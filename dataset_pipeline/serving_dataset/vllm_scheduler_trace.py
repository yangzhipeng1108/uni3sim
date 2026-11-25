"""Low-overhead in-process vLLM Scheduler trace instrumentation."""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import ctypes
import json
import inspect
import multiprocessing
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, NamedTuple

from dataset_pipeline.serving_dataset.pd_p2p_request_id import transfer_request_id


_OUTPUT_ENV = "MIDLAYER_SCHEDULER_TRACE_OUTPUT"
_REQUEST_ADD_OUTPUT_ENV = "MIDLAYER_SCHEDULER_REQUEST_ADD_OUTPUT"
_STAGE_THREAD_CPU_ENV = "MIDLAYER_SCHEDULER_STAGE_THREAD_CPU"
_STAGE_RECORDERS: list = []
_COUNTER_ONLY_OUTPUT_ENV = "MIDLAYER_A800_COUNTER_ONLY_OUTPUT"
_REQUEST_COUNT_ENV = "MIDLAYER_SCHEDULER_TRACE_REQUEST_COUNT"
_CONFIGURED_GRAPH_MODE_ENV = (
    "MIDLAYER_SCHEDULER_TRACE_CONFIGURED_CUDAGRAPH_MODE"
)
_MAX_IN_FLIGHT_ENV = "MIDLAYER_SCHEDULER_TRACE_MAX_IN_FLIGHT"
_HARDWARE_ID_ENV = "MIDLAYER_SCHEDULER_TRACE_HARDWARE_ID"
_WORKER_RANK_COUNT_ENV = "MIDLAYER_SCHEDULER_TRACE_WORKER_RANK_COUNT"
_A800_NSYS_STEP_NVTX_ENV = "MIDLAYER_A800_NSYS_STEP_NVTX"
_C500_GRAPH_LAUNCH_CONTEXT_ENV = "MIDLAYER_C500_GRAPH_LAUNCH_CONTEXT"
_C500_GRAPH_LAUNCH_LOG_ENV = "MIDLAYER_C500_GRAPH_LAUNCH_LOG"
_C500_KERNEL_LAUNCH_LOG_ENV = "MIDLAYER_C500_KERNEL_LAUNCH_LOG"
_C500_GRAPH_LAUNCH_INTERPOSER_ENV = "MIDLAYER_C500_GRAPH_LAUNCH_INTERPOSER"
_ASCEND_MSPROF_STEP_TIMESTAMP_ENV = "MIDLAYER_910B_MSPROF_STEP_TIMESTAMP"
_ASCEND_MSPROFTX_STEP_MARK_ENV = "MIDLAYER_910B_MSPROFTX_STEP_MARK"
_ASCEND_STEP_DEVICE_WINDOW_ENV = "MIDLAYER_910B_STEP_DEVICE_WINDOW"
_ASCEND_THREAD_PROFILE_STEP_ENV = "MIDLAYER_910B_THREAD_PROFILE_STEP_ID"
_ASCEND_STACK_SAMPLE_STEP_ENV = "MIDLAYER_910B_STACK_SAMPLE_STEP_ID"
_ASCEND_NATIVE_PERF_STEP_ENV = "MIDLAYER_910B_NATIVE_PERF_STEP_ID"
_ASCEND_FRAMEWORK_TIMING_STEP_ENV = "MIDLAYER_910B_FRAMEWORK_TIMING_STEP_ID"
_ASCEND_FRAMEWORK_TIMING_TORCH_DISPATCH_ENV = (
    "MIDLAYER_910B_FRAMEWORK_TIMING_TORCH_DISPATCH")
_ASCEND_FRAMEWORK_TIMING_TORCH_DISPATCH_COUNT_ENV = (
    "MIDLAYER_910B_FRAMEWORK_TIMING_TORCH_DISPATCH_COUNT")
_ASCEND_FRAMEWORK_TIMING_SELECTED_TORCH_OP_ENV = (
    "MIDLAYER_910B_FRAMEWORK_TIMING_SELECTED_TORCH_OP")
_ASCEND_TORCH_PROFILER_STEP_ENV = "MIDLAYER_910B_TORCH_PROFILER_STEP_ID"
_SOURCE_IDENTITY_ENV = "MIDLAYER_SCHEDULER_TRACE_SOURCE_IDENTITY"
_SPAWN_WRAPPER_ENV = "MIDLAYER_SCHEDULER_TRACE_SPAWN_WRAPPER"
_SPAWN_PYTHON_ENV = "MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHON"
_SPAWN_PYTHONPATH_ENV = "MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHONPATH"
_BOOTSTRAP_MARKER_ENV = "MIDLAYER_SCHEDULER_TRACE_BOOTSTRAP_MARKER"
_PD_RUN_ID_ENV = "MIDLAYER_PD_RUN_ID"
_PD_RUN_ORIGIN_ENV = "MIDLAYER_PD_RUN_ORIGIN_MONOTONIC_NS"
_PD_HOST_BOOT_ID_ENV = "MIDLAYER_PD_HOST_BOOT_ID"
_PD_INSTANCE_ROLE_ENV = "MIDLAYER_PD_INSTANCE_ROLE"
_PD_INSTANCE_ID_ENV = "MIDLAYER_PD_INSTANCE_ID"
_PD_HANDOFF_EVENTS_DIR_ENV = "MIDLAYER_PD_HANDOFF_EVENTS_DIR"
_PD_HANDOFF_CONNECTOR_ENV = "MIDLAYER_PD_HANDOFF_CONNECTOR"
_REQUEST_ID = re.compile(
    r"(?:cmpl-)?request-(?P<index>[0-9]+)(?:-[0-9]+(?:-[0-9a-f]{8})?)?"
)
_REQUEST_BASE_ID = re.compile(
    r"(?P<base>request-[0-9]+)(?:-[0-9]+(?:-[0-9a-f]{8})?)?$"
)
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_ACTIVE_TRACE_STATE: Any | None = None
_ENGINE_CORE_ORIGINAL_TARGET: object | None = None
_P2P_CAPTURE_BOOTSTRAP_STATE = threading.local()
_P2P_CAPTURE_BOOTSTRAP_MARKER = "_midlayer_p2p_capture_bootstrap_installed"
_A800_COUNTER_FIELDS = (
    "num_graphs_seen",
    "num_piecewise_graphs_seen",
    "num_backend_compilations",
    "num_inductor_compiles",
    "num_eager_compiles",
    "num_cache_entries_updated",
    "num_compiled_artifacts_saved",
    "num_cudagraph_captured",
)
_CURRENT_REQUEST_INDEX: contextvars.ContextVar[int] = contextvars.ContextVar(
    "midlayer_request_stage_index"
)


_H200_WORKER_DISPATCH_RELEASES = ("0.11.0",)
_H200_RUNTIME_STATS_RELEASES = ("0.30.0",)


class _TraceInstrumentation(NamedTuple):
    """How one hardware's Scheduler trace observes per-step CUDA Graph facts.

    ``worker_dispatch`` is True when the capture wraps the V1 worker's graph
    dispatcher (instrumentation v4, vLLM 0.11.0 only) and False when the same
    facts arrive in ``ModelRunnerOutput.cudagraph_stats``. ``deferred_io`` is
    True when trace file I/O leaves the EngineCore busy-loop thread. Both
    values are ``None`` for hardware whose choice follows the installed vLLM
    release instead of being fixed; ``release_worker_dispatch`` names the
    releases that select the worker-dispatch form for such hardware.
    """

    worker_dispatch: bool | None
    deferred_io: bool | None
    release_worker_dispatch: tuple = ()


# Hardware whose instrumentation is fixed by what its canonical captures were
# taken with, and hardware whose form follows the installed release.
_TRACE_INSTRUMENTATION = {
    "nvidia_a800": _TraceInstrumentation(True, False),
    "metax_c500": _TraceInstrumentation(False, False),
    "ascend_910b": _TraceInstrumentation(False, False),
    "nvidia_h200": _TraceInstrumentation(None, None,
                                         _H200_WORKER_DISPATCH_RELEASES),
}
# A trace started without an identity (import-time environment probe) uses the
# runtime-stats form the later runtimes report.
# vLLM 0.11's MultiprocExecutor waits only for the output rank's
# ``execute_model`` reply (``unique_reply_rank``), so another TP/EP rank can
# append its worker observation for step N after the Scheduler has already
# run ``update_from_output`` for step N.  Rank r must finish step N before its
# TP/EP collectives of step N+1 can complete on the output rank, so every
# non-final step's observation is on disk by the next formal update.  Only the
# final step has no successor; sealing the trace waits this long for it and
# then fails loudly.  The worker writes the observation right after its
# forward returns, so a healthy lag is microseconds to milliseconds.
_WORKER_OBSERVATION_ARRIVAL_TIMEOUT_S = 30.0
_WORKER_OBSERVATION_POLL_S = 0.001


def _installed_vllm_release() -> str:
    from importlib.metadata import version

    return version("vllm").split("+", 1)[0]


def _uses_worker_dispatch_observations(hardware_id: str | None) -> bool:
    """Select how per-step CUDA Graph evidence is observed.

    vLLM 0.11.0 exposes no per-step graph statistics, so the A800 capture
    wraps the V1 worker's CUDA Graph dispatcher and writes per-rank
    observations (instrumentation v4).  Later runtimes report the same facts
    in ``ModelRunnerOutput.cudagraph_stats`` when ``--cudagraph-metrics`` is
    set (instrumentation v2, as on C500 0.24.0 and 910B 0.18.0).  H200 runs
    both runtimes, so its choice follows the installed release exactly.
    """

    if hardware_id is None:
        return False
    if hardware_id not in _TRACE_INSTRUMENTATION:
        raise ValueError(
            f"Scheduler trace hardware identity is unsupported: {hardware_id!r}")
    declared = _TRACE_INSTRUMENTATION[hardware_id]
    if declared.worker_dispatch is not None:
        return declared.worker_dispatch
    release = _installed_vllm_release()
    if release in declared.release_worker_dispatch:
        return True
    if release in _H200_RUNTIME_STATS_RELEASES:
        return False
    raise ValueError(
        f"H200 Scheduler trace does not support vLLM {release!r}")


def _uses_deferred_trace_io(hardware_id: str | None) -> bool:
    """Select whether trace file I/O leaves the EngineCore busy-loop thread.

    Only H200 vLLM 0.30.0 defers.  There the synchronous marker appends and
    depth-evidence path resolution on the shared filesystem ran inside the
    first formal step (request 0 TTFT +18-20 ms in every trace-on run), and
    the final trace serialisation ran inside the last ``update_from_output``,
    holding the last request's output for 88-260 ms.  Every other runtime
    keeps the synchronous behaviour its canonical captures were taken with.
    The recorded content is identical either way.
    """

    # Deferred I/O is opt-in: an unidentified or unknown hardware keeps the
    # synchronous behaviour the other runtimes were captured with.
    if hardware_id and hardware_id in _TRACE_INSTRUMENTATION:
        declared = _TRACE_INSTRUMENTATION[hardware_id]
    else:
        declared = None
    if declared is None or declared.deferred_io is False:
        return False
    release = _installed_vllm_release()
    if release in _H200_RUNTIME_STATS_RELEASES:
        return True
    if release in declared.release_worker_dispatch:
        return False
    raise ValueError(
        f"H200 Scheduler trace does not support vLLM {release!r}")


def flush_request_stage_recorders() -> None:
    """Write out every buffered request stage span in this process.

    Called on the normal completion path so the trace is on disk before the
    run reports success, rather than depending on interpreter exit.
    """

    for recorder in _STAGE_RECORDERS:
        recorder.flush()


class _RequestStageRecorder:
    """Write request-path spans without sharing buffered state across processes."""

    # Every stage span is taken on the serving Host's event-loop thread, so any
    # syscall or file write performed inline is stolen directly from request
    # progress and is amplified by the number of streams in flight.  Records are
    # therefore buffered as raw dicts and serialised in one batch, and the
    # per-span thread-CPU clock -- which costs two extra syscalls per span and is
    # diagnostic only -- is opt-in.  The wall clock the decomposition actually
    # needs is a single ``perf_counter_ns`` per boundary.
    # Buffered records only reach disk on a flush, so the threshold bounds what
    # a hard kill can destroy.  A capture is terminated mid-run often enough --
    # a stalled run, an abandoned model load -- that losing the whole buffer
    # would take the diagnostic evidence with it.  At eight spans per request
    # this bounds the loss to roughly sixty requests' worth while still
    # replacing per-span file operations with one batched write.
    _FLUSH_THRESHOLD = 512

    def __init__(self, request_add_output_path: Path) -> None:
        self.path = Path(
            f"{request_add_output_path}.stages.{os.getpid()}.jsonl"
        )
        self.lock = threading.Lock()
        self.buffer = []
        # A request crosses about eight spans, so parsing its index out of the
        # request ID once per span repeats the same string work eight times.
        self.index_cache: dict = {}
        self.capture_thread_cpu = _STAGE_THREAD_CPU_ENV in os.environ
        # The completion path flushes explicitly; this is only the backstop for
        # an exit that does not run it.
        atexit.register(self.flush)
        _STAGE_RECORDERS.append(self)

    def flush(self) -> None:
        """Serialise and append everything buffered so far."""

        with self.lock:
            pending = self.buffer
            if not pending:
                return
            self.buffer = []
            lines = "".join(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ) + "\n"
                for record in pending
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(lines)

    def _append(self, record) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self._FLUSH_THRESHOLD:
            self.flush()

    def record_runtime_stage_absence(self, stage: str) -> None:
        """Record that this runtime structurally has no such request stage.

        Written once per process so a consumer can tell a stage that the
        runtime never executes from a stage the tracer failed to capture.
        """

        record = {
            "schema_version": 1,
            "clock_domain": "shared_host_time.perf_counter_ns",
            "process_pid": os.getpid(),
            "request_idx": None,
            "request_id": None,
            "stage": stage,
            "runtime_stage_status": "absent_in_runtime_source",
        }
        self._append(record)

    def record(
        self,
        stage: str,
        request: object,
        started_ns: int,
        ended_ns: int,
        started_thread_cpu_ns: int,
        ended_thread_cpu_ns: int,
        *,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        request_id = (
            request if isinstance(request, str) else request.request_id
        )
        cache = self.index_cache
        if request_id in cache:
            request_idx = cache[request_id]
        else:
            request_idx = _request_index(request_id)
            cache[request_id] = request_idx
        if request_idx is None:
            return
        if ended_ns < started_ns:
            raise RuntimeError("request stage timing is not monotonic")
        if (started_thread_cpu_ns is not None
                and ended_thread_cpu_ns is not None
                and ended_thread_cpu_ns < started_thread_cpu_ns):
            raise RuntimeError("request stage CPU timing is not monotonic")
        record: dict[str, object] = {
            "schema_version": 1,
            "clock_domain": "shared_host_time.perf_counter_ns",
            "process_pid": os.getpid(),
            "request_idx": request_idx,
            "request_id": request_id,
            "stage": stage,
            "start_monotonic_ns": started_ns,
            "end_monotonic_ns": ended_ns,
        }
        if started_thread_cpu_ns is not None and ended_thread_cpu_ns is not None:
            record["thread_cpu_duration_ns"] = (
                ended_thread_cpu_ns - started_thread_cpu_ns
            )
        if extra is not None:
            for key, value in extra.items():
                if key in record:
                    raise ValueError(f"duplicate request stage field: {key}")
                record[key] = value
        self._append(record)


def _timed_call(
    recorder: _RequestStageRecorder,
    stage: str,
    request: object,
    function: object,
    *args: object,
    **kwargs: object,
) -> object:
    capture_cpu = recorder.capture_thread_cpu
    started_ns = time.perf_counter_ns()
    started_thread_cpu_ns = time.thread_time_ns() if capture_cpu else None
    result = function(*args, **kwargs)
    ended_thread_cpu_ns = time.thread_time_ns() if capture_cpu else None
    ended_ns = time.perf_counter_ns()
    recorder.record(
        stage,
        request,
        started_ns,
        ended_ns,
        started_thread_cpu_ns,
        ended_thread_cpu_ns,
    )
    return result


def _resolve_optional_runtime_symbol(
    candidates: "tuple[tuple[str, str], ...]",
):
    """Return a symbol a runtime may legitimately not have, else ``None``.

    This is only for stages whose very existence is a runtime-version fact.
    Required symbols must keep using :func:`_resolve_runtime_symbol`, which
    fails closed.
    """

    for module_name, symbol_name in candidates:
        try:
            module = __import__(module_name, fromlist=[symbol_name])
        except ImportError:
            continue
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name)
    return None


def _resolve_runtime_symbol(
    label: str,
    candidates: "tuple[tuple[str, str], ...]",
) -> type:
    """Return the first candidate symbol the installed runtime actually has.

    Candidates are an explicit, ordered enumeration of the module paths the
    pinned runtimes use.  Nothing is guessed and nothing is skipped: when no
    candidate resolves this raises, so an unrecognised runtime fails closed
    rather than producing a trace with a silently missing stage.
    """

    attempted = []
    for module_name, symbol_name in candidates:
        try:
            module = __import__(module_name, fromlist=[symbol_name])
        except ImportError:
            attempted.append(f"{module_name} (absent)")
            continue
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name)
        attempted.append(f"{module_name} (no {symbol_name})")
    raise RuntimeError(
        f"request-stage instrumentation cannot resolve the {label} on this "
        "runtime; tried " + ", ".join(attempted))


def _install_request_stage_wrappers(
    recorder: _RequestStageRecorder,
) -> None:
    """Trace active pre-Scheduler spans and the explicit queues between them."""

    from functools import partial

    from transformers.tokenization_utils_base import BatchEncoding
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.core_client import AsyncMPClient, EngineCoreRequestType
    from vllm.v1.engine.output_processor import OutputProcessor

    # Three request-path symbols were relocated between the pinned runtimes
    # this tracer collects on (A800 vLLM 0.11.0, 910B vLLM 0.18.0).  The
    # candidates are enumerated explicitly and resolution fails closed, so a
    # future relocation raises here instead of silently dropping a stage.
    OpenAIServingCompletion = _resolve_runtime_symbol(
        "OpenAI completion entrypoint",
        (
            ("vllm.entrypoints.openai.serving_completion",
             "OpenAIServingCompletion"),
            ("vllm.entrypoints.openai.completion.serving",
             "OpenAIServingCompletion"),
        ),
    )
    # The batching tokenizer is a runtime-version feature, not a universal
    # stage: vLLM 0.11.0 and 0.18.0 have it, 0.24.0 tokenizes through a plain
    # executor hop instead.  Its absence is a recorded structural fact, so the
    # remaining stages still bound the Host admission path end to end.
    AsyncMicrobatchTokenizer = _resolve_optional_runtime_symbol(
        (
            ("vllm.utils", "AsyncMicrobatchTokenizer"),
            ("vllm.utils.async_utils", "AsyncMicrobatchTokenizer"),
        ),
    )
    Processor = _resolve_runtime_symbol(
        "engine input processor",
        (
            ("vllm.v1.engine.processor", "Processor"),
            ("vllm.v1.engine.input_processor", "InputProcessor"),
        ),
    )

    original_create_completion = OpenAIServingCompletion.create_completion

    async def traced_create_completion(
        serving: object,
        request: object,
        raw_request: object = None,
    ) -> object:
        request_id = None
        if raw_request is not None and "x-request-id" in raw_request.headers:
            request_id = raw_request.headers["x-request-id"]
        elif hasattr(request, "request_id"):
            request_id = request.request_id
        request_idx = _request_index(request_id) if request_id is not None else None
        if request_idx is None:
            return await original_create_completion(serving, request, raw_request)
        token = _CURRENT_REQUEST_INDEX.set(request_idx)
        started_ns = time.perf_counter_ns()
        started_thread_cpu_ns = time.thread_time_ns()
        try:
            return await original_create_completion(serving, request, raw_request)
        finally:
            ended_thread_cpu_ns = time.thread_time_ns()
            ended_ns = time.perf_counter_ns()
            recorder.record(
                "openai_create_completion",
                request_id,
                started_ns,
                ended_ns,
                started_thread_cpu_ns,
                ended_thread_cpu_ns,
            )
            _CURRENT_REQUEST_INDEX.reset(token)

    OpenAIServingCompletion.create_completion = traced_create_completion

    original_tokenizer_call = (
        AsyncMicrobatchTokenizer.__call__
        if AsyncMicrobatchTokenizer is not None else None)
    pending_tokenization: dict[
        asyncio.Future, tuple[int, int]
    ] = {}

    async def traced_tokenizer_call(
        tokenizer: object, prompt: object, **kwargs: object
    ) -> object:
        context = contextvars.copy_context()
        if _CURRENT_REQUEST_INDEX not in context:
            return await original_tokenizer_call(tokenizer, prompt, **kwargs)
        request_idx = context[_CURRENT_REQUEST_INDEX]
        result_future = tokenizer._loop.create_future()
        key = tokenizer._queue_key("encode", kwargs)
        queue = tokenizer._get_queue(tokenizer._loop, key)
        enqueue_ns = time.perf_counter_ns()
        pending_tokenization[result_future] = (request_idx, enqueue_ns)
        await queue.put((prompt, kwargs, result_future))
        try:
            return await result_future
        finally:
            if result_future in pending_tokenization:
                del pending_tokenization[result_future]

    async def traced_batch_encode_loop(
        tokenizer: object, queue: asyncio.Queue, can_batch: bool
    ) -> None:
        while True:
            prompt, kwargs, result_future = await getattr(queue, "get")()
            prompts = [prompt]
            kwargs_list = [kwargs]
            result_futures = [result_future]
            deadline = tokenizer._loop.time() + tokenizer.batch_wait_timeout_s
            while len(prompts) < tokenizer.max_batch_size:
                timeout = deadline - tokenizer._loop.time()
                if timeout <= 0:
                    break
                try:
                    prompt, kwargs, result_future = await asyncio.wait_for(
                        getattr(queue, "get")(), timeout
                    )
                    prompts.append(prompt)
                    result_futures.append(result_future)
                    if not can_batch:
                        kwargs_list.append(kwargs)
                except asyncio.TimeoutError:
                    break
            try:
                execution_start_ns_box: list[int] = []
                execution_end_ns_box: list[int] = []
                execution_cpu_ns_box: list[int] = []

                def execute_batch() -> object:
                    execution_start_ns_box.append(time.perf_counter_ns())
                    cpu_start_ns = time.thread_time_ns()
                    if can_batch and len(prompts) > 1:
                        batch_encode_fn = partial(
                            tokenizer.tokenizer, prompts, **kwargs
                        )
                        value = batch_encode_fn()
                    else:
                        value = [
                            tokenizer.tokenizer(item, **item_kwargs)
                            for item, item_kwargs in zip(prompts, kwargs_list)
                        ]
                    execution_cpu_ns_box.append(
                        time.thread_time_ns() - cpu_start_ns
                    )
                    execution_end_ns_box.append(time.perf_counter_ns())
                    return value

                results = await tokenizer._loop.run_in_executor(
                    tokenizer._executor, execute_batch
                )
                if (
                    len(execution_start_ns_box) != 1
                    or len(execution_end_ns_box) != 1
                    or len(execution_cpu_ns_box) != 1
                ):
                    raise RuntimeError("tokenizer execution timing is incomplete")
                execution_start_ns = execution_start_ns_box[0]
                execution_end_ns = execution_end_ns_box[0]
                execution_cpu_ns = execution_cpu_ns_box[0]
                batch_size = len(result_futures)
                for index, future in enumerate(result_futures):
                    if future in pending_tokenization:
                        request_idx, enqueue_ns = pending_tokenization[future]
                        recorder.record(
                            "tokenizer_batch",
                            f"request-{request_idx}",
                            enqueue_ns,
                            execution_end_ns,
                            0,
                            execution_cpu_ns,
                            extra={
                                "tokenizer_queue_duration_ns": (
                                    execution_start_ns - enqueue_ns
                                ),
                                "tokenizer_execution_start_monotonic_ns": (
                                    execution_start_ns
                                ),
                                "tokenizer_execution_wall_duration_ns": (
                                    execution_end_ns - execution_start_ns
                                ),
                                "tokenizer_batch_size": batch_size,
                            },
                        )
                    if not future.done():
                        if can_batch and batch_size > 1:
                            data = {
                                key: value[index]
                                for key, value in results.items()
                            }
                            future.set_result(BatchEncoding(data))
                        else:
                            future.set_result(results[index])
            except Exception as error:
                for future in result_futures:
                    if not future.done():
                        future.set_exception(error)
                raise

    if AsyncMicrobatchTokenizer is not None:
        AsyncMicrobatchTokenizer.__call__ = traced_tokenizer_call
        AsyncMicrobatchTokenizer._batch_encode_loop = traced_batch_encode_loop
    else:
        recorder.record_runtime_stage_absence("tokenizer_batch")

    original_process_inputs = Processor.process_inputs

    def traced_process_inputs(
        processor: object, request_id: str, *args: object, **kwargs: object
    ) -> object:
        return _timed_call(
            recorder,
            "engine_input_processor",
            request_id,
            original_process_inputs,
            processor,
            request_id,
            *args,
            **kwargs,
        )

    Processor.process_inputs = traced_process_inputs

    original_output_add_request = OutputProcessor.add_request

    def traced_output_add_request(
        output_processor: object,
        request: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        return _timed_call(
            recorder,
            "output_processor_add_request",
            request,
            original_output_add_request,
            output_processor,
            request,
            *args,
            **kwargs,
        )

    OutputProcessor.add_request = traced_output_add_request

    original_send_input = AsyncMPClient._send_input

    def traced_send_input(
        client: object,
        request_type: object,
        request: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        if request_type != EngineCoreRequestType.ADD:
            return original_send_input(
                client, request_type, request, *args, **kwargs
            )
        return _timed_call(
            recorder,
            "engine_core_ipc_enqueue",
            request,
            original_send_input,
            client,
            request_type,
            request,
            *args,
            **kwargs,
        )

    AsyncMPClient._send_input = traced_send_input

    original_preprocess_add_request = EngineCore.preprocess_add_request

    def traced_preprocess_add_request(
        engine_core: object,
        request: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        return _timed_call(
            recorder,
            "engine_core_preprocess",
            request,
            original_preprocess_add_request,
            engine_core,
            request,
            *args,
            **kwargs,
        )

    EngineCore.preprocess_add_request = traced_preprocess_add_request

    original_engine_core_add_request = EngineCore.add_request

    def traced_engine_core_add_request(
        engine_core: object,
        request: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        return _timed_call(
            recorder,
            "engine_core_add_request",
            request,
            original_engine_core_add_request,
            engine_core,
            request,
            *args,
            **kwargs,
        )

    EngineCore.add_request = traced_engine_core_add_request


class _PDHandoffRecorder:
    """Append one-process connector evidence for a shared PD run."""

    def __init__(
        self, identity: Mapping[str, object], *, write_through: bool = False
    ) -> None:
        if _PD_HANDOFF_EVENTS_DIR_ENV not in os.environ:
            raise RuntimeError(
                "PD Scheduler tracing requires MIDLAYER_PD_HANDOFF_EVENTS_DIR"
            )
        if _PD_HANDOFF_CONNECTOR_ENV not in os.environ:
            raise RuntimeError(
                "PD Scheduler tracing requires MIDLAYER_PD_HANDOFF_CONNECTOR"
            )
        output_dir = Path(os.environ[_PD_HANDOFF_EVENTS_DIR_ENV]).resolve()
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError("PD handoff event path must be a directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        connector = os.environ[_PD_HANDOFF_CONNECTOR_ENV]
        if connector not in (
            "P2pNcclConnector", "P2pMcclConnector", "P2pHcclConnector", "MooncakeConnector"
        ):
            raise ValueError("PD handoff connector identity is unsupported")
        instance_role = identity["instance_role"]
        if not isinstance(instance_role, str):
            raise TypeError("PD handoff instance role must be text")
        run_id = identity["run_id"]
        if not isinstance(run_id, str):
            raise TypeError("PD handoff run ID must be text")
        self.identity = identity
        self.connector = connector
        recorder_kind = "runtime" if write_through else "scheduler"
        self.path = output_dir / (
            f"{instance_role}.connector.{recorder_kind}.{os.getpid()}.jsonl"
        )
        # A V1 engine may construct more than one connector object in one
        # worker process.  Runtime handoff evidence is intentionally
        # write-through, so those objects must append to that process's single
        # evidence stream; treating the second construction as a second run
        # aborts a valid P2P receive.  Buffered traces still own their output
        # atomically and must retain the no-overwrite guard.
        if self.path.exists() and not write_through:
            raise FileExistsError(
                f"refusing to overwrite PD connector evidence: {self.path}"
            )
        self.lock = threading.Lock()
        self.records: list[dict[str, object]] = []
        self.written = False
        self.write_through = write_through
        self.write_through_handle = None
        if self.write_through:
            # Runtime connector records occur on the critical NCCL receive
            # path.  Keep Python's bounded file buffer in that path and flush
            # it when the Scheduler trace seals or the worker exits normally.
            # An unsuccessful process never yields a candidate, so it must
            # not trade the measured workload's integrity for synchronous
            # storage latency on every connector boundary.
            atexit.register(self.write_complete)

    def record(
        self,
        event: str,
        *,
        request_id: str,
        kv_tokens: int | None,
        kv_blocks: int | None,
        byte_count: int | None,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        if not request_id:
            raise ValueError("PD connector event requires a request ID")
        for label, value in (
            ("KV tokens", kv_tokens),
            ("KV blocks", kv_blocks),
            ("byte count", byte_count),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise TypeError(f"PD connector {label} must be non-negative or null")
        origin_ns = self.identity["run_origin_monotonic_ns"]
        if isinstance(origin_ns, bool) or not isinstance(origin_ns, int):
            raise TypeError("PD handoff run origin must be a non-negative integer")
        timestamp_ns = time.perf_counter_ns()
        if timestamp_ns < origin_ns:
            raise RuntimeError("PD connector event predates the shared run origin")
        payload: dict[str, object] = {
            "schema_version": 1,
            "event_kind": "pd_connector_handoff",
            "event": event,
            # Keep raw connector observations in the same explicit run-relative
            # clock as Scheduler and router records.  The shared origin itself is
            # carried below, so an archive never needs to infer a clock offset.
            "timestamp_ns": timestamp_ns - origin_ns,
            "clock_domain": "shared_host_time.perf_counter_ns",
            "process_pid": os.getpid(),
            "connector": self.connector,
            "request_id": request_id,
            "base_request_id": _base_request_id(request_id),
            "kv_tokens": kv_tokens,
            "kv_blocks": kv_blocks,
            "byte_count": byte_count,
            **self.identity,
        }
        if extra is not None:
            for key, value in extra.items():
                if key in payload:
                    raise ValueError(f"PD connector event duplicate field: {key}")
                payload[key] = value
        with self.lock:
            if self.written:
                raise RuntimeError("PD connector event arrived after trace completion")
            if self.write_through:
                if self.write_through_handle is None:
                    self.write_through_handle = self.path.open(
                        "a", encoding="utf-8"
                    )
                self.write_through_handle.write(json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ) + "\n")
                return
            self.records.append(payload)

    def write_complete(self) -> None:
        """Atomically persist buffered connector evidence once per process."""
        with self.lock:
            if self.write_through:
                if self.write_through_handle is not None:
                    self.write_through_handle.flush()
                return
            if self.written:
                raise RuntimeError("PD connector evidence attempted to write twice")
            if not self.records:
                raise RuntimeError("PD connector evidence completed without events")
            records = tuple(self.records)
            self.written = True
        lines = tuple(json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) for record in records)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        if self.path.exists() or temporary.exists():
            raise FileExistsError(
                f"refusing to replace PD connector evidence: {self.path}"
            )
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


class _TraceState:
    def __init__(
        self,
        output_path: Path,
        request_count: int,
        *,
        configured_cudagraph_mode: str | None = None,
        max_in_flight: int | None = None,
        hardware_id: str | None = None,
        worker_rank_count: int | None = None,
        counter_only: bool = False,
        pd_run_identity: Mapping[str, object] | None = None,
        pd_handoff_recorder: _PDHandoffRecorder | None = None,
        request_add_output_path: Path | None = None,
    ) -> None:
        self.output_path = output_path
        self.request_count = request_count
        self.configured_cudagraph_mode = configured_cudagraph_mode
        self.max_in_flight = max_in_flight
        self.hardware_id = hardware_id
        self.worker_dispatch_profile = _uses_worker_dispatch_observations(
            hardware_id)
        self.worker_rank_count = worker_rank_count
        self.counter_only = counter_only
        self.pd_run_identity = pd_run_identity
        self.pd_handoff_recorder = pd_handoff_recorder
        self.request_add_output_path = request_add_output_path
        self.request_add_end_ns: dict[int, int] = {}
        self.lock = threading.Lock()
        self.next_iteration = 0
        self.pending: dict[int, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []
        self.finished_request_indexes: set[int] = set()
        self.warmup_step_count = 0
        self.warmup_mixed_step_count = 0
        self.warmup_max_scheduled_requests = 0
        self.warmup_scheduled_tokens = 0
        self.warmup_request_ids: set[str] = set()
        self.written = False
        self.worker_rank: int | None = None
        self.worker_counter_after_warmup_written = False
        self.worker_counter_after_benchmark_written = False
        self.next_counter_only_step = 0
        self.worker_execution = threading.local()
        self.configured_depth_evidence: Mapping[str, object] | None = None
        self.a800_rank_observation_offsets: dict[int, int] = {}
        self.a800_rank_observation_cache: dict[
            int, dict[str, dict[str, object]]
        ] = {}
        # Recorded formal steps whose per-rank worker observations have not
        # all arrived yet, oldest first (see
        # _WORKER_OBSERVATION_ARRIVAL_TIMEOUT_S).
        self.a800_unresolved_steps: list[dict[str, Any]] = []
        self.bootstrap_marker_events: set[str] = set()
        self.request_identity_cache: dict[str, tuple[int | None, bool]] = {}
        self.deferred_io = _uses_deferred_trace_io(hardware_id)
        # One worker keeps marker appends and the final publication in their
        # original order; its thread is joined at interpreter exit.
        self.io_executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="midlayer-scheduler-trace-io",
            )
            if self.deferred_io
            else None
        )
        self.io_error: BaseException | None = None
        self.depth_evidence_probe_ns: int | None = None

    def _raise_deferred_io_error(self) -> None:
        if self.io_error is not None:
            raise RuntimeError(
                "Scheduler trace deferred file I/O failed"
            ) from self.io_error

    def _submit_io(self, function: Any, *args: object) -> None:
        """Run trace file I/O on the trace I/O thread, failing loud."""

        self._raise_deferred_io_error()
        if self.io_executor is None:
            raise RuntimeError("Scheduler trace deferred I/O is not enabled")

        def run() -> None:
            try:
                function(*args)
            except BaseException as error:
                self.io_error = error
                print(
                    "midlayer Scheduler trace deferred I/O failed:",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                raise

        self.io_executor.submit(run)

    def wait_for_deferred_io(self) -> None:
        """Block until queued trace I/O has run; raise its failure."""

        if self.io_executor is not None:
            self.io_executor.submit(lambda: None).result()
        self._raise_deferred_io_error()

    def _new_bootstrap_marker(self, event: str) -> dict[str, Any] | None:
        if _BOOTSTRAP_MARKER_ENV not in os.environ:
            return None
        if event in self.bootstrap_marker_events:
            return None
        self.bootstrap_marker_events.add(event)
        return {
            "event": event,
            "process_pid": os.getpid(),
            "process_name": multiprocessing.current_process().name,
        }

    def record_bootstrap_marker_event(self, event: str) -> None:
        """Record one opt-in control-flow marker for a traced process."""

        marker = self._new_bootstrap_marker(event)
        if marker is None:
            return
        if self.deferred_io:
            # Resolving and appending on the shared filesystem is slow and
            # variable; keep it off the Scheduler thread.
            self._submit_io(
                _append_marker, os.environ[_BOOTSTRAP_MARKER_ENV], marker)
            return
        _append_jsonl(Path(os.environ[_BOOTSTRAP_MARKER_ENV]).resolve(), marker)

    def _record_bootstrap_marker_failure(
        self, event: str, error: Exception
    ) -> None:
        """Persist opt-in write failure evidence before propagating it."""

        if _BOOTSTRAP_MARKER_ENV not in os.environ:
            return
        _append_jsonl(Path(os.environ[_BOOTSTRAP_MARKER_ENV]).resolve(), {
            "event": event,
            "process_pid": os.getpid(),
            "process_name": multiprocessing.current_process().name,
            "error_type": type(error).__name__,
            "error_message": str(error),
        })

    def after_add_request(self, request: object, ended_ns: int) -> None:
        """Record when a formal request has entered the Scheduler."""

        request_id = request.request_id
        request_idx = _request_index(request_id)
        if request_idx is None:
            return
        with self.lock:
            if request_idx in self.request_add_end_ns:
                raise RuntimeError(
                    "Scheduler trace observed duplicate request admission: "
                    f"{request_id}"
                )
            self.request_add_end_ns[request_idx] = ended_ns

    def before_schedule(self) -> tuple[int, int]:
        started_ns = time.perf_counter_ns()
        with self.lock:
            iteration = self.next_iteration
            self.next_iteration += 1
        return iteration, started_ns

    def after_schedule(
        self,
        scheduler: object,
        scheduler_output: object,
        iteration: int,
        started_ns: int,
    ) -> None:
        schedule_end_ns = time.perf_counter_ns()
        scheduled_requests = []
        warmup_requests = []
        identities = self.request_identity_cache
        for request_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            # Parsing the request ID is the dominant per-step cost of this
            # hook and runs on the Scheduler's critical path every step; a
            # request keeps its ID for its whole life, so parse it once.
            if request_id in identities:
                request_idx, warmup_request = identities[request_id]
            else:
                request_idx = _request_index(request_id)
                warmup_request = _is_warmup_request(request_id)
                identities[request_id] = (request_idx, warmup_request)
            if request_idx is None and not warmup_request:
                continue
            if request_id not in scheduler.requests:
                raise RuntimeError(
                    "Scheduler trace cannot recover post-schedule request state: "
                    f"{request_id}"
                )
            request = scheduler.requests[request_id]
            computed_tokens = request.num_computed_tokens - scheduled_tokens
            if computed_tokens < 0:
                raise RuntimeError(
                    "Scheduler trace recovered a negative pre-schedule token "
                    f"count for {request_id}"
                )
            prompt_tokens = request.num_prompt_tokens
            remaining_prefill = max(prompt_tokens - computed_tokens, 0)
            prefill_tokens = min(scheduled_tokens, remaining_prefill)
            decode_tokens = scheduled_tokens - prefill_tokens
            scheduled = (
                request_idx, prefill_tokens, decode_tokens, scheduled_tokens
            )
            if warmup_request:
                warmup_requests.append((request_id, *scheduled[1:]))
            else:
                scheduled_requests.append(scheduled)
        if warmup_requests:
            with self.lock:
                self.warmup_step_count += 1
                self.warmup_max_scheduled_requests = max(
                    self.warmup_max_scheduled_requests,
                    len(warmup_requests),
                )
                warmup_prefill = sum(value[1] for value in warmup_requests)
                warmup_decode = sum(value[2] for value in warmup_requests)
                if warmup_prefill > 0 and warmup_decode > 0:
                    self.warmup_mixed_step_count += 1
                self.warmup_scheduled_tokens += sum(
                    value[3] for value in warmup_requests
                )
                self.warmup_request_ids.update(
                    value[0] for value in warmup_requests
                )
        if not scheduled_requests:
            return
        self.record_bootstrap_marker_event("formal_scheduler_schedule")
        scheduled_requests.sort(key=lambda value: value[0])
        with self.lock:
            key = id(scheduler_output)
            if key in self.pending:
                raise RuntimeError("Scheduler trace output identity was reused")
            in_flight_step_count = len(self.pending) + 1
            if (
                self.max_in_flight is not None
                and in_flight_step_count > self.max_in_flight
            ):
                raise RuntimeError(
                    "Scheduler trace exceeded configured in-flight depth: "
                    f"{in_flight_step_count} > {self.max_in_flight}"
                )
            self.pending[key] = {
                "source_iteration_id": iteration,
                "trace_step_id": f"vllm:{iteration}",
                "schedule_start_ns": started_ns,
                "schedule_end_ns": schedule_end_ns,
                "scheduled_requests": scheduled_requests,
                "in_flight_step_count": in_flight_step_count,
                "device_execution_start_ns": None,
                "device_execution_end_ns": None,
            }
            if (
                self.configured_cudagraph_mode is not None
                and not self.worker_dispatch_profile
            ):
                compilation_config = scheduler.vllm_config.compilation_config
                capture_limit = compilation_config.max_cudagraph_capture_size
                if (
                    self.configured_cudagraph_mode == "NONE"
                    and capture_limit == 0
                ):
                    capture_limit = None
                if (
                    capture_limit is not None
                    and (
                        isinstance(capture_limit, bool)
                        or not isinstance(capture_limit, int)
                        or capture_limit < 1
                    )
                ):
                    raise TypeError(
                        "runtime CUDAGraph capture limit must be a positive int"
                    )
                self.pending[key]["capture_limit_tokens"] = capture_limit

    def executor_started(
        self,
        executor_or_scheduler_output: object,
        scheduler_output_or_started_ns: object,
        started_ns: int | None = None,
    ) -> None:
        if self.configured_cudagraph_mode is None:
            return
        if started_ns is None:
            # Keep the pre-PD unit helper signature for old, non-runtime trace
            # tests.  A real PD trace always follows the three-argument path
            # below and must record executor-derived depth evidence.
            executor = None
            scheduler_output = executor_or_scheduler_output
            started = scheduler_output_or_started_ns
            if isinstance(started, bool) or not isinstance(started, int):
                raise TypeError("Scheduler trace execution start must be an integer")
        else:
            executor = executor_or_scheduler_output
            scheduler_output = scheduler_output_or_started_ns
            started = started_ns
        with self.lock:
            if (
                self.deferred_io
                and executor is not None
                and self.configured_depth_evidence is None
            ):
                # Resolve the depth evidence at the first (warmup) submission
                # so no formal step pays its filesystem lookups.  The value is
                # a runtime constant and is cached for the formal steps.
                probe_started_ns = time.monotonic_ns()
                self._runtime_depth_evidence(executor)
                self.depth_evidence_probe_ns = (
                    time.monotonic_ns() - probe_started_ns
                )
            key = id(scheduler_output)
            if key not in self.pending:
                return
            pending = self.pending[key]
            evidence = (
                None
                if executor is None
                else self._runtime_depth_evidence(executor)
            )
            if pending["device_execution_start_ns"] is not None:
                raise RuntimeError(
                    "Scheduler trace observed duplicate executor submission"
                )
            setattr(scheduler_output, "_midlayer_scheduler_trace_step_id", (
                pending["trace_step_id"]
            ))
            pending["device_execution_start_ns"] = started
            if evidence is not None:
                pending["scheduler_pipeline_depth_evidence"] = evidence

    def _runtime_depth_evidence(self, executor: object) -> Mapping[str, object]:
        if self.configured_depth_evidence is not None:
            return self.configured_depth_evidence
        executor_class = type(executor)
        if hasattr(executor, "max_concurrent_batches"):
            configured_depth = executor.max_concurrent_batches
            source_kind = "runtime_executor_property"
            source_field = "max_concurrent_batches"
            source_owner = executor_class
        elif hasattr(executor, "vllm_config"):
            vllm_config = executor.vllm_config
            if not hasattr(vllm_config, "max_concurrent_batches"):
                raise AttributeError(
                    "runtime executor vllm_config is missing max_concurrent_batches"
                )
            configured_depth = vllm_config.max_concurrent_batches
            source_kind = "runtime_vllm_config_field"
            source_field = "vllm_config.max_concurrent_batches"
            source_owner = type(vllm_config)
        else:
            raise AttributeError(
                "runtime executor is missing max_concurrent_batches evidence"
            )
        if (
            isinstance(configured_depth, bool)
            or not isinstance(configured_depth, int)
            or configured_depth < 1
        ):
            raise TypeError(
                "runtime executor max_concurrent_batches must be a positive int"
            )
        if self.max_in_flight is not None and configured_depth != self.max_in_flight:
            raise RuntimeError(
                "runtime executor depth disagrees with trace configuration: "
                f"{configured_depth} != {self.max_in_flight}"
            )
        parallel_config = executor.vllm_config.parallel_config
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if (
            isinstance(pipeline_parallel_size, bool)
            or not isinstance(pipeline_parallel_size, int)
            or pipeline_parallel_size < 1
        ):
            raise TypeError("runtime executor pipeline parallel size is invalid")
        source_path = inspect.getfile(source_owner)
        import vllm

        evidence: Mapping[str, object] = {
            "source_kind": source_kind,
            "source_field": source_field,
            "configured_depth": configured_depth,
            "executor_class": (
                f"{executor_class.__module__}.{executor_class.__qualname__}"
            ),
            "pipeline_parallel_size": pipeline_parallel_size,
            "runtime_version": vllm.__version__,
            "source_identity": {
                "module_file": str(Path(source_path).resolve()),
                "module_name": executor_class.__module__,
            },
        }
        self.configured_depth_evidence = evidence
        return evidence

    def executor_finished(self, scheduler_output: object, ended_ns: int) -> None:
        if self.configured_cudagraph_mode is None:
            return
        with self.lock:
            key = id(scheduler_output)
            if key not in self.pending:
                return
            pending = self.pending[key]
            if pending["device_execution_start_ns"] is None:
                raise RuntimeError(
                    "Scheduler trace executor completed before submission"
                )
            if pending["device_execution_end_ns"] is not None:
                raise RuntimeError(
                    "Scheduler trace observed duplicate executor completion"
                )
            pending["device_execution_end_ns"] = ended_ns

    def executor_completion_observed_by_scheduler(
        self, scheduler_output: object, observed_ns: int
    ) -> None:
        """Record the completion boundary when Scheduler starts consuming output.

        An async executor Future can be complete while Python has not yet run
        its done callback.  ``Scheduler.update_from_output`` is only entered
        after that Future's result is available, so its entry is a real host
        completion observation.  It is used only when the callback did not
        already provide the earlier completion timestamp.
        """

        if self.configured_cudagraph_mode is None:
            return
        with self.lock:
            key = id(scheduler_output)
            if key not in self.pending:
                return
            pending = self.pending[key]
            if pending["device_execution_start_ns"] is None:
                raise RuntimeError(
                    "Scheduler trace observed executor completion before submission"
                )
            if pending["device_execution_end_ns"] is None:
                pending["device_execution_end_ns"] = observed_ns

    def after_update(
        self,
        scheduler_output: object,
        update_start_ns: int,
        update_end_ns: int,
        engine_outputs: object,
        model_runner_output: object | None = None,
    ) -> None:
        self._raise_deferred_io_error()
        with self.lock:
            key = id(scheduler_output)
            if key not in self.pending:
                return
            pending = self.pending.pop(key)
            self.record_bootstrap_marker_event("formal_scheduler_update")
            source_iteration_id = pending["source_iteration_id"]
            scheduled_requests = pending["scheduled_requests"]
            finished_indexes = _finished_request_indexes(engine_outputs)
            duplicate = finished_indexes & self.finished_request_indexes
            if duplicate:
                raise RuntimeError(
                    "Scheduler trace observed duplicate request completion: "
                    f"{sorted(duplicate)}"
                )
            self.finished_request_indexes.update(finished_indexes)
            prefill_tokens = sum(value[1] for value in scheduled_requests)
            decode_tokens = sum(value[2] for value in scheduled_requests)
            record = {
                **pending,
                "scheduler_update_start_ns": update_start_ns,
                "scheduler_update_end_ns": update_end_ns,
                "prefill_request_count": sum(
                    value[1] > 0 for value in scheduled_requests
                ),
                "decode_request_count": sum(
                    value[2] > 0 for value in scheduled_requests
                ),
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "finished_request_indexes": tuple(sorted(finished_indexes)),
            }
            if self.configured_cudagraph_mode is not None:
                if self.worker_dispatch_profile:
                    # Non-output ranks may still be appending this step's
                    # observation; attach them as they arrive.  The trace is
                    # sealed only after every step is resolved.
                    self.a800_unresolved_steps.append({
                        "record": record,
                        "trace_step_id": pending["trace_step_id"],
                        "scheduled_tokens": prefill_tokens + decode_tokens,
                        "observations": {},
                    })
                    self._resolve_a800_rank_observations()
                elif self.hardware_id in (
                    None, "metax_c500", "ascend_910b", "nvidia_h200"
                ):
                    record.update(self._runtime_graph_record(
                        model_runner_output,
                        prefill_tokens + decode_tokens,
                        pending["capture_limit_tokens"],
                    ))
                else:
                    raise ValueError(
                        f"unsupported Scheduler trace hardware: {self.hardware_id!r}"
                    )
                if record["device_execution_start_ns"] is None:
                    raise RuntimeError(
                        "Scheduler trace is missing executor submission timing"
                    )
                if record["device_execution_end_ns"] is None:
                    raise RuntimeError(
                        "Scheduler trace is missing executor completion timing"
                    )
                if (
                    self.pd_run_identity is not None
                    and "scheduler_pipeline_depth_evidence" not in pending
                ):
                    raise RuntimeError(
                        "Scheduler trace is missing runtime executor depth evidence"
                    )
                if "scheduler_pipeline_depth_evidence" in pending:
                    record["scheduler_pipeline_depth_evidence"] = pending[
                        "scheduler_pipeline_depth_evidence"
                    ]
            self.records.append(record)
            # With async scheduling, request completion can be observed while
            # a previously submitted depth-2 step is still pending its update.
            # Preserve that real final step and serialize only after the
            # pipeline has drained.
            if (
                len(self.finished_request_indexes) == self.request_count
                and not self.pending
            ):
                self.record_bootstrap_marker_event("scheduler_trace_complete")
                self.record_bootstrap_marker_event("scheduler_trace_write_start")
                if self.deferred_io:
                    # This update's outputs (the last request's final token)
                    # are returned now; the runner waits for the atomic
                    # publication before it stops the server.
                    self._submit_io(self._publish_complete_trace)
                    return
                try:
                    self._write_complete_trace()
                except Exception as error:
                    self._record_bootstrap_marker_failure(
                        "scheduler_trace_write_failed", error
                    )
                    raise
                self.record_bootstrap_marker_event("scheduler_trace_write_complete")

    def _publish_complete_trace(self) -> None:
        """Write the sealed trace on the trace I/O thread."""

        with self.lock:
            publish_started_ns = time.monotonic_ns()
            try:
                self._write_complete_trace()
            except Exception as error:
                self._record_bootstrap_marker_failure(
                    "scheduler_trace_write_failed", error
                )
                raise
            publish_ns = time.monotonic_ns() - publish_started_ns
            # Already on the trace I/O thread: append directly so the marker
            # is covered by ``wait_for_deferred_io`` barriers.
            complete_marker = self._new_bootstrap_marker(
                "scheduler_trace_write_complete")
            if complete_marker is not None:
                _append_marker(
                    os.environ[_BOOTSTRAP_MARKER_ENV], complete_marker)
        probe = (
            "not run"
            if self.depth_evidence_probe_ns is None
            else f"{self.depth_evidence_probe_ns / 1e6:.3f} ms"
        )
        print(
            "midlayer Scheduler trace published off the Scheduler thread: "
            f"{len(self.records)} steps in {publish_ns / 1e6:.1f} ms; "
            f"pre-benchmark depth-evidence probe {probe}",
            file=sys.stderr,
            flush=True,
        )

    def _read_a800_rank_observations(self, rank: int) -> None:
        """Move newly appended complete rows of one rank into its cache.

        A row is consumed only once its terminating newline is visible, so a
        concurrently appended row is re-read whole on the next call.
        """

        path = _worker_observation_path(self.output_path, rank)
        if rank not in self.a800_rank_observation_cache:
            self.a800_rank_observation_cache[rank] = {}
        cache = self.a800_rank_observation_cache[rank]
        if rank not in self.a800_rank_observation_offsets:
            self.a800_rank_observation_offsets[rank] = 0
        if not path.is_file():
            # The first formal append creates the file; a non-output rank may
            # not have reached it yet.
            return
        offset = self.a800_rank_observation_offsets[rank]
        with path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read()
        complete_length = data.rfind(b"\n") + 1
        if complete_length == 0:
            return
        for line_index, line in enumerate(
            data[:complete_length].decode("utf-8").split("\n")[:-1]
        ):
            if not line.strip():
                raise ValueError(
                    "A800 worker observation contains an empty JSONL row: "
                    f"rank={rank}, offset={offset}, line={line_index}"
                )
            value = json.loads(line)
            observation_step_id = value["trace_step_id"]
            if observation_step_id in cache:
                raise RuntimeError(
                    "A800 worker observation duplicated a trace step: "
                    f"rank={rank}, step={observation_step_id}"
                )
            cache[observation_step_id] = value
        self.a800_rank_observation_offsets[rank] = offset + complete_length

    def _resolve_a800_rank_observations(self) -> None:
        """Attach every arrived worker observation to its recorded step."""

        if self.worker_rank_count is None:
            raise RuntimeError("A800 Scheduler trace is missing worker rank count")
        for rank in range(self.worker_rank_count):
            self._read_a800_rank_observations(rank)
        unresolved = []
        for entry in self.a800_unresolved_steps:
            trace_step_id = entry["trace_step_id"]
            observations = entry["observations"]
            for rank in range(self.worker_rank_count):
                if rank in observations:
                    continue
                cache = self.a800_rank_observation_cache[rank]
                if trace_step_id not in cache:
                    continue
                observation = cache.pop(trace_step_id)
                _validate_a800_rank_observation(
                    observation,
                    expected_rank=rank,
                    expected_step_id=trace_step_id,
                    expected_scheduled_tokens=entry["scheduled_tokens"],
                    expected_configured_mode=self.configured_cudagraph_mode,
                )
                observations[rank] = observation
            if len(observations) == self.worker_rank_count:
                entry["record"]["rank_observations"] = [
                    observations[rank]
                    for rank in range(self.worker_rank_count)
                ]
            else:
                unresolved.append(entry)
        self.a800_unresolved_steps = unresolved

    def _wait_for_a800_rank_observations(self) -> None:
        """Block until every recorded step has all rank observations.

        Fails loudly after ``_WORKER_OBSERVATION_ARRIVAL_TIMEOUT_S``; nothing
        is dropped or synthesized.
        """

        if not self.a800_unresolved_steps:
            return
        deadline_ns = time.monotonic_ns() + int(
            _WORKER_OBSERVATION_ARRIVAL_TIMEOUT_S * 1e9
        )
        self._resolve_a800_rank_observations()
        while self.a800_unresolved_steps:
            if time.monotonic_ns() >= deadline_ns:
                missing = [
                    (
                        entry["trace_step_id"],
                        [
                            rank
                            for rank in range(self.worker_rank_count)
                            if rank not in entry["observations"]
                        ],
                    )
                    for entry in self.a800_unresolved_steps
                ]
                files = {
                    rank: str(_worker_observation_path(self.output_path, rank))
                    for rank in range(self.worker_rank_count)
                }
                raise RuntimeError(
                    "A800 worker observation did not arrive within "
                    f"{_WORKER_OBSERVATION_ARRIVAL_TIMEOUT_S} s of sealing the "
                    f"Scheduler trace: missing (step, ranks)={missing}, "
                    f"files={files}"
                )
            time.sleep(_WORKER_OBSERVATION_POLL_S)
            self._resolve_a800_rank_observations()

    def _runtime_graph_record(
        self,
        model_runner_output: object | None,
        scheduled_tokens: int,
        capture_limit_tokens: int | None,
    ) -> dict[str, Any]:
        if model_runner_output is None or not hasattr(
            model_runner_output, "cudagraph_stats"
        ):
            raise RuntimeError(
                "Scheduler trace requires ModelRunnerOutput.cudagraph_stats"
            )
        stats = model_runner_output.cudagraph_stats
        if stats is None:
            raise RuntimeError(
                "Scheduler trace requires --cudagraph-metrics runtime stats"
            )
        values = {
            "num_unpadded_tokens": stats.num_unpadded_tokens,
            "padded_total_tokens": stats.num_padded_tokens,
            "padding_tokens": stats.num_paddings,
        }
        for field, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"runtime CUDAGraph {field} must be a non-negative int")
        if values["num_unpadded_tokens"] != scheduled_tokens:
            raise RuntimeError(
                "runtime CUDAGraph unpadded tokens disagree with Scheduler: "
                f"{values['num_unpadded_tokens']} != {scheduled_tokens}"
            )
        if values["padded_total_tokens"] != (
            values["num_unpadded_tokens"] + values["padding_tokens"]
        ):
            raise RuntimeError("runtime CUDAGraph padding totals are inconsistent")
        runtime_mode = _runtime_cudagraph_mode(stats.runtime_mode)
        graph_executed = runtime_mode != "NONE"
        fallback_reason = None
        if not graph_executed:
            if self.configured_cudagraph_mode == "NONE":
                if capture_limit_tokens is not None:
                    raise RuntimeError(
                        "disabled CUDAGraph unexpectedly has a capture limit"
                    )
            else:
                if capture_limit_tokens is None:
                    raise RuntimeError(
                        "enabled CUDAGraph requires a capture limit"
                    )
                fallback_reason = (
                    "above_capture_limit"
                    if scheduled_tokens > capture_limit_tokens
                    else "runtime_dispatch_no_matching_graph"
                )
        return {
            "configured_cudagraph_mode": self.configured_cudagraph_mode,
            "runtime_cudagraph_mode": runtime_mode,
            **values,
            "capture_bucket_tokens": (
                values["padded_total_tokens"] if graph_executed else None
            ),
            "capture_limit_tokens": capture_limit_tokens,
            "graph_fallback_reason": fallback_reason,
            "device_execution_clock": "time.perf_counter_ns",
            "device_execution_source": "host_monotonic_executor_rpc_envelope",
        }

    def worker_execute_started(
        self, worker_rank: object, scheduler_output: object
    ) -> None:
        if not self.worker_dispatch_profile:
            return
        if hasattr(scheduler_output, "_midlayer_scheduler_trace_step_id"):
            trace_step_id = scheduler_output._midlayer_scheduler_trace_step_id
        elif self.counter_only:
            if not _scheduler_output_has_formal_request(scheduler_output):
                return
            trace_step_id = f"counter-only:{self.next_counter_only_step}"
            self.next_counter_only_step += 1
        else:
            return
        if isinstance(worker_rank, bool) or not isinstance(worker_rank, int):
            raise TypeError("A800 worker rank must be an integer")
        if self.worker_rank_count is None or not 0 <= worker_rank < self.worker_rank_count:
            raise ValueError("A800 worker rank is outside the configured trace range")
        if not isinstance(trace_step_id, str) or not trace_step_id:
            raise TypeError("A800 worker trace step identity must be non-empty text")
        scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (
            isinstance(scheduled_tokens, bool)
            or not isinstance(scheduled_tokens, int)
            or scheduled_tokens < 1
        ):
            raise ValueError("A800 worker trace requires positive scheduled tokens")
        self.worker_rank = worker_rank
        if not self.worker_counter_after_warmup_written:
            self._write_worker_counter_snapshot(
                "after_warmup_before_formal", trace_step_id
            )
            self.worker_counter_after_warmup_written = True
        if hasattr(self.worker_execution, "current"):
            raise RuntimeError("A800 worker trace observed overlapping executions")
        self.worker_execution.current = {
            "rank": worker_rank,
            "trace_step_id": trace_step_id,
            "scheduled_tokens": scheduled_tokens,
            "worker_execution_start_ns": time.perf_counter_ns(),
            "dispatches": [],
        }

    def worker_dispatch(
        self,
        dispatcher: object,
        dispatch_input: object,
        runtime_mode: object,
        selected_graph_key: object | None,
    ) -> None:
        if not hasattr(self.worker_execution, "current"):
            return
        current = self.worker_execution.current
        dispatches = current["dispatches"]
        if dispatches:
            raise RuntimeError("A800 worker trace observed multiple dispatcher calls")
        input_key = _batch_descriptor_record(dispatch_input)
        selected_key = (
            None
            if selected_graph_key is None
            else _batch_descriptor_record(selected_graph_key)
        )
        normalized_runtime_mode = _runtime_cudagraph_mode(str(runtime_mode))
        configured_mode = _configured_cudagraph_mode(
            dispatcher.compilation_config.cudagraph_mode
        )
        fallback_reason = None
        if normalized_runtime_mode == "NONE":
            if selected_key is not None:
                raise RuntimeError("NONE CUDAGraph dispatch selected a graph key")
            fallback_reason = (
                "dispatch_keys_uninitialized"
                if not dispatcher.keys_initialized
                else "runtime_dispatch_no_matching_graph"
            )
        elif selected_key is None:
            raise RuntimeError("Graph CUDAGraph dispatch omitted selected graph key")
        current["dispatches"].append({
            "configured_cudagraph_mode": configured_mode,
            "runtime_cudagraph_mode": normalized_runtime_mode,
            "dispatch_input_key": input_key,
            "selected_graph_key": selected_key,
            "graph_fallback_reason": fallback_reason,
        })

    def worker_execute_finished(self) -> None:
        if not hasattr(self.worker_execution, "current"):
            return
        current = self.worker_execution.current
        del self.worker_execution.current
        if self.counter_only:
            return
        dispatches = current["dispatches"]
        if len(dispatches) != 1:
            raise RuntimeError(
                "A800 worker trace requires exactly one runtime dispatcher result"
            )
        dispatch = dispatches[0]
        scheduled_tokens = current["scheduled_tokens"]
        padded_total_tokens = dispatch["dispatch_input_key"]["num_tokens"]
        if padded_total_tokens < scheduled_tokens:
            raise RuntimeError(
                "A800 worker dispatcher key has fewer tokens than Scheduler"
            )
        observation = {
            "schema_version": 1,
            "source": "vllm_v1_worker_cudagraph_dispatch",
            "rank": current["rank"],
            "trace_step_id": current["trace_step_id"],
            "scheduled_tokens": scheduled_tokens,
            **dispatch,
            "num_unpadded_tokens": scheduled_tokens,
            "padded_total_tokens": padded_total_tokens,
            "padding_tokens": padded_total_tokens - scheduled_tokens,
            "worker_execution_start_ns": current["worker_execution_start_ns"],
            "worker_execution_end_ns": time.perf_counter_ns(),
            "worker_execution_clock": "time.perf_counter_ns",
            "worker_execution_source": "vllm_v1_worker_execute_model_envelope",
        }
        _append_jsonl(_worker_observation_path(self.output_path, current["rank"]), observation)

    def worker_execute_failed(self) -> None:
        if hasattr(self.worker_execution, "current"):
            del self.worker_execution.current

    def worker_snapshot_after_benchmark(self) -> None:
        if not self.worker_dispatch_profile or self.worker_rank is None:
            return
        if not self.worker_counter_after_warmup_written:
            raise RuntimeError(
                "A800 worker benchmark snapshot arrived before warmup snapshot"
            )
        if self.worker_counter_after_benchmark_written:
            raise RuntimeError("A800 worker benchmark counter snapshot duplicated")
        self._write_worker_counter_snapshot("after_formal_benchmark", None)
        self.worker_counter_after_benchmark_written = True

    def _write_worker_counter_snapshot(
        self, phase: str, trace_step_id: str | None
    ) -> None:
        if self.worker_rank is None:
            raise RuntimeError("A800 worker counter snapshot has no worker rank")
        from dataclasses import asdict
        from vllm.compilation.counter import compilation_counter

        counter = asdict(compilation_counter)
        for field in _A800_COUNTER_FIELDS:
            value = counter[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    "A800 compilation counter field must be a non-negative int: "
                    f"{field}"
                )
        _append_jsonl(_worker_counter_path(self.output_path, self.worker_rank), {
            "schema_version": 1,
            "source": "vllm.compilation.counter.compilation_counter",
            "rank": self.worker_rank,
            "phase": phase,
            "trace_step_id": trace_step_id,
            "timestamp_ns": time.perf_counter_ns(),
            "counter": counter,
        })

    def _write_complete_trace(self) -> None:
        if self.written:
            raise RuntimeError("Scheduler trace attempted to write twice")
        # Get the request stage spans on disk before this run is reported
        # complete; atexit is only the backstop.
        flush_request_stage_recorders()
        if self.pending:
            raise RuntimeError("Scheduler trace completed with pending steps")
        if self.worker_dispatch_profile:
            self._wait_for_a800_rank_observations()
        ordered = sorted(
            self.records, key=lambda value: value["source_iteration_id"]
        )
        if not ordered:
            raise RuntimeError("Scheduler trace completed without records")
        origin_ns = self._trace_origin_ns(ordered[0]["schedule_start_ns"])
        cumulative_tokens = 0
        cumulative_completed = 0
        lines = []
        for step_index, source in enumerate(ordered):
            source_iteration_id = source["source_iteration_id"]
            scheduled_requests = source["scheduled_requests"]
            prefill_tokens = source["prefill_tokens"]
            decode_tokens = source["decode_tokens"]
            finished_request_indexes = source["finished_request_indexes"]
            scheduled_tokens = prefill_tokens + decode_tokens
            cumulative_tokens += scheduled_tokens
            cumulative_completed += len(finished_request_indexes)
            record = {
                "step_index": step_index,
                "step_id": f"vllm:{source_iteration_id}",
                "source_iteration_id": source_iteration_id,
                "step_start_ns": source["schedule_start_ns"] - origin_ns,
                "schedule_end_ns": source["schedule_end_ns"] - origin_ns,
                "update_start_ns": (
                    source["scheduler_update_start_ns"] - origin_ns
                ),
                "step_end_ns": source["scheduler_update_end_ns"] - origin_ns,
                "scheduled_requests": [
                    {
                        "request_idx": value[0],
                        "prefill_tokens": value[1],
                        "decode_tokens": value[2],
                        "scheduled_tokens": value[3],
                    }
                    for value in scheduled_requests
                ],
                "scheduled_request_count": len(scheduled_requests),
                "prefill_request_count": source["prefill_request_count"],
                "decode_request_count": source["decode_request_count"],
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "scheduled_tokens": scheduled_tokens,
                "finished_request_indexes": list(finished_request_indexes),
                "cumulative_scheduled_tokens": cumulative_tokens,
                "cumulative_completed_requests": cumulative_completed,
            }
            if self.pd_run_identity is not None:
                record.update(self.pd_run_identity)
            if self.configured_cudagraph_mode is not None:
                record["schedule_start_ns"] = (
                    source["schedule_start_ns"] - origin_ns
                )
                record["scheduler_update_start_ns"] = (
                    source["scheduler_update_start_ns"] - origin_ns
                )
                record["scheduler_update_end_ns"] = (
                    source["scheduler_update_end_ns"] - origin_ns
                )
                for field in (
                    "configured_cudagraph_mode",
                    "runtime_cudagraph_mode",
                    "num_unpadded_tokens",
                    "capture_bucket_tokens",
                    "capture_limit_tokens",
                    "padded_total_tokens",
                    "padding_tokens",
                    "graph_fallback_reason",
                    "device_execution_clock",
                    "device_execution_source",
                    "in_flight_step_count",
                ):
                    if not self.worker_dispatch_profile:
                        record[field] = source[field]
                if not self.worker_dispatch_profile:
                    if "scheduler_pipeline_depth_evidence" in source:
                        record["scheduler_pipeline_depth_evidence"] = source[
                            "scheduler_pipeline_depth_evidence"
                        ]
                    elif self.pd_run_identity is not None:
                        raise RuntimeError(
                            "PD Scheduler trace is missing runtime executor "
                            "depth evidence"
                        )
                record["device_execution_start_ns"] = (
                    source["device_execution_start_ns"] - origin_ns
                )
                record["device_execution_end_ns"] = (
                    source["device_execution_end_ns"] - origin_ns
                )
                if self.worker_dispatch_profile:
                    record["device_execution_clock"] = (
                        "time.perf_counter_ns"
                    )
                    record["device_execution_source"] = (
                        "host_monotonic_executor_rpc_envelope"
                    )
                    record["in_flight_step_count"] = source[
                        "in_flight_step_count"
                    ]
                    if self.pd_run_identity is not None:
                        record["scheduler_pipeline_depth_evidence"] = source[
                            "scheduler_pipeline_depth_evidence"
                        ]
                    observations = []
                    for source_observation in source["rank_observations"]:
                        observation = dict(source_observation)
                        observation["worker_execution_start_ns"] = (
                            observation["worker_execution_start_ns"] - origin_ns
                        )
                        observation["worker_execution_end_ns"] = (
                            observation["worker_execution_end_ns"] - origin_ns
                        )
                        observations.append(observation)
                    record["rank_observations"] = observations
            lines.append(json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
        if cumulative_completed != self.request_count:
            raise RuntimeError("Scheduler trace completion count is inconsistent")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(
            f".{self.output_path.name}.{os.getpid()}.tmp"
        )
        if self.output_path.exists() or temporary.exists():
            raise FileExistsError(
                f"refusing to replace Scheduler trace output: {self.output_path}"
            )
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not self.deferred_io:
            os.replace(temporary, self.output_path)
        if self.configured_depth_evidence is not None:
            self._write_runtime_metadata(origin_ns)
        if self.request_add_output_path is not None:
            self._write_request_add_trace()
        self._write_warmup_evidence()
        if self.pd_handoff_recorder is not None:
            self.pd_handoff_recorder.write_complete()
        if self.deferred_io:
            # The runner treats the step trace as the publication signal and
            # reads the sidecars as soon as it appears; off the Scheduler
            # thread the sidecars are therefore published first.
            os.replace(temporary, self.output_path)
        self.written = True

    def _write_request_add_trace(self) -> None:
        if self.request_add_output_path is None:
            raise RuntimeError("Scheduler request-add output path is missing")
        expected = set(range(self.request_count))
        if set(self.request_add_end_ns) != expected:
            raise RuntimeError(
                "Scheduler request-add trace does not cover every formal request"
            )
        records = [
            {
                "request_idx": request_idx,
                "scheduler_add_end_monotonic_ns": self.request_add_end_ns[
                    request_idx
                ],
            }
            for request_idx in range(self.request_count)
        ]
        payload = {
            "schema_version": 1,
            "clock_domain": "shared_host_time.perf_counter_ns",
            "boundary": "Scheduler.add_request return",
            "request_count": self.request_count,
            "records": records,
        }
        path = self.request_add_output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        if path.exists() or temporary.exists():
            raise FileExistsError(
                f"refusing to replace Scheduler request-add trace: {path}"
            )
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _write_runtime_metadata(self, trace_origin_monotonic_ns: int) -> None:
        if self.configured_depth_evidence is None:
            if self.configured_cudagraph_mode is None:
                return
            raise RuntimeError(
                "Scheduler trace is missing live Executor depth evidence"
            )
        evidence = dict(self.configured_depth_evidence)
        configured_depth = evidence["configured_depth"]
        if (
            isinstance(configured_depth, bool)
            or not isinstance(configured_depth, int)
            or configured_depth < 1
        ):
            raise TypeError("Scheduler runtime depth must be a positive integer")
        source_kind = evidence["source_kind"]
        if source_kind == "runtime_executor_property":
            del evidence["configured_depth"]
            if _SOURCE_IDENTITY_ENV not in os.environ:
                raise RuntimeError(
                    f"{_SOURCE_IDENTITY_ENV} is required for Scheduler trace"
                )
            source_identity = os.environ[_SOURCE_IDENTITY_ENV]
            if not source_identity:
                raise ValueError(
                    f"{_SOURCE_IDENTITY_ENV} must be non-empty"
                )
            evidence["source_identity"] = source_identity
        elif source_kind == "runtime_vllm_config_field":
            pass
        else:
            raise ValueError("Scheduler runtime depth source is unsupported")
        runtime_path = Path(f"{self.output_path}.runtime.json")
        clock_path = Path(f"{self.output_path}.clock.json")
        for path in (runtime_path, clock_path):
            if path.exists():
                raise FileExistsError(
                    f"refusing to replace Scheduler metadata: {path}"
                )
        runtime_payload = {
            "schema_version": 1,
            "scheduler_pipeline_depth": configured_depth,
            "scheduler_pipeline_depth_evidence": evidence,
        }
        boot_id_path = Path("/proc/sys/kernel/random/boot_id")
        host_boot_id = boot_id_path.read_text(encoding="utf-8").strip()
        if not host_boot_id:
            raise RuntimeError("host boot ID is empty")
        clock_payload = {
            "schema_version": 1,
            "clock_domain": "shared_host_time.perf_counter_ns",
            "host_boot_id": host_boot_id,
            "trace_origin_monotonic_ns": trace_origin_monotonic_ns,
        }
        for path, payload in (
            (runtime_path, runtime_payload),
            (clock_path, clock_payload),
        ):
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            if temporary.exists():
                raise FileExistsError(
                    f"refusing to replace Scheduler metadata temporary: {temporary}"
                )
            temporary.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)

    def _trace_origin_ns(self, first_schedule_start_ns: int) -> int:
        if self.pd_run_identity is None:
            return first_schedule_start_ns
        origin = self.pd_run_identity["run_origin_monotonic_ns"]
        if isinstance(origin, bool) or not isinstance(origin, int) or origin < 0:
            raise TypeError("PD run origin must be a non-negative integer")
        if first_schedule_start_ns < origin:
            raise RuntimeError("PD Scheduler trace predates the shared run origin")
        return origin

    def _write_warmup_evidence(self) -> None:
        warmup_path = Path(f"{self.output_path}.warmup.json")
        temporary = warmup_path.with_name(
            f".{warmup_path.name}.{os.getpid()}.tmp"
        )
        if warmup_path.exists() or temporary.exists():
            raise FileExistsError(
                f"refusing to replace Scheduler warmup evidence: {warmup_path}"
            )
        evidence = {
            "schema_version": 1,
            "source": "vllm_scheduler_runtime",
            "step_count": self.warmup_step_count,
            "mixed_prefill_decode_step_count": self.warmup_mixed_step_count,
            "max_scheduled_request_count": self.warmup_max_scheduled_requests,
            "scheduled_tokens": self.warmup_scheduled_tokens,
            "request_ids": sorted(self.warmup_request_ids),
        }
        temporary.write_text(
            json.dumps(
                evidence,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, warmup_path)


def _worker_observation_path(output_path: Path, rank: int) -> Path:
    return Path(f"{output_path}.rank-{rank}.jsonl")


def _worker_counter_path(output_path: Path, rank: int) -> Path:
    return Path(f"{output_path}.compilation-counter.rank-{rank}.jsonl")


def _append_marker(marker_path: str, value: Mapping[str, Any]) -> None:
    _append_jsonl(Path(marker_path).resolve(), value)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
        handle.flush()


def _batch_descriptor_record(value: object) -> dict[str, object]:
    if not hasattr(value, "num_tokens") or not hasattr(value, "uniform_decode"):
        raise TypeError("CUDAGraph batch descriptor is missing required fields")
    num_tokens = value.num_tokens
    uniform_decode = value.uniform_decode
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
        raise TypeError("CUDAGraph batch descriptor num_tokens must be positive int")
    if not isinstance(uniform_decode, bool):
        raise TypeError("CUDAGraph batch descriptor uniform_decode must be bool")
    return {
        "num_tokens": num_tokens,
        "uniform_decode": uniform_decode,
    }


def _configured_cudagraph_mode(value: object) -> str:
    normalized = value.name if hasattr(value, "name") else str(value)
    if not isinstance(normalized, str):
        raise TypeError("configured CUDAGraph mode must be text")
    normalized = normalized.rsplit(".", 1)[-1]
    if normalized != "FULL_AND_PIECEWISE":
        raise ValueError(
            "A800 runtime must configure FULL_AND_PIECEWISE CUDAGraph mode"
        )
    return normalized


def _validate_a800_rank_observation(
    observation: object,
    *,
    expected_rank: int,
    expected_step_id: str,
    expected_scheduled_tokens: int,
    expected_configured_mode: str | None,
) -> None:
    if not isinstance(observation, Mapping):
        raise TypeError("A800 rank observation must be an object")
    required = {
        "schema_version",
        "source",
        "rank",
        "trace_step_id",
        "scheduled_tokens",
        "configured_cudagraph_mode",
        "runtime_cudagraph_mode",
        "dispatch_input_key",
        "selected_graph_key",
        "graph_fallback_reason",
        "num_unpadded_tokens",
        "padded_total_tokens",
        "padding_tokens",
        "worker_execution_start_ns",
        "worker_execution_end_ns",
        "worker_execution_clock",
        "worker_execution_source",
    }
    if set(observation) != required:
        raise ValueError("A800 rank observation fields differ from the contract")
    if observation["schema_version"] != 1:
        raise ValueError("A800 rank observation schema version is invalid")
    if observation["source"] != "vllm_v1_worker_cudagraph_dispatch":
        raise ValueError("A800 rank observation source is invalid")
    if observation["rank"] != expected_rank:
        raise ValueError("A800 rank observation rank disagrees with file identity")
    if observation["trace_step_id"] != expected_step_id:
        raise ValueError("A800 rank observation step identity disagrees")
    if observation["scheduled_tokens"] != expected_scheduled_tokens:
        raise ValueError("A800 rank observation Scheduler tokens disagree")
    if observation["configured_cudagraph_mode"] != expected_configured_mode:
        raise ValueError("A800 rank observation configured mode disagrees")
    runtime_mode = observation["runtime_cudagraph_mode"]
    if runtime_mode not in {"FULL", "PIECEWISE", "NONE"}:
        raise ValueError("A800 rank observation runtime mode is invalid")
    input_key = observation["dispatch_input_key"]
    if not isinstance(input_key, Mapping) or set(input_key) != {
        "num_tokens", "uniform_decode"
    }:
        raise TypeError("A800 dispatch input key is invalid")
    _validate_batch_descriptor_mapping(input_key, "A800 dispatch input key")
    selected_key = observation["selected_graph_key"]
    if runtime_mode == "NONE":
        if selected_key is not None:
            raise ValueError("NONE A800 rank observation selected a graph key")
        fallback_reason = observation["graph_fallback_reason"]
        if not isinstance(fallback_reason, str) or not fallback_reason:
            raise ValueError("NONE A800 rank observation needs a fallback reason")
    elif runtime_mode in ("FULL", "PIECEWISE"):
        if not isinstance(selected_key, Mapping) or set(selected_key) != {
            "num_tokens", "uniform_decode"
        }:
            raise TypeError("Graph A800 rank observation selected key is invalid")
        _validate_batch_descriptor_mapping(
            selected_key, "A800 selected graph key"
        )
        if observation["graph_fallback_reason"] is not None:
            raise ValueError("Graph A800 rank observation has a fallback reason")
    else:
        raise ValueError(f"unsupported A800 runtime Graph mode: {runtime_mode!r}")
    for field in (
        "num_unpadded_tokens",
        "padded_total_tokens",
        "padding_tokens",
        "worker_execution_start_ns",
        "worker_execution_end_ns",
    ):
        value = observation[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"A800 rank observation {field} is invalid")
    if observation["num_unpadded_tokens"] != expected_scheduled_tokens:
        raise ValueError("A800 rank observation unpadded tokens disagree")
    if observation["padded_total_tokens"] != (
        expected_scheduled_tokens + observation["padding_tokens"]
    ):
        raise ValueError("A800 rank observation padding tokens disagree")
    if input_key["num_tokens"] != observation["padded_total_tokens"]:
        raise ValueError("A800 dispatch input key does not encode padded tokens")
    if observation["worker_execution_start_ns"] > observation[
        "worker_execution_end_ns"
    ]:
        raise ValueError("A800 worker execution timestamps are not causal")
    if observation["worker_execution_clock"] != "time.perf_counter_ns":
        raise ValueError("A800 worker execution clock is invalid")
    if observation["worker_execution_source"] != (
        "vllm_v1_worker_execute_model_envelope"
    ):
        raise ValueError("A800 worker execution source is invalid")


def _validate_batch_descriptor_mapping(
    value: Mapping[str, object], description: str
) -> None:
    num_tokens = value["num_tokens"]
    uniform_decode = value["uniform_decode"]
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
        raise TypeError(f"{description} num_tokens is invalid")
    if not isinstance(uniform_decode, bool):
        raise TypeError(f"{description} uniform_decode is invalid")


def _request_index(request_id: object) -> int | None:
    if not isinstance(request_id, str):
        raise TypeError("vLLM Scheduler request IDs must be strings")
    if "___decode_addr_" in request_id or "___prefill_addr_" in request_id:
        request_id = transfer_request_id(request_id)
    if request_id.startswith("request-"):
        canonical_index = request_id[8:]
        if canonical_index.isdecimal():
            return int(canonical_index)
    match = _REQUEST_ID.fullmatch(request_id)
    if match is None:
        return None
    return int(match.group("index"))


def _scheduler_output_has_formal_request(scheduler_output: object) -> bool:
    if not hasattr(scheduler_output, "num_scheduled_tokens"):
        raise TypeError("A800 worker Scheduler output is missing scheduled tokens")
    scheduled = scheduler_output.num_scheduled_tokens
    if not isinstance(scheduled, Mapping):
        raise TypeError("A800 worker Scheduler scheduled tokens must be a mapping")
    return any(_request_index(request_id) is not None for request_id in scheduled)


def _is_warmup_request(request_id: object) -> bool:
    if not isinstance(request_id, str):
        raise TypeError("vLLM Scheduler request IDs must be strings")
    normalized = request_id[5:] if request_id.startswith("cmpl-") else request_id
    return normalized.startswith("warmup-") or normalized == "sharegpt-warmup"


def _runtime_cudagraph_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("runtime CUDAGraph mode must be a string")
    normalized = value.rsplit(".", 1)[-1]
    if normalized not in {"FULL", "PIECEWISE", "NONE"}:
        raise ValueError(f"unsupported runtime CUDAGraph mode: {value!r}")
    return normalized


def _finished_request_indexes(engine_outputs: object) -> set[int]:
    if not isinstance(engine_outputs, Mapping):
        raise TypeError("vLLM Scheduler update output must be a mapping")
    finished = set()
    for output_batch in engine_outputs.values():
        if hasattr(output_batch, "outputs"):
            for output in output_batch.outputs:
                if output.finished:
                    request_idx = _request_index(output.request_id)
                    if request_idx is not None:
                        finished.add(request_idx)
        if hasattr(output_batch, "finished_requests"):
            finished_requests = output_batch.finished_requests
            if finished_requests is not None:
                for request_id in finished_requests:
                    request_idx = _request_index(request_id)
                    if request_idx is not None:
                        finished.add(request_idx)
    return finished


def _pd_run_identity_from_environment() -> Mapping[str, object] | None:
    names = (
        _PD_RUN_ID_ENV,
        _PD_RUN_ORIGIN_ENV,
        _PD_HOST_BOOT_ID_ENV,
        _PD_INSTANCE_ROLE_ENV,
        _PD_INSTANCE_ID_ENV,
    )
    present = tuple(name in os.environ for name in names)
    if not any(present):
        return None
    if not all(present):
        missing = [name for name in names if name not in os.environ]
        raise RuntimeError(
            "PD Scheduler trace identity must be complete; missing: "
            + ", ".join(missing)
        )
    run_id = os.environ[_PD_RUN_ID_ENV]
    host_boot_id = os.environ[_PD_HOST_BOOT_ID_ENV]
    instance_role = os.environ[_PD_INSTANCE_ROLE_ENV]
    instance_id = os.environ[_PD_INSTANCE_ID_ENV]
    for label, value in (
        ("PD run ID", run_id),
        ("PD host boot ID", host_boot_id),
        ("PD instance ID", instance_id),
    ):
        if not value:
            raise ValueError(f"{label} must be non-empty")
    if instance_role not in ("prefill", "decode"):
        raise ValueError("PD instance role must be prefill or decode")
    origin_text = os.environ[_PD_RUN_ORIGIN_ENV]
    if not origin_text.isdecimal():
        raise ValueError("PD run origin must be a non-negative integer")
    return {
        "run_id": run_id,
        "run_origin_monotonic_ns": int(origin_text),
        "host_boot_id": host_boot_id,
        "instance_role": instance_role,
        "instance_id": instance_id,
        "clock_domain": "shared_host_time.perf_counter_ns",
    }


def _base_request_id(request_id: str) -> str:
    if not isinstance(request_id, str) or not request_id:
        raise TypeError("PD connector request ID must be non-empty text")
    separator = request_id.find("___")
    base = request_id if separator < 0 else request_id[:separator]
    if base.startswith("cmpl-"):
        base = base[5:]
    if not base:
        raise ValueError("PD connector request ID has an empty base identity")
    match = _REQUEST_BASE_ID.fullmatch(base)
    return base if match is None else match["base"]


def _integer_attribute(value: object, attribute: str, label: str) -> int:
    if not hasattr(value, attribute):
        raise RuntimeError(f"PD connector {label} lacks {attribute}")
    result = getattr(value, attribute)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise TypeError(f"PD connector {label}.{attribute} must be non-negative int")
    return result


def _attribute(value: object, attribute: str, label: str) -> object:
    if not hasattr(value, attribute):
        raise RuntimeError(f"PD connector {label} lacks {attribute}")
    return getattr(value, attribute)


def _request_identifier(value: object, label: str) -> str:
    if not hasattr(value, "request_id"):
        raise RuntimeError(f"PD connector {label} lacks request_id")
    request_id = getattr(value, "request_id")
    if not isinstance(request_id, str) or not request_id:
        raise TypeError(f"PD connector {label}.request_id must be non-empty text")
    return request_id


def _block_count(value: object, attribute: str, label: str) -> int:
    if not hasattr(value, attribute):
        raise RuntimeError(f"PD connector {label} lacks {attribute}")
    blocks = getattr(value, attribute)
    try:
        count = len(blocks)
    except TypeError as error:
        raise TypeError(f"PD connector {label}.{attribute} is not sized") from error
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TypeError(f"PD connector {label}.{attribute} length is invalid")
    return count


def _tensor_byte_count(tensor: object) -> int | None:
    if tensor is None:
        return None
    if not hasattr(tensor, "element_size") or not hasattr(tensor, "numel"):
        raise TypeError("P2P connector transfer payload is not a tensor")
    element_size = tensor.element_size()
    element_count = tensor.numel()
    if (
        isinstance(element_size, bool)
        or not isinstance(element_size, int)
        or element_size < 1
        or isinstance(element_count, bool)
        or not isinstance(element_count, int)
        or element_count < 0
    ):
        raise TypeError("P2P connector tensor byte size is invalid")
    return element_size * element_count


def _install_pd_handoff_wrappers(recorder: _PDHandoffRecorder) -> None:
    if recorder.connector == "P2pNcclConnector":
        _install_p2p_handoff_wrappers(recorder)
        return
    if recorder.connector == "P2pMcclConnector":
        _install_p2p_mccl_handoff_wrappers(recorder)
        return
    if recorder.connector == "P2pHcclConnector":
        _install_p2p_hccl_handoff_wrappers(recorder)
        return
    if recorder.connector == "MooncakeConnector":
        _install_mooncake_018_handoff_wrappers(recorder)
        return
    raise AssertionError("PD handoff connector was validated")


def _install_p2p_handoff_wrappers(recorder: _PDHandoffRecorder) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
        P2pNcclConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
        P2pNcclEngine,
    )

    _install_p2p_handoff_wrappers_for(
        recorder, P2pNcclConnector, P2pNcclEngine)


def _install_p2p_capture_bootstrap_for(
    connector_class: type,
    model_runner_class: type,
) -> None:
    """Keep P2P KV sends out of vLLM's metadata-free Graph dummy runs."""

    if hasattr(connector_class, _P2P_CAPTURE_BOOTSTRAP_MARKER):
        return
    original_capture_model = model_runner_class.capture_model
    original_save_kv_layer = connector_class.save_kv_layer
    original_build_connector_meta = connector_class.build_connector_meta
    original_request_finished = connector_class.request_finished
    original_start_load_kv = connector_class.start_load_kv
    completed_prefill_attribute = "_midlayer_completed_prefill_request_ids"
    runtime_recorder_attribute = "_midlayer_p2p_runtime_handoff_recorder"

    def capture_model(model_runner: object, *args: object, **kwargs: object) -> object:
        if hasattr(_P2P_CAPTURE_BOOTSTRAP_STATE, "active"):
            raise RuntimeError("nested P2P CUDA Graph bootstrap is unsupported")
        _P2P_CAPTURE_BOOTSTRAP_STATE.active = True
        try:
            return original_capture_model(model_runner, *args, **kwargs)
        finally:
            del _P2P_CAPTURE_BOOTSTRAP_STATE.active

    def save_kv_layer(
        connector: object, *args: object, **kwargs: object
    ) -> object:
        if hasattr(_P2P_CAPTURE_BOOTSTRAP_STATE, "active"):
            if not hasattr(connector, "_connector_metadata"):
                raise RuntimeError("P2P connector lacks metadata state")
            if connector._connector_metadata is None:
                return None
        return original_save_kv_layer(connector, *args, **kwargs)

    def build_connector_meta(
        connector: object, scheduler_output: object
    ) -> object:
        if not hasattr(connector, "is_producer"):
            raise RuntimeError("P2P connector lacks is_producer")
        if connector.is_producer is not True:
            return original_build_connector_meta(connector, scheduler_output)
        if not hasattr(connector, completed_prefill_attribute):
            setattr(connector, completed_prefill_attribute, set())
        completed = getattr(connector, completed_prefill_attribute)
        if not isinstance(completed, set):
            raise TypeError("P2P completed-Prefill state must be a set")
        for request in scheduler_output.scheduled_new_reqs:
            if request.prompt_token_ids is None:
                raise RuntimeError("P2P producer request lacks prompt tokens")
            scheduled = scheduler_output.num_scheduled_tokens[request.req_id]
            if scheduled + request.num_computed_tokens >= len(
                request.prompt_token_ids
            ):
                completed.add(request.req_id)
        cached = scheduler_output.scheduled_cached_reqs
        keep = tuple(
            index
            for index, request_id in enumerate(cached.req_ids)
            if request_id not in completed or request_id in connector.chunked_prefill
        )
        if len(keep) == len(cached.req_ids):
            return original_build_connector_meta(connector, scheduler_output)
        from copy import copy

        filtered_output = copy(scheduler_output)
        filtered_cached = copy(cached)
        for field in (
            "req_ids",
            "resumed_from_preemption",
            "new_token_ids",
            "new_block_ids",
            "num_computed_tokens",
        ):
            values = getattr(cached, field)
            setattr(filtered_cached, field, [values[index] for index in keep])
        filtered_output.scheduled_cached_reqs = filtered_cached
        return original_build_connector_meta(connector, filtered_output)

    def request_finished(
        connector: object, request: object, *args: object, **kwargs: object
    ) -> object:
        if hasattr(connector, completed_prefill_attribute):
            completed = getattr(connector, completed_prefill_attribute)
            if not isinstance(completed, set):
                raise TypeError("P2P completed-Prefill state must be a set")
            if not hasattr(request, "request_id"):
                raise RuntimeError("P2P finished request lacks request_id")
            completed.discard(request.request_id)
        return original_request_finished(connector, request, *args, **kwargs)

    def traced_start_load_kv(
        connector: object, *args: object, **kwargs: object
    ) -> object:
        """Record actual P2P GET receives in the worker process.

        The Scheduler process can observe only connector metadata.  The NCCL
        receive occurs inside the worker-local engine, so the boundaries below
        surround its real ``recv_tensor`` call and stream synchronization.
        """

        if not hasattr(connector, "is_producer"):
            raise RuntimeError("P2P connector lacks is_producer")
        if connector.is_producer:
            return original_start_load_kv(connector, *args, **kwargs)
        if hasattr(connector, "p2p_nccl_engine"):
            engine = connector.p2p_nccl_engine
        elif hasattr(connector, "p2p_mccl_engine"):
            engine = connector.p2p_mccl_engine
        elif hasattr(connector, "p2p_hccl_engine"):
            engine = connector.p2p_hccl_engine
        else:
            raise RuntimeError("P2P consumer connector lacks its transport engine")
        if engine is None or not hasattr(engine, "recv_tensor"):
            raise RuntimeError("P2P consumer transport engine lacks recv_tensor")
        if not hasattr(connector, runtime_recorder_attribute):
            identity = _pd_run_identity_from_environment()
            if identity is None:
                raise RuntimeError(
                    "P2P worker handoff observation requires complete PD identity"
                )
            setattr(
                connector,
                runtime_recorder_attribute,
                _PDHandoffRecorder(identity, write_through=True),
            )
        recorder = getattr(connector, runtime_recorder_attribute)
        if not isinstance(recorder, _PDHandoffRecorder):
            raise TypeError("P2P worker handoff recorder has an invalid type")
        original_recv_tensor = engine.recv_tensor
        received: dict[str, tuple[int, int | None, str]] = {}
        load_completed = False

        def traced_recv_tensor(
            tensor_id: object, remote_address: object = None
        ) -> object:
            if not isinstance(tensor_id, str) or not tensor_id:
                raise TypeError("P2P receive tensor ID must be non-empty text")
            request_id = tensor_id.split("#", 1)[0]
            if request_id not in received:
                received[request_id] = (0, 0, tensor_id)
                for event in ("transfer_start", "consumer_load_start"):
                    recorder.record(
                        event,
                        request_id=request_id,
                        kv_tokens=None,
                        kv_blocks=None,
                        byte_count=None,
                        extra={"first_tensor_id": tensor_id},
                    )
            try:
                result = original_recv_tensor(tensor_id, remote_address)
            except BaseException:
                recorder.record(
                    "consumer_load_failed",
                    request_id=request_id,
                    kv_tokens=None,
                    kv_blocks=None,
                    byte_count=None,
                    extra={"tensor_id": tensor_id},
                )
                raise
            byte_count = _tensor_byte_count(result)
            tensor_count, total_byte_count, first_tensor_id = received[request_id]
            received[request_id] = (
                tensor_count + 1,
                None if total_byte_count is None or byte_count is None else (
                    total_byte_count + byte_count
                ),
                first_tensor_id,
            )
            return result

        engine.recv_tensor = traced_recv_tensor
        try:
            result = original_start_load_kv(connector, *args, **kwargs)
            load_completed = True
            return result
        finally:
            engine.recv_tensor = original_recv_tensor
            if load_completed:
                for request_id, (tensor_count, byte_count, first_tensor_id) in received.items():
                    for event in ("transfer_end", "consumer_load_end"):
                        recorder.record(
                            event,
                            request_id=request_id,
                            kv_tokens=None,
                            kv_blocks=None,
                            byte_count=byte_count,
                            extra={
                                "first_tensor_id": first_tensor_id,
                                "tensor_count": tensor_count,
                            },
                        )

    model_runner_class.capture_model = capture_model
    connector_class.save_kv_layer = save_kv_layer
    connector_class.build_connector_meta = build_connector_meta
    connector_class.request_finished = request_finished
    connector_class.start_load_kv = traced_start_load_kv
    setattr(connector_class, _P2P_CAPTURE_BOOTSTRAP_MARKER, True)


def _install_p2p_capture_bootstrap_from_environment() -> None:
    if _PD_HANDOFF_CONNECTOR_ENV not in os.environ:
        return
    connector = os.environ[_PD_HANDOFF_CONNECTOR_ENV]
    if connector not in ("P2pNcclConnector", "P2pMcclConnector", "P2pHcclConnector"):
        return
    if connector == "P2pNcclConnector":
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import P2pNcclConnector
        connector_class = P2pNcclConnector
    elif connector == "P2pMcclConnector":
        from dataset_pipeline.serving_dataset.metax_p2p_mccl import P2pMcclConnector
        connector_class = P2pMcclConnector
    elif connector == "P2pHcclConnector":
        from dataset_pipeline.serving_dataset.ascend_p2p_hccl import P2pHcclConnector
        connector_class = P2pHcclConnector
    else:
        raise ValueError(f"unsupported PD handoff connector: {connector!r}")
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    _install_p2p_capture_bootstrap_for(connector_class, GPUModelRunner)


def _install_p2p_mccl_handoff_wrappers(
    recorder: _PDHandoffRecorder,
) -> None:
    from dataset_pipeline.serving_dataset.metax_p2p_mccl import (
        P2pMcclConnector,
    )
    from dataset_pipeline.serving_dataset.metax_p2p_mccl.p2p_mccl_engine import (
        P2pMcclEngine,
    )

    _install_p2p_handoff_wrappers_for(
        recorder, P2pMcclConnector, P2pMcclEngine)


def _install_p2p_hccl_handoff_wrappers(
    recorder: _PDHandoffRecorder,
) -> None:
    from dataset_pipeline.serving_dataset.ascend_p2p_hccl import P2pHcclConnector
    from dataset_pipeline.serving_dataset.ascend_p2p_hccl.p2p_hccl_engine import P2pHcclEngine

    _install_p2p_handoff_wrappers_for(
        recorder, P2pHcclConnector, P2pHcclEngine)


def _install_p2p_handoff_wrappers_for(
    recorder: _PDHandoffRecorder,
    connector_class: type,
    engine_class: type,
) -> None:
    original_build_connector_meta = connector_class.build_connector_meta

    def traced_build_connector_meta(
        connector: object, *args: object, **kwargs: object
    ) -> object:
        metadata = original_build_connector_meta(connector, *args, **kwargs)
        if not hasattr(metadata, "requests"):
            raise RuntimeError("P2P connector metadata lacks requests")
        requests = metadata.requests
        if not isinstance(requests, list):
            raise TypeError("P2P connector metadata requests must be a list")
        if not hasattr(connector, "is_producer"):
            raise RuntimeError("P2P connector lacks is_producer")
        is_producer = connector.is_producer
        if not isinstance(is_producer, bool):
            raise TypeError("P2P connector is_producer must be boolean")
        event = "producer_ready" if is_producer else "consumer_load_scheduled"
        for request in requests:
            recorder.record(
                event,
                request_id=_request_identifier(request, "metadata request"),
                kv_tokens=_integer_attribute(
                    request, "num_tokens", "metadata request"
                ),
                kv_blocks=_block_count(
                    request, "block_ids", "metadata request"
                ),
                byte_count=None,
            )
        return metadata

    connector_class.build_connector_meta = traced_build_connector_meta
    original_send_sync = engine_class.send_sync

    def traced_send_sync(engine: object, item: object) -> object:
        if not hasattr(item, "tensor_id") or not hasattr(item, "tensor"):
            raise RuntimeError("P2P send item lacks tensor identity or payload")
        tensor_id = item.tensor_id
        if not isinstance(tensor_id, str) or not tensor_id:
            raise TypeError("P2P send tensor ID must be non-empty text")
        byte_count = _tensor_byte_count(item.tensor)
        request_id = tensor_id.split("#", 1)[0]
        recorder.record(
            "transfer_start",
            request_id=request_id,
            kv_tokens=None,
            kv_blocks=None,
            byte_count=byte_count,
            extra={"tensor_id": tensor_id},
        )
        try:
            result = original_send_sync(engine, item)
        except BaseException:
            recorder.record(
                "transfer_failed",
                request_id=request_id,
                kv_tokens=None,
                kv_blocks=None,
                byte_count=byte_count,
                extra={"tensor_id": tensor_id},
            )
            raise
        if result is not True:
            recorder.record(
                "transfer_failed",
                request_id=request_id,
                kv_tokens=None,
                kv_blocks=None,
                byte_count=byte_count,
                extra={"tensor_id": tensor_id},
            )
            return result
        recorder.record(
            "transfer_end",
            request_id=request_id,
            kv_tokens=None,
            kv_blocks=None,
            byte_count=byte_count,
            extra={"tensor_id": tensor_id},
        )
        return result

    engine_class.send_sync = traced_send_sync
    if hasattr(connector_class, _P2P_CAPTURE_BOOTSTRAP_MARKER):
        # The worker-local capture bootstrap already wraps the complete
        # ``start_load_kv`` operation.  It records one interval per request
        # around the real NCCL receives, including the aggregate byte count.
        # Adding this per-tensor wrapper beneath it writes two synchronous
        # JSON records per layer and materially changes the measured PD
        # workload without adding another independent boundary.
        return

    original_recv_tensor = engine_class.recv_tensor

    def traced_recv_tensor(
        engine: object, tensor_id: object, remote_address: object = None
    ) -> object:
        if not isinstance(tensor_id, str) or not tensor_id:
            raise TypeError("P2P receive tensor ID must be non-empty text")
        request_id = tensor_id.split("#", 1)[0]
        recorder.record(
            "consumer_load_start",
            request_id=request_id,
            kv_tokens=None,
            kv_blocks=None,
            byte_count=None,
            extra={"tensor_id": tensor_id},
        )
        try:
            result = original_recv_tensor(engine, tensor_id, remote_address)
        except BaseException:
            recorder.record(
                "consumer_load_failed",
                request_id=request_id,
                kv_tokens=None,
                kv_blocks=None,
                byte_count=None,
                extra={"tensor_id": tensor_id},
            )
            raise
        recorder.record(
            "consumer_load_end",
            request_id=request_id,
            kv_tokens=None,
            kv_blocks=None,
            byte_count=_tensor_byte_count(result),
            extra={"tensor_id": tensor_id},
        )
        return result

    engine_class.recv_tensor = traced_recv_tensor




def _install_mooncake_018_handoff_wrappers(recorder: _PDHandoffRecorder) -> None:
    """Instrument the actual 910B vLLM 0.18.0 Mooncake connector surface."""

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
            MooncakeConnector,
            MooncakeConnectorWorker,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "910B PD handoff tracing requires vLLM 0.18.0 MooncakeConnector"
        ) from error

    original_request_finished = MooncakeConnector.request_finished

    def traced_request_finished(
        connector: object, request: object, block_ids: object
    ) -> object:
        result = original_request_finished(connector, request, block_ids)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Mooncake request_finished returned an invalid result")
        scheduler = _attribute(connector, "connector_scheduler", "Mooncake connector")
        if scheduler is None:
            raise RuntimeError("Mooncake connector has no Scheduler role")
        is_producer = _attribute(scheduler, "is_kv_producer", "Mooncake Scheduler")
        if not isinstance(is_producer, bool):
            raise TypeError("Mooncake Scheduler is_kv_producer must be boolean")
        if not is_producer:
            return result
        request_id = _request_identifier(request, "Mooncake producer request")
        prompt_ids = _attribute(request, "prompt_token_ids", "Mooncake producer request")
        try:
            token_count = len(prompt_ids)
            block_count = len(block_ids)
        except TypeError as error:
            raise TypeError("Mooncake producer token or block sequence is invalid") from error
        if token_count < 1 or block_count < 1:
            raise ValueError("Mooncake producer emitted an empty KV handoff")
        recorder.record(
            "producer_ready",
            request_id=request_id,
            kv_tokens=token_count,
            kv_blocks=block_count,
            byte_count=None,
        )
        return result

    MooncakeConnector.request_finished = traced_request_finished
    original_send_kv_to_decode = MooncakeConnectorWorker.send_kv_to_decode

    async def traced_send_kv_to_decode(
        worker: object, identity: object, socket: object, metadata: object
    ) -> object:
        request_blocks = _attribute(metadata, "req_blocks", "Mooncake transfer metadata")
        if not isinstance(request_blocks, Mapping) or not request_blocks:
            raise TypeError("Mooncake transfer metadata req_blocks must be non-empty")
        records: list[tuple[str, int]] = []
        for request_id, value in request_blocks.items():
            if not isinstance(request_id, str) or not request_id:
                raise TypeError("Mooncake transfer request ID must be non-empty text")
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError("Mooncake transfer request blocks are invalid")
            try:
                block_count = len(value[1])
            except TypeError as error:
                raise TypeError("Mooncake remote block IDs are not sized") from error
            if block_count < 1:
                raise ValueError("Mooncake transfer has no remote blocks")
            records.append((request_id, block_count))
            recorder.record(
                "transfer_start",
                request_id=request_id,
                kv_tokens=None,
                kv_blocks=block_count,
                byte_count=None,
            )
        try:
            result = await original_send_kv_to_decode(worker, identity, socket, metadata)
        except BaseException:
            for request_id, block_count in records:
                recorder.record(
                    "transfer_failed",
                    request_id=request_id,
                    kv_tokens=None,
                    kv_blocks=block_count,
                    byte_count=None,
                )
            raise
        for request_id, block_count in records:
            recorder.record(
                "transfer_end",
                request_id=request_id,
                kv_tokens=None,
                kv_blocks=block_count,
                byte_count=None,
            )
        return result

    MooncakeConnectorWorker.send_kv_to_decode = traced_send_kv_to_decode
    original_start_load_kv = MooncakeConnectorWorker.start_load_kv

    def traced_start_load_kv(worker: object, metadata: object) -> object:
        request_groups = _attribute(metadata, "reqs_to_recv", "Mooncake metadata")
        if not isinstance(request_groups, Mapping):
            raise TypeError("Mooncake receive metadata must be an object")
        for requests in request_groups.values():
            if not isinstance(requests, Mapping):
                raise TypeError("Mooncake receive request group must be an object")
            for request_id, pull_meta in requests.items():
                if not isinstance(request_id, str) or not request_id:
                    raise TypeError("Mooncake receive request ID must be non-empty text")
                local_blocks = _attribute(pull_meta, "local_block_ids", "Mooncake pull metadata")
                try:
                    block_count = len(local_blocks)
                except TypeError as error:
                    raise TypeError("Mooncake local block IDs are not sized") from error
                if block_count < 1:
                    raise ValueError("Mooncake receive has no local blocks")
                recorder.record(
                    "consumer_load_start",
                    request_id=request_id,
                    kv_tokens=None,
                    kv_blocks=block_count,
                    byte_count=None,
                )
        return original_start_load_kv(worker, metadata)

    MooncakeConnectorWorker.start_load_kv = traced_start_load_kv
    original_process_pulling_result = MooncakeConnectorWorker.process_pulling_result

    def traced_process_pulling_result(
        worker: object, response: object, pull_metas: object
    ) -> object:
        result = original_process_pulling_result(worker, response, pull_metas)
        ok_requests = _attribute(response, "ok_reqs", "Mooncake transfer response")
        if ok_requests is None:
            return result
        if not isinstance(ok_requests, list):
            raise TypeError("Mooncake transfer response ok_reqs must be a list")
        if not isinstance(pull_metas, Mapping):
            raise TypeError("Mooncake pull metadata must be an object")
        completed = _attribute(worker, "finished_recving_reqs", "Mooncake worker")
        if not isinstance(completed, set):
            raise TypeError("Mooncake finished receive set is invalid")
        for request_id in ok_requests:
            if not isinstance(request_id, str) or not request_id:
                raise TypeError("Mooncake completed request ID must be non-empty text")
            if request_id not in completed:
                continue
            if request_id not in pull_metas:
                raise RuntimeError("Mooncake completed request lacks pull metadata")
            pull_meta = pull_metas[request_id]
            local_blocks = _attribute(pull_meta, "local_block_ids", "Mooncake pull metadata")
            try:
                block_count = len(local_blocks)
            except TypeError as error:
                raise TypeError("Mooncake completed local block IDs are not sized") from error
            recorder.record(
                "consumer_load_end",
                request_id=request_id,
                kv_tokens=None,
                kv_blocks=block_count,
                byte_count=None,
            )
        return result

    MooncakeConnectorWorker.process_pulling_result = traced_process_pulling_result


def _run_engine_core_with_scheduler_trace(
    *args: object, **kwargs: object
) -> object:
    """Install tracing in the spawned EngineCore before its busy loop begins."""

    install_from_environment()
    target = _ENGINE_CORE_ORIGINAL_TARGET
    if target is None or not callable(target):
        raise RuntimeError("EngineCore Scheduler trace target is not installed")
    return target(*args, **kwargs)


def _install_engine_core_spawn_target() -> None:
    """Make the EngineCore spawn target independently bootstrap tracing.

    vLLM starts EngineCore through its own spawn context.  The child cannot
    rely solely on ``sitecustomize``: an upstream launcher can replace the
    multiprocessing executable after the API process has installed it.  A
    module-level target is pickleable under spawn and performs the same
    fail-closed bootstrap at the exact EngineCore boundary.
    """

    from vllm.v1.engine.core import EngineCoreProc

    global _ENGINE_CORE_ORIGINAL_TARGET
    current = EngineCoreProc.run_engine_core
    if current is _run_engine_core_with_scheduler_trace:
        if _ENGINE_CORE_ORIGINAL_TARGET is None:
            raise RuntimeError("EngineCore trace target lost its original target")
        return
    if _ENGINE_CORE_ORIGINAL_TARGET is not None:
        raise RuntimeError("EngineCore trace target was installed inconsistently")
    _ENGINE_CORE_ORIGINAL_TARGET = current
    EngineCoreProc.run_engine_core = staticmethod(
        _run_engine_core_with_scheduler_trace
    )


def _write_bootstrap_marker() -> None:
    """Record which spawned vLLM processes ran the opt-in bootstrap."""

    if _BOOTSTRAP_MARKER_ENV not in os.environ:
        return
    marker = Path(os.environ[_BOOTSTRAP_MARKER_ENV]).resolve()
    _append_jsonl(marker, {
        "process_pid": os.getpid(),
        "process_parent_pid": os.getppid(),
        "process_name": multiprocessing.current_process().name,
        "argv": list(sys.argv),
        "trace_output": os.environ[_OUTPUT_ENV],
    })


def finalize_scheduler_trace_from_environment() -> bool:
    """Seal an in-process trace after its frontend has drained.

    Async EngineCore users can observe ``has_unfinished_requests() == False``
    immediately after the final Scheduler update, without another scheduling
    call that would revisit the automatic seal condition.  A selected-step
    runner has an explicit end-of-run boundary, so it uses this hook to make
    the same completeness checks and atomically write the trace.  Spawned
    EngineCore captures continue to seal automatically in their child process.
    """

    if _OUTPUT_ENV not in os.environ:
        return False
    with _INSTALL_LOCK:
        state = _ACTIVE_TRACE_STATE
    if state is None:
        raise RuntimeError("Scheduler trace state is not installed in this process")
    if state.deferred_io:
        # An automatic seal may already be queued on the trace I/O thread.
        state.wait_for_deferred_io()
    with state.lock:
        if state.written:
            return True
        if state.pending:
            raise RuntimeError(
                "Scheduler trace cannot be finalized with pending steps: "
                f"{len(state.pending)}"
            )
        if len(state.finished_request_indexes) != state.request_count:
            raise RuntimeError(
                "Scheduler trace cannot be finalized before all requests finish: "
                f"{len(state.finished_request_indexes)} != {state.request_count}"
            )
        state.record_bootstrap_marker_event("scheduler_trace_complete")
        state.record_bootstrap_marker_event("scheduler_trace_write_start")
        try:
            state._write_complete_trace()
        except Exception as error:
            state._record_bootstrap_marker_failure(
                "scheduler_trace_write_failed", error
            )
            raise
        state.record_bootstrap_marker_event("scheduler_trace_write_complete")
    return True


def scheduler_trace_pending_step_count_from_environment() -> int:
    """Return the number of already-submitted in-process trace steps."""

    if _OUTPUT_ENV not in os.environ:
        raise RuntimeError("Scheduler trace is not enabled")
    with _INSTALL_LOCK:
        state = _ACTIVE_TRACE_STATE
    if state is None:
        raise RuntimeError("Scheduler trace state is not installed in this process")
    with state.lock:
        return len(state.pending)


def install_from_environment() -> bool:
    """Install the V1 Scheduler wrappers when trace environment is present."""

    global _ACTIVE_TRACE_STATE, _INSTALLED
    _install_p2p_capture_bootstrap_from_environment()
    trace_enabled = _OUTPUT_ENV in os.environ
    counter_only_enabled = _COUNTER_ONLY_OUTPUT_ENV in os.environ
    request_add_enabled = _REQUEST_ADD_OUTPUT_ENV in os.environ
    if request_add_enabled and not trace_enabled:
        raise RuntimeError(
            f"{_REQUEST_ADD_OUTPUT_ENV} requires {_OUTPUT_ENV}"
        )
    if not trace_enabled and not counter_only_enabled:
        pd_run_identity = _pd_run_identity_from_environment()
        if pd_run_identity is None:
            return False
        with _INSTALL_LOCK:
            if _INSTALLED:
                return True
            _install_pd_handoff_wrappers(
                _PDHandoffRecorder(pd_run_identity, write_through=True))
            _INSTALLED = True
            return True
    if trace_enabled and counter_only_enabled:
        raise RuntimeError(
            "Scheduler trace and A800 counter-only instrumentation are mutually exclusive"
        )
    if trace_enabled:
        _write_bootstrap_marker()
    if trace_enabled or counter_only_enabled:
        _configure_spawn_bootstrap()
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True
        if trace_enabled:
            if _REQUEST_COUNT_ENV not in os.environ:
                raise RuntimeError(
                    f"{_REQUEST_COUNT_ENV} is required when {_OUTPUT_ENV} is set"
                )
            request_count_text = os.environ[_REQUEST_COUNT_ENV]
            if not request_count_text.isdecimal() or int(request_count_text) < 1:
                raise ValueError(
                    f"{_REQUEST_COUNT_ENV} must be a positive integer"
                )
            request_count = int(request_count_text)
            output_path = Path(os.environ[_OUTPUT_ENV]).resolve()
        else:
            request_count = 0
            output_path = Path(os.environ[_COUNTER_ONLY_OUTPUT_ENV]).resolve()
        if output_path.exists():
            raise FileExistsError(
                f"Scheduler instrumentation output already exists: {output_path}"
            )
        request_add_output_path = (
            Path(os.environ[_REQUEST_ADD_OUTPUT_ENV]).resolve()
            if request_add_enabled
            else None
        )
        if (
            request_add_output_path is not None
            and request_add_output_path.exists()
        ):
            raise FileExistsError(
                "Scheduler request-add trace already exists: "
                f"{request_add_output_path}"
            )
        hardware_id = None
        if _HARDWARE_ID_ENV in os.environ:
            hardware_id = os.environ[_HARDWARE_ID_ENV]
        if hardware_id not in (
            None,
            "nvidia_a800",
            "nvidia_h200",
            "metax_c500",
            "ascend_910b",
        ):
            raise ValueError(
                "Scheduler trace hardware identity is unsupported: "
                f"{hardware_id!r}"
            )
        worker_dispatch_profile = _uses_worker_dispatch_observations(
            hardware_id)
        configured_cudagraph_mode = None
        if _CONFIGURED_GRAPH_MODE_ENV in os.environ:
            configured_cudagraph_mode = os.environ[_CONFIGURED_GRAPH_MODE_ENV]
        max_in_flight = None
        if configured_cudagraph_mode is not None:
            if _MAX_IN_FLIGHT_ENV not in os.environ:
                raise RuntimeError(
                    f"{_MAX_IN_FLIGHT_ENV} is required for rich Scheduler trace"
                )
            max_in_flight_text = os.environ[_MAX_IN_FLIGHT_ENV]
            if not max_in_flight_text.isdecimal() or int(max_in_flight_text) < 1:
                raise ValueError(f"{_MAX_IN_FLIGHT_ENV} must be a positive integer")
            max_in_flight = int(max_in_flight_text)
        worker_rank_count = None
        if worker_dispatch_profile:
            if trace_enabled and configured_cudagraph_mode != "FULL_AND_PIECEWISE":
                raise ValueError(
                    "A800 Scheduler trace requires FULL_AND_PIECEWISE mode"
                )
            if _WORKER_RANK_COUNT_ENV not in os.environ:
                raise RuntimeError(
                    f"{_WORKER_RANK_COUNT_ENV} is required for A800 Scheduler trace"
                )
            worker_rank_count_text = os.environ[_WORKER_RANK_COUNT_ENV]
            if (
                not worker_rank_count_text.isdecimal()
                or int(worker_rank_count_text) < 1
            ):
                raise ValueError(
                    f"{_WORKER_RANK_COUNT_ENV} must be a positive integer"
                )
            worker_rank_count = int(worker_rank_count_text)
            for rank in range(worker_rank_count):
                for path in (
                    _worker_counter_path(output_path, rank),
                ):
                    if path.exists():
                        raise FileExistsError(
                            "A800 Scheduler trace sidecar already exists: "
                            f"{path}"
                        )
                if trace_enabled:
                    observation_path = _worker_observation_path(output_path, rank)
                    if observation_path.exists():
                        raise FileExistsError(
                            "A800 Scheduler trace sidecar already exists: "
                            f"{observation_path}"
                        )
        pd_run_identity = _pd_run_identity_from_environment()
        pd_handoff_recorder = (
            None
            if pd_run_identity is None
            # KV transfer callbacks can finish after the last Scheduler update
            # has sealed its step trace.  Connector evidence must therefore be
            # append-only runtime evidence, not a buffer coupled to that trace.
            else _PDHandoffRecorder(pd_run_identity, write_through=True)
        )
        state = _TraceState(
            output_path,
            request_count,
            configured_cudagraph_mode=configured_cudagraph_mode,
            max_in_flight=max_in_flight,
            hardware_id=hardware_id,
            worker_rank_count=worker_rank_count,
            counter_only=counter_only_enabled,
            pd_run_identity=pd_run_identity,
            pd_handoff_recorder=pd_handoff_recorder,
            request_add_output_path=request_add_output_path,
        )
        _ACTIVE_TRACE_STATE = state
        request_stage_recorder = (
            _RequestStageRecorder(request_add_output_path)
            if request_add_output_path is not None
            else None
        )
        if request_stage_recorder is not None:
            _install_request_stage_wrappers(request_stage_recorder)
        if trace_enabled:
            try:
                from vllm.v1.core.sched.scheduler import Scheduler
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "Scheduler trace instrumentation requires vLLM V1 Scheduler"
                ) from error
            original_schedule = Scheduler.schedule
            original_update = Scheduler.update_from_output

            def traced_schedule(
                scheduler: object, *args: object, **kwargs: object
            ) -> object:
                iteration, started_ns = state.before_schedule()
                scheduler_output = original_schedule(scheduler, *args, **kwargs)
                state.record_bootstrap_marker_event("scheduler_schedule")
                state.after_schedule(
                    scheduler, scheduler_output, iteration, started_ns
                )
                return scheduler_output

            def traced_update(
                scheduler: object,
                scheduler_output: object,
                model_runner_output: object,
            ) -> object:
                update_start_ns = time.perf_counter_ns()
                state.record_bootstrap_marker_event("scheduler_update")
                state.executor_completion_observed_by_scheduler(
                    scheduler_output, update_start_ns
                )
                engine_outputs = original_update(
                    scheduler, scheduler_output, model_runner_output
                )
                update_end_ns = time.perf_counter_ns()
                state.after_update(
                    scheduler_output,
                    update_start_ns,
                    update_end_ns,
                    engine_outputs,
                    model_runner_output,
                )
                return engine_outputs

            if request_add_enabled:
                original_add_request = Scheduler.add_request

                def traced_add_request(
                    scheduler: object, request: object
                ) -> object:
                    started_ns = time.perf_counter_ns()
                    started_thread_cpu_ns = time.thread_time_ns()
                    result = original_add_request(scheduler, request)
                    ended_thread_cpu_ns = time.thread_time_ns()
                    ended_ns = time.perf_counter_ns()
                    if request_stage_recorder is None:
                        raise AssertionError("request stage recorder is missing")
                    request_stage_recorder.record(
                        "scheduler_add_request",
                        request,
                        started_ns,
                        ended_ns,
                        started_thread_cpu_ns,
                        ended_thread_cpu_ns,
                    )
                    state.after_add_request(request, ended_ns)
                    return result

                Scheduler.add_request = traced_add_request
            Scheduler.schedule = traced_schedule
            Scheduler.update_from_output = traced_update
            _install_engine_core_spawn_target()
            if configured_cudagraph_mode is not None:
                _install_executor_wrappers(state)
        if worker_dispatch_profile:
            _install_a800_worker_wrappers(state)
            _install_a800_counter_signal_handler(state)
        if hardware_id == "metax_c500":
            _install_c500_worker_wrappers()
        if hardware_id == "ascend_910b":
            _install_ascend_worker_wrappers()
            _install_ascend_step_device_window_wrappers()
        if pd_handoff_recorder is not None:
            _install_pd_handoff_wrappers(pd_handoff_recorder)
        _INSTALLED = True
        return True


def _configure_spawn_bootstrap() -> None:
    """Ensure spawned EngineCore processes import the trace bootstrap."""

    required = (
        _SPAWN_WRAPPER_ENV,
        _SPAWN_PYTHON_ENV,
        _SPAWN_PYTHONPATH_ENV,
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Scheduler trace spawn bootstrap is missing environment fields: "
            + ", ".join(missing)
        )
    wrapper = Path(os.environ[_SPAWN_WRAPPER_ENV]).resolve()
    python = Path(os.environ[_SPAWN_PYTHON_ENV]).resolve()
    bootstrap_pythonpath = os.environ[_SPAWN_PYTHONPATH_ENV]
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise RuntimeError("Scheduler trace spawn wrapper is not executable")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("Scheduler trace spawn Python is not executable")
    if not bootstrap_pythonpath:
        raise ValueError("Scheduler trace spawn Python path is empty")
    multiprocessing.set_executable(str(wrapper))


def _install_executor_wrappers(state: _TraceState) -> None:
    from concurrent.futures import Future
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    if state.worker_dispatch_profile:
        # vLLM 0.11.0 defines UniProcExecutor next to the Executor base.
        from vllm.v1.executor.abstract import UniProcExecutor
    elif state.hardware_id in ("metax_c500", "ascend_910b", "nvidia_h200"):
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
    else:
        raise ValueError(
            f"unsupported executor-wrapper hardware: {state.hardware_id!r}"
        )

    for executor_class in (UniProcExecutor, MultiprocExecutor):
        original_execute_model = executor_class.execute_model

        def traced_execute_model(
            executor: object,
            scheduler_output: object,
            non_block: bool = False,
            *,
            _original: object = original_execute_model,
        ) -> object:
            started_ns = time.perf_counter_ns()
            state.executor_started(executor, scheduler_output, started_ns)
            result = _original(executor, scheduler_output, non_block=non_block)
            if isinstance(result, Future):
                result.add_done_callback(
                    lambda _future, output=scheduler_output: state.executor_finished(
                        output, time.perf_counter_ns()
                    )
                )
            else:
                state.executor_finished(scheduler_output, time.perf_counter_ns())
            return result

        executor_class.execute_model = traced_execute_model


def _install_a800_worker_wrappers(state: _TraceState) -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    original_worker_execute_model = Worker.execute_model
    original_runner_execute_model = GPUModelRunner.execute_model

    def traced_worker_execute_model(
        worker: object, scheduler_output: object
    ) -> object:
        state.worker_execute_started(worker.rank, scheduler_output)
        nvtx_pushed = False
        if _A800_NSYS_STEP_NVTX_ENV in os.environ:
            if os.environ[_A800_NSYS_STEP_NVTX_ENV] != "1":
                raise ValueError(
                    f"{_A800_NSYS_STEP_NVTX_ENV} must be exactly '1' when set"
                )
            trace_step_id = (
                scheduler_output._midlayer_scheduler_trace_step_id
                if hasattr(
                    scheduler_output,
                    "_midlayer_scheduler_trace_step_id")
                else None)
            # Warmup Scheduler outputs intentionally have no formal step ID.
            # Only formal steps enter the capture range, keeping compilation and
            # graph-capture work out of the diagnostic Nsight timeline.
            if trace_step_id is not None:
                if not isinstance(trace_step_id, str) or not trace_step_id:
                    raise TypeError("A800 NVTX Scheduler step ID is invalid")
                import torch

                torch.cuda.nvtx.range_push(f"midlayer-step:{trace_step_id}")
                nvtx_pushed = True
        try:
            result = original_worker_execute_model(worker, scheduler_output)
        except BaseException:
            state.worker_execute_failed()
            raise
        finally:
            if nvtx_pushed:
                import torch

                torch.cuda.nvtx.range_pop()
        state.worker_execute_finished()
        return result

    def traced_runner_execute_model(
        runner: object,
        scheduler_output: object,
        intermediate_tensors: object | None = None,
    ) -> object:
        if state.counter_only or not hasattr(state.worker_execution, "current"):
            return original_runner_execute_model(
                runner, scheduler_output, intermediate_tensors
            )
        dispatcher = runner.cudagraph_dispatcher
        original_dispatch = dispatcher.dispatch

        def traced_dispatch(dispatch_input: object) -> tuple[object, object]:
            runtime_mode, selected_graph_key = original_dispatch(dispatch_input)
            state.worker_dispatch(
                dispatcher,
                dispatch_input,
                runtime_mode,
                selected_graph_key,
            )
            return runtime_mode, selected_graph_key

        dispatcher.dispatch = traced_dispatch
        try:
            return original_runner_execute_model(
                runner, scheduler_output, intermediate_tensors
            )
        finally:
            dispatcher.dispatch = original_dispatch

    Worker.execute_model = traced_worker_execute_model
    GPUModelRunner.execute_model = traced_runner_execute_model



def _install_ascend_step_device_window_wrappers() -> None:
    """Record one device timing window per formal Scheduler step.

    The Scheduler trace carries a host enqueue envelope, not the device window,
    and msprof cannot supply the latter without destroying what it measures: an
    msprof-attached run of the same case spends 478.97 ms per Scheduler interval
    against 20.5 ms without it, about 460 ms of added host time per step.  A
    ``torch.npu.Event`` pair costs no resolvable host time instead -- measured
    against an un-instrumented submission of the same chain at 579 ops per step,
    the difference stays inside the run-to-run spread with +17.79 us per step as
    the largest positive reading.

    Reaching that cost depends on one detail: nothing on the submission path may
    synchronise.  Both events are recorded and kept, and every ``elapsed_time``
    read is deferred to process exit.  Synchronising on the closing event each
    step instead costs +563 us per step at the same op count.

    Two windows are recorded per step because vLLM's async path splits the work
    across two executor calls: ``execute_model`` runs the forward and
    ``sample_tokens`` runs sampling and the output copy.  A step's output only
    becomes readable after the second, so bracketing the forward alone leaves the
    sampling window outside and makes the forward end look 18-29 ms early.
    """

    if _ASCEND_STEP_DEVICE_WINDOW_ENV not in os.environ:
        return
    if os.environ[_ASCEND_STEP_DEVICE_WINDOW_ENV] != "1":
        raise ValueError(
            f"{_ASCEND_STEP_DEVICE_WINDOW_ENV} must be exactly '1' when set"
        )
    for conflicting_name in (
        _ASCEND_MSPROF_STEP_TIMESTAMP_ENV,
        _ASCEND_MSPROFTX_STEP_MARK_ENV,
    ):
        if conflicting_name in os.environ:
            raise ValueError(
                "910B profiler step boundary mechanisms are mutually "
                f"exclusive: {_ASCEND_STEP_DEVICE_WINDOW_ENV} with "
                f"{conflicting_name}"
            )
    if _OUTPUT_ENV not in os.environ:
        raise RuntimeError(
            f"{_ASCEND_STEP_DEVICE_WINDOW_ENV} requires {_OUTPUT_ENV}"
        )
    # Every rank runs its own copy of this wrapper in its own process, so the
    # file carries the rank the way the worker observation files already do
    # (``_worker_observation_path``).  Without it a TP>1 capture has two
    # processes writing one path.
    output_root = os.environ[_OUTPUT_ENV]
    recorded_rank: dict[str, int] = {}
    import torch

    from vllm_ascend.worker.worker import NPUWorker

    original_worker_execute_model = NPUWorker.execute_model
    original_worker_sample_tokens = NPUWorker.sample_tokens
    recorded: list[tuple[str, str, object, object]] = []
    recorded_lock = threading.Lock()
    # One reference event puts every window on a single device timeline, so a
    # step's start and end can be compared against another step's -- an
    # elapsed_time pair alone only gives one window's own length.
    reference_event = torch.npu.Event(enable_timing=True)
    reference_recorded = {"done": False}
    # sample_tokens takes the grammar output, not the Scheduler output, so the
    # step it belongs to is the one the preceding execute_model announced on
    # this worker.
    current_step_id: dict[int, str] = {}

    def traced_worker_execute_model(
        worker: object, scheduler_output: object, *args: object, **kwargs: object
    ) -> object:
        # Warmup and capture Scheduler outputs carry no formal step ID; leaving
        # them untimed keeps compilation and graph capture out of the window.
        if not hasattr(scheduler_output, "_midlayer_scheduler_trace_step_id"):
            with recorded_lock:
                if id(worker) in current_step_id:
                    del current_step_id[id(worker)]
            return original_worker_execute_model(
                worker, scheduler_output, *args, **kwargs)
        step_id = scheduler_output._midlayer_scheduler_trace_step_id
        if not isinstance(step_id, str) or not step_id:
            raise TypeError("910B step device window step ID is invalid")
        if not hasattr(worker, "rank"):
            raise AttributeError("910B step device window worker has no rank")
        rank = worker.rank
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise TypeError("910B step device window worker rank is invalid")
        with recorded_lock:
            current_step_id[id(worker)] = step_id
            if "rank" in recorded_rank and recorded_rank["rank"] != rank:
                raise RuntimeError(
                    "910B step device window saw two ranks in one process")
            recorded_rank["rank"] = rank
        opening = torch.npu.Event(enable_timing=True)
        closing = torch.npu.Event(enable_timing=True)
        with recorded_lock:
            if not reference_recorded["done"]:
                reference_event.record()
                reference_recorded["done"] = True
        opening.record()
        try:
            return original_worker_execute_model(
                worker, scheduler_output, *args, **kwargs)
        finally:
            closing.record()
            with recorded_lock:
                recorded.append((step_id, "forward", opening, closing))

    def traced_worker_sample_tokens(
        worker: object, *args: object, **kwargs: object
    ) -> object:
        with recorded_lock:
            step_id = (
                current_step_id[id(worker)]
                if id(worker) in current_step_id else None)
        if step_id is None:
            return original_worker_sample_tokens(worker, *args, **kwargs)
        opening = torch.npu.Event(enable_timing=True)
        closing = torch.npu.Event(enable_timing=True)
        opening.record()
        try:
            return original_worker_sample_tokens(worker, *args, **kwargs)
        finally:
            closing.record()
            with recorded_lock:
                recorded.append((step_id, "sample", opening, closing))

    # A FULL ACL-graph step synchronises the stream and only then replays
    # (vllm_ascend/compilation/acl_graph.py:207-209), so an event recorded just
    # before the replay splits the forward window into the pre-replay part
    # (input-preparation kernels plus the synchronisation hand-off) and the
    # replayed graph itself.
    original_graph_replay = torch.npu.NPUGraph.replay
    replay_points: list[tuple[str, int, object]] = []
    replay_counts: dict[str, int] = {}
    replay_step_by_thread: dict[int, str] = {}

    def traced_graph_replay(graph: object, *args: object, **kwargs: object) -> object:
        thread_id = threading.get_ident()
        with recorded_lock:
            step_id = (
                replay_step_by_thread[thread_id]
                if thread_id in replay_step_by_thread else None)
        if step_id is not None:
            point = torch.npu.Event(enable_timing=True)
            point.record()
            with recorded_lock:
                index = replay_counts[step_id] if step_id in replay_counts else 0
                replay_counts[step_id] = index + 1
                replay_points.append((step_id, index, point))
        return original_graph_replay(graph, *args, **kwargs)

    def forward_with_replay_scope(
        worker: object, scheduler_output: object, *args: object, **kwargs: object
    ) -> object:
        thread_id = threading.get_ident()
        formal = hasattr(scheduler_output, "_midlayer_scheduler_trace_step_id")
        if formal:
            with recorded_lock:
                replay_step_by_thread[thread_id] = (
                    scheduler_output._midlayer_scheduler_trace_step_id)
        try:
            return traced_worker_execute_model(
                worker, scheduler_output, *args, **kwargs)
        finally:
            if formal:
                with recorded_lock:
                    del replay_step_by_thread[thread_id]

    torch.npu.NPUGraph.replay = traced_graph_replay
    NPUWorker.execute_model = forward_with_replay_scope
    NPUWorker.sample_tokens = traced_worker_sample_tokens

    def write_step_device_windows() -> None:
        with recorded_lock:
            pending = list(recorded)
            have_reference = reference_recorded["done"]
            pending_replays = list(replay_points)
        if not pending:
            return
        if not have_reference:
            raise RuntimeError("910B step device windows have no reference event")
        if "rank" not in recorded_rank:
            raise RuntimeError("910B step device windows have no worker rank")
        output_path = Path(
            f"{output_root}.device_window.rank-{recorded_rank['rank']}.jsonl")
        if output_path.exists():
            raise FileExistsError(
                f"refusing to replace step device windows: {output_path}"
            )
        torch.npu.synchronize()
        with output_path.open("w", encoding="utf-8") as handle:
            for step_id, phase, opening, closing in pending:
                window_ms = float(opening.elapsed_time(closing))
                if window_ms <= 0:
                    raise RuntimeError(
                        "910B step device window is not positive: "
                        f"{step_id}/{phase}"
                    )
                handle.write(json.dumps({
                    "schema_version": 3,
                    "step_id": step_id,
                    "phase": phase,
                    "device_window_ms": window_ms,
                    "device_start_ms": float(
                        reference_event.elapsed_time(opening)),
                    "device_end_ms": float(
                        reference_event.elapsed_time(closing)),
                    "rank": recorded_rank["rank"],
                }, sort_keys=True) + "\n")
            for step_id, index, point in pending_replays:
                handle.write(json.dumps({
                    "schema_version": 3,
                    "step_id": step_id,
                    "phase": f"replay_start_{index}",
                    "device_start_ms": float(
                        reference_event.elapsed_time(point)),
                    "rank": recorded_rank["rank"],
                }, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    atexit.register(write_step_device_windows)


def _install_ascend_worker_wrappers() -> None:
    """Emit formal Scheduler-step boundaries onto the 910B device stream.

    The current path enqueues one MSTX mark after each successful formal step;
    adjacent end marks delimit the intervening device work.  The legacy
    ``aclprofGetStepTimestamp`` path remains independently selectable.  The
    sidecar binds device marker order to formal Scheduler trace IDs.
    """

    enabled = {
        name
        for name in (
            _ASCEND_MSPROF_STEP_TIMESTAMP_ENV,
            _ASCEND_MSPROFTX_STEP_MARK_ENV,
        )
        if name in os.environ
    }
    if not enabled:
        return
    if len(enabled) != 1:
        raise ValueError(
            "910B profiler step boundary mechanisms are mutually exclusive"
        )
    enabled_name = next(iter(enabled))
    if os.environ[enabled_name] != "1":
        raise ValueError(
            f"{enabled_name} must be exactly '1' when set"
        )
    if _OUTPUT_ENV not in os.environ:
        raise RuntimeError(
            f"{enabled_name} requires {_OUTPUT_ENV}"
        )
    # Ascend's V1 executor calls NPUWorker directly; the generic GPU Worker
    # does not own this execution path.
    from vllm_ascend.worker.worker import NPUWorker

    framework_timing_context = threading.local()
    framework_timing_step_id = (
        os.environ[_ASCEND_FRAMEWORK_TIMING_STEP_ENV]
        if _ASCEND_FRAMEWORK_TIMING_STEP_ENV in os.environ
        else None
    )
    if framework_timing_step_id == "":
        raise ValueError(
            f"{_ASCEND_FRAMEWORK_TIMING_STEP_ENV} must be non-empty when set")
    # Timing every Torch dispatch changes the workload substantially.  A
    # framework-timing capture therefore defaults to the low-overhead source
    # wrappers above; full dispatch timing is an explicit diagnostic mode
    # only.  This keeps a missing environment setting from silently
    # contaminating the timing used to partition the outer model-runner path.
    include_framework_torch_dispatch = False
    if _ASCEND_FRAMEWORK_TIMING_TORCH_DISPATCH_ENV in os.environ:
        include_dispatch_value = os.environ[
            _ASCEND_FRAMEWORK_TIMING_TORCH_DISPATCH_ENV]
        if include_dispatch_value not in ("0", "1"):
            raise ValueError(
                f"{_ASCEND_FRAMEWORK_TIMING_TORCH_DISPATCH_ENV} must be "
                "exactly '0' or '1' when set")
        include_framework_torch_dispatch = include_dispatch_value == "1"
    count_framework_torch_dispatch = False
    if _ASCEND_FRAMEWORK_TIMING_TORCH_DISPATCH_COUNT_ENV in os.environ:
        count_dispatch_value = os.environ[
            _ASCEND_FRAMEWORK_TIMING_TORCH_DISPATCH_COUNT_ENV]
        if count_dispatch_value not in ("0", "1"):
            raise ValueError(
                f"{_ASCEND_FRAMEWORK_TIMING_TORCH_DISPATCH_COUNT_ENV} must "
                "be exactly '0' or '1' when set")
        count_framework_torch_dispatch = count_dispatch_value == "1"
    selected_framework_torch_op = (
        os.environ[_ASCEND_FRAMEWORK_TIMING_SELECTED_TORCH_OP_ENV]
        if _ASCEND_FRAMEWORK_TIMING_SELECTED_TORCH_OP_ENV in os.environ
        else None
    )
    if selected_framework_torch_op is not None:
        if selected_framework_torch_op != "vllm::qkv_rmsnorm_rope.default":
            raise ValueError(
                f"{_ASCEND_FRAMEWORK_TIMING_SELECTED_TORCH_OP_ENV} must be "
                "exactly 'vllm::qkv_rmsnorm_rope.default' when set")
    if include_framework_torch_dispatch and count_framework_torch_dispatch:
        raise ValueError(
            "910B Torch dispatch timing and count-only capture are mutually "
            "exclusive")
    if selected_framework_torch_op is not None and (
        include_framework_torch_dispatch or count_framework_torch_dispatch
    ):
        raise ValueError(
            "910B selected Torch-op timing is mutually exclusive with "
            "Torch dispatch timing and count-only capture")

    def timed_framework_call(
        label: str,
        function: object,
        *args: object,
        detail_labels: tuple[str, ...] = (),
        **kwargs: object,
    ) -> object:
        state = getattr(framework_timing_context, "state", None)
        record_range = (
            hasattr(framework_timing_context, "record_ranges")
            and framework_timing_context.record_ranges
        )
        if state is None and not record_range:
            return function(*args, **kwargs)
        if state is None:
            import torch

            with torch.profiler.record_function(f"midlayer::{label}"):
                return function(*args, **kwargs)
        frame = {
            "label": label,
            "wall_start_ns": time.perf_counter_ns(),
            "thread_cpu_start_ns": time.thread_time_ns(),
            "child_wall_ns": 0,
            "child_thread_cpu_ns": 0,
        }
        state["stack"].append(frame)
        try:
            return function(*args, **kwargs)
        finally:
            thread_cpu_ns = time.thread_time_ns() - frame["thread_cpu_start_ns"]
            wall_ns = time.perf_counter_ns() - frame["wall_start_ns"]
            popped = state["stack"].pop()
            if popped is not frame:
                raise RuntimeError("910B framework timing stack is corrupted")
            exclusive_wall_ns = wall_ns - frame["child_wall_ns"]
            exclusive_thread_cpu_ns = (
                thread_cpu_ns - frame["child_thread_cpu_ns"]
            )
            if exclusive_wall_ns < 0 or exclusive_thread_cpu_ns < 0:
                raise RuntimeError("910B framework exclusive timing is negative")
            records = state["records"]
            for record_label in (label, *detail_labels):
                if record_label not in records:
                    records[record_label] = {
                        "call_count": 0,
                        "inclusive_wall_ns": 0,
                        "inclusive_thread_cpu_ns": 0,
                        "exclusive_wall_ns": 0,
                        "exclusive_thread_cpu_ns": 0,
                        "minimum_wall_ns": wall_ns,
                        "maximum_wall_ns": wall_ns,
                    }
                record = records[record_label]
                record["call_count"] += 1
                record["inclusive_wall_ns"] += wall_ns
                record["inclusive_thread_cpu_ns"] += thread_cpu_ns
                record["exclusive_wall_ns"] += exclusive_wall_ns
                record["exclusive_thread_cpu_ns"] += exclusive_thread_cpu_ns
                record["minimum_wall_ns"] = min(
                    record["minimum_wall_ns"], wall_ns)
                record["maximum_wall_ns"] = max(
                    record["maximum_wall_ns"], wall_ns)
            if state["stack"]:
                parent = state["stack"][-1]
                parent["child_wall_ns"] += wall_ns
                parent["child_thread_cpu_ns"] += thread_cpu_ns

    torch_profiler_step_id = (
        os.environ[_ASCEND_TORCH_PROFILER_STEP_ENV]
        if _ASCEND_TORCH_PROFILER_STEP_ENV in os.environ
        else None
    )
    if torch_profiler_step_id == "":
        raise ValueError(
            f"{_ASCEND_TORCH_PROFILER_STEP_ENV} must be non-empty when set")

    if (
        framework_timing_step_id is not None
        or torch_profiler_step_id is not None
    ):
        from torch_npu.dynamo.torchair.npu_fx_compiler import _CompiledFxGraph
        from vllm.compilation.piecewise_backend import PiecewiseBackend
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.v1.utils import CpuGpuBuffer
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        from vllm_ascend.device.device_op import DeviceOperator
        from vllm_ascend.worker import model_runner_v1 as ascend_model_runner
        from vllm_ascend.worker.block_table import BlockTable
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        from vllm_ascend.ops.fused_moe.fused_moe import (
            AscendFusedMoE,
            AscendMoERunner,
            AscendSharedFusedMoE,
            AscendUnquantizedFusedMoEMethod,
        )
        from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommMethod
        from vllm_ascend.ops.fused_moe.token_dispatcher import (
            TokenDispatcherWithAllGather,
        )

        def wrap_method(owner: object, name: str, label: str) -> None:
            original = getattr(owner, name)

            def timed(*args: object, **kwargs: object) -> object:
                return timed_framework_call(label, original, *args, **kwargs)

            setattr(owner, name, timed)

        def wrap_module_function(
            module: object, name: str, label: str
        ) -> None:
            """Time a source module's direct global function reference."""

            original = getattr(module, name)

            def timed(*args: object, **kwargs: object) -> object:
                return timed_framework_call(label, original, *args, **kwargs)

            setattr(module, name, timed)

        def wrap_method_inside_parent(
            owner: object, name: str, label: str, parent_label: str
        ) -> None:
            """Time a shared helper only from one source-owned boundary.

            ``torch`` and ``CpuGpuBuffer`` are process-global helpers.  Their
            selected-step cost is diagnostic only, but wrapping every use
            would collapse unrelated model and output calls into one record.
            Restrict the wrapper to the active source frame instead, so this
            boundary partitions only ``NPUModelRunner._prepare_inputs``.
            """

            original = getattr(owner, name)

            def timed(*args: object, **kwargs: object) -> object:
                state = getattr(framework_timing_context, "state", None)
                if state is None or not any(
                    frame["label"] == parent_label
                    for frame in state["stack"]
                ):
                    return original(*args, **kwargs)
                return timed_framework_call(label, original, *args, **kwargs)

            setattr(owner, name, timed)

        def wrap_context_factory(
            module: object, name: str, label: str
        ) -> None:
            """Split context setup/teardown without timing its enclosed body."""

            original = getattr(module, name)

            def timed(*args: object, **kwargs: object) -> object:
                context = original(*args, **kwargs)

                class TimedContext:
                    def __enter__(self) -> object:
                        return timed_framework_call(
                            f"{label}_enter", context.__enter__)

                    def __exit__(
                        self,
                        exception_type: object,
                        exception: object,
                        traceback: object,
                    ) -> object:
                        return timed_framework_call(
                            f"{label}_exit",
                            context.__exit__,
                            exception_type,
                            exception,
                            traceback,
                        )

                return TimedContext()

            setattr(module, name, timed)

        model_class_compute_logits_wrapped: set[type] = set()
        model_logits_component_classes_wrapped: set[tuple[type, str]] = set()

        def wrap_model_logits_method(
            component: object, name: str, label: str
        ) -> None:
            """Wrap one concrete logits component method after warmup.

            Qwen3MoE's source-level ``compute_logits`` calls its
            ``logits_processor`` with the ``lm_head``.  Measuring those two
            calls separately keeps the selected-step diagnostic from turning
            the whole output path (which includes planned Device work) into
            a fictional Host constant.
            """

            component_class = type(component)
            key = (component_class, label)
            if key in model_logits_component_classes_wrapped:
                return
            original = getattr(component_class, name)

            def timed(*args: object, **kwargs: object) -> object:
                return timed_framework_call(label, original, *args, **kwargs)

            setattr(component_class, name, timed)
            model_logits_component_classes_wrapped.add(key)

        def wrap_model_compute_logits(model: object) -> None:
            """Wrap the concrete model class once after it has been loaded.

            ``compute_logits`` is called after the compiled decoder body and
            outside ``NPUModelRunner._model_forward``.  Keeping it separate
            lets the low-overhead trace split that step-level Host work from
            the remaining model-runner bookkeeping.  The wrapper is
            diagnostic-only and is inert unless the target timing state is
            active.
            """

            # 910B's FULL-configured runtime stores the model in an
            # ``ACLGraphWrapper`` even when this target executes in NONE
            # mode.  ``compute_logits`` is delegated dynamically, so patch
            # the wrapped runnable's concrete class rather than the wrapper
            # class itself.
            if hasattr(model, "unwrap"):
                model = model.unwrap()
            model_class = type(model)
            if model_class in model_class_compute_logits_wrapped:
                return
            original = getattr(model_class, "compute_logits")

            def timed(*args: object, **kwargs: object) -> object:
                return timed_framework_call(
                    "model_runner_compute_logits", original, *args, **kwargs)

            setattr(model_class, "compute_logits", timed)
            model_class_compute_logits_wrapped.add(model_class)

            # The source contract is explicit for this target model.  Do not
            # infer another model's output structure from its class name: a
            # missing component is a diagnostic failure rather than a timing
            # fallback.
            logits_processor = getattr(model, "logits_processor")
            lm_head = getattr(model, "lm_head")
            wrap_model_logits_method(
                logits_processor,
                "forward",
                "model_compute_logits_processor",
            )
            # The Qwen3MoE source calls the quant-method ``apply`` path,
            # rather than ``lm_head.forward``.  Split it from the processor's
            # TP gather and remaining control work at the exact source
            # boundaries; none of these diagnostic durations is a parameter.
            wrap_model_logits_method(
                logits_processor,
                "_get_logits",
                "model_compute_logits_get_logits",
            )
            wrap_model_logits_method(
                logits_processor,
                "_gather_logits",
                "model_compute_logits_tp_gather",
            )
            wrap_model_logits_method(
                getattr(lm_head, "quant_method"),
                "apply",
                "model_compute_logits_lm_head",
            )

        def wrap_execute_model() -> None:
            """Install the instance-dependent logits wrapper before execution."""

            original = NPUModelRunner.execute_model

            def timed(*args: object, **kwargs: object) -> object:
                if not args:
                    raise RuntimeError("NPUModelRunner.execute_model lost self")
                # Do not alter the compiler's warmup/capture path.  The
                # concrete model method is wrapped only for the selected
                # diagnostic step, after that path has already been compiled.
                if getattr(framework_timing_context, "state", None) is not None:
                    runner = args[0]
                    model = getattr(runner, "model")
                    if model is None:
                        raise RuntimeError("NPUModelRunner has no loaded model")
                    wrap_model_compute_logits(model)
                return timed_framework_call(
                    "model_runner_execute", original, *args, **kwargs)

            NPUModelRunner.execute_model = timed

        def wrap_bound_class_call(owner: object, name: str, label: str) -> None:
            original = getattr(owner, name)

            def timed(*args: object, **kwargs: object) -> object:
                return timed_framework_call(label, original, *args, **kwargs)

            setattr(owner, name, staticmethod(timed))

        def wrap_named_torch_call(
            owner: object, name: str, label: str, operator_name: object
        ) -> None:
            original = getattr(owner, name)

            def timed(*args: object, **kwargs: object) -> object:
                detail_label = f"{label}::{operator_name(args[0])}"
                return timed_framework_call(
                    label,
                    original,
                    *args,
                    detail_labels=(detail_label,),
                    **kwargs,
                )

            setattr(owner, name, timed)

        def wrap_torch_dispatch_count(
            owner: object, name: str, dispatch_kind: str
        ) -> None:
            original = getattr(owner, name)

            def counted(*args: object, **kwargs: object) -> object:
                state = getattr(framework_timing_context, "state", None)
                if state is not None:
                    stack = state["stack"]
                    source_label = (
                        stack[-1]["label"]
                        if stack else "outside_instrumented_source"
                    )
                    counts = state["torch_dispatch_count_by_source"]
                    source_counts = (
                        counts[source_label]
                        if source_label in counts else None
                    )
                    if source_counts is None:
                        source_counts = {
                            "op_overload": 0,
                            "op_overload_packet": 0,
                        }
                        counts[source_label] = source_counts
                    source_counts[dispatch_kind] += 1
                return original(*args, **kwargs)

            setattr(owner, name, counted)

        def wrap_selected_torch_op(
            owner: object, name: str, operator_name: str
        ) -> None:
            original = getattr(owner, name)

            def timed(*args: object, **kwargs: object) -> object:
                if not args:
                    raise RuntimeError(
                        "Torch OpOverload call has no operator argument")
                operator = args[0]
                observed_name = (
                    f"{operator._schema.name}."
                    f"{operator._schema.overload_name or 'default'}"
                )
                if observed_name != operator_name:
                    return original(*args, **kwargs)
                return timed_framework_call(
                    f"torch_op_selected::{operator_name}",
                    original,
                    *args,
                    **kwargs,
                )

            setattr(owner, name, timed)

        wrap_method(
            PiecewiseBackend,
            "_find_range_for_shape",
            "piecewise_range_lookup",
        )
        # The range lookup itself is negligible.  Time the already-compiled
        # runnable boundary separately so the low-overhead selected-step
        # diagnostic can distinguish its asynchronous Host execution from
        # the source wrappers nested beneath it.  This is diagnostic-only:
        # the resulting production-workload duration is never a Forward
        # parameter.
        wrap_method(
            PiecewiseBackend,
            "__call__",
            "piecewise_compiled_execute",
        )
        # ``runtime_graph_mode=NONE`` still invokes TorchAir's runtime
        # compiled-FX graph.  Keep this boundary separate from
        # PiecewiseBackend so a selected eager-step diagnostic can distinguish
        # executor service from the Python/CANN work outside the compiled
        # backend.  It is diagnostic-only and cannot provide a Forward
        # duration parameter.
        wrap_method(
            _CompiledFxGraph,
            "__call__",
            "npu_fx_compiler_execute",
        )
        # These three source boundaries partition the large unlabelled
        # Torch/CANN dispatch population without assigning a duration to an
        # individual operator.  They are diagnostic-only: the measurements
        # distinguish model-runner input preparation from compiled model
        # execution before any generic runtime mechanism is proposed.
        wrap_execute_model()
        wrap_method(
            NPUModelRunner,
            "_update_states",
            "model_runner_state_update",
        )
        wrap_method(
            NPUModelRunner,
            "_prepare_inputs",
            "model_runner_input_prepare",
        )
        # Partition the largest unmodeled outer-runner prepare boundary using
        # direct source calls.  These wrappers are diagnostic-only: they
        # identify the next independent generic mechanism to benchmark and
        # never turn selected-serving durations into Forward parameters.
        wrap_method(
            NPUModelRunner,
            "_build_attn_state",
            "model_runner_input_prepare_attn_state",
        )
        wrap_method(
            GPUModelRunner,
            "_get_cumsum_and_arange",
            "model_runner_input_prepare_cumsum_arange",
        )
        wrap_method(
            GPUModelRunner,
            "_prepare_input_ids",
            "model_runner_input_prepare_input_ids",
        )
        wrap_method(
            BlockTable,
            "commit_block_table",
            "model_runner_input_prepare_block_table_commit",
        )
        wrap_method(
            BlockTable,
            "compute_slot_mapping",
            "model_runner_input_prepare_slot_mapping_compute",
        )
        wrap_method(
            BlockTable,
            "commit_slot_mapping",
            "model_runner_input_prepare_slot_mapping_commit",
        )
        # ``_prepare_inputs`` retains a larger unclassified exclusive slice
        # than any of its first-pass child boundaries.  Split its two generic
        # shared helper families before measuring a smaller child: the
        # framework's pinned CPU-to-NPU buffer commits and the CPU-side token
        # gather.  The parent guard excludes compiled-model and output-path
        # uses of the same process-global helpers.
        wrap_method_inside_parent(
            CpuGpuBuffer,
            "copy_to_gpu",
            "model_runner_input_prepare_pinned_h2d_commit",
            "model_runner_input_prepare",
        )
        wrap_method_inside_parent(
            ascend_model_runner.torch,
            "index_select",
            "model_runner_input_prepare_index_select",
            "model_runner_input_prepare",
        )
        wrap_method(
            NPUModelRunner,
            "_model_forward",
            "model_forward",
        )
        # Split the largest remaining source-level outer-runner boundary into
        # its three source-authored subpaths.  These are diagnostic labels
        # only: their production-workload durations cannot become Forward
        # parameters.  The target step has a mixed batch, so each is required
        # to determine whether a later independent mechanism benchmark should
        # scale with requests, scheduled tokens, or neither.
        wrap_method(
            NPUModelRunner,
            "_determine_batch_execution_and_padding",
            "model_runner_batch_descriptor",
        )
        wrap_method(
            NPUModelRunner,
            "_build_attention_metadata",
            "model_runner_attention_metadata",
        )
        wrap_method(
            NPUModelRunner,
            "_preprocess",
            "model_runner_preprocess",
        )
        # These calls live in ``execute_model`` outside the existing method
        # wrappers above.  They partition the largest remaining outer-runner
        # interval using source boundaries without enabling per-dispatch
        # instrumentation.  They are diagnostic labels only.
        wrap_module_function(
            ascend_model_runner,
            "maybe_create_ubatch_slices",
            "model_runner_ubatch_slices",
        )
        wrap_module_function(
            ascend_model_runner,
            "update_cos_sin",
            "model_runner_update_cos_sin",
        )
        wrap_context_factory(
            ascend_model_runner,
            "set_ascend_forward_context",
            "model_runner_forward_context",
        )
        if (
            framework_timing_step_id is not None
            and include_framework_torch_dispatch
        ):
            from torch._ops import OpOverload, OpOverloadPacket

            wrap_named_torch_call(
                OpOverload,
                "__call__",
                "torch_op_overload_dispatch",
                lambda operator: (
                    f"{operator._schema.name}."
                    f"{operator._schema.overload_name or 'default'}"
                ),
            )
            wrap_named_torch_call(
                OpOverloadPacket,
                "__call__",
                "torch_op_packet_dispatch",
                lambda operator: operator._qualified_op_name,
            )
        elif (
            framework_timing_step_id is not None
            and count_framework_torch_dispatch
        ):
            from torch._ops import OpOverload, OpOverloadPacket

            wrap_torch_dispatch_count(
                OpOverload,
                "__call__",
                "op_overload",
            )
            wrap_torch_dispatch_count(
                OpOverloadPacket,
                "__call__",
                "op_overload_packet",
            )
        elif (
            framework_timing_step_id is not None
            and selected_framework_torch_op is not None
        ):
            from torch._ops import OpOverload

            wrap_selected_torch_op(
                OpOverload,
                "__call__",
                selected_framework_torch_op,
            )
        wrap_method(
            AscendMoERunner,
            "forward_impl",
            "moe_runner_forward",
        )
        wrap_method(
            AscendFusedMoE,
            "forward_impl",
            "moe_fused_forward",
        )
        wrap_method(
            AscendSharedFusedMoE,
            "forward_impl",
            "moe_shared_forward",
        )
        wrap_method(
            AscendUnquantizedFusedMoEMethod,
            "apply",
            "moe_expert_apply",
        )
        wrap_method(
            MoECommMethod,
            "prepare",
            "moe_communication_prepare",
        )
        wrap_method(
            MoECommMethod,
            "fused_experts",
            "moe_fused_experts",
        )
        wrap_method(
            MoECommMethod,
            "_apply_mlp",
            "moe_mlp",
        )
        wrap_method(
            MoECommMethod,
            "finalize",
            "moe_communication_finalize",
        )
        wrap_method(
            TokenDispatcherWithAllGather,
            "token_dispatch",
            "moe_token_dispatch",
        )
        wrap_method(
            TokenDispatcherWithAllGather,
            "token_combine",
            "moe_token_combine",
        )
        wrap_method(
            AscendAttentionBackendImpl,
            "forward",
            "attention_forward",
        )
        wrap_method(
            AscendAttentionBackendImpl,
            "forward_impl",
            "attention_forward_impl",
        )
        wrap_method(
            AscendAttentionBackendImpl,
            "forward_fused_infer_attention",
            "attention_fia",
        )
        wrap_method(
            AscendAttentionBackendImpl,
            "reshape_and_cache",
            "attention_reshape_and_cache",
        )
        wrap_method(
            AscendAttentionBackendImpl,
            "_get_fia_params",
            "fia_parameter_prepare",
        )
        wrap_bound_class_call(
            DeviceOperator,
            "moe_gating_top_k",
            "moe_gating_top_k",
        )
        wrap_bound_class_call(
            DeviceOperator,
            "npu_moe_init_routing",
            "moe_init_routing",
        )
        wrap_bound_class_call(
            DeviceOperator,
            "reshape_and_cache",
            "reshape_and_cache",
        )

    use_mstx_mark = enabled_name == _ASCEND_MSPROFTX_STEP_MARK_ENV
    profiler = None
    if not use_mstx_mark:
        profiler = ctypes.CDLL("libmsprofiler.so")
        profiler.aclprofCreateStepInfo.restype = ctypes.c_void_p
        profiler.aclprofGetStepTimestamp.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        profiler.aclprofGetStepTimestamp.restype = ctypes.c_int
        profiler.aclprofDestroyStepInfo.argtypes = (ctypes.c_void_p,)
    marker_path_prefix = Path(os.environ[_OUTPUT_ENV]).resolve()
    profile_step_id = (
        os.environ[_ASCEND_THREAD_PROFILE_STEP_ENV]
        if _ASCEND_THREAD_PROFILE_STEP_ENV in os.environ
        else None
    )
    if profile_step_id == "":
        raise ValueError(
            f"{_ASCEND_THREAD_PROFILE_STEP_ENV} must be non-empty when set")
    stack_sample_step_id = (
        os.environ[_ASCEND_STACK_SAMPLE_STEP_ENV]
        if _ASCEND_STACK_SAMPLE_STEP_ENV in os.environ
        else None
    )
    if stack_sample_step_id == "":
        raise ValueError(
            f"{_ASCEND_STACK_SAMPLE_STEP_ENV} must be non-empty when set")
    native_perf_step_id = (
        os.environ[_ASCEND_NATIVE_PERF_STEP_ENV]
        if _ASCEND_NATIVE_PERF_STEP_ENV in os.environ
        else None
    )
    if native_perf_step_id == "":
        raise ValueError(
            f"{_ASCEND_NATIVE_PERF_STEP_ENV} must be non-empty when set")
    profile_targets = tuple(
        step_id for step_id in (
            profile_step_id,
            stack_sample_step_id,
            native_perf_step_id,
            framework_timing_step_id,
            torch_profiler_step_id,
        )
        if step_id is not None
    )
    if len(profile_targets) != len(set(profile_targets)):
        raise ValueError(
            "910B CPU profilers cannot target the same step"
        )
    original_worker_execute_model = NPUWorker.execute_model
    marker_ordinals: dict[int, int] = {}
    # Keep diagnostic boundary events alive until worker teardown.  Destroying
    # an event immediately after enqueue can race its device-side record and
    # would make the msprof connection ID unusable as a step boundary.
    boundary_events: dict[int, list[object]] = {}

    def traced_worker_execute_model(
        worker: object, scheduler_output: object
    ) -> object:
        nonlocal framework_timing_step_id, profile_step_id, native_perf_step_id
        trace_step_id = (
            scheduler_output._midlayer_scheduler_trace_step_id
            if hasattr(scheduler_output, "_midlayer_scheduler_trace_step_id")
            else None
        )
        if trace_step_id is None:
            return original_worker_execute_model(worker, scheduler_output)
        if not isinstance(trace_step_id, str) or not trace_step_id:
            raise TypeError("910B profiler Scheduler step ID is invalid")
        # Warmup can consume a different number of Scheduler iterations.
        # Resolve this opt-in selector from the first source-authored formal
        # step on each worker, never from a guessed warmup iteration count.
        if framework_timing_step_id == "first_formal":
            framework_timing_step_id = trace_step_id
        if profile_step_id == "first_formal":
            profile_step_id = trace_step_id
        if native_perf_step_id == "first_formal":
            native_perf_step_id = trace_step_id
        resolved_targets = tuple(
            step_id for step_id in (
                profile_step_id,
                stack_sample_step_id,
                native_perf_step_id,
                framework_timing_step_id,
                torch_profiler_step_id,
            )
            if step_id is not None
        )
        if len(resolved_targets) != len(set(resolved_targets)):
            raise ValueError("910B CPU profilers cannot target the same step")
        worker_execution_start_ns = time.perf_counter_ns()
        worker_thread_cpu_start_ns = time.thread_time_ns()
        worker_rank = worker.rank
        if (
            isinstance(worker_rank, bool)
            or not isinstance(worker_rank, int)
            or worker_rank < 0
        ):
            raise TypeError("910B profiler worker rank is invalid")
        process_pid = os.getpid()
        marker_path = Path(
            f"{marker_path_prefix}.ascend-step-marker."
            f"rank-{worker_rank}.pid-{process_pid}.jsonl"
        )
        marker_ordinal = (
            marker_ordinals[worker_rank] if worker_rank in marker_ordinals else 0
        )
        import torch_npu

        stream = torch_npu.npu.current_stream()
        stream_handle = stream.npu_stream
        if not isinstance(stream_handle, int) or stream_handle < 1:
            raise RuntimeError("910B profiler stream handle is invalid")
        if use_mstx_mark:
            framework_timing_state = None
            if trace_step_id == framework_timing_step_id:
                framework_timing_state = {
                    "records": {},
                    "stack": [],
                    "torch_dispatch_count_by_source": {},
                }
                framework_timing_context.state = framework_timing_state
            if trace_step_id == profile_step_id:
                import cProfile

                cpu_profile = cProfile.Profile(timer=time.thread_time)
                cpu_profile.enable()
                try:
                    result = original_worker_execute_model(
                        worker, scheduler_output)
                finally:
                    cpu_profile.disable()
                profile_path = Path(
                    f"{marker_path_prefix}.thread-cpu-profile."
                    f"rank-{worker_rank}.pid-{process_pid}.pstats"
                )
                if profile_path.exists():
                    raise FileExistsError(
                        f"refusing to replace CPU profile: {profile_path}")
                cpu_profile.dump_stats(profile_path)
            elif trace_step_id == stack_sample_step_id:
                sample_interval_s = 0.001
                sampled_thread_ident = threading.get_ident()
                sampled_thread_native_id = threading.get_native_id()
                sampler_ready = threading.Event()
                sampler_stop = threading.Event()
                samples: list[dict[str, object]] = []

                def sample_worker_stack() -> None:
                    sampler_ready.set()
                    while not sampler_stop.wait(sample_interval_s):
                        frame = sys._current_frames()[sampled_thread_ident]
                        stack = []
                        while frame is not None:
                            stack.append({
                                "name": frame.f_code.co_name,
                                "file": frame.f_code.co_filename,
                                "line": frame.f_lineno,
                            })
                            frame = frame.f_back
                        samples.append({
                            "wall_ns": time.time_ns(),
                            "stack": tuple(reversed(stack)),
                        })

                sampler_thread = threading.Thread(
                    target=sample_worker_stack,
                    name="midlayer-selected-step-stack-sampler",
                    daemon=True,
                )
                sampler_thread.start()
                if not sampler_ready.wait(timeout=1.0):
                    raise RuntimeError("selected-step stack sampler did not start")
                worker_call_start_wall_ns = time.time_ns()
                try:
                    result = original_worker_execute_model(
                        worker, scheduler_output)
                finally:
                    worker_call_end_wall_ns = time.time_ns()
                    sampler_stop.set()
                    sampler_thread.join(timeout=1.0)
                    if sampler_thread.is_alive():
                        raise RuntimeError(
                            "selected-step stack sampler did not stop")
                profile_path = Path(
                    f"{marker_path_prefix}.stack-sample-cpu-profile."
                    f"rank-{worker_rank}.pid-{process_pid}.json"
                )
                if profile_path.exists():
                    raise FileExistsError(
                        f"refusing to replace CPU profile: {profile_path}")
                profile_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "source": "python_current_frames_statistical_sampling",
                            "process_pid": process_pid,
                            "worker_rank": worker_rank,
                            "worker_thread_native_id": sampled_thread_native_id,
                            "trace_step_id": trace_step_id,
                            "sample_interval_s": sample_interval_s,
                            "worker_call_start_wall_ns": worker_call_start_wall_ns,
                            "worker_call_end_wall_ns": worker_call_end_wall_ns,
                            "samples": samples,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ) + "\n",
                    encoding="utf-8",
                )
            elif trace_step_id == native_perf_step_id:
                import select

                profile_path = Path(
                    f"{marker_path_prefix}.native-perf."
                    f"rank-{worker_rank}.pid-{process_pid}.data"
                )
                metadata_path = profile_path.with_suffix(".json")
                if profile_path.exists() or metadata_path.exists():
                    raise FileExistsError(
                        "refusing to replace native perf profile: "
                        f"{profile_path}")
                control_read, control_write = os.pipe()
                ack_read, ack_write = os.pipe()
                command = (
                    "perf", "record", "-q", "-F", "997", "-g",
                    "--call-graph", "dwarf", "-D", "-1",
                    f"--control=fd:{control_read},{ack_write}",
                    "-o", str(profile_path),
                    "-p", str(process_pid),
                )
                native_perf = None
                try:
                    native_perf = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        pass_fds=(control_read, ack_write),
                    )
                    # Wait for perf's explicit enable acknowledgment: a fixed
                    # sleep can finish before perf has even attached and lose
                    # the entire target step. Timings remain diagnostic-only.
                    os.write(control_write, b"enable\n")
                    if not select.select([ack_read], [], [], 30.0)[0]:
                        raise RuntimeError("native perf enable acknowledgment timed out")
                    if os.read(ack_read, 64) not in (b"ack\n", b"ack\n\x00"):
                        raise RuntimeError("native perf enable acknowledgment is invalid")
                    result = original_worker_execute_model(
                        worker, scheduler_output)
                finally:
                    try:
                        if native_perf is not None:
                            native_perf.send_signal(signal.SIGINT)
                            stdout, stderr = native_perf.communicate(timeout=10.0)
                    finally:
                        os.close(control_read)
                        os.close(control_write)
                        os.close(ack_read)
                        os.close(ack_write)
                if native_perf is not None:
                    if native_perf.returncode not in (0, -signal.SIGINT):
                        raise RuntimeError(
                            "native perf target sampling failed: "
                            f"returncode={native_perf.returncode}; "
                            f"stdout={stdout!r}; stderr={stderr!r}")
                    if not profile_path.exists() or profile_path.stat().st_size == 0:
                        raise RuntimeError(
                            "native perf target sampling produced no data")
                    metadata_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "source": (
                                    "perf_target_pid_native_stack_sampling"),
                                "trace_step_id": trace_step_id,
                                "process_pid": process_pid,
                                "worker_rank": worker_rank,
                                "worker_thread_native_id": threading.get_native_id(),
                                "attach_handshake": "perf_control_enable_ack",
                                "record_returncode": native_perf.returncode,
                                "command": command,
                                "sampling_boundary": (
                                    "NPUWorker.execute_model entry to return; "
                                    "wall duration is diagnostic-only"),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ) + "\n",
                        encoding="utf-8",
                    )
            elif trace_step_id == torch_profiler_step_id:
                import torch

                profile_path = Path(
                    f"{marker_path_prefix}.torch-npu-profile."
                    f"rank-{worker_rank}.pid-{process_pid}"
                )
                if profile_path.exists():
                    raise FileExistsError(
                        "refusing to replace torch_npu profile: "
                        f"{profile_path}")
                trace_handler = torch_npu.profiler.tensorboard_trace_handler(
                    str(profile_path), analyse_flag=True, async_mode=False)
                experimental = torch_npu.profiler._ExperimentalConfig(
                    profiler_level="Level0",
                    aic_metrics="ACL_AICORE_NONE",
                    l2_cache=False,
                    data_simplification=True,
                    record_op_args=False,
                )
                framework_timing_context.record_ranges = True
                try:
                    with torch_npu.profiler.profile(
                        activities=(
                            torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU,
                        ),
                        schedule=torch_npu.profiler.schedule(
                            wait=0, warmup=0, active=1, repeat=1),
                        on_trace_ready=trace_handler,
                        record_shapes=True,
                        profile_memory=False,
                        with_stack=True,
                        with_flops=False,
                        with_modules=False,
                        experimental_config=experimental,
                    ) as npu_profile:
                        with torch.profiler.record_function(
                            "midlayer::worker_execute_model"
                        ):
                            result = original_worker_execute_model(
                                worker, scheduler_output)
                        torch_npu.npu.synchronize()
                        npu_profile.step()
                finally:
                    del framework_timing_context.record_ranges
            else:
                try:
                    result = original_worker_execute_model(
                        worker, scheduler_output)
                finally:
                    if framework_timing_state is not None:
                        if framework_timing_state["stack"]:
                            raise RuntimeError(
                                "910B framework timing stack did not drain")
                        del framework_timing_context.state
                        timing_path = Path(
                            f"{marker_path_prefix}.framework-timing."
                            f"rank-{worker_rank}.pid-{process_pid}.json"
                        )
                        if timing_path.exists():
                            raise FileExistsError(
                                "refusing to replace framework timing: "
                                f"{timing_path}")
                        timing_path.write_text(
                            json.dumps(
                                {
                                    "schema_version": 2,
                                    "source": (
                                        "selected_function_aggregate_timing"),
                                    "process_pid": process_pid,
                                    "worker_rank": worker_rank,
                                    "trace_step_id": trace_step_id,
                                    "wall_clock": "time.perf_counter_ns",
                                    "thread_cpu_clock": "time.thread_time_ns",
                                    "nesting_semantics": (
                                        "exclusive subtracts only nested "
                                        "instrumented functions"),
                                    "records": framework_timing_state["records"],
                                    "torch_dispatch_count_by_source": (
                                        framework_timing_state[
                                            "torch_dispatch_count_by_source"]
                                    ),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ) + "\n",
                            encoding="utf-8",
                        )
            worker_thread_cpu_end_ns = time.thread_time_ns()
            worker_execution_end_ns = time.perf_counter_ns()
            mark_issue_start_ns = time.perf_counter_ns()
            torch_npu.npu.mstx.mark(
                f"midlayer-step-end:{trace_step_id}",
                stream,
                "midlayer-selected-step",
            )
            mark_issue_end_ns = time.perf_counter_ns()
            boundary_event = torch_npu.npu.Event()
            event_issue_start_wall_ns = time.time_ns()
            event_issue_start_ns = time.perf_counter_ns()
            boundary_event.record(stream)
            event_issue_end_ns = time.perf_counter_ns()
            event_issue_end_wall_ns = time.time_ns()
            if worker_rank not in boundary_events:
                boundary_events[worker_rank] = []
            boundary_events[worker_rank].append(boundary_event)
            _append_jsonl(marker_path, {
                "schema_version": 3,
                "source": "ascend_mstx_device_end_mark",
                "process_pid": process_pid,
                "worker_rank": worker_rank,
                "marker_ordinal": marker_ordinal,
                "trace_step_id": trace_step_id,
                "stream_handle": stream_handle,
                "mark_issue_start_ns": mark_issue_start_ns,
                "mark_issue_end_ns": mark_issue_end_ns,
            })
            event_path = Path(
                f"{marker_path_prefix}.ascend-event-boundary."
                f"rank-{worker_rank}.pid-{process_pid}.jsonl"
            )
            _append_jsonl(event_path, {
                "schema_version": 2,
                "source": "ascend_aclrt_event_record_boundary",
                "process_pid": process_pid,
                "worker_rank": worker_rank,
                "marker_ordinal": marker_ordinal,
                "trace_step_id": trace_step_id,
                "stream_handle": stream_handle,
                "event_handle": boundary_event.npu_event,
                "event_issue_start_ns": event_issue_start_ns,
                "event_issue_end_ns": event_issue_end_ns,
                "event_issue_start_wall_ns": event_issue_start_wall_ns,
                "event_issue_end_wall_ns": event_issue_end_wall_ns,
                "worker_execution_start_ns": worker_execution_start_ns,
                "worker_execution_end_ns": worker_execution_end_ns,
                "worker_execution_clock": "time.perf_counter_ns",
                "worker_thread_cpu_start_ns": worker_thread_cpu_start_ns,
                "worker_thread_cpu_end_ns": worker_thread_cpu_end_ns,
                "worker_thread_cpu_clock": "time.thread_time_ns",
            })
            marker_ordinals[worker_rank] = marker_ordinal + 1
            return result
        assert profiler is not None
        step_info = profiler.aclprofCreateStepInfo()
        if not step_info:
            raise RuntimeError("CANN failed to create 910B profiler step boundary")
        succeeded = False
        try:
            if profiler.aclprofGetStepTimestamp(
                step_info, 0, ctypes.c_void_p(stream_handle)
            ) != 0:
                raise RuntimeError("CANN failed to start 910B profiler step boundary")
            result = original_worker_execute_model(worker, scheduler_output)
            succeeded = True
            return result
        finally:
            try:
                if profiler.aclprofGetStepTimestamp(
                    step_info, 1, ctypes.c_void_p(stream_handle)
                ) != 0:
                    raise RuntimeError("CANN failed to end 910B profiler step boundary")
            finally:
                profiler.aclprofDestroyStepInfo(step_info)
            if succeeded:
                _append_jsonl(marker_path, {
                    "schema_version": 2,
                    "source": "ascend_aclprof_step_timestamp",
                    "process_pid": process_pid,
                    "worker_rank": worker_rank,
                    "marker_ordinal": marker_ordinal,
                    "trace_step_id": trace_step_id,
                })
                marker_ordinals[worker_rank] = marker_ordinal + 1

    NPUWorker.execute_model = traced_worker_execute_model


def _install_c500_worker_wrappers() -> None:
    """Expose the active formal step to the mcGraphLaunch interposer."""

    environment_names = (
        _C500_GRAPH_LAUNCH_CONTEXT_ENV,
        _C500_GRAPH_LAUNCH_LOG_ENV,
        _C500_KERNEL_LAUNCH_LOG_ENV,
        _C500_GRAPH_LAUNCH_INTERPOSER_ENV,
    )
    present = {name for name in environment_names if name in os.environ}
    if not present:
        return
    if present != set(environment_names):
        missing = sorted(set(environment_names) - present)
        raise RuntimeError(
            "C500 Graph launch correlation environment is incomplete: "
            + ", ".join(missing)
        )
    for name in (
        _C500_GRAPH_LAUNCH_CONTEXT_ENV,
        _C500_GRAPH_LAUNCH_LOG_ENV,
        _C500_KERNEL_LAUNCH_LOG_ENV,
    ):
        output_prefix = Path(os.environ[name])
        if not output_prefix.is_absolute():
            raise ValueError(f"{name} must be an absolute path")
        if not output_prefix.parent.is_dir():
            raise FileNotFoundError(f"{name} parent does not exist")
    interposer_path = Path(os.environ[_C500_GRAPH_LAUNCH_INTERPOSER_ENV])
    if not interposer_path.is_absolute() or not interposer_path.is_file():
        raise FileNotFoundError(
            "C500 Graph launch interposer must be an existing absolute file"
        )

    from vllm.v1.worker.gpu_worker import Worker

    original_worker_execute_model = Worker.execute_model
    interposer = ctypes.CDLL(str(interposer_path))
    set_context = interposer.midlayer_c500_set_step_context
    set_context.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64)
    set_context.restype = None
    clear_context = interposer.midlayer_c500_clear_step_context
    clear_context.argtypes = ()
    clear_context.restype = None
    context_ordinal = 0
    context_lock = threading.Lock()

    def traced_worker_execute_model(
        worker: object, scheduler_output: object
    ) -> object:
        nonlocal context_ordinal
        trace_step_id = (
            scheduler_output._midlayer_scheduler_trace_step_id
            if hasattr(scheduler_output, "_midlayer_scheduler_trace_step_id")
            else None
        )
        if trace_step_id is None:
            return original_worker_execute_model(worker, scheduler_output)
        if (
            not isinstance(trace_step_id, str)
            or re.fullmatch(r"vllm:[0-9]+", trace_step_id) is None
        ):
            raise TypeError("C500 Graph launch Scheduler step ID is invalid")
        rank = worker.rank
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise TypeError("C500 Graph launch worker rank is invalid")
        with context_lock:
            ordinal = context_ordinal
            context_ordinal += 1
            set_context(trace_step_id.encode("utf-8"), rank, ordinal)
            try:
                return original_worker_execute_model(worker, scheduler_output)
            finally:
                clear_context()

    Worker.execute_model = traced_worker_execute_model


def _install_a800_counter_signal_handler(state: _TraceState) -> None:
    def snapshot_handler(_signum: int, _frame: object) -> None:
        state.worker_snapshot_after_benchmark()

    signal.signal(signal.SIGUSR1, snapshot_handler)
