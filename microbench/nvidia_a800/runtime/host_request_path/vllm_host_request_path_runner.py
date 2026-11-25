#!/usr/bin/env python3
"""Measure the per-request Host request path primitives of one serving Host.

The Serving Forward charges a per-request Host admission cost and a per-request
first-token delivery cost.  Both are built from primitives that this runner
measures independently of any production Serving workload: no model is loaded,
no device is touched, and no production operator, kernel or Serving duration is
read.  Every primitive below is a standalone operation of the pinned runtime on
this Host.

Primitives
----------
``api_entry``
    Client socket write to OpenAI-entrypoint handler entry, over the same
    uvicorn/Starlette stack the vLLM API server runs.
``tokenizer_microbatch_window``
    One uncontended pass through the runtime's own asynchronous tokenizer
    wrapper.  Runtimes whose source batches encode requests behind a wait
    window pay that window once per request; runtimes without the wrapper
    report ``unavailable`` so the mechanism stays out of their domain.
``executor_dispatch``
    One ``loop.run_in_executor`` round trip on a single-worker pool.
``tokenizer_encode``
    Real tokenizer encode, reported as an affine fit over prompt length whose
    slope and intercept are both measured here.
``request_build``
    ``EngineCoreRequest`` construction plus the msgpack encode the client
    performs before the IPC send.
``ipc_transit``
    One-way ZeroMQ transit plus decode between two processes that share this
    Host's monotonic clock.
``first_token_delivery``
    Incremental detokenize, ``RequestOutput`` assembly and one streamed HTTP
    chunk write to a connected client.
``output_fanout``
    The same streamed chunk write when one engine step releases an output into
    every one of ``n`` connected streams, which is what a step carrying ``n``
    scheduled requests does.  Reported as the marginal cost of one additional
    output in the step, measured over a fan-out sweep on this Host.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from typing import Callable, Sequence


_NS_PER_S = 1_000_000_000.0


def _median_ns(samples: Sequence[int]) -> float:
    if not samples:
        raise ValueError("a primitive measurement produced no samples")
    return float(statistics.median(samples))


def _summary(samples: Sequence[int]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_s": _median_ns(ordered) / _NS_PER_S,
        "mean_s": statistics.fmean(ordered) / _NS_PER_S,
        "p10_s": ordered[len(ordered) // 10] / _NS_PER_S,
        "p90_s": ordered[(9 * len(ordered)) // 10] / _NS_PER_S,
        "sample_count": len(ordered),
    }


def _time_calls(operation: Callable[[], None], trials: int, warmup: int) -> list[int]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(trials):
        started_ns = time.perf_counter_ns()
        operation()
        samples.append(time.perf_counter_ns() - started_ns)
    return samples


# --------------------------------------------------------------------------
# executor dispatch
# --------------------------------------------------------------------------


def measure_executor_dispatch(trials: int, warmup: int) -> dict[str, object]:
    """One asyncio -> single-worker thread pool -> asyncio round trip."""

    async def run() -> list[int]:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            def noop() -> None:
                return None

            samples = []
            for index in range(trials + warmup):
                started_ns = time.perf_counter_ns()
                await loop.run_in_executor(executor, noop)
                if index >= warmup:
                    samples.append(time.perf_counter_ns() - started_ns)
            return samples
        finally:
            executor.shutdown(wait=True)

    return _summary(asyncio.run(run()))


# --------------------------------------------------------------------------
# tokenizer micro-batch window
# --------------------------------------------------------------------------


def _async_microbatch_tokenizer_class():
    """Return the runtime's asynchronous batching tokenizer, if it has one."""

    for module_name in ("vllm.utils", "vllm.utils.async_utils"):
        try:
            module = __import__(module_name, fromlist=["AsyncMicrobatchTokenizer"])
        except ImportError:
            continue
        if hasattr(module, "AsyncMicrobatchTokenizer"):
            return getattr(module, "AsyncMicrobatchTokenizer")
    return None


def measure_tokenizer_microbatch_window(
    tokenizer, trials: int, warmup: int,
) -> dict[str, object]:
    """One uncontended encode through the runtime's async tokenizer wrapper.

    A single in-flight request can never fill ``max_batch_size``, so the source
    batching loop always waits its full ``batch_wait_timeout_s`` before it
    dispatches.  That is the occurrence this measurement reproduces.
    """

    wrapper_class = _async_microbatch_tokenizer_class()
    if wrapper_class is None:
        return {
            "status": "unavailable_runtime_has_no_batching_tokenizer",
            "source_symbol": None,
        }

    prompt = "host request path window probe"

    async def run() -> tuple[list[int], object]:
        wrapper = wrapper_class(tokenizer)
        samples = []
        for index in range(trials + warmup):
            started_ns = time.perf_counter_ns()
            await wrapper(prompt)
            if index >= warmup:
                samples.append(time.perf_counter_ns() - started_ns)
        return samples, wrapper

    samples, wrapper = asyncio.run(run())
    result = dict(_summary(samples))
    result["status"] = "available"
    result["source_symbol"] = (
        f"{wrapper_class.__module__}.{wrapper_class.__qualname__}"
    )
    result["declared_batch_wait_timeout_s"] = float(
        getattr(wrapper, "batch_wait_timeout_s")
    )
    result["declared_max_batch_size"] = int(getattr(wrapper, "max_batch_size"))
    return result


# --------------------------------------------------------------------------
# tokenizer encode
# --------------------------------------------------------------------------


def measure_tokenizer_encode(
    tokenizer, lengths: Sequence[int], trials: int, warmup: int,
) -> dict[str, object]:
    """Affine encode cost over prompt length, both terms measured here."""

    points = []
    for target in lengths:
        text = " ".join(["token"] * target)
        encoded = tokenizer(text)
        token_count = len(encoded["input_ids"])
        samples = _time_calls(lambda: tokenizer(text), trials, warmup)
        points.append((token_count, _median_ns(samples) / _NS_PER_S))
    if len(points) < 2:
        raise ValueError("tokenizer encode fit requires at least two lengths")
    points.sort()
    # The two terms are read straight off measured endpoints rather than fitted
    # through a regression: the shortest prompt measures the fixed call cost and
    # the endpoint difference measures the marginal per-token cost.
    (short_tokens, short_cost) = points[0]
    (long_tokens, long_cost) = points[-1]
    if long_tokens <= short_tokens:
        raise ValueError("tokenizer encode probe requires distinct lengths")
    slope = (long_cost - short_cost) / (long_tokens - short_tokens)
    intercept = short_cost - slope * short_tokens
    if slope <= 0.0:
        raise ValueError("tokenizer encode cost must increase with length")
    if intercept <= 0.0:
        # A non-positive constant term means the shortest probe is already
        # dominated by marginal cost; charge the measured short-prompt call.
        intercept = short_cost
        slope = (long_cost - short_cost) / long_tokens
    return {
        "intercept_s": intercept,
        "per_token_s": slope,
        "points": [
            {"prompt_tokens": count, "median_s": cost} for count, cost in points
        ],
    }


def measure_completion_render_region(model_path: str, lengths: Sequence[int],
                                    trials: int, warmup: int) -> dict[str, object]:
    """The region a renderer runtime's completion handler runs to reach the engine.

    vLLM 0.18.0 and 0.24.0 do not dispatch a bare encode from the handler.  The
    handler calls its own ``render_completion_request``, which checks the model
    and then runs ``OpenAIServingRender.render_completion`` -- the render chain
    that reaches the tokenizer, whichever way that runtime reaches it.  Driving
    that region measures what the request actually pays instead of a primitive
    standing in for it, and it subsumes the encode, so a runtime priced this way
    does not also carry separate encode terms.

    The runtime supplies this path without an engine: ``OpenAIModelRegistry`` is
    documented as a model view with no engine dependency, for CPU-only render
    contexts, and the engine configuration is built with a dummy weight format.
    """

    import asyncio as _asyncio
    import os as _os

    _os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    renderer_builder = _resolve_symbol((
        ("vllm.renderers.registry", "renderer_from_config"),
        ("vllm.renderers", "renderer_from_config"),
    ))
    if renderer_builder is None:
        return {"status": "unavailable_runtime_has_no_renderer"}
    registry_class = _resolve_symbol((
        ("vllm.entrypoints.openai.models.serving", "OpenAIModelRegistry"),))
    path_class = _resolve_symbol((
        ("vllm.entrypoints.openai.models.protocol", "BaseModelPath"),))
    serving_class = _resolve_symbol((
        ("vllm.entrypoints.serve.render.serving", "OpenAIServingRender"),))
    request_class = _resolve_symbol((
        ("vllm.entrypoints.openai.completion.protocol", "CompletionRequest"),
        ("vllm.entrypoints.openai.protocol", "CompletionRequest"),
    ))
    missing = [name for name, value in (
        ("OpenAIModelRegistry", registry_class), ("BaseModelPath", path_class),
        ("OpenAIServingRender", serving_class),
        ("CompletionRequest", request_class)) if value is None]
    if missing:
        return {"status": "unavailable_render_serving_symbols_absent",
                "missing": missing}

    from vllm.engine.arg_utils import EngineArgs

    config = EngineArgs(
        model=model_path, load_format="dummy", max_model_len=4096,
        enforce_eager=True, gpu_memory_utilization=0.1).create_engine_config()
    model_config = config.model_config
    registry = registry_class(
        model_config, [path_class(name=model_path, model_path=model_path)])
    serving = serving_class(
        model_config, renderer_builder(config), registry, request_logger=None,
        chat_template=None, chat_template_content_format="auto")

    async def run() -> dict[str, float]:
        medians = {}
        for length in lengths:
            text = " ".join(["token"] * length)
            request = request_class(model=model_path, prompt=text, max_tokens=16)
            samples = []
            for index in range(trials + warmup):
                started_ns = time.perf_counter_ns()
                await registry.check_model(request.model)
                await serving.render_completion(request)
                if index >= warmup:
                    samples.append(time.perf_counter_ns() - started_ns)
            medians[length] = _median_ns(samples) / _NS_PER_S
        return medians

    medians = _asyncio.run(run())
    points = sorted(medians.items())
    if len(points) < 2:
        raise ValueError("render region fit requires at least two lengths")
    (short_tokens, short_cost) = points[0]
    (long_tokens, long_cost) = points[-1]
    slope = (long_cost - short_cost) / (long_tokens - short_tokens)
    intercept = short_cost - slope * short_tokens
    if slope <= 0.0 or intercept <= 0.0:
        raise ValueError(
            "render region cost must rise with prompt length from a positive "
            "fixed cost")
    return {
        "status": "available",
        "intercept_s": intercept,
        "per_token_s": slope,
        "median_s_by_prompt_tokens": {str(k): v for k, v in points},
        "source_symbol": (
            f"{serving_class.__module__}.{serving_class.__qualname__}"
            ".render_completion"),
    }


# --------------------------------------------------------------------------
# request build and IPC transit
# --------------------------------------------------------------------------


def measure_input_processor(model_path: str, prompt_tokens, trials: int,
                            warmup: int) -> dict[str, object]:
    """Real ``process_inputs`` calls on a Host-only engine configuration.

    The engine configuration is built with a dummy weight format so nothing is
    loaded onto a device; only the Host-side request validation and
    ``EngineCoreRequest`` construction that every request pays are exercised.

    The call copies and validates the prompt's token ids, so its cost is affine
    in prompt length.  Passing several lengths reports both terms off measured
    endpoints, the same way the tokenizer encode probe does.
    """

    import os as _os

    _os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams

    engine_args = EngineArgs(
        model=model_path, load_format="dummy", max_model_len=4096,
        enforce_eager=True, gpu_memory_utilization=0.1)
    config = engine_args.create_engine_config()
    # The input processor takes a different collaborator in each pinned
    # runtime and the two forms are not interchangeable: 0.11.0's ``Processor``
    # is handed the tokenizer, while 0.18.0 and 0.24.0 hand ``InputProcessor``
    # a renderer that it builds from the config itself when none is passed.
    # Passing a tokenizer where a renderer belongs is what made this probe
    # report a construction failure on the two newer runtimes.
    processor_forms = (
        ("vllm.v1.engine.input_processor", "InputProcessor", "renderer_from_config"),
        ("vllm.v1.engine.processor", "Processor", "tokenizer"),
    )
    processor_class = None
    for module_name, symbol_name, form in processor_forms:
        candidate = _resolve_symbol(((module_name, symbol_name),))
        if candidate is not None:
            processor_class, processor_form = candidate, form
            break
    if processor_class is None:
        return {"status": "unavailable_runtime_has_no_input_processor"}
    # The helper that builds a runtime tokenizer from a config was renamed
    # across the pinned runtimes; each candidate is tried explicitly and the
    # probe records why it could not run rather than skipping silently.
    processor = None
    construction_errors = []
    if processor_form == "renderer_from_config":
        # The renderer defaults to one built from this same config, which is
        # Host-only: it constructs a cached tokenizer and nothing else.
        try:
            processor = processor_class(config)
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            construction_errors.append(
                f"InputProcessor(config): {type(error).__name__}: {error}")
    else:
        tokenizer_builders = (
            ("vllm.transformers_utils.tokenizer", "init_tokenizer_from_configs",
             lambda fn: fn(config.model_config)),
            ("vllm.transformers_utils.tokenizer", "get_tokenizer",
             lambda fn: fn(model_path)),
        )
        for module_name, symbol_name, invoke in tokenizer_builders:
            builder = _resolve_symbol(((module_name, symbol_name),))
            if builder is None:
                construction_errors.append(f"{symbol_name} absent")
                continue
            try:
                processor = processor_class(config, invoke(builder))
                break
            except Exception as error:  # noqa: BLE001 - reported, never swallowed
                construction_errors.append(f"{symbol_name}: {type(error).__name__}")
    if processor is None:
        return {
            "status": "unavailable_input_processor_construction_failed",
            "attempts": construction_errors,
        }
    sampling_params = SamplingParams(max_tokens=16)
    lengths = ((prompt_tokens,) if isinstance(prompt_tokens, int)
               else tuple(prompt_tokens))
    if any(length <= 0 for length in lengths):
        raise ValueError("input processor probe needs positive prompt lengths")
    token_ids = list(range(2, 2 + lengths[0]))
    counter = [0]
    # 0.18.0 and 0.24.0 validate the request against the tasks the engine
    # supports, so that tuple is part of the call rather than optional; the
    # generate task is the one this path serves.
    import inspect
    takes_supported_tasks = (
        "supported_tasks" in inspect.signature(processor.process_inputs).parameters)
    extra = (("generate",),) if takes_supported_tasks else ()

    def call() -> None:
        counter[0] += 1
        processor.process_inputs(
            f"host-request-path-{counter[0]}",
            {"prompt_token_ids": token_ids},
            sampling_params,
            *extra,
        )

    points = []
    summaries = {}
    for length in lengths:
        token_ids = list(range(2, 2 + length))
        summary = _summary(_time_calls(call, trials, warmup))
        summaries[str(length)] = summary
        points.append((length, summary["median_s"]))
    result = dict(summaries[str(lengths[0])])
    if len(points) >= 2:
        points.sort()
        (short_tokens, short_cost) = points[0]
        (long_tokens, long_cost) = points[-1]
        slope = (long_cost - short_cost) / (long_tokens - short_tokens)
        intercept = short_cost - slope * short_tokens
        if slope <= 0.0 or intercept <= 0.0:
            raise ValueError(
                "input processor cost must rise with prompt length from a "
                "positive fixed call cost")
        result["intercept_s"] = intercept
        result["per_token_s"] = slope
    result["per_length"] = summaries
    result["status"] = "available"
    result["prompt_tokens"] = lengths[0]
    result["source_symbol"] = (
        f"{processor_class.__module__}.{processor_class.__qualname__}")
    result["construction_form"] = processor_form
    result["supported_tasks_in_call"] = takes_supported_tasks
    return result


def _resolve_symbol(candidates):
    for module_name, symbol_name in candidates:
        try:
            module = __import__(module_name, fromlist=[symbol_name])
        except ImportError:
            continue
        if hasattr(module, symbol_name):
            return getattr(module, symbol_name)
    return None


def measure_request_build(tokenizer, prompt_tokens: int, trials: int,
                          warmup: int) -> dict[str, object]:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackEncoder

    encoder = MsgpackEncoder()
    token_ids = list(range(2, 2 + prompt_tokens))
    sampling_params = SamplingParams(max_tokens=16)

    def build() -> None:
        request = _build_engine_core_request(
            EngineCoreRequest, token_ids, sampling_params, tokenizer)
        encoder.encode(request)

    return _summary(_time_calls(build, trials, warmup))


def _build_engine_core_request(request_class, token_ids, sampling_params,
                               tokenizer):
    """Construct the pinned runtime's EngineCoreRequest without guessing fields."""

    candidate = {
        "request_id": "host-request-path-probe",
        "prompt_token_ids": token_ids,
        "mm_features": None,
        "sampling_params": sampling_params,
        "pooling_params": None,
        "eos_token_id": tokenizer.eos_token_id,
        "arrival_time": 0.0,
        "lora_request": None,
        "cache_salt": None,
        "data_parallel_rank": None,
    }
    annotations = getattr(request_class, "__annotations__", {})
    if annotations:
        candidate = {
            name: value for name, value in candidate.items() if name in annotations
        }
    return request_class(**candidate)


_IPC_REGION_CHILD_MARKER = "MIDLAYER_IPC_REGION_CHILD"
_IPC_CHILD_MARKER = "MIDLAYER_HOST_REQUEST_PATH_IPC_CHILD"


def measure_ipc_transit(payload_bytes: int, trials: int,
                        warmup: int) -> dict[str, object]:
    """One-way ZeroMQ transit measured on a single shared monotonic clock.

    Sender and receiver are separate processes on this Host, so the ``perf_counter``
    domain is shared and the one-way delta is meaningful without any correction.
    """

    import subprocess

    import zmq

    address = f"ipc:///tmp/midlayer-host-request-path-{os.getpid()}"
    environment = dict(os.environ)
    environment[_IPC_CHILD_MARKER] = address
    child = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--ipc-child"],
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        context = zmq.Context(io_threads=1)
        sender = context.socket(zmq.PUSH)
        sender.bind(address)
        payload = b"x" * payload_bytes
        for _ in range(warmup + trials):
            sender.send(time.perf_counter_ns().to_bytes(8, "little") + payload)
            time.sleep(0.001)
        sender.send(b"")
        sender.close(linger=2000)
        context.term()
        stdout, _ = child.communicate(timeout=120)
    finally:
        if child.poll() is None:
            child.kill()
    samples = [int(line) for line in stdout.split() if line]
    if len(samples) < warmup + trials:
        raise RuntimeError("IPC transit child returned too few samples")
    return _summary(samples[warmup:])


def measure_admission_enqueue_region(tokenizer, prompt_tokens: int,
                                     trials: int, warmup: int) -> dict[str, object]:
    """Registering a request and handing it to the EngineCore socket.

    Between the input processor and the IPC hop the API server registers the
    request with ``OutputProcessor.add_request`` -- which builds that request's
    state, its incremental detokenizer and its logprobs processor -- and then
    serialises the ``EngineCoreRequest`` for the socket.  The earlier
    ``request_build`` probe only constructed the request object and encoded it,
    which is the smaller half; this drives the region the serving trace
    measures as ``output_processor_add_request`` plus ``engine_core_ipc_enqueue``.
    """

    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackEncoder

    processor_class = _resolve_symbol((
        ("vllm.v1.engine.output_processor", "OutputProcessor"),))
    if processor_class is None:
        return {"status": "unavailable_runtime_has_no_output_processor"}
    # 0.11.0 takes log_stats positionally; 0.18.0 and 0.24.0 make it
    # keyword-only.  Both forms are tried explicitly and the one that built is
    # recorded, so a later signature change shows up rather than falling back.
    processor = None
    form = None
    for label, build in (
        ("keyword_log_stats", lambda: processor_class(tokenizer, log_stats=False)),
        ("positional_log_stats", lambda: processor_class(tokenizer, False)),
    ):
        try:
            processor = build()
            form = label
            break
        except TypeError:
            continue
    if processor is None:
        return {"status": "unavailable_output_processor_construction_failed"}

    encoder = MsgpackEncoder()
    token_ids = list(range(2, 2 + prompt_tokens))
    sampling_params = SamplingParams(max_tokens=16)
    counter = [0]

    def call() -> None:
        counter[0] += 1
        request_id = f"host-request-path-{counter[0]}"
        request = _build_engine_core_request(
            EngineCoreRequest, token_ids, sampling_params, tokenizer)
        request.request_id = request_id
        # 0.24.0 keeps the client-facing id alongside the engine one and
        # asserts it is set before building request state; on runtimes without
        # the field this is simply absent.
        if hasattr(request, "external_req_id"):
            request.external_req_id = request_id
        processor.add_request(request, None)
        encoder.encode(request)

    result = dict(_summary(_time_calls(call, trials, warmup)))
    result["status"] = "available"
    result["prompt_tokens"] = prompt_tokens
    result["construction_form"] = form
    result["source_symbol"] = (
        f"{processor_class.__module__}.{processor_class.__qualname__}"
        ".add_request + MsgpackEncoder.encode")
    return result


def measure_engine_core_ipc_region(tokenizer, prompt_tokens: int, trials: int,
                                   warmup: int) -> dict[str, object]:
    """A real request's transit to EngineCore, decode included.

    The serving trace measures this region from the send-side enqueue ending to
    ``engine_core_preprocess`` ending, so it contains the cross-process wake and
    the receiving side's msgpack decode, not only the wire hop.  The earlier
    ``ipc_transit`` probe sent a fixed opaque payload and never decoded it.
    """

    import subprocess

    import zmq
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackEncoder

    address = f"ipc:///tmp/midlayer-engine-core-ipc-{os.getpid()}"
    environment = dict(os.environ)
    environment[_IPC_REGION_CHILD_MARKER] = address
    # The child imports vLLM, whose logger writes to stdout; the samples are
    # the only thing this channel carries, so quiet it and read digits only.
    environment["VLLM_LOGGING_LEVEL"] = "ERROR"
    child = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--ipc-region-child"],
        env=environment, stdout=subprocess.PIPE, text=True)
    try:
        context = zmq.Context(io_threads=1)
        sender = context.socket(zmq.PUSH)
        sender.bind(address)
        encoder = MsgpackEncoder()
        token_ids = list(range(2, 2 + prompt_tokens))
        sampling_params = SamplingParams(max_tokens=16)
        for index in range(warmup + trials):
            request = _build_engine_core_request(
                EngineCoreRequest, token_ids, sampling_params, tokenizer)
            request.request_id = f"host-request-path-{index}"
            if hasattr(request, "external_req_id"):
                request.external_req_id = request.request_id
            payload = encoder.encode(request)
            frames = (list(payload)
                      if isinstance(payload, (list, tuple)) else [payload])
            sender.send_multipart(
                [time.perf_counter_ns().to_bytes(8, "little")] + frames)
            time.sleep(0.001)
        sender.send_multipart([b""])
        sender.close(linger=2000)
        context.term()
        stdout, _ = child.communicate(timeout=180)
    finally:
        if child.poll() is None:
            child.kill()
    samples = [int(token) for token in stdout.split() if token.isdigit()]
    if len(samples) < warmup + trials:
        raise RuntimeError("EngineCore IPC region child returned too few samples")
    result = dict(_summary(samples[warmup:]))
    result["status"] = "available"
    result["prompt_tokens"] = prompt_tokens
    return result


def _ipc_region_child_main() -> None:
    import zmq
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackDecoder

    address = os.environ[_IPC_REGION_CHILD_MARKER]
    decoder = MsgpackDecoder(EngineCoreRequest)
    context = zmq.Context(io_threads=1)
    receiver = context.socket(zmq.PULL)
    receiver.connect(address)
    while True:
        try:
            frames = receiver.recv_multipart()
        except zmq.ZMQError:
            break
        if len(frames) < 2:
            break
        sent_ns = int.from_bytes(frames[0], "little")
        decoder.decode(frames[1:])
        print(time.perf_counter_ns() - sent_ns, flush=True)


def _ipc_child_main() -> None:
    import zmq

    address = os.environ[_IPC_CHILD_MARKER]
    context = zmq.Context(io_threads=1)
    receiver = context.socket(zmq.PULL)
    receiver.connect(address)
    while True:
        try:
            message = receiver.recv()
        except zmq.ZMQError:
            break
        if not message:
            break
        sent_ns = int.from_bytes(message[:8], "little")
        print(time.perf_counter_ns() - sent_ns, flush=True)


# --------------------------------------------------------------------------
# HTTP entry and streamed first-token delivery
# --------------------------------------------------------------------------


def measure_http_path(trials: int, warmup: int) -> dict[str, object]:
    """Client write to handler entry, and one streamed chunk back to the client.

    The server is the same uvicorn/Starlette stack the vLLM OpenAI entrypoint
    serves with; only the handler body is replaced so that no model is needed.
    """

    import threading

    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    async def handler(request):
        entry_ns = time.perf_counter_ns()
        body = await request.json()

        async def stream():
            yield json.dumps({
                "entry_ns": entry_ns,
                "sent_ns": body["sent_ns"],
                "chunk_ns": time.perf_counter_ns(),
            }).encode("utf-8")

        return StreamingResponse(stream(), media_type="text/event-stream")

    application = Starlette(routes=[Route("/probe", handler, methods=["POST"])])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(application, host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("probe HTTP server did not start")
        time.sleep(0.01)
    try:
        import requests as http_client

        entry_samples = []
        chunk_samples = []
        session = http_client.Session()
        url = f"http://127.0.0.1:{port}/probe"
        for index in range(warmup + trials):
            sent_ns = time.perf_counter_ns()
            response = session.post(url, json={"sent_ns": sent_ns}, stream=True)
            payload = json.loads(response.raw.read())
            received_ns = time.perf_counter_ns()
            response.close()
            if index >= warmup:
                entry_samples.append(payload["entry_ns"] - payload["sent_ns"])
                chunk_samples.append(received_ns - payload["chunk_ns"])
        session.close()
    finally:
        server.should_exit = True
        thread.join(timeout=30.0)
    return {
        "api_entry": _summary(entry_samples),
        "stream_chunk_return": _summary(chunk_samples),
    }


def measure_detokenize_step(tokenizer, trials: int,
                            warmup: int) -> dict[str, object]:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer

    token_ids = tokenizer("host request path delivery probe")["input_ids"]
    request = _build_engine_core_request(
        EngineCoreRequest, list(token_ids), SamplingParams(max_tokens=256),
        tokenizer)
    detokenizer = IncrementalDetokenizer.from_new_request(tokenizer, request)
    next_token = int(token_ids[0])

    def step() -> None:
        detokenizer.update([next_token], False)
        detokenizer.get_next_output_text(False, False)

    return _summary(_time_calls(step, trials, warmup))


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Output fan-out: one engine step carries one output per request in the step
# --------------------------------------------------------------------------


def measure_output_delivery_region(tokenizer, fanouts: Sequence[int],
                                  trials: int, warmup: int) -> dict[str, object]:
    """Superseded; kept only to document what it did measure.

    This encodes and decodes the step's message inline on the API server's own
    event loop, so the cross-process hop is outside it, and it reads its streams
    with one thread each, which inflates the single-output term.  The published
    delivery terms come from the three-process harness with the evidence.

    The whole delivery path a produced token travels, with the real classes.

    A step's outputs leave EngineCore as one msgpack message carrying one
    ``EngineCoreOutput`` per request the step scheduled.  The API server decodes
    that message once, then walks the entries: each one advances its own
    request's incremental detokenizer and becomes a streamed completion chunk
    that the same event loop writes.  A request's own chunk therefore leaves
    behind the other outputs of its step.

    ``measure_output_fanout`` starts after all of that, with a payload already
    sitting in each stream's queue, so its slope covers only the final write.
    This probe starts where the step's outputs do and reports the region, so its
    intercept is the single-output delivery and its slope the marginal cost of
    one further output in the same step.
    """

    import threading

    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                                EngineCoreRequest)
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    response_class = _resolve_symbol((
        ("vllm.entrypoints.openai.completion.protocol",
         "CompletionStreamResponse"),
        ("vllm.entrypoints.openai.protocol", "CompletionStreamResponse"),
    ))
    choice_class = _resolve_symbol((
        ("vllm.entrypoints.openai.completion.protocol",
         "CompletionResponseStreamChoice"),
        ("vllm.entrypoints.openai.protocol", "CompletionResponseStreamChoice"),
    ))
    if response_class is None or choice_class is None:
        return {"status": "unavailable_streaming_response_symbols_absent"}

    prompt_ids = list(tokenizer("output delivery region probe")["input_ids"])
    largest = max(fanouts)
    detokenizers = []
    for index in range(largest):
        request = _build_engine_core_request(
            EngineCoreRequest, prompt_ids, SamplingParams(max_tokens=256),
            tokenizer)
        detokenizers.append(
            IncrementalDetokenizer.from_new_request(tokenizer, request))
    encoder, decoder = MsgpackEncoder(), MsgpackDecoder(EngineCoreOutputs)
    next_token = int(prompt_ids[0])
    queues: dict[int, asyncio.Queue] = {}
    ready = threading.Semaphore(0)
    fanout_box = [largest]

    async def stream_endpoint(request):
        index = int(request.path_params["index"])
        queue: asyncio.Queue = asyncio.Queue()
        queues[index] = queue

        async def body():
            yield b" "
            ready.release()
            while True:
                payload = await queue.get()
                if payload is None:
                    return
                yield payload

        return StreamingResponse(body(), media_type="text/event-stream")

    async def fire_endpoint(request):
        """One engine step's outputs, from the message to the written chunks."""
        fanout = fanout_box[0]
        fired_ns = time.perf_counter_ns()
        message = encoder.encode(EngineCoreOutputs(engine_index=0, outputs=[
            EngineCoreOutput(request_id=f"cmpl-request-{index:012d}-0",
                             new_token_ids=[next_token], finish_reason=None)
            for index in range(fanout)]))
        buffer = (b"".join(message)
                  if isinstance(message, (list, tuple)) else message)
        for output in decoder.decode(buffer).outputs:
            index = int(output.request_id.split("-")[2])
            detokenizer = detokenizers[index]
            detokenizer.update(output.new_token_ids, False)
            text = detokenizer.get_next_output_text(False, False)
            chunk = response_class(
                id=output.request_id, created=1758000000, model="probe",
                choices=[choice_class(index=0, text=text, logprobs=None,
                                      finish_reason=None)])
            queues[index].put_nowait(
                ("data: " + chunk.model_dump_json(exclude_unset=True)
                 + "\n\n").encode("utf-8"))
        return JSONResponse({"fired_ns": fired_ns})

    application = Starlette(routes=[
        Route("/stream/{index:int}", stream_endpoint),
        Route("/fire", fire_endpoint, methods=["POST"]),
    ])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        application, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("output delivery probe server did not start")
        time.sleep(0.01)

    import requests as http_client

    base = f"http://127.0.0.1:{port}"
    per_fanout = {}
    try:
        for fanout in fanouts:
            if fanout < 1:
                raise ValueError("output fan-out must be at least one output")
            queues.clear()
            fanout_box[0] = fanout
            receipts: dict[int, list[int]] = {i: [] for i in range(fanout)}
            errors: list[BaseException] = []

            def read_stream(index: int) -> None:
                try:
                    response = http_client.get(
                        f"{base}/stream/{index}", stream=True, timeout=120.0)
                    lines = response.iter_lines()
                    for _ in range(warmup + trials):
                        line = next(lines)
                        while not line:
                            line = next(lines)
                        receipts[index].append(time.perf_counter_ns())
                    response.close()
                except BaseException as error:  # surfaced after the join
                    errors.append(error)

            readers = []
            for index in range(fanout):
                reader = threading.Thread(target=read_stream, args=(index,))
                reader.start()
                readers.append(reader)
            for _ in range(fanout):
                if not ready.acquire(timeout=120.0):
                    raise RuntimeError("output delivery streams did not open")
            last_ns = []
            for round_index in range(warmup + trials):
                before = {i: len(receipts[i]) for i in receipts}
                fired = http_client.post(f"{base}/fire", timeout=60.0).json()
                round_deadline = time.monotonic() + 120.0
                while any(len(receipts[i]) == before[i] for i in receipts):
                    if time.monotonic() > round_deadline:
                        raise RuntimeError("output delivery round did not finish")
                    time.sleep(0.0002)
                if round_index >= warmup:
                    last_ns.append(max(receipts[i][-1] for i in receipts)
                                   - fired["fired_ns"])
            for queue in list(queues.values()):
                queue.put_nowait(None)
            for reader in readers:
                reader.join(timeout=60.0)
            if errors:
                raise errors[0]
            per_fanout[str(fanout)] = _summary(last_ns)
    finally:
        server.should_exit = True
        thread.join(timeout=30.0)

    points = sorted((int(key), value["median_s"])
                    for key, value in per_fanout.items())
    # A single-output step does not walk any other output, so its cost is a
    # different shape and measures far noisier than the rest of the sweep.  The
    # marginal cost is read off the endpoints of the range that does walk more
    # than one output, and the single-output cost is that line at one -- which
    # is how the model uses the pair.  Every measured fan-out is reported so
    # the read can be checked.
    walking = [point for point in points if point[0] >= 2]
    if len(walking) < 2:
        raise ValueError(
            "output delivery region needs at least two multi-output fan-outs")
    (short_outputs, short_cost) = walking[0]
    (long_outputs, long_cost) = walking[-1]
    slope = (long_cost - short_cost) / (long_outputs - short_outputs)
    intercept = short_cost - slope * (short_outputs - 1)
    if slope <= 0.0 or intercept <= 0.0:
        raise ValueError(
            "output delivery must rise with the step's output count from a "
            "positive single-output cost")
    return {
        "status": "available",
        "per_fanout": per_fanout,
        "fit_fanouts": [point[0] for point in walking],
        "single_output_s": intercept,
        "marginal_s_per_output": slope,
    }


def measure_output_fanout(fanouts: Sequence[int], trials: int,
                          warmup: int) -> dict[str, object]:
    """Superseded; kept only to document what it did measure.

    This begins after the step's outputs are already decoded and queued per
    stream, so its slope covers the final write alone.  The published delivery
    terms come from the three-process harness with the evidence.

    Delivery latency of one stream when a step delivers ``n`` outputs.

    ``OutputProcessor.process_outputs`` walks one ``EngineCoreOutput`` per
    request scheduled in the step and hands each one to that request's own
    streaming response, and every one of those responses is written by the
    same API-server event loop.  A request therefore waits behind the other
    outputs of its own step before its chunk reaches the client.  This probe
    reproduces that structure with the same uvicorn/Starlette stack and no
    model: ``n`` streams are opened, one trigger releases one chunk into each
    of them, and the probe reports how long the last of them takes.
    """

    import threading

    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    queues: dict[int, asyncio.Queue] = {}
    ready = threading.Semaphore(0)

    async def stream_endpoint(request):
        index = int(request.path_params["index"])
        queue: asyncio.Queue = asyncio.Queue()
        queues[index] = queue

        async def body():
            # Flush the response head so the client is parked on the chunk
            # read before the trigger runs, exactly as a streaming completion
            # is parked between tokens.
            yield b" " * 1
            ready.release()
            while True:
                payload = await queue.get()
                if payload is None:
                    return
                yield json.dumps(payload).encode("utf-8") + b"\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    async def fire_endpoint(request):
        fired_ns = time.perf_counter_ns()
        for queue in list(queues.values()):
            queue.put_nowait({"fired_ns": fired_ns})
        return JSONResponse({"fired_ns": fired_ns})

    application = Starlette(routes=[
        Route("/stream/{index:int}", stream_endpoint),
        Route("/fire", fire_endpoint, methods=["POST"]),
    ])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(application, host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("output fan-out probe server did not start")
        time.sleep(0.01)

    import requests as http_client

    base = f"http://127.0.0.1:{port}"
    per_fanout = {}
    try:
        for fanout in fanouts:
            if fanout < 1:
                raise ValueError("output fan-out must be at least one output")
            queues.clear()
            readers = []
            receipts: dict[int, list[int]] = {index: [] for index in range(fanout)}
            errors: list[BaseException] = []

            def read_stream(index: int) -> None:
                try:
                    response = http_client.get(
                        f"{base}/stream/{index}", stream=True, timeout=120.0)
                    lines = response.iter_lines()
                    for _ in range(warmup + trials):
                        line = next(lines)
                        while not line:
                            line = next(lines)
                        receipts[index].append(time.perf_counter_ns())
                    response.close()
                except BaseException as error:  # surfaced after the join
                    errors.append(error)

            for index in range(fanout):
                reader = threading.Thread(target=read_stream, args=(index,))
                reader.start()
                readers.append(reader)
            for _ in range(fanout):
                if not ready.acquire(timeout=120.0):
                    raise RuntimeError("output fan-out streams did not open")
            if len(queues) != fanout:
                raise RuntimeError("output fan-out stream registration is short")
            last_ns = []
            for round_index in range(warmup + trials):
                before = {index: len(receipts[index]) for index in receipts}
                fired = http_client.post(f"{base}/fire", timeout=60.0).json()
                deadline_ns = time.monotonic() + 120.0
                while any(len(receipts[index]) == before[index]
                          for index in receipts):
                    if time.monotonic() > deadline_ns:
                        raise RuntimeError("output fan-out round did not finish")
                    time.sleep(0.0002)
                if round_index >= warmup:
                    last_ns.append(
                        max(receipts[index][-1] for index in receipts)
                        - fired["fired_ns"])
            for queue in list(queues.values()):
                queue.put_nowait(None)
            for reader in readers:
                reader.join(timeout=60.0)
            if errors:
                raise errors[0]
            per_fanout[str(fanout)] = _summary(last_ns)
    finally:
        server.should_exit = True
        thread.join(timeout=30.0)

    points = sorted((int(key), value["median_s"])
                    for key, value in per_fanout.items())
    if len(points) < 3:
        raise ValueError("output fan-out needs at least three fan-out points")
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0:
        raise ValueError("output fan-out points must span more than one value")
    slope_s = sum((point[0] - mean_x) * (point[1] - mean_y)
                  for point in points) / denominator
    return {
        "per_fanout": per_fanout,
        "marginal_s_per_output": slope_s,
        "intercept_s": mean_y - slope_s * mean_x,
    }


def _host_identity() -> dict[str, object]:
    import platform

    model_name = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model_name = line.split(":", 1)[1].strip()
                break
    import vllm

    return {
        "host_cpu": model_name,
        "operating_system": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "vllm_version": vllm.__version__,
        "vllm_path": os.path.dirname(vllm.__file__),
    }


def main() -> None:
    if _IPC_CHILD_MARKER in os.environ and "--ipc-child" in sys.argv:
        _ipc_child_main()
        return
    if (_IPC_REGION_CHILD_MARKER in os.environ
            and "--ipc-region-child" in sys.argv):
        _ipc_region_child_main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--window-trials", type=int, default=60)
    parser.add_argument("--ipc-trials", type=int, default=200)
    parser.add_argument("--ipc-payload-bytes", type=int, default=2048)
    parser.add_argument("--encode-lengths", type=int, nargs="+",
                        default=[16, 128, 512, 1024, 2048])
    parser.add_argument("--build-prompt-tokens", type=int, default=512)
    parser.add_argument("--input-processor-lengths", type=int, nargs="+",
                        default=[16, 128, 1024])
    parser.add_argument("--output-fanouts", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--fanout-trials", type=int, default=40)
    # ``admission_ipc_regions`` drives only the two trace-bounded regions
    # (``measure_admission_enqueue_region`` / ``measure_engine_core_ipc_region``)
    # so each independent repeat is its own process.
    parser.add_argument("--mode", choices=("primitives", "admission_ipc_regions"),
                        default="primitives")
    parser.add_argument("--region-prompt-tokens", type=int, default=86)
    arguments = parser.parse_args()
    if arguments.trials < 20 or arguments.warmup < 1:
        raise ValueError("host request path probes require real repetition")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)
    if arguments.mode == "admission_ipc_regions":
        measurements = {
            "admission_enqueue_region": measure_admission_enqueue_region(
                tokenizer, arguments.region_prompt_tokens, arguments.trials,
                arguments.warmup),
            "engine_core_ipc_region": measure_engine_core_ipc_region(
                tokenizer, arguments.region_prompt_tokens, arguments.trials,
                arguments.warmup),
        }
    else:
        measurements = _primitive_measurements(arguments, tokenizer)
    payload = {
        "schema_version": 1,
        "hardware_id": arguments.hardware_id,
        "measurement_boundary": "host_only_no_device_no_model",
        "production_or_evaluation_duration_consumed": False,
        "tokenizer_path": arguments.tokenizer,
        "host": _host_identity(),
        "runner": {
            "path": os.path.relpath(os.path.abspath(__file__)),
            "mode": arguments.mode,
            "trials": arguments.trials,
            "warmup": arguments.warmup,
        },
        "measurements": measurements,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["measurements"], indent=2, sort_keys=True))


def _primitive_measurements(arguments, tokenizer) -> dict[str, object]:
    return {
        "api_entry_and_stream": measure_http_path(
            arguments.trials, arguments.warmup),
        "executor_dispatch": measure_executor_dispatch(
            arguments.trials, arguments.warmup),
        "tokenizer_microbatch_window": measure_tokenizer_microbatch_window(
            tokenizer, arguments.window_trials, 5),
        "tokenizer_encode": measure_tokenizer_encode(
            tokenizer, arguments.encode_lengths, 50, 10),
        "input_processor": measure_input_processor(
            arguments.tokenizer, tuple(arguments.input_processor_lengths),
            max(40, arguments.trials // 3), 5),
        "request_build": measure_request_build(
            tokenizer, arguments.build_prompt_tokens, arguments.trials,
            arguments.warmup),
        "ipc_transit": measure_ipc_transit(
            arguments.ipc_payload_bytes, arguments.ipc_trials, 20),
        "detokenize_step": measure_detokenize_step(
            tokenizer, arguments.trials, arguments.warmup),
        # The delivery terms are not produced here any more: both probes below
        # measure less than the region the trace bounds, and the published
        # values come from the three-process harness kept with the evidence
        # (experiment_logs/serving/ttft_wait_undercoverage/delivery_v4.py),
        # which sends the step's outputs from a separate EngineCore stand-in
        # and reads them with the benchmark client the capture itself uses.
        # Running them here would hand a later reader stale numbers for a
        # published term.
    }


if __name__ == "__main__":
    main()
