#!/usr/bin/env python3
"""Measure the vLLM EngineCore host round cadence with real loop mechanics.

The probe reproduces the EngineCore busy-loop structure from source
(``v1/engine/core.py`` ``run_busy_loop``/``step_with_batch_queue`` with the
batch queue of depth 2, ``v1/executor/uniproc_executor.py`` non-blocking
collective_rpc with one WorkerAsyncOutput thread, ``v1/worker/gpu_model_runner.py``
``synchronize_input_prep``/``AsyncGPUModelRunnerOutput`` wake path, and the
vllm-ascend ``ACLGraphWrapper`` pre-replay ``current_stream().synchronize()``
barrier) around a real vLLM Scheduler and a controlled NPU device stub.

The device stub duration is a controlled experiment variable, never a Forward
parameter.  The graph lane replays one captured ``torch.npu.NPUGraph`` whose
device duration is independent of the request count, so any request-count
dependence of the measured round cadence is host-side by construction.  The
eager lane enqueues a fixed op chain one op at a time.

Measured per round, with wall clocks: input-queue poll, Scheduler
``schedule``, prepare-inputs event synchronization, per-request input-state
host preparation, pinned H2D enqueue, pre-replay stream synchronization,
replay/eager enqueue, sampler submit with async D2H output construction,
thread-pool submit, queue decision, wake block on the oldest future (event
synchronize + ``tolist`` inside the WorkerAsyncOutput thread), Scheduler
``update_from_output``, and per-request output-queue puts.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import queue as queue_module
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import torch_npu
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


HARDWARE_ID = "ascend_910b"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
SCHEMA_VERSION = 1

REQUEST_COUNTS = (1, 8, 32, 64, 120, 128)
GRAPH_CHAIN_OPS = (2, 64)
GRAPH_MATRIX_SIZE = 1024
EAGER_MATRIX_SIZE = 256
EAGER_REQUEST_SWEEP_OPS = 700
EAGER_REQUEST_COUNTS = (1, 32, 120, 128)
EAGER_OP_SWEEP_REQUEST_COUNT = 32
EAGER_OP_COUNTS = (64, 256, 700)
PROMPT_TOKENS = 64
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 128
MAX_NUM_BATCHED_TOKENS = 8192
SAMPLER_STUB_WIDTH = 1
BLOCK_QUEUE_SIZE = 2


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    owner_size = len(OUTPUT_OWNER_PARTS)
    offsets = tuple(
        index
        for index in range(len(resolved.parts) - owner_size + 1)
        if resolved.parts[index:index + owner_size] == OUTPUT_OWNER_PARTS
    )
    if len(offsets) != 1 or len(resolved.parts) <= offsets[0] + owner_size:
        raise ValueError("--output-dir must be a run below the Ascend owner")
    if resolved.exists():
        raise FileExistsError("--output-dir must not already exist")
    return resolved


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--measured-rounds", type=int, default=200)
    parser.add_argument("--warmup-rounds", type=int, default=30)
    args = parser.parse_args()
    args.output_dir = _validate_output_dir(args.output_dir)
    if args.device_id not in (0, 1, 2, 3, 4, 5, 6, 7):
        parser.error("--device-id must be an authorized NPU index 0-7")
    if len(args.baseline_commit) != 40 or any(
            character not in "0123456789abcdef"
            for character in args.baseline_commit):
        parser.error("--baseline-commit must be a full lowercase Git SHA")
    if len(args.expected_source_sha256) != 64:
        parser.error("--expected-source-sha256 must be a SHA256 digest")
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    if args.measured_rounds < 40 or args.warmup_rounds < 8:
        parser.error("round counts must keep the predeclared minimums")
    if not Path(args.model_path).is_dir():
        parser.error("--model-path must be a local model directory")
    return args


class _RoundTimings:
    """Per-round wall timestamps collected on the EngineCore host thread."""

    __slots__ = (
        "round_start_ns", "poll_ns", "plan_ns", "prep_sync_ns",
        "state_host_ns", "h2d_ns", "pre_replay_sync_ns", "submit_lane_ns",
        "sample_submit_ns", "thread_submit_ns", "queue_decision_ns",
        "wake_block_ns", "d2h_event_sync_ns", "tolist_ns", "update_ns",
        "put_ns", "blocked_round",
    )

    def __init__(self):
        for name in self.__slots__:
            setattr(self, name, 0)
        self.round_start_ns = time.perf_counter_ns()
        self.blocked_round = False

    def row(self, sequence, request_count, lane, stub_label):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "request_count": request_count,
            "lane": lane,
            "stub": stub_label,
            "blocked_round": self.blocked_round,
        }
        for name in self.__slots__:
            payload[name] = int(getattr(self, name))
        return payload


class _StubAsyncOutput:
    """Source-faithful AsyncGPUModelRunnerOutput wake primitive.

    Mirrors ``vllm/v1/worker/gpu_model_runner.py``: an async D2H copy of the
    sampled token ids on a dedicated copy stream with a completion event;
    ``get_output`` blocks on the event and materializes the per-request
    ``tolist`` result.
    """

    def __init__(self, sampled_cpu, ready_event, req_ids, req_id_to_index):
        self._sampled_cpu = sampled_cpu
        self._ready_event = ready_event
        self._req_ids = req_ids
        self._req_id_to_index = req_id_to_index
        self.d2h_event_sync_ns = 0
        self.tolist_ns = 0

    def get_output(self) -> ModelRunnerOutput:
        start = time.perf_counter_ns()
        self._ready_event.synchronize()
        mid = time.perf_counter_ns()
        request_count = len(self._req_ids)
        sampled = self._sampled_cpu[:request_count].tolist()
        end = time.perf_counter_ns()
        self.d2h_event_sync_ns = mid - start
        self.tolist_ns = end - mid
        return ModelRunnerOutput(
            req_ids=list(self._req_ids),
            req_id_to_index=dict(self._req_id_to_index),
            sampled_token_ids=sampled,
        )


class _StubWorker:
    """Source-faithful driver-worker data plane on a controlled NPU stub.

    ``execute_model`` follows the source order: prepare-inputs event
    synchronization (``synchronize_input_prep``), per-request input-state
    host preparation, pinned H2D input enqueue with the event record, the
    ACLGraphWrapper pre-replay stream barrier, and the device stub submit.
    ``sample_tokens`` follows the sampler submit and the async output copy
    construction.  Device stub durations are calibration evidence only.
    """

    def __init__(self, device_id: int, lane: str, stub):
        self._device = torch.device(f"npu:{device_id}")
        self.lane = lane
        self.stub = stub
        self.prepare_inputs_event = torch.npu.Event()
        self.output_copy_stream = torch.npu.Stream(device=self._device)
        max_reqs = MAX_NUM_SEQS
        self._input_ids_host = torch.zeros(
            max_reqs, dtype=torch.int32, pin_memory=True)
        self._positions_host = torch.zeros(
            max_reqs, dtype=torch.int64, pin_memory=True)
        self._input_ids_device = torch.zeros(
            max_reqs, dtype=torch.int32, device=self._device)
        self._positions_device = torch.zeros(
            max_reqs, dtype=torch.int64, device=self._device)
        self._sampled_source = torch.zeros(
            (max_reqs, SAMPLER_STUB_WIDTH), dtype=torch.int64,
            device=self._device)
        self._sampled_host = torch.zeros(
            (max_reqs, SAMPLER_STUB_WIDTH), dtype=torch.int64,
            pin_memory=True)
        self._base_positions = np.arange(max_reqs, dtype=np.int64)
        self._token_source = torch.arange(max_reqs * 4, dtype=torch.int32)
        self._round_counter = 0
        self._req_ids: list[str] = []
        self._req_id_to_index: dict[str, int] = {}
        self._thread_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="WorkerAsyncOutput")
        self.last_async_output: _StubAsyncOutput | None = None

    def _state_host_prepare(self, request_count: int):
        """Mirror the source per-request input index arithmetic on CPU."""
        scheduled = np.ones(request_count, dtype=np.int32)
        req_indices = np.repeat(
            np.arange(request_count, dtype=np.int32), scheduled)
        arange = np.arange(request_count, dtype=np.int32)
        positions = self._base_positions[req_indices] + self._round_counter
        token_indices = positions + req_indices * max(request_count, 1)
        source = self._token_source
        index = torch.from_numpy(token_indices % len(self._token_source))
        selected = torch.index_select(source, 0, index)
        if selected.numel() != request_count:
            raise RuntimeError("state preparation changed token count")
        self._input_ids_host[:request_count] = selected
        self._positions_host[:request_count] = torch.from_numpy(positions)
        self._round_counter += 1

    def execute_model(self, scheduler_output, timings: _RoundTimings):
        req_ids = list(scheduler_output.num_scheduled_tokens)
        self._req_ids = req_ids
        self._req_id_to_index = {
            request_id: index
            for index, request_id in enumerate(req_ids)
        }
        request_count = len(req_ids)
        if request_count == 0:
            raise RuntimeError("probe round scheduled no requests")
        start = time.perf_counter_ns()
        self.prepare_inputs_event.synchronize()
        timings.prep_sync_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        self._state_host_prepare(request_count)
        timings.state_host_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        self._input_ids_device[:request_count].copy_(
            self._input_ids_host[:request_count], non_blocking=True)
        self._positions_device[:request_count].copy_(
            self._positions_host[:request_count], non_blocking=True)
        timings.h2d_ns = time.perf_counter_ns() - start
        self.prepare_inputs_event.record()
        if self.lane == "graph":
            start = time.perf_counter_ns()
            torch.npu.current_stream(self._device).synchronize()
            timings.pre_replay_sync_ns = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            self.stub.replay()
            timings.submit_lane_ns = time.perf_counter_ns() - start
        else:
            start = time.perf_counter_ns()
            self.stub()
            timings.submit_lane_ns = time.perf_counter_ns() - start
        return None

    def sample_tokens(self, timings: _RoundTimings) -> Future:
        start = time.perf_counter_ns()
        request_count = len(self._req_ids)
        self._sampled_source[:request_count].add_(1)
        ready_event = torch.npu.Event()
        current = torch.npu.current_stream(self._device)
        with torch.npu.stream(self.output_copy_stream):
            self.output_copy_stream.wait_stream(current)
            self._sampled_host[:request_count].copy_(
                self._sampled_source[:request_count], non_blocking=True)
            ready_event.record()
        timings.sample_submit_ns = time.perf_counter_ns() - start
        output = _StubAsyncOutput(
            self._sampled_host, ready_event,
            self._req_ids, self._req_id_to_index)
        self.last_async_output = output
        start = time.perf_counter_ns()
        future = self._thread_pool.submit(output.get_output)
        timings.thread_submit_ns = time.perf_counter_ns() - start
        return future

    def collect_wake_timings(self, timings: _RoundTimings):
        output = self.last_async_output
        if output is None:
            raise RuntimeError("wake timings requested before sampling")
        timings.d2h_event_sync_ns = output.d2h_event_sync_ns
        timings.tolist_ns = output.tolist_ns

    def close(self):
        self._thread_pool.shutdown(wait=True)


class _StubExecutor:
    """Source-faithful UniProcExecutor non-blocking collective_rpc seam."""

    def __init__(self, worker: _StubWorker):
        self._worker = worker

    def execute_model(self, scheduler_output, timings) -> Future:
        result = self._worker.execute_model(scheduler_output, timings)
        future = Future()
        future.set_result(result)
        return future

    def sample_tokens(self, timings) -> Future:
        return self._worker.sample_tokens(timings)


class _GraphStub:
    """One captured NPUGraph replaying a fixed mm chain (fixed duration)."""

    def __init__(self, device, chain_ops: int, matrix_size: int):
        self._left = torch.ones(
            (matrix_size, matrix_size), dtype=torch.float16, device=device)
        self._right = torch.eye(
            matrix_size, dtype=torch.float16, device=device)
        self._first = torch.empty(
            (matrix_size, matrix_size), dtype=torch.float16, device=device)
        self._second = torch.empty(
            (matrix_size, matrix_size), dtype=torch.float16, device=device)
        self._chain_ops = chain_ops
        self._graph = torch.npu.NPUGraph()
        side_stream = torch.npu.Stream(device=device)
        side_stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(side_stream):
            for _ in range(3):
                self._run_chain(chain_ops)
        torch.npu.current_stream(device).wait_stream(side_stream)
        with torch.npu.graph(self._graph):
            self._run_chain(chain_ops)
        self.device_duration_ns = 0

    def _run_chain(self, chain_ops: int):
        source, destination = self._left, self._first
        for _ in range(chain_ops):
            torch.mm(source, self._right, out=destination)
            source, destination = destination, source
        return source

    def calibrate(self) -> int:
        events = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        for _ in range(4):
            self.replay()
        torch.npu.current_stream().synchronize()
        samples = []
        for _ in range(10):
            events[0].record()
            self.replay()
            events[1].record()
            torch.npu.current_stream().synchronize()
            samples.append(events[0].elapsed_time(events[1]) * 1_000_000)
        self.device_duration_ns = int(statistics.median(samples))
        return self.device_duration_ns

    def replay(self):
        self._graph.replay()


class _EagerStub:
    """A fixed per-op eager mm chain enqueued one op at a time."""

    def __init__(self, device, chain_ops: int, matrix_size: int):
        self._left = torch.ones(
            (matrix_size, matrix_size), dtype=torch.float16, device=device)
        self._right = torch.eye(
            matrix_size, dtype=torch.float16, device=device)
        self._first = torch.empty(
            (matrix_size, matrix_size), dtype=torch.float16, device=device)
        self._second = torch.empty(
            (matrix_size, matrix_size), dtype=torch.float16, device=device)
        self._chain_ops = chain_ops
        self.device_duration_ns = 0

    def __call__(self):
        source, destination = self._left, self._first
        for _ in range(self._chain_ops):
            torch.mm(source, self._right, out=destination)
            source, destination = destination, source

    def calibrate(self) -> int:
        events = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        for _ in range(3):
            self()
        torch.npu.current_stream().synchronize()
        samples = []
        for _ in range(10):
            events[0].record()
            self()
            events[1].record()
            torch.npu.current_stream().synchronize()
            samples.append(events[0].elapsed_time(events[1]) * 1_000_000)
        self.device_duration_ns = int(statistics.median(samples))
        return self.device_duration_ns


def _scheduler(model_path: str) -> AsyncScheduler:
    """Real AsyncScheduler, the class async_scheduling selects in vLLM 0.18.

    ``SchedulerConfig.get_scheduler_cls`` returns ``AsyncScheduler`` when
    ``async_scheduling`` is set, and only that subclass keeps the
    output-placeholder bookkeeping that lets a step be scheduled while the
    previous step is still executing (the U(k)=P(k+2) overlap the EngineCore
    batch queue creates).
    """
    model = ModelConfig(
        model=model_path,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        async_scheduling=True,
        is_encoder_decoder=model.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    config = VllmConfig(
        model_config=model,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )
    kv_cache = KVCacheConfig(
        num_blocks=8192,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(
            ["layer"], FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=1,
                dtype=torch.float32))],
    )
    cache_config.num_gpu_blocks = 8192
    return AsyncScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache,
        block_size=16,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(config),
    )


def _admit_requests(
    scheduler: AsyncScheduler, request_count: int, sample: int, max_tokens: int
):
    init_none_hash(sha256)
    hasher = get_request_block_hasher(16, sha256)
    for index in range(request_count):
        prompt = [
            (sample * 131 + index * 17 + position) % 32000
            for position in range(PROMPT_TOKENS)
        ]
        scheduler.add_request(Request(
            request_id=f"{sample}-{index}",
            prompt_token_ids=prompt,
            sampling_params=SamplingParams(max_tokens=max_tokens),
            pooling_params=None,
            block_hasher=hasher,
        ))


def _run_cell(
    request_count: int,
    lane: str,
    stub,
    stub_label: str,
    scheduler: AsyncScheduler,
    executor: _StubExecutor,
    worker: _StubWorker,
    input_queue,
    output_queue,
    measured_rounds: int,
    warmup_rounds: int,
    trials_path: Path,
    sequence_start: int,
):
    batch_queue = deque(maxlen=BLOCK_QUEUE_SIZE)
    rounds_executed = 0
    sequence = sequence_start
    rounds = []
    rows = []
    total_rounds = warmup_rounds + measured_rounds

    def _step_with_batch_queue(timings: _RoundTimings):
        # Mirrors core.py step_with_batch_queue with batch_queue_size == 2.
        model_executed = False
        if scheduler.has_requests():
            start = time.perf_counter_ns()
            output = scheduler.schedule()
            timings.plan_ns = time.perf_counter_ns() - start
            exec_future = executor.execute_model(output, timings)
            model_executed = output.total_num_scheduled_tokens > 0
            future = executor.sample_tokens(timings)
            batch_queue.appendleft((future, output, exec_future))
            start = time.perf_counter_ns()
            early_return = (
                model_executed
                and len(batch_queue) < BLOCK_QUEUE_SIZE
                and not batch_queue[-1][0].done())
            timings.queue_decision_ns = time.perf_counter_ns() - start
            if early_return:
                return None, True
        elif not batch_queue:
            return None, False
        future, output, _exec_future = batch_queue.pop()
        timings.blocked_round = True
        start = time.perf_counter_ns()
        model_output = future.result()
        timings.wake_block_ns = time.perf_counter_ns() - start
        worker.collect_wake_timings(timings)
        start = time.perf_counter_ns()
        outputs = scheduler.update_from_output(output, model_output)
        timings.update_ns = time.perf_counter_ns() - start
        return outputs, True

    def _step(timings: _RoundTimings):
        # Mirrors _process_input_queue + _process_engine_step.
        start = time.perf_counter_ns()
        while not input_queue.empty():
            input_queue.get_nowait()
        timings.poll_ns = time.perf_counter_ns() - start
        outputs, _model_executed = _step_with_batch_queue(timings)
        start = time.perf_counter_ns()
        if outputs is not None:
            for item in outputs.items():
                output_queue.put_nowait(item)
        timings.put_ns = time.perf_counter_ns() - start
        return outputs

    while ((scheduler.has_requests() or batch_queue)
           and rounds_executed < total_rounds):
        timings = _RoundTimings()
        _step(timings)
        rounds_executed += 1
        if rounds_executed > warmup_rounds:
            sequence += 1
            rounds.append(timings)
            rows.append(timings.row(sequence, request_count, lane, stub_label))
    # Write the trials outside the measured loop: serializing every row
    # inside the loop would add file I/O to the round cadence itself.
    with trials_path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    # Drain pending pipeline entries so the cell ends with a clean queue.
    while batch_queue:
        future, output, _exec_future = batch_queue.pop()
        model_output = future.result()
        worker.collect_wake_timings(_RoundTimings())
        scheduler.update_from_output(output, model_output)
    while not output_queue.empty():
        output_queue.get_nowait()
    return rounds, sequence


def _median(values):
    if not values:
        raise ValueError("median requires samples")
    return int(statistics.median(values))


_SEGMENT_KEYS = (
    "poll_ns", "plan_ns", "prep_sync_ns", "state_host_ns", "h2d_ns",
    "pre_replay_sync_ns", "submit_lane_ns", "sample_submit_ns",
    "thread_submit_ns", "queue_decision_ns", "wake_block_ns",
    "d2h_event_sync_ns", "tolist_ns", "update_ns", "put_ns",
)


def _cell_summary(rounds, request_count, lane, stub_label, stub_duration_ns):
    starts = [row.round_start_ns for row in rounds]
    cadences = [
        starts[index + 1] - starts[index]
        for index in range(len(starts) - 1)
    ]
    blocked = [row for row in rounds if row.blocked_round]
    summary = {
        "request_count": request_count,
        "lane": lane,
        "stub": stub_label,
        "stub_device_duration_ns": stub_duration_ns,
        "sample_count": len(rounds),
        "blocked_round_count": len(blocked),
        "round_cadence_ns": _median(cadences),
    }
    for key in _SEGMENT_KEYS:
        summary[key] = _median([getattr(row, key) for row in rounds])
    return summary


def _linear_fit(points):
    count = len(points)
    x_sum = sum(x for x, _ in points)
    y_sum = sum(y for _, y in points)
    xx_sum = sum(x * x for x, _ in points)
    xy_sum = sum(x * y for x, y in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("request slope fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    return float(intercept), float(slope)


def _fit_request_slope(cells, key):
    points = [
        (int(cell["request_count"]), int(cell[key]))
        for cell in cells
    ]
    if len(points) < 2:
        raise RuntimeError("request slope fit requires two cells")
    return _linear_fit(points)


def main():
    args = _parse_args()
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256(source_path)
    if source_sha256 != args.expected_source_sha256:
        raise RuntimeError("runner source SHA256 differs from the audited snapshot")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "manifest.json"
    trials_path = args.output_dir / "trials.jsonl"
    summary_path = args.output_dir / "summary.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "kind": "ascend_engine_core_host_round_microbenchmark",
        "hardware_id": HARDWARE_ID,
        "baseline_commit": args.baseline_commit,
        "container_name": args.container_name,
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "source": {"path": str(source_path), "sha256": source_sha256},
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "model_path": str(Path(args.model_path).resolve()),
        },
        "contract": {
            "hypothesis": (
                "the EngineCore host round cadence is a serial host chain "
                "(plan + submit + wake + update + puts) whose per-request "
                "slope is host-side and independent of device stub duration"
            ),
            "loop_source": (
                "vllm 0.18 EngineCore run_busy_loop / step_with_batch_queue "
                "with batch queue depth 2; UniProcExecutor non-blocking "
                "collective_rpc with one WorkerAsyncOutput thread; "
                "GPUModelRunner synchronize_input_prep event barrier; "
                "ACLGraphWrapper pre-replay current_stream synchronize; "
                "AsyncGPUModelRunnerOutput copy-stream D2H wake with tolist"
            ),
            "request_counts": list(REQUEST_COUNTS),
            "graph_chain_ops": list(GRAPH_CHAIN_OPS),
            "eager_request_counts": list(EAGER_REQUEST_COUNTS),
            "eager_op_counts": list(EAGER_OP_COUNTS),
            "prompt_tokens": PROMPT_TOKENS,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "measured_rounds": args.measured_rounds,
            "warmup_rounds": args.warmup_rounds,
            "timers": ["time.perf_counter_ns"],
            "device_stub_durations_are_forward_parameters": False,
            "production_or_evaluation_duration_consumed": False,
            "scheduler_requests_synthetic": True,
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
    }
    _write_json(manifest_path, manifest)
    try:
        torch.npu.set_device(args.device_id)
        device = torch.device(f"npu:{args.device_id}")
        input_queue = queue_module.Queue()
        output_queue = queue_module.Queue()
        cells = []
        sequence = 0
        max_tokens = args.warmup_rounds + args.measured_rounds + 8

        def _graph_cell(request_count, sample_index, chain_ops, stub_label):
            nonlocal sequence
            scheduler = _scheduler(args.model_path)
            _admit_requests(scheduler, request_count, sample_index, max_tokens)
            stub = _GraphStub(device, chain_ops, GRAPH_MATRIX_SIZE)
            stub_duration_ns = stub.calibrate()
            worker = _StubWorker(args.device_id, "graph", stub)
            executor = _StubExecutor(worker)
            rounds, sequence = _run_cell(
                request_count, "graph", stub, stub_label, scheduler,
                executor, worker, input_queue, output_queue,
                args.measured_rounds, args.warmup_rounds,
                trials_path, sequence)
            worker.close()
            return _cell_summary(
                rounds, request_count, "graph", stub_label, stub_duration_ns)

        def _eager_cell(request_count, sample_index, chain_ops):
            nonlocal sequence
            scheduler = _scheduler(args.model_path)
            _admit_requests(scheduler, request_count, sample_index, max_tokens)
            stub = _EagerStub(device, chain_ops, EAGER_MATRIX_SIZE)
            stub_duration_ns = stub.calibrate()
            worker = _StubWorker(args.device_id, "eager", stub)
            executor = _StubExecutor(worker)
            rounds, sequence = _run_cell(
                request_count, "eager", stub, f"ops{chain_ops}", scheduler,
                executor, worker, input_queue, output_queue,
                args.measured_rounds, args.warmup_rounds,
                trials_path, sequence)
            worker.close()
            return _cell_summary(
                rounds, request_count, "eager", f"ops{chain_ops}",
                stub_duration_ns)

        for sample_index, request_count in enumerate(REQUEST_COUNTS):
            for chain_ops, stub_label in (
                    (GRAPH_CHAIN_OPS[0], "small"),
                    (GRAPH_CHAIN_OPS[1], "large")):
                cells.append(_graph_cell(
                    request_count, sample_index, chain_ops, stub_label))
        for sample_index, request_count in enumerate(EAGER_REQUEST_COUNTS):
            cells.append(_eager_cell(
                request_count, sample_index, EAGER_REQUEST_SWEEP_OPS))
        for op_count in EAGER_OP_COUNTS[:-1]:
            cells.append(_eager_cell(
                EAGER_OP_SWEEP_REQUEST_COUNT, len(REQUEST_COUNTS), op_count))

        graph_large = [
            cell for cell in cells
            if cell["lane"] == "graph" and cell["stub"] == "large"]
        graph_small = [
            cell for cell in cells
            if cell["lane"] == "graph" and cell["stub"] == "small"]
        eager_sweep = [
            cell for cell in cells
            if cell["lane"] == "eager" and cell["stub"] == "ops700"]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "parameter": "engine_core_host_round_cadence",
            "cells": cells,
            "request_slope_fits": {
                "graph_large": {
                    key: _fit_request_slope(graph_large, key)
                    for key in _SEGMENT_KEYS
                },
                "graph_small": {
                    key: _fit_request_slope(graph_small, key)
                    for key in _SEGMENT_KEYS
                },
                "eager_ops700": {
                    key: _fit_request_slope(eager_sweep, key)
                    for key in _SEGMENT_KEYS
                },
            },
        }
        summary["request_slope_fits"]["round_cadence"] = {
            "graph_large": _fit_request_slope(
                graph_large, "round_cadence_ns"),
            "graph_small": _fit_request_slope(
                graph_small, "round_cadence_ns"),
            "eager_ops700": _fit_request_slope(
                eager_sweep, "round_cadence_ns"),
        }
        _write_json(summary_path, summary)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = _utc_now()
        manifest["artifacts"] = {
            "trials": {"path": str(trials_path), "sha256": _sha256(trials_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        }
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        _write_json(manifest_path, manifest)
    print(json.dumps(json.loads(summary_path.read_text())["cells"], indent=2))


if __name__ == "__main__":
    main()
