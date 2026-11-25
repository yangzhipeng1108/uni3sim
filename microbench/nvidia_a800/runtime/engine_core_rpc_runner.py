#!/usr/bin/env python3
"""Probe vLLM 0.11 EngineCore RPC transport with synthetic Scheduler outputs.

The worker acknowledges an actual MultiprocExecutor RPC with a normal
ModelRunnerOutput.  It performs no model work and uses no GPU.  The round trip
therefore measures transport and Python control service, not Device service or
the complete EngineCore round.  It is diagnostic and cannot directly price
Serving Forward because real Device/Host overlap is deliberately absent.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import threading
import time
from types import SimpleNamespace

from vllm.config import ModelConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.sampling_params import SamplingParams
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.executor import multiproc_executor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from microbench.nvidia_a800.runtime.scheduler_host_mechanism_runner import (
    _scheduler,
    _update,
)


COUNTS = (1, 8, 32, 64, 128)
PHASES = ("prefill", "decode")
PROMPT_TOKENS = 64
BLOCK_SIZE = 16


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker(input_handle, handle_pipe, cpu: int) -> None:
    os.sched_setaffinity(0, {cpu})
    incoming = MessageQueue.create_from_handle(input_handle, 0)
    outgoing = MessageQueue(1, 1)
    handle_pipe.send(outgoing.export_handle())
    handle_pipe.close()
    incoming.wait_until_ready()
    outgoing.wait_until_ready()
    while True:
        method, args, kwargs, reply_rank = incoming.dequeue(timeout=20)
        if method == "probe_stop":
            if args or kwargs or reply_rank != 0:
                raise RuntimeError("invalid synthetic worker stop message")
            return
        if method != "execute_model" or len(args) != 1 or kwargs or reply_rank != 0:
            raise RuntimeError("unexpected synthetic worker RPC")
        scheduled = args[0]
        ids = list(scheduled.num_scheduled_tokens)
        output = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={request_id: index
                             for index, request_id in enumerate(ids)},
            sampled_token_ids=[[1] for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None for _ in ids],
        )
        # perf_counter_ns uses the same monotonic clock in both local processes.
        # This timestamp follows synthetic output construction and precedes
        # the installed worker response queue's serialization and notification.
        output._probe_ready_ns = time.perf_counter_ns()
        outgoing.enqueue((multiproc_executor.WorkerProc.ResponseStatus.SUCCESS,
                          output))


def _scheduler_output(model: ModelConfig, count: int, phase: str):
    scheduler = _scheduler(model)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    for index in range(count):
        scheduler.add_request(Request(
            request_id=f"synthetic-{phase}-{count}-{index}",
            prompt_token_ids=[index + 1] * PROMPT_TOKENS,
            sampling_params=SamplingParams(max_tokens=3),
            pooling_params=None,
            eos_token_id=None,
            block_hasher=hasher,
        ))
    first = scheduler.schedule()
    if (len(first.num_scheduled_tokens) != count
            or first.total_num_scheduled_tokens != count * PROMPT_TOKENS):
        raise RuntimeError("synthetic prefill batch differs from declared state")
    if phase == "prefill":
        return first
    if phase != "decode":
        raise ValueError("unsupported synthetic phase")
    _update(scheduler, first)
    second = scheduler.schedule()
    if (len(second.num_scheduled_tokens) != count
            or second.total_num_scheduled_tokens != count):
        raise RuntimeError("synthetic decode batch differs from declared state")
    return second


def _executor(cpu: int):
    context = multiprocessing.get_context("spawn")
    incoming = MessageQueue(1, 1)
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker,
                              args=(incoming.export_handle(), child, cpu))
    process.start()
    child.close()
    if not parent.poll(20):
        process.terminate()
        process.join(timeout=10)
        raise TimeoutError("synthetic worker did not publish response queue")
    response_handle = parent.recv()
    parent.close()
    outgoing = MessageQueue.create_from_handle(response_handle, 0)
    incoming.wait_until_ready()
    outgoing.wait_until_ready()
    executor = object.__new__(multiproc_executor.MultiprocExecutor)
    executor.is_failed = False
    executor.rpc_broadcast_mq = incoming
    executor.workers = (SimpleNamespace(worker_response_mq=outgoing),)
    executor.output_rank = 0
    executor.has_connector = False
    executor.io_thread_pool = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mp_exec_io")
    executor.shutdown_event = threading.Event()
    return executor, process


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--counts", type=int, nargs="+", default=COUNTS)
    parser.add_argument("--phases", nargs="+", default=PHASES)
    args = parser.parse_args()
    if importlib.metadata.version("vllm") != "0.11.0":
        raise RuntimeError("EngineCore RPC probe requires vLLM 0.11.0")
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError("requested CPU is outside process affinity")
    if args.warmup < 10 or args.samples < 40:
        raise ValueError("probe needs at least 10 warmup and 40 measured rounds")
    if not args.counts or any(count not in COUNTS for count in args.counts):
        raise ValueError("counts must be a nonempty subset of the declared grid")
    if len(set(args.counts)) != len(args.counts):
        raise ValueError("counts must be unique")
    if not args.phases or any(phase not in PHASES for phase in args.phases):
        raise ValueError("phases must be a nonempty subset of the declared phases")
    if len(set(args.phases)) != len(args.phases):
        raise ValueError("phases must be unique")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {args.cpu})
    model_dir = args.output.parent / "synthetic_model"
    model_dir.mkdir(exist_ok=True)
    local_config = model_dir / "config.json"
    if not local_config.exists():
        local_config.write_bytes(args.model_config.read_bytes())
    elif local_config.read_bytes() != args.model_config.read_bytes():
        raise ValueError("synthetic model config differs from declared source")
    model = ModelConfig(
        model=str(model_dir), dtype="bfloat16", max_model_len=4096,
        skip_tokenizer_init=True, enforce_eager=True)
    init_none_hash(sha256)
    payloads = {
        (phase, count): _scheduler_output(model, count, phase)
        for phase in args.phases for count in args.counts
    }
    executor, process = _executor(args.cpu)
    rows = []
    try:
        for phase in args.phases:
            for count in args.counts:
                scheduled = payloads[(phase, count)]
                submit_ns = []
                round_trip_ns = []
                ready_to_receive_ns = []
                for index in range(args.warmup + args.samples):
                    start = time.perf_counter_ns()
                    future = executor.execute_model(scheduled, non_block=True)
                    submitted = time.perf_counter_ns()
                    output = future.result(timeout=10)
                    finished = time.perf_counter_ns()
                    if tuple(output.req_ids) != tuple(scheduled.num_scheduled_tokens):
                        raise RuntimeError("synthetic response lost scheduled requests")
                    ready_ns = output._probe_ready_ns
                    if not start <= ready_ns <= finished:
                        raise RuntimeError("cross-process timestamp ordering failed")
                    if index >= args.warmup:
                        submit_ns.append(submitted - start)
                        round_trip_ns.append(finished - start)
                        ready_to_receive_ns.append(finished - ready_ns)
                rows.append({
                    "phase": phase,
                    "scheduled_request_count": count,
                    "scheduled_tokens": scheduled.total_num_scheduled_tokens,
                    "submit_ns": submit_ns,
                    "round_trip_ns": round_trip_ns,
                    "ready_to_receive_ns": ready_to_receive_ns,
                    "median_submit_ns": statistics.median(submit_ns),
                    "median_round_trip_ns": statistics.median(round_trip_ns),
                    "median_ready_to_receive_ns": statistics.median(
                        ready_to_receive_ns),
                })
    finally:
        executor.rpc_broadcast_mq.enqueue(("probe_stop", (), {}, 0))
        process.join(timeout=10)
        executor.io_thread_pool.shutdown(wait=True)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            raise RuntimeError("synthetic worker did not stop")
        if process.exitcode != 0:
            raise RuntimeError(f"synthetic worker exited {process.exitcode}")
    source_root = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    result = {
        "boundary": "vllm_0_11_multiproc_executor_rpc_synthetic_no_model_no_gpu",
        "vllm_version": "0.11.0",
        "cpu": args.cpu,
        "warmup_rounds_per_cell": args.warmup,
        "measured_rounds_per_cell": args.samples,
        "counts": args.counts,
        "phases": args.phases,
        "prompt_tokens_per_request": PROMPT_TOKENS,
        "source_sha256": {
            "core.py": _digest(source_root / "v1/engine/core.py"),
            "multiproc_executor.py": _digest(
                source_root / "v1/executor/multiproc_executor.py"),
            "shm_broadcast.py": _digest(
                source_root / "distributed/device_communicators/shm_broadcast.py"),
            "runner.py": _digest(Path(__file__).resolve()),
            "model_config.json": _digest(args.model_config),
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
