"""Observe vLLM worker Host calls versus prior NPU completion in memory."""

import json
import os
from pathlib import Path
import time

import torch
import torch_npu  # noqa: F401
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


_ORIGINAL_EXECUTE = NPUModelRunner.execute_model
_OUTPUT_DIR = Path(os.environ["MID_WORKER_OVERLAP_OUTPUT_DIR"])
_SEQUENCE = 0
_LAST_EVENT = None
_GATE = None
_GATE_ENABLED = os.environ["MID_WORKER_OVERLAP_GATE"] == "1"
_GATE_NODES = int(os.environ["MID_WORKER_OVERLAP_GATE_NODES"])
_GATE_MODE = os.environ["MID_WORKER_OVERLAP_GATE_MODE"]
if _GATE_NODES <= 0 or _GATE_NODES > 4096:
    raise ValueError("gate node count must be in 1..4096")
if _GATE_MODE not in ("graph", "eager_add"):
    raise ValueError("gate mode must be graph or eager_add")


def _replay_gate():
    global _GATE
    if _GATE_MODE == "eager_add":
        with torch.inference_mode():
            if _GATE is None:
                source = torch.zeros(64 * 1024 * 1024,
                                     dtype=torch.float32, device="npu")
                destination = torch.empty_like(source)
                increment = torch.ones_like(source)
                _GATE = (source, destination, increment)
            source, destination, increment = _GATE
            for _ in range(_GATE_NODES):
                torch.add(source, increment, out=destination)
                source, destination = destination, source
        return
    model_stream = torch.npu.current_stream()
    if _GATE is None:
        gate_stream = torch.npu.Stream()
        source = torch.zeros(8 * 1024 * 1024, dtype=torch.float32,
                             device="npu")
        destination = torch.empty_like(source)
        increment = torch.ones_like(source)
        graph = torch.npu.NPUGraph()
        pool = torch.npu.graph_pool_handle()
        with torch.npu.graph(graph, pool=pool, stream=gate_stream):
            for _ in range(_GATE_NODES):
                torch.add(source, increment, out=destination)
                source, destination = destination, source
        gate_stream.synchronize()
        _GATE = (graph, gate_stream, source, destination, increment)
    graph, gate_stream, _, _, _ = _GATE
    gate_stream.wait_stream(model_stream)
    with torch.npu.stream(gate_stream):
        graph.replay()
    model_stream.wait_stream(gate_stream)


def _traced_execute(self, *args, **kwargs):
    global _SEQUENCE, _LAST_EVENT
    runner_id = id(self)
    entry_ns = time.monotonic_ns()
    if _LAST_EVENT is not None:
        prior_device_complete_at_entry = bool(_LAST_EVENT.query())
    else:
        prior_device_complete_at_entry = None
    _SEQUENCE += 1
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (_OUTPUT_DIR / f"worker_{os.getpid()}.jsonl").open(
            "a", encoding="utf-8") as output:
        output.write(json.dumps({
            "schema_version": 1,
            "sequence": _SEQUENCE,
            "runner_id": runner_id,
            "host_entry_ns": entry_ns,
            "prior_device_complete_at_entry": prior_device_complete_at_entry,
        }, sort_keys=True) + "\n")
    result = _ORIGINAL_EXECUTE(self, *args, **kwargs)
    if _GATE_ENABLED:
        _replay_gate()
    device_end = torch.npu.Event(enable_timing=False)
    device_end.record(torch.npu.current_stream())
    _LAST_EVENT = device_end
    return result


NPUModelRunner.execute_model = _traced_execute
