#!/usr/bin/env python3
"""Measure the torchair AclGraph eager-fallback Host dispatch per FX node.

The canonical Ascend serving engine compiles with ``enable_npugraph_ex=True``
(vllm-ascend ``AscendCompiler`` -> ``npugraph_ex_compile``), so the runnable
the engine calls is a torchair reduce-overhead AclGraph, not a plain
torch.compile artifact.  Steps whose batch exceeds the capture limit
(runtime_cudagraph_mode NONE) dispatch as ``AclGraph.__call__`` ->
``fallback_to_eager`` -> ``fx_run_eagerly``, i.e. the FX forward executes
node-by-node on the Host.  This runner measures exactly that mechanism:
it compiles a synthetic opaque-node FX chain through the same torchair
backend configuration, forces the static-capture-limit eager fallback, and
times the Host submit of the already-fallback call by node count.  Device
synchronization stays outside the timed interval; no Serving workload,
model, or production duration is consumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
import torch_npu.dynamo.torchair as torchair
from torch._ops import OpOverload, OpOverloadPacket


HARDWARE_ID = "ascend_910b"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", HARDWARE_ID)
CHAIN_LENGTHS = (1, 8, 64, 512)
FIT_CHAIN_LENGTHS = frozenset((1, 8, 64))
TRIALS = 12
WARMUP_REPEATS = 4
# One warm-up shape is captured, the second distinct shape trips the limit
# and latches fallback_to_eager; every subsequent call runs fx_run_eagerly.
CAPTURE_SIZE_LIMIT = 1
WARMUP_SHAPE = (256, 256)
MEASURED_SHAPE = (256, 257)
SCHEMA_VERSION = 1


@torch.library.custom_op(
    "midlayer_torchair_fallback_probe::opaque_boundary", mutates_args=())
def _opaque_boundary(source: torch.Tensor) -> torch.Tensor:
    """Keep a generic backend node opaque to the graph optimizers."""

    return source.clone()


@torch.library.register_fake(
    "midlayer_torchair_fallback_probe::opaque_boundary")
def _opaque_boundary_fake(source: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(source)


class _SyntheticTorchairFxChain(torch.nn.Module):
    """One fixed-size graph whose nodes cannot be fused together."""

    def __init__(self, node_count: int) -> None:
        super().__init__()
        self.node_count = node_count

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        value = source
        for _ in range(self.node_count):
            value = _opaque_boundary(value)
        return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--baseline-commit", type=str, required=True)
    arguments = parser.parse_args()
    if not arguments.container_name.strip():
        parser.error("--container-name must be nonempty")
    return arguments


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _append_json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires samples")
    return float(statistics.median(values))


def _linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("torchair fallback fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        raise RuntimeError("torchair fallback fit is invalid")
    return float(intercept), float(slope)


def _compiled_graph(node_count: int) -> torch.nn.Module:
    if node_count not in CHAIN_LENGTHS:
        raise ValueError("torchair fallback node count is not predeclared")
    torch.npu.set_compile_mode(jit_compile=False)
    config = torchair.CompilerConfig()
    config.mode = "reduce-overhead"
    config.debug.run_eagerly = True
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    config.debug.aclgraph.static_capture_size_limit = CAPTURE_SIZE_LIMIT
    backend = torchair.get_npu_backend(compiler_config=config)
    return torch.compile(
        _SyntheticTorchairFxChain(node_count).eval(),
        fullgraph=True,
        dynamic=False,
        backend=backend,
    )


def _dispatch_counts(
    compiled: torch.nn.Module,
    source: torch.Tensor,
) -> dict[str, int]:
    """Count Python dispatcher entries for one already-warm fallback call."""

    counts = {"op_overload_calls": 0, "op_overload_packet_calls": 0}
    original_overload_call = OpOverload.__call__
    original_packet_call = OpOverloadPacket.__call__

    def counted_overload(*args: object, **kwargs: object) -> object:
        counts["op_overload_calls"] += 1
        return original_overload_call(*args, **kwargs)

    def counted_packet(*args: object, **kwargs: object) -> object:
        counts["op_overload_packet_calls"] += 1
        return original_packet_call(*args, **kwargs)

    OpOverload.__call__ = counted_overload
    OpOverloadPacket.__call__ = counted_packet
    try:
        result = compiled(source)
    finally:
        OpOverload.__call__ = original_overload_call
        OpOverloadPacket.__call__ = original_packet_call
    if result.device != source.device:
        raise RuntimeError("torchair fallback count probe changed the device")
    if counts["op_overload_calls"] <= 0:
        raise RuntimeError("torchair fallback count probe observed no op call")
    return counts


def _measure_chain(
    node_count: int,
    warmup: torch.Tensor,
    measured: torch.Tensor,
    trials_path: Path,
) -> dict[str, object]:
    compiled = _compiled_graph(node_count)
    # First call warms up and captures the single allowed graph; a distinct
    # shape then trips static_capture_size_limit and latches eager fallback.
    for _ in range(WARMUP_REPEATS):
        compiled(warmup)
    torch.npu.synchronize()
    compiled(measured)
    torch.npu.synchronize()
    dispatch_counts = _dispatch_counts(compiled, measured)
    wall_values = []
    cpu_values = []
    for trial_index in range(TRIALS):
        cpu_start = time.thread_time_ns()
        wall_start = time.perf_counter_ns()
        result = compiled(measured)
        wall_end = time.perf_counter_ns()
        cpu_end = time.thread_time_ns()
        if result.device != measured.device:
            raise RuntimeError("torchair fallback probe changed the device")
        wall_ns = wall_end - wall_start
        cpu_ns = cpu_end - cpu_start
        if wall_ns <= 0 or cpu_ns <= 0:
            raise RuntimeError("torchair fallback timed interval must be positive")
        _append_json_line(trials_path, {
            "chain_length": node_count,
            "trial_index": trial_index,
            "host_call_wall_ns": wall_ns,
            "host_call_thread_cpu_ns": cpu_ns,
        })
        wall_values.append(float(wall_ns))
        cpu_values.append(float(cpu_ns))
    return {
        "chain_length": node_count,
        "host_call_thread_cpu_ns": {"median": _median(cpu_values)},
        "host_call_wall_ns": {"median": _median(wall_values)},
        "sample_count": TRIALS,
        "fallback_dispatch_counts": dispatch_counts,
    }


def main() -> None:
    arguments = _arguments()
    if isinstance(arguments.device_id, bool) or arguments.device_id < 0:
        raise ValueError("--device-id must be a non-negative integer")
    if (len(arguments.baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in arguments.baseline_commit)):
        raise ValueError("--baseline-commit must be a lowercase full Git SHA")
    output_dir = _validate_output_dir(arguments.output_dir)
    output_dir.mkdir(parents=True)
    trials_path = output_dir / "trials.jsonl"
    torch.npu.set_device(arguments.device_id)
    warmup = torch.empty(WARMUP_SHAPE, dtype=torch.float32,
                         device=f"npu:{arguments.device_id}")
    measured = torch.empty(MEASURED_SHAPE, dtype=torch.float32,
                           device=f"npu:{arguments.device_id}")
    torch.npu.synchronize()
    chains = [
        _measure_chain(node_count, warmup, measured, trials_path)
        for node_count in CHAIN_LENGTHS
    ]
    fit_points = [
        (chain["chain_length"],
         chain["host_call_thread_cpu_ns"]["median"])
        for chain in chains
        if chain["chain_length"] in FIT_CHAIN_LENGTHS
    ]
    cpu_intercept, cpu_slope = _linear_fit(fit_points)
    wall_points = [
        (chain["chain_length"],
         chain["host_call_wall_ns"]["median"])
        for chain in chains
        if chain["chain_length"] in FIT_CHAIN_LENGTHS
    ]
    wall_intercept, wall_slope = _linear_fit(wall_points)
    holdout = chains[-1]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ascend_torchair_fallback_dispatch_microbenchmark",
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "torchair reduce-overhead AclGraph latched to fallback_to_eager "
            "via static_capture_size_limit; each call dispatches the FX "
            "forward node-by-node on the Host (fx_run_eagerly)"),
        "measurement_boundary": (
            "compiled graph call entry to return; device synchronization is "
            "outside the timed interval"),
        "measurement_semantics": {
            "warmup_shape": list(WARMUP_SHAPE),
            "measured_shape": list(MEASURED_SHAPE),
            "static_capture_size_limit": CAPTURE_SIZE_LIMIT,
            "backend_config": {
                "mode": "reduce-overhead",
                "run_eagerly": True,
                "jit_compile": False,
            },
        },
        "predeclared": {
            "chain_lengths": list(CHAIN_LENGTHS),
            "fit_chain_lengths": sorted(FIT_CHAIN_LENGTHS),
            "holdout_chain_length": CHAIN_LENGTHS[-1],
            "trials": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
        },
        "chains": chains,
        "fit": {
            "thread_cpu_intercept_ns": cpu_intercept,
            "thread_cpu_slope_ns_per_node": cpu_slope,
            "wall_intercept_ns": wall_intercept,
            "wall_slope_ns_per_node": wall_slope,
            "holdout_thread_cpu_measured_ns":
                holdout["host_call_thread_cpu_ns"]["median"],
            "holdout_thread_cpu_predicted_ns":
                cpu_intercept + cpu_slope * holdout["chain_length"],
            "holdout_thread_cpu_signed_error_percent": (
                (holdout["host_call_thread_cpu_ns"]["median"]
                 - (cpu_intercept + cpu_slope * holdout["chain_length"]))
                / (cpu_intercept + cpu_slope * holdout["chain_length"]) * 100.0),
            "holdout_wall_measured_ns":
                holdout["host_call_wall_ns"]["median"],
            "holdout_wall_predicted_ns":
                wall_intercept + wall_slope * holdout["chain_length"],
            "holdout_wall_signed_error_percent": (
                (holdout["host_call_wall_ns"]["median"]
                 - (wall_intercept + wall_slope * holdout["chain_length"]))
                / (wall_intercept + wall_slope * holdout["chain_length"]) * 100.0),
        },
        "environment": {
            "container": arguments.container_name,
            "device_id": arguments.device_id,
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "baseline_commit": arguments.baseline_commit,
        "completed_at": _utc_now(),
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps({
        "summary_path": str(summary_path),
        "thread_cpu_slope_ns_per_node": cpu_slope,
        "wall_slope_ns_per_node": wall_slope,
        "thread_cpu_intercept_ns": cpu_intercept,
    }))


if __name__ == "__main__":
    main()
