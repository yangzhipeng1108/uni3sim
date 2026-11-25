#!/usr/bin/env python3
"""Measure generic Ascend compiled-FX executor Host service by node count.

This is a mechanism microbenchmark.  It deliberately uses synthetic tensor
allocation/copy/view nodes and consumes no Serving workload, model, operator
duration, or profiler duration.  The measured boundary is from invocation of
an already-compiled graph to asynchronous return; device synchronization is
outside the timed interval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
import torch_npu
from torch._ops import OpOverload, OpOverloadPacket


HARDWARE_ID = "ascend_910b"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", HARDWARE_ID)
CHAIN_LENGTHS = (1, 8, 64, 512)
FIT_CHAIN_LENGTHS = frozenset((1, 8, 64))
TRIALS = 12
WARMUP_REPEATS = 4
SCHEMA_VERSION = 1


@torch.library.custom_op(
    "midlayer_compiled_fx_probe::opaque_boundary", mutates_args=())
def _opaque_boundary(source: torch.Tensor) -> torch.Tensor:
    """Keep a generic backend entry opaque to the compiled graph optimizer."""

    return source.clone()


@torch.library.register_fake("midlayer_compiled_fx_probe::opaque_boundary")
def _opaque_boundary_fake(source: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(source)


class _SyntheticCompiledFxChain(torch.nn.Module):
    """One fixed-size graph whose backend entries cannot be fused together."""

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
        raise RuntimeError("compiled-FX fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        raise RuntimeError("compiled-FX fit is invalid")
    return float(intercept), float(slope)


def _compiled_graph(node_count: int) -> torch.nn.Module:
    if node_count not in CHAIN_LENGTHS:
        raise ValueError("compiled-FX node count is not predeclared")
    return torch.compile(
        _SyntheticCompiledFxChain(node_count).eval(),
        fullgraph=True,
        dynamic=False,
    )


def _dispatch_counts(
    compiled: torch.nn.Module,
    source: torch.Tensor,
) -> dict[str, int]:
    """Count Python dispatcher entries for one already-warm compiled call.

    This is deliberately outside the timed boundary.  It establishes whether
    the synthetic one-node chain and the real generated FX graph use the same
    OpOverload/OpOverloadPacket cardinality before a backend-entry descriptor
    is changed.
    """

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
        raise RuntimeError("compiled-FX count probe changed the device")
    if counts["op_overload_calls"] <= 0:
        raise RuntimeError("compiled-FX count probe observed no OpOverload")
    return counts


def _measure_chain(
    node_count: int,
    source: torch.Tensor,
    trials_path: Path,
) -> dict[str, object]:
    compiled = _compiled_graph(node_count)
    for _ in range(WARMUP_REPEATS):
        compiled(source)
    torch.npu.synchronize()
    dispatch_counts = _dispatch_counts(compiled, source)
    wall_values = []
    cpu_values = []
    for trial_index in range(TRIALS):
        cpu_start = time.thread_time_ns()
        wall_start = time.perf_counter_ns()
        result = compiled(source)
        wall_end = time.perf_counter_ns()
        cpu_end = time.thread_time_ns()
        if result.device != source.device:
            raise RuntimeError("compiled-FX probe changed the device")
        wall_ns = wall_end - wall_start
        cpu_ns = cpu_end - cpu_start
        if wall_ns <= 0 or cpu_ns <= 0:
            raise RuntimeError("compiled-FX timed interval must be positive")
        _append_json_line(trials_path, {
            "chain_length": node_count,
            "trial_index": trial_index,
            "host_call_wall_ns": wall_ns,
            "host_call_thread_cpu_ns": cpu_ns,
        })
        wall_values.append(float(wall_ns))
        cpu_values.append(float(cpu_ns))
        torch.npu.synchronize()
    return {
        "chain_length": node_count,
        "dispatch_counts_per_compiled_call": dispatch_counts,
        "sample_count": TRIALS,
        "host_call_wall_ns": {"median": _median(wall_values)},
        "host_call_thread_cpu_ns": {"median": _median(cpu_values)},
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
    device = f"npu:{arguments.device_id}"
    source = torch.ones(256, dtype=torch.float32, device=device)
    summaries = [
        _measure_chain(node_count, source, trials_path)
        for node_count in CHAIN_LENGTHS
    ]
    by_count = {summary["chain_length"]: summary for summary in summaries}
    wall_fit = _linear_fit([
        (node_count, by_count[node_count]["host_call_wall_ns"]["median"])
        for node_count in CHAIN_LENGTHS if node_count in FIT_CHAIN_LENGTHS
    ])
    cpu_fit = _linear_fit([
        (node_count, by_count[node_count]["host_call_thread_cpu_ns"]["median"])
        for node_count in CHAIN_LENGTHS if node_count in FIT_CHAIN_LENGTHS
    ])
    holdout = by_count[512]
    wall_prediction = wall_fit[0] + wall_fit[1] * 512
    cpu_prediction = cpu_fit[0] + cpu_fit[1] * 512
    summary = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "synthetic precompiled FX executor entry with one opaque generic "
            "normal-output backend node per count, asynchronous return"),
        "measurement_boundary": (
            "compiled graph call entry to return; device synchronization is "
            "outside the timed interval"),
        "source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "baseline_commit": arguments.baseline_commit,
        },
        "environment": {
            "container": arguments.container_name,
            "device_id": arguments.device_id,
            "host": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "predeclared": {
            "chain_lengths": CHAIN_LENGTHS,
            "fit_chain_lengths": tuple(sorted(FIT_CHAIN_LENGTHS)),
            "holdout_chain_length": 512,
            "trials": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
        },
        "chains": summaries,
        "fit": {
            "wall_intercept_ns": wall_fit[0],
            "wall_slope_ns_per_node": wall_fit[1],
            "thread_cpu_intercept_ns": cpu_fit[0],
            "thread_cpu_slope_ns_per_node": cpu_fit[1],
            "holdout_wall_measured_ns": holdout["host_call_wall_ns"]["median"],
            "holdout_wall_predicted_ns": wall_prediction,
            "holdout_wall_signed_error_percent": (
                100.0 * (wall_prediction - holdout["host_call_wall_ns"]["median"])
                / holdout["host_call_wall_ns"]["median"]),
            "holdout_thread_cpu_measured_ns": (
                holdout["host_call_thread_cpu_ns"]["median"]),
            "holdout_thread_cpu_predicted_ns": cpu_prediction,
            "holdout_thread_cpu_signed_error_percent": (
                100.0 * (cpu_prediction
                         - holdout["host_call_thread_cpu_ns"]["median"])
                / holdout["host_call_thread_cpu_ns"]["median"]),
        },
        "forward_eligibility": (
            "diagnostic_only_until a source-static compiled-FX node descriptor "
            "is added and its count is independently validated"),
        "completed_at": _utc_now(),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", {
        "schema_version": 1,
        "kind": "ascend_compiled_fx_executor_microbenchmark",
        "summary_sha256": _sha256(output_dir / "summary.json"),
        "trials_sha256": _sha256(trials_path),
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
