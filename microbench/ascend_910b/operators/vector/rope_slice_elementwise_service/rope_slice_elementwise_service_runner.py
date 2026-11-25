#!/usr/bin/env python3
"""Measure the per-launch device completion of the RoPE slice elementwise
services without a profiler.

The 910B llama-3.1 rotary path is not fused: ``rotate_half`` slices the head
dimension into two halves, and the cos/sin application lowers to independent
slice-copy, multiply and add launches on small BF16 tensors.  This probe
measures the independent mechanism law of those three single-op services on
dedicated tensors: a fixed per-program completion term plus a steady
per-element rate.  It never replays the production rotary chain, never touches
the serving dataset, and consumes no production or evaluation duration.

The slice probe reproduces the strided source geometry of the production
slice (a contiguous destination filled from ``base[:, :inner]`` of a
``[rows, 2 * inner]`` operand) rather than a contiguous memcpy, because the
stride is what selects the kernel.  ``inner`` is fixed at the 64-element half
head dimension and ``rows`` sweeps the element domain.

Timing uses ``torch.npu.Event(enable_timing=True)`` around a captured chain of
CHAIN_LENGTH independent identical ops replayed through an NPU graph, so the
measured span excludes host submission pacing.  An eager back-to-back chain
and a single-op event pair are recorded alongside as cross-checks.
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
import re
import statistics
import sys
import time
import traceback

import torch
import torch_npu


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")

# The production rotary element domain is rows * 64, where rows is the head
# count times the token count.  The 910B dense cells populate 8 and 32 heads,
# so the observed domain spans 128 elements (two rows) through the largest
# multi-token blocks.  The sweep keeps that domain and adds headroom points so
# the steady per-element rate is separable from the fixed completion term.
HALF_HEAD_DIMENSION = 64
ELEMENT_COUNTS = (128, 256, 512, 1024, 1536, 2048, 4096, 8192, 12288,
                  16384, 65536, 131072)
OP_NAMES = ("slice_copy", "mul", "add")
MODES = ("graph", "eager", "single")
CHAIN_LENGTH = 256
TRIALS = 30
WARMUP_REPEATS = 5
SCHEMA_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _cann_identity():
    ascend_home = Path(os.environ["ASCEND_HOME_PATH"])
    version_path = ascend_home / "opp" / "version.info"
    return {
        "ascend_home_path": str(ascend_home),
        "opp_version_path": str(version_path),
        "opp_version_text": version_path.read_text(encoding="utf-8"),
        "opp_version_sha256": _sha256(version_path),
    }


def _device_identity(device_id):
    properties = torch.npu.get_device_properties(device_id)
    return {
        "device_id": device_id,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "cube_core_count": int(properties.cube_core_num),
        "vector_core_count": int(properties.vector_core_num),
        "properties_repr": repr(properties),
    }


def _validate_output_dir(path):
    resolved = path.resolve()
    owner_size = len(OUTPUT_OWNER_PARTS)
    owner_offsets = tuple(
        index
        for index in range(len(resolved.parts) - owner_size + 1)
        if resolved.parts[index:index + owner_size] == OUTPUT_OWNER_PARTS
    )
    if len(owner_offsets) != 1:
        raise ValueError(
            "--output-dir must be under exactly one "
            "profiler_output/microbench/ascend_910b owner")
    if len(resolved.parts) <= owner_offsets[0] + owner_size:
        raise ValueError("--output-dir must include a run-specific directory")
    if resolved.exists():
        raise FileExistsError("--output-dir must not already exist")
    return resolved


def _rows_for(elements):
    if elements % HALF_HEAD_DIMENSION != 0:
        raise ValueError("element count must be a half-head-dimension multiple")
    return elements // HALF_HEAD_DIMENSION


def _slice_copy_tensors(elements, device):
    rows = _rows_for(elements)
    base = torch.arange(
        rows * 2 * HALF_HEAD_DIMENSION, dtype=torch.float32, device=device
    ).reshape(rows, 2 * HALF_HEAD_DIMENSION).to(torch.bfloat16)
    source = base[:, :HALF_HEAD_DIMENSION]
    destination = torch.empty(
        (rows, HALF_HEAD_DIMENSION), dtype=torch.bfloat16, device=device)
    expected = source.contiguous()
    return base, source, destination, expected


def _binary_tensors(elements, device):
    rows = _rows_for(elements)
    shape = (rows, HALF_HEAD_DIMENSION)
    left = torch.arange(
        elements, dtype=torch.float32, device=device
    ).reshape(shape).to(torch.bfloat16)
    right = torch.arange(
        elements, 0, -1, dtype=torch.float32, device=device
    ).reshape(shape).to(torch.bfloat16)
    destination = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return left, right, destination


def _run_slice_copy_chain(source, destination, chain_length):
    for _ in range(chain_length):
        destination.copy_(source)


def _run_mul_chain(left, right, destination, chain_length):
    for _ in range(chain_length):
        torch.mul(left, right, out=destination)


def _run_add_chain(left, right, destination, chain_length):
    for _ in range(chain_length):
        torch.add(left, right, out=destination)


def _capture_graph(stream, pool, run_chain):
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=pool, stream=stream):
        run_chain(CHAIN_LENGTH)
    return graph


class _Cell:
    def __init__(self, op_name, elements, device):
        self.op_name = op_name
        self.elements = elements
        if op_name == "slice_copy":
            base, source, destination, expected = _slice_copy_tensors(
                elements, device)
            self.operands = (base, source)
            self.run_chain = (
                lambda chain_length: _run_slice_copy_chain(
                    source, destination, chain_length))
        elif op_name in ("mul", "add"):
            left, right, destination = _binary_tensors(elements, device)
            self.operands = (left, right)
            if op_name == "mul":
                expected = left * right
                self.run_chain = (
                    lambda chain_length: _run_mul_chain(
                        left, right, destination, chain_length))
            else:
                expected = left + right
                self.run_chain = (
                    lambda chain_length: _run_add_chain(
                        left, right, destination, chain_length))
        else:
            raise AssertionError(f"unknown probe op: {op_name!r}")
        self.destination = destination
        self.expected = expected
        self.graph = None

    def check(self):
        self.run_chain(1)
        torch.npu.synchronize()
        if not bool(torch.equal(self.destination, self.expected)):
            raise RuntimeError(
                f"probe op {self.op_name!r} at {self.elements} elements "
                "produced a wrong result")

    def capture(self, stream, pool):
        self.graph = _capture_graph(stream, pool, self.run_chain)

    def measure(self, stream, mode):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record(stream)
        if mode == "graph":
            if self.graph is None:
                raise RuntimeError("graph was not captured for this cell")
            self.graph.replay()
        elif mode == "eager":
            self.run_chain(CHAIN_LENGTH)
        elif mode == "single":
            self.run_chain(1)
        else:
            raise ValueError(f"unsupported mode {mode!r}")
        end_event.record(stream)
        end_event.synchronize()
        device_elapsed_ns = (
            float(start_event.elapsed_time(end_event)) * 1_000_000.0)
        op_count = 1 if mode == "single" else CHAIN_LENGTH
        if (not math.isfinite(device_elapsed_ns) or device_elapsed_ns <= 0
                or device_elapsed_ns / op_count <= 0):
            raise RuntimeError("timing sample contains a non-positive value")
        return {
            "op": self.op_name,
            "elements": self.elements,
            "mode": mode,
            "device_elapsed_ns": device_elapsed_ns,
            "per_op_ns": device_elapsed_ns / op_count,
        }


def _ordered_cells(trial_index, cells):
    rotation = trial_index % len(cells)
    rotated = cells[rotation:] + cells[:rotation]
    return rotated if trial_index % 2 == 0 else tuple(reversed(rotated))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if args.run_index < 1:
        parser.error("--run-index must be positive")
    if BASELINE_PATTERN.fullmatch(args.baseline_commit) is None:
        parser.error("--baseline-commit must be a full lowercase Git commit")
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    if len(args.expected_source_sha256) != 64:
        parser.error("--expected-source-sha256 must be a SHA256 digest")
    args.output_dir = _validate_output_dir(args.output_dir)
    return args


def main():
    args = _parse_args()
    source_path = Path(__file__).resolve()
    actual_source_sha256 = _sha256(source_path)
    if actual_source_sha256 != args.expected_source_sha256:
        raise RuntimeError(
            "runner source SHA256 differs from the audited snapshot")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "manifest.json"
    trials_path = args.output_dir / "samples.csv"
    summary_path = args.output_dir / "summary.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "run_index": args.run_index,
        "baseline_commit": args.baseline_commit,
        "source": {
            "path": str(source_path),
            "sha256": actual_source_sha256,
        },
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "container_name": args.container_name,
        "device": None,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "cann": None,
        },
        "contract": {
            "mechanism": (
                "independent single-op RoPE slice-copy (strided BF16 half "
                "head slice to contiguous destination), multiply and add "
                "device completion on dedicated probe tensors"),
            "ops": list(OP_NAMES),
            "half_head_dimension": HALF_HEAD_DIMENSION,
            "element_counts": list(ELEMENT_COUNTS),
            "modes": list(MODES),
            "chain_length": CHAIN_LENGTH,
            "trials_per_cell_per_mode": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "schedule": "paired_forward_reverse_rotations",
            "event_clock": "torch.npu.Event(enable_timing=True)",
            "graph_api": "torch.npu.graph(NPUGraph, pool, explicit_stream)",
            "production_operator_replayed": False,
            "production_or_evaluation_duration_consumed": False,
        },
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
        "outputs": {
            "samples": str(trials_path),
            "summary": str(summary_path),
        },
    }
    _write_json(manifest_path, manifest)
    try:
        manifest["environment"]["cann"] = _cann_identity()
        torch.npu.set_device(args.device_id)
        manifest["device"] = _device_identity(args.device_id)
        device = f"npu:{args.device_id}"
        stream = torch.npu.Stream(device=device)
        pool = torch.npu.graph_pool_handle()
        with torch.npu.stream(stream):
            cells = tuple(
                _Cell(op_name, elements, device)
                for op_name in OP_NAMES
                for elements in ELEMENT_COUNTS)
            for cell in cells:
                cell.check()
            for cell in cells:
                cell.capture(stream, pool)
            # Warm every cell's graph replay and eager chain once so the
            # first timed trial does not carry one-time graph or allocator
            # effects.
            for cell in cells:
                for mode in MODES:
                    for _ in range(WARMUP_REPEATS):
                        cell.measure(stream, mode)
            samples = []
            for trial in range(TRIALS):
                submit_started_ns = time.perf_counter_ns()
                for cell in _ordered_cells(trial, cells):
                    for mode in MODES:
                        samples.append(cell.measure(stream, mode))
                submit_ended_ns = time.perf_counter_ns()
                if submit_ended_ns - submit_started_ns <= 0:
                    raise RuntimeError("host trial clock is non-monotonic")
            with trials_path.open("w", encoding="utf-8") as handle:
                handle.write(
                    "op,elements,mode,trial,device_elapsed_ns,per_op_ns\n")
                for index, sample in enumerate(samples):
                    trial = index // (len(cells) * len(MODES))
                    handle.write(
                        f"{sample['op']},{sample['elements']},"
                        f"{sample['mode']},{trial},"
                        f"{sample['device_elapsed_ns']:.3f},"
                        f"{sample['per_op_ns']:.6f}\n")
                handle.flush()
                os.fsync(handle.fileno())
        summary = {}
        for op_name in OP_NAMES:
            for elements in ELEMENT_COUNTS:
                for mode in MODES:
                    values = [
                        sample["per_op_ns"] for sample in samples
                        if sample["op"] == op_name
                        and sample["elements"] == elements
                        and sample["mode"] == mode]
                    if len(values) != TRIALS:
                        raise RuntimeError(
                            "sample count disagrees with the trial contract")
                    summary[f"{op_name}/{elements}/{mode}"] = {
                        "median_per_op_ns": statistics.median(values),
                        "min_per_op_ns": min(values),
                        "max_per_op_ns": max(values),
                    }
        _write_json(summary_path, summary)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
    except Exception:
        manifest["status"] = "failed"
        manifest["failure"] = traceback.format_exc()
        _write_json(manifest_path, manifest)
        raise
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "run_index": args.run_index,
        "samples": str(trials_path),
    }))


if __name__ == "__main__":
    main()
