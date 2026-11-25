#!/usr/bin/env python3
"""Validate three frozen resident Split-K tile captures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


COUNTS = (64, 256, 1024, 4096)
TILE_K_VALUES = (384, 256)
CELLS = tuple(
    (tile_k, count) for tile_k in TILE_K_VALUES for count in COUNTS)
FIELDS = (
    "iterations", "block_dim", "tile_m", "tile_n", "tile_k",
    "max_cycles", "max_ns", "worker_cycles", "validation",
)
SYSTEM_COUNTER_TICK_NS = 20.0
STABILITY_LIMIT = 0.10
AFFINE_ERROR_LIMIT = 0.05


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path, expected_order):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"CSV schema drifted for {path}")
        rows = list(reader)
    row_order = tuple(
        (int(row["tile_k"]), int(row["iterations"])) for row in rows)
    if row_order != expected_order:
        raise ValueError(f"cell order drifted for {path}")
    parsed = {}
    for row in rows:
        count = int(row["iterations"])
        tile_k = int(row["tile_k"])
        cycles = tuple(int(value) for value in row["worker_cycles"].split(";"))
        maximum = int(row["max_cycles"])
        if (
            int(row["block_dim"]) != 24
            or int(row["tile_m"]) != 128
            or int(row["tile_n"]) != 128
            or tile_k not in TILE_K_VALUES
            or row["validation"] != "exact"
            or len(cycles) != 24
            or any(value <= 0 for value in cycles)
            or max(cycles) != maximum
            or float(row["max_ns"]) != maximum * SYSTEM_COUNTER_TICK_NS
        ):
            raise ValueError(f"cell contract drifted for {path}: {count}")
        parsed[(tile_k, count)] = maximum
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    if len(args.run) != 3:
        raise ValueError("exactly three independent --run files are required")
    orders = (CELLS, tuple(reversed(CELLS)), CELLS)
    runs = [_load(path, order) for path, order in zip(args.run, orders)]

    count_spans = {}
    slopes_by_tile_k = {}
    slope_spans = {}
    affine_errors = []
    for tile_k in TILE_K_VALUES:
        count_spans[str(tile_k)] = {}
        for count in COUNTS:
            values = [run[(tile_k, count)] for run in runs]
            relative_span = (
                (max(values) - min(values)) / statistics.median(values))
            if relative_span > STABILITY_LIMIT:
                raise ValueError(
                    f"K{tile_k} count {count} is unstable: {relative_span}")
            count_spans[str(tile_k)][str(count)] = relative_span
        slopes = [
            (run[(tile_k, 4096)] - run[(tile_k, 1024)])
            / (4096 - 1024)
            for run in runs
        ]
        if any(value <= 0.0 for value in slopes):
            raise ValueError(f"K{tile_k} steady tile service must be positive")
        median_slope = statistics.median(slopes)
        slope_span = (max(slopes) - min(slopes)) / median_slope
        if slope_span > STABILITY_LIMIT:
            raise ValueError(
                f"K{tile_k} steady service is unstable: {slope_span}")
        slopes_by_tile_k[str(tile_k)] = slopes
        slope_spans[str(tile_k)] = slope_span
        for run, slope in zip(runs, slopes):
            startup = run[(tile_k, 1024)] - slope * 1024
            for count in (64, 256):
                predicted = startup + count * slope
                affine_errors.append(
                    abs(predicted - run[(tile_k, count)])
                    / run[(tile_k, count)])
    if max(affine_errors) > AFFINE_ERROR_LIMIT:
        raise ValueError(
            f"held-out affine error exceeds gate: {max(affine_errors)}")

    source_root = Path(__file__).resolve().parent
    source_names = (
        "CMakeLists.txt", "README.md", "single_core_split_k_tile_kernel.cpp",
        "single_core_split_k_tile_main.cpp",
    )
    repository_source = Path(
        "microbench/ascend_910b/operators/gemm/single_core_split_k_tile")
    reference = {
        "schema": "ascend_910b_matmul_v3_split_k_resident_tile_reference_v2",
        "hardware_id": "ascend_910b",
        "software_revision": "CANN 8.5.1",
        "source_commit": "aff7fd42a0787e22eae435033543217626ed8b85",
        "mechanism": "matmul_v3_key65632_resident_base_tile_completion",
        "scope": {
            "dtype": "bfloat16",
            "tile_m": 128,
            "tile_n": 128,
            "tile_k_values": list(TILE_K_VALUES),
            "active_aic_workers": 24,
            "included": [
                "SetSingleShape", "A1_B1_tensor_binding", "MTE1_to_Cube",
                "M_to_FixPipe_handoff", "FP32_workspace_tile_store",
                "Matmul_End", "ping_pong_L0C_lifetime",
            ],
            "excluded": [
                "GM_to_L1", "paired_AIV_cast_store",
                "production_outer_loops", "complete_production_operator",
            ],
        },
        "source_artifacts": [
            {
                "path": str(repository_source / name),
                "sha256": _sha256(source_root / name),
            }
            for name in source_names
        ],
        "cann_source": {
            "path": "/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel_gm_to_l1.h",
            "sha256": "fd77d477b43422eb3f5da0c83f542c532e9d66ddf1c6480af49b65174878f223",
        },
        "capture": {
            "physical_device_id": 4,
            "container": args.container_name,
            "iteration_counts": list(COUNTS),
            "run_order": [
                [list(cell) for cell in order] for order in orders],
            "raw": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in args.run
            ],
        },
        "validation": {
            "count_relative_spans": count_spans,
            "steady_cycles_per_tile_by_run": slopes_by_tile_k,
            "steady_relative_span": slope_spans,
            "maximum_held_out_affine_error": max(affine_errors),
            "stability_limit": STABILITY_LIMIT,
            "affine_error_limit": AFFINE_ERROR_LIMIT,
            "status": "pass",
        },
        "result": {
            "unit": "ns_per_aic_resident_base_tile_completion",
            "system_counter_tick_ns": SYSTEM_COUNTER_TICK_NS,
            "median_cycles_per_tile_by_tile_k": {
                tile_k: statistics.median(slopes_by_tile_k[str(tile_k)])
                for tile_k in TILE_K_VALUES
            },
            "median_ns_per_tile_by_tile_k": {
                tile_k: (
                    statistics.median(slopes_by_tile_k[str(tile_k)])
                    * SYSTEM_COUNTER_TICK_NS)
                for tile_k in TILE_K_VALUES
            },
            "tensor_operations_per_tile_by_tile_k": {
                tile_k: 2 * 128 * 128 * tile_k
                for tile_k in TILE_K_VALUES
            },
        },
        "production_or_evaluation_duration_consumed": False,
        "forward_eligible": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(reference, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
