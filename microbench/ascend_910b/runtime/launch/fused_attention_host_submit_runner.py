#!/usr/bin/env python3
"""Measure paged-TND Ascend FIA Host preparation and asynchronous submission.

The probe uses tiny tensors but the same TND page-attention argument family
as the Ascend vLLM FIA path: KV cache, block table, causal mask, and per-request
sequence lengths.  It measures only the host interval from the warm-cache
Torch-NPU FIA call to return; every sample synchronizes *after* that interval.
It is therefore an independent runtime-mechanism microbenchmark, not a
Serving or kernel-duration benchmark.
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


HARDWARE_ID = "ascend_910b"
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", HARDWARE_ID)
CHAIN_LENGTHS = (1, 8, 64, 512)
FIT_CHAIN_LENGTHS = frozenset((1, 8, 64))
BATCH_SIZES = (1, 8, 32)
FIT_BATCH_SIZES = frozenset((1, 8))
QUERY_TOKEN_COUNTS = (1, 32, 128, 512)
QUERY_TOKEN_SWEEP_BATCH_SIZE = 1
SOURCE_WRAPPER_BATCH_SIZES = (1, 8, 32)
TRIALS = 12
WARMUP_REPEATS = 4
SCHEMA_VERSION = 3


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


def _linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    count = len(points)
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    xx_sum = sum(point[0] * point[0] for point in points)
    xy_sum = sum(point[0] * point[1] for point in points)
    denominator = count * xx_sum - x_sum * x_sum
    if denominator <= 0:
        raise RuntimeError("FIA host-submit fit is singular")
    slope = (count * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / count
    if not math.isfinite(intercept) or not math.isfinite(slope) or slope <= 0:
        raise RuntimeError("FIA host-submit fit is invalid")
    return float(intercept), float(slope)


def _fused_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    attention_mask: torch.Tensor,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
) -> torch.Tensor:
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        block_table=block_table,
        atten_mask=attention_mask,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_heads=2,
        num_key_value_heads=2,
        input_layout="TND",
        block_size=128,
        scale=0.125,
        sparse_mode=3,
    )
    return output


def _source_static_fia_wrapper(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    attention_mask: torch.Tensor,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
    output: torch.Tensor,
) -> torch.Tensor:
    """Generic structural analogue of the non-Graph TND FIA source path.

    It retains cache flattening, query narrowing, the FIA API boundary, output
    reshape, and preallocated output write.  Tensor dimensions are synthetic;
    this is an independent runtime-control benchmark, not a Serving replay.
    """

    num_blocks, block_size, _, _ = key_cache.shape
    key = key_cache.view(num_blocks, block_size, -1)
    value = value_cache.view(num_blocks, block_size, -1)
    num_tokens = actual_seq_lengths[-1]
    attention_output = _fused_attention(
        query[:num_tokens],
        key,
        value,
        block_table,
        attention_mask,
        actual_seq_lengths,
        actual_seq_lengths_kv,
    )
    output[:num_tokens] = attention_output.view(num_tokens, 2, 64)
    return output


def _measure_chain(
    chain_length: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    attention_mask: torch.Tensor,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
    trials_path: Path,
    sample_metadata: dict[str, object],
) -> dict[str, object]:
    for _ in range(WARMUP_REPEATS):
        for _ in range(chain_length):
            _fused_attention(
                query,
                key,
                value,
                block_table,
                attention_mask,
                actual_seq_lengths,
                actual_seq_lengths_kv,
            )
    torch.npu.synchronize()
    wall_values: list[float] = []
    cpu_values: list[float] = []
    for trial_index in range(TRIALS):
        cpu_start = time.thread_time_ns()
        wall_start = time.perf_counter_ns()
        result = query
        for _ in range(chain_length):
            result = _fused_attention(
                query,
                key,
                value,
                block_table,
                attention_mask,
                actual_seq_lengths,
                actual_seq_lengths_kv,
            )
        wall_end = time.perf_counter_ns()
        cpu_end = time.thread_time_ns()
        if result.device != query.device or result.shape != query.shape:
            raise RuntimeError("FIA probe returned an unexpected output")
        wall_ns = wall_end - wall_start
        cpu_ns = cpu_end - cpu_start
        if wall_ns <= 0 or cpu_ns <= 0:
            raise RuntimeError("FIA host-submit interval must be positive")
        _append_json_line(trials_path, {
            **sample_metadata,
            "chain_length": chain_length,
            "trial_index": trial_index,
            "host_call_wall_ns": wall_ns,
            "host_call_thread_cpu_ns": cpu_ns,
        })
        wall_values.append(float(wall_ns))
        cpu_values.append(float(cpu_ns))
        torch.npu.synchronize()
    return {
        "chain_length": chain_length,
        "sample_count": TRIALS,
        "host_call_wall_ns": {"median": float(statistics.median(wall_values))},
        "host_call_thread_cpu_ns": {
            "median": float(statistics.median(cpu_values))},
    }


def _chain_fit(chains: list[dict[str, object]]) -> dict[str, float]:
    by_count = {summary["chain_length"]: summary for summary in chains}
    wall_fit = _linear_fit([
        (chain_length, by_count[chain_length]["host_call_wall_ns"]["median"])
        for chain_length in CHAIN_LENGTHS
        if chain_length in FIT_CHAIN_LENGTHS
    ])
    cpu_fit = _linear_fit([
        (chain_length,
         by_count[chain_length]["host_call_thread_cpu_ns"]["median"])
        for chain_length in CHAIN_LENGTHS
        if chain_length in FIT_CHAIN_LENGTHS
    ])
    holdout = by_count[512]
    holdout_wall = holdout["host_call_wall_ns"]["median"]
    holdout_cpu = holdout["host_call_thread_cpu_ns"]["median"]
    return {
        "wall_intercept_ns": wall_fit[0],
        "wall_slope_ns_per_call": wall_fit[1],
        "thread_cpu_intercept_ns": cpu_fit[0],
        "thread_cpu_slope_ns_per_call": cpu_fit[1],
        "holdout_wall_measured_ns": holdout_wall,
        "holdout_wall_predicted_ns": wall_fit[0] + wall_fit[1] * 512,
        "holdout_wall_signed_error_percent": (
            100.0 * ((wall_fit[0] + wall_fit[1] * 512)
                     / holdout_wall - 1.0)),
        "holdout_thread_cpu_measured_ns": holdout_cpu,
        "holdout_thread_cpu_predicted_ns": cpu_fit[0] + cpu_fit[1] * 512,
        "holdout_thread_cpu_signed_error_percent": (
            100.0 * ((cpu_fit[0] + cpu_fit[1] * 512)
                     / holdout_cpu - 1.0)),
    }


def _measure_source_wrapper_chain(
    chain_length: int,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    attention_mask: torch.Tensor,
    actual_seq_lengths: list[int],
    actual_seq_lengths_kv: list[int],
    output: torch.Tensor,
    trials_path: Path,
    sample_metadata: dict[str, object],
) -> dict[str, object]:
    for _ in range(WARMUP_REPEATS):
        for _ in range(chain_length):
            _source_static_fia_wrapper(
                query, key_cache, value_cache, block_table, attention_mask,
                actual_seq_lengths, actual_seq_lengths_kv, output)
    torch.npu.synchronize()
    wall_values: list[float] = []
    cpu_values: list[float] = []
    for trial_index in range(TRIALS):
        cpu_start = time.thread_time_ns()
        wall_start = time.perf_counter_ns()
        result = output
        for _ in range(chain_length):
            result = _source_static_fia_wrapper(
                query, key_cache, value_cache, block_table, attention_mask,
                actual_seq_lengths, actual_seq_lengths_kv, output)
        wall_end = time.perf_counter_ns()
        cpu_end = time.thread_time_ns()
        if result is not output:
            raise RuntimeError("FIA source wrapper must preserve output storage")
        wall_ns = wall_end - wall_start
        cpu_ns = cpu_end - cpu_start
        if wall_ns <= 0 or cpu_ns <= 0:
            raise RuntimeError("FIA source-wrapper interval must be positive")
        _append_json_line(trials_path, {
            **sample_metadata,
            "chain_length": chain_length,
            "trial_index": trial_index,
            "host_call_wall_ns": wall_ns,
            "host_call_thread_cpu_ns": cpu_ns,
        })
        wall_values.append(float(wall_ns))
        cpu_values.append(float(cpu_ns))
        torch.npu.synchronize()
    return {
        "chain_length": chain_length,
        "sample_count": TRIALS,
        "host_call_wall_ns": {"median": float(statistics.median(wall_values))},
        "host_call_thread_cpu_ns": {
            "median": float(statistics.median(cpu_values))},
    }


def _source_wrapper_batch_sweep(
    device: str,
    attention_mask: torch.Tensor,
    trials_path: Path,
) -> list[dict[str, object]]:
    measurements = []
    for batch_size in SOURCE_WRAPPER_BATCH_SIZES:
        query = torch.randn(
            (batch_size, 2, 64), dtype=torch.float16, device=device)
        key_cache = torch.randn(
            (batch_size, 128, 2, 64), dtype=torch.float16, device=device)
        value_cache = torch.randn_like(key_cache)
        block_table = torch.arange(
            batch_size, dtype=torch.int32, device=device).view(batch_size, 1)
        actual_seq_lengths = list(range(1, batch_size + 1))
        actual_seq_lengths_kv = [128] * batch_size
        output = torch.empty_like(query)
        chains = [
            _measure_source_wrapper_chain(
                chain_length,
                query,
                key_cache,
                value_cache,
                block_table,
                attention_mask,
                actual_seq_lengths,
                actual_seq_lengths_kv,
                output,
                trials_path,
                {
                    "batch_size": batch_size,
                    "sweep": "source_static_wrapper_batch_size",
                },
            )
            for chain_length in CHAIN_LENGTHS
        ]
        measurements.append({
            "batch_size": batch_size,
            "query_shape": tuple(query.shape),
            "key_value_cache_shape": tuple(key_cache.shape),
            "block_table_shape": tuple(block_table.shape),
            "chains": chains,
            "chain_fit": _chain_fit(chains),
        })
    return measurements


def _query_token_sweep(
    device: str,
    attention_mask: torch.Tensor,
    trials_path: Path,
) -> list[dict[str, object]]:
    """Measure paged-query metadata scaling without a Serving workload."""

    measurements = []
    for query_tokens in QUERY_TOKEN_COUNTS:
        block_count = math.ceil(query_tokens / 128)
        query = torch.randn(
            (query_tokens, 2, 64), dtype=torch.float16, device=device)
        key = torch.randn(
            (block_count, 2, 128, 64), dtype=torch.float16, device=device)
        block_table = torch.arange(
            block_count, dtype=torch.int32, device=device).view(1, block_count)
        chains = [
            _measure_chain(
                chain_length,
                query,
                key,
                key,
                block_table,
                attention_mask,
                [query_tokens],
                [query_tokens],
                trials_path,
                {
                    "batch_size": QUERY_TOKEN_SWEEP_BATCH_SIZE,
                    "query_tokens": query_tokens,
                    "sweep": "query_tokens",
                },
            )
            for chain_length in CHAIN_LENGTHS
        ]
        measurements.append({
            "batch_size": QUERY_TOKEN_SWEEP_BATCH_SIZE,
            "query_tokens": query_tokens,
            "query_shape": tuple(query.shape),
            "key_value_shape": tuple(key.shape),
            "block_table_shape": tuple(block_table.shape),
            "chains": chains,
            "chain_fit": _chain_fit(chains),
        })
    return measurements


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
    attention_mask = torch.ones((2048, 2048), dtype=torch.bool, device=device)
    batch_measurements = []
    for batch_size in BATCH_SIZES:
        query = torch.randn(
            (batch_size, 2, 64), dtype=torch.float16, device=device)
        key = torch.randn(
            (batch_size, 2, 128, 64), dtype=torch.float16, device=device)
        block_table = torch.arange(
            batch_size, dtype=torch.int32, device=device).view(batch_size, 1)
        actual_seq_lengths = list(range(1, batch_size + 1))
        actual_seq_lengths_kv = [16] * batch_size
        chains = [
            _measure_chain(
                chain_length,
                query,
                key,
                key,
                block_table,
                attention_mask,
                actual_seq_lengths,
                actual_seq_lengths_kv,
                trials_path,
                {
                    "batch_size": batch_size,
                    "query_tokens": batch_size,
                    "sweep": "batch_size",
                },
            )
            for chain_length in CHAIN_LENGTHS
        ]
        batch_measurements.append({
            "batch_size": batch_size,
            "query_shape": tuple(query.shape),
            "key_value_shape": tuple(key.shape),
            "block_table_shape": tuple(block_table.shape),
            "chains": chains,
            "chain_fit": _chain_fit(chains),
        })
    query_token_measurements = _query_token_sweep(
        device, attention_mask, trials_path)
    source_wrapper_measurements = _source_wrapper_batch_sweep(
        device, attention_mask, trials_path)
    measurement_by_batch = {
        measurement["batch_size"]: measurement
        for measurement in batch_measurements
    }
    request_wall_fit = _linear_fit([
        (batch_size, measurement_by_batch[batch_size]["chain_fit"]
         ["wall_slope_ns_per_call"])
        for batch_size in BATCH_SIZES
        if batch_size in FIT_BATCH_SIZES
    ])
    request_cpu_fit = _linear_fit([
        (batch_size, measurement_by_batch[batch_size]["chain_fit"]
         ["thread_cpu_slope_ns_per_call"])
        for batch_size in BATCH_SIZES
        if batch_size in FIT_BATCH_SIZES
    ])
    batch_holdout = measurement_by_batch[32]["chain_fit"]
    batch_holdout_wall = batch_holdout["wall_slope_ns_per_call"]
    batch_holdout_cpu = batch_holdout["thread_cpu_slope_ns_per_call"]
    source_wrapper_by_batch = {
        measurement["batch_size"]: measurement
        for measurement in source_wrapper_measurements
    }
    source_wrapper_wall_fit = _linear_fit([
        (batch_size, source_wrapper_by_batch[batch_size]["chain_fit"]
         ["wall_slope_ns_per_call"])
        for batch_size in SOURCE_WRAPPER_BATCH_SIZES
        if batch_size in FIT_BATCH_SIZES
    ])
    source_wrapper_cpu_fit = _linear_fit([
        (batch_size, source_wrapper_by_batch[batch_size]["chain_fit"]
         ["thread_cpu_slope_ns_per_call"])
        for batch_size in SOURCE_WRAPPER_BATCH_SIZES
        if batch_size in FIT_BATCH_SIZES
    ])
    source_wrapper_holdout = source_wrapper_by_batch[32]["chain_fit"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": (
            "Torch-NPU fused-infer-attention descriptor preparation and "
            "asynchronous submit with tiny synthetic paged TND tensors"),
        "measurement_boundary": (
            "torch_npu.npu_fused_infer_attention_score entry to return; "
            "device synchronization occurs only after the timed interval"),
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
            "batch_sizes": BATCH_SIZES,
            "fit_batch_sizes": tuple(sorted(FIT_BATCH_SIZES)),
            "holdout_batch_size": 32,
            "query_token_counts": QUERY_TOKEN_COUNTS,
            "query_token_sweep_batch_size": QUERY_TOKEN_SWEEP_BATCH_SIZE,
            "source_wrapper_batch_sizes": SOURCE_WRAPPER_BATCH_SIZES,
            "trials": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "input_layout": "TND",
            "attention_mask_shape": tuple(attention_mask.shape),
        },
        "batch_measurements": batch_measurements,
        "query_token_measurements": query_token_measurements,
        "source_static_wrapper_measurements": source_wrapper_measurements,
        "request_metadata_fit": {
            "wall_intercept_ns": request_wall_fit[0],
            "wall_slope_ns_per_request": request_wall_fit[1],
            "thread_cpu_intercept_ns": request_cpu_fit[0],
            "thread_cpu_slope_ns_per_request": request_cpu_fit[1],
            "holdout_wall_measured_ns_per_call": batch_holdout_wall,
            "holdout_wall_predicted_ns_per_call": (
                request_wall_fit[0] + request_wall_fit[1] * 32),
            "holdout_wall_signed_error_percent": (
                100.0 * ((request_wall_fit[0] + request_wall_fit[1] * 32)
                         / batch_holdout_wall - 1.0)),
            "holdout_thread_cpu_measured_ns_per_call": batch_holdout_cpu,
            "holdout_thread_cpu_predicted_ns_per_call": (
                request_cpu_fit[0] + request_cpu_fit[1] * 32),
            "holdout_thread_cpu_signed_error_percent": (
                100.0 * ((request_cpu_fit[0] + request_cpu_fit[1] * 32)
                         / batch_holdout_cpu - 1.0)),
        },
        "source_static_wrapper_request_fit": {
            "wall_intercept_ns": source_wrapper_wall_fit[0],
            "wall_slope_ns_per_request": source_wrapper_wall_fit[1],
            "thread_cpu_intercept_ns": source_wrapper_cpu_fit[0],
            "thread_cpu_slope_ns_per_request": source_wrapper_cpu_fit[1],
            "holdout_wall_measured_ns_per_call": (
                source_wrapper_holdout["wall_slope_ns_per_call"]),
            "holdout_wall_predicted_ns_per_call": (
                source_wrapper_wall_fit[0] + source_wrapper_wall_fit[1] * 32),
            "holdout_wall_signed_error_percent": (
                100.0 * ((source_wrapper_wall_fit[0]
                          + source_wrapper_wall_fit[1] * 32)
                         / source_wrapper_holdout["wall_slope_ns_per_call"]
                         - 1.0)),
            "holdout_thread_cpu_measured_ns_per_call": (
                source_wrapper_holdout["thread_cpu_slope_ns_per_call"]),
            "holdout_thread_cpu_predicted_ns_per_call": (
                source_wrapper_cpu_fit[0] + source_wrapper_cpu_fit[1] * 32),
            "holdout_thread_cpu_signed_error_percent": (
                100.0 * ((source_wrapper_cpu_fit[0]
                          + source_wrapper_cpu_fit[1] * 32)
                         / source_wrapper_holdout["thread_cpu_slope_ns_per_call"]
                         - 1.0)),
        },
        "forward_eligibility": (
            "diagnostic_only_until the source-static FIA call descriptor is "
            "independently validated"),
        "completed_at": _utc_now(),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", {
        "schema_version": 1,
        "kind": "ascend_fused_attention_host_submit_microbenchmark",
        "summary_sha256": _sha256(output_dir / "summary.json"),
        "trials_sha256": _sha256(trials_path),
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
