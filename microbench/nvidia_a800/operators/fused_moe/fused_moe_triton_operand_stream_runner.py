#!/usr/bin/env python3
"""Measure one source-exact Triton grouped-MoE operand stream on A800.

The probe invokes only the grouped matrix kernel, uses a deterministic
synthetic routing distribution, and requires non-production K/N dimensions.
It never executes or consumes the duration of a complete production MoE op.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "profiler_output/microbench/nvidia_a800"
VLLM_SOURCE = (
    Path(os.environ["MIDLAYER_A800_VLLM_SOURCE"])
    / "model_executor/layers/fused_moe/fused_moe.py"
)
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
PREFERRED_DEVICES = (2, 6)
DEFAULT_EXPERTS = 8
DEFAULT_TOPK = 2
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
GROUP_SIZE_M = 8
PRODUCTION_SHAPES = (
    (1024, 4096),
    (2880, 2880),
    (2880, 5760),
    (6144, 32768),
)
SOURCE_SNIPPETS = (
    "num_pid_in_group = GROUP_SIZE_M * num_pid_n",
    "offs_token[:, None] // top_k * stride_am",
    "off_experts * stride_be",
    "accumulator += tl.dot(a, b)",
    "if MUL_ROUTED_WEIGHT:",
    "accumulator = accumulator * moe_weight[:, None]",
)


def _run(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(device_id, *, require_idle):
    result = _run([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,"
        "memory.used,clocks.current.sm,clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ])
    fields = tuple(value.strip() for value in result.stdout.split(","))
    if len(fields) != 9:
        raise ValueError("unexpected nvidia-smi device row")
    row = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "gpu_utilization_pct": int(fields[3]),
        "memory_utilization_pct": int(fields[4]),
        "memory_used_mib": int(fields[5]), "sm_clock_mhz": int(fields[6]),
        "memory_clock_mhz": int(fields[7]), "driver_version": fields[8],
    }
    if row["index"] != device_id or row["name"] != DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    applications = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]).stdout
    if (require_idle and any(
            line.strip() and line.split(",", 1)[0].strip() == row["uuid"]
            for line in applications.splitlines())):
        raise RuntimeError("target GPU has a compute process")
    if (require_idle
            and (row["gpu_utilization_pct"] != 0
                 or row["memory_utilization_pct"] != 0
                 or row["memory_used_mib"] > 8)):
        raise RuntimeError(f"target GPU is not idle: {row}")
    return row


def _audit_source():
    source = VLLM_SOURCE.read_text(encoding="utf-8")
    missing = tuple(fragment for fragment in SOURCE_SNIPPETS
                    if fragment not in source)
    if missing:
        raise ValueError(f"vLLM grouped-MoE source changed: {missing}")
    return _sha256(VLLM_SOURCE)


def _wait_idle(device_id):
    observations = []
    for _ in range(40):
        try:
            return _device(device_id, require_idle=True)
        except RuntimeError as error:
            observations.append(str(error))
            time.sleep(0.25)
    raise RuntimeError(
        f"target GPU did not become idle: {observations[-3:]}")


def _parse_m_tile_counts(text):
    values = tuple(int(value) for value in text.split(","))
    if not values or any(value < 0 for value in values):
        raise ValueError("m-tile counts must be a non-empty non-negative list")
    return values


def _parse_tokens_per_expert(text):
    values = tuple(int(value) for value in text.split(","))
    if not values or any(value < 0 for value in values):
        raise ValueError(
            "tokens per expert must be a non-empty non-negative list")
    return values


def _synthetic_counts(
        tokens, experts, topk, route_seed, m_tile_counts,
        tokens_per_expert):
    routed_rows = tokens * topk
    if tokens_per_expert is not None:
        if len(tokens_per_expert) != experts:
            raise ValueError(
                "tokens per expert must contain one entry per expert")
        if sum(tokens_per_expert) != routed_rows:
            raise ValueError(
                "tokens per expert must conserve the requested routed rows")
        if m_tile_counts is not None:
            actual_m_tiles = tuple(
                math.ceil(rows / BLOCK_M) if rows else 0
                for rows in tokens_per_expert)
            if actual_m_tiles != m_tile_counts:
                raise ValueError(
                    "tokens per expert does not match the declared M-tile "
                    "census")
        return tokens_per_expert
    if m_tile_counts is not None:
        if len(m_tile_counts) != experts:
            raise ValueError("m-tile counts must contain one entry per expert")
        lower_rows = sum(
            (m_tiles - 1) * BLOCK_M + 1
            for m_tiles in m_tile_counts if m_tiles)
        upper_rows = sum(m_tiles * BLOCK_M for m_tiles in m_tile_counts)
        if not lower_rows <= routed_rows <= upper_rows:
            raise ValueError(
                "m-tile counts cannot represent the requested routed rows")
        remaining_trim = upper_rows - routed_rows
        counts = []
        for m_tiles in m_tile_counts:
            if m_tiles == 0:
                counts.append(0)
                continue
            trim = min(remaining_trim, BLOCK_M - 1)
            counts.append(m_tiles * BLOCK_M - trim)
            remaining_trim -= trim
        if remaining_trim:
            raise AssertionError("m-tile count trim did not consume all rows")
        return tuple(counts)
    if experts % 2:
        raise ValueError("synthetic routing requires an even expert count")
    if route_seed is not None:
        generator = random.Random(route_seed)
        counts = [0] * experts
        for _ in range(routed_rows):
            counts[generator.randrange(experts)] += 1
        return tuple(counts)
    if routed_rows % experts:
        raise ValueError("balanced routes must divide evenly across experts")
    average = routed_rows // experts
    if average <= 1:
        raise ValueError(
            "synthetic balanced routing requires at least two rows per "
            "expert")
    counts = (
        (average + 1,) * (experts // 2)
        + (average - 1,) * (experts // 2))
    if sum(counts) != routed_rows:
        raise AssertionError("synthetic routing does not conserve rows")
    return counts


def _measure(
        tokens, experts, topk, source_topk, k, n, trials,
        mul_routed_weight, route_seed, m_tile_counts, tokens_per_expert,
        maximum_m_tiles):
    import torch
    import triton.language as tl
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_kernel)
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    from dataset_pipeline.kernel_dataset.nvidia_a800_event_capture import (
        build_nvidia_a800_moe_topk_ids)

    routed_rows = tokens * topk
    synthetic_counts = _synthetic_counts(
        tokens, experts, topk, route_seed, m_tile_counts, tokens_per_expert)
    topk_ids = torch.tensor(build_nvidia_a800_moe_topk_ids(
        token_counts=list(synthetic_counts),
        num_tokens=tokens,
        num_experts=experts,
        top_k=topk,
        context="A800 synthetic grouped-MoE routing",
    ), dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, post_padded = moe_align_block_size(
        topk_ids, BLOCK_M, experts, None)
    actual_post_padded = int(post_padded.item())
    active_m_tiles = actual_post_padded // BLOCK_M
    expected_active_m_tiles = sum(
        math.ceil(count / BLOCK_M) for count in synthetic_counts)
    if active_m_tiles != expected_active_m_tiles:
        raise AssertionError("synthetic routing has the wrong active M tiles")
    natural_maximum_m_tiles = math.ceil(sorted_ids.numel() / BLOCK_M)
    if maximum_m_tiles is None:
        maximum_m_tiles = natural_maximum_m_tiles
    if maximum_m_tiles < natural_maximum_m_tiles:
        raise ValueError("maximum M tiles cannot truncate the source workspace")
    if maximum_m_tiles != natural_maximum_m_tiles:
        expanded_sorted_ids = torch.empty(
            (maximum_m_tiles * BLOCK_M,), dtype=sorted_ids.dtype,
            device=sorted_ids.device)
        expanded_sorted_ids[:sorted_ids.numel()].copy_(sorted_ids)
        sorted_ids = expanded_sorted_ids

    source_rows = routed_rows // source_topk
    activation = torch.ones(
        (source_rows, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((experts, n, k), dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(
        (source_rows, source_topk, n), dtype=torch.bfloat16, device="cuda")
    routed_weights = torch.full(
        (source_rows, source_topk), 0.5,
        dtype=torch.bfloat16, device="cuda")
    config = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
    }

    def operation():
        invoke_fused_moe_kernel(
            activation, weight, output, None, None, None,
            routed_weights if mul_routed_weight else None,
            sorted_ids, expert_ids, post_padded, mul_routed_weight,
            source_topk, config,
            tl.bfloat16, False, False, False, False, False)

    for _ in range(2):
        operation()
        torch.cuda.synchronize()
    durations_ms = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        durations_ms.append(float(start.elapsed_time(stop)))
    expected_output = k * (0.5 if mul_routed_weight else 1.0)
    if float(output[0, 0, 0].item()) != expected_output:
        raise RuntimeError("grouped-MoE output validation failed")

    n_tiles = math.ceil(n / BLOCK_N)
    activation_bytes = routed_rows * k * 2 * n_tiles
    weight_bytes = active_m_tiles * n_tiles * k * BLOCK_N * 2
    logical_bytes = activation_bytes + weight_bytes
    median_ms = statistics.median(durations_ms)
    return {
        "tokens": tokens,
        "experts": experts,
        "topk": topk,
        "source_topk": source_topk,
        "source_rows": source_rows,
        "mul_routed_weight": mul_routed_weight,
        "route_seed": route_seed,
        "m_tile_counts": list(m_tile_counts) if m_tile_counts else None,
        "tokens_per_expert_source": (
            "exact" if tokens_per_expert is not None else "synthetic"),
        "tokens_per_expert": list(synthetic_counts),
        "k": k,
        "n": n,
        "tile_m": BLOCK_M,
        "tile_n": BLOCK_N,
        "tile_k": BLOCK_K,
        "group_size_m": GROUP_SIZE_M,
        "block_threads": 128,
        "active_m_tiles": active_m_tiles,
        "maximum_m_tiles": maximum_m_tiles,
        "natural_maximum_m_tiles": natural_maximum_m_tiles,
        "n_tiles": n_tiles,
        "logical_operand_bytes": logical_bytes,
        "duration_ms": durations_ms,
        "median_duration_ms": median_ms,
        "logical_operand_capacity_bytes_per_s": (
            logical_bytes / (median_ms * 1.0e-3)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--mul-routed-weight", action="store_true")
    parser.add_argument("--route-seed", type=int)
    parser.add_argument("--source-topk", type=int)
    parser.add_argument("--m-tile-counts")
    parser.add_argument("--tokens-per-expert")
    parser.add_argument("--maximum-m-tiles", type=int)
    arguments = parser.parse_args()
    if arguments.device_id not in PREFERRED_DEVICES:
        raise ValueError(f"device must be one of {PREFERRED_DEVICES}")
    if (arguments.k, arguments.n) in PRODUCTION_SHAPES:
        raise ValueError("probe dimensions must not equal the production shape")
    source_topk = (
        arguments.topk if arguments.source_topk is None
        else arguments.source_topk)
    m_tile_counts = (
        None if arguments.m_tile_counts is None
        else _parse_m_tile_counts(arguments.m_tile_counts))
    tokens_per_expert = (
        None if arguments.tokens_per_expert is None
        else _parse_tokens_per_expert(arguments.tokens_per_expert))
    if (arguments.tokens <= 0 or arguments.experts <= 0
            or (arguments.experts % 2 and m_tile_counts is None)
            or arguments.topk <= 0
            or arguments.topk > arguments.experts or arguments.k <= 0
            or arguments.n <= 0 or arguments.trials <= 0
            or (arguments.maximum_m_tiles is not None
                and arguments.maximum_m_tiles <= 0)
            or arguments.k % BLOCK_K or arguments.n % BLOCK_N
            or source_topk <= 0 or source_topk > arguments.topk
            or arguments.tokens * arguments.topk % source_topk):
        raise ValueError("K/N/trials violate the probe geometry contract")
    if m_tile_counts is not None and len(m_tile_counts) != arguments.experts:
        raise ValueError("m-tile counts must contain one entry per expert")
    if (tokens_per_expert is not None
            and len(tokens_per_expert) != arguments.experts):
        raise ValueError("tokens per expert must contain one entry per expert")
    output_dir = arguments.output_dir.resolve()
    if (output_dir == OUTPUT_ROOT.resolve()
            or OUTPUT_ROOT.resolve() not in output_dir.parents):
        raise ValueError("output directory must be below the A800 root")
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    source_sha256 = _audit_source()
    before = _wait_idle(arguments.device_id)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    result = _measure(
        arguments.tokens, arguments.experts, arguments.topk, source_topk,
        arguments.k, arguments.n, arguments.trials,
        arguments.mul_routed_weight, arguments.route_seed, m_tile_counts,
        tokens_per_expert,
        arguments.maximum_m_tiles)
    after = _device(arguments.device_id, require_idle=False)
    payload = {
        "schema": "a800_fused_moe_triton_operand_stream_v1",
        "owner": "research/kernel-accuracy-fusedmoe",
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "started_at_utc": started_at_utc,
        "baseline_commit": _run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "production_or_evaluation_duration_consumed": False,
        "timing_domain": "CUDA events around one synthetic-routing grouped matrix kernel",
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "vllm_source": str(VLLM_SOURCE),
        "vllm_source_sha256": source_sha256,
        "device_before": before,
        "device_after": after,
        "result": result,
    }
    payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path = output_dir / "result.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
