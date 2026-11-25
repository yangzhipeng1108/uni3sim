#!/usr/bin/env python3
"""Measure one sparse BM16 grouped-MatMul source program (H200).

H200 owner copy of
microbench/nvidia_a800/operators/fused_moe/sm80_bf16_bm16_sparse_w2_source_runner.py.
Routing, workspace expansion, operands, BM16/BN32/BK64/GSM1 config, the
128 MiB polluter and the W2 activation refresh before every timed launch,
CUDA-event timing, validation and the logical-byte formula are unchanged.
Hardware/stack coupling only (shared helpers come from the H200 TRI copy
``fused_moe_triton_operand_stream_runner.py``):
  - device check ``NVIDIA H200``, devices (0, 3), output root
    ``profiler_output/microbench/nvidia_h200``;
  - kernel source: the byte-identical vLLM 0.11 ``invoke_fused_moe_kernel`` /
    ``moe_align_block_size`` segments from ``vllm_0_11_fused_moe_kernel.py``
    (audited against ``--vllm-011-source-dir``), compiled by Triton 3.6 over
    the vLLM 0.24 ``_moe_C`` sort op;
  - the routing builder is executed from its verbatim source segment
    (its module imports packages absent on H200);
  - ``--baseline-commit`` replaces ``git rev-parse HEAD``;
  - clock gate: one untimed launch compiles the kernel (on A800 the first
    timed trial also carried the JIT compile; the median of 9 is unaffected),
    a 0.8 s L2-resident FP32 add spin raises the SM clock, and the whole process is
    sampled by ``nvidia-smi -lms 50``; the capture is accepted only if every
    sample in [first timed launch - 0.1 s, last timed launch + 0.1 s] reads
    >= 1980 MHz (``result.clock_gate``).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_moe_triton_operand_stream_runner import (  # noqa: E402
    ClockGate, KERNEL_MODULE, TOPK_BUILDER_SOURCE, _load_topk_builder,
    _prewarm, _software)

OUTPUT_ROOT = ROOT / "profiler_output/microbench/nvidia_h200"
DEVICE_NAME = "NVIDIA H200"
PREFERRED_DEVICES = (0, 3)
BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 64
GROUP_SIZE_M = 1
POLLUTER_BYTES = 128 * 1024 * 1024
PRODUCTION_SHAPES = (
    (512, 2048),
    (1024, 4096),
    (1536, 4096),
    (4096, 3072),
)
SOURCE_SNIPPETS = (
    "num_pid_in_group = GROUP_SIZE_M * num_pid_n",
    "offs_token[:, None] // top_k * stride_am",
    "if MUL_ROUTED_WEIGHT:",
    "accumulator = accumulator * moe_weight[:, None]",
)


def _run(command):
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(device_id, *, require_idle):
    row = _run([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,"
        "memory.used,clocks.current.sm,clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ]).stdout.strip().split(",")
    fields = tuple(value.strip() for value in row)
    if len(fields) != 9:
        raise ValueError("unexpected nvidia-smi device row")
    result = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "gpu_utilization_pct": int(fields[3]),
        "memory_utilization_pct": int(fields[4]),
        "memory_used_mib": int(fields[5]), "sm_clock_mhz": int(fields[6]),
        "memory_clock_mhz": int(fields[7]), "driver_version": fields[8],
    }
    if result["index"] != device_id or result["name"] != DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {result}")
    applications = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]).stdout
    if require_idle and any(
            line.strip() and line.split(",", 1)[0].strip() == result["uuid"]
            for line in applications.splitlines()):
        raise RuntimeError("target GPU has a compute process")
    if (require_idle
            and (result["gpu_utilization_pct"] != 0
                 or result["memory_utilization_pct"] != 0
                 or result["memory_used_mib"] > 8)):
        raise RuntimeError(f"target GPU is not idle: {result}")
    return result


def _audit_source(source_dir):
    vllm_source = source_dir / "fused_moe.py"
    source = vllm_source.read_text(encoding="utf-8")
    missing = tuple(fragment for fragment in SOURCE_SNIPPETS
                    if fragment not in source)
    if missing:
        raise ValueError(f"vLLM grouped-MoE source changed: {missing}")
    from vllm_0_11_fused_moe_kernel import audit_against_source
    audit_against_source(vllm_source, source_dir / "moe_align_block_size.py")
    return vllm_source, _sha256(vllm_source)


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


def _parse_tokens_per_expert(text):
    values = tuple(int(value) for value in text.split(","))
    if not values or any(value < 0 for value in values):
        raise ValueError(
            "tokens per expert must be a non-empty non-negative list")
    return values


def _measure(
        tokens, experts, topk, active_experts, k, n, trials, role,
        route_seed, expected_active_m_tiles, source_topk, tokens_per_expert,
        maximum_m_tiles, clock_gate):
    import torch
    import triton.language as tl
    from vllm_0_11_fused_moe_kernel import (
        invoke_fused_moe_kernel, moe_align_block_size)
    build_nvidia_a800_moe_topk_ids = _load_topk_builder()

    routed_rows = tokens * topk
    if routed_rows % source_topk:
        raise ValueError("routed rows must divide exactly by source top-k")
    source_rows = routed_rows // source_topk
    if not 1 <= active_experts <= min(routed_rows, experts):
        raise ValueError("active expert count is outside the routing domain")
    if tokens_per_expert is not None:
        if len(tokens_per_expert) != experts:
            raise ValueError(
                "tokens per expert must contain one entry per expert")
        if sum(tokens_per_expert) != routed_rows:
            raise ValueError(
                "tokens per expert must conserve the routed rows")
        route_ids = build_nvidia_a800_moe_topk_ids(
            token_counts=list(tokens_per_expert),
            num_tokens=tokens,
            num_experts=experts,
            top_k=topk,
            context="A800 BM16 exact grouped-MoE routing",
        )
    elif route_seed is None:
        route_ids = tuple(index % active_experts for index in range(routed_rows))
    else:
        generator = random.Random(route_seed)
        counts = [0] * experts
        for _ in range(routed_rows):
            counts[generator.randrange(active_experts)] += 1
        route_ids = build_nvidia_a800_moe_topk_ids(
            token_counts=counts,
            num_tokens=tokens,
            num_experts=experts,
            top_k=topk,
            context="A800 BM16 synthetic grouped-MoE routing",
        )
    topk_ids = torch.tensor(
        route_ids, dtype=torch.int32, device="cuda").view(tokens, topk)
    sorted_ids, expert_ids, post_padded = moe_align_block_size(
        topk_ids, BLOCK_M, experts, None)
    active_m_tiles = int(post_padded.item()) // BLOCK_M
    if expected_active_m_tiles is None:
        expected_active_m_tiles = active_experts
    if active_m_tiles != expected_active_m_tiles:
        raise RuntimeError(
            "synthetic routing has the wrong active M-tile count: "
            f"{active_m_tiles} != {expected_active_m_tiles}")
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
        expanded_sorted_ids[sorted_ids.numel():].fill_(routed_rows)
        sorted_ids = expanded_sorted_ids
        expanded_expert_ids = torch.empty(
            (maximum_m_tiles,), dtype=expert_ids.dtype,
            device=expert_ids.device)
        expanded_expert_ids[:expert_ids.numel()].copy_(expert_ids)
        expanded_expert_ids[expert_ids.numel():].zero_()
        expert_ids = expanded_expert_ids
    activation = torch.ones(
        (source_rows, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((experts, n, k), dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(
        (source_rows, source_topk, n),
        dtype=torch.bfloat16,
        device="cuda")
    routed_weights = torch.full(
        (source_rows, source_topk), 0.5, dtype=torch.bfloat16, device="cuda")
    if role == "w1":
        routed_weight_argument = None
        multiply_routed_weight = False
        expected_output_scale = 1.0
    elif role == "w2":
        routed_weight_argument = routed_weights
        multiply_routed_weight = True
        expected_output_scale = 0.5
    else:
        raise ValueError(f"unsupported grouped-MoE role: {role!r}")
    polluter = torch.randint(
        0, 256, (POLLUTER_BYTES,), dtype=torch.uint8, device="cuda")
    config = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
    }

    def operation():
        invoke_fused_moe_kernel(
            activation, weight, output, None, None, None,
            routed_weight_argument,
            sorted_ids, expert_ids, post_padded, multiply_routed_weight,
            source_topk, config,
            tl.bfloat16, False, False, False, False, False)

    operation()
    torch.cuda.synchronize()
    _prewarm(torch)
    samples = []
    window_begin = time.time()
    for _ in range(trials):
        torch.sum(polluter)
        if role == "w2":
            activation.fill_(1.0)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) * 1.0e6)
    window_end = time.time()
    clock = clock_gate.stop(window_begin, window_end)
    expected_output = k * expected_output_scale
    if float(output[0, 0, 0].item()) != expected_output:
        raise RuntimeError("grouped-MoE output validation failed")
    n_tiles = math.ceil(n / BLOCK_N)
    logical_bytes = (
        routed_rows * k * 2 * n_tiles
        + active_m_tiles * n_tiles * k * BLOCK_N * 2)
    median_ns = statistics.median(samples)
    return {
        "tokens": tokens,
        "experts": experts,
        "topk": topk,
        "source_topk": source_topk,
        "source_rows": source_rows,
        "active_experts": active_experts,
        "route_seed": route_seed,
        "tokens_per_expert_source": (
            "exact" if tokens_per_expert is not None else "synthetic"),
        "tokens_per_expert": (
            list(tokens_per_expert) if tokens_per_expert is not None else None),
        "role": role,
        "mul_routed_weight": role == "w2",
        "activation_refresh_before_timed_launch": role == "w2",
        "k": k,
        "n": n,
        "active_m_tiles": active_m_tiles,
        "maximum_m_tiles": maximum_m_tiles,
        "natural_maximum_m_tiles": natural_maximum_m_tiles,
        "grid": maximum_m_tiles * n_tiles,
        "logical_operand_bytes": logical_bytes,
        "duration_ns": samples,
        "median_duration_ns": median_ns,
        "logical_operand_capacity_bytes_per_s": logical_bytes / (median_ns * 1e-9),
        "clock_gate": clock,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--experts", required=True, type=int)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--source-topk", type=int)
    parser.add_argument("--active-experts", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--role", choices=("w1", "w2"), default="w2")
    parser.add_argument("--route-seed", type=int)
    parser.add_argument("--expected-active-m-tiles", type=int)
    parser.add_argument("--tokens-per-expert")
    parser.add_argument("--maximum-m-tiles", type=int)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--vllm-011-source-dir", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    arguments = parser.parse_args()
    if arguments.device_id not in PREFERRED_DEVICES:
        raise ValueError(f"device must be one of {PREFERRED_DEVICES}")
    if (arguments.k, arguments.n) in PRODUCTION_SHAPES:
        raise ValueError("probe dimensions must not equal a production shape")
    if (arguments.tokens <= 0 or arguments.experts <= 0
            or arguments.topk <= 0 or arguments.topk > arguments.experts
            or arguments.active_experts <= 0
            or arguments.active_experts > min(
                arguments.tokens * arguments.topk, arguments.experts)
            or (arguments.expected_active_m_tiles is not None
                and arguments.expected_active_m_tiles <= 0)
            or (arguments.maximum_m_tiles is not None
                and arguments.maximum_m_tiles <= 0)
            or arguments.k <= 0 or arguments.n <= 0
            or arguments.k % BLOCK_K or arguments.n % BLOCK_N
            or not 3 <= arguments.trials <= 11):
        raise ValueError("probe arguments violate the source contract")
    source_topk = (
        arguments.topk if arguments.source_topk is None
        else arguments.source_topk)
    if (source_topk <= 0
            or arguments.tokens * arguments.topk % source_topk):
        raise ValueError("source top-k must divide the routed-row count")
    tokens_per_expert = (
        None if arguments.tokens_per_expert is None
        else _parse_tokens_per_expert(arguments.tokens_per_expert))
    if (tokens_per_expert is not None
            and len(tokens_per_expert) != arguments.experts):
        raise ValueError(
            "tokens per expert must contain one entry per expert")
    output = arguments.output.resolve()
    if OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("output must be below the H200 profiler root")
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    before = _wait_idle(arguments.device_id)
    vllm_source, source_sha256 = _audit_source(
        arguments.vllm_011_source_dir.resolve())
    started_at_utc = datetime.now(timezone.utc).isoformat()
    clock_gate = ClockGate(
        arguments.device_id, output.parent / "clock.csv")
    clock_gate.start()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    result = _measure(
        arguments.tokens, arguments.experts, arguments.topk,
        arguments.active_experts, arguments.k, arguments.n,
        arguments.trials, arguments.role, arguments.route_seed,
        arguments.expected_active_m_tiles, source_topk, tokens_per_expert,
        arguments.maximum_m_tiles, clock_gate)
    payload = {
        "schema": "h200_bm16_sparse_grouped_source_v1",
        "hardware_id": "nvidia_h200",
        "owner": "research/kernel-accuracy-fusedmoe",
        "baseline_commit": arguments.baseline_commit,
        "command": [str(Path(__file__).resolve()), *os.sys.argv[1:]],
        "pid": os.getpid(),
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_or_evaluation_duration_consumed": False,
        "timing_domain": (
            "CUDA events around one role-specific grouped source launch"),
        "cache_contract": (
            "128 MiB polluter before each launch; W2 additionally refreshes "
            "the full activation, while W1 preserves its source input"),
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "vllm_source": str(vllm_source),
        "vllm_source_sha256": source_sha256,
        "kernel_module": str(KERNEL_MODULE),
        "kernel_module_sha256": _sha256(KERNEL_MODULE),
        "routing_builder_source": str(TOPK_BUILDER_SOURCE),
        "routing_builder_source_sha256": _sha256(TOPK_BUILDER_SOURCE),
        "software": _software(),
        "device_before": before,
        "device_after": _device(arguments.device_id, require_idle=False),
        "result": result,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
