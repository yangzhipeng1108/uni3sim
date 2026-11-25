#!/usr/bin/env python3
"""Run a synthetic-routing Triton grouped-operand stream on MetaX C500.

The existing C++ gather-cache benchmark reproduces aggregate L2 bytes, but it
issues scalar global loads.  This probe isolates the vectorized A/B load and
dot pipeline while replacing production routing with a deterministic,
workload-independent random top-k assignment and requiring non-production K/N
geometry.  Event timing is an eligible mechanism capacity for that exact
launch/pipeline domain; no production or evaluation duration is read by this
runner.  Profiler mode remains available for non-duration counter validation.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl

from dataset_pipeline.kernel_dataset.metax_c500_fused_moe_kernels import (
    fused_moe_kernel as _production_fused_moe_kernel,
)


SCHEMA_VERSION = 5
HARDWARE_ID = "metax_c500"
MECHANISM = "triton_grouped_operand_pipeline"
BASELINE_COMMIT = "057480a20da5efdaa666140b8e65d333e30a66b5"
PRODUCTION_SOURCE_RELATIVE_PATHS = (
    "scripts/kernel_dataset/metax_c500/fused_moe_runner.py",
    "dataset_pipeline/kernel_dataset/metax_c500_fused_moe_kernels.py",
)
PRODUCTION_SOURCE_SHA256S = {
    "scripts/kernel_dataset/metax_c500/fused_moe_runner.py": (
        "436bee88cb5fe444ec0aa5204e7df158edb51119891424ed15237e9dc9c13474"),
    "dataset_pipeline/kernel_dataset/metax_c500_fused_moe_kernels.py": (
        "40e0cd0e9439a6e060cb73ebe36f344ae25d0ba51efddd1ad38d154ece5f064b"),
}
PRODUCTION_SOURCE_BUNDLE_SHA256 = (
    "91b2278fb593e47c8b71965e1b4564969b484326f8ec77559721263fd4eda2a3")
DEFAULT_ROUTE_SEED = 20260824
DEFAULT_EVENT_TRIALS = 3
EXPECTED_WAVE_SIZE = 64
UINT64_MASK = (1 << 64) - 1
SUPPORTED_SIGNATURES = {
    "bm16_bk64_bn32_gsm1": (16, 64, 32, 1),
    "bm64_bk32_bn64_gsm8": (64, 32, 64, 8),
    "bm32_bk128_bn64_gsm1": (32, 128, 64, 1),
    "bm16_bk128_bn64_gsm1": (16, 128, 64, 1),
    # vLLM-metax 0.24.0's BF16 default rule above 64 batch tokens.
    "bm32_bk64_bn128_gsm1": (32, 64, 128, 1),
    "bm64_bk64_bn128_gsm1": (64, 64, 128, 1),
    "bm128_bk64_bn128_gsm1": (128, 64, 128, 1),
}
# Deployed AOT compiler resource facts per signature when the launch pins
# num_warps/num_stages explicitly (midlayer
# src/kernel/launch_config/metax_c500/fused_moe_launch.py keeps the same
# table for the production selector).
EXPECTED_DYNAMIC_SHARED_BYTES = {
    ("bm32_bk128_bn64_gsm1", 4, 3): 49_152,
    ("bm16_bk128_bn64_gsm1", 4, 4): 40_960,
    ("bm32_bk64_bn128_gsm1", 4, 3): 40_960,
    # BLOCK_SIZE_M=64 compiles to two distinct programs that share every other
    # resource fact; both are deployed (four warps at most 128 batch tokens,
    # eight above).
    ("bm64_bk64_bn128_gsm1", 4, 3): 49_152,
    ("bm64_bk64_bn128_gsm1", 8, 3): 49_152,
    ("bm128_bk64_bn128_gsm1", 8, 3): 65_536,
}


@dataclass(frozen=True)
class RouteLayout:
    sorted_route_ids: tuple[int, ...]
    expert_ids: tuple[int, ...]
    tokens_per_expert: tuple[int, ...]
    valid_routes: int
    post_padded_routes: int
    maximum_padded_routes: int
    active_m_tiles: int
    maximum_m_tiles: int
    sha256: str


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{label} must be a positive integer")
    return value


def _route_seed(value):
    if (isinstance(value, bool) or not isinstance(value, int)
            or value < 0 or value > UINT64_MASK):
        raise TypeError("route_seed must be an unsigned 64-bit integer")
    return value


def _signature(name):
    if name not in SUPPORTED_SIGNATURES:
        raise ValueError(f"unknown Triton operand-stream signature: {name!r}")
    return SUPPORTED_SIGNATURES[name]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _production_source_bundle_sha256(source_sha256s):
    if set(source_sha256s) != set(PRODUCTION_SOURCE_RELATIVE_PATHS):
        raise ValueError("production FusedMoE source bundle is incomplete")
    digest = hashlib.sha256(
        b"midlayer-c500-fused-moe-production-source-v1\0")
    for relative_path in PRODUCTION_SOURCE_RELATIVE_PATHS:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(source_sha256s[relative_path]))
    return digest.hexdigest()


def audit_production_source():
    repository_root = Path(__file__).resolve().parents[4]
    source_sha256s = {}
    sources = []
    paths = []
    for relative_path in PRODUCTION_SOURCE_RELATIVE_PATHS:
        source_path = repository_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"production FusedMoE source is missing: {source_path}")
        actual_sha256 = _sha256(source_path)
        if actual_sha256 != PRODUCTION_SOURCE_SHA256S[relative_path]:
            raise ValueError(
                "production FusedMoE source differs from the frozen "
                f"baseline: {relative_path}")
        source_sha256s[relative_path] = actual_sha256
        sources.append(source_path.read_text(encoding="utf-8"))
        paths.append(source_path)
    if (_production_source_bundle_sha256(source_sha256s)
            != PRODUCTION_SOURCE_BUNDLE_SHA256):
        raise ValueError(
            "production FusedMoE source bundle differs from the frozen "
            "baseline")
    source = "\n".join(sources)
    required_fragments = (
        "offs_token[:, None] // top_k * stride_am",
        "off_experts * stride_be",
        "a = tl.load(a_ptrs",
        "b = tl.load(b_ptrs",
        "accumulator += tl.dot(a, b)",
        "a_ptrs += BLOCK_SIZE_K * stride_ak",
        "b_ptrs += BLOCK_SIZE_K * stride_bk",
    )
    if any(fragment not in source for fragment in required_fragments):
        raise ValueError("production FusedMoE load/dot program changed")
    return tuple(paths)


def _splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return value ^ (value >> 31)


def _sample_experts(token, experts, topk, seed):
    if experts == 1:
        return (0,)
    base = _splitmix64(seed ^ token) % experts
    stride = _splitmix64(seed ^ token ^ 0xD1B54A32D192ED03) % experts
    if stride == 0:
        stride = 1
    while math.gcd(stride, experts) != 1:
        stride = (stride + 1) % experts
        if stride == 0:
            stride = 1
    return tuple((base + slot * stride) % experts for slot in range(topk))


def build_random_topk_route(tokens, experts, topk, block_m, seed):
    """Build the same expert-major padded interface as moe_align_block_size."""

    for value, label in (
            (tokens, "tokens"), (experts, "experts"), (topk, "topk"),
            (block_m, "block_m")):
        _positive_integer(value, label)
    _route_seed(seed)
    if topk > experts:
        raise ValueError("topk cannot exceed experts")

    buckets = [[] for _ in range(experts)]
    for token in range(tokens):
        selected = _sample_experts(token, experts, topk, seed)
        for slot, expert in enumerate(selected):
            buckets[expert].append(token * topk + slot)

    valid_routes = tokens * topk
    sentinel = valid_routes
    maximum_padded_routes = valid_routes + experts * (block_m - 1)
    sorted_route_ids = []
    expert_ids = []
    tokens_per_expert = []
    for expert, bucket in enumerate(buckets):
        tokens_per_expert.append(len(bucket))
        padded = math.ceil(len(bucket) / block_m) * block_m
        sorted_route_ids.extend(bucket)
        sorted_route_ids.extend([sentinel] * (padded - len(bucket)))
        expert_ids.extend([expert] * (padded // block_m))
    post_padded_routes = len(sorted_route_ids)
    maximum_m_tiles = math.ceil(maximum_padded_routes / block_m)
    maximum_padded_storage = maximum_m_tiles * block_m
    sorted_route_ids.extend(
        [sentinel] * (maximum_padded_storage - post_padded_routes))
    expert_ids.extend([0] * (maximum_m_tiles - len(expert_ids)))

    if sum(tokens_per_expert) != valid_routes:
        raise AssertionError("random top-k routing did not conserve tokens*topk")
    if len(sorted_route_ids) != maximum_padded_storage:
        raise AssertionError("route storage does not cover the maximum grid")
    if len(expert_ids) != maximum_m_tiles:
        raise AssertionError("expert IDs do not cover the maximum M-tile grid")
    if any(value < 0 or value > sentinel for value in sorted_route_ids):
        raise AssertionError("route IDs are outside the valid/sentinel domain")

    digest = hashlib.sha256()
    digest.update(struct.pack(
        "<6Q", tokens, experts, topk, block_m, seed,
        post_padded_routes))
    for value in sorted_route_ids:
        digest.update(struct.pack("<I", value))
    for value in expert_ids:
        digest.update(struct.pack("<I", value))
    return RouteLayout(
        sorted_route_ids=tuple(sorted_route_ids),
        expert_ids=tuple(expert_ids),
        tokens_per_expert=tuple(tokens_per_expert),
        valid_routes=valid_routes,
        post_padded_routes=post_padded_routes,
        maximum_padded_routes=maximum_padded_routes,
        active_m_tiles=post_padded_routes // block_m,
        maximum_m_tiles=maximum_m_tiles,
        sha256=digest.hexdigest(),
    )


def build_contract(
        signature, tokens, experts, topk, source_topk, k, n, route_seed,
        mul_routed_weight=False, use_production_kernel=False,
        refresh_activation_before_trial=False, weight_only_source_order=False,
        early_return_prefix_only=False, all_empty_prefix_control=False,
        num_warps=None, num_stages=None):
    block_m, block_k, block_n, group_size_m = _signature(signature)
    for value, label in (
            (tokens, "tokens"), (experts, "experts"), (topk, "topk"),
            (source_topk, "source_topk"), (k, "k"), (n, "n")):
        _positive_integer(value, label)
    _route_seed(route_seed)
    if topk > experts:
        raise ValueError("topk cannot exceed experts")
    if weight_only_source_order and (
            mul_routed_weight or use_production_kernel
            or refresh_activation_before_trial):
        raise ValueError(
            "weight-only source-order probe cannot combine production, "
            "activation, or routed-weight paths")
    if early_return_prefix_only and (
            mul_routed_weight or use_production_kernel
            or refresh_activation_before_trial or weight_only_source_order):
        raise ValueError(
            "early-return prefix probe cannot combine operand, production, "
            "activation, routed-weight, or weight-stream paths")
    if all_empty_prefix_control and not early_return_prefix_only:
        raise ValueError(
            "all-empty prefix control requires the early-return prefix probe")
    if tokens * topk % source_topk:
        raise ValueError("tokens*topk must be divisible by source_topk")
    if k % block_k:
        raise ValueError("k must be divisible by the signature block_k")
    if n % block_n:
        raise ValueError("n must be divisible by the signature block_n")
    for launch_fact, label in ((num_warps, "num_warps"), (num_stages, "num_stages")):
        if launch_fact is not None:
            _positive_integer(launch_fact, label)
    layout = build_random_topk_route(
        tokens, experts, topk, block_m, route_seed)
    source_rows = layout.valid_routes // source_topk
    n_tiles = n // block_n
    launch_padded_routes = layout.maximum_padded_routes
    if source_rows < block_m:
        launch_padded_routes = min(
            launch_padded_routes, source_rows * source_topk * block_m)
    launch_m_tiles = math.ceil(launch_padded_routes / block_m)
    grid = launch_m_tiles * n_tiles
    element_bytes = 2
    operand_replay = 2 if block_m == 64 and block_k == 32 else 1
    activation_l2_read_bytes = (
        layout.valid_routes * k * element_bytes * n_tiles * operand_replay)
    weight_l2_read_bytes = (
        layout.active_m_tiles * n_tiles * k * block_n * element_bytes
        * operand_replay)
    output_write_bytes = layout.valid_routes * n * element_bytes
    prefix_post_padded_routes = (
        0 if all_empty_prefix_control else layout.post_padded_routes)
    prefix_active_m_tiles = math.ceil(prefix_post_padded_routes / block_m)
    prefix_active_ctas = prefix_active_m_tiles * n_tiles
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "baseline_commit": BASELINE_COMMIT,
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "production_source_files": [
            {
                "path": relative_path,
                "sha256": PRODUCTION_SOURCE_SHA256S[relative_path],
            }
            for relative_path in PRODUCTION_SOURCE_RELATIVE_PATHS
        ],
        "production_source_bundle_sha256": (
            PRODUCTION_SOURCE_BUNDLE_SHA256),
        "production_or_evaluation_duration_consumed": False,
        "signature": signature,
        "tokens": tokens,
        "experts": experts,
        "topk": topk,
        "source_topk": source_topk,
        "source_rows": source_rows,
        "k": k,
        "n": n,
        "block_m": block_m,
        "block_k": block_k,
        "block_n": block_n,
        "group_size_m": group_size_m,
        "route_seed": route_seed,
        "route_algorithm": "splitmix64_affine_without_replacement_v1",
        "route_sha256": layout.sha256,
        "valid_routes": layout.valid_routes,
        "post_padded_routes": layout.post_padded_routes,
        "maximum_padded_routes": layout.maximum_padded_routes,
        "active_m_tiles": layout.active_m_tiles,
        "active_experts": sum(
            1 for count in layout.tokens_per_expert if count > 0),
        "maximum_expert_m_tiles": max(
            (math.ceil(count / block_m)
             for count in layout.tokens_per_expert if count > 0),
            default=0),
        "maximum_m_tiles": layout.maximum_m_tiles,
        "launch_padded_routes": launch_padded_routes,
        "launch_m_tiles": launch_m_tiles,
        "n_tiles": n_tiles,
        "grid": grid,
        "block": 4 * EXPECTED_WAVE_SIZE,
        "operand_replay": operand_replay,
        "activation_l2_read_bytes": activation_l2_read_bytes,
        "weight_l2_read_bytes": weight_l2_read_bytes,
        "expected_operand_l2_read_bytes": (
            activation_l2_read_bytes + weight_l2_read_bytes),
        "output_write_bytes": output_write_bytes,
        "mul_routed_weight": bool(mul_routed_weight),
        "use_production_kernel": bool(use_production_kernel),
        "refresh_activation_before_trial": bool(
            refresh_activation_before_trial),
        "weight_only_source_order": bool(weight_only_source_order),
        "early_return_prefix_only": bool(early_return_prefix_only),
        "all_empty_prefix_control": bool(all_empty_prefix_control),
        "prefix_post_padded_routes": prefix_post_padded_routes,
        "prefix_active_m_tiles": prefix_active_m_tiles,
        "prefix_active_ctas": prefix_active_ctas,
        "routed_weight_read_bytes": (
            layout.valid_routes * element_bytes if mul_routed_weight else 0),
        "activation_unique_bytes": source_rows * k * element_bytes,
        "weight_unique_bytes": experts * n * k * element_bytes,
        "route_layout": layout,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


@triton.jit
def _grouped_operand_stream(
        activation, weight, output, routed_weights, sorted_route_ids, expert_ids,
        post_padded_routes, valid_routes: tl.constexpr,
        MAXIMUM_PADDED_ROUTES: tl.constexpr,
        K: tl.constexpr, N: tl.constexpr, TOPK: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(MAXIMUM_PADDED_ROUTES, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    post_padded_route_count = tl.load(post_padded_routes)
    if pid_m * BLOCK_M >= post_padded_route_count:
        return
    route_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    route_ids = tl.load(sorted_route_ids + route_offsets)
    route_mask = route_ids < valid_routes
    source_rows = route_ids // TOPK
    expert = tl.load(expert_ids + pid_m).to(tl.int64)
    n_offsets = (
        pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
    k_offsets = tl.arange(0, BLOCK_K)
    activation_ptrs = (
        activation + source_rows[:, None] * K + k_offsets[None, :])
    weight_ptrs = (
        weight + expert * N * K + n_offsets[None, :] * K
        + k_offsets[:, None])
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_step in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = k_offsets < K - k_step * BLOCK_K
        activation_tile = tl.load(
            activation_ptrs,
            mask=route_mask[:, None] & k_mask[None, :],
            other=0.0)
        weight_tile = tl.load(
            weight_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator += tl.dot(activation_tile, weight_tile)
        activation_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            routed_weights + route_ids, mask=route_mask, other=0.0)
        accumulator *= routed_weight[:, None]

    output_tile = accumulator.to(tl.bfloat16)
    output_offsets = (
        route_ids[:, None].to(tl.int64) * N +
        pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
    output_mask = route_mask[:, None] & (
        pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(output + output_offsets, output_tile, mask=output_mask)


@triton.jit
def _grouped_weight_source_order_stream(
        weight, expert_ids, post_padded_routes, checksum,
        MAXIMUM_PADDED_ROUTES: tl.constexpr, K: tl.constexpr,
        N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Issue only the production B descriptor stream in source PID order.

    This intentionally excludes token routing dereferences, activation loads,
    tensor-core work, activation materialization, and output tiles.  The
    grouped pid mapping and B pointer construction mirror the frozen Triton
    source, so layer-1 counters isolate only the cache/downstream behaviour
    of the weight stream across logical M groups.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(MAXIMUM_PADDED_ROUTES, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    post_padded_route_count = tl.load(post_padded_routes)
    if pid_m * BLOCK_M >= post_padded_route_count:
        return
    expert = tl.load(expert_ids + pid_m).to(tl.int64)
    n_offsets = (
        pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
    k_offsets = tl.arange(0, BLOCK_K)
    weight_ptrs = (
        weight + expert * N * K + n_offsets[None, :] * K
        + k_offsets[:, None])
    partial = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        weight_tile = tl.load(weight_ptrs)
        partial += tl.sum(weight_tile.to(tl.float32), axis=0)
        weight_ptrs += BLOCK_K
    tl.store(checksum + pid, tl.sum(partial, axis=0))


@triton.jit
def _grouped_early_return_prefix(
        post_padded_routes, checksum,
        MAXIMUM_PADDED_ROUTES: tl.constexpr, N: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr):
    """Isolate the source-visible grouped PID prefix and early return.

    The active path contains only a checksum store so the runtime cannot
    dead-code-eliminate the data-dependent branch.  It deliberately omits
    routing, A/B operands, tensor work, and production output storage.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(MAXIMUM_PADDED_ROUTES, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)

    post_padded_route_count = tl.load(post_padded_routes)
    if pid_m * BLOCK_M >= post_padded_route_count:
        return
    tl.store(checksum + pid, 1.0)


def _launch_options(contract):
    options = {}
    if contract["num_warps"] is not None:
        options["num_warps"] = contract["num_warps"]
    if contract["num_stages"] is not None:
        options["num_stages"] = contract["num_stages"]
    return options


def _run_once(contract, activation, weight, output, routed_weights,
              sorted_route_ids, expert_ids, post_padded_routes):
    if contract["use_production_kernel"]:
        _production_fused_moe_kernel[(contract["grid"],)](
            activation,
            weight,
            output,
            None,
            None,
            None,
            routed_weights,
            sorted_route_ids,
            expert_ids,
            post_padded_routes,
            contract["n"],
            contract["k"],
            contract["launch_padded_routes"],
            contract["valid_routes"],
            activation.stride(0),
            activation.stride(1),
            weight.stride(0),
            weight.stride(2),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            MUL_ROUTED_WEIGHT=contract["mul_routed_weight"],
            top_k=contract["source_topk"],
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            per_channel_quant=False,
            HAS_BIAS=False,
            BLOCK_SIZE_M=contract["block_m"],
            BLOCK_SIZE_N=contract["block_n"],
            BLOCK_SIZE_K=contract["block_k"],
            GROUP_SIZE_M=contract["group_size_m"],
            **_launch_options(contract),
        )
        return
    _grouped_operand_stream[(contract["grid"],)](
        activation,
        weight,
        output,
        routed_weights,
        sorted_route_ids,
        expert_ids,
        post_padded_routes,
        valid_routes=contract["valid_routes"],
        MAXIMUM_PADDED_ROUTES=contract["launch_padded_routes"],
        K=contract["k"],
        N=contract["n"],
        TOPK=contract["source_topk"],
        MUL_ROUTED_WEIGHT=contract["mul_routed_weight"],
        BLOCK_M=contract["block_m"],
        BLOCK_N=contract["block_n"],
        BLOCK_K=contract["block_k"],
        GROUP_SIZE_M=contract["group_size_m"],
        **_launch_options(contract),
    )


def _run_weight_source_order_once(
        contract, weight, expert_ids, post_padded_routes, checksum):
    _grouped_weight_source_order_stream[(contract["grid"],)](
        weight,
        expert_ids,
        post_padded_routes,
        checksum,
        MAXIMUM_PADDED_ROUTES=contract["launch_padded_routes"],
        K=contract["k"],
        N=contract["n"],
        BLOCK_M=contract["block_m"],
        BLOCK_N=contract["block_n"],
        BLOCK_K=contract["block_k"],
        GROUP_SIZE_M=contract["group_size_m"],
        **_launch_options(contract),
    )


def _run_early_return_prefix_once(
        contract, post_padded_routes, checksum):
    _grouped_early_return_prefix[(contract["grid"],)](
        post_padded_routes,
        checksum,
        MAXIMUM_PADDED_ROUTES=contract["launch_padded_routes"],
        N=contract["n"],
        BLOCK_M=contract["block_m"],
        BLOCK_N=contract["block_n"],
        GROUP_SIZE_M=contract["group_size_m"],
        **_launch_options(contract),
    )


def _device_tensors(contract):
    layout = contract["route_layout"]
    device_name = torch.cuda.get_device_name(0)
    if "C500" not in device_name:
        raise RuntimeError(
            f"expected a MetaX C500 device, found {device_name!r}")
    activation = torch.ones(
        (contract["source_rows"], contract["k"]),
        dtype=torch.bfloat16,
        device="cuda")
    weight = torch.empty(
        (contract["experts"], contract["n"], contract["k"]),
        dtype=torch.bfloat16,
        device="cuda")
    populated_experts = torch.tensor(
        [
            expert
            for expert, count in enumerate(layout.tokens_per_expert)
            if count > 0
        ],
        dtype=torch.int64,
        device="cuda",
    )
    weight.index_fill_(0, populated_experts, 1.0)
    output = torch.zeros(
        (contract["valid_routes"], contract["n"]),
        dtype=torch.bfloat16,
        device="cuda")
    routed_weights = torch.ones(
        (contract["valid_routes"],), dtype=torch.bfloat16, device="cuda")
    sorted_route_ids = torch.tensor(
        layout.sorted_route_ids, dtype=torch.int32, device="cuda")
    expert_ids = torch.tensor(
        layout.expert_ids, dtype=torch.int32, device="cuda")
    post_padded_routes = torch.tensor(
        [layout.post_padded_routes], dtype=torch.int32, device="cuda")
    return (
        layout, activation, weight, output, routed_weights, sorted_route_ids,
        expert_ids, post_padded_routes)


def _device_weight_source_order_tensors(contract):
    """Allocate only the B stream and compact source-control inputs."""

    layout = contract["route_layout"]
    device_name = torch.cuda.get_device_name(0)
    if "C500" not in device_name:
        raise RuntimeError(
            f"expected a MetaX C500 device, found {device_name!r}")
    weight = torch.ones(
        (contract["experts"], contract["n"], contract["k"]),
        dtype=torch.bfloat16,
        device="cuda")
    expert_ids = torch.tensor(
        layout.expert_ids, dtype=torch.int32, device="cuda")
    post_padded_routes = torch.tensor(
        [layout.post_padded_routes], dtype=torch.int32, device="cuda")
    checksum = torch.zeros(
        (contract["grid"],), dtype=torch.float32, device="cuda")
    return weight, expert_ids, post_padded_routes, checksum


def _device_early_return_prefix_tensors(contract):
    """Allocate only the scalar source control and non-production checksum."""

    device_name = torch.cuda.get_device_name(0)
    if "C500" not in device_name:
        raise RuntimeError(
            f"expected a MetaX C500 device, found {device_name!r}")
    post_padded_routes = torch.tensor(
        [contract["prefix_post_padded_routes"]], dtype=torch.int32,
        device="cuda")
    checksum = torch.zeros(
        (contract["grid"],), dtype=torch.float32, device="cuda")
    return post_padded_routes, checksum


def _compiled_resource_report(jit_fn):
    """Best-effort compiled-kernel resource fact extraction for provenance."""

    def _metadata_field(metadata, name):
        if metadata is None:
            return None
        if isinstance(metadata, dict):
            return metadata.get(name)
        return getattr(metadata, name, None)

    try:
        cache = getattr(jit_fn, "cache", None)
        if cache is None:
            device_caches = getattr(jit_fn, "device_caches", None)
            if device_caches:
                cache = device_caches[0][0] if isinstance(
                    device_caches, dict) else None
        if not cache:
            return {"introspection": "unavailable"}
        compiled = None
        for per_device in (
                cache.values() if isinstance(cache, dict) else ()):
            if isinstance(per_device, dict) and per_device:
                compiled = next(iter(per_device.values()))
                break
        if compiled is None:
            return {"introspection": "unavailable"}
        metadata = getattr(compiled, "metadata", None)
        report = {
            "introspection": "ok",
            "shared": _metadata_field(metadata, "shared"),
            "num_warps": _metadata_field(metadata, "num_warps"),
            "num_stages": _metadata_field(metadata, "num_stages"),
        }
        name = getattr(compiled, "name", None)
        if name is not None:
            report["name"] = str(name)
        return report
    except Exception as exc:  # provenance only; never fail the run here
        return {"introspection": "error", "detail": repr(exc)}


def _assert_compiled_resources(contract, jit_fn):
    report = _compiled_resource_report(jit_fn)
    sys.stderr.write(f"compiled_resources {json.dumps(report)}\n")
    expected = EXPECTED_DYNAMIC_SHARED_BYTES.get(
        (contract["signature"], contract["num_warps"],
         contract["num_stages"]))
    if expected is None or report.get("introspection") != "ok":
        return
    if report.get("shared") != expected:
        raise RuntimeError(
            "compiled dynamic shared memory differs from the deployed AOT "
            f"fact: expected {expected}, observed {report.get('shared')}")


def _run(
        contract, profile_once, attach_wait_seconds, event_trials,
        event_repetitions):
    if contract["early_return_prefix_only"]:
        post_padded_routes, checksum = _device_early_return_prefix_tensors(
            contract)
        def run_once():
            _run_early_return_prefix_once(
                contract, post_padded_routes, checksum)

        def validate():
            observed_active_ctas = int(checksum.sum().item())
            # The active path stores an idempotent one per CTA, so repeated
            # warmup/timed launches preserve this source-derived census.
            expected_active_ctas = contract["prefix_active_ctas"]
            if observed_active_ctas != expected_active_ctas:
                raise RuntimeError(
                    "early-return prefix active CTA count mismatch: "
                    f"expected {expected_active_ctas}, got "
                    f"{observed_active_ctas}")
            return {"active_prefix_ctas": observed_active_ctas}

        activation = None
    elif contract["weight_only_source_order"]:
        weight, expert_ids, post_padded_routes, checksum = (
            _device_weight_source_order_tensors(contract))

        def run_once():
            _run_weight_source_order_once(
                contract, weight, expert_ids, post_padded_routes, checksum)

        def validate():
            return {"output_checksum": float(checksum.sum().item())}

        activation = None
    else:
        tensors = _device_tensors(contract)
        arguments = tensors[1:]
        activation = arguments[0]

        def run_once():
            _run_once(contract, *arguments)

        def validate():
            output = arguments[2]
            return {
                "first_output": float(output[0, 0].item()),
                "last_output": float(output[-1, -1].item()),
            }

    run_once()
    torch.cuda.synchronize()
    run_once()
    torch.cuda.synchronize()
    active_jit = (
        _grouped_early_return_prefix if contract["early_return_prefix_only"]
        else _grouped_weight_source_order_stream
        if contract["weight_only_source_order"]
        else _production_fused_moe_kernel if contract["use_production_kernel"]
        else _grouped_operand_stream)
    _assert_compiled_resources(contract, active_jit)

    event_samples = []
    if profile_once:
        if event_repetitions != 1:
            raise ValueError(
                "profile-once mode requires exactly one event repetition")
        if attach_wait_seconds <= 0:
            raise ValueError("profile attach wait must be positive")
        time.sleep(attach_wait_seconds)
        if activation is not None and contract["refresh_activation_before_trial"]:
            activation.fill_(1.0)
        torch.cuda.profiler.start()
        run_once()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    else:
        _positive_integer(event_trials, "event_trials")
        _positive_integer(event_repetitions, "event_repetitions")
        for _ in range(event_trials):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            if activation is not None and contract["refresh_activation_before_trial"]:
                activation.fill_(1.0)
            start.record()
            for _ in range(event_repetitions):
                run_once()
            stop.record()
            stop.synchronize()
            event_samples.append(
                float(start.elapsed_time(stop)) / event_repetitions)

    validation = validate()
    if (not contract["early_return_prefix_only"]
            and any(value == 0.0 for value in validation.values())):
        raise RuntimeError("operand-stream output validation remained zero")
    event_median_ms = (
        sorted(event_samples)[len(event_samples) // 2]
        if event_samples else 0.0)
    return {
        "mode": "profile_once" if profile_once else "event",
        "event_trials": len(event_samples),
        "event_repetitions": event_repetitions,
        "event_median_ms": event_median_ms,
        "operand_pipeline_capacity_bytes_per_s": (
            contract["expected_operand_l2_read_bytes"]
            / (event_median_ms / 1000.0)
            if event_samples else 0.0),
        **validation,
    }


def _public_contract(contract):
    return {
        key: value for key, value in contract.items()
        if key != "route_layout"
    }


def _print_csv(contract, result):
    public = _public_contract(contract)
    fields = tuple(public) + tuple(result)
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow((
        "metadata", SCHEMA_VERSION, HARDWARE_ID, MECHANISM,
        "production_or_evaluation_duration_consumed=0"))
    writer.writerow(("sample_header",) + fields)
    writer.writerow(("sample",) + tuple(
        public[field] if field in public else result[field]
        for field in fields))
    print(output.getvalue(), end="", flush=True)


def main(argv=None):
    audit_production_source()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signature", choices=tuple(SUPPORTED_SIGNATURES), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--source-topk", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--route-seed", type=int, default=DEFAULT_ROUTE_SEED)
    parser.add_argument("--mul-routed-weight", action="store_true")
    parser.add_argument("--production-kernel", action="store_true")
    parser.add_argument(
        "--weight-only-source-order", action="store_true",
        help=(
            "Isolate only the source-equivalent B descriptor stream and its "
            "grouped PID order; excludes A, dot, routing, and output work."))
    parser.add_argument(
        "--early-return-prefix-only", action="store_true",
        help=(
            "Run only the source PID mapping, scalar post-padded-count load, "
            "and early-return predicate; excludes routing, A/B, dot, and "
            "production output."))
    parser.add_argument(
        "--all-empty-prefix-control", action="store_true",
        help=(
            "With --early-return-prefix-only, set the runtime post-padded "
            "count to zero while preserving the compiled source grid."))
    parser.add_argument(
        "--refresh-activation-before-trial", action="store_true",
        help=(
            "Queue a full activation-buffer write immediately before each "
            "timed launch, matching a grouped W2 source whose activation "
            "input is produced by the preceding elementwise stage."))
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--profile-once", action="store_true")
    parser.add_argument("--attach-wait-seconds", type=int, default=5)
    parser.add_argument(
        "--event-trials", type=int, default=DEFAULT_EVENT_TRIALS)
    parser.add_argument("--event-repetitions", type=int, default=1)
    parser.add_argument(
        "--num-warps", type=int, default=None,
        help=(
            "Pin the launch num_warps instead of the compiler default; used "
            "to reproduce the deployed AOT launch configuration exactly."))
    parser.add_argument(
        "--num-stages", type=int, default=None,
        help=(
            "Pin the launch num_stages instead of the compiler default; used "
            "to reproduce the deployed AOT launch configuration exactly."))
    args = parser.parse_args(argv)
    contract = build_contract(
        args.signature, args.tokens, args.experts, args.topk, args.source_topk,
        args.k, args.n, args.route_seed, args.mul_routed_weight,
        args.production_kernel, args.refresh_activation_before_trial,
        args.weight_only_source_order, args.early_return_prefix_only,
        args.all_empty_prefix_control, args.num_warps, args.num_stages)
    if args.describe:
        print(json.dumps(_public_contract(contract), indent=2, sort_keys=True))
        return 0
    result = _run(
        contract, args.profile_once, args.attach_wait_seconds,
        args.event_trials, args.event_repetitions)
    _print_csv(contract, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
