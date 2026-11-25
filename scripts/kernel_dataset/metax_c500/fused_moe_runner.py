"""Standalone vLLM-equivalent MetaX fused-MoE runner without importing vLLM."""

import argparse
import heapq
import json
import math
from pathlib import Path
import sys
import time

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    add_event_cli_arguments,
    measure_graph_replay_events,
    parse_event_run_options,
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
    write_event_case_record,
)
from dataset_pipeline.kernel_dataset.metax_c500_fused_moe_kernels import (
    fused_moe_kernel,
    fused_moe_kernel_w2,
)


def moe_align_block_size(topk_ids, block_size, num_experts):
    num_tokens = topk_ids.numel()
    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    sorted_ids = torch.full((max_num_tokens_padded,), num_tokens,
                            dtype=torch.int32, device=topk_ids.device)
    expert_ids_out = torch.zeros(
        (triton.cdiv(max_num_tokens_padded, block_size),),
        dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.zeros(1, dtype=torch.int32,
                                      device=topk_ids.device)

    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32,
                                    device='cpu')
    flat_ids = topk_ids.view(-1).cpu()
    for i in range(flat_ids.numel()):
        tokens_per_expert[flat_ids[i].item()] += 1

    cumsum = 0
    expert_start = []
    for e in range(num_experts):
        expert_start.append(cumsum)
        cnt = tokens_per_expert[e].item()
        padded = ((cnt + block_size - 1) // block_size) * block_size
        cumsum += padded

    num_tokens_post_pad[0] = cumsum

    fill_idx = list(expert_start)
    sorted_ids_cpu = sorted_ids.cpu()
    for i in range(flat_ids.numel()):
        e = flat_ids[i].item()
        sorted_ids_cpu[fill_idx[e]] = i
        fill_idx[e] += 1

    expert_ids_cpu = expert_ids_out.cpu()
    block_idx = 0
    for e in range(num_experts):
        cnt = tokens_per_expert[e].item()
        padded = ((cnt + block_size - 1) // block_size) * block_size
        num_blocks = padded // block_size
        for _ in range(num_blocks):
            expert_ids_cpu[block_idx] = e
            block_idx += 1

    sorted_ids = sorted_ids_cpu.to(topk_ids.device)
    expert_ids_out = expert_ids_cpu.to(topk_ids.device)
    num_tokens_post_pad = num_tokens_post_pad.to(topk_ids.device)

    return sorted_ids, expert_ids_out, num_tokens_post_pad
def get_default_config(M, E):
    """Tile config matching vLLM 0.10.2's get_default_config (the version in
    the maca 3.2.1 serving container). Verified 2026-07-16: vLLM 0.10.2
    returns BM=16,BN=32,BK=64,GSM=1 when M<=E and
    BM=64,BN=64,BK=32,GSM=8 otherwise.  The broader tiered 16/32/64/128 logic
    in newer vLLM versions is not what this frozen runner uses.
    """
    if M <= E:
        return {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
    }
def invoke_fused_moe_kernel(A, B, C, topk_weights, sorted_token_ids,
                            expert_ids, num_tokens_post_padded,
                            mul_routed_weight, top_k, config, projection="w1"):
    M = A.size(0)
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0),
                 A.size(0) * top_k * config['BLOCK_SIZE_M'])

    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) *
                         triton.cdiv(B.size(1), META['BLOCK_SIZE_N']),)

    BLOCK_SIZE_K = config["BLOCK_SIZE_K"]
    cfg = {k: v for k, v in config.items() if k != "BLOCK_SIZE_K"}

    # Select kernel by projection so mcProfiler attributes w1/w2 separately
    # (they share code but have distinct names → distinct profiler reports).
    if projection == "w1":
        _kernel = fused_moe_kernel
    elif projection == "w2":
        _kernel = fused_moe_kernel_w2
    else:
        raise ValueError(f"unknown FusedMoE projection: {projection!r}")
    _kernel[grid](
        A, B, C,
        None,  # b_bias
        None,  # a_scale
        None,  # b_scale
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.size(1), B.size(2), EM, M * top_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        0, 0,  # a_scale strides
        0, 0, 0,  # b_scale strides
        0, 0,  # bias strides
        0, 0,  # group_n, group_k
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        HAS_BIAS=False,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        **cfg,
    )
def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                  inplace=False, activation='silu'):
    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    top_k_num = topk_ids.size(1)

    config = get_default_config(num_tokens, E)

    intermediate_cache1 = torch.empty(
        (num_tokens, top_k_num, N), device=hidden_states.device,
        dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty(
        (num_tokens * top_k_num, N // 2), device=hidden_states.device,
        dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty(
        (num_tokens, top_k_num, K), device=hidden_states.device,
        dtype=hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)

    # Stage 1: hidden @ w1 (with router weight on input)
    invoke_fused_moe_kernel(
        hidden_states, w1, intermediate_cache1,
        topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True, top_k=top_k_num, config=config,
        projection="w1")

    # SiLU activation with gate multiply (pure PyTorch replacement)
    ic1_flat = intermediate_cache1.view(-1, N)
    gate = ic1_flat[:, :N // 2]
    up = ic1_flat[:, N // 2:]
    intermediate_cache2.copy_(torch.nn.functional.silu(gate) * up)

    # Stage 2: activated @ w2
    invoke_fused_moe_kernel(
        intermediate_cache2, w2, intermediate_cache3,
        topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False, top_k=1, config=config,
        projection="w2")

    # moe_sum: sum across top_k dimension
    out = intermediate_cache3.sum(dim=1)
    return out


def prepare_fused_experts_gpu_sequence(
        hidden_states, w1, w2, topk_weights, topk_ids):
    """Prepare a graphable replay of the canonical GPU kernel sequence.

    The standalone production call performs CPU routing between three GPU fill
    kernels and the two expert projections.  The routing result is invariant
    for fixed inputs, so it is computed once here.  Replay emits the same three
    fill kernels and the same projection/activation/reduction kernels while
    excluding the intervening CPU loops, synchronization, and host copies.
    """

    num_tokens = hidden_states.size(0)
    num_experts, projection_width, _ = w1.size()
    hidden_size = w2.size(1)
    top_k = topk_ids.size(1)
    config = get_default_config(num_tokens, num_experts)
    intermediate_cache1 = torch.empty(
        (num_tokens, top_k, projection_width), device=hidden_states.device,
        dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty(
        (num_tokens * top_k, projection_width // 2),
        device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty(
        (num_tokens, top_k, hidden_size), device=hidden_states.device,
        dtype=hidden_states.dtype)
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], num_experts))
    max_num_tokens_padded = (
        topk_ids.numel()
        + num_experts * (config["BLOCK_SIZE_M"] - 1))
    expert_id_count = triton.cdiv(
        max_num_tokens_padded, config["BLOCK_SIZE_M"])

    def operation():
        fill_sorted_ids = torch.full(
            (max_num_tokens_padded,), topk_ids.numel(),
            dtype=torch.int32, device=topk_ids.device)
        fill_expert_ids = torch.zeros(
            (expert_id_count,), dtype=torch.int32, device=topk_ids.device)
        fill_num_tokens = torch.zeros(
            1, dtype=torch.int32, device=topk_ids.device)
        invoke_fused_moe_kernel(
            hidden_states, w1, intermediate_cache1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded, mul_routed_weight=True, top_k=top_k,
            config=config, projection="w1")
        flattened = intermediate_cache1.view(-1, projection_width)
        gate = flattened[:, :projection_width // 2]
        up = flattened[:, projection_width // 2:]
        intermediate_cache2.copy_(torch.nn.functional.silu(gate) * up)
        invoke_fused_moe_kernel(
            intermediate_cache2, w2, intermediate_cache3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded, mul_routed_weight=False, top_k=1,
            config=config, projection="w2")
        output = intermediate_cache3.sum(dim=1)
        return output, fill_sorted_ids, fill_expert_ids, fill_num_tokens

    return operation


def routing_ids_from_expert_counts(token_counts, num_tokens, top_k):
    """Construct deterministic valid top-k rows with exact expert counts."""

    if (not isinstance(token_counts, list) or not token_counts
            or any(isinstance(count, bool) or not isinstance(count, int)
                   or count < 0 for count in token_counts)):
        raise ValueError("tok_per_expert must be a non-empty integer list")
    if sum(token_counts) != num_tokens * top_k:
        raise ValueError(
            "tok_per_expert sum must equal num_tokens * top_k")
    if max(token_counts) > num_tokens:
        raise ValueError(
            "one expert cannot occur more than once per token")
    heap = [(-count, expert) for expert, count in enumerate(token_counts)
            if count]
    heapq.heapify(heap)
    rows = []
    for _ in range(num_tokens):
        if len(heap) < top_k:
            raise ValueError(
                "tok_per_expert cannot form distinct top-k experts per token")
        selected = [heapq.heappop(heap) for _ in range(top_k)]
        rows.append([expert for _, expert in selected])
        for negative_count, expert in selected:
            remaining = negative_count + 1
            if remaining < 0:
                heapq.heappush(heap, (remaining, expert))
    if heap:
        raise ValueError("tok_per_expert routing construction left work")
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def run_fused_moe(T, E, TOPK, H, I, dtype, tiling_path=None, profiler_wait=0,
                  use_marker=False, event_options=None,
                  graph_repetitions=None, canonical_token_counts=None):
    if event_options is not None:
        validate_cache_eviction_mib(
            event_options.cache_eviction_mib, minimum_nonzero_mib=64,
            hardware_id="metax_c500")
    if dtype != "bf16":
        raise ValueError(
            f"MetaX standalone FusedMoE only supports bf16, got {dtype!r}")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    hidden = torch.randn(T, H, dtype=torch.bfloat16, device='cuda')
    router_logits = torch.randn(
        T, E, dtype=torch.float32, device='cuda')
    routing_weights = torch.softmax(router_logits, dim=-1)
    tw, ti = torch.topk(routing_weights, TOPK, dim=-1)
    tw = tw.to(torch.bfloat16)
    ti = ti.to(torch.int32)
    if canonical_token_counts is not None:
        if len(canonical_token_counts) != E:
            raise ValueError(
                "tok_per_expert length must equal num_experts")
        ti = routing_ids_from_expert_counts(
            canonical_token_counts, T, TOPK)
    w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device='cuda')
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device='cuda')
    token_counts = [int((ti == expert).sum().item()) for expert in range(E)]

    # Extract tiling info
    if tiling_path:
        config = get_default_config(T, E)
        BM = config["BLOCK_SIZE_M"]
        BN = config["BLOCK_SIZE_N"]
        BK = config["BLOCK_SIZE_K"]
        GSM = config["GROUP_SIZE_M"]

        sorted_ids, _, _ = moe_align_block_size(ti, BM, E)
        sorted_sz = sorted_ids.size(0)
        EM1 = sorted_sz
        if T < BM:
            EM1 = min(sorted_sz, T * TOPK * BM)
        EM2 = sorted_sz
        if T * TOPK < BM:
            EM2 = min(sorted_sz, T * TOPK * BM)
        N1, N2 = 2 * I, H
        tiling = {
            "w1": {"N": N1, "K": H, "BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN,
                   "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GSM, "EM": EM1,
                   "grid": math.ceil(EM1 / BM) * math.ceil(N1 / BN)},
            "w2": {"N": N2, "K": I, "BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN,
                   "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GSM, "EM": EM2,
                   "grid": math.ceil(EM2 / BM) * math.ceil(N2 / BN)},
            "tok_per_expert": token_counts,
        }
        with open(tiling_path, "w", encoding="utf-8") as f:
            json.dump(tiling, f, indent=2)

    def operation():
        with torch.no_grad():
            return fused_experts(
                hidden, w1, w2, tw, ti, inplace=False, activation='silu')

    if event_options is not None:
        if use_marker or profiler_wait != 0 or tiling_path is not None:
            raise ValueError(
                "MetaX Event mode cannot enable profiler or tiling options")
        if graph_repetitions is None:
            raise ValueError(
                "MetaX FusedMoE Event mode requires graph repetitions and "
                "the preprepared GPU sequence")
        cache_eviction_operation, cache_eviction_bytes = (
            prepare_bf16_cache_eviction_sweep(
                torch_module=torch, device="cuda",
                cache_eviction_mib=event_options.cache_eviction_mib))
        gpu_sequence = prepare_fused_experts_gpu_sequence(
            hidden, w1, w2, tw, ti)
        measurement = measure_graph_replay_events(
            gpu_sequence, device_module=torch.cuda,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            repetition_counts=graph_repetitions,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes)
        timer_mode = "external_events_around_cuda_graph_replay"
        sequence_evidence = (
            "fixed routing metadata and workspaces are prepared before "
            "capture; replay preserves the canonical three fill, two "
            "expert projection, activation/multiply/copy, and reduction "
            "GPU kernel sequence while excluding CPU routing and copies")
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        write_event_case_record(
            options=event_options, hardware_id="metax_c500",
            family="fused_moe", runner_path=Path(__file__),
            timer={
                "api": "torch.cuda.Event", "returned_unit": "millisecond",
                "implementation": "wcudaEvent_to_maca_mcEvent",
                "nanoseconds_per_unit": 1_000_000,
                "cupti_identity": "not_proven", "mode": timer_mode,
                "cache_condition": (
                    "queued_pre_replay_device_sweep_with_intra_replay_reuse"
                    if cache_eviction_bytes else "ordinary_warm_state")},
            stream_contract={
                "event_stream": "torch.cuda.current_stream",
                "all_op_streams_joined": True,
                "evidence": (
                    "the ordered Triton/PyTorch GPU launch sequence uses the "
                    "current MetaX stream; the runner creates no secondary "
                    f"stream; {sequence_evidence}")},
            allocation_contract={
                "inputs_preallocated": True,
                "output_behavior": "graph_capture_pool",
                "workspace_behavior": "preallocated_for_gpu_sequence",
                "evidence": (
                    "hidden states, routing inputs and expert weights are "
                    "constructed before timing; graph mode also precomputes "
                    "CPU routing and workspaces and retains replay outputs "
                    "through the end Event")},
            workload={
                "TOK": T, "E": E, "TOPK": TOPK, "H": H, "INT": I,
                "dtype": dtype, "tok_per_expert": token_counts,
                "cache_eviction_bytes_before_replay": cache_eviction_bytes},
            environment={
                "torch_version": torch.__version__,
                "torch_cuda_version_field": torch.version.cuda,
                "triton_version": triton.__version__, "device_index": device_index,
                "device_name": properties.name,
                "device_total_memory_bytes": properties.total_memory,
                "input_seed": 42},
            measurement=measurement)
        return None

    # Pre-compile Triton kernels before profiler window
    torch.cuda.synchronize()
    _ = operation()
    torch.cuda.synchronize()
    print("Triton compilation done, waiting for profiler...")

    if profiler_wait > 0:
        time.sleep(profiler_wait)

    # Warmup (kernels already compiled)
    _ = operation()
    torch.cuda.synchronize()
    time.sleep(0.2)

    # Profiled run. Layer-1 mcProfiler uses the marker window to exclude
    # tensor construction, Triton compilation, and warmup launches.
    if use_marker:
        torch.cuda.profiler.start()
    _ = operation()
    torch.cuda.synchronize()
    if use_marker:
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_tokens", type=int, required=True)
    p.add_argument("--num_experts", type=int, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--hidden_size", type=int, required=True)
    p.add_argument("--intermediate_size", type=int, required=True)
    p.add_argument("--dtype", required=True, choices=("bf16",))
    p.add_argument("--tiling_path", type=str, default=None)
    p.add_argument("--profiler_wait", type=int, default=0)
    p.add_argument("--use_marker", action="store_true")
    add_event_cli_arguments(p)
    p.add_argument("--event-graph-repetitions", nargs=2, type=int)
    p.add_argument("--tok-per-expert-json")
    args = p.parse_args()
    event_options = parse_event_run_options(args)
    graph_repetitions = None
    if args.event_graph_repetitions is not None:
        if event_options is None:
            raise ValueError(
                "event_graph_repetitions requires complete Event options")
        graph_repetitions = tuple(args.event_graph_repetitions)
    canonical_token_counts = None
    if args.tok_per_expert_json is not None:
        canonical_token_counts = json.loads(args.tok_per_expert_json)
    run_fused_moe(args.num_tokens, args.num_experts, args.top_k,
                  args.hidden_size, args.intermediate_size,
                  args.dtype, args.tiling_path, args.profiler_wait,
                  args.use_marker, event_options, graph_repetitions,
                  canonical_token_counts)
