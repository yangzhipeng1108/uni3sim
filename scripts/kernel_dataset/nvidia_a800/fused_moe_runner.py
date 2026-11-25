"""Fused-MoE runner for the A800 single-op NCU dataset (vLLM fused_experts)."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import torch
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts, try_get_optimal_moe_config)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    add_event_cli_arguments,
    build_nvidia_smi_telemetry_observer,
    measure_current_stream_events,
    parse_event_run_options,
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
    write_event_case_record,
)
from dataset_pipeline.kernel_dataset.nvidia_a800_event_capture import (
    build_nvidia_a800_moe_topk_ids,
)


DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def run_fused_moe(
    num_tokens,
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    nvtx_range_name,
    dtype="fp16",
    device="cuda",
    tiling_output=None,
    event_options=None,
    expected_token_counts=None,
    pre_nvtx_cache_eviction_mib=0,
    pre_nvtx_warmup_iterations=0,
    steady_state_public_op_repetitions=1,
):
    if (isinstance(pre_nvtx_warmup_iterations, bool)
            or not isinstance(pre_nvtx_warmup_iterations, int)
            or pre_nvtx_warmup_iterations < 0):
        raise ValueError(
            "A800 FusedMoE pre-NVTX warmup iterations must be a non-negative integer")
    if (isinstance(steady_state_public_op_repetitions, bool)
            or not isinstance(steady_state_public_op_repetitions, int)
            or steady_state_public_op_repetitions not in {1, 8, 32}):
        raise ValueError(
            "A800 FusedMoE steady-state Event repetitions must be one of 1, 8, 32")
    if event_options is not None and expected_token_counts is None:
        raise ValueError(
            "A800 FusedMoE Event mode requires canonical routing counts")
    if event_options is not None:
        validate_cache_eviction_mib(
            event_options.cache_eviction_mib, minimum_nonzero_mib=320,
            hardware_id="nvidia_a800")
    torch_dtype = DTYPE_MAP[dtype.lower()]
    canonical_topk_ids = None
    if expected_token_counts is not None:
        canonical_topk_ids = build_nvidia_a800_moe_topk_ids(
            token_counts=expected_token_counts,
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            context="A800 FusedMoE canonical routing",
        )

    # Fixed seed: NCU --set full re-launches the process multiple times.
    # Without a fixed seed, each pass generates different topk_ids → different
    # routing → different grids. The tiling info file gets overwritten by the
    # last pass but ncu-rep merges metrics across passes with inconsistent grids.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch_dtype, device=device)
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device)
    routing_weights = torch.softmax(router_logits, dim=-1)
    topk_weights, random_topk_ids = torch.topk(
        routing_weights, top_k, dim=-1)
    if canonical_topk_ids is None:
        topk_ids = random_topk_ids
    else:
        topk_ids = torch.tensor(
            canonical_topk_ids, dtype=torch.int32, device=device)
    topk_weights = topk_weights.to(torch_dtype)
    topk_ids = topk_ids.to(torch.int32)

    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=torch_dtype, device=device)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch_dtype, device=device)

    # ─── Extract tiling config using the SAME topk_ids that will be profiled ───
    if tiling_output is not None:
        tiling_info = _extract_tiling_info(
            w1, w2, topk_ids, num_tokens, num_experts, top_k,
            hidden_size, intermediate_size)
        with open(tiling_output, "w") as f:
            json.dump(tiling_info, f, indent=2)

    def operation():
        with torch.no_grad():
            return fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
        )

    if event_options is None:
        if steady_state_public_op_repetitions != 1:
            raise ValueError(
                "A800 FusedMoE steady-state Event repetitions require Event mode")
        if not isinstance(nvtx_range_name, str) or not nvtx_range_name:
            raise ValueError("normal A800 FusedMoE mode requires an NVTX name")
        validate_cache_eviction_mib(
            pre_nvtx_cache_eviction_mib, minimum_nonzero_mib=320,
            hardware_id="nvidia_a800")
        cache_eviction_operation, _ = prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="cuda",
            cache_eviction_mib=pre_nvtx_cache_eviction_mib)
        torch.cuda.synchronize()
        for _ in range(pre_nvtx_warmup_iterations):
            operation()
        torch.cuda.synchronize()
        if cache_eviction_operation is not None:
            cache_eviction_operation()
            torch.cuda.current_stream().synchronize()
        torch.cuda.nvtx.range_push(nvtx_range_name)
        output = operation()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return output
    if nvtx_range_name is not None or tiling_output is not None:
        raise ValueError(
            "A800 FusedMoE Event mode cannot enable NVTX or tiling output")
    if pre_nvtx_warmup_iterations:
        raise ValueError(
            "A800 FusedMoE Event mode cannot enable a pre-NVTX warmup")
    if pre_nvtx_cache_eviction_mib:
        raise ValueError(
            "A800 FusedMoE Event mode cannot enable a pre-NVTX cache sweep")
    token_counts = [
        int((topk_ids == expert).sum().item())
        for expert in range(num_experts)
    ]
    if token_counts != expected_token_counts:
        raise ValueError(
            "A800 FusedMoE actual routing disagrees with canonical Attr: "
            f"actual={token_counts}, canonical={expected_token_counts}")
    cache_eviction_operation, cache_eviction_bytes = (
        prepare_bf16_cache_eviction_sweep(
            torch_module=torch, device="cuda",
            cache_eviction_mib=event_options.cache_eviction_mib))
    telemetry_observer = None
    if event_options.telemetry_gpu_index is not None:
        telemetry_observer = build_nvidia_smi_telemetry_observer(
            event_options.telemetry_gpu_index)
    measurement = measure_current_stream_events(
        operation,
        device_module=torch.cuda,
        warmup_iterations=event_options.warmup_iterations,
        sample_count=event_options.sample_count,
        cache_eviction_operation=cache_eviction_operation,
        cache_eviction_bytes=cache_eviction_bytes,
        operation_repetitions_per_sample=steady_state_public_op_repetitions,
        telemetry_observer=telemetry_observer,
    )
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    write_event_case_record(
        options=event_options,
        hardware_id="nvidia_a800",
        family="fused_moe",
        runner_path=Path(__file__),
        timer={
            "api": "torch.cuda.Event",
            "implementation": "CUDA_Runtime_Event",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000,
            "cupti_identity": "not_used",
            "mode": (
                "one_complete_public_op_per_event_interval"
                if steady_state_public_op_repetitions == 1 else
                "fixed_count_complete_public_ops_in_one_current_stream_event_"
                "reported_per_op"),
            "public_op_repetition_count": steady_state_public_op_repetitions,
            "cache_condition": (
                "queued_pre_sample_device_sweep"
                if cache_eviction_bytes else "ordinary_warm_state"),
        },
        stream_contract={
            "event_stream": "torch.cuda.current_stream",
            "all_op_streams_joined": True,
            "evidence": (
                "the canonical fused_experts implementation dispatches its "
                "ordered launch graph on the current CUDA stream"),
        },
        allocation_contract={
            "inputs_preallocated": True,
            "output_behavior": "exact_runner_call",
            "workspace_behavior": "exact_runner_call",
            "evidence": (
                "hidden states, routing tensors and expert weights are built "
                "before timing; Event mode brackets the exact fused_experts "
                "call and retains its result through the end Event"),
        },
        workload={
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "top_k": top_k,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "dtype": dtype,
            "tok_per_expert": token_counts,
            "routing_source": "canonical_attr_tok_per_expert",
            "routing_assignment": (
                "deterministic_distinct_topk_max_remaining_count"),
            "cache_eviction_bytes_before_sample": cache_eviction_bytes,
            "steady_state_event_contract": {
                "public_op_repetition_count": steady_state_public_op_repetitions,
                "cache_state": (
                    "one_op_per_cleaned_event_interval"
                    if steady_state_public_op_repetitions == 1 else
                    "cache_eviction_before_aggregate_with_intra_aggregate_reuse"),
            },
        },
        environment={
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "vllm_version": importlib.metadata.version("vllm"),
            "device_index": device_index,
            "device_name": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "input_seed": 42,
            "telemetry_physical_gpu_index": event_options.telemetry_gpu_index,
        },
        measurement=measurement,
    )
    return None


def _extract_tiling_info(w1, w2, topk_ids, num_tokens, num_experts, top_k,
                         hidden_size, intermediate_size):
    """Extract the vllm triton shared tiling config + padded EM for both
    projections.

    vllm's `fused_experts` (triton) selects ONE config
    (try_get_optimal_moe_config keyed on w1/w2 shapes + M=TOK) and applies the
    SAME BLOCK_SIZE_M/N/K + GROUP_SIZE_M to both stage1 (w1) and stage2 (w2).
    We therefore autotune ONCE and derive both projections from that shared
    config — NOT a separate per-projection autotune (the old code's bug, which
    produced a w2 grid that never matched the real dispatched kernel).
    """
    N_w1 = 2 * intermediate_size  # w1 output dim
    N_w2 = hidden_size            # w2 output dim

    # Single shared config (same call the runtime makes).
    config = try_get_optimal_moe_config(
        w1.size(), w2.size(), top_k, dtype=None, M=num_tokens)
    BM = config["BLOCK_SIZE_M"]
    BN = config["BLOCK_SIZE_N"]
    BK = config["BLOCK_SIZE_K"]
    GSM = config["GROUP_SIZE_M"]

    # EM (effective padded token count). invoke_fused_moe_triton_kernel applies
    # a small-batch reduction keyed on A.size(0): TOK for w1, TOK*top_k for w2.
    sorted_ids, _, _ = moe_align_block_size(topk_ids, BM, num_experts, None)
    sorted_sz = sorted_ids.size(0)

    EM_w1 = sorted_sz
    if num_tokens < BM:
        EM_w1 = min(sorted_sz, num_tokens * top_k * BM)

    EM_w2 = sorted_sz
    a_size_w2 = num_tokens * top_k
    if a_size_w2 < BM:
        EM_w2 = min(sorted_sz, num_tokens * top_k * BM)

    # Predicted grids
    grid_w1 = math.ceil(EM_w1 / BM) * math.ceil(N_w1 / BN)
    grid_w2 = math.ceil(EM_w2 / BM) * math.ceil(N_w2 / BN)

    # Per-expert token distribution
    tok_per_expert = []
    for e in range(num_experts):
        tok_per_expert.append(int((topk_ids == e).sum().item()))

    return {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "w1": {
            "N": N_w1, "K": hidden_size,
            "BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GSM,
            "EM": EM_w1, "grid": grid_w1,
        },
        "w2": {
            "N": N_w2, "K": intermediate_size,
            "BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GSM,
            "EM": EM_w2, "grid": grid_w2,
        },
        "tok_per_expert": tok_per_expert,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, required=True)
    parser.add_argument("--num_experts", type=int, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--intermediate_size", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--nvtx_range_name", type=str)
    parser.add_argument("--tiling_output", type=str, default=None,
                        help="Path to write tiling info JSON (same run as NCU)")
    parser.add_argument("--pre-nvtx-cache-eviction-mib", type=int, default=0)
    parser.add_argument("--pre-nvtx-warmup-iterations", type=int, default=0)
    parser.add_argument(
        "--event-steady-state-public-op-repetitions", type=int, default=1)
    parser.add_argument("--expected-tok-per-expert-json")
    add_event_cli_arguments(parser)

    args = parser.parse_args()
    event_options = parse_event_run_options(args)
    if event_options is None:
        expected_token_counts = (
            None
            if args.expected_tok_per_expert_json is None
            else json.loads(args.expected_tok_per_expert_json))
    else:
        if args.expected_tok_per_expert_json is None:
            raise ValueError(
                "Event mode requires --expected-tok-per-expert-json")
        expected_token_counts = json.loads(args.expected_tok_per_expert_json)
    torch.cuda.set_device(0)

    run_fused_moe(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        nvtx_range_name=args.nvtx_range_name,
        dtype=args.dtype,
        tiling_output=args.tiling_output,
        event_options=event_options,
        expected_token_counts=expected_token_counts,
        pre_nvtx_cache_eviction_mib=args.pre_nvtx_cache_eviction_mib,
        pre_nvtx_warmup_iterations=args.pre_nvtx_warmup_iterations,
        steady_state_public_op_repetitions=(
            args.event_steady_state_public_op_repetitions),
    )


if __name__ == "__main__":
    main()
