"""Standalone vLLM-Ascend fused-MoE runner for 910B dataset profiling.

Replicates the vLLM-Ascend unquantized fused-MoE serving path:
  gating (softmax topk) -> npu_moe_init_routing (sort tokens by expert, build
  group_list) -> npu_grouped_matmul(w1) -> npu_swiglu -> npu_grouped_matmul(w2)
  -> npu_moe_finalize_routing (scatter back, weighted sum).

This is the SAME kernel sequence vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp
issues (verified against source), driven standalone so msprof op can profile each
kernel. Fixed seed so routing (hence group_list / grid) is reproducible across the
warmup and profiled launches.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch_npu


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.device_event import (
    add_event_cli_arguments,
    measure_current_stream_events,
    parse_event_run_options,
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
    write_event_case_record,
)

DTYPE_MAP = {
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}


def _topk_rows_from_tokens_per_expert(
        tokens_per_expert, tokens, experts, topk):
    """Build one legal top-k route with an exact per-expert row census.

    This is a launch-observation utility. It lets a source-only capture use
    the canonical routing topology without reading an operator duration or
    manufacturing a random routing distribution. Every token receives
    distinct expert IDs, matching the semantic contract of top-k routing.
    """

    if not isinstance(tokens_per_expert, list):
        raise TypeError("tokens_per_expert must be a JSON list")
    if len(tokens_per_expert) != experts:
        raise ValueError(
            "tokens_per_expert length must equal experts: "
            f"{len(tokens_per_expert)} != {experts}")
    if (any(isinstance(count, bool) or not isinstance(count, int)
            for count in tokens_per_expert)
            or any(count < 0 for count in tokens_per_expert)):
        raise TypeError("tokens_per_expert values must be non-negative ints")
    if sum(tokens_per_expert) != tokens * topk:
        raise ValueError(
            "tokens_per_expert must sum to tokens * topk: "
            f"{sum(tokens_per_expert)} != {tokens * topk}")
    if topk > experts:
        raise ValueError("topk cannot exceed experts")
    if max(tokens_per_expert) > tokens:
        raise ValueError(
            "tokens_per_expert cannot assign one expert more than once "
            "per token")

    remaining = list(tokens_per_expert)
    rows = []
    for token_index in range(tokens):
        ordered = sorted(range(experts), key=lambda expert: (
            -remaining[expert], expert))
        selected = ordered[:topk]
        if any(remaining[expert] <= 0 for expert in selected):
            raise ValueError(
                "tokens_per_expert cannot form a legal distinct top-k route "
                f"at token {token_index}")
        for expert in selected:
            remaining[expert] -= 1
        rows.append(selected)
    if any(remaining):
        raise ValueError("tokens_per_expert route construction left rows")
    return rows


def _topk_ids_from_tokens_per_expert(
        tokens_per_expert, tokens, experts, topk):
    """Materialize an exact-census route on the NPU for launch capture."""

    rows = _topk_rows_from_tokens_per_expert(
        tokens_per_expert, tokens, experts, topk)
    return torch.tensor(rows, dtype=torch.int32, device="npu")


def _build_moe_tensors(T, E, TOPK, H, I, dtype):
    hidden = torch.randn(T, H, dtype=dtype, device="npu")
    # Router logits -> softmax topk gating (mirrors select_experts)
    router_logits = torch.randn(T, E, dtype=torch.float32, device="npu")
    routing = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(routing, TOPK, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    # Expert weights ALREADY in npu_grouped_matmul layout (no per-forward
    # transpose): w1 = gate+up [E, H, 2I], w2 = down [E, I, H].
    w1 = torch.randn(E, H, 2 * I, dtype=dtype, device="npu")
    w2 = torch.randn(E, I, H, dtype=dtype, device="npu")
    return hidden, router_logits, topk_weights, topk_ids, w1, w2


def _routing(hidden, topk_ids, E, T, TOPK):
    """init_routing: sort tokens by expert, produce expanded rows + group_list."""
    row_idx = torch.arange(T * TOPK, dtype=torch.int32, device="npu").view(TOPK, T).transpose(0, 1).contiguous()
    expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
        hidden, row_idx=row_idx, expert_idx=topk_ids, active_num=T * TOPK)
    # tokens per expert -> group_list (cumulative counts, group_list_type=1 -> counts)
    # npu_moe_compute_expert_tokens returns CUMULATIVE token counts per expert
    # -> consumed by npu_grouped_matmul with group_list_type=0 (cumsum form).
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, E)
    return expanded_x, expanded_row_idx, expert_tokens.to(torch.int64)


def run_fused_moe(T, E, TOPK, H, I, dtype="bf16", tiling_path=None,
                  event_options=None, expected_token_counts=None,
                  profile_cache_eviction_mib=0,
                  serving_exact_public_call=False,
                  tokens_per_expert=None):
    validate_cache_eviction_mib(
        profile_cache_eviction_mib, minimum_nonzero_mib=1536,
        hardware_id="ascend_910b")
    if event_options is not None:
        if profile_cache_eviction_mib != 0:
            raise ValueError(
                "profile_cache_eviction_mib cannot be mixed with Event mode")
        validate_cache_eviction_mib(
            event_options.cache_eviction_mib, minimum_nonzero_mib=1536,
            hardware_id="ascend_910b")
    torch.npu.set_device(0)
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    dt = DTYPE_MAP[dtype.lower()]

    hidden, router_logits, topk_weights, topk_ids, w1, w2 = (
        _build_moe_tensors(T, E, TOPK, H, I, dt))
    if tokens_per_expert is not None:
        if serving_exact_public_call:
            raise ValueError(
                "tokens_per_expert is only valid for the explicit "
                "GroupedMatmul route")
        topk_ids = _topk_ids_from_tokens_per_expert(
            tokens_per_expert, T, E, TOPK)
    # NOTE: npu_grouped_matmul internally reformats ND weights to FRACTAL_NZ on
    # first use, emitting a one-time Cast/TransData reading the full weight bytes.
    # In real serving this is amortized at model load (maybe_trans_nz). We do NOT
    # pre-cast with npu_format_cast (it needs TBE JIT which fails under msprof op).
    # Instead the runner does many warmup forwards and build_910b's high
    # moe_warmup_skip drops those first cold iterations so msprof captures only
    # steady-state kernels.

    def _legacy_forward():
        expanded_x, expanded_row_idx, group_list = _routing(hidden, topk_ids, E, T, TOPK)
        gate_up = torch_npu.npu_grouped_matmul(
            x=[expanded_x], weight=[w1], split_item=2,
            group_list_type=0, group_type=0, group_list=group_list)[0]
        act = torch_npu.npu_swiglu(gate_up)
        down = torch_npu.npu_grouped_matmul(
            x=[act], weight=[w2], split_item=2,
            group_list_type=0, group_type=0, group_list=group_list)[0]
        out = torch_npu.npu_moe_finalize_routing(
            down, skip1=None, skip2=None, bias=None,
            scales=topk_weights.to(dt), expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids)
        return out, group_list, topk_ids

    def _serving_exact_topk():
        return torch_npu.npu_moe_gating_top_k(
            router_logits, k=TOPK, k_group=1, group_count=1,
            group_select_mode=1, renorm=0, norm_type=0,
            routed_scaling_factor=1.0, eps=1e-20)

    def _serving_exact_forward():
        """Mirror the eager vLLM-Ascend TP1 MoE task sequence.

        The legacy standalone path precomputed a torch softmax/topk outside the
        public call, used the original routing op, and finalized with
        ``npu_moe_finalize_routing``.  Eager Serving instead emits
        MoeGatingTopK -> MoeInitRoutingCustom -> GMM -> SwiGlu -> GMM -> Abs
        -> MoeTokenUnpermute.  Keep this explicit route Timeline-only so
        existing Event/canonical collection semantics are unchanged.
        """
        runtime_weights, runtime_ids, _ = _serving_exact_topk()
        # The eager Qwen Serving trace has no post-gating ReduceSum/Div tasks:
        # this routing configuration consumes the softmax top-k probabilities
        # directly, without a separate top-k renormalization kernel sequence.
        expanded_x, expanded_row_idx, group_list, _ = (
            torch_npu.npu_moe_init_routing_v2(
                hidden, runtime_ids.to(torch.int32),
                active_num=T * TOPK, expert_num=E,
                expert_tokens_num_type=1, expert_tokens_num_flag=True,
                active_expert_range=[0, E], quant_mode=-1))
        group_list = group_list.to(torch.int64)
        gate_up = torch_npu.npu_grouped_matmul(
            x=[expanded_x], weight=[w1], split_item=2,
            group_list_type=1, group_type=0, group_list=group_list)[0]
        act = torch_npu.npu_swiglu(gate_up)
        down = torch_npu.npu_grouped_matmul(
            x=[act], weight=[w2], split_item=2,
            group_list_type=1, group_type=0, group_list=group_list)[0]
        out = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=down,
            sorted_indices=torch.abs(expanded_row_idx),
            probs=runtime_weights)
        return out, group_list, runtime_ids

    if serving_exact_public_call:
        if event_options is not None:
            raise ValueError(
                "serving_exact_public_call is Timeline-only and cannot use Event")
        forward = _serving_exact_forward
    else:
        forward = _legacy_forward

    if serving_exact_public_call:
        # The tiling sidecar is a provenance observation, not part of the
        # final profiled public call.  Materialize it before runner warmup so
        # its scalar reductions cannot extend the Timeline suffix.
        _, setup_topk_ids, _ = _serving_exact_topk()
        torch.npu.synchronize()
        token_counts = [
            int((setup_topk_ids == expert).sum().item())
            for expert in range(E)]
    else:
        token_counts = [
            int((topk_ids == expert).sum().item()) for expert in range(E)]
    if event_options is not None:
        if expected_token_counts is None:
            raise ValueError(
                "Ascend FusedMoE Event mode requires canonical routing counts")
        if token_counts != expected_token_counts:
            raise ValueError(
                "Ascend FusedMoE routing does not match canonical Attr: "
                f"generated={token_counts}, canonical={expected_token_counts}")
        if tiling_path is not None:
            raise ValueError("Ascend FusedMoE Event mode cannot write tiling")
        cache_eviction_operation, cache_eviction_bytes = (
            prepare_bf16_cache_eviction_sweep(
                torch_module=torch, device="npu",
                cache_eviction_mib=event_options.cache_eviction_mib))
        measurement = measure_current_stream_events(
            forward,
            device_module=torch.npu,
            warmup_iterations=event_options.warmup_iterations,
            sample_count=event_options.sample_count,
            cache_eviction_operation=cache_eviction_operation,
            cache_eviction_bytes=cache_eviction_bytes,
        )
        device_index = torch.npu.current_device()
        properties = torch.npu.get_device_properties(device_index)
        write_event_case_record(
            options=event_options,
            hardware_id="ascend_910b",
            family="fused_moe",
            runner_path=Path(__file__),
            timer={
                "api": "torch.npu.Event",
                "implementation": "ACL_Runtime_Event_via_torch_npu",
                "returned_unit": "millisecond",
                "nanoseconds_per_unit": 1_000_000,
                "cupti_identity": "not_applicable",
                "cache_condition": (
                    "queued_pre_sample_device_sweep"
                    if cache_eviction_bytes else "ordinary_warm_state"),
            },
            stream_contract={
                "event_stream": "torch.npu.current_stream",
                "all_op_streams_joined": True,
                "evidence": (
                    "the exact routing/grouped-matmul/activation/finalize "
                    "sequence is submitted through the current NPU stream; "
                    "the runner creates no secondary stream"),
            },
            allocation_contract={
                "inputs_preallocated": True,
                "output_behavior": "exact_runner_call",
                "workspace_behavior": "exact_runner_call",
                "evidence": (
                    "hidden states, route IDs and expert weights are built "
                    "before timing; Event mode preserves the exact _forward "
                    "sequence and retains its returned tensors through the "
                    "end Event"),
            },
            workload={
                "TOK": T, "E": E, "TOPK": TOPK, "H": H, "INT": I,
                "dtype": dtype, "tok_per_expert": token_counts,
                "cache_eviction_bytes_before_sample": cache_eviction_bytes,
            },
            environment={
                "torch_version": torch.__version__,
                "torch_npu_version": torch_npu.__version__,
                "device_index": device_index,
                "device_name": properties.name,
                "device_total_memory_bytes": properties.total_memory,
                "input_seed": 42,
            },
            measurement=measurement,
        )
        return None

    profile_sweep, _ = prepare_bf16_cache_eviction_sweep(
        torch_module=torch, device="npu",
        cache_eviction_mib=profile_cache_eviction_mib,
        sweep_operation="fill")

    # Warmup: multiple iterations so all sub-kernels (Cast/routing/grouped_matmul)
    # are compiled/autotuned and reach steady state before profiling. A single
    # warmup leaves cold-compiled Cast launches (msprof captured 4.5ms->2.2ms
    # decaying artifacts otherwise).
    torch.npu.synchronize()
    for _ in range(8):
        if profile_sweep is not None:
            profile_sweep()
        out, group_list, _ = forward()
    torch.npu.synchronize()

    if tiling_path is not None:
        import json
        gl = group_list.detach().cpu().tolist()
        with open(tiling_path, "w") as f:
            json.dump({"tok_per_expert": token_counts, "T": T, "E": E, "TOPK": TOPK,
                       "H": H, "I": I}, f, indent=2)

    # Profiled launches (steady state). msprof averages over invocations.
    for _ in range(3):
        if profile_sweep is not None:
            profile_sweep()
        out, _, _ = forward()
    torch.npu.synchronize()
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tok", type=int, required=True)
    p.add_argument("--experts", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--intermediate", type=int, required=True)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--tiling", default=None)
    p.add_argument("--profile-cache-eviction-mib", type=int, default=0)
    p.add_argument("--serving-exact-public-call", action="store_true")
    p.add_argument("--expected-tok-per-expert-json")
    p.add_argument("--tokens-per-expert-json")
    add_event_cli_arguments(p)
    a = p.parse_args()
    event_options = parse_event_run_options(a)
    if event_options is None:
        expected_token_counts = None
    else:
        if a.expected_tok_per_expert_json is None:
            raise ValueError(
                "Event mode requires --expected-tok-per-expert-json")
        expected_token_counts = json.loads(a.expected_tok_per_expert_json)
        if (not isinstance(expected_token_counts, list)
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value < 0 for value in expected_token_counts)):
            raise TypeError("expected token counts must be non-negative ints")
    tokens_per_expert = (
        None if a.tokens_per_expert_json is None
        else json.loads(a.tokens_per_expert_json))
    run_fused_moe(
        a.tok, a.experts, a.topk, a.hidden, a.intermediate, a.dtype,
        a.tiling, event_options=event_options,
        expected_token_counts=expected_token_counts,
        profile_cache_eviction_mib=a.profile_cache_eviction_mib,
        serving_exact_public_call=a.serving_exact_public_call,
        tokens_per_expert=tokens_per_expert)
    print("MOE done")
