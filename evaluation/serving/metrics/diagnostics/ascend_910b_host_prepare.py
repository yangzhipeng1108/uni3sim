"""Source-authored 910B eager runtime-prepare occurrence candidates.

This module only produces diagnostic event positions.  It does not contain
durations and does not make a candidate eligible for production Forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Tuple

from src.kernel.kernel_spec import KernelKind
from src.serving.host_timing import RuntimePrepareEvent
from src.serving.runtime_prepare import (
    ascend_910b_rotary_cache_update_events as _production_rotary_cache_update_events,
)
from src.serving.kernel_lowering.execution_timeline import call_host_lanes
from src.serving.kernel_lowering.kernel_plan import KernelPlan


@dataclass(frozen=True)
class AscendRuntimePrepareRule:
    kernel_kind: KernelKind
    variants: Tuple[str, ...]
    mechanisms: Tuple[str, ...]
    source_path: str
    source_symbol: str


ASCEND_910B_RUNTIME_PREPARE_RULES = (
    AscendRuntimePrepareRule(
        KernelKind.VECTOR,
        ("qkv_rmsnorm_rope",),
        ("triton_dynamic_argument_prepare",),
        "vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py",
        "split_qkv_rmsnorm_rope_kernel",
    ),
    AscendRuntimePrepareRule(
        KernelKind.VECTOR,
        ("moe_gating_topk",),
        ("stream_record_event_prepare", "moe_gating_prepare"),
        "vllm_ascend/device/device_op.py",
        "BaseDeviceAdaptor.moe_gating_top_k",
    ),
    AscendRuntimePrepareRule(
        KernelKind.VECTOR,
        ("moe_init_routing_v2",),
        ("stream_record_event_prepare", "moe_token_dispatch_prepare"),
        "vllm_ascend/ops/fused_moe/token_dispatcher.py",
        "TokenDispatcherWithAllGather.token_dispatch",
    ),
    AscendRuntimePrepareRule(
        KernelKind.FUSED_MOE,
        ("w1", "w2"),
        ("moe_grouped_gemm_prepare",),
        "vllm_ascend/ops/fused_moe/moe_mlp.py",
        "unquant_apply_mlp",
    ),
    AscendRuntimePrepareRule(
        KernelKind.GEMM,
        ("gemm",),
        ("unquantized_gemm_prepare",),
        "vllm_ascend/patch/worker/patch_unquantized_gemm.py",
        "unquantized_gemm",
    ),
    AscendRuntimePrepareRule(
        KernelKind.VECTOR,
        ("moe_swiglu",),
        ("moe_swiglu_prepare",),
        "vllm_ascend/ops/fused_moe/moe_mlp.py",
        "unquant_apply_mlp",
    ),
    AscendRuntimePrepareRule(
        KernelKind.VECTOR,
        ("moe_token_unpermute",),
        ("stream_record_event_prepare", "moe_token_combine_prepare"),
        "vllm_ascend/ops/fused_moe/token_dispatcher.py",
        "TokenDispatcherWithAllGather.token_combine",
    ),
    AscendRuntimePrepareRule(
        KernelKind.KV_CACHE,
        ("paged_kv_cache",),
        ("reshape_cache_prepare",),
        "vllm_ascend/attention/attention_v1.py",
        "AscendAttentionBackendImpl.reshape_and_cache",
    ),
    AscendRuntimePrepareRule(
        KernelKind.FLASH_ATTN,
        ("prefill", "decode"),
        ("python_argument_prepare",),
        "vllm_ascend/attention/attention_v1.py",
        "AscendAttentionBackendImpl._get_fia_params",
    ),
)


_MOE_WRAPPER_ANCHOR = re.compile(
    r":layer-([01]):moe:ep-rank-([01]):0$")
_MOE_WRAPPER_MECHANISM = "compiled_fx_moe_wrapper_control"
_MOE_UNQUANT_MLP_MECHANISM = "moe_unquant_mlp_wrapper_prepare"
_QKV_TRITON_COMPOSITE_MECHANISM = "triton_three_output_composite_prepare"
_FIA_SOURCE_WRAPPER_MECHANISM = "fused_attention_source_wrapper_prepare"
_MODEL_RUNNER_INPUT_PREPARE_MECHANISM = "model_runner_input_prepare"
_MODEL_RUNNER_FORWARD_METADATA_MECHANISM = (
    "model_runner_forward_metadata_prepare"
)
_MODEL_RUNNER_LOGITS_PREPARE_MECHANISM = "model_runner_logits_prepare"
_MODEL_RUNNER_TOKEN_EMBEDDING_ANCHOR = re.compile(
    r"(?:^|:)token-embedding(?:\:tp-rank-([0-9]+))?$"
)
_MODEL_RUNNER_LM_HEAD_ANCHOR = re.compile(
    r"(?:^|:)lm-head(?:\:tp-rank-([0-9]+))?$"
)


def ascend_910b_model_runner_outer_events(
    plan: KernelPlan,
    *,
    enabled_ranks: Tuple[int, ...] | None = None,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Place one source-defined ModelRunner outer event set per Host lane.

    ``NPUModelRunner.execute_model`` performs input preparation, attention and
    forward-context metadata construction before invoking ``_model_forward``.
    It later prepares the logits call after the transformer forward returns.
    The compact Qwen3 plan represents these two source boundaries with its
    token-embedding and LM-head calls respectively.  They are deliberately
    step-level runtime mechanisms, not kernel-name timing lookups: the source
    anchors only establish their ordering on each concrete Host lane.

    The descriptor is intentionally restricted to the current Qwen3 TP plan.
    A plan without one token-embedding and one LM-head source anchor for every
    Host rank is not silently treated as equivalent.
    """

    if not isinstance(plan, KernelPlan):
        raise TypeError("ModelRunner outer descriptor requires a KernelPlan")
    hardware_ids = {call.kernel_spec.hardware_id for call in plan.calls}
    if hardware_ids != {"ascend_910b"}:
        raise ValueError("ModelRunner outer descriptor requires an Ascend-only plan")

    embedding_by_rank = {}
    lm_head_by_rank = {}
    embedding_labels = {}
    lm_head_labels = {}
    for call in plan.calls:
        embedding_match = _MODEL_RUNNER_TOKEN_EMBEDDING_ANCHOR.search(
            call.call_id)
        lm_head_match = _MODEL_RUNNER_LM_HEAD_ANCHOR.search(call.call_id)
        if embedding_match is not None:
            encoded_rank = embedding_match.group(1)
            if encoded_rank is not None:
                encoded_rank = int(encoded_rank)
            if call.rank in embedding_by_rank:
                raise ValueError("ModelRunner token-embedding anchor is duplicated")
            embedding_by_rank[call.rank] = call
            embedding_labels[call.rank] = encoded_rank
        if lm_head_match is not None:
            encoded_rank = lm_head_match.group(1)
            if encoded_rank is not None:
                encoded_rank = int(encoded_rank)
            if call.rank in lm_head_by_rank:
                raise ValueError("ModelRunner LM-head anchor is duplicated")
            lm_head_by_rank[call.rank] = call
            lm_head_labels[call.rank] = encoded_rank

    if not embedding_by_rank or set(embedding_by_rank) != set(lm_head_by_rank):
        raise ValueError(
            "ModelRunner outer descriptor requires matching token-embedding "
            "and LM-head anchors for every Host rank")
    embedding_label_values = tuple(embedding_labels.values())
    lm_head_label_values = tuple(lm_head_labels.values())
    embedding_ranked_forms = {
        value is not None for value in embedding_label_values
    }
    lm_head_ranked_forms = {
        value is not None for value in lm_head_label_values
    }
    if len(embedding_ranked_forms) != 1 or len(lm_head_ranked_forms) != 1:
        raise ValueError(
            "ModelRunner token-embedding and LM-head anchors must be "
            "uniformly ranked or uniformly unranked")
    if (any(value is None for value in embedding_label_values)
            != any(value is None for value in lm_head_label_values)):
        raise ValueError(
            "ModelRunner token-embedding and LM-head anchors must use the "
            "same ranked or unranked form")
    if (all(value is not None for value in embedding_label_values)
            and (set(embedding_label_values) != set(lm_head_label_values)
                 or len(set(embedding_label_values)) != len(embedding_label_values)
                 or len(set(lm_head_label_values)) != len(lm_head_label_values))):
        raise ValueError(
            "ModelRunner token-embedding and LM-head logical ranks differ")
    if enabled_ranks is None:
        selected_ranks = tuple(sorted(embedding_by_rank))
    else:
        if (not isinstance(enabled_ranks, tuple)
                or not enabled_ranks
                or any(isinstance(rank, bool) or not isinstance(rank, int)
                       or rank < 0 for rank in enabled_ranks)
                or len(set(enabled_ranks)) != len(enabled_ranks)):
            raise ValueError(
                "ModelRunner outer descriptor enabled ranks must be unique "
                "non-negative integers")
        selected_ranks = tuple(sorted(enabled_ranks))
        if set(selected_ranks) - set(embedding_by_rank):
            raise ValueError(
                "ModelRunner outer descriptor enabled ranks lack source anchors")

    events = {}
    for rank in selected_ranks:
        embedding_call = embedding_by_rank[rank]
        lm_head_call = lm_head_by_rank[rank]
        for pool_name, lane_rank in call_host_lanes(embedding_call):
            if lane_rank != rank:
                raise ValueError("token-embedding anchor moved ModelRunner rank")
            key = (pool_name, lane_rank, embedding_call.call_id, -1)
            events[key] = (
                RuntimePrepareEvent(_MODEL_RUNNER_INPUT_PREPARE_MECHANISM),
                RuntimePrepareEvent(_MODEL_RUNNER_FORWARD_METADATA_MECHANISM),
            )
        for pool_name, lane_rank in call_host_lanes(lm_head_call):
            if lane_rank != rank:
                raise ValueError("LM-head anchor moved ModelRunner rank")
            key = (pool_name, lane_rank, lm_head_call.call_id, -1)
            if key in events:
                raise AssertionError("ModelRunner outer event anchor is duplicated")
            events[key] = (
                RuntimePrepareEvent(_MODEL_RUNNER_LOGITS_PREPARE_MECHANISM),
            )
    return events


def ascend_910b_rotary_cache_update_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Expose the production source occurrence contract for diagnostics."""

    return dict(_production_rotary_cache_update_events(plan))


def ascend_910b_moe_wrapper_control_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Place one FX custom-wrapper control event before each MoE layer/rank.

    The frozen 910B Qwen3MoE FX graph has one ``vllm.moe_forward`` higher
    order node for each decoder layer and TP rank.  The compact KernelPlan
    represents that source point with expert-zero in explicit layer zero and
    repeated layer one.  Other plan layouts are rejected instead of guessing
    an equivalent wrapper position.
    """

    if not isinstance(plan, KernelPlan):
        raise TypeError("MoE wrapper descriptor requires a KernelPlan")
    if {call.kernel_spec.hardware_id for call in plan.calls} != {"ascend_910b"}:
        raise ValueError("MoE wrapper descriptor requires an Ascend-only plan")
    repeat_by_call = {
        call_id: repeat
        for repeat in plan.block_repeats
        for call_id in repeat.call_ids
    }
    anchors = {}
    for call in plan.calls:
        match = _MOE_WRAPPER_ANCHOR.search(call.call_id)
        if match is None:
            continue
        layer, encoded_rank = (int(value) for value in match.groups())
        if encoded_rank != call.rank:
            raise ValueError("MoE wrapper anchor rank differs from call rank")
        key = (layer, encoded_rank)
        if key in anchors:
            raise ValueError("MoE wrapper anchor is duplicated")
        anchors[key] = call
    expected = {(layer, rank) for layer in (0, 1) for rank in (0, 1)}
    if set(anchors) != expected:
        raise ValueError("MoE wrapper anchors are incomplete")
    events = {}
    for (layer, rank), call in anchors.items():
        repeat = repeat_by_call.get(call.call_id)
        if layer == 0:
            if repeat is not None:
                raise ValueError("explicit MoE wrapper anchor cannot repeat")
            occurrence_indices = (-1,)
        else:
            if repeat is None:
                raise ValueError("template MoE wrapper anchor must repeat")
            occurrence_indices = range(repeat.repeat_count)
        for occurrence_index in occurrence_indices:
            for pool_name, lane_rank in call_host_lanes(call):
                if lane_rank != rank:
                    raise ValueError("MoE wrapper anchor moved Host rank")
                event_key = (
                    pool_name,
                    lane_rank,
                    call.call_id,
                    occurrence_index,
                )
                if event_key in events:
                    raise AssertionError("MoE wrapper event is duplicated")
                events[event_key] = (
                    RuntimePrepareEvent(_MOE_WRAPPER_MECHANISM),)
    return events


def ascend_910b_unquant_mlp_wrapper_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Place the unquantized MoE MLP wrapper before its first GMM child.

    ``unified_apply_mlp`` dispatches the unquantized path once for each source
    MoE invocation.  Its first Device child is the ``w1`` grouped matmul, so
    that call is the only source-static position that can carry the wrapper.
    This diagnostic mapping is deliberately restricted to explicit ``w1``
    FusedMoE calls; it does not infer a quantization path for another layout.
    """

    if not isinstance(plan, KernelPlan):
        raise TypeError("MoE MLP wrapper descriptor requires a KernelPlan")
    if {call.kernel_spec.hardware_id for call in plan.calls} != {"ascend_910b"}:
        raise ValueError("MoE MLP wrapper descriptor requires an Ascend-only plan")
    repeat_by_call = {
        call_id: repeat
        for repeat in plan.block_repeats
        for call_id in repeat.call_ids
    }
    events = {}
    for call in plan.calls:
        if (call.kernel_spec.kind is not KernelKind.FUSED_MOE
                or call.kernel_spec.variant != "w1"):
            continue
        occurrence_indices = (
            range(repeat_by_call[call.call_id].repeat_count)
            if call.call_id in repeat_by_call else (-1,)
        )
        for occurrence_index in occurrence_indices:
            for pool_name, rank in call_host_lanes(call):
                event_key = (
                    pool_name,
                    rank,
                    call.call_id,
                    occurrence_index,
                )
                if event_key in events:
                    raise AssertionError("MoE MLP wrapper event is duplicated")
                events[event_key] = (
                    RuntimePrepareEvent(_MOE_UNQUANT_MLP_MECHANISM),)
    if not events:
        raise ValueError("MoE MLP wrapper descriptor has no w1 source anchor")
    return events


def ascend_910b_qkv_triton_composite_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Replace the QKV source site's generic Triton event with its composite.

    The source site has one Torch custom-op boundary, three output allocations,
    and one source-static Triton launch.  This descriptor maps only its static
    occurrence; the independent generic mechanism benchmark supplies timing.
    It must not be combined with ``triton_dynamic_argument_prepare``.
    """

    dynamic_events = ascend_910b_runtime_prepare_events(
        plan,
        enabled_mechanisms=("triton_dynamic_argument_prepare",),
    )
    return {
        key: (RuntimePrepareEvent(_QKV_TRITON_COMPOSITE_MECHANISM),)
        for key in dynamic_events
    }


def ascend_910b_fia_source_wrapper_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Replace a raw FIA API event with its source-wrapper composite."""

    api_events = ascend_910b_runtime_prepare_events(
        plan,
        enabled_mechanisms=("python_argument_prepare",),
    )
    return {
        key: (RuntimePrepareEvent(_FIA_SOURCE_WRAPPER_MECHANISM),)
        for key in api_events
    }


def ascend_910b_runtime_prepare_events(
    plan: KernelPlan,
    *,
    enabled_mechanisms: Tuple[str, ...] | None = None,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Map source-defined prepare sites to concrete plan occurrences.

    The mapping is intentionally 910B-specific diagnostic evidence.  A later
    production contract must express the same backend mechanisms without
    branching on operator names or retaining per-operator durations.
    """

    if not isinstance(plan, KernelPlan):
        raise TypeError("910B runtime prepare mapping requires a KernelPlan")
    hardware_ids = {
        call.kernel_spec.hardware_id for call in plan.calls
    }
    if hardware_ids != {"ascend_910b"}:
        raise ValueError(
            "910B runtime prepare mapping requires an Ascend-only plan")
    if enabled_mechanisms is not None:
        if (not isinstance(enabled_mechanisms, (list, tuple))
                or any(not isinstance(value, str) or not value
                       for value in enabled_mechanisms)):
            raise TypeError(
                "enabled runtime prepare mechanisms must be non-empty strings")
        enabled = frozenset(enabled_mechanisms)
        if len(enabled) != len(enabled_mechanisms):
            raise ValueError(
                "enabled runtime prepare mechanisms must be unique")
    else:
        enabled = None
    rule_by_kind_variant = {}
    for rule in ASCEND_910B_RUNTIME_PREPARE_RULES:
        for variant in rule.variants:
            key = (rule.kernel_kind, variant)
            if key in rule_by_kind_variant:
                raise AssertionError(
                    "910B runtime prepare rules overlap")
            rule_by_kind_variant[key] = rule
    repeat_by_call = {
        call_id: repeat
        for repeat in plan.block_repeats
        for call_id in repeat.call_ids
    }
    events = {}
    for call in plan.calls:
        variant = call.kernel_spec.variant
        key = (call.kernel_spec.kind, variant)
        mechanisms = []
        if key in rule_by_kind_variant:
            mechanisms.extend(rule_by_kind_variant[key].mechanisms)
        if call.kernel_spec.kind is KernelKind.COMMUNICATION:
            transfer = call.kernel_spec.transfer
            if transfer is None:
                raise ValueError(
                    "910B communication call requires a transfer descriptor")
            collective_mechanisms = {
                "all_reduce": "collective_all_reduce_prepare",
                "all_gather": "collective_all_gather_prepare",
            }
            if transfer.operation in collective_mechanisms:
                mechanisms.append(
                    collective_mechanisms[transfer.operation])
        if enabled is not None:
            mechanisms = [
                mechanism for mechanism in mechanisms
                if mechanism in enabled
            ]
        if not mechanisms:
            continue
        occurrence_indices = (
            range(repeat_by_call[call.call_id].repeat_count)
            if call.call_id in repeat_by_call else (-1,)
        )
        for occurrence_index in occurrence_indices:
            for pool_name, rank in call_host_lanes(call):
                event_key = (
                    pool_name,
                    rank,
                    call.call_id,
                    occurrence_index,
                )
                if event_key in events:
                    raise AssertionError(
                        "910B runtime prepare occurrence is duplicated")
                events[event_key] = tuple(
                    RuntimePrepareEvent(mechanism)
                    for mechanism in mechanisms
                )
    return events
