"""Source-static eager runtime-prepare occurrence rules for Serving.

Rules describe where generic runtime mechanisms occur.  They contain no
duration: service values are supplied only by independently measured Host
runtime-prepare profiles.  One engine interprets declarative rule sets; a
runtime stack contributes only data — variant rules, collective
mechanisms, anchor charges, and the completeness expectations its source
structure must satisfy.

Known source-evidenced boundaries not expressed by the rules:

- ``fused_experts`` splits its internal launches into chunks of
  ``VLLM_FUSED_MOE_CHUNK_SIZE`` (default 32768) tokens; the wrapper rules
  charge one prepare per occurrence, which is exact only while a Serving
  step stays below one chunk (chunked prefill keeps steps far below).
- Rare attention paths launch more than one varlen kernel per wrapper
  call (CUDA cascade attention; MetaX ``VLLM_METAX_ENABLE_FA_SPLIT_FORWARD``
  and chunked-context paths); the attention rule charges one wrapper
  prepare per occurrence, matching the single-launch common path.
- ``moe_gating_prepare`` profiles the ``fused_topk`` routing of
  Qwen3-class models (softmax + renormalize); models routing through
  ``grouped_topk`` (DeepSeek-style) would need their own measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Tuple

from src.kernel.kernel_spec import KernelKind
from src.serving.host_timing import RuntimePrepareEvent, WorkerRequestState, WORKER_REQUEST_STATE_UNITS
from src.serving.compiled_fx import (
    ascend_910b_compiled_fx_dense_metadata_events,
    ascend_910b_compiled_fx_rotary_events,
    ascend_910b_compiled_fx_reshape_events,
    repeat_indices_from_plan,
)
from src.serving.kernel_lowering.execution_timeline import call_host_lanes
from src.serving.kernel_lowering.kernel_plan import KernelPlan

@dataclass(frozen=True)
class RuntimePrepareRule:
    kernel_kind: KernelKind
    variants: Tuple[str, ...]
    mechanisms: Tuple[str, ...]
    call_id_pattern: str | None = None
    gate_pattern: str | None = None
    attribute_constraints: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.call_id_pattern is not None:
            if not isinstance(self.call_id_pattern, str) or not self.call_id_pattern:
                raise ValueError(
                    "runtime prepare rule call_id_pattern must be non-empty")
            re.compile(self.call_id_pattern)
        if self.gate_pattern is not None:
            if not isinstance(self.gate_pattern, str) or not self.gate_pattern:
                raise ValueError(
                    "runtime prepare rule gate_pattern must be non-empty")
            re.compile(self.gate_pattern)
        names = tuple(name for name, _ in self.attribute_constraints)
        if (len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names)
                or any(not values or any(not isinstance(value, str) or not value for value in values)
                       for _, values in self.attribute_constraints)):
            raise ValueError("runtime prepare rule attribute constraints must be unique non-empty strings")


@dataclass(frozen=True)
class RuntimePrepareAnchorCharge:
    """One source anchor: a call-ID suffix charged ``count`` entries."""

    suffix: str
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.suffix, str) or not self.suffix:
            raise ValueError("runtime prepare anchor suffix must be non-empty")
        if (isinstance(self.count, bool) or not isinstance(self.count, int)
                or self.count < 1):
            raise ValueError("runtime prepare anchor count must be positive")


@dataclass(frozen=True)
class RuntimePrepareAnchorGroup:
    """Completeness-checked per-layer anchor charges for one mechanism.

    Anchors live inside repeated decoder layers: every ``required_layers``
    bucket must appear, ``template_layer`` is the layer carried by a block
    repeat, and each of ``required_ranks`` must carry exactly the declared
    ranked and unranked anchor suffixes.  ``endpoint_prefixes`` are
    layer-external source boundaries charged once per rank.  When
    ``gate_pattern`` is given the whole group applies only to plans that
    contain that source signature; other plans emit nothing and are not
    validated.
    """

    name: str
    mechanism: str
    layer_pattern: str
    required_layers: Tuple[int, ...]
    template_layer: int
    required_ranks: Tuple[int, ...]
    ranked_charges: Tuple[RuntimePrepareAnchorCharge, ...]
    unranked_charges: Tuple[RuntimePrepareAnchorCharge, ...]
    endpoint_prefixes: Tuple[str, ...]
    gate_pattern: str | None = None

    def __post_init__(self) -> None:
        for name in ("name", "mechanism", "layer_pattern"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"runtime prepare anchor group {name} must be non-empty")
        if self.template_layer not in self.required_layers:
            raise ValueError(
                "runtime prepare anchor template layer must be required")
        if not self.required_ranks:
            raise ValueError("runtime prepare anchor group requires ranks")
        if self.gate_pattern is not None and not self.gate_pattern:
            raise ValueError("runtime prepare anchor gate must be non-empty")


@dataclass(frozen=True)
class RuntimePrepareWrapperAnchor:
    """One regex-anchored wrapper charge repeated per layer and rank."""

    name: str
    mechanism: str
    anchor_pattern: str
    required_layers: Tuple[int, ...]
    template_layer: int
    required_ranks: Tuple[int, ...]
    gate_pattern: str | None = None

    def __post_init__(self) -> None:
        for name in ("name", "mechanism", "anchor_pattern"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"runtime prepare wrapper anchor {name} must be non-empty")
        if self.template_layer not in self.required_layers:
            raise ValueError(
                "runtime prepare wrapper template layer must be required")
        if not self.required_ranks:
            raise ValueError("runtime prepare wrapper anchor requires ranks")
        if self.gate_pattern is not None and not self.gate_pattern:
            raise ValueError("runtime prepare wrapper gate must be non-empty")


@dataclass(frozen=True)
class RuntimePrepareRuleSet:
    """Declarative source occurrence rules for one runtime stack."""

    hardware_ids: Tuple[str, ...]
    variant_rules: Tuple[RuntimePrepareRule, ...]
    collective_mechanisms: Tuple[Tuple[str, str], ...]
    exclusive_mechanism_groups: Tuple[Tuple[str, ...], ...] = ()
    anchor_groups: Tuple[RuntimePrepareAnchorGroup, ...] = ()
    wrapper_anchors: Tuple[RuntimePrepareWrapperAnchor, ...] = ()

    def __post_init__(self) -> None:
        if (not self.hardware_ids
                or any(not isinstance(value, str) or not value
                       for value in self.hardware_ids)):
            raise ValueError("runtime prepare rule set requires hardware IDs")
        if len(set(self.hardware_ids)) != len(self.hardware_ids):
            raise ValueError("runtime prepare rule set hardware IDs repeat")


_COLLECTIVE_MECHANISMS = (
    ("all_reduce", "collective_all_reduce_prepare"),
    ("all_gather", "collective_all_gather_prepare"),
)

# The measured profile is hardware/runtime scoped.  Source-specific anchor
# groups are emitted only when their physical source anchors are present in
# the plan; model identity never gates the whole profile.
_MOE_DECODER_GRAPH_SIGNATURE = r":layer-([01]):moe:ep-rank-([0-9]+):0$"

# The dense 910B decoder lowers into one unsplit torchair FX graph, so its
# eager lane is priced by per-node class dispatch, not by per-wrapper
# prepare mechanisms.  Those wrapper mechanisms describe the split-piece
# MoE executor contract and stay gated to MoE-signature plans so a dense
# plan is never charged both structures.
_DENSE_SHARED_RULE_GATE = _MOE_DECODER_GRAPH_SIGNATURE

_ASCEND_910B_RULE_SET = RuntimePrepareRuleSet(
    hardware_ids=("ascend_910b",),
    variant_rules=(
        RuntimePrepareRule(
            KernelKind.VECTOR,
            ("qkv_rmsnorm_rope",),
            (
                "triton_dynamic_argument_prepare",
                "triton_three_output_composite_prepare",
            ),
            gate_pattern=_DENSE_SHARED_RULE_GATE,
        ),
        RuntimePrepareRule(
            KernelKind.VECTOR,
            ("moe_gating_topk",),
            ("stream_record_event_prepare", "moe_gating_prepare"),
        ),
        RuntimePrepareRule(
            KernelKind.VECTOR,
            ("moe_init_routing_v2",),
            ("stream_record_event_prepare", "moe_token_dispatch_prepare"),
        ),
        RuntimePrepareRule(
            KernelKind.FUSED_MOE,
            ("w1",),
            (
                "moe_grouped_gemm_prepare",
                "moe_unquant_mlp_wrapper_prepare",
                # vLLM-Ascend's token dispatcher performs one source-local
                # expert-map Index, sentinel NotEqual, and weight-mask Mul
                # chain before each unquantized W1 dispatch.  The independent
                # runtime-call microbench measures that complete host/device
                # submission chain as one mechanism; attach it to the W1
                # anchor so each source MoE invocation is charged exactly
                # once without inventing per-op duration values.
                "moe_routing_index_mask_mul_prepare",
            ),
        ),
        RuntimePrepareRule(
            KernelKind.FUSED_MOE,
            ("w2",),
            ("moe_grouped_gemm_prepare",),
        ),
        RuntimePrepareRule(
            KernelKind.GEMM,
            ("gemm",),
            ("unquantized_gemm_prepare",),
            gate_pattern=_DENSE_SHARED_RULE_GATE,
        ),
        RuntimePrepareRule(
            KernelKind.VECTOR,
            ("moe_swiglu",),
            ("moe_swiglu_prepare",),
        ),
        RuntimePrepareRule(
            KernelKind.VECTOR,
            ("moe_token_unpermute",),
            ("stream_record_event_prepare", "moe_token_combine_prepare"),
        ),
        RuntimePrepareRule(
            KernelKind.KV_CACHE,
            ("paged_kv_cache",),
            ("reshape_cache_prepare",),
            gate_pattern=_DENSE_SHARED_RULE_GATE,
        ),
        RuntimePrepareRule(
            KernelKind.FLASH_ATTN,
            ("prefill", "decode"),
            ("fused_attention_source_wrapper_prepare",),
            gate_pattern=_DENSE_SHARED_RULE_GATE,
        ),
    ),
    collective_mechanisms=_COLLECTIVE_MECHANISMS,
    exclusive_mechanism_groups=(
        (
            "triton_dynamic_argument_prepare",
            "triton_three_output_composite_prepare",
        ),
    ),
    anchor_groups=(
        RuntimePrepareAnchorGroup(
            name="compiled-FX",
            mechanism="compiled_fx_backend_entry",
            layer_pattern=r":layer-([01]):(.+)$",
            required_layers=(0, 1),
            template_layer=1,
            required_ranks=(0, 1),
            ranked_charges=(
                RuntimePrepareAnchorCharge("moe:ep-rank-{rank}:0", 1),
                RuntimePrepareAnchorCharge("input-norm:tp-rank-{rank}", 2),
                RuntimePrepareAnchorCharge("qkv:tp-rank-{rank}", 1),
                RuntimePrepareAnchorCharge("q-norm:tp-rank-{rank}", 3),
                RuntimePrepareAnchorCharge("attention:tp-rank-{rank}", 1),
                RuntimePrepareAnchorCharge("o-proj:tp-rank-{rank}", 1),
                RuntimePrepareAnchorCharge("post-attn-norm:tp-rank-{rank}", 2),
                RuntimePrepareAnchorCharge("router:tp-rank-{rank}", 1),
            ),
            unranked_charges=(
                RuntimePrepareAnchorCharge("o-proj:tp-all-reduce", 1),
                RuntimePrepareAnchorCharge("moe:ep-output-all-reduce", 1),
            ),
            endpoint_prefixes=(
                "token-embedding:tp-rank-",
                "final-norm:tp-rank-",
            ),
            gate_pattern=_MOE_DECODER_GRAPH_SIGNATURE,
        ),
    ),
    wrapper_anchors=(
        RuntimePrepareWrapperAnchor(
            name="MoE wrapper",
            mechanism="compiled_fx_moe_wrapper_control",
            anchor_pattern=_MOE_DECODER_GRAPH_SIGNATURE,
            required_layers=(0, 1),
            template_layer=1,
            required_ranks=(0, 1),
            gate_pattern=_MOE_DECODER_GRAPH_SIGNATURE,
        ),
    ),
)

_CUDA_LIKE_RULE_SET = RuntimePrepareRuleSet(
    hardware_ids=("nvidia_a800", "metax_c500", "nvidia_h200"),
    variant_rules=(
        # Explicit mathematical source programs preserve the modular router
        # and prepared-expert Host wrappers even without native launch specs.
        # Legacy vLLM0.11 independent services are declared CPU proxies for
        # the same allocation/dispatch mechanisms in vLLM0.30; no new price.
        # Graph capture suppresses these events through the shared timeline.
        # Do not infer an FX piece entry from the router's location: modern
        # modular custom-op boundaries do not prove that partition structure.
        RuntimePrepareRule(
            KernelKind.VECTOR, ("moe_topk",), ("moe_gating_prepare",),
            attribute_constraints=(("source_program", ("vllm_0_30_bf16_softmax_topk",)),),
        ),
        RuntimePrepareRule(
            KernelKind.FUSED_MOE, ("routed_experts",),
            ("compiled_fx_moe_wrapper_control", "moe_unquant_mlp_wrapper_prepare"),
            attribute_constraints=(("mathematical_program", ("unquantized_bf16_swiglu_v1",
                                          "unquantized_bf16_grouped_swiglu_v1")),),
        ),
        # Piecewise compilation splits the eager program at the attention
        # ops, yielding num_layers + 1 compiled pieces per forward
        # (vllm compilation/backends.py split_graph).  Each decoder layer
        # gating occurrence anchors one inter-layer piece; the first piece
        # (embedding + layer-0 pre-attention) contains no gating call, so
        # the embedding occurrence anchors its single FX backend entry.
        RuntimePrepareRule(
            KernelKind.VECTOR,
            ("gather",),
            ("compiled_fx_backend_entry",),
            call_id_pattern="token-embedding",
        ),
        RuntimePrepareRule(
            KernelKind.VECTOR,
            ("moe_gating_topk",),
            ("compiled_fx_backend_entry", "moe_gating_prepare"),
        ),
        RuntimePrepareRule(
            KernelKind.FUSED_MOE,
            ("w1",),
            (
                "compiled_fx_moe_wrapper_control",
                "moe_unquant_mlp_wrapper_prepare",
            ),
        ),
        RuntimePrepareRule(
            KernelKind.GEMM,
            ("gemm",),
            ("unquantized_gemm_prepare",),
        ),
        RuntimePrepareRule(
            KernelKind.KV_CACHE,
            ("paged_kv_cache",),
            ("reshape_cache_prepare",),
        ),
        RuntimePrepareRule(
            KernelKind.FLASH_ATTN,
            ("prefill", "decode"),
            ("fused_attention_source_wrapper_prepare",),
        ),
    ),
    collective_mechanisms=_COLLECTIVE_MECHANISMS,
)

_RUNTIME_PREPARE_RULE_SETS = (
    _ASCEND_910B_RULE_SET,
    _CUDA_LIKE_RULE_SET,
)

_ASCEND_910B_TOKEN_EMBEDDING_ANCHOR = re.compile(
    r"(?:^|:)(?:token-embedding|embedding(?::mask-ids)?)(?:\:tp-rank-([0-9]+))?$"
)


_ASCEND_910B_LM_HEAD_ANCHOR = re.compile(
    r"(?:^|:)lm-head(?::projection)?(?:\:tp-rank-([0-9]+))?$")


def _ascend_910b_model_runner_outer_events(
    plan: KernelPlan,
) -> Mapping[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Place source-authored ModelRunner outer preparation events.

    ``NPUModelRunner.execute_model`` prepares inputs and attention metadata
    before entering the model, then selects sampled hidden states after the
    forward returns.  The compact workload has no separate nodes for those
    host-only regions, so source token-embedding and LM-head boundaries carry
    the events.  Both ranked and unranked endpoint IDs are accepted; every
    concrete Host lane must expose one and only one endpoint of each kind, and
    ranked labels must agree when present.  Incomplete or duplicated plans
    fail closed instead of borrowing a duration.
    """

    if not isinstance(plan, KernelPlan):
        raise TypeError("ModelRunner outer descriptor requires a KernelPlan")
    if {call.kernel_spec.hardware_id for call in plan.calls} != {"ascend_910b"}:
        raise ValueError("ModelRunner outer descriptor requires Ascend-only plan")
    embedding_by_rank = {}
    lm_head_by_rank = {}
    embedding_labels = {}
    lm_head_labels = {}
    for call in plan.calls:
        for pattern, target, label in (
            (_ASCEND_910B_TOKEN_EMBEDDING_ANCHOR, embedding_by_rank,
             "token-embedding"),
            (_ASCEND_910B_LM_HEAD_ANCHOR, lm_head_by_rank, "LM-head"),
        ):
            match = pattern.search(call.call_id)
            if match is None:
                continue
            # TP multi-row logits also materialize a VECTOR named lm-head.
            # ModelRunner preparation belongs to the linear projection.
            if label == "LM-head" and call.kernel_spec.kind is not KernelKind.GEMM:
                continue
            encoded_rank = match.group(1)
            if call.rank in target:
                raise ValueError(f"ModelRunner {label} anchor is duplicated")
            target[call.rank] = call
            if encoded_rank is not None:
                encoded_rank = int(encoded_rank)
                if encoded_rank < 0:
                    raise ValueError(f"{label} anchor rank must be non-negative")
            labels = embedding_labels if label == "token-embedding" else lm_head_labels
            labels[call.rank] = encoded_rank
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
    if all(value is not None for value in embedding_label_values):
        if (len(set(embedding_label_values)) != len(embedding_label_values)
                or len(set(lm_head_label_values)) != len(lm_head_label_values)
                or set(embedding_label_values) != set(lm_head_label_values)):
            raise ValueError(
                "ModelRunner token-embedding and LM-head logical ranks differ")
    events = {}
    for rank, embedding_call in embedding_by_rank.items():
        lm_head_call = lm_head_by_rank[rank]
        for pool_name, lane_rank in call_host_lanes(embedding_call):
            if lane_rank != rank:
                raise ValueError("token-embedding anchor moved ModelRunner rank")
            key = (pool_name, lane_rank, embedding_call.call_id, -1)
            events[key] = (
                RuntimePrepareEvent(
                    "model_runner_input_prepare", graph_capturable=False),
                RuntimePrepareEvent(
                    "model_runner_forward_metadata_prepare",
                    graph_capturable=False),
            )
        for pool_name, lane_rank in call_host_lanes(lm_head_call):
            if lane_rank != rank:
                raise ValueError("LM-head anchor moved ModelRunner rank")
            key = (pool_name, lane_rank, lm_head_call.call_id, -1)
            if key in events:
                raise AssertionError("ModelRunner outer anchor is duplicated")
            events[key] = (
                RuntimePrepareEvent(
                    "model_runner_logits_prepare", graph_capturable=False),
            )
    return events


def ascend_910b_rotary_cache_update_events(
    plan: KernelPlan,
) -> Mapping[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Place the source-authored rotary-cache update before decoder entry.

    Ascend's ``NPUModelRunner.execute_model`` updates the rotary cosine/sine
    cache after ModelRunner input preparation and before entering the decoder.
    The compact plan has no separate call for that helper, so the per-rank
    token-embedding boundary is the only source anchor.  The source call is
    not specific to the fused Qwen3 QKV/RoPE child: Llama's native-RoPE path
    executes the same helper when it is a non-VL model with RoPE.  Therefore
    every Ascend plan with concrete token-embedding Host anchors is eligible;
    missing or duplicated anchors still fail closed.
    """

    if not isinstance(plan, KernelPlan):
        raise TypeError("rotary-cache descriptor requires a KernelPlan")
    if {call.kernel_spec.hardware_id for call in plan.calls} != {"ascend_910b"}:
        raise ValueError("rotary-cache descriptor requires an Ascend-only plan")
    anchors = {}
    ranked_forms = set()
    for call in plan.calls:
        match = _ASCEND_910B_TOKEN_EMBEDDING_ANCHOR.search(call.call_id)
        if match is None:
            continue
        encoded_rank = match.group(1)
        ranked_forms.add(encoded_rank is not None)
        if (encoded_rank is not None
                and int(encoded_rank) != call.rank):
            raise ValueError("rotary-cache anchor rank differs from call rank")
        if call.rank in anchors:
            raise ValueError("rotary-cache token-embedding anchor is duplicated")
        anchors[call.rank] = call
    if not anchors:
        raise ValueError(
            "rotary-cache descriptor has no token-embedding anchors; "
            f"call_ids={[call.call_id for call in plan.calls if 'embedding' in call.call_id]!r}")
    if len(ranked_forms) != 1:
        raise ValueError(
            "rotary-cache token-embedding anchors must be uniformly ranked "
            "or uniformly unranked")
    rope_variants = {"rotary", "qkv_rmsnorm_rope", "mla_rope"}
    has_rope = any(
        call.kernel_spec.variant in rope_variants
        or ":rotary-" in call.call_id
        or ":rotary" in call.call_id
        for call in plan.calls
    )
    if not has_rope:
        raise ValueError(
            "rotary-cache descriptor requires an explicit RoPE source call")
    has_mla = any(
        (isinstance(call.kernel_spec.variant, str)
         and call.kernel_spec.variant.startswith("mla_"))
        or ":mla-" in call.call_id
        or ":main-rotary" in call.call_id
        or ":indexer-rotary" in call.call_id
        for call in plan.calls
    )
    if has_mla:
        raise ValueError(
            "rotary-cache descriptor is unsupported for MLA plans")
    has_vl = any(
        any(token in call.call_id.lower()
            for token in ("vision", "multimodal", "multi-modal", ":vl-"))
        for call in plan.calls
    )
    if has_vl:
        raise ValueError(
            "rotary-cache descriptor requires a non-VL plan")

    events = {}
    for rank, call in anchors.items():
        for pool_name, lane_rank in call_host_lanes(call):
            if lane_rank != rank:
                raise ValueError("rotary-cache anchor moved Host rank")
            key = (pool_name, lane_rank, call.call_id, -1)
            if key in events:
                raise AssertionError("rotary-cache event is duplicated")
            events[key] = (
                RuntimePrepareEvent(
                    "rotary_cache_update_prepare", graph_capturable=False),
            )
    return events


def _retain_enabled_mechanisms(events, mechanisms, enabled, *, drop_empty):
    """Drop events whose mechanism is not enabled.

    Two Ascend metadata families build their full event set and then keep only
    the mechanisms the caller enabled.  The removals are applied in the
    caller's order.  ``drop_empty`` additionally removes entries that lost all
    their events: the model-runner family drops them, the dense-metadata family
    keeps them (its consumers index by call id and tolerate an empty tuple),
    so this is a declared difference rather than a shared step.
    """

    for mechanism in mechanisms:
        if mechanism in enabled:
            continue
        events = {
            key: tuple(
                event for event in entry if event.mechanism != mechanism)
            for key, entry in events.items()
        }
    if not drop_empty:
        return events
    return {key: entry for key, entry in events.items() if entry}


def _merge_runtime_prepare_events(
    target: dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]],
    additions: Mapping[
        tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]
    ],
) -> None:
    for key, added in additions.items():
        current = target[key] if key in target else ()
        names = tuple(event.mechanism for event in current + added)
        if len(names) != len(set(names)):
            raise ValueError("runtime prepare mechanism is duplicated at one source site")
        target[key] = current + added


def _ranked_anchor_multiplicity(group, call_id, rank):
    for charge in group.ranked_charges:
        suffix = charge.suffix.replace("{rank}", str(rank))
        if (call_id.endswith(suffix)
                or re.search(re.escape(suffix) + r":[0-9]+$", call_id)):
            return charge.count
    return None


def _ranked_anchor_suffix(group, call_id, rank):
    for charge in group.ranked_charges:
        suffix = charge.suffix.replace("{rank}", str(rank))
        if (call_id.endswith(suffix)
                or re.search(re.escape(suffix) + r":[0-9]+$", call_id)):
            return suffix
    return None


def _ranked_anchor_encoded_rank(group, call_id):
    """Return the physical rank encoded by a ranked anchor call ID.

    ``required_ranks`` declares the number of rank lanes in the source
    contract; physical rank IDs themselves come from the current plan.  This
    keeps EP plans with remapped ranks (for example ``91`` and ``7``)
    equivalent to the ordinary ``0``/``1`` layout while still validating the
    encoded source IDs.
    """

    matches = []
    for charge in group.ranked_charges:
        marker = "{rank}"
        if marker not in charge.suffix:
            continue
        prefix, suffix = charge.suffix.split(marker, 1)
        match = re.search(
            re.escape(prefix) + r"([0-9]+)" + re.escape(suffix)
            + r"(?::[0-9]+)?$",
            call_id,
        )
        if match is not None:
            matches.append(int(match.group(1)))
    if len(set(matches)) > 1:
        raise ValueError("runtime prepare ranked anchor encodes multiple ranks")
    return matches[0] if matches else None


def _anchor_group_events(
    plan: KernelPlan,
    group: RuntimePrepareAnchorGroup,
    repeat_indices,
) -> Mapping[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Count generated operator entries at one group's source anchors."""

    layer_regex = re.compile(group.layer_pattern)
    layer_calls = {layer: [] for layer in group.required_layers}
    endpoints = {}
    for call in plan.calls:
        match = layer_regex.search(call.call_id)
        if match is not None:
            layer_calls[int(match.group(1))].append(call)
            continue
        if ":layer-" in call.call_id:
            raise ValueError(
                f"{group.name} descriptor does not support this layer ID")
        for endpoint in group.endpoint_prefixes:
            if endpoint not in call.call_id:
                continue
            suffix = call.call_id.rsplit(endpoint, 1)[1]
            if not suffix.isdigit():
                raise ValueError(f"{group.name} endpoint rank is malformed")
            key = (endpoint, int(suffix))
            if key in endpoints:
                raise ValueError(f"{group.name} endpoint is duplicated")
            endpoints[key] = call
    if any(not layer_calls[layer] for layer in group.required_layers):
        raise ValueError(
            f"{group.name} descriptor requires layers zero and one")
    template_repeats = {
        repeat_indices[call.call_id]
        for call in layer_calls[group.template_layer]
    }
    if len(template_repeats) != 1 or next(iter(template_repeats)) == (-1,):
        raise ValueError(
            f"{group.name} layer-one calls must share one repeat")

    events = {}
    required_rank_count = len(group.required_ranks)
    observed_ranks = None
    for calls in layer_calls.values():
        ranked = {}
        unranked = {}
        for call in calls:
            encoded_rank = _ranked_anchor_encoded_rank(group, call.call_id)
            multiplicity = _ranked_anchor_multiplicity(
                group, call.call_id, call.rank)
            if encoded_rank is not None:
                if encoded_rank != call.rank:
                    raise ValueError(
                        f"{group.name} anchor rank differs from call rank")
                if multiplicity is None:
                    raise ValueError(
                        f"{group.name} ranked decoder anchor is unknown")
                ranked[(call.rank, call.call_id)] = (call, multiplicity)
                continue
            for charge in group.unranked_charges:
                if call.call_id.endswith(charge.suffix):
                    if charge.suffix in unranked:
                        raise ValueError(
                            f"{group.name} collective anchor is duplicated")
                    unranked[charge.suffix] = (call, charge.count)
                    break
        ranks = {rank for rank, _ in ranked}
        if len(ranks) != required_rank_count:
            raise ValueError(
                f"{group.name} descriptor requires {required_rank_count} rank lanes")
        if observed_ranks is None:
            observed_ranks = ranks
        elif ranks != observed_ranks:
            raise ValueError(
                f"{group.name} decoder rank lanes differ between layers")
        for rank in ranks:
            expected = {
                charge.suffix.replace("{rank}", str(rank))
                for charge in group.ranked_charges
            }
            actual = {
                _ranked_anchor_suffix(group, call_id, event_rank)
                for event_rank, call_id in ranked
                if event_rank == rank
            }
            if actual != expected:
                raise ValueError(
                    f"{group.name} ranked decoder anchors are incomplete "
                    f"for rank {rank}: expected={sorted(expected)!r} "
                    f"actual={sorted(actual)!r} "
                    f"calls={[call.call_id for call in calls if call.rank == rank]!r}")
        if set(unranked) != {
                charge.suffix for charge in group.unranked_charges}:
            raise ValueError(
                f"{group.name} collective decoder anchors are incomplete")
        for (rank, call_id), (call, multiplicity) in ranked.items():
            for occurrence_index in repeat_indices[call_id]:
                for pool_name, lane_rank in call_host_lanes(call):
                    if lane_rank != rank:
                        raise AssertionError(
                            f"{group.name} rank-local decoder anchor changed "
                            "Host lane")
                    events[(pool_name, rank, call_id, occurrence_index)] = (
                        RuntimePrepareEvent(group.mechanism, multiplicity),)
        for call, multiplicity in unranked.values():
            for occurrence_index in repeat_indices[call.call_id]:
                for pool_name, rank in call_host_lanes(call):
                    events[(pool_name, rank, call.call_id, occurrence_index)] = (
                        RuntimePrepareEvent(group.mechanism, multiplicity),)
    endpoint_ranks = {rank for _, rank in endpoints}
    if (observed_ranks is None
            or endpoint_ranks != observed_ranks
            or len(endpoints)
            != len(group.endpoint_prefixes) * required_rank_count):
        raise ValueError(
            f"{group.name} descriptor requires all endpoint rank lanes")
    for call in endpoints.values():
        for pool_name, rank in call_host_lanes(call):
            events[(pool_name, rank, call.call_id, -1)] = (
                RuntimePrepareEvent(group.mechanism),)
    return events


def _wrapper_anchor_events(
    plan: KernelPlan,
    wrapper: RuntimePrepareWrapperAnchor,
    repeat_by_call,
) -> Mapping[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Place one generated wrapper charge per declared layer and rank."""

    pattern = re.compile(wrapper.anchor_pattern)
    anchors = {}
    for call in plan.calls:
        match = pattern.search(call.call_id)
        if match is None:
            continue
        layer, encoded_rank = (int(value) for value in match.groups())
        if encoded_rank != call.rank:
            raise ValueError(
                f"{wrapper.name} anchor rank differs from call rank")
        key = (layer, encoded_rank)
        if key in anchors:
            raise ValueError(f"{wrapper.name} anchor is duplicated")
        anchors[key] = call
    required_rank_count = len(wrapper.required_ranks)
    observed_ranks = {rank for _, rank in anchors}
    if len(observed_ranks) != required_rank_count:
        raise ValueError(
            f"{wrapper.name} descriptor requires {required_rank_count} rank lanes")
    expected = {
        (layer, rank)
        for layer in wrapper.required_layers
        for rank in observed_ranks
    }
    if set(anchors) != expected:
        raise ValueError(f"{wrapper.name} anchors are incomplete")
    events = {}
    for (layer, rank), call in anchors.items():
        if layer == wrapper.template_layer:
            if call.call_id not in repeat_by_call:
                raise ValueError(
                    f"template {wrapper.name} anchor must repeat")
            occurrence_indices = range(repeat_by_call[call.call_id].repeat_count)
        else:
            if call.call_id in repeat_by_call:
                raise ValueError(
                    f"explicit {wrapper.name} anchor cannot repeat")
            occurrence_indices = (-1,)
        for occurrence_index in occurrence_indices:
            for pool_name, lane_rank in call_host_lanes(call):
                if lane_rank != rank:
                    raise ValueError(f"{wrapper.name} anchor moved Host rank")
                events[(pool_name, rank, call.call_id, occurrence_index)] = (
                    RuntimePrepareEvent(wrapper.mechanism),)
    return events


# Source node classes for the dense 910B decoder FX graph.  Keys are the
# plan's (KernelKind, variant); values are (mechanism, node multiplicity)
# per call occurrence, pinned by the canonical qps64 prefill kernel census
# (145 aclnnMatmul = 144 gemm + 1 gemv lm-head, 73 RmsNorm = 2 per fused
# qkv/rope + layer-0, 72 AddRmsNormBias, 36 SwiGlu, 217 copy-shape
# TensorMove/Slice, 36 rope, 36 FusedInferAttentionScore, 36
# ReshapeAndCache).  A call outside the table fails the mapping closed.
_ASCEND_910B_DENSE_NODE_CLASSES: Mapping[
    tuple[KernelKind, str], tuple[tuple[str, int], ...]
] = {
    (KernelKind.GEMM, "gemm"): (
        ("compiled_fx_aten_linear_node_dispatch", 1),),
    (KernelKind.GEMM, "gemv"): (
        ("compiled_fx_aten_linear_node_dispatch", 1),),
    (KernelKind.FLASH_ATTN, "prefill"): (
        ("compiled_fx_fia_node_dispatch", 1),
        ("compiled_fx_aten_copy_node_dispatch", 1),
    ),
    # Decode attention dispatches through a torch_npu pybind light-class
    # call with the same host-lane machinery as the norm class; the norm
    # chain is the measured vehicle for that class.
    (KernelKind.FLASH_ATTN, "decode"): (
        ("compiled_fx_torch_npu_norm_node_dispatch", 1),),
    (KernelKind.KV_CACHE, "paged_kv_cache"): (
        ("compiled_fx_reshape_cache_node_dispatch", 1),),
    # AscendRMSNorm.forward_oot takes three different branches, and the three
    # branches were measured separately (production_node_dispatch_20260926_r7):
    # the residual branch issues vllm::maybe_chunk_residual then
    # _C_ascend::npu_add_rms_norm_bias, the no-residual branch issues
    # torch_npu.npu_rms_norm and the weight-prefetch postprocess, and SwiGlu is
    # npu::npu_swiglu.  One price cannot stand for all three.
    (KernelKind.VECTOR, "add_rmsnorm"): (
        ("compiled_fx_torch_npu_norm_node_dispatch", 1),),
    (KernelKind.VECTOR, "rmsnorm"): (
        ("compiled_fx_norm_no_residual_node_dispatch", 1),),
    (KernelKind.VECTOR, "silu"): (
        ("compiled_fx_swiglu_node_dispatch", 1),),
    # SymbolicEmbedding preserves the source ids.long(): ATen _to_copy
    # allocates the result and dispatches the same copy_ primitive measured
    # by the existing copy class. No new Host duration is introduced.
    (KernelKind.VECTOR, "vllm_embedding_id_widen"): (
        ("compiled_fx_aten_copy_node_dispatch", 1),),
    (KernelKind.VECTOR, "elementwise"): (
        ("compiled_fx_aten_copy_node_dispatch", 1),),
    (KernelKind.VECTOR, "gather"): (
        ("compiled_fx_aten_copy_node_dispatch", 1),),
    # The Q/K Norm and RoPE nodes lower separately (the fused
    # split_qkv_rmsnorm_rope signature is absent from every captured 910B
    # launch observation), so RoPE carries its own dense node dispatch.  The
    # per-class mechanism is the same one the fused entry below charges for
    # its RoPE member; only the occurrence moves with the node.
    (KernelKind.VECTOR, "rotary"): (
        ("compiled_fx_rope_node_dispatch", 1),),
    (KernelKind.VECTOR, "qkv_rmsnorm_rope"): (
        ("compiled_fx_torch_npu_norm_node_dispatch", 2),
        ("compiled_fx_aten_copy_node_dispatch", 3),
        ("compiled_fx_rope_node_dispatch", 1),
    ),
}


def _ascend_masked_embedding_node_classes(spec):
    """Expand the ordinary Ascend0.18 TP2 helper's actual FX operations.

    AscendVocabParallelEmbedding._get_masked_input_and_mask has no separate
    compiled decorator. The installed npugraph_ex optimized FX retains seven
    mask-ids and two mask-output ATen calls. Independent complete-submit
    measurements replace the occurrence's generic enqueue charge. Missing
    mechanism evidence fails closed.
    """
    attrs = spec.attributes
    if (attrs["mask_program"] != "vllm_ascend_no_added_vocab"
            or attrs["tensor_parallel_size"] != 2):
        raise ValueError("Ascend TP Host mask dispatch requires its no-added-vocabulary TP2 source")
    stage = attrs["stage"]
    if stage == "mask_ids":
        scalar_mul = "scalar_zero_mul_bool" if attrs["tensor_parallel_rank"] == 0 else "scalar_mul_bool"
        primitives = ("ge_scalar", "lt_scalar", "bool_and", scalar_mul,
                      "sub_int32_int64", "bool_mul_int64", "bool_invert")
    elif stage == "mask_output":
        if spec.dtype != "bf16":
            raise ValueError("Ascend TP Host output mask requires its BF16 source")
        primitives = ("unsqueeze", "masked_fill_inplace")
    else:
        raise ValueError("Ascend TP Host mask dispatch requires an explicit stage")
    return tuple((f"aten_{primitive}_host_submit", 1) for primitive in primitives)


def _ascend_source_moe_node_classes(plan, ascend_compilation_mode):
    """Bind the ordinary 0.18 AllGather MoE wrapper to all its source calls.

    The source model exposes routing and owner masking separately from the
    six lowered expert children. This is one opaque moe_forward invocation,
    not six compiled-FX entries. Legacy split-piece descriptors stay separate.
    """
    routes = {}
    for call in plan.calls:
        spec = call.kernel_spec
        if (spec.kind is KernelKind.VECTOR and spec.variant == "moe_topk"
                and spec.attributes["source_program"] == "vllm_0_18_bf16_softmax_topk"
                and spec.attributes["stage"] == "route"):
            match = re.search(r":layer-([0-9]+):route$", call.call_id)
            if match is None:
                raise ValueError("Ascend source MoE route has no layer anchor")
            key = (int(match.group(1)), call.rank)
            if key in routes:
                raise ValueError("Ascend source MoE route is duplicated")
            routes[key] = call
    if not routes:
        return {}
    if ascend_compilation_mode == "NONE":
        wrapper_mechanism = "compiled_fx_moe_wrapper_control"
    elif ascend_compilation_mode in ("FULL", "FULL_DECODE_ONLY", "PIECEWISE"):
        wrapper_mechanism = "compiled_fx_moe_source_wrapper_prepare"
    else:
        raise ValueError("Ascend source MoE requires its configured compilation mode")
    layers = {layer for layer, _ in routes}
    if (layers != set(range(max(layers) + 1))
            or set(routes) != {(layer, rank) for layer in layers for rank in (0, 1)}):
        raise ValueError("Ascend source MoE requires complete TP2/EP2 layer lanes")
    calls_by_id = {call.call_id: call for call in plan.calls}
    result = {}
    child_contract = (
        # The legacy token-dispatch price also contains owner-mask work.
        # This source exposes that mask as its own three complete submits.
        (KernelKind.VECTOR, "moe_init_routing", ("stream_record_event_prepare", "moe_init_routing_prepare")),
        (KernelKind.FUSED_MOE, "w1", ("moe_grouped_gemm_prepare", "moe_unquant_mlp_wrapper_prepare")),
        (KernelKind.VECTOR, "moe_swiglu", ("moe_swiglu_prepare",)),
        (KernelKind.FUSED_MOE, "w2", ("moe_grouped_gemm_prepare",)),
        (KernelKind.VECTOR, "moe_abs", ("aten_abs_int32_host_submit",)),
        (KernelKind.VECTOR, "moe_token_unpermute", ("stream_record_event_prepare", "moe_token_combine_prepare")),
    )
    for (_, rank), route in routes.items():
        prefix = route.call_id.removesuffix(":route")
        owner_id = prefix + ":owner-mask"
        if owner_id not in calls_by_id:
            raise ValueError("Ascend source MoE owner mask is missing")
        owner = calls_by_id[owner_id]
        if (owner.rank != rank or owner.kernel_spec.variant != "moe_topk"
                or owner.kernel_spec.attributes["source_program"] != "vllm_0_18_bf16_softmax_topk"
                or owner.kernel_spec.attributes["stage"] != "owner_mask"):
            raise ValueError("Ascend source MoE owner mask disagrees with its route")
        # When compiled, this is one opaque FX node. The legacy wrapper-control
        # price subtracts the generic node entry, which its legacy anchor
        # rules charge separately. Source plans have no such anchor; use the
        # complete independently measured TorchAir wrapper boundary here,
        # once only. Its compiler/source capture is independent of the
        # legacy Inductor control-plus-generic-entry arithmetic.
        # Configured NONE keeps control-only pricing without an FX entry.
        # The measurement stubs Device children and stream/Event APIs, so
        # their submits and three event records remain separate charges.
        result[route.call_id] = ((wrapper_mechanism, 1),
                                ("stream_record_event_prepare", 1),
                                ("moe_gating_prepare", 1))
        result[owner_id] = tuple((name, 1) for name in (
            "aten_index_int32_host_submit", "aten_ne_scalar_host_submit", "aten_bf16_mul_bool_host_submit"))
        first_child_id = f"{prefix}:experts:0"
        if first_child_id not in calls_by_id:
            raise ValueError("Ascend source MoE expert child is missing")
        first_spec = calls_by_id[first_child_id].kernel_spec
        if (first_spec.variant != "moe_init_routing"
                or first_spec.attributes["mathematical_program"] != "ascend_0_18_moe_dispatch_math_v1"):
            raise ValueError("Ascend source MoE requires its explicit dispatch math")
        empty_local = first_spec.attributes["routed_rows"] == 0
        # Lowering omits W1/activation/W2 device math for an empty EP rank.
        # unquant_apply_mlp still calls all three APIs. Retain their Host
        # prepare and enqueue group after dispatch and before abs/combine.
        contract = (tuple(child_contract[index] for index in (0, 4, 5))
                    if empty_local else child_contract)
        source_ids = set()
        for index, (kind, variant, mechanisms) in enumerate(contract):
            call_id = f"{prefix}:experts:{index}"
            if call_id not in calls_by_id:
                raise ValueError("Ascend source MoE expert child is missing")
            child = calls_by_id[call_id]
            if (child.rank != rank or child.kernel_spec.kind is not kind
                    or child.kernel_spec.variant != variant):
                raise ValueError("Ascend source MoE expert child order or rank disagrees")
            if variant in ("moe_init_routing", "moe_abs", "moe_token_unpermute"):
                if (child.kernel_spec.attributes["mathematical_program"] != "ascend_0_18_moe_dispatch_math_v1"
                        or child.kernel_spec.attributes["routed_rows"] != first_spec.attributes["routed_rows"]):
                    raise ValueError("Ascend source MoE dispatch children disagree on local routed rows")
            source_ids.add(child.source_node_ids)
            result[call_id] = tuple((name, 1) for name in mechanisms)
            if empty_local and variant == "moe_abs":
                result[call_id] = (("moe_unquant_mlp_wrapper_prepare", 1),
                    ("moe_grouped_gemm_prepare", 2), ("moe_swiglu_prepare", 1)) + result[call_id]
        if len(source_ids) != 1:
            raise ValueError("Ascend source MoE expert children must share one wrapper source")
    return result


def _dense_node_dispatch_events(
    plan: KernelPlan,
    repeat_by_call,
    enabled: frozenset,
    source_moe_classes,
    ascend_compilation_mode: str,
) -> Mapping[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Charge the dense torchair graph entry and per-node class dispatch.

    Ascend0.18 FULL/FULL_DECODE_ONLY clears splitting_ops: the model enters
    one compiled graph at embedding, including above-capture-limit calls.
    PIECEWISE splits at attention and enters one further piece per layer.
    NONE disables compilation. The configured source mode, not the current
    step's replay mode, determines those eager graph-entry occurrences.
    Every plan call is mapped to its source node classes through a fail-closed
    table; a call outside the table aborts the mapping instead of
    borrowing a neighbouring class.  Llama's ``rotary-*`` helper children
    are folded into one compiled rotary source call and are skipped.

    MoE plans use the split-piece anchor descriptors above because fused
    MoE wrappers do not expose one node per lowered child call; charging
    both contracts would double those plans.
    """

    if any(":moe:" in call.call_id for call in plan.calls):
        return {}
    layer_ids = {
        int(match.group(1))
        for call in plan.calls
        for match in [re.search(r":layer-([0-9]+):", call.call_id)]
        if match is not None
    }
    if layer_ids and layer_ids != set(range(max(layer_ids) + 1)):
        raise ValueError("dense node dispatch layer descriptor is incomplete")
    graph_entry_calls = []
    post_attention_candidates = {}
    events = {}
    for call in plan.calls:
        if ":rotary-" in call.call_id or ":rotary:" in call.call_id:
            continue
        # LogitsProcessor/ModelRunner selection and Sampler are outside
        # model.forward. Their device calls retain ordinary launch pricing;
        # outer preparation and sampler submission are charged separately.
        if call.kernel_spec.kind is KernelKind.VECTOR and call.kernel_spec.variant in (
                "vllm_logits_row_select", "vllm_logits_fp32_cast", "vllm_plain_greedy_sample",
                "vllm_logits_gather_reorder"):
            continue
        if call.kernel_spec.kind is KernelKind.COMMUNICATION:
            # TP collective nodes are not in the single-rank census that
            # pinned this table, and their Host lane is already priced by
            # the collective prepare mechanisms (all_reduce/all_gather);
            # mapping them to a compute node class would borrow an
            # unmeasured class and double-charge the same lane.
            continue
        class_key = (call.kernel_spec.kind, call.kernel_spec.variant)
        if call.call_id in source_moe_classes:
            node_classes = source_moe_classes[call.call_id]
            missing = {name for name, _ in node_classes} - enabled
            if missing:
                raise ValueError(f"Ascend source MoE Host dispatch lacks independent measurements: {sorted(missing)!r}")
        elif class_key == (KernelKind.VECTOR, "vllm_masked_embedding"):
            node_classes = _ascend_masked_embedding_node_classes(call.kernel_spec)
            missing = {name for name, _ in node_classes} - enabled
            if missing:
                raise ValueError(
                    "Ascend TP embedding Host dispatch lacks independent primitive "
                    f"measurements: {sorted(missing)!r}")
        elif class_key not in _ASCEND_910B_DENSE_NODE_CLASSES:
            raise ValueError(
                "dense node dispatch has no node class for "
                f"({call.kernel_spec.kind.name}, "
                f"{call.kernel_spec.variant!r}) at {call.call_id!r}")
        else:
            node_classes = _ASCEND_910B_DENSE_NODE_CLASSES[class_key]
        if _ASCEND_910B_TOKEN_EMBEDDING_ANCHOR.search(call.call_id) is not None:
            graph_entry_calls.append(call)
        match = re.search(r":layer-([0-9]+):(attention:output-copy|o-proj)$", call.call_id)
        if match is not None:
            post_attention_candidates[(int(match.group(1)), call.rank, match.group(2))] = call
        selected = tuple(
            (mechanism, count) for mechanism, count in node_classes
            if mechanism in enabled)
        if not selected:
            continue
        occurrence_indices = (
            range(repeat_by_call[call.call_id].repeat_count)
            if call.call_id in repeat_by_call else (-1,)
        )
        for occurrence_index in occurrence_indices:
            for pool_name, rank in call_host_lanes(call):
                key = (pool_name, rank, call.call_id, occurrence_index)
                if key in events:
                    raise ValueError(
                        "dense node dispatch occurrence is duplicated")
                # production_node_dispatch measures each native operator
                # entry through its asynchronous return, including enqueue.
                # The atomic *_host_submit mechanisms use that same boundary.
                # Existing MoE *_prepare prices remain increments over enqueue.
                complete_submit = any(
                    mechanism.endswith(("_host_submit", "_node_dispatch"))
                    for mechanism, _ in selected)
                events[key] = tuple(
                    RuntimePrepareEvent(mechanism, count, replaces_eager_enqueue=(
                        complete_submit and index == len(selected) - 1),
                        additional_eager_enqueue_count=(count if (
                            call.call_id in source_moe_classes
                            and call.kernel_spec.variant == "moe_abs"
                            and mechanism in ("moe_grouped_gemm_prepare", "moe_swiglu_prepare")) else 0))
                    for index, (mechanism, count) in enumerate(selected))
    ranks = {call.rank for call in plan.calls if call.rank is not None}
    expected_entry_layers = {(layer, rank) for layer in layer_ids for rank in ranks}
    entry_layers = {(layer, rank) for layer, rank, _ in post_attention_candidates}
    if layer_ids and (not graph_entry_calls or entry_layers != expected_entry_layers):
        raise ValueError(
            "dense node dispatch requires one post-attention graph-entry "
            "anchor per layer and Host lane")
    if ascend_compilation_mode == "PIECEWISE":
        for layer, rank in sorted(entry_layers):
            graph_entry_calls.append(post_attention_candidates[
                (layer, rank, "attention:output-copy") if
                (layer, rank, "attention:output-copy") in post_attention_candidates else
                (layer, rank, "o-proj")])
    elif ascend_compilation_mode == "NONE":
        graph_entry_calls.clear()
    elif ascend_compilation_mode in ("FULL", "FULL_DECODE_ONLY"):
        pass
    else:
        raise ValueError("unsupported Ascend source compilation mode")
    for call in graph_entry_calls:
        occurrence_indices = (
            range(repeat_by_call[call.call_id].repeat_count)
            if call.call_id in repeat_by_call else (-1,)
        )
        for occurrence_index in occurrence_indices:
            for pool_name, rank in call_host_lanes(call):
                key = (pool_name, rank, call.call_id, occurrence_index)
                if key in events:
                    current = events[key]
                    if any(event.mechanism == "compiled_fx_graph_entry"
                           for event in current):
                        raise ValueError(
                            "dense graph entry is duplicated")
                    events[key] = current + (
                        RuntimePrepareEvent("compiled_fx_graph_entry"),)
                else:
                    events[key] = (
                        RuntimePrepareEvent("compiled_fx_graph_entry"),)
    return events


VLLM030_V2_INPUT_PREPARE = "vllm030_v2_input_prepare"
WORKER_REQUEST_STATE_MECHANISMS = frozenset(WORKER_REQUEST_STATE_UNITS)


def worker_request_state_prepare_events(plan, state, *, enabled_mechanisms):
    """Source counts before V2 input preparation, once on every worker lane."""
    enabled = frozenset(enabled_mechanisms) & WORKER_REQUEST_STATE_MECHANISMS
    if not enabled:
        return {}
    if enabled != WORKER_REQUEST_STATE_MECHANISMS:
        raise ValueError("worker request state profile requires its complete service group")
    if not isinstance(state, WorkerRequestState):
        raise ValueError("worker request state profile requires native non-duration source counts")
    if (state.max_requests, state.max_model_len, state.kv_block_size_tokens) != (256, 8192, 16):
        raise ValueError("worker request state mechanism requires measured256 slots/8192 context/block16")
    entries = [("fixed", 1), ("cached_request", state.cached_requests)]
    if state.finished_requests:
        entries.extend((("finish_nonempty", 1), ("finished_request", state.finished_requests)))
    if state.admitted_requests:
        entries.extend((("admit_nonempty", 1), ("admitted_request", state.admitted_requests),
            ("admitted_token", state.admitted_token_ids)))
    events = tuple(RuntimePrepareEvent("vllm030_v2_state_" + name, count,
        graph_capturable=False) for name, count in entries if count)
    # The existing input primitive follows state mutation in execute_model.
    # These anchors describe one outer call per rank, not one call per layer.
    return {key: events for key in _vllm030_v2_input_prepare_events(plan)}


def _model_execution_versions(plan: KernelPlan):
    """Model-source versions, excluding unversioned generic primitives."""
    return frozenset(call.kernel_spec.attributes["model_execution_version"]
                     for call in plan.calls if "model_execution_version" in call.kernel_spec.attributes
                     and call.kernel_spec.attributes["model_execution_version"] is not None)


def _vllm030_v2_input_prepare_events(plan: KernelPlan):
    """One declared V2 preparation before the first submit on each worker.

    Selecting this versioned mechanism in a Host profile explicitly declares
    the V2 runner. Its independent measurement includes gather_batch_req_state,
    prepare_inputs and prepare_attn, excluding the model/AOT metadata builder.
    The compact model graph has no calls for these input kernels/copies;
    their CPU enqueue service therefore replaces no existing model submit.
    This Host service remains outside model capture and may overlap earlier
    GPU work through the existing submit-lane and device-floor schedule.
    """
    if {call.kernel_spec.hardware_id for call in plan.calls} != {"nvidia_h200"}:
        raise ValueError("V2 input preparation profile requires an H200-only plan")
    if _model_execution_versions(plan) != {"vllm_0_30_0"}:
        raise ValueError("V2 input preparation requires explicit vLLM0.30 model source")
    repeats = {call_id for repeat in plan.block_repeats for call_id in repeat.call_ids}
    first = {}
    for call in plan.topological_order():
        for lane in call_host_lanes(call):
            if lane not in first:
                first[lane] = (call, 0 if call.call_id in repeats else -1)
    if not first:
        raise ValueError("V2 input preparation requires a worker Host lane")
    return {(pool, rank, call.call_id, index): (
                RuntimePrepareEvent(VLLM030_V2_INPUT_PREPARE, graph_capturable=False),)
            for (pool, rank), (call, index) in first.items()}


def runtime_prepare_source_signature(plan: KernelPlan, prefix: str):
    """Source fields consumed by Host rules beyond the scheduling topology."""
    variants = frozenset(("vllm_masked_embedding", "moe_topk", "moe_init_routing", "routed_experts",
                          "w1", "moe_swiglu", "w2", "moe_abs", "moe_token_unpermute"))
    fields = ("stage", "mask_program", "tensor_parallel_size", "tensor_parallel_rank", "source_program",
              "mathematical_program", "routed_rows")
    local = tuple((call.call_id.removeprefix(prefix), call.kernel_spec.dtype,
                  tuple((name, call.kernel_spec.attributes[name]) for name in fields
                        if name in call.kernel_spec.attributes),
                  tuple(name.removeprefix(prefix) for name in call.source_node_ids))
                 for call in plan.calls if call.kernel_spec.variant in variants)
    return local, _model_execution_versions(plan)


def runtime_prepare_events(
    plan: KernelPlan,
    *,
    enabled_mechanisms: Tuple[str, ...],
    ascend_compilation_mode: str | None = None,
    worker_request_state: WorkerRequestState | None = None,
) -> Mapping[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Map source-declared runtime mechanisms to concrete Host-lane events.

    The engine is shared by every runtime stack; the stack's declarative
    rule set is selected by the plan's hardware identity.  Internal
    dispatch/GEMM/activation/reduction work covered by one measured
    fused-experts wrapper event is deliberately not charged again per
    child kernel.
    """

    if not isinstance(plan, KernelPlan):
        raise TypeError("runtime prepare mapping requires a KernelPlan")
    hardware_ids = {call.kernel_spec.hardware_id for call in plan.calls}
    if len(hardware_ids) != 1:
        raise ValueError("runtime prepare mapping requires one hardware")
    hardware_id = next(iter(hardware_ids))
    rule_set = None
    for candidate in _RUNTIME_PREPARE_RULE_SETS:
        if hardware_id in candidate.hardware_ids:
            rule_set = candidate
            break
    if rule_set is None:
        raise ValueError(
            "runtime prepare mapping has no rule set for hardware "
            f"{hardware_id!r}")
    if (not isinstance(enabled_mechanisms, tuple) or not enabled_mechanisms
            or any(not isinstance(value, str) or not value
                   for value in enabled_mechanisms)):
        raise TypeError("enabled runtime prepare mechanisms must be non-empty strings")
    enabled = frozenset(enabled_mechanisms)
    if len(enabled) != len(enabled_mechanisms):
        raise ValueError("enabled runtime prepare mechanisms must be unique")
    if ascend_compilation_mode is not None:
        if hardware_id != "ascend_910b":
            raise ValueError("Ascend compilation mode requires ascend_910b")
        if ascend_compilation_mode not in ("NONE", "FULL", "FULL_DECODE_ONLY", "PIECEWISE"):
            raise ValueError("unsupported Ascend source compilation mode")
    if (hardware_id == "ascend_910b" and "compiled_fx_graph_entry" in enabled
            and ascend_compilation_mode is None):
        raise ValueError("Ascend Host graph entries require the configured source compilation mode")
    for exclusive_group in rule_set.exclusive_mechanism_groups:
        if set(exclusive_group) <= enabled:
            raise ValueError(
                "enabled runtime prepare mechanisms overlap within an "
                f"exclusive group: {sorted(exclusive_group)!r}")

    rule_by_kind_variant = {}
    for rule in rule_set.variant_rules:
        for variant in rule.variants:
            key = (rule.kernel_kind, variant)
            if key in rule_by_kind_variant:
                raise AssertionError("runtime prepare rules overlap")
            rule_by_kind_variant[key] = rule
    gated_rules = {
        rule
        for rule in rule_set.variant_rules
        if rule.gate_pattern is not None and not any(
            re.search(rule.gate_pattern, call.call_id) is not None
            for call in plan.calls)
    }
    collective_mechanisms = dict(rule_set.collective_mechanisms)
    repeat_by_call = {
        call_id: repeat
        for repeat in plan.block_repeats
        for call_id in repeat.call_ids
    }
    source_moe_classes = (_ascend_source_moe_node_classes(plan, ascend_compilation_mode)
                          if hardware_id == "ascend_910b" else {})
    if source_moe_classes and ("compiled_fx_graph_entry" not in enabled
            or any(":moe:" in call.call_id for call in plan.calls)):
        raise ValueError("Ascend source MoE requires one explicit source graph contract, without legacy split anchors")
    events = {}
    for call in plan.calls:
        if call.call_id in source_moe_classes:
            # The source-aware descriptor below covers these calls in full;
            # generic legacy rules would charge owner masking again at W1.
            continue
        mechanisms = []
        rule_key = (call.kernel_spec.kind, call.kernel_spec.variant)
        if rule_key in rule_by_kind_variant:
            rule = rule_by_kind_variant[rule_key]
            if rule not in gated_rules and (
                    rule.call_id_pattern is None
                    or re.search(rule.call_id_pattern, call.call_id)) and all(
                        name in call.kernel_spec.attributes and call.kernel_spec.attributes[name] in values
                        for name, values in rule.attribute_constraints):
                mechanisms.extend(rule.mechanisms)
        if call.kernel_spec.kind is KernelKind.COMMUNICATION:
            transfer = call.kernel_spec.transfer
            if transfer is None:
                raise ValueError("communication call requires a transfer descriptor")
            if transfer.operation in collective_mechanisms:
                mechanisms.append(collective_mechanisms[transfer.operation])
        selected = tuple(
            mechanism for mechanism in mechanisms if mechanism in enabled)
        if not selected:
            continue
        occurrence_indices = (
            range(repeat_by_call[call.call_id].repeat_count)
            if call.call_id in repeat_by_call else (-1,)
        )
        for occurrence_index in occurrence_indices:
            for pool_name, rank in call_host_lanes(call):
                key = (pool_name, rank, call.call_id, occurrence_index)
                if key in events:
                    raise AssertionError(
                        "runtime prepare occurrence is duplicated")
                events[key] = tuple(
                    RuntimePrepareEvent(mechanism) for mechanism in selected)

    repeat_indices = repeat_indices_from_plan(plan)
    for group in rule_set.anchor_groups:
        if group.mechanism not in enabled:
            continue
        if (group.gate_pattern is not None and not any(
                re.search(group.gate_pattern, call.call_id) is not None
                for call in plan.calls)):
            continue
        _merge_runtime_prepare_events(
            events, _anchor_group_events(plan, group, repeat_indices))
    for wrapper in rule_set.wrapper_anchors:
        if wrapper.mechanism not in enabled:
            continue
        if (wrapper.gate_pattern is not None and not any(
                re.search(wrapper.gate_pattern, call.call_id) is not None
                for call in plan.calls)):
            continue
        _merge_runtime_prepare_events(
            events, _wrapper_anchor_events(plan, wrapper, repeat_by_call))
    if (hardware_id == "ascend_910b"
            and "compiled_fx_graph_entry" in enabled):
        _merge_runtime_prepare_events(
            events, _dense_node_dispatch_events(
                plan, repeat_by_call, enabled, source_moe_classes,
                ascend_compilation_mode))
    if (hardware_id == "ascend_910b"
            and "compiled_fx_rotary_composite_entry" in enabled
            and any(":rotary-" in call.call_id or ":rotary:" in call.call_id for call in plan.calls)):
        _merge_runtime_prepare_events(
            events, ascend_910b_compiled_fx_rotary_events(plan))
    if (hardware_id == "ascend_910b"
            and "rotary_cache_update_prepare" in enabled):
        _merge_runtime_prepare_events(
            events, ascend_910b_rotary_cache_update_events(plan))
    if (hardware_id == "ascend_910b"
            and "compiled_fx_aten_reshape_metadata" in enabled
            and any(
                re.search(
                    r":layer-[0-9]+:(?:attention|o-proj|router)"
                    r"(?::tp-rank-[0-9]+)?(?::[0-9]+)?$",
                    call.call_id,
                ) is not None
                for call in plan.calls
            )):
        _merge_runtime_prepare_events(
            events, ascend_910b_compiled_fx_reshape_events(plan))
    if (hardware_id == "ascend_910b"
            and ({"compiled_fx_aten_view_metadata", "compiled_fx_aten_empty_metadata"}
                 & enabled)
            and any(":layer-" in call.call_id for call in plan.calls)):
        metadata_events = _retain_enabled_mechanisms(
            ascend_910b_compiled_fx_dense_metadata_events(plan),
            ("compiled_fx_aten_view_metadata",
             "compiled_fx_aten_empty_metadata"),
            enabled, drop_empty=False)
        _merge_runtime_prepare_events(events, metadata_events)
    if (hardware_id == "ascend_910b"
            and {
                "model_runner_input_prepare",
                "model_runner_forward_metadata_prepare",
                "model_runner_logits_prepare",
            } & enabled):
        outer_events = _retain_enabled_mechanisms(
            _ascend_910b_model_runner_outer_events(plan),
            ("model_runner_input_prepare",
             "model_runner_forward_metadata_prepare",
             "model_runner_logits_prepare"),
            enabled, drop_empty=True)
        _merge_runtime_prepare_events(events, outer_events)
    if VLLM030_V2_INPUT_PREPARE in enabled:
        _merge_runtime_prepare_events(events, _vllm030_v2_input_prepare_events(plan))
    state_events = worker_request_state_prepare_events(plan, worker_request_state,
        enabled_mechanisms=enabled)
    for key, mechanisms in state_events.items():
        events[key] = mechanisms + (events[key] if key in events else ())
    return events
