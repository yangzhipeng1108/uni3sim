"""Source-static compiled-FX occurrence contracts for Ascend Serving.

These descriptors count source-authored generated-Python entries only.  They
never contain timing values; independent runtime microbench profiles provide
the service for each mechanism.
"""

from __future__ import annotations

import re

from src.serving.host_timing import RuntimePrepareEvent
from src.serving.kernel_lowering.execution_timeline import call_host_lanes
from src.serving.kernel_lowering.kernel_plan import KernelPlan


_LAYER_PATTERN = re.compile(r":layer-([01]):(.+)$")
RANKED_MULTIPLICITY = (
    ("input-norm:tp-rank-", 2),
    ("qkv:tp-rank-", 1),
    ("q-norm:tp-rank-", 3),
    ("attention:tp-rank-", 1),
    ("o-proj:tp-rank-", 1),
    ("post-attn-norm:tp-rank-", 2),
    ("router:tp-rank-", 1),
)
UNRANKED_MULTIPLICITY = (
    ("o-proj:tp-all-reduce", 1),
    ("moe:ep-output-all-reduce", 1),
)
_MECHANISM = "compiled_fx_backend_entry"
_RESHAPE_MECHANISM = "compiled_fx_aten_reshape_metadata"
_VIEW_MECHANISM = "compiled_fx_aten_view_metadata"
_EMPTY_MECHANISM = "compiled_fx_aten_empty_metadata"
_ROTARY_MECHANISM = "compiled_fx_rotary_composite_entry"
_MOE_RESHAPE_ANCHORS = (("attention", 3), ("o-proj", 2), ("router", 1))
_DENSE_RESHAPE_ANCHORS = (("attention", 3), ("o-proj", 2))


def repeat_indices_from_plan(plan: KernelPlan) -> dict[str, tuple[int, ...]]:
    """Repeat index set per call id: ``(-1,)`` outside a repeat block."""

    indices = {call.call_id: (-1,) for call in plan.calls}
    for repeat in plan.block_repeats:
        for call_id in repeat.call_ids:
            indices[call_id] = tuple(range(repeat.repeat_count))
    return indices


def _ranked_multiplicity(call_id: str, rank: int) -> int | None:
    if call_id.endswith(f"moe:ep-rank-{rank}:0"):
        return 1
    for prefix, multiplicity in RANKED_MULTIPLICITY:
        if call_id.endswith(f"{prefix}{rank}"):
            return multiplicity
    return None


def require_compiled_fx_plan(plan, descriptor: str) -> None:
    """Require the frozen Ascend compiled-FX plan shape.

    The compiled-FX contracts describe one Ascend 910B plan only; any other
    plan is unsupported rather than a look-up failure.
    """

    if not isinstance(plan, KernelPlan):
        raise TypeError(f"{descriptor} requires a KernelPlan")
    if {call.kernel_spec.hardware_id for call in plan.calls} != {"ascend_910b"}:
        raise ValueError(f"{descriptor} requires an Ascend-only plan")


def ascend_910b_compiled_fx_backend_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Expand the frozen TP2 Qwen3-MoE compiled-FX source contract."""
    require_compiled_fx_plan(plan, "compiled-FX descriptor")
    repeat_indices = repeat_indices_from_plan(plan)
    layer_calls = {0: [], 1: []}
    endpoints = {}
    for call in plan.calls:
        match = _LAYER_PATTERN.search(call.call_id)
        if match is not None:
            layer_calls[int(match.group(1))].append(call)
            continue
        if ":layer-" in call.call_id:
            raise ValueError("compiled-FX descriptor does not support this layer ID")
        for prefix in ("token-embedding:tp-rank-", "final-norm:tp-rank-"):
            if prefix in call.call_id:
                suffix = call.call_id.rsplit(prefix, 1)[1]
                if not suffix.isdigit():
                    raise ValueError("compiled-FX endpoint rank is malformed")
                key = (prefix, int(suffix))
                if key in endpoints:
                    raise ValueError("compiled-FX endpoint is duplicated")
                endpoints[key] = call
    if not layer_calls[0] or not layer_calls[1]:
        raise ValueError("compiled-FX descriptor requires layers zero and one")
    template_repeats = {repeat_indices[call.call_id] for call in layer_calls[1]}
    if len(template_repeats) != 1 or next(iter(template_repeats)) == (-1,):
        raise ValueError("compiled-FX layer-one calls must share one repeat")
    events = {}
    for calls in layer_calls.values():
        ranked, unranked = {}, {}
        for call in calls:
            multiplicity = _ranked_multiplicity(call.call_id, call.rank)
            if multiplicity is not None:
                ranked[(call.rank, call.call_id)] = (call, multiplicity)
                continue
            for suffix, count in UNRANKED_MULTIPLICITY:
                if call.call_id.endswith(suffix):
                    if suffix in unranked:
                        raise ValueError("compiled-FX collective anchor duplicates")
                    unranked[suffix] = (call, count)
                    break
        ranks = {rank for rank, _ in ranked}
        if ranks != {0, 1}:
            raise ValueError("compiled-FX descriptor requires TP ranks zero and one")
        for rank in ranks:
            expected = {f"input-norm:tp-rank-{rank}", f"qkv:tp-rank-{rank}",
                        f"q-norm:tp-rank-{rank}", f"attention:tp-rank-{rank}",
                        f"o-proj:tp-rank-{rank}",
                        f"post-attn-norm:tp-rank-{rank}",
                        f"router:tp-rank-{rank}", f"moe:ep-rank-{rank}:0"}
            actual = {
                re.search(r":layer-[01]:(.+)$", call_id).group(1)
                for event_rank, call_id in ranked
                if event_rank == rank
            }
            if actual != expected:
                raise ValueError("compiled-FX ranked decoder anchors are incomplete")
        if set(unranked) != {suffix for suffix, _ in UNRANKED_MULTIPLICITY}:
            raise ValueError("compiled-FX collective decoder anchors are incomplete")
        for (rank, call_id), (call, count) in ranked.items():
            for index in repeat_indices[call_id]:
                for pool_name, lane_rank in call_host_lanes(call):
                    if lane_rank != rank:
                        raise AssertionError("rank-local decoder anchor changed lane")
                    events[(pool_name, rank, call_id, index)] = (
                        RuntimePrepareEvent(_MECHANISM, count),)
        for call, count in unranked.values():
            for index in repeat_indices[call.call_id]:
                for pool_name, rank in call_host_lanes(call):
                    events[(pool_name, rank, call.call_id, index)] = (
                        RuntimePrepareEvent(_MECHANISM, count),)
    if {rank for _, rank in endpoints} != {0, 1} or len(endpoints) != 4:
        raise ValueError("compiled-FX descriptor requires both endpoint ranks")
    for call in endpoints.values():
        for pool_name, rank in call_host_lanes(call):
            events[(pool_name, rank, call.call_id, -1)] = (
                RuntimePrepareEvent(_MECHANISM),)
    return events


def ascend_910b_compiled_fx_reshape_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Count source-authored ``aten.reshape`` metadata nodes per layer/rank.

    Qwen3-MoE has attention/o-proj/router anchors (six nodes); dense decoder
    layers have the attention/o-proj subset (five nodes).  Split lowering
    children are accepted only when their aggregate count remains the static
    source count.
    """
    require_compiled_fx_plan(plan, "compiled-FX reshape descriptor")
    repeat_indices = repeat_indices_from_plan(plan)
    has_moe = any(":moe:" in call.call_id for call in plan.calls)
    has_source_moe = any(call.kernel_spec.variant == "moe_topk"
        and call.kernel_spec.attributes["source_program"] == "vllm_0_18_bf16_softmax_topk" for call in plan.calls)
    anchor_definitions = (
        _MOE_RESHAPE_ANCHORS if has_moe or has_source_moe else _DENSE_RESHAPE_ANCHORS)
    # Endpoint-only plans (for example a transfer boundary without a decoder
    # body) have no generated FX metadata anchors.  They consume no metadata
    # service; decoder-shaped plans are validated strictly below.
    if not any(
        re.search(
            r":layer-[0-9]+:(?:attention|o-proj|router)"
            r"(?::tp-rank-[0-9]+)?(?::[0-9]+)?$",
            call.call_id,
        )
        for call in plan.calls
    ):
        return {}
    layer_pattern = _LAYER_PATTERN if has_moe else re.compile(r":layer-([0-9]+):(.+)$")
    anchors = {}
    for call in plan.calls:
        match = layer_pattern.search(call.call_id)
        if match is None:
            continue
        layer, suffix = int(match.group(1)), match.group(2)
        for name, count in anchor_definitions:
            expected = f"{name}:tp-rank-{call.rank}"
            dense_expected = (name, re.fullmatch(re.escape(name) + r":[0-9]+", suffix))
            if (suffix == expected
                    or re.fullmatch(re.escape(expected) + r":[0-9]+", suffix)
                    or (not has_moe and (suffix == name or dense_expected[1]))):
                key = (layer, call.rank, name)
                if key not in anchors:
                    anchors[key] = [count]
                anchors[key].append(call)
                break
    layers = {key[0] for key in anchors}
    if (has_moe and layers != {0, 1}) or (not has_moe and (
            not layers or layers != set(range(max(layers) + 1)))):
        raise ValueError("compiled-FX reshape descriptor has incomplete source layers")
    ranks = {rank for _, rank, _ in anchors}
    expected = {(layer, rank, name) for layer in layers for rank in ranks
                for name, _ in anchor_definitions}
    if set(anchors) != expected:
        raise ValueError(
            "compiled-FX reshape anchors are incomplete: "
            f"expected={sorted(expected)!r} actual={sorted(anchors)!r}")
    events = {}
    for (layer, expected_rank, _), values in anchors.items():
        count, calls = values[0], tuple(values[1:])
        if len(calls) > count:
            raise ValueError(
                "compiled-FX reshape anchor has too many split calls: "
                f"key={(layer, expected_rank)!r} count={count} calls={len(calls)}")
        counts = [1] * len(calls)
        counts[0] += count - len(calls)
        for call, event_count in zip(calls, counts):
            indices = repeat_indices[call.call_id]
            if has_moe and ((layer == 0 and indices != (-1,))
                    or (layer == 1 and indices == (-1,))):
                raise ValueError("compiled-FX reshape layer repeat contract is invalid")
            for index in indices:
                for pool_name, rank in call_host_lanes(call):
                    if rank != expected_rank:
                        raise ValueError("compiled-FX reshape anchor moved Host rank")
                    key = (pool_name, rank, call.call_id, index)
                    if key in events:
                        raise AssertionError("compiled-FX reshape event is duplicated")
                    events[key] = (RuntimePrepareEvent(_RESHAPE_MECHANISM, event_count),)
    return events


def ascend_910b_compiled_fx_dense_metadata_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Count dense Qwen/Llama source metadata operations in eager fallback.

    The vLLM source has two statically distinct dense attention contracts:
    Qwen3's q/k head normalization adds four ``view`` nodes at the QKV
    boundary, while Llama's path does not.  Both paths enter the shared
    attention helper, whose source contains five ``view`` and one ``empty``
    metadata operations.  Numeric attention split children are one source
    attention call and therefore partition, rather than multiply, that count.
    """
    require_compiled_fx_plan(plan, "dense FX metadata descriptor")
    if any(":moe:" in call.call_id for call in plan.calls):
        return {}
    layer_pattern = re.compile(r":layer-([0-9]+):(.+)$")
    layers: dict[int, list[object]] = {}
    for call in plan.calls:
        match = layer_pattern.search(call.call_id)
        if match is not None:
            layer = int(match.group(1))
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(call)
    if not layers:
        return {}
    if set(layers) != set(range(max(layers) + 1)):
        raise ValueError("dense FX metadata descriptor has incomplete layers")
    repeats = repeat_indices_from_plan(plan)
    events = {}
    for layer, calls in layers.items():
        qkv_by_rank = {}
        attention_by_rank = {}
        has_q_norm = False
        for call in calls:
            suffix = layer_pattern.search(call.call_id).group(2)
            rank = call.rank
            if suffix == "qkv" or re.fullmatch(r"qkv:tp-rank-[0-9]+", suffix):
                if rank in qkv_by_rank:
                    raise ValueError("dense FX metadata qkv anchor is duplicated")
                qkv_by_rank[rank] = call
            if (suffix == "q-norm"
                    or re.fullmatch(r"q-norm:tp-rank-[0-9]+", suffix)):
                has_q_norm = True
            if (suffix == "attention"
                    or re.fullmatch(r"attention(?::[0-9]+)?", suffix)
                    or re.fullmatch(r"attention:tp-rank-[0-9]+(?::[0-9]+)?", suffix)):
                if rank not in attention_by_rank:
                    attention_by_rank[rank] = []
                attention_by_rank[rank].append(call)
        ranks = set(qkv_by_rank) | set(attention_by_rank)
        if ranks != set(qkv_by_rank) or not ranks:
            raise ValueError("dense FX metadata anchors are incomplete")
        for rank in ranks:
            qkv = qkv_by_rank[rank]
            attention_calls = (
                attention_by_rank[rank] if rank in attention_by_rank else ())
            if not attention_calls:
                raise ValueError("dense FX metadata attention anchor is incomplete")
            qkv_view_count = 4 if has_q_norm else 0
            counts = [1] * len(attention_calls)
            counts[0] += 5 - len(attention_calls)
            if len(attention_calls) > 5:
                raise ValueError("dense FX metadata has too many attention splits")
            for attention_index, (call, attention_count) in enumerate(
                    zip(attention_calls, counts)):
                indices = repeats[call.call_id]
                for index in indices:
                    for pool_name, lane_rank in call_host_lanes(call):
                        if lane_rank != rank:
                            raise ValueError("dense FX metadata attention moved Host rank")
                        key = (pool_name, rank, call.call_id, index)
                        metadata = [
                            RuntimePrepareEvent(_VIEW_MECHANISM, attention_count),
                        ]
                        if attention_index == 0:
                            metadata.append(RuntimePrepareEvent(_EMPTY_MECHANISM))
                        events[key] = tuple(metadata)
            for index in repeats[qkv.call_id]:
                for pool_name, lane_rank in call_host_lanes(qkv):
                    if lane_rank != rank:
                        raise ValueError("dense FX metadata qkv moved Host rank")
                    key = (pool_name, rank, qkv.call_id, index)
                    if qkv_view_count:
                        events[key] = (RuntimePrepareEvent(_VIEW_MECHANISM, qkv_view_count),)
    return events


def ascend_910b_compiled_fx_rotary_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Charge one fallback-dispatch entry for each native RoPE helper call.

    Llama's out-of-tree ``Llama3RotaryEmbedding`` is not registered by the
    Ascend plugin.  vLLM therefore executes the native composite helper, and
    lowering expands its tensor operations into a ``rotary-*`` child group.
    Those children are physical device launches and are already charged by
    the normal eager submit path; the source helper itself contributes one
    compiled-FX fallback entry.  The ``rotary-cos-sin-gather`` child is the
    stable source anchor for that one entry.  Other dense paths either use a
    registered fused RoPE op or the QKV/RMSNorm/RoPE fusion and expose no
    native-composite group, so they consume no event here.
    """
    require_compiled_fx_plan(plan, "compiled-FX rotary descriptor")
    if any(":moe:" in call.call_id for call in plan.calls):
        return {}
    layer_pattern = re.compile(r":layer-([0-9]+):(.+)$")
    anchors: dict[tuple[int, int], object] = {}
    groups: dict[tuple[int, int], list[object]] = {}
    for call in plan.calls:
        match = layer_pattern.search(call.call_id)
        if match is None:
            continue
        layer = int(match.group(1))
        suffix = match.group(2)
        if not suffix.startswith(("rotary-", "rotary:")):
            continue
        key = (layer, call.rank)
        if key not in groups:
            groups[key] = []
        groups[key].append(call)
        if suffix in ("rotary-cos-sin-gather", "rotary:cos-sin"):
            if key in anchors:
                raise ValueError(
                    "compiled-FX rotary source anchor is duplicated")
            anchors[key] = call
    if not groups:
        return {}
    if set(anchors) != set(groups):
        raise ValueError(
            "compiled-FX rotary source anchor is incomplete")
    layers = {layer for layer, _ in groups}
    if layers != set(range(max(layers) + 1)):
        raise ValueError(
            "compiled-FX rotary descriptor has incomplete layers")
    repeats = repeat_indices_from_plan(plan)
    events = {}
    for (layer, expected_rank), call in anchors.items():
        for index in repeats[call.call_id]:
            for pool_name, lane_rank in call_host_lanes(call):
                if lane_rank != expected_rank:
                    raise ValueError(
                        "compiled-FX rotary anchor moved Host rank")
                key = (pool_name, lane_rank, call.call_id, index)
                if key in events:
                    raise AssertionError(
                        "compiled-FX rotary event is duplicated")
                events[key] = (
                    RuntimePrepareEvent(_ROTARY_MECHANISM),)
    return events
