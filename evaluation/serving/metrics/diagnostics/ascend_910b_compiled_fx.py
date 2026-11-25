"""Source-static compiled-FX backend-entry descriptor for 910B Qwen3MoE.

The descriptor counts generated ``torch.ops`` entries from the frozen vLLM
compiled GraphModule.  It has no duration: the generic backend-entry service
is supplied independently by the synthetic compiled-FX microbenchmark.
"""

from __future__ import annotations

import re

from src.serving.host_timing import RuntimePrepareEvent
from src.serving.kernel_lowering.execution_timeline import call_host_lanes
from src.serving.kernel_lowering.kernel_plan import KernelPlan
from src.serving.compiled_fx import (
    RANKED_MULTIPLICITY,
    UNRANKED_MULTIPLICITY,
    require_compiled_fx_plan,
    ascend_910b_compiled_fx_backend_events as _production_backend_events,
    ascend_910b_compiled_fx_reshape_events as _production_reshape_events,
)


_LAYER_PATTERN = re.compile(r":layer-([01]):(.+)$")
_MECHANISM = "compiled_fx_backend_entry"
_RESHAPE_MECHANISM = "compiled_fx_aten_reshape_metadata"
_RESHAPE_ANCHORS = (
    ("attention", 3),
    ("o-proj", 2),
    ("router", 1),
)


def _repeat_indices(plan: KernelPlan) -> dict[str, tuple[int, ...]]:
    indices = {call.call_id: (-1,) for call in plan.calls}
    for repeat in plan.block_repeats:
        for call_id in repeat.call_ids:
            indices[call_id] = tuple(range(repeat.repeat_count))
    return indices


def _ranked_multiplicity(call_id: str, rank: int) -> int | None:
    expected_moe = f"moe:ep-rank-{rank}:0"
    if call_id.endswith(expected_moe):
        return 1
    for prefix, multiplicity in RANKED_MULTIPLICITY:
        if call_id.endswith(f"{prefix}{rank}"):
            return multiplicity
    return None


def ascend_910b_compiled_fx_backend_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Expand the source-static 14 decoder entries/layer plus two endpoints.

    This applies only to the compact Qwen3MoE KernelPlan used by the frozen
    Ascend runtime: explicit layer zero plus one repeated layer-one template.
    It rejects any other plan rather than silently guessing a compiler graph.
    """

    require_compiled_fx_plan(plan)
    repeat_indices = _repeat_indices(plan)
    layer_calls = {0: [], 1: []}
    top_level_ranked = {}
    for call in plan.calls:
        match = _LAYER_PATTERN.search(call.call_id)
        if match is not None:
            layer_calls[int(match.group(1))].append(call)
            continue
        if ":layer-" in call.call_id:
            raise ValueError("compiled-FX descriptor does not support this layer ID")
        for endpoint in ("token-embedding:tp-rank-", "final-norm:tp-rank-"):
            if endpoint in call.call_id:
                suffix = call.call_id.rsplit(endpoint, 1)[1]
                if not suffix.isdigit():
                    raise ValueError("compiled-FX endpoint rank is malformed")
                key = (endpoint, int(suffix))
                if key in top_level_ranked:
                    raise ValueError("compiled-FX endpoint is duplicated")
                top_level_ranked[key] = call
    if not layer_calls[0] or not layer_calls[1]:
        raise ValueError("compiled-FX descriptor requires layers zero and one")
    template_repeats = {
        repeat_indices[call.call_id] for call in layer_calls[1]
    }
    if len(template_repeats) != 1 or next(iter(template_repeats)) == (-1,):
        raise ValueError("compiled-FX layer-one calls must share one repeat")
    events = {}
    for layer_index, calls in layer_calls.items():
        ranked_counts = {}
        unranked_counts = {}
        for call in calls:
            ranked = _ranked_multiplicity(call.call_id, call.rank)
            if ranked is not None:
                key = (call.rank, call.call_id)
                ranked_counts[key] = ranked
                continue
            for suffix, multiplicity in UNRANKED_MULTIPLICITY:
                if call.call_id.endswith(suffix):
                    if suffix in unranked_counts:
                        raise ValueError("compiled-FX collective anchor duplicates")
                    unranked_counts[suffix] = (call, multiplicity)
                    break
        ranks = {call.rank for call in calls if call.rank in (0, 1)}
        if ranks != {0, 1}:
            raise ValueError("compiled-FX descriptor requires TP ranks zero and one")
        for rank in ranks:
            expected = {
                f"input-norm:tp-rank-{rank}",
                f"qkv:tp-rank-{rank}",
                f"q-norm:tp-rank-{rank}",
                f"attention:tp-rank-{rank}",
                f"o-proj:tp-rank-{rank}",
                f"post-attn-norm:tp-rank-{rank}",
                f"router:tp-rank-{rank}",
                f"moe:ep-rank-{rank}:0",
            }
            actual = {
                _LAYER_PATTERN.search(call_id).group(2)
                for event_rank, call_id in ranked_counts
                if event_rank == rank
            }
            if actual != expected:
                raise ValueError(
                    "compiled-FX ranked decoder anchors are incomplete")
        if set(unranked_counts) != {
                suffix for suffix, _ in UNRANKED_MULTIPLICITY}:
            raise ValueError("compiled-FX collective decoder anchors are incomplete")
        for (rank, call_id), multiplicity in ranked_counts.items():
            call = next(call for call in calls if call.call_id == call_id)
            for occurrence_index in repeat_indices[call.call_id]:
                for pool_name, lane_rank in call_host_lanes(call):
                    if lane_rank != rank:
                        raise AssertionError("rank-local decoder anchor changed lane")
                    events[(pool_name, lane_rank, call.call_id, occurrence_index)] = (
                        RuntimePrepareEvent(_MECHANISM, multiplicity),)
        for call, multiplicity in unranked_counts.values():
            for occurrence_index in repeat_indices[call.call_id]:
                for pool_name, rank in call_host_lanes(call):
                    events[(pool_name, rank, call.call_id, occurrence_index)] = (
                        RuntimePrepareEvent(_MECHANISM, multiplicity),)
    endpoint_ranks = {
        rank for _, rank in top_level_ranked
    }
    if endpoint_ranks != {0, 1} or len(top_level_ranked) != 4:
        raise ValueError("compiled-FX descriptor requires both endpoint ranks")
    for call in top_level_ranked.values():
        for pool_name, rank in call_host_lanes(call):
            events[(pool_name, rank, call.call_id, -1)] = (
                RuntimePrepareEvent(_MECHANISM),)
    return events


def ascend_910b_compiled_fx_reshape_events(
    plan: KernelPlan,
) -> dict[tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]]:
    """Place the six static ``aten.reshape`` nodes in every decoder layer.

    The exported 910B Qwen3MoE compiled FX GraphModule has, per TP rank and
    decoder layer, three reshapes after QKV before attention, two after
    attention before O projection, and one before router GEMM.  The compact
    KernelPlan's explicit layer zero plus repeated layer-one template is the
    source-static counterpart; other plans fail closed rather than assuming a
    compiler graph layout.
    """

    if not isinstance(plan, KernelPlan):
        raise TypeError("compiled-FX reshape descriptor requires a KernelPlan")
    if {call.call_id for call in plan.calls} == set():
        raise ValueError("compiled-FX reshape descriptor requires plan calls")
    if {call.kernel_spec.hardware_id for call in plan.calls} != {"ascend_910b"}:
        raise ValueError(
            "compiled-FX reshape descriptor requires an Ascend-only plan")
    repeat_indices = _repeat_indices(plan)
    anchors = {}
    for call in plan.calls:
        match = _LAYER_PATTERN.search(call.call_id)
        if match is None:
            continue
        layer = int(match.group(1))
        suffix = match.group(2)
        for name, count in _RESHAPE_ANCHORS:
            expected = f"{name}:tp-rank-{call.rank}"
            if suffix != expected:
                continue
            key = (layer, call.rank, name)
            if key in anchors:
                raise ValueError("compiled-FX reshape anchor is duplicated")
            anchors[key] = (call, count)
            break
    expected = {
        (layer, rank, name)
        for layer in (0, 1)
        for rank in (0, 1)
        for name, _ in _RESHAPE_ANCHORS
    }
    if set(anchors) != expected:
        raise ValueError("compiled-FX reshape anchors are incomplete")
    events = {}
    for (layer, expected_rank, _), (call, count) in anchors.items():
        indices = repeat_indices[call.call_id]
        if layer == 0 and indices != (-1,):
            raise ValueError("compiled-FX layer-zero reshape anchor cannot repeat")
        if layer == 1 and indices == (-1,):
            raise ValueError("compiled-FX layer-one reshape anchor must repeat")
        for occurrence_index in indices:
            for pool_name, rank in call_host_lanes(call):
                if rank != expected_rank:
                    raise ValueError("compiled-FX reshape anchor moved Host rank")
                key = (pool_name, rank, call.call_id, occurrence_index)
                if key in events:
                    raise AssertionError("compiled-FX reshape event is duplicated")
                events[key] = (RuntimePrepareEvent(_RESHAPE_MECHANISM, count),)
    return events


# Keep the historical evaluation import path as a compatibility shim.  The
# source occurrence contract is owned by ``src.serving.compiled_fx`` so
# production cannot depend on evaluation diagnostics.
ascend_910b_compiled_fx_backend_events = _production_backend_events
ascend_910b_compiled_fx_reshape_events = _production_reshape_events
