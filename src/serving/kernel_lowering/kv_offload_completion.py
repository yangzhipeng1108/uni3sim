"""Join one KV transfer intent to its canonical predicted timeline interval."""

import math
from typing import Mapping

from src.kernel.kernel_spec import KernelKind, KernelPrediction
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.kv_cache.kv_cache_tier_manager import (
    KVCachePhysicalTransferCompletion,
    KVCacheTransferCompletion,
)
from src.serving.workload.workload_graph import WorkloadGraph, WorkloadKind

from .execution_timeline import ExecutionTimeline
from .kernel_plan import KernelPlan


def build_kv_cache_transfer_completion(
    event: KVCacheEvent,
    workload: WorkloadGraph,
    plan: KernelPlan,
    predictions: Mapping[str, KernelPrediction],
    timeline: ExecutionTimeline,
    *,
    expected_pool_name: str,
    expected_ranks: tuple[int, ...],
) -> KVCacheTransferCompletion:
    if not isinstance(event, KVCacheEvent):
        raise TypeError("KV transfer completion requires a KVCacheEvent")
    if event.action not in (KVCacheAction.LOAD, KVCacheAction.EVICT,
                            KVCacheAction.STORE):
        raise ValueError("KV transfer completion requires load, evict or store intent")
    if not isinstance(workload, WorkloadGraph):
        raise TypeError("KV transfer completion requires a WorkloadGraph")
    if not isinstance(plan, KernelPlan):
        raise TypeError("KV transfer completion requires a KernelPlan")
    if not isinstance(predictions, Mapping):
        raise TypeError("KV transfer completion predictions must be a mapping")
    if not isinstance(timeline, ExecutionTimeline):
        raise TypeError("KV transfer completion requires an ExecutionTimeline")
    if not isinstance(expected_pool_name, str) or not expected_pool_name:
        raise ValueError(
            "KV transfer completion expected_pool_name must be non-empty")
    if not isinstance(expected_ranks, (list, tuple)):
        raise TypeError(
            "KV transfer completion expected_ranks must be a list or tuple")
    expected_ranks = tuple(expected_ranks)
    if (not expected_ranks
            or any(
                isinstance(rank, bool) or not isinstance(rank, int)
                or rank < 0
                for rank in expected_ranks)):
        raise ValueError(
            "KV transfer completion expected_ranks must be non-negative "
            "integers")
    if (expected_ranks != tuple(sorted(expected_ranks))
            or len(expected_ranks) != len(set(expected_ranks))):
        raise ValueError(
            "KV transfer completion expected_ranks must be unique and "
            "ascending")
    call_ids = {call.call_id for call in plan.calls}
    if set(predictions) != call_ids:
        raise ValueError(
            "KV transfer completion predictions must cover the plan exactly")
    interval_ids = {interval.call_id for interval in timeline.intervals}
    if len(interval_ids) != len(timeline.intervals) or interval_ids != call_ids:
        raise ValueError(
            "KV transfer completion timeline must cover the plan exactly")

    nodes = [
        node for node in workload.nodes
        if node.kind is WorkloadKind.KV_OFFLOAD
        and node.attributes["ready_event"] == event.ready_event
    ]
    if len(nodes) != len(expected_ranks):
        raise ValueError(
            "KV transfer intent must match exactly one workload node per "
            "expected rank")
    if any(node.rank is None for node in nodes):
        raise ValueError(
            "KV transfer workload nodes require explicit physical ranks")
    nodes = sorted(nodes, key=lambda item: item.rank)
    if (tuple(node.rank for node in nodes) != expected_ranks
            or any(node.pool_name != expected_pool_name for node in nodes)):
        raise ValueError(
            "KV transfer workload pool/ranks disagree with its owner")
    nodes_by_id = {node.node_id: node for node in workload.nodes}
    transfer_only = all(
        node.kind is WorkloadKind.KV_OFFLOAD for node in workload.nodes)
    for node in nodes:
        predecessors = tuple(
            nodes_by_id[edge.source]
            for edge in workload.edges
            if edge.destination == node.node_id
        )
        successors = tuple(
            nodes_by_id[edge.destination]
            for edge in workload.edges
            if edge.source == node.node_id
        )
        if event.action is KVCacheAction.LOAD:
            if predecessors or not any(
                    successor.kind is not WorkloadKind.KV_OFFLOAD
                    and successor.pool_name == expected_pool_name
                    and successor.rank == node.rank
                    for successor in successors):
                raise ValueError(
                    "KV load workload must be the dependency predecessor of "
                    "compute on every owner rank")
        elif event.action in (KVCacheAction.EVICT, KVCacheAction.STORE):
            if successors or not (event.action is KVCacheAction.STORE
                    and transfer_only and not predecessors) and not any(
                    predecessor.kind is not WorkloadKind.KV_OFFLOAD
                    and predecessor.pool_name == expected_pool_name
                    and (
                        predecessor.rank == node.rank
                        or (
                            predecessor.kind is WorkloadKind.COLLECTIVE
                            and node.rank
                            in predecessor.attributes["participants"]
                        )
                    )
                    for predecessor in predecessors):
                raise ValueError(
                    "KV evict workload must be the dependency successor of "
                    "compute on every owner rank")
        else:
            raise RuntimeError(
                f"unknown KV transfer action: {event.action.value!r}")

    physical_completions = []
    for node in nodes:
        calls = [
            call for call in plan.calls if call.call_id == node.node_id]
        intervals = [
            interval for interval in timeline.intervals
            if interval.call_id == node.node_id
        ]
        if len(calls) != 1 or len(intervals) != 1:
            raise ValueError(
                "KV transfer workload node must match one call and one "
                "interval")
        call = calls[0]
        interval = intervals[0]
        prediction = predictions[call.call_id]
        if call.kernel_spec.kind is not KernelKind.KV_OFFLOAD:
            raise ValueError("KV transfer call must use KernelKind.KV_OFFLOAD")
        physical = call.kernel_spec.transfer
        if physical is None:
            raise ValueError("KV transfer call lacks PhysicalTransferConfig")
        if (node.attributes["request_id"] != event.request_id
                or node.attributes["token_start"] != event.token_start
                or node.attributes["token_end"] != event.token_end
                or node.attributes["bytes"] != event.bytes
                or node.attributes["direction"] != event.action.value
                or node.attributes["source_tier"] != event.source_tier
                or node.attributes["destination_tier"]
                != event.destination_tier
                or node.attributes["tier"] != event.tier):
            raise ValueError(
                "KV transfer workload provenance disagrees with its intent")
        if event.source_request_id is None:
            if "source_request_id" in node.attributes:
                raise ValueError("request-private KV transfer has a shared source")
        elif ("source_request_id" not in node.attributes
              or node.attributes["source_request_id"] != event.source_request_id):
            raise ValueError("shared KV transfer source identity disagrees")
        if (call.call_id != node.node_id
                or call.source_node_ids != (node.node_id,)
                or call.pool_name != expected_pool_name
                or call.rank != node.rank
                or interval.pool_name != expected_pool_name
                or interval.rank != node.rank):
            raise ValueError(
                "KV transfer node/call/interval ownership disagrees")
        expected_physical_direction = (
            "evict" if event.action is KVCacheAction.STORE
            else event.action.value)
        if (physical.bytes != event.bytes
                or physical.direction != expected_physical_direction
                or physical.tier != event.tier
                or physical.link != node.attributes["link"]
                or call.resource != physical.link
                or interval.resource != physical.link):
            raise ValueError(
                "KV transfer physical provenance disagrees with its intent")
        duration_s = interval.end_time_s - interval.start_time_s
        if (not math.isclose(
                duration_s,
                prediction.t_total_s,
                rel_tol=1e-12,
                abs_tol=1e-15)
                or duration_s <= 0):
            raise ValueError(
                "KV transfer interval disagrees with its canonical prediction")
        physical_completions.append(KVCachePhysicalTransferCompletion(
            node_id=node.node_id,
            call_id=call.call_id,
            pool_name=call.pool_name,
            rank=call.rank,
            bytes=physical.bytes,
            link=physical.link,
            start_time_s=interval.start_time_s,
            end_time_s=interval.end_time_s,
        ))
    return KVCacheTransferCompletion(
        ready_event=event.ready_event,
        action=event.action,
        bytes=event.bytes,
        source_tier=event.source_tier,
        destination_tier=event.destination_tier,
        link=physical_completions[0].link,
        physical_transfers=tuple(physical_completions),
    )
