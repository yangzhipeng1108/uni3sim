"""Source-derived live activation storage, independent of startup observations."""

from src.kernel.prediction_cache import runtime_lru_cache, RuntimeCache

from collections.abc import Mapping
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import threading
from src.kernel.memory_access import MemoryAllocation
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.workload.model_config import ModelConfig
from src.serving.workload.traced_forward import ForwardActivationGroup, ForwardCollectiveGroup, ForwardKVCacheGroup, ForwardPipelineGroup, ForwardTrace
from src.serving.workload.workload_graph import WorkloadGraph, WorkloadKind, WorkloadNode


_peak_identity_cache = RuntimeCache()
_capture_identity_cache = RuntimeCache()
_storage_identity_lock = threading.Lock()


def _storage_identity_lookup(cache, identity, allocations, events):
    with _storage_identity_lock:
        if identity not in cache:
            return None
        retained_allocations, retained_events, result = cache[identity]
        if retained_allocations is not allocations or retained_events is not events:
            raise RuntimeError("storage input identity collision")
        cache.move_to_end(identity)
        return result


def _storage_identity_remember(cache, identity, allocations, events, result):
    with _storage_identity_lock:
        cache[identity] = (allocations, events, result)
        if len(cache) > 128:
            cache.popitem(last=False)


def peak_live_activation_bytes(
    allocations: tuple[MemoryAllocation, ...], event_order: tuple[str, ...],
    *, start_event: str | None = None, end_event: str | None = None,
) -> tuple[int, tuple[str, ...]]:
    """Sweep exact lifetimes, reusing only complete immutable storage inputs."""
    native_scope = ((start_event is None or type(start_event) is str)
        and (end_event is None or type(end_event) is str))
    identity = (id(allocations), id(event_order), start_event, end_event) if native_scope else None
    if identity is not None:
        cached = _storage_identity_lookup(_peak_identity_cache, identity, allocations, event_order)
        if cached is not None:
            return cached
    if (type(allocations) is tuple and type(event_order) is tuple
            and all(type(allocation) is MemoryAllocation for allocation in allocations)
            and all(type(event) is str for event in event_order)
            and (start_event is None or type(start_event) is str)
            and (end_event is None or type(end_event) is str)):
        result = _cached_peak_live_activation_bytes(allocations, event_order, start_event, end_event)
        _storage_identity_remember(_peak_identity_cache, identity, allocations, event_order, result)
        return result
    return _peak_live_activation_bytes(allocations, event_order,
        start_event=start_event, end_event=end_event)


def source_capture_activation_peak(trace, capture):
    """Reuse this exact capture's strict allocation window, never global peak."""
    from src.serving.workload.traced_forward import _source_activation_allocations
    allocations, events = _source_activation_allocations(trace), trace.event_order
    begin_event, end_event = capture["begin_event"], capture["end_event"]
    cacheable = (type(allocations) is tuple and type(events) is tuple
        and type(begin_event) is str and type(end_event) is str)
    identity = (id(allocations), id(events), begin_event, end_event) if cacheable else None
    if identity is not None:
        cached = _storage_identity_lookup(_capture_identity_cache, identity, allocations, events)
        if cached is not None:
            return cached
    indices = {event: index for index, event in enumerate(events)}
    begin, end = indices[begin_event], indices[end_event]
    window = tuple(allocation for allocation in allocations if begin < indices[allocation.allocated_event] < end)
    result = peak_live_activation_bytes(window, events)
    # The global call has already validated canonical input in the ordinary
    # observer; the complete window sweep above still owns this result.
    if (cacheable and all(type(value) is MemoryAllocation for value in allocations)
            and all(type(value) is str for value in events)):
        _storage_identity_remember(_capture_identity_cache, identity, allocations, events, result)
    return result


@runtime_lru_cache(maxsize=128)
def _cached_peak_live_activation_bytes(allocations, event_order, start_event, end_event):
    # Frozen canonical allocations contain every geometry and lifetime field;
    # equal IDs alone are insufficient. Eviction simply repeats the same sweep.
    return _peak_live_activation_bytes(allocations, event_order,
        start_event=start_event, end_event=end_event)


def _peak_live_activation_bytes(
    allocations: tuple[MemoryAllocation, ...], event_order: tuple[str, ...],
    *, start_event: str | None = None, end_event: str | None = None,
) -> tuple[int, tuple[str, ...]]:
    """Maximum unique live allocation bytes on an explicit total event order.

    Lifetimes are half-open [allocated_event, freed_event); None lasts to
    the end. Views do not create MemoryAllocations. The caller must encode
    asynchronous overlap in the event order, rather than freeing at enqueue.
    Activation callers exclude weights, persistent state and KV. Profiling
    can also sweep every source object, then subtract its final live state;
    an optional event scope excludes unrelated setup/teardown peaks.
    Scope sampling includes both named boundary events; storage lifetimes
    themselves remain half-open.
    """
    if (not isinstance(event_order, tuple) or not event_order
            or any(type(event) is not str or not event for event in event_order)
            or len(set(event_order)) != len(event_order)):
        raise ValueError("activation liveness requires unique ordered event IDs")
    if not isinstance(allocations, tuple) or any(not isinstance(a, MemoryAllocation) for a in allocations):
        raise TypeError("activation liveness requires canonical MemoryAllocation objects")
    if len({a.allocation_id for a in allocations}) != len(allocations):
        raise ValueError("activation allocations must have unique storage IDs; aliases are views")
    if len({a.device_id for a in allocations}) > 1:
        raise ValueError("activation liveness must be computed per physical rank")
    indices = {event: index for index, event in enumerate(event_order)}
    if (start_event is None) != (end_event is None):
        raise ValueError("a storage measurement scope needs both start and end events")
    if start_event is None:
        scope_start, scope_end = 0, len(event_order)
    else:
        if start_event not in indices or end_event not in indices or indices[start_event] >= indices[end_event]:
            raise ValueError("storage measurement scope must reference ordered known events")
        scope_start, scope_end = indices[start_event], indices[end_event] + 1
    intervals = []
    for allocation in allocations:
        if allocation.allocated_event not in indices or (allocation.freed_event is not None and allocation.freed_event not in indices):
            raise ValueError("activation lifetime references an unknown event")
        start = indices[allocation.allocated_event]
        end = len(event_order) if allocation.freed_event is None else indices[allocation.freed_event]
        if start >= end:
            raise ValueError("activation allocation must be freed after allocation")
        start, end = max(start, scope_start), min(end, scope_end)
        if start < end:
            intervals.append((allocation, start, end))
    # Sweep metadata once rather than rescanning every allocation per event.
    # Cost depends on graph size, never the number of represented tensor values.
    deltas = [0] * (len(event_order) + 1)
    for allocation, start, end in intervals:
        deltas[start] += allocation.size_bytes
        deltas[end] -= allocation.size_bytes
    peak, size, peak_index = 0, 0, 0
    for event_index, delta in enumerate(deltas[:-1]):
        size += delta
        if size > peak:
            peak, peak_index = size, event_index
    peak_ids = tuple(a.allocation_id for a, start, end in intervals if start <= peak_index < end)
    return peak, peak_ids


def source_profile_storage_headroom(trace, *, start_event, end_event):
    """Separate peak source storage from state retained after profiling.

    vLLM0.30 memory_profiling subtracts after-profile allocated memory from
    the allocation peak for transient_peak_headroom. Cached buffers created
    late in the run therefore cannot simply be excluded from every event.
    This accounts only for the source objects represented by this trace.
    """
    parameters = trace.parameter_allocation_ids
    return _profile_storage_measurements(tuple(a for a in trace.allocations if a.allocation_id not in parameters), trace.event_order,
        start_event=start_event, end_event=end_event)


def _profile_storage_measurements(allocations, events, *, start_event, end_event):
    peak, live = peak_live_activation_bytes(allocations, events,
        start_event=start_event, end_event=end_event)
    indices = {event: index for index, event in enumerate(events)}

    def snapshot(event):
        index = indices[event]
        objects = tuple(a for a in allocations if indices[a.allocated_event] <= index
            and (a.freed_event is None or index < indices[a.freed_event]))
        return sum(a.size_bytes for a in objects), tuple(a.allocation_id for a in objects)

    before, before_ids = snapshot(start_event)
    after, after_ids = snapshot(end_event)
    if peak < after:
        raise AssertionError("profile storage peak cannot be below its final live state")
    retained = set(after_ids)
    nonretained = tuple(a for a in allocations if a.allocation_id not in retained)
    temporary_peak, temporary_ids = peak_live_activation_bytes(nonretained, events,
        start_event=start_event, end_event=end_event)
    # A capacity ledger can charge the complete final retained set once,
    # plus the maximum simultaneous storage outside that set. This also
    # covers old arenas/aliases during growth. It is a source budget bound,
    # not the profile metric (peak - after), or an observed point estimate.
    if after + temporary_peak < peak:
        raise AssertionError("retained and nonretained source partition does not cover its peak")
    return {"peak_source_storage_bytes": peak, "peak_live_allocation_ids": live,
        "before_profile_source_bytes": before, "before_profile_allocation_ids": before_ids,
        "after_profile_source_bytes": after, "after_profile_allocation_ids": after_ids,
        "retained_source_growth_bytes": after - before,
        "transient_source_headroom_bytes": peak - after,
        "nonretained_source_peak_bytes": temporary_peak,
        "nonretained_peak_live_allocation_ids": temporary_ids,
        "retained_plus_nonretained_source_budget_bytes": after + temporary_peak,
        "scope": "traced source objects only; runtime/library/allocator state not implied"}


def _validated_dependency_liveness(trace, escaped_allocation_ids, start_event, end_event):
    if not isinstance(trace, ForwardTrace):
        raise TypeError("dependency liveness requires one actual rank-local ForwardTrace")
    if (not isinstance(escaped_allocation_ids, tuple)
            or any(type(name) is not str or not name for name in escaped_allocation_ids)
            or len(set(escaped_allocation_ids)) != len(escaped_allocation_ids)):
        raise ValueError("escaped storage must be unique explicit allocation IDs")
    allocations, views, events, graph = trace.allocations, trace.views, trace.event_order, trace.graph
    by_id = {allocation.allocation_id: allocation for allocation in allocations}
    view_ids = {view.view_id: view.allocation_id for view in views}
    if any(name not in by_id for name in escaped_allocation_ids):
        raise ValueError("escaped storage is outside the source allocation contract")
    # Reuse the canonical interval validator before deriving shorter lives.
    source_peak, source_live = peak_live_activation_bytes(trace.activation_allocations, events,
        start_event=start_event, end_event=end_event)
    indices = {event: index for index, event in enumerate(events)}
    returns = {}
    for index, event in enumerate(events):
        parts = event.split(":", 2)
        if len(parts) == 3 and parts[1] == "return":
            payload = parts[2]
            if payload in returns:
                raise ValueError("dependency liveness requires one return per source operation")
            returns[payload] = index
    if tuple(returns) != tuple(node.node_id for node in graph.nodes):
        raise ValueError("dependency liveness requires the complete synchronous source operation order")
    if any(returns[edge.source] >= returns[edge.destination] for edge in graph.edges):
        raise ValueError("source execution order violates the declared dependency graph")
    last_use, final_writes = {}, set()
    for node in graph.nodes:
        if node.rank != trace.rank or any(field not in node.attributes
                for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs")):
            raise ValueError("dependency liveness requires complete typed rank-local ports")
        ports = [node.attributes["symbolic_tensor_inputs"], node.attributes["symbolic_tensor_outputs"]]
        if "symbolic_tensor_controls" in node.attributes:
            ports.append(node.attributes["symbolic_tensor_controls"])
        if any(not isinstance(port, tuple) or any(name not in view_ids for name in port) for port in ports):
            raise ValueError("dependency liveness encountered undeclared tensor views")
        touched = {view_ids[name] for port in ports for name in port}
        writes = {view_ids[name] for name in ports[1]}
        for name in touched:
            allocation = by_id[name]
            position = returns[node.node_id]
            end = len(events) if allocation.freed_event is None else indices[allocation.freed_event]
            if not indices[allocation.allocated_event] <= position < end:
                raise ValueError("typed operation uses storage outside its source lifetime")
            last_use[name] = position
            final_writes.discard(name)
        final_writes.update(writes)
    activation_ids = {allocation.allocation_id for allocation in trace.activation_allocations}
    protected = (set(by_id) - activation_ids) | set(escaped_allocation_ids) | final_writes
    protected.update(name for name, allocation in by_id.items()
        if allocation.freed_event is None or name not in last_use)
    return (allocations, events, graph, by_id, view_ids, source_peak, source_live, indices, returns, last_use, activation_ids, protected)


def dependency_last_use_storage_report(trace, *, escaped_allocation_ids, start_event=None, end_event=None):
    """Analyze one synchronous typed graph without rewriting its eager trace.

    All aliases share their allocation's last input/output/control use. An
    eligible object retires at the first source event after that operation
    returns, while retaining its original birth and thus producer overlap.
    Persistent objects, explicit escapes, still-live objects, final writes
    without consumers, and objects without typed graph ports retain their
    source intervals. This is a defined analytical retirement policy, not a
    compiler allocation plan or a prediction of arbitrary asynchronous use.
    """
    (allocations, events, graph, by_id, view_ids, source_peak, source_live,
        indices, returns, last_use, activation_ids, protected) = _validated_dependency_liveness(
            trace, escaped_allocation_ids, start_event, end_event)
    projected, retired = [], []
    for allocation in allocations:
        name = allocation.allocation_id
        if name not in protected and last_use[name] + 1 < indices[allocation.freed_event]:
            retired.append(name)
            allocation = replace(allocation, freed_event=events[last_use[name] + 1])
        projected.append(allocation)
    projected = tuple(projected)
    analytical = tuple(allocation for allocation in projected if allocation.allocation_id in activation_ids)
    peak, live = peak_live_activation_bytes(analytical, events, start_event=start_event, end_event=end_event)
    if peak > source_peak:
        raise AssertionError("last-use retirement cannot increase the same source-order peak")
    return {"analytical_activation_peak_bytes": peak, "analytical_peak_live_allocation_ids": live,
        "eager_source_activation_peak_bytes": source_peak, "eager_source_peak_live_allocation_ids": source_live,
        "retired_after_dependency_last_use": tuple(retired), "protected_source_lifetimes": tuple(sorted(protected)),
        "escaped_allocation_ids": escaped_allocation_ids,
        "source_graph_nodes": len(graph.nodes), "physical_rank": trace.rank,
        "retirement_policy": "first_source_event_after_final_typed_operation_return",
        "scope": "synchronous_complete_typed_graph_policy_with_conservative_escape_and_opaque_storage_retention",
        "actual_compiler_plan_bound": False, "complete_worker_memory_accepted": False,
        "limitations": ("undeclared_external_consumers_are_outside_this_closed_typed_graph_scenario",
            "no_cross_stream_retirement_or_library_workspace_inferred", "source_intervals_and_graph_are_unchanged")}


def segment_reference_storage_report(trace, *, split_node_ids, escaped_allocation_ids,
        start_event, end_event, segment_start_event, segment_end_event):
    """Source partition policy; caller retains segment inputs until return.

    Split nodes and model scope are explicit source contracts, never inferred
    from shape or an observation. Original graph/intervals remain unchanged.
    """
    (allocations, events, graph, by_id, view_ids, source_peak, source_live,
        indices, returns, last_use, activation_ids, protected) = _validated_dependency_liveness(
            trace, escaped_allocation_ids, start_event, end_event)
    if (not isinstance(split_node_ids, tuple) or not split_node_ids
            or any(not isinstance(name, str) or not name for name in split_node_ids)
            or len(set(split_node_ids)) != len(split_node_ids)):
        raise ValueError("segment policy requires unique explicit split-node IDs")
    if segment_start_event not in indices or segment_end_event not in indices:
        raise ValueError("segment scope contains unknown source events")
    begin, end = indices[segment_start_event], indices[segment_end_event]
    if begin < indices[start_event] or end > indices[end_event]:
        raise ValueError("segment scope must be inside the measured source profile")
    if begin >= end:
        raise ValueError("segment policy requires an ordered explicit source scope")
    nodes = tuple(node for node in graph.nodes if begin <= returns[node.node_id] < end)
    node_ids = tuple(node.node_id for node in nodes)
    if not set(split_node_ids) <= set(node_ids) or tuple(
            name for name in node_ids if name in split_node_ids) != split_node_ids:
        raise ValueError("split nodes must follow the declared model scope order")
    groups, current = [], []
    for node in nodes:
        if node.node_id in split_node_ids:
            if current:
                groups.append(tuple(current))
                current = []
            groups.append((node,))
        else:
            current.append(node)
    if current:
        groups.append(tuple(current))
    # Preserve all storage participating in operations outside the selected
    # compiled model scope; this never reinterprets the eager worker/head.
    for node in graph.nodes:
        if node.node_id not in node_ids:
            for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_tensor_controls"):
                if field in node.attributes:
                    protected.update(view_ids[view] for view in node.attributes[field])
    produced, incoming_groups = {}, []
    for group_index, group in enumerate(groups):
        incoming = set()
        for node in group:
            inputs = node.attributes["symbolic_tensor_inputs"]
            outputs = node.attributes["symbolic_tensor_outputs"]
            controls = () if "symbolic_tensor_controls" not in node.attributes else node.attributes["symbolic_tensor_controls"]
            for view in (*inputs, *controls):
                storage = view_ids[view]
                if storage not in produced or produced[storage] < group_index:
                    incoming.add(storage)
            for view in outputs:
                if view_ids[view] not in produced:
                    produced[view_ids[view]] = group_index
        group_return = returns[group[-1].node_id]
        for storage in incoming:
            last_use[storage] = max(last_use[storage], group_return)
        incoming_groups.append(tuple(sorted(incoming)))
    projected = tuple(replace(allocation, freed_event=events[last_use[allocation.allocation_id] + 1])
        if allocation.allocation_id not in protected else allocation for allocation in allocations)
    analytical = tuple(a for a in projected if a.allocation_id in activation_ids)
    peak, live = peak_live_activation_bytes(analytical, events)
    parameters = trace.parameter_allocation_ids
    measurements = _profile_storage_measurements(tuple(a for a in projected if a.allocation_id not in parameters), events, start_event=start_event, end_event=end_event)
    return {"peak_bytes": peak, "peak_live_allocation_ids": live,
        "peak_live_allocations": tuple({"allocation_id": name, "bytes": by_id[name].size_bytes} for name in live),
        "allocation_count": len(projected), "source_workspace_capacity_bytes": trace.workspace_capacity_bytes,
        "persistent_source_objects": tuple({"allocation_id": a.allocation_id, "bytes": a.size_bytes}
            for a in allocations if a.allocation_id not in activation_ids and a.allocation_id not in parameters),
        "profile_storage_measurements": measurements,
        "segments": tuple({"nodes": tuple(n.node_id for n in group), "incoming_storage_ids": incoming}
            for group, incoming in zip(groups, incoming_groups)),
        "escaped_allocation_ids": escaped_allocation_ids, "protected_source_lifetimes": tuple(sorted(protected)),
        "retirement_policy": "typed_last_use_with_segment_caller_input_retention",
        "actual_compiler_plan_bound": False, "complete_worker_memory_accepted": False,
        "normal_capacity_accepted": False}


def _native_observer_report(source, graph, source_profile, traces, compute):
    if source_profile != "cuda_eager_inplace" or type(source) is not ForwardCollectiveGroup or len(traces) != 2:
        return None
    from copy import deepcopy
    from src.serving.kernel_lowering.memory_bindings import _memory_fields
    from src.serving.workload.traced_forward import (
        ForwardCollectiveSourceProgram, _NativeSourceFrame, _NativeStorageOverlay)

    frame = graph.__dict__["_native_source_frame"] if "_native_source_frame" in graph.__dict__ else None
    if type(frame) is not _NativeSourceFrame:
        return None
    owner = frame.owner()
    if type(owner) is not ForwardCollectiveSourceProgram or source._recorded_graph is not graph:
        return None
    if any(not rank.geometry_rebind for rank in owner.rank_programs):
        return None
    snapshots = tuple(rank.__dict__["_native_storage_template"] if "_native_storage_template" in rank.__dict__ else None
        for rank in owner.rank_programs)
    if any(snapshot is None for snapshot in snapshots):
        return None
    rank_keys = []
    for trace, snapshot in zip(traces, snapshots, strict=True):
        owned = trace.__dict__["_owned_memory_storage"] if "_owned_memory_storage" in trace.__dict__ else None
        if (type(owned) is not tuple or len(owned) != 6
                or type(owned[2]) is not _NativeStorageOverlay or type(owned[3]) is not _NativeStorageOverlay
                or trace._allocations is not owned[0] or trace._views is not owned[1]
                or trace._persistent_ids is not owned[4] or trace._events is not owned[5]
                or trace._persistent_ids is not snapshot[2] or trace._events is not snapshot[6]):
            return None
        captures = []
        for capture in trace.source_captures:
            inputs = tuple((name, port["shape"], port["dtype"], port["bytes"], port["storage_id"])
                for name, port in capture["inputs"].items())
            output = capture["output"]
            captures.append((capture["logical_tokens"], capture["model_input_tokens"],
                capture["key"], capture["runtime_mode"], capture["source_stage"],
                capture["begin_event"], capture["end_event"], inputs,
                (output["shape"], output["dtype"], output["bytes"], output["storage_id"])))
        rank_keys.append((trace.rank, len(trace._views),
            tuple(sorted(trace.workspace_capacity_bytes.items())), tuple(captures)))
    key = id(frame.template), source_profile, tuple(rank_keys)
    if "_native_observer_reports" not in owner.__dict__:
        owner.__dict__["_native_observer_reports"] = OrderedDict()
    cache = owner.__dict__["_native_observer_reports"]
    if key in cache:
        cached_snapshots, expected, report = cache[key]
        if all(current is previous for current, previous in zip(snapshots, cached_snapshots, strict=True)):
            current = tuple(tuple((type(allocation), _memory_fields(allocation))
                for allocation in trace.activation_allocations) for trace in traces)
            if current == expected:
                cache.move_to_end(key)
                return deepcopy(report)
    report = compute()
    expected = tuple(tuple((type(allocation), _memory_fields(allocation))
        for allocation in trace.activation_allocations) for trace in traces)
    cache[key] = snapshots, expected, deepcopy(report)
    cache.move_to_end(key)
    if len(cache) > 64:
        cache.popitem(last=False)
    return report


def symbolic_forward_storage_report(source, graph, *, source_profile):
    if isinstance(source, ForwardTrace):
        traces = (source,)
    elif isinstance(source, (ForwardActivationGroup, ForwardCollectiveGroup, ForwardKVCacheGroup, ForwardPipelineGroup)):
        traces = source.traces
    else:
        raise TypeError("source memory requires the actual symbolic forward trace")
    if not isinstance(graph, WorkloadGraph) or not isinstance(source_profile, str) or not source_profile:
        raise TypeError("source memory requires its workload graph and declared source profile")
    cached = _native_observer_report(source, graph, source_profile, traces,
        lambda: _compute_symbolic_forward_storage_report(source, graph, source_profile=source_profile))
    if cached is not None:
        return cached
    return _compute_symbolic_forward_storage_report(source, graph, source_profile=source_profile)


def kimi_k3_source_cache_storage_report(source, *, native_workers, physical_ranks):
    """Reconcile logical K3 source pools with validated native page metadata.

    Logical per-layer tensors retain dense coordinates for dependency/traffic
    analysis. They are projections into the native arena, not extra allocations
    to add to that arena. This check does not bind GPU tensors or change their
    source storage identities, allocator policy or acceptance status.
    """
    from src.kernel.kernel_spec import DTYPE_BYTES

    if isinstance(source, ForwardTrace):
        traces = (source,)
    elif isinstance(source, (ForwardActivationGroup, ForwardCollectiveGroup, ForwardKVCacheGroup, ForwardPipelineGroup)):
        traces = source.traces
    else:
        raise TypeError("K3 cache reconciliation requires the actual symbolic source")
    if (not isinstance(native_workers, tuple) or not isinstance(physical_ranks, tuple)
            or len(traces) != len(native_workers) or len(traces) != len(physical_ranks)
            or len(set(physical_ranks)) != len(physical_ranks)
            or tuple(trace.rank for trace in traces) != physical_ranks):
        raise ValueError("K3 cache reconciliation requires matching physical rank order")
    dtype_names = {"bfloat16": "bf16", "float32": "fp32"}
    ranks = {}
    for trace, native in zip(traces, native_workers, strict=True):
        rows, logical_bytes = [], 0
        blocks, arena = native["num_blocks"], native["shared_arena_bytes"]
        if type(blocks) is not int or blocks <= 0 or type(arena) is not int or arena <= 0:
            raise ValueError("K3 cache reconciliation requires positive native capacities")
        owners = set()
        for layer in native["layers"]:
            owner = layer["owner"]
            parts = owner.split(".")
            if (len(parts) != 5 or parts[:3] != ["language_model", "model", "layers"]
                    or parts[4] != "self_attn" or not parts[3].isdigit() or owner in owners):
                raise ValueError("K3 cache reconciliation requires unique source attention owners")
            owners.add(owner)
            prefix = f"layer-{int(parts[3])}:attention"
            if layer["spec_type"] == "MambaSpec":
                if len(layer["states"]) != 2:
                    raise ValueError("K3 cache reconciliation requires conv and recurrent native states")
                first_state, second_state = layer["states"]
                if (first_state["offset_within_page_bytes"] != 0
                        or second_state["offset_within_page_bytes"] != first_state["state_content_bytes"]
                        or sum(state["state_content_bytes"] for state in layer["states"]) != layer["state_content_size_bytes"]
                        or layer["state_content_size_bytes"] > layer["page_size_bytes"]):
                    raise ValueError("K3 cache reconciliation has inconsistent packed native states")
                projections = []
                for suffix, state in zip(("conv-state", "recurrent-state"), layer["states"], strict=True):
                    shape, dtype = tuple(state["shape"]), dtype_names[state["dtype"]]
                    element_bytes = DTYPE_BYTES[dtype]
                    strides, stride = [], element_bytes
                    for dimension in reversed(shape):
                        strides.append(stride)
                        stride *= dimension
                    strides = tuple(reversed(strides))
                    transform = "identity"
                    if suffix == "conv-state":
                        if len(shape) != 2 or layer["conv_state_layout"] not in ("SD", "DS"):
                            raise ValueError("K3 cache reconciliation requires source-defined conv state orientation")
                        if layer["conv_state_layout"] == "SD":
                            # K3 forward transposes SD storage to the DS
                            # kernel view before all prefill/decode branches.
                            shape, strides = tuple(reversed(shape)), tuple(reversed(strides))
                            transform = "transpose_last_two_view_no_copy"
                    projections.append((suffix, (blocks, *shape), dtype,
                        blocks * state["state_content_bytes"], state["offset_within_page_bytes"], strides, transform))
            elif layer["spec_type"] == "MLAAttentionSpec":
                if layer["num_kv_heads"] != 1:
                    raise ValueError("K3 cache reconciliation requires one latent KV head")
                projections = (("cache", (blocks * layer["block_tokens"], layer["head_size"]),
                    dtype_names[layer["dtype"]], blocks * layer["page_size_bytes"], 0,
                    (layer["head_size"] * DTYPE_BYTES[dtype_names[layer["dtype"]]], DTYPE_BYTES[dtype_names[layer["dtype"]]]),
                    "slot_to_block_and_token_in_page"),)
            else:
                raise NotImplementedError("K3 cache reconciliation requires native Mamba/MLA leaves")
            if (type(layer["block_stride_bytes"]) is not int or layer["block_stride_bytes"] <= 0
                    or layer["last_page_end_bytes"] != layer["block0_offset_bytes"]
                        + (blocks - 1) * layer["block_stride_bytes"] + layer["page_size_bytes"]
                    or not 0 <= layer["block0_offset_bytes"] < layer["last_page_end_bytes"] <= arena):
                raise ValueError("K3 cache reconciliation native page exceeds arena")
            for suffix, shape, dtype, size, state_offset, strides, transform in projections:
                key = trace.qualify(prefix + ":" + suffix)
                view, allocation = trace._views[key], trace._allocations[key]
                if (key not in trace._persistent_ids or view.allocation_id != key
                        or view.shape != shape or view.dtype != dtype or allocation.size_bytes != size):
                    raise ValueError(f"K3 source logical cache differs from native state geometry: {key}; "
                        f"source={(view.shape, view.dtype, allocation.size_bytes)}, native={(shape, dtype, size)}")
                logical_bytes += size
                rows.append({"allocation_id": key, "owner": owner, "logical_projection_bytes": size,
                    "group_index": layer["group_index"], "block0_offset_bytes": layer["block0_offset_bytes"] + state_offset,
                    "block_stride_bytes": layer["block_stride_bytes"],
                    "native_intra_page_strides_bytes": strides, "logical_to_native_transform": transform})
        source_pools = {key for key in trace._persistent_ids
            if key.endswith((":attention:conv-state", ":attention:recurrent-state", ":attention:cache"))}
        if not rows or {row["allocation_id"] for row in rows} != source_pools:
            raise ValueError("K3 cache reconciliation requires all stage-local source pools")
        ranks[str(trace.rank)] = {"native_shared_arena_bytes": arena,
            "logical_source_projection_bytes": logical_bytes, "additional_cache_charge_bytes": 0,
            "projections": tuple(rows)}
    return {"ranks": ranks, "normal_capacity_accepted": False, "gpu_cache_binding_executed": False,
        "scope": "source_logical_shapes_match_native_pages; physical_capacity_charges_native_arena_once"}


def _compute_symbolic_forward_storage_report(source, graph, *, source_profile):
    """Summarize the same virtual forward, without replaying model code.

    Canonical allocation lifetimes remain the authority. Only peak witnesses
    are retained; tensor handles/events for the full forward can be released.
    Unrepresented workspace/runtime state prevents complete capacity claims.
    """
    if isinstance(source, ForwardTrace):
        traces = (source,)
    elif isinstance(source, (ForwardActivationGroup, ForwardCollectiveGroup, ForwardKVCacheGroup, ForwardPipelineGroup)):
        traces = source.traces
    else:
        raise TypeError("source memory requires the actual symbolic forward trace")
    if not isinstance(graph, WorkloadGraph) or not isinstance(source_profile, str) or not source_profile:
        raise TypeError("source memory requires its workload graph and declared source profile")
    from src.serving.workload.traced_forward import _source_bound_storage_projection
    projection = _source_bound_storage_projection(source, graph)
    if projection is None:
        ranks = {}
        for trace in traces:
            allocations = trace.activation_allocations
            peak, live = peak_live_activation_bytes(allocations, trace.event_order)
            live_ids = set(live)
            objects = tuple({"allocation_id": a.allocation_id, "size_bytes": a.size_bytes,
                "allocated_event": a.allocated_event, "freed_event": a.freed_event}
                for a in allocations if a.allocation_id in live_ids)
            if sum(a["size_bytes"] for a in objects) != peak:
                raise AssertionError("source activation peak disagrees with its unique live storage")
            ranks[str(trace.rank)] = {"known_activation_peak_bytes": peak, "peak_live_storage": objects,
                "activation_allocation_count": len(allocations), "view_count": len(trace._views),
                "source_workspace_capacity_bytes": trace.workspace_capacity_bytes}
        graph_nodes, graph_edges = len(graph.nodes), len(graph.edges)
    else:
        if type(projection) is not tuple or len(projection) != 3:
            raise TypeError("compiled source storage projection requires graph counts and rank rows")
        graph_nodes, graph_edges, rows = projection
        if (type(graph_nodes) is not int or graph_nodes <= 0
                or type(graph_edges) is not int or graph_edges < 0 or type(rows) is not tuple):
            raise TypeError("compiled source storage projection has invalid counts or rank rows")
        traces_by_rank = {trace.rank: trace for trace in traces}
        ranks = {}
        for row in rows:
            if type(row) is not tuple or len(row) != 4:
                raise TypeError("compiled source storage rank rows require all existing storage fields")
            rank, allocations, view_count, workspaces = row
            if (type(rank) is not int or rank not in traces_by_rank or str(rank) in ranks
                    or type(allocations) is not tuple or type(view_count) is not int or view_count < 0
                    or not isinstance(workspaces, Mapping)):
                raise TypeError("compiled source storage rank row disagrees with its current source")
            peak, live = peak_live_activation_bytes(allocations, traces_by_rank[rank].event_order)
            live_ids = set(live)
            objects = tuple({"allocation_id": allocation.allocation_id, "size_bytes": allocation.size_bytes,
                "allocated_event": allocation.allocated_event, "freed_event": allocation.freed_event}
                for allocation in allocations if allocation.allocation_id in live_ids)
            if sum(allocation["size_bytes"] for allocation in objects) != peak:
                raise AssertionError("source activation peak disagrees with its unique live storage")
            ranks[str(rank)] = {"known_activation_peak_bytes": peak, "peak_live_storage": objects,
                "activation_allocation_count": len(allocations), "view_count": view_count,
                "source_workspace_capacity_bytes": dict(workspaces)}
        if set(ranks) != {str(rank) for rank in traces_by_rank}:
            raise ValueError("compiled source storage projection must cover every actual rank")
    limitations = ("unrepresented_workspace_compiler_and_runtime_lifetimes",
        "logical_cache_is_not_native_physical_KV_capacity", "persistent_state_and_weights_are_separate")
    if isinstance(source, ForwardKVCacheGroup):
        limitations += ("host_kv_pages_and_retirement_owned_by_tier_manager_not_activation_storage_or_DMA_workspace",)
    return {"source_profile": source_profile, "graph_nodes": graph_nodes, "graph_edges": graph_edges,
        "ranks": ranks, "complete": False, "normal_capacity_accepted": False,
        "scope": "same_forward_known_source_storage_only_not_complete_worker_peak_or_bound",
        "limitations": limitations}


def source_forward_state_summary(source, graph):
    """Bind every source allocation's identity, location, capacity and lifetime."""
    if isinstance(source, ForwardTrace):
        traces = (source,)
    elif isinstance(source, (ForwardActivationGroup, ForwardCollectiveGroup, ForwardKVCacheGroup, ForwardPipelineGroup)):
        traces = source.traces
    else:
        raise TypeError("source state audit requires the actual symbolic forward trace")
    if not isinstance(graph, WorkloadGraph):
        raise TypeError("source state audit requires its actual workload graph")
    ranks = {}
    for trace in traces:
        records = []
        domains = {}
        freed_count = freed_bytes = live_count = live_bytes = 0
        for allocation in trace.allocations:
            record = {
                "allocation_id": allocation.allocation_id,
                "device_id": allocation.device_id,
                "memory_domain": allocation.memory_domain,
                "size_bytes": allocation.size_bytes,
                "address_level": allocation.address_level.value,
                "base_address_bytes": allocation.base_address_bytes,
                "mapping_ref": allocation.mapping_ref,
                "allocated_event": allocation.allocated_event,
                "freed_event": allocation.freed_event,
            }
            records.append(record)
            domain_key = (allocation.memory_domain
                if allocation.memory_domain is not None else "not_declared")
            domain = domains[domain_key] if domain_key in domains else {
                "allocation_count": 0, "allocation_bytes": 0}
            domain["allocation_count"] += 1
            domain["allocation_bytes"] += allocation.size_bytes
            domains[domain_key] = domain
            if allocation.freed_event is None:
                live_count += 1
                live_bytes += allocation.size_bytes
            else:
                freed_count += 1
                freed_bytes += allocation.size_bytes
        records.sort(key=lambda record: record["allocation_id"])
        encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ranks[str(trace.rank)] = {
            "allocation_count": len(records),
            "allocation_bytes": sum(record["size_bytes"] for record in records),
            "freed_allocation_count": freed_count,
            "freed_allocation_bytes": freed_bytes,
            "live_allocation_count": live_count,
            "live_allocation_bytes": live_bytes,
            "memory_domains": domains,
            "allocation_state_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return {"ranks": ranks,
        "scope": "source_forward_allocation_lifecycle_not_complete_runtime_state"}


class SourceActivationMemoryAccumulator:
    """Retain scalar per-step peaks and one witness per pool/physical rank.

    Full trace objects are never retained. Missing source reports are not
    observed zeros. Maxima of independent rank/forward snapshots do not prove
    a global asynchronous worker peak or a complete runtime capacity budget.
    """

    def __init__(self, *, rank_pools=None):
        if rank_pools is not None and (not isinstance(rank_pools, Mapping) or not rank_pools
                or any(not isinstance(rank, str) or not rank.isascii() or not rank.isdecimal()
                    or not isinstance(pool, str) or not pool for rank, pool in rank_pools.items())):
            raise ValueError("source memory placement needs explicit physical rank to pool ownership")
        self._rank_pools = None if rank_pools is None else dict(rank_pools)
        self._steps = set()
        self._series = []
        self._pools = {}

    def consume(self, pool_name, step_id, report):
        if not isinstance(pool_name, str) or not pool_name or not isinstance(step_id, str) or not step_id:
            raise ValueError("source memory needs explicit pool and step identities")
        if step_id in self._steps:
            raise ValueError("duplicate source memory step")
        required = {
                "source_profile", "graph_nodes", "graph_edges", "ranks", "complete",
                "normal_capacity_accepted", "scope", "limitations"}
        if (not isinstance(report, Mapping) or not required <= set(report)
                or set(report) - required - {"host_capture_timing"}):
            raise ValueError("source memory requires its complete declared report contract")
        if "host_capture_timing" in report:
            timing = report["host_capture_timing"]
            if (not isinstance(timing, Mapping) or set(timing) != {
                    "source_runtime_mode", "step_host_coverage_mode", "scope"}
                    or timing["source_runtime_mode"] not in ("FULL", "PIECEWISE")
                    or timing["step_host_coverage_mode"] not in ("FULL", "PIECEWISE")
                    or not isinstance(timing["scope"], str) or not timing["scope"]):
                raise ValueError("source memory Host capture annotation must declare valid source and step coverage")
        if (not isinstance(report["source_profile"], str) or not report["source_profile"]
                or not isinstance(report["ranks"], Mapping) or not report["ranks"]):
            raise ValueError("source memory requires a source identity and physical ranks")
        if any(type(report[name]) is not int or report[name] < minimum
               for name, minimum in (("graph_nodes", 1), ("graph_edges", 0))):
            raise ValueError("source memory graph counts are invalid")
        if report["complete"] is not False or report["normal_capacity_accepted"] is not False:
            raise ValueError("source memory snapshots cannot assert complete runtime capacity")
        for rank, record in report["ranks"].items():
            if not isinstance(rank, str) or not rank.isascii() or not rank.isdecimal():
                raise ValueError("source memory requires explicit nonnegative physical rank identities")
            peak = record["known_activation_peak_bytes"]
            objects = record["peak_live_storage"]
            if (type(peak) is not int or peak < 0
                    or len({a["allocation_id"] for a in objects}) != len(objects)
                    or sum(a["size_bytes"] for a in objects) != peak):
                raise ValueError("source memory peak disagrees with its unique live allocation witness")
        if self._rank_pools is not None and (pool_name not in self._rank_pools.values()
                or set(report["ranks"]) - set(self._rank_pools)):
            raise ValueError("source memory rank coverage is outside its configured pool placement")
        partitions, rank_pools, peaks = {}, {}, {}
        for rank, record in report["ranks"].items():
            owner = pool_name if self._rank_pools is None else self._rank_pools[rank]
            rank_pools[rank] = owner
            if owner not in partitions:
                partitions[owner] = {}
            partitions[owner][rank] = record
            peaks[rank] = record["known_activation_peak_bytes"]
        for owner, records in partitions.items():
            if self._rank_pools is not None and set(records) != {rank for rank, pool in self._rank_pools.items() if pool == owner}:
                raise ValueError("source memory report must cover every physical rank of its pool")
            if owner in self._pools:
                if set(self._pools[owner]["ranks"]) != set(records):
                    raise ValueError("source memory pool changed its physical rank coverage")
                if self._pools[owner]["source_profile"] != report["source_profile"]:
                    raise ValueError("one source memory pool cannot change forward source profiles")
        for owner, records in partitions.items():
            if owner not in self._pools:
                self._pools[owner] = {"source_profile": report["source_profile"], "ranks": {}}
            pool = self._pools[owner]
            for rank, record in records.items():
                peak = record["known_activation_peak_bytes"]
                if rank not in pool["ranks"] or peak > pool["ranks"][rank]["known_activation_peak_bytes"]:
                    pool["ranks"][rank] = {**record, "step_id": step_id,
                        "peak_live_storage": tuple(dict(a) for a in record["peak_live_storage"])}
        if not peaks:
            raise ValueError("source memory needs at least one physical rank")
        self._series.append({"step_id": step_id, "pool_name": pool_name,
            "known_activation_peak_bytes_by_rank": peaks,
            **({"rank_pool_names": rank_pools} if len(partitions) > 1 else {})})
        self._steps.add(step_id)

    def manifest(self, *, total_steps):
        if type(total_steps) is not int or total_steps < len(self._series):
            raise ValueError("source memory coverage exceeds total simulation steps")
        if not self._series:
            return None
        return {"reported_steps": len(self._series), "total_simulation_steps": total_steps,
            "all_steps_reported": len(self._series) == total_steps, "pools": self._pools,
            "series": tuple(self._series), "complete": False, "normal_capacity_accepted": False,
            "scope": "per_forward_known_storage_maxima_not_complete_worker_peak_or_bound",
            "retention": "scalar_step_peaks_and_one_live_storage_witness_per_pool_rank"}


def dense_decoder_mlp_forward_trace(
    model: ModelConfig, parallelism: ParallelConfig, *, tokens: int,
) -> ForwardTrace:
    """Dense CUDA eager SwiGLU MLP peak including retained caller tensors.

    vLLM Qwen3 imports Qwen2MLP. Qwen2Model.forward holds its incoming
    hidden_states across DecoderLayer.forward; the layer holds the MLP input
    and residual. Qwen2MLP.forward retains gate_up until return, replaces x
    after activation, then allocates down_proj output before replacing x.
    CUDA SiluAndMul.forward_cuda allocates one output; its views do not copy.
    Thus the down-projection return boundary retains four hidden tensors and
    gate/up + activation (three local intermediate widths). LlamaMLP instead
    overwrites x after activation, releasing gate/up before down_proj; its
    activation return boundary and down return are checked separately.
    Layers execute serially and reuse
    activation storage; number of layers must not multiply the live peak.

    This is the MLP caller tensor peak, not a claim of complete worker peak:
    attention, sampler, compiler lifetime changes, library workspace and
    persistent/driver/allocator state require their own source programs.
    No observation, measured reserve or error-selected coefficient is read.
    """
    if not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig):
        raise TypeError("activation projection requires ModelConfig and ParallelConfig")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("activation token capacity must be a positive integer")
    if (model.model_type not in ("qwen3", "llama") or model.compute_dtype not in ("bf16", "fp16")
            or model.weight_dtype != model.compute_dtype or model.num_hidden_layers < 2):
        raise NotImplementedError("dense activation source requires an unquantized multi-layer Qwen3/Llama")
    if (not parallelism.is_rank_local or parallelism.pipeline_parallel_size != 1
            or parallelism.enable_expert_parallel or parallelism.tensor_parallel_size != 1):
        raise NotImplementedError("initial dense activation source requires TP1/PP1/EP-off")
    trace = ForwardTrace()
    h, i = model.hidden_size, model.intermediate_size
    dtype = model.compute_dtype

    def linear(name, x, n, k):
        return trace.call(WorkloadNode(name, WorkloadKind.LINEAR, "default",
            {"b": 1, "m": tokens, "n": n, "k": k}, dtype=dtype,
            variant="gemm", rank=0),
            (x,), shape=(tokens, n), dtype=dtype)

    def activation(x):
        return trace.call(WorkloadNode("mlp.activation", WorkloadKind.ACTIVATION,
            "default", {"elements": tokens * i, "tok": tokens, "intermediate": i},
            dtype=dtype, variant="silu", rank=0), (x,), shape=(tokens, i), dtype=dtype)

    def mlp(x):
        if model.model_type == "qwen3":
            gate_up = linear("mlp.gate_up", x, 2 * i, h)
            x = activation(gate_up)
        else:
            x = linear("mlp.gate_up", x, 2 * i, h)
            x = activation(x)
        x = linear("mlp.down_output", x, h, i)
        return x

    def layer():
        # Attention is outside this component. These are its distinct source
        # caller tensors, retained by the layer/model frames across MLP calls.
        caller_hidden = trace.tensor("model.caller_hidden", (tokens, h), dtype)
        residual = trace.tensor("layer.residual", (tokens, h), dtype)
        mlp_input = trace.tensor("layer.mlp_input", (tokens, h), dtype)
        return mlp(mlp_input)

    output = layer()
    # No graph record retains output (or any other symbolic tensor). Frame
    # return automatically records its free, just like all intermediate locals.
    return trace


def dense_decoder_mlp_activation_peak(
    model: ModelConfig, parallelism: ParallelConfig, *, tokens: int,
) -> dict[str, object]:
    """Evaluate the automatically traced source MLP's unique live storage."""
    trace = dense_decoder_mlp_forward_trace(model, parallelism, tokens=tokens)
    peak, live_ids = peak_live_activation_bytes(trace.allocations, trace.event_order)
    return {"peak_bytes": peak, "peak_live_allocation_ids": live_ids,
        "source_program": "cuda_eager_dense_swiglu_caller_lifetimes",
        "lifetime_derivation": "symbolic_forward_cpython_references",
        "graph_node_ids": trace.graph.topological_order(),
        "graph_edge_count": len(trace.graph.edges),
        "scope": "MLP caller tensor peak; not complete worker activation or normal memory capacity",
        "uncovered": ("attention", "sampler", "library_workspace", "compiler_lifetimes", "runtime_persistent_state")}


def symbolic_model_activation_peak(model: ModelConfig, parallelism: ParallelConfig, *, tokens: int):
    """Use the same complete symbolic model graph for its caller-storage peak.

    The dense program's storage is shape-dependent; IDs are explicit synthetic
    metadata, not model-generated tokens. This runs the ordinary greedy model
    call, not the worker's separate random-sampler warmup/profile program.
    """
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
    from src.serving.workload.traced_models import trace_qwen3_forward
    if not isinstance(parallelism, ParallelConfig) or (not parallelism.is_rank_local
            or parallelism.tensor_parallel_size != 1 or parallelism.pipeline_parallel_size != 1
            or parallelism.enable_expert_parallel):
        raise NotImplementedError("initial symbolic activation source requires TP1/PP1/EP-off")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("symbolic activation profile needs a positive token capacity")
    step = ScheduleStep("activation-profile", "main", (ScheduledRequest("profile", "prefill", tokens, 0),))
    trace = trace_qwen3_forward(model, step, input_ids=(0,) * tokens,
        positions=tuple(range(tokens)), slot_mapping=tuple(range(tokens)),
        cache_capacity_tokens=tokens, source_profile="cuda_eager_inplace")
    peak, live = peak_live_activation_bytes(trace.activation_allocations, trace.event_order)
    return {"peak_bytes": peak, "peak_live_allocation_ids": live,
        "source_program": "symbolic_complete_dense_greedy_forward",
        "lifetime_derivation": "symbolic_forward_cpython_references",
        "graph_nodes": len(trace.graph.nodes),
        "scope": "complete ordinary greedy caller-storage program; not whole startup worker profile",
        "uncovered": ("startup_random_sampler_profile", "library_workspace", "compiler_lifetimes", "runtime_persistent_state"),
        "complete_worker_memory_accepted": False}


def _startup_storage_record(trace):
    """One rank, one source scope; no model-specific peak formula."""
    activations = trace.activation_allocations
    peak, live = peak_live_activation_bytes(activations, trace.event_order)
    activation_ids = {a.allocation_id for a in activations}
    parameters = trace.parameter_allocation_ids
    persistent = tuple(a for a in trace.allocations if a.allocation_id not in activation_ids and a.allocation_id not in parameters)
    measurements = source_profile_storage_headroom(trace,
        start_event=trace.checkpoints["worker.profile.begin"], end_event=trace.checkpoints["worker.profile.end"])
    return {"peak_bytes": peak, "peak_live_allocation_ids": live,
        "peak_live_allocations": tuple({"allocation_id": a.allocation_id, "bytes": a.size_bytes}
            for a in activations if a.allocation_id in live),
        "allocation_count": len(trace.allocations),
        "source_workspace_capacity_bytes": trace.workspace_capacity_bytes,
        "persistent_source_objects": tuple({"allocation_id": a.allocation_id, "bytes": a.size_bytes}
            for a in persistent),
        "profile_storage_measurements": measurements}


def symbolic_startup_storage_report(source, *, source_profile):
    """Use the same source storage sweep for single-rank or TP startup graphs.

    Report each physical rank separately. Top-level storage fields witness
    the rank with the largest represented activation peak, never a sum of
    independent rank maxima or an implied complete-worker capacity budget.
    """
    if isinstance(source, ForwardTrace):
        traces = (source,)
    elif isinstance(source, ForwardCollectiveGroup):
        traces = source.traces
    else:
        raise TypeError("startup storage requires the actual virtual source trace")
    if not isinstance(source_profile, str) or not source_profile:
        raise ValueError("startup storage requires its explicit source profile")
    records = {str(trace.rank): _startup_storage_record(trace) for trace in traces}
    witness = max(records, key=lambda rank: records[rank]["peak_bytes"])
    return {**records[witness], "ranks": records, "peak_activation_rank": int(witness),
        "source_program": source_profile, "graph_nodes": len(source.graph.nodes),
        "lifetime_derivation": "symbolic_forward_cpython_references",
        "execution": "pure_python_tensor_metadata_only",
        "scope": "conditional eager startup caller-storage program; not complete worker headroom or normal capacity",
        "uncovered": ("library_workspace", "compiler_lifetimes", "runtime_persistent_state", "allocator_reservation"),
        "graph_has_complete_timing_contract": False, "actual_provider_bound": False,
        "complete_worker_memory_accepted": False}


def symbolic_startup_activation_peak(model: ModelConfig, parallelism: ParallelConfig, *,
                                     tokens: int, max_requests: int, source_profile: str, num_compute_units=None,
                                     collective_configs=(), model_execution_version="vllm_0_30_0",
                                     source_lifetime_policy=None, source_compilation_options=None):
    """Trace the startup model/sampler scopes together, not a sum of peaks.

    The Qwen3/Llama source modules share the worker/sampler lifecycle sweep.
    No startup observation is an input. The explicit conditional CUDA source
    profile must be chosen from execution facts rather than numerical error.
    Unknown workspace and runtime state remain uncovered, never assumed zero.
    """
    from src.serving.workload.traced_models import trace_qwen3_startup_profile, ascend018_segment_source_scope
    if not isinstance(model, ModelConfig):
        raise TypeError("symbolic startup activation requires ModelConfig")
    if (not isinstance(parallelism, ParallelConfig) or not parallelism.is_rank_local
            or parallelism.pipeline_parallel_size != 1
            or parallelism.enable_expert_parallel):
        raise NotImplementedError("symbolic dense startup source requires ordinary TP, PP1 EP-off")
    trace = trace_qwen3_startup_profile(model, tokens=tokens, max_requests=max_requests,
        source_profile=source_profile, num_compute_units=num_compute_units,
        tensor_parallel_size=parallelism.tensor_parallel_size, collective_configs=collective_configs,
        model_execution_version=model_execution_version, record_model_scope=source_lifetime_policy is not None)
    report = symbolic_startup_storage_report(trace, source_profile=source_profile)
    if source_lifetime_policy is None:
        if source_compilation_options is not None:
            raise ValueError("compilation options require an explicit source lifetime policy")
    else:
        scope = ascend018_segment_source_scope(trace, model=model, source_profile=source_profile,
            model_execution_version=model_execution_version, tensor_parallel_size=parallelism.tensor_parallel_size,
            source_lifetime_policy=source_lifetime_policy, source_compilation_options=source_compilation_options)
        record = segment_reference_storage_report(trace, **scope)
        report = {**report, **record, "ranks": {str(trace.rank): record},
            "lifetime_derivation": source_lifetime_policy,
            "scope": "explicit source segment-reference storage policy; incomplete worker/runtime ledger"}
    return {**report, "num_compute_units": num_compute_units, "model_execution_version": model_execution_version,
        "tensor_parallel_size": parallelism.tensor_parallel_size,
        "uncovered": report["uncovered"] + (() if parallelism.tensor_parallel_size == 1 else ("collective_workspace",)),
        "collective_storage_source": None if parallelism.tensor_parallel_size == 1 else
            "ordinary CUDA fresh-output collectives; selected provider and collective scratch unbound"}


def symbolic_v4_startup_activation_peak(model: ModelConfig, parallelism: ParallelConfig, *, hardware,
        pool_name: str, tokens: int, max_requests: int, source_max_model_len: int, source_profile: str,
        source_use_deep_gemm: bool, workspace_initial_capacity_bytes: tuple[int, ...], indexer_max_logits_mb: int,
        expert_routing):
    """One production source program shared by reports and capacity projection.

    The normalized deployment describes synchronous collective output
    storage only; it supplies no library workspace or actual provider claim.
    Dynamic routing, workspace history and source limits stay explicit.
    """
    from src.kernel.hardware_spec import HWSpec
    from src.serving.deployment.deployment_config import CollectiveConfig, DeploymentConfig, WorkerPoolConfig
    from src.serving.deployment.deployment_plan import build_deployment_plan
    from src.serving.workload.traced_models import trace_deepseek_v4_startup_profile
    if (not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig)
            or not isinstance(hardware, HWSpec)):
        raise TypeError("V4 startup storage requires normalized model, parallelism and hardware")
    collectives = tuple(CollectiveConfig(operation, "ring", "nvlink", "ring", "declared-source-fabric", "analytical_roofline_v1")
        for operation in ("all_reduce", "all_gather"))
    deployment = build_deployment_plan(DeploymentConfig("colocated", (WorkerPoolConfig(pool_name, "colocated",
        hardware.hardware_id, parallelism, collectives=collectives),)), num_layers=model.num_hidden_layers)
    source = trace_deepseek_v4_startup_profile(model, deployment.worker_pools[pool_name], hardware, expert_routing,
        tokens=tokens, max_requests=max_requests, source_max_model_len=source_max_model_len,
        source_use_deep_gemm=source_use_deep_gemm, workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
        indexer_max_logits_mb=indexer_max_logits_mb, source_profile=source_profile)
    prediction = symbolic_startup_storage_report(source, source_profile=source_profile)
    prediction["source_inputs"] = {"tensor_parallel_size": parallelism.tensor_parallel_size,
        "max_model_len": source_max_model_len, "use_deep_gemm": source_use_deep_gemm,
        "workspace_initial_capacity_bytes": workspace_initial_capacity_bytes, "indexer_max_logits_mb": indexer_max_logits_mb,
        "expert_routing": {"source_kind": expert_routing.source_kind, "source_id": expert_routing.source_id,
            "root_seed": expert_routing.root_seed}}
    return prediction


def symbolic_glm_startup_activation_peak(model: ModelConfig, parallelism: ParallelConfig, *, hardware,
        pool_name: str, tokens: int, max_requests: int, source_max_model_len: int,
        source_profile: str, workspace_initial_capacity_bytes: tuple[int, ...], indexer_max_logits_mb: int,
        expert_routing, include_mtp: bool, num_speculative_tokens: int):
    """Trace constructor-sized GLM target/drafter/sampler scopes together.

    This closes the source startup program, not a full runtime memory ledger.
    Dynamic routed storage remains conditional on the explicitly supplied
    routing input; the maximum token/request constructor is not a proof of
    maximum workspace across every possible routing or compiler/provider.
    """
    from src.kernel.hardware_spec import HWSpec
    from src.serving.deployment.deployment_config import CollectiveConfig, DeploymentConfig, WorkerPoolConfig
    from src.serving.deployment.deployment_plan import build_deployment_plan
    from src.serving.workload.traced_glm_mtp import trace_glm_startup_profile
    if (not isinstance(model, ModelConfig) or not isinstance(parallelism, ParallelConfig)
            or not isinstance(hardware, HWSpec)):
        raise TypeError("GLM startup requires normalized model, parallelism and hardware")
    collectives = (() if parallelism.tensor_parallel_size == 1 else tuple(
        CollectiveConfig(op, "ring", "nvlink", "ring", "declared-source-fabric", "analytical_roofline_v1")
        for op in ("all_reduce", "all_gather")))
    deployment = build_deployment_plan(DeploymentConfig("colocated", (WorkerPoolConfig(pool_name, "colocated",
        hardware.hardware_id, parallelism, collectives=collectives),)), num_layers=model.num_hidden_layers)
    source = trace_glm_startup_profile(model, deployment.worker_pools[pool_name], expert_routing, tokens=tokens,
        max_requests=max_requests, source_max_model_len=source_max_model_len,
        workspace_initial_capacity_bytes=workspace_initial_capacity_bytes, indexer_max_logits_mb=indexer_max_logits_mb,
        source_profile=source_profile, include_mtp=include_mtp, num_speculative_tokens=num_speculative_tokens)
    program = "vllm_0_30_glm_no_metadata_target_" + (
        f"mtp{num_speculative_tokens}_" if include_mtp else "") + "pytorch_sort_fp32_noise"
    prediction = symbolic_startup_storage_report(source, source_profile=program)
    prediction["source_inputs"] = {"tensor_parallel_size": parallelism.tensor_parallel_size,
        "max_batched_tokens": tokens, "max_requests": max_requests, "max_model_len": source_max_model_len,
        "include_mtp": include_mtp, "num_speculative_tokens": num_speculative_tokens,
        "workspace_initial_capacity_bytes": workspace_initial_capacity_bytes, "indexer_max_logits_mb": indexer_max_logits_mb,
        "linear_source_profile": source_profile,
        "expert_routing": {"source_kind": expert_routing.source_kind, "source_id": expert_routing.source_id,
            "root_seed": expert_routing.root_seed}}
    prediction["uncovered"] += ("routing_conditioned_not_all_routing_maximum", "sparse_index_group_metadata_state",
        "complete_speculative_worker_state", "collective_workspace")
    return prediction
