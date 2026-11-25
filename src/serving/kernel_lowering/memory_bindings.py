"""Project the current source trace into typed per-call memory bindings.

This module owns no allocations. Every backing and view comes from the actual
ForwardTrace; unsupported operand mappings remain explicit on the KernelCall.
"""
from dataclasses import dataclass, fields, replace
from math import prod
from operator import is_
from types import MappingProxyType
from collections.abc import Mapping
import weakref

from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    EventKind, EventRef, MemoryAllocation, ProgramEvent, Strided, TensorView, Unmapped,
)
from src.kernel.memory_access_builder import ElementwiseBindings, GEMMWork, MemoryBindings
from src.serving.workload.workload_graph import WorkloadKind


@dataclass(frozen=True)
class SourceCallMemory:
    node_id: str
    rank: int
    pool_name: str
    inputs: tuple[TensorView, ...]
    outputs: tuple[TensorView, ...]
    input_producers: tuple[str | None, ...]
    allocations: tuple[MemoryAllocation, ...]
    predecessors: tuple[str, ...]
    kind: WorkloadKind
    variant: str | None
    attributes: Mapping[str, object]

    def __post_init__(self):
        self._validate(None)

    def _validate(self, validated):
        if not isinstance(self.attributes, Mapping):
            raise TypeError("memory source attributes require the actual source metadata mapping")
        object.__setattr__(self, "attributes", _source_metadata(self.attributes, frozen=True))
        _validate_source_call_header(self, self.rank, self.inputs, self.outputs,
            self.input_producers, self.allocations, self.predecessors)
        allocations = {item.allocation_id: item for item in self.allocations}
        if len(allocations) != len(self.allocations):
            raise ValueError("duplicate source allocations")
        for allocation in self.allocations:
            if type(allocation) is not MemoryAllocation:
                raise TypeError("source allocation must use the canonical memory contract")
            _validate_source_memory_value(allocation, validated)
        for view in (*self.inputs, *self.outputs):
            if type(view) is not TensorView or view.allocation_id not in allocations:
                raise ValueError("source port must reference its actual backing")
            _validate_source_memory_value(view, validated)

    def __reduce__(self):
        return type(self), tuple(_source_metadata(getattr(self, item.name), frozen=False)
            if item.name == "attributes" else getattr(self, item.name) for item in fields(self))


def _validate_source_call_header(node, rank, inputs, outputs, producers, allocations, predecessors):
    if not isinstance(node.kind, WorkloadKind):
        raise TypeError("memory source requires its actual operation kind")
    if node.variant is not None and (type(node.variant) is not str or not node.variant):
        raise ValueError("memory source variant must be explicit or absent")
    if (type(node.node_id) is not str or not node.node_id or type(rank) is not int or rank < 0
            or type(node.pool_name) is not str or not node.pool_name):
        raise ValueError("memory source requires a named physical-rank call")
    if any(type(values) is not tuple for values in (inputs, outputs, producers, allocations, predecessors)):
        raise TypeError("memory source facts must be immutable tuples")
    if len(inputs) != len(producers):
        raise ValueError("source readiness must cover every input port")
    if any(producer is not None and (type(producer) is not str or not producer) for producer in producers):
        raise ValueError("source producer identities must be nonempty or external")


def _validate_source_memory_value(value, validated, stable=None):
    # This memo exists for one synchronous snapshot only. The complete first
    # validation still runs for every actual object, including aliases.
    if validated is not None and id(value) in validated:
        if validated[id(value)] is not value:
            raise RuntimeError("source memory validation identity collision")
        return True
    if stable is not None:
        key = (type(value), value.view_id if type(value) is TensorView else value.allocation_id)
        if key in stable and _same_memory_referents((value,), stable[key]):
            validated[id(value)] = value
            return True
    value.__post_init__()
    if validated is not None:
        validated[id(value)] = value
    if stable is not None:
        stable[key] = _memory_referents((value,))
    return False


_IMMUTABLE_METADATA_SCALARS = (str, int, float, bool, bytes, type(None))


def _immutable_metadata_tuple(value, immutable):
    identity = id(value)
    if identity in immutable:
        if immutable[identity] is not value:
            raise RuntimeError("source metadata identity collision")
        return True
    for item in value:
        if type(item) in _IMMUTABLE_METADATA_SCALARS:
            continue
        if type(item) is not tuple or not _immutable_metadata_tuple(item, immutable):
            return False
    immutable[identity] = value
    return True


def _source_metadata(value, *, frozen, immutable=None):
    if type(value) in _IMMUTABLE_METADATA_SCALARS:
        return value
    if (frozen and immutable is not None and type(value) is tuple
            and _immutable_metadata_tuple(value, immutable)):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("source attribute keys must be strings")
        result = {key: _source_metadata(item, frozen=frozen, immutable=immutable) for key, item in value.items()}
        return MappingProxyType(result) if frozen else result
    if isinstance(value, (tuple, list)):
        return tuple(_source_metadata(item, frozen=frozen, immutable=immutable) for item in value)
    return value


def _same_source_metadata(current, frozen):
    """Compare the normalized metadata without merging scalar types."""
    if current is frozen:
        return True
    if isinstance(current, Mapping):
        if (not isinstance(frozen, Mapping) or current.keys() != frozen.keys()
                or any(type(key) is not str for key in current)):
            return False
        # Exactly the same children would each take the identity branch
        # above. Compare their ordered keys as well, so dictionary reordering
        # cannot associate a value with the wrong field. Mutable descendants
        # were detached when freezing and therefore take the recursive path.
        if all(map(is_, current, frozen)) and all(map(is_, current.values(), frozen.values())):
            return True
        return all(_same_source_metadata(value, frozen[key]) for key, value in current.items())
    if isinstance(current, (tuple, list)):
        return (type(frozen) is tuple and len(current) == len(frozen)
            and all(_same_source_metadata(a, b) for a, b in zip(current, frozen, strict=True)))
    return type(current) is type(frozen) and current == frozen


@dataclass(frozen=True)
class TraceMemoryContext:
    """A detached receipt of current source-owned ports, never a storage arena."""
    nodes: tuple[SourceCallMemory, ...]

    def __post_init__(self):
        if type(self.nodes) is not tuple or any(type(node) is not SourceCallMemory for node in self.nodes):
            raise TypeError("trace memory context requires immutable source-call facts")
        names = tuple(node.node_id for node in self.nodes)
        if len(names) != len(set(names)):
            raise ValueError("trace memory source identities must be unique")
        object.__setattr__(self, "_by_node", MappingProxyType({node.node_id: node for node in self.nodes}))

    def __getattribute__(self, name):
        if name == "nodes":
            state = object.__getattribute__(self, "__dict__")
            if "_pending_nodes" in state:
                pending = state["_pending_nodes"]
                result = tuple(pending[name] for name in pending)
                object.__setattr__(self, "nodes", result)
                del state["_pending_nodes"]
                return result
        return object.__getattribute__(self, name)

    @classmethod
    def _from_pending(cls, rows, cache, validated):
        lookup = _DeferredMemorySources(rows, cache, validated)
        result = object.__new__(cls)
        object.__setattr__(result, "_by_node", lookup)
        object.__setattr__(result, "_pending_nodes", lookup)
        return result

    def __reduce__(self):
        # Rebuild the immutable derived index at deserialization; only the
        # actual source facts cross an execution/export process boundary.
        return type(self), (self.nodes,)

    def bind_call(self, call):
        if len(call.source_node_ids) != 1:
            return replace(call, memory_bindings=None, memory_source=None,
                memory_unbound_reason="fused_source_operand_mapping_unavailable")
        node_id, = call.source_node_ids
        if node_id not in self._by_node:
            return replace(call, memory_bindings=None, memory_source=None,
                memory_unbound_reason="source_tensor_ports_unavailable")
        source = self._by_node[node_id]
        if source.rank != call.rank or source.pool_name != call.pool_name:
            raise ValueError("kernel call and source memory rank disagree")
        spec = call.kernel_spec
        if (spec.launch is not None and spec.launch.algorithm in (
                "h200_vllm030_torch213_fp16_external_aggregate_gemm",
                "h200_vllm030_torch213_fp16_external_aggregate_reduce",
                "h200_cublaslt129_bf16_external_aggregate_gemm",
                "h200_cublaslt129_bf16_external_aggregate_reduce")):
            # The public caller's final C is not the main's modeled scratch.
            # Keep the source receipt without inventing native partial views.
            binding, reason = None, "modeled_external_gemm_partial_ABI_unbound"
        elif (spec.kind is KernelKind.VECTOR and "mathematical_program" in spec.attributes
                and spec.attributes["mathematical_program"] == "vllm_0_30_dspark_typed_role_bytes_v1"):
            # Each spec is one typed role already projected from this current
            # receipt. Keep provenance; no native whole-node binding or copy.
            binding, reason = None, "dspark_typed_role_payload_projected_native_binding_unbound"
        elif spec.kind in (KernelKind.KV_CACHE, KernelKind.FLASH_ATTN, KernelKind.SPARSE_ATTN, KernelKind.SPARSE_ATTN_INDEXER):
            from .attention_memory_bindings import bind_attention_memory
            binding, reason = bind_attention_memory(call, source)
        elif ("symbolic_zero_extent_parameter_roles" in source.attributes or spec.kind is KernelKind.FUSED_MOE or spec.kind is KernelKind.VECTOR
                and isinstance(spec.variant, str) and (spec.variant.startswith("moe_") or spec.variant.endswith("_moe_route"))):
            from .moe_memory_bindings import bind_moe_memory
            binding, reason = bind_moe_memory(call, source)
        elif spec.kind is KernelKind.GEMM:
            binding, reason = _bind_gemm(call, source)
        elif call.kernel_spec.kind is KernelKind.COMMUNICATION:
            binding, reason = _bind_communication(call, source)
        elif call.kernel_spec.kind is KernelKind.VECTOR:
            binding, reason = _bind_elementwise(call, source)
        else:
            binding, reason = None, "source_family_operand_mapping_unavailable"
        return replace(call, memory_bindings=binding, memory_source=source,
            memory_unbound_reason=reason)


_SOURCE_MEMORY_RECEIPTS = weakref.WeakKeyDictionary()
_SOURCE_CALL_FIELDS = tuple(field.name for field in fields(SourceCallMemory))
_SOURCE_CALL_POST_INIT = (SourceCallMemory.__post_init__, SourceCallMemory.__post_init__.__code__)
_SOURCE_CALL_VALIDATE = (SourceCallMemory._validate, SourceCallMemory._validate.__code__)


def _can_defer_source_receipts():
    return ((SourceCallMemory.__post_init__, SourceCallMemory.__post_init__.__code__) == _SOURCE_CALL_POST_INIT
        and (SourceCallMemory._validate, SourceCallMemory._validate.__code__) == _SOURCE_CALL_VALIDATE)


def _receipt_cache(graph):
    """Cache only under a live run-owned source program, never a global shape."""
    from src.serving.workload.traced_forward import _NativeSourceFrame
    if "_native_source_frame" not in graph.__dict__ or not _can_defer_source_receipts():
        return None
    frame = graph.__dict__["_native_source_frame"]
    owner = frame.owner() if type(frame) is _NativeSourceFrame else None
    if owner is None:
        return None
    from src.kernel import memory_access
    from src.kernel.kernel_spec import DTYPE_BYTES
    validators = (SourceCallMemory.__post_init__, SourceCallMemory._validate,
        _source_metadata, _same_source_metadata, _immutable_metadata_tuple, _validate_source_call_header, _validate_source_memory_value,
        _memory_referents, _same_memory_referents, _compile_trace_projection, _reuse_trace_projection,
        _reuse_group_projection, _join_collective_memory,
        MemoryAllocation.__post_init__, TensorView.__post_init__,
        memory_access.Typed.__post_init__, memory_access.Strided.__post_init__,
        memory_access.region_end, memory_access.check_type,
        memory_access._contract_field_checks, memory_access._compile_type_check,
        memory_access._contract_type_hints)
    # Once per snapshot, include the actual inherited schema path and wrapped
    # compiler functions. Never rescan these dependencies per source node.
    signatures = []
    for function in validators:
        while True:
            signatures.append((function, function.__code__ if hasattr(function, "__code__") else None))
            if not hasattr(function, "__wrapped__"):
                break
            function = function.__wrapped__
    contract = (SourceCallMemory, tuple(signatures), frame.recipe_signature,
        memory_access.U64_MAX, tuple(DTYPE_BYTES.items()), _IMMUTABLE_METADATA_SCALARS)
    if owner not in _SOURCE_MEMORY_RECEIPTS or _SOURCE_MEMORY_RECEIPTS[owner][0] != contract:
        _SOURCE_MEMORY_RECEIPTS[owner] = (contract, {}, {}, {}, {})
    return _SOURCE_MEMORY_RECEIPTS[owner][1:]


def _memory_referents(values):
    # Keep actual objects alive and remember every stored field, including the
    # derived view region. A forged in-place field change cannot hit the cache.
    snapshots = []
    for value in values:
        state = vars(value)
        if type(value) is TensorView:
            actual_region = value.region()
            region_state = vars(actual_region)
            region = (actual_region, tuple(region_state), tuple(region_state.values()))
        else:
            region = None
        snapshots.append((value, tuple(state), tuple(state.values()), region))
    return tuple(snapshots)


def _same_memory_referents(values, previous):
    if len(values) != len(previous):
        return False
    for value, (old, names, fields, region) in zip(values, previous, strict=True):
        current_fields = vars(value)
        # Ordered names bind the positional identity comparison to each actual
        # field. Dict reordering merely misses reuse; equality cannot merge
        # False with zero, and no field/region check is omitted.
        if (value is not old or tuple(current_fields) != names
                or not all(map(is_, current_fields.values(), fields))):
            return False
        if region is not None:
            current = value.region()
            state = vars(current)
            if (current is not region[0] or tuple(state) != region[1]
                    or not all(map(is_, state.values(), region[2]))):
                return False
    return True


def _source_call_receipt(cache, node, rank, inputs, outputs, producers, allocations, predecessors, validated):
    if cache is not None and node.node_id in cache:
        old, referents = cache[node.node_id]
        if (rank is old.rank and node.pool_name == old.pool_name and node.kind is old.kind
                and node.variant == old.variant and producers == old.input_producers
                and predecessors == old.predecessors and _same_source_metadata(node.attributes, old.attributes)
                and _same_memory_referents((*inputs, *outputs, *allocations), referents)):
            return old
    values = (node.node_id, rank, node.pool_name, inputs, outputs,
        producers, allocations, predecessors, node.kind, node.variant, node.attributes)
    if (SourceCallMemory.__post_init__, SourceCallMemory.__post_init__.__code__) != _SOURCE_CALL_POST_INIT:
        result = SourceCallMemory(*values)
    else:
        result = object.__new__(SourceCallMemory)
        for name, value in zip(_SOURCE_CALL_FIELDS, values, strict=True):
            object.__setattr__(result, name, value)
        result._validate(validated)
    if cache is not None:
        cache[node.node_id] = result, _memory_referents((*inputs, *outputs, *allocations))
    return result


class _DeferredMemorySources(Mapping):
    """Complete checked source rows; instantiate only the consumer's calls."""

    def __init__(self, rows, cache, validated):
        self.rows = {row[0].node_id: row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError("trace memory source identities must be unique")
        self.cache, self.validated, self.materialized = cache, validated, {}

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, name):
        if name not in self.materialized:
            self.materialized[name] = _source_call_receipt(
                self.cache, *self.rows[name], self.validated)
        return self.materialized[name]


def _closed_metadata(value):
    return (type(value) in _IMMUTABLE_METADATA_SCALARS
        or type(value) is tuple and all(_closed_metadata(item) for item in value))


_OWNED_MEMORY_FIELDS = {kind: tuple(item.name for item in fields(kind))
    for kind in (TensorView, MemoryAllocation, Strided, SourceCallMemory)}


def _export_owned_memory(kind, values):
    """Copy already validated private values without rerunning the input boundary."""
    result = object.__new__(kind)
    result.__dict__.update(zip(_OWNED_MEMORY_FIELDS[kind], values, strict=True))
    return result


def _copyable_metadata(value):
    if type(value) in _IMMUTABLE_METADATA_SCALARS:
        return True
    if isinstance(value, Mapping):
        return all(type(key) is str and _copyable_metadata(item) for key, item in value.items())
    return type(value) in (tuple, list) and all(_copyable_metadata(item) for item in value)


class _OwnedMemorySources(Mapping):
    """Private scalar/tuple facts; public receipts materialize independent ports."""

    def __init__(self, rows, views, allocations):
        self.rows, self.views, self.allocations = rows, views, allocations
        self.materialized = {}

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, name):
        if name not in self.materialized:
            identity, rank, pool, inputs, outputs, producers, backings, parents, kind, variant, attrs = self.rows[name]
            views = {}
            for key in (*inputs, *outputs):
                values, region = self.views[key]
                view = _export_owned_memory(TensorView, values)
                object.__setattr__(view, "_region", _export_owned_memory(Strided, region))
                views[key] = view
            allocations = tuple(_export_owned_memory(MemoryAllocation, self.allocations[key]) for key in backings)
            self.materialized[name] = _export_owned_memory(SourceCallMemory, (identity, rank, pool,
                tuple(views[key] for key in inputs), tuple(views[key] for key in outputs), producers,
                allocations, parents, WorkloadKind(kind), variant, MappingProxyType(dict(attrs))))
        return self.materialized[name]


def _owned_context(rows, views, allocations):
    lookup = _OwnedMemorySources(rows, views, allocations)
    result = object.__new__(TraceMemoryContext)
    object.__setattr__(result, "_by_node", lookup)
    object.__setattr__(result, "_pending_nodes", lookup)
    return result


def _memory_fields(value):
    return tuple(getattr(value, item.name) for item in fields(value))


def _snapshot_execution_memory(source, graph):
    """Snapshot a closed internal bind; public source submissions stay checked."""
    from src.serving.workload.traced_forward import ForwardTrace, ForwardCollectiveGroup, _SourceFrameNodes
    if type(source) not in (ForwardTrace, ForwardCollectiveGroup):
        return snapshot_trace_memory(source, graph)
    traces = (source,) if type(source) is ForwardTrace else source.traces
    if any(type(trace._nodes) is not _SourceFrameNodes or "_owned_memory_storage" not in trace.__dict__
            for trace in traces):
        return snapshot_trace_memory(source, graph)
    memo = _receipt_cache(graph)
    if memo is None:
        return snapshot_trace_memory(source, graph)
    storage = tuple(trace.__dict__["_owned_memory_storage"] for trace in traces)
    if any(type(item) is not tuple or len(item) != 6
            or trace._allocations is not item[0] or trace._views is not item[1]
            or trace._persistent_ids is not item[4] or trace._events is not item[5]
            for trace, item in zip(traces, storage, strict=True)):
        return snapshot_trace_memory(source, graph)
    _, _, _, projection = memo
    identity = tuple(value for trace, (_, _, allocation_overlay, view_overlay, _, _) in zip(traces, storage, strict=True)
        for value in (trace._nodes.base, allocation_overlay.base, view_overlay.base,
            allocation_overlay.bindings, view_overlay.bindings))
    node_maps = tuple((trace._nodes, trace._collective_ids) for trace in traces)
    if type(source) is ForwardCollectiveGroup:
        joined = graph.__dict__["_native_source_frame"].joined
        if type(joined) is not _SourceFrameNodes:
            return snapshot_trace_memory(source, graph)
        node_maps = (*node_maps, (joined, ()))
    frame = graph.__dict__["_native_source_frame"]
    identity = (*identity, frame.template)
    membership = (() if type(source) is ForwardTrace else
        tuple((name, tuple(parts.items())) for name, parts in source._submissions.items()))
    previous = projection["owned"] if "owned" in projection else None
    if previous is not None and membership != previous[4]:
        previous = None
    compile_owned = previous is None or not all(a is b for a, b in zip(identity, previous[0], strict=True))
    if compile_owned:
        if any(not all(type(key) is str and (_closed_metadata(value)
                or name in nodes.updates and key in nodes.updates[name][1])
                for key, value in node.attributes.items())
                for nodes, _ in node_maps for name, node in nodes.base.items()):
            return snapshot_trace_memory(source, graph)
        # Actual current source validation happens before compiling any reuse.
        context = snapshot_trace_memory(source, graph)
        # The ordinary projector has checked every pending row already.
        # Do not instantiate and retain a second full set of public receipts.
        pending = tuple(context._by_node.rows.values())
        if any(not all(type(key) is str and _copyable_metadata(value)
                for key, value in row[0].attributes.items()) for row in pending):
            return context
        rows = {node.node_id: (node.node_id, rank, node.pool_name,
            tuple(view.view_id for view in inputs), tuple(view.view_id for view in outputs),
            producers, tuple(value.allocation_id for value in backings),
            parents, node.kind.value, node.variant, tuple(node.attributes.items()))
            for node, rank, inputs, outputs, producers, backings, parents in pending}
        views = {view.view_id: (_memory_fields(view), _memory_fields(view.region()))
            for _, _, inputs, outputs, _, _, _ in pending for view in (*inputs, *outputs)}
        allocations = {value.allocation_id: _memory_fields(value)
            for _, _, _, _, _, backings, _ in pending for value in backings}
        previous = (identity, rows, views, allocations, membership)
    _, old_rows, old_views, old_allocations, _ = previous
    rows, views, allocations = dict(old_rows), dict(old_views), dict(old_allocations)
    for nodes, local_collectives in node_maps:
        for name, (shape, attrs) in nodes.updates.items():
            if name in local_collectives:
                continue
            if (name not in rows or any(type(value) is not int or value < 0 for value in shape.values())
                    or any(type(key) is not str or not _copyable_metadata(value) for key, value in attrs.items())
                    or "symbolic_tensor_inputs" in attrs or "symbolic_tensor_outputs" in attrs):
                return snapshot_trace_memory(source, graph)
            row = rows[name]
            current_attrs = dict(row[-1])
            current_attrs.update(_source_metadata(attrs, frozen=True))
            rows[name] = (*row[:-1], tuple(current_attrs.items()))
    for _, _, allocation_overlay, view_overlay, _, _ in storage:
        for name, origin in allocation_overlay.bindings.items():
            current = allocation_overlay.current_values[origin]
            old = allocation_overlay.base[name]
            bound = current.with_lifetime(name, old.allocated_event, old.freed_event)
            bound.__post_init__()
            allocations[name] = _memory_fields(bound)
        for name, origin in view_overlay.bindings.items():
            current = view_overlay.current_values[origin]
            bound = current.with_identity(name, name)
            bound.__post_init__()
            if name not in allocations or allocations[name][3] is None:
                return snapshot_trace_memory(source, graph)
            from src.kernel.memory_access import region_end
            if region_end(bound.region()) > allocations[name][3]:
                raise ValueError("bound source view exceeds its actual backing allocation")
            views[name] = (_memory_fields(bound), _memory_fields(bound.region()))
    if compile_owned:
        # Replace the owner's entry rather than clearing maps potentially
        # retained by an earlier exported deferred snapshot.
        owner = frame.owner(check_source=False)
        if owner is None:
            return snapshot_trace_memory(source, graph)
        contract = _SOURCE_MEMORY_RECEIPTS[owner][0]
        _SOURCE_MEMORY_RECEIPTS[owner] = (contract, {}, {}, {}, {"owned": previous})
    return _owned_context(rows, views, allocations)


def _pending_source_call(node, rank, inputs, outputs, producers, allocations, predecessors,
                         validated, stable, metadata, immutable):
    # Snapshot all live ports now, including calls not consumed by a compact
    # predictor. Lazy construction never postpones backing/schema validation.
    _validate_source_call_header(node, rank, inputs, outputs, producers, allocations, predecessors)
    for value in (*inputs, *outputs, *allocations):
        if type(value) not in (MemoryAllocation, TensorView):
            raise TypeError("source ports require canonical memory contracts")
        _validate_source_memory_value(value, validated, stable)
    if node.node_id not in metadata or not _same_source_metadata(node.attributes, metadata[node.node_id]):
        metadata[node.node_id] = _source_metadata(node.attributes, frozen=True, immutable=immutable)
    # Only metadata is reusable. Rank, kind, variant and pool remain the
    # actual current node's facts, even when no compact consumer reads it.
    frozen_node = node.with_attributes(metadata[node.node_id])
    return (frozen_node, rank, inputs, outputs, producers, allocations, predecessors)


def _compile_trace_projection(source, graph, rows, state, canonical=None):
    """Retain a validated local topology, never the trace or its program owner."""
    nodes = tuple(source._nodes.values())
    if (len(nodes) != len(rows) or any(node.kind in (WorkloadKind.P2P, WorkloadKind.AF_ACTIVATION_TRANSFER)
            for node in nodes) or any(node.node_id != row[0].node_id for node, row in zip(nodes, rows))):
        state.clear()
        return
    canonical = tuple(node.node_id for node in nodes) if canonical is None else canonical
    referenced_views = {view.view_id: view for row in rows for view in (*row[2], *row[3])}
    referenced_allocations = {value.allocation_id: value for row in rows for value in row[5]}
    state["local"] = (
        source.rank, tuple(node.node_id for node in nodes), tuple(node.node_id for node in graph.nodes), canonical,
        source.event_order, tuple(view.view_id for view in source.views),
        tuple(value.allocation_id for value in source.allocations),
        tuple((name, value, value.allocation_id) for name, value in referenced_views.items()),
        tuple((name, value, value.allocated_event, value.freed_event)
            for name, value in referenced_allocations.items()), rows,
        tuple(frozenset((*[view.view_id for view in (*row[2], *row[3])],
            *[value.allocation_id for value in row[5]])) for row in rows),
    )


def _reuse_trace_projection(source, graph, state, receipts, stable, metadata):
    if "local" not in state:
        return None
    rank, names, graph_names, canonical, events, view_names, allocation_names, view_rows, allocation_rows, previous, dependencies = state["local"]
    nodes = tuple(source._nodes.values())
    current_events = source.event_order
    if (source.rank is not rank or len(nodes) != len(names)
            or any(node.node_id is not name for node, name in zip(nodes, names))
            or tuple(node.node_id for node in graph.nodes) != graph_names
            or len(current_events) != len(events) or not all(map(is_, current_events, events))
            or any(node.kind in (WorkloadKind.P2P, WorkloadKind.AF_ACTIVATION_TRANSFER) for node in nodes)):
        return None
    views = source.views
    allocations = source.allocations
    if (len(views) != len(view_names) or len(allocations) != len(allocation_names)
            or any(view.view_id is not name for view, name in zip(views, view_names))
            or any(value.allocation_id is not name for value, name in zip(allocations, allocation_names))):
        return None
    by_view = {view.view_id: view for view in views}
    by_allocation = {value.allocation_id: value for value in allocations}
    # Alias and lifetime relations are the compiled part. A changed relation
    # goes through the complete ordinary projection, which rebuilds or rejects.
    if (any(by_view[name].allocation_id is not backing for name, _, backing in view_rows)
            or any(by_allocation[name].allocated_event is not allocated
                or by_allocation[name].freed_event is not freed
                for name, _, allocated, freed in allocation_rows)):
        return None
    for node, row, name in zip(nodes, previous, canonical, strict=True):
        attrs = node.attributes
        if ("symbolic_tensor_inputs" not in attrs or "symbolic_tensor_outputs" not in attrs
                or not _same_source_metadata(attrs["symbolic_tensor_inputs"], row[0].attributes["symbolic_tensor_inputs"])
                or not _same_source_metadata(attrs["symbolic_tensor_outputs"], row[0].attributes["symbolic_tensor_outputs"])
                or not _same_source_metadata(tuple(graph.predecessors(name)), row[6])):
            return None
    validated, changed = {}, set()
    for name, old, _ in view_rows:
        value = by_view[name]
        if type(value) is not TensorView:
            return None
        unchanged = _validate_source_memory_value(value, validated, stable)
        if value is not old or not unchanged:
            changed.add(name)
    for name, old, _, _ in allocation_rows:
        value = by_allocation[name]
        if type(value) is not MemoryAllocation:
            return None
        unchanged = _validate_source_memory_value(value, validated, stable)
        if value is not old or not unchanged:
            changed.add(name)
    rows, immutable = [], {}
    for node, old, depends in zip(nodes, previous, dependencies, strict=True):
        header = (node.node_id, node.pool_name, node.kind, node.variant)
        old_header = (old[0].node_id, old[0].pool_name, old[0].kind, old[0].variant)
        if (depends.isdisjoint(changed) and all(map(is_, header, old_header))
                and _same_source_metadata(node.attributes, old[0].attributes)):
            rows.append(old)
        else:
            inputs = tuple(by_view[view.view_id] for view in old[2])
            outputs = tuple(by_view[view.view_id] for view in old[3])
            backings = tuple(by_allocation[value.allocation_id] for value in old[5])
            rows.append(_pending_source_call(node, rank, inputs, outputs, old[4], backings, old[6],
                validated, stable, metadata, immutable))
    result = tuple(rows)
    state["local"] = (rank, names, graph_names, canonical, events, view_names, allocation_names,
        tuple(row if by_view[row[0]] is row[1] else (row[0], by_view[row[0]], row[2]) for row in view_rows),
        tuple(row if by_allocation[row[0]] is row[1] else
            (row[0], by_allocation[row[0]], row[2], row[3]) for row in allocation_rows),
        result, dependencies)
    return TraceMemoryContext._from_pending(result, receipts, validated)


def _join_collective_memory(collective_parts, graph, receipt_cache, validated, defer_receipts,
                            stable_values, metadata, immutable_metadata):
    records = []
    for canonical, parts in collective_parts.items():
        node = graph.node(canonical)
        # Rank-local parts have already passed the same header, metadata,
        # backing and lifecycle checks. Joining needs their actual ports,
        # not a temporary SourceCallMemory object for each discarded part.
        operands = (tuple((row[2], row[3], row[4], row[5]) for row in parts) if defer_receipts else
            tuple((part.inputs, part.outputs, part.input_producers, part.allocations) for part in parts))
        ports = {view.view_id: view for reads, writes, _, _ in operands for view in (*reads, *writes)}
        ready_by_port = {view.view_id: producer for reads, _, producers, _ in operands
            for view, producer in zip(reads, producers, strict=True)}
        inputs = tuple(ports[name] for name in node.attributes["symbolic_tensor_inputs"])
        outputs = tuple(ports[name] for name in node.attributes["symbolic_tensor_outputs"])
        allocations = tuple({item.allocation_id: item for _, _, _, backings in operands for item in backings}.values())
        values = (node, node.rank, inputs, outputs,
            tuple(ready_by_port[view.view_id] for view in inputs), allocations, tuple(graph.predecessors(canonical)))
        records.append(_source_call_receipt(receipt_cache, *values, validated) if not defer_receipts else
            _pending_source_call(*values, validated, stable_values, metadata, immutable_metadata))
    return records


def _reuse_group_projection(source, graph, state, receipts, stable, metadata):
    if "group" not in state:
        return None
    submissions, states = state["group"]
    current = tuple((canonical, tuple(parts.values())) for canonical, parts in source._submissions.items())
    if not _same_source_metadata(current, submissions) or len(source.traces) != len(states):
        return None
    joined = {name: canonical for canonical, parts in current for name in parts}
    records, collective_parts, validated = [], {}, {}
    for trace, local_state in zip(source.traces, states, strict=True):
        local = _reuse_trace_projection(trace, graph, local_state, receipts, stable, metadata)
        if local is None:
            return None
        validated.update(local._by_node.validated)
        for row in local._by_node.rows.values():
            name = row[0].node_id
            if name in joined:
                canonical = joined[name]
                if canonical not in collective_parts:
                    collective_parts[canonical] = []
                collective_parts[canonical].append(row)
            else:
                records.append(row)
    records.extend(_join_collective_memory(collective_parts, graph, receipts, validated, True,
        stable, metadata, {}))
    return TraceMemoryContext._from_pending(tuple(records), receipts, validated)


def snapshot_trace_memory(source, graph):
    """Read actual ports in source call order, including alias writer changes.

    Final trace._writers is intentionally not used: an earlier input must see
    the writer that preceded that invocation, not the final writer of storage.
    """
    from src.serving.workload.traced_forward import (
        ForwardTrace, ForwardCollectiveGroup, ForwardPipelineGroup, ForwardActivationGroup, ForwardKVCacheGroup,
    )
    from src.serving.workload.workload_graph import WorkloadKind
    traces = (source,) if isinstance(source, ForwardTrace) else source.traces
    records = []
    validated = {}
    # Only recursively immutable tuples can be remembered. Mutable mappings
    # and lists are always read and detached, even within this one snapshot.
    immutable_metadata = {}
    memo = _receipt_cache(graph)
    defer_receipts = _can_defer_source_receipts()
    # Snapshot-local lazy records do not require a reusable source program.
    # Only native owners may retain anything across consecutive snapshots.
    receipt_cache, stable_values, metadata, projection = memo if memo is not None else (None, None, {}, {})
    if memo is not None and type(source) is ForwardTrace:
        reused = _reuse_trace_projection(source, graph, projection, receipt_cache, stable_values, metadata)
        if reused is not None:
            return reused
    if memo is not None and type(source) is ForwardCollectiveGroup:
        reused = _reuse_group_projection(source, graph, projection, receipt_cache, stable_values, metadata)
        if reused is not None:
            return reused
    local_projections = []
    compile_group = memo is not None and type(source) is ForwardCollectiveGroup
    graph_ids = {item.node_id for item in graph.nodes}
    owner = source._source if isinstance(source, ForwardKVCacheGroup) else source
    if isinstance(owner, ForwardTrace):
        groups = ()
    elif isinstance(owner, ForwardCollectiveGroup):
        groups = (owner, *owner.subgroups)
    elif isinstance(owner, (ForwardPipelineGroup, ForwardActivationGroup)):
        groups = owner.collective_groups
    else:
        raise TypeError("memory source must be an actual supported forward owner")
    joined_ids = {local_id: canonical for group in groups
        for canonical, submissions in group._submissions.items() for local_id in submissions.values()}
    transport_kinds = (WorkloadKind.P2P, WorkloadKind.AF_ACTIVATION_TRANSFER)
    collective_parts = {}
    all_views, all_allocations, allocation_events = {}, {}, {}
    for trace in traces:
        events = {event: index for index, event in enumerate(trace.event_order)}
        for view in trace.views:
            if view.view_id in all_views:
                raise ValueError("source ranks must have distinct view identities")
            all_views[view.view_id] = view
        for allocation in trace.allocations:
            if allocation.allocation_id in all_allocations:
                raise ValueError("source ranks must have distinct backing identities")
            all_allocations[allocation.allocation_id] = allocation
            allocation_events[allocation.allocation_id] = events
    for trace in traces:
        local_rows, canonical_names = [], []
        allocations = {item.allocation_id: item for item in trace.allocations}
        views = {item.view_id: item for item in trace.views}
        writers = {}
        # Receives are declared by the sending rank, before the destination
        # consumes that backing. Preserve their canonical producer identity.
        for node in graph.nodes:
            if node.kind in transport_kinds:
                for name in node.attributes["symbolic_tensor_outputs"]:
                    if name in views:
                        writers[views[name].allocation_id] = node.node_id
        event_positions = {event: index for index, event in enumerate(trace.event_order)}
        return_positions = {event.split(":", 2)[2]: index for event, index in event_positions.items()
            if event.split(":", 2)[1] == "return"}
        for node in trace._nodes.values():
            canonical = joined_ids[node.node_id] if node.node_id in joined_ids else node.node_id
            if canonical not in graph_ids:
                raise ValueError("memory source trace is not part of the executed graph")
            if "symbolic_tensor_inputs" not in node.attributes or "symbolic_tensor_outputs" not in node.attributes:
                continue
            inputs = tuple(views[name] for name in node.attributes["symbolic_tensor_inputs"])
            outputs = tuple(all_views[name] for name in node.attributes["symbolic_tensor_outputs"])
            ready = tuple(writers[view.allocation_id] if view.allocation_id in writers else None for view in inputs)
            ids = tuple(dict.fromkeys(view.allocation_id for view in (*inputs, *outputs)))
            transport = node.kind in transport_kinds
            if not transport and node.node_id not in return_positions:
                raise ValueError("source tensor call lacks its lifecycle boundary")
            input_ids = {view.allocation_id for view in inputs}
            for name in ids:
                allocation = all_allocations[name]
                positions = allocation_events[name]
                if transport:
                    label = ("send:" if name in input_ids else "receive:") + node.node_id
                    boundaries = tuple(index for event, index in positions.items()
                        if event.split(":", 1)[1] == label)
                    if len(boundaries) != 1:
                        raise ValueError("transport source needs its exact send/receive lifecycle boundary")
                    position, = boundaries
                else:
                    position = return_positions[node.node_id]
                if (positions[allocation.allocated_event] >= position
                        or allocation.freed_event is not None and positions[allocation.freed_event] <= position):
                    raise ValueError("source tensor backing is not live at its call boundary")
            values = (node, trace.rank, inputs, outputs, ready,
                tuple(all_allocations[name] for name in ids), tuple(graph.predecessors(canonical)))
            record = (_source_call_receipt(receipt_cache, *values, validated)
                if not defer_receipts else
                _pending_source_call(*values, validated, stable_values, metadata, immutable_metadata))
            if compile_group:
                local_rows.append(record)
                canonical_names.append(canonical)
            if node.node_id in joined_ids:
                if canonical not in collective_parts:
                    collective_parts[canonical] = []
                collective_parts[canonical].append(record)
            else:
                records.append(record)
            for view in outputs:
                writers[view.allocation_id] = canonical
        if compile_group:
            local_state = {}
            _compile_trace_projection(trace, graph, tuple(local_rows), local_state, tuple(canonical_names))
            local_projections.append(local_state)
    records.extend(_join_collective_memory(collective_parts, graph, receipt_cache, validated,
        defer_receipts, stable_values, metadata, immutable_metadata))
    if compile_group:
        projection["group"] = (
            tuple((canonical, tuple(parts.values())) for canonical, parts in source._submissions.items()),
            tuple(local_projections))
    if memo is not None and type(source) is ForwardTrace:
        _compile_trace_projection(source, graph, tuple(records), projection)
    return (TraceMemoryContext(tuple(records)) if not defer_receipts else
        TraceMemoryContext._from_pending(tuple(records), receipt_cache, validated))


def _bind_gemm(call, source):
    spec = call.kernel_spec
    if (source.kind not in (WorkloadKind.LINEAR, WorkloadKind.LM_HEAD, WorkloadKind.ROUTER)
            or spec.kind is not KernelKind.GEMM or spec.variant not in ("gemm", "gemv")):
        return None, "source_family_operand_mapping_unavailable"
    if "epilogue" in spec.attributes and spec.attributes["epilogue"] not in (None, "none"):
        return None, "gemm_epilogue_source_operand_mapping_unavailable"
    if spec.shape["b"] != 1 or len(source.inputs) != 2 or len(source.outputs) != 1:
        return None, "gemm_source_operand_arity_or_batch_unavailable"
    a, b = source.inputs
    c, = source.outputs
    m, n, k = (spec.shape[name] for name in ("m", "n", "k"))
    if (a.shape != (m, k) or b.shape != (n, k) or c.shape != (m, n)
            or a.dtype != spec.dtype or b.dtype != spec.dtype or c.dtype != spec.dtype):
        return None, "gemm_source_operand_geometry_or_dtype_unavailable"
    if a.allocation_id == c.allocation_id or b.allocation_id == c.allocation_id:
        return None, "inplace_gemm_source_unavailable"
    if any(view.strides_bytes[1] != view.element_bytes
            or view.strides_bytes[0] < view.shape[1] * view.element_bytes for view in (a, b, c)):
        return None, "gemm_source_operand_layout_unavailable"
    from src.kernel.memory_cann_gemm import (
        has_cann_gemm_source, has_cann_single_core_gemm_source,
        has_cann_deterministic_gemm_source, cann_split_execution_kernel,
    )
    from src.kernel.memory_gpu_split_gemm import has_gpu_split_gemm_abi
    from src.kernel.memory_access_builder import has_public_tensor_gemm_source
    execution = cann_split_execution_kernel(spec)
    if (has_gpu_split_gemm_abi(spec) or has_cann_single_core_gemm_source(execution)
            or has_cann_deterministic_gemm_source(execution)):
        return None, "gemm_native_split_workspace_source_ports_unavailable"
    if has_public_tensor_gemm_source(spec):
        return None, "gemm_opaque_backend_operand_mapping_unavailable"
    if has_cann_gemm_source(execution) and execution.shape != spec.shape:
        return None, "gemm_captured_physical_operand_mapping_unavailable"
    if has_cann_gemm_source(execution):
        # CANN exposes the same logical matrices, but does not identify the
        # backend ND/NZ format. Keep actual parents and advertise only logical
        # operand availability, never a native address/format mapping.
        roles = (
            replace(a, view_id="A", shape=(1, m, k), strides_bytes=(m * a.strides_bytes[0], *a.strides_bytes)),
            replace(b, view_id="B"),
            replace(c, view_id="C", shape=(1, m, n), strides_bytes=(m * c.strides_bytes[0], *c.strides_bytes)),
        )
        allocations, events, ready, visibility = source_memory_events(call, source,
            tuple(zip((a, b), source.input_producers, strict=True)))
        events = [replace(event, view_id=None) if event.view_id is not None else event for event in events]
        logical_ready = []
        for view, ref in zip(roles[:2], ready, strict=True):
            event_id = call.call_id + ":logical-native:" + view.view_id
            events.append(ProgramEvent(event_id, EventKind.LOGICAL_AVAILABLE, view.allocation_id,
                Unmapped(view.shape, view.dtype, view.element_bytes, event_id), visibility, (ref,), view.view_id))
            logical_ready.append(EventRef(event_id, EventKind.LOGICAL_AVAILABLE))
        return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations, roles,
            tuple(events), ("A", "B"), ("C",), tuple(logical_ready), call.rank, "source-forward", visibility), None
    work = GEMMWork(m, n, k, spec.dtype, a.element_bytes, trans_b=True)
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(source.inputs, source.input_producers, strict=True)))
    binding = MemoryBindings(work, call.call_id + ":source-memory", call.call_id, allocations,
        tuple(dict((view.view_id, view) for view in (a, b, c)).values()), events,
        a.view_id, b.view_id, c.view_id, *ready, call.rank, "source-forward", visibility)
    return binding, None


def source_memory_events(call, source, operands):
    """Project caller-ready facts without extending the source storage lifetime."""
    binding_id = call.call_id + ":source-memory"
    visibility = f"{call.pool_name}:rank:{call.rank}"
    # The receipt keeps original lifetimes. A single kernel cannot free its
    # caller's storage; its subprogram sees the same actual live backing.
    allocations = tuple(item.with_lifetime(item.allocation_id, item.allocated_event, None)
        for item in source.allocations)
    events = [ProgramEvent(item.allocated_event, EventKind.ALLOCATED,
        item.allocation_id, None, None, ()) for item in allocations]
    barriers = []
    for index, predecessor in enumerate(source.predecessors):
        event_id = f"{binding_id}:predecessor:{index}:{predecessor}"
        events.append(ProgramEvent(event_id, EventKind.BARRIER, None, None, None, ()))
        barriers.append(EventRef(event_id, EventKind.BARRIER))
    ready = []
    for index, (view, producer) in enumerate(operands):
        allocated = next(item.allocated_event for item in allocations if item.allocation_id == view.allocation_id)
        dependencies = (EventRef(allocated, EventKind.ALLOCATED), *barriers)
        if producer is not None:
            done = f"{binding_id}:input:{index}:source-done:{producer}"
            events.append(ProgramEvent(done, EventKind.COMPUTE_DONE, None, None, None, ()))
            dependencies += (EventRef(done, EventKind.COMPUTE_DONE),)
        initialized = f"{binding_id}:input:{index}:available"
        events.append(ProgramEvent(initialized, EventKind.INITIALIZED,
            view.allocation_id, view.region(), visibility, dependencies, view.view_id))
        ready.append(EventRef(initialized, EventKind.INITIALIZED))
    return allocations, tuple(events), tuple(ready), visibility


def _bind_elementwise(call, source):
    spec = call.kernel_spec
    if (spec.variant == "elementwise" and "mathematical_program" in spec.attributes
            and spec.attributes["mathematical_program"] == "vllm_0_30_fa3_aot_tail_clear"):
        return None, "fa3_metadata_clear_native_launch_unbound"
    if spec.variant in ("rmsnorm", "add_rmsnorm"):
        return _bind_normalization(call, source)
    if spec.variant in ("activation", "silu"):
        if source.kind is not WorkloadKind.ACTIVATION or source.variant != "silu":
            return None, "activation_source_operation_mapping_unavailable"
        if len(source.inputs) != 1 or len(source.outputs) != 1 or "tok" not in spec.shape:
            return None, "activation_source_packed_operand_unavailable"
        packed, = source.inputs
        output, = source.outputs
        rows, width = spec.shape["tok"], spec.shape["intermediate"]
        if (packed.shape != (rows, 2 * width) or output.shape != (rows, width)
                or packed.dtype != spec.dtype or output.dtype != spec.dtype):
            return None, "activation_source_operand_geometry_unavailable"
        # The source SiluAndMul reads two columns of the same packed tensor.
        # These role-local views alias that exact backing; no storage is made.
        left = replace(packed, view_id=call.call_id + ":gate", shape=(rows, width))
        right = replace(packed, view_id=call.call_id + ":up", shape=(rows, width),
            offset_bytes=packed.offset_bytes + width * packed.strides_bytes[-1])
        reads, writes = (left, right), (output,)
        producers = source.input_producers * 2
    elif spec.variant in ("elementwise", "elementwise_pytorch_vec4"):
        if source.kind is not WorkloadKind.ELEMENTWISE:
            return None, "elementwise_source_operation_mapping_unavailable"
        arity = {"add": (2, 1), "mul": (2, 1), "unary": (1, 1), "copy": (1, 1),
            "fill": (0, 1), "read": (1, 0)}
        operation = spec.attributes["operation"]
        if operation not in arity or (len(source.inputs), len(source.outputs)) != arity[operation]:
            return None, "elementwise_source_operand_arity_unavailable"
        reads, writes, producers = source.inputs, source.outputs, source.input_producers
        ports = (*reads, *writes)
        if (any(prod(view.shape) != spec.shape["element_count"] or view.dtype != spec.dtype for view in ports)
                or len({view.shape for view in ports}) != 1
                or "input_materialization" in spec.attributes):
            return None, "elementwise_source_operand_geometry_unavailable"
    else:
        return None, "source_family_operand_mapping_unavailable"
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(reads, producers, strict=True)))
    views = tuple({view.view_id: view for view in (*reads, *writes)}.values())
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id,
        allocations, views, events, tuple(view.view_id for view in reads),
        tuple(view.view_id for view in writes), ready, call.rank, "source-forward", visibility), None


def _bind_normalization(call, source):
    spec = call.kernel_spec
    fused = spec.variant == "add_rmsnorm"
    if source.kind is not WorkloadKind.NORM or source.variant != spec.variant:
        return None, "normalization_source_operation_mapping_unavailable"
    if (len(source.inputs), len(source.outputs)) != ((3, 2) if fused else (2, 1)):
        return None, "normalization_source_operand_arity_unavailable"
    names = ("input", "residual", "weight") if fused else ("input", "weight")
    reads = tuple(replace(view, view_id=name) for name, view in zip(names, source.inputs, strict=True))
    if fused:
        if source.outputs != source.inputs[:2]:
            return None, "normalization_source_inplace_mapping_unavailable"
        writes = reads[:2]
    else:
        writes = (replace(source.outputs[0], view_id="output"),)
    rows, hidden, width = (spec.shape[name] for name in ("rows", "hidden", "dtype_bytes"))
    for view in (*reads, *writes):
        shape = (hidden,) if view.view_id == "weight" else (rows, hidden)
        strides = (width,) if view.view_id == "weight" else (hidden * width, width)
        if (view.shape, view.strides_bytes, view.dtype, view.element_bytes) != (shape, strides, spec.dtype, width):
            return None, "normalization_source_operand_layout_unavailable"
    if len({view.allocation_id for view in reads}) != len(reads):
        return None, "normalization_source_input_alias_unavailable"
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(reads, source.input_producers, strict=True)))
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations,
        tuple({view.view_id: view for view in (*reads, *writes)}.values()), events,
        names, tuple(view.view_id for view in writes), ready, call.rank, "source-forward", visibility), None


def _dense_role_view(view, *, numeric):
    stride = view.element_bytes
    for dimension, actual in zip(reversed(view.shape), reversed(view.strides_bytes), strict=True):
        if dimension != 1 and actual != stride:
            return None
        stride *= dimension
    if numeric:
        return replace(view, view_id=view.view_id + ":scalar-role", shape=(prod(view.shape),),
            strides_bytes=(view.element_bytes,))
    return replace(view, view_id=view.view_id + ":byte-role", shape=(prod(view.shape) * view.element_bytes,),
        strides_bytes=(1,), dtype="uint8", element_bytes=1)


def _bind_communication(call, source):
    from src.kernel.memory_communication import communication_operand_layouts, has_communication_source
    spec = call.kernel_spec
    if (source.kind not in (WorkloadKind.COLLECTIVE, WorkloadKind.P2P, WorkloadKind.AF_ACTIVATION_TRANSFER)
            or not has_communication_source(spec)):
        return None, "communication_source_operation_mapping_unavailable"
    input_layouts, output_layouts = communication_operand_layouts(spec)
    if (len(source.inputs), len(source.outputs)) != (len(input_layouts), len(output_layouts)):
        return None, "communication_source_participant_ports_unavailable"
    numeric = spec.transfer.operation in ("all_reduce", "all_gatherv", "reduce_scatterv")
    reads = tuple(_dense_role_view(view, numeric=numeric) for view in source.inputs)
    writes = tuple(_dense_role_view(view, numeric=numeric) for view in source.outputs)
    if any(view is None for view in (*reads, *writes)):
        return None, "communication_source_contiguous_payload_unavailable"
    if any(prod(view.shape) * view.element_bytes != size for view, (_, size)
            in zip((*reads, *writes), (*input_layouts, *output_layouts), strict=True)):
        return None, "communication_source_payload_extent_unavailable"
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(reads, source.input_producers, strict=True)))
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations,
        tuple({view.view_id: view for view in (*reads, *writes)}.values()), events,
        tuple(view.view_id for view in reads), tuple(view.view_id for view in writes), ready,
        call.rank, "source-forward", visibility), None


def bind_trace_memory(plan, context):
    from .kernel_plan import KernelPlan
    if type(context) is not TraceMemoryContext:
        raise TypeError("source memory binding requires a current trace receipt")
    return KernelPlan(tuple(context.bind_call(call) for call in plan.calls),
        plan.block_repeats, memory_context=context)
