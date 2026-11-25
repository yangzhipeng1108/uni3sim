"""CPU symbolic eager forward tracing into the existing workload/storage contracts.

No tensor payloads or weights are allocated. Python references retain symbolic
storage exactly as the described forward does; graph records retain IDs only.
This models synchronous CPython eager scopes, not compiler liveness, asynchronous
GPU retirement, allocator reservation, or undocumented library workspace.
"""

from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping, MutableMapping, Sequence
from types import MappingProxyType
from collections import OrderedDict
from copy import copy
from functools import lru_cache, wraps
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import sys
import weakref
import operator
from math import prod, isfinite

from src.kernel.kernel_spec import DTYPE_BYTES
from src.kernel.memory_access import AddressLevel, MemoryAllocation, TensorView, U64_MAX, region_end
from src.serving.deployment.deployment_config import (
    CollectiveConfig, PipelineP2PConfig, worker_role_operator_group, worker_role_phase,
)
from .workload_graph import WorkloadEdge, WorkloadGraph, WorkloadKind, WorkloadNode, _workload_node_field_updates, _workload_node_field_updates_read, _node_update_source_unchanged


def _frame_rebinding(value, prefix, view_counter, bound_names, collective_prefixes=()):
    """Compile binding leaves without a recursive closure retaining its frame."""
    if isinstance(value, str):
        if (value in bound_names or value == prefix or value.startswith(prefix + ":")
                or value == view_counter or value.startswith(view_counter + ":")
                or any(value == item or value.startswith(item + ":") for item in collective_prefixes)):
            return lambda rename: rename(value)
        return None
    if isinstance(value, (tuple, list, Mapping)):
        items = value.items() if isinstance(value, Mapping) else enumerate(value)
        changes = tuple((key, program) for key, item in items
            if (program := _frame_rebinding(item, prefix, view_counter, bound_names,
                collective_prefixes)) is not None)
        if not changes:
            return None

        def apply(rename):
            result = dict(value) if isinstance(value, Mapping) else list(value)
            for key, program in changes:
                result[key] = program(rename)
            return tuple(result) if isinstance(value, tuple) else result

        return apply
    return None


class _ModuleFrameStorage(MutableMapping):
    """Exact source storage records, materialized only when a consumer reads them.

    Dead internal views/allocations need no Python contract objects during
    execution. Their original event IDs and typed geometry remain expandable;
    live output handles use the ordinary TensorView/MemoryAllocation owners.
    """
    def __init__(self, values, kind, stats):
        self.rows, self.kind, self.stats = dict(values), kind, stats

    def __len__(self): return len(self.rows)
    def __iter__(self): return iter(self.rows)
    def __contains__(self, name): return name in self.rows
    def __setitem__(self, name, value): self.rows[name] = value
    def __delitem__(self, name): del self.rows[name]

    def __getitem__(self, name):
        value = self.rows[name]
        if type(value) is tuple:
            template, rename, events = value
            if self.kind == "allocation":
                value = template.with_lifetime(name, events[template.allocated_event],
                    None if template.freed_event is None else events[template.freed_event])
            else:
                value = template.with_identity(name, rename(template.allocation_id))
            self.rows[name] = value
            self.stats[self.kind + "_materialized"] += 1
        return value

    def bind(self, name, template, rename, events):
        self.rows[name] = (template, rename, events)
        self.stats[self.kind + "_bound"] += 1

    def copy(self):
        result = object.__new__(_ModuleFrameStorage)
        result.rows, result.kind, result.stats = dict(self.rows), self.kind, self.stats
        return result


def _module_storage_snapshot(values):
    return values.copy() if type(values) is _ModuleFrameStorage else dict(values)


class _ForwardModuleFrame:
    """Detached metadata program recorded from one actual source invocation."""

    def __init__(self, trace, prefix, arguments, result, before):
        self.supported = False
        if (not isinstance(result, tuple) or not result
                or any(not isinstance(value, ForwardTensor) for value in result)
                or not before["persistent"] <= trace._persistent_ids
                or before["checkpoints"] != trace._checkpoints
                or before["workspaces"] != trace._workspace_capacities
                or before["captures"] != len(trace._captures)
                or not before["collectives"] <= trace._collective_ids
                or before["group"] is None and before["collectives"] != trace._collective_ids):
            return
        self.collectives = frozenset(trace._collective_ids - before["collectives"])
        self.collective_records = None
        self.collective_sequence = ()
        self.collective_prefix = None
        self.collective_sequence_prefix = None
        if before["group"] is not None:
            group, sequence_length, submissions, group_prefix = before["group"]
            rank = trace.rank
            self.collective_sequence_prefix = group_prefix[len(group.namespace) + 1:]
            self.collective_records = tuple(
                (identifier, group._contracts[identifier], group._layouts[identifier], current[rank])
                for identifier, current in group._submissions.items()
                if rank in current and (identifier not in submissions or submissions[identifier] != current[rank]))
            self.collective_sequence = tuple(group._sequences[rank][sequence_length:])
            self.collective_prefix = group_prefix
            if (len(self.collective_records) != len(self.collectives)
                    or len(self.collective_sequence) != len(self.collectives)):
                return
        self.persistent = frozenset(trace._persistent_ids - before["persistent"])
        self.prefix = prefix
        self.dependencies = before["dependencies"]
        self.arguments = tuple(None if value is None else value.view for value in arguments)
        self.readonly_views = frozenset(self.arguments[index].view_id
            for index in before["readonly_arguments"])
        self.readonly_allocations = frozenset(self.arguments[index].allocation_id
            for index in before["readonly_arguments"])
        self.argument_writers = tuple(None if value is None else before["writers"][value.view.allocation_id]
            for value in arguments)
        self.outputs = tuple((value.view.view_id,
            next((index for index, argument in enumerate(arguments) if value is argument), None))
            for value in result)
        self.nodes = tuple(trace._nodes.values())[before["nodes"]:]
        self.edges = tuple(trace._edges[before["edges"]:])
        self.views = tuple(trace._views[name] for name in trace._views if name not in before["views"])
        self.allocations = tuple(trace._allocations[name] for name in trace._allocations if name not in before["allocations"])
        self.events = tuple(trace._events[before["events"]:])
        if any(event.split(":", 2)[1] not in ("allocate", "return", "free") for event in self.events):
            return
        self.view_counter = trace.qualify("view")
        bound_names = {name for view in self.arguments if view is not None
            for name in (view.view_id, view.allocation_id)}
        bound_names.update(writer for writer in self.argument_writers if writer is not None)
        bound_names.update(self.dependencies)

        collective_prefixes = (() if self.collective_prefix is None else
            (self.collective_prefix, trace.qualify(self.collective_prefix)))
        self.node_rebindings = tuple((node, tuple((name, program) for name, value in node.attributes.items()
            if (program := _frame_rebinding(value, prefix, self.view_counter, bound_names,
                collective_prefixes)) is not None))
            for node in self.nodes)
        self.names = tuple((name, before["names"][name] if name in before["names"] else None, value)
            for name, value in trace._names.items() if name not in before["names"] or value != before["names"][name])
        ports = set()
        written = set()
        for node in self.nodes:
            for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_tensor_controls", "symbolic_parameter_inputs"):
                if field in node.attributes:
                    ports.update(node.attributes[field])
            written.update(trace._views[name].allocation_id for name in node.attributes["symbolic_tensor_outputs"])
        touched = {trace._views[name].allocation_id for name in ports}
        touched.update(view.allocation_id for view in self.views)
        touched.update(allocation.allocation_id for allocation in self.allocations)
        self.external_views = tuple(before["views"][name] for name in ports if name in before["views"])
        self.external = {name: (before["allocations"][name], before["references"][name],
            before["writers"][name], before["readers"][name])
            for name in touched if name in before["allocations"]}
        if (self.readonly_allocations.intersection(written)
                or any(view.allocation_id in self.readonly_allocations for view in self.views)
                or any(value.view.allocation_id in self.readonly_allocations for value in result)):
            return
        self.writers = {name: trace._writers[name] for name in touched}
        self.readers = {name: set(trace._readers[name]) for name in touched}
        self.written = written
        output_views = {name for name, argument in self.outputs if argument is None}
        counts = {}
        for name in output_views:
            storage = trace._views[name].allocation_id
            if storage not in counts:
                counts[storage] = 0
            counts[storage] += 1
        for name in touched:
            expected = counts[name] if name in counts else 0
            if name in self.external:
                expected += before["references"][name]
                if name not in written and not before["readers"][name] <= trace._readers[name]:
                    return
            if trace._references[name] != expected:
                return
        self.supported = True
        self.node_templates = {node.node_id: node for node in self.nodes}
        for node in self.nodes:
            object.__setattr__(node, "_module_source_frame", weakref.ref(self))

    def replay(self, trace, prefix, arguments, group=None, group_prefix=None, dependencies=()):
        substitutions = {}
        for source, target in zip(self.dependencies, dependencies, strict=True):
            substitutions[source] = target
        for old, old_writer, current in zip(self.arguments, self.argument_writers, arguments, strict=True):
            if old is None:
                continue
            pairs = [(old.view_id, current.view.view_id), (old.allocation_id, current.view.allocation_id)]
            if old_writer is not None:
                writer = trace._writers[current.view.allocation_id]
                if writer is None:
                    return None
                pairs.append((old_writer, writer))
            for source, target in pairs:
                if source in substitutions and substitutions[source] != target:
                    return None
                substitutions[source] = target
        old_view_start, new_view_start = None, None
        for name, initial, _ in self.names:
            if name == self.view_counter:
                old_view_start = -1 if initial is None else initial
                new_view_start = trace._names[name] if name in trace._names else -1

        local_collective_prefix = (None if self.collective_prefix is None else
            trace.qualify(self.collective_prefix))
        trace_namespace = trace.namespace

        def rename(value):
            if value in substitutions:
                return substitutions[value]
            if (local_collective_prefix is not None
                    and (value == local_collective_prefix or value.startswith(local_collective_prefix + ":"))):
                result = trace_namespace + ":" + group_prefix + value[len(local_collective_prefix):]
            elif (self.collective_prefix is not None
                    and (value == self.collective_prefix or value.startswith(self.collective_prefix + ":"))):
                result = group_prefix + value[len(self.collective_prefix):]
            elif value == self.prefix or value.startswith(self.prefix + ":"):
                result = prefix + value[len(self.prefix):]
            elif old_view_start is not None and (value == self.view_counter or value.startswith(self.view_counter + ":")):
                index = 0 if value == self.view_counter else int(value[len(self.view_counter) + 1:])
                if index > old_view_start:
                    index += new_view_start - old_view_start
                    result = self.view_counter if index == 0 else self.view_counter + ":" + str(index)
                else:
                    result = value
            else:
                result = value
            substitutions[value] = result
            return result

        def geometry(view):
            return view.offset_bytes, view.shape, view.strides_bytes, view.dtype, view.element_bytes

        for old in self.external_views:
            name = rename(old.view_id)
            if name not in trace._views:
                return None
            current_geometry, old_geometry = geometry(trace._views[name]), geometry(old)
            if ((current_geometry[1:] != old_geometry[1:]) if old.view_id in self.readonly_views
                    else current_geometry != old_geometry):
                return None
            if trace._views[name].allocation_id != rename(old.allocation_id):
                return None
        for name, (allocation, references, writer, readers) in self.external.items():
            target = rename(name)
            if target not in trace._allocations:
                return None
            actual = trace._allocations[target]
            readonly = name in self.readonly_allocations
            if (not readonly and actual.size_bytes != allocation.size_bytes or actual.memory_domain != allocation.memory_domain
                    or actual.device_id != allocation.device_id or not readonly and trace._references[target] != references
                    or trace._writers[target] != (None if writer is None else rename(writer))):
                return None
            if name in self.written and trace._readers[target] != {rename(reader) for reader in readers}:
                return None
        names = {}
        for name, initial, final in self.names:
            target = name if name == self.view_counter else rename(name)
            actual = trace._names[target] if target in trace._names else None
            if name == self.view_counter:
                names[target] = (-1 if actual is None else actual) + final - (-1 if initial is None else initial)
            else:
                if actual != initial:
                    return None
                names[target] = final
        if (any(rename(node.node_id) in trace._nodes for node in self.nodes)
                or any(rename(view.view_id) in trace._views for view in self.views)
                or any(rename(a.allocation_id) in trace._allocations for a in self.allocations)):
            return None
        collective_records = []
        if self.collective_records is not None:
            if (group is None or group_prefix is None or type(group) is not ForwardCollectiveGroup
                    or not any(trace is current for current in group.traces)):
                return None
            for identifier, contract, layout, submission in self.collective_records:
                if identifier == self.collective_prefix or identifier.startswith(self.collective_prefix + ":"):
                    target = group_prefix + identifier[len(self.collective_prefix):]
                else:
                    return None
                local_id = rename(submission)
                if target in group._submissions and trace.rank in group._submissions[target]:
                    return None
                current = contract.with_attributes(contract.attributes, node_id=target)
                if (target in group._contracts and group._contracts[target] != current
                        or target in group._layouts and group._layouts[target] != layout):
                    return None
                collective_records.append((target, current, layout, local_id))
            sequence_prefix = group_prefix[len(group.namespace) + 1:]
            collective_sequence = []
            for name in self.collective_sequence:
                if name == self.collective_sequence_prefix or name.startswith(self.collective_sequence_prefix + ":"):
                    collective_sequence.append(sequence_prefix + name[len(self.collective_sequence_prefix):])
                else:
                    return None
        events = {}
        for index, event in enumerate(self.events, start=len(trace._events)):
            _, kind, payload = event.split(":", 2)
            events[event] = f"{index}:{kind}:{rename(payload)}"

        trace._events.extend(events[event] for event in self.events)
        trace._names.update(names)
        deferred_storage = bool(self.readonly_allocations)
        if deferred_storage and type(trace._allocations) is not _ModuleFrameStorage:
            trace._allocations = _ModuleFrameStorage(trace._allocations, "allocation", trace._module_frame_stats)
        if deferred_storage and type(trace._views) is not _ModuleFrameStorage:
            trace._views = _ModuleFrameStorage(trace._views, "view", trace._module_frame_stats)
        for allocation in self.allocations:
            target = rename(allocation.allocation_id)
            if deferred_storage:
                trace._allocations.bind(target, allocation, rename, events)
            else:
                trace._allocations[target] = allocation.with_lifetime(target, events[allocation.allocated_event],
                    None if allocation.freed_event is None else events[allocation.freed_event])
            trace._references[target] = 0
        trace._persistent_ids.update(rename(name) for name in self.persistent)
        for view in self.views:
            target = rename(view.view_id)
            if deferred_storage:
                trace._views.bind(target, view, rename, events)
            else:
                trace._views[target] = view.with_identity(target, rename(view.allocation_id))
        for node, programs in self.node_rebindings:
            target = rename(node.node_id)
            attributes = {}
            for name, program in programs:
                attributes[name] = program(rename)
            trace._nodes[target] = node.with_updates({}, attributes, node_id=target, compose=True)
        edges = tuple(WorkloadEdge(rename(edge.source), rename(edge.destination), edge.kind) for edge in self.edges)
        if self.dependencies:
            # Binding external predecessors can change their lexical order.
            # Preserve call()'s data-then-barrier, sorted-source convention.
            positions = {rename(node.node_id): index for index, node in enumerate(self.nodes)}
            edges = tuple(sorted(edges, key=lambda edge: (
                positions[edge.destination], edge.kind != "data", edge.source)))
        trace._edges.extend(edges)
        if self.collective_records is not None:
            for target, contract, layout, local_id in collective_records:
                group._contracts[target] = contract
                group._layouts[target] = layout
                if target not in group._submissions:
                    group._submissions[target] = {}
                group._submissions[target][trace.rank] = local_id
            group._sequences[trace.rank].extend(collective_sequence)
        trace._collective_ids.update(rename(name) for name in self.collectives)
        for name, writer in self.writers.items():
            trace._writers[rename(name)] = None if writer is None else rename(writer)
        for name, readers in self.readers.items():
            target = rename(name)
            if name in self.external and name not in self.written:
                trace._readers[target].update(rename(reader) for reader in readers - self.external[name][3])
            else:
                trace._readers[target] = {rename(reader) for reader in readers}
        handles = {}
        result = []
        for name, argument in self.outputs:
            if argument is not None:
                result.append(arguments[argument])
            else:
                target = rename(name)
                if target not in handles:
                    handles[target] = trace._handle(trace._views[target])
                result.append(handles[target])
        return tuple(result)


_SOURCE_LAYOUT_CACHES = ContextVar("serving_source_layout_caches", default=None)
_SOURCE_LAYOUT_FACTORIES = ()


def _source_layout_cache(maxsize):
    """Keep the original bounded C LRU, owned by one experiment context."""
    def decorate(builder):
        global _SOURCE_LAYOUT_FACTORIES
        index = len(_SOURCE_LAYOUT_FACTORIES)
        _SOURCE_LAYOUT_FACTORIES += ((builder, maxsize),)
        default = lru_cache(maxsize=maxsize)(builder)

        def current():
            caches = _SOURCE_LAYOUT_CACHES.get()
            return default if caches is None else caches[index]

        @wraps(builder)
        def cached(*args, **kwargs):
            return current()(*args, **kwargs)

        cached.cache_info = lambda: current().cache_info()
        cached.cache_clear = lambda: current().cache_clear()
        cached.cache_parameters = default.cache_parameters
        return cached
    return decorate


@contextmanager
def _source_layout_cache_scope():
    caches = tuple(lru_cache(maxsize=limit)(builder)
        for builder, limit in _SOURCE_LAYOUT_FACTORIES)
    token = _SOURCE_LAYOUT_CACHES.set(caches)
    try:
        yield
    finally:
        _SOURCE_LAYOUT_CACHES.reset(token)
        for cache in caches:
            cache.cache_clear()


@_source_layout_cache(maxsize=4096)
def _source_allocation_template(device_id, memory_domain, size_bytes):
    """Reuse validated source storage geometry; each call binds fresh IDs/events."""
    return MemoryAllocation("source-allocation-layout", device_id, memory_domain,
        size_bytes, AddressLevel.OBJECT, None, None, "source-allocation-layout", None)


@_source_layout_cache(maxsize=2048)
def _dense_layout(shape, dtype):
    """Reuse a fully validated immutable layout, without retaining a trace.

    The caller checks concrete dimension types and dtype before lookup.
    New views validate their identities and reuse only validated geometry.
    """
    stride = DTYPE_BYTES[dtype]
    strides = []
    for dimension in reversed(shape):
        strides.append(stride)
        stride *= dimension
    return TensorView("layout", "layout", 0, shape, tuple(reversed(strides)), dtype, DTYPE_BYTES[dtype])


@_source_layout_cache(maxsize=4096)
def _strided_layout(shape, strides_bytes, dtype, offset_bytes):
    return TensorView("layout", "layout", offset_bytes, shape, strides_bytes, dtype, DTYPE_BYTES[dtype])


def _validate_layout_input(name, shape, dtype):
    if (not isinstance(name, str) or not name or not isinstance(shape, tuple)
            or not shape or any(type(d) is not int or d <= 0 for d in shape)):
        raise ValueError("symbolic tensors require a name and concrete positive dimensions")
    if dtype not in DTYPE_BYTES:
        raise NotImplementedError("symbolic forward requires an unpacked canonical dtype")
    if type(shape) is not tuple:
        raise TypeError("contract sequences must be immutable tuples")
    if type(dtype) is not str:
        raise ValueError("expected nonempty string")


def _release_forward_tensor(trace_reference, storage):
    # finalize keeps its callback alive globally. A bound trace method would
    # pin captured tensor handles and their entire trace after every step.
    trace = trace_reference()
    if trace is not None:
        trace._release(storage)


@dataclass(frozen=True)
class ForwardTensor:
    """A reference to a canonical view; Python aliases keep this handle alive."""

    trace: "ForwardTrace"
    view: TensorView

    @property
    def shape(self):
        return self.view.shape

    @property
    def dtype(self):
        return self.view.dtype

    @property
    def is_contiguous(self):
        stride = self.view.element_bytes
        for size, actual in zip(reversed(self.shape), reversed(self.view.strides_bytes), strict=True):
            if size != 1 and actual != stride:
                return False
            stride *= size
        return True

    def split(self, sizes, dim=-1):
        if dim not in (-1, len(self.shape) - 1) or sum(sizes) != self.shape[-1]:
            raise NotImplementedError("symbolic split requires a complete last-axis partition")
        offset = self.view.offset_bytes
        outputs = []
        for size in sizes:
            if type(size) is not int or size <= 0:
                raise ValueError("symbolic split sizes must be positive integers")
            outputs.append(self.trace.alias(self, self.trace.unique_name("view"),
                shape=(*self.shape[:-1], size), strides_bytes=self.view.strides_bytes,
                offset_bytes=offset))
            offset += size * self.view.strides_bytes[-1]
        return tuple(outputs)

    def narrow(self, dim, start, length):
        """A source slice retaining its full backing, including head padding."""
        if type(dim) is not int or not -len(self.shape) <= dim < len(self.shape):
            raise ValueError("symbolic narrow dimension must identify an existing axis")
        dim %= len(self.shape)
        if (type(start) is not int or type(length) is not int or start < 0 or length <= 0
                or start + length > self.shape[dim]):
            raise ValueError("symbolic narrow requires a nonempty in-bounds slice")
        shape = tuple(length if axis == dim else size for axis, size in enumerate(self.shape))
        return self.trace.alias(self, self.trace.unique_name("view"), shape=shape,
            strides_bytes=self.view.strides_bytes,
            offset_bytes=self.view.offset_bytes + start * self.view.strides_bytes[dim])

    def transpose(self, dim0, dim1):
        """Swap two metadata axes without copying or releasing the backing."""
        ndim = len(self.shape)
        if any(type(dim) is not int or not -ndim <= dim < ndim for dim in (dim0, dim1)):
            raise ValueError("symbolic transpose dimensions must identify existing axes")
        dim0, dim1 = dim0 % ndim, dim1 % ndim
        axes = list(range(ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self.trace.alias(self, self.trace.unique_name("view"),
            shape=tuple(self.shape[axis] for axis in axes),
            strides_bytes=tuple(self.view.strides_bytes[axis] for axis in axes),
            offset_bytes=self.view.offset_bytes)

    def unsqueeze(self, dim):
        """Insert a singleton view axis, preserving all original strides."""
        axes = len(self.shape) + 1
        if type(dim) is not int or not -axes <= dim < axes:
            raise ValueError("symbolic unsqueeze dimension must identify an insertion axis")
        dim %= axes
        stride = self.view.element_bytes if dim == len(self.shape) else self.shape[dim] * self.view.strides_bytes[dim]
        shape = (*self.shape[:dim], 1, *self.shape[dim:])
        strides = (*self.view.strides_bytes[:dim], stride, *self.view.strides_bytes[dim:])
        return self.trace.alias(self, self.trace.unique_name("view"), shape=shape,
            strides_bytes=strides, offset_bytes=self.view.offset_bytes)

    def expand(self, *shape):
        """Broadcast singleton axes as zero-stride views, without allocating."""
        if len(shape) != len(self.shape):
            raise NotImplementedError("symbolic expand requires explicit existing axes")
        dimensions, strides = [], []
        for original, requested, stride in zip(self.shape, shape, self.view.strides_bytes, strict=True):
            if requested == -1:
                requested = original
            if type(requested) is not int or requested <= 0 or (original != requested and original != 1):
                raise ValueError("symbolic expand only broadcasts singleton dimensions")
            dimensions.append(requested)
            strides.append(0 if original != requested else stride)
        return self.trace.alias(self, self.trace.unique_name("view"), shape=tuple(dimensions),
            strides_bytes=tuple(strides), offset_bytes=self.view.offset_bytes)

    def view_as_shape(self, *shape):
        """Dense reshape, or split/merge contiguous trailing axes of row views."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if -1 in shape:
            if shape.count(-1) != 1 or any(type(d) is not int or d == 0 or d < -1 for d in shape):
                raise ValueError("symbolic view accepts one inferred dimension and positive explicit dimensions")
            known = prod(d for d in shape if d != -1)
            if prod(self.shape) % known:
                raise ValueError("symbolic inferred view must preserve the number of elements")
            shape = tuple(prod(self.shape) // known if d == -1 else d for d in shape)
        if not shape or any(type(d) is not int or d <= 0 for d in shape) or prod(shape) != prod(self.shape):
            raise ValueError("symbolic view must preserve the number of elements")
        dense = self.trace._dense_view("layout", self.shape, self.dtype)
        result = self.trace._dense_view("layout", tuple(shape), self.dtype)
        strides = result.strides_bytes
        if not self.is_contiguous:
            if (len(self.shape) < 2 or len(shape) < 2 or shape[0] != self.shape[0]
                    or self.view.strides_bytes[1:] != dense.strides_bytes[1:]):
                raise NotImplementedError("symbolic view cannot merge noncontiguous storage chunks")
            strides = (self.view.strides_bytes[0], *strides[1:])
        return self.trace.alias(self, self.trace.unique_name("view"), shape=tuple(shape),
            strides_bytes=strides, offset_bytes=self.view.offset_bytes)


    def dim(self):
        return len(self.shape)


class ForwardTrace:
    """Record one rank's declared synchronous forward without real tensors.

    Operators supply existing WorkloadNodes and their source output layouts.
    Data and mutation dependencies, allocation/free events and alias ownership
    are inferred. Caller-local references remain live until reassigned, deleted
    or their function returns. Views retain their backing allocation. Internal
    workspace must be explicitly described by the operator's source program.
    """

    def __init__(self, *, rank: int = 0, namespace: str | None = None):
        if sys.implementation.name != "cpython":
            raise NotImplementedError("eager reference lifetimes require CPython")
        if type(rank) is not int or rank < 0:
            raise ValueError("forward trace requires a nonnegative physical rank")
        if namespace is not None and (not isinstance(namespace, str) or not namespace):
            raise ValueError("forward namespace must be nonempty when declared")
        self.rank = rank
        self.namespace = namespace
        self._allocations = {}
        self._references = {}
        self._views = {}
        self._writers = {}
        self._readers = {}
        self._events = []
        self._nodes = {}
        self._edges = []
        self._names = {}
        self._persistent_ids = set()
        self._external_input_origins = {}
        self._checkpoints = {}
        self._collective_ids = set()
        self._workspace_capacities = {}
        self._capture = None
        self._captures = []
        self._source_program_cache = SourceProgramCache()
        # Whole-source bindings and repeated source bodies use the same
        # run-local program owner and eviction policy.
        self._module_frames = self._source_program_cache.programs
        self._module_frame_stats = {"hits": 0, "misses": 0, "declined": 0,
            "allocation_bound": 0, "allocation_materialized": 0,
            "view_bound": 0, "view_materialized": 0, "node_bound": 0}
        self._module_program_count = 0
        self._module_program_types = {}
        self._recorded_graph = None

    def _has_mla_slot_proof(self, slots, capacity, dummy_rows):
        """Reuse only a retained native tuple's validated range/disjointness."""
        if "_mla_slot_proofs" not in self.__dict__:
            return False
        key = (id(slots), capacity, dummy_rows)
        if key not in self._mla_slot_proofs:
            return False
        if self._mla_slot_proofs[key] is not slots:
            raise RuntimeError("MLA slot proof identity collision")
        return True

    def _remember_mla_slot_proof(self, slots, capacity, dummy_rows):
        # The constructor calls this only after every original check passes.
        # Retaining the tuple prevents id reuse; the trace owns its lifetime.
        if "_mla_slot_proofs" not in self.__dict__:
            self._mla_slot_proofs = {}
        self._mla_slot_proofs[id(slots), capacity, dummy_rows] = slots
        if len(self._mla_slot_proofs) > 32:
            del self._mla_slot_proofs[next(iter(self._mla_slot_proofs))]

    def _module_program(self, module_type):
        """Name one homogeneous constructor scope without retaining its model."""
        self._module_program_count += 1
        self._module_program_types[self._module_program_count] = module_type
        return self._module_program_count

    def _call_module_frame(self, program, prefix, module, arguments, group=None, *, dependencies=(),
            readonly_arguments=()):
        """Replay one homogeneous frame with current tensor and dependency ports.

        Dependency ports bind actual preceding readers, such as a shared
        workspace's WAR predecessors. Replay still checks the complete live
        reader set before mutating state; missing ports decline the binding.
        """
        if type(program) is not int or program not in self._module_program_types:
            raise ValueError("module frame requires its registered constructor scope")
        if type(module) is not self._module_program_types[program]:
            raise ValueError("module frame type differs from its homogeneous constructor scope")
        if not isinstance(arguments, tuple):
            raise TypeError("module frame arguments must be a tuple")
        if (type(dependencies) is not tuple
                or any(type(name) is not str or name not in self._nodes for name in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            raise ValueError("module frame dependency ports must name distinct actual preceding nodes")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("module frame needs its nonempty source prefix")
        if group is not None and (type(group) is not ForwardCollectiveGroup
                or not any(self is trace for trace in group.traces)):
            raise ValueError("module frame collective group must contain this source rank")
        tensors = tuple(value for value in arguments if value is not None)
        for tensor in tensors:
            self._check_tensor(tensor)
        if (type(readonly_arguments) is not tuple or len(set(readonly_arguments)) != len(readonly_arguments)
                or any(type(index) is not int or not 0 <= index < len(arguments)
                    or arguments[index] is None for index in readonly_arguments)):
            raise ValueError("readonly module ports must name distinct actual input arguments")
        layouts = tuple(None if value is None else (
            0 if index in readonly_arguments else value.view.offset_bytes,
            value.shape, value.view.strides_bytes, value.dtype,
            0 if index in readonly_arguments else self._references[value.view.allocation_id])
            for index, value in enumerate(arguments))
        aliases = tuple(left.view.allocation_id == right.view.allocation_id
            for left in tensors for right in tensors)
        writers = tuple(self._writers[value.view.allocation_id] for value in tensors)
        writer_aliases = tuple(left == right for left in writers for right in writers)
        key = (program, type(module).__call__, layouts, aliases, tuple(value is None for value in writers), writer_aliases,
            len(dependencies), readonly_arguments)
        group_prefix = None if group is None else group.namespace + ":" + prefix
        if key in self._module_frames:
            frame = self._source_program_cache.lookup(key)
            if frame is not None:
                replay = frame.replay(self, self.qualify(prefix), arguments,
                    group=group, group_prefix=group_prefix, dependencies=dependencies)
                if replay is not None:
                    self._module_frame_stats["hits"] += 1
                    self._module_frame_stats["node_bound"] += len(frame.nodes)
                    return replay
            self._source_program_cache.store_module_frame(key, None)
            self._module_frame_stats["declined"] += 1
            return module(*arguments)
        self._module_frame_stats["misses"] += 1
        before = {
            "readonly_arguments": readonly_arguments,
            "dependencies": dependencies,
            "views": _module_storage_snapshot(self._views), "allocations": _module_storage_snapshot(self._allocations),
            "references": dict(self._references), "writers": dict(self._writers),
            "readers": {name: set(readers) for name, readers in self._readers.items()},
            "names": dict(self._names), "events": len(self._events),
            "nodes": len(self._nodes), "edges": len(self._edges),
            "persistent": set(self._persistent_ids), "checkpoints": dict(self._checkpoints),
            "workspaces": dict(self._workspace_capacities), "captures": len(self._captures),
            "collectives": set(self._collective_ids),
            "group": None if group is None else (group, len(group._sequences[self.rank]),
                {identifier: submissions[self.rank] for identifier, submissions in group._submissions.items()
                    if self.rank in submissions}, group_prefix),
        }
        result = module(*arguments)
        frame = _ForwardModuleFrame(self, self.qualify(prefix), arguments, result, before)
        self._source_program_cache.store_module_frame(key, frame if frame.supported else None)
        return result

    def bind_launch_policy(self, policy):
        """Bind kernel selection while preserving transport-owned contracts."""
        if policy not in ("analytical_launch", "injected_launch"):
            raise ValueError("virtual source requires an explicit supported launch policy")
        if (type(self._nodes) is _SourceFrameNodes and self._nodes.launch_policy == policy
                and all("launch_policy" not in attrs for _, attrs in self._nodes.updates.values())):
            return
        kernel_kinds = {WorkloadKind.LINEAR, WorkloadKind.LM_HEAD, WorkloadKind.ROUTER,
            WorkloadKind.FLASH_ATTN, WorkloadKind.EAGER_ATTN, WorkloadKind.FUSED_MOE,
            WorkloadKind.NORM, WorkloadKind.ACTIVATION, WorkloadKind.ROTARY,
            WorkloadKind.ELEMENTWISE, WorkloadKind.REDUCE, WorkloadKind.GATHER,
            WorkloadKind.VECTOR, WorkloadKind.KV_UPDATE}
        transport_kinds = {WorkloadKind.COLLECTIVE, WorkloadKind.P2P, WorkloadKind.PD_KV_TRANSFER,
            WorkloadKind.AF_ACTIVATION_TRANSFER, WorkloadKind.MEMCPY, WorkloadKind.KV_OFFLOAD}
        bound = {}
        for identifier, node in self._nodes.items():
            if node.kind in kernel_kinds:
                bound[identifier] = (node if "launch_policy" in node.attributes
                    and node.attributes["launch_policy"] == policy else
                    node.with_attributes({**node.attributes, "launch_policy": policy}))
            elif node.kind in transport_kinds:
                bound[identifier] = node
            else:
                raise ValueError(f"launch policy has no contract for workload kind {node.kind!r}")
        if any(bound[name] is not node for name, node in self._nodes.items()):
            self._nodes = bound

    def begin_capture(self, *, key, logical_tokens, model_input_tokens, inputs, runtime_mode="FULL", eager_kinds=(),
            padding_program="vllm_masked_minus1", source_stage=None, extra_inputs=None,
            source_graph_batch_size=None):
        """Bind static source ports while tracing the actual padded forward.

        This records a source graph's object identities, not CUDA pool
        reservation or a compiled allocation/replay observation.
        """
        fa3_aot = (source_stage is None and isinstance(extra_inputs,dict)
            and set(extra_inputs)=={"fa3_cu_query","fa3_used_key","fa3_scheduler_metadata"})
        pipeline = source_stage == "vllm_pipeline"
        af = source_stage == "midlayer_af"
        local_stage = pipeline or af
        pipeline_ports = ({"input_ids", "positions", "slot_mapping"},
            {"hidden_states", "residual", "positions", "slot_mapping"})
        stage_ports = (*pipeline_ports, {"hidden_states", "residual"}) if af else pipeline_ports
        if (self._capture is not None or not isinstance(key, str) or not key
                or type(logical_tokens) is not int or type(model_input_tokens) is not int
                or not 0 < logical_tokens <= model_input_tokens
                or (set(inputs) not in stage_ports if local_stage else set(inputs) != pipeline_ports[0])):
            raise ValueError("capture requires one explicit bucket and complete static source ports")
        if (runtime_mode not in ("FULL", "PIECEWISE") or not isinstance(eager_kinds, tuple)
                or len(eager_kinds) != len(set(eager_kinds))
                or (runtime_mode == "FULL" and eager_kinds)
                or (runtime_mode == "PIECEWISE" and set(eager_kinds) != {"flash_attn", "kv_update"})):
            raise ValueError("piecewise source capture requires explicit eager boundaries")
        if padding_program not in ("vllm_masked_minus1", "sglang_zero_slot0_seq1"):
            raise ValueError("source capture requires a declared padding program")
        if padding_program == "sglang_zero_slot0_seq1" and (runtime_mode != "FULL" or logical_tokens >= model_input_tokens):
            raise ValueError("SGLang zero-slot padding requires an underfilled FULL capture")
        if source_stage not in (None, "vllm_pipeline", "midlayer_af", "sglang_eagle_draft_decode", "sglang_eagle_target_verify", "sglang_eagle_draft_extend"):
            raise ValueError("source capture stage must name a supported source boundary")
        if source_stage is not None and not local_stage and (runtime_mode != "FULL"
                or (logical_tokens != model_input_tokens and padding_program != "sglang_zero_slot0_seq1")):
            raise NotImplementedError("EAGLE source stage padding requires its explicit SGLang FULL program")
        if source_graph_batch_size is not None and (source_stage is None
                or type(source_graph_batch_size) is not int or source_graph_batch_size <= 0):
            raise ValueError("source graph request-batch identity requires a positive staged declaration")
        if extra_inputs is not None and ((source_stage is None and not fa3_aot) or not isinstance(extra_inputs, dict)
                or any(not isinstance(role, str) or not role or role in inputs for role in extra_inputs)):
            raise ValueError("additional capture inputs require distinct named staged source ports")
        if fa3_aot:
            for tensor in extra_inputs.values():self._check_tensor(tensor)
            seq=extra_inputs["fa3_used_key"]
            if runtime_mode!="FULL" or seq.dtype!="int32" or len(seq.shape)!=1:
                raise ValueError("FA3 FULL AOT requires its actual INT32 sequence input")
            b=seq.shape[0]
            n=(3 if b<=992 else 2)*((b+3)//4*4)+1
            if (not logical_tokens<=b<=model_input_tokens
                    or extra_inputs["fa3_cu_query"].dtype!="int32" or extra_inputs["fa3_cu_query"].shape!=(b+1,)
                    or extra_inputs["fa3_scheduler_metadata"].dtype!="int32" or extra_inputs["fa3_scheduler_metadata"].shape!=(n,)):
                raise ValueError("FA3 FULL AOT static ports must preserve actual padded metadata geometry")
        if local_stage and (extra_inputs is not None or source_graph_batch_size is not None):
            raise ValueError("pipeline capture takes only its original model stage ports")
        if source_stage is not None and not local_stage:
            batch = source_graph_batch_size
            if type(batch) is not int or batch <= 0:
                raise ValueError("EAGLE staged capture requires its explicit positive request batch")
            draft = source_stage == "sglang_eagle_draft_decode"
            if logical_tokens % (1 if draft else 3):
                raise ValueError("EAGLE stage logical rows must contain complete request windows")
            if model_input_tokens != batch * (1 if draft else 3):
                raise ValueError("EAGLE stage physical rows must be draft B or verify/extend3B")
            extras = {} if extra_inputs is None else extra_inputs
            expected = ({"hidden_states", "topk_p", "topk_index"} if draft else
                {"hidden_states", "num_correct_drafts", "num_accept_tokens"}
                if source_stage == "sglang_eagle_draft_extend" else set())
            if set(extras) != expected:
                raise ValueError("EAGLE stage requires every declared hidden/topk/acceptance input port")
            for role, tensor in extras.items():
                self._check_tensor(tensor)
                if role == "hidden_states":
                    valid = tensor.dtype == "bf16" and len(tensor.shape) == 2 and tensor.shape[0] == model_input_tokens
                elif role in ("topk_p", "topk_index"):
                    valid = tensor.shape == (batch, 1) and tensor.dtype == ("fp32" if role == "topk_p" else "int64")
                else:
                    valid = tensor.shape == (batch,) and tensor.dtype == "int32"
                if not valid:
                    raise ValueError("EAGLE stage input dtype/shape must match its source role")
        if any(capture["key"] == key for capture in self._captures):
            raise ValueError("source capture keys must be distinct within a forward")
        ports = {}
        all_inputs = {**inputs, **({} if extra_inputs is None else extra_inputs)}
        for role, tensor in all_inputs.items():
            self._check_tensor(tensor)
            if local_stage and role in ("hidden_states", "residual"):
                if tensor.dtype != "bf16" or len(tensor.shape) != 2 or tensor.shape[0] != model_input_tokens:
                    raise ValueError("pipeline hidden/residual ports require physical BF16 rows")
            elif role in inputs and tensor.shape != (model_input_tokens,):
                raise ValueError("capture source port must contain its physical token extent")
            if local_stage and role not in ("hidden_states", "residual") and tensor.dtype != (
                    "int32" if role == "input_ids" else "int64"):
                raise ValueError("pipeline capture token/position/slot dtype must match its source port")
            self._persistent_ids.add(tensor.view.allocation_id)
            self._retain_capture_tensor(tensor)
            ports[role] = {"storage_id": f"rank:{self.rank}:{key}:{role}",
                "allocation_id": tensor.view.allocation_id, "shape": tensor.shape, "dtype": tensor.dtype,
                "bytes": self._allocations[tensor.view.allocation_id].size_bytes}
        self._capture = {"key": key, "logical_tokens": logical_tokens,
            "source_stage": source_stage,
            "source_graph_batch_size": source_graph_batch_size,
            "runtime_mode": runtime_mode, "eager_kinds": tuple(eager_kinds),
            "model_input_tokens": model_input_tokens, "inputs": ports, "output": None,
            "begin_event": self._event("capture.begin"), "end_event": None,
            "node_start": len(self._nodes), "node_ids": (), "allocator_reserved_bytes": None,
            "padding_ids": "unknown_retained_in_vocabulary_source_buffer_values",
            "padding_positions": "unknown_retained_in_range_source_buffer_values",
            "padding_slot_mapping": -1, "actual_provider_bound": False,
            "input_backing_capacity_bound": False,
            "port_identity_scope": "logical_rank_bucket_role_not_physical_backing_identity",
            "cross_bucket_backing_relationship": None,
            "cross_bucket_allocator_reserved_bytes": None,
            "static_input_scope": "captured_active_port_slices_not_complete_worker_buffer_capacity_graph_local_allocation_ids_not_cross_step_physical_backing"}
        if padding_program == "sglang_zero_slot0_seq1":
            self._capture.update(padding_ids="caller_declared_retained_source_values",
                padding_positions=0, padding_slot_mapping=0, padding_sequence_length=1,
                anonymous_dummy_kv_write_rows=model_input_tokens - logical_tokens,
                dummy_cache_disjointness="caller_declared_not_observed")
            if source_stage is not None:
                self._capture["padding_attention_kv_length"] = 4 if source_stage == "sglang_eagle_target_verify" else 1
                if source_stage == "sglang_eagle_draft_decode":
                    self._capture["padding_ids"] = 0

    def end_capture(self, output=None, *, outputs=None):
        if self._capture is None or self._capture["end_event"] is not None:
            raise ValueError("capture end requires one active source capture")
        if self._capture["source_stage"] is None:
            if outputs is not None:
                raise ValueError("ordinary capture retains its single-output source contract")
            self._check_tensor(output)
            named_outputs = {"hidden_states": output}
        else:
            if (output is not None or not isinstance(outputs, dict) or not outputs
                    or any(not isinstance(role, str) or not role for role in outputs)):
                raise ValueError("staged capture requires its complete named source output ports")
            pipeline = self._capture["source_stage"] in ("vllm_pipeline", "midlayer_af")
            draft = self._capture["source_stage"] == "sglang_eagle_draft_decode"
            expected = {"parent_list", "top_scores_index", "draft_tokens"} if draft else {"next_token_logits", "hidden_states"}
            if ((set(outputs) not in ({"hidden_states"}, {"hidden_states", "residual"})) if pipeline else set(outputs) != expected):
                raise ValueError("staged capture requires its complete named source output ports")
            for role, tensor in outputs.items():
                self._check_tensor(tensor)
                if pipeline:
                    valid = tensor.dtype == "bf16" and len(tensor.shape) == 2 and tensor.shape[0] == self._capture["model_input_tokens"]
                elif draft:
                    valid = tensor.dtype == "int64" and tensor.shape == (self._capture["source_graph_batch_size"], 2)
                else:
                    valid = (tensor.dtype == ("fp32" if role == "next_token_logits" else "bf16")
                        and len(tensor.shape) == 2 and tensor.shape[0] == self._capture["model_input_tokens"])
                if not valid:
                    raise ValueError("EAGLE stage output dtype/shape must match its complete source port")
            named_outputs = outputs
        ports = {}
        for role, tensor in named_outputs.items():
            self._check_tensor(tensor)
            self._persistent_ids.add(tensor.view.allocation_id)
            self._retain_capture_tensor(tensor)
            ports[role] = {"storage_id": f"rank:{self.rank}:{self._capture['key']}:{role}",
                "allocation_id": tensor.view.allocation_id, "shape": tensor.shape, "dtype": tensor.dtype,
                "bytes": self._allocations[tensor.view.allocation_id].size_bytes}
        if self._capture["source_stage"] is None:
            self._capture["output"] = ports["hidden_states"]
        else:
            self._capture["outputs"] = ports
        model_nodes = tuple(self._nodes)[self._capture["node_start"]:]
        eager = {name for name in model_nodes if self._nodes[name].kind.value in self._capture["eager_kinds"]
            or (self._capture["runtime_mode"] == "PIECEWISE" and self._nodes[name].kind is WorkloadKind.ELEMENTWISE
                and "source_attention_program" in self._nodes[name].attributes
                and self._nodes[name].attributes["source_attention_program"] == "ascend_0_18_piecewise_fia"
                and "source_component" in self._nodes[name].attributes
                and self._nodes[name].attributes["source_component"] == "symbolic_ascend_fia_output_copy")}
        segments, current = [], []
        for node in model_nodes:
            if node in eager:
                if current:
                    segments.append(tuple(current))
                    current = []
            else:
                current.append(node)
        if current:
            segments.append(tuple(current))
        self._capture["model_node_ids"] = model_nodes
        self._capture["eager_node_ids"] = tuple(node for node in model_nodes if node in eager)
        self._capture["node_ids"] = tuple(node for node in model_nodes if node not in eager)
        self._capture["source_segments"] = tuple(segments)
        self._capture["segment_scope"] = "source_op_split_regions_not_compiler_partitions_or_actual_replay_launches"
        self._capture["end_event"] = self._event("capture.end")
        self._captures.append(self._capture)
        self._capture = None

    @property
    def source_capture(self):
        if len(self._captures) > 1:
            raise ValueError("multi-stage source graphs require source_captures")
        return self._capture if self._capture is not None else (self._captures[0] if self._captures else None)

    @property
    def source_captures(self):
        if self._capture is not None:
            raise ValueError("source capture sequence cannot omit an unfinished active capture")
        return tuple(self._captures)

    @property
    def workspace_capacity_bytes(self):
        """Final declared arena sizes, without retaining managers or tensors."""
        return dict(self._workspace_capacities)

    def checkpoint(self, label):
        """Mark a source-program measurement boundary without tensor work."""
        if not isinstance(label, str) or not label or label in self._checkpoints:
            raise ValueError("forward checkpoint labels must be nonempty and unique")
        event = self._event(label)
        self._checkpoints[label] = event
        return event

    @property
    def checkpoints(self):
        return dict(self._checkpoints)

    def unique_name(self, prefix):
        """Stable module invocation/view IDs without per-model counters."""
        if self._recorded_graph is not None:
            raise ValueError("a bound source program is a closed forward trace")
        prefix = self.qualify(prefix)
        if prefix not in self._names:
            self._names[prefix] = 0
            return prefix
        self._names[prefix] += 1
        return f"{prefix}:{self._names[prefix]}"

    def qualify(self, name):
        if self.namespace is None or name.startswith(self.namespace + ":"):
            return name
        return self.namespace + ":" + name

    def _event(self, label):
        if self._recorded_graph is not None:
            raise ValueError("a bound source program is a closed forward trace")
        event = f"{len(self._events)}:{label}"
        self._events.append(event)
        return event

    def _handle(self, view):
        self._views[view.view_id] = view
        storage = view.allocation_id
        self._references[storage] += 1
        tensor = ForwardTensor(self, view)
        object.__setattr__(tensor, "_release_finalizer",
            weakref.finalize(tensor, _release_forward_tensor, weakref.ref(self), storage))
        return tensor

    def _retain_capture_tensor(self, tensor):
        # Capture owns this logical handle for the remaining trace lifetime.
        # Keep its reference count/lifetime unchanged when Python callers drop
        # it, without retaining tensor -> trace through a trace-owned list.
        # Detaching twice for an aliased static port is deliberately idempotent.
        tensor._release_finalizer.detach()

    def _release(self, storage):
        self._references[storage] -= 1
        if self._references[storage] == 0:
            allocation = self._allocations[storage]
            self._allocations[storage] = allocation.with_lifetime(storage,
                allocation.allocated_event, self._event(f"free:{storage}"))

    def _check_tensor(self, tensor):
        if not isinstance(tensor, ForwardTensor) or tensor.trace is not self:
            raise ValueError("forward tensor belongs to a different trace")

    @staticmethod
    def _dense_view(name, shape, dtype):
        # Tuple/str subclasses compare equal to canonical cache keys, but the
        # memory contract rejects them. Preserve that boundary before lookup.
        _validate_layout_input(name, shape, dtype)
        return _dense_layout(shape, dtype).with_identity(name, name)

    def tensor(self, name, shape, dtype, *, persistent=False, memory_domain="dram"):
        """Create independent activation storage, e.g. a source input or scratch."""
        if type(persistent) is not bool:
            raise TypeError("symbolic persistent flag must be boolean")
        name = self.qualify(name)
        view = self._dense_view(name, shape, dtype)
        return self._tensor_from_view(view, persistent=persistent, memory_domain=memory_domain)

    def parameter_view(self, name, shape, dtype):
        """Declare a readonly module parameter in this trace's storage owner.

        This is metadata only. The module source supplies its shape and dtype;
        no weight payload is loaded. Unlike activation handles, parameters stay
        live throughout the forward and never enter activation accounting.
        """
        name = self.qualify(name)
        layout = self._dense_view(name, shape, dtype)
        if name in self._views:
            existing = self._views[name]
            if (name not in self._persistent_ids or existing != layout
                    or self._writers[name] is not None):
                raise ValueError("source parameter identity or geometry changed")
            return existing
        if self._recorded_graph is not None:
            raise ValueError("cannot add parameters to a bound source trace")
        if name in self._allocations:
            raise ValueError("source parameter conflicts with an existing allocation")
        event = self._event(f"allocate:{name}")
        self._allocations[name] = _source_allocation_template(
            f"rank:{self.rank}", "dram", region_end(layout.region())).with_lifetime(name, event, None)
        self._views[name] = layout
        self._references[name] = 0
        self._writers[name] = None
        self._readers[name] = set()
        self._persistent_ids.add(name)
        return layout

    def _tensor_from_view(self, view, *, persistent=False, memory_domain="dram"):
        """Allocate from the exact view already validated by this call."""
        if memory_domain not in ("dram", "ddr", "buffer"):
            raise ValueError("forward allocation requires a declared dram, ddr or unbound on-chip buffer memory domain")
        name = view.view_id
        if name in self._allocations or name in self._views:
            raise ValueError("forward storage/view names must be unique")
        size_bytes = region_end(view.region())
        allocated_event = self._event(f"allocate:{name}")
        device_id = f"rank:{self.rank}"
        if type(memory_domain) is str and type(size_bytes) is int:
            self._allocations[name] = _source_allocation_template(device_id, memory_domain, size_bytes).with_lifetime(
                name, allocated_event, None)
        else:
            # Noncanonical values retain the full public constructor boundary.
            self._allocations[name] = MemoryAllocation(name, device_id, memory_domain,
                size_bytes, AddressLevel.OBJECT, None, None, allocated_event, None)
        self._references[name] = 0
        self._writers[name] = None
        self._readers[name] = set()
        if persistent:
            self._persistent_ids.add(name)
        return self._handle(view)

    def alias(self, tensor, name, *, shape, strides_bytes, offset_bytes=0, dtype=None):
        """Describe a typed view with an absolute offset into the same backing."""
        self._check_tensor(tensor)
        name = self.qualify(name)
        if not isinstance(name, str) or not name or name in self._views or name in self._allocations:
            raise ValueError("forward view names must be nonempty and unique")
        original = tensor.view
        dtype = original.dtype if dtype is None else dtype
        _validate_layout_input(name, shape, dtype)
        if type(strides_bytes) is not tuple:
            raise TypeError("contract sequences must be immutable tuples")
        if (type(offset_bytes) is not int or not 0 <= offset_bytes <= U64_MAX
                or any(type(stride) is not int or not 0 <= stride <= U64_MAX for stride in strides_bytes)):
            raise ValueError("expected unsigned 64-bit integer")
        width = DTYPE_BYTES[dtype]
        if offset_bytes % width or any(stride % width for stride in strides_bytes):
            raise ValueError("typed forward view offsets/strides must align to its element size")
        view = _strided_layout(shape, strides_bytes, dtype, offset_bytes).with_identity(name, original.allocation_id)
        if region_end(view.region()) > self._allocations[view.allocation_id].size_bytes:
            raise ValueError("forward view exceeds its backing storage")
        return self._handle(view)

    def declare_legacy_dummy_PP_input(self, tensor, *, role):
        """Declare inherited runner storage without a device write or value."""
        self._check_tensor(tensor)
        if (role not in ("hidden_states", "residual") or tensor.dtype != "bf16" or len(tensor.shape) != 2
                or not tensor.is_contiguous or tensor.view.offset_bytes != 0
                or tensor.view.allocation_id != self.qualify("worker.intermediate:" + role)
                or tensor.view.allocation_id not in self._persistent_ids
                or self._writers[tensor.view.allocation_id] is not None):
            raise ValueError("legacy dummy PP input requires its own persistent unknown active prefix")
        self._external_input_origins[tensor.view.view_id] = (role, tensor.view)

    def _legacy_dummy_PP_origin(self, tensor):
        self._check_tensor(tensor)
        if self._writers[tensor.view.allocation_id] is not None or tensor.view.view_id not in self._external_input_origins:
            return None
        role, original = self._external_input_origins[tensor.view.view_id]
        if original != tensor.view or tensor.view.allocation_id not in self._persistent_ids:
            raise ValueError("legacy dummy external input changed its bound layout or backing")
        return {"source": "vllm_0_30_legacy_dummy_PP_retained", "rank": self.rank, "role": role,
            "values_bound": False, "view": asdict(tensor.view),
            "allocation_bytes": self._allocations[tensor.view.allocation_id].size_bytes}

    def _complete_producer_view(self, tensor):
        """Resolve a same-layout alias to its one complete producer port."""
        self._check_tensor(tensor)
        writer = self._writers[tensor.view.allocation_id]
        if writer is None:
            raise ValueError("transfer requires an explicit source tensor producer")
        ports = tuple(self._views[view_id] for view_id in self._nodes[writer].attributes["symbolic_tensor_outputs"]
            if self._views[view_id].allocation_id == tensor.view.allocation_id
            and self._views[view_id].shape == tensor.view.shape
            and self._views[view_id].strides_bytes == tensor.view.strides_bytes
            and self._views[view_id].offset_bytes == tensor.view.offset_bytes
            and self._views[view_id].dtype == tensor.view.dtype)
        if len(ports) != 1:
            raise ValueError("transfer must match exactly one complete producer region")
        return ports[0]

    def call(self, node: WorkloadNode, inputs: tuple[ForwardTensor, ...], *, shape, dtype, out=None, writes=(), controls=(), parameters=()):
        """Record an operator, allocating its output unless source declares out.

        Reading aliases sees the current storage writer. Reusing out also waits
        for earlier readers/writers of that storage (WAR/WAW). Source operator
        contracts remain responsible for math, layout and supported branches.
        """
        if not isinstance(node, WorkloadNode) or node.rank != self.rank:
            raise ValueError("forward operator must declare this physical rank")
        node_id = self.qualify(node.node_id)
        if node_id in self._nodes:
            raise ValueError("forward operator IDs must be unique")
        if not isinstance(inputs, tuple) or (not inputs and out is None):
            raise ValueError("forward operators require tensor inputs or an explicit write-only output")
        for tensor in inputs:
            self._check_tensor(tensor)
        if not isinstance(writes, tuple):
            raise TypeError("symbolic mutation targets must be a tuple")
        for tensor in writes:
            self._check_tensor(tensor)
        if not isinstance(controls, tuple):
            raise TypeError("symbolic control dependencies must be a tuple")
        for tensor in controls:
            self._check_tensor(tensor)
        if out is not None:
            self._check_tensor(out)
            if out.view.shape != shape or out.view.dtype != dtype:
                raise ValueError("forward out layout disagrees with the declared result")
        if type(parameters) is not tuple:
            raise TypeError("source parameter ports must be a tuple")
        for view in parameters:
            if (type(view) is not TensorView or view.view_id not in self._views
                    or self._views[view.view_id] is not view
                    or view.allocation_id not in self._persistent_ids
                    or self._writers[view.allocation_id] is not None):
                raise ValueError("parameter port requires this trace's readonly parameter")
        # Validate before recording any events or allocating output storage.
        layout = self._dense_view(node_id, shape, dtype)
        dependencies = set()
        for tensor in inputs:
            writer = self._writers[tensor.view.allocation_id]
            if writer is not None:
                dependencies.add(writer)
        for target in ((*writes, out) if out is not None else writes):
            storage = target.view.allocation_id
            writer = self._writers[storage]
            if writer is not None:
                dependencies.add(writer)
            dependencies.update(self._readers[storage])
        control_dependencies = {self._writers[tensor.view.allocation_id] for tensor in controls}
        control_dependencies.discard(None)
        result = self._tensor_from_view(layout) if out is None else out
        self._nodes[node_id] = node.with_attributes({**node.attributes,
            "symbolic_tensor_inputs": (*tuple(t.view.view_id for t in inputs), *tuple(view.view_id for view in parameters)),
            "symbolic_tensor_outputs": tuple(t.view.view_id for t in (result, *writes)),
            **({"symbolic_parameter_inputs": tuple(view.view_id for view in parameters)} if parameters else {}),
            **({"symbolic_tensor_controls": tuple(t.view.view_id for t in controls)} if controls else {})}, node_id=node_id)
        self._edges.extend(WorkloadEdge(source, node_id) for source in sorted(dependencies))
        self._edges.extend(WorkloadEdge(source, node_id, "barrier")
            for source in sorted(control_dependencies - dependencies))
        for tensor in inputs:
            self._readers[tensor.view.allocation_id].add(node_id)
        for view in parameters:
            self._readers[view.allocation_id].add(node_id)
        for target in (*writes, result):
            storage = target.view.allocation_id
            self._writers[storage] = node_id
            self._readers[storage].clear()
        self._event(f"return:{node_id}")
        return result

    def call_effect(self, node, inputs, *, writes=(), controls=()):
        """Record a source launch with only declared in-place or no writes.

        A conditional kernel may launch while every lane skips its stores.
        Partial writes name real alias regions; this does not add element-level
        initialization tracking to the allocation-level dependency model.
        """
        if self._recorded_graph is not None:
            raise ValueError('a bound source program is a closed forward trace')
        if not isinstance(node, WorkloadNode) or node.rank != self.rank:
            raise ValueError("forward effect must declare this physical rank")
        name=self.qualify(node.node_id)
        if name in self._nodes:
            raise ValueError("forward effect operator IDs must be unique")
        for values in (inputs,writes,controls):
            if type(values) is not tuple:
                raise TypeError("forward effect ports must be explicit tuples")
            for tensor in values:
                self._check_tensor(tensor)
        dependencies={self._writers[tensor.view.allocation_id] for tensor in (*inputs,*writes)}
        dependencies.discard(None)
        for tensor in writes:
            dependencies.update(self._readers[tensor.view.allocation_id])
        barriers={self._writers[tensor.view.allocation_id] for tensor in controls}
        barriers.discard(None)
        self._nodes[name]=node.with_attributes({**node.attributes,
            'symbolic_tensor_inputs':tuple(tensor.view.view_id for tensor in inputs),
            'symbolic_tensor_outputs':tuple(tensor.view.view_id for tensor in writes),
            **({'symbolic_tensor_controls':tuple(tensor.view.view_id for tensor in controls)} if controls else {})},node_id=name)
        self._edges.extend(WorkloadEdge(source,name) for source in sorted(dependencies))
        self._edges.extend(WorkloadEdge(source,name,'barrier') for source in sorted(barriers-dependencies))
        for tensor in inputs:
            self._readers[tensor.view.allocation_id].add(name)
        for tensor in writes:
            storage=tensor.view.allocation_id
            self._writers[storage]=name
            self._readers[storage].clear()
        self._event(f'return:{name}')
        return name

    def call_outputs(self, node, inputs, *, outputs, workspace=(), controls=(), parameters=()):
        """One source operation with multiple independent fresh tensor outputs.

        Each output is (role, shape, dtype). Roles name storage; the call and
        input dependencies remain single. Optional source workspace uses the
        same layouts and is retained only within the operator scope. No tuple
        or graph record pins a handle after the caller releases it.
        """
        if not isinstance(node, WorkloadNode) or node.rank != self.rank:
            raise ValueError("forward operator must declare this physical rank")
        name = self.qualify(node.node_id)
        if name in self._nodes or not isinstance(inputs, tuple) or not inputs:
            raise ValueError("multi-output call needs a fresh node and explicit inputs")
        for tensor in inputs:
            self._check_tensor(tensor)
        if not isinstance(controls, tuple):
            raise TypeError("symbolic control dependencies must be a tuple")
        for tensor in controls:
            self._check_tensor(tensor)
        if not isinstance(outputs, tuple) or not outputs:
            raise ValueError("multi-output call requires output role/shape/dtype tuples")
        if not isinstance(workspace, tuple):
            raise TypeError("operator workspace requires role/shape/dtype tuples")
        layouts = []
        for output in (*outputs, *workspace):
            if not isinstance(output, tuple) or len(output) != 3:
                raise TypeError("output layout must contain role, shape and dtype")
            role, shape, dtype = output
            if not isinstance(role, str) or not role:
                raise ValueError("output role must be a nonempty string")
            layout = self._dense_view(name + ":" + role, shape, dtype)
            if layout.view_id in self._views or layout.allocation_id in self._allocations:
                raise ValueError("multi-output storage IDs must be fresh")
            layouts.append(layout)
        if len({view.view_id for view in layouts}) != len(layouts):
            raise ValueError("multi-output roles must be unique")
        results = tuple(self._tensor_from_view(view) for view in layouts[:len(outputs)])
        scratch = tuple(self._tensor_from_view(view) for view in layouts[len(outputs):])
        self.call(node, inputs, shape=results[0].shape, dtype=results[0].dtype,
            out=results[0], writes=results[1:], controls=controls, parameters=parameters)
        # Keep scratch across the source operation, then let its local scope
        # release it. It is neither a returned tensor nor a separately priced op.
        del scratch
        return results

    @property
    def graph(self):
        if self._collective_ids:
            raise ValueError("rank-local collective traces require the joined group graph")
        if self._recorded_graph is not None:
            nodes, graph = self._recorded_graph
            if nodes is self._nodes and graph is not None and len(nodes) == len(graph.__dict__["nodes"]):
                return graph
        return WorkloadGraph(tuple(self._nodes.values()), self._capture_program_edges())

    def _capture_program_edges(self):
        """Preserve synchronous source-stream boundaries around capture regions.

        Only capture entry/exit and PIECEWISE split boundaries are ordered.
        Adjacent region sinks join at the next region's first source operation,
        which gates its other independent roots. This linear-size boundary hub
        preserves the remaining intra-region DAG concurrency; it does not
        serialize every operation or infer compiler/internal stream parallelism.
        ForwardTrace describes one synchronous source program, not multistream
        execution. No new operator, tensor access or timing parameter is added.
        Ordinary traces without capture retain their exact original edges.
        """
        if not self._captures and self._capture is None:
            return tuple(self._edges)
        if self._capture is not None:
            raise ValueError("capture program boundaries require a finished source capture")
        names = tuple(self._nodes)
        regions = []

        def append_region(members, captured):
            if not members:
                return
            if not captured and regions and not regions[-1][1]:
                regions[-1][0].extend(members)
            else:
                regions.append((list(members), captured))

        cursor = 0
        for capture in self._captures:
            start = capture["node_start"]
            stop = start + len(capture["model_node_ids"])
            append_region(names[cursor:start], False)
            members, captured = [], None
            for name in names[start:stop]:
                node = self._nodes[name]
                inside = not (node.kind.value in capture["eager_kinds"]
                    or (capture["runtime_mode"] == "PIECEWISE" and node.kind is WorkloadKind.ELEMENTWISE
                        and "source_attention_program" in node.attributes
                        and node.attributes["source_attention_program"] == "ascend_0_18_piecewise_fia"
                        and "source_component" in node.attributes
                        and node.attributes["source_component"] == "symbolic_ascend_fia_output_copy"))
                if members and inside != captured:
                    append_region(members, captured)
                    members = []
                members.append(name)
                captured = inside
            append_region(members, captured)
            cursor = stop
        append_region(names[cursor:], False)
        ownership = {name: index for index, (members, _) in enumerate(regions) for name in members}
        incoming, outgoing = set(), set()
        existing = {(edge.source, edge.destination) for edge in self._edges}
        for source, destination in existing:
            if source in ownership and destination in ownership and ownership[source] == ownership[destination]:
                incoming.add(destination)
                outgoing.add(source)
        barriers = []

        def barrier(source, destination):
            if (source, destination) not in existing:
                existing.add((source, destination))
                barriers.append(WorkloadEdge(source, destination, "barrier"))

        for (before, before_capture), (after, after_capture) in zip(regions, regions[1:]):
            if not (before_capture or after_capture):
                continue
            first = after[0]
            if first in incoming:
                raise ValueError("capture boundary hub must be the next region's first source root")
            for source in before:
                if source not in outgoing:
                    barrier(source, first)
            for destination in after[1:]:
                if destination not in incoming:
                    barrier(first, destination)
        return (*self._edges, *barriers)

    @property
    def operator_order_by_rank(self):
        """Host forward call order, independent of legal DAG schedules."""
        if self._collective_ids:
            raise ValueError("collective traces require their joined group's call order")
        return {self.rank: tuple(self._nodes)}

    @property
    def allocations(self):
        return tuple(self._allocations.values())

    @property
    def activation_allocations(self):
        if (type(self) is ForwardTrace and self._recorded_graph is not None
                and type(self._allocations) is MappingProxyType
                and type(self._persistent_ids) is frozenset and type(self._events) is tuple
                and "_closed_storage_inputs" in self.__dict__):
            allocations, persistent, events, activations = self.__dict__["_closed_storage_inputs"]
            if self._allocations is allocations and self._persistent_ids is persistent and self._events is events:
                if "_owned_memory_storage" in self.__dict__:
                    return tuple(self._allocations[value.allocation_id] for value in activations)
                return activations
        return tuple(a for name, a in self._allocations.items() if name not in self._persistent_ids)

    @property
    def parameter_allocation_ids(self):
        """Readonly module parameters, charged by the model-weight ledger.

        These descriptors may be declared lazily while tracing a forward;
        their declaration is not a runtime parameter allocation or profile
        workspace growth. Only explicit validated parameter ports qualify.
        """
        return frozenset(self._views[name].allocation_id for node in self._nodes.values()
            if "symbolic_parameter_inputs" in node.attributes
            for name in node.attributes["symbolic_parameter_inputs"])

    @property
    def views(self):
        return tuple(self._views.values())

    @property
    def event_order(self):
        return tuple(self._events)


def _copy_capture_dicts(value):
    """Detach public mutable capture metadata; source tuples remain immutable."""
    if type(value) is dict:
        return {key: _copy_capture_dicts(item) for key, item in value.items()}
    return value


class ForwardSourceProgram:
    """Bind a recorded complete source to values from its actual source prefix.

    Bindings describe source-produced metadata fields, not a second model DAG.
    Every static operator, typed view, alias, lifetime and capture event remains
    in the complete trace. IDs are local to each graph; reusing their spelling
    never declares cross-step physical allocation or KV backing identity.
    """

    def __init__(self, source, prefix, *, node_bindings, allocation_bindings, collective_groups=None,
                 transient_allocation_bindings=(), view_bindings=(), capture_bindings=(), geometry_rebind=False):
        if type(geometry_rebind) is not bool:
            raise TypeError("source geometry rebinding requires an explicit boolean")
        self.geometry_rebind = geometry_rebind
        if not isinstance(source, ForwardTrace) or not isinstance(prefix, ForwardTrace):
            raise TypeError("source program requires actual complete and prefix traces")
        if (source.namespace != prefix.namespace or source.rank != prefix.rank
                or source._capture is not None or prefix._capture is not None
                or source._checkpoints or source._workspace_capacities
                or (source._collective_ids and collective_groups is None)):
            raise ValueError("source program requires a closed ordinary rank-local source")
        self.source, self.prefix = source, prefix
        self.rank_programs = (self,)
        self.collective_attribute_bindings = ()
        self.collective_geometry_bindings = ()
        self.node_bindings = tuple(node_bindings)
        persistent_bindings = tuple(allocation_bindings)
        transient_bindings = tuple(transient_allocation_bindings)
        self.allocation_bindings = (*persistent_bindings, *transient_bindings)
        self.view_bindings, self.capture_bindings = tuple(view_bindings), tuple(capture_bindings)
        if len({target for target, _ in self.allocation_bindings}) != len(self.allocation_bindings):
            raise ValueError("source geometry allocation targets must be unique")
        if len({target for target, _ in self.view_bindings}) != len(self.view_bindings):
            raise ValueError("source geometry view targets must be unique")
        self.transient_origins = {origin for _, origin in transient_bindings}
        self.bound_prefix_views = {origin for _, origin in self.view_bindings}
        self.capture_fields = {}
        capture_targets = set()
        capture_geometry = ("logical_tokens", "model_input_tokens", "key", "inputs", "output")
        for target, origin, fields in self.capture_bindings:
            if (type(target) is not int or type(origin) is not int
                    or (fields != ("logical_tokens",)
                        and not (geometry_rebind and fields == capture_geometry))
                    or target in capture_targets or origin in self.capture_fields
                    or not 0 <= target < len(source.source_captures)
                    or not 0 <= origin < len(prefix.source_captures)
                    or any(type(capture["logical_tokens"]) is not int
                        or type(capture["model_input_tokens"]) is not int
                        or not 0 < capture["logical_tokens"] <= capture["model_input_tokens"]
                        for capture in (source.source_captures[target], prefix.source_captures[origin]))
                    or not geometry_rebind and source.source_captures[target]["logical_tokens"] != prefix.source_captures[origin]["logical_tokens"]):
                raise ValueError("source capture binding requires explicit matching logical token extents")
            capture_targets.add(target)
            self.capture_fields[origin] = fields
        for target, origin in self.view_bindings:
            left, right = source._views[target], prefix._views[origin]
            if ((left.dtype, left.element_bytes, len(left.shape)) != (
                    right.dtype, right.element_bytes, len(right.shape))
                    or not geometry_rebind and (left.shape, left.strides_bytes, left.offset_bytes) != (
                        right.shape, right.strides_bytes, right.offset_bytes)):
                raise ValueError("initial source view binding has incompatible typed geometry")
        if collective_groups is not None:
            if (not isinstance(collective_groups, tuple) or len(collective_groups) != 2
                    or any(not isinstance(group, ForwardCollectiveGroup) for group in collective_groups)
                    or not any(source is trace for trace in collective_groups[0].traces)
                    or not any(prefix is trace for trace in collective_groups[1].traces)):
                raise ValueError("rank program requires its complete source and prefix collective groups")
            self.graph = WorkloadGraph(tuple(source._nodes.values()), source._capture_program_edges())
        else:
            self.graph = source.graph
        ignored_shapes, ignored_attributes = {}, {}
        for target, origin, shapes, attributes in self.node_bindings:
            node, current = source._nodes[target], prefix._nodes[origin]
            if not geometry_rebind and any(node.shape[name] != current.shape[name] for name in shapes):
                raise ValueError("source program initial shape binding disagrees with complete source")
            if not geometry_rebind and any(node.attributes[name] != current.attributes[name] for name in attributes):
                raise ValueError("source program initial attribute binding disagrees with complete source")
            if origin not in ignored_shapes:
                ignored_shapes[origin], ignored_attributes[origin] = set(), set()
            ignored_shapes[origin].update(shapes)
            ignored_attributes[origin].update(attributes)
        self.prefix_contract = tuple((name, node.kind, node.pool_name, node.dtype, node.variant, node.rank,
            {key: value for key, value in node.shape.items()
                if name not in ignored_shapes or key not in ignored_shapes[name]},
            {key: value for key, value in node.attributes.items()
                if name not in ignored_attributes or key not in ignored_attributes[name]})
            for name, node in prefix._nodes.items())
        self.ignored_shapes, self.ignored_attributes = ignored_shapes, ignored_attributes
        self.bound_prefix_allocations = {origin for _, origin in self.allocation_bindings}
        for target, origin in self.allocation_bindings:
            transient = origin in self.transient_origins
            if ((target in source._persistent_ids) != (not transient)
                    or (origin in prefix._persistent_ids) != (not transient)):
                raise ValueError("source geometry binding must preserve its declared persistent/transient role")
            left_allocation = self._allocation_geometry(source._allocations[target])
            right_allocation = self._allocation_geometry(prefix._allocations[origin])
            if (left_allocation[1:] if geometry_rebind else left_allocation) != (
                    right_allocation[1:] if geometry_rebind else right_allocation):
                raise ValueError("source program initial persistent allocation geometry disagrees")
            left, right = source._views[target], prefix._views[origin]
            if left.allocation_id != target or right.allocation_id != origin:
                raise ValueError("persistent program geometry requires the complete backing view")
            if ((left.dtype, left.element_bytes, len(left.shape)) != (
                    right.dtype, right.element_bytes, len(right.shape))
                    or not geometry_rebind and (left.shape, left.strides_bytes, left.offset_bytes) != (
                        right.shape, right.strides_bytes, right.offset_bytes)):
                raise ValueError("source program initial persistent view geometry disagrees")

    @staticmethod
    def _allocation_geometry(allocation):
        return (allocation.size_bytes, allocation.device_id, allocation.memory_domain,
            allocation.address_level, allocation.base_address_bytes, allocation.mapping_ref)

    def _capture_contract(self, trace):
        if not self.capture_fields:
            return trace.source_captures
        return tuple({name: value for name, value in capture.items()
            if index not in self.capture_fields or name not in self.capture_fields[index]}
            for index, capture in enumerate(trace.source_captures))

    def bind(self, prefix, *, native_frame=False):
        """Return a complete bound trace, or a cache miss for changed source structure."""
        if not isinstance(prefix, ForwardTrace):
            raise TypeError("source binding requires an actual prefix trace")
        if (prefix.namespace != self.source.namespace or prefix.rank != self.source.rank
                or tuple(prefix._nodes) != tuple(item[0] for item in self.prefix_contract)):
            return None
        if len(prefix.source_captures) != len(self.prefix.source_captures):
            return None
        for _, origin, _ in self.capture_bindings:
            capture = prefix.source_captures[origin]
            if (type(capture["logical_tokens"]) is not int or type(capture["model_input_tokens"]) is not int
                    or not 0 < capture["logical_tokens"] <= capture["model_input_tokens"]):
                return None
        if (prefix.event_order != self.prefix.event_order or self._capture_contract(prefix) != self._capture_contract(self.prefix)
                or prefix._persistent_ids != self.prefix._persistent_ids
                or prefix._references != self.prefix._references or prefix._writers != self.prefix._writers
                or prefix._readers != self.prefix._readers
                or tuple(prefix._allocations) != tuple(self.prefix._allocations)
                or tuple(prefix._views) != tuple(self.prefix._views)):
            return None
        for name, allocation in prefix._allocations.items():
            if name not in self.bound_prefix_allocations:
                if allocation != self.prefix._allocations[name]:
                    return None
            elif tuple(getattr(allocation, field) for field in ("device_id", "memory_domain", "address_level", "base_address_bytes", "mapping_ref", "allocated_event", "freed_event")) != tuple(
                    getattr(self.prefix._allocations[name], field) for field in ("device_id", "memory_domain", "address_level", "base_address_bytes", "mapping_ref", "allocated_event", "freed_event")):
                return None
        for name, view in prefix._views.items():
            if name not in self.bound_prefix_allocations and name not in self.bound_prefix_views:
                if view != self.prefix._views[name]:
                    return None
            else:
                old = self.prefix._views[name]
                if (view.allocation_id, view.offset_bytes, view.dtype, view.element_bytes, len(view.shape)) != (
                        old.allocation_id, old.offset_bytes, old.dtype, old.element_bytes, len(old.shape)):
                    return None
        for name, kind, pool, dtype, variant, rank, shape, attributes in self.prefix_contract:
            node = prefix._nodes[name]
            if (node.kind, node.pool_name, node.dtype, node.variant, node.rank) != (kind, pool, dtype, variant, rank):
                return None
            actual_shape = {key: value for key, value in node.shape.items()
                if name not in self.ignored_shapes or key not in self.ignored_shapes[name]}
            actual_attributes = {key: value for key, value in node.attributes.items()
                if name not in self.ignored_attributes or key not in self.ignored_attributes[name]}
            if actual_shape != shape or actual_attributes != attributes:
                return None
        result = copy(self.source)
        if not native_frame:
            result._nodes, result._allocations, result._views = (
                dict(self.source._nodes), dict(self.source._allocations), dict(self.source._views))
        if native_frame:
            updates = {}
            for target, origin, shapes, attributes in self.node_bindings:
                current = prefix._nodes[origin]
                shape = {name: current.shape[name] for name in shapes}
                if any(type(value) is not int or value < 0 for value in shape.values()):
                    raise ValueError("source frame requires canonical bound shape dimensions")
                updates[target] = (shape, {name: current.attributes[name] for name in attributes})
            result._nodes = _source_rank_frame(self, updates)
            if result._nodes is None: return None
        elif self.source._collective_ids:
            for target, origin, shapes, attributes in self.node_bindings:
                node, current = result._nodes[target], prefix._nodes[origin]
                result._nodes[target] = node.with_updates(
                    {name: current.shape[name] for name in shapes},
                    {name: current.attributes[name] for name in attributes})
        else:
            validated_shapes = {}
            for target, origin, shapes, attributes in self.node_bindings:
                node, current = result._nodes[target], prefix._nodes[origin]
                bound_attributes = dict(node.attributes)
                for name in attributes:
                    bound_attributes[name] = current.attributes[name]
                if shapes:
                    shape = dict(node.shape)
                    for name in shapes:
                        shape[name] = current.shape[name]
                    # Bindings often update the same homogeneous attention shape
                    # in every layer. Validate once per exact shape in this bind;
                    # node IDs, rank and source annotations remain independently
                    # bound. Noncanonical values retain the ordinary constructor.
                    canonical = (type(node) is WorkloadNode and all(
                        type(name) is str and type(value) is int
                        for name, value in shape.items()))
                    key = tuple(shape.items()) if canonical else None
                    if canonical and key in validated_shapes:
                        bound = node.with_attributes(bound_attributes)
                        object.__setattr__(bound, "shape", validated_shapes[key])
                    else:
                        bound = replace(node, shape=shape, attributes=bound_attributes)
                        if canonical:
                            validated_shapes[key] = bound.shape
                    result._nodes[target] = bound
                else:
                    result._nodes[target] = node.with_attributes(bound_attributes)
        native_storage = native_frame and _bind_native_storage(self, result, prefix._allocations, prefix._views)
        if not native_storage:
            if native_frame:
                result._allocations, result._views = dict(self.source._allocations), dict(self.source._views)
            self._bind_storage_parameters(result, prefix._allocations, prefix._views, prefix.source_captures)
        else:
            self._bind_capture_parameters(result, prefix.source_captures)
        if native_frame:
            result._recorded_graph = (result._nodes, None)
            _seal_native_bound_storage(self, result)
            if not self.source._collective_ids:
                result._recorded_graph = (result._nodes, _issue_source_frame(self, result, {}))
        else:
            graph = self.graph.with_nodes(result._nodes.values())
            result._recorded_graph = (result._nodes, graph)
        result._captures = [_copy_capture_dicts(capture) for capture in result.source_captures]
        return result


    def _bind_storage_parameters(self, result, allocations, views, captures):
        for target, origin in self.allocation_bindings:
            old = result._allocations[target]
            result._allocations[target] = allocations[origin].with_lifetime(
                target, old.allocated_event, old.freed_event)
            result._views[target] = views[origin].with_identity(target, target)
        for target, origin in self.view_bindings:
            old = result._views[target]
            result._views[target] = views[origin].with_identity(target, old.allocation_id)
        if self.view_bindings or self.transient_origins:
            for view in result._views.values():
                if region_end(view.region()) > result._allocations[view.allocation_id].size_bytes:
                    raise ValueError("bound source view exceeds its actual backing allocation")
        self._bind_capture_parameters(result, captures)

    def _bind_capture_parameters(self, result, captures):
        if self.capture_bindings:
            result._captures = [dict(capture) for capture in self.source.source_captures]
            for target, origin, fields in self.capture_bindings:
                for name in fields:
                    if name != "output":
                        result._captures[target][name] = _copy_capture_dicts(captures[origin][name])
                        continue
                    source_capture = self.source.source_captures[target]
                    output = source_capture["output"]
                    storage_id = f"rank:{result.rank}:{source_capture['key']}:hidden_states"
                    if output["storage_id"] != storage_id:
                        raise ValueError("capture geometry requires its declared terminal hidden-state port")
                    view = result._views[output["allocation_id"]]
                    allocation = result._allocations[view.allocation_id]
                    result._captures[target][name] = {**output,
                        "storage_id": f"rank:{result.rank}:{captures[origin]['key']}:hidden_states",
                        "shape": view.shape, "bytes": allocation.size_bytes}


def _source_activation_allocations(trace):
    """Private closed storage for internal reporters; public receipts own copies."""
    if (type(trace) is ForwardTrace and "_owned_memory_storage" in trace.__dict__
            and "_closed_storage_inputs" in trace.__dict__):
        allocations, persistent, events, activations = trace.__dict__["_closed_storage_inputs"]
        if trace._allocations is allocations and trace._persistent_ids is persistent and trace._events is events:
            return activations
    return trace.activation_allocations


def _seal_native_bound_storage(program, trace):
    """Own readonly closed storage; ordinary mutable source traces stay live.

    A source may legitimately edit its template after an earlier bind. Compare
    actual referents rather than IDs or hashes; changed input gets a fresh
    tuple and the storage owner's complete validation on first use.
    """
    if (type(program) is not ForwardSourceProgram or type(trace) is not ForwardTrace
            or trace._recorded_graph is None or type(trace._allocations) is not dict
            or any(type(value) is not MemoryAllocation for value in trace._allocations.values())
            or any(type(value) is not str for value in trace._persistent_ids)):
        return
    activations = tuple(value for name, value in trace._allocations.items() if name not in trace._persistent_ids)
    events = tuple(trace._events)
    if "_native_storage_tuples" in program.__dict__:
        old_activations, old_events = program.__dict__["_native_storage_tuples"]
        if (len(activations) == len(old_activations)
                and all(map(operator.is_, activations, old_activations))):
            activations = old_activations
        if len(events) == len(old_events) and all(map(operator.is_, events, old_events)):
            events = old_events
    program.__dict__["_native_storage_tuples"] = (activations, events)
    trace._allocations = MappingProxyType(trace._allocations)
    trace._persistent_ids = frozenset(trace._persistent_ids)
    trace._events = events
    trace.__dict__["_closed_storage_inputs"] = (trace._allocations, trace._persistent_ids, events, activations)


class _NativeStorageOverlay(Mapping):
    """Closed source mapping with independently bound current backing values."""

    __slots__ = ("base", "bindings", "current_values", "allocation", "preserve_backing", "cache", "values_cache")

    def __init__(self, base, bindings, values, *, allocation, preserve_backing=False):
        if type(preserve_backing) is not bool:
            raise TypeError("native view backing preservation requires an explicit boolean")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "bindings", bindings)
        # Current inputs cross this closed-frame boundary by value. Keeping
        # only declared origins also avoids retaining unrelated prefix ports.
        object.__setattr__(self, "current_values", MappingProxyType({
            name: replace(values[name]) for name in dict.fromkeys(bindings.values())}))
        object.__setattr__(self, "allocation", allocation)
        object.__setattr__(self, "preserve_backing", preserve_backing)
        object.__setattr__(self, "cache", {})
        object.__setattr__(self, "values_cache", None)

    def __setattr__(self, name, value):
        raise TypeError("bound source storage is immutable")

    def __delattr__(self, name):
        raise TypeError("bound source storage is immutable")

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        return iter(self.base)

    def values(self):
        if self.values_cache is None:
            object.__setattr__(self, "values_cache", tuple(self[name] for name in self.base))
        return self.values_cache

    def __getitem__(self, name):
        if name not in self.cache:
            if name not in self.bindings:
                value = replace(self.base[name])
            else:
                value = self.current_values[self.bindings[name]]
                if self.allocation:
                    old = self.base[name]
                    value = value.with_lifetime(name, old.allocated_event, old.freed_event)
                else:
                    value = value.with_identity(name,
                        self.base[name].allocation_id if self.preserve_backing else name)
            self.cache[name] = value
        return self.cache[name]


def _source_view_layouts(trace):
    """Read current view geometry without reconstructing immutable static views.

    This projection exposes layout facts only. Callers that consume a port,
    allocation lifetime or public memory receipt still use the actual frame.
    """
    owned = trace.__dict__["_owned_memory_storage"] if "_owned_memory_storage" in trace.__dict__ else None
    if (type(trace) is ForwardTrace and type(owned) is tuple and len(owned) == 6
            and trace._allocations is owned[0] and trace._views is owned[1]
            and trace._persistent_ids is owned[4] and trace._events is owned[5]
            and type(owned[1]) is MappingProxyType
            and type(owned[3]) is _NativeStorageOverlay and owned[3].allocation is False
            and type(owned[3].base) is MappingProxyType):
        overlay = owned[3]
        return (overlay[name] if name in overlay.bindings else view
            for name, view in overlay.base.items())
    return trace._views.values()


def _bind_native_storage(program, trace, allocations, views):
    """Reuse detached storage after comparing the actual source template."""
    source = program.source
    dynamic = program.geometry_rebind
    if not dynamic and (program.view_bindings or program.capture_bindings or program.transient_origins):
        return False
    mutable_source = (type(source._allocations) is dict and type(source._views) is dict
        and type(source._persistent_ids) is set and type(source._events) is list)
    sealed_source = False
    if not mutable_source:
        owned = source.__dict__["_owned_memory_storage"] if "_owned_memory_storage" in source.__dict__ else None
        sealed_source = (not dynamic and type(owned) is tuple and len(owned) == 6
            and type(source._allocations) is MappingProxyType and type(source._views) is MappingProxyType
            and type(source._persistent_ids) is frozenset and type(source._events) is tuple
            and source._allocations is owned[0] and source._views is owned[1]
            and type(owned[2]) is _NativeStorageOverlay and owned[2].allocation is True
            and type(owned[3]) is _NativeStorageOverlay and owned[3].allocation is False
            and source._persistent_ids is owned[4] and source._events is owned[5])
        if not sealed_source:
            return False
    for _, origin in program.allocation_bindings:
        if type(allocations[origin]) is not MemoryAllocation or type(views[origin]) is not TensorView:
            return False
    for _, origin in program.view_bindings:
        if type(views[origin]) is not TensorView:
            return False
    snapshot = program.__dict__["_native_storage_template"] if "_native_storage_template" in program.__dict__ else None
    if (snapshot is None or tuple(source._allocations) != snapshot[8] or tuple(source._views) != snapshot[9]
            or not all(map(operator.is_, source._allocations.values(), snapshot[0].values()))
            or not all(map(operator.is_, source._views.values(), snapshot[1].values()))
            or source._persistent_ids != snapshot[2]
            or source._events != (snapshot[6] if sealed_source else snapshot[3])
            or program.allocation_bindings is not snapshot[11]
            or program.view_bindings is not snapshot[12] or program.capture_bindings is not snapshot[13]
            or program.transient_origins != snapshot[14] or dynamic is not snapshot[15]):
        if (any(type(value) is not MemoryAllocation for value in source._allocations.values())
                or any(type(value) is not TensorView for value in source._views.values())
                or any(type(name) is not str for name in source._persistent_ids)
                or not dynamic and any(target not in source._persistent_ids for target, _ in program.allocation_bindings)):
            return False
        allocation_base, view_base = dict(source._allocations), dict(source._views)
        owned_allocations = {name: replace(value) for name, value in allocation_base.items()}
        owned_views = {name: replace(value) for name, value in view_base.items()}
        persistent, events = frozenset(source._persistent_ids), list(source._events)
        activations = tuple(value for name, value in owned_allocations.items() if name not in persistent)
        view_bindings = dict(program.allocation_bindings)
        view_bindings.update(program.view_bindings)
        snapshot = (allocation_base, view_base, persistent, events,
            MappingProxyType(owned_allocations), MappingProxyType(owned_views), tuple(events), activations,
            tuple(allocation_base), tuple(view_base), MappingProxyType(dict(program.allocation_bindings)),
            program.allocation_bindings, program.view_bindings, program.capture_bindings,
            frozenset(program.transient_origins), dynamic, MappingProxyType(view_bindings))
        program.__dict__["_native_storage_template"] = snapshot
    if dynamic:
        bound_allocations = dict(program.allocation_bindings)
        for target, origin in program.view_bindings:
            target_root = source._views[target].allocation_id
            allocation = (allocations[bound_allocations[target_root]] if target_root in bound_allocations
                else source._allocations[target_root])
            if region_end(views[origin].region()) > allocation.size_bytes:
                raise ValueError("bound source view exceeds its actual backing allocation")
    allocation_overlay = _NativeStorageOverlay(snapshot[4], snapshot[10], allocations, allocation=True)
    view_overlay = _NativeStorageOverlay(snapshot[5], snapshot[16], views,
        allocation=False, preserve_backing=True)
    trace._allocations = MappingProxyType(allocation_overlay)
    trace._views = MappingProxyType(view_overlay)
    trace._persistent_ids, trace._events = snapshot[2], snapshot[6]
    trace.__dict__["_owned_memory_storage"] = (trace._allocations, trace._views,
        allocation_overlay, view_overlay, trace._persistent_ids, trace._events)
    if not program.transient_origins:
        trace.__dict__["_closed_storage_inputs"] = (trace._allocations, trace._persistent_ids, trace._events, snapshot[7])
    return True


class SourceProgramCache:
    """Bounded run-local source programs; eviction resumes actual source execution."""

    def __init__(self, max_entries=128, *, glm_mtp_structural_reuse=False,
                 glm_ordinary_structural_reuse=True):
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("source program cache capacity must be a positive integer")
        if type(glm_mtp_structural_reuse) is not bool:
            raise TypeError("GLM MTP structural reuse must be an explicit boolean")
        if type(glm_ordinary_structural_reuse) is not bool:
            raise TypeError("ordinary GLM structural reuse must be an explicit boolean")
        self.glm_mtp_structural_reuse = glm_mtp_structural_reuse
        self.glm_ordinary_structural_reuse = glm_ordinary_structural_reuse
        self.max_entries = max_entries
        self.programs = OrderedDict()
        self.next_program_id = 0
        # Live DSpark recurrence is run-owned state, separate from evictable
        # recorded templates. Losing it cannot restart a request with new KV.
        self.dspark_sources = {}
        # Legacy dummy calls read the last prepared input buffer. These value
        # declarations live with the run, independently of template eviction.
        self.qwen_legacy_inputs = {}
        self.glm_legacy_inputs = {}
        # MTP dummy_run retains the real proposer's indexer toggle. This is
        # structural run history, independent of evictable graph templates.
        self.glm_mtp_legacy_states = {}

    def namespace(self, model_type, pool, rank, source_version, *, cache_key=None):
        if cache_key is None:
            identity = self.next_program_id
            self.next_program_id += 1
        else:
            identity = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()
        return f"source-program:{model_type}:{pool}:rank-{rank}:{source_version}:{identity}"

    def lookup(self, key):
        if key not in self.programs:
            return None
        self.programs.move_to_end(key)
        return self.programs[key]

    def store(self, key, program):
        if not isinstance(program, ForwardSourceProgram):
            raise TypeError("source cache entries must be actual recorded source programs")
        self._retain(key, program)

    def store_module_frame(self, key, frame):
        if frame is not None and (type(frame) is not _ForwardModuleFrame or not frame.supported):
            raise TypeError("source fragments require an actual supported recorded module body")
        self._retain(key, frame)

    def _retain(self, key, program):
        self.programs[key] = program
        self.programs.move_to_end(key)
        while len(self.programs) > self.max_entries:
            self.programs.popitem(last=False)

class ForwardWorkspace:
    """One explicitly initialized, aligned source workspace lane.

    Mirrors vLLM0.30 WorkspaceManager.get_simultaneous/_ensure_workspace_size.
    Views share a UINT8 arena. Growth drops the manager reference before new
    allocation; caller-retained views alone determine whether old storage
    overlaps. This models source live allocations, not allocator reserved
    bytes, asynchronous lanes, a warm runtime history or an installed GPU.
    """

    def __init__(self, trace, name, *, alignment_bytes, initial_capacity_bytes, locked):
        if not isinstance(trace, ForwardTrace) or not isinstance(name, str) or not name:
            raise TypeError("workspace requires a forward trace and nonempty source name")
        if (type(alignment_bytes) is not int or alignment_bytes <= 0
                or type(initial_capacity_bytes) is not int or initial_capacity_bytes < 0
                or type(locked) is not bool):
            raise ValueError("workspace requires explicit positive alignment, nonnegative initial size and lock state")
        self.trace, self.name = trace, name
        if name in trace._workspace_capacities:
            raise ValueError("source workspace names must be unique within a rank forward")
        self.alignment_bytes, self.locked = alignment_bytes, locked
        self._current = (None if initial_capacity_bytes == 0 else
            trace.tensor(trace.unique_name(name), (initial_capacity_bytes,), "uint8"))
        trace._workspace_capacities[name] = self.capacity_bytes

    @property
    def capacity_bytes(self):
        return 0 if self._current is None else self._current.shape[0]

    def get_simultaneous(self, *shapes_and_dtypes):
        if not shapes_and_dtypes:
            raise ValueError("workspace needs at least one nonempty typed layout")
        layouts, offsets, total = [], [], 0
        for item in shapes_and_dtypes:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("workspace layout must be a shape/dtype pair")
            shape, dtype = item
            layout = self.trace._dense_view("layout", shape, dtype)
            if self.alignment_bytes % layout.element_bytes:
                raise ValueError("workspace alignment must align every requested dtype")
            layouts.append(layout)
            offsets.append(total)
            size = region_end(layout.region())
            total += -(-size // self.alignment_bytes) * self.alignment_bytes
        if total > self.capacity_bytes:
            if self.locked:
                raise ValueError("locked source workspace cannot grow")
            self._current = None
            self._current = self.trace.tensor(self.trace.unique_name(self.name), (total,), "uint8")
            self.trace._workspace_capacities[self.name] = self.capacity_bytes
        return tuple(self.trace.alias(self._current, self.trace.unique_name(self.name + ":view"),
            shape=layout.shape, strides_bytes=layout.strides_bytes, offset_bytes=offset, dtype=layout.dtype)
            for layout, offset in zip(layouts, offsets, strict=True))


class ForwardPipelineGroup:
    """Synchronous PP transport between independent physical rank traces.

    Source scopes retain send buffers; receives allocate on the destination.
    This records data/control edges, not transport metadata, async retirement
    or a distributed runtime. Equal TP stages use flattened lane views,
    full receiver backings and separately traced destination all-gathers.
    """

    def __init__(self, traces, *, namespace, collective_groups=(), tensor_roles=("hidden_states", "residual"),
                 stage_tensor_roles=None):
        if (not isinstance(traces, tuple) or len(traces) < 2
                or any(not isinstance(trace, ForwardTrace) or trace.namespace is None for trace in traces)
                or len({trace.rank for trace in traces}) != len(traces)
                or len({trace.namespace for trace in traces}) != len(traces)
                or not isinstance(namespace, str) or not namespace):
            raise ValueError("pipeline tracing requires distinct, namespaced stage ranks")
        auxiliary_roles = ("hidden_states", "aux_hidden_states_0", "aux_hidden_states_1", "aux_hidden_states_2")
        if tensor_roles not in (("hidden_states", "residual"), ("hidden_states",), auxiliary_roles):
            raise ValueError("pipeline tensor roles must match the source intermediate dictionary")
        self.traces, self.namespace, self.tensor_roles = traces, namespace, tensor_roles
        if collective_groups:
            if (not isinstance(collective_groups, tuple) or len(collective_groups) < 2
                    or any(not isinstance(group, ForwardCollectiveGroup) or len(group.traces) < 2
                        for group in collective_groups)
                    or len({len(group.traces) for group in collective_groups}) != 1
                    or tuple(trace for group in collective_groups for trace in group.traces) != traces):
                raise ValueError("sliced pipeline requires ordered equal multi-lane stage groups")
        self.collective_groups = collective_groups
        stage_count = len(traces) // (len(collective_groups[0].traces) if collective_groups else 1)
        if tensor_roles == auxiliary_roles:
            if (type(stage_tensor_roles) is not tuple or len(stage_tensor_roles) != stage_count
                    or any(type(roles) is not tuple or not roles or roles[0] != "hidden_states"
                        or len(set(roles)) != len(roles) or not set(roles) <= set(auxiliary_roles)
                        or tuple(sorted(roles[1:])) != auxiliary_roles[1:len(roles)] for roles in stage_tensor_roles)
                    or set(stage_tensor_roles[-1]) != set(auxiliary_roles)
                    or any(not set(before) <= set(after) for before, after in zip(stage_tensor_roles, stage_tensor_roles[1:]))):
                raise ValueError("auxiliary PP requires every stage's exact ordered tensor roles")
        elif stage_tensor_roles is not None:
            raise ValueError("ordinary PP tensor roles cannot carry auxiliary stage declarations")
        self.stage_tensor_roles = stage_tensor_roles
        self._previous = {trace.rank: None for trace in traces}

    def send(self, tensor, *, config, pool_name, tensor_name, frontier, out=None):
        if not isinstance(config, PipelineP2PConfig) or tensor_name not in self.tensor_roles:
            raise ValueError("pipeline transport requires canonical config and intermediate tensor role")
        tp = len(self.collective_groups[0].traces) if self.collective_groups else 1
        if not 0 <= config.source_stage < config.destination_stage < len(self.traces) // tp:
            raise ValueError("pipeline stages are outside the declared rank placement")
        if tp == 1:
            source, destination = self.traces[config.source_stage], self.traces[config.destination_stage]
            lane = 0
        else:
            source_group = self.collective_groups[config.source_stage]
            if not any(tensor.trace is trace for trace in source_group.traces):
                raise ValueError("pipeline slice belongs to another source stage")
            lane = source_group.traces.index(tensor.trace)
            source = source_group.traces[lane]
            destination = self.collective_groups[config.destination_stage].traces[lane]
        source._check_tensor(tensor)
        source._check_tensor(frontier)
        expanded = self.tensor_roles == ("hidden_states",) or self.stage_tensor_roles is not None
        full_rank = 3 if expanded and tensor_name == "hidden_states" else 2
        if tensor.dtype not in ("bf16", "fp16") or len(tensor.shape) != (full_rank if tp == 1 else 1) or not tensor.is_contiguous:
            raise NotImplementedError("pipeline requires a contiguous 16-bit full tensor or TP lane")
        if (tp == 1 and out is not None) or (tp > 1 and out is None):
            raise ValueError("sliced PP requires an explicit full-backing destination view")
        suffix = "" if tp == 1 else f":lane-{lane}"
        boundary = f"{self.namespace}:pp-{config.source_stage}-to-{config.destination_stage}"
        name = f"{boundary}:{tensor_name}{suffix}"
        if name in source._nodes:
            raise ValueError("pipeline tensor has already been sent across this boundary")
        roles = (self.tensor_roles if self.stage_tensor_roles is None else self.stage_tensor_roles[config.source_stage])
        if tensor_name not in roles:
            raise ValueError("pipeline tensor is not present at this stage's source boundary")
        index = roles.index(tensor_name)
        expected_previous = None if index == 0 else f"{boundary}:{roles[index - 1]}{suffix}"
        if self._previous[source.rank] != expected_previous:
            raise ValueError("pipeline sends must follow hidden_states then residual order" if self.stage_tensor_roles is None else
                "pipeline sends must follow the declared source tensor-dictionary order")
        writer = source._writers[tensor.view.allocation_id]
        ready = source._writers[frontier.view.allocation_id]
        if writer is None or ready is None:
            raise ValueError("pipeline transport requires produced data and a completed stage frontier")
        if out is None:
            received = destination.tensor("receive:" + name, tensor.shape, tensor.dtype)
        else:
            destination._check_tensor(out)
            if out.shape != tensor.shape or out.dtype != tensor.dtype or not out.is_contiguous:
                raise ValueError("pipeline receive view must match the transmitted lane")
            received = out
        size = prod(tensor.shape) * tensor.view.element_bytes
        source._nodes[name] = WorkloadNode(name, WorkloadKind.P2P, pool_name, {}, rank=source.rank,
            attributes={"algorithm": config.algorithm, "bytes": size, "bytes_semantics": "payload",
                "backend": config.backend, "backend_config_sha256": config.backend_config_sha256,
                "direction": "point_to_point", "link": config.link, "topology": config.topology,
                "participants": (source.rank, destination.rank), "operation": "send_recv",
                "source_rank": source.rank, "destination_rank": destination.rank,
                "ring_order": (), "reduction": None, "resource_id": config.resource_id,
                "source_stage": config.source_stage, "destination_stage": config.destination_stage,
                "payload_tensor": {"name": tensor_name, "shape": tensor.shape, "dtype": tensor.dtype, "bytes": size},
                "symbolic_tensor_inputs": (tensor.view.view_id,),
                "symbolic_tensor_outputs": (received.view.view_id,),
                "symbolic_tensor_output_allocations": (received.view.allocation_id,),
                **({"symbolic_pipeline_views": {
                    "source": asdict(tensor.view), "destination": asdict(received.view),
                    "source_backing_bytes": source._allocations[tensor.view.allocation_id].size_bytes,
                    "destination_backing_bytes": destination._allocations[received.view.allocation_id].size_bytes,
                    "source_producer_views": tuple(asdict(source._views[view_id])
                        for view_id in source._nodes[writer].attributes["symbolic_tensor_outputs"])}})})
        source._edges.append(WorkloadEdge(writer, name))
        previous = self._previous[source.rank]
        for dependency in (ready, previous):
            if dependency is not None and dependency != writer:
                edge = WorkloadEdge(dependency, name, "barrier")
                if edge not in source._edges:
                    source._edges.append(edge)
        source._readers[tensor.view.allocation_id].add(name)
        destination._writers[received.view.allocation_id] = name
        self._previous[source.rank] = name
        source._event("send:" + name)
        destination._event("receive:" + name)
        return received

    @property
    def operator_order_by_rank(self):
        return {rank: order for source in (self.collective_groups or self.traces)
            for rank, order in source.operator_order_by_rank.items()}

    @property
    def graph(self):
        if self.collective_groups:
            parts = tuple(group._joined_parts() for group in self.collective_groups)
            graph = WorkloadGraph(tuple(node for nodes, _ in parts for node in nodes),
                tuple(edge for _, edges in parts for edge in edges))
            graph.topological_order()
            return graph
        if any(trace._collective_ids for trace in self.traces):
            raise NotImplementedError("full-tensor PP group does not join TP collectives")
        graph = WorkloadGraph(tuple(node for trace in self.traces for node in trace._nodes.values()),
            tuple(edge for trace in self.traces for edge in trace._capture_program_edges()))
        graph.topological_order()
        return graph


class ForwardActivationGroup:
    """Blocking full hidden/residual exchanges for the existing AF contract.

    This is the source projection of the configured AF service, not a claim
    that upstream vanilla vLLM implements its transport/provider. Each receive
    has independent storage; Python forward scopes determine its lifetime.
    """

    def __init__(self, deployment, pool_traces, *, collective_groups=(), graph_token_shape=None, graph_eager_kinds=()):
        if (deployment.mode not in ("af_disagg", "pd_af_disagg") or len(pool_traces) != 2
                or not set(pool_traces) <= set(deployment.worker_pools)
                or (deployment.mode == "af_disagg" and set(pool_traces) != set(deployment.worker_pools))):
            raise ValueError("activation tracing requires the two configured AF pools")
        selected = tuple(deployment.worker_pools[name] for name in pool_traces)
        phases = {worker_role_phase(pool.role) for pool in selected}
        if ({worker_role_operator_group(pool.role) for pool in selected} != {"attention", "ffn"}
                or len(phases) != 1
                or (deployment.mode == "af_disagg" and phases != {"colocated"})
                or (deployment.mode == "pd_af_disagg" and not phases <= {"prefill", "decode"})):
            raise ValueError("activation tracing requires an Attention/FFN pair in the same phase")
        for name, traces in pool_traces.items():
            if (not isinstance(traces, tuple) or tuple(trace.rank for trace in traces) != deployment.worker_pools[name].ranks
                    or any(not isinstance(trace, ForwardTrace) or trace.namespace is None for trace in traces)):
                raise ValueError("activation tracing requires each pool's ordered physical rank traces")
        self.traces = tuple(trace for traces in pool_traces.values() for trace in traces)
        if len({trace.rank for trace in self.traces}) != len(self.traces):
            raise ValueError("activation tracing requires independent physical workers")
        if collective_groups and (not isinstance(collective_groups, tuple)
                or {trace.rank for group in collective_groups for trace in group.traces} != {trace.rank for trace in self.traces}):
            raise ValueError("activation collective groups must cover the declared workers")
        self.deployment, self.pool_traces, self.collective_groups = deployment, pool_traces, collective_groups
        self.graph_token_shape, self.graph_eager_kinds = graph_token_shape, graph_eager_kinds
        self.attention_capture_inputs = {}

    def begin_local_capture(self, trace, inputs, *, boundary):
        """A single worker's compute segment; AF sends remain outside it."""
        if self.graph_token_shape is None:
            return
        shape = self.graph_token_shape
        if trace.rank in self.attention_capture_inputs:
            inputs = {**inputs, **self.attention_capture_inputs[trace.rank]}
        trace.begin_capture(key=f"midlayer_af:{boundary}:{shape.runtime_mode}:{shape.model_input_tokens}",
            logical_tokens=shape.logical_tokens, model_input_tokens=shape.model_input_tokens,
            inputs=inputs, runtime_mode=shape.runtime_mode, eager_kinds=self.graph_eager_kinds,
            source_stage="midlayer_af")
        trace._capture["execution_scope"] = "midlayer_af_service_compute_segment_not_upstream_native_runtime"

    def end_local_capture(self, trace, outputs):
        if self.graph_token_shape is not None:
            trace.end_capture(outputs=outputs)

    def exchange(self, hidden_states, residual, *, source_pool, destination_pool, boundary):
        """Carry both tensors on matching TP lanes without a TP reshard."""
        if (source_pool == destination_pool or source_pool not in self.pool_traces
                or destination_pool not in self.pool_traces or not isinstance(boundary, str) or not boundary):
            raise ValueError("activation exchange requires a named cross-pool boundary")
        traces = self.pool_traces[source_pool]
        if not any(hidden_states.trace is trace for trace in traces):
            raise ValueError("activation tensor must belong to the sending pool")
        lane = traces.index(hidden_states.trace)
        source = traces[lane]
        destination = self.pool_traces[destination_pool][lane]
        source._check_tensor(hidden_states)
        source._check_tensor(residual)
        if (hidden_states.shape != residual.shape or hidden_states.dtype != residual.dtype
                or hidden_states.dtype != "bf16" or len(hidden_states.shape) != 2
                or not hidden_states.is_contiguous or not residual.is_contiguous):
            raise NotImplementedError("activation exchange requires equal contiguous full BF16 tensors")
        routes = tuple(route for route in self.deployment.routes if route.payload == "activation"
            and route.source_pool == source_pool and route.destination_pool == destination_pool)
        if len(routes) != 1:
            raise ValueError("activation exchange requires exactly one configured route")
        route = routes[0]
        received, previous = {}, None
        for role, tensor in (("hidden_states", hidden_states), ("residual", residual)):
            name = source.qualify(f"{boundary}:{role}:transfer")
            if name in source._nodes:
                raise ValueError("activation boundary cannot be emitted twice on one lane")
            # A model may return a same-layout view of its last output. Bind
            # the send to that exact producer region without inventing a copy
            # or relaxing the lowering boundary to accept unrelated aliases.
            port = source._complete_producer_view(tensor)
            writer = source._writers[tensor.view.allocation_id]
            output = destination.tensor("receive:" + name, tensor.shape, tensor.dtype)
            size = prod(tensor.shape) * tensor.view.element_bytes
            source._nodes[name] = WorkloadNode(name, WorkloadKind.AF_ACTIVATION_TRANSFER, source_pool, {}, rank=source.rank,
                attributes={"algorithm": route.algorithm, "bytes": size, "bytes_semantics": "payload",
                    "backend": route.backend, "backend_config_sha256": route.backend_config_sha256,
                    "direction": "point_to_point", "link": route.link, "topology": route.topology,
                    "participants": (source.rank, destination.rank), "operation": "send_recv",
                    "source_rank": source.rank, "destination_rank": destination.rank,
                    "ring_order": (), "reduction": None, "resource_id": route.resource_id,
                    "destination_pool": destination_pool,
                    "payload_tensor": {"name": role, "shape": tensor.shape, "dtype": tensor.dtype, "bytes": size},
                    "symbolic_tensor_inputs": (port.view_id,), "symbolic_tensor_outputs": (output.view.view_id,),
                    "symbolic_af_boundary": boundary})
            source._edges.append(WorkloadEdge(writer, name, "ready"))
            if previous is not None:
                source._edges.append(WorkloadEdge(previous, name, "barrier"))
            source._readers[tensor.view.allocation_id].add(name)
            destination._writers[output.view.allocation_id] = name
            source._event("send:" + name)
            destination._event("receive:" + name)
            received[role] = output
            previous = name
        return received["hidden_states"], received["residual"]

    @property
    def operator_order_by_rank(self):
        return {rank: order for source in (self.collective_groups or self.traces)
            for rank, order in source.operator_order_by_rank.items()}

    @property
    def graph(self):
        parts = (tuple(group._joined_parts() for group in self.collective_groups) if self.collective_groups else
            tuple((tuple(trace._nodes.values()), trace._capture_program_edges()) for trace in self.traces))
        graph = WorkloadGraph(tuple(node for nodes, _ in parts for node in nodes),
            tuple(edge for _, edges in parts for edge in edges))
        graph.topological_order()
        return graph


class ForwardCollectiveGroup:
    """Join synchronous virtual TP calls without retaining tensor handles.

    Each rank runs its source forward independently. Local collective calls
    update that rank's storage writers; joining maps them to one communication
    node with all producer/consumer dependencies. Rank execution order in
    Python is never a cross-device storage lifetime order. Collective scratch,
    asynchronous retirement and actual provider selection are not inferred.
    """

    def __init__(self, traces, *, namespace, participant_nic_ids=(), ring_order=(), subgroups=()):
        if (not isinstance(traces, tuple) or len(traces) < 2
                or any(not isinstance(trace, ForwardTrace) for trace in traces)):
            raise ValueError("collective tracing requires at least two rank traces")
        if len({trace.rank for trace in traces}) != len(traces):
            raise ValueError("collective traces must have unique physical ranks")
        if (any(trace.namespace is None for trace in traces)
                or len({trace.namespace for trace in traces}) != len(traces)):
            raise ValueError("collective traces require distinct rank namespaces")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("collective group requires a nonempty namespace")
        if (not isinstance(subgroups, tuple) or any(not isinstance(group, ForwardCollectiveGroup)
                or group.subgroups or not set(group.traces) <= set(traces) for group in subgroups)
                or len({group.namespace for group in subgroups}) != len(subgroups)
                or any(group.namespace == namespace for group in subgroups)):
            raise ValueError("collective subgroups must be distinct direct groups over these actual rank traces")
        self.subgroups = subgroups
        self.traces = traces
        self.namespace = namespace
        self.participants = tuple(trace.rank for trace in traces)
        if (not isinstance(participant_nic_ids, tuple)
                or (participant_nic_ids and (len(participant_nic_ids) != len(traces)
                    or any(not isinstance(nic, str) or not nic or nic != nic.strip() for nic in participant_nic_ids)))):
            raise ValueError("collective NIC mapping requires an explicit immutable physical ID per rank")
        self.participant_nic_ids = participant_nic_ids
        if (not isinstance(ring_order, tuple) or ring_order and (
                len(ring_order) != len(self.participants)
                or any(type(rank) is not int for rank in ring_order)
                or set(ring_order) != set(self.participants))):
            raise ValueError("collective ring order must be an immutable permutation of physical participants")
        self.ring_order = self.participants if not ring_order else ring_order
        self._explicit_ring_order = bool(ring_order)
        self._sequences = {rank: [] for rank in self.participants}
        self._contracts = {}
        self._layouts = {}
        self._submissions = {}
        self._recorded_graph = None

    def call(self, name, tensor, *, pool_name, config, inplace, dim=None, moe_invocation_id=None, output_shape=None,
             implementation=None, residual=None, rmsnorm_weight=None, rmsnorm_epsilon=None, group_dimension="tensor"):
        """All-reduce (fresh or in-place) or fresh, axis-expanding all-gather.

        Input bytes are per-rank logical bytes, as in PhysicalTransferConfig.
        Layout, dtype, storage mode and call order must agree on every rank.
        Invocation IDs are derived from module calls, not a model DAG table.
        Optional output_shape records the source's dense post-gather view;
        it allocates no additional backing and remains the public output port.
        """
        if self._recorded_graph is not None:
            raise ValueError("a bound collective source program is a closed forward trace")
        if not isinstance(name, str) or not name:
            raise ValueError("symbolic collective requires a nonempty module name")
        if not isinstance(tensor, ForwardTensor) or not any(tensor.trace is trace for trace in self.traces):
            raise ValueError("symbolic collective input belongs to another rank group")
        if not isinstance(config, CollectiveConfig) or type(inplace) is not bool:
            raise TypeError("symbolic collective requires canonical config and explicit storage mode")
        if group_dimension not in ("tensor", "context") or (group_dimension == "context" and config.operation != "all_gather"):
            raise ValueError("ordinary source collectives require TP or an explicit PCP all-gather")
        if config.link == "roce" and not self.participant_nic_ids:
            raise ValueError("RoCE collective requires explicit shared physical NIC placement")
        if self._explicit_ring_order and (config.algorithm != "ring" or config.backend != "analytical_roofline_v1"):
            raise ValueError("explicit virtual ring order requires the analytical ring backend")
        if moe_invocation_id is not None and (not isinstance(moe_invocation_id, str) or not moe_invocation_id
                or config.operation != "all_reduce"):
            raise ValueError("MoE collective requires its explicit invocation and all-reduce")
        if not tensor.is_contiguous:
            raise NotImplementedError("symbolic collective requires a contiguous source input")
        if implementation not in (None, "sglang_custom_all_reduce_v2", "sglang_flashinfer_allreduce_residual_rmsnorm"):
            raise ValueError("collective implementation requires its existing source contract")
        fused_norm = implementation == "sglang_flashinfer_allreduce_residual_rmsnorm"
        if implementation is not None and (config.operation != "all_reduce" or config.link != "nvlink"):
            raise ValueError("SGLang collective implementation requires an NVLink all-reduce")
        if fused_norm:
            if (inplace or dim is not None or output_shape is not None or tensor.dtype != "bf16" or len(tensor.shape) != 2
                    or not isinstance(residual, ForwardTensor) or not isinstance(rmsnorm_weight, ForwardTensor)):
                raise ValueError("fused all-reduce/RMSNorm requires BF16 rank2 input, residual and weight with fresh outputs")
            tensor.trace._check_tensor(residual)
            tensor.trace._check_tensor(rmsnorm_weight)
            if (residual.shape != tensor.shape or residual.dtype != tensor.dtype or not residual.is_contiguous
                    or rmsnorm_weight.shape != (tensor.shape[-1],) or rmsnorm_weight.dtype != tensor.dtype
                    or not rmsnorm_weight.is_contiguous or isinstance(rmsnorm_epsilon, bool)
                    or not isinstance(rmsnorm_epsilon, (int, float)) or not isfinite(rmsnorm_epsilon) or rmsnorm_epsilon <= 0):
                raise ValueError("fused all-reduce/RMSNorm requires matching contiguous source ports and positive epsilon")
        elif residual is not None or rmsnorm_weight is not None or rmsnorm_epsilon is not None:
            raise ValueError("residual/RMSNorm ports require the explicit fused collective implementation")
        shape = tensor.shape
        if config.operation == "all_reduce":
            if dim is not None:
                raise ValueError("all-reduce has no gather dimension")
        elif config.operation == "all_gather":
            if inplace or type(dim) is not int or not -len(shape) <= dim < len(shape):
                raise ValueError("all-gather requires fresh output and an explicit valid dimension")
            dim %= len(shape)
            shape = tuple(size * len(self.traces) if axis == dim else size
                for axis, size in enumerate(shape))
        else:
            raise NotImplementedError("symbolic TP group supports only all-reduce and all-gather")
        if output_shape is not None and (config.operation != "all_gather"
                or not isinstance(output_shape, tuple) or not output_shape
                or any(type(size) is not int or size <= 0 for size in output_shape)
                or prod(output_shape) != prod(shape)):
            raise ValueError("all-gather output reshape must preserve its fresh dense allocation")
        rank = tensor.trace.rank
        sequence = self._sequences[rank]
        invocation = sequence.count(name)
        identifier = self.namespace + ":" + name + (f":{invocation}" if invocation else "")
        contract = WorkloadNode(identifier, WorkloadKind.COLLECTIVE, pool_name, {},
            dtype=tensor.dtype, rank=self.participants[0], attributes={
                "algorithm": config.algorithm, "backend": config.backend,
                "backend_config_sha256": config.backend_config_sha256,
                "bytes": prod(tensor.shape) * tensor.view.element_bytes,
                "bytes_semantics": "per_rank_input", "direction": "collective",
                "link": config.link, "topology": config.topology,
                "participants": self.participants, "operation": config.operation,
                **({"collective_group": group_dimension} if group_dimension != "tensor" else {}),
                **({"participant_nic_ids": self.participant_nic_ids} if config.link == "roce" else {}),
                "source_rank": None, "destination_rank": None,
                "ring_order": self.ring_order if config.algorithm == "ring" else (),
                "reduction": "sum" if config.operation == "all_reduce" else None,
                "resource_id": config.resource_id,
                "symbolic_collective_storage": "inplace" if inplace else "fresh",
                "symbolic_collective_input_shape": tensor.shape,
                "symbolic_collective_dim": dim,
                **({"implementation": implementation} if implementation is not None else {}),
                **({"rmsnorm_epsilon": rmsnorm_epsilon, "symbolic_workspace_complete": False,
                    "symbolic_fused_outputs": ("normalized", "residual"), "symbolic_inputs_mutated": False}
                    if fused_norm else {}),
                **({"moe_invocation_id": moe_invocation_id} if moe_invocation_id is not None else {}),
            })
        layout = (tensor.shape, tensor.dtype, shape, output_shape,
            (residual.shape, residual.dtype, rmsnorm_weight.shape, rmsnorm_weight.dtype) if fused_norm else None)
        for other in self._sequences.values():
            if len(other) > len(sequence) and other[len(sequence)] != name:
                raise ValueError("symbolic collective order disagrees across ranks")
        if identifier in self._contracts:
            if contract != self._contracts[identifier] or layout != self._layouts[identifier]:
                raise ValueError("symbolic collective contract/layout disagrees across ranks")
            if rank in self._submissions[identifier]:
                raise ValueError("symbolic collective invocation collides with another module name")
        local = replace(contract, rank=rank)
        trace = tensor.trace
        input_producer = trace._writers[tensor.view.allocation_id]
        external_origin = trace._legacy_dummy_PP_origin(tensor)
        if external_origin is not None and (config.operation != "all_reduce" or inplace or dim is not None
                or implementation is not None or group_dimension != "tensor" or external_origin["role"] != "hidden_states"):
            raise ValueError("legacy dummy external collective input is confined to fresh stage-entry TP all-reduce")
        producer_port = tensor.view if external_origin is not None else None
        if moe_invocation_id is not None and external_origin is None:
            producer = trace._nodes[input_producer] if input_producer is not None else None
            if producer is not None and "symbolic_glm_moe_sp_restore" in producer.attributes:
                proof = producer.attributes["symbolic_glm_moe_sp_restore"]
                if set(proof) != {"logical_rows", "gathered_rows", "tensor_parallel_size"}:
                    raise ValueError("GLM MoE SP restore requires its complete physical prefix geometry")
                rows, padded, lanes = (proof[key] for key in ("logical_rows", "gathered_rows", "tensor_parallel_size"))
                ports = tuple(trace._views[name] for name in producer.attributes["symbolic_tensor_outputs"])
                if (any(type(value) is not int or value <= 0 for value in (rows, padded, lanes))
                        or lanes != len(self.traces) or padded != -(-rows // lanes) * lanes
                        or producer.kind is not WorkloadKind.COLLECTIVE or producer.attributes["operation"] != "all_gather"
                        or producer.attributes["participants"] != self.participants
                        or len(tensor.shape) != 2 or tensor.shape[0] != rows or tensor.dtype != "bf16"
                        or tensor.view.offset_bytes != 0 or tensor.view.strides_bytes != (2 * tensor.shape[1], 2)
                        or not any(view.allocation_id == tensor.view.allocation_id and view.shape == (padded, tensor.shape[1])
                            and view.offset_bytes == 0 and view.dtype == "bf16" and view.strides_bytes == tensor.view.strides_bytes for view in ports)
                        or producer.attributes["symbolic_collective_input_shape"] != (padded // lanes, tensor.shape[1])):
                    raise ValueError("GLM MoE SP restore must consume the real-row prefix of its actual padded TP gather")
                producer_port = tensor.view
            else:
                producer_port = trace._complete_producer_view(tensor)
        if fused_norm:
            # FlashInfer's wrapper allocates both empty_like outputs; it does
            # not mutate/alias x or residual. Workspace remains unbound.
            result = trace.call_outputs(local, (tensor, residual, rmsnorm_weight), outputs=(
                ("normalized", tensor.shape, tensor.dtype), ("residual", residual.shape, residual.dtype)))
        else:
            result = trace.call(local, (tensor,), shape=shape, dtype=tensor.dtype,
                out=tensor if inplace else None)
        local_id = trace.qualify(identifier)
        if producer_port is not None:
            attrs = trace._nodes[local_id].attributes
            trace._nodes[local_id] = trace._nodes[local_id].with_attributes({**attrs,
                "symbolic_tensor_inputs": (producer_port.view_id, *attrs["symbolic_tensor_inputs"][1:])})
        if output_shape is not None:
            result = result.view_as_shape(output_shape)
            trace._nodes[local_id] = trace._nodes[local_id].with_attributes({
                **trace._nodes[local_id].attributes, "symbolic_tensor_outputs": (result.view.view_id,)})
        trace._nodes[local_id] = trace._nodes[local_id].with_attributes({**trace._nodes[local_id].attributes,
            "symbolic_collective_input_producers": (input_producer,),
            **({"symbolic_collective_external_inputs": (external_origin,)} if external_origin is not None else {})})
        if sequence:
            previous_name = sequence[-1]
            previous_count = sequence.count(previous_name) - 1
            previous_id = self.namespace + ":" + previous_name + (f":{previous_count}" if previous_count else "")
            trace._edges.append(WorkloadEdge(trace.qualify(previous_id), local_id, "barrier"))
        trace._collective_ids.add(local_id)
        sequence.append(name)
        if identifier not in self._contracts:
            self._contracts[identifier] = contract
            self._layouts[identifier] = layout
            self._submissions[identifier] = {}
        self._submissions[identifier][rank] = local_id
        return result

    def call_variable(self, name, trace, tensor, *, pool_name, config, rows_by_rank, row_shape, dtype,
                      moe_invocation_id=None, group_dimension="expert"):
        """One rank's variable-row expert gather or sum-scatter submission.

        Empty inputs/outputs have no fake tensor or allocation. Every rank
        still submits the operation, preserving process-group ordering.
        """
        if (self._recorded_graph is not None or trace not in self.traces
                or not isinstance(name, str) or not name
                or not isinstance(config, CollectiveConfig)
                or config.operation not in ("all_gatherv", "reduce_scatterv")):
            raise ValueError("variable collective requires its current source rank and canonical operation")
        if (not isinstance(rows_by_rank, tuple) or len(rows_by_rank) != len(self.traces)
                or any(type(rows) is not int or rows < 0 for rows in rows_by_rank) or not sum(rows_by_rank)
                or not isinstance(row_shape, tuple) or not row_shape
                or any(type(size) is not int or size <= 0 for size in row_shape) or dtype not in DTYPE_BYTES):
            raise ValueError("variable collective needs explicit nonnegative row extents and scalar row geometry")
        if group_dimension not in ("expert", "data", "data_context", "context", "tensor"):
            raise ValueError("variable source requires an explicit expert or data process group")
        if self._explicit_ring_order or self.participant_nic_ids:
            raise ValueError("variable direct-pairwise source does not infer ring/shared-NIC routing")
        if moe_invocation_id is not None and (not isinstance(moe_invocation_id, str) or not moe_invocation_id):
            raise ValueError("MoE invocation must be explicit and nonempty")
        rank, gather = trace.rank, config.operation == "all_gatherv"
        local_rows, total_rows = rows_by_rank[self.participants.index(rank)], sum(rows_by_rank)
        input_rows, output_rows = (local_rows, total_rows) if gather else (total_rows, local_rows)
        if input_rows:
            if (not isinstance(tensor, ForwardTensor) or tensor.trace is not trace or not tensor.is_contiguous
                    or tensor.shape != (input_rows, *row_shape) or tensor.dtype != dtype):
                raise ValueError("variable collective input disagrees with this participant's exact rows")
            trace._check_tensor(tensor)
        elif tensor is not None:
            raise ValueError("zero-row gather participants must not invent payload storage")
        input_producers = (trace._writers[tensor.view.allocation_id],) if input_rows else ()
        sequence = self._sequences[rank]
        invocation = sequence.count(name)
        identifier = self.namespace + ":" + name + (f":{invocation}" if invocation else "")
        for other in self._sequences.values():
            if len(other) > len(sequence) and other[len(sequence)] != name:
                raise ValueError("symbolic collective order disagrees across ranks")
        partitions = tuple(rows * prod(row_shape) * DTYPE_BYTES[dtype] for rows in rows_by_rank)
        contract = WorkloadNode(identifier, WorkloadKind.COLLECTIVE, pool_name, {}, dtype=dtype,
            rank=self.participants[0], attributes={
                "algorithm": config.algorithm, "backend": config.backend, "backend_config_sha256": config.backend_config_sha256,
                "bytes": sum(partitions), "bytes_semantics": "aggregate_input" if gather else "per_rank_input",
                "direction": "collective", "link": config.link, "topology": config.topology,
                "participants": self.participants, "operation": config.operation,
                "source_rank": None, "destination_rank": None, "ring_order": (),
                "reduction": None if gather else "sum", "resource_id": config.resource_id,
                "partition_bytes": partitions, "collective_group": group_dimension,
                "symbolic_variable_rows": rows_by_rank, "symbolic_variable_row_shape": row_shape,
                **({"moe_invocation_id": moe_invocation_id} if moe_invocation_id is not None else {})})
        layout = (rows_by_rank, row_shape, dtype)
        if identifier in self._contracts and (self._contracts[identifier] != contract
                or self._layouts[identifier] != layout or rank in self._submissions[identifier]):
            raise ValueError("variable collective contract or invocation disagrees across participants")
        local, local_id = replace(contract, rank=rank), trace.qualify(identifier)
        if output_rows:
            if tensor is None:
                result = trace.tensor(identifier + ":output", (output_rows, *row_shape), dtype)
                trace.call(local, (), shape=result.shape, dtype=dtype, out=result)
            else:
                result = trace.call(local, (tensor,), shape=(output_rows, *row_shape), dtype=dtype)
        else:
            # An empty reduce-scatter destination still contributes its full
            # expert result. Record the read and completion without storage.
            if local_id in trace._nodes:
                raise ValueError("variable collective invocation is not fresh")
            trace._nodes[local_id] = local.with_attributes({**local.attributes,
                "symbolic_tensor_inputs": (tensor.view.view_id,), "symbolic_tensor_outputs": ()}, node_id=local_id)
            writer = trace._writers[tensor.view.allocation_id]
            if writer is not None:
                trace._edges.append(WorkloadEdge(writer, local_id))
            trace._readers[tensor.view.allocation_id].add(local_id)
            trace._event(f"return:{local_id}")
            result = None
        trace._nodes[local_id] = trace._nodes[local_id].with_attributes({**trace._nodes[local_id].attributes,
            "symbolic_collective_input_producers": input_producers})
        if sequence:
            previous = sequence[-1]
            previous_id = self.namespace + ":" + previous
            count = sequence.count(previous) - 1
            if count:
                previous_id += f":{count}"
            trace._edges.append(WorkloadEdge(trace.qualify(previous_id), local_id, "barrier"))
        trace._collective_ids.add(local_id)
        sequence.append(name)
        if identifier not in self._contracts:
            self._contracts[identifier], self._layouts[identifier], self._submissions[identifier] = contract, layout, {}
        self._submissions[identifier][rank] = local_id
        return result

    @property
    def operator_order_by_rank(self):
        """Preserve each source program's order with canonical collective IDs."""
        if self._recorded_graph is not None:
            frame = (self._recorded_graph.__dict__["_native_source_frame"]
                if "_native_source_frame" in self._recorded_graph.__dict__ else None)
            if type(frame) is _NativeSourceFrame:
                owner = frame.owner()
                if (type(owner) is ForwardCollectiveSourceProgram
                        and self._sequences == owner.source._sequences
                        and self._submissions == owner.source._submissions):
                    if "_operator_order_by_rank" not in owner.__dict__:
                        owner.__dict__["_operator_order_by_rank"] = owner.source._joined_parts(order_only=True)
                    return owner.__dict__["_operator_order_by_rank"]
        return self._joined_parts(order_only=True)

    def _joined_parts(self, *, order_only=False):
        mapping, joined = {}, {}
        for group in (self, *self.subgroups):
            reference = group._sequences[group.participants[0]]
            if any(sequence != reference for sequence in group._sequences.values()):
                raise ValueError("symbolic collective group has missing rank submissions")
            for identifier, submissions in group._submissions.items():
                if identifier in joined or set(submissions) != set(group.participants):
                    raise ValueError("symbolic collective group has missing or colliding rank submissions")
                inputs, outputs, producers, external_inputs = [], [], [], []
                for trace in group.traces:
                    local_id = submissions[trace.rank]
                    if local_id in mapping:
                        raise ValueError("one source call cannot belong to multiple collective groups")
                    mapping[local_id] = identifier
                    local = trace._nodes[local_id]
                    inputs.extend(local.attributes["symbolic_tensor_inputs"])
                    outputs.extend(local.attributes["symbolic_tensor_outputs"])
                    if "symbolic_collective_input_producers" in local.attributes:
                        producers.extend(local.attributes["symbolic_collective_input_producers"])
                        external_inputs.extend(local.attributes["symbolic_collective_external_inputs"]
                            if "symbolic_collective_external_inputs" in local.attributes else
                            (None,) * len(local.attributes["symbolic_collective_input_producers"]))
                joined[identifier] = replace(group._contracts[identifier], attributes={
                    **group._contracts[identifier].attributes,
                    "symbolic_tensor_inputs": tuple(inputs), "symbolic_tensor_outputs": tuple(outputs),
                    "symbolic_collective_input_producers": tuple(producers),
                    **({"symbolic_collective_external_inputs": tuple(external_inputs)} if any(value is not None for value in external_inputs) else {})})
        for identifier, node in tuple(joined.items()):
            if "symbolic_collective_input_producers" in node.attributes:
                joined[identifier] = node.with_attributes({**node.attributes,
                    "symbolic_collective_input_producers": tuple(mapping[name] if name in mapping else name
                        for name in node.attributes["symbolic_collective_input_producers"])})
        nodes, edges = [], set()
        emitted = set()
        for trace in self.traces:
            if trace._collective_ids - set(mapping):
                raise ValueError("rank trace contains collectives from another group")
            if not order_only:
                for local_id, node in trace._nodes.items():
                    if local_id in mapping:
                        identifier = mapping[local_id]
                        if identifier not in emitted:
                            nodes.append(joined[identifier])
                            emitted.add(identifier)
                    else:
                        nodes.append(node)
            for edge in trace._capture_program_edges():
                source = mapping[edge.source] if edge.source in mapping else edge.source
                destination = mapping[edge.destination] if edge.destination in mapping else edge.destination
                # The order query retains typed-edge and capture validation,
                # but does not retain, deduplicate or sort an unused graph.
                joined_edge = WorkloadEdge(source, destination, edge.kind)
                if not order_only:
                    edges.add(joined_edge)
        if order_only:
            return {trace.rank: tuple(mapping[name] if name in mapping else name for name in trace._nodes)
                for trace in self.traces}
        # A data edge already enforces the process-group order. Avoid duplicate
        # KernelCall dependencies when ranks contributed both edge kinds.
        edges = {edge for edge in edges if edge.kind != "barrier"
            or WorkloadEdge(edge.source, edge.destination) not in edges}
        return tuple(nodes), tuple(sorted(edges, key=lambda edge:
            (edge.source, edge.destination, edge.kind)))

    @property
    def graph(self):
        if self._recorded_graph is not None:
            return self._recorded_graph
        graph = WorkloadGraph(*self._joined_parts())
        graph.topological_order()
        return graph



class ForwardCollectiveSourceProgram(ForwardSourceProgram):
    """Replay a complete source group with original submissions and local storage.

    Full collective contracts/layouts/submissions come from the complete source,
    never from extrapolating prefix payloads. The actual dual-rank prefix checks
    current static contracts and supplies only current attention/KV parameters.
    """

    def __init__(self, source, prefix, *, node_bindings, allocation_bindings, collective_attribute_bindings=(),
                 transient_allocation_bindings=(), view_bindings=(), capture_bindings=(), collective_geometry_bindings=(),
                 geometry_rebind=False):
        if not isinstance(source, ForwardCollectiveGroup) or not isinstance(prefix, ForwardCollectiveGroup):
            raise TypeError("collective source program requires actual source/prefix groups")
        source_parts = source._joined_parts()
        prefix._joined_parts()
        if (source.namespace != prefix.namespace or source.participants != prefix.participants
                or source.subgroups or prefix.subgroups
                or len(source.traces) != 2 or source.participant_nic_ids != prefix.participant_nic_ids
                or source.ring_order != prefix.ring_order or source._explicit_ring_order != prefix._explicit_ring_order):
            raise ValueError("collective source programs require matching explicit dual-rank groups")
        self.collective_geometry_bindings = tuple(collective_geometry_bindings)
        for target, origin in self.collective_geometry_bindings:
            if not self._collective_geometry_matches(source, target) or not self._collective_geometry_matches(prefix, origin):
                raise ValueError("initial collective geometry must be actual matching source facts")
            actual = prefix._contracts[origin]
            expected = replace(source._contracts[target], node_id=actual.node_id)
            if geometry_rebind:
                expected = expected.with_updates({}, {"bytes": actual.attributes["bytes"]})
            if expected != actual or not geometry_rebind and source._layouts[target] != prefix._layouts[origin]:
                raise ValueError("initial collective geometry must be actual matching source facts")
        self.collective_attribute_bindings = tuple(collective_attribute_bindings)
        targets = tuple(binding[0] for binding in self.collective_attribute_bindings)
        if len(targets) != len(set(targets)):
            raise ValueError("collective attribute binding targets must be unique")
        local_bindings = []
        ignored = {}
        for target, origin, attributes in self.collective_attribute_bindings:
            if (target not in source._contracts or origin not in prefix._contracts
                    or attributes != ("moe_invocation_id",)):
                raise ValueError("collective binding requires declared MoE invocation identity only")
            for name in attributes:
                if (name not in source._contracts[target].attributes or name not in prefix._contracts[origin].attributes
                        or source._contracts[target].attributes[name] != prefix._contracts[origin].attributes[name]
                        or type(prefix._contracts[origin].attributes[name]) is not str
                        or not prefix._contracts[origin].attributes[name]):
                    raise ValueError("initial collective attribute binding disagrees with complete source")
            ignored[origin] = attributes
            for trace, current in zip(source.traces, prefix.traces, strict=True):
                source_id, prefix_id = source._submissions[target][trace.rank], prefix._submissions[origin][current.rank]
                if any(trace._nodes[source_id].attributes[name] != source._contracts[target].attributes[name]
                        or current._nodes[prefix_id].attributes[name] != prefix._contracts[origin].attributes[name]
                        for name in attributes):
                    raise ValueError("collective local and joined invocation identities disagree")
                local_bindings.append((source_id, prefix_id, (), attributes))
        for target, origin in self.collective_geometry_bindings:
            geometry_attributes = ("bytes", "symbolic_collective_input_shape")
            if origin in ignored and ignored[origin] != geometry_attributes:
                raise ValueError("collective geometry cannot overlap annotation bindings")
            ignored[origin] = geometry_attributes
            for trace, current in zip(source.traces, prefix.traces, strict=True):
                local_bindings.append((source._submissions[target][trace.rank], prefix._submissions[origin][current.rank], (), geometry_attributes))
        self.ignored_collective_attributes = ignored
        self.prefix_collective_contracts = {identifier: self._collective_signature(node,
            ignored[identifier] if identifier in ignored else ()) for identifier, node in prefix._contracts.items()}
        node_bindings, allocation_bindings = (*tuple(node_bindings), *local_bindings), tuple(allocation_bindings)
        for bindings, field in ((node_bindings, "_nodes"), ((*allocation_bindings, *tuple(transient_allocation_bindings)), "_allocations"), (tuple(view_bindings), "_views")):
            targets = tuple(binding[0] for binding in bindings)
            if len(targets) != len(set(targets)):
                raise ValueError("collective source program binding targets must be unique")
            for binding in bindings:
                target, origin = binding[:2]
                owners = tuple(index for index, trace in enumerate(source.traces)
                    if target in getattr(trace, field))
                if len(owners) != 1 or origin not in getattr(prefix.traces[owners[0]], field):
                    raise ValueError("collective source program binding must belong to exactly one matching source/prefix rank")
        # The same constructor already joined and validated this source.
        # Reuse only that local result; do not cache/close either public group.
        if source._recorded_graph is None:
            graph = WorkloadGraph(*source_parts)
            graph.topological_order()
        else:
            graph = source.graph
        self.source, self.prefix, self.graph = source, prefix, graph
        self.rank_programs = tuple(ForwardSourceProgram(trace, current,
            node_bindings=tuple(binding for binding in node_bindings if binding[0] in trace._nodes),
            allocation_bindings=tuple(binding for binding in allocation_bindings if binding[0] in trace._allocations),
            collective_groups=(source, prefix),
            transient_allocation_bindings=tuple(binding for binding in transient_allocation_bindings if binding[0] in trace._allocations),
            view_bindings=tuple(binding for binding in view_bindings if binding[0] in trace._views),
            capture_bindings=capture_bindings, geometry_rebind=geometry_rebind)
            for trace, current in zip(source.traces, prefix.traces, strict=True))

    @staticmethod
    def _collective_signature(node, ignored):
        return (node.node_id, node.kind, node.pool_name, node.shape, node.dtype, node.variant, node.rank,
            {name: value for name, value in node.attributes.items() if name not in ignored})

    @staticmethod
    def _collective_geometry_matches(group, identifier):
        """Validate actual typed ports, including the all-reduce alias contract."""
        contract, layout = group._contracts[identifier], group._layouts[identifier]
        operation, storage = contract.attributes["operation"], contract.attributes["symbolic_collective_storage"]
        if (operation not in ("all_gather", "all_reduce")
                or storage not in (("fresh", "inplace") if operation == "all_reduce" else ("fresh",))
                or contract.attributes["symbolic_collective_dim"] != (0 if operation == "all_gather" else None)
                or layout[4] is not None):
            return False
        value = contract.attributes["bytes"]
        if (type(value) is not int or value <= 0
                or any(type(size) is not int or size <= 0 for size in layout[0])
                or contract.attributes["symbolic_collective_input_shape"] != layout[0]
                or value != prod(layout[0]) * DTYPE_BYTES[layout[1]]
                or layout[2] != ((len(group.participants) * layout[0][0], *layout[0][1:])
                    if operation == "all_gather" else layout[0])):
            return False
        output_shape = layout[2] if layout[3] is None else layout[3]
        if (any(type(size) is not int or size <= 0 for size in output_shape)
                or prod(output_shape) != prod(layout[2])
                or (storage == "inplace" and layout[3] is not None)):
            return False
        for trace in group.traces:
            local = trace._nodes[group._submissions[identifier][trace.rank]]
            if any(local.attributes[field] != contract.attributes[field] for field in (
                    "operation", "symbolic_collective_storage", "symbolic_collective_dim", "bytes",
                    "symbolic_collective_input_shape")):
                return False
            inputs, outputs = local.attributes["symbolic_tensor_inputs"], local.attributes["symbolic_tensor_outputs"]
            if len(inputs) != 1 or len(outputs) != 1:
                return False
            actual_input, actual_output = trace._views[inputs[0]], trace._views[outputs[0]]
            if (actual_input.shape, actual_input.dtype, actual_output.shape, actual_output.dtype, local.dtype) != (
                    layout[0], layout[1], output_shape, layout[1], layout[1]):
                return False
            if storage == "inplace":
                if inputs != outputs:
                    return False
            elif actual_input.allocation_id == actual_output.allocation_id:
                return False
            for view in (actual_input, actual_output):
                stride = view.element_bytes
                for size, actual in zip(reversed(view.shape), reversed(view.strides_bytes), strict=True):
                    if size != 1 and actual != stride:
                        return False
                    stride *= size
        return True

    def bind(self, prefix):
        if not isinstance(prefix, ForwardCollectiveGroup):
            raise TypeError("collective source binding requires its actual dual-rank prefix")
        native_frame = _native_source_scope(self, prefix)
        # Validate the same submissions without constructing a temporary
        # joined DAG that binding discards. Full signatures and rank-prefix
        # contracts are still checked below before rebinding the actual graph.
        reference = prefix._sequences[prefix.participants[0]]
        if any(sequence != reference for sequence in prefix._sequences.values()):
            raise ValueError("symbolic collective group has missing rank submissions")
        local_ids = set()
        for identifier, submissions in prefix._submissions.items():
            if set(submissions) != set(prefix.participants):
                raise ValueError("symbolic collective group has missing rank submissions")
            for trace in prefix.traces:
                local_id = submissions[trace.rank]
                local = trace._nodes[local_id]
                # Preserve the original required-port and iterable boundaries.
                tuple(local.attributes["symbolic_tensor_inputs"])
                tuple(local.attributes["symbolic_tensor_outputs"])
                local_ids.add(local_id)
        for trace in prefix.traces:
            if trace._collective_ids - local_ids:
                raise ValueError("rank trace contains collectives from another group")
        if any(getattr(prefix, name) != getattr(self.prefix, name) for name in (
                "namespace", "participants", "participant_nic_ids", "ring_order", "_explicit_ring_order",
                "_sequences", "_submissions")):
            return None
        geometry_origins = {origin for _, origin in self.collective_geometry_bindings}
        if set(prefix._layouts) != set(self.prefix._layouts):
            return None
        for name, layout in prefix._layouts.items():
            old = self.prefix._layouts[name]
            if name not in geometry_origins:
                if layout != old:
                    return None
            elif (len(layout) != len(old) or tuple(layout[index] for index in (1, 3, 4)) != tuple(old[index] for index in (1, 3, 4))):
                return None
        contracts = {identifier: self._collective_signature(node,
            self.ignored_collective_attributes[identifier] if identifier in self.ignored_collective_attributes else ())
            for identifier, node in prefix._contracts.items()}
        if contracts != self.prefix_collective_contracts:
            return None
        for target, origin, attributes in self.collective_attribute_bindings:
            for name in attributes:
                if name not in prefix._contracts[origin].attributes:
                    return None
                value = prefix._contracts[origin].attributes[name]
                if type(value) is not str or not value or any(
                        name not in trace._nodes[prefix._submissions[origin][trace.rank]].attributes
                        or trace._nodes[prefix._submissions[origin][trace.rank]].attributes[name] != value
                        for trace in prefix.traces):
                    return None
        for _, origin in self.collective_geometry_bindings:
            if not self._collective_geometry_matches(prefix, origin):
                return None
        traces = tuple(program.bind(trace, native_frame=native_frame) for program, trace in zip(self.rank_programs, prefix.traces, strict=True))
        if any(trace is None for trace in traces):
            return None
        result = copy(self.source)
        result.traces = traces
        if native_frame:
            result._layouts = dict(self.source._layouts)
            updates = {}
            for target, origin, attributes in self.collective_attribute_bindings:
                updates[target] = ({}, {name: prefix._contracts[origin].attributes[name] for name in attributes})
            for target, origin in self.collective_geometry_bindings:
                changes = {name: prefix._contracts[origin].attributes[name]
                    for name in ("bytes", "symbolic_collective_input_shape")}
                updates[target] = ({}, changes)
                result._layouts[target] = prefix._layouts[origin]
            result._contracts = _SourceFrameNodes(MappingProxyType(dict(self.source._contracts)),updates,None)
            joined = _SourceFrameNodes(MappingProxyType({target:self.graph.node(target) for target in updates}),updates,None)
        else:
            result._contracts = dict(self.source._contracts)
            result._layouts = dict(self.source._layouts)
            joined = {}
            for target, origin, attributes in self.collective_attribute_bindings:
                changes = {name: prefix._contracts[origin].attributes[name] for name in attributes}
                contract = result._contracts[target]
                result._contracts[target] = contract.with_updates({}, changes)
                original = self.graph.node(target)
                joined[target] = original.with_updates({}, changes)
            for target, origin in self.collective_geometry_bindings:
                contract = result._contracts[target]
                changes = {name: prefix._contracts[origin].attributes[name]
                    for name in ("bytes", "symbolic_collective_input_shape")}
                result._contracts[target] = contract.with_updates({}, changes)
                result._layouts[target] = prefix._layouts[origin]
                original = self.graph.node(target)
                joined[target] = original.with_updates({}, changes)
        # Every collective node, full submission and edge is retained from the
        # original complete group. Only current per-rank compute nodes change.
        if native_frame:
            result._recorded_graph = _issue_source_frame(self, result, joined)
        else:
            local = {name: node for trace in traces for name, node in trace._nodes.items()
                if name not in trace._collective_ids}
            result._recorded_graph = self.graph.with_nodes(
                local[node.node_id] if node.node_id in local else joined[node.node_id] if node.node_id in joined else node
                for node in self.graph.nodes)
            if type(self) is ForwardCollectiveSourceProgram:
                graph = result._recorded_graph
                if type(graph) is WorkloadGraph:
                    object.__setattr__(graph, "_ep_source_binding", _EPSourceBinding(self, graph, result.traces))
        return result


class _EPSourceBinding:
    """Immutable receipt from one actual complete source-program bind."""

    __slots__ = ("program_reference", "graph_reference", "nodes", "edges", "token_domains", "node_layout")

    def __init__(self, program, graph, traces):
        if type(program) is not ForwardCollectiveSourceProgram or type(graph) is not WorkloadGraph:
            raise TypeError("source receipt requires canonical source program and graph")
        object.__setattr__(self, "program_reference", weakref.ref(program))
        object.__setattr__(self, "graph_reference", weakref.ref(graph))
        object.__setattr__(self, "nodes", graph.nodes)
        object.__setattr__(self, "edges", graph.edges)
        object.__setattr__(self, "node_layout", _lineage_source_node_layout(program, graph))
        domains = []
        for trace in traces:
            if type(trace) is not ForwardTrace or type(trace.rank) is not int:
                domains = None
                break
            captures = trace.source_captures
            if len(captures) != 1:
                domains = None
                break
            capture = captures[0]
            if ("logical_tokens" not in capture or "model_input_tokens" not in capture
                    or type(capture["logical_tokens"]) is not int
                    or type(capture["model_input_tokens"]) is not int):
                domains = None
                break
            domains.append((trace.rank, capture["logical_tokens"], capture["model_input_tokens"]))
        object.__setattr__(self, "token_domains", None if domains is None else tuple(domains))

    def __setattr__(self, name, value):
        raise TypeError("source EP receipt is immutable")


def _ep_source_owner(graph, *, check_source=True):
    """Return the actual binding owner while its exact source receipt holds."""
    if type(graph) is not WorkloadGraph or "_ep_source_binding" not in graph.__dict__:
        return None
    receipt = graph.__dict__["_ep_source_binding"]
    if type(receipt) is _NativeSourceFrame:
        return receipt.owner(check_source=check_source)
    if (type(receipt) is not _EPSourceBinding or receipt.graph_reference() is not graph
            or receipt.nodes is not graph.nodes or receipt.edges is not graph.edges):
        return None
    owner = receipt.program_reference()
    if type(owner) is not ForwardCollectiveSourceProgram:
        return None
    if check_source:
        from . import traced_models as source
        if not source._DENSE_METADATA_SOURCE_CONTRACT.unchanged():
            return None
    return owner


def _ep_source_token_domains(graph):
    """Return native capture pairs copied by the actual source issuer."""
    if _ep_source_owner(graph, check_source=False) is None:
        return None
    return graph.__dict__["_ep_source_binding"].token_domains


_SOURCE_LINEAGE_RECIPES = weakref.WeakKeyDictionary()


def _lineage_binding_recipe(program):
    if (type(program) not in (ForwardSourceProgram, ForwardCollectiveSourceProgram) or type(program.graph) is not WorkloadGraph
            or type(program.rank_programs) is not tuple):
        return None
    if any(type(rank) is not ForwardSourceProgram for rank in program.rank_programs):return None
    if any(type(node) is not WorkloadNode or type(node.node_id) is not str for node in program.graph.nodes):return None
    declarations = (tuple(rank.node_bindings for rank in program.rank_programs),
        program.collective_attribute_bindings, program.collective_geometry_bindings)
    if program in _SOURCE_LINEAGE_RECIPES:
        graph_reference, previous, bindings = _SOURCE_LINEAGE_RECIPES[program]
        if graph_reference() is program.graph and len(declarations[0]) == len(previous[0]) and all(current is old
                for current, old in zip(declarations[0], previous[0], strict=True)) and all(
                    current is old for current, old in zip(declarations[1:], previous[1:], strict=True)):
            return bindings
    targets = {node.node_id for node in program.graph.nodes}
    writes = {}
    def write(target, origin, shapes, attrs):
        if (type(target) is not str or type(shapes) is not tuple or type(attrs) is not tuple
                or any(type(name) is not str for name in (*shapes, *attrs)) or target not in targets):
            return False
        if target in writes:
            return False
        writes[target] = (origin, shapes, attrs)
        return True
    for index, rank in enumerate(program.rank_programs):
        if type(rank) is not ForwardSourceProgram or type(rank.node_bindings) is not tuple:
            return None
        for binding in rank.node_bindings:
            if type(binding) is not tuple or len(binding) != 4:
                return None
            target, origin, shapes, attrs = binding
            if type(origin) is not str:
                return None
            if target not in targets:
                if target not in rank.source._collective_ids:
                    return None
                continue
            if not write(target, (index, origin), shapes, attrs):return None
    for binding in program.collective_attribute_bindings:
        if type(binding) is not tuple or len(binding) != 3:return None
        target, origin, attrs = binding
        if type(origin) is not str or not write(target, ("collective", origin), (), attrs):return None
    for binding in program.collective_geometry_bindings:
        if type(binding) is not tuple or len(binding) != 2:return None
        target, origin = binding
        if type(origin) is not str or not write(target, ("collective", origin), (),
                ("bytes", "symbolic_collective_input_shape")):return None
    bindings = tuple((node.node_id, *writes[node.node_id]) for node in program.graph.nodes if node.node_id in writes)
    _SOURCE_LINEAGE_RECIPES[program] = (weakref.ref(program.graph), declarations, bindings)
    return bindings


def _lineage_source_node_layout(program, graph):
    # Receipt payload is an untrusted recipe descriptor. Only the accessor
    # below establishes the actual-node facts from owner-issued lineage.
    bindings = _lineage_binding_recipe(program)
    template = program.graph
    if (bindings is None or len(template.nodes) != len(graph.nodes)
            or template.edges is not graph.edges):return None
    return weakref.ref(template), template.nodes, bindings


def _source_bound_node_layout(graph):
    if "_native_source_frame" in graph.__dict__:
        frame = graph.__dict__["_native_source_frame"]
        if type(frame) is _NativeSourceFrame and frame.owner() is not None:
            return frame.layout
        return None
    if _ep_source_owner(graph) is None or not _node_update_source_unchanged():
        return None
    receipt = graph.__dict__["_ep_source_binding"]
    layout = receipt.node_layout
    if layout is None:
        return None
    template = layout[0]()
    if (type(template) is not WorkloadGraph or template.nodes is not layout[1]
            or template.edges is not graph.edges or template.block_repeats is not graph.block_repeats):
        return None
    # Recheck intrinsic seals, including parent/actual shape+attribute refs;
    # no mutable prefix validator or program-history audit is involved.
    bound = {target: (origin, shapes, attrs) for target, origin, shapes, attrs in layout[2]}
    referents = {}
    for base, actual in zip(template.nodes, graph.nodes, strict=True):
        if base.node_id not in bound:
            if actual is not base:return None
        else:
            updates = _workload_node_field_updates_read(actual)
            origin, shapes, attrs = bound[base.node_id]
            if updates is None or updates[0] is not base or updates[1] != shapes or updates[2] != attrs:return None
            for mapping, fields, kind in ((actual.shape, shapes, "shape"), (actual.attributes, attrs, "attributes")):
                for name in fields:
                    key = (origin, kind, name)
                    value = mapping[name]
                    if key in referents:
                        if referents[key] is not value:return None
                    else:referents[key] = value
    return layout


class ForwardKVCacheGroup:
    """Join an executed virtual forward with externally owned KV DMA effects.

    KVCacheTierManager owns the persistent Host/HBM pages and their retirement.
    These events carry exact per-rank bytes; they do not manufacture activation
    tensors, physical page addresses or an unobserved DMA workspace lifetime.
    """

    def __init__(self, source, *, pool, step_id, transfer_attributes):
        if isinstance(source, ForwardTrace):
            self.traces = (source,)
        elif isinstance(source, (ForwardCollectiveGroup, ForwardPipelineGroup, ForwardActivationGroup)):
            self.traces = source.traces
        else:
            raise TypeError("KV effects require the actual virtual model forward")
        if not isinstance(transfer_attributes, tuple) or not transfer_attributes:
            raise ValueError("KV effects require explicit validated transfer intents")
        if not set(pool.ranks) <= {trace.rank for trace in self.traces}:
            raise ValueError("KV transfer owner ranks are absent from the virtual forward")
        identifiers = tuple(attributes["ready_event"] for attributes in transfer_attributes)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("KV offload ready_event values must be unique")
        graph = source.graph
        incoming = {edge.destination for edge in graph.edges}
        outgoing = {edge.source for edge in graph.edges}
        roots = tuple(node.node_id for node in graph.nodes if node.node_id not in incoming)
        leaves = tuple(node.node_id for node in graph.nodes if node.node_id not in outgoing)
        if not roots or not leaves:
            raise ValueError("KV effects require a nonempty model dependency graph")
        before, after, edges = [], [], list(graph.edges)
        for rank in pool.ranks:
            loads, evicts = [], []
            for attributes in transfer_attributes:
                name = f"{step_id}:kv-offload:{attributes['ready_event']}"
                if len(pool.ranks) > 1:
                    name += f":tp-rank-{rank}"
                node = WorkloadNode(name, WorkloadKind.KV_OFFLOAD, pool.name, {}, rank=rank,
                    attributes=attributes)
                if attributes["direction"] == "load":
                    loads.append(node)
                elif attributes["direction"] in ("evict", "store"):
                    evicts.append(node)
                else:
                    raise ValueError("KV effects only accept load, evict and store directions")
            # Each intent must independently be a pre-compute root or a
            # post-compute leaf, as required by the existing completion
            # contract. Copy engines/link resources serialize overlapping
            # transfers; unrelated page intents are not payload dependencies.
            edges.extend(WorkloadEdge(load.node_id, root, "barrier") for load in loads for root in roots)
            edges.extend(WorkloadEdge(leaf, evict.node_id, "barrier") for evict in evicts for leaf in leaves)
            before.extend(loads)
            after.extend(evicts)
        self._graph = WorkloadGraph((*before, *graph.nodes, *after), tuple(edges))
        self._graph.topological_order()
        self._source = source

    @property
    def operator_order_by_rank(self):
        # Capture offsets index the original trace's calls. External DMA
        # nodes have no trace positions and remain outside every capture.
        return self._source.operator_order_by_rank

    @property
    def graph(self):
        return self._graph


class ForwardLocalGroup:
    """One TP rank: the source's collective is an identity, not a transfer."""
    def __init__(self, trace):
        if not isinstance(trace, ForwardTrace):
            raise TypeError("local source group requires a ForwardTrace")
        self.traces = (trace,)
        self.participants = (trace.rank,)

    def call(self, name, tensor, *, pool_name, config, inplace):
        if config is not None or type(inplace) is not bool or not name or not pool_name:
            raise ValueError("TP1 source collective requires explicit absent communication config")
        self.traces[0]._check_tensor(tensor)
        return tensor


class _SourceParameterSlots(Mapping):
    """Cold node-to-slot layout with one current parameter packet per slot.

    Layouts are private immutable recipes. Values belong to this binding only;
    node IDs and each node's static fields remain in the recorded source.
    """
    __slots__ = ("layout", "packets")
    def __init__(self, layout, packets):
        self.layout = layout
        self.packets = packets
    def __setattr__(self, name, value):
        if hasattr(self, name): raise TypeError("source parameter slots are immutable")
        object.__setattr__(self, name, value)
    def __len__(self): return len(self.layout)
    def __iter__(self): return iter(self.layout)
    def __getitem__(self, name): return self.packets[self.layout[name]]


class _SourceFrameNodes(Mapping):
    """One immutable actual binding; canonical node objects are inspection views."""
    __slots__ = ("base", "updates", "cache", "launch_policy")
    def __init__(self, base, updates, launch_policy):
        self.base = base
        if type(updates) is _SourceParameterSlots:
            self.updates = _SourceParameterSlots(updates.layout,
                tuple((MappingProxyType(shape), MappingProxyType(attrs))
                    for shape, attrs in updates.packets))
        else:
            self.updates = MappingProxyType({name: (MappingProxyType(shape), MappingProxyType(attrs))
                for name, (shape, attrs) in updates.items()})
        self.cache = {}
        self.launch_policy = launch_policy
    def __setattr__(self, name, value):
        if hasattr(self, name): raise TypeError("source frame node binding is immutable")
        object.__setattr__(self, name, value)
    def __len__(self): return len(self.base)
    def __iter__(self): return iter(self.base)
    def __getitem__(self, name):
        if name not in self.cache:
            if name in self.updates:
                shape, attrs = self.updates[name]
                self.cache[name] = self.base[name].with_updates(dict(shape), dict(attrs))
            else:
                self.cache[name] = self.base[name].with_attributes(self.base[name].attributes)
        return self.cache[name]


_NATIVE_RANK_LAYOUTS = weakref.WeakKeyDictionary()


def _source_rank_frame(program, updates):
    base_nodes = program.graph.nodes
    current_nodes = tuple(program.source._nodes.values())
    if (len(base_nodes) != len(current_nodes)
            or any(type(actual) is not WorkloadNode or node.node_id != actual.node_id
                for node, actual in zip(base_nodes, current_nodes, strict=True))):
        return None
    if program in _NATIVE_RANK_LAYOUTS:
        graph_reference, previous_nodes, base, policy = _NATIVE_RANK_LAYOUTS[program]
        if (graph_reference() is program.graph and all(current is previous
                for current, previous in zip(current_nodes, previous_nodes, strict=True))):
            return _SourceFrameNodes(base, updates, policy)
    policies = frozenset(node.attributes["launch_policy"] for node in current_nodes if "launch_policy" in node.attributes)
    policy = next(iter(policies)) if len(policies) == 1 else None
    base = MappingProxyType(dict(program.source._nodes))
    _NATIVE_RANK_LAYOUTS[program] = (weakref.ref(program.graph), current_nodes, base, policy)
    return _SourceFrameNodes(base, updates, policy)


class _SourceFrameSequence(Sequence):
    """Own node-view inputs without retaining the frame that owns this view."""
    def __init__(self, frame):
        self.ids, self.origins = frame.ids, frame.origins
        self.base_nodes, self.node_maps = frame.base_nodes, frame.node_maps
        self.joined, self.detached_nodes = frame.joined, frame.detached_nodes
    def __setattr__(self, name, value):
        if hasattr(self, name): raise TypeError("source node sequence binding is immutable")
        object.__setattr__(self, name, value)
    def __len__(self): return len(self.ids)
    def __getitem__(self, index):
        if isinstance(index, slice): return tuple(self.node(name) for name in self.ids[index])
        return self.node(self.ids[index])
    def node(self, name):
        origin = self.origins[name]
        if origin is None:
            if name not in self.detached_nodes:
                self.detached_nodes[name] = self.base_nodes[name].with_attributes(self.base_nodes[name].attributes)
            return self.detached_nodes[name]
        if origin == "collective": return self.joined[name]
        return self.node_maps[origin][name]


class _SourceFrameLookup(Mapping):
    def __init__(self, frame): self.frame = frame
    def __len__(self): return len(self.frame.ids)
    def __iter__(self): return iter(self.frame.ids)
    def __getitem__(self, name): return self.frame.node(name)


_NATIVE_SOURCE_RECIPES = weakref.WeakKeyDictionary()


def _source_frame_traces(source):
    if type(source) is ForwardTrace:
        return (source,)
    if type(source) is ForwardCollectiveGroup:
        return source.traces
    raise TypeError("native source frames require actual local or collective traces")


class _NativeSourceFrame:
    def __init__(self, program, result, graph, joined, compiled):
        self.program_reference, self.graph_reference = weakref.ref(program), weakref.ref(graph)
        self.template_reference = weakref.ref(program.graph)
        self.recipe_signature = _native_recipe_signature(program)
        self.trace_references = tuple(weakref.ref(trace) for trace in _source_frame_traces(result))
        self.trace_types = tuple(type(trace) for trace in _source_frame_traces(result))
        self.node_maps = tuple(trace._nodes for trace in _source_frame_traces(result))
        self.joined = joined
        self.ids, self.origins, self.layout, self.workload_contract, self.routing_ids, self.storage_rows, self.base_nodes, self.expected_rank_bases = compiled
        self.materialized = None
        self.detached_nodes = {}
        self.nodes = _SourceFrameSequence(self)
        self.exported_topology = {}
        self.token_domains = tuple((trace.rank, trace.source_captures[0]["logical_tokens"],
            trace.source_captures[0]["model_input_tokens"]) for trace in _source_frame_traces(result))
    def owner(self, *, check_source=True):
        owner, graph = self.program_reference(), self.graph_reference()
        if (type(owner) not in (ForwardSourceProgram, ForwardCollectiveSourceProgram) or owner.graph is not self.template
                or graph is None or graph.__dict__["_ep_source_binding"] is not self
                or not _native_recipe_matches(owner,self.recipe_signature)
                or self.expected_rank_bases is None
                or any(nodes.base is not expected for nodes, expected in zip(
                    self.node_maps, self.expected_rank_bases, strict=True))
                or any(reference() is not None and reference()._nodes is not nodes
                    for reference, nodes in zip(self.trace_references, self.node_maps, strict=True))):
            return None
        if check_source:
            if (type(owner.source) in (ForwardTrace, ForwardCollectiveGroup)
                    and all(type(rank) is ForwardSourceProgram for rank in owner.rank_programs)
                    and type(self.trace_types) is tuple
                    and len(self.trace_types) == len(self.trace_references)
                    and all(kind is ForwardTrace for kind in self.trace_types)
                    and all(reference() is None or type(reference()) is ForwardTrace for reference in self.trace_references)
                    and all(type(nodes) is _SourceFrameNodes for nodes in self.node_maps)):
                if not _native_consumer_source_unchanged(): return None
            else:
                from . import traced_models as source
                if not source._DENSE_METADATA_SOURCE_CONTRACT.unchanged(): return None
        return owner
    def node(self, name):
        return self.nodes.node(name)
    def pool_name(self, name):
        # Pool membership cannot be changed by shape/attribute bindings. If
        # the current frame has already exported a node, honor that input.
        origin = self.origins[name]
        if origin is None:
            nodes = self.detached_nodes
        elif origin == "collective":
            nodes = self.joined.cache
        else:
            nodes = self.node_maps[origin].cache
        return (nodes[name] if name in nodes else self.base_nodes[name]).pool_name
    def public_topology(self, name):
        if name not in self.exported_topology:
            if name == "edges":
                value = tuple(replace(edge) for edge in self.workload_contract[1])
            elif name == "block_repeats":
                value = tuple(replace(repeat,
                    feedback_edges=tuple(replace(edge) for edge in repeat.feedback_edges))
                    for repeat in self.workload_contract[2])
            else:
                raise ValueError("unknown source topology export")
            self.exported_topology[name] = value
        return self.exported_topology[name]
    def fields(self, name):
        base = self.base_nodes[name]
        origin = self.origins[name]
        if origin is None: return base.shape, base.attributes
        if origin == "collective":
            base = self.joined.base[name]
            shape, attrs = self.joined.updates[name]
            return _OverlayFields(base.shape, shape), _OverlayFields(base.attributes, attrs)
        nodes = self.node_maps[origin]
        if name not in nodes.updates: return base.shape, base.attributes
        shape, attrs = nodes.updates[name]
        return _OverlayFields(base.shape, shape), _OverlayFields(base.attributes, attrs)
    def public_nodes(self):
        if self.materialized is None:
            object.__setattr__(self,"materialized",tuple(self.node(name) for name in self.ids))
        return self.materialized

    @property
    def template(self):
        return self.template_reference()
    def __setattr__(self, name, value):
        if name in self.__dict__: raise TypeError("source frame binding is immutable")
        object.__setattr__(self,name,value)


class _OverlayFields(Mapping):
    def __init__(self, base, updates): self.base, self.updates = base, updates
    def __len__(self): return len(self.base)
    def __iter__(self): return iter(self.base)
    def __getitem__(self, name):
        return self.updates[name] if name in self.updates else self.base[name]


def _issue_source_frame(program, result, joined):
    if program in _NATIVE_SOURCE_RECIPES and not _native_recipe_matches(program,_NATIVE_SOURCE_RECIPES[program][0]):
        del _NATIVE_SOURCE_RECIPES[program]
    if program not in _NATIVE_SOURCE_RECIPES:
        template = program.graph
        bindings = _lineage_binding_recipe(program)
        origins = {}
        for node in template.nodes:
            if node.node_id in joined: origins[node.node_id] = "collective"
            else:
                indices = tuple(index for index, trace in enumerate(_source_frame_traces(result))
                    if node.node_id in trace._nodes and node.node_id not in trace._collective_ids)
                if len(indices) > 1: raise ValueError("source frame has ambiguous rank node")
                origins[node.node_id] = indices[0] if indices else None
        # Only unchanged activation storage may use the compiled projection.
        # Dynamic transient/view bindings retain their ordinary reporter path.
        static_storage = not any(rank.transient_origins for rank in program.rank_programs)
        rows = None if not static_storage else tuple((trace.rank, trace.activation_allocations,
            len(trace._views), MappingProxyType(trace.workspace_capacity_bytes)) for trace in _source_frame_traces(program.source))
        rank_bases = tuple(trace._nodes.base for trace in _source_frame_traces(result))
        static_matches = all(node is template.node(name)
            for trace, base in zip(_source_frame_traces(result), rank_bases, strict=True)
            for name, node in base.items() if name not in trace._collective_ids)
        compiled = (tuple(node.node_id for node in template.nodes), MappingProxyType(origins),
            (weakref.ref(template), template.nodes, bindings),
            (tuple((node.node_id,node.kind.value,node.pool_name,node.rank) for node in template.nodes),
                template.edges, template.block_repeats, template.logical_node_count,template.logical_edge_count),
            tuple(node.node_id for node in template.nodes if node.kind is WorkloadKind.FUSED_MOE), rows,
            MappingProxyType({node.node_id: node for node in template.nodes}),
            rank_bases if static_matches else None)
        _NATIVE_SOURCE_RECIPES[program] = (_native_recipe_signature(program),compiled)
    graph = object.__new__(WorkloadGraph)
    graph.__dict__.update(program.graph.__dict__)
    frame = _NativeSourceFrame(program, result, graph, joined, _NATIVE_SOURCE_RECIPES[program][1])
    object.__setattr__(graph, "_native_source_frame", frame)
    object.__setattr__(graph, "_ep_source_binding", frame)
    object.__setattr__(graph, "_is_node_binding", True)
    object.__setattr__(graph, "_nodes_by_id", _SourceFrameLookup(frame))
    return graph


def _native_recipe_signature(program):
    ranks = (id(program.rank_programs), tuple(weakref.ref(rank) for rank in program.rank_programs))
    return (weakref.ref(program.graph), ranks, tuple((rank.node_bindings,rank.allocation_bindings,
            rank.view_bindings,rank.capture_bindings) for rank in program.rank_programs),
        program.collective_attribute_bindings,program.collective_geometry_bindings,
        program.__dict__["_retained_full_execution_context"] if "_retained_full_execution_context" in program.__dict__ else None)


def _native_recipe_matches(program, signature):
    graph_reference,ranks,bindings,attrs,geometry,context = signature
    return (program.graph is graph_reference() and id(program.rank_programs) == ranks[0]
        and len(program.rank_programs) == len(ranks[1])
        and all(rank is reference() for rank, reference in zip(program.rank_programs, ranks[1], strict=True))
        and all(all(current is old for current,old in zip((rank.node_bindings,rank.allocation_bindings,
            rank.view_bindings,rank.capture_bindings),binding,strict=True)) for rank,binding in zip(program.rank_programs,bindings,strict=True))
        and program.collective_attribute_bindings is attrs and program.collective_geometry_bindings is geometry
        and (program.__dict__["_retained_full_execution_context"]
            if "_retained_full_execution_context" in program.__dict__ else None) is context)


def _native_capture_output_contract(group, trace, capture):
    """Prove a capture's own terminal typed port, without renaming its backing."""
    port = capture["output"]
    if set(port) != {"storage_id", "allocation_id", "shape", "dtype", "bytes"}:
        return None
    if (type(port["shape"]) is not tuple or not port["shape"]
            or any(type(size) is not int or size <= 0 for size in port["shape"])
            or type(port["dtype"]) is not str):
        raise ValueError("source capture output requires exact typed dimensions and dtype")
    name = port["allocation_id"]
    if (name not in trace._views or name not in trace._allocations
            or name not in trace._persistent_ids or name not in trace._writers):
        raise ValueError("source capture output requires its actual persistent backing")
    view, allocation = trace._views[name], trace._allocations[name]
    if type(view) is not TensorView or type(allocation) is not MemoryAllocation:
        raise TypeError("source capture output requires canonical typed storage")
    # Revalidate current submitted values, including bool offsets and rank.
    checked_view = replace(view)
    replace(allocation)
    writer = trace._writers[name]
    if not capture["model_node_ids"]:
        return None
    terminal = capture["model_node_ids"][-1]
    expected_writer = (group._submissions[terminal][trace.rank]
        if type(group) is ForwardCollectiveGroup and terminal in group._submissions else terminal)
    if (view.view_id != name or view.allocation_id != name
            or allocation.allocation_id != name or allocation.device_id != f"rank:{trace.rank}"
            or region_end(checked_view.region()) > allocation.size_bytes
            or port["storage_id"] != f"rank:{trace.rank}:{capture['key']}:hidden_states"
            or port["shape"] != view.shape or port["dtype"] != view.dtype
            or type(port["bytes"]) is not int or port["bytes"] != allocation.size_bytes
            or writer != expected_writer
            or writer not in trace._nodes
            or view.view_id not in trace._nodes[writer].attributes["symbolic_tensor_outputs"]):
        raise ValueError("source capture output disagrees with its terminal typed port")
    if (view.offset_bytes != 0 or prod(view.shape) * view.element_bytes != allocation.size_bytes
            or len(view.shape) != 2 or view.shape[0] != capture["model_input_tokens"]):
        return None
    stride = view.element_bytes
    for size, actual in zip(reversed(view.shape), reversed(view.strides_bytes), strict=True):
        if actual != stride:
            return None
        stride *= size
    return (port["storage_id"], view.shape, view.dtype, allocation.size_bytes,
        view.offset_bytes, view.strides_bytes, allocation.device_id)


def _native_source_scope(program, prefix):
    single = type(program) is ForwardSourceProgram
    if single:
        if (type(prefix) is not ForwardTrace
                or "_retained_full_execution_context" not in program.__dict__
                or program.rank_programs != (program,)):
            return False
    elif type(program) is not ForwardCollectiveSourceProgram or len(prefix.traces) != 2:
        return False
    traces = _source_frame_traces(prefix)
    dynamic = all(rank.geometry_rebind and (
            rank.source.source_captures[0]["runtime_mode"] == "PIECEWISE"
            or (rank.source.source_captures[0]["runtime_mode"] == "FULL"
                and rank.source._nodes[rank.source.qualify("layer-0:attention")].attributes[
                    "source_model_execution_version"] == "vllm_0_11_0"))
        for rank in program.rank_programs)
    if dynamic and any(trace.source_captures[0]["runtime_mode"] == "FULL"
            and _native_capture_output_contract(group, trace, trace.source_captures[0]) is None
            for group in (program.source, program.prefix, prefix) for trace in _source_frame_traces(group)):
        return False
    if not dynamic and (program.collective_geometry_bindings
            or any(rank.transient_origins or any(target not in rank.source._persistent_ids
                for target,origin in rank.allocation_bindings) for rank in program.rank_programs)):
        return False
    for index,trace in enumerate(traces):
        captures = trace.source_captures
        if (len(captures) != 1 or type(captures[0]["logical_tokens"]) is not int
                or captures[0]["logical_tokens"] <= 0
                or type(captures[0]["model_input_tokens"]) is not int
                or captures[0]["logical_tokens"] > captures[0]["model_input_tokens"]): return False
        if not dynamic and captures[0]["logical_tokens"] != captures[0]["model_input_tokens"]:
            capture = captures[0]
            exact_fields = ("key","logical_tokens","model_input_tokens","runtime_mode","source_stage","source_graph_batch_size","inputs")
            if (not set(exact_fields) <= set(capture) or capture["runtime_mode"] != "FULL"
                    or capture["source_stage"] is not None): return False
            output = _native_capture_output_contract(prefix, trace, capture)
            if output is None or capture["output"] != _source_frame_traces(program.prefix)[index].source_captures[0]["output"]:
                return False
            for group in (program.source, program.prefix):
                recorded = _source_frame_traces(group)[index]
                actual = recorded.source_captures
                if (len(actual) != 1 or recorded.rank != trace.rank or not set(exact_fields) <= set(actual[0])
                        or any(capture[name] != actual[0][name] for name in exact_fields)
                        or _native_capture_output_contract(group, recorded, actual[0]) != output): return False
    if "_native_frame_scope" in program.__dict__:
        signature, eligible = program.__dict__["_native_frame_scope"]
        if _native_recipe_matches(program,signature): return eligible
    attention = tuple(node for node in program.graph.nodes if node.kind is WorkloadKind.FLASH_ATTN)
    eligible = (bool(attention) and (dynamic or all(node.shape["batch"] > 0 and (node.shape["q_len"] == 1
            or ("mathematical_program" in node.attributes
                and node.attributes["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1"
                and "q_lens" in node.attributes and type(node.attributes["q_lens"]) is tuple
                and bool(node.attributes["q_lens"])
                and all(type(length) is int and length == 1 for length in node.attributes["q_lens"]))) for node in attention))
        and all(len(trace.source_captures) == 1 for trace in _source_frame_traces(program.source))
        and (not single or all(node.attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode"
            for node in attention)))
    program._native_frame_scope = (_native_recipe_signature(program),eligible)
    return eligible


def _source_bound_pool_names(graph, names):
    """Read every requested node's current pool without exporting its fields."""
    if "_native_source_frame" in graph.__dict__:
        frame = graph.__dict__["_native_source_frame"]
        if type(frame) is _NativeSourceFrame and frame.owner() is not None:
            return {frame.pool_name(name) for name in names}
    return {graph.node(name).pool_name for name in names}


def _source_bound_topology(graph):
    """Private topology identity for consumers of a checked native frame."""
    if "_native_source_frame" in graph.__dict__:
        frame = graph.__dict__["_native_source_frame"]
        if type(frame) is _NativeSourceFrame and frame.owner() is not None:
            return frame.workload_contract[1:3]
    return graph.edges, graph.block_repeats


def _source_bound_execution_projection(graph):
    if "_native_source_frame" not in graph.__dict__:
        if "_glm_decode_binding" not in graph.__dict__:
            return None
        from .traced_models import _decode_source_execution_projection
        return _decode_source_execution_projection(graph)
    frame = graph.__dict__["_native_source_frame"]
    if type(frame) is not _NativeSourceFrame or frame.owner() is None: return None
    routing = []
    for name in frame.routing_ids:
        _, attrs = frame.fields(name)
        routing.append((name,attrs["moe_invocation_id"],attrs["tokens_per_expert"],attrs["routing_provenance"]))
    return frame.workload_contract, tuple(routing)


def _source_bound_storage_projection(source, graph):
    if "_native_source_frame" not in graph.__dict__: return None
    frame = graph.__dict__["_native_source_frame"]
    traces = _source_frame_traces(source)
    recorded = source._recorded_graph[1] if type(source) is ForwardTrace else source._recorded_graph
    if (type(frame) is not _NativeSourceFrame or frame.owner() is None or recorded is not graph
            or len(traces) != len(frame.trace_references) or frame.storage_rows is None
            or any(trace is not reference() for trace,reference in zip(traces,frame.trace_references,strict=True))): return None
    rows = tuple((row[0], _source_activation_allocations(trace), len(trace._views),
        MappingProxyType(trace.workspace_capacity_bytes))
        for row, trace in zip(frame.storage_rows, traces, strict=True))
    return len(frame.ids), len(frame.workload_contract[1]), rows


def _source_bound_node_sequence(graph):
    if "_native_source_frame" in graph.__dict__:
        frame = graph.__dict__["_native_source_frame"]
        if type(frame) is _NativeSourceFrame and frame.owner() is not None: return frame.nodes
    return graph.nodes


def _bind_native_source_parameters(program, rank_values, collective_values):
    """Execute the already validated source recipe on actual current fields."""
    traces = []
    for rank_program, (updates, allocations, views) in zip(program.rank_programs, rank_values, strict=True):
        result = copy(rank_program.source)
        result._nodes = _source_rank_frame(rank_program, updates)
        if result._nodes is None:
            return None
        result._recorded_graph = (result._nodes, None)
        if not _bind_native_storage(rank_program, result, allocations, views):
            result._allocations, result._views = dict(rank_program.source._allocations), dict(rank_program.source._views)
            rank_program._bind_storage_parameters(result, allocations, views, rank_program.prefix.source_captures)
            _seal_native_bound_storage(rank_program, result)
        result._captures = [_copy_capture_dicts(capture) for capture in result.source_captures]
        traces.append(result)
    if type(program) is ForwardSourceProgram:
        if collective_values or len(traces) != 1:
            raise ValueError("rank-local source parameters cannot contain collective updates")
        result, = traces
        result._recorded_graph = (result._nodes, _issue_source_frame(program, result, {}))
        return result
    result = copy(program.source)
    result.traces = tuple(traces)
    result._layouts = dict(program.source._layouts)
    result._contracts = _SourceFrameNodes(MappingProxyType(dict(program.source._contracts)), collective_values, None)
    joined = _SourceFrameNodes(MappingProxyType({name:program.graph.node(name) for name in collective_values}), collective_values, None)
    result._recorded_graph = _issue_source_frame(program, result, joined)
    return result


# Issued frames read an already recorded graph. Builders retain the complete
# traced_models source guard; consumers validate only their execution surface.
def _native_consumer_source_unchanged():
    if not _node_update_source_unchanged(): return False
    for name, cls, bases, keys, reader, values, identical in _NATIVE_CONSUMER_CLASSES:
        if globals()[name] is not cls or cls.__bases__ != bases: return False
        namespace = vars(cls)
        if cls in (ForwardTrace, ForwardCollectiveGroup, MemoryAllocation) and "__slotnames__" in namespace:
            if type(namespace["__slotnames__"]) is not list or namespace["__slotnames__"]: return False
            current_keys = namespace.keys() - {"__slotnames__"}
        else:
            current_keys = namespace.keys()
        if current_keys != keys: return False
        # The complete key-set check above also proves every declared absent
        # member stays absent. Read present members in C, then compare identity;
        # equality-overloaded descriptors cannot satisfy this source guard.
        if len(values) == 1:
            if reader(namespace) is not values[0]: return False
        elif values and not all(map(identical, reader(namespace), values)):
            return False
    valid = (all(globals()[name] is function for name, function in _NATIVE_CONSUMER_GLOBALS)
        and all(function.__code__ is code and function.__defaults__ == defaults
            and function.__kwdefaults__ == kwdefaults
            for function, code, defaults, kwdefaults in _NATIVE_CONSUMER_FUNCTIONS))
    return valid


def _native_consumer_snapshot():
    from types import FunctionType
    declarations = (
        ('_NativeSourceFrame', tuple(vars(_NativeSourceFrame))),
        ('_SourceFrameNodes', tuple(vars(_SourceFrameNodes))),
        ('_SourceParameterSlots', tuple(vars(_SourceParameterSlots))),
        ('_SourceFrameSequence', tuple(vars(_SourceFrameSequence))),
        ('_SourceFrameLookup', tuple(vars(_SourceFrameLookup))),
        ('_OverlayFields', tuple(vars(_OverlayFields))),
        ('_NativeStorageOverlay', tuple(vars(_NativeStorageOverlay))),
        ('WorkloadGraph', tuple(vars(WorkloadGraph))),
        ('WorkloadNode', tuple(vars(WorkloadNode))),
        ('MemoryAllocation', tuple(vars(MemoryAllocation))),
        ('Mapping', tuple(vars(Mapping))), ('Sequence', tuple(vars(Sequence))),
        ('ForwardTrace', ('source_capture', 'source_captures', 'workspace_capacity_bytes',
            'activation_allocations', 'event_order', 'allocations', 'views', 'graph',
            '_nodes', '_allocations', '_persistent_ids', '_events', '_views', '_captures', '_capture', '_recorded_graph')),
        ('ForwardCollectiveGroup', ('graph', '_recorded_graph', 'traces')),
        ('ForwardSourceProgram', ('graph', 'source', 'rank_programs', 'node_bindings', 'allocation_bindings', 'view_bindings', 'capture_bindings', 'collective_attribute_bindings', 'collective_geometry_bindings')),
        ('ForwardCollectiveSourceProgram', ('graph', 'source', 'rank_programs', 'node_bindings', 'allocation_bindings', 'view_bindings', 'capture_bindings', 'collective_attribute_bindings', 'collective_geometry_bindings')))
    classes, functions = [], []
    for name, names in declarations:
        cls = globals()[name]; namespace = vars(cls)
        names = tuple(dict.fromkeys((*names, '__getattribute__', '__getattr__', '__setattr__')))
        if cls in (ForwardTrace, ForwardCollectiveGroup, MemoryAllocation):
            # Pickle may have populated this empty CPython cache before import.
            names = tuple(key for key in names if key != "__slotnames__")
        members = tuple((key, key in namespace, namespace[key] if key in namespace else None) for key in names)
        keys = frozenset(key for key in namespace if key != "__slotnames__"
            or cls not in (ForwardTrace, ForwardCollectiveGroup, MemoryAllocation))
        present_names = tuple(key for key, present, _ in members if present)
        values = tuple(value for _, present, value in members if present)
        reader = operator.itemgetter(*present_names) if present_names else None
        classes.append((name, cls, cls.__bases__, keys, reader, values, operator.is_))
        for _, present, member in members:
            if present and type(member) is FunctionType: functions.append(member)
            elif present and type(member) is property:
                functions.extend(function for function in (member.fget, member.fset, member.fdel) if function is not None)
    names = ('_source_frame_traces', '_native_consumer_source_unchanged', '_native_recipe_matches', '_source_bound_node_layout',
        '_source_bound_execution_projection', '_source_bound_storage_projection', '_source_bound_node_sequence',
        '_source_activation_allocations', '_source_bound_topology', '_source_bound_pool_names',
        '_ep_source_owner', '_ep_source_token_domains', '_node_update_source_unchanged')
    global_functions = tuple((name, globals()[name]) for name in names)
    functions.extend(function for _, function in global_functions)
    return (tuple(classes), global_functions, tuple((function, function.__code__, function.__defaults__,
        None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)) for function in functions))


_NATIVE_CONSUMER_CLASSES, _NATIVE_CONSUMER_GLOBALS, _NATIVE_CONSUMER_FUNCTIONS = _native_consumer_snapshot()
