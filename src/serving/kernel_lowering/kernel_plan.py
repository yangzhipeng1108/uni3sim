"""Structured kernel calls and dependencies after workload lowering."""

from dataclasses import dataclass, field, fields, replace
from typing import Iterable, Iterator, Tuple
from collections.abc import Mapping
from types import MappingProxyType

from src.kernel.kernel_spec import KernelSpec, PhysicalTransferConfig
from src.kernel.launch_config import LaunchConfig
from src.kernel.memory_access_builder import (
    CacheUpdateBindings, ElementwiseBindings, FusedMoEBindings, MemoryBindings, SparseAttentionBindings,
)
from .memory_bindings import SourceCallMemory, TraceMemoryContext




@dataclass(frozen=True)
class KernelCall:
    call_id: str
    kernel_spec: KernelSpec
    pool_name: str
    rank: int
    resource: str
    source_node_ids: Tuple[str, ...]
    dependencies: Tuple[str, ...] = ()
    memory_bindings: MemoryBindings | ElementwiseBindings | CacheUpdateBindings | SparseAttentionBindings | FusedMoEBindings | None = None
    memory_source: SourceCallMemory | None = None
    memory_unbound_reason: str | None = None
    source_launches: Tuple[Tuple[str, LaunchConfig | PhysicalTransferConfig], ...] | None = None

    def __post_init__(self) -> None:
        if self.source_launches is not None:
            if not isinstance(self.source_launches, tuple) or any(
                    not isinstance(item, tuple) or len(item) != 2
                    or type(item[0]) is not str or not item[0]
                    or not isinstance(item[1], (LaunchConfig, PhysicalTransferConfig)) for item in self.source_launches):
                raise TypeError("source launches require immutable named launch/transfer descriptors")
            roles = tuple(role for role, _ in self.source_launches)
            if len(set(roles)) != len(roles):
                raise ValueError("source launch stage roles must be unique within a logical call")
        for name in ("call_id", "pool_name", "resource"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"kernel call {name} must be a non-empty string")
        if not isinstance(self.kernel_spec, KernelSpec):
            raise TypeError("kernel call requires a KernelSpec")
        if self.memory_source is not None:
            if type(self.memory_source) is not SourceCallMemory or self.memory_source.rank != self.rank:
                raise ValueError("kernel memory provenance requires its source rank")
            if self.source_node_ids != (self.memory_source.node_id,):
                raise ValueError("kernel memory provenance requires its actual source node")
        if self.memory_bindings is not None:
            if not isinstance(self.memory_bindings, (MemoryBindings, ElementwiseBindings, CacheUpdateBindings, SparseAttentionBindings, FusedMoEBindings)):
                raise TypeError("kernel memory binding requires a typed source contract")
            if (self.memory_source is None or self.memory_bindings.call_id != self.call_id
                    or self.memory_bindings.rank != self.rank or self.memory_unbound_reason is not None):
                raise ValueError("kernel memory binding identity or provenance mismatch")
        if self.memory_unbound_reason is not None and (type(self.memory_unbound_reason) is not str or not self.memory_unbound_reason):
            raise ValueError("unavailable memory bindings require an explicit reason")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("kernel call rank must be an integer")
        if self.rank < 0:
            raise ValueError("kernel call rank cannot be negative")
        if not isinstance(self.source_node_ids, (list, tuple)):
            raise TypeError(
                "kernel call source_node_ids must be a list or tuple")
        object.__setattr__(self, "source_node_ids", tuple(self.source_node_ids))
        if (not self.source_node_ids
                or any(not isinstance(node_id, str) or not node_id
                       for node_id in self.source_node_ids)):
            raise ValueError(
                "kernel call source_node_ids must contain non-empty strings")
        if len(self.source_node_ids) != len(set(self.source_node_ids)):
            raise ValueError("kernel call source_node_ids must be unique")
        if not isinstance(self.dependencies, (list, tuple)):
            raise TypeError("kernel call dependencies must be a list or tuple")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if any(not isinstance(dependency, str) or not dependency
               for dependency in self.dependencies):
            raise ValueError(
                "kernel call dependencies must contain non-empty strings")


    def with_dependencies(self, dependencies):
        """Bind current resource predecessors to an already validated call."""
        if type(self) is not KernelCall or type(dependencies) is not tuple:
            raise TypeError("dependency binding requires a canonical call and immutable tuple")
        if any(not isinstance(name, str) or not name or name == self.call_id for name in dependencies):
            raise ValueError("kernel dependencies require preceding call identities")
        if dependencies == self.dependencies:
            return self
        result = object.__new__(KernelCall)
        result.__dict__.update(self.__dict__)
        object.__setattr__(result, "dependencies", dependencies)
        return result

    def bind_source_call(self, call_id, source_node_id, dependencies):
        """Instantiate a compiled one-node lowering recipe at its actual ports."""
        if (type(self) is not KernelCall or self.memory_source is not None
                or self.memory_bindings is not None or self.memory_unbound_reason is not None
                or len(self.source_node_ids) != 1):
            raise ValueError("source call recipes require one unbound canonical source node")
        if (not isinstance(call_id, str) or not call_id or not isinstance(source_node_id, str) or not source_node_id
                or type(dependencies) is not tuple
                or any(not isinstance(name, str) or not name or name == call_id for name in dependencies)):
            raise ValueError("source call binding requires current IDs and predecessor ports")
        result = object.__new__(KernelCall)
        result.__dict__.update(self.__dict__)
        object.__setattr__(result, "call_id", call_id)
        object.__setattr__(result, "source_node_ids", (source_node_id,))
        object.__setattr__(result, "dependencies", dependencies)
        return result

    def with_kernel_spec(self, kernel_spec):
        """Bind a validated spec without changing this call's execution lane."""
        if type(self) is not KernelCall or type(kernel_spec) is not KernelSpec:
            raise TypeError("compiled call binding requires canonical KernelCall and KernelSpec")
        if kernel_spec is self.kernel_spec:
            return self
        if self.source_launches is not None and any(
                getattr(kernel_spec, name) != getattr(self.kernel_spec, name)
                for name in ("shape", "dtype", "variant", "attributes")):
            raise ValueError("changing source-bound call geometry requires reselecting its source launches")
        from src.kernel.kernel_spec import KernelKind
        if kernel_spec.transfer is None:
            resource = "compute"
        elif kernel_spec.kind is KernelKind.COMMUNICATION:
            resource = kernel_spec.transfer.resource_id
        else:
            resource = kernel_spec.transfer.link
        if (kernel_spec.kind is not self.kernel_spec.kind
                or kernel_spec.hardware_id != self.kernel_spec.hardware_id
                or resource != self.resource):
            raise ValueError("spec binding changes call structure; compile a corresponding plan")
        old_transfer = self.kernel_spec.transfer
        new_transfer = kernel_spec.transfer
        if (old_transfer is None) != (new_transfer is None):
            raise ValueError("spec binding changes transfer structure; compile a corresponding plan")
        if old_transfer is not None:
            # Payload volume may change with a bound tensor shape. The
            # algorithm, participants, NICs and all other lane facts may not.
            if type(new_transfer) is not type(old_transfer) or any(
                    getattr(new_transfer, item.name) != getattr(old_transfer, item.name)
                    for item in fields(old_transfer) if item.name != "bytes"):
                raise ValueError("spec binding changes transfer topology; compile a corresponding plan")
        result = object.__new__(KernelCall)
        result.__dict__.update(self.__dict__)
        object.__setattr__(result, "kernel_spec", kernel_spec)
        if self.memory_source is not None:
            return TraceMemoryContext((self.memory_source,)).bind_call(result)
        return result


@dataclass(frozen=True)
class KernelBlockRepeat:
    """One predicted block plan with recurrence dependencies."""

    block_id: str
    call_ids: Tuple[str, ...]
    repeat_count: int
    feedback_dependencies: Tuple[Tuple[str, Tuple[str, ...]], ...]
    trace_node_ids: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("kernel block repeat ID must be non-empty")
        if not isinstance(self.call_ids, (list, tuple)):
            raise TypeError("kernel block repeat call IDs must be a sequence")
        call_ids = tuple(self.call_ids)
        if (not call_ids
                or any(not isinstance(call_id, str) or not call_id
                       for call_id in call_ids)):
            raise ValueError(
                "kernel block repeat call IDs must be non-empty strings")
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("kernel block repeat call IDs must be unique")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)
                or self.repeat_count < 2):
            raise ValueError("kernel block repeat count must be an integer >= 2")
        if not isinstance(self.feedback_dependencies, (list, tuple)):
            raise TypeError(
                "kernel block feedback dependencies must be a sequence")
        feedback_dependencies = tuple(
            (destination, tuple(sources))
            for destination, sources in self.feedback_dependencies
        )
        known = set(call_ids)
        destinations = []
        for destination, sources in feedback_dependencies:
            if destination not in known:
                raise ValueError(
                    "kernel block feedback destination is outside the block")
            if (not sources
                    or any(source not in known for source in sources)):
                raise ValueError(
                    "kernel block feedback sources must be inside the block")
            destinations.append(destination)
        if len(destinations) != len(set(destinations)):
            raise ValueError(
                "kernel block feedback destinations must be unique")
        object.__setattr__(self, "call_ids", call_ids)
        object.__setattr__(
            self, "feedback_dependencies", feedback_dependencies)
        if not isinstance(self.trace_node_ids, (tuple, list)):
            raise TypeError("kernel repeat source trace must be a sequence")
        traces = tuple((name, tuple(occurrences)) for name, occurrences in self.trace_node_ids)
        names, logical = [], []
        for name, occurrences in traces:
            if (not isinstance(name, str) or not name or len(occurrences) != self.repeat_count
                    or occurrences[0] != name
                    or any(not isinstance(value, str) or not value for value in occurrences)):
                raise ValueError("kernel repeat source trace must cover every actual occurrence")
            names.append(name)
            logical.extend(occurrences)
        if len(names) != len(set(names)) or len(logical) != len(set(logical)):
            raise ValueError("kernel repeat source trace identities must be unique")
        object.__setattr__(self, "trace_node_ids", traces)


@dataclass(frozen=True)
class KernelPlan:
    calls: Tuple[KernelCall, ...]
    block_repeats: Tuple[KernelBlockRepeat, ...]
    memory_context: TraceMemoryContext | None = None
    _topological_calls: Tuple[KernelCall, ...] = field(
        init=False, repr=False, compare=False)

    def __init__(self, calls: Iterable[KernelCall],
                 block_repeats: Iterable[KernelBlockRepeat] = (), *,
                 memory_context: TraceMemoryContext | None = None):
        if memory_context is not None and type(memory_context) is not TraceMemoryContext:
            raise TypeError("kernel plan memory context must be a trace receipt")
        object.__setattr__(self, "memory_context", memory_context)
        call_tuple = tuple(calls)
        block_repeat_tuple = tuple(block_repeats)
        if not call_tuple:
            raise ValueError("KernelPlan requires at least one kernel call")
        if any(not isinstance(call, KernelCall) for call in call_tuple):
            raise TypeError("KernelPlan requires KernelCall objects")
        ids = [call.call_id for call in call_tuple]
        if len(ids) != len(set(ids)):
            raise ValueError("kernel call IDs must be unique")
        known = set(ids)
        calls_by_id = {
            call.call_id: call for call in call_tuple
        }
        for call in call_tuple:
            unknown = set(call.dependencies) - known
            if unknown:
                raise ValueError(f"kernel call has unknown dependencies: {sorted(unknown)}")
            if call.call_id in call.dependencies:
                raise ValueError("kernel calls cannot depend on themselves")
        if any(
                not isinstance(repeat, KernelBlockRepeat)
                for repeat in block_repeat_tuple):
            raise TypeError(
                "KernelPlan block repeats require KernelBlockRepeat objects")
        block_ids = [repeat.block_id for repeat in block_repeat_tuple]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("kernel block repeat IDs must be unique")
        repeated_call_ids = []
        for repeat in block_repeat_tuple:
            unknown = set(repeat.call_ids) - known
            if unknown:
                raise ValueError(
                    "kernel block repeat references unknown calls: "
                    f"{sorted(unknown)}")
            repeated_call_ids.extend(repeat.call_ids)
            if repeat.trace_node_ids:
                template_sources = {source for call_id in repeat.call_ids
                    for source in calls_by_id[call_id].source_node_ids}
                if {name for name, _ in repeat.trace_node_ids} != template_sources:
                    raise ValueError("kernel repeat source trace must cover every template source")
            feedback_by_destination = {
                destination: sources
                for destination, sources in repeat.feedback_dependencies
            }
            repeated = set(repeat.call_ids)
            for call_id in repeat.call_ids:
                call = calls_by_id[call_id]
                external_dependencies = tuple(
                    dependency for dependency in call.dependencies
                    if dependency not in repeated
                )
                if external_dependencies and call_id not in feedback_by_destination:
                    raise ValueError(
                        "kernel block external dependency lacks feedback: "
                        f"{call_id!r}")
                if (not external_dependencies
                        and call_id in feedback_by_destination):
                    raise ValueError(
                        "kernel block feedback has no first-iteration "
                        f"dependency: {call_id!r}")
        logical_sources = tuple(source for repeat in block_repeat_tuple
            for _, occurrences in repeat.trace_node_ids for source in occurrences)
        repeated = set(repeated_call_ids)
        unrepeated_sources = {source for call in call_tuple if call.call_id not in repeated
            for source in call.source_node_ids}
        if (len(logical_sources) != len(set(logical_sources))
                or set(logical_sources).intersection(unrepeated_sources)):
            raise ValueError("kernel repeat source occurrences overlap another source region")
        if len(repeated_call_ids) != len(set(repeated_call_ids)):
            raise ValueError(
                "kernel calls cannot belong to multiple block repeats")
        object.__setattr__(self, "calls", call_tuple)
        object.__setattr__(self, "block_repeats", block_repeat_tuple)
        object.__setattr__(
            self, "_topological_calls", self._compute_topological_order())

    @property
    def logical_call_count(self) -> int:
        return len(self.calls) + sum(
            (repeat.repeat_count - 1) * len(repeat.call_ids)
            for repeat in self.block_repeats
        )

    @property
    def memory_binding_coverage(self):
        """Explicit coverage of source-owned bindings, including folded calls."""
        if self.memory_context is None:
            return {"bound_logical_calls": 0, "total_logical_calls": self.logical_call_count,
                "complete": self.logical_call_count == 0,
                "unavailable": {"source_trace_not_supplied": self.logical_call_count}}
        bound, unavailable = 0, {}
        # Count actual occurrences: a representative cannot prove that every
        # folded layer has the same source port geometry or storage mapping.
        for call in self.logical_calls():
            if call.memory_bindings is not None:
                bound += 1
            else:
                reason = call.memory_unbound_reason
                if reason is None:
                    raise ValueError("source memory coverage requires an explicit unavailable reason")
                unavailable[reason] = (unavailable[reason] if reason in unavailable else 0) + 1
        return {"bound_logical_calls": bound, "total_logical_calls": self.logical_call_count,
            "complete": bound == self.logical_call_count, "unavailable": unavailable}

    def logical_calls(self):
        """Yield complete source-derived calls without expanding normal summaries.

        Legacy hand-authored repeats may omit source traces; callers asking
        for actual source identities must then receive an explicit refusal.
        """
        by_id = {call.call_id: call for call in self.calls}
        owners, source_maps, feedback = {}, {}, {}
        for repeat in self.block_repeats:
            if not repeat.trace_node_ids:
                raise NotImplementedError("logical source calls require repeat trace_node_ids")
            source_maps[repeat.block_id] = dict(repeat.trace_node_ids)
            feedback[repeat.block_id] = dict(repeat.feedback_dependencies)
            for name in repeat.call_ids:
                owners[name] = repeat

        def identity(name, occurrence):
            call = by_id[name]
            repeat = owners[name]
            primary = call.source_node_ids[0]
            if not call.call_id.startswith(primary):
                raise NotImplementedError("source call identity is not derived from its primary node")
            return source_maps[repeat.block_id][primary][occurrence] + call.call_id[len(primary):]

        def final_identity(name):
            return identity(name, owners[name].repeat_count - 1) if name in owners else name

        emitted = set()
        for call in self.calls:
            if call.call_id not in owners:
                yield replace(call, dependencies=tuple(final_identity(name) for name in call.dependencies))
                continue
            repeat = owners[call.call_id]
            if repeat.block_id in emitted:
                continue
            emitted.add(repeat.block_id)
            members = tuple(by_id[name] for name in repeat.call_ids)
            for occurrence in range(repeat.repeat_count):
                for member in members:
                    dependencies = []
                    inserted_feedback = False
                    for name in member.dependencies:
                        if name in owners and owners[name] is repeat:
                            dependencies.append(identity(name, occurrence))
                        elif occurrence == 0:
                            dependencies.append(final_identity(name))
                        elif not inserted_feedback:
                            dependencies.extend(identity(previous, occurrence - 1)
                                for previous in feedback[repeat.block_id][member.call_id])
                            inserted_feedback = True
                    logical = KernelCall(identity(member.call_id, occurrence), member.kernel_spec,
                        member.pool_name, member.rank, member.resource,
                        tuple(source_maps[repeat.block_id][name][occurrence] for name in member.source_node_ids),
                        tuple(dependencies), source_launches=member.source_launches)
                    if self.memory_context is not None:
                        logical = self.memory_context.bind_call(logical)
                    elif member.memory_source is not None or member.memory_bindings is not None:
                        raise ValueError("repeated memory bindings require actual occurrence source receipts")
                    yield logical

    def __iter__(self) -> Iterator[KernelCall]:
        return iter(self.calls)

    def topological_order(self) -> Tuple[KernelCall, ...]:
        return self._topological_calls

    def bind_kernel_specs(self, updates, source_launches=None):
        """Reuse this immutable DAG while binding current validated specs.

        Structural branch selection belongs to the compiled source program.
        This operation cannot add calls, move hardware/resources, alter
        source identities, or change dependencies/repeat layout.
        """
        if type(self) is not KernelPlan or not isinstance(updates, Mapping):
            raise TypeError("compiled plan binding requires KernelPlan and a named spec mapping")
        updates = dict(updates)
        if any(type(name) is not str for name in updates):
            raise TypeError("compiled spec binding names must be native strings")
        if source_launches is not None:
            if not isinstance(source_launches, Mapping) or not set(source_launches).issubset(updates):
                raise ValueError("reselected source launches require corresponding current specs")
        if "_binding_positions" not in self.__dict__:
            if any(type(call) is not KernelCall or type(call.call_id) is not str for call in self.calls):
                raise TypeError("compiled plan binding requires canonical named calls")
            positions = {call.call_id: index for index, call in enumerate(self.calls)}
            order = tuple(positions[call.call_id] for call in self._topological_calls)
            object.__setattr__(self, "_binding_positions", MappingProxyType(positions))
            object.__setattr__(self, "_binding_topological_indices", order)
        positions = self._binding_positions
        unknown = set(updates) - positions.keys()
        if unknown:
            raise ValueError(f"unknown compiled call bindings: {sorted(unknown)}")
        if "_schedule_structure_token" not in self.__dict__:
            canonical = (all(type(call.kernel_spec) is KernelSpec
                and (call.kernel_spec.transfer is None or type(call.kernel_spec.transfer) is PhysicalTransferConfig)
                for call in self.calls) and all(type(repeat) is KernelBlockRepeat for repeat in self.block_repeats))
            object.__setattr__(self, "_schedule_structure_token", object() if canonical else None)
        changed = {}
        for name, spec in updates.items():
            index = positions[name]
            old = self.calls[index]
            if source_launches is not None and name in source_launches:
                # The caller has reselected the actual source stages. Keep
                # the existing lane/transfer checks while replacing the old
                # stage descriptors; never retain a stale bound launch.
                unbound = old if old.source_launches is None else replace(old, source_launches=None)
                call = replace(unbound.with_kernel_spec(spec), source_launches=source_launches[name])
            else:
                call = old.with_kernel_spec(spec)
            if call is not self.calls[index]:
                changed[index] = call
        if not changed:
            return self
        calls = tuple(changed[index] if index in changed else call
            for index, call in enumerate(self.calls))
        result = object.__new__(KernelPlan)
        result.__dict__.update(self.__dict__)
        if (self._schedule_structure_token is not None
                and any(spec.variant != self.calls[positions[name]].kernel_spec.variant for name, spec in updates.items())):
            # Kind, hardware and transfer topology are checked by
            # with_kernel_spec(). Variant is also part of schedule structure.
            object.__setattr__(result, "_schedule_structure_token", object())
        object.__setattr__(result, "calls", calls)
        object.__setattr__(result, "_topological_calls", tuple(calls[index]
            for index in self._binding_topological_indices))
        previous = _bound_kernel_spec_projection(self)
        template, inherited = (self, ()) if previous is None else previous
        object.__setattr__(result, "_bound_spec_projection", _BoundSpecProjection(
            result, template, tuple(sorted(set(inherited) | changed.keys()))))
        return result

    def _compute_topological_order(self) -> Tuple[KernelCall, ...]:
        position = {
            call.call_id: index for index, call in enumerate(self.calls)
        }
        by_id = {call.call_id: call for call in self.calls}
        indegree = {
            call.call_id: len(set(call.dependencies)) for call in self.calls
        }
        successors = {call.call_id: [] for call in self.calls}
        for call in self.calls:
            for dependency in set(call.dependencies):
                successors[dependency].append(call.call_id)
        ready = [
            call.call_id for call in self.calls if indegree[call.call_id] == 0
        ]
        ordered = []
        while ready:
            next_ready = []
            for call_id in ready:
                ordered.append(by_id[call_id])
                for successor in successors[call_id]:
                    indegree[successor] -= 1
                    if indegree[successor] == 0:
                        next_ready.append(successor)
            ready = sorted(next_ready, key=position.__getitem__)
        if len(ordered) != len(self.calls):
            raise ValueError("KernelPlan dependencies must be acyclic")
        return tuple(ordered)


def hardware_for_call(call: KernelCall, hardware_by_id):
    """Return the HWSpec one planned call resolves to, or fail loud.

    A plan whose call names a hardware the caller supplied no spec for cannot
    be priced; the failure names the missing hardware rather than defaulting.
    """

    return require_hardware(call.kernel_spec.hardware_id, hardware_by_id)


def require_hardware(hardware_id: str, hardware_by_id):
    """Return the HWSpec for one hardware id, or fail loud when absent."""

    if hardware_id not in hardware_by_id:
        raise ValueError(f"no HWSpec supplied for {hardware_id}")
    return hardware_by_id[hardware_id]


class _BoundSpecProjection:
    """Run-local changed slots in the same immutable plan body."""
    __slots__ = ("owner_id", "calls", "topological_calls", "template", "base_calls", "repeats", "positions")

    def __init__(self, plan, template, positions):
        for name, value in (("owner_id", id(plan)), ("calls", plan.calls),
                ("topological_calls", plan._topological_calls),
                ("template", template), ("base_calls", template.calls),
                ("repeats", plan.block_repeats), ("positions", positions)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise TypeError("bound spec projections are immutable")

    def __reduce_ex__(self, protocol):
        # Transport preserves the public plan, not this process's owner proof.
        return type(None), ()


def _bound_kernel_spec_projection(plan):
    """Return the checked template/changed slots, or use complete consumers."""
    if (type(plan) is not KernelPlan or type(plan.calls) is not tuple
            or "__getattribute__" in vars(KernelPlan)
            or any(getattr(owner, name) is not function or function.__code__ is not code
                for owner, name, function, code in _SPEC_BINDING_SOURCE)):
        return None
    binding = plan.__dict__["_bound_spec_projection"] if "_bound_spec_projection" in plan.__dict__ else None
    if (type(binding) is not _BoundSpecProjection or binding.owner_id != id(plan)
            or binding.calls is not plan.calls or binding.topological_calls is not plan._topological_calls
            or binding.repeats is not plan.block_repeats
            or type(binding.template) is not KernelPlan
            or type(binding.template.calls) is not tuple
            or binding.base_calls is not binding.template.calls
            or binding.template.block_repeats is not binding.repeats):
        return None
    return binding.template, binding.positions


_SPEC_BINDING_SOURCE = tuple((owner, name, getattr(owner, name), getattr(owner, name).__code__)
    for owner, name in ((KernelPlan, "bind_kernel_specs"), (KernelCall, "with_kernel_spec")))
