"""Public KernelPlan prediction entry point."""

from collections import OrderedDict
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from typing import Mapping

from src.kernel.kernel_base import predict, prediction_equivalence_fields
from src.kernel.kernel_spec import KernelKind, KernelPrediction, KernelSpec
from src.kernel.memory_access_builder import (memory_prediction_source_epoch, memory_prediction_source_scope,
    prediction_memory_program_reuse, bound_prediction_numeric_reuse_key)

from .kernel_plan import KernelCall, KernelPlan, hardware_for_call, _bound_kernel_spec_projection


_NONE_CONTRACT = ("none",)
# One frozen contract shape per concrete type, resolved once by the same
# ordered predicate chain the shapes are defined by.  The chain is a function
# of the type alone, so caching its answer cannot change any frozen contract.
_CONTRACT_FREEZERS: dict = {}
_CONTRACT_DATACLASS_FIELD_NAMES: dict = {}
_PLAN_GROUP_ORDER_SOURCE = (KernelPlan.topological_order, KernelPlan.topological_order.__code__)
_PLAN_GROUP_HARDWARE_SOURCE = (hardware_for_call, hardware_for_call.__code__)


def _freeze_none(value):
    del value
    return _NONE_CONTRACT


def _freeze_enum(value):
    kind = type(value)
    return (
        "enum",
        kind.__module__,
        kind.__qualname__,
        freeze_contract_value(value.value),
    )


def _freeze_bool(value):
    return ("bool", value)


def _freeze_int(value):
    return ("int", value)


def _freeze_float(value):
    return ("float", value.hex())


def _freeze_str(value):
    return ("str", value)


def _freeze_mapping(value):
    keys = tuple(value)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError(
            "kernel prediction memo mappings require string keys")
    return (
        "mapping",
        tuple(
            (key, freeze_contract_value(value[key]))
            for key in sorted(keys)
        ),
    )


def _freeze_tuple(value):
    # Native shape/index tuples need no per-element type dispatch. Subclasses
    # retain ordinary iteration and mixed values retain their exact type tags.
    if type(value) is tuple and all(type(item) is int for item in value):
        return ("tuple", tuple(("int", item) for item in value))
    return ("tuple", tuple(freeze_contract_value(item) for item in value))


def _freeze_list(value):
    return ("list", tuple(freeze_contract_value(item) for item in value))


def _freeze_dataclass(value):
    kind = type(value)
    if kind in _CONTRACT_DATACLASS_FIELD_NAMES:
        names = _CONTRACT_DATACLASS_FIELD_NAMES[kind]
    else:
        names = tuple(field.name for field in fields(value))
        _CONTRACT_DATACLASS_FIELD_NAMES[kind] = names
    return (
        "dataclass",
        kind.__module__,
        kind.__qualname__,
        tuple(
            (name, freeze_contract_value(getattr(value, name)))
            for name in names
        ),
    )


def _resolve_contract_freezer(value):
    """Pick the frozen contract shape for one concrete type, in chain order."""

    if value is None:
        return _freeze_none
    if isinstance(value, Enum):
        return _freeze_enum
    if isinstance(value, bool):
        return _freeze_bool
    if isinstance(value, int):
        return _freeze_int
    if isinstance(value, float):
        return _freeze_float
    if isinstance(value, str):
        return _freeze_str
    if isinstance(value, Mapping):
        return _freeze_mapping
    if isinstance(value, tuple):
        return _freeze_tuple
    if isinstance(value, list):
        return _freeze_list
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze_dataclass
    raise TypeError(
        "kernel prediction memo cannot freeze contract value of type "
        f"{type(value).__name__}")


def freeze_contract_value(value):
    """Build an exact hash key from an immutable production contract."""

    kind = type(value)
    if kind in _CONTRACT_FREEZERS:
        return _CONTRACT_FREEZERS[kind](value)
    freezer = _resolve_contract_freezer(value)
    _CONTRACT_FREEZERS[kind] = freezer
    return freezer(value)


def _contract_cache_key(value):
    """Return the exact immutable contract used for run-local reuse."""

    return freeze_contract_value(value)


def _prediction_class_key(kernel_spec: KernelSpec):
    """Return the frozen contract that names this spec's prediction class.

    Most families are identified by their exact contract. A family that
    declares canonical equivalence fields reads part of its contract only
    through reductions coarser than the field itself, so every spec in one
    physical class is frozen as the canonical member and shares one
    prediction.  The substitution happens on the frozen contract, so the key
    is exactly the one the canonical spec would have produced.
    """

    substitutions = prediction_equivalence_fields(kernel_spec)
    if substitutions is None:
        return _contract_cache_key(kernel_spec)
    return (
        "dataclass",
        type(kernel_spec).__module__,
        type(kernel_spec).__qualname__,
        tuple(
            (
                field.name,
                freeze_contract_value(
                    substitutions[field.name] if field.name in substitutions
                    else getattr(kernel_spec, field.name)),
            )
            for field in fields(kernel_spec)
        ),
    )


class KernelPredictionMemo:
    """Bounded run-local LRUs for exact predictions and source-spec keys.

    The engineering capacity limits metadata retention; eviction recomputes
    the unchanged predictor and never removes a KernelPlan call.
    """

    def __init__(self, max_entries: int = 4096):
        if type(max_entries) is not int:
            raise TypeError("kernel prediction memo max_entries must be a positive int")
        if max_entries <= 0:
            raise ValueError("kernel prediction memo max_entries must be a positive int")
        self._max_entries = max_entries
        self._predictions = OrderedDict()
        self._numeric_predictions = OrderedDict()
        self._execution_specs = OrderedDict()
        self._execution_source_epoch = None
        self._hardware_objects = {}
        self._contract_keys_by_spec_identity = OrderedDict()
        self._plan_identity_groups = OrderedDict()
        self._hit_count = 0
        self._miss_count = 0

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def entry_count(self) -> int:
        return len(self._predictions)

    @property
    def contract_key_entry_count(self) -> int:
        return len(self._contract_keys_by_spec_identity)

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    def predict(
        self,
        kernel_spec: KernelSpec,
        hardware,
        method: str,
        *,
        contract_key=None,
        memory_bindings=None,
        equivalent_call_count: int = 1,
    ) -> KernelPrediction:
        if not isinstance(kernel_spec, KernelSpec):
            raise TypeError("kernel prediction memo requires KernelSpec")
        if (isinstance(equivalent_call_count, bool)
                or not isinstance(equivalent_call_count, int)):
            raise TypeError(
                "equivalent kernel call count must be an integer")
        if equivalent_call_count <= 0:
            raise ValueError(
                "equivalent kernel call count must be positive")
        hardware_identity = id(hardware)
        if hardware_identity in self._hardware_objects:
            if self._hardware_objects[hardware_identity] is not hardware:
                raise RuntimeError(
                    "kernel prediction memo hardware identity collision")
        else:
            self._hardware_objects[hardware_identity] = hardware
        if memory_bindings is not None:
            # Bound programs retain exact geometry and source identities, even
            # where an unbound predictor declares a coarser physical class.
            contract_key = _contract_cache_key(kernel_spec)
        elif contract_key is None:
            contract_key = _prediction_class_key(kernel_spec)
        key = (
            method,
            hardware_identity,
            contract_key,
            freeze_contract_value(memory_bindings),
            memory_prediction_source_epoch(),
        )
        if key in self._predictions:
            self._predictions.move_to_end(key)
            self._hit_count += equivalent_call_count
            return self._predictions[key]
        if self._execution_source_epoch is not key[-1]:
            self._execution_specs.clear()
            self._execution_source_epoch = key[-1]
        with prediction_memory_program_reuse(execution_cache=self._execution_specs):
            projection = bound_prediction_numeric_reuse_key(kernel_spec,hardware,memory_bindings,method)
            numeric_key = None if projection is None else (method,hardware_identity,contract_key,projection,key[-1])
            reused = numeric_key is not None and numeric_key in self._numeric_predictions
            if reused:
                self._numeric_predictions.move_to_end(numeric_key)
                prediction = self._numeric_predictions[numeric_key]
            else:
                prediction = _predict_with_memory(kernel_spec, hardware, method, memory_bindings)
                if not isinstance(prediction, KernelPrediction):
                    raise TypeError("kernel_base.predict must return KernelPrediction")
                if numeric_key is not None:
                    self._numeric_predictions[numeric_key] = prediction
                    if len(self._numeric_predictions) > self._max_entries:
                        self._numeric_predictions.popitem(last=False)
        if not isinstance(prediction, KernelPrediction):
            raise TypeError(
                "kernel_base.predict must return KernelPrediction")
        self._predictions[key] = prediction
        if len(self._predictions) > self._max_entries:
            self._predictions.popitem(last=False)
        self._miss_count += 0 if reused else 1
        self._hit_count += equivalent_call_count if reused else equivalent_call_count - 1
        return prediction

    def contract_key(self, kernel_spec: KernelSpec):
        if not isinstance(kernel_spec, KernelSpec):
            raise TypeError("kernel prediction memo requires KernelSpec")
        identity = id(kernel_spec)
        if identity in self._contract_keys_by_spec_identity:
            retained, contract_key = (
                self._contract_keys_by_spec_identity[identity])
            if retained is not kernel_spec:
                raise RuntimeError(
                    "kernel prediction memo KernelSpec identity collision")
            self._contract_keys_by_spec_identity.move_to_end(identity)
            return contract_key
        contract_key = _prediction_class_key(kernel_spec)
        self._contract_keys_by_spec_identity[identity] = (
            kernel_spec, contract_key)
        if len(self._contract_keys_by_spec_identity) > self._max_entries:
            self._contract_keys_by_spec_identity.popitem(last=False)
        return contract_key


    def plan_identity_groups(self, plan, hardware_by_id):
        """Reuse grouping membership, never prices or mutable spec contents.

        The existing binder proves which call slots changed. Unchanged groups
        retain their exact KernelSpec identities; all groups still enter the
        ordinary contract-key and prediction memo on every invocation.
        Source-bound memory groups keep their complete existing path.
        """
        if (type(hardware_by_id) is not dict
                or KernelPlan.topological_order is not _PLAN_GROUP_ORDER_SOURCE[0]
                or KernelPlan.topological_order.__code__ is not _PLAN_GROUP_ORDER_SOURCE[1]
                or hardware_for_call is not _PLAN_GROUP_HARDWARE_SOURCE[0]
                or hardware_for_call.__code__ is not _PLAN_GROUP_HARDWARE_SOURCE[1]):
            return None
        binding = _bound_kernel_spec_projection(plan)
        if binding is None:
            return None
        template, changed = binding
        identity = id(template)
        if identity not in self._plan_identity_groups:
            if (template.memory_context is not None
                    or any(type(call) is not KernelCall or type(call.kernel_spec) is not KernelSpec
                        or call.memory_source is not None or call.memory_bindings is not None
                        for call in template.calls)):
                return None
            positions = {call.call_id: index for index, call in enumerate(template.calls)}
            groups = {}
            for call in template.topological_order():
                key = id(call.kernel_spec)
                if key not in groups:
                    groups[key] = (call.kernel_spec, [])
                groups[key][1].append(positions[call.call_id])
            rows = tuple((spec, frozenset(indices), tuple(indices),
                tuple(template.calls[index].call_id for index in indices))
                for spec, indices in groups.values())
            self._plan_identity_groups[identity] = (template, template.calls, template._topological_calls, rows)
            if len(self._plan_identity_groups) > 128:
                self._plan_identity_groups.popitem(last=False)
        retained, calls, order, rows = self._plan_identity_groups[identity]
        if retained is not template:
            raise RuntimeError("prediction grouping template identity collision")
        if calls is not template.calls or order is not template._topological_calls:
            del self._plan_identity_groups[identity]
            return None
        self._plan_identity_groups.move_to_end(identity)
        changed = frozenset(changed)
        result = []
        for spec, members, indices, names in rows:
            touched = members & changed
            if touched:
                names = tuple(template.calls[index].call_id for index in indices if index not in touched)
            if names:
                hardware = hardware_for_call(template.calls[indices[0]], hardware_by_id)
                result.append((spec, hardware, names))
        updates = {}
        for index in changed:
            call = plan.calls[index]
            if call.memory_source is not None or call.memory_bindings is not None:
                return None
            key = id(call.kernel_spec)
            if key not in updates:
                updates[key] = (call.kernel_spec, hardware_for_call(call, hardware_by_id), [])
            updates[key][2].append(call.call_id)
        result.extend(updates.values())
        return result


def _predict_with_memory(kernel_spec, hardware, method, memory_bindings):
    if memory_bindings is None:
        return predict(kernel_spec, hardware, method)
    return predict(kernel_spec, hardware, method, memory_bindings=memory_bindings)


def _bound_fp8_source_call(call, method):
    kernel = call.kernel_spec
    return (method == "midlayer" and call.source_launches is not None
        and kernel.kind is KernelKind.FUSED_MOE and kernel.variant == "routed_experts"
        and kernel.attributes["routed_weight_dtype"] == "fp8"
        and "mathematical_program" in kernel.attributes
        and kernel.attributes["mathematical_program"] == "block_fp8_swiglu_v1")


def _fp8_diagnostics(predictions, scope, gaps):
    from src.kernel.model_contract import PredictionDiagnostics
    compute = {}
    memory = []
    access_classes = []
    limitations = [scope, *gaps]
    for role, prediction in predictions:
        for resource, seconds in prediction.diagnostics.compute_time_s.items():
            compute[resource] = (compute[resource] if resource in compute else 0.0) + seconds
        memory.extend(prediction.diagnostics.memory)
        access_classes.extend(prediction.diagnostics.memory_access_classes)
        for limitation in prediction.diagnostics.limitations:
            if limitation not in limitations:
                limitations.append(limitation)
    return PredictionDiagnostics(
        compute_time_s=compute,
        memory=tuple(memory),
        memory_access_classes=tuple(access_classes),
        memory_arbitration=(),
        pipeline=None,
        critical_path=tuple(role for role, _ in predictions),
        support_level="partial",
        limitations=tuple(limitations))


def _predict_bound_fp8_source_call(call, hardware, method, predict_stage):
    from src.kernel.fused_moe_physical_specs import build_bound_fp8_expert_stage_specs
    from src.kernel.kernel_spec import KernelPrediction
    from src.kernel.predictors.fused_moe import fp8_block_tile_support_reason
    stages = build_bound_fp8_expert_stage_specs(call.kernel_spec, call.source_launches)
    for role, spec in stages:
        if role not in ("w1", "w2"):
            continue
        reason = fp8_block_tile_support_reason(spec, hardware)
        if reason is not None:
            fallback = _predict_with_memory(
                call.kernel_spec, hardware, method, call.memory_bindings)
            return replace(fallback, diagnostics=_fp8_diagnostics((),
                "fp8_expert_mathematical_scope_no_physical_tile", (reason,)))
    predictions = tuple((role, predict_stage(spec)) for role, spec in stages)
    per_group = {f"fp8_expert/{role}": prediction.t_total_s
        for role, prediction in predictions}
    per_resource = {}
    for _, prediction in predictions:
        for role, seconds in prediction.t_per_resource.items():
            per_resource[role] = (per_resource[role] if role in per_resource else 0.0) + seconds
    bottleneck = max(predictions, key=lambda item: item[1].t_total_s)[0]
    return KernelPrediction(
        t_total_s=sum(prediction.t_total_s for _, prediction in predictions),
        t_per_group=per_group,
        t_launch_s=sum(prediction.t_launch_s for _, prediction in predictions),
        bottleneck_group=f"fp8_expert/{bottleneck}",
        t_per_resource=per_resource,
        diagnostics=_fp8_diagnostics(predictions,
            "fp8_expert_physical_stage_service",
            ("serial_stage_service_sum_not_device_utilization",)))


@memory_prediction_source_scope()
def predict_kernel_plan(plan: KernelPlan, hardware_by_id: Mapping[str, object],
                        method: str = "midlayer",
                        memo: KernelPredictionMemo | None = None):
    if memo is not None and not isinstance(memo, KernelPredictionMemo):
        raise TypeError("kernel prediction memo must be KernelPredictionMemo")
    # Group reuse needs this classification only for actual source-bound MoE
    # candidates. Ordinary calls still follow their complete prediction path.
    has_bound_fp8 = (memo is not None and method == "midlayer" and any(
        call.source_launches is not None and call.kernel_spec.kind is KernelKind.FUSED_MOE
        and _bound_fp8_source_call(call, method) for call in plan.calls))
    retained_groups = (None if memo is None or has_bound_fp8
        else memo.plan_identity_groups(plan, hardware_by_id))
    ordered_calls = () if retained_groups is not None else plan.topological_order()
    from .attention_memory_bindings import _FA3SourceCallContext, fa3_plan_call_groups, predict_fa3_plan_call
    fa3_context = _FA3SourceCallContext(ordered_calls)
    fa3_groups = fa3_plan_call_groups(ordered_calls, source_context=fa3_context)
    if memo is None:
        predictions = {}
        for call in ordered_calls:
            if call.call_id in fa3_groups:
                predictions[call.call_id] = predict_fa3_plan_call(
                    call, fa3_groups[call.call_id], hardware_for_call(call, hardware_by_id), method,
                    _predict_with_memory, source_calls=fa3_context)
                continue
            hardware = hardware_for_call(call, hardware_by_id)
            if _bound_fp8_source_call(call, method):
                predictions[call.call_id] = _predict_bound_fp8_source_call(
                    call, hardware, method,
                    lambda spec: _predict_with_memory(spec, hardware, method, None))
                continue
            predictions[call.call_id] = _predict_with_memory(
                call.kernel_spec, hardware, method, call.memory_bindings)
        return predictions

    # The plan still carries every rank-local call and dependency.  Only the
    # pure predictor invocation is grouped by the complete frozen KernelSpec,
    # prediction method, and exact HWSpec object.  A rank whose spec differs
    # in any field therefore forms its own group automatically.
    identity_groups = {}
    predictions = {}
    for call in ordered_calls:
        hardware = hardware_for_call(call, hardware_by_id)
        if call.call_id in fa3_groups:
            predictions[call.call_id] = predict_fa3_plan_call(
                call, fa3_groups[call.call_id], hardware, method, memo.predict, source_calls=fa3_context)
            continue
        if _bound_fp8_source_call(call, method):
            predictions[call.call_id] = _predict_bound_fp8_source_call(
                call, hardware, method,
                lambda spec: memo.predict(spec, hardware, method))
            continue
        if call.memory_bindings is not None:
            predictions[call.call_id] = memo.predict(
                call.kernel_spec, hardware, method, memory_bindings=call.memory_bindings)
            continue
        identity_key = (method, id(hardware), id(call.kernel_spec))
        if identity_key not in identity_groups:
            identity_groups[identity_key] = [
                call.kernel_spec, hardware, []]
        elif identity_groups[identity_key][0] is not call.kernel_spec:
            raise RuntimeError("plan-local KernelSpec identity collision")
        identity_groups[identity_key][2].append(call.call_id)

    groups = {}
    for kernel_spec, hardware, call_ids in (identity_groups.values() if retained_groups is None else retained_groups):
        contract_key = memo.contract_key(kernel_spec)
        group_key = (method, id(hardware), contract_key)
        if group_key not in groups:
            groups[group_key] = [kernel_spec, hardware, []]
        groups[group_key][2].extend(call_ids)

    for _group_key, (kernel_spec, hardware, call_ids) in groups.items():
        prediction = memo.predict(
            kernel_spec,
            hardware,
            method,
            contract_key=_group_key[2],
            equivalent_call_count=len(call_ids),
        )
        for call_id in call_ids:
            predictions[call_id] = prediction
    return predictions
