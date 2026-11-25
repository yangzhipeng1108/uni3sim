"""Shared physical model input and arithmetic for pure predictors."""

from src.kernel.prediction_cache import RuntimeCache

import dataclasses
import itertools
import math
import threading
from fractions import Fraction
from typing import Dict, Mapping, Optional, Tuple, Union

from src.kernel.kernel_spec import (
    KernelPrediction, KernelSpec, graph_replayed_launch, prediction_from_ns,
)
from src.kernel.model_contract import (
    PredictionDiagnostics,
    PredictionInput,
    AIV_COMPLETE_TASK_FLOOR_ONLY,
    AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    AIV_SOURCE_PIPELINE_LAUNCH_ONLY,
    AIV_SOURCE_ENTRY_FILL,
    AIV_WORKER_START_FILL,
    GPU_SOURCE_PIPELINE_OUTER_ENVELOPE,
    AnalyticalDependencyPath,
    AnalyticalPhase,
    AnalyticalPhaseLedger,
    AttentionPipelineEnvelopeDemand,
    BarrierPipelineEnvelopeDemand,
    CannGroupedMatmulEnvelopeDemand,
    CannSingleCoreSplitKEnvelopeDemand,
    CANN_AIV_TILING_MATERIALIZATION_48B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_56B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_80B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_192B_PROLOGUE,
    CANN_AIV_TILING_MATERIALIZATION_PROLOGUE,
    ComputeDemand,
    ComputeServiceDemand,
    FiaCrossDomainWorkPlan,
    FiaSplitFuseWorkPlan,
    GpuGroupedGemmPipelineEnvelopeDemand,
    MemoryAccessClassDemand,
    MemoryAccessClassSignature,
    MemoryNodeDemand,
    MemoryPathDemand,
    MemorySourceDemand,
    MetaxGroupedOperandPipelineDemand,
    PhaseComputeDemand,
    PhaseMemoryDemand,
    PipelineBuffer,
    PipelineStageDemand,
    PipelineTaskDemand,
    PipelineTaskProgramDemand,
    PipelineTaskSequenceDemand,
    PipelineWorkPlan,
    TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR,
    TENSOR_ITERATOR_FP32_RANK3_COPY_TASK_FLOOR,
    PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
    PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_CLASS,
    WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
)


# The production model intentionally makes no per-kernel residency estimate.
# Grid tail is modeled separately; resource occupancy is explicitly full until
# a source-bound physical residency contract is introduced.
FULL_RESOURCE_OCCUPANCY = 1.0


_PIPELINE_STAGE_TEMPLATE_CACHE_SIZE = 4096
_pipeline_stage_templates = RuntimeCache()
_pipeline_stage_templates_lock = threading.Lock()
_TYPED_COMPUTE_PIPELINE_STAGE_TEMPLATE_CACHE_SIZE = 4096
_typed_compute_pipeline_stage_templates = RuntimeCache()
_typed_compute_pipeline_stage_templates_lock = threading.Lock()
_MEMORY_ACCESS_TEMPLATE_CACHE_SIZE = 4096
_memory_access_templates = RuntimeCache()
_memory_access_templates_lock = threading.Lock()


def bind_pipeline_stage_template(
        name, service, work_fraction, include_first_request_latency=False,
        *, predecessors=None, start_predecessors=(), prior_resources=(),
        absolute_duration_s=None):
    """Bind dynamic source work to a shared, validated stage template.

    A hardware adapter owns the launch/source facts which choose the stage
    name, service and dependency topology.  A800, C500 and 910B adapters can
    therefore use the same cache and fail-closed binding path while retaining
    their own physical launch rules.  The cache contains no shape, byte,
    duration or prediction value: each invocation validates and binds exactly
    one dynamic work fraction or mechanism duration.
    """

    is_absolute = absolute_duration_s is not None
    if is_absolute == (work_fraction is not None):
        raise ValueError(
            "pipeline stage template requires exactly one dynamic work or "
            "duration")
    key = (
        name, service, include_first_request_latency, predecessors,
        start_predecessors, prior_resources, is_absolute,
    )
    with _pipeline_stage_templates_lock:
        if key not in _pipeline_stage_templates:
            if (len(_pipeline_stage_templates)
                    >= _PIPELINE_STAGE_TEMPLATE_CACHE_SIZE):
                _pipeline_stage_templates.clear()
            _pipeline_stage_templates[key] = PipelineStageDemand(
                name, service, None if is_absolute else 1.0,
                include_first_request_latency,
                predecessors=predecessors,
                start_predecessors=start_predecessors,
                prior_resources=prior_resources,
                absolute_duration_s=1.0 if is_absolute else None,
            )
        template = _pipeline_stage_templates[key]
    return PipelineStageDemand.bind_validated_dynamic_value(
        template,
        work_fraction=work_fraction,
        absolute_duration_s=absolute_duration_s,
    )


def bind_typed_compute_pipeline_stage_template(
        name, service, typed_services, operations, *,
        serial_invocations=1, predecessors=None, start_predecessors=(),
        prior_resources=()):
    """Bind dynamic typed compute work to a reusable source-stage contract.

    Like ``bind_pipeline_stage_template``, the cache stores static source
    topology only.  ``operations`` are validated and rebound for each
    prediction, never used as duration or prediction-result keys.  It is
    shared by all hardware adapters whose source programs use typed compute
    services, while launch/residency facts remain adapter-owned.
    """

    if (not isinstance(typed_services, tuple) or not typed_services
            or any(not isinstance(value, str) or not value
                   for value in typed_services)):
        raise TypeError("typed compute stage services must be non-empty strings")
    if len(typed_services) != len(set(typed_services)):
        raise ValueError("typed compute stage services cannot repeat")
    if (isinstance(serial_invocations, bool)
            or not isinstance(serial_invocations, int)
            or serial_invocations <= 0):
        raise ValueError(
            "typed compute stage serial invocations must be positive integer")
    if (not isinstance(operations, tuple)
            or len(operations) != len(typed_services)):
        raise ValueError(
            "typed compute stage operations must align with services")
    key = (
        name, service, typed_services, serial_invocations,
        predecessors, start_predecessors, prior_resources,
    )
    with _typed_compute_pipeline_stage_templates_lock:
        if key not in _typed_compute_pipeline_stage_templates:
            if (len(_typed_compute_pipeline_stage_templates)
                    >= _TYPED_COMPUTE_PIPELINE_STAGE_TEMPLATE_CACHE_SIZE):
                _typed_compute_pipeline_stage_templates.clear()
            _typed_compute_pipeline_stage_templates[key] = PipelineStageDemand(
                name, service, None,
                compute_service_demands=tuple(
                    ComputeServiceDemand(
                        typed_service, 1.0,
                        serial_invocations=serial_invocations)
                    for typed_service in typed_services),
                predecessors=predecessors,
                start_predecessors=start_predecessors,
                prior_resources=prior_resources,
            )
        template = _typed_compute_pipeline_stage_templates[key]
    return PipelineStageDemand.bind_validated_typed_compute_operations(
        template, operations)


def bind_memory_access_class_template(
        service, ingress_resource_role, logical_bytes, signature,
        replaces_ingress_path_service=False):
    """Bind dynamic bytes to a shared checked access-class source template.

    A hardware adapter selects the source service and immutable signature;
    only the current shape's logical bytes are dynamic.  This is the typed
    memory counterpart to ``bind_pipeline_stage_template`` and contains no
    traffic prediction, duration, or workload-result cache.
    """

    key = (
        service,
        ingress_resource_role,
        signature,
        replaces_ingress_path_service,
    )
    with _memory_access_templates_lock:
        if key not in _memory_access_templates:
            if (len(_memory_access_templates)
                    >= _MEMORY_ACCESS_TEMPLATE_CACHE_SIZE):
                _memory_access_templates.clear()
            _memory_access_templates[key] = MemoryAccessClassDemand(
                service=service,
                ingress_resource_role=ingress_resource_role,
                logical_bytes=1.0,
                signature=signature,
                replaces_ingress_path_service=replaces_ingress_path_service,
            )
        template = _memory_access_templates[key]
    return MemoryAccessClassDemand.bind_validated_logical_bytes(
        template,
        logical_bytes=logical_bytes,
    )


def attach_structured_contract(
        inp: PredictionInput, hardware, *,
        transaction_bytes: Optional[float],
        path_name: Optional[str],
        access_kind: str = "read",
        source_producers_per_unit: float = 1.0,
        independent_requests_per_producer: Optional[float] = 1.0,
        frontend_rate_per_unit_s: Optional[float] = None,
        prepend_npu_aiv_mte2_ingress: bool = False) -> PredictionInput:
    """Attach the one canonical compute and memory-path contract.

    Per-kind predictors still own physical operation/traffic derivation.  This
    function only normalizes those already-derived quantities into the shared
    solver schema; it does not infer a kernel variant or fit a duration.
    """

    import dataclasses

    if not isinstance(inp, PredictionInput):
        raise TypeError("attach_structured_contract requires PredictionInput")
    if inp.compute_demands or inp.memory_paths or inp.pipeline_plan is not None:
        raise ValueError(
            "structured contract must be attached exactly once")
    if not isinstance(prepend_npu_aiv_mte2_ingress, bool):
        raise TypeError("prepend_npu_aiv_mte2_ingress must be boolean")
    demand = dict(inp.demand_per_resource)
    if not hardware.l1_cache_enabled and inp.demand_per_resource["l1_cache"]:
        demand["l1_cache"] = 0.0
    if inp.mem_model == "npu_tile":
        # The aggregate-only NPU contract currently establishes Cube/AIV
        # operations plus L2/GM/MTE3 byte demand.  GPU-derived L1, SMEM and
        # crossbar traffic have no source-backed AIC/UB tile lifecycle and
        # must not be priced through similarly named HWSpec resources.
        for role in ("l1_cache", "smem", "xbar"):
            demand[role] = 0.0
    compute_demands = []
    for role in (
            "compute_tensor", "compute_vector", "compute_reduction",
            "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
            "scalar_aiv"):
        if demand[role] <= 0:
            continue
        compute_demands.append(ComputeDemand(
            resource_role=role,
            operations=float(demand[role]),
            shape_efficiency=1.0,
            schedule_efficiency=(
                inp.eta * inp.wave_efficiency),
        ))
    compute_demands = tuple(compute_demands)

    l2_bytes = float(demand["l2_cache"])
    dram_bytes = float(demand["dram"])
    if l2_bytes == 0.0:
        if dram_bytes != 0.0:
            raise ValueError(
                "positive DRAM traffic requires a structured ingress path")
        memory_paths = ()
    elif inp.mem_model in ("gpu_mshr", "npu_tile"):
        if inp.mem_model == "gpu_mshr":
            if transaction_bytes is not None:
                raise ValueError(
                    "GPU transaction sizes come from explicit per-level "
                    "HWSpec fields, not cacheline_bytes")
            transaction_bytes_by_role = {
                "l2_cache": float(hardware.l2_transaction_bytes),
                "dram": float(hardware.dram_transaction_bytes),
            }
        elif inp.mem_model == "npu_tile":
            if transaction_bytes is not None:
                raise ValueError(
                    "NPU transaction size must remain None until physical "
                    "MTE transaction geometry is independently established")
            transaction_bytes_by_role = {
                "l2_cache": None,
                "dram": None,
            }
        else:
            raise AssertionError(
                f"unhandled target memory model: {inp.mem_model!r}")
        if path_name is None or not isinstance(path_name, str) or not path_name:
            raise ValueError(
                "target memory path requires a non-empty path name")
        l2_transaction_bytes = transaction_bytes_by_role["l2_cache"]
        nodes = [MemoryNodeDemand(
            resource_role="l2_cache",
            transactions=(
                None if l2_transaction_bytes is None
                else l2_bytes / l2_transaction_bytes),
            bytes=l2_bytes,
            transaction_bytes=l2_transaction_bytes,
        )]
        if prepend_npu_aiv_mte2_ingress:
            if inp.mem_model != "npu_tile":
                raise ValueError(
                    "NPU AIV MTE2 ingress requires npu_tile memory")
            if hardware.kernel_memory_model != "npu_tile":
                raise ValueError(
                    "NPU AIV MTE2 ingress requires npu_tile hardware")
            if inp.launch_task_kind != "aiv":
                raise ValueError(
                    "NPU AIV MTE2 ingress requires an AIV-only task")
            mte2_units = hardware.resource_by_role("mte2").total_units
            if inp.active_units > mte2_units:
                raise ValueError(
                    "NPU MTE2 active units exceed the physical AIV domain: "
                    f"active={inp.active_units}, physical={mte2_units}")
            nodes.insert(0, MemoryNodeDemand(
                resource_role="mte2",
                transactions=None,
                bytes=l2_bytes,
                transaction_bytes=None,
            ))
        if dram_bytes > 0:
            dram_transaction_bytes = transaction_bytes_by_role["dram"]
            nodes.append(MemoryNodeDemand(
                resource_role="dram",
                transactions=(
                    None if dram_transaction_bytes is None
                    else dram_bytes / dram_transaction_bytes),
                bytes=dram_bytes,
                transaction_bytes=dram_transaction_bytes,
            ))
        memory_paths = (MemoryPathDemand(
            name=path_name,
            access_kind=access_kind,
            initial_transactions=(
                None if l2_transaction_bytes is None
                else l2_bytes / l2_transaction_bytes),
            nodes=tuple(nodes),
            source=MemorySourceDemand(
                active_units=inp.active_units,
                producers_per_unit=source_producers_per_unit,
                independent_requests_per_producer=(
                    independent_requests_per_producer),
                frontend_rate_per_unit_s=frontend_rate_per_unit_s,
            ),
        ),)
    elif inp.mem_model in ("pim_bank", "systolic_scratchpad"):
        # These explicit compatibility architectures are outside the current
        # three-hardware acceptance scope and retain their existing physical
        # formula until a transaction contract exists for their memory path.
        if transaction_bytes is not None or path_name is not None:
            raise ValueError(
                "compatibility memory models cannot declare target paths")
        memory_paths = ()
    else:
        raise ValueError(f"unknown PredictionInput.mem_model: {inp.mem_model!r}")
    return dataclasses.replace(
        inp,
        demand_per_resource=demand,
        compute_demands=compute_demands,
        memory_paths=memory_paths,
    )


def replace_structured_demands(
        inp: PredictionInput, replacements: Mapping[str, float]
        ) -> PredictionInput:
    """Replace resource demand while preserving the structured contract.

    This is the canonical observation/ablation boundary.  A replacement of a
    flat compatibility demand must update the corresponding typed compute
    demand and every structured memory node; otherwise the formula would price
    one workload while diagnostics described another one.
    """

    import dataclasses

    if not isinstance(inp, PredictionInput):
        raise TypeError("replace_structured_demands requires PredictionInput")
    if not isinstance(replacements, Mapping):
        raise TypeError("structured demand replacements must be a mapping")
    unknown = set(replacements) - set(inp.demand_per_resource)
    if unknown:
        raise ValueError(
            "structured demand replacement has unknown roles: "
            f"{sorted(unknown)}")
    normalized = {}
    for role, value in replacements.items():
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0):
            raise ValueError(
                "structured demand replacements must be finite and "
                f"non-negative: {role}={value!r}")
        normalized[role] = float(value)

    annotated_ingress_roles = {
        path.nodes[0].resource_role
        for path in inp.memory_paths
        if path.access_class_demands
    }
    replaced_annotated_ingress = (
        annotated_ingress_roles.intersection(normalized))
    if replaced_annotated_ingress:
        raise ValueError(
            "structured demand replacement cannot replace an ingress with "
            "typed memory access-class demand: "
            f"{sorted(replaced_annotated_ingress)}")

    flat = dict(inp.demand_per_resource)
    flat.update(normalized)
    compute_by_role = {
        demand.resource_role: demand for demand in inp.compute_demands
    }
    typed_pipeline_scale = {}
    compute_roles = {
        "compute_tensor", "compute_vector", "compute_reduction",
        "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
    }

    def _rescale_service(service, scale):
        worker_operations = (
            tuple(
                float(Fraction(value) * scale)
                for value in service.operations_by_worker)
            if service.operations_by_worker else ())
        return dataclasses.replace(
            service,
            operations=float(Fraction(service.operations) * scale),
            operations_by_worker=worker_operations,
        )

    for role in compute_roles.intersection(normalized):
        operations = normalized[role]
        if role in compute_by_role:
            if operations == 0.0:
                del compute_by_role[role]
            else:
                prior = compute_by_role[role]
                if prior.exact_request_demands:
                    exact_operations = math.fsum(
                        demand.operations
                        for demand in prior.exact_request_demands)
                    regular_prior = prior.operations - exact_operations
                    regular_replacement = operations - exact_operations
                    if regular_replacement < 0.0:
                        raise ValueError(
                            "structured demand replacement cannot reduce "
                            "operations below a source-defined exact request "
                            "schedule")
                    if not prior.service_demands:
                        if regular_replacement != 0.0:
                            raise ValueError(
                                "structured demand replacement cannot "
                                "rescale a source-defined exact request "
                                "schedule without a replaceable regular "
                                "service")
                        service_demands = ()
                    else:
                        if regular_prior <= 0.0:
                            raise ValueError(
                                "exact request demand has no positive regular "
                                "service remainder")
                        service_demands = tuple(
                            _rescale_service(
                                service,
                                Fraction(regular_replacement)
                                / Fraction(regular_prior))
                            for service in prior.service_demands
                            if regular_replacement > 0.0)
                    # Exact request sizes and worker ownership are deliberately
                    # preserved.  Oracle/ablation replacements may only alter
                    # the remaining regular service demand.
                else:
                    typed_pipeline_scale[f"compute:{role}"] = (
                        Fraction(operations) / Fraction(prior.operations))
                    service_demands = tuple(
                        _rescale_service(
                            service,
                            Fraction(operations)
                            / Fraction(prior.operations))
                        for service in prior.service_demands)
                compute_by_role[role] = dataclasses.replace(
                    prior,
                    operations=operations,
                    service_demands=service_demands,
                )
        elif operations > 0.0:
            raise ValueError(
                "cannot introduce a new compute role without its shape and "
                f"schedule efficiency contract: {role!r}")

    def _rescale_pipeline_stages(source_stages):
        stages = []
        for stage in source_stages:
            if (not stage.compute_service_demands
                    or stage.service not in typed_pipeline_scale):
                stages.append(stage)
                continue
            scale = typed_pipeline_scale[stage.service]
            stages.append(dataclasses.replace(
                stage,
                compute_service_demands=tuple(
                    _rescale_service(service, scale)
                    for service in stage.compute_service_demands),
            ))
        return tuple(stages)

    def _rescale_pipeline_task(task):
        return dataclasses.replace(
            task, stages=_rescale_pipeline_stages(task.stages))

    def _rescale_pipeline_sequence(sequence):
        return dataclasses.replace(
            sequence,
            tasks=tuple(
                _rescale_pipeline_task(task)
                for task in sequence.tasks),
        )

    if inp.pipeline_plan is None or not typed_pipeline_scale:
        pipeline_plan = inp.pipeline_plan
    elif isinstance(inp.pipeline_plan, AttentionPipelineEnvelopeDemand):
        rescaled_phase_by_demand = {}
        rescaled_phases_by_identity = {}

        def _rescale_attention_phase(phase):
            if phase not in rescaled_phase_by_demand:
                rescaled_phase_by_demand[phase] = dataclasses.replace(
                    phase,
                    stages=_rescale_pipeline_stages(phase.stages),
                )
            return rescaled_phase_by_demand[phase]

        def _rescale_attention_phases(phases):
            identity = id(phases)
            if identity not in rescaled_phases_by_identity:
                rescaled_phases_by_identity[identity] = tuple(
                    _rescale_attention_phase(phase) for phase in phases)
            return rescaled_phases_by_identity[identity]

        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan,
            classes=tuple(
                dataclasses.replace(
                    work_class,
                    phases=_rescale_attention_phases(work_class.phases),
                )
                for work_class in inp.pipeline_plan.classes),
        )
    elif isinstance(inp.pipeline_plan, BarrierPipelineEnvelopeDemand):
        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan,
            classes=tuple(
                _rescale_pipeline_sequence(sequence)
                for sequence in inp.pipeline_plan.classes),
        )
    elif isinstance(inp.pipeline_plan, FiaCrossDomainWorkPlan):
        def _rescale_fia_stage(stage):
            service_name = f"compute:{stage.resource_role}"
            if service_name not in typed_pipeline_scale:
                return stage
            scale = typed_pipeline_scale[service_name]
            return dataclasses.replace(
                stage,
                logical_operations=float(
                    Fraction(stage.logical_operations) * scale),
                physical_operations=float(
                    Fraction(stage.physical_operations) * scale),
                typed_services=tuple(
                    _rescale_service(service, scale)
                    for service in stage.typed_services),
            )

        def _rescale_gqa_source_order(source_order):
            """Keep literal GQA panels consistent with replaced Cube work."""

            if source_order is None:
                return None
            return dataclasses.replace(
                source_order,
                rounds=tuple(
                    dataclasses.replace(
                        round_demand,
                        panels=tuple(
                            dataclasses.replace(
                                panel,
                                mm1=_rescale_fia_stage(panel.mm1),
                                mm2=_rescale_fia_stage(panel.mm2),
                            )
                            for panel in round_demand.panels),
                    )
                    for round_demand in source_order.rounds),
            )

        fia_groups = []
        for group in inp.pipeline_plan.groups:
            tasks = []
            for task in group.tasks:
                tasks.append(dataclasses.replace(
                    task,
                    mm1=_rescale_fia_stage(task.mm1),
                    vec1=_rescale_fia_stage(task.vec1),
                    mm2=_rescale_fia_stage(task.mm2),
                    vec2=_rescale_fia_stage(task.vec2),
                    vec1_update=(
                        None if task.vec1_update is None
                        else _rescale_fia_stage(task.vec1_update)),
                    vec1_peer=(
                        None if task.vec1_peer is None
                        else _rescale_fia_stage(task.vec1_peer)),
                    vec1_peer_update=(
                        None if task.vec1_peer_update is None
                        else _rescale_fia_stage(task.vec1_peer_update)),
                    vec2_peer=(
                        None if task.vec2_peer is None
                        else _rescale_fia_stage(task.vec2_peer)),
                    gqa_source_order=_rescale_gqa_source_order(
                        task.gqa_source_order),
                ))
            fia_groups.append(dataclasses.replace(
                group, tasks=tuple(tasks)))
        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan,
            groups=tuple(fia_groups),
            terminal_vector_stages=tuple(
                _rescale_fia_stage(stage)
                for stage in inp.pipeline_plan.terminal_vector_stages),
            aiv_initialization_stages=tuple(
                _rescale_fia_stage(stage)
                for stage in inp.pipeline_plan.aiv_initialization_stages),
        )
    elif isinstance(inp.pipeline_plan, FiaSplitFuseWorkPlan):
        tensor_scale = (
            typed_pipeline_scale["compute:compute_tensor"]
            if "compute:compute_tensor" in typed_pipeline_scale
            else Fraction(1, 1))
        vector_scale = (
            typed_pipeline_scale["compute:compute_vector"]
            if "compute:compute_vector" in typed_pipeline_scale
            else Fraction(1, 1))
        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan,
            tensor_operation_scale=float(
                Fraction(inp.pipeline_plan.tensor_operation_scale)
                * tensor_scale),
            vector_operation_scale=float(
                Fraction(inp.pipeline_plan.vector_operation_scale)
                * vector_scale),
        )
    elif isinstance(inp.pipeline_plan, CannSingleCoreSplitKEnvelopeDemand):
        # The split-K envelope stores source geometry rather than structured
        # task demands.  Its role durations are supplied separately by the
        # formula, so observation replacements must preserve the plan itself.
        pipeline_plan = inp.pipeline_plan
    else:
        pipeline_entries = []
        for entry in inp.pipeline_plan.tasks:
            if isinstance(entry, PipelineTaskProgramDemand):
                pipeline_entries.append(dataclasses.replace(
                    entry,
                    patterns=tuple(
                        _rescale_pipeline_sequence(sequence)
                        for sequence in entry.patterns),
                ))
            elif isinstance(entry, PipelineTaskSequenceDemand):
                pipeline_entries.append(_rescale_pipeline_sequence(entry))
            elif isinstance(entry, PipelineTaskDemand):
                pipeline_entries.append(_rescale_pipeline_task(entry))
            else:
                raise TypeError(
                    "pipeline plan contains an unsupported task demand: "
                    f"{type(entry).__name__}")
        pipeline_plan = dataclasses.replace(
            inp.pipeline_plan, tasks=tuple(pipeline_entries))

    memory_roles = {
        node.resource_role
        for path in inp.memory_paths
        for node in path.nodes
    }
    replaced_memory_roles = memory_roles.intersection(normalized)
    bytes_by_role = {
        role: sum(
            node.bytes
            for path in inp.memory_paths
            for node in path.nodes
            if node.resource_role == role)
        for role in replaced_memory_roles
    }
    invalid_zero_bases = {
        role for role in replaced_memory_roles
        if bytes_by_role[role] == 0.0 and normalized[role] > 0.0
    }
    if invalid_zero_bases:
        raise ValueError(
            "cannot distribute positive observed demand across an empty "
            f"structured role: {sorted(invalid_zero_bases)}")

    paths = []
    for path in inp.memory_paths:
        nodes = []
        for node in path.nodes:
            role = node.resource_role
            if role not in replaced_memory_roles:
                nodes.append(node)
                continue
            role_total = bytes_by_role[role]
            replacement_total = normalized[role]
            byte_count = (
                0.0 if role_total == 0.0 else
                replacement_total * node.bytes / role_total)
            if byte_count == 0.0:
                continue
            nodes.append(dataclasses.replace(
                node,
                transactions=(
                    None if node.transaction_bytes is None
                    else byte_count / node.transaction_bytes),
                bytes=byte_count,
            ))
        if not nodes:
            continue
        if nodes[0].resource_role != path.nodes[0].resource_role:
            raise ValueError(
                "a structured demand replacement cannot remove a path "
                f"ingress while retaining downstream nodes: {path.name!r}")
        paths.append(dataclasses.replace(
            path,
            initial_transactions=nodes[0].transactions,
            nodes=tuple(nodes),
        ))

    if pipeline_plan is None or isinstance(
            pipeline_plan,
            (
                CannSingleCoreSplitKEnvelopeDemand,
                FiaCrossDomainWorkPlan,
                FiaSplitFuseWorkPlan,
            )):
        pipeline_stages = ()
    elif isinstance(pipeline_plan, AttentionPipelineEnvelopeDemand):
        unique_phases = {}
        for work_class in pipeline_plan.classes:
            for phase in work_class.phases:
                unique_phases[id(phase)] = phase
        pipeline_stages = tuple(
            stage
            for phase in unique_phases.values()
            for stage in phase.stages)
    elif isinstance(pipeline_plan, BarrierPipelineEnvelopeDemand):
        pipeline_stages = tuple(
            stage
            for sequence in pipeline_plan.classes
            for task in sequence.tasks
            for stage in task.stages)
    elif isinstance(pipeline_plan, PipelineWorkPlan):
        pipeline_entries = pipeline_plan.tasks
        logical_pipeline_tasks = tuple(
            logical
            for entry in pipeline_entries
            for sequence in (
                entry.patterns
                if isinstance(entry, PipelineTaskProgramDemand)
                else (entry,))
            for logical in (
                sequence.tasks
                if isinstance(sequence, PipelineTaskSequenceDemand)
                else (sequence,)))
        pipeline_stages = tuple(
            stage
            for task in logical_pipeline_tasks
            for stage in task.stages)
    else:
        raise TypeError(
            "structured demand replacement produced an unknown pipeline plan")
    used_services = {
        stage.service
        for stage in pipeline_stages
    }
    available_services = {
        *(f"compute:{role}" for role in compute_by_role),
        *(f"memory:{path.name}" for path in paths),
    }
    removed_services = {
        service for service in used_services
        if (service.startswith("compute:") or service.startswith("memory:"))
        and service not in available_services
    }
    if removed_services:
        raise ValueError(
            "structured demand replacement removes a live pipeline service: "
            f"{sorted(removed_services)}")

    if isinstance(pipeline_plan, FiaSplitFuseWorkPlan):
        l2_scale = (
            Fraction(normalized["l2_cache"])
            / Fraction(inp.demand_per_resource["l2_cache"])
            if "l2_cache" in normalized else Fraction(1, 1))
        mte3_scale = (
            Fraction(normalized["mte3"])
            / Fraction(inp.demand_per_resource["mte3"])
            if "mte3" in normalized else Fraction(1, 1))
        pipeline_plan = dataclasses.replace(
            pipeline_plan,
            l2_byte_scale=float(
                Fraction(pipeline_plan.l2_byte_scale) * l2_scale),
            mte3_byte_scale=float(
                Fraction(pipeline_plan.mte3_byte_scale) * mte3_scale),
        )

    phase_ledger = (
        None if inp.phase_ledger is None
        else inp.phase_ledger.rescaled(
            tuple(compute_by_role.values()), tuple(paths)))
    return dataclasses.replace(
        inp,
        demand_per_resource=flat,
        compute_demands=tuple(compute_by_role.values()),
        memory_paths=tuple(paths),
        pipeline_plan=pipeline_plan,
        phase_ledger=phase_ledger,
    )


def gpu_memory_source(hardware) -> Tuple[float, Optional[float]]:
    """Return the occupancy-free GPU memory-source contract.

    The model has no source-bound per-kernel resident-warp or independent
    request contract.  Represent one generic producer per active unit and let
    the hardware byte/request capacities price the resource demand.
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise ValueError("gpu_memory_source requires gpu_mshr hardware")
    return 1.0, None


def gpu_partial_limitations(hardware, *additional: str) -> Tuple[str, ...]:
    """Describe evidence gaps from capabilities, never hardware identity."""

    if any(not isinstance(value, str) or not value for value in additional):
        raise ValueError("additional GPU limitations must be non-empty strings")
    local_path = (
        "l1_and_local_queue_evidence_unavailable"
        if hardware.l1_cache_enabled
        else "local_queue_evidence_unavailable"
    )
    common = additional + (
        local_path,
        "noncoalesced_transaction_geometry_unavailable",
        "workload_producer_independence_unavailable",
        "hardware_source_ceiling_evidence_unavailable",
        "request_frontend_capacity_evidence_unavailable",
    )
    if hardware.dram_arbitration_mode == "shared_directional_capacity":
        return common + (
            "dram_direction_switch_schedule_unavailable",
            "dram_turnaround_unmodeled",
        )
    if hardware.dram_arbitration_mode == "independent_directional_capacity":
        return common
    if hardware.dram_arbitration_mode == "shared_common_capacity":
        return common + (
            "write_capacity_evidence_unavailable",
            "dram_directional_capacity_and_arbitration_unavailable",
        )
    raise ValueError(
        "GPU hardware has no supported DRAM arbitration capability")


def npu_launch_floor_contract(
        hardware, launch_task_kind: str) -> Tuple[bool, Tuple[str, ...]]:
    """Return minimum task-duration evidence status for one NPU task class."""

    if not hardware.aic_aiv_separated:
        raise ValueError(
            "NPU launch-floor contract requires separated AIC/AIV hardware")
    if launch_task_kind not in ("aic", "aiv", "mix_aiv", "mixed"):
        raise ValueError(
            "NPU launch-floor task kind must be 'aic', 'aiv', 'mix_aiv', "
            "or 'mixed'")
    available = hardware.has_task_duration_floor_for_task(
        launch_task_kind)
    if available:
        return True, ()
    return False, ("launch_task_duration_evidence_unavailable",)


def split_global_memory_paths(
        inp: PredictionInput, *, read_l2_bytes: float,
        read_dram_bytes: float, write_l2_bytes: float,
        write_dram_bytes: float,
        dependency_paths: Optional[Tuple[AnalyticalDependencyPath, ...]] = None
        ) -> PredictionInput:
    """Split the canonical global path into physical read and write streams."""

    import dataclasses

    if len(inp.memory_paths) != 1:
        raise ValueError("global memory split requires exactly one path")
    original = inp.memory_paths[0]
    if original.access_class_demands:
        raise ValueError(
            "global memory split cannot redistribute a path with typed "
            "memory access-class demand")
    if original.name not in ("global_memory", "mte2_read"):
        raise ValueError(
            f"cannot split unexpected memory path {original.name!r}")
    totals = {
        "read_l2_bytes": read_l2_bytes,
        "read_dram_bytes": read_dram_bytes,
        "write_l2_bytes": write_l2_bytes,
        "write_dram_bytes": write_dram_bytes,
    }
    invalid = {
        name: value for name, value in totals.items()
        if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0)
    }
    if invalid:
        raise ValueError(
            f"memory path split bytes must be finite and non-negative: {invalid}")
    original_l2 = inp.demand_per_resource["l2_cache"]
    original_dram = inp.demand_per_resource["dram"]
    if not math.isclose(
            read_l2_bytes + write_l2_bytes, original_l2,
            rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("read/write L2 bytes do not conserve global traffic")
    if not math.isclose(
            read_dram_bytes + write_dram_bytes, original_dram,
            rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("read/write DRAM bytes do not conserve global traffic")
    transaction_bytes_by_role = {
        node.resource_role: node.transaction_bytes
        for node in original.nodes
    }

    def make_path(name, l2_bytes, dram_bytes):
        if l2_bytes <= 0:
            if dram_bytes != 0:
                raise ValueError(
                    f"{name} has DRAM bytes without ingress bytes")
            return None
        if dram_bytes > l2_bytes:
            raise ValueError(f"{name} DRAM bytes cannot exceed L2 bytes")
        l2_transaction_bytes = transaction_bytes_by_role["l2_cache"]
        if l2_transaction_bytes is None:
            raise ValueError(
                "global GPU path split requires resolved L2 transactions")
        nodes = [MemoryNodeDemand(
            resource_role="l2_cache",
            transactions=l2_bytes / l2_transaction_bytes,
            bytes=l2_bytes,
            transaction_bytes=l2_transaction_bytes,
        )]
        if dram_bytes > 0:
            dram_transaction_bytes = transaction_bytes_by_role["dram"]
            if dram_transaction_bytes is None:
                raise ValueError(
                    "global GPU path split requires resolved DRAM transactions")
            nodes.append(MemoryNodeDemand(
                resource_role="dram",
                transactions=dram_bytes / dram_transaction_bytes,
                bytes=dram_bytes,
                transaction_bytes=dram_transaction_bytes,
            ))
        return MemoryPathDemand(
            name=name,
            access_kind=("read" if name.endswith("read") else "write"),
            initial_transactions=l2_bytes / l2_transaction_bytes,
            nodes=tuple(nodes),
            source=original.source,
            queues=original.queues,
        )

    paths = tuple(
        path for path in (
            make_path("global_read", read_l2_bytes, read_dram_bytes),
            make_path("global_write", write_l2_bytes, write_dram_bytes),
        )
        if path is not None
    )
    replacement = {"memory_paths": paths}
    if dependency_paths is not None:
        replacement["dependency_paths"] = dependency_paths
    return dataclasses.replace(inp, **replacement)


def append_npu_write_path(
        inp: PredictionInput, hardware, *,
        dram_write_bytes: Optional[float] = None,
        l2_write_bytes: Optional[float] = None,
        write_active_units: Optional[int] = None,
        prepend_aiv_mte2_ingress: bool = False) -> PredictionInput:
    """Attach a byte-only UB/MTE3 stream and proven L2/GM continuations."""

    import dataclasses

    if inp.mem_model != "npu_tile":
        raise ValueError("NPU write path requires mem_model='npu_tile'")
    if not isinstance(prepend_aiv_mte2_ingress, bool):
        raise TypeError("prepend_aiv_mte2_ingress must be boolean")
    if any(path.name == "mte3_write" for path in inp.memory_paths):
        raise ValueError("NPU MTE3 write path must be attached exactly once")
    write_bytes = float(inp.demand_per_resource["mte3"])
    if write_bytes <= 0:
        raise ValueError("NPU MTE3 write path requires positive write bytes")
    if dram_write_bytes is None:
        normalized_dram_write_bytes = write_bytes
    else:
        if (isinstance(dram_write_bytes, bool)
                or not isinstance(dram_write_bytes, (int, float))
                or not math.isfinite(dram_write_bytes)
                or dram_write_bytes < 0):
            raise ValueError(
                "NPU MTE3 DRAM write bytes must be finite and non-negative")
        normalized_dram_write_bytes = float(dram_write_bytes)
    if normalized_dram_write_bytes > write_bytes:
        raise ValueError(
            "NPU MTE3 DRAM continuation cannot exceed MTE3 source bytes")
    if l2_write_bytes is None:
        normalized_l2_write_bytes = 0.0
    else:
        if (isinstance(l2_write_bytes, bool)
                or not isinstance(l2_write_bytes, (int, float))
                or not math.isfinite(l2_write_bytes)
                or l2_write_bytes < 0):
            raise ValueError(
                "NPU MTE3 L2 write bytes must be finite and non-negative")
        normalized_l2_write_bytes = float(l2_write_bytes)
    if normalized_l2_write_bytes > write_bytes:
        raise ValueError(
            "NPU MTE3 L2 continuation cannot exceed MTE3 source bytes")
    if (normalized_l2_write_bytes > 0
            and normalized_dram_write_bytes > normalized_l2_write_bytes):
        raise ValueError(
            "NPU MTE3 DRAM continuation cannot exceed its proven L2 path")
    if hardware.kernel_memory_model != "npu_tile":
        raise ValueError("NPU write path requires npu_tile hardware")
    if write_active_units is not None:
        if (isinstance(write_active_units, bool)
                or not isinstance(write_active_units, int)):
            raise TypeError("NPU MTE3 active units must be an integer")
        normalized_write_active_units = write_active_units
    elif inp.launch_task_kind == "aiv":
        normalized_write_active_units = inp.active_units
    elif inp.launch_task_kind == "mixed":
        # Mixed CANN tasks express the launch grid in AIC block-dim units.  The
        # MTE3 producer is the corresponding AIV domain, so preserve the same
        # active fraction while converting to physical AIV cores.
        mixed_domain_units = hardware.execution_units_for_task("mixed")
        aiv_domain_units = hardware.execution_units_for_task("aiv")
        scaled_units = inp.active_units * aiv_domain_units
        if scaled_units % mixed_domain_units:
            raise ValueError(
                "mixed NPU task does not map to an integral AIV active count")
        normalized_write_active_units = scaled_units // mixed_domain_units
    elif inp.launch_task_kind == "aic":
        raise ValueError("AIC-only task cannot publish an AIV MTE3 write path")
    else:
        raise AssertionError(
            f"unhandled NPU write launch task: {inp.launch_task_kind!r}")
    mte3_units = hardware.resource_by_role("mte3").total_units
    if (normalized_write_active_units <= 0
            or normalized_write_active_units > mte3_units):
        raise ValueError(
            "NPU MTE3 active units exceed the physical AIV domain: "
            f"active={normalized_write_active_units}, physical={mte3_units}")
    nodes = [MemoryNodeDemand(
        resource_role="mte3",
        transactions=None,
        bytes=write_bytes,
        transaction_bytes=None,
    )]
    if normalized_l2_write_bytes > 0:
        nodes.append(MemoryNodeDemand(
            resource_role="l2_cache",
            transactions=None,
            bytes=normalized_l2_write_bytes,
            transaction_bytes=None,
        ))
    if normalized_dram_write_bytes > 0:
        nodes.append(MemoryNodeDemand(
            resource_role="dram",
            transactions=None,
            bytes=normalized_dram_write_bytes,
            transaction_bytes=None,
        ))
    path = MemoryPathDemand(
        name="mte3_write",
        access_kind="write",
        initial_transactions=None,
        nodes=tuple(nodes),
        source=MemorySourceDemand(
            active_units=normalized_write_active_units,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
    )
    flat = dict(inp.demand_per_resource)
    flat["l2_cache"] += normalized_l2_write_bytes
    flat["dram"] += normalized_dram_write_bytes
    memory_paths = (
        _npu_aiv_mte2_ingress_paths(inp, hardware)
        if prepend_aiv_mte2_ingress else inp.memory_paths)
    return dataclasses.replace(
        inp,
        demand_per_resource=flat,
        memory_paths=memory_paths + (path,),
    )


def _npu_aiv_mte2_ingress_paths(inp: PredictionInput, hardware):
    """Return memory paths with the source-defined AIV MTE2 ingress."""

    if (inp.mem_model != "npu_tile"
            or hardware.kernel_memory_model != "npu_tile"):
        raise ValueError("NPU AIV MTE2 ingress requires npu_tile memory")
    if inp.launch_task_kind != "aiv":
        raise ValueError("NPU AIV MTE2 ingress requires an AIV-only task")
    matches = tuple(
        (index, path)
        for index, path in enumerate(inp.memory_paths)
        if path.name == "mte2_read")
    if not matches:
        if inp.demand_per_resource["l2_cache"] != 0.0:
            raise ValueError(
                "positive NPU AIV reads require an mte2_read path")
        return inp.memory_paths
    if len(matches) != 1:
        raise ValueError(
            "NPU AIV MTE2 ingress requires exactly one mte2_read path")
    path_index, original = matches[0]
    if original.access_kind != "read":
        raise ValueError("mte2_read must be a read path")
    if original.nodes[0].resource_role != "l2_cache":
        raise ValueError(
            "NPU AIV MTE2 ingress requires an L2-first continuation path")
    if any(node.resource_role == "mte2" for node in original.nodes):
        raise ValueError("NPU AIV MTE2 ingress must be attached exactly once")
    mte2_units = hardware.resource_by_role("mte2").total_units
    if inp.active_units > mte2_units:
        raise ValueError(
            "NPU MTE2 active units exceed the physical AIV domain: "
            f"active={inp.active_units}, physical={mte2_units}")
    ingress = MemoryNodeDemand(
        resource_role="mte2",
        transactions=None,
        bytes=original.nodes[0].bytes,
        transaction_bytes=None,
    )
    paths = list(inp.memory_paths)
    paths[path_index] = dataclasses.replace(
        original, nodes=(ingress,) + original.nodes)
    return tuple(paths)


def prepend_npu_aiv_mte2_ingress(
        inp: PredictionInput, hardware) -> PredictionInput:
    """Prepend the physical per-AIV MTE2 endpoint to an NPU read path."""

    import dataclasses

    paths = _npu_aiv_mte2_ingress_paths(inp, hardware)
    if paths is inp.memory_paths:
        return inp
    return dataclasses.replace(inp, memory_paths=paths)


def attach_uniform_pipeline(
        inp: PredictionInput, hardware, *, task_count: int,
        stages: Tuple[Tuple[str, str, bool], ...], buffer_slots: int
        ) -> PredictionInput:
    """Attach a direct-grid homogeneous worker plan and remove old tail eta."""

    import dataclasses

    if isinstance(task_count, bool) or not isinstance(task_count, int):
        raise TypeError("pipeline task_count must be an integer")
    if task_count <= 0:
        raise ValueError("pipeline task_count must be positive")
    if isinstance(buffer_slots, bool) or not isinstance(buffer_slots, int):
        raise TypeError("pipeline buffer_slots must be an integer")
    if buffer_slots <= 0:
        raise ValueError("pipeline buffer_slots must be positive")
    if inp.pipeline_plan is not None:
        raise ValueError("pipeline plan must be attached exactly once")
    if not isinstance(stages, tuple) or not stages:
        raise TypeError("pipeline stages must be a non-empty tuple")
    worker_count = min(task_count, inp.active_units)
    fraction = 1.0 / task_count
    tasks = tuple(
        PipelineTaskDemand(
            name=f"worker{worker}_tiles",
            worker=worker,
            stages=tuple(
                bind_pipeline_stage_template(
                    name=stage_name,
                    service=service,
                    work_fraction=fraction,
                    include_first_request_latency=include_first,
                )
                for stage_name, service, include_first in stages
            ),
            repeat_count=(task_count - worker + worker_count - 1)
            // worker_count,
        )
        for worker in range(worker_count)
    )
    buffers = ()
    stage_names = tuple(stage[0] for stage in stages)
    if "load" in stage_names:
        load_index = stage_names.index("load")
        if load_index + 1 >= len(stage_names):
            if len(stage_names) != 1:
                raise ValueError(
                    "pipeline load stage must have a consumer")
        else:
            buffers += (PipelineBuffer(
                name="load_compute_buffer",
                producer_stage="load",
                consumer_stage=stage_names[load_index + 1],
                slots=buffer_slots,
                scope="worker",
            ),)
    if "store" in stage_names:
        store_index = stage_names.index("store")
        if store_index == 0:
            if len(stage_names) != 1:
                raise ValueError(
                    "pipeline store stage must have a producer")
        else:
            buffers += (PipelineBuffer(
                name="compute_store_buffer",
                producer_stage=stage_names[store_index - 1],
                consumer_stage="store",
                slots=buffer_slots,
                scope="worker",
            ),)

    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=tasks,
        buffers=buffers,
    )
    return attach_pipeline_work_plan(inp, hardware, plan)


def attach_pipeline_work_plan(
        inp: PredictionInput, hardware,
        plan: PipelineWorkPlan, *,
        promote_npu_partial: bool = False) -> PredictionInput:
    """Attach an explicit plan and remove only the legacy tail-wave factor."""

    import dataclasses

    if not isinstance(plan, PipelineWorkPlan):
        raise TypeError("plan must be PipelineWorkPlan")
    if inp.pipeline_plan is not None:
        raise ValueError("pipeline plan must be attached exactly once")
    if not isinstance(promote_npu_partial, bool):
        raise TypeError("promote_npu_partial must be boolean")
    if promote_npu_partial:
        if (inp.mem_model != "npu_tile"
                or inp.support_level != "aggregate_only"):
            raise ValueError(
                "NPU partial promotion requires an aggregate-only npu_tile "
                "input")
    elif inp.mem_model == "npu_tile":
        raise ValueError(
            "npu_tile pipeline attachment requires explicit partial-support "
            "promotion by its source-owning predictor")
    old_wave_efficiency = inp.wave_efficiency
    if not 0 < old_wave_efficiency <= 1:
        raise ValueError("invalid legacy wave efficiency")
    compute_demands = tuple(
        dataclasses.replace(
            demand,
            schedule_efficiency=min(
                1.0,
                demand.schedule_efficiency / old_wave_efficiency),
        )
        for demand in inp.compute_demands
    )
    support_level = "partial" if promote_npu_partial else inp.support_level
    return dataclasses.replace(
        inp,
        compute_demands=compute_demands,
        pipeline_plan=plan,
        wave_efficiency=1.0,
        support_level=support_level,
    )


def merge_operand_read_paths(paths, *, name="operand_read"):
    """Merge untyped requests competing for the same ordered ingress path.

    This preserves bytes and transactions at every node. It does not merge
    independent producers or typed services with different physical meanings.
    """
    if not paths or any(path.access_kind != "read" for path in paths):
        raise ValueError("operand merge requires nonempty read paths")
    first = paths[0]
    if any(path.source != first.source or path.queues != first.queues
           or path.access_class_demands for path in paths):
        raise ValueError("operand merge requires one untyped request source")
    roles = tuple(node.resource_role for node in first.nodes)
    if any(tuple(node.resource_role for node in path.nodes) != roles for path in paths):
        raise ValueError("operand merge requires matching hierarchy nodes")
    nodes = []
    for index, original in enumerate(first.nodes):
        items = tuple(path.nodes[index] for path in paths)
        if any((node.transaction_bytes, node.independent_service_streams,
                node.capacity_ceiling_resource_role) != (
                original.transaction_bytes, original.independent_service_streams,
                original.capacity_ceiling_resource_role) for node in items):
            raise ValueError("operand merge requires matching transaction semantics")
        nodes.append(dataclasses.replace(original, bytes=math.fsum(node.bytes for node in items),
            transactions=(None if original.transactions is None
                          else math.fsum(node.transactions for node in items))))
    return dataclasses.replace(first, name=name, nodes=tuple(nodes),
        initial_transactions=(None if first.initial_transactions is None
                              else math.fsum(path.initial_transactions for path in paths)))


def _attach_matmul_startup_issue_work(inp, hardware, task_count,
                                      startup_entry_warp_instructions):
    """Add source entry work once, independently of the scheduling envelope."""
    if startup_entry_warp_instructions is None:
        if any(d.resource_role == "warp_issue" for d in inp.compute_demands):
            if inp.phase_ledger is None:
                raise ValueError("existing warp issue requires an entry-startup phase contract")
            entry_work = math.fsum(d.operations
                for phase in inp.phase_ledger.phases for d in phase.compute_demands
                if d.resource_role == "warp_issue"
                and d.operation == "conservative_entry_startup_issue")
            output_work = math.fsum(d.operations
                for phase in inp.phase_ledger.phases for d in phase.compute_demands
                if d.resource_role == "warp_issue"
                and d.operation == "packed_output_BF16_conversion")
            issue_work = math.fsum(d.operations for d in inp.compute_demands
                if d.resource_role == "warp_issue")
            if entry_work + output_work != issue_work:
                raise ValueError("existing warp issue requires an entry-startup phase contract")
        return inp
    if (isinstance(startup_entry_warp_instructions, bool)
            or not isinstance(startup_entry_warp_instructions, int)
            or startup_entry_warp_instructions <= 0):
        raise ValueError("startup entry issue must be a positive instruction count")
    if (inp.demand_per_resource["warp_issue"] != 0.0
            or any(d.resource_role == "warp_issue" for d in inp.compute_demands)):
        raise ValueError("startup-only issue requires an unassigned warp issue resource")
    operations = float(startup_entry_warp_instructions * task_count
        * math.ceil(inp.block_size / hardware.parallel_width))
    demand = dict(inp.demand_per_resource)
    demand["warp_issue"] = operations
    inp = dataclasses.replace(inp, demand_per_resource=demand,
        compute_demands=inp.compute_demands + (ComputeDemand(
            resource_role="warp_issue", operations=operations,
            shape_efficiency=1.0, schedule_efficiency=1.0),))
    return inp


def attach_recurrent_matmul_pipeline(inp, hardware, *, task_count, k_tiles,
                                     buffer_slots, read_path_name, write_path_name,
                                     resident_ctas_per_unit=1, output_relayout_bytes=0.0,
                                     output_conversion_warp_instructions=0,
                                     task_memory_bytes=None,
                                     startup_entry_warp_instructions=None,
                                     matrix_preparation_source_work=None):
    """Coarse resident-CTA load/MMA/store schedule using existing services.

    The source owner supplies its stage count and populated workgroups. CTA
    assignment is a uniform analytical worker model, not observed placement.
    Concurrent CTA lanes share each unit's capacity equally; multiplying the
    lane count does not multiply any hardware resource capacity.
    Every read panel is a distinct asynchronous request burst. Its byte-port
    occupancy and all-return dependency are separate; the finite operand ring
    determines how much of that return latency can be hidden by other work.

    Optional source-ordered (read, write) byte pairs retain masked task tails.
    They partition the existing paths, including their aggregate downstream
    ratios; they do not infer per-task cache hits or change global traffic.
    A closed post-wait preparation prefix adds its issue work before each
    matrix stage on the same scheduler resource as entry work. It neither
    serializes an entire scalar loop nor adds an instruction latency price.
    """
    for name, value in (("task_count", task_count), ("k_tiles", k_tiles),
                        ("buffer_slots", buffer_slots),
                        ("resident_ctas_per_unit", resident_ctas_per_unit)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if hardware.gpu_store_completion_scope != "l2":
        raise ValueError("recurrent matmul store retirement requires coherent L2")
    if resident_ctas_per_unit > hardware.max_blocks_per_unit:
        raise ValueError("resident CTA count exceeds hardware block capacity")
    if (isinstance(output_relayout_bytes, bool)
            or not isinstance(output_relayout_bytes, (int, float))
            or not math.isfinite(output_relayout_bytes) or output_relayout_bytes < 0):
        raise ValueError("output_relayout_bytes must be finite non-negative bytes")
    if (isinstance(output_conversion_warp_instructions, bool)
            or not isinstance(output_conversion_warp_instructions, int)
            or output_conversion_warp_instructions < 0):
        raise ValueError("output conversion issue work must be a non-negative integer")
    max_read_scale = 1.0
    if task_memory_bytes is not None:
        if (not isinstance(task_memory_bytes, tuple)
                or len(task_memory_bytes) != task_count
                or any(not isinstance(pair, tuple) or len(pair) != 2
                       for pair in task_memory_bytes)):
            raise ValueError("task_memory_bytes requires one read/write pair per task")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0
               for pair in task_memory_bytes for value in pair):
            raise ValueError("task_memory_bytes must contain positive finite bytes")
        totals = []
        for index, name in enumerate((read_path_name, write_path_name)):
            paths = tuple(path for path in inp.memory_paths if path.name == name)
            if len(paths) != 1:
                raise ValueError("task byte partition requires one named memory path")
            nodes = tuple(node for node in paths[0].nodes
                          if node.resource_role == "l2_cache")
            if len(nodes) != 1:
                raise ValueError("task byte partition requires one L2 node per path")
            total = math.fsum(pair[index] for pair in task_memory_bytes)
            if not math.isclose(total, nodes[0].bytes, rel_tol=1e-12, abs_tol=1e-6):
                raise ValueError("task bytes must conserve the aggregate L2 path")
            totals.append(total)
        read_mean, write_mean = (total / task_count for total in totals)
        max_read_scale = max(pair[0] / read_mean for pair in task_memory_bytes)
        if len(set(task_memory_bytes)) == 1:
            task_memory_bytes = None
    operand_shared_fraction = 0.0
    output_shared_fraction = 0.0
    if output_relayout_bytes > 0:
        shared_bytes = inp.demand_per_resource["smem"]
        if output_relayout_bytes >= shared_bytes:
            raise ValueError("output relayout requires positive shared operand work")
        if ({d.resource_role for d in inp.compute_demands} - {"compute_tensor", "warp_issue"}
                or not any(d.resource_role == "compute_tensor" for d in inp.compute_demands)):
            raise ValueError("shared output relayout requires Tensor and optional startup issue")
        if {"smem", "compute_tensor"} & set(inp.resource_time_overrides_ns):
            raise ValueError("shared operand dominance cannot use time overrides")
        from src.kernel.formula import PerformanceFormula
        tensor, = tuple(d for d in inp.compute_demands if d.resource_role == "compute_tensor")
        tensor_capacity = PerformanceFormula(hardware)._compute_capacity(
            inp, hardware.resource_by_role("compute_tensor"))
        shared_capacity = hardware.resource_by_role("smem").effective_capacity(
            inp.active_units, sm_freq_hz=inp.sm_freq_hz, dram_freq_hz=inp.dram_freq_hz)
        operand_bytes = shared_bytes - output_relayout_bytes
        tensor_time = tensor.operations / (tensor_capacity * tensor.shape_efficiency
            * min(1.0, tensor.schedule_efficiency / inp.wave_efficiency))
        if operand_bytes * max_read_scale / shared_capacity > tensor_time:
            raise ValueError("operand shared service must finish within Tensor stage")
        # The operand SMEM capacity branch runs beside Tensor work. Its
        # shorter service on every task lets the matrix buffer retain its existing
        # release. The output fraction alone follows matrix completion.
        operand_shared_fraction = operand_bytes / shared_bytes
        output_shared_fraction = output_relayout_bytes / shared_bytes
    next_task_prior = f"memory:{write_path_name}"
    inp = _attach_matmul_startup_issue_work(inp, hardware, task_count,
        startup_entry_warp_instructions)
    has_startup_issue = any(d.resource_role == "warp_issue" for d in inp.compute_demands)
    entry_issue_fraction = 1.0 if has_startup_issue else 0.0
    preparation_issue_fraction = 0.0
    if matrix_preparation_source_work is not None:
        if (not isinstance(matrix_preparation_source_work, tuple)
                or len(matrix_preparation_source_work) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                       for v in matrix_preparation_source_work)):
            raise ValueError("matrix preparation requires positive source issue work and launch capacity")
        count, matrix_launch_capacity = matrix_preparation_source_work
        if "warp_issue" in inp.resource_time_overrides_ns:
            raise ValueError("matrix preparation cannot use warp issue time overrides")
        issues = tuple(d for d in inp.compute_demands if d.resource_role == "warp_issue")
        if len(issues) > 1:
            raise ValueError("matrix preparation requires one conserved issue resource")
        entry_operations = issues[0].operations if issues else 0.0
        if inp.demand_per_resource["warp_issue"] != entry_operations:
            raise ValueError("matrix preparation must conserve existing entry issue work")
        preparation_operations = float(count * task_count * k_tiles
            * math.ceil(inp.block_size / hardware.parallel_width))
        total_operations = entry_operations + preparation_operations
        demand = dict(inp.demand_per_resource)
        demand["warp_issue"] = total_operations
        if issues:
            if issues[0].shape_efficiency != 1.0 or issues[0].schedule_efficiency != 1.0:
                raise ValueError("source issue work requires unmodified scheduler capacity")
            issue_demand = dataclasses.replace(issues[0], operations=total_operations)
        else:
            issue_demand = ComputeDemand("warp_issue", total_operations, 1.0, 1.0)
        inp = dataclasses.replace(inp, demand_per_resource=demand,
            compute_demands=tuple(d for d in inp.compute_demands
                if d.resource_role != "warp_issue") + (issue_demand,),
            limitations=inp.limitations + (
                "matmul_post_wait_matrix_preparation_issue_from_source",
                "matmul_preparation_predicated_work_and_dependency_latency_omitted",))
        entry_issue_fraction = entry_operations / total_operations
        preparation_issue_fraction = preparation_operations / total_operations
    output_issue_fraction = 0.0
    if output_conversion_warp_instructions:
        if (inp.phase_ledger is not None
                and any(phase.name == "output_conversion" for phase in inp.phase_ledger.phases)):
            raise ValueError("output conversion must be attached exactly once")
        if "warp_issue" in inp.resource_time_overrides_ns:
            raise ValueError("output conversion cannot use warp issue time overrides")
        issues = tuple(d for d in inp.compute_demands if d.resource_role == "warp_issue")
        if len(issues) > 1:
            raise ValueError("output conversion requires one conserved issue resource")
        entry_operations = issues[0].operations if issues else 0.0
        if inp.demand_per_resource["warp_issue"] != entry_operations:
            raise ValueError("output conversion must conserve existing issue work")
        total_operations = entry_operations + output_conversion_warp_instructions
        retained_fraction = entry_operations / total_operations
        entry_issue_fraction *= retained_fraction
        preparation_issue_fraction *= retained_fraction
        output_issue_fraction = output_conversion_warp_instructions / total_operations
        demand = dict(inp.demand_per_resource)
        demand["warp_issue"] = total_operations
        if issues:
            issue_demand = dataclasses.replace(issues[0], operations=total_operations)
        else:
            issue_demand = ComputeDemand("warp_issue", total_operations, 1.0, 1.0)
        inp = dataclasses.replace(inp, demand_per_resource=demand,
            compute_demands=tuple(d for d in inp.compute_demands
                if d.resource_role != "warp_issue") + (issue_demand,),
            limitations=inp.limitations + (
                "matmul_output_packed_conversion_issue_from_compiled_source",
                "matmul_output_conversion_execution_capacity_remains_unpriced",))
    workers = min(task_count, inp.active_units * resident_ctas_per_unit)
    counts = tuple((task_count - worker + workers - 1) // workers for worker in range(workers))
    def cta_pattern(worker, read_scale=1.0, write_scale=1.0):
        def tick(name, *, first, repeat=1, after_preparation=False):
            stages = (
                PipelineStageDemand("operand_load", f"memory:{read_path_name}",
                    read_scale / (task_count * k_tiles),
                    include_first_request_latency=True,
                    burst_drains_at_path_capacity=True,
                    prior_resources=(("compute:warp_issue",) if after_preparation
                        else (next_task_prior,) if first else ())),
            )
            if preparation_issue_fraction:
                stages += (PipelineStageDemand("matrix_preparation_issue", "compute:warp_issue",
                    preparation_issue_fraction / (task_count * k_tiles)),)
            stages += (PipelineStageDemand("matrix", "compute:compute_tensor",
                1.0 / (task_count * k_tiles)),)
            if first and has_startup_issue:
                stages = (PipelineStageDemand("source_entry_issue", "compute:warp_issue",
                    entry_issue_fraction / task_count, predecessors=(),
                    prior_resources=(next_task_prior,)),) + stages
            if operand_shared_fraction:
                stages += (PipelineStageDemand("operand_shared", "resource:smem",
                    operand_shared_fraction * read_scale / (task_count * k_tiles),
                    predecessors=("operand_load",)),)
            return PipelineTaskDemand(name, worker, stages, repeat_count=repeat)
        if matrix_preparation_source_work is not None and buffer_slots > 1 and k_tiles >= buffer_slots:
            tasks = tuple(tick(f"k_{panel}", first=panel == 0,
                after_preparation=panel == buffer_slots - 1)
                for panel in range(buffer_slots))
            if k_tiles > buffer_slots:
                tasks += (tick("remaining_k_tiles", first=False,
                    repeat=k_tiles - buffer_slots),)
        else:
            tasks = (tick("first_k_tile", first=True),)
            if k_tiles > 1:
                tasks += (tick("remaining_k_tiles", first=False, repeat=k_tiles - 1),)
        store_stages = ()
        if output_issue_fraction:
            store_stages += (PipelineStageDemand(
                "output_conversion", "compute:warp_issue",
                output_issue_fraction / task_count, predecessors=(),
                prior_resources=("compute:compute_tensor",)),)
        if output_shared_fraction:
            store_stages += (PipelineStageDemand("output_shared", "resource:smem",
                output_shared_fraction / task_count,
                predecessors=("output_conversion",) if output_issue_fraction else (),
                prior_resources=("compute:compute_tensor",)),)
        store_stages += (PipelineStageDemand(
            "output_store", f"memory:{write_path_name}", write_scale / task_count,
            predecessors=("output_shared",) if output_shared_fraction else
                (("output_conversion",) if output_issue_fraction else ()),
            prior_resources=("compute:compute_tensor",),
            completion_resource_roles=("l2_cache",)),)
        tasks += (PipelineTaskDemand("store", worker, store_stages),)
        return PipelineTaskSequenceDemand("cta", worker, tuple(tasks))

    programs = []
    if task_memory_bytes is None:
        for count in dict.fromkeys(counts):
            physical = tuple(worker for worker, value in enumerate(counts) if value == count)
            worker = physical[0]
            programs.append(PipelineTaskProgramDemand(f"worker_{worker}_ctas", worker,
                (cta_pattern(worker),), (0,), repeat_count=count,
                worker_multiplicity=len(physical), physical_workers=physical))
    else:
        by_profile = {}
        for worker in range(workers):
            profile = task_memory_bytes[worker::workers]
            if profile not in by_profile:
                by_profile[profile] = []
            by_profile[profile].append(worker)
        for profile, physical in by_profile.items():
            worker = physical[0]
            patterns, order, indices = [], [], {}
            for pair, same in itertools.groupby(profile):
                repeats = sum(1 for _ in same)
                key = (pair, repeats)
                if key not in indices:
                    indices[key] = len(patterns)
                    sequence = cta_pattern(worker, pair[0] / read_mean, pair[1] / write_mean)
                    patterns.append(dataclasses.replace(sequence,
                        name=f"cta_{len(patterns)}", repeat_count=repeats))
                order.append(indices[key])
            programs.append(PipelineTaskProgramDemand(f"worker_{worker}_ctas", worker,
                tuple(patterns), tuple(order), worker_multiplicity=len(physical),
                physical_workers=tuple(physical)))
    buffers = (PipelineBuffer("operand_ring", "operand_load", "matrix", buffer_slots, "worker"),)
    if preparation_issue_fraction:
        buffers += (PipelineBuffer("matrix_launch_queue", "matrix_preparation_issue",
            "matrix", matrix_launch_capacity, "worker"),)
    plan = PipelineWorkPlan(workers, tuple(programs), buffers)
    return attach_pipeline_work_plan(inp, hardware, plan)



def attach_recurrent_operand_copy_issue(inp, *, task_count, k_tiles,
                                       copy_warp_instructions):
    """Add source-counted copy issue to an existing untyped resident graph.

    This optional transform preserves every existing service and task. Copy
    starts acquire the original operand ring; transport starts with issue,
    while preparation/consumption join issue and return. Prologue/tail work
    included in the supplied count is averaged over useful K panels. This
    does not claim native enqueue chronology or add a copy latency price.
    Typed complete-progress and single-panel aggregate envelopes require
    their own dependency treatment and are deliberately outside this API.
    """
    for name, value in (("task_count", task_count), ("k_tiles", k_tiles),
                        ("copy_warp_instructions", copy_warp_instructions)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    plan = inp.pipeline_plan
    if (not isinstance(plan, PipelineWorkPlan)
            or not all(isinstance(p, PipelineTaskProgramDemand) for p in plan.tasks)):
        raise ValueError("copy issue requires an existing recurrent resident graph")
    rings = tuple(b for b in plan.buffers if b.name == "operand_ring")
    if (len(rings) != 1 or rings[0].producer_stage != "operand_load"
            or rings[0].consumer_stage != "matrix" or rings[0].scope != "worker"):
        raise ValueError("copy issue requires the original worker operand ring")
    if any(p.access_class_demands for p in inp.memory_paths):
        raise ValueError("copy issue cannot reprice a typed complete-progress envelope")
    issues = tuple(d for d in inp.compute_demands if d.resource_role == "warp_issue")
    if (len(issues) > 1 or any(d.service_demands or d.exact_request_demands
            or d.shape_efficiency != 1.0 or d.schedule_efficiency != 1.0 for d in issues)
            or "warp_issue" in inp.resource_time_overrides_ns):
        raise ValueError("copy issue requires ordinary unmodified scheduler capacity")
    old_work = issues[0].operations if issues else 0.0
    if inp.demand_per_resource["warp_issue"] != old_work:
        raise ValueError("copy issue requires conserved original scheduler work")
    ledger = inp.phase_ledger
    if ledger is None or any(p.name == "operand_copy_issue" for p in ledger.phases):
        raise ValueError("copy issue requires a unique completed original phase ledger")
    total_work = old_work + copy_warp_instructions
    old_fraction = old_work / total_work
    copy_fraction = copy_warp_instructions / (total_work * task_count * k_tiles)
    programs = []
    ctas = 0
    for program in plan.tasks:
        patterns = []
        for pattern_index, sequence in enumerate(program.patterns):
            panels = sum(task.repeat_count for task in sequence.tasks
                         if any(s.name == "operand_load" for s in task.stages))
            if panels != k_tiles:
                raise ValueError("copy issue K count disagrees with recurrent source graph")
            occurrences = (program.order.count(pattern_index) * program.repeat_count
                           + program.tail_order.count(pattern_index))
            ctas += occurrences * sequence.repeat_count * program.worker_multiplicity
            tasks = []
            for task in sequence.tasks:
                names = tuple(s.name for s in task.stages)
                if "operand_copy_issue" in names:
                    raise ValueError("copy issue is already present in recurrent graph")
                stages = []
                for index, stage in enumerate(task.stages):
                    predecessors = stage.predecessors
                    if predecessors is None:
                        predecessors = (task.stages[index-1].name,) if index else ()
                    updated = stage
                    if stage.service == "compute:warp_issue":
                        if (stage.work_fraction is None or stage.compute_service_demands
                                or stage.compute_exact_request_demands):
                            raise ValueError("copy issue requires fractional old scheduler stages")
                        updated = dataclasses.replace(stage,
                            work_fraction=stage.work_fraction * old_fraction)
                    if stage.name == "operand_load":
                        if "matrix" not in names:
                            raise ValueError("copy issue panel has no matrix consumer")
                        stages.append(PipelineStageDemand("operand_copy_issue",
                            "compute:warp_issue", copy_fraction, predecessors=predecessors,
                            start_predecessors=stage.start_predecessors,
                            prior_resources=stage.prior_resources))
                        updated = dataclasses.replace(updated, predecessors=predecessors,
                            start_predecessors=stage.start_predecessors + ("operand_copy_issue",))
                    elif ("operand_load" in names and (stage.name in
                            ("matrix_preparation_issue", "matrix", "operand_shared"))):
                        updated = dataclasses.replace(updated,
                            predecessors=predecessors + ("operand_copy_issue",))
                    stages.append(updated)
                tasks.append(dataclasses.replace(task, stages=tuple(stages)))
            patterns.append(dataclasses.replace(sequence, tasks=tuple(tasks)))
        programs.append(dataclasses.replace(program, patterns=tuple(patterns)))
    if ctas != task_count:
        raise ValueError("copy issue CTA count disagrees with recurrent source graph")
    new_plan = dataclasses.replace(plan, tasks=tuple(programs), buffers=tuple(
        dataclasses.replace(b, producer_stage="operand_copy_issue") if b is rings[0] else b
        for b in plan.buffers))
    issue = (dataclasses.replace(issues[0], operations=total_work) if issues
             else ComputeDemand("warp_issue", total_work, 1.0, 1.0))
    demands = tuple(issue if d is issues[0] else d for d in inp.compute_demands) if issues else (
        inp.compute_demands + (issue,))
    # The ledger records a distinct capacity inventory, not a launch-wide
    # startup fence. Local issue/transport joins are solely in the work plan.
    phase = AnalyticalPhase("operand_copy_issue", (),
        (PhaseComputeDemand("warp_issue", "source_operand_copy_warp_issue",
                            float(copy_warp_instructions)),), ())
    ledger = dataclasses.replace(ledger,
        phases=ledger.phases[:-1] + (phase,) + ledger.phases[-1:])
    demand = dict(inp.demand_per_resource)
    demand["warp_issue"] = total_work
    return dataclasses.replace(inp, pipeline_plan=new_plan, phase_ledger=ledger,
        demand_per_resource=demand, compute_demands=demands,
        limitations=inp.limitations + (
            "source_copy_issue_quantity_added_to_existing_resident_graph",
            "source_copy_prefill_tail_issue_averaged_over_useful_K_panels",
            "source_copy_transport_starts_at_issue_and_consumer_joins_completion",))


def attach_prefetched_recurrent_operand_copy_issue(inp, *, task_count, k_tiles,
                                                  prefetch_panels,
                                                  copy_warp_instructions):
    """Order one coarse CTA as source prefill, consume/refill, and output.

    Unlike putting copy and preparation in every K task, source prefill
    issues upcoming panels before reserving the first post-return issue
    stage. All original services remain; cross-panel dependency edges retain
    the original storage and matrix-launch capacities. K is expanded only
    inside a representative CTA, never across all physical CTAs or warps.
    The supplied count includes prefill plus every loop refill, including
    zero-source tail copies. Tail issue is retained without inventing global
    bytes for unused panels or repricing the original shared-byte proxy.
    """
    result = attach_recurrent_operand_copy_issue(inp, task_count=task_count,
        k_tiles=k_tiles, copy_warp_instructions=copy_warp_instructions)
    plan = inp.pipeline_plan
    ring, = tuple(b for b in plan.buffers if b.name == "operand_ring")
    if (isinstance(prefetch_panels, bool) or not isinstance(prefetch_panels, int)
            or not 1 <= prefetch_panels <= ring.slots):
        raise ValueError("source prefetch must fit the physical operand ring")
    queues = tuple(b for b in plan.buffers if b.name == "matrix_launch_queue")
    if len(queues) > 1 or any(b.name not in ("operand_ring", "matrix_launch_queue")
                              or b.release != "completion" for b in plan.buffers):
        raise ValueError("source prefetch requires completion-owned resident queues")
    total_issue = result.demand_per_resource["warp_issue"]
    old_fraction = inp.demand_per_resource["warp_issue"] / total_issue
    copy_fraction = copy_warp_instructions / (
        total_issue * task_count * (k_tiles + prefetch_panels))
    programs = []
    for program in plan.tasks:
        patterns = []
        for sequence in program.patterns:
            first = sequence.tasks[0]
            steady = sequence.tasks[1] if k_tiles > 1 else first
            store = sequence.tasks[-1]
            if len(sequence.tasks) != (3 if k_tiles > 1 else 2):
                raise ValueError("source prefetch requires first/steady/store CTA patterns")
            stages = []
            entries = tuple(s for s in first.stages if s.name == "source_entry_issue")
            if len(entries) > 1:
                raise ValueError("source prefetch has ambiguous entry work")
            entry_names = tuple(s.name for s in entries)
            stages.extend(dataclasses.replace(s, work_fraction=s.work_fraction * old_fraction)
                          for s in entries)
            def produce(panel):
                load, = tuple(s for s in first.stages if s.name == "operand_load")
                ready = entry_names
                if panel >= ring.slots:
                    ready += (f"matrix_{panel-ring.slots}",)
                copy_name = f"operand_copy_issue_{panel}"
                stages.append(PipelineStageDemand(copy_name, "compute:warp_issue",
                    copy_fraction, predecessors=ready,
                    prior_resources=load.prior_resources if panel == 0 else ()))
                if panel < k_tiles:
                    source_task = first if panel == 0 else steady
                    load, = tuple(s for s in source_task.stages if s.name == "operand_load")
                    stages.append(dataclasses.replace(load, name=f"operand_load_{panel}",
                        predecessors=entry_names, start_predecessors=(copy_name,),
                        prior_resources=load.prior_resources if panel == 0 else ()))
            for panel in range(prefetch_panels):
                produce(panel)
            for panel in range(k_tiles):
                task = first if panel == 0 else steady
                ready = (f"operand_load_{panel}", f"operand_copy_issue_{panel}")
                preparation = tuple(s for s in task.stages if s.name == "matrix_preparation_issue")
                if bool(preparation) != bool(queues) or len(preparation) > 1:
                    raise ValueError("source prefetch preparation queue disagrees with its work")
                if preparation:
                    capacity = queues[0].slots
                    if panel >= capacity:
                        ready += (f"matrix_{panel-capacity}",)
                    prep, = preparation
                    name = f"matrix_preparation_issue_{panel}"
                    stages.append(dataclasses.replace(prep, name=name, predecessors=ready,
                        work_fraction=prep.work_fraction * old_fraction))
                    ready += (name,)
                matrix, = tuple(s for s in task.stages if s.name == "matrix")
                stages.append(dataclasses.replace(matrix, name=f"matrix_{panel}", predecessors=ready))
                shared = tuple(s for s in task.stages if s.name == "operand_shared")
                stages.extend(dataclasses.replace(s, name=f"operand_shared_{panel}",
                    predecessors=(f"operand_load_{panel}", f"operand_copy_issue_{panel}"))
                    for s in shared)
                expected = {"source_entry_issue", "operand_load", "matrix_preparation_issue",
                            "matrix", "operand_shared"}
                if any(s.name not in expected for s in task.stages):
                    raise ValueError("source prefetch encountered an unknown resident stage")
                produce(panel + prefetch_panels)
            output_ready = (f"matrix_{k_tiles-1}",
                            f"operand_copy_issue_{k_tiles+prefetch_panels-1}")
            # The old resident envelope already proves operand shared service
            # finishes within matrix; keep its output work and retirement.
            for index, stage in enumerate(store.stages):
                stages.append(dataclasses.replace(stage,
                    predecessors=output_ready if index == 0 else (store.stages[index-1].name,)))
            task = PipelineTaskDemand("source_prefetched_cta", sequence.worker, tuple(stages))
            patterns.append(dataclasses.replace(sequence, tasks=(task,)))
        programs.append(dataclasses.replace(program, patterns=tuple(patterns)))
    # The explicit matrix_{i-S} and preparation-capacity edges above encode
    # the original physical queues, rather than changing their slot counts.
    plan = dataclasses.replace(plan, tasks=tuple(programs), buffers=())
    return dataclasses.replace(result, pipeline_plan=plan,
        limitations=inp.limitations + (
            "source_prefill_precedes_post_return_scheduler_reservations",
            "source_explicit_CTA_K_edges_preserve_ring_and_matrix_queue_capacity",
            "source_zero_source_tail_copy_issue_retained_with_original_byte_budgets",))


def attach_recurrent_matmul_scalar_epilogue(inp, *, task_count, read_path,
                                           scalar_multiplications,
                                           prefetch_read=False):
    """Attach a source scalar epilogue with a late or prefetched read.

    The scalar multiply waits for both the matrix result and its weight
    read, then precedes the original conversion/layout/store work. The
    default retains the explicit late-read diagnostic endpoint. Prefetch
    mode issues the independent address read with the first K task; it is an
    analytical overlap approximation, not an observed native schedule. The
    caller binds the read's source, dtype, traffic and cache assumptions.
    """
    for name, value in (("task_count", task_count),
                        ("scalar_multiplications", scalar_multiplications)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(prefetch_read, bool):
        raise TypeError("scalar epilogue prefetch_read must be boolean")
    plan = inp.pipeline_plan
    if (not isinstance(plan, PipelineWorkPlan)
            or not all(isinstance(p, PipelineTaskProgramDemand) for p in plan.tasks)):
        raise ValueError("scalar epilogue requires an existing recurrent CTA graph")
    if (not isinstance(read_path, MemoryPathDemand) or read_path.access_kind != "read"
            or any(p.name == read_path.name for p in inp.memory_paths)
            or read_path.access_class_demands):
        raise ValueError("scalar epilogue requires a distinct ordinary read path")
    if (any(d.resource_role == "compute_fma" for d in inp.compute_demands)
            or inp.demand_per_resource["compute_fma"] != 0
            or "compute_fma" in inp.resource_time_overrides_ns):
        raise ValueError("scalar epilogue requires an unassigned FP32 resource")
    ledger = inp.phase_ledger
    if (ledger is None or ledger.phases[-1].name != "output_writeback"
            or any(p.name == "source_scalar_epilogue_read" for p in ledger.phases)):
        raise ValueError("scalar epilogue requires the original output ledger")
    programs = []; ctas = 0
    for program in plan.tasks:
        patterns = []
        for index, sequence in enumerate(program.patterns):
            if sequence.tasks[-1].name != "store":
                raise ValueError("scalar epilogue requires one final CTA store task")
            ctas += ((program.order.count(index)*program.repeat_count
                      + program.tail_order.count(index))
                     * sequence.repeat_count * program.worker_multiplicity)
            store = sequence.tasks[-1]
            if prefetch_read:
                first_task = sequence.tasks[0]
                if first_task.stages[0].predecessors not in (None, ()):
                    raise ValueError("scalar read prefetch requires an independent first stage")
                read_stage = PipelineStageDemand(
                    "source_scalar_epilogue_read", f"memory:{read_path.name}",
                    1.0/task_count, predecessors=(),
                    prior_resources=(store.stages[-1].service,),
                    include_first_request_latency=True, burst_drains_at_path_capacity=True)
                original_first = dataclasses.replace(
                    first_task.stages[0], predecessors=())
                first_task = dataclasses.replace(
                    first_task, stages=(read_stage, original_first) + first_task.stages[1:])
                multiply = PipelineStageDemand(
                    "source_scalar_epilogue_multiply", "compute:compute_fma",
                    1.0/task_count, predecessors=(),
                    prior_resources=(f"memory:{read_path.name}", "compute:compute_tensor"))
                sequence = dataclasses.replace(sequence, tasks=(first_task,) + sequence.tasks[1:])
            else:
                multiply = PipelineStageDemand(
                    "source_scalar_epilogue_multiply", "compute:compute_fma",
                    1.0/task_count, predecessors=("source_scalar_epilogue_read",))
                read_stage = PipelineStageDemand(
                    "source_scalar_epilogue_read", f"memory:{read_path.name}",
                    1.0/task_count, predecessors=(),
                    prior_resources=("compute:compute_tensor",),
                    include_first_request_latency=True, burst_drains_at_path_capacity=True)
            prefix = (multiply,) if prefetch_read else (read_stage, multiply)
            first = dataclasses.replace(store.stages[0],
                predecessors=("source_scalar_epilogue_multiply",))
            store = dataclasses.replace(store, stages=prefix+(first,)+store.stages[1:])
            patterns.append(dataclasses.replace(sequence,tasks=sequence.tasks[:-1]+(store,)))
        programs.append(dataclasses.replace(program,patterns=tuple(patterns)))
    if ctas != task_count:
        raise ValueError("scalar epilogue CTA count disagrees with the source graph")
    demand = dict(inp.demand_per_resource)
    demand["compute_fma"] = float(scalar_multiplications)
    for node in read_path.nodes:
        demand[node.resource_role] += node.bytes
    scalar = ComputeDemand("compute_fma",float(scalar_multiplications),0.5,1.0)
    output = ledger.phases[-1]
    conversion = (ledger.phases[-2]
        if len(ledger.phases) >= 2 and ledger.phases[-2].name == "output_conversion" else None)
    anchor = output if conversion is None else conversion
    read_predecessors = () if prefetch_read else anchor.predecessors
    multiply_predecessors = (
        ("source_scalar_epilogue_read",) + anchor.predecessors
        if prefetch_read else ("source_scalar_epilogue_read",))
    read = AnalyticalPhase("source_scalar_epilogue_read",read_predecessors,(),
        tuple(PhaseMemoryDemand(read_path.name,n.resource_role,n.bytes) for n in read_path.nodes))
    multiply = AnalyticalPhase("source_scalar_epilogue_multiply",multiply_predecessors,
        (PhaseComputeDemand("compute_fma","FP32_scalar_row_multiply",
                            float(scalar_multiplications)),),())
    if conversion is None:
        phases = ledger.phases[:-1] + (read, multiply,
            dataclasses.replace(output, predecessors=(multiply.name,)))
    else:
        phases = ledger.phases[:-2] + (read, multiply,
            dataclasses.replace(conversion, predecessors=(multiply.name,)), output)
    ledger = dataclasses.replace(ledger, phases=phases)
    limitations = (
        "source_scalar_epilogue_prefetched_read_is_analytical_not_native_observation",
    ) if prefetch_read else (
        "source_scalar_epilogue_late_read_is_diagnostic_not_native_hoisting_proof",
    )
    return dataclasses.replace(inp,demand_per_resource=demand,
        compute_demands=inp.compute_demands+(scalar,),memory_paths=inp.memory_paths+(read_path,),
        pipeline_plan=dataclasses.replace(plan,tasks=tuple(programs)),phase_ledger=ledger,
        limitations=inp.limitations+(
            "source_scalar_row_epilogue_after_matrix_before_output",) + limitations)


def attach_single_panel_prefetched_scalar_epilogue(inp, *, read_path,
                                                  scalar_multiplications):
    """Add a prefetched scalar read and post-matrix multiply to one aggregate task."""
    if (isinstance(scalar_multiplications, bool)
            or not isinstance(scalar_multiplications, int) or scalar_multiplications <= 0):
        raise ValueError("scalar_multiplications must be a positive integer")
    plan = inp.pipeline_plan
    if not isinstance(plan, PipelineWorkPlan) or len(plan.tasks) != 1:
        raise ValueError("single-panel scalar epilogue requires one aggregate task")
    task = plan.tasks[0]
    if (not isinstance(task, PipelineTaskDemand)
            or task.name != "single_panel_aggregate" or task.repeat_count != 1):
        raise ValueError("single-panel scalar epilogue requires one aggregate task")
    if (not isinstance(read_path, MemoryPathDemand) or read_path.access_kind != "read"
            or any(path.name == read_path.name for path in inp.memory_paths)
            or read_path.access_class_demands):
        raise ValueError("scalar epilogue requires a distinct ordinary read path")
    if (any(demand.resource_role == "compute_fma" for demand in inp.compute_demands)
            or inp.demand_per_resource["compute_fma"] != 0
            or "compute_fma" in inp.resource_time_overrides_ns):
        raise ValueError("scalar epilogue requires an unassigned FP32 resource")
    ledger = inp.phase_ledger
    if (ledger is None or ledger.phases[-1].name != "output_writeback"
            or any(phase.name == "source_scalar_epilogue_read" for phase in ledger.phases)):
        raise ValueError("scalar epilogue requires the original output ledger")
    names = tuple(stage.name for stage in task.stages)
    if names.count("matrix") != 1 or names.count("output_store") != 1:
        raise ValueError("single-panel scalar epilogue requires matrix and output_store stages")
    output_name = ("output_conversion" if "output_conversion" in names else
                   "output_shared" if "output_shared" in names else "output_store")
    output_index = names.index(output_name)
    read_stage = PipelineStageDemand(
        "source_scalar_epilogue_read", f"memory:{read_path.name}", 1.0,
        predecessors=(), include_first_request_latency=True,
        burst_drains_at_path_capacity=True)
    multiply = PipelineStageDemand(
        "source_scalar_epilogue_multiply", "compute:compute_fma", 1.0,
        predecessors=("matrix", "source_scalar_epilogue_read"))
    output = dataclasses.replace(
        task.stages[output_index],
        predecessors=task.stages[output_index].predecessors + (
            "source_scalar_epilogue_multiply",))
    stages = ((read_stage,) + task.stages[:output_index]
              + (multiply, output) + task.stages[output_index + 1:])
    demand = dict(inp.demand_per_resource)
    demand["compute_fma"] = float(scalar_multiplications)
    for node in read_path.nodes:
        demand[node.resource_role] += node.bytes
    scalar = ComputeDemand("compute_fma", float(scalar_multiplications), 0.5, 1.0)
    output = ledger.phases[-1]
    conversion = (ledger.phases[-2]
        if len(ledger.phases) >= 2 and ledger.phases[-2].name == "output_conversion" else None)
    anchor = output if conversion is None else conversion
    read = AnalyticalPhase("source_scalar_epilogue_read", (), (),
        tuple(PhaseMemoryDemand(read_path.name, node.resource_role, node.bytes)
              for node in read_path.nodes))
    multiply_phase = AnalyticalPhase(
        "source_scalar_epilogue_multiply",
        ("source_scalar_epilogue_read",) + anchor.predecessors,
        (PhaseComputeDemand("compute_fma", "FP32_scalar_row_multiply",
                            float(scalar_multiplications)),), ())
    if conversion is None:
        phases = ledger.phases[:-1] + (read, multiply_phase,
            dataclasses.replace(output, predecessors=(multiply_phase.name,)))
    else:
        phases = ledger.phases[:-2] + (read, multiply_phase,
            dataclasses.replace(conversion, predecessors=(multiply_phase.name,)), output)
    ledger = dataclasses.replace(ledger, phases=phases)
    return dataclasses.replace(
        inp, demand_per_resource=demand,
        compute_demands=inp.compute_demands + (scalar,),
        memory_paths=inp.memory_paths + (read_path,),
        pipeline_plan=dataclasses.replace(
            plan, tasks=(dataclasses.replace(task, stages=stages),)),
        phase_ledger=ledger,
        limitations=inp.limitations + (
            "source_scalar_row_epilogue_after_matrix_before_output",
            "single_panel_scalar_epilogue_prefetched_read_is_analytical_not_native_observation",))


def _source_tiled_shared_budget(original_bytes, *, task_count, k_tiles,
                                panel_bytes, output_bytes, panel_read_bytes=None):
    """Logical padded operand write/read work, retaining every old byte.

    Runtime masks suppress global reads, but the staged dot operands still
    represent full A/B tiles. One logical write and read per panel is a
    coarse transport estimate, not a bank-transaction observation. Explicit
    panel_read_bytes can retain source-bound register-load work or repeated
    shared-operand feeds across output warpgroups. Existing larger budgets
    are retained on the read side; output work is unchanged.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
           for v in (task_count, k_tiles, panel_bytes)):
        raise ValueError("source tile work requires positive physical counts/bytes")
    if (any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) for v in (original_bytes, output_bytes))
            or original_bytes <= 0 or output_bytes < 0):
        raise ValueError("source shared/output budgets require finite physical bytes")
    if panel_read_bytes is None:
        panel_read_bytes = panel_bytes
    if (isinstance(panel_read_bytes, bool) or not isinstance(panel_read_bytes, int)
            or panel_read_bytes <= 0):
        raise ValueError("source panel reads require positive bound bytes")
    ingress = task_count * k_tiles * panel_bytes
    reads = task_count * k_tiles * panel_read_bytes
    total = max(original_bytes, ingress + reads + output_bytes)
    return ingress, total - ingress - output_bytes, output_bytes


def _single_panel_source_cta_plan(inp, *, task_count, k_tiles,
        resident_ctas_per_unit, read_path_name, write_path_name,
        shared_fractions, task_memory_bytes):
    """Compatibility entry for the source-level one-buffer prototype."""
    return _source_tiled_cta_plan(inp, task_count=task_count, k_tiles=k_tiles,
        buffer_slots=1, resident_ctas_per_unit=resident_ctas_per_unit,
        read_path_name=read_path_name, write_path_name=write_path_name,
        shared_fractions=shared_fractions, task_memory_bytes=task_memory_bytes)


def _source_tiled_cta_plan(inp, *, task_count, k_tiles, buffer_slots,
        resident_ctas_per_unit, read_path_name, write_path_name,
        shared_fractions, task_memory_bytes, preparation_fraction=0.0,
        preparation_slots=None, post_read_path_name=None, post_compute_role=None,
        copy_issue_fraction=0.0):
    """Bounded-prefetch K loops within independent output-tile CTAs.

    Source tl.dot accumulation orders each CTA's K steps. Its private operand
    buffer does not impose a launch-wide load/compute/store barrier. Resident
    workers retain the existing equal capacity sharing approximation.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
           for v in (task_count, k_tiles, buffer_slots, resident_ctas_per_unit)):
        raise ValueError("source requires positive CTA/K/buffer/residency counts")
    if (not isinstance(shared_fractions, tuple) or len(shared_fractions) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v < 0 for v in shared_fractions)
            or shared_fractions[0] <= 0 or shared_fractions[1] <= 0
            or not math.isclose(math.fsum(shared_fractions), 1.0, rel_tol=1e-12)):
        raise ValueError("source shared phases must conserve one complete shared demand")
    if task_memory_bytes is None:
        task_memory_bytes = ((1.0, 1.0),) * task_count
    if (not isinstance(task_memory_bytes, tuple) or len(task_memory_bytes) != task_count
            or any(not isinstance(pair, tuple) or len(pair) != 2
                   or any(isinstance(value, bool) or not isinstance(value, (int, float))
                          or not math.isfinite(value) or value <= 0 for value in pair)
                   for pair in task_memory_bytes)):
        raise ValueError("single-panel tasks require positive source read/write pairs")
    read_mean = math.fsum(pair[0] for pair in task_memory_bytes) / task_count
    write_mean = math.fsum(pair[1] for pair in task_memory_bytes) / task_count
    workers = min(task_count, int(inp.active_units) * resident_ctas_per_unit)
    ingress, reads, output = shared_fractions
    startup = any(d.resource_role == "warp_issue" for d in inp.compute_demands)
    if (isinstance(preparation_fraction, bool)
            or not isinstance(preparation_fraction, (int, float))
            or not math.isfinite(preparation_fraction) or not 0 <= preparation_fraction <= 1):
        raise ValueError("source preparation fraction must lie within issue work")
    if (isinstance(copy_issue_fraction,bool) or not isinstance(copy_issue_fraction,(int,float))
            or not math.isfinite(copy_issue_fraction) or not 0<=copy_issue_fraction<=1
            or preparation_fraction+copy_issue_fraction>1):
        raise ValueError("source copy/preparation fractions must conserve issue work")
    if copy_issue_fraction and not startup:
        raise ValueError("source copy issue requires explicit warp-issue work")
    if preparation_fraction:
        if (not startup or isinstance(preparation_slots, bool)
                or not isinstance(preparation_slots, int) or preparation_slots <= 0):
            raise ValueError("source preparation requires issue work and bounded queue")
    elif preparation_slots is not None:
        raise ValueError("source preparation queue has no corresponding work")
    if (post_read_path_name is None) != (post_compute_role is None):
        raise ValueError("source post-matmul read and compute must be bound together")
    if post_read_path_name is not None:
        if (not isinstance(post_read_path_name, str)
                or post_read_path_name in (read_path_name, write_path_name)
                or sum(p.name == post_read_path_name and p.access_kind == "read"
                       for p in inp.memory_paths) != 1
                or sum(d.resource_role == post_compute_role and d.operations > 0
                       for d in inp.compute_demands) != 1):
            raise ValueError("source post-matmul work requires distinct bound services")

    def cta(worker, pair, name, repeats):
        read_scale, write_scale = pair[0] / read_mean, pair[1] / write_mean
        def tick(first):
            stages = ()
            if first and startup and preparation_fraction+copy_issue_fraction < 1:
                stages += (PipelineStageDemand("source_entry_issue", "compute:warp_issue",
                    (1.0 - preparation_fraction - copy_issue_fraction) / task_count, predecessors=(),
                    prior_resources=(f"memory:{write_path_name}",)),)
            if copy_issue_fraction:
                stages += (PipelineStageDemand("operand_copy_issue","compute:warp_issue",
                    copy_issue_fraction/(task_count*k_tiles),predecessors=()),)
            stages += (
                PipelineStageDemand("operand_load", f"memory:{read_path_name}",
                    read_scale / (task_count * k_tiles),
                    predecessors=() if copy_issue_fraction else None,
                    start_predecessors=("operand_copy_issue",) if copy_issue_fraction else (),
                    include_first_request_latency=True, burst_drains_at_path_capacity=True,
                    prior_resources=(f"memory:{write_path_name}",) if first else ()),
                PipelineStageDemand("operand_shared_ingress_bound", "resource:smem",
                    ingress / (task_count * k_tiles), predecessors=(),
                    start_predecessors=("operand_load",)),
            )
            ready = ("operand_load", "operand_shared_ingress_bound")
            if copy_issue_fraction:ready+=("operand_copy_issue",)
            matrix_ready = ready
            if preparation_fraction:
                stages += (PipelineStageDemand("matrix_preparation_issue", "compute:warp_issue",
                    preparation_fraction / (task_count * k_tiles), predecessors=(
                        ("operand_load","operand_copy_issue") if copy_issue_fraction else ("operand_load",))),)
                matrix_ready += ("matrix_preparation_issue",)
            stages += (
                PipelineStageDemand("matrix", "compute:compute_tensor",
                    1.0 / (task_count * k_tiles), predecessors=matrix_ready),
                PipelineStageDemand("operand_shared_read", "resource:smem",
                    reads / (task_count * k_tiles), predecessors=ready),
            )
            return PipelineTaskDemand("first_k_tile" if first else "remaining_k_tiles",
                worker, stages, repeat_count=1 if first else k_tiles - 1)
        tasks = (tick(True),)
        if k_tiles > 1:
            tasks += (tick(False),)
        stages = ()
        if post_read_path_name is not None:
            # Uniform per-CTA read share is an explicit aggregate approximation;
            # it does not reconstruct the valid row count of each router load.
            stages += (
                PipelineStageDemand("source_post_read", f"memory:{post_read_path_name}",
                    1.0 / task_count, predecessors=(),
                    include_first_request_latency=True, burst_drains_at_path_capacity=True,
                    prior_resources=("compute:compute_tensor", "resource:smem")),
                PipelineStageDemand("source_post_compute", f"compute:{post_compute_role}",
                    1.0 / task_count, predecessors=("source_post_read",)),
            )
        if output:
            stages += (PipelineStageDemand("output_shared", "resource:smem",
                output / task_count,
                predecessors=("source_post_compute",) if post_read_path_name is not None else (),
                prior_resources=("compute:compute_tensor", "resource:smem")),)
        stages += (PipelineStageDemand("output_store", f"memory:{write_path_name}",
            write_scale / task_count, predecessors=(("output_shared",) if output else
                ("source_post_compute",) if post_read_path_name is not None else ()),
            prior_resources=("compute:compute_tensor", "resource:smem"),
            completion_resource_roles=("l2_cache",)),)
        tasks += (PipelineTaskDemand("store", worker, stages),)
        return PipelineTaskSequenceDemand(name, worker, tasks, repeat_count=repeats)

    profiles = {}
    for worker in range(workers):
        profile = task_memory_bytes[worker::workers]
        if profile not in profiles:
            profiles[profile] = []
        profiles[profile].append(worker)
    programs = []
    for profile, physical in profiles.items():
        worker = physical[0]; patterns, order, indices = [], [], {}
        for pair, grouped in itertools.groupby(profile):
            repeats = sum(1 for _ in grouped); key = pair, repeats
            if key not in indices:
                indices[key] = len(patterns)
                patterns.append(cta(worker, pair, f"cta_{len(patterns)}", repeats))
            order.append(indices[key])
        programs.append(PipelineTaskProgramDemand(f"worker_{worker}_ctas", worker,
            tuple(patterns), tuple(order), worker_multiplicity=len(physical),
            physical_workers=tuple(physical)))
    buffers = (
        PipelineBuffer("operand_ring", "operand_load", "matrix", buffer_slots, "worker"),
        PipelineBuffer("operand_read_reuse", "operand_load", "operand_shared_read", buffer_slots, "worker"),
    )
    if preparation_fraction:
        buffers += (PipelineBuffer("matrix_launch_queue", "matrix_preparation_issue",
            "matrix", preparation_slots, "worker"),)
    return PipelineWorkPlan(workers, tuple(programs), buffers)


def attach_single_panel_matmul_pipeline(inp, hardware, *, task_count, dtype,
                                        tile_m, tile_k, panel_bytes,
                                        resident_ctas_per_unit, row_stride_bytes,
                                        read_path_name, write_path_name,
                                        startup_entry_warp_instructions=None,
                                        source_shared_work=None,
                                        output_conversion_warp_instructions=0):
    """Aggregate a measured single-panel class with reclaimable shared service.

    The byte-only partial-cohort evidence admits work conservation instead of
    permanently reserving bandwidth for completed CTA lanes. Whole operand,
    matrix/shared and output phases remain serialized as a coarse envelope.
    When compiled shared byte facts are supplied, asynchronous destination
    writes compete during ingress, register-operand reads follow ingress, and
    output rearrangement follows matrix completion. Native placement, fine
    overlap and per-task cache ratios stay unbound.
    """
    if (inp.mem_model != "gpu_mshr" or inp.pipeline_plan is not None
            or inp.sm_freq_hz != hardware.nominal_sm_freq_hz):
        return inp
    name = (f"single_panel_async_copy_threads{inp.block_size}"
            f"_resident{resident_ctas_per_unit}_b{panel_bytes}")
    signature = MemoryAccessClassSignature(access_class=name, dtype=dtype,
        tile_m=tile_m, tile_k=tile_k, logical_row_bytes=row_stride_bytes)
    services = tuple(s for s in hardware.memory_access_class_services
        if s.service == name and s.signature == signature
        and s.access_kind == "read" and s.ingress_resource_role == "l2_cache")
    if not services:
        return inp
    if len(services) != 1:
        raise ValueError("single-panel progress requires one exact service")
    service, = services
    table = service.logical_byte_capacity_per_s_by_grid_ctas
    if table is None:
        raise ValueError("single-panel progress requires measured cohorts")
    if task_count not in tuple(grid for grid, _ in table):
        return inp
    paths = tuple(p for p in inp.memory_paths if p.name == read_path_name)
    if len(paths) != 1 or paths[0].access_kind != "read":
        raise ValueError("single-panel envelope requires one operand path")
    path, = paths
    if path.access_class_demands:
        return inp
    if (isinstance(output_conversion_warp_instructions, bool)
            or not isinstance(output_conversion_warp_instructions, int)
            or output_conversion_warp_instructions < 0):
        raise ValueError("single-panel output conversion issue must be a non-negative integer")
    inp = _attach_matmul_startup_issue_work(inp, hardware, task_count,
        startup_entry_warp_instructions)
    if source_shared_work is None:
        stages = (
            PipelineStageDemand("operand_load", f"memory:{read_path_name}", 1.0,
                include_first_request_latency=True),
            PipelineStageDemand("matrix", "compute:compute_tensor", 1.0),
            PipelineStageDemand("operand_shared", "resource:smem", 1.0,
                predecessors=("operand_load",)),
            PipelineStageDemand("output_store", f"memory:{write_path_name}", 1.0,
                predecessors=("matrix", "operand_shared"),
                completion_resource_roles=("l2_cache",)),
        )
    else:
        if (not isinstance(source_shared_work, tuple) or len(source_shared_work) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v < 0 for v in source_shared_work)
                or source_shared_work[0] <= 0 or source_shared_work[1] <= 0):
            raise ValueError("source shared work requires positive ingress/read and nonnegative output bytes")
        shared = inp.demand_per_resource["smem"]
        if not math.isclose(math.fsum(source_shared_work), shared, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError("source shared phases must conserve aggregate shared work")
        ingress, reads, output = (value/shared for value in source_shared_work)
        # This is a competing write-port capacity branch during async ingress,
        # not a claim that data are written before their global return. The
        # whole operand-load endpoint already closes destination readiness.
        ready = ("operand_load", "operand_shared_ingress_bound")
        stages = (
            PipelineStageDemand("operand_load", f"memory:{read_path_name}", 1.0,
                include_first_request_latency=True, predecessors=()),
            PipelineStageDemand("operand_shared_ingress_bound", "resource:smem", ingress,
                predecessors=()),
            PipelineStageDemand("matrix", "compute:compute_tensor", 1.0, predecessors=ready),
            PipelineStageDemand("operand_shared_read", "resource:smem", reads, predecessors=ready),
        )
        output_ready = ("matrix", "operand_shared_read")
        if output:
            stages += (PipelineStageDemand("output_shared", "resource:smem", output,
                predecessors=output_ready),)
            output_ready = ("output_shared",)
        stages += (PipelineStageDemand("output_store", f"memory:{write_path_name}", 1.0,
            predecessors=output_ready, completion_resource_roles=("l2_cache",)),)
        inp = dataclasses.replace(inp, limitations=inp.limitations + (
            "single_panel_shared_writes_are_ingress_capacity_work",
            "single_panel_shared_output_follows_matrix_and_operand_reads",))
    if any(d.resource_role == "warp_issue" for d in inp.compute_demands):
        # This coarse envelope has no per-CTA admission timeline. Preserve
        # startup work as an independent aggregate capacity branch; do not
        # invent a launch-wide startup-to-ingress barrier.
        stages += (PipelineStageDemand("source_entry_issue", "compute:warp_issue",
            1.0, predecessors=()),)
        inp = dataclasses.replace(inp, limitations=inp.limitations + (
            "single_panel_startup_issue_progress_is_aggregate_only",))
    if output_conversion_warp_instructions:
        if (inp.phase_ledger is not None
                and any(phase.name == "output_conversion" for phase in inp.phase_ledger.phases)):
            raise ValueError("single-panel conversion must be attached exactly once")
        if "warp_issue" in inp.resource_time_overrides_ns:
            raise ValueError("single-panel conversion cannot use issue time overrides")
        issues = tuple(d for d in inp.compute_demands if d.resource_role == "warp_issue")
        if len(issues) > 1:
            raise ValueError("single-panel conversion requires one conserved issue resource")
        old_operations = issues[0].operations if issues else 0.0
        total_operations = old_operations + output_conversion_warp_instructions
        retained_fraction = old_operations / total_operations
        stages = tuple(dataclasses.replace(stage,
            work_fraction=stage.work_fraction * retained_fraction)
            if stage.service == "compute:warp_issue" else stage for stage in stages)
        target_index = next(index for index, stage in enumerate(stages)
                            if stage.name in ("output_shared", "output_store"))
        target = stages[target_index]
        conversion = PipelineStageDemand(
            "output_conversion", "compute:warp_issue",
            output_conversion_warp_instructions / total_operations,
            predecessors=(), prior_resources=("compute:compute_tensor",))
        target = dataclasses.replace(target,
            predecessors=(target.predecessors or ()) + (conversion.name,))
        stages = stages[:target_index] + (conversion, target) + stages[target_index + 1:]
        demand = dict(inp.demand_per_resource)
        demand["warp_issue"] = total_operations
        issue_demand = (dataclasses.replace(issues[0], operations=total_operations)
                        if issues else ComputeDemand("warp_issue", total_operations, 1.0, 1.0))
        inp = dataclasses.replace(inp, demand_per_resource=demand,
            compute_demands=tuple(d for d in inp.compute_demands
                if d.resource_role != "warp_issue") + (issue_demand,),
            limitations=inp.limitations + (
                "matmul_output_packed_conversion_issue_from_compiled_source",
                "matmul_output_conversion_execution_capacity_remains_unpriced",))
    plan = PipelineWorkPlan(1, (PipelineTaskDemand("single_panel_aggregate", 0, stages),))
    result = attach_pipeline_work_plan(inp, hardware, plan)
    return dataclasses.replace(result, limitations=result.limitations + (
        "single_panel_work_conserving_aggregate_operand_matrix_output_envelope",
        "single_panel_partial_cohort_and_native_overlap_remain_analytical",))


def attach_async_copy_ring_capacity(inp, hardware, *, read_path_name, dtype,
                                    tile_m, tile_k, panel_bytes, buffer_slots,
                                    resident_ctas_per_unit, k_tiles):
    """Bind an independently measured byte-ring capacity to its physical class.

    This is a frontend/byte-copy lower bound beside generic hierarchy service,
    not a replacement L2 port rate or a sum of cache-hit and cache-miss times.
    The published primitive spans 64..1024 rounds with a fully populated chip.
    Source owners retain their own stage, residency and operand-byte derivation.
    """
    if (inp.mem_model != "gpu_mshr" or not 64 <= k_tiles <= 1024
            or inp.active_units != hardware.num_compute_units
            or inp.sm_freq_hz != hardware.nominal_sm_freq_hz):
        return inp
    name = (f"async_copy_ring_threads{inp.block_size}_depth{buffer_slots}"
            f"_resident{resident_ctas_per_unit}")
    signature = MemoryAccessClassSignature(
        access_class=name, dtype=dtype, tile_m=tile_m, tile_k=tile_k,
        logical_row_bytes=panel_bytes)
    services = tuple(service for service in hardware.memory_access_class_services
                     if service.service == name and service.access_kind == "read"
                     and service.ingress_resource_role == "l2_cache"
                     and service.signature == signature)
    if not services:
        return inp
    if len(services) != 1:
        raise ValueError("async copy ring capacity requires one exact service")
    paths = tuple(path for path in inp.memory_paths if path.name == read_path_name)
    if len(paths) != 1 or paths[0].access_kind != "read":
        raise ValueError("async copy ring capacity requires one operand read")
    path, = paths
    if path.access_class_demands:
        raise ValueError("async copy ring capacity cannot replace existing typed services")
    if path.nodes[0].resource_role != "l2_cache":
        raise ValueError("async copy ring requires L2 ingress")
    demand = MemoryAccessClassDemand(
        service=name, ingress_resource_role="l2_cache",
        logical_bytes=path.nodes[0].bytes, signature=signature)
    typed_path = dataclasses.replace(path, access_class_demands=(demand,))
    return dataclasses.replace(inp,
        memory_paths=tuple(typed_path if item is path else item for item in inp.memory_paths),
        limitations=inp.limitations + (
            "async_copy_ring_capacity_is_independent_byte_frontend_bound",
            "async_copy_ring_masked_and_cold_bytes_use_aggregate_capacity_approximation",))


def attach_coalesced_async_copy_progress_floor(inp, hardware, *, read_path_name,
        dtype, tile_m, tile_k, panel_bytes, buffer_slots, row_stride_bytes,
        requesting_B_bytes, unique_B_bytes, maximum_fanout,
        resident_ctas_per_unit=1):
    """Compete compulsory same-address request progress with original work.

    DRAM request merging reduces unique traffic but does not make pending
    requesters into completed cache hits. The source supplies first-wave
    duplicate work, treating every later weight replay as free in this floor.
    Independent byte-only fanout capacities retain full copy/control work.
    Multi-resident primitives have distinct capability identities; their
    occupancy cannot borrow the historical one-resident service.
    """
    if (isinstance(resident_ctas_per_unit,bool)
            or not isinstance(resident_ctas_per_unit,int)
            or not 1<=resident_ctas_per_unit<=hardware.max_blocks_per_unit):
        raise ValueError('coalesced progress requires physical resident CTA capacity')
    if (inp.mem_model != "gpu_mshr"
            or inp.sm_freq_hz != hardware.nominal_sm_freq_hz
            or not 1 <= maximum_fanout <= 16):
        return inp
    if not (math.isfinite(requesting_B_bytes) and math.isfinite(unique_B_bytes)
            and 0 < unique_B_bytes <= requesting_B_bytes):
        raise ValueError("coalesced B work requires positive conserved bytes")
    fanout = requesting_B_bytes/unique_B_bytes
    if fanout > maximum_fanout:
        raise ValueError("average request fanout exceeds maximum")
    lower = 2**int(math.floor(math.log2(fanout)))
    upper = lower if fanout == lower else lower*2
    seconds_per_byte = []
    resident_tag='' if resident_ctas_per_unit==1 else f'_resident{resident_ctas_per_unit}'
    for count in (lower,upper):
        name = (f"coalesced_async_copy_threads{inp.block_size}_slots{buffer_slots}"
                f"_b{panel_bytes}{resident_tag}_fanout{count}")
        signature = MemoryAccessClassSignature(access_class=name,dtype=dtype,
            tile_m=tile_m,tile_k=tile_k,logical_row_bytes=row_stride_bytes)
        services = tuple(s for s in hardware.memory_access_class_services
            if s.service == name and s.signature == signature
            and s.access_kind == "read" and s.ingress_resource_role == "l2_cache")
        if not services:
            return inp
        if len(services) != 1:
            raise ValueError("coalesced progress requires one exact capability")
        service, = services
        table = service.logical_byte_capacity_per_s_by_grid_ctas
        if table is None:
            raise ValueError("coalesced progress requires measured grid points")
        points = dict(table)
        if inp.grid_size not in points:
            return inp
        seconds_per_byte.append(1/points[inp.grid_size])
    paths = tuple(p for p in inp.memory_paths if p.name == read_path_name)
    if len(paths) != 1 or paths[0].access_kind != "read":
        raise ValueError("coalesced progress requires one B read path")
    path, = paths
    if path.access_class_demands:
        return inp
    ingress, = (n for n in path.nodes if n.resource_role == "l2_cache")
    if requesting_B_bytes > ingress.bytes*(1+1e-12):
        raise ValueError("compulsory request work exceeds B ingress")
    # Interpolate only independently measured adjacent fanouts. The physical
    # launch grid is exact; failed or absent endpoints are never bridged.
    share = 0 if lower == upper else (fanout-lower)/(upper-lower)
    service_time = requesting_B_bytes*(seconds_per_byte[0]*(1-share)
                                      + seconds_per_byte[1]*share)*1e9
    if inp.execution_time_floor_ns is not None:
        service_time = max(service_time,inp.execution_time_floor_ns)
    return dataclasses.replace(inp,execution_time_floor_ns=service_time,
        limitations=inp.limitations + (
            "compulsory_coalesced_B_request_progress_is_competing_byte_only_floor",
            "coalesced_B_uniform_wave_and_mean_fanout_are_analytical",
            "coalesced_B_later_weight_replays_and_nonrequesting_CTAs_are_free_in_floor",
            'coalesced_B_progress_capability_retains_resident_CTA_capacity',))


def attach_finite_async_copy_progress_floor(inp, hardware, *, read_path_name,
                                           dtype, tile_m, tile_k, panel_bytes,
                                           buffer_slots, row_stride_bytes):
    """Compete an independent full byte-ring body with the existing program.

    Rates retain copy-group/control/synchronization progress. They must not
    replace L2 capacity, inflate global payload, or be added to the matrix
    pipeline. Only exact eligible measured grids bind; holes stay unbound.
    The source owner proves the matching padded tile copy allocation. The
    primitive's broadcast A layout is optimistic for native gather tails;
    inactive CTAs and B hits contribute no cold bytes to this floor. The
    original physical launch and complete A/cache work remain priced.
    """
    if (inp.mem_model != "gpu_mshr"
            or inp.sm_freq_hz != hardware.nominal_sm_freq_hz):
        return inp
    name = (f"finite_async_copy_threads{inp.block_size}_slots{buffer_slots}"
            f"_b{panel_bytes}")
    signature = MemoryAccessClassSignature(
        access_class=name, dtype=dtype, tile_m=tile_m, tile_k=tile_k,
        logical_row_bytes=row_stride_bytes)
    services = tuple(service for service in hardware.memory_access_class_services
                     if service.service == name and service.access_kind == "read"
                     and service.ingress_resource_role == "l2_cache"
                     and service.signature == signature)
    if not services:
        return inp
    if len(services) != 1:
        raise ValueError("finite async copy requires one exact service")
    service, = services
    table = service.logical_byte_capacity_per_s_by_grid_ctas
    if table is None:
        raise ValueError("finite async progress requires measured grid points")
    if inp.grid_size not in tuple(grid for grid, _ in table):
        return inp
    paths = tuple(path for path in inp.memory_paths if path.name == read_path_name)
    if len(paths) != 1 or paths[0].access_kind != "read":
        raise ValueError("finite async progress requires one B read path")
    path, = paths
    nodes = {node.resource_role: node for node in path.nodes}
    if (path.access_class_demands or set(nodes) != {"l2_cache", "dram"}
            or nodes["dram"].bytes <= 0):
        return inp
    if nodes["dram"].bytes > nodes["l2_cache"].bytes:
        raise ValueError("finite async cold bytes exceed B ingress")
    # Resolve the existing typed capability without attaching its finite
    # progress to each K stage (which would charge startup/control twice).
    # A mixed stream retains the complete original cache/DRAM pipeline.
    # This competing cold-substream floor treats B hits as free and scales
    # even the primitive's control work down with its cold byte fraction.
    # It is deliberately optimistic; it neither serializes two cache tiers
    # nor assigns a measured all-cold duration to the complete mixed stream.
    demand = MemoryAccessClassDemand(service=name, ingress_resource_role="l2_cache",
                                    logical_bytes=nodes["dram"].bytes,
                                    signature=signature)
    from src.kernel.memory_path_solver import solve_memory_access_classes
    diagnostic, = solve_memory_access_classes(
        (dataclasses.replace(path, access_class_demands=(demand,)),),
        services, grid_size=inp.grid_size)
    floor = diagnostic.service_time_s * 1e9
    if inp.execution_time_floor_ns is not None:
        floor = max(floor, inp.execution_time_floor_ns)
    return dataclasses.replace(inp, execution_time_floor_ns=floor,
        limitations=inp.limitations + (
            "finite_async_byte_ring_complete_body_is_competing_progress_floor",
            "finite_async_byte_ring_cold_substream_treats_B_hits_as_free",
            "finite_async_byte_ring_broadcast_A_and_native_layout_are_approximations",))


def attach_attention_pipeline_envelope(
        inp: PredictionInput, hardware,
        envelope: AttentionPipelineEnvelopeDemand) -> PredictionInput:
    """Attach a constant-size attention envelope and remove tail scaling."""

    import dataclasses

    if not isinstance(envelope, AttentionPipelineEnvelopeDemand):
        raise TypeError("envelope must be AttentionPipelineEnvelopeDemand")
    if inp.pipeline_plan is not None:
        raise ValueError("pipeline plan must be attached exactly once")
    if inp.mem_model == "npu_tile":
        raise ValueError("attention pipeline envelope is GPU-only")
    old_wave_efficiency = inp.wave_efficiency
    if not 0 < old_wave_efficiency <= 1:
        raise ValueError("invalid legacy wave efficiency")
    compute_demands = tuple(
        dataclasses.replace(
            demand,
            schedule_efficiency=min(
                1.0,
                demand.schedule_efficiency / old_wave_efficiency),
        )
        for demand in inp.compute_demands
    )
    return dataclasses.replace(
        inp,
        compute_demands=compute_demands,
        pipeline_plan=envelope,
        wave_efficiency=1.0,
    )


def pipeline_compute_service(
        inp: PredictionInput, hardware,
        roles: Optional[Tuple[str, ...]] = None) -> str:
    """Return one declared service covering all positive compute demands."""

    positive_roles = tuple(
        demand.resource_role for demand in inp.compute_demands
        if demand.operations > 0)
    if roles is None:
        arithmetic_roles = tuple(
            role for role in positive_roles if role != "warp_issue")
        roles = arithmetic_roles if arithmetic_roles else positive_roles
    elif (not isinstance(roles, tuple)
          or not roles
          or any(role not in positive_roles for role in roles)):
        raise ValueError(
            "requested pipeline compute roles must be a non-empty subset of "
            f"positive demands: requested={roles}, positive={positive_roles}")
    if not roles:
        raise ValueError("pipeline requires positive compute work")
    resource_names = {
        hardware.resource_by_role(role).name for role in roles
    }
    covering_groups = tuple(
        group.name for group in hardware.resource_groups
        if resource_names.issubset(set(group.members)))
    if len(covering_groups) == 1:
        return f"group:{covering_groups[0]}"
    if len(roles) == 1:
        return f"compute:{roles[0]}"
    if not covering_groups:
        raise NotImplementedError(
            "pipeline cannot serialize or overlap compute roles without a "
            f"shared hardware group: {roles}")
    raise ValueError(
        "compute demands are covered by multiple ambiguous resource groups: "
        f"{covering_groups}")


def pipeline_compute_stages(
        inp: PredictionInput, hardware
        ) -> Tuple[Tuple[str, str, bool], ...]:
    """Partition positive compute work into ordered physical group services."""

    positive_roles = tuple(
        demand.resource_role for demand in inp.compute_demands
        if demand.operations > 0 and demand.resource_role != "warp_issue")
    if not positive_roles:
        positive_roles = tuple(
            demand.resource_role for demand in inp.compute_demands
            if demand.operations > 0)
    if not positive_roles:
        raise ValueError("pipeline requires positive compute work")

    services = []
    for role in positive_roles:
        resource_name = hardware.resource_by_role(role).name
        covering_groups = tuple(
            group.name for group in hardware.resource_groups
            if resource_name in group.members)
        if len(covering_groups) == 0:
            service = f"compute:{role}"
        elif len(covering_groups) == 1:
            service = f"group:{covering_groups[0]}"
        else:
            raise ValueError(
                "compute resource belongs to multiple ambiguous groups: "
                f"role={role!r}, groups={covering_groups}")
        if service not in services:
            services.append(service)

    if len(services) == 1:
        return (("compute", services[0], False),)
    return tuple(
        (f"compute_{index}", service, False)
        for index, service in enumerate(services))


def build_forward_demand(l2_total: float, dram_bytes: float,
                         is_non_tensor: bool, flops: Dict[str, float]) -> Dict[str, float]:
    """Build the complete resource demand used by the physical formula."""

    compute_roles = {
        "compute_tensor", "compute_vector", "compute_reduction",
        "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
        "scalar_aiv",
    }
    unknown_roles = set(flops) - compute_roles
    if unknown_roles:
        raise ValueError(f"unknown compute demand roles: {sorted(unknown_roles)}")
    demand = {
        "l1_cache": l2_total if is_non_tensor else 0.0,
        "smem": 0.0 if is_non_tensor else l2_total * 2.0,
        "xbar": l2_total,
        "l2_cache": l2_total,
        "dram": dram_bytes,
        "mte3": 0.0,
        **{role: 0.0 for role in compute_roles},
    }
    demand.update({name: float(value) for name, value in flops.items()})
    return demand


def validated_l2_miss_rate(dram_bytes: float, l2_bytes: float) -> float:
    """Return the physical DRAM/L2 ratio without hiding inconsistent traffic."""

    if dram_bytes < 0 or l2_bytes < 0:
        raise ValueError("DRAM and L2 traffic cannot be negative")
    if l2_bytes == 0:
        if dram_bytes != 0:
            raise ValueError("positive DRAM traffic requires positive L2 traffic")
        return 0.0
    rate = dram_bytes / l2_bytes
    if not 0.0 <= rate <= 1.0:
        raise ValueError(
            f"DRAM/L2 traffic ratio must be within [0, 1], got {rate}")
    return rate


def causal_attention_pairs(
        q_lens: Tuple[int, ...], kv_lens: Tuple[int, ...], *,
        capacity: Optional[int] = None) -> int:
    """Count source-visible causal query/key pairs without shape padding."""

    if len(q_lens) != len(kv_lens) or not q_lens:
        raise ValueError("causal attention lengths must be non-empty and align")
    if capacity is not None and (
            isinstance(capacity, bool) or not isinstance(capacity, int)
            or capacity <= 0):
        raise ValueError("causal attention capacity must be positive")
    total = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        if (isinstance(q_len, bool) or not isinstance(q_len, int)
                or isinstance(kv_len, bool) or not isinstance(kv_len, int)
                or q_len <= 0 or kv_len < q_len):
            raise ValueError(
                "causal attention requires positive integer lengths and "
                "kv_len >= q_len")
        first_visible = kv_len - q_len + 1
        last_visible = kv_len
        if capacity is None or capacity >= last_visible:
            total += q_len * (first_visible + last_visible) // 2
            continue
        uncapped_terms = max(0, min(q_len, capacity - first_visible + 1))
        if uncapped_terms:
            total += uncapped_terms * (
                2 * first_visible + uncapped_terms - 1) // 2
        total += (q_len - uncapped_terms) * capacity
    if total <= 0:
        raise ValueError("causal attention pair count must be positive")
    return total


def launch_wave_geometry(kernel: KernelSpec, hardware, *,
                         execution_units: Optional[int] = None,
                         active_workgroups: Optional[int] = None,
                         ) -> Tuple[float, float, int]:
    """Return fixed resource occupancy, tail-wave efficiency and active units.

    ``smem``, registers, CTA residency and resident warps are deliberately
    absent.  A finite grid still exposes a physical tail because workgroups
    beyond the active units require a later wave. Source-proven early-return
    workgroups can be excluded from useful-work capacity without changing
    the physical launch or its dispatch boundary.
    """

    launch = kernel.launch
    workgroups = launch.grid_size
    if active_workgroups is not None:
        if isinstance(active_workgroups, bool) or not isinstance(active_workgroups, int):
            raise TypeError("active_workgroups must be an integer")
        if not 0 < active_workgroups <= launch.grid_size:
            raise ValueError("active_workgroups must be within the physical launch")
        workgroups = active_workgroups
    if execution_units is None:
        unit_count = hardware.num_compute_units
    else:
        if (isinstance(execution_units, bool)
                or not isinstance(execution_units, int)):
            raise TypeError("execution_units must be an integer")
        if execution_units <= 0:
            raise ValueError("execution_units must be positive")
        unit_count = execution_units
    wave_size = min(workgroups, unit_count)
    if wave_size <= 0:
        raise ValueError("launch produced a non-positive wave size")
    waves = workgroups / wave_size
    wave_efficiency = waves / math.ceil(waves)
    active_units = min(workgroups, unit_count)
    if hardware.kernel_memory_model not in (
            "gpu_mshr", "npu_tile", "pim_bank", "systolic_scratchpad"):
        raise ValueError(
            "unknown kernel_memory_model for launch resources: "
            f"{hardware.kernel_memory_model!r}")
    return (
        FULL_RESOURCE_OCCUPANCY,
        wave_efficiency,
        active_units,
    )


def sum_serial_kernel_predictions(stages) -> KernelPrediction:
    parts = tuple(stages)
    if not parts:
        raise ValueError("serial prediction requires at least one stage")
    names = tuple(name for name, _ in parts)
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("serial prediction stage names must be non-empty and unique")
    if any(prediction.diagnostics is None for _, prediction in parts):
        raise ValueError("serial prediction stages require structured diagnostics")
    if any(prediction.diagnostics.pipeline is not None for _, prediction in parts):
        raise ValueError("serial composition of nested pipeline diagnostics is unsupported")
    resources = {}
    compute = {}
    arbitration = {}
    for _, prediction in parts:
        for role, seconds in prediction.t_per_resource.items():
            resources[role] = (resources[role] if role in resources else 0.0) + seconds
        for role, seconds in prediction.diagnostics.compute_time_s.items():
            compute[role] = (compute[role] if role in compute else 0.0) + seconds
        for item in prediction.diagnostics.memory_arbitration:
            role = item.resource_role
            if role not in arbitration:
                arbitration[role] = []
            arbitration[role].append(item)
    combined_arbitration = []
    for role, items in arbitration.items():
        first = items[0]
        if any(item.arbitration_mode != first.arbitration_mode
                or item.read_capacity_per_s != first.read_capacity_per_s
                or item.write_capacity_per_s != first.write_capacity_per_s
                for item in items[1:]):
            raise ValueError(f"serial memory arbitration capacities differ for {role}")
        combined_arbitration.append(dataclasses.replace(first,
            read_bytes=math.fsum(item.read_bytes for item in items),
            write_bytes=math.fsum(item.write_bytes for item in items),
            read_service_time_s=math.fsum(item.read_service_time_s for item in items),
            write_service_time_s=math.fsum(item.write_service_time_s for item in items),
            analytical_time_s=math.fsum(item.analytical_time_s for item in items),
            effective_time_s=math.fsum(item.effective_time_s for item in items),
            overridden=any(item.overridden for item in items)))
    support_levels = {prediction.diagnostics.support_level for _, prediction in parts}
    support_level = next(iter(support_levels)) if len(support_levels) == 1 else "partial"
    limitations = tuple(dict.fromkeys(
        limitation for _, prediction in parts
        for limitation in prediction.diagnostics.limitations))
    diagnostics = PredictionDiagnostics(
        compute_time_s=compute,
        memory=tuple(dataclasses.replace(item, path_name=f"{name}:{item.path_name}")
            for name, prediction in parts for item in prediction.diagnostics.memory),
        memory_access_classes=tuple(dataclasses.replace(item, path_name=f"{name}:{item.path_name}")
            for name, prediction in parts for item in prediction.diagnostics.memory_access_classes),
        memory_arbitration=tuple(combined_arbitration),
        pipeline=None,
        critical_path=tuple(f"{name}:{component}" for name, prediction in parts
            for component in prediction.diagnostics.critical_path),
        support_level=support_level,
        limitations=limitations + ("serial_stage_support_and_critical_path_combined",),
    )
    groups = {name: prediction.t_total_s for name, prediction in parts}
    return KernelPrediction(
        t_total_s=sum(groups.values()),
        t_per_group=groups,
        t_launch_s=sum(prediction.t_launch_s for _, prediction in parts),
        bottleneck_group=max(groups, key=groups.__getitem__),
        t_per_resource=resources,
        diagnostics=diagnostics,
    )


def build_aggregate_gpu_input(
        kernel: KernelSpec, hardware, demand: Dict[str, float], *,
        write_l2_bytes: float, write_dram_bytes: float,
        limitations: Tuple[str, ...]) -> PredictionInput:
    """Build a source-work aggregate for an uncalibrated GPU kernel.

    This boundary is intentionally limited to static semantic work and tensor
    traffic.  It does not infer residency, pipeline overlap, or a workload
    timing correction when the retained source does not expose those facts.
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise NotImplementedError(
            f"{kernel.kind.value}/{kernel.variant} currently has only a "
            "GPU analytical execution contract")
    if not isinstance(limitations, tuple) or any(
            not isinstance(value, str) or not value for value in limitations):
        raise TypeError("aggregate GPU limitations must be non-empty strings")
    l2_total = float(demand["l2_cache"])
    dram_total = float(demand["dram"])
    for name, value, total in (
            ("write_l2_bytes", write_l2_bytes, l2_total),
            ("write_dram_bytes", write_dram_bytes, dram_total)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0 or value > total):
            raise ValueError(
                f"{name} must be finite and within its total traffic")
    eta, wave_efficiency, active_units = launch_wave_geometry(
        kernel, hardware)
    inp = PredictionInput(
        demand_per_resource=dict(demand),
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=kernel.variant,
        mem_model=hardware.kernel_memory_model,
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind="generic",
        dram_capacity_multiplier=hardware.dram_protocol_efficiency_l1cached,
        include_dispatch_overhead=True,
        support_level="aggregate_only",
        limitations=gpu_partial_limitations(hardware, *limitations),
    )
    producers, requests_per_producer = gpu_memory_source(hardware)
    structured = attach_structured_contract(
        inp,
        hardware,
        transaction_bytes=None,
        path_name="global_memory",
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=None,
    )
    return split_global_memory_paths(
        structured,
        read_l2_bytes=l2_total - float(write_l2_bytes),
        read_dram_bytes=dram_total - float(write_dram_bytes),
        write_l2_bytes=float(write_l2_bytes),
        write_dram_bytes=float(write_dram_bytes),
    )


def roofline_ns(demand: Dict[str, float], hardware, compulsory_bytes: float) -> float:
    compute_seconds = 0.0
    for role in (
            "compute_tensor", "compute_vector", "compute_reduction",
            "compute_fma", "compute_sfu", "compute_alu"):
        amount = float(demand[role])
        if amount <= 0:
            continue
        if role not in hardware._role_map:
            raise ValueError(
                f"hardware has no {role!r} resource for positive demand {amount}")
        capacity = hardware.resource_by_role(role).capacity
        if capacity <= 0:
            raise ValueError(f"hardware {role!r} capacity must be positive")
        compute_seconds = max(compute_seconds, amount / capacity)
    dram_capacity = hardware.resource_by_role("dram").capacity
    if compulsory_bytes < 0 or dram_capacity <= 0:
        raise ValueError(
            "roofline bytes cannot be negative and DRAM capacity must be positive")
    memory_seconds = compulsory_bytes / dram_capacity
    return max(compute_seconds, memory_seconds) * 1e9


def source_stage_roofline_ns(demand, hardware, l2_bytes, dram_bytes):
    """Price explicit source-role L2 and compulsory DRAM traffic with compute."""
    if l2_bytes < 0 or dram_bytes < 0 or dram_bytes > l2_bytes:
        raise ValueError("source stage traffic requires 0 <= DRAM bytes <= L2 bytes")
    compute_ns = roofline_ns(demand, hardware, 0.0)
    l2_capacity = hardware.resource_by_role("l2_cache").capacity
    dram_capacity = hardware.resource_by_role("dram").capacity
    if l2_capacity <= 0 or dram_capacity <= 0:
        raise ValueError("source stage L2/DRAM capacities must be positive")
    return max(compute_ns, l2_bytes / l2_capacity * 1e9,
        dram_bytes / dram_capacity * 1e9)


def _with_graph_regime(inp, kernel):
    """Tag the input with whether this launch is a replayed graph node."""

    replayed = graph_replayed_launch(kernel)
    if replayed == inp.graph_replayed:
        return inp
    return dataclasses.replace(inp, graph_replayed=replayed)
