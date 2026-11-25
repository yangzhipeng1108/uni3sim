"""Canonical structured analytical formula for one kernel prediction."""

import dataclasses
import math

from src.kernel.kernel_spec import KernelPrediction
from src.kernel.memory_path_solver import (
    memory_hierarchy_for_path,
    solve_memory_access_classes,
    solve_memory_arbitration,
    solve_memory_path,
)
from src.kernel.model_contract import (
    PredictionInput,
    AIV_COMPLETE_TASK_FLOOR_ONLY,
    AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    AIV_SOURCE_PIPELINE_LAUNCH_ONLY,
    AIV_SOURCE_ENTRY_FILL,
    AIV_WORKER_START_FILL,
    GPU_SOURCE_PIPELINE_OUTER_ENVELOPE,
    AttentionPipelineClass,
    AttentionPipelineEnvelope,
    AttentionPipelineEnvelopeDemand,
    AttentionPipelinePhase,
    BarrierPipelineEnvelope,
    BarrierPipelineEnvelopeDemand,
    CannGroupedMatmulEnvelopeDemand,
    CannSingleCoreSplitKEnvelopeDemand,
    FiaCrossDomainWorkPlan,
    FiaSplitFuseWorkPlan,
    GpuGroupedGemmPipelineEnvelopeDemand,
    MetaxGroupedOperandPipelineDemand,
    MEMORY_DEPENDENCY_TASK_INCREMENT_SERVICE,
    GLOBAL_STORE_COMPLETION_SERVICE,
    PipelineDiagnostics,
    PipelinePlan,
    PipelineResource,
    PipelineStage,
    PipelineTask,
    PipelineTaskProgram,
    PipelineTaskProgramDemand,
    PipelineTaskDiagnostic,
    PipelineTaskSequence,
    PipelineTaskSequenceDemand,
    PipelineWorkPlan,
    PredictionDiagnostics,
    WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
)
from src.kernel.pipeline_model import (
    _evaluate_barrier_pipeline_sequence,
    _maxplus_apply,
    _maxplus_identity,
    _maxplus_power_matrix,
    _maxplus_powers,
    apply_maxplus_transition_programs,
    apply_repeated_maxplus_transition,
    compose_maxplus_transitions,
    evaluate_attention_pipeline_envelope,
    evaluate_barrier_pipeline_envelope,
    evaluate_pipeline,
)


def _split_fuse_mm2_source_order_time(
        stack_widths, packed_rows, head_dim,
        mte2_descriptor_duration, tensor_panel_duration,
        output_duration):
    """Solve the non-paged SplitFuse PV program with two alternating L1 slots."""

    if (not isinstance(stack_widths, tuple) or not stack_widths
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or value <= 0 for value in stack_widths)):
        raise ValueError("SplitFuse MM2 stack widths must be positive integers")
    for name, value in (
            ("packed_rows", packed_rows), ("head_dim", head_dim)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"SplitFuse MM2 {name} must be a positive integer")
    if (not isinstance(output_duration, (int, float))
            or isinstance(output_duration, bool)
            or not math.isfinite(output_duration)
            or output_duration < 0.0):
        raise ValueError(
            "SplitFuse MM2 output duration must be finite and non-negative")

    mte2_available = 0.0
    tensor_available = 0.0
    output_available = 0.0
    l1_buffer_ready = [0.0, 0.0]
    l1_chunk_ordinal = 0
    mte2_descriptor_ordinal = 0

    def issue_mte2(byte_count):
        nonlocal mte2_available, mte2_descriptor_ordinal
        duration = mte2_descriptor_duration(
            byte_count, mte2_descriptor_ordinal)
        if (not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(duration) or duration < 0.0):
            raise ValueError(
                "SplitFuse MM2 descriptor duration must be finite and "
                "non-negative")
        mte2_available += duration
        mte2_descriptor_ordinal += 1
        return mte2_available

    for columns in stack_widths:
        remaining = columns
        while remaining:
            l1_columns = min(256, remaining)
            l1_slot = l1_chunk_ordinal % len(l1_buffer_ready)
            mte2_available = max(
                mte2_available, l1_buffer_ready[l1_slot])
            p_ready = issue_mte2(2 * packed_rows * l1_columns)
            panel_count, tail_columns = divmod(l1_columns, 128)
            panel_widths = (
                (128,) * panel_count
                + (() if tail_columns == 0 else (tail_columns,)))
            for panel_columns in panel_widths:
                v_ready = issue_mte2(2 * panel_columns * head_dim)
                duration = tensor_panel_duration(panel_columns)
                if (not isinstance(duration, (int, float))
                        or isinstance(duration, bool)
                        or not math.isfinite(duration) or duration < 0.0):
                    raise ValueError(
                        "SplitFuse MM2 Tensor duration must be finite and "
                        "non-negative")
                tensor_start = max(tensor_available, p_ready, v_ready)
                tensor_available = tensor_start + duration
            l1_buffer_ready[l1_slot] = tensor_available
            l1_chunk_ordinal += 1
            remaining -= l1_columns
        output_start = max(output_available, tensor_available)
        output_available = output_start + output_duration
    return max(mte2_available, tensor_available, output_available)


def _resolve_small_kernel_span_floor(hardware, inp) -> float:
    """Return the measured lower bound of a memory-touching launch's span.

    The empty-task scheduler describes a launch that touches nothing.  A
    launch that performs any global access does not reach that duration: the
    measured floor for one is 2080 ns at grid 48, where an empty CTA reaches
    1760.  It is a floor, so it composes by maximum and never adds.

    The probe measured grids 48-216 only; the motivating case (a GEMM
    split-K reduce kernel) sits exactly at grid 48, inside that domain.
    Below it the surface is not measured, and the only population that
    actually depends on extrapolating there is the MoE per-expert-row
    routing bookkeeping (moe_align, moe_align_small_batch): both launch a
    single administrative CTA (grid=1, independent of token count or QPS)
    that scans the routed-token/expert-offset bookkeeping once per MoE
    layer per step.  A permanently grid-1 launch is categorically outside a
    probe whose measured population starts at grid 48 -- it is never, at
    any batch size, one of the launches the probe describes.  Keep the
    measured floor at grid >= 48 for every variant and do not extrapolate
    it below 48 for this always-grid-1 bookkeeping family; every other
    memory-touching launch (including the token-count-scaled MoE
    sort/reduce/activation kernels, whose grid does grow with batch size)
    keeps the same floor as before this exclusion existed.
    """

    if not (inp.memory_paths and inp.include_dispatch_overhead):
        return 0.0
    return small_kernel_span_floor_s(
        hardware, inp.grid_size, inp.variant, inp.graph_replayed)


def small_kernel_span_floor_s(hardware, grid_size, variant,
                              graph_replayed) -> float:
    """Return the published span floor of one memory-touching launch.

    Zero when the card publishes no floor.  See
    ``_resolve_small_kernel_span_floor`` for the measured domain and the
    grid-1 routing-bookkeeping exclusion.
    """

    span_floor_profile = hardware.small_kernel_span_floor
    if span_floor_profile is None:
        return 0.0
    # A graph-replayed node and a standalone launch do not span the same
    # time, and the floor above was measured on standalone launches.  Where
    # the graph span is published, use it for graph-replayed launches.
    graph_span_profile = hardware.cuda_graph_small_node_span
    if graph_replayed and graph_span_profile is not None:
        table = graph_span_profile["span_ns_by_grid_ctas"]
    else:
        table = span_floor_profile["floor_ns_by_grid_ctas"]
    minimum_measured_grid_ctas = table[0][0]
    if (grid_size < minimum_measured_grid_ctas
            and variant in ("moe_align", "moe_align_small_batch")):
        return 0.0
    resolved = table[0][1]
    for grid, floor_ns in table:
        if grid <= grid_size:
            resolved = floor_ns
    return resolved * 1e-9


class PerformanceFormula:
    """Compose structured demand, capacity, memory, pipeline, and launch."""

    def __init__(self, hw_spec):
        self.hw = hw_spec

    def _compute_capacity(self, inp, resource):
        """Price compute on the active fraction of its execution domain."""

        if resource.type != "compute":
            raise ValueError(
                f"resource {resource.name!r} is not a compute resource")
        capacity = resource.effective_capacity(
            inp.active_units,
            sm_freq_hz=inp.sm_freq_hz,
            dram_freq_hz=inp.dram_freq_hz)
        if resource.scope == "shared":
            domain_units = self.hw.execution_units_for_task(
                inp.launch_task_kind)
            capacity *= min(inp.active_units, domain_units) / domain_units
        elif resource.scope == "per_unit":
            pass
        else:
            raise ValueError(
                f"unknown compute resource scope: {resource.scope!r}")
        return capacity

    def _compatibility_memory_time(self, inp, resource, demand):
        """Retained cacheless path outside the three-hardware acceptance set."""

        memory_model = inp.mem_model
        if memory_model == "pim_bank":
            if resource.role != "l2_cache":
                raise ValueError(
                    "PIM compatibility memory must terminate at l2_cache")
            memory = self.hw.memory_subsystem
            columns = int(memory["pim_columns_per_row"])
            column_ns = float(memory["pim_column_access_ns"])
            row_activate_s = float(memory["pim_row_activate_ns"]) * 1e-9
            row_bytes = float(memory["pim_row_bytes"])
            latency_s = row_activate_s + columns * column_ns * 1e-9
            capacity = self.hw.num_compute_units * row_bytes / latency_s
        elif memory_model == "systolic_scratchpad":
            if resource.role != "l2_cache":
                raise ValueError(
                    "systolic compatibility memory must terminate at l2_cache")
            dma_bpc = float(
                self.hw.memory_subsystem["systolic_dma_bus_bpc"])
            if dma_bpc <= 0:
                raise ValueError("systolic_dma_bus_bpc must be positive")
            dma_capacity = dma_bpc * inp.sm_freq_hz
            dram_capacity = self.hw.resource_by_role(
                "dram").effective_capacity(
                    inp.active_units,
                    sm_freq_hz=inp.sm_freq_hz,
                    dram_freq_hz=inp.dram_freq_hz)
            capacity = min(dma_capacity, dram_capacity)
        else:
            raise ValueError(
                f"unknown compatibility memory model: {memory_model!r}")
        if capacity <= 0:
            raise ValueError("compatibility memory capacity must be positive")
        return demand / capacity

    @staticmethod
    def _burst_memory_stage_service(memory, work_fraction, worker_count):
        """Return (all-return completion, byte-port occupancy) of one burst.

        A cache hit may return first while a miss is still outstanding. The
        consumer joins the node endpoints, rather than treating that first
        packet's average latency as the endpoint of the entire burst. Node
        traffic uses the caller's declared per-stage path fractions. A node
        fraction below one transaction retains a fluid expected-cardinality
        approximation; it is not an observed per-panel cache hit or miss.
        """
        prefix = completion = occupancy = 0.0
        for node in memory.nodes:
            if node.transactions is None:
                raise ValueError("burst completion requires node transaction geometry")
            transactions = node.transactions * work_fraction
            rate = node.service_capacity_per_s / worker_count
            occupancy = max(occupancy, transactions / rate)
            prefix += min(1.0, transactions) * node.unloaded_latency_s
            completion = max(completion,
                prefix + max(transactions - 1.0, 0.0) / rate)
        return max(completion, occupancy), occupancy

    @staticmethod
    def _local_memory_source_completion_time(
            stage, memory, worker_count):
        """Resolve one worker's finite-source memory completion chain.

        Aggregate memory diagnostics describe the whole launch.  This helper
        consumes only path geometry and per-unit source facts from that
        diagnostic: stage bytes recover their own transaction ratios, so a
        launch-wide cache-sharing change cannot shorten an unrelated CTA's
        local request lifecycle.
        """

        if memory.rate_unit != "transactions":
            raise ValueError(
                "local memory source completion requires a "
                "transaction-resolved memory path")
        if memory.initial_transactions is None:
            raise ValueError(
                "local memory source completion requires aggregate initial "
                "transactions")
        if memory.source_slots_per_unit is None:
            raise ValueError(
                "local memory source completion requires a per-unit finite "
                "request-source population")
        if memory.queues:
            raise ValueError(
                "local memory source completion does not support an "
                "unresolved shared queue lifecycle")
        if worker_count <= 0:
            raise ValueError(
                "local memory source completion requires positive workers")

        completion_roles = (
            set(stage.completion_resource_roles)
            if stage.completion_resource_roles is not None else None)
        demand_by_role = {}
        for demand in (
                stage.memory_node_demands
                + stage.launch_shared_memory_node_demands):
            if (completion_roles is not None
                    and demand.resource_role not in completion_roles):
                continue
            if demand.resource_role in demand_by_role:
                raise ValueError(
                    "local memory source completion cannot repeat a path "
                    f"node: {demand.resource_role!r}")
            demand_by_role[demand.resource_role] = demand

        node_by_role = {node.resource_role: node for node in memory.nodes}
        ingress_role = memory.nodes[0].resource_role
        if ingress_role not in demand_by_role:
            raise ValueError(
                "local memory source completion requires bytes on the path "
                f"ingress node {ingress_role!r}")
        stage_transactions_by_role = {}
        for role, demand in demand_by_role.items():
            if role not in node_by_role:
                raise ValueError(
                    "local memory source completion references an unknown "
                    f"path node: {role!r}")
            node = node_by_role[role]
            if (node.transactions is None
                    or node.transactions <= 0.0
                    or node.bytes <= 0.0):
                raise ValueError(
                    "local memory source completion requires transaction "
                    f"geometry for path node {role!r}")
            transaction_bytes = node.bytes / node.transactions
            stage_transactions_by_role[role] = (
                demand.bytes / transaction_bytes)

        initial_transactions = stage_transactions_by_role[ingress_role]
        if initial_transactions <= 0.0:
            raise ValueError(
                "local memory source completion requires positive ingress "
                "transactions")
        source_lifetime_s = math.fsum(
            (stage_transactions_by_role[role] / initial_transactions)
            * node_by_role[role].unloaded_latency_s
            for role in stage_transactions_by_role)
        if not math.isfinite(source_lifetime_s) or source_lifetime_s <= 0.0:
            raise ValueError(
                "local memory source completion requires positive unloaded "
                "node latency evidence")

        local_rate_candidates = (
            memory.source_slots_per_unit / source_lifetime_s,)
        if memory.frontend_rate_capacity_per_s is not None:
            local_rate_candidates += (
                memory.frontend_rate_capacity_per_s / worker_count,)
        local_rate_per_s = min(local_rate_candidates)
        if not math.isfinite(local_rate_per_s) or local_rate_per_s <= 0.0:
            raise ValueError(
                "local memory source completion produced a non-positive "
                "request rate")
        if stage.local_memory_source_completion == "sequence_start":
            return math.fsum((
                source_lifetime_s,
                max(initial_transactions - 1.0, 0.0) / local_rate_per_s,
            ))
        if stage.local_memory_source_completion == "steady":
            return initial_transactions / local_rate_per_s
        raise ValueError(
            "unknown local memory source-completion mode: "
            f"{stage.local_memory_source_completion!r}")

    def _resolve_pipeline(
            self, work_plan, compute_time_by_role, memory_diagnostics,
            memory_service_time, memory_access_class_diagnostics,
            overridden_memory_services,
            t_per_resource, compute_demands, active_units,
            launch_task_kind, *, allow_partial_worker_services=False):
        if isinstance(work_plan, AttentionPipelineEnvelopeDemand):
            return self._resolve_attention_pipeline_envelope(
                work_plan,
                compute_time_by_role,
                memory_diagnostics,
                memory_service_time,
                memory_access_class_diagnostics,
                overridden_memory_services,
                t_per_resource,
            )
        if isinstance(work_plan, BarrierPipelineEnvelopeDemand):
            return self._resolve_barrier_pipeline_envelope(
                work_plan,
                compute_time_by_role,
                memory_diagnostics,
                memory_service_time,
                memory_access_class_diagnostics,
                overridden_memory_services,
                t_per_resource,
                compute_demands,
                active_units,
                launch_task_kind,
            )
        if isinstance(work_plan, CannGroupedMatmulEnvelopeDemand):
            return self._resolve_cann_grouped_matmul_envelope(
                work_plan,
                memory_diagnostics,
                memory_access_class_diagnostics,
                compute_demands,
                active_units,
                launch_task_kind,
            )
        if isinstance(work_plan, CannSingleCoreSplitKEnvelopeDemand):
            return self._resolve_cann_single_core_split_k_envelope(
                work_plan,
                memory_diagnostics,
                compute_time_by_role,
                active_units,
                launch_task_kind,
            )
        if isinstance(work_plan, GpuGroupedGemmPipelineEnvelopeDemand):
            return self._resolve_gpu_grouped_gemm_pipeline_envelope(
                work_plan, active_units, launch_task_kind,
                launched_ctas=work_plan.active_ctas)
        if isinstance(
                work_plan, MetaxGroupedOperandPipelineDemand):
            raise TypeError(
                "MetaxGroupedOperandPipelineDemand resolves only as "
                "a top-level pipeline plan with its launched grid")
        service_time = {
            f"compute:{role}": duration
            for role, duration in compute_time_by_role.items()
        }
        memory_by_service = {}
        for diagnostic in memory_diagnostics:
            key = f"memory:{diagnostic.path_name}"
            if key not in memory_service_time:
                raise ValueError(
                    f"pipeline memory service time is missing {key!r}")
            service_time[key] = memory_service_time[key]
            memory_by_service[key] = diagnostic
        extra_memory_services = (
            set(memory_service_time) - set(memory_by_service))
        if extra_memory_services:
            raise ValueError(
                "pipeline memory service times contain unknown paths: "
                f"{sorted(extra_memory_services)}")
        memory_access_by_identity = {}
        memory_access_time_by_service = {}
        for diagnostic in memory_access_class_diagnostics:
            identity = (
                diagnostic.path_name,
                diagnostic.service,
                diagnostic.ingress_resource_role,
                diagnostic.signature,
            )
            if identity in memory_access_by_identity:
                raise ValueError(
                    "pipeline memory access-class diagnostics repeat an exact "
                    f"identity: {identity!r}")
            memory_access_by_identity[identity] = diagnostic
            service_name = f"memory_access:{diagnostic.service}"
            if service_name in memory_access_time_by_service:
                memory_access_time_by_service[service_name] = math.fsum((
                    memory_access_time_by_service[service_name],
                    diagnostic.service_time_s,
                ))
            else:
                memory_access_time_by_service[service_name] = (
                    diagnostic.service_time_s)
        service_time.update(memory_access_time_by_service)
        if not isinstance(overridden_memory_services, set):
            raise TypeError(
                "overridden_memory_services must be a set")
        unknown_overridden_services = (
            overridden_memory_services - set(memory_by_service))
        if unknown_overridden_services:
            raise ValueError(
                "overridden pipeline memory services are unknown: "
                f"{sorted(unknown_overridden_services)}")
        for role, resource in self.hw._role_map.items():
            service_time[f"resource:{role}"] = t_per_resource[resource.name]
        for group in self.hw.resource_groups:
            service_time[f"group:{group.name}"] = sum(
                t_per_resource[member] for member in group.members)

        logical_work_tasks = tuple(
            logical
            for entry in work_plan.tasks
            for sequence in (
                entry.patterns
                if isinstance(entry, PipelineTaskProgramDemand)
                else (entry,))
            for logical in (
                sequence.tasks
                if isinstance(sequence, PipelineTaskSequenceDemand)
                else (sequence,)))
        if any(
                stage.absolute_duration_s is not None
                for task in logical_work_tasks
                for stage in task.stages):
            raise ValueError(
                "absolute-duration pipeline stages are supported only by "
                "attention pipeline envelopes")
        initial_resource_names = {
            name for name, _duration_s
            in work_plan.initial_resource_availability_s}
        aiv_grouped_stages = tuple(
            stage
            for task in logical_work_tasks
            for stage in task.stages
            if stage.aiv_ingress_group_size is not None)
        if aiv_grouped_stages:
            if any(
                    isinstance(
                        entry,
                        (PipelineTaskProgramDemand, PipelineTaskSequenceDemand))
                    or entry.repeat_count != 1
                    for entry in work_plan.tasks):
                raise ValueError(
                    "AIV ingress-group scheduling requires explicit "
                    "one-pass worker tasks")
            group_size = aiv_grouped_stages[0].aiv_ingress_group_size
            if any(stage.aiv_ingress_group_size != group_size
                   for stage in aiv_grouped_stages):
                raise ValueError(
                    "one pipeline work plan cannot mix AIV ingress group "
                    "sizes")
            aiv_group_count = self.hw.num_compute_units
            mte2_units = self.hw.resource_by_role("mte2").total_units
            if aiv_group_count * group_size != mte2_units:
                raise ValueError(
                    "AIV ingress-group topology disagrees with MTE2 units: "
                    f"groups={aiv_group_count}, group_size={group_size}, "
                    f"mte2_units={mte2_units}")
        else:
            group_size = None
            aiv_group_count = None
        used_services = {
            stage.service
            for task in logical_work_tasks
            for stage in task.stages
        }
        used_services.update(
            service
            for task in logical_work_tasks
            for stage in task.stages
            for service in stage.prior_resources)
        unknown_services = (
            used_services - set(service_time) - initial_resource_names)
        if unknown_services:
            raise ValueError(
                "pipeline references unknown services: "
                f"{sorted(unknown_services)}")
        fractional_services = {
            stage.service
            for task in logical_work_tasks
            for stage in task.stages
            if not (
                stage.compute_service_demands
                or stage.compute_exact_request_demands
                or stage.memory_node_demands
                or stage.launch_shared_memory_node_demands
                or stage.memory_access_class_demands)
        }
        typed_compute_services = {
            stage.service
            for task in logical_work_tasks
            for stage in task.stages
            if (stage.compute_service_demands
                or stage.compute_exact_request_demands)
        }
        absolute_memory_node_services = {
            stage.service
            for task in logical_work_tasks
            for stage in task.stages
            if (stage.memory_node_demands
                or stage.launch_shared_memory_node_demands)
        }
        local_memory_source_completion_services = {
            stage.service
            for task in logical_work_tasks
            for stage in task.stages
            if stage.local_memory_source_completion is not None
        }
        absolute_memory_access_services = {
            stage.service
            for task in logical_work_tasks
            for stage in task.stages
            if stage.memory_access_class_demands
        }
        absolute_services = (
            typed_compute_services
            | absolute_memory_node_services
            | absolute_memory_access_services)
        mixed_services = fractional_services & absolute_services
        if mixed_services:
            raise ValueError(
                "pipeline services cannot mix fractional and absolute "
                f"stages: {sorted(mixed_services)}")
        if (typed_compute_services
                and work_plan.worker_count != active_units):
            raise ValueError(
                "typed worker pipeline count must equal active launch units")
        program_count_cache = {}

        def _sequence_counts(entry):
            if isinstance(entry, PipelineTaskProgramDemand):
                count_key = (
                    len(entry.patterns), entry.order,
                    entry.repeat_count, entry.tail_order)
                if count_key not in program_count_cache:
                    pattern_counts = [0] * len(entry.patterns)
                    for pattern_index in entry.order:
                        pattern_counts[pattern_index] += entry.repeat_count
                    for pattern_index in entry.tail_order:
                        pattern_counts[pattern_index] += 1
                    program_count_cache[count_key] = tuple(pattern_counts)
                return tuple(zip(
                    entry.patterns, program_count_cache[count_key]))
            if isinstance(entry, PipelineTaskSequenceDemand):
                return ((entry, 1),)
            return ()

        def _task_occurrences(entry):
            sequence_counts = _sequence_counts(entry)
            if not sequence_counts:
                return ((entry, entry.repeat_count),)
            worker_multiplicity = (
                entry.worker_multiplicity
                if isinstance(entry, PipelineTaskProgramDemand) else 1)
            return tuple(
                (task,
                 task.repeat_count * sequence.repeat_count * count
                 * worker_multiplicity)
                for sequence, count in sequence_counts
                if count > 0
                for task in sequence.tasks)

        # Per-worker absolute memory may follow an asymmetric repeated source
        # program; its bytes are conserved below with the exact occurrence
        # multiplicity.  Launch-shared bytes are different: every worker must
        # issue one identical source stage exactly once, while the path sees
        # that compulsory footprint only once for the whole launch.
        launch_shared_memory_occurrences = {}
        for entry in work_plan.tasks:
            physical_workers = (
                entry.physical_workers
                if isinstance(entry, PipelineTaskProgramDemand)
                else ())
            for task, multiplier in _task_occurrences(entry):
                for stage in task.stages:
                    if not stage.launch_shared_memory_node_demands:
                        continue
                    key = (stage.name, stage.service)
                    if physical_workers:
                        if multiplier % len(physical_workers):
                            raise ValueError(
                                "compressed launch-shared stage has a "
                                "non-integral physical-worker occurrence: "
                                f"stage={key!r}, multiplier={multiplier}, "
                                f"physical_workers={physical_workers}")
                        local_multiplier = multiplier // len(physical_workers)
                        occurrences = tuple(
                            (worker, stage, local_multiplier)
                            for worker in physical_workers)
                    else:
                        occurrences = ((task.worker, stage, multiplier),)
                    if key in launch_shared_memory_occurrences:
                        launch_shared_memory_occurrences[key].extend(
                            occurrences)
                    else:
                        launch_shared_memory_occurrences[key] = list(
                            occurrences)

        staged_launch_shared_memory_bytes = {
            name: {} for name in absolute_memory_node_services}
        expected_workers = set(range(work_plan.worker_count))
        for stage_key, occurrences in (
                launch_shared_memory_occurrences.items()):
            workers = tuple(worker for worker, _, _ in occurrences)
            multipliers = tuple(multiplier for _, _, multiplier in occurrences)
            if any(multiplier != 1 for multiplier in multipliers):
                raise ValueError(
                    "launch-shared pipeline memory stages must have unit "
                    "occurrence multiplicity: "
                    f"stage={stage_key!r}, multipliers={multipliers}")
            if (len(occurrences) != work_plan.worker_count
                    or set(workers) != expected_workers
                    or len(workers) != len(set(workers))):
                raise ValueError(
                    "launch-shared pipeline memory stages must occur exactly "
                    "once "
                    f"on every worker: stage={stage_key!r}, "
                    f"workers={workers}")
            _, representative, _ = occurrences[0]
            if any(stage != representative
                   for _, stage, _ in occurrences[1:]):
                raise ValueError(
                    "launch-shared pipeline memory stages and source "
                    "dependencies must be symmetric across workers: "
                    f"stage={stage_key!r}")
            shared_bytes = staged_launch_shared_memory_bytes[
                representative.service]
            for demand in (
                    representative.launch_shared_memory_node_demands):
                key = (demand.path_name, demand.resource_role)
                if key in shared_bytes:
                    shared_bytes[key] += demand.bytes
                else:
                    shared_bytes[key] = demand.bytes
        fraction_by_service = {
            name: 0.0 for name in fractional_services}
        staged_operations = {
            name: {} for name in typed_compute_services}
        staged_operation_contribution_counts = {
            name: {} for name in typed_compute_services}
        staged_worker_operations = {
            name: {
                worker: {} for worker in range(work_plan.worker_count)
            }
            for name in typed_compute_services
        }
        staged_invocations = {
            name: {
                worker: {} for worker in range(work_plan.worker_count)
            }
            for name in typed_compute_services
        }
        staged_exact_requests = {
            name: {
                worker: {} for worker in range(work_plan.worker_count)
            }
            for name in typed_compute_services
        }
        staged_memory_bytes = {
            name: {} for name in absolute_memory_node_services}
        staged_memory_access_bytes = {}

        def _accumulate_task(task, multiplier, physical_workers=()):
            workers = physical_workers or (task.worker,)
            population = len(workers)
            for stage in task.stages:
                if (stage.compute_service_demands
                        or stage.compute_exact_request_demands):
                    operations = staged_operations[stage.service]
                    contribution_counts = (
                        staged_operation_contribution_counts[stage.service])
                    for demand in stage.compute_service_demands:
                        contribution = demand.operations * multiplier
                        aggregate_contribution = contribution * population
                        if demand.service in operations:
                            operations[demand.service] += aggregate_contribution
                        else:
                            operations[demand.service] = aggregate_contribution
                        if demand.service in contribution_counts:
                            contribution_counts[demand.service] += population
                        else:
                            contribution_counts[demand.service] = population
                        invocation_contribution = (
                            demand.serial_invocations * multiplier)
                        for worker in workers:
                            worker_operations = staged_worker_operations[
                                stage.service][worker]
                            worker_invocations = staged_invocations[
                                stage.service][worker]
                            if demand.service in worker_operations:
                                worker_operations[demand.service] += contribution
                            else:
                                worker_operations[demand.service] = contribution
                            if demand.service in worker_invocations:
                                worker_invocations[demand.service] += (
                                    invocation_contribution)
                            else:
                                worker_invocations[demand.service] = (
                                    invocation_contribution)
                    for demand in stage.compute_exact_request_demands:
                        local_runs, = demand.request_size_runs
                        for worker in workers:
                            worker_requests = staged_exact_requests[
                                stage.service][worker]
                            if demand.service not in worker_requests:
                                worker_requests[demand.service] = {}
                            service_runs = worker_requests[demand.service]
                            for request_size, count in local_runs:
                                if request_size in service_runs:
                                    service_runs[request_size] += (
                                        count * multiplier)
                                else:
                                    service_runs[request_size] = (
                                        count * multiplier)
                elif (stage.memory_node_demands
                        or stage.launch_shared_memory_node_demands
                        or stage.memory_access_class_demands):
                    if stage.service in staged_memory_bytes:
                        memory_bytes = staged_memory_bytes[stage.service]
                    else:
                        memory_bytes = None
                    for demand in stage.memory_node_demands:
                        if memory_bytes is None:
                            raise AssertionError(
                                "memory-node stage service was not indexed")
                        key = (demand.path_name, demand.resource_role)
                        contribution = demand.bytes * multiplier * population
                        if key in memory_bytes:
                            memory_bytes[key] += contribution
                        else:
                            memory_bytes[key] = contribution
                    path_name = stage.service.removeprefix("memory:")
                    for demand in stage.memory_access_class_demands:
                        key = (
                            path_name,
                            demand.service,
                            demand.ingress_resource_role,
                            demand.signature,
                        )
                        contribution = (
                            demand.logical_bytes * multiplier * population)
                        if key in staged_memory_access_bytes:
                            staged_memory_access_bytes[key] += contribution
                        else:
                            staged_memory_access_bytes[key] = contribution
                else:
                    fraction_by_service[stage.service] += (
                        stage.work_fraction * multiplier * population)

        for entry in work_plan.tasks:
            sequence_counts = _sequence_counts(entry)
            if not sequence_counts:
                _accumulate_task(entry, entry.repeat_count)
            worker_multiplicity = (
                entry.worker_multiplicity
                if isinstance(entry, PipelineTaskProgramDemand) else 1)
            physical_workers = (
                entry.physical_workers
                if isinstance(entry, PipelineTaskProgramDemand) else ())
            for sequence, count in sequence_counts:
                if count == 0:
                    continue
                for task in sequence.tasks:
                    multiplier = (
                        task.repeat_count * sequence.repeat_count * count)
                    if not physical_workers:
                        multiplier *= worker_multiplicity
                    _accumulate_task(
                        task,
                        multiplier,
                        physical_workers)
        invalid_fractions = {
            name: value for name, value in fraction_by_service.items()
            if not math.isclose(value, 1.0, rel_tol=1e-9, abs_tol=1e-12)
        }
        if invalid_fractions:
            raise ValueError(
                "pipeline work fractions must sum to one per service: "
                f"{invalid_fractions}")
        compute_by_pipeline_service = {}
        for demand in compute_demands:
            service_name = f"compute:{demand.resource_role}"
            if service_name in compute_by_pipeline_service:
                raise ValueError(
                    "pipeline compute roles must have one aggregate demand: "
                    f"{service_name!r}")
            compute_by_pipeline_service[service_name] = demand
        required_worker_compute_services = {
            f"compute:{demand.resource_role}"
            for demand in compute_demands
            if (demand.exact_request_demands
                or any(service.operations_by_worker
                       for service in demand.service_demands))
        }
        unstaged_worker_compute_services = (
            required_worker_compute_services - typed_compute_services)
        if (unstaged_worker_compute_services
                and not allow_partial_worker_services):
            raise ValueError(
                "worker-scheduled compute demands require source-closed "
                "pipeline stages: "
                f"{sorted(unstaged_worker_compute_services)}")
        for service_name in typed_compute_services:
            if service_name not in compute_by_pipeline_service:
                raise ValueError(
                    "typed pipeline stage has no aggregate compute demand: "
                    f"{service_name!r}")
            aggregate = compute_by_pipeline_service[service_name]
            aggregate_by_typed_service = {
                demand.service: demand
                for demand in aggregate.service_demands
            }
            staged_by_typed_service = staged_operations[service_name]
            if (set(staged_by_typed_service)
                    != set(aggregate_by_typed_service)):
                raise ValueError(
                    "pipeline typed compute services disagree with aggregate "
                    f"demand for {service_name!r}: "
                    f"staged={sorted(staged_by_typed_service)}, "
                    f"aggregate={sorted(aggregate_by_typed_service)}")
            for typed_service, aggregate_demand in (
                    aggregate_by_typed_service.items()):
                staged_amount = staged_by_typed_service[typed_service]
                contribution_count = staged_operation_contribution_counts[
                    service_name][typed_service]
                # Each typed stage is independently rounded when an
                # observation/ablation rescales the source-defined pipeline.
                # Bound the conservation check by the number of accumulated
                # floating-point contributions, not merely the number of
                # logical worker tasks: one task can contain hundreds of
                # typed source stages.
                tolerance = max(
                    math.ulp(staged_amount),
                    math.ulp(aggregate_demand.operations),
                ) * max(1, contribution_count)
                if abs(staged_amount - aggregate_demand.operations) > (
                        tolerance):
                    raise ValueError(
                        "pipeline typed compute operations must conserve the "
                        f"aggregate demand for {service_name!r}/"
                        f"{typed_service!r}: staged={staged_amount}, "
                        f"aggregate={aggregate_demand.operations}")
                per_worker = staged_invocations[service_name]
                invocation_counts = []
                for worker in range(work_plan.worker_count):
                    worker_invocations = per_worker[worker]
                    if typed_service in worker_invocations:
                        invocation_counts.append(
                            worker_invocations[typed_service])
                    else:
                        invocation_counts.append(0)
                critical_invocations = max(invocation_counts)
                if (critical_invocations
                        != aggregate_demand.serial_invocations):
                    raise ValueError(
                        "pipeline typed compute critical-worker invocation "
                        f"count disagrees for {service_name!r}/"
                        f"{typed_service!r}: staged={critical_invocations}, "
                        f"aggregate={aggregate_demand.serial_invocations}")
                if aggregate_demand.operations_by_worker:
                    staged_worker_amounts = []
                    for worker in range(work_plan.worker_count):
                        worker_operations = staged_worker_operations[
                            service_name][worker]
                        if typed_service in worker_operations:
                            staged_worker_amounts.append(
                                worker_operations[typed_service])
                        else:
                            staged_worker_amounts.append(0.0)
                    staged_by_worker = tuple(staged_worker_amounts)
                    if len(aggregate_demand.operations_by_worker) != (
                            work_plan.worker_count):
                        raise ValueError(
                            "pipeline typed compute aggregate worker count "
                            f"disagrees for {service_name!r}/"
                            f"{typed_service!r}")
                    if any(not math.isclose(
                            staged, aggregate_worker,
                            rel_tol=1e-12, abs_tol=1e-9)
                           for staged, aggregate_worker in zip(
                               staged_by_worker,
                               aggregate_demand.operations_by_worker)):
                        raise ValueError(
                            "pipeline typed compute worker operations must "
                            f"conserve {service_name!r}/{typed_service!r}: "
                            f"staged={staged_by_worker}, aggregate="
                            f"{aggregate_demand.operations_by_worker}")
            aggregate_by_exact_service = {
                demand.service: demand
                for demand in aggregate.exact_request_demands
            }
            staged_exact_services = {
                exact_service
                for worker_requests in staged_exact_requests[
                    service_name].values()
                for exact_service in worker_requests
            }
            if staged_exact_services != set(aggregate_by_exact_service):
                raise ValueError(
                    "pipeline exact request services disagree with aggregate "
                    f"demand for {service_name!r}: "
                    f"staged={sorted(staged_exact_services)}, aggregate="
                    f"{sorted(aggregate_by_exact_service)}")
            for exact_service, aggregate_demand in (
                    aggregate_by_exact_service.items()):
                staged_worker_requests = []
                for worker in range(work_plan.worker_count):
                    worker_requests = staged_exact_requests[
                        service_name][worker]
                    if exact_service in worker_requests:
                        staged_worker_requests.append(
                            tuple(sorted(
                                worker_requests[exact_service].items())))
                    else:
                        staged_worker_requests.append(())
                staged_by_worker = tuple(staged_worker_requests)
                if staged_by_worker != aggregate_demand.request_size_runs:
                    raise ValueError(
                        "pipeline exact requests must conserve the aggregate "
                        f"worker schedule for {service_name!r}/"
                        f"{exact_service!r}: staged={staged_by_worker}, "
                        f"aggregate={aggregate_demand.request_size_runs}")
        for service_name in absolute_memory_node_services:
            if service_name in overridden_memory_services:
                raise ValueError(
                    "absolute pipeline memory stages cannot consume an "
                    f"Oracle-overridden path: {service_name!r}")
            memory = memory_by_service[service_name]
            if (memory.rate_unit != "bytes"
                    and service_name
                    not in local_memory_source_completion_services):
                raise ValueError(
                    "transaction-resolved absolute pipeline memory stages "
                    "require an explicit local source-completion contract: "
                    f"{service_name!r}")
            aggregate_bytes = {
                (memory.path_name, node.resource_role): node.bytes
                for node in memory.nodes
            }
            staged_bytes = staged_memory_bytes[service_name]
            shared_bytes = staged_launch_shared_memory_bytes[service_name]
            combined_bytes = dict(staged_bytes)
            for key, byte_count in shared_bytes.items():
                if key in combined_bytes:
                    combined_bytes[key] += byte_count
                else:
                    combined_bytes[key] = byte_count
            if set(combined_bytes) != set(aggregate_bytes):
                raise ValueError(
                    "pipeline memory nodes disagree with aggregate path "
                    f"{service_name!r}: staged={sorted(combined_bytes)}, "
                    f"aggregate={sorted(aggregate_bytes)}")
            for key, aggregate_amount in aggregate_bytes.items():
                staged_amount = combined_bytes[key]
                if not math.isclose(
                        staged_amount, aggregate_amount,
                        rel_tol=1e-12, abs_tol=1e-6):
                    raise ValueError(
                        "pipeline memory-node bytes must conserve aggregate "
                        f"demand for {key}: staged={staged_amount}, "
                        f"aggregate={aggregate_amount}")
        if absolute_memory_access_services:
            if set(staged_memory_access_bytes) != set(
                    memory_access_by_identity):
                raise ValueError(
                    "pipeline memory access-class demands disagree with "
                    "aggregate paths: staged="
                    f"{sorted(map(str, staged_memory_access_bytes))}, "
                    "aggregate="
                    f"{sorted(map(str, memory_access_by_identity))}")
            for identity, diagnostic in memory_access_by_identity.items():
                staged_amount = staged_memory_access_bytes[identity]
                if not math.isclose(
                        staged_amount, diagnostic.logical_bytes,
                        rel_tol=1e-12, abs_tol=1e-6):
                    raise ValueError(
                        "pipeline memory access-class bytes must conserve the "
                        f"aggregate demand for {identity}: "
                        f"staged={staged_amount}, "
                        f"aggregate={diagnostic.logical_bytes}")
        non_positive_services = {
            name: service_time[name] for name in fractional_services
            if service_time[name] <= 0
        }
        if non_positive_services:
            raise ValueError(
                "pipeline services must have positive modeled time: "
                f"{non_positive_services}")

        resolved_task_cache = {}
        resolved_stages_cache = {}
        resource_scope_by_name = {}

        def _scheduler_resource(stage, worker):
            if stage.aiv_ingress_group_size is None:
                return stage.service, "worker"
            if group_size is None or aiv_group_count is None:
                raise AssertionError("missing validated AIV ingress topology")
            name = (
                f"{stage.service}:aiv_group_"
                f"{worker % aiv_group_count}")
            return name, "shared"

        def _record_scheduler_resource(name, scope):
            if name in resource_scope_by_name:
                if resource_scope_by_name[name] != scope:
                    raise ValueError(
                        "pipeline scheduler resource has inconsistent scope: "
                        f"{name!r}")
            else:
                resource_scope_by_name[name] = scope

        def _resolve_task(task):
            cache_key = id(task)
            if cache_key in resolved_task_cache:
                source_task, resolved_task = resolved_task_cache[cache_key]
                if source_task is not task:
                    raise RuntimeError("pipeline task identity cache collision")
                return resolved_task
            stages_key = id(task.stages)
            cacheable_stages = not any(
                stage.aiv_ingress_group_size is not None
                for stage in task.stages)
            if cacheable_stages and stages_key in resolved_stages_cache:
                source_stages, resolved_stages = resolved_stages_cache[
                    stages_key]
                if source_stages is not task.stages:
                    raise RuntimeError(
                        "pipeline stage identity cache collision")
                resolved_task = PipelineTask(
                    name=task.name,
                    worker=task.worker,
                    stages=resolved_stages,
                    repeat_count=task.repeat_count,
                )
                resolved_task_cache[cache_key] = (task, resolved_task)
                return resolved_task
            stages = []
            for stage in task.stages:
                if (stage.compute_service_demands
                        or stage.compute_exact_request_demands):
                    aggregate = compute_by_pipeline_service[stage.service]
                    resource = self.hw.resource_by_role(
                        aggregate.resource_role)
                    execution_domain_units = (
                        self.hw.execution_units_for_task(launch_task_kind))
                    if resource.total_units % execution_domain_units:
                        raise ValueError(
                            "typed worker pipeline requires an integral "
                            "physical-resource count per execution-domain "
                            "worker")
                    typed_durations = []
                    typed_occupancies = []
                    for demand in stage.compute_service_demands:
                        capacity = resource.effective_service_capacity(
                            demand.service,
                            active_units=work_plan.worker_count,
                            execution_domain_units=execution_domain_units,
                        ) / work_plan.worker_count
                        effective_capacity = (
                            capacity * aggregate.shape_efficiency
                            * aggregate.schedule_efficiency)
                        if effective_capacity <= 0:
                            raise ValueError(
                                "typed pipeline compute service has no "
                                f"capacity: {stage.service!r}/"
                                f"{demand.service!r}")
                        duration = demand.operations / effective_capacity
                        typed_occupancies.append(duration)
                        if demand.serial_invocations > 0:
                            duration += (
                                demand.serial_invocations
                                * resource.setup_time_for_service(
                                    demand.service))
                        typed_durations.append(duration)
                    for demand in stage.compute_exact_request_demands:
                        local_runs, = demand.request_size_runs
                        request_duration = math.fsum(
                            resource.exact_request_service_time_s(
                                demand.service, request_size) * count
                            for request_size, count in local_runs)
                        typed_durations.append(request_duration)
                        typed_occupancies.append(
                            0.0 if demand.pipelined_completion
                            else request_duration)
                    duration = math.fsum(typed_durations)
                    occupancy = math.fsum(typed_occupancies)
                elif (stage.memory_node_demands
                        or stage.launch_shared_memory_node_demands
                        or stage.memory_access_class_demands):
                    memory = memory_by_service[stage.service]
                    capacity_by_role = {
                        node.resource_role: node.byte_capacity_per_s
                        for node in memory.nodes
                    }
                    unknown_roles = {
                        demand.resource_role
                        for demand in (
                            stage.memory_node_demands
                            + stage.launch_shared_memory_node_demands)
                        if demand.resource_role not in capacity_by_role
                    }
                    if unknown_roles:
                        raise ValueError(
                            "pipeline stage references unknown memory nodes "
                            f"for {stage.service!r}: {sorted(unknown_roles)}")
                    completion_roles = (
                        set(stage.completion_resource_roles)
                        if stage.completion_resource_roles is not None
                        else None)
                    replaced_capacity_role = (
                        stage.memory_exact_request_demands[0].resource_role
                        if stage.memory_exact_request_replaces_resource_capacity
                        else None)
                    per_worker_durations = tuple(
                        demand.bytes * work_plan.worker_count
                        / (capacity_by_role[demand.resource_role]
                           * (stage.aiv_ingress_group_size
                              if demand.resource_role == "mte2"
                              and stage.aiv_ingress_group_size is not None
                              else 1))
                        for demand in stage.memory_node_demands
                        if demand.resource_role != replaced_capacity_role
                        and (completion_roles is None
                             or demand.resource_role in completion_roles))
                    shared_durations = tuple(
                        demand.bytes
                        / capacity_by_role[demand.resource_role]
                        for demand in (
                            stage.launch_shared_memory_node_demands)
                        if demand.resource_role != replaced_capacity_role
                        and (completion_roles is None
                             or demand.resource_role in completion_roles))
                    durations = per_worker_durations + shared_durations
                    path_name = stage.service.removeprefix("memory:")
                    access_class_durations = []
                    for demand in stage.memory_access_class_demands:
                        identity = (
                            path_name,
                            demand.service,
                            demand.ingress_resource_role,
                            demand.signature,
                        )
                        if identity not in memory_access_by_identity:
                            raise ValueError(
                                "pipeline stage references an unknown memory "
                                f"access class: {identity!r}")
                        diagnostic = memory_access_by_identity[identity]
                        access_class_durations.append(
                            demand.logical_bytes * work_plan.worker_count
                            / diagnostic.logical_byte_capacity_per_s)
                    durations += tuple(access_class_durations)
                    if (not durations
                            and not stage.memory_exact_request_demands):
                        raise ValueError(
                            "absolute memory pipeline stage has no duration")
                    if durations:
                        duration = max(durations)
                    else:
                        duration = 0.0
                    if stage.local_memory_source_completion:
                        if stage.service in overridden_memory_services:
                            raise ValueError(
                                "local memory source completion cannot "
                                "consume an Oracle-overridden path: "
                                f"{stage.service!r}")
                        duration = max(
                            duration,
                            self._local_memory_source_completion_time(
                                stage, memory, work_plan.worker_count),
                        )
                    occupancy = duration
                else:
                    service_fraction = (
                        stage.work_fraction * work_plan.worker_count)
                    if stage.completion_resource_roles is not None:
                        # A coherent store can retire at an earlier node while
                        # its downstream bytes continue to drain. All nodes
                        # retain their aggregate resource/arbitration bounds.
                        if stage.service in overridden_memory_services:
                            raise ValueError(
                                "filtered memory completion cannot consume an overridden path")
                        memory = memory_by_service[stage.service]
                        if any(item.path_name == memory.path_name
                               for item in memory_access_class_diagnostics):
                            raise ValueError(
                                "filtered memory capacity cannot replace a typed memory service")
                        roles = set(stage.completion_resource_roles)
                        unknown = roles - {node.resource_role for node in memory.nodes}
                        if unknown:
                            raise ValueError(
                                f"pipeline completion references unknown memory nodes: {sorted(unknown)}")
                        duration = max(node.capacity_time_s for node in memory.nodes
                                       if node.resource_role in roles) * service_fraction
                    else:
                        duration = service_time[stage.service] * service_fraction
                    if stage.aiv_ingress_group_size is not None:
                        duration /= stage.aiv_ingress_group_size
                    service_occupancy = duration
                    if (stage.include_first_request_latency
                            and stage.service
                            not in overridden_memory_services):
                        memory = memory_by_service[stage.service]
                        if (memory.rate_unit != "transactions"
                                or memory.initial_transactions is None
                                or memory.first_request_latency_s is None):
                            raise ValueError(
                                "pipeline first-request latency requires a "
                                "transaction-resolved memory path")
                        # ``lambda_initial`` is the full-chip rate, while a
                        # task's work fraction is its actual share of
                        # transactions.  The worker-local stage below therefore
                        # uses lambda/W; scaling both the transaction count and
                        # the rate by W would charge extra completions after the
                        # first request on short tiles.
                        stage_transactions = (
                            memory.initial_transactions
                            * stage.work_fraction)
                        # The remaining transactions of this request burst are
                        # issued together with the first one, so they drain at
                        # the path's own service capacity.  ``lambda_initial``
                        # is the launch-average rate the solver derived from
                        # this very dependency, and it already contains the
                        # idle the dependency creates between bursts; reusing
                        # it here, on top of the full first-request latency,
                        # would charge that idle a second time.  Capacity is a
                        # path property, so it stays an upper bound on the
                        # burst and never lowers the solved aggregate envelope.
                        burst_rate_per_s = (
                            max(memory.lambda_initial_per_s,
                                memory.lambda_capacity_per_s)
                            if stage.burst_drains_at_path_capacity
                            else memory.lambda_initial_per_s)
                        worker_rate = (
                            burst_rate_per_s / work_plan.worker_count)
                        duration = (
                            memory.first_request_latency_s
                            + max(stage_transactions - 1.0, 0.0)
                            / worker_rate)
                    if stage.pipelined_memory_completion:
                        # Latency delays this operand's consumers, but does
                        # not reserve the copy service for the whole round
                        # trip. Keep its existing aggregate service share and
                        # let the retained buffers carry completion/reuse.
                        duration = max(duration, service_occupancy)
                        occupancy = service_occupancy
                    elif (stage.include_first_request_latency
                            and stage.burst_drains_at_path_capacity
                            and stage.service not in overridden_memory_services):
                        # A burst's return remains a dependency of its
                        # consumer; it does not occupy the byte port while
                        # another buffered burst is ready to issue. Charge
                        # every transaction to capacity, including the first.
                        duration, occupancy = self._burst_memory_stage_service(
                            memory, stage.work_fraction, work_plan.worker_count)
                    else:
                        occupancy = duration
                if stage.memory_exact_request_demands:
                    if stage.service in overridden_memory_services:
                        raise ValueError(
                            "exact memory completion cannot consume an "
                            f"Oracle-overridden path: {stage.service!r}")
                    memory = memory_by_service[stage.service]
                    path_roles = {
                        node.resource_role for node in memory.nodes}
                    exact_demand, = stage.memory_exact_request_demands
                    if exact_demand.resource_role not in path_roles:
                        raise ValueError(
                            "pipeline exact memory completion resource must "
                            f"belong to {stage.service!r}: got "
                            f"{exact_demand.resource_role!r}, path roles="
                            f"{sorted(path_roles)}")
                    resource = self.hw.resource_by_role(
                        exact_demand.resource_role)
                    if resource.type != "memory":
                        raise ValueError(
                            "pipeline exact memory completion requires a "
                            f"memory resource: {exact_demand.resource_role!r}")
                    local_requests, = (
                        exact_demand.request_sizes_by_worker)
                    exact_completion = math.fsum(
                        self.hw.memory_exact_request_service_time_s(
                            exact_demand.resource_role,
                            exact_demand.service, request_size)
                        for request_size in local_requests)
                    duration = max(duration, exact_completion)
                    if not exact_demand.pipelined_completion:
                        occupancy = duration
                scheduler_resource, scheduler_scope = _scheduler_resource(
                    stage, task.worker)
                _record_scheduler_resource(
                    scheduler_resource, scheduler_scope)
                for prior_resource in stage.prior_resources:
                    _record_scheduler_resource(prior_resource, "worker")
                stages.append(PipelineStage._from_validated_demand(
                    stage, duration, occupancy,
                    resource=scheduler_resource))
            resolved_stages = tuple(stages)
            if cacheable_stages:
                resolved_stages_cache[stages_key] = (
                    task.stages, resolved_stages)
            resolved_task = PipelineTask(
                name=task.name,
                worker=task.worker,
                stages=resolved_stages,
                repeat_count=task.repeat_count,
            )
            resolved_task_cache[cache_key] = (task, resolved_task)
            return resolved_task

        tasks = []
        for entry in work_plan.tasks:
            if isinstance(entry, PipelineTaskProgramDemand):
                tasks.append(PipelineTaskProgram(
                    name=entry.name,
                    worker=entry.worker,
                    patterns=tuple(PipelineTaskSequence(
                        name=pattern.name,
                        worker=pattern.worker,
                        tasks=tuple(
                            _resolve_task(task) for task in pattern.tasks),
                        repeat_count=pattern.repeat_count,
                    ) for pattern in entry.patterns),
                    order=entry.order,
                    repeat_count=entry.repeat_count,
                    tail_order=entry.tail_order,
                    worker_multiplicity=entry.worker_multiplicity,
                    physical_workers=entry.physical_workers,
                ))
            elif isinstance(entry, PipelineTaskSequenceDemand):
                tasks.append(PipelineTaskSequence(
                    name=entry.name,
                    worker=entry.worker,
                    tasks=tuple(_resolve_task(task) for task in entry.tasks),
                    repeat_count=entry.repeat_count,
                ))
            else:
                tasks.append(_resolve_task(entry))
        resources = tuple(
            PipelineResource(
                name=name, scope=resource_scope_by_name[name], slots=1)
            for name in sorted(resource_scope_by_name)
        )
        return PipelinePlan(
            worker_count=work_plan.worker_count,
            resources=resources,
            tasks=tuple(tasks),
            buffers=work_plan.buffers,
            initial_resource_availability_s=(
                work_plan.initial_resource_availability_s),
        )

    def _resolve_barrier_pipeline_envelope(
            self, envelope, compute_time_by_role, memory_diagnostics,
            memory_service_time, memory_access_class_diagnostics,
            overridden_memory_services, t_per_resource, compute_demands,
            active_units, launch_task_kind):
        """Resolve one representative of each full-barrier tile class."""

        if not isinstance(envelope, BarrierPipelineEnvelopeDemand):
            raise TypeError("expected BarrierPipelineEnvelopeDemand")
        if envelope.resolved_envelope is not None:
            resolved = envelope.resolved_envelope
            if (resolved.worker_count != envelope.worker_count
                    or resolved.counts_by_worker != envelope.counts_by_worker
                    or resolved.buffers != envelope.buffers):
                raise ValueError(
                    "barrier envelope resolved template disagrees with "
                    "its source histogram")
            return resolved
        class_counts = tuple(
            sum(row[index] for row in envelope.counts_by_worker)
            for index in range(len(envelope.classes)))
        validation_plan = PipelineWorkPlan(
            worker_count=envelope.worker_count,
            tasks=tuple(
                dataclasses.replace(sequence, repeat_count=count)
                for sequence, count in zip(envelope.classes, class_counts)),
            buffers=envelope.buffers,
        )
        resolved = self._resolve_pipeline(
            validation_plan,
            compute_time_by_role,
            memory_diagnostics,
            memory_service_time,
            memory_access_class_diagnostics,
            overridden_memory_services,
            t_per_resource,
            compute_demands,
            active_units,
            launch_task_kind,
        )
        if not isinstance(resolved, PipelinePlan):
            raise RuntimeError("barrier envelope resolution did not return a plan")
        if len(resolved.tasks) != len(envelope.classes):
            raise RuntimeError("barrier envelope class count changed during resolution")
        if any(not isinstance(sequence, PipelineTaskSequence)
               for sequence in resolved.tasks):
            raise RuntimeError(
                "barrier envelope resolution returned a non-sequence class")
        resolved_classes = tuple(
            PipelineTaskSequence(
                name=sequence.name,
                worker=0,
                tasks=tuple(dataclasses.replace(task, worker=0)
                            for task in sequence.tasks),
                repeat_count=1,
            )
            for sequence in resolved.tasks)
        class_durations = tuple(
            _evaluate_barrier_pipeline_sequence(sequence, envelope.buffers)
            for sequence in resolved_classes)
        return BarrierPipelineEnvelope(
            worker_count=envelope.worker_count,
            classes=resolved_classes,
            counts_by_worker=envelope.counts_by_worker,
            resources=resolved.resources,
            buffers=envelope.buffers,
            class_durations_s=class_durations,
            class_stage_service_s=tuple(
                tuple((stage.name, stage.duration_s * task.repeat_count)
                      for task in sequence.tasks for stage in task.stages)
                for sequence in resolved_classes),
        )

    def _resolve_cann_single_core_split_k_envelope(
            self, envelope, memory_diagnostics, compute_time_by_role,
            active_units, launch_task_kind):
        """Resolve key-65632's source-ordered AIC/AIV recurrence.

        The source owns one effective A1 slot per M tile in the inner stripe,
        two B1 slots and ping-pong L0C.  B is
        fetched one N tile ahead; at terminal N the next K group's B0 and
        A0/A1 are issued around the final M0/M1/M2 resident tiles.  A complete
        inner-M stripe then notifies its paired AIVs and drains the AIC pipes.
        The state below is constant sized regardless of M, N, or K.
        """

        if not isinstance(envelope, CannSingleCoreSplitKEnvelopeDemand):
            raise TypeError("expected CannSingleCoreSplitKEnvelopeDemand")
        if (launch_task_kind != "mixed"
                or active_units < envelope.worker_count):
            raise ValueError(
                "CANN single-core split-K envelope requires its AIC domain")
        path_by_name = {
            diagnostic.path_name: diagnostic
            for diagnostic in memory_diagnostics}
        required_paths = {
            "mte2_read", "aiv_workspace_mte2_read", "mte3_write"}
        if not required_paths.issubset(path_by_name):
            raise ValueError(
                "CANN single-core split-K envelope lacks required paths")

        def path_seconds_per_local_byte(path_name):
            diagnostic = path_by_name[path_name]
            byte_counts = {
                node.bytes for node in diagnostic.nodes
                if node.bytes > 0.0}
            if not byte_counts:
                raise ValueError(
                    f"CANN split-K path {path_name!r} has no byte demand")
            # Each path's steady service is a full-chip bound.  One AIC (or
            # its paired AIVs) owns one of ``worker_count`` concurrent blocks.
            # Scale a local source request by that same physical concurrency.
            path_bytes = max(byte_counts)
            return (
                diagnostic.steady_time_s * envelope.worker_count / path_bytes)

        operand_byte_s = path_seconds_per_local_byte("mte2_read")
        workspace_byte_s = path_seconds_per_local_byte(
            "aiv_workspace_mte2_read")
        output_byte_s = path_seconds_per_local_byte("mte3_write")
        total_output_elements = math.fsum(
            multiplicity * m_extent * n_extent
            for (m_extent, n_extent), multiplicity in zip(
                envelope.class_shapes, envelope.class_multiplicities))
        if total_output_elements <= 0.0:
            raise ValueError("CANN split-K envelope has no output elements")
        if "compute_vector" not in compute_time_by_role:
            raise ValueError("CANN split-K envelope has no Vector service")
        vector_second_per_local_element = (
            compute_time_by_role["compute_vector"]
            * envelope.worker_count / total_output_elements)
        resident_tile_time_cache = {}

        def resident_tile_time_ns(m_rows, n_columns, k_columns):
            key = (m_rows, n_columns, k_columns)
            if key not in resident_tile_time_cache:
                resident_tile_time_cache[key] = (
                    self.hw.aic_matmul_v3_split_k_resident_tile_time_ns(
                        envelope.dtype, envelope.worker_count,
                        m_rows, n_columns, k_columns))
            return resident_tile_time_cache[key]

        source_stripe_completion_cache = {}

        def source_stripe_completion_s(stripe_m, block_n):
            cache_key = (stripe_m, block_n)
            if cache_key in source_stripe_completion_cache:
                return source_stripe_completion_cache[cache_key]
            m_tiles = math.ceil(stripe_m / envelope.base_m)
            n_tiles = math.ceil(block_n / envelope.base_n)
            k_groups = math.ceil(
                envelope.projection_k / envelope.single_core_k)
            source_completion_ns = (
                self.hw.
                aic_mte2_gemm_capacity_constrained_source_completion_time_ns(
                    envelope.base_m, envelope.single_core_k,
                    m_tiles * k_groups,
                    envelope.base_n, envelope.single_core_k,
                    n_tiles * k_groups,
                    envelope.worker_count))
            if source_completion_ns is not None:
                resident_service = math.fsum(
                    resident_tile_time_ns(
                        min(envelope.base_m,
                            stripe_m - m_index * envelope.base_m),
                        min(envelope.base_n,
                            block_n - n_index * envelope.base_n),
                        min(envelope.single_core_k,
                            envelope.projection_k
                            - k_index * envelope.single_core_k))
                    for k_index in range(k_groups)
                    for n_index in range(n_tiles)
                    for m_index in range(m_tiles)) * 1e-9
                # MTE2/MTE1 and resident Cube/FixPipe are an overlapped source
                # pipeline.  Only the final resident tile drains after ingress
                # completion; steady work is the slower complete service.
                tail_resident_s = resident_tile_time_ns(
                    min(envelope.base_m, stripe_m),
                    min(envelope.base_n, block_n),
                    min(envelope.single_core_k, envelope.projection_k)
                ) * 1e-9
                completion = (
                    max(source_completion_ns * 1e-9, resident_service)
                    + tail_resident_s)
                result = completion, (
                    source_completion_ns * 1e-9, resident_service)
                source_stripe_completion_cache[cache_key] = result
                return result
            mte2_ready = 0.0
            resident_ready = 0.0
            a_slot_ready = [0.0] * envelope.a1_buffer_slots
            b_slot_ready = [0.0] * envelope.b1_buffer_slots
            a_data_ready = [0.0] * m_tiles
            b_data_ready = [0.0] * envelope.b1_buffer_slots
            b_ping = 0
            mte2_service = 0.0
            resident_service = 0.0

            def copy_a(tile_index, k_columns):
                nonlocal mte2_ready, mte2_service
                rows = min(
                    envelope.base_m,
                    stripe_m - tile_index * envelope.base_m)
                duration = (
                    rows * k_columns * envelope.element_bytes
                    * operand_byte_s)
                slot = tile_index % envelope.a1_buffer_slots
                mte2_ready = max(
                    mte2_ready, a_slot_ready[slot]) + duration
                a_data_ready[tile_index] = mte2_ready
                mte2_service += duration

            def copy_b(slot, n_columns, k_columns):
                nonlocal mte2_ready, mte2_service
                duration = (
                    n_columns * k_columns * envelope.element_bytes
                    * operand_byte_s)
                mte2_ready = max(
                    mte2_ready, b_slot_ready[slot]) + duration
                b_data_ready[slot] = mte2_ready
                mte2_service += duration

            initial_k = min(
                envelope.single_core_k, envelope.projection_k)
            for m_index in range(min(2, m_tiles)):
                copy_a(m_index, initial_k)
            copy_b(b_ping, min(envelope.base_n, block_n), initial_k)

            for k_index in range(k_groups):
                k_offset = k_index * envelope.single_core_k
                k_columns = min(
                    envelope.single_core_k,
                    envelope.projection_k - k_offset)
                last_k = k_index + 1 == k_groups
                if m_tiles == envelope.step_m:
                    copy_a(m_tiles - 1, k_columns)
                if n_tiles > 1:
                    next_slot = (b_ping + 1) % envelope.b1_buffer_slots
                    copy_b(
                        next_slot,
                        min(envelope.base_n, block_n - envelope.base_n),
                        k_columns)

                for n_index in range(n_tiles):
                    last_n = n_index + 1 == n_tiles
                    first_n = n_index == 0
                    current_b = b_ping
                    if not last_n and not first_n:
                        next_slot = (
                            b_ping + 1) % envelope.b1_buffer_slots
                        next_n = min(
                            envelope.base_n,
                            block_n - (n_index + 1) * envelope.base_n)
                        copy_b(next_slot, next_n, k_columns)
                    if last_n and not last_k:
                        next_slot = (
                            b_ping + 1) % envelope.b1_buffer_slots
                        next_k = min(
                            envelope.single_core_k,
                            envelope.projection_k
                            - (k_index + 1) * envelope.single_core_k)
                        copy_b(
                            next_slot, min(envelope.base_n, block_n), next_k)

                    n_columns = min(
                        envelope.base_n,
                        block_n - n_index * envelope.base_n)
                    for m_index in range(m_tiles):
                        m_rows = min(
                            envelope.base_m,
                            stripe_m - m_index * envelope.base_m)
                        resident_ns = resident_tile_time_ns(
                            m_rows, n_columns, k_columns)
                        duration = resident_ns * 1e-9
                        resident_ready = max(
                            resident_ready,
                            a_data_ready[m_index],
                            b_data_ready[current_b]) + duration
                        resident_service += duration
                        if last_n:
                            a_slot = m_index % envelope.a1_buffer_slots
                            a_slot_ready[a_slot] = resident_ready
                            if m_index < min(2, m_tiles) and not last_k:
                                next_k = min(
                                    envelope.single_core_k,
                                    envelope.projection_k
                                    - (k_index + 1)
                                    * envelope.single_core_k)
                                copy_a(m_index, next_k)
                    b_slot_ready[current_b] = resident_ready
                    b_ping = (b_ping + 1) % envelope.b1_buffer_slots
            completion = max(mte2_ready, resident_ready), (
                mte2_service, resident_service)
            source_stripe_completion_cache[cache_key] = completion
            return completion

        class_durations = []
        class_services = []
        classes = []
        inner_block_m = envelope.step_m * envelope.base_m
        for class_index, (block_m, block_n) in enumerate(
                envelope.class_shapes):
            aic_ready = 0.0
            aiv_ready = 0.0
            total_mte2 = 0.0
            total_resident = 0.0
            total_workspace = 0.0
            total_vector = 0.0
            total_output = 0.0
            for stripe_start in range(0, block_m, inner_block_m):
                stripe_m = min(inner_block_m, block_m - stripe_start)
                aic_duration, services = source_stripe_completion_s(
                    stripe_m, block_n)
                aic_ready += aic_duration
                total_mte2 += services[0]
                total_resident += services[1]
                stripe_elements = stripe_m * block_n
                workspace_s = (
                    stripe_elements * envelope.partial_element_bytes
                    * workspace_byte_s)
                vector_s = (
                    stripe_elements * vector_second_per_local_element)
                output_s = (
                    stripe_elements * envelope.element_bytes
                    * output_byte_s)
                aiv_ready = max(aiv_ready, aic_ready) + (
                    workspace_s + vector_s + output_s)
                total_workspace += workspace_s
                total_vector += vector_s
                total_output += output_s
            completion = max(aic_ready, aiv_ready)
            stage = PipelineStage(
                name="source_completion",
                resource="cann_single_core_split_k_source",
                duration_s=completion)
            classes.append(PipelineTaskSequence(
                name=f"single_core_split_k_class{class_index}",
                worker=0,
                tasks=(PipelineTask(
                    name="source_program", worker=0, stages=(stage,)),),
            ))
            class_durations.append(completion)
            class_services.append((
                ("operand_mte2", total_mte2),
                ("resident_tile", total_resident),
                ("aiv_workspace_read", total_workspace),
                ("aiv_cast", total_vector),
                ("aiv_output_write", total_output),
            ))
        counts_by_worker = []
        for class_index, multiplicity in enumerate(
                envelope.class_multiplicities):
            row = tuple(
                int(index == class_index)
                for index in range(len(envelope.class_shapes)))
            counts_by_worker.extend((row,) * multiplicity)
        return BarrierPipelineEnvelope(
            worker_count=envelope.worker_count,
            classes=tuple(classes),
            counts_by_worker=tuple(counts_by_worker),
            resources=(PipelineResource(
                name="cann_single_core_split_k_source",
                scope="worker", slots=1),),
            class_durations_s=tuple(class_durations),
            class_stage_service_s=tuple(class_services),
        )

    def _resolve_cann_grouped_matmul_envelope(
            self, envelope, memory_diagnostics,
            memory_access_class_diagnostics, compute_demands,
            active_units, launch_task_kind):
        """Price one CANN GroupedMatmul source tile class analytically.

        The CANN source gives a full resource barrier before every output
        tile.  Its A/B copy, Cube and final writeback stages therefore compose
        serially within a tile and only the launch-provided per-AIC histogram
        remains.  This avoids materializing the equivalent generic stage DAG
        once for every M-tail class.
        """

        if not isinstance(envelope, CannGroupedMatmulEnvelopeDemand):
            raise TypeError("expected CannGroupedMatmulEnvelopeDemand")
        if launch_task_kind != "aic" or active_units != envelope.worker_count:
            raise ValueError(
                "CANN GroupedMatmul envelope must cover the active AIC domain")
        grouped_matmul = envelope.source_kind == "grouped_matmul"
        persistent_matmul = envelope.source_kind == "matmul_v2"
        scan_demands = tuple(
            exact_demand
            for compute in compute_demands
            if compute.resource_role == "compute_fma"
            for exact_demand in compute.exact_request_demands
            if exact_demand.service
            == "cann_grouped_matmul_finite_group_scan_control")
        if len(scan_demands) > 1:
            raise ValueError(
                "CANN GroupedMatmul has repeated finite group-scan requests")
        if not grouped_matmul:
            control_s = 0.0
            control_stage_name = "matmul_output_control"
        elif scan_demands:
            scan_demand, = scan_demands
            if scan_demand.worker_count != envelope.worker_count:
                raise ValueError(
                    "CANN GroupedMatmul group-scan workers disagree with AICs")
            scan_resource = self.hw.resource_by_role("compute_fma")
            scan_time_by_worker = tuple(
                math.fsum(
                    scan_resource.exact_request_service_time_s(
                        scan_demand.service, request_size) * count
                    for request_size, count in worker_runs)
                for worker_runs in scan_demand.request_size_runs)
            if len(set(scan_time_by_worker)) != 1:
                raise ValueError(
                    "CANN GroupedMatmul group-scan request must match all AICs")
            control_s = scan_time_by_worker[0]
            control_stage_name = "finite_group_scan_control"
        else:
            control_s = (
                envelope.zero_expert_count
                * self.hw.resource_by_role(
                    "compute_fma").exact_request_service_time_s(
                        "cann_grouped_matmul_zero_expert_control", 1))
            control_stage_name = "zero_expert_control"
        entry_metadata_prelude_s = 0.0
        if (grouped_matmul
                and self.hw.aic_grouped_matmul_entry_metadata_pair_supports(
                    envelope.dtype, envelope.worker_count,
                    envelope.source_kind, envelope.projection_k,
                    envelope.projection_n)):
            # The CANN BF16 GMM_FLOAT/GMM_CUBE_IMP entry materializes its
            # typed GMMBaseParams then TCubeTiling members before GMMProcess.
            # This launch-wide AIC prelude is independent of the later
            # output-tile recurrence, so charge it once, never per tile.
            entry_metadata_prelude_s = (
                self.hw.aic_grouped_matmul_entry_metadata_pair_time_ns(
                    envelope.dtype, envelope.worker_count,
                    envelope.source_kind, envelope.projection_k,
                    envelope.projection_n) * 1e-9)
        path_by_name = {diagnostic.path_name: diagnostic
                        for diagnostic in memory_diagnostics}
        input_dram_path_name = (
            "operand_dram_read"
            if "operand_dram_read" in path_by_name else "mte2_read")
        required_paths = {
            "mte2_read", input_dram_path_name,
            envelope.output_path_name}
        if not required_paths.issubset(path_by_name):
            raise ValueError("CANN GroupedMatmul envelope lacks required paths")
        access_by_service = {
            diagnostic.service: diagnostic
            for diagnostic in memory_access_class_diagnostics}
        allowed_access_services = {
            "cann_grouped_moe_operand_a_l2_hit",
            "cann_grouped_moe_operand_a_dram_miss",
            "cann_grouped_moe_operand_b_l2_hit",
            "cann_grouped_moe_operand_b_k2880_stride_l2_hit",
            "cann_grouped_moe_operand_b_dram_miss",
            "cann_grouped_moe_operand_b_k2880_strided_dram_miss",
            "cann_grouped_moe_operand_b_k6144_m1_dram_miss",
        }
        if not set(access_by_service).issubset(allowed_access_services):
            raise ValueError(
                "CANN GroupedMatmul contains an unknown operand service")
        typed_a_available = any(
            service in access_by_service for service in (
                "cann_grouped_moe_operand_a_l2_hit",
                "cann_grouped_moe_operand_a_dram_miss"))
        typed_b_available = any(
            service in access_by_service for service in (
                "cann_grouped_moe_operand_b_l2_hit",
                "cann_grouped_moe_operand_b_k2880_stride_l2_hit",
                "cann_grouped_moe_operand_b_dram_miss",
                "cann_grouped_moe_operand_b_k2880_strided_dram_miss",
                "cann_grouped_moe_operand_b_k6144_m1_dram_miss"))
        tensor = self.hw.resource_by_role("compute_tensor")
        tensor_capacity = tensor.effective_capacity(
            1,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz)
        domain_units = self.hw.execution_units_for_task("aic")
        if tensor.scope == "shared":
            tensor_capacity /= domain_units
        elif tensor.scope == "per_unit":
            pass
        else:
            raise ValueError("CANN GroupedMatmul Tensor scope is unsupported")
        if tensor_capacity <= 0.0:
            raise ValueError("CANN GroupedMatmul has no Tensor capacity")
        l0a = self.hw.resource_by_role("l0a_buffer")
        l0b = self.hw.resource_by_role("l0b_buffer")
        l0a_capacity = l0a.effective_capacity(
            1,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        )
        l0b_capacity = l0b.effective_capacity(
            1,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        )
        if l0a_capacity <= 0.0 or l0b_capacity <= 0.0:
            raise ValueError("CANN GroupedMatmul has no MTE1 capacity")
        node_capacity = {
            path_name: {
                node.resource_role: node.byte_capacity_per_s
                for node in diagnostic.nodes}
            for path_name, diagnostic in path_by_name.items()}
        for path_name in required_paths:
            if any(capacity <= 0.0
                   for capacity in node_capacity[path_name].values()):
                raise ValueError("CANN GroupedMatmul memory capacity is invalid")
        k_tiles = math.ceil(envelope.projection_k / envelope.tile_k)
        if k_tiles <= 0:
            raise ValueError("CANN GroupedMatmul has no K tiles")
        output_tiles_per_worker = tuple(
            sum(counts) for counts in envelope.counts_by_worker)
        small_m_base_k_control_eligible = (
            persistent_matmul
            and self.hw.aic_matmul_v2_small_m_base_k_control_service
                is not None
            and len(set(output_tiles_per_worker)) == 1
            and envelope.a1_buffer_slots == envelope.b1_buffer_slots
            and all(
                logical_m % 16 and logical_n == envelope.tile_n
                for logical_m, logical_n in envelope.class_shapes)
            and self.hw.aic_matmul_v2_small_m_base_k_control_supports(
                envelope.dtype,
                envelope.worker_count,
                envelope.tile_m,
                envelope.tile_n,
                envelope.tile_k,
                envelope.projection_k,
                output_tiles_per_worker[0],
                envelope.a_step_k_tiles * envelope.tile_k,
                envelope.b_step_k_tiles * envelope.tile_k,
                envelope.a1_buffer_slots,
                True))
        small_m_base_k_control_s = (
            self.hw.aic_matmul_v2_small_m_base_k_control_time_ns() * 1e-9
            if small_m_base_k_control_eligible else 0.0)
        persistent_full_a = (
            persistent_matmul
            and envelope.a1_buffer_slots == 1
            and envelope.a_step_k_tiles == k_tiles)
        if "l2_cache" not in node_capacity["mte2_read"]:
            raise ValueError("CANN GEMM MTE2 path has no L2 service")
        l2_capacity = node_capacity["mte2_read"]["l2_cache"]
        if "dram" in node_capacity[input_dram_path_name]:
            dram_capacity = node_capacity[input_dram_path_name]["dram"]
        elif (envelope.a_l2_hit_fraction == 1.0
              and envelope.b_l2_hit_fraction == 1.0):
            # A fully resident persistent program has no operand continuation
            # to DRAM.  Its output write remains a separate FixPipe path; an
            # infinite unused miss capacity keeps the same finite recurrence
            # without manufacturing a DRAM edge.
            dram_capacity = math.inf
        else:
            raise ValueError("CANN GEMM operand path has no DRAM service")
        mte2_mechanism = self.hw.aic_mte2_nd2nz_service
        operand_group_service_cache = {}

        def _capacity_bound(byte_count, miss_fraction):
            return max(
                byte_count * envelope.worker_count / l2_capacity,
                byte_count * miss_fraction * envelope.worker_count
                / dram_capacity,
            )

        def _generic_a_service(
                byte_count, miss_fraction, *, first_descriptor):
            """Price one source A1 descriptor with a monotone capacity tail."""

            bound_s = _capacity_bound(byte_count, miss_fraction)
            if mte2_mechanism is None:
                return bound_s
            descriptor_bytes = math.ceil(byte_count)
            contiguous_ns = (
                None
                if grouped_matmul
                else self.hw.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
                    descriptor_bytes, envelope.worker_count,
                    stream=not first_descriptor,
                    dram_miss_fraction=miss_fraction)
            )
            if contiguous_ns is not None:
                return max(bound_s, contiguous_ns * 1e-9)
            points = (
                mte2_mechanism["descriptor_bytes"]
                if first_descriptor
                else mte2_mechanism["stream_descriptor_bytes"])
            descriptor_bytes = max(descriptor_bytes, points[0])
            service = (
                self.hw.aic_mte2_nd2nz_serial_time_ns
                if first_descriptor
                else self.hw.aic_mte2_nd2nz_stream_time_ns)
            if descriptor_bytes > points[-1]:
                # The largest independently measured descriptor remains a
                # completion lower bound for every larger request.  Dropping
                # directly to the aggregate byte-capacity bound at this edge
                # made a larger M tail complete faster than a smaller one.
                return max(
                    bound_s,
                    service(points[-1], envelope.worker_count) * 1e-9,
                )
            return max(
                bound_s,
                service(descriptor_bytes, envelope.worker_count) * 1e-9,
            )

        def _typed_or_capacity_service(
                service, byte_count, miss_fraction):
            if byte_count == 0.0:
                return 0.0
            bound_s = _capacity_bound(byte_count, miss_fraction)
            if service is None:
                return bound_s
            if service not in access_by_service:
                raise ValueError(
                    "CANN GroupedMatmul source class service is missing "
                    "from its aggregate memory path")
            access_s = (
                byte_count * envelope.worker_count
                / access_by_service[service].logical_byte_capacity_per_s)
            return max(bound_s, access_s)

        def _operand_group_service(
                *, operand, logical_extent, group_k, hit_fraction,
                typed_available, first_descriptor):
            cache_key = (
                operand, logical_extent, group_k, hit_fraction,
                typed_available, first_descriptor)
            if cache_key in operand_group_service_cache:
                return operand_group_service_cache[cache_key]
            byte_count = float(
                logical_extent * group_k * envelope.element_bytes)
            full_k_tiles = group_k // envelope.tile_k
            if operand == "a":
                tile_extent = envelope.tile_m
                hit_service = "cann_grouped_moe_operand_a_l2_hit"
                miss_service = "cann_grouped_moe_operand_a_dram_miss"
            elif operand == "b":
                tile_extent = envelope.tile_n
                hit_service = (
                    "cann_grouped_moe_operand_b_k2880_stride_l2_hit"
                    if "cann_grouped_moe_operand_b_k2880_stride_l2_hit"
                    in access_by_service
                    else "cann_grouped_moe_operand_b_l2_hit")
                miss_service = next((service for service in (
                        "cann_grouped_moe_operand_b_k2880_strided_dram_miss",
                        "cann_grouped_moe_operand_b_k6144_m1_dram_miss",
                        "cann_grouped_moe_operand_b_dram_miss")
                        if service in access_by_service), None)
            else:
                raise ValueError(
                    f"unsupported CANN GroupedMatmul operand: {operand!r}")
            typed_bytes = float(
                full_k_tiles * tile_extent * envelope.tile_k
                * envelope.element_bytes
                if typed_available and logical_extent == tile_extent else 0.0)
            tail_bytes = max(byte_count - typed_bytes, 0.0)
            typed_capacity_s = math.fsum((
                _typed_or_capacity_service(
                    hit_service, typed_bytes * hit_fraction, 0.0),
                _typed_or_capacity_service(
                    miss_service, typed_bytes * (1.0 - hit_fraction), 1.0),
                _typed_or_capacity_service(
                    None, tail_bytes, 1.0 - hit_fraction),
            ))
            if operand == "a":
                if not grouped_matmul and logical_extent != tile_extent:
                    # MatmulImpl's dynamic M tail is applied by the base-K
                    # copy loop.  The independent step-K sweep shows that
                    # widening its A1 group beyond 256 K elements does not
                    # collapse the partial-M copies into one descriptor: the
                    # complete worker program is unchanged (and slightly
                    # slower) for K-group 256 -> 512.  Preserve 256 as the
                    # source-visible descriptor span while the launch's
                    # larger A1 group still controls buffer credit/release.
                    maximum_descriptor_k = (
                        self.hw.
                        aic_matmul_v2_dynamic_m_tail_maximum_descriptor_k())
                    descriptor_services = []
                    remaining_k = group_k
                    descriptor_index = 0
                    while remaining_k:
                        descriptor_k = min(
                            remaining_k, maximum_descriptor_k)
                        descriptor_services.append(_generic_a_service(
                            float(logical_extent * descriptor_k
                                  * envelope.element_bytes),
                            1.0 - hit_fraction,
                            first_descriptor=(
                                first_descriptor and descriptor_index == 0)))
                        remaining_k -= descriptor_k
                        descriptor_index += 1
                    generic_service_s = math.fsum(descriptor_services)
                    result = max(typed_capacity_s, generic_service_s)
                    operand_group_service_cache[cache_key] = result
                    return result
                if not grouped_matmul and typed_bytes == byte_count:
                    # The typed pair stream already measures the completed
                    # base-K A issue interval at this exact physical tile.
                    # Repeating the isolated whole-descriptor ND2NZ span for
                    # every larger A1 group serializes startup that the
                    # source-visible two-slot stream overlaps.  Tails remain
                    # on the generic descriptor service below because the
                    # typed full-tile mechanism does not cover them.
                    return typed_capacity_s
                generic_service_s = _generic_a_service(
                    byte_count, 1.0 - hit_fraction,
                    first_descriptor=first_descriptor)
                if grouped_matmul:
                    result = (
                        generic_service_s
                        if typed_bytes == 0.0 else typed_capacity_s)
                else:
                    # The typed full-tile service and generic ND2NZ descriptor
                    # service constrain the same physical MTE2 transfer. Keep
                    # both bounds for every A extent instead of switching
                    # models at the exact base-M boundary.
                    result = max(typed_capacity_s, generic_service_s)
            else:
                result = typed_capacity_s
                if (not grouped_matmul
                        and envelope.element_bytes == 2
                        and logical_extent % envelope.tile_n == 0
                        and self.hw.aic_mte2_nd2nz_cold_exact_supports(
                            envelope.tile_n, group_k, envelope.projection_k,
                            envelope.worker_count)):
                    # A B1 group issues one source N-major slab descriptor
                    # per output tile.  The descriptor walks the strided
                    # dimension: with the physical K as the source row
                    # stride and B stored [N, K] row-major (b_layout=
                    # row_major, trans_b=true in the canonical GEMM launch
                    # observation), consecutive rows step across N, so the
                    # measured geometry is (N extent x K extent), not the
                    # transpose.  When that exact slab geometry at this
                    # launch's AIC count owns an independently measured cold
                    # stream service, blend it with the resident descriptor
                    # service by the launch's own L2 hit fraction (the same
                    # resident-plus-miss structure as the FIA cold completion
                    # model) and keep it as a per-descriptor lower bound next
                    # to the aggregate byte capacity.  The resident side is
                    # the generic serial table clamped at its largest
                    # measured descriptor, a lower bound for these larger
                    # slabs.  Geometries outside the measured domain keep the
                    # capacity-only bound.
                    cold_slab_ns = (
                        self.hw.aic_mte2_nd2nz_cold_exact_time_ns(
                            envelope.tile_n, group_k,
                            envelope.projection_k, envelope.worker_count,
                            stream=not first_descriptor))
                    resident_slab_ns = (
                        self.hw.aic_mte2_nd2nz_serial_time_ns(
                            min(
                                group_k * envelope.tile_n
                                * envelope.element_bytes,
                                mte2_mechanism["descriptor_bytes"][-1]),
                            envelope.worker_count))
                    blended_slab_ns = math.fsum((
                        hit_fraction * resident_slab_ns,
                        (1.0 - hit_fraction) * cold_slab_ns,
                    ))
                    result = max(
                        result,
                        blended_slab_ns
                        * (logical_extent // envelope.tile_n) * 1e-9)
            # A measured resident descriptor completion and the aggregate
            # byte capacity constrain the same physical MTE2 transfer, so the
            # measured completion is a lower bound on it - the same
            # composition this function already applies to the cold slab
            # above, and the one the resident-descriptor reference declares.
            # The operand group issues one descriptor of
            # (tile extent x group K) at the physical K stride, which is the
            # geometry the reference is keyed on.  Launches outside the
            # measured domain keep the capacity-only bound.
            if (logical_extent >= tile_extent
                    and self.hw.aic_mte2_gemm_large_resident_supports(
                        tile_extent, group_k, envelope.projection_k,
                        envelope.worker_count)):
                if first_descriptor:
                    resident_ns = (
                        self.hw.aic_mte2_gemm_large_resident_serial_time_ns(
                            tile_extent, group_k, envelope.projection_k,
                            envelope.worker_count))
                else:
                    resident_ns = (
                        self.hw.aic_mte2_gemm_large_resident_stream_time_ns(
                            tile_extent, group_k, envelope.projection_k,
                            envelope.worker_count))
                result = max(
                    result,
                    resident_ns * 1e-9 * (logical_extent // tile_extent))
            operand_group_service_cache[cache_key] = result
            return result

        a_hit = envelope.a_l2_hit_fraction
        b_hit = envelope.b_l2_hit_fraction
        finite_stream_mechanism = (
            self.hw.aic_grouped_matmul_finite_stream_completion_service
            if grouped_matmul else None)
        # The independent tile-boundary campaign shows that the base-M16
        # physical pair stream continues across output-tile boundaries.  The
        # CANN MoE base-M128 program instead invokes IterateAll once per output
        # tile; its exact K2880 boundary is handled below by a separate finite
        # completion service.
        finite_stream_eligible = (
            finite_stream_mechanism is not None
            and envelope.worker_count == finite_stream_mechanism[
                "active_aic_groups"]
            and envelope.tile_m in (
                finite_stream_mechanism["a_rows"],
                2 * finite_stream_mechanism["a_rows"])
            and envelope.tile_k == finite_stream_mechanism["a_columns"]
            and envelope.tile_k == finite_stream_mechanism["b_rows"]
            and envelope.tile_n == finite_stream_mechanism["b_columns"]
            and envelope.element_bytes == 2
            and envelope.a_step_k_tiles == 1
            and envelope.b_step_k_tiles == finite_stream_mechanism[
                "drain_interval_pairs"]
            and b_hit == 0.0)
        finite_stream_supports = (
            self.hw.aic_grouped_matmul_finite_stream_completion_supports)
        finite_stream_time_ns = (
            self.hw.aic_grouped_matmul_finite_stream_completion_time_ns)
        wide_tail_mechanism = (
            self.hw.aic_grouped_matmul_k2880_wide_base_tail_completion_service
            if grouped_matmul else None)
        # M80 has the same captured K program as M64 (A1=24, B1=8,
        # Ka/Kb=12/4 and two A/B L0 buffers).  Its independent operand-pair
        # probe closes the same finite M16/M32-tail completion domain; select
        # it by that complete emitted program, never by routed-token count.
        m80_wide_tail_program = (
            envelope.tile_m == 80
            and envelope.a_step_k_tiles == 12
            and envelope.b_step_k_tiles == 4
            and envelope.a1_buffer_slots == 2
            and envelope.b1_buffer_slots == 2
            and envelope.l0a_buffer_slots == 2
            and envelope.l0b_buffer_slots == 2
            and envelope.l0c_buffer_slots == 1)
        # The completion service is indexed by the source-program base-M.
        # M80's captured program is exactly M64's program, so it shares that
        # service identity; its larger descriptor allocation is not a second
        # completion mechanism.
        wide_tail_service_base_m = (
            64 if m80_wide_tail_program else envelope.tile_m)
        wide_tail_eligible = (
            wide_tail_mechanism is not None
            and (envelope.tile_m in wide_tail_mechanism["base_m_values"]
                 or m80_wide_tail_program)
            and envelope.projection_k == wide_tail_mechanism["projection_k"]
            and envelope.worker_count == wide_tail_mechanism[
                "active_aic_groups"]
            and envelope.tile_k == wide_tail_mechanism["a_columns"]
            and envelope.tile_k == wide_tail_mechanism["b_rows"]
            and envelope.tile_n == wide_tail_mechanism["b_columns"]
            and envelope.element_bytes == 2
            and envelope.b_step_k_tiles == wide_tail_mechanism[
                "drain_interval_pairs"]
            and b_hit == 0.0)
        output_tile_mechanism = (
            self.hw.aic_grouped_matmul_k2880_output_tile_completion_service
            if grouped_matmul else None)
        output_tile_eligible = (
            output_tile_mechanism is not None
            and envelope.tile_m in output_tile_mechanism["base_m_values"]
            and envelope.projection_k == output_tile_mechanism["projection_k"]
            and envelope.worker_count == output_tile_mechanism[
                "active_aic_groups"]
            and envelope.tile_k == output_tile_mechanism["a_columns"]
            and envelope.tile_k == output_tile_mechanism["b_rows"]
            and envelope.tile_n == output_tile_mechanism["b_columns"]
            and envelope.element_bytes == 2
            and envelope.b_step_k_tiles == output_tile_mechanism[
                "drain_interval_pairs"]
            and k_tiles == output_tile_mechanism["output_tile_pairs"]
            and b_hit == 0.0)
        if not grouped_matmul:
            step4_role = None
        elif envelope.projection_k == 4096:
            step4_role = "w1"
        elif envelope.projection_k == 1024:
            step4_role = "w2"
        else:
            step4_role = None
        step4_worker_counts = tuple(
            sum(counts) for counts in envelope.counts_by_worker)
        step4_worker_sequence_eligible = (
            step4_role is not None
            and b_hit == 0.0
            and all(
                math.ceil(logical_m / 16) * 16 == 16
                and logical_n == 256
                for logical_m, logical_n in envelope.class_shapes)
            and all(
                self.hw.aic_grouped_matmul_step4_worker_sequence_supports(
                    step4_role, envelope.dtype, envelope.worker_count,
                    envelope.tile_m, 16, envelope.projection_k,
                    envelope.projection_n, envelope.tile_n, envelope.tile_k,
                    envelope.a1_buffer_slots * envelope.a_step_k_tiles,
                    envelope.b1_buffer_slots * envelope.b_step_k_tiles,
                    envelope.a_step_k_tiles, envelope.b_step_k_tiles,
                    envelope.l0a_buffer_slots, envelope.l0b_buffer_slots,
                    envelope.l0c_buffer_slots, output_tile_count)
                for output_tile_count in step4_worker_counts))
        short_prelude_mechanism = (
            self.hw.aic_grouped_matmul_short_source_prelude_control_service)
        short_prelude_tail_mechanism = (
            self.hw.aic_grouped_matmul_short_source_tail_prelude_control_service)
        m32_short_prelude_mechanism = (
            self.hw.aic_grouped_matmul_m32_short_source_prelude_control_service)
        short_prelude_common_arguments = (
            envelope.dtype, envelope.worker_count, envelope.tile_m, 16,
            envelope.projection_k, envelope.tile_n, envelope.tile_k,
            envelope.a1_buffer_slots * envelope.a_step_k_tiles,
            envelope.b1_buffer_slots * envelope.b_step_k_tiles,
            envelope.a_step_k_tiles, envelope.b_step_k_tiles,
            envelope.l0a_buffer_slots, envelope.l0b_buffer_slots,
            envelope.l0c_buffer_slots)
        short_prelude_base_eligible = (
            short_prelude_mechanism is not None
            and all(
                math.ceil(logical_m / 16) * 16 == 16
                for logical_m, _ in envelope.class_shapes)
            and self.hw.aic_grouped_matmul_short_source_prelude_control_supports(
                *short_prelude_common_arguments))
        short_prelude_eligible = (
            short_prelude_base_eligible
            and all(
                logical_n == envelope.tile_n
                or (short_prelude_tail_mechanism is not None
                    and self.hw.
                    aic_grouped_matmul_short_source_tail_prelude_control_supports(
                        envelope.dtype, envelope.worker_count,
                        envelope.tile_m, 16, logical_n,
                        envelope.projection_k, envelope.tile_n,
                        envelope.tile_k,
                        envelope.a1_buffer_slots * envelope.a_step_k_tiles,
                        envelope.b1_buffer_slots * envelope.b_step_k_tiles,
                        envelope.a_step_k_tiles, envelope.b_step_k_tiles,
                        envelope.l0a_buffer_slots, envelope.l0b_buffer_slots,
                        envelope.l0c_buffer_slots))
                for _, logical_n in envelope.class_shapes))
        m32_short_prelude_eligible = (
            m32_short_prelude_mechanism is not None
            and envelope.tile_m == 32
            and all(
                self.hw.
                aic_grouped_matmul_m32_short_source_prelude_control_supports(
                    envelope.dtype, envelope.worker_count, envelope.tile_m,
                    32, logical_n, envelope.projection_k, envelope.tile_n,
                    envelope.tile_k,
                    envelope.a1_buffer_slots * envelope.a_step_k_tiles,
                    envelope.b1_buffer_slots * envelope.b_step_k_tiles,
                    envelope.a_step_k_tiles, envelope.b_step_k_tiles,
                    envelope.l0a_buffer_slots, envelope.l0b_buffer_slots,
                    envelope.l0c_buffer_slots)
                for _, logical_n in envelope.class_shapes))

        def short_prelude_time_s(logical_n):
            if short_prelude_eligible and logical_n == envelope.tile_n:
                return (
                    self.hw.
                    aic_grouped_matmul_short_source_prelude_control_time_ns(
                        *short_prelude_common_arguments) * 1e-9)
            if short_prelude_eligible:
                return (
                    self.hw.
                    aic_grouped_matmul_short_source_tail_prelude_control_time_ns(
                        envelope.dtype, envelope.worker_count, envelope.tile_m,
                        16, logical_n, envelope.projection_k, envelope.tile_n,
                        envelope.tile_k,
                        envelope.a1_buffer_slots * envelope.a_step_k_tiles,
                        envelope.b1_buffer_slots * envelope.b_step_k_tiles,
                        envelope.a_step_k_tiles, envelope.b_step_k_tiles,
                        envelope.l0a_buffer_slots, envelope.l0b_buffer_slots,
                        envelope.l0c_buffer_slots) * 1e-9)
            if m32_short_prelude_eligible:
                return (
                    self.hw.
                    aic_grouped_matmul_m32_short_source_prelude_control_time_ns(
                        envelope.dtype, envelope.worker_count, envelope.tile_m,
                        32, logical_n, envelope.projection_k, envelope.tile_n,
                        envelope.tile_k,
                        envelope.a1_buffer_slots * envelope.a_step_k_tiles,
                        envelope.b1_buffer_slots * envelope.b_step_k_tiles,
                        envelope.a_step_k_tiles, envelope.b_step_k_tiles,
                        envelope.l0a_buffer_slots, envelope.l0b_buffer_slots,
                        envelope.l0c_buffer_slots) * 1e-9)
            return 0.0
        # GMMProcess calls WaitPreTaskEnd before the group loop and
        # SetNextTaskStart after it.  GMMCompute uses IterateAll<false> inside
        # that loop, so the A1/B1, L0 and execution engines retain their
        # readiness state across output tiles.  Compile one source tile into
        # a small max-plus transition and apply the launch-derived RLE below;
        # this keeps the exact diagonal source order without one task per CTA.
        mte2_index = 0
        mte1_index = 1
        cube_index = 2
        store_index = 3
        prelude_index = 4
        a1_indexes = tuple(
            5 + offset for offset in range(envelope.a1_buffer_slots))
        b1_indexes = tuple(
            5 + envelope.a1_buffer_slots + offset
            for offset in range(envelope.b1_buffer_slots))
        l0a_indexes = tuple(
            5 + envelope.a1_buffer_slots + envelope.b1_buffer_slots + offset
            for offset in range(envelope.l0a_buffer_slots))
        l0b_indexes = tuple(
            5 + envelope.a1_buffer_slots + envelope.b1_buffer_slots
            + envelope.l0a_buffer_slots + offset
            for offset in range(envelope.l0b_buffer_slots))
        controller_state_size = 5 + sum(map(len, (
            a1_indexes, b1_indexes, l0a_indexes, l0b_indexes)))
        controller_identity = _maxplus_identity(controller_state_size)

        def controller_after_one(vector, duration_s):
            if (not math.isfinite(duration_s)
                    or duration_s < 0.0):
                raise ValueError(
                    "CANN controller stage duration must be non-negative")
            return [
                value + duration_s if math.isfinite(value) else value
                for value in vector]

        def controller_after_two(left, right, duration_s):
            if (not math.isfinite(duration_s)
                    or duration_s < 0.0):
                raise ValueError(
                    "CANN controller stage duration must be non-negative")
            return [
                (left_value if left_value >= right_value else right_value)
                + duration_s
                for left_value, right_value in zip(left, right, strict=True)]

        def controller_after_three(first, second, third, duration_s):
            if (not math.isfinite(duration_s)
                    or duration_s < 0.0):
                raise ValueError(
                    "CANN controller stage duration must be non-negative")
            # The staged comparisons preserve the previous tie ordering:
            # first row, then second, then third.  Iterating the three rows
            # together removes indexed loads from this inner controller loop.
            return [
                (first_value
                 if second_value <= first_value and third_value <= first_value
                 else second_value
                 if third_value <= second_value else third_value) + duration_s
                for first_value, second_value, third_value in zip(
                    first, second, third, strict=True)]

        class_matmul_programs = []
        mte1_pair_completion_cache = {}

        def controller_transition(
                logical_m, logical_n, *, cold_first_a=False):
            """Compile one IterateAll<false> call without resetting state."""

            padded_m = math.ceil(logical_m / 16) * 16
            padded_n = math.ceil(logical_n / 16) * 16
            prelude_s = short_prelude_time_s(logical_n)
            state = [list(row) for row in controller_identity]
            load_s = 0.0
            a_load_service_s = 0.0
            mte1_s = 0.0
            compute_s = 0.0
            a_group_ready = None
            b_group_ready = None
            a_group_index = -1
            b_group_index = -1
            if prelude_s:
                # MMCompute runs this scalar control before the same tile's
                # IterateAll.  It can overlap prior engines, but its first A
                # descriptor cannot issue before this service completes.
                state[prelude_index] = controller_after_one(
                    state[prelude_index], prelude_s)
            matmul_steps = []
            for k_index in range(k_tiles):
                k_offset = k_index * envelope.tile_k
                base_k = min(
                    envelope.tile_k, envelope.projection_k - k_offset)
                a_load_step_s = None
                b_load_step_s = None
                if k_index % envelope.a_step_k_tiles == 0:
                    a_group_index += 1
                    group_tiles = min(
                        envelope.a_step_k_tiles, k_tiles - k_index)
                    group_k = min(
                        group_tiles * envelope.tile_k,
                        envelope.projection_k - k_offset)
                    a_load_s = _operand_group_service(
                        operand="a", logical_extent=logical_m,
                        group_k=group_k, hit_fraction=a_hit,
                        typed_available=typed_a_available,
                        first_descriptor=(
                            a_group_index == 0
                            and (grouped_matmul or cold_first_a)))
                    a_load_step_s = a_load_s
                    a_slot = a1_indexes[
                        a_group_index % envelope.a1_buffer_slots]
                    if a_group_index == 0 and prelude_s:
                        state[mte2_index] = controller_after_three(
                            state[mte2_index], state[a_slot],
                            state[prelude_index], a_load_s)
                    else:
                        state[mte2_index] = controller_after_two(
                            state[mte2_index], state[a_slot], a_load_s)
                    a_group_ready = state[mte2_index]
                    load_s += a_load_s
                    a_load_service_s += a_load_s
                if k_index % envelope.b_step_k_tiles == 0:
                    b_group_index += 1
                    group_tiles = min(
                        envelope.b_step_k_tiles, k_tiles - k_index)
                    group_k = min(
                        group_tiles * envelope.tile_k,
                        envelope.projection_k - k_offset)
                    b_load_s = _operand_group_service(
                        operand="b", logical_extent=logical_n,
                        group_k=group_k, hit_fraction=b_hit,
                        typed_available=typed_b_available,
                        first_descriptor=(
                            grouped_matmul and b_group_index == 0))
                    b_load_step_s = b_load_s
                    b_slot = b1_indexes[
                        b_group_index % envelope.b1_buffer_slots]
                    state[mte2_index] = controller_after_two(
                        state[mte2_index], state[b_slot], b_load_s)
                    b_group_ready = state[mte2_index]
                    load_s += b_load_s
                if a_group_ready is None or b_group_ready is None:
                    raise AssertionError(
                        "CANN controller did not issue initial operand groups")
                l0a_slot = l0a_indexes[k_index % envelope.l0a_buffer_slots]
                l0b_slot = l0b_indexes[k_index % envelope.l0b_buffer_slots]
                a_mte1_s = (
                    padded_m * base_k * envelope.element_bytes
                    / l0a_capacity)
                b_mte1_s = (
                    padded_n * base_k * envelope.element_bytes
                    / l0b_capacity)
                if (grouped_matmul
                        and self.hw.aic_grouped_matmul_mte1_pair_completion_service
                        is not None
                        and padded_n == envelope.tile_n
                        and base_k == envelope.tile_k):
                    pair_key = (padded_m, padded_n, base_k)
                    if pair_key not in mte1_pair_completion_cache:
                        mte1_pair_completion_cache[pair_key] = (
                            self.hw.aic_grouped_matmul_mte1_pair_time_ns(
                                padded_m, padded_n, base_k, 1) * 1e-9)
                    pair_completion_s = mte1_pair_completion_cache[pair_key]
                    b_mte1_s = max(
                        b_mte1_s,
                        pair_completion_s - a_mte1_s,
                    )
                a_mte1_end = controller_after_three(
                    state[mte1_index], a_group_ready, state[l0a_slot],
                    a_mte1_s)
                b_mte1_end = controller_after_three(
                    a_mte1_end, b_group_ready, state[l0b_slot], b_mte1_s)
                state[mte1_index] = b_mte1_end
                base_compute_s = (
                    2.0 * padded_m * padded_n * base_k
                    / tensor_capacity)
                if k_index == 0 and not grouped_matmul:
                    # MM_CFG_NO_PRELOAD has one L0C output slot.  GetTensorC
                    # can overlap the next output's A1/B1 prefetch, but the
                    # next Cube program cannot reuse L0C until the preceding
                    # FixPipe store completes.
                    cube_end = controller_after_three(
                        state[cube_index], b_mte1_end, state[store_index],
                        base_compute_s)
                else:
                    cube_end = controller_after_two(
                        state[cube_index], b_mte1_end, base_compute_s)
                state[cube_index] = cube_end
                state[l0a_slot] = cube_end.copy()
                state[l0b_slot] = cube_end.copy()
                mte1_s += a_mte1_s + b_mte1_s
                compute_s += base_compute_s
                if ((k_index + 1) % envelope.a_step_k_tiles == 0
                        or k_index + 1 == k_tiles):
                    a_slot = a1_indexes[
                        a_group_index % envelope.a1_buffer_slots]
                    state[a_slot] = a_mte1_end
                if ((k_index + 1) % envelope.b_step_k_tiles == 0
                        or k_index + 1 == k_tiles):
                    b_slot = b1_indexes[
                        b_group_index % envelope.b1_buffer_slots]
                    state[b_slot] = b_mte1_end
                matmul_steps.append((
                    a_load_step_s,
                    b_load_step_s,
                    a_mte1_s,
                    b_mte1_s,
                    base_compute_s,
                    ((k_index + 1) % envelope.a_step_k_tiles == 0
                     or k_index + 1 == k_tiles),
                    ((k_index + 1) % envelope.b_step_k_tiles == 0
                     or k_index + 1 == k_tiles),
                ))
            output_bytes = float(
                logical_m * logical_n * envelope.element_bytes)
            store_s = max(
                output_bytes * envelope.worker_count / capacity
                for capacity in node_capacity[
                    envelope.output_path_name].values())
            state[store_index] = controller_after_two(
                state[cube_index], state[store_index], store_s)
            return (
                tuple(tuple(row) for row in state),
                (("source_prelude_control", prelude_s),
                 ("load", load_s), ("mte1", mte1_s),
                 ("compute", compute_s), ("store", store_s)),
                tuple(matmul_steps),
                store_s,
                a_load_service_s,
            )

        class_transitions = []
        class_durations = []
        class_services = []
        class_a_load_services = []
        class_cold_a_load_services = []
        for logical_m, logical_n in envelope.class_shapes:
            (transition, services, matmul_steps, store_s,
             a_load_service_s) = controller_transition(logical_m, logical_n)
            tail_control_s = (
                self.hw.aic_matmul_v2_dynamic_m_tail_control_time_ns()
                * 1e-9
                if persistent_matmul and logical_m % 16 else 0.0)
            if small_m_base_k_control_s:
                services = services + ((
                    "base_k_control", small_m_base_k_control_s * k_tiles),)
            if tail_control_s:
                services = services + ((
                    "dynamic_m_tail_control", tail_control_s * k_tiles),)
            class_transitions.append(transition)
            class_services.append(services)
            class_a_load_services.append(a_load_service_s)
            initial_state = (0.0,) * controller_state_size
            class_durations.append(
                _maxplus_apply(transition, initial_state)[store_index])
            if not grouped_matmul:
                # MM_CFG_NO_PRELOAD enables Cube/FixPipe unit flags.  The
                # source therefore retains A1/B1, L0A/L0B, Cube, and FixPipe
                # credits across successive output tiles even when dbL0C is
                # one.  Build the finite source program as a fixed-size
                # max-plus recurrence.  Buffer phases are part of the state
                # transition cycle, so odd K-group counts cannot reuse the
                # wrong ping-pong slot at an output boundary.
                mte2_index = 0
                mte1_index = 1
                cube_index = 2
                store_index = 3
                a_group_index_state = 4
                b_group_index_state = 5
                a_end_index = 6
                b_end_index = 7
                tail_scalar_index = 8
                base_scalar_index = 9
                a1_offset = 10
                b1_offset = a1_offset + envelope.a1_buffer_slots
                l0a_offset = b1_offset + envelope.b1_buffer_slots
                l0b_offset = l0a_offset + envelope.l0a_buffer_slots
                constant_index = l0b_offset + envelope.l0b_buffer_slots
                state_size = constant_index + 1

                def apply_event(state, target, sources, duration_s=0.0):
                    """Apply one sparse event directly to the transition rows.

                    Every event matrix is identity except for ``target``.
                    Composing it with the accumulated transition therefore
                    only replaces that one row; materializing and multiplying
                    a dense matrix for each source event is unnecessary.
                    """

                    if len(sources) == 1:
                        state[target] = [
                            value + duration_s
                            for value in state[sources[0]]]
                    elif len(sources) == 2:
                        first, second = (state[source] for source in sources)
                        state[target] = [
                            (first_value
                             if first_value >= second_value else second_value)
                            + duration_s
                            for first_value, second_value in zip(
                                first, second, strict=True)]
                    elif len(sources) == 3:
                        first, second, third = (
                            state[source] for source in sources)
                        state[target] = [
                            (first_value
                             if (second_value <= first_value
                                 and third_value <= first_value)
                             else second_value
                             if third_value <= second_value else third_value)
                            + duration_s
                            for first_value, second_value, third_value in zip(
                                first, second, third, strict=True)]
                    else:
                        raise ValueError(
                            "CANN MatMul event must have one to three sources")

                a_groups_per_tile = math.ceil(
                    k_tiles / envelope.a_step_k_tiles)
                b_groups_per_tile = math.ceil(
                    k_tiles / envelope.b_step_k_tiles)
                phase_count = math.lcm(
                    envelope.a1_buffer_slots // math.gcd(
                        envelope.a1_buffer_slots, a_groups_per_tile),
                    envelope.b1_buffer_slots // math.gcd(
                        envelope.b1_buffer_slots, b_groups_per_tile),
                    envelope.l0a_buffer_slots // math.gcd(
                        envelope.l0a_buffer_slots, k_tiles),
                    envelope.l0b_buffer_slots // math.gcd(
                        envelope.l0b_buffer_slots, k_tiles),
                )

                def phase_state_permutation(phase):
                    """Map phase-zero buffer state indices to one phase."""

                    indices = list(range(state_size))
                    for offset, slots, advance in (
                            (a1_offset, envelope.a1_buffer_slots,
                             a_groups_per_tile),
                            (b1_offset, envelope.b1_buffer_slots,
                             b_groups_per_tile),
                            (l0a_offset, envelope.l0a_buffer_slots,
                             k_tiles),
                            (l0b_offset, envelope.l0b_buffer_slots,
                             k_tiles)):
                        for slot in range(slots):
                            indices[offset + slot] = offset + (
                                (phase * advance + slot) % slots)
                    return tuple(indices)

                def relabel_phase_transition(transition, indices):
                    """Conjugate a phase-zero max-plus transition by slots."""

                    relabeled = [
                        [float("-inf")] * state_size
                        for _ in range(state_size)
                    ]
                    for target, row in enumerate(transition):
                        relabeled_target = relabeled[indices[target]]
                        for source, value in enumerate(row):
                            relabeled_target[indices[source]] = value
                    return tuple(tuple(row) for row in relabeled)

                def compile_phase_transition(
                        steps, phase, *, load_resident_a):
                    state = [list(row) for row in _maxplus_identity(state_size)]
                    a_group_index = -1
                    b_group_index = -1
                    for k_index, (
                            a_load_step_s, b_load_step_s,
                            a_mte1_s, b_mte1_s, base_compute_s,
                            release_a1, release_b1) in enumerate(steps):
                        if small_m_base_k_control_s:
                            # The MatmulImpl base-K controller advances on
                            # its own Scalar stream.  The corresponding A2
                            # request consumes that state below, while prior
                            # MTE2/MTE1/Cube/FixPipe work remains in flight.
                            apply_event(
                                state, base_scalar_index,
                                (base_scalar_index,),
                                small_m_base_k_control_s)
                        if tail_control_s:
                            # The dynamic partial-M branch also orders the
                            # next source issue after the current MTE2 group.
                            apply_event(
                                state, tail_scalar_index,
                                (tail_scalar_index, mte2_index),
                                tail_control_s)
                            apply_event(
                                state, mte2_index,
                                (mte2_index, tail_scalar_index))
                        if a_load_step_s is not None:
                            a_group_index += 1
                            a1_slot = (
                                phase * a_groups_per_tile + a_group_index
                            ) % envelope.a1_buffer_slots
                            if not persistent_full_a or load_resident_a:
                                apply_event(
                                    state,
                                    mte2_index,
                                    (mte2_index, a1_offset + a1_slot),
                                    a_load_step_s)
                                apply_event(
                                    state, a_group_index_state, (mte2_index,))
                        if b_load_step_s is not None:
                            b_group_index += 1
                            b1_slot = (
                                phase * b_groups_per_tile + b_group_index
                            ) % envelope.b1_buffer_slots
                            apply_event(
                                state,
                                mte2_index,
                                (mte2_index, b1_offset + b1_slot),
                                b_load_step_s)
                            apply_event(
                                state, b_group_index_state, (mte2_index,))
                        l0a_slot = (
                            phase * k_tiles + k_index
                        ) % envelope.l0a_buffer_slots
                        l0b_slot = (
                            phase * k_tiles + k_index
                        ) % envelope.l0b_buffer_slots
                        if tail_control_s or small_m_base_k_control_s:
                            apply_event(
                                state, a_end_index,
                                 (mte1_index, a_group_index_state,
                                 l0a_offset + l0a_slot))
                            if tail_control_s:
                                apply_event(
                                    state, a_end_index,
                                    (a_end_index, tail_scalar_index))
                            if small_m_base_k_control_s:
                                apply_event(
                                    state, a_end_index,
                                    (a_end_index, base_scalar_index),
                                    a_mte1_s)
                            else:
                                apply_event(
                                    state, a_end_index, (a_end_index,),
                                    a_mte1_s)
                        else:
                            apply_event(
                                state,
                                a_end_index,
                                (mte1_index, a_group_index_state,
                                 l0a_offset + l0a_slot),
                                a_mte1_s)
                        apply_event(state, mte1_index, (a_end_index,))
                        apply_event(
                            state,
                            b_end_index,
                            (mte1_index, b_group_index_state,
                             l0b_offset + l0b_slot),
                            b_mte1_s)
                        apply_event(state, mte1_index, (b_end_index,))
                        apply_event(
                            state,
                            cube_index, (cube_index, b_end_index),
                            base_compute_s)
                        apply_event(
                            state, l0a_offset + l0a_slot, (cube_index,))
                        apply_event(
                            state, l0b_offset + l0b_slot, (cube_index,))
                        if release_a1 and not persistent_full_a:
                            apply_event(
                                state, a1_offset + a1_slot, (a_end_index,))
                        if release_b1:
                            apply_event(
                                state, b1_offset + b1_slot, (b_end_index,))
                    apply_event(
                        state,
                        store_index, (store_index, cube_index), store_s)
                    return tuple(tuple(row) for row in state)

                phase_transitions = [
                    compile_phase_transition(
                        matmul_steps, 0, load_resident_a=False)]
                base_phase_transition, = phase_transitions
                phase_transitions.extend(
                    relabel_phase_transition(
                        base_phase_transition, phase_state_permutation(phase))
                    for phase in range(1, phase_count))
                (_, _, cold_matmul_steps, _,
                 cold_a_load_service_s) = controller_transition(
                     logical_m, logical_n, cold_first_a=True)
                cold_first_transition = compile_phase_transition(
                    cold_matmul_steps, 0, load_resident_a=True)
                class_cold_a_load_services.append(cold_a_load_service_s)
                class_matmul_programs.append((
                    tuple(phase_transitions), cold_first_transition,
                    state_size, store_index, constant_index,
                ))
        finite_stream_class_indices = tuple(
            index
            for index, (logical_m, logical_n) in enumerate(
                envelope.class_shapes)
            if (finite_stream_mechanism is not None
                and math.ceil(logical_m / 16) * 16
                == finite_stream_mechanism["a_rows"]
                and logical_n == envelope.tile_n))
        wide_tail_class_indices = tuple(
            index
            for index, (logical_m, logical_n) in enumerate(
                envelope.class_shapes)
            if (wide_tail_mechanism is not None
                and math.ceil(logical_m / 16) * 16
                in wide_tail_mechanism["active_m_values"]
                and logical_n == envelope.tile_n))
        output_tile_class_indices = tuple(
            index
            for index, (logical_m, logical_n) in enumerate(
                envelope.class_shapes)
            if (output_tile_mechanism is not None
                and math.ceil(logical_m / 16) * 16
                in output_tile_mechanism["active_m_values"]
                and logical_n == envelope.tile_n))
        finite_stream_floor_s = {}
        base_k_lifecycle_floor_s = {}
        step4_worker_sequence_s = {}
        controller_continuous_s = {}
        worker_makespan = {}
        finite_stream_critical_workers = set()
        base_k_lifecycle_critical_workers = set()
        controller_continuous_workers = set()
        transition_slice_cache = {}

        def class_transition_slice(class_index, start_phase, length):
            """Compose one exact finite source-program phase slice once."""

            cache_key = (class_index, start_phase, length)
            if cache_key not in transition_slice_cache:
                phase_transitions = class_matmul_programs[class_index][0]
                transition_slice_cache[cache_key] = (
                    compose_maxplus_transitions(
                        phase_transitions[start_phase:start_phase + length]))
            return transition_slice_cache[cache_key]

        grouped_controller_states = None
        if grouped_matmul and not step4_worker_sequence_eligible:
            initial_state = (0.0,) * controller_state_size
            worker_programs = tuple(
                tuple(
                    (class_transitions[class_index], repeat_count)
                    for class_index, repeat_count in worker_runs)
                for worker_runs in envelope.source_class_runs_by_worker)
            # The source's expert-ring distribution often gives several AICs
            # the same complete RLE program.  All controller states begin at
            # the same closed initial state, so equal programs have exactly
            # equal max-plus results.  Run each full recurrence once and
            # restore the physical worker order below; this does not collapse
            # any source work or alter its class/run identity.
            program_class_by_worker = []
            unique_programs = []
            program_class_by_identity = {}
            for program in worker_programs:
                if program in program_class_by_identity:
                    program_class = program_class_by_identity[program]
                else:
                    program_class = len(unique_programs)
                    program_class_by_identity[program] = program_class
                    unique_programs.append(program)
                program_class_by_worker.append(program_class)
            unique_states = apply_maxplus_transition_programs(
                tuple(unique_programs),
                (initial_state,) * len(unique_programs))
            grouped_controller_states = tuple(
                unique_states[program_class]
                for program_class in program_class_by_worker)
        for worker, counts in enumerate(envelope.counts_by_worker):
            worker_key = str(worker)
            duration = entry_metadata_prelude_s + control_s + math.fsum(
                count * class_durations[index]
                for index, count in enumerate(counts))
            if not grouped_matmul:
                phase_counts = {
                    len(program[0]) for program in class_matmul_programs}
                if len(phase_counts) != 1:
                    raise ValueError(
                        "CANN MatMul tile classes disagree on buffer phases")
                phase_count, = phase_counts
                (_, _, state_size, store_index, constant_index) = (
                    class_matmul_programs[0])
                initial_state = [0.0] * state_size
                initial_state[constant_index] = 0.0
                final_state = tuple(initial_state)
                worker_runs = envelope.source_class_runs_by_worker[worker]
                first_class_index, first_repeat_count = worker_runs[0]
                cold_first_transition = class_matmul_programs[
                    first_class_index][1]
                final_state = _maxplus_apply(
                    cold_first_transition, final_state)
                phase = 1 % phase_count
                remaining_runs = (
                    ((first_class_index, first_repeat_count - 1),)
                    + worker_runs[1:])
                for class_index, repeat_count in remaining_runs:
                    remaining = repeat_count
                    if not remaining:
                        continue
                    if phase:
                        prefix = min(remaining, phase_count - phase)
                        transition = class_transition_slice(
                            class_index, phase, prefix)
                        final_state = apply_repeated_maxplus_transition(
                            transition, 1, final_state)
                        phase = (phase + prefix) % phase_count
                        remaining -= prefix
                    full_cycles, tail = divmod(remaining, phase_count)
                    if full_cycles:
                        transition = class_transition_slice(
                            class_index, 0, phase_count)
                        final_state = apply_repeated_maxplus_transition(
                            transition, full_cycles, final_state)
                    if tail:
                        transition = class_transition_slice(
                            class_index, 0, tail)
                        final_state = apply_repeated_maxplus_transition(
                            transition, 1, final_state)
                        phase = tail
                duration = (
                    entry_metadata_prelude_s + control_s
                    + final_state[store_index])
                lifecycle_mechanism = (
                    self.hw.aic_matmul_v2_base_k_lifecycle_service)
                output_tile_count = sum(counts)
                partial_m_output_tiles = sum(
                    count
                    for count, (logical_m, _) in zip(
                        counts, envelope.class_shapes, strict=True)
                    if logical_m % 16)
                if (lifecycle_mechanism is not None
                        and self.hw.aic_matmul_v2_base_k_lifecycle_supports(
                            envelope.dtype,
                            envelope.worker_count,
                            envelope.tile_m,
                            envelope.tile_n,
                            envelope.tile_k,
                            envelope.projection_k,
                            output_tile_count,
                            envelope.a_step_k_tiles * envelope.tile_k,
                            envelope.b_step_k_tiles * envelope.tile_k,
                            envelope.a1_buffer_slots,
                            partial_m_output_tiles > 0)):
                    lifecycle_s = math.fsum((
                        self.hw.aic_matmul_v2_base_k_lifecycle_time_ns(
                            envelope.tile_k, output_tile_count, k_tiles),
                        partial_m_output_tiles * k_tiles
                        * self.hw.
                        aic_matmul_v2_dynamic_m_tail_control_time_ns(),
                    )) * 1e-9
                    base_k_lifecycle_floor_s[worker_key] = lifecycle_s
                    lifecycle_bound = (
                        entry_metadata_prelude_s + control_s + lifecycle_s)
                    if lifecycle_bound > duration:
                        duration = lifecycle_bound
                        base_k_lifecycle_critical_workers.add(worker_key)
                controller_continuous_s[worker_key] = final_state[store_index]
                controller_continuous_workers.add(worker_key)
            elif step4_worker_sequence_eligible:
                completion_s = (
                    self.hw.aic_grouped_matmul_step4_worker_sequence_time_ns(
                        step4_role, envelope.dtype, envelope.worker_count,
                        envelope.tile_m, 16, envelope.projection_k,
                        envelope.projection_n, envelope.tile_n,
                        envelope.tile_k,
                        envelope.a1_buffer_slots * envelope.a_step_k_tiles,
                        envelope.b1_buffer_slots * envelope.b_step_k_tiles,
                        envelope.a_step_k_tiles, envelope.b_step_k_tiles,
                        envelope.l0a_buffer_slots, envelope.l0b_buffer_slots,
                        envelope.l0c_buffer_slots,
                        step4_worker_counts[worker]) * 1e-9)
                step4_worker_sequence_s[worker_key] = completion_s
                duration = (
                    entry_metadata_prelude_s + control_s + completion_s)
            elif (output_tile_eligible or finite_stream_eligible
                    or wide_tail_eligible):
                if output_tile_eligible:
                    stream_class_indices = output_tile_class_indices
                elif wide_tail_eligible:
                    stream_class_indices = wide_tail_class_indices
                else:
                    stream_class_indices = finite_stream_class_indices
                output_tile_count = sum(
                    counts[index] for index in stream_class_indices)
                finite_stream_pair_count = (
                    output_tile_count * k_tiles)
                if output_tile_eligible:
                    stream_supported = (
                        output_tile_count > 0
                        and all(
                            counts[index] == 0
                            or self.hw.aic_grouped_matmul_k2880_output_tile_supports(
                                envelope.tile_m,
                                math.ceil(envelope.class_shapes[index][0] / 16)
                                * 16,
                                envelope.projection_k, envelope.tile_k,
                                envelope.tile_k, envelope.tile_n,
                                envelope.worker_count,
                                envelope.b_step_k_tiles, k_tiles)
                            for index in stream_class_indices))
                elif wide_tail_eligible:
                    stream_supported = (
                        self.hw.aic_grouped_matmul_k2880_wide_base_tail_supports(
                            wide_tail_service_base_m,
                            wide_tail_mechanism["active_m_values"][0],
                            envelope.projection_k, envelope.tile_k,
                            envelope.tile_k, envelope.tile_n,
                            envelope.worker_count, envelope.b_step_k_tiles,
                            finite_stream_pair_count)
                        if finite_stream_pair_count > 0 else False)
                else:
                    stream_supported = (
                        finite_stream_pair_count > 0
                        and finite_stream_supports(finite_stream_pair_count))
                if stream_supported:
                    if output_tile_eligible:
                        completion_ns = math.fsum(
                            counts[index]
                            * self.hw.aic_grouped_matmul_k2880_output_tile_time_ns(
                                envelope.tile_m,
                                math.ceil(envelope.class_shapes[index][0] / 16)
                                * 16,
                                envelope.projection_k, envelope.tile_k,
                                envelope.tile_k, envelope.tile_n,
                                envelope.worker_count,
                                envelope.b_step_k_tiles, k_tiles)
                            for index in stream_class_indices)
                    elif wide_tail_eligible:
                        completion_ns = (
                            self.hw.aic_grouped_matmul_k2880_wide_base_tail_time_ns(
                                wide_tail_service_base_m,
                                wide_tail_mechanism["active_m_values"][0],
                                envelope.projection_k, envelope.tile_k,
                                envelope.tile_k, envelope.tile_n,
                                envelope.worker_count,
                                envelope.b_step_k_tiles,
                                finite_stream_pair_count))
                    else:
                        completion_ns = finite_stream_time_ns(
                            finite_stream_mechanism["a_rows"], envelope.tile_k,
                            envelope.tile_k, envelope.tile_n,
                            envelope.worker_count, envelope.b_step_k_tiles,
                            finite_stream_pair_count)
                    completion_s = completion_ns * 1e-9
                    tail_analytical_s = math.fsum(
                        count * class_durations[index]
                        for index, count in enumerate(counts)
                        if index not in stream_class_indices)
                    finite_stream_floor_s[worker_key] = completion_s
                    completion_bound = math.fsum((
                        entry_metadata_prelude_s,
                        control_s,
                        completion_s,
                        tail_analytical_s,
                    ))
                    if completion_bound > duration:
                        duration = completion_bound
                        finite_stream_critical_workers.add(worker_key)
            if (grouped_matmul
                    and not step4_worker_sequence_eligible
                    and worker_key not in finite_stream_critical_workers):
                if grouped_controller_states is None:
                    raise AssertionError(
                        "CANN GroupedMatmul controller states are missing")
                controller_state = grouped_controller_states[worker]
                completion_s = controller_state[store_index]
                controller_continuous_s[worker_key] = completion_s
                duration = (
                    entry_metadata_prelude_s + control_s + completion_s)
                controller_continuous_workers.add(worker_key)
            worker_makespan[worker_key] = duration
        class_totals = tuple(
            sum(row[index] for row in envelope.counts_by_worker)
            for index in range(len(envelope.class_shapes)))
        stage_service = {
            "entry_metadata_materialization": (
                entry_metadata_prelude_s * envelope.worker_count),
            "source_prelude_control": 0.0,
            "base_k_control": 0.0,
            "dynamic_m_tail_control": 0.0,
            "load": 0.0,
            "mte1": 0.0,
            "compute": 0.0,
            "store": 0.0,
            control_stage_name: control_s * envelope.worker_count,
        }
        if finite_stream_floor_s:
            stage_service["finite_stream_completion_floor"] = math.fsum(
                finite_stream_floor_s.values())
        if base_k_lifecycle_floor_s:
            stage_service["base_k_control_lifecycle_floor"] = math.fsum(
                base_k_lifecycle_floor_s.values())
        if step4_worker_sequence_s:
            stage_service["step4_worker_sequence_completion"] = math.fsum(
                step4_worker_sequence_s.values())
        if controller_continuous_s:
            stage_service["source_order_controller_continuity"] = math.fsum(
                controller_continuous_s.values())
        if (not step4_worker_sequence_s
                and (controller_continuous_s or not finite_stream_floor_s)):
            for class_index, (services, total) in enumerate(zip(
                    class_services, class_totals, strict=True)):
                for name, duration in services:
                    if name == "load" and persistent_full_a:
                        duration -= class_a_load_services[class_index]
                    stage_service[name] += total * duration
            if persistent_full_a:
                # A full-K MatMulV2 source loads the one worker-local M slab
                # into its single A1 slot before the N-major output loop.  B1,
                # MTE1, Cube, and FixPipe still execute for every output tile;
                # only the diagnostic load sum needs the one cold A service
                # per worker instead of one A service per output tile.
                stage_service["load"] += math.fsum(
                    class_cold_a_load_services[worker_runs[0][0]]
                    for worker_runs in envelope.source_class_runs_by_worker)
        critical_class = max(
            range(len(class_durations)), key=lambda index: class_durations[index])
        critical_worker = max(
            worker_makespan, key=lambda worker: worker_makespan[worker])
        entry_metadata_path = (
            ("cann_grouped_matmul/entry_metadata_materialization",)
            if entry_metadata_prelude_s else ())
        control_path = (
            (f"cann_grouped_matmul/{control_stage_name}",)
            if control_s else ())
        finite_stream_path = (
            ("cann_grouped_matmul/finite_stream_completion_floor",)
            if critical_worker in finite_stream_critical_workers else ())
        if critical_worker in base_k_lifecycle_critical_workers:
            execution_path = (
                "cann_grouped_matmul/base_k_control_lifecycle_floor",)
        elif critical_worker in step4_worker_sequence_s:
            execution_path = (
                "cann_grouped_matmul/step4_worker_sequence_completion",)
        elif critical_worker in controller_continuous_workers:
            execution_path = (
                "cann_grouped_matmul/source_order_controller_continuity",)
        else:
            execution_path = (
                "cann_grouped_matmul/"
                f"m{envelope.class_shapes[critical_class][0]}_"
                f"n{envelope.class_shapes[critical_class][1]}",)
        return PipelineDiagnostics(
            execution_time_s=max(worker_makespan.values()),
            initiation_interval_s=0.0,
            worker_makespan_s=worker_makespan,
            stage_service_s=stage_service,
            critical_path=(
                entry_metadata_path + control_path + finite_stream_path
                + execution_path),
            tasks=(),
        )

    def _resolve_gpu_grouped_gemm_pipeline_envelope(
            self, envelope, active_units, launch_task_kind, *, launched_ctas):
        """Price grouped-GEMM fixed span and K rounds without CTA expansion."""

        if not isinstance(envelope, GpuGroupedGemmPipelineEnvelopeDemand):
            raise TypeError(
                "expected GpuGroupedGemmPipelineEnvelopeDemand")
        if launch_task_kind != "generic":
            raise ValueError(
                "GPU grouped-GEMM pipeline requires a generic launch")
        if envelope.fp8_block_program is None:
            mechanism = self.hw.grouped_gemm_tile_pipeline
            source_special_cases = True
        else:
            # FP8 block tile programs carry the same fixed-span, K-round and
            # empty-tail coefficients but no BF16 long-K or sparse-refill
            # special case; those remain bound to their own BF16 probes.
            family = self.hw.grouped_gemm_fp8_block_tile_pipelines
            if family is None or envelope.fp8_block_program not in family:
                raise ValueError(
                    "GPU grouped-GEMM FP8 block pipeline has no hardware "
                    f"mechanism for {envelope.fp8_block_program!r}")
            mechanism = family[envelope.fp8_block_program]
            source_special_cases = False
        if mechanism is None:
            raise ValueError(
                "GPU grouped-GEMM pipeline has no hardware mechanism")
        if self.hw.dram_latency_ns is None:
            raise ValueError(
                "GPU grouped-GEMM pipeline requires DRAM latency")
        resident_ctas = (
            self.hw.num_compute_units
            * envelope.resident_ctas_per_sm)
        if (isinstance(launched_ctas, bool)
                or not isinstance(launched_ctas, int)):
            raise TypeError("GPU grouped-GEMM launched_ctas must be an integer")
        if launched_ctas < envelope.active_ctas:
            raise ValueError(
                "GPU grouped-GEMM launch cannot contain fewer CTAs than its "
                "active pipeline")
        waves = math.ceil(envelope.active_ctas / resident_ctas)
        dram = self.hw.resource_by_role("dram")
        dram_capacity = dram.effective_capacity(
            active_units,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        )
        if dram_capacity <= 0.0:
            raise ValueError(
                "GPU grouped-GEMM pipeline has no DRAM capacity")
        dram_bytes_per_round_wave = (
            envelope.dram_read_bytes / envelope.k_tiles / waves)
        dram_round_s = math.fsum((
            dram_bytes_per_round_wave / dram_capacity,
            self.hw.dram_latency_ns * 1e-9 / mechanism["pipeline_depth"],
        ))
        local_round_s = mechanism["local_k_round_interval_ns"] * 1e-9
        long_k_pipeline = (
            source_special_cases
            and launched_ctas == mechanism["long_k_grid_ctas"]
            and envelope.active_ctas == launched_ctas
            and waves == 1
            and envelope.k_tiles >= mechanism["long_k_anchor_tiles"])
        if long_k_pipeline:
            round_s = mechanism["long_k_round_interval_ns"] * 1e-9
            anchor_s = mechanism["long_k_anchor_completion_ns"] * 1e-9
            fixed_span_s = anchor_s - (
                mechanism["long_k_anchor_tiles"] * round_s)
            active_wave_s = envelope.k_tiles * round_s
            critical_round = "long_k_pipeline"
        else:
            round_s = max(local_round_s, dram_round_s)
            fixed_span_s = mechanism["fixed_span_ns"] * 1e-9
            active_wave_s = envelope.k_tiles * round_s
            if local_round_s >= dram_round_s:
                critical_round = "local_issue"
            else:
                critical_round = "dram_service_plus_hidden_latency"
        empty_tail_s = 0.0
        if (envelope.active_ctas >= resident_ctas
                and launched_ctas > envelope.active_ctas):
            empty_tail_rounds = math.ceil(
                (launched_ctas - envelope.active_ctas)
                / self.hw.num_compute_units)
            empty_tail_s = (
                empty_tail_rounds
                * mechanism["empty_tail_round_ns"] * 1e-9)
        # A source-exact sparse-W1 probe brackets the 31-active/32-maximum
        # BM16 launch at adjacent N widths.  Relative to the same source with
        # one resident wave, its second wave pays the existing independently
        # measured pipeline fill once more.  Keep the rule within the probed
        # two-wave, K32 and sub-SM early-return-tail domain.
        source_tail_refill = (
            source_special_cases
            and envelope.active_ctas == mechanism["sparse_refill_active_ctas"]
            and launched_ctas == mechanism["sparse_refill_grid_ctas"]
            and envelope.k_tiles == mechanism["sparse_refill_k_tiles"])
        wave_fill_s = fixed_span_s * (
            mechanism["sparse_refill_wave_count"]
            if source_tail_refill else 1)
        later_wave_s = active_wave_s
        later_wave_refill_critical = False
        if not source_special_cases:
            # FP8 block tile rows: every resident wave after the first also
            # pays its independently measured L2-resident per-wave cost
            # (CTA refill plus full-residency K rounds) unless the wave's
            # DRAM round service is longer.
            local_wave_s = math.fsum((
                mechanism["wave_refill_ns"] * 1e-9,
                envelope.k_tiles
                * mechanism["wave_k_round_interval_ns"] * 1e-9,
            ))
            dram_wave_s = envelope.k_tiles * dram_round_s
            # Where the tile program's DRAM-streamed multi-wave service was
            # measured as well (within its probed K depth), a later wave also
            # costs at least that DRAM-fed refill plus its full-residency K
            # rounds.
            if ("dram_wave_refill_ns" in mechanism
                    and envelope.k_tiles <= mechanism["dram_wave_max_k_tiles"]):
                dram_fed_wave_s = math.fsum((
                    mechanism["dram_wave_refill_ns"] * 1e-9,
                    envelope.k_tiles
                    * mechanism["dram_wave_k_round_interval_ns"] * 1e-9,
                ))
            else:
                dram_fed_wave_s = 0.0
            later_wave_s = max(
                active_wave_s, local_wave_s, dram_wave_s, dram_fed_wave_s)
            later_wave_refill_critical = (
                waves > 1 and later_wave_s in (local_wave_s, dram_fed_wave_s)
                and later_wave_s > active_wave_s)
        remaining_active_s = (waves - 1) * later_wave_s
        steady_s = math.fsum((
            active_wave_s,
            max(remaining_active_s, empty_tail_s),
        ))
        execution_s = wave_fill_s + steady_s
        return PipelineDiagnostics(
            execution_time_s=execution_s,
            initiation_interval_s=round_s,
            worker_makespan_s={"resident_wave_envelope": execution_s},
            stage_service_s={
                "fixed_span": wave_fill_s,
                "local_issue": (
                    waves * envelope.k_tiles * local_round_s),
                "dram_service_plus_hidden_latency": (
                    waves * envelope.k_tiles * dram_round_s),
                "long_k_pipeline": (
                    active_wave_s if long_k_pipeline else 0.0),
                "empty_cta_tail": empty_tail_s,
            },
            critical_path=(
                "gpu_grouped_gemm/fixed_span",
                f"gpu_grouped_gemm/{critical_round}",
                *(
                    ("gpu_grouped_gemm/later_wave_refill",)
                    if later_wave_refill_critical else ()),
                *(
                    ("gpu_grouped_gemm/empty_cta_tail",)
                    if empty_tail_s > remaining_active_s else ()),
            ),
            tasks=(),
        )

    def _resolve_metax_grouped_operand_pipeline_envelope(
            self, envelope, active_units, launch_task_kind, *, launched_ctas):
        """Price the C500 serial per-CTA grouped-operand pipeline.

        One resident CTA per AP drains its expert operand stream: every
        useful CTA pays the independently measured fixed span once and the
        per-round service for each complete K tile; the per-round service is
        the larger of the measured local issue interval and the analytic
        DRAM round service (round bytes at the wave's fair-share capacity
        plus the pipeline-depth-amortized DRAM latency).  Empty grid CTAs
        retire after the source-visible early-return predicate and add the
        measured per-wave slot cost.  All coefficients come from the
        synthetic-routing mechanism reference; no production duration or
        case identity participates.
        """

        if not isinstance(
                envelope, MetaxGroupedOperandPipelineDemand):
            raise TypeError(
                "expected MetaxGroupedOperandPipelineDemand")
        if launch_task_kind != "generic":
            raise ValueError(
                "MetaX grouped operand pipeline requires a generic launch")
        mechanism = getattr(self.hw, envelope.mechanism_key)
        if mechanism is None:
            raise ValueError(
                "MetaX grouped operand pipeline has no hardware mechanism "
                f"for {envelope.mechanism_key}")
        if self.hw.dram_latency_ns is None:
            raise ValueError(
                "MetaX grouped operand pipeline requires DRAM latency")
        resident_ctas = (
            self.hw.num_compute_units * mechanism["resident_ctas_per_ap"])
        if (isinstance(launched_ctas, bool)
                or not isinstance(launched_ctas, int)):
            raise TypeError(
                "MetaX grouped operand pipeline launched_ctas must be an "
                "integer")
        if launched_ctas < envelope.active_ctas:
            raise ValueError(
                "MetaX grouped operand-pipeline launch cannot contain fewer "
                "CTAs than its active pipeline")
        waves = math.ceil(envelope.active_ctas / resident_ctas)
        dram = self.hw.resource_by_role("dram")
        dram_capacity = dram.effective_capacity(
            active_units,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        )
        if dram_capacity <= 0.0:
            raise ValueError(
                "MetaX grouped operand pipeline has no DRAM capacity")
        dram_bytes_per_round_wave = (
            envelope.dram_read_bytes / envelope.k_tiles / waves)
        dram_round_s = math.fsum((
            dram_bytes_per_round_wave / dram_capacity,
            self.hw.dram_latency_ns * 1e-9 / mechanism["pipeline_depth"],
        ))
        local_round_s = mechanism["local_k_round_interval_ns"] * 1e-9
        round_s = max(local_round_s, dram_round_s)
        fixed_span_s = mechanism["fixed_span_ns"] * 1e-9
        empty_tail_s = 0.0
        if launched_ctas > envelope.active_ctas:
            empty_waves = math.ceil(
                (launched_ctas - envelope.active_ctas)
                / self.hw.num_compute_units)
            empty_tail_s = (
                empty_waves * mechanism["empty_wave_slot_ns"] * 1e-9)
        steady_s = waves * envelope.k_tiles * round_s
        fill_s = waves * fixed_span_s
        execution_s = math.fsum((fill_s, steady_s, empty_tail_s))
        if local_round_s >= dram_round_s:
            critical_round = "local_issue"
        else:
            critical_round = "dram_service_plus_hidden_latency"
        return PipelineDiagnostics(
            execution_time_s=execution_s,
            initiation_interval_s=round_s,
            worker_makespan_s={
                "resident_cta_operand_pipeline": execution_s},
            stage_service_s={
                "fixed_span": fill_s,
                "local_issue": waves * envelope.k_tiles * local_round_s,
                "dram_service_plus_hidden_latency": (
                    waves * envelope.k_tiles * dram_round_s),
                "empty_cta_tail": empty_tail_s,
            },
            critical_path=(
                "metax_grouped_operand_pipeline/fixed_span",
                f"metax_grouped_operand_pipeline/{critical_round}",
                *(
                    ("metax_grouped_operand_pipeline/empty_cta_tail",)
                    if empty_tail_s > 0.0 else ()),
            ),
            tasks=(),
        )

    def _resolve_attention_pipeline_envelope(
            self, envelope, compute_time_by_role, memory_diagnostics,
            memory_service_time, memory_access_class_diagnostics,
            overridden_memory_services,
            t_per_resource):
        """Resolve a constant-size attention envelope into stage durations."""

        if not isinstance(envelope, AttentionPipelineEnvelopeDemand):
            raise TypeError(
                "expected AttentionPipelineEnvelopeDemand")
        service_time = {
            f"compute:{role}": duration
            for role, duration in compute_time_by_role.items()
        }
        memory_by_service = {}
        for diagnostic in memory_diagnostics:
            key = f"memory:{diagnostic.path_name}"
            if key not in memory_service_time:
                raise ValueError(
                    f"attention envelope memory service is missing {key!r}")
            service_time[key] = memory_service_time[key]
            memory_by_service[key] = diagnostic
        extra_memory_services = (
            set(memory_service_time) - set(memory_by_service))
        if extra_memory_services:
            raise ValueError(
                "attention envelope memory services contain unknown paths: "
                f"{sorted(extra_memory_services)}")
        typed_time_by_path = {}
        for diagnostic in memory_access_class_diagnostics:
            path_name = diagnostic.path_name
            if f"memory:{path_name}" not in memory_by_service:
                raise ValueError(
                    "attention access-class service has an unknown path: "
                    f"{path_name!r}")
            typed_time_by_path[path_name] = (
                typed_time_by_path[path_name] + diagnostic.service_time_s
                if path_name in typed_time_by_path
                else diagnostic.service_time_s)
        for path_name, typed_duration in typed_time_by_path.items():
            service_name = f"memory:{path_name}"
            service_time[service_name] = max(
                service_time[service_name], typed_duration)
        if not isinstance(overridden_memory_services, set):
            raise TypeError(
                "overridden_memory_services must be a set")
        unknown_overridden_services = (
            overridden_memory_services - set(memory_by_service))
        if unknown_overridden_services:
            raise ValueError(
                "overridden attention memory services are unknown: "
                f"{sorted(unknown_overridden_services)}")
        tail_latency_s_per_panel = 0.0
        tail_latency_s_per_group = 0.0
        if envelope.resident_tail_latency_service is not None:
            service = envelope.resident_tail_latency_service
            if service not in memory_by_service:
                raise ValueError(
                    "attention resident-tail latency references an unknown "
                    f"memory service: {service!r}")
            if service in overridden_memory_services:
                raise ValueError(
                    "attention resident-tail latency cannot use an Oracle "
                    f"memory override: {service!r}")
            memory = memory_by_service[service]
            if memory.first_request_latency_s is None:
                raise ValueError(
                    "attention resident-tail latency requires a resolved "
                    "first-request latency")
            tail_latency_s_per_panel = (
                envelope.resident_tail_requests_per_panel
                * memory.first_request_latency_s)
            tail_latency_s_per_group = (
                envelope.resident_tail_requests_per_group
                * memory.first_request_latency_s)
        for role, resource in self.hw._role_map.items():
            service_time[f"resource:{role}"] = t_per_resource[resource.name]
        for group in self.hw.resource_groups:
            service_time[f"group:{group.name}"] = math.fsum(
                t_per_resource[member] for member in group.members)

        phase_by_identity = {}
        multiplier_by_phase_identity = {}
        for work_class in envelope.classes:
            for phase in work_class.phases:
                identity = id(phase)
                phase_by_identity[identity] = phase
                multiplier = (
                    envelope.source_repeat_count
                    * work_class.count
                    * phase.repeat_count)
                multiplier_by_phase_identity[identity] = (
                    multiplier_by_phase_identity[identity] + multiplier
                    if identity in multiplier_by_phase_identity
                    else multiplier)
        stages = tuple(
            stage
            for phase in phase_by_identity.values()
            for stage in phase.stages)
        if any(
                stage.compute_service_demands
                or stage.compute_exact_request_demands
                or stage.memory_node_demands
                or stage.launch_shared_memory_node_demands
                for stage in stages):
            raise ValueError(
                "attention pipeline envelope requires fractional aggregate "
                "stage demands")
        absolute_services = {
            stage.service
            for stage in stages
            if stage.absolute_duration_s is not None
        }
        unknown_services = {
            service
            for stage in stages
            for service in (stage.service,) + stage.prior_resources
            if service not in service_time
            and service not in absolute_services
        }
        if unknown_services:
            raise ValueError(
                "attention envelope references unknown services: "
                f"{sorted(unknown_services)}")
        fraction_by_service = {}
        for identity, phase in phase_by_identity.items():
            multiplier = multiplier_by_phase_identity[identity]
            for stage in phase.stages:
                if stage.absolute_duration_s is not None:
                    continue
                contribution = stage.work_fraction * multiplier
                fraction_by_service[stage.service] = (
                    fraction_by_service[stage.service] + contribution
                    if stage.service in fraction_by_service
                    else contribution)
        invalid_fractions = {
            name: value for name, value in fraction_by_service.items()
            if not math.isclose(value, 1.0, rel_tol=1e-9, abs_tol=1e-12)
        }
        if invalid_fractions:
            raise ValueError(
                "attention envelope work fractions must sum to one per "
                f"service: {invalid_fractions}")

        resident_local_frontend = (
            self.hw.hardware_id in ("nvidia_a800", "nvidia_h200")
            and envelope.resident_tail_latency_service
            == "memory:global_read"
        )

        def _resolve_stage(stage):
            if stage.absolute_duration_s is not None:
                duration = stage.absolute_duration_s
            else:
                duration = (
                    service_time[stage.service]
                    * stage.work_fraction
                    * envelope.active_lane_count)
            if (stage.absolute_duration_s is None
                    and resident_local_frontend
                    and stage.service == "memory:global_read"):
                memory = memory_by_service[stage.service]
                if memory.initial_transactions is None:
                    raise ValueError(
                        "resident FA2 LDGSTS stage requires transactions")
                # The aggregate service bound above still uses the saturated
                # full-chip path.  Only this CTA-local dependency duration
                # uses the independently measured active-SM issue rate.
                local_rate = (
                    self.hw.ldgsts_initial_32b_transactions_per_active_sm_cycle
                    * self.hw.nominal_sm_freq_hz)
                duration = (
                    memory.initial_transactions * stage.work_fraction
                    / local_rate)
            if (stage.include_first_request_latency
                    and stage.service not in overridden_memory_services):
                memory = memory_by_service[stage.service]
                if (memory.rate_unit != "transactions"
                        or memory.initial_transactions is None
                        or memory.first_request_latency_s is None):
                    raise ValueError(
                        "attention envelope first-request latency requires a "
                        "transaction-resolved memory path")
                stage_transactions = (
                    memory.initial_transactions * stage.work_fraction)
                # Same burst reasoning as the work-plan stage above: the
                # request burst drains at the path's service capacity, not at
                # the launch-average rate this dependency itself produces.
                lane_rate = (
                    (max(memory.lambda_initial_per_s,
                         memory.lambda_capacity_per_s)
                     if stage.burst_drains_at_path_capacity
                     else memory.lambda_initial_per_s)
                    / envelope.active_lane_count)
                duration = (
                    memory.first_request_latency_s
                    + max(stage_transactions - 1.0, 0.0) / lane_rate)
            return PipelineStage._from_validated_demand(stage, duration)

        resolved_phase_by_structure = {}
        resolved_stages_by_identity = {}
        resolved_phases_by_identity = {}

        def _resolve_phase(phase):
            stages_identity = id(phase.stages)
            if stages_identity not in resolved_stages_by_identity:
                resolved_stages_by_identity[stages_identity] = tuple(
                    _resolve_stage(stage) for stage in phase.stages)
            structure = (
                phase.name, stages_identity, phase.repeat_count)
            if structure not in resolved_phase_by_structure:
                resolved_phase_by_structure[structure] = (
                    AttentionPipelinePhase._from_validated_demand(
                        phase,
                        resolved_stages_by_identity[stages_identity]))
            return resolved_phase_by_structure[structure]

        def _resolve_phases(phases):
            identity = id(phases)
            if identity not in resolved_phases_by_identity:
                resolved_phases_by_identity[identity] = tuple(
                    _resolve_phase(phase) for phase in phases)
            return resolved_phases_by_identity[identity]

        return AttentionPipelineEnvelope._from_validated_demand(
            envelope,
            tuple(
                AttentionPipelineClass._from_validated_demand(
                    work_class, _resolve_phases(work_class.phases))
                for work_class in envelope.classes),
            tail_latency_s_per_panel=tail_latency_s_per_panel,
            tail_latency_s_per_group=tail_latency_s_per_group,
        )

    def _resolve_fia_cross_domain_work_plan(
            self, work_plan, active_units, launch_task_kind,
            resource_time_overrides_ns, memory_diagnostics=()):
        """Resolve one source FIA DAG with an unpartitioned ring-span edge."""

        if not isinstance(work_plan, FiaCrossDomainWorkPlan):
            raise TypeError("expected FiaCrossDomainWorkPlan")
        if launch_task_kind != "mixed":
            raise ValueError("FIA cross-domain work plan requires mixed task")
        if active_units != work_plan.active_group_count:
            raise ValueError(
                "FIA cross-domain active units must equal source AIC groups")
        mechanism = self.hw.fia_cross_domain_three_handoff
        if mechanism is None or mechanism["model"] != work_plan.mechanism_id:
            raise ValueError(
                "FIA cross-domain work plan has no matching HW mechanism")
        execution_domain_units = self.hw.execution_units_for_task("mixed")
        tensor_resource = self.hw.resource_by_role("compute_tensor")
        vector_resource = self.hw.resource_by_role("compute_vector")
        aiv_mte2_resource = self.hw.resource_by_role("mte2")
        aiv_mte3_resource = self.hw.resource_by_role("mte3")
        if tensor_resource.total_units != execution_domain_units:
            raise ValueError(
                "FIA source program requires one Tensor unit per AIC")
        if vector_resource.total_units != 2 * execution_domain_units:
            raise ValueError(
                "FIA source program requires two Vector units per AIC")
        if (aiv_mte2_resource.total_units != vector_resource.total_units
                or aiv_mte3_resource.total_units
                != vector_resource.total_units):
            raise ValueError(
                "FIA AIV memory pipes must map one-to-one onto Vector units")
        mte2_read_diagnostics = tuple(
            diagnostic for diagnostic in memory_diagnostics
            if diagnostic.path_name == "mte2_read")
        if memory_diagnostics and len(mte2_read_diagnostics) != 1:
            raise ValueError(
                "FIA cold MTE2 completion requires exactly one mte2_read "
                "memory path")
        if mte2_read_diagnostics:
            mte2_read_nodes = {
                node.resource_role: node
                for node in mte2_read_diagnostics[0].nodes}
            if set(mte2_read_nodes) != {"l2_cache", "dram"}:
                raise ValueError(
                    "FIA cold MTE2 completion requires L2 and DRAM nodes")
            mte2_l2_bytes = mte2_read_nodes["l2_cache"].bytes
            mte2_dram_bytes = mte2_read_nodes["dram"].bytes
            if (mte2_l2_bytes <= 0.0 or mte2_dram_bytes < 0.0
                    or mte2_dram_bytes > mte2_l2_bytes):
                raise ValueError(
                    "FIA predicted MTE2 miss bytes are outside L2 traffic")
            mte2_dram_miss_fraction = mte2_dram_bytes / mte2_l2_bytes
        else:
            mte2_dram_miss_fraction = None
        mte2_descriptors = tuple(
            descriptor
            for group in work_plan.groups
            for task in group.tasks
            for stage in (task.mm1_mte2, task.mm2_mte2)
            for descriptor in stage.descriptors)
        # The independent cold panel validates the standard GQA source with
        # four Q rows per KV head.  The latency-bandwidth law prices the same
        # source chain for other multi-row GQA panels by descriptor bytes.
        # MHA's single-row chain remains outside this mechanism's domain.
        small_descriptor_rows = {
            descriptor.rows for descriptor in mte2_descriptors
            if (descriptor.rows <= 16
                and descriptor.source_row_stride_elements is not None)}

        def _gqa_group_fits_finite_stream(group):
            for stage_name in ("mm1_mte2", "mm2_mte2"):
                large_descriptors = tuple(
                    descriptor
                    for task in group.tasks
                    for descriptor in getattr(task, stage_name).descriptors
                    if descriptor.rows > 16)
                if any(
                        descriptor.rows > 128 or descriptor.columns != 128
                        for descriptor in large_descriptors):
                    return False
                descriptor_count = sum(
                    task.repeat_count * descriptor.count
                    for task in group.tasks
                    for descriptor in getattr(task, stage_name).descriptors
                    if descriptor.rows > 16)
                # The versioned finite-program validation keeps the shared
                # cold-stream lower bound stable through 160 drained D128
                # descriptors per worker.  Longer programs remain outside
                # this mechanism's measured domain.
                if descriptor_count > 160:
                    return False
            return True

        grouped_gqa_geometry = (
            len(small_descriptor_rows) == 1
            and 2 <= next(iter(small_descriptor_rows)) <= 16
        )
        grouped_gqa_cold_completion_eligible = (
            grouped_gqa_geometry
            and all(descriptor.source_row_stride_elements is not None
                    for descriptor in mte2_descriptors)
        )
        grouped_gqa_cross_task_finite_drain_eligible = (
            grouped_gqa_cold_completion_eligible
            and work_plan.active_group_count == 24)
        grouped_gqa_finite_stream_eligible = (
            grouped_gqa_geometry
            and any(descriptor.source_row_stride_elements is None
                    for descriptor in mte2_descriptors)
            and all(
                _gqa_group_fits_finite_stream(group)
                for group in work_plan.groups)
        )
        # CANN 8.5.1 regular FIA does not preload its AIV instruction cache.
        # A source-exact, workload-independent Vec1/Vec2 probe establishes a
        # first-Vec1 cold-fill lower bound for the statically covered
        # M4/M5/M8, exact S2 and D64/D128 program classes.  The cold fill
        # occurs once per resident AIV and is independent of later tasks.
        regular_large_descriptor_columns = {
            descriptor.columns for descriptor in mte2_descriptors
            if descriptor.rows > 16}
        regular_aiv_instruction_cold_fill_eligible = (
            grouped_gqa_geometry
            and small_descriptor_rows <= {4, 5, 8}
            and len(regular_large_descriptor_columns) == 1
            and all(
                task.aic_pipeline_kind == "serial_stages"
                for group in work_plan.groups
                for task in group.tasks))
        mla_multitask_dependency_startup_s = (
            max(
                mechanism["startup_local_ns"],
                work_plan.active_group_count
                * mechanism["startup_shared_ns_per_active_group"],
            ) * 1e-9
            if (any(count >= 2 for count in work_plan.task_counts)
                and all(
                    task.aic_pipeline_kind
                    == "fia_block_cube_nonquant_mla_multibuffer"
                    for group in work_plan.groups
                    for task in group.tasks))
            else 0.0)
        def aiv_instruction_cold_fill_s(group):
            first_task = group.tasks[0]
            source_s2 = sum(
                descriptor.columns * descriptor.count
                for descriptor in first_task.mm2_mte2.descriptors
                if descriptor.rows in small_descriptor_rows)
            if regular_aiv_instruction_cold_fill_eligible:
                key = (
                    work_plan.active_group_count,
                    source_s2,
                    next(iter(regular_large_descriptor_columns)),
                )
                duration_ns = (
                    self.hw.fia_regular_aiv_instruction_cold_fill_time_ns(
                        *key))
            else:
                duration_ns = None
            if duration_ns is not None:
                return duration_ns * 1e-9
            return 0.0
        mla_generic_q_rope_geometry = (64, 64, 64)
        mla_strided_completion_eligible = (
            mte2_dram_miss_fraction is not None
            and all(
                task.aic_pipeline_kind
                == "fia_block_cube_nonquant_mla_multibuffer"
                for group in work_plan.groups
                for task in group.tasks)
            and all(
                (
                    descriptor.rows,
                    descriptor.columns,
                    descriptor.source_row_stride_elements,
                ) == mla_generic_q_rope_geometry
                or (
                    descriptor.source_row_stride_elements is not None
                    and self.hw.aic_mte2_fia_mla_strided_supports(
                        descriptor.rows,
                        descriptor.columns,
                        descriptor.source_row_stride_elements,
                        work_plan.active_group_count))
                for descriptor in mte2_descriptors)
        )
        mha_finite_drain_geometries = {
            (1, 128, 128),
            (256, 128, 4096),
        }
        mha_resident_probability_geometries = {
            (1, 256, 512),
            (1, 256, 1024),
            (1, 256, 2048),
            (1, 256, 4096),
            (1, 256, 8192),
        }

        def _mha_group_fits_finite_drain(group):
            if len(group.tasks) != 1:
                return False
            task = group.tasks[0]
            for stage in (task.mm1_mte2, task.mm2_mte2):
                for descriptor in stage.descriptors:
                    geometry = (
                        descriptor.rows,
                        descriptor.columns,
                        descriptor.source_row_stride_elements,
                    )
                    total_count = task.repeat_count * descriptor.count
                    if geometry in mha_finite_drain_geometries:
                        if not self.hw.aic_mte2_fia_mha_finite_drain_supports(
                                *geometry, work_plan.active_group_count,
                                total_count):
                            return False
                    elif geometry not in mha_resident_probability_geometries:
                        return False
            return True

        mha_finite_drain_cold_completion_eligible = (
            small_descriptor_rows == {1}
            and all(
                _mha_group_fits_finite_drain(group)
                for group in work_plan.groups)
        )
        cold_completion_plan_eligible = (
            grouped_gqa_cold_completion_eligible)
        gqa_l0_buffer_eligible = grouped_gqa_cold_completion_eligible

        def stage_duration(stage):
            if stage.resource_role == "compute_tensor":
                capacity = tensor_resource.effective_capacity(
                    1,
                    sm_freq_hz=self.hw.nominal_sm_freq_hz,
                    dram_freq_hz=self.hw.nominal_dram_freq_hz,
                )
                if tensor_resource.scope == "shared":
                    capacity /= execution_domain_units
                elif tensor_resource.scope == "per_unit":
                    pass
                else:
                    raise ValueError(
                        "FIA Tensor resource has an unknown scope")
                if capacity <= 0:
                    raise ValueError("FIA Tensor stage has no capacity")
                return stage.physical_operations / capacity
            elif stage.resource_role == "compute_vector":
                duration = 0.0
                for service in stage.typed_services:
                    rate = vector_resource.service_rate_for_one_unit(
                        service.service)
                    if rate <= 0:
                        raise ValueError(
                            "FIA Vector typed service has no capacity")
                    duration += service.operations / rate
                    if service.serial_invocations:
                        duration += (
                            service.serial_invocations
                            * vector_resource.setup_time_for_service(
                                service.service))
                return duration
            else:
                raise ValueError(
                    f"unknown FIA stage resource: {stage.resource_role!r}")

        def _compute_mte2_descriptor_duration(
                descriptor, *, drain_each_descriptor,
                finite_repeat_count=None):
            source_stride = descriptor.source_row_stride_elements
            normalized_source_stride = (
                descriptor.columns
                if source_stride is None else source_stride)
            finite_geometry = (
                descriptor.rows,
                descriptor.columns,
                normalized_source_stride,
            )
            physical_descriptor_bytes = max(
                descriptor.bytes_per_descriptor,
                self.hw.aic_mte2_nd2nz_service["descriptor_bytes"][0],
            )
            use_mla_strided_completion = (
                mla_strided_completion_eligible
                and self.hw.aic_mte2_fia_mla_strided_supports(
                    descriptor.rows,
                    descriptor.columns,
                    normalized_source_stride,
                    work_plan.active_group_count))
            if use_mla_strided_completion:
                serial_ns = self.hw.aic_mte2_fia_mla_strided_time_ns(
                    descriptor.rows,
                    descriptor.columns,
                    normalized_source_stride,
                    work_plan.active_group_count,
                    mte2_dram_miss_fraction,
                    stream=False,
                )
            elif (finite_repeat_count is not None
                    and finite_geometry in mha_finite_drain_geometries
                    and self.hw.aic_mte2_fia_mha_finite_drain_supports(
                        descriptor.rows,
                        descriptor.columns,
                        normalized_source_stride,
                        work_plan.active_group_count,
                        finite_repeat_count * descriptor.count)):
                finite_ns = self.hw.aic_mte2_fia_mha_finite_drain_time_ns(
                    descriptor.rows,
                    descriptor.columns,
                    normalized_source_stride,
                    work_plan.active_group_count,
                    finite_repeat_count * descriptor.count)
                return finite_ns / finite_repeat_count * 1e-9
            else:
                use_cold_completion = (
                    drain_each_descriptor
                    and cold_completion_plan_eligible
                    and mte2_dram_miss_fraction is not None
                    and self.hw.aic_mte2_nd2nz_cold_completion_supports(
                        descriptor.rows,
                        descriptor.columns,
                        normalized_source_stride,
                        work_plan.active_group_count))
                if use_cold_completion:
                    # One descriptor's L2-resident and DRAM-missed lines share
                    # the same MTE2 stream, so the resident completion and the
                    # cold service owed on the missed fraction overlap instead
                    # of accumulating.  An independent 100%-cold drained probe
                    # at this exact geometry reproduces, at eight and sixteen
                    # AIC, the aggregate DRAM rate implied by the production
                    # launch's own missed bytes, which the additive blend
                    # cannot satisfy.  Compose the two constraints by max.
                    fully_cold_ns = (
                        self.hw.aic_mte2_nd2nz_cold_completion_time_ns(
                            descriptor.rows,
                            descriptor.columns,
                            normalized_source_stride,
                            work_plan.active_group_count,
                            1.0,
                        ))
                    # The geometry-matched strided service is this
                    # descriptor's resident completion; the contiguous entry
                    # of equal byte count describes a different shape and is
                    # not an alternative bound on it.
                    if (source_stride is not None
                            and source_stride != descriptor.columns
                            and descriptor.rows > 16):
                        resident_ns = (
                            self.hw.aic_mte2_nd2nz_strided_serial_time_ns(
                                descriptor.rows,
                                descriptor.columns,
                                source_stride,
                                work_plan.active_group_count,
                            ))
                    else:
                        resident_ns = (
                            self.hw.aic_mte2_nd2nz_serial_time_ns(
                                physical_descriptor_bytes,
                                work_plan.active_group_count,
                            ))
                    serial_ns = max(
                        resident_ns,
                        mte2_dram_miss_fraction * fully_cold_ns,
                    )
                else:
                    serial_ns = self.hw.aic_mte2_nd2nz_serial_time_ns(
                        physical_descriptor_bytes,
                        work_plan.active_group_count,
                    )
                    if (source_stride is not None
                            and source_stride != descriptor.columns
                            and descriptor.rows > 16):
                        serial_ns = max(
                            serial_ns,
                            self.hw.aic_mte2_nd2nz_strided_serial_time_ns(
                                descriptor.rows,
                                descriptor.columns,
                                source_stride,
                                work_plan.active_group_count,
                            ))
            # FiaBlockCubeNonQuantGqa closes every CopyQ/K/P/VToL1 call
            # with its source MTE2->MTE1 SetFlag/WaitFlag edge.  A repeated
            # geometry therefore consists of repeated serial completions,
            # not one fill followed by an undrained stream.  The pinned MLA
            # multibuffer source retains descriptors in flight.
            if (drain_each_descriptor
                    and source_stride is None
                    and descriptor.rows == 128
                    and descriptor.columns == 128):
                # Paged regular FIA splits each 256-row CopyK/CopyV helper
                # into two 128-row block-table transfers, then drains MTE2
                # once when the helper returns.  The flattened descriptor
                # run therefore contains one serial fill and one streamed
                # transfer per source helper, not two independent drains.
                serial_count = math.ceil(descriptor.count / 2)
                stream_count = descriptor.count - serial_count
                if (physical_descriptor_bytes
                        < self.hw.aic_mte2_nd2nz_service[
                            "stream_descriptor_bytes"][0]):
                    stream_ns = serial_ns
                else:
                    stream_ns = self.hw.aic_mte2_nd2nz_stream_time_ns(
                        physical_descriptor_bytes,
                        work_plan.active_group_count,
                    )
            elif drain_each_descriptor:
                serial_count = descriptor.count
                stream_count = 0
                stream_ns = 0.0
            else:
                serial_count = 1
                stream_count = descriptor.count - 1
                if use_mla_strided_completion:
                    stream_ns = self.hw.aic_mte2_fia_mla_strided_time_ns(
                        descriptor.rows,
                        descriptor.columns,
                        normalized_source_stride,
                        work_plan.active_group_count,
                        mte2_dram_miss_fraction,
                        stream=True,
                    )
                elif (physical_descriptor_bytes
                        < self.hw.aic_mte2_nd2nz_service[
                            "stream_descriptor_bytes"][0]):
                    stream_ns = serial_ns
                else:
                    stream_ns = self.hw.aic_mte2_nd2nz_stream_time_ns(
                        physical_descriptor_bytes,
                        work_plan.active_group_count,
                    )
                if (not use_mla_strided_completion
                        and source_stride is not None
                        and source_stride != descriptor.columns
                        and descriptor.rows > 16):
                    stream_ns = max(
                        stream_ns,
                        self.hw.aic_mte2_nd2nz_strided_stream_time_ns(
                            descriptor.rows,
                            descriptor.columns,
                            source_stride,
                            work_plan.active_group_count,
                        ))
            return (
                serial_count * serial_ns + stream_count * stream_ns) * 1e-9

        mte2_descriptor_duration_cache = {}

        def mte2_descriptor_duration(
                descriptor, *, drain_each_descriptor,
                finite_repeat_count=None):
            key = (
                descriptor, drain_each_descriptor, finite_repeat_count)
            if key not in mte2_descriptor_duration_cache:
                mte2_descriptor_duration_cache[key] = (
                    _compute_mte2_descriptor_duration(
                        descriptor,
                        drain_each_descriptor=drain_each_descriptor,
                        finite_repeat_count=finite_repeat_count))
            return mte2_descriptor_duration_cache[key]

        def gqa_finite_drain_program_duration(descriptor):
            """Return an exact GQA drained-program service, when published.

            The source-order resolver expands a regular FIA K/V program back
            into its literal descriptor events.  Its MTE2 service must remain
            the same finite, drained program used by the aggregate stage
            resolver; only the event placement differs.  Dividing this exact
            homogeneous program by its literal descriptor count preserves the
            source scheduler's per-copy completion edges without replacing it
            with an unconstrained stream model.
            """

            source_stride = descriptor.source_row_stride_elements
            if (not grouped_gqa_cold_completion_eligible
                    or mte2_dram_miss_fraction is None
                    or source_stride is None
                    or not self.hw.aic_mte2_fia_gqa_finite_drain_supports(
                        descriptor.rows,
                        descriptor.columns,
                        source_stride,
                        work_plan.active_group_count,
                        descriptor.count)):
                return None
            # A strided K/V descriptor has its own measured service; the
            # contiguous entry of the same byte count describes a different
            # descriptor shape, so it is not an alternative lower bound for
            # this one.
            if source_stride != descriptor.columns:
                serial_ns = self.hw.aic_mte2_nd2nz_strided_serial_time_ns(
                    descriptor.rows,
                    descriptor.columns,
                    source_stride,
                    work_plan.active_group_count,
                )
            else:
                serial_ns = self.hw.aic_mte2_nd2nz_serial_time_ns(
                    descriptor.bytes_per_descriptor,
                    work_plan.active_group_count,
                )
            resident_program_ns = descriptor.count * serial_ns
            cold_program_ns = self.hw.aic_mte2_fia_gqa_finite_drain_time_ns(
                descriptor.rows,
                descriptor.columns,
                source_stride,
                work_plan.active_group_count,
                descriptor.count,
            )
            # One descriptor's L2-resident and DRAM-missed lines travel the
            # same MTE2 stream, so the two constraints overlap rather than
            # accumulate: the program cannot beat its resident completion, nor
            # the cold service owed on the fraction that actually misses.
            # The independent probe confirms the composition -- its 100%-cold
            # aggregate DRAM rate at eight AIC reproduces the rate implied by
            # the production launch's own missed bytes.
            return max(
                resident_program_ns,
                mte2_dram_miss_fraction * cold_program_ns,
            ) * 1e-9

        def mte2_duration(
                stage, *, drain_each_descriptor, finite_repeat_count=None):
            resident_s = math.fsum(
                mte2_descriptor_duration(
                    descriptor,
                    drain_each_descriptor=drain_each_descriptor,
                    finite_repeat_count=finite_repeat_count)
                for descriptor in stage.descriptors)
            large_descriptors = tuple(
                descriptor for descriptor in stage.descriptors
                if descriptor.rows > 16)
            if (drain_each_descriptor
                    and grouped_gqa_cold_completion_eligible
                    and mte2_dram_miss_fraction is not None
                    and len(large_descriptors) == 1):
                descriptor = large_descriptors[0]
                source_stride = descriptor.source_row_stride_elements
                if source_stride is None:
                    raise RuntimeError(
                        "eligible FIA GQA finite drain has no source stride")
                repeat_count = (
                    1 if finite_repeat_count is None
                    else finite_repeat_count)
                total_descriptor_count = repeat_count * descriptor.count
                finite_program_s = gqa_finite_drain_program_duration(
                    dataclasses.replace(
                        descriptor, count=total_descriptor_count))
                if finite_program_s is not None:
                    modeled_s = mte2_descriptor_duration(
                        descriptor, drain_each_descriptor=True)
                    resident_s = math.fsum((
                        resident_s,
                        -modeled_s,
                        finite_program_s / repeat_count,
                    ))
            if (not drain_each_descriptor
                    or not grouped_gqa_finite_stream_eligible
                    or mte2_dram_miss_fraction is None):
                return resident_s
            cold_stream_ns = self.hw.aic_mte2_nd2nz_cold_stream_time_ns(
                stage.total_bytes
                * work_plan.active_group_count
                * mte2_dram_miss_fraction,
                work_plan.active_group_count,
            )
            return max(resident_s, cold_stream_ns * 1e-9)

        def mte1_duration(stage):
            l0a = self.hw.resource_by_role("l0a_buffer")
            l0b = self.hw.resource_by_role("l0b_buffer")
            l0a_capacity = l0a.effective_capacity(
                1,
                sm_freq_hz=self.hw.nominal_sm_freq_hz,
                dram_freq_hz=self.hw.nominal_dram_freq_hz,
            )
            l0b_capacity = l0b.effective_capacity(
                1,
                sm_freq_hz=self.hw.nominal_sm_freq_hz,
                dram_freq_hz=self.hw.nominal_dram_freq_hz,
            )
            if l0a_capacity <= 0 or l0b_capacity <= 0:
                raise ValueError("FIA MTE1 stage has no L0 service capacity")
            bandwidth_duration = (
                stage.l0a_bytes / l0a_capacity
                + stage.l0b_bytes / l0b_capacity
            )
            if stage.load_pair is None:
                return bandwidth_duration
            completion_duration = (
                self.hw.aic_mte1_load_pair_time_ns(
                    stage.load_pair.stage_kind,
                    stage.load_pair.head_dim,
                    stage.load_pair.count,
                ) * 1e-9
            )
            return max(bandwidth_duration, completion_duration)

        def aiv_memory_durations(stage):
            if stage is None:
                return 0.0, 0.0, 0.0
            capacities = []
            for resource in (aiv_mte2_resource, aiv_mte3_resource):
                aggregate_capacity = resource.effective_capacity(
                    resource.total_units,
                    sm_freq_hz=self.hw.nominal_sm_freq_hz,
                    dram_freq_hz=self.hw.nominal_dram_freq_hz,
                )
                capacities.append(
                    aggregate_capacity / resource.total_units)
            if min(capacities) <= 0:
                raise ValueError("FIA AIV memory stage has no pipe capacity")
            read_s = stage.mte2_bytes / capacities[0]
            write_s = stage.mte3_bytes / capacities[1]
            bandwidth_s = read_s + write_s
            if stage.completion_service is None:
                return read_s, write_s, bandwidth_s
            completion_s = (
                self.hw.fia_aiv_workspace_transfer_time_ns(
                    stage.completion_service,
                    work_plan.active_group_count,
                ) * 1e-9)
            return read_s, write_s, max(bandwidth_s, completion_s)

        def fixpipe_duration(stage):
            return math.fsum(
                descriptor.count
                * self.hw.aic_fixpipe_full_tile_time_ns(
                    descriptor.rows,
                    descriptor.columns,
                    descriptor.destination_stride_elements,
                )
                * 1e-9
                for descriptor in stage.descriptors
            )

        def block_reduce_sum_duration(task):
            return math.fsum(
                descriptor.count
                * self.hw.fia_block_reduce_sum_time_ns(
                    descriptor.row_chunk_round,
                    descriptor.column_count,
                    descriptor.column_alignment,
                )
                * 1e-9
                for descriptor in task.vec1_block_reduce_sum
            )

        def block_reduce_max_duration(descriptors):
            return math.fsum(
                descriptor.count
                * self.hw.fia_block_reduce_max_time_ns(
                    descriptor.row_chunk_round,
                    descriptor.column_count,
                    descriptor.column_alignment,
                )
                * 1e-9
                for descriptor in descriptors
            )

        def brcb_duration(descriptors):
            return math.fsum(
                descriptor.count
                * self.hw.fia_brcb_time_ns(descriptor.row_chunk_round)
                * 1e-9
                for descriptor in descriptors
            )

        def buffered_stage_span(stage_durations, item_count):
            if (isinstance(item_count, bool)
                    or not isinstance(item_count, int)
                    or item_count <= 0):
                raise ValueError(
                    "FIA buffered stage item count must be positive")
            per_item = tuple(
                duration / item_count for duration in stage_durations)
            return (
                math.fsum(per_item)
                + (item_count - 1) * max(per_item))

        ring_span_s = self.hw.fia_cross_domain_three_handoff_time_ns(
            work_plan.task_counts) * 1e-9
        if work_plan.tiling_data_prelude is None:
            aic_tiling_s = 0.0
            aiv_tiling_s = 0.0
        else:
            aic_tiling_ns, aiv_tiling_ns = (
                self.hw.fia_tiling_data_materialization_times_ns(
                    work_plan.tiling_data_prelude.active_aic_groups,
                    work_plan.tiling_data_prelude.payload_bytes,
                    work_plan.tiling_data_prelude.aiv_transfer_bytes,
                ))
            aic_tiling_s = aic_tiling_ns * 1e-9
            aiv_tiling_s = aiv_tiling_ns * 1e-9
        initial_sync_all_s = self.hw.fia_aiv_sync_all_time_ns(
            2 * work_plan.active_group_count) * 1e-9
        initialization_durations = tuple(
            stage_duration(stage)
            for stage in work_plan.aiv_initialization_stages)
        initial_vector_s = (
            max(initialization_durations)
            if initialization_durations else 0.0)
        initial_aic_ready_s = aic_tiling_s
        initial_aiv_ready_s = (
            aiv_tiling_s + initial_vector_s + initial_sync_all_s)
        ring_ready_s = max(initial_aic_ready_s, initial_aiv_ready_s)
        stage_service_s = {
            "compute_tensor": 0.0,
            "compute_vector": math.fsum(initialization_durations),
            "aic_mte2_nd2nz": 0.0,
            "aic_mte1_l1_to_l0": 0.0,
            "aic_fixpipe_l0c_to_gm": 0.0,
            "aiv_block_reduce_sum": 0.0,
            "aiv_block_reduce_max": 0.0,
            "aiv_brcb": 0.0,
            "aiv_queue_control": 0.0,
            "aiv_mte2_gm_to_ub": 0.0,
            "aiv_mte3_ub_to_gm": 0.0,
            "fia_aic_tiling_data_materialization": aic_tiling_s,
            "fia_aiv_tiling_data_materialization": aiv_tiling_s,
            "fia_aiv_initialization": math.fsum(
                initialization_durations),
            "fia_output_init_syncall": initial_sync_all_s,
            "fia_cross_domain_program_span": ring_span_s,
            "fia_regular_aiv_instruction_cold_fill": 0.0,
            "fia_paged_kv_block_table_lookup": 0.0,
            "fia_mla_multitask_dependency_startup": (
                mla_multitask_dependency_startup_s),
        }
        worker_makespan_s = {}
        task_diagnostics = []
        group_completion = []
        compute_critical_group = None
        compute_critical_time = -1.0
        task_duration_cache = {}

        def task_durations(task):
            # FIA task runs are shallow wrappers around source task-class
            # templates.  Identity keys preserve exact duration semantics and
            # avoid repeatedly hashing descriptor trees merely to recognize a
            # repeated source program.
            cache_key = (
                id(task.mm1_mte2.descriptors),
                id(task.mm1_mte1),
                task.mm1.physical_operations,
                id(task.mm1_fixpipe.descriptors),
                id(task.vec1_memory),
                id(task.vec1.typed_services),
                id(task.vec1_block_reduce_sum),
                id(None if task.vec1_update is None
                   else task.vec1_update.typed_services),
                id(task.vec1_update_memory),
                id(None if task.vec1_peer is None
                   else task.vec1_peer.typed_services),
                id(task.vec1_peer_memory),
                id(task.vec1_peer_block_reduce_sum),
                id(None if task.vec1_peer_update is None
                   else task.vec1_peer_update.typed_services),
                id(task.vec1_peer_update_memory),
                id(task.mm2_mte2.descriptors),
                id(task.mm2_mte1),
                task.mm2.physical_operations,
                id(task.mm2_fixpipe.descriptors),
                id(task.gqa_source_order),
                id(task.mla_source_order),
                id(task.vec2_memory),
                id(task.vec2.typed_services),
                id(None if task.vec2_peer is None
                   else task.vec2_peer.typed_services),
                id(task.vec2_peer_memory),
                task.aic_pipeline_kind,
                (task.repeat_count
                 if (mha_finite_drain_cold_completion_eligible
                     or grouped_gqa_cross_task_finite_drain_eligible)
                 else None),
            )
            if cache_key in task_duration_cache:
                return task_duration_cache[cache_key]
            drain_each_mte2_descriptor = (
                task.aic_pipeline_kind == "serial_stages")
            # Regular FIA owns two source queue lifetimes in each Vec1/Vec2
            # lane.  The MLA multibuffer source does not use those queues.
            if task.aic_pipeline_kind == "serial_stages":
                queue_control_cycle_s = self.hw.resource_by_role(
                    "scalar_aiv").exact_request_service_time_s(
                        "aiv_queue_control_cycle", 1)
            elif (task.aic_pipeline_kind
                  == "fia_block_cube_nonquant_mla_multibuffer"):
                queue_control_cycle_s = 0.0
            else:
                raise ValueError("unknown FIA AIC pipeline kind")
            finite_repeat_count = (
                task.repeat_count
                if (mha_finite_drain_cold_completion_eligible
                    or grouped_gqa_cross_task_finite_drain_eligible)
                else None)
            mm1_mte2 = mte2_duration(
                task.mm1_mte2,
                drain_each_descriptor=drain_each_mte2_descriptor,
                finite_repeat_count=finite_repeat_count)
            mm1_mte1 = mte1_duration(task.mm1_mte1)
            mm1_compute = stage_duration(task.mm1)
            mm1_fixpipe = fixpipe_duration(task.mm1_fixpipe)
            reduction = block_reduce_sum_duration(task)
            reduction_max = block_reduce_max_duration(
                task.vec1_block_reduce_sum)
            brcb = brcb_duration(task.vec1_block_reduce_sum)
            (vec1_primary_read, vec1_primary_write,
             vec1_primary_memory) = (
                aiv_memory_durations(task.vec1_memory))
            vec1_primary = (
                vec1_primary_memory + stage_duration(task.vec1)
                + reduction + reduction_max + brcb
                + 2 * queue_control_cycle_s)
            (update_primary_read, update_primary_write,
             update_primary_memory) = (
                aiv_memory_durations(task.vec1_update_memory))
            update_primary = (
                0.0 if task.vec1_update is None
                else update_primary_memory
                + stage_duration(task.vec1_update))
            if task.vec1_peer is None:
                peer_reduction = 0.0
                peer_reduction_max = 0.0
                peer_brcb = 0.0
                vec1 = vec1_primary
                update_peer = 0.0
                vec2_peer = 0.0
                peer_memory_durations = (0.0, 0.0, 0.0, 0.0)
            else:
                peer_reduction = math.fsum(
                    descriptor.count
                    * self.hw.fia_block_reduce_sum_time_ns(
                        descriptor.row_chunk_round,
                        descriptor.column_count,
                        descriptor.column_alignment,
                    )
                    * 1e-9
                    for descriptor in task.vec1_peer_block_reduce_sum
                )
                peer_reduction_max = block_reduce_max_duration(
                    task.vec1_peer_block_reduce_sum)
                peer_brcb = brcb_duration(
                    task.vec1_peer_block_reduce_sum)
                (vec1_peer_read, vec1_peer_write,
                 vec1_peer_memory) = (
                    aiv_memory_durations(task.vec1_peer_memory))
                vec1 = max(
                    vec1_primary,
                    vec1_peer_memory + stage_duration(task.vec1_peer)
                    + peer_reduction + peer_reduction_max + peer_brcb
                    + 2 * queue_control_cycle_s)
                (update_peer_read, update_peer_write,
                 update_peer_memory) = (
                    aiv_memory_durations(task.vec1_peer_update_memory))
                update_peer = (
                    0.0 if task.vec1_peer_update is None
                    else update_peer_memory
                    + stage_duration(task.vec1_peer_update))
                (vec2_peer_read, vec2_peer_write,
                 vec2_peer_memory) = (
                    aiv_memory_durations(task.vec2_peer_memory))
                vec2_peer = (
                    vec2_peer_memory + stage_duration(task.vec2_peer)
                    + 2 * queue_control_cycle_s)
                peer_memory_durations = (
                    vec1_peer_read + update_peer_read + vec2_peer_read,
                    vec1_peer_write + update_peer_write + vec2_peer_write,
                    update_peer_read,
                    update_peer_write,
                )
            vec1_update = max(update_primary, update_peer)
            mm2_mte2 = mte2_duration(
                task.mm2_mte2,
                drain_each_descriptor=drain_each_mte2_descriptor,
                finite_repeat_count=finite_repeat_count)
            mm2_mte1 = mte1_duration(task.mm2_mte1)
            mm2_compute = stage_duration(task.mm2)
            mm2_fixpipe = fixpipe_duration(task.mm2_fixpipe)
            # Keep the service ledger separate from a source schedule's
            # terminal Fixpipe edge.  The latter can overlap earlier output
            # columns, while the former must continue to account for every
            # typed Fixpipe call.
            mm1_fixpipe_service_s = mm1_fixpipe
            mm2_fixpipe_service_s = mm2_fixpipe
            mla_source_schedule = None
            mla_outer_source_profile = None
            gqa_outer_source_profile = None
            (vec2_primary_read, vec2_primary_write,
             vec2_primary_memory) = (
                aiv_memory_durations(task.vec2_memory))
            vec2_primary = (
                vec2_primary_memory + stage_duration(task.vec2)
                + 2 * queue_control_cycle_s)
            vec2 = max(vec2_primary, vec2_peer)
            mm1_stage = (
                mm1_mte2 + mm1_mte1 + mm1_compute + mm1_fixpipe)
            mm2_pre_stage = mm2_mte2 + mm2_mte1 + mm2_compute

            # The CANN 8.5.1 MLA program is not an aggregate MM1/MM2
            # pipeline.  ``FiaBlockCubeNonQuantMla`` issues, for every
            # source-S2=512 task, four literal K pairs
            # ``(256, 32, 256, 32)`` and two K-L1 halves per pair.  Each
            # half publishes one MTE2->MTE1 edge; its three KV L1 slots are
            # released only after the matching MTE1 consumer has issued.
            # MM2 follows the same two-K-half order, with its two P panels
            # resident across the four output-D panels.  Preserve that
            # source topology rather than summing heterogeneous descriptor
            # completions into an artificial serial stage.
            #
            # This is selected by the complete literal descriptor program,
            # not by a workload shape or a kernel name.  Every descriptor
            # service remains the existing source-bound, versioned MLA MTE2
            # service; the change is only the static dependency graph.
            mla_source_order = task.mla_source_order
            mla_mm1_descriptors = (
                () if mla_source_order is None
                else mla_source_order.mm1_descriptors)
            mla_mm2_descriptors = (
                () if mla_source_order is None
                else mla_source_order.mm2_descriptors)
            mla_mm1_fixpipe_descriptors = task.mm1_fixpipe.descriptors
            mla_mm2_fixpipe_descriptors = task.mm2_fixpipe.descriptors
            mla_query_prefix_count = (
                3 if len(mla_mm1_descriptors) == 19 else 0)
            mla_query_descriptors = mla_mm1_descriptors[
                :mla_query_prefix_count]
            mla_k_descriptors = mla_mm1_descriptors[
                mla_query_prefix_count:]
            mla_source_program_eligible = (
                task.aic_pipeline_kind
                == "fia_block_cube_nonquant_mla_multibuffer"
                and mla_source_order is not None
                and work_plan.source_s2_base_size == 512
                and mla_strided_completion_eligible
                and mte2_dram_miss_fraction is not None
                and len(mla_k_descriptors) == 16
                and all(descriptor.count == 1
                        for descriptor in mla_mm1_descriptors)
                and (
                    not mla_query_descriptors
                    or (
                        len(mla_query_descriptors) == 3
                        and all(
                            descriptor.rows == 64
                            for descriptor in mla_query_descriptors)
                        and tuple(
                            (descriptor.columns,
                             descriptor.source_row_stride_elements)
                            for descriptor in mla_query_descriptors)
                        == ((256, 512), (64, 64), (256, 512))
                    ))
                and all(
                    descriptor.rows == 128
                    and descriptor.count == 1
                    for descriptor in mla_k_descriptors)
                and tuple(
                    (descriptor.columns, descriptor.source_row_stride_elements)
                    for descriptor in mla_k_descriptors)
                == ((256, 512), (32, 64), (32, 64), (256, 512)) * 4
                and len(mla_mm2_descriptors) == 18
                and all(descriptor.count == 1
                        for descriptor in mla_mm2_descriptors)
                and tuple(
                    (descriptor.rows, descriptor.columns,
                     descriptor.source_row_stride_elements)
                    for descriptor in mla_mm2_descriptors)
                == (
                    ((128, 128, 512), (128, 128, 512), (64, 256, 512))
                    * 2
                    + ((128, 128, 512),) * 12
                )
                and len(mla_mm1_fixpipe_descriptors) == 1
                and len(mla_mm2_fixpipe_descriptors) == 1
                and (
                    mla_mm1_fixpipe_descriptors[0].rows,
                    mla_mm1_fixpipe_descriptors[0].columns,
                    mla_mm1_fixpipe_descriptors[0].count,
                ) == (64, 128, 4)
                and (
                    mla_mm2_fixpipe_descriptors[0].rows,
                    mla_mm2_fixpipe_descriptors[0].columns,
                    mla_mm2_fixpipe_descriptors[0].count,
                ) == (64, 128, 4)
            )
            if mla_source_program_eligible:
                def mla_descriptor_time(descriptor, *, stream):
                    stride = descriptor.source_row_stride_elements
                    if stride is None:
                        raise AssertionError(
                            "literal MLA source descriptor lost its stride")
                    if not self.hw.aic_mte2_fia_mla_strided_supports(
                            descriptor.rows,
                            descriptor.columns,
                            stride,
                            work_plan.active_group_count):
                        if (descriptor.rows, descriptor.columns, stride) != (
                                64, 64, 64):
                            raise AssertionError(
                                "literal MLA source descriptor is outside "
                                "its published service domain")
                        # The Q/RoPE descriptor is the one statically known
                        # MLA geometry without a dedicated stream record.
                        # Keep its existing one-descriptor completion rather
                        # than borrowing a K/RoPE service for it.
                        return mte2_descriptor_duration(
                            descriptor, drain_each_descriptor=False)
                    return self.hw.aic_mte2_fia_mla_strided_time_ns(
                        descriptor.rows,
                        descriptor.columns,
                        stride,
                        work_plan.active_group_count,
                        mte2_dram_miss_fraction,
                        stream=stream,
                    ) * 1e-9

                def mla_source_burst(descriptors):
                    if not descriptors:
                        raise ValueError(
                            "literal MLA source burst must not be empty")
                    return math.fsum(
                        mla_descriptor_time(descriptor, stream=index > 0)
                        for index, descriptor in enumerate(descriptors))

                mm1_mte1_per_k_s = mm1_mte1 / 8
                mm1_tensor_per_k_s = mm1_compute / 8
                mm1_fixpipe_per_n_s = mm1_fixpipe / 4
                mm1_pair_services = tuple(
                    mla_source_burst(mla_k_descriptors[
                        2 * pair_ordinal:2 * pair_ordinal + 2])
                    for pair_ordinal in range(8))
                mm1_query_services = (
                    () if not mla_query_descriptors else (
                        mla_source_burst(mla_query_descriptors[:2]),
                        mla_source_burst(mla_query_descriptors[2:]),
                    ))
                mte2_available_s = 0.0
                mte1_available_s = 0.0
                tensor_available_s = 0.0
                fixpipe_available_s = 0.0
                mm1_mte2_service_s = 0.0
                kv_l1_ready_s = [0.0, 0.0, 0.0]
                query_ready_s = [0.0, 0.0]
                for n_ordinal in range(4):
                    for k_ordinal in range(2):
                        pair_ordinal = 2 * n_ordinal + k_ordinal
                        kv_slot = pair_ordinal % len(kv_l1_ready_s)
                        pair_service_s = mm1_pair_services[pair_ordinal]
                        mm1_mte2_service_s += pair_service_s
                        mte2_available_s = max(
                            mte2_available_s, kv_l1_ready_s[kv_slot],
                        ) + pair_service_s
                        key_ready_s = mte2_available_s
                        if n_ordinal == 0 and mla_query_descriptors:
                            query_service_s = mm1_query_services[k_ordinal]
                            mm1_mte2_service_s += query_service_s
                            mte2_available_s += query_service_s
                            query_ready_s[k_ordinal] = mte2_available_s
                        mte1_available_s = max(
                            mte1_available_s,
                            key_ready_s,
                            query_ready_s[k_ordinal],
                        ) + mm1_mte1_per_k_s
                        tensor_available_s = max(
                            tensor_available_s, mte1_available_s,
                        ) + mm1_tensor_per_k_s
                        kv_l1_ready_s[kv_slot] = mte1_available_s
                        if k_ordinal == 1:
                            fixpipe_available_s = max(
                                fixpipe_available_s, tensor_available_s,
                            ) + mm1_fixpipe_per_n_s
                mm1_source_span_s = max(
                    mte2_available_s,
                    mte1_available_s,
                    tensor_available_s,
                    fixpipe_available_s,
                )
                mm1_source_mte2_span_s = mte2_available_s

                probability_descriptors = (
                    mla_mm2_descriptors[2], mla_mm2_descriptors[5])
                value_descriptors = tuple(
                    descriptor
                    for ordinal, descriptor in enumerate(mla_mm2_descriptors)
                    if ordinal not in (2, 5))
                mm2_mte1_per_k_s = mm2_mte1 / 8
                mm2_tensor_per_k_s = mm2_compute / 8
                mm2_fixpipe_per_n_s = mm2_fixpipe / 4
                mm2_value_pair_services = tuple(
                    mla_source_burst(value_descriptors[
                        2 * pair_ordinal:2 * pair_ordinal + 2])
                    for pair_ordinal in range(8))
                mm2_probability_services = tuple(
                    mla_source_burst((descriptor,))
                    for descriptor in probability_descriptors)
                mte2_available_s = 0.0
                mte1_available_s = 0.0
                tensor_available_s = 0.0
                fixpipe_available_s = 0.0
                mm2_mte2_service_s = 0.0
                kv_l1_ready_s = [0.0, 0.0, 0.0]
                probability_ready_s = [0.0, 0.0]
                for n_ordinal in range(4):
                    for k_ordinal in range(2):
                        pair_ordinal = 2 * n_ordinal + k_ordinal
                        kv_slot = pair_ordinal % len(kv_l1_ready_s)
                        value_service_s = mm2_value_pair_services[
                            pair_ordinal]
                        mm2_mte2_service_s += value_service_s
                        mte2_available_s = max(
                            mte2_available_s, kv_l1_ready_s[kv_slot],
                        ) + value_service_s
                        value_ready_s = mte2_available_s
                        if n_ordinal == 0:
                            probability_service_s = (
                                mm2_probability_services[k_ordinal])
                            mm2_mte2_service_s += probability_service_s
                            mte2_available_s += probability_service_s
                            probability_ready_s[k_ordinal] = mte2_available_s
                        mte1_available_s = max(
                            mte1_available_s,
                            value_ready_s,
                            probability_ready_s[k_ordinal],
                        ) + mm2_mte1_per_k_s
                        tensor_available_s = max(
                            tensor_available_s, mte1_available_s,
                        ) + mm2_tensor_per_k_s
                        kv_l1_ready_s[kv_slot] = mte1_available_s
                        if k_ordinal == 1:
                            fixpipe_available_s = max(
                                fixpipe_available_s, tensor_available_s,
                            ) + mm2_fixpipe_per_n_s
                mm2_source_span_s = max(
                    mte2_available_s,
                    mte1_available_s,
                    tensor_available_s,
                    fixpipe_available_s,
                )
                mm2_source_mte2_span_s = mte2_available_s
                mm1_mte2 = mm1_mte2_service_s
                mm2_mte2 = mm2_mte2_service_s
                mm1_stage = mm1_source_span_s
                mm2_pre_stage = (
                    mm2_source_span_s - mm2_fixpipe_per_n_s)
                mm2_fixpipe = mm2_fixpipe_per_n_s
                mla_source_schedule = (
                    max(
                        mm1_source_mte2_span_s,
                        mm1_mte1,
                        mm1_compute,
                        mm1_fixpipe_service_s,
                    ),
                    max(
                        mm2_source_mte2_span_s,
                        mm2_mte1,
                        mm2_compute,
                        mm2_fixpipe_service_s,
                    ),
                )
                # ``ExecuteTask`` first invokes ``ComputeMm1(current)`` and
                # then ``ComputeMm2(previous)``.  It does not issue a
                # task-wide pipe drain between them: only the source-owned
                # MTE2/MTE1/Tensor/Fixpipe waits and the C1->V1->C2 flags
                # constrain the next operation.  Retain the literal event
                # services so the outer resolver can share those pipes
                # instead of serializing two completed task makespans.
                mla_outer_source_profile = (
                    mm1_pair_services,
                    mm1_query_services,
                    mm1_mte1_per_k_s,
                    mm1_tensor_per_k_s,
                    mm1_fixpipe_per_n_s,
                    mm2_value_pair_services,
                    mm2_probability_services,
                    mm2_mte1_per_k_s,
                    mm2_tensor_per_k_s,
                    mm2_fixpipe_per_n_s,
                )
            mm1_buffered_l0_stage = None
            mm2_buffered_l0_stage = None
            # The regular GQA source does not drain every K (or P/V) L1
            # round before its first L1->L0/MMAD round.  The CANN
            # ``FiaBlockCubeNonQuantGqa`` program fixes this topology: one
            # 256-column K/P/V L1 round emits two 128-column L0 panels,
            # while the physical K/V copy can be one 256-row descriptor or
            # two 128-row paged descriptors.
            # ``FiaBlockCubeNonQuantGqa`` rotates three KP and two V L1
            # slots, two L0AB slots, and two L0C slots.  Once a round has
            # issued its MTE1 load, the next free L1 slot may accept an MTE2
            # copy while the Cube pipe consumes the prior round.  This is a
            # This is a finite source-event schedule, not a generic
            # serial-to-stream relaxation.  Select it from the complete
            # source topology rather than a particular captured grid or
            # shape: Q is optional after its source L1 snapshot; MM1 has one
            # K class; MM2 has P then V; and both stages expose the two
            # 128-column panels belonging to every 256-column round.
            source_order_rounds = None
            source_order_k_descriptors_per_round = None
            source_order_p_descriptors_per_round = None
            source_order_v_descriptors_per_round = None
            source_order_descriptors = None
            if (task.aic_pipeline_kind == "serial_stages"
                    # D64 source tasks retain the same literal Q2/V2/KP3
                    # program but publish no terminal MM2 Fixpipe demand.
                    # A zero service is still a complete source topology;
                    # it must not force a task-level drain.
                    and task.mm2_fixpipe.total_calls in (0, 1)
                    and task.mm1_mte1.load_pair is not None
                    and task.mm2_mte1.load_pair is not None
                    and len(task.mm1_mte2.descriptors) in (1, 2)
                    and len(task.mm2_mte2.descriptors) == 2):
                if len(task.mm1_mte2.descriptors) == 2:
                    q_descriptor, k_descriptor = task.mm1_mte2.descriptors
                else:
                    q_descriptor = None
                    k_descriptor, = task.mm1_mte2.descriptors
                p_descriptor, v_descriptor = task.mm2_mte2.descriptors
                round_count = p_descriptor.count
                if (round_count > 0
                        and k_descriptor.count % round_count == 0
                        and v_descriptor.count % round_count == 0):
                    k_per_round = k_descriptor.count // round_count
                    v_per_round = v_descriptor.count // round_count
                    panel_count = 2 * round_count
                    if (
                            k_per_round in (1, 2)
                            and v_per_round in (1, 2)
                            and p_descriptor.columns == 256
                            and k_per_round * k_descriptor.rows == 256
                            and v_per_round * v_descriptor.rows == 256
                            and k_descriptor.columns == (
                                task.mm1_mte1.load_pair.head_dim)
                            and v_descriptor.columns == (
                                task.mm2_mte1.load_pair.head_dim)
                            and task.mm1_fixpipe.total_calls == panel_count
                            and task.mm1_mte1.load_pair.count == panel_count
                            and task.mm2_mte1.load_pair.count == panel_count
                            and (q_descriptor is None or (
                                q_descriptor.count == 1
                                and q_descriptor.columns
                                == k_descriptor.columns))):
                        source_order_rounds = round_count
                        source_order_k_descriptors_per_round = k_per_round
                        source_order_p_descriptors_per_round = 1
                        source_order_v_descriptors_per_round = v_per_round
                        source_order_descriptors = (
                            q_descriptor, k_descriptor,
                            p_descriptor, v_descriptor)
            regular_gqa_source_order_eligible = (
                source_order_rounds is not None
                and source_order_k_descriptors_per_round is not None
                and source_order_p_descriptors_per_round is not None
                and source_order_v_descriptors_per_round is not None
                and source_order_descriptors is not None)

            # ``CopyKvGmToL1::ProcessPageAttention`` resolves the page-table
            # entry with ``GlobalTensor<int32_t>::GetValue`` before issuing
            # each K/V ND2NZ copy.  This source-only field is published by
            # the regular-FIA builder rather than inferred from a missing
            # stride: paged runs can have a known physical stride too.  Its
            # source-order model accepts the same source topology for
            # contiguous (all zero) and paged (one lookup per K/V physical
            # descriptor) routes; Q and P never own a page lookup.
            source_order_paged_lookup_eligible = (
                regular_gqa_source_order_eligible
                and (source_order_descriptors[0] is None
                     or source_order_descriptors[0].block_table_lookup_count
                     == 0)
                and source_order_descriptors[2].block_table_lookup_count == 0
                and source_order_descriptors[1].block_table_lookup_count
                in (0, 1)
                and source_order_descriptors[3].block_table_lookup_count
                in (0, 1)
            )
            # The compact stage ledger deliberately coalesces equal K/V
            # descriptors.  For a ragged final 256-KV source round that
            # destroys the literal boundary needed by the KP/V slot rings.
            # The builder supplies that topology separately, and this branch
            # accepts only the same literal GQA source shape modelled below.
            generic_gqa_source_order = task.gqa_source_order
            generic_gqa_source_order_eligible = (
                generic_gqa_source_order is not None
                and not regular_gqa_source_order_eligible
                and task.aic_pipeline_kind == "serial_stages"
                and task.mm2_fixpipe.total_calls in (0, 1)
                and all(
                    0 < round_demand.probability_descriptor.columns <= 256
                    and len(round_demand.key_descriptors) in (1, 2)
                    and len(round_demand.value_descriptors) in (1, 2)
                    and len(round_demand.panels) in (1, 2)
                    and all(
                        panel.mm1_mte1.load_pair is None
                        and panel.mm2_mte1.load_pair is None
                        for panel in round_demand.panels)
                    for round_demand in generic_gqa_source_order.rounds)
                # A source round with two L0 panels has the KP/V and L0 slot
                # lifetimes represented by this recurrence, even when both
                # panels are full width.  A one-panel round has no such
                # overlap opportunity and stays on the aggregate path.
                and any(
                    len(round_demand.panels) == 2
                    for round_demand in generic_gqa_source_order.rounds
                ))
            generic_gqa_paged_lookup_eligible = (
                generic_gqa_source_order_eligible
                and (generic_gqa_source_order.query_descriptor is None
                     or generic_gqa_source_order.query_descriptor
                     .block_table_lookup_count == 0)
                and all(
                    round_demand.probability_descriptor
                    .block_table_lookup_count == 0
                    and all(descriptor.block_table_lookup_count in (0, 1)
                            for descriptor in round_demand.key_descriptors)
                    and all(descriptor.block_table_lookup_count in (0, 1)
                            for descriptor in round_demand.value_descriptors)
                    for round_demand in generic_gqa_source_order.rounds))

            if regular_gqa_source_order_eligible:
                def _source_descriptor_completion_duration(descriptor):
                    """Return one source MTE2-to-MTE1 completion edge.

                    The published finite-drain program contains a full-pipe
                    drain after every descriptor.  That is valid aggregate
                    MTE2 demand evidence, but regular
                    FiaBlockCubeNonQuantGqa does not put that drain on its
                    K/P/V source loop: it releases the L1 slot once the final
                    MTE1 issue has been made.  Its literal MTE2-to-MTE1 wait
                    therefore consumes the existing independent
                    one-descriptor MTE2 service, not a fraction of a fully
                    drained program completion.

                    This does not change the MTE2 service ledger assembled
                    above.  It changes only the dependency edge used by the
                    source-order DAG, where every descriptor must finish its
                    own DMA before MTE1 can consume that slot.
                    """

                    source_stride = descriptor.source_row_stride_elements
                    descriptor_ns = self.hw.aic_mte2_nd2nz_serial_time_ns(
                        max(
                            descriptor.bytes_per_descriptor,
                            self.hw.aic_mte2_nd2nz_service[
                                "descriptor_bytes"][0],
                        ),
                        work_plan.active_group_count,
                    )
                    if (source_stride is not None
                            and source_stride != descriptor.columns
                            and descriptor.rows > 16):
                        descriptor_ns = max(
                            descriptor_ns,
                            self.hw.aic_mte2_nd2nz_strided_serial_time_ns(
                                descriptor.rows,
                                descriptor.columns,
                                source_stride,
                                work_plan.active_group_count,
                            ),
                        )
                    return descriptor_ns * 1e-9

                def _source_descriptor_stream_duration(descriptor):
                    """Return one back-to-back DMA issue interval.

                    ``CopyKvGmToL1`` emits paired physical paged copies before
                    its single MTE2-to-MTE1 completion edge.  The first copy
                    therefore uses the measured cold completion; later copies
                    use the independent stream issue service.  This only
                    changes source-burst placement; released L1 slots still
                    gate subsequent rounds in the outer DAG.
                    """

                    source_stride = descriptor.source_row_stride_elements
                    physical_descriptor_bytes = max(
                        descriptor.bytes_per_descriptor,
                        self.hw.aic_mte2_nd2nz_service[
                            "descriptor_bytes"][0],
                    )
                    if (physical_descriptor_bytes
                            < self.hw.aic_mte2_nd2nz_service[
                                "stream_descriptor_bytes"][0]):
                        return _source_descriptor_completion_duration(
                            descriptor)
                    descriptor_ns = self.hw.aic_mte2_nd2nz_stream_time_ns(
                        physical_descriptor_bytes,
                        work_plan.active_group_count,
                    )
                    if (source_stride is not None
                            and source_stride != descriptor.columns
                            and descriptor.rows > 16):
                        descriptor_ns = max(
                            descriptor_ns,
                            self.hw.aic_mte2_nd2nz_strided_stream_time_ns(
                                descriptor.rows,
                                descriptor.columns,
                                source_stride,
                                work_plan.active_group_count,
                            ))
                    return descriptor_ns * 1e-9

                paged_lookup_s = (
                    self.hw.resource_by_role(
                        "compute_fma").exact_request_service_time_s(
                            "aic_scalar_global_get_value_serialized", 1)
                    if source_order_paged_lookup_eligible else 0.0)

                def _source_order_mte2_burst_duration(
                        descriptor, descriptor_count):
                    """Issue one literal source burst with page lookup edges.

                    CANN has one scalar AIC dependency before each paged DMA
                    command, but its scalar pipe can advance to the next page
                    while the preceding MTE2 copy completes.  Keep separate
                    availability clocks rather than serially adding all
                    lookups to the finished DMA service.
                    """

                    if descriptor_count <= 0:
                        raise ValueError(
                            "FIA source-order descriptor count must be "
                            "positive")
                    descriptor_s = _source_descriptor_completion_duration(
                        descriptor)
                    stream_s = _source_descriptor_stream_duration(descriptor)
                    scalar_available_s = 0.0
                    mte2_available_s = 0.0
                    for descriptor_ordinal in range(descriptor_count):
                        scalar_available_s += (
                            descriptor.block_table_lookup_count
                            * paged_lookup_s)
                        mte2_available_s = max(
                            mte2_available_s, scalar_available_s,
                        ) + (descriptor_s if descriptor_ordinal == 0
                             else stream_s)
                    return mte2_available_s

                (q_descriptor, k_descriptor,
                 p_descriptor, v_descriptor) = source_order_descriptors
                q_mte2 = (
                    0.0 if q_descriptor is None
                    else _source_descriptor_completion_duration(q_descriptor))
                round_count = source_order_rounds
                k_mte2 = _source_order_mte2_burst_duration(
                    k_descriptor, source_order_k_descriptors_per_round)
                p_mte2 = _source_order_mte2_burst_duration(
                    p_descriptor, source_order_p_descriptors_per_round)
                v_mte2 = _source_order_mte2_burst_duration(
                    v_descriptor, source_order_v_descriptors_per_round)
                panel_count = 2 * round_count
                mm1_mte1_panel = mm1_mte1 / task.mm1_fixpipe.total_calls
                mm2_mte1_panel = mm2_mte1 / panel_count
                mm1_tensor_panel = mm1_compute / task.mm1_fixpipe.total_calls
                mm2_tensor_panel = mm2_compute / panel_count
                mm1_fixpipe_panel = (
                    mm1_fixpipe / task.mm1_fixpipe.total_calls)

                # MM1 emits K(i), then (only for i=0) Q, followed by two
                # 128-column L0/MMAD/Fixpipe panels.  KP releases after the
                # MTE1 consumers; AB and C releases wait for Cube and
                # Fixpipe respectively.  All service durations are the
                # existing typed descriptor/pair services divided only along
                # their literal, equal source calls.
                mte2_available = 0.0
                mte1_available = 0.0
                tensor_available = 0.0
                fixpipe_available = 0.0
                kp_ready = [0.0, 0.0, 0.0]
                ab_ready = [0.0, 0.0]
                c_ready = [0.0, 0.0]
                q_ready = 0.0
                panel_ordinal = 0
                for round_ordinal in range(round_count):
                    kp_slot = round_ordinal % len(kp_ready)
                    mte2_available = max(
                        mte2_available, kp_ready[kp_slot]) + k_mte2
                    k_ready = mte2_available
                    if round_ordinal == 0 and q_descriptor is not None:
                        mte2_available += q_mte2
                        q_ready = mte2_available
                    for _ in range(2):
                        ab_slot = panel_ordinal % len(ab_ready)
                        c_slot = panel_ordinal % len(c_ready)
                        mte1_available = max(
                            mte1_available, k_ready, q_ready,
                            ab_ready[ab_slot], c_ready[c_slot],
                        ) + mm1_mte1_panel
                        tensor_available = max(
                            tensor_available, mte1_available,
                        ) + mm1_tensor_panel
                        ab_ready[ab_slot] = tensor_available
                        fixpipe_available = max(
                            fixpipe_available, tensor_available,
                        ) + mm1_fixpipe_panel
                        c_ready[c_slot] = fixpipe_available
                        panel_ordinal += 1
                    kp_ready[kp_slot] = mte1_available
                mm1_stage = max(
                    mte2_available, mte1_available,
                    tensor_available, fixpipe_available,
                )

                # MM2 emits P(i), V(i), and two 128-column L0/MMAD panels
                # for every 256-column K round.  P uses the three KP slots;
                # V uses its two source slots.  The one final Fixpipe call is
                # issued only after the complete M loop, exactly as in the
                # CANN source.
                mte2_available = 0.0
                mte1_available = 0.0
                tensor_available = 0.0
                kp_ready = [0.0, 0.0, 0.0]
                v_ready = [0.0, 0.0]
                ab_ready = [0.0, 0.0]
                for round_ordinal in range(round_count):
                    kp_slot = round_ordinal % len(kp_ready)
                    v_slot = round_ordinal % len(v_ready)
                    mte2_available = max(
                        mte2_available, kp_ready[kp_slot],
                    ) + p_mte2
                    p_ready = mte2_available
                    mte2_available = max(
                        mte2_available, v_ready[v_slot],
                    ) + v_mte2
                    current_v_ready = mte2_available
                    for panel_offset in range(2):
                        ab_slot = (2 * round_ordinal + panel_offset) % len(ab_ready)
                        mte1_available = max(
                            mte1_available, p_ready, current_v_ready,
                            ab_ready[ab_slot],
                        ) + mm2_mte1_panel
                        tensor_available = max(
                            tensor_available, mte1_available,
                        ) + mm2_tensor_panel
                        ab_ready[ab_slot] = tensor_available
                    kp_ready[kp_slot] = mte1_available
                    v_ready[v_slot] = mte1_available
                mm2_pre_stage = max(
                    mte2_available, mte1_available, tensor_available,
                )
                mm2_fixpipe += mm2_pre_stage
                mm2_pre_stage = 0.0
                # ``FiaBlockCubeNonQuantGqa::ExecuteTask`` does not drain
                # these source resources at the task boundary.  Keep the
                # literal descriptor and panel services so the outer task
                # resolver can retain Q2, KP3, V2 and L0 rings across
                # ``MM1(current) -> MM2(previous)``.  This is deliberately
                # a topology record, not a new aggregate service: the
                # accounting ledgers above remain authoritative.
                gqa_outer_source_profile = (
                    "regular_gqa",
                    q_mte2,
                    tuple(
                        (
                            k_mte2,
                            p_mte2,
                            v_mte2,
                            (
                                (mm1_mte1_panel, mm1_tensor_panel,
                                 mm1_fixpipe_panel),
                                (mm1_mte1_panel, mm1_tensor_panel,
                                 mm1_fixpipe_panel),
                            ),
                            (
                                (mm2_mte1_panel, mm2_tensor_panel),
                                (mm2_mte1_panel, mm2_tensor_panel),
                            ),
                        )
                        for _ in range(round_count)
                    ),
                    mm2_fixpipe_service_s,
                )
            if generic_gqa_source_order_eligible:
                # This is the same finite KP/V slot recurrence as the
                # homogeneous branch above, with the literal short source
                # descriptors *and* their non-uniform L0/Cube/Fixpipe panels
                # preserved within each CANN K round.  It is not a generic
                # stream relaxation: MTE2 waits for the released three-KP/
                # two-V rings, while every exact panel remains on its source
                # dependency path.
                generic_paged_lookup_s = (
                    self.hw.resource_by_role(
                        "compute_fma").exact_request_service_time_s(
                            "aic_scalar_global_get_value_serialized", 1)
                    if generic_gqa_paged_lookup_eligible else 0.0)

                def generic_source_descriptor_duration(descriptor):
                    source_stride = descriptor.source_row_stride_elements
                    descriptor_ns = self.hw.aic_mte2_nd2nz_serial_time_ns(
                        max(
                            descriptor.bytes_per_descriptor,
                            self.hw.aic_mte2_nd2nz_service[
                                "descriptor_bytes"][0],
                        ),
                        work_plan.active_group_count,
                    )
                    if (source_stride is not None
                            and source_stride != descriptor.columns
                            and descriptor.rows > 16):
                        descriptor_ns = max(
                            descriptor_ns,
                            self.hw.aic_mte2_nd2nz_strided_serial_time_ns(
                                descriptor.rows,
                                descriptor.columns,
                                source_stride,
                                work_plan.active_group_count,
                            ),
                        )
                    return descriptor_ns * 1e-9

                def generic_source_descriptor_stream_duration(descriptor):
                    """Return the source-local interval after a DMA fill."""

                    source_stride = descriptor.source_row_stride_elements
                    physical_descriptor_bytes = max(
                        descriptor.bytes_per_descriptor,
                        self.hw.aic_mte2_nd2nz_service[
                            "descriptor_bytes"][0],
                    )
                    if (physical_descriptor_bytes
                            < self.hw.aic_mte2_nd2nz_service[
                                "stream_descriptor_bytes"][0]):
                        return generic_source_descriptor_duration(descriptor)
                    descriptor_ns = self.hw.aic_mte2_nd2nz_stream_time_ns(
                        physical_descriptor_bytes,
                        work_plan.active_group_count,
                    )
                    if (source_stride is not None
                            and source_stride != descriptor.columns
                            and descriptor.rows > 16):
                        descriptor_ns = max(
                            descriptor_ns,
                            self.hw.aic_mte2_nd2nz_strided_stream_time_ns(
                                descriptor.rows,
                                descriptor.columns,
                                source_stride,
                                work_plan.active_group_count,
                            ))
                    return descriptor_ns * 1e-9

                def generic_source_burst_duration(descriptors):
                    scalar_available_s = 0.0
                    mte2_available_s = 0.0
                    for descriptor_ordinal, descriptor in enumerate(
                            descriptors):
                        scalar_available_s += (
                            descriptor.block_table_lookup_count
                            * generic_paged_lookup_s)
                        mte2_available_s = max(
                            mte2_available_s, scalar_available_s,
                        ) + (
                            generic_source_descriptor_duration(descriptor)
                            if descriptor_ordinal == 0 else
                            generic_source_descriptor_stream_duration(
                                descriptor))
                    return mte2_available_s

                generic_rounds = generic_gqa_source_order.rounds
                query_descriptor = generic_gqa_source_order.query_descriptor
                q_mte2 = (
                    0.0 if query_descriptor is None
                    else generic_source_burst_duration((query_descriptor,)))

                mte2_available = 0.0
                mte1_available = 0.0
                tensor_available = 0.0
                fixpipe_available = 0.0
                kp_ready = [0.0, 0.0, 0.0]
                ab_ready = [0.0, 0.0]
                c_ready = [0.0, 0.0]
                q_ready = 0.0
                panel_ordinal = 0
                for round_ordinal, round_demand in enumerate(generic_rounds):
                    kp_slot = round_ordinal % len(kp_ready)
                    mte2_available = max(
                        mte2_available, kp_ready[kp_slot],
                    ) + generic_source_burst_duration(
                        round_demand.key_descriptors)
                    k_ready = mte2_available
                    if round_ordinal == 0 and query_descriptor is not None:
                        mte2_available += q_mte2
                        q_ready = mte2_available
                    for panel in round_demand.panels:
                        ab_slot = panel_ordinal % len(ab_ready)
                        c_slot = panel_ordinal % len(c_ready)
                        mte1_available = max(
                            mte1_available, k_ready, q_ready,
                            ab_ready[ab_slot], c_ready[c_slot],
                        ) + mte1_duration(panel.mm1_mte1)
                        tensor_available = max(
                            tensor_available, mte1_available,
                        ) + stage_duration(panel.mm1)
                        ab_ready[ab_slot] = tensor_available
                        if panel.mm1_fixpipe.descriptors:
                            fixpipe_available = max(
                                fixpipe_available, tensor_available,
                            ) + fixpipe_duration(panel.mm1_fixpipe)
                            c_ready[c_slot] = fixpipe_available
                        else:
                            c_ready[c_slot] = tensor_available
                        panel_ordinal += 1
                    kp_ready[kp_slot] = mte1_available
                mm1_stage = max(
                    mte2_available, mte1_available,
                    tensor_available, fixpipe_available,
                )

                mte2_available = 0.0
                mte1_available = 0.0
                tensor_available = 0.0
                kp_ready = [0.0, 0.0, 0.0]
                v_ready = [0.0, 0.0]
                ab_ready = [0.0, 0.0]
                panel_ordinal = 0
                for round_ordinal, round_demand in enumerate(generic_rounds):
                    kp_slot = round_ordinal % len(kp_ready)
                    v_slot = round_ordinal % len(v_ready)
                    mte2_available = max(
                        mte2_available, kp_ready[kp_slot],
                    ) + generic_source_burst_duration((
                        round_demand.probability_descriptor,))
                    p_ready = mte2_available
                    mte2_available = max(
                        mte2_available, v_ready[v_slot],
                    ) + generic_source_burst_duration(
                        round_demand.value_descriptors)
                    v_round_ready = mte2_available
                    for panel in round_demand.panels:
                        ab_slot = panel_ordinal % len(ab_ready)
                        mte1_available = max(
                            mte1_available, p_ready, v_round_ready,
                            ab_ready[ab_slot],
                        ) + mte1_duration(panel.mm2_mte1)
                        tensor_available = max(
                            tensor_available, mte1_available,
                        ) + stage_duration(panel.mm2)
                        ab_ready[ab_slot] = tensor_available
                        panel_ordinal += 1
                    kp_ready[kp_slot] = mte1_available
                    v_ready[v_slot] = mte1_available
                mm2_pre_stage = max(
                    mte2_available, mte1_available, tensor_available,
                )
                mm2_fixpipe += mm2_pre_stage
                mm2_pre_stage = 0.0
                # The generic descriptor builder preserves a ragged final
                # source round.  Carry that exact round/panel program into
                # the outer solver as well; replacing it by homogeneous
                # average panels would lose the source identity that makes
                # the persistent ring lifetime meaningful.
                gqa_outer_source_profile = (
                    "generic_gqa",
                    q_mte2,
                    tuple(
                        (
                            generic_source_burst_duration(
                                round_demand.key_descriptors),
                            generic_source_burst_duration((
                                round_demand.probability_descriptor,)),
                            generic_source_burst_duration(
                                round_demand.value_descriptors),
                            tuple(
                                (
                                    mte1_duration(panel.mm1_mte1),
                                    stage_duration(panel.mm1),
                                    fixpipe_duration(panel.mm1_fixpipe),
                                )
                                for panel in round_demand.panels
                            ),
                            tuple(
                                (
                                    mte1_duration(panel.mm2_mte1),
                                    stage_duration(panel.mm2),
                                )
                                for panel in round_demand.panels
                            ),
                        )
                        for round_demand in generic_rounds
                    ),
                    mm2_fixpipe_service_s,
                )
            paged_lookup_service_s = (
                math.fsum(
                    descriptor.count * descriptor.block_table_lookup_count
                    for stage in (task.mm1_mte2, task.mm2_mte2)
                    for descriptor in stage.descriptors)
                * (
                    paged_lookup_s if source_order_paged_lookup_eligible
                    else generic_paged_lookup_s)
                if (source_order_paged_lookup_eligible
                    or generic_gqa_paged_lookup_eligible) else 0.0)
            mla_full_panel_l0_buffer_eligible = (
                task.aic_pipeline_kind
                == "fia_block_cube_nonquant_mla_multibuffer"
                and work_plan.active_group_count == 1
                and work_plan.task_counts == (1,)
                and task.mm1_fixpipe.total_calls == 4
                and task.mm2_fixpipe.total_calls == 4)
            if (not regular_gqa_source_order_eligible
                    and not generic_gqa_source_order_eligible
                    and (gqa_l0_buffer_eligible
                         or mla_full_panel_l0_buffer_eligible)):
                # CANN's MLA Cube source retains three K/V L1 panels, four
                # Q/P L1 panels, and ping-pong L0A/L0B/L0C buffers.  A full
                # single-group, single-task S2=512 MLA program has four
                # identical 128-column output panels, so its L1->L0, MMAD,
                # and Fixpipe services form the same exact finite buffered
                # span as the regular-GQA source.  Multi-task programs retain
                # their outer three-slot recurrence instead of folding the
                # same buffering into both levels.
                # Keep MTE2 outside this overlap: sparse K/V gathers remain
                # source ordered and are not represented as panel classes.
                mm1_buffered_l0_stage = buffered_stage_span(
                    (mm1_mte1, mm1_compute, mm1_fixpipe),
                    task.mm1_fixpipe.total_calls)
                mm1_stage = mm1_mte2 + mm1_buffered_l0_stage
                mm2_load_count = (
                    task.mm2_fixpipe.total_calls
                    if mla_full_panel_l0_buffer_eligible
                    else (
                        1 if task.mm2_mte1.load_pair is None
                        else task.mm2_mte1.load_pair.count))
                mm2_buffered_l0_stage = buffered_stage_span(
                    (mm2_mte1, mm2_compute), mm2_load_count)
                mm2_pre_stage = mm2_mte2 + mm2_buffered_l0_stage
            if task.aic_pipeline_kind == "serial_stages":
                if (mm1_buffered_l0_stage is not None
                        and mm2_buffered_l0_stage is not None):
                    # FiaBlockCubeNonQuantGqa keeps three source tasks in
                    # flight and owns double-buffered L0A/B/C plus its K/P
                    # and V buffer rings.  The first two tasks remain the
                    # exact fill path above.  From the third task onward,
                    # the next task's MTE2 transfer can occupy its own pipe
                    # while the resident task drains the source-ordered L0
                    # stages.  This changes only their steady initiation
                    # interval and conserves every stage service.
                    mm1_steady = max(mm1_mte2, mm1_buffered_l0_stage)
                    mm2_steady = max(
                        mm2_mte2, mm2_buffered_l0_stage, mm2_fixpipe)
                else:
                    mm1_steady = mm1_stage
                    mm2_steady = mm2_pre_stage + mm2_fixpipe
            elif (task.aic_pipeline_kind
                  == "fia_block_cube_nonquant_mla_multibuffer"):
                if mla_source_schedule is None:
                    mm1_steady = max(
                        mm1_mte2, mm1_mte1, mm1_compute, mm1_fixpipe)
                    mm2_steady = max(
                        mm2_mte2, mm2_mte1, mm2_compute, mm2_fixpipe)
                else:
                    mm1_steady, mm2_steady = mla_source_schedule
            else:
                raise ValueError("unknown FIA AIC pipeline kind")
            result = (
                mm1_stage,
                vec1,
                mm2_pre_stage,
                mm2_fixpipe,
                vec1_update,
                vec2,
                mm1_mte2 + mm2_mte2,
                mm1_mte1 + mm2_mte1,
                mm1_compute + mm2_compute,
                (
                    mm1_fixpipe_service_s + mm2_fixpipe_service_s
                    if mla_source_schedule is not None
                    else mm1_fixpipe + mm2_fixpipe
                ),
                reduction + peer_reduction,
                stage_duration(task.vec1) + reduction + reduction_max + brcb
                + (0.0 if task.vec1_peer is None else (
                    stage_duration(task.vec1_peer) + peer_reduction
                    + peer_reduction_max + peer_brcb))
                + (0.0 if task.vec1_update is None
                   else stage_duration(task.vec1_update))
                + (0.0 if task.vec1_peer_update is None
                   else stage_duration(task.vec1_peer_update))
                + stage_duration(task.vec2)
                + (0.0 if task.vec2_peer is None
                   else stage_duration(task.vec2_peer)),
                mm1_steady,
                mm2_steady,
                vec1_primary_read + update_primary_read
                + vec2_primary_read + peer_memory_durations[0],
                vec1_primary_write + update_primary_write
                + vec2_primary_write + peer_memory_durations[1],
                reduction_max + peer_reduction_max,
                brcb + peer_brcb,
                (4 + (4 if task.vec1_peer is not None else 0))
                * queue_control_cycle_s,
                paged_lookup_service_s,
                mla_outer_source_profile,
                gqa_outer_source_profile,
            )
            task_duration_cache[cache_key] = result
            return result

        groups_by_program = {}
        for group in work_plan.groups:
            cold_fill_s = aiv_instruction_cold_fill_s(group)
            program_key = tuple(
                (task_durations(task), task.repeat_count)
                for task in group.tasks)
            program_key = (cold_fill_s, program_key)
            if program_key in groups_by_program:
                groups_by_program[program_key].append(group)
            else:
                groups_by_program[program_key] = [group]

        for matching_groups in groups_by_program.values():
            group = matching_groups[0]
            group_multiplier = sum(
                value.class_count for value in matching_groups)
            group_cold_fill_s = aiv_instruction_cold_fill_s(group)
            aic_available = initial_aic_ready_s
            # InitOutputSingleCore is AIV-only.  Its terminal SyncAll delays
            # the first Vector stage, while the AIC can initialize and begin
            # MM1 concurrently.
            aiv_available = initial_aiv_ready_s
            pending = []
            tasks_started = 0
            # The literal MLA source program exposes every MTE2/MTE1/Cube/
            # Fixpipe visit.  Use it only when all task classes in this
            # compressed worker program carry that complete topology; mixed
            # or aggregate programs retain the established task-stage path.
            mla_outer_source_eligible = all(
                task_durations(task)[20] is not None
                for task in group.tasks)
            mla_outer_state = (
                {
                    "mte2_available": 0.0,
                    "mte1_available": 0.0,
                    "tensor_available": 0.0,
                    "fixpipe_available": 0.0,
                    "kv_l1_ready": [0.0, 0.0, 0.0],
                    "kv_l1_ordinal": 0,
                }
                if mla_outer_source_eligible else None)
            # The regular-GQA source path has the same outer controller
            # ordering as the MLA path, but a different literal program:
            # Q is retained by a two-slot ring, K/P by a three-slot ring,
            # V by a two-slot ring, and the L0 panel rings remain live while
            # the following task enters MM1.  Only complete literal source
            # profiles may enter this resolver.  Aggregate or mixed task
            # programs retain the established task-stage model.
            gqa_outer_source_eligible = (
                not mla_outer_source_eligible
                and all(task_durations(task)[21] is not None
                        for task in group.tasks))
            gqa_outer_state = (
                {
                    "mte2_available": 0.0,
                    "mte1_available": 0.0,
                    "tensor_available": 0.0,
                    "fixpipe_available": 0.0,
                    "q_l1_ready": [0.0, 0.0],
                    "q_l1_ordinal": 0,
                    "kp_l1_ready": [0.0, 0.0, 0.0],
                    "kp_l1_ordinal": 0,
                    "v_l1_ready": [0.0, 0.0],
                    "v_l1_ordinal": 0,
                    "l0ab_ready": [0.0, 0.0],
                    "l0c_ready": [0.0, 0.0],
                    "l0ab_ordinal": 0,
                    "l0c_ordinal": 0,
                }
                if gqa_outer_source_eligible else None)

            vec1_cold_fill_pending = group_cold_fill_s > 0.0

            def vec1_duration(durations):
                nonlocal vec1_cold_fill_pending
                duration = durations[1]
                if vec1_cold_fill_pending:
                    duration += group_cold_fill_s
                    vec1_cold_fill_pending = False
                return duration

            def schedule_mla_mm1(durations, task_ordinal):
                """Issue the literal current-task MM1 pipe events.

                CANN's controller invokes this whole source routine before
                it enters the previous task's MM2 routine.  The calls only
                order pipe *issue*; final MTE2/MTE1/Cube/Fixpipe completion
                remains visible to the subsequent routine through the
                corresponding shared availability and L1-slot states.
                """

                if mla_outer_state is None:
                    raise AssertionError("MLA source state is unavailable")
                profile = durations[20]
                if profile is None:
                    raise AssertionError("MLA source profile is unavailable")
                (mm1_pair_services, mm1_query_services,
                 mm1_mte1_per_k_s, mm1_tensor_per_k_s,
                 mm1_fixpipe_per_n_s, _, _, _, _, _) = profile
                source_gate_s = (
                    initial_aic_ready_s if task_ordinal == 0 else 0.0)
                for n_ordinal in range(4):
                    for k_ordinal in range(2):
                        pair_ordinal = 2 * n_ordinal + k_ordinal
                        kv_slot = (
                            mla_outer_state["kv_l1_ordinal"]
                            % len(mla_outer_state["kv_l1_ready"]))
                        mla_outer_state["kv_l1_ordinal"] += 1
                        mte2_start_s = max(
                            mla_outer_state["mte2_available"],
                            mla_outer_state["kv_l1_ready"][kv_slot],
                            source_gate_s,
                        )
                        mla_outer_state["mte2_available"] = (
                            mte2_start_s + mm1_pair_services[pair_ordinal])
                        key_ready_s = mla_outer_state["mte2_available"]
                        source_gate_s = 0.0
                        query_ready_s = 0.0
                        if n_ordinal == 0 and mm1_query_services:
                            mla_outer_state["mte2_available"] += (
                                mm1_query_services[k_ordinal])
                            query_ready_s = mla_outer_state[
                                "mte2_available"]
                        mla_outer_state["mte1_available"] = max(
                            mla_outer_state["mte1_available"],
                            key_ready_s,
                            query_ready_s,
                        ) + mm1_mte1_per_k_s
                        mla_outer_state["tensor_available"] = max(
                            mla_outer_state["tensor_available"],
                            mla_outer_state["mte1_available"],
                        ) + mm1_tensor_per_k_s
                        mla_outer_state["kv_l1_ready"][kv_slot] = (
                            mla_outer_state["mte1_available"])
                        if k_ordinal == 1:
                            mla_outer_state["fixpipe_available"] = max(
                                mla_outer_state["fixpipe_available"],
                                mla_outer_state["tensor_available"],
                            ) + mm1_fixpipe_per_n_s
                return mla_outer_state["fixpipe_available"]

            def schedule_mla_mm2(durations, vec1_end_s):
                """Issue the literal previous-task MM2 pipe events.

                ``ExecuteTask`` calls ``ComputeVec1(previous)`` before
                ``ComputeMm2(previous)``.  ``ProcessVec1L`` publishes
                ``syncV1C2`` after Vec1, so MM2 is gated by that completion.
                Its later ``ProcessAmlaNupdate`` instead waits for
                ``syncC2V1``, which Cube publishes only after MM2.  Do not
                fold that post-MM2 AIV handoff into the MM2 input edge.
                """

                if mla_outer_state is None:
                    raise AssertionError("MLA source state is unavailable")
                profile = durations[20]
                if profile is None:
                    raise AssertionError("MLA source profile is unavailable")
                (_, _, _, _, _, mm2_value_pair_services,
                 mm2_probability_services, mm2_mte1_per_k_s,
                 mm2_tensor_per_k_s, mm2_fixpipe_per_n_s) = profile
                source_gate_s = vec1_end_s
                for n_ordinal in range(4):
                    for k_ordinal in range(2):
                        pair_ordinal = 2 * n_ordinal + k_ordinal
                        kv_slot = (
                            mla_outer_state["kv_l1_ordinal"]
                            % len(mla_outer_state["kv_l1_ready"]))
                        mla_outer_state["kv_l1_ordinal"] += 1
                        mte2_start_s = max(
                            mla_outer_state["mte2_available"],
                            mla_outer_state["kv_l1_ready"][kv_slot],
                            source_gate_s,
                        )
                        mla_outer_state["mte2_available"] = (
                            mte2_start_s
                            + mm2_value_pair_services[pair_ordinal])
                        value_ready_s = mla_outer_state["mte2_available"]
                        source_gate_s = 0.0
                        probability_ready_s = 0.0
                        if n_ordinal == 0:
                            mla_outer_state["mte2_available"] += (
                                mm2_probability_services[k_ordinal])
                            probability_ready_s = mla_outer_state[
                                "mte2_available"]
                        mla_outer_state["mte1_available"] = max(
                            mla_outer_state["mte1_available"],
                            value_ready_s,
                            probability_ready_s,
                        ) + mm2_mte1_per_k_s
                        mla_outer_state["tensor_available"] = max(
                            mla_outer_state["tensor_available"],
                            mla_outer_state["mte1_available"],
                        ) + mm2_tensor_per_k_s
                        mla_outer_state["kv_l1_ready"][kv_slot] = (
                            mla_outer_state["mte1_available"])
                        if k_ordinal == 1:
                            mla_outer_state["fixpipe_available"] = max(
                                mla_outer_state["fixpipe_available"],
                                mla_outer_state["tensor_available"],
                            ) + mm2_fixpipe_per_n_s
                return mla_outer_state["fixpipe_available"]

            def schedule_gqa_mm1(durations, task_ordinal):
                """Issue the literal regular-GQA MM1 source program.

                The CANN controller reuses its Q/KP/L0 buffers after the
                source consumer has issued, rather than inserting a task
                boundary drain.  Thus each descriptor and panel remains
                represented by the existing typed duration while its buffer
                release is recorded on the persistent state below.
                """

                if gqa_outer_state is None:
                    raise AssertionError("GQA source state is unavailable")
                profile = durations[21]
                if profile is None:
                    raise AssertionError("GQA source profile is unavailable")
                (_, query_service_s, rounds, _) = profile
                query_slot = None
                query_ready_s = 0.0
                for round_ordinal, (
                        key_service_s, _, _, mm1_panels, _) in enumerate(
                            rounds):
                    kp_slot = (
                        gqa_outer_state["kp_l1_ordinal"]
                        % len(gqa_outer_state["kp_l1_ready"]))
                    gqa_outer_state["kp_l1_ordinal"] += 1
                    gqa_outer_state["mte2_available"] = max(
                        gqa_outer_state["mte2_available"],
                        gqa_outer_state["kp_l1_ready"][kp_slot],
                    ) + key_service_s
                    key_ready_s = gqa_outer_state["mte2_available"]
                    if round_ordinal == 0 and query_service_s:
                        query_slot = (
                            gqa_outer_state["q_l1_ordinal"]
                            % len(gqa_outer_state["q_l1_ready"]))
                        gqa_outer_state["q_l1_ordinal"] += 1
                        gqa_outer_state["mte2_available"] = max(
                            gqa_outer_state["mte2_available"],
                            gqa_outer_state["q_l1_ready"][query_slot],
                        ) + query_service_s
                        query_ready_s = gqa_outer_state["mte2_available"]
                    for (mte1_service_s, tensor_service_s,
                         fixpipe_service_s) in mm1_panels:
                        l0ab_slot = (
                            gqa_outer_state["l0ab_ordinal"]
                            % len(gqa_outer_state["l0ab_ready"]))
                        gqa_outer_state["l0ab_ordinal"] += 1
                        l0c_slot = (
                            gqa_outer_state["l0c_ordinal"]
                            % len(gqa_outer_state["l0c_ready"]))
                        gqa_outer_state["mte1_available"] = max(
                            gqa_outer_state["mte1_available"],
                            key_ready_s,
                            query_ready_s,
                            gqa_outer_state["l0ab_ready"][l0ab_slot],
                            gqa_outer_state["l0c_ready"][l0c_slot],
                        ) + mte1_service_s
                        gqa_outer_state["tensor_available"] = max(
                            gqa_outer_state["tensor_available"],
                            gqa_outer_state["mte1_available"],
                        ) + tensor_service_s
                        gqa_outer_state["l0ab_ready"][l0ab_slot] = (
                            gqa_outer_state["tensor_available"])
                        if fixpipe_service_s:
                            gqa_outer_state["fixpipe_available"] = max(
                                gqa_outer_state["fixpipe_available"],
                                gqa_outer_state["tensor_available"],
                            ) + fixpipe_service_s
                            gqa_outer_state["l0c_ready"][l0c_slot] = (
                                gqa_outer_state["fixpipe_available"])
                        else:
                            gqa_outer_state["l0c_ready"][l0c_slot] = (
                                gqa_outer_state["tensor_available"])
                        gqa_outer_state["l0c_ordinal"] += 1
                    # The source calls release KP only after the final MTE1
                    # consumer of the round.  Q remains live for all K
                    # rounds and is released below after the whole MM1 call.
                    gqa_outer_state["kp_l1_ready"][kp_slot] = (
                        gqa_outer_state["mte1_available"])
                if query_slot is not None:
                    gqa_outer_state["q_l1_ready"][query_slot] = (
                        gqa_outer_state["mte1_available"])
                return max(
                    gqa_outer_state["mte2_available"],
                    gqa_outer_state["mte1_available"],
                    gqa_outer_state["tensor_available"],
                    gqa_outer_state["fixpipe_available"],
                )

            def schedule_gqa_mm2(durations, update_end_s):
                """Issue the literal previous-task MM2 source program."""

                if gqa_outer_state is None:
                    raise AssertionError("GQA source state is unavailable")
                profile = durations[21]
                if profile is None:
                    raise AssertionError("GQA source profile is unavailable")
                (_, _, rounds, terminal_fixpipe_s) = profile
                # ComputeMm2 waits on the current L0C buffer's FIX_M event
                # before it begins its P/V source loop.  L0C rotates once
                # per terminal MM2 Fixpipe, not once per L0AB panel, so keep
                # its ordinal separate from the double-buffered loads.
                l0c_slot = (
                    gqa_outer_state["l0c_ordinal"]
                    % len(gqa_outer_state["l0c_ready"]))
                source_gate_s = max(
                    update_end_s,
                    gqa_outer_state["l0c_ready"][l0c_slot],
                )
                for (_, probability_service_s, value_service_s, _,
                     mm2_panels) in rounds:
                    kp_slot = (
                        gqa_outer_state["kp_l1_ordinal"]
                        % len(gqa_outer_state["kp_l1_ready"]))
                    gqa_outer_state["kp_l1_ordinal"] += 1
                    v_slot = (
                        gqa_outer_state["v_l1_ordinal"]
                        % len(gqa_outer_state["v_l1_ready"]))
                    gqa_outer_state["v_l1_ordinal"] += 1
                    gqa_outer_state["mte2_available"] = max(
                        gqa_outer_state["mte2_available"],
                        gqa_outer_state["kp_l1_ready"][kp_slot],
                        source_gate_s,
                    ) + probability_service_s
                    probability_ready_s = gqa_outer_state["mte2_available"]
                    source_gate_s = 0.0
                    gqa_outer_state["mte2_available"] = max(
                        gqa_outer_state["mte2_available"],
                        gqa_outer_state["v_l1_ready"][v_slot],
                    ) + value_service_s
                    value_ready_s = gqa_outer_state["mte2_available"]
                    for mte1_service_s, tensor_service_s in mm2_panels:
                        l0ab_slot = (
                            gqa_outer_state["l0ab_ordinal"]
                            % len(gqa_outer_state["l0ab_ready"]))
                        gqa_outer_state["l0ab_ordinal"] += 1
                        gqa_outer_state["mte1_available"] = max(
                            gqa_outer_state["mte1_available"],
                            probability_ready_s,
                            value_ready_s,
                            gqa_outer_state["l0ab_ready"][l0ab_slot],
                        ) + mte1_service_s
                        gqa_outer_state["tensor_available"] = max(
                            gqa_outer_state["tensor_available"],
                            gqa_outer_state["mte1_available"],
                        ) + tensor_service_s
                        gqa_outer_state["l0ab_ready"][l0ab_slot] = (
                            gqa_outer_state["tensor_available"])
                    gqa_outer_state["kp_l1_ready"][kp_slot] = (
                        gqa_outer_state["mte1_available"])
                    gqa_outer_state["v_l1_ready"][v_slot] = (
                        gqa_outer_state["mte1_available"])
                gqa_outer_state["fixpipe_available"] = max(
                    gqa_outer_state["fixpipe_available"],
                    gqa_outer_state["mte2_available"],
                    gqa_outer_state["mte1_available"],
                    gqa_outer_state["tensor_available"],
                ) + terminal_fixpipe_s
                gqa_outer_state["l0c_ready"][l0c_slot] = (
                    gqa_outer_state["fixpipe_available"])
                gqa_outer_state["l0c_ordinal"] += 1
                return gqa_outer_state["fixpipe_available"]

            def advance_one(durations, *, steady):
                nonlocal aic_available, aiv_available, tasks_started
                if mla_outer_state is not None:
                    mm1_end = schedule_mla_mm1(durations, tasks_started)
                    pending.append({
                        "durations": durations,
                        "mm1_end": mm1_end,
                    })
                    tasks_started += 1
                    if len(pending) >= 2:
                        previous = pending[-2]
                        vec1_end = max(
                            aiv_available, previous["mm1_end"],
                        ) + vec1_duration(previous["durations"])
                        previous["mm2_end"] = schedule_mla_mm2(
                            previous["durations"], vec1_end)
                        # ``ProcessAmlaNupdate`` waits for the previous
                        # task's C2->V1 event.  It therefore begins only
                        # after Cube has completed MM2, not before it.
                        update_end = max(
                            vec1_end, previous["mm2_end"],
                        ) + previous["durations"][4]
                        aiv_available = update_end
                    if len(pending) >= 3:
                        oldest = pending.pop(0)
                        aiv_available = max(
                            aiv_available, oldest["mm2_end"],
                        ) + oldest["durations"][5]
                    aic_available = max(
                        mla_outer_state["mte2_available"],
                        mla_outer_state["mte1_available"],
                        mla_outer_state["tensor_available"],
                        mla_outer_state["fixpipe_available"],
                    )
                    return
                if gqa_outer_state is not None:
                    mm1_end = schedule_gqa_mm1(durations, tasks_started)
                    pending.append({
                        "durations": durations,
                        "mm1_end": mm1_end,
                    })
                    tasks_started += 1
                    if len(pending) >= 2:
                        previous = pending[-2]
                        vec1_end = max(
                            aiv_available, previous["mm1_end"],
                        ) + vec1_duration(previous["durations"])
                        update_end = vec1_end + previous["durations"][4]
                        aiv_available = update_end
                        previous["mm2_end"] = schedule_gqa_mm2(
                            previous["durations"], update_end)
                    if len(pending) >= 3:
                        oldest = pending.pop(0)
                        aiv_available = max(
                            aiv_available, oldest["mm2_end"],
                        ) + oldest["durations"][5]
                    aic_available = max(
                        gqa_outer_state["mte2_available"],
                        gqa_outer_state["mte1_available"],
                        gqa_outer_state["tensor_available"],
                        gqa_outer_state["fixpipe_available"],
                    )
                    return
                (mm1_stage, vec1_stage, mm2_pre_stage, mm2_fixpipe_stage,
                 vec1_update_stage, vec2_stage) = durations[:6]
                mm1_duration = durations[12] if steady else mm1_stage
                mm1_end = aic_available + mm1_duration
                aic_available = mm1_end
                pending.append({
                    "durations": durations,
                    "mm1_end": mm1_end,
                    "steady": steady,
                })
                tasks_started += 1
                if len(pending) >= 2:
                    previous = pending[-2]
                    vec1_end = max(
                        aiv_available, previous["mm1_end"]
                    ) + vec1_duration(previous["durations"])
                    update_end = vec1_end + previous["durations"][4]
                    aiv_available = update_end
                    if previous["steady"]:
                        mm2_end = max(
                            max(aic_available, vec1_end)
                            + previous["durations"][13],
                            update_end,
                        )
                    else:
                        mm2_pre_end = max(
                            aic_available, vec1_end
                        ) + previous["durations"][2]
                        mm2_end = max(
                            mm2_pre_end, update_end
                        ) + previous["durations"][3]
                    aic_available = mm2_end
                    previous["mm2_end"] = mm2_end
                if len(pending) >= 3:
                    oldest = pending.pop(0)
                    aiv_available = max(
                        aiv_available, oldest["mm2_end"]
                    ) + oldest["durations"][5]

            for task in group.tasks:
                durations = task_durations(task)
                count = task.repeat_count
                stage_service_s["aic_mte2_nd2nz"] += (
                    group_multiplier * count * durations[6])
                stage_service_s["fia_paged_kv_block_table_lookup"] += (
                    group_multiplier * count * durations[19])
                stage_service_s["aic_mte1_l1_to_l0"] += (
                    group_multiplier * count * durations[7])
                stage_service_s["compute_tensor"] += (
                    group_multiplier * count * durations[8])
                stage_service_s["aic_fixpipe_l0c_to_gm"] += (
                    group_multiplier * count * durations[9])
                stage_service_s["compute_vector"] += (
                    group_multiplier * count * durations[11])
                stage_service_s["aiv_block_reduce_sum"] += (
                    group_multiplier * count * durations[10])
                stage_service_s["aiv_block_reduce_max"] += (
                    group_multiplier * count * durations[16])
                stage_service_s["aiv_brcb"] += (
                    group_multiplier * count * durations[17])
                stage_service_s["aiv_queue_control"] += (
                    group_multiplier * count * durations[18])
                if (tasks_started == 0
                        and group_cold_fill_s > 0.0):
                    stage_service_s[
                        "fia_regular_aiv_instruction_cold_fill"] += (
                            group_multiplier * group_cold_fill_s)
                stage_service_s["aiv_mte2_gm_to_ub"] += (
                    group_multiplier * count * durations[14])
                stage_service_s["aiv_mte3_ub_to_gm"] += (
                    group_multiplier * count * durations[15])
                # Materialize through the first steady task.  The repeated
                # transition assumes the newest pending task already uses
                # the multibuffer service interval; the first two tasks are
                # still fill work and therefore cannot seed that recurrence.
                explicit = (
                    count if (mla_outer_state is not None
                              or gqa_outer_state is not None)
                    else min(count, max(1, 3 - tasks_started)))
                for _ in range(explicit):
                    advance_one(durations, steady=tasks_started >= 2)
                remaining = count - explicit
                if remaining:
                    if len(pending) != 2:
                        raise AssertionError(
                            "FIA homogeneous run did not fill its pipeline")
                    (aic_stage, vec1_stage, mm2_pre_stage,
                     mm2_fixpipe_stage, vec1_update_stage,
                     vec2_stage) = durations[:6]
                    steady_aic_stage = durations[12]
                    steady_mm2_stage = durations[13]
                    vec1_to_mm2_end = (
                        vec1_stage
                        + max(steady_mm2_stage, vec1_update_stage))
                    negative = float("-inf")
                    transition = (
                        (
                            steady_aic_stage + steady_mm2_stage,
                            vec1_to_mm2_end,
                            vec1_to_mm2_end,
                            negative,
                            negative,
                        ),
                        (
                            negative,
                            vec1_stage + vec1_update_stage + vec2_stage,
                            vec1_stage + vec1_update_stage + vec2_stage,
                            vec2_stage,
                            negative,
                        ),
                        (steady_aic_stage, negative, negative, negative,
                         negative),
                        (
                            steady_aic_stage + steady_mm2_stage,
                            vec1_to_mm2_end,
                            vec1_to_mm2_end,
                            negative,
                            negative,
                        ),
                        (negative, negative, negative, negative, 0.0),
                    )
                    state = apply_repeated_maxplus_transition(
                        transition,
                        remaining,
                        (
                            aic_available,
                            aiv_available,
                            pending[-1]["mm1_end"],
                            pending[-2]["mm2_end"],
                            0.0,
                        ),
                    )
                    aic_available, aiv_available = state[:2]
                    pending = [
                        {
                            "durations": durations,
                            "mm2_end": state[3],
                            "steady": True,
                        },
                        {
                            "durations": durations,
                            "mm1_end": state[2],
                            "steady": True,
                        },
                    ]
                    tasks_started += remaining

            if gqa_outer_state is not None:
                if len(pending) == 2:
                    oldest, newest = pending
                    vec1_end = max(
                        aiv_available, newest["mm1_end"],
                    ) + vec1_duration(newest["durations"])
                    update_end = vec1_end + newest["durations"][4]
                    aiv_available = update_end
                    newest["mm2_end"] = schedule_gqa_mm2(
                        newest["durations"], update_end)
                    aiv_available = max(
                        aiv_available, oldest["mm2_end"],
                    ) + oldest["durations"][5]
                    aiv_available = max(
                        aiv_available, newest["mm2_end"],
                    ) + newest["durations"][5]
                elif len(pending) == 1:
                    newest, = pending
                    vec1_end = max(
                        aiv_available, newest["mm1_end"],
                    ) + vec1_duration(newest["durations"])
                    update_end = vec1_end + newest["durations"][4]
                    aiv_available = update_end
                    newest["mm2_end"] = schedule_gqa_mm2(
                        newest["durations"], update_end)
                    aiv_available = max(
                        aiv_available, newest["mm2_end"],
                    ) + newest["durations"][5]
                else:
                    raise RuntimeError("FIA GQA group received no task runs")
                aic_available = max(
                    gqa_outer_state["mte2_available"],
                    gqa_outer_state["mte1_available"],
                    gqa_outer_state["tensor_available"],
                    gqa_outer_state["fixpipe_available"],
                )
            elif mla_outer_state is not None:
                if len(pending) == 2:
                    oldest, newest = pending
                    vec1_end = max(
                        aiv_available, newest["mm1_end"],
                    ) + vec1_duration(newest["durations"])
                    newest["mm2_end"] = schedule_mla_mm2(
                        newest["durations"], vec1_end)
                    # ``ProcessVec1L`` publishes V1->C2 immediately after
                    # Vec1, then waits for C2->V1 before it runs
                    # ProcessAmlaNupdate.  The final pipeline drain obeys
                    # the same order as the steady controller: nUpdate is
                    # post-MM2 work, never an MM2 input dependency.
                    update_end = max(
                        vec1_end, newest["mm2_end"],
                    ) + newest["durations"][4]
                    aiv_available = update_end
                    aiv_available = max(
                        aiv_available, oldest["mm2_end"],
                    ) + oldest["durations"][5]
                    aiv_available = max(
                        aiv_available, newest["mm2_end"],
                    ) + newest["durations"][5]
                elif len(pending) == 1:
                    newest, = pending
                    vec1_end = max(
                        aiv_available, newest["mm1_end"],
                    ) + vec1_duration(newest["durations"])
                    newest["mm2_end"] = schedule_mla_mm2(
                        newest["durations"], vec1_end)
                    update_end = max(
                        vec1_end, newest["mm2_end"],
                    ) + newest["durations"][4]
                    aiv_available = update_end
                    aiv_available = max(
                        aiv_available, newest["mm2_end"],
                    ) + newest["durations"][5]
                else:
                    raise RuntimeError("FIA MLA group received no task runs")
                aic_available = max(
                    mla_outer_state["mte2_available"],
                    mla_outer_state["mte1_available"],
                    mla_outer_state["tensor_available"],
                    mla_outer_state["fixpipe_available"],
                )
            elif len(pending) == 2:
                oldest, newest = pending
                vec1_end = max(
                    aiv_available, newest["mm1_end"]
                ) + vec1_duration(newest["durations"])
                update_end = vec1_end + newest["durations"][4]
                aiv_available = update_end
                if newest["steady"]:
                    newest["mm2_end"] = max(
                        max(aic_available, vec1_end)
                        + newest["durations"][13],
                        update_end,
                    )
                else:
                    mm2_pre_end = max(
                        aic_available, vec1_end
                    ) + newest["durations"][2]
                    newest["mm2_end"] = max(
                        mm2_pre_end, update_end
                    ) + newest["durations"][3]
                aic_available = newest["mm2_end"]
                aiv_available = max(
                    aiv_available, oldest["mm2_end"]
                ) + oldest["durations"][5]
                aiv_available = max(
                    aiv_available, newest["mm2_end"]
                ) + newest["durations"][5]
            elif len(pending) == 1:
                newest, = pending
                vec1_end = max(
                    aiv_available, newest["mm1_end"]
                ) + vec1_duration(newest["durations"])
                update_end = vec1_end + newest["durations"][4]
                aiv_available = update_end
                if newest["steady"]:
                    newest["mm2_end"] = max(
                        max(aic_available, vec1_end)
                        + newest["durations"][13],
                        update_end,
                    )
                else:
                    mm2_pre_end = max(
                        aic_available, vec1_end
                    ) + newest["durations"][2]
                    newest["mm2_end"] = max(
                        mm2_pre_end, update_end
                    ) + newest["durations"][3]
                aic_available = newest["mm2_end"]
                aiv_available = max(
                    aiv_available, newest["mm2_end"]
                ) + newest["durations"][5]
            else:
                raise RuntimeError("FIA AIC group received no task runs")

            compute_completion = (
                aiv_available + mla_multitask_dependency_startup_s)
            completion = max(
                compute_completion, ring_ready_s + ring_span_s)
            group_completion.extend(
                completion for _ in range(group_multiplier))
            for physical_group in matching_groups:
                has_peer = any(
                    task.vec1_peer is not None
                    for task in physical_group.tasks)
                for aic_worker in physical_group.aic_workers:
                    worker_makespan_s[
                        f"aic:{aic_worker}"] = aic_available
                    worker_makespan_s[
                        f"aiv:{2 * aic_worker}"] = aiv_available
                    if has_peer:
                        worker_makespan_s[
                            f"aiv:{2 * aic_worker + 1}"] = aiv_available
                    else:
                        worker_makespan_s[
                            f"aiv:{2 * aic_worker + 1}"] = initial_aiv_ready_s
                    worker_makespan_s[
                        f"group:{aic_worker}"] = completion
            if compute_completion > compute_critical_time:
                compute_critical_time = compute_completion
                compute_critical_group = group.aic_worker
            for physical_group in matching_groups:
                for aic_worker in physical_group.aic_workers:
                    if (len(physical_group.tasks) == 1
                            and physical_group.tasks[0].repeat_count == 1):
                        only_task = physical_group.tasks[0]
                        only_durations = task_durations(only_task)
                        mm1_end = initial_aic_ready_s + only_durations[0]
                        vec1_start = max(initial_aiv_ready_s, mm1_end)
                        vec1_end = vec1_start + only_durations[1]
                        update_end = vec1_end + only_durations[4]
                        mm2_pre_end = vec1_end + only_durations[2]
                        mm2_fixpipe_start = max(mm2_pre_end, update_end)
                        mm2_end = mm2_fixpipe_start + only_durations[3]
                        vec2_end = mm2_end + only_durations[5]
                        task_diagnostics.append(PipelineTaskDiagnostic(
                            task_name=f"fia_group{aic_worker}_task0",
                            worker=aic_worker,
                            stage_start_s={
                                "mm1_fixpipe": initial_aic_ready_s,
                                "vec1": vec1_start,
                                "vec1_update": vec1_end,
                                "mm2_pre": vec1_end,
                                "mm2_fixpipe": mm2_fixpipe_start,
                                "vec2": mm2_end,
                            },
                            stage_end_s={
                                "mm1_fixpipe": mm1_end,
                                "vec1": vec1_end,
                                "vec1_update": update_end,
                                "mm2_pre": mm2_pre_end,
                                "mm2_fixpipe": mm2_end,
                                "vec2": vec2_end,
                            },
                        ))
                        continue
                    for task in physical_group.tasks:
                        task_diagnostics.append(PipelineTaskDiagnostic(
                            task_name=(
                                f"fia_group{aic_worker}_"
                                f"task_run{task.ordinal}"),
                            worker=aic_worker,
                            stage_start_s={"program": 0.0},
                            stage_end_s={"program": compute_completion},
                            repeat_count=task.repeat_count,
                        ))
        analytical_main_execution_time = max(group_completion)
        if ring_ready_s + ring_span_s >= compute_critical_time:
            if initial_aiv_ready_s >= initial_aic_ready_s:
                prelude_path = (() if aiv_tiling_s == 0.0 else (
                    "fia_aiv_tiling_data_materialization",))
                prelude_path += (() if initial_vector_s == 0.0 else (
                    "fia_aiv_initialization",))
                prelude_path += ("fia_output_init_syncall",)
            else:
                prelude_path = (
                    "fia_aic_tiling_data_materialization",)
            main_critical_path = prelude_path + (
                "fia_cross_domain_program_span",)
        else:
            main_critical_path = (
                f"fia_group{compute_critical_group}_source_compute_dag",)
            if mla_multitask_dependency_startup_s > 0.0:
                main_critical_path += (
                    "fia_mla_multitask_dependency_startup",)
        tensor_override_s = (
            resource_time_overrides_ns["compute_tensor"] * 1e-9
            if "compute_tensor" in resource_time_overrides_ns else 0.0)
        main_execution_time = max(
            analytical_main_execution_time, tensor_override_s)
        if tensor_override_s > analytical_main_execution_time:
            main_critical_path = (
                "resource_time_override:compute_tensor",)
        terminal_critical_time = 0.0
        terminal_critical_worker = None
        if work_plan.terminal_vector_stages:
            if self.hw.fia_flash_decode_terminal_prelude_service is None:
                terminal_prelude_s = 0.0
            else:
                terminal_prelude_s = (
                    self.hw.fia_flash_decode_terminal_prelude_time_ns(
                        work_plan.active_group_count) * 1e-9)
            stage_service_s[
                "fia_flash_decode_terminal_prelude"] = terminal_prelude_s
            terminal_start = main_execution_time + terminal_prelude_s
            for worker, completion in tuple(worker_makespan_s.items()):
                if worker.startswith(("aic:", "aiv:")):
                    worker_makespan_s[worker] = max(
                        completion, terminal_start)
            for stage in work_plan.terminal_vector_stages:
                duration = stage_duration(stage)
                start = terminal_start
                end = start + duration
                stage_service_s["compute_vector"] += duration
                worker_makespan_s[f"aiv:{stage.physical_worker}"] = end
                task_diagnostics.append(PipelineTaskDiagnostic(
                    task_name=(
                        "fia_flash_decode_terminal_aiv"
                        f"{stage.physical_worker}"),
                    worker=stage.physical_worker,
                    stage_start_s={"flash_decode_terminal": start},
                    stage_end_s={"flash_decode_terminal": end},
                ))
                if duration > terminal_critical_time:
                    terminal_critical_time = duration
                    terminal_critical_worker = stage.physical_worker
            if terminal_critical_worker is None:
                raise RuntimeError(
                    "FIA FlashDecode terminal program produced no work")
            stage_service_s["fia_flash_decode_terminal_critical"] = (
                terminal_critical_time)
            dependency_completion = terminal_start + terminal_critical_time
            dependency_critical_path = main_critical_path + (
                "fia_flash_decode_terminal_prelude",
                f"fia_flash_decode_terminal_aiv{terminal_critical_worker}",
            )
        else:
            dependency_completion = main_execution_time
            dependency_critical_path = main_critical_path
        vector_override_s = (
            resource_time_overrides_ns["compute_vector"] * 1e-9
            if "compute_vector" in resource_time_overrides_ns else 0.0)
        execution_time = max(dependency_completion, vector_override_s)
        if vector_override_s > dependency_completion:
            critical_path = ("resource_time_override:compute_vector",)
        else:
            critical_path = dependency_critical_path
        steady_s = max(
            mechanism["steady_local_ns"],
            work_plan.active_group_count
            * mechanism["steady_shared_ns_per_active_group"],
        ) * 1e-9
        return PipelineDiagnostics(
            execution_time_s=execution_time,
            initiation_interval_s=steady_s,
            worker_makespan_s=worker_makespan_s,
            stage_service_s=stage_service_s,
            critical_path=critical_path,
            tasks=tuple(task_diagnostics),
        )

    def _resolve_fia_split_fuse_work_plan(
            self, work_plan, active_units, launch_task_kind,
            resource_time_overrides_ns, memory_diagnostics=()):
        """Solve the compressed SplitFuse AIC-to-dual-AIV source pipeline."""

        if not isinstance(work_plan, FiaSplitFuseWorkPlan):
            raise TypeError("expected FiaSplitFuseWorkPlan")
        if launch_task_kind != "mixed":
            raise ValueError("SplitFuse work plan requires a mixed launch")
        if active_units != work_plan.aic_worker_count:
            raise ValueError(
                "SplitFuse active units must equal its AIC worker count")
        execution_domain_units = self.hw.execution_units_for_task("mixed")
        if execution_domain_units != work_plan.aic_worker_count:
            raise ValueError(
                "SplitFuse work plan requires the full physical AIC launch")
        # The launch keeps all 24 AIC/48 AIV actors, but only groups assigned
        # a source task issue tensor or memory traffic.  Shared L2 bandwidth
        # and descriptor-service contention therefore use the source-active
        # group count; using the launch width charges idle groups as traffic
        # contenders and especially overstates short-Q packed tasks.
        active_aic_count = sum(
            task_count > 0 for task_count in work_plan.physical_task_counts)
        handoff_aic_count = sum(task_count > 0 for task_count in work_plan.task_counts)
        active_aiv_count = 2 * active_aic_count
        if active_aic_count <= 0:
            raise ValueError("SplitFuse work plan has no active AIC group")
        mte2_paths = tuple(
            diagnostic for diagnostic in memory_diagnostics
            if diagnostic.path_name == "mte2_read")
        if memory_diagnostics and len(mte2_paths) != 1:
            raise ValueError("SplitFuse requires one mte2_read memory path")
        if mte2_paths:
            nodes = {node.resource_role: node for node in mte2_paths[0].nodes}
            if set(nodes) != {"l2_cache", "dram"}:
                raise ValueError("SplitFuse MTE2 path requires L2 and DRAM")
            if (nodes["l2_cache"].bytes <= 0
                    or not 0 <= nodes["dram"].bytes
                    <= nodes["l2_cache"].bytes):
                raise ValueError("SplitFuse MTE2 miss bytes are invalid")
            mte2_dram_miss_fraction = (
                nodes["dram"].bytes / nodes["l2_cache"].bytes)
        else:
            mte2_dram_miss_fraction = None
        # CANN 8.5.1 SplitFuse has no ICachePreLoad.  A workload-independent
        # FIA Vec1/Vec2 probe at the same 24-group, dual-AIV topology measures
        # this conservative first-Vec1 cold-fill lower bound.
        split_fuse_aiv_instruction_cold_fill_s = (
            self.hw.fia_split_fuse_aiv_instruction_cold_fill_time_ns(
                execution_domain_units) * 1e-9)
        # flash_attention_regular.h (SplitFuse) has no normal-path output
        # initialization or SyncAll. The regular FIA InitOutput barrier is
        # a different source program and cannot seed this pipeline's clock.
        initial_sync_all_s = 0.0
        # CANN 8.5.1 SplitFuse is a cross-core producer-consumer ring: each
        # fused-loop stack crosses QK_READY (AIC -> AIV before the softmax,
        # flash_attention_regular.h CrossCoreWaitFlag(qkReady)),
        # SOFTMAX_READY (AIV -> AIC before the PV MMAD, set on PIPE_MTE3 and
        # waited in block_mmad_pv.hpp CrossCoreWaitFlag(softmaxFlag)) and
        # PV_READY (AIC -> AIV before Vec2).  The measured empty three-handoff
        # ring prices one depth (one MM1 -> Vec1 -> MM2 -> Vec2 alternation,
        # exactly one task's three source crossings) at
        # max(steady_local, groups * steady_shared); the symmetric three-flag
        # source structure splits that depth evenly, so each modeled
        # cross-core edge carries one third of the measured per-depth cost.
        # Charged as edge delays inside the max-plus graph they only extend
        # the critical path where the source pipeline really waits on them;
        # deep-stack PRE_LAUNCH lookahead hides them, keeping closed
        # long-KV buckets from regressing.  The per-depth unit depends only
        # on the active-group count, so the depth-count bound that gates the
        # whole-program span sum does not apply here.
        handoff_edge_s = 0.0
        handoff_mechanism = self.hw.fia_cross_domain_three_handoff
        if handoff_mechanism is not None and (
                handoff_aic_count
                in handoff_mechanism["single_task_active_aic_groups"]):
            handoff_depth_s = max(
                handoff_mechanism["steady_local_ns"],
                handoff_aic_count
                * handoff_mechanism[
                    "steady_shared_ns_per_active_group"]) * 1e-9
            handoff_edge_s = handoff_depth_s / 3.0
        tensor_resource = self.hw.resource_by_role("compute_tensor")
        vector_resource = self.hw.resource_by_role("compute_vector")
        l2_resource = self.hw.resource_by_role("l2_cache")
        l0a_resource = self.hw.resource_by_role("l0a_buffer")
        l0b_resource = self.hw.resource_by_role("l0b_buffer")
        mte3_resource = self.hw.resource_by_role("mte3")
        if tensor_resource.total_units != execution_domain_units:
            raise ValueError("SplitFuse requires one Tensor unit per AIC")
        if vector_resource.total_units != 2 * execution_domain_units:
            raise ValueError("SplitFuse requires two Vector units per AIC")
        if mte3_resource.total_units != 2 * execution_domain_units:
            raise ValueError("SplitFuse requires one MTE3 unit per AIV")
        if self.hw.fia_brcb_service is None:
            raise ValueError("SplitFuse requires the FIA Brcb source service")

        tensor_capacity = tensor_resource.effective_capacity(
            1,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        )
        if tensor_resource.scope == "shared":
            tensor_capacity /= execution_domain_units
        elif tensor_resource.scope == "per_unit":
            pass
        else:
            raise ValueError("SplitFuse Tensor resource has unknown scope")
        l2_capacity = l2_resource.effective_capacity(
            execution_domain_units,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        )
        l0a_capacity = l0a_resource.effective_capacity(
            1,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        )
        l0b_capacity = l0b_resource.effective_capacity(
            1,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        )
        mte3_capacity = mte3_resource.effective_capacity(
            active_aiv_count,
            sm_freq_hz=self.hw.nominal_sm_freq_hz,
            dram_freq_hz=self.hw.nominal_dram_freq_hz,
        )
        if min(
                tensor_capacity, l2_capacity, l0a_capacity, l0b_capacity,
                mte3_capacity) <= 0:
            raise ValueError("SplitFuse pipeline has a non-positive capacity")

        generic_rate = vector_resource.service_rate_for_one_unit("generic")
        exp_rate = vector_resource.service_rate_for_one_unit("exp")
        div_rate = vector_resource.service_rate_for_one_unit("div")
        muls_rate = vector_resource.service_rate_for_one_unit(
            "muls_serialized")
        cast_fp32_to_bf16_rate = vector_resource.service_rate_for_one_unit(
            "cast_fp32_to_bf16_serialized")
        exp_setup = vector_resource.setup_time_for_service("exp")
        div_setup = vector_resource.setup_time_for_service("div")
        muls_setup = vector_resource.setup_time_for_service(
            "muls_serialized")
        cast_fp32_to_bf16_setup = vector_resource.setup_time_for_service(
            "cast_fp32_to_bf16_serialized")
        paged_lookup_s = (
            self.hw.resource_by_role(
                "compute_fma").exact_request_service_time_s(
                    "aic_scalar_global_get_value_serialized", 1)
            if work_plan.paged_kv_cache else 0.0)
        # Source tasks repeatedly issue the same physical descriptor sizes.
        # Keep these exact HWSpec service results for this prediction only;
        # the active group count and hardware object are fixed in this scope.
        serial_mte2_ns = {}
        stream_mte2_ns = {}

        def mte2_service_ns(byte_count, *, stream):
            cache = stream_mte2_ns if stream else serial_mte2_ns
            if byte_count not in cache:
                cache[byte_count] = (
                    self.hw.aic_mte2_nd2nz_stream_time_ns(byte_count, active_aic_count)
                    if stream else
                    self.hw.aic_mte2_nd2nz_serial_time_ns(byte_count, active_aic_count))
            return cache[byte_count]

        aic_count = work_plan.aic_worker_count
        aiv_count = 2 * aic_count
        tensor_scale = work_plan.tensor_operation_scale
        vector_scale = work_plan.vector_operation_scale
        l2_scale = work_plan.l2_byte_scale
        mte3_scale = work_plan.mte3_byte_scale
        embed_round = math.ceil(work_plan.head_dim / 16) * 16
        vec2_row_tile = (8192 // embed_round // 8) * 8
        div_calls_per_loop = math.ceil(work_plan.head_dim / 64)
        if vec2_row_tile <= 0:
            raise ValueError("SplitFuse Vec2 row tile is empty")
        def aic_l2_time(byte_count):
            return (
                byte_count * l2_scale * active_aic_count
                / l2_capacity)

        def aiv_l2_time(byte_count):
            return (
                byte_count * l2_scale * active_aiv_count
                / l2_capacity)

        def aiv_write_time(byte_count):
            return max(
                aiv_l2_time(byte_count),
                byte_count * mte3_scale * active_aiv_count / mte3_capacity,
            )

        def aic_mte2_sequence_time(
                descriptor_bytes, *, paged_address_dependency=False,
                paged_kv_source=False):
            """Return one source copy burst with optional address lookups."""

            serial_minimum = self.hw.aic_mte2_nd2nz_service[
                "descriptor_bytes"][0]
            stream_minimum = self.hw.aic_mte2_nd2nz_service[
                "stream_descriptor_bytes"][0]
            scalar_available_s = 0.0
            mte2_available_s = 0.0
            for ordinal, byte_count in enumerate(descriptor_bytes):
                physical_bytes = max(byte_count, serial_minimum)
                if ordinal == 0 or byte_count < stream_minimum:
                    duration_ns = mte2_service_ns(physical_bytes, stream=False)
                else:
                    duration_ns = mte2_service_ns(physical_bytes, stream=True)
                if paged_address_dependency:
                    scalar_available_s += paged_lookup_s
                mte2_available_s = (
                    max(mte2_available_s, scalar_available_s)
                    + duration_ns * 1e-9)
            if (paged_kv_source and work_plan.head_dim == 128
                    and work_plan.kv_head_count == 8
                    and active_aic_count == 24
                    and mte2_dram_miss_fraction is not None
                    and len(descriptor_bytes) >= 3):
                # A measured full-page prefix must still constrain a longer
                # source stream whose final descriptor is only a partial page.
                finite_s = max((
                    self.hw.split_fuse_mte2_short_tail_time_ns(
                        count, 24, mte2_dram_miss_fraction,
                        two_slot_overlap=True) * 1e-9
                    for count in (3, 4, 5, 7)
                    if len(descriptor_bytes) >= count
                    and all(byte_count == 32768
                            for byte_count in descriptor_bytes[:count])
                ), default=0.0)
                # The finite probe supplies a lower bound on the full ingress
                # path; it cannot erase the existing descriptor service.
                return max(finite_s, mte2_available_s)
            return mte2_available_s

        def axis_slices(size, block_size):
            full, tail = divmod(size, block_size)
            return (block_size,) * full + (() if tail == 0 else (tail,))

        def three_stage_pipeline_time(stage_rows):
            available = [0.0, 0.0, 0.0]
            for row in stage_rows:
                if len(row) != 3 or any(value < 0.0 for value in row):
                    raise ValueError(
                        "SplitFuse internal pipeline requires three "
                        "non-negative stage durations")
                for stage, duration in enumerate(row):
                    predecessor = available[stage - 1] if stage else 0.0
                    start = max(available[stage], predecessor)
                    available[stage] = start + duration
            return available[-1]

        def mte2_tensor_output_pipeline_time(
                stage_rows, *, paged_address_dependency=False,
                paged_kv_source=False):
            """Solve a two-buffer descriptor stream through AIC pipes."""

            scalar_available = 0.0
            mte2_available = 0.0
            tensor_available = 0.0
            output_available = 0.0
            for descriptor_bytes, tensor_time, output_time in stage_rows:
                if (descriptor_bytes <= 0 or tensor_time < 0.0
                        or output_time < 0.0):
                    raise ValueError(
                        "SplitFuse descriptor pipeline requires positive "
                        "bytes and non-negative stage durations")
                serial_minimum = self.hw.aic_mte2_nd2nz_service[
                    "descriptor_bytes"][0]
                physical_bytes = max(descriptor_bytes, serial_minimum)
                serial_time = max(
                    aic_l2_time(physical_bytes),
                    mte2_service_ns(physical_bytes, stream=False) * 1e-9,
                )
                if descriptor_bytes < self.hw.aic_mte2_nd2nz_service[
                        "stream_descriptor_bytes"][0]:
                    stream_time = serial_time
                else:
                    stream_time = max(
                        aic_l2_time(physical_bytes),
                        mte2_service_ns(physical_bytes, stream=True) * 1e-9,
                    )
                if paged_address_dependency:
                    scalar_available += paged_lookup_s
                mte2_start = max(mte2_available, scalar_available)
                mte2_end = mte2_start + serial_time
                mte2_available = mte2_start + stream_time
                tensor_start = max(tensor_available, mte2_end)
                tensor_available = tensor_start + tensor_time
                output_start = max(output_available, tensor_available)
                output_available = output_start + output_time
            completion = max(mte2_available, tensor_available, output_available)
            if (paged_kv_source and work_plan.head_dim == 128
                    and work_plan.kv_head_count == 8
                    and active_aic_count == 24
                    and mte2_dram_miss_fraction is not None
                    and len(stage_rows) >= 3):
                # Preserve every measured full-page prefix as a completion
                # lower bound when the source stream continues past it.
                finite_s = max((
                    self.hw.split_fuse_mte2_short_tail_time_ns(
                        count, 24, mte2_dram_miss_fraction,
                        two_slot_overlap=False) * 1e-9
                    for count in (3, 4, 5, 7)
                    if len(stage_rows) >= count
                    and all(row[0] == 32768
                            for row in stage_rows[:count])
                ), default=0.0)
                completion = max(completion, finite_s)
            return completion

        task_duration_cache = {}

        def task_durations(query_rows, packed_rows, attended_kv):
            cache_key = (query_rows, packed_rows, attended_kv)
            if cache_key in task_duration_cache:
                return task_duration_cache[cache_key]
            mm_rows = math.ceil(packed_rows / 16) * 16
            output_elements = packed_rows * work_plan.head_dim
            stack_widths = tuple(
                min(
                    work_plan.kv_stack_ceiling,
                    attended_kv - stack_start)
                for stack_start in range(
                    0, attended_kv, work_plan.kv_stack_ceiling))
            q_bytes = 2 * packed_rows * work_plan.head_dim
            mm1 = (
                max(
                    aic_l2_time(q_bytes),
                    aic_mte2_sequence_time((q_bytes,)),
                )
                + mte2_tensor_output_pipeline_time(tuple(
                    (
                        2 * n_rows * work_plan.head_dim,
                        (
                            2 * mm_rows * work_plan.head_dim / l0a_capacity
                            + 2 * work_plan.head_dim * n_rows / l0b_capacity
                            + 2 * work_plan.head_dim * mm_rows * n_rows
                            * tensor_scale / tensor_capacity
                        ),
                        aic_l2_time(4 * packed_rows * n_rows),
                    )
                    for columns in stack_widths
                    for n_rows in axis_slices(columns, 128)),
                    paged_address_dependency=work_plan.paged_kv_cache,
                    paged_kv_source=work_plan.paged_kv_cache)
            )
            if work_plan.paged_kv_cache:
                # CANN's fixed ``PRE_LAUNCH=2`` means that only a V prefix
                # with two later source stacks has an outer QK iteration in
                # which it can issue before the Softmax wait.  The final two
                # drain stacks retain the aggregate post-Vec1 completion.
                mm2 = three_stage_pipeline_time(tuple(
                    (
                        max(
                            aic_l2_time(
                                2 * packed_rows * columns
                                + 2 * columns * work_plan.head_dim),
                            math.fsum((
                                aic_mte2_sequence_time(tuple(
                                    2 * v_rows * work_plan.head_dim
                                    for v_rows in axis_slices(columns, 128)),
                                    paged_address_dependency=True,
                                    paged_kv_source=True),
                                aic_mte2_sequence_time(tuple(
                                    2 * packed_rows * k_rows
                                    for k_rows in axis_slices(columns, 256))),
                            ))),
                        (
                            2 * mm_rows * columns / l0a_capacity
                            + 2 * columns * work_plan.head_dim / l0b_capacity
                            + 2 * work_plan.head_dim * mm_rows * columns
                            * tensor_scale / tensor_capacity
                        ),
                        aic_l2_time(4 * output_elements),
                    )
                    for columns in stack_widths))
            else:
                serial_minimum = self.hw.aic_mte2_nd2nz_service[
                    "descriptor_bytes"][0]
                stream_minimum = self.hw.aic_mte2_nd2nz_service[
                    "stream_descriptor_bytes"][0]

                def mm2_mte2_duration(byte_count, descriptor_ordinal):
                    physical_bytes = max(byte_count, serial_minimum)
                    if (descriptor_ordinal == 0
                            or byte_count < stream_minimum):
                        duration_ns = mte2_service_ns(physical_bytes, stream=False)
                    else:
                        duration_ns = mte2_service_ns(physical_bytes, stream=True)
                    return max(
                        aic_l2_time(physical_bytes), duration_ns * 1e-9)

                def mm2_tensor_duration(panel_columns):
                    return (
                        2 * mm_rows * panel_columns / l0a_capacity
                        + 2 * panel_columns * work_plan.head_dim
                        / l0b_capacity
                        + 2 * work_plan.head_dim * mm_rows
                        * panel_columns * tensor_scale / tensor_capacity)

                mm2 = _split_fuse_mm2_source_order_time(
                    stack_widths,
                    packed_rows,
                    work_plan.head_dim,
                    mm2_mte2_duration,
                    mm2_tensor_duration,
                    aic_l2_time(4 * output_elements),
                )
            # BlockMmadPV loads the V tile before its softmaxReady wait for
            # both PAGED_CACHE_FLAG specializations.  The outer source loop
            # can issue only stacks with two later QK iterations before that
            # wait is on the critical path; its final two stacks are drains.
            # Preserve the total PV service in ``mm2`` and move only this
            # source-defined prefix into the earlier AIC interval.
            mm2_v_ingress_s = math.fsum(
                max(
                    aic_l2_time(2 * columns * work_plan.head_dim),
                    aic_mte2_sequence_time(
                        tuple(
                            2 * v_rows * work_plan.head_dim
                            for v_rows in axis_slices(columns, 128)),
                        paged_address_dependency=work_plan.paged_kv_cache,
                        paged_kv_source=work_plan.paged_kv_cache),
                )
                for stack_index, columns in enumerate(stack_widths)
                if stack_index + 2 < len(stack_widths)
            ) if work_plan.pv_v_ingress_before_softmax else 0.0
            if mm2_v_ingress_s > mm2:
                raise AssertionError(
                    "SplitFuse V ingress cannot exceed PV completion")
            mm2_l0 = math.fsum(
                2 * mm_rows * columns / l0a_capacity
                + 2 * columns * work_plan.head_dim / l0b_capacity
                for columns in stack_widths)
            vec1 = []
            vec2 = []
            vec1_block_reduce_sum = []
            vec1_block_reduce_max = []
            vec1_brcb = []
            vec1_queue_control = []
            vec2_queue_control = []
            for lane in range(2):
                rows = (
                    packed_rows // 2 if lane == 0
                    else packed_rows - packed_rows // 2)
                if rows == 0:
                    vec1.append(0.0)
                    vec2.append(0.0)
                    vec1_block_reduce_sum.extend(
                        0.0 for _ in stack_widths)
                    vec1_block_reduce_max.extend(
                        0.0 for _ in stack_widths)
                    vec1_brcb.extend(0.0 for _ in stack_widths)
                    continue
                lane_outputs = rows * work_plan.head_dim
                vec1_rows = []
                for stack_start in range(
                        0, attended_kv, work_plan.kv_stack_ceiling):
                    columns = min(
                        work_plan.kv_stack_ceiling,
                        attended_kv - stack_start)
                    stack_end = stack_start + columns
                    is_first_stack = stack_start == 0
                    masked = attended_kv - query_rows < stack_end - 1
                    alignment = 32 if masked else 16
                    rounded_columns = (
                        math.ceil(columns / alignment) * alignment)
                    mask_bytes = rows * rounded_columns if masked else 0
                    row_tile = min(
                        (8192 // rounded_columns // 8) * 8, 64)
                    if row_tile <= 0:
                        raise ValueError("SplitFuse Vec1 row tile is empty")
                    lane_scores = rows * columns
                    source_row_chunks = math.ceil(rows / row_tile)
                    # The pinned online-softmax source has one score-wise
                    # generic operation (CalcExp::Sub), followed by its
                    # typed Exp.  ScaleS and DownCastP use their own typed
                    # Muls and Cast services.  In every non-first KV stack,
                    # UpdateGlobalRowMax/RowSum instead issue row-wise
                    # Max/Sub/Exp/Mul/Add.  The causal stack alone also runs
                    # ApplyMask's score-wise Muls/Add.  Do not charge the
                    # latter operations to every score of every stack.
                    mask_elements = mask_bytes
                    state_rows = 0 if is_first_stack else rows
                    vec1_compute = vector_scale * (
                        (lane_scores + mask_elements + 4 * state_rows)
                        / generic_rate
                        + (lane_scores + mask_elements) / muls_rate
                        + lane_scores / cast_fp32_to_bf16_rate
                        + (lane_scores + state_rows) / exp_rate)
                    vec1_compute += source_row_chunks * (
                        muls_setup + cast_fp32_to_bf16_setup + exp_setup)
                    if masked:
                        vec1_compute += source_row_chunks * muls_setup
                    if not is_first_stack:
                        vec1_compute += source_row_chunks * exp_setup
                    # The pinned SplitFuse source uses statically allocated
                    # LocalTensor buffers: its regular, online-softmax, and
                    # rescale headers contain no runtime TQue lifetime.
                    # Do not charge the unrelated generic AIV queue-control
                    # mechanism to these source row chunks.
                    vec1_queue_control.append(0.0)
                    block_reduce_sum_s = 0.0
                    block_reduce_max_s = 0.0
                    brcb_s = 0.0
                    for row_start in range(0, rows, row_tile):
                        actual_rows = min(row_tile, rows - row_start)
                        row_chunk_round = math.ceil(actual_rows / 8) * 8
                        block_reduce_sum_s += (
                            self.hw.fia_block_reduce_sum_time_ns(
                                row_chunk_round,
                                columns,
                                alignment,
                            ) * 1e-9)
                        block_reduce_max_s += (
                            self.hw.fia_block_reduce_max_time_ns(
                                row_chunk_round,
                                columns,
                                alignment,
                            ) * 1e-9)
                        brcb_s += self.hw.fia_brcb_time_ns(
                            row_chunk_round) * 1e-9
                    vec1_compute += (
                        block_reduce_sum_s + block_reduce_max_s + brcb_s)
                    vec1_rows.append((
                        aiv_l2_time(4 * lane_scores + mask_bytes),
                        vec1_compute,
                        aiv_write_time(2 * lane_scores),
                    ))
                    vec1_block_reduce_sum.append(block_reduce_sum_s)
                    vec1_block_reduce_max.append(block_reduce_max_s)
                    vec1_brcb.append(brcb_s)
                vec1.append(three_stage_pipeline_time(tuple(vec1_rows)))
                div_invocations = (
                    math.ceil(rows / vec2_row_tile) * div_calls_per_loop)
                cast_invocations = math.ceil(rows / vec2_row_tile)
                # The terminal source path serializes its output Cast before
                # the MTE3 write, so it uses the same measured typed service.
                vec2_compute = vector_scale * (
                    lane_outputs / cast_fp32_to_bf16_rate
                    + lane_outputs / div_rate
                    # block_epilogue_rescale_o.hpp: every non-first stack
                    # rescales the retained output with Mul then Add. This
                    # work also executes when no GM accumulator is needed.
                    + 2 * lane_outputs * (len(stack_widths) - 1)
                    / generic_rate)
                vec2_compute += (
                    div_invocations * div_setup
                    + cast_invocations * cast_fp32_to_bf16_setup)
                vec2_row_chunks = math.ceil(rows / vec2_row_tile)
                # See the Vec1 source audit above: this source program has
                # no runtime TQue lifecycle in its Vec2 rescale path either.
                vec2_queue_control.append(0.0)
                vec2.append(
                    aiv_l2_time(
                        4 * lane_outputs * len(stack_widths))
                    + vec2_compute
                    + aiv_write_time(2 * lane_outputs))
            result = (
                mm1,
                tuple(vec1),
                mm2,
                tuple(vec2),
                tuple(vec1_block_reduce_sum),
                tuple(vec1_block_reduce_max),
                tuple(vec1_brcb),
                mm2_l0,
                tuple(vec1_queue_control),
                tuple(vec2_queue_control),
                mm2_v_ingress_s,
            )
            task_duration_cache[cache_key] = result
            return result

        state_classes = [(
            tuple(range(aic_count)),
            {
                "aic": 0.0,
                "aiv": [initial_sync_all_s, initial_sync_all_s],
                "pending": [],
                "vec1_cold_fill_pending": [True, True],
            },
        )]
        stage_service_s = {
            "split_fuse_mm1": 0.0,
            "split_fuse_vec1": 0.0,
            "split_fuse_mm2": 0.0,
            "split_fuse_vec2": 0.0,
            "split_fuse_block_reduce_sum": 0.0,
            "split_fuse_block_reduce_max": 0.0,
            "split_fuse_brcb": 0.0,
            "split_fuse_mm2_l0": 0.0,
            "split_fuse_vec1_queue_control": 0.0,
            "split_fuse_vec2_queue_control": 0.0,
            "split_fuse_paged_kv_lookup": 0.0,
            "split_fuse_empty_kv_q_load": 0.0,
            "split_fuse_empty_kv_output_init": 0.0,
            "split_fuse_aiv_instruction_cold_fill": (
                2 * aic_count * split_fuse_aiv_instruction_cold_fill_s),
            "fia_output_init_syncall": initial_sync_all_s,
            "fia_cross_domain_program_span": 0.0,
            "fia_split_fuse_cross_core_handoff": (
                3.0 * handoff_edge_s * sum(
                    count * math.ceil(kv / work_plan.kv_stack_ceiling)
                    for geometries in work_plan.worker_geometry_counts
                    for _, _, kv, count in geometries)),
        }

        negative = float("-inf")

        source_task_cache = {}

        def source_task_stacks(query_rows, packed_rows, attended_kv):
            """Materialize source stack stages, with one Q ingress per task."""
            key = (query_rows, packed_rows, attended_kv)
            if key in source_task_cache:
                return source_task_cache[key]
            q_bytes = 2 * packed_rows * work_plan.head_dim
            q_load = max(aic_l2_time(q_bytes), aic_mte2_sequence_time((q_bytes,)))
            stacks = []
            for offset in range(0, attended_kv, work_plan.kv_stack_ceiling):
                columns = min(work_plan.kv_stack_ceiling, attended_kv - offset)
                first = offset == 0
                last = offset + columns == attended_kv
                masked = attended_kv - query_rows < offset + columns - 1
                base = task_durations(query_rows if masked else 1, packed_rows, columns)
                vec1, vec2 = [], []
                vec2_brcb = []
                alignment = 32 if masked else 16
                rounded = math.ceil(columns / alignment) * alignment
                row_tile = min((8192 // rounded // 8) * 8, 64)
                for lane in range(2):
                    rows = packed_rows // 2 if lane == 0 else packed_rows - packed_rows // 2
                    if rows == 0:
                        vec1.append(0.0)
                        vec2.append(0.0)
                        vec2_brcb.append(0.0)
                        continue
                    outputs = rows * work_plan.head_dim
                    v1 = base[1][lane]
                    if not first:
                        # UpdateGlobalRowMax/RowSum are per non-first stack,
                        # independent of whether its causal mask is active.
                        v1 += vector_scale * (4 * rows / generic_rate + rows / exp_rate)
                        v1 += math.ceil(rows / row_tile) * exp_setup
                    vec1.append(v1)
                    # The first stack seeds the UB accumulator. Subsequent
                    # stacks execute Mul/Add; only the final stack Div/Casts
                    # and stores the public output. This retains de9b02abc.
                    v2 = aiv_l2_time(4 * outputs)
                    # block_epilogue_rescale_o.hpp:185-190,243-249:
                    # broadcast dm for each non-first stack and gl for the
                    # terminal stack, once per source output row chunk.
                    brcb = (int(not first) + int(last)) * math.fsum(
                        self.hw.fia_brcb_time_ns(
                            math.ceil(min(vec2_row_tile, rows - start) / 8) * 8
                        ) * 1e-9
                        for start in range(0, rows, vec2_row_tile))
                    v2 += brcb
                    vec2_brcb.append(brcb)
                    if rows > vec2_row_tile:
                        # The accumulator cannot stay in UB across row
                        # loops: nonterminal stacks spill gUpdate and later
                        # stacks reload it (source:192-198,343-347).
                        if not first:
                            v2 += aiv_l2_time(4 * outputs)
                        if not last:
                            v2 += aiv_write_time(4 * outputs)
                    if not first:
                        v2 += vector_scale * 2 * outputs / generic_rate
                    if last:
                        v2 += base[3][lane] - aiv_l2_time(4 * outputs)
                    vec2.append(v2)
                v_ingress = max(aic_l2_time(2 * columns * work_plan.head_dim),
                    aic_mte2_sequence_time(tuple(
                        2 * rows * work_plan.head_dim
                        for rows in axis_slices(columns, 128)),
                        paged_address_dependency=work_plan.paged_kv_cache,
                        paged_kv_source=work_plan.paged_kv_cache)
                    ) if work_plan.pv_v_ingress_before_softmax else 0.0
                if v_ingress > base[2]:
                    raise ValueError("SplitFuse V ingress exceeds its PV service")
                stacks.append((base[0] if first else base[0] - q_load,
                    tuple(vec1), base[2], tuple(vec2),
                    base[4], base[5], base[6] + tuple(vec2_brcb),
                    base[7], base[8], base[9], v_ingress))
            source_task_cache[key] = tuple(stacks)
            return source_task_cache[key]

        source_transition_cache = {}

        def source_task_transition(geometry, cold):
            """Three-actor source loop, with PRE_LAUNCH inside one task.

            QK(k), Vec1(k), PV(k-2), Vec2(k-2) retain source order. The
            complete stack loop drains before the next request/head task;
            AIC and both AIV clocks still overlap between adjacent tasks.
            AIC stages retain the existing internal pipe approximation.
            """
            key = (*geometry, *cold)
            if key in source_transition_cache:
                return source_transition_cache[key]
            stacks = source_task_stacks(*geometry)
            aic = (0.0, negative, negative)
            aiv = [(negative, 0.0, negative), (negative, negative, 0.0)]
            softmax_ready = []
            pending_cold = list(cold)

            def delayed(value, duration):
                return tuple(entry + duration for entry in value)

            def joined(*values):
                return tuple(max(entries) for entries in zip(*values))

            # flash_attention_regular.h:297,311,420,499-501. The two drain
            # iterations belong to this task, not the next two tasks.
            for iteration in range(len(stacks) + 2):
                if iteration < len(stacks):
                    stack = stacks[iteration]
                    aic = delayed(aic, stack[0])
                    for lane in range(2):
                        duration = stack[1][lane]
                        if pending_cold[lane]:
                            duration += split_fuse_aiv_instruction_cold_fill_s
                            pending_cold[lane] = False
                        aiv[lane] = delayed(joined(aiv[lane],
                            delayed(aic, handoff_edge_s)), duration)
                    softmax_ready.append(joined(*aiv))
                if iteration >= 2:
                    ordinal = iteration - 2
                    stack = stacks[ordinal]
                    # BlockMmadPV loads V before waiting for softmaxReady.
                    aic = delayed(aic, stack[10])
                    aic = delayed(joined(aic,
                        delayed(softmax_ready[ordinal], handoff_edge_s)),
                        stack[2] - stack[10])
                    for lane in range(2):
                        aiv[lane] = delayed(joined(aiv[lane],
                            delayed(aic, handoff_edge_s)), stack[3][lane])
            transition = (aic, *aiv)
            source_transition_cache[key] = transition
            return transition

        def advance_program(state, program, repeat_count):
            if repeat_count <= 0 or not program:
                raise ValueError("SplitFuse source program requires positive repeats and tasks")
            vector = (state["aic"], *state["aiv"])
            if any(state["vec1_cold_fill_pending"]):
                for query_rows, packed_rows, attended_kv, count in program:
                    geometry = (query_rows, packed_rows, attended_kv)
                    vector = apply_repeated_maxplus_transition(
                        source_task_transition(geometry, state["vec1_cold_fill_pending"]),
                        1, vector)
                    state["vec1_cold_fill_pending"] = [False, False]
                    if count > 1:
                        vector = apply_repeated_maxplus_transition(
                            source_task_transition(geometry, (False, False)),
                            count - 1, vector)
                repeat_count -= 1
            if repeat_count:
                transitions = tuple(_maxplus_power_matrix(
                    _maxplus_powers(source_task_transition((q, p, k), (False, False)), count),
                    count, 3) for q, p, k, count in program)
                vector = apply_repeated_maxplus_transition(
                    compose_maxplus_transitions(transitions), repeat_count, vector)
            state["aic"], state["aiv"] = vector[0], list(vector[1:])

        compressed_task_count = 0
        for geometries in work_plan.worker_geometry_counts:
            for (query_rows, packed_rows, attended_kv,
                 multiplier) in geometries:
                stacks = source_task_stacks(query_rows, packed_rows, attended_kv)
                durations = tuple(
                    math.fsum(stack[index] for stack in stacks)
                    if index in (0, 2, 7, 10) else
                    tuple(math.fsum(stack[index][lane] for stack in stacks)
                          for lane in range(2))
                    if index in (1, 3) else
                    tuple(value for stack in stacks for value in stack[index])
                    for index in range(11))
                compressed_task_count += multiplier
                stage_service_s["split_fuse_mm1"] += (
                    multiplier * durations[0])
                stage_service_s["split_fuse_vec1"] += (
                    multiplier * math.fsum(durations[1]))
                stage_service_s["split_fuse_mm2"] += (
                    multiplier * durations[2])
                stage_service_s["split_fuse_vec2"] += (
                    multiplier * math.fsum(durations[3]))
                stage_service_s["split_fuse_block_reduce_sum"] += (
                    multiplier * math.fsum(durations[4]))
                stage_service_s["split_fuse_block_reduce_max"] += (
                    multiplier * math.fsum(durations[5]))
                stage_service_s["split_fuse_brcb"] += (
                    multiplier * math.fsum(durations[6]))
                stage_service_s["split_fuse_mm2_l0"] += (
                    multiplier * durations[7])
                stage_service_s["split_fuse_vec1_queue_control"] += (
                    multiplier * math.fsum(durations[8]))
                stage_service_s["split_fuse_vec2_queue_control"] += (
                    multiplier * math.fsum(durations[9]))
                stage_service_s["split_fuse_paged_kv_lookup"] += (
                    multiplier
                    * work_plan.paged_descriptor_lookup_count(attended_kv)
                    * paged_lookup_s)
        for block_repeat, worker_programs in (
                work_plan.iter_worker_program_blocks()):
            next_classes = []
            for workers, state in state_classes:
                workers_by_program = {}
                for worker in workers:
                    program = worker_programs[worker]
                    if program in workers_by_program:
                        workers_by_program[program].append(worker)
                    else:
                        workers_by_program[program] = [worker]
                for program, matching_workers in workers_by_program.items():
                    class_state = {
                        "aic": state["aic"],
                        "aiv": list(state["aiv"]),
                        "pending": [dict(task) for task in state["pending"]],
                        "vec1_cold_fill_pending": list(
                            state["vec1_cold_fill_pending"]),
                    }
                    if program:
                        advance_program(class_state, program, block_repeat)
                    next_classes.append((
                        tuple(matching_workers), class_state))
            state_classes = next_classes
        if compressed_task_count != work_plan.task_count:
            raise AssertionError(
                "SplitFuse periodic programs lost source tasks")

        states = [None] * aic_count
        for workers, state in state_classes:
            for worker in workers:
                states[worker] = state
        if any(state is None for state in states):
            raise AssertionError(
                "SplitFuse worker-class folding lost an AIC")

        # The independent zero-KV request is last in the source task order.
        # Its AIC loads Q; its AIV runs only the local zero-output epilogue.
        # These actor streams have no QK/SOFTMAX/PV cross-domain edges and
        # must not enter the positive-KV max-plus handoff recurrence.
        for worker, geometries in enumerate(work_plan.empty_kv_tail_worker_geometry):
            if not geometries:
                continue
            state = {"aic": states[worker]["aic"], "aiv": list(states[worker]["aiv"])}
            for rows, heads, count in geometries:
                q_bytes = 2 * rows * heads * work_plan.head_dim
                q_load_s = max(aic_l2_time(q_bytes), aic_mte2_sequence_time((q_bytes,)))
                state["aic"] += count * q_load_s
                stage_service_s["split_fuse_empty_kv_q_load"] += count * q_load_s
                for lane in range(2):
                    lane_rows = (rows // 2 if lane == 0 else rows - rows // 2) if heads == 1 else rows
                    lane_heads = 1 if heads == 1 else (heads // 2 if lane == 0 else heads - heads // 2)
                    if not lane_rows or not lane_heads:
                        continue
                    dup_s = lane_rows * embed_round * vector_scale / generic_rate
                    output_s = aiv_write_time(2 * lane_rows * lane_heads * work_plan.head_dim)
                    service_s = dup_s + output_s
                    state["aiv"][lane] += count * service_s
                    stage_service_s["split_fuse_empty_kv_output_init"] += count * service_s
            states[worker] = state

        group_completion = tuple(
            max(state["aic"], *state["aiv"]) for state in states)
        analytical_time = max(group_completion)
        mechanism = self.hw.fia_cross_domain_three_handoff
        active_task_counts = tuple(
            task_count for task_count in work_plan.task_counts
            if task_count > 0)
        if (mechanism is not None
                and len(active_task_counts) in mechanism[
                    "single_task_active_aic_groups"]
                and all(
                    mechanism["minimum_task_count"] <= task_count
                    <= mechanism["maximum_task_count"]
                    for task_count in active_task_counts)):
            ring_span_s = self.hw.fia_cross_domain_three_handoff_time_ns(
                active_task_counts) * 1e-9
            stage_service_s["fia_cross_domain_program_span"] = ring_span_s
        else:
            ring_span_s = 0.0
        execution_time = max(analytical_time, ring_span_s)
        for role in ("compute_tensor", "compute_vector"):
            if role in resource_time_overrides_ns:
                execution_time = max(
                    execution_time,
                    resource_time_overrides_ns[role] * 1e-9)
        critical_worker = max(
            range(aic_count), key=lambda worker: group_completion[worker])
        worker_makespan_s = {}
        diagnostics = []
        for worker, state in enumerate(states):
            completion = group_completion[worker]
            worker_makespan_s[f"aic:{worker}"] = state["aic"]
            worker_makespan_s[f"aiv:{2 * worker}"] = state["aiv"][0]
            worker_makespan_s[f"aiv:{2 * worker + 1}"] = state["aiv"][1]
            worker_makespan_s[f"group:{worker}"] = completion
            if work_plan.physical_task_counts[worker]:
                diagnostics.append(PipelineTaskDiagnostic(
                    task_name=(
                        f"split_fuse_group{worker}_compressed_program"),
                    worker=worker,
                    stage_start_s={"program": 0.0},
                    stage_end_s={"program": completion},
                    repeat_count=work_plan.physical_task_counts[worker],
                ))
        critical_path = (
            ("fia_cross_domain_program_span",)
            if ring_span_s >= analytical_time
            else (f"split_fuse_group{critical_worker}_source_pipeline",))
        initiation_interval = max(
            group_completion[worker] / work_plan.physical_task_counts[worker]
            for worker in range(aic_count)
            if work_plan.physical_task_counts[worker])
        return PipelineDiagnostics(
            execution_time_s=execution_time,
            initiation_interval_s=initiation_interval,
            worker_makespan_s=worker_makespan_s,
            stage_service_s=stage_service_s,
            critical_path=critical_path,
            tasks=tuple(diagnostics),
        )

    def predict_kernel(self, inp):
        """Predict one canonical ``PredictionInput`` without hardware-ID logic."""

        if not isinstance(inp, PredictionInput):
            raise TypeError(
                "predict_kernel requires PredictionInput, got "
                f"{type(inp).__name__}")
        demand_per_resource = inp.demand_per_resource

        positive_compute_roles = {
            role for role in (
                "compute_tensor", "compute_vector", "compute_reduction",
                "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
                "scalar_aiv")
            if demand_per_resource[role] > 0
        }
        structured_compute_roles = {
            demand.resource_role for demand in inp.compute_demands
            if demand.operations > 0
        }
        if positive_compute_roles != structured_compute_roles:
            raise ValueError(
                "compute demand/schema mismatch: "
                f"flat={sorted(positive_compute_roles)}, "
                f"structured={sorted(structured_compute_roles)}")
        compute_amount_mismatch = {
            demand.resource_role: (
                demand_per_resource[demand.resource_role],
                demand.operations,
            )
            for demand in inp.compute_demands
            if not math.isclose(
                demand_per_resource[demand.resource_role],
                demand.operations,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        }
        if compute_amount_mismatch:
            raise ValueError(
                "flat and structured compute demand amounts disagree: "
                f"{compute_amount_mismatch}")

        target_memory_model = inp.mem_model in ("gpu_mshr", "npu_tile")
        if target_memory_model:
            expected_path = (
                demand_per_resource["l2_cache"] > 0
                or (inp.mem_model == "npu_tile"
                    and demand_per_resource["mte3"] > 0))
            if expected_path != bool(inp.memory_paths):
                raise ValueError(
                    "structured memory path is required exactly when target "
                    "hardware has positive read or write traffic")
        elif inp.mem_model in ("pim_bank", "systolic_scratchpad"):
            if inp.memory_paths:
                raise ValueError(
                    "cacheless compatibility models cannot consume target "
                    "memory paths")
        else:
            raise ValueError(
                f"unknown PredictionInput.mem_model: {inp.mem_model!r}")
        structured_memory_bytes = {}
        for path in inp.memory_paths:
            for node in path.nodes:
                if node.resource_role in structured_memory_bytes:
                    structured_memory_bytes[node.resource_role] += node.bytes
                else:
                    structured_memory_bytes[node.resource_role] = node.bytes
        if inp.mem_model == "gpu_mshr":
            structured_memory_roles = ("l2_cache", "dram")
        elif inp.mem_model == "npu_tile":
            structured_memory_roles = ("l2_cache", "dram")
            if "mte3" in structured_memory_bytes:
                structured_memory_roles += ("mte3",)
            elif demand_per_resource["mte3"] > 0:
                if "mte3" not in inp.resource_time_overrides_ns:
                    raise ValueError(
                        "positive unstructured MTE3 traffic is allowed only "
                        "for an explicit Oracle pipe-time override")
        elif inp.mem_model in ("pim_bank", "systolic_scratchpad"):
            structured_memory_roles = ()
        else:
            raise AssertionError(
                f"unhandled structured memory model: {inp.mem_model!r}")
        for role in structured_memory_roles:
            if role not in structured_memory_bytes:
                structured_memory_bytes[role] = 0.0
        memory_amount_mismatch = {
            role: (
                demand_per_resource[role],
                structured_memory_bytes[role],
            )
            for role in structured_memory_roles
            if not math.isclose(
                demand_per_resource[role],
                structured_memory_bytes[role],
                rel_tol=1e-12, abs_tol=1e-6)
        }
        if memory_amount_mismatch:
            raise ValueError(
                "flat and structured memory demand amounts disagree: "
                f"{memory_amount_mismatch}")

        t_per_resource = {resource.name: 0.0 for resource in self.hw.resources}
        compute_time_by_role = {}
        for compute in inp.compute_demands:
            if compute.operations == 0:
                continue
            if compute.resource_role not in self.hw._role_map:
                raise ValueError(
                    "hardware has no resource for positive compute role "
                    f"{compute.resource_role!r}")
            resource = self.hw.resource_by_role(compute.resource_role)
            if resource.type != "compute":
                raise ValueError(
                    f"compute role {compute.resource_role!r} is not compute")
            if (compute.service_demands
                    or compute.exact_request_demands):
                execution_domain_units = self.hw.execution_units_for_task(
                    inp.launch_task_kind)
                service_durations = []
                worker_scheduled = bool(
                    compute.service_demands
                    and compute.service_demands[0].operations_by_worker)
                worker_count = None
                if worker_scheduled or compute.exact_request_demands:
                    active_domain_units = min(
                        inp.active_units, execution_domain_units)
                    worker_numerator = (
                        resource.total_units * active_domain_units)
                    if worker_numerator % execution_domain_units:
                        raise ValueError(
                            f"compute role {compute.resource_role!r} active "
                            "resource worker count is fractional")
                    worker_count = (
                        worker_numerator // execution_domain_units)
                    if worker_count <= 0:
                        raise ValueError(
                            f"compute role {compute.resource_role!r} has no "
                            "active physical workers")
                if worker_scheduled:
                    if (compute.shape_efficiency != 1.0
                            or compute.schedule_efficiency != 1.0):
                        raise ValueError(
                            "worker-scheduled typed compute requires explicit "
                            "physical operations and unit efficiencies")
                    worker_durations = [0.0] * worker_count
                    for service_demand in compute.service_demands:
                        if (len(service_demand.operations_by_worker)
                                != worker_count):
                            raise ValueError(
                                f"compute role {compute.resource_role!r} typed "
                                f"service {service_demand.service!r} worker "
                                "count disagrees with active physical units")
                        rate = resource.service_rate_for_one_unit(
                            service_demand.service)
                        setup = (
                            resource.setup_time_for_service(
                                service_demand.service)
                            if service_demand.serial_invocations > 0
                            else 0.0)
                        for worker, (operations, invocations) in enumerate(zip(
                                service_demand.operations_by_worker,
                                service_demand.serial_invocations_by_worker)):
                            worker_durations[worker] += (
                                operations / rate + invocations * setup)
                else:
                    for service_demand in compute.service_demands:
                        capacity = resource.effective_service_capacity(
                            service_demand.service,
                            active_units=inp.active_units,
                            execution_domain_units=execution_domain_units,
                        )
                        effective_capacity = (
                            capacity * compute.shape_efficiency
                            * compute.schedule_efficiency)
                        if effective_capacity <= 0:
                            raise ValueError(
                                f"compute role {compute.resource_role!r} typed "
                                f"service {service_demand.service!r} has no "
                                "capacity")
                        steady_time = (
                            service_demand.operations / effective_capacity)
                        if service_demand.serial_invocations > 0:
                            setup_time = (
                                service_demand.serial_invocations
                                * resource.setup_time_for_service(
                                    service_demand.service))
                        else:
                            setup_time = 0.0
                        service_durations.append(steady_time + setup_time)
                exact_worker_durations = (
                    [0.0] * worker_count
                    if compute.exact_request_demands else None)
                for exact_demand in compute.exact_request_demands:
                    if exact_demand.worker_count != worker_count:
                        raise ValueError(
                            f"compute role {compute.resource_role!r} exact "
                            f"service {exact_demand.service!r} worker count "
                            "disagrees with active physical units")
                    for worker, worker_runs in enumerate(
                            exact_demand.request_size_runs):
                        if exact_demand.pipelined_completion:
                            exact_time = max(
                                resource.exact_request_service_time_s(
                                    exact_demand.service, request_size)
                                for request_size, _count in worker_runs)
                        else:
                            exact_time = math.fsum(
                                resource.exact_request_service_time_s(
                                    exact_demand.service, request_size) * count
                                for request_size, count in worker_runs)
                        exact_worker_durations[worker] += exact_time
                if exact_worker_durations is not None:
                    if worker_scheduled:
                        for worker, exact_time in enumerate(
                                exact_worker_durations):
                            worker_durations[worker] += exact_time
                    else:
                        service_durations.append(
                            max(exact_worker_durations))
                if worker_scheduled:
                    # Typed and exact request services on one AIV serialize.
                    # Physical AIVs execute concurrently, so take one maximum
                    # over the complete worker programs, not one maximum per
                    # service followed by a sum.
                    service_durations.append(max(worker_durations))
                duration = math.fsum(service_durations)
            else:
                capacity = self._compute_capacity(inp, resource)
                # Preserve the historical left-associated arithmetic exactly
                # when no typed service is active.  This is the production
                # path for every current hardware/operator pair.
                effective_capacity = (
                    capacity * compute.shape_efficiency
                    * compute.schedule_efficiency)
                if effective_capacity <= 0:
                    raise ValueError(
                        f"compute role {compute.resource_role!r} has no "
                        "capacity")
                duration = compute.operations / effective_capacity
            t_per_resource[resource.name] += duration
            compute_time_by_role[compute.resource_role] = duration

        memory_diagnostics = []
        memory_access_class_diagnostics = ()
        memory_arbitration_diagnostics = []
        covered_memory_roles = set()
        memory_path_time = {}
        memory_path_roles = {}
        replacing_memory_paths = set()
        for path in inp.memory_paths:
            replacing_access_classes = tuple(
                demand for demand in path.access_class_demands
                if demand.replaces_ingress_path_service)
            if replacing_access_classes:
                replacing_memory_paths.add(path.name)
                replaced_ingress_bytes = math.fsum(
                    demand.logical_bytes
                    for demand in replacing_access_classes)
                ingress_bytes = path.nodes[0].bytes
                if len(replacing_access_classes) != len(
                        path.access_class_demands):
                    raise ValueError(
                        "a memory path cannot mix replacing and independent "
                        "access-class services")
                if not math.isclose(
                        replaced_ingress_bytes, ingress_bytes,
                        rel_tol=1e-12, abs_tol=1e-6):
                    raise ValueError(
                        "replacing memory access-class services must cover "
                        "the complete ingress path")
            hierarchy = memory_hierarchy_for_path(
                self.hw,
                path,
                sm_freq_hz=inp.sm_freq_hz,
                dram_freq_hz=inp.dram_freq_hz,
                dram_capacity_multiplier=inp.dram_capacity_multiplier,
            )
            diagnostic = solve_memory_path(path, hierarchy)
            memory_diagnostics.append(diagnostic)
            roles = tuple(node.resource_role for node in path.nodes)
            memory_path_roles[path.name] = roles
            # A non-empty read cannot complete before its first returned
            # request, even when the workload-side independent-request count
            # is unknown and the steady-state solver therefore exposes only a
            # byte-capacity lower bound.  Store completion is intentionally
            # left unchanged: the current GPU contract has no independently
            # established write-drain latency.
            if replacing_access_classes:
                # A full source-program service owns request generation and
                # the ingress resource.  Keep only independently priced
                # downstream hierarchy nodes as the generic path bound; the
                # typed service is joined with this bound below.
                downstream_capacity_times = tuple(
                    node.capacity_time_s for node in diagnostic.nodes[1:])
                if downstream_capacity_times:
                    memory_path_time[path.name] = max(
                        downstream_capacity_times)
                else:
                    memory_path_time[path.name] = 0.0
            elif (path.access_kind == "read"
                    and diagnostic.first_request_latency_s is not None):
                memory_path_time[path.name] = max(
                    diagnostic.steady_time_s,
                    diagnostic.first_request_latency_s,
                )
            else:
                memory_path_time[path.name] = diagnostic.steady_time_s
            for node in diagnostic.nodes:
                resource = self.hw.resource_by_role(node.resource_role)
                replaces_ingress = (
                    replacing_access_classes
                    and node.resource_role == path.nodes[0].resource_role)
                if node.resource_role != "dram" and not replaces_ingress:
                    t_per_resource[resource.name] += node.capacity_time_s
                covered_memory_roles.add(node.resource_role)
        memory_access_class_diagnostics = solve_memory_access_classes(
            inp.memory_paths,
            self.hw.memory_access_class_services,
            grid_size=inp.grid_size,
        )
        if "dram" in covered_memory_roles:
            read_capacity = self.hw.dram_capacity_for_access(
                "read",
                active_units=inp.active_units,
                sm_freq_hz=inp.sm_freq_hz,
                dram_freq_hz=inp.dram_freq_hz,
            ) * inp.dram_capacity_multiplier
            write_capacity = self.hw.dram_capacity_for_access(
                "write",
                active_units=inp.active_units,
                sm_freq_hz=inp.sm_freq_hz,
                dram_freq_hz=inp.dram_freq_hz,
            ) * inp.dram_capacity_multiplier
            arbitration = solve_memory_arbitration(
                inp.memory_paths,
                tuple(memory_diagnostics),
                resource_role="dram",
                arbitration_mode=self.hw.dram_arbitration_mode,
                read_capacity_per_s=read_capacity,
                write_capacity_per_s=write_capacity,
            )
            dram_resource = self.hw.resource_by_role("dram")
            t_per_resource[dram_resource.name] += (
                arbitration.effective_time_s)
            memory_arbitration_diagnostics.append(arbitration)

        missing_positive_roles = {
            role: demand
            for role, demand in demand_per_resource.items()
            if demand > 0 and role not in self.hw._role_map
        }
        if missing_positive_roles:
            raise ValueError(
                "hardware has no resource for positive demand roles: "
                f"{missing_positive_roles}")

        compute_role_names = {
            "compute_tensor", "compute_vector", "compute_reduction",
            "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
        }
        for resource in self.hw.resources:
            role = resource.role
            if role in compute_role_names or role in covered_memory_roles:
                continue
            if role not in demand_per_resource:
                continue
            demand = demand_per_resource[role]
            if demand <= 0:
                continue
            if role == "dram" and inp.mem_model in (
                    "pim_bank", "systolic_scratchpad"):
                # The explicit compatibility l2_cache path owns its DRAM/DMA
                # envelope.  This is not used by the target production models.
                t_per_resource[resource.name] = 0.0
                continue
            if role == "l2_cache" and inp.mem_model in (
                    "pim_bank", "systolic_scratchpad"):
                t_per_resource[resource.name] = (
                    self._compatibility_memory_time(inp, resource, demand))
                continue
            capacity = resource.effective_capacity(
                inp.active_units,
                sm_freq_hz=inp.sm_freq_hz,
                dram_freq_hz=inp.dram_freq_hz)
            if role == "mte3":
                if inp.mem_model != "npu_tile":
                    raise ValueError(
                        "positive MTE3 demand requires npu_tile memory")
                if resource.total_units <= 0:
                    raise ValueError("MTE3 resource has no physical units")
                active_capacity = (
                    capacity
                    * min(inp.active_units, resource.total_units)
                    / resource.total_units)
                dram_capacity = self.hw.resource_by_role(
                    "dram").effective_capacity(
                        inp.active_units,
                        sm_freq_hz=inp.sm_freq_hz,
                        dram_freq_hz=inp.dram_freq_hz)
                capacity = min(active_capacity, dram_capacity)
            if capacity <= 0:
                raise ValueError(
                    f"resource {resource.name!r} has no capacity")
            t_per_resource[resource.name] += demand / capacity

        unknown_override_roles = (
            set(inp.resource_time_overrides_ns) - set(self.hw._role_map))
        if unknown_override_roles:
            raise ValueError(
                "hardware has no resource for time-override roles: "
                f"{sorted(unknown_override_roles)}")
        for role in inp.resource_time_overrides_ns:
            matching_paths = tuple(
                path_name for path_name, roles in memory_path_roles.items()
                if role in roles)
            if len(matching_paths) > 1:
                raise ValueError(
                    f"aggregate resource override {role!r} reaches multiple "
                    f"memory paths {matching_paths}; a source-backed per-path "
                    "allocation is required")
        for role, duration_ns in inp.resource_time_overrides_ns.items():
            resource = self.hw.resource_by_role(role)
            t_per_resource[resource.name] = duration_ns * 1e-9
            if role in compute_time_by_role:
                compute_time_by_role[role] = duration_ns * 1e-9
        memory_arbitration_diagnostics = [
            dataclasses.replace(
                diagnostic,
                effective_time_s=(
                    inp.resource_time_overrides_ns[
                        diagnostic.resource_role] * 1e-9),
                overridden=True,
            )
            if diagnostic.resource_role in inp.resource_time_overrides_ns
            else diagnostic
            for diagnostic in memory_arbitration_diagnostics
        ]

        t_per_group = {}
        for group in self.hw.resource_groups:
            t_per_group[group.name] = sum(
                t_per_resource[member] for member in group.members)
        access_time_by_service = {}
        access_time_by_path_and_service = {}
        for diagnostic in memory_access_class_diagnostics:
            if diagnostic.service in access_time_by_service:
                access_time_by_service[diagnostic.service] = math.fsum((
                    access_time_by_service[diagnostic.service],
                    diagnostic.service_time_s,
                ))
            else:
                access_time_by_service[diagnostic.service] = (
                    diagnostic.service_time_s)
            path_service = (diagnostic.path_name, diagnostic.service)
            if path_service in access_time_by_path_and_service:
                access_time_by_path_and_service[path_service] = math.fsum((
                    access_time_by_path_and_service[path_service],
                    diagnostic.service_time_s,
                ))
            else:
                access_time_by_path_and_service[path_service] = (
                    diagnostic.service_time_s)
        for service, duration in access_time_by_service.items():
            group_name = f"memory_access:{service}"
            if group_name in t_per_group:
                raise ValueError(
                    "memory access-class service collides with an existing "
                    f"resource group: {group_name!r}")
            t_per_group[group_name] = duration
        resolved_memory_service_time = {}
        dependency_memory_service_time = {}
        overridden_memory_services = set()
        for path_name, duration in memory_path_time.items():
            typed_path_times = tuple(
                typed_duration
                for (typed_path, _service), typed_duration
                in access_time_by_path_and_service.items()
                if typed_path == path_name)
            overridden = tuple(
                role for role in memory_path_roles[path_name]
                if role in inp.resource_time_overrides_ns)
            if overridden:
                duration = max(
                    inp.resource_time_overrides_ns[role] * 1e-9
                    for role in overridden)
                overridden_memory_services.add(f"memory:{path_name}")
            service_name = f"memory:{path_name}"
            resolved_memory_service_time[service_name] = duration
            if overridden or not typed_path_times:
                dependency_memory_service_time[service_name] = duration
            elif path_name in replacing_memory_paths:
                # Replacing classes partition one source-program ingress.
                # Distinct classes are consecutive portions of that program
                # (for example L2-hit and DRAM-miss requests), not parallel
                # alternative ceilings.  Downstream hierarchy service remains
                # an independent lower bound.
                dependency_memory_service_time[service_name] = max(
                    duration, math.fsum(typed_path_times))
            else:
                dependency_memory_service_time[service_name] = max(
                    duration, *typed_path_times)
            t_per_group[service_name] = duration
        dependency_critical_paths = {}
        dependency_service_time = {
            **{
                f"compute:{role}": duration
                for role, duration in compute_time_by_role.items()
            },
            **dependency_memory_service_time,
        }
        if inp.global_store_completion_ns is not None:
            dependency_service_time[GLOBAL_STORE_COMPLETION_SERVICE] = (
                inp.global_store_completion_ns * 1e-9)
        for local_role in ("l1_cache", "smem"):
            if inp.demand_per_resource[local_role] > 0:
                local_resource = self.hw.resource_by_role(local_role)
                dependency_service_time[f"local_memory:{local_role}"] = (
                    t_per_resource[local_resource.name])
        dependency_service_time.update({
            f"memory_latency:{diagnostic.path_name}":
            diagnostic.first_request_latency_s
            for diagnostic in memory_diagnostics
            if diagnostic.first_request_latency_s is not None
        })
        if self.hw.launch_latency_model is not None:
            launch_model = self.hw.launch_latency_model
            if launch_model["model"] == "isolated_nonempty_task_scheduler_v1":
                # Only host submission is a proven predecessor of the first
                # production task body.  CTA descriptor issue and later wave
                # scheduling belong to the complete non-empty-task curve, but
                # can overlap already-issued Load -> Compute -> Store work.
                # The complete curve remains a whole-task lower bound below;
                # serializing it here would charge scheduler tail twice and
                # impose an unsupported global barrier before the first Load.
                launch_submit_ns = launch_model["submit_ns"]
            elif launch_model["model"] == "isolated_empty_task_scheduler_v1":
                # The empty-task geometry tail may overlap production work;
                # only its independently decomposed submission component is
                # ordered before the program path.
                launch_submit_ns = launch_model["submit_ns"]
            else:
                raise ValueError(
                    "unsupported analytical launch dependency model: "
                    f"{launch_model['model']!r}")
            dependency_service_time["launch:submit"] = (
                launch_submit_ns * 1e-9)
        if any(
                MEMORY_DEPENDENCY_TASK_INCREMENT_SERVICE in path.services
                for path in inp.dependency_paths):
            contract = WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR
            if (inp.mem_model != "gpu_mshr"
                    or inp.launch_task_kind != "generic"
                    or not self.hw.has_task_duration_floor_contract(contract)
                    or self.hw.launch_latency_model is None):
                raise ValueError(
                    "memory-dependency task increment requires matching "
                    "GPU task and empty-launch mechanisms")
            service_grid_size = inp.active_units
            service_grid = (service_grid_size, 1, 1)
            increment_ns = self.hw.memory_dependency_task_increment_ns(
                contract,
                service_grid_size,
                inp.block_size,
                service_grid,
                inp.launch_block,
                inp.sm_freq_hz,
            )
            if increment_ns <= 0:
                raise ValueError(
                    "memory-dependency task increment must be positive")
            dependency_service_time[
                MEMORY_DEPENDENCY_TASK_INCREMENT_SERVICE] = (
                    increment_ns * 1e-9)
        for path in inp.dependency_paths:
            unknown_services = (
                set(path.services) - set(dependency_service_time))
            if unknown_services:
                raise ValueError(
                    f"analytical dependency path {path.name!r} references "
                    f"unknown services: {sorted(unknown_services)}")
            group_name = f"dependency:{path.name}"
            if group_name in t_per_group:
                raise ValueError(
                    "analytical dependency path collides with an existing "
                    f"resource group: {group_name!r}")
            concurrent_by_service = {
                service: group
                for group in path.concurrent_service_groups
                for service in group
            }
            serial_counts = {
                service: 1 for service in path.services
            }
            serial_counts.update(dict(path.serial_service_counts))

            priced_groups = set()
            stage_times = []
            critical_services = []
            for service in path.services:
                if service not in concurrent_by_service:
                    stage_times.append(
                        dependency_service_time[service]
                        * serial_counts[service])
                    critical_services.append(service)
                    continue
                concurrent = concurrent_by_service[service]
                if concurrent in priced_groups:
                    continue
                priced_groups.add(concurrent)
                critical = max(
                    concurrent,
                    key=lambda name: (
                        dependency_service_time[name]
                        * serial_counts[name]),
                )
                stage_times.append(
                    dependency_service_time[critical]
                    * serial_counts[critical])
                critical_services.append(critical)
            t_per_group[group_name] = math.fsum(stage_times)
            dependency_critical_paths[group_name] = tuple(
                f"{path.name}/{service}" for service in critical_services)
        if not t_per_group:
            raise RuntimeError("hardware specification has no resource groups")

        pipeline_diagnostic = None
        if inp.pipeline_plan is None:
            execution_time = max(t_per_group.values())
            bottleneck_group = max(
                t_per_group, key=lambda name: t_per_group[name])
        elif isinstance(inp.pipeline_plan, FiaCrossDomainWorkPlan):
            pipeline_diagnostic = self._resolve_fia_cross_domain_work_plan(
                inp.pipeline_plan,
                inp.active_units,
                inp.launch_task_kind,
                inp.resource_time_overrides_ns,
                tuple(memory_diagnostics),
            )
            staged_resources = {
                self.hw.resource_by_role("compute_tensor").name,
                self.hw.resource_by_role("compute_vector").name,
            }
            for group in self.hw.resource_groups:
                t_per_group[group.name] = math.fsum(
                    t_per_resource[member]
                    for member in group.members
                    if member not in staged_resources)
            resource_time = max(t_per_group.values())
            pipeline_time = pipeline_diagnostic.execution_time_s
            execution_time = max(pipeline_time, resource_time)
            t_per_group["pipeline"] = execution_time
            if pipeline_time >= resource_time:
                bottleneck_group = "pipeline"
            else:
                bottleneck_group = max(
                    (name for name in t_per_group if name != "pipeline"),
                    key=lambda name: t_per_group[name])
        elif isinstance(inp.pipeline_plan, FiaSplitFuseWorkPlan):
            pipeline_diagnostic = self._resolve_fia_split_fuse_work_plan(
                inp.pipeline_plan,
                inp.active_units,
                inp.launch_task_kind,
                inp.resource_time_overrides_ns,
                tuple(memory_diagnostics),
            )
            staged_resources = {
                self.hw.resource_by_role("compute_tensor").name,
                self.hw.resource_by_role("compute_vector").name,
                self.hw.resource_by_role("l2_cache").name,
                self.hw.resource_by_role("mte3").name,
            }
            for group in self.hw.resource_groups:
                t_per_group[group.name] = math.fsum(
                    t_per_resource[member]
                    for member in group.members
                    if member not in staged_resources)
            resource_time = max(t_per_group.values())
            pipeline_time = pipeline_diagnostic.execution_time_s
            execution_time = max(pipeline_time, resource_time)
            t_per_group["pipeline"] = execution_time
            if pipeline_time >= resource_time:
                bottleneck_group = "pipeline"
            else:
                bottleneck_group = max(
                    (name for name in t_per_group if name != "pipeline"),
                    key=lambda name: t_per_group[name])
        elif isinstance(inp.pipeline_plan, CannGroupedMatmulEnvelopeDemand):
            pipeline_diagnostic = self._resolve_cann_grouped_matmul_envelope(
                inp.pipeline_plan,
                memory_diagnostics,
                memory_access_class_diagnostics,
                inp.compute_demands,
                inp.active_units,
                inp.launch_task_kind,
            )
            resource_time = max(t_per_group.values())
            pipeline_time = pipeline_diagnostic.execution_time_s
            execution_time = max(pipeline_time, resource_time)
            t_per_group["pipeline"] = execution_time
            if pipeline_time >= resource_time:
                bottleneck_group = "pipeline"
            else:
                bottleneck_group = max(
                    (name for name in t_per_group if name != "pipeline"),
                    key=lambda name: t_per_group[name])
        elif isinstance(inp.pipeline_plan, CannSingleCoreSplitKEnvelopeDemand):
            resolved_pipeline = self._resolve_cann_single_core_split_k_envelope(
                inp.pipeline_plan,
                memory_diagnostics,
                compute_time_by_role,
                inp.active_units,
                inp.launch_task_kind,
            )
            pipeline_diagnostic = evaluate_barrier_pipeline_envelope(
                resolved_pipeline)
            # The compact source envelope already owns AIC MTE2/MTE1/Cube,
            # paired-AIV read/cast/write, and their local queue lifetimes.
            # Retain only independent shared hierarchy bounds outside it;
            # otherwise the same operand bytes are charged once by the
            # source completion and again by aggregate MTE2 throughput.
            staged_resources = {
                self.hw.resource_by_role(role).name
                for role in (
                    "mte2", "compute_tensor", "compute_vector", "mte3")
            }
            for group in self.hw.resource_groups:
                t_per_group[group.name] = math.fsum(
                    t_per_resource[member]
                    for member in group.members
                    if member not in staged_resources)
            for diagnostic in memory_diagnostics:
                if (diagnostic.nodes
                        and diagnostic.nodes[0].resource_role == "mte2"):
                    downstream = tuple(
                        node.capacity_time_s
                        for node in diagnostic.nodes[1:])
                    t_per_group[f"memory:{diagnostic.path_name}"] = (
                        max(downstream) if downstream else 0.0)
            resource_time = max(t_per_group.values())
            pipeline_time = pipeline_diagnostic.execution_time_s
            execution_time = max(pipeline_time, resource_time)
            t_per_group["pipeline"] = execution_time
            if pipeline_time >= resource_time:
                bottleneck_group = "pipeline"
            else:
                bottleneck_group = max(
                    (name for name in t_per_group if name != "pipeline"),
                    key=lambda name: t_per_group[name])
        elif isinstance(
                inp.pipeline_plan,
                GpuGroupedGemmPipelineEnvelopeDemand):
            pipeline_diagnostic = (
                self._resolve_gpu_grouped_gemm_pipeline_envelope(
                    inp.pipeline_plan,
                    inp.active_units,
                    inp.launch_task_kind,
                    launched_ctas=inp.grid_size,
                ))
            resource_time = max(t_per_group.values())
            pipeline_time = pipeline_diagnostic.execution_time_s
            execution_time = max(pipeline_time, resource_time)
            t_per_group["pipeline"] = execution_time
            if pipeline_time >= resource_time:
                bottleneck_group = "pipeline"
            else:
                bottleneck_group = max(
                    (name for name in t_per_group if name != "pipeline"),
                    key=lambda name: t_per_group[name])
        elif isinstance(
                inp.pipeline_plan,
                MetaxGroupedOperandPipelineDemand):
            pipeline_diagnostic = (
                self._resolve_metax_grouped_operand_pipeline_envelope(
                    inp.pipeline_plan,
                    inp.active_units,
                    inp.launch_task_kind,
                    launched_ctas=inp.grid_size,
                ))
            resource_time = max(t_per_group.values())
            pipeline_time = pipeline_diagnostic.execution_time_s
            execution_time = max(pipeline_time, resource_time)
            t_per_group["pipeline"] = execution_time
            if pipeline_time >= resource_time:
                bottleneck_group = "pipeline"
            else:
                bottleneck_group = max(
                    (name for name in t_per_group if name != "pipeline"),
                    key=lambda name: t_per_group[name])
        else:
            resolved_pipeline = self._resolve_pipeline(
                inp.pipeline_plan,
                compute_time_by_role,
                memory_diagnostics,
                resolved_memory_service_time,
                memory_access_class_diagnostics,
                overridden_memory_services,
                t_per_resource,
                inp.compute_demands,
                inp.active_units,
                inp.launch_task_kind,
            )
            if isinstance(resolved_pipeline, AttentionPipelineEnvelope):
                pipeline_diagnostic = evaluate_attention_pipeline_envelope(
                    resolved_pipeline)
            elif isinstance(resolved_pipeline, BarrierPipelineEnvelope):
                pipeline_diagnostic = evaluate_barrier_pipeline_envelope(
                    resolved_pipeline)
            else:
                pipeline_diagnostic = evaluate_pipeline(resolved_pipeline)
            if inp.competing_pipeline_floor is not None:
                resolved_floor = self._resolve_pipeline(
                    inp.competing_pipeline_floor,
                    compute_time_by_role,
                    memory_diagnostics,
                    resolved_memory_service_time,
                    memory_access_class_diagnostics,
                    overridden_memory_services,
                    t_per_resource,
                    inp.compute_demands,
                    inp.active_units,
                    inp.launch_task_kind,
                    allow_partial_worker_services=True,
                )
                if not isinstance(resolved_floor, PipelinePlan):
                    raise TypeError(
                        "competing pipeline floor must resolve to a "
                        "PipelinePlan")
                floor_diagnostic = evaluate_pipeline(resolved_floor)
                if (floor_diagnostic.execution_time_s
                        > pipeline_diagnostic.execution_time_s):
                    pipeline_diagnostic = floor_diagnostic
            resource_time = max(t_per_group.values())
            pipeline_time = pipeline_diagnostic.execution_time_s
            execution_time = max(pipeline_time, resource_time)
            t_per_group["pipeline"] = execution_time
            if pipeline_time >= resource_time:
                bottleneck_group = "pipeline"
            else:
                bottleneck_group = max(
                    (name for name in t_per_group if name != "pipeline"),
                    key=lambda name: t_per_group[name])

        if inp.execution_time_floor_ns is not None:
            execution_floor_s = inp.execution_time_floor_ns * 1e-9
            if execution_floor_s > execution_time:
                execution_time = execution_floor_s
                bottleneck_group = "analytical_execution_floor"
            t_per_group["analytical_execution_floor"] = execution_floor_s

        aiv_ordered_pipeline_completion_s = None
        aiv_descriptor_interval_s = None
        aiv_pipeline_worker_makespan = None
        aiv_descriptor_prologue_pipelined = False
        task_prologue_s = 0.0
        resident_cta_cohort_refill_s = 0.0
        if inp.include_dispatch_overhead:
            runtime_launch_scheduler = (
                inp.runtime_task_class is not None
                and self.hw.runtime_task_class_is_launch_scheduler(
                    inp.runtime_task_class))
            if runtime_launch_scheduler:
                dispatch_s = self.hw.runtime_task_class_duration_ns(
                    inp.runtime_task_class, inp.runtime_task_elements,
                    inp.grid_size, inp.block_size) * 1e-9
            else:
                dispatch_s = self.hw.dispatch_overhead_ns * 1e-9
            # Measured cost of dispatching this launch's CTAs.  The launch
            # model bills a full-chip round, so CTA count inside one wave is
            # free there, and a geometry-controlled probe finds the real cost
            # independent of block size and dynamic shared memory and additive
            # rather than hidden by execution.  A CTA cannot begin before it
            # is dispatched, so the cost is ordered before the program.  Cards
            # publishing no measurement return zero and are unaffected.
            dispatch_s = math.fsum((
                dispatch_s,
                self.hw.cta_dispatch_cost_ns(inp.grid_size) * 1e-9,
            ))
            if (inp.task_duration_floor_contract
                    == AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM):
                dispatch_s = self.hw.empty_kernel_duration_for_task(
                    inp.launch_task_kind,
                    inp.grid_size,
                    inp.block_size,
                    inp.launch_grid,
                    inp.launch_block,
                ) * 1e-9
            if (not runtime_launch_scheduler
                    and inp.pipeline_plan is not None
                    and not isinstance(
                        inp.pipeline_plan,
                        (GpuGroupedGemmPipelineEnvelopeDemand,
                         MetaxGroupedOperandPipelineDemand))
                    and self.hw.launch_latency_model is not None
                    and self.hw.launch_latency_model["model"] in (
                        "isolated_empty_task_scheduler_v1",
                        "isolated_nonempty_task_scheduler_v1")):
                # A source-closed worker pipeline replaces the aggregate
                # dependency path, so order the independently decomposed host
                # submission prefix here.  Descriptor issue and later wave
                # scheduling remain only in the complete-task floor below;
                # they may overlap work from CTAs that are already resident.
                if (inp.task_duration_floor_contract
                        == GPU_SOURCE_PIPELINE_OUTER_ENVELOPE):
                    dispatch_s = math.fsum((
                        dispatch_s,
                        self.hw.fa2_source_task_outer_envelope_ns(
                            inp.grid_size, inp.block_size) * 1e-9,
                    ))
                else:
                    dispatch_s = math.fsum((
                        dispatch_s,
                        self.hw.launch_latency_model["submit_ns"] * 1e-9,
                    ))
            if isinstance(inp.pipeline_plan, FiaCrossDomainWorkPlan):
                if inp.launch_task_kind != "mixed":
                    raise ValueError(
                        "FIA cross-domain work requires a mixed launch task")
                # The regular-FIA mixed-core probe measures the complete
                # device interval and its simultaneous internal
                # InitOutput+SyncAll interval.  Their difference is therefore
                # ordered outside, rather than overlapping or replacing, the
                # regular FIA source-closed DAG.  The mixed-empty complete-
                # task curve below remains only a lower bound.  SplitFuse has
                # a distinct source program and does not run this sequence.
                active_aiv_workers = (
                    2 * inp.pipeline_plan.active_group_count
                    if isinstance(inp.pipeline_plan, FiaCrossDomainWorkPlan)
                    else 2 * inp.pipeline_plan.aic_worker_count)
                dispatch_s = math.fsum((
                    dispatch_s,
                    self.hw.fia_mixed_task_outer_envelope_time_ns(
                        active_aiv_workers) * 1e-9,
                ))
            if (inp.launch_task_kind == "aiv"
                    and inp.task_duration_floor_contract
                    == AIV_SOURCE_PIPELINE_LAUNCH_ONLY):
                # The source DAG and its explicit prologue account for the
                # task body.  An explicit first-instruction worker-fill
                # contract supplies the launch-wide admission span directly;
                # otherwise retain the legacy independently measured common
                # submission prefix.  The fitted complete-task intercept must
                # never be added when the explicit worker fill is selected.
                if inp.aiv_worker_start_contract is None:
                    if self.hw.aiv_nonempty_task_latency_model is None:
                        raise ValueError(
                            "AIV source pipeline requires an independently "
                            "measured task submission prefix")
                    dispatch_s = math.fsum((
                        dispatch_s,
                        self.hw.aiv_nonempty_task_latency_model["submit_ns"]
                        * 1e-9,
                    ))
            elif (inp.launch_task_kind == "aiv"
                    and inp.task_duration_floor_contract not in (
                        AIV_COMPLETE_TASK_FLOOR_ONLY,
                        AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
                        AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM)
                    and self.hw.aiv_nonempty_task_latency_model is not None):
                # Legacy AIV pipelines retain their existing decomposed
                # scheduler composition.  A source-closed caller can instead
                # select the complete-task-only contract when the fitted
                # benchmark coefficients are not independent descriptor
                # observations.
                scheduler = self.hw.aiv_nonempty_task_latency_model
                dispatch_s = math.fsum((
                    dispatch_s,
                    scheduler["submit_ns"] * 1e-9,
                ))
                if pipeline_diagnostic is not None:
                    worker_makespan = (
                        pipeline_diagnostic.worker_makespan_s)
                    expected_workers = tuple(
                        str(worker) for worker in range(inp.active_units))
                    if tuple(worker_makespan) != expected_workers:
                        raise ValueError(
                            "AIV pipeline workers must be contiguous and "
                            "ordered like launch descriptors")
                    descriptor_s = (
                        scheduler["descriptor_ns_per_aiv"] * 1e-9)
                    aiv_descriptor_interval_s = descriptor_s
                    aiv_pipeline_worker_makespan = worker_makespan
                    aiv_ordered_pipeline_completion_s = max(
                        dispatch_s + (worker + 1) * descriptor_s
                        + worker_makespan[str(worker)]
                        for worker in range(inp.active_units)
                    )
            if inp.aiv_worker_start_contract == AIV_WORKER_START_FILL:
                if pipeline_diagnostic is None:
                    raise ValueError(
                        "AIV worker-start fill requires pipeline diagnostics")
                worker_fill_s = self.hw.aiv_worker_start_fill_ns(
                    inp.active_units) * 1e-9
                # The device counter observes a launch-wide admission span,
                # relative to the first AIV entry.  It is neither a host
                # submission prefix nor the slope of the complete-task floor.
                # Row-group workers are balanced except for one shorter tail,
                # so the final full worker exposes the launch fill once while
                # earlier workers execute concurrently.
                aiv_ordered_pipeline_completion_s = math.fsum((
                    execution_time,
                    worker_fill_s,
                ))
            if inp.aiv_source_entry_fill_contract == AIV_SOURCE_ENTRY_FILL:
                if (pipeline_diagnostic is None
                        or aiv_ordered_pipeline_completion_s is None):
                    raise ValueError(
                        "AIV source-entry fill requires legacy descriptor "
                        "pipeline completion")
                source_entry_fill_s = self.hw.aiv_worker_start_fill_ns(
                    inp.active_units) * 1e-9
                # The source entry span begins only after the task scheduler
                # has made each AIV descriptor ready.  It is therefore a
                # single internal prefix after, not a replacement for, the
                # outer descriptor schedule.  The one-tile q2 source program
                # gives every worker the same first CopyIn boundary.
                aiv_ordered_pipeline_completion_s = math.fsum((
                    aiv_ordered_pipeline_completion_s,
                    source_entry_fill_s,
                ))
            if inp.task_prologue_contract is not None:
                task_prologue_s = (
                    inp.task_prologue_repetitions
                    * self.hw.task_prologue_duration_ns(
                        inp.task_prologue_contract, inp.active_units)
                    * 1e-9)
                if (aiv_descriptor_interval_s is not None
                        and aiv_pipeline_worker_makespan is not None
                        and inp.aiv_descriptor_prologue_pipeline
                        and inp.task_prologue_repetitions == 1
                        and inp.task_duration_floor_contract is None):
                    # Descriptor admission and the per-AIV typed-tiling
                    # materialization form a two-stage launch pipeline.  The
                    # independently measured prologue curve is the completion
                    # of the active worker set, not a second launch-wide span
                    # after every descriptor has already been admitted.
                    # Reconstruct its exact incremental completion curve and
                    # compose it with descriptor readiness in source order.
                    prologue_ready_s = None
                    pipelined_completion_s = 0.0
                    previous_prologue_s = 0.0
                    for worker in range(inp.active_units):
                        cumulative_prologue_s = (
                            self.hw.task_prologue_duration_ns(
                                inp.task_prologue_contract, worker + 1)
                            * 1e-9)
                        prologue_increment_s = (
                            cumulative_prologue_s - previous_prologue_s)
                        if prologue_increment_s < 0:
                            raise ValueError(
                                "AIV task prologue completion curve must be "
                                "monotonic")
                        descriptor_ready_s = (
                            dispatch_s
                            + (worker + 1) * aiv_descriptor_interval_s)
                        prologue_ready_s = math.fsum((
                            descriptor_ready_s
                            if prologue_ready_s is None else max(
                                descriptor_ready_s, prologue_ready_s),
                            prologue_increment_s,
                        ))
                        pipelined_completion_s = max(
                            pipelined_completion_s,
                            prologue_ready_s
                            + aiv_pipeline_worker_makespan[str(worker)],
                        )
                        previous_prologue_s = cumulative_prologue_s
                    aiv_ordered_pipeline_completion_s = (
                        pipelined_completion_s)
                    aiv_descriptor_prologue_pipelined = True
            if inp.resident_cta_cohort_refills is not None:
                resident_cta_cohort_refill_s = (
                    inp.resident_cta_cohort_refills
                    * self.hw.resident_cta_cohort_refill_ns(
                        inp.active_units, inp.block_size)
                    * 1e-9)
            if (inp.task_duration_floor_contract
                    == AIV_SOURCE_PIPELINE_LAUNCH_ONLY):
                task_floor_s = 0.0
            else:
                task_floor_ns = self.hw.task_duration_floor_for_task(
                    inp.launch_task_kind, inp.grid_size, inp.block_size,
                    inp.launch_grid, inp.launch_block,
                    task_duration_floor_contract=(
                        inp.task_duration_floor_contract),
                    sm_freq_hz=inp.sm_freq_hz,
                    runtime_task_class=(
                        None if runtime_launch_scheduler
                        else inp.runtime_task_class),
                    runtime_task_elements=(
                        None if runtime_launch_scheduler
                        else inp.runtime_task_elements))
                task_floor_s = task_floor_ns * 1e-9
        else:
            dispatch_s = 0.0
            task_floor_s = 0.0
        launch_and_prologue_s = math.fsum((
            dispatch_s,
            task_prologue_s,
            resident_cta_cohort_refill_s,
            (inp.finite_mixed_operand_stream_startup_ns or 0.0) * 1e-9,
        ))
        dispatched_execution = (
            dispatch_s + resident_cta_cohort_refill_s + execution_time
            if aiv_descriptor_prologue_pipelined
            else launch_and_prologue_s + execution_time)
        if aiv_ordered_pipeline_completion_s is not None:
            ordered_prologue_s = (
                0.0
                if (inp.task_duration_floor_contract
                    in (
                        AIV_EMPTY_TASK_SHELL_COMPETING_SOURCE_FLOOR,
                        AIV_SOURCE_PIPELINE_LAUNCH_ONLY)
                    or aiv_descriptor_prologue_pipelined)
                else task_prologue_s
            )
            # A source pipeline's launch-wide first-instruction fill and its
            # complete tiling-materialization phase start from the same task
            # admission boundary.  They are competing prefixes, not two
            # serial phases.  Complete-floor callers retain the legacy
            # ordered prologue composition.
            dispatched_execution = max(
                dispatched_execution,
                aiv_ordered_pipeline_completion_s + ordered_prologue_s,
            )
        # CTAs that start, reject their tile and return still occupy their
        # AP's resident slots, in series with the real work that AP does, so
        # their measured cost adds to the dispatched execution rather than
        # composing with it by maximum.
        dispatched_execution += (
            inp.early_exit_cta_retire_ns or 0.0) * 1e-9
        span_floor_s = _resolve_small_kernel_span_floor(self.hw, inp)
        total_time = max(task_floor_s, dispatched_execution, span_floor_s)
        if (span_floor_s >= dispatched_execution
                and span_floor_s >= task_floor_s and span_floor_s > 0):
            bottleneck_group = "launch"
        elif task_floor_s >= dispatched_execution and task_floor_s > 0:
            bottleneck_group = "launch"
        elif (launch_and_prologue_s > execution_time
              and launch_and_prologue_s > 0):
            bottleneck_group = "launch"

        if bottleneck_group in dependency_critical_paths:
            critical_path = dependency_critical_paths[bottleneck_group]
        elif (pipeline_diagnostic is not None
                and bottleneck_group == "pipeline"):
            critical_path = pipeline_diagnostic.critical_path
        else:
            critical_path = (bottleneck_group,)
        diagnostics = PredictionDiagnostics(
            compute_time_s=compute_time_by_role,
            memory=tuple(memory_diagnostics),
            memory_access_classes=memory_access_class_diagnostics,
            memory_arbitration=tuple(memory_arbitration_diagnostics),
            pipeline=pipeline_diagnostic,
            critical_path=critical_path,
            support_level=inp.support_level,
            limitations=inp.limitations,
        )
        return KernelPrediction(
            t_total_s=total_time,
            t_per_group=t_per_group,
            t_launch_s=launch_and_prologue_s,
            bottleneck_group=bottleneck_group,
            t_per_resource=t_per_resource,
            diagnostics=diagnostics,
        )
