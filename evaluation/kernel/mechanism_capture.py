"""Mechanism capture; extracted without changing the established contracts."""

from __future__ import annotations

from evaluation.json_shape import file_sha256 as sha256

from evaluation.json_shape import file_sha256
from dataclasses import fields, is_dataclass
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping
from evaluation.kernel.dataset import ATTENTION_OP_TYPES, MOE_OP_TYPES, VECTOR_OP_TYPES, to_float
from evaluation.kernel.oracle_observation import single_counterfactual_inputs
from src.kernel.kernel_base import create_predictor
from src.kernel.model_contract import (
    AttentionPipelineEnvelopeDemand,
    BarrierPipelineEnvelopeDemand,
    CannGroupedMatmulEnvelopeDemand,
    CannSingleCoreSplitKEnvelopeDemand,
    FiaCrossDomainWorkPlan,
    FiaSplitFuseWorkPlan,
    GpuGroupedGemmPipelineEnvelopeDemand,
    MetaxGroupedOperandPipelineDemand,
    PipelineWorkPlan,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "experiment_logs/evaluation/kernel/mechanism_validation"
)


HARDWARE = (
    ("nvidia_a800", "NVIDIA A800"),
    ("metax_c500", "MetaX C500"),
    ("ascend_910b", "Ascend 910B"),
)


DATASET_FAMILIES = ("gemm", "flash_attn", "fused_moe", "vector")


FAMILY_LABELS = ("GEMM", "FlashAttention", "FusedMoE", "Vector")


VECTOR_TYPES = VECTOR_OP_TYPES


DEMAND_ROLES = (
    "compute_tensor", "compute_vector", "compute_reduction", "compute_fma",
    "compute_sfu", "compute_alu", "warp_issue", "l1_cache", "smem",
    "xbar", "l2_cache", "dram", "mte3",
)


TIME_ROLES = (
    "compute_tensor", "compute_vector", "compute_reduction", "compute_fma",
    "compute_sfu", "compute_alu", "warp_issue", "l1_cache", "xbar",
    "l2_cache", "dram", "mte3", "scalar_aiv",
)


NVIDIA_L2_SECTOR_BYTES = 32.0


DIRECTIONAL_TRAFFIC_COMPONENTS = (
    "traffic:l2c_source_read",
    "traffic:downstream_global_read",
    "traffic:downstream_global_write",
    "traffic:xbar_read",
    "traffic:xbar_write",
    "traffic:l2_source_read",
    "traffic:l2_source_write",
    "traffic:dram_read_in_window",
    "traffic:dram_writeback_in_window",
    "traffic:aic_mte2_gm_to_l1",
    "traffic:aiv_mte2_gm_to_ub",
    "traffic:aiv_mte3_ub_to_gm",
    "traffic:dram_read_derived",
)


PROFILER_EVIDENCE_KINDS = frozenset({"profiler", "profiler_derived"})


BUCKET_SCHEMA_VERSION = "physical_bucket_v1"


TASK_AGGREGATION_CONTRACT = "serial_sum_of_forward_kernel_predictions_v2"


BUCKET_MAPE_LIMIT_PCT = 15.0


CASE_APE_LIMIT_PCT = 30.0


SUPPORT_LEVELS = (
    "complete",
    "partial",
    "aggregate_only",
    "compatibility",
)


COMPARISON_STATUSES = (
    "comparable",
    "not_comparable",
    "unobserved",
)


CURRENT_DATASET_COMPARISON_REASON = (
    "capture_software_source_and_binary_identity_not_recorded"
)


PHYSICAL_BUCKET_DISCRETE_CONTROLS = (
    "aiv_mixed",
    "group_size_m",
    "operand_replay_a",
    "operand_replay_b",
    "expert_parallel_size",
    "cann_tiling_key",
)


SEVERE_DURATION_FIELDS = (
    "result_scope",
    "hardware_id",
    "family",
    "case_id",
    "kernel_ordinal",
    "kernel_kind",
    "variant",
    "kernel_count",
    "timing_source",
    "predicted_ns",
    "observed_ns",
    "absolute_duration_error_ns",
    "signed_error_pct",
    "absolute_error_pct",
    "physical_bucket_id",
    "comparison_status",
    "comparison_reason",
)


CASE_IMPACT_FIELDS = (
    "impact_rank",
    "result_scope",
    "hardware_id",
    "family",
    "case_id",
    "kernel_ordinal",
    "kernel_kind",
    "variant_or_execution_graph",
    "kernel_count",
    "timing_source",
    "measured_duration_field",
    "measured_ns",
    "predicted_ns",
    "absolute_duration_error_ns",
    "signed_error_pct",
    "absolute_error_pct",
    "case_ape_gate_passed",
    "physical_bucket_id",
    "bucket_mape_pct",
    "bucket_mape_gate_passed",
    "bucket_max_ape_pct",
    "bucket_max_ape_gate_passed",
    "support_level",
    "comparison_status",
    "comparison_reason",
)


C500_SILU_MECHANISM_FIELDS = (
    "result_scope",
    "case_id",
    "tokens",
    "intermediate",
    "production_duration_ns",
    "timing_source",
    "predicted_duration_ns",
    "predicted_over_production",
    "absolute_error_pct",
    "grid_size",
    "waves",
    "predicted_instruction_lower_bound",
    "observed_total_instructions",
    "instruction_predicted_over_observed",
    "instruction_comparison_status",
    "instruction_comparison_reason",
    "predicted_global_bytes",
    "observed_global_bytes",
    "traffic_predicted_over_observed",
    "instrumented_duration_ns",
    "instrumented_core_clock_ghz",
    "instrumented_ap_active_cycles",
    "instrumented_mean_active_aps",
    "instrumented_instruction_per_active_cycle",
    "instrumented_average_wave_life_cycles",
    "instrumented_dnoc_read_latency_cycles",
    "instrumented_vls_pipeline_stall_cycles",
    "instrumented_vls_wdata_stall_cycles",
    "timing_scope_note",
)


BEFORE_AFTER_CASE_FIELDS = (
    "result_scope",
    "hardware_id",
    "family",
    "case_id",
    "kernel_ordinal",
    "kernel_kind",
    "before_variant_or_execution_graph",
    "variant_or_execution_graph",
    "before_model_identity",
    "after_model_identity",
    "model_identity_changed",
    "timing_source",
    "measured_ns",
    "before_predicted_ns",
    "after_predicted_ns",
    "prediction_change_ns",
    "before_signed_error_pct",
    "after_signed_error_pct",
    "before_absolute_error_pct",
    "after_absolute_error_pct",
    "ape_change_pct_points",
    "before_absolute_duration_error_ns",
    "after_absolute_duration_error_ns",
    "absolute_duration_error_change_ns",
    "before_case_gate_passed",
    "after_case_gate_passed",
    "before_support_level",
    "after_support_level",
    "support_level_changed",
    "bucket_schema",
    "physical_bucket_id",
    "physical_bucket_short_id",
    "bucket_dimensions",
    "bucket_basis",
    "comparison_status",
    "comparison_reason",
)


BEFORE_AFTER_BUCKET_FIELDS = (
    "result_scope",
    "hardware_id",
    "family",
    "physical_bucket_id",
    "physical_bucket_short_id",
    "bucket_schema",
    "bucket_dimensions",
    "bucket_basis",
    "row_count",
    "comparable_count",
    "not_comparable_count",
    "unobserved_count",
    "comparison_coverage_pct",
    "comparison_coverage_gate_passed",
    "before_mape_pct",
    "after_mape_pct",
    "mape_change_pct_points",
    "before_mdape_pct",
    "after_mdape_pct",
    "before_p90_ape_pct",
    "after_p90_ape_pct",
    "before_p99_ape_pct",
    "after_p99_ape_pct",
    "before_max_ape_pct",
    "after_max_ape_pct",
    "max_ape_change_pct_points",
    "before_mean_signed_error_pct",
    "after_mean_signed_error_pct",
    "before_median_signed_error_pct",
    "after_median_signed_error_pct",
    "before_case_violation_count",
    "after_case_violation_count",
    "before_diagnostic_mape_gate_passed",
    "after_diagnostic_mape_gate_passed",
    "before_diagnostic_max_ape_gate_passed",
    "after_diagnostic_max_ape_gate_passed",
    "before_diagnostic_accuracy_gate_passed",
    "after_diagnostic_accuracy_gate_passed",
    "before_mape_gate_passed",
    "after_mape_gate_passed",
    "before_max_ape_gate_passed",
    "after_max_ape_gate_passed",
    "before_accuracy_gate_passed",
    "after_accuracy_gate_passed",
)


BEFORE_AFTER_SUMMARY_FIELDS = (
    "result_scope",
    "hardware_id",
    "family",
    "row_count",
    "before_mape_pct",
    "after_mape_pct",
    "mape_change_pct_points",
    "before_mdape_pct",
    "after_mdape_pct",
    "before_p90_ape_pct",
    "after_p90_ape_pct",
    "before_p99_ape_pct",
    "after_p99_ape_pct",
    "before_max_ape_pct",
    "after_max_ape_pct",
    "before_mean_signed_error_pct",
    "after_mean_signed_error_pct",
    "before_median_signed_error_pct",
    "after_median_signed_error_pct",
    "before_case_pass_count",
    "after_case_pass_count",
    "before_case_violation_count",
    "after_case_violation_count",
)


SEMANTICALLY_EXCLUDED_COMPONENTS = frozenset({
    ("ascend_910b", "demand:compute_tensor"),
    ("ascend_910b", "demand:compute_vector"),
    ("ascend_910b", "demand:compute_reduction"),
    ("ascend_910b", "demand:compute_fma"),
    ("ascend_910b", "demand:compute_sfu"),
    ("ascend_910b", "demand:compute_alu"),
    ("ascend_910b", "demand:warp_issue"),
    # mcProfiler's l2c_memacs is an upstream read-side request count, whereas
    # the production xbar/L2 roles include both read and write nodes.  The
    # downstream all_memacs counter is read+write at a different transaction
    # boundary.  Neither is a like-for-like observation of the aggregate
    # xbar/L2 demand, so compare the directional memory paths instead.
    ("metax_c500", "demand:xbar"),
    ("metax_c500", "demand:l2_cache"),
    # The native directional counters are retained separately below.  These
    # aggregate rows would otherwise hide read/write mistakes and, for L2,
    # compare against a service envelope rather than one counter interface.
    ("nvidia_a800", "demand:l2_cache"),
    ("nvidia_a800", "demand:dram"),
    # NPU MTE2 reads, MTE3 writes, and DRAM-read misses are distinct paths.
    # Flat xbar/L2/DRAM/MTE3 replacement loses that direction and can inject a
    # positive GPU-shaped xbar role which the npu_tile contract intentionally
    # leaves at zero.
    ("ascend_910b", "demand:xbar"),
    ("ascend_910b", "demand:l2_cache"),
    ("ascend_910b", "demand:dram"),
    ("ascend_910b", "demand:mte3"),
})


def is_semantically_comparable_component(
        hardware_id: str, family: str, variant: str,
        component: str, profiler_op_type: str | None) -> bool:
    if (hardware_id, component) in SEMANTICALLY_EXCLUDED_COMPONENTS:
        return False
    if hardware_id == "ascend_910b":
        if profiler_op_type not in ("cube", "vector", "mix"):
            raise ValueError(
                "Ascend semantic comparison requires a known profiler "
                f"Op Type, got {profiler_op_type!r}")
        if profiler_op_type == "mix" and component == "demand:dram":
            # Historical canonical mixed-domain records applied one mean AIC
            # hit rate to the combined AIC+AIV request volume.  The per-row
            # join needed to reconstruct DRAM misses was not retained, so the
            # resulting aggregate is not a valid comparison target.
            return False
        if (family == "FusedMoE"
                and variant in ("w1", "w2")
                and component in ("demand:l2_cache", "demand:dram")):
            # Forward now exposes separate GroupedMatmul read and output
            # interfaces.  The generic aggregate component does not have that
            # split and therefore is not the matching validation boundary.
            return False
    return True


def family(op_type: str) -> str:
    if op_type == "GEMM":
        return "GEMM"
    elif op_type in ATTENTION_OP_TYPES:
        return "FlashAttention"
    elif op_type in MOE_OP_TYPES:
        return "FusedMoE"
    elif op_type in VECTOR_TYPES:
        return "Vector"
    else:
        raise ValueError(f"unsupported mechanism-validation op type: {op_type!r}")


def canonical_capture_comparison_evidence(
        hardware_id: str) -> Mapping[str, str]:
    """Return the fail-closed status of current timing and counter captures.

    All three canonical hardware datasets preserve a timer source and exact
    kernel ordinal, and most preserve profiler counters.  None records the
    capture-time runtime, source snapshot, runner binary, and operator binary
    identities needed to bind those observations to the current production
    Forward.  A missing identity is ``unobserved``; it is not evidence of a
    version mismatch and must not be called ``not_comparable``.
    """

    if hardware_id not in {item[0] for item in HARDWARE}:
        raise ValueError(
            f"unknown duration-comparison hardware: {hardware_id!r}")
    return {
        "comparison_status": "unobserved",
        "comparison_reason": CURRENT_DATASET_COMPARISON_REASON,
    }


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(name): json_value(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if hasattr(value, "value"):
        return json_value(value.value)
    raise TypeError(
        f"mechanism report cannot serialize {type(value).__name__}")




def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def lfs_oids() -> Mapping[str, str]:
    output = git_output("lfs", "ls-files", "-l")
    identities = {}
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"unexpected git-lfs identity line: {line!r}")
        oid, marker, path = parts
        if marker not in ("*", "-"):
            raise ValueError(f"unexpected git-lfs worktree marker: {line!r}")
        if path.startswith("kernel_dataset/") or path.startswith("launch_data/"):
            identities[path] = oid
    return identities


def tiling_summary(tiling) -> Mapping[str, Any] | None:
    if tiling is None:
        return None
    return {
        "tile_shape": dict(tiling.tile_shape),
        "pipeline_depth": tiling.pipeline_depth,
        "split_k": tiling.split_k,
        "num_kv_splits": tiling.num_kv_splits,
    }


def launch_summary(launch) -> Mapping[str, Any]:
    return {
        "algorithm": launch.algorithm,
        "grid": list(launch.grid),
        "grid_size": launch.grid_size,
        "block": list(launch.block),
        "block_size": launch.block_size,
        "tiling": tiling_summary(launch.tiling),
        "cluster": None if launch.cluster is None else list(launch.cluster),
        "parameters": dict(launch.parameters),
        "execution_variant": launch.execution_variant,
        "source_program_type": (
            None if launch.source_program is None
            else type(launch.source_program).__name__),
        "source_program": (
            None if launch.source_program is None
            else json_value(launch.source_program)),
    }


def compute_demand_summary(prediction_input) -> list[Mapping[str, Any]]:
    result = []
    for demand in prediction_input.compute_demands:
        services = []
        for service in demand.service_demands:
            services.append({
                "service": service.service,
                "operations": service.operations,
                "serial_invocations": service.serial_invocations,
                "worker_count": len(service.operations_by_worker),
                "max_worker_operations": (
                    max(service.operations_by_worker)
                    if service.operations_by_worker else None),
            })
        exact_requests = []
        for request in demand.exact_request_demands:
            request_counts = tuple(
                sum(count for _, count in worker_runs)
                for worker_runs in request.request_size_runs)
            exact_requests.append({
                "service": request.service,
                "operations": request.operations,
                "worker_count": request.worker_count,
                "request_count": sum(request_counts),
                "max_worker_request_count": max(request_counts),
            })
        result.append({
            "resource_role": demand.resource_role,
            "operations": demand.operations,
            "shape_efficiency": demand.shape_efficiency,
            "schedule_efficiency": demand.schedule_efficiency,
            "service_demands": services,
            "exact_request_demands": exact_requests,
        })
    return result


def memory_path_summary(prediction_input) -> list[Mapping[str, Any]]:
    result = []
    for path in prediction_input.memory_paths:
        result.append({
            "name": path.name,
            "access_kind": path.access_kind,
            "rate_unit": path.rate_unit,
            "initial_transactions": path.initial_transactions,
            "nodes": [
                {
                    "resource_role": node.resource_role,
                    "transactions": node.transactions,
                    "bytes": node.bytes,
                    "transaction_bytes": node.transaction_bytes,
                }
                for node in path.nodes
            ],
            "source": {
                "active_units": path.source.active_units,
                "producers_per_unit": path.source.producers_per_unit,
                "independent_requests_per_producer": (
                    path.source.independent_requests_per_producer),
                "frontend_rate_per_unit_s": path.source.frontend_rate_per_unit_s,
                "workload_slots_per_unit": path.source.workload_slots_per_unit,
            },
            "queues": [
                {
                    "name": queue.name,
                    "allocations_per_initial_request": (
                        queue.allocations_per_initial_request),
                    "lifetime_weights": [list(item) for item in queue.lifetime_weights],
                }
                for queue in path.queues
            ],
            "access_classes": [json_value(value) for value in path.access_class_demands],
        })
    return result


def pipeline_input_summary(prediction_input) -> Mapping[str, Any] | None:
    plan = prediction_input.pipeline_plan
    if plan is None:
        return None
    # The report schema uses null for fields absent from a supported plan's
    # contract. Unknown plan types must not silently produce an empty summary.
    worker_count = None
    tasks_count = None
    buffers_count = None
    if isinstance(plan, PipelineWorkPlan):
        worker_count = plan.worker_count
        tasks_count = len(plan.tasks)
        buffers_count = len(plan.buffers)
    elif isinstance(plan, BarrierPipelineEnvelopeDemand):
        worker_count = plan.worker_count
        buffers_count = len(plan.buffers)
    elif isinstance(plan, AttentionPipelineEnvelopeDemand):
        buffers_count = len(plan.buffers)
    elif isinstance(plan, (
            CannGroupedMatmulEnvelopeDemand, CannSingleCoreSplitKEnvelopeDemand)):
        worker_count = plan.worker_count
    elif isinstance(plan, (
            GpuGroupedGemmPipelineEnvelopeDemand, MetaxGroupedOperandPipelineDemand,
            FiaCrossDomainWorkPlan, FiaSplitFuseWorkPlan)):
        pass
    else:
        raise TypeError(f"unsupported pipeline plan type: {type(plan).__name__}")
    return {
        "type": type(plan).__name__,
        "worker_count": worker_count,
        "active_aic_units": None,
        "active_aiv_units": None,
        "resources_encoded_count": None,
        "tasks_encoded_count": tasks_count,
        "buffers_encoded_count": buffers_count,
        "aic_groups_encoded_count": None,
    }


def prediction_input_summary(prediction_input) -> Mapping[str, Any]:
    return {
        "demand_per_resource": dict(prediction_input.demand_per_resource),
        "eta": prediction_input.eta,
        "wave_efficiency": prediction_input.wave_efficiency,
        "active_units": prediction_input.active_units,
        "sm_freq_hz": prediction_input.sm_freq_hz,
        "dram_freq_hz": prediction_input.dram_freq_hz,
        "grid_size": prediction_input.grid_size,
        "block_size": prediction_input.block_size,
        "variant": prediction_input.variant,
        "mem_model": prediction_input.mem_model,
        "compute_demands": compute_demand_summary(prediction_input),
        "memory_paths": memory_path_summary(prediction_input),
        "pipeline_plan": pipeline_input_summary(prediction_input),
        "phase_ledger_present": prediction_input.phase_ledger is not None,
        "dram_capacity_multiplier": prediction_input.dram_capacity_multiplier,
        "include_dispatch_overhead": prediction_input.include_dispatch_overhead,
        "support_level": prediction_input.support_level,
        "limitations": list(prediction_input.limitations),
        "resource_time_overrides_ns": dict(
            prediction_input.resource_time_overrides_ns),
        "launch_task_kind": prediction_input.launch_task_kind,
        "task_duration_floor_contract": (
            prediction_input.task_duration_floor_contract),
        "execution_time_floor_ns": prediction_input.execution_time_floor_ns,
    }


def prediction_summary(prediction, hardware) -> Mapping[str, Any]:
    diagnostics = prediction.diagnostics
    if diagnostics is None:
        raise RuntimeError("mechanism validation requires Forward diagnostics")
    resource_time_by_role_ns = {}
    for role in TIME_ROLES:
        if role in hardware._role_map:
            resource_name = hardware.resource_by_role(role).name
            if resource_name in prediction.t_per_resource:
                resource_time_by_role_ns[role] = (
                    prediction.t_per_resource[resource_name] * 1e9)
    memory = []
    for path in diagnostics.memory:
        memory.append({
            "path_name": path.path_name,
            "access_kind": path.access_kind,
            "rate_unit": path.rate_unit,
            "initial_transactions": path.initial_transactions,
            "lambda_initial_per_s": path.lambda_initial_per_s,
            "lambda_capacity_per_s": path.lambda_capacity_per_s,
            "workload_slots_per_unit": path.workload_slots_per_unit,
            "hardware_source_slots_per_unit": (
                path.hardware_source_slots_per_unit),
            "source_slots_per_unit": path.source_slots_per_unit,
            "source_lifetime_s": path.source_lifetime_s,
            "source_rate_capacity_per_s": path.source_rate_capacity_per_s,
            "frontend_rate_capacity_per_s": path.frontend_rate_capacity_per_s,
            "steady_time_ns": path.steady_time_s * 1e9,
            "first_request_latency_ns": (
                None if path.first_request_latency_s is None
                else path.first_request_latency_s * 1e9),
            "bottleneck": path.bottleneck,
            "capacity_bound": path.capacity_bound,
            "nodes": [
                {
                    "resource_role": node.resource_role,
                    "transactions": node.transactions,
                    "bytes": node.bytes,
                    "byte_capacity_per_s": node.byte_capacity_per_s,
                    "request_capacity_per_s": node.request_capacity_per_s,
                    "service_capacity_per_s": node.service_capacity_per_s,
                    "arrival_rate_per_s": node.arrival_rate_per_s,
                    "rho": node.rho,
                    "queue_wait_ns": (
                        None if node.queue_wait_s is None
                        else node.queue_wait_s * 1e9),
                    "capacity_time_ns": node.capacity_time_s * 1e9,
                    "saturated": node.saturated,
                }
                for node in path.nodes
            ],
            "queues": [json_value(queue) for queue in path.queues],
        })
    pipeline = None
    if diagnostics.pipeline is not None:
        pipeline = {
            "execution_time_ns": diagnostics.pipeline.execution_time_s * 1e9,
            "initiation_interval_ns": (
                diagnostics.pipeline.initiation_interval_s * 1e9),
            "worker_makespan_ns": {
                name: value * 1e9
                for name, value in diagnostics.pipeline.worker_makespan_s.items()
            },
            "stage_service_ns": {
                name: value * 1e9
                for name, value in diagnostics.pipeline.stage_service_s.items()
            },
            "critical_path": list(diagnostics.pipeline.critical_path),
            "encoded_task_diagnostic_count": len(diagnostics.pipeline.tasks),
        }
    return {
        "total_ns": prediction.t_total_ns,
        "launch_dispatch_ns": prediction.t_launch_s * 1e9,
        "bottleneck_group": prediction.bottleneck_group,
        "time_per_group_ns": {
            name: value * 1e9 for name, value in prediction.t_per_group.items()
        },
        "time_per_resource_ns": {
            name: value * 1e9
            for name, value in prediction.t_per_resource.items()
        },
        "resource_time_by_role_ns": resource_time_by_role_ns,
        "compute_time_by_role_ns": {
            name: value * 1e9
            for name, value in diagnostics.compute_time_s.items()
        },
        "memory": memory,
        "memory_access_classes": [
            json_value(value) for value in diagnostics.memory_access_classes
        ],
        "memory_arbitration": [
            json_value(value) for value in diagnostics.memory_arbitration
        ],
        "pipeline": pipeline,
        "critical_path": list(diagnostics.critical_path),
        "support_level": diagnostics.support_level,
        "limitations": list(diagnostics.limitations),
    }


def nvidia_l2_directional_summary(
        sections: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return native NCU L2 source traffic without merging directions."""

    l2_table = sections["Memory Workload Analysis Tables"]["L2 Cache"]
    sector_fields = {
        "l2_source_read_bytes": (
            "lts__t_sectors_srcunit_tex_op_read.sum [sector]"),
        "l2_source_write_bytes": (
            "lts__t_sectors_srcunit_tex_op_write.sum [sector]"),
        "l2_read_lookup_hit_bytes": (
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum [sector]"),
        "l2_read_lookup_miss_bytes": (
            "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum [sector]"),
        "l2_write_lookup_hit_bytes": (
            "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum [sector]"),
        "l2_write_lookup_miss_bytes": (
            "lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum [sector]"),
        "l2_source_atomic_alu_bytes": (
            "lts__t_sectors_srcunit_tex_op_atom_dot_alu.sum [sector]"),
        "l2_source_atomic_cas_bytes": (
            "lts__t_sectors_srcunit_tex_op_atom_dot_cas.sum [sector]"),
        "l2_source_reduction_bytes": (
            "lts__t_sectors_srcunit_tex_op_red.sum [sector]"),
    }
    values = {}
    invalid_fields = []
    for name, field in sector_fields.items():
        sectors = to_float(l2_table[field])
        if not math.isfinite(sectors):
            raise ValueError(
                f"NCU L2 sector count {field!r} must be finite: "
                f"{sectors!r}")
        if sectors < 0:
            # This profiler field is a raw observation, not a physical model
            # input.  A negative sector count cannot establish a directional
            # traffic contract, but it must not prevent the report from
            # retaining the remaining valid evidence.  Preserve the missing
            # comparison explicitly instead of turning the value into zero.
            invalid_fields.append(field)
            values[name] = None
        else:
            values[name] = sectors * NVIDIA_L2_SECTOR_BYTES
    for direction in ("read", "write"):
        total = values[f"l2_source_{direction}_bytes"]
        hit = values[f"l2_{direction}_lookup_hit_bytes"]
        miss = values[f"l2_{direction}_lookup_miss_bytes"]
        classified = None if hit is None or miss is None else hit + miss
        # NCU may collect these metrics in different replay passes.  Preserve
        # both native values and record whether they close; never force one to
        # be reconstructed from the other.
        values[f"l2_{direction}_lookup_partition_closes"] = (
            total is not None
            and classified is not None
            and math.isclose(total, classified, rel_tol=0.0, abs_tol=0.0))
    values["l2_directional_read_write_comparable"] = (
        not invalid_fields
        and values["l2_source_atomic_alu_bytes"] == 0.0
        and values["l2_source_atomic_cas_bytes"] == 0.0
        and values["l2_source_reduction_bytes"] == 0.0)
    values["l2_directional_counter_status"] = (
        "complete" if not invalid_fields else "invalid_raw_counter")
    values["l2_directional_invalid_fields"] = invalid_fields
    return values


def optional_ascend_kb_bytes(
        raw: Mapping[str, Any], field: str) -> float | None:
    if field not in raw or raw[field] == "NA":
        return None
    kilobytes = to_float(raw[field])
    if not math.isfinite(kilobytes) or kilobytes < 0:
        raise ValueError(
            f"Ascend counter {field!r} must be finite and non-negative: "
            f"{kilobytes!r}")
    return kilobytes * 1024.0


def optional_ascend_percent(
        raw: Mapping[str, Any], field: str) -> float | None:
    if field not in raw or raw[field] in ("", "NA", "N/A"):
        return None
    percentage = to_float(raw[field])
    if (not math.isfinite(percentage)
            or percentage < 0.0 or percentage > 100.0):
        raise ValueError(
            f"Ascend counter {field!r} must be within [0, 100]: "
            f"{percentage!r}")
    return percentage


def observation_summary(profiler_record, profiler_metrics) -> Mapping[str, Any]:
    sections = profiler_record["Sections"]
    memory_chart = sections["Memory Workload Analysis Chart"]["Memory Chart"]
    instruction_count = None
    if "Instruction Statistics" in sections:
        instruction_header = sections["Instruction Statistics"]["Header"]
        instruction_count = to_float(
            instruction_header["Executed Instructions [inst]"])
    native = None
    pipe_mte_pct = None
    pipe_ste_pct = None
    l2_source_interface_bytes = profiler_metrics["l2_total"]
    downstream_global_bytes = profiler_metrics["dram_bytes"]
    canonical_l2_service_bytes = None
    canonical_l2_service_status = None
    if "MetaX" in profiler_record:
        metax = profiler_record["MetaX"]
        scalars = metax["all_scalars"]
        roofline = metax["roofline"]
        pipe_mte_pct = to_float(scalars["AP MTE Duty ratio"])
        pipe_ste_pct = to_float(scalars["AP STE Duty ratio"])
        if "l2c_memacs" in roofline:
            l2c_memory_access_bytes = to_float(roofline["l2c_memacs"])
        else:
            l2c_memory_access_bytes = None
        if "all_memacs" in roofline:
            global_memory_access_bytes = to_float(roofline["all_memacs"])
        else:
            global_memory_access_bytes = None
        for name, value in (
                ("l2c_memacs", l2c_memory_access_bytes),
                ("all_memacs", global_memory_access_bytes)):
            if value is not None and value < 0:
                raise ValueError(
                    f"MetaX {name} cannot be negative: {value}")
        l2_source_interface_bytes = l2c_memory_access_bytes
        downstream_global_bytes = global_memory_access_bytes
        if (l2c_memory_access_bytes is not None
                and global_memory_access_bytes is not None):
            canonical_l2_service_bytes = max(
                l2c_memory_access_bytes, global_memory_access_bytes)
            canonical_l2_service_status = (
                "max_metax_l2c_source_and_downstream_global")
        else:
            canonical_l2_service_status = (
                "evidence_pending_missing_metax_traffic_counter")
        if "core_clk" in roofline:
            core_clock_ghz = roofline["core_clk"]
        else:
            core_clock_ghz = None
        native = {
            "backend": "metax_mcprofiler",
            "source": metax["source"],
            "duration_source": metax["dur_source"],
            "instrumented_during_ns": metax["during"],
            "core_clock_ghz": core_clock_ghz,
            "workgroups": scalars["WORKGROUPS"],
            "waves": scalars["WAVES"],
            "total_instructions": scalars["Total Instructions"],
            "average_wave_life_cycles": scalars[
                "Average Wave life cycles"],
            "dnoc_read_average_latency_cycles": scalars[
                "Dnoc Read Average Latency"],
            "vls_pipeline_stall_cycles": metax["charts"][
                "ISU stall cycles layout"]["vls_pipeline_stall"],
            "vls_wdata_stall_cycles": metax["charts"][
                "ISU stall cycles layout"]["vls_wdata_stall"],
            "mma_duty_pct": to_float(scalars["AP MMA Duty ratio"]),
            "mte_duty_pct": pipe_mte_pct,
            "ste_duty_pct": pipe_ste_pct,
            "compute_instructions": scalars["Compute Instructions"],
            "memory_instructions": scalars["Memory Instructions"],
            "global_read_instructions": scalars["Global Read Instructions"],
            "global_write_instructions": scalars["Global Write Instructions"],
            "constant_read_instructions": scalars[
                "Constant Read Instructions"],
            "constant_read_sl1_instructions": scalars[
                "Constant Read SL1 Instructions"],
            "constant_read_l2_instructions": scalars[
                "Constant Read L2 Instructions"],
            "constant_sl1_hit_rate_pct": scalars["Constant SL1 Hit Rate"],
            "vl1_read_instructions": scalars["VL1 Read Instructions"],
            "vl1_write_instructions": scalars["VL1 Write Instructions"],
            "global_read_bytes": scalars["Global Memory Read bytes"],
            "global_write_bytes": scalars["Global Memory Write bytes"],
            "dnoc_read_requests": scalars["Dnoc Read Req"],
            "dnoc_write_requests": scalars["Dnoc Write Req"],
            "l2c_read_instructions": scalars["L2C Read Instructions"],
            "l2c_write_instructions": scalars["L2C Write Instructions"],
            "l2c_hit_rate_pct": scalars["L2C Hit Rate"],
            "vl1_hit_rate_pct": scalars["VL1 Hit Rate"],
            "l2c_memory_access_bytes": l2c_memory_access_bytes,
            "global_memory_access_bytes": global_memory_access_bytes,
            "canonical_l2_service_bytes": canonical_l2_service_bytes,
        }
    elif "Ascend" in profiler_record:
        ascend = profiler_record["Ascend"]
        raw = ascend["raw"]
        if raw == {}:
            read_main_memory_kb = None
            write_main_memory_kb = None
            gm_to_l1_kb = None
            gm_to_ub_kb = None
            gm_to_local_kb = None
            ub_to_gm_kb = None
            l2_read_hit_rate_pct = None
            l2_write_hit_rate_pct = None
        else:
            read_main_memory_kb = raw["read_main_memory_datas(KB)"]
            write_main_memory_kb = raw["write_main_memory_datas(KB)"]
            if "GM_to_L1_datas(KB)" in raw:
                gm_to_l1_kb = raw["GM_to_L1_datas(KB)"]
            else:
                gm_to_l1_kb = None
            if "GM_to_UB_datas(KB)" in raw:
                gm_to_ub_kb = raw["GM_to_UB_datas(KB)"]
            else:
                gm_to_ub_kb = None
            ub_to_gm_kb = raw["UB_to_GM_datas(KB)"]
            if ascend["op_type"] in ("cube", "mix"):
                gm_to_local_kb = gm_to_l1_kb
                l2_read_hit_rate_pct = optional_ascend_percent(
                    raw, "aic_read_hit_rate(%)")
                l2_write_hit_rate_pct = optional_ascend_percent(
                    raw, "aic_write_hit_rate(%)")
            elif ascend["op_type"] == "vector":
                gm_to_local_kb = gm_to_ub_kb
                l2_read_hit_rate_pct = optional_ascend_percent(
                    raw, "aiv_read_hit_rate(%)")
                l2_write_hit_rate_pct = optional_ascend_percent(
                    raw, "aiv_write_hit_rate(%)")
            else:
                raise ValueError(
                    "unknown Ascend execution domain in mechanism report: "
                    f"{ascend['op_type']!r}")
        read_main_memory_bytes = optional_ascend_kb_bytes(
            raw, "read_main_memory_datas(KB)")
        write_main_memory_bytes = optional_ascend_kb_bytes(
            raw, "write_main_memory_datas(KB)")
        if (read_main_memory_bytes is None
                or l2_read_hit_rate_pct is None):
            derived_dram_read_bytes = None
        else:
            derived_dram_read_bytes = (
                read_main_memory_bytes
                * (1.0 - l2_read_hit_rate_pct / 100.0))
        canonical_l2_service_status = (
            "not_comparable_as_one_aggregate_for_ascend_mte2_mte3")
        native = {
            "backend": "ascend_msprof",
            "op_type": ascend["op_type"],
            "block_dim": ascend["block_dim"],
            "mix_block_dim": ascend["mix_block_dim"],
            "cube_fops": ascend["cube_fops"],
            "legacy_dram_read_bytes": ascend["dram_read_bytes"],
            "dram_read_bytes": derived_dram_read_bytes,
            "gm_to_l1_bytes": ascend["gm_to_l1_bytes"],
            "l2_hit_rate": ascend["l2_hit_rate"],
            "mte2_tile_count": ascend["mte2_tile_count"],
            "read_main_memory_kb": read_main_memory_kb,
            "write_main_memory_kb": write_main_memory_kb,
            "read_main_memory_bytes": read_main_memory_bytes,
            "write_main_memory_bytes": write_main_memory_bytes,
            "gm_to_l1_kb": gm_to_l1_kb,
            "gm_to_ub_kb": gm_to_ub_kb,
            "gm_to_local_kb": gm_to_local_kb,
            "ub_to_gm_kb": ub_to_gm_kb,
            "raw_gm_to_l1_bytes": optional_ascend_kb_bytes(
                raw, "GM_to_L1_datas(KB)"),
            "gm_to_ub_bytes": optional_ascend_kb_bytes(
                raw, "GM_to_UB_datas(KB)"),
            "gm_to_local_bytes": (
                None if gm_to_local_kb is None
                else to_float(gm_to_local_kb) * 1024.0),
            "ub_to_gm_bytes": optional_ascend_kb_bytes(
                raw, "UB_to_GM_datas(KB)"),
            "l0c_to_gm_bytes": optional_ascend_kb_bytes(
                raw, "L0C_to_GM_datas(KB)"),
            "l2_read_hit_rate_pct": l2_read_hit_rate_pct,
            "l2_write_hit_rate_pct": l2_write_hit_rate_pct,
            "aic_cube_time_ns": ascend["aic_cube_time_ns"],
            "aic_mte2_time_ns": ascend["aic_mte2_time_ns"],
            "aic_scalar_time_ns": ascend["aic_scalar_time_ns"],
            "aiv_scalar_time_ns": ascend["aiv_scalar_time_ns"],
            "aiv_vec_time_ns": ascend["aiv_vec_time_ns"],
        }
    else:
        nvidia_l2 = nvidia_l2_directional_summary(sections)
        native = {
            "backend": "nvidia_ncu",
            "executed_instructions": instruction_count,
            **nvidia_l2,
            "dram_read_bytes": to_float(
                memory_chart["dram__bytes_read.sum [byte]"]),
            "dram_write_bytes": to_float(
                memory_chart["dram__bytes_write.sum [byte]"]),
        }
        canonical_l2_service_bytes = max(
            l2_source_interface_bytes, downstream_global_bytes)
        canonical_l2_service_status = (
            "max_ncu_source_interface_and_downstream_dram")
    if canonical_l2_service_status is None:
        raise RuntimeError("observation has no canonical L2 service status")
    return {
        "profiler_header_duration_ns_not_scoring_target": (
            profiler_metrics["duration_ns"]),
        "sm_frequency_hz": profiler_metrics["sm_freq_hz"],
        "dram_frequency_hz": profiler_metrics["dram_freq_hz"],
        "elapsed_cycles": profiler_metrics["elapsed_cycles"],
        "active_cycles": profiler_metrics["active_cycles"],
        "sm_active_fraction": profiler_metrics["sm_active_frac"],
        "pipe_tensor_pct": profiler_metrics["tensor_pct"],
        "pipe_fma_pct": profiler_metrics["fma_pct"],
        "pipe_alu_pct": profiler_metrics["alu_pct"],
        "pipe_sfu_pct": profiler_metrics["sfu_pct"],
        "pipe_lsu_pct": profiler_metrics["lsu_pct"],
        # MetaX MTE is the vector-compute/global-memory engine and STE is the
        # scalar/control engine.  They are not NVIDIA FMA/ALU pipe aliases;
        # preserve their native names for cross-hardware reporting.
        "pipe_mte_pct": pipe_mte_pct,
        "pipe_ste_pct": pipe_ste_pct,
        "l2_interface_read_bytes": to_float(
            memory_chart["l1tex__m_xbar2l1tex_read_bytes.sum [byte]"]),
        "l2_interface_write_bytes": to_float(
            memory_chart["l1tex__m_l1tex2xbar_write_bytes.sum [byte]"]),
        "l2_interface_total_bytes": profiler_metrics["l2_total"],
        "l2_source_interface_bytes": l2_source_interface_bytes,
        "downstream_global_bytes": downstream_global_bytes,
        "canonical_l2_service_bytes": canonical_l2_service_bytes,
        "canonical_l2_service_status": canonical_l2_service_status,
        "dram_read_bytes": to_float(
            memory_chart["dram__bytes_read.sum [byte]"]),
        "dram_write_bytes": to_float(
            memory_chart["dram__bytes_write.sum [byte]"]),
        "dram_total_bytes": profiler_metrics["dram_bytes"],
        "l1_hit_rate": profiler_metrics["l1_hit_rate"],
        "l2_hit_rate": profiler_metrics["l2_hit_rate"],
        "ldgsts_bypass_bytes": profiler_metrics["ldgsts_bypass"],
        "grid": [
            profiler_metrics["grid_x"],
            profiler_metrics["grid_y"],
            profiler_metrics["grid_z"],
        ],
        "grid_size": profiler_metrics["grid_size"],
        "block_size": profiler_metrics["block_size"],
        "registers_per_thread": profiler_metrics["regs_per_thread"],
        "shared_memory_per_block_bytes": profiler_metrics["smem_per_block"],
        "achieved_occupancy_pct": profiler_metrics["achieved_occupancy_pct"],
        "theoretical_occupancy_pct": (
            profiler_metrics["theoretical_occupancy_pct"]),
        "block_limit_registers": profiler_metrics["block_limit_registers"],
        "native": native,
    }


def oracle_summary(
        oracle_observation, oracle_unavailable_reason) -> Mapping[str, Any]:
    if oracle_observation is None:
        if (not isinstance(oracle_unavailable_reason, str)
                or not oracle_unavailable_reason):
            raise ValueError(
                "unavailable Oracle observation requires a non-empty reason")
        return {
            "available": False,
            "unavailable_reason": oracle_unavailable_reason,
            "backend": None,
            "coverage": None,
            "target_derived_paths": [],
            "field_provenance": {},
            "unavailable_fields": {},
            "observed_input": None,
        }
    provenance = {
        path: {
            "kind": source.kind,
            "provider": source.provider,
            "metric": source.metric,
            "uses_target_duration": source.uses_target_duration,
        }
        for path, source in oracle_observation.field_provenance.items()
    }
    return {
        "available": True,
        "unavailable_reason": None,
        "backend": oracle_observation.backend,
        "coverage": oracle_observation.coverage,
        "target_derived_paths": list(oracle_observation.target_derived_paths),
        "field_provenance": provenance,
        "unavailable_fields": dict(oracle_observation.unavailable_fields),
        "observed_input": prediction_input_summary(
            oracle_observation.prediction_input),
    }


def single_counterfactual_summary(
        kernel_spec, oracle_observation, hardware) -> list[Mapping[str, Any]]:
    """Evaluate every profiler-backed formula input one at a time."""

    if oracle_observation is None:
        return []
    predictor = create_predictor(kernel_spec.kind)
    rows = []
    for candidate in single_counterfactual_inputs(oracle_observation):
        prediction = predictor.oracle(
            kernel_spec, hardware, candidate.prediction_input)
        rows.append({
            "component": candidate.component,
            "input_path": candidate.input_path,
            "unit": candidate.unit,
            "provider": candidate.provenance.provider,
            "metric": candidate.provenance.metric,
            "provenance_kind": candidate.provenance.kind,
            "uses_target_duration": candidate.provenance.uses_target_duration,
            "predicted_ns": prediction.t_total_ns,
            "bottleneck_group": prediction.bottleneck_group,
        })
    return rows


def capture_kernel_mechanisms(
        *, kernel_spec, profiler_record, profiler_metrics, hardware,
        forward_prediction, oracle_observation, oracle_unavailable_reason,
        timing_target_ns,
        timing_target_source, timing_target_unavailable_reason,
        kernel_ordinal) -> Mapping[str, Any]:
    """Capture existing model inputs/outputs without recomputing a formula."""

    analytical_input, compulsory_bytes = create_predictor(
        kernel_spec.kind).forward_resource_input(kernel_spec, hardware)
    # Oracle deliberately owns a separately normalized input.  For CANN it
    # may also retain an expanded source DAG while Forward uses the equivalent
    # compact worker envelope.  Capture both representations independently;
    # requiring structural equality would reject valid Forward diagnostics.
    if timing_target_ns is None:
        if timing_target_source is not None:
            raise ValueError(
                "unobserved kernel timing cannot have a timing source")
        if (not isinstance(timing_target_unavailable_reason, str)
                or not timing_target_unavailable_reason.strip()):
            raise ValueError(
                "unobserved kernel timing requires an unavailable reason")
        timing_target = {
            "availability": "unobserved",
            "duration_ns": None,
            "source": None,
            "unavailable_reason": timing_target_unavailable_reason,
        }
    else:
        if (not isinstance(timing_target_ns, (int, float))
                or isinstance(timing_target_ns, bool)
                or not math.isfinite(float(timing_target_ns))
                or timing_target_ns <= 0):
            raise ValueError(
                "observed kernel timing must be finite and positive")
        if (not isinstance(timing_target_source, str)
                or not timing_target_source.strip()):
            raise ValueError(
                "observed kernel timing requires a timing source")
        if timing_target_unavailable_reason is not None:
            raise ValueError(
                "observed kernel timing cannot have an unavailable reason")
        timing_target = {
            "availability": "observed",
            "duration_ns": timing_target_ns,
            "source": timing_target_source,
            "unavailable_reason": None,
        }
    return {
        "kernel_ordinal": kernel_ordinal,
        "kernel_name": profiler_metrics["kernel_name"],
        "kernel_kind": kernel_spec.kind.value,
        "hardware_id": kernel_spec.hardware_id,
        "shape": dict(kernel_spec.shape),
        "dtype": kernel_spec.dtype,
        "variant": kernel_spec.variant,
        "attributes": json_value(kernel_spec.attributes),
        "observed_launch": launch_summary(kernel_spec.launch),
        "timing_target": timing_target,
        "observation": observation_summary(
            profiler_record, profiler_metrics),
        "analytical_input": prediction_input_summary(analytical_input),
        "compulsory_bytes": compulsory_bytes,
        "forward": prediction_summary(forward_prediction, hardware),
        "oracle": oracle_summary(
            oracle_observation, oracle_unavailable_reason),
        "single_counterfactuals": single_counterfactual_summary(
            kernel_spec, oracle_observation, hardware),
    }
