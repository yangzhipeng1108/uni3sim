"""Normalize profiler evidence into the canonical Oracle formula input.

Oracle is a formula-ceiling evaluation, not a second predictor.  It starts
from the production predictor's analytical :class:`PredictionInput`, replaces
every formula intermediate for which the profiler has hardware-native
evidence, and sends the result through the same ``PerformanceFormula`` used by
Midlayer.  Every replacement carries provenance; unavailable evidence remains
explicit instead of being mistaken for profiler truth.
"""

from dataclasses import dataclass, fields, replace
import math
from types import MappingProxyType
from typing import Mapping

from evaluation.kernel.dataset import to_float
from evaluation.kernel.profiler_observation import extract_ncu_metrics
from src.kernel.kernel_base import KernelPredictor, create_predictor
from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config.tiling_spec import (
    flash_attention_packed_varlen_scheduled_workgroups,
    gemm_logical_workgroups,
    parse_flash_attention_tile,
)
from src.kernel.model_contract import (
    BarrierPipelineEnvelopeDemand,
    CannGroupedMatmulEnvelopeDemand,
    GpuGroupedGemmPipelineEnvelopeDemand,
    PipelineTaskDemand,
    PipelineTaskProgramDemand,
    PipelineTaskSequenceDemand,
    PipelineWorkPlan,
)
from src.kernel.predictors.common import (
    PredictionInput,
    replace_structured_demands,
)


_SOURCE_KINDS = {
    "analytical",
    "config",
    "launch_contract",
    "profiler",
    "profiler_derived",
}
_BACKENDS = {"nvidia_ncu", "metax_mcprofiler", "ascend_msprof"}


def _live_analytical_formula_paths(prediction_input, provenance):
    """Return analytical numeric inputs still live in the formula result.

    A resource-time override replaces the corresponding demand/capacity path.
    Every other positive analytical demand, and every analytical scalar that
    modifies such a path, must be reported as unavailable before an Oracle
    observation can claim complete profiler-evidence coverage.
    """

    overridden_roles = set(prediction_input.resource_time_overrides_ns)
    live_roles = {
        role for role, demand in prediction_input.demand_per_resource.items()
        if demand > 0 and role not in overridden_roles
    }
    live_paths = set()

    def include(path, active):
        if not active:
            return
        if path not in provenance:
            raise KeyError(
                f"Oracle provenance is missing formula input {path!r}")
        if provenance[path].kind == "analytical":
            live_paths.add(path)

    for role in live_roles:
        include(f"demand_per_resource.{role}", True)
    include("compute_demands", bool(prediction_input.compute_demands))
    include("memory_paths", bool(prediction_input.memory_paths))
    include("pipeline_plan", prediction_input.pipeline_plan is not None)
    include(
        "dram_capacity_multiplier",
        "dram" in live_roles
        and prediction_input.dram_capacity_multiplier != 1.0)
    return frozenset(live_paths)


@dataclass(frozen=True)
class OracleFieldProvenance:
    """Evidence used for one Oracle formula-input path."""

    kind: str
    provider: str
    metric: str
    uses_target_duration: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _SOURCE_KINDS:
            raise ValueError(f"unknown Oracle provenance kind: {self.kind!r}")
        for name in ("provider", "metric"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"Oracle provenance {name} must be a non-empty string")
        if not isinstance(self.uses_target_duration, bool):
            raise TypeError(
                "Oracle provenance uses_target_duration must be boolean")
        if self.uses_target_duration:
            raise ValueError(
                "Formula Ceiling evidence must not use the kernel timing "
                "target")


@dataclass(frozen=True)
class OracleObservation:
    """One normalized Oracle input plus field-level evidence coverage."""

    hardware_id: str
    backend: str
    analytical_input: PredictionInput
    prediction_input: PredictionInput
    field_provenance: Mapping[str, OracleFieldProvenance]
    unavailable_fields: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.hardware_id, str) or not self.hardware_id:
            raise TypeError("OracleObservation.hardware_id must be non-empty")
        if self.backend not in _BACKENDS:
            raise ValueError(f"unknown Oracle profiler backend: {self.backend!r}")
        for name in ("analytical_input", "prediction_input"):
            if not isinstance(getattr(self, name), PredictionInput):
                raise TypeError(
                    f"OracleObservation.{name} must be PredictionInput")
        if not isinstance(self.field_provenance, Mapping):
            raise TypeError("Oracle field_provenance must be a mapping")
        if not isinstance(self.unavailable_fields, Mapping):
            raise TypeError("Oracle unavailable_fields must be a mapping")
        provenance = dict(self.field_provenance)
        unavailable = dict(self.unavailable_fields)
        invalid_provenance = {
            path: source for path, source in provenance.items()
            if (not isinstance(path, str) or not path
                or not isinstance(source, OracleFieldProvenance))
        }
        if invalid_provenance:
            raise TypeError(
                "Oracle field provenance requires non-empty paths and "
                "OracleFieldProvenance values: "
                f"{invalid_provenance}")
        target_assisted = {
            path: source for path, source in provenance.items()
            if source.uses_target_duration
        }
        if target_assisted:
            raise ValueError(
                "Formula Ceiling observations cannot contain target-assisted "
                f"evidence: {sorted(target_assisted)}")
        invalid_unavailable = {
            path: reason for path, reason in unavailable.items()
            if (not isinstance(path, str) or not path
                or not isinstance(reason, str) or not reason)
        }
        if invalid_unavailable:
            raise TypeError(
                "Oracle unavailable fields require non-empty paths/reasons: "
                f"{invalid_unavailable}")
        unaccounted = (
            _live_analytical_formula_paths(self.prediction_input, provenance)
            - set(unavailable)
        )
        if unaccounted:
            raise ValueError(
                "Formula Ceiling coverage omits live analytical formula "
                f"inputs: {sorted(unaccounted)}")
        object.__setattr__(
            self, "field_provenance", MappingProxyType(provenance))
        object.__setattr__(
            self, "unavailable_fields", MappingProxyType(unavailable))

    @property
    def target_derived_paths(self) -> tuple[str, ...]:
        return tuple(sorted(
            path for path, source in self.field_provenance.items()
            if source.uses_target_duration))

    @property
    def coverage(self) -> str:
        if self.unavailable_fields:
            return "partial"
        return "complete"


@dataclass(frozen=True)
class SingleCounterfactualInput:
    """One independently observed replacement for diagnostic evaluation."""

    component: str
    input_path: str
    unit: str
    prediction_input: PredictionInput
    provenance: OracleFieldProvenance

    def __post_init__(self) -> None:
        for name in ("component", "input_path", "unit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"SingleCounterfactualInput.{name} must be non-empty")
        if not isinstance(self.prediction_input, PredictionInput):
            raise TypeError(
                "SingleCounterfactualInput.prediction_input must be "
                "PredictionInput")
        if not isinstance(self.provenance, OracleFieldProvenance):
            raise TypeError(
                "SingleCounterfactualInput.provenance must be "
                "OracleFieldProvenance")
        if self.provenance.kind not in {"profiler", "profiler_derived"}:
            raise ValueError(
                "single-counterfactual evidence must be profiler-derived")
        if self.provenance.uses_target_duration:
            raise ValueError(
                "single-counterfactual evidence cannot use target duration")


def single_counterfactual_inputs(
        observation: OracleObservation) -> tuple[SingleCounterfactualInput, ...]:
    """Build one-at-a-time formula inputs from comparable profiler evidence.

    The returned inputs always start from the production analytical input and
    replace exactly one normalized formula field.  Structured demand ledgers
    are updated through the same conservation-preserving helper used by the
    full Formula Ceiling normalization.
    """

    if not isinstance(observation, OracleObservation):
        raise TypeError(
            "single_counterfactual_inputs requires OracleObservation")
    analytical = observation.analytical_input
    observed = observation.prediction_input
    provenance = observation.field_provenance
    unavailable = observation.unavailable_fields
    result = []

    for role, observed_value in observed.demand_per_resource.items():
        path = f"demand_per_resource.{role}"
        if path in unavailable:
            continue
        if path not in provenance:
            raise KeyError(
                f"single-counterfactual provenance is missing {path!r}")
        source = provenance[path]
        if source.kind not in {"profiler", "profiler_derived"}:
            continue
        candidate = _replace_demand(analytical, {role: observed_value})
        if role in {
                "l1_cache", "smem", "xbar", "l2_cache", "dram", "mte3"}:
            unit = "bytes"
        elif role in {
                "compute_tensor", "compute_vector", "compute_reduction",
                "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
                "scalar_aiv"}:
            unit = "operations"
        else:
            raise ValueError(f"unknown demand role unit: {role!r}")
        result.append(SingleCounterfactualInput(
            component=f"demand:{role}",
            input_path=path,
            unit=unit,
            prediction_input=candidate,
            provenance=source,
        ))

    for role, observed_value in observed.resource_time_overrides_ns.items():
        path = f"resource_time_overrides_ns.{role}"
        if path in unavailable:
            continue
        if path not in provenance:
            raise KeyError(
                f"single-counterfactual provenance is missing {path!r}")
        source = provenance[path]
        if source.kind not in {"profiler", "profiler_derived"}:
            continue
        overrides = dict(analytical.resource_time_overrides_ns)
        overrides[role] = observed_value
        result.append(SingleCounterfactualInput(
            component=f"service_time:{role}",
            input_path=path,
            unit="ns",
            prediction_input=replace(
                analytical, resource_time_overrides_ns=overrides),
            provenance=source,
        ))

    for field_name, component in (
            ("sm_freq_hz", "frequency:sm"),
            ("dram_freq_hz", "frequency:dram")):
        if field_name in unavailable:
            continue
        if field_name not in provenance:
            raise KeyError(
                "single-counterfactual provenance is missing "
                f"{field_name!r}")
        source = provenance[field_name]
        if source.kind not in {"profiler", "profiler_derived"}:
            continue
        result.append(SingleCounterfactualInput(
            component=component,
            input_path=field_name,
            unit="Hz",
            prediction_input=replace(
                analytical, **{field_name: getattr(observed, field_name)}),
            provenance=source,
        ))

    components = [candidate.component for candidate in result]
    if len(components) != len(set(components)):
        raise RuntimeError(
            "single-counterfactual components must be unique: "
            f"{components}")
    return tuple(result)


def _source(kind, provider, metric, *, uses_target_duration=False):
    return OracleFieldProvenance(
        kind=kind,
        provider=provider,
        metric=metric,
        uses_target_duration=uses_target_duration,
    )


def _base_provenance(analytical):
    provenance = {}
    launch_fields = {
        "active_units", "grid_size", "block_size", "variant", "mem_model",
        "wave_efficiency", "launch_task_kind", "include_dispatch_overhead",
        "support_level", "limitations",
    }
    config_fields = {"sm_freq_hz", "dram_freq_hz"}
    for field_info in fields(PredictionInput):
        name = field_info.name
        if name == "demand_per_resource":
            for role in analytical.demand_per_resource:
                provenance[f"demand_per_resource.{role}"] = _source(
                    "analytical", "production_predictor", role)
        elif name == "resource_time_overrides_ns":
            provenance[name] = _source(
                "analytical", "performance_formula", "no override")
            for role in analytical.resource_time_overrides_ns:
                provenance[f"{name}.{role}"] = _source(
                    "analytical", "production_predictor", role)
        elif name in launch_fields:
            provenance[name] = _source(
                "launch_contract", "production_predictor", name)
        elif name in config_fields:
            provenance[name] = _source(
                "config", "hardware_spec", name)
        else:
            provenance[name] = _source(
                "analytical", "production_predictor", name)
    return provenance


def _finite_number(mapping, key, context, *, non_negative=True):
    if key not in mapping:
        raise KeyError(f"{context} is missing {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{context}.{key} must be numeric, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context}.{key} must be finite, got {number}")
    if non_negative and number < 0:
        raise ValueError(f"{context}.{key} cannot be negative, got {number}")
    return number


def _replace_demand(analytical, replacements):
    """Return a contract-consistent input with observed demand replacements."""

    unknown = set(replacements) - set(analytical.demand_per_resource)
    if unknown:
        raise ValueError(
            f"Oracle demand replacement has unknown roles: {sorted(unknown)}")
    annotated_ingress_roles = {
        path.nodes[0].resource_role
        for path in analytical.memory_paths
        if path.access_class_demands
    }
    replaced_annotated_ingress = (
        annotated_ingress_roles.intersection(replacements))
    if replaced_annotated_ingress:
        raise NotImplementedError(
            "Oracle demand replacement is unsupported for ingress roles "
            "with typed memory access-class demand: "
            f"{sorted(replaced_annotated_ingress)}")

    # A source-closed pipeline may carry exact requests and absolute memory
    # ownership that intentionally cannot be inferred from aggregate profiler
    # counters.  Rescaling only its aggregate demand would make the Oracle
    # input disagree with the staged source program; rescaling the stages
    # would fabricate per-worker/per-launch attribution.  Keep this Oracle
    # case explicitly unavailable instead of weakening production
    # conservation or silently discarding the source DAG.
    pipeline = analytical.pipeline_plan
    if isinstance(
            pipeline,
            (CannGroupedMatmulEnvelopeDemand,
             GpuGroupedGemmPipelineEnvelopeDemand)):
        raise NotImplementedError(
            "Oracle aggregate counters cannot replace a grouped-GEMM "
            "source tile pipeline recurrence")
    if isinstance(pipeline, (BarrierPipelineEnvelopeDemand, PipelineWorkPlan)):
        entries = (
            pipeline.classes
            if isinstance(pipeline, BarrierPipelineEnvelopeDemand)
            else pipeline.tasks)
        logical_tasks = tuple(
            task
            for entry in entries
            for sequence in (
                entry.patterns
                if isinstance(entry, PipelineTaskProgramDemand)
                else (entry,))
            for task in (
                sequence.tasks
                if isinstance(sequence, PipelineTaskSequenceDemand)
                else (sequence,))
        )
        if any(not isinstance(task, PipelineTaskDemand)
               for task in logical_tasks):
            raise TypeError(
                "Oracle pipeline normalization encountered a non-task "
                "logical entry")
        conflicts = set()
        for task in logical_tasks:
            for stage in task.stages:
                if stage.compute_exact_request_demands:
                    role = stage.service.removeprefix("compute:")
                    if role in replacements:
                        conflicts.add(role)
                for demand in (
                        stage.memory_node_demands
                        + stage.launch_shared_memory_node_demands):
                    if demand.resource_role in replacements:
                        conflicts.add(demand.resource_role)
        if conflicts:
            raise NotImplementedError(
                "Oracle aggregate counters cannot replace source-closed "
                "pipeline exact/absolute ownership without per-stage "
                f"evidence: {sorted(conflicts)}")
    return replace_structured_demands(analytical, replacements)


def _mark_consumed_analytical_inputs_unavailable(
        normalized, provenance, unavailable, backend):
    """Expose live formula terms still consumed without profiler evidence.

    A native resource-time override makes the corresponding analytical demand
    irrelevant to the formula result.  Every other positive demand remains a
    live formula intermediate.  The same applies to efficiency, issue, and
    pipeline terms that alter those un-overridden resource paths.
    """

    overridden_roles = set(normalized.resource_time_overrides_ns)
    live_roles = {
        role for role, demand in normalized.demand_per_resource.items()
        if demand > 0 and role not in overridden_roles
    }
    for role, demand in normalized.demand_per_resource.items():
        path = f"demand_per_resource.{role}"
        source = provenance[path]
        if (demand > 0
                and role not in overridden_roles
                and source.kind == "analytical"
                and path not in unavailable):
            unavailable[path] = (
                f"{backend} provides no like-for-like observation for this "
                "positive resource demand")
    for role in normalized.resource_time_overrides_ns:
        path = f"resource_time_overrides_ns.{role}"
        if (provenance[path].kind == "analytical"
                and path not in unavailable):
            unavailable[path] = (
                f"{backend} provides no like-for-like observation for this "
                "analytical resource service time")

    def mark(path, active, reason):
        if (active
                and provenance[path].kind == "analytical"
                and path not in unavailable):
            unavailable[path] = reason

    mark(
        "compute_demands",
        bool(normalized.compute_demands),
        f"{backend} does not replace every structured compute demand",
    )
    mark(
        "memory_paths",
        bool(normalized.memory_paths),
        f"{backend} does not independently observe every transaction split, "
        "source slot, and queue lifetime in the structured memory paths",
    )
    mark(
        "pipeline_plan",
        normalized.pipeline_plan is not None,
        f"{backend} does not independently observe the complete worker/tile "
        "pipeline plan",
    )
    mark(
        "dram_capacity_multiplier",
        "dram" in live_roles and normalized.dram_capacity_multiplier != 1.0,
        f"{backend} provides no independent like-for-like DRAM access-class "
        "capacity multiplier",
    )


def _metax_counter_volume_issue(
        scalars, byte_count, instruction_names,
        max_bytes_per_instruction, label):
    """Return an integrity error for an impossible MetaX byte/count pair.

    ``*_memacs`` and the corresponding hierarchy instruction counters are
    independent count-like fields in the same per-invocation report.  Their
    ratio therefore validates attribution without using either mcTracer target
    duration or mcProfiler's instrumented replay duration.
    """

    instruction_count = sum(
        _finite_number(
            scalars, name, "MetaX.all_scalars")
        for name in instruction_names)
    if byte_count == 0:
        return None
    if instruction_count <= 0:
        return (
            f"MetaX {label} reports {byte_count} bytes with no hierarchy "
            "instructions")
    bytes_per_instruction = byte_count / instruction_count
    if bytes_per_instruction > max_bytes_per_instruction:
        return (
            f"MetaX {label} byte count is incompatible with its hierarchy "
            "instruction count: "
            f"bytes_per_instruction={bytes_per_instruction}, "
            f"maximum={max_bytes_per_instruction}")
    return None


def _metax_global_memory_volume_issue(
        scalars, global_bytes, max_bytes_per_request):
    """Return an integrity error for impossible global-byte/DNoC counts."""

    issues = []
    directional_bytes = {}
    for direction in ("Read", "Write"):
        byte_count = _finite_number(
            scalars, f"Global Memory {direction} bytes",
            "MetaX.all_scalars")
        request_count = _finite_number(
            scalars, f"Dnoc {direction} Req", "MetaX.all_scalars")
        directional_bytes[direction] = byte_count
        if byte_count == 0:
            continue
        if request_count <= 0:
            issues.append(
                f"{direction.lower()} reports {byte_count} bytes with no "
                "DNoC requests")
            continue
        bytes_per_request = byte_count / request_count
        if bytes_per_request > max_bytes_per_request:
            issues.append(
                f"{direction.lower()} bytes_per_request="
                f"{bytes_per_request}, maximum={max_bytes_per_request}")
    scalar_total = sum(directional_bytes.values())
    if not math.isclose(
            global_bytes, scalar_total, rel_tol=0.0, abs_tol=1e-6):
        issues.append(
            f"roofline all_memacs={global_bytes} differs from scalar global "
            f"read+write bytes={scalar_total}")
    if issues:
        return (
            "MetaX global-memory byte count is incompatible with its DNoC "
            "request count: " + "; ".join(issues))
    return None


def _normalize_nvidia(analytical, metrics, hardware, provenance, unavailable):
    l2_interface_bytes = float(metrics["l2_total"])
    dram_bytes = float(metrics["dram_bytes"])
    bypass_bytes = float(metrics["ldgsts_bypass"])
    if l2_interface_bytes < 0 or dram_bytes < 0 or bypass_bytes < 0:
        raise ValueError("NCU memory traffic cannot be negative")
    l2_service_bytes = max(l2_interface_bytes, dram_bytes)
    measured_demand = {
        "xbar": l2_interface_bytes,
        "l2_cache": l2_service_bytes,
    }
    for role in measured_demand:
        provenance[f"demand_per_resource.{role}"] = _source(
            "profiler", "nvidia_ncu", role)
    # The aggregate NCU DRAM counter is read+write traffic.  Replacing the flat
    # role would make ``replace_structured_demands`` distribute that total over
    # the Forward read/write paths in their *predicted* proportions.  That
    # creates a direction split which NCU did not observe and is especially
    # wrong for write-back stores.  Keep the production demand untouched; the
    # report compares the native read and write counters to matching paths.
    unavailable["demand_per_resource.dram"] = (
        "NCU exposes separate DRAM read/write counters; an aggregate role "
        "replacement would redistribute them using the predicted path mix")

    l1_hit_rate = float(metrics["l1_hit_rate"])
    if 0.0 <= l1_hit_rate < 1.0:
        cached_l2_bytes = (
            0.0 if bypass_bytes >= l2_interface_bytes
            else l2_interface_bytes - bypass_bytes)
        measured_demand["l1_cache"] = (
            cached_l2_bytes / (1.0 - l1_hit_rate))
        measured_demand["smem"] = 2.0 * bypass_bytes
        provenance["demand_per_resource.l1_cache"] = _source(
            "profiler_derived", "nvidia_ncu",
            "L1/L2 interface bytes and L1 hit rate")
        provenance["demand_per_resource.smem"] = _source(
            "profiler_derived", "nvidia_ncu", "2 * LDGSTS bypass bytes")
    else:
        reason = f"NCU L1 hit rate outside [0, 1): {l1_hit_rate}"
        unavailable["demand_per_resource.l1_cache"] = reason
        unavailable["demand_per_resource.smem"] = reason

    active_cycles = float(metrics["active_cycles"])
    sm_frequency_hz = float(metrics["sm_freq_hz"])
    if active_cycles < 0:
        raise ValueError(
            f"NCU active cycles cannot be negative: {active_cycles}")
    if sm_frequency_hz <= 0:
        raise ValueError(
            f"NCU SM frequency must be positive: {sm_frequency_hz}")
    pipe_percentages = {
        "compute_tensor": float(metrics["tensor_pct"]),
        "compute_fma": float(metrics["fma_pct"]),
        "compute_sfu": float(metrics["sfu_pct"]),
        "compute_alu": float(metrics["alu_pct"]),
    }
    resource_times = dict(analytical.resource_time_overrides_ns)
    for role, percentage in pipe_percentages.items():
        if not 0.0 <= percentage <= 100.0:
            raise ValueError(
                f"NCU {role} utilization must be within [0, 100]: "
                f"{percentage}")
        # NCU reports pipe utilization over SM-active cycles.  Multiplying
        # those two independent counters gives the pipe-busy interval.  Feed
        # that interval to the canonical formula directly: reconstructing an
        # operation demand and then dividing it by SM-active/elapsed as eta
        # algebraically recreates the whole kernel elapsed time and makes the
        # alleged Formula Ceiling target-assisted.
        resource_times[role] = (
            percentage / 100.0
            * active_cycles / sm_frequency_hz * 1e9)
        provenance[f"resource_time_overrides_ns.{role}"] = _source(
            "profiler_derived", "nvidia_ncu",
            f"{role} utilization * SM active cycles / SM frequency")

    normalized = _replace_demand(analytical, measured_demand)
    replacements = {
        "resource_time_overrides_ns": resource_times,
        "sm_freq_hz": sm_frequency_hz,
        "dram_freq_hz": float(metrics["dram_freq_hz"]),
    }
    provenance["resource_time_overrides_ns"] = _source(
        "profiler_derived", "nvidia_ncu",
        "pipe utilization over SM active cycles")
    provenance["sm_freq_hz"] = _source(
        "profiler", "nvidia_ncu", "SM Frequency")
    provenance["dram_freq_hz"] = _source(
        "profiler", "nvidia_ncu", "DRAM Frequency")

    # NCU provides independently counted traffic, but every exposed L2 rate or
    # duty metric is normalized by the complete kernel elapsed window.  The
    # absence matters even when the analytical pre-pass currently selects a
    # different bottleneck: using the modeled L2 time to decide whether the
    # true L2 time is relevant is circular and mislabeled slow prefill kernels
    # as complete.  Preserve the formula result, but report that its L2 service
    # time is conditional rather than independently observed.
    if l2_service_bytes > 0:
        unavailable["resource_time_overrides_ns.l2_cache"] = (
            "NCU exposes L2 sectors and elapsed-normalized rates, but no "
            "independent L2 service interval; deriving the interval from the "
            "kernel elapsed window would make Formula Ceiling target-assisted")

    if analytical.dram_capacity_multiplier != 1.0 and dram_bytes > 0:
        unavailable["dram_capacity_multiplier"] = (
            "NCU achieved DRAM throughput is normalized by the complete "
            "kernel elapsed interval and is not an independent resource "
            "access-class capacity observation")
    return replace(normalized, **replacements)


def _expected_metax_profiled_workgroups(kernel, analytical, metrics):
    """Return the WORKGROUPS contract for one structured C500 launch."""

    physical_workgroups = int(analytical.grid_size)
    if kernel.kind is KernelKind.GEMM:
        if (kernel.variant != "gemm"
                or kernel.launch.algorithm != "observed_mcblas_gemm:gemm"):
            return physical_workgroups
        tiling = kernel.launch.tiling
        if tiling is None:
            raise ValueError(
                "observed C500 mcBLAS GEMM requires structured tiling")
        tile_shape = tiling.tile_shape
        if {"m", "n"} - set(tile_shape):
            raise ValueError(
                "observed C500 mcBLAS GEMM requires M/N tile dimensions")
        if tiling.split_k is None:
            raise ValueError(
                "observed C500 mcBLAS GEMM requires explicit split_k")
        return gemm_logical_workgroups(
            batch=int(kernel.shape["b"]),
            m=int(kernel.shape["m"]),
            n=int(kernel.shape["n"]),
            tile_m=int(tile_shape["m"]),
            tile_n=int(tile_shape["n"]),
            split_k=int(tiling.split_k),
        )
    if kernel.kind is not KernelKind.FLASH_ATTN:
        return physical_workgroups
    if (kernel.variant != "prefill"
            or kernel.launch.algorithm not in {
                "observed_metax_flash_attention_packed_varlen_queue",
                "observed_metax_flash_attention_trait_packed:"
                "prefill_q64_full_tile",
            }):
        return physical_workgroups
    # This is an evaluation-only counter integrity rule.  Reparse the raw
    # symbol here instead of publishing its template integers as production
    # semantic tiling.  The dataset contract narrowly establishes that the
    # tile64/block256 chunked variant reports one padded job per query head.
    observed_tile = parse_flash_attention_tile(metrics["kernel_name"])
    if (observed_tile != (64, 64)
            or metrics["block_size"] != 256):
        return physical_workgroups
    tiling = kernel.launch.tiling
    if tiling is None:
        raise ValueError(
            "packed C500 FLASH_ATTN launch requires split metadata")
    if tiling.num_kv_splits is None:
        raise ValueError(
            "packed tile64 C500 FLASH_ATTN requires num_kv_splits")
    return flash_attention_packed_varlen_scheduled_workgroups(
        batch=int(kernel.shape["batch"]),
        q_len=int(kernel.shape["q_len"]),
        tile_q=int(observed_tile[0]),
        query_heads=int(kernel.shape["num_query_heads"]),
        query_heads_per_workgroup=1,
        splits=int(tiling.num_kv_splits),
    )


def _normalize_metax(
        kernel, analytical, profiler_record, metrics, hardware,
        provenance, unavailable):
    metax = profiler_record["MetaX"]
    scalars = metax["all_scalars"]
    roofline = metax["roofline"]
    if metax["source"] != "per_invocation":
        raise ValueError(
            "MetaX Formula Ceiling requires per-invocation counters, got "
            f"{metax['source']!r}")

    average_busy_cycles = _finite_number(
        scalars, "Average AP busy cycles", "MetaX.all_scalars")
    core_clock_ghz = hardware.nominal_sm_freq_hz / 1e9
    if "core_clk" in roofline:
        reported_clock = _finite_number(
            roofline, "core_clk", "MetaX.roofline")
        if reported_clock > 0:
            core_clock_ghz = reported_clock
    if core_clock_ghz <= 0:
        raise ValueError("MetaX core clock must be positive")

    replacements = {}
    duty_to_role = {
        "AP MMA Duty ratio": "compute_tensor",
        "AP STE Duty ratio": "compute_alu",
        "AP MTE Duty ratio": "xbar",
    }
    duties = {
        metric_name: _finite_number(
            scalars, metric_name, "MetaX.all_scalars")
        for metric_name in duty_to_role
    }
    integrity_issues = [
        f"{metric_name}={duty} is outside [0, 100]"
        for metric_name, duty in duties.items()
        if not 0.0 <= duty <= 100.0
    ]
    workgroups = _finite_number(
        scalars, "WORKGROUPS", "MetaX.all_scalars")
    expected_workgroups = _expected_metax_profiled_workgroups(
        kernel, analytical, metrics)
    if abs(workgroups - expected_workgroups) > 0.1:
        integrity_issues.append(
            "profiled workgroup count is incompatible with the launch "
            f"contract: observed={workgroups}, "
            f"expected={expected_workgroups}, "
            f"physical_grid={analytical.grid_size}")
    # AP-active/elapsed is not the formula's per-active-unit wave efficiency.
    # It mixes idle APs into the ratio and, when elapsed comes from mcTracer,
    # also divides counters from an instrumented replay by the evaluation
    # target.  Retain the production wave model instead of manufacturing a
    # target-assisted Oracle value.
    # The AP-cycle denominator is tool-version dependent: normal reports may
    # average over every physical AP while roofline-crash reports may average
    # only over participating APs.  Its ratio cannot prove launch identity or
    # replace the launch-derived active-unit count.
    if average_busy_cycles <= 0:
        integrity_issues.append("Average AP busy cycles is zero")

    # A layer-1 filename match is necessary but not sufficient provenance:
    # older C500 captures contain reports whose duty fields exceed 100% or
    # whose active-AP count belongs to another launch.  In that state none of
    # the report's counters can be attributed to this KernelSpec.  Keep the
    # analytical input and expose the rejection instead of mixing true launch
    # metadata with unrelated profiler evidence.
    if integrity_issues:
        reason = "MetaX layer-1 counter provenance failed: " + "; ".join(
            integrity_issues)
        unavailable["active_units"] = reason
        for role in duty_to_role.values():
            unavailable[f"resource_time_overrides_ns.{role}"] = reason
        unavailable["resource_time_overrides_ns.l2_cache"] = reason
        unavailable["resource_time_overrides_ns"] = reason
        for role in ("l1_cache", "xbar", "l2_cache", "dram"):
            unavailable[f"demand_per_resource.{role}"] = reason
        return analytical

    # mcProfiler serializes/instruments C500 collection.  The instruction and
    # byte counters remain count-like evidence, but Average AP busy cycles and
    # every duty ratio are measured inside that instrumented replay.  Real
    # examples with an exact WORKGROUPS match show 47 us in mcTracer versus
    # 17 ms in the layer-1 report.  Turning replay busy cycles into an absolute
    # pipe interval would therefore measure profiler overhead, not a production
    # formula intermediate.  Keep production pipe demand and mark the missing
    # independent interval explicitly.
    pipe_reason = (
        "MetaX AP busy cycles and duty ratios are collected in an instrumented "
        "mcProfiler replay and do not provide an independent production pipe "
        "service interval")
    for role in duty_to_role.values():
        unavailable[f"resource_time_overrides_ns.{role}"] = pipe_reason
    unavailable["resource_time_overrides_ns"] = pipe_reason

    # L2C duty is a controller duty percentage, not an L2 service interval.
    # It can be tiny even for a large memory-bound GEMV and therefore cannot
    # replace the formula's L2 resource time.
    unavailable["resource_time_overrides_ns.l2_cache"] = (
        "MetaX L2C Duty ratio is not an independently measured L2 service "
        "interval")

    tool_limitation = None
    if "tool_limitation" in metax:
        tool_limitation = metax["tool_limitation"]
    measured_demand = {}
    traffic_roles = ("l1_cache", "xbar", "l2_cache", "dram")
    if not hardware.profiler_per_kernel_traffic_reliable:
        reason = (
            tool_limitation if tool_limitation is not None else
            "MetaX HWSpec declares mcProfiler per-kernel traffic counters "
            "unreliable")
        for role in traffic_roles:
            unavailable[f"demand_per_resource.{role}"] = reason
    elif tool_limitation is None:
        required_roofline = {
            "l2c_memacs", "vl1_memacs", "all_memacs"}
        missing_roofline = required_roofline - set(roofline)
        if missing_roofline:
            raise KeyError(
                "MetaX.roofline is missing fields: "
                f"{sorted(missing_roofline)}")
        l2_bytes_raw = _finite_number(
            roofline, "l2c_memacs", "MetaX.roofline")
        vl1_bytes = _finite_number(
            roofline, "vl1_memacs", "MetaX.roofline")
        global_bytes = _finite_number(
            roofline, "all_memacs", "MetaX.roofline")
        # The upstream L2C byte counter and downstream global-memory counter
        # use different transaction granularities.  A partial or unaligned L2
        # request can fetch a complete cache line from global memory, so the
        # physical service envelope is their maximum rather than an assumed
        # byte-wise hierarchy ordering.
        l2_service_bytes = max(l2_bytes_raw, global_bytes)
        candidate_demand = {
            "l1_cache": vl1_bytes,
            "xbar": l2_bytes_raw,
            "l2_cache": l2_service_bytes,
            "dram": global_bytes,
        }
        # mcProfiler occasionally emits a correctly named, correctly sized
        # per-kernel file whose VL1/L2 byte totals are nevertheless inherited
        # from another device-wide collection window.  Physical-throughput
        # validation cannot detect that when the instrumented replay happens
        # to be long.  Validate the independent byte and hierarchy-instruction
        # counters instead.  MetaX L2C instructions address one 128-byte cache
        # line; allow two lines for aggregation slack.  VL1 vector requests can
        # cover two lines per half-wave; allow four lines.  Both bounds are
        # deliberately above every valid counter encoding, but reject the
        # corrupted 1.6 KiB--84 MiB per-instruction reports.
        cacheline_bytes = float(hardware.cacheline_bytes)
        vl1_issue = _metax_counter_volume_issue(
            scalars,
            vl1_bytes,
            (
                "VL1 Read Instructions",
                "VL1 Write Instructions",
                "VL1 Atomic Instructions",
            ),
            4.0 * cacheline_bytes,
            "VL1",
        )
        l2_issue = _metax_counter_volume_issue(
            scalars,
            l2_bytes_raw,
            (
                "L2C Read Instructions",
                "L2C Write Instructions",
                "L2C Atomic Instructions",
            ),
            2.0 * cacheline_bytes,
            "L2C",
        )
        # DNoC request counts independently validate the downstream bytes.
        # Canonical C500 evidence clusters around one 128-byte cache line per
        # request; allow two lines as a conservative encoding/aggregation
        # bound.  This is intentionally independent of replay duration.
        global_issue = _metax_global_memory_volume_issue(
            scalars, global_bytes, 2.0 * cacheline_bytes)
        counter_issues = {}
        if vl1_issue is not None:
            counter_issues["l1_cache"] = vl1_issue
        if l2_issue is not None:
            for role in ("xbar", "l2_cache"):
                counter_issues[role] = l2_issue
        if global_issue is not None:
            counter_issues["l2_cache"] = global_issue
            counter_issues["dram"] = global_issue
        profiler_during_ns = _finite_number(
            metax, "during", "MetaX")
        if profiler_during_ns <= 0:
            reason = (
                "MetaX report has no positive profiler-local collection "
                "window for traffic integrity validation")
            for role in traffic_roles:
                unavailable[f"demand_per_resource.{role}"] = reason
        else:
            # ``during`` is used only as an internal integrity check, never as
            # a formula input or in a derived prediction.  A count requiring
            # more than 125% of the hardware's physical peak during its own
            # report window cannot belong to one invocation (several legacy
            # reports contain 8 GB of L2 traffic for a 50 us, 80-CTA GEMM).
            # The slack covers clock rounding and measured-vs-theoretical peak.
            for role, demand_bytes in candidate_demand.items():
                if role in counter_issues:
                    unavailable[f"demand_per_resource.{role}"] = (
                        counter_issues[role])
                    continue
                resource = hardware.resource_by_role(role)
                peak_bytes_per_second = resource.effective_capacity(
                    analytical.active_units,
                    sm_freq_hz=core_clock_ghz * 1e9,
                    dram_freq_hz=hardware.nominal_dram_freq_hz,
                )
                observed_bytes_per_second = (
                    demand_bytes / (profiler_during_ns * 1e-9))
                if observed_bytes_per_second <= 1.25 * peak_bytes_per_second:
                    measured_demand[role] = demand_bytes
                    provenance[f"demand_per_resource.{role}"] = _source(
                        "profiler", "metax_mcprofiler",
                        f"roofline {role} bytes")
                else:
                    unavailable[f"demand_per_resource.{role}"] = (
                        "MetaX traffic counter exceeds the physical peak in "
                        "its profiler-local window: "
                        f"observed={observed_bytes_per_second} B/s, "
                        f"peak={peak_bytes_per_second} B/s")
            memory_pair = {"l2_cache", "dram"}
            measured_memory_pair = memory_pair.intersection(measured_demand)
            if measured_memory_pair and measured_memory_pair != memory_pair:
                # The unified GPU memory formula consumes DRAM/L2 as one
                # physical hierarchy and derives its miss fraction from their
                # ratio.  Replacing only one level can combine a profiler count
                # with an analytical count from a different traffic model and
                # manufacture DRAM > L2.  Treat the pair as one observation.
                pair_reason = (
                    "MetaX L2 and DRAM traffic counters must both pass the "
                    "per-invocation integrity gates")
                for role in memory_pair:
                    if role in measured_demand:
                        del measured_demand[role]
                    provenance[f"demand_per_resource.{role}"] = _source(
                        "analytical", "production_predictor", role)
                    path = f"demand_per_resource.{role}"
                    if path not in unavailable:
                        unavailable[path] = pair_reason
    else:
        for role in traffic_roles:
            unavailable[f"demand_per_resource.{role}"] = tool_limitation

    normalized = (
        _replace_demand(analytical, measured_demand)
        if measured_demand else analytical)
    return replace(normalized, **replacements)


def _raw_ascend_time_ns(raw, key):
    if key not in raw or raw[key] == "NA":
        return None
    value = to_float(raw[key])
    if value < 0:
        raise ValueError(f"Ascend raw {key} cannot be negative: {value}")
    return value * 1000.0


def _raw_ascend_count(raw, key, scale):
    if key not in raw or raw[key] == "NA":
        return None
    value = to_float(raw[key]) * scale
    if value < 0:
        raise ValueError(f"Ascend raw {key} cannot be negative: {value}")
    return value


def _normalize_ascend(
        analytical, metrics, hardware, provenance, unavailable):
    ascend = metrics["ascend"]
    if ascend is None:
        raise ValueError("Ascend hardware requires an Ascend native payload")
    raw = ascend["raw"]
    if not isinstance(raw, dict):
        raise TypeError("Ascend.raw must be a mapping")

    measured_demand = {}
    analytical_compute_roles = {
        demand.resource_role for demand in analytical.compute_demands
    }
    cube_fops = _finite_number(ascend, "cube_fops", "Ascend")
    if cube_fops > 0 and "compute_tensor" in analytical_compute_roles:
        measured_demand["compute_tensor"] = cube_fops
        provenance["demand_per_resource.compute_tensor"] = _source(
            "profiler", "ascend_msprof", "aic_cube_fops")
    vector_fops = _raw_ascend_count(raw, "aiv_vec_fops", 1.0)
    if (vector_fops is not None
            and "compute_vector" in analytical_compute_roles):
        measured_demand["compute_vector"] = vector_fops
        provenance["demand_per_resource.compute_vector"] = _source(
            "profiler", "ascend_msprof", "aiv_vec_fops")
    elif vector_fops is not None:
        # A mixed task can report work on a secondary domain that is absent
        # from the production static schedule.  The native pipe-time override
        # below still prices that domain.  Do not invent a structured compute
        # demand with an unknown shape/schedule efficiency merely to carry the
        # observed operation count.
        pass
    elif metrics["op_type"] == "cube":
        measured_demand["compute_vector"] = 0.0
        provenance["demand_per_resource.compute_vector"] = _source(
            "profiler", "ascend_msprof",
            "inactive AIV domain for pure Cube execution")
    elif _finite_number(
            ascend, "aiv_vec_time_ns", "Ascend") > 0:
        # Some timeline-only vector captures expose native pipe time but not
        # the underlying FLOP count.  The time override below is sufficient to
        # replace this resource in the formula; retaining a zero analytical
        # demand does not affect the overridden result.
        pass
    else:
        unavailable["demand_per_resource.compute_vector"] = (
            "Ascend aiv_vec_fops is unavailable")
    if metrics["op_type"] == "vector":
        # AIV vector work executes on the separate Vector domain.  GPU-shaped
        # FMA/SFU/ALU demands in the compatibility record are placeholders and
        # must not remain alongside the native Vector demand.
        for role in ("compute_tensor", "compute_fma", "compute_sfu", "compute_alu"):
            measured_demand[role] = 0.0
            provenance[f"demand_per_resource.{role}"] = _source(
                "profiler", "ascend_msprof",
                "not applicable to pure AIV vector execution")

    gm_to_l1_bytes = _finite_number(
        ascend, "gm_to_l1_bytes", "Ascend")
    gm_to_ub_bytes = _raw_ascend_count(
        raw, "GM_to_UB_datas(KB)", 1024.0)
    if gm_to_ub_bytes is None:
        gm_to_ub_bytes = 0.0
    ub_to_gm_bytes = _raw_ascend_count(
        raw, "UB_to_GM_datas(KB)", 1024.0)
    if ub_to_gm_bytes is None:
        ub_to_gm_bytes = 0.0
    read_interface_bytes = gm_to_l1_bytes + gm_to_ub_bytes
    write_interface_bytes = ub_to_gm_bytes
    _finite_number(ascend, "dram_read_bytes", "Ascend")
    memory_path_names = tuple(path.name for path in analytical.memory_paths)
    is_grouped_matmul_split = (
        set(memory_path_names) == {"mte2_read", "aic_output_l2_write"}
        and len(memory_path_names) == 2)
    if is_grouped_matmul_split:
        reason = (
            "Ascend GroupedMatmul memory observations are validated against "
            "the separate GM-to-L1 read and AIC output-write interfaces; no "
            "aggregate Oracle replacement is semantically valid")
        for role in ("xbar", "l2_cache", "dram", "mte3"):
            unavailable[f"demand_per_resource.{role}"] = reason
    else:
        if metrics["op_type"] not in ("cube", "vector", "mix"):
            raise ValueError(
                "unknown Ascend execution domain in Oracle normalization: "
                f"{metrics['op_type']!r}")
        if read_interface_bytes > 0 or write_interface_bytes > 0:
            reason = (
                "Ascend exposes directional MTE2/MTE3 and derived DRAM-read "
                "observations; a flat role replacement would discard their "
                "direction and allocate bytes using the predicted path mix")
        else:
            reason = "Ascend directional memory traffic counters are unavailable"
        for role in ("xbar", "l2_cache", "dram", "mte3"):
            unavailable[f"demand_per_resource.{role}"] = reason

    resource_times = {}
    summary_times = {
        "compute_tensor": _finite_number(
            ascend, "aic_cube_time_ns", "Ascend"),
        "compute_alu": _finite_number(
            ascend, "aic_scalar_time_ns", "Ascend"),
        "compute_vector": _finite_number(
            ascend, "aiv_vec_time_ns", "Ascend"),
        "scalar_aiv": _finite_number(
            ascend, "aiv_scalar_time_ns", "Ascend"),
    }
    for role, duration_ns in summary_times.items():
        if duration_ns > 0:
            resource_times[role] = duration_ns
            provenance[f"resource_time_overrides_ns.{role}"] = _source(
                "profiler", "ascend_msprof", f"native {role} pipe time")

    aic_mte2_ns = _finite_number(
        ascend, "aic_mte2_time_ns", "Ascend")
    aiv_mte2_ns = _raw_ascend_time_ns(raw, "aiv_mte2_time(us)")
    if aic_mte2_ns > 0:
        l2_paths = tuple(
            path.name
            for path in analytical.memory_paths
            if any(
                node.resource_role == "l2_cache"
                for node in path.nodes)
        )
        unavailable["resource_time_overrides_ns.l2_cache"] = (
            "Ascend aic_mte2_time is the mean per-AIC read-side pipe "
            "interval, not the full-chip shared L2 service makespan; it "
            "cannot override aggregate L2 service for structured paths: "
            f"{l2_paths}")
    if aiv_mte2_ns is not None and aiv_mte2_ns > 0:
        resource_times["ub"] = aiv_mte2_ns
        provenance["resource_time_overrides_ns.ub"] = _source(
            "profiler", "ascend_msprof", "aiv_mte2_time")
    unavailable["resource_time_overrides_ns.xbar"] = (
        "Ascend MTE2 pipe intervals are domain-local transfer times, not an "
        "observation of the predictor's xbar resource")
    aiv_mte3_ns = _raw_ascend_time_ns(raw, "aiv_mte3_time(us)")
    if aiv_mte3_ns is not None and aiv_mte3_ns > 0:
        resource_times["mte3"] = aiv_mte3_ns
        provenance["resource_time_overrides_ns.mte3"] = _source(
            "profiler", "ascend_msprof", "aiv_mte3_time")
    aic_mte1_ns = _raw_ascend_time_ns(raw, "aic_mte1_time(us)")
    if aic_mte1_ns is not None and aic_mte1_ns > 0:
        resource_times["l1_cache"] = aic_mte1_ns
        provenance["resource_time_overrides_ns.l1_cache"] = _source(
            "profiler", "ascend_msprof", "aic_mte1_time")

    pipeline = analytical.pipeline_plan
    if isinstance(pipeline, (BarrierPipelineEnvelopeDemand, PipelineWorkPlan)):
        entries = (
            pipeline.classes
            if isinstance(pipeline, BarrierPipelineEnvelopeDemand)
            else pipeline.tasks)
        logical_tasks = tuple(
            task
            for entry in entries
            for sequence in (
                entry.patterns
                if isinstance(entry, PipelineTaskProgramDemand)
                else (entry,))
            for task in (
                sequence.tasks
                if isinstance(sequence, PipelineTaskSequenceDemand)
                else (sequence,))
        )
        source_bound_memory_roles = {
            demand.resource_role
            for task in logical_tasks
            for stage in task.stages
            for demand in (
                stage.memory_node_demands
                + stage.launch_shared_memory_node_demands
                + stage.memory_exact_request_demands)
        }
        for role in tuple(resource_times):
            if role not in source_bound_memory_roles:
                continue
            del resource_times[role]
            provenance_path = f"resource_time_overrides_ns.{role}"
            if provenance_path not in provenance:
                raise RuntimeError(
                    "Ascend resource-time override has no provenance before "
                    f"pipeline compatibility filtering: {role!r}")
            del provenance[provenance_path]
            unavailable[provenance_path] = (
                "Ascend aggregate pipe time cannot replace a source-closed "
                "pipeline path with exact request or absolute per-stage "
                "ownership")

    replacements = {
        "resource_time_overrides_ns": resource_times,
    }
    provenance["resource_time_overrides_ns"] = _source(
        "profiler", "ascend_msprof", "native AIC/AIV pipe times")
    normalized = (
        _replace_demand(analytical, measured_demand)
        if measured_demand else analytical)

    if not resource_times:
        unavailable["resource_time_overrides_ns"] = (
            "Ascend native pipe times are unavailable")
    return replace(normalized, **replacements)


def normalize_oracle_observation(
        kernel, profiler_record, hardware, *, predictor=None):
    """Return a provenance-bearing Formula Ceiling observation."""

    if predictor is None:
        predictor = create_predictor(kernel.kind)
    elif not isinstance(predictor, KernelPredictor):
        raise TypeError(
            "normalize_oracle_observation predictor must be a "
            f"KernelPredictor, got {type(predictor).__name__}")
    elif predictor.kind is not kernel.kind:
        raise ValueError(
            "normalize_oracle_observation predictor kind does not match "
            f"kernel kind: {predictor.kind.value!r} != "
            f"{kernel.kind.value!r}")
    analytical, _ = predictor.oracle_resource_input(kernel, hardware)
    if not isinstance(analytical, PredictionInput):
        raise TypeError(
            "production predictor oracle_resource_input must return "
            "PredictionInput")
    metrics = extract_ncu_metrics(profiler_record)
    if metrics["record_kind"] != "ncu_sections":
        raise NotImplementedError(
            "Oracle observation requires profiler counter sections; "
            f"{hardware.hardware_id!r} records are {metrics['record_kind']!r}")
    provenance = _base_provenance(analytical)
    unavailable = {}
    hardware_id = hardware.hardware_id
    if hardware_id == "nvidia_a800":
        normalized = _normalize_nvidia(
            analytical, metrics, hardware, provenance, unavailable)
        backend = "nvidia_ncu"
    elif hardware_id == "metax_c500":
        normalized = _normalize_metax(
            kernel, analytical, profiler_record, metrics, hardware,
            provenance, unavailable)
        backend = "metax_mcprofiler"
    elif hardware_id == "ascend_910b":
        normalized = _normalize_ascend(
            analytical, metrics, hardware, provenance, unavailable)
        backend = "ascend_msprof"
    else:
        raise NotImplementedError(
            f"Oracle observation is not implemented for {hardware_id!r}")
    _mark_consumed_analytical_inputs_unavailable(
        normalized, provenance, unavailable, backend)
    return OracleObservation(
        hardware_id=hardware_id,
        backend=backend,
        analytical_input=analytical,
        prediction_input=normalized,
        field_provenance=provenance,
        unavailable_fields=unavailable,
    )
