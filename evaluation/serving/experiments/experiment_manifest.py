"""Load and validate Serving experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import yaml

from evaluation.serving.metrics.interactivity_tgs import (
    load_interactivity_tgs_contract,
)
from evaluation.serving.metrics.slo_goodput import load_slo_contract
from src.kernel.hardware_ids import canonical_hardware_id
from src.serving.backend.vllm import (
    VLLMSchedulerConfig,
    mtp_scenario_manifest,
    normalize_mtp_scenario_manifest,
)
from src.serving.deployment.af_disagg import build_af_disagg_plan
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    RouteConfig,
    RouterConfig,
    WorkerPoolConfig,
)
from src.serving.kv_cache.kv_lifecycle_replay import (
    load_kv_lifecycle_replay,
)
from src.serving.parallelism.parallel_config import ParallelConfig, ParallelConfigError
from src.serving.request_gen.production_trace import (
    ProductionTraceReaderConfig,
    SuccessfulRequestFilter,
    load_production_trace,
    production_trace_manifest,
)
from src.serving.schedule.backend_protocol import BackendConfig, backend_config_sha256
from src.serving.workload.model_config import (
    is_validated_qwen3_tp2_host_kv_model_scope,
    ModelConfig,
    load_model_config,
)
from src.serving.workload.runtime_inputs import load_expert_routing_replay


RUNTIME_MEMORY_POLICY_PATH_TEMPLATE = (
    "configs/runtime_memory_policy/{hardware_id}/{model_id}.json"
)


def resolve_runtime_memory_policy_path(
    template: object,
    *,
    repository_root: str | Path,
    hardware_id: str,
    model_id: str,
) -> Path:
    """Resolve one canonical hardware/model policy without an implicit fallback."""

    if template != RUNTIME_MEMORY_POLICY_PATH_TEMPLATE:
        raise ValueError("runtime memory policy path template mismatch")
    for name, value in (("hardware_id", hardware_id), ("model_id", model_id)):
        if not isinstance(value, str) or not value or "/" in value:
            raise ValueError(
                f"runtime memory policy {name} must be a non-empty path atom")
    relative_path = template.format(hardware_id=hardware_id, model_id=model_id)
    return (Path(repository_root) / relative_path).resolve(strict=False)


VLLM_CONFORMANCE_SCHEDULER_DEFAULTS = {
    "max_num_seqs": 1,
    "max_num_batched_tokens": 8192,
    "block_size": 16,
    "num_gpu_blocks_override": 40000,
}

_QWEN3_TP1_SCHEDULER_MODEL_CANONICAL_CONFIG_SHA256 = (
    "a464d3ea8d95f4fe2e9c3c7ffaf330a412a5c158c0dfb84b7ba4343a2966e489"
)
_QWEN3_TP2_SCHEDULER_MODEL_CANONICAL_CONFIG_SHA256 = (
    "ac7267c100ed2ddc531f72c054dbcebbc438eb4a9488e4f1e15b7f483c7cf907"
)


def _parallel_worker_count_util(parallelism: dict[str, Any]) -> int:
    config = ParallelConfig(
        tensor_parallel_size=_require_positive_int_value(
            parallelism["tensor_parallel_size"], "tensor_parallel_size"),
        pipeline_parallel_size=_require_positive_int_value(
            parallelism["pipeline_parallel_size"], "pipeline_parallel_size"),
        data_parallel_size=_require_positive_int_value(
            parallelism["data_parallel_size"], "data_parallel_size"),
    )
    return config.worker_count


def _validate_gpu_capacity_util(
    label: str, gpus: list[int], parallelism: dict[str, Any],
) -> None:
    required = _parallel_worker_count_util(parallelism)
    if len(gpus) < required:
        raise ParallelConfigError(
            f"{label}: requires {required} GPUs but only {len(gpus)} provided: {gpus}")


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentManifest:
    version: int
    cells: list["ExperimentCell | ProductionTraceSweepCell | SimulationCell"]
    path: Path
    source_sha256: str
    case_manifest_path: Path | None


@dataclass(frozen=True)
class ExperimentCell:
    id: str
    case_id: str
    model: Path
    deployment: str
    workload: dict[str, Any]
    sampling: dict[str, Any]
    benchmark: dict[str, Any]
    prediction: dict[str, Any]
    metrics: list[str]
    gpus: list[int] | None
    parallelism: dict[str, Any] | None
    model_type: str
    prefill: dict[str, Any] | None
    decode: dict[str, Any] | None
    router: dict[str, Any] | None
    kv_connector: str | None
    moe: dict[str, Any] | None
    expert_routing: dict[str, Any] | None

    @property
    def required_gpu_count(self) -> int:
        if self.deployment == "disaggregated":
            if self.prefill is None or self.decode is None:
                raise ManifestValidationError(
                    f"{self.id}: disaggregated cell requires prefill/decode")
            return _worker_count(self.prefill) + _worker_count(self.decode)
        if self.deployment == "colocated":
            if self.parallelism is None:
                raise ManifestValidationError(
                    f"{self.id}: colocated cell requires parallelism")
            return _parallel_worker_count(self.parallelism)
        raise ManifestValidationError(
            f"{self.id}: unknown deployment {self.deployment!r}")


@dataclass(frozen=True)
class ProductionTraceSweepCell:
    """One immutable, CPU-only projected Production Trace experiment."""

    id: str
    input_kind: str
    trace_path: Path
    trace: dict[str, Any]
    arrival_rate_scales: tuple[float, ...]
    model: Path
    model_config_sha256: str
    simulation: dict[str, Any]
    analysis: dict[str, Any]


@dataclass(frozen=True)
class SimulationCell:
    """A CPU simulation using the existing simulator's option contract."""

    id: str
    options: dict[str, Any]


def load_experiment_manifest(
    path: str | Path,
    project_root: str | Path | None = None,
) -> ExperimentManifest:
    manifest_path = Path(path)
    if project_root is None:
        project_root_path = manifest_path.resolve().parent
    else:
        project_root_path = Path(project_root)

    try:
        source_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ManifestValidationError(
            f"cannot read experiment manifest: {manifest_path}") from error
    try:
        raw = json.loads(
            source_bytes,
            object_pairs_hook=_unique_manifest_json_object,
            parse_constant=_reject_manifest_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestValidationError(
            f"invalid experiment manifest JSON: {manifest_path}") from error
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    if not isinstance(raw, dict):
        raise ManifestValidationError("manifest root must be an object")
    version = _require_field(raw, "version", "manifest")
    if version == 1:
        return _load_v1_experiment_manifest(
            raw, manifest_path, source_sha256, project_root_path)
    if version in (2, 3, 4, 5, 6, 7, 8):
        return _load_production_trace_experiment_manifest(
            raw, manifest_path, source_sha256, project_root_path, version)
    if version == 9:
        return _load_simulation_experiment_manifest(raw, manifest_path, source_sha256)
    raise ManifestValidationError(
        "version must be 1, 2, 3, 4, 5, 6, 7, 8, or 9")


def _load_simulation_experiment_manifest(
    raw: dict[str, Any], manifest_path: Path, source_sha256: str,
) -> ExperimentManifest:
    _require_exact_fields(raw, {"version", "cells"}, "manifest v9")
    raw_cells = raw["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ManifestValidationError("manifest v9 cells must be a non-empty list")
    cells = []
    for index, item in enumerate(raw_cells):
        label = f"manifest v9 cells[{index}]"
        if not isinstance(item, dict):
            raise ManifestValidationError(f"{label} must be an object")
        _require_exact_fields(item, {"id", "simulation"}, label)
        cell_id = _require_non_empty_str(item, "id", label)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cell_id) is None:
            raise ManifestValidationError(f"{label}.id must be a safe path component")
        options = item["simulation"]
        if not isinstance(options, dict) or not options:
            raise ManifestValidationError(f"{label}.simulation must be a non-empty object")
        for key in options:
            if key.startswith("no-") and key[3:] in options:
                raise ManifestValidationError(
                    f"{label}: conflicting options {key!r} and {key[3:]!r}")
        for key, value in options.items():
            if re.fullmatch(r"[a-z][a-z0-9-]*", key) is None or key in ("output", "settings-summary", "quiet"):
                raise ManifestValidationError(f"{label}: invalid or reserved option {key!r}")
            if value is None:
                continue
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                continue
            if (isinstance(value, list) and value
                    and all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value)):
                continue
            raise ManifestValidationError(f"{label}: option {key!r} requires a scalar, non-empty scalar list, or null flag")
        cells.append(SimulationCell(cell_id, dict(options)))
    _validate_unique_cells(cells)
    return ExperimentManifest(9, cells, manifest_path, source_sha256, None)


def _load_v1_experiment_manifest(
    raw: dict[str, Any],
    manifest_path: Path,
    source_sha256: str,
    project_root_path: Path,
) -> ExperimentManifest:
    _reject_unknown_fields(
        raw, {"version", "case_manifest", "cells"}, "manifest")
    case_manifest_value = _require_field(raw, "case_manifest", "manifest")
    case_manifest_path = _resolve_project_path(
        case_manifest_value, project_root_path)
    cases = _load_case_manifest(case_manifest_path, project_root_path)
    raw_cells = _require_field(raw, "cells", "manifest")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ManifestValidationError("cells must be a non-empty list")

    cells = [
        _load_cell(cell, cases, project_root_path)
        for cell in raw_cells
    ]
    _validate_unique_cells(cells)
    return ExperimentManifest(
        version=1,
        cells=cells,
        path=manifest_path,
        source_sha256=source_sha256,
        case_manifest_path=case_manifest_path,
    )


def _load_production_trace_experiment_manifest(
    raw: dict[str, Any],
    manifest_path: Path,
    source_sha256: str,
    project_root: Path,
    version: int,
) -> ExperimentManifest:
    if version not in (2, 3, 4, 5, 6, 7, 8):
        raise AssertionError("Production Trace manifest version is invalid")
    label = f"manifest v{version}"
    _reject_unknown_fields(raw, {"version", "cells"}, label)
    raw_cells = _require_field(raw, "cells", label)
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ManifestValidationError(
            f"{label} cells must be a non-empty list")
    cells = [
        _load_production_trace_sweep_cell(
            raw_cell, project_root, f"{label} cells[{index}]", version)
        for index, raw_cell in enumerate(raw_cells)
    ]
    _validate_unique_cells(cells)
    return ExperimentManifest(
        version=version,
        cells=cells,
        path=manifest_path,
        source_sha256=source_sha256,
        case_manifest_path=None,
    )


def _load_production_trace_sweep_cell(
    raw: Any,
    project_root: Path,
    label: str,
    manifest_version: int,
) -> ProductionTraceSweepCell:
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"{label} must be an object")
    _require_exact_fields(
        raw,
        {"id", "input", "model", "simulation", "analysis"},
        label,
    )
    cell_id = _require_non_empty_str(raw, "id", label)
    input_config = _load_production_trace_input(
        _require_dict(raw, "input", label), project_root, cell_id)
    model_path, model_config_sha256, model = _load_production_trace_model(
        _require_dict(raw, "model", label), project_root, cell_id)
    simulation = _load_production_trace_simulation(
        _require_dict(raw, "simulation", label),
        project_root,
        cell_id,
        model,
        manifest_version,
    )
    _validate_production_trace_model_simulation(
        model,
        simulation,
        input_config["trace"],
        input_config["request_shapes"],
        cell_id,
        manifest_version,
    )
    if manifest_version == 7:
        _validate_v7_pure_pd_slice(
            model,
            simulation,
            input_config["trace"],
            input_config["arrival_rate_scales"],
            cell_id,
        )
    if ((simulation["scheduler"]["enable_expert_parallel"]
         or simulation["mtp"]["acceptance_policy"] != "disabled"
         or simulation["kv_lifecycle_replay"] is not None
         or simulation["deployment"] in ("af_disagg", "pd_af_disagg"))
            and input_config["arrival_rate_scales"] != (1.0,)):
        raise ManifestValidationError(
            f"{cell_id}.input.arrival_rate_scales must be [1.0] for the "
            "validated EP, MTP, Host-KV lifecycle replay, AF, or PD+AF "
            "slice")
    analysis = _load_production_trace_analysis(
        _require_dict(raw, "analysis", label), project_root, cell_id)
    return ProductionTraceSweepCell(
        id=cell_id,
        input_kind=input_config["kind"],
        trace_path=input_config["trace_path"],
        trace=input_config["trace"],
        arrival_rate_scales=input_config["arrival_rate_scales"],
        model=model_path,
        model_config_sha256=model_config_sha256,
        simulation=simulation,
        analysis=analysis,
    )


def _load_production_trace_input(
    raw: dict[str, Any],
    project_root: Path,
    cell_id: str,
) -> dict[str, Any]:
    label = f"{cell_id}.input"
    _require_exact_fields(raw, {"kind", "trace", "arrival_rate_scales"}, label)
    kind = _require_non_empty_str(raw, "kind", label)
    if kind != "production_trace_arrival_rate_sweep":
        raise ManifestValidationError(
            f"{label}.kind must be 'production_trace_arrival_rate_sweep'")
    trace_config = _require_dict(raw, "trace", label)
    trace_path, trace_manifest, request_shapes = _load_production_trace_source(
        trace_config, project_root, cell_id)
    raw_scales = _require_field(raw, "arrival_rate_scales", label)
    if not isinstance(raw_scales, list) or not raw_scales:
        raise ManifestValidationError(
            f"{label}.arrival_rate_scales must be a non-empty list")
    scales = []
    for index, value in enumerate(raw_scales):
        if (not _is_finite_real_number(value)
                or value <= 0):
            raise ManifestValidationError(
                f"{label}.arrival_rate_scales[{index}] must be finite and positive")
        scales.append(float(value))
    if len(scales) != len(set(scales)):
        raise ManifestValidationError(
            f"{label}.arrival_rate_scales must be unique")
    return {
        "kind": kind,
        "trace_path": trace_path,
        "trace": trace_manifest,
        "request_shapes": request_shapes,
        "arrival_rate_scales": tuple(scales),
    }


def _load_production_trace_source(
    raw: dict[str, Any],
    project_root: Path,
    cell_id: str,
) -> tuple[Path, dict[str, Any], tuple[tuple[float, int, int, int], ...]]:
    label = f"{cell_id}.input.trace"
    _require_exact_fields(
        raw,
        {
            "path", "source_sha256", "source_format", "input_time_unit",
            "field_mapping", "successful_only", "success_filter",
        },
        label,
    )
    if _require_field(raw, "successful_only", label) is not True:
        raise ManifestValidationError(f"{label}.successful_only must be true")
    source_sha256 = _require_sha256(raw, "source_sha256", label)
    trace_path = _resolve_existing_file(
        _require_field(raw, "path", label), project_root, f"{label}.path")
    actual_sha256 = _file_sha256(trace_path)
    if actual_sha256 != source_sha256:
        raise ManifestValidationError(
            f"{label}.source_sha256 disagrees with {trace_path}: "
            f"expected={source_sha256}, actual={actual_sha256}")
    mapping = _require_dict(raw, "field_mapping", label)
    _require_exact_fields(
        mapping,
        {
            "arrival_time", "input_tokens", "output_tokens",
            "prefix_hit_tokens", "request_id",
        },
        f"{label}.field_mapping",
    )
    for field in (
        "arrival_time", "input_tokens", "output_tokens", "prefix_hit_tokens",
    ):
        _require_non_empty_str(mapping, field, f"{label}.field_mapping")
    request_id_field = mapping["request_id"]
    if request_id_field is not None and (
            not isinstance(request_id_field, str) or not request_id_field):
        raise ManifestValidationError(
            f"{label}.field_mapping.request_id must be null or a non-empty string")
    success_raw = _require_dict(raw, "success_filter", label)
    _require_exact_fields(
        success_raw,
        {"mode", "field_name", "accepted_value", "provenance"},
        f"{label}.success_filter",
    )
    try:
        success_filter = SuccessfulRequestFilter(
            mode=_require_non_empty_str(
                success_raw, "mode", f"{label}.success_filter"),
            field_name=success_raw["field_name"],
            accepted_value=success_raw["accepted_value"],
            provenance=_require_non_empty_str(
                success_raw, "provenance", f"{label}.success_filter"),
        )
        reader_config = ProductionTraceReaderConfig(
            source_format=_require_non_empty_str(raw, "source_format", label),
            arrival_time_field=mapping["arrival_time"],
            input_tokens_field=mapping["input_tokens"],
            output_tokens_field=mapping["output_tokens"],
            prefix_hit_tokens_field=mapping["prefix_hit_tokens"],
            request_id_field=request_id_field,
            input_time_unit=_require_non_empty_str(raw, "input_time_unit", label),
            successful_request_filter=success_filter,
        )
        trace = load_production_trace(trace_path, reader_config)
    except (TypeError, ValueError) as error:
        raise ManifestValidationError(
            f"{label} is invalid: {error}") from error
    trace_manifest = production_trace_manifest(trace)
    if trace_manifest["source_sha256"] != source_sha256:
        raise AssertionError("validated Production Trace SHA256 changed while loading")
    request_shapes = tuple(
        (
            request.arrival_time_s,
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
        )
        for request in trace.requests
    )
    return trace_path, trace_manifest, request_shapes


def _load_production_trace_model(
    raw: dict[str, Any],
    project_root: Path,
    cell_id: str,
) -> tuple[Path, str, ModelConfig]:
    label = f"{cell_id}.model"
    _require_exact_fields(raw, {"path", "config_sha256"}, label)
    model_path = _resolve_project_path(_require_field(raw, "path", label), project_root)
    expected_sha256 = _require_sha256(raw, "config_sha256", label)
    try:
        model = load_model_config(str(model_path))
    except (OSError, TypeError, ValueError) as error:
        raise ManifestValidationError(f"{label} is invalid: {error}") from error
    if model.config_sha256 != expected_sha256:
        raise ManifestValidationError(
            f"{label}.config_sha256 disagrees with the model: "
            f"expected={expected_sha256}, actual={model.config_sha256}")
    if (model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"):
        raise ManifestValidationError(
            f"{label} currently requires BF16 standard-GQA with BF16 weights")
    return model_path, expected_sha256, model


def _load_production_trace_simulation(
    raw: dict[str, Any],
    project_root: Path,
    cell_id: str,
    model: ModelConfig,
    manifest_version: int,
) -> dict[str, Any]:
    label = f"{cell_id}.simulation"
    deployment = _require_non_empty_str(raw, "deployment", label)
    base_fields = {
        "backend", "deployment", "hardware", "prediction_method",
        "launch_policy", "attention_kv_layout",
        "attention_kv_distribution", "scheduler", "expert_routing",
    }
    if manifest_version == 2:
        pass
    elif manifest_version == 3:
        base_fields = base_fields | {"mtp"}
    elif manifest_version == 4:
        base_fields = base_fields | {"kv_lifecycle_replay"}
    elif manifest_version == 5:
        base_fields = base_fields | {"af_disagg"}
    elif manifest_version == 6:
        base_fields = base_fields | {"pd_disagg", "af_disagg"}
    elif manifest_version == 7:
        base_fields = base_fields | {"pd_disagg"}
    elif manifest_version == 8:
        base_fields = base_fields | {"af_disagg", "kv_lifecycle_replay"}
    else:
        raise AssertionError("Production Trace manifest version is invalid")
    if manifest_version in (5, 8):
        if deployment != "af_disagg":
            raise ManifestValidationError(
                f"{label}.deployment must be 'af_disagg' for manifest "
                f"v{manifest_version}")
        simulation_fields = base_fields
    elif manifest_version == 6:
        if deployment != "pd_af_disagg":
            raise ManifestValidationError(
                f"{label}.deployment must be 'pd_af_disagg' for manifest v6")
        simulation_fields = base_fields
    elif manifest_version == 7:
        if deployment != "pd_disagg":
            raise ManifestValidationError(
                f"{label}.deployment must be 'pd_disagg' for manifest v7")
        simulation_fields = base_fields
    elif deployment == "colocated":
        simulation_fields = base_fields
    elif deployment == "pd_disagg":
        simulation_fields = base_fields | {"pd_disagg"}
    else:
        raise ManifestValidationError(
            f"{label}.deployment must be 'colocated' or 'pd_disagg' for "
            "the current Production Trace slice")
    _require_exact_fields(
        raw,
        simulation_fields,
        label,
    )
    fixed_values = {
        "backend": "vllm",
        "attention_kv_layout": "paged_varlen",
        "attention_kv_distribution": "uniform",
    }
    normalized: dict[str, Any] = {"deployment": deployment}
    for field, expected in fixed_values.items():
        value = _require_non_empty_str(raw, field, label)
        if value != expected:
            raise ManifestValidationError(
                f"{label}.{field} must be {expected!r} for the current "
                "Production Trace slice")
        normalized[field] = value
    prediction_method = _require_non_empty_str(
        raw, "prediction_method", label)
    if prediction_method not in {"midlayer", "roofline"}:
        raise ManifestValidationError(
            f"{label}.prediction_method must be 'midlayer' or 'roofline'")
    normalized["prediction_method"] = prediction_method
    launch_policy = _require_non_empty_str(raw, "launch_policy", label)
    if launch_policy not in {"injected_launch", "analytical_launch"}:
        raise ManifestValidationError(
            f"{label}.launch_policy must be 'injected_launch' or 'analytical_launch'")
    normalized["launch_policy"] = launch_policy
    hardware = _require_dict(raw, "hardware", label)
    _require_exact_fields(
        hardware, {"hardware_id", "config_sha256"}, f"{label}.hardware")
    hardware_id = _require_non_empty_str(
        hardware, "hardware_id", f"{label}.hardware")
    config_sha256 = _require_sha256(
        hardware, "config_sha256", f"{label}.hardware")
    hardware_path = _hardware_config_path(hardware_id, project_root)
    actual_sha256 = _file_sha256(hardware_path)
    if actual_sha256 != config_sha256:
        raise ManifestValidationError(
            f"{label}.hardware.config_sha256 disagrees with {hardware_path}: "
            f"expected={config_sha256}, actual={actual_sha256}")
    normalized["hardware"] = {
        "hardware_id": hardware_id,
        "config_path": hardware_path,
        "config_sha256": config_sha256,
    }
    scheduler = _require_dict(raw, "scheduler", label)
    scheduler_base_fields = {
        "max_requests", "max_batched_tokens",
        "kv_block_size_tokens", "max_model_len", "prefill_tp_size",
        "prefill_pp_size", "scheduler_pipeline_depth",
        "enable_expert_parallel",
    }
    if deployment in ("pd_disagg", "pd_af_disagg"):
        scheduler_base_fields = scheduler_base_fields | {"decode_tp_size"}
    capacity_fields = set(scheduler) - scheduler_base_fields
    if capacity_fields not in (
            {"runtime_memory_policy"}, {"debug_num_kv_blocks"}):
        raise ManifestValidationError(
            f"{label}.scheduler must select exactly one capacity input: "
            "runtime_memory_policy or debug_num_kv_blocks")
    scheduler_fields = scheduler_base_fields | capacity_fields
    _require_exact_fields(scheduler, scheduler_fields, f"{label}.scheduler")
    normalized_scheduler = {}
    for field in scheduler_base_fields - {"enable_expert_parallel"}:
        normalized_scheduler[field] = _require_positive_int_value(
            scheduler[field], f"{label}.scheduler.{field}")
    if capacity_fields == {"runtime_memory_policy"}:
        policy_path = _resolve_existing_file(
            scheduler["runtime_memory_policy"],
            project_root,
            f"{label}.scheduler.runtime_memory_policy",
        )
        normalized_scheduler["capacity_mode"] = "analytical"
        normalized_scheduler["runtime_memory_policy"] = policy_path
        normalized_scheduler["num_kv_blocks"] = None
    elif capacity_fields == {"debug_num_kv_blocks"}:
        normalized_scheduler["capacity_mode"] = "debug_explicit_block_count"
        normalized_scheduler["runtime_memory_policy"] = None
        normalized_scheduler["num_kv_blocks"] = _require_positive_int_value(
            scheduler["debug_num_kv_blocks"],
            f"{label}.scheduler.debug_num_kv_blocks",
        )
    else:
        raise AssertionError("validated Scheduler capacity mode disappeared")
    enable_ep = scheduler["enable_expert_parallel"]
    if not isinstance(enable_ep, bool):
        raise ManifestValidationError(
            f"{label}.scheduler.enable_expert_parallel must be a boolean")
    normalized_scheduler["enable_expert_parallel"] = enable_ep
    parallelism = (
        normalized_scheduler["prefill_tp_size"],
        normalized_scheduler["prefill_pp_size"],
    )
    if (parallelism not in ((1, 1), (2, 1), (1, 2), (2, 2))
            or (enable_ep and parallelism != (2, 1))):
        raise ManifestValidationError(
            f"{label}.scheduler currently requires TP/PP in "
            "{(1, 1), (2, 1), (1, 2), (2, 2)}, with expert parallelism "
            "allowed only for TP/PP=(2, 1)")
    if normalized_scheduler["kv_block_size_tokens"] != 16:
        raise ManifestValidationError(
            f"{label}.scheduler.kv_block_size_tokens must be 16 for the "
            "pinned vLLM Scheduler")
    if deployment in ("pd_disagg", "pd_af_disagg"):
        if deployment == "pd_disagg":
            deployment_slice = "PD slice"
        elif deployment == "pd_af_disagg":
            deployment_slice = "PD+AF slice"
        else:
            raise AssertionError(
                "validated Production Trace deployment vanished")
        if parallelism != (1, 1):
            raise ManifestValidationError(
                f"{label}.scheduler prefill TP/PP must be (1, 1) for the "
                f"validated {deployment_slice}")
        if normalized_scheduler["decode_tp_size"] != 1:
            raise ManifestValidationError(
                f"{label}.scheduler.decode_tp_size must be 1 for the "
                f"validated {deployment_slice}")
        if normalized_scheduler["max_requests"] != 1:
            raise ManifestValidationError(
                f"{label}.scheduler.max_requests must be 1 for the "
                f"validated {deployment_slice}")
    normalized["scheduler"] = normalized_scheduler
    if manifest_version in (2, 4, 5, 6, 7, 8):
        normalized_mtp = mtp_scenario_manifest(VLLMSchedulerConfig(
            f"experiment-manifest-v{manifest_version}-disabled-mtp"))
    elif manifest_version == 3:
        mtp_raw = _require_dict(raw, "mtp", label)
        try:
            normalized_mtp = normalize_mtp_scenario_manifest(mtp_raw)
        except (TypeError, ValueError) as error:
            raise ManifestValidationError(
                f"{label}.mtp is invalid: {error}") from error
        if normalized_mtp["acceptance_policy"] not in (
                "disabled", "fixed_count", "maximum_entropy"):
            raise ManifestValidationError(
                f"{label}.mtp currently supports only disabled or "
                "explicit-assumption acceptance")
    else:
        raise AssertionError("Production Trace manifest version is invalid")
    normalized["mtp"] = normalized_mtp
    model_kind = "dense" if model.num_experts == 0 else "moe"
    expert_routing = _validate_expert_routing_config(
        cell_id,
        model_kind,
        _require_field(raw, "expert_routing", label),
        project_root,
    )
    if (expert_routing is not None
            and expert_routing["source_kind"] != "explicit_assumption"):
        raise ManifestValidationError(
            f"{label}.expert_routing currently requires "
            "source_kind='explicit_assumption'")
    normalized["expert_routing"] = expert_routing
    if deployment == "colocated":
        normalized["pd_disagg"] = None
        normalized["af_disagg"] = None
    elif deployment == "pd_disagg":
        pd_instance_pairs = (
            {(1, 1)} if manifest_version == 7 else None)
        normalized["pd_disagg"] = _load_production_trace_pd_disagg(
            _require_dict(raw, "pd_disagg", label),
            label,
            allowed_instance_pairs=pd_instance_pairs,
        )
        normalized["af_disagg"] = None
    elif deployment == "af_disagg":
        normalized["pd_disagg"] = None
        normalized["af_disagg"] = _load_production_trace_af_disagg(
            _require_dict(raw, "af_disagg", label), label)
    elif deployment == "pd_af_disagg":
        normalized["pd_disagg"] = _load_production_trace_pd_disagg(
            _require_dict(raw, "pd_disagg", label),
            label,
            allowed_instance_pairs={(1, 1), (1, 2), (2, 1), (2, 2)},
        )
        normalized["af_disagg"] = (
            _load_production_trace_pd_af_disagg(
                _require_dict(raw, "af_disagg", label), label)
        )
    else:
        raise AssertionError("validated Production Trace deployment vanished")
    if manifest_version in (2, 3):
        normalized["kv_lifecycle_replay"] = None
    elif manifest_version in (4, 8):
        normalized["kv_lifecycle_replay"] = (
            _load_production_trace_kv_lifecycle_replay(
                _require_dict(raw, "kv_lifecycle_replay", label),
                project_root,
                label,
            )
        )
    elif manifest_version in (5, 6, 7):
        normalized["kv_lifecycle_replay"] = None
    else:
        raise AssertionError("Production Trace manifest version is invalid")
    return normalized


def _load_production_trace_kv_lifecycle_replay(
    raw: dict[str, Any],
    project_root: Path,
    simulation_label: str,
) -> dict[str, Any]:
    label = f"{simulation_label}.kv_lifecycle_replay"
    _require_exact_fields(raw, {"path", "source_sha256"}, label)
    replay_path = _resolve_existing_file(
        _require_field(raw, "path", label), project_root, f"{label}.path")
    expected_sha256 = _require_sha256(raw, "source_sha256", label)
    try:
        replay = load_kv_lifecycle_replay(replay_path)
    except (OSError, TypeError, ValueError, NotImplementedError) as error:
        raise ManifestValidationError(f"{label} is invalid: {error}") from error
    if replay.source_sha256 != expected_sha256:
        raise ManifestValidationError(
            f"{label}.source_sha256 disagrees with {replay_path}: "
            f"expected={expected_sha256}, actual={replay.source_sha256}")
    return {
        "path": replay_path,
        "source_sha256": replay.source_sha256,
        "artifact_id": replay.artifact_id,
        "provenance": {
            "source_kind": replay.provenance.source_kind,
            "producer": replay.provenance.producer,
            "producer_version": replay.provenance.producer_version,
            "source_reference": replay.provenance.source_reference,
        },
    }


def _validate_production_trace_model_simulation(
    model: ModelConfig,
    simulation: dict[str, Any],
    trace: dict[str, Any],
    request_shapes: tuple[tuple[float, int, int, int], ...],
    cell_id: str,
    manifest_version: int,
) -> None:
    scheduler = simulation["scheduler"]
    parallelism = (
        scheduler["prefill_tp_size"],
        scheduler["prefill_pp_size"],
    )
    enable_ep = scheduler["enable_expert_parallel"]
    mtp = simulation["mtp"]
    mtp_enabled = mtp["acceptance_policy"] != "disabled"
    kv_lifecycle_identity = simulation["kv_lifecycle_replay"]
    af_disagg = simulation["af_disagg"]
    if simulation["deployment"] == "pd_af_disagg":
        pd_disagg = simulation["pd_disagg"]
        pd_instance_pair = (
            pd_disagg["num_prefill_instances"],
            pd_disagg["num_decode_instances"],
        ) if pd_disagg is not None else None
        validated_trace_scope = (
            (pd_instance_pair == (1, 1)
             and request_shapes == ((0.0, 4, 2, 0),))
            or (pd_instance_pair == (1, 2)
                and request_shapes == (
                    (0.0, 1, 8, 0),
                    (0.0, 1, 8, 0),
                    (0.0, 1, 8, 0),
                ))
            or (pd_instance_pair == (2, 2)
                and request_shapes == (
                    (0.0, 4, 2, 0),
                    (0.0, 4, 2, 0),
                ))
            or (pd_instance_pair == (2, 1)
                and request_shapes == (
                    (0.0, 4, 2, 0),
                    (0.0, 4, 2, 0),
                ))
        )
        if (pd_disagg is None
                or af_disagg is None
                or pd_disagg["router"]["resource_id"] != "router_cpu"
                or parallelism != (1, 1)
                or scheduler["decode_tp_size"] != 1
                or enable_ep
                or mtp_enabled
                or kv_lifecycle_identity is not None
                or simulation["expert_routing"] is not None
                or scheduler["max_requests"] != 1
                or scheduler["max_batched_tokens"] != 128
                or scheduler["num_kv_blocks"] != 128
                or scheduler["kv_block_size_tokens"] != 16
                or scheduler["max_model_len"] != 128
                or simulation["hardware"]["hardware_id"] != "nvidia_a800"
                or simulation["launch_policy"] != "analytical_launch"
                or model.model_type != "qwen3"
                or model.num_experts != 0
                or model.num_hidden_layers != 1
                or model.canonical_config_sha256
                != _QWEN3_TP1_SCHEDULER_MODEL_CANONICAL_CONFIG_SHA256
                or not validated_trace_scope):
            raise ManifestValidationError(
                f"{cell_id}.simulation PD+AF currently requires either one "
                "no-prefix 4->2 request on exactly 1P x 1D, or three "
                "simultaneous no-prefix 1->8 requests on exactly 1P x 2D, "
                "or two simultaneous no-prefix 4->2 requests on exactly "
                "2P x 1D or 2P x 2D, using the exact canonical nvidia_a800 "
                "single-layer Dense BF16 Qwen3 fixture with "
                "one Attention/FFN pair per P or D instance, "
                "Attention TP=PP=1, FFN TP=1, EP/MTP/Host-KV=false, "
                "max_requests=1, max_batched_tokens=128, "
                "num_kv_blocks=128, max_model_len=128, and "
                "launch_policy='analytical_launch'")
        return
    if simulation["deployment"] == "af_disagg":
        if manifest_version == 8:
            validated_trace_and_hbm_scope = (
                (request_shapes == ((0.0, 4, 2, 0),)
                 and scheduler["num_kv_blocks"] == 2)
                or
                (request_shapes == ((0.0, 17, 2, 16),)
                 and scheduler["num_kv_blocks"] == 3)
            )
            if (af_disagg is None
                    or simulation["pd_disagg"] is not None
                    or parallelism != (1, 1)
                    or af_disagg["ffn_tp_size"] != 1
                    or enable_ep
                    or mtp_enabled
                    or kv_lifecycle_identity is None
                    or simulation["expert_routing"] is not None
                    or scheduler["max_requests"] != 1
                    or scheduler["max_batched_tokens"] != 128
                    or scheduler["kv_block_size_tokens"] != 16
                    or scheduler["max_model_len"] != 128
                    or simulation["hardware"]["hardware_id"]
                    != "nvidia_a800"
                    or simulation["launch_policy"] != "analytical_launch"
                    or model.model_type != "qwen3"
                    or model.num_experts != 0
                    or model.canonical_config_sha256
                    != _QWEN3_TP1_SCHEDULER_MODEL_CANONICAL_CONFIG_SHA256
                    or model.hidden_size != 64
                    or model.num_hidden_layers != 1
                    or model.num_attention_heads != 1
                    or model.num_key_value_heads != 1
                    or not validated_trace_and_hbm_scope):
                raise ManifestValidationError(
                    f"{cell_id}.simulation manifest v8 AF+Host-KV "
                    "currently requires either one exact no-prefix 4->2 "
                    "request with two HBM blocks, or one exact block-aligned "
                    "17->2/H16 request with three HBM blocks, "
                    "on one canonical nvidia_a800 single-layer Dense BF16 "
                    "Qwen3 Attention/FFN pair, Attention/FFN TP=1, PP=1, "
                    "EP/MTP=false, max_requests=1, max_batched_tokens=128, "
                    "16-token blocks and exactly one Host block, "
                    "max_model_len=128, and launch_policy='analytical_launch'")
            replay = _reload_production_trace_kv_lifecycle_replay(
                kv_lifecycle_identity,
                cell_id,
            )
            transfer = af_disagg["activation_transfer"]
            router = af_disagg["router"]
            deployment = build_af_disagg_plan(
                DeploymentConfig(
                    "af_disagg",
                    (
                        WorkerPoolConfig(
                            "attention",
                            "attention",
                            simulation["hardware"]["hardware_id"],
                            ranks=(0,),
                            kv_cache=replay.cache_config,
                        ),
                        WorkerPoolConfig(
                            "ffn",
                            "ffn",
                            simulation["hardware"]["hardware_id"],
                            ranks=(1,),
                        ),
                    ),
                    (
                        RouteConfig(
                            "attention", "ffn", transfer["payload"],
                            transfer["algorithm"], transfer["link"],
                            transfer["topology"],
                            resource_id=transfer["resource_id"],
                            backend=transfer["backend"],
                        ),
                        RouteConfig(
                            "ffn", "attention", transfer["payload"],
                            transfer["algorithm"], transfer["link"],
                            transfer["topology"],
                            resource_id=transfer["resource_id"],
                            backend=transfer["backend"],
                        ),
                    ),
                    RouterConfig(
                        router["resource_id"],
                        router["dispatch_service_time_s"],
                    ),
                ),
                num_layers=model.num_hidden_layers,
                num_experts=model.num_experts,
            )
            expected_kv_bytes_per_token = (
                model.kv_bytes_per_token_per_parallel_rank(1, 1))
            backend_config = BackendConfig(
                "attention",
                "attention",
                scheduler["max_batched_tokens"],
                scheduler["max_requests"],
                scheduler["num_kv_blocks"],
            kv_block_size_tokens=scheduler["kv_block_size_tokens"],
            kv_bytes_per_token=expected_kv_bytes_per_token,
            scheduler_pipeline_depth=scheduler["scheduler_pipeline_depth"])
            _validate_exact_production_trace_kv_lifecycle_replay(
                replay,
                model,
                simulation,
                trace,
                cell_id,
                deployment,
                backend_config,
                pool_name="attention",
                backend_role="attention",
                expected_hbm_blocks=(
                    3 if request_shapes == ((0.0, 17, 2, 16),) else 2),
            )
            return
        if manifest_version != 5:
            raise AssertionError("validated AF manifest version vanished")
        validated_af_scope = (
            (parallelism == (1, 1)
             and af_disagg is not None
             and af_disagg["ffn_tp_size"] == 1
             and scheduler["max_batched_tokens"] == 128
             and scheduler["num_kv_blocks"] == 128
             and model.canonical_config_sha256
             == _QWEN3_TP1_SCHEDULER_MODEL_CANONICAL_CONFIG_SHA256
             and model.hidden_size == 64
             and model.num_hidden_layers == 1
             and model.num_attention_heads == 1
             and model.num_key_value_heads == 1
             and request_shapes == ((0.0, 4, 2, 0),))
            or (parallelism == (2, 1)
                and af_disagg is not None
                and af_disagg["ffn_tp_size"] == 2
                and scheduler["max_batched_tokens"] == 4
                and scheduler["num_kv_blocks"] == 8
                and model.canonical_config_sha256
                == _QWEN3_TP2_SCHEDULER_MODEL_CANONICAL_CONFIG_SHA256
                and model.hidden_size == 256
                and model.num_hidden_layers == 2
                and model.num_attention_heads == 4
                and model.num_key_value_heads == 2
                and request_shapes == ((0.0, 2, 1, 0),))
        )
        if (af_disagg is None
                or simulation["pd_disagg"] is not None
                or enable_ep
                or mtp_enabled
                or kv_lifecycle_identity is not None
                or simulation["expert_routing"] is not None
                or scheduler["max_requests"] != 1
                or scheduler["kv_block_size_tokens"] != 16
                or scheduler["max_model_len"] != 128
                or simulation["hardware"]["hardware_id"] != "nvidia_a800"
                or simulation["launch_policy"] != "analytical_launch"
                or model.model_type != "qwen3"
                or model.num_experts != 0
                or not validated_af_scope):
            raise ManifestValidationError(
                f"{cell_id}.simulation AF currently requires either the "
                "exact TP=1/FFN-TP=1 Qwen3 4->2/H0 slice with batch=128 "
                "and 128 KV blocks, or the exact TP=2/FFN-TP=2 Qwen3 "
                "2->1/H0 slice with batch=4 and 8 KV blocks; both require "
                "their exact canonical Dense BF16 Qwen3 fixture, one "
                "nvidia_a800 Attention/FFN pair, PP=1, "
                "EP/MTP/Host-KV=false, max_requests=1, 16-token KV blocks, "
                "max_model_len=128, and launch_policy='analytical_launch'")
        return
    if af_disagg is not None:
        raise AssertionError("non-AF Production Trace retained AF config")
    if kv_lifecycle_identity is not None:
        statistics = trace["statistics"]
        tp_size, pp_size = parallelism
        tp1_scope = tp_size == 1 and pp_size == 1
        tp2_scope = (
            tp_size == 2
            and pp_size == 1
            and is_validated_qwen3_tp2_host_kv_model_scope(model)
            and scheduler["max_batched_tokens"] == 8
            and scheduler["max_model_len"] == 128
            and statistics["input_tokens_total"] == 4
        )
        if (simulation["deployment"] != "colocated"
                or simulation["pd_disagg"] is not None
                or not (tp1_scope or tp2_scope)
                or enable_ep
                or mtp_enabled
                or simulation["expert_routing"] is not None
                or scheduler["max_requests"] != 1
                or simulation["hardware"]["hardware_id"] != "nvidia_a800"
                or simulation["launch_policy"] != "analytical_launch"
                or model.model_type != "qwen3"
                or model.num_experts != 0
                or statistics["request_count"] != 1
                or statistics["output_tokens_total"] != 2
                or statistics["prefix_hit_tokens_total"] != 0):
            raise ManifestValidationError(
                f"{cell_id}.simulation Host-KV lifecycle replay currently "
                "requires one no-prefix request with a two-token completion "
                "on one colocated nvidia_a800 Dense BF16 Qwen3 pool, "
                "either TP=PP=1 or the exact source-backed TP=2/PP=1 "
                "4->2 fixture with max_batched_tokens=8 and "
                "max_model_len=128, EP/MTP=false, max_requests=1, and "
                "launch_policy='analytical_launch'")
        replay = _reload_production_trace_kv_lifecycle_replay(
            kv_lifecycle_identity,
            cell_id,
        )
        expected_kv_bytes_per_token = (
            model.kv_bytes_per_token_per_parallel_rank(tp_size, pp_size))
        hbm_tier = next(
            tier for tier in replay.cache_config.tiers
            if tier.name == "hbm")
        hbm_block_bytes = (
            scheduler["kv_block_size_tokens"]
            * expected_kv_bytes_per_token)
        if hbm_tier.capacity_bytes % hbm_block_bytes:
            raise ManifestValidationError(
                f"{cell_id}.simulation Host-KV replay HBM capacity is not "
                "an integer number of Scheduler KV blocks")
        replay_hbm_blocks = hbm_tier.capacity_bytes // hbm_block_bytes
        backend_config = BackendConfig(
            "main",
            "colocated",
            scheduler["max_batched_tokens"],
            scheduler["max_requests"],
            replay_hbm_blocks,
            kv_block_size_tokens=scheduler["kv_block_size_tokens"],
            kv_bytes_per_token=expected_kv_bytes_per_token,
        scheduler_pipeline_depth=scheduler["scheduler_pipeline_depth"])
        collectives = (
            ()
            if tp_size == 1
            else tuple(
                CollectiveConfig(
                    operation,
                    "ring",
                    "nvlink",
                    "ring",
                    "nvlink:main:tensor-fabric",
                    "analytical_roofline_v1",
                )
                for operation in ("all_reduce", "all_gather")
            )
        )
        deployment = build_colocated_plan(
            DeploymentConfig(
                "colocated",
                (WorkerPoolConfig(
                    "main",
                    "colocated",
                    simulation["hardware"]["hardware_id"],
                    ParallelConfig(tensor_parallel_size=tp_size),
                    kv_cache=replay.cache_config,
                    collectives=collectives,
                ),),
            ),
            num_layers=model.num_hidden_layers,
            num_experts=model.num_experts,
        )
        _validate_exact_production_trace_kv_lifecycle_replay(
            replay,
            model,
            simulation,
            trace,
            cell_id,
            deployment,
            backend_config,
            pool_name="main",
            backend_role="colocated",
            expected_hbm_blocks=2,
            tensor_parallel_size=tp_size,
        )
        return
    if mtp_enabled:
        if (mtp["acceptance_policy"] == "maximum_entropy"
                and mtp["acceptance_assumption"][
                    "request_stream_sha256"]
                != trace["normalized_request_stream_sha256"]):
            raise ManifestValidationError(
                f"{cell_id}.simulation maximum-entropy MTP assumption must "
                "bind the normalized Production Trace request stream")
        statistics = trace["statistics"]
        trace_scope = (
            statistics["request_count"],
            statistics["input_tokens_total"],
            statistics["output_tokens_total"],
            statistics["prefix_hit_tokens_total"],
        )
        validated_trace_scope = (
            trace_scope == (1, 4, 2, 0)
            or (
                trace_scope == (1, 17, 2, 16)
                and mtp["fixed_accepted_tokens"] == 1
            )
        )
        if (simulation["deployment"] != "colocated"
                or simulation["pd_disagg"] is not None
                or parallelism != (1, 1)
                or enable_ep
                or scheduler["max_requests"] != 1
                or simulation["hardware"]["hardware_id"] != "nvidia_a800"
                or simulation["launch_policy"] != "analytical_launch"
                or model.model_type != "glm4_moe"
                or model.mtp_architecture != "glm4_moe_mtp"
                or model.num_mtp_layers != 1
                or model.num_hidden_layers != 2
                or model.num_experts != 2
                or model.num_experts_per_tok != 1
                or model.num_dense_layers != 1
                or model.num_shared_experts != 1
                or mtp["num_speculative_tokens"] != 2
                or mtp["fixed_accepted_tokens"] not in (0, 1, 2)
                or mtp["acceptance_replay"] is not None
                or mtp["validation_status"]
                != "projected_explicit_assumption"
                or not validated_trace_scope):
            raise ManifestValidationError(
                f"{cell_id}.simulation MTP currently requires one exact "
                "4->2 no-prefix request with fixed-count or maximum-entropy "
                "acceptance, "
                "or one exact 17->2 request with Prefix=16 and fixed "
                "acceptance one, on one colocated "
                "nvidia_a800 GLM4-MoE BF16 pool, "
                "TP=PP=1, EP=false, two model layers, one dense MTP layer, "
                "two experts with top-k=1, max_requests=1, exactly two "
                "speculative tokens, an explicit acceptance assumption, and "
                "launch_policy='analytical_launch'")
        if simulation["expert_routing"] is None:
            raise ManifestValidationError(
                f"{cell_id}.simulation MTP MoE requires an expert_routing "
                "object")
        return
    if parallelism == (1, 1) and not enable_ep:
        if model.num_experts != 0:
            raise ManifestValidationError(
                f"{cell_id}.simulation MoE currently requires TP=EP=2")
        return
    if parallelism not in ((2, 1), (1, 2), (2, 2)):
        raise AssertionError(
            "validated Production Trace parallelism vanished")
    label = f"{cell_id}.simulation"
    if enable_ep:
        if (parallelism != (2, 1)
                or simulation["deployment"] != "colocated"
                or simulation["pd_disagg"] is not None
                or scheduler["max_requests"] != 1
                or simulation["hardware"]["hardware_id"] != "nvidia_a800"
                or simulation["launch_policy"] != "analytical_launch"
                or model.model_type != "qwen3_moe"
                or model.num_experts != 2
                or model.num_experts_per_tok != 1
                or model.num_dense_layers != 0
                or model.num_shared_experts != 0
                or model.num_mtp_layers != 0
                or model.attention_layout != "standard_gqa"
                or model.compute_dtype != "bf16"
                or model.weight_dtype != "bf16"):
            raise ManifestValidationError(
                f"{label} EP currently requires one colocated nvidia_a800 "
                "Qwen3MoE BF16 pool with TP=EP=2, two experts, top-k=1, no "
                "dense/shared/MTP layers, max_requests=1, and "
                "launch_policy='analytical_launch'")
        return
    if (simulation["deployment"] != "colocated"
            or simulation["pd_disagg"] is not None
            or scheduler["max_requests"] != 1
            or simulation["hardware"]["hardware_id"] != "nvidia_a800"
            or simulation["launch_policy"] != "analytical_launch"
            or model.model_type != "qwen3"
            or model.num_experts != 0
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"):
        raise ManifestValidationError(
            f"{label} TP/PP={parallelism} currently requires one colocated "
            "nvidia_a800 Dense BF16 Qwen3 pool, EP=false, max_requests=1, and "
            "launch_policy='analytical_launch'")
    if (parallelism[0] == 2
            and (model.num_attention_heads % 2
                 or model.intermediate_size % 2)):
        raise ManifestValidationError(
            f"{label} TP=2 model dimensions are not evenly shardable")
    if (parallelism[1] == 2
            and (model.num_hidden_layers % 2
                 or model.tie_word_embeddings)):
        raise ManifestValidationError(
            f"{label} PP=2 requires an even number of model layers and "
            "tie_word_embeddings=false")
    try:
        model.kv_bytes_per_token_per_parallel_rank(*parallelism)
    except ValueError as error:
        raise ManifestValidationError(
            f"{label} TP/PP={parallelism} model sharding is invalid: "
            f"{error}") from error


def _reload_production_trace_kv_lifecycle_replay(
    replay_identity: dict[str, Any],
    cell_id: str,
) -> Any:
    try:
        replay = load_kv_lifecycle_replay(replay_identity["path"])
    except (OSError, TypeError, ValueError, NotImplementedError) as error:
        raise ManifestValidationError(
            f"{cell_id}.simulation Host-KV lifecycle replay changed after "
            "validation") from error
    if replay.source_sha256 != replay_identity["source_sha256"]:
        raise ManifestValidationError(
            f"{cell_id}.simulation Host-KV lifecycle replay content changed "
            "after validation")
    return replay


def _validate_exact_production_trace_kv_lifecycle_replay(
    replay: Any,
    model: ModelConfig,
    simulation: dict[str, Any],
    trace: dict[str, Any],
    cell_id: str,
    deployment: Any,
    backend_config: BackendConfig,
    *,
    pool_name: str,
    backend_role: str,
    expected_hbm_blocks: int,
    tensor_parallel_size: int = 1,
) -> None:
    scheduler = simulation["scheduler"]
    expected_kv_bytes_per_token = (
        model.kv_bytes_per_token_per_parallel_rank(
            tensor_parallel_size, 1))
    replay_actions = tuple(
        event.action.value
        for step in replay.steps
        for event in step.events
    )
    tiers_by_name = {
        tier.name: tier for tier in replay.cache_config.tiers
    }
    if set(tiers_by_name) != {"hbm", "host"}:
        raise ManifestValidationError(
            f"{cell_id}.simulation Host-KV lifecycle replay requires "
            "exactly the canonical HBM and Host tiers")
    hbm_tier = tiers_by_name["hbm"]
    host_tier = tiers_by_name["host"]
    block_bytes = 16 * expected_kv_bytes_per_token
    if expected_hbm_blocks == 2:
        expected_hbm_description = "two"
    elif expected_hbm_blocks == 3:
        expected_hbm_description = "three"
    else:
        raise AssertionError("unsupported formal Host-KV HBM block count")
    if scheduler["capacity_mode"] == "analytical":
        scheduler_capacity_matches = scheduler["num_kv_blocks"] is None
    elif scheduler["capacity_mode"] == "debug_explicit_block_count":
        scheduler_capacity_matches = (
            scheduler["num_kv_blocks"] == expected_hbm_blocks)
    else:
        raise AssertionError("validated Scheduler capacity mode vanished")
    if (not scheduler_capacity_matches
            or replay.cache_config.hbm_blocks != expected_hbm_blocks
            or hbm_tier.capacity_bytes != expected_hbm_blocks * block_bytes
            or host_tier.capacity_bytes != block_bytes
            or hbm_tier.block_size_tokens != 16
            or host_tier.block_size_tokens != 16):
        raise ManifestValidationError(
            f"{cell_id}.simulation Host-KV lifecycle replay currently "
            f"requires exactly {expected_hbm_description} HBM blocks and "
            "one Host block, each with 16 tokens")
    if (replay.context.model_config_sha256 != model.config_sha256
            or replay.context.request_stream_sha256
            != trace["normalized_request_stream_sha256"]
            or replay.context.deployment_id != deployment.deployment_id
            or replay.context.scheduler_config_sha256
            != backend_config_sha256(backend_config)
            or replay.context.pool_name != pool_name
            or replay.context.backend_role != backend_role
            or replay.cache_config.bytes_per_token
            != expected_kv_bytes_per_token
            or hbm_tier.block_size_tokens
            != scheduler["kv_block_size_tokens"]
            or replay.context.scheduler_step_count != 2
            or replay_actions != ("evict", "load")):
        raise ManifestValidationError(
            f"{cell_id}.simulation Host-KV lifecycle replay context "
            "disagrees with the model, deployment, Scheduler, or exact "
            "EVICT/LOAD slice")


def _validate_v7_pure_pd_slice(
    model: ModelConfig,
    simulation: dict[str, Any],
    trace: dict[str, Any],
    arrival_rate_scales: tuple[float, ...],
    cell_id: str,
) -> None:
    scheduler = simulation["scheduler"]
    pd_disagg = simulation["pd_disagg"]
    statistics = trace["statistics"]
    if (simulation["deployment"] != "pd_disagg"
            or pd_disagg is None
            or simulation["af_disagg"] is not None
            or pd_disagg["num_prefill_instances"] != 1
            or pd_disagg["num_decode_instances"] != 1
            or pd_disagg["router"]["resource_id"] != "router_cpu"
            or arrival_rate_scales != (1.0,)
            or scheduler["prefill_tp_size"] != 1
            or scheduler["prefill_pp_size"] != 1
            or scheduler["decode_tp_size"] != 1
            or scheduler["enable_expert_parallel"]
            or scheduler["max_requests"] != 1
            or scheduler["max_batched_tokens"] != 128
            or scheduler["num_kv_blocks"] != 128
            or scheduler["kv_block_size_tokens"] != 16
            or scheduler["max_model_len"] != 128
            or simulation["mtp"]["acceptance_policy"] != "disabled"
            or simulation["kv_lifecycle_replay"] is not None
            or simulation["expert_routing"] is not None
            or simulation["hardware"]["hardware_id"] != "nvidia_a800"
            or simulation["launch_policy"] != "analytical_launch"
            or model.model_type != "qwen3"
            or model.num_experts != 0
            or model.num_hidden_layers != 1
            or statistics["request_count"] != 1
            or statistics["input_tokens_total"] != 4
            or statistics["output_tokens_total"] != 2
            or statistics["prefix_hit_tokens_total"] != 0):
        raise ManifestValidationError(
            f"{cell_id}.simulation manifest v7 pure PD currently requires "
            "arrival_rate_scales=[1.0], one no-prefix 4-token prompt with a "
            "2-token completion, one nvidia_a800 single-layer Dense BF16 "
            "Qwen3 fixture, exactly 1P x 1D, TP=PP=1, EP/MTP/Host-KV=false, "
            "max_requests=1, max_batched_tokens=128, num_kv_blocks=128, "
            "max_model_len=128, router_cpu, and "
            "launch_policy='analytical_launch'")


def _load_production_trace_af_disagg(
    raw: dict[str, Any],
    simulation_label: str,
) -> dict[str, Any]:
    label = f"{simulation_label}.af_disagg"
    _require_exact_fields(
        raw,
        {"ffn_tp_size", "activation_transfer", "router"},
        label,
    )
    ffn_tp_size = _require_positive_int_value(
        raw["ffn_tp_size"], f"{label}.ffn_tp_size")
    if ffn_tp_size not in (1, 2):
        raise ManifestValidationError(
            f"{label}.ffn_tp_size must be 1 or 2 for the validated AF slice")
    transfer = _require_dict(raw, "activation_transfer", label)
    _require_exact_fields(
        transfer,
        {
            "payload", "algorithm", "link", "topology", "backend",
            "resource_id",
        },
        f"{label}.activation_transfer",
    )
    expected_transfer = {
        "payload": "activation",
        "algorithm": "send_recv",
        "link": "nvlink",
        "topology": "point_to_point",
        "backend": "analytical_roofline_v1",
        "resource_id": "nvlink:attention--ffn",
    }
    normalized_transfer = {}
    for field, expected in expected_transfer.items():
        value = _require_non_empty_str(
            transfer, field, f"{label}.activation_transfer")
        if value != expected:
            raise ManifestValidationError(
                f"{label}.activation_transfer.{field} must be {expected!r}")
        normalized_transfer[field] = value
    router = _require_dict(raw, "router", label)
    _require_exact_fields(
        router,
        {"resource_id", "dispatch_service_time_s"},
        f"{label}.router",
    )
    router_resource_id = _require_non_empty_str(
        router, "resource_id", f"{label}.router")
    if router_resource_id != "router_cpu":
        raise ManifestValidationError(
            f"{label}.router.resource_id must be 'router_cpu' for the "
            "validated AF slice")
    dispatch_service_time_s = router["dispatch_service_time_s"]
    if (not _is_finite_real_number(dispatch_service_time_s)
            or float(dispatch_service_time_s) != 0.0):
        raise ManifestValidationError(
            f"{label}.router.dispatch_service_time_s must be 0.0 for the "
            "validated AF slice")
    return {
        "ffn_tp_size": ffn_tp_size,
        "activation_transfer": normalized_transfer,
        "router": {
            "resource_id": router_resource_id,
            "dispatch_service_time_s": 0.0,
        },
    }


def _load_production_trace_pd_af_disagg(
    raw: dict[str, Any],
    simulation_label: str,
) -> dict[str, Any]:
    label = f"{simulation_label}.af_disagg"
    _require_exact_fields(
        raw,
        {"ffn_tp_size", "activation_transfer"},
        label,
    )
    ffn_tp_size = _require_positive_int_value(
        raw["ffn_tp_size"], f"{label}.ffn_tp_size")
    if ffn_tp_size != 1:
        raise ManifestValidationError(
            f"{label}.ffn_tp_size must be 1 for the validated PD+AF slice")
    transfer = _require_dict(raw, "activation_transfer", label)
    _require_exact_fields(
        transfer,
        {"payload", "algorithm", "link", "topology", "backend"},
        f"{label}.activation_transfer",
    )
    expected_transfer = {
        "payload": "activation",
        "algorithm": "send_recv",
        "link": "nvlink",
        "topology": "point_to_point",
        "backend": "analytical_roofline_v1",
    }
    normalized_transfer = {}
    for field, expected in expected_transfer.items():
        value = _require_non_empty_str(
            transfer, field, f"{label}.activation_transfer")
        if value != expected:
            raise ManifestValidationError(
                f"{label}.activation_transfer.{field} must be {expected!r}")
        normalized_transfer[field] = value
    return {
        "ffn_tp_size": ffn_tp_size,
        "activation_transfer": normalized_transfer,
    }


def _load_production_trace_pd_disagg(
    raw: dict[str, Any],
    simulation_label: str,
    *,
    allowed_instance_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    label = f"{simulation_label}.pd_disagg"
    _require_exact_fields(
        raw,
        {
            "num_prefill_instances", "num_decode_instances",
            "routing_policy", "kv_handoff_mode", "kv_transfer", "router",
        },
        label,
    )
    num_prefill_instances = _require_positive_int_value(
        raw["num_prefill_instances"], f"{label}.num_prefill_instances")
    num_decode_instances = _require_positive_int_value(
        raw["num_decode_instances"], f"{label}.num_decode_instances")
    instance_pair = (num_prefill_instances, num_decode_instances)
    if allowed_instance_pairs is None:
        if instance_pair not in {(1, 2), (2, 1), (2, 2)}:
            raise ManifestValidationError(
                f"{label} currently supports exactly 1P x 2D, 2P x 1D, or "
                "2P x 2D")
    elif instance_pair not in allowed_instance_pairs:
        if allowed_instance_pairs == {(1, 1)}:
            supported_slice = "exactly 1P x 1D"
        elif allowed_instance_pairs == {(1, 1), (1, 2)}:
            supported_slice = "exactly 1P x 1D or 1P x 2D"
        elif allowed_instance_pairs == {(1, 1), (1, 2), (2, 2)}:
            supported_slice = "exactly 1P x 1D, 1P x 2D, or 2P x 2D"
        elif allowed_instance_pairs == {
                (1, 1), (1, 2), (2, 1), (2, 2)}:
            supported_slice = (
                "exactly 1P x 1D, 1P x 2D, 2P x 1D, or 2P x 2D")
        else:
            raise AssertionError(
                "unknown validated Production Trace PD instance set")
        raise ManifestValidationError(
            f"{label} must be {supported_slice} for the validated manifest "
            "slice")
    routing_policy = _require_non_empty_str(raw, "routing_policy", label)
    if routing_policy != "least_queued_tokens":
        raise ManifestValidationError(
            f"{label}.routing_policy must be 'least_queued_tokens'")
    kv_handoff_mode = _require_non_empty_str(raw, "kv_handoff_mode", label)
    if kv_handoff_mode != "metadata_only_private_destination_kv":
        raise ManifestValidationError(
            f"{label}.kv_handoff_mode must be "
            "'metadata_only_private_destination_kv'")
    kv_transfer = _require_dict(raw, "kv_transfer", label)
    _require_exact_fields(
        kv_transfer,
        {"payload", "algorithm", "link", "topology", "backend"},
        f"{label}.kv_transfer",
    )
    expected_transfer = {
        "payload": "kv",
        "algorithm": "send_recv",
        "link": "nvlink",
        "topology": "point_to_point",
        "backend": "analytical_roofline_v1",
    }
    normalized_transfer = {}
    for field, expected in expected_transfer.items():
        value = _require_non_empty_str(
            kv_transfer, field, f"{label}.kv_transfer")
        if value != expected:
            raise ManifestValidationError(
                f"{label}.kv_transfer.{field} must be {expected!r}")
        normalized_transfer[field] = value
    router = _require_dict(raw, "router", label)
    _require_exact_fields(
        router,
        {"resource_id", "dispatch_service_time_s"},
        f"{label}.router",
    )
    router_resource_id = _require_non_empty_str(
        router, "resource_id", f"{label}.router")
    if not router_resource_id.strip():
        raise ManifestValidationError(
            f"{label}.router.resource_id must contain a non-whitespace "
            "character")
    dispatch_service_time_s = router["dispatch_service_time_s"]
    if (not _is_finite_real_number(dispatch_service_time_s)
            or float(dispatch_service_time_s) != 0.0):
        raise ManifestValidationError(
            f"{label}.router.dispatch_service_time_s must be 0.0 for the "
            "validated PD slice")
    return {
        "num_prefill_instances": num_prefill_instances,
        "num_decode_instances": num_decode_instances,
        "routing_policy": routing_policy,
        "kv_handoff_mode": kv_handoff_mode,
        "kv_transfer": normalized_transfer,
        "router": {
            "resource_id": router_resource_id,
            "dispatch_service_time_s": 0.0,
        },
    }


def _load_production_trace_analysis(
    raw: dict[str, Any],
    project_root: Path,
    cell_id: str,
) -> dict[str, Any]:
    label = f"{cell_id}.analysis"
    fields = {"slo_contract", "interactivity_tgs_contract"}
    _require_exact_fields(raw, fields, label)
    return {
        "slo_contract": _load_analysis_contract(
            raw["slo_contract"], project_root, f"{label}.slo_contract",
            load_slo_contract),
        "interactivity_tgs_contract": _load_analysis_contract(
            raw["interactivity_tgs_contract"], project_root,
            f"{label}.interactivity_tgs_contract",
            load_interactivity_tgs_contract),
    }


def _load_analysis_contract(
    value: Any,
    project_root: Path,
    label: str,
    loader,
) -> dict[str, Any]:
    if value is None:
        raise ManifestValidationError(
            f"{label} must be a non-null contract object for a formal "
            "Production Trace experiment")
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{label} must be an object or null")
    _require_exact_fields(value, {"path", "source_sha256"}, label)
    path = _resolve_existing_file(value["path"], project_root, f"{label}.path")
    expected_sha256 = _require_sha256(value, "source_sha256", label)
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ManifestValidationError(
            f"{label}.source_sha256 disagrees with {path}: "
            f"expected={expected_sha256}, actual={actual_sha256}")
    try:
        contract = loader(path)
    except (OSError, TypeError, ValueError) as error:
        raise ManifestValidationError(f"{label} is invalid: {error}") from error
    if contract.source_sha256 != expected_sha256:
        raise AssertionError("validated analysis contract SHA256 changed while loading")
    return {
        "path": path,
        "source_sha256": expected_sha256,
        "contract_id": contract.contract_id,
    }


def _load_case_manifest(
    path: Path,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    with open(path) as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ManifestValidationError("case manifest root must be an object")
    _reject_unknown_fields(raw, {"schema_version", "cases"}, "case manifest")
    schema_version = _require_field(raw, "schema_version", "case manifest")
    if schema_version != 1:
        raise ManifestValidationError("case manifest schema_version must be 1")
    raw_cases = _require_field(raw, "cases", "case manifest")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ManifestValidationError(
            "case manifest.cases must be a non-empty list")
    cases: dict[str, dict[str, Any]] = {}
    for index, raw_case in enumerate(raw_cases):
        case = _load_case(
            raw_case, project_root, f"case manifest.cases[{index}]")
        case_id = case["id"]
        if case_id in cases:
            raise ManifestValidationError(f"duplicate case id: {case_id}")
        cases[case_id] = case
    return cases


def _load_case(
    raw: Any,
    project_root: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"{label} must be an object")
    fields = {
        "id", "model", "model_type", "workload", "sampling",
        "benchmark", "metrics",
    }
    _reject_unknown_fields(raw, fields, label)
    for field in fields:
        _require_field(raw, field, label)
    case_id = _require_non_empty_str(raw, "id", label)
    model_type = _require_non_empty_str(raw, "model_type", label)
    if model_type not in {"dense", "moe"}:
        raise ManifestValidationError(
            f"{case_id}: unsupported model_type: {model_type}")
    return {
        "id": case_id,
        "model": _resolve_project_path(raw["model"], project_root),
        "model_type": model_type,
        "workload": _validate_workload(
            _require_dict(raw, "workload", case_id), case_id),
        "sampling": validate_sampling_config(
            _require_dict(raw, "sampling", case_id), case_id),
        "benchmark": _validate_benchmark(
            _require_dict(raw, "benchmark", case_id), case_id),
        "metrics": _validate_metrics(
            _require_str_list(raw, "metrics", case_id), case_id),
    }


def _load_cell(
    raw: Any,
    cases: dict[str, dict[str, Any]],
    project_root: Path,
) -> ExperimentCell:
    if not isinstance(raw, dict):
        raise ManifestValidationError("cell must be an object")

    required = {
        "id",
        "case_id",
        "deployment",
        "prediction",
        "moe",
        "expert_routing",
    }
    optional = {
        "gpus", "parallelism", "prefill", "decode", "router", "kv_connector",
    }
    _reject_unknown_fields(raw, required | optional, "cell")
    for field in required:
        if field not in raw:
            raise ManifestValidationError(f"missing required field: {field}")

    cell_id = _require_non_empty_str(raw, "id", "manifest cell")
    case_id = _require_non_empty_str(raw, "case_id", "manifest cell")
    if case_id not in cases:
        raise ManifestValidationError(
            f"{cell_id}: unknown Serving case: {case_id}")
    case = cases[case_id]
    deployment = _require_non_empty_str(raw, "deployment", "manifest cell")
    if deployment not in {"colocated", "disaggregated"}:
        raise ManifestValidationError(
            f"{cell_id}: unsupported deployment: {deployment}")

    prediction = _validate_prediction(
        _require_dict(raw, "prediction", cell_id),
        deployment,
        cell_id,
        project_root,
    )
    if deployment == "colocated":
        gpus = _require_gpu_list(raw, "gpus", cell_id)
        parallelism = _normalize_parallelism(
            _require_dict(raw, "parallelism", cell_id),
            f"{cell_id}.parallelism",
            num_requests=case["workload"]["num_requests"],
        )
        _validate_gpu_capacity(cell_id, gpus, parallelism)
        prefill = None
        decode = None
        router = None
        kv_connector = None
    elif deployment == "disaggregated":
        gpus = None
        parallelism = None
        kv_connector = _require_non_empty_str(
            raw, "kv_connector", "manifest cell")
        prefill = _load_instance_config(raw, "prefill", cell_id)
        decode = _load_instance_config(raw, "decode", cell_id)
        router = _load_router_config(raw, cell_id, prefill, decode)
        _validate_disaggregated_layout(cell_id, prefill, decode, router)
    else:
        raise ManifestValidationError(
            f"{cell_id}: unsupported deployment: {deployment}")

    moe = _require_field(raw, "moe", cell_id)
    if moe is not None and not isinstance(moe, dict):
        raise ManifestValidationError(f"{cell_id}: moe must be an object")
    _validate_moe_config(
        cell_id, case["model_type"], parallelism, prefill, decode, moe)
    expert_routing = _validate_expert_routing_config(
        cell_id,
        case["model_type"],
        _require_field(raw, "expert_routing", cell_id),
        project_root,
    )

    return ExperimentCell(
        id=cell_id,
        case_id=case_id,
        model=case["model"],
        deployment=deployment,
        workload=case["workload"],
        sampling=case["sampling"],
        benchmark=case["benchmark"],
        prediction=prediction,
        metrics=case["metrics"],
        gpus=gpus,
        parallelism=parallelism,
        model_type=case["model_type"],
        prefill=prefill,
        decode=decode,
        router=router,
        kv_connector=kv_connector,
        moe=moe,
        expert_routing=expert_routing,
    )


def _hardware_config_path(hardware_id: Any, project_root: Path) -> Path:
    if not isinstance(hardware_id, str) or not hardware_id:
        raise ManifestValidationError(
            "prediction hardware ID must be a non-empty string")
    canonical_id = canonical_hardware_id(hardware_id)
    if canonical_id != hardware_id:
        raise ManifestValidationError(
            f"prediction hardware ID must be canonical: {hardware_id!r}")
    path = (
        project_root / "configs" / "hardware" / f"{canonical_id}.yaml"
    ).resolve(strict=False)
    if not path.is_file():
        raise ManifestValidationError(
            f"prediction hardware config does not exist: {path}")
    return path


def _validate_unique_cells(
    cells: list[ExperimentCell | ProductionTraceSweepCell | SimulationCell],
) -> None:
    seen_ids: set[str] = set()
    for cell in cells:
        if cell.id in seen_ids:
            raise ManifestValidationError(f"duplicate cell id: {cell.id}")
        seen_ids.add(cell.id)


def _validate_workload(workload: dict[str, Any], cell_id: str) -> dict[str, Any]:
    _reject_unknown_fields(
        workload,
        {"name", "num_requests", "prompt_len", "output_len", "seed"},
        f"{cell_id}.workload",
    )
    name = _require_field(workload, "name", f"{cell_id}.workload")
    if not isinstance(name, str) or not name:
        raise ManifestValidationError(f"{cell_id}: workload.name is required")
    for key in ("num_requests", "prompt_len", "output_len"):
        _require_positive_int(workload, key, f"{cell_id}: workload")
    _require_non_negative_int(workload, "seed", f"{cell_id}: workload")
    return dict(workload)


def validate_sampling_config(sampling: dict[str, Any], cell_id: str) -> dict[str, Any]:
    supported = {"min_tokens", "max_tokens", "ignore_eos", "temperature"}
    unsupported = sorted(set(sampling) - supported)
    if unsupported:
        raise ManifestValidationError(
            f"{cell_id}: unsupported sampling fields: {', '.join(unsupported)}")
    max_tokens = _require_positive_int(sampling, "max_tokens", f"{cell_id}: sampling")
    min_tokens = _require_positive_int(sampling, "min_tokens", f"{cell_id}: sampling")
    if min_tokens != max_tokens:
        raise ManifestValidationError(
            f"{cell_id}: sampling min_tokens must match max_tokens")
    if _require_field(sampling, "ignore_eos", f"{cell_id}.sampling") is not True:
        raise ManifestValidationError(f"{cell_id}: sampling ignore_eos must be true")
    temperature = _require_field(sampling, "temperature", f"{cell_id}.sampling")
    if not _is_real_number(temperature) or temperature != 0.0:
        raise ManifestValidationError(
            f"{cell_id}: sampling.temperature must be 0.0 for shared greedy execution")
    sampling = dict(sampling)
    sampling["temperature"] = 0.0
    return dict(sampling)


def _validate_benchmark(benchmark: dict[str, Any], cell_id: str) -> dict[str, Any]:
    _reject_unknown_fields(
        benchmark,
        {"request_rate", "metric_percentiles"},
        f"{cell_id}.benchmark",
    )
    request_rate = _require_field(
        benchmark, "request_rate", f"{cell_id}.benchmark")
    if (not _is_finite_real_number(request_rate)
            or request_rate <= 0):
        raise ManifestValidationError(
            f"{cell_id}: benchmark.request_rate must be finite and positive")
    percentiles = _require_field(
        benchmark, "metric_percentiles", f"{cell_id}.benchmark")
    if not isinstance(percentiles, list) or not percentiles:
        raise ManifestValidationError(
            f"{cell_id}: benchmark.metric_percentiles is required")
    for percentile in percentiles:
        if not _is_real_number(percentile) or percentile not in (50, 90, 99):
            raise ManifestValidationError(
                f"{cell_id}: benchmark.metric_percentiles only supports 50, 90, and 99")
    return dict(benchmark)


def _validate_prediction(
    prediction: dict[str, Any],
    deployment: str,
    cell_id: str,
    project_root: Path,
) -> dict[str, Any]:
    if deployment == "colocated":
        _reject_unknown_fields(
            prediction, {"hardware_id"}, f"{cell_id}.prediction")
        hardware_id = _require_field(
            prediction, "hardware_id", f"{cell_id}.prediction")
        return {
            "hw_config": _hardware_config_path(hardware_id, project_root),
        }
    elif deployment == "disaggregated":
        _reject_unknown_fields(
            prediction,
            {"prefill_hardware_id", "decode_hardware_id"},
            f"{cell_id}.prediction",
        )
        missing = [
            key for key in ("prefill_hardware_id", "decode_hardware_id")
            if key not in prediction
        ]
        if missing:
            raise ManifestValidationError(
                f"{cell_id}: prediction missing {', '.join(missing)}")
        return {
            "prefill_hw_config": _hardware_config_path(
                prediction["prefill_hardware_id"], project_root),
            "decode_hw_config": _hardware_config_path(
                prediction["decode_hardware_id"], project_root),
        }
    else:
        raise ManifestValidationError(
            f"{cell_id}: unsupported deployment: {deployment}")


def _validate_metrics(metrics: list[str], cell_id: str) -> list[str]:
    allowed = {
        "throughput",
        "request_throughput",
        "output_throughput",
        "token_throughput",
        "ttft",
        "tpot",
        "itl",
        "e2el",
    }
    unknown = [metric for metric in metrics if metric not in allowed]
    if unknown:
        raise ManifestValidationError(
            f"{cell_id}: unsupported metrics: {', '.join(unknown)}")
    required_families = [
        {"throughput", "request_throughput", "output_throughput", "token_throughput"},
        {"ttft"},
        {"tpot", "itl"},
    ]
    for family in required_families:
        if not any(metric in family for metric in metrics):
            raise ManifestValidationError(
                f"{cell_id}: metrics must include throughput, ttft, and tpot/itl")
    return list(metrics)


def _load_instance_config(
    raw: dict[str, Any],
    key: str,
    cell_id: str,
) -> dict[str, Any]:
    if key not in raw or not isinstance(raw[key], dict):
        raise ManifestValidationError(f"{cell_id}: {key} config is required")
    cfg = dict(raw[key])
    _reject_unknown_fields(
        cfg,
        {
            "gpus", "engine_args", "tp_size", "kv_port", "http_port",
            "trace_output", "kv_ip", "kv_buffer_size",
        },
        f"{cell_id}.{key}",
    )
    for field in (
        "gpus", "engine_args", "tp_size", "kv_port", "http_port",
        "trace_output", "kv_ip", "kv_buffer_size",
    ):
        _require_field(cfg, field, f"{cell_id}.{key}")
    gpus = _require_gpu_list(cfg, "gpus", f"{cell_id}.{key}")
    engine_args = cfg["engine_args"]
    if not isinstance(engine_args, dict):
        raise ManifestValidationError(f"{cell_id}.{key}: engine_args must be an object")
    cfg["engine_args"] = dict(engine_args)
    cfg["gpus"] = gpus
    cfg["effective_parallelism"] = _effective_instance_parallelism(cfg, engine_args, cell_id, key)
    cfg["tp_size"] = cfg["effective_parallelism"]["tensor_parallel_size"]
    cfg["kv_port"] = _require_port(cfg, "kv_port", f"{cell_id}.{key}")
    if cfg["http_port"] is not None:
        cfg["http_port"] = _require_port(cfg, "http_port", f"{cell_id}.{key}")
    trace_output = cfg["trace_output"]
    if not isinstance(trace_output, str) or not trace_output:
        raise ManifestValidationError(
            f"{cell_id}.{key}: trace_output must be a non-empty relative path")
    trace_path = Path(trace_output)
    if trace_path.is_absolute() or ".." in trace_path.parts:
        raise ManifestValidationError(
            f"{cell_id}.{key}: trace_output must stay inside the runtime cell directory")
    cfg["trace_output"] = trace_path
    if cfg["kv_ip"] is not None and (
        not isinstance(cfg["kv_ip"], str) or not cfg["kv_ip"]
    ):
        raise ManifestValidationError(
            f"{cell_id}.{key}: kv_ip must be null or a non-empty string")
    if cfg["kv_buffer_size"] is not None:
        _require_positive_int_value(
            cfg["kv_buffer_size"], f"{cell_id}.{key}: kv_buffer_size")
    _validate_gpu_capacity(f"{cell_id}.{key}", gpus, cfg["effective_parallelism"])
    return cfg


def _load_router_config(
    raw: dict[str, Any],
    cell_id: str,
    prefill: dict[str, Any],
    decode: dict[str, Any],
) -> dict[str, Any] | None:
    router_value = _require_field(raw, "router", cell_id)
    real_run_shape = (
        prefill["http_port"] is not None or decode["http_port"] is not None
    )
    if router_value is None:
        if real_run_shape:
            raise ManifestValidationError(
                f"{cell_id}: router config is required when prefill/decode http_port is set")
        return None
    router = _require_dict(raw, "router", cell_id)
    host = _require_field(router, "host", f"{cell_id}.router")
    if not isinstance(host, str) or not host:
        raise ManifestValidationError(f"{cell_id}: router.host must be a non-empty string")
    router["host"] = host
    router["http_port"] = _require_port(router, "http_port", f"{cell_id}.router")
    router["proxy_port"] = _require_port(router, "proxy_port", f"{cell_id}.router")
    return router


def _effective_instance_parallelism(
    cfg: dict[str, Any],
    engine_args: dict[str, Any],
    cell_id: str,
    role: str,
) -> dict[str, int]:
    explicit_tp = _require_positive_int_value(
        cfg["tp_size"], f"{cell_id}.{role}: tp_size")
    engine_tp = _require_positive_int_value(
        _require_field(
            engine_args, "tensor_parallel_size",
            f"{cell_id}.{role}.engine_args",
        ),
        f"{cell_id}.{role}: engine_args.tensor_parallel_size",
    )
    if explicit_tp != engine_tp:
        raise ManifestValidationError(
            f"{cell_id}.{role}: tp_size conflicts with engine_args.tensor_parallel_size")

    return {
        "tensor_parallel_size": explicit_tp,
        "pipeline_parallel_size": _require_positive_int_value(
            _require_field(
                engine_args, "pipeline_parallel_size",
                f"{cell_id}.{role}.engine_args",
            ),
            f"{cell_id}.{role}: engine_args.pipeline_parallel_size",
        ),
        "data_parallel_size": _require_positive_int_value(
            _require_field(
                engine_args, "data_parallel_size",
                f"{cell_id}.{role}.engine_args",
            ),
            f"{cell_id}.{role}: engine_args.data_parallel_size",
        ),
    }


def _validate_disaggregated_layout(
    cell_id: str,
    prefill: dict[str, Any],
    decode: dict[str, Any],
    router: dict[str, Any] | None = None,
) -> None:
    overlap = set(prefill["gpus"]) & set(decode["gpus"])
    if overlap:
        raise ManifestValidationError(
            f"{cell_id}: prefill/decode GPU sets overlap: {sorted(overlap)}")

    prefill_ports = _port_range(prefill["kv_port"], _worker_count(prefill))
    decode_ports = _port_range(decode["kv_port"], _worker_count(decode))
    port_overlap = prefill_ports & decode_ports
    if port_overlap:
        raise ManifestValidationError(
            f"{cell_id}: prefill/decode port range overlap: {sorted(port_overlap)}")
    if router is not None:
        occupied = set(prefill_ports) | set(decode_ports)
        for instance in (prefill, decode):
            if instance["http_port"] is not None:
                occupied.add(instance["http_port"])
        router_ports = {router["http_port"], router["proxy_port"]}
        if len(router_ports) != 2:
            raise ManifestValidationError(
                f"{cell_id}: router http_port and proxy_port must not overlap")
        overlap = occupied & router_ports
        if overlap:
            raise ManifestValidationError(
                f"{cell_id}: router ports overlap with instance ports: {sorted(overlap)}")


def _validate_moe_config(
    cell_id: str,
    model_type: str,
    parallelism: dict[str, Any] | None,
    prefill: dict[str, Any] | None,
    decode: dict[str, Any] | None,
    moe: dict[str, Any] | None,
) -> None:
    enable_ep = any(
        _expert_parallel_enabled(config, label)
        for config, label in (
            (parallelism, f"{cell_id}.parallelism"),
            (None if prefill is None else prefill["engine_args"],
             f"{cell_id}.prefill.engine_args"),
            (None if decode is None else decode["engine_args"],
             f"{cell_id}.decode.engine_args"),
        )
    )
    if model_type == "dense":
        if enable_ep:
            raise ManifestValidationError(
                f"{cell_id}: dense model cannot enable expert parallel")
        if moe is not None:
            raise ManifestValidationError(
                f"{cell_id}: dense model requires moe=null")
    elif model_type == "moe":
        if moe is None:
            raise ManifestValidationError(
                f"{cell_id}: moe model requires an explicit moe object")
        _reject_unknown_fields(
            moe, {"all2all_backend", "eplb_enabled"}, f"{cell_id}.moe")
        if set(moe) != {"all2all_backend", "eplb_enabled"}:
            raise ManifestValidationError(
                f"{cell_id}: moe requires all2all_backend and eplb_enabled")
        backend = moe["all2all_backend"]
        if not isinstance(backend, str) or not backend:
            raise ManifestValidationError(
                f"{cell_id}: moe.all2all_backend must be a non-empty string")
        if not isinstance(moe["eplb_enabled"], bool):
            raise ManifestValidationError(
                f"{cell_id}: moe.eplb_enabled must be a boolean")
    else:
        raise ManifestValidationError(
            f"{cell_id}: unsupported model_type: {model_type}")
    if model_type == "moe" and not enable_ep:
        raise ManifestValidationError(
            f"{cell_id}: moe model must enable expert parallel execution")


def _validate_expert_routing_config(
    cell_id: str,
    model_type: str,
    value: Any,
    project_root: Path,
) -> dict[str, Any] | None:
    if model_type == "dense":
        if value is not None:
            raise ManifestValidationError(
                f"{cell_id}: dense model requires expert_routing=null")
        return None
    if model_type != "moe":
        raise ManifestValidationError(
            f"{cell_id}: unsupported model_type: {model_type}")
    if not isinstance(value, dict):
        raise ManifestValidationError(
            f"{cell_id}: MoE model requires an expert_routing object")
    source_kind = _require_non_empty_str(
        value, "source_kind", f"{cell_id}.expert_routing")
    if source_kind == "explicit_assumption":
        uniform_fields = {"source_kind", "source_id", "root_seed"}
        skew_fields = uniform_fields | {
            "hot_expert_count", "hot_expert_weight"}
        _reject_unknown_fields(
            value, skew_fields, f"{cell_id}.expert_routing")
        if set(value) not in (uniform_fields, skew_fields):
            raise ManifestValidationError(
                f"{cell_id}.expert_routing explicit assumption requires "
                "source_kind, source_id, and root_seed, with both or neither "
                "of hot_expert_count and hot_expert_weight")
        source_id = _require_non_empty_str(
            value, "source_id", f"{cell_id}.expert_routing")
        root_seed = value["root_seed"]
        if (isinstance(root_seed, bool) or not isinstance(root_seed, int)
                or root_seed < 0):
            raise ManifestValidationError(
                f"{cell_id}.expert_routing.root_seed must be a "
                "non-negative integer")
        normalized = {
            "source_kind": source_kind,
            "source_id": source_id,
            "root_seed": root_seed,
        }
        if set(value) == skew_fields:
            hot_expert_count = value["hot_expert_count"]
            hot_expert_weight = value["hot_expert_weight"]
            if (isinstance(hot_expert_count, bool)
                    or not isinstance(hot_expert_count, int)
                    or hot_expert_count < 1):
                raise ManifestValidationError(
                    f"{cell_id}.expert_routing.hot_expert_count must be a "
                    "positive integer")
            if (isinstance(hot_expert_weight, bool)
                    or not isinstance(hot_expert_weight, int)
                    or hot_expert_weight <= 1):
                raise ManifestValidationError(
                    f"{cell_id}.expert_routing.hot_expert_weight must be an "
                    "integer greater than one")
            normalized["hot_expert_count"] = hot_expert_count
            normalized["hot_expert_weight"] = hot_expert_weight
        return normalized
    if source_kind == "replay_observation":
        expected = {"source_kind", "path"}
        _reject_unknown_fields(value, expected, f"{cell_id}.expert_routing")
        if set(value) != expected:
            raise ManifestValidationError(
                f"{cell_id}.expert_routing replay requires source_kind and "
                "path")
        path = _resolve_project_path(value["path"], project_root)
        try:
            load_expert_routing_replay(path)
        except (OSError, TypeError, ValueError) as error:
            raise ManifestValidationError(
                f"{cell_id}.expert_routing replay is invalid: {error}"
            ) from error
        return {"source_kind": source_kind, "path": str(path)}
    if source_kind == "profile_observation":
        expected = {"source_kind", "source_id", "manifest_path"}
        _reject_unknown_fields(value, expected, f"{cell_id}.expert_routing")
        if set(value) != expected:
            raise ManifestValidationError(
                f"{cell_id}.expert_routing profile requires source_kind, "
                "source_id, and manifest_path")
        source_id = _require_non_empty_str(
            value, "source_id", f"{cell_id}.expert_routing")
        manifest_path = _resolve_project_path(
            value["manifest_path"], project_root)
        if not manifest_path.is_file():
            raise ManifestValidationError(
                f"{cell_id}.expert_routing profile manifest does not exist: "
                f"{manifest_path}")
        return {
            "source_kind": source_kind,
            "source_id": source_id,
            "manifest_path": str(manifest_path),
        }
    raise ManifestValidationError(
        f"{cell_id}.expert_routing has unknown source_kind: "
        f"{source_kind!r}")


def _validate_gpu_capacity(
    cell_id: str,
    gpus: list[int],
    parallelism: dict[str, Any],
) -> None:
    try:
        _validate_gpu_capacity_util(cell_id, gpus, parallelism)
    except ParallelConfigError as e:
        raise ManifestValidationError(str(e)) from e


def _parallel_worker_count(parallelism: dict[str, Any]) -> int:
    return _parallel_worker_count_util(parallelism)


def _worker_count(instance_config: dict[str, Any]) -> int:
    return _parallel_worker_count(instance_config["effective_parallelism"])


def _port_range(start: int, count: int) -> set[int]:
    return set(range(start, start + count))


def _resolve_project_path(value: Any, project_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestValidationError("path value must be a non-empty string")
    models_prefix = "${MODELS_ROOT}/"
    if value.startswith(models_prefix):
        if "MODELS_ROOT" not in os.environ:
            raise ManifestValidationError(
                "MODELS_ROOT is required to resolve experiment model paths")
        models_root = os.environ["MODELS_ROOT"]
        if not models_root:
            raise ManifestValidationError("MODELS_ROOT must not be empty")
        return (Path(models_root) / value[len(models_prefix):]).resolve(
            strict=False)
    if "${" in value:
        raise ManifestValidationError(
            f"unsupported environment placeholder in path: {value!r}")
    path = Path(value)
    if path.is_absolute():
        return path.resolve(strict=False)
    return (project_root / path).resolve(strict=False)


def _resolve_existing_file(
    value: Any,
    project_root: Path,
    label: str,
) -> Path:
    path = _resolve_project_path(value, project_root)
    if not path.is_file():
        raise ManifestValidationError(f"{label} is not a file: {path}")
    return path


def _file_sha256(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ManifestValidationError(
            f"cannot read immutable experiment input: {path}") from error
    return hashlib.sha256(payload).hexdigest()


def _require_non_empty_str(
    raw: dict[str, Any], key: str, label: str,
) -> str:
    value = _require_field(raw, key, label)
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(
            f"{label}.{key} must be a non-empty string")
    return value


def _expert_parallel_enabled(config: dict[str, Any] | None, label: str) -> bool:
    if config is None:
        return False
    value = _require_field(config, "enable_expert_parallel", label)
    if not isinstance(value, bool):
        raise ManifestValidationError(
            f"{label}.enable_expert_parallel must be a boolean")
    return value


def _require_dict(raw: dict[str, Any], key: str, cell_id: str) -> dict[str, Any]:
    value = _require_field(raw, key, cell_id)
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{cell_id}: {key} must be an object")
    return dict(value)


def _require_str_list(raw: dict[str, Any], key: str, cell_id: str) -> list[str]:
    value = _require_field(raw, key, cell_id)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise ManifestValidationError(f"{cell_id}: {key} must be a non-empty string list")
    return list(value)


def _require_gpu_list(raw: dict[str, Any], key: str, cell_id: str) -> list[int]:
    value = _require_field(raw, key, cell_id)
    if not isinstance(value, list) or not value:
        raise ManifestValidationError(f"{cell_id}: {key} must be a non-empty list")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value):
        raise ManifestValidationError(f"{cell_id}: {key} must contain non-negative integers")
    if len(set(value)) != len(value):
        raise ManifestValidationError(f"{cell_id}: {key} contains duplicate GPU IDs")
    return list(value)


def _require_positive_int(raw: dict[str, Any], key: str, label: str) -> int:
    if key not in raw:
        raise ManifestValidationError(f"{label}.{key} is required")
    return _require_positive_int_value(raw[key], f"{label}.{key}")


def _require_non_negative_int(raw: dict[str, Any], key: str, label: str) -> int:
    value = _require_field(raw, key, label)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManifestValidationError(f"{label}.{key} must be a non-negative integer")
    return value


def _require_positive_int_value(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ManifestValidationError(f"{label} must be an integer >= 1")
    return value


def _require_sha256(
    raw: dict[str, Any],
    key: str,
    label: str,
) -> str:
    value = _require_field(raw, key, label)
    if (not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ManifestValidationError(
            f"{label}.{key} must be a lowercase SHA256")
    return value


def _require_port(raw: dict[str, Any], key: str, label: str) -> int:
    value = _require_field(raw, key, label)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > 65535:
        raise ManifestValidationError(
            f"{label}.{key} must be an integer TCP port in [1, 65535]")
    return value


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_real_number(value: Any) -> bool:
    if not _is_real_number(value):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _require_field(raw: dict[str, Any], key: str, label: str) -> Any:
    if key not in raw:
        raise ManifestValidationError(f"{label}.{key} is required")
    return raw[key]


def _normalize_parallelism(
    raw: dict[str, Any], label: str, *, num_requests: int,
) -> dict[str, Any]:
    base_fields = {
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "enable_expert_parallel",
    }
    scheduler_fields = {
        *VLLM_CONFORMANCE_SCHEDULER_DEFAULTS,
        "enable_chunked_prefill",
        "enable_prefix_caching",
    }
    _reject_unknown_fields(raw, base_fields | scheduler_fields, label)
    normalized = dict(raw)
    for key in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
    ):
        normalized[key] = _require_positive_int_value(
            _require_field(normalized, key, label), f"{label}.{key}")
    value = _require_field(normalized, "enable_expert_parallel", label)
    if not isinstance(value, bool):
        raise ManifestValidationError(
            f"{label}.enable_expert_parallel must be a boolean")
    if num_requests > 1:
        missing = sorted(
            set(VLLM_CONFORMANCE_SCHEDULER_DEFAULTS) - set(normalized))
        if missing:
            raise ManifestValidationError(
                f"{label} must explicitly define scheduler capacity for a "
                "multi-request workload: " + ", ".join(missing))
    for key, default in VLLM_CONFORMANCE_SCHEDULER_DEFAULTS.items():
        if key in normalized:
            normalized[key] = _require_positive_int_value(
                normalized[key], f"{label}.{key}")
        else:
            normalized[key] = default
    for key, default in (
        ("enable_chunked_prefill", True),
        ("enable_prefix_caching", False),
    ):
        if key in normalized:
            if not isinstance(normalized[key], bool):
                raise ManifestValidationError(
                    f"{label}.{key} must be a boolean")
        else:
            normalized[key] = default
    return normalized


def _reject_unknown_fields(
    raw: dict[str, Any], allowed: set[str], label: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ManifestValidationError(
            f"{label} contains unsupported fields: {', '.join(unknown)}")


def _require_exact_fields(
    raw: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    supplied = set(raw)
    if supplied != expected:
        raise ManifestValidationError(
            f"{label} fields disagree with schema: "
            f"missing={sorted(expected - supplied)}, "
            f"unexpected={sorted(supplied - expected)}")


def _unique_manifest_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ManifestValidationError(
                f"duplicate experiment manifest JSON field: {key!r}")
        payload[key] = value
    return payload


def _reject_manifest_json_constant(value: str) -> object:
    raise ManifestValidationError(
        f"non-standard experiment manifest JSON constant: {value!r}")
