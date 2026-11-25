"""Compact ShareGPT suite and case configuration contract: suite configs,
cells, canonical model/hardware identity, and case binding validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Protocol, Sequence

from evaluation.serving.experiments.experiment_manifest import (
    RUNTIME_MEMORY_POLICY_PATH_TEMPLATE,
)
from src.serving.backend.vllm_scheduler_process import (
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
    VLLM030_SOURCE_COMMIT,
)
from src.kernel.predictors.protobridge.adapter import (
    development_profile_config_sha256,
)
from src.serving.workload.graph_padding import GraphExecutionConfig
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS

from evaluation.json_shape import require_object_value as _mapping

from evaluation.serving.metrics.compact_shape_validators import (
    _unique_json_object,
    _confined_path,
    _relative_path,
    _exact_fields,
    _sequence,
    _string,
    _sha256,
    _integer,
)


_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")

_SUITE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CASE_ID = re.compile(
    r"tp(?P<tp>[1-9][0-9]*)(?:_ep(?P<ep>[1-9][0-9]*))?"
    r"_qps(?P<qps>[1-9][0-9]*)"
)


_MODEL_FIELDS = {
    "dataset_model_id",
    "simulation_model_path",
    "model_config_sha256",
}
_HARDWARE_FIELDS = {
    "hardware_id",
    "hardware_config_path",
    "hardware_config_sha256",
}
_GRAPH_HARDWARE_FIELDS = _HARDWARE_FIELDS | {"graph_execution"}
_GRAPH_EXECUTION_FIELDS = {
    "configured_mode",
    "uniform_decode_query_len",
    "capture_sizes",
    "piecewise_eager_workload_kinds",
}
_PREDICTION_FIELDS = {
    "semantics",
    "backend",
    "scheduler_version",
    "scheduler_source_commit",
    "deployment",
    "compute_dtype",
    "prediction_method",
    "tensor_collective_backend",
    "tensor_all_reduce_profile_sha256",
    "tensor_all_gather_profile_sha256",
    "launch_policy",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "expert_parallel_size",
    "max_model_len",
    "max_batched_tokens",
    "max_requests",
    "runtime_memory_policy_path",
    "kv_block_size_tokens",
    "attention_kv_layout",
    "attention_kv_distribution",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "router_resource_id",
    "router_dispatch_service_time_s",
    "mtp_num_speculative_tokens",
    "mtp_acceptance_policy",
    "mtp_fixed_accepted_tokens",
    "mtp_acceptance_source_id",
}
_CONFIG_FIELDS = {
    "schema_version",
    "suite_id",
    "dataset_root",
    "models",
    "hardware",
    "qps_values",
    "cells",
    "prediction_contract",
}
_CELL_FIELDS = {
    "hardware_id",
    "model_id",
    "case_id",
    "manifest_sha256",
    "requests_sha256",
}


_CANONICAL_MODEL_SPECS = {
    "qwen3-4b": (
        "configs/model/qwen3-4b.json",
        "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    ),
    "qwen3-8b": (
        "configs/model/qwen3-8b.json",
        "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    ),
    "llama-3.1-8b": (
        "configs/model/llama-3.1-8b-instruct.json",
        "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    ),
    "qwen3-30b-a3b": (
        "configs/model/qwen3-30b-a3b.json",
        "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297",
    ),
    "qwen3-32b": (
        "configs/model/qwen3-32b.json",
        "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb",
    ),
    "llama-3-70b": (
        "${MODELS_ROOT}/Meta-Llama-3-70B-Instruct",
        "bfd937677ee1c3d5b535ccaeec3223c06eca9452f3ac0dabe2ff87deb0f818a4",
    ),
    "llama-3.1-70b": (
        "${MODELS_ROOT}/Llama-3.1-70B-Instruct",
        "fa6e9124e4621df77aecf96fbfaf7975814013d2d5ab1c972e965000588a9749",
    ),
}
_MODEL_CONFIG_ALTERNATES = {
    "qwen3-32b": (
        (
            "configs/model/qwen3-32b-fp16.json",
            "8150eeaa087d37397bde1f2f128a6d120916e078bc51b1c7e95fc5a6b9905316",
        ),
    ),
}
_LEGACY_MODEL_IDS = {
    "Qwen3-4B": "qwen3-4b",
    "Qwen3-8B": "qwen3-8b",
    "Llama-3.1-8B": "llama-3.1-8b",
    "Qwen3-30B-A3B": "qwen3-30b-a3b",
    "Qwen3-32B": "qwen3-32b",
}
_EXPECTED_MODELS = {
    **_CANONICAL_MODEL_SPECS,
    **{
        legacy: _CANONICAL_MODEL_SPECS[canonical]
        for legacy, canonical in _LEGACY_MODEL_IDS.items()
    },
}
_EXPECTED_HARDWARE = {
    "nvidia_a800": (
        "configs/hardware/nvidia_a800.yaml",
        "4fd8c3cbed54268219dc8dac76ed51d747f53b5d0c7d0fdb2033454979f2bf0c",
    ),
    "metax_c500": (
        "configs/hardware/metax_c500.yaml",
        "3eff3be984ab4c42acce85efb053155760685a95b0b08e6544c479cf5ff8b929",
    ),
    "ascend_910b": (
        "configs/hardware/ascend_910b.yaml",
        "e0fe91381d31955cecc29c97484c959f5a4bf8ce941de4f38c47c0c42b48eff0",
    ),
    "nvidia_h200": (
        "configs/hardware/nvidia_h200.yaml",
        "1b67bb9e4c555976a9d96386967bf50ca36bf11287ad33aedff5adca8ebf406a",
    ),
}
# The frozen 45-cell official matrix and the phase-1 suite cover these three
# hardware; H200 cells join a run only through an explicit vLLM version below.
_OFFICIAL_HARDWARE = EVALUATED_HARDWARE_IDS
_EXPECTED_PREDICTION = {
    "semantics": "current_pinned_vllm_not_capture_replay",
    "backend": "vllm",
    "scheduler_version": VLLM_SCHEDULER_VERSION,
    "scheduler_source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
    "deployment": "colocated",
    "compute_dtype": "bf16",
    "prediction_method": "midlayer",
    "tensor_collective_backend": "protobridge",
    "tensor_all_reduce_profile_sha256": None,
    "tensor_all_gather_profile_sha256": None,
    "launch_policy": "injected_launch",
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "expert_parallel_size": 1,
    "max_model_len": 8192,
    "max_batched_tokens": 8192,
    "max_requests": 256,
    "runtime_memory_policy_path": RUNTIME_MEMORY_POLICY_PATH_TEMPLATE,
    "kv_block_size_tokens": 16,
    "attention_kv_layout": "paged_varlen",
    "attention_kv_distribution": "uniform",
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "router_resource_id": "router_cpu",
    "router_dispatch_service_time_s": 0.0,
    "mtp_num_speculative_tokens": 0,
    "mtp_acceptance_policy": "disabled",
    "mtp_fixed_accepted_tokens": 0,
    "mtp_acceptance_source_id": "disabled",
}
OFFICIAL_METRICS_SUITE_SELECTIONS = (
    (
        "configs/evaluation/experiments/sharegpt_phase1.json",
        ("nvidia_a800", "metax_c500"),
    ),
    (
        "configs/evaluation/experiments/ascend_prefix_cache_phase1_suite.json",
        ("ascend_910b",),
    ),
    ("configs/evaluation/experiments/nvidia_a800_qwen3_30b_a3b_suite.json", None),
    ("configs/evaluation/experiments/metax_c500_qwen3_30b_a3b_suite.json", None),
    ("configs/evaluation/experiments/ascend_qwen3_30b_a3b_suite.json", None),
    ("configs/evaluation/experiments/qwen3_32b_qps1_suite.json", None),
    ("configs/evaluation/experiments/qwen3_32b_qps4_qps16_suite.json", None),
    ("configs/evaluation/experiments/metax_c500_qwen3_32b_suite.json", None),
    ("configs/evaluation/experiments/ascend_qwen3_32b_suite.json", None),
)
# H200 ShareGPT suites per captured vLLM version.  Their cells pin candidate
# datasets outside serving_dataset/, so they are evaluated only when the
# caller selects one version and supplies its candidate dataset root.
NVIDIA_H200_SUITE_SELECTIONS = {
    version: tuple(
        f"configs/evaluation/experiments/nvidia_h200_{tag}_{group}_suite.json"
        for group in ("sharegpt_tp1", "qwen3_30b_a3b", "qwen3_32b")
    )
    for version, tag in (("0.30.0", "vllm030"), ("0.11.0", "vllm011"))
}


class ValidatedCompactCase(Protocol):
    case_dir: Path
    manifest: Mapping[str, object]
    scheduler_pipeline_depth: int
    requests: Sequence[object]
    metrics: Mapping[str, object] | None
    scheduler_steps: Sequence[Mapping[str, object]] | None
    alignment_steps: Mapping[str, object | None]
    operator_signatures: Mapping[str, Mapping[str, object]]
    launch_signatures: Mapping[str, Mapping[str, object]]


def scheduler_pipeline_depth_from_validated_case(
    case: ValidatedCompactCase,
) -> int:
    """Consume Scheduler capacity normalized by the dataset boundary."""

    return case.scheduler_pipeline_depth


def scheduler_async_scheduling_from_validated_case(
    case: ValidatedCompactCase,
) -> bool:
    """Resolve vLLM async scheduling from the observed executor capacity."""

    manifest = _mapping(case.manifest, "compact manifest")
    if manifest["schema_version"] not in (5, 7):
        raise ValueError(
            "runtime Scheduler evidence is required to resolve async mode"
        )
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    depth = scheduler_pipeline_depth_from_validated_case(case)
    pipeline_parallel_size = runtime["pipeline_parallel_size"]
    if (isinstance(pipeline_parallel_size, bool)
            or not isinstance(pipeline_parallel_size, int)
            or pipeline_parallel_size < 1):
        raise ValueError(
            "runtime.pipeline_parallel_size must be a positive non-bool integer"
        )
    evidence = _mapping(
        runtime["scheduler_pipeline_depth_evidence"],
        "runtime scheduler_pipeline_depth_evidence",
    )
    if evidence["pipeline_parallel_size"] != pipeline_parallel_size:
        raise ValueError(
            "Scheduler depth evidence pipeline size disagrees with runtime"
        )
    if pipeline_parallel_size != 1:
        raise NotImplementedError(
            "compact runtime must explicitly record async scheduling when "
            "pipeline_parallel_size exceeds one"
        )
    if depth not in (1, 2):
        raise ValueError(
            "observed PP=1 vLLM executor capacity must be one or two"
        )
    return depth == 2


@dataclass(frozen=True)
class ModelEntry:
    dataset_model_id: str
    simulation_model_path: str
    model_config_sha256: str


@dataclass(frozen=True)
class HardwareEntry:
    hardware_id: str
    hardware_config_path: str
    hardware_config_sha256: str
    graph_execution: GraphExecutionConfig


@dataclass(frozen=True)
class PredictionContract:
    values: Mapping[str, object]


@dataclass(frozen=True)
class Phase1Config:
    source_path: Path
    source_sha256: str
    suite_id: str
    dataset_root: str
    models: tuple[ModelEntry, ...]
    hardware: tuple[HardwareEntry, ...]
    qps_values: tuple[int, ...]
    cell_order: tuple[str, ...]
    cell_hashes: Mapping[str, tuple[str, str]]
    prediction: PredictionContract


@dataclass(frozen=True)
class Phase1Cell:
    cell_id: str
    model: ModelEntry
    hardware: HardwareEntry
    qps: int
    case_id: str
    manifest_sha256: str
    requests_sha256: str


def load_phase1_config(path: str | Path) -> Phase1Config:
    """Load a closed compact-suite contract, including the phase-1 suite."""

    source_path = Path(path)
    source_bytes = source_path.read_bytes()
    try:
        payload = json.loads(source_bytes, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid phase-1 JSON config: {error}") from error
    root = _mapping(payload, "phase-1 config")
    _exact_fields(root, _CONFIG_FIELDS, "phase-1 config")
    if (type(root["schema_version"]) is not int
            or root["schema_version"] not in (2, 4)):
        raise ValueError("phase-1 config schema_version must equal 2 or 4")
    schema_version = root["schema_version"]
    suite_id = _string(root["suite_id"], "suite_id")
    if _SUITE_ID.fullmatch(suite_id) is None:
        raise ValueError("suite_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if root["dataset_root"] != "serving_dataset/sharegpt":
        raise ValueError(
            "phase-1 dataset_root must equal 'serving_dataset/sharegpt'"
        )

    raw_models = _sequence(root["models"], "phase-1 models")
    models = []
    for index, raw_model in enumerate(raw_models):
        model = _mapping(raw_model, f"phase-1 model {index}")
        _exact_fields(model, _MODEL_FIELDS, f"phase-1 model {index}")
        model_id = _string(model["dataset_model_id"], "dataset_model_id")
        path_value = _string(
            model["simulation_model_path"], "simulation_model_path"
        )
        digest = _sha256(model["model_config_sha256"], "model config SHA256")
        if model_id not in _EXPECTED_MODELS:
            raise ValueError(f"unknown compact-suite model: {model_id!r}")
        alternates = ()
        if model_id in _MODEL_CONFIG_ALTERNATES:
            alternates = _MODEL_CONFIG_ALTERNATES[model_id]
        expected_model_specs = (_EXPECTED_MODELS[model_id], *alternates)
        if (path_value, digest) not in expected_model_specs:
            raise ValueError(
                f"phase-1 model mapping disagrees with frozen contract: {model_id}"
            )
        models.append(ModelEntry(model_id, path_value, digest))
    if not models:
        raise ValueError("compact suite must contain at least one model")
    if len(models) != len({model.dataset_model_id for model in models}):
        raise ValueError("compact-suite models contain duplicates")

    raw_hardware = _sequence(root["hardware"], "phase-1 hardware")
    hardware_entries = []
    for index, raw_entry in enumerate(raw_hardware):
        entry = _mapping(raw_entry, f"phase-1 hardware {index}")
        expected_hardware_fields = (
            _HARDWARE_FIELDS if schema_version == 2 else
            _GRAPH_HARDWARE_FIELDS)
        _exact_fields(
            entry, expected_hardware_fields, f"phase-1 hardware {index}")
        hardware_id = _string(entry["hardware_id"], "hardware_id")
        config_path = _relative_path(
            entry["hardware_config_path"], "hardware_config_path"
        )
        digest = _sha256(
            entry["hardware_config_sha256"], "hardware config SHA256"
        )
        if hardware_id not in _EXPECTED_HARDWARE:
            raise ValueError(f"unknown phase-1 hardware: {hardware_id!r}")
        if config_path != _EXPECTED_HARDWARE[hardware_id][0]:
            raise ValueError(
                "phase-1 hardware mapping disagrees with frozen contract: "
                f"{hardware_id}"
            )
        if schema_version == 2:
            graph_execution = GraphExecutionConfig.disabled()
        else:
            graph = _mapping(
                entry["graph_execution"],
                f"phase-1 hardware {index} graph_execution",
            )
            _exact_fields(
                graph,
                _GRAPH_EXECUTION_FIELDS,
                f"phase-1 hardware {index} graph_execution",
            )
            graph_execution = GraphExecutionConfig(
                _string(graph["configured_mode"], "configured graph mode"),
                tuple(
                    _integer(value, "graph capture size")
                    for value in _sequence(
                        graph["capture_sizes"], "graph capture sizes")
                ),
                tuple(
                    _string(value, "piecewise eager workload kind")
                    for value in _sequence(
                        graph["piecewise_eager_workload_kinds"],
                        "piecewise eager workload kinds",
                    )
                ),
                _integer(
                    graph["uniform_decode_query_len"],
                    "uniform decode query length",
                ),
            )
        hardware_entries.append(HardwareEntry(
            hardware_id,
            config_path,
            digest,
            graph_execution,
        ))
    if not hardware_entries:
        raise ValueError("compact suite must contain at least one hardware entry")
    if len(hardware_entries) != len(
        {entry.hardware_id for entry in hardware_entries}
    ):
        raise ValueError("compact-suite hardware entries contain duplicates")

    qps_values = _sequence(root["qps_values"], "phase-1 qps_values")
    if any(type(value) is not int for value in qps_values):
        raise TypeError("phase-1 qps_values must contain exact integers")
    if not qps_values or any(value <= 0 for value in qps_values):
        raise ValueError("compact-suite qps_values must be positive")
    if len(qps_values) != len(set(qps_values)):
        raise ValueError("compact-suite qps_values contain duplicates")

    raw_cells = _sequence(root["cells"], "phase-1 cells")
    cell_hashes = {}
    cell_order = []
    for index, raw_cell in enumerate(raw_cells):
        cell = _mapping(raw_cell, f"phase-1 cell {index}")
        _exact_fields(cell, _CELL_FIELDS, f"phase-1 cell {index}")
        hardware_id = _string(cell["hardware_id"], "cell hardware_id")
        model_id = _string(cell["model_id"], "cell model_id")
        case_id = _string(cell["case_id"], "cell case_id")
        if hardware_id not in {entry.hardware_id for entry in hardware_entries}:
            raise ValueError(f"cell uses undeclared hardware: {hardware_id!r}")
        if model_id not in {entry.dataset_model_id for entry in models}:
            raise ValueError(f"cell uses undeclared model: {model_id!r}")
        case_match = _CASE_ID.fullmatch(case_id)
        if case_match is None:
            raise ValueError(f"cell case_id is not canonical: {case_id!r}")
        if int(case_match.group("qps")) not in qps_values:
            raise ValueError(f"cell case_id QPS is absent from qps_values: {case_id!r}")
        cell_id = f"{hardware_id}/{model_id}/{case_id}"
        if cell_id in cell_hashes:
            raise ValueError(f"duplicate phase-1 cell: {cell_id}")
        cell_order.append(cell_id)
        cell_hashes[cell_id] = (
            _sha256(cell["manifest_sha256"], "cell manifest SHA256"),
            _sha256(cell["requests_sha256"], "cell requests SHA256"),
        )

    prediction = _mapping(root["prediction_contract"], "prediction contract")
    _exact_fields(prediction, _PREDICTION_FIELDS, "prediction contract")
    _validate_prediction_contract(prediction)
    collective_backend = prediction["tensor_collective_backend"]
    if collective_backend == "protobridge":
        configured_profiles = (
            prediction["tensor_all_reduce_profile_sha256"],
            prediction["tensor_all_gather_profile_sha256"],
        )
        if (prediction["tensor_parallel_size"] == 1
                and configured_profiles == (None, None)):
            pass
        elif len(hardware_entries) != 1:
            raise ValueError(
                "ProtoBridge compact suite requires exactly one hardware "
                "profile")
        else:
            hardware_id = hardware_entries[0].hardware_id
            expected_profiles = (
                development_profile_config_sha256(hardware_id, "all_reduce"),
                development_profile_config_sha256(hardware_id, "all_gather"),
            )
            if configured_profiles != expected_profiles:
                raise ValueError(
                    "ProtoBridge prediction profiles disagree with the "
                    "declared hardware and operations")
    elif collective_backend == "analytical_roofline_v1":
        pass
    else:
        raise ValueError(
            f"unknown tensor collective backend: {collective_backend!r}")
    cell_qps_values = set()
    for cell_id in cell_order:
        case_id = cell_id.rsplit("/", maxsplit=1)[1]
        _validate_case_id_contract(case_id, prediction)
        case_match = _CASE_ID.fullmatch(case_id)
        if case_match is None:
            raise AssertionError("validated case ID unexpectedly became invalid")
        cell_qps_values.add(int(case_match.group("qps")))
    if cell_qps_values != set(qps_values):
        raise ValueError("cell case_id QPS values disagree with qps_values")
    if not cell_order:
        raise ValueError("compact suite must contain at least one cell")
    if suite_id == "sharegpt_phase1":
        expected_models = ("qwen3-4b", "qwen3-8b", "llama-3.1-8b")
        actual_models = tuple(
            _LEGACY_MODEL_IDS[model.dataset_model_id]
            if model.dataset_model_id in _LEGACY_MODEL_IDS
            else model.dataset_model_id
            for model in models
        )
        if actual_models != expected_models:
            raise ValueError("phase-1 models must contain the frozen ordered model set")
        if tuple(entry.hardware_id for entry in hardware_entries) != (
            _OFFICIAL_HARDWARE
        ):
            raise ValueError("phase-1 hardware must contain the frozen ordered set")
        if tuple(qps_values) != (4, 16, 64):
            raise ValueError("phase-1 qps_values must equal [4, 16, 64]")
        for field, expected in _EXPECTED_PREDICTION.items():
            actual = prediction[field]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    "prediction contract disagrees with the frozen baseline at "
                    f"{field}"
                )
        expected_cell_order = [
            f"{hardware.hardware_id}/{model.dataset_model_id}/tp1_qps{qps}"
            for hardware in hardware_entries
            for model in models
            for qps in qps_values
        ]
        if cell_order != expected_cell_order:
            raise ValueError(
                "phase-1 cells must equal the frozen ordered 27-cell matrix"
            )
    return Phase1Config(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        suite_id=suite_id,
        dataset_root=root["dataset_root"],
        models=tuple(models),
        hardware=tuple(hardware_entries),
        qps_values=tuple(qps_values),
        cell_order=tuple(cell_order),
        cell_hashes=cell_hashes,
        prediction=PredictionContract(dict(prediction)),
    )


def phase1_cells(config: Phase1Config) -> tuple[Phase1Cell, ...]:
    """Return compact-suite cells in their declared stable order."""

    hardware_by_id = {entry.hardware_id: entry for entry in config.hardware}
    model_by_id = {entry.dataset_model_id: entry for entry in config.models}
    cells = []
    for cell_id in config.cell_order:
        hardware_id, model_id, case_id = cell_id.split("/", maxsplit=2)
        case_match = _CASE_ID.fullmatch(case_id)
        if case_match is None:
            raise AssertionError("validated compact-suite case lost its identity")
        hashes = config.cell_hashes[cell_id]
        cells.append(Phase1Cell(
            cell_id=cell_id,
            model=model_by_id[model_id],
            hardware=hardware_by_id[hardware_id],
            qps=int(case_match.group("qps")),
            case_id=case_id,
            manifest_sha256=hashes[0],
            requests_sha256=hashes[1],
        ))
    if config.suite_id == "sharegpt_phase1" and len(cells) != 27:
        raise AssertionError("frozen first-phase matrix must contain 27 cells")
    return tuple(cells)


def official_metrics_suite_runs(
    repository_root: str | Path,
    *,
    nvidia_h200_vllm_version: str | None = None,
) -> tuple[tuple[Phase1Config, tuple[Phase1Cell, ...]], ...]:
    """Return the non-overlapping suite selections covering all 45 truth cells.

    ``nvidia_h200_vllm_version`` additionally selects the 15 H200 cells of one
    captured vLLM version.
    """

    root = Path(repository_root).resolve()
    selections = list(OFFICIAL_METRICS_SUITE_SELECTIONS)
    expected = _official_metrics_case_keys()
    if nvidia_h200_vllm_version is not None:
        if nvidia_h200_vllm_version not in NVIDIA_H200_SUITE_SELECTIONS:
            raise ValueError(
                f"unknown H200 vLLM version: {nvidia_h200_vllm_version!r}")
        selections.extend(
            (path, ("nvidia_h200",))
            for path in NVIDIA_H200_SUITE_SELECTIONS[nvidia_h200_vllm_version])
        expected = expected | _matrix_case_keys(("nvidia_h200",))
    runs = []
    observed = set()
    for relative_path, hardware_ids in selections:
        config = load_phase1_config(root / relative_path)
        cells = tuple(
            cell
            for cell in phase1_cells(config)
            if hardware_ids is None or cell.hardware.hardware_id in hardware_ids
        )
        if not cells:
            raise ValueError(f"official metrics suite selects no cells: {relative_path}")
        for cell in cells:
            key = (
                cell.hardware.hardware_id,
                _canonical_dataset_model_id(cell.model.dataset_model_id),
                cell.case_id,
            )
            if key in observed:
                raise ValueError(f"duplicate official metrics cell: {'/'.join(key)}")
            observed.add(key)
        runs.append((config, cells))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "official metrics suites do not cover the selected cell matrix; "
            f"missing={missing}, extra={extra}"
        )
    return tuple(runs)


def _official_metrics_case_keys() -> set[tuple[str, str, str]]:
    keys = _matrix_case_keys(_OFFICIAL_HARDWARE)
    if len(keys) != 45:
        raise AssertionError("official ShareGPT metric matrix must contain 45 cells")
    return keys


def _matrix_case_keys(hardware_ids: Sequence[str]) -> set[tuple[str, str, str]]:
    keys = {
        (hardware_id, model_id, f"tp1_qps{qps}")
        for hardware_id in hardware_ids
        for model_id in ("qwen3-4b", "qwen3-8b", "llama-3.1-8b")
        for qps in (4, 16, 64)
    }
    keys.update({
        (hardware_id, model_id, f"tp2_ep{ep}_qps{qps}")
        for hardware_id in hardware_ids
        for model_id, ep in (("qwen3-30b-a3b", 2), ("qwen3-32b", 1))
        for qps in (1, 4, 16)
    })
    return keys


def _canonical_cell_id(cell: Phase1Cell) -> str:
    return "/".join((
        cell.hardware.hardware_id,
        _canonical_dataset_model_id(cell.model.dataset_model_id),
        cell.case_id,
    ))


def select_phase1_cells(
    config: Phase1Config,
    requested_cell_ids: Sequence[str],
    *,
    all_cases: bool,
) -> tuple[Phase1Cell, ...]:
    """Select explicit cells; an empty CLI never starts all 27 implicitly."""

    cells = phase1_cells(config)
    if all_cases:
        if requested_cell_ids:
            raise ValueError("--all-cases cannot be combined with --case")
        return cells
    if not requested_cell_ids:
        raise ValueError("select at least one --case or pass --all-cases")
    if len(requested_cell_ids) != len(set(requested_cell_ids)):
        raise ValueError("phase-1 case selection contains duplicates")
    by_id = {cell.cell_id: cell for cell in cells}
    unknown = [cell_id for cell_id in requested_cell_ids if cell_id not in by_id]
    if unknown:
        raise ValueError(f"unknown phase-1 cell IDs: {unknown}")
    return tuple(by_id[cell_id] for cell_id in requested_cell_ids)


def compact_case_path(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
) -> Path:
    root = Path(repository_root).resolve()
    dataset_root = _confined_path(root, config.dataset_root, "dataset_root")
    _validate_case_id_contract(cell.case_id, config.prediction.values)
    expected_cell_id = (
        f"{cell.hardware.hardware_id}/{cell.model.dataset_model_id}/{cell.case_id}"
    )
    if cell.cell_id != expected_cell_id:
        raise ValueError("compact cell_id disagrees with its path components")
    canonical_model_id = _canonical_dataset_model_id(cell.model.dataset_model_id)
    case_path = (
        dataset_root
        / cell.hardware.hardware_id
        / canonical_model_id
        / cell.case_id
    ).resolve()
    if dataset_root not in case_path.parents:
        raise ValueError(f"compact case path escapes dataset_root: {cell.cell_id}")
    return case_path


def _validate_case_binding(
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
) -> None:
    manifest = _mapping(case.manifest, "compact manifest")
    identity = _mapping(manifest["identity"], "manifest identity")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    expected_identity = {
        "hardware_id": cell.hardware.hardware_id,
        "model_id": _canonical_dataset_model_id(cell.model.dataset_model_id),
        "case_id": cell.case_id,
        "source_kind": "sharegpt",
    }
    for field, expected in expected_identity.items():
        if identity[field] != expected:
            raise ValueError(
                f"{cell.cell_id} manifest {field} mismatch: "
                f"{identity[field]!r} != {expected!r}"
            )
    runtime_expected = {
        "tensor_parallel_size": config.prediction.values[
            "tensor_parallel_size"
        ],
        "pipeline_parallel_size": config.prediction.values[
            "pipeline_parallel_size"
        ],
        "expert_parallel_size": config.prediction.values[
            "expert_parallel_size"
        ],
        "target_qps": float(cell.qps),
    }
    for field, expected in runtime_expected.items():
        if runtime[field] != expected:
            raise ValueError(
                f"{cell.cell_id} capture runtime {field} mismatch: "
                f"{runtime[field]!r} != {expected!r}"
            )
    provenance = _mapping(manifest["provenance"], "manifest provenance")
    if case.metrics is None:
        raise ValueError(f"{cell.cell_id} requires available official metrics")
    _sequence(provenance["gaps"], "manifest provenance gaps")


def validate_case_data_pins(cell: Phase1Cell, case: ValidatedCompactCase) -> None:
    """Bind a loaded case's actual files to the selected suite cell."""

    case_dir = Path(case.case_dir).resolve()
    manifest_path = case_dir / "manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha != cell.manifest_sha256:
        raise ValueError(f"{cell.cell_id} suite manifest SHA256 differs from case bytes")
    requests = _mapping(case.manifest["requests"], "manifest requests")
    requests_path = (case_dir / _relative_path(
        requests["path"], "manifest requests path")).resolve()
    if case_dir not in requests_path.parents:
        raise ValueError(f"{cell.cell_id} requests path escapes case directory")
    requests_sha = hashlib.sha256(requests_path.read_bytes()).hexdigest()
    if (requests_sha != cell.requests_sha256
            or requests_sha != _sha256(requests["sha256"], "manifest requests SHA256")):
        raise ValueError(f"{cell.cell_id} suite requests SHA256 differs from case bytes")


def _canonical_dataset_model_id(model_id: str) -> str:
    if model_id in _LEGACY_MODEL_IDS:
        return _LEGACY_MODEL_IDS[model_id]
    if model_id in _CANONICAL_MODEL_SPECS:
        return model_id
    raise ValueError(f"unsupported compact dataset model_id: {model_id!r}")


def _validate_prediction_contract(prediction: Mapping[str, object]) -> None:
    # Prediction runtime identity is independent of capture provenance. Bind
    # each supported CPU Scheduler version to its authoritative source owner.
    scheduler_sources = {
        VLLM_SCHEDULER_VERSION: VLLM_SCHEDULER_SOURCE_COMMIT,
        "0.30.0": VLLM030_SOURCE_COMMIT,
    }
    version = prediction["scheduler_version"]
    if type(version) is not str or version not in scheduler_sources:
        raise ValueError(f"prediction contract has unsupported scheduler_version: {version!r}")
    if prediction["scheduler_source_commit"] != scheduler_sources[version]:
        raise ValueError("prediction scheduler_source_commit disagrees with its declared version")
    fixed_fields = {
        field: expected
        for field, expected in _EXPECTED_PREDICTION.items()
        if field not in {
            "scheduler_version",
            "scheduler_source_commit",
            "launch_policy",
            "tensor_collective_backend",
            "tensor_all_reduce_profile_sha256",
            "tensor_all_gather_profile_sha256",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "expert_parallel_size",
            "enable_prefix_caching",
            "compute_dtype",
        }
    }
    for field, expected in fixed_fields.items():
        actual = prediction[field]
        if field == "router_dispatch_service_time_s":
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                raise TypeError(
                    "prediction router_dispatch_service_time_s must be numeric"
                )
            if not math.isfinite(float(actual)) or float(actual) != 0.0:
                raise ValueError(
                    "prediction router_dispatch_service_time_s must equal zero"
                )
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"prediction contract has unsupported {field}: {actual!r}"
            )
    if prediction["compute_dtype"] not in ("bf16", "fp16"):
        raise ValueError("prediction contract has unsupported compute_dtype")
    if prediction["launch_policy"] not in (
        "injected_launch", "analytical_launch"
    ):
        raise ValueError("prediction contract has unknown launch_policy")
    collective_backend = prediction["tensor_collective_backend"]
    profile_values = (
        prediction["tensor_all_reduce_profile_sha256"],
        prediction["tensor_all_gather_profile_sha256"],
    )
    if collective_backend == "analytical_roofline_v1":
        if any(value is not None for value in profile_values):
            raise ValueError(
                "analytical collective prediction requires null profiles")
    elif collective_backend == "protobridge":
        if prediction["tensor_parallel_size"] == 1:
            if profile_values != (None, None):
                raise ValueError(
                    "TP=1 ProtoBridge prediction must not declare unused "
                    "collective profiles")
        else:
            for value in profile_values:
                _sha256(value, "ProtoBridge profile SHA256")
    else:
        raise ValueError(
            f"prediction contract has unknown tensor collective backend: "
            f"{collective_backend!r}")
    integer_fields = (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
    )
    for field in integer_fields:
        value = prediction[field]
        if type(value) is not int:
            raise ValueError(f"prediction {field} must be an exact integer")
        if value < 1:
            raise ValueError(f"prediction {field} must be positive")
    if prediction["pipeline_parallel_size"] != 1:
        raise ValueError("compact evaluator currently supports PP=1 only")
    if prediction["expert_parallel_size"] not in (
        1, prediction["tensor_parallel_size"]
    ):
        raise ValueError("expert_parallel_size must equal one or TP size")
    if type(prediction["enable_prefix_caching"]) is not bool:
        raise TypeError("prediction enable_prefix_caching must be a boolean")


def _validate_case_id_contract(
    case_id: str,
    prediction: Mapping[str, object],
) -> None:
    match = _CASE_ID.fullmatch(case_id)
    if match is None:
        raise ValueError(f"cell case_id is not canonical: {case_id!r}")
    case_tp = int(match.group("tp"))
    case_ep_text = match.group("ep")
    case_ep = 1 if case_ep_text is None else int(case_ep_text)
    expected_tp = prediction["tensor_parallel_size"]
    expected_ep = prediction["expert_parallel_size"]
    if case_tp != expected_tp:
        raise ValueError(
            f"case_id {case_id!r} TP={case_tp} disagrees with "
            f"prediction TP={expected_tp}"
        )
    if case_ep != expected_ep:
        raise ValueError(
            f"case_id {case_id!r} EP={case_ep} disagrees with "
            f"prediction EP={expected_ep}"
        )
    if case_ep_text is None and (expected_tp, expected_ep) not in (
        (1, 1),
        (4, 1),
    ):
        raise ValueError(
            f"case_id {case_id!r} may omit EP only for TP1 or TP4 with EP1"
        )


def _model_config_path(
    repository_root: str | Path,
    value: object,
) -> Path:
    path = _confined_path(
        Path(repository_root).resolve(), value, "simulation model config")
    if not path.is_file():
        raise ValueError(f"simulation model config is not a file: {path}")
    return path
