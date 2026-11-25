"""Build the complete static ShareGPT GEMM launch-selection domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from dataset_pipeline.kernel_dataset.config import file_sha256
from dataset_pipeline.launch_data.json_validators import (
    canonical_sequence_sha256,
)
from src.kernel.launch_config.gemm_workload import (
    GEMM_WORKLOAD_KEY_FIELDS,
    SERVING_MODEL_PARALLELISM,
    SERVING_SCHEDULER_CONTRACT,
    derive_model_gemm_workload_domain,
)
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS


HARDWARE_IDS = EVALUATED_HARDWARE_IDS
PHYSICAL_OPERATION_FIELDS = ("m", "n", "k", "dtype", "has_bias")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return value


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def _repository_file(repository_root: Path, path: Path, context: str) -> Path:
    resolved = path.resolve()
    if repository_root not in resolved.parents:
        raise ValueError(f"{context} must stay below the repository root")
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} is absent: {resolved}")
    return resolved


def _validate_scheduler_contract(value: object) -> dict[str, object]:
    contract = _mapping(value, "scheduler_contract")
    if set(contract) != set(SERVING_SCHEDULER_CONTRACT):
        raise ValueError(
            "scheduler_contract fields must be exactly "
            f"{sorted(SERVING_SCHEDULER_CONTRACT)}")
    normalized = dict(contract)
    if normalized != SERVING_SCHEDULER_CONTRACT:
        raise ValueError(
            "scheduler_contract differs from the complete launch-domain "
            f"boundary: expected={SERVING_SCHEDULER_CONTRACT}, "
            f"actual={normalized}")
    return normalized


def _validate_inventory(dataset_root: Path) -> dict[str, list[str]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"ShareGPT dataset root is absent: {dataset_root}")
    actual_hardware = sorted(
        path.name for path in dataset_root.iterdir() if path.is_dir())
    if actual_hardware != sorted(HARDWARE_IDS):
        raise ValueError(
            "ShareGPT hardware directory set differs from the exact launch "
            f"domain: expected={sorted(HARDWARE_IDS)}, actual={actual_hardware}")
    # The canonical ShareGPT dataset addresses a model by its lowercase
    # directory slug, which the serving alignment suite enforces; the launch
    # domain names the same models by their model identity.  Compare the two
    # over the shared slug so the inventory check tracks the dataset contract
    # instead of a second spelling of it.
    expected_models = sorted(
        model_id.lower() for model_id in SERVING_MODEL_PARALLELISM)
    if len(expected_models) != len(set(expected_models)):
        raise ValueError("launch-domain model slugs are not unique")
    inventory = {}
    for hardware_id in HARDWARE_IDS:
        hardware_root = dataset_root / hardware_id
        actual_models = sorted(
            path.name for path in hardware_root.iterdir() if path.is_dir())
        if actual_models != expected_models:
            raise ValueError(
                f"{hardware_id} ShareGPT model directory set differs from the "
                f"exact launch domain: expected={expected_models}, "
                f"actual={actual_models}")
        inventory[hardware_id] = actual_models
    return inventory


def _model_path(models_root: Path, configured_path: object, model_id: str) -> Path:
    path_text = _nonempty_string(
        configured_path, f"model {model_id} model_path")
    expected = f"${{MODELS_ROOT}}/{model_id}"
    if model_id == "Llama-3.1-8B":
        expected = "${MODELS_ROOT}/Llama-3.1-8B-Instruct"
    if path_text != expected:
        raise ValueError(
            f"model {model_id} path must be exactly {expected!r}, got "
            f"{path_text!r}")
    relative = Path(path_text.removeprefix("${MODELS_ROOT}/"))
    resolved = (models_root / relative).resolve()
    if models_root not in resolved.parents or not resolved.is_dir():
        raise FileNotFoundError(
            f"model {model_id} directory is absent below --models-root: "
            f"{resolved}")
    return resolved


def build_gemm_workload_candidate(
        repository_root: Path, config_path: Path, *,
        models_root: Path) -> dict[str, object]:
    """Return the exhaustive, timing-free legal GEMM launch domain."""

    repository_root = Path(repository_root).resolve()
    if not repository_root.is_dir():
        raise FileNotFoundError(f"repository root is absent: {repository_root}")
    config_path = _repository_file(
        repository_root, Path(config_path), "launch-domain config")
    models_root = Path(models_root).resolve()
    if not models_root.is_dir():
        raise FileNotFoundError(f"--models-root is absent: {models_root}")
    config = _mapping(json.loads(config_path.read_text()), "launch-domain config")
    expected_config_fields = {
        "schema_version", "artifact_type", "dataset_root", "hardware_ids",
        "scheduler_contract", "models",
    }
    if set(config) != expected_config_fields:
        raise ValueError(
            "launch-domain config fields must be exactly "
            f"{sorted(expected_config_fields)}")
    if config["schema_version"] != 1:
        raise ValueError("launch-domain config schema_version must be 1")
    if config["artifact_type"] != "gemm_launch_workload_domain_config":
        raise ValueError("launch-domain config artifact_type is unsupported")
    hardware_ids = _sequence(config["hardware_ids"], "hardware_ids")
    if tuple(hardware_ids) != HARDWARE_IDS:
        raise ValueError(
            f"hardware_ids must be exactly {list(HARDWARE_IDS)}")
    dataset_relative = Path(_nonempty_string(
        config["dataset_root"], "dataset_root"))
    if (dataset_relative.is_absolute()
            or dataset_relative.as_posix() != config["dataset_root"]
            or ".." in dataset_relative.parts):
        raise ValueError("dataset_root must be a canonical relative path")
    dataset_root = (repository_root / dataset_relative).resolve()
    inventory = _validate_inventory(dataset_root)
    scheduler = _validate_scheduler_contract(config["scheduler_contract"])

    configured_models = _sequence(config["models"], "models")
    model_records = {}
    for index, value in enumerate(configured_models):
        record = _mapping(value, f"models[{index}]")
        expected_model_fields = {
            "model_id", "model_path", "model_config_sha256",
            "tensor_parallel_size", "expert_parallel_size",
        }
        if set(record) != expected_model_fields:
            raise ValueError(
                f"models[{index}] fields must be exactly "
                f"{sorted(expected_model_fields)}")
        model_id = _nonempty_string(record["model_id"], f"models[{index}] model_id")
        if model_id not in SERVING_MODEL_PARALLELISM:
            raise ValueError(f"unknown launch-domain model {model_id!r}")
        if model_id in model_records:
            raise ValueError(f"duplicate launch-domain model {model_id!r}")
        tp = _positive_integer(
            record["tensor_parallel_size"], f"model {model_id} TP")
        ep = _positive_integer(
            record["expert_parallel_size"], f"model {model_id} EP")
        if (tp, ep) != SERVING_MODEL_PARALLELISM[model_id]:
            raise ValueError(
                f"model {model_id} parallelism must be "
                f"TP={SERVING_MODEL_PARALLELISM[model_id][0]}, "
                f"EP={SERVING_MODEL_PARALLELISM[model_id][1]}")
        config_sha256 = _nonempty_string(
            record["model_config_sha256"],
            f"model {model_id} model_config_sha256")
        if (len(config_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in config_sha256)):
            raise ValueError(
                f"model {model_id} model_config_sha256 must be lowercase SHA256")
        path = _model_path(models_root, record["model_path"], model_id)
        actual_sha256 = file_sha256(path / "config.json")
        if actual_sha256 != config_sha256:
            raise ValueError(
                f"model {model_id} config.json SHA256 mismatch: "
                f"expected={config_sha256}, actual={actual_sha256}")
        model_domain = derive_model_gemm_workload_domain(
            str(path), model_id,
            tensor_parallel_size=tp,
            expert_parallel_size=ep,
            max_batched_tokens=scheduler["max_batched_tokens"],
            max_requests=scheduler["max_requests"],
        )
        if model_domain["config_sha256"] != config_sha256:
            raise RuntimeError(
                f"canonical load_model_config hash disagrees for {model_id}")
        if (model_domain["compute_dtype"] != "bf16"
                or model_domain["weight_dtype"] != "bf16"):
            raise ValueError(
                f"model {model_id} must use BF16 compute and weights")
        model_records[model_id] = (record, model_domain, tp, ep)
    if set(model_records) != set(SERVING_MODEL_PARALLELISM):
        raise ValueError(
            "configured model set differs from the exact launch domain: "
            f"expected={sorted(SERVING_MODEL_PARALLELISM)}, "
            f"actual={sorted(model_records)}")

    all_keys = set()
    all_physical_operations = set()
    model_domains = []
    for model_id in SERVING_MODEL_PARALLELISM:
        record, model_domain, tp, ep = model_records[model_id]
        keys = model_domain["workload_keys"]
        physical_operations = model_domain["physical_operations"]
        positions = model_domain["positions"]
        all_keys.update(keys)
        all_physical_operations.update(physical_operations)
        model_domains.append({
            "model_id": model_id,
            "model_path": record["model_path"],
            "model_config_sha256": record["model_config_sha256"],
            "model_type": model_domain["model_type"],
            "tensor_parallel_size": tp,
            "expert_parallel_size": ep,
            "positions": positions,
            "semantic_key_count": len(keys),
            "physical_operation_count": len(physical_operations),
        })
    workload_keys = sorted(all_keys)
    domain_sha256 = canonical_sequence_sha256(workload_keys)
    hardware_domains = {
        hardware_id: {
            "status": "exact",
            "domain_sha256": domain_sha256,
            "semantic_key_count": len(workload_keys),
            "physical_operation_count": len(all_physical_operations),
        }
        for hardware_id in HARDWARE_IDS
    }
    return {
        "schema_version": 2,
        "artifact_type": "gemm_workload_candidate",
        "source_config": str(config_path.relative_to(repository_root)),
        "source_config_sha256": file_sha256(config_path),
        "duration_consumed": False,
        "workload_key_fields": list(GEMM_WORKLOAD_KEY_FIELDS),
        "physical_operation_fields": list(PHYSICAL_OPERATION_FIELDS),
        "scheduler_contract": scheduler,
        "coverage_scope": {
            "kind": "static_complete_legal_launch_selector_domain",
            "request_trajectory_consumed": False,
            "scheduler_alignment_consumed": False,
            "operator_trace_consumed": False,
            "timing_consumed": False,
            "fused_moe_included": False,
            "serving_end_to_end_executability_claimed": False,
        },
        "limitations": [
            {
                "scope": "serving_workload_lowering",
                "affected_hardware_ids": ["metax_c500", "ascend_910b"],
                "affected_model_ids": ["Qwen3-32B"],
                "reason": (
                    "current colocated TP=2 workload emission is restricted "
                    "to nvidia_a800"),
            },
            {
                "scope": "serving_workload_lowering",
                "affected_hardware_ids": list(HARDWARE_IDS),
                "affected_model_ids": ["Qwen3-30B-A3B"],
                "reason": (
                    "current EP=2 workload emission supports only two routed "
                    "experts with top-k=1; this model has 128 experts and top-k=8"),
            },
        ],
        "dataset_model_inventory": inventory,
        "model_domains": model_domains,
        "workload_keys": workload_keys,
        "semantic_key_count": len(workload_keys),
        "physical_operation_count": len(all_physical_operations),
        "hardware_domains": hardware_domains,
        "cross_hardware": {
            "status": "exact_equal",
            "hardware_ids": list(HARDWARE_IDS),
            "domain_sha256": domain_sha256,
            "semantic_key_count": len(workload_keys),
            "physical_operation_count": len(all_physical_operations),
        },
    }
