"""Case contract loading for ShareGPT ground-truth capture:
repository root, supported hardware, runtime retargeting, and
file identity."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS


ROOT = Path(__file__).resolve().parents[3]


SUPPORTED_HARDWARE_IDS = EVALUATED_HARDWARE_IDS
# A hardware without its own canonical ShareGPT cases replays the identical
# request file and arrival schedule of a template hardware's canonical case.
# The template identity is recorded in the result; only the hardware and the
# runtime release change.
REQUEST_TEMPLATE_HARDWARE_BY_TARGET = {"nvidia_h200": "nvidia_a800"}
H200_RUNTIME_VERSIONS = ("0.11.0", "0.30.0")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_vllm_release() -> str:
    from importlib.metadata import version

    return version("vllm").split("+", 1)[0]


def _vllm_source_identity(release: str) -> str:
    """Identify the installed vLLM source the same way canonical A800 cases do.

    Canonical runtime.source_commit values have the form
    ``vllm=<release>;core=<sha256[:8]>...<sha256[-6:]>;executor=<...>`` over
    ``vllm/v1/engine/core.py`` and ``vllm/v1/executor/multiproc_executor.py``.
    """

    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("installed vLLM package cannot be located")
    package = Path(spec.origin).resolve().parent

    def digest(relative: str) -> str:
        value = sha256_file(package / relative)
        return f"{value[:8]}...{value[-6:]}"

    return (
        f"vllm={release};core={digest('v1/engine/core.py')};"
        f"executor={digest('v1/executor/multiproc_executor.py')}"
    )


def _retarget_case_contract(
    case: dict[str, Any],
    *,
    target_hardware_id: str,
    target_runtime_version: str,
) -> dict[str, Any]:
    """Relabel a template hardware's canonical case for another hardware."""

    if target_hardware_id not in REQUEST_TEMPLATE_HARDWARE_BY_TARGET:
        raise ValueError(
            f"unsupported request-template target hardware: {target_hardware_id!r}"
        )
    template_hardware_id = REQUEST_TEMPLATE_HARDWARE_BY_TARGET[target_hardware_id]
    if case["request_template"] is not None:
        raise ValueError("case contract was already retargeted")
    if case["identity"]["hardware_id"] != template_hardware_id:
        raise ValueError(
            f"{target_hardware_id} replays {template_hardware_id} cases only, got "
            f"{case['identity']['hardware_id']!r}"
        )
    if target_runtime_version not in H200_RUNTIME_VERSIONS:
        raise ValueError(
            f"unsupported H200 runtime version: {target_runtime_version!r}")
    installed = _installed_vllm_release()
    if installed != target_runtime_version:
        raise RuntimeError(
            "requested runtime version disagrees with the installed vLLM: "
            f"{target_runtime_version} != {installed}"
        )
    template_runtime = case["runtime"]
    identity = dict(case["identity"])
    identity["hardware_id"] = target_hardware_id
    runtime = {
        "dtype": template_runtime["dtype"],
        "tensor_parallel_size": template_runtime["tensor_parallel_size"],
        "pipeline_parallel_size": template_runtime["pipeline_parallel_size"],
        "expert_parallel_size": template_runtime["expert_parallel_size"],
        "target_qps": template_runtime["target_qps"],
        "runtime_version": target_runtime_version,
        "source_commit": _vllm_source_identity(installed),
    }
    return {
        **case,
        "identity": identity,
        "runtime": runtime,
        "request_template": {
            "hardware_id": template_hardware_id,
            "case_manifest_path": case["manifest_path"],
            "case_manifest_sha256": case["manifest_sha256"],
            "requests_sha256": case["requests_sha256"],
        },
    }


# The dtypes the production ground-truth runner accepts per hardware: the
# MetaX capture retains a float16 cell as well as bfloat16, the others
# bfloat16 only.
_PRODUCTION_DTYPES = {
    "nvidia_a800": ("bfloat16",),
    "nvidia_h200": ("bfloat16",),
    "metax_c500": ("bfloat16", "float16"),
    "ascend_910b": ("bfloat16",),
}


def _load_case_contract(
    manifest_path: str | Path,
    *,
    model_path: str | Path,
    suite_manifest_path: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve(strict=True)
    raw = json.loads(path.read_text())
    identity = raw["identity"]
    requests = raw["requests"]
    runtime = raw["runtime"]
    if identity["source_kind"] != "sharegpt":
        raise ValueError("ground-truth runner requires a ShareGPT case")
    if identity["hardware_id"] not in SUPPORTED_HARDWARE_IDS:
        raise ValueError(
            "unsupported production ground-truth hardware_id: "
            f"{identity['hardware_id']!r}"
        )
    suite_path = Path(suite_manifest_path).resolve(strict=True)
    suite = json.loads(suite_path.read_text())
    dataset_root = (ROOT / suite["dataset_root"]).resolve(strict=True)
    expected_case_path = (
        dataset_root
        / identity["hardware_id"]
        / identity["model_id"]
        / identity["case_id"]
        / "manifest.json"
    ).resolve(strict=True)
    if path != expected_case_path:
        raise ValueError(
            "case manifest path disagrees with suite identity: "
            f"{path} != {expected_case_path}"
        )
    matching_cells = [
        cell
        for cell in suite["cells"]
        if cell["hardware_id"] == identity["hardware_id"]
        and cell["model_id"] == identity["model_id"]
        and cell["case_id"] == identity["case_id"]
    ]
    if len(matching_cells) != 1:
        raise ValueError(
            "suite must contain exactly one matching hardware/model/case cell"
        )
    cell = matching_cells[0]
    manifest_sha256 = sha256_file(path)
    if manifest_sha256 != cell["manifest_sha256"]:
        raise ValueError("case manifest SHA256 disagrees with suite manifest")
    requests_path = (path.parent / requests["path"]).resolve(strict=True)
    requests_sha256 = sha256_file(requests_path)
    if requests_sha256 != requests["sha256"]:
        raise ValueError("requests.csv SHA256 disagrees with case manifest")
    if requests_sha256 != cell["requests_sha256"]:
        raise ValueError("requests.csv SHA256 disagrees with suite manifest")
    matching_models = [
        model
        for model in suite["models"]
        if model["dataset_model_id"] == identity["model_id"]
    ]
    if len(matching_models) != 1:
        raise ValueError("suite must contain exactly one matching model identity")
    model_directory = Path(model_path).resolve(strict=True)
    model_config_path = (model_directory / "config.json").resolve(strict=True)
    model_config_sha256 = sha256_file(model_config_path)
    if model_config_sha256 != matching_models[0]["model_config_sha256"]:
        raise ValueError("local model config SHA256 disagrees with suite manifest")
    tensor_parallel_size = runtime["tensor_parallel_size"]
    if tensor_parallel_size not in (1, 2):
        raise ValueError("production ground-truth runner requires TP=1 or TP=2")
    if runtime["pipeline_parallel_size"] != 1:
        raise ValueError("production ground-truth runner requires PP=1")
    expert_parallel_size = runtime["expert_parallel_size"]
    if expert_parallel_size not in (1, 2):
        raise ValueError("production ground-truth runner requires EP=1 or EP=2")
    if expert_parallel_size == 2 and tensor_parallel_size != 2:
        raise ValueError("production ground-truth EP=2 requires TP=2")
    if identity["hardware_id"] not in _PRODUCTION_DTYPES:
        raise ValueError(
            f"unsupported production hardware: {identity['hardware_id']!r}"
        )
    if runtime["dtype"] not in _PRODUCTION_DTYPES[identity["hardware_id"]]:
        raise ValueError(
            "production ground-truth runner received an unsupported dtype"
        )
    target_qps = runtime["target_qps"]
    if (
        isinstance(target_qps, bool)
        or not isinstance(target_qps, (int, float))
        or not math.isfinite(target_qps)
        or target_qps <= 0
    ):
        raise ValueError("case manifest target_qps must be finite and positive")
    return {
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
        "suite_manifest_path": str(suite_path),
        "suite_manifest_sha256": sha256_file(suite_path),
        "identity": dict(identity),
        "runtime": dict(runtime),
        "model_config_path": str(model_config_path),
        "model_config_sha256": model_config_sha256,
        "requests_path": str(requests_path),
        "requests_sha256": requests_sha256,
        "requests_row_count": requests["row_count"],
        "request_template": None,
    }
