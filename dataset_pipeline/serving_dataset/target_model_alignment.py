"""Writer and validator for partial target-model alignment candidates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from .compact_case import ALIGNMENT_FILENAME
from .compressed_artifact import (
    COMPRESSION_ALGORITHM,
    compress_canonical_bytes,
    read_compressed_canonical_bytes,
    write_compressed_canonical_bytes,
)

from dataset_pipeline.serving_dataset.json_shape import (
    compact_mapping as _mapping,
    compact_sequence as _sequence,
)


TARGET_MODEL_ALIGNMENT_SCHEMA_VERSION = 4
_ALIGNMENT_KIND = "sharegpt_alignment_steps"
_PHASES = ("prefill", "decode", "mixed")
_TARGET_MODELS = {
    "glm-5.2": "zai-org/GLM-5.2",
    "kimi-k2.6": "moonshotai/Kimi-K2.6",
}


def canonical_target_model_alignment_bytes(
    payload: Mapping[str, object],
) -> bytes:
    """Validate and serialize one versioned partial alignment payload."""

    validate_target_model_alignment(payload)
    return _json_bytes(payload)


def validate_target_model_alignment(payload: Mapping[str, object]) -> None:
    """Validate a source-pinned actual-to-Midlayer partial alignment."""

    root = _mapping(payload, "target-model alignment")
    _exact_keys(
        root,
        (
            "schema_version",
            "kind",
            "source_identity",
            "operator_signatures",
            "launch_signatures",
            "phases",
        ),
        "target-model alignment",
    )
    if root["schema_version"] != TARGET_MODEL_ALIGNMENT_SCHEMA_VERSION:
        raise ValueError("target-model alignment schema_version must be 4")
    if root["kind"] != _ALIGNMENT_KIND:
        raise ValueError(
            "target-model alignment kind must be sharegpt_alignment_steps")
    _reject_duration(root, "target-model alignment")
    identity = _validate_source_identity(root["source_identity"])
    signatures = _validate_operator_signatures(root["operator_signatures"])
    launches = _mapping(root["launch_signatures"], "launch signatures")
    if launches:
        raise ValueError(
            "target-model module-slice alignment cannot claim launch capture")
    phases = _mapping(root["phases"], "target-model alignment phases")
    _exact_keys(phases, _PHASES, "target-model alignment phases")
    available = []
    for phase in _PHASES:
        entry = _mapping(phases[phase], f"alignment phase {phase}")
        if entry["status"] == "available":
            _exact_keys(entry, ("status", "samples"), f"alignment phase {phase}")
            samples = _sequence(entry["samples"], f"alignment phase {phase} samples")
            if not samples:
                raise ValueError(f"available alignment phase {phase} requires samples")
            sample_ids = []
            for value in samples:
                sample = _mapping(value, f"alignment phase {phase} sample")
                sample_ids.append(_validate_sample(
                    sample,
                    phase=phase,
                    model_id=identity["model_id"],
                    signatures=signatures,
                ))
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError(
                    f"alignment phase {phase} sample IDs must be unique")
            available.append(phase)
        elif entry["status"] == "not_observed":
            _exact_keys(
                entry,
                ("status", "reason"),
                f"alignment phase {phase}",
            )
            _nonempty_string(entry["reason"], f"alignment phase {phase} reason")
        else:
            raise ValueError(
                f"unsupported target-model alignment phase status: "
                f"{entry['status']!r}")
    if tuple(available) != ("prefill", "decode"):
        raise ValueError(
            "target-model alignment requires Prefill and Decode samples and "
            "an explicit Mixed not_observed boundary")


def write_target_model_alignment_candidate(
    source_case_dir: str | Path,
    output_case_dir: str | Path,
    payload: Mapping[str, object],
    *,
    candidate_root: str | Path,
) -> Path:
    """Write a non-overwriting candidate beside the canonical source case."""

    source = Path(source_case_dir).resolve()
    output = Path(output_case_dir).resolve()
    allowed_root = Path(candidate_root).resolve()
    if (
        allowed_root.name != "target_model_alignment"
        or allowed_root.parent.name != "profiler_output"
    ):
        raise ValueError(
            "target-model candidate root must be exactly "
            "<execution-root>/profiler_output/target_model_alignment")
    if output == allowed_root or allowed_root not in output.parents:
        raise ValueError("target-model candidate must be below candidate_root")
    if not source.is_dir():
        raise ValueError(f"source case is not a directory: {source}")
    if output.exists():
        raise FileExistsError(f"candidate case already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"candidate parent is not a directory: {output.parent}")
    manifest = _json_object(source / "manifest.json", "source manifest")
    identity = _mapping(manifest["identity"], "source manifest identity")
    _exact_keys(
        identity,
        (
            "case_id",
            "hardware_id",
            "model_id",
            "model_revision",
            "source_kind",
            "tokenizer_revision",
        ),
        "source manifest identity",
    )
    if identity["source_kind"] != "production":
        raise ValueError("target-model alignment requires a Production case")
    if identity["case_id"] != "debug_workload_1s":
        raise ValueError(
            "target-model alignment is restricted to debug_workload_1s")
    if identity["model_id"] not in _TARGET_MODELS:
        raise ValueError("target-model alignment received an unsupported model")
    payload_identity = _mapping(
        payload["source_identity"], "target-model source_identity")
    if payload_identity["canonical_model_id"] != identity["model_id"]:
        raise ValueError(
            "target-model alignment model disagrees with the case manifest")
    relative = output.relative_to(allowed_root)
    if relative.parts != (identity["model_id"], identity["case_id"]):
        raise ValueError(
            "candidate output path must match model_id/case_id identity")
    requests_path = source / "requests.csv"
    request_bytes = requests_path.read_bytes()
    requests = _mapping(manifest["requests"], "source manifest requests")
    if requests["requests_sha256"] != hashlib.sha256(request_bytes).hexdigest():
        raise ValueError("source requests SHA256 is inconsistent")
    alignment_bytes = canonical_target_model_alignment_bytes(payload)
    alignment_sha = hashlib.sha256(alignment_bytes).hexdigest()
    phases = _mapping(payload["phases"], "target-model alignment phases")
    phase_manifest = {}
    for phase in _PHASES:
        entry = _mapping(phases[phase], f"alignment phase {phase}")
        if entry["status"] == "available":
            rank_ids = [0]
        elif entry["status"] == "not_observed":
            rank_ids = []
        else:
            raise ValueError(
                f"unsupported target-model alignment phase status: "
                f"{entry['status']!r}")
        phase_manifest[phase] = {
            "status": entry["status"],
            "rank_ids": rank_ids,
        }
    updated_manifest = dict(manifest)
    compressed_alignment = compress_canonical_bytes(alignment_bytes)
    updated_manifest["alignment_steps"] = {
        "status": "available",
        "path": ALIGNMENT_FILENAME,
        "sha256": alignment_sha,
        "compressed_sha256": hashlib.sha256(
            compressed_alignment).hexdigest(),
        "compression": COMPRESSION_ALGORITHM,
        "phases": phase_manifest,
    }
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.candidate-", dir=output.parent))
    try:
        (temporary / "requests.csv").write_bytes(request_bytes)
        write_compressed_canonical_bytes(
            temporary / ALIGNMENT_FILENAME, alignment_bytes)
        (temporary / "manifest.json").write_bytes(_json_bytes(updated_manifest))
        validate_target_model_alignment_case(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def validate_target_model_alignment_case(case_dir: str | Path) -> None:
    """Validate one debug Production case and its partial alignment binding."""

    root = Path(case_dir)
    if not root.is_dir():
        raise ValueError(f"target-model alignment case is not a directory: {root}")
    actual_files = {
        path.relative_to(root) for path in root.rglob("*") if path.is_file()
    }
    expected_files = {
        Path("manifest.json"),
        Path("requests.csv"),
        Path(ALIGNMENT_FILENAME),
    }
    if actual_files != expected_files:
        raise ValueError(
            "target-model alignment case file set must contain only manifest, "
            "requests, and alignment_steps")
    manifest = _json_object(root / "manifest.json", "case manifest")
    if manifest["schema_version"] != 1:
        raise ValueError("target-model source case manifest must use schema version 1")
    identity = _mapping(manifest["identity"], "case manifest identity")
    if (
        identity["source_kind"] != "production"
        or identity["case_id"] != "debug_workload_1s"
        or identity["model_id"] not in _TARGET_MODELS
    ):
        raise ValueError("target-model alignment case identity is invalid")
    request_bytes = (root / "requests.csv").read_bytes()
    requests = _mapping(manifest["requests"], "case manifest requests")
    if requests["requests_sha256"] != hashlib.sha256(request_bytes).hexdigest():
        raise ValueError("case requests SHA256 is inconsistent")
    entry = _mapping(manifest["alignment_steps"], "manifest alignment_steps")
    _exact_keys(
        entry,
        (
            "status", "path", "sha256", "compressed_sha256",
            "compression", "phases",
        ),
        "manifest alignment_steps",
    )
    if entry["status"] != "available" or entry["path"] != ALIGNMENT_FILENAME:
        raise ValueError("manifest alignment_steps must be available")
    if entry["compression"] != COMPRESSION_ALGORITHM:
        raise ValueError("manifest alignment_steps compression is invalid")
    alignment_bytes = read_compressed_canonical_bytes(
        root / ALIGNMENT_FILENAME,
        compressed_sha256=entry["compressed_sha256"],
        content_sha256=entry["sha256"],
        label=ALIGNMENT_FILENAME,
    )
    payload = _mapping(json.loads(alignment_bytes), "alignment_steps")
    validate_target_model_alignment(payload)
    source_identity = _mapping(payload["source_identity"], "source_identity")
    if source_identity["canonical_model_id"] != identity["model_id"]:
        raise ValueError("alignment source identity disagrees with manifest")
    manifest_phases = _mapping(entry["phases"], "manifest alignment phases")
    _exact_keys(manifest_phases, _PHASES, "manifest alignment phases")
    alignment_phases = _mapping(payload["phases"], "alignment phases")
    for phase in _PHASES:
        declared = _mapping(
            manifest_phases[phase], f"manifest alignment phase {phase}")
        _exact_keys(
            declared,
            ("status", "rank_ids"),
            f"manifest alignment phase {phase}",
        )
        actual = _mapping(alignment_phases[phase], f"alignment phase {phase}")
        if declared["status"] != actual["status"]:
            raise ValueError(
                f"manifest phase {phase} status disagrees with alignment")
        if actual["status"] == "available":
            expected_ranks = [0]
        elif actual["status"] == "not_observed":
            expected_ranks = []
        else:
            raise ValueError(
                f"unsupported target-model alignment phase status: "
                f"{actual['status']!r}")
        if declared["rank_ids"] != expected_ranks:
            raise ValueError(
                f"manifest phase {phase} rank_ids disagree with alignment")


def _validate_source_identity(value: object) -> Mapping[str, object]:
    identity = _mapping(value, "target-model source_identity")
    _exact_keys(
        identity,
        (
            "canonical_model_id",
            "model_id",
            "model_revision",
            "official_config_sha256",
            "midlayer_config_sha256",
            "midlayer_baseline",
            "validation_run_sha256",
            "environment",
            "sources",
        ),
        "target-model source_identity",
    )
    canonical_model_id = identity["canonical_model_id"]
    if canonical_model_id not in _TARGET_MODELS:
        raise ValueError("source_identity canonical_model_id is unsupported")
    if identity["model_id"] != _TARGET_MODELS[canonical_model_id]:
        raise ValueError("source_identity model ID disagrees with canonical ID")
    for field in ("model_revision", "midlayer_baseline"):
        _git_commit(identity[field], f"source_identity {field}")
    for field in (
        "official_config_sha256",
        "midlayer_config_sha256",
        "validation_run_sha256",
    ):
        _sha256(identity[field], f"source_identity {field}")
    environment = _mapping(identity["environment"], "source environment")
    _exact_keys(
        environment,
        (
            "device",
            "cuda_visible_devices",
            "python",
            "torch",
            "transformers",
        ),
        "source environment",
    )
    for field in environment:
        _nonempty_string(environment[field], f"source environment {field}")
    sources = _mapping(identity["sources"], "source checkouts")
    _exact_keys(sources, ("model", "transformers", "vllm"), "source checkouts")
    for name in ("model", "transformers", "vllm"):
        source = _mapping(sources[name], f"source checkout {name}")
        _exact_keys(
            source,
            ("commit", "version", "execution_status", "files"),
            f"source checkout {name}",
        )
        _git_commit(source["commit"], f"source checkout {name} commit")
        if source["version"] is not None:
            _nonempty_string(source["version"], f"source checkout {name} version")
        _nonempty_string(
            source["execution_status"],
            f"source checkout {name} execution_status",
        )
        files = _sequence(source["files"], f"source checkout {name} files")
        if not files:
            raise ValueError(f"source checkout {name} requires pinned files")
        for position, item in enumerate(files):
            record = _mapping(item, f"source checkout {name} file {position}")
            _exact_keys(
                record,
                ("relative_path", "sha256"),
                f"source checkout {name} file {position}",
            )
            _nonempty_string(
                record["relative_path"],
                f"source checkout {name} file path",
            )
            _sha256(record["sha256"], f"source checkout {name} file SHA256")
    return identity


def _validate_operator_signatures(
    value: object,
) -> Mapping[str, Mapping[str, object]]:
    table = _mapping(value, "operator signatures")
    if not table:
        raise ValueError("operator signatures cannot be empty")
    normalized = {}
    for identifier, payload in table.items():
        _sha256(identifier, "operator signature ID")
        signature = _mapping(payload, f"operator signature {identifier}")
        _exact_keys(signature, ("operation", "attrs"), "operator signature")
        _nonempty_string(signature["operation"], "operator signature operation")
        _mapping(signature["attrs"], "operator signature attrs")
        if _canonical_sha256(signature) != identifier:
            raise ValueError("operator signature ID disagrees with its content")
        normalized[identifier] = signature
    return normalized


def _validate_sample(
    sample: Mapping[str, object],
    *,
    phase: str,
    model_id: object,
    signatures: Mapping[str, Mapping[str, object]],
) -> str:
    _exact_keys(
        sample,
        (
            "sample_id",
            "phase",
            "scope",
            "actual",
            "midlayer",
            "comparisons",
            "differences",
            "unsupported",
            "provenance",
            "content_sha256",
        ),
        f"alignment phase {phase} sample",
    )
    sample_id = _nonempty_string(sample["sample_id"], "alignment sample_id")
    if sample["phase"] != phase:
        raise ValueError("alignment sample phase disagrees with its container")
    _validate_scope(sample["scope"])
    _validate_actual(sample["actual"], signatures)
    _validate_midlayer(sample["midlayer"], phase, model_id)
    comparisons = _sequence(sample["comparisons"], "alignment comparisons")
    if not comparisons:
        raise ValueError("alignment sample requires semantic comparisons")
    for position, value in enumerate(comparisons):
        comparison = _mapping(value, f"alignment comparison {position}")
        _exact_keys(
            comparison,
            (
                "mechanism",
                "actual_evidence",
                "midlayer_evidence",
                "status",
            ),
            f"alignment comparison {position}",
        )
        for field in ("mechanism", "actual_evidence", "midlayer_evidence"):
            _nonempty_string(
                comparison[field], f"alignment comparison {field}")
        if comparison["status"] != "matched":
            raise ValueError(
                "alignment comparisons must be matched; use differences or "
                "unsupported for other outcomes")
    differences = _sequence(sample["differences"], "alignment differences")
    if differences:
        raise ValueError(
            "publishable target-model candidates cannot contain unresolved "
            "alignment differences")
    unsupported = _sequence(sample["unsupported"], "alignment unsupported")
    for position, value in enumerate(unsupported):
        boundary = _mapping(value, f"unsupported boundary {position}")
        _exact_keys(
            boundary,
            ("capability", "reason"),
            f"unsupported boundary {position}",
        )
        _nonempty_string(boundary["capability"], "unsupported capability")
        _nonempty_string(boundary["reason"], "unsupported reason")
    provenance = _mapping(sample["provenance"], "alignment sample provenance")
    _exact_keys(
        provenance,
        (
            "actual_execution",
            "production_shape_source",
            "midlayer_generation",
        ),
        "alignment sample provenance",
    )
    for field in provenance:
        _nonempty_string(provenance[field], f"alignment provenance {field}")
    expected_sha = _canonical_sha256({
        key: sample[key] for key in sample if key != "content_sha256"
    })
    if sample["content_sha256"] != expected_sha:
        raise ValueError("alignment sample content SHA256 is inconsistent")
    return sample_id


def _validate_scope(value: object) -> None:
    scope = _mapping(value, "alignment sample scope")
    _exact_keys(
        scope,
        (
            "coverage",
            "actual_layers",
            "midlayer_layers",
            "rank_ids",
            "subgraphs",
            "omitted",
        ),
        "alignment sample scope",
    )
    if scope["coverage"] != "partial":
        raise ValueError("target-model execution slices must declare partial coverage")
    for field in ("actual_layers", "midlayer_layers", "rank_ids"):
        values = _sequence(scope[field], f"alignment scope {field}")
        if not values or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in values
        ):
            raise ValueError(
                f"alignment scope {field} must contain non-negative integers")
    for field in ("subgraphs", "omitted"):
        values = _sequence(scope[field], f"alignment scope {field}")
        if not values:
            raise ValueError(f"alignment scope {field} cannot be empty")
        for item in values:
            _nonempty_string(item, f"alignment scope {field} entry")


def _validate_actual(
    value: object,
    signatures: Mapping[str, Mapping[str, object]],
) -> None:
    actual = _mapping(value, "alignment actual slice")
    _exact_keys(
        actual,
        ("framework", "execution_mode", "events", "contracts"),
        "alignment actual slice",
    )
    if actual["framework"] != "transformers":
        raise ValueError("actual target-model execution must use Transformers")
    if actual["execution_mode"] != "a800_cuda_bf16":
        raise ValueError("actual target-model execution mode must be A800 CUDA BF16")
    events = _sequence(actual["events"], "alignment actual events")
    if not events:
        raise ValueError("alignment actual events cannot be empty")
    for order, value in enumerate(events):
        event = _mapping(value, f"alignment actual event {order}")
        _exact_keys(
            event,
            ("order", "signature_id", "layer", "rank", "inputs", "output"),
            f"alignment actual event {order}",
        )
        if event["order"] != order:
            raise ValueError("alignment actual event order must be contiguous")
        if event["signature_id"] not in signatures:
            raise ValueError("alignment actual event references unknown signature")
        for field in ("layer", "rank"):
            if (
                isinstance(event[field], bool)
                or not isinstance(event[field], int)
                or event[field] < 0
            ):
                raise ValueError(
                    f"alignment actual event {field} must be non-negative")
        _json_value(event["inputs"], "alignment actual event inputs")
        _json_value(event["output"], "alignment actual event output")
    contracts = _mapping(actual["contracts"], "alignment actual contracts")
    if not contracts:
        raise ValueError("alignment actual contracts cannot be empty")


def _validate_midlayer(value: object, phase: str, model_id: object) -> None:
    midlayer = _mapping(value, "alignment Midlayer slice")
    _exact_keys(
        midlayer,
        (
            "model_id",
            "phase",
            "layer",
            "rank",
            "schedule",
            "operators",
            "kernel_calls",
        ),
        "alignment Midlayer slice",
    )
    if midlayer["model_id"] != model_id or midlayer["phase"] != phase:
        raise ValueError("Midlayer alignment identity disagrees with sample")
    for field in ("layer", "rank"):
        if (
            isinstance(midlayer[field], bool)
            or not isinstance(midlayer[field], int)
            or midlayer[field] < 0
        ):
            raise ValueError(f"Midlayer alignment {field} must be non-negative")
    _mapping(midlayer["schedule"], "Midlayer alignment schedule")
    operators = _sequence(midlayer["operators"], "Midlayer alignment operators")
    calls = _sequence(midlayer["kernel_calls"], "Midlayer alignment kernel calls")
    if not operators or not calls:
        raise ValueError(
            "Midlayer alignment requires logical operators and KernelSpecs")
    node_ids = set()
    for position, value in enumerate(operators):
        operator = _mapping(value, f"Midlayer operator {position}")
        _exact_keys(
            operator,
            (
                "node_id",
                "kind",
                "pool_name",
                "shape",
                "dtype",
                "variant",
                "rank",
                "attributes",
            ),
            f"Midlayer operator {position}",
        )
        node_ids.add(operator["node_id"])
    if len(node_ids) != len(operators):
        raise ValueError("Midlayer operator IDs must be unique")
    call_ids = set()
    covered_nodes = set()
    for position, value in enumerate(calls):
        call = _mapping(value, f"Midlayer KernelCall {position}")
        _exact_keys(
            call,
            (
                "call_id",
                "pool_name",
                "rank",
                "resource",
                "source_node_ids",
                "dependencies",
                "kernel_spec",
            ),
            f"Midlayer KernelCall {position}",
        )
        call_ids.add(call["call_id"])
        sources = _sequence(
            call["source_node_ids"], f"Midlayer KernelCall {position} sources")
        covered_nodes.update(sources)
        spec = _mapping(
            call["kernel_spec"], f"Midlayer KernelCall {position} KernelSpec")
        _exact_keys(
            spec,
            (
                "kind",
                "hardware_id",
                "shape",
                "dtype",
                "variant",
                "launch",
                "transfer",
                "attributes",
                "execution_support",
                "unsupported_reason",
            ),
            f"Midlayer KernelCall {position} KernelSpec",
        )
        if spec["hardware_id"] != "nvidia_a800":
            raise ValueError("target-model KernelSpec must target nvidia_a800")
        if spec["execution_support"] not in ("supported", "unsupported"):
            raise ValueError("KernelSpec execution_support is invalid")
        if spec["execution_support"] == "supported" and spec["launch"] is None:
            raise ValueError("supported KernelSpec must contain a launch")
        if spec["execution_support"] == "unsupported" and (
            spec["launch"] is not None or spec["unsupported_reason"] is None
        ):
            raise ValueError(
                "unsupported KernelSpec requires null launch and a reason")
    if len(call_ids) != len(calls):
        raise ValueError("Midlayer KernelCall IDs must be unique")
    if covered_nodes != node_ids:
        raise ValueError(
            "Midlayer KernelSpecs must cover every aligned logical operator")


def _reject_duration(value: object, description: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{description} keys must be strings")
            if "duration" in key.lower():
                raise ValueError(f"{description} must not contain duration fields")
            _reject_duration(item, description)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _reject_duration(item, description)






def _exact_keys(
    value: Mapping[str, object],
    expected: Sequence[str],
    description: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(
            f"{description} fields must be exactly {tuple(expected)}; "
            f"got {tuple(value)}")


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def _sha256(value: object, description: str) -> str:
    text = _nonempty_string(value, description)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{description} must be a lowercase SHA256")
    return text


def _git_commit(value: object, description: str) -> str:
    text = _nonempty_string(value, description)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{description} must be a full lowercase Git commit")
    return text


def _json_value(value: object, description: str) -> object:
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{description} must be finite JSON data") from error
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    _json_value(value, "JSON payload")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_object(path: Path, description: str) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must contain valid UTF-8 JSON") from error
    return _mapping(value, description)
