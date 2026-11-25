"""Strict loading of canonical model-level expert-frequency profiles."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Tuple

import numpy as np


_PROFILE_FIELDS = {
    "canonical_model_id",
    "model_ids",
    "filename",
    "source_path",
    "source_repository_commit",
    "sha256",
    "shape",
    "dtype",
    "dimension_semantics",
    "value_semantics",
    "source_workload",
    "provenance_notes",
}
_SOURCE_WORKLOAD_FIELDS = {"status", "description"}
_KEBAB_NPY = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\.npy")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate expert-frequency manifest key: {key!r}")
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    value = _require_string(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _require_git_commit(value: object, label: str) -> str:
    value = _require_string(value, label)
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a 40-character lowercase Git commit")
    return value


def _require_string_tuple(value: object, label: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty list")
    result = tuple(_require_string(item, f"{label} entry") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} entries must be unique")
    return result


@dataclass(frozen=True)
class ExpertFrequencyProfile:
    canonical_model_id: str
    model_ids: Tuple[str, ...]
    filename: str
    source_path: str
    source_repository_commit: str
    sha256: str
    shape: Tuple[int, int]
    dtype: str
    dimension_semantics: Tuple[str, str]
    value_semantics: str
    source_workload: Mapping[str, str]
    provenance_notes: Tuple[str, ...]
    mean_probabilities: Tuple[float, ...]
    manifest_sha256: str
    # Row i is the i-th MoE layer in model execution order.
    layer_probabilities: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.canonical_model_id not in self.model_ids:
            raise ValueError(
                "expert-frequency canonical_model_id must appear in model_ids")
        if len(self.mean_probabilities) != self.shape[1]:
            raise ValueError(
                "expert-frequency mean probability width disagrees with shape")
        if any(not math.isfinite(value) or value < 0 for value in self.mean_probabilities):
            raise ValueError(
                "expert-frequency mean probabilities must be finite and non-negative")
        if not math.isclose(sum(self.mean_probabilities), 1.0, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError("expert-frequency mean probabilities must sum to one")
        if len(self.layer_probabilities) != self.shape[0] or any(
                len(row) != self.shape[1] for row in self.layer_probabilities):
            raise ValueError(
                "expert-frequency layer probabilities disagree with shape")
        object.__setattr__(self, "source_workload", MappingProxyType(dict(self.source_workload)))

    def manifest(self) -> dict[str, object]:
        return {
            "canonical_model_id": self.canonical_model_id,
            "model_ids": self.model_ids,
            "filename": self.filename,
            "source_path": self.source_path,
            "source_repository_commit": self.source_repository_commit,
            "sha256": self.sha256,
            "shape": self.shape,
            "dtype": self.dtype,
            "dimension_semantics": self.dimension_semantics,
            "value_semantics": self.value_semantics,
            "source_workload": dict(self.source_workload),
            "provenance_notes": self.provenance_notes,
            "manifest_sha256": self.manifest_sha256,
        }


def load_expert_frequency_profile(
    manifest_path: str | Path,
    model_id: str,
    *,
    expected_model_id: str,
    expected_moe_layers: int,
    expected_num_experts: int,
) -> ExpertFrequencyProfile:
    """Resolve one exact manifest model ID and validate its unchanged NPY bytes."""

    manifest_path = Path(manifest_path)
    model_id = _require_string(model_id, "expert-frequency model_id")
    expected_model_id = _require_string(
        expected_model_id, "expert-frequency expected_model_id")
    for label, value in (
            ("expected_moe_layers", expected_moe_layers),
            ("expected_num_experts", expected_num_experts)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    raw = json.loads(
        manifest_bytes,
        object_pairs_hook=_unique_json_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant in expert-frequency manifest: {value}")),
    )
    if not isinstance(raw, dict):
        raise TypeError("expert-frequency manifest root must be an object")
    if set(raw) != {"schema_version", "profiles"}:
        raise ValueError(
            "expert-frequency manifest fields must be exactly schema_version and profiles")
    if raw["schema_version"] != 1:
        raise ValueError("expert-frequency manifest schema_version must be 1")
    profiles = raw["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise TypeError("expert-frequency profiles must be a non-empty list")

    entries = []
    mapped_model_ids = set()
    filenames = set()
    for index, entry in enumerate(profiles):
        label = f"expert-frequency profiles[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{label} must be an object")
        if set(entry) != _PROFILE_FIELDS:
            raise ValueError(f"{label} fields must be exactly {sorted(_PROFILE_FIELDS)}")
        canonical_model_id = _require_string(
            entry["canonical_model_id"], f"{label}.canonical_model_id")
        model_ids = _require_string_tuple(entry["model_ids"], f"{label}.model_ids")
        if canonical_model_id not in model_ids:
            raise ValueError(f"{label}.canonical_model_id must appear in model_ids")
        duplicate_model_ids = mapped_model_ids.intersection(model_ids)
        if duplicate_model_ids:
            raise ValueError(
                "expert-frequency model IDs must map exactly once: "
                f"{sorted(duplicate_model_ids)}")
        mapped_model_ids.update(model_ids)
        filename = _require_string(entry["filename"], f"{label}.filename")
        if _KEBAB_NPY.fullmatch(filename) is None or Path(filename).name != filename:
            raise ValueError(f"{label}.filename must be a lowercase kebab-case .npy name")
        if filename in filenames:
            raise ValueError(f"duplicate expert-frequency filename: {filename!r}")
        filenames.add(filename)
        source_path = _require_string(entry["source_path"], f"{label}.source_path")
        if not Path(source_path).is_absolute():
            raise ValueError(f"{label}.source_path must be absolute")
        source_repository_commit = _require_git_commit(
            entry["source_repository_commit"],
            f"{label}.source_repository_commit",
        )
        sha256 = _require_sha256(entry["sha256"], f"{label}.sha256")
        shape_raw = entry["shape"]
        if (not isinstance(shape_raw, list) or len(shape_raw) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                       for item in shape_raw)):
            raise ValueError(f"{label}.shape must contain two positive integers")
        shape = (shape_raw[0], shape_raw[1])
        dtype = _require_string(entry["dtype"], f"{label}.dtype")
        try:
            normalized_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"{label}.dtype is not a NumPy dtype: {dtype!r}") from exc
        if normalized_dtype.kind != "f":
            raise ValueError(f"{label}.dtype must be floating point")
        dimension_semantics = _require_string_tuple(
            entry["dimension_semantics"], f"{label}.dimension_semantics")
        if len(dimension_semantics) != 2:
            raise ValueError(f"{label}.dimension_semantics must describe two axes")
        value_semantics = _require_string(
            entry["value_semantics"], f"{label}.value_semantics")
        source_workload = entry["source_workload"]
        if not isinstance(source_workload, dict):
            raise TypeError(f"{label}.source_workload must be an object")
        if set(source_workload) != _SOURCE_WORKLOAD_FIELDS:
            raise ValueError(
                f"{label}.source_workload fields must be exactly "
                f"{sorted(_SOURCE_WORKLOAD_FIELDS)}")
        normalized_source_workload = {
            field: _require_string(
                source_workload[field], f"{label}.source_workload.{field}")
            for field in sorted(_SOURCE_WORKLOAD_FIELDS)
        }
        provenance_notes_raw = entry["provenance_notes"]
        if not isinstance(provenance_notes_raw, list):
            raise TypeError(f"{label}.provenance_notes must be a list")
        provenance_notes = tuple(
            _require_string(note, f"{label}.provenance_notes entry")
            for note in provenance_notes_raw
        )
        entries.append({
            "canonical_model_id": canonical_model_id,
            "model_ids": model_ids,
            "filename": filename,
            "source_path": source_path,
            "source_repository_commit": source_repository_commit,
            "sha256": sha256,
            "shape": shape,
            "dtype": normalized_dtype,
            "dimension_semantics": dimension_semantics,
            "value_semantics": value_semantics,
            "source_workload": normalized_source_workload,
            "provenance_notes": provenance_notes,
        })

    matches = [entry for entry in entries if model_id in entry["model_ids"]]
    if not matches:
        raise KeyError(
            f"expert-frequency manifest has no explicit model ID mapping for {model_id!r}")
    if len(matches) != 1:
        raise RuntimeError("expert-frequency model ID mapping is ambiguous")
    entry = matches[0]
    if expected_model_id not in entry["model_ids"]:
        raise ValueError(
            "expert-frequency profile is not mapped to the executing model: "
            f"selected={model_id!r}, executing={expected_model_id!r}")
    if entry["shape"] != (expected_moe_layers, expected_num_experts):
        raise ValueError(
            "expert-frequency manifest shape disagrees with model: "
            f"manifest={entry['shape']}, expected={(expected_moe_layers, expected_num_experts)}")
    profile_path = manifest_path.parent / entry["filename"]
    actual_sha256 = _sha256_file(profile_path)
    if actual_sha256 != entry["sha256"]:
        raise ValueError(
            "expert-frequency profile SHA256 mismatch: "
            f"expected={entry['sha256']}, actual={actual_sha256}")
    values = np.load(profile_path, allow_pickle=False)
    if not isinstance(values, np.ndarray):
        raise TypeError("expert-frequency profile must contain a NumPy array")
    if values.ndim != 2:
        raise ValueError("expert-frequency profile must be two-dimensional")
    if values.shape != entry["shape"]:
        raise ValueError(
            f"expert-frequency array shape mismatch: {values.shape} != {entry['shape']}")
    if values.dtype != entry["dtype"]:
        raise ValueError(
            f"expert-frequency array dtype mismatch: {values.dtype} != {entry['dtype']}")
    if not np.isfinite(values).all():
        raise ValueError("expert-frequency profile values must be finite")
    if (values < 0).any():
        raise ValueError("expert-frequency profile values must be non-negative")
    row_sums = values.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("every expert-frequency profile row must sum to one")
    mean_probabilities = tuple(
        float(value) for value in values.mean(axis=0, dtype=np.float64))
    return ExpertFrequencyProfile(
        canonical_model_id=entry["canonical_model_id"],
        model_ids=entry["model_ids"],
        filename=entry["filename"],
        source_path=entry["source_path"],
        source_repository_commit=entry["source_repository_commit"],
        sha256=entry["sha256"],
        shape=entry["shape"],
        dtype=entry["dtype"].name,
        dimension_semantics=entry["dimension_semantics"],
        value_semantics=entry["value_semantics"],
        source_workload=entry["source_workload"],
        provenance_notes=entry["provenance_notes"],
        mean_probabilities=mean_probabilities,
        manifest_sha256=manifest_sha256,
        layer_probabilities=tuple(
            tuple(float(value) for value in row) for row in values),
    )
