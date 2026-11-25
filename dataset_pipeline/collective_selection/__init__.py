"""Collective launch-selection observation parsers.

The three backends (NCCL, MCCL, HCCL) each parse a profiler's collective
records into the same observation shape.  The key sets that shape is validated
against, the validation/hashing helpers each parser needs, and the run-metadata
spine their ``build_selection_observations`` share are declared here, once:
all three must agree on them, and a per-backend copy could drift silently.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


DEVICE_KEYS = {"device_name", "hostname", "physical_device"}

OBSERVATION_KEYS = {
    "dtype",
    "in_place",
    "message_bytes",
    "operation",
    "participant_count",
    "rank",
    "reduction",
    "repeat",
    "validation",
}


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON array")
    return value


def require_exact_keys(
        value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def require_keys(value: dict[str, Any], required: set[str], context: str) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")


def load_collective_json(path: Path) -> Any:
    """Read one backend's JSON artifact, reporting a read/parse failure as one
    ``ValueError`` so the three parsers surface the same error shape."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error


def metadata_observation_key(observation: dict[str, Any]) -> tuple[Any, ...]:
    """The fields that must agree across ranks for one metadata observation."""

    return (
        observation["operation"],
        observation["message_bytes"],
        observation["participant_count"],
        observation["dtype"],
        observation["reduction"],
        observation["in_place"],
        observation["repeat"],
        observation["validation"],
    )


def metadata_files(
        run_dir: Path, allowed_world_sizes: tuple[int, ...],
        allowed_description: str) -> tuple[Path, ...]:
    """Locate one run's contiguous ``rank-*.json`` metadata files.

    ``allowed_description`` is the English rendering of
    ``allowed_world_sizes`` used in the count error, so each backend keeps the
    exact wording it has always raised.
    """

    paths = tuple(sorted(run_dir.glob("rank-*.json")))
    if len(paths) not in allowed_world_sizes:
        raise ValueError(
            f"run directory must contain {allowed_description} rank files, "
            f"got {paths}")
    expected = tuple(run_dir / f"rank-{rank}.json" for rank in range(len(paths)))
    if paths != expected:
        raise ValueError(
            f"run directory rank files must be contiguous, got {paths}")
    return paths


def validate_metadata_agreement(
        metadata: tuple[dict[str, Any], ...], keys: set[str],
        differing_fields: set[str]) -> None:
    """Require per-rank metadata to agree outside its rank-local fields."""

    for rank, payload in enumerate(metadata[1:], start=1):
        for key in keys - differing_fields:
            if metadata[0][key] != payload[key]:
                raise ValueError(f"rank {rank} metadata differs for {key}")
    observations = []
    for rank, payload in enumerate(metadata):
        rank_observations = require_list(
            payload["observations"], f"rank {rank} observations")
        observations.append(tuple(
            metadata_observation_key(
                require_mapping(value, f"rank {rank} observation"))
            for value in rank_observations
        ))
    for rank, rank_observations in enumerate(observations[1:], start=1):
        if observations[0] != rank_observations:
            raise ValueError(f"rank {rank} metadata observations differ")


def load_run_metadata(
        run_dir: Path, *, allowed_world_sizes: tuple[int, ...],
        allowed_description: str, validate_metadata: Any, keys: set[str],
        differing_fields: set[str],
        ) -> tuple[Path, tuple[Path, ...], tuple[dict[str, Any], ...]]:
    """Resolve, list, validate, and cross-check one run's rank metadata."""

    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    paths = metadata_files(run_dir, allowed_world_sizes, allowed_description)
    world_size = len(paths)
    metadata = tuple(
        validate_metadata(path, rank, world_size)
        for rank, path in enumerate(paths)
    )
    validate_metadata_agreement(metadata, keys, differing_fields)
    return run_dir, paths, metadata


def rank_zero_observations(
        metadata: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Return the shared observation contract stated by rank 0."""

    return tuple(
        require_mapping(value, f"rank 0 observations[{index}]")
        for index, value in enumerate(metadata[0]["observations"])
    )


def require_consistent_ranks(
        items_by_rank: Any, signature: Any, error_label: str) -> None:
    """Require every rank's signature sequence to equal rank 0's.

    ``error_label`` is the full clause preceding ``across ranks 0 and`` in the
    error each backend has always raised.
    """

    reference = tuple(signature(item) for item in items_by_rank[0])
    for rank, items in enumerate(items_by_rank[1:], start=1):
        if reference != tuple(signature(item) for item in items):
            raise ValueError(f"{error_label} across ranks 0 and {rank}")


def require_consistent_repeats(
        normalized: Any, signature: Any, error_label: str) -> None:
    """Require one selection signature per (operation, message_bytes)."""

    selections: dict[tuple[str, int], set[tuple[Any, ...]]] = defaultdict(set)
    for observation in normalized:
        selections[(observation["operation"], observation["message_bytes"])].add(
            signature(observation)
        )
    inconsistent = {
        key: value for key, value in selections.items() if len(value) != 1}
    if inconsistent:
        raise ValueError(f"{error_label}: {inconsistent}")


def require_integer(value: Any, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_observations(run_dir: Path, payload: dict[str, Any]) -> Path:
    """Write one parser's selection observations next to its run directory.

    The three parsers differ in how they build the payload and nowhere else,
    so the destination, the refuse-to-overwrite check and the atomic replace
    are shared.  ``selection_observations.json`` is the published name.
    """

    output = run_dir.resolve() / "selection_observations.json"
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    return output
