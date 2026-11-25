"""Strict publication of complete Scheduler capture candidates."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Sequence

from .compact_case import (
    ALIGNMENT_FILENAME,
    LATEST_SCHEMA_VERSION,
    OBSERVER_QUALITY_SCHEMA_VERSION,
    SCHEDULER_STEPS_FILENAME,
    load_compact_case,
)


_CASE_FILES = frozenset({
    ALIGNMENT_FILENAME,
    "manifest.json",
    "metrics.json",
    "requests.csv",
    SCHEDULER_STEPS_FILENAME,
})
_REQUIRED_RAW_SOURCE_IDS = frozenset({
    "accepted-ground-truth-v3",
    "accepted-scheduler-trace-v3",
})


def publish_scheduler_capture_candidates(
    *,
    candidate_root: str | Path,
    dataset_root: str | Path,
    cases: Sequence[str],
    new_case_reference_hardware: str | None = None,
) -> tuple[Path, ...]:
    """Atomically publish audited canonical cases from complete candidates.

    An existing canonical case is replaced only by a candidate with the same
    request set.  A case with no canonical predecessor (a newly captured
    hardware) is published only when ``new_case_reference_hardware`` names a
    hardware whose canonical case of the same model and case ID holds the
    same request set, so a new hardware never introduces a new workload.
    """

    candidates = Path(candidate_root).resolve(strict=True)
    canonical = Path(dataset_root).resolve(strict=True)
    if candidates.name != "serving_dataset_compaction" or candidates.parent.name != "profiler_output":
        raise ValueError(
            "candidate root must be exactly "
            "<repository>/profiler_output/serving_dataset_compaction"
        )
    if not canonical.is_dir():
        raise ValueError(f"Serving dataset root is not a directory: {canonical}")
    if canonical == candidates or canonical in candidates.parents or candidates in canonical.parents:
        raise ValueError("candidate and canonical dataset roots must not overlap")
    relative_cases = _relative_cases(cases)
    staged = []
    for relative_case in relative_cases:
        candidate_case = candidates / relative_case
        canonical_case = canonical / relative_case
        if canonical_case.exists() or canonical_case.is_symlink():
            _validate_case_pair(candidate_case, canonical_case, relative_case)
        else:
            if new_case_reference_hardware is None:
                raise ValueError(
                    f"no canonical case to replace and no reference hardware "
                    f"for a new case: {relative_case}")
            reference_case = (canonical / new_case_reference_hardware
                              / relative_case.parts[1] / relative_case.parts[2])
            if reference_case == canonical_case:
                raise ValueError("reference hardware must differ from the "
                                 f"published hardware: {relative_case}")
            _validate_case_pair(candidate_case, reference_case, relative_case,
                                reference_hardware=new_case_reference_hardware)
            canonical_case.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{relative_case.name}.publish-", dir=canonical_case.parent
        ))
        try:
            for filename in sorted(_CASE_FILES):
                shutil.copyfile(candidate_case / filename, temporary / filename)
            load_compact_case(temporary)
        except BaseException:
            shutil.rmtree(temporary)
            raise
        staged.append((relative_case, canonical_case, temporary))

    published = []
    for relative_case, canonical_case, temporary in staged:
        if not canonical_case.exists():
            try:
                os.replace(temporary, canonical_case)
                load_compact_case(canonical_case)
            except BaseException:
                if canonical_case.exists():
                    shutil.rmtree(canonical_case)
                raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            published.append(canonical_case)
            continue
        backup = canonical_case.with_name(
            f".{canonical_case.name}.prepublish-backup"
        )
        if backup.exists() or backup.is_symlink():
            raise FileExistsError(f"publication backup path already exists: {backup}")
        try:
            os.replace(canonical_case, backup)
            try:
                os.replace(temporary, canonical_case)
            except BaseException:
                os.replace(backup, canonical_case)
                raise
            try:
                load_compact_case(canonical_case)
            except BaseException:
                os.replace(canonical_case, temporary)
                os.replace(backup, canonical_case)
                raise
            shutil.rmtree(backup)
            published.append(canonical_case)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            if backup.exists():
                shutil.rmtree(backup)
    return tuple(published)


def _relative_cases(cases: Sequence[str]) -> tuple[Path, ...]:
    if not cases:
        raise ValueError("at least one case is required for publication")
    normalized = []
    for value in cases:
        if not isinstance(value, str) or not value:
            raise TypeError("publication case must be a non-empty string")
        pure = PurePosixPath(value)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 3:
            raise ValueError(
                "publication case must be hardware_id/model_id/case_id"
            )
        relative = Path(*pure.parts)
        if relative.as_posix() != value:
            raise ValueError("publication case must be a normalized POSIX path")
        normalized.append(relative)
    if len(set(normalized)) != len(normalized):
        raise ValueError("publication cases must be unique")
    return tuple(sorted(normalized))


def _validate_case_pair(
    candidate_case: Path, canonical_case: Path, relative_case: Path,
    reference_hardware: str | None = None,
) -> None:
    if not candidate_case.is_dir() or candidate_case.is_symlink():
        raise ValueError(f"candidate case is not a directory: {candidate_case}")
    if not canonical_case.is_dir() or canonical_case.is_symlink():
        raise ValueError(f"canonical case is not a directory: {canonical_case}")
    if _case_files(candidate_case) != _CASE_FILES:
        raise ValueError(f"candidate case file set is invalid: {candidate_case}")
    if _case_files(canonical_case) != _CASE_FILES:
        raise ValueError(f"canonical case file set is invalid: {canonical_case}")
    candidate = load_compact_case(candidate_case)
    canonical = load_compact_case(canonical_case)
    candidate_identity = candidate.manifest["identity"]
    canonical_identity = canonical.manifest["identity"]
    expected_identity = tuple(relative_case.parts)
    canonical_expected_identity = (
        expected_identity if reference_hardware is None
        else (reference_hardware, *expected_identity[1:]))
    candidate_identity_tuple = tuple(
        candidate_identity[key] for key in ("hardware_id", "model_id", "case_id")
    )
    canonical_identity_tuple = tuple(
        canonical_identity[key] for key in ("hardware_id", "model_id", "case_id")
    )
    if candidate_identity_tuple != expected_identity:
        raise ValueError(f"candidate identity disagrees with path: {relative_case}")
    if canonical_identity_tuple != canonical_expected_identity:
        raise ValueError(f"canonical identity disagrees with path: {relative_case}")
    if candidate.manifest["schema_version"] not in (
            LATEST_SCHEMA_VERSION, OBSERVER_QUALITY_SCHEMA_VERSION):
        raise ValueError(f"candidate compact schema is not publishable: {relative_case}")
    if candidate.requests != canonical.requests:
        raise ValueError(f"candidate requests differ from canonical case: {relative_case}")
    capture = candidate.manifest["capture"]
    if capture["metrics_source_archive_id"] != "accepted-ground-truth-v3":
        raise ValueError(f"candidate metrics source is invalid: {relative_case}")
    if capture["scheduler_source_archive_id"] != "accepted-scheduler-trace-v3":
        raise ValueError(f"candidate scheduler source is invalid: {relative_case}")
    raw_source_ids = {
        source["archive_id"] for source in candidate.manifest["raw_sources"]
    }
    if not _REQUIRED_RAW_SOURCE_IDS <= raw_source_ids:
        raise ValueError(f"candidate raw-source coverage is incomplete: {relative_case}")


def _case_files(root: Path) -> frozenset[str]:
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"case payload cannot contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"case payload has unsupported file type: {path}")
        files.add(path.relative_to(root).as_posix())
    return frozenset(files)
