"""Repository-wide compact Serving schema and structural-alignment gate."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import time
from typing import Callable

from evaluation.serving.metrics import sharegpt_compact_evaluation as compact


EXPECTED_CASE_COUNT = 45


def run_compact_alignment_suite(
    repository_root: str | Path,
    load_compact_case: Callable[[Path], object],
    *,
    source_commit: str,
    jobs: int | None = None,
    nvidia_h200_vllm_version: str | None = None,
) -> dict[str, object]:
    """Validate every canonical case and align cases with frozen configs."""

    resolved_jobs = _resolve_jobs(jobs)
    started = time.perf_counter()
    root = Path(repository_root).resolve()
    dataset_root = root / "serving_dataset" / "sharegpt"
    frozen = _frozen_case_mappings(root, nvidia_h200_vllm_version=nvidia_h200_vllm_version)
    expected_case_count = EXPECTED_CASE_COUNT if nvidia_h200_vllm_version is None else len(frozen)
    case_dirs = tuple(sorted(
        manifest.parent for manifest in dataset_root.glob("*/*/*/manifest.json")
    ))
    if len(case_dirs) != expected_case_count:
        raise ValueError(
            "canonical compact alignment suite must contain exactly "
            f"{expected_case_count} cases; found {len(case_dirs)}"
        )
    case_keys = {
        tuple(case_dir.relative_to(dataset_root).parts) for case_dir in case_dirs
    }
    frozen_keys = set(frozen)
    if case_keys != frozen_keys:
        missing = sorted(case_keys - frozen_keys)
        extra = sorted(frozen_keys - case_keys)
        raise ValueError(
            "compact alignment mappings must exactly cover the canonical "
            f"{expected_case_count}-case matrix; missing={missing}, extra={extra}"
        )
    results = _run_cases(
        dataset_root,
        case_dirs,
        frozen,
        load_compact_case,
        resolved_jobs,
    )
    matched_count = _status_count(results, "matched")
    hardware_summaries = _hardware_summaries(results)
    return {
        "schema_version": 1,
        "suite_id": f"sharegpt_compact_alignment_{expected_case_count}",
        "nvidia_h200_vllm_version": nvidia_h200_vllm_version,
        "source_commit": source_commit,
        "jobs": resolved_jobs,
        "wall_time_seconds": time.perf_counter() - started,
        "case_count": len(results),
        "schema_validated_count": len(results),
        "alignment_stage_ran_count": sum(
            result["alignment"]["alignment_stage_ran"] is True
            for result in results
        ),
        "matched_count": matched_count,
        "mismatched_count": _status_count(results, "mismatched"),
        "unsupported_count": _status_count(results, "unsupported"),
        "alignment_checked": matched_count == len(results),
        "formal_conclusion_allowed": matched_count == len(results),
        "hardware_summaries": hardware_summaries,
        "cases": list(results),
    }


def _resolve_jobs(jobs: int | None) -> int:
    if jobs is None:
        return os.cpu_count() or 1
    compact._validate_jobs(jobs)
    return jobs


def _run_cases(
    dataset_root: Path,
    case_dirs: tuple[Path, ...],
    frozen: dict[tuple[str, str, str], tuple],
    load_compact_case: Callable[[Path], object],
    jobs: int,
) -> tuple[dict[str, object], ...]:
    if jobs == 1:
        return tuple(
            _validate_and_align_case(
                dataset_root, case_dir, frozen, load_compact_case
            )
            for case_dir in case_dirs
        )
    result_by_index: dict[int, dict[str, object]] = {}
    executor = ProcessPoolExecutor(max_workers=min(jobs, len(case_dirs)))
    future_to_index = {
        executor.submit(
            _validate_and_align_case,
            dataset_root,
            case_dir,
            frozen,
            load_compact_case,
        ): index
        for index, case_dir in enumerate(case_dirs)
    }
    try:
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_by_index[index] = future.result()
            except Exception as error:
                for pending in future_to_index:
                    pending.cancel()
                relative_case = case_dirs[index].relative_to(dataset_root)
                raise RuntimeError(
                    "compact alignment worker failed for "
                    f"{'/'.join(relative_case.parts)}"
                ) from error
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return tuple(result_by_index[index] for index in range(len(case_dirs)))


def _frozen_case_mappings(root: Path, *, nvidia_h200_vllm_version=None) -> dict[tuple[str, str, str], tuple]:
    frozen = {}
    suite_runs = compact.official_metrics_suite_runs(root, nvidia_h200_vllm_version=nvidia_h200_vllm_version)
    for config, cells in suite_runs:
        for cell in cells:
            key = (
                cell.hardware.hardware_id,
                cell.model.dataset_model_id.lower(),
                cell.case_id,
            )
            if key in frozen:
                raise AssertionError(f"duplicate frozen compact case mapping: {key}")
            frozen[key] = (config, cell)
    return frozen


def _validate_and_align_case(dataset_root, case_dir, frozen, load_compact_case):
    hardware_id, model_id, case_id = case_dir.relative_to(dataset_root).parts
    if model_id != model_id.lower():
        raise ValueError(
            f"compact dataset model directory must be lowercase: {model_id!r}"
        )
    case = load_compact_case(case_dir)
    identity = case.manifest["identity"]
    key = (hardware_id, model_id, case_id)
    actual_key = (
        identity["hardware_id"], identity["model_id"], identity["case_id"]
    )
    if actual_key != key:
        raise ValueError(
            "compact case path disagrees with manifest identity: "
            f"path={key}, manifest={actual_key}"
        )
    if key not in frozen:
        raise ValueError(f"compact case has no frozen alignment mapping: {key}")
    config, cell = frozen[key]
    result = compact.validate_compact_alignment(
        dataset_root.parents[1], config, cell, case
    )
    return {
        "cell_id": "/".join(key),
        "schema_validated": True,
        "alignment": result,
    }

def _status_count(results, status):
    return sum(result["alignment"]["status"] == status for result in results)


def _hardware_summaries(
    results: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    hardware_ids = sorted({
        result["cell_id"].split("/", maxsplit=1)[0] for result in results
    })
    summaries = []
    for hardware_id in hardware_ids:
        hardware_results = tuple(
            result
            for result in results
            if result["cell_id"].split("/", maxsplit=1)[0] == hardware_id
        )
        summaries.append({
            "hardware_id": hardware_id,
            "case_count": len(hardware_results),
            "schema_validated_count": sum(
                result["schema_validated"] is True for result in hardware_results
            ),
            "alignment_stage_ran_count": sum(
                result["alignment"]["alignment_stage_ran"] is True
                for result in hardware_results
            ),
            "matched_count": _status_count(hardware_results, "matched"),
            "mismatched_count": _status_count(hardware_results, "mismatched"),
            "unsupported_count": _status_count(hardware_results, "unsupported"),
        })
    return summaries
