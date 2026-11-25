"""Suite-level drivers and report writers for the official compact
ShareGPT metrics evaluation."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

from evaluation.serving.metrics.sharegpt_evaluation_report import (
    STATISTICS,
    write_sharegpt_evaluation_report,
)

from evaluation.json_shape import require_object_value as _mapping

from evaluation.serving.metrics.compact_shape_validators import (
    _sequence,
    _string,
    _finite_number,
)
from evaluation.serving.metrics.compact_case_contract import (
    _GIT_COMMIT,
    ValidatedCompactCase,
    Phase1Config,
    Phase1Cell,
    official_metrics_suite_runs,
    _official_metrics_case_keys,
    _canonical_cell_id,
    compact_case_path,
)
from evaluation.serving.metrics.compact_run_environment import (
    create_run_directory,
)
from evaluation.serving.metrics.compact_metric_comparison import (
    _LATENCY_FIELDS,
    _THROUGHPUT_FIELDS,
)
from evaluation.serving.metrics.compact_case_evaluation import (
    _validated_precomputed_alignment,
    evaluate_validated_case,
    evaluate_validated_case_three_modes,
    _validate_execution_mode,
)


def run_official_metrics_evaluation(
    repository_root: str | Path,
    run_id: str,
    source_commit: str,
    *,
    jobs: int,
    metrics_only: bool,
    compact_alignment_summary: Mapping[str, object] | None,
    isolated_case_worker: Callable[
        [
            str,
            Phase1Config,
            Phase1Cell,
            str,
            bool | Mapping[str, object],
            str,
        ],
        dict[str, object],
    ],
    execution_mode: str = "accelerated",
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run 45 metrics cells after one shared 45-case alignment gate."""

    _validate_jobs(jobs)
    alignment_by_cell = _official_alignment_inputs(
        compact_alignment_summary,
        source_commit=source_commit,
        metrics_only=metrics_only,
    )
    _validate_execution_mode(execution_mode)
    if not isinstance(debug_analytical_launch, bool):
        raise TypeError("debug_analytical_launch must be boolean")
    metrics_started = time.perf_counter()
    suite_runs = official_metrics_suite_runs(repository_root)
    if host_launch_profile_paths is not None:
        required_hardware = {
            cell.hardware.hardware_id
            for _config, cells in suite_runs
            for cell in cells
        }
        unexpected_hardware = set(host_launch_profile_paths) - required_hardware
        if unexpected_hardware:
            raise ValueError(
                "three-mode official evaluation profiles contain unselected "
                f"hardware: {sorted(unexpected_hardware)}")
    if scheduler_host_profile_paths is not None:
        required_hardware = {
            cell.hardware.hardware_id
            for _config, cells in suite_runs
            for cell in cells
        }
        unexpected_scheduler = (
            set(scheduler_host_profile_paths) - required_hardware)
        if unexpected_scheduler:
            raise ValueError(
                "official evaluation Scheduler Host profiles contain "
                f"unselected hardware: {sorted(unexpected_scheduler)}")
    prepared_runs = []
    parallel_cases = []
    for config, cells in suite_runs:
        output_dir, alignment_inputs = _prepare_phase1_evaluation(
            repository_root,
            config,
            cells,
            f"{run_id}--{config.suite_id}",
            metrics_only=metrics_only,
            precomputed_alignment=(
                None
                if alignment_by_cell is None
                else {
                    _canonical_cell_id(cell): alignment_by_cell[
                        _canonical_cell_id(cell)
                    ]
                    for cell in cells
                }
            ),
        )
        start_index = len(parallel_cases)
        parallel_cases.extend(
            (config, cell, alignment_input)
            for cell, alignment_input in zip(
                cells, alignment_inputs, strict=True
            )
        )
        prepared_runs.append((
            config,
            cells,
            output_dir,
            start_index,
            len(parallel_cases),
        ))
    all_results, completed_at = _run_parallel_case_jobs(
        repository_root,
        parallel_cases,
        source_commit,
        jobs,
        isolated_case_worker,
        execution_mode=execution_mode,
        debug_analytical_launch=debug_analytical_launch,
        host_launch_profile_paths=host_launch_profile_paths,
        scheduler_host_profile_paths=scheduler_host_profile_paths,
    )
    metrics_wall_time_seconds = time.perf_counter() - metrics_started
    summaries = []
    for config, _cells, output_dir, start_index, stop_index in prepared_runs:
        results = all_results[start_index:stop_index]
        suite_wall_time_seconds = (
            max(completed_at[start_index:stop_index]) - metrics_started
        )
        summaries.append(_persist_phase1_evaluation(
            repository_root,
            config,
            source_commit,
            output_dir,
            results,
            jobs=jobs,
            execution_mode=execution_mode,
            evaluation_wall_time_seconds=suite_wall_time_seconds,
            debug_num_kv_blocks=None,
            debug_analytical_launch=debug_analytical_launch,
        ))
    root = Path(repository_root).resolve()
    report_output_dir = create_run_directory(root, run_id)
    report_cells = []
    for summary in summaries:
        result_path = root / summary["artifact_paths"]["json"]
        result = json.loads(result_path.read_text())
        report_cells.extend(_sequence(result["cells"], "suite result cells"))
    report_paths = write_sharegpt_evaluation_report(
        report_output_dir,
        report_cells,
        run_elapsed_wall_time_seconds=metrics_wall_time_seconds,
    )
    if compact_alignment_summary is None:
        alignment_report = {
            "suite_id": "sharegpt_compact_alignment_45",
            "status": "skipped_metrics_only_debug",
            "wall_time_seconds": 0.0,
            "alignment_stage_ran_count": 0,
            "matched_count": 0,
            "mismatched_count": 0,
            "unsupported_count": 0,
            "alignment_checked": False,
            "formal_conclusion_allowed": False,
        }
    else:
        alignment_report = dict(compact_alignment_summary)
    alignment_stage_ran = (
        alignment_report["alignment_stage_ran_count"] == 45
    )
    alignment_checked = alignment_report["alignment_checked"] is True
    formal_conclusion_allowed = (
        alignment_report["formal_conclusion_allowed"] is True
        and all(summary["formal_conclusion_allowed"] is True for summary in summaries)
    )
    alignment_wall_time_seconds = alignment_report["wall_time_seconds"]
    return {
        "schema_version": 1,
        "suite_id": "sharegpt_official_metrics_45",
        "metrics_only_debug": metrics_only,
        "execution_mode": execution_mode,
        "evaluation_wall_time_seconds": (
            alignment_wall_time_seconds + metrics_wall_time_seconds
        ),
        "alignment_wall_time_seconds": alignment_wall_time_seconds,
        "metrics_evaluation_wall_time_seconds": metrics_wall_time_seconds,
        "cell_count": sum(summary["cell_count"] for summary in summaries),
        "completed_count": sum(summary["completed_count"] for summary in summaries),
        "unsupported_count": sum(summary["unsupported_count"] for summary in summaries),
        "alignment_stage_ran": alignment_stage_ran,
        "alignment_checked": alignment_checked,
        "formal_conclusion_allowed": formal_conclusion_allowed,
        "dataset_validation_and_alignment": alignment_report,
        "suite_runs": summaries,
        "run_directory": str(report_output_dir.relative_to(root)),
        "report_artifact_paths": {
            name: str(path.relative_to(root))
            for name, path in report_paths.items()
        },
    }


def _official_alignment_inputs(
    summary_value: Mapping[str, object] | None,
    *,
    source_commit: str,
    metrics_only: bool,
) -> Mapping[str, Mapping[str, object]] | None:
    if metrics_only:
        if summary_value is not None:
            raise ValueError(
                "metrics-only debug evaluation cannot consume alignment results"
            )
        return None
    if summary_value is None:
        raise ValueError(
            "official metrics evaluation requires the complete 45-case alignment report"
        )
    summary = _mapping(summary_value, "compact alignment summary")
    if summary["suite_id"] != "sharegpt_compact_alignment_45":
        raise ValueError("compact alignment summary suite_id is invalid")
    if summary["source_commit"] != source_commit:
        raise ValueError("compact alignment summary source_commit mismatch")
    if summary["case_count"] != 45 or summary["schema_validated_count"] != 45:
        raise ValueError("compact alignment summary must validate exactly 45 cases")
    cases = _sequence(summary["cases"], "compact alignment summary cases")
    if len(cases) != 45:
        raise ValueError("compact alignment summary must contain exactly 45 cases")
    expected = _official_metrics_case_keys()
    alignment_by_cell = {}
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"compact alignment case {index}")
        cell_id = _string(case["cell_id"], "compact alignment cell_id")
        if cell_id in alignment_by_cell:
            raise ValueError(f"duplicate compact alignment result: {cell_id}")
        alignment_by_cell[cell_id] = _validated_precomputed_alignment(
            case["alignment"]
        )
    if set(alignment_by_cell) != {"/".join(key) for key in expected}:
        raise ValueError(
            "compact alignment report does not cover the exact 45-case matrix"
        )
    status_counts = {
        status: sum(
            alignment["status"] == status
            for alignment in alignment_by_cell.values()
        )
        for status in ("matched", "mismatched", "unsupported")
    }
    summary_expectations = {
        "alignment_stage_ran_count": 45,
        "matched_count": status_counts["matched"],
        "mismatched_count": status_counts["mismatched"],
        "unsupported_count": status_counts["unsupported"],
        "alignment_checked": status_counts["matched"] == 45,
        "formal_conclusion_allowed": status_counts["matched"] == 45,
    }
    for field, expected_value in summary_expectations.items():
        if summary[field] != expected_value:
            raise ValueError(
                f"compact alignment summary {field} is inconsistent"
            )
    return {
        cell_id: alignment_by_cell[cell_id]
        for cell_id in sorted(alignment_by_cell)
        if tuple(cell_id.split("/", maxsplit=2)) in _official_metrics_case_keys()
    }


def _case_alignment_inputs(
    cells: Sequence[Phase1Cell],
    *,
    metrics_only: bool,
    precomputed_alignment: Mapping[str, Mapping[str, object]] | None,
) -> tuple[bool | Mapping[str, object], ...]:
    if metrics_only:
        if precomputed_alignment is not None:
            raise ValueError(
                "metrics-only debug evaluation cannot consume alignment results"
            )
        return tuple(False for _ in cells)
    if precomputed_alignment is None:
        return tuple(True for _ in cells)
    expected_ids = tuple(_canonical_cell_id(cell) for cell in cells)
    if set(precomputed_alignment) != set(expected_ids):
        raise ValueError(
            "precomputed alignment cells must exactly match selected metric cells"
        )
    return tuple(
        _validated_precomputed_alignment(precomputed_alignment[cell_id])
        for cell_id in expected_ids
    )


def run_phase1_evaluation(
    repository_root: str | Path,
    config: Phase1Config,
    cells: Sequence[Phase1Cell],
    run_id: str,
    source_commit: str,
    load_validated_case: Callable[[Path], ValidatedCompactCase] | None = None,
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]] | None = None,
    *,
    jobs: int = 1,
    metrics_only: bool = False,
    execution_mode: str = "accelerated",
    isolated_case_worker: Callable[
        [
            str,
            Phase1Config,
            Phase1Cell,
            str,
            bool | Mapping[str, object],
            str,
        ],
        dict[str, object],
    ] | None = None,
    precomputed_alignment: Mapping[str, Mapping[str, object]] | None = None,
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Validate inputs, isolate case execution, and persist in parent order."""

    _validate_jobs(jobs)
    _validate_execution_mode(execution_mode)
    if debug_num_kv_blocks is not None:
        if type(debug_num_kv_blocks) is not int or debug_num_kv_blocks < 1:
            raise ValueError("debug_num_kv_blocks must be a positive integer")
        if not metrics_only or len(cells) != 1:
            raise ValueError(
                "debug_num_kv_blocks requires metrics_only and one cell")
    if not isinstance(debug_analytical_launch, bool):
        raise TypeError("debug_analytical_launch must be boolean")
    if host_launch_profile_paths is not None:
        required_hardware = {cell.hardware.hardware_id for cell in cells}
        unexpected_hardware = set(host_launch_profile_paths) - required_hardware
        if unexpected_hardware:
            raise ValueError(
                "three-mode evaluation profiles contain unselected hardware: "
                f"{sorted(unexpected_hardware)}")
    if scheduler_host_profile_paths is not None:
        required_hardware = {cell.hardware.hardware_id for cell in cells}
        unexpected_scheduler = (
            set(scheduler_host_profile_paths) - required_hardware)
        if unexpected_scheduler:
            raise ValueError(
                "evaluation Scheduler Host profiles contain unselected "
                f"hardware: {sorted(unexpected_scheduler)}")
    started = time.perf_counter()
    if isolated_case_worker is None:
        if jobs != 1:
            raise ValueError(
                "parallel compact evaluation requires an isolated case worker"
            )
        if load_validated_case is None or run_simulation is None:
            raise ValueError(
                "serial compact evaluation requires loader and simulator"
            )
    elif load_validated_case is not None or run_simulation is not None:
        raise ValueError(
            "isolated compact evaluation cannot use in-process dependencies"
        )
    output_dir, alignment_inputs = _prepare_phase1_evaluation(
        repository_root,
        config,
        cells,
        run_id,
        metrics_only=metrics_only,
        precomputed_alignment=precomputed_alignment,
    )
    if isolated_case_worker is None:
        assert load_validated_case is not None
        assert run_simulation is not None
        results = [
            _evaluate_one_case(
                repository_root,
                config,
                cell,
                source_commit,
                load_validated_case,
                run_simulation,
                alignment_input,
                execution_mode,
                debug_num_kv_blocks,
                debug_analytical_launch,
                host_launch_profile_paths,
                scheduler_host_profile_paths,
            )
            for cell, alignment_input in zip(cells, alignment_inputs, strict=True)
        ]
    else:
        results = _run_parallel_cases(
            repository_root,
            config,
            cells,
            source_commit,
            jobs,
            isolated_case_worker,
            alignment_inputs=alignment_inputs,
            execution_mode=execution_mode,
            debug_num_kv_blocks=debug_num_kv_blocks,
            debug_analytical_launch=debug_analytical_launch,
            host_launch_profile_paths=host_launch_profile_paths,
            scheduler_host_profile_paths=scheduler_host_profile_paths,
        )
    return _persist_phase1_evaluation(
        repository_root,
        config,
        source_commit,
        output_dir,
        results,
        jobs=jobs,
        execution_mode=execution_mode,
        evaluation_wall_time_seconds=time.perf_counter() - started,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
    )


def _prepare_phase1_evaluation(
    repository_root: str | Path,
    config: Phase1Config,
    cells: Sequence[Phase1Cell],
    run_id: str,
    *,
    metrics_only: bool,
    precomputed_alignment: Mapping[str, Mapping[str, object]] | None,
) -> tuple[Path, tuple[bool | Mapping[str, object], ...]]:
    alignment_inputs = _case_alignment_inputs(
        cells,
        metrics_only=metrics_only,
        precomputed_alignment=precomputed_alignment,
    )
    output_dir = create_run_directory(repository_root, run_id)
    for cell in cells:
        case_path = compact_case_path(repository_root, config, cell)
        for path in (case_path / "manifest.json", case_path / "requests.csv"):
            if not path.is_file():
                raise ValueError(f"phase-1 case input is missing: {path}")
    return output_dir, alignment_inputs


def _persist_phase1_evaluation(
    repository_root: str | Path,
    config: Phase1Config,
    source_commit: str,
    output_dir: Path,
    results: Sequence[Mapping[str, object]],
    *,
    jobs: int,
    execution_mode: str,
    evaluation_wall_time_seconds: float,
    debug_num_kv_blocks: int | None,
    debug_analytical_launch: bool,
) -> dict[str, object]:
    alignment_stage_ran = bool(results) and all(
        _mapping(result["alignment"], "cell alignment")["alignment_stage_ran"] is True
        for result in results
    )
    alignment_checked = bool(results) and all(
        _mapping(result["alignment"], "cell alignment")["alignment_checked"] is True
        for result in results
    )
    formal_conclusion_allowed = (
        debug_num_kv_blocks is None
        and not debug_analytical_launch
        and bool(results)
        and all(
        _mapping(result["alignment"], "cell alignment")[
            "formal_conclusion_allowed"
        ] is True
        for result in results
        )
    )
    paths = write_phase1_results(
        output_dir,
        config=config,
        source_commit=source_commit,
        cells=results,
        execution_mode=execution_mode,
        evaluation_wall_time_seconds=evaluation_wall_time_seconds,
        alignment_stage_ran=alignment_stage_ran,
        alignment_checked=alignment_checked,
        formal_conclusion_allowed=formal_conclusion_allowed,
    )
    report_paths = write_sharegpt_evaluation_report(
        output_dir,
        results,
        run_elapsed_wall_time_seconds=evaluation_wall_time_seconds,
    )
    paths = {**paths, **report_paths}
    root = Path(repository_root).resolve()
    return {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "execution_mode": execution_mode,
        "evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        "run_directory": str(output_dir.relative_to(root)),
        "cell_count": len(results),
        "completed_count": sum(
            result["prediction_status"] == "completed" for result in results
        ),
        "unsupported_count": sum(
            result["prediction_status"] == "unsupported_launch_coverage"
            for result in results
        ),
        "jobs": jobs,
        "alignment_stage_ran": alignment_stage_ran,
        "alignment_checked": alignment_checked,
        "formal_conclusion_allowed": formal_conclusion_allowed,
        "provisional_observer_case_count": sum(
            result["observation_quality"]["status"] == "provisional"
            for result in results),
        "artifact_paths": {
            name: str(path.relative_to(root)) for name, path in paths.items()
        },
    }


def _evaluate_one_case(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    source_commit: str,
    load_validated_case: Callable[[Path], ValidatedCompactCase],
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    alignment_checked: bool | Mapping[str, object],
    execution_mode: str,
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    case_path = compact_case_path(repository_root, config, cell)
    validated_case = load_validated_case(case_path)
    hardware_id = cell.hardware.hardware_id
    profile_path = None
    if (host_launch_profile_paths is not None
            and hardware_id in host_launch_profile_paths):
        profile_path = host_launch_profile_paths[hardware_id]
    evaluator = (
        evaluate_validated_case_three_modes
        if profile_path is not None else evaluate_validated_case)
    extra = (
        {"host_launch_profile_path": profile_path}
        if profile_path is not None else {})
    if profile_path is not None:
        if (scheduler_host_profile_paths is None
                or hardware_id not in scheduler_host_profile_paths):
            raise ValueError(
                "three-mode evaluation requires a Scheduler Host profile "
                f"for {hardware_id}")
        extra["scheduler_host_profile_path"] = (
            scheduler_host_profile_paths[hardware_id])
    elif scheduler_host_profile_paths is not None:
        raise ValueError(
            "device-only evaluation cannot consume Scheduler Host profiles")
    return evaluator(
        repository_root,
        config,
        cell,
        validated_case,
        run_simulation,
        source_commit=source_commit,
        alignment_checked=alignment_checked,
        execution_mode=execution_mode,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
        **extra,
    )


def _run_parallel_cases(
    repository_root: str | Path,
    config: Phase1Config,
    cells: Sequence[Phase1Cell],
    source_commit: str,
    jobs: int,
    isolated_case_worker: Callable[
        [
            str,
            Phase1Config,
            Phase1Cell,
            str,
            bool | Mapping[str, object],
            str,
        ],
        dict[str, object],
    ],
    alignment_checked: bool = True,
    *,
    alignment_inputs: Sequence[bool | Mapping[str, object]] | None = None,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    if not cells:
        return []
    if alignment_inputs is None:
        resolved_alignment_inputs = tuple(alignment_checked for _ in cells)
    else:
        resolved_alignment_inputs = tuple(alignment_inputs)
        if len(resolved_alignment_inputs) != len(cells):
            raise ValueError("alignment input count must match selected cells")
    results, _completed_at = _run_parallel_case_jobs(
        repository_root,
        tuple(
            (config, cell, alignment_input)
            for cell, alignment_input in zip(
                cells, resolved_alignment_inputs, strict=True
            )
        ),
        source_commit,
        jobs,
        isolated_case_worker,
        execution_mode=execution_mode,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
        host_launch_profile_paths=host_launch_profile_paths,
        scheduler_host_profile_paths=scheduler_host_profile_paths,
    )
    return results


def _run_parallel_case_jobs(
    repository_root: str | Path,
    case_jobs: Sequence[
        tuple[
            Phase1Config,
            Phase1Cell,
            bool | Mapping[str, object],
        ]
    ],
    source_commit: str,
    jobs: int,
    isolated_case_worker: Callable[
        [
            str,
            Phase1Config,
            Phase1Cell,
            str,
            bool | Mapping[str, object],
            str,
        ],
        dict[str, object],
    ],
    *,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, object]], tuple[float, ...]]:
    if not case_jobs:
        return [], ()
    result_by_index: dict[int, dict[str, object]] = {}
    completed_at_by_index: dict[int, float] = {}
    executor = ProcessPoolExecutor(max_workers=min(jobs, len(case_jobs)))
    future_to_index = {}
    try:
        for index, (config, cell, alignment_input) in enumerate(case_jobs):
            args = (
                str(Path(repository_root).resolve()),
                config,
                cell,
                source_commit,
                alignment_input,
                execution_mode,
            )
            try:
                optional_args = []
                if debug_num_kv_blocks is not None or debug_analytical_launch:
                    optional_args.extend((
                        debug_num_kv_blocks, debug_analytical_launch))
                if host_launch_profile_paths is not None:
                    if not optional_args:
                        optional_args.extend((None, False))
                    optional_args.append(dict(host_launch_profile_paths))
                if scheduler_host_profile_paths is not None:
                    if not optional_args:
                        optional_args.extend((None, False, None))
                    elif len(optional_args) == 2:
                        optional_args.append(None)
                    optional_args.append(dict(scheduler_host_profile_paths))
                future = (
                    executor.submit(isolated_case_worker, *args, *optional_args)
                )
            except Exception as error:
                for pending in future_to_index:
                    pending.cancel()
                raise RuntimeError(
                    "compact evaluation worker submission failed for "
                    f"{_canonical_cell_id(cell)}"
                ) from error
            future_to_index[future] = index
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_by_index[index] = future.result()
                completed_at_by_index[index] = time.perf_counter()
            except Exception as error:
                for pending in future_to_index:
                    pending.cancel()
                raise RuntimeError(
                    "compact evaluation worker failed for "
                    f"{_canonical_cell_id(case_jobs[index][1])}"
                ) from error
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return (
        [result_by_index[index] for index in range(len(case_jobs))],
        tuple(completed_at_by_index[index] for index in range(len(case_jobs))),
    )


def write_phase1_results(
    output_dir: str | Path,
    *,
    config: Phase1Config,
    source_commit: str,
    cells: Sequence[Mapping[str, object]],
    execution_mode: str = "accelerated",
    evaluation_wall_time_seconds: float = 0.0,
    alignment_stage_ran: bool = False,
    alignment_checked: bool = False,
    formal_conclusion_allowed: bool = False,
) -> Mapping[str, Path]:
    """Persist only aggregate, request-safe experiment artifacts."""

    destination = Path(output_dir)
    if not destination.is_dir():
        raise ValueError(f"phase-1 output directory must already exist: {destination}")
    if not isinstance(source_commit, str) or _GIT_COMMIT.fullmatch(
        source_commit
    ) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    _validate_execution_mode(execution_mode)
    evaluation_wall_time_seconds = _finite_number(
        evaluation_wall_time_seconds, "evaluation wall time"
    )
    if evaluation_wall_time_seconds < 0.0:
        raise ValueError("evaluation wall time must be nonnegative")
    for cell in cells:
        if cell["execution_mode"] != execution_mode:
            raise ValueError(
                "persisted cell execution_mode disagrees with run mode")
    quality_counts = {
        status: sum(cell["observation_quality"]["status"] == status
                    for cell in cells)
        for status in ("accepted", "provisional", "legacy_contract")
    }
    if sum(quality_counts.values()) != len(cells):
        raise ValueError("unknown observation quality status")
    scored_count = sum(cell["prediction_status"] == "completed" for cell in cells)
    result = {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "interpretation": "observed_vs_current_prediction_not_capture_replay",
        "config_sha256": config.source_sha256,
        "source_commit": source_commit,
        "execution_mode": execution_mode,
        "evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        "cell_count": len(cells),
        "observation_quality_counts": quality_counts,
        "formal_observation_denominator": scored_count,
        "alignment_stage_ran": alignment_stage_ran,
        "alignment_checked": alignment_checked,
        "formal_conclusion_allowed": formal_conclusion_allowed,
        "cells": list(cells),
    }
    paths = {
        "json": destination / "results.json",
        "csv": destination / "results.csv",
        "markdown": destination / "report.md",
        "manifest": destination / "run_manifest.json",
        "hashes": destination / "hashes.json",
    }
    _write_json(paths["json"], result)
    _write_csv(paths["csv"], cells)
    paths["markdown"].write_text(_render_markdown(result))
    manifest = {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "interpretation": "observed_vs_current_prediction_not_capture_replay",
        "config_path": str(config.source_path),
        "config_sha256": config.source_sha256,
        "source_commit": source_commit,
        "execution_mode": execution_mode,
        "evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        "alignment_stage_ran": alignment_stage_ran,
        "alignment_checked": alignment_checked,
        "formal_conclusion_allowed": formal_conclusion_allowed,
        "observation_quality_counts": quality_counts,
        "formal_observation_denominator": scored_count,
        "prediction_contract": dict(config.prediction.values),
        "model_mappings": [
            {
                "dataset_model_id": model.dataset_model_id,
                "simulation_model_path": model.simulation_model_path,
                "model_config_sha256": model.model_config_sha256,
            }
            for model in config.models
        ],
        "hardware_mappings": [
            {
                "hardware_id": hardware.hardware_id,
                "hardware_config_path": hardware.hardware_config_path,
                "hardware_config_sha256": hardware.hardware_config_sha256,
                "graph_execution": hardware.graph_execution.manifest(),
            }
            for hardware in config.hardware
        ],
        "selected_cell_inputs": [
            {
                "cell_id": cell["cell_id"],
                "manifest_sha256": cell["manifest_sha256"],
                "requests_sha256": cell["requests_sha256"],
                "alignment": cell["alignment_files"],
            }
            for cell in cells
        ],
        "persisted_content": (
            "aggregate_plus_synthetic_request_index_paired_latency"
        ),
    }
    _write_json(paths["manifest"], manifest)
    digests = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
        if name != "hashes"
    }
    _write_json(paths["hashes"], {"schema_version": 1, "sha256": digests})
    return paths


def _write_csv(path: Path, cells: Sequence[Mapping[str, object]]) -> None:
    fieldnames = (
        "cell_id",
        "hardware_id",
        "model_id",
        "case_id",
        "target_qps",
        "execution_mode",
        "evaluation_wall_time_seconds",
        "prediction_status",
        "performance_timing_status",
        "accuracy_claim_status",
        "observer_quality_status",
        "observer_failed_metrics",
        "capture_contract_status",
        "metric",
        "statistic",
        "observed",
        "predicted",
        "error",
        "absolute_percentage_error",
        "value",
        "eligible_count",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            metrics = _mapping(cell["metrics"], "cell metrics")
            latency = _mapping(metrics["latency"], "cell latency")
            for metric_name in _LATENCY_FIELDS:
                distribution = _mapping(latency[metric_name], metric_name)
                for statistic in STATISTICS:
                    writer.writerow(_csv_row(cell, metric_name, statistic, distribution[statistic]))
            throughput = _mapping(metrics["throughput"], "cell throughput")
            for metric_name in _THROUGHPUT_FIELDS:
                writer.writerow(_csv_row(cell, metric_name, "scalar", throughput[metric_name]))
            paired = _mapping(
                metrics["paired_request_error"], "paired request error"
            )
            for metric_name in _LATENCY_FIELDS:
                summary = _mapping(paired[metric_name], metric_name)
                eligible_count = summary[
                    "absolute_percentage_error_eligible_count"
                ]
                writer.writerow(_paired_csv_row(
                    cell, metric_name, "mae_ms", summary["mae_ms"], None
                ))
                writer.writerow(_paired_csv_row(
                    cell,
                    metric_name,
                    "mape_percent",
                    summary["mape_percent"],
                    eligible_count,
                ))
                writer.writerow(_paired_csv_row(
                    cell,
                    metric_name,
                    "absolute_percentage_error_eligible_count",
                    eligible_count,
                    eligible_count,
                ))
                for statistic in ("p50", "p90", "p99"):
                    writer.writerow(_paired_csv_row(
                        cell,
                        metric_name,
                        f"absolute_error_ms_{statistic}",
                        _mapping(
                            summary["absolute_error_ms"],
                            "paired absolute errors",
                        )[statistic],
                        None,
                    ))
                    writer.writerow(_paired_csv_row(
                        cell,
                        metric_name,
                        f"absolute_percentage_error_{statistic}",
                        _mapping(
                            summary["absolute_percentage_error"],
                            "paired percentage errors",
                        )[statistic],
                        eligible_count,
                    ))


def _csv_row(
    cell: Mapping[str, object],
    metric: str,
    statistic: str,
    comparison_value: object,
) -> Mapping[str, object]:
    comparison = _mapping(comparison_value, "metric comparison")
    return {
        "cell_id": cell["cell_id"],
        "hardware_id": cell["hardware_id"],
        "model_id": cell["model_id"],
        "case_id": cell["case_id"],
        "target_qps": cell["target_qps"],
        "execution_mode": cell["execution_mode"],
        "evaluation_wall_time_seconds": cell[
            "evaluation_wall_time_seconds"],
        "prediction_status": cell["prediction_status"],
        "performance_timing_status": cell["performance_timing_status"],
        "accuracy_claim_status": cell["accuracy_claim_status"],
        "observer_quality_status": cell["observation_quality"]["status"],
        "observer_failed_metrics": ",".join(
            cell["observation_quality"]["failed_metrics"]),
        "capture_contract_status": cell["capture_contract_status"],
        "metric": metric,
        "statistic": statistic,
        "observed": comparison["observed"],
        "predicted": comparison["predicted"],
        "error": comparison["error"],
        "absolute_percentage_error": comparison["absolute_percentage_error"],
        "value": None,
        "eligible_count": None,
    }


def _paired_csv_row(
    cell: Mapping[str, object],
    metric: str,
    statistic: str,
    value: object,
    eligible_count: object,
) -> Mapping[str, object]:
    return {
        "cell_id": cell["cell_id"],
        "hardware_id": cell["hardware_id"],
        "model_id": cell["model_id"],
        "case_id": cell["case_id"],
        "target_qps": cell["target_qps"],
        "execution_mode": cell["execution_mode"],
        "evaluation_wall_time_seconds": cell[
            "evaluation_wall_time_seconds"],
        "prediction_status": cell["prediction_status"],
        "performance_timing_status": cell["performance_timing_status"],
        "accuracy_claim_status": cell["accuracy_claim_status"],
        "observer_quality_status": cell["observation_quality"]["status"],
        "observer_failed_metrics": ",".join(
            cell["observation_quality"]["failed_metrics"]),
        "capture_contract_status": cell["capture_contract_status"],
        "metric": metric,
        "statistic": statistic,
        "observed": None,
        "predicted": None,
        "error": None,
        "absolute_percentage_error": None,
        "value": value,
        "eligible_count": eligible_count,
    }


def _render_markdown(result: Mapping[str, object]) -> str:
    if result["formal_conclusion_allowed"] is True:
        conclusion = "FORMAL: alignment and observer quality gates passed."
    elif result["alignment_stage_ran"] is False:
        conclusion = (
            "NON-FORMAL DEBUG OUTPUT: the mandatory compact alignment stage "
            "did not run."
        )
    else:
        conclusion = (
            "NON-FORMAL OUTPUT: alignment or observer quality gate failed."
        )
    lines = [
        "# ShareGPT compact-suite evaluation",
        "",
        f"**{conclusion}**",
        "",
        "This report compares historical compact observations with the current pinned vLLM prediction contract; it is not capture-exact replay.",
        "",
        f"Execution mode: `{result['execution_mode']}`. Evaluation wall time: {float(result['evaluation_wall_time_seconds']):.6f} s.",
        f"Formal observation denominator: {result['formal_observation_denominator']}/{result['cell_count']}; capture differences above 5% are diagnostic labels only ({result['observation_quality_counts']['provisional']} cases).",
        "",
        "| Cell | Prediction | Timing status | Accuracy claim | Observer quality | Failed metrics | Capture contract | TTFT MAPE | TPOT MAPE | E2E MAPE |",
        "|---|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for cell_value in _sequence(result["cells"], "result cells"):
        cell = _mapping(cell_value, "result cell")
        metrics = _mapping(cell["metrics"], "cell metrics")
        paired = _mapping(metrics["paired_request_error"], "paired errors")
        mape_values = []
        for metric_name in _LATENCY_FIELDS:
            summary = _mapping(paired[metric_name], metric_name)
            mape = summary["mape_percent"]
            if cell["prediction_status"] == "completed" and mape is not None:
                mape_values.append(f"{_finite_number(mape, metric_name):.2f}%")
            else:
                mape_values.append("--")
        lines.append(
            f"| {cell['cell_id']} | {cell['prediction_status']} | "
            f"{cell['performance_timing_status']} | "
            f"{cell['accuracy_claim_status']} | "
            f"{cell['observation_quality']['status']} | "
            f"{', '.join(cell['observation_quality']['failed_metrics']) or '—'} | "
            f"{cell['capture_contract_status']} | "
            f"{' | '.join(mape_values)} |"
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_jobs(jobs: int) -> None:
    if type(jobs) is not int:
        raise TypeError("jobs must be an exact integer")
    if jobs < 1:
        raise ValueError("jobs must be a positive integer")


def run_virtual_forward_report(
    repository_root: str | Path,
    run_id: str,
    selected_ids: Sequence[str],
    *,
    jobs: int,
    python: Path,
    sdk_python: Path | None,
    source_root: Path | None,
    load_validated_case: Callable,
) -> dict[str, object]:
    """Evaluate virtual migration with the original case prediction contract.

    Unsupported source bindings remain visible instead of changing methods,
    capture, scheduling or capacity to make a case execute. Empty selection
    retains all 45 non-H200 and 15 captured H2000.30 cells; the captured runtime
    version does not overwrite the prediction contract's Scheduler version.
    """
    from evaluation.serving.metrics.symbolic_model import (
        run_virtual_serving_evaluation,
        write_virtual_serving_evaluation_report,
    )

    _validate_jobs(jobs)
    output_dir = create_run_directory(repository_root, run_id)
    result_path = run_virtual_serving_evaluation(
        Path(repository_root).resolve(), output_dir / "virtual_forward",
        selected_ids, jobs, python, sdk_python, source_root,
        load_validated_case, h200_version="0.30.0")
    report_path = output_dir / "serving_report.md"
    write_virtual_serving_evaluation_report(result_path, report_path)
    result = _mapping(json.loads(result_path.read_text()), "virtual Serving results")
    cells = _sequence(result["cells"], "virtual Serving cells")
    return {
        "suite_id": "sharegpt_virtual_tensor",
        "workload_path": "virtual_tensor",
        "case_count": len(cells),
        "completed_case_count": sum(cell["prediction_status"] == "completed" for cell in cells),
        "prediction_status_counts": {
            status: sum(cell["prediction_status"] == status for cell in cells)
            for status in sorted({cell["prediction_status"] for cell in cells})
        },
        "formal_conclusion_allowed": False,
        "normal_capacity_accepted": False,
        "goal_complete": False,
        "results_path": str(result_path),
        "serving_report_path": str(report_path),
    }
