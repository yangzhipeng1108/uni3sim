"""Shared driver for compact ShareGPT per-case study runs.

The Scheduler depth diagnostics studies resolve official compact cells,
confine one run directory under ``experiment_logs``, and execute independent
per-case Forward jobs in worker processes with order-preserving results.
This module holds that single skeleton; each study keeps its own payload
schema, cache identity validation, and artifact layout.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Callable, Sequence


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def require_run_id(run_id: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    return run_id


def require_positive_jobs(jobs: int) -> int:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    return jobs


def select_configured_cells(all_configured, selected_cell_ids):
    """Return the configured cells the caller asked for, in request order.

    A requested id the configuration does not contain is refused rather than
    silently dropped; a duplicate request is refused because the result would
    be ambiguous.  With no explicit selection every configured cell is used.
    """

    if not selected_cell_ids:
        return all_configured
    requested = tuple(selected_cell_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("selected cell IDs contain duplicates")
    available = {cell.cell_id for _config, cell in all_configured}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"unknown selected cells: {unknown}")
    return tuple(
        (config, cell)
        for requested_id in requested
        for config, cell in all_configured
        if cell.cell_id == requested_id
    )


def compact_dataset_case_path(dataset_base: Path, cell) -> Path:
    return (
        dataset_base
        / cell.hardware.hardware_id
        / cell.model.dataset_model_id
        / cell.case_id
    ).resolve(strict=True)


def resolve_experiment_run_directory(
    root: Path,
    relative_output_root: Path,
    run_id: str,
    *,
    label: str,
) -> Path:
    output_root = root / relative_output_root
    destination = (output_root / run_id).resolve()
    if output_root.resolve() not in destination.parents:
        raise ValueError(f"{label} must remain in experiment_logs")
    return destination


def run_case_jobs(
    jobspecs: Sequence[tuple[object, ...]],
    jobs: int,
    worker: Callable[..., dict[str, object]],
    read_cached: Callable[[tuple[object, ...]], dict[str, object] | None]
    | None = None,
) -> list[dict[str, object]]:
    """Run every per-case job in a worker process, preserving spec order.

    ``read_cached`` returns an already valid cached result for one spec or
    ``None`` when the spec must execute; its identity checks stay with the
    owning study.
    """

    results = {}
    pending = {}
    for index, spec in enumerate(jobspecs):
        cached = None if read_cached is None else read_cached(spec)
        if cached is None:
            pending[index] = spec
        else:
            results[index] = cached
    if not pending:
        return [results[index] for index in range(len(jobspecs))]
    with ProcessPoolExecutor(max_workers=min(jobs, len(pending))) as executor:
        futures = {
            executor.submit(worker, *spec): index
            for index, spec in pending.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[index] for index in range(len(jobspecs))]
