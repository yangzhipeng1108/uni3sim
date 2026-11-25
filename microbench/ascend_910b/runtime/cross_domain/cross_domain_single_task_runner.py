#!/usr/bin/env python3
"""Capture exact single-task active-AIC-group points of the CANN FIA ring.

The frozen campaign in ``cross_domain_runner.py`` measured a fixed block-dim
set, so ``scheduling.fia_cross_domain_three_handoff.single_task_active_aic_
groups`` covers only part of the 1..24 domain.  Grids outside that domain make
the consumer drop the whole program plan, so the missing points have to be
measured rather than interpolated.

This driver adds points without disturbing the published campaign.  It never
rebuilds or rewrites an existing report: it drives the already-published
benchmark binary directly and leaves every file named in the frozen runner's
``SOURCE_FILENAMES`` untouched, so the snapshot digest those reports are bound
to stays valid.

It reproduces the frozen campaign's estimator rather than inventing a weaker
one.  A single six-trial median of this quantity is not reproducible: four
independent six-trial medians of the identical ``b4_n1`` cell came out at 960,
980, 1030 and 1100 ns, a 14.6% relative span, which is wider than the ~10%
step between adjacent AIC groups.  The published campaign already carries that
property -- its own cross-run relative span reached 13.0% at group 1 and its
gate tolerates 15%.  So each trial runs every cell inside one child process in
balanced forward/reverse order, exactly as ``cross_domain_runner.py`` does, and
``aggregate`` takes the median of two independent runs' per-cell medians, the
same estimator ``cross_domain_report.py`` freezes, gated on the same 0.15
cross-run relative span with a third run held out as validation.

The measured quantity is the whole-program span of an empty MM1 -> Vec1 ->
MM2 -> Vec2 three-handoff ring at one task per active AIC group.  It carries
no production arithmetic, tensor traffic or operator duration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "cross_domain_contract.h",
    "cross_domain_rt_kernel.cpp",
    "cross_domain_rt_main.cpp",
    "cross_domain_runner.py",
    "cross_domain_report.py",
)
SYSTEM_COUNTER_HZ = 50_000_000
SYSTEM_COUNTER_TICK_NS = 20.0
TRIALS = 6
PARAMETER_RUN_COUNT = 2
CROSS_RUN_SPAN_LIMIT = 0.15
CHILD_HEADER = (
    "block_dim,task_counts,aic_cycles,aiv0_cycles,aiv1_cycles,"
    "host_wall_ns,validation"
)
RAW_FIELDS = (
    "trial",
    "order_direction",
    "order_position",
    "active_aic_groups",
    "task_counts",
    "aic_cycles",
    "aiv0_cycles",
    "aiv1_cycles",
    "completion_cycles",
    "completion_ns",
    "host_wall_ns_health_only",
    "validation",
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(benchmark_root: Path) -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILENAMES:
        path = benchmark_root / name
        payload = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _cycles(text: str, expected_length: int, label: str) -> list[int]:
    values = [int(item) for item in text.split(";")]
    if len(values) != expected_length:
        raise ValueError(f"{label} reported {len(values)} of {expected_length}")
    for value in values:
        if value <= 0:
            raise ValueError(f"{label} reported a non-positive cycle count")
    return values


def _build_command(binary: Path, block_dims: list[int]) -> list[str]:
    command = [str(binary), "--batch"]
    for block_dim in block_dims:
        command.extend((str(block_dim), ";".join(["1"] * block_dim)))
    return command


def _parse_row(line: str, block_dim: int) -> dict:
    fields = line.split(",")
    if len(fields) != 7:
        raise ValueError("benchmark stdout row does not match the contract")
    expected_task_counts = ";".join(["1"] * block_dim)
    if int(fields[0]) != block_dim or fields[1] != expected_task_counts:
        raise ValueError("benchmark echoed a different cell than requested")
    if fields[6] != "ok":
        raise ValueError(f"benchmark self-validation reported {fields[6]!r}")
    aic = _cycles(fields[2], block_dim, "aic_cycles")
    aiv0 = _cycles(fields[3], block_dim, "aiv0_cycles")
    aiv1 = _cycles(fields[4], block_dim, "aiv1_cycles")
    completion_cycles = max(max(aic), max(aiv0), max(aiv1))
    return {
        "active_aic_groups": block_dim,
        "task_counts": fields[1],
        "aic_cycles": fields[2],
        "aiv0_cycles": fields[3],
        "aiv1_cycles": fields[4],
        "completion_cycles": completion_cycles,
        "completion_ns": completion_cycles * SYSTEM_COUNTER_TICK_NS,
        "host_wall_ns_health_only": int(fields[5]),
        "validation": fields[6],
    }


def _npu_smi(npu_smi: Path, destination: Path) -> None:
    completed = subprocess.run([str(npu_smi), "info"], capture_output=True,
                               text=True, check=True, timeout=120)
    destination.write_text(completed.stdout, encoding="utf-8")


def _cell_summary(completions: list[float]) -> dict:
    median = statistics.median(completions)
    return {
        "single_task_duration_ns": median,
        "minimum_ns": min(completions),
        "maximum_ns": max(completions),
        "median_absolute_deviation_ns": statistics.median(
            [abs(value - median) for value in completions]),
    }


def _run(args, benchmark_root: Path) -> None:
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("run id contains unsupported characters")
    if not COMMIT_PATTERN.fullmatch(args.source_commit):
        raise ValueError("source commit must be a full Git commit")
    if not SHA256_PATTERN.fullmatch(args.expected_snapshot_sha256):
        raise ValueError("expected snapshot digest must be SHA256 hex")
    if not SHA256_PATTERN.fullmatch(args.expected_binary_sha256):
        raise ValueError("expected binary digest must be SHA256 hex")
    snapshot = _snapshot_sha256(benchmark_root)
    if snapshot != args.expected_snapshot_sha256:
        raise ValueError(
            f"benchmark source snapshot is {snapshot}, expected "
            f"{args.expected_snapshot_sha256}")
    binary_digest = _sha256(args.binary)
    if binary_digest != args.expected_binary_sha256:
        raise ValueError(
            f"benchmark binary is {binary_digest}, expected "
            f"{args.expected_binary_sha256}")
    block_dims = [int(item) for item in args.block_dims.split(",")]
    if len(set(block_dims)) != len(block_dims):
        raise ValueError("block dims repeat a point")
    for block_dim in block_dims:
        if not 1 <= block_dim <= 24:
            raise ValueError(f"block dim {block_dim} is outside 1..24")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _npu_smi(args.npu_smi, args.output_dir / "npu_smi_pre.txt")

    environment = dict(os.environ)
    environment["ASCEND_DEVICE_ID"] = "0"
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)

    rows = []
    trial_artifacts = []
    for trial in range(1, TRIALS + 1):
        forward = trial % 2 == 1
        direction = "forward" if forward else "reverse"
        # Every cell of a trial runs inside one child process in balanced
        # forward/reverse order, matching the frozen campaign: a per-cell
        # process would measure each cell's cold first launch instead.
        ordered = block_dims if forward else list(reversed(block_dims))
        command = _build_command(args.binary, ordered)
        completed = subprocess.run(command, env=environment,
                                   capture_output=True, text=True, check=True,
                                   timeout=args.child_timeout_seconds)
        stdout_path = args.output_dir / f"trial_{trial}.stdout"
        stderr_path = args.output_dir / f"trial_{trial}.stderr"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines or lines[0] != CHILD_HEADER:
            raise ValueError("benchmark stdout header changed")
        if len(lines) != len(ordered) + 1:
            raise ValueError("benchmark stdout cell count changed")
        for position, (block_dim, line) in enumerate(
                zip(ordered, lines[1:]), start=1):
            row = _parse_row(line, block_dim)
            row["trial"] = trial
            row["order_direction"] = direction
            row["order_position"] = position
            rows.append(row)
        trial_artifacts.append({
            "trial": trial,
            "order_direction": direction,
            "command": " ".join(command),
            "stdout": stdout_path.name,
            "stdout_sha256": _sha256(stdout_path),
            "stderr": stderr_path.name,
            "stderr_sha256": _sha256(stderr_path),
        })

    raw_path = args.output_dir / "raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(RAW_FIELDS) + "\n")
        for row in rows:
            handle.write(",".join(str(row[name]) for name in RAW_FIELDS) + "\n")

    points = []
    for block_dim in sorted(block_dims):
        cell = [row for row in rows if row["active_aic_groups"] == block_dim]
        if len(cell) != TRIALS:
            raise ValueError(f"block dim {block_dim} lost a trial")
        completions = [row["completion_ns"] for row in cell]
        summary = _cell_summary(completions)
        summary["active_aic_groups"] = block_dim
        summary["trial_completion_ns"] = completions
        for direction in ("forward", "reverse"):
            summary[f"{direction}_median_ns"] = statistics.median(
                [row["completion_ns"] for row in cell
                 if row["order_direction"] == direction])
        points.append(summary)

    _npu_smi(args.npu_smi, args.output_dir / "npu_smi_post.txt")
    report = {
        "schema": "ascend_910b_cross_domain_single_task_run_v1",
        "run_id": args.run_id,
        "hardware_id": "ascend_910b",
        "mechanism_id": "cann_fia_1_2_three_handoff_v5",
        "scope": (
            "empty source-bound MM1_to_Vec1_to_MM2_to_Vec2 three-handoff ring "
            "program span"),
        "task_counts_per_group": 1,
        "clock": {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
        },
        "trials": TRIALS,
        "order": "balanced_forward_reverse",
        "cells_per_child_process": len(block_dims),
        "points": points,
        "recommended_consumer": (
            "one input run of this driver's aggregate step; a single run is "
            "not a publishable LUT value on its own"),
        "production_or_evaluation_duration_consumed": False,
    }
    _write_json_exclusive(args.output_dir / "report.json", report)

    provenance = {
        "schema": "ascend_910b_cross_domain_single_task_run_provenance_v1",
        "run_id": args.run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "source_identity_mode": "audited_source_snapshot",
        "source_snapshot_sha256": snapshot,
        "source_files": {
            name: _sha256(benchmark_root / name)
            for name in SOURCE_FILENAMES
        },
        "binary": {
            "path": str(args.binary),
            "sha256": binary_digest,
            "rebuilt": False,
        },
        "machine": {"host": args.host, "container": args.container},
        "device": {
            "model": "910B2",
            "physical_device_id": args.device_id,
            "logical_device_id": 0,
            "selection_environment": {
                "ASCEND_DEVICE_ID": "0",
                "ASCEND_RT_VISIBLE_DEVICES": str(args.device_id),
            },
        },
        "idle_telemetry": {"pre": "npu_smi_pre.txt", "post": "npu_smi_post.txt"},
        "clock": {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
        },
        "measurement_scope": (
            "empty CANN 8.5.1 FIA-shaped three-handoff ring at one task per "
            "active AIC group; no production arithmetic, tensor traffic or "
            "operator duration; every cell of a trial runs inside one child "
            "process in balanced forward/reverse order"),
        "production_or_evaluation_duration_consumed": False,
        "campaign": {
            "block_dims": block_dims,
            "task_count_per_group": 1,
            "trials": TRIALS,
            "order": "balanced_forward_reverse",
        },
        "capture": {
            "raw": raw_path.name,
            "raw_sha256": _sha256(raw_path),
            "raw_rows": len(rows),
            "runner_pid": os.getpid(),
            "argv": list(sys.argv),
            "trials": trial_artifacts,
        },
    }
    _write_json_exclusive(args.output_dir / "provenance.json", provenance)
    (args.output_dir / "status.txt").write_text("complete", encoding="utf-8")
    print(args.output_dir / "report.json")


def _relative_span(values: list[float]) -> float:
    for value in values:
        if value <= 0:
            raise ValueError("relative-span values must be positive")
    return (max(values) - min(values)) / statistics.median(values)


def _aggregate(args) -> None:
    if len(args.run_dir) <= PARAMETER_RUN_COUNT:
        raise ValueError(
            f"aggregate needs more than {PARAMETER_RUN_COUNT} independent "
            "runs so one is held out for validation")
    reports = []
    for run_dir in args.run_dir:
        with (run_dir / "report.json").open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report["schema"] != "ascend_910b_cross_domain_single_task_run_v1":
            raise ValueError(f"{run_dir} is not a single-task run report")
        with (run_dir / "provenance.json").open(encoding="utf-8") as handle:
            provenance = json.load(handle)
        reports.append((run_dir, report, provenance))
    for field in ("source_snapshot_sha256", "source_commit"):
        values = {provenance[field] for _, _, provenance in reports}
        if len(values) != 1:
            raise ValueError(f"independent runs disagree on {field}")
    binaries = {provenance["binary"]["sha256"] for _, _, provenance in reports}
    if len(binaries) != 1:
        raise ValueError("independent runs disagree on the benchmark binary")
    run_ids = [report["run_id"] for _, report, _ in reports]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("independent runs repeat a run id")

    medians_by_run = []
    for _, report, _ in reports:
        medians_by_run.append({
            point["active_aic_groups"]: point["single_task_duration_ns"]
            for point in report["points"]
        })
    domains = {tuple(sorted(medians)) for medians in medians_by_run}
    if len(domains) != 1:
        raise ValueError("independent runs disagree on the measured domain")
    block_dims = sorted(medians_by_run[0])

    points = []
    for block_dim in block_dims:
        per_run = [medians[block_dim] for medians in medians_by_run]
        parameter_run_values = per_run[:PARAMETER_RUN_COUNT]
        # Exactly the estimator cross_domain_report.py freezes: the median of
        # the parameter runs' per-cell medians, gated on the cross-run span.
        value = statistics.median(parameter_run_values)
        points.append({
            "active_aic_groups": block_dim,
            "single_task_duration_ns": value,
            "parameter_run_medians_ns": parameter_run_values,
            "parameter_run_cross_run_relative_span": _relative_span(
                parameter_run_values),
            "all_run_medians_ns": per_run,
            "all_run_cross_run_relative_span": _relative_span(per_run),
            "validation_run_medians_ns": per_run[PARAMETER_RUN_COUNT:],
            "validation_relative_error": max(
                abs(held - value) / value
                for held in per_run[PARAMETER_RUN_COUNT:]),
        })

    gates = {
        "parameter_run_cross_run_span": all(
            point["parameter_run_cross_run_relative_span"]
            <= CROSS_RUN_SPAN_LIMIT for point in points),
        "all_run_cross_run_span": all(
            point["all_run_cross_run_relative_span"] <= CROSS_RUN_SPAN_LIMIT
            for point in points),
        "validation_run_relative_error": all(
            point["validation_relative_error"] <= CROSS_RUN_SPAN_LIMIT
            for point in points),
    }
    controls = {}
    for block_dim, published in args.control:
        matches = [point for point in points
                   if point["active_aic_groups"] == block_dim]
        if len(matches) != 1:
            raise ValueError(f"control {block_dim} was not measured")
        value = matches[0]["single_task_duration_ns"]
        controls[str(block_dim)] = {
            "published_ns": published,
            "measured_ns": value,
            "relative_error": abs(value - published) / published,
        }
    gates["published_control_agreement"] = all(
        control["relative_error"] <= CROSS_RUN_SPAN_LIMIT
        for control in controls.values()) if controls else False

    payload = {
        "schema": "ascend_910b_cross_domain_single_task_points_candidate_v2",
        "hardware_id": "ascend_910b",
        "mechanism_id": "cann_fia_1_2_three_handoff_v5",
        "estimator": (
            "median over the first "
            f"{PARAMETER_RUN_COUNT} independent runs of each run's median "
            f"over {TRIALS} balanced forward/reverse trials; remaining runs "
            "are held out as validation"),
        "cross_run_relative_span_limit": CROSS_RUN_SPAN_LIMIT,
        "input_runs": [
            {
                "run_id": report["run_id"],
                "path": str(run_dir),
                "role": ("parameter" if index < PARAMETER_RUN_COUNT
                         else "validation"),
                "report_sha256": _sha256(run_dir / "report.json"),
                "raw_sha256": provenance["capture"]["raw_sha256"],
            }
            for index, (run_dir, report, provenance) in enumerate(reports)
        ],
        "source_commit": reports[0][2]["source_commit"],
        "source_snapshot_sha256": reports[0][2]["source_snapshot_sha256"],
        "binary_sha256": reports[0][2]["binary"]["sha256"],
        "points": points,
        "published_controls": controls,
        "gates": gates,
        "recommended_consumer": (
            "exact active_aic_groups entries in "
            "scheduling.fia_cross_domain_three_handoff."
            "single_task_active_aic_groups"),
        "production_or_evaluation_duration_consumed": False,
        "limitations": [
            "This is an independent empty handoff microbenchmark, not a "
            "production kernel duration.",
            "It isolates only the three-handoff scheduler/control span and "
            "does not include MLA arithmetic, traffic, Fixpipe, or nUpdate "
            "atomic service.",
            "A six-trial median of this quantity reproduces only to roughly "
            "the 0.15 relative span the published campaign already gates on, "
            "so no point here resolves its immediate neighbour.",
        ],
    }
    _write_json_exclusive(args.output, payload)
    print(args.output)


def _control(text: str) -> tuple[int, float]:
    block_dim, published = text.split("=")
    return int(block_dim), float(published)


def main() -> None:
    benchmark_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("snapshot")
    run = subparsers.add_parser("run")
    run.add_argument("--binary", required=True, type=Path)
    run.add_argument("--block-dims", required=True)
    run.add_argument("--device-id", required=True, type=int)
    run.add_argument("--run-id", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--expected-snapshot-sha256", required=True)
    run.add_argument("--expected-binary-sha256", required=True)
    run.add_argument("--npu-smi", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--host", required=True)
    run.add_argument("--container", required=True)
    run.add_argument("--child-timeout-seconds", type=int, default=300)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-dir", required=True, action="append",
                           type=Path)
    aggregate.add_argument("--control", action="append", default=[],
                           type=_control)
    aggregate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        print(_snapshot_sha256(benchmark_root))
    elif args.command == "run":
        if args.child_timeout_seconds <= 0:
            raise ValueError("child timeout must be positive")
        _run(args, benchmark_root)
    elif args.command == "aggregate":
        _aggregate(args)
    else:
        raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
