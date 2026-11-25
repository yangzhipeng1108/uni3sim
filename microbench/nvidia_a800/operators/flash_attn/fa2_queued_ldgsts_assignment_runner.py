#!/usr/bin/env python3
"""Capture and analyze source-ordered heterogeneous SM80 CTA assignment."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("fa2_queued_ldgsts_assignment.cu")
RUNNER = Path(__file__).resolve()
OUTPUT_ROOT = ROOT / "profiler_output/microbench/nvidia_a800"
NVCC = "/usr/local/cuda-12.2/bin/nvcc"
SM_COUNT = 108
RESIDENT_PER_SM = 2
RESIDENT_COHORT = SM_COUNT * RESIDENT_PER_SM
GRID = 512
CELLS = ((8, 0), (8, 8), (16, 0), (16, 16), (32, 0), (32, 32))


def _run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, timeout=600,
                          **kwargs)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _device_rows():
    query = "index,name,uuid,memory.used,utilization.gpu"
    result = _run([
        "nvidia-smi", f"--query-gpu={query}",
        "--format=csv,noheader,nounits"], cwd=ROOT, capture_output=True)
    rows = []
    for columns in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(columns) != 5:
            raise ValueError("unexpected nvidia-smi row")
        rows.append({
            "index": int(columns[0]), "name": columns[1], "uuid": columns[2],
            "memory_used_mib": int(columns[3]),
            "utilization_percent": int(columns[4]),
        })
    return rows


def _require_idle(device):
    rows = [row for row in _device_rows() if row["index"] == device]
    if len(rows) != 1:
        raise ValueError(f"device {device} did not resolve exactly once")
    row = rows[0]
    if (row["name"] != "NVIDIA A800-SXM4-80GB"
            or row["memory_used_mib"] > 8
            or row["utilization_percent"] != 0):
        raise RuntimeError(f"device {device} is not an idle A800: {row}")
    processes = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], cwd=ROOT,
        capture_output=True).stdout
    if any(line.split(",", 1)[0].strip() == row["uuid"]
           for line in processes.splitlines() if line.strip()):
        raise RuntimeError(f"device {device} has a compute process")
    return row


def _wait_for_idle(device):
    last_error = None
    for _ in range(20):
        try:
            return _require_idle(device)
        except RuntimeError as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"device {device} did not return idle: {last_error}")


def _parse_raw(path, trials, maximum_panels, period_offset):
    records = []
    metadata = {}
    header = None
    for columns in csv.reader(path.read_text().splitlines()):
        if columns[0] == "meta":
            metadata[columns[1]] = columns[2]
        elif columns[0] == "raw" and columns[1] == "trial":
            header = tuple(columns[1:])
        elif columns[0] == "raw":
            records.append(dict(zip(
                header, (int(value) for value in columns[1:]))))
        else:
            raise ValueError("unknown native output row")
    if len(records) != trials * GRID:
        raise ValueError("native record matrix is incomplete")
    if metadata != {
            "device": "0", "device_name": "NVIDIA A800-SXM4-80GB",
            "sm_count": str(SM_COUNT),
            "resident_ctas_per_sm": str(RESIDENT_PER_SM),
            "threads": "128", "dynamic_shared_bytes": str(76 * 1024),
            "grid": str(GRID), "maximum_panels": str(maximum_panels),
            "period_offset": str(period_offset)}:
        raise ValueError(f"native metadata changed: {metadata}")
    return records


def _pearson(xs, ys):
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y)
                    for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys))
    return 0.0 if denominator == 0.0 else numerator / denominator


def _analyze_trial(rows):
    if {row["source_ordinal"] for row in rows} != set(range(GRID)):
        raise ValueError("source ordinals are not complete")
    if {row["start_ticket"] for row in rows} != set(range(GRID)):
        raise ValueError("start tickets are not a permutation")
    starts = sorted(rows, key=lambda row: row["start_ticket"])
    initial = starts[:RESIDENT_COHORT]
    if {row["source_ordinal"] for row in initial} != set(
            range(RESIDENT_COHORT)):
        raise ValueError("first resident cohort changed source membership")
    by_sm = {sm: [] for sm in range(SM_COUNT)}
    for row in rows:
        if row["sm_id"] not in by_sm or row["end_ns"] <= row["begin_ns"]:
            raise ValueError("CTA record is outside physical bounds")
        by_sm[row["sm_id"]].append(row)
    for sm_rows in by_sm.values():
        if {row["resident_ordinal"] for row in sm_rows} != set(
                range(len(sm_rows))):
            raise ValueError("per-SM resident ordinals are not contiguous")
        sm_rows.sort(key=lambda row: row["resident_ordinal"])
    initial_pairs = {
        sm: tuple(sorted(row["source_ordinal"] for row in sm_rows[:2]))
        for sm, sm_rows in by_sm.items()}
    adjacent_initial_pairing = (
        sorted(initial_pairs.values())
        == [(2 * index, 2 * index + 1) for index in range(SM_COUNT)])
    initial_sm_by_source = {
        row["source_ordinal"]: row["sm_id"] for row in initial}
    round_robin_matches = 0
    queued = starts[RESIDENT_COHORT:]
    for row in queued:
        expected_sm = initial_sm_by_source[
            row["source_ordinal"] % RESIDENT_COHORT]
        round_robin_matches += row["sm_id"] == expected_sm
    global_slot_ends = {
        sm: [row["end_ns"] for row in sm_rows[:2]]
        for sm, sm_rows in by_sm.items()}
    global_ready_excesses = []
    global_ready_ranks = []
    for row in queued:
        ready_by_sm = {
            sm: min(slot_ends)
            for sm, slot_ends in global_slot_ends.items()}
        actual_ready = ready_by_sm[row["sm_id"]]
        ordered_ready = sorted(ready_by_sm.values())
        global_ready_excesses.append(actual_ready - ordered_ready[0])
        global_ready_ranks.append(ordered_ready.index(actual_ready))
        slot_ends = global_slot_ends[row["sm_id"]]
        slot = slot_ends.index(min(slot_ends))
        slot_ends[slot] = row["end_ns"]
    adjacent_tail_pairs = 0
    tail_pair_count = 0
    rows_by_source = {row["source_ordinal"]: row for row in rows}
    for lower in range(RESIDENT_COHORT, GRID - 1, 2):
        tail_pair_count += 1
        adjacent_tail_pairs += (
            rows_by_source[lower]["sm_id"]
            == rows_by_source[lower + 1]["sm_id"])
    availability_violations = 0
    ready_excesses = []
    initial_panel_sums = []
    tail_counts = []
    for sm_rows in by_sm.values():
        initial_panel_sums.append(sum(
            row["panel_count"] for row in sm_rows[:2]))
        tail_counts.append(max(0, len(sm_rows) - 2))
        slot_ends = [sm_rows[0]["end_ns"], sm_rows[1]["end_ns"]]
        for row in sm_rows[2:]:
            earliest = min(slot_ends)
            if row["begin_ns"] < earliest:
                availability_violations += 1
            ready_excesses.append(max(0, row["begin_ns"] - earliest))
            slot = slot_ends.index(earliest)
            slot_ends[slot] = row["end_ns"]
    displacement = [
        abs(RESIDENT_COHORT + index - row["source_ordinal"])
        for index, row in enumerate(queued)]
    return {
        "adjacent_initial_pairing": adjacent_initial_pairing,
        "round_robin_match_fraction": round_robin_matches / len(queued),
        "median_global_ready_excess_ns": sorted(global_ready_excesses)[
            len(global_ready_excesses) // 2],
        "p99_global_ready_rank": sorted(global_ready_ranks)[
            int(0.99 * (len(global_ready_ranks) - 1))],
        "maximum_global_ready_rank": max(global_ready_ranks),
        "fraction_within_2048ns_of_global_ready": sum(
            value <= 2048 for value in global_ready_excesses)
            / len(global_ready_excesses),
        "adjacent_tail_pair_fraction": adjacent_tail_pairs / tail_pair_count,
        "maximum_start_order_displacement": max(displacement),
        "p99_start_order_displacement": sorted(displacement)[
            int(0.99 * (len(displacement) - 1))],
        "same_sm_slot_availability_violations": availability_violations,
        "median_same_sm_ready_excess_ns": sorted(ready_excesses)[
            len(ready_excesses) // 2],
        "initial_work_vs_tail_count_pearson": _pearson(
            initial_panel_sums, tail_counts),
        "normalized_assignment": [
            initial_pairs[row["sm_id"]][0] // 2 for row in rows_by_source.values()
        ],
    }


def _analyze(records, trials, maximum_panels, period_offset):
    trial_reports = []
    normalized = []
    for trial in range(trials):
        report = _analyze_trial([
            row for row in records if row["trial"] == trial])
        normalized.append(tuple(report.pop("normalized_assignment")))
        report["trial"] = trial
        trial_reports.append(report)
    return {
        "maximum_panels": maximum_panels,
        "period_offset": period_offset,
        "trial_reports": trial_reports,
        "all_adjacent_initial_pairing": all(
            report["adjacent_initial_pairing"] for report in trial_reports),
        "maximum_round_robin_match_fraction": max(
            report["round_robin_match_fraction"]
            for report in trial_reports),
        "minimum_round_robin_match_fraction": min(
            report["round_robin_match_fraction"]
            for report in trial_reports),
        "cross_trial_normalized_assignment_stable": all(
            value == normalized[0] for value in normalized[1:]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.trials < 3 or args.trials > 11:
        raise ValueError("trials must be in [3, 11]")
    output = args.output.resolve()
    if output.exists() or OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("output must be new and below the A800 profiler root")
    preflight = _require_idle(args.device)
    output.mkdir(parents=True)
    (output / "source").mkdir()
    (output / "build").mkdir()
    shutil.copy2(SOURCE, output / "source" / SOURCE.name)
    shutil.copy2(RUNNER, output / "source" / RUNNER.name)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                  capture_output=True).stdout.strip()
    status = _run(["git", "status", "--short"], cwd=ROOT,
                  capture_output=True).stdout
    (output / "baseline_commit.txt").write_text(commit + "\n")
    (output / "tracked_worktree_status.txt").write_text(status)
    binary = output / "build/fa2_queued_ldgsts_assignment"
    compile_command = [
        NVCC, "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo",
        str(output / "source" / SOURCE.name), "-o", str(binary)]
    compile_result = _run(compile_command, cwd=ROOT, capture_output=True)
    (output / "build/compile.stdout").write_text(compile_result.stdout)
    (output / "build/compile.stderr").write_text(compile_result.stderr)
    cell_reports = []
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    for maximum_panels, period_offset in CELLS:
        stem = f"max{maximum_panels}_offset{period_offset}"
        command = [
            str(binary), str(args.trials), str(GRID),
            str(maximum_panels), str(period_offset), "0"]
        result = _run(command, cwd=ROOT, env=environment,
                      capture_output=True)
        raw = output / f"{stem}.csv"
        raw.write_text(result.stdout)
        (output / f"{stem}.stderr").write_text(result.stderr)
        records = _parse_raw(
            raw, args.trials, maximum_panels, period_offset)
        report = _analyze(
            records, args.trials, maximum_panels, period_offset)
        report["raw_sha256"] = _sha256(raw)
        cell_reports.append(report)
    postflight = _wait_for_idle(args.device)
    analysis = {
        "schema": "nvidia_a800_fa2_queued_ldgsts_assignment_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "source_ordered_heterogeneous_ldgsts_cta_assignment",
        "grid_ctas": GRID,
        "resident_ctas_per_sm": RESIDENT_PER_SM,
        "resident_cohort_ctas": RESIDENT_COHORT,
        "trials": args.trials,
        "cells": cell_reports,
        "production_or_evaluation_duration_consumed": False,
    }
    _write_json(output / "analysis.json", analysis)
    _write_json(output / "provenance.json", {
        "schema": "nvidia_a800_fa2_queued_ldgsts_assignment_capture_v1",
        "baseline_commit": commit,
        "owner": "research-kernel-accuracy-fa",
        "pid": os.getpid(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "physical_device": preflight,
        "postflight_device": postflight,
        "source_sha256": _sha256(output / "source" / SOURCE.name),
        "runner_sha256": _sha256(output / "source" / RUNNER.name),
        "binary_sha256": _sha256(binary),
        "production_or_evaluation_duration_consumed": False,
    })
    print(json.dumps({"output": str(output), "analysis": analysis},
                     sort_keys=True))


if __name__ == "__main__":
    main()
