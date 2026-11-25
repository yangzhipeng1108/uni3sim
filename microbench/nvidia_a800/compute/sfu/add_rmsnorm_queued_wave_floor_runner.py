#!/usr/bin/env python3
"""Capture an assignment-invariant queued-wave floor for Add-RMS source rounds."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
from statistics import median
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output/microbench/nvidia_a800"
SOURCE_PATH = Path(__file__).with_name("add_rmsnorm_round_service.cu")
RUNNER_PATH = Path(__file__).resolve()
BASE_RUNNER_PATH = Path(__file__).with_name(
    "add_rmsnorm_round_service_runner.py")
BASE_SPEC = importlib.util.spec_from_file_location(
    "add_rmsnorm_round_service_runner", BASE_RUNNER_PATH)
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

EXPECTED_TRIALS = 12
TRAIN_TRIALS = tuple(range(6))
HOLDOUT_TRIALS = tuple(range(6, 12))
FUNCTIONS = ("add_square", "scale_multiply")
CELLS = {"resident": 864, "queued_wave": 1024}
MAXIMUM_RELATIVE_SPREAD = 0.10
RAW_FIELDS = BASE.RAW_FIELDS


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _relative_spread(values):
    center = median(values)
    if center <= 0:
        raise ValueError("spread inputs must have a positive median")
    return (max(values) - min(values)) / center


def _expected_order():
    order = []
    for trial in range(EXPECTED_TRIALS):
        for function in FUNCTIONS:
            if trial % 2 == 0:
                cells = ("queued_wave", "resident")
            else:
                cells = ("resident", "queued_wave")
            order.extend((function, cell, trial) for cell in cells)
    return tuple(order)


def load_raw(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("queued-wave raw header changed")
        raw_rows = list(reader)
    if len(raw_rows) != EXPECTED_TRIALS * len(FUNCTIONS) * len(CELLS):
        raise ValueError("queued-wave raw matrix is incomplete")
    normalized = []
    order = []
    for row in raw_rows:
        function = row["function"]
        cell = row["cell"]
        trial = int(row["trial"])
        order.append((function, cell, trial))
        if (function not in FUNCTIONS or cell not in CELLS
                or trial not in range(EXPECTED_TRIALS)):
            raise ValueError(
                f"invalid queued-wave function/cell/trial: "
                f"{function}/{cell}/{trial}")
        integers = {
            name: int(row[name]) for name in RAW_FIELDS
            if name not in ("function", "cell", "checksum")}
        if (integers["grid_blocks"] != CELLS[cell]
                or integers["block_threads"] != BASE.EXPECTED_THREADS
                or integers["rounds"] != BASE.EXPECTED_ROUNDS
                or integers["resident_cta_limit"]
                != BASE.EXPECTED_RESIDENT_CTAS
                or integers["active_sms"] != BASE.EXPECTED_SM_COUNT):
            raise ValueError(
                f"queued-wave launch contract changed for "
                f"{function}/{cell}/{trial}")
        if cell == "resident":
            assignment_valid = (
                integers["minimum_ctas_per_sm"]
                == BASE.EXPECTED_RESIDENT_CTAS
                == integers["maximum_ctas_per_sm"])
        elif cell == "queued_wave":
            assignment_valid = (
                BASE.EXPECTED_RESIDENT_CTAS
                <= integers["minimum_ctas_per_sm"]
                <= integers["maximum_ctas_per_sm"]
                <= 2 * BASE.EXPECTED_RESIDENT_CTAS
                and math.ceil(
                    integers["maximum_ctas_per_sm"]
                    / BASE.EXPECTED_RESIDENT_CTAS) == 2)
        else:
            raise ValueError(f"unknown queued-wave cell: {cell}")
        if not assignment_valid:
            raise ValueError(
                f"queued-wave assignment contract changed for "
                f"{function}/{cell}/{trial}")
        checksum = float(row["checksum"])
        if (integers["maximum_sm_span_ns"] <= 0
                or integers["median_block_ns"] <= 0
                or not math.isfinite(checksum)):
            raise ValueError(
                f"queued-wave timing/checksum failed for "
                f"{function}/{cell}/{trial}")
        normalized.append({
            "function": function,
            "cell": cell,
            "trial": trial,
            "checksum": checksum,
            **integers,
        })
    if tuple(order) != _expected_order():
        raise ValueError("queued-wave balanced cell order changed")
    return tuple(normalized)


def derive_candidate(rows, resources, sass_contract):
    by_key = {
        (row["function"], row["cell"], row["trial"]): row
        for row in rows}
    expected_keys = {
        (function, cell, trial)
        for function in FUNCTIONS
        for cell in CELLS
        for trial in range(EXPECTED_TRIALS)}
    if set(by_key) != expected_keys:
        raise ValueError("queued-wave rows do not form the frozen matrix")

    functions = {}
    all_positive = True
    all_cells_stable = True
    all_deltas_stable = True
    holdout_preserves_floor = True
    all_two_wave_assignments = True
    for function in FUNCTIONS:
        cell_spreads = {}
        for cell in CELLS:
            spans = [
                by_key[(function, cell, trial)]["maximum_sm_span_ns"]
                for trial in range(EXPECTED_TRIALS)]
            spread = _relative_spread(spans)
            cell_spreads[cell] = spread
            all_cells_stable = (
                all_cells_stable and spread <= MAXIMUM_RELATIVE_SPREAD)
        deltas = {
            trial: (
                by_key[(function, "queued_wave", trial)][
                    "maximum_sm_span_ns"]
                - by_key[(function, "resident", trial)][
                    "maximum_sm_span_ns"])
                / BASE.EXPECTED_ROUNDS
            for trial in range(EXPECTED_TRIALS)}
        all_positive = all_positive and all(value > 0 for value in deltas.values())
        delta_spread = _relative_spread(tuple(deltas.values()))
        all_deltas_stable = (
            all_deltas_stable
            and delta_spread <= MAXIMUM_RELATIVE_SPREAD)
        training = tuple(deltas[trial] for trial in TRAIN_TRIALS)
        holdout = tuple(deltas[trial] for trial in HOLDOUT_TRIALS)
        training_floor = min(training)
        holdout_margins = tuple(value - training_floor for value in holdout)
        holdout_pass = min(holdout_margins) >= 0
        holdout_preserves_floor = holdout_preserves_floor and holdout_pass
        queued_assignments = tuple(
            {
                "minimum_ctas_per_sm": by_key[
                    (function, "queued_wave", trial)][
                        "minimum_ctas_per_sm"],
                "maximum_ctas_per_sm": by_key[
                    (function, "queued_wave", trial)][
                        "maximum_ctas_per_sm"],
            }
            for trial in range(EXPECTED_TRIALS))
        assignment_pass = all(
            BASE.EXPECTED_RESIDENT_CTAS
            <= entry["minimum_ctas_per_sm"]
            <= entry["maximum_ctas_per_sm"]
            <= 2 * BASE.EXPECTED_RESIDENT_CTAS
            and math.ceil(
                entry["maximum_ctas_per_sm"]
                / BASE.EXPECTED_RESIDENT_CTAS) == 2
            for entry in queued_assignments)
        all_two_wave_assignments = (
            all_two_wave_assignments and assignment_pass)
        functions[function] = {
            "cell_maximum_relative_spread": cell_spreads,
            "additional_wave_ns_per_round": {
                "training_minimum": training_floor,
                "training_median": median(training),
                "holdout_minimum": min(holdout),
                "holdout_median": median(holdout),
                "holdout_minimum_margin_above_training_floor": (
                    min(holdout_margins)),
                "all_trial_relative_spread": delta_spread,
            },
            "queued_assignment_observed_ranges": {
                "minimum_ctas_per_sm": sorted({
                    entry["minimum_ctas_per_sm"]
                    for entry in queued_assignments}),
                "maximum_ctas_per_sm": sorted({
                    entry["maximum_ctas_per_sm"]
                    for entry in queued_assignments}),
            },
            "training_floor_held_out": holdout_pass,
        }

    static_resource_contract = (
        len(resources) == len(FUNCTIONS)
        and all(
            value["registers_per_thread"] <= 32
            and value["static_shared_bytes"] == 0
            and value["local_bytes"] == 0
            for value in resources.values()))
    static_sass_contract = (
        sass_contract["required_opcodes"]
        == ["LDG.E.128", "STG.E.128", "BAR.SYNC"]
        and sass_contract["forbidden_opcodes_absent"]
        == ["LDL", "STL", "ATOM"])
    gates = {
        "frozen_balanced_train_holdout_order": tuple(
            TRAIN_TRIALS + HOLDOUT_TRIALS) == tuple(range(EXPECTED_TRIALS)),
        "all_additional_wave_intervals_positive": all_positive,
        "all_cells_stable_within_10_percent": all_cells_stable,
        "all_additional_wave_intervals_stable_within_10_percent": (
            all_deltas_stable),
        "held_out_intervals_preserve_training_floor": (
            holdout_preserves_floor),
        "all_queued_launches_are_two_assignment_waves": (
            all_two_wave_assignments),
        "static_resource_contract": static_resource_contract,
        "static_sass_contract": static_sass_contract,
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        published_floors = {}
    else:
        published_floors = {
            function: functions[function]["additional_wave_ns_per_round"][
                "training_minimum"]
            for function in FUNCTIONS
        }
    publication = {
        "allowed": not failed,
        "failed_gates": failed,
        "additional_queued_wave_floor_ns_per_round": published_floors,
    }
    return {
        "schema": "nvidia_a800_add_rmsnorm_queued_wave_floor_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "independent_add_rms_source_round_queued_wave_floor",
        "block_threads": BASE.EXPECTED_THREADS,
        "rounds": BASE.EXPECTED_ROUNDS,
        "resident_ctas_per_sm": BASE.EXPECTED_RESIDENT_CTAS,
        "resident_grid_blocks": CELLS["resident"],
        "queued_grid_blocks": CELLS["queued_wave"],
        "training_trials": list(TRAIN_TRIALS),
        "holdout_trials": list(HOLDOUT_TRIALS),
        "split_rule": (
            "trial identity and alternating resident/queued order were frozen "
            "before capture"),
        "functions": functions,
        "gates": gates,
        "publication": publication,
        "forward_eligible": False,
        "production_or_evaluation_duration_consumed": False,
        "production_predictor_consumption_allowed": False,
        "allowed_conclusion": (
            "Only a held-out lower bound for the additional whole-launch "
            "source-round span of the second assignment wave may be published."),
        "excluded_conclusions": [
            "a complete Add-RMSNorm duration",
            "an exact queued CTA service rate",
            "an exact or average queued-tail CTA assignment",
            "a per-grid or per-shape timing lookup table",
            "generalization beyond the exact block-256 grid-1024 two-wave domain",
            "direct Forward consumption before composition review",
        ],
    }


def capture(args):
    if args.device < 0:
        raise ValueError("device must be nonnegative")
    if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", args.owner) is None:
        raise ValueError("owner must match [A-Za-z0-9._-]{1,64}")
    output = BASE._validated_output(args.output)
    started_at = _utc_now()
    device = BASE._require_idle_device(args.device, args.nvidia_smi)
    output.mkdir(parents=True)
    for directory in (output / "build", output / "source_snapshot"):
        directory.mkdir()
    snapshots = {}
    for path in (SOURCE_PATH, RUNNER_PATH, BASE_RUNNER_PATH):
        destination = output / "source_snapshot" / path.name
        shutil.copy2(path, destination)
        snapshots[path.name] = destination
    commit = BASE._run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout.strip()
    status = BASE._run(
        ["git", "status", "--short"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout
    (output / "baseline_commit.txt").write_text(commit + "\n")
    (output / "tracked_worktree_status.txt").write_text(status)

    tools = {
        "nvcc": BASE._tool_identity(
            args.nvcc, ("--version",), BASE.EXPECTED_NVCC_SHA256),
        "cuobjdump": BASE._tool_identity(
            args.cuobjdump, ("--version",), BASE.EXPECTED_CUOBJDUMP_SHA256),
        "nvidia_smi": BASE._tool_identity(
            args.nvidia_smi,
            ("--query-gpu=driver_version", "--format=csv,noheader")),
    }
    production_artifact = BASE._audit_production_artifact(
        args.production_binary, args.cuobjdump, output / "build")
    binary = output / "build/add_rmsnorm_queued_wave_floor"
    compile_command = [
        str(args.nvcc), "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo",
        str(snapshots[SOURCE_PATH.name]), "-o", str(binary),
    ]
    with (output / "build/compile.stdout").open("w") as stdout, (
            output / "build/compile.stderr").open("w") as stderr:
        BASE._run(
            compile_command, cwd=REPOSITORY_ROOT,
            stdout=stdout, stderr=stderr)
    resource_result = BASE._run([
        str(args.cuobjdump), "--dump-resource-usage", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True)
    resource_path = output / "build/probe.resources.txt"
    resource_path.write_text(resource_result.stdout)
    resources = BASE._parse_resource_usage(resource_result.stdout)
    sass_result = BASE._run([
        str(args.cuobjdump), "--dump-sass", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True)
    sass_path = output / "build/probe.sass"
    sass_path.write_text(sass_result.stdout)
    sass_contract = BASE._validate_sass(sass_result.stdout)

    raw_path = output / "raw.csv"
    native_command = [
        str(binary), str(args.device), str(EXPECTED_TRIALS),
        "queued_wave_floor",
    ]
    with raw_path.open("w") as stdout, (
            output / "native.stderr").open("w") as stderr:
        native_pid = BASE._run_native(
            native_command, cwd=REPOSITORY_ROOT,
            stdout=stdout, stderr=stderr)
    rows = load_raw(raw_path)
    candidate = derive_candidate(rows, resources, sass_contract)
    _write_json(output / "candidate.json", candidate)
    postflight_device = BASE._wait_for_idle_device(
        args.device, args.nvidia_smi)
    completed_at = _utc_now()
    provenance = {
        "schema": "nvidia_a800_add_rmsnorm_queued_wave_floor_capture_v1",
        "owner": args.owner,
        "runner_pid": os.getpid(),
        "native_pid": native_pid,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "baseline_commit": commit,
        "tracked_worktree_status": status,
        "command": shlex.join([sys.executable, *sys.argv]),
        "native_command": shlex.join(native_command),
        "device": device,
        "postflight_device": postflight_device,
        "preflight_cleanup": "target_device_idle_no_compute_processes",
        "postflight_cleanup": "target_device_idle_no_compute_processes",
        "tool_identities": tools,
        "source_sha256": _sha256(snapshots[SOURCE_PATH.name]),
        "runner_sha256": _sha256(snapshots[RUNNER_PATH.name]),
        "base_runner_sha256": _sha256(snapshots[BASE_RUNNER_PATH.name]),
        "probe_binary_sha256": _sha256(binary),
        "probe_resource_dump_sha256": _sha256(resource_path),
        "probe_sass_dump_sha256": _sha256(sass_path),
        "probe_resources": resources,
        "probe_sass_contract": sass_contract,
        "production_artifact": production_artifact,
        "candidate_sha256": _sha256(output / "candidate.json"),
        "raw_sha256": _sha256(raw_path),
        "production_or_evaluation_duration_consumed": False,
        "production_predictor_consumption_allowed": False,
    }
    _write_json(output / "provenance.json", provenance)
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--nvcc", type=Path, default=BASE.DEFAULT_NVCC)
    parser.add_argument("--cuobjdump", type=Path, default=BASE.DEFAULT_CUOBJDUMP)
    parser.add_argument("--nvidia-smi", type=Path, default=BASE.DEFAULT_NVIDIA_SMI)
    parser.add_argument(
        "--production-binary", type=Path,
        default=BASE.DEFAULT_PRODUCTION_BINARY)
    args = parser.parse_args()
    candidate = capture(args)
    print(json.dumps(candidate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
