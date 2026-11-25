#!/usr/bin/env python3
"""Derive the source-bound A800 Add-RMSNorm reduction contract."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


RUNNER_PATH = Path(__file__).with_name("block_reduce_latency_runner.py")
SOURCE_PATH = Path(__file__).with_name("block_reduce_latency.cu")
REPORT_PATH = Path(__file__).resolve()
EXPECTED_FIELDS = (
    "run_id", "trial", "position", "physical_device", "device_uuid",
    "pci_bus_id", "driver_version", "pstate", "sm_clock_before_mhz",
    "sm_clock_after_mhz", "memory_clock_before_mhz",
    "memory_clock_after_mhz", "sample", "device_ordinal", "sm_count",
    "threads", "grid", "block_median_cycles", "sm_local_span_cycles",
    "grid_span_ns", "validation_checksum", "validation_status",
)
EXPECTED_TRIALS = (0, 1, 2)
TRAINING_TRIALS = (0, 1)
VALIDATION_TRIALS = (2,)
BLOCK_THREADS = (256, 1024)
SM_COUNT = 108
GRID_MULTIPLIERS = (0, 1, 2, 4, 8, 16)
GRIDS = tuple(
    1 if multiplier == 0 else SM_COUNT * multiplier
    for multiplier in GRID_MULTIPLIERS)
SAMPLES_PER_CELL = 200
ROTATION_STRIDE = 7
MAX_VALIDATION_MAPE_PERCENT = 10.0
MAX_VALIDATION_MAX_APE_PERCENT = 30.0


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(text, name, *, positive=False, non_negative=False):
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(value) != text:
        raise ValueError(f"{name} must be a canonical integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_float(text, name):
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _read_json_object(path, name):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _expected_schedule():
    cells = tuple(
        (threads, grid) for threads in BLOCK_THREADS for grid in GRIDS)
    schedule = {}
    for trial in EXPECTED_TRIALS:
        offset = (trial * ROTATION_STRIDE) % len(cells)
        for position in range(len(cells)):
            threads, grid = cells[(position + offset) % len(cells)]
            schedule[(trial, threads, grid)] = position
    return schedule


def validate_manifest(path, raw_path):
    manifest = _read_json_object(path, "capture manifest")
    if manifest["schema"] != "a800_block_reduce_latency_capture_v3":
        raise ValueError("capture manifest schema is unsupported")
    if (not isinstance(manifest["baseline_commit"], str)
            or re.fullmatch(r"[0-9a-f]{40}", manifest["baseline_commit"])
            is None):
        raise ValueError("capture manifest baseline commit is invalid")
    if manifest["production_duration_consumed"]:
        raise ValueError("capture manifest consumed production duration")
    if tuple(manifest["block_threads"]) != BLOCK_THREADS:
        raise ValueError("capture block-thread domain changed")
    if tuple(manifest["grid_multipliers"]) != GRID_MULTIPLIERS:
        raise ValueError("capture grid domain changed")
    if manifest["samples_per_cell"] != SAMPLES_PER_CELL:
        raise ValueError("capture samples-per-cell changed")
    if manifest["trials"] != len(EXPECTED_TRIALS):
        raise ValueError("capture trial count changed")
    if manifest["sm_count"] != SM_COUNT:
        raise ValueError("capture SM count changed")
    if manifest["timer_domains"] != {
            "block_median_cycles": "clock64_sm_cycles",
            "sm_local_span_cycles": "clock64_same_smid_cycles",
            "grid_span_ns": "globaltimer_ns"}:
        raise ValueError("capture timer domain changed")
    for field, current_path in (
            ("source_sha256", SOURCE_PATH),
            ("runner_sha256", RUNNER_PATH),
            ("reporter_sha256", REPORT_PATH)):
        if manifest[field] != _sha256(current_path):
            raise ValueError(f"capture {field} differs from the current source")
    if manifest["raw_sha256"] != _sha256(raw_path):
        raise ValueError("capture raw SHA256 mismatch")
    proof_path = path.parent / "static_equivalence.json"
    if not proof_path.is_file():
        raise ValueError("capture static-equivalence proof is missing")
    if manifest["static_equivalence_sha256"] != _sha256(proof_path):
        raise ValueError("capture static-equivalence proof SHA256 mismatch")
    proof = _read_json_object(proof_path, "static-equivalence proof")
    if (proof["schema"]
            != "a800_add_rmsnorm_reduction_static_equivalence_v2"
            or proof["status"] != "pass"):
        raise ValueError("capture static-equivalence proof did not pass")
    for timer_name, timer_domain in (
            ("globaltimer", "globaltimer_ns"),
            ("clock64", "clock64_sm_cycles")):
        timer_proof = proof[timer_name]
        if (timer_proof["status"] != "pass"
                or timer_proof["timer_domain"] != timer_domain
                or timer_proof["global_memory_in_timed_region"]
                or timer_proof["other_timer_domain_in_timed_region"]
                or not timer_proof["normal_fast_path_closed"]):
            raise ValueError(
                f"capture {timer_name} static proof did not pass")
    resource_usage = proof["resource_usage"]
    if (resource_usage["registers_per_thread"] != 24
            or resource_usage["shared_memory_bytes_per_cta"] != 168
            or resource_usage["sm_register_capacity"] != 65_536
            or resource_usage["sm_shared_memory_bytes"] != 163_840
            or resource_usage["sm_thread_capacity"] != 2_048
            or resource_usage["sm_block_capacity"] != 32
            or resource_usage["resident_ctas_by_threads"]
            != {"256": 8, "1024": 2}):
        raise ValueError("capture static resource/occupancy proof changed")
    production = manifest["production_artifact"]
    if (production["extension_sha256"]
            != "984de21ce38cc8083d207663a380ca75acc6ad3dc230858092f59e363c4eebd5"
            or production["cub_version"] != 200700):
        raise ValueError("capture production artifact identity changed")
    toolchain = manifest["toolchain"]
    if toolchain["cub_version"] != 200700:
        raise ValueError("capture toolchain CUB identity changed")
    return manifest, proof


def read_raw(path, manifest):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise ValueError(
                "A800 block-reduction raw fields disagree with the schema")
        rows = []
        identities = set()
        expected_schedule = _expected_schedule()
        device = manifest["device"]
        for ordinal, raw in enumerate(reader, start=2):
            row = {
                "trial": _integer(
                    raw["trial"], f"line {ordinal}.trial", non_negative=True),
                "position": _integer(
                    raw["position"], f"line {ordinal}.position",
                    non_negative=True),
                "sample": _integer(
                    raw["sample"], f"line {ordinal}.sample",
                    non_negative=True),
                "sm_count": _integer(
                    raw["sm_count"], f"line {ordinal}.sm_count",
                    positive=True),
                "threads": _integer(
                    raw["threads"], f"line {ordinal}.threads",
                    positive=True),
                "grid": _integer(
                    raw["grid"], f"line {ordinal}.grid", positive=True),
                "block_median_cycles": _integer(
                    raw["block_median_cycles"],
                    f"line {ordinal}.block_median_cycles", positive=True),
                "sm_local_span_cycles": _integer(
                    raw["sm_local_span_cycles"],
                    f"line {ordinal}.sm_local_span_cycles", positive=True),
                "grid_span_ns": _integer(
                    raw["grid_span_ns"], f"line {ordinal}.grid_span_ns",
                    non_negative=True),
            }
            physical_device = _integer(
                raw["physical_device"], f"line {ordinal}.physical_device",
                non_negative=True)
            device_ordinal = _integer(
                raw["device_ordinal"], f"line {ordinal}.device_ordinal",
                non_negative=True)
            sm_clock_before = _integer(
                raw["sm_clock_before_mhz"],
                f"line {ordinal}.sm_clock_before_mhz", positive=True)
            sm_clock_after = _integer(
                raw["sm_clock_after_mhz"],
                f"line {ordinal}.sm_clock_after_mhz", positive=True)
            memory_clock_before = _integer(
                raw["memory_clock_before_mhz"],
                f"line {ordinal}.memory_clock_before_mhz", positive=True)
            memory_clock_after = _integer(
                raw["memory_clock_after_mhz"],
                f"line {ordinal}.memory_clock_after_mhz", positive=True)
            if raw["run_id"] != manifest["run_id"]:
                raise ValueError(f"line {ordinal} run_id differs from manifest")
            if (physical_device != manifest["physical_device"]
                    or device_ordinal != 0
                    or raw["device_uuid"] != device["device_uuid"]
                    or raw["pci_bus_id"] != device["pci_bus_id"]
                    or raw["driver_version"] != device["driver_version"]):
                raise ValueError(f"line {ordinal} device identity changed")
            if sm_clock_before != sm_clock_after:
                raise ValueError(f"line {ordinal} SM clock changed within a cell")
            if memory_clock_before != memory_clock_after:
                raise ValueError(
                    f"line {ordinal} memory clock changed within a cell")
            row["sm_clock_mhz"] = sm_clock_before
            if not raw["pstate"]:
                raise ValueError(f"line {ordinal} pstate is empty")
            if raw["validation_status"] != "pass":
                raise ValueError(
                    f"line {ordinal} has a failed device validation")
            _positive_float(
                raw["validation_checksum"],
                f"line {ordinal}.validation_checksum")
            if row["sm_count"] != SM_COUNT:
                raise ValueError(
                    f"line {ordinal} did not execute on {SM_COUNT} SMs")
            schedule_key = (row["trial"], row["threads"], row["grid"])
            if (schedule_key not in expected_schedule
                    or expected_schedule[schedule_key] != row["position"]):
                raise ValueError(f"line {ordinal} schedule position changed")
            identity = (
                row["trial"], row["threads"], row["grid"], row["sample"])
            if identity in identities:
                raise ValueError(
                    f"duplicate A800 block-reduction sample: {identity}")
            identities.add(identity)
            rows.append(row)

    expected = {
        (trial, threads, grid, sample)
        for trial in EXPECTED_TRIALS
        for threads in BLOCK_THREADS
        for grid in GRIDS
        for sample in range(SAMPLES_PER_CELL)
    }
    if identities != expected:
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise ValueError(
            "A800 block-reduction raw matrix is incomplete: "
            f"missing={missing[:4]}, extra={extra[:4]}")
    clock_values = {row["sm_clock_mhz"] for row in rows}
    if len(clock_values) != 1:
        raise ValueError(
            "A800 block-reduction capture changed SM clocks between cells")
    return tuple(rows)


def _parameters(rows, resident_ctas_by_threads):
    training = tuple(
        row for row in rows if row["trial"] in TRAINING_TRIALS)
    maximum_grid = GRIDS[-1]
    maximum_rounds = maximum_grid // SM_COUNT
    if maximum_rounds <= 1:
        raise AssertionError("steady-service endpoint requires multiple rounds")
    parameters = []
    for threads in BLOCK_THREADS:
        resident_ctas = resident_ctas_by_threads[str(threads)]
        resident_grid = SM_COUNT * resident_ctas
        first_block_cycles = statistics.median(
            row["block_median_cycles"] for row in training
            if row["threads"] == threads and row["grid"] == 1)
        sm_clock_mhz = statistics.median(
            row["sm_clock_mhz"] for row in training
            if row["threads"] == threads and row["grid"] == 1)
        latency_ns = first_block_cycles * 1000.0 / sm_clock_mhz
        resident_span_cycles = statistics.median(
            row["sm_local_span_cycles"] for row in training
            if row["threads"] == threads and row["grid"] == resident_grid)
        maximum_span_cycles = statistics.median(
            row["sm_local_span_cycles"] for row in training
            if row["threads"] == threads and row["grid"] == maximum_grid)
        resident_service_cycles = (
            (resident_span_cycles - first_block_cycles)
            / (resident_ctas - 1))
        queued_service_cycles = (
            (maximum_span_cycles - resident_span_cycles)
            / (maximum_rounds - resident_ctas))
        if not 0 < resident_service_cycles < first_block_cycles:
            raise ValueError(
                "resident block-reduction endpoints do not identify a "
                f"positive hidden-latency service for {threads} threads: "
                f"latency_cycles={first_block_cycles}, "
                f"service_cycles={resident_service_cycles}")
        if not 0 < queued_service_cycles < first_block_cycles:
            raise ValueError(
                "queued block-reduction endpoints do not identify a positive "
                f"service for {threads} threads: "
                f"service_cycles={queued_service_cycles}")
        resident_service_ns = (
            resident_service_cycles * 1000.0 / sm_clock_mhz)
        queued_service_ns = queued_service_cycles * 1000.0 / sm_clock_mhz
        parameters.append({
            "threads": threads,
            "resident_ctas_per_sm": resident_ctas,
            "first_block_latency_cycles": first_block_cycles,
            "measured_sm_clock_mhz": sm_clock_mhz,
            "first_block_latency_ns": latency_ns,
            "resident_block_service_cycles_per_sm": resident_service_cycles,
            "resident_block_service_ns_per_sm": resident_service_ns,
            "resident_block_rate_per_sm_s": 1.0e9 / resident_service_ns,
            "queued_block_service_cycles_per_sm": queued_service_cycles,
            "queued_block_service_ns_per_sm": queued_service_ns,
            "queued_block_rate_per_sm_s": 1.0e9 / queued_service_ns,
        })
    return tuple(parameters)


def _predict_grid_span_cycles(row, by_threads):
    parameters = by_threads[row["threads"]]
    ctas_per_sm = (
        1 if row["grid"] == 1 else math.ceil(row["grid"] / SM_COUNT))
    resident_ctas = parameters["resident_ctas_per_sm"]
    return (
        parameters["first_block_latency_cycles"]
        + (min(ctas_per_sm, resident_ctas) - 1)
        * parameters["resident_block_service_cycles_per_sm"]
        + max(0, ctas_per_sm - resident_ctas)
        * parameters["queued_block_service_cycles_per_sm"])


def _metrics(rows, by_threads):
    errors = []
    absolute_errors = []
    absolute_percentages = []
    observed_total = 0.0
    for row in rows:
        observed = row["sm_local_span_cycles"]
        predicted = _predict_grid_span_cycles(row, by_threads)
        error = predicted - observed
        errors.append(error / observed)
        absolute_errors.append(abs(error))
        absolute_percentages.append(abs(error) / observed)
        observed_total += observed
    return {
        "sample_count": len(rows),
        "mape_percent": statistics.fmean(absolute_percentages) * 100.0,
        "wmae_percent": sum(absolute_errors) / observed_total * 100.0,
        "signed_mean_percentage_error": statistics.fmean(errors) * 100.0,
        "maximum_absolute_percentage_error": (
            max(absolute_percentages) * 100.0),
    }


def build_report(path, manifest_path):
    manifest, proof = validate_manifest(manifest_path, path)
    rows = read_raw(path, manifest)
    parameters = _parameters(
        rows, proof["resource_usage"]["resident_ctas_by_threads"])
    by_threads = {row["threads"]: row for row in parameters}
    training = tuple(
        row for row in rows if row["trial"] in TRAINING_TRIALS)
    validation = tuple(
        row for row in rows if row["trial"] in VALIDATION_TRIALS)
    diagnostics = {
        "training": _metrics(training, by_threads),
        "validation": _metrics(validation, by_threads),
    }
    validation_metrics = diagnostics["validation"]
    if validation_metrics["mape_percent"] > MAX_VALIDATION_MAPE_PERCENT:
        raise ValueError("mechanism validation MAPE exceeds the frozen gate")
    if (validation_metrics["maximum_absolute_percentage_error"]
            > MAX_VALIDATION_MAX_APE_PERCENT):
        raise ValueError("mechanism validation maximum APE exceeds the frozen gate")
    return {
        "schema_version": 2,
        "status": "candidate",
        "hardware_id": "nvidia_a800",
        "mechanism": "source_bound_add_rmsnorm_reduction_and_publication",
        "duration_source": (
            "clock64_same_smid_span_with_globaltimer_diagnostic"),
        "production_or_evaluation_duration_consumed": False,
        "validity_domain": {
            "dtype": "bfloat16",
            "vector_width_elements": 8,
            "block_threads": list(BLOCK_THREADS),
            "hidden_size": 4096,
            "normal_fdiv_path": True,
        },
        "input": {
            "path": str(path),
            "sha256": _sha256(path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "static_equivalence_sha256": manifest[
                "static_equivalence_sha256"],
            "baseline_commit": manifest["baseline_commit"],
            "production_extension_sha256": manifest[
                "production_artifact"]["extension_sha256"],
            "toolchain_nvcc_sha256": manifest["toolchain"]["nvcc_sha256"],
            "trials": list(EXPECTED_TRIALS),
            "training_trials": list(TRAINING_TRIALS),
            "validation_trials": list(VALIDATION_TRIALS),
            "samples_per_cell": SAMPLES_PER_CELL,
            "block_threads": list(BLOCK_THREADS),
            "grids": list(GRIDS),
        },
        "formula": (
            "first_block_latency_cycles[threads] + "
            "(min(max(1, ceil(grid / 108)), resident_ctas) - 1) * "
            "resident_block_service_cycles_per_sm[threads] + "
            "max(0, ceil(grid / 108) - resident_ctas) * "
            "queued_block_service_cycles_per_sm[threads]; then multiply "
            "by 1000 / measured_sm_clock_mhz for nanoseconds"),
        "derivation": (
            "first-block latency is the pooled clock64 median at grid=1; "
            "resident service is the incremental same-SMID clock64 span from "
            "one CTA to the static-resource occupancy boundary, and queued "
            "service is the increment from that boundary to 16 CTAs/SM. "
            "Resident CTA counts 8/2 follow only from 24 registers/thread, "
            "168 B SMEM/CTA and A800 SM capacities. CUDA blocks do not migrate "
            "between SMs. Cycles are converted by the bracketed stable SM "
            "clock. Globaltimer is retained only as a quantized diagnostic. "
            "No production/evaluation duration or residual is used"),
        "parameters": list(parameters),
        "diagnostics": diagnostics,
        "publication_gate": {
            "maximum_validation_mape_percent": MAX_VALIDATION_MAPE_PERCENT,
            "maximum_validation_maximum_ape_percent": (
                MAX_VALIDATION_MAX_APE_PERCENT),
            "passed": True,
        },
        "static_equivalence": {
            "schema": proof["schema"],
            "status": proof["status"],
            "globaltimer_status": proof["globaltimer"]["status"],
            "clock64_status": proof["clock64"]["status"],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.input, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
