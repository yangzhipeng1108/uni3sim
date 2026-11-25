#!/usr/bin/env python3
"""Audit three captures and validate the A800 non-recurrent SiLU envelope."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from silu_nonrecurrent_envelope_runner import (
    CELLS,
    EXPECTED_COMPUTE_CAPABILITY,
    EXPECTED_DEVICE_NAME,
    EXPECTED_RESOURCES,
    EXPECTED_SM_COUNT,
    EXPECTED_THREADS,
    GATES_PATH,
    GRIDS,
    MAXIMUM_IDLE_DRIVER_MEMORY_MIB,
    REQUIRED_INDEPENDENT_RUNS,
    SCHEDULE_NAME,
    TRIALS,
    VARIANTS,
    WIDTHS,
    _probe_resources,
    _probe_sass_contract,
    _source_math_spine,
    _validate_gates,
    cell_index_for,
)


RAW_FIELDS = (
    "run_id", "trial", "order_position", "cell_index", "variant",
    "grid_ctas", "intermediate_elements_per_cta", "loop_rounds_per_cta",
    "scalar_evaluations", "block_threads", "active_sms",
    "minimum_ctas_per_sm", "maximum_ctas_per_sm",
    "maximum_sm_span_cycles", "median_cta_cycles", "checksum")
MANIFEST_FIELDS = {
    "schema", "hardware_id", "owner", "run_id", "run_rotation",
    "runner_pid", "native_pid", "baseline_commit", "tracked_worktree_status",
    "command", "native_command", "device_preflight", "device_postflight",
    "schedule", "trials", "cycle_unit", "work_unit", "service_rate_unit",
    "source_sha256", "runner_sha256", "report_sha256", "gates_sha256",
    "probe_binary_sha256", "probe_resource_sha256", "probe_sass_sha256",
    "probe_resources", "probe_sass_contract", "production_binary_path",
    "production_binary_sha256", "production_cubin_sha256",
    "production_sass_sha256", "production_sass_contract", "raw_sha256",
    "production_or_evaluation_duration_consumed",
    "production_parameter_published",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_spread(values):
    maximum = max(values)
    return 0.0 if maximum == 0 else (maximum - min(values)) / maximum


def _validated_output(path):
    resolved = path.resolve()
    root = (
        Path(__file__).resolve().parents[4]
        / "profiler_output/microbench/nvidia_a800").resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a child of {root}")
    if resolved.exists():
        raise FileExistsError(f"output already exists: {resolved}")
    if not resolved.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {resolved.parent}")
    return resolved


def _read_native_csv(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = {}
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("meta,"):
            fields = next(csv.reader([line]))
            if len(fields) != 3 or fields[1] in metadata:
                raise ValueError(f"{path} has invalid metadata")
            metadata[fields[1]] = fields[2]
        elif line.startswith("raw,"):
            fields = next(csv.reader([line]))
            if tuple(fields[1:]) != RAW_FIELDS:
                raise ValueError(f"{path} has an invalid raw schema")
            header_index = index
            break
        else:
            raise ValueError(f"{path} has an unexpected preamble row")
    if header_index is None:
        raise ValueError(f"{path} has no raw header")
    reader = csv.DictReader(
        io.StringIO("\n".join(lines[header_index:])),
        fieldnames=("record", *RAW_FIELDS))
    next(reader)
    rows = []
    integer_fields = (
        "trial", "order_position", "cell_index", "grid_ctas",
        "intermediate_elements_per_cta", "loop_rounds_per_cta",
        "scalar_evaluations", "block_threads", "active_sms",
        "minimum_ctas_per_sm", "maximum_ctas_per_sm",
        "maximum_sm_span_cycles", "median_cta_cycles")
    for raw in reader:
        if raw["record"] != "raw":
            raise ValueError(f"{path} contains a non-raw row")
        integers = {name: int(raw[name]) for name in integer_fields}
        checksum = float(raw["checksum"])
        if not math.isfinite(checksum):
            raise ValueError(f"{path} contains a non-finite checksum")
        rows.append({
            "run_id": raw["run_id"], "variant": raw["variant"],
            **integers, "checksum": checksum})
    return metadata, rows


def _validate_device(manifest, field):
    row = manifest[field]
    if set(row) != {
            "index", "name", "uuid", "memory_used_mib",
            "utilization_gpu_pct", "compute_capability", "sm_clock_mhz",
            "memory_clock_mhz", "driver_version"}:
        raise ValueError(f"{field} device schema changed")
    if (row["name"] != EXPECTED_DEVICE_NAME
            or row["compute_capability"] != EXPECTED_COMPUTE_CAPABILITY
            or row["memory_used_mib"] > MAXIMUM_IDLE_DRIVER_MEMORY_MIB
            or row["utilization_gpu_pct"] != 0):
        raise ValueError(f"{field} does not identify an idle A800")
    return row


def _audit_capture(capture, gates):
    capture = capture.resolve()
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError(f"{capture} manifest schema changed")
    if (manifest["schema"]
            != "nvidia_a800_silu_nonrecurrent_envelope_capture_v2"
            or manifest["hardware_id"] != "nvidia_a800"
            or manifest["schedule"] != SCHEDULE_NAME
            or manifest["trials"] != TRIALS
            or manifest["cycle_unit"] != gates["cycle_unit"]
            or manifest["work_unit"] != gates["work_unit"]
            or manifest["service_rate_unit"] != gates["service_rate_unit"]
            or manifest["production_or_evaluation_duration_consumed"]
            or manifest["production_parameter_published"]):
        raise ValueError(f"{capture} manifest contract changed")
    rotation = manifest["run_rotation"]
    if rotation not in range(REQUIRED_INDEPENDENT_RUNS):
        raise ValueError(f"{capture} has an invalid rotation")
    preflight = _validate_device(manifest, "device_preflight")
    postflight = _validate_device(manifest, "device_postflight")
    if preflight["uuid"] != postflight["uuid"]:
        raise ValueError(f"{capture} changed devices")
    hash_paths = {
        "source_sha256": "source_snapshot/silu_nonrecurrent_envelope.cu",
        "runner_sha256": "source_snapshot/silu_nonrecurrent_envelope_runner.py",
        "report_sha256": "source_snapshot/silu_nonrecurrent_envelope_report.py",
        "gates_sha256": "source_snapshot/silu_nonrecurrent_envelope_gates.json",
        "probe_binary_sha256": "build/silu_nonrecurrent_envelope",
        "probe_resource_sha256": "build/probe_resource_usage.txt",
        "probe_sass_sha256": "build/probe.sass",
        "production_cubin_sha256": "build/production_silu_sm80.cubin",
        "production_sass_sha256": "build/production_silu_bf16.sass",
        "raw_sha256": "raw.csv",
    }
    for field, relative in hash_paths.items():
        if manifest[field] != _sha256(capture / relative):
            raise ValueError(f"{capture} has a stale {field}")
    if manifest["gates_sha256"] != _sha256(GATES_PATH):
        raise ValueError(f"{capture} was not captured with the frozen gates")
    resources = _probe_resources(
        (capture / "build/probe_resource_usage.txt").read_text(encoding="utf-8"))
    if resources != EXPECTED_RESOURCES or resources != manifest["probe_resources"]:
        raise ValueError(f"{capture} resource evidence disagrees")
    probe_contract = _probe_sass_contract(
        (capture / "build/probe.sass").read_text(encoding="utf-8"))
    production_contract = _source_math_spine(
        (capture / "build/production_silu_bf16.sass").read_text(encoding="utf-8"))
    if (probe_contract != manifest["probe_sass_contract"]
            or production_contract != manifest["production_sass_contract"]
            or probe_contract["source_math_spine"] != production_contract):
        raise ValueError(f"{capture} SASS identity evidence disagrees")

    metadata, rows = _read_native_csv(capture / "raw.csv")
    expected_metadata = {
        "device_index": "0", "device_name": EXPECTED_DEVICE_NAME,
        "compute_capability": EXPECTED_COMPUTE_CAPABILITY,
        "sm_count": str(EXPECTED_SM_COUNT),
        "threads_per_cta": str(EXPECTED_THREADS),
        "resident_cta_limit_per_sm": "2",
        "run_id": manifest["run_id"], "run_rotation": str(rotation),
        "schedule": SCHEDULE_NAME, "cycle_unit": gates["cycle_unit"],
        "work_unit": gates["work_unit"],
        "production_or_evaluation_duration_consumed": "false",
    }
    if metadata != expected_metadata:
        raise ValueError(f"{capture} native metadata changed")
    if len(rows) != TRIALS * len(CELLS):
        raise ValueError(f"{capture} raw row count changed")
    by_cell = {cell: [] for cell in CELLS}
    seen = set()
    for row in rows:
        trial_index = row["trial"] - 1
        position = row["order_position"]
        if trial_index not in range(TRIALS) or position not in range(len(CELLS)):
            raise ValueError(f"{capture} row is outside the schedule")
        expected_index = cell_index_for(trial_index, position, rotation)
        cell = (
            row["variant"], row["grid_ctas"],
            row["intermediate_elements_per_cta"])
        identity = (trial_index, position)
        expected_minimum = row["grid_ctas"] // EXPECTED_SM_COUNT
        expected_maximum = math.ceil(row["grid_ctas"] / EXPECTED_SM_COUNT)
        if (row["run_id"] != manifest["run_id"]
                or row["cell_index"] != expected_index
                or cell != CELLS[expected_index] or identity in seen
                or row["loop_rounds_per_cta"]
                != math.ceil(row["intermediate_elements_per_cta"] / EXPECTED_THREADS)
                or row["scalar_evaluations"]
                != row["grid_ctas"] * row["intermediate_elements_per_cta"]
                or row["block_threads"] != EXPECTED_THREADS
                or row["active_sms"] != EXPECTED_SM_COUNT
                or row["minimum_ctas_per_sm"] != expected_minimum
                or row["maximum_ctas_per_sm"] != expected_maximum
                or row["maximum_sm_span_cycles"] <= 0
                or row["median_cta_cycles"] <= 0):
            raise ValueError(f"{capture} violates its row contract")
        seen.add(identity)
        by_cell[cell].append(row["maximum_sm_span_cycles"])
    if any(len(values) != TRIALS for values in by_cell.values()):
        raise ValueError(f"{capture} does not replicate every cell")
    replication = {}
    within_limit = gates["capture_success_gates"][
        "within_run_interquartile_relative_spread_max"]
    for cell, values in by_cell.items():
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        spread = _relative_spread((q1, q3))
        replication[cell] = {
            "median_sm_clock_cycles": statistics.median(values),
            "q1_sm_clock_cycles": q1, "q3_sm_clock_cycles": q3,
            "interquartile_relative_spread": spread,
            "stability_gate": spread <= within_limit,
        }
    if not all(result["stability_gate"] for result in replication.values()):
        raise ValueError(f"{capture} failed within-run stability")
    return {
        "capture": str(capture), "manifest": manifest,
        "device_uuid": preflight["uuid"], "replication": replication,
    }


def _campaign(captures):
    gates = _validate_gates()
    if len(captures) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(f"campaign requires {REQUIRED_INDEPENDENT_RUNS} captures")
    audits = tuple(_audit_capture(capture, gates) for capture in captures)
    audits = tuple(sorted(audits, key=lambda audit: audit["manifest"]["run_rotation"]))
    manifests = tuple(audit["manifest"] for audit in audits)
    if tuple(manifest["run_rotation"] for manifest in manifests) != (0, 1, 2):
        raise ValueError("captures must cover rotations 0, 1, and 2")
    run_ids = tuple(manifest["run_id"] for manifest in manifests)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("capture run IDs must be distinct")
    identity_fields = (
        "baseline_commit", "source_sha256", "runner_sha256", "report_sha256",
        "gates_sha256", "probe_resources", "probe_sass_contract",
        "production_binary_sha256",
        "production_cubin_sha256", "production_sass_sha256",
        "production_sass_contract")
    for field in identity_fields:
        if len({json.dumps(manifest[field], sort_keys=True) for manifest in manifests}) != 1:
            raise ValueError(f"captures disagree on {field}")
    if len({audit["device_uuid"] for audit in audits}) != 1:
        raise ValueError("captures disagree on device UUID")

    cross_run = {}
    cross_limit = gates["capture_success_gates"][
        "cross_run_median_relative_spread_max"]
    for cell in CELLS:
        medians = tuple(
            audit["replication"][cell]["median_sm_clock_cycles"]
            for audit in audits)
        spread = _relative_spread(medians)
        cross_run[cell] = {
            "median_sm_clock_cycles_by_run": dict(zip(run_ids, medians)),
            "cross_run_relative_spread": spread,
            "stability_gate": spread <= cross_limit,
        }
    if not all(result["stability_gate"] for result in cross_run.values()):
        raise ValueError("campaign failed cross-run stability")
    maximum_within_run_spread = max(
        result["interquartile_relative_spread"]
        for audit in audits for result in audit["replication"].values())
    maximum_cross_run_spread = max(
        result["cross_run_relative_spread"] for result in cross_run.values())

    segmented = gates["segmented_service_model"]
    resident_ctas = segmented["resident_ctas_per_sm"]
    calibration_grids = tuple(segmented["calibration_grid_ctas"])
    validation_grids = tuple(segmented["validation_grid_ctas"])
    endpoint_widths = tuple(segmented["calibration_endpoint_widths"])
    service_segments = tuple(segmented["service_segments"])
    if (resident_ctas != 2
            or calibration_grids != (108, 216, 1024)
            or validation_grids != (256,)
            or endpoint_widths != (WIDTHS[0], WIDTHS[-1])
            or service_segments != ("first", "resident", "queued")):
        raise ValueError("segmented service model changed")

    def _calibration_cycles(grid, width):
        return statistics.median(tuple(
            audit["replication"][("source_silu", grid, width)][
                "median_sm_clock_cycles"]
            for audit in audits[:2]))

    def _services_at_width(width):
        first = _calibration_cycles(108, width)
        two_cta = _calibration_cycles(216, width)
        ten_cta = _calibration_cycles(1024, width)
        return {
            "first": first,
            "resident": two_cta - first,
            "queued": (ten_cta - two_cta) / 8.0,
        }

    endpoint_services = {
        width: _services_at_width(width) for width in endpoint_widths}
    endpoint_rounds = tuple(
        math.ceil(width / EXPECTED_THREADS) for width in endpoint_widths)
    service_parameters = {}
    for segment in service_segments:
        lower = endpoint_services[endpoint_widths[0]][segment]
        upper = endpoint_services[endpoint_widths[1]][segment]
        round_cycles = (
            (upper - lower) / (endpoint_rounds[1] - endpoint_rounds[0]))
        entry_cycles = lower - endpoint_rounds[0] * round_cycles
        if entry_cycles <= 0 or round_cycles <= 0:
            raise ValueError(
                f"{segment} affine service is not physically positive")
        service_parameters[segment] = {
            "entry_cycles_per_cta": entry_cycles,
            "round_cycles_per_cta": round_cycles,
            "round_service_rate_per_sm_cycle": 1.0 / round_cycles,
        }

    def _predict_cycles(grid, width):
        ctas_per_sm = math.ceil(grid / EXPECTED_SM_COUNT)
        rounds = math.ceil(width / EXPECTED_THREADS)
        def _service(segment):
            parameters = service_parameters[segment]
            return (
                parameters["entry_cycles_per_cta"]
                + rounds * parameters["round_cycles_per_cta"])
        return (
            _service("first")
            + (min(ctas_per_sm, resident_ctas) - 1)
            * _service("resident")
            + max(0, ctas_per_sm - resident_ctas)
            * _service("queued"))

    held_out = audits[2]
    held_out_rows = {}
    heldout_limit = gates["capture_success_gates"][
        "held_out_segmented_duration_ape_max"]
    for grid in GRIDS:
        for width in WIDTHS:
            observed = held_out["replication"][("source_silu", grid, width)][
                "median_sm_clock_cycles"]
            predicted = _predict_cycles(grid, width)
            ape = abs(predicted - observed) / observed
            held_out_rows[(grid, width)] = {
                "observed_sm_clock_cycles": observed,
                "predicted_sm_clock_cycles": predicted,
                "absolute_percentage_error": ape,
                "validation_gate": ape <= heldout_limit,
            }
    source_to_control_ratios = []
    for grid in GRIDS:
        for width in WIDTHS:
            source = statistics.median(tuple(
                audit["replication"][("source_silu", grid, width)][
                    "median_sm_clock_cycles"] for audit in audits))
            control = statistics.median(tuple(
                audit["replication"][("arithmetic_control", grid, width)][
                    "median_sm_clock_cycles"] for audit in audits))
            if source <= control:
                raise ValueError(
                    f"source math is not distinguishable from control at grid={grid}, width={width}")
            source_to_control_ratios.append(source / control)

    heldout_errors = tuple(
        result["absolute_percentage_error"] for result in held_out_rows.values())
    segmented_service_gate = max(heldout_errors) <= heldout_limit
    decision = (
        "positive_mechanism_candidate" if segmented_service_gate
        else "negative_evidence_only")

    return {
        "schema": "nvidia_a800_silu_nonrecurrent_envelope_audit_v2",
        "hardware_id": "nvidia_a800", "run_ids": list(run_ids),
        "capture_paths": [audit["capture"] for audit in audits],
        "capture_artifact_sha256": {
            manifest["run_id"]: {
                "probe_binary_sha256": manifest["probe_binary_sha256"],
                "probe_resource_sha256": manifest["probe_resource_sha256"],
                "probe_sass_sha256": manifest["probe_sass_sha256"],
                "raw_sha256": manifest["raw_sha256"],
            }
            for manifest in manifests},
        "device_uuid": audits[0]["device_uuid"],
        "baseline_commit": manifests[0]["baseline_commit"],
        "units": {
            "cycle": gates["cycle_unit"], "work": gates["work_unit"],
            "service_rate": gates["service_rate_unit"]},
        "probe_variants": list(VARIANTS), "threads_per_cta": EXPECTED_THREADS,
        "intermediate_elements_per_cta": list(WIDTHS), "grid_ctas": list(GRIDS),
        "trials_per_cell_per_run": TRIALS,
        "position_balanced_cyclic_schedule": True,
        "within_run_stability_gate": True, "cross_run_stability_gate": True,
        "maximum_within_run_interquartile_relative_spread":
            maximum_within_run_spread,
        "maximum_cross_run_median_relative_spread": maximum_cross_run_spread,
        "calibration_rotations": [0, 1], "held_out_rotation": 2,
        "segmented_service_model": {
            "resident_ctas_per_sm": resident_ctas,
            "calibration_grid_ctas": list(calibration_grids),
            "validation_grid_ctas": list(validation_grids),
            "calibration_endpoint_widths": list(endpoint_widths),
            "service_parameters": service_parameters,
            "formula": (
                "service(segment, rounds) = entry_cycles_per_cta + "
                "rounds * round_cycles_per_cta; total = first + "
                "(min(ceil(grid/108), 2) - 1) * resident + "
                "max(0, ceil(grid/108) - 2) * queued"),
        },
        "segmented_service_gate": segmented_service_gate,
        "held_out_validation": {
            f"grid{grid}/width{width}": result
            for (grid, width), result in held_out_rows.items()},
        "held_out_validation_gate": segmented_service_gate,
        "held_out_mean_absolute_percentage_error": statistics.mean(heldout_errors),
        "held_out_maximum_absolute_percentage_error": max(heldout_errors),
        "source_silu_slower_than_arithmetic_control_for_every_cell": True,
        "source_to_arithmetic_control_duration_ratio_minimum":
            min(source_to_control_ratios),
        "source_to_arithmetic_control_duration_ratio_maximum":
            max(source_to_control_ratios),
        "probe_and_production_source_math_spine_identical": True,
        "audited_semantic_identity": {
            "probe_resources": manifests[0]["probe_resources"],
            "probe_sass_contract": manifests[0]["probe_sass_contract"],
            "production_sass_contract": manifests[0]["production_sass_contract"],
        },
        "decision": decision,
        "mechanism_publishable": segmented_service_gate,
        "forward_eligible": segmented_service_gate,
        "bounded_conclusion": (
            "The register-resident non-recurrent source SiLU math sequence is "
            "stable within and across captures, and its occupancy-segmented "
            "first/resident/queued affine service validates without "
            "production durations."
            if segmented_service_gate else
            "The register-resident non-recurrent source SiLU math sequence is "
            "stable within and across captures, but its occupancy-segmented "
            "affine service fails the frozen held-out duration gate."),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
        "production_predictor_consumption_allowed": segmented_service_gate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = _campaign(args.capture)
    output = _validated_output(args.output)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print("within_run_stability_gate=pass")
    print("cross_run_stability_gate=pass")
    print(
        "held_out_validation_gate="
        f"{'pass' if audit['held_out_validation_gate'] else 'fail'}")
    print(f"decision={audit['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
