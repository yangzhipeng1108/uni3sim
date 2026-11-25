#!/usr/bin/env python3
"""Plan and audit the C500 low-wave program-span mechanism campaign."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import statistics


SCHEMA_VERSION = 1
HARDWARE_ID = "metax_c500"
MECHANISM = "mctracer_register_program_span"
TIMER_DOMAIN = "mctracer_gpu_kernel_event"
NUM_COMPUTE_UNITS = 104
PARALLEL_WIDTH = 64
WARMUP_LAUNCHES_PER_CONFIGURATION = 8
BODIES = ("looped", "unrolled", "cold")
COUNTS = (1, 8, 32, 128, 512)
GRIDS = (1, 4, 104)
BLOCKS = (64, 256, 512, 1024)
FULL_REPEATS = 15
SMOKE_REPEATS = 3
FULL_TRIALS = (1, 2, 3)
TRAINING_TRIALS = (1, 2)
VALIDATION_TRIALS = (3,)
# Every retained mcTracer event is an integer multiple of 256 ns.  The tool
# does not document whether the timestamp is rounded or truncated, so one
# endpoint is conservatively treated as uncertain by a complete tick.  A
# difference of two independently timestamped bodies therefore has a
# two-tick interval.
MCTRACER_QUANTIZATION_NS = 256.0
DIFFERENCE_UNCERTAINTY_NS = 2.0 * MCTRACER_QUANTIZATION_NS
INFERENCE_COUNT = 512
SECONDARY_VALIDATION_COUNT = 128
PLAN_FIELDS = (
    "sample_id", "trial", "order_position", "body", "count",
    "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix(mode):
    if mode == "smoke":
        return BODIES, (1, 128, 512), (1, 104), (64, 512, 1024), SMOKE_REPEATS
    if mode == "full":
        return BODIES, COUNTS, GRIDS, BLOCKS, FULL_REPEATS
    raise ValueError(f"unknown program-span campaign mode: {mode!r}")


def build_plan(trial, mode):
    if isinstance(trial, bool) or not isinstance(trial, int) or trial <= 0:
        raise ValueError("program-span trial must be a positive integer")
    bodies, counts, grids, blocks, repeats = _matrix(mode)
    scheduled = []
    for repeat in range(1, repeats + 1):
        for body in bodies:
            for count in counts:
                for grid in grids:
                    for block in blocks:
                        scheduled.append((
                            repeat, body, count, grid, block))
    random.Random(500372 + 7919 * trial).shuffle(scheduled)
    rows = []
    for order_position, (repeat, body, count, grid, block) in enumerate(
            scheduled):
        sample_id = (
            f"{body}_c{count}_g{grid}_b{block}_"
            f"t{trial}_r{repeat}"
        )
        if len(sample_id) > 96:
            raise AssertionError("program-span sample ID is too long")
        rows.append({
            "sample_id": sample_id,
            "trial": trial,
            "order_position": order_position,
            "body": body,
            "count": count,
            "grid_x": grid,
            "grid_y": 1,
            "grid_z": 1,
            "block_x": block,
            "block_y": 1,
            "block_z": 1,
        })
    return tuple(rows)


def plan_text(rows):
    return "".join(
        " ".join(str(row[field]) for field in PLAN_FIELDS) + "\n"
        for row in rows
    )


def _configuration(row):
    return tuple(row[field] for field in (
        "body", "count", "grid_x", "grid_y", "grid_z",
        "block_x", "block_y", "block_z",
    ))


def expected_event_rows(plan):
    body_order = {body: index for index, body in enumerate(BODIES)}
    configurations = sorted(
        {_configuration(row) for row in plan},
        key=lambda value: (body_order[value[0]],) + value[1:],
    )
    warmups = tuple(
        configuration
        for configuration in configurations
        for _ in range(WARMUP_LAUNCHES_PER_CONFIGURATION)
    )
    samples = tuple(_configuration(row) for row in plan)
    return warmups + samples


def validate_binary_stdout(path, plan):
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    if len(rows) != len(plan) + 3:
        raise ValueError("program-span binary stdout row count is invalid")
    if len(rows[0]) != 1 or rows[0][0].strip() != "mctxAutoStart called!":
        raise ValueError("program-span mcTracer injection banner is invalid")
    prefix = "mcToolsExtPid_UsedPort:"
    if len(rows[1]) != 1 or not rows[1][0].startswith(prefix):
        raise ValueError("program-span mcTracer endpoint is invalid")
    endpoint = rows[1][0].removeprefix(prefix).split("-")
    if len(endpoint) != 2 or any(not value.isdigit() for value in endpoint):
        raise ValueError("program-span mcTracer endpoint is invalid")
    metadata = rows[2]
    if (len(metadata) != 8 or metadata[0] != "metadata"
            or metadata[1] != "1" or metadata[2] != "MetaX C500"
            or metadata[5] != str(NUM_COMPUTE_UNITS)
            or metadata[6] != str(PARALLEL_WIDTH)
            or not metadata[7].isdigit() or int(metadata[7]) <= 0):
        raise ValueError("program-span binary metadata is invalid")
    for output, planned in zip(rows[3:], plan):
        expected = (
            "sample", planned["sample_id"], str(planned["trial"]),
            str(planned["order_position"]), planned["body"],
            str(planned["count"]), str(planned["grid_x"]),
            str(planned["grid_y"]), str(planned["grid_z"]),
            str(planned["block_x"]), str(planned["block_y"]),
            str(planned["block_z"]),
        )
        if tuple(output) != expected:
            raise ValueError(
                "program-span binary output differs from the frozen plan")


def _require_positive_int(mapping, key, context):
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return value


def _event_body(name):
    if "looped_chain_kernel" in name:
        return "looped"
    if "unrolled_chain_kernel" in name:
        return "unrolled"
    if "cold_footprint_kernel" in name:
        return "cold"
    raise ValueError(f"unknown program-span kernel identity: {name!r}")


def _template_count(name, body):
    if body == "looped":
        return None
    if body == "unrolled":
        token = "unrolled_chain_kernel"
    elif body == "cold":
        token = "cold_footprint_kernel"
    else:
        raise ValueError(f"unknown program-span body: {body!r}")
    match = re.search(re.escape(token) + r"(?:<|ILi)([0-9]+)", name)
    if match is None:
        raise ValueError(
            f"program-span template count is absent from {name!r}")
    return int(match.group(1))


def parse_trace(path, plan, device_id):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        events = payload["traceEvents"]
    elif isinstance(payload, list):
        events = payload
    else:
        raise TypeError("program-span mcTracer root must be a list or mapping")
    if not isinstance(events, list):
        raise TypeError("program-span traceEvents must be a list")
    selected = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise TypeError(f"program-span trace event {index} is not a mapping")
        if "name" not in event or not isinstance(event["name"], str):
            continue
        name = event["name"]
        if not any(token in name for token in (
                "looped_chain_kernel", "unrolled_chain_kernel",
                "cold_footprint_kernel")):
            continue
        if event["ph"] != "X":
            raise ValueError("program-span kernel event must be complete")
        args = event["args"]
        if not isinstance(args, dict):
            raise TypeError("program-span kernel args must be a mapping")
        if _require_positive_int(args, "device_id", "kernel args") != device_id:
            raise ValueError("program-span trace changed physical device")
        grid = args["grid"]
        block = args["block"]
        memory = args["mem"]
        if not isinstance(grid, dict) or not isinstance(block, dict):
            raise TypeError("program-span launch dimensions must be mappings")
        if not isinstance(memory, dict):
            raise TypeError("program-span resource metadata must be a mapping")
        duration_ns = event["dur"]
        if (isinstance(duration_ns, bool)
                or not isinstance(duration_ns, (int, float))
                or not math.isfinite(float(duration_ns))
                or float(duration_ns) <= 0):
            raise ValueError("program-span duration must be finite and positive")
        selected.append({
            "name": name,
            "body": _event_body(name),
            "template_count": _template_count(name, _event_body(name)),
            "grid": tuple(_require_positive_int(
                grid, axis, "kernel grid") for axis in ("x", "y", "z")),
            "block": tuple(_require_positive_int(
                block, axis, "kernel block") for axis in ("x", "y", "z")),
            "registers_per_thread": _require_positive_int(
                memory, "registers_per_thread", "kernel memory"),
            "dynamic_shared_bytes": memory["dynamic_shared"],
            "static_shared_bytes": memory["static_shared"],
            "duration_ns": float(duration_ns),
        })
    expected = expected_event_rows(plan)
    if len(selected) != len(expected):
        raise ValueError(
            "program-span trace has an unexpected kernel event count: "
            f"expected={len(expected)}, actual={len(selected)}")
    for ordinal, (event, configuration) in enumerate(zip(selected, expected)):
        body, count, *dimensions = configuration
        expected_grid = tuple(dimensions[:3])
        expected_block = tuple(dimensions[3:])
        if event["body"] != body:
            raise ValueError(
                f"program-span body differs at ordinal {ordinal}")
        if (event["template_count"] is not None
                and event["template_count"] != count):
            raise ValueError(
                f"program-span template count differs at ordinal {ordinal}")
        if event["grid"] != expected_grid or event["block"] != expected_block:
            raise ValueError(
                f"program-span geometry differs at ordinal {ordinal}")
        if (event["dynamic_shared_bytes"] != 0
                or event["static_shared_bytes"] != 0):
            raise ValueError("program-span benchmark must be register-only")
    sample_events = selected[len(selected) - len(plan):]
    return tuple({
        "sample_id": row["sample_id"],
        "trial": row["trial"],
        "body": row["body"],
        "count": row["count"],
        "grid": list(event["grid"]),
        "block": list(event["block"]),
        "waves_per_cta": math.ceil(math.prod(event["block"]) / PARALLEL_WIDTH),
        "registers_per_thread": event["registers_per_thread"],
        "duration_ns": event["duration_ns"],
        "kernel_name": event["name"],
    } for row, event in zip(plan, sample_events))


def build_report(mode, trial, plan_path, stdout_path, trace_path, device_id):
    expected = build_plan(trial, mode)
    if plan_path.read_text(encoding="utf-8") != plan_text(expected):
        raise ValueError("program-span plan differs from the frozen generator")
    validate_binary_stdout(stdout_path, expected)
    samples = parse_trace(trace_path, expected, device_id)
    cell_medians = []
    identities = sorted({
        (sample["body"], sample["count"], tuple(sample["grid"]),
         tuple(sample["block"]))
        for sample in samples
    })
    for body, count, grid, block in identities:
        durations = [
            sample["duration_ns"] for sample in samples
            if (sample["body"], sample["count"], tuple(sample["grid"]),
                tuple(sample["block"])) == (body, count, grid, block)
        ]
        cell_medians.append({
            "body": body,
            "count": count,
            "grid": list(grid),
            "block": list(block),
            "sample_count": len(durations),
            "median_duration_ns": statistics.median(durations),
            "minimum_duration_ns": min(durations),
            "maximum_duration_ns": max(durations),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "timer_domain": TIMER_DOMAIN,
        "software_revision": "maca_3.7.2",
        "production_or_evaluation_duration_consumed": False,
        "mode": mode,
        "trial": trial,
        "physical_device_id": device_id,
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "binary_stdout": {
            "path": str(stdout_path), "sha256": _sha256(stdout_path)},
        "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
        "samples": list(samples),
        "cell_medians": cell_medians,
    }


def _load_report(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "hardware_id", "mechanism", "timer_domain",
        "software_revision", "production_or_evaluation_duration_consumed",
        "mode", "trial", "physical_device_id", "plan", "binary_stdout",
        "trace", "samples", "cell_medians",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("program-span run report schema is invalid")
    if (payload["schema_version"] != SCHEMA_VERSION
            or payload["hardware_id"] != HARDWARE_ID
            or payload["mechanism"] != MECHANISM
            or payload["timer_domain"] != TIMER_DOMAIN
            or payload["software_revision"] != "maca_3.7.2"
            or payload["production_or_evaluation_duration_consumed"]
            is not False):
        raise ValueError("program-span run report contract is invalid")
    if not isinstance(payload["cell_medians"], list):
        raise TypeError("program-span cell_medians must be a list")
    return payload


def _cell_index(report):
    expected_plan = build_plan(report["trial"], report["mode"])
    expected_identities = {
        _configuration(row) for row in expected_plan
    }
    indexed = {}
    for ordinal, cell in enumerate(report["cell_medians"]):
        if not isinstance(cell, dict):
            raise TypeError(
                f"program-span cell median {ordinal} is not a mapping")
        required = {
            "body", "count", "grid", "block", "sample_count",
            "median_duration_ns", "minimum_duration_ns",
            "maximum_duration_ns",
        }
        if set(cell) != required:
            raise ValueError("program-span cell median schema is invalid")
        grid = cell["grid"]
        block = cell["block"]
        if (not isinstance(grid, list) or len(grid) != 3
                or not isinstance(block, list) or len(block) != 3):
            raise TypeError("program-span cell launch dimensions are invalid")
        identity = (
            cell["body"], cell["count"], *grid, *block)
        if identity in indexed:
            raise ValueError("program-span cell median identity is duplicated")
        if cell["sample_count"] != FULL_REPEATS:
            raise ValueError("program-span full cell sample count is invalid")
        durations = (
            cell["minimum_duration_ns"], cell["median_duration_ns"],
            cell["maximum_duration_ns"],
        )
        if (any(isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value <= 0
                for value in durations)
                or durations[0] > durations[1]
                or durations[1] > durations[2]):
            raise ValueError("program-span cell durations are invalid")
        indexed[identity] = float(cell["median_duration_ns"])
    if set(indexed) != expected_identities:
        raise ValueError("program-span full report changed the frozen matrix")
    return indexed


def _intersect(intervals, context):
    lower = max(interval[0] for interval in intervals)
    upper = min(interval[1] for interval in intervals)
    if lower > upper:
        raise ValueError(f"program-span {context} intervals do not intersect")
    return lower, upper


def _difference_interval(unrolled_ns, cold_ns, count):
    if count <= 1:
        raise ValueError("program-span difference requires count > 1")
    observed = unrolled_ns - cold_ns
    return (
        (observed - DIFFERENCE_UNCERTAINTY_NS) / (count - 1),
        (observed + DIFFERENCE_UNCERTAINTY_NS) / (count - 1),
    )


def _prediction_interval(count, waves, latency_interval, issue_interval):
    lower = max(latency_interval[0], waves * issue_interval[0])
    upper = max(latency_interval[1], waves * issue_interval[1])
    return (count - 1) * lower, (count - 1) * upper


def _overlaps(first, second):
    return first[0] <= second[1] and second[0] <= first[1]


def aggregate_reports(paths):
    reports = [_load_report(path) for path in paths]
    actual_trials = tuple(sorted(report["trial"] for report in reports))
    if actual_trials != FULL_TRIALS or len(reports) != len(FULL_TRIALS):
        raise ValueError("program-span aggregate requires full trials 1, 2, 3")
    if any(report["mode"] != "full" for report in reports):
        raise ValueError("program-span aggregate requires full reports")
    device_ids = {report["physical_device_id"] for report in reports}
    if len(device_ids) != 1:
        raise ValueError("program-span aggregate mixes physical devices")
    indexed = {
        report["trial"]: _cell_index(report) for report in reports
    }

    training_step_intervals = {waves: [] for waves in (1, 4, 8, 16)}
    for trial in TRAINING_TRIALS:
        cells = indexed[trial]
        for grid in GRIDS:
            for block in BLOCKS:
                waves = math.ceil(block / PARALLEL_WIDTH)
                unrolled = cells[(
                    "unrolled", INFERENCE_COUNT, grid, 1, 1,
                    block, 1, 1)]
                cold = cells[(
                    "cold", INFERENCE_COUNT, grid, 1, 1,
                    block, 1, 1)]
                training_step_intervals[waves].append(
                    _difference_interval(unrolled, cold, INFERENCE_COUNT))
    step_intervals = {
        waves: _intersect(intervals, f"wave-{waves} training")
        for waves, intervals in training_step_intervals.items()
    }

    # W=1 closes the dependent-latency plateau.  W=8 and W=16 both have a
    # lower service bound above that plateau, so they independently close the
    # issue interval.  W=4 straddles the crossover and is validation only.
    latency_interval = step_intervals[1]
    if not (step_intervals[8][0] > latency_interval[1]
            and step_intervals[16][0] > latency_interval[1]):
        raise ValueError(
            "program-span high-wave cells do not separate issue service")
    issue_interval = _intersect((
        (step_intervals[8][0] / 8.0, step_intervals[8][1] / 8.0),
        (step_intervals[16][0] / 16.0,
         step_intervals[16][1] / 16.0),
    ), "high-wave issue")

    diagnostics = {}
    for label, trials, count in (
            ("training_high_count", TRAINING_TRIALS, INFERENCE_COUNT),
            ("frozen_high_count", VALIDATION_TRIALS, INFERENCE_COUNT),
            ("secondary_count", FULL_TRIALS, SECONDARY_VALIDATION_COUNT)):
        violations = []
        checked = 0
        for trial in trials:
            cells = indexed[trial]
            for grid in GRIDS:
                for block in BLOCKS:
                    waves = math.ceil(block / PARALLEL_WIDTH)
                    observed_delta = (
                        cells[("unrolled", count, grid, 1, 1,
                               block, 1, 1)]
                        - cells[("cold", count, grid, 1, 1,
                                 block, 1, 1)])
                    observed_interval = (
                        observed_delta - DIFFERENCE_UNCERTAINTY_NS,
                        observed_delta + DIFFERENCE_UNCERTAINTY_NS,
                    )
                    predicted_interval = _prediction_interval(
                        count, waves, latency_interval, issue_interval)
                    checked += 1
                    if not _overlaps(observed_interval, predicted_interval):
                        violations.append({
                            "trial": trial,
                            "count": count,
                            "grid": grid,
                            "block": block,
                            "waves": waves,
                            "observed_delta_ns": observed_delta,
                            "observed_interval_ns": list(observed_interval),
                            "predicted_interval_ns": list(predicted_interval),
                        })
        diagnostics[label] = {
            "cell_count": checked,
            "violation_count": len(violations),
            "violations": violations,
        }

    high_count_passed = (
        diagnostics["training_high_count"]["violation_count"] == 0
        and diagnostics["frozen_high_count"]["violation_count"] == 0)
    secondary_passed = (
        diagnostics["secondary_count"]["violation_count"] == 0)
    publishable = high_count_passed and secondary_passed
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "timer_domain": TIMER_DOMAIN,
        "software_revision": "maca_3.7.2",
        "production_or_evaluation_duration_consumed": False,
        "physical_device_id": next(iter(device_ids)),
        "run_reports": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "formula": "(count-1)*max(dependent_latency_ns, waves*issue_ns)",
        "quantization_contract": {
            "mctracer_tick_ns": MCTRACER_QUANTIZATION_NS,
            "paired_difference_uncertainty_ns": DIFFERENCE_UNCERTAINTY_NS,
        },
        "parameter_intervals": {
            "dependent_latency_ns_per_add": list(latency_interval),
            "issue_ns_per_wave_instruction": list(issue_interval),
            "crossover_waves": [
                latency_interval[0] / issue_interval[1],
                latency_interval[1] / issue_interval[0],
            ],
        },
        "training_step_intervals_ns": {
            str(waves): list(interval)
            for waves, interval in step_intervals.items()
        },
        "diagnostics": diagnostics,
        "publishable": publishable,
        "disposition": (
            "eligible_register_program_span_interval" if publishable
            else "not_publishable_requires_higher_resolution_mechanism"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="C500 low-wave program-span mechanism campaign")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    plan_parser.add_argument("--trial", type=int, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    analyze_parser.add_argument("--trial", type=int, required=True)
    analyze_parser.add_argument("--plan", type=Path, required=True)
    analyze_parser.add_argument("--stdout", type=Path, required=True)
    analyze_parser.add_argument("--trace", type=Path, required=True)
    analyze_parser.add_argument("--device-id", type=int, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--report", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        args.output.write_text(
            plan_text(build_plan(args.trial, args.mode)), encoding="utf-8")
        return
    if args.command == "analyze":
        report = build_report(
            args.mode, args.trial, args.plan, args.stdout, args.trace,
            args.device_id)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return
    if args.command == "aggregate":
        aggregate = aggregate_reports(args.report)
        args.output.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return
    raise AssertionError(f"unknown program-span command: {args.command!r}")


if __name__ == "__main__":
    main()
