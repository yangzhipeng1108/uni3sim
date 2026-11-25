#!/usr/bin/env python3
"""Generate and audit the C500 mcTracer scheduler-floor campaign."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics


SCHEMA_VERSION = 2
HARDWARE_ID = "metax_c500"
NUM_COMPUTE_UNITS = 104
PARALLEL_WIDTH = 64
WARMUP_LAUNCHES_PER_GEOMETRY = 16
FULL_SAMPLES_PER_CELL = 30
SMOKE_SAMPLES_PER_CELL = 5
FULL_TRIALS = (1, 2, 3)
TRAINING_TRIALS = (1, 2)
VALIDATION_TRIALS = (3,)
MAX_TRAINING_MAPE_PERCENT = 7.0
MAX_TRAINING_WMAE_PERCENT = 5.0
MAX_VALIDATION_MAPE_PERCENT = 7.0
MAX_VALIDATION_WMAE_PERCENT = 5.0
MAX_VALIDATION_ABSOLUTE_PERCENT_ERROR = 40.0
MAX_AXIS_MAPE_PERCENT = 7.0
MAX_AXIS_WMAE_PERCENT = 5.0
MAX_AXIS_ABSOLUTE_PERCENT_ERROR = 15.0
PLAN_FIELDS = (
    "sample_id", "trial", "order_position", "pair_id", "geometry_id",
    "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z",
    "launch_count",
)
SAMPLE_HEADER = (
    "sample_header", "sample_id", "trial", "order_position", "pair_id",
    "geometry_id", "grid_x", "grid_y", "grid_z", "block_x", "block_y",
    "block_z", "launch_count", "host_start_ns", "host_end_ns",
    "batch_device_ns", "per_launch_device_ns", "device_elapsed_ms_hex",
    "event_status",
)
ONE_D_GRIDS = (1, 52, 104, 208, 832, 6656, 53248, 425984)
ONE_D_BLOCKS = (1, 64, 128, 256, 512, 1024)


@dataclass(frozen=True)
class Geometry:
    geometry_id: str
    family: str
    grid: tuple
    block: tuple

    @property
    def grid_size(self):
        return math.prod(self.grid)

    @property
    def block_size(self):
        return math.prod(self.block)


def _geometry_id(grid, block):
    return (
        f"g{grid[0]}x{grid[1]}x{grid[2]}_"
        f"b{block[0]}x{block[1]}x{block[2]}"
    )


def _geometry(grid, block, family):
    if (not isinstance(grid, tuple) or len(grid) != 3
            or not isinstance(block, tuple) or len(block) != 3):
        raise TypeError("launch geometry must use two dim3 tuples")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in grid + block):
        raise ValueError("launch geometry axes must be positive integers")
    if math.prod(block) > 1024:
        raise ValueError("C500 launch block cannot exceed 1024 threads")
    return Geometry(_geometry_id(grid, block), family, grid, block)


def full_geometries():
    one_d = tuple(
        _geometry((grid, 1, 1), (block, 1, 1), "one_d")
        for grid in ONE_D_GRIDS for block in ONE_D_BLOCKS
    )
    axis = (
        _geometry((52, 2, 1), (256, 1, 1), "axis_holdout"),
        _geometry((13, 2, 4), (256, 1, 1), "axis_holdout"),
        _geometry((104, 64, 1), (256, 1, 1), "axis_holdout"),
        _geometry((13, 8, 64), (256, 1, 1), "axis_holdout"),
        _geometry((104, 1, 1), (8, 8, 1), "axis_holdout"),
        _geometry((104, 1, 1), (4, 4, 4), "axis_holdout"),
        _geometry((6656, 1, 1), (16, 16, 1), "axis_holdout"),
        _geometry((6656, 1, 1), (8, 8, 4), "axis_holdout"),
    )
    geometries = one_d + axis
    identities = [geometry.geometry_id for geometry in geometries]
    if len(identities) != len(set(identities)):
        raise AssertionError("C500 scheduler matrix contains duplicate dim3")
    return geometries


def smoke_geometries():
    return (
        _geometry((1, 1, 1), (64, 1, 1), "one_d"),
        _geometry((104, 1, 1), (256, 1, 1), "one_d"),
        _geometry((832, 1, 1), (1024, 1, 1), "one_d"),
        _geometry((52, 2, 1), (256, 1, 1), "axis_holdout"),
        _geometry((104, 1, 1), (4, 4, 4), "axis_holdout"),
    )


def _matrix(mode):
    if mode == "smoke":
        return smoke_geometries(), SMOKE_SAMPLES_PER_CELL
    elif mode == "full":
        return full_geometries(), FULL_SAMPLES_PER_CELL
    else:
        raise ValueError(f"unknown C500 scheduler campaign mode: {mode!r}")


def build_plan(body, trial, mode):
    if body not in {"empty", "minimal", "memory"}:
        raise ValueError("kernel body must be empty, minimal, or memory")
    if isinstance(trial, bool) or not isinstance(trial, int) or trial <= 0:
        raise ValueError("trial must be a positive integer")
    if mode == "full" and trial not in FULL_TRIALS:
        raise ValueError("full campaign trial must be one, two, or three")
    if mode == "smoke" and trial != 1:
        raise ValueError("smoke campaign has exactly one trial")
    geometries, samples_per_cell = _matrix(mode)
    scheduled = []
    for repeat in range(1, samples_per_cell + 1):
        for geometry_index, geometry in enumerate(geometries):
            scheduled.append((repeat, geometry_index, geometry))
    if body == "empty":
        body_offset = 0
    elif body == "minimal":
        body_offset = 1000003
    elif body == "memory":
        body_offset = 2000003
    else:
        raise ValueError(f"unknown C500 scheduler body: {body!r}")
    rng = random.Random(500104 + 7919 * trial + body_offset)
    rng.shuffle(scheduled)
    rows = []
    for order_position, (repeat, geometry_index, geometry) in enumerate(
            scheduled):
        row = {
            "sample_id": (
                f"{body}_t{trial}_r{repeat}_g{geometry_index}"
            ),
            "trial": trial,
            "order_position": order_position,
            "pair_id": f"mctracer_{geometry.family}",
            "geometry_id": geometry.geometry_id,
            "grid_x": geometry.grid[0],
            "grid_y": geometry.grid[1],
            "grid_z": geometry.grid[2],
            "block_x": geometry.block[0],
            "block_y": geometry.block[1],
            "block_z": geometry.block[2],
            "launch_count": 1,
        }
        if len(row["sample_id"]) > 96:
            raise AssertionError("generated sample ID exceeds source contract")
        rows.append(row)
    return tuple(rows)


def plan_text(rows):
    lines = []
    for row in rows:
        lines.append(" ".join(str(row[field]) for field in PLAN_FIELDS))
    return "\n".join(lines) + "\n"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_positive_int(text, label):
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer") from error
    if str(value) != text or value <= 0:
        raise ValueError(f"{label} must be a canonical positive integer")
    return value


def validate_binary_stdout(path, plan):
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    if len(rows) != len(plan) + 4:
        raise ValueError("C500 scheduler binary output row count is invalid")
    tracer_banner = rows[0]
    tracer_endpoint = rows[1]
    if (len(tracer_banner) != 1
            or tracer_banner[0].strip() != "mctxAutoStart called!"):
        raise ValueError("C500 scheduler mcTracer stdout banner is invalid")
    endpoint_prefix = "mcToolsExtPid_UsedPort:"
    if (len(tracer_endpoint) != 1
            or not tracer_endpoint[0].startswith(endpoint_prefix)):
        raise ValueError("C500 scheduler mcTracer stdout endpoint is invalid")
    endpoint_fields = tracer_endpoint[0].removeprefix(endpoint_prefix).split("-")
    if len(endpoint_fields) != 2:
        raise ValueError("C500 scheduler mcTracer stdout endpoint is invalid")
    _parse_positive_int(endpoint_fields[0], "mcTracer target PID")
    port = _parse_positive_int(endpoint_fields[1], "mcTracer RPC port")
    if port > 65535:
        raise ValueError("C500 scheduler mcTracer RPC port is invalid")
    rows = rows[2:]
    metadata = rows[0]
    if (len(metadata) != 10 or metadata[0] != "metadata"
            or metadata[2] != "MetaX C500"
            or metadata[5] != str(NUM_COMPUTE_UNITS)
            or metadata[6] != str(PARALLEL_WIDTH)
            or metadata[8] != "CLOCK_MONOTONIC_RAW"
            or metadata[9] != "mcEvent"):
        raise ValueError("C500 scheduler binary metadata is invalid")
    if tuple(rows[1]) != SAMPLE_HEADER:
        raise ValueError("C500 scheduler binary sample header is invalid")
    planned_fields = PLAN_FIELDS
    for output, planned in zip(rows[2:], plan):
        if len(output) != 19 or output[0] != "sample":
            raise ValueError("C500 scheduler binary sample row is invalid")
        values = (
            output[1], _parse_positive_int(output[2], "trial"),
            int(output[3]), output[4], output[5],
            _parse_positive_int(output[6], "grid_x"),
            _parse_positive_int(output[7], "grid_y"),
            _parse_positive_int(output[8], "grid_z"),
            _parse_positive_int(output[9], "block_x"),
            _parse_positive_int(output[10], "block_y"),
            _parse_positive_int(output[11], "block_z"),
            _parse_positive_int(output[12], "launch_count"),
        )
        parsed = dict(zip(planned_fields, values))
        if parsed != planned:
            raise ValueError("C500 scheduler binary changed the frozen plan")


def _trace_events(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("cannot parse C500 scheduler mcTracer JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"traceEvents"}:
        raise ValueError("C500 scheduler mcTracer root schema is invalid")
    events = payload["traceEvents"]
    if not isinstance(events, list):
        raise TypeError("C500 scheduler traceEvents must be a list")
    return events


def parse_trace(path, body, plan, device_id):
    if body == "empty":
        token = "empty_kernel"
    elif body == "minimal":
        token = "minimal_register_kernel"
    elif body == "memory":
        token = "memory_dependency_kernel"
    else:
        raise ValueError(f"unknown C500 scheduler body: {body!r}")
    gpu_events = []
    cpu_launch_counts = {}
    for event_index, event in enumerate(_trace_events(path)):
        if not isinstance(event, dict):
            raise TypeError(f"mcTracer event {event_index} must be a mapping")
        if ("name" in event and event["name"] in {
                "mcLaunchKernel", "mcModuleLaunchKernel"}):
            if "args" not in event or not isinstance(event["args"], dict):
                raise ValueError("mcTracer CPU launch is missing args")
            arguments = event["args"]
            if "co_id" not in arguments:
                raise ValueError("mcTracer CPU launch is missing co_id")
            co_id = arguments["co_id"]
            if isinstance(co_id, bool) or not isinstance(co_id, int):
                raise TypeError("mcTracer CPU launch co_id must be an integer")
            if co_id not in cpu_launch_counts:
                cpu_launch_counts[co_id] = 0
            cpu_launch_counts[co_id] += 1
        if ("args" not in event or not isinstance(event["args"], dict)
                or "name" not in event["args"]
                or token not in str(event["args"]["name"])):
            continue
        arguments = event["args"]
        required = {"co_id", "device_id", "grid", "block", "mem"}
        missing = required - set(arguments)
        if missing:
            raise ValueError(
                f"mcTracer GPU event is missing fields: {sorted(missing)}")
        if "dur" not in event or "ph" not in event or event["ph"] != "X":
            raise ValueError("mcTracer GPU event is missing complete duration")
        duration_ns = event["dur"]
        if (isinstance(duration_ns, bool) or not isinstance(duration_ns, int)
                or duration_ns <= 0):
            raise ValueError("mcTracer GPU duration must be a positive integer")
        if arguments["device_id"] != device_id:
            raise ValueError("mcTracer GPU event used the wrong device")
        dimensions = []
        for name in ("grid", "block"):
            dim3 = arguments[name]
            if not isinstance(dim3, dict) or set(dim3) != {"x", "y", "z"}:
                raise ValueError(f"mcTracer {name} must be an exact dim3")
            normalized = tuple(dim3[axis] for axis in ("x", "y", "z"))
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in normalized):
                raise ValueError(f"mcTracer {name} axes must be positive")
            dimensions.append(normalized)
        memory = arguments["mem"]
        if not isinstance(memory, dict):
            raise TypeError("mcTracer kernel memory metadata must be a mapping")
        for field in ("registers_per_thread", "dynamic_shared", "static_shared"):
            if field not in memory:
                raise ValueError(f"mcTracer kernel memory lacks {field}")
        if (memory["dynamic_shared"] != 0 or memory["static_shared"] != 0
                or isinstance(memory["registers_per_thread"], bool)
                or not isinstance(memory["registers_per_thread"], int)
                or memory["registers_per_thread"] <= 0):
            raise ValueError("scheduler probe must be register-only")
        gpu_events.append({
            "co_id": arguments["co_id"],
            "grid": dimensions[0],
            "block": dimensions[1],
            "duration_ns": duration_ns,
            "registers_per_thread": memory["registers_per_thread"],
        })
    unique_geometries = sorted({
        (
            row["grid_x"], row["grid_y"], row["grid_z"],
            row["block_x"], row["block_y"], row["block_z"],
        )
        for row in plan
    })
    expected_warmup = [
        (geometry[:3], geometry[3:])
        for geometry in unique_geometries
        for _ in range(WARMUP_LAUNCHES_PER_GEOMETRY)
    ]
    expected_samples = [
        (
            (row["grid_x"], row["grid_y"], row["grid_z"]),
            (row["block_x"], row["block_y"], row["block_z"]),
        )
        for row in plan
    ]
    expected_sequence = expected_warmup + expected_samples
    actual_sequence = [
        (event["grid"], event["block"]) for event in gpu_events
    ]
    if actual_sequence != expected_sequence:
        raise ValueError("mcTracer kernel sequence differs from the frozen plan")
    for event in gpu_events:
        co_id = event["co_id"]
        if co_id not in cpu_launch_counts or cpu_launch_counts[co_id] != 1:
            raise ValueError("mcTracer GPU event lacks unique CPU correlation")
    measured = gpu_events[len(expected_warmup):]
    return tuple({
        "sample_id": row["sample_id"],
        "trial": row["trial"],
        "family": row["pair_id"].removeprefix("mctracer_"),
        "geometry_id": row["geometry_id"],
        "grid": list(event["grid"]),
        "block": list(event["block"]),
        "duration_ns": event["duration_ns"],
        "registers_per_thread": event["registers_per_thread"],
    } for row, event in zip(plan, measured))


def build_run_report(body, trial, mode, plan_path, stdout_path, trace_path,
                     device_id):
    expected_plan = build_plan(body, trial, mode)
    if plan_path.read_text(encoding="utf-8") != plan_text(expected_plan):
        raise ValueError("C500 scheduler plan differs from the frozen generator")
    validate_binary_stdout(stdout_path, expected_plan)
    samples = parse_trace(trace_path, body, expected_plan, device_id)
    mechanism = {
        "memory": "mctracer_memory_dependency_task_duration",
        "empty": "mctracer_nonempty_task_scheduler_duration",
        "minimal": "mctracer_nonempty_task_scheduler_duration",
    }[body]
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": mechanism,
        "timer_domain": "mctracer_gpu_kernel_event",
        "duration_consumed": True,
        "production_or_evaluation_duration_consumed": False,
        "mode": mode,
        "body": body,
        "trial": trial,
        "device_id": device_id,
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "binary_stdout": {
            "path": str(stdout_path), "sha256": _sha256(stdout_path),
        },
        "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
        "samples": list(samples),
    }


def _group_medians(report):
    grouped = {}
    for sample in report["samples"]:
        identity = sample["geometry_id"]
        if identity not in grouped:
            grouped[identity] = []
        grouped[identity].append(sample["duration_ns"])
    if report["mode"] == "smoke":
        expected_count = SMOKE_SAMPLES_PER_CELL
    elif report["mode"] == "full":
        expected_count = FULL_SAMPLES_PER_CELL
    else:
        raise ValueError(
            f"unknown C500 scheduler report mode: {report['mode']!r}")
    if any(len(values) != expected_count for values in grouped.values()):
        raise ValueError("C500 scheduler report has incomplete geometry samples")
    return {
        identity: float(statistics.median(values))
        for identity, values in grouped.items()
    }


def _load_report(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "hardware_id", "mechanism", "timer_domain",
        "duration_consumed", "production_or_evaluation_duration_consumed",
        "mode", "body", "trial", "device_id", "plan", "binary_stdout",
        "trace", "samples",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("C500 scheduler run report schema is invalid")
    if payload["body"] == "memory":
        expected_mechanism = "mctracer_memory_dependency_task_duration"
    elif payload["body"] in ("empty", "minimal"):
        expected_mechanism = "mctracer_nonempty_task_scheduler_duration"
    else:
        raise ValueError("C500 scheduler report body is invalid")
    if (payload["schema_version"] != SCHEMA_VERSION
            or payload["hardware_id"] != HARDWARE_ID
            or payload["mechanism"] != expected_mechanism
            or payload["timer_domain"] != "mctracer_gpu_kernel_event"
            or payload["duration_consumed"] is not True
            or payload["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError("C500 scheduler report contract is invalid")
    return payload


def _features(grid_size, block_size):
    rounds = math.ceil(grid_size / NUM_COMPUTE_UNITS)
    waves = math.ceil(block_size / PARALLEL_WIDTH)
    return (
        1.0,
        float(grid_size),
        float(rounds if waves == 1 else 0),
        float(rounds * waves if waves > 1 else 0),
    )


def _fit_model(rows):
    # NumPy is needed only for the predeclared full-campaign least-squares
    # fit.  Plan generation, trace validation, and smoke aggregation remain
    # runnable in the deliberately minimal profiler container.
    import numpy as np

    durations = np.asarray([row["duration_ns"] for row in rows], dtype=float)
    design = np.asarray([
        _features(row["grid_size"], row["block_size"])
        for row in rows
    ], dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(
        design / durations[:, None], np.ones(len(rows)), rcond=None)
    if np.any(coefficients <= 0) or not np.all(np.isfinite(coefficients)):
        raise ValueError(
            "non-empty scheduler services must fit finite positive values")
    (submit_ns, cta_descriptor_ns, single_wave_round_ns,
     multi_wave_round_ns) = coefficients
    relative = (design @ coefficients - durations) / durations
    return {
        "submit_ns": float(submit_ns),
        "cta_descriptor_ns": float(cta_descriptor_ns),
        "single_wave_round_ns": float(single_wave_round_ns),
        "multi_wave_round_ns": float(multi_wave_round_ns),
        "relative_squared_error": float(np.dot(relative, relative)),
    }


def _predict(grid_size, block_size, model):
    features = _features(grid_size, block_size)
    coefficients = (
        model["submit_ns"],
        model["cta_descriptor_ns"],
        model["single_wave_round_ns"],
        model["multi_wave_round_ns"],
    )
    return math.fsum(
        feature * coefficient
        for feature, coefficient in zip(features, coefficients))


def _metrics(rows, model):
    observed = tuple(float(row["duration_ns"]) for row in rows)
    predicted = tuple(
        _predict(row["grid_size"], row["block_size"], model)
        for row in rows)
    errors = tuple(
        predicted_value - observed_value
        for predicted_value, observed_value in zip(predicted, observed))
    absolute_percentage = tuple(
        abs(error) / observed_value
        for error, observed_value in zip(errors, observed))
    return {
        "cell_count": len(rows),
        "mape_percent": (
            math.fsum(absolute_percentage) / len(rows) * 100.0),
        "wmae_percent": (
            math.fsum(abs(error) for error in errors)
            / math.fsum(observed) * 100.0),
        "signed_mean_percentage_error": (
            math.fsum(
                error / observed_value
                for error, observed_value in zip(errors, observed))
            / len(rows) * 100.0),
        "maximum_absolute_percentage_error": (
            max(absolute_percentage) * 100.0),
    }


def aggregate_reports(paths, mode):
    reports = [_load_report(path) for path in paths]
    if mode == "smoke":
        expected_trials = (1,)
    elif mode == "full":
        expected_trials = FULL_TRIALS
    else:
        raise ValueError(f"unknown C500 scheduler aggregate mode: {mode!r}")
    expected = {
        (body, trial) for body in ("empty", "minimal")
        for trial in expected_trials
    }
    actual = {(report["body"], report["trial"]) for report in reports}
    if actual != expected or len(reports) != len(expected):
        raise ValueError("C500 scheduler aggregate run set is incomplete")
    if any(report["mode"] != mode for report in reports):
        raise ValueError("C500 scheduler aggregate mixes campaign modes")
    device_ids = {report["device_id"] for report in reports}
    if len(device_ids) != 1:
        raise ValueError("C500 scheduler aggregate mixes devices")
    by_identity = {
        (report["body"], report["trial"]): report for report in reports
    }
    empty_median = _group_medians(by_identity[("empty", expected_trials[0])])
    minimal_median = _group_medians(
        by_identity[("minimal", expected_trials[0])])
    if set(empty_median) != set(minimal_median):
        raise ValueError("empty/minimal scheduler geometries differ")
    median_violations = sorted(
        identity for identity in empty_median
        if minimal_median[identity] < empty_median[identity]
    )
    base = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": "mctracer_nonempty_task_scheduler_duration",
        "timer_domain": "mctracer_gpu_kernel_event",
        "production_or_evaluation_duration_consumed": False,
        "mode": mode,
        "device_id": next(iter(device_ids)),
        "run_reports": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "empty_vs_minimal_median_violations": median_violations,
    }
    if mode == "smoke":
        return {
            **base,
            "publishable": False,
            "disposition": "smoke_only_requires_full_frozen_campaign",
        }
    geometry_by_id = {
        geometry.geometry_id: geometry for geometry in full_geometries()
    }
    cell_rows = []
    for body in ("empty", "minimal"):
        for trial in FULL_TRIALS:
            medians = _group_medians(by_identity[(body, trial)])
            if set(medians) != set(geometry_by_id):
                raise ValueError("full scheduler report changed the matrix")
            for identity, duration_ns in medians.items():
                geometry = geometry_by_id[identity]
                cell_rows.append({
                    "body": body,
                    "trial": trial,
                    "family": geometry.family,
                    "grid_size": geometry.grid_size,
                    "block_size": geometry.block_size,
                    "duration_ns": duration_ns,
                })
    training = tuple(
        row for row in cell_rows
        if row["body"] == "minimal" and row["trial"] in TRAINING_TRIALS
        and row["family"] == "one_d"
    )
    validation = tuple(
        row for row in cell_rows
        if row["body"] == "minimal" and row["trial"] in VALIDATION_TRIALS
        and row["family"] == "one_d"
    )
    axis_validation = tuple(
        row for row in cell_rows
        if row["body"] == "minimal" and row["family"] == "axis_holdout"
    )
    model = _fit_model(training)
    training_metrics = _metrics(training, model)
    validation_metrics = _metrics(validation, model)
    axis_metrics = _metrics(axis_validation, model)
    publishable = (
        training_metrics["mape_percent"]
        <= MAX_TRAINING_MAPE_PERCENT
        and training_metrics["wmae_percent"]
        <= MAX_TRAINING_WMAE_PERCENT
        and validation_metrics["mape_percent"]
        <= MAX_VALIDATION_MAPE_PERCENT
        and validation_metrics["wmae_percent"]
        <= MAX_VALIDATION_WMAE_PERCENT
        and validation_metrics["maximum_absolute_percentage_error"]
        <= MAX_VALIDATION_ABSOLUTE_PERCENT_ERROR
        and axis_metrics["mape_percent"] <= MAX_AXIS_MAPE_PERCENT
        and axis_metrics["wmae_percent"] <= MAX_AXIS_WMAE_PERCENT
        and axis_metrics["maximum_absolute_percentage_error"]
        <= MAX_AXIS_ABSOLUTE_PERCENT_ERROR
    )
    return {
        **base,
        "formula": (
            "submit_ns + grid * cta_descriptor_ns + ceil(grid/104) * "
            "(single_wave_round_ns if ceil(block/64) == 1 else "
            "multi_wave_round_ns * ceil(block/64))"
        ),
        "model_body": "register_only_single_add_nonempty_task",
        "parameters": {
            "submit_ns": model["submit_ns"],
            "cta_descriptor_ns": model["cta_descriptor_ns"],
            "single_wave_round_ns": model["single_wave_round_ns"],
            "multi_wave_round_ns": model["multi_wave_round_ns"],
            "extra_grid_dimension_ns": 0.0,
            "extra_block_dimension_ns": 0.0,
        },
        "fit_diagnostics": {
            "training": training_metrics,
            "validation": validation_metrics,
            "axis_holdout": axis_metrics,
        },
        "publication_gates": {
            "training_mape_percent": MAX_TRAINING_MAPE_PERCENT,
            "training_wmae_percent": MAX_TRAINING_WMAE_PERCENT,
            "validation_mape_percent": MAX_VALIDATION_MAPE_PERCENT,
            "validation_wmae_percent": MAX_VALIDATION_WMAE_PERCENT,
            "validation_maximum_absolute_percentage_error": (
                MAX_VALIDATION_ABSOLUTE_PERCENT_ERROR),
            "axis_mape_percent": MAX_AXIS_MAPE_PERCENT,
            "axis_wmae_percent": MAX_AXIS_WMAE_PERCENT,
            "axis_maximum_absolute_percentage_error": (
                MAX_AXIS_ABSOLUTE_PERCENT_ERROR),
        },
        "publishable": publishable,
        "disposition": (
            "eligible_nonempty_task_scheduler_duration" if publishable
            else "not_publishable_without_a_new_predeclared_campaign"
        ),
    }


def aggregate_memory_dependency_reports(paths, mode):
    """Fit the complete same-timer-domain memory-dependency task floor."""

    reports = [_load_report(path) for path in paths]
    if mode == "smoke":
        expected_trials = (1,)
    elif mode == "full":
        expected_trials = FULL_TRIALS
    else:
        raise ValueError(
            f"unknown C500 memory-dependency aggregate mode: {mode!r}")
    expected = {("memory", trial) for trial in expected_trials}
    actual = {(report["body"], report["trial"]) for report in reports}
    if actual != expected or len(reports) != len(expected):
        raise ValueError(
            "C500 memory-dependency aggregate run set is incomplete")
    if any(report["mode"] != mode for report in reports):
        raise ValueError(
            "C500 memory-dependency aggregate mixes campaign modes")
    device_ids = {report["device_id"] for report in reports}
    if len(device_ids) != 1:
        raise ValueError(
            "C500 memory-dependency aggregate mixes devices")
    base = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": "mctracer_memory_dependency_task_duration",
        "timer_domain": "mctracer_gpu_kernel_event",
        "production_or_evaluation_duration_consumed": False,
        "mode": mode,
        "device_id": next(iter(device_ids)),
        "run_reports": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
    }
    if mode == "smoke":
        report, = reports
        return {
            **base,
            "cell_median_ns": _group_medians(report),
            "publishable": False,
            "disposition": "smoke_only_requires_full_frozen_campaign",
        }

    geometry_by_id = {
        geometry.geometry_id: geometry for geometry in full_geometries()
    }
    cell_rows = []
    for report in reports:
        medians = _group_medians(report)
        if set(medians) != set(geometry_by_id):
            raise ValueError(
                "full memory-dependency report changed the matrix")
        for identity, duration_ns in medians.items():
            geometry = geometry_by_id[identity]
            cell_rows.append({
                "trial": report["trial"],
                "family": geometry.family,
                "grid_size": geometry.grid_size,
                "block_size": geometry.block_size,
                "duration_ns": duration_ns,
            })
    training = tuple(
        row for row in cell_rows
        if row["trial"] in TRAINING_TRIALS and row["family"] == "one_d")
    validation = tuple(
        row for row in cell_rows
        if row["trial"] in VALIDATION_TRIALS and row["family"] == "one_d")
    axis_validation = tuple(
        row for row in cell_rows if row["family"] == "axis_holdout")
    model = _fit_model(training)
    training_metrics = _metrics(training, model)
    validation_metrics = _metrics(validation, model)
    axis_metrics = _metrics(axis_validation, model)
    publishable = (
        training_metrics["mape_percent"] <= MAX_TRAINING_MAPE_PERCENT
        and training_metrics["wmae_percent"] <= MAX_TRAINING_WMAE_PERCENT
        and validation_metrics["mape_percent"]
        <= MAX_VALIDATION_MAPE_PERCENT
        and validation_metrics["wmae_percent"]
        <= MAX_VALIDATION_WMAE_PERCENT
        and validation_metrics["maximum_absolute_percentage_error"]
        <= MAX_VALIDATION_ABSOLUTE_PERCENT_ERROR
        and axis_metrics["mape_percent"] <= MAX_AXIS_MAPE_PERCENT
        and axis_metrics["wmae_percent"] <= MAX_AXIS_WMAE_PERCENT
        and axis_metrics["maximum_absolute_percentage_error"]
        <= MAX_AXIS_ABSOLUTE_PERCENT_ERROR)
    return {
        **base,
        "formula": (
            "submit_ns + grid * cta_descriptor_ns + ceil(grid/104) * "
            "(single_wave_round_ns if ceil(block/64) == 1 else "
            "multi_wave_round_ns * ceil(block/64))"
        ),
        "model_body": "one_coalesced_load_dependent_add_store_per_wave",
        "bounds": {
            "minimum_grid_size": min(
                geometry.grid_size for geometry in geometry_by_id.values()),
            "maximum_grid_size": max(
                geometry.grid_size for geometry in geometry_by_id.values()),
            "minimum_block_size": min(
                geometry.block_size for geometry in geometry_by_id.values()),
            "maximum_block_size": max(
                geometry.block_size for geometry in geometry_by_id.values()),
            "maximum_launch_rank": 2,
        },
        "parameters": {
            "submit_ns": model["submit_ns"],
            "cta_descriptor_ns": model["cta_descriptor_ns"],
            "single_wave_round_ns": model["single_wave_round_ns"],
            "multi_wave_round_ns": model["multi_wave_round_ns"],
        },
        "fit_diagnostics": {
            "training": training_metrics,
            "validation": validation_metrics,
            "axis_holdout": axis_metrics,
        },
        "publication_gates": {
            "training_mape_percent": MAX_TRAINING_MAPE_PERCENT,
            "training_wmae_percent": MAX_TRAINING_WMAE_PERCENT,
            "validation_mape_percent": MAX_VALIDATION_MAPE_PERCENT,
            "validation_wmae_percent": MAX_VALIDATION_WMAE_PERCENT,
            "validation_maximum_absolute_percentage_error": (
                MAX_VALIDATION_ABSOLUTE_PERCENT_ERROR),
            "axis_mape_percent": MAX_AXIS_MAPE_PERCENT,
            "axis_wmae_percent": MAX_AXIS_WMAE_PERCENT,
            "axis_maximum_absolute_percentage_error": (
                MAX_AXIS_ABSOLUTE_PERCENT_ERROR),
        },
        "publishable": publishable,
        "disposition": (
            "eligible_complete_memory_dependency_task_floor"
            if publishable
            else "not_publishable_without_a_new_predeclared_campaign"),
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument(
        "--body", choices=("empty", "minimal", "memory"), required=True)
    plan_parser.add_argument("--trial", type=int, required=True)
    plan_parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    plan_parser.add_argument("--output", type=Path, required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument(
        "--body", choices=("empty", "minimal", "memory"), required=True)
    analyze_parser.add_argument("--trial", type=int, required=True)
    analyze_parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    analyze_parser.add_argument("--device-id", type=int, required=True)
    analyze_parser.add_argument("--plan", type=Path, required=True)
    analyze_parser.add_argument("--stdout", type=Path, required=True)
    analyze_parser.add_argument("--trace", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    aggregate_parser.add_argument("--report", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)

    memory_aggregate_parser = subparsers.add_parser("aggregate-memory")
    memory_aggregate_parser.add_argument(
        "--mode", choices=("smoke", "full"), required=True)
    memory_aggregate_parser.add_argument(
        "--report", type=Path, action="append", required=True)
    memory_aggregate_parser.add_argument(
        "--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "plan":
        rows = build_plan(args.body, args.trial, args.mode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(plan_text(rows), encoding="utf-8")
    elif args.command == "analyze":
        report = build_run_report(
            args.body, args.trial, args.mode, args.plan, args.stdout,
            args.trace, args.device_id)
        _write_json(args.output, report)
    elif args.command == "aggregate":
        _write_json(args.output, aggregate_reports(args.report, args.mode))
    elif args.command == "aggregate-memory":
        _write_json(
            args.output,
            aggregate_memory_dependency_reports(args.report, args.mode))
    else:
        raise AssertionError(f"unhandled command: {args.command!r}")


if __name__ == "__main__":
    main()
