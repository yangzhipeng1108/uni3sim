"""Strict trace-off/trace-on observer-effect gate for Serving capture."""

from __future__ import annotations

from evaluation.json_shape import file_sha256

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.json_shape import require_object_value as _mapping


_THRESHOLD_PERCENT = 3.0
_METRICS = (
    ("request_throughput", "request_throughput_per_s", None),
    ("output_token_throughput", "output_token_throughput_per_s", None),
    ("ttft", "ttft", "mean_ms"),
    ("tpot", "tpot", "mean_ms"),
    ("e2e", "e2e", "mean_ms"),
)


def compare_scheduler_trace_observer_effect(
    trace_off_path: str | Path,
    trace_on_path: str | Path,
    *,
    threshold_percent: float = _THRESHOLD_PERCENT,
) -> Mapping[str, object]:
    """Classify paired no-profiler runs at an explicit evaluation threshold.

    The historical default remains three percent. A diagnostic may request
    another threshold without editing the capture or its published verdict.
    """

    threshold = _positive(threshold_percent, "observer-effect threshold percent")

    off_path = Path(trace_off_path).resolve(strict=True)
    on_path = Path(trace_on_path).resolve(strict=True)
    off = _json_object(off_path, "trace-off result")
    on = _json_object(on_path, "trace-on result")
    _validate_result(off, trace_enabled=False)
    _validate_result(on, trace_enabled=True)
    _validate_pair(off, on)
    off_metrics = _mapping(off["metrics"], "trace-off metrics")
    on_metrics = _mapping(on["metrics"], "trace-on metrics")
    deltas = {}
    values = {}
    for name, section, statistic in _METRICS:
        if statistic is None:
            off_value = _positive(off_metrics[section], f"trace-off {name}")
            on_value = _positive(on_metrics[section], f"trace-on {name}")
        else:
            off_value = _positive(
                _mapping(off_metrics[section], f"trace-off {section}")[statistic],
                f"trace-off {name}",
            )
            on_value = _positive(
                _mapping(on_metrics[section], f"trace-on {section}")[statistic],
                f"trace-on {name}",
            )
        delta = abs(on_value - off_value) / off_value * 100.0
        deltas[name] = delta
        values[name] = {"trace_off": off_value, "trace_on": on_value}
    passed = all(delta <= threshold for delta in deltas.values())
    identity = _mapping(
        _mapping(on["workload"], "trace-on workload")["identity"],
        "trace-on identity",
    )
    validation_case_id = "/".join(str(identity[field]) for field in (
        "hardware_id", "model_id", "case_id"
    ))
    return {
        "schema_version": 1,
        "kind": "scheduler_trace_observer_effect",
        "validation_case_id": validation_case_id,
        "trace_off": {
            "run_id": _run_id(off_path),
            "path": str(off_path),
            "sha256": _sha256(off_path),
        },
        "trace_on": {
            "run_id": _run_id(on_path),
            "path": str(on_path),
            "sha256": _sha256(on_path),
        },
        "threshold_percent": threshold,
        "metric_values": values,
        "metric_delta_percent": deltas,
        "passed": passed,
    }


def require_observer_effect_passed(
    comparison: Mapping[str, object],
) -> None:
    """Fail loudly unless all five observer-effect metrics pass."""

    payload = _mapping(comparison, "observer-effect comparison")
    if payload["passed"] is not True:
        deltas = _mapping(
            payload["metric_delta_percent"], "observer-effect metric deltas"
        )
        raise RuntimeError(
            f"Scheduler trace observer effect exceeds {payload['threshold_percent']:g}%: "
            + ", ".join(
                f"{name}={float(deltas[name]):.3f}%" for name, _, _ in _METRICS
            )
        )


def _validate_result(result: Mapping[str, object], *, trace_enabled: bool) -> None:
    if result["schema_version"] != 1 or result["status"] != "success":
        raise ValueError("observer-effect input must be a successful schema-v1 run")
    server = _mapping(result["server"], "ground-truth server")
    profiler = _mapping(server["profiler"], "ground-truth profiler")
    if profiler["requested"] is not False:
        raise ValueError("observer-effect runs must not use a profiler")
    trace = _mapping(server["scheduler_trace"], "server scheduler_trace")
    if trace["enabled"] is not trace_enabled:
        raise ValueError("ground-truth Scheduler trace mode is inconsistent")
    if trace_enabled:
        if result["scheduler_trace"] is None:
            raise ValueError("trace-on result is missing Scheduler trace summary")
    elif "scheduler_trace" in result and result["scheduler_trace"] is not None:
        raise ValueError("trace-off result must not contain Scheduler trace data")


def _validate_pair(
    trace_off: Mapping[str, object], trace_on: Mapping[str, object]
) -> None:
    off_workload = _mapping(trace_off["workload"], "trace-off workload")
    on_workload = _mapping(trace_on["workload"], "trace-on workload")
    for field in (
        "request_count", "expected_qps", "seed", "identity",
        "model_config_sha256", "requests_sha256", "request_stream_sha256",
        "dataset_sha256",
    ):
        if off_workload[field] != on_workload[field]:
            raise ValueError(f"observer-effect workload mismatch: {field}")
    off_server = _mapping(trace_off["server"], "trace-off server")
    on_server = _mapping(trace_on["server"], "trace-on server")
    for field in ("hardware_id", "devices", "engine"):
        if off_server[field] != on_server[field]:
            raise ValueError(f"observer-effect server mismatch: {field}")
    off_metrics = _mapping(trace_off["metrics"], "trace-off metrics")
    on_metrics = _mapping(trace_on["metrics"], "trace-on metrics")
    off_requests = _sequence(off_metrics["per_request"], "trace-off per_request")
    on_requests = _sequence(on_metrics["per_request"], "trace-on per_request")
    if len(off_requests) != len(on_requests):
        raise ValueError("observer-effect request counts disagree")
    for index, (off_value, on_value) in enumerate(
        zip(off_requests, on_requests, strict=True)
    ):
        off_request = _mapping(off_value, f"trace-off request {index}")
        on_request = _mapping(on_value, f"trace-on request {index}")
        for field in (
            "request_id", "scheduled_arrival_s", "input_tokens", "output_tokens"
        ):
            if off_request[field] != on_request[field]:
                raise ValueError(
                    f"observer-effect request {index} mismatch: {field}"
                )


def _run_id(path: Path) -> str:
    return path.parent.name + "/" + path.name


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    value = json.loads(path.read_text())
    return _mapping(value, label)




def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _positive(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)
