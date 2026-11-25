"""Strict trace-off/trace-on observer-effect evidence for one PD case."""

from __future__ import annotations

from evaluation.serving.metrics.diagnostics.scheduler_trace_observer_effect import (
    _METRICS,
)

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.json_shape import require_object_value as _mapping

from evaluation.json_shape import require_text as _text




def compare_pd_observer_effect(
    trace_off_path: str | Path, trace_on_path: str | Path
) -> Mapping[str, object]:
    """Compare paired real PD runs and return the candidate-ready gate record."""

    off_path = Path(trace_off_path).resolve(strict=True)
    on_path = Path(trace_on_path).resolve(strict=True)
    off = _object(off_path, "PD trace-off ground truth")
    on = _object(on_path, "PD trace-on ground truth")
    _validate_run(off, trace_enabled=False)
    _validate_run(on, trace_enabled=True)
    _validate_pair(off, on)
    off_metrics = _mapping(off["metrics"], "PD trace-off metrics")
    on_metrics = _mapping(on["metrics"], "PD trace-on metrics")
    deltas: dict[str, float] = {}
    for name, field, statistic in _METRICS:
        off_value = _metric_value(off_metrics, field, statistic, f"PD trace-off {name}")
        on_value = _metric_value(on_metrics, field, statistic, f"PD trace-on {name}")
        deltas[name] = abs(on_value - off_value) / off_value * 100.0
    identity = _mapping(_mapping(on["workload"], "PD trace-on workload")["identity"], "PD identity")
    validation_case_id = "/".join(
        _text(identity[field], f"PD identity {field}")
        for field in ("hardware_id", "model_id", "case_id")
    )
    return {
        "status": "measured_for_this_case",
        "validation_case_id": validation_case_id,
        "baseline_run_id": _run_id(off_path),
        "instrumented_run_id": _run_id(on_path),
        "threshold_percent": 3.0,
        "metric_delta_percent": deltas,
        "passed": all(value <= 3.0 for value in deltas.values()),
    }


def write_pd_observer_effect(
    output_path: str | Path, trace_off_path: str | Path, trace_on_path: str | Path
) -> Mapping[str, object]:
    """Write one non-overwriting observer-effect report and enforce the gate."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PD observer report: {output}")
    report = compare_pd_observer_effect(trace_off_path, trace_on_path)
    if report["passed"] is not True:
        deltas = _mapping(report["metric_delta_percent"], "PD observer deltas")
        raise RuntimeError(
            "PD observer effect exceeds 3%: "
            + ", ".join(f"{key}={float(value):.3f}%" for key, value in deltas.items())
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _validate_run(value: Mapping[str, object], *, trace_enabled: bool) -> None:
    if value["schema_version"] != 1 or value["status"] != "success":
        raise ValueError("PD observer input must be a successful schema-v1 run")
    server = _mapping(value["server"], "PD observer server")
    if server["profiler_attached"] is not False:
        raise ValueError("PD observer input must not attach a profiler")
    trace = _mapping(server["scheduler_trace"], "PD observer Scheduler trace")
    if trace["enabled"] is not trace_enabled:
        raise ValueError("PD observer trace mode disagrees with the selected run")


def _validate_pair(off: Mapping[str, object], on: Mapping[str, object]) -> None:
    off_workload = _mapping(off["workload"], "PD trace-off workload")
    on_workload = _mapping(on["workload"], "PD trace-on workload")
    for field in ("request_count", "expected_qps", "seed", "identity", "requests_sha256", "request_stream_sha256", "dataset_sha256"):
        if off_workload[field] != on_workload[field]:
            raise ValueError(f"PD observer workload mismatch: {field}")
    off_rows = _sequence(_mapping(off["metrics"], "PD trace-off metrics")["per_request"], "PD trace-off requests")
    on_rows = _sequence(_mapping(on["metrics"], "PD trace-on metrics")["per_request"], "PD trace-on requests")
    if len(off_rows) != len(on_rows):
        raise ValueError("PD observer request counts differ")
    for index, (off_row, on_row) in enumerate(zip(off_rows, on_rows, strict=True)):
        off_request = _mapping(off_row, f"PD trace-off request {index}")
        on_request = _mapping(on_row, f"PD trace-on request {index}")
        for field in ("request_id", "scheduled_arrival_s", "input_tokens", "output_tokens"):
            if off_request[field] != on_request[field]:
                raise ValueError(f"PD observer request {index} differs at {field}")


def _metric_value(metrics: Mapping[str, object], field: str, statistic: str | None, label: str) -> float:
    value = metrics[field] if statistic is None else _mapping(metrics[field], label)[statistic]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _run_id(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON") from error
    return _mapping(value, label)




def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return value
