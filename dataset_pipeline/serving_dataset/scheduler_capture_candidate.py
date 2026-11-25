"""Build one versioned Serving candidate from an accepted trace-on run."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from .compact_candidate_writer import (
    AvailabilityPayload,
    CompactCasePayload,
    write_compact_case_candidate,
)
from .compact_case import (
    ALIGNMENT_FILENAME,
    CANONICAL_REQUEST_COLUMNS,
    LATEST_SCHEMA_VERSION,
    OBSERVER_QUALITY_SCHEMA_VERSION,
    RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
    SCHEDULER_CAPTURE_SCHEMA_VERSION,
    CanonicalRequestRow,
    raw_source_set_sha256,
    read_canonical_request_rows,
    validate_converter_source_commit,
)
from .compressed_artifact import read_compressed_canonical_bytes
from dataset_pipeline.serving_dataset.json_shape import (
    compact_mapping,
    file_sha256,
)


_REQUEST_ID = re.compile(r"request-(?P<index>[0-9]+)")
_TIMING_BOUNDARY = {
    "clock": "time.perf_counter_ns",
    "origin": "first_scheduler_step_start",
    "step_start": "immediately_before_scheduler_schedule",
    "schedule_end": "immediately_after_scheduler_schedule",
    "update_start": "immediately_before_scheduler_output_update",
    "step_end": "immediately_after_scheduler_output_update",
}


def build_scheduler_capture_candidate(
    *,
    canonical_case_dir: str | Path,
    trace_on_result_path: str | Path,
    scheduler_trace_path: str | Path,
    observer_effect_path: str | Path,
    candidate_root: str | Path,
    converter_source_commit: str,
    observer_validation_status: str = "measured_for_this_case",
) -> Path:
    """Write one atomic case whose requests, metrics, and steps share a run."""

    validate_converter_source_commit(converter_source_commit)
    source_case = Path(canonical_case_dir).resolve(strict=True)
    manifest = _json_object(source_case / "manifest.json", "source manifest")
    alignment_entry = _mapping(
        manifest["alignment_steps"], "source alignment entry")
    alignment = _json_bytes_object(
        read_compressed_canonical_bytes(
            source_case / ALIGNMENT_FILENAME,
            compressed_sha256=alignment_entry["compressed_sha256"],
            content_sha256=alignment_entry["sha256"],
            label=ALIGNMENT_FILENAME,
        ),
        "source alignment_steps",
    )
    trace_on_path = Path(trace_on_result_path).resolve(strict=True)
    scheduler_path = Path(scheduler_trace_path).resolve(strict=True)
    observer_path = Path(observer_effect_path).resolve(strict=True)
    trace_on = _json_object(trace_on_path, "trace-on result")
    server = _mapping(trace_on["server"], "trace-on server")
    observer = _normalize_observer_effect(
        _json_object(observer_path, "observer-effect result"),
        observer_validation_status=observer_validation_status,
    )
    _validate_capture_inputs(
        trace_on,
        scheduler_path,
        observer,
        trace_on_path,
        observer_path,
        observer_validation_status,
    )
    trace_on_sha = _sha256(trace_on_path)
    request_rows, old_to_new, metrics = _accepted_requests_and_metrics(
        trace_on, trace_on_sha, source_case / "requests.csv"
    )
    scheduler_steps = _remapped_scheduler_steps(
        scheduler_path, old_to_new
    )
    scheduler_step_schema_version = _capture_schema_version(scheduler_steps)
    if scheduler_step_schema_version not in (
        SCHEDULER_CAPTURE_SCHEMA_VERSION,
        RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
        LATEST_SCHEMA_VERSION,
    ):
        raise ValueError(
            "Scheduler depth candidates require runtime-graph Scheduler rows")
    runtime_metadata_path = Path(f"{scheduler_path}.runtime.json")
    runtime_metadata, runtime_metadata_path = _runtime_depth_metadata(
        runtime_metadata_path, scheduler_steps
    )
    scheduler_steps = tuple(
        {
            key: value
            for key, value in step.items()
            if key != "scheduler_pipeline_depth_evidence"
        }
        for step in scheduler_steps
    )
    identity = dict(_mapping(manifest["identity"], "source identity"))
    capture_schema_version = (
        OBSERVER_QUALITY_SCHEMA_VERSION
        if identity["hardware_id"] == "nvidia_h200"
        else LATEST_SCHEMA_VERSION
    )
    if (capture_schema_version == OBSERVER_QUALITY_SCHEMA_VERSION
            and observer_validation_status != "measured_for_this_case"):
        raise ValueError("H200 observer quality requires own-case measurement")
    replaced_source_ids = {
        "accepted-ground-truth-v3",
        "accepted-scheduler-trace-v3",
        "accepted-scheduler-runtime-depth-v1",
        "scheduler-observer-effect-v1",
    }
    raw_sources = [
        dict(_mapping(value, "source raw source"))
        for value in _sequence(manifest["raw_sources"], "source raw_sources")
        if _mapping(value, "source raw source")["archive_id"]
        not in replaced_source_ids
    ]
    scheduler_instrumentation = _mapping(
        server["scheduler_trace"], "trace-on scheduler instrumentation"
    )
    new_sources = (
        _raw_source(
            "accepted-ground-truth-v3",
            "accepted-ground-truth",
            trace_on_path,
            "midlayer-sharegpt-ground-truth",
            "1",
        ),
        _raw_source(
            "accepted-scheduler-trace-v3",
            "accepted-scheduler-trace",
            scheduler_path,
            scheduler_instrumentation["instrumentation_id"],
            str(scheduler_instrumentation["instrumentation_version"]),
        ),
        _raw_source(
            "scheduler-observer-effect-v1",
            "scheduler-observer-effect",
            observer_path,
            "midlayer-scheduler-trace-ab",
            "1",
        ),
    )
    raw_sources.extend(new_sources)
    if runtime_metadata_path is not None:
        raw_sources.append(_raw_source(
            "accepted-scheduler-runtime-depth-v1",
            "accepted-scheduler-runtime-depth",
            runtime_metadata_path,
            "midlayer-vllm-scheduler-trace",
            "1",
        ))
    raw_set_sha = raw_source_set_sha256(raw_sources)
    alignment_payload = _rebound_alignment(alignment, raw_set_sha)
    cached_hit_evidence = {
        "kind": "scheduler_equivalent_representative",
        "logical_prefix_hit_observability": "unavailable",
        "representative_value": 0,
        "net_prefix_work_reduction_tokens": 0,
        "derivation": "full_scheduler_token_conservation",
        "source_archive_ids": [
            "accepted-ground-truth-v3",
            "accepted-scheduler-trace-v3",
        ],
        "raw_source_set_sha256": raw_set_sha,
    }
    runtime = dict(_mapping(manifest["runtime"], "source runtime"))
    engine = _mapping(server["engine"], "trace-on engine")
    runtime["dtype"] = engine["dtype"]
    runtime["seed"] = engine["seed"]
    runtime["runtime_version"] = server["client_vllm_version"]
    if runtime_metadata["scheduler_pipeline_depth_evidence"]["source_kind"] == (
            "runtime_executor_property"):
        runtime["source_commit"] = runtime_metadata[
            "scheduler_pipeline_depth_evidence"
        ]["source_identity"]
    runtime["scheduler_pipeline_depth"] = runtime_metadata[
        "scheduler_pipeline_depth"
    ]
    runtime["scheduler_pipeline_depth_evidence"] = runtime_metadata[
        "scheduler_pipeline_depth_evidence"
    ]
    runtime["capture_enable_prefix_caching"] = engine[
        "enable_prefix_caching"
    ]
    observer_trace_on = _mapping(observer["trace_on"], "observer trace_on")
    run_id = _run_id(trace_on_path)
    validation = {
        "status": observer_validation_status,
        "validation_case_id": observer["validation_case_id"],
        "baseline_run_id": _mapping(
            observer["trace_off"], "observer trace_off"
        )["run_id"],
        "instrumented_run_id": observer_trace_on["run_id"],
        "threshold_percent": observer["threshold_percent"],
        "metric_delta_percent": {
            key: _mapping(
                observer["metric_delta_percent"], "observer metric deltas"
            )[key]
            for key in (
                "request_throughput", "output_token_throughput", "ttft",
                "tpot", "e2e",
            )
        },
        "passed": observer["passed"],
    }
    if capture_schema_version == OBSERVER_QUALITY_SCHEMA_VERSION:
        validation["source_archive_id"] = "scheduler-observer-effect-v1"
    payload = CompactCasePayload(
        identity=identity,
        runtime=runtime,
        requests=request_rows,
        request_field_provenance={
            CANONICAL_REQUEST_COLUMNS[0]: {
                "kind": "requested",
                "source_archive_id": "accepted-ground-truth-v3",
                "reason": None,
            },
            CANONICAL_REQUEST_COLUMNS[1]: {
                "kind": "requested",
                "source_archive_id": "accepted-ground-truth-v3",
                "reason": None,
            },
            CANONICAL_REQUEST_COLUMNS[2]: {
                "kind": "requested",
                "source_archive_id": "accepted-ground-truth-v3",
                "reason": None,
            },
            CANONICAL_REQUEST_COLUMNS[3]: {
                "kind": "scheduler_equivalent_representative",
                "source_archive_id": None,
                "reason": (
                    "logical H unavailable; accepted Scheduler trace proves "
                    "equivalent zero net prefix-work reduction"
                ),
            },
        },
        cached_hit_value_evidence=cached_hit_evidence,
        metrics=AvailabilityPayload("available", None, metrics),
        alignment_steps=alignment_payload,
        raw_sources=tuple(raw_sources),
        provenance_gaps=tuple(_sequence(
            _mapping(manifest["provenance"], "source provenance")["gaps"],
            "source provenance gaps",
        )),
        converter={
            "implementation_id": (
                "midlayer.scheduler_capture_candidate.v7"
                if capture_schema_version == OBSERVER_QUALITY_SCHEMA_VERSION
                else "midlayer.scheduler_capture_candidate.v5"
            ),
            "source_commit": converter_source_commit,
            "selection_rule": "middle_complete_work_bearing_step_v1",
        },
        schema_version=capture_schema_version,
        scheduler_steps=scheduler_steps,
        scheduler_step_schema_version=scheduler_step_schema_version,
        scheduler_timing_boundary=_TIMING_BOUNDARY,
        scheduler_provenance={
            "source_archive_id": "accepted-scheduler-trace-v3",
            "source_sha256": _sha256(scheduler_path),
            "capture_run_id": run_id,
            "instrumentation_id": scheduler_instrumentation[
                "instrumentation_id"
            ],
            "instrumentation_version": scheduler_instrumentation[
                "instrumentation_version"
            ],
            "buffering": "in_memory_append_then_single_write",
            "profiler_attached": False,
        },
        capture={
            "run_id": run_id,
            "requests_source_archive_ids": ["accepted-ground-truth-v3"],
            "metrics_source_archive_id": "accepted-ground-truth-v3",
            "scheduler_source_archive_id": "accepted-scheduler-trace-v3",
            "scheduler_trace_enabled": True,
            "profiler_attached": False,
            "observer_effect_validation": validation,
        },
    )
    destination = (
        Path(candidate_root)
        / identity["hardware_id"]
        / identity["model_id"]
        / identity["case_id"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return write_compact_case_candidate(
        destination, payload, candidate_root=candidate_root
    )


def _accepted_requests_and_metrics(
    trace_on: Mapping[str, object],
    source_sha256: str,
    requests_path: Path,
) -> tuple[
    tuple[CanonicalRequestRow, ...],
    Mapping[int, int],
    Mapping[str, object],
]:
    metrics = _mapping(trace_on["metrics"], "trace-on metrics")
    per_request = _sequence(metrics["per_request"], "trace-on per_request")
    requests = read_canonical_request_rows(requests_path)
    metric_rows_by_index = {}
    seen = set()
    for value in per_request:
        row = _mapping(value, "trace-on request")
        request_id = row["request_id"]
        if not isinstance(request_id, str):
            raise TypeError("trace-on request_id must be text")
        match = _REQUEST_ID.fullmatch(request_id)
        if match is None:
            raise ValueError(f"trace-on request_id is not canonical: {request_id}")
        old_index = int(match.group("index"))
        if old_index in seen:
            raise ValueError("trace-on result contains duplicate request_id")
        if old_index >= len(requests):
            raise ValueError("trace-on request_id exceeds canonical request rows")
        seen.add(old_index)
        expected = requests[old_index]
        if (
            row["input_tokens"] != expected.input_tokens
            or row["output_tokens"] != expected.output_tokens
        ):
            raise ValueError(
                "trace-on request token counts disagree with canonical requests"
            )
        scheduled_arrival_s = row["scheduled_arrival_s"]
        if (
            isinstance(scheduled_arrival_s, bool)
            or not isinstance(scheduled_arrival_s, (int, float))
            or not math.isfinite(scheduled_arrival_s)
            or not math.isclose(
                scheduled_arrival_s,
                expected.arrival_time_ms / 1000.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "trace-on scheduled arrival disagrees with canonical requests"
            )
        metric_rows_by_index[old_index] = {
            "request_idx": old_index,
            "actual_output_tokens": row["output_tokens"],
            "ttft_ms": row["ttft_ms"],
            "tpot_ms": row["tpot_ms"],
            "e2e_latency_ms": row["e2e_ms"],
        }
    if not metric_rows_by_index:
        raise ValueError("trace-on result contains no requests")
    if seen != set(range(len(requests))):
        raise ValueError("trace-on requests do not cover canonical requests")
    metric_rows = [
        metric_rows_by_index[request_idx]
        for request_idx in range(len(requests))
    ]
    old_to_new = {request_idx: request_idx for request_idx in range(len(requests))}
    duration_s = metrics["duration_s"]
    total_tokens = sum(
        row.input_tokens + row.output_tokens for row in requests
    )
    return requests, old_to_new, {
        "metric_provenance": {
            "source_kind": "serving_runtime",
            "source_archive_id": "accepted-ground-truth-v3",
            "source_sha256": source_sha256,
            "profiler_attached": False,
        },
        "per_request": metric_rows,
        "summary": {
            "request_count": len(requests),
            "request_throughput_per_s": metrics["request_throughput_per_s"],
            "output_token_throughput_per_s": metrics[
                "output_token_throughput_per_s"
            ],
            "total_token_throughput_per_s": total_tokens / duration_s,
        },
    }


def _remapped_scheduler_steps(
    path: Path, old_to_new: Mapping[int, int]
) -> tuple[Mapping[str, object], ...]:
    steps = []
    for line_index, line in enumerate(path.read_text().splitlines()):
        if not line:
            raise ValueError("raw scheduler trace contains an empty row")
        source = _json_mapping(line, f"raw scheduler row {line_index}")
        if source["step_index"] != line_index:
            raise ValueError("raw scheduler step indexes must be contiguous")
        step = dict(source)
        scheduled = []
        for value in _sequence(
            source["scheduled_requests"], "raw scheduled_requests"
        ):
            work = dict(_mapping(value, "raw scheduled request"))
            old_index = work["request_idx"]
            if old_index not in old_to_new:
                raise ValueError("scheduler request is absent from accepted metrics")
            work["request_idx"] = old_to_new[old_index]
            scheduled.append(work)
        scheduled.sort(key=lambda value: value["request_idx"])
        step["scheduled_requests"] = scheduled
        finished = []
        for old_index in _sequence(
            source["finished_request_indexes"], "raw finished requests"
        ):
            if old_index not in old_to_new:
                raise ValueError("finished request is absent from accepted metrics")
            finished.append(old_to_new[old_index])
        step["finished_request_indexes"] = sorted(finished)
        steps.append(step)
    return tuple(steps)


def _capture_schema_version(
    steps: Sequence[Mapping[str, object]],
) -> int:
    if not steps:
        raise ValueError("accepted scheduler trace contains no steps")
    first = _mapping(steps[0], "accepted scheduler step 0")
    if "rank_observations" in first:
        return RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION
    return SCHEDULER_CAPTURE_SCHEMA_VERSION


def _runtime_depth_metadata(
    path: Path, steps: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], Path | None]:
    if path.is_file():
        metadata = _json_object(path, "Scheduler runtime depth metadata")
        return _validate_runtime_depth_metadata(metadata), path
    if not steps:
        raise ValueError("Scheduler trace contains no steps for depth evidence")
    first = _mapping(steps[0], "Scheduler runtime depth trace row 0")
    if "scheduler_pipeline_depth_evidence" not in first:
        raise ValueError(
            "Scheduler trace is missing runtime depth metadata and embedded "
            "depth evidence")
    evidence = _mapping(
        first["scheduler_pipeline_depth_evidence"],
        "Scheduler runtime depth trace evidence",
    )
    for index, value in enumerate(steps[1:], start=1):
        row = _mapping(value, f"Scheduler runtime depth trace row {index}")
        if row["scheduler_pipeline_depth_evidence"] != evidence:
            raise ValueError("Scheduler trace has inconsistent embedded depth evidence")
    metadata = {
        "schema_version": 1,
        "scheduler_pipeline_depth": evidence["configured_depth"],
        "scheduler_pipeline_depth_evidence": evidence,
    }
    return _validate_runtime_depth_metadata(metadata), None


def _validate_runtime_depth_metadata(
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    if set(metadata) != {
            "schema_version", "scheduler_pipeline_depth",
            "scheduler_pipeline_depth_evidence"}:
        raise ValueError("Scheduler runtime depth metadata keys are invalid")
    if metadata["schema_version"] != 1:
        raise ValueError("Scheduler runtime depth metadata schema is invalid")
    depth = metadata["scheduler_pipeline_depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError(
            "Scheduler runtime metadata depth must be a positive non-bool "
            "integer")
    evidence = _mapping(
        metadata["scheduler_pipeline_depth_evidence"],
        "Scheduler runtime depth evidence",
    )
    source_pair = (evidence["source_kind"], evidence["source_field"])
    if source_pair == ("runtime_executor_property", "max_concurrent_batches"):
        expected_keys = {
            "source_kind", "source_field", "executor_class",
            "pipeline_parallel_size", "runtime_version", "source_identity"}
    elif source_pair == (
            "runtime_vllm_config_field", "vllm_config.max_concurrent_batches"):
        expected_keys = {
            "source_kind", "source_field", "configured_depth", "executor_class",
            "pipeline_parallel_size", "runtime_version", "source_identity"}
        if evidence["configured_depth"] != depth:
            raise ValueError("Scheduler runtime depth evidence configured depth is invalid")
    else:
        raise ValueError("Scheduler runtime depth evidence source is invalid")
    if set(evidence) != expected_keys:
        raise ValueError("Scheduler runtime depth evidence keys are invalid")
    for field in ("executor_class", "runtime_version"):
        if not isinstance(evidence[field], str) or not evidence[field]:
            raise ValueError(
                f"Scheduler runtime depth evidence {field} is invalid")
    pp = evidence["pipeline_parallel_size"]
    if isinstance(pp, bool) or not isinstance(pp, int) or pp < 1:
        raise ValueError("Scheduler runtime depth evidence PP is invalid")
    if source_pair == ("runtime_executor_property", "max_concurrent_batches"):
        if (not isinstance(evidence["source_identity"], str)
                or not evidence["source_identity"]):
            raise ValueError("Scheduler runtime depth evidence source identity is invalid")
    elif source_pair == (
            "runtime_vllm_config_field", "vllm_config.max_concurrent_batches"):
        identity = _mapping(
            evidence["source_identity"], "Scheduler runtime depth source identity"
        )
        if set(identity) != {"module_file", "module_name"}:
            raise ValueError("Scheduler runtime depth source identity keys are invalid")
        for field in ("module_file", "module_name"):
            if not isinstance(identity[field], str) or not identity[field]:
                raise ValueError(
                    f"Scheduler runtime depth source identity {field} is invalid")
    else:
        raise ValueError("Scheduler runtime depth evidence source is invalid")
    return metadata


def _rebound_alignment(
    source: Mapping[str, object], raw_source_sha: str
) -> Mapping[str, object]:
    alignment = json.loads(json.dumps(source, allow_nan=False))
    if alignment["schema_version"] not in (
        2,
        3,
        RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
        LATEST_SCHEMA_VERSION,
    ):
        raise ValueError("source alignment must use a supported schema")
    del alignment["schema_version"]
    if alignment["kind"] != "sharegpt_alignment_steps":
        raise ValueError("source alignment kind is invalid")
    del alignment["kind"]
    phases = _mapping(alignment["phases"], "source alignment phases")
    for phase in ("prefill", "decode", "mixed"):
        entry = _mapping(phases[phase], f"source alignment {phase}")
        if entry["status"] == "not_observed":
            continue
        if entry["status"] != "available":
            raise ValueError("source alignment phase status is invalid")
        sample = _mapping(entry["sample"], f"source alignment {phase} sample")
        provenance = dict(_mapping(
            sample["provenance"], f"source alignment {phase} provenance"
        ))
        provenance["raw_source_set_sha256"] = raw_source_sha
        sample["provenance"] = provenance
    return alignment


def _validate_capture_inputs(
    trace_on: Mapping[str, object],
    scheduler_path: Path,
    observer: Mapping[str, object],
    trace_on_path: Path,
    observer_path: Path,
    observer_validation_status: str,
) -> None:
    if observer_validation_status not in (
        "measured_for_this_case",
        "inherited_from_representative_case",
    ):
        raise ValueError("observer validation status is invalid")
    if trace_on["schema_version"] != 1 or trace_on["status"] != "success":
        raise ValueError("accepted trace-on input must be a successful schema-v1 run")
    server = _mapping(trace_on["server"], "trace-on server")
    if _mapping(server["profiler"], "trace-on profiler")["requested"] is not False:
        raise ValueError("accepted trace-on input must not use a profiler")
    scheduler = _mapping(
        trace_on["scheduler_trace"], "trace-on scheduler trace summary"
    )
    if scheduler["sha256"] != _sha256(scheduler_path):
        raise ValueError("trace-on Scheduler SHA256 disagrees with supplied file")
    if observer["kind"] != "scheduler_trace_observer_effect":
        raise ValueError("observer-effect input kind is invalid")
    if (observer_validation_status == "inherited_from_representative_case"
            and observer["passed"] is not True):
        raise ValueError("observer-effect gate did not pass")
    observer_trace_on = _mapping(observer["trace_on"], "observer trace_on")
    identity = _mapping(
        _mapping(trace_on["workload"], "trace-on workload")["identity"],
        "trace-on identity",
    )
    case_id = "/".join(str(identity[field]) for field in (
        "hardware_id", "model_id", "case_id"
    ))
    validation_case_id = observer["validation_case_id"]
    if not isinstance(validation_case_id, str):
        raise TypeError("observer validation_case_id must be text")
    if observer_validation_status == "measured_for_this_case":
        if validation_case_id != case_id:
            raise ValueError("measured observer-effect case identity disagrees")
        if observer_trace_on["sha256"] != _sha256(trace_on_path):
            raise ValueError("observer-effect trace-on SHA256 disagrees")
        if identity["hardware_id"] == "nvidia_h200":
            if observer_path.parent != trace_on_path.parent:
                raise ValueError("own observer must share the trace-on run directory")
            baseline = _mapping(observer["trace_off"], "observer trace_off")
            baseline_run = baseline["run_id"]
            if (not isinstance(baseline_run, str)
                    or re.fullmatch(
                        r"trace_off(?:_[A-Za-z0-9_]+)?/ground_truth\.json",
                        baseline_run) is None):
                raise ValueError("own observer trace-off run identity is invalid")
            trace_off_path = observer_path.parent.parent / baseline_run
            if (observer_trace_on["run_id"] != _run_id(trace_on_path)
                    or baseline["sha256"] != _sha256(trace_off_path)):
                raise ValueError("own observer run or trace-off SHA256 disagrees")
            trace_off = _json_object(trace_off_path, "trace-off result")
            if (trace_off["schema_version"] != 1
                    or trace_off["status"] != "success"
                    or _mapping(trace_off["server"], "trace-off server")
                    ["profiler"]["requested"] is not False):
                raise ValueError("own observer trace-off result is invalid")
            off_metrics = _mapping(trace_off["metrics"], "trace-off metrics")
            on_metrics = _mapping(trace_on["metrics"], "trace-on metrics")
            deltas = _mapping(observer["metric_delta_percent"], "observer deltas")
            values = _mapping(observer["metric_values"], "observer metric values")
            metric_paths = {
                "request_throughput": ("request_throughput_per_s", None),
                "output_token_throughput": ("output_token_throughput_per_s", None),
                "ttft": ("ttft", "mean_ms"),
                "tpot": ("tpot", "mean_ms"),
                "e2e": ("e2e", "mean_ms"),
            }
            if set(deltas) != set(metric_paths) or set(values) != set(metric_paths):
                raise ValueError("own observer must preserve all five metrics")
            for field, (section, statistic) in metric_paths.items():
                pair = _mapping(values[field], f"observer {field} values")
                if set(pair) != {"trace_off", "trace_on"}:
                    raise ValueError("observer metric value pair is incomplete")
                off, on = pair["trace_off"], pair["trace_on"]
                if (not isinstance(off, (int, float)) or isinstance(off, bool)
                        or not math.isfinite(off) or off <= 0
                        or not isinstance(on, (int, float)) or isinstance(on, bool)
                        or not math.isfinite(on) or on <= 0):
                    raise ValueError("observer metric values are invalid")
                actual_off = (off_metrics[section] if statistic is None else
                              _mapping(off_metrics[section],
                                       f"trace-off {section}")[statistic])
                actual_on = (on_metrics[section] if statistic is None else
                             _mapping(on_metrics[section],
                                      f"trace-on {section}")[statistic])
                if (type(actual_off) not in (int, float)
                        or type(actual_on) not in (int, float)
                        or not math.isfinite(actual_off)
                        or not math.isfinite(actual_on)
                        or actual_off <= 0 or actual_on <= 0
                        or off != actual_off or on != actual_on):
                    raise ValueError(
                        "observer metric values disagree with bound ground truth")
                expected = abs(on - off) / off * 100.0
                if not math.isclose(deltas[field], expected, rel_tol=1e-10,
                                    abs_tol=1e-10):
                    raise ValueError("observer delta disagrees with raw metric values")
            if observer["passed"] is not all(
                    deltas[field] <= observer["threshold_percent"]
                    for field in metric_paths):
                raise ValueError("observer pass flag disagrees with five deltas")
    elif validation_case_id.split("/", 1)[0] != identity["hardware_id"]:
        raise ValueError(
            "inherited observer-effect evidence must use the same hardware"
        )


def _normalize_observer_effect(
    observer: Mapping[str, object], *, observer_validation_status: str,
) -> Mapping[str, object]:
    """Accept the explicitly selected legacy representative evidence format."""

    if "kind" in observer:
        return observer
    legacy_keys = {
        "baseline_run_id", "instrumented_run_id", "metric_delta_percent", "passed",
        "status", "threshold_percent", "validation_case_id",
    }
    if set(observer) != legacy_keys:
        raise ValueError("observer-effect result has an unsupported schema")
    if observer_validation_status != "inherited_from_representative_case":
        raise ValueError(
            "legacy observer evidence may only be used as an inherited representative"
        )
    if observer["status"] != "measured_for_this_case":
        raise ValueError("legacy observer evidence must record a measured source case")
    metric_deltas = _mapping(observer["metric_delta_percent"], "legacy metric deltas")
    if set(metric_deltas) != {
        "request_throughput", "output_token_throughput", "ttft", "tpot", "e2e",
    }:
        raise ValueError("legacy observer metric deltas are incomplete")
    return {
        "kind": "scheduler_trace_observer_effect",
        "validation_case_id": observer["validation_case_id"],
        "trace_off": {"run_id": observer["baseline_run_id"]},
        "trace_on": {"run_id": observer["instrumented_run_id"]},
        "threshold_percent": observer["threshold_percent"],
        "metric_delta_percent": metric_deltas,
        "passed": observer["passed"],
    }


def _run_id(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _raw_source(
    archive_id: str,
    role: str,
    path: Path,
    tool: str,
    tool_version: str,
) -> Mapping[str, object]:
    return {
        "archive_id": archive_id,
        "role": role,
        "relative_path": f"{path.parent.name}/{path.name}",
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "tool": tool,
        "tool_version": tool_version,
    }


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    return _json_mapping(path.read_text(), label)


def _json_bytes_object(content: bytes, label: str) -> Mapping[str, object]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    return _json_mapping(text, label)


def _json_mapping(text: str, label: str) -> Mapping[str, object]:
    value = json.loads(text)
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    return compact_mapping(value, label)


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value
