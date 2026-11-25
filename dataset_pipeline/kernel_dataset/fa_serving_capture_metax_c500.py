"""Fail-closed MetaX C500 candidate audits for FA serving capture.

This owner validates only evidence produced by the frozen serving manifest.  A
single observed final kernel is valid when the public operation executes with
one split; a separate reduction kernel is required only when the observation
actually contains multiple split bodies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from dataset_pipeline.kernel_dataset.metax_c500 import (
    _allk_real_files,
    _join_layer1_observations,
    _parse_trace_json,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path, context: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON mapping: {path}")
    return value


def _positive_number(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{context} must be finite and positive")
    return number


def _nonnegative_number(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context} must be finite and non-negative")
    return number


def _require_hex(value, length: int, context: str) -> str:
    if (not isinstance(value, str) or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{context} must be lower hexadecimal")
    return value


def _resolve_under(path: Path, root: Path, context: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        common = Path(os.path.commonpath((resolved_root, resolved_path)))
    except ValueError as error:
        raise ValueError(f"{context} is outside the C500 case root") from error
    if common != resolved_root:
        raise ValueError(f"{context} is outside the C500 case root")
    return resolved_path


def _event_repetition_records(event: dict) -> tuple[dict, dict]:
    measurement = event["measurement"]
    if measurement["mode"] != "external_events_around_cuda_graph_replay":
        raise ValueError("C500 FA Event timing must use Graph replay")
    runs = measurement["repetition_runs"]
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("C500 FA Event timing requires exactly 50/200 runs")
    by_repetition = {}
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("C500 Event repetition run must be a mapping")
        repetitions = run["repetition_count"]
        if repetitions not in (50, 200):
            raise ValueError("C500 FA Event repetition count is not 50 or 200")
        if repetitions in by_repetition:
            raise ValueError("C500 FA Event repetition count is duplicated")
        if run["sample_count"] != 20:
            raise ValueError("C500 FA Event sample count must be 20")
        statistics = run["statistics"]
        _positive_number(statistics["median_ns"], "C500 Event median")
        if _nonnegative_number(
                statistics["coefficient_of_variation"], "C500 Event CV") > 0.05:
            raise ValueError("C500 FA Event selected samples exceed the 5% CV gate")
        by_repetition[repetitions] = run
    if set(by_repetition) != {50, 200}:
        raise ValueError("C500 FA Event repetitions are incomplete")
    return by_repetition[50], by_repetition[200]


def _require_event_identity(
        event, case_id, source_snapshot_sha256, logical_identity_sha256,
        workload_digest):
    """Require one C500 Event record to carry the capture's own identity.

    The Event top level and its workload capture identity must both name this
    hardware, this family and this case, and must agree with the captured
    source snapshot, logical identity and workload digest.  Any disagreement
    means the Event belongs to another capture.
    """

    for field, expected in (
            ("hardware_id", "metax_c500"),
            ("family", "flash_attn"),
            ("case_id", case_id),
            ("source_snapshot_sha256", source_snapshot_sha256),
            ("canonical_attributes_sha256", logical_identity_sha256),
            ("status", "measured")):
        if event[field] != expected:
            raise ValueError(f"C500 Event {field} disagrees with capture contract")
    identity = event["workload"]["capture_identity"]
    for field, expected in (
            ("case_id", case_id),
            ("hardware_id", "metax_c500"),
            ("logical_identity_sha256", logical_identity_sha256),
            ("workload_digest", workload_digest),
            ("requested_num_splits", 1),
            ("kv_page_size_tokens", 16)):
        if identity[field] != expected:
            raise ValueError(f"C500 Event capture identity {field} disagrees")


def build_c500_case_evidence(*, case_id: str, workload_digest: str,
                              logical_identity_sha256: str,
                              source_snapshot_sha256: str,
                              case_root: Path, event_path: Path, trace_paths: list[Path],
                              allk_dir: Path) -> tuple[dict, dict]:
    """Build timing and duration-free counter evidence for one C500 case."""
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("C500 case_id must be non-empty")
    _require_hex(workload_digest, 64, "C500 workload digest")
    _require_hex(logical_identity_sha256, 64, "C500 logical identity")
    _require_hex(source_snapshot_sha256, 64, "C500 source snapshot")
    if not trace_paths or len(set(trace_paths)) != len(trace_paths):
        raise ValueError("C500 trace paths must be non-empty and unique")
    case_root = case_root.resolve()
    if not case_root.is_dir():
        raise FileNotFoundError(f"C500 case root does not exist: {case_root}")
    event_path = _resolve_under(Path(event_path), case_root, "C500 Event raw")
    allk_dir = _resolve_under(Path(allk_dir), case_root, "C500 layer-1 directory")
    trace_paths = [
        _resolve_under(Path(path), case_root, "C500 mcTracer raw")
        for path in trace_paths
    ]

    event = _load_mapping(event_path, "C500 Event raw")
    _require_event_identity(
        event, case_id, source_snapshot_sha256, logical_identity_sha256,
        workload_digest)

    replay_50, replay_200 = _event_repetition_records(event)
    median_50 = _positive_number(
        replay_50["statistics"]["median_ns"], "C500 50-replay median")
    median_200 = _positive_number(
        replay_200["statistics"]["median_ns"], "C500 200-replay median")
    median_difference = abs(median_50 - median_200)
    median_tolerance = max(500.0, 0.05 * max(median_50, median_200))
    if median_difference > median_tolerance:
        raise ValueError("C500 FA 50/200 Event medians exceed the stability gate")
    selected_statistics = event["measurement"]["statistics"]
    selected_median = _positive_number(
        selected_statistics["median_ns"], "C500 selected Event median")
    if _nonnegative_number(
            selected_statistics["coefficient_of_variation"],
            "C500 selected Event CV") > 0.05:
        raise ValueError("C500 FA selected Event samples exceed the 5% CV gate")

    parsed_traces = []
    trace_records = []
    expected_sequence = None
    for path in trace_paths:
        trace = _parse_trace_json(str(path))
        sequence = tuple(
            (record["name"], record["grid_x"], record["grid_y"],
             record["grid_z"], record["block_size"],
             record["regs_per_thread"], record["dynamic_smem"],
             record["static_smem"])
            for record in trace)
        if expected_sequence is None:
            expected_sequence = sequence
        elif sequence != expected_sequence:
            raise ValueError("C500 mcTracer shards have different launch sequences")
        parsed_traces.append(trace)
        trace_records.append({
            "path": str(path),
            "sha256": _sha256(path),
        })
    if expected_sequence is None:
        raise AssertionError("C500 trace sequence was not initialized")
    trace = parsed_traces[0]
    allk_files = _allk_real_files(str(allk_dir))
    joined, unused = _join_layer1_observations(str(allk_dir), trace, case_id)
    if unused:
        raise ValueError("C500 layer-1 counter files were not consumed")
    if len(joined) != len(trace) or len(allk_files) != len(trace):
        raise ValueError("C500 trace and layer-1 counter ordinal counts differ")

    physical_timing = []
    physical_counters = []
    for ordinal, ((launch, counters), counter_path) in enumerate(
            zip(joined, allk_files, strict=True), start=0):
        physical_timing.append({
            "ordinal": ordinal,
            "program": launch["name"],
            "grid": [launch["grid_x"], launch["grid_y"], launch["grid_z"]],
            "block_size": launch["block_size"],
            "kernel_wall_time_ns": _positive_number(
                launch["dur_ns"], "C500 mcTracer kernel wall time"),
        })
        physical_counters.append({
            "ordinal": ordinal,
            "program": launch["name"],
            "grid": [launch["grid_x"], launch["grid_y"], launch["grid_z"]],
            "block_size": launch["block_size"],
            "registers_per_thread": launch["regs_per_thread"],
            "dynamic_shared_bytes": launch["dynamic_smem"],
            "static_shared_bytes": launch["static_smem"],
            "counter_path": str(counter_path),
            "counter_sha256": _sha256(Path(counter_path)),
            "counters": {
                "total_cycles": counters["total_cycles"],
                "ap_active_cycles": counters["ap_active_cycles"],
                "ap_mma_duty_pct": counters["ap_mma_duty_pct"],
                "ap_mte_duty_pct": counters["ap_mte_duty_pct"],
                "ap_ste_duty_pct": counters["ap_ste_duty_pct"],
                "vls_duty_pct": counters["vls_duty_pct"],
                "l2c_duty_pct": counters["l2c_duty_pct"],
                "vl1_hit_rate_pct": counters["vl1_hit_rate_pct"],
                "l2c_hit_rate_pct": counters["l2c_hit_rate_pct"],
                "global_memory_read_bytes": counters["global_memory_read_bytes"],
                "global_memory_write_bytes": counters["global_memory_write_bytes"],
                "achieved_waves": counters["achieved_waves"],
                "dispatched_waves": counters["dispatched_waves"],
            },
        })

    observed_split_kernel_count = sum(
        "flash_fwd_splitkv_kernel" in record["program"]
        for record in physical_timing)
    if observed_split_kernel_count != 1:
        raise ValueError(
            "C500 split=1 public call must expose exactly one final SplitKV kernel")
    timing = {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_metax_c500_event_timing",
        "status": "pass",
        "hardware": {"hardware_id": "metax_c500"},
        "case_id": case_id,
        "workload_digest": workload_digest,
        "source_snapshot": {"sha256": source_snapshot_sha256},
        "event": {
            "path": str(event_path),
            "sha256": _sha256(event_path),
            "selection": "200-replay Graph Device Event median after 50/200 agreement",
            "median_duration_ns": selected_median,
            "replay_50_median_ns": median_50,
            "replay_200_median_ns": median_200,
            "median_difference_ns": median_difference,
            "median_difference_tolerance_ns": median_tolerance,
            "selected_coefficient_of_variation": selected_statistics[
                "coefficient_of_variation"],
        },
        "mctracer": {
            "shards": trace_records,
            "kernels": physical_timing,
        },
        "observed_execution": {
            "requested_num_splits": 1,
            "separate_combine_kernel_required": False,
            "reason": "one split produces one final result and no reduction stage",
        },
    }
    candidate = {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_metax_c500_counter",
        "hardware_id": "metax_c500",
        "case_id": case_id,
        "workload_digest": workload_digest,
        "source_snapshot_sha256": source_snapshot_sha256,
        "counter_contract": {
            "raw_tool": "mcProfiler layer-1 per-kernel",
            "duration_consumed": False,
            "excluded_metric_classes": [
                "during", "kernel_wall_time", "event_duration", "frequency"],
        },
        "timing_launch_join": {
            "join_key": "ordinal + normalized program + grid + block",
            "mctracer_shards": trace_records,
        },
        "kernels": physical_counters,
    }
    return timing, candidate


def build_c500_timing_without_counter(*, case_id: str, workload_digest: str,
                                      logical_identity_sha256: str,
                                      source_snapshot_sha256: str,
                                      event_path: Path,
                                      trace_paths: list[Path]) -> dict:
    """Build Event/trace timing evidence when per-invocation counters are absent.

    This is intentionally timing-only.  It refuses an aggregate mcProfiler
    report as a counter substitute; publication records the counter absence
    separately rather than turning diagnostic aggregate data into a kernel
    observation.
    """
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("C500 case_id must be non-empty")
    _require_hex(workload_digest, 64, "C500 workload digest")
    _require_hex(logical_identity_sha256, 64, "C500 logical identity")
    _require_hex(source_snapshot_sha256, 64, "C500 source snapshot")
    if not trace_paths or len(set(trace_paths)) != len(trace_paths):
        raise ValueError("C500 trace paths must be non-empty and unique")
    event_path = Path(event_path).resolve()
    event = _load_mapping(event_path, "C500 Event raw")
    _require_event_identity(
        event, case_id, source_snapshot_sha256, logical_identity_sha256,
        workload_digest)
    replay_50, replay_200 = _event_repetition_records(event)
    median_50 = _positive_number(replay_50["statistics"]["median_ns"], "C500 50-replay median")
    median_200 = _positive_number(replay_200["statistics"]["median_ns"], "C500 200-replay median")
    median_difference = abs(median_50 - median_200)
    median_tolerance = max(500.0, 0.05 * max(median_50, median_200))
    if median_difference > median_tolerance:
        raise ValueError("C500 FA 50/200 Event medians exceed the stability gate")
    statistics = event["measurement"]["statistics"]
    selected_median = _positive_number(statistics["median_ns"], "C500 selected Event median")
    if _nonnegative_number(statistics["coefficient_of_variation"], "C500 Event CV") > 0.05:
        raise ValueError("C500 FA selected Event samples exceed the 5% CV gate")
    traces = []
    trace_records = []
    sequence = None
    for raw_path in trace_paths:
        path = Path(raw_path).resolve()
        parsed = _parse_trace_json(str(path))
        current_sequence = tuple(
            (record["name"], record["grid_x"], record["grid_y"], record["grid_z"],
             record["block_size"]) for record in parsed)
        if sequence is None:
            sequence = current_sequence
        elif current_sequence != sequence:
            raise ValueError("C500 mcTracer shards have different launch sequences")
        traces.append(parsed)
        trace_records.append({"path": str(path), "sha256": _sha256(path)})
    if sequence is None:
        raise AssertionError("C500 trace sequence was not initialized")
    physical_timing = []
    for ordinal, launch in enumerate(traces[0]):
        physical_timing.append({
            "ordinal": ordinal, "program": launch["name"],
            "grid": [launch["grid_x"], launch["grid_y"], launch["grid_z"]],
            "block_size": launch["block_size"],
            "kernel_wall_time_ns": _positive_number(
                launch["dur_ns"], "C500 mcTracer kernel wall time"),
        })
    if sum("flash_fwd_splitkv_kernel" in item["program"] for item in physical_timing) != 1:
        raise ValueError("C500 split=1 call must expose exactly one final SplitKV kernel")
    if any("flash_fwd_splitkv_combine_kernel" in item["program"] for item in physical_timing):
        raise ValueError("C500 split=1 call emitted an unexpected combine kernel")
    return {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_metax_c500_event_timing",
        "status": "pass",
        "hardware": {"hardware_id": "metax_c500"},
        "case_id": case_id,
        "workload_digest": workload_digest,
        "source_snapshot": {"sha256": source_snapshot_sha256},
        "event": {
            "path": str(event_path), "sha256": _sha256(event_path),
            "selection": "200-replay Graph Device Event median after 50/200 agreement",
            "median_duration_ns": selected_median,
            "replay_50_median_ns": median_50,
            "replay_200_median_ns": median_200,
            "median_difference_ns": median_difference,
            "median_difference_tolerance_ns": median_tolerance,
            "selected_coefficient_of_variation": statistics["coefficient_of_variation"],
        },
        "mctracer": {"shards": trace_records, "kernels": physical_timing},
        "observed_execution": {
            "requested_num_splits": 1,
            "separate_combine_kernel_required": False,
            "reason": "one split produces one final result and no reduction stage",
        },
        "counter_status": "unavailable",
        "counter_unavailable_reason": (
            "mcProfiler did not emit a per-invocation kernel report after 15 s, "
            "60 s, and 120 s attach waits; aggregate report is diagnostic-only"),
    }


def build_c500_canary_closure(*, case_id: str, workload_digest: str,
                               timing_audit_path: Path,
                               counter_candidate_path: Path) -> dict:
    """Close a C500 canary only after its Event, trace, and counters agree."""
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("C500 canary case_id must be non-empty")
    _require_hex(workload_digest, 64, "C500 canary workload digest")
    timing_path = Path(timing_audit_path)
    candidate_path = Path(counter_candidate_path)
    timing = _load_mapping(timing_path, "C500 canary timing audit")
    candidate = _load_mapping(candidate_path, "C500 canary counter candidate")
    if timing["status"] != "pass":
        raise ValueError("C500 canary Event timing did not pass")
    if timing["hardware"]["hardware_id"] != "metax_c500":
        raise ValueError("C500 canary timing hardware does not match")
    if candidate["hardware_id"] != "metax_c500":
        raise ValueError("C500 canary counter hardware does not match")
    for field, expected in (("case_id", case_id), ("workload_digest", workload_digest)):
        if timing[field] != expected or candidate[field] != expected:
            raise ValueError(f"C500 canary {field} does not match")
    if (timing["source_snapshot"]["sha256"]
            != candidate["source_snapshot_sha256"]):
        raise ValueError("C500 canary timing and counter snapshots differ")
    if candidate["counter_contract"]["duration_consumed"] is not False:
        raise ValueError("C500 canary counter candidate consumed duration")
    timing_kernels = timing["mctracer"]["kernels"]
    counter_kernels = candidate["kernels"]
    if not timing_kernels or len(timing_kernels) != len(counter_kernels):
        raise ValueError("C500 canary launch join is incomplete")
    if sum("flash_fwd_splitkv_kernel" in kernel["program"]
           for kernel in timing_kernels) != 1:
        raise ValueError("C500 canary has an invalid one-split final kernel sequence")
    if timing["mctracer"]["shards"] != candidate["timing_launch_join"]["mctracer_shards"]:
        raise ValueError("C500 canary trace provenance differs between timing and counters")
    return {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_canary_closure",
        "status": "pass",
        "hardware_id": "metax_c500",
        "case_id": case_id,
        "workload_digest": workload_digest,
        "source_snapshot_sha256": timing["source_snapshot"]["sha256"],
        "timing": {
            "path": str(timing_path),
            "sha256": _sha256(timing_path),
            "stability_pass": True,
        },
        "counter_candidate": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
            "duration_consumed": False,
        },
        "gate": {
            "runner_validation": True,
            "golden_timing": True,
            "non_duration_counter": True,
            "ordinal_launch_join": True,
            "provenance": True,
            "candidate_aggregation": True,
            "stability": True,
        },
    }
