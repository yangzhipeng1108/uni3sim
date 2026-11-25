#!/usr/bin/env python3
"""Replay the canonical 910B schedule with current Forward step durations."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping, Sequence

from evaluation.serving.metrics import (
    ascend_910b_same_trajectory_reduction as same_trajectory,
)
from evaluation.serving.traces import scheduling_trajectory_replay
from evaluation.serving.metrics import (
    sharegpt_compact_evaluation as compact_evaluation,
)
from src.serving.host_timing import load_host_launch_profile
from evaluation.json_shape import (
    file_sha256 as _sha256,
    percentile_value as _percentile,
    require_finite as _finite,
    require_list_or_tuple as _sequence,
    require_object_dict as _mapping,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CELL_ID = "ascend_910b/qwen3-4b/tp1_qps16"
CASE_RELATIVE_PATH = Path(
    "serving_dataset/sharegpt/ascend_910b/qwen3-4b/tp1_qps16"
)
HOST_PROFILE_RELATIVE_PATH = Path(
    "configs/host_timing/host_launch_profiles/ascend_910b.json"
)
EXPECTED_REQUESTS_SHA256 = (
    "fcd3a78294381b34d0d4fe4f8ef5940c6728c2a9c02b70aa5db1f3c7b95c3286"
)
EXPECTED_SCHEDULER_SHA256 = (
    "e6ca8297975c222eeb74087d624499c80b76a3e7a0b3b7acdb62653fe03a8326"
)
LATENCY_FIELDS = ("ttft_ms", "tpot_ms", "e2e_latency_ms")


def _request_contract(case: object) -> list[dict[str, object]]:
    requests = []
    for index, request in enumerate(case.requests):
        if request.cached_hit_tokens != 0:
            raise ValueError("canonical counterfactual requires zero effective prefix hit")
        if request.output_tokens <= 1:
            raise ValueError("canonical counterfactual requires at least two output tokens")
        requests.append(
            {
                "request_id": f"request-{index:012d}",
                "arrival_time_s": request.arrival_time_ms / 1000.0,
                "input_tokens": request.input_tokens,
                "output_tokens": request.output_tokens,
            }
        )
    return requests


def _summarize_values(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        **{
            f"p{percentile}": _percentile(values, percentile)
            for percentile in (0, 50, 90, 99, 100)
        },
    }


def _request_metrics(
    scheduler_rows: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    completion_times_s: Sequence[float],
) -> dict[str, object]:
    return scheduling_trajectory_replay.reconstruct_request_metrics(
        scheduler_rows, requests, completion_times_s
    )


def _measured_metrics(case: object) -> dict[str, object]:
    metrics = _mapping(case.metrics, "canonical metrics")
    rows = _sequence(metrics["per_request"], "canonical per-request metrics")
    if len(rows) != len(case.requests):
        raise ValueError("canonical metrics request count differs")
    per_request = []
    for index, value in enumerate(rows):
        row = _mapping(value, f"canonical metric request {index}")
        if row["request_idx"] != index:
            raise ValueError("canonical metric request indexes are not contiguous")
        per_request.append(
            {
                "request_idx": index,
                **{field: _finite(row[field], field) for field in LATENCY_FIELDS},
            }
        )
    summary = _mapping(metrics["summary"], "canonical metric summary")
    return {
        "request_count": len(per_request),
        "request_throughput_per_s": _finite(
            summary["request_throughput_per_s"], "measured request throughput"
        ),
        "latency": {
            field: _summarize_values([float(row[field]) for row in per_request])
            for field in LATENCY_FIELDS
        },
        "per_request": per_request,
    }


def _standard_prediction(path: Path) -> dict[str, object]:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "evaluation")
    matches = [
        _mapping(value, "evaluation cell")
        for value in _sequence(payload["cells"], "evaluation cells")
        if _mapping(value, "evaluation cell")["cell_id"] == CELL_ID
    ]
    if len(matches) != 1:
        raise ValueError("standard evaluation must contain the exact target cell")
    cell = matches[0]
    metrics = _mapping(cell["metrics"], "standard metrics")
    latency = _mapping(metrics["latency"], "standard latency")
    throughput = _mapping(metrics["throughput"], "standard throughput")
    host_timing = _mapping(cell["host_timing"], "standard Host timing")
    return {
        "step_count": len(_sequence(host_timing["steps"], "standard Host steps")),
        "request_throughput_per_s": _finite(
            _mapping(
                throughput["request_throughput_per_s"],
                "standard request throughput",
            )["predicted"],
            "standard predicted request throughput",
        ),
        "latency": {
            field: {
                "mean": _finite(
                    _mapping(
                        _mapping(latency[field], field)["mean"],
                        f"standard {field} mean",
                    )["predicted"],
                    f"standard predicted {field}",
                )
            }
            for field in LATENCY_FIELDS
        },
    }


def run_counterfactual(
    repository_root: Path,
    case: object,
    standard_results_path: Path,
) -> dict[str, object]:
    root = repository_root.resolve()
    case_path = (root / CASE_RELATIVE_PATH).resolve()
    if case.case_dir.resolve() != case_path:
        raise ValueError("counterfactual received the wrong canonical case")
    requests_path = case_path / "requests.csv"
    scheduler_path = case_path / "scheduler_steps.jsonl.zst"
    if _sha256(requests_path) != EXPECTED_REQUESTS_SHA256:
        raise ValueError("canonical request stream SHA256 changed")
    if case.manifest["scheduler_steps"]["sha256"] != EXPECTED_SCHEDULER_SHA256:
        raise ValueError("canonical Scheduler trace SHA256 changed")
    requests = _request_contract(case)
    scheduler_rows, scheduler_summary = scheduling_trajectory_replay.load_and_validate_scheduler_trace(
        scheduler_path, requests, schema_version=3
    )
    config_path = root / "configs/evaluation/experiments/sharegpt_phase1.json"
    config = compact_evaluation.load_phase1_config(config_path)
    cell = compact_evaluation.select_phase1_cells(config, [CELL_ID], all_cases=False)[0]
    profile_path = root / HOST_PROFILE_RELATIVE_PATH
    profile = load_host_launch_profile(profile_path)
    predictions, contributions, predictor_identity = (
        scheduling_trajectory_replay.predict_scheduling_trajectory_steps(
            root,
            config_path,
            CELL_ID,
            scheduler_rows,
            requests,
            cell.model.model_config_sha256,
            host_launch_profile=profile,
        )
    )
    device_s = [
        float(prediction["predicted_timeline_ns"]) / 1_000_000_000.0
        for prediction in predictions
    ]
    device_launch_s = [
        float(prediction["predicted_device_plus_launch_ns"])
        / 1_000_000_000.0
        for prediction in predictions
    ]
    modes = {}
    for name, services, scheduler_cpu in (
        ("device_only", device_s, False),
        ("device_plus_launch", device_launch_s, False),
        ("device_plus_launch_plus_observed_scheduler_cpu", device_launch_s, True),
    ):
        completion_times, timing = scheduling_trajectory_replay.replay_scheduling_trajectory(
            scheduler_rows,
            requests,
            services,
            include_observed_scheduler_cpu=scheduler_cpu,
            pipeline_depth=2,
        )
        modes[name] = {
            "timing": timing,
            "metrics": scheduling_trajectory_replay.reconstruct_request_metrics(
                scheduler_rows, requests, completion_times
            ),
        }
    trace_completion_s = [
        int(row["step_end_ns"]) / 1_000_000_000.0 for row in scheduler_rows
    ]
    trace_metrics = scheduling_trajectory_replay.reconstruct_request_metrics(
        scheduler_rows, requests, trace_completion_s
    )
    measured = _measured_metrics(case)
    standard = _standard_prediction(standard_results_path)
    _, observed_service = scheduling_trajectory_replay.observed_serialized_service_times(
        scheduler_rows
    )
    prediction_totals = scheduling_trajectory_replay.summarize_prediction_totals(
        scheduler_rows, predictions
    )
    actual_total_s = float(observed_service["total_s"])
    predicted_total_s = float(prediction_totals["device"]["total_s"])
    phase_service_error = {}
    for phase in same_trajectory.PHASES:
        actual_phase_s = float(observed_service["by_phase_s"][phase])
        predicted_phase_s = float(prediction_totals["device"]["by_phase_s"][phase])
        phase_service_error[phase] = {
            "observed_model_runtime_service_s": actual_phase_s,
            "predicted_device_s": predicted_phase_s,
            "prediction_over_observed": predicted_phase_s / actual_phase_s,
            "underprediction_percent": 100.0 * (actual_phase_s - predicted_phase_s) / actual_phase_s,
        }
    selected_replay = modes[
        "device_plus_launch_plus_observed_scheduler_cpu"
    ]["metrics"]
    return {
        "schema_version": 2,
        "cell_id": CELL_ID,
        "status": "formal_canonical_scheduling_trajectory_replay",
        "forward_duration_observation_consumed": False,
        "counterfactual_contract": {
            "schedule_membership_and_order": "canonical_observed_1271_steps",
            "arrival_times": "canonical_requests_csv",
            "pipeline": "two_in_flight_single_serial_device",
            "first_token": "completion_of_step_reaching_full_prompt",
            "decode_emission": "one_output_token_per_decode_schedule",
            "completion": "completion_of_finished_request_step",
            "caveat": (
                "The canonical trace has no common absolute clock with request "
                "timestamps. Zero origin offset is retained; the resulting request "
                "boundary residual is reported rather than fitted away."
            ),
        },
        "source_sha256": {
            "requests": _sha256(requests_path),
            "scheduler_trace": _sha256(scheduler_path),
            "standard_evaluation": _sha256(standard_results_path),
            "host_launch_profile": _sha256(profile_path),
        },
        "scheduler_trace": scheduler_summary,
        "predictor_source": predictor_identity,
        "standard_simulation": standard,
        "measured": measured,
        "trace_reconstructed": trace_metrics,
        "scheduling_trajectory_replay_modes": modes,
        "formal_latency_gap_attribution": scheduling_trajectory_replay.attribute_latency_gap(
            measured, trace_metrics, selected_replay, standard
        ),
        "canonical_step_service": {
            "observed": observed_service,
            "prediction": prediction_totals,
            "overall_device_underprediction_percent": (
                100.0 * (actual_total_s - predicted_total_s) / actual_total_s
            ),
            "by_phase": phase_service_error,
        },
        "predicted_kernel_contributions_by_phase": contributions,
        "interpretation": (
            "The request-level attribution is exact algebra for the selected "
            "counterfactual, not a fitted correction. Observed model/runtime "
            "service includes all serialized work between execution completions "
            "and therefore is not labeled pure Kernel time."
        ),
    }


def main(
    load_compact_case: Callable[[Path], object],
    repository_root: Path = REPOSITORY_ROOT,
    standard_results_path: Path | None = None,
) -> None:
    root = repository_root.resolve()
    case = load_compact_case(root / CASE_RELATIVE_PATH)
    if standard_results_path is None:
        standard_results_path = root / (
            "experiment_logs/serving/sharegpt_phase1/"
            "device-gap-910b-phase1-full-20260831/results.json"
        )
    else:
        standard_results_path = standard_results_path.resolve()
    result = run_counterfactual(root, case, standard_results_path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
