#!/usr/bin/env python3
"""Measure how one cell's predicted TTFT responds to a controlled admission shift.

The per-request Host request path delays only the copy of the arrival stream
that the Scheduler sees, so a shift of that service time is the smallest
well-defined perturbation the queueing model accepts: latency is still measured
against the unshifted client arrivals, and nothing else about the deployment
changes.  A purely additive admission cost would move every request's predicted
TTFT by exactly the shift, giving a response ratio of one.  Any departure from
one is produced downstream of admission, by the Scheduler choosing different
batches once a request crosses a step boundary.

The sweep therefore reports, per cell and per shift, the response ratio

    (predicted TTFT mean at the shift - predicted TTFT mean unshifted) / shift

which is the same quantity the attribution report calls the amplification
factor, generalised from the mechanism's own magnitude to an arbitrary probe.
The shift is a diagnostic probe rather than a model parameter: it is written
into a scratch copy of the published profile and never into a published one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from evaluation.serving.metrics import sharegpt_compact_evaluation as compact
from evaluation.serving.simulation import simulation_runner


PROFILE_DIR = Path("configs/host_timing/host_request_path_profiles")

# The two disjoint regions of the request path, named by the service time that
# carries the whole of each.  ``api_entry`` is the first term of the admission
# sum, so shifting it shifts admission by the same amount for every request
# regardless of prompt length.  ``output_delivery_per_output`` is charged once
# per output of the producing step, so shifting it probes the delivery side,
# which reaches the step loop rather than the arrival stream.
SHIFTABLE_FIELDS = ("api_entry", "tokenizer_encode_per_token",
                    "output_delivery_per_output", "first_token_delivery")

# A shift of a per-request constant such as ``api_entry`` delays every arrival
# by the same amount, which is a pure translation of the whole arrival stream:
# the Scheduler sees an identical relative pattern and can only reproduce the
# same batches, so predicted TTFT must move by exactly the shift.  The only
# part of admission that changes relative arrival spacing is the per-token
# encode, because it charges long prompts more than short ones.  Probing it
# needs a prompt length to express the shift against, and 1000 tokens is the
# order of the ShareGPT prompts these cases carry, so a shift of one ms means
# one ms of extra admission for a 1000-token prompt and proportionally less
# for a shorter one.
REFERENCE_PROMPT_TOKENS = 1000


@dataclass(frozen=True)
class SweepPoint:
    """One cell evaluated under one admission shift."""

    cell_id: str
    shift_ms: float
    shift_field: str
    predicted_ttft_mean_ms: float
    observed_ttft_mean_ms: float
    paired_mape_percent: float
    signed_error_percent: float
    predicted_wait_mean_ms: float
    predicted_processing_mean_ms: float


def write_shifted_profile(
    repository_root: Path,
    hardware_id: str,
    shift_ms: float,
    destination: Path,
    shift_field: str = "api_entry",
) -> Path:
    """Copy one published request path profile with its API entry time shifted.

    ``api_entry`` is the first term of the admission sum, so shifting it shifts
    admission by exactly ``shift_ms`` for every request regardless of prompt
    length.  The measurement block, including the microbench artifact digests
    the loader verifies, is copied unchanged: the scratch profile stays a
    faithful copy of a measured profile plus one declared probe offset.
    """

    if shift_field not in SHIFTABLE_FIELDS:
        raise ValueError(f"unshiftable request path field: {shift_field!r}")
    source = repository_root / PROFILE_DIR / f"{hardware_id}.json"
    payload = json.loads(source.read_text())
    shift_s = shift_ms / 1000.0
    if shift_field == "tokenizer_encode_per_token":
        shift_s /= REFERENCE_PROMPT_TOKENS
    shifted = payload["service_time_s"][shift_field] + shift_s
    if shifted <= 0.0:
        raise ValueError(
            f"shift {shift_ms} ms drives {shift_field} non-positive")
    payload["service_time_s"][shift_field] = shifted
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def evaluate_shift(
    repository_root: Path,
    config: compact.Phase1Config,
    cell: compact.Phase1Cell,
    source_commit: str,
    shift_ms: float,
    scratch_dir: Path,
    shift_field: str = "api_entry",
    *,
    load_validated_case: Callable[[Path], compact.ValidatedCompactCase],
) -> SweepPoint:
    """Run one cell once under one admission shift and read its TTFT back."""

    hardware_id = cell.hardware.hardware_id
    overrides = None
    if shift_ms != 0.0:
        overrides = {
            hardware_id: str(write_shifted_profile(
                repository_root,
                hardware_id,
                shift_ms,
                scratch_dir / (
                    f"{hardware_id}_{shift_field}_{shift_ms:+.3f}ms.json"),
                shift_field,
            )),
        }
    launch_paths = compact.resolve_host_launch_profile_paths(
        repository_root, (), {hardware_id})
    scheduler_paths = compact.resolve_scheduler_host_profile_paths(
        repository_root, (), {hardware_id})
    # The alignment stage validates the captured trace against the case and is
    # independent of the admission shift, so the sweep skips it and compares
    # prediction against prediction.
    result = compact.run_isolated_case(
        str(repository_root),
        config,
        cell,
        source_commit,
        False,
        "accelerated",
        load_validated_case=load_validated_case,
        run_simulation=simulation_runner.run_cli_args,
        host_launch_profile_paths=launch_paths,
        scheduler_host_profile_paths=scheduler_paths,
        host_request_path_profile_paths=overrides,
    )
    metrics = result["metrics"]
    ttft = metrics["latency"]["ttft_ms"]["mean"]
    predicted = ttft["predicted"]
    observed = ttft["observed"]
    # The wait/processing split is the decomposition the attribution report
    # uses: an additive admission cost lands entirely in wait, so processing
    # moving at all is the Scheduler having chosen different batches.
    records = metrics["ttft_breakdown"]["records"]
    record_count = len(records)
    wait_mean = sum(
        record["predicted_wait_ms"] for record in records) / record_count
    processing_mean = sum(
        record["predicted_processing_ms"] for record in records) / record_count
    return SweepPoint(
        cell_id=cell.cell_id,
        shift_ms=shift_ms,
        shift_field=shift_field,
        predicted_ttft_mean_ms=predicted,
        observed_ttft_mean_ms=observed,
        paired_mape_percent=(
            metrics["paired_request_error"]["ttft_ms"]["mape_percent"]),
        signed_error_percent=100.0 * (predicted - observed) / observed,
        predicted_wait_mean_ms=wait_mean,
        predicted_processing_mean_ms=processing_mean,
    )


def response_ratios(
    points: Sequence[SweepPoint],
) -> list[Mapping[str, object]]:
    """Divide each shifted point's TTFT change by the shift that produced it."""

    baselines = {
        point.cell_id: point for point in points if point.shift_ms == 0.0
    }
    rows = []
    for point in points:
        if point.shift_ms == 0.0:
            continue
        if point.cell_id not in baselines:
            raise ValueError(f"no unshifted baseline for {point.cell_id}")
        base = baselines[point.cell_id]
        delta = point.predicted_ttft_mean_ms - base.predicted_ttft_mean_ms
        rows.append({
            "cell_id": point.cell_id,
            "shift_ms": point.shift_ms,
            "shift_field": point.shift_field,
            "predicted_wait_delta_ms": (
                point.predicted_wait_mean_ms - base.predicted_wait_mean_ms),
            "predicted_processing_delta_ms": (
                point.predicted_processing_mean_ms
                - base.predicted_processing_mean_ms),
            "baseline_predicted_ttft_ms": base.predicted_ttft_mean_ms,
            "shifted_predicted_ttft_ms": point.predicted_ttft_mean_ms,
            "predicted_ttft_delta_ms": delta,
            "response_ratio": delta / point.shift_ms,
            "baseline_signed_error_percent": base.signed_error_percent,
            "shifted_signed_error_percent": point.signed_error_percent,
        })
    return rows
