"""Invariant checks for a finished standalone AF run-output artifact.

Evaluation-owner home for the af-disagg-1013 Gate A2 run-artifact contract:
the resident test suite must not depend on an ignored experiment snapshot,
so the full-artifact checks live here and are invoked explicitly (see
``scripts/evaluation/serving/check_af_run_artifact.py``), e.g.::

    python scripts/evaluation/serving/check_af_run_artifact.py \
        --run-output experiment_logs/evaluation/diagnostics/af-disagg-1013/runs/af_tp1tp1.json

The artifact under test is a standalone AF (Attention/FFN disaggregated)
simulation output produced by ``scripts/run_serving_simulation.py`` with
``--deployment af_disagg``.  All checks are fail-closed: any missing field,
missing file, or drifted invariant raises.

Scope note: the artifact exposes activation-transfer *counts* (one per
emitted AF_ACTIVATION_TRANSFER workload node), not per-transfer byte
records.  This module asserts the count-level conservation equation: every scheduler step
runs one full forward over all decoder layers, and each layer crosses the
AF pool boundary twice (Attention -> FFN, FFN -> Attention) with two
tensors (hidden_states, residual) per crossing per tensor-parallel lane,
so the expected count is ``4 * tp_size * num_hidden_layers * num_steps``
with ``tp_size`` derived from the artifact's own pool topology.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

from src.serving.workload.model_config import load_model_config


AF_STEP_PHASES = ("prefill", "decode", "mixed")


def _require_mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"AF run artifact {label} must be a mapping")
    return value


def _check_deployment_and_model_binding(
        artifact: Mapping, *, model_path: str | Path) -> None:
    if artifact["deployment"] != "af_disagg":
        raise ValueError("AF run artifact deployment is not af_disagg")
    deployment_plan = _require_mapping(
        artifact["deployment_plan"], "deployment_plan")
    if deployment_plan["mode"] != "af_disagg":
        raise ValueError("AF run artifact deployment plan mode drifted")
    roles = {
        _require_mapping(pool, "worker pool")["role"]
        for pool in deployment_plan["worker_pools"]
    }
    if roles != {"attention", "ffn"}:
        raise ValueError(
            f"AF run artifact pool roles drifted: {sorted(roles)!r}")
    routes = {
        (
            _require_mapping(route, "route")["source_pool"],
            route["destination_pool"],
            route["payload"],
        )
        for route in deployment_plan["routes"]
    }
    if routes != {
            ("attention", "ffn", "activation"),
            ("ffn", "attention", "activation"),
    }:
        raise ValueError(
            f"AF run artifact activation routes drifted: {sorted(routes)!r}")
    model = load_model_config(model_path)
    provenance = _require_mapping(
        artifact["model_config_provenance"], "model_config_provenance")
    if provenance["config_sha256"] != model.config_sha256:
        raise ValueError(
            "AF run artifact model config SHA256 disagrees with "
            f"{model_path}")


def _check_step_ordering_and_phases(artifact: Mapping) -> None:
    progress = _require_mapping(
        artifact["scheduler_progress"], "scheduler_progress")
    series = progress["series"]
    summary = _require_mapping(progress["summary"], "scheduler summary")
    counts = _require_mapping(
        artifact["execution_artifact"], "execution_artifact")["counts"]
    if not (summary["step_count"] == len(series) == artifact["num_steps"]):
        raise ValueError("AF run artifact step counts disagree")
    if counts["schedule_steps"] != len(series):
        raise ValueError(
            "AF run artifact execution schedule step count disagrees")
    for index, entry in enumerate(series):
        if entry["step_index"] != index:
            raise ValueError("AF run artifact steps are not ordered")
        if entry["end_time_s"] < entry["start_time_s"]:
            raise ValueError("AF run artifact step end precedes start")
        if index and entry["start_time_s"] != series[index - 1]["end_time_s"]:
            # The scheduler timeline is contiguous: each step starts exactly
            # where its predecessor ended.
            raise ValueError("AF run artifact scheduler timeline is not "
                             "contiguous")
        if entry["scheduled_tokens"] != (
                entry["prefill_tokens"] + entry["decode_tokens"]):
            raise ValueError("AF run artifact step token counts do not add "
                             "up")
    host_steps = _require_mapping(
        artifact["host_timing"], "host_timing")["steps"]
    if len(host_steps) != len(series):
        raise ValueError("AF run artifact Host timing steps disagree with "
                         "the schedule")
    observed_phases = set()
    for host_step, entry in zip(host_steps, series):
        if host_step["step_id"] != entry["step_id"]:
            raise ValueError("AF run artifact Host timing step IDs disagree")
        if host_step["phase"] not in AF_STEP_PHASES:
            raise ValueError(
                f"AF run artifact unknown step phase: {host_step['phase']!r}")
        observed_phases.add(host_step["phase"])
        # Standalone AF keeps a single Attention scheduler pool; the FFN
        # pool executes only activation-fed work, never its own scheduling.
        if host_step["pool_name"] != "attention":
            raise ValueError("AF run artifact scheduled a step outside the "
                             "Attention pool")
    if observed_phases != set(AF_STEP_PHASES):
        raise ValueError(
            "AF run artifact did not exercise every step phase: "
            f"{sorted(observed_phases)!r}")


def _check_completion_and_token_conservation(
        artifact: Mapping, *, requests_path: str | Path) -> None:
    with Path(requests_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    request_count = len(rows)
    input_tokens = sum(int(row["In Token [tokens]"]) for row in rows)
    output_tokens = sum(int(row["Out Token [tokens]"]) for row in rows)
    progress = _require_mapping(
        artifact["scheduler_progress"], "scheduler_progress")
    series = progress["series"]
    summary = _require_mapping(progress["summary"], "scheduler summary")
    if artifact["request_source"]["row_count"] != request_count:
        raise ValueError("AF run artifact request source row count drifted")
    if summary["request_count"] != request_count:
        raise ValueError("AF run artifact request count drifted")
    if summary["completed_requests"] != request_count:
        raise ValueError("AF run artifact did not complete every request")
    if series[-1]["cumulative_completed_requests"] != request_count:
        raise ValueError("AF run artifact final cumulative completion "
                         "drifted")
    cumulative = [entry["cumulative_completed_requests"] for entry in series]
    if any(later < earlier
           for earlier, later in zip(cumulative, cumulative[1:])):
        raise ValueError("AF run artifact cumulative completion regressed")
    # Decode tokens exclude the first (prefill-emitted) token per request.
    if summary["prefill_tokens"] != input_tokens:
        raise ValueError("AF run artifact prefill tokens disagree with the "
                         "request source")
    if summary["decode_tokens"] != output_tokens - request_count:
        raise ValueError("AF run artifact decode tokens disagree with the "
                         "request source")
    if summary["scheduled_tokens"] != (
            summary["prefill_tokens"] + summary["decode_tokens"]):
        raise ValueError("AF run artifact scheduled tokens do not add up")
    if summary["scheduled_tokens"] != sum(
            entry["scheduled_tokens"] for entry in series):
        raise ValueError("AF run artifact scheduled tokens disagree with "
                         "the step series")
    metrics = _require_mapping(artifact["metrics"], "metrics")
    if not (metrics["token_throughput_per_s"] > 0
            and metrics["request_throughput_per_s"] > 0):
        raise ValueError("AF run artifact throughputs must be positive")


def _af_pair_tensor_parallel_size(artifact: Mapping) -> int:
    """Derive the per-pool TP size from the artifact's verified topology.

    Fail-closed: exactly one Attention and one FFN pool, rank-local
    parallelism (PP=CP=DP=SP=1), tensor_parallel_size equal to the pool's
    rank count, and equal TP sizes on both pools — anything else is not a
    topology this checker can reason about.
    """

    deployment_plan = _require_mapping(
        artifact["deployment_plan"], "deployment_plan")
    pools = {}
    for pool in deployment_plan["worker_pools"]:
        pool = _require_mapping(pool, "worker pool")
        role = pool["role"]
        if role in pools:
            raise ValueError(
                f"AF run artifact has more than one {role!r} pool")
        pools[role] = pool
    if set(pools) != {"attention", "ffn"}:
        raise ValueError(
            f"AF run artifact pool roles drifted: {sorted(pools)!r}")
    tp_sizes = set()
    occupied_ranks = set()
    for role, pool in pools.items():
        parallelism = _require_mapping(pool["parallelism"], "parallelism")
        for axis in (
                "pipeline_parallel_size", "context_parallel_size",
                "data_parallel_size", "sequence_parallel_size"):
            if parallelism[axis] != 1:
                raise ValueError(
                    "AF transfer-count conservation requires rank-local "
                    f"parallelism; {role} pool has {axis}="
                    f"{parallelism[axis]}")
        tp_size = parallelism["tensor_parallel_size"]
        if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 1:
            raise ValueError("AF tensor_parallel_size must be a positive integer")
        ranks = pool["ranks"]
        if not isinstance(ranks, (list, tuple)) or any(
                isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
                for rank in ranks):
            raise ValueError("AF pool ranks must be nonnegative integers")
        if len(set(ranks)) != len(ranks) or occupied_ranks.intersection(ranks):
            raise ValueError("AF pool ranks must be unique and disjoint")
        occupied_ranks.update(ranks)
        if tp_size != len(ranks):
            raise ValueError(
                f"AF run artifact {role} pool tensor_parallel_size disagrees "
                "with its rank list")
        tp_sizes.add(tp_size)
    if len(tp_sizes) != 1:
        raise ValueError(
            "AF transfer-count conservation requires equal Attention/FFN "
            f"tensor-parallel sizes; found {sorted(tp_sizes)!r}")
    return tp_sizes.pop()


def _check_transfer_count_conservation(
        artifact: Mapping, *, model_path: str | Path) -> None:
    model = load_model_config(model_path)
    counts = _require_mapping(
        artifact["execution_artifact"], "execution_artifact")["counts"]
    if counts["simulation_transfers"] != 0:
        raise ValueError("standalone AF must not emit SimulationTransfer "
                         "records")
    if artifact["num_transfers"] != counts["trace_transfers"]:
        raise ValueError("AF run artifact num_transfers disagrees with the "
                         "execution artifact")
    # Per scheduler step and decoder layer the forward crosses the AF pool
    # boundary twice (Attention -> FFN, FFN -> Attention); each crossing
    # emits two tensors (hidden_states, residual) per tensor-parallel lane
    # (traced_models.trace_af_forward activation ports). The lane count
    # comes from the artifact's own verified
    # pool topology, so TP1 and TP2 runs are both covered without
    # case-specific branches.
    tp_size = _af_pair_tensor_parallel_size(artifact)
    expected_transfers = (
        4 * tp_size * model.num_hidden_layers * artifact["num_steps"])
    if counts["trace_transfers"] != expected_transfers:
        raise ValueError(
            "AF activation transfer count conservation failed: "
            f"trace_transfers={counts['trace_transfers']} != "
            f"4 * {tp_size} * {model.num_hidden_layers} * "
            f"{artifact['num_steps']}")


def validate_af_run_artifact(
        artifact: Mapping,
        *,
        model_path: str | Path,
        requests_path: str | Path) -> dict[str, object]:
    """Validate one standalone AF run-output artifact; raise on any drift.

    Returns a small summary mapping on success.
    """

    _require_mapping(artifact, "root")
    _check_deployment_and_model_binding(artifact, model_path=model_path)
    _check_step_ordering_and_phases(artifact)
    _check_completion_and_token_conservation(
        artifact, requests_path=requests_path)
    _check_transfer_count_conservation(artifact, model_path=model_path)
    return {
        "validation_status": "validated",
        "deployment": artifact["deployment"],
        "num_steps": artifact["num_steps"],
        "num_transfers": artifact["num_transfers"],
        "completed_requests": artifact["scheduler_progress"]["summary"][
            "completed_requests"],
    }


def validate_af_run_artifact_file(
        run_output_path: str | Path,
        *,
        model_path: str | Path,
        requests_path: str | Path) -> dict[str, object]:
    """Load a run-output JSON and validate it; fail when it is missing."""

    path = Path(run_output_path)
    if not path.is_file():
        raise RuntimeError(
            "AF run artifact is missing; regenerate it with "
            "scripts/run_serving_simulation.py --deployment af_disagg: "
            f"{path}")
    return validate_af_run_artifact(
        json.loads(path.read_text()),
        model_path=model_path,
        requests_path=requests_path,
    )
