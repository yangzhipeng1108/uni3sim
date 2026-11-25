"""Comparable ShareGPT evaluation for colocated and PD deployments."""

from __future__ import annotations

from evaluation.json_shape import file_sha256

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics
import time
from typing import Callable, Mapping, Sequence

from evaluation.serving.experiments.experiment_plan import git_code_identity
from evaluation.serving.metrics import sharegpt_compact_evaluation as compact
from evaluation.serving.experiments.experiment_manifest import (
    resolve_runtime_memory_policy_path,
)
from src.kernel.predictors.protobridge.adapter import (
    development_profile_config_sha256,
    resolve_registered_profile,
)
from src.serving.kv_cache.kv_capacity import load_runtime_memory_policy
from src.serving.request_gen.canonical_requests import load_canonical_requests
from src.serving.workload.model_config import load_model_config


TOPOLOGIES = {
    "1p1d": (1, 1),
    "1p2d": (1, 2),
    "2p1d": (2, 1),
    "2p2d": (2, 2),
}
_RESULT_COLUMNS = (
    "hardware_id", "model_id", "case_id", "topology", "deployment",
    "status", "accelerator_count", "request_throughput_per_s",
    "output_token_throughput_per_s", "token_throughput_per_s",
    "ttft_p50_s", "ttft_p95_s", "ttft_p99_s", "tpot_p50_s",
    "tpot_p95_s", "tpot_p99_s", "e2e_p50_s", "e2e_p95_s",
    "e2e_p99_s", "prefill_queue_mean_s", "decode_queue_mean_s",
    "routing_wait_mean_s", "kv_transfer_mean_s", "kv_transfer_total_s",
    "prefill_utilization", "decode_utilization", "colocated_utilization",
    "critical_path_request_id", "critical_path_duration_s",
    "critical_path_dominant_category", "schedule_end_time_s",
    "simulator_wall_time_s",
    "relative_request_throughput_percent", "relative_output_tokens_percent",
    "relative_ttft_p50_percent", "relative_tpot_p50_percent",
    "relative_e2e_p50_percent", "blocker",
)
_SUMMARY_COLUMNS = (
    "request_throughput_per_s", "output_token_throughput_per_s",
    "token_throughput_per_s", "ttft_p50_s", "ttft_p95_s", "ttft_p99_s",
    "tpot_p50_s", "tpot_p95_s", "tpot_p99_s", "e2e_p50_s",
    "e2e_p95_s", "e2e_p99_s", "prefill_queue_mean_s",
    "decode_queue_mean_s", "routing_wait_mean_s", "kv_transfer_mean_s",
    "kv_transfer_total_s", "prefill_utilization", "decode_utilization",
    "colocated_utilization", "schedule_end_time_s",
    "simulator_wall_time_s",
)


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _replace_option(argv: list[str], option: str, value: str) -> None:
    index = argv.index(option)
    argv[index + 1] = value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def derive_runtime_memory_policy(
    source_path: str | Path,
    output_path: str | Path,
    *,
    deployment: str,
    pool_roles: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Rebind one validated per-rank policy to explicit identical pools."""

    source = Path(source_path).resolve()
    validated = load_runtime_memory_policy(source)
    if tuple(validated.policies) != ("main",):
        raise ValueError(
            "PD study source runtime policy must contain exactly pool 'main'")
    raw = json.loads(source.read_text())
    template = raw["scheduler_pools"][0]
    if template["pool_name"] != "main":
        raise ValueError("validated source runtime policy lost pool 'main'")
    entries = []
    for pool_name, role in pool_roles:
        entry = deepcopy(template)
        entry["pool_name"] = pool_name
        entry["binding"]["deployment"] = deployment
        entry["binding"]["worker_role"] = role
        entries.append(entry)
    payload = {
        "schema_version": 2,
        "scheduler_pools": entries,
    }
    destination = Path(output_path)
    _write_json(destination, payload)
    load_runtime_memory_policy(destination)
    return {
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "derived_path": str(destination.resolve()),
        "derived_sha256": _sha256(destination),
        "derivation": (
            "exact per-rank runtime policy replicated across independent "
            "scheduler pools; deployment, worker role, and pool name rebound"
        ),
    }


def _build_variant_argv(
    repository_root: Path,
    config: compact.Phase1Config,
    cell: compact.Phase1Cell,
    case: compact.ValidatedCompactCase,
    *,
    topology: str,
    deployment: str,
    policy_path: Path,
    scheduler_pipeline_depth: int,
    scheduler_async_scheduling: bool,
) -> list[str]:
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown PD topology: {topology!r}")
    prefill_instances, decode_instances = TOPOLOGIES[topology]
    argv = compact.build_simulation_argv(
        repository_root,
        config,
        cell,
        case,
        execution_mode="accelerated",
        scheduler_pipeline_depth=scheduler_pipeline_depth,
        scheduler_async_scheduling=scheduler_async_scheduling,
    )
    _replace_option(argv, "--runtime-memory-policy", str(policy_path))
    argv.extend(("--step-retention", "full"))
    hardware_path = str(
        (repository_root / cell.hardware.hardware_config_path).resolve())
    if deployment == "pd_disagg":
        # The compact study models intra-node PD colocation, so its KV
        # transfer follows the tensor-collective scale-up fabric.
        scale_up_link = resolve_registered_profile(
            development_profile_config_sha256(
                cell.hardware.hardware_id, "all_reduce")).link
        p2p_profile = resolve_registered_profile(
            development_profile_config_sha256(
                cell.hardware.hardware_id, "send_recv", link=scale_up_link))
        _replace_option(argv, "--deployment", "pd_disagg")
        argv.extend((
            "--decode-hardware-config", hardware_path,
            "--decode-tp-size", str(config.prediction.values[
                "tensor_parallel_size"]),
            "--num-prefill-instances", str(prefill_instances),
            "--num-decode-instances", str(decode_instances),
            "--transfer-algorithm", "send_recv",
            "--transfer-link", p2p_profile.link,
        ))
    elif deployment == "colocated":
        replica_count = prefill_instances + decode_instances
        for _index in range(replica_count - 1):
            argv.extend((
                "--additional-colocated-hardware-config", hardware_path,
            ))
    else:
        raise ValueError(f"unknown study deployment: {deployment!r}")
    return argv


def _pd_case_path(
    repository_root: Path,
    cell: compact.Phase1Cell,
) -> Path:
    return (
        repository_root
        / "serving_dataset"
        / "sharegpt_pd_disaggregated"
        / cell.hardware.hardware_id
        / cell.model.dataset_model_id
        / cell.case_id
    )


def _scheduler_pipeline_depth_from_pd_case(
    case: object,
    cell: compact.Phase1Cell,
) -> int:
    manifest = compact._mapping(case.manifest, "PD compact manifest")
    if manifest["schema_version"] != 6:
        raise ValueError("PD study requires a schema-v6 PD compact case")
    identity = compact._mapping(manifest["identity"], "PD compact identity")
    expected_identity = {
        "hardware_id": cell.hardware.hardware_id,
        "model_id": cell.model.dataset_model_id,
        "case_id": cell.case_id,
        "source_kind": "sharegpt_pd_disaggregated",
    }
    for field, expected in expected_identity.items():
        if identity[field] != expected:
            raise ValueError(
                f"PD compact {field} mismatch: "
                f"{identity[field]!r} != {expected!r}")
    requests = compact._mapping(manifest["requests"], "PD compact requests")
    if requests["sha256"] != cell.requests_sha256:
        raise ValueError("PD and colocated request streams do not match")
    runtime = compact._mapping(manifest["runtime"], "PD compact runtime")
    instances = compact._mapping(runtime["instances"], "PD runtime instances")
    depths = []
    for role in ("prefill", "decode"):
        instance = compact._mapping(
            instances[role], f"PD runtime {role} instance")
        depth = instance["scheduler_pipeline_depth"]
        if (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1):
            raise ValueError(
                f"PD {role} scheduler_pipeline_depth must be a positive "
                "non-bool integer")
        depths.append(depth)
    if depths[0] != depths[1]:
        raise NotImplementedError(
            "PD study requires equal Prefill and Decode Scheduler depths "
            "until the simulator accepts per-pool depth")
    return depths[0]


def _scheduler_async_scheduling_from_pd_case(
    case: object,
    pipeline_parallel_size: int,
) -> bool:
    """Resolve the common native async mode from both PD schedulers."""

    manifest = compact._mapping(case.manifest, "PD compact manifest")
    if manifest["schema_version"] != 6:
        raise ValueError("PD study requires a schema-v6 PD compact case")
    runtime = compact._mapping(manifest["runtime"], "PD compact runtime")
    instances = compact._mapping(runtime["instances"], "PD runtime instances")
    modes = []
    for role in ("prefill", "decode"):
        instance = compact._mapping(
            instances[role], f"PD runtime {role} instance")
        observed_pipeline_parallel_size = instance["pipeline_parallel_size"]
        if observed_pipeline_parallel_size != pipeline_parallel_size:
            raise ValueError(
                f"PD {role} pipeline_parallel_size disagrees with config")
        if observed_pipeline_parallel_size != 1:
            raise NotImplementedError(
                "PD compact runtime must explicitly record async scheduling "
                "when pipeline_parallel_size exceeds one")
        depth = instance["scheduler_pipeline_depth"]
        if depth not in (1, 2):
            raise ValueError(
                "observed PP=1 PD executor capacity must be one or two")
        modes.append(depth == 2)
    if modes[0] is not modes[1]:
        raise NotImplementedError(
            "PD study requires equal Prefill and Decode async scheduling modes")
    return modes[0]


def _pool_roles(deployment: str, topology: str) -> tuple[tuple[str, str], ...]:
    prefill_instances, decode_instances = TOPOLOGIES[topology]
    if deployment == "pd_disagg":
        return tuple(
            (f"prefill-{index}", "prefill")
            for index in range(prefill_instances)
        ) + tuple(
            (f"decode-{index}", "decode")
            for index in range(decode_instances)
        )
    if deployment == "colocated":
        return tuple(
            (f"colocated-{index}", "colocated")
            for index in range(prefill_instances + decode_instances)
        )
    raise ValueError(f"unknown study deployment: {deployment!r}")


def _request_specs(case_dir: Path):
    manifest = json.loads((case_dir / "manifest.json").read_text())
    requests_path = case_dir / manifest["requests"]["path"]
    if _sha256(requests_path) != manifest["requests"]["sha256"]:
        raise ValueError("PD study requests SHA256 disagrees with manifest")
    return load_canonical_requests(requests_path)


def _is_discarded_async_terminal_artifact_request(
    scheduled: Mapping[str, object],
    generated_tokens: Mapping[str, object],
    finished_request_ids: Sequence[str],
) -> bool:
    request_id = scheduled["request_id"]
    return (
        scheduled["phase"] == "decode"
        and scheduled["input_tokens"] == 0
        and scheduled["output_tokens"] == 1
        and scheduled["recomputed_tokens"] == 0
        and scheduled["prefix_replay_tokens"] == 0
        and scheduled["resumed_from_preemption"] is False
        and scheduled["raw_model_output_tokens"] == 1
        and scheduled["mtp_proposer_mode"] == "disabled"
        and scheduled["mtp_acceptance_source_kind"] == "disabled"
        and scheduled["mtp_acceptance_source_id"] == "disabled"
        and scheduled["mtp_proposed_tokens"] == 0
        and scheduled["mtp_verified_tokens"] == 0
        and scheduled["mtp_accepted_tokens"] == 0
        and scheduled["mtp_proposer_tokens"] == 0
        and scheduled["mtp_next_proposed_tokens"] == 0
        and generated_tokens[request_id] == 0
        and request_id not in finished_request_ids
    )


def validate_pd_conservation(
    payload: Mapping[str, object],
    requests,
    model,
) -> dict[str, object]:
    """Prove request, token, KV, transfer, and routing invariants."""

    if payload["deployment"] != "pd_disagg":
        raise ValueError("PD conservation requires a pd_disagg result")
    pools = payload["deployment_plan"]["worker_pools"]
    pool_by_name = {pool["name"]: pool for pool in pools}
    role_by_pool = {pool["name"]: pool["role"] for pool in pools}
    tensor_parallel_sizes = tuple(
        int(pool["parallelism"]["tensor_parallel_size"])
        for pool in pools)
    if any(tp_size not in (1, 2) for tp_size in tensor_parallel_sizes):
        raise ValueError("PD conservation supports independent P/D TP=1 or 2")
    prefill_pools = {
        name for name, role in role_by_pool.items() if role == "prefill"
    }
    decode_pools = {
        name for name, role in role_by_pool.items() if role == "decode"
    }
    if not prefill_pools or not decode_pools:
        raise ValueError("PD conservation requires P and D pools")
    request_by_id = {request.request_id: request for request in requests}
    emitted = {
        request_id: {"prefill": 0, "decode": 0}
        for request_id in request_by_id
    }
    first_decode = {}
    artifact = payload["execution_artifact"]
    for step in artifact["steps"]:
        pool_name = step["pool_name"]
        if pool_name in prefill_pools:
            role = "prefill"
        elif pool_name in decode_pools:
            role = "decode"
        else:
            raise ValueError(
                f"PD execution step uses an unknown pool: {pool_name!r}")
        backend_step = step["backend_step"]
        generated = backend_step["generated_tokens"]
        finished_request_ids = backend_step["schedule"][
            "finished_request_ids"]
        for scheduled in backend_step["schedule"]["requests"]:
            request_id = scheduled["request_id"]
            if request_id not in request_by_id:
                raise ValueError(
                    f"PD execution scheduled unknown request {request_id!r}")
            if scheduled["phase"] != role:
                if (role == "prefill"
                        and _is_discarded_async_terminal_artifact_request(
                            scheduled, generated, finished_request_ids)):
                    continue
                raise ValueError(
                    f"PD {role} pool executed {scheduled['phase']} work for "
                    f"{request_id!r}")
            if role == "decode":
                if scheduled["input_tokens"] != 0:
                    raise ValueError(
                        f"PD decode recomputed prefill for {request_id!r}")
                if request_id not in first_decode:
                    first_decode[request_id] = {
                        "start_time_s": step["schedule_interval"][
                            "start_time_s"],
                        "computed_tokens": scheduled["computed_tokens"],
                        "output_tokens": scheduled["output_tokens"],
                    }
            emitted[request_id][role] += generated[request_id]

    transfers_by_request = {request_id: [] for request_id in request_by_id}
    for transfer in artifact["schedule_trace"]["transfers"]:
        if transfer["payload"] != "kv":
            raise ValueError("pure PD study produced a non-KV transfer")
        if len(transfer["request_ids"]) != 1:
            raise ValueError("PD KV transfer must carry exactly one request")
        request_id = transfer["request_ids"][0]
        if request_id not in transfers_by_request:
            raise ValueError(
                f"PD transfer references unknown request {request_id!r}")
        transfers_by_request[request_id].append(transfer)

    route_decisions = artifact["schedule_trace"]["route_decisions"]
    ingress_counts = {request_id: 0 for request_id in request_by_id}
    decode_route_counts = {request_id: 0 for request_id in request_by_id}
    assignments_by_stage = {
        "ingress_to_prefill": {
            pool_name: 0 for pool_name in sorted(prefill_pools)},
        "prefill_to_decode": {
            pool_name: 0 for pool_name in sorted(decode_pools)},
    }
    for decision in route_decisions:
        request_id = decision["request_id"]
        if request_id not in request_by_id:
            raise ValueError(
                f"PD route references unknown request {request_id!r}")
        candidates = decision["candidate_queued_tokens"]
        expected_engine = min(candidates, key=lambda item: (item[1], item[0]))[0]
        if (decision["policy"] != "least_queued_tokens"
                or decision["selected_engine_id"] != expected_engine):
            raise ValueError(
                f"PD route violates least_queued_tokens for {request_id!r}")
        if decision["stage"] == "ingress_to_prefill":
            ingress_counts[request_id] += 1
        elif decision["stage"] == "prefill_to_decode":
            decode_route_counts[request_id] += 1
        else:
            raise ValueError(
                f"PD study found unexpected route stage {decision['stage']!r}")
        assignments_by_stage[decision["stage"]][expected_engine] += 1

    transferred_bytes = 0
    handoffs_by_route = {}
    for request_id, request in request_by_id.items():
        counts = emitted[request_id]
        if counts["prefill"] + counts["decode"] != request.output_tokens:
            raise ValueError(
                f"PD output-token conservation failed for {request_id!r}")
        if ingress_counts[request_id] != 1:
            raise ValueError(
                f"PD ingress routing count is not one for {request_id!r}")
        requires_decode = counts["prefill"] < request.output_tokens
        expected_handoff_count = 1 if requires_decode else 0
        transfers = transfers_by_request[request_id]
        if decode_route_counts[request_id] != expected_handoff_count:
            raise ValueError(
                f"PD decode routing count mismatch for {request_id!r}")
        if requires_decode:
            if counts["prefill"] != 1:
                raise ValueError(
                    f"PD prefill did not emit one first token for {request_id!r}")
            admission = first_decode[request_id]
            if (admission["computed_tokens"] != request.input_tokens
                    or admission["output_tokens"] != 1):
                raise ValueError(
                    f"PD decode admission state is discontinuous for "
                    f"{request_id!r}")
            transfer_completion_s = max(
                transfer["end_time_s"] for transfer in transfers)
            if admission["start_time_s"] < transfer_completion_s:
                raise ValueError(
                    f"PD decode started before transfer for {request_id!r}")
            route_pairs = {
                (
                    transfer["source_pool_id"],
                    transfer["destination_pool_id"],
                )
                for transfer in transfers
            }
            if len(route_pairs) != 1:
                raise ValueError(
                    f"PD KV physical edges disagree on route for "
                    f"{request_id!r}")
            source_pool_id, destination_pool_id = next(iter(route_pairs))
            source_tp = int(pool_by_name[source_pool_id]["parallelism"][
                "tensor_parallel_size"])
            destination_tp = int(pool_by_name[destination_pool_id][
                "parallelism"]["tensor_parallel_size"])
            expected_transfer_count = max(source_tp, destination_tp)
            if len(transfers) != expected_transfer_count:
                raise ValueError(
                    f"PD KV physical edge count mismatch for {request_id!r}")
            if model.attention_layout == "standard_gqa":
                if model.num_key_value_heads % expected_transfer_count:
                    raise ValueError(
                        "PD KV heads must divide the physical edge count")
                expected_bytes = (
                    request.input_tokens * 2 * model.num_hidden_layers
                    * (model.num_key_value_heads // expected_transfer_count)
                    * model.head_dim * model.dtype_bytes)
            elif expected_transfer_count == 1:
                expected_bytes = (
                    request.input_tokens
                    * model.kv_bytes_per_token_per_parallel_rank(1, 1))
            else:
                raise NotImplementedError(
                    "heterogeneous PD TP conservation requires standard GQA")
            for transfer in transfers:
                if transfer["byte_count"] != expected_bytes:
                    raise ValueError(
                        f"PD KV transfer bytes mismatch for {request_id!r}: "
                        f"{transfer['byte_count']} != {expected_bytes}")
            transfer_ids = tuple(
                transfer["transfer_id"] for transfer in transfers)
            if len(transfer_ids) != len(set(transfer_ids)):
                raise ValueError(
                    f"PD KV physical lane IDs repeat for {request_id!r}")
            transferred_bytes += sum(
                transfer["byte_count"] for transfer in transfers)
            route = f"{source_pool_id}->{destination_pool_id}"
            if route not in handoffs_by_route:
                handoffs_by_route[route] = 0
            handoffs_by_route[route] += 1
        else:
            if transfers:
                raise ValueError(
                    f"PD-complete request has a KV transfer for {request_id!r}")
            if counts["decode"] != 0 or request_id in first_decode:
                raise ValueError(
                    f"PD-complete request entered decode for {request_id!r}")
    if set(payload["metrics"]["requests"]) != set(request_by_id):
        raise ValueError("PD metrics request IDs disagree with input requests")
    return {
        "status": "passed",
        "request_count": len(request_by_id),
        "handoff_request_count": sum(
            bool(transfers) for transfers in transfers_by_request.values()),
        "transfer_count": sum(
            len(transfers) for transfers in transfers_by_request.values()),
        "transferred_bytes": transferred_bytes,
        "routing": {
            "policy": "least_queued_tokens",
            "immediate_reservation": True,
            "stable_tie_break": "engine_id",
            "assignments_by_stage_and_pool": assignments_by_stage,
        },
        "handoffs_by_pool_route": handoffs_by_route,
        "checks": {
            "prefill_only_on_p": True,
            "decode_only_on_d": True,
            "exactly_one_handoff_when_needed": True,
            "no_empty_handoff": True,
            "decode_after_transfer": True,
            "no_decode_prefill_recompute": True,
            "request_and_output_token_conservation": True,
            "kv_token_continuity": True,
            "least_queued_tokens_and_stable_tie_break": True,
        },
    }


def _quantile(distribution: Mapping[str, object], percentile: int) -> float:
    quantiles = distribution["quantiles"]
    percentiles = list(quantiles["percentiles"])
    return float(quantiles["values_s"][percentiles.index(percentile)])


def summarize_payload(
    payload: Mapping[str, object],
    *,
    simulator_wall_time_s: float,
) -> dict[str, object]:
    metrics = payload["metrics"]
    distributions = metrics["latency_distributions"]
    resources = metrics["system_resources"]
    pools = payload["deployment_plan"]["worker_pools"]
    roles = {pool["name"]: pool["role"] for pool in pools}
    utilizations = resources["worker_pool_utilization"]

    def mean_utilization(role: str) -> float | None:
        values = [
            float(utilizations[name]["utilization"])
            for name in sorted(utilizations)
            if roles[name] == role
        ]
        return None if not values else statistics.fmean(values)

    queue_by_stage = resources["scheduler_queue"]["by_stage"]
    transfers_by_request = {}
    for transfer in payload["execution_artifact"]["schedule_trace"][
            "transfers"]:
        if transfer["payload"] != "kv":
            continue
        request_ids = transfer["request_ids"]
        if len(request_ids) != 1:
            raise ValueError(
                "PD study KV transfer must carry exactly one request")
        request_id = request_ids[0]
        if request_id not in transfers_by_request:
            transfers_by_request[request_id] = []
        transfers_by_request[request_id].append(transfer)
    transfer_durations = [
        float(
            max(transfer["end_time_s"] for transfer in transfers)
            - min(transfer["start_time_s"] for transfer in transfers)
        )
        for transfers in transfers_by_request.values()
    ]
    critical_paths = metrics["critical_path"]["system_completion"][
        "terminal_observation_paths"]
    critical_path = max(
        critical_paths,
        key=lambda item: item["path_duration_s"],
    )
    return {
        "request_throughput_per_s": metrics["request_throughput_per_s"],
        "output_token_throughput_per_s": (
            metrics["output_token_throughput_per_s"]),
        "token_throughput_per_s": metrics["token_throughput_per_s"],
        "ttft_p50_s": _quantile(distributions["ttft_s"], 50),
        "ttft_p95_s": _quantile(distributions["ttft_s"], 95),
        "ttft_p99_s": _quantile(distributions["ttft_s"], 99),
        "tpot_p50_s": _quantile(distributions["tpot_s"], 50),
        "tpot_p95_s": _quantile(distributions["tpot_s"], 95),
        "tpot_p99_s": _quantile(distributions["tpot_s"], 99),
        "e2e_p50_s": _quantile(distributions["latency_s"], 50),
        "e2e_p95_s": _quantile(distributions["latency_s"], 95),
        "e2e_p99_s": _quantile(distributions["latency_s"], 99),
        "prefill_queue_mean_s": (
            queue_by_stage["ingress_to_prefill"]["mean_wait_s"]
            if "ingress_to_prefill" in queue_by_stage else None),
        "decode_queue_mean_s": (
            queue_by_stage["prefill_to_decode"]["mean_wait_s"]
            if "prefill_to_decode" in queue_by_stage else None),
        "routing_wait_mean_s": resources["router_queue"]["overall"][
            "mean_wait_s"],
        "kv_transfer_mean_s": (
            statistics.fmean(transfer_durations)
            if transfer_durations else None),
        "kv_transfer_total_s": sum(transfer_durations),
        "prefill_utilization": mean_utilization("prefill"),
        "decode_utilization": mean_utilization("decode"),
        "colocated_utilization": mean_utilization("colocated"),
        "critical_path": critical_path,
        "schedule_end_time_s": payload["schedule_end_time_s"],
        "simulator_wall_time_s": simulator_wall_time_s,
    }


def compare_summaries(
    baseline: Mapping[str, object],
    pd: Mapping[str, object],
) -> dict[str, float]:
    def throughput_delta(name: str) -> float:
        return 100.0 * (float(pd[name]) / float(baseline[name]) - 1.0)

    def latency_delta(name: str) -> float:
        return 100.0 * (float(baseline[name]) - float(pd[name])) / float(
            baseline[name])

    return {
        "request_throughput_percent": throughput_delta(
            "request_throughput_per_s"),
        "output_token_throughput_percent": throughput_delta(
            "output_token_throughput_per_s"),
        "ttft_p50_percent": latency_delta("ttft_p50_s"),
        "tpot_p50_percent": latency_delta("tpot_p50_s"),
        "e2e_p50_percent": latency_delta("e2e_p50_s"),
    }


def select_study_cells(
    repository_root: str | Path,
    config_arguments: Sequence[str],
    requested_cell_ids: Sequence[str],
) -> tuple[tuple[compact.Phase1Config, compact.Phase1Cell], ...]:
    """Resolve an exact cell selection across existing suite configs."""

    root = Path(repository_root).resolve()
    if not config_arguments:
        raise ValueError("select at least one PD study config")
    if not requested_cell_ids:
        raise ValueError("select at least one PD study case")
    configs = tuple(
        compact.load_phase1_config(compact.resolve_repository_input(
            root, argument, "experiment config"))
        for argument in config_arguments
    )
    selected_cells = []
    matched_ids = set()
    for config in configs:
        available_ids = set(config.cell_order)
        selected_ids = tuple(
            cell_id for cell_id in requested_cell_ids
            if cell_id in available_ids)
        if selected_ids:
            cells = compact.select_phase1_cells(
                config, selected_ids, all_cases=False)
            selected_cells.extend((config, cell) for cell in cells)
            matched_ids.update(selected_ids)
    unknown = [
        cell_id for cell_id in requested_cell_ids
        if cell_id not in matched_ids
    ]
    if unknown:
        raise ValueError(
            f"selected cases are absent from the supplied configs: {unknown}")
    if len(selected_cells) != len(requested_cell_ids):
        raise ValueError("supplied configs select duplicate study cells")
    return tuple(selected_cells)


def run_study_cell(
    repository_root: str | Path,
    config: compact.Phase1Config,
    cell: compact.Phase1Cell,
    case: compact.ValidatedCompactCase,
    pd_case: object,
    topology: str,
    output_dir: str | Path,
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
) -> dict[str, object]:
    root = Path(repository_root).resolve()
    destination = Path(output_dir).resolve()
    prediction = config.prediction.values
    tensor_parallel_size = int(prediction["tensor_parallel_size"])
    case_dir = compact.compact_case_path(root, config, cell)
    requests = _request_specs(case_dir)
    model_path = (root / cell.model.simulation_model_path).resolve()
    model = load_model_config(model_path)
    scheduler_pipeline_depth = _scheduler_pipeline_depth_from_pd_case(
        pd_case, cell)
    scheduler_async_scheduling = _scheduler_async_scheduling_from_pd_case(
        pd_case, int(prediction["pipeline_parallel_size"]))
    source_policy = resolve_runtime_memory_policy_path(
        prediction["runtime_memory_policy_path"],
        repository_root=root,
        hardware_id=cell.hardware.hardware_id,
        model_id=cell.model.dataset_model_id,
    )
    pair_root = (
        destination
        / cell.hardware.hardware_id
        / cell.model.dataset_model_id
        / cell.case_id
        / topology
    )
    policies = {}
    payloads = {}
    wall_times = {}
    for deployment in ("colocated", "pd_disagg"):
        policy_path = pair_root / f"{deployment}_runtime_memory_policy.json"
        policies[deployment] = derive_runtime_memory_policy(
            source_policy,
            policy_path,
            deployment=deployment,
            pool_roles=_pool_roles(deployment, topology),
        )
        argv = _build_variant_argv(
            root,
            config,
            cell,
            case,
            topology=topology,
            deployment=deployment,
            policy_path=policy_path,
            scheduler_pipeline_depth=scheduler_pipeline_depth,
            scheduler_async_scheduling=scheduler_async_scheduling,
        )
        started = time.perf_counter()
        payloads[deployment] = run_simulation(argv)
        wall_times[deployment] = time.perf_counter() - started
    conservation = validate_pd_conservation(
        payloads["pd_disagg"], requests, model)
    summaries = {
        deployment: summarize_payload(
            payloads[deployment],
            simulator_wall_time_s=wall_times[deployment],
        )
        for deployment in ("colocated", "pd_disagg")
    }
    return {
        "hardware_id": cell.hardware.hardware_id,
        "model_id": cell.model.dataset_model_id,
        "case_id": cell.case_id,
        "topology": topology,
        "status": "completed",
        "blocker": None,
        "accelerator_count": (
            sum(TOPOLOGIES[topology]) * tensor_parallel_size),
        "fairness": {
            "same_hardware": True,
            "same_request_stream_sha256": cell.requests_sha256,
            "same_model_config_sha256": cell.model.model_config_sha256,
            "same_scheduler_limits": True,
            "scheduler_pipeline_depth": scheduler_pipeline_depth,
            "same_memory_policy_source_sha256": policies["colocated"][
                "source_sha256"],
            "pd_worker_count": (
                sum(TOPOLOGIES[topology]) * tensor_parallel_size),
            "colocated_replica_count": sum(TOPOLOGIES[topology]),
            "colocated_worker_count": (
                sum(TOPOLOGIES[topology]) * tensor_parallel_size),
        },
        "runtime_memory_policies": policies,
        "conservation": conservation,
        "colocated": summaries["colocated"],
        "pd_disagg": summaries["pd_disagg"],
        "relative_to_colocated": compare_summaries(
            summaries["colocated"], summaries["pd_disagg"]),
    }


def _flatten_rows(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        if case["status"] != "completed":
            row = {name: None for name in _RESULT_COLUMNS}
            row.update({
                "hardware_id": case["hardware_id"],
                "model_id": case["model_id"],
                "case_id": case["case_id"],
                "topology": case["topology"],
                "status": case["status"],
                "accelerator_count": case["accelerator_count"],
                "blocker": case["blocker"],
            })
            rows.append(row)
            continue
        relative = case["relative_to_colocated"]
        for deployment in ("colocated", "pd_disagg"):
            metrics = case[deployment]
            row = {name: None for name in _RESULT_COLUMNS}
            row.update({
                "hardware_id": case["hardware_id"],
                "model_id": case["model_id"],
                "case_id": case["case_id"],
                "topology": case["topology"],
                "deployment": deployment,
                "status": case["status"],
                "accelerator_count": case["accelerator_count"],
                "blocker": None,
            })
            row.update({name: metrics[name] for name in _SUMMARY_COLUMNS})
            critical_path = metrics["critical_path"]
            row.update({
                "critical_path_request_id": critical_path["request_id"],
                "critical_path_duration_s": critical_path["path_duration_s"],
                "critical_path_dominant_category": critical_path[
                    "dominant_category"],
            })
            if deployment == "pd_disagg":
                row.update({
                    "relative_request_throughput_percent": relative[
                        "request_throughput_percent"],
                    "relative_output_tokens_percent": relative[
                        "output_token_throughput_percent"],
                    "relative_ttft_p50_percent": relative[
                        "ttft_p50_percent"],
                    "relative_tpot_p50_percent": relative[
                        "tpot_p50_percent"],
                    "relative_e2e_p50_percent": relative[
                        "e2e_p50_percent"],
                })
            rows.append(row)
    return rows


def write_study_results(
    output_dir: str | Path,
    cases: Sequence[Mapping[str, object]],
    *,
    source_identity: Mapping[str, object],
    wall_time_s: float,
) -> dict[str, str]:
    destination = Path(output_dir).resolve()
    result = {
        "schema_version": 1,
        "study": "sharegpt_qps16_pd_disaggregation",
        "source_identity": dict(source_identity),
        "wall_time_s": wall_time_s,
        "cases": list(cases),
    }
    json_path = destination / "results.json"
    csv_path = destination / "chart_data.csv"
    markdown_path = destination / "report.md"
    _write_json(json_path, result)
    rows = _flatten_rows(cases)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# ShareGPT QPS16 PD disaggregation study",
        "",
        "Positive relative latency values mean PD is faster; positive throughput values mean PD is higher.",
        "",
        "| Hardware | Model | Topology | Status | Req/s PD | Req/s baseline | TTFT p50 PD (s) | TPOT p50 PD (s) | E2E p50 PD (s) | Req/s change | Blocker |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        if case["status"] == "completed":
            pd = case["pd_disagg"]
            baseline = case["colocated"]
            relative = case["relative_to_colocated"]
            values = (
                case["hardware_id"], case["model_id"], case["topology"],
                "completed", f"{pd['request_throughput_per_s']:.6g}",
                f"{baseline['request_throughput_per_s']:.6g}",
                f"{pd['ttft_p50_s']:.6g}", f"{pd['tpot_p50_s']:.6g}",
                f"{pd['e2e_p50_s']:.6g}",
                f"{relative['request_throughput_percent']:+.2f}%", "",
            )
        elif case["status"] == "blocked":
            values = (
                case["hardware_id"], case["model_id"], case["topology"],
                "blocked", "", "", "", "", "", "", case["blocker"],
            )
        else:
            raise ValueError(
                f"unknown PD study case status: {case['status']!r}")
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n")
    return {
        "results": str(json_path),
        "chart_data": str(csv_path),
        "report": str(markdown_path),
    }


def run_study(
    repository_root: str | Path,
    selected_cells: Sequence[
        tuple[compact.Phase1Config, compact.Phase1Cell]
    ],
    topologies: Sequence[str],
    run_id: str,
    load_case: Callable[[Path], compact.ValidatedCompactCase],
    load_pd_case: Callable[[Path], object],
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    *,
    jobs: int,
) -> dict[str, object]:
    root = Path(repository_root).resolve()
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run_id):
        raise ValueError("run_id contains unsupported characters")
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    normalized_topologies = tuple(topologies)
    if (not normalized_topologies
            or any(topology not in TOPOLOGIES
                   for topology in normalized_topologies)):
        raise ValueError("select at least one supported topology")
    if len(normalized_topologies) != len(set(normalized_topologies)):
        raise ValueError("PD study topologies contain duplicates")
    if not selected_cells:
        raise ValueError("select at least one PD study cell")
    cell_ids = tuple(cell.cell_id for _config, cell in selected_cells)
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("PD study cells contain duplicates")
    output_dir = root / "experiment_logs" / "serving" / "pd_disaggregation" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    jobs_to_run = []
    blocked = []
    for config, cell in selected_cells:
        for topology in normalized_topologies:
            case = load_case(compact.compact_case_path(root, config, cell))
            pd_path = _pd_case_path(root, cell)
            if not pd_path.is_dir():
                blocked.append({
                    "hardware_id": cell.hardware.hardware_id,
                    "model_id": cell.model.dataset_model_id,
                    "case_id": cell.case_id,
                    "topology": topology,
                    "status": "blocked",
                    "accelerator_count": (
                        sum(TOPOLOGIES[topology])
                        * int(config.prediction.values[
                            "tensor_parallel_size"])),
                    "blocker": (
                        "ValueError: schema-v6 PD compact case is not "
                        f"published: {pd_path}"),
                })
                continue
            pd_case = load_pd_case(pd_path)
            jobs_to_run.append((config, cell, case, pd_case, topology))

    completed = []
    worker_count = min(jobs, len(jobs_to_run) or 1)
    if worker_count == 1:
        for config, cell, case, pd_case, topology in jobs_to_run:
            try:
                completed.append(run_study_cell(
                    root, config, cell, case, pd_case, topology, output_dir,
                    run_simulation,
                ))
            except (NotImplementedError, RuntimeError, ValueError) as error:
                completed.append({
                    "hardware_id": cell.hardware.hardware_id,
                    "model_id": cell.model.dataset_model_id,
                    "case_id": cell.case_id,
                    "topology": topology,
                    "status": "blocked",
                    "accelerator_count": (
                        sum(TOPOLOGIES[topology])
                        * int(config.prediction.values[
                            "tensor_parallel_size"])),
                    "blocker": f"{type(error).__name__}: {error}",
                })
    elif worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    run_study_cell,
                    root,
                    config,
                    cell,
                    case,
                    pd_case,
                    topology,
                    output_dir,
                    run_simulation,
                ): (config, cell, topology)
                for config, cell, case, pd_case, topology in jobs_to_run
            }
            for future in as_completed(futures):
                config, cell, topology = futures[future]
                try:
                    completed.append(future.result())
                except (NotImplementedError, RuntimeError, ValueError) as error:
                    completed.append({
                        "hardware_id": cell.hardware.hardware_id,
                        "model_id": cell.model.dataset_model_id,
                        "case_id": cell.case_id,
                        "topology": topology,
                        "status": "blocked",
                        "accelerator_count": (
                            sum(TOPOLOGIES[topology])
                            * int(config.prediction.values[
                                "tensor_parallel_size"])),
                        "blocker": f"{type(error).__name__}: {error}",
                    })
    else:
        raise RuntimeError("PD study resolved a non-positive worker count")
    order = {
        (cell.cell_id, topology): index
        for index, (cell, topology) in enumerate(
            (item for _config, cell in selected_cells for item in (
                (cell, topology) for topology in normalized_topologies)))
    }
    cases = sorted(
        blocked + completed,
        key=lambda case: order[(
            f"{case['hardware_id']}/{case['model_id']}/{case['case_id']}",
            case["topology"],
        )],
    )
    artifacts = write_study_results(
        output_dir,
        cases,
        source_identity=git_code_identity(root),
        wall_time_s=time.perf_counter() - started,
    )
    return {
        "schema_version": 1,
        "run_directory": str(output_dir),
        "completed_count": sum(case["status"] == "completed" for case in cases),
        "blocked_count": sum(case["status"] == "blocked" for case in cases),
        "artifacts": artifacts,
    }
