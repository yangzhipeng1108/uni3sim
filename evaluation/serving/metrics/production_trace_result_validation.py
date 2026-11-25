"""Result-content validation for Production Trace sweep points:
latency distributions, SLO goodput, system resources, critical
path, and sweep analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from evaluation.serving.metrics.interactivity_tgs import (
    arrival_rate_sweep_analysis_manifest,
    InteractivityTGSContract,
    sweep_metric_point_from_manifest,
)
from evaluation.serving.metrics.slo_goodput import SLOContract

from evaluation.json_shape import require_object as _mapping

from evaluation.serving.metrics.production_trace_report_shape import (
    _sequence,
    _exact_fields,
    _required_fields,
    _positive_number,
    _nonnegative_number,
    _nonnegative_integer,
    _nondecreasing_nonnegative,
    _canonical_json_bytes,
)


def _validate_latency_distributions(value: object, point_label: str) -> None:
    distributions = _mapping(value, f"{point_label} latency distributions")
    expected_fields = {
        "schema_version",
        "quantile_representation",
        "percentile_method",
        "cdf_representation",
        "cdf_method",
        "ttft_s",
        "tpot_s",
        "latency_s",
    }
    _exact_fields(
        distributions, expected_fields, f"{point_label} latency distributions")
    if distributions["schema_version"] != 2:
        raise ValueError(f"{point_label} latency distribution schema must be 2")
    for metric_name in ("ttft_s", "tpot_s", "latency_s"):
        distribution = _mapping(
            distributions[metric_name], f"{point_label} {metric_name} distribution")
        _exact_fields(
            distribution,
            {
                "availability",
                "eligible_request_count",
                "excluded_request_count",
                "exclusion_reason",
                "quantiles",
                "empirical_cdf",
            },
            f"{point_label} {metric_name} distribution",
        )
        eligible = _nonnegative_integer(
            distribution["eligible_request_count"],
            f"{point_label} {metric_name} eligible count",
        )
        _nonnegative_integer(
            distribution["excluded_request_count"],
            f"{point_label} {metric_name} excluded count",
        )
        quantiles = _mapping(
            distribution["quantiles"], f"{point_label} {metric_name} quantiles")
        _exact_fields(
            quantiles, {"percentiles", "values_s"},
            f"{point_label} {metric_name} quantiles")
        percentiles = tuple(_sequence(
            quantiles["percentiles"], f"{point_label} {metric_name} percentiles"))
        values = tuple(_sequence(
            quantiles["values_s"], f"{point_label} {metric_name} quantile values"))
        cdf = _mapping(
            distribution["empirical_cdf"], f"{point_label} {metric_name} CDF")
        _exact_fields(
            cdf, {"values_s", "cumulative_probabilities"},
            f"{point_label} {metric_name} CDF")
        cdf_values = tuple(_sequence(
            cdf["values_s"], f"{point_label} {metric_name} CDF values"))
        probabilities = tuple(_sequence(
            cdf["cumulative_probabilities"],
            f"{point_label} {metric_name} CDF probabilities",
        ))
        availability = distribution["availability"]
        if availability == "available":
            if eligible == 0:
                raise ValueError(f"{point_label} available {metric_name} is empty")
            if percentiles != tuple(range(101)) or len(values) != 101:
                raise ValueError(
                    f"{point_label} {metric_name} must contain P0-P100")
            _nondecreasing_nonnegative(values, f"{point_label} {metric_name} quantiles")
            if len(cdf_values) != len(probabilities) or not cdf_values:
                raise ValueError(f"{point_label} {metric_name} CDF is incomplete")
            _nondecreasing_nonnegative(cdf_values, f"{point_label} {metric_name} CDF")
            previous = 0.0
            for probability in probabilities:
                normalized = _nonnegative_number(
                    probability, f"{point_label} {metric_name} CDF probability")
                if normalized <= previous or normalized > 1:
                    raise ValueError(
                        f"{point_label} {metric_name} CDF probabilities are invalid")
                previous = normalized
            if previous != 1.0:
                raise ValueError(f"{point_label} {metric_name} CDF must end at 1")
        elif availability == "unavailable_no_eligible_requests":
            if eligible != 0 or percentiles or values or cdf_values or probabilities:
                raise ValueError(
                    f"{point_label} unavailable {metric_name} contains values")
        else:
            raise ValueError(
                f"{point_label} unknown {metric_name} availability: {availability!r}")


def _validate_slo_goodput(value: object, point_label: str) -> None:
    slo = _mapping(value, f"{point_label} SLO goodput")
    _exact_fields(
        slo,
        {
            "schema_version",
            "availability",
            "contract",
            "population",
            "violation_counts",
            "request_goodput_per_s",
            "token_goodput_per_s",
        },
        f"{point_label} SLO goodput",
    )
    if slo["schema_version"] != 1:
        raise ValueError(f"{point_label} SLO goodput schema_version must be 1")
    population = _mapping(
        slo["population"], f"{point_label} SLO population")
    _exact_fields(
        population,
        {
            "total_request_count",
            "eligible_request_count",
            "excluded_request_count",
            "slo_met_request_count",
            "violated_request_count",
            "eligible_token_count",
            "slo_met_token_count",
        },
        f"{point_label} SLO population",
    )
    availability = slo["availability"]
    if availability == "unavailable_no_slo_contract":
        if (slo["contract"] is not None
                or slo["violation_counts"] is not None
                or slo["request_goodput_per_s"] is not None
                or slo["token_goodput_per_s"] is not None):
            raise ValueError(
                f"{point_label} unavailable SLO goodput contains results")
    elif availability == "available":
        contract = _mapping(slo["contract"], f"{point_label} SLO contract")
        _slo_contract(contract, point_label)
        violations = _mapping(
            slo["violation_counts"], f"{point_label} SLO violations")
        _exact_fields(
            violations,
            {"ttft_only", "tpot_only", "ttft_and_tpot"},
            f"{point_label} SLO violations",
        )
        _nonnegative_number(
            slo["request_goodput_per_s"], f"{point_label} request SLO goodput")
        _nonnegative_number(
            slo["token_goodput_per_s"], f"{point_label} token SLO goodput")
    else:
        raise ValueError(f"{point_label} unknown SLO availability: {availability!r}")


def _slo_contract(
    contract: Mapping[str, object],
    point_label: str,
) -> SLOContract:
    expected_fields = {
        "schema_version",
        "contract_id",
        "ttft_threshold_s",
        "tpot_threshold_s",
        "comparison",
        "threshold_composition",
        "request_eligibility",
        "single_output_tpot",
        "token_basis",
        "duration_basis",
        "source_path",
        "source_sha256",
    }
    _exact_fields(contract, expected_fields, f"{point_label} SLO contract")
    return SLOContract(
        schema_version=contract["schema_version"],
        contract_id=contract["contract_id"],
        ttft_threshold_s=contract["ttft_threshold_s"],
        tpot_threshold_s=contract["tpot_threshold_s"],
        comparison=contract["comparison"],
        threshold_composition=contract["threshold_composition"],
        request_eligibility=contract["request_eligibility"],
        single_output_tpot=contract["single_output_tpot"],
        token_basis=contract["token_basis"],
        duration_basis=contract["duration_basis"],
        source_path=contract["source_path"],
        source_sha256=contract["source_sha256"],
    )


def _validate_system_resources(value: object, point_label: str) -> None:
    resources = _mapping(value, f"{point_label} system resources")
    _required_fields(
        resources, {"schema_version"}, f"{point_label} system resources")
    common_fields = {
        "schema_version",
        "observation_window",
        "worker_pool_activity_basis",
        "link_activity_basis",
        "communication_resource_activity_basis",
        "worker_pool_utilization",
        "link_utilization",
        "communication_resource_utilization",
        "router_queue",
        "scheduler_queue",
    }
    if resources["schema_version"] == 4:
        _exact_fields(
            resources, common_fields, f"{point_label} system resources")
    elif resources["schema_version"] == 5:
        _exact_fields(
            resources,
            common_fields | {
                "host_kv_link_activity_basis",
                "host_kv_link_utilization",
            },
            f"{point_label} system resources",
        )
    else:
        raise ValueError(
            f"{point_label} system resource schema_version must be 4 or 5")
    window = _mapping(
        resources["observation_window"], f"{point_label} observation window")
    _required_fields(window, {"basis", "duration_s"},
                     f"{point_label} observation window")
    _positive_number(window["duration_s"], f"{point_label} observation duration")
    for field in (
            "worker_pool_utilization",
            "link_utilization",
            "communication_resource_utilization",
            "router_queue",
            "scheduler_queue"):
        _mapping(resources[field], f"{point_label} {field}")
    if resources["schema_version"] == 5:
        _validate_host_kv_link_resources(resources, point_label)
    worker_pools = _mapping(
        resources["worker_pool_utilization"],
        f"{point_label} worker-pool utilization",
    )
    if not worker_pools:
        raise ValueError(f"{point_label} has no worker-pool utilization")
    for pool_name, record_value in worker_pools.items():
        record = _mapping(
            record_value, f"{point_label} worker pool {pool_name}")
        _exact_fields(
            record,
            {"kernel_interval_count", "busy_time_s", "idle_time_s", "utilization"},
            f"{point_label} worker pool {pool_name}",
        )
        _nonnegative_integer(
            record["kernel_interval_count"],
            f"{point_label} worker pool {pool_name} interval count",
        )
        for field in ("busy_time_s", "idle_time_s", "utilization"):
            _nonnegative_number(
                record[field], f"{point_label} worker pool {pool_name} {field}")
    router = _mapping(resources["router_queue"], f"{point_label} router queue")
    _required_fields(
        router, {"availability", "overall", "by_stage"},
        f"{point_label} router queue")
    if router["availability"] == "modeled_serial_dispatch_resource":
        _exact_fields(
            router,
            {
                "availability",
                "timing_semantics",
                "resource_id",
                "configured_dispatch_service_time_s",
                "overall",
                "by_stage",
                "dispatch_service",
            },
            f"{point_label} router queue",
        )
    elif router["availability"] == "not_applicable_single_pool_without_router":
        _exact_fields(
            router, {"availability", "overall", "by_stage"},
            f"{point_label} router queue")
    else:
        raise ValueError(
            f"{point_label} router queue availability is unsupported")
    scheduler = _mapping(
        resources["scheduler_queue"], f"{point_label} scheduler queue")
    _required_fields(
        scheduler,
        {"availability", "admission_basis", "overall", "by_stage", "by_pool"},
        f"{point_label} scheduler queue",
    )
    _exact_fields(
        scheduler,
        {"availability", "admission_basis", "overall", "by_stage", "by_pool"},
        f"{point_label} scheduler queue",
    )


def _validate_host_kv_link_resources(
    resources: Mapping[str, object],
    point_label: str,
) -> None:
    if resources["host_kv_link_activity_basis"] != (
            "per_rank_resource_union_of_canonical_kv_offload_physical_"
            "completion_intervals"):
        raise ValueError(
            f"{point_label} Host-KV link activity basis is unsupported")
    host_resources = _mapping(
        resources["host_kv_link_utilization"],
        f"{point_label} Host-KV link utilization",
    )
    window = _mapping(
        resources["observation_window"],
        f"{point_label} observation window",
    )
    duration_s = _positive_number(
        window["duration_s"], f"{point_label} observation duration")
    for resource_id, raw_record in host_resources.items():
        label = f"{point_label} Host-KV resource {resource_id}"
        record = _mapping(raw_record, label)
        _exact_fields(
            record,
            {
                "pool_name", "rank", "hardware_id", "link",
                "physical_transfer_count", "action_counts",
                "bytes_by_action", "transferred_bytes", "busy_time_s",
                "idle_time_s", "utilization",
            },
            label,
        )
        pool_name = record["pool_name"]
        hardware_id = record["hardware_id"]
        link = record["link"]
        if (not isinstance(pool_name, str) or not pool_name
                or not isinstance(hardware_id, str) or not hardware_id
                or not isinstance(link, str) or not link):
            raise ValueError(f"{label} identity fields must be non-empty")
        rank = _nonnegative_integer(record["rank"], f"{label} rank")
        if resource_id != f"pool:{pool_name}:rank:{rank}:{link}":
            raise ValueError(f"{label} ID disagrees with pool/rank/link")
        physical_count = _nonnegative_integer(
            record["physical_transfer_count"],
            f"{label} physical transfer count",
        )
        if physical_count == 0:
            raise ValueError(f"{label} must contain a physical transfer")
        action_counts = _mapping(
            record["action_counts"], f"{label} action counts")
        bytes_by_action = _mapping(
            record["bytes_by_action"], f"{label} bytes by action")
        if (not action_counts
                or set(action_counts) != set(bytes_by_action)
                or not set(action_counts).issubset({"evict", "load"})):
            raise ValueError(f"{label} action coverage is invalid")
        normalized_counts = {
            action: _nonnegative_integer(
                action_counts[action], f"{label} {action} count")
            for action in action_counts
        }
        normalized_bytes = {
            action: _nonnegative_integer(
                bytes_by_action[action], f"{label} {action} bytes")
            for action in bytes_by_action
        }
        if (any(value == 0 for value in normalized_counts.values())
                or any(value == 0 for value in normalized_bytes.values())
                or sum(normalized_counts.values()) != physical_count):
            raise ValueError(f"{label} action totals are invalid")
        transferred_bytes = _nonnegative_integer(
            record["transferred_bytes"], f"{label} transferred bytes")
        if (transferred_bytes == 0
                or transferred_bytes != sum(normalized_bytes.values())):
            raise ValueError(f"{label} byte total is invalid")
        busy_time_s = _nonnegative_number(
            record["busy_time_s"], f"{label} busy time")
        idle_time_s = _nonnegative_number(
            record["idle_time_s"], f"{label} idle time")
        utilization = _nonnegative_number(
            record["utilization"], f"{label} utilization")
        if (not math.isclose(
                busy_time_s + idle_time_s,
                duration_s,
                rel_tol=1e-12,
                abs_tol=1e-15)
                or not math.isclose(
                    utilization,
                    busy_time_s / duration_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15)):
            raise ValueError(f"{label} utilization arithmetic is invalid")


def _validate_host_kv_resource_binding(
    artifact_value: object,
    resources_value: object,
    point_label: str,
) -> None:
    artifact = _mapping(
        artifact_value, f"{point_label} execution artifact")
    resources = _mapping(
        resources_value, f"{point_label} system resources")
    modeled = _mapping(
        artifact["modeled_kv_offload"],
        f"{point_label} modeled KV offload",
    )
    if (artifact["schema_version"] not in (7, 8)
            or modeled["schema_version"] != 4):
        if (resources["schema_version"] == 5
                and _mapping(
                    resources["host_kv_link_utilization"],
                    f"{point_label} Host-KV link utilization")):
            raise ValueError(
                f"{point_label} legacy/disabled execution artifact cannot "
                "claim Host-KV link utilization")
        return
    if resources["schema_version"] != 5:
        raise ValueError(
            f"{point_label} execution artifact v7 requires "
            "SystemResourceManifest v5")

    if modeled["schema_version"] != 4:
        raise ValueError(
            f"{point_label} execution artifact v7 requires modeled KV "
            "offload v4")
    steps_by_id = {}
    for raw_step in _sequence(
            artifact["steps"], f"{point_label} execution steps"):
        step = _mapping(raw_step, f"{point_label} execution step")
        step_id = step["step_id"]
        if step_id in steps_by_id:
            raise ValueError(
                f"{point_label} has duplicate execution step IDs")
        steps_by_id[step_id] = step

    records_by_resource = {}
    for raw_transfer in _sequence(
            modeled["transfers"], f"{point_label} modeled KV transfers"):
        transfer = _mapping(
            raw_transfer, f"{point_label} modeled KV transfer")
        step_id = transfer["step_id"]
        if step_id not in steps_by_id:
            raise ValueError(
                f"{point_label} Host-KV transfer references unknown step")
        step = steps_by_id[step_id]
        calls_by_id = {
            call["call_id"]: call
            for call in (
                _mapping(raw_call, f"{point_label} kernel call")
                for raw_call in _sequence(
                    _mapping(
                        step["kernel_plan"],
                        f"{point_label} kernel plan",
                    )["calls"],
                    f"{point_label} kernel calls",
                )
            )
        }
        action = _mapping(
            transfer["intent"], f"{point_label} KV intent")["action"]
        for raw_physical in _sequence(
                transfer["physical_transfers"],
                f"{point_label} physical KV transfers"):
            physical = _mapping(
                raw_physical, f"{point_label} physical KV transfer")
            resource_id = physical["resource_id"]
            call_id = physical["call_id"]
            if call_id not in calls_by_id:
                raise ValueError(
                    f"{point_label} Host-KV transfer lacks a kernel call")
            call = calls_by_id[call_id]
            completion = _mapping(
                physical["completion"],
                f"{point_label} physical KV completion",
            )
            interval = _mapping(
                physical["absolute_interval"],
                f"{point_label} physical KV absolute interval",
            )
            contract = (
                physical["pool_name"],
                physical["rank"],
                _mapping(
                    call["kernel_spec"],
                    f"{point_label} KV KernelSpec",
                )["hardware_id"],
                completion["link"],
            )
            if resource_id not in records_by_resource:
                records_by_resource[resource_id] = {
                    "contract": contract,
                    "intervals": [],
                    "physical_transfer_count": 0,
                    "action_counts": {},
                    "bytes_by_action": {},
                }
            record = records_by_resource[resource_id]
            if record["contract"] != contract:
                raise ValueError(
                    f"{point_label} Host-KV resource contract changed")
            record["intervals"].append((
                interval["start_time_s"], interval["end_time_s"]))
            record["physical_transfer_count"] += 1
            if action not in record["action_counts"]:
                record["action_counts"][action] = 0
                record["bytes_by_action"][action] = 0
            record["action_counts"][action] += 1
            record["bytes_by_action"][action] += completion["bytes"]

    window = _mapping(
        resources["observation_window"],
        f"{point_label} observation window",
    )
    duration_s = _positive_number(
        window["duration_s"], f"{point_label} observation duration")
    schedule_entries = _sequence(
        _mapping(
            artifact["schedule_trace"],
            f"{point_label} schedule trace",
        )["entries"],
        f"{point_label} schedule entries",
    )
    if not schedule_entries:
        raise ValueError(
            f"{point_label} execution artifact v7 has no schedule entries")
    schedule_completion_s = max(
        _nonnegative_number(
            _mapping(
                raw_entry,
                f"{point_label} schedule entry {index}",
            )["end_time_s"],
            f"{point_label} schedule entry {index} end time",
        )
        for index, raw_entry in enumerate(schedule_entries)
    )
    if not math.isclose(
            duration_s,
            schedule_completion_s,
            rel_tol=1e-12,
            abs_tol=1e-15):
        raise ValueError(
            f"{point_label} Host-KV observation window disagrees with "
            "schedule completion")
    expected = {}
    for resource_id in sorted(records_by_resource):
        record = records_by_resource[resource_id]
        intervals = sorted(record["intervals"])
        previous_end_s = None
        busy_time_s = 0.0
        for start_time_s, end_time_s in intervals:
            if (start_time_s < 0.0
                    or end_time_s <= start_time_s
                    or end_time_s > duration_s + 1e-15):
                raise ValueError(
                    f"{point_label} Host-KV interval lies outside the "
                    "observation window")
            if (previous_end_s is not None
                    and start_time_s < previous_end_s - 1e-15):
                raise ValueError(
                    f"{point_label} Host-KV intervals overlap on "
                    f"{resource_id!r}")
            busy_time_s += end_time_s - start_time_s
            previous_end_s = end_time_s
        idle_time_s = duration_s - busy_time_s
        if idle_time_s < -1e-15:
            raise ValueError(
                f"{point_label} Host-KV busy time exceeds observation window")
        idle_time_s = max(0.0, idle_time_s)
        pool_name, rank, hardware_id, link = record["contract"]
        bytes_by_action = {
            action: record["bytes_by_action"][action]
            for action in sorted(record["bytes_by_action"])
        }
        expected[resource_id] = {
            "pool_name": pool_name,
            "rank": rank,
            "hardware_id": hardware_id,
            "link": link,
            "physical_transfer_count": record["physical_transfer_count"],
            "action_counts": {
                action: record["action_counts"][action]
                for action in sorted(record["action_counts"])
            },
            "bytes_by_action": bytes_by_action,
            "transferred_bytes": sum(bytes_by_action.values()),
            "busy_time_s": busy_time_s,
            "idle_time_s": idle_time_s,
            "utilization": busy_time_s / duration_s,
        }
    actual = _mapping(
        resources["host_kv_link_utilization"],
        f"{point_label} Host-KV link utilization",
    )
    if actual != expected or not expected:
        raise ValueError(
            f"{point_label} Host-KV link utilization disagrees with "
            "execution artifact v7")


def _validate_critical_path(value: object, point_label: str) -> None:
    path = _mapping(value, f"{point_label} critical path")
    _exact_fields(
        path,
        {"schema_version", "semantics", "steps", "requests", "system_completion"},
        f"{point_label} critical path",
    )
    if path["schema_version"] != 2:
        raise ValueError(f"{point_label} critical path schema_version must be 2")
    _mapping(path["semantics"], f"{point_label} critical-path semantics")
    steps = _sequence(path["steps"], f"{point_label} critical-path steps")
    for index, step_value in enumerate(steps):
        step = _mapping(step_value, f"{point_label} critical-path step {index}")
        _required_fields(
            step,
            {
                "step_id",
                "pool_name",
                "duration_s",
                "canonical_path_unique",
                "contribution_s_by_category",
                "dominant_category",
            },
            f"{point_label} critical-path step {index}",
        )
        _nonnegative_number(
            step["duration_s"],
            f"{point_label} critical-path step {index} duration",
        )
        if not isinstance(step["canonical_path_unique"], bool):
            raise ValueError(
                f"{point_label} critical-path step {index} unique flag "
                "must be boolean")
        _validate_contributions(
            step["contribution_s_by_category"],
            f"{point_label} critical-path step {index}",
        )
    requests = _mapping(
        path["requests"], f"{point_label} critical-path requests")
    for request_id, request_value in requests.items():
        request = _mapping(
            request_value, f"{point_label} critical-path request {request_id}")
        _exact_fields(
            request, {"arrival_time_s", "attribution", "ttft", "e2e"},
            f"{point_label} critical-path request {request_id}")
        _nonnegative_number(
            request["arrival_time_s"],
            f"{point_label} critical-path request {request_id} arrival",
        )
        if not isinstance(request["attribution"], str):
            raise ValueError(
                f"{point_label} critical-path request {request_id} "
                "attribution must be a string")
        for target_name in ("ttft", "e2e"):
            target = _mapping(
                request[target_name],
                f"{point_label} request {request_id} {target_name}",
            )
            _required_fields(
                target, {"availability", "duration_s", "dominant_category"},
                f"{point_label} request {request_id} {target_name}",
            )
            if target["availability"] == "available":
                _nonnegative_number(
                    target["duration_s"],
                    f"{point_label} request {request_id} {target_name} duration",
                )
            elif (target["availability"]
                  == "not_applicable_zero_output_tokens"):
                if (target["duration_s"] is not None
                        or target["dominant_category"] is not None):
                    raise ValueError(
                        f"{point_label} unavailable request path has values")
            else:
                raise ValueError(
                    f"{point_label} request path availability is unsupported")
    completion = _mapping(
        path["system_completion"], f"{point_label} system completion")
    _required_fields(
        completion,
        {"time_s", "terminal_request_ids", "terminal_observation_paths"},
        f"{point_label} system completion",
    )
    _positive_number(completion["time_s"], f"{point_label} system completion time")
    terminal_paths = _sequence(
        completion["terminal_observation_paths"],
        f"{point_label} terminal observation paths",
    )
    if not terminal_paths:
        raise ValueError(f"{point_label} has no terminal critical path")
    for index, terminal_value in enumerate(terminal_paths):
        terminal = _mapping(
            terminal_value, f"{point_label} terminal path {index}")
        _exact_fields(
            terminal,
            {
                "request_id",
                "path_start_time_s",
                "path_duration_s",
                "unattributed_pre_arrival_window_s",
                "contribution_s_by_category",
                "dominant_category",
            },
            f"{point_label} terminal path {index}",
        )
        if not isinstance(terminal["request_id"], str):
            raise ValueError(
                f"{point_label} terminal path {index} request ID is invalid")
        for field in (
                "path_start_time_s",
                "path_duration_s",
                "unattributed_pre_arrival_window_s"):
            _nonnegative_number(
                terminal[field], f"{point_label} terminal path {index} {field}")
        _validate_contributions(
            terminal["contribution_s_by_category"],
            f"{point_label} terminal path {index}",
        )


def _validate_contributions(value: object, label: str) -> None:
    contributions = _mapping(value, f"{label} contributions")
    for category, duration in contributions.items():
        if not isinstance(category, str) or not category:
            raise ValueError(f"{label} contribution category is invalid")
        _nonnegative_number(duration, f"{label} contribution {category}")


def _validate_sweep_analysis(
    sweep: Mapping[str, object],
    points: Sequence[Mapping[str, object]],
) -> None:
    tradeoff = _mapping(sweep["interactivity_tgs"], "Interactivity-TGS analysis")
    peak = _mapping(sweep["peak_selection"], "sampled peak selection")
    saturation = _mapping(sweep["saturation_analysis"], "saturation analysis")
    _required_fields(
        tradeoff,
        {"availability", "contract", "points", "pareto_frontier"},
        "Interactivity-TGS analysis",
    )
    _required_fields(
        peak, {"availability", "contract"}, "sampled peak selection")
    _required_fields(
        saturation,
        {"availability", "contract", "intervals"},
        "saturation analysis",
    )
    if tradeoff["contract"] is None:
        expected = arrival_rate_sweep_analysis_manifest(None, None)
    else:
        contract = _interactivity_tgs_contract(tradeoff["contract"])
        canonical_points = tuple(
            sweep_metric_point_from_manifest(
                point["arrival_rate_scale"],
                _mapping(point["result"], "sweep point result")["metrics"],
                contract,
            )
            for point in points
        )
        expected = arrival_rate_sweep_analysis_manifest(
            canonical_points, contract)
    for analysis_name, actual in (
            ("interactivity_tgs", tradeoff),
            ("peak_selection", peak),
            ("saturation_analysis", saturation)):
        if (_canonical_json_bytes(actual)
                != _canonical_json_bytes(expected[analysis_name])):
            raise ValueError(
                f"{analysis_name} disagrees with canonical sweep metrics")


def _interactivity_tgs_contract(value: object) -> InteractivityTGSContract:
    contract = _mapping(value, "Interactivity-TGS contract")
    expected_fields = {
        "schema_version",
        "contract_id",
        "interactivity_metric",
        "interactivity_percentile",
        "interactivity_unit",
        "interactivity_aggregation",
        "interactivity_eligibility",
        "tgs_metric",
        "tgs_unit",
        "tgs_aggregation",
        "peak_selection",
        "saturation_rule",
        "saturation_marginal_gain_ratio_threshold",
        "saturation_minimum_consecutive_intervals",
        "source_path",
        "source_sha256",
    }
    _exact_fields(contract, expected_fields, "Interactivity-TGS contract")
    return InteractivityTGSContract(
        schema_version=contract["schema_version"],
        contract_id=contract["contract_id"],
        interactivity_metric=contract["interactivity_metric"],
        interactivity_percentile=contract["interactivity_percentile"],
        interactivity_unit=contract["interactivity_unit"],
        interactivity_aggregation=contract["interactivity_aggregation"],
        interactivity_eligibility=contract["interactivity_eligibility"],
        tgs_metric=contract["tgs_metric"],
        tgs_unit=contract["tgs_unit"],
        tgs_aggregation=contract["tgs_aggregation"],
        peak_selection=contract["peak_selection"],
        saturation_rule=contract["saturation_rule"],
        saturation_marginal_gain_ratio_threshold=contract[
            "saturation_marginal_gain_ratio_threshold"],
        saturation_minimum_consecutive_intervals=contract[
            "saturation_minimum_consecutive_intervals"],
        source_path=contract["source_path"],
        source_sha256=contract["source_sha256"],
    )
