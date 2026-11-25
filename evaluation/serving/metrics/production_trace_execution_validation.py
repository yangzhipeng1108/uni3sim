"""Execution-side validation for Production Trace sweep points:
Host timing profiles, source traces, execution provenance, and
modeled KV-offload artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re

from evaluation.serving.simulation.execution_artifact import performance_timing_status
from src.serving.host_timing import (
    LAUNCH_KINDS,
    STEP_PHASES,
    validate_serving_timing_mode,
)

from evaluation.serving.metrics.sharegpt_compact_evaluation import (
    EXECUTION_ARTIFACT_FIELDS,
)

from evaluation.json_shape import require_object as _mapping

from evaluation.serving.metrics.production_trace_report_shape import (
    _SOURCE_TRACE_FIELDS,
    _EXECUTION_PROVENANCE_FIELDS,
    _sequence,
    _exact_fields,
    _required_fields,
    _sha256,
    _string,
    _positive_number,
    _nonnegative_number,
    _nonnegative_integer,
)
from evaluation.serving.metrics.compact_execution_payload_validation import validate_execution_repository_provenance


def _validate_execution_and_host_timing(
    result: Mapping[str, object],
    label: str,
) -> None:
    execution_mode = result["execution_mode"]
    if execution_mode not in ("accelerated", "reference", "fast"):
        raise ValueError(f"{label} execution mode is invalid")
    mode = validate_serving_timing_mode(result["serving_timing_mode"])
    hardware_ids = tuple(
        _string(item, f"{label} hardware ID")
        for item in _sequence(result["hardware_ids"], f"{label} hardware IDs")
    )
    if not hardware_ids or len(hardware_ids) != len(set(hardware_ids)):
        raise ValueError(f"{label} hardware IDs must be non-empty and unique")
    timing = _mapping(result["host_timing"], f"{label} Host timing")
    _exact_fields(timing, {
        "schema_version", "timing_mode", "scheduler_measurement_boundary",
        "launch_measurement_boundary", "profiles", "runtime_prepare_profiles",
        "aggregate", "steps",
    }, f"{label} Host timing")
    if timing["schema_version"] != 1 or timing["timing_mode"] != mode:
        raise ValueError(f"{label} Host timing schema or mode is invalid")
    if timing["scheduler_measurement_boundary"] != (
            "in_scheduler_process_schedule_and_output_update_only"):
        raise ValueError(f"{label} Host timing Scheduler boundary is invalid")
    if timing["launch_measurement_boundary"] != (
            "host_submit_only_no_device_sync"):
        raise ValueError(f"{label} Host timing launch boundary is invalid")

    profiles = tuple(
        _validate_host_launch_profile_manifest(profile, label)
        for profile in _sequence(
            timing["profiles"], f"{label} Host launch profiles")
    )
    profile_hardware_ids = tuple(profile["hardware_id"] for profile in profiles)
    if mode == "device_only":
        expected_profile_hardware_ids = ()
    elif mode in ("device_plus_launch", "device_plus_launch_plus_scheduler"):
        expected_profile_hardware_ids = tuple(sorted(hardware_ids))
    else:
        raise AssertionError("validated Serving timing mode is unknown")
    if profile_hardware_ids != expected_profile_hardware_ids:
        raise ValueError(
            f"{label} Host launch profiles disagree with timing mode or hardware")
    runtime_prepare_profiles = tuple(
        _validate_host_runtime_prepare_profile_manifest(profile, label)
        for profile in _sequence(
            timing["runtime_prepare_profiles"],
            f"{label} Host runtime prepare profiles")
    )
    runtime_prepare_hardware_ids = tuple(
        profile["hardware_id"] for profile in runtime_prepare_profiles)
    if any(hardware_id not in hardware_ids
           for hardware_id in runtime_prepare_hardware_ids):
        raise ValueError(
            f"{label} Host runtime prepare profile hardware is not deployed")

    aggregate = _mapping(
        timing["aggregate"], f"{label} Host timing aggregate")
    summed_fields = {
        "predicted_device_s", "launch_cpu_s", "launch_critical_path_s",
        "host_device_overlap_s", "scheduler_plan_cpu_s",
        "scheduler_update_cpu_s", "execution_wall_s", "step_wall_s",
        "eager_dispatch_count", "graph_replay_count",
    }
    aggregate_fields = summed_fields | {
        "scheduler_cpu_s", "schedule_makespan_s", "step_count",
    }
    _exact_fields(aggregate, aggregate_fields, f"{label} Host timing aggregate")
    count_fields = {
        "eager_dispatch_count", "graph_replay_count", "step_count",
    }
    for field in aggregate_fields - count_fields:
        _nonnegative_number(aggregate[field], f"{label} Host timing {field}")
    for field in count_fields:
        _nonnegative_integer(aggregate[field], f"{label} Host timing {field}")
    if not math.isclose(
            aggregate["scheduler_cpu_s"],
            aggregate["scheduler_plan_cpu_s"]
            + aggregate["scheduler_update_cpu_s"],
            rel_tol=1e-12,
            abs_tol=1e-15):
        raise ValueError(f"{label} Host Scheduler CPU aggregate is inconsistent")

    steps = tuple(
        _validate_host_timing_step(step, label)
        for step in _sequence(timing["steps"], f"{label} Host timing steps")
    )
    if not steps or len(steps) != aggregate["step_count"]:
        raise ValueError(f"{label} Host timing step count is inconsistent")
    for field in summed_fields:
        observed_sum = sum(step[field] for step in steps)
        if not math.isclose(
                aggregate[field], observed_sum, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"{label} Host timing aggregate {field} is inconsistent")


def _validate_host_launch_profile_manifest(
    value: object,
    label: str,
) -> Mapping[str, object]:
    profile = _mapping(value, f"{label} Host launch profile")
    if "schema_version" not in profile:
        raise ValueError(f"{label} Host launch profile lacks schema_version")
    schema_version = profile["schema_version"]
    if type(schema_version) is not int:
        raise TypeError(f"{label} Host launch profile schema must be an integer")
    common_fields = {
        "schema_version", "hardware_id", "host_cpu", "operating_system",
        "driver_version", "runtime", "framework", "service_time_s",
        "measurement", "source",
    }
    if schema_version == 1:
        _exact_fields(
            profile,
            common_fields | {"phase_launch_kind"},
            f"{label} Host launch profile",
        )
        phases = _mapping(
            profile["phase_launch_kind"],
            f"{label} Host launch phase mapping",
        )
        _exact_fields(
            phases, set(STEP_PHASES),
            f"{label} Host launch phase mapping",
        )
        if any(phases[phase] not in LAUNCH_KINDS for phase in STEP_PHASES):
            raise ValueError(f"{label} Host launch phase mapping is invalid")
    elif schema_version == 2:
        # ``device_launch_gap`` is the schema's one optional section.
        _exact_fields(
            profile,
            common_fields | (
                {"device_launch_gap"} if "device_launch_gap" in profile
                else set()),
            f"{label} Host launch profile",
        )
    else:
        raise ValueError(f"{label} Host launch profile schema is invalid")
    for field in (
            "hardware_id", "host_cpu", "operating_system", "driver_version"):
        _string(profile[field], f"{label} Host launch profile {field}")
    for field in ("runtime", "framework"):
        identity = _mapping(
            profile[field], f"{label} Host launch profile {field}")
        _exact_fields(
            identity, {"name", "version"},
            f"{label} Host launch profile {field}")
        _string(identity["name"], f"{label} Host launch profile {field} name")
        _string(
            identity["version"], f"{label} Host launch profile {field} version")
    service = _mapping(
        profile["service_time_s"], f"{label} Host launch service time")
    _exact_fields(
        service, {"eager_enqueue", "graph_replay_enqueue"},
        f"{label} Host launch service time")
    for field in ("eager_enqueue", "graph_replay_enqueue"):
        _positive_number(service[field], f"{label} Host launch {field}")
    measurement = _mapping(
        profile["measurement"], f"{label} Host launch measurement")
    _exact_fields(
        measurement, {"artifact", "sha256", "boundary"},
        f"{label} Host launch measurement")
    _string(measurement["artifact"], f"{label} Host launch artifact")
    _sha256(measurement["sha256"], f"{label} Host launch measurement SHA256")
    if measurement["boundary"] != "host_submit_only_no_device_sync":
        raise ValueError(f"{label} Host launch measurement boundary is invalid")
    source = _mapping(profile["source"], f"{label} Host launch source")
    _exact_fields(
        source, {"commit", "path", "sha256"}, f"{label} Host launch source")
    commit = _string(source["commit"], f"{label} Host launch source commit")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"{label} Host launch source commit is invalid")
    _string(source["path"], f"{label} Host launch source path")
    _sha256(source["sha256"], f"{label} Host launch source SHA256")
    return profile


def _validate_host_runtime_prepare_profile_manifest(
    value: object,
    label: str,
) -> Mapping[str, object]:
    profile = _mapping(value, f"{label} Host runtime prepare profile")
    _exact_fields(profile, {
        "hardware_id", "service_time_s_by_mechanism",
        "lane_service_time_s_by_mechanism", "profile_path", "profile_sha256",
    }, f"{label} Host runtime prepare profile")
    _string(profile["hardware_id"], f"{label} runtime prepare hardware ID")
    services = _mapping(
        profile["service_time_s_by_mechanism"],
        f"{label} runtime prepare services")
    if not services:
        raise ValueError(f"{label} runtime prepare services are empty")
    for mechanism, service_time_s in services.items():
        _string(mechanism, f"{label} runtime prepare mechanism")
        _positive_number(
            service_time_s, f"{label} runtime prepare mechanism service")
    lanes = _sequence(
        profile["lane_service_time_s_by_mechanism"],
        f"{label} runtime prepare lane services")
    if lanes:
        raise ValueError(
            f"{label} runtime prepare lane services are unsupported in this report")
    _string(profile["profile_path"], f"{label} runtime prepare profile path")
    _sha256(profile["profile_sha256"], f"{label} runtime prepare profile SHA256")
    return profile


def _validate_host_timing_step(
    value: object,
    label: str,
) -> Mapping[str, object]:
    step = _mapping(value, f"{label} Host timing step")
    count_fields = {
        "scheduled_request_count", "prefill_token_count", "decode_token_count",
        "eager_dispatch_count", "graph_replay_count",
    }
    duration_fields = {
        "predicted_device_s", "launch_cpu_s", "launch_critical_path_s",
        "host_device_overlap_s", "scheduler_plan_cpu_s",
        "scheduler_update_cpu_s", "execution_wall_s", "step_wall_s",
    }
    _exact_fields(step, {
        "step_id", "pool_name", "phase", "launch_kinds",
    } | count_fields | duration_fields, f"{label} Host timing step")
    _string(step["step_id"], f"{label} Host timing step ID")
    _string(step["pool_name"], f"{label} Host timing pool name")
    if step["phase"] not in STEP_PHASES:
        raise ValueError(f"{label} Host timing phase is invalid")
    for field in count_fields:
        _nonnegative_integer(step[field], f"{label} Host timing step {field}")
    for field in duration_fields:
        _nonnegative_number(step[field], f"{label} Host timing step {field}")
    launch_kinds = _sequence(
        step["launch_kinds"], f"{label} Host timing launch kinds")
    if any(kind not in LAUNCH_KINDS for kind in launch_kinds):
        raise ValueError(f"{label} Host timing launch kind is invalid")
    return step


def _validate_source_trace(value: object) -> Mapping[str, object]:
    trace = _mapping(value, "production Trace source provenance")
    _exact_fields(
        trace, _SOURCE_TRACE_FIELDS, "production Trace source provenance")
    if trace["schema_version"] != 1:
        raise ValueError("production Trace source schema_version must be 1")
    if trace["successful_only"] is not True:
        raise ValueError("production Trace report requires successful_only=True")
    if not isinstance(trace["source_path"], str) or not trace["source_path"]:
        raise ValueError("production Trace source path must be non-empty")
    _sha256(trace["source_sha256"], "production Trace source SHA256")
    _sha256(
        trace["normalized_request_stream_sha256"],
        "production Trace normalized request stream SHA256",
    )
    _mapping(trace["statistics"], "production Trace statistics")
    return trace


def _validate_execution_provenance(value: object) -> Mapping[str, object]:
    provenance = _mapping(value, "execution provenance")
    _exact_fields(
        provenance, _EXECUTION_PROVENANCE_FIELDS, "execution provenance")
    repository = validate_execution_repository_provenance(provenance, require_unchanged=False)
    _required_fields(
        repository,
        {"git_commit", "worktree_state", "git_diff_hash"},
        "repository provenance",
    )
    commit = repository["git_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)):
        raise ValueError("execution provenance git_commit must be lowercase hex")
    if not isinstance(repository["worktree_state"], str):
        raise ValueError("execution provenance worktree_state must be a string")
    _sha256(
        repository["git_diff_hash"],
        "execution provenance git_diff_hash",
    )
    hardware_configs = _mapping(
        provenance["hardware_configs"], "hardware config provenance")
    _exact_fields(
        hardware_configs,
        {"prefill", "decode", "additional_colocated"},
        "hardware config provenance",
    )
    hardware_ids = _sequence(
        provenance["loaded_hardware_ids"], "loaded hardware IDs")
    if not hardware_ids or any(
            not isinstance(hardware_id, str) or not hardware_id
            for hardware_id in hardware_ids):
        raise ValueError("execution provenance loaded_hardware_ids are invalid")
    _mapping(provenance["runtime_inputs"], "runtime input provenance")
    launch_inputs = _mapping(
        provenance["launch_inputs"], "launch input provenance")
    _required_fields(
        launch_inputs, {"launch_policy", "launch_data"},
        "launch input provenance")
    _mapping(launch_inputs["launch_data"], "launch-data provenance")
    expected_timing_status = performance_timing_status(
        launch_inputs["launch_policy"], tuple(hardware_ids))
    if provenance["performance_timing_status"] != expected_timing_status:
        raise ValueError(
            "execution provenance performance timing status disagrees with "
            "launch policy and loaded hardware")
    _mapping(
        provenance["microbenchmark_inputs"],
        "microbenchmark input provenance",
    )
    if (not isinstance(provenance["prediction_method"], str)
            or not provenance["prediction_method"]):
        raise ValueError("execution provenance prediction_method is invalid")
    return provenance


def _validate_execution_artifact(
    value: object,
    point_label: str,
    expected_timing_status: str,
) -> None:
    artifact = _mapping(value, f"{point_label} execution artifact")
    _exact_fields(artifact, EXECUTION_ARTIFACT_FIELDS, f"{point_label} execution artifact")
    if artifact["schema_version"] not in (5, 6, 7, 8):
        raise ValueError(
            f"{point_label} execution artifact schema_version must be 5, 6, "
            "7, or 8")
    modeled_kv_offload = _mapping(
        artifact["modeled_kv_offload"],
        f"{point_label} modeled KV offload",
    )
    expected_offload_schema = {
        5: 2,
        6: 3,
        7: 4,
        8: modeled_kv_offload["schema_version"],
    }[artifact["schema_version"]]
    if (artifact["schema_version"] == 8
            and expected_offload_schema not in (2, 4)):
        raise ValueError(
            f"{point_label} compact execution artifact has unsupported "
            "modeled KV offload schema")
    if modeled_kv_offload["schema_version"] != expected_offload_schema:
        raise ValueError(
            f"{point_label} modeled KV offload schema_version disagrees with "
            "the execution artifact")
    _validate_modeled_kv_offload_artifact(
        artifact, modeled_kv_offload, point_label)
    if artifact["performance_timing_status"] != expected_timing_status:
        raise ValueError(
            f"{point_label} execution artifact performance timing status "
            "disagrees with result")
    if artifact["hash_method"] != "canonical_json_sort_keys_compact_utf8_v1":
        raise ValueError(
            f"{point_label} execution artifact hash_method is unsupported")
    expected = _sha256(
        artifact["artifact_sha256"],
        f"{point_label} execution artifact SHA256",
    )
    body = {
        key: artifact[key] for key in artifact if key != "artifact_sha256"
    }
    try:
        serialized = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{point_label} execution artifact is not canonical JSON") from error
    actual = hashlib.sha256(serialized).hexdigest()
    if actual != expected:
        raise ValueError(f"{point_label} execution artifact SHA256 is invalid")


def _validate_modeled_kv_offload_artifact(
    artifact: Mapping[str, object],
    modeled: Mapping[str, object],
    point_label: str,
) -> None:
    label = f"{point_label} modeled KV offload"
    counts = _mapping(
        artifact["counts"], f"{point_label} execution artifact counts")
    _required_fields(
        counts,
        {"modeled_kv_offload_transfers"},
        f"{point_label} execution artifact counts",
    )
    expected_transfer_count = _nonnegative_integer(
        counts["modeled_kv_offload_transfers"],
        f"{point_label} modeled KV offload transfer count",
    )

    if (artifact["schema_version"] in (7, 8)
            and modeled["schema_version"] == 4):
        _validate_modeled_kv_offload_v4(
            artifact,
            modeled,
            expected_transfer_count,
            label,
        )
        return

    if (artifact["schema_version"] in (5, 8)
            and modeled["schema_version"] == 2):
        _required_fields(modeled, {"availability"}, label)
        if modeled["availability"] == "not_enabled":
            _exact_fields(
                modeled,
                {
                    "schema_version", "availability", "intent_source",
                    "transfers", "final_tier_state",
                },
                label,
            )
            transfers = _sequence(
                modeled["transfers"], f"{label} transfers")
            if len(transfers) != expected_transfer_count:
                raise ValueError(
                    f"{label} transfers disagree with execution artifact "
                    "counts")
            if transfers or modeled["final_tier_state"] is not None:
                raise ValueError(
                    f"{label} disabled contract cannot contain transfers or "
                    "tier state")
            return
        _exact_fields(
            modeled,
            {
                "schema_version", "availability", "mode",
                "validation_status", "intent_source", "scope", "transfers",
                "final_tier_state",
            },
            label,
        )
        transfers = _sequence(
            modeled["transfers"], f"{label} transfers")
        if len(transfers) != expected_transfer_count:
            raise ValueError(
                f"{label} transfers disagree with execution artifact counts")
        if modeled["availability"] != "available_modeled_serial":
            raise ValueError(f"{label} v2 availability is invalid")
        for index, raw_transfer in enumerate(transfers):
            transfer = _mapping(
                raw_transfer, f"{label} transfer {index}")
            _exact_fields(
                transfer,
                {
                    "step_id", "ready_event", "intent", "node_id",
                    "call_id", "relative_interval", "absolute_interval",
                    "prediction_total_s", "completion",
                },
                f"{label} transfer {index}",
            )
        _mapping(modeled["final_tier_state"], f"{label} final tier state")
        return
    if artifact["schema_version"] != 6:
        raise AssertionError("validated artifact schema version vanished")

    _exact_fields(
        modeled,
        {
            "schema_version", "availability", "mode", "validation_status",
            "intent_source", "ledger_semantics", "physical_transfer_count",
            "scope", "transfers", "final_tier_state",
        },
        label,
    )
    transfers = _sequence(modeled["transfers"], f"{label} transfers")
    if len(transfers) != expected_transfer_count:
        raise ValueError(
            f"{label} transfers disagree with execution artifact counts")
    if modeled["availability"] != "available_modeled_rank_local":
        raise ValueError(f"{label} v3 availability is invalid")
    if modeled["mode"] != "modeled_host_kv_rank_local_lockstep":
        raise ValueError(f"{label} v3 mode is invalid")
    if modeled["ledger_semantics"] != (
            "logical_request_blocks_with_lockstep_per_rank_capacity"):
        raise ValueError(f"{label} v3 ledger semantics are invalid")
    if not transfers:
        raise ValueError(f"{label} v3 requires logical transfers")

    physical_count = 0
    for index, raw_transfer in enumerate(transfers):
        transfer_label = f"{label} transfer {index}"
        transfer = _mapping(raw_transfer, transfer_label)
        _exact_fields(
            transfer,
            {
                "step_id", "ready_event", "intent", "logical_completion",
                "physical_transfers",
            },
            transfer_label,
        )
        intent = _mapping(transfer["intent"], f"{transfer_label} intent")
        _required_fields(
            intent,
            {
                "ready_event", "action", "bytes", "source_tier",
                "destination_tier",
            },
            f"{transfer_label} intent",
        )
        logical = _mapping(
            transfer["logical_completion"],
            f"{transfer_label} logical completion",
        )
        _exact_fields(
            logical,
            {
                "ready_event", "action", "bytes_per_rank", "source_tier",
                "destination_tier", "link", "physical_transfer_count",
                "relative_interval", "absolute_interval",
            },
            f"{transfer_label} logical completion",
        )
        bytes_per_rank = _nonnegative_integer(
            logical["bytes_per_rank"],
            f"{transfer_label} bytes per rank",
        )
        if bytes_per_rank == 0:
            raise ValueError(f"{transfer_label} bytes per rank must be positive")
        if (transfer["ready_event"] != intent["ready_event"]
                or transfer["ready_event"] != logical["ready_event"]
                or intent["action"] != logical["action"]
                or intent["bytes"] != bytes_per_rank
                or intent["source_tier"] != logical["source_tier"]
                or intent["destination_tier"]
                != logical["destination_tier"]):
            raise ValueError(
                f"{transfer_label} logical provenance is inconsistent")
        physical_transfers = _sequence(
            transfer["physical_transfers"],
            f"{transfer_label} physical transfers",
        )
        if len(physical_transfers) != 2:
            raise ValueError(
                f"{transfer_label} requires exactly two physical transfers")
        if logical["physical_transfer_count"] != 2:
            raise ValueError(
                f"{transfer_label} logical physical count must be two")
        ranks = []
        for physical_index, raw_physical in enumerate(physical_transfers):
            physical_label = (
                f"{transfer_label} physical transfer {physical_index}")
            physical = _mapping(raw_physical, physical_label)
            _exact_fields(
                physical,
                {
                    "rank", "pool_name", "node_id", "call_id",
                    "relative_interval", "absolute_interval",
                    "prediction_total_s", "completion",
                },
                physical_label,
            )
            completion = _mapping(
                physical["completion"], f"{physical_label} completion")
            _exact_fields(
                completion,
                {
                    "node_id", "call_id", "pool_name", "rank", "bytes",
                    "link", "start_time_s", "end_time_s",
                },
                f"{physical_label} completion",
            )
            rank = _nonnegative_integer(
                physical["rank"], f"{physical_label} rank")
            ranks.append(rank)
            if (completion["rank"] != rank
                    or completion["pool_name"] != physical["pool_name"]
                    or completion["node_id"] != physical["node_id"]
                    or completion["call_id"] != physical["call_id"]
                    or completion["bytes"] != bytes_per_rank
                    or completion["link"] != logical["link"]):
                raise ValueError(
                    f"{physical_label} completion provenance is inconsistent")
            interval = _mapping(
                physical["relative_interval"],
                f"{physical_label} relative interval",
            )
            _exact_fields(
                interval, {"start_time_s", "end_time_s"},
                f"{physical_label} relative interval")
            start = _nonnegative_number(
                interval["start_time_s"], f"{physical_label} start")
            end = _nonnegative_number(
                interval["end_time_s"], f"{physical_label} end")
            prediction = _positive_number(
                physical["prediction_total_s"],
                f"{physical_label} prediction")
            if (end <= start
                    or not math.isclose(
                        end - start, prediction,
                        rel_tol=1e-12, abs_tol=1e-15)
                    or completion["start_time_s"] != interval["start_time_s"]
                    or completion["end_time_s"] != interval["end_time_s"]):
                raise ValueError(
                    f"{physical_label} interval/prediction is inconsistent")
        if tuple(ranks) != (0, 1):
            raise ValueError(
                f"{transfer_label} physical ranks must be exactly (0, 1)")
        physical_count += len(physical_transfers)
    declared_physical_count = _nonnegative_integer(
        modeled["physical_transfer_count"],
        f"{label} physical transfer count",
    )
    if declared_physical_count != physical_count:
        raise ValueError(f"{label} physical transfer count is inconsistent")
    _mapping(modeled["final_tier_state"], f"{label} final tier state")


def _validate_modeled_kv_offload_v4(
    artifact: Mapping[str, object],
    modeled: Mapping[str, object],
    expected_transfer_count: int,
    label: str,
) -> None:
    _exact_fields(
        modeled,
        {
            "schema_version", "availability", "mode", "validation_status",
            "intent_source", "ledger_semantics", "physical_transfer_count",
            "scope", "transfers", "final_tier_state",
        },
        label,
    )
    if modeled["schema_version"] != 4:
        raise ValueError(f"{label} schema_version must be 4")
    schedule_trace = _mapping(
        artifact["schedule_trace"], f"{label} schedule trace")
    schedule_entries = _sequence(
        schedule_trace["entries"], f"{label} schedule entries")
    schedule_entries_by_id = {}
    for index, raw_entry in enumerate(schedule_entries):
        entry = _mapping(raw_entry, f"{label} schedule entry {index}")
        _required_fields(
            entry, {"step_id", "pool_name", "start_time_s", "end_time_s"},
            f"{label} schedule entry {index}")
        step_id = entry["step_id"]
        if step_id in schedule_entries_by_id:
            raise ValueError(f"{label} has duplicate schedule step IDs")
        schedule_entries_by_id[step_id] = entry
    steps = _sequence(artifact["steps"], f"{label} artifact steps")
    steps_by_id = {}
    for index, raw_step in enumerate(steps):
        step = _mapping(raw_step, f"{label} artifact step {index}")
        _required_fields(
            step,
            {
                "step_id", "pool_name", "schedule_interval",
                "workload_graph", "kernel_plan", "predictions",
                "execution_timeline",
            },
            f"{label} artifact step {index}",
        )
        step_id = step["step_id"]
        if step_id in steps_by_id:
            raise ValueError(f"{label} has duplicate artifact step IDs")
        if step_id not in schedule_entries_by_id:
            raise ValueError(f"{label} artifact step lacks a schedule entry")
        entry = schedule_entries_by_id[step_id]
        schedule_interval = _mapping(
            step["schedule_interval"],
            f"{label} step {step_id} schedule interval",
        )
        _exact_fields(
            schedule_interval, {"start_time_s", "end_time_s"},
            f"{label} step {step_id} schedule interval")
        if (step["pool_name"] != entry["pool_name"]
                or schedule_interval["start_time_s"] != entry["start_time_s"]
                or schedule_interval["end_time_s"] != entry["end_time_s"]):
            raise ValueError(
                f"{label} artifact step disagrees with its schedule entry")
        steps_by_id[step_id] = step
    if set(steps_by_id) != set(schedule_entries_by_id):
        raise ValueError(f"{label} step/schedule coverage is inconsistent")
    availability = modeled["availability"]
    if availability == "available_modeled_serial":
        expected_mode = "modeled_host_kv_serial"
        expected_ledger = (
            "logical_request_blocks_with_single_rank_capacity")
        expected_physical_per_logical = 1
        expected_multi_rank = "unsupported"
    elif availability == "available_modeled_rank_local":
        expected_mode = "modeled_host_kv_rank_local_lockstep"
        expected_ledger = (
            "logical_request_blocks_with_lockstep_per_rank_capacity")
        expected_physical_per_logical = 2
        expected_multi_rank = "validated_colocated_qwen3_tp2_lockstep"
    else:
        raise ValueError(f"{label} availability is invalid")
    if (modeled["mode"] != expected_mode
            or modeled["ledger_semantics"] != expected_ledger):
        raise ValueError(f"{label} mode or ledger semantics are invalid")
    scope = _mapping(modeled["scope"], f"{label} scope")
    _exact_fields(
        scope,
        {
            "vllm_native_offload", "ramulator", "host_dram_timing",
            "host_dram_queue", "host_link_utilization", "prefetch",
            "compute_overlap", "multi_pool", "multi_rank",
        },
        f"{label} scope",
    )
    if scope["host_link_utilization"] != (
            "available_in_system_resource_manifest_v5_"
            "host_kv_link_utilization_per_rank"):
        raise ValueError(
            f"{label} does not bind SystemResourceManifest v5 Host-KV "
            "utilization")
    if scope["multi_rank"] != expected_multi_rank:
        raise ValueError(f"{label} multi-rank scope is inconsistent")

    transfers = _sequence(modeled["transfers"], f"{label} transfers")
    if len(transfers) != expected_transfer_count or not transfers:
        raise ValueError(
            f"{label} transfers disagree with execution artifact counts")
    physical_count = 0
    for index, raw_transfer in enumerate(transfers):
        transfer_label = f"{label} transfer {index}"
        transfer = _mapping(raw_transfer, transfer_label)
        _exact_fields(
            transfer,
            {
                "step_id", "ready_event", "intent", "logical_completion",
                "physical_transfers",
            },
            transfer_label,
        )
        intent = _mapping(transfer["intent"], f"{transfer_label} intent")
        _required_fields(
            intent,
            {
                "ready_event", "action", "bytes", "source_tier",
                "destination_tier", "tier",
            },
            f"{transfer_label} intent",
        )
        logical = _mapping(
            transfer["logical_completion"],
            f"{transfer_label} logical completion",
        )
        _exact_fields(
            logical,
            {
                "ready_event", "action", "bytes_per_rank", "source_tier",
                "destination_tier", "link", "physical_transfer_count",
                "relative_interval", "absolute_interval",
            },
            f"{transfer_label} logical completion",
        )
        bytes_per_rank = _nonnegative_integer(
            logical["bytes_per_rank"],
            f"{transfer_label} bytes per rank",
        )
        if bytes_per_rank == 0:
            raise ValueError(f"{transfer_label} bytes per rank must be positive")
        if (transfer["ready_event"] != intent["ready_event"]
                or transfer["ready_event"] != logical["ready_event"]
                or intent["action"] not in ("evict", "load")
                or intent["action"] != logical["action"]
                or intent["bytes"] != bytes_per_rank
                or intent["source_tier"] != logical["source_tier"]
                or intent["destination_tier"]
                != logical["destination_tier"]):
            raise ValueError(
                f"{transfer_label} logical provenance is inconsistent")
        physical_transfers = _sequence(
            transfer["physical_transfers"],
            f"{transfer_label} physical transfers",
        )
        if len(physical_transfers) != expected_physical_per_logical:
            raise ValueError(
                f"{transfer_label} physical transfer count is invalid")
        if logical["physical_transfer_count"] != len(physical_transfers):
            raise ValueError(
                f"{transfer_label} logical physical count is inconsistent")

        step_id = transfer["step_id"]
        if step_id not in steps_by_id:
            raise ValueError(f"{transfer_label} references an unknown step")
        step = steps_by_id[step_id]
        step_origin_s = _nonnegative_number(
            schedule_entries_by_id[step_id]["start_time_s"],
            f"{transfer_label} step origin",
        )
        nodes = _sequence(
            _mapping(
                step["workload_graph"],
                f"{transfer_label} workload graph",
            )["nodes"],
            f"{transfer_label} workload nodes",
        )
        nodes_by_id = {}
        for raw_node in nodes:
            node = _mapping(raw_node, f"{transfer_label} workload node")
            node_id = node["node_id"]
            if node_id in nodes_by_id:
                raise ValueError(
                    f"{transfer_label} has duplicate workload node IDs")
            nodes_by_id[node_id] = node
        calls = _sequence(
            _mapping(
                step["kernel_plan"], f"{transfer_label} kernel plan",
            )["calls"],
            f"{transfer_label} kernel calls",
        )
        calls_by_id = {}
        for raw_call in calls:
            call = _mapping(raw_call, f"{transfer_label} kernel call")
            call_id = call["call_id"]
            if call_id in calls_by_id:
                raise ValueError(
                    f"{transfer_label} has duplicate kernel call IDs")
            calls_by_id[call_id] = call
        predictions = _sequence(
            step["predictions"], f"{transfer_label} predictions")
        predictions_by_id = {}
        for raw_prediction in predictions:
            prediction = _mapping(
                raw_prediction, f"{transfer_label} prediction")
            call_id = prediction["call_id"]
            if call_id in predictions_by_id:
                raise ValueError(
                    f"{transfer_label} has duplicate prediction IDs")
            predictions_by_id[call_id] = prediction
        timeline = _mapping(
            step["execution_timeline"], f"{transfer_label} timeline")
        if timeline["origin_time_s"] != step_origin_s:
            raise ValueError(
                f"{transfer_label} timeline origin disagrees with schedule")
        intervals = _sequence(
            timeline["intervals"], f"{transfer_label} intervals")
        intervals_by_id = {}
        for raw_interval in intervals:
            interval = _mapping(
                raw_interval, f"{transfer_label} interval")
            call_id = interval["call_id"]
            if call_id in intervals_by_id:
                raise ValueError(
                    f"{transfer_label} has duplicate interval IDs")
            intervals_by_id[call_id] = interval

        ranks = []
        pools = []
        relative_intervals = []
        absolute_intervals = []
        for physical_index, raw_physical in enumerate(physical_transfers):
            physical_label = (
                f"{transfer_label} physical transfer {physical_index}")
            physical = _mapping(raw_physical, physical_label)
            _exact_fields(
                physical,
                {
                    "resource_id", "rank", "pool_name", "node_id",
                    "call_id", "relative_interval", "absolute_interval",
                    "prediction_total_s", "completion",
                },
                physical_label,
            )
            completion = _mapping(
                physical["completion"], f"{physical_label} completion")
            _exact_fields(
                completion,
                {
                    "node_id", "call_id", "pool_name", "rank", "bytes",
                    "link", "start_time_s", "end_time_s",
                },
                f"{physical_label} completion",
            )
            rank = _nonnegative_integer(
                physical["rank"], f"{physical_label} rank")
            ranks.append(rank)
            pool_name = physical["pool_name"]
            if (not isinstance(pool_name, str) or not pool_name
                    or not isinstance(logical["link"], str)
                    or not logical["link"]):
                raise ValueError(
                    f"{physical_label} pool/link must be non-empty")
            pools.append(pool_name)
            expected_resource_id = (
                f"pool:{pool_name}:rank:{rank}:"
                f"{logical['link']}")
            if (physical["resource_id"] != expected_resource_id
                    or completion["rank"] != rank
                    or completion["pool_name"] != physical["pool_name"]
                    or completion["node_id"] != physical["node_id"]
                    or completion["call_id"] != physical["call_id"]
                    or completion["bytes"] != bytes_per_rank
                    or completion["link"] != logical["link"]):
                raise ValueError(
                    f"{physical_label} completion/resource provenance is "
                    "inconsistent")
            node_id = physical["node_id"]
            call_id = physical["call_id"]
            if (node_id not in nodes_by_id
                    or call_id not in calls_by_id
                    or call_id not in predictions_by_id
                    or call_id not in intervals_by_id):
                raise ValueError(
                    f"{physical_label} lacks a canonical node/call/"
                    "prediction/interval")
            node = nodes_by_id[node_id]
            node_attributes = _mapping(
                node["attributes"], f"{physical_label} node attributes")
            call = calls_by_id[call_id]
            spec = _mapping(call["kernel_spec"], f"{physical_label} spec")
            transfer_config = _mapping(
                spec["transfer"], f"{physical_label} transfer config")
            prediction_record = predictions_by_id[call_id]
            interval_record = intervals_by_id[call_id]
            if (node["kind"] != "kv_offload"
                    or node["pool_name"] != pool_name
                    or node["rank"] != rank
                    or node_attributes["ready_event"]
                    != transfer["ready_event"]
                    or node_attributes["bytes"] != bytes_per_rank
                    or node_attributes["direction"] != logical["action"]
                    or node_attributes["source_tier"]
                    != logical["source_tier"]
                    or node_attributes["destination_tier"]
                    != logical["destination_tier"]
                    or node_attributes["link"] != logical["link"]
                    or call["pool_name"] != pool_name
                    or call["rank"] != rank
                    or call["resource"] != logical["link"]
                    or tuple(_sequence(
                        call["source_node_ids"],
                        f"{physical_label} source node IDs")) != (node_id,)
                    or spec["kind"] != "KV_OFFLOAD"
                    or transfer_config["algorithm"] != "dma"
                    or transfer_config["bytes"] != bytes_per_rank
                    or transfer_config["direction"] != logical["action"]
                    or transfer_config["link"] != logical["link"]
                    or transfer_config["tier"] != intent["tier"]
                    or prediction_record["t_total_s"]
                    != physical["prediction_total_s"]
                    or interval_record["pool_name"] != pool_name
                    or interval_record["rank"] != rank
                    or interval_record["resource"] != logical["link"]
                    or tuple(_sequence(
                        interval_record["reserved_resource_ids"],
                        f"{physical_label} reserved resources"))
                    != (physical["resource_id"],)):
                raise ValueError(
                    f"{physical_label} disagrees with canonical execution "
                    "components")
            relative = _mapping(
                physical["relative_interval"],
                f"{physical_label} relative interval",
            )
            absolute = _mapping(
                physical["absolute_interval"],
                f"{physical_label} absolute interval",
            )
            _exact_fields(
                relative, {"start_time_s", "end_time_s"},
                f"{physical_label} relative interval")
            _exact_fields(
                absolute, {"start_time_s", "end_time_s"},
                f"{physical_label} absolute interval")
            relative_start = _nonnegative_number(
                relative["start_time_s"], f"{physical_label} relative start")
            relative_end = _nonnegative_number(
                relative["end_time_s"], f"{physical_label} relative end")
            absolute_start = _nonnegative_number(
                absolute["start_time_s"], f"{physical_label} absolute start")
            absolute_end = _nonnegative_number(
                absolute["end_time_s"], f"{physical_label} absolute end")
            prediction = _positive_number(
                physical["prediction_total_s"],
                f"{physical_label} prediction")
            if (relative_end <= relative_start
                    or not math.isclose(
                        relative_end - relative_start,
                        prediction,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        absolute_end - absolute_start,
                        prediction,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        absolute_start - relative_start,
                        step_origin_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        absolute_end - relative_end,
                        step_origin_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or completion["start_time_s"] != relative_start
                    or completion["end_time_s"] != relative_end
                    or interval_record["start_time_s"] != relative_start
                    or interval_record["end_time_s"] != relative_end
                    or interval_record["absolute_start_time_s"]
                    != absolute_start
                    or interval_record["absolute_end_time_s"]
                    != absolute_end):
                raise ValueError(
                    f"{physical_label} interval/prediction is inconsistent")
            relative_intervals.append((relative_start, relative_end))
            absolute_intervals.append((absolute_start, absolute_end))
        if (ranks != sorted(ranks)
                or len(ranks) != len(set(ranks))
                or len(set(pools)) != 1):
            raise ValueError(
                f"{transfer_label} physical ranks/pools are inconsistent")
        if (expected_physical_per_logical == 1 and len(ranks) != 1
                or expected_physical_per_logical == 2
                and tuple(ranks) != (0, 1)):
            raise ValueError(
                f"{transfer_label} physical ranks are outside validated "
                "scope")

        logical_relative = _mapping(
            logical["relative_interval"],
            f"{transfer_label} logical relative interval",
        )
        logical_absolute = _mapping(
            logical["absolute_interval"],
            f"{transfer_label} logical absolute interval",
        )
        _exact_fields(
            logical_relative, {"start_time_s", "end_time_s"},
            f"{transfer_label} logical relative interval")
        _exact_fields(
            logical_absolute, {"start_time_s", "end_time_s"},
            f"{transfer_label} logical absolute interval")
        if (logical_relative["start_time_s"]
                != min(item[0] for item in relative_intervals)
                or logical_relative["end_time_s"]
                != max(item[1] for item in relative_intervals)
                or logical_absolute["start_time_s"]
                != min(item[0] for item in absolute_intervals)
                or logical_absolute["end_time_s"]
                != max(item[1] for item in absolute_intervals)
                or not math.isclose(
                    logical_absolute["start_time_s"]
                    - logical_relative["start_time_s"],
                    step_origin_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    logical_absolute["end_time_s"]
                    - logical_relative["end_time_s"],
                    step_origin_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{transfer_label} logical interval does not cover its "
                "physical transfers")
        physical_count += len(physical_transfers)

    declared_physical_count = _nonnegative_integer(
        modeled["physical_transfer_count"],
        f"{label} physical transfer count",
    )
    if declared_physical_count != physical_count:
        raise ValueError(f"{label} physical transfer count is inconsistent")
    _mapping(modeled["final_tier_state"], f"{label} final tier state")
