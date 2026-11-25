"""Validate HCCL INFO logs and extract non-duration selection facts."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from dataset_pipeline.collective_selection import (
    DEVICE_KEYS,
    OBSERVATION_KEYS,
    file_sha256,
    load_collective_json,
    load_run_metadata,
    rank_zero_observations,
    require_consistent_ranks,
    require_consistent_repeats,
    require_exact_keys,
    require_integer,
    require_list,
    require_mapping,
    write_observations,
)


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SELECT_PATTERN = re.compile(
    r"HCCL\((\d+),[^)]*\):.*\[(\d+)\]\[SelectAlgfor910B\] "
    r"(AllReduce|AllGather) SelectAlgfor910B is algName "
    r"\[([A-Za-z0-9_]+)\](.*)$"
)
COUNT_PATTERN = re.compile(
    r"HCCL\((\d+),[^)]*\):.*\[(\d+)\]NSLB-HCCL "
    r"algValue:\[(\d+)\],nslbAlg\[(\d+)\], count:\[(\d+)\]\.$"
)
MODE_SUFFIX_PATTERN = re.compile(r", current mode is \[(\d+)\]")
OPERATION_NAMES = {"AllGather": "all_gather", "AllReduce": "all_reduce"}
EXECUTORS = {
    "AllGatherMeshOpbaseExecutor": "mesh",
    "AllReduceMeshOpbaseLoopExecutor": "mesh",
    "AllReduceMeshSmallCountExecutor": "mesh",
}
METADATA_KEYS = {
    "cann_version_file_sha256",
    "communication_library",
    "device",
    "driver_version",
    "driver_version_file_sha256",
    "hardware_id",
    "library_sha256",
    "library_version",
    "local_rank",
    "log_environment",
    "observation_method",
    "observations",
    "process_id",
    "rank",
    "rank_mapping",
    "run_id",
    "runner_sha256",
    "schema_version",
    "source_commit",
    "source_identity",
    "topology",
    "torch_npu_version",
    "torch_version",
}
SOURCE_IDENTITY_KEYS = {"kind", "manifest_sha256"}
LOG_ENVIRONMENT_KEYS = {
    "ASCEND_GLOBAL_LOG_LEVEL",
    "ASCEND_PROCESS_LOG_PATH",
    "ASCEND_SLOG_PRINT_TO_STDOUT",
}


def _validate_metadata(
    path: Path, expected_rank: int, expected_world_size: int
) -> dict[str, Any]:
    payload = require_mapping(load_collective_json(path), str(path))
    require_exact_keys(payload, METADATA_KEYS, str(path))
    if payload["schema_version"] != 1:
        raise ValueError(f"{path} schema_version must be 1")
    if payload["hardware_id"] != "ascend_910b":
        raise ValueError(f"{path} hardware_id must be ascend_910b")
    if payload["communication_library"] != "HCCL":
        raise ValueError(f"{path} communication_library must be HCCL")
    if payload["observation_method"] != "HCCL INFO selection log":
        raise ValueError(f"{path} has an unsupported observation method")
    if payload["topology"] != "single_node_full_hccs":
        raise ValueError(f"{path} has an unsupported topology")
    if payload["rank"] != expected_rank or payload["local_rank"] != expected_rank:
        raise ValueError(f"{path} rank identity does not match its filename")
    for key, pattern in (
        ("source_commit", COMMIT_PATTERN),
        ("cann_version_file_sha256", SHA256_PATTERN),
        ("driver_version_file_sha256", SHA256_PATTERN),
        ("library_sha256", SHA256_PATTERN),
        ("runner_sha256", SHA256_PATTERN),
    ):
        value = payload[key]
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise ValueError(f"{path} {key} is not a full lowercase identity")
    source_identity = require_mapping(
        payload["source_identity"], f"{path} source_identity"
    )
    require_exact_keys(
        source_identity, SOURCE_IDENTITY_KEYS, f"{path} source_identity"
    )
    if source_identity["kind"] != "audited_source_snapshot":
        raise ValueError(f"{path} must use an audited source snapshot")
    manifest_sha256 = source_identity["manifest_sha256"]
    if (
        not isinstance(manifest_sha256, str)
        or SHA256_PATTERN.fullmatch(manifest_sha256) is None
    ):
        raise ValueError(f"{path} source snapshot manifest identity is invalid")
    device = require_mapping(payload["device"], f"{path} device")
    require_exact_keys(device, DEVICE_KEYS, f"{path} device")
    physical_device = require_integer(
        device["physical_device"], f"{path} device physical_device"
    )
    rank_mapping = require_list(payload["rank_mapping"], f"{path} rank_mapping")
    if (
        len(rank_mapping) != expected_world_size
        or len(set(rank_mapping)) != expected_world_size
    ):
        raise ValueError(
            f"{path} rank_mapping must contain {expected_world_size} unique devices"
        )
    if physical_device != rank_mapping[expected_rank]:
        raise ValueError(f"{path} device does not match rank_mapping")
    require_integer(payload["process_id"], f"{path} process_id", positive=True)
    log_environment = require_mapping(
        payload["log_environment"], f"{path} log_environment"
    )
    require_exact_keys(
        log_environment, LOG_ENVIRONMENT_KEYS, f"{path} log_environment"
    )
    if log_environment["ASCEND_GLOBAL_LOG_LEVEL"] != "1":
        raise ValueError(f"{path} did not enable INFO logs")
    if log_environment["ASCEND_SLOG_PRINT_TO_STDOUT"] != "0":
        raise ValueError(f"{path} did not isolate process logs")
    log_path = Path(log_environment["ASCEND_PROCESS_LOG_PATH"])
    if log_path.name != "hccl_logs" or log_path.parent.name != payload["run_id"]:
        raise ValueError(f"{path} process log path is not bound to its run")
    observations = require_list(payload["observations"], f"{path} observations")
    if not observations:
        raise ValueError(f"{path} observations must not be empty")
    for index, raw_observation in enumerate(observations):
        context = f"{path} observations[{index}]"
        observation = require_mapping(raw_observation, context)
        require_exact_keys(observation, OBSERVATION_KEYS, context)
        operation = observation["operation"]
        if operation not in ("all_gather", "all_reduce"):
            raise ValueError(f"{context} has an unsupported operation")
        if observation["rank"] != expected_rank:
            raise ValueError(f"{context} rank does not match metadata rank")
        if observation["participant_count"] != expected_world_size:
            raise ValueError(
                f"{context} participant_count must be {expected_world_size}"
            )
        if observation["dtype"] != "bfloat16":
            raise ValueError(f"{context} must use bfloat16")
        if operation == "all_reduce":
            expected_reduction = "sum"
            expected_in_place = True
        elif operation == "all_gather":
            expected_reduction = None
            expected_in_place = False
        else:
            raise ValueError(f"{context} has an unsupported operation")
        if observation["reduction"] != expected_reduction:
            raise ValueError(f"{context} reduction differs")
        if observation["in_place"] is not expected_in_place:
            raise ValueError(f"{context} in_place differs")
        require_integer(
            observation["message_bytes"], f"{context} message_bytes", positive=True
        )
        require_integer(observation["repeat"], f"{context} repeat")
        if observation["validation"] != "passed":
            raise ValueError(f"{context} workload validation did not pass")
    return payload


def _rank_log_files(run_dir: Path, process_id: int) -> tuple[Path, ...]:
    log_dir = run_dir / "hccl_logs" / "debug" / "plog"
    paths = tuple(sorted(log_dir.glob(f"plog-{process_id}_*.log")))
    if not paths:
        raise ValueError(f"no HCCL INFO log found for process {process_id}")
    return paths


def _selection_events(
    paths: tuple[Path, ...], expected_process_id: int
) -> tuple[dict[str, Any], ...]:
    pending: dict[int, tuple[str, str, int | None]] = {}
    selections = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"cannot read HCCL log {path}: {error}") from error
        for line in lines:
            select_match = SELECT_PATTERN.search(line)
            if select_match is not None:
                process_id = int(select_match.group(1))
                thread_id = int(select_match.group(2))
                if process_id != expected_process_id:
                    raise ValueError(f"{path} contains a selection for another process")
                if thread_id in pending:
                    raise ValueError(f"{path} has an unpaired HCCL selection")
                operation = OPERATION_NAMES[select_match.group(3)]
                executor = select_match.group(4)
                suffix = select_match.group(5)
                if operation == "all_reduce":
                    if suffix != ".":
                        raise ValueError(f"{path} has an unknown AllReduce log suffix")
                    mode = None
                elif operation == "all_gather":
                    mode_match = MODE_SUFFIX_PATTERN.fullmatch(suffix)
                    if mode_match is None:
                        raise ValueError(f"{path} has an unknown AllGather log suffix")
                    mode = int(mode_match.group(1))
                else:
                    raise ValueError(
                        f"{path} has an unsupported operation {operation!r}"
                    )
                if executor not in EXECUTORS:
                    raise ValueError(f"{path} has unknown HCCL executor {executor!r}")
                pending[thread_id] = (operation, executor, mode)
                continue
            count_match = COUNT_PATTERN.search(line)
            if count_match is None:
                continue
            process_id = int(count_match.group(1))
            thread_id = int(count_match.group(2))
            if process_id != expected_process_id:
                raise ValueError(f"{path} contains a count for another process")
            if thread_id not in pending:
                raise ValueError(f"{path} has an HCCL count without a selection")
            operation, executor, mode = pending.pop(thread_id)
            count_bytes = int(count_match.group(5))
            if count_bytes <= 0:
                raise ValueError(f"{path} has a nonpositive HCCL count")
            selections.append({
                "algorithm_value": int(count_match.group(3)),
                "count_bytes": count_bytes,
                "mode": mode,
                "nslb_algorithm": int(count_match.group(4)),
                "operation": operation,
                "vendor_executor": executor,
            })
    if pending:
        raise ValueError(f"HCCL selections lack count records: {pending}")
    if not selections:
        raise ValueError("HCCL logs contain no complete selection records")
    return tuple(selections)


def _expected_count(observation: dict[str, Any]) -> int:
    if observation["operation"] == "all_reduce":
        return observation["message_bytes"]
    return observation["message_bytes"] * observation["participant_count"]


def _select_targets(
    events: tuple[dict[str, Any], ...], observations: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    operations = {observation["operation"] for observation in observations}
    if len(operations) != 1:
        raise ValueError("one HCCL run must contain exactly one target operation")
    operation = next(iter(operations))
    candidates = tuple(event for event in events if event["operation"] == operation)
    expected = tuple(_expected_count(observation) for observation in observations)
    counts = tuple(event["count_bytes"] for event in candidates)
    starts = tuple(
        start
        for start in range(len(candidates) - len(expected) + 1)
        if counts[start:start + len(expected)] == expected
    )
    if len(starts) != 1:
        raise ValueError(
            "runner observation sequence does not uniquely match HCCL logs: "
            f"match_count={len(starts)}"
        )
    start = starts[0]
    if start + len(expected) != len(candidates):
        raise ValueError("unexpected target-like HCCL selections follow the workload")
    return candidates[start:]


def _normalize_target(
    event: dict[str, Any], observation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "algorithm": EXECUTORS[event["vendor_executor"]],
        "algorithm_value": event["algorithm_value"],
        "channels": None,
        "message_bytes": observation["message_bytes"],
        "mode": event["mode"],
        "nslb_algorithm": event["nslb_algorithm"],
        "operation": observation["operation"],
        "protocol": None,
        "repeat": observation["repeat"],
        "vendor_executor": event["vendor_executor"],
    }


def _selection_signature(observation: dict[str, Any]) -> tuple[Any, ...]:
    return (
        observation["algorithm"],
        observation["algorithm_value"],
        observation["mode"],
        observation["nslb_algorithm"],
        observation["protocol"],
        observation["vendor_executor"],
    )


def build_selection_observations(run_dir: Path) -> dict[str, Any]:
    """Build validated HCCL selection facts without consuming log timestamps."""
    run_dir, metadata_paths, metadata = load_run_metadata(
        run_dir,
        allowed_world_sizes=(2, 4, 8),
        allowed_description="2, 4, or 8",
        validate_metadata=_validate_metadata,
        keys=METADATA_KEYS,
        differing_fields={
            "device", "local_rank", "observations", "process_id", "rank",
        },
    )
    raw_observations = rank_zero_observations(metadata)
    log_paths_by_rank = []
    normalized_by_rank = []
    for rank, payload in enumerate(metadata):
        process_id = payload["process_id"]
        log_paths = _rank_log_files(run_dir, process_id)
        log_paths_by_rank.append(log_paths)
        events = _selection_events(log_paths, process_id)
        targets = _select_targets(events, raw_observations)
        normalized_by_rank.append(tuple(
            _normalize_target(event, observation)
            for event, observation in zip(targets, raw_observations)
        ))
    require_consistent_ranks(
        normalized_by_rank, _selection_signature, "HCCL selection differs")
    require_consistent_repeats(
        normalized_by_rank[0],
        _selection_signature,
        "HCCL selection differs across repeats",
    )
    return {
        "schema_version": 1,
        "hardware_id": metadata[0]["hardware_id"],
        "run_id": metadata[0]["run_id"],
        "source_commit": metadata[0]["source_commit"],
        "source_identity": metadata[0]["source_identity"],
        "runner_sha256": metadata[0]["runner_sha256"],
        "communication_library": metadata[0]["communication_library"],
        "library_version": metadata[0]["library_version"],
        "library_sha256": metadata[0]["library_sha256"],
        "driver_version": metadata[0]["driver_version"],
        "topology": metadata[0]["topology"],
        "rank_mapping": metadata[0]["rank_mapping"],
        "channel_count_observation": "unavailable_in_hccl_info_log",
        "protocol_observation": "unavailable_in_hccl_info_log",
        "raw_files": [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(path),
            }
            for path in (
                *metadata_paths,
                *(path for paths in log_paths_by_rank for path in paths),
            )
        ],
        "observations": list(normalized_by_rank[0]),
    }


def write_selection_observations(run_dir: Path) -> Path:
    return write_observations(run_dir, build_selection_observations(run_dir))
