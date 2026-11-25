"""Validate MCCL mcTracer captures and extract non-duration selection facts."""

from __future__ import annotations

import math
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
    require_keys,
    require_list,
    require_mapping,
    write_observations,
)


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
KERNEL_PATTERN = re.compile(
    r"^mcclKernel_(AllReduce|AllGather)_(RING|TREE|FC2)"
    r"(?:_(LL|SIMPLE))?_Sum_(mccl_bfloat16|int8_t)\((.*)\)$"
)
ALL_GATHER_FC2_PATTERN = re.compile(
    r"^mcclkernel_AllGather_FC2(?:_(LL))?_int8_t\(fcinfo\)$"
)
SETUP_PATTERN = re.compile(
    r"^mcclKernel_AllGather_RING_LL_Sum_int8_t"
    r"\(mcclDevComm\*, unsigned long, mcclWork\*\)$"
)
BARRIER_PATTERN = re.compile(
    r"^mcclKernel_AllReduce_(?:RING|TREE)_(?:LL|SIMPLE)_Sum_uint8_t"
    r"\(mcclDevComm\*, unsigned long, mcclWork\*\)$"
)
OPERATION_NAMES = {"all_gather": "AllGather", "all_reduce": "AllReduce"}
KERNEL_DTYPES = {"all_gather": "int8_t", "all_reduce": "mccl_bfloat16"}
ALGORITHMS = {"FC2": "fc2", "RING": "ring", "TREE": "tree"}
PROTOCOLS = {"LL": "ll", "SIMPLE": "simple"}
METADATA_KEYS_V1 = {
    "communication_library",
    "device",
    "hardware_id",
    "library_sha256",
    "library_version",
    "local_rank",
    "observation_method",
    "observations",
    "rank",
    "rank_mapping",
    "run_id",
    "runner_sha256",
    "schema_version",
    "source_commit",
    "source_identity",
    "topology",
    "torch_version",
}
METADATA_KEYS_V2 = METADATA_KEYS_V1 | {"profiler_attach_wait_seconds"}
SOURCE_IDENTITY_KEYS = {"kind", "manifest_sha256"}


def _validate_metadata(
    path: Path, expected_rank: int, expected_world_size: int
) -> dict[str, Any]:
    payload = require_mapping(load_collective_json(path), str(path))
    if "schema_version" not in payload:
        raise ValueError(f"{path} is missing schema_version")
    schema_version = payload["schema_version"]
    if schema_version == 1:
        require_exact_keys(payload, METADATA_KEYS_V1, str(path))
    elif schema_version == 2:
        require_exact_keys(payload, METADATA_KEYS_V2, str(path))
    else:
        raise ValueError(f"{path} schema_version must be 1 or 2")
    if payload["hardware_id"] != "metax_c500":
        raise ValueError(f"{path} hardware_id must be metax_c500")
    if payload["communication_library"] != "MCCL":
        raise ValueError(f"{path} communication_library must be MCCL")
    if payload["observation_method"] != "mcTracer kernel symbol":
        raise ValueError(f"{path} has an unsupported observation method")
    if payload["topology"] != "single_node_metaxlink":
        raise ValueError(f"{path} has an unsupported topology")
    if payload["rank"] != expected_rank or payload["local_rank"] != expected_rank:
        raise ValueError(f"{path} rank identity does not match its filename")
    for key, pattern in (
        ("source_commit", COMMIT_PATTERN),
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
    if schema_version == 1:
        normalized = dict(payload)
        normalized["profiler_attach_wait_seconds"] = None
    else:
        attach_wait = payload["profiler_attach_wait_seconds"]
        if (
            isinstance(attach_wait, bool)
            or not isinstance(attach_wait, (int, float))
            or not math.isfinite(attach_wait)
            or attach_wait < 0
        ):
            raise ValueError(f"{path} profiler attach wait must be nonnegative")
        normalized = payload
    observations = require_list(payload["observations"], f"{path} observations")
    if not observations:
        raise ValueError(f"{path} observations must not be empty")
    for index, raw_observation in enumerate(observations):
        context = f"{path} observations[{index}]"
        observation = require_mapping(raw_observation, context)
        require_exact_keys(observation, OBSERVATION_KEYS, context)
        operation = observation["operation"]
        if operation not in OPERATION_NAMES:
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
    return normalized


def _trace_files(run_dir: Path, world_size: int) -> tuple[Path, ...]:
    paths = tuple(sorted(run_dir.glob("tracer_out-*.json")))
    nested = tuple(sorted(run_dir.glob("tracer_out_*/tracer_out-*.json")))
    if paths and nested:
        raise ValueError("run directory mixes flat and nested mcTracer layouts")
    selected = paths if paths else nested
    if len(selected) != world_size:
        raise ValueError(
            f"run directory must contain {world_size} mcTracer files, got {selected}"
        )
    return selected


def _kernel_events(path: Path) -> tuple[dict[str, Any], ...]:
    payload = require_mapping(load_collective_json(path), str(path))
    require_exact_keys(payload, {"traceEvents"}, str(path))
    events = require_list(payload["traceEvents"], f"{path} traceEvents")
    kernels = []
    for index, raw_event in enumerate(events):
        event = require_mapping(raw_event, f"{path} traceEvents[{index}]")
        if "name" not in event or "ph" not in event:
            continue
        name = event["name"]
        if not isinstance(name, str) or not name.lower().startswith("mcclkernel_"):
            continue
        if event["ph"] != "X":
            raise ValueError(f"{path} MCCL event must be a complete event")
        require_keys(event, {"args"}, f"{path} MCCL event")
        arguments = require_mapping(event["args"], f"{path} MCCL event args")
        require_keys(
            arguments, {"block", "device_id", "grid", "name"},
            f"{path} MCCL event args",
        )
        require_integer(arguments["device_id"], f"{path} MCCL device_id")
        if not isinstance(arguments["name"], str):
            raise TypeError(f"{path} MCCL kernel symbol must be a string")
        for field in ("block", "grid"):
            geometry = require_mapping(
                arguments[field], f"{path} MCCL {field}"
            )
            require_exact_keys(geometry, {"x", "y", "z"}, f"{path} MCCL {field}")
            for dimension in ("x", "y", "z"):
                require_integer(
                    geometry[dimension],
                    f"{path} MCCL {field}.{dimension}",
                    positive=True,
                )
        kernels.append(event)
    if not kernels:
        raise ValueError(f"{path} contains no MCCL kernel events")
    return tuple(kernels)


def _trace_rank(path: Path, events: tuple[dict[str, Any], ...]) -> int:
    device_ids = {event["args"]["device_id"] for event in events}
    if len(device_ids) != 1:
        raise ValueError(f"{path} MCCL events span multiple devices: {device_ids}")
    return next(iter(device_ids))


def _target_events(
    events: tuple[dict[str, Any], ...],
    observations: tuple[dict[str, Any], ...],
    context: str,
) -> tuple[dict[str, Any], ...]:
    barrier_indices = tuple(
        index
        for index, event in enumerate(events)
        if BARRIER_PATTERN.fullmatch(event["name"]) is not None
    )
    if len(barrier_indices) != 1:
        raise ValueError(
            f"{context} must contain one uint8 barrier kernel, got {barrier_indices}"
        )
    barrier_index = barrier_indices[0]
    for event in events[:barrier_index]:
        if SETUP_PATTERN.fullmatch(event["name"]) is None:
            raise ValueError(f"{context} contains an unknown pre-barrier MCCL kernel")
    targets = events[barrier_index + 1:]
    if len(targets) != len(observations):
        raise ValueError(
            f"{context} target count differs: expected={len(observations)}, "
            f"actual={len(targets)}"
        )
    return targets


def _geometry(event: dict[str, Any], field: str) -> list[int]:
    geometry = event["args"][field]
    return [geometry["x"], geometry["y"], geometry["z"]]


def _normalize_target_event(
    event: dict[str, Any], observation: dict[str, Any], context: str
) -> dict[str, Any]:
    match = KERNEL_PATTERN.fullmatch(event["name"])
    if match is not None:
        operation_name, algorithm_token, protocol_token, dtype_token, arguments = (
            match.groups()
        )
    else:
        all_gather_fc2_match = ALL_GATHER_FC2_PATTERN.fullmatch(event["name"])
        if all_gather_fc2_match is None:
            raise ValueError(
                f"{context} has an unknown MCCL kernel {event['name']!r}"
            )
        operation_name = "AllGather"
        algorithm_token = "FC2"
        protocol_token = all_gather_fc2_match.group(1)
        dtype_token = "int8_t"
        arguments = "fcinfo"
    operation = observation["operation"]
    if operation_name != OPERATION_NAMES[operation]:
        raise ValueError(f"{context} operation differs from runner metadata")
    if dtype_token != KERNEL_DTYPES[operation]:
        raise ValueError(f"{context} datatype differs from runner metadata")
    if algorithm_token == "FC2":
        if arguments != "fcinfo":
            raise ValueError(f"{context} FC2 kernel arguments differ")
    elif arguments != "mcclDevComm*, unsigned long, mcclWork*":
        raise ValueError(f"{context} MCCL kernel arguments differ")
    if protocol_token is None and algorithm_token != "FC2":
        raise ValueError(f"{context} omits protocol for a non-FC2 algorithm")
    protocol = None if protocol_token is None else PROTOCOLS[protocol_token]
    return {
        "algorithm": ALGORITHMS[algorithm_token],
        "channels": None,
        "kernel_block": _geometry(event, "block"),
        "kernel_grid": _geometry(event, "grid"),
        "kernel_name": event["name"],
        "kernel_symbol": event["args"]["name"],
        "message_bytes": observation["message_bytes"],
        "operation": operation,
        "protocol": protocol,
        "repeat": observation["repeat"],
    }


def _selection_signature(observation: dict[str, Any]) -> tuple[Any, ...]:
    return (
        observation["algorithm"],
        observation["protocol"],
        tuple(observation["kernel_grid"]),
        tuple(observation["kernel_block"]),
        observation["kernel_name"],
        observation["kernel_symbol"],
    )


def build_selection_observations(run_dir: Path) -> dict[str, Any]:
    """Build validated MCCL selection facts without consuming timestamps or duration."""
    run_dir, metadata_paths, metadata = load_run_metadata(
        run_dir,
        allowed_world_sizes=(2, 4),
        allowed_description="2 or 4",
        validate_metadata=_validate_metadata,
        keys=METADATA_KEYS_V2,
        differing_fields={"device", "local_rank", "observations", "rank"},
    )
    world_size = len(metadata_paths)
    trace_paths = _trace_files(run_dir, world_size)
    traces_by_rank: dict[int, tuple[Path, tuple[dict[str, Any], ...]]] = {}
    for path in trace_paths:
        events = _kernel_events(path)
        rank = _trace_rank(path, events)
        if rank not in range(world_size):
            raise ValueError(f"{path} has out-of-range trace rank {rank}")
        if rank in traces_by_rank:
            raise ValueError(f"duplicate mcTracer file for rank {rank}")
        traces_by_rank[rank] = (path, events)
    if set(traces_by_rank) != set(range(world_size)):
        raise ValueError("mcTracer files do not cover every rank")
    raw_observations = rank_zero_observations(metadata)
    normalized_by_rank = []
    for rank in range(world_size):
        path, events = traces_by_rank[rank]
        targets = _target_events(events, raw_observations, str(path))
        normalized_by_rank.append(tuple(
            _normalize_target_event(
                event, observation, f"rank {rank} target event {index}"
            )
            for index, (event, observation) in enumerate(
                zip(targets, raw_observations)
            )
        ))
    require_consistent_ranks(
        normalized_by_rank, _selection_signature, "MCCL selection differs")
    require_consistent_repeats(
        normalized_by_rank[0],
        _selection_signature,
        "MCCL selection differs across repeats",
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
        "profiler_attach_wait_seconds": metadata[0][
            "profiler_attach_wait_seconds"
        ],
        "topology": metadata[0]["topology"],
        "rank_mapping": metadata[0]["rank_mapping"],
        "channel_count_observation": "unavailable_in_mctracer",
        "raw_files": [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(path),
            }
            for path in (
                *metadata_paths,
                *(traces_by_rank[rank][0] for rank in range(world_size)),
            )
        ],
        "observations": list(normalized_by_rank[0]),
    }


def write_selection_observations(run_dir: Path) -> Path:
    return write_observations(run_dir, build_selection_observations(run_dir))
