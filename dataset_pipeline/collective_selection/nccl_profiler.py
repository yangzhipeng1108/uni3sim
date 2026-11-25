"""Validate NCCL profiler traces and extract non-duration selection facts."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from dataset_pipeline.collective_selection import (
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


PROFILE_NAME_PATTERN = re.compile(r"nccl-profile_[0-9]+_([0-9]+)\.json")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ALGORITHMS = {
    "COLLNET_CHAIN": "collnet_chain",
    "COLLNET_DIRECT": "collnet_direct",
    "NVLS": "nvls",
    "NVLS_TREE": "nvls_tree",
    "PAT": "pat",
    "RING": "ring",
    "TREE": "tree",
}
PROTOCOLS = {"LL": "ll", "LL128": "ll128", "SIMPLE": "simple"}
OPERATION_NAMES = {"all_gather": "AllGather", "all_reduce": "AllReduce"}
PROFILE_DATATYPES = {
    "all_gather": "ncclInt8",
    "all_reduce": "ncclBfloat16",
}
METADATA_KEYS = {
    "communication_library",
    "cuda_runtime_version",
    "device",
    "hardware_id",
    "library_version",
    "local_rank",
    "observation_method",
    "observations",
    "profiler_plugin_sha256",
    "profiler_plugin_source_commit",
    "rank",
    "rank_mapping",
    "run_id",
    "runner_sha256",
    "schema_version",
    "source_commit",
    "topology",
    "torch_version",
}
COLLECTIVE_ARGUMENT_KEYS = {
    "Algorithm",
    "CommHash",
    "Count",
    "Datatype",
    "Protocol",
    "Rank",
    "SeqNum",
    "nChannels",
}


def _profile_files(run_dir: Path, world_size: int) -> dict[int, Path]:
    paths = tuple(sorted(run_dir.glob("nccl-profile_*.json")))
    if len(paths) != world_size:
        raise ValueError(
            f"run directory must contain {world_size} NCCL profiles, got {paths}"
        )
    profiles: dict[int, Path] = {}
    communicator_ids: set[str] = set()
    for path in paths:
        match = PROFILE_NAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected NCCL profile filename: {path.name}")
        rank = int(match.group(1))
        if rank in profiles:
            raise ValueError(f"duplicate NCCL profile for rank {rank}")
        profiles[rank] = path
        communicator_ids.add(path.name.rsplit("_", 1)[0])
    expected_ranks = set(range(world_size))
    if set(profiles) != expected_ranks:
        raise ValueError(
            f"NCCL profiles must cover ranks {sorted(expected_ranks)}, got {sorted(profiles)}"
        )
    if len(communicator_ids) != 1:
        raise ValueError("NCCL profile filenames do not share one communicator ID")
    return profiles


def _validate_metadata(
    path: Path, expected_rank: int, expected_world_size: int
) -> dict[str, Any]:
    payload = require_mapping(load_collective_json(path), str(path))
    require_exact_keys(payload, METADATA_KEYS, str(path))
    if payload["schema_version"] != 1:
        raise ValueError(f"{path} schema_version must be 1")
    if payload["hardware_id"] != "nvidia_a800":
        raise ValueError(f"{path} hardware_id must be nvidia_a800")
    if payload["communication_library"] != "NCCL":
        raise ValueError(f"{path} communication_library must be NCCL")
    if payload["observation_method"] != "NCCL profiler plugin API v4":
        raise ValueError(f"{path} has an unsupported observation method")
    if payload["rank"] != expected_rank or payload["local_rank"] != expected_rank:
        raise ValueError(f"{path} rank identity does not match its filename")
    for key in (
        "source_commit",
        "profiler_plugin_source_commit",
        "profiler_plugin_sha256",
        "runner_sha256",
    ):
        value = payload[key]
        expected_pattern = SHA256_PATTERN if key.endswith("sha256") else re.compile(r"[0-9a-f]{40}")
        if not isinstance(value, str) or expected_pattern.fullmatch(value) is None:
            raise ValueError(f"{path} {key} is not a full lowercase identity")
    rank_mapping = require_list(payload["rank_mapping"], f"{path} rank_mapping")
    if (
        len(rank_mapping) != expected_world_size
        or len(set(rank_mapping)) != expected_world_size
    ):
        raise ValueError(
            f"{path} rank_mapping must contain {expected_world_size} unique devices"
        )
    device = require_mapping(payload["device"], f"{path} device")
    require_keys(device, {"device_uuid", "physical_device"}, f"{path} device")
    if device["device_uuid"] != rank_mapping[expected_rank]:
        raise ValueError(f"{path} device UUID does not match rank_mapping")
    observations = require_list(payload["observations"], f"{path} observations")
    if not observations:
        raise ValueError(f"{path} observations must not be empty")
    for index, raw_observation in enumerate(observations):
        context = f"{path} observations[{index}]"
        observation = require_mapping(raw_observation, context)
        require_exact_keys(observation, OBSERVATION_KEYS, context)
        if observation["rank"] != expected_rank:
            raise ValueError(f"{context} rank does not match metadata rank")
        if observation["dtype"] != "bfloat16":
            raise ValueError(f"{context} must use bfloat16")
        if observation["operation"] not in OPERATION_NAMES:
            raise ValueError(f"{context} has an unsupported operation")
        if observation["operation"] == "all_reduce":
            expected_reduction = "sum"
            expected_in_place = True
        elif observation["operation"] == "all_gather":
            expected_reduction = None
            expected_in_place = False
        else:
            raise AssertionError("validated NCCL operation vanished")
        if observation["reduction"] != expected_reduction:
            raise ValueError(f"{context} reduction differs")
        if observation["in_place"] is not expected_in_place:
            raise ValueError(f"{context} in_place differs")
        if observation["participant_count"] != expected_world_size:
            raise ValueError(
                f"{context} participant_count must be {expected_world_size}"
            )
        require_integer(observation["message_bytes"], f"{context} message_bytes", positive=True)
        require_integer(observation["repeat"], f"{context} repeat")
        if observation["validation"] != "passed":
            raise ValueError(f"{context} workload validation did not pass")
    return payload


def _collective_begins(path: Path, expected_rank: int) -> list[dict[str, Any]]:
    events = require_list(load_collective_json(path), str(path))
    collective_events = []
    for index, raw_event in enumerate(events):
        event = require_mapping(raw_event, f"{path} event[{index}]")
        if not event and index == len(events) - 1:
            continue
        require_keys(event, {"cat"}, f"{path} event[{index}]")
        if event["cat"] == "COLL":
            collective_events.append((index, event))
    if not collective_events:
        raise ValueError(f"{path} contains no collective events")
    if len(collective_events) % 2:
        raise ValueError(f"{path} contains an odd number of collective events")
    begins = []
    for pair_index in range(0, len(collective_events), 2):
        begin_index, begin = collective_events[pair_index]
        end_index, end = collective_events[pair_index + 1]
        require_keys(begin, {"args", "id", "name", "ph"}, f"{path} event[{begin_index}]")
        require_keys(end, {"id", "name", "ph"}, f"{path} event[{end_index}]")
        if begin["ph"] != "b" or end["ph"] != "e":
            raise ValueError(f"{path} collective events are not complete begin/end pairs")
        if begin["id"] != end["id"] or begin["name"] != end["name"]:
            raise ValueError(f"{path} collective begin/end pair identity differs")
        arguments = require_mapping(begin["args"], f"{path} event[{begin_index}] args")
        require_keys(arguments, COLLECTIVE_ARGUMENT_KEYS, f"{path} event[{begin_index}] args")
        if arguments["Rank"] != expected_rank:
            raise ValueError(f"{path} contains an event for a different rank")
        require_integer(arguments["Count"], f"{path} event[{begin_index}] Count", positive=True)
        require_integer(arguments["SeqNum"], f"{path} event[{begin_index}] SeqNum")
        require_integer(arguments["CommHash"], f"{path} event[{begin_index}] CommHash", positive=True)
        require_integer(arguments["nChannels"], f"{path} event[{begin_index}] nChannels", positive=True)
        begins.append(begin)
    return begins


def _raw_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    arguments = event["args"]
    return (
        event["name"],
        arguments["SeqNum"],
        arguments["CommHash"],
        arguments["Count"],
        arguments["Datatype"],
        arguments["Algorithm"],
        arguments["Protocol"],
        arguments["nChannels"],
    )


def _normalize_target_event(
    event: dict[str, Any], observation: dict[str, Any], context: str
) -> dict[str, Any]:
    arguments = event["args"]
    expected_name = OPERATION_NAMES[observation["operation"]]
    if event["name"] != expected_name:
        raise ValueError(f"{context} operation differs from runner metadata")
    operation = observation["operation"]
    if operation == "all_gather":
        count_bytes = arguments["Count"]
    elif operation == "all_reduce":
        count_bytes = arguments["Count"] * 2
    else:
        raise ValueError(f"{context} has an unsupported operation")
    if count_bytes != observation["message_bytes"]:
        raise ValueError(f"{context} element count differs from runner metadata")
    algorithm = arguments["Algorithm"]
    protocol = arguments["Protocol"]
    if algorithm not in ALGORITHMS:
        raise ValueError(f"{context} has unknown NCCL algorithm {algorithm!r}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"{context} has unknown NCCL protocol {protocol!r}")
    return {
        "algorithm": ALGORITHMS[algorithm],
        "channels": arguments["nChannels"],
        "message_bytes": observation["message_bytes"],
        "operation": observation["operation"],
        "protocol": PROTOCOLS[protocol],
        "repeat": observation["repeat"],
    }


def _target_signature(observation: dict[str, Any]) -> tuple[str, str, int]:
    operation = observation["operation"]
    if operation == "all_gather":
        count = observation["message_bytes"]
    elif operation == "all_reduce":
        count = observation["message_bytes"] // 2
    else:
        raise ValueError(f"unsupported operation: {operation!r}")
    return OPERATION_NAMES[operation], PROFILE_DATATYPES[operation], count


def _select_target_events(
    begins: list[dict[str, Any]], observations: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    operations = {observation["operation"] for observation in observations}
    if len(operations) != 1:
        raise ValueError("one profiler run must contain exactly one target operation")
    operation = next(iter(operations))
    expected = tuple(_target_signature(observation) for observation in observations)
    candidates = tuple(
        event
        for event in begins
        if event["name"] == OPERATION_NAMES[operation]
        and event["args"]["Datatype"] == PROFILE_DATATYPES[operation]
    )
    candidate_signatures = tuple(
        (event["name"], event["args"]["Datatype"], event["args"]["Count"])
        for event in candidates
    )
    starts = tuple(
        start
        for start in range(len(candidates) - len(expected) + 1)
        if candidate_signatures[start:start + len(expected)] == expected
    )
    if len(starts) != 1:
        raise ValueError(
            "runner observation sequence does not uniquely match NCCL events: "
            f"match_count={len(starts)}"
        )
    start = starts[0]
    if start + len(expected) != len(candidates):
        raise ValueError("unexpected target-like NCCL events follow the matched workload")
    return candidates[start:]


def build_selection_observations(run_dir: Path) -> dict[str, Any]:
    """Build validated selection facts while ignoring all profiler timestamps."""
    run_dir, metadata_paths, metadata = load_run_metadata(
        run_dir,
        allowed_world_sizes=(2, 4, 8),
        allowed_description="2, 4, or 8",
        validate_metadata=_validate_metadata,
        keys=METADATA_KEYS,
        differing_fields={"device", "local_rank", "observations", "rank"},
    )
    world_size = len(metadata_paths)
    profile_paths = _profile_files(run_dir, world_size)
    begins = tuple(
        _collective_begins(profile_paths[rank], rank) for rank in range(world_size)
    )
    require_consistent_ranks(
        begins, _raw_signature, "NCCL collective events differ")
    raw_observations = rank_zero_observations(metadata)
    target_events = tuple(
        _select_target_events(rank_events, raw_observations)
        for rank_events in begins
    )
    normalized = tuple(
        _normalize_target_event(event, observation, f"target event {index}")
        for index, (event, observation) in enumerate(zip(target_events[0], raw_observations))
    )
    require_consistent_repeats(
        normalized,
        lambda observation: (
            observation["algorithm"],
            observation["protocol"],
            observation["channels"],
        ),
        "NCCL selection differs across repeats",
    )
    return {
        "schema_version": 1,
        "hardware_id": metadata[0]["hardware_id"],
        "run_id": metadata[0]["run_id"],
        "source_commit": metadata[0]["source_commit"],
        "runner_sha256": metadata[0]["runner_sha256"],
        "communication_library": metadata[0]["communication_library"],
        "library_version": metadata[0]["library_version"],
        "profiler_plugin_source_commit": metadata[0]["profiler_plugin_source_commit"],
        "profiler_plugin_sha256": metadata[0]["profiler_plugin_sha256"],
        "topology": metadata[0]["topology"],
        "rank_mapping": metadata[0]["rank_mapping"],
        "raw_files": [
            {"path": path.name, "sha256": file_sha256(path)}
            for path in (
                *metadata_paths,
                *(profile_paths[rank] for rank in range(world_size)),
            )
        ],
        "observations": list(normalized),
    }


def write_selection_observations(run_dir: Path) -> Path:
    return write_observations(run_dir, build_selection_observations(run_dir))
