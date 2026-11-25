"""Audit normalized collective observations against production selectors."""

import json
from pathlib import Path
from typing import Mapping, Sequence

from src.kernel.launch_config.select_launch_config import (
    select_collective_launch,
)


from dataset_pipeline.collective_selection import require_mapping as _mapping


def _operation_contract(operation: str) -> tuple[str, str | None, bool]:
    if operation == "all_reduce":
        return "bfloat16", "sum", True
    if operation == "all_gather":
        return "bfloat16", None, False
    raise ValueError(f"unsupported collective audit operation: {operation!r}")


def _vendor_executor(
    hardware_id: str,
    observation: Mapping[str, object],
) -> str | None:
    if hardware_id == "ascend_910b":
        executor = observation["vendor_executor"]
        if not isinstance(executor, str) or not executor:
            raise ValueError("HCCL observation vendor_executor must be non-empty")
        return executor
    if hardware_id in {"nvidia_a800", "metax_c500"}:
        return None
    raise ValueError(
        f"unsupported collective audit hardware_id: {hardware_id!r}")


def audit_selection_observations(
    manifest_paths: Sequence[Path],
) -> dict[str, int]:
    """Require every normalized observation to match one selector result."""

    paths = tuple(Path(path) for path in manifest_paths)
    if not paths:
        raise ValueError("collective rule audit requires at least one manifest")
    if len(set(paths)) != len(paths):
        raise ValueError("collective rule audit manifest paths must be unique")

    observation_count = 0
    unique_inputs = set()
    for path in paths:
        payload = _mapping(
            json.loads(path.read_text(encoding="utf-8")), path)
        hardware_id = payload["hardware_id"]
        communication_library = payload["communication_library"]
        library_version = payload["library_version"]
        topology = payload["topology"]
        rank_mapping = payload["rank_mapping"]
        observations = payload["observations"]
        for field_name, value in (
                ("hardware_id", hardware_id),
                ("communication_library", communication_library),
                ("library_version", library_version),
                ("topology", topology)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{path} {field_name} must be non-empty")
        if not isinstance(rank_mapping, list) or len(rank_mapping) <= 1:
            raise ValueError(f"{path} rank_mapping must contain multiple ranks")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"{path} observations must be a non-empty list")
        participant_count = len(rank_mapping)

        for index, raw_observation in enumerate(observations):
            context = f"{path}:observations[{index}]"
            observation = _mapping(raw_observation, context)
            operation = observation["operation"]
            message_bytes = observation["message_bytes"]
            algorithm = observation["algorithm"]
            protocol = observation["protocol"]
            if not isinstance(operation, str):
                raise TypeError(f"{context} operation must be a string")
            if (isinstance(message_bytes, bool)
                    or not isinstance(message_bytes, int)):
                raise TypeError(f"{context} message_bytes must be an integer")
            if not isinstance(algorithm, str) or not algorithm:
                raise ValueError(f"{context} algorithm must be non-empty")
            if protocol is not None and (
                    not isinstance(protocol, str) or not protocol):
                raise ValueError(
                    f"{context} protocol must be non-empty or null")
            dtype, reduction, in_place = _operation_contract(operation)
            executor = _vendor_executor(hardware_id, observation)
            selection = select_collective_launch(
                hardware_id,
                communication_library=communication_library,
                library_version=library_version,
                operation=operation,
                message_bytes=message_bytes,
                participant_count=participant_count,
                dtype=dtype,
                reduction=reduction,
                in_place=in_place,
                topology=topology,
            )
            observed_selection = (algorithm, protocol, executor)
            selected = (
                selection.vendor_algorithm,
                selection.vendor_protocol,
                selection.vendor_executor,
            )
            if selected != observed_selection:
                raise ValueError(
                    f"{context} selector differs: observed="
                    f"{observed_selection!r}, selected={selected!r}")
            unique_inputs.add((
                hardware_id,
                communication_library,
                library_version,
                topology,
                participant_count,
                operation,
                message_bytes,
            ))
            observation_count += 1
    return {
        "manifest_count": len(paths),
        "observation_count": observation_count,
        "unique_input_count": len(unique_inputs),
    }
