"""Run environment resolution for compact ShareGPT evaluation: run
directories, repository inputs, Host timing profiles, and source
identity."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

from src.serving.host_timing import (
    load_host_launch_profile,
    load_scheduler_host_profile,
)

from evaluation.serving.metrics.compact_case_contract import (
    _GIT_COMMIT,
)


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


# Hardware whose Host timing profiles exist once per captured vLLM version:
# every profile family is bound by the case's recorded runtime version.
_HOST_PROFILE_VARIANTS = {
    "nvidia_h200": {"0.30.0": "nvidia_h200_vllm030", "0.11.0": "nvidia_h200_vllm011"},
}
_HOST_PROFILE_DIRECTORIES = {
    "host_launch": ("host_launch_profiles", ""),
    "scheduler_host": ("scheduler_host_profiles", ""),
    "host_request_path": ("host_request_path_profiles", ""),
    "host_runtime_prepare": ("host_runtime_prepare_profiles", "_eager"),
}


def create_run_directory(
    repository_root: str | Path,
    run_id: str,
) -> Path:
    """Create one fresh run below the repository experiment_logs tree."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    root = Path(repository_root).resolve()
    output_root = root / "experiment_logs" / "serving" / "sharegpt_phase1"
    destination = (output_root / run_id).resolve()
    if output_root.resolve() not in destination.parents:
        raise ValueError("phase-1 output must remain under experiment_logs")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def resolve_repository_input(
    repository_root: str | Path,
    value: str | Path,
    label: str,
) -> Path:
    """Resolve one CLI input while preventing repository-root escape."""

    root = Path(repository_root).resolve()
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} must remain inside the repository")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def resolve_host_launch_profile_paths(
    repository_root: str | Path,
    explicit_paths: Sequence[str],
    selected_hardware_ids: set[str] | None,
) -> Mapping[str, str] | None:
    """Resolve explicit profiles or discover canonical profiles by hardware."""

    root = Path(repository_root).resolve()
    values = tuple(explicit_paths)
    use_defaults = not values
    if use_defaults:
        profile_dir = (
            root / "configs/host_timing/host_launch_profiles")
        values = tuple(str(path) for path in sorted(profile_dir.glob("*.json")))
    profiles = {}
    for value in values:
        path = resolve_repository_input(root, value, "Host launch profile")
        profile = load_host_launch_profile(path)
        if use_defaults and profile.hardware_id in _HOST_PROFILE_VARIANTS:
            continue
        if profile.hardware_id in profiles:
            raise ValueError(
                f"duplicate Host launch profile for {profile.hardware_id}")
        if (use_defaults and selected_hardware_ids is not None
                and profile.hardware_id not in selected_hardware_ids):
            continue
        profiles[profile.hardware_id] = str(path)
    return profiles if profiles else None


def resolve_scheduler_host_profile_paths(
    repository_root: str | Path,
    explicit_paths: Sequence[str],
    selected_hardware_ids: set[str] | None,
) -> Mapping[str, str] | None:
    """Resolve explicit profiles or discover canonical profiles by hardware."""

    root = Path(repository_root).resolve()
    values = tuple(explicit_paths)
    use_defaults = not values
    if use_defaults:
        profile_dir = (
            root / "configs/host_timing/scheduler_host_profiles")
        values = tuple(str(path) for path in sorted(profile_dir.glob("*.json")))
    profiles = {}
    for value in values:
        path = resolve_repository_input(root, value, "Scheduler Host profile")
        profile = load_scheduler_host_profile(path)
        if use_defaults and profile.hardware_id in _HOST_PROFILE_VARIANTS:
            continue
        if profile.hardware_id in profiles:
            raise ValueError(
                f"duplicate Scheduler Host profile for {profile.hardware_id}")
        if (use_defaults and selected_hardware_ids is not None
                and profile.hardware_id not in selected_hardware_ids):
            continue
        profiles[profile.hardware_id] = str(path)
    return profiles if profiles else None


def host_profile_variant_hardware() -> frozenset[str]:
    """Hardware whose Host profiles are bound per captured vLLM version."""

    return frozenset(_HOST_PROFILE_VARIANTS)


def case_host_profile_paths(
    repository_root: str | Path,
    hardware_id: str,
    runtime_version: str,
) -> Mapping[str, Path]:
    """Return the four Host profiles of one versioned hardware case."""

    if hardware_id not in _HOST_PROFILE_VARIANTS:
        raise ValueError(f"{hardware_id} Host profiles are not versioned")
    variants = _HOST_PROFILE_VARIANTS[hardware_id]
    if runtime_version not in variants:
        raise ValueError(
            f"no {hardware_id} Host profiles for vLLM {runtime_version!r}")
    root = Path(repository_root).resolve()
    return {
        family: (
            root / "configs/host_timing" / directory
            / f"{variants[runtime_version]}{suffix}.json"
        ).resolve(strict=True)
        for family, (directory, suffix) in _HOST_PROFILE_DIRECTORIES.items()
    }


def repository_source_commit(repository_root: str | Path) -> str:
    """Return the immutable source identity used for this evaluation run."""

    root = Path(repository_root)
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("git rev-parse HEAD did not return a full commit SHA")
    return commit
