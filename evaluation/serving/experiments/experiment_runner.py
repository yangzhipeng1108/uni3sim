#!/usr/bin/env python3
"""Select, resume, and aggregate Serving experiment cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import signal
import site
import subprocess
import sys
import sysconfig
import time
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3]

from evaluation.serving.experiments.experiment_plan import (
    git_code_identity,
    materialize_cpu_simulation_plan,
    simulation_explicit_input_files,
    load_model_tokenizer,
    materialize_experiment_plan,
    validate_vllm_experiment_environment,
    validate_vllm_scheduler_environment,
)
from evaluation.serving.experiments.experiment_stages import (
    execute_experiment_stages,
    validate_production_trace_outputs,
)
from evaluation.serving.experiments.experiment_manifest import (
    ExperimentManifest,
    ManifestValidationError,
    ProductionTraceSweepCell,
    SimulationCell,
    load_experiment_manifest,
)

from evaluation.json_shape import write_json_atomic as _write_json_atomic, write_json_file as _write_json
from src.serving.deployment.deployment_config import worker_role_phase


def _raise_directory_scan_error(error: OSError) -> None:
    raise error


def site_package_content_identity(roots: list[str]) -> dict[str, Any]:
    """Fingerprint Python library directories, including unregistered additions."""
    digest = hashlib.sha256()
    count = 0
    resolved_roots = sorted({Path(path).resolve() for path in roots})
    for root in resolved_roots:
        if not root.is_dir():
            return {"status": "unavailable", "path": str(root)}
        if any(parent in resolved_roots for parent in root.parents):
            continue
        for parent, directories, filenames in os.walk(root, onerror=_raise_directory_scan_error):
            directories.sort()
            for name in directories:
                path = Path(parent) / name
                if path.is_symlink():
                    return {"status": "unavailable", "path": str(path)}
            for name in sorted(filenames):
                path = Path(parent) / name
                if not path.is_file():
                    return {"status": "unavailable", "path": str(path)}
                file_digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        file_digest.update(chunk)
                digest.update(json.dumps([str(path), file_digest.hexdigest()], ensure_ascii=True,
                                         separators=(",", ":")).encode("utf-8"))
                digest.update(b"\n")
                count += 1
    return {"status": "observed", "file_count": count, "sha256": digest.hexdigest()}


def python_native_library_paths(prefixes: list[str]) -> list[str] | None:
    """List shared libraries beside the selected Python installations."""
    paths = []
    for root in sorted({(Path(prefix) / "lib").resolve() for prefix in prefixes}):
        if not root.is_dir():
            return None
        with os.scandir(root) as entries:
            for entry in entries:
                if ".so" not in entry.name:
                    continue
                if not entry.is_file():
                    return None
                paths.append(entry.path)
    return sorted(paths)


def system_library_content_identity() -> dict[str, Any]:
    """Fingerprint registered libraries and other libraries in their directories."""
    linker = shutil.which("ldconfig")
    if linker is None:
        return {"status": "unavailable"}
    result = subprocess.run([linker, "-p"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"status": "unavailable"}
    paths = {line.rsplit("=>", 1)[1].strip() for line in result.stdout.splitlines() if "=>" in line}
    if not paths:
        return {"status": "unavailable"}
    registered = {Path(name).resolve() for name in paths}
    links = []
    for root in {Path(name).parent.resolve() for name in paths}:
        for parent, directories, filenames in os.walk(root, onerror=_raise_directory_scan_error):
            for name in directories:
                path = Path(parent) / name
                if path.is_symlink():
                    target = path.resolve()
                    if target != root and root not in target.parents:
                        return {"status": "unavailable"}
                    links.append((str(path), str(target)))
            for name in filenames:
                if ".so" not in name:
                    continue
                path = Path(parent) / name
                if path.is_file() and path.resolve() not in registered:
                    paths.add(str(path.resolve()))
    python_libraries = python_native_library_paths([sys.prefix, sys.base_prefix])
    if python_libraries is None:
        return {"status": "unavailable"}
    paths.update(python_libraries)
    paths.add("/etc/ld.so.cache")
    digest = hashlib.sha256()
    for link in sorted(links):
        digest.update(json.dumps(link, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    for name in sorted(paths):
        path = Path(name)
        if not path.is_file():
            return {"status": "unavailable"}
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                file_digest.update(chunk)
        digest.update(json.dumps([name, file_digest.hexdigest()], ensure_ascii=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return {"status": "observed", "file_count": len(paths), "sha256": digest.hexdigest()}


def editable_sources_observation() -> list[dict[str, str]]:
    """Record editable sources; existing sources require an explicit identity."""
    observations = []
    for distribution in metadata.distributions():
        text = distribution.read_text("direct_url.json")
        if text is None:
            continue
        data = json.loads(text)
        if "dir_info" not in data or "editable" not in data["dir_info"] or not data["dir_info"]["editable"]:
            continue
        url = urlsplit(data["url"])
        if url.scheme != "file" or url.netloc not in ("", "localhost"):
            observations.append({"name": distribution.metadata["Name"], "status": "unavailable"})
            continue
        path = Path(unquote(url.path))
        if not path.is_absolute():
            observations.append({"name": distribution.metadata["Name"], "status": "unavailable"})
            continue
        try:
            path.stat()
        except FileNotFoundError:
            status = "missing"
        else:
            status = "present_unfingerprinted"
        observations.append({"name": distribution.metadata["Name"], "path": str(path),
                             "status": status})
    return sorted(observations, key=lambda value: (value["name"], value["status"]))


def system_configuration_identity(paths: list[Path]) -> list[dict[str, Any]]:
    """Observe system configuration files read by the supported CPU scenarios."""
    from evaluation.serving.simulation.simulation_artifacts import file_provenance

    observations = []
    for path in sorted(paths):
        try:
            path.stat()
        except FileNotFoundError:
            observations.append({"path": str(path), "status": "missing"})
            continue
        if not path.is_file():
            observations.append({"path": str(path), "status": "unavailable"})
            continue
        observations.append({"status": "observed", "provenance": file_provenance(path)})
    return observations


def python_search_path_observation(paths: list[str]) -> list[dict[str, str]]:
    """Record whether each Python search path exists, including missing ZIP paths."""
    observations = []
    for raw in paths:
        path = Path(raw or ".").resolve()
        try:
            path.stat()
        except FileNotFoundError:
            status = "missing"
        else:
            status = "present"
        observations.append({"path": str(path), "status": status})
    return observations


def require_reusable_python_search_paths(environment: dict[str, Any], project_root: Path,
                                         source_roots: list[Path]) -> None:
    """Reject existing import paths outside recorded libraries and source owners."""
    project = project_root.resolve()
    roots = [*(project / name for name in ("src", "evaluation", "scripts")),
             *(path.resolve() for path in source_roots),
             *(Path(path).resolve() for path in environment["python_library_roots"])]
    for entry in environment["python_search_paths"]:
        if entry["status"] == "missing":
            continue
        if entry["status"] != "present":
            raise ManifestValidationError("cannot reuse with an unknown Python search path state")
        path = Path(entry["path"])
        if path != project and not any(path == root or root in path.parents for root in roots):
            raise ManifestValidationError(
                f"cannot reuse with an unfingerprinted Python search path: {path}; "
                "use --resume --rerun-success --cpu-run")


def simulation_python_environment() -> dict[str, Any]:
    """Describe the launcher; this is not a full dependency content identity."""
    from evaluation.serving.simulation.simulation_artifacts import file_provenance

    versions = {}
    installation_records = {}
    for name in ("numpy", "pydantic", "torch", "vllm"):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
            installation_records[name] = {"status": "not_installed"}
        else:
            versions[name] = distribution.version
            record = distribution.read_text("RECORD")
            if record is None:
                installation_records[name] = {"status": "unavailable"}
            else:
                installation_records[name] = {
                    "status": "observed",
                    "sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
                }
    system = os.uname()
    if any(not isinstance(path, str) for path in sys.path):
        raise ManifestValidationError("launcher Python search path must contain only strings")
    site_roots = list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        site_roots.append(site.getusersitepackages())
    startup_sites = []
    for root in sorted({Path(path).resolve() for path in site_roots}):
        entries = []
        if root.is_dir():
            for candidate in sorted(root.glob("*.pth")):
                if candidate.is_file():
                    entries.append({"status": "file", "provenance": file_provenance(candidate)})
                else:
                    entries.append({"status": "unavailable", "path": str(candidate)})
        startup_sites.append({"path": str(root), "is_directory": root.is_dir(), "pth_files": entries})
    startup_modules = {}
    for name in ("sitecustomize", "usercustomize"):
        if name not in sys.modules:
            startup_modules[name] = {"status": "not_loaded"}
        else:
            module = sys.modules[name]
            if module is None or not hasattr(module, "__file__") or not isinstance(module.__file__, str):
                startup_modules[name] = {"status": "loaded_without_file"}
            elif Path(module.__file__).is_file():
                startup_modules[name] = {"status": "file", "provenance": file_provenance(module.__file__)}
            else:
                startup_modules[name] = {"status": "loaded_unavailable", "path": module.__file__}
    return {"executable": sys.executable,
            "executable_provenance": file_provenance(sys.executable), "version": sys.version,
            "launcher_module_search_path": list(sys.path),
            "python_search_paths": python_search_path_observation(list(sys.path)),
            "python_library_roots": sorted({str(Path(path).resolve())
                                             for path in [*site_roots, sysconfig.get_path("stdlib")]}),
            "launcher_user_site_enabled": site.ENABLE_USER_SITE,
            "launcher_site_startup_observation": {"site_disabled": bool(sys.flags.no_site),
                "sites": startup_sites, "customize_modules": startup_modules,
                "scope": "launcher_only_not_child_or_complete_dependency_identity"},
            "prefix": sys.prefix, "platform": {"system": system.sysname,
                "release": system.release, "machine": system.machine},
            "package_versions": versions, "launcher_package_installation_records": installation_records,
            "site_package_contents": site_package_content_identity([*site_roots, sysconfig.get_path("stdlib")]),
            "system_library_contents": system_library_content_identity(),
            "system_configurations": system_configuration_identity(
                [Path("/etc/gitconfig"), Path("/etc/ssl/openssl.cnf")]),
            "editable_sources": editable_sources_observation(),
            "identity_scope": "launcher_executable_search_path_and_versions_not_dependency_content_identity"}


def simulation_native_runtime_identity(project_root: Path) -> dict[str, Any]:
    """Observe the native library loaded by the predictor, not all dependencies."""
    from evaluation.serving.simulation.simulation_artifacts import file_provenance

    return {"lru_sim": file_provenance(project_root / "build" / "lru_sim.so")}


def simulation_project_bytecode_identity(project_root: Path) -> dict[str, Any]:
    """Observe ignored bytecode in project code directories, separately from Git."""
    from evaluation.serving.simulation.simulation_artifacts import file_provenance

    files = []
    for name in ("src", "evaluation", "scripts"):
        root = project_root / name
        if root.is_dir():
            for path in sorted(root.rglob("*.pyc")):
                if path.is_file():
                    files.append({"relative_path": str(path.relative_to(project_root)),
                                  "provenance": file_provenance(path)})
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"file_count": len(files), "sha256": hashlib.sha256(payload).hexdigest()}


def simulation_subprocess_environment() -> dict[str, str]:
    """Prepare the environment actually passed to the simulation process."""
    environment = dict(os.environ)
    # The orchestration call ID changes per invocation and is not a
    # simulation input; remove it from the child as well as its identity.
    if "CODEX_TOOL_CALL_ID" in environment:
        del environment["CODEX_TOOL_CALL_ID"]
    environment["PYTHONHASHSEED"] = "0"
    return environment


def simulation_process_environment_digest() -> str:
    """Bind the actual child environment without recording names or values."""
    serialized = json.dumps(sorted(simulation_subprocess_environment().items()), ensure_ascii=True,
                            separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def simulation_sdk_python_environment(options: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Observe the explicitly selected SDK Python using the existing environment recorder."""
    if "vllm-sdk-python" not in options:
        return {"status": "not_requested"}
    executable = options["vllm-sdk-python"]
    if isinstance(executable, list) and len(executable) == 1:
        executable = executable[0]
    if not isinstance(executable, str) or not executable:
        raise ManifestValidationError("vllm-sdk-python must name an executable")
    environment = simulation_subprocess_environment()
    if "vllm-source-root" in options:
        source_root = options["vllm-source-root"]
        if isinstance(source_root, list) and len(source_root) == 1:
            source_root = source_root[0]
        if not isinstance(source_root, str) or not source_root:
            raise ManifestValidationError("vllm-source-root must name a directory")
        environment["PYTHONPATH"] = source_root + os.pathsep + str(project_root)
    environment.update({"CUDA_VISIBLE_DEVICES": "", "VLLM_TARGET_DEVICE": "cpu", "VLLM_PLUGINS": "",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1", "VLLM_NO_USAGE_STATS": "1"})
    script = ("import json; from evaluation.serving.experiments.experiment_runner "
              "import simulation_python_environment; print(json.dumps(simulation_python_environment()))")
    completed = subprocess.run([executable, "-c", script], cwd=project_root, env=environment,
                               capture_output=True, text=True, check=True)
    return {"status": "observed", "environment": json.loads(completed.stdout)}


class _CpuSimulationTerminated(BaseException):
    def __init__(self, signum: int):
        self.signum = signum


def _raise_cpu_termination(signum: int, frame: Any) -> None:
    raise _CpuSimulationTerminated(signum)


def _cpu_process_identity() -> dict[str, Any]:
    pid = os.getpid()
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat.rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid_namespace": os.readlink("/proc/self/ns/pid")}


def _require_stopped_cpu_attempt(record: dict[str, Any]) -> None:
    """Allow retry only when the recorded owner and its process group are absent."""
    if "runner_process" not in record or "child_process_group" not in record:
        raise ManifestValidationError("running attempt lacks process identity; cannot establish it stopped")
    identity = record["runner_process"]
    group = record["child_process_group"]
    if (not isinstance(identity, dict) or set(identity) != {"pid", "start_ticks", "boot_id", "pid_namespace"}
            or type(identity["pid"]) is not int or identity["pid"] <= 1
            or type(identity["start_ticks"]) is not int or identity["start_ticks"] < 0
            or type(group) is not int or group <= 1):
        raise ManifestValidationError("running attempt has invalid process identity")
    current = _cpu_process_identity()
    if identity["boot_id"] != current["boot_id"] or identity["pid_namespace"] != current["pid_namespace"]:
        raise ManifestValidationError("running attempt belongs to another boot or PID namespace")
    try:
        text = Path(f"/proc/{identity['pid']}/stat").read_text()
    except FileNotFoundError:
        text = None
    if text is not None and int(text.rsplit(")", 1)[1].split()[19]) == identity["start_ticks"]:
        raise ManifestValidationError("running attempt owner is still alive")
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise ManifestValidationError("running attempt process group cannot be inspected") from error
    raise ManifestValidationError("running attempt process group is still alive")


def _execute_cpu_process(command: list[str], *, cwd: Path, env: dict[str, str],
                         stdout: Any, stderr: Any, on_started: Any) -> subprocess.CompletedProcess:
    """Own the simulation process group so cancellation reaps its workers."""
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout,
                               stderr=stderr, start_new_session=True)
    try:
        on_started(process.pid)
        returncode = process.wait()
    except BaseException:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    return subprocess.CompletedProcess(command, returncode)


def require_cpu_worktree_identity(identity: dict[str, Any]) -> None:
    """Reject declared-only snapshots whose checked-out bytes are not verified."""
    if identity["worktree_state"] not in (
            "clean_reproducible_by_commit", "dirty_not_reproducible_by_commit_alone"):
        raise ManifestValidationError(
            "CPU experiment requires a verifiable Git worktree; a declared "
            "SOURCE_COMMIT alone does not bind snapshot contents")


def render_simulation_settings(result: dict[str, Any], result_path: Path) -> str:
    """Render a compact view of observed simulation fields, with a source link.

    Different backends expose different runtime fields. Missing fields are
    explicitly unknown; in particular they are not inferred from CLI defaults.
    """
    def value(mapping: dict[str, Any], name: str) -> str:
        if name not in mapping:
            return "unknown (not reported)"
        if mapping[name] is None:
            return "unknown or not applicable (reported null)"
        return json.dumps(mapping[name], ensure_ascii=False, sort_keys=True)

    model = result["model_config_provenance"]
    plan = result["deployment_plan"]
    runtime_inputs = result["execution_provenance"]["runtime_inputs"]
    lines = [
        "# Simulation settings", "",
        "Execution type: CPU simulation; not a hardware collection.", "",
        f"Complete machine-readable result: `{result_path}`", "",
        "## Model and prediction", "",
    ]
    for label, mapping, key in (
        ("Model architecture", model, "architecture"),
        ("Model identity", model, "model_id"),
        ("Model revision", model, "revision"),
        ("Model source URL", model, "source_url"),
        ("Model scope", model, "scope"),
        ("Model configuration SHA256", model, "config_sha256"),
        ("Weight precision", model, "weight_dtype"),
        ("Compute precision", model, "compute_dtype"),
        ("Prediction method", result, "prediction_method"),
        ("Launch policy", result, "launch_policy"),
        ("Timing scope", result, "serving_timing_mode"),
        ("Performance timing status", result, "performance_timing_status"),
        ("Execution mode", result, "execution_mode"),
        ("Graph execution", runtime_inputs, "graph_execution"),
        ("Attention KV layout", runtime_inputs, "attention_kv_layout"),
        ("Attention KV distribution", runtime_inputs, "attention_kv_distribution"),
        ("Model execution source", runtime_inputs, "model_execution_version"),
        ("Symbolic forward source", runtime_inputs, "symbolic_forward_source"),
        ("Conditional model sources", runtime_inputs, "conditional_model_sources"),
    ):
        lines.append(f"- {label}: {value(mapping, key)}")
    lines.append("- KV precision: unknown (not reported as an effective runtime value)")
    provenance = result["execution_provenance"]
    if "hardware_configs" in provenance:
        hardware_configs = provenance["hardware_configs"]
        for role in ("prefill", "decode", "additional_colocated"):
            lines.append(f"- Hardware config {role}: {value(hardware_configs, role)}")
    else:
        lines.append("- Hardware config provenance: unknown (not reported)")
    if "launch_inputs" in provenance and "launch_data" in provenance["launch_inputs"]:
        for hardware_id, source in provenance["launch_inputs"]["launch_data"].items():
            lines.append(f"- Launch source {json.dumps(hardware_id)}: path={value(source, 'path')}, "
                         f"tree_sha256={value(source, 'tree_sha256')}, file_count={value(source, 'file_count')}")
        lines.append("- Launch source fingerprints describe checkout file bytes; LFS payload "
                     "materialization and runtime use are not established by this fingerprint alone.")
    else:
        lines.append("- Launch source identity: unknown (not reported)")
    lines.extend(("", "## Deployment and Scheduler", ""))
    lines.append(f"- Deployment mode: {value(plan, 'mode')}")
    lines.append(f"- Routes and links: {value(plan, 'routes')}")
    for pool in plan["worker_pools"]:
        name = pool["name"]
        lines.append(f"- Pool {json.dumps(name)}: role={value(pool, 'role')}, "
                     f"hardware={value(pool, 'hardware_id')}, logical ranks={value(pool, 'ranks')}, "
                     f"parallelism={value(pool, 'parallelism')}, "
                     f"collectives={value(pool, 'collectives')}, "
                     f"pipeline links={value(pool, 'pipeline_p2p')}")
    for name, runtime in result["scheduler_runtime"].items():
        lines.append(f"- Scheduler {json.dumps(name)}:")
        for key in ("version", "source_commit", "source_identity", "package_python_manifest_sha256",
                    "platform", "device_guard",
                    "gpu_worker_processes_started", "weights_loaded", "model_executed",
                    "effective_dtype", "effective_tensor_parallel_size", "effective_pipeline_parallel_size",
                    "effective_expert_parallel", "effective_async_scheduling", "scheduler_policy",
                    "max_model_len", "max_num_seqs", "max_num_batched_tokens", "kv_block_size_tokens",
                    "max_num_partial_prefills", "max_long_partial_prefills", "long_prefill_token_threshold",
                    "requested_gpu_memory_utilization", "effective_gpu_memory_utilization",
                    "requested_chunked_prefill", "requested_prefix_caching",
                    "effective_max_num_batched_tokens", "effective_max_num_scheduled_tokens",
                    "effective_cache_block_size", "effective_chunked_prefill", "effective_prefix_caching",
                    "available_memory_by_rank", "capacity_source", "capacity_input_source", "cache_specs_source",
                    "normal_capacity_accepted",
                    "model_feedback_source", "mtp_scenario"):
            lines.append(f"  - {key}: {value(runtime, key)}")
        if "native_allocation" in runtime and isinstance(runtime["native_allocation"], dict):
            lines.append("  - native_allocator_package_python_manifest_sha256: "
                         + value(runtime["native_allocation"], "package_python_manifest_sha256"))
    lines.extend(("", "## Request and execution", ""))
    for label, mapping, key in (
        ("Request source", result, "request_source"),
        ("Workload artifact hash", result, "workload_hash"),
        ("Request stream and assumptions", result, "workload_assumption_binding"),
        ("Arrival process", result, "arrival_process"),
        ("Routing and token assumptions", runtime_inputs, "expert_routing"),
        ("KV lifecycle replay", result, "kv_lifecycle_replay"),
        ("KV handoff mode", result, "kv_handoff_mode"),
        ("MTP scenario", result, "mtp_scenario"),
        ("Execution status", result, "main_chain_status"),
        ("Simulation wall time (s)", result, "simulation_wall_time_s"),
        ("Execution wall breakdown (s)", result, "evaluation_wall_time_breakdown_s"),
        ("Execution parallelism", result, "execution_parallelism"),
        ("Scheduler progress", result, "scheduler_progress"),
    ):
        if key == "scheduler_progress" and key in mapping:
            lines.append(f"- {label}: {value(mapping[key], 'summary')}")
        else:
            lines.append(f"- {label}: {value(mapping, key)}")
    lines.extend(("", "## Capacity", ""))
    if "kv_capacity" in result:
        capacity = result["kv_capacity"]
        lines.append(f"- Source: {value(capacity, 'source')}")
        for pool in capacity["scheduler_pools"]:
            lines.append(f"- Pool {value(pool, 'pool_name')}: "
                         f"source={value(pool, 'source_mode')}, "
                         f"block tokens={value(pool, 'kv_block_size_tokens')}, "
                         f"block count={value(pool, 'num_kv_blocks')}, "
                         f"rank budgets={value(pool, 'rank_budgets')}")
    else:
        lines.append("- Capacity: unknown (not reported)")
    if "kv_lifecycle_replay" not in result or result["kv_lifecycle_replay"] is None:
        lines.append("- Host KV capacity and offload enablement/strategy: unknown (no lifecycle evidence reported)")
        lines.append("- HBM/Host hit ratios, initial cache state and replay rounding: unknown (not reported)")
    else:
        lines.append("- Host capacity, tier policy, hit ratios, initial state and rounding: "
                     "see `kv_lifecycle_replay` in the result; unreported values are unknown")
    lines.extend(("", "## Predictions", ""))
    metrics = result["metrics"]
    if "execution_artifact" in result:
        lines.append(f"- Execution artifact schema version: {value(result['execution_artifact'], 'schema_version')}")
    else:
        lines.append("- Execution artifact schema version: unknown (not reported)")
    lines.append(f"- Persistence projection: {value(result, 'persistence')}")
    lines.append("- Saved `metrics.requests` field: " + ("present" if "requests" in metrics else "not present"))
    for key in ("request_throughput_per_s", "output_token_throughput_per_s"):
        lines.append(f"- {key}: {value(metrics, key)}")
    if "latency_distributions" in metrics:
        distributions = metrics["latency_distributions"]
        for name in ("ttft_s", "tpot_s", "latency_s"):
            if (name in distributions and distributions[name]["availability"] == "available"
                    and 50 in distributions[name]["quantiles"]["percentiles"]):
                quantiles = distributions[name]["quantiles"]
                index = quantiles["percentiles"].index(50)
                lines.append(f"- {name} p50: {quantiles['values_s'][index]}")
            else:
                lines.append(f"- {name} p50: unknown or unavailable")
    else:
        lines.append("- TTFT, TPOT and end-to-end distributions: unknown (not reported)")
    lines.extend(("", "Physical device mapping, capture framework and GPU runtime are unknown "
                  "unless separately evidenced; simulated ranks are not GPU device IDs. "
                  "A null KV lifecycle replay does not establish a measured offload state. "
                  "GPU collection/runtime and CPU Scheduler environment are separate; "
                  "unreported GPU runtime or capture facts are unknown. Capacity details, "
                  "request progress, provenance and metrics are in the result above.", ""))
    if "interactive_report" in result:
        receipt = result["interactive_report"]
        if "path" in receipt:
            lines.extend(["", "## Offline interactive report", "",
                "[Timeline HTML](" + str(receipt["path"]) + ") · [Regeneration input](" + str(receipt["data_path"]) + ")", ""])
        elif receipt["status"] == "disabled":
            lines.extend(["", "Report explicitly disabled: " + str(receipt["reason"]), ""])
    return "\n".join(lines)


def render_simulation_setting_comparison(plan: dict[str, Any], result: dict[str, Any]) -> str:
    """Compare parsed settings with runtime reports, retaining missing evidence."""
    preview = plan["normalized_preview"]
    lines = ["\n## Parsed settings and runtime reports", "",
             "The parsed preview is not evidence that a setting was executed. "
             "A difference without a runtime explanation remains unexplained.", ""]
    if "model" in plan["requested"]:
        lines.append(f"- Requested model configuration: {json.dumps(plan['requested']['model'], ensure_ascii=False)}")
    lines.append(f"- Parsed instance counts: prefill={json.dumps(preview['num_prefill_instances'])}, "
                 f"decode={json.dumps(preview['num_decode_instances'])}; "
                 "effective pools and roles are listed above")
    if "request_preview" in plan:
        lines.extend(("- Input preview (not execution observation): "
                      + json.dumps(plan["request_preview"], ensure_ascii=False, sort_keys=True), ""))

    def row(label: str, requested: object, runtime: dict[str, Any], key: str | None,
            difference_reason: str | None = None) -> None:
        requested_text = json.dumps(requested, ensure_ascii=False, sort_keys=True)
        if key is None or key not in runtime:
            lines.append(f"- {label}: parsed={requested_text}; effective=unknown (not reported)")
        else:
            actual = runtime[key]
            actual_text = json.dumps(actual, ensure_ascii=False, sort_keys=True)
            if requested == actual:
                difference = ""
            elif difference_reason is None:
                difference = "; difference reason not reported"
            else:
                difference = f"; {difference_reason}"
            lines.append(f"- {label}: parsed={requested_text}; effective={actual_text}{difference}")

    inputs = result["execution_provenance"]["runtime_inputs"]
    if "graph_execution" in inputs:
        row("Graph mode (runtime configuration)", preview["graph_mode"], inputs["graph_execution"], "configured_mode")
    else:
        lines.append("- Graph mode (runtime configuration): effective=unknown (not reported)")
    pools = {pool["name"]: pool for pool in result["deployment_plan"]["worker_pools"]}
    for name, runtime in result["scheduler_runtime"].items():
        if name not in pools:
            raise ValueError(f"Scheduler pool {name!r} is missing from the deployment plan")
        phase = worker_role_phase(pools[name]["role"])
        if phase in ("colocated", "prefill"):
            requested_tp = preview["prefill_tp_size"]
            requested_pp = preview["prefill_pp_size"]
        elif phase == "decode":
            requested_tp = preview["decode_tp_size"]
            requested_pp = None  # No separate decode PP option is exposed by this CLI.
        else:
            raise AssertionError(f"unsupported Scheduler phase {phase!r}")
        if "version" in runtime and runtime["version"] == "0.11.0":
            batch_key, block_key = "max_num_batched_tokens", "kv_block_size_tokens"
            model_limit_source, model_limit_key = runtime, "max_model_len"
            request_limit_source, request_limit_key = runtime, "max_num_seqs"
        elif "version" in runtime and runtime["version"] == "0.30.0":
            batch_key, block_key = "effective_max_num_batched_tokens", "effective_cache_block_size"
            model_limit_source = (runtime["native_allocation"]
                if "native_allocation" in runtime and isinstance(runtime["native_allocation"], dict) else {})
            model_limit_key = "max_model_len"
            request_limit_source, request_limit_key = runtime, None
        else:
            batch_key, block_key = None, None
            model_limit_source, model_limit_key = runtime, None
            request_limit_source, request_limit_key = runtime, None
        row(f"{name} max model length", preview["max_model_len"], model_limit_source, model_limit_key)
        row(f"{name} max requests", preview["max_requests"], request_limit_source, request_limit_key)
        row(f"{name} pipeline parallel size", requested_pp, runtime, "effective_pipeline_parallel_size",
            "decode PP has no independent CLI option" if phase == "decode" else None)
        for label, requested, key in (
            ("tensor parallel size", requested_tp, "effective_tensor_parallel_size"),
            ("expert parallel", preview["enable_expert_parallel"], "effective_expert_parallel"),
            ("async scheduling", preview["vllm_async_scheduling"], "effective_async_scheduling"),
            ("speculative tokens", preview["mtp_num_speculative_tokens"], "mtp_num_speculative_tokens"),
            ("chunked prefill", preview["enable_chunked_prefill"], "effective_chunked_prefill"),
            ("prefix cache", preview["enable_prefix_caching"], "effective_prefix_caching"),
            ("max batched tokens", preview["max_batched_tokens"], batch_key),
            ("KV block tokens", preview["kv_block_size_tokens"], block_key),
        ):
            row(f"{name} {label}", requested, runtime, key)
    lines.append("")
    return "\n".join(lines)


def load_existing_experiment_cell(cell_root: Path) -> dict[str, Any] | None:
    path = cell_root / "command_manifest.json"
    if not path.exists():
        return None
    return _read_json(path)


def experiment_cell_output_exists(cell_root: Path) -> bool:
    return cell_root.exists() and any(cell_root.iterdir())


def experiment_cell_status(command_manifest: dict[str, Any]) -> str:
    manifest_status = command_manifest["status"]
    if not isinstance(manifest_status, str):
        raise ManifestValidationError("command manifest status must be a string")
    status_path = Path(command_manifest["outputs"]["status"])
    if status_path.exists():
        status = _read_json(status_path)
        if "status" not in status or not isinstance(status["status"], str):
            raise ManifestValidationError(
                f"status artifact has no string status: {status_path}")
        artifact_status = status["status"]
        if artifact_status != manifest_status:
            raise ManifestValidationError(
                "command manifest and status artifact disagree: "
                f"{manifest_status!r} != {artifact_status!r}")
        if (artifact_status == "success"
                and "experiment_type" in command_manifest):
            experiment_type = command_manifest["experiment_type"]
            if experiment_type == "production_trace":
                validate_production_trace_outputs(command_manifest)
            else:
                raise ManifestValidationError(
                    f"unknown experiment type: {experiment_type!r}")
        return artifact_status
    if manifest_status in {"dry_run", "running"}:
        return manifest_status
    elif manifest_status in {"success", "failed", "blocked"}:
        raise ManifestValidationError(
            f"terminal experiment state {manifest_status!r} is missing "
            f"status artifact: {status_path}")
    else:
        raise ManifestValidationError(
            f"unknown command manifest status: {manifest_status!r}")


def experiment_resume_decision(status: str | None, *, dry_run: bool) -> str:
    if status == "success":
        return "skip"
    elif dry_run and status == "dry_run":
        return "skip"
    elif status in {"failed", "blocked"}:
        return "rerun"
    elif status is None or status == "running" or (
            status == "dry_run" and not dry_run):
        return "reject"
    else:
        raise ManifestValidationError(f"unknown experiment status: {status!r}")


def validate_resume_manifest_identity(
    command_manifest: dict[str, Any],
    manifest: ExperimentManifest,
    *,
    expected_plan: dict[str, Any] | None = None,
) -> None:
    """Reject resume when an existing cell belongs to another manifest."""
    if manifest.version == 1:
        if ("experiment_manifest" in command_manifest
                or "experiment_type" in command_manifest):
            raise ManifestValidationError(
                "existing cell belongs to a different experiment manifest "
                "version; use --force to replace it")
        return
    elif manifest.version in (2, 3, 4, 5, 6, 7, 8):
        if expected_plan is None:
            raise ManifestValidationError(
                "Production Trace resume requires a plan rematerialized from "
                "the current experiment manifest")
        if "experiment_manifest" not in command_manifest:
            raise ManifestValidationError(
                "existing Production Trace cell has no experiment manifest "
                "identity; use --force to replace it")
        existing_identity = command_manifest["experiment_manifest"]
        if not isinstance(existing_identity, dict):
            raise ManifestValidationError(
                "existing Production Trace cell experiment manifest identity "
                "must be an object")
        required_fields = {"version", "path", "source_sha256"}
        if set(existing_identity) != required_fields:
            raise ManifestValidationError(
                "existing Production Trace cell experiment manifest identity "
                "fields must be "
                f"{sorted(required_fields)}")
        if existing_identity["version"] != manifest.version:
            raise ManifestValidationError(
                "existing cell experiment manifest version disagrees with the "
                "current manifest; use --force to replace it")
        if existing_identity["source_sha256"] != manifest.source_sha256:
            raise ManifestValidationError(
                "existing cell was produced by a different experiment manifest; "
                "use --force to replace it")
        if "provenance" not in command_manifest:
            raise ManifestValidationError(
                "existing Production Trace cell has no code provenance; use "
                "--force to replace it")
        existing_provenance = command_manifest["provenance"]
        if not isinstance(existing_provenance, dict):
            raise ManifestValidationError(
                "existing Production Trace cell code provenance must be an "
                "object")
        required_provenance = {"git_commit", "git_diff_hash"}
        if not required_provenance.issubset(existing_provenance):
            raise ManifestValidationError(
                "existing Production Trace cell code provenance is missing "
                "required fields: "
                f"{sorted(required_provenance - set(existing_provenance))}")
        current_code_identity = git_code_identity(ROOT)
        if existing_provenance["git_commit"] != current_code_identity[
                "git_commit"]:
            raise ManifestValidationError(
                "existing cell was produced by a different code commit; "
                "use --force to replace it")
        if existing_provenance["git_diff_hash"] != current_code_identity[
                "git_diff_hash"]:
            raise ManifestValidationError(
                "existing cell was produced by different worktree contents; "
                "use --force to replace it")
        _validate_resume_static_plan(command_manifest, expected_plan)
        return
    else:
        raise ManifestValidationError(
            f"unsupported experiment manifest version: {manifest.version}")


def _validate_resume_static_plan(
    command_manifest: dict[str, Any],
    expected_plan: dict[str, Any],
) -> None:
    """Bind a resumable Production Trace cell to its canonical plan."""
    static_fields = {
        "version",
        "experiment_type",
        "workflow",
        "cell_id",
        "output_root",
        "input",
        "model",
        "simulation",
        "analysis",
        "environment",
        "validation_status",
        "outputs",
        "commands",
        "command_manifest_path",
    }
    runtime_fields = {
        "experiment_manifest",
        "provenance",
        "status",
        "runtime",
        "failed_stage",
    }
    expected_fields = static_fields | {"provenance", "experiment_manifest"}
    if set(expected_plan) != expected_fields:
        raise AssertionError(
            "rematerialized Production Trace plan fields disagree with the "
            "runner contract: "
            f"expected {sorted(expected_fields)}, got {sorted(expected_plan)}")
    unknown_fields = set(command_manifest) - static_fields - runtime_fields
    if unknown_fields:
        raise ManifestValidationError(
            "existing Production Trace command manifest has unknown fields: "
            f"{sorted(unknown_fields)}")
    missing_fields = static_fields - set(command_manifest)
    if missing_fields:
        raise ManifestValidationError(
            "existing Production Trace command manifest is missing static "
            "plan fields: "
            f"{sorted(missing_fields)}")
    for field in sorted(static_fields):
        existing_value = json.dumps(
            command_manifest[field],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        expected_value = json.dumps(
            expected_plan[field],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if existing_value != expected_value:
            raise ManifestValidationError(
                "existing Production Trace command manifest static plan "
                "disagrees with the "
                f"current experiment manifest at field {field!r}; use --force "
                "to replace it")


def summarize_experiment_cell(
    command_manifest: dict[str, Any],
    status: str,
    *,
    resumed: bool = False,
) -> dict[str, Any]:
    common = {
        "cell_id": command_manifest["cell_id"],
        "status": status,
        "output_root": command_manifest["output_root"],
        "command_manifest": command_manifest["command_manifest_path"],
    }
    if "experiment_type" in command_manifest:
        experiment_type = command_manifest["experiment_type"]
        if experiment_type != "production_trace":
            raise ManifestValidationError(
                f"unknown experiment type: {experiment_type!r}")
        summary = {
            **common,
            "input_kind": command_manifest["input"]["kind"],
            "trace_source_sha256": command_manifest["input"]["trace"][
                "source_sha256"],
            "sweep": command_manifest["outputs"]["sweep"],
            "report_markdown": command_manifest["outputs"][
                "report_markdown"],
        }
    else:
        summary = {
            **common,
            "case_id": command_manifest["case_id"],
            "workload_hash": command_manifest["workload"]["hash"],
        }
    if resumed:
        summary["resumed"] = True
    return summary


def run_experiments(
    argv: list[str] | None = None,
    *,
    subprocess_run=subprocess.run,
    subprocess_popen=subprocess.Popen,
    sleep=time.sleep,
    readiness_probe=None,
    tokenizer_loader=load_model_tokenizer,
) -> dict[str, Any]:
    invocation = _experiment_invocation(argv)
    args = _parse_args(argv)
    mode = _run_mode(args)
    project_root = Path(args.project_root).resolve(strict=False)
    manifest = load_experiment_manifest(args.manifest, project_root=project_root)
    if manifest.version == 9:
        return run_cpu_simulation_experiments(args, manifest, project_root)
    if args.rerun_success:
        raise ManifestValidationError("--rerun-success requires a version 9 simulation manifest")
    if mode == "cpu_run":
        raise ManifestValidationError("--cpu-run requires a version 9 simulation manifest")
    selected = [
        cell for cell in manifest.cells
        if args.only is None or cell.id == args.only
    ]
    if args.only is not None and not selected:
        raise ManifestValidationError(f"unknown cell id for --only: {args.only}")
    if mode == "real_run":
        if manifest.version == 1:
            validate_vllm_experiment_environment()
        elif manifest.version in (2, 3, 4, 5, 6, 7, 8):
            validate_vllm_scheduler_environment()
        else:
            raise ManifestValidationError(
                f"unsupported experiment manifest version: {manifest.version}")

    run_dir = Path(args.run_dir).resolve(strict=False)
    run_dir.mkdir(parents=True, exist_ok=True)

    cell_summaries = []
    tokenizers: dict[str, Any] = {}
    for cell in selected:
        cell_root = run_dir / cell.id
        experiment_plan: dict[str, Any] | None = None
        existing_cell = load_existing_experiment_cell(cell_root)
        if existing_cell is not None and not args.force:
            if args.resume:
                if manifest.version in (2, 3, 4, 5, 6, 7, 8):
                    if not isinstance(cell, ProductionTraceSweepCell):
                        raise AssertionError(
                            "Production Trace manifest produced a "
                            "non-Production-Trace cell")
                    experiment_plan = materialize_experiment_plan(
                        cell,
                        cell_root,
                        project_root=project_root,
                        invocation=invocation,
                        tokenizer=None,
                    )
                    experiment_plan["experiment_manifest"] = {
                        "version": manifest.version,
                        "path": str(manifest.path.resolve(strict=False)),
                        "source_sha256": manifest.source_sha256,
                    }
                validate_resume_manifest_identity(
                    existing_cell,
                    manifest,
                    expected_plan=experiment_plan,
                )
            existing_status = experiment_cell_status(existing_cell)
            if args.resume:
                decision = experiment_resume_decision(
                    existing_status,
                    dry_run=args.dry_run,
                )
                if decision == "skip":
                    cell_summaries.append(summarize_experiment_cell(
                        existing_cell,
                        "skipped",
                        resumed=True,
                    ))
                    continue
                elif decision == "reject":
                    raise ManifestValidationError(
                        "unresumable existing cell state "
                        f"{existing_status!r} for {cell_root}; use --force or inspect "
                        "and clean up the cell output manually")
                elif decision == "rerun":
                    pass
                else:
                    raise ManifestValidationError(
                        f"unknown resume decision: {decision!r}")
            if not args.resume:
                raise ManifestValidationError(
                    f"cell output already exists: {cell_root}; use --resume or --force")
        elif (
            existing_cell is None
            and experiment_cell_output_exists(cell_root)
            and not args.force
        ):
            raise ManifestValidationError(
                f"cell output already exists: {cell_root}; use --resume or --force")

        if experiment_plan is None:
            if isinstance(cell, ProductionTraceSweepCell):
                tokenizer = None
            else:
                model_key = str(cell.model)
                if model_key not in tokenizers:
                    tokenizers[model_key] = tokenizer_loader(cell.model)
                tokenizer = tokenizers[model_key]
            experiment_plan = materialize_experiment_plan(
                cell,
                cell_root,
                project_root=project_root,
                invocation=invocation,
                tokenizer=tokenizer,
            )
            if manifest.version in (2, 3, 4, 5, 6, 7, 8):
                experiment_plan["experiment_manifest"] = {
                    "version": manifest.version,
                    "path": str(manifest.path.resolve(strict=False)),
                    "source_sha256": manifest.source_sha256,
                }
        result = execute_experiment_stages(
            experiment_plan,
            mode=mode,
            subprocess_run=subprocess_run,
            subprocess_popen=subprocess_popen,
            server_wait_seconds=args.server_wait_seconds,
            readiness_poll_interval_seconds=args.readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
        cell_summaries.append(summarize_experiment_cell(
            result,
            result["status"],
        ))

    summary = {
        "version": manifest.version,
        "mode": mode,
        "manifest": str(Path(args.manifest)),
        "run_dir": str(run_dir),
        "status": _aggregate_status([cell["status"] for cell in cell_summaries]),
        "cells": cell_summaries,
    }
    if manifest.version == 1:
        if manifest.case_manifest_path is None:
            raise AssertionError("v1 experiment manifest lost its case manifest")
        summary["case_manifest"] = str(manifest.case_manifest_path)
    elif manifest.version not in (2, 3, 4, 5, 6, 7, 8):
        raise ManifestValidationError(
            f"unsupported experiment manifest version: {manifest.version}")
    else:
        summary["manifest_sha256"] = manifest.source_sha256
    _write_json(run_dir / "run_summary.json", summary)
    return summary


def validate_cpu_success_record_inputs(record: dict[str, Any], manifest: ExperimentManifest,
                                       expected_plan: dict[str, Any], project_root: Path) -> None:
    """Validate the existing owners' identities without asserting dependency coverage."""
    from evaluation.serving.simulation.simulation_artifacts import file_provenance, launch_data_provenance

    if record["status"] != "success" or record["returncode"] != 0:
        raise ManifestValidationError("saved CPU simulation did not complete successfully")
    if expected_plan["normalized_preview"]["prediction_method"] == "pipeweave":
        raise ManifestValidationError(
            "cannot reuse PipeWeave results: its dependencies are not fingerprinted; "
            "use --resume --rerun-success --cpu-run")
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "PYTHONPATH"):
        if name in os.environ and os.environ[name]:
            raise ManifestValidationError(
                f"cannot reuse with {name}: custom search paths are not fingerprinted; "
                "use --resume --rerun-success --cpu-run")
    if record["manifest_sha256"] != manifest.source_sha256:
        raise ManifestValidationError("saved CPU simulation manifest differs")
    if json.dumps(record["plan"], sort_keys=True) != json.dumps(expected_plan, sort_keys=True):
        raise ManifestValidationError("saved CPU simulation plan differs")
    current_code_identity = git_code_identity(project_root)
    require_cpu_worktree_identity(current_code_identity)
    if json.dumps(record["code_identity"], sort_keys=True) != json.dumps(current_code_identity, sort_keys=True):
        raise ManifestValidationError("saved CPU simulation code identity differs")
    if ("simulation_python_environment_at_end" not in record
            or record["simulation_python_environment"] != record["simulation_python_environment_at_end"]
            or record["simulation_python_environment_at_end"] != simulation_python_environment()):
        raise ManifestValidationError("saved CPU simulation Python environment observation differs")
    if ("sdk_python_environment_at_start" not in record or "sdk_python_environment_at_end" not in record
            or record["sdk_python_environment_at_start"] != record["sdk_python_environment_at_end"]
            or record["sdk_python_environment_at_end"] != simulation_sdk_python_environment(
                expected_plan["requested"], project_root)):
        raise ManifestValidationError("saved CPU simulation SDK Python environment observation differs")
    if record["simulation_python_environment_at_end"]["site_package_contents"]["status"] != "observed":
        raise ManifestValidationError("saved CPU simulation Python package contents are unavailable")
    if record["simulation_python_environment_at_end"]["system_library_contents"]["status"] != "observed":
        raise ManifestValidationError("saved CPU simulation system library contents are unavailable")
    if any(entry["status"] == "unavailable" for entry in record["simulation_python_environment_at_end"]["system_configurations"]):
        raise ManifestValidationError("saved CPU simulation system configuration identity is unavailable")
    if any(source["status"] != "missing" for source in record["simulation_python_environment_at_end"]["editable_sources"]):
        raise ManifestValidationError("cannot reuse with an unfingerprinted editable Python source")
    sdk_environment = record["sdk_python_environment_at_end"]
    if (sdk_environment["status"] == "observed"
            and sdk_environment["environment"]["site_package_contents"]["status"] != "observed"):
        raise ManifestValidationError("saved CPU simulation SDK package contents are unavailable")
    if (sdk_environment["status"] == "observed"
            and sdk_environment["environment"]["system_library_contents"]["status"] != "observed"):
        raise ManifestValidationError("saved CPU simulation SDK system library contents are unavailable")
    if (sdk_environment["status"] == "observed" and any(
            entry["status"] == "unavailable" for entry in sdk_environment["environment"]["system_configurations"])):
        raise ManifestValidationError("saved CPU simulation SDK system configuration identity is unavailable")
    if (sdk_environment["status"] == "observed"
            and any(source["status"] != "missing" for source in sdk_environment["environment"]["editable_sources"])):
        raise ManifestValidationError("cannot reuse with an unfingerprinted editable SDK source")
    if ("process_environment_digest_at_start" not in record
            or "process_environment_digest_at_end" not in record
            or record["process_environment_digest_at_start"] != record["process_environment_digest_at_end"]
            or record["process_environment_digest_at_end"] != simulation_process_environment_digest()):
        raise ManifestValidationError("saved CPU simulation process environment differs")
    if "native_runtime_at_start" not in record or "native_runtime_at_end" not in record:
        raise ManifestValidationError("saved CPU simulation native runtime identity was not recorded")
    if (record["native_runtime_at_start"] != record["native_runtime_at_end"]
            or record["native_runtime_at_end"] != simulation_native_runtime_identity(project_root)):
        raise ManifestValidationError("saved CPU simulation native runtime identity differs")
    if ("project_bytecode_at_start" not in record or "project_bytecode_at_end" not in record
            or record["project_bytecode_at_start"] != record["project_bytecode_at_end"]
            or record["project_bytecode_at_end"] != simulation_project_bytecode_identity(project_root)):
        raise ManifestValidationError("saved CPU simulation project bytecode identity differs")
    current_inputs = simulation_explicit_input_files(expected_plan["requested"])
    if (json.dumps(record["explicit_inputs_at_start"], sort_keys=True)
            != json.dumps(record["explicit_inputs_at_end"], sort_keys=True)
            or json.dumps(record["explicit_inputs_at_end"], sort_keys=True)
            != json.dumps(current_inputs, sort_keys=True)):
        raise ManifestValidationError("saved CPU simulation explicit input identity differs")
    for option, entries in current_inputs["files"].items():
        for entry in entries:
            if entry["status"] == "file":
                continue
            if (option == "vllm-source-root"
                    and entry["status"] == "git_directory_partial_identity"
                    and entry["scope"] == "git_contents_and_vllm_package_files"
                    and entry["package_python_sources"]["status"] == "observed"
                    and entry["package_auxiliary"]["status"] == "observed"):
                continue
            raise ManifestValidationError(
                f"saved CPU simulation explicit input {option!r} is not "
                f"fingerprinted: {entry['status']}")
    source_roots = []
    if "vllm-source-root" in expected_plan["requested"]:
        source_value = expected_plan["requested"]["vllm-source-root"]
        if isinstance(source_value, list) and len(source_value) == 1:
            source_value = source_value[0]
        if not isinstance(source_value, str) or not source_value:
            raise ManifestValidationError("saved CPU simulation vLLM source root is invalid")
        source_roots.append(Path(source_value))
    require_reusable_python_search_paths(record["simulation_python_environment_at_end"],
                                         project_root, source_roots)
    if sdk_environment["status"] == "observed":
        require_reusable_python_search_paths(sdk_environment["environment"], project_root, source_roots)
    if (record["request_preview_binding"] != "matched"
            or record["wrapper_and_simulation_entry_identities_match"] is not True
            or record["explicit_input_observations_match"] is not True):
        raise ManifestValidationError("saved CPU simulation has unbound execution identity")
    for name, key in (("result", "output"), ("settings_summary", "settings_summary")):
        if file_provenance(expected_plan[key]) != record["artifacts"][name]:
            raise ManifestValidationError(f"saved CPU simulation {name} artifact identity differs")
    result = _read_json(Path(expected_plan["output"]))
    provenance = result["execution_provenance"]
    current_launch = launch_data_provenance(
        tuple(provenance["loaded_hardware_ids"]), repository_root=project_root)
    if ("launch_data_at_end" not in record or "launch_observations_match" not in record
            or record["launch_observations_match"] is not True
            or json.dumps(record["launch_data_at_end"], sort_keys=True) != json.dumps(
                provenance["launch_inputs"]["launch_data"], sort_keys=True)
            or json.dumps(current_launch, sort_keys=True) != json.dumps(record["launch_data_at_end"], sort_keys=True)):
        raise ManifestValidationError("saved CPU simulation launch data identity differs")


def run_cpu_simulation_experiments(args: argparse.Namespace, manifest: ExperimentManifest,
                                   project_root: Path) -> dict[str, Any]:
    """Preview or execute v9 CPU cells, preserving unsuccessful attempts."""
    if Path.cwd().resolve() != project_root:
        raise ManifestValidationError("version 9 must be invoked from --project-root")
    if args.mock_run or args.real_run:
        raise ManifestValidationError("version 9 accepts only --dry-run or --cpu-run")
    if args.force:
        raise ManifestValidationError("version 9 does not support --force")
    if args.rerun_success and (not args.resume or args.dry_run):
        raise ManifestValidationError("--rerun-success requires --resume --cpu-run")
    selected = [cell for cell in manifest.cells if args.only is None or cell.id == args.only]
    if not selected:
        raise ManifestValidationError(f"unknown cell id for --only: {args.only}")
    run_dir = Path(args.run_dir).resolve()
    logs_root = (project_root / "experiment_logs").resolve()
    if logs_root not in run_dir.parents:
        raise ManifestValidationError("version 9 run directory must be below experiment_logs")
    run_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cell in selected:
        if not isinstance(cell, SimulationCell):
            raise AssertionError("v9 manifest contains a non-simulation cell")
        cell_root = run_dir / cell.id
        rerun_of = None
        recovery_of = None
        if cell_root.resolve() != cell_root:
            raise ManifestValidationError("v9 cell output cannot be a symlink")
        if args.resume:
            if args.dry_run or not cell_root.is_dir():
                raise ManifestValidationError("v9 resume requires an existing executed cell")
            attempts = sorted(p for p in cell_root.glob("attempt-[0-9][0-9][0-9][0-9]") if p.is_dir())
            previous_dir = attempts[-1] if attempts else cell_root
            if previous_dir.resolve() != previous_dir:
                raise ManifestValidationError("v9 attempt output cannot be a symlink")
            previous = _read_json(previous_dir / "command_manifest.json")
            if previous["manifest_sha256"] != manifest.source_sha256:
                raise ManifestValidationError("cannot resume a different manifest")
            if previous["status"] == "success":
                if not args.rerun_success:
                    expected_plan = materialize_cpu_simulation_plan(cell, previous_dir)
                    validate_cpu_success_record_inputs(previous, manifest, expected_plan, project_root)
                    summaries.append({"cell_id": cell.id, "status": "success", "reused": True,
                                      "command_manifest": str(previous_dir / "command_manifest.json"),
                                      "output": expected_plan["output"]})
                    continue
                rerun_of = str(previous_dir / "command_manifest.json")
            elif args.rerun_success:
                raise ManifestValidationError("--rerun-success requires a successful previous attempt")
            elif previous["status"] == "running":
                _require_stopped_cpu_attempt(previous)
                recovery_of = {"command_manifest": str(previous_dir / "command_manifest.json"),
                               "observation": "owner_and_process_group_absent",
                               "previous_status": "running", "termination_cause": "unknown"}
            elif previous["status"] not in ("failed", "cancelled", "interrupted"):
                raise ManifestValidationError(f"cannot resume cell status {previous['status']!r}")
            if len(attempts) >= 9999:
                raise ManifestValidationError("v9 attempt limit reached")
            execution_root = cell_root / f"attempt-{len(attempts) + 1:04d}"
        elif cell_root.exists():
            raise ManifestValidationError(f"cell output already exists: {cell_root}")
        else:
            execution_root = cell_root
        plan = materialize_cpu_simulation_plan(cell, execution_root)
        if args.dry_run:
            summaries.append({"cell_id": cell.id, "status": "dry_run", "plan": plan})
            continue
        explicit_inputs_at_start = simulation_explicit_input_files(cell.options)
        native_runtime_at_start = simulation_native_runtime_identity(project_root)
        code_identity = git_code_identity(project_root)
        require_cpu_worktree_identity(code_identity)
        execution_root.mkdir()
        record = {"plan": plan, "manifest_sha256": manifest.source_sha256,
                  "code_identity": code_identity, "status": "running",
                  "rerun_of": rerun_of,
                  "recovery_of": recovery_of,
                  "runner_process": _cpu_process_identity(), "child_process_group": None,
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "cpu_affinity": sorted(os.sched_getaffinity(0)),
                  "wall_time_budget_s": None,
                  "wall_time_budget_source": "no_subprocess_timeout",
                  "simulation_python_environment": simulation_python_environment(),
                  "process_environment_digest_at_start": simulation_process_environment_digest(),
                  "explicit_inputs_at_start": explicit_inputs_at_start,
                  "native_runtime_at_start": native_runtime_at_start}
        record_path = execution_root / "command_manifest.json"
        _write_json_atomic(record_path, record)
        start = time.monotonic()
        postprocessing_started = None
        environment = simulation_subprocess_environment()
        previous_sigterm = signal.signal(signal.SIGTERM, _raise_cpu_termination)
        try:
            record["project_bytecode_at_start"] = simulation_project_bytecode_identity(project_root)
            record["sdk_python_environment_at_start"] = simulation_sdk_python_environment(cell.options, project_root)
            _write_json_atomic(record_path, record)

            def on_started(process_group: int) -> None:
                record["child_process_group"] = process_group
                _write_json_atomic(record_path, record)

            with (execution_root / "stdout.log").open("w") as stdout, (execution_root / "stderr.log").open("w") as stderr:
                process = _execute_cpu_process(plan["command"], cwd=project_root, env=environment,
                                               stdout=stdout, stderr=stderr, on_started=on_started)
            postprocessing_started = time.monotonic()
            record["returncode"] = process.returncode
            if process.returncode < 0:
                record["termination_signal"] = -process.returncode
                record["termination_cause"] = "unknown"
            if process.returncode == 0 and Path(plan["output"]).is_file() and Path(plan["settings_summary"]).is_file():
                result = _read_json(Path(plan["output"]))
                if "interactive_report" not in result:
                    raise ValueError("new CPU experiment omitted its default report receipt")
                receipt = result["interactive_report"]
                if plan["report_enabled"]:
                    for field in ("path", "data_path", "cost_receipt_path"):
                        if not Path(receipt[field]).is_file():
                            raise ValueError("default report receipt references a missing artifact")
                elif "status" not in receipt or receipt["status"] != "disabled" or not receipt["reason"]:
                    raise ValueError("report opt-out lacks a recorded reason")
                record["interactive_report"] = receipt
                progress = result["scheduler_progress"]["summary"]
                complete = progress["completed_requests"] == progress["request_count"]
                repository = result["execution_provenance"]["repository"]
                unchanged = repository["identity_changed_between_boundaries"] is False
                wrapper_identity = json.dumps(record["code_identity"], sort_keys=True)
                child_identity = json.dumps(repository["run_start_identity"], sort_keys=True)
                record["wrapper_and_simulation_entry_identities_match"] = wrapper_identity == child_identity
                request_preview = plan["request_preview"]
                if request_preview["status"] == "input_parsed_not_executed":
                    request_binding = (request_preview["request_stream_sha256"]
                                       == result["workload_assumption_binding"]["request_stream_sha256"])
                    record["request_preview_binding"] = "matched" if request_binding else "mismatch"
                elif request_preview["status"] == "not_materialized_large_input":
                    request_binding = True
                    record["request_preview_binding"] = "not_checked_large_input"
                else:
                    raise ValueError(f"unknown request preview status: {request_preview['status']!r}")
                record["explicit_inputs_at_end"] = simulation_explicit_input_files(cell.options)
                same_observations = record["explicit_inputs_at_start"] == record["explicit_inputs_at_end"]
                record["explicit_input_observations_match"] = same_observations
                record["native_runtime_at_end"] = simulation_native_runtime_identity(project_root)
                same_native = record["native_runtime_at_start"] == record["native_runtime_at_end"]
                record["project_bytecode_at_end"] = simulation_project_bytecode_identity(project_root)
                from evaluation.serving.simulation.simulation_artifacts import launch_data_provenance
                execution_provenance = result["execution_provenance"]
                record["launch_data_at_end"] = launch_data_provenance(
                    tuple(execution_provenance["loaded_hardware_ids"]), repository_root=project_root)
                same_launch = (json.dumps(record["launch_data_at_end"], sort_keys=True)
                               == json.dumps(execution_provenance["launch_inputs"]["launch_data"], sort_keys=True))
                record["launch_observations_match"] = same_launch
                record["simulation_python_environment_at_end"] = simulation_python_environment()
                same_python = record["simulation_python_environment"] == record["simulation_python_environment_at_end"]
                record["sdk_python_environment_at_end"] = simulation_sdk_python_environment(cell.options, project_root)
                same_sdk_python = (record["sdk_python_environment_at_start"]
                                   == record["sdk_python_environment_at_end"])
                record["process_environment_digest_at_end"] = simulation_process_environment_digest()
                same_process_environment = (record["process_environment_digest_at_start"]
                                            == record["process_environment_digest_at_end"])
                record["status"] = "success" if (result["main_chain_status"]["status"] == "executed"
                    and complete and unchanged and record["wrapper_and_simulation_entry_identities_match"]
                    and request_binding and same_observations and same_native and same_launch and same_python
                    and same_sdk_python
                    and same_process_environment) else "failed"
                with Path(plan["settings_summary"]).open("a", encoding="utf-8") as stream:
                    stream.write(render_simulation_setting_comparison(plan, result))
                    stream.write("\n## Experiment record\n\n"
                                 f"Requested options, normalized preview, exact command, CPU affinity and "
                                 f"execution status: `{record_path}`.\n\n"
                                 "The preview is parsed configuration; the runtime fields above are "
                                 "the simulation's reported effective values. Ground-truth capture "
                                 "environment and actual GPU runtime are not supplied by this CPU run.\n")
                from evaluation.serving.simulation.simulation_artifacts import file_provenance
                record["artifacts"] = {"result": file_provenance(plan["output"]),
                                       "settings_summary": file_provenance(plan["settings_summary"])}
            elif process.returncode in (-signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL):
                record["status"] = "interrupted"
            else:
                record["status"] = "failed"
        except _CpuSimulationTerminated as error:
            record["status"] = "interrupted"
            record["termination_signal"] = error.signum
            record["termination_cause"] = "unknown"
        except KeyboardInterrupt:
            record["status"] = "cancelled"
            raise
        except Exception as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            record["postprocessing_wall_time_s"] = (None if postprocessing_started is None
                                                    else time.monotonic() - postprocessing_started)
            if plan["report_enabled"] and "interactive_report" not in record:
                from evaluation.reporting.serving_interactive import write_status_report
                record["interactive_report"] = write_status_report({
                    "main_chain_status": {"status": record["status"],
                        "reason": "CPU experiment did not produce a verified completed report; see command_manifest and process logs"},
                    "request_source": {"requested_cli_options": plan["requested"]}}, plan["interactive_report"])
            record["plotting"] = "not_performed"
            record["wall_time_s"] = time.monotonic() - start
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            try:
                _write_json_atomic(record_path, record)
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
        summaries.append({"cell_id": cell.id, "status": record["status"],
                          "command_manifest": str(record_path), "output": plan["output"],
                          **{key: record[key] for key in ("interactive_report",) if key in record}})
    summary = {"version": 9, "mode": "dry_run" if args.dry_run else "cpu_run",
               "manifest": str(manifest.path), "manifest_sha256": manifest.source_sha256,
               "run_dir": str(run_dir), "cells": summaries,
               "status": _aggregate_status([item["status"] for item in summaries])}
    _write_json(run_dir / "run_summary.json", summary)
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manifest-driven real vLLM E2E experiments")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--only", default=None)
    rerun = parser.add_mutually_exclusive_group()
    rerun.add_argument("--resume", action="store_true")
    rerun.add_argument("--force", action="store_true")
    parser.add_argument("--rerun-success", action="store_true",
                        help="with --resume --cpu-run, execute a new attempt after success")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--mock-run", action="store_true")
    mode.add_argument("--real-run", action="store_true")
    mode.add_argument("--cpu-run", action="store_true", help="run a version 9 CPU simulation")
    parser.add_argument(
        "--server-wait-seconds",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for vLLM HTTP readiness in real-run mode.",
    )
    parser.add_argument(
        "--readiness-poll-interval-seconds",
        type=float,
        default=1.0,
        help="Seconds between vLLM readiness probes in real-run mode.",
    )
    args = parser.parse_args(argv)
    if args.readiness_poll_interval_seconds <= 0:
        raise ManifestValidationError(
            "--readiness-poll-interval-seconds must be positive")
    return args


def _run_mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry_run"
    elif args.mock_run:
        return "mock_run"
    elif args.real_run:
        return "real_run"
    elif args.cpu_run:
        return "cpu_run"
    else:
        raise ManifestValidationError(
            "runner requires --dry-run, --mock-run, or --real-run")


def _experiment_invocation(argv: list[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    return ["scripts/experiments/run_serving_experiments.py", *args]


def _aggregate_status(statuses: list[str]) -> str:
    allowed = {"dry_run", "success", "skipped", "blocked", "failed", "interrupted"}
    unknown = sorted(set(statuses) - allowed)
    if unknown:
        raise ManifestValidationError(
            f"cannot aggregate unknown experiment statuses: {unknown}")
    if not statuses:
        return "empty"
    elif all(status == "dry_run" for status in statuses):
        return "dry_run"
    elif all(status in {"success", "skipped"} for status in statuses):
        return "success"
    elif any(status == "blocked" for status in statuses):
        return "blocked"
    elif any(status == "failed" for status in statuses):
        return "failed"
    elif any(status == "interrupted" for status in statuses):
        return "interrupted"
    else:
        raise ManifestValidationError(
            f"unhandled experiment status combination: {statuses}")




def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ManifestValidationError(f"JSON root must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    summary = run_experiments(argv)
    if summary["version"] == 9 and summary["status"] in ("failed", "interrupted"):
        return 1
    return 0
