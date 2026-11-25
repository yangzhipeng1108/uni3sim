"""Execute one immutable real 1P1D capture plan after idle evidence exists."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping, Sequence

from evaluation.serving.experiments.pd_capture_plan import MAX_DEVICE_ID
from urllib import error as urllib_error
from urllib import request as urllib_request

from evaluation.json_shape import require_object_value as _mapping

from evaluation.json_shape import require_text as _text


_ROLES = ("prefill", "decode", "router")
_FORBIDDEN_PROFILER_TOKENS = ("msprof", "nsys", "torch.profiler")


def execute_pd_capture_plan(
    plan_path: str | Path,
    *,
    readiness_timeout_s: float = 900.0,
    benchmark_timeout_s: float = 7200.0,
    poll_interval_s: float = 1.0,
) -> Mapping[str, object]:
    """Run P, D, router and benchmark; always stop only owned processes.

    The caller is responsible for supplying the Goal-required sixty-second
    idle evidence before this function is invoked.  The executor never selects
    devices, never attaches a profiler, and never removes a pre-existing path.
    """

    if readiness_timeout_s <= 0 or benchmark_timeout_s <= 0 or poll_interval_s <= 0:
        raise ValueError("PD executor timeouts and poll interval must be positive")
    plan = _load_plan(Path(plan_path))
    raw_root = Path(_text(plan["raw_root"], "PD plan raw_root")).resolve(strict=True)
    _validate_setting_parity(plan)
    execution_path = raw_root / "execution.json"
    if execution_path.exists():
        raise FileExistsError(f"refusing to overwrite PD execution evidence: {execution_path}")
    commands = _mapping(plan["commands"], "PD plan commands")
    for role in _ROLES:
        _validate_server_command(_mapping(commands[role], f"PD {role} command"), role)
    benchmark = _mapping(commands["benchmark"], "PD benchmark command")
    benchmark_argv = _argv(benchmark["argv"], "PD benchmark argv")
    _assert_no_profiler(benchmark_argv, {})
    logs = raw_root / "runtime_logs"
    if logs.exists():
        raise FileExistsError(f"refusing to overwrite PD runtime logs: {logs}")
    logs.mkdir()
    started_ns = time.perf_counter_ns()
    processes: dict[str, subprocess.Popen[bytes]] = {}
    handles = []
    benchmark_result: subprocess.CompletedProcess[bytes] | None = None
    failure: BaseException | None = None
    try:
        # P2P connectors may establish their communicator while each vLLM
        # server is becoming healthy.  Launch both peers before waiting for
        # either health endpoint, otherwise Prefill can wait forever for a
        # Decode process the executor has not started yet.
        for role in ("prefill", "decode"):
            command = _mapping(commands[role], f"PD {role} command")
            stdout = (logs / f"{role}.stdout.log").open("xb")
            stderr = (logs / f"{role}.stderr.log").open("xb")
            handles.extend((stdout, stderr))
            argv = _argv(command["argv"], f"PD {role} argv")
            environment = _environment(command["environment"], f"PD {role} environment")
            _assert_no_profiler(argv, environment)
            process = subprocess.Popen(
                argv,
                cwd=raw_root,
                env={**os.environ, **environment},
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            processes[role] = process
        for role in ("prefill", "decode"):
            command = _mapping(commands[role], f"PD {role} command")
            _wait_for_health(
                _text(_mapping(command["endpoint"], f"PD {role} endpoint")["health_url"], f"PD {role} health URL"),
                processes[role],
                readiness_timeout_s,
                poll_interval_s,
            )
        router = _mapping(commands["router"], "PD router command")
        stdout = (logs / "router.stdout.log").open("xb")
        stderr = (logs / "router.stderr.log").open("xb")
        handles.extend((stdout, stderr))
        router_argv = _argv(router["argv"], "PD router argv")
        router_environment = _environment(
            router["environment"], "PD router environment"
        )
        _assert_no_profiler(router_argv, router_environment)
        processes["router"] = subprocess.Popen(
            router_argv,
            cwd=raw_root,
            env={**os.environ, **router_environment},
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        _wait_for_health(
            _text(_mapping(router["endpoint"], "PD router endpoint")["health_url"], "PD router health URL"),
            processes["router"], readiness_timeout_s, poll_interval_s,
        )
        benchmark_stdout = (logs / "benchmark.stdout.log").open("xb")
        benchmark_stderr = (logs / "benchmark.stderr.log").open("xb")
        handles.extend((benchmark_stdout, benchmark_stderr))
        benchmark_environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "ASCEND_RT_VISIBLE_DEVICES": "",
            "MACA_VISIBLE_DEVICES": "",
        }
        benchmark_result = subprocess.run(
            benchmark_argv,
            cwd=raw_root,
            env=benchmark_environment,
            stdout=benchmark_stdout,
            stderr=benchmark_stderr,
            timeout=benchmark_timeout_s,
            check=False,
        )
        if benchmark_result.returncode != 0:
            raise RuntimeError(f"PD benchmark failed with exit code {benchmark_result.returncode}")
    except BaseException as error:
        failure = error
        raise
    finally:
        for process in processes.values():
            _terminate_owned_process(process)
        for handle in handles:
            handle.close()
        finished_ns = time.perf_counter_ns()
        _write_execution_evidence(
            execution_path,
            plan=plan,
            processes=processes,
            logs=logs,
            benchmark_result=benchmark_result,
            started_ns=started_ns,
            finished_ns=finished_ns,
            failure=failure,
        )
    _validate_trace_artifacts(plan)
    return _load_object(execution_path, "PD execution evidence")


def _load_plan(path: Path) -> Mapping[str, object]:
    plan = _load_object(path.resolve(strict=True), "PD capture plan")
    _exact_keys(
        plan,
        (
            "schema_version", "kind", "identity", "run_contract", "raw_root",
            "devices", "commands", "artifacts", "profiler_attached",
        ),
        "PD capture plan",
    )
    if plan["schema_version"] != 1 or plan["kind"] != "sharegpt_pd_capture_plan":
        raise ValueError("PD capture plan identity is invalid")
    if plan["profiler_attached"] is not False:
        raise ValueError("PD capture plan may not attach a profiler")
    identity = _mapping(plan["identity"], "PD plan identity")
    hardware_id = _text(identity["hardware_id"], "PD plan hardware ID")
    if hardware_id not in MAX_DEVICE_ID:
        raise ValueError(f"unsupported PD plan hardware: {hardware_id!r}")
    maximum = MAX_DEVICE_ID[hardware_id]
    devices = _mapping(plan["devices"], "PD plan devices")
    _exact_keys(devices, ("prefill", "decode"), "PD plan devices")
    values: list[int] = []
    for role in ("prefill", "decode"):
        role_devices = _sequence(devices[role], f"PD {role} devices")
        if not role_devices:
            raise ValueError("PD role device allocation may not be empty")
        for device in role_devices:
            if (
                isinstance(device, bool)
                or not isinstance(device, int)
                or not 0 <= device <= maximum
            ):
                raise ValueError(
                    f"PD executor only permits {hardware_id} devices zero "
                    f"through {maximum}"
                )
            values.append(device)
    if len(values) != len(set(values)):
        raise ValueError("PD Prefill and Decode device allocations overlap")
    return plan


def _validate_server_command(command: Mapping[str, object], role: str) -> None:
    if role == "router":
        expected = ("argv", "environment", "endpoint")
    elif role in {"prefill", "decode"}:
        expected = (
            "argv", "environment", "endpoint", "engine",
            "scheduler_trace", "profiler_attached",
        )
    else:
        raise ValueError(f"unsupported PD command role: {role!r}")
    _exact_keys(command, expected, f"PD {role} command")
    _argv(command["argv"], f"PD {role} argv")
    _environment(command["environment"], f"PD {role} environment")
    endpoint = _mapping(command["endpoint"], f"PD {role} endpoint")
    _exact_keys(endpoint, ("base_url", "health_url"), f"PD {role} endpoint")
    _text(endpoint["base_url"], f"PD {role} base URL")
    _text(endpoint["health_url"], f"PD {role} health URL")
    if role != "router" and command["profiler_attached"] is not False:
        raise ValueError("PD server command may not attach a profiler")


def _wait_for_health(
    health_url: str,
    process: subprocess.Popen[bytes],
    timeout_s: float,
    interval_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"PD server exited before readiness: {exit_code}")
        try:
            with urllib_request.urlopen(health_url, timeout=min(interval_s, 5.0)) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib_error.URLError) as error:
            last_error = str(error)
        time.sleep(interval_s)
    raise TimeoutError(f"PD server did not become ready: {health_url}; last error: {last_error}")


def _terminate_owned_process(process: subprocess.Popen[bytes]) -> None:
    """Stop the complete session created for one owned server command.

    vLLM can let its API-server parent exit before its worker descendants do.
    Those descendants remain in the session created by ``start_new_session``;
    checking ``process.poll()`` first would therefore leak them.
    """

    process_group = process.pid
    for termination_signal, timeout_s in (
        (signal.SIGTERM, 10.0),
        (signal.SIGKILL, 10.0),
    ):
        try:
            os.killpg(process_group, termination_signal)
        except ProcessLookupError:
            return
        if _wait_for_process_group_exit(process, timeout_s):
            return


def _wait_for_process_group_exit(
    process: subprocess.Popen[bytes], timeout_s: float
) -> bool:
    process_group = process.pid
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        # Reap an exited API-server parent before probing its session.  Without
        # this, a zombie group leader keeps killpg(..., 0) successful even
        # after all workers are gone, making every completed PD case wait for
        # the full signal timeout.
        process.poll()
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return True
    return False


def _validate_trace_artifacts(plan: Mapping[str, object]) -> None:
    artifacts = _mapping(plan["artifacts"], "PD plan artifacts")
    for role, field in (("prefill", "prefill_scheduler"), ("decode", "decode_scheduler")):
        path_value = artifacts[field]
        if path_value is None:
            continue
        path = Path(_text(path_value, f"PD {role} Scheduler trace")).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"PD {role} Scheduler trace is missing or empty")


def _validate_setting_parity(plan: Mapping[str, object]) -> None:
    artifacts = _mapping(plan["artifacts"], "PD plan artifacts")
    report_path = Path(
        _text(artifacts["setting_parity"], "PD setting-parity report")
    ).resolve(strict=True)
    report = _load_object(report_path, "PD setting-parity report")
    _exact_keys(
        report,
        (
            "schema_version", "kind", "identity", "source",
            "allowed_differences", "engine_fields", "differences", "passed",
        ),
        "PD setting-parity report",
    )
    if report["schema_version"] != 1 or report["kind"] != "sharegpt_pd_setting_parity":
        raise ValueError("PD setting-parity report identity is invalid")
    if report["passed"] is not True:
        raise ValueError("PD setting-parity report did not pass")
    report_identity = _mapping(report["identity"], "PD setting-parity identity")
    plan_identity = _mapping(plan["identity"], "PD plan identity")
    for field in ("case_id", "hardware_id", "model_id"):
        if report_identity[field] != plan_identity[field]:
            raise ValueError("PD setting-parity identity differs from capture plan")
    differences = _mapping(report["differences"], "PD setting-parity differences")
    _exact_keys(differences, ("prefill", "decode"), "PD setting-parity differences")
    for role in ("prefill", "decode"):
        values = _sequence(differences[role], f"PD setting-parity {role} differences")
        if values:
            raise ValueError("PD setting-parity report has non-topology differences")


def _write_execution_evidence(
    path: Path,
    *,
    plan: Mapping[str, object],
    processes: Mapping[str, subprocess.Popen[bytes]],
    logs: Path,
    benchmark_result: subprocess.CompletedProcess[bytes] | None,
    started_ns: int,
    finished_ns: int,
    failure: BaseException | None,
) -> None:
    identity = _mapping(plan["identity"], "PD plan identity")
    value = {
        "schema_version": 1,
        "kind": "sharegpt_pd_execution",
        "identity": dict(identity),
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "profiler_attached": False,
        "logs": str(logs),
        "processes": {
            role: {"pid": process.pid, "returncode": process.returncode}
            for role, process in processes.items()
        },
        "benchmark_returncode": None if benchmark_result is None else benchmark_result.returncode,
        "status": "success" if failure is None else "failed",
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_no_profiler(argv: Sequence[str], environment: Mapping[str, str]) -> None:
    combined = " ".join(argv) + " " + " ".join(f"{key}={value}" for key, value in environment.items())
    normalized = combined.lower()
    if any(token in normalized for token in _FORBIDDEN_PROFILER_TOKENS):
        raise ValueError("PD executor forbids profiler attachment")


def _load_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return _mapping(value, label)


def _argv(value: object, label: str) -> list[str]:
    values = _sequence(value, label)
    if not values:
        raise ValueError(f"{label} must be non-empty")
    result = []
    for index, item in enumerate(values):
        result.append(_text(item, f"{label}[{index}]"))
    return result


def _environment(value: object, label: str) -> dict[str, str]:
    mapping = _mapping(value, label)
    result = {}
    for key, item in mapping.items():
        normalized_key = _text(key, f"{label} key")
        if not isinstance(item, str):
            raise ValueError(f"{label}[{key}] must be text")
        result[normalized_key] = item
    return result




def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return value




def _exact_keys(value: Mapping[str, object], expected: Sequence[str], label: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{label} field set is invalid")
