"""Collect fail-closed host process evidence for MetaX C500 Event runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any, Mapping


MX_SMI_PATH = Path("/opt/mxdriver/bin/mx-smi")
EXPECTED_DEVICE_COUNT = 4
_ATTACHED_PATTERN = re.compile(r"^Attached GPUs\s*:\s*(\d+)\s*$")
_PROCESS_PATTERN = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s+\S.*?\s+(\d+)\s*\|\s*$")


def parse_mx_smi_processes(output: str) -> Mapping[int, tuple[int, ...]]:
    """Parse the complete device-file process table from ``mx-smi``."""
    if not isinstance(output, str) or not output:
        raise ValueError("mx-smi process output is empty")
    attached_count = None
    saw_process_table = False
    saw_end = False
    device_pids: dict[int, set[int]] = {
        device: set() for device in range(EXPECTED_DEVICE_COUNT)
    }
    for line in output.splitlines():
        stripped = line.strip()
        attached = _ATTACHED_PATTERN.fullmatch(stripped)
        if attached is not None:
            attached_count = int(attached.group(1))
        if stripped.startswith("| Process:"):
            saw_process_table = True
        if stripped == "End of Log":
            saw_end = True
        process = _PROCESS_PATTERN.fullmatch(stripped)
        if process is None:
            continue
        device = int(process.group(1))
        pid = int(process.group(2))
        memory_mib = int(process.group(3))
        if device not in device_pids or pid <= 0:
            raise ValueError("mx-smi process row has an invalid device or PID")
        if memory_mib > 0:
            device_pids[device].add(pid)
    if (attached_count != EXPECTED_DEVICE_COUNT
            or not saw_process_table or not saw_end):
        raise ValueError("mx-smi process output is incomplete")
    return {
        device: tuple(sorted(pids)) for device, pids in device_pids.items()
    }


def _read_process(proc_root: Path, pid: int) -> Mapping[str, Any] | None:
    process_root = proc_root / str(pid)
    try:
        stat = (process_root / "stat").read_text(encoding="utf-8")
        cmdline = (process_root / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    close = stat.rfind(")")
    if close < 0:
        raise ValueError(f"invalid /proc stat record for PID {pid}")
    prefix = stat[:close + 1]
    open_parenthesis = prefix.find("(")
    if open_parenthesis < 0:
        raise ValueError(f"invalid /proc stat command for PID {pid}")
    fields = stat[close + 2:].split()
    if len(fields) < 3:
        raise ValueError(f"truncated /proc stat record for PID {pid}")
    ppid = int(fields[1])
    process_group_id = int(fields[2])
    argv = [
        value.decode("utf-8", errors="surrogateescape")
        for value in cmdline.split(b"\0") if value
    ]
    if not argv:
        argv = [f"[{prefix[open_parenthesis + 1:-1]}]"]
    return {
        "pid": pid,
        "ppid": ppid,
        "process_group_id": process_group_id,
        "argv": argv,
        "physical_devices": [],
    }


def _collect_full_system_process_snapshot_once(
        *, mx_smi_path: Path = MX_SMI_PATH, proc_root: Path = Path("/proc"),
        host: str | None = None) -> Mapping[str, Any]:
    """Collect one internally consistent host/device process snapshot."""
    command = [str(mx_smi_path.resolve()), "--show-process"]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=15)
    device_pids = parse_mx_smi_processes(completed.stdout)
    process_ids = sorted(
        int(path.name) for path in proc_root.iterdir() if path.name.isdigit())
    process_index = {}
    for pid in process_ids:
        process = _read_process(proc_root, pid)
        if process is not None:
            process_index[pid] = dict(process)
    device_processes = []
    for device, pids in device_pids.items():
        for pid in pids:
            if pid not in process_index:
                raise RuntimeError(
                    f"mx-smi PID disappeared during process snapshot: {pid}")
            process_index[pid]["physical_devices"].append(device)
            device_processes.append({"physical_device": device, "pid": pid})
    processes = []
    for pid in sorted(process_index):
        process = process_index[pid]
        process["physical_devices"] = sorted(process["physical_devices"])
        processes.append(process)
    return {
        "schema_version": 1,
        "artifact_kind": "metax_c500_full_system_process_snapshot",
        "scope": "host_all_processes",
        "collection_complete": True,
        "host": socket.gethostname() if host is None else host,
        "timestamp_wall_time_ns": time.time_ns(),
        "processes": processes,
        "device_processes": device_processes,
    }


def collect_full_system_process_snapshot(
        *, mx_smi_path: Path = MX_SMI_PATH, proc_root: Path = Path("/proc"),
        host: str | None = None, transient_attempts: int = 3,
        transient_interval_seconds: float = 0.05) -> Mapping[str, Any]:
    """Collect a snapshot, retrying the complete read on mx-smi/proc races."""
    if (isinstance(transient_attempts, bool)
            or not isinstance(transient_attempts, int)
            or transient_attempts <= 0
            or transient_interval_seconds < 0):
        raise ValueError("snapshot transient retry policy is invalid")
    failures = []
    for attempt in range(transient_attempts):
        try:
            return _collect_full_system_process_snapshot_once(
                mx_smi_path=mx_smi_path, proc_root=proc_root, host=host)
        except RuntimeError as error:
            if "mx-smi PID disappeared during process snapshot:" not in str(error):
                raise
            failures.append(f"attempt {attempt + 1}: {type(error).__name__}: {error}")
            if attempt + 1 < transient_attempts:
                time.sleep(transient_interval_seconds)
    raise RuntimeError(
        "full-system snapshot exhausted transient mx-smi/proc retries: "
        + "; ".join(failures))


def write_process_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    """Write one immutable snapshot without replacing existing evidence."""
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")
    descriptor = os.open(
        output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError):
        if output.exists():
            output.unlink()
        raise
