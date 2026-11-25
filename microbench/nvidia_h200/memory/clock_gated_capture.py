#!/usr/bin/env python3
"""Clock-gated process capture for the H200 memory mechanism runners.

The H200 SM clock cannot be locked in the capture container, so every timed
probe process is sampled with ``nvidia-smi -lms 50`` for its whole lifetime.
A capture is accepted only when it has at least one busy sample
(``utilization.gpu > 0``) and every busy sample reports at least the 1980 MHz
boost clock; otherwise the process is rejected and re-run.  Accepted and
rejected attempts are both returned so the published reference can record
them.  The helper runs on the capture host with the system Python (stdlib
only).
"""

import hashlib
import json
import os
import statistics
import subprocess
import shutil
import time
from pathlib import Path

BOOST_MHZ = 1980
DEVICE_NAME = "NVIDIA H200"
NVCC = Path(shutil.which("nvcc") or "nvcc")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_state(device):
    result = subprocess.run(
        ["nvidia-smi", f"--id={device}",
         "--query-gpu=index,name,uuid,pci.bus_id,memory.used,utilization.gpu,"
         "clocks.current.sm,clocks.current.memory",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=60)
    values = [item.strip() for item in result.stdout.split(",")]
    if len(values) != 8:
        raise ValueError(f"unexpected nvidia-smi row: {result.stdout!r}")
    state = {
        "index": int(values[0]), "name": values[1], "uuid": values[2],
        "pci_bus_id": values[3], "memory_used_mib": int(values[4]),
        "gpu_utilization_pct": int(values[5]),
        "sm_clock_mhz": int(values[6]), "memory_clock_mhz": int(values[7]),
    }
    if state["name"] != DEVICE_NAME:
        raise RuntimeError(f"device {device} is {state['name']}, not {DEVICE_NAME}")
    return state


def require_idle(device, *, settle_seconds=0.0):
    deadline = time.monotonic() + settle_seconds
    while True:
        state = device_state(device)
        apps = subprocess.run(
            ["nvidia-smi", f"--id={device}",
             "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True, text=True, capture_output=True, timeout=60).stdout.split()
        if (state["gpu_utilization_pct"] == 0 and state["memory_used_mib"] <= 64
                and not apps):
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"device {device} is not idle: {state} apps={apps}")
        time.sleep(0.5)


def probe_environment(device):
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    return environment


def _clock_stats(log_path):
    samples = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        clock, utilization = (item.strip() for item in line.split(","))
        samples.append((int(clock), int(utilization)))
    busy = [clock for clock, utilization in samples if utilization > 0]
    return {
        "samples": len(samples),
        "busy_samples": len(busy),
        "busy_sm_clock_mhz_min": min(busy) if busy else None,
        "busy_sm_clock_mhz_median": statistics.median(busy) if busy else None,
        "busy_sm_clock_mhz_max": max(busy) if busy else None,
        "all_sm_clock_mhz_min": min(c for c, _ in samples) if samples else None,
    }


def run_clock_gated(command, *, device, log_stem, attempts=3, timeout=3600):
    """Run ``command`` on ``device`` until one attempt passes the clock gate.

    Returns ``(stdout, accepted, rejected)`` where ``accepted`` is the clock
    statistics of the accepted attempt and ``rejected`` the list of rejected
    attempt statistics.  Raises when no attempt is accepted.
    """

    log_stem = Path(log_stem)
    rejected = []
    for attempt in range(attempts):
        log_path = log_stem.with_name(f"{log_stem.name}.attempt{attempt}.clock.csv")
        with log_path.open("w") as log:
            sampler = subprocess.Popen(
                ["nvidia-smi", "-i", str(device),
                 "--query-gpu=clocks.sm,utilization.gpu",
                 "--format=csv,noheader,nounits", "-lms", "50"],
                stdout=log, stderr=subprocess.DEVNULL)
            try:
                time.sleep(0.2)
                result = subprocess.run(
                    command, check=True, text=True, capture_output=True,
                    env=probe_environment(device), timeout=timeout)
                time.sleep(0.2)
            finally:
                sampler.terminate()
                sampler.wait()
        stats = _clock_stats(log_path)
        stats["attempt"] = attempt
        stats["clock_log"] = log_path.name
        if stats["busy_samples"] > 0 and stats["busy_sm_clock_mhz_min"] >= BOOST_MHZ:
            (log_stem.with_name(f"{log_stem.name}.stdout")).write_text(result.stdout)
            (log_stem.with_name(f"{log_stem.name}.stderr")).write_text(result.stderr)
            return result.stdout, stats, rejected
        rejected.append(stats)
    raise RuntimeError(
        f"no clock-accepted attempt for {command}: {json.dumps(rejected)}")
