"""In-runner H200 capture session for the NSYS NVTX kernel-timing protocol.

This module is the NVIDIA H200 hardware-coupling layer added to owner copies
of the A800 kernel-dataset runners (``scripts/kernel_dataset/nvidia_h200``).
It is stdlib-only so the runners can import it on the remote H200 host.

The A800 canonical timing protocol (``nsys_nvtx_kernel_median``: one exact
public-op call inside a unique NVTX range, CUDA device kernel start->end from
``nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none``, three
independent process windows, median whole-trace window) ran on SM clocks
locked at 1140 MHz.  The H200 container cannot lock clocks
(``nvidia-smi -lgc`` is denied), so this module adds only what that hardware
difference requires:

* a BF16 GEMM power warm-up immediately before the runner's own warm-up
  (same 4096^3 x 1000 GEMM as the A800 Vector power warm-up constants),
  followed by an idle wait until nvidia-smi reports the 1980 MHz boost clock
  again (the GEMM trips the SW power cap).  The bracketing clock samples do
  not establish the frequency during the measured kernel;
* an SM-clock query immediately before the NVTX push and after the NVTX pop,
  written to a telemetry side file; the campaign rejects every window whose
  clocks are below 1980 MHz;
* an exclusive GPU lock taken before the first CUDA call and held until
  process exit, so several pre-imported runner processes can be staged while
  exactly one of them touches the GPU at any time;
* a device identity check (``NVIDIA H200``, 132 SMs, 60 MiB L2).
"""

from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping


H200_HARDWARE_ID = "nvidia_h200"
H200_DEVICE_NAME = "NVIDIA H200"
H200_SM_COUNT = 132
H200_L2_BYTES = 60 * 1024 * 1024
H200_REQUIRED_SM_CLOCK_MHZ = 1980
# Eight H200 L2 capacities; the A800 equivalent was 320 MiB (8 x 40 MiB).
H200_MINIMUM_CACHE_EVICTION_MIB = 8 * 60
H200_POWER_WARMUP_GEMM_DIMENSION = 4096
H200_POWER_WARMUP_GEMM_ITERATIONS = 1000
H200_CLOCK_SETTLE_MAXIMUM_QUERIES = 20
H200_PRE_NVTX_SETTLE_SECONDS = 30.0
H200_PRE_NVTX_SAMPLE_INTERVAL_SECONDS = 0.25
H200_TELEMETRY_SCHEMA_VERSION = 1


# vLLM releases the H200 capture runners support.  Runners branch explicitly
# on this stack id (no try/except import fallback) and record it, plus the
# call path they took, in their launch evidence.
H200_VLLM_STACKS = {"0.24.0": "vllm024", "0.30.0": "vllm030"}


def installed_vllm_stack() -> str:
    """Return the H200 stack id of the installed vLLM; fail on any other."""

    version = importlib.metadata.version("vllm")
    release = version.split("+")[0]
    if release not in H200_VLLM_STACKS:
        raise RuntimeError(
            f"H200 capture runners support vLLM {sorted(H200_VLLM_STACKS)}, "
            f"installed {version!r}")
    return H200_VLLM_STACKS[release]


def query_sm_clock(physical_gpu_index: int) -> Mapping[str, Any]:
    """Return one fail-closed nvidia-smi SM clock/utilization sample."""

    if (isinstance(physical_gpu_index, bool)
            or not isinstance(physical_gpu_index, int)
            or physical_gpu_index < 0):
        raise ValueError("physical_gpu_index must be a nonnegative integer")
    fields = ("index", "clocks.sm", "clocks.mem", "utilization.gpu",
              "pstate", "power.draw", "clocks_event_reasons.active")
    completed = subprocess.run(
        ("nvidia-smi", f"--id={physical_gpu_index}",
         f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"),
        text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi query failed: stderr={completed.stderr!r}")
    rows = [line.strip() for line in completed.stdout.splitlines()
            if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"nvidia-smi returned {len(rows)} rows: {rows!r}")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"nvidia-smi field count mismatch: {rows[0]!r}")
    index = int(values[0])
    if index != physical_gpu_index:
        raise RuntimeError(
            f"nvidia-smi returned GPU {index}, expected {physical_gpu_index}")
    return {
        "epoch_ns": time.time_ns(),
        "physical_gpu_index": index,
        "sm_clock_mhz": int(values[1]),
        "memory_clock_mhz": int(values[2]),
        "utilization_gpu_pct": int(values[3]),
        "pstate": values[4],
        "power_w": float(values[5]),
        "clocks_event_reasons_active": values[6],
    }


class H200CaptureSession:
    """Lock, identity, power warm-up and clock telemetry for one runner."""

    def __init__(self, *, gpu_lock_path: str, telemetry_output: str,
                 physical_gpu_index: int, marker: str):
        for value, name in ((gpu_lock_path, "gpu_lock_path"),
                            (telemetry_output, "telemetry_output"),
                            (marker, "marker")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (isinstance(physical_gpu_index, bool)
                or not isinstance(physical_gpu_index, int)
                or physical_gpu_index < 0):
            raise ValueError("physical_gpu_index must be nonnegative")
        self._telemetry_output = Path(telemetry_output)
        self._physical_gpu_index = physical_gpu_index
        self._record: dict[str, Any] = {
            "schema_version": H200_TELEMETRY_SCHEMA_VERSION,
            "hardware_id": H200_HARDWARE_ID,
            "marker": marker,
            "physical_gpu_index": physical_gpu_index,
            "required_sm_clock_mhz": H200_REQUIRED_SM_CLOCK_MHZ,
            "process_start_epoch_ns": time.time_ns(),
        }
        self._lock_stream = open(gpu_lock_path, "a+")
        fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX)
        self._record["gpu_lock_acquired_epoch_ns"] = time.time_ns()

    def validate_device(self, torch_module: Any) -> None:
        properties = torch_module.cuda.get_device_properties(
            torch_module.cuda.current_device())
        if properties.name != H200_DEVICE_NAME:
            raise RuntimeError(
                f"H200 runner executed on {properties.name!r}")
        if properties.multi_processor_count != H200_SM_COUNT:
            raise RuntimeError(
                "H200 runner SM count mismatch: "
                f"{properties.multi_processor_count}")
        l2_bytes = properties.L2_cache_size
        if l2_bytes != H200_L2_BYTES:
            raise RuntimeError(f"H200 runner L2 size mismatch: {l2_bytes}")
        self._record["device"] = {
            "name": properties.name,
            "multi_processor_count": properties.multi_processor_count,
            "l2_cache_bytes": l2_bytes,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }

    def power_warmup(self, torch_module: Any) -> None:
        """Run the fixed BF16 GEMM power warm-up and synchronize."""

        if "power_warmup" in self._record:
            raise RuntimeError("H200 power warm-up may run only once")
        dimension = H200_POWER_WARMUP_GEMM_DIMENSION
        left = torch_module.randn(
            dimension, dimension, dtype=torch_module.bfloat16, device="cuda")
        right = torch_module.randn(
            dimension, dimension, dtype=torch_module.bfloat16, device="cuda")
        output = torch_module.empty(
            dimension, dimension, dtype=torch_module.bfloat16, device="cuda")
        torch_module.cuda.synchronize()
        start_ns = time.time_ns()
        for _ in range(H200_POWER_WARMUP_GEMM_ITERATIONS):
            torch_module.mm(left, right, out=output)
        torch_module.cuda.synchronize()
        end_ns = time.time_ns()
        del left, right, output
        # The dense GEMM trips the H200 SW power cap (clock dips to
        # 1590-1965 MHz); wait, GPU idle, until the boost clock is back so
        # the runner's own warm-up and NVTX call start from 1980 MHz.
        settle_samples = []
        for _ in range(H200_CLOCK_SETTLE_MAXIMUM_QUERIES):
            sample = query_sm_clock(self._physical_gpu_index)
            settle_samples.append(sample)
            if sample["sm_clock_mhz"] >= H200_REQUIRED_SM_CLOCK_MHZ:
                break
        self._record["power_warmup"] = {
            "kind": "bf16_gemm_power_warmup_before_runner_warmup",
            "matrix_dimension": dimension,
            "iterations": H200_POWER_WARMUP_GEMM_ITERATIONS,
            "start_epoch_ns": start_ns,
            "end_epoch_ns": end_ns,
            "included_in_nvtx_range": False,
            "clock_settle_samples": settle_samples,
        }

    def before_nvtx(self) -> None:
        if "power_warmup" not in self._record:
            raise RuntimeError("H200 NVTX range requires the power warm-up")
        # The runner's own warm-up may leave the GPU below its 1980-MHz boost
        # clock even after the earlier power warm-up settled.  Wait outside
        # the measured NVTX range for two spaced, power-cap-free samples.
        # A timeout still records the low clock; finish() keeps the existing
        # fail-closed before/after quality gate.
        deadline = time.monotonic() + H200_PRE_NVTX_SETTLE_SECONDS
        samples = []
        consecutive_ready = 0
        while True:
            sample = query_sm_clock(self._physical_gpu_index)
            samples.append(sample)
            power_capped = int(sample["clocks_event_reasons_active"], 16) & 0x4
            consecutive_ready = (consecutive_ready + 1
                                 if sample["sm_clock_mhz"] >= H200_REQUIRED_SM_CLOCK_MHZ
                                 and not power_capped else 0)
            if consecutive_ready >= 2 or time.monotonic() >= deadline:
                break
            time.sleep(H200_PRE_NVTX_SAMPLE_INTERVAL_SECONDS)
        self._record["pre_nvtx_clock_settle_samples"] = samples
        self._record["clock_before_nvtx"] = samples[-1]
        self._record["nvtx_push_epoch_ns"] = time.time_ns()

    def after_nvtx(self) -> None:
        self._record["nvtx_pop_synchronized_epoch_ns"] = time.time_ns()
        self._record["clock_after_nvtx"] = query_sm_clock(
            self._physical_gpu_index)

    def add_launch_evidence(self, name: str, payload: Any) -> None:
        if name in self._record:
            raise ValueError(f"duplicate H200 telemetry field: {name}")
        self._record[name] = payload

    def finish(self) -> None:
        for name in ("device", "power_warmup", "clock_before_nvtx",
                     "clock_after_nvtx"):
            if name not in self._record:
                raise RuntimeError(f"H200 capture session missing {name}")
        self._record["finish_epoch_ns"] = time.time_ns()
        before = self._record["clock_before_nvtx"]["sm_clock_mhz"]
        after = self._record["clock_after_nvtx"]["sm_clock_mhz"]
        self._record["clock_accepted"] = (
            before >= H200_REQUIRED_SM_CLOCK_MHZ
            and after >= H200_REQUIRED_SM_CLOCK_MHZ)
        self._telemetry_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._telemetry_output.with_suffix(
            self._telemetry_output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._record, indent=1, sort_keys=True),
            encoding="utf-8")
        os.replace(temporary, self._telemetry_output)


def add_h200_capture_arguments(parser: Any) -> None:
    parser.add_argument("--h200-gpu-lock", required=True)
    parser.add_argument("--h200-telemetry-output", required=True)
    parser.add_argument("--h200-physical-gpu", type=int, required=True)


def open_h200_capture_session(args: Any, marker: str) -> H200CaptureSession:
    return H200CaptureSession(
        gpu_lock_path=args.h200_gpu_lock,
        telemetry_output=args.h200_telemetry_output,
        physical_gpu_index=args.h200_physical_gpu,
        marker=marker,
    )
