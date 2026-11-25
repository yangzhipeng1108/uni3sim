#!/usr/bin/env python3
"""Capture the independent FA2 local-stack spill-round service on H200.

H200 owner copy of microbench/nvidia_a800/memory/l1/
fa2_local_spill_round_runner.py.  The probe, rounds, trials, the
local-minus-register-control derivation (median over trials of per-trial
CTA-median cycles, one-round increment) and the one-/two-CTA agreement gate
are unchanged.  Hardware-coupling changes only:
  - device contract "NVIDIA A800-SXM4-80GB" -> "NVIDIA H200", -arch=sm_80 ->
    -arch=sm_90 (same CUDA 12.2 nvcc/cuobjdump, plus -cudart static);
  - SM count 108 -> 132, grids 108/216 -> 132/264;
  - output root profiler_output/microbench/nvidia_h200;
  - ``build`` (local: compile + probe static stack/site audit) and
    ``capture`` (remote H200) stages, explicit baseline commit (the remote
    copy is not a git checkout);
  - the pinned flash_attn 2.8.3 SM80 production cubin audit is not
    re-executed: that binary is an A800 (sm_80) artifact that is not
    installed on the H200 host.  The probe's own sm_90 static geometry is
    still audited against the same pinned stack/site contract, and the
    A800 production audit hashes are carried as provenance;
  - 50 ms ClockGate (unlocked SM clock); a capture whose logged live/busy
    SM clock falls below 1980 MHz is rejected and re-run (bounded).
"""

import argparse
import atexit
import datetime
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from statistics import median


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output/microbench/nvidia_h200"
SOURCE_PATH = Path(__file__).with_name("fa2_local_spill_round.cu")
RUNNER_PATH = Path(__file__).resolve()
NVCC = "/usr/local/cuda-12.2/bin/nvcc"
CUOBJDUMP = "/usr/local/cuda-12.2/bin/cuobjdump"
EXPECTED_DEVICE = "NVIDIA H200"
EXPECTED_SM_COUNT = 132
# A800 production audit identity (flash_attn 2.8.3 SM80 cubin), carried as
# provenance only; see the module docstring.
A800_PRODUCTION_BINARY_SHA256 = (
    "7c68f17400fac3f69c079842bd5304d3590e28dd5da850ff1465ff7ce86a1e93")
A800_PRODUCTION_SASS_SHA256 = {
    "d64_chunked": (
        "6ce176e0ccb0b11ba72a9ee6c0fbc0c48fafddd786b9d7a94dfc4d6ecbd225cf"),
    "d128_causal": (
        "2cc41bc199f2268cb97c083acc703a2b70f1b55e734009c9ed7b0062e932341f"),
}
PRODUCTION_CONTRACTS = {
    "d64_chunked": {
        "symbol": (
            "_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traits"
            "ILi64ELi64ELi256ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_"
            "kernel_traitsILi64ELi64ELi256ELi4ES3_EELb1ELb0ELb0ELb0ELb1E"
            "Lb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE"),
        "stack_bytes_per_thread": 56,
        "scalar_load_sites": 13,
        "scalar_store_sites": 13,
        "artifact_name": "production_d64_chunked",
        "mechanism": (
            "sm80_128_thread_cta_56_byte_stack_13_scalar_store_"
            "13_scalar_load_round"),
    },
    "d128_causal": {
        "symbol": (
            "_ZN5flash16flash_fwd_kernelI23Flash_fwd_kernel_traitsILi128E"
            "Li128ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_"
            "traitsILi128ELi128ELi64ELi4ES3_EELb0ELb1ELb0ELb0ELb1ELb1E"
            "Lb0ELb0EEEvNS_16Flash_fwd_paramsE"),
        "stack_bytes_per_thread": 176,
        "scalar_load_sites": 53,
        "scalar_store_sites": 51,
        "artifact_name": "production_d128_causal",
        "mechanism": (
            "sm80_128_thread_cta_176_byte_stack_51_scalar_store_"
            "53_scalar_load_round"),
    },
}
RAW_FIELDS = (
    "trial", "grid", "rounds", "resident_limit", "active_sms",
    "min_ctas", "max_ctas", "median_block_cycles", "max_sm_span_cycles",
    "checksum",
)
GRIDS = (132, 264)
ROUNDS = (0, 1, 4, 16, 64, 256)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd, capture_output=False, stdout=None, stderr=None):
    return subprocess.run(
        command, cwd=cwd, check=True, text=True,
        capture_output=capture_output, stdout=stdout, stderr=stderr,
        timeout=600)


def _validated_output(path):
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be below {root}")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def _device_rows():
    result = _run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT, capture_output=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise ValueError("unexpected nvidia-smi device row")
        rows.append({
            "index": int(row[0]), "name": row[1], "uuid": row[2],
            "pci_bus_id": row[3], "memory_used_mib": int(row[4]),
            "utilization_percent": int(row[5]),
        })
    return rows


def _compute_processes():
    result = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT, capture_output=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if row:
            if len(row) != 3:
                raise ValueError("unexpected compute-process row")
            rows.append((row[0], int(row[1]), int(row[2])))
    return rows


def _require_idle_device(index):
    matches = [row for row in _device_rows() if row["index"] == index]
    if len(matches) != 1:
        raise ValueError(f"device {index} does not resolve exactly once")
    device = matches[0]
    if device["name"] != EXPECTED_DEVICE:
        raise ValueError(f"unexpected device: {device['name']}")
    if device["memory_used_mib"] > 8 or device["utilization_percent"] != 0:
        raise RuntimeError(f"device {index} is not idle: {device}")
    processes = [row for row in _compute_processes()
                 if row[0] == device["uuid"]]
    if processes:
        raise RuntimeError(f"device {index} has compute processes: {processes}")
    return device


def _wait_for_idle_device(index):
    last_error = None
    for _ in range(10):
        try:
            return _require_idle_device(index)
        except RuntimeError as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"device {index} did not return idle: {last_error}")


def _local_site_counts(sass):
    return {
        "scalar_load_sites": len(re.findall(r"\sLDL(?:\.|\s)", sass)),
        "scalar_store_sites": len(re.findall(r"\sSTL(?:\.|\s)", sass)),
    }


def _compile(output, source, *, control, specialization):
    name = "register_control" if control else "local_spill"
    binary = output / "build" / name
    command = [
        NVCC, "-O3", "-std=c++17", "-arch=sm_90", "-cudart", "static", "-lineinfo",
        "-Xptxas=-O0,-v",
    ]
    if control:
        command.append("-DREGISTER_CONTROL")
    if specialization == "d128_causal":
        command.append("-DD128_CAUSAL")
    command.extend((str(source), "-o", str(binary)))
    with (output / "build" / f"{name}.compile.stdout").open("w") as stdout, (
            output / "build" / f"{name}.compile.stderr").open("w") as stderr:
        _run(command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    resources = _run(
        [CUOBJDUMP, "--dump-resource-usage", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True).stdout
    sass = _run(
        [CUOBJDUMP, "--dump-sass", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True).stdout
    (output / "build" / f"{name}.resources.txt").write_text(resources)
    sass_path = output / "build" / f"{name}.sass"
    sass_path.write_text(sass)
    stack_matches = re.findall(r"STACK:(\d+)", resources)
    if len(stack_matches) != 1:
        raise ValueError(f"{name} resource record changed")
    sites = _local_site_counts(sass)
    contract = PRODUCTION_CONTRACTS[specialization]
    expected = {
        "stack_bytes_per_thread": (
            0 if control else contract["stack_bytes_per_thread"]),
        "scalar_load_sites": (
            0 if control else contract["scalar_load_sites"]),
        "scalar_store_sites": (
            0 if control else contract["scalar_store_sites"]),
    }
    actual = {"stack_bytes_per_thread": int(stack_matches[0]), **sites}
    if actual != expected:
        raise ValueError(f"{name} static contract changed: {actual}")
    return binary, sass_path, actual


def _capture(binary, output, device, trials, name):
    raw = output / f"{name}.csv"
    with raw.open("w") as stdout, (output / f"{name}.stderr").open("w") as stderr:
        _run([str(binary), str(device), str(trials)],
             cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    with raw.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"{name} raw header changed")
        rows = list(reader)
    if len(rows) != trials * len(GRIDS) * len(ROUNDS):
        raise ValueError(f"{name} raw row count changed")
    normalized = []
    for row in rows:
        value = {field: int(row[field]) for field in RAW_FIELDS}
        if value["grid"] not in GRIDS or value["rounds"] not in ROUNDS:
            raise ValueError(f"{name} contains an unknown cell")
        expected_ctas = value["grid"] // EXPECTED_SM_COUNT
        if (value["active_sms"] != EXPECTED_SM_COUNT
                or value["min_ctas"] != expected_ctas
                or value["max_ctas"] != expected_ctas):
            raise ValueError(f"{name} CTA assignment changed")
        if value["resident_limit"] < 2:
            raise ValueError(f"{name} cannot admit two resident CTAs")
        normalized.append(value)
    return raw, normalized


def _median(rows, grid, rounds):
    values = [row["median_block_cycles"] for row in rows
              if row["grid"] == grid and row["rounds"] == rounds]
    if not values:
        raise ValueError("missing measurement cell")
    if (max(values) - min(values)) / median(values) > 0.05:
        raise ValueError("measurement cell is unstable")
    return float(median(values))


# --- H200 clock gate (hardware coupling: the H200 SM clock cannot be locked
# in this container).  A 50 ms nvidia-smi logger runs for the whole capture;
# every cell's wall-clock window is judged afterwards.  A cell is rejected
# when any busy sample (utilization > 0) inside its window, or any sample
# while its CUDA context is alive, reports an SM clock below the 1980 MHz
# boost clock; rejected cells are re-run (bounded passes) and counted.
CLOCK_TARGET_MHZ = 1980
CLOCK_MAX_PASSES = 4


class ClockGate:
    def __init__(self, device_id, log_path):
        self.log_path = Path(log_path)
        # nvidia-smi -f block-buffers its file; line-buffered stdout keeps
        # the log current so cells can be judged during the capture.
        self._stream = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen([
            "stdbuf", "-oL", "nvidia-smi", "-i", str(device_id),
            "--query-gpu=timestamp,clocks.sm,utilization.gpu",
            "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=self._stream, stderr=subprocess.DEVNULL)
        self.pid = self.process.pid
        # Stop the logger on every interpreter exit, including failures.
        atexit.register(self.stop)
        deadline = time.time() + 10.0
        while not self._samples_after(time.time() - 0.5):
            if time.time() > deadline:
                raise RuntimeError("clock logger produced no samples")
            time.sleep(0.1)

    def _samples(self):
        samples = []
        with self.log_path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 3 or not fields[1].isdigit():
                    continue
                stamp = datetime.datetime.strptime(
                    fields[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                samples.append((stamp, int(fields[1]), int(fields[2])))
        return samples

    def _samples_after(self, stamp):
        if not self.log_path.exists():
            return []
        return [sample for sample in self._samples() if sample[0] > stamp]

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=30)
        self._stream.close()

    def judge(self, windows):
        """windows: {key: (start_s, end_s)} -> ({key: verdict}, samples)."""
        deadline = time.time() + 10.0
        last_end = max(end for _, end in windows.values())
        while not self._samples_after(last_end + 0.3):
            if time.time() > deadline:
                raise RuntimeError("clock logger stopped advancing")
            time.sleep(0.1)
        samples = self._samples()
        verdicts = {}
        for key, (start, end) in windows.items():
            inside = [s for s in samples if start <= s[0] <= end]
            busy = [s for s in samples
                    if start <= s[0] <= end + 0.25 and s[2] > 0]
            context = [s for s in inside if s[1] >= CLOCK_TARGET_MHZ]
            # Samples before the context raises the clock (process start-up)
            # are not busy; after the first boosted sample every sample in
            # the window belongs to the live context.
            live = ([s for s in inside if s[0] >= context[0][0]]
                    if context else [])
            low = [s for s in busy + live if s[1] < CLOCK_TARGET_MHZ]
            verdicts[key] = {
                "accepted": bool(context) and not low,
                "samples_in_window": len(inside),
                "busy_samples": len(busy),
                "min_live_or_busy_clock_mhz": (
                    min(s[1] for s in busy + live) if busy + live else None),
            }
        return verdicts


def _clock_summary(gate, windows):
    samples = gate._samples()
    live = [s[1] for s in samples if s[2] > 0]
    return {
        "logger": "nvidia-smi -lms 50 timestamp,clocks.sm,utilization.gpu",
        "logger_pid": gate.pid,
        "log_path": str(gate.log_path.name),
        "target_sm_clock_mhz": CLOCK_TARGET_MHZ,
        "total_samples": len(samples),
        "busy_samples": len(live),
        "busy_sm_clock_mhz_min": min(live) if live else None,
        "busy_sm_clock_mhz_max": max(live) if live else None,
        "busy_sm_clock_mhz_mean": (
            sum(live) / len(live) if live else None),
    }
# --- end H200 clock gate


def build(args):
    output = _validated_output(args.output)
    output.mkdir(parents=True)
    (output / "build").mkdir()
    (output / "source_snapshot").mkdir()
    source = output / "source_snapshot" / SOURCE_PATH.name
    runner = output / "source_snapshot" / RUNNER_PATH.name
    shutil.copy2(SOURCE_PATH, source)
    shutil.copy2(RUNNER_PATH, runner)
    local_binary, local_sass, local_static = _compile(
        output, source, control=False,
        specialization=args.specialization)
    control_binary, control_sass, control_static = _compile(
        output, source, control=True,
        specialization=args.specialization)
    (output / "build" / "build_manifest.json").write_text(json.dumps({
        "specialization": args.specialization,
        "source_sha256": _sha256(source),
        "runner_sha256": _sha256(runner),
        "local_binary_sha256": _sha256(local_binary),
        "local_sass_sha256": _sha256(local_sass),
        "control_binary_sha256": _sha256(control_binary),
        "control_sass_sha256": _sha256(control_sass),
        "local_static": local_static,
        "control_static": control_static,
        "nvcc": NVCC,
        "nvcc_version": _run(
            [NVCC, "--version"], cwd=REPOSITORY_ROOT,
            capture_output=True).stdout.strip(),
    }, indent=2, sort_keys=True) + "\n")


def _gated_capture(gate, binary, output, trials, name, passes):
    for pass_index in range(CLOCK_MAX_PASSES):
        start = time.time()
        raw, rows = _capture(binary, output, 0, trials, name)
        verdict = gate.judge({name: (start, time.time())})[name]
        passes.append({"capture": name, "pass": pass_index, **verdict})
        if verdict["accepted"]:
            return raw, rows
    raise RuntimeError(f"{name} capture never passed the clock gate")


def capture(args):
    output = args.output.resolve()
    root = OUTPUT_ROOT.resolve()
    if output == root or root not in output.parents or not output.is_dir():
        raise ValueError(f"build output must exist below {root}")
    build_manifest = json.loads(
        (output / "build" / "build_manifest.json").read_text())
    specialization = build_manifest["specialization"]
    source = output / "source_snapshot" / SOURCE_PATH.name
    local_binary = output / "build" / "local_spill"
    control_binary = output / "build" / "register_control"
    if (_sha256(source) != build_manifest["source_sha256"]
            or _sha256(SOURCE_PATH) != build_manifest["source_sha256"]
            or _sha256(local_binary) != build_manifest["local_binary_sha256"]
            or _sha256(control_binary)
            != build_manifest["control_binary_sha256"]):
        raise ValueError("build artifacts differ from the build manifest")
    if (output / "report.json").exists():
        raise FileExistsError("capture already exists")
    device = _require_idle_device(args.device)
    # The binary receives CUDA ordinal 0 inside a single-GPU visibility mask
    # so the physical nvidia-smi index and the CUDA ordinal cannot diverge.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    gate = ClockGate(args.device, output / "clock_log.csv")
    passes = []
    local_raw, local_rows = _gated_capture(
        gate, local_binary, output, args.trials, "local", passes)
    control_raw, control_rows = _gated_capture(
        gate, control_binary, output, args.trials, "control", passes)
    clock = _clock_summary(gate, {})
    gate.stop()
    increments = {}
    for grid in GRIDS:
        increments[str(grid)] = {
            str(rounds): _median(local_rows, grid, rounds)
            - _median(control_rows, grid, rounds)
            for rounds in ROUNDS
        }
    one_round = tuple(increments[str(grid)]["1"] for grid in GRIDS)
    steady = tuple(
        (increments[str(grid)]["256"] - increments[str(grid)]["64"])
        / (256 - 64) for grid in GRIDS)
    if (not math.isclose(one_round[0], one_round[1], abs_tol=1.0)
            or not math.isclose(
                steady[0], steady[1], abs_tol=4.0 / (256 - 64))):
        raise ValueError(
            "one-CTA and two-CTA spill services disagree beyond clock64 "
            "median endpoint resolution: "
            f"one_round={one_round}, steady={steady}")
    if one_round[0] <= 0 or steady[0] <= 0:
        raise ValueError("spill service must be positive")
    contract = PRODUCTION_CONTRACTS[specialization]
    local_static = build_manifest["local_static"]
    control_static = build_manifest["control_static"]
    report = {
        "schema": "nvidia_h200_local_spill_round_candidate_v2",
        "hardware_id": "nvidia_h200", "device_index": args.device,
        "device_uuid": device["uuid"], "baseline_commit": args.baseline_commit,
        "owner": args.owner, "specialization": specialization,
        "mechanism": contract["mechanism"],
        "production_static_contract": {
            "status": "not_rechecked_on_h200",
            "reason": (
                "the pinned flash_attn 2.8.3 SM80 production cubin is an "
                "A800 artifact not installed on the H200 host; the probe "
                "models that SM80 source program unchanged and its own sm_90 "
                "build is audited against the same stack/site contract"),
            "a800_binary_sha256": A800_PRODUCTION_BINARY_SHA256,
            "a800_sass_sha256": A800_PRODUCTION_SASS_SHA256[specialization],
            "registers_per_thread": 255,
            "stack_bytes_per_thread": contract["stack_bytes_per_thread"],
            "scalar_load_sites": contract["scalar_load_sites"],
            "scalar_store_sites": contract["scalar_store_sites"],
        },
        "probe_static_contract": {
            "source_sha256": build_manifest["source_sha256"],
            "local_binary_sha256": build_manifest["local_binary_sha256"],
            "local_sass_sha256": build_manifest["local_sass_sha256"],
            "control_binary_sha256": build_manifest["control_binary_sha256"],
            "control_sass_sha256": build_manifest["control_sass_sha256"],
            **local_static,
            "control_stack_bytes_per_thread": control_static[
                "stack_bytes_per_thread"],
            "control_local_sites": (
                control_static["scalar_load_sites"]
                + control_static["scalar_store_sites"]),
        },
        "measurement": {
            "timer_domain": "clock64_same_cta_envelope_cycles",
            "trials": args.trials, "grid_blocks": list(GRIDS),
            "rounds": list(ROUNDS), "local_minus_control_cycles": increments,
            "one_round_local_minus_register_control_cycles": one_round[0],
            "steady_local_minus_register_control_cycles_per_round": steady[0],
            "one_cta_two_cta_max_clock64_tick_difference": max(
                abs(one_round[0] - one_round[1]),
                abs(steady[0] - steady[1]) * (256 - 64)),
            "local_raw_sha256": _sha256(local_raw),
            "control_raw_sha256": _sha256(control_raw),
        },
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "A source-matched local stack with the recorded scalar store/load "
            "site geometry adds the measured SM cycles for its single "
            "complete 128-thread CTA spill round."),
        "excluded_conclusions": [
            "production FlashAttention duration", "per-shape latency correction",
            "local spill service for a different stack or SASS site geometry",
        ],
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "pid": os.getpid(),
            "runner_sha256": build_manifest["runner_sha256"],
            "nvcc": build_manifest["nvcc"],
            "clock_gate": {**clock, "passes": passes},
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    _wait_for_idle_device(args.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "capture"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--owner", default="h200-port-wp6_fa2dep")
    parser.add_argument("--baseline-commit")
    parser.add_argument(
        "--specialization", choices=tuple(PRODUCTION_CONTRACTS),
        default="d64_chunked")
    args = parser.parse_args()
    if args.device < 0 or not 3 <= args.trials <= 11:
        raise ValueError("device must be nonnegative and trials must be in [3, 11]")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", args.owner):
        raise ValueError("invalid owner")
    if args.stage == "build":
        build(args)
    else:
        if args.baseline_commit is None:
            raise ValueError("capture requires --baseline-commit")
        capture(args)


if __name__ == "__main__":
    main()
