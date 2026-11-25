#!/usr/bin/env python3
"""Capture one position-balanced A800 SiLU BF16 load-pair run."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time


HARDWARE_ID = "nvidia_a800"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("silu_bf16_load_pair.cu")
RUNNER_PATH = Path(__file__)
REPORT_PATH = Path(__file__).with_name("silu_bf16_load_pair_report.py")
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / HARDWARE_ID)
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
EXPECTED_COMPUTE_CAPABILITY = "8.0"
MAXIMUM_IDLE_DRIVER_MEMORY_MIB = 8
TRIALS = 9
WIDTHS = (6144, 9728, 12288, 14336, 25600, 28672, 53248)
GRIDS = (108, 216, 432)
VARIANTS = ("control", "single_bf16_load", "paired_bf16_load")
CELLS = tuple(
    (variant, grid, width)
    for width in WIDTHS for grid in GRIDS for variant in VARIANTS)
REQUIRED_INDEPENDENT_RUNS = 3
SCHEDULE_NAME = "silu_bf16_load_pair_cyclic_v1"
PRODUCTION_SYMBOL = (
    "_ZN4vllm18act_and_mul_kernelIN3c108BFloat16EXadL_ZNS_11silu_kernel"
    "IS2_EET_RKS4_EELb0EEEvPS4_PS5_i")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_ACTIVE_PROCESS = None
_DEFERRED_TERMINATION_SIGNAL = None


def cell_index_for(trial, order_position, run_rotation):
    return (
        order_position + 7 * trial
        + 20 * run_rotation
    ) % len(CELLS)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd, env=None):
    command = [str(item) for item in command]
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def _terminate_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _handle_termination(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128 + signum)


def _install_signal_handlers():
    for signum in TERMINATION_SIGNALS:
        signal.signal(signum, _handle_termination)


def _defer_termination(signum, _frame):
    global _DEFERRED_TERMINATION_SIGNAL

    if _DEFERRED_TERMINATION_SIGNAL is None:
        _DEFERRED_TERMINATION_SIGNAL = signum


def _start_native_process(native_command, environment):
    global _ACTIVE_PROCESS, _DEFERRED_TERMINATION_SIGNAL

    previous_handlers = tuple(
        (signum, signal.signal(signum, _defer_termination))
        for signum in TERMINATION_SIGNALS)
    try:
        process = subprocess.Popen(
            native_command, cwd=REPOSITORY_ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        _ACTIVE_PROCESS = process
    finally:
        for signum, previous_handler in previous_handlers:
            signal.signal(signum, previous_handler)
    if _DEFERRED_TERMINATION_SIGNAL is not None:
        signum = _DEFERRED_TERMINATION_SIGNAL
        _DEFERRED_TERMINATION_SIGNAL = None
        _handle_termination(signum, None)
    return process


def _validated_output(path):
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a child of {root}")
    if resolved.exists():
        raise FileExistsError(f"output already exists: {resolved}")
    return resolved


def _device_row(nvidia_smi, physical_device, *, require_idle):
    result = _run([
        nvidia_smi, f"--id={physical_device}",
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu,"
        "compute_cap,clocks.current.sm,clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT)
    rows = tuple(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 9:
        raise ValueError("nvidia-smi returned an unexpected device row")
    values = tuple(value.strip() for value in rows[0])
    row = {
        "index": int(values[0]),
        "name": values[1],
        "uuid": values[2],
        "memory_used_mib": int(values[3]),
        "utilization_gpu_pct": int(values[4]),
        "compute_capability": values[5],
        "sm_clock_mhz": int(values[6]),
        "memory_clock_mhz": int(values[7]),
        "driver_version": values[8],
    }
    if (row["index"] != physical_device
            or row["name"] != EXPECTED_DEVICE_NAME
            or row["compute_capability"] != EXPECTED_COMPUTE_CAPABILITY):
        raise RuntimeError(f"unexpected target GPU: {row}")
    applications = _run([
        nvidia_smi, "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT).stdout
    for line in applications.splitlines():
        fields = next(csv.reader([line], skipinitialspace=True))
        if fields and fields[0].strip() == row["uuid"]:
            raise RuntimeError(f"target GPU has a compute process: {line}")
    if require_idle and (
            row["memory_used_mib"] > MAXIMUM_IDLE_DRIVER_MEMORY_MIB
            or row["utilization_gpu_pct"] != 0):
        raise RuntimeError(f"target GPU is not idle: {row}")
    return row


def _wait_for_idle_device(nvidia_smi, physical_device):
    last_row = None
    for _ in range(20):
        try:
            return _device_row(
                nvidia_smi, physical_device, require_idle=True)
        except RuntimeError as error:
            last_row = error
            time.sleep(0.5)
    raise RuntimeError(
        "target GPU did not become idle within 10 seconds") from last_row


def _extract_function_sass(sass, symbol):
    marker = f"Function : {symbol}"
    start = sass.find(marker)
    if start < 0:
        raise ValueError(f"production SASS is missing {symbol}")
    next_function = sass.find("Function : ", start + len(marker))
    return sass[start:] if next_function < 0 else sass[start:next_function]


def _validate_production_sass(function_sass):
    instructions = tuple(
        line.strip() for line in function_sass.splitlines()
        if re.search(r"/\*[0-9a-f]{4}\*/", line))
    loads = tuple(
        index for index, line in enumerate(instructions)
        if "LDG.E.U16.CONSTANT" in line)
    if len(loads) != 2 or loads[1] != loads[0] + 1:
        raise ValueError("production SiLU must contain two adjacent BF16 loads")
    if "/*0170*/" not in instructions[loads[0]]:
        raise ValueError("production SiLU first load offset changed")
    if "/*0180*/" not in instructions[loads[1]]:
        raise ValueError("production SiLU second load offset changed")
    if not any("BRA 0xe0" in line for line in instructions):
        raise ValueError("production SiLU loop-back offset changed")
    return {
        "symbol": PRODUCTION_SYMBOL,
        "bf16_load_instruction_count": len(loads),
        "bf16_load_offsets": ["0x170", "0x180"],
        "loads_are_adjacent": True,
        "loop_back_offset": "0xe0",
    }


def _extract_production_artifact(
        production_binary, cuobjdump, build_dir):
    with tempfile.TemporaryDirectory(prefix="silu_prod_elf_") as temporary:
        temporary_path = Path(temporary)
        _run(
            [cuobjdump, "--extract-elf", "all", production_binary],
            cwd=temporary_path)
        matches = tuple(
            path for path in temporary_path.glob("*.sm_80.cubin")
            if PRODUCTION_SYMBOL.encode("ascii") in path.read_bytes())
        if len(matches) != 1:
            raise ValueError(
                "expected exactly one SM80 cubin containing production SiLU")
        cubin = build_dir / "production_silu_sm80.cubin"
        shutil.copy2(matches[0], cubin)
    sass = _run([cuobjdump, "--dump-sass", cubin], cwd=build_dir).stdout
    function_sass = _extract_function_sass(sass, PRODUCTION_SYMBOL)
    sass_path = build_dir / "production_silu_bf16.sass"
    sass_path.write_text(function_sass, encoding="utf-8")
    return cubin, sass_path, _validate_production_sass(function_sass)


def _parse_probe_resources(text):
    expected = {
        "silu_load_control": 11,
        "silu_single_bf16_load": 13,
        "silu_paired_bf16_load": 20,
    }
    found = {}
    current = None
    for line in text.splitlines():
        match = re.search(r"Function (silu_[A-Za-z0-9_]+):", line)
        if match is not None:
            current = match.group(1)
        register_match = re.search(r"REG:([0-9]+)", line)
        if current in expected and register_match is not None:
            found[current] = int(register_match.group(1))
    if found != expected:
        raise ValueError(f"probe resource contract changed: {found}")
    return found


def _parse_probe_sass(text):
    sections = {
        symbol: _extract_function_sass(text, symbol)
        for symbol in (
            "silu_load_control", "silu_single_bf16_load",
            "silu_paired_bf16_load")
    }
    counts = {
        symbol: section.count("LDG.E.U16.CONSTANT")
        for symbol, section in sections.items()}
    if counts != {
            "silu_load_control": 0,
            "silu_single_bf16_load": 1,
            "silu_paired_bf16_load": 2}:
        raise ValueError(f"probe load instruction contract changed: {counts}")
    paired_instructions = tuple(
        line.strip() for line in sections["silu_paired_bf16_load"].splitlines()
        if re.search(r"/\*[0-9a-f]{4}\*/", line))
    load_indices = tuple(
        index for index, line in enumerate(paired_instructions)
        if "LDG.E.U16.CONSTANT" in line)
    if load_indices[1] != load_indices[0] + 1:
        raise ValueError("probe paired loads are no longer adjacent")
    return {"bf16_load_instruction_counts": counts,
            "paired_loads_are_adjacent": True}


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rotation", required=True, type=int)
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--nvcc", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument(
        "--cuobjdump", type=Path,
        default=Path(shutil.which("cuobjdump") or "cuobjdump"))
    parser.add_argument(
        "--nvidia-smi", type=Path, default=Path("/usr/bin/nvidia-smi"))
    parser.add_argument(
        "--production-binary", type=Path,
        default=(Path(os.environ["MIDLAYER_A800_VLLM_BINARY"])
                 if "MIDLAYER_A800_VLLM_BINARY" in os.environ else None))
    args = parser.parse_args()
    if args.production_binary is None:
        parser.error("--production-binary or MIDLAYER_A800_VLLM_BINARY is required")
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.rotation not in range(REQUIRED_INDEPENDENT_RUNS):
        parser.error("--rotation must be 0, 1, or 2")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if RUN_ID_PATTERN.fullmatch(args.owner) is None:
        parser.error("--owner must match [A-Za-z0-9._-]{1,96}")
    paths = (args.nvcc, args.cuobjdump, args.nvidia_smi,
             args.production_binary, SOURCE_PATH, REPORT_PATH)
    if any(not path.resolve().is_file() for path in paths):
        parser.error("all tool, source, report, and production paths must exist")
    _install_signal_handlers()

    output = _validated_output(args.output)
    preflight = _wait_for_idle_device(
        args.nvidia_smi.resolve(), args.device)
    output.mkdir(parents=True)
    build_dir = output / "build"
    snapshot_dir = output / "source_snapshot"
    build_dir.mkdir()
    snapshot_dir.mkdir()
    for path in (SOURCE_PATH, RUNNER_PATH, REPORT_PATH):
        shutil.copy2(path, snapshot_dir / path.name)
    source_snapshot = snapshot_dir / SOURCE_PATH.name
    runner_snapshot = snapshot_dir / RUNNER_PATH.name
    report_snapshot = snapshot_dir / REPORT_PATH.name

    baseline = _run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT).stdout.strip()
    status = _run(
        ["git", "status", "--short"], cwd=REPOSITORY_ROOT).stdout
    binary = build_dir / "silu_bf16_load_pair"
    compile_command = [
        args.nvcc.resolve(), "-O3", "-std=c++17", "-arch=sm_80",
        "-Xptxas=-v", source_snapshot.name, "-o",
        Path("../build/silu_bf16_load_pair")]
    compile_result = _run(compile_command, cwd=snapshot_dir)
    (build_dir / "compile.stdout").write_text(
        compile_result.stdout, encoding="utf-8")
    (build_dir / "compile.stderr").write_text(
        compile_result.stderr, encoding="utf-8")
    resources_text = _run([
        args.cuobjdump.resolve(), "--dump-resource-usage", binary],
        cwd=REPOSITORY_ROOT).stdout
    resource_path = build_dir / "probe_resource_usage.txt"
    resource_path.write_text(resources_text, encoding="utf-8")
    resources = _parse_probe_resources(resources_text)
    probe_sass_text = _run([
        args.cuobjdump.resolve(), "--dump-sass", binary],
        cwd=REPOSITORY_ROOT).stdout
    probe_sass_path = build_dir / "probe.sass"
    probe_sass_path.write_text(probe_sass_text, encoding="utf-8")
    probe_sass_contract = _parse_probe_sass(probe_sass_text)
    production_cubin, production_sass, production_sass_contract = (
        _extract_production_artifact(
            args.production_binary.resolve(), args.cuobjdump.resolve(),
            build_dir))

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    native_command = [str(binary), "0", args.run_id, str(args.rotation)]
    process = _start_native_process(native_command, environment)
    try:
        stdout, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        raise RuntimeError(
            f"native PID {process.pid} exceeded 300 seconds") from error
    except BaseException:
        _terminate_process_group(process)
        raise
    finally:
        _ACTIVE_PROCESS = None
    if process.returncode != 0:
        raise RuntimeError(
            f"native PID {process.pid} failed ({process.returncode}): {stderr}")
    raw_path = output / "raw.csv"
    raw_path.write_text(stdout, encoding="utf-8")
    (output / "native.stderr").write_text(stderr, encoding="utf-8")
    postflight = _wait_for_idle_device(
        args.nvidia_smi.resolve(), args.device)

    manifest = {
        "schema": "nvidia_a800_silu_bf16_load_pair_capture_v1",
        "hardware_id": HARDWARE_ID,
        "owner": args.owner,
        "run_id": args.run_id,
        "run_rotation": args.rotation,
        "runner_pid": os.getpid(),
        "native_pid": process.pid,
        "baseline_commit": baseline,
        "tracked_worktree_status": status.splitlines(),
        "command": shlex.join([sys.executable, os.fspath(Path(__file__)),
                                *sys.argv[1:]]),
        "native_command": shlex.join(native_command),
        "device_preflight": preflight,
        "device_postflight": postflight,
        "schedule": SCHEDULE_NAME,
        "trials": TRIALS,
        "source_sha256": _sha256(source_snapshot),
        "runner_sha256": _sha256(runner_snapshot),
        "report_sha256": _sha256(report_snapshot),
        "probe_binary_sha256": _sha256(binary),
        "probe_resource_sha256": _sha256(resource_path),
        "probe_sass_sha256": _sha256(probe_sass_path),
        "probe_resources": resources,
        "probe_sass_contract": probe_sass_contract,
        "production_binary_path": str(args.production_binary.resolve()),
        "production_binary_sha256": _sha256(args.production_binary.resolve()),
        "production_cubin_sha256": _sha256(production_cubin),
        "production_sass_sha256": _sha256(production_sass),
        "production_sass_contract": production_sass_contract,
        "raw_sha256": _sha256(raw_path),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
