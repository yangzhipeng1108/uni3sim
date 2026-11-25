#!/usr/bin/env python3
"""Capture the source-exact A800 memory-dependency multi-wave campaign."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import sqlite3
import statistics
import subprocess
from typing import NamedTuple


EXPECTED_TRIALS = (1, 2, 3)
EXPECTED_GRIDS = (256, 512, 1024, 2048, 4096)
EXPECTED_BLOCKS = (128, 256)
EXPECTED_SAMPLES_PER_CELL = 200
EXPECTED_HARDWARE_NAME = "NVIDIA A800-SXM4-80GB"
DEFAULT_NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
DEFAULT_CUOBJDUMP = Path("/usr/local/cuda-12.2/bin/cuobjdump")
DEFAULT_NSYS = Path(shutil.which("nsys") or "nsys")
DEFAULT_NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
MEASUREMENT_FIELDS = (
    "trial", "seq", "nvtx_range", "grid", "block", "duration_ns",
    "correlationId", "range_start", "range_end", "launch_start",
    "launch_end", "kernel_start", "kernel_end")
SUMMARY_FIELDS = (
    "trial", "grid", "block", "sample_count", "median_duration_ns")
DEVICE_QUERY = (
    "index,uuid,pci.bus_id,name,memory.used,utilization.gpu,clocks.mem,"
    "clocks.sm,pstate,driver_version")
TELEMETRY_QUERY = (
    "timestamp,index,uuid,memory.used,utilization.gpu,clocks.mem,"
    "clocks.sm,pstate")
COMPUTE_PROCESS_QUERY = "gpu_uuid,pid,process_name,used_gpu_memory"
AGGREGATE_SQL = """
SELECT nvtx.text, nvtx.start, nvtx.end,
       runtime.correlationId, runtime.start, runtime.end,
       kernel.gridX, kernel.gridY, kernel.gridZ,
       kernel.blockX, kernel.blockY, kernel.blockZ,
       kernel.start, kernel.end, kernel_name.value, runtime_name.value
FROM NVTX_EVENTS AS nvtx
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
  ON runtime.start >= nvtx.start AND runtime.end <= nvtx.end
JOIN StringIds AS runtime_name ON runtime_name.id = runtime.nameId
JOIN CUPTI_ACTIVITY_KIND_KERNEL AS kernel
  ON kernel.correlationId = runtime.correlationId
JOIN StringIds AS kernel_name ON kernel_name.id = kernel.shortName
WHERE nvtx.text GLOB 'memory_dependency_grid_*_block_*'
  AND runtime_name.value LIKE 'cudaLaunchKernel%'
  AND kernel_name.value LIKE '%memory_dependency_task%'
ORDER BY nvtx.start
""".strip()
MANAGED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_ACTIVE_PROCESSES = {}


class ToolPaths(NamedTuple):
    nvcc: Path
    cuobjdump: Path
    nsys: Path
    nvidia_smi: Path


def _unblock_managed_signals_in_child():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, MANAGED_SIGNALS)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_tool(path, name):
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{name} tool does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} tool is not a regular file: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise ValueError(f"{name} tool is not executable: {resolved}")
    return resolved


def _validated_tools(args):
    return ToolPaths(
        nvcc=_validated_tool(args.nvcc, "nvcc"),
        cuobjdump=_validated_tool(args.cuobjdump, "cuobjdump"),
        nsys=_validated_tool(args.nsys, "nsys"),
        nvidia_smi=_validated_tool(args.nvidia_smi, "nvidia-smi"),
    )


def _nvcc_command(nvcc, source, executable):
    return (
        str(nvcc), "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo",
        str(source), "-lnvToolsExt", "-o", str(executable),
    )


def _cuobjdump_command(cuobjdump, action, executable):
    if action not in ("--dump-sass", "--dump-resource-usage"):
        raise ValueError(f"unsupported cuobjdump action: {action}")
    return (str(cuobjdump), action, str(executable))


def _nsys_profile_command(nsys, prefix, executable):
    return (
        str(nsys), "profile", "--trace=cuda,nvtx", "--sample=none",
        "--cpuctxsw=none", "--force-overwrite=true", f"--output={prefix}",
        str(executable), "--sweep",
    )


def _nsys_export_command(nsys, sqlite_path, report_path):
    return (
        str(nsys), "export", "--type=sqlite", "--force-overwrite=true",
        f"--output={sqlite_path}", str(report_path),
    )


def _nvidia_smi_command(nvidia_smi, query, device, *options):
    return (
        str(nvidia_smi), f"--query-gpu={query}",
        "--format=csv,noheader,nounits", *options, "-i", str(device),
    )


def _nvidia_smi_compute_command(nvidia_smi, device):
    return (
        str(nvidia_smi), f"--query-compute-apps={COMPUTE_PROCESS_QUERY}",
        "--format=csv,noheader,nounits", "-i", str(device),
    )


def _nvidia_smi_version_command(nvidia_smi, device):
    return (str(nvidia_smi), "-i", str(device))


def _nvidia_smi_version(text):
    matches = re.findall(r"\bNVIDIA-SMI\s+([0-9]+(?:\.[0-9]+)+)\b", text)
    if len(matches) != 1:
        raise ValueError("nvidia-smi banner lacks one exact tool version")
    return matches[0]


def _start_process(command, *, cwd, env=None, stdout=None, stderr=None):
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, MANAGED_SIGNALS)
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, text=True, stdout=stdout,
            stderr=stderr, start_new_session=True,
            preexec_fn=_unblock_managed_signals_in_child)
        _ACTIVE_PROCESSES[process.pid] = process
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    return process


def _unregister_process(process):
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, MANAGED_SIGNALS)
    try:
        del _ACTIVE_PROCESSES[process.pid]
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _terminate_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            process.wait()
            return
        process.wait(timeout=2)


def _terminate_and_unregister(process):
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, MANAGED_SIGNALS)
    try:
        _terminate_process_group(process)
    finally:
        del _ACTIVE_PROCESSES[process.pid]
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _run(command, *, cwd, env=None, stdout=None, stderr=None):
    process = _start_process(
        command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    try:
        output, error = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=output, stderr=error)
        return subprocess.CompletedProcess(
            command, process.returncode, output, error)
    except BaseException:
        _terminate_and_unregister(process)
        raise
    else:
        _unregister_process(process)


def _handle_signal(signum, _frame):
    raise SystemExit(128 + signum)


@contextmanager
def _supervise_children():
    if _ACTIVE_PROCESSES:
        raise RuntimeError("child-process supervisor entered while active")
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in MANAGED_SIGNALS}
    for signum in MANAGED_SIGNALS:
        signal.signal(signum, _handle_signal)
    try:
        yield
    finally:
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, MANAGED_SIGNALS)
        try:
            for process in tuple(_ACTIVE_PROCESSES.values()):
                _terminate_and_unregister(process)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _output_path(repo, value):
    output = Path(value).resolve()
    profiler_root = (repo / "profiler_output").resolve()
    try:
        output.relative_to(profiler_root)
    except ValueError as error:
        raise ValueError(
            "capture output must be below the repository profiler_output") \
            from error
    if output.exists():
        raise FileExistsError(output)
    return output


def _query_device(repo, nvidia_smi, device):
    completed = _run(
        _nvidia_smi_command(nvidia_smi, DEVICE_QUERY, device),
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rows = list(csv.reader(completed.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 10:
        raise ValueError("nvidia-smi returned an unexpected device record")
    fields = tuple(field.strip() for field in rows[0])
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "pci_bus_id": fields[2],
        "name": fields[3],
        "memory_used_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
        "memory_clock_mhz": int(fields[6]),
        "sm_clock_mhz": int(fields[7]),
        "pstate": fields[8],
        "driver_version": fields[9],
    }


def _query_compute_processes(repo, nvidia_smi, device):
    return _run(
        _nvidia_smi_compute_command(nvidia_smi, device), cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _require_no_compute_processes(text):
    if text.strip():
        raise ValueError("target A800 has an active compute process")


def _require_idle_device(
        identity, device, expected_uuid, expected_memory_used_mib):
    if (identity["index"] != device
            or identity["uuid"] != expected_uuid
            or identity["name"] != EXPECTED_HARDWARE_NAME):
        raise ValueError("target A800 identity differs from the request")
    if (identity["memory_used_mib"] != expected_memory_used_mib
            or identity["utilization_percent"] != 0
            or identity["pstate"] != "P0"):
        raise ValueError(
            "target A800 must match its frozen idle-memory baseline, have "
            "zero utilization, and be in P0")


def _write_device_row(path, identity, trial):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"trial{trial}|0|{identity['index']}|"
            f"{identity['uuid'].removeprefix('GPU-')}|"
            f"{identity['pci_bus_id']}|{identity['name']}\n")


def _capture_trial(
        repo, output, executable, tools, device, expected_uuid,
        idle_memory_used_mib, trial):
    run_dir = output / "run"
    telemetry_dir = output / "telemetry"
    identity = _query_device(repo, tools.nvidia_smi, device)
    _require_idle_device(
        identity, device, expected_uuid, idle_memory_used_mib)
    preflight_processes = _query_compute_processes(
        repo, tools.nvidia_smi, device)
    _require_no_compute_processes(preflight_processes)
    (telemetry_dir / f"trial{trial}_preflight_compute_processes.csv").write_text(
        preflight_processes, encoding="utf-8")
    _write_device_row(run_dir / "device_identity.txt", identity, trial)
    live_path = telemetry_dir / f"trial{trial}_live_clock.csv"
    live_stderr_path = telemetry_dir / f"trial{trial}_live_clock.stderr"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    prefix = run_dir / f"memory_dependency_trial{trial}"
    with live_path.open("w") as live_stdout, live_stderr_path.open(
            "w") as live_stderr:
        telemetry = _start_process(
            _nvidia_smi_command(
                tools.nvidia_smi, TELEMETRY_QUERY, device, "--loop-ms=100"),
            cwd=repo, stdout=live_stdout, stderr=live_stderr)
        try:
            with (run_dir / f"trial{trial}.stdout").open("w") as stdout, (
                    run_dir / f"trial{trial}.stderr").open("w") as stderr:
                _run(
                    _nsys_profile_command(tools.nsys, prefix, executable),
                    cwd=repo, env=environment, stdout=stdout, stderr=stderr)
        finally:
            _terminate_and_unregister(telemetry)
    report_path = prefix.with_suffix(".nsys-rep")
    if not report_path.is_file():
        raise ValueError(f"nsys did not create {report_path}")
    sqlite_path = prefix.with_suffix(".sqlite")
    with (run_dir / f"trial{trial}_export.stdout").open("w") as stdout, (
            run_dir / f"trial{trial}_export.stderr").open("w") as stderr:
        _run(
            _nsys_export_command(tools.nsys, sqlite_path, report_path),
            cwd=repo, stdout=stdout, stderr=stderr)
    postflight = _query_device(repo, tools.nvidia_smi, device)
    _require_idle_device(
        postflight, device, expected_uuid, idle_memory_used_mib)
    postflight_processes = _query_compute_processes(
        repo, tools.nvidia_smi, device)
    _require_no_compute_processes(postflight_processes)
    (telemetry_dir / f"trial{trial}_postflight_compute_processes.csv").write_text(
        postflight_processes, encoding="utf-8")
    with (telemetry_dir / f"trial{trial}_postflight.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow((
            postflight["index"], postflight["uuid"],
            postflight["memory_used_mib"],
            postflight["utilization_percent"],
            postflight["memory_clock_mhz"], postflight["sm_clock_mhz"],
            postflight["pstate"],
        ))


def _parse_geometry(label):
    match = re.fullmatch(
        r"memory_dependency_grid_([1-9][0-9]*)_block_([1-9][0-9]*)",
        label)
    if match is None:
        raise ValueError(f"unexpected memory-task NVTX label: {label!r}")
    return int(match.group(1)), int(match.group(2))


def _read_trial(output, trial):
    path = output / "run" / f"memory_dependency_trial{trial}.sqlite"
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        raw_rows = connection.execute(AGGREGATE_SQL).fetchall()
    finally:
        connection.close()
    rows = []
    identities = set()
    correlations = set()
    for sequence, raw in enumerate(raw_rows, start=1):
        (label, range_start, range_end, correlation, launch_start,
         launch_end, grid_x, grid_y, grid_z, block_x, block_y, block_z,
         kernel_start, kernel_end, kernel_name, runtime_name) = raw
        grid, block = _parse_geometry(label)
        if (grid not in EXPECTED_GRIDS or block not in EXPECTED_BLOCKS
                or (grid_x, grid_y, grid_z) != (grid, 1, 1)
                or (block_x, block_y, block_z) != (block, 1, 1)
                or "memory_dependency_task" not in kernel_name
                or not runtime_name.startswith("cudaLaunchKernel")):
            raise ValueError(
                f"trial {trial} SQLite geometry or symbol drifted")
        identity = (grid, block)
        correlations.add(correlation)
        identities.add(identity)
        rows.append({
            "trial": trial,
            "seq": sequence,
            "nvtx_range": label,
            "grid": grid,
            "block": block,
            "duration_ns": kernel_end - kernel_start,
            "correlationId": correlation,
            "range_start": range_start,
            "range_end": range_end,
            "launch_start": launch_start,
            "launch_end": launch_end,
            "kernel_start": kernel_start,
            "kernel_end": kernel_end,
        })
    expected_rows = (
        len(EXPECTED_GRIDS) * len(EXPECTED_BLOCKS)
        * EXPECTED_SAMPLES_PER_CELL)
    expected_cells = {
        (grid, block) for grid in EXPECTED_GRIDS for block in EXPECTED_BLOCKS}
    if (len(rows) != expected_rows or len(correlations) != expected_rows
            or identities != expected_cells):
        raise ValueError(f"trial {trial} SQLite sweep is incomplete")
    return rows


def _write_aggregates(output):
    all_rows = []
    validation = []
    for trial in EXPECTED_TRIALS:
        rows = _read_trial(output, trial)
        all_rows.extend(rows)
        validation.append(
            f"{trial}|{len(rows)}|"
            f"{len(EXPECTED_GRIDS) * len(EXPECTED_BLOCKS)}|0|0|0|0\n")
    run_dir = output / "run"
    measurements = run_dir / "measurements_3trials.csv"
    with measurements.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = run_dir / "summary_by_trial.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for trial in EXPECTED_TRIALS:
            for grid in EXPECTED_GRIDS:
                for block in EXPECTED_BLOCKS:
                    durations = [
                        row["duration_ns"] for row in all_rows
                        if row["trial"] == trial and row["grid"] == grid
                        and row["block"] == block]
                    if len(durations) != EXPECTED_SAMPLES_PER_CELL:
                        raise ValueError("summary cell is incomplete")
                    writer.writerow({
                        "trial": trial,
                        "grid": grid,
                        "block": block,
                        "sample_count": len(durations),
                        "median_duration_ns": statistics.median(durations),
                    })
    (run_dir / "validation_3trials.txt").write_text(
        "".join(validation), encoding="utf-8")
    return summary, measurements


def _write_manifest(output):
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{_sha256(path)}  {path.resolve()}\n")
    (output / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--expected-device-uuid", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--nvcc", type=Path, default=DEFAULT_NVCC)
    parser.add_argument("--cuobjdump", type=Path, default=DEFAULT_CUOBJDUMP)
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--nvidia-smi", type=Path, default=DEFAULT_NVIDIA_SMI)
    args = parser.parse_args(argv)
    if args.device < 0 or not args.owner:
        raise ValueError("device must be nonnegative and owner nonempty")
    tools = _validated_tools(args)
    runner = Path(__file__).resolve()
    source = runner.with_name("memory_dependency_task_duration.cu")
    reporter = runner.with_name("memory_dependency_task_duration_report.py")
    repo = runner.parents[4]
    output = _output_path(repo, args.output)
    commit = _run(
        ["git", "rev-parse", "HEAD"], cwd=repo,
        stdout=subprocess.PIPE).stdout.strip()
    status = _run(
        ["git", "status", "--short", "--untracked-files=no"], cwd=repo,
        stdout=subprocess.PIPE).stdout
    if status:
        raise ValueError("tracked worktree must be clean before capture")
    identity = _query_device(repo, tools.nvidia_smi, args.device)
    idle_memory_used_mib = identity["memory_used_mib"]
    if idle_memory_used_mib < 0:
        raise ValueError("target A800 idle-memory baseline is negative")
    _require_idle_device(
        identity, args.device, args.expected_device_uuid,
        idle_memory_used_mib)
    preflight_processes = _query_compute_processes(
        repo, tools.nvidia_smi, args.device)
    _require_no_compute_processes(preflight_processes)
    output.mkdir(parents=True)
    for path in (
            output / "build", output / "run",
            output / "source_snapshot", output / "telemetry"):
        path.mkdir()
    snapshot = output / "source_snapshot"
    for path in (source, reporter, runner):
        shutil.copy2(path, snapshot / path.name)
    (output / "baseline_commit.txt").write_text(
        commit + "\n", encoding="utf-8")
    with (output / "telemetry/preflight.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow((
            identity["index"], identity["uuid"], identity["name"],
            identity["memory_used_mib"], identity["utilization_percent"],
            identity["memory_clock_mhz"], identity["sm_clock_mhz"],
            identity["pstate"],
        ))
    (output / "telemetry/preflight_compute_processes.csv").write_text(
        preflight_processes, encoding="utf-8")
    executable = output / "build/memory_dependency_task_duration"
    with (output / "build/compile.stdout").open("w") as stdout, (
            output / "build/compile.stderr").open("w") as stderr:
        _run(
            _nvcc_command(tools.nvcc, snapshot / source.name, executable),
            cwd=repo, stdout=stdout, stderr=stderr)
    sass = output / "build/memory_dependency_task_duration.sass"
    with sass.open("w", encoding="utf-8") as stdout:
        stdout.write(
            f"arch = sm_80\nidentifier = {(snapshot / source.name).resolve()}\n")
        _run(
            _cuobjdump_command(tools.cuobjdump, "--dump-sass", executable),
            cwd=repo, stdout=stdout)
    resources = output / "build/memory_dependency_task_duration.resources"
    with resources.open("w", encoding="utf-8") as stdout:
        _run(
            _cuobjdump_command(
                tools.cuobjdump, "--dump-resource-usage", executable),
            cwd=repo, stdout=stdout)
    (output / "aggregate.sql").write_text(
        AGGREGATE_SQL + "\n", encoding="utf-8")
    (output / "run/started_at.txt").write_text(
        datetime.now().astimezone().isoformat() + "\n", encoding="utf-8")
    for trial in EXPECTED_TRIALS:
        _capture_trial(
            repo, output, executable, tools, args.device,
            args.expected_device_uuid, idle_memory_used_mib, trial)
    (output / "run/completed_at.txt").write_text(
        datetime.now().astimezone().isoformat() + "\n", encoding="utf-8")
    summary, measurements = _write_aggregates(output)
    provenance = {
        "hardware_id": "nvidia_a800",
        "owner": args.owner,
        "baseline_commit": commit,
        "device_index": args.device,
        "device_uuid": identity["uuid"],
        "device_pci_bus_id": identity["pci_bus_id"],
        "timer_domain": "nsys_gpu_kernel",
        "owner_shell_pid": os.getppid(),
        "nsys_version": _run(
            [str(tools.nsys), "--version"], cwd=repo,
            stdout=subprocess.PIPE).stdout.strip(),
        "compiler_version": _run(
            [str(tools.nvcc), "--version"], cwd=repo,
            stdout=subprocess.PIPE).stdout.strip(),
        "cuobjdump_version": _run(
            [str(tools.cuobjdump), "--version"], cwd=repo,
            stdout=subprocess.PIPE).stdout.strip(),
        "nvidia_smi_version": _run(
            _nvidia_smi_version_command(tools.nvidia_smi, args.device),
            cwd=repo, stdout=subprocess.PIPE).stdout,
        "driver_version": identity["driver_version"],
        "nvcc_path": str(tools.nvcc),
        "nvcc_sha256": _sha256(tools.nvcc),
        "cuobjdump_path": str(tools.cuobjdump),
        "cuobjdump_sha256": _sha256(tools.cuobjdump),
        "nsys_path": str(tools.nsys),
        "nsys_sha256": _sha256(tools.nsys),
        "nvidia_smi_path": str(tools.nvidia_smi),
        "nvidia_smi_sha256": _sha256(tools.nvidia_smi),
        "source_sha256": _sha256(snapshot / source.name),
        "report_source_sha256": _sha256(snapshot / reporter.name),
        "runner_source_sha256": _sha256(snapshot / runner.name),
        "binary_sha256": _sha256(executable),
        "resource_sha256": _sha256(resources),
        "sass_sha256": _sha256(sass),
        "summary_sha256": _sha256(summary),
        "measurement_sha256": _sha256(measurements),
        "production_or_evaluation_duration_consumed": False,
        "cleanup": "complete",
    }
    provenance["nvidia_smi_version"] = _nvidia_smi_version(
        provenance["nvidia_smi_version"])
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    _write_manifest(output)


def main(argv=None):
    with _supervise_children():
        _main(argv)


if __name__ == "__main__":
    main()
