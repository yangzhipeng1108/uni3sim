#!/usr/bin/env python3
"""Capture balanced Ascend 910B AIV workspace-transfer trials."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


AIC_GROUPS = (8, 16)
MODES = (
    "vec1_m4_s512_ordered_chain",
    "vec1_m4_s1024_ordered_chain",
    "vec2_m4_final_ordered_chain",
)
MODE_IDS = {
    "vec1_m4_s512_ordered_chain": 6,
    "vec1_m4_s1024_ordered_chain": 7,
    "vec2_m4_final_ordered_chain": 8,
}
CELLS = tuple((mode, groups) for mode in MODES for groups in AIC_GROUPS)
TRIALS = 36
ITERATIONS = 4096
SYSTEM_COUNTER_HZ = 50_000_000
SYSTEM_COUNTER_TICK_NS = 20.0
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aiv_workspace_transfer_contract.h",
    "aiv_workspace_transfer_kernel.cpp",
    "aiv_workspace_transfer_main.cpp",
    "aiv_workspace_transfer_report.py",
    "aiv_workspace_transfer_runner.py",
)
RAW_FIELDS = (
    "run_id", "trial", "order_direction", "order_position",
    "process_pid", "mode", "aic_groups", "aiv_workers",
    "active_vec1_workers", "active_vec2_workers", "iterations",
    "vec1_actor_cycle_pairs", "vec2_actor_cycle_pairs", "tick_ns",
    "vec1_critical_completion_ns", "vec2_critical_completion_ns",
    "critical_completion_ns", "validation",
)
CHILD_FIELDS = RAW_FIELDS[5:]
CHILD_HEADER = ",".join(CHILD_FIELDS)
CANN_ROOT = Path("/usr/local/Ascend/cann-8.5.1")
CANN_VERSION_FILE = CANN_ROOT / "aarch64-linux/ascend_toolkit_install.info"
EXPECTED_CANN = {
    "package": "Ascend-cann-toolkit",
    "version": "8.5.1",
    "innerversion": "V100R001C25SPC002B220",
}
FIA_VECTOR_SOURCE = (
    CANN_ROOT
    / "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc"
    / "common/arch32/fia_block_vec_nonquant.h"
)
EXPECTED_FIA_VECTOR_SOURCE_SHA256 = (
    "259f5700812eafe32bcffd075f59d90c82cb71458f62bd47cb3d6f17f171e762"
)
FIA_VECTOR_SOURCE_SYMBOLS = (
    "FiaBlockVecNonQuant<FIAT>::DealBmm1ResBaseBlock",
    "FiaBlockVecNonQuant<FIAT>::DealBmm2ResBaseBlock",
)
FIA_VECTOR_SOURCE_MARKERS = (
    "DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);",
    "DataCopy(vec1ResGm[inOutGmOffset], vec1ResUb, computeSize);",
    "SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);",
    "WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);",
    "DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);",
    "DataCopy(vec2ResGm[vec2ResGmOffset], tmpBmm2Res, vec2ComputeSize);",
    "if (GetBlockIdx() % 2 == 1)",
    "mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;",
)
NPU_SMI = Path("/usr/local/bin/npu-smi")
LOGICAL_DEVICE_ID = 0
CHILD_TIMEOUT_SECONDS = 900
_ACTIVE_PROCESS = None
_CHILD_SIGNALS = frozenset((signal.SIGINT, signal.SIGTERM))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(benchmark_root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILENAMES:
        path = benchmark_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"snapshot source is missing: {path}")
        payload = path.read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _cell_index(trial_index, order_position):
    rotation = trial_index % len(CELLS)
    if trial_index < TRIALS // 2:
        return (rotation + order_position) % len(CELLS)
    return (rotation - order_position) % len(CELLS)


def _ordered_cells(trial_index):
    if trial_index < 0 or trial_index >= TRIALS:
        raise ValueError("trial index is outside the frozen campaign")
    direction = "forward" if trial_index < TRIALS // 2 else "reverse"
    return direction, tuple(
        CELLS[_cell_index(trial_index, position)]
        for position in range(len(CELLS))
    )


def _validate_schedule():
    trials_per_phase = TRIALS // 2
    expected_position_count = trials_per_phase // len(CELLS)
    for phase_start in (0, trials_per_phase):
        position_counts = [
            {cell: 0 for cell in CELLS} for _ in CELLS
        ]
        for trial_index in range(phase_start, phase_start + trials_per_phase):
            _direction, ordered = _ordered_cells(trial_index)
            if len(ordered) != len(CELLS) or set(ordered) != set(CELLS):
                raise AssertionError("one trial does not cover every cell")
            for position, cell in enumerate(ordered):
                position_counts[position][cell] += 1
        if any(
                count != expected_position_count
                for counts in position_counts for count in counts.values()):
            raise AssertionError("campaign phase is not position-balanced")


def _process_group_exists(process_group_id):
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _signal_process_group(process_group_id, signum):
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_group_exit(process, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(process.pid):
        process.poll()
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    process.poll()
    return True


def _terminate_process_group(
        process, *, term_timeout_seconds=10, kill_timeout_seconds=10):
    if not _process_group_exists(process.pid):
        process.poll()
        return
    if not _signal_process_group(process.pid, signal.SIGTERM):
        process.poll()
        return
    if _wait_for_process_group_exit(process, term_timeout_seconds):
        return
    if not _signal_process_group(process.pid, signal.SIGKILL):
        process.poll()
        return
    if not _wait_for_process_group_exit(process, kill_timeout_seconds):
        raise RuntimeError(
            f"child process group {process.pid} survived SIGKILL")


def _handle_signal(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128 + signum)


def _unblock_child_ownership_signals():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _CHILD_SIGNALS)


def _spawn_child(command, environment):
    global _ACTIVE_PROCESS
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _CHILD_SIGNALS)
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment, start_new_session=True,
            preexec_fn=_unblock_child_ownership_signals)
        _ACTIVE_PROCESS = process
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    return process


def _communicate_child(process, timeout_seconds):
    try:
        return process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        raise RuntimeError(
            f"child PID {process.pid} timed out after "
            f"{timeout_seconds} seconds") from error
    finally:
        if _process_group_exists(process.pid):
            _terminate_process_group(process)


def _run_checked(command, *, environment=None, timeout=30):
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True,
        env=environment, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command returned {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return completed.stdout


def _run_captured(command, stdout_path, stderr_path, *, timeout):
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    artifact = {
        "command": command,
        "return_code": completed.returncode,
        "stdout": stdout_path.name,
        "stdout_sha256": _sha256(stdout_path),
        "stderr": stderr_path.name,
        "stderr_sha256": _sha256(stderr_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"build command returned {completed.returncode}; "
            f"see {stdout_path} and {stderr_path}")
    return artifact


def _parse_idle_usage(text):
    if not isinstance(text, str) or not text.strip():
        raise TypeError("npu-smi usage telemetry must be non-empty")
    labels = {
        "aic_percent": "Aicore Usage Rate(%)",
        "aiv_percent": "Aivector Usage Rate(%)",
        "hbm_bandwidth_percent": "HBM Bandwidth Usage Rate(%)",
    }
    result = {}
    for field, label in labels.items():
        matches = re.findall(
            rf"^\s*{re.escape(label)}\s*:\s*([0-9]+)\s*$",
            text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"usage telemetry has ambiguous {field}")
        value = int(matches[0])
        if value != 0:
            raise RuntimeError(f"selected NPU is busy: {field}={value}")
        result[field] = value
    return result


def _parse_device_identity(global_text, board_text, physical_device_id):
    rows = re.findall(
        r"^\s*\|\s*([0-9]+)\s+((?:Ascend\s*)?910B2)\s*\|.*$",
        global_text, flags=re.IGNORECASE | re.MULTILINE)
    selected = [
        model for index, model in rows if int(index) == physical_device_id]
    if len(selected) != 1:
        raise ValueError(
            f"npu-smi must identify physical device {physical_device_id} "
            "exactly once as 910B2")
    board_ids = re.findall(
        r"^\s*NPU\s+ID\s*:\s*([0-9]+)\s*$", board_text,
        flags=re.IGNORECASE | re.MULTILINE)
    if board_ids != [str(physical_device_id)]:
        raise ValueError("board telemetry does not match the selected NPU")
    pci = re.findall(
        r"^\s*(?:PCIe?\s+Bus\s+Info|PCIe?\s+Bus-Id|Bus-Id)\s*:\s*"
        r"([0-9a-fA-F:.]+)\s*$", board_text,
        flags=re.IGNORECASE | re.MULTILINE)
    board = re.findall(
        r"^\s*Board\s+ID\s*:\s*([^\s]+)\s*$", board_text,
        flags=re.IGNORECASE | re.MULTILINE)
    if len(pci) != 1 or len(board) != 1:
        raise ValueError(
            "board telemetry must contain one PCI identity and board ID")
    return {
        "physical_device_id": physical_device_id,
        "logical_device_id": LOGICAL_DEVICE_ID,
        "model": selected[0],
        "board_id": board[0],
        "pci_identity": pci[0],
        "selection_environment": {
            "ASCEND_DEVICE_ID": str(LOGICAL_DEVICE_ID),
            "ASCEND_RT_VISIBLE_DEVICES": str(physical_device_id),
        },
    }


def _build_binary(output_dir, benchmark_root, source_snapshot_sha256):
    build_dir = output_dir / "build"
    configure_stdout = output_dir / "build_configure.stdout"
    configure_stderr = output_dir / "build_configure.stderr"
    build_stdout = output_dir / "build_compile.stdout"
    build_stderr = output_dir / "build_compile.stderr"
    configure_command = [
        "cmake", "-S", str(benchmark_root), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DASCEND_HOME_PATH={CANN_ROOT}",
        f"-DASCEND_CANN_PACKAGE_PATH={CANN_ROOT}",
        "-DSOC_VERSION=Ascend910B2",
    ]
    build_command = [
        "cmake", "--build", str(build_dir), "--target",
        "aiv_workspace_transfer_bench", "--parallel", "4",
    ]
    configure = _run_captured(
        configure_command, configure_stdout, configure_stderr, timeout=300)
    compile_artifact = _run_captured(
        build_command, build_stdout, build_stderr, timeout=900)
    binary = build_dir / "aiv_workspace_transfer_bench"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("source-bound build did not produce the benchmark")
    if _snapshot_sha256(benchmark_root) != source_snapshot_sha256:
        raise RuntimeError("source snapshot changed during source-bound build")
    return binary, {
        "source_snapshot_sha256": source_snapshot_sha256,
        "build_directory": str(build_dir),
        "configure": configure,
        "compile": compile_artifact,
        "binary_path": str(binary),
        "binary_sha256": _sha256(binary),
    }


def _parse_cann_version(text):
    source_fields = {
        "package": "package_name",
        "version": "version",
        "innerversion": "innerversion",
    }
    result = {}
    for field, expected in EXPECTED_CANN.items():
        matches = re.findall(
            rf"^\s*{source_fields[field]}\s*=\s*([^\s]+)\s*$",
            text, flags=re.IGNORECASE | re.MULTILINE)
        if matches != [expected]:
            raise ValueError(
                f"CANN {field} must be exactly {expected}, got {matches!r}")
        result[field] = expected
    return result


def _validate_cann_source():
    if not FIA_VECTOR_SOURCE.is_file():
        raise FileNotFoundError(
            f"CANN FIA vector source is missing: {FIA_VECTOR_SOURCE}")
    observed = _sha256(FIA_VECTOR_SOURCE)
    if observed != EXPECTED_FIA_VECTOR_SOURCE_SHA256:
        raise ValueError(
            "CANN FIA vector source SHA256 changed: "
            f"expected {EXPECTED_FIA_VECTOR_SOURCE_SHA256}, got {observed}")
    source = FIA_VECTOR_SOURCE.read_text(encoding="utf-8")
    missing = tuple(
        marker for marker in FIA_VECTOR_SOURCE_MARKERS
        if marker not in source)
    if missing:
        raise ValueError(f"CANN FIA source lost required markers: {missing}")
    return {
        "path": str(FIA_VECTOR_SOURCE.resolve()),
        "sha256": observed,
        "symbols": list(FIA_VECTOR_SOURCE_SYMBOLS),
        "required_markers": list(FIA_VECTOR_SOURCE_MARKERS),
    }


def _parse_actor_cycle_pairs(text, expected_actor_ids, label):
    expected_count = len(expected_actor_ids)
    if expected_count == 0:
        if text != "none":
            raise ValueError(f"{label} must be none for an inactive domain")
        return ()
    encoded_pairs = text.split(";")
    if len(encoded_pairs) != expected_count:
        raise ValueError(f"{label} has an invalid active-actor count")
    pairs = []
    for encoded in encoded_pairs:
        fields = encoded.split(":")
        if len(fields) != 3:
            raise ValueError(f"{label} has an invalid actor cycle pair")
        actor_id, short_cycles, long_cycles = (
            int(value) for value in fields)
        if short_cycles <= 0 or long_cycles <= short_cycles:
            raise ValueError(f"{label} has invalid paired cycles")
        pairs.append((actor_id, short_cycles, long_cycles))
    observed_actor_ids = tuple(pair[0] for pair in pairs)
    if observed_actor_ids != expected_actor_ids:
        raise ValueError(f"{label} changed the physical actor mapping")
    return tuple(pairs)


def _critical_paired_completion_ns(pairs, iterations):
    if not pairs:
        return 0.0
    actor_completion_ns = tuple(
        (long_cycles - short_cycles) * SYSTEM_COUNTER_TICK_NS
        / (3 * iterations)
        for _actor_id, short_cycles, long_cycles in pairs
    )
    return max(actor_completion_ns)


def _active_counts(mode, groups):
    if mode.startswith("vec1_"):
        return groups, 0
    if mode.startswith("vec2_"):
        return 0, groups
    raise ValueError(f"unsupported mode {mode}")


def _parse_child_row(line, expected_cell):
    fields = line.split(",")
    if len(fields) != len(CHILD_FIELDS):
        raise ValueError("child result has an invalid field count")
    raw = dict(zip(CHILD_FIELDS, fields))
    mode, groups = expected_cell
    if raw["mode"] != mode or int(raw["aic_groups"]) != groups:
        raise ValueError("child changed the requested cell")
    expected_vec1, expected_vec2 = _active_counts(mode, groups)
    integers = {
        name: int(raw[name])
        for name in (
            "aiv_workers", "active_vec1_workers", "active_vec2_workers",
            "iterations")
    }
    if integers != {
            "aiv_workers": 2 * groups,
            "active_vec1_workers": expected_vec1,
            "active_vec2_workers": expected_vec2,
            "iterations": ITERATIONS}:
        raise ValueError("child changed the topology or iteration count")
    expected_actor_ids = tuple(2 * index for index in range(groups))
    vec1_pairs = _parse_actor_cycle_pairs(
        raw["vec1_actor_cycle_pairs"],
        expected_actor_ids if expected_vec1 else (), "Vec1")
    vec2_pairs = _parse_actor_cycle_pairs(
        raw["vec2_actor_cycle_pairs"],
        expected_actor_ids if expected_vec2 else (), "Vec2")
    tick_ns = float(raw["tick_ns"])
    vec1_ns = float(raw["vec1_critical_completion_ns"])
    vec2_ns = float(raw["vec2_critical_completion_ns"])
    critical_ns = float(raw["critical_completion_ns"])
    if tick_ns != SYSTEM_COUNTER_TICK_NS:
        raise ValueError("child changed the system-counter tick")
    expected_vec1_ns = _critical_paired_completion_ns(
        vec1_pairs, ITERATIONS)
    expected_vec2_ns = _critical_paired_completion_ns(
        vec2_pairs, ITERATIONS)
    expected_values = (expected_vec1_ns, expected_vec2_ns)
    observed_values = (vec1_ns, vec2_ns)
    if any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=5.1e-10)
            for observed, expected in zip(observed_values, expected_values)):
        raise ValueError("child completion time is inconsistent with cycles")
    if (not math.isclose(
            critical_ns, max(vec1_ns, vec2_ns),
            rel_tol=0.0, abs_tol=5.1e-10)
            or not math.isfinite(critical_ns) or critical_ns <= 0.0
            or raw["validation"] != "pass"):
        raise ValueError("child critical completion or validation is invalid")
    return raw


def _write_json_exclusive(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validate_output_path(path, repository_root):
    resolved = path.resolve()
    profiler_root = (repository_root / "profiler_output").resolve()
    if profiler_root not in resolved.parents:
        raise ValueError("output directory must be below profiler_output/")
    relative = resolved.relative_to(profiler_root)
    if "ascend_910b" not in relative.parts:
        raise ValueError("output directory must be scoped under ascend_910b")
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")


def _run_campaign(args, benchmark_root):
    global _ACTIVE_PROCESS
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID contains unsupported characters")
    if COMMIT_PATTERN.fullmatch(args.source_commit) is None:
        raise ValueError("source commit must be 40 lowercase hex digits")
    if SHA256_PATTERN.fullmatch(args.expected_snapshot_sha256) is None:
        raise ValueError("expected snapshot must be a lowercase SHA256")
    if args.physical_device_id < 0:
        raise ValueError("physical device ID must be non-negative")
    _validate_schedule()
    repository_root = benchmark_root.parents[4]
    output_dir = args.output_dir.resolve()
    _validate_output_path(output_dir, repository_root)
    observed_snapshot = _snapshot_sha256(benchmark_root)
    if observed_snapshot != args.expected_snapshot_sha256:
        raise ValueError("deployed source snapshot does not match expectation")
    git_marker = repository_root / ".git"
    if git_marker.exists():
        head = _run_checked([
            "git", "-C", str(repository_root), "rev-parse", "HEAD"]
        ).strip()
        if head != args.source_commit:
            raise ValueError("source commit does not match checkout HEAD")
        relative_root = benchmark_root.relative_to(repository_root)
        relative_paths = [str(relative_root / name) for name in SOURCE_FILENAMES]
        _run_checked([
            "git", "-C", str(repository_root), "diff", "--exit-code",
            args.source_commit, "--", *relative_paths])
        source_identity_mode = "git_commit_and_source_snapshot"
    else:
        source_identity_mode = "audited_source_snapshot"
    source_semantics = _validate_cann_source()
    if not CANN_VERSION_FILE.is_file():
        raise FileNotFoundError(f"CANN version file is missing: {CANN_VERSION_FILE}")
    cann_identity = _parse_cann_version(
        CANN_VERSION_FILE.read_text(encoding="utf-8"))
    if not NPU_SMI.is_file() or not os.access(NPU_SMI, os.X_OK):
        raise FileNotFoundError(f"npu-smi is unavailable: {NPU_SMI}")

    output_dir.mkdir(parents=True)
    binary, build = _build_binary(
        output_dir, benchmark_root, observed_snapshot)
    environment = os.environ.copy()
    environment["ASCEND_DEVICE_ID"] = str(LOGICAL_DEVICE_ID)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_device_id)
    global_command = [str(NPU_SMI), "info"]
    board_command = [
        str(NPU_SMI), "info", "-t", "board", "-i",
        str(args.physical_device_id)]
    usage_command = [
        str(NPU_SMI), "info", "-t", "usages", "-i",
        str(args.physical_device_id)]
    global_text = _run_checked(global_command, environment=environment)
    board_text = _run_checked(board_command, environment=environment)
    device_identity = _parse_device_identity(
        global_text, board_text, args.physical_device_id)
    global_path = output_dir / "npu_smi_global.txt"
    board_path = output_dir / "npu_smi_board.txt"
    global_path.write_text(global_text, encoding="utf-8")
    board_path.write_text(board_text, encoding="utf-8")
    pre_text = _run_checked(usage_command, environment=environment)
    pre_usage = _parse_idle_usage(pre_text)
    pre_path = output_dir / "npu_smi_pre.txt"
    pre_path.write_text(pre_text, encoding="utf-8")
    raw_path = output_dir / "raw.csv"
    trial_artifacts = []
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for trial_index in range(TRIALS):
            direction, cells = _ordered_cells(trial_index)
            command = [str(binary), "--batch"]
            for mode, groups in cells:
                command.extend((
                    str(MODE_IDS[mode]), str(groups), str(ITERATIONS)))
            process = _spawn_child(command, environment)
            try:
                stdout, stderr = _communicate_child(
                    process, args.child_timeout_seconds)
            finally:
                _ACTIVE_PROCESS = None
            stdout_path = output_dir / f"trial_{trial_index + 1}.stdout"
            stderr_path = output_dir / f"trial_{trial_index + 1}.stderr"
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(
                    f"trial child failed with {process.returncode}; "
                    "see captured stdout/stderr")
            lines = stdout.strip().splitlines()
            if not lines or lines[0] != CHILD_HEADER:
                raise RuntimeError("child output header changed")
            if len(lines) != len(cells) + 1:
                raise RuntimeError("child output cell count changed")
            for position, (cell, line) in enumerate(zip(cells, lines[1:])):
                parsed = _parse_child_row(line, cell)
                writer.writerow({
                    "run_id": args.run_id,
                    "trial": trial_index + 1,
                    "order_direction": direction,
                    "order_position": position,
                    "process_pid": process.pid,
                    **parsed,
                })
            stream.flush()
            os.fsync(stream.fileno())
            trial_artifacts.append({
                "trial": trial_index + 1,
                "direction": direction,
                "process_pid": process.pid,
                "command": command,
                "stdout": stdout_path.name,
                "stdout_sha256": _sha256(stdout_path),
                "stderr": stderr_path.name,
                "stderr_sha256": _sha256(stderr_path),
            })

    post_text = _run_checked(usage_command, environment=environment)
    post_usage = _parse_idle_usage(post_text)
    post_path = output_dir / "npu_smi_post.txt"
    post_path.write_text(post_text, encoding="utf-8")
    if _snapshot_sha256(benchmark_root) != observed_snapshot:
        raise RuntimeError("source snapshot changed after source-bound build")
    if _sha256(binary) != build["binary_sha256"]:
        raise RuntimeError("source-bound benchmark binary changed during capture")
    manifest = {
        "schema": "ascend_910b_aiv_workspace_transfer_manifest_v4",
        "run_id": args.run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "source_identity_mode": source_identity_mode,
        "source_snapshot_sha256": observed_snapshot,
        "source_files": {
            name: _sha256(benchmark_root / name) for name in SOURCE_FILENAMES},
        "build": build,
        "cann": {
            "root": str(CANN_ROOT),
            "identity": cann_identity,
            "version_file": str(CANN_VERSION_FILE),
            "version_file_sha256": _sha256(CANN_VERSION_FILE),
            "fia_vector_source": source_semantics,
        },
        "source_semantics": {
            "vec1": {
                "symbol": FIA_VECTOR_SOURCE_SYMBOLS[0],
                "actual_m": 4,
                "s512_gm_to_ub_bytes_per_task": 8192,
                "s512_ub_to_gm_bytes_per_task": 4096,
                "s1024_gm_to_ub_bytes_per_task": 16384,
                "s1024_ub_to_gm_bytes_per_task": 8192,
            },
            "vec2": {
                "symbol": FIA_VECTOR_SOURCE_SYMBOLS[1],
                "actual_m": 4,
                "final_state_read_bytes_per_task": 1024,
                "final_state_write_bytes_per_task": 512,
                "dependency": "MTE3_MTE2 event/wait before state reuse",
            },
            "topology": "KERNEL_TYPE_MIX_AIC_1_2",
            "actor_mapping": (
                "D64 mDealSize=4 runs both Vec1 and Vec2 on primary AIV; "
                "secondary AIV has vecDealM=0"),
        },
        "device": {
            **device_identity,
            "identity_telemetry": {
                "global": {
                    "command": global_command,
                    "path": global_path.name,
                    "sha256": _sha256(global_path),
                },
                "board": {
                    "command": board_command,
                    "path": board_path.name,
                    "sha256": _sha256(board_path),
                },
            },
        },
        "idle_telemetry": {
            "pre": {
                "values": pre_usage,
                "path": pre_path.name,
                "sha256": _sha256(pre_path),
                "command": usage_command,
            },
            "post": {
                "values": post_usage,
                "path": post_path.name,
                "sha256": _sha256(post_path),
                "command": usage_command,
            },
        },
        "clock": {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
            "derivation": (
                "max_i((long_i-short_i)*tick/(3*iterations)) over "
                "same-invocation physical-actor pairs"),
        },
        "measurement_scope": (
            "byte-count-matched actor-local ordered transfer chains, with "
            "N and 4N intervals "
            "paired inside one persistent invocation on each selected "
            "physical primary AIV actor; excludes "
            "Vec1/Vec2 overlap, attention arithmetic, MatMul, cross-core "
            "handoffs, host synchronization, and kernel launch time"),
        "production_or_evaluation_duration_consumed": False,
        "campaign": {
            "aic_groups": list(AIC_GROUPS),
            "aiv_per_aic": 2,
            "modes": list(MODES),
            "iterations": ITERATIONS,
            "trials": TRIALS,
            "cell_count": len(CELLS),
            "order": "repeated_position_balanced_forward_reverse_phases",
        },
        "capture": {
            "raw": raw_path.name,
            "raw_sha256": _sha256(raw_path),
            "raw_rows": TRIALS * len(CELLS),
            "runner_pid": os.getpid(),
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
            "trials": trial_artifacts,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    print(manifest_path)


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    benchmark_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("snapshot")
    run = subparsers.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--expected-snapshot-sha256", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--physical-device-id", required=True, type=int)
    run.add_argument(
        "--child-timeout-seconds", type=int, default=CHILD_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.command == "snapshot":
        print(_snapshot_sha256(benchmark_root))
    elif args.command == "run":
        if args.child_timeout_seconds <= 0:
            parser.error("--child-timeout-seconds must be positive")
        _run_campaign(args, benchmark_root)
    else:
        raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
