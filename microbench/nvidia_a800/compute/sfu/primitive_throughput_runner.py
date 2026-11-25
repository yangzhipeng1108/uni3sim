#!/usr/bin/env python3
"""Capture the fixed A800 matched-control primitive mechanism matrix."""

import argparse
from dataclasses import dataclass
from functools import partial
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
from statistics import median
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = Path(__file__).resolve()
SOURCE_PATH = Path(__file__).with_name("primitive_throughput.cu")
OUTPUT_ROOT = REPO_ROOT / "profiler_output" / "microbench" / "nvidia_a800"
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
EXPECTED_COMPUTE_CAPABILITY = "8.0"
TARGET_PRIMITIVES = (
    "fma_f32",
    "add_f32",
    "rsqrt_approx_f32",
)
LATENCY_TARGET_PRIMITIVES = (
    "fma_f32",
    "add_f32",
    "mul_f32",
    "rsqrt_approx_f32",
)
OBSERVED_TARGET_PRIMITIVES = TARGET_PRIMITIVES + ("mul_f32",)
CONTROL_PRIMITIVE = "control_xor_b32"
ALL_PRIMITIVES = TARGET_PRIMITIVES + (CONTROL_PRIMITIVE,)
LATENCY_UNROLLS = (4, 8, 16, 32)
THROUGHPUT_UNROLL = 128
THROUGHPUT_CHAINS = (1, 2, 4, 8, 16)
THROUGHPUT_WARPS = (1, 4, 8, 16, 32)
RESIDENCY_CELLS = ("one", "max")
TRIALS = 31
SM_COUNT = 108
SUBCORES_PER_SM = 4
WARP_SIZE = 32
MAX_TIMED_OPCODES_PER_THREAD = 2048
COMMAND_TIMEOUT_SECONDS = 120
COMPILE_TIMEOUT_SECONDS = 600
PROCESS_GROUP_GRACE_SECONDS = 5
POSTFLIGHT_IDLE_ATTEMPTS = 10
POSTFLIGHT_IDLE_INTERVAL_SECONDS = 1.0
NOT_APPLICABLE = "not_applicable"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
DEVICE_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
PCI_BUS_ID_PATTERN = re.compile(
    r"(?:[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):"
    r"[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")
RAW_FIELDS = (
    "trial",
    "mode",
    "primitive",
    "clock_domain",
    "device_name",
    "device_uuid",
    "pci_bus_id",
    "device_property_clock_khz",
    "sm_count",
    "grid_blocks",
    "block_threads",
    "warps_per_cta",
    "residency_cell",
    "paired_resident_ctas_per_sm",
    "specialization_occupancy_limit_ctas_per_sm",
    "chains",
    "unroll",
    "timed_opcode_count_per_thread",
    "issued_warp_opcodes",
    "sm_clock_start_cycle",
    "sm_clock_end_cycle",
    "elapsed_sm_cycles",
    "active_sm_count",
    "min_local_elapsed_sm_cycles",
    "max_local_elapsed_sm_cycles",
    "sum_local_elapsed_sm_cycles",
    "lane_ops_per_subcore_cycle",
    "fma_flops_per_subcore_cycle",
    "warp_instructions_per_subcore_cycle",
)
BINARY_RAW_HEADER = ",".join(RAW_FIELDS)
BINARY_ABI_LINE = "abi,primitive_throughput,6," + BINARY_RAW_HEADER
OCCUPANCY_FIELDS = (
    "record_type",
    "mode",
    "primitive",
    "unroll",
    "chains",
    "warps",
    "occupancy_limit_ctas_per_sm",
    "sm_count",
    "cooperative_grid_limit",
)
RUNNER_FIELDS = (
    "run_id",
    "cell_index",
    "pair_id",
    "pair_order",
    "child_pid",
    "target_occupancy_limit_ctas_per_sm",
    "control_occupancy_limit_ctas_per_sm",
)
RUNNER_RAW_HEADER = ",".join(RUNNER_FIELDS + RAW_FIELDS)
LATENCY_DERIVATION_FIELDS = (
    "primitive",
    "clock_domain",
    "unit",
    "evidence_status",
    "sample_count_per_unroll",
    "median_elapsed_cycles_u4",
    "median_elapsed_cycles_u8",
    "median_elapsed_cycles_u16",
    "median_elapsed_cycles_u32",
    "canonical_span",
    "latency_sm_cycles_per_dependent_warp_opcode",
    "adjacent_slope_u4_u8",
    "adjacent_slope_u8_u16",
    "adjacent_slope_u16_u32",
    "control_subtracted",
)
LATENCY_DERIVATION_HEADER = ",".join(LATENCY_DERIVATION_FIELDS)
DEVICE_QUERY_FIELDS = (
    "uuid,name,pci.bus_id,driver_version,pstate,temperature.gpu,"
    "memory.used,utilization.gpu,clocks.sm,clocks.max.sm,compute_cap")
COMPUTE_PROCESS_QUERY_FIELDS = "pid,gpu_uuid,used_gpu_memory"
LIVE_PROCESS_FIELDS = (
    "gpu_owner_contract,nvidia_smi_pid,gpu_uuid,used_gpu_memory")
PRIMITIVE_ENUM = {
    "fma_f32": 0,
    "add_f32": 1,
    "ex2_approx_f32": 2,
    "rcp_approx_f32": 3,
    "rsqrt_approx_f32": 4,
    "control_xor_b32": 5,
    "mul_f32": 6,
}
PRIMITIVE_OPCODE = {
    0: "FFMA",
    1: "FADD",
    2: "MUFU.EX2",
    3: "MUFU.RCP",
    4: "MUFU.RSQ",
    5: "LOP3.LUT",
    6: "FMUL",
}
TARGET_OPCODES = frozenset((
    "FFMA", "FADD", "FMUL", "MUFU.EX2", "MUFU.RCP", "MUFU.RSQ"))
DEPENDENT_SYMBOL = re.compile(
    r"^_Z16dependent_kernelIL9Primitive([0-6])ELi(4|8|16|32)EE")
THROUGHPUT_SYMBOL = re.compile(
    r"^_Z17throughput_kernelIL9Primitive([0-5])ELi(1|2|4|8|16)EE")
FUNCTION_MARKER = re.compile(r"Function\s*:\s*(\S+)")
INSTRUCTION = re.compile(
    r"\*/\s+(?:@[!A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b")
FORBIDDEN_TIMED_PREFIXES = (
    "LD", "ST", "CALL", "BRA", "JMP", "RET", "EXIT", "BAR", "ATOM",
    "RED", "MEMBAR")
PTXAS_FUNCTION = re.compile(r"Function properties for\s+'?(\S+?)'?$")
PTXAS_RESOURCES = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
    r"(\d+) bytes spill loads")
PTXAS_REGISTERS = re.compile(r"Used (\d+) registers")
SASS_INSTRUCTION = re.compile(
    r"\*/\s+(?:(@[!A-Z0-9]+)\s+)?([A-Z][A-Z0-9_.]*)"
    r"(?:\s+(.+?))?\s*;")
REGISTER_OPERAND = re.compile(
    r"[-~|]?(U?R)(Z|[0-9]+)(?:\.(64))?(?:\.reuse)?[|]?")
SASS_IMMEDIATE = re.compile(
    r"[+-]?(?:0x[0-9a-fA-F]+|"
    r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?)")
ACTIVE_PROCESS_GROUPS = {}
PROCESS_STAGE_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,160}")
ENDPOINT_XOR_IMMEDIATE = "0x6a09e667"
FORBIDDEN_ENVIRONMENT_NAMES = frozenset((
    "CUDA_FORCE_PTX_JIT",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
))
FORBIDDEN_ENVIRONMENT_PREFIXES = ("CUDA_", "NVCC_", "LD_")
TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
NVIDIA_SMI_IDENTITY_ARGS = ("-h",)


class RunnerTerminationSignal(RuntimeError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(
            f"runner received termination signal {signal.Signals(signum).name}")


class DeviceNotIdleError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class GitBlob:
    object_id: str
    content: bytes
    sha256: str


def _termination_signal_handler(signum, _frame):
    raise RunnerTerminationSignal(signum)


def _install_signal_handlers():
    previous = {}
    for signum in TERMINATION_SIGNALS:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _termination_signal_handler)
    return previous


def _restore_signal_handlers(previous):
    for signum in TERMINATION_SIGNALS:
        signal.signal(signum, previous[signum])


@dataclass(frozen=True)
class Cell:
    pair_id: str
    pair_order: str
    mode: str
    primitive: str
    unroll: int
    chains: int
    warps: int
    residency: str


@dataclass(frozen=True)
class Pair:
    pair_id: str
    target: Cell
    control: Cell
    execution_order: tuple


def _make_pair(pair_id, mode, target, unroll, chains, warps, residency,
               target_first):
    target_order = "1" if target_first else "2"
    control_order = "2" if target_first else "1"
    target_cell = Cell(
        pair_id, target_order, mode, target, unroll, chains, warps, residency)
    control_cell = Cell(
        pair_id, control_order, mode, CONTROL_PRIMITIVE, unroll, chains,
        warps, residency)
    if target_first:
        execution_order = (target_cell, control_cell)
    else:
        execution_order = (control_cell, target_cell)
    return Pair(pair_id, target_cell, control_cell, execution_order)


def _fixed_pairs():
    pairs = []
    pair_index = 0
    for unroll in LATENCY_UNROLLS:
        for target in LATENCY_TARGET_PRIMITIVES:
            pair_id = f"latency_u{unroll}_{target}"
            pairs.append(_make_pair(
                pair_id, "latency", target, unroll, 1, 1, "one",
                pair_index % 2 == 0))
            pair_index += 1
    for chains in THROUGHPUT_CHAINS:
        for warps in THROUGHPUT_WARPS:
            for residency in RESIDENCY_CELLS:
                for target in TARGET_PRIMITIVES:
                    pair_id = (
                        f"throughput_u{THROUGHPUT_UNROLL}_c{chains}_"
                        f"w{warps}_r{residency}_{target}")
                    pairs.append(_make_pair(
                        pair_id, "throughput", target, THROUGHPUT_UNROLL,
                        chains, warps, residency, pair_index % 2 == 0))
                    pair_index += 1
    return tuple(pairs)


FIXED_PAIRS = _fixed_pairs()
FIXED_MATRIX = tuple(
    cell for pair in FIXED_PAIRS for cell in pair.execution_order)
SMOKE_PAIRS = (
    _make_pair(
        "smoke_latency_u4_add_f32", "latency", "add_f32", 4, 1, 1,
        "one", True),
    _make_pair(
        "smoke_throughput_u128_c1_w1_one_fma_f32", "throughput",
        "fma_f32", THROUGHPUT_UNROLL, 1, 1, "one", False),
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fd_identity(fd):
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"held FD {fd} is not a regular file")
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(fd)
    before_fields = (
        before.st_dev, before.st_ino, before.st_mode, before.st_size,
        before.st_mtime_ns)
    after_fields = (
        after.st_dev, after.st_ino, after.st_mode, after.st_size,
        after.st_mtime_ns)
    if after_fields != before_fields:
        raise RuntimeError(f"held FD {fd} changed while hashing")
    return FileIdentity(*after_fields, digest.hexdigest())


def _open_held_file(path, label, require_executable=False):
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for capture identity binding")
    flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        identity = _fd_identity(fd)
        if require_executable and identity.mode & 0o111 == 0:
            raise RuntimeError(f"{label} is not executable")
        _assert_held_file(path, fd, identity, label)
    except BaseException:
        os.close(fd)
        raise
    return fd, identity


def _create_held_file(path, content, label, mode=0o644):
    if not isinstance(content, bytes):
        raise TypeError(f"{label} content must be bytes")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for capture identity binding")
    flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise RuntimeError(f"failed to write complete {label}")
            remaining = remaining[written:]
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        identity = _fd_identity(fd)
        content_sha256 = hashlib.sha256(content).hexdigest()
        if identity.sha256 != content_sha256:
            raise RuntimeError(f"held {label} differs from supplied bytes")
        _assert_held_file(path, fd, identity, label)
    except BaseException:
        os.close(fd)
        raise
    return fd, identity


def _assert_held_file(path, fd, expected, label):
    current = _fd_identity(fd)
    if current != expected:
        raise RuntimeError(f"held {label} content or stat identity changed")
    _assert_path_identity(path, expected, label)


def _assert_path_identity(path, expected, label):
    path_stat = os.stat(path, follow_symlinks=False)
    path_fields = (
        path_stat.st_dev, path_stat.st_ino, path_stat.st_mode,
        path_stat.st_size, path_stat.st_mtime_ns)
    expected_fields = (
        expected.device, expected.inode, expected.mode, expected.size,
        expected.mtime_ns)
    if path_fields != expected_fields or not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"{label} pathname no longer names the held inode")


def _format_file_identity(prefix, identity):
    return (
        f"{prefix}_device={identity.device}\n"
        f"{prefix}_inode={identity.inode}\n"
        f"{prefix}_mode={identity.mode:o}\n"
        f"{prefix}_size={identity.size}\n"
        f"{prefix}_mtime_ns={identity.mtime_ns}\n"
        f"{prefix}_sha256={identity.sha256}\n")


def _proc_fd_path(fd):
    return f"/proc/self/fd/{fd}"


def _require_executable(path, argument_name):
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{argument_name} must name an executable file")
    return resolved


def _capture_environment(source_environment, device_uuid):
    forbidden = sorted(
        name for name in source_environment
        if name in FORBIDDEN_ENVIRONMENT_NAMES
        or name.startswith(FORBIDDEN_ENVIRONMENT_PREFIXES))
    if forbidden:
        raise RuntimeError(
            "capture environment contains forbidden tool/runtime variables: "
            + ", ".join(forbidden))
    environment = source_environment.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device_uuid
    return environment


def _prepare_output_dir(output_dir, run_id):
    resolved = output_dir.resolve()
    allowed = OUTPUT_ROOT.resolve()
    if resolved.parent != allowed:
        raise ValueError(f"--output-dir must be a direct child of {allowed}")
    if resolved.name != run_id:
        raise ValueError("--output-dir basename must equal --run-id")
    if resolved.exists():
        raise FileExistsError(f"refusing to reuse output directory {resolved}")
    allowed.mkdir(parents=True, exist_ok=True)
    resolved.mkdir()
    for name in ("build", "logs", "results", "source_snapshot", "telemetry"):
        (resolved / name).mkdir()
    (resolved / "logs" / "processes").mkdir()
    return resolved


def _process_group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_group_exit(pgid, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _register_process(process, cleanup_records):
    pgid = process.pid
    if pgid in ACTIVE_PROCESS_GROUPS:
        raise RuntimeError(f"process group {pgid} is already registered")
    try:
        actual_pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        actual_pgid = pgid
    if actual_pgid != pgid:
        raise RuntimeError(
            f"child PID {process.pid} has unexpected PGID {actual_pgid}")
    ACTIVE_PROCESS_GROUPS[pgid] = process
    cleanup_records.append(
        f"pid={process.pid} pgid={pgid} group=registered")


def _release_process(process, cleanup_records):
    pgid = process.pid
    if pgid not in ACTIVE_PROCESS_GROUPS:
        raise RuntimeError(f"process group {pgid} is not registered")
    if ACTIVE_PROCESS_GROUPS[pgid] is not process:
        raise RuntimeError(f"process group {pgid} registration changed owner")
    if _process_group_exists(pgid):
        raise RuntimeError(f"cannot release live process group {pgid}")
    del ACTIVE_PROCESS_GROUPS[pgid]
    cleanup_records.append(
        f"pid={process.pid} pgid={pgid} group=released")


def _terminate_process_group(process, cleanup_records, reason):
    pgid = process.pid
    leader_state = "running" if process.poll() is None else "exited"
    if not _process_group_exists(pgid):
        cleanup_records.append(
            f"pid={process.pid} pgid={pgid} leader={leader_state} "
            f"group=absent reason={reason}")
        return
    os.killpg(pgid, signal.SIGTERM)
    if process.poll() is None:
        try:
            process.wait(timeout=PROCESS_GROUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            cleanup_records.append(
                f"pid={process.pid} pgid={pgid} leader=running "
                f"group=term_pending reason={reason}")
    if _wait_for_process_group_exit(pgid, PROCESS_GROUP_GRACE_SECONDS):
        cleanup_records.append(
            f"pid={process.pid} pgid={pgid} leader={leader_state} "
            f"group=terminated reason={reason}")
    else:
        os.killpg(pgid, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=PROCESS_GROUP_GRACE_SECONDS)
        if not _wait_for_process_group_exit(
                pgid, PROCESS_GROUP_GRACE_SECONDS):
            raise RuntimeError(
                f"process group {pgid} survived SIGKILL during {reason}")
        cleanup_records.append(
            f"pid={process.pid} pgid={pgid} leader={leader_state} "
            f"group=killed reason={reason}")
    if process.poll() is None:
        process.wait(timeout=PROCESS_GROUP_GRACE_SECONDS)


def _cleanup_active_process_groups(cleanup_records, reason):
    errors = []
    for pgid in tuple(sorted(ACTIVE_PROCESS_GROUPS)):
        process = ACTIVE_PROCESS_GROUPS[pgid]
        try:
            _terminate_process_group(process, cleanup_records, reason)
            _release_process(process, cleanup_records)
        except BaseException as exc:
            cleanup_records.append(
                f"pid={process.pid} pgid={pgid} group=cleanup_failed "
                f"reason={reason} error_type={type(exc).__name__} error={exc}")
            errors.append((pgid, exc))
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
    if errors:
        details = "; ".join(
            f"PGID {pgid}: {type(exc).__name__}: {exc}"
            for pgid, exc in errors)
        raise RuntimeError(f"process-group cleanup failed: {details}")


def _exception_cleanup_reason(exc):
    if isinstance(exc, RunnerTerminationSignal):
        return f"signal_{signal.Signals(exc.signum).name}"
    return f"exception_{type(exc).__name__}"


def _prepare_process_artifacts(process_dir, process_records, stage, command):
    if PROCESS_STAGE_PATTERN.fullmatch(stage) is None:
        raise ValueError(f"invalid subprocess stage {stage!r}")
    if not process_dir.is_dir():
        raise RuntimeError(f"missing subprocess artifact directory {process_dir}")
    sequence = len(process_records) + 1
    stem = f"{sequence:04d}_{stage}"
    stdout_path = process_dir / f"{stem}.stdout"
    stderr_path = process_dir / f"{stem}.stderr"
    events_path = process_dir / f"{stem}.process"
    stdout_path.touch(exist_ok=False)
    stderr_path.touch(exist_ok=False)
    events_path.write_text(
        "event=prepared\n"
        f"stage={stage}\n"
        f"command={_format_command(command)}\n"
        f"stdout={stdout_path.name}\n"
        f"stderr={stderr_path.name}\n")
    with (process_dir / "process_manifest.txt").open("a") as manifest_stream:
        manifest_stream.write(
            f"sequence={sequence} stage={stage} events={events_path.name} "
            f"stdout={stdout_path.name} stderr={stderr_path.name}\n")
        manifest_stream.flush()
        os.fsync(manifest_stream.fileno())
    process_records.append(stem)
    return stdout_path, stderr_path, events_path


def _append_process_event(events_path, event, fields):
    if "\n" in event:
        raise ValueError("subprocess event contains a newline")
    lines = [f"event={event}"]
    for key, value in fields:
        text = str(value)
        if "\n" in key or "\n" in text:
            raise ValueError("subprocess event field contains a newline")
        lines.append(f"{key}={text}")
    with events_path.open("a") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _spawn_registered_process(
        command, *, cwd, environment, cleanup_records, events_path,
        stdout_stream, stderr_stream, pass_fds, stdin=None):
    if not hasattr(signal, "pthread_sigmask"):
        raise RuntimeError("pthread_sigmask is required for atomic child spawn")
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, TERMINATION_SIGNALS)
    process = None
    try:
        try:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=environment,
                    stdin=stdin,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    text=True,
                    start_new_session=True,
                    pass_fds=tuple(pass_fds),
                    preexec_fn=partial(
                        signal.pthread_sigmask, signal.SIG_SETMASK,
                        previous_mask),
                )
            except BaseException as exc:
                _append_process_event(
                    events_path, "spawn_failed",
                    (("error_type", type(exc).__name__), ("error", exc)))
                raise
            _register_process(process, cleanup_records)
            _append_process_event(
                events_path, "spawned",
                (("pid", process.pid), ("pgid", process.pid)))
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        return process
    except BaseException as exc:
        if process is not None:
            registered = process.pid in ACTIVE_PROCESS_GROUPS
            if (registered
                    and ACTIVE_PROCESS_GROUPS[process.pid] is not process):
                raise RuntimeError(
                    f"process group {process.pid} registration changed owner"
                ) from exc
            _terminate_process_group(
                process, cleanup_records, _exception_cleanup_reason(exc))
            if registered:
                _release_process(process, cleanup_records)
            _append_process_event(
                events_path, "interrupted",
                (("pid", process.pid), ("pgid", process.pid),
                 ("error_type", type(exc).__name__), ("error", exc)))
        raise


def _run_capture(command, *, cwd, environment, cleanup_records,
                 timeout_seconds, process_dir, process_records, stage,
                 pass_fds=(), stdout_bytes=False):
    stdout_path, stderr_path, events_path = _prepare_process_artifacts(
        process_dir, process_records, stage, command)
    stdout_stream = stdout_path.open("a")
    stderr_stream = stderr_path.open("a")
    try:
        process = _spawn_registered_process(
            command, cwd=cwd, environment=environment,
            cleanup_records=cleanup_records, events_path=events_path,
            stdout_stream=stdout_stream, stderr_stream=stderr_stream,
            pass_fds=pass_fds)
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process, cleanup_records, "timeout")
            _release_process(process, cleanup_records)
            _append_process_event(
                events_path, "timeout",
                (("pid", process.pid), ("pgid", process.pid),
                 ("timeout_seconds", timeout_seconds)))
            raise RuntimeError(
                f"command timed out after {timeout_seconds}s: "
                f"{_format_command(command)}; partial stdout={stdout_path}; "
                f"partial stderr={stderr_path}") from exc
        except BaseException as exc:
            _terminate_process_group(
                process, cleanup_records, _exception_cleanup_reason(exc))
            _release_process(process, cleanup_records)
            _append_process_event(
                events_path, "interrupted",
                (("pid", process.pid), ("pgid", process.pid),
                 ("error_type", type(exc).__name__), ("error", exc)))
            raise
    finally:
        stdout_stream.flush()
        stderr_stream.flush()
        stdout_stream.close()
        stderr_stream.close()
    returncode = process.returncode
    try:
        if _process_group_exists(process.pid):
            _terminate_process_group(
                process, cleanup_records, "leader_exited_with_descendants")
            _release_process(process, cleanup_records)
            _append_process_event(
                events_path, "descendants_rejected",
                (("pid", process.pid), ("pgid", process.pid)))
            raise RuntimeError(
                f"command leader exited but PGID {process.pid} retained "
                f"descendants: {_format_command(command)}")
        cleanup_records.append(
            f"pid={process.pid} pgid={process.pid} leader=exited group=absent "
            f"returncode={returncode}")
        _release_process(process, cleanup_records)
    finally:
        _append_process_event(
            events_path, "exited",
            (("pid", process.pid), ("pgid", process.pid),
             ("returncode", returncode)))
    if stdout_bytes:
        stdout = stdout_path.read_bytes()
    else:
        stdout = stdout_path.read_text()
    stderr = stderr_path.read_text()
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit {returncode}: "
            f"{_format_command(command)}; stdout={stdout_path}; "
            f"stderr={stderr_path}\nstderr:\n{stderr}")
    return process.pid, stdout, stderr


def _git_identity(environment, cleanup_records, process_dir, process_records):
    _, baseline, baseline_stderr = _run_capture(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        environment=environment, cleanup_records=cleanup_records,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
        process_records=process_records, stage="git_rev_parse")
    if baseline_stderr:
        raise RuntimeError("git rev-parse emitted stderr")
    baseline = baseline.strip()
    if re.fullmatch(r"[0-9a-f]{40}", baseline) is None:
        raise RuntimeError("git rev-parse did not return a commit hash")
    _, status, status_stderr = _run_capture(
        ["git", "status", "--short"], cwd=REPO_ROOT,
        environment=environment, cleanup_records=cleanup_records,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
        process_records=process_records, stage="git_status")
    if status_stderr:
        raise RuntimeError("git status emitted stderr")
    if status:
        raise RuntimeError(
            "hardware capture requires a clean deployed baseline:\n" + status)
    return baseline


def _git_blob(
        baseline, relative_path, environment, cleanup_records, process_dir,
        process_records):
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Git blob path must be repository-relative")
    revision = f"{baseline}:{relative_path.as_posix()}"
    _, object_id, object_stderr = _run_capture(
        ["git", "rev-parse", revision], cwd=REPO_ROOT,
        environment=environment, cleanup_records=cleanup_records,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
        process_records=process_records, stage="git_source_blob_oid")
    if object_stderr:
        raise RuntimeError("git source blob rev-parse emitted stderr")
    object_id = object_id.strip()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id) is None:
        raise RuntimeError("git source blob has invalid object identity")
    _, content, content_stderr = _run_capture(
        ["git", "show", revision], cwd=REPO_ROOT,
        environment=environment, cleanup_records=cleanup_records,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
        process_records=process_records, stage="git_source_blob_content",
        stdout_bytes=True)
    if content_stderr:
        raise RuntimeError("git show source blob emitted stderr")
    return GitBlob(
        object_id=object_id,
        content=content,
        sha256=hashlib.sha256(content).hexdigest())


def _expected_sass_kernels():
    expected = []
    latency_enums = tuple(
        PRIMITIVE_ENUM[name]
        for name in LATENCY_TARGET_PRIMITIVES + (CONTROL_PRIMITIVE,))
    for primitive in latency_enums:
        for unroll in LATENCY_UNROLLS:
            expected.append(("latency", primitive, unroll, 1))
    for primitive in (
            PRIMITIVE_ENUM[name]
            for name in TARGET_PRIMITIVES + (CONTROL_PRIMITIVE,)):
        for chains in THROUGHPUT_CHAINS:
            expected.append(
                ("throughput", primitive, THROUGHPUT_UNROLL, chains))
    return tuple(expected)


def _sass_sections(sass_text):
    sections = {}
    symbol = None
    lines = []
    for line in sass_text.splitlines():
        marker = FUNCTION_MARKER.search(line)
        if marker is not None:
            if symbol is not None:
                if symbol in sections:
                    raise ValueError(f"duplicate SASS function {symbol}")
                sections[symbol] = tuple(lines)
            symbol = marker.group(1)
            lines = []
        elif symbol is not None:
            lines.append(line)
    if symbol is not None:
        if symbol in sections:
            raise ValueError(f"duplicate SASS function {symbol}")
        sections[symbol] = tuple(lines)
    if not sections:
        raise ValueError("cuobjdump output contains no SASS function")
    return sections


def _decode_kernel_symbol(symbol):
    dependent = DEPENDENT_SYMBOL.match(symbol)
    if dependent is not None:
        return (
            "latency", int(dependent.group(1)), int(dependent.group(2)), 1)
    throughput = THROUGHPUT_SYMBOL.match(symbol)
    if throughput is not None:
        return (
            "throughput", int(throughput.group(1)), THROUGHPUT_UNROLL,
            int(throughput.group(2)))
    return None


def _parse_sass_instruction(symbol, line):
    parsed = SASS_INSTRUCTION.search(line)
    if parsed is None:
        raise ValueError(f"{symbol} cannot parse SASS instruction {line!r}")
    predicate, opcode, operands = parsed.groups()
    return predicate, opcode, operands


def _timed_instructions(symbol, lines, mode):
    if mode == "latency":
        expected_timer = "SR_CLOCKLO"
        expected_domain = "clock64"
    elif mode == "throughput":
        expected_timer = "SR_CLOCKLO"
        expected_domain = "clock64"
    else:
        raise ValueError(f"{symbol} has unknown timing mode {mode!r}")
    timer_lines = [
        (index, line) for index, line in enumerate(lines)
        if "CS2R" in line
        and ("SR_CLOCKLO" in line or "SR_GLOBALTIMERLO" in line)
    ]
    wrong_timer_lines = [
        line for _, line in timer_lines if expected_timer not in line]
    if wrong_timer_lines:
        raise ValueError(
            f"{symbol} {mode} SASS mixes timing domains: "
            f"{wrong_timer_lines!r}")
    timer_positions = [
        index for index, line in timer_lines if expected_timer in line
    ]
    if len(timer_positions) != 2:
        raise ValueError(
            f"{symbol} must contain exactly two {expected_domain} SASS reads")
    start, end = timer_positions
    if end <= start + 1:
        raise ValueError(f"{symbol} has an empty timed SASS region")
    instructions = []
    for line in lines[start + 1:end]:
        instruction = INSTRUCTION.search(line)
        if instruction is not None:
            parsed = _parse_sass_instruction(symbol, line)
            if parsed[1] != instruction.group(1):
                raise ValueError(
                    f"{symbol} SASS opcode parsing disagreed for {line!r}")
            instructions.append(parsed)
    if not instructions:
        raise ValueError(f"{symbol} timed SASS region contains no opcode")
    start_timer = _parse_sass_instruction(symbol, lines[start])
    end_timer = _parse_sass_instruction(symbol, lines[end])
    for label, timer in (("start", start_timer), ("end", end_timer)):
        predicate, opcode, operands = timer
        if (predicate is not None or opcode != "CS2R"
                or operands is None
                or not operands.endswith(f", {expected_timer}")):
            raise ValueError(
                f"{symbol} {label} timer is not an unpredicated "
                f"{expected_domain} "
                "CS2R")
    return tuple(instructions)


def _register_components(token):
    stripped = token.strip()
    match = REGISTER_OPERAND.fullmatch(stripped)
    if match is None:
        if re.search(r"\bU?R(?:Z|[0-9]+)", stripped) is not None:
            raise ValueError(f"unsupported SASS register operand {token!r}")
        return frozenset()
    register_kind, register_index, pair_width = match.groups()
    if register_index == "Z":
        return frozenset()
    base = int(register_index)
    registers = {f"{register_kind}{base}"}
    if pair_width == "64":
        registers.add(f"{register_kind}{base + 1}")
    return frozenset(registers)


def _parse_target_instruction(symbol, opcode, operand_text):
    if operand_text is None:
        raise ValueError(
            f"{symbol} cannot parse operands for target opcode {opcode}")
    operands = tuple(part.strip() for part in operand_text.split(","))
    if opcode == "FFMA":
        expected_arity = 4
        data_source_indices = (1, 2, 3)
    elif opcode in ("FADD", "FMUL"):
        expected_arity = 3
        data_source_indices = (1, 2)
    elif opcode in ("MUFU.EX2", "MUFU.RCP", "MUFU.RSQ"):
        expected_arity = 2
        data_source_indices = (1,)
    elif opcode == "LOP3.LUT":
        expected_arity = 6
        data_source_indices = (1, 2, 3)
    else:
        raise ValueError(f"{symbol} has unsupported target opcode {opcode}")
    if len(operands) != expected_arity:
        raise ValueError(
            f"{symbol} {opcode} has {len(operands)} operands; "
            f"expected {expected_arity}")
    destination = _register_components(operands[0])
    if not destination:
        raise ValueError(
            f"{symbol} {opcode} destination is not a register: {operands[0]!r}")
    sources = set()
    for index in data_source_indices:
        source = operands[index]
        source_registers = _register_components(source)
        if (not source_registers
                and REGISTER_OPERAND.fullmatch(source) is None
                and SASS_IMMEDIATE.fullmatch(source) is None):
            raise ValueError(
                f"{symbol} {opcode} has unsupported source operand {source!r}")
        sources.update(source_registers)
    if opcode == "LOP3.LUT":
        if re.fullmatch(r"0x[0-9a-fA-F]+", operands[4]) is None:
            raise ValueError(
                f"{symbol} LOP3.LUT has unsupported LUT operand {operands[4]!r}")
        if re.fullmatch(r"!?PT", operands[5]) is None:
            raise ValueError(
                f"{symbol} LOP3.LUT has unsupported predicate {operands[5]!r}")
    return destination, frozenset(sources)


def _validate_dependency_contract(symbol, mode, parsed_instructions):
    if mode != "latency":
        raise ValueError(
            f"{symbol} single-chain validator requires latency mode")
    prior_outputs = set()
    previous_output = None
    for index, (destination, sources) in enumerate(
            parsed_instructions, start=1):
        if previous_output is not None:
            if not previous_output.issubset(sources):
                raise ValueError(
                    f"{symbol} latency dependency chain breaks at "
                    f"instruction {index}")
            earlier_inputs = sources.intersection(prior_outputs)
            if earlier_inputs != set(previous_output):
                raise ValueError(
                    f"{symbol} latency region mixes dependency chains at "
                    f"instruction {index}")
        prior_outputs.update(destination)
        previous_output = destination
    return previous_output


def _validate_throughput_dependency_chains(
        symbol, parsed_instructions, expected_chains, expected_rounds):
    last_writer = {}
    instruction_chain = {}
    child_count = {}
    chain_lengths = {}
    next_chain = 0
    for index, (destination, sources) in enumerate(parsed_instructions):
        predecessors = {
            last_writer[register]
            for register in sources if register in last_writer
        }
        if len(predecessors) > 1:
            raise ValueError(
                f"{symbol} throughput instruction {index + 1} merges "
                f"multiple prior target outputs")
        if predecessors:
            predecessor = next(iter(predecessors))
            chain = instruction_chain[predecessor]
            child_count[predecessor] += 1
        else:
            chain = next_chain
            next_chain += 1
            chain_lengths[chain] = 0
        instruction_chain[index] = chain
        child_count[index] = 0
        chain_lengths[chain] += 1
        for register in destination:
            if (register in last_writer
                    and instruction_chain[last_writer[register]] != chain
                    and child_count[last_writer[register]] == 0):
                raise ValueError(
                    f"{symbol} throughput instruction {index + 1} has "
                    f"cross-chain WAW on live output {register}")
            last_writer[register] = index
    if next_chain != expected_chains:
        raise ValueError(
            f"{symbol} throughput dependency graph has {next_chain} roots; "
            f"expected {expected_chains} independent chains")
    for chain in range(expected_chains):
        if chain_lengths[chain] != expected_rounds:
            raise ValueError(
                f"{symbol} throughput chain {chain} has "
                f"{chain_lengths[chain]} instructions; expected "
                f"{expected_rounds}")
        chain_instructions = [
            index for index, observed_chain in instruction_chain.items()
            if observed_chain == chain
        ]
        terminals = [
            index for index in chain_instructions if child_count[index] == 0]
        branches = [
            index for index in chain_instructions if child_count[index] > 1]
        if len(terminals) != 1 or branches:
            raise ValueError(
                f"{symbol} throughput chain {chain} is not one linear "
                "recurrence")


def _validate_latency_endpoint(symbol, endpoint_instructions, final_output):
    if len(endpoint_instructions) != 3:
        raise ValueError(
            f"{symbol} latency endpoint must contain exactly three "
            "control instructions")
    xor_predicate, xor_opcode, xor_operands_text = endpoint_instructions[0]
    if xor_predicate is not None or xor_opcode != "LOP3.LUT":
        raise ValueError(f"{symbol} latency endpoint must start with LOP3.LUT")
    xor_destination, xor_sources = _parse_target_instruction(
        symbol, xor_opcode, xor_operands_text)
    xor_operands = tuple(
        part.strip() for part in xor_operands_text.split(","))
    if (xor_sources != final_output
            or xor_operands[2].lower() != ENDPOINT_XOR_IMMEDIATE
            or int(xor_operands[2], 16) == 0
            or xor_operands[3:] != ("RZ", "0x3c", "!PT")):
        raise ValueError(
            f"{symbol} latency endpoint XOR is not a nonzero transform of "
            "the final target output")

    compare_predicate, compare_opcode, compare_operands_text = (
        endpoint_instructions[1])
    if (compare_predicate is not None
            or compare_opcode != "ISETP.NE.U32.AND"
            or compare_operands_text is None):
        raise ValueError(
            f"{symbol} latency endpoint lacks the exact predicate producer")
    compare_operands = tuple(
        part.strip() for part in compare_operands_text.split(","))
    if len(compare_operands) != 5:
        raise ValueError(
            f"{symbol} latency endpoint predicate has invalid operands")
    endpoint_predicate = compare_operands[0]
    if (re.fullmatch(r"P[0-7]", endpoint_predicate) is None
            or compare_operands[1] != "PT"
            or _register_components(compare_operands[2]) != final_output
            or _register_components(compare_operands[3]) != xor_destination
            or compare_operands[4] != "PT"):
        raise ValueError(
            f"{symbol} latency endpoint predicate does not compare the final "
            "output with its nonzero XOR")

    exit_predicate, exit_opcode, exit_operands = endpoint_instructions[2]
    if (exit_predicate != f"@!{endpoint_predicate}"
            or exit_opcode != "EXIT" or exit_operands is not None):
        raise ValueError(
            f"{symbol} latency endpoint does not guard the end timer with "
            "the complementary failure exit")


def _validate_sass(sass_text, expected_kernels=None):
    expected = (
        _expected_sass_kernels()
        if expected_kernels is None else tuple(expected_kernels))
    expected_set = set(expected)
    sections = _sass_sections(sass_text)
    decoded = {}
    for symbol, lines in sections.items():
        identity = _decode_kernel_symbol(symbol)
        if identity is None:
            continue
        if identity in decoded:
            raise ValueError(f"duplicate instantiated kernel identity {identity}")
        decoded[identity] = (symbol, lines)
    actual_set = set(decoded)
    if actual_set != expected_set:
        raise ValueError(
            "SASS instantiated-kernel mismatch: "
            f"missing={sorted(expected_set - actual_set)}, "
            f"unexpected={sorted(actual_set - expected_set)}")
    for identity in expected:
        symbol, lines = decoded[identity]
        _, primitive, unroll, chains = identity
        expected_opcode = PRIMITIVE_OPCODE[primitive]
        expected_count = unroll * chains
        if expected_count > MAX_TIMED_OPCODES_PER_THREAD:
            raise ValueError(
                f"{symbol} timed footprint {expected_count} exceeds "
                f"{MAX_TIMED_OPCODES_PER_THREAD}")
        instructions = _timed_instructions(symbol, lines, identity[0])
        if identity[0] == "latency":
            if len(instructions) != expected_count + 3:
                raise ValueError(
                    f"{symbol} has {len(instructions)} timed instructions; "
                    f"expected {expected_count} target opcodes plus three "
                    "endpoint-control instructions")
            target_instructions = instructions[:expected_count]
            endpoint_instructions = instructions[expected_count:]
        elif identity[0] == "throughput":
            target_instructions = instructions
            endpoint_instructions = ()
        else:
            raise ValueError(f"{symbol} has unknown mode {identity[0]!r}")
        opcodes = tuple(
            opcode for _, opcode, _ in target_instructions)
        forbidden = tuple(
            opcode for opcode in opcodes
            if opcode.startswith(FORBIDDEN_TIMED_PREFIXES))
        if forbidden:
            raise ValueError(
                f"{symbol} timed region contains forbidden opcodes {forbidden}")
        if any(opcode != expected_opcode for opcode in opcodes):
            raise ValueError(
                f"{symbol} timed region is not pure {expected_opcode}: {opcodes}")
        if len(opcodes) != expected_count:
            raise ValueError(
                f"{symbol} has {len(opcodes)} timed {expected_opcode} opcodes; "
                f"expected {expected_count}")
        if primitive == PRIMITIVE_ENUM[CONTROL_PRIMITIVE]:
            if any(opcode in TARGET_OPCODES for opcode in opcodes):
                raise ValueError(
                    f"{symbol} control region contains a target opcode")
        parsed_instructions = tuple(
            _parse_target_instruction(symbol, opcode, operand_text)
            for _, opcode, operand_text in target_instructions)
        if identity[0] == "latency":
            final_output = _validate_dependency_contract(
                symbol, identity[0], parsed_instructions)
            _validate_latency_endpoint(
                symbol, endpoint_instructions, final_output)
        elif identity[0] == "throughput":
            _validate_throughput_dependency_chains(
                symbol, parsed_instructions, chains, unroll)
        else:
            raise ValueError(f"{symbol} has unknown mode {identity[0]!r}")
    return len(expected)


def _validate_ptxas(ptxas_text, expected_kernels=None):
    expected = set(
        _expected_sass_kernels()
        if expected_kernels is None else tuple(expected_kernels))
    lines = ptxas_text.splitlines()
    validated = set()
    for index, line in enumerate(lines):
        marker = PTXAS_FUNCTION.search(line.strip())
        if marker is None:
            continue
        symbol = marker.group(1)
        identity = _decode_kernel_symbol(symbol)
        if identity is None:
            continue
        if identity in validated:
            raise ValueError(f"duplicate ptxas function properties {identity}")
        evidence = " ".join(lines[index + 1:index + 4])
        resources = PTXAS_RESOURCES.search(evidence)
        registers = PTXAS_REGISTERS.search(evidence)
        if resources is None or registers is None:
            raise ValueError(f"ptxas resource evidence missing for {symbol}")
        if tuple(int(resources.group(item)) for item in (1, 2, 3)) != (0, 0, 0):
            raise ValueError(f"ptxas reports stack or spills for {symbol}")
        if int(registers.group(1)) <= 0:
            raise ValueError(f"ptxas reports invalid register use for {symbol}")
        validated.add(identity)
    if validated != expected:
        raise ValueError(
            "ptxas instantiated-kernel mismatch: "
            f"missing={sorted(expected - validated)}, "
            f"unexpected={sorted(validated - expected)}")
    return len(validated)


def _validate_cubin_package(list_elf_text, dump_ptx_text):
    elf_lines = tuple(
        line.strip() for line in list_elf_text.splitlines() if line.strip())
    if not elf_lines:
        raise ValueError("cuobjdump listed no embedded cubin")
    for line in elf_lines:
        if re.fullmatch(
                r"ELF file\s+[0-9]+:\s+\S+\.sm_80\.cubin", line) is None:
            raise ValueError(
                f"binary contains a non-sm_80 cubin entry: {line!r}")
    if ("Fatbin ptx code:" in dump_ptx_text
            or re.search(r"^\.version\s", dump_ptx_text, re.MULTILINE)
            is not None):
        raise ValueError("binary contains embedded PTX and permits runtime JIT")
    return len(elf_lines)


def _parse_positive_int(text, field):
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"raw field {field} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"raw field {field} must be positive")
    return value


def _parse_finite_positive(text, field):
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"raw field {field} must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"raw field {field} must be finite and positive")
    return value


def _pci_identity(text):
    if PCI_BUS_ID_PATTERN.fullmatch(text) is None:
        raise ValueError(f"invalid PCI bus identity {text!r}")
    domain_text, bus_text, device_function = text.split(":")
    device_text, function_text = device_function.split(".")
    return (
        int(domain_text, 16), int(bus_text, 16), int(device_text, 16),
        int(function_text, 16))


def _parse_occupancy_output(stdout, stderr, cell):
    if stderr:
        raise ValueError("occupancy query emitted stderr")
    lines = stdout.splitlines()
    if len(lines) != 1:
        raise ValueError("occupancy query must emit exactly one line")
    fields = lines[0].split(",")
    if len(fields) != len(OCCUPANCY_FIELDS):
        raise ValueError("occupancy query schema changed")
    row = dict(zip(OCCUPANCY_FIELDS, fields))
    exact = {
        "record_type": "occupancy",
        "mode": cell.mode,
        "primitive": cell.primitive,
        "unroll": str(cell.unroll),
        "chains": str(cell.chains),
        "warps": str(cell.warps),
        "sm_count": str(SM_COUNT),
    }
    for field, expected in exact.items():
        if row[field] != expected:
            raise ValueError(
                f"occupancy query changed {field}: {row[field]!r}")
    occupancy = _parse_positive_int(
        row["occupancy_limit_ctas_per_sm"],
        "occupancy_limit_ctas_per_sm")
    grid_limit = _parse_positive_int(
        row["cooperative_grid_limit"], "cooperative_grid_limit")
    if grid_limit != occupancy * SM_COUNT:
        raise ValueError("occupancy cooperative grid limit is inconsistent")
    return occupancy


def _validate_raw_output(stdout, cell, expected_uuid, expected_pci,
                         paired_resident, specialization_occupancy):
    lines = stdout.splitlines()
    if len(lines) != TRIALS + 1:
        raise ValueError(
            f"binary emitted {len(lines)} lines; expected {TRIALS + 1}")
    if lines[0] != BINARY_RAW_HEADER:
        raise ValueError("binary raw header does not match ABI")
    validated = []
    for expected_trial, line in enumerate(lines[1:], start=1):
        fields = line.split(",")
        if len(fields) != len(RAW_FIELDS):
            raise ValueError("binary emitted an invalid raw field count")
        row = dict(zip(RAW_FIELDS, fields))
        trial = _parse_positive_int(row["trial"], "trial")
        if trial != expected_trial:
            raise ValueError("binary trial order changed")
        if cell.mode not in ("latency", "throughput"):
            raise ValueError(f"unknown matrix mode {cell.mode!r}")
        exact = {
            "mode": cell.mode,
            "primitive": cell.primitive,
            "clock_domain": "clock64",
            "device_name": EXPECTED_DEVICE_NAME,
            "device_uuid": expected_uuid,
            "residency_cell": cell.residency,
        }
        for field, expected in exact.items():
            if row[field] != expected:
                raise ValueError(
                    f"binary changed {field}: {row[field]!r} != {expected!r}")
        if _pci_identity(row["pci_bus_id"]) != _pci_identity(expected_pci):
            raise ValueError("binary changed the selected PCI identity")
        _parse_positive_int(
            row["device_property_clock_khz"], "device_property_clock_khz")
        sm_count = _parse_positive_int(row["sm_count"], "sm_count")
        if sm_count != SM_COUNT:
            raise ValueError(f"binary reported {sm_count} SMs instead of {SM_COUNT}")
        grid = _parse_positive_int(row["grid_blocks"], "grid_blocks")
        block = _parse_positive_int(row["block_threads"], "block_threads")
        warps = _parse_positive_int(row["warps_per_cta"], "warps_per_cta")
        resident = _parse_positive_int(
            row["paired_resident_ctas_per_sm"],
            "paired_resident_ctas_per_sm")
        occupancy = _parse_positive_int(
            row["specialization_occupancy_limit_ctas_per_sm"],
            "specialization_occupancy_limit_ctas_per_sm")
        chains = _parse_positive_int(row["chains"], "chains")
        unroll = _parse_positive_int(row["unroll"], "unroll")
        if block != cell.warps * WARP_SIZE or warps != cell.warps:
            raise ValueError("binary changed the requested CTA warp geometry")
        if chains != cell.chains or unroll != cell.unroll:
            raise ValueError("binary changed the requested chain/unroll geometry")
        if resident != paired_resident or occupancy != specialization_occupancy:
            raise ValueError("binary changed the proven paired occupancy geometry")
        if resident > occupancy:
            raise ValueError("paired residency exceeds specialization occupancy")
        if cell.residency == "one" and resident != 1:
            raise ValueError("one-residency cell did not use one CTA per SM")
        if cell.mode == "latency":
            expected_grid = 1
        elif cell.mode == "throughput":
            expected_grid = SM_COUNT * resident
        else:
            raise ValueError(f"unknown matrix mode {cell.mode!r}")
        if grid != expected_grid:
            raise ValueError("binary grid is not the proven paired grid")
        opcode_count = _parse_positive_int(
            row["timed_opcode_count_per_thread"],
            "timed_opcode_count_per_thread")
        expected_opcode_count = cell.unroll * cell.chains
        if opcode_count != expected_opcode_count:
            raise ValueError("binary timed opcode count drifted")
        if opcode_count > MAX_TIMED_OPCODES_PER_THREAD:
            raise ValueError("binary timed instruction footprint is too large")
        issued = _parse_positive_int(
            row["issued_warp_opcodes"], "issued_warp_opcodes")
        expected_issued = grid * cell.warps * expected_opcode_count
        if issued != expected_issued:
            raise ValueError("binary issued-warp-opcode total drifted")
        active_sm_count = _parse_positive_int(
            row["active_sm_count"], "active_sm_count")
        if cell.mode == "latency":
            if active_sm_count != 1:
                raise ValueError("latency must retain exactly one active SM")
            sm_start = _parse_positive_int(
                row["sm_clock_start_cycle"], "sm_clock_start_cycle")
            sm_end = _parse_positive_int(
                row["sm_clock_end_cycle"], "sm_clock_end_cycle")
            elapsed_cycles = _parse_positive_int(
                row["elapsed_sm_cycles"], "elapsed_sm_cycles")
            if (sm_end <= sm_start
                    or elapsed_cycles != sm_end - sm_start):
                raise ValueError("binary clock64 interval is inconsistent")
            for field in (
                    "min_local_elapsed_sm_cycles",
                    "max_local_elapsed_sm_cycles",
                    "sum_local_elapsed_sm_cycles",
                    "lane_ops_per_subcore_cycle",
                    "fma_flops_per_subcore_cycle",
                    "warp_instructions_per_subcore_cycle"):
                if row[field] != NOT_APPLICABLE:
                    raise ValueError(f"latency field {field} must be not_applicable")
        elif cell.mode == "throughput":
            if active_sm_count != SM_COUNT:
                raise ValueError(
                    "throughput did not retain every expected active SM")
            for field in (
                    "sm_clock_start_cycle", "sm_clock_end_cycle",
                    "elapsed_sm_cycles"):
                if row[field] != NOT_APPLICABLE:
                    raise ValueError(
                        f"throughput field {field} must be not_applicable")
            minimum_elapsed = _parse_positive_int(
                row["min_local_elapsed_sm_cycles"],
                "min_local_elapsed_sm_cycles")
            maximum_elapsed = _parse_positive_int(
                row["max_local_elapsed_sm_cycles"],
                "max_local_elapsed_sm_cycles")
            sum_elapsed = _parse_positive_int(
                row["sum_local_elapsed_sm_cycles"],
                "sum_local_elapsed_sm_cycles")
            if active_sm_count == 1:
                extrema_consistent = (
                    minimum_elapsed == maximum_elapsed == sum_elapsed)
            else:
                spread = maximum_elapsed - minimum_elapsed
                extrema_consistent = (
                    minimum_elapsed <= maximum_elapsed
                    and sum_elapsed
                        >= minimum_elapsed * active_sm_count + spread
                    and sum_elapsed
                        <= maximum_elapsed * active_sm_count - spread)
            if not extrema_consistent:
                raise ValueError(
                    "binary local-SM clock64 aggregate extrema are "
                    "inconsistent")
            lane_rate = _parse_finite_positive(
                row["lane_ops_per_subcore_cycle"],
                "lane_ops_per_subcore_cycle")
            warp_rate = _parse_finite_positive(
                row["warp_instructions_per_subcore_cycle"],
                "warp_instructions_per_subcore_cycle")
            elapsed_subcore_cycles = sum_elapsed * SUBCORES_PER_SM
            if cell.primitive == "fma_f32":
                operation_scale = 2
            elif cell.primitive in ALL_PRIMITIVES[1:]:
                operation_scale = 1
            else:
                raise ValueError(f"unknown matrix primitive {cell.primitive!r}")
            expected_lane_rate = (
                issued * WARP_SIZE * operation_scale
                / elapsed_subcore_cycles)
            expected_warp_rate = issued / elapsed_subcore_cycles
            if not math.isclose(
                    lane_rate, expected_lane_rate,
                    rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("binary lane rate is arithmetically inconsistent")
            if not math.isclose(
                    warp_rate, expected_warp_rate,
                    rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("binary warp rate is arithmetically inconsistent")
            if cell.primitive == "fma_f32":
                fma_rate = _parse_finite_positive(
                    row["fma_flops_per_subcore_cycle"],
                    "fma_flops_per_subcore_cycle")
                if not math.isclose(
                        fma_rate, expected_lane_rate,
                        rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError("binary FMA rate is arithmetically inconsistent")
            elif row["fma_flops_per_subcore_cycle"] != NOT_APPLICABLE:
                raise ValueError("non-FMA rate must be not_applicable")
        else:
            raise ValueError(f"unknown matrix mode {cell.mode!r}")
        validated.append(line)
    return tuple(validated)


def _derive_latency_cycles(rows):
    grouped = {
        primitive: {unroll: [] for unroll in LATENCY_UNROLLS}
        for primitive in LATENCY_TARGET_PRIMITIVES
    }
    for row in rows:
        if row["mode"] != "latency":
            raise ValueError("latency derivation received a non-latency row")
        primitive = row["primitive"]
        if primitive == CONTROL_PRIMITIVE:
            continue
        if primitive not in grouped:
            raise ValueError(
                f"latency derivation received unknown primitive {primitive!r}")
        unroll = _parse_positive_int(row["unroll"], "unroll")
        if unroll not in LATENCY_UNROLLS:
            raise ValueError(
                f"latency derivation received unsupported unroll {unroll}")
        elapsed = _parse_positive_int(
            row["elapsed_sm_cycles"], "elapsed_sm_cycles")
        grouped[primitive][unroll].append(elapsed)
    records = []
    for primitive in LATENCY_TARGET_PRIMITIVES:
        medians = {}
        for unroll in LATENCY_UNROLLS:
            samples = grouped[primitive][unroll]
            if len(samples) != TRIALS:
                raise ValueError(
                    f"{primitive} unroll {unroll} has {len(samples)} "
                    f"latency samples; expected {TRIALS}")
            medians[unroll] = median(samples)
        for lower, upper in zip(LATENCY_UNROLLS, LATENCY_UNROLLS[1:]):
            if medians[upper] <= medians[lower]:
                raise ValueError(
                    f"{primitive} clock64 medians are not strictly "
                    f"increasing from unroll {lower} to {upper}")
        adjacent = (
            (medians[8] - medians[4]) / 4,
            (medians[16] - medians[8]) / 8,
            (medians[32] - medians[16]) / 16,
        )
        canonical = (medians[32] - medians[4]) / 28
        records.append({
            "primitive": primitive,
            "clock_domain": "clock64",
            "unit": "sm_cycles_per_dependent_warp_opcode",
            "evidence_status": "mechanism_candidate_only",
            "sample_count_per_unroll": str(TRIALS),
            "median_elapsed_cycles_u4": str(medians[4]),
            "median_elapsed_cycles_u8": str(medians[8]),
            "median_elapsed_cycles_u16": str(medians[16]),
            "median_elapsed_cycles_u32": str(medians[32]),
            "canonical_span": "u4_to_u32_fixed",
            "latency_sm_cycles_per_dependent_warp_opcode": (
                f"{canonical:.12f}"),
            "adjacent_slope_u4_u8": f"{adjacent[0]:.12f}",
            "adjacent_slope_u8_u16": f"{adjacent[1]:.12f}",
            "adjacent_slope_u16_u32": f"{adjacent[2]:.12f}",
            "control_subtracted": "false",
        })
    return tuple(records)


def _write_latency_derivation(path, rows):
    records = _derive_latency_cycles(rows)
    with path.open("x") as stream:
        stream.write(LATENCY_DERIVATION_HEADER + "\n")
        for record in records:
            stream.write(
                ",".join(record[field] for field in LATENCY_DERIVATION_FIELDS)
                + "\n")
    return records


def _validate_abi(stdout, stderr):
    if stderr:
        raise ValueError("ABI smoke command emitted stderr")
    if stdout.splitlines() != [BINARY_ABI_LINE]:
        raise ValueError("binary ABI smoke output does not match runner contract")


def _validate_device_query(text, expected_uuid, expected_pci, require_idle):
    lines = text.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("nvidia-smi query must resolve exactly one device")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 11:
        raise ValueError("nvidia-smi query schema changed")
    if fields[0] != expected_uuid:
        raise ValueError("nvidia-smi resolved a different device UUID")
    if fields[1] != EXPECTED_DEVICE_NAME:
        raise ValueError(
            f"expected exact device name {EXPECTED_DEVICE_NAME!r}, "
            f"got {fields[1]!r}")
    if _pci_identity(fields[2]) != _pci_identity(expected_pci):
        raise ValueError("nvidia-smi resolved a different PCI identity")
    if fields[10] != EXPECTED_COMPUTE_CAPABILITY:
        raise ValueError("nvidia-smi reported an unexpected compute capability")
    if not fields[3] or not fields[4]:
        raise ValueError("nvidia-smi omitted driver or pstate identity")
    try:
        int(fields[5])
        memory_used_mib = int(fields[6])
        utilization_percent = int(fields[7])
        sm_clock_mhz = int(fields[8])
        maximum_sm_clock_mhz = int(fields[9])
    except ValueError as exc:
        raise ValueError("nvidia-smi numeric fields are not integers") from exc
    if sm_clock_mhz <= 0 or maximum_sm_clock_mhz <= 0:
        raise ValueError("nvidia-smi reported a non-positive SM clock")
    if require_idle is True:
        if memory_used_mib != 0 or utilization_percent != 0:
            raise DeviceNotIdleError(
                "selected A800 is not idle: "
                f"memory_used_mib={memory_used_mib}, "
                f"utilization_percent={utilization_percent}")
    elif require_idle is False and memory_used_mib <= 0:
        raise RuntimeError(
            "overlapping live telemetry did not observe benchmark memory")
    elif require_idle is not None and require_idle is not False:
        raise TypeError("require_idle must be True, False, or None")
    if utilization_percent < 0 or utilization_percent > 100:
        raise ValueError("nvidia-smi utilization is outside [0, 100]")
    return lines[0]


def _validate_compute_process_query(
        text, benchmark_pid, expected_uuid, context_ready_verified):
    if context_ready_verified is not True:
        raise RuntimeError(
            "compute-process ownership requires a verified CUDA-context "
            "readiness handshake")
    lines = text.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(
            "overlapping telemetry must observe exactly one compute process")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 3:
        raise ValueError("nvidia-smi compute-process schema changed")
    try:
        observed_pid = int(fields[0])
        used_memory_mib = int(fields[2])
    except ValueError as exc:
        raise ValueError(
            "nvidia-smi compute-process numeric fields are invalid") from exc
    if observed_pid <= 0:
        raise ValueError("nvidia-smi reported a non-positive process PID")
    if fields[1] != expected_uuid:
        raise ValueError("compute-process query resolved a different GPU UUID")
    if used_memory_mib <= 0:
        raise RuntimeError("benchmark compute process has no resident GPU memory")
    if observed_pid == benchmark_pid:
        owner_contract = "direct_pid_match"
    else:
        owner_contract = "exclusive_ready_context_across_pid_namespace"
    return owner_contract, lines[0]


def _require_live_process(process, stage):
    returncode = process.poll()
    if returncode is not None:
        raise RuntimeError(
            f"benchmark process exited before {stage}: returncode={returncode}")


def _write_manifest(output_dir):
    manifest_path = output_dir / "manifest.sha256"
    entries = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != manifest_path:
            entries.append(f"{_sha256(path)}  {path.relative_to(output_dir)}")
    if not entries:
        raise RuntimeError("cannot write an empty raw manifest")
    manifest_path.write_text("\n".join(entries) + "\n")
    return tuple(entries)


def _format_command(command):
    if any("\n" in argument for argument in command):
        raise ValueError("command argument contains a newline")
    return shlex.join(command)


def _cell_command(binary, cell, expected_uuid, expected_pci, action,
                  grid_blocks, paired_resident):
    return [
        str(binary),
        "--action", action,
        "--primitive", cell.primitive,
        "--mode", cell.mode,
        "--unroll", str(cell.unroll),
        "--chains", str(cell.chains),
        "--warps", str(cell.warps),
        "--residency", cell.residency,
        "--trials", str(TRIALS),
        "--expected-device-uuid", expected_uuid,
        "--expected-pci-bus-id", expected_pci,
        "--grid-blocks", str(grid_blocks),
        "--paired-resident-ctas-per-sm", str(paired_resident),
    ]


def _device_query_command(nvidia_smi, expected_uuid):
    return [
        str(nvidia_smi), f"--query-gpu={DEVICE_QUERY_FIELDS}",
        "--format=csv,noheader,nounits", "-i", expected_uuid,
    ]


def _compute_process_query_command(nvidia_smi, expected_uuid):
    return [
        str(nvidia_smi),
        f"--query-compute-apps={COMPUTE_PROCESS_QUERY_FIELDS}",
        "--format=csv,noheader,nounits", "-i", expected_uuid,
    ]


def _query_device(nvidia_smi, expected_uuid, expected_pci, environment,
                  cleanup_records, command_records, require_idle, process_dir,
                  process_records, stage):
    command = _device_query_command(nvidia_smi, expected_uuid)
    command_records.append(_format_command(command))
    _, stdout, stderr = _run_capture(
        command, cwd=REPO_ROOT, environment=environment,
        cleanup_records=cleanup_records,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
        process_records=process_records, stage=stage)
    if stderr:
        raise RuntimeError("nvidia-smi query emitted stderr")
    return _validate_device_query(
        stdout, expected_uuid, expected_pci, require_idle)


def _query_device_until_idle(
        nvidia_smi, expected_uuid, expected_pci, environment,
        cleanup_records, command_records, process_dir, process_records,
        attempts_path):
    with attempts_path.open("x") as stream:
        stream.write("attempt," + DEVICE_QUERY_FIELDS + "\n")
        for attempt in range(1, POSTFLIGHT_IDLE_ATTEMPTS + 1):
            row = _query_device(
                nvidia_smi, expected_uuid, expected_pci, environment,
                cleanup_records, command_records, None, process_dir,
                process_records, f"device_post_attempt_{attempt}")
            stream.write(f"{attempt},{row}\n")
            stream.flush()
            try:
                return _validate_device_query(
                    row, expected_uuid, expected_pci, True)
            except DeviceNotIdleError:
                if attempt == POSTFLIGHT_IDLE_ATTEMPTS:
                    raise
                time.sleep(POSTFLIGHT_IDLE_INTERVAL_SECONDS)
    raise AssertionError("postflight idle loop terminated without a result")


def _query_compute_process(nvidia_smi, expected_uuid, benchmark_pid,
                           environment, cleanup_records, command_records,
                           process_dir, process_records, stage):
    command = _compute_process_query_command(nvidia_smi, expected_uuid)
    command_records.append(_format_command(command))
    _, stdout, stderr = _run_capture(
        command, cwd=REPO_ROOT, environment=environment,
        cleanup_records=cleanup_records,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
        process_records=process_records, stage=stage)
    if stderr:
        raise RuntimeError("nvidia-smi compute-process query emitted stderr")
    return _validate_compute_process_query(
        stdout, benchmark_pid, expected_uuid, True)


def _wait_for_telemetry_ready(process, stderr_path, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    with stderr_path.open() as stderr_stream:
        while True:
            line = stderr_stream.readline()
            if line:
                break
            _require_live_process(process, "telemetry-ready handshake")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"benchmark PID {process.pid} did not request live "
                    f"telemetry within {timeout_seconds}s; partial stderr="
                    f"{stderr_path}")
            time.sleep(0.01)
    if line != "telemetry_ready\n":
        raise RuntimeError(
            f"benchmark PID {process.pid} emitted invalid telemetry "
            f"handshake {line!r}")
    _require_live_process(process, "overlapping telemetry start")


def _run_benchmark_with_live_telemetry(
        command, *, cwd, environment, cleanup_records, timeout_seconds,
        nvidia_smi, expected_uuid, expected_pci, command_records, live_path,
        stage, process_dir, process_records, pass_fds=()):
    stdout_path, stderr_path, events_path = _prepare_process_artifacts(
        process_dir, process_records, stage, command)
    stdout_stream = stdout_path.open("a")
    stderr_stream = stderr_path.open("a")
    try:
        process = _spawn_registered_process(
            command, cwd=cwd, environment=environment,
            cleanup_records=cleanup_records, events_path=events_path,
            stdout_stream=stdout_stream, stderr_stream=stderr_stream,
            pass_fds=pass_fds, stdin=subprocess.PIPE)
        deadline = time.monotonic() + timeout_seconds
        try:
            _wait_for_telemetry_ready(process, stderr_path, timeout_seconds)
            device_query = _query_device(
                nvidia_smi, expected_uuid, expected_pci, environment,
                cleanup_records, command_records, False, process_dir,
                process_records, f"{stage}_live_device")
            owner_contract, process_query = _query_compute_process(
                nvidia_smi, expected_uuid, process.pid, environment,
                cleanup_records, command_records, process_dir,
                process_records, f"{stage}_live_process")
            _require_live_process(process, "overlapping telemetry completion")
            with live_path.open("a") as live_stream:
                live_stream.write(
                    f"{stage},{process.pid},{process.pid},{device_query},"
                    f"{owner_contract},{process_query}\n")
            process.stdin.write("G\n")
            process.stdin.flush()
            process.stdin.close()
            process.stdin = None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process, cleanup_records, "timeout")
            _release_process(process, cleanup_records)
            _append_process_event(
                events_path, "timeout",
                (("pid", process.pid), ("pgid", process.pid),
                 ("timeout_seconds", timeout_seconds)))
            raise RuntimeError(
                f"benchmark command timed out after {timeout_seconds}s: "
                f"{_format_command(command)}; partial stdout={stdout_path}; "
                f"partial stderr={stderr_path}") from exc
        except BaseException as exc:
            _terminate_process_group(
                process, cleanup_records, _exception_cleanup_reason(exc))
            _release_process(process, cleanup_records)
            _append_process_event(
                events_path, "interrupted",
                (("pid", process.pid), ("pgid", process.pid),
                 ("error_type", type(exc).__name__), ("error", exc)))
            raise
    finally:
        stdout_stream.flush()
        stderr_stream.flush()
        stdout_stream.close()
        stderr_stream.close()
    returncode = process.returncode
    try:
        if _process_group_exists(process.pid):
            _terminate_process_group(
                process, cleanup_records, "leader_exited_with_descendants")
            _release_process(process, cleanup_records)
            _append_process_event(
                events_path, "descendants_rejected",
                (("pid", process.pid), ("pgid", process.pid)))
            raise RuntimeError(
                f"benchmark leader exited but PGID {process.pid} retained "
                f"descendants: {_format_command(command)}")
        cleanup_records.append(
            f"pid={process.pid} pgid={process.pid} leader=exited group=absent "
            f"returncode={returncode} live_telemetry=overlapped")
        _release_process(process, cleanup_records)
    finally:
        _append_process_event(
            events_path, "exited",
            (("pid", process.pid), ("pgid", process.pid),
             ("returncode", returncode),
             ("live_telemetry", "overlapped")))
    stdout = stdout_path.read_text()
    complete_stderr = stderr_path.read_text()
    if not complete_stderr.startswith("telemetry_ready\n"):
        raise RuntimeError(
            f"benchmark PID {process.pid} lost its telemetry handshake "
            f"provenance in {stderr_path}")
    stderr = complete_stderr[len("telemetry_ready\n"):]
    if returncode != 0:
        raise RuntimeError(
            f"benchmark command failed with exit {returncode}: "
            f"{_format_command(command)}; stdout={stdout_path}; "
            f"stderr={stderr_path}\nstderr:\n{stderr}")
    return process.pid, stdout, stderr


def _record_tool_identity(tool, name, version_args, output_dir, environment,
                          cleanup_records, command_records, process_dir,
                          process_records):
    command = [str(tool)] + list(version_args)
    command_records.append(_format_command(command))
    _, stdout, stderr = _run_capture(
        command, cwd=REPO_ROOT, environment=environment,
        cleanup_records=cleanup_records,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
        process_records=process_records, stage=f"{name}_version")
    if not stdout and not stderr:
        raise RuntimeError(f"{name} version command emitted no identity")
    (output_dir / "logs" / f"{name}_version.stdout").write_text(stdout)
    (output_dir / "logs" / f"{name}_version.stderr").write_text(stderr)
    (output_dir / "logs" / f"{name}_identity.txt").write_text(
        f"path={tool}\nsha256={_sha256(tool)}\n")


def _paired_residency(pair, target_occupancy, control_occupancy):
    if pair.target.residency != pair.control.residency:
        raise ValueError("paired cells changed residency semantics")
    if pair.target.residency == "one":
        if target_occupancy < 1 or control_occupancy < 1:
            raise ValueError("paired specialization has no one-CTA residency")
        return 1
    if pair.target.residency == "max":
        return min(target_occupancy, control_occupancy)
    raise ValueError(f"unknown paired residency {pair.target.residency!r}")


def _capture_main(args):
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("--run-id must match [A-Za-z0-9._-]{1,96}")
    if DEVICE_UUID_PATTERN.fullmatch(args.device_uuid) is None:
        raise ValueError("--device-uuid must be a canonical NVIDIA GPU UUID")
    if PCI_BUS_ID_PATTERN.fullmatch(args.pci_bus_id) is None:
        raise ValueError("--pci-bus-id must be a canonical PCI bus identity")
    nvcc = _require_executable(args.nvcc, "--nvcc")
    cuobjdump = _require_executable(args.cuobjdump, "--cuobjdump")
    nvidia_smi = _require_executable(args.nvidia_smi, "--nvidia-smi")
    if not SOURCE_PATH.is_file():
        raise RuntimeError(f"missing canonical source {SOURCE_PATH}")
    source_environment = os.environ.copy()
    environment = None
    cleanup_records = []
    if ACTIVE_PROCESS_GROUPS:
        raise RuntimeError(
            "capture cannot start with registered external process groups")
    output_dir = _prepare_output_dir(args.output_dir, args.run_id)
    process_dir = output_dir / "logs" / "processes"
    process_records = []
    command_records = []
    status_path = output_dir / "logs" / "status.txt"
    capture_error = None
    post_error = None
    cleanup_errors = []
    held_fds = []
    source_fd = None
    source_identity = None
    runner_fd = None
    runner_identity = None
    binary_fd = None
    binary_identity = None
    sass_fd = None
    sass_identity = None
    baseline_verified = False
    pairs = SMOKE_PAIRS if args.smoke else FIXED_PAIRS
    live_path = output_dir / "telemetry" / "live.csv"
    try:
        environment = _capture_environment(source_environment, args.device_uuid)
        (output_dir / "logs" / "environment_gate.txt").write_text(
            "status=pass\n"
            "input_cuda_nvcc_ld_variables=none\n"
            f"forced_cuda_visible_devices={args.device_uuid}\n"
            "runtime_code_policy=embedded_sm_80_cubin_only_no_ptx_jit\n")
        baseline = _git_identity(
            environment, cleanup_records, process_dir, process_records)
        baseline_verified = True
        source_snapshot = output_dir / "source_snapshot" / SOURCE_PATH.name
        runner_snapshot = output_dir / "source_snapshot" / RUNNER_PATH.name
        source_relative = SOURCE_PATH.relative_to(REPO_ROOT)
        source_blob = _git_blob(
            baseline, source_relative, environment, cleanup_records,
            process_dir, process_records)
        source_fd, source_identity = _create_held_file(
            source_snapshot, source_blob.content, "source snapshot")
        held_fds.append(source_fd)
        if source_identity.sha256 != source_blob.sha256:
            raise RuntimeError(
                "source snapshot differs from baseline Git blob bytes")
        shutil.copy2(RUNNER_PATH, runner_snapshot)
        runner_fd, runner_identity = _open_held_file(
            runner_snapshot, "runner snapshot")
        held_fds.append(runner_fd)
        mode = "smoke" if args.smoke else "full"
        top_level_argv = _format_command(sys.argv)
        provenance = (
            f"baseline={baseline}\n"
            f"source={source_relative}\n"
            f"source_git_blob_oid={source_blob.object_id}\n"
            f"source_git_blob_sha256={source_blob.sha256}\n"
            f"source_sha256={source_identity.sha256}\n"
            f"runner={RUNNER_PATH.relative_to(REPO_ROOT)}\n"
            f"runner_sha256={runner_identity.sha256}\n"
            f"top_level_argv={top_level_argv}\n"
            f"device_name={EXPECTED_DEVICE_NAME}\n"
            f"device_uuid={args.device_uuid}\n"
            f"pci_bus_id={args.pci_bus_id}\n"
            f"capture_mode={mode}\n"
            f"matrix_pairs={len(pairs)}\n"
            f"matrix_cells={sum(len(pair.execution_order) for pair in pairs)}\n"
            f"trials_per_cell={TRIALS}\n"
            "latency_timing=single_cta_single_warp_clock64_"
            "min_start_max_end_cycles\n"
            "throughput_timing=cooperative_grid_per_sm_clock64_"
            "local_min_start_max_end_cycles\n"
            "throughput_aggregation=sum_issued_warp_opcodes_over_"
            "four_subcores_times_sum_local_sm_cycles\n"
            "paired_grid=min_target_control_occupancy\n"
            "control_policy=independent_observation_no_subtraction\n"
            "latency_ex2_approx_f32=not_identifiable_"
            "pure_finite_recurrence_unavailable\n"
            "latency_rcp_approx_f32=not_identifiable_"
            "cuda_ptxas_reciprocal_pair_collapse\n"
            "throughput_rcp_approx_f32=not_identifiable_"
            "cuda_ptxas_reciprocal_chain_collapse\n"
            "throughput_ex2_approx_f32=not_identifiable_"
            "finite_pure_recurrence_operand_class_unavailable\n"
            "throughput_register_reuse=only_after_unique_successor_"
            "consumed_previous_output\n"
            "latency_mul_f32=replacement_mechanism_observation\n"
            "latency_endpoint=final_output_nonzero_xor_predicate_"
            "complementary_exit_then_clock64\n"
            "latency_denominator=target_opcodes_only_endpoint_excluded\n"
        )
        (output_dir / "logs" / "provenance.txt").write_text(provenance)
        (output_dir / "top_level_argv.txt").write_text(top_level_argv + "\n")
        (output_dir / "results" / "not_identifiable.csv").write_text(
            "mode,primitive,status,reason\n"
            "latency,ex2_approx_f32,not_identifiable,"
            "pure_finite_dependent_recurrence_unavailable\n"
            "latency,rcp_approx_f32,not_identifiable,"
            "cuda_ptxas_reciprocal_pair_collapse\n"
            "throughput,rcp_approx_f32,not_identifiable,"
            "cuda_ptxas_reciprocal_chain_collapse\n"
            "throughput,ex2_approx_f32,not_identifiable,"
            "finite_pure_recurrence_operand_class_unavailable\n")
        (output_dir / "logs" / "source_identity.txt").write_text(
            f"source_baseline={baseline}\n"
            f"source_git_path={source_relative}\n"
            f"source_git_blob_oid={source_blob.object_id}\n"
            f"source_git_blob_sha256={source_blob.sha256}\n"
            + _format_file_identity("source_pre_compile", source_identity)
            + _format_file_identity("runner_snapshot", runner_identity))

        _record_tool_identity(
            nvcc, "nvcc", ("--version",), output_dir, environment,
            cleanup_records, command_records, process_dir, process_records)
        _record_tool_identity(
            cuobjdump, "cuobjdump", ("--version",), output_dir, environment,
            cleanup_records, command_records, process_dir, process_records)
        _record_tool_identity(
            nvidia_smi, "nvidia_smi", NVIDIA_SMI_IDENTITY_ARGS,
            output_dir, environment,
            cleanup_records, command_records, process_dir, process_records)

        _assert_held_file(
            source_snapshot, source_fd, source_identity,
            "source snapshot immediately before compile")
        binary = output_dir / "build" / "primitive_throughput"
        compile_command = [
            str(nvcc), "-O3", "-std=c++17", "-x", "cu",
            "--generate-code=arch=compute_80,code=sm_80", "-lineinfo",
            "--ptxas-options=-v", "-o", str(binary),
            _proc_fd_path(source_fd),
        ]
        command_records.append(_format_command(compile_command))
        _, compile_stdout, compile_stderr = _run_capture(
            compile_command, cwd=REPO_ROOT, environment=environment,
            cleanup_records=cleanup_records,
            timeout_seconds=COMPILE_TIMEOUT_SECONDS, process_dir=process_dir,
            process_records=process_records, stage="compile_sm80_cubin",
            pass_fds=(source_fd,))
        (output_dir / "logs" / "compile.stdout").write_text(compile_stdout)
        (output_dir / "logs" / "compile.stderr").write_text(compile_stderr)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError("nvcc did not produce an executable binary")
        _assert_held_file(
            source_snapshot, source_fd, source_identity,
            "source snapshot after compile")
        _assert_held_file(
            runner_snapshot, runner_fd, runner_identity,
            "runner snapshot after compile")
        with (output_dir / "logs" / "source_identity.txt").open("a") as stream:
            stream.write(
                _format_file_identity(
                    "source_post_compile", _fd_identity(source_fd)))
        binary_fd, binary_identity = _open_held_file(
            binary, "benchmark binary", require_executable=True)
        held_fds.append(binary_fd)
        binary_command_path = _proc_fd_path(binary_fd)
        validated_ptxas_count = _validate_ptxas(compile_stderr)
        (output_dir / "logs" / "ptxas_gate.txt").write_text(
            f"status=pass\nvalidated_kernels={validated_ptxas_count}\n")

        list_elf_command = [
            str(cuobjdump), "--list-elf", binary_command_path]
        command_records.append(_format_command(list_elf_command))
        _, list_elf_stdout, list_elf_stderr = _run_capture(
            list_elf_command, cwd=REPO_ROOT, environment=environment,
            cleanup_records=cleanup_records,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
            process_records=process_records, stage="cubin_list_elf",
            pass_fds=(binary_fd,))
        if list_elf_stderr:
            raise RuntimeError("cuobjdump --list-elf emitted stderr")
        dump_ptx_command = [
            str(cuobjdump), "--dump-ptx", binary_command_path]
        command_records.append(_format_command(dump_ptx_command))
        _, dump_ptx_stdout, dump_ptx_stderr = _run_capture(
            dump_ptx_command, cwd=REPO_ROOT, environment=environment,
            cleanup_records=cleanup_records,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
            process_records=process_records, stage="cubin_dump_ptx",
            pass_fds=(binary_fd,))
        if dump_ptx_stderr:
            raise RuntimeError("cuobjdump --dump-ptx emitted stderr")
        validated_cubin_count = _validate_cubin_package(
            list_elf_stdout, dump_ptx_stdout)
        (output_dir / "logs" / "cubin_gate.txt").write_text(
            "status=pass\n"
            f"validated_sm_80_cubins={validated_cubin_count}\n"
            "embedded_ptx=absent\n")
        (output_dir / "logs" / "cubin_list_elf.txt").write_text(
            list_elf_stdout)
        (output_dir / "logs" / "cubin_dump_ptx.txt").write_text(
            dump_ptx_stdout)

        sass_path = output_dir / "build" / "primitive_throughput.sass"
        sass_command = [str(cuobjdump), "--dump-sass", binary_command_path]
        command_records.append(_format_command(sass_command))
        _, sass_bytes, sass_stderr = _run_capture(
            sass_command, cwd=REPO_ROOT, environment=environment,
            cleanup_records=cleanup_records,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
            process_records=process_records, stage="cubin_dump_sass",
            pass_fds=(binary_fd,), stdout_bytes=True)
        (output_dir / "logs" / "cuobjdump.stderr").write_text(sass_stderr)
        try:
            sass_stdout = sass_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("cuobjdump SASS is not valid UTF-8") from exc
        validated_sass_count = _validate_sass(sass_stdout)
        sass_fd, sass_identity = _create_held_file(
            sass_path, sass_bytes, "validated SASS")
        held_fds.append(sass_fd)
        sass_stdout_sha256 = hashlib.sha256(sass_bytes).hexdigest()
        if sass_identity.sha256 != sass_stdout_sha256:
            raise RuntimeError(
                "held validated SASS differs from validated stdout bytes")
        (output_dir / "logs" / "sass_gate.txt").write_text(
            f"status=pass\nvalidated_kernels={validated_sass_count}\n"
            f"validated_stdout_sha256={sass_stdout_sha256}\n"
            "dependency_contract=latency_single_raw_chain_"
            "throughput_128_round_independent_recurrence_chains\n"
            "throughput_register_reuse=dead_output_only_"
            "live_or_terminal_cross_chain_waw_rejected\n"
            "latency_endpoint=final_dest_to_nonzero_xor_to_predicate_"
            "to_complementary_exit_to_end_cs2r\n"
            "latency_timer=clock64_cycles\n"
            "throughput_timer=per_sm_clock64_cycles\n"
            "throughput_clock_rule=no_cross_sm_absolute_clock_comparison\n"
            "latency_denominator=target_opcodes_only_endpoint_excluded\n"
            "dependency_status=pass\n")
        (output_dir / "logs" / "binary_identity.txt").write_text(
            _format_file_identity("binary_pre_execution", binary_identity)
            + _format_file_identity("sass_pre_execution", sass_identity))

        _assert_path_identity(binary, binary_identity, "benchmark binary")
        _assert_path_identity(sass_path, sass_identity, "validated SASS")
        abi_command = [binary_command_path, "--describe-abi"]
        command_records.append(_format_command(abi_command))
        _, abi_stdout, abi_stderr = _run_capture(
            abi_command, cwd=REPO_ROOT, environment=environment,
            cleanup_records=cleanup_records,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS, process_dir=process_dir,
            process_records=process_records, stage="binary_abi",
            pass_fds=(binary_fd,))
        _validate_abi(abi_stdout, abi_stderr)
        (output_dir / "logs" / "abi_smoke.txt").write_text(abi_stdout)

        pre_query = _query_device(
            nvidia_smi, args.device_uuid, args.pci_bus_id, environment,
            cleanup_records, command_records, True, process_dir,
            process_records, "device_pre")
        (output_dir / "telemetry" / "pre.csv").write_text(pre_query + "\n")
        live_path.write_text(
            "stage,benchmark_pid,benchmark_pgid," + DEVICE_QUERY_FIELDS
            + "," + LIVE_PROCESS_FIELDS + "\n")

        raw_path = output_dir / "results" / "raw.csv"
        cell_index = 0
        latency_rows = []
        with raw_path.open("x") as raw_stream:
            raw_stream.write(RUNNER_RAW_HEADER + "\n")
            for pair_index, pair in enumerate(pairs, start=1):
                _assert_path_identity(
                    binary, binary_identity, "benchmark binary")
                _assert_path_identity(
                    sass_path, sass_identity, "validated SASS")
                target_occupancy_command = _cell_command(
                    binary_command_path, pair.target, args.device_uuid,
                    args.pci_bus_id,
                    "occupancy", 1, 1)
                command_records.append(_format_command(target_occupancy_command))
                _, target_stdout, target_stderr = _run_capture(
                    target_occupancy_command, cwd=REPO_ROOT,
                    environment=environment, cleanup_records=cleanup_records,
                    timeout_seconds=COMMAND_TIMEOUT_SECONDS,
                    process_dir=process_dir, process_records=process_records,
                    stage=f"pair_{pair_index}_target_occupancy",
                    pass_fds=(binary_fd,))
                target_occupancy = _parse_occupancy_output(
                    target_stdout, target_stderr, pair.target)

                control_occupancy_command = _cell_command(
                    binary_command_path, pair.control, args.device_uuid,
                    args.pci_bus_id,
                    "occupancy", 1, 1)
                command_records.append(_format_command(control_occupancy_command))
                _, control_stdout, control_stderr = _run_capture(
                    control_occupancy_command, cwd=REPO_ROOT,
                    environment=environment, cleanup_records=cleanup_records,
                    timeout_seconds=COMMAND_TIMEOUT_SECONDS,
                    process_dir=process_dir, process_records=process_records,
                    stage=f"pair_{pair_index}_control_occupancy",
                    pass_fds=(binary_fd,))
                control_occupancy = _parse_occupancy_output(
                    control_stdout, control_stderr, pair.control)
                paired_resident = _paired_residency(
                    pair, target_occupancy, control_occupancy)
                if pair.target.mode == "latency":
                    grid_blocks = 1
                elif pair.target.mode == "throughput":
                    grid_blocks = SM_COUNT * paired_resident
                else:
                    raise ValueError(f"unknown pair mode {pair.target.mode!r}")

                for cell in pair.execution_order:
                    _assert_path_identity(
                        binary, binary_identity, "benchmark binary")
                    _assert_path_identity(
                        sass_path, sass_identity, "validated SASS")
                    if cell.primitive == CONTROL_PRIMITIVE:
                        specialization_occupancy = control_occupancy
                    elif cell.primitive in OBSERVED_TARGET_PRIMITIVES:
                        specialization_occupancy = target_occupancy
                    else:
                        raise ValueError(
                            f"unknown pair primitive {cell.primitive!r}")
                    command = _cell_command(
                        binary_command_path, cell, args.device_uuid,
                        args.pci_bus_id,
                        "run", grid_blocks, paired_resident)
                    command_records.append(_format_command(command))
                    child_pid, stdout, stderr = _run_benchmark_with_live_telemetry(
                        command, cwd=REPO_ROOT, environment=environment,
                        cleanup_records=cleanup_records,
                        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
                        nvidia_smi=nvidia_smi,
                        expected_uuid=args.device_uuid,
                        expected_pci=args.pci_bus_id,
                        command_records=command_records,
                        live_path=live_path,
                        stage=f"pair_{pair_index}_cell_{cell.pair_order}",
                        process_dir=process_dir,
                        process_records=process_records,
                        pass_fds=(binary_fd,))
                    if stderr:
                        raise RuntimeError(
                            f"cell {cell.pair_id} emitted unexpected stderr: "
                            f"{stderr}")
                    validated_lines = _validate_raw_output(
                        stdout, cell, args.device_uuid, args.pci_bus_id,
                        paired_resident, specialization_occupancy)
                    cell_index += 1
                    for line in validated_lines:
                        row = dict(zip(RAW_FIELDS, line.split(",")))
                        if row["mode"] == "latency":
                            latency_rows.append(row)
                        raw_stream.write(
                            f"{args.run_id},{cell_index},{cell.pair_id},"
                            f"{cell.pair_order},{child_pid},{target_occupancy},"
                            f"{control_occupancy},{line}\n")
                    raw_stream.flush()
        if args.smoke:
            (output_dir / "results" / "latency_derivation_not_run.txt").write_text(
                "status=not_run\nreason=smoke_does_not_contain_full_fixed_"
                "multi_unroll_matrix\n")
        else:
            _write_latency_derivation(
                output_dir / "results" / "latency_cycles.csv",
                latency_rows)
        _assert_held_file(
            source_snapshot, source_fd, source_identity,
            "source snapshot final")
        _assert_held_file(
            runner_snapshot, runner_fd, runner_identity,
            "runner snapshot final")
        _assert_held_file(
            binary, binary_fd, binary_identity, "benchmark binary final")
        _assert_held_file(
            sass_path, sass_fd, sass_identity, "validated SASS final")
        with (output_dir / "logs" / "binary_identity.txt").open("a") as stream:
            stream.write(
                _format_file_identity(
                    "binary_post_execution", _fd_identity(binary_fd))
                + _format_file_identity(
                    "sass_post_execution", _fd_identity(sass_fd)))
    except BaseException as exc:
        capture_error = exc
        raise
    finally:
        try:
            _cleanup_active_process_groups(
                cleanup_records, "capture_finally")
        except BaseException as exc:
            cleanup_errors.append(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
        for held_fd in reversed(held_fds):
            try:
                os.close(held_fd)
                cleanup_records.append(f"held_fd={held_fd} status=closed")
            except OSError as exc:
                cleanup_errors.append(exc)
                cleanup_records.append(
                    f"held_fd={held_fd} status=close_failed error={exc}")
        capture_was_signalled = isinstance(
            capture_error, RunnerTerminationSignal)
        if capture_was_signalled:
            signame = signal.Signals(capture_error.signum).name
            (output_dir / "telemetry" / "post_skipped_due_signal.txt").write_text(
                f"signal={signame}\nreason=runner_termination_signal\n")
        elif not baseline_verified:
            (output_dir / "telemetry" / "post_skipped_due_baseline_failure.txt").write_text(
                "reason=deployed_baseline_not_verified\n")
        elif cleanup_errors:
            (output_dir / "telemetry" / "post_skipped_due_cleanup_failure.txt").write_text(
                "reason=active_process_group_cleanup_failed\n")
        else:
            try:
                post_query = _query_device_until_idle(
                    nvidia_smi, args.device_uuid, args.pci_bus_id, environment,
                    cleanup_records, command_records, process_dir,
                    process_records,
                    output_dir / "telemetry" / "post_attempts.csv")
                (output_dir / "telemetry" / "post.csv").write_text(
                    post_query + "\n")
            except BaseException as exc:
                post_error = exc
                (output_dir / "telemetry" / "post_failure.txt").write_text(
                    f"error_type={type(exc).__name__}\nerror={exc}\n")
                if isinstance(
                        exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
        try:
            _cleanup_active_process_groups(
                cleanup_records, "post_capture_finally")
        except BaseException as exc:
            cleanup_errors.append(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
        errors = []
        if capture_error is not None:
            errors.append(
                f"capture_error_type={type(capture_error).__name__}\n"
                f"capture_error={capture_error}")
            if capture_was_signalled:
                errors.append(
                    "capture_signal="
                    f"{signal.Signals(capture_error.signum).name}")
        if post_error is not None:
            errors.append(
                f"post_error_type={type(post_error).__name__}\n"
                f"post_error={post_error}")
        for index, cleanup_error in enumerate(cleanup_errors, start=1):
            errors.append(
                f"cleanup_error_{index}_type="
                f"{type(cleanup_error).__name__}\n"
                f"cleanup_error_{index}={cleanup_error}")
        if errors:
            status_path.write_text(
                "status=failure\n" + "\n".join(errors) + "\n")
        else:
            status_path.write_text("status=success\n")
        (output_dir / "commands.txt").write_text(
            "\n".join(command_records) + "\n")
        (output_dir / "logs" / "cleanup.txt").write_text(
            "\n".join(cleanup_records) + "\n")
        _write_manifest(output_dir)
        if capture_error is None:
            if cleanup_errors:
                raise cleanup_errors[0]
            if post_error is not None:
                raise post_error
    print(f"capture_complete={output_dir}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--pci-bus-id", required=True)
    parser.add_argument("--nvcc", required=True, type=Path)
    parser.add_argument("--cuobjdump", required=True, type=Path)
    parser.add_argument("--nvidia-smi", required=True, type=Path)
    parser.add_argument(
        "--smoke", action="store_true",
        help="capture two minimal target/control pairs instead of the full matrix")
    args = parser.parse_args()
    previous_signal_handlers = _install_signal_handlers()
    try:
        try:
            _capture_main(args)
        except (FileExistsError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    finally:
        _restore_signal_handlers(previous_signal_handlers)


if __name__ == "__main__":
    main()
