#!/usr/bin/env python3
"""Build and retain raw empty-task launch captures on Ascend 910B.

This runner deliberately emits no fitted curve or HWSpec-ready parameter.  It
keeps the complete msprof profiles and transcribes target rows from the
provider's ``task_time`` table without aggregating their durations.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import shlex
import shutil
import sqlite3
import stat
import subprocess
import sys
import tarfile
import time


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
SOC_VERSION = "Ascend910B2"
MIN_CANDIDATE_RUNS = 2
RAW_TASK_TIME_SCHEMA_VERSION = 3
CAPTURE_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
TRUSTED_LD_LIBRARY_ROOTS = (
    Path("/usr/local/Ascend/cann-8.5.1"),
    Path("/usr/local/Ascend/driver"),
)
TRUSTED_LD_LIBRARY_OWNER_UID = 0
SOURCE_FILES = (
    "CMakeLists.txt",
    "empty_aic_kernel.cpp",
    "empty_aic_main.cpp",
    "empty_aiv_kernel.cpp",
    "empty_aiv_main.cpp",
    "nonempty_aiv_kernel.cpp",
    "nonempty_aiv_main.cpp",
    "nonempty_mix_aiv_kernel.cpp",
    "nonempty_mix_aiv_main.cpp",
    "nonempty_aiv_report.py",
    "aiv_worker_start_kernel.cpp",
    "aiv_worker_start_main.cpp",
    "aiv_memory_task_floor_kernel.cpp",
    "aiv_memory_task_floor_main.cpp",
    "aiv_norm_control_task_floor_kernel.cpp",
    "aiv_norm_control_task_floor_main.cpp",
    "empty_mixed_kernel.cpp",
    "empty_mixed_main.cpp",
    "empty_mix_aic_kernel.cpp",
    "empty_mix_aic_main.cpp",
    "launch_capture_runner.py",
)
KERNEL_TARGETS = (
    "empty_aic_kernel", "empty_aiv_kernel", "nonempty_aiv_kernel",
    "nonempty_mix_aiv_kernel",
    "aiv_memory_task_floor_kernel", "aiv_norm_control_task_floor_kernel",
    "empty_mixed_kernel", "empty_mix_aic_kernel")
HEALTH_PATTERN = re.compile(
    r"health_check_only=true task_class="
    r"(aic_only|aiv_only|aiv_nonempty|mix_aiv_nonempty|aiv_memory_task|"
    r"aiv_norm_control_task|mix_aic_1_2|mix_aic_1_0) "
    r"block_dim=([0-9]+) mix_block_dim=([0-9]+) launches=([0-9]+) "
    r"elapsed_ms=([0-9]+(?:\.[0-9]+)?)"
)


@dataclass(frozen=True)
class CaptureCell:
    task_class: str
    binary_name: str
    kernel_name: str
    requested_block_dim: int
    requested_mix_block_dim: int
    expected_ge_task_type: str
    expected_task_time_task_type: str


class CaptureInterrupted(BaseException):
    def __init__(self, signum):
        super().__init__(f"capture interrupted by signal {signum}")
        self.signum = signum


class PostlaunchIntegrityError(RuntimeError):
    def __init__(self, integrity, cause):
        super().__init__(f"postlaunch integrity failed: {cause}")
        self.integrity = integrity


def _capture_signal_handler(signum, _frame):
    raise CaptureInterrupted(signum)


def _install_capture_signal_handlers():
    previous = {}
    for signum in CAPTURE_SIGNALS:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _capture_signal_handler)
    return previous


def _restore_capture_signal_handlers(previous):
    for signum in CAPTURE_SIGNALS:
        signal.signal(signum, previous[signum])


def _capture_failure_signal(error):
    if isinstance(error, CaptureInterrupted):
        return error.signum
    return None


def build_cells(campaign="empty"):
    """Return the complete canonical exact-point sweep in stable order."""
    if campaign == "aiv_norm_control_task":
        return tuple(
            CaptureCell(
                "aiv_norm_control_task", "aiv_norm_control_task_floor_bench",
                "aiv_norm_control_task_floor", block_dim, 0,
                "AI_VECTOR_CORE", "AI_CORE")
            for block_dim in range(1, 49)
        )
    if campaign == "aiv_memory_task":
        return tuple(
            CaptureCell(
                "aiv_memory_task", "aiv_memory_task_floor_bench",
                "aiv_memory_task_floor", block_dim, 0,
                "AI_VECTOR_CORE", "AI_CORE")
            for block_dim in range(1, 49)
        )
    if campaign == "nonempty_aiv":
        return tuple(
            CaptureCell(
                "aiv_nonempty", "nonempty_aiv_bench", "nonempty_aiv",
                block_dim, 0, "AI_VECTOR_CORE", "AI_CORE")
            for block_dim in range(1, 49)
        )
    if campaign == "nonempty_mix_aiv":
        return tuple(
            CaptureCell(
                "mix_aiv_nonempty", "nonempty_mix_aiv_bench",
                "nonempty_mix_aiv", block_dim, 0,
                "MIX_AIV", "AI_VECTOR_CORE")
            for block_dim in range(1, 49)
        )
    if campaign == "mix_aic_1_0":
        return tuple(
            CaptureCell(
                "mix_aic_1_0", "empty_mix_aic_bench", "empty_mix_aic",
                block_dim, 0, "MIX_AIC", "AI_CORE")
            for block_dim in range(1, 25)
        )
    if campaign != "empty":
        raise ValueError(f"unsupported launch campaign {campaign!r}")
    cells = []
    for block_dim in range(1, 25):
        cells.append(CaptureCell(
            "aic_only", "empty_aic_bench", "empty_aic",
            block_dim, 0, "AI_CORE", "AI_CORE"))
    for block_dim in range(1, 49):
        cells.append(CaptureCell(
            "aiv_only", "empty_aiv_bench", "empty_aiv",
            block_dim, 0, "AI_VECTOR_CORE", "AI_CORE"))
    for block_dim in range(1, 25):
        cells.append(CaptureCell(
            "mix_aic_1_2", "empty_mixed_bench", "empty_mixed",
            block_dim, 2 * block_dim, "MIX_AIC", "AI_CORE"))
    return tuple(cells)


def ordered_cells(run_index, campaign="empty"):
    """Return paired forward/reverse rotations for candidate capture."""
    cells = build_cells(campaign)
    if not isinstance(run_index, int) or isinstance(run_index, bool):
        raise TypeError("run_index must be an integer")
    if run_index <= 0:
        raise ValueError("run_index must be positive")
    pair_index = (run_index - 1) // 2
    offset = pair_index % len(cells)
    rotated = cells[offset:] + cells[:offset]
    if run_index % 2 == 1:
        return rotated
    return tuple(reversed(rotated))


def smoke_cells(
        aic_block_dim, aiv_block_dim, mixed_block_dim, campaign="empty"):
    """Return at most one explicitly requested point per task class."""
    if campaign in (
            "nonempty_aiv", "nonempty_mix_aiv", "aiv_memory_task",
            "aiv_norm_control_task"):
        if aic_block_dim is not None or mixed_block_dim is not None:
            raise ValueError(
                f"{campaign} smoke accepts only --smoke-aiv-block-dim")
        if aiv_block_dim is None:
            raise ValueError(
                f"{campaign} smoke requires --smoke-aiv-block-dim")
        if aiv_block_dim <= 0 or aiv_block_dim > 48:
            raise ValueError(
                f"smoke {campaign} block_dim must be in 1..48")
        return (build_cells(campaign)[aiv_block_dim - 1],)
    if campaign == "mix_aic_1_0":
        if aiv_block_dim is not None or mixed_block_dim is not None:
            raise ValueError(
                "mix_aic_1_0 smoke accepts only --smoke-aic-block-dim")
        if aic_block_dim is None:
            raise ValueError(
                "mix_aic_1_0 smoke requires --smoke-aic-block-dim")
        if aic_block_dim <= 0 or aic_block_dim > 24:
            raise ValueError("smoke mix_aic_1_0 block_dim must be in 1..24")
        return (build_cells(campaign)[aic_block_dim - 1],)
    if campaign != "empty":
        raise ValueError(f"unsupported launch campaign {campaign!r}")
    requested = (
        ("aic_only", aic_block_dim, 24),
        ("aiv_only", aiv_block_dim, 48),
        ("mix_aic_1_2", mixed_block_dim, 24),
    )
    cells = []
    for task_class, block_dim, maximum in requested:
        if block_dim is None:
            continue
        if block_dim <= 0 or block_dim > maximum:
            raise ValueError(
                f"smoke {task_class} block_dim must be in 1..{maximum}")
        matches = [
            cell for cell in build_cells(campaign)
            if cell.task_class == task_class
            and cell.requested_block_dim == block_dim
        ]
        if len(matches) != 1:
            raise AssertionError(f"missing canonical smoke cell for {task_class}")
        cells.append(matches[0])
    if not cells:
        raise ValueError("--smoke requires at least one scoped block-dim option")
    return tuple(cells)


def validate_capture_selection(
        smoke, runs, aic_block_dim, aiv_block_dim, mixed_block_dim,
        campaign="empty"):
    smoke_requested = any(value is not None for value in (
        aic_block_dim, aiv_block_dim, mixed_block_dim))
    if smoke:
        if runs != 1:
            raise ValueError("--smoke requires --runs 1")
        return smoke_cells(
            aic_block_dim, aiv_block_dim, mixed_block_dim, campaign)
    if smoke_requested:
        raise ValueError("scoped smoke block dimensions require --smoke")
    if runs < MIN_CANDIDATE_RUNS or runs % 2 != 0:
        raise ValueError(
            f"candidate capture requires an even --runs >= "
            f"{MIN_CANDIDATE_RUNS}")
    return ()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path, root):
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _positive_integer(text):
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_integer(text):
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _repository_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("cannot locate repository root containing pyproject.toml")


def _validate_output_path(path, repository_root):
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(
            "--output-dir must be inside the current repository root") from error
    if relative.parts[:len(OUTPUT_OWNER_PARTS)] != OUTPUT_OWNER_PARTS:
        raise ValueError(
            "--output-dir must be a run directory under "
            "profiler_output/microbench/ascend_910b")
    if len(relative.parts) <= len(OUTPUT_OWNER_PARTS):
        raise ValueError("--output-dir must include a run-specific directory")
    if resolved.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {resolved}")
    return resolved


def _resolve_executable(value, name):
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"required {name} executable not found: {value}")
    path = Path(resolved).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise PermissionError(f"{name} is not an executable file: {path}")
    return path


def _executable_identity(executable):
    executable_stat = executable.stat()
    return {
        "executable": str(executable),
        "executable_bytes": executable_stat.st_size,
        "executable_sha256": _sha256(executable),
        "executable_mode_octal": f"{stat.S_IMODE(executable_stat.st_mode):04o}",
        "executable_owner_uid": executable_stat.st_uid,
        "executable_owner_gid": executable_stat.st_gid,
        "executable_st_dev": executable_stat.st_dev,
        "executable_st_ino": executable_stat.st_ino,
    }


def _sha256_file_descriptor(descriptor):
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _verify_executable_stat(identity, observed, label):
    expected = {
        "mode": int(identity["executable_mode_octal"], 8),
        "owner_uid": identity["executable_owner_uid"],
        "owner_gid": identity["executable_owner_gid"],
        "st_dev": identity["executable_st_dev"],
        "st_ino": identity["executable_st_ino"],
        "bytes": identity["executable_bytes"],
    }
    actual = {
        "mode": stat.S_IMODE(observed.st_mode),
        "owner_uid": observed.st_uid,
        "owner_gid": observed.st_gid,
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "bytes": observed.st_size,
    }
    if not stat.S_ISREG(observed.st_mode):
        raise PermissionError(f"{label} is not a regular file")
    if actual != expected:
        raise RuntimeError(
            f"{label} file identity changed: {actual!r} != {expected!r}")


def _bind_exact_executable(path, label):
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError(f"{label} is not executable: {resolved}")
    identity = _executable_identity(resolved)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(f"O_NOFOLLOW is required to bind {label}")
    descriptor = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        _verify_executable_stat(identity, os.fstat(descriptor), f"{label} FD")
        actual_sha256 = _sha256_file_descriptor(descriptor)
        if actual_sha256 != identity["executable_sha256"]:
            raise RuntimeError(f"{label} FD SHA256 changed")
        bound_path = Path(
            "/proc", str(os.getpid()), "fd", str(descriptor))
        if not bound_path.exists():
            raise FileNotFoundError(
                f"{label} bound descriptor path is unavailable: {bound_path}")
        return descriptor, {
            **identity,
            "descriptor": descriptor,
            "bound_path": str(bound_path),
            "binding": "parent_held_o_nofollow_fd_via_procfs",
        }
    except BaseException:
        os.close(descriptor)
        raise


def _verify_bound_exact_executable(binding, label):
    descriptor = binding["descriptor"]
    _verify_executable_stat(binding, os.fstat(descriptor), f"{label} FD")
    actual_sha256 = _sha256_file_descriptor(descriptor)
    if actual_sha256 != binding["executable_sha256"]:
        raise RuntimeError(f"{label} FD SHA256 changed")
    return {
        "status": "verified",
        "sha256": actual_sha256,
        "mode_octal": binding["executable_mode_octal"],
        "owner_uid": binding["executable_owner_uid"],
        "owner_gid": binding["executable_owner_gid"],
        "st_dev": binding["executable_st_dev"],
        "st_ino": binding["executable_st_ino"],
        "bytes": binding["executable_bytes"],
    }


def _interpreter_chain_probe_content(shebang):
    return (
        f"{shebang}\n"
        "import sys\n"
        "if not (sys.flags.isolated == 1 "
        "and sys.flags.ignore_environment == 1 "
        "and sys.flags.no_site == 1):\n"
        "    raise RuntimeError(f'unisolated flags: {sys.flags!r}')\n"
        "import json\n"
        "print(json.dumps({\n"
        "    'isolated': sys.flags.isolated,\n"
        "    'ignore_environment': sys.flags.ignore_environment,\n"
        "    'no_site': sys.flags.no_site,\n"
        "    'executable': sys.executable,\n"
        "    'version': sys.version,\n"
        "}, sort_keys=True))\n"
    )


def _audit_ld_library_path(
        environment, trusted_roots=TRUSTED_LD_LIBRARY_ROOTS,
        trusted_owner_uid=TRUSTED_LD_LIBRARY_OWNER_UID):
    if "LD_LIBRARY_PATH" not in environment:
        return {
            "status": "absent",
            "value": None,
            "directories": [],
        }
    value = environment["LD_LIBRARY_PATH"]
    directories = []
    normalized_entries = []
    resolved_trusted_roots = tuple(
        root.resolve(strict=True) for root in trusted_roots)
    for entry in value.split(":"):
        if not entry:
            raise ValueError(
                "LD_LIBRARY_PATH must not contain an empty current-directory "
                "entry")
        requested = Path(entry)
        if not requested.is_absolute():
            raise ValueError(
                f"LD_LIBRARY_PATH entry must be absolute: {entry!r}")
        resolved = requested.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(
                f"LD_LIBRARY_PATH entry is not a directory: {resolved}")
        directory_stat = resolved.stat()
        matching_roots = []
        for trusted_root in resolved_trusted_roots:
            if resolved == trusted_root or trusted_root in resolved.parents:
                matching_roots.append(trusted_root)
        if len(matching_roots) != 1:
            raise PermissionError(
                f"LD_LIBRARY_PATH entry must be under exactly one trusted "
                f"provider root: {resolved}")
        trusted_root = matching_roots[0]
        if stat.S_IMODE(directory_stat.st_mode) & 0o022:
            raise PermissionError(
                f"LD_LIBRARY_PATH entry must not be group/world writable: "
                f"{resolved}")
        if directory_stat.st_uid != trusted_owner_uid:
            raise PermissionError(
                f"LD_LIBRARY_PATH entry must be owned by uid "
                f"{trusted_owner_uid}: {resolved}")
        ancestry = []
        ancestor = resolved
        while True:
            ancestor_stat = ancestor.stat()
            ancestor_mode = stat.S_IMODE(ancestor_stat.st_mode)
            if (ancestor_stat.st_uid != trusted_owner_uid
                    or ancestor_mode & 0o022):
                raise PermissionError(
                    f"LD_LIBRARY_PATH ancestry must be uid "
                    f"{trusted_owner_uid} owned and not group/world writable: "
                    f"{ancestor}")
            ancestry.append({
                "path": str(ancestor),
                "mode_octal": f"{ancestor_mode:04o}",
                "owner_uid": ancestor_stat.st_uid,
                "owner_gid": ancestor_stat.st_gid,
                "st_dev": ancestor_stat.st_dev,
                "st_ino": ancestor_stat.st_ino,
            })
            if ancestor == trusted_root:
                break
            ancestor = ancestor.parent
        normalized_entries.append(str(resolved))
        directories.append({
            "requested_path": str(requested),
            "resolved_path": str(resolved),
            "mode_octal": f"{stat.S_IMODE(directory_stat.st_mode):04o}",
            "owner_uid": directory_stat.st_uid,
            "owner_gid": directory_stat.st_gid,
            "st_dev": directory_stat.st_dev,
            "st_ino": directory_stat.st_ino,
            "trusted_root": str(trusted_root),
            "trusted_owner_uid": trusted_owner_uid,
            "read_only_ancestry": ancestry,
        })
    normalized_value = ":".join(normalized_entries)
    environment["LD_LIBRARY_PATH"] = normalized_value
    return {
        "status": "preserved_after_absolute_directory_audit",
        "original_value": value,
        "value": normalized_value,
        "directories": directories,
    }


def _audit_capture_environment(environment):
    observed_ld_variables = sorted(
        key for key in environment if key.startswith("LD_"))
    rejected_loader_variables = [
        key for key in observed_ld_variables if key != "LD_LIBRARY_PATH"]
    if "GLIBC_TUNABLES" in environment:
        rejected_loader_variables.append("GLIBC_TUNABLES")
    if rejected_loader_variables:
        raise ValueError(
            "capture environment defines forbidden dynamic-loader variables: "
            f"{sorted(rejected_loader_variables)!r}")
    return {
        "status": "audited_before_capture_side_effects",
        "policy": "reject_all_LD_star_except_audited_LD_LIBRARY_PATH",
        "observed_ld_variables": observed_ld_variables,
        "rejected_loader_variables": [],
        "glibc_tunables": None,
        "ld_library_path": _audit_ld_library_path(environment),
    }


def _prepare_interpreter_chain(
        point_dir, environment, env_path=None, python_path=None):
    loader_policy = _audit_capture_environment(environment)
    if "PYTHONPATH" in environment:
        pythonpath = environment["PYTHONPATH"]
    else:
        pythonpath = None
    if "PYTHONHOME" in environment:
        pythonhome = environment["PYTHONHOME"]
    else:
        pythonhome = None
    exact_env_path = Path("/usr/bin/env") if env_path is None else Path(env_path)
    exact_python_path = (
        Path(sys.executable).resolve()
        if python_path is None else Path(python_path))
    env_descriptor = None
    python_descriptor = None
    try:
        env_descriptor, env_binding = _bind_exact_executable(
            exact_env_path, "wrapper env interpreter")
        python_descriptor, python_binding = _bind_exact_executable(
            exact_python_path, "wrapper Python interpreter")
        env_version = _run_process_group_captured(
            [env_binding["bound_path"], "--version"], environment, 60)
        if env_version["process"]["exit_code"] != 0:
            raise RuntimeError("bound env --version failed")
        shebang = (
            f"#!{env_binding['bound_path']} -S "
            f"{python_binding['bound_path']} -I -S")
        probe_path = point_dir.resolve() / "interpreter_chain_probe.py"
        probe_content = _interpreter_chain_probe_content(shebang)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        probe_descriptor = os.open(probe_path, flags, 0o500)
        with os.fdopen(probe_descriptor, "w", encoding="utf-8") as stream:
            stream.write(probe_content)
        probe_path.chmod(0o500)
        probe_result = _run_process_group_captured(
            [str(probe_path)], environment, 60)
        if probe_result["process"]["exit_code"] != 0:
            raise RuntimeError(
                "isolated interpreter-chain probe failed with "
                f"{probe_result['process']['exit_code']}: "
                f"{probe_result['stderr']}")
        probe_flags = json.loads(probe_result["stdout"])
        expected_flags = {
            "isolated": 1,
            "ignore_environment": 1,
            "no_site": 1,
        }
        actual_flags = {
            "isolated": probe_flags["isolated"],
            "ignore_environment": probe_flags["ignore_environment"],
            "no_site": probe_flags["no_site"],
        }
        if actual_flags != expected_flags:
            raise RuntimeError(
                f"interpreter-chain probe flags {actual_flags!r} do not "
                f"match {expected_flags!r}")
        return {
            "shebang": shebang,
            "env": {
                **env_binding,
                "version_command": [env_binding["bound_path"], "--version"],
                "version_stdout": env_version["stdout"],
                "version_stderr": env_version["stderr"],
                "version_process": env_version["process"],
            },
            "python": {
                **python_binding,
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "sys_version": sys.version,
            },
            "python_flags": expected_flags,
            "environment_policy": {
                **loader_policy,
                "ignored_pythonpath": pythonpath,
                "ignored_pythonhome": pythonhome,
            },
            "probe": {
                "path": str(probe_path),
                "content": probe_content,
                "sha256": _sha256(probe_path),
                "mode_octal": "0500",
                "command": [str(probe_path)],
                "return_code": probe_result["process"]["exit_code"],
                "stdout": probe_result["stdout"],
                "stderr": probe_result["stderr"],
                "process": probe_result["process"],
                "observed": probe_flags,
            },
        }
    except BaseException:
        if python_descriptor is not None:
            os.close(python_descriptor)
        if env_descriptor is not None:
            os.close(env_descriptor)
        raise


def _close_interpreter_chain(interpreter_chain):
    os.close(interpreter_chain["python"]["descriptor"])
    os.close(interpreter_chain["env"]["descriptor"])


def _close_application_resources(wrapper_descriptor, interpreter_chain):
    if wrapper_descriptor is not None:
        os.close(wrapper_descriptor)
    _close_interpreter_chain(interpreter_chain)


def _application_wrapper_content(
        benchmark, health_path, stderr_path, interpreter_chain):
    benchmark_contract = {
        "path": benchmark["executable"],
        "argv": benchmark["argv"],
        "bytes": benchmark["executable_bytes"],
        "sha256": benchmark["executable_sha256"],
        "mode": int(benchmark["executable_mode_octal"], 8),
        "owner_uid": benchmark["executable_owner_uid"],
        "owner_gid": benchmark["executable_owner_gid"],
        "st_dev": benchmark["executable_st_dev"],
        "st_ino": benchmark["executable_st_ino"],
    }
    return (
        f"{interpreter_chain['shebang']}\n"
        "import sys\n"
        "if not (sys.flags.isolated == 1 "
        "and sys.flags.ignore_environment == 1 "
        "and sys.flags.no_site == 1):\n"
        "    raise RuntimeError(f'unisolated flags: {sys.flags!r}')\n"
        "import hashlib\n"
        "import os\n"
        "import stat\n"
        f"BENCHMARK = {benchmark_contract!r}\n"
        f"STDOUT_PATH = {str(health_path)!r}\n"
        f"STDERR_PATH = {str(stderr_path)!r}\n"
        "if not hasattr(os, 'O_NOFOLLOW'):\n"
        "    raise RuntimeError('O_NOFOLLOW is required')\n"
        "output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW\n"
        "stdout_fd = os.open(STDOUT_PATH, output_flags, 0o600)\n"
        "stderr_fd = os.open(STDERR_PATH, output_flags, 0o600)\n"
        "os.dup2(stdout_fd, 1)\n"
        "os.dup2(stderr_fd, 2)\n"
        "os.close(stdout_fd)\n"
        "os.close(stderr_fd)\n"
        "if len(sys.argv) != 1:\n"
        "    raise RuntimeError('application wrapper rejects arguments')\n"
        "benchmark_fd = os.open(\n"
        "    BENCHMARK['path'], os.O_RDONLY | os.O_NOFOLLOW)\n"
        "observed = os.fstat(benchmark_fd)\n"
        "actual = {\n"
        "    'mode': stat.S_IMODE(observed.st_mode),\n"
        "    'owner_uid': observed.st_uid,\n"
        "    'owner_gid': observed.st_gid,\n"
        "    'st_dev': observed.st_dev,\n"
        "    'st_ino': observed.st_ino,\n"
        "    'bytes': observed.st_size,\n"
        "}\n"
        "expected = {key: BENCHMARK[key] for key in actual}\n"
        "if not stat.S_ISREG(observed.st_mode) or actual != expected:\n"
        "    raise RuntimeError(\n"
        "        f'benchmark FD identity changed: {actual!r} != {expected!r}')\n"
        "digest = hashlib.sha256()\n"
        "while True:\n"
        "    chunk = os.read(benchmark_fd, 1024 * 1024)\n"
        "    if not chunk:\n"
        "        break\n"
        "    digest.update(chunk)\n"
        "if digest.hexdigest() != BENCHMARK['sha256']:\n"
        "    raise RuntimeError('benchmark FD SHA256 changed')\n"
        "os.lseek(benchmark_fd, 0, os.SEEK_SET)\n"
        "os.set_inheritable(benchmark_fd, True)\n"
        "os.execve(\n"
        "    f'/proc/self/fd/{benchmark_fd}', BENCHMARK['argv'], os.environ)\n"
    )


def _create_application_wrapper(
        point_dir, binary, block_dim, launches, interpreter_chain):
    resolved_point_dir = point_dir.resolve()
    resolved_binary = binary.resolve()
    if not resolved_point_dir.is_dir():
        raise FileNotFoundError(
            f"point raw directory does not exist: {resolved_point_dir}")
    point_stat = resolved_point_dir.lstat()
    point_mode = stat.S_IMODE(point_stat.st_mode)
    if point_mode != 0o700:
        raise PermissionError(
            f"point raw directory mode must be 0700, got {point_mode:#05o}")
    if not resolved_binary.is_file() or not os.access(resolved_binary, os.X_OK):
        raise PermissionError(
            f"benchmark is not an executable file: {resolved_binary}")
    wrapper_path = resolved_point_dir / "msprof_application_wrapper.py"
    health_path = resolved_point_dir / "cann_event_health.txt"
    stderr_path = resolved_point_dir / "application_stderr.txt"
    benchmark_argv = [
        str(resolved_binary), str(block_dim), str(launches)]
    benchmark = {
        **_executable_identity(resolved_binary),
        "argv": benchmark_argv,
    }
    content = _application_wrapper_content(
        benchmark, health_path, stderr_path, interpreter_chain)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(wrapper_path, flags, 0o500)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
    wrapper_path.chmod(0o500)
    wrapper_stat = wrapper_path.lstat()
    return {
        "path": str(wrapper_path),
        "content": content,
        "bytes": wrapper_stat.st_size,
        "sha256": _sha256(wrapper_path),
        "mode_octal": "0500",
        "owner_uid": wrapper_stat.st_uid,
        "owner_gid": wrapper_stat.st_gid,
        "st_dev": wrapper_stat.st_dev,
        "st_ino": wrapper_stat.st_ino,
        "interpreter_chain": interpreter_chain,
        "point_directory": {
            "path": str(resolved_point_dir),
            "mode_octal": "0700",
            "owner_uid": point_stat.st_uid,
        },
        "application_argv": [str(wrapper_path)],
        "benchmark": benchmark,
        "stdout_path": str(health_path),
        "stderr_path": str(stderr_path),
        "prelaunch_integrity": {
            "status": "pending",
            "checked_at_utc": None,
            "failure": None,
        },
        "postlaunch_integrity": None,
        "threat_boundary": (
            "point directory mode 0700 excludes different-uid path writers; "
            "the runner executes the verified wrapper through a parent-held "
            "O_NOFOLLOW file descriptor; env and Python are separately bound "
            "and executed through parent-held descriptors with Python -I -S; "
            "the wrapper verifies and executes the benchmark through one "
            "O_NOFOLLOW file descriptor; a process "
            "with this runner uid can still mutate retained pathname artifacts, "
            "which postlaunch integrity and the top-level inventory detect"),
    }


def _verify_application_wrapper(wrapper):
    wrapper_path = Path(wrapper["path"])
    if not wrapper_path.is_absolute():
        raise ValueError("msprof application wrapper path must be absolute")
    point_directory = wrapper["point_directory"]
    point_path = Path(point_directory["path"])
    if not point_path.is_absolute() or wrapper_path.parent != point_path:
        raise ValueError(
            "msprof application wrapper must be directly inside its point "
            "raw directory")
    point_stat = point_path.lstat()
    if not stat.S_ISDIR(point_stat.st_mode):
        raise PermissionError(
            f"point raw directory is not a directory: {point_path}")
    point_mode = stat.S_IMODE(point_stat.st_mode)
    if point_mode != int(point_directory["mode_octal"], 8):
        raise PermissionError("point raw directory mode changed")
    if point_stat.st_uid != point_directory["owner_uid"]:
        raise PermissionError("point raw directory owner changed")
    for output_key in ("stdout_path", "stderr_path"):
        output_path = Path(wrapper[output_key])
        if not output_path.is_absolute() or output_path.parent != point_path:
            raise ValueError(
                f"application {output_key} must be directly inside the "
                "point raw directory")
    wrapper_stat = wrapper_path.lstat()
    if not stat.S_ISREG(wrapper_stat.st_mode):
        raise PermissionError(
            f"msprof application wrapper is not a regular file: {wrapper_path}")
    if wrapper_path.resolve(strict=True) != wrapper_path:
        raise PermissionError(
            f"msprof application wrapper path contains a symlink: {wrapper_path}")
    expected_mode = int(wrapper["mode_octal"], 8)
    actual_mode = stat.S_IMODE(wrapper_stat.st_mode)
    if actual_mode != expected_mode:
        raise PermissionError(
            f"msprof application wrapper mode changed: "
            f"{actual_mode:#05o} != {expected_mode:#05o}")
    if wrapper_stat.st_uid != wrapper["owner_uid"]:
        raise PermissionError("msprof application wrapper owner changed")
    if wrapper_stat.st_gid != wrapper["owner_gid"]:
        raise PermissionError("msprof application wrapper group changed")
    if wrapper_stat.st_dev != wrapper["st_dev"]:
        raise RuntimeError("msprof application wrapper device changed")
    if wrapper_stat.st_ino != wrapper["st_ino"]:
        raise RuntimeError("msprof application wrapper inode changed")
    if wrapper_stat.st_size != wrapper["bytes"]:
        raise RuntimeError("msprof application wrapper size changed")
    actual_sha256 = _sha256(wrapper_path)
    if actual_sha256 != wrapper["sha256"]:
        raise RuntimeError("msprof application wrapper SHA256 changed")
    if wrapper_path.read_text(encoding="utf-8") != wrapper["content"]:
        raise RuntimeError("msprof application wrapper content changed")
    if not os.access(wrapper_path, os.X_OK):
        raise PermissionError(
            f"msprof application wrapper is not executable: {wrapper_path}")
    benchmark = wrapper["benchmark"]
    benchmark_path = Path(benchmark["executable"])
    if benchmark_path.resolve(strict=True) != benchmark_path:
        raise PermissionError(
            f"benchmark path contains a symlink: {benchmark_path}")
    if not benchmark_path.is_file() or not os.access(benchmark_path, os.X_OK):
        raise PermissionError(
            f"benchmark is not an executable file: {benchmark_path}")
    _verify_executable_stat(
        benchmark, benchmark_path.stat(), "benchmark before msprof launch")
    if _sha256(benchmark_path) != benchmark["executable_sha256"]:
        raise RuntimeError("benchmark SHA256 changed before msprof launch")
    interpreter_chain = wrapper["interpreter_chain"]
    env_integrity = _verify_bound_exact_executable(
        interpreter_chain["env"], "wrapper env interpreter")
    python_integrity = _verify_bound_exact_executable(
        interpreter_chain["python"], "wrapper Python interpreter")
    return {
        "status": "verified",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure": None,
        "actual_sha256": actual_sha256,
        "actual_mode_octal": f"{actual_mode:04o}",
        "env_fd": env_integrity,
        "python_fd": python_integrity,
    }


def _bound_application_wrapper_integrity(wrapper, descriptor, status):
    observed = os.fstat(descriptor)
    wrapper_identity = {
        "executable_mode_octal": wrapper["mode_octal"],
        "executable_owner_uid": wrapper["owner_uid"],
        "executable_owner_gid": wrapper["owner_gid"],
        "executable_st_dev": wrapper["st_dev"],
        "executable_st_ino": wrapper["st_ino"],
        "executable_bytes": wrapper["bytes"],
    }
    _verify_executable_stat(
        wrapper_identity, observed, "msprof application wrapper FD")
    actual_sha256 = _sha256_file_descriptor(descriptor)
    if actual_sha256 != wrapper["sha256"]:
        raise RuntimeError("msprof application wrapper FD SHA256 changed")
    bound_path = Path("/proc") / str(os.getpid()) / "fd" / str(descriptor)
    if not bound_path.exists():
        raise FileNotFoundError(
            f"bound wrapper descriptor path is unavailable: {bound_path}")
    interpreter_chain = wrapper["interpreter_chain"]
    return {
        "status": status,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure": None,
        "actual_sha256": actual_sha256,
        "actual_mode_octal": f"{stat.S_IMODE(observed.st_mode):04o}",
        "actual_owner_uid": observed.st_uid,
        "actual_owner_gid": observed.st_gid,
        "actual_st_dev": observed.st_dev,
        "actual_st_ino": observed.st_ino,
        "holder_pid": os.getpid(),
        "descriptor": descriptor,
        "bound_application_path": str(bound_path),
        "mechanism": "parent_held_o_nofollow_fd_via_procfs",
        "interpreter_chain": {
            "env_fd": _verify_bound_exact_executable(
                interpreter_chain["env"], "wrapper env interpreter"),
            "python_fd": _verify_bound_exact_executable(
                interpreter_chain["python"], "wrapper Python interpreter"),
        },
    }


def _bind_application_wrapper(wrapper):
    _verify_application_wrapper(wrapper)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required to bind the wrapper")
    wrapper_path = Path(wrapper["path"])
    descriptor = os.open(wrapper_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        integrity = _bound_application_wrapper_integrity(
            wrapper, descriptor, "verified_and_bound")
        return descriptor, integrity
    except BaseException:
        os.close(descriptor)
        raise


def _postlaunch_application_integrity(wrapper, descriptor):
    result = {
        "status": "checking",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "bound_fd": {
            "status": "pending",
            "evidence": None,
            "failure": None,
        },
        "retained_path": {
            "status": "pending",
            "evidence": None,
            "failure": None,
        },
        "failure": None,
    }
    try:
        bound_evidence = _bound_application_wrapper_integrity(
            wrapper, descriptor, "verified_after_msprof_exit")
        result["bound_fd"] = {
            "status": "verified",
            "evidence": bound_evidence,
            "failure": None,
        }
    except BaseException as error:
        result["bound_fd"] = {
            "status": "failed",
            "evidence": None,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        result["status"] = "failed"
        result["failure"] = result["bound_fd"]["failure"]
        raise PostlaunchIntegrityError(result, error) from error
    try:
        retained_evidence = _verify_application_wrapper(wrapper)
        result["retained_path"] = {
            "status": "verified",
            "evidence": retained_evidence,
            "failure": None,
        }
    except BaseException as error:
        result["retained_path"] = {
            "status": "failed",
            "evidence": None,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        result["status"] = "failed"
        result["failure"] = result["retained_path"]["failure"]
        raise PostlaunchIntegrityError(result, error) from error
    result["status"] = "verified"
    return result


def _tool_identity(
        executable, arguments, allowed_return_codes, environment):
    command = [str(executable), *arguments]
    result = _run_process_group_captured(command, environment, 60)
    return_code = result["process"]["exit_code"]
    if return_code not in allowed_return_codes:
        raise RuntimeError(
            f"tool identity command returned {return_code}, expected "
            f"one of {allowed_return_codes}: {command}")
    return {
        **_executable_identity(executable),
        "command": command,
        "allowed_return_codes": list(allowed_return_codes),
        "return_code": return_code,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "process": result["process"],
    }


def _cann_package_identity(cann_path):
    root = Path(cann_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CANN package path is not a directory: {root}")
    candidates = (
        root / "version.cfg",
        root / "version.info",
        root / "ascend_toolkit_install.info",
        root / "compiler" / "version.info",
    )
    version_files = []
    for path in candidates:
        if path.is_file():
            version_files.append({
                "path": str(path),
                "sha256": _sha256(path),
                "content": path.read_text(encoding="utf-8"),
            })
    if not version_files:
        raise FileNotFoundError(
            f"CANN package exposes none of the required version files: {root}")
    return {
        "package_path": str(root),
        "package_directory_name": root.name,
        "version_files": version_files,
    }


UTILIZATION_PATTERNS = {
    "aic_percent": (
        r"(?:AI\s*Core|AICore)(?:\s*(?:Usage|Utilization|Rate))*\s*\(%\)",
    ),
    "aiv_percent": (
        r"(?:AI\s*Vector|AIVector|Vector\s*Core)"
        r"(?:\s*(?:Usage|Utilization|Rate))*\s*\(%\)",
    ),
    "hbm_bandwidth_percent": (
        r"(?:HBM\s*(?:Bandwidth|BW))"
        r"(?:\s*(?:Usage|Utilization|Rate))*\s*\(%\)",
    ),
}


def parse_idle_telemetry(text):
    """Require scoped AIC, AIV, and HBM-bandwidth utilization to be zero."""
    if not isinstance(text, str) or not text.strip():
        raise TypeError("npu-smi usage telemetry must be non-empty text")
    parsed = {}
    for field, patterns in UTILIZATION_PATTERNS.items():
        values = []
        for pattern in patterns:
            matches = re.findall(
                pattern + r"[^0-9+-]*([0-9]+(?:\.[0-9]+)?)",
                text,
                flags=re.IGNORECASE,
            )
            values.extend(Decimal(value) for value in matches)
        distinct = set(values)
        if len(distinct) != 1:
            raise ValueError(
                f"npu-smi telemetry must contain one unambiguous {field}, "
                f"found {sorted(str(value) for value in distinct)}")
        value = distinct.pop()
        if value != 0:
            raise RuntimeError(f"selected NPU is busy: {field}={value}")
        parsed[field] = float(value)
    return parsed


def parse_device_identity(global_text, board_text, device_id):
    """Bind a global 910B2 table row to the selected board ID and PCI bus."""
    if not isinstance(device_id, int) or isinstance(device_id, bool):
        raise TypeError("device_id must be an integer")
    if device_id < 0:
        raise ValueError("device_id must be non-negative")
    if not isinstance(global_text, str) or not global_text.strip():
        raise TypeError("global npu-smi telemetry must be non-empty text")
    if not isinstance(board_text, str) or not board_text.strip():
        raise TypeError("board npu-smi telemetry must be non-empty text")
    global_rows = re.findall(
        r"^\s*\|\s*([0-9]+)\s+((?:Ascend\s*)?910B2)\s*\|.*$",
        global_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    selected_rows = [
        (int(row_device_id), model)
        for row_device_id, model in global_rows
        if int(row_device_id) == device_id
    ]
    if len(selected_rows) != 1:
        raise ValueError(
            f"global npu-smi table must contain exactly one selected "
            f"device {device_id} with model 910B2, found {selected_rows!r}")
    board_ids = re.findall(
        r"^\s*NPU\s+ID\s*:\s*([0-9]+)\s*$",
        board_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if board_ids != [str(device_id)]:
        raise ValueError(
            f"board telemetry NPU ID does not uniquely match {device_id}: "
            f"{board_ids!r}")
    pci_matches = re.findall(
        r"^\s*(?:PCIe?\s+Bus\s+Info|PCIe?\s+Bus-Id|Bus-Id)\s*:\s*"
        r"([0-9a-fA-F:.]+)\s*$",
        board_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if len(pci_matches) != 1:
        raise ValueError(
            f"board telemetry must contain one PCI identity, found "
            f"{pci_matches!r}")
    header_lines = [
        line.strip()
        for line in global_text.splitlines()
        if re.search(r"npu-smi|Version", line, flags=re.IGNORECASE)
    ]
    if not header_lines:
        raise ValueError("global npu-smi table is missing its version header")
    return {
        "soc_version": SOC_VERSION,
        "global_device_id": device_id,
        "global_model": selected_rows[0][1],
        "board_npu_id": int(board_ids[0]),
        "board_pci_identity": pci_matches[0],
        "global_header_lines": header_lines,
    }


def _create_source_archive(source_dir, output_dir):
    snapshot_dir = output_dir / "source" / "tree"
    snapshot_dir.mkdir()
    archive_path = output_dir / "source" / "launch_source.tar"
    identities = []
    for relative in SOURCE_FILES:
        source_path = source_dir / relative
        if not source_path.is_file():
            raise FileNotFoundError(
                f"required launch source is missing: {source_path}")
        snapshot_path = snapshot_dir / relative
        shutil.copy2(source_path, snapshot_path)
        identities.append(_file_identity(snapshot_path, snapshot_dir))
    with tarfile.open(archive_path, "w") as archive:
        for relative in SOURCE_FILES:
            snapshot_path = snapshot_dir / relative
            info = archive.gettarinfo(str(snapshot_path), arcname=relative)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with snapshot_path.open("rb") as stream:
                archive.addfile(info, stream)
    return snapshot_dir, archive_path, identities


def verify_deployed_source_archive(archive_path, source_dir, baseline_commit):
    """Bind every active launch source byte to the deployed source archive."""
    source_prefix = "microbench/ascend_910b/runtime/launch"
    verified = []
    try:
        archive = tarfile.open(archive_path, "r:*")
    except tarfile.TarError as error:
        raise ValueError(
            f"deployed source archive is not a readable tar: {archive_path}") \
            from error
    with archive:
        if "comment" not in archive.pax_headers:
            raise ValueError("deployed tar is not a Git archive with commit PAX")
        archive_commit = archive.pax_headers["comment"]
        if archive_commit != baseline_commit:
            raise ValueError(
                f"git archive commit {archive_commit!r} does not match "
                f"baseline {baseline_commit}")
        members = [member for member in archive.getmembers() if member.isfile()]
        for filename in SOURCE_FILES:
            suffix = f"{source_prefix}/{filename}"
            matches = [
                member for member in members
                if member.name == suffix or member.name.endswith(f"/{suffix}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"deployed source archive must contain exactly one "
                    f"{suffix}, found {len(matches)}")
            stream = archive.extractfile(matches[0])
            if stream is None:
                raise ValueError(f"cannot read archived launch source {suffix}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            archived_sha256 = digest.hexdigest()
            current_sha256 = _sha256(source_dir / filename)
            if archived_sha256 != current_sha256:
                raise ValueError(
                    f"deployed archive source differs from current {suffix}")
            verified.append({
                "archive_member": matches[0].name,
                "current_path": str((source_dir / filename).resolve()),
                "sha256": current_sha256,
            })
    return {
        "git_archive_commit": archive_commit,
        "verified_launch_sources": verified,
    }


def _git_head(repository_root, environment):
    git_metadata = repository_root / ".git"
    if not git_metadata.exists():
        return {
            "head": None,
            "process": None,
            "reason": "repository_has_no_git_metadata",
        }
    result = _run_process_group_captured(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        environment,
        60,
    )
    if result["process"]["exit_code"] != 0:
        raise RuntimeError("git rev-parse HEAD failed")
    head = result["stdout"].strip()
    if BASELINE_PATTERN.fullmatch(head) is None:
        raise ValueError(f"git returned an invalid HEAD identity: {head!r}")
    return {
        "head": head,
        "process": result["process"],
        "reason": None,
    }


def _required_columns(connection, table, expected):
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if not rows:
        raise ValueError(f"msprof database is missing required table {table!r}")
    actual = {row[1] for row in rows}
    missing = set(expected) - actual
    if missing:
        raise ValueError(
            f"msprof table {table!r} is missing columns {sorted(missing)}")


def read_sqlite_schema(database_path):
    """Retain provider schema evidence before interpreting task rows."""
    with sqlite3.connect(database_path) as connection:
        sqlite_master = connection.execute(
            "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master "
            "ORDER BY type, name"
        ).fetchall()
        pragmas = {}
        for table in ("task_time", "ge_summary"):
            pragmas[table] = connection.execute(
                f"PRAGMA table_info({table})").fetchall()
    if not sqlite_master:
        raise ValueError("msprof sqlite_master is empty")
    if any(not pragmas[table] for table in ("task_time", "ge_summary")):
        raise ValueError("required msprof PRAGMA schema is empty")
    return {
        "sqlite_master": [list(row) for row in sqlite_master],
        "pragma_table_info": {
            table: [list(row) for row in pragmas[table]]
            for table in ("task_time", "ge_summary")
        },
    }


def _required_integer(value, field, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{field} must be numeric, got {type(value).__name__}")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(
                f"{field} must be a finite integer, got {value!r}")
        result = int(value)
    elif isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except InvalidOperation as error:
            raise ValueError(f"{field} is not numeric: {value!r}") from error
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(
                f"{field} must be a finite integer, got {value!r}")
        result = int(number)
    else:
        raise AssertionError(f"unhandled numeric type for {field}")
    if positive and result <= 0:
        raise ValueError(f"{field} must be positive, got {result}")
    if not positive and result < 0:
        raise ValueError(f"{field} must be non-negative, got {result}")
    return result


def _optional_nonnegative_integer(value, field):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in ("", "NA", "N/A"):
        return None
    return _required_integer(value, field)


def _normalize_task_time_row(row, cell, expected_task_time_task_type):
    (task_id, stream_id, batch_id, start_time, duration_time,
     task_time_task_type, observed_block_dim, observed_mix_block_dim,
     ge_task_type) = row
    if (not isinstance(task_time_task_type, str)
            or not task_time_task_type.strip()):
        raise TypeError("task_time.task_type must be a non-empty string")
    task_time_task_type = task_time_task_type.strip()
    if task_time_task_type != expected_task_time_task_type:
        raise ValueError(
            f"{cell.task_class} expected task_time.task_type "
            f"{expected_task_time_task_type!r}, found "
            f"{task_time_task_type!r}")
    if not isinstance(ge_task_type, str) or not ge_task_type.strip():
        raise TypeError("ge_summary.task_type must be a non-empty string")
    ge_task_type = ge_task_type.strip()
    if ge_task_type != cell.expected_ge_task_type:
        raise ValueError(
            f"{cell.task_class} expected ge_summary.task_type "
            f"{cell.expected_ge_task_type!r}, found {ge_task_type!r}")
    observed_block_dim = _required_integer(
        observed_block_dim, "ge_summary.block_dim", positive=True)
    observed_mix_block_dim = _optional_nonnegative_integer(
        observed_mix_block_dim, "ge_summary.mix_block_dim")
    normalized_mix_block_dim = (
        0 if observed_mix_block_dim is None else observed_mix_block_dim)
    if observed_block_dim != cell.requested_block_dim:
        raise ValueError(
            f"requested block_dim={cell.requested_block_dim}, "
            f"observed {observed_block_dim}")
    if normalized_mix_block_dim != cell.requested_mix_block_dim:
        raise ValueError(
            f"requested mix_block_dim={cell.requested_mix_block_dim}, "
            f"observed {observed_mix_block_dim!r}")
    return {
        "task_id": _required_integer(task_id, "task_time.task_id"),
        "stream_id": _required_integer(stream_id, "task_time.stream_id"),
        "batch_id": _required_integer(batch_id, "task_time.batch_id"),
        "start_time_ns": _required_integer(
            start_time, "task_time.start_time"),
        "duration_time_ns": _required_integer(
            duration_time, "task_time.duration_time", positive=True),
        "task_time_task_type": task_time_task_type,
        "ge_summary_task_type": ge_task_type,
        "observed_block_dim": observed_block_dim,
        "observed_mix_block_dim": observed_mix_block_dim,
    }


def parse_task_time_database(database_path, cell, launches):
    """Read exact target task rows, validating provider launch metadata."""
    if not isinstance(cell, CaptureCell):
        raise TypeError("cell must be a CaptureCell")
    if not isinstance(launches, int) or isinstance(launches, bool):
        raise TypeError("launches must be an integer")
    if launches <= 0:
        raise ValueError("launches must be positive")
    with sqlite3.connect(database_path) as connection:
        _required_columns(connection, "task_time", (
            "task_id", "stream_id", "batch_id", "start_time",
            "duration_time", "task_type",
        ))
        _required_columns(connection, "ge_summary", (
            "task_id", "stream_id", "batch_id", "op_name", "block_dim",
            "mix_block_dim", "task_type",
        ))
        rows = connection.execute(
            "SELECT t.task_id, t.stream_id, t.batch_id, t.start_time, "
            "t.duration_time, t.task_type, g.block_dim, g.mix_block_dim, "
            "g.task_type "
            "FROM task_time t JOIN ge_summary g "
            "ON t.task_id IS g.task_id AND t.stream_id IS g.stream_id "
            "AND t.batch_id IS g.batch_id WHERE g.op_name=? "
            "ORDER BY t.start_time, t.stream_id, t.task_id, t.batch_id",
            (cell.kernel_name,),
        ).fetchall()

    if cell.task_class in {
            "mix_aic_1_2", "mix_aic_1_0", "mix_aiv_nonempty"}:
        component_task_type = (
            "MIX_AIC"
            if cell.task_class in {"mix_aic_1_2", "mix_aic_1_0"}
            else "MIX_AIV")
        expected_row_count = 2 * launches
        if len(rows) != expected_row_count:
            raise ValueError(
                f"expected {expected_row_count} {cell.kernel_name!r} mixed "
                f"component task_time rows, found {len(rows)}")
        grouped_rows = {}
        for row in rows:
            identity = row[:3]
            if identity not in grouped_rows:
                grouped_rows[identity] = []
            grouped_rows[identity].append(row)
        if len(grouped_rows) != launches:
            raise ValueError(
                f"expected {launches} mixed task identities, found "
                f"{len(grouped_rows)}")
        ordered_groups = sorted(
            grouped_rows.values(),
            key=lambda group: min(
                _required_integer(row[3], "task_time.start_time")
                for row in group),
        )
        records = []
        for group in ordered_groups:
            task_types = []
            for row in group:
                if not isinstance(row[5], str) or not row[5].strip():
                    raise TypeError(
                        "task_time.task_type must be a non-empty string")
                task_types.append(row[5].strip())
            if sorted(task_types) != ["AI_CORE", component_task_type]:
                raise ValueError(
                    "mixed task identity must contain exactly one AI_CORE "
                    f"and one {component_task_type} row, found "
                    f"{task_types!r}")
            complete_row = group[task_types.index("AI_CORE")]
            component_row = group[task_types.index(component_task_type)]
            complete = _normalize_task_time_row(
                complete_row, cell, "AI_CORE")
            component = _normalize_task_time_row(
                component_row, cell, component_task_type)
            complete_end = (
                complete["start_time_ns"] + complete["duration_time_ns"])
            component_end = (
                component["start_time_ns"] + component["duration_time_ns"])
            if (complete["start_time_ns"] > component["start_time_ns"]
                    or complete_end < component_end):
                raise ValueError(
                    "mixed AI_CORE task_time row does not enclose its "
                    f"{component_task_type} component row")
            complete_component = {
                **complete, "task_time_role": "complete_mixed_envelope"}
            mixed_component = {
                **component, "task_time_role": "contained_mixed_component"}
            complete["task_time_role"] = "complete_mixed_envelope"
            complete["task_time_components"] = [
                complete_component, mixed_component]
            records.append(complete)
        return records
    elif cell.task_class in {
            "aic_only", "aiv_only", "aiv_nonempty", "aiv_memory_task",
            "aiv_norm_control_task"}:
        if len(rows) != launches:
            raise ValueError(
                f"expected {launches} {cell.kernel_name!r} task_time rows, "
                f"found {len(rows)}")
        records = []
        for row in rows:
            record = _normalize_task_time_row(
                row, cell, cell.expected_task_time_task_type)
            component = {**record, "task_time_role": "complete_task"}
            record["task_time_role"] = "complete_task"
            record["task_time_components"] = [component]
            records.append(record)
        return records
    else:
        raise ValueError(f"unsupported task class {cell.task_class!r}")


def validate_health_output(path, cell, launches):
    """Validate the CANN-event health line without using it as duration data."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError(f"health output must contain exactly one line: {path}")
    match = HEALTH_PATTERN.fullmatch(lines[0])
    if match is None:
        raise ValueError(f"invalid health output schema: {lines[0]!r}")
    task_class, block_dim, mix_block_dim, emitted_launches, elapsed_ms = (
        match.groups())
    actual = (
        task_class, int(block_dim), int(mix_block_dim), int(emitted_launches))
    expected = (
        cell.task_class, cell.requested_block_dim,
        cell.requested_mix_block_dim, launches)
    if actual != expected:
        raise ValueError(
            f"health output does not match request: {actual!r} != {expected!r}")
    elapsed = float(elapsed_ms)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("CANN-event health elapsed_ms must be positive")
    return {
        "status": "passed",
        "source": "cann_event_health_check_only",
        "elapsed_ms": elapsed,
        "eligible_as_task_duration": False,
    }


def _find_profile_database(profile_dir):
    databases = sorted(profile_dir.glob(
        "PROF_*/device_*/sqlite/ai_core_op_summary.db"))
    if len(databases) != 1:
        raise FileNotFoundError(
            f"expected exactly one ai_core_op_summary.db under {profile_dir}, "
            f"found {len(databases)}")
    return databases[0]


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _write_manifest(path, payload):
    _write_json(path, payload)
    digest = _sha256(path)
    checksum_path = path.with_suffix(".sha256")
    temporary = checksum_path.with_suffix(".sha256.tmp")
    temporary.write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    temporary.replace(checksum_path)


def _append_json_lines(path, records):
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")


def _telemetry(command, output_path, environment):
    _run_logged(command, output_path, environment, timeout=60)


def _device_telemetry(npu_smi, device_id, global_path, board_path, usage_path,
                      environment):
    global_command = [str(npu_smi), "info"]
    board_command = [
        str(npu_smi), "info", "-t", "board", "-i", str(device_id)]
    usage_command = [
        str(npu_smi), "info", "-t", "usages", "-i", str(device_id)]
    _telemetry(global_command, global_path, environment)
    _telemetry(board_command, board_path, environment)
    _telemetry(usage_command, usage_path, environment)
    identity = parse_device_identity(
        global_path.read_text(encoding="utf-8"),
        board_path.read_text(encoding="utf-8"),
        device_id,
    )
    utilization = parse_idle_telemetry(
        usage_path.read_text(encoding="utf-8"))
    return {
        "global_command": global_command,
        "board_command": board_command,
        "usage_command": usage_command,
        "identity": identity,
        "utilization": utilization,
    }


def _process_group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(pgid, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _spawn_process_group(spawn_state, command, **popen_arguments):
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, CAPTURE_SIGNALS)
    try:
        process = subprocess.Popen(
            [str(value) for value in command],
            start_new_session=True,
            preexec_fn=lambda: signal.pthread_sigmask(
                signal.SIG_SETMASK, previous_mask),
            **popen_arguments,
        )
        spawn_state["process"] = process
        spawn_state["pid"] = process.pid
        spawn_state["pgid"] = process.pid
        return process
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _terminate_process_group(process, terminate_timeout_seconds=30):
    pgid = process.pid
    if _process_group_exists(pgid):
        os.killpg(pgid, signal.SIGTERM)
    if process.poll() is None:
        try:
            process.wait(timeout=terminate_timeout_seconds)
        except subprocess.TimeoutExpired:
            if process.poll() is not None:
                raise AssertionError(
                    "process completed while wait reported a timeout")
    if not _wait_for_process_group_exit(pgid, terminate_timeout_seconds):
        os.killpg(pgid, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=30)
        if not _wait_for_process_group_exit(pgid, 30):
            raise RuntimeError(f"process group {pgid} survived SIGKILL")


def _run_process_group_captured(
        command, environment, timeout_seconds, merge_stderr=False):
    metadata = {
        "command": [str(value) for value in command],
        "timeout_seconds": timeout_seconds,
        "pid": None,
        "pgid": None,
        "started_at_utc": None,
        "ended_at_utc": None,
        "exit_code": None,
        "timed_out": False,
        "failure": None,
    }
    spawn_state = {"process": None, "pid": None, "pgid": None}
    process = None
    started_monotonic = time.monotonic()
    stderr_target = subprocess.STDOUT if merge_stderr else subprocess.PIPE
    try:
        process = _spawn_process_group(
            spawn_state,
            metadata["command"],
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            env=environment,
            text=True,
        )
        metadata["pid"] = spawn_state["pid"]
        metadata["pgid"] = spawn_state["pgid"]
        metadata["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        remaining = timeout_seconds - (time.monotonic() - started_monotonic)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        stdout, stderr = process.communicate(timeout=remaining)
        if _process_group_exists(spawn_state["pgid"]):
            raise RuntimeError(
                f"captured command leader exited while process group "
                f"{spawn_state['pgid']} survived: {command}")
        metadata["exit_code"] = process.returncode
        metadata["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        return {
            "stdout": stdout,
            "stderr": stderr,
            "process": metadata,
        }
    except subprocess.TimeoutExpired as error:
        metadata["timed_out"] = True
        failure = TimeoutError(f"captured command timed out: {command}")
        failure_stdout = ""
        failure_stderr = None if merge_stderr else ""
        while True:
            try:
                process = spawn_state["process"]
                if process is not None:
                    _terminate_process_group(process)
                    failure_stdout, failure_stderr = process.communicate(
                        timeout=30)
                    metadata["pid"] = spawn_state["pid"]
                    metadata["pgid"] = spawn_state["pgid"]
                    metadata["exit_code"] = process.returncode
                metadata["ended_at_utc"] = datetime.now(
                    timezone.utc).isoformat()
                metadata["failure"] = {
                    "type": type(failure).__name__,
                    "message": str(failure),
                }
                break
            except CaptureInterrupted as interrupted:
                failure = interrupted
        failure.process_metadata = metadata
        failure.process_stdout = failure_stdout
        failure.process_stderr = failure_stderr
        raise failure from error
    except BaseException as error:
        failure = error
        failure_stdout = ""
        failure_stderr = None if merge_stderr else ""
        while True:
            try:
                process = spawn_state["process"]
                if process is not None:
                    _terminate_process_group(process)
                    failure_stdout, failure_stderr = process.communicate(
                        timeout=30)
                    metadata["pid"] = spawn_state["pid"]
                    metadata["pgid"] = spawn_state["pgid"]
                    metadata["exit_code"] = process.returncode
                metadata["ended_at_utc"] = datetime.now(
                    timezone.utc).isoformat()
                metadata["failure"] = {
                    "type": type(failure).__name__,
                    "message": str(failure),
                }
                break
            except CaptureInterrupted as interrupted:
                failure = interrupted
        failure.process_metadata = metadata
        failure.process_stdout = failure_stdout
        failure.process_stderr = failure_stderr
        raise failure


def _run_logged(command, log_path, environment, timeout=None):
    if timeout is None:
        raise ValueError("logged command timeout must be explicit")
    process_path = log_path.with_suffix(log_path.suffix + ".process.json")
    try:
        result = _run_process_group_captured(
            command, environment, timeout, merge_stderr=True)
    except BaseException as error:
        if not hasattr(error, "process_metadata"):
            raise RuntimeError(
                "captured command failed without process metadata") from error
        log_path.write_text(error.process_stdout, encoding="utf-8")
        _write_json(process_path, error.process_metadata)
        raise
    log_path.write_text(result["stdout"], encoding="utf-8")
    _write_json(process_path, result["process"])
    if result["process"]["exit_code"] != 0:
        raise RuntimeError(
            f"logged command failed with exit "
            f"{result['process']['exit_code']}: {command}")


def _finalize_logged_process_failure(
        spawn_state, metadata, process_manifest_path, error):
    failure = error
    while True:
        try:
            process = spawn_state["process"]
            if process is not None:
                if metadata["pid"] is None:
                    metadata["pid"] = spawn_state["pid"]
                    metadata["pgid"] = spawn_state["pgid"]
                    metadata["started_at_utc"] = datetime.now(
                        timezone.utc).isoformat()
                _terminate_process_group(process)
                metadata["exit_code"] = process.returncode
            metadata["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["failure"] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
            _write_json(process_manifest_path, metadata)
            return failure
        except CaptureInterrupted as interrupted:
            failure = interrupted


def _run_process_group_logged(
        command, log_path, process_manifest_path, environment,
        timeout_seconds):
    metadata = {
        "command": [str(value) for value in command],
        "signal_contract": {
            "handled": [signal.Signals(value).name
                        for value in CAPTURE_SIGNALS],
            "spawn_parent_mask": "blocked_until_process_reference_is_stored",
            "spawn_child_mask": "restored_to_parent_pre_spawn_mask",
            "failure_finalization": "cleanup_and_manifest_retry_on_signal",
        },
        "timeout_seconds": timeout_seconds,
        "pid": None,
        "pgid": None,
        "started_at_utc": None,
        "ended_at_utc": None,
        "exit_code": None,
        "timed_out": False,
        "failure": None,
    }
    _write_json(process_manifest_path, metadata)
    spawn_state = {"process": None, "pid": None, "pgid": None}
    process = None
    started_monotonic = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            process = _spawn_process_group(
                spawn_state,
                metadata["command"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
            )
            metadata["pid"] = spawn_state["pid"]
            metadata["pgid"] = spawn_state["pgid"]
            metadata["started_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(process_manifest_path, metadata)
            remaining = timeout_seconds - (time.monotonic() - started_monotonic)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            return_code = process.wait(timeout=remaining)
        if process is not None and _process_group_exists(process.pid):
            raise RuntimeError(
                f"command leader exited while process group {process.pid} "
                f"survived: {command}")
        metadata["exit_code"] = return_code
        metadata["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        if return_code != 0:
            raise RuntimeError(
                f"command failed with exit {return_code}: {command}")
        _write_json(process_manifest_path, metadata)
        return metadata
    except subprocess.TimeoutExpired as error:
        metadata["timed_out"] = True
        timeout_error = TimeoutError(f"command timed out: {command}")
        finalized = _finalize_logged_process_failure(
            spawn_state, metadata, process_manifest_path, timeout_error)
        raise finalized from error
    except BaseException as error:
        finalized = _finalize_logged_process_failure(
            spawn_state, metadata, process_manifest_path, error)
        raise finalized


def _host_stub_build_command(cmake, build_dir):
    return [
        str(cmake), "--build", str(build_dir), "--target",
        *(f"{kernel_target}_host_stub_obj"
          for kernel_target in KERNEL_TARGETS),
        "-j4",
    ]


def _extract_provider_pack_recipe(build_make_path, ascend_home_path):
    """Read the exact provider PRE_LINK pack recipe from generated Makefiles."""
    lines = build_make_path.read_text(encoding="utf-8").splitlines()
    logical_lines = []
    pending = ""
    for line in lines:
        pending += line
        if pending.endswith("\\"):
            pending = pending[:-1]
        else:
            logical_lines.append(pending)
            pending = ""
    if pending:
        raise ValueError(f"unterminated Make recipe in {build_make_path}")
    recipes = []
    for line in logical_lines:
        if not line.startswith("\t") or "ascendc_pack_kernel" not in line:
            continue
        recipe = line.lstrip()
        if "$" in recipe:
            raise ValueError(
                f"provider pack recipe contains unresolved Make variables: "
                f"{recipe}")
        try:
            tokens = shlex.split(recipe)
        except ValueError as error:
            raise ValueError(
                f"provider pack recipe has invalid shell quoting: {recipe}") \
                from error
        wrappers = [
            token for token in tokens
            if Path(token).name == "ascendc_pack_kernel.sh"
        ]
        if wrappers:
            recipes.append((recipe, tokens, wrappers))
    if len(recipes) != 1:
        raise ValueError(
            f"expected one provider ascendc_pack_kernel.sh PRE_LINK recipe in "
            f"{build_make_path}, found {len(recipes)}")
    recipe, tokens, wrappers = recipes[0]
    if len(wrappers) != 1:
        raise ValueError(
            f"provider pack recipe has ambiguous pack wrapper: {recipe}")

    def option_value(option):
        positions = [
            index for index, token in enumerate(tokens) if token == option
        ]
        if len(positions) != 1:
            raise ValueError(
                f"provider pack recipe must contain one {option}: {recipe}")
        position = positions[0]
        if position + 1 >= len(tokens) or tokens[position + 1].startswith("--"):
            raise ValueError(
                f"provider pack recipe has no value for {option}: {recipe}")
        return tokens[position + 1]

    pack_tool_value = option_value("--pack_tool")
    elf_input_value = option_value("--elf_in")
    add_directory_value = option_value("--add_dir")
    pack_tool_tokens = [
        token for token in tokens
        if Path(token).name == "ascendc_pack_kernel"
    ]
    if (len(pack_tool_tokens) != 1
            or pack_tool_tokens[0] != pack_tool_value):
        raise ValueError(
            f"provider pack recipe has ambiguous --pack_tool executable: "
            f"{recipe}")

    def absolute_path(value, label):
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(
                f"provider pack {label} must be an absolute path: {value}")
        return path.resolve()

    pack_wrapper = absolute_path(wrappers[0], "wrapper")
    pack_tool = absolute_path(pack_tool_value, "tool")
    elf_input = absolute_path(elf_input_value, "--elf_in")
    add_directory = absolute_path(add_directory_value, "--add_dir")
    ascend_home = Path(ascend_home_path).resolve()
    for label, path in (
            ("wrapper", pack_wrapper), ("tool", pack_tool)):
        try:
            path.relative_to(ascend_home)
        except ValueError as error:
            raise ValueError(
                f"provider pack {label} is outside ASCEND_HOME_PATH: {path}") \
                from error
        if not path.is_file():
            raise FileNotFoundError(
                f"provider pack {label} is not a file: {path}")
        if label == "tool" and not os.access(path, os.X_OK):
            raise FileNotFoundError(
                f"provider pack tool is not executable: {path}")
    return {
        "recipe": recipe,
        "tokens": tokens,
        "pack_wrapper": pack_wrapper,
        "pack_tool": pack_tool,
        "elf_input": elf_input,
        "add_directory": add_directory,
    }


def _provider_pack_stage(
        *, kernel_target, expected_device_names, build_dir, ascend_home_path,
        logs_dir, environment, timeout_seconds):
    build_make_path = (
        build_dir / "CMakeFiles" / f"{kernel_target}.dir" / "build.make")
    if not build_make_path.is_file():
        raise FileNotFoundError(
            f"generated provider build metadata is missing: {build_make_path}")
    pack_recipe = _extract_provider_pack_recipe(
        build_make_path, ascend_home_path)
    expected_stub_object = (
        build_dir / "CMakeFiles" / f"{kernel_target}_host_stub_obj.dir"
        / "auto_gen" / kernel_target / "host_stub.cpp.o").resolve()
    stub_object = pack_recipe["elf_input"]
    if stub_object != expected_stub_object:
        raise ValueError(
            f"provider --elf_in does not identify {kernel_target} host stub: "
            f"{stub_object}")
    if not stub_object.is_file():
        raise FileNotFoundError(f"host stub does not exist before pack: {stub_object}")
    device_root = pack_recipe["add_directory"]
    expected_device_root = (
        build_dir / f"{kernel_target}_merge_obj_dir").resolve()
    if device_root != expected_device_root:
        raise ValueError(
            f"provider --add_dir does not equal {kernel_target} merge-object "
            f"directory: {device_root}")
    if not device_root.is_dir():
        raise FileNotFoundError(
            f"provider device artifact directory does not exist: {device_root}")
    device_artifacts = sorted(
        path for path in device_root.rglob("device*.o") if path.is_file())
    artifact_names = sorted(path.name for path in device_artifacts)
    expected_names = sorted(expected_device_names)
    if artifact_names != expected_names:
        raise ValueError(
            f"{kernel_target} device artifact set is {artifact_names}, "
            f"expected exactly {expected_names}")
    device_artifact_identities_before = [
        _file_identity(path, build_dir) for path in device_artifacts]
    pack_wrapper_sha256 = _sha256(pack_recipe["pack_wrapper"])
    pack_tool_sha256 = _sha256(pack_recipe["pack_tool"])
    stub_sha256_before = _sha256(stub_object)
    command = ["/bin/bash", "-c", pack_recipe["recipe"]]
    process = _run_process_group_logged(
        command,
        logs_dir / f"{kernel_target}_provider_pack.log",
        logs_dir / f"{kernel_target}_provider_pack_process.json",
        environment,
        timeout_seconds,
    )
    device_artifacts_after = sorted(
        path for path in device_root.rglob("device*.o") if path.is_file())
    device_artifact_identities_after = [
        _file_identity(path, build_dir) for path in device_artifacts_after]
    if device_artifact_identities_after != device_artifact_identities_before:
        raise RuntimeError(
            f"provider pack changed {kernel_target} device artifact set")
    stub_sha256_after = _sha256(stub_object)
    if stub_sha256_after == stub_sha256_before:
        raise RuntimeError(
            f"provider pack did not change host stub for {kernel_target}")
    if _sha256(pack_recipe["pack_wrapper"]) != pack_wrapper_sha256:
        raise RuntimeError("provider pack wrapper changed during execution")
    if _sha256(pack_recipe["pack_tool"]) != pack_tool_sha256:
        raise RuntimeError("provider pack tool changed during execution")
    return {
        "kernel_target": kernel_target,
        "build_make": _file_identity(build_make_path, build_dir),
        "recipe_argv": pack_recipe["tokens"],
        "pack_wrapper": {
            "path": str(pack_recipe["pack_wrapper"]),
            "sha256": pack_wrapper_sha256,
        },
        "pack_tool": {
            "path": str(pack_recipe["pack_tool"]),
            "sha256": pack_tool_sha256,
        },
        "command": command,
        "process": process,
        "artifact_directory": str(device_root),
        "device_artifacts": device_artifact_identities_before,
        "device_artifact_integrity": {
            "status": "unchanged_after_provider_pack",
            "before_pack": device_artifact_identities_before,
            "after_pack": device_artifact_identities_after,
        },
        "host_stub": {
            "path": str(stub_object),
            "sha256_before_pack": stub_sha256_before,
            "sha256_after_pack": stub_sha256_after,
        },
    }


def _finalize_capture_point_failure(
        spawn_state, invocation, point_manifest_path, error):
    failure = error
    while True:
        try:
            process = spawn_state["process"]
            if process is not None:
                if invocation["pid"] is None:
                    invocation["pid"] = spawn_state["pid"]
                    invocation["pgid"] = spawn_state["pgid"]
                    invocation["started_at_utc"] = datetime.now(
                        timezone.utc).isoformat()
                _terminate_process_group(process)
                invocation["exit_code"] = process.returncode
            invocation["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            invocation["failure"] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
            _write_json(point_manifest_path, invocation)
            return failure
        except CaptureInterrupted as interrupted:
            failure = interrupted


def _capture_point(
        *, cell, run_index, order, launches, binary, msprof, npu_smi,
        device_id, raw_root, telemetry_root, environment,
        msprof_timeout_seconds):
    point_name = (
        f"run_{run_index:03d}_order_{order:03d}_{cell.task_class}_"
        f"bd_{cell.requested_block_dim:02d}")
    point_dir = raw_root / point_name
    point_dir.mkdir()
    point_dir.chmod(0o700)
    interpreter_chain = _prepare_interpreter_chain(
        point_dir, environment)
    try:
        application_wrapper = _create_application_wrapper(
            point_dir, binary, cell.requested_block_dim, launches,
            interpreter_chain)
    except BaseException:
        _close_interpreter_chain(interpreter_chain)
        raise
    health_path = Path(application_wrapper["stdout_path"])
    application_path = application_wrapper["path"]
    msprof_command = [
        str(msprof),
        f"--application={application_path}",
        "--ai-core=off",
        f"--output={point_dir}",
    ]
    msprof_log = point_dir / "msprof.log"
    during_path = telemetry_root / "during" / f"{point_name}.txt"
    point_manifest_path = point_dir / "capture_process.json"
    invocation = {
        "schema_version": 6,
        "point_name": point_name,
        "run": run_index,
        "order": order,
        "cell": asdict(cell),
        "signal_contract": {
            "handled": [signal.Signals(value).name
                        for value in CAPTURE_SIGNALS],
            "spawn_parent_mask": "blocked_until_process_reference_is_stored",
            "spawn_child_mask": "restored_to_parent_pre_spawn_mask",
            "failure_finalization": "cleanup_and_manifest_retry_on_signal",
        },
        "application_wrapper": application_wrapper,
        "msprof_command": msprof_command,
        "msprof_timeout_seconds": msprof_timeout_seconds,
        "msprof_log": str(msprof_log),
        "raw_profile_directory": str(point_dir),
        "task_time_database": None,
        "sqlite_schema_raw": None,
        "during_idle_telemetry": str(during_path),
        "health_check": None,
        "pid": None,
        "pgid": None,
        "started_at_utc": None,
        "ended_at_utc": None,
        "exit_code": None,
        "timed_out": False,
        "failure": None,
    }
    _write_json(point_manifest_path, invocation)
    wrapper_descriptor = None
    spawn_state = {"process": None, "pid": None, "pgid": None}
    process = None
    started_monotonic = time.monotonic()
    try:
        with msprof_log.open("w", encoding="utf-8") as log_stream:
            try:
                wrapper_descriptor, prelaunch_integrity = (
                    _bind_application_wrapper(application_wrapper))
                invocation["application_wrapper"]["prelaunch_integrity"] = (
                    prelaunch_integrity)
                application_path = prelaunch_integrity[
                    "bound_application_path"]
                invocation["application_wrapper"]["application_argv"] = [
                    application_path]
                msprof_command[1] = f"--application={application_path}"
            except BaseException as error:
                invocation["application_wrapper"]["prelaunch_integrity"] = {
                    "status": "failed",
                    "checked_at_utc": datetime.now(
                        timezone.utc).isoformat(),
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
                raise
            _write_json(point_manifest_path, invocation)
            process = _spawn_process_group(
                spawn_state,
                msprof_command,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
            )
            invocation["pid"] = spawn_state["pid"]
            invocation["pgid"] = spawn_state["pgid"]
            invocation["started_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(point_manifest_path, invocation)
            _telemetry(
                [str(npu_smi), "info", "-t", "usages", "-i",
                 str(device_id)],
                during_path,
                environment,
            )
            remaining = msprof_timeout_seconds - (
                time.monotonic() - started_monotonic)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    msprof_command, msprof_timeout_seconds)
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                invocation["timed_out"] = True
                raise TimeoutError(
                    f"msprof timed out for {point_name}") from error
        if process is not None and not _wait_for_process_group_exit(
                process.pid, 5):
            raise RuntimeError(
                f"msprof leader exited while process group {process.pid} "
                f"survived for {point_name}")
        invocation["exit_code"] = return_code
        invocation["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(point_manifest_path, invocation)
        if return_code != 0:
            raise RuntimeError(
                f"msprof failed for {point_name} with exit {return_code}")
        try:
            postlaunch_integrity = _postlaunch_application_integrity(
                application_wrapper, wrapper_descriptor)
            invocation["application_wrapper"]["postlaunch_integrity"] = (
                postlaunch_integrity)
            _write_json(point_manifest_path, invocation)
        except PostlaunchIntegrityError as error:
            invocation["application_wrapper"]["postlaunch_integrity"] = (
                error.integrity)
            _write_json(point_manifest_path, invocation)
            raise
        health = validate_health_output(health_path, cell, launches)
        database = _find_profile_database(point_dir)
        schema_path = point_dir / "sqlite_schema_raw.json"
        _write_json(schema_path, read_sqlite_schema(database))
        task_records = parse_task_time_database(database, cell, launches)
        invocation["task_time_database"] = str(database)
        invocation["sqlite_schema_raw"] = str(schema_path)
        invocation["health_check"] = health
        _write_json(point_manifest_path, invocation)
    except BaseException as error:
        finalized = _finalize_capture_point_failure(
            spawn_state, invocation, point_manifest_path, error)
        _close_application_resources(wrapper_descriptor, interpreter_chain)
        raise finalized
    _close_application_resources(wrapper_descriptor, interpreter_chain)
    return invocation, task_records


def _inventory_files(root, excluded):
    excluded_resolved = {path.resolve() for path in excluded}
    return [
        _file_identity(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() not in excluded_resolved
    ]


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Retain exact-point Ascend 910B AIC/AIV/mixed launch raw data."))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--device-id", required=True, type=_nonnegative_integer)
    parser.add_argument("--runs", required=True, type=_positive_integer)
    parser.add_argument("--launches", required=True, type=_positive_integer)
    parser.add_argument(
        "--campaign",
        choices=(
            "empty", "nonempty_aiv", "nonempty_mix_aiv", "aiv_memory_task",
            "aiv_norm_control_task", "mix_aic_1_0"),
        default="empty")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-aic-block-dim", type=_positive_integer)
    parser.add_argument(
        "--smoke-aiv-block-dim", type=_positive_integer)
    parser.add_argument(
        "--smoke-mixed-block-dim", type=_positive_integer)
    parser.add_argument(
        "--msprof-timeout-seconds", type=_positive_integer, default=300)
    parser.add_argument(
        "--build-timeout-seconds", type=_positive_integer, default=1200)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--msprof", default="msprof")
    parser.add_argument("--npu-smi", default="npu-smi")
    parser.add_argument("--ccec", default="ccec")
    parser.add_argument("--atc", default="atc")
    parser.add_argument("--host-cxx", default="g++")
    return parser


def main(argv=None):
    args = _build_argument_parser().parse_args(argv)
    environment = os.environ.copy()
    environment_policy = _audit_capture_environment(environment)
    previous_signal_handlers = _install_capture_signal_handlers()
    try:
        return _main_after_environment_audit(
            args, argv, environment, environment_policy)
    finally:
        _restore_capture_signal_handlers(previous_signal_handlers)


def _main_after_environment_audit(
        args, argv, environment, environment_policy):
    if BASELINE_PATTERN.fullmatch(args.baseline_commit) is None:
        raise ValueError("--baseline-commit must be a full lowercase commit hash")
    if SHA256_PATTERN.fullmatch(args.source_archive_sha256) is None:
        raise ValueError("--source-archive-sha256 must be lowercase SHA-256")
    deployed_source_archive = args.source_archive.resolve()
    if not deployed_source_archive.is_file():
        raise FileNotFoundError(
            f"--source-archive is not a file: {deployed_source_archive}")
    deployed_source_archive_sha256 = _sha256(deployed_source_archive)
    if deployed_source_archive_sha256 != args.source_archive_sha256:
        raise ValueError(
            "deployed source archive SHA-256 does not match the explicitly "
            "provided identity")
    selected_smoke_cells = validate_capture_selection(
        args.smoke,
        args.runs,
        args.smoke_aic_block_dim,
        args.smoke_aiv_block_dim,
        args.smoke_mixed_block_dim,
        args.campaign,
    )
    repository_root = _repository_root()
    output_dir = _validate_output_path(args.output_dir, repository_root)
    output_dir.mkdir(parents=True)
    (output_dir / "source").mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "raw").mkdir()
    (output_dir / "telemetry" / "during").mkdir(parents=True)
    manifest_path = output_dir / "manifest.json"
    manifest_checksum_path = output_dir / "manifest.sha256"
    invocations_path = output_dir / "invocations.jsonl"
    task_time_path = output_dir / "task_time_raw.jsonl"
    source_dir = Path(__file__).resolve().parent
    verified_deployed_sources = verify_deployed_source_archive(
        deployed_source_archive, source_dir, args.baseline_commit)
    git_process_path = output_dir / "logs" / "git_head_process.json"
    try:
        git_identity = _git_head(repository_root, environment)
    except BaseException as error:
        process_metadata = None
        if hasattr(error, "process_metadata"):
            process_metadata = error.process_metadata
            _write_json(git_process_path, process_metadata)
        _write_manifest(manifest_path, {
            "schema_version": 13,
            "hardware_id": "ascend_910b",
            "capture_state": "failed_preflight_git",
            "environment_policy": environment_policy,
            "git_process": process_metadata,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "signal": _capture_failure_signal(error),
            },
        })
        raise
    if git_identity["process"] is not None:
        _write_json(git_process_path, git_identity["process"])
    git_head = git_identity["head"]
    if git_head is not None and git_head != args.baseline_commit:
        raise ValueError(
            f"git HEAD {git_head} does not match baseline "
            f"{args.baseline_commit}")
    if "ASCEND_HOME_PATH" not in environment:
        raise EnvironmentError("ASCEND_HOME_PATH must be set explicitly")
    if "ASCEND_CANN_PACKAGE_PATH" not in environment:
        raise EnvironmentError("ASCEND_CANN_PACKAGE_PATH must be set explicitly")
    # ASCEND_RT_VISIBLE_DEVICES selects the requested physical device and
    # remaps it to logical device zero inside the benchmark process.  Passing
    # the physical id to aclrtSetDevice after that remapping works only for
    # physical device zero and makes every non-zero selection fail with
    # ACL_ERROR_INVALID_DEVICE.
    environment["ASCEND_DEVICE_ID"] = "0"
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)

    cmake = _resolve_executable(args.cmake, "cmake")
    msprof = _resolve_executable(args.msprof, "msprof")
    npu_smi = _resolve_executable(args.npu_smi, "npu-smi")
    ccec = _resolve_executable(args.ccec, "ccec")
    atc = _resolve_executable(args.atc, "atc")
    host_cxx = _resolve_executable(args.host_cxx, "host C++ compiler")

    snapshot_dir, archive_path, source_identities = _create_source_archive(
        source_dir, output_dir)
    actual_source_archive_sha256 = _sha256(archive_path)
    build_dir = output_dir / "build"
    configure_command = [
        str(cmake), "-S", str(snapshot_dir), "-B", str(build_dir),
        f"-DSOC_VERSION={SOC_VERSION}",
        f"-DASCEND_HOME_PATH={environment['ASCEND_HOME_PATH']}",
        f"-DASCEND_CANN_PACKAGE_PATH={environment['ASCEND_CANN_PACKAGE_PATH']}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    host_stub_build_command = _host_stub_build_command(cmake, build_dir)
    cann_arch_root = Path(environment["ASCEND_HOME_PATH"]) / (
        f"{platform.machine()}-linux")
    if not cann_arch_root.is_dir():
        raise FileNotFoundError(
            f"CANN architecture directory is missing: {cann_arch_root}")
    ascendc_runtime_library = (
        cann_arch_root / "lib64" / "libascendc_runtime.a")
    if not ascendc_runtime_library.is_file():
        raise FileNotFoundError(
            f"AscendC runtime library is missing: {ascendc_runtime_library}")
    link_commands = []
    stub_objects = []
    for kernel_target, source_name, binary_name in (
        ("empty_aic_kernel", "empty_aic_main.cpp", "empty_aic_bench"),
        ("empty_aiv_kernel", "empty_aiv_main.cpp", "empty_aiv_bench"),
        ("nonempty_aiv_kernel", "nonempty_aiv_main.cpp",
         "nonempty_aiv_bench"),
        ("nonempty_mix_aiv_kernel", "nonempty_mix_aiv_main.cpp",
         "nonempty_mix_aiv_bench"),
        ("aiv_memory_task_floor_kernel", "aiv_memory_task_floor_main.cpp",
         "aiv_memory_task_floor_bench"),
        ("aiv_norm_control_task_floor_kernel",
         "aiv_norm_control_task_floor_main.cpp",
         "aiv_norm_control_task_floor_bench"),
        ("empty_mixed_kernel", "empty_mixed_main.cpp", "empty_mixed_bench"),
        ("empty_mix_aic_kernel", "empty_mix_aic_main.cpp",
         "empty_mix_aic_bench"),
    ):
        stub_object = (
            build_dir / "CMakeFiles" / f"{kernel_target}_host_stub_obj.dir"
            / "auto_gen" / kernel_target / "host_stub.cpp.o")
        stub_objects.append(stub_object)
        link_commands.append([
            str(host_cxx), "-std=c++17", "-O2",
            f"-I{cann_arch_root / 'include'}",
            f"-I{cann_arch_root / 'runtime' / 'include'}",
            f"-I{build_dir / 'auto_gen' / kernel_target}",
            str(stub_object), str(snapshot_dir / source_name),
            f"-L{cann_arch_root / 'lib64'}",
            f"-L{cann_arch_root / 'runtime' / 'lib64'}",
            f"-L{Path(environment['ASCEND_HOME_PATH']) / 'lib64'}",
            "-Wl,--whole-archive",
            str(ascendc_runtime_library),
            "-Wl,--no-whole-archive",
            "-lascendcl", "-lruntime", "-lprofapi", "-lmmpa", "-lc_sec",
            "-lstdc++", "-lpthread", "-ldl",
            "-o", str(build_dir / binary_name),
        ])
    manifest = {
        "schema_version": 13,
        "hardware_id": "ascend_910b",
        "soc_version": SOC_VERSION,
        "baseline_commit": args.baseline_commit,
        "git_head": git_head,
        "git_identity": git_identity,
        "capture_state": "building",
        "campaign": args.campaign,
        "capture_scope": "ABI_smoke" if args.smoke else "candidate_capture",
        "evidence_status": "ABI_smoke" if args.smoke else "candidate_only",
        "candidate_eligibility": (
            "not_candidate" if args.smoke else "candidate_only"),
        "hw_spec_eligibility": "not_hw_spec_eligible",
        "duration_policy": "exact_raw_task_time_only_no_fit_no_interpolation",
        "production_or_evaluation_duration_consumed": False,
        "checksum_algorithm": "sha256",
        "manifest_checksum": str(manifest_checksum_path),
        "output_directory": str(output_dir),
        "runner_command": [sys.executable, str(Path(__file__).resolve()),
                           *(sys.argv[1:] if argv is None else argv)],
        "runner_cwd": str(Path.cwd()),
        "environment_policy": environment_policy,
        "device": {
            "requested_device_id": args.device_id,
            "ASCEND_DEVICE_ID": environment["ASCEND_DEVICE_ID"],
            "ASCEND_RT_VISIBLE_DEVICES": (
                environment["ASCEND_RT_VISIBLE_DEVICES"]),
            "observed": None,
        },
        "runs": args.runs,
        "launches_per_point": args.launches,
        "point_count_per_run": (
            len(selected_smoke_cells)
            if args.smoke else len(build_cells(args.campaign))),
        "schedule": (
            "scoped_single_run_smoke"
            if args.smoke else "paired_forward_reverse_rotations"),
        "minimum_candidate_runs": MIN_CANDIDATE_RUNS,
        "msprof_timeout_seconds": args.msprof_timeout_seconds,
        "source_archive": {
            "path": str(archive_path),
            "sha256": actual_source_archive_sha256,
            "scope": "runtime_launch_source_subset",
        },
        "deployed_source_archive": {
            "path": str(deployed_source_archive),
            "expected_sha256": args.source_archive_sha256,
            "actual_sha256": deployed_source_archive_sha256,
            "baseline_commit": args.baseline_commit,
            "git_archive_commit": (
                verified_deployed_sources["git_archive_commit"]),
            "verified_launch_sources": (
                verified_deployed_sources["verified_launch_sources"]),
        },
        "source_snapshot_directory": str(snapshot_dir),
        "source_files": source_identities,
        "tools": None,
        "build_contract": "cann_8_5_1_provider_pack_host_stub_direct_link",
        "build_commands": [
            configure_command, host_stub_build_command, *link_commands],
        "build_timeout_seconds": args.build_timeout_seconds,
        "build_processes": [],
        "provider_pack_evidence": [],
        "telemetry": {
            "pre_global": str(output_dir / "telemetry" / "pre_global.txt"),
            "pre_board": str(output_dir / "telemetry" / "pre_board.txt"),
            "pre_usage": str(output_dir / "telemetry" / "pre_usage.txt"),
            "during_directory": str(output_dir / "telemetry" / "during"),
            "post_global": str(output_dir / "telemetry" / "post_global.txt"),
            "post_board": str(output_dir / "telemetry" / "post_board.txt"),
            "post_usage": str(output_dir / "telemetry" / "post_usage.txt"),
        },
        "preflight": None,
        "postflight": None,
        "invocations": str(invocations_path),
        "raw_task_time": str(task_time_path),
        "raw_task_time_schema_version": RAW_TASK_TIME_SCHEMA_VERSION,
        "capture_process_schema_version": 6,
        "invocations_schema_version": 6,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "failure": None,
        "binary_files": None,
        "checksum_inventory": None,
    }
    _write_manifest(manifest_path, manifest)
    try:
        manifest["tools"] = {
            "cmake": _tool_identity(
                cmake, ("--version",), (0,), environment),
            "msprof": _tool_identity(
                msprof, ("--help",), (0,), environment),
            "npu_smi": _tool_identity(
                npu_smi, ("-v",), (0,), environment),
            "compiler_ccec": _tool_identity(
                ccec, ("--version",), (0,), environment),
            "cann_atc": _tool_identity(
                atc, ("--help",), (0,), environment),
            "host_cxx": _tool_identity(
                host_cxx, ("--version",), (0,), environment),
            "cann_package": _cann_package_identity(
                environment["ASCEND_CANN_PACKAGE_PATH"]),
        }
        _write_manifest(manifest_path, manifest)
        configure_process = _run_process_group_logged(
            configure_command, output_dir / "logs" / "cmake_configure.log",
            output_dir / "logs" / "cmake_configure_process.json",
            environment, args.build_timeout_seconds)
        manifest["build_processes"].append(configure_process)
        host_stub_process = _run_process_group_logged(
            host_stub_build_command,
            output_dir / "logs" / "ascendc_host_stub_build.log",
            output_dir / "logs" / "ascendc_host_stub_build_process.json",
            environment,
            args.build_timeout_seconds,
        )
        manifest["build_processes"].append(host_stub_process)
        for stub_object in stub_objects:
            if not stub_object.is_file():
                raise FileNotFoundError(
                    f"AscendC host-stub target did not produce {stub_object}")
        pack_requirements = (
            ("empty_aic_kernel", ("device_aic.o",)),
            ("empty_aiv_kernel", ("device_aiv.o",)),
            ("nonempty_aiv_kernel", ("device_aiv.o",)),
            ("nonempty_mix_aiv_kernel", ("device_aiv.o",)),
            ("aiv_memory_task_floor_kernel", ("device_aiv.o",)),
            ("aiv_norm_control_task_floor_kernel", ("device_aiv.o",)),
            ("empty_mixed_kernel", ("device.o",)),
            ("empty_mix_aic_kernel", ("device_aic.o",)),
        )
        pack_commands = []
        for kernel_target, required_device_names in pack_requirements:
            evidence = _provider_pack_stage(
                kernel_target=kernel_target,
                expected_device_names=required_device_names,
                build_dir=build_dir,
                ascend_home_path=environment["ASCEND_HOME_PATH"],
                logs_dir=output_dir / "logs",
                environment=environment,
                timeout_seconds=args.build_timeout_seconds,
            )
            manifest["provider_pack_evidence"].append(evidence)
            manifest["build_processes"].append(evidence["process"])
            pack_commands.append(evidence["command"])
        manifest["build_commands"] = [
            configure_command,
            host_stub_build_command,
            *pack_commands,
            *link_commands,
        ]
        _write_manifest(manifest_path, manifest)
        for index, link_command in enumerate(link_commands, start=1):
            link_process = _run_process_group_logged(
                link_command,
                output_dir / "logs" / f"direct_link_{index}.log",
                output_dir / "logs" / f"direct_link_{index}_process.json",
                environment,
                args.build_timeout_seconds,
            )
            manifest["build_processes"].append(link_process)
        _write_manifest(manifest_path, manifest)
        binaries = {}
        for name in (
                "empty_aic_bench", "empty_aiv_bench",
                "nonempty_aiv_bench", "aiv_memory_task_floor_bench",
                "aiv_norm_control_task_floor_bench",
                "empty_mixed_bench", "empty_mix_aic_bench",
                "nonempty_mix_aiv_bench"):
            path = build_dir / name
            if not path.is_file() or not os.access(path, os.X_OK):
                raise FileNotFoundError(f"build did not produce executable {path}")
            binaries[name] = path
        manifest["binary_files"] = [
            _file_identity(binaries[name], output_dir)
            for name in sorted(binaries)
        ]
        manifest["capture_state"] = "capturing"
        _write_manifest(manifest_path, manifest)

        manifest["preflight"] = _device_telemetry(
            npu_smi,
            args.device_id,
            Path(manifest["telemetry"]["pre_global"]),
            Path(manifest["telemetry"]["pre_board"]),
            Path(manifest["telemetry"]["pre_usage"]),
            environment,
        )
        manifest["device"]["observed"] = manifest["preflight"]["identity"]
        _write_manifest(manifest_path, manifest)
        for run_index in range(1, args.runs + 1):
            run_cells = (
                selected_smoke_cells
                if args.smoke else ordered_cells(run_index, args.campaign))
            for order, cell in enumerate(run_cells, start=1):
                invocation, task_records = _capture_point(
                    cell=cell,
                    run_index=run_index,
                    order=order,
                    launches=args.launches,
                    binary=binaries[cell.binary_name],
                    msprof=msprof,
                    npu_smi=npu_smi,
                    device_id=args.device_id,
                    raw_root=output_dir / "raw",
                    telemetry_root=output_dir / "telemetry",
                    environment=environment,
                    msprof_timeout_seconds=args.msprof_timeout_seconds,
                )
                _append_json_lines(invocations_path, (invocation,))
                enriched = []
                for trial, record in enumerate(task_records, start=1):
                    enriched.append({
                        "run": run_index,
                        "trial": trial,
                        "order": order,
                        "task_class": cell.task_class,
                        "requested_block_dim": cell.requested_block_dim,
                        "requested_mix_block_dim": (
                            cell.requested_mix_block_dim),
                        "raw_profile_directory": (
                            invocation["raw_profile_directory"]),
                        **record,
                    })
                _append_json_lines(task_time_path, enriched)
        manifest["postflight"] = _device_telemetry(
            npu_smi,
            args.device_id,
            Path(manifest["telemetry"]["post_global"]),
            Path(manifest["telemetry"]["post_board"]),
            Path(manifest["telemetry"]["post_usage"]),
            environment,
        )
        manifest["capture_state"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["checksum_inventory"] = _inventory_files(
            output_dir, (manifest_path, manifest_checksum_path))
        _write_manifest(manifest_path, manifest)
    except BaseException as error:
        manifest["capture_state"] = "failed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "signal": None,
            "post_failure_telemetry": None,
            "post_telemetry_failure": None,
        }
        manifest["failure"]["signal"] = _capture_failure_signal(error)
        pre_path = Path(manifest["telemetry"]["pre_usage"])
        post_path = Path(manifest["telemetry"]["post_usage"])
        if pre_path.is_file() and not post_path.exists():
            try:
                manifest["failure"]["post_failure_telemetry"] = (
                    _device_telemetry(
                        npu_smi,
                        args.device_id,
                        Path(manifest["telemetry"]["post_global"]),
                        Path(manifest["telemetry"]["post_board"]),
                        post_path,
                        environment,
                    ))
            except Exception as telemetry_error:
                manifest["failure"]["post_telemetry_failure"] = {
                    "type": type(telemetry_error).__name__,
                    "message": str(telemetry_error),
                }
                manifest["checksum_inventory"] = _inventory_files(
                    output_dir, (manifest_path, manifest_checksum_path))
                _write_manifest(manifest_path, manifest)
                raise RuntimeError(
                    "capture failed and post-capture telemetry also failed; "
                    f"original {type(error).__name__}: {error}") from telemetry_error
        manifest["checksum_inventory"] = _inventory_files(
            output_dir, (manifest_path, manifest_checksum_path))
        _write_manifest(manifest_path, manifest)
        raise
    return 0


if __name__ == "__main__":
    main()
