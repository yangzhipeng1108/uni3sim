#!/usr/bin/env python3
"""Build and capture the CANN 8.5.1 FIA BlockReduceMax mechanism."""

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


ROW_CHUNKS = tuple(range(8, 65, 8))
TAIL_COLUMNS = (
    1, 32, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255,
    257, 319, 320, 321, 383, 384, 385, 447, 448, 449, 511,
    513, 576, 640, 704, 768, 832, 896, 960, 1024,
)
COLUMN_ALIGNMENTS = (16, 32)
SPECIAL_COLUMNS = (256, 512)
MAXIMUM_SOURCE_ELEMENTS = 8192
ITERATIONS = 2048
TRIALS = 6
SYSTEM_COUNTER_HZ = 50_000_000
SYSTEM_COUNTER_TICK_NS = 20.0
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "block_reduce_max_contract.h",
    "block_reduce_max_kernel.cpp",
    "block_reduce_max_main.cpp",
    "block_reduce_max_report.py",
    "block_reduce_max_runner.py",
)
RAW_FIELDS = (
    "run_id", "trial", "order_direction", "order_position",
    "process_pid", "branch", "row_chunk_round", "column_count",
    "column_count_round", "column_alignment", "iterations",
    "system_cycles", "per_iteration_ns", "first_row_max",
    "last_row_max", "validation",
)
CHILD_HEADER = (
    "row_chunk_round,column_count,column_count_round,column_alignment,"
    "iterations,system_cycles,per_iteration_ns,first_row_max,"
    "last_row_max,validation"
)
CANN_ROOT = Path("/usr/local/Ascend/cann-8.5.1")
CANN_VERSION_FILE = CANN_ROOT / "aarch64-linux/ascend_toolkit_install.info"
PRODUCTION_SOURCE = (
    CANN_ROOT
    / "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc"
    / "fused_infer_attention_score/attn_infra/epilogue/block"
    / "block_epilogue_online_softmax.hpp"
)
EXPECTED_CANN = {
    "package": "Ascend-cann-toolkit",
    "version": "8.5.1",
    "innerversion": "V100R001C25SPC002B220",
}
EXPECTED_PRODUCTION_SOURCE_SHA256 = (
    "60f5cf14d3c7613f349b17ef3919cf796fba89304dd7d22429f58a57d3c58d2c"
)
PRODUCTION_SOURCE_MARKERS = (
    "void RowmaxSPECTILE512(",
    "void RowmaxSPECTILE256(",
    "void RowmaxTAILTILE(",
    "void CalcLocalRowMax(",
    "AscendC::BlockReduceMax<float, false>(",
    "AscendC::PipeBarrier<PIPE_V>();",
)
BENCHMARK_SOURCE_MARKERS = PRODUCTION_SOURCE_MARKERS
CMAKE = Path("/usr/bin/cmake")
ASCEND_HOME_PATH = CANN_ROOT
NPU_SMI = Path("/usr/local/bin/npu-smi")
LOGICAL_DEVICE_ID = 0
BUILD_TIMEOUT_SECONDS = 900
CHILD_TIMEOUT_SECONDS = 900
MAX_RELATIVE_TRIAL_SPREAD = 0.15
_ACTIVE_PROCESS = None


def _round_up(value, alignment):
    if value <= 0 or alignment <= 0:
        raise ValueError("round-up operands must be positive")
    return (value + alignment - 1) // alignment * alignment


def _branch(column_count):
    if column_count == 512:
        return "special_512"
    if column_count == 256:
        return "special_256"
    if 1 <= column_count <= 1024 and column_count not in SPECIAL_COLUMNS:
        return "tail"
    raise ValueError("column count is outside the production path")


def _cells():
    cells = []
    for column_count in SPECIAL_COLUMNS:
        for row_chunk_round in ROW_CHUNKS:
            if row_chunk_round * column_count <= MAXIMUM_SOURCE_ELEMENTS:
                cells.append((
                    _branch(column_count), row_chunk_round,
                    column_count, column_count, 16,
                ))
    for column_alignment in COLUMN_ALIGNMENTS:
        for column_count in TAIL_COLUMNS:
            column_count_round = _round_up(
                column_count, column_alignment)
            for row_chunk_round in ROW_CHUNKS:
                if (row_chunk_round * column_count_round
                        <= MAXIMUM_SOURCE_ELEMENTS):
                    cells.append((
                        "tail", row_chunk_round, column_count,
                        column_count_round, column_alignment,
                    ))
    result = tuple(cells)
    if len(result) != len(set(result)):
        raise AssertionError("campaign contains duplicate cells")
    return result


CELLS = _cells()


def _ordered_cells(trial_index):
    if trial_index < 0 or trial_index >= TRIALS:
        raise ValueError("trial index is outside the frozen campaign")
    offset = (trial_index // 2 * 17) % len(CELLS)
    rotated = CELLS[offset:] + CELLS[:offset]
    if trial_index % 2 == 0:
        return "forward", rotated
    return "reverse", tuple(reversed(rotated))


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


def _run_checked(command, *, environment=None, timeout=30):
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True,
        env=environment, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command returned {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return completed.stdout


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


def _validate_source(path, expected_sha256, markers, label):
    if not path.is_file():
        raise FileNotFoundError(f"required {label} source is missing: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, "
            f"got {observed}")
    source = path.read_text(encoding="utf-8")
    missing = tuple(marker for marker in markers if marker not in source)
    if missing:
        raise ValueError(f"{label} source lost required markers: {missing}")
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "required_markers": list(markers),
    }


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


def _parse_global_identity(text, physical_device_id):
    if not isinstance(text, str) or not text.strip():
        raise TypeError("npu-smi global telemetry must be non-empty")
    version = re.findall(
        r"npu-smi\s+([0-9]+(?:\.[0-9]+)+)", text,
        flags=re.IGNORECASE)
    header_version = re.findall(
        r"Version\s*:\s*([0-9]+(?:\.[0-9]+)+)", text,
        flags=re.IGNORECASE)
    if len(version) != 1 or header_version != version:
        raise ValueError("npu-smi version identity is ambiguous")
    model_pattern = rf"^\|\s*{physical_device_id}\s+910B2\s+\|"
    if len(re.findall(model_pattern, text, flags=re.MULTILINE)) != 1:
        raise ValueError(
            "npu-smi does not identify the selected physical device as "
            "910B2")
    return {
        "physical_device_id": physical_device_id,
        "logical_device_id": LOGICAL_DEVICE_ID,
        "model": "910B2",
        "npu_smi_version": version[0],
    }


def _parse_child_row(line, expected_cell):
    fields = line.split(",")
    if len(fields) != 10:
        raise ValueError("child result does not contain ten fields")
    if fields[9] != "pass":
        raise ValueError("child validation did not pass")
    branch, rows, columns, columns_round, alignment = expected_cell
    integers = tuple(int(value) for value in fields[:6])
    expected = (
        rows, columns, columns_round, alignment, ITERATIONS,
    )
    if integers[:5] != expected or integers[5] <= 0:
        raise ValueError("child result changed the requested cell")
    per_iteration_ns = float(fields[6])
    first_row_max = float(fields[7])
    last_row_max = float(fields[8])
    recomputed = integers[5] * SYSTEM_COUNTER_TICK_NS / ITERATIONS
    if not math.isclose(
            per_iteration_ns, recomputed, rel_tol=0.0, abs_tol=5.1e-10):
        raise ValueError("child per-iteration time is inconsistent")
    if first_row_max != 1.0 or last_row_max != 1.0:
        raise ValueError("child row-max validation value is inconsistent")
    return {
        "branch": branch,
        "row_chunk_round": rows,
        "column_count": columns,
        "column_count_round": columns_round,
        "column_alignment": alignment,
        "iterations": ITERATIONS,
        "system_cycles": integers[5],
        "per_iteration_ns": f"{per_iteration_ns:.9f}",
        "first_row_max": f"{first_row_max:.6f}",
        "last_row_max": f"{last_row_max:.6f}",
        "validation": "pass",
    }


def _build_command(binary, cells):
    command = [str(binary), "--batch"]
    for _branch_name, rows, columns, _round, alignment in cells:
        command.extend((
            str(rows), str(columns), str(alignment), str(ITERATIONS)))
    return command


def _terminate_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _handle_signal(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128 + signum)


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


def _build(benchmark_root, output_dir):
    if not CMAKE.is_file() or not os.access(CMAKE, os.X_OK):
        raise FileNotFoundError(f"fixed CMake binary is unavailable: {CMAKE}")
    toolchain = (
        ASCEND_HOME_PATH
        / "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
    if not toolchain.is_file():
        raise FileNotFoundError(f"AscendC toolchain is unavailable: {toolchain}")
    build_dir = output_dir / "build"
    configure_command = [
        str(CMAKE), "-S", str(benchmark_root), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DASCEND_HOME_PATH={ASCEND_HOME_PATH}",
        f"-DASCEND_CANN_PACKAGE_PATH={ASCEND_HOME_PATH}",
        "-DSOC_VERSION=Ascend910B2",
    ]
    build_command = [
        str(CMAKE), "--build", str(build_dir), "--target",
        "block_reduce_max_service", "--parallel", "4",
    ]
    environment = os.environ.copy()
    configure = subprocess.run(
        configure_command, check=False, capture_output=True, text=True,
        env=environment, timeout=BUILD_TIMEOUT_SECONDS)
    (output_dir / "cmake_configure.stdout").write_text(
        configure.stdout, encoding="utf-8")
    (output_dir / "cmake_configure.stderr").write_text(
        configure.stderr, encoding="utf-8")
    if configure.returncode != 0:
        raise RuntimeError("CMake configuration failed; see captured logs")
    built = subprocess.run(
        build_command, check=False, capture_output=True, text=True,
        env=environment, timeout=BUILD_TIMEOUT_SECONDS)
    (output_dir / "cmake_build.stdout").write_text(
        built.stdout, encoding="utf-8")
    (output_dir / "cmake_build.stderr").write_text(
        built.stderr, encoding="utf-8")
    if built.returncode != 0:
        raise RuntimeError("CMake build failed; see captured logs")
    binary = build_dir / "block_reduce_max_service"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("build did not produce the expected executable")
    return binary, configure_command, build_command


def _run_campaign(args, benchmark_root):
    global _ACTIVE_PROCESS
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID contains unsupported characters")
    if COMMIT_PATTERN.fullmatch(args.source_commit) is None:
        raise ValueError("source commit must be 40 lowercase hex digits")
    if SHA256_PATTERN.fullmatch(args.expected_snapshot_sha256) is None:
        raise ValueError("expected snapshot must be a lowercase SHA256")
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
    if not CANN_VERSION_FILE.is_file():
        raise FileNotFoundError(f"CANN version file is missing: {CANN_VERSION_FILE}")
    cann_identity = _parse_cann_version(
        CANN_VERSION_FILE.read_text(encoding="utf-8"))
    production_identity = _validate_source(
        PRODUCTION_SOURCE, EXPECTED_PRODUCTION_SOURCE_SHA256,
        PRODUCTION_SOURCE_MARKERS, "production FIA online-softmax")
    benchmark_identity = {
        "path": str((benchmark_root / "block_reduce_max_kernel.cpp").resolve()),
        "sha256": _sha256(benchmark_root / "block_reduce_max_kernel.cpp"),
        "required_markers": list(BENCHMARK_SOURCE_MARKERS),
    }
    benchmark_source = (
        benchmark_root / "block_reduce_max_kernel.cpp"
    ).read_text(encoding="utf-8")
    missing = tuple(
        marker for marker in BENCHMARK_SOURCE_MARKERS
        if marker not in benchmark_source)
    if missing:
        raise ValueError(f"benchmark source lost required markers: {missing}")

    output_dir.mkdir(parents=True)
    binary, configure_command, build_command = _build(
        benchmark_root, output_dir)
    if _snapshot_sha256(benchmark_root) != observed_snapshot:
        raise RuntimeError("audited source snapshot changed during build")

    global_command = [str(NPU_SMI), "info"]
    usage_command = [
        str(NPU_SMI), "info", "-t", "usages", "-i",
        str(args.physical_device_id)]
    global_text = _run_checked(global_command)
    pre_usage_text = _run_checked(usage_command)
    device_identity = _parse_global_identity(
        global_text, args.physical_device_id)
    pre_usage = _parse_idle_usage(pre_usage_text)
    (output_dir / "npu_smi_global.txt").write_text(
        global_text, encoding="utf-8")
    (output_dir / "npu_smi_pre.txt").write_text(
        pre_usage_text, encoding="utf-8")

    environment = os.environ.copy()
    environment["ASCEND_DEVICE_ID"] = str(LOGICAL_DEVICE_ID)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_device_id)
    raw_path = output_dir / "raw.csv"
    trial_artifacts = []
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for trial_index in range(TRIALS):
            direction, cells = _ordered_cells(trial_index)
            command = _build_command(binary, cells)
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=environment, start_new_session=True)
            _ACTIVE_PROCESS = process
            try:
                stdout, stderr = process.communicate(
                    timeout=CHILD_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                _terminate_process_group(process)
                raise RuntimeError(
                    f"trial child PID {process.pid} timed out") from error
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
            for order_position, (cell, line) in enumerate(
                    zip(cells, lines[1:])):
                parsed = _parse_child_row(line, cell)
                writer.writerow({
                    "run_id": args.run_id,
                    "trial": trial_index + 1,
                    "order_direction": direction,
                    "order_position": order_position,
                    "process_pid": process.pid,
                    **parsed,
                })
            stream.flush()
            os.fsync(stream.fileno())
            trial_artifacts.append({
                "trial": trial_index + 1,
                "direction": direction,
                "pid": process.pid,
                "command": command,
                "stdout": stdout_path.name,
                "stdout_sha256": _sha256(stdout_path),
                "stderr": stderr_path.name,
                "stderr_sha256": _sha256(stderr_path),
            })

    post_usage_text = _run_checked(usage_command)
    post_usage = _parse_idle_usage(post_usage_text)
    (output_dir / "npu_smi_post.txt").write_text(
        post_usage_text, encoding="utf-8")
    provenance = {
        "schema": "ascend_910b_fia_block_reduce_max_provenance_v1",
        "run_id": args.run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "source_identity_mode": source_identity_mode,
        "source_snapshot_sha256": observed_snapshot,
        "source_files": {
            name: _sha256(benchmark_root / name) for name in SOURCE_FILENAMES},
        "binary": {
            "path": str(binary.resolve()),
            "sha256": _sha256(binary),
            "configure_command": configure_command,
            "build_command": build_command,
            "configure_stdout_sha256": _sha256(
                output_dir / "cmake_configure.stdout"),
            "configure_stderr_sha256": _sha256(
                output_dir / "cmake_configure.stderr"),
            "build_stdout_sha256": _sha256(output_dir / "cmake_build.stdout"),
            "build_stderr_sha256": _sha256(output_dir / "cmake_build.stderr"),
        },
        "cann": {
            "identity": cann_identity,
            "root": str(CANN_ROOT),
            "version_file": str(CANN_VERSION_FILE),
            "version_file_sha256": _sha256(CANN_VERSION_FILE),
            "production_source": production_identity,
        },
        "static_instruction_evidence": {
            "claim": (
                "the benchmark mirrors CalcLocalRowMax's special-512, "
                "special-256, and tail BlockReduceMax/Max instruction paths "
                "with every source-required PIPE_V barrier"),
            "production": production_identity,
            "benchmark": benchmark_identity,
            "disassembly_claimed": False,
        },
        "device": {
            **device_identity,
            "selection_environment": {
                "ASCEND_DEVICE_ID": str(LOGICAL_DEVICE_ID),
                "ASCEND_RT_VISIBLE_DEVICES": str(args.physical_device_id),
            },
        },
        "idle_telemetry": {
            "pre": {
                "values": pre_usage,
                "path": "npu_smi_pre.txt",
                "sha256": _sha256(output_dir / "npu_smi_pre.txt"),
                "command": usage_command,
            },
            "post": {
                "values": post_usage,
                "path": "npu_smi_post.txt",
                "sha256": _sha256(output_dir / "npu_smi_post.txt"),
                "command": usage_command,
            },
        },
        "clock": {
            "source": "GetSystemCycle",
            "frequency_hz": SYSTEM_COUNTER_HZ,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
        },
        "measurement_scope": (
            "UB-resident local row max only: exact CANN 8.5.1 FIA "
            "CalcLocalRowMax BlockReduceMax/Max/mask path and required "
            "PIPE_V barriers; excludes GM traffic, other softmax stages, "
            "cross-domain synchronization, and host/kernel launch time"),
        "production_or_evaluation_duration_consumed": False,
        "campaign": {
            "row_chunks": list(ROW_CHUNKS),
            "tail_columns": list(TAIL_COLUMNS),
            "special_columns": list(SPECIAL_COLUMNS),
            "column_alignments": list(COLUMN_ALIGNMENTS),
            "iterations": ITERATIONS,
            "trials": TRIALS,
            "cell_count": len(CELLS),
            "order": "rotated_balanced_forward_reverse",
            "maximum_relative_trial_spread": MAX_RELATIVE_TRIAL_SPREAD,
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
    _write_json_exclusive(output_dir / "provenance.json", provenance)
    print(output_dir / "provenance.json")


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
    run.add_argument(
        "--physical-device-id", type=int, choices=tuple(range(8)),
        required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        print(_snapshot_sha256(benchmark_root))
    elif args.command == "run":
        _run_campaign(args, benchmark_root)
    else:
        raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
