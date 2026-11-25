#!/usr/bin/env python3
"""Capture the source-bound A800 Add-RMSNorm reduction mechanism."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = Path(__file__).resolve()
SOURCE_PATH = RUNNER_PATH.with_name("block_reduce_latency.cu")
REPORT_PATH = RUNNER_PATH.with_name("block_reduce_latency_report.py")
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / "nvidia_a800")
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
EXPECTED_CUDA_RELEASE = "12.8"
EXPECTED_NVCC_BUILD = "V12.8.93"
EXPECTED_CUB_VERSION = 200700
SM_REGISTER_CAPACITY = 65_536
SM_SHARED_MEMORY_BYTES = 163_840
SM_THREAD_CAPACITY = 2_048
SM_BLOCK_CAPACITY = 32
EXPECTED_PRODUCTION_BINARY_SHA256 = (
    "984de21ce38cc8083d207663a380ca75acc6ad3dc230858092f59e363c4eebd5")
EXPECTED_PRODUCTION_KERNEL = (
    "_ZN4vllm25fused_add_rms_norm_kernelIN3c108BFloat16ELi8EEENSt9enable_"
    "ifIXaagtT0_Li0Esr4vllm12_typeConvertIT_EE6existsEvE4typeEPS4_lS7_"
    "PKS4_fii")
BLOCK_THREADS = (256, 1024)
GRID_MULTIPLIERS = (0, 1, 2, 4, 8, 16)
DEFAULT_SAMPLES = 200
DEFAULT_TRIALS = 3
ROTATION_STRIDE = 7
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
SASS_INSTRUCTION_PATTERN = re.compile(
    r"/\*([0-9a-fA-F]+)\*/\s+(@!?P\d+\s+)?"
    r"([A-Z][A-Z0-9_.]*)\s+(.*?)\s*;")
COMMON_BINARY_FIELDS = (
    "sample", "device_ordinal", "sm_count", "threads", "grid",
)
GLOBAL_BINARY_FIELDS = COMMON_BINARY_FIELDS + (
    "block_median_ns", "grid_span_ns", "validation_checksum",
    "validation_status",
)
CYCLE_BINARY_FIELDS = COMMON_BINARY_FIELDS + (
    "block_median_cycles", "sm_local_span_cycles", "validation_checksum",
    "validation_status",
)
RAW_FIELDS = (
    "run_id", "trial", "position", "physical_device", "device_uuid",
    "pci_bus_id", "driver_version", "pstate", "sm_clock_before_mhz",
    "sm_clock_after_mhz", "memory_clock_before_mhz",
    "memory_clock_after_mhz",
) + COMMON_BINARY_FIELDS + (
    "block_median_cycles", "sm_local_span_cycles", "grid_span_ns",
    "validation_checksum", "validation_status",
)
DEVICE_QUERY_FIELDS = (
    "uuid,name,pci.bus_id,driver_version,pstate,memory.used,"
    "utilization.gpu,utilization.memory,clocks.sm,clocks.mem")


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, environment=None, timeout=600):
    result = subprocess.run(
        tuple(str(value) for value in command),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "command failed: "
            f"command={tuple(str(value) for value in command)!r}, "
            f"returncode={result.returncode}, stdout={result.stdout!r}, "
            f"stderr={result.stderr!r}")
    return result


def _git_identity():
    commit = _run(
        ("git", "rev-parse", "HEAD"), timeout=30).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("git HEAD is not a full commit identity")
    for path in (SOURCE_PATH, RUNNER_PATH, REPORT_PATH):
        _run(("git", "ls-files", "--error-unmatch", path), timeout=30)
    status = _run(
        ("git", "status", "--short", "--untracked-files=no", "--",
         SOURCE_PATH, RUNNER_PATH, REPORT_PATH),
        timeout=30,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "benchmark source, runner, and reporter must be committed before "
            f"capture: {status}")
    return commit


def _parse_device_row(
        text, physical_device, *, require_zero_utilization=True):
    fields = tuple(part.strip() for part in text.strip().split(","))
    if len(fields) != 10:
        raise ValueError(
            "nvidia-smi device query did not return exactly ten fields")
    integers = {}
    for name, value in (
            ("memory_used_mib", fields[5]),
            ("gpu_utilization_percent", fields[6]),
            ("memory_utilization_percent", fields[7]),
            ("sm_clock_mhz", fields[8]),
            ("memory_clock_mhz", fields[9])):
        integers[name] = int(value)
        if integers[name] < 0:
            raise ValueError(f"nvidia-smi returned negative {name}")
    if fields[1] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(
            f"device {physical_device} is {fields[1]!r}, expected "
            f"{EXPECTED_DEVICE_NAME!r}")
    if require_zero_utilization and integers["gpu_utilization_percent"] != 0:
        raise RuntimeError(
            f"device {physical_device} is not idle: GPU utilization is "
            f"{integers['gpu_utilization_percent']}%")
    return {
        "physical_device": physical_device,
        "device_uuid": fields[0],
        "device_name": fields[1],
        "pci_bus_id": fields[2],
        "driver_version": fields[3],
        "pstate": fields[4],
        **integers,
    }


def _query_device(physical_device, *, require_zero_utilization=True):
    result = _run((
        "nvidia-smi", "-i", str(physical_device),
        f"--query-gpu={DEVICE_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    ), timeout=30)
    rows = tuple(line for line in result.stdout.splitlines() if line.strip())
    if len(rows) != 1:
        raise RuntimeError(
            "nvidia-smi must return exactly one selected device row")
    return _parse_device_row(
        rows[0], physical_device,
        require_zero_utilization=require_zero_utilization)


def _require_no_compute_processes(device_uuid):
    result = _run((
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    ), timeout=30)
    rows = tuple(line.strip() for line in result.stdout.splitlines()
                 if line.strip())
    active = []
    for row in rows:
        if row.startswith("No running processes found"):
            continue
        fields = tuple(part.strip() for part in row.split(","))
        if len(fields) != 2:
            raise ValueError(
                "nvidia-smi compute-process row has an unknown schema")
        if fields[1] == device_uuid:
            active.append(fields[0])
    if active:
        raise RuntimeError(
            f"selected device has active compute processes: {active}")


def _cells():
    return tuple(
        (threads, (1 if multiplier == 0 else EXPECTED_SM_COUNT * multiplier))
        for threads in BLOCK_THREADS
        for multiplier in GRID_MULTIPLIERS
    )


def _schedule(trials):
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")
    cells = _cells()
    for trial in range(trials):
        offset = (trial * ROTATION_STRIDE) % len(cells)
        for position in range(len(cells)):
            threads, grid = cells[(position + offset) % len(cells)]
            yield trial, position, threads, grid


def _parse_binary_rows(
        text, expected_threads, expected_grid, samples, timer_domain):
    if not isinstance(text, str):
        raise TypeError("benchmark stdout must be text")
    if timer_domain == "globaltimer_ns":
        expected_fields = GLOBAL_BINARY_FIELDS
        duration_fields = ("block_median_ns", "grid_span_ns")
    elif timer_domain == "clock64_sm_cycles":
        expected_fields = CYCLE_BINARY_FIELDS
        duration_fields = ("block_median_cycles", "sm_local_span_cycles")
    else:
        raise ValueError(f"unsupported timer domain: {timer_domain!r}")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or tuple(reader.fieldnames) != expected_fields:
        raise ValueError("benchmark stdout has an unknown CSV schema")
    rows = tuple(reader)
    if len(rows) != samples:
        raise ValueError(
            f"benchmark returned {len(rows)} rows, expected {samples}")
    observed_samples = set()
    for row in rows:
        if None in row.values():
            raise ValueError("benchmark row has a missing CSV value")
        integers = {
            name: int(row[name]) for name in (
                *COMMON_BINARY_FIELDS, *duration_fields)
        }
        if integers["sample"] in observed_samples:
            raise ValueError("benchmark sample identifiers must be unique")
        observed_samples.add(integers["sample"])
        if integers["device_ordinal"] != 0:
            raise ValueError("CUDA_VISIBLE_DEVICES must expose one ordinal-0 GPU")
        if integers["sm_count"] != EXPECTED_SM_COUNT:
            raise ValueError("benchmark did not execute on a 108-SM A800")
        if integers["threads"] != expected_threads:
            raise ValueError("benchmark threads disagree with the scheduled cell")
        if integers["grid"] != expected_grid:
            raise ValueError("benchmark grid disagrees with the scheduled cell")
        if timer_domain == "clock64_sm_cycles":
            if any(integers[name] <= 0 for name in duration_fields):
                raise ValueError("clock64 durations must be positive")
        elif timer_domain == "globaltimer_ns":
            if any(integers[name] < 0 for name in duration_fields):
                raise ValueError("globaltimer durations must be non-negative")
        else:
            raise AssertionError(f"unhandled timer domain: {timer_domain!r}")
        checksum = float(row["validation_checksum"])
        if not math.isfinite(checksum) or checksum <= 0:
            raise ValueError("benchmark checksum must be finite and positive")
        if row["validation_status"] != "pass":
            raise ValueError("benchmark validation did not pass")
    if observed_samples != set(range(samples)):
        raise ValueError("benchmark sample identifiers are incomplete")
    return rows


def _require_descendant(path, root, name):
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} resolves outside cuda_root") from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{name} must resolve to an executable file")
    return resolved


def _resolve_toolchain(cuda_root):
    root = cuda_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("cuda_root must be a directory")
    nvcc = _require_descendant(root / "bin" / "nvcc", root, "nvcc")
    cuobjdump = _require_descendant(
        root / "bin" / "cuobjdump", root, "cuobjdump")
    host_cxx = _require_descendant(
        root / "bin" / "x86_64-conda-linux-gnu-g++", root, "host_cxx")
    cub_version = (
        root / "targets" / "x86_64-linux" / "include" / "cub"
        / "version.cuh").resolve(strict=True)
    cub_text = cub_version.read_text(encoding="utf-8")
    match = re.search(r"^#define CUB_VERSION (\d+)\b", cub_text, re.MULTILINE)
    if match is None or int(match.group(1)) != EXPECTED_CUB_VERSION:
        raise RuntimeError(
            f"toolchain must provide CUB {EXPECTED_CUB_VERSION}")
    nvcc_version = _run((nvcc, "--version"), timeout=30).stdout
    if (f"release {EXPECTED_CUDA_RELEASE}" not in nvcc_version
            or EXPECTED_NVCC_BUILD not in nvcc_version):
        raise RuntimeError(
            "nvcc identity does not match the pinned CUDA 12.8.93 compiler")
    cuobjdump_version = _run((cuobjdump, "--version"), timeout=30).stdout
    if f"release {EXPECTED_CUDA_RELEASE}" not in cuobjdump_version:
        raise RuntimeError("cuobjdump is not from CUDA 12.8")
    return {
        "root": str(root),
        "nvcc": str(nvcc),
        "nvcc_sha256": _sha256(nvcc),
        "nvcc_version": nvcc_version.strip(),
        "cuobjdump": str(cuobjdump),
        "cuobjdump_sha256": _sha256(cuobjdump),
        "cuobjdump_version": cuobjdump_version.strip(),
        "host_cxx": str(host_cxx),
        "host_cxx_sha256": _sha256(host_cxx),
        "cub_version_header": str(cub_version),
        "cub_version_header_sha256": _sha256(cub_version),
        "cub_version": EXPECTED_CUB_VERSION,
    }


def _read_json_object(path, name):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _checksum_manifest(path):
    entries = {}
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise ValueError(
                f"invalid production checksum line {line_number}")
        name = Path(fields[1].strip()).name
        if name in entries:
            raise ValueError(f"duplicate production checksum basename: {name}")
        entries[name] = fields[0]
    return entries


def _validate_production_artifact(artifact_dir, production_python):
    root = artifact_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("production_artifact_dir must be a directory")
    report_path = root / "report.json"
    sass_path = root / "add_rmsnorm_bf16_vec8_sm80.sass"
    cfg_path = root / "add_rmsnorm_bf16_vec8_sm80_cfg.dot"
    symbols_path = root / "sm80_symbols.txt"
    checksums_path = root / "sha256.txt"
    for path in (
            report_path, sass_path, cfg_path, symbols_path, checksums_path):
        if not path.is_file():
            raise ValueError(
                f"production artifact is missing required file: {path.name}")
    checksums = _checksum_manifest(checksums_path)
    for path in (sass_path, cfg_path, symbols_path):
        if path.name not in checksums or _sha256(path) != checksums[path.name]:
            raise ValueError(
                f"production artifact checksum mismatch: {path.name}")
    report = _read_json_object(report_path, "production report")
    if report["artifact_kind"] != "static_sass_cfg_lower_bound":
        raise ValueError("production artifact kind is unsupported")
    if report["hardware_id"] != "nvidia_a800" or report["duration_consumed"]:
        raise ValueError("production artifact hardware/duration contract is invalid")
    source_binary = report["source_binary"]
    kernel = report["kernel"]
    domain = report["tail_free_domain"]
    if source_binary["sha256"] != EXPECTED_PRODUCTION_BINARY_SHA256:
        raise ValueError("production report extension hash is not pinned")
    if kernel["symbol"] != EXPECTED_PRODUCTION_KERNEL:
        raise ValueError("production report kernel symbol is not pinned")
    if kernel["sass_sha256"] != _sha256(sass_path):
        raise ValueError("production report SASS hash mismatch")
    if kernel["basic_block_cfg_sha256"] != _sha256(cfg_path):
        raise ValueError("production report CFG hash mismatch")
    if (domain["dtype"] != "bfloat16"
            or domain["vector_width_elements"] != 8
            or domain["hidden_multiple"] != 256
            or tuple(domain["supported_block_sizes"]) != BLOCK_THREADS):
        raise ValueError("production report tail-free launch domain changed")
    symbol_text = symbols_path.read_text(encoding="utf-8")
    if "CUB_200700" not in symbol_text or EXPECTED_PRODUCTION_KERNEL not in symbol_text:
        raise ValueError("production symbols do not prove CUB/kernel identity")
    python_path = production_python.resolve(strict=True)
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ValueError("production_python must be an executable file")
    probe = _run((
        python_path, "-c",
        "import importlib.util; s=importlib.util.find_spec('vllm._C'); "
        "assert s is not None and s.origin is not None; print(s.origin)",
    ), timeout=60)
    lines = tuple(line.strip() for line in probe.stdout.splitlines()
                  if line.strip())
    if len(lines) != 1:
        raise ValueError("production Python returned an ambiguous extension path")
    extension_path = Path(lines[0]).resolve(strict=True)
    if _sha256(extension_path) != EXPECTED_PRODUCTION_BINARY_SHA256:
        raise ValueError("live production extension hash differs from the artifact")
    return {
        "root": str(root),
        "report_sha256": _sha256(report_path),
        "sass_path": str(sass_path),
        "sass_sha256": _sha256(sass_path),
        "cfg_sha256": _sha256(cfg_path),
        "symbols_sha256": _sha256(symbols_path),
        "kernel_symbol": EXPECTED_PRODUCTION_KERNEL,
        "production_python": str(python_path),
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
        "cub_version": EXPECTED_CUB_VERSION,
    }


def _sass_instructions(text):
    instructions = []
    for match in SASS_INSTRUCTION_PATTERN.finditer(text):
        predicate = match.group(2)
        if predicate is None:
            predicate = ""
        else:
            predicate = predicate.strip()
        instructions.append({
            "offset": int(match.group(1), 16),
            "predicate": predicate,
            "opcode": match.group(3),
            "operands": match.group(4).strip(),
        })
    if not instructions:
        raise ValueError("SASS contains no decoded instructions")
    return tuple(instructions)


def _ordered_subsequence(values, expected):
    position = 0
    for value in values:
        if position < len(expected) and value == expected[position]:
            position += 1
    return position == len(expected)


def _resource_usage(text):
    matches = re.findall(
        r"Function\s+[^\n]+:\s*\n\s+REG:(\d+)\s+STACK:\d+\s+"
        r"SHARED:(\d+)\b",
        text,
    )
    if len(matches) != 1:
        raise ValueError(
            "cuobjdump resource usage must contain one device function")
    registers_per_thread, shared_memory_bytes = (
        int(value) for value in matches[0])
    if registers_per_thread <= 0 or shared_memory_bytes <= 0:
        raise ValueError("candidate resource usage must be positive")
    resident_ctas = {}
    for threads in BLOCK_THREADS:
        register_limit = (
            SM_REGISTER_CAPACITY // (registers_per_thread * threads))
        shared_limit = SM_SHARED_MEMORY_BYTES // shared_memory_bytes
        thread_limit = SM_THREAD_CAPACITY // threads
        resident = min(
            SM_BLOCK_CAPACITY, register_limit, shared_limit, thread_limit)
        if resident <= 1:
            raise ValueError(
                "candidate must admit at least two resident CTAs per SM")
        resident_ctas[str(threads)] = resident
    return {
        "registers_per_thread": registers_per_thread,
        "shared_memory_bytes_per_cta": shared_memory_bytes,
        "sm_register_capacity": SM_REGISTER_CAPACITY,
        "sm_shared_memory_bytes": SM_SHARED_MEMORY_BYTES,
        "sm_thread_capacity": SM_THREAD_CAPACITY,
        "sm_block_capacity": SM_BLOCK_CAPACITY,
        "resident_ctas_by_threads": resident_ctas,
    }


def _static_equivalence(candidate_sass, production_sass, timer_domain):
    candidate = _sass_instructions(candidate_sass)
    production = _sass_instructions(production_sass)
    if timer_domain == "globaltimer_ns":
        timer_token = "SR_GLOBALTIMER"
        forbidden_timer_token = "SR_CLOCK"
    elif timer_domain == "clock64_sm_cycles":
        timer_token = "SR_CLOCK"
        forbidden_timer_token = "SR_GLOBALTIMER"
    else:
        raise ValueError(f"unsupported timer domain: {timer_domain!r}")
    timer_indices = tuple(
        index for index, instruction in enumerate(candidate)
        if timer_token in instruction["operands"])
    if len(timer_indices) != 2:
        raise RuntimeError(
            f"candidate must contain exactly two {timer_domain} instructions")
    first_timer, second_timer = timer_indices
    if first_timer >= second_timer:
        raise RuntimeError("candidate timer ordering is invalid")
    region = candidate[first_timer + 1:second_timer]
    if not region or not region[-1]["opcode"].startswith("BAR.SYNC"):
        raise RuntimeError(
            "second timer must immediately follow the publication barrier")
    forbidden_tokens = (forbidden_timer_token, "SR_VIRTUAL_FUNCTION")
    if any(token in instruction["operands"] for instruction in region
           for token in forbidden_tokens):
        raise RuntimeError("candidate timed region contains another timer domain")
    forbidden_prefixes = (
        "LDG", "STG", "ATOM", "RED", "SULD", "SUST", "LDGSTS")
    if any(instruction["opcode"].startswith(forbidden_prefixes)
           for instruction in region):
        raise RuntimeError(
            "candidate timed region contains global-memory traffic")
    candidate_shuffles = tuple(
        instruction for instruction in region
        if instruction["opcode"] == "SHFL.DOWN")
    production_shuffles = tuple(
        instruction for instruction in production
        if instruction["opcode"] == "SHFL.DOWN")
    expected_shifts = ("0x1", "0x2", "0x4", "0x8", "0x10") * 2
    for name, shuffles in (
            ("candidate", candidate_shuffles),
            ("production", production_shuffles)):
        if len(shuffles) != len(expected_shifts):
            raise RuntimeError(
                f"{name} reduction does not contain two five-step shuffle trees")
        shifts = tuple(
            next(token for token in instruction["operands"].split(", ")
                 if token in expected_shifts)
            for instruction in shuffles)
        if shifts != expected_shifts:
            raise RuntimeError(f"{name} reduction shuffle tree changed")
    fast_normalization = (
        "MUFU.RCP", "FCHK", "FFMA", "FFMA", "FFMA", "FFMA", "FFMA",
        "BRA", "BSYNC", "FADD", "FSETP.GEU.AND", "FMUL", "MUFU.RSQ",
        "FMUL", "STS", "BSYNC", "BAR.SYNC.DEFER_BLOCKING",
    )
    for name, instructions in (("candidate", region), ("production", production)):
        opcodes = tuple(instruction["opcode"] for instruction in instructions)
        if not _ordered_subsequence(opcodes, fast_normalization):
            raise RuntimeError(
                f"{name} lacks the source-closed normal fast-path sequence")
    candidate_barriers = sum(
        instruction["opcode"].startswith("BAR.SYNC")
        for instruction in region)
    production_barriers = sum(
        instruction["opcode"].startswith("BAR.SYNC")
        for instruction in production)
    if candidate_barriers != 3 or production_barriers < 3:
        raise RuntimeError("reduction/publication barrier structure changed")
    return {
        "schema": "a800_add_rmsnorm_reduction_static_equivalence_v1",
        "status": "pass",
        "production_kernel": EXPECTED_PRODUCTION_KERNEL,
        "cub_version": EXPECTED_CUB_VERSION,
        "timer_domain": timer_domain,
        "candidate_timer_instruction_count": len(timer_indices),
        "candidate_timed_instruction_count": len(region),
        "candidate_shuffle_down_count": len(candidate_shuffles),
        "candidate_barrier_count": candidate_barriers,
        "production_shuffle_down_count": len(production_shuffles),
        "production_barrier_count": production_barriers,
        "global_memory_in_timed_region": False,
        "other_timer_domain_in_timed_region": False,
        "normal_fast_path_closed": True,
        "limitations": [
            "the production FCHK slow path is retained but is outside the "
            "normal-value timing validity domain",
            "the proof covers BF16 vec8 Add-RMSNorm with block 256 or 1024",
        ],
    }


def _compile_timer_binary(
        output_dir, toolchain, production_sass, timer_domain):
    nvcc = Path(toolchain["nvcc"])
    cuobjdump = Path(toolchain["cuobjdump"])
    host_cxx = Path(toolchain["host_cxx"])
    include_root = (
        Path(toolchain["root"]) / "targets" / "x86_64-linux" / "include")
    if timer_domain == "globaltimer_ns":
        suffix = "globaltimer"
        timer_flags = ()
    elif timer_domain == "clock64_sm_cycles":
        suffix = "clock64"
        timer_flags = ("-DA800_REDUCTION_CLOCK64=1",)
    else:
        raise ValueError(f"unsupported timer domain: {timer_domain!r}")
    binary = output_dir / f"block_reduce_latency_{suffix}"
    compile_command = (
        nvcc, "-O3", "-lineinfo", "-std=c++17",
        "--generate-code", "arch=compute_80,code=sm_80",
        "-ccbin", host_cxx, "-I", include_root,
        *timer_flags, SOURCE_PATH, "-o", binary,
    )
    _run(compile_command, timeout=600)
    sass_path = output_dir / f"block_reduce_latency_{suffix}.sass"
    sass = _run((cuobjdump, "--dump-sass", binary), timeout=120).stdout
    sass_arches = tuple(re.findall(r"^arch = (sm_\d+)$", sass, re.MULTILINE))
    if not sass_arches or set(sass_arches) != {"sm_80"}:
        raise RuntimeError("candidate binary must contain only sm_80 images")
    if sass.count("Function : ") != 1:
        raise RuntimeError("candidate SASS must contain exactly one device function")
    ptx = _run((cuobjdump, "--dump-ptx", binary), timeout=120).stdout
    if re.search(r"^\.version\s", ptx, re.MULTILINE):
        raise RuntimeError("candidate binary must not retain a PTX fallback")
    sass_path.write_text(sass, encoding="utf-8")
    resource_usage = _resource_usage(
        _run((cuobjdump, "--dump-resource-usage", binary),
             timeout=120).stdout)
    proof = _static_equivalence(sass, production_sass, timer_domain)
    proof.update({
        "candidate_binary_sha256": _sha256(binary),
        "candidate_sass_sha256": _sha256(sass_path),
        "toolchain_nvcc_sha256": toolchain["nvcc_sha256"],
        "toolchain_cub_header_sha256": toolchain["cub_version_header_sha256"],
    })
    return {
        "binary": binary,
        "sass_path": sass_path,
        "compile_command": tuple(str(value) for value in compile_command),
        "proof": proof,
        "resource_usage": resource_usage,
    }


def _build(output_dir, toolchain, production_artifact):
    production_sass_path = Path(production_artifact["sass_path"])
    production_sass = production_sass_path.read_text(encoding="utf-8")
    globaltimer = _compile_timer_binary(
        output_dir, toolchain, production_sass, "globaltimer_ns")
    clock64 = _compile_timer_binary(
        output_dir, toolchain, production_sass, "clock64_sm_cycles")
    if globaltimer["resource_usage"] != clock64["resource_usage"]:
        raise RuntimeError("timer binaries have different static resources")
    proof = {
        "schema": "a800_add_rmsnorm_reduction_static_equivalence_v2",
        "status": "pass",
        "production_sass_sha256": production_artifact["sass_sha256"],
        "resource_usage": globaltimer["resource_usage"],
        "globaltimer": globaltimer["proof"],
        "clock64": clock64["proof"],
    }
    proof_path = output_dir / "static_equivalence.json"
    proof_path.write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "globaltimer_binary": globaltimer["binary"],
        "globaltimer_sass_path": globaltimer["sass_path"],
        "globaltimer_compile_command": globaltimer["compile_command"],
        "clock64_binary": clock64["binary"],
        "clock64_sass_path": clock64["sass_path"],
        "clock64_compile_command": clock64["compile_command"],
        "proof_path": proof_path,
    }


def _static_manifest(
        run_id, baseline, toolchain, production_artifact, build):
    return {
        "schema": "a800_block_reduce_static_capture_v1",
        "run_id": run_id,
        "baseline_commit": baseline,
        "block_threads": list(BLOCK_THREADS),
        "mechanism": "source_bound_add_rmsnorm_reduction_and_publication",
        "production_duration_consumed": False,
        "source_sha256": _sha256(SOURCE_PATH),
        "runner_sha256": _sha256(RUNNER_PATH),
        "reporter_sha256": _sha256(REPORT_PATH),
        "globaltimer_binary_sha256": _sha256(build["globaltimer_binary"]),
        "globaltimer_sass_sha256": _sha256(
            build["globaltimer_sass_path"]),
        "clock64_binary_sha256": _sha256(build["clock64_binary"]),
        "clock64_sass_sha256": _sha256(build["clock64_sass_path"]),
        "static_equivalence_sha256": _sha256(build["proof_path"]),
        "globaltimer_compile_command": list(
            build["globaltimer_compile_command"]),
        "clock64_compile_command": list(build["clock64_compile_command"]),
        "toolchain": toolchain,
        "production_artifact": production_artifact,
    }


def _write_manifest(path, manifest):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--production-python", type=Path, required=True)
    parser.add_argument("--production-artifact-dir", type=Path, required=True)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument(
        "--samples", type=_positive_integer, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--trials", type=_positive_integer, default=DEFAULT_TRIALS)
    args = parser.parse_args()
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run_id contains unsupported characters")
    if args.device_id is not None and args.device_id < 0:
        raise ValueError("device_id must be non-negative")
    if not args.static_only and args.device_id is None:
        raise ValueError("device_id is required unless --static-only is used")

    baseline = _git_identity()
    toolchain = _resolve_toolchain(args.cuda_root)
    production_artifact = _validate_production_artifact(
        args.production_artifact_dir, args.production_python)
    output_dir = (OUTPUT_ROOT / args.run_id).resolve()
    if output_dir.parent != OUTPUT_ROOT.resolve():
        raise ValueError("run_id escaped the A800 profiler output root")
    output_dir.mkdir(parents=True, exist_ok=False)
    build = _build(output_dir, toolchain, production_artifact)
    static_manifest = _static_manifest(
        args.run_id, baseline, toolchain, production_artifact, build)
    if args.static_only:
        _write_manifest(output_dir / "manifest.json", static_manifest)
        print(f"static_capture={output_dir}")
        return 0

    preflight = _query_device(args.device_id)
    _require_no_compute_processes(preflight["device_uuid"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    # An idle A800 normally reports its 210 MHz floor immediately before the
    # first process.  Run one discarded mechanism cell after ownership and
    # idle checks so every retained row is bracketed in the steady P0 clock
    # state; this is not a production/evaluation duration observation.
    conditioning = _run(
        (build["clock64_binary"], BLOCK_THREADS[0], 1, 1),
        environment=environment, timeout=180)
    _parse_binary_rows(
        conditioning.stdout, BLOCK_THREADS[0], expected_grid=1, samples=1,
        timer_domain="clock64_sm_cycles")
    conditioned_preflight = _query_device(
        args.device_id, require_zero_utilization=False)
    _require_no_compute_processes(conditioned_preflight["device_uuid"])
    if (preflight["device_uuid"] != conditioned_preflight["device_uuid"]
            or preflight["pci_bus_id"]
            != conditioned_preflight["pci_bus_id"]):
        raise RuntimeError("selected device identity changed during conditioning")
    raw_path = output_dir / "raw.csv"
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for trial, position, threads, grid in _schedule(args.trials):
            before = _query_device(
                args.device_id, require_zero_utilization=False)
            _require_no_compute_processes(before["device_uuid"])
            cycle_result = _run(
                (build["clock64_binary"], threads, grid, args.samples),
                environment=environment, timeout=180)
            cycle_rows = _parse_binary_rows(
                cycle_result.stdout, threads, grid, args.samples,
                timer_domain="clock64_sm_cycles")
            globaltimer_result = _run(
                (build["globaltimer_binary"], threads, grid, args.samples),
                environment=environment, timeout=180)
            globaltimer_rows = _parse_binary_rows(
                globaltimer_result.stdout, threads, grid, args.samples,
                timer_domain="globaltimer_ns")
            after = _query_device(
                args.device_id, require_zero_utilization=False)
            _require_no_compute_processes(after["device_uuid"])
            if (before["device_uuid"] != after["device_uuid"]
                    or before["pci_bus_id"] != after["pci_bus_id"]):
                raise RuntimeError("selected device identity changed during capture")
            if (before["sm_clock_mhz"] != after["sm_clock_mhz"]
                    or before["memory_clock_mhz"]
                    != after["memory_clock_mhz"]):
                raise RuntimeError("selected device clocks changed during a cell")
            for cycle_row, globaltimer_row in zip(
                    cycle_rows, globaltimer_rows, strict=True):
                for field in COMMON_BINARY_FIELDS:
                    if cycle_row[field] != globaltimer_row[field]:
                        raise RuntimeError(
                            "clock64/globaltimer sample identities disagree")
                if not math.isclose(
                        float(cycle_row["validation_checksum"]),
                        float(globaltimer_row["validation_checksum"]),
                        rel_tol=1.0e-7, abs_tol=1.0e-7):
                    raise RuntimeError(
                        "clock64/globaltimer validation checksums disagree")
                writer.writerow({
                    "run_id": args.run_id,
                    "trial": trial,
                    "position": position,
                    "physical_device": args.device_id,
                    "device_uuid": before["device_uuid"],
                    "pci_bus_id": before["pci_bus_id"],
                    "driver_version": before["driver_version"],
                    "pstate": before["pstate"],
                    "sm_clock_before_mhz": before["sm_clock_mhz"],
                    "sm_clock_after_mhz": after["sm_clock_mhz"],
                    "memory_clock_before_mhz": before["memory_clock_mhz"],
                    "memory_clock_after_mhz": after["memory_clock_mhz"],
                    **{
                        field: cycle_row[field]
                        for field in COMMON_BINARY_FIELDS
                    },
                    "block_median_cycles": cycle_row[
                        "block_median_cycles"],
                    "sm_local_span_cycles": cycle_row[
                        "sm_local_span_cycles"],
                    "grid_span_ns": globaltimer_row["grid_span_ns"],
                    "validation_checksum": cycle_row[
                        "validation_checksum"],
                    "validation_status": cycle_row["validation_status"],
                })
            stream.flush()
            print(
                f"trial={trial} position={position} threads={threads} "
                f"grid={grid} samples={args.samples}",
                flush=True)

    postflight = _query_device(
        args.device_id, require_zero_utilization=False)
    _require_no_compute_processes(postflight["device_uuid"])
    if (preflight["device_uuid"] != postflight["device_uuid"]
            or preflight["pci_bus_id"] != postflight["pci_bus_id"]):
        raise RuntimeError("selected device identity changed after capture")
    manifest = {
        **static_manifest,
        "schema": "a800_block_reduce_latency_capture_v3",
        "physical_device": args.device_id,
        "device": preflight,
        "conditioned_preflight": conditioned_preflight,
        "clock_conditioning": {
            "threads": BLOCK_THREADS[0],
            "grid": 1,
            "retained_samples": 0,
        },
        "postflight": postflight,
        "samples_per_cell": args.samples,
        "trials": args.trials,
        "grid_multipliers": list(GRID_MULTIPLIERS),
        "sm_count": EXPECTED_SM_COUNT,
        "timer_domains": {
            "block_median_cycles": "clock64_sm_cycles",
            "sm_local_span_cycles": "clock64_same_smid_cycles",
            "grid_span_ns": "globaltimer_ns",
        },
        "raw_sha256": _sha256(raw_path),
    }
    _write_manifest(output_dir / "manifest.json", manifest)
    print(f"capture={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
