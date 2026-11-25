#!/usr/bin/env python3
"""Capture the independent block-256 Add-RMSNorm source-round probe."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
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
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output/microbench/nvidia_a800"
SOURCE_PATH = Path(__file__).with_name("add_rmsnorm_round_service.cu")
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
DEFAULT_CUOBJDUMP = Path(shutil.which("cuobjdump") or "cuobjdump")
DEFAULT_NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
DEFAULT_PRODUCTION_BINARY = (
    Path(os.environ["MIDLAYER_A800_VLLM_SOURCE"]) / "_C.abi3.so"
)
EXPECTED_NVCC_SHA256 = (
    "6b05c8048bff95eb892bcd53cc55fd4fe231aa27bb9dd7289661e746408d8b7d")
EXPECTED_CUOBJDUMP_SHA256 = (
    "ad1d0f0699f46603416eb58e7b9fdf9a293e95f9312c9df7fd6fda56a6c30d41")
EXPECTED_PRODUCTION_BINARY_SHA256 = (
    "984de21ce38cc8083d207663a380ca75acc6ad3dc230858092f59e363c4eebd5")
EXPECTED_PRODUCTION_CUBIN = "_C.abi3.59.sm_80.cubin"
REJECTED_ACTIVATION_CUBIN = "_C.abi3.52.sm_80.cubin"
EXPECTED_PRODUCTION_CUBIN_SHA256 = (
    "166d5a3ad91de17854f489ae49038b1a39179121fe744edc5d5449a5ea225700")
EXPECTED_PRODUCTION_SYMBOL = (
    "_ZN4vllm25fused_add_rms_norm_kernelIN3c108BFloat16ELi8EEE"
    "NSt9enable_ifIXaagtT0_Li0Esr4vllm12_typeConvertIT_EE6existsEvE4typeE"
    "PS4_lS7_PKS4_fii")
EXPECTED_PRODUCTION_RESOURCES = {
    "registers_per_thread": 32,
    "static_shared_bytes": 168,
    "local_bytes": 0,
}
EXPECTED_PRODUCTION_SASS_SHA256 = (
    "b1f9aaf1a8a221567499e1214b053592c62ff594815742bdc881b912e971cf4b")
EXPECTED_PRODUCTION_INSTRUCTION_COUNT = 432
EXPECTED_DEVICE = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
EXPECTED_THREADS = 256
EXPECTED_RESIDENT_CTAS = 8
EXPECTED_ROUNDS = 512
MINIMUM_TRIALS = 3
MAXIMUM_TRIALS = 11
MAXIMUM_IDLE_MEMORY_MIB = 8
FUNCTIONS = ("add_square", "scale_multiply")
CELLS = {
    "first": EXPECTED_SM_COUNT,
    "resident": EXPECTED_RESIDENT_CTAS * EXPECTED_SM_COUNT,
}
RAW_FIELDS = (
    "trial", "function", "cell", "grid_blocks", "block_threads",
    "rounds", "resident_cta_limit", "active_sms",
    "minimum_ctas_per_sm", "maximum_ctas_per_sm",
    "maximum_sm_span_ns", "median_block_ns", "checksum",
)
PROVENANCE_FIELDS = (
    "schema", "owner", "runner_pid", "native_pid", "started_at_utc",
    "completed_at_utc", "baseline_commit", "command", "native_command",
    "device", "postflight_device", "preflight_cleanup",
    "postflight_cleanup", "trials", "tool_identities", "source_sha256",
    "runner_sha256", "probe_binary_sha256", "probe_resource_dump_sha256",
    "probe_sass_dump_sha256", "probe_resources", "probe_sass_contract",
    "production_artifact", "candidate_sha256", "raw_sha256",
    "forward_eligible", "production_or_evaluation_duration_consumed",
)
SYMBOL_PATTERN = re.compile(r"Function(?:\s*:\s*|\s+)(\S+?):?$")
REGISTER_PATTERN = re.compile(r"REG:(\d+)")
SHARED_PATTERN = re.compile(r"SHARED:(\d+)")
LOCAL_PATTERN = re.compile(r"LOCAL:(\d+)")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd, stdout=None, stderr=None, capture_output=False):
    return subprocess.run(
        command, cwd=cwd, check=True, text=True, stdout=stdout,
        stderr=stderr, capture_output=capture_output, timeout=600)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _tool_identity(path, version_args, expected_sha256=None):
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"required executable is unavailable: {resolved}")
    digest = _sha256(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"tool identity changed for {resolved}: {digest} != "
            f"{expected_sha256}")
    version = _run(
        [str(resolved), *version_args], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout.strip()
    return {"path": str(resolved), "sha256": digest, "version": version}


def _validate_production_cubin_name(name):
    if name == REJECTED_ACTIVATION_CUBIN:
        raise ValueError(
            "activation cubin .52 is a decoy and cannot prove Add-RMSNorm")
    if name != EXPECTED_PRODUCTION_CUBIN:
        raise ValueError(
            f"production Add-RMSNorm requires {EXPECTED_PRODUCTION_CUBIN}, "
            f"got {name}")
    return name


def _parse_elf_listing(text):
    names = tuple(re.findall(r"ELF file\s+\d+:\s+(\S+)", text))
    if names.count(EXPECTED_PRODUCTION_CUBIN) != 1:
        raise ValueError(
            "production binary must contain exactly one Add-RMSNorm .59 cubin")
    if REJECTED_ACTIVATION_CUBIN not in names:
        raise ValueError(
            "production binary no longer contains the expected activation "
            "decoy used by the fail-loud audit")
    return names


def _parse_production_resources(text, cubin_name):
    _validate_production_cubin_name(cubin_name)
    resources = {}
    current = None
    for line in text.splitlines():
        match = SYMBOL_PATTERN.search(line)
        if match:
            current = match.group(1)
            continue
        if current != EXPECTED_PRODUCTION_SYMBOL:
            continue
        registers = REGISTER_PATTERN.search(line)
        shared = SHARED_PATTERN.search(line)
        local = LOCAL_PATTERN.search(line)
        if registers and shared and local:
            if current in resources:
                raise ValueError("production Add-RMSNorm symbol is duplicated")
            resources[current] = {
                "registers_per_thread": int(registers.group(1)),
                "static_shared_bytes": int(shared.group(1)),
                "local_bytes": int(local.group(1)),
            }
    if set(resources) != {EXPECTED_PRODUCTION_SYMBOL}:
        raise ValueError(
            "production .59 cubin lacks the exact BF16 vec8 Add-RMSNorm symbol")
    resource = resources[EXPECTED_PRODUCTION_SYMBOL]
    if resource != EXPECTED_PRODUCTION_RESOURCES:
        raise ValueError(
            f"production Add-RMSNorm resources changed: {resource}")
    return resource


def _validate_production_sass(
        text, cubin_name, expected_sha256=EXPECTED_PRODUCTION_SASS_SHA256):
    _validate_production_cubin_name(cubin_name)
    headers = re.findall(r"^\s*Function\s*:\s*(\S+)\s*$", text, re.MULTILINE)
    if headers != [EXPECTED_PRODUCTION_SYMBOL]:
        raise ValueError(
            "production SASS must contain exactly the BF16 vec8 Add-RMSNorm "
            "symbol")
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"production Add-RMSNorm function SASS changed: {digest}")
    opcodes = re.findall(
        r"/\*[0-9a-f]+\*/\s+(?:@[!P0-9]+\s+)?([A-Z][A-Z0-9_.]*)",
        text)
    if len(opcodes) != EXPECTED_PRODUCTION_INSTRUCTION_COUNT:
        raise ValueError(
            "production Add-RMSNorm instruction count changed: "
            f"{len(opcodes)}")
    required = {
        "LDG.E.128", "LDG.E.U16.CONSTANT", "STG.E.128", "HFMA2.BF16_V2",
        "F2FP.BF16.PACK_AB", "MUFU.RSQ", "BAR.SYNC.DEFER_BLOCKING",
    }
    missing = sorted(required - set(opcodes))
    if missing:
        raise ValueError(
            f"production Add-RMSNorm SASS lacks required opcodes: {missing}")
    return {
        "symbol": EXPECTED_PRODUCTION_SYMBOL,
        "sha256": digest,
        "instruction_count": len(opcodes),
        "required_opcodes": sorted(required),
    }


def _audit_production_artifact(production_binary, cuobjdump, build_dir):
    binary = production_binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    binary_sha256 = _sha256(binary)
    if binary_sha256 != EXPECTED_PRODUCTION_BINARY_SHA256:
        raise ValueError(
            f"production vLLM binary changed: {binary_sha256}")
    listing = _run(
        [str(cuobjdump), "--list-elf", str(binary)],
        cwd=build_dir, capture_output=True)
    (build_dir / "production_elf_listing.txt").write_text(listing.stdout)
    _parse_elf_listing(listing.stdout)
    production_dir = build_dir / "production_cubin"
    production_dir.mkdir()
    _run([
        str(cuobjdump), "--extract-elf", EXPECTED_PRODUCTION_CUBIN,
        str(binary)], cwd=production_dir, capture_output=True)
    cubin = production_dir / EXPECTED_PRODUCTION_CUBIN
    if not cubin.is_file():
        raise FileNotFoundError(
            f"cuobjdump did not extract {EXPECTED_PRODUCTION_CUBIN}")
    cubin_sha256 = _sha256(cubin)
    if cubin_sha256 != EXPECTED_PRODUCTION_CUBIN_SHA256:
        raise ValueError(
            f"production Add-RMSNorm cubin changed: {cubin_sha256}")
    resources_result = _run([
        str(cuobjdump), "--dump-resource-usage", str(cubin)],
        cwd=build_dir, capture_output=True)
    resources_path = build_dir / "production_add_rmsnorm.resources.txt"
    resources_path.write_text(resources_result.stdout)
    resources = _parse_production_resources(
        resources_result.stdout, cubin.name)
    sass_result = _run([
        str(cuobjdump), "--dump-sass", "--function",
        EXPECTED_PRODUCTION_SYMBOL, str(cubin)],
        cwd=build_dir, capture_output=True)
    sass_path = build_dir / "production_add_rmsnorm.sass"
    sass_path.write_text(sass_result.stdout)
    sass = _validate_production_sass(sass_result.stdout, cubin.name)
    return {
        "binary_path": str(binary),
        "binary_sha256": binary_sha256,
        "cubin_name": cubin.name,
        "cubin_sha256": cubin_sha256,
        "symbol": EXPECTED_PRODUCTION_SYMBOL,
        "resources": resources,
        "resource_dump_sha256": _sha256(resources_path),
        "sass": sass,
        "sass_dump_sha256": _sha256(sass_path),
        "rejected_activation_cubin": REJECTED_ACTIVATION_CUBIN,
    }


def _validated_output(path):
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be below {root}")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def _device_rows(nvidia_smi):
    query = "index,name,uuid,pci.bus_id,memory.used,utilization.gpu"
    result = _run([
        str(nvidia_smi), f"--query-gpu={query}",
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


def _compute_process_rows(nvidia_smi):
    result = _run([
        str(nvidia_smi),
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT, capture_output=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if row:
            if len(row) != 3:
                raise ValueError("unexpected nvidia-smi compute-process row")
            rows.append({
                "uuid": row[0], "pid": int(row[1]),
                "memory_mib": int(row[2]),
            })
    return rows


def _require_idle_device(device, nvidia_smi=DEFAULT_NVIDIA_SMI):
    matching = [
        row for row in _device_rows(nvidia_smi) if row["index"] == device]
    if len(matching) != 1:
        raise ValueError(f"device {device} does not resolve exactly once")
    row = matching[0]
    if row["name"] != EXPECTED_DEVICE:
        raise ValueError(f"device {device} is {row['name']}, not {EXPECTED_DEVICE}")
    if row["memory_used_mib"] > MAXIMUM_IDLE_MEMORY_MIB:
        raise RuntimeError(f"device {device} has {row['memory_used_mib']} MiB allocated")
    if row["utilization_percent"] != 0:
        raise RuntimeError(f"device {device} has nonzero utilization")
    processes = [
        entry for entry in _compute_process_rows(nvidia_smi)
        if entry["uuid"] == row["uuid"]]
    if processes:
        raise RuntimeError(f"device {device} has compute processes: {processes}")
    return row


def _wait_for_idle_device(device, nvidia_smi=DEFAULT_NVIDIA_SMI):
    error = None
    for _ in range(10):
        try:
            return _require_idle_device(device, nvidia_smi)
        except RuntimeError as current:
            error = current
            time.sleep(0.5)
    raise RuntimeError(f"device {device} did not return idle: {error}")


def _parse_resource_usage(text):
    resources = {}
    current = None
    for line in text.splitlines():
        match = SYMBOL_PATTERN.search(line)
        if match:
            current = match.group(1)
            continue
        if current is None or "add_rmsnorm_round_service" not in current:
            continue
        registers = REGISTER_PATTERN.search(line)
        shared = SHARED_PATTERN.search(line)
        local = LOCAL_PATTERN.search(line)
        if registers and shared and local:
            resources[current] = {
                "registers_per_thread": int(registers.group(1)),
                "static_shared_bytes": int(shared.group(1)),
                "local_bytes": int(local.group(1)),
            }
    if len(resources) != 2:
        raise ValueError(f"expected two kernel specializations, found {len(resources)}")
    for symbol, resource in resources.items():
        if (resource["registers_per_thread"] > 32
                or resource["static_shared_bytes"] != 0
                or resource["local_bytes"] != 0):
            raise ValueError(f"probe residency resource contract changed: {symbol}")
    return resources


def _validate_sass(text):
    required = ("LDG.E.128", "STG.E.128", "BAR.SYNC")
    missing = [opcode for opcode in required if opcode not in text]
    if missing:
        raise ValueError(f"SASS is missing required opcodes: {missing}")
    forbidden = ("LDL", "STL", "ATOM")
    present = [opcode for opcode in forbidden if opcode in text]
    if present:
        raise ValueError(f"SASS contains forbidden opcodes: {present}")
    return {
        "required_opcodes": list(required),
        "forbidden_opcodes_absent": list(forbidden),
    }


def _load_raw(path, trials):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("raw header changed")
        rows = list(reader)
    if len(rows) != trials * len(FUNCTIONS) * len(CELLS):
        raise ValueError("raw row count changed")
    seen = set()
    normalized = []
    for row in rows:
        function = row["function"]
        cell = row["cell"]
        trial = int(row["trial"])
        key = (function, cell, trial)
        if (function not in FUNCTIONS or cell not in CELLS or key in seen
                or trial not in range(trials)):
            raise ValueError(f"invalid function/cell/trial key {key}")
        seen.add(key)
        integers = {
            name: int(row[name]) for name in RAW_FIELDS
            if name not in ("function", "cell", "checksum")}
        if integers["grid_blocks"] != CELLS[cell]:
            raise ValueError(f"geometry changed for {function}/{cell}")
        if (integers["block_threads"] != EXPECTED_THREADS
                or integers["rounds"] != EXPECTED_ROUNDS
                or integers["resident_cta_limit"] != EXPECTED_RESIDENT_CTAS):
            raise ValueError(f"source contract changed for {function}/{cell}")
        expected_ctas = CELLS[cell] // EXPECTED_SM_COUNT
        if (integers["active_sms"] != EXPECTED_SM_COUNT
                or integers["minimum_ctas_per_sm"] != expected_ctas
                or integers["maximum_ctas_per_sm"] != expected_ctas):
            raise ValueError(f"CTA assignment changed for {function}/{cell}")
        if (integers["maximum_sm_span_ns"] <= 0
                or integers["median_block_ns"] <= 0):
            raise ValueError(f"timing contract failed for {function}/{cell}")
        normalized.append({
            "function": function, "cell": cell, "trial": trial,
            "checksum": float(row["checksum"]), **integers,
        })
    return normalized


def _derive_candidate(rows):
    functions = {}
    stable = True
    positive_increments = True
    for function in FUNCTIONS:
        cell_summaries = {}
        spans = {}
        for cell in CELLS:
            values = [
                row["maximum_sm_span_ns"] for row in rows
                if row["function"] == function and row["cell"] == cell]
            center = median(values)
            spread = (max(values) - min(values)) / center
            stable = stable and spread <= 0.10
            spans[cell] = center
            cell_summaries[cell] = {
                "median_maximum_sm_span_ns": center,
                "maximum_relative_spread": spread,
            }
        first_ns = spans["first"] / EXPECTED_ROUNDS
        resident_ns = (
            (spans["resident"] - spans["first"])
            / ((EXPECTED_RESIDENT_CTAS - 1) * EXPECTED_ROUNDS))
        positive_increments = positive_increments and min(
            first_ns, resident_ns) > 0
        functions[function] = {
            "cell_summaries": cell_summaries,
            "incremental_ns_per_cta_round": {
                "first": first_ns,
                "resident": resident_ns,
            },
            "service_rates_per_sm_s": {
                "first": 1e9 / first_ns if first_ns > 0 else None,
                "resident": 1e9 / resident_ns if resident_ns > 0 else None,
            },
        }
    return {
        "schema": "nvidia_a800_add_rmsnorm_round_service_candidate_v1",
        "hardware_id": "nvidia_a800",
        "block_threads": EXPECTED_THREADS,
        "resident_ctas_per_sm": EXPECTED_RESIDENT_CTAS,
        "functions": functions,
        "all_cells_stable_within_10_percent": stable,
        "all_incremental_services_positive": positive_increments,
        "forward_eligible": False,
        "production_or_evaluation_duration_consumed": False,
        "production_predictor_consumption_allowed": False,
        "allowed_conclusion": (
            "The independent block-256 source-round probe may characterize "
            "first and resident CTA service only after resource, "
            "SASS, stability, and positive-increment gates pass."),
        "excluded_conclusions": [
            "a complete Add-RMSNorm duration",
            "a reduction or reciprocal-square-root service",
            "a production-specific correction factor",
            "queued CTA service or grids above 864",
            "direct Forward consumption before composition review",
        ],
    }


def _validate_candidate(candidate):
    if set(candidate) != {
            "schema", "hardware_id", "block_threads",
            "resident_ctas_per_sm", "functions",
            "all_cells_stable_within_10_percent",
            "all_incremental_services_positive", "forward_eligible",
            "production_or_evaluation_duration_consumed",
            "production_predictor_consumption_allowed", "allowed_conclusion",
            "excluded_conclusions"}:
        raise ValueError("candidate schema changed")
    if (candidate["schema"]
            != "nvidia_a800_add_rmsnorm_round_service_candidate_v1"
            or candidate["hardware_id"] != "nvidia_a800"
            or candidate["block_threads"] != EXPECTED_THREADS
            or candidate["resident_ctas_per_sm"] != EXPECTED_RESIDENT_CTAS):
        raise ValueError("candidate identity or geometry changed")
    if (candidate["forward_eligible"] is not False
            or candidate["production_or_evaluation_duration_consumed"] is not False
            or candidate["production_predictor_consumption_allowed"] is not False):
        raise ValueError("candidate became consumable by production Forward")
    if candidate["excluded_conclusions"] != [
            "a complete Add-RMSNorm duration",
            "a reduction or reciprocal-square-root service",
            "a production-specific correction factor",
            "queued CTA service or grids above 864",
            "direct Forward consumption before composition review"]:
        raise ValueError("candidate exclusion scope changed")
    return candidate


def _run_native(command, *, cwd, stdout, stderr):
    process = subprocess.Popen(
        command, cwd=cwd, text=True, stdout=stdout, stderr=stderr,
        start_new_session=True)
    try:
        returncode = process.wait(timeout=600)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        raise
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return process.pid


def _validate_provenance(provenance):
    if set(provenance) != set(PROVENANCE_FIELDS):
        raise ValueError("capture provenance fields changed")
    if (provenance["schema"]
            != "nvidia_a800_add_rmsnorm_round_service_capture_v1"
            or provenance["forward_eligible"] is not False
            or provenance["production_or_evaluation_duration_consumed"] is not False
            or provenance["preflight_cleanup"]
            != "target_device_idle_no_compute_processes"
            or provenance["postflight_cleanup"]
            != "target_device_idle_no_compute_processes"):
        raise ValueError("capture provenance decision or cleanup changed")
    tools = provenance["tool_identities"]
    if (set(tools) != {"nvcc", "cuobjdump", "nvidia_smi"}
            or tools["nvcc"]["sha256"] != EXPECTED_NVCC_SHA256
            or tools["cuobjdump"]["sha256"] != EXPECTED_CUOBJDUMP_SHA256):
        raise ValueError("capture tool identity changed")
    production = provenance["production_artifact"]
    if (production["binary_sha256"] != EXPECTED_PRODUCTION_BINARY_SHA256
            or production["cubin_name"] != EXPECTED_PRODUCTION_CUBIN
            or production["cubin_sha256"] != EXPECTED_PRODUCTION_CUBIN_SHA256
            or production["symbol"] != EXPECTED_PRODUCTION_SYMBOL
            or production["resources"] != EXPECTED_PRODUCTION_RESOURCES
            or production["sass"]["sha256"]
            != EXPECTED_PRODUCTION_SASS_SHA256):
        raise ValueError("production artifact provenance changed")
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--nvcc", type=Path, default=DEFAULT_NVCC)
    parser.add_argument("--cuobjdump", type=Path, default=DEFAULT_CUOBJDUMP)
    parser.add_argument("--nvidia-smi", type=Path, default=DEFAULT_NVIDIA_SMI)
    parser.add_argument(
        "--production-binary", type=Path,
        default=DEFAULT_PRODUCTION_BINARY)
    args = parser.parse_args()
    if args.device < 0:
        raise ValueError("device must be nonnegative")
    if args.trials < MINIMUM_TRIALS or args.trials > MAXIMUM_TRIALS:
        raise ValueError(f"trials must be in [{MINIMUM_TRIALS}, {MAXIMUM_TRIALS}]")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", args.owner):
        raise ValueError("owner must match [A-Za-z0-9._-]{1,64}")
    nvcc = args.nvcc.resolve()
    cuobjdump = args.cuobjdump.resolve()
    nvidia_smi = args.nvidia_smi.resolve()
    production_binary = args.production_binary.resolve()
    output = _validated_output(args.output)
    started_at = _utc_now()
    device = _require_idle_device(args.device, nvidia_smi)
    output.mkdir(parents=True)
    for directory in (output / "build", output / "source_snapshot"):
        directory.mkdir()
    source_snapshot = output / "source_snapshot" / SOURCE_PATH.name
    runner_snapshot = output / "source_snapshot" / RUNNER_PATH.name
    shutil.copy2(SOURCE_PATH, source_snapshot)
    shutil.copy2(RUNNER_PATH, runner_snapshot)
    commit = _run([
        "git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout.strip()
    status = _run([
        "git", "status", "--short"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout
    (output / "baseline_commit.txt").write_text(commit + "\n")
    (output / "tracked_worktree_status.txt").write_text(status)
    tools = {
        "nvcc": _tool_identity(
            nvcc, ("--version",), EXPECTED_NVCC_SHA256),
        "cuobjdump": _tool_identity(
            cuobjdump, ("--version",), EXPECTED_CUOBJDUMP_SHA256),
        "nvidia_smi": _tool_identity(
            nvidia_smi,
            ("--query-gpu=driver_version", "--format=csv,noheader")),
    }
    production_artifact = _audit_production_artifact(
        production_binary, cuobjdump, output / "build")
    binary = output / "build/add_rmsnorm_round_service"
    compile_command = [
        str(nvcc), "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo",
        "-Xptxas=-v", str(source_snapshot), "-o", str(binary),
    ]
    with (output / "build/compile.stdout").open("w") as stdout, (
            output / "build/compile.stderr").open("w") as stderr:
        _run(compile_command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    resource_result = _run([
        str(cuobjdump), "--dump-resource-usage", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True)
    resource_path = output / "build/resource_usage.txt"
    resource_path.write_text(resource_result.stdout)
    resources = _parse_resource_usage(resource_result.stdout)
    sass_result = _run([
        str(cuobjdump), "--dump-sass", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True)
    sass_path = output / "build/add_rmsnorm_round_service.sass"
    sass_path.write_text(sass_result.stdout)
    sass_contract = _validate_sass(sass_result.stdout)
    run_command = [str(binary), str(args.device), str(args.trials)]
    with (output / "raw.csv").open("w") as stdout, (
            output / "native.stderr").open("w") as stderr:
        native_pid = _run_native(
            run_command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    rows = _load_raw(output / "raw.csv", args.trials)
    candidate = _validate_candidate(_derive_candidate(rows))
    (output / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    postflight = _wait_for_idle_device(args.device, nvidia_smi)
    completed_at = _utc_now()
    provenance = _validate_provenance({
        "schema": "nvidia_a800_add_rmsnorm_round_service_capture_v1",
        "owner": args.owner, "runner_pid": os.getpid(),
        "native_pid": native_pid,
        "started_at_utc": started_at, "completed_at_utc": completed_at,
        "baseline_commit": commit,
        "command": shlex.join([sys.executable, *sys.argv]),
        "native_command": shlex.join(run_command),
        "device": device, "postflight_device": postflight,
        "preflight_cleanup": "target_device_idle_no_compute_processes",
        "postflight_cleanup": "target_device_idle_no_compute_processes",
        "trials": args.trials,
        "tool_identities": tools,
        "source_sha256": _sha256(source_snapshot),
        "runner_sha256": _sha256(runner_snapshot),
        "probe_binary_sha256": _sha256(binary),
        "probe_resource_dump_sha256": _sha256(resource_path),
        "probe_sass_dump_sha256": _sha256(sass_path),
        "probe_resources": resources, "probe_sass_contract": sass_contract,
        "production_artifact": production_artifact,
        "candidate_sha256": _sha256(output / "candidate.json"),
        "raw_sha256": _sha256(output / "raw.csv"),
        "forward_eligible": False,
        "production_or_evaluation_duration_consumed": False,
    })
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
