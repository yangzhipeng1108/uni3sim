#!/usr/bin/env python3
"""Gate the H8192 A800 Add-RMS independent and chained phase capture."""

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import sys


RUNNER_PATH = Path(__file__).with_name(
    "add_rmsnorm_round_service_1024_h8192_runner.py")
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "add_rmsnorm_round_service_1024_h8192_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(RUNNER)

SOURCE_PATH = RUNNER.SOURCE_PATH
REPORT_PATH = Path(__file__).resolve()
REDUCTION_SOURCE_PATH = (
    RUNNER.REPOSITORY_ROOT
    / "microbench/nvidia_a800/compute/reduction/block_reduce_latency.cu")
REDUCTION_RUNNER_PATH = REDUCTION_SOURCE_PATH.with_name(
    "block_reduce_latency_runner.py")
REDUCTION_REPORT_PATH = REDUCTION_SOURCE_PATH.with_name(
    "block_reduce_latency_report.py")
EXPECTED_REDUCTION_HASHES = {
    "source_sha256": (
        "28851369b346e438c14f2d3dba168a11b8cd83a690677106687df191e12aa853"),
    "runner_sha256": (
        "587c0805b3bb03bd663b67dedd3ef76911b3545faeba7019c6a9d5f3b5a793e6"),
    "report_sha256": (
        "5794b6ddb0fcfd9ed7e83c75a0d10d522540841570bf7eb33439bd0bf56941f1"),
}
RAW_FIELDS = (
    "trial", "position", "sample", "function", "grid_blocks",
    "block_threads", "hidden_size", "active_threads", "active_warps",
    "registers_per_thread", "static_shared_bytes", "resident_cta_limit", "active_sms",
    "minimum_ctas_per_active_sm", "maximum_ctas_per_active_sm",
    "median_block_clock64_cycles", "checksum",
)
MAXIMUM_RELATIVE_SPREAD = 0.10
DERIVATION_TRIALS = (0, 1)
VALIDATION_TRIAL = 2
MAXIMUM_EXCESS_RELATIVE_SPREAD = 0.10
MAXIMUM_VALIDATION_APE_PERCENT = 10.0
MANIFEST_FIELDS = {
    "schema", "owner", "runner_pid", "native_pid", "started_at_utc",
    "completed_at_utc", "command", "native_command", "trials",
    "samples_per_trial_cell",
    "runtime_variance_scale", "timer_source", "measurement_unit",
    "device", "postflight_device",
    "preflight_cleanup", "postflight_cleanup", "nvidia_smi_identity",
    "static_contract", "static_contract_sha256", "raw_sha256",
    "production_or_evaluation_duration_consumed", "forward_eligible",
}
STATIC_FIELDS = {
    "schema", "baseline_commit", "production_or_evaluation_duration_consumed",
    "source_contract", "production_source_pin", "upstream_reference",
    "tool_identities", "production_artifact", "compile_command",
    "source_sha256", "runner_sha256", "report_sha256", "source_pin_sha256",
    "upstream_reference_sha256",
    "base_runner_sha256", "probe_binary_sha256", "probe_resource_sha256",
    "probe_sass_sha256", "probe_resources", "occupancy_proof",
    "probe_sass_contract",
}
EXPECTED_TIMED_OPCODE_COUNTS = {
    "add_square": {
        "CS2R": 1, "MOV": 1, "IMAD.WIDE": 2,
        "LDG.E.128.CONSTANT": 1, "LDG.E.128": 1, "STG.E.128": 1,
        "HFMA2.BF16_V2": 4, "PRMT": 8, "FMUL": 4,
        "FFMA": 4, "FADD": 4, "LDS": 0, "IMAD.WIDE.U32": 0,
        "F2FP.BF16.PACK_AB": 0,
    },
    "scale_multiply": {
        "CS2R": 1, "MOV": 0, "IMAD.WIDE": 1,
        "IMAD.WIDE.U32": 1,
        "LDG.E.128.CONSTANT": 2, "STG.E.128": 1, "LDS": 1,
        "PRMT": 8, "FMUL": 8, "F2FP.BF16.PACK_AB": 4,
        "HFMA2.BF16_V2": 4, "LDG.E.128": 0, "FFMA": 0, "FADD": 0,
    },
    "publication_barrier": {
        "CS2R": 2, "MOV": 2, "FFMA": 1, "STS": 1,
        "BAR.SYNC": 1, "LDS": 1, "ISETP.GT.U32.AND": 1,
        "BSSY": 1, "BRA": 1, "LOP3.LUT": 1,
    },
    "phase_chain": {
        "CS2R": 2, "MOV": 1, "LDG.E.128": 3,
        "HFMA2.BF16_V2": 8, "PRMT": 16, "FMUL": 12,
        "FFMA": 5, "FADD": 4, "STG.E.128": 2,
        "BSYNC": 1, "ISETP.NE.AND": 1, "BSSY": 1,
        "STS": 1, "BAR.SYNC": 1, "BRA": 1,
        "IMAD.WIDE.U32": 1, "LDS": 1,
        "LDG.E.128.CONSTANT": 1, "LOP3.LUT": 1,
        "F2FP.BF16.PACK_AB": 4,
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redump_probe(binary_path, cuobjdump_path):
    resource = RUNNER.BASE._run([
        str(cuobjdump_path), "--dump-resource-usage", str(binary_path)],
        cwd=RUNNER.REPOSITORY_ROOT, capture_output=True).stdout
    sass = RUNNER.BASE._run([
        str(cuobjdump_path), "--dump-sass", str(binary_path)],
        cwd=RUNNER.REPOSITORY_ROOT, capture_output=True).stdout
    return resource, sass


def _integer(text, label, *, positive=False, non_negative=False):
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer") from error
    if str(value) != text:
        raise ValueError(f"{label} must be canonical")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _expected_schedule():
    cells = tuple(
        (function, grid)
        for grid in RUNNER.EXPECTED_GRIDS
        for function in RUNNER.EXPECTED_FUNCTIONS)
    expected = {}
    for trial in range(RUNNER.EXPECTED_TRIALS):
        offset = (trial * 3) % len(cells)
        for position in range(len(cells)):
            expected[(trial, cells[(position + offset) % len(cells)])] = position
    return expected


def validate_manifest(path, raw_path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("capture manifest fields changed")
    if (manifest["schema"]
            != "a800_add_rmsnorm_phase_chain_1024_h8192_clock64_capture_v2"):
        raise ValueError("capture schema changed")
    if (manifest["trials"] != RUNNER.EXPECTED_TRIALS
            or manifest["samples_per_trial_cell"] != RUNNER.EXPECTED_SAMPLES
            or manifest["runtime_variance_scale"] != 0.875
            or manifest["timer_source"] != "clock64"
            or manifest["measurement_unit"] != "cycles"
            or manifest["production_or_evaluation_duration_consumed"] is not False
            or manifest["forward_eligible"] is not False
            or manifest["preflight_cleanup"]
            != "target_device_idle_no_compute_processes"
            or manifest["postflight_cleanup"]
            != "target_device_idle_no_compute_processes"):
        raise ValueError("capture execution or duration contract changed")
    if (not isinstance(manifest["owner"], str)
            or re.fullmatch(
                r"[A-Za-z0-9._-]{1,64}", manifest["owner"]) is None
            or not isinstance(manifest["runner_pid"], int)
            or manifest["runner_pid"] <= 0
            or not isinstance(manifest["native_pid"], int)
            or manifest["native_pid"] <= 0):
        raise ValueError("capture owner or process identity changed")
    try:
        started_at = datetime.fromisoformat(manifest["started_at_utc"])
        completed_at = datetime.fromisoformat(manifest["completed_at_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("capture timestamps are not ISO-8601") from error
    if started_at.tzinfo is None or completed_at.tzinfo is None:
        raise ValueError("capture timestamps must include timezone")
    if completed_at < started_at:
        raise ValueError("capture timestamps are reversed")
    if manifest["raw_sha256"] != _sha256(raw_path):
        raise ValueError("capture raw SHA256 mismatch")
    static_path = path.parent / "static_contract.json"
    if manifest["static_contract_sha256"] != _sha256(static_path):
        raise ValueError("capture static contract SHA256 mismatch")
    static = manifest["static_contract"]
    if set(static) != STATIC_FIELDS:
        raise ValueError("capture static contract fields changed")
    if static != json.loads(static_path.read_text(encoding="utf-8")):
        raise ValueError("embedded and standalone static contracts differ")
    if (static["schema"]
            != "a800_add_rmsnorm_phase_chain_1024_h8192_clock64_static_v2"
            or static["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError("captured static contract identity changed")
    if (static["source_sha256"] != _sha256(SOURCE_PATH)
            or static["runner_sha256"] != _sha256(RUNNER_PATH)
            or static["report_sha256"] != _sha256(REPORT_PATH)
            or static["base_runner_sha256"]
            != _sha256(RUNNER.BASE_RUNNER_PATH)):
        raise ValueError("capture validator source identity changed")
    artifact_paths = {
        "probe_binary_sha256": (
            path.parent / "build/add_rmsnorm_round_service_1024_h8192"),
        "probe_resource_sha256": (
            path.parent / "build/probe_resource_usage.txt"),
        "probe_sass_sha256": (
            path.parent / "build/add_rmsnorm_round_service_1024_h8192.sass"),
        "source_sha256": path.parent / "source_snapshot" / SOURCE_PATH.name,
        "runner_sha256": path.parent / "source_snapshot" / RUNNER_PATH.name,
        "report_sha256": path.parent / "source_snapshot" / REPORT_PATH.name,
        "source_pin_sha256": (
            path.parent / "source_snapshot" / RUNNER.SOURCE_PIN_PATH.name),
        "upstream_reference_sha256": (
            path.parent / "source_snapshot"
            / RUNNER.UPSTREAM_REFERENCE_PATH.name),
    }
    for field, artifact_path in artifact_paths.items():
        if not artifact_path.is_file() or static[field] != _sha256(artifact_path):
            raise ValueError(f"capture {field} artifact drift")
    snapshot = path.parent / "source_snapshot"
    captured_source_contract = RUNNER._validate_probe_source(
        snapshot / SOURCE_PATH.name)
    if static["source_contract"] != captured_source_contract:
        raise ValueError("captured source contract is not derived from snapshot")
    captured_source_pin = RUNNER._validate_source_pin(
        snapshot / RUNNER.SOURCE_PIN_PATH.name)
    if static["production_source_pin"] != captured_source_pin:
        raise ValueError("captured production source pin changed")
    captured_upstream_reference = RUNNER._validate_upstream_reference(
        snapshot / RUNNER.UPSTREAM_REFERENCE_PATH.name)
    if static["upstream_reference"] != captured_upstream_reference:
        raise ValueError("captured upstream reference changed")
    source_contract = captured_source_contract
    if (source_contract["threads_per_cta"] != RUNNER.EXPECTED_THREADS
            or source_contract["hidden_size"] != RUNNER.EXPECTED_HIDDEN_SIZE
            or source_contract["active_threads"]
            != RUNNER.EXPECTED_ACTIVE_THREADS
            or source_contract["active_warps"] != RUNNER.EXPECTED_ACTIVE_WARPS
            or source_contract["registers_per_thread"]
            != RUNNER.EXPECTED_REGISTERS
            or source_contract["static_shared_bytes"]
            != RUNNER.EXPECTED_STATIC_SHARED_BYTES
            or source_contract["resident_ctas_per_sm"]
            != RUNNER.EXPECTED_RESIDENT_CTAS
            or tuple(source_contract["grids"]) != RUNNER.EXPECTED_GRIDS
            or source_contract["samples_per_trial_cell"]
            != RUNNER.EXPECTED_SAMPLES
            or source_contract["source_rounds_per_kernel_invocation"] != 1
            or source_contract[
                "phase_chain_source_rounds_per_kernel_invocation"] != 2
            or source_contract["timer_source"] != "clock64"
            or source_contract["timer_unit"] != "cycles"
            or source_contract["timer_comparison_scope"]
            != "same_cta_same_sm"
            or source_contract["input_and_residual_restored_before_each_sample"]
            is not True
            or source_contract["add_residual_load_store_same_address"]
            is not True
            or source_contract["scale_residual_load_and_input_store"]
            is not True
            or source_contract[
                "timer_values_stay_in_registers_until_post_timed_barrier"]
            is not True
            or source_contract["all_threads_reach_post_timed_barrier"]
            is not True
            or source_contract["reduction_or_rsqrt_absent"] is not True
            or source_contract["checksum_or_resource_padding_in_timed_region"]
            is not False
            or source_contract["scale_value_source"]
            != "runtime_kernel_argument_written_to_shared_s_variance"
            or source_contract["phase_chain_middle_stage"]
            != "thread0_dependency_publication_then_cta_barrier"
            or source_contract["phase_chain_dependency_source"]
            != "thread0_add_square_variance"
            or source_contract["publication_control_dependency_source"]
            != "runtime_kernel_argument"):
        raise ValueError("captured source-round contract changed")
    resource_path = path.parent / "build/probe_resource_usage.txt"
    captured_resources = RUNNER._parse_probe_resources(
        resource_path.read_text(encoding="utf-8"))
    if static["probe_resources"] != captured_resources:
        raise ValueError("captured probe resources are not derived from dump")
    captured_occupancy = RUNNER._occupancy_proof(captured_resources)
    if static["occupancy_proof"] != captured_occupancy:
        raise ValueError("captured occupancy proof is not derived from resources")
    sass_path = path.parent / "build/add_rmsnorm_round_service_1024_h8192.sass"
    captured_sass = RUNNER._validate_probe_sass(
        sass_path.read_text(encoding="utf-8"), captured_resources)
    if static["probe_sass_contract"] != captured_sass:
        raise ValueError("captured probe SASS contract is not derived from dump")
    if set(captured_resources) != set(RUNNER.EXPECTED_FUNCTIONS):
        raise ValueError("captured probe specialization set changed")
    for function in RUNNER.EXPECTED_FUNCTIONS:
        resource = captured_resources[function]
        sass = captured_sass[function]
        occupancy = captured_occupancy["by_function"][function]
        expected_dependency_barriers = {
            "add_square": 0,
            "scale_multiply": 0,
            "publication_barrier": 1,
            "phase_chain": 1,
        }[function]
        expected_global_stores = {
            "add_square": ["STG.E.128"],
            "scale_multiply": ["STG.E.128"],
            "publication_barrier": [],
            "phase_chain": ["STG.E.128", "STG.E.128"],
        }[function]
        expected_shared_stores = {
            "add_square": [],
            "scale_multiply": [],
            "publication_barrier": ["STS"],
            "phase_chain": ["STS"],
        }[function]
        if (resource["registers_per_thread"] != RUNNER.EXPECTED_REGISTERS
                or resource["static_shared_bytes"]
                != RUNNER.EXPECTED_STATIC_SHARED_BYTES
                or resource["stack_bytes"] != 0
                or resource["local_bytes"] != 0
                or occupancy["register_limit"] != 2
                or occupancy["thread_limit"] != 2
                or occupancy["resident_ctas_per_sm"] != 2
                or sass["reduction_or_rsqrt_absent"] is not True
                or sass["timed_barrier_absent"]
                is not (expected_dependency_barriers == 0)
                or sass["timed_dependency_barrier_count"]
                != expected_dependency_barriers
                or sass["timed_global_stores"] != expected_global_stores
                or sass["timed_shared_stores"] != expected_shared_stores
                or sass[
                    "post_timer_barrier_precedes_padding_and_timestamp_stores"]
                is not True
                or sass["post_timer_block_barrier_unpredicated"] is not True
                or sass["packed_bf16_semantics_exact"] is not True
                or sass["timed_opcode_counts"]
                != EXPECTED_TIMED_OPCODE_COUNTS[function]
                or sass["clock64_reads"] != 2
                or sass["globaltimer_reads"] != 0):
            raise ValueError(f"captured static identity changed for {function}")
    tool_specs = {
        "nvcc": (("--version",), RUNNER.BASE.EXPECTED_NVCC_SHA256),
        "cuobjdump": (("--version",), RUNNER.BASE.EXPECTED_CUOBJDUMP_SHA256),
    }
    if set(static["tool_identities"]) != set(tool_specs):
        raise ValueError("captured tool identity set changed")
    for name, (version_args, expected_sha256) in tool_specs.items():
        captured_identity = static["tool_identities"][name]
        live_identity = RUNNER.BASE._tool_identity(
            Path(captured_identity["path"]), version_args, expected_sha256)
        if captured_identity != live_identity:
            raise ValueError(f"captured {name} identity differs from live tool")
    nvidia_smi_identity = manifest["nvidia_smi_identity"]
    if set(nvidia_smi_identity) != {"path", "sha256", "version"}:
        raise ValueError("captured nvidia-smi identity is not structured")
    live_nvidia_smi_identity = RUNNER.BASE._tool_identity(
        Path(nvidia_smi_identity["path"]),
        ("--query-gpu=driver_version", "--format=csv,noheader"))
    if nvidia_smi_identity != live_nvidia_smi_identity:
        raise ValueError("captured nvidia-smi identity differs from live tool")
    expected_compile_command = [
        static["tool_identities"]["nvcc"]["path"], "-O3", "-std=c++17",
        "-arch=sm_80", "-lineinfo", "-Xptxas=-v",
        str(snapshot / SOURCE_PATH.name),
        "-o", str(path.parent / "build/add_rmsnorm_round_service_1024_h8192"),
    ]
    if static["compile_command"] != shlex.join(expected_compile_command):
        raise ValueError("captured compile command is not reconstructible")
    probe_binary_path = path.parent / "build/add_rmsnorm_round_service_1024_h8192"
    expected_native_command = [
        str(probe_binary_path), str(manifest["device"]["index"]), "0.875"]
    if shlex.split(manifest["native_command"]) != expected_native_command:
        raise ValueError("captured native command does not bind binary/device/scale")
    expected_runner_command = [
        sys.executable, str(RUNNER_PATH), "--run-id", path.parent.name,
        "--device", str(manifest["device"]["index"]),
        "--owner", manifest["owner"],
    ]
    actual_runner_command = shlex.split(manifest["command"])
    if (len(actual_runner_command) != len(expected_runner_command)
            or Path(actual_runner_command[0]).resolve()
            != Path(expected_runner_command[0]).resolve()
            or Path(actual_runner_command[1]).resolve()
            != RUNNER_PATH.resolve()
            or actual_runner_command[2:] != expected_runner_command[2:]):
        raise ValueError("captured runner command changed")
    live_resource_text, live_sass_text = _redump_probe(
        probe_binary_path,
        Path(static["tool_identities"]["cuobjdump"]["path"]))
    saved_resource_text = resource_path.read_text(encoding="utf-8")
    saved_sass_text = sass_path.read_text(encoding="utf-8")
    if live_resource_text != saved_resource_text:
        raise ValueError("live probe resource dump differs from captured artifact")
    if live_sass_text != saved_sass_text:
        raise ValueError("live probe SASS dump differs from captured artifact")
    live_resources = RUNNER._parse_probe_resources(live_resource_text)
    if live_resources != captured_resources:
        raise ValueError("live probe resources differ from captured parse")
    live_sass = RUNNER._validate_probe_sass(live_sass_text, live_resources)
    if live_sass != captured_sass:
        raise ValueError("live probe SASS differs from captured parse")

    production = static["production_artifact"]
    if (production["binary_sha256"] != RUNNER.BASE.EXPECTED_PRODUCTION_BINARY_SHA256
            or production["cubin_sha256"]
            != RUNNER.BASE.EXPECTED_PRODUCTION_CUBIN_SHA256
            or production["sass"]["sha256"]
            != RUNNER.BASE.EXPECTED_PRODUCTION_SASS_SHA256
            or production["resources"] != RUNNER.BASE.EXPECTED_PRODUCTION_RESOURCES):
        raise ValueError("captured production SASS/resource pin changed")
    production_paths = {
        "resource_dump_sha256": (
            path.parent / "build/production_add_rmsnorm.resources.txt"),
        "sass_dump_sha256": (
            path.parent / "build/production_add_rmsnorm.sass"),
    }
    for field, artifact_path in production_paths.items():
        if not artifact_path.is_file() or production[field] != _sha256(artifact_path):
            raise ValueError(f"capture production {field} artifact drift")
    production_resources = RUNNER.BASE._parse_production_resources(
        production_paths["resource_dump_sha256"].read_text(encoding="utf-8"),
        RUNNER.BASE.EXPECTED_PRODUCTION_CUBIN)
    if production["resources"] != production_resources:
        raise ValueError("captured production resources are not derived from dump")
    production_sass = RUNNER.BASE._validate_production_sass(
        production_paths["sass_dump_sha256"].read_text(encoding="utf-8"),
        RUNNER.BASE.EXPECTED_PRODUCTION_CUBIN)
    if production["sass"] != production_sass:
        raise ValueError("captured production SASS is not derived from dump")
    expected_device_fields = {
        "index", "name", "uuid", "pci_bus_id", "memory_used_mib",
        "utilization_percent"}
    device = manifest["device"]
    postflight = manifest["postflight_device"]
    if (set(device) != expected_device_fields
            or set(postflight) != expected_device_fields
            or device["name"] != RUNNER.BASE.EXPECTED_DEVICE
            or postflight["name"] != RUNNER.BASE.EXPECTED_DEVICE
            or device["index"] != postflight["index"]
            or device["uuid"] != postflight["uuid"]
            or device["pci_bus_id"] != postflight["pci_bus_id"]
            or device["memory_used_mib"] > RUNNER.BASE.MAXIMUM_IDLE_MEMORY_MIB
            or postflight["memory_used_mib"]
            > RUNNER.BASE.MAXIMUM_IDLE_MEMORY_MIB
            or device["utilization_percent"] != 0
            or postflight["utilization_percent"] != 0):
        raise ValueError("capture device or postflight identity changed")
    return manifest, static


def read_raw(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("raw fields changed")
        raw_rows = list(reader)
    expected_schedule = _expected_schedule()
    expected_count = (
        RUNNER.EXPECTED_TRIALS * RUNNER.EXPECTED_SAMPLES
        * len(RUNNER.EXPECTED_FUNCTIONS) * len(RUNNER.EXPECTED_GRIDS))
    if len(raw_rows) != expected_count:
        raise ValueError("raw matrix is incomplete")
    rows = []
    seen = set()
    for line, raw in enumerate(raw_rows, start=2):
        function = raw["function"]
        if function not in RUNNER.EXPECTED_FUNCTIONS:
            raise ValueError(f"line {line} has unsupported function")
        integers = {
            name: _integer(
                raw[name], f"line {line}.{name}",
                positive=name not in (
                    "trial", "position", "sample",
                    "median_block_clock64_cycles"),
                non_negative=name in (
                    "trial", "position", "sample",
                    "median_block_clock64_cycles"))
            for name in RAW_FIELDS
            if name not in ("function", "checksum")
        }
        schedule_key = (
            integers["trial"], (function, integers["grid_blocks"]))
        key = (*schedule_key, integers["sample"])
        if (key in seen or schedule_key not in expected_schedule
                or integers["sample"] >= RUNNER.EXPECTED_SAMPLES):
            raise ValueError(f"line {line} duplicates or changes a matrix cell")
        seen.add(key)
        if integers["position"] != expected_schedule[schedule_key]:
            raise ValueError(f"line {line} changed the frozen rotation")
        if (integers["block_threads"] != RUNNER.EXPECTED_THREADS
                or integers["hidden_size"] != RUNNER.EXPECTED_HIDDEN_SIZE
                or integers["active_threads"] != RUNNER.EXPECTED_ACTIVE_THREADS
                or integers["active_warps"] != RUNNER.EXPECTED_ACTIVE_WARPS
                or integers["registers_per_thread"] != RUNNER.EXPECTED_REGISTERS
                or integers["static_shared_bytes"]
                != RUNNER.EXPECTED_STATIC_SHARED_BYTES
                or integers["resident_cta_limit"]
                != RUNNER.EXPECTED_RESIDENT_CTAS):
            raise ValueError(f"line {line} changed the source/resource identity")
        grid = integers["grid_blocks"]
        active_sms = integers["active_sms"]
        minimum_ctas = integers["minimum_ctas_per_active_sm"]
        maximum_ctas = integers["maximum_ctas_per_active_sm"]
        if (active_sms > grid or minimum_ctas > maximum_ctas
                or maximum_ctas > RUNNER.EXPECTED_RESIDENT_CTAS
                or active_sms * minimum_ctas > grid
                or active_sms * maximum_ctas < grid):
            raise ValueError(f"line {line} has impossible CTA assignment")
        try:
            checksum = float(raw["checksum"])
        except ValueError as error:
            raise ValueError(f"line {line}.checksum must be numeric") from error
        if not math.isfinite(checksum):
            raise ValueError(f"line {line}.checksum must be finite")
        rows.append({"function": function, "checksum": checksum, **integers})
    if len(seen) != expected_count:
        raise ValueError("raw matrix is incomplete")
    topology_by_cell = {}
    for row in rows:
        key = (row["function"], row["grid_blocks"])
        topology = (
            row["active_sms"], row["minimum_ctas_per_active_sm"],
            row["maximum_ctas_per_active_sm"])
        if key in topology_by_cell and topology_by_cell[key] != topology:
            raise ValueError("CTA topology changed across trials")
        topology_by_cell[key] = topology
    for (_, grid), topology in topology_by_cell.items():
        if topology != (grid, 1, 1):
            raise ValueError("CTA topology does not match the frozen grid")
    return tuple(rows)


def _validate_reduction_partition():
    paths = {
        "source_sha256": REDUCTION_SOURCE_PATH,
        "runner_sha256": REDUCTION_RUNNER_PATH,
        "report_sha256": REDUCTION_REPORT_PATH,
    }
    for name, path in paths.items():
        if _sha256(path) != EXPECTED_REDUCTION_HASHES[name]:
            raise ValueError(f"existing reduction {name} changed")
    text = REDUCTION_SOURCE_PATH.read_text(encoding="utf-8")
    begin = text.index("const std::uint64_t begin_tick = mechanism_timer();")
    reduce = text.index("BlockReduce(reduce_storage).Reduce(")
    reciprocal = text.index("inverse_root = rsqrtf(variance);")
    barrier = text.index("__syncthreads();", reciprocal)
    end = text.index("const std::uint64_t end_tick = mechanism_timer();")
    if not begin < reduce < reciprocal < barrier < end:
        raise ValueError("existing reduction timed partition changed")
    return {
        **EXPECTED_REDUCTION_HASHES,
        "timed_phase": "reduction_and_publication",
        "global_memory_in_timed_region": False,
        "source_round_work_in_timed_region": False,
    }


def derive_report(rows, static):
    stages = []
    medians = {}
    all_stable = True
    all_positive = True
    for function in RUNNER.EXPECTED_FUNCTIONS:
        cells = []
        for grid in RUNNER.EXPECTED_GRIDS:
            trial_medians = []
            for trial in range(RUNNER.EXPECTED_TRIALS):
                samples = [
                    row["median_block_clock64_cycles"] for row in rows
                    if row["function"] == function
                    and row["grid_blocks"] == grid
                    and row["trial"] == trial]
                if len(samples) != RUNNER.EXPECTED_SAMPLES:
                    raise ValueError("derived sample matrix is incomplete")
                trial_medians.append(statistics.median(samples))
            durations = trial_medians
            if len(durations) != RUNNER.EXPECTED_TRIALS:
                raise ValueError("derived matrix is incomplete")
            center = statistics.median(durations)
            if center <= 0 or any(duration <= 0 for duration in durations):
                raise ValueError("source-round trial median is not positive")
            spread = (max(durations) - min(durations)) / center
            service = center
            stable = spread <= MAXIMUM_RELATIVE_SPREAD
            positive = service > 0.0
            all_stable = all_stable and stable
            all_positive = all_positive and positive
            cells.append({
                "grid_blocks": grid,
                "median_block_clock64_cycles": center,
                "trial_median_block_clock64_cycles": trial_medians,
                "independent_clock64_cycles_per_h8192_source_round": service,
                "relative_spread": spread,
                "stable_within_10_percent": stable,
                "positive_independent_service": positive,
            })
            medians[(function, grid)] = tuple(trial_medians)
        stages.append({"function": function, "cells": cells})
    if not all_stable:
        raise ValueError("source-round trial spread exceeds the frozen 10% gate")
    if not all_positive:
        raise ValueError("source-round independent service is not positive")
    reduction = _validate_reduction_partition()
    source_pin = static["production_source_pin"]
    if source_pin["phase_order"] != [
            "add_square_source_round", "reduction_and_publication",
            "scale_multiply_source_round"]:
        raise ValueError("production source phase partition changed")
    phase_chain_excess = []
    publishable_grids = []
    for grid in RUNNER.EXPECTED_GRIDS:
        excess_by_trial = tuple(
            medians[("phase_chain", grid)][trial]
            - medians[("add_square", grid)][trial]
            - medians[("publication_barrier", grid)][trial]
            - medians[("scale_multiply", grid)][trial]
            for trial in range(RUNNER.EXPECTED_TRIALS))
        derived = statistics.median(
            excess_by_trial[trial] for trial in DERIVATION_TRIALS)
        derivation_spread = (
            (max(excess_by_trial[trial] for trial in DERIVATION_TRIALS)
             - min(excess_by_trial[trial] for trial in DERIVATION_TRIALS))
            / abs(derived)
            if derived != 0 else math.inf)
        validation_measured = medians[("phase_chain", grid)][VALIDATION_TRIAL]
        validation_predicted = (
            medians[("add_square", grid)][VALIDATION_TRIAL]
            + medians[("publication_barrier", grid)][VALIDATION_TRIAL]
            + medians[("scale_multiply", grid)][VALIDATION_TRIAL]
            + derived)
        validation_ape = (
            abs(validation_predicted - validation_measured)
            / validation_measured * 100.0)
        positive_derivation = (
            derived > 0
            and all(excess_by_trial[trial] > 0
                    for trial in DERIVATION_TRIALS))
        stable_derivation = (
            positive_derivation
            and derivation_spread <= MAXIMUM_EXCESS_RELATIVE_SPREAD)
        validated = (
            excess_by_trial[VALIDATION_TRIAL] > 0
            and validation_ape <= MAXIMUM_VALIDATION_APE_PERCENT)
        publishable = stable_derivation and validated
        if publishable:
            publishable_grids.append(grid)
        phase_chain_excess.append({
            "grid_blocks": grid,
            "trial_excess_clock64_cycles": list(excess_by_trial),
            "derived_excess_clock64_cycles": derived,
            "derivation_relative_spread": derivation_spread,
            "positive_in_both_derivation_trials": positive_derivation,
            "stable_derivation_within_10_percent": stable_derivation,
            "validation_measured_chain_clock64_cycles": validation_measured,
            "validation_predicted_chain_clock64_cycles": validation_predicted,
            "validation_ape_percent": validation_ape,
            "validation_positive_excess": (
                excess_by_trial[VALIDATION_TRIAL] > 0),
            "publishable": publishable,
        })
    return {
        "schema": "a800_add_rmsnorm_phase_chain_1024_h8192_clock64_report_v2",
        "status": "mechanism_candidate",
        "hardware_id": "nvidia_a800",
        "production_or_evaluation_duration_consumed": False,
        "forward_eligible": False,
        "validity_domain": {
            "dtype": "bfloat16",
            "vector_width_elements": 8,
            "block_threads": 1024,
            "hidden_size": RUNNER.EXPECTED_HIDDEN_SIZE,
            "rows": list(RUNNER.EXPECTED_GRIDS),
            "grids": list(RUNNER.EXPECTED_GRIDS),
            "registers_per_thread": 32,
            "static_shared_bytes": 168,
            "resident_ctas_per_sm": 2,
        },
        "normalization": {
            "active_threads": RUNNER.EXPECTED_ACTIVE_THREADS,
            "active_warps": RUNNER.EXPECTED_ACTIVE_WARPS,
            "source_rounds_per_kernel_invocation": 1,
            "host_samples_per_trial_cell": RUNNER.EXPECTED_SAMPLES,
            "timer_source": "clock64",
            "timer_comparison_scope": "same_cta_same_sm",
            "service_unit": "clock64_cycles_per_h8192_source_round",
            "derivation_trials": list(DERIVATION_TRIALS),
            "validation_trial": VALIDATION_TRIAL,
        },
        "measurement_precondition": {
            "input_seed": "cudaMemset(input, 1) immediately before each sample",
            "residual_seed": (
                "cudaMemset(residual, 2) immediately before each sample"),
            "stream_order": (
                "seed writes and measured kernel use the same CUDA default stream"),
            "cache_state": "write_hot_after_host_issued_cudaMemset",
            "service_scope": "conditional_on_this_write_hot_cache_precondition",
        },
        "stages": stages,
        "phase_chain_excess": phase_chain_excess,
        "publishable_exact_grids": publishable_grids,
        "gates": {
            "three_trials_per_cell": True,
            "maximum_relative_spread": MAXIMUM_RELATIVE_SPREAD,
            "all_cells_stable_within_10_percent": True,
            "all_independent_services_positive": True,
            "maximum_excess_relative_spread": (
                MAXIMUM_EXCESS_RELATIVE_SPREAD),
            "maximum_validation_ape_percent": (
                MAXIMUM_VALIDATION_APE_PERCENT),
            "probe_resource_and_occupancy_identity_exact": True,
            "production_source_sass_and_resource_identity_exact": True,
            "composition_has_no_reduction_double_count": True,
        },
        "composition_contract": {
            "production_phase_order": source_pin["phase_order"],
            "source_round_timed_phases": [
                "add_square_source_round", "scale_multiply_source_round"],
            "control_timed_phase": "dependency_publication_and_cta_barrier",
            "phase_chain_timed_phases": [
                "add_square_source_round",
                "dependency_publication_and_cta_barrier",
                "scale_multiply_source_round",
            ],
            "existing_reduction_timed_phase": "reduction_and_publication",
            "phase_intersection": [],
            "composition_expression": (
                "add_square_source_service + reduction_and_publication_service "
                "+ scale_multiply_source_service + phase_chain_excess"),
            "phase_chain_excess_expression": (
                "phase_chain - add_square - publication_barrier "
                "- scale_multiply"),
            "publication_control_is_subtracted_before_forward_use": True,
            "each_phase_may_be_composed_at_most_once": True,
            "source_round_probe_reduction_rsqrt_and_checksum_math_absent": True,
            "reduction_probe_source_round_work_absent": True,
            "reduction_contract": reduction,
            "complete_operator_duration_term_allowed": False,
            "residual_or_duration_fit_allowed": False,
        },
        "allowed_conclusion": (
            "For the exact pinned BF16 vec8 block-1024 H=4096 source and grids "
            "1/4/16/64/108, the probe measures independent source phases, an "
            "isolated dependency-publication control, and their exact chained "
            "schedule; only publishable_exact_grids may supply a positive "
            "dependency excess after separate Forward integration review."),
        "excluded_conclusions": [
            "a complete Add-RMSNorm duration",
            "a production-duration correction or residual",
            "rows other than 1/4/16/64/108 or hidden sizes other than 4096",
            "block widths other than 1024",
            "direct Forward consumption without a separate integration review",
        ],
    }


def build_report(raw_path, manifest_path):
    manifest, static = validate_manifest(manifest_path, raw_path)
    report = derive_report(read_raw(raw_path), static)
    report["input"] = {
        "raw_path": str(raw_path),
        "raw_sha256": manifest["raw_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "static_contract_sha256": manifest["static_contract_sha256"],
        "baseline_commit": static["baseline_commit"],
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.input, args.manifest)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
