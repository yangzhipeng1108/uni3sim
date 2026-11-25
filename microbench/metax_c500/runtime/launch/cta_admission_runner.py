#!/usr/bin/env python3
"""Build, run, and validate the C500 CTA-admission mechanism probe."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics
import subprocess
import time

import zstandard


SCHEMA_VERSION = 2
HARDWARE_ID = "metax_c500"
MECHANISM = "resource_matched_cta_admission"
AP_COUNT = 104
WAVE_SIZE = 64
L2_BYTES = 8 * 1024 * 1024
BLOCK_THREADS = 256
DYNAMIC_SHARED_BYTES = 32 * 1024
TARGET_REGISTERS_PER_THREAD = 238
REGISTER_PRESSURE_SCALARS = 224
ADMISSION_HOLD_CYCLES = 2250000
TRIALS = 7
SCHEDULE_SEED = 500829
SOURCE_RELATIVE_PATH = (
    "microbench/metax_c500/runtime/launch/cta_admission.cpp")
RUNNER_RELATIVE_PATH = (
    "microbench/metax_c500/runtime/launch/cta_admission_runner.py")
OUTPUT_RELATIVE_ROOT = "profiler_output/microbench/metax_c500"
TARGET_DATASET_RELATIVE_PATH = "kernel_dataset/metax_c500/flash_attn.json.zst"
TARGET_DATASET_BLOB_SHA256 = (
    "983e9f40f762ccf79d1ec3a05389c6ec23006a2d88c96311eabda8b12dbd2511")
TARGET_CASE_ID = "FLASHATTN_PREFILL_B4_S512_HQ32_HKV8_D128_C1"
TARGET_KERNEL_INDEX = 0
MXCC_PATH = Path("/opt/maca/mxgpu_llvm/bin/mxcc")
MX_SMI_PATH = Path("/opt/mxdriver/bin/mx-smi")
TOOL_QUERY_TIMEOUT_SECONDS = 30
COMPILE_TIMEOUT_SECONDS = 300
CAPTURE_TIMEOUT_SECONDS = 300
PROCESS_TERMINATION_GRACE_SECONDS = 5
MAX_IDLE_HBM_THROUGHPUT_MBYTES_PER_SECOND = 1.0
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
MX_SMI_DEVICE_PATTERN = re.compile(
    r"^\s*GPU#(?P<device>\d+)\s+MXC500\s+"
    r"(?P<bus>[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\s*$",
    re.IGNORECASE | re.MULTILINE)
MX_SMI_GPU_USAGE_PATTERN = re.compile(
    r"^\s*\|?\s*GPU\s*(?:\||:)\s*(\d+(?:\.\d+)?)\s*%\s*\|?\s*$",
    re.IGNORECASE | re.MULTILINE)
MX_SMI_HBM_PATTERN = re.compile(
    r"^\s*\|?\s*throughput\s*(?:\||:)\s*"
    r"(\d+(?:\.\d+)?)\s*MBytes/s\s*\|?\s*$",
    re.IGNORECASE | re.MULTILINE)
MX_SMI_NO_PROCESS_PATTERN = re.compile(
    r"^[ \t]*\|?[ \t]*"
    r"(?:No process(?: found)?|No running processes(?: found)?)\.?"
    r"[ \t]*\|?[ \t]*$", re.IGNORECASE | re.MULTILINE)
GEOMETRIES = (
    ("control_x_1", 1, 1, 1, "control"),
    ("control_x_52", 52, 1, 1, "control"),
    ("control_x_104", 104, 1, 1, "control"),
    ("control_x_208", 208, 1, 1, "control"),
    ("rank1_x_512", 512, 1, 1, "rank1"),
    ("rank2_xy_512", 32, 16, 1, "rank2"),
    ("rank2_xz_512", 32, 1, 16, "rank2"),
    ("rank2_yz_512", 1, 32, 16, "rank2"),
    ("rank3_xyz_512", 32, 4, 4, "rank3"),
    ("rank3_yxz_512", 4, 32, 4, "rank3"),
    ("rank3_zyx_512", 4, 4, 32, "rank3"),
)
PLAN_FIELDS = (
    "sample_id", "trial", "order_position", "geometry_id",
    "grid_x", "grid_y", "grid_z",
)


def _repository_root():
    return Path(__file__).resolve().parents[4]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_contained(path, root, label):
    resolved = path.resolve()
    allowed = root.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"{label} must stay below {allowed}")
    return resolved


def _write_json_exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _copy_exclusive(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_stream, destination.open("xb") as target:
        shutil.copyfileobj(source_stream, target)


def _git_commit(repository_root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=False,
        timeout=TOOL_QUERY_TIMEOUT_SECONDS)
    commit = result.stdout.strip()
    if result.returncode != 0 or COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("cannot resolve exact baseline commit")
    return commit


def _verify_baseline_source_files(repository_root, baseline_commit):
    verified = []
    for repository_path in (SOURCE_RELATIVE_PATH, RUNNER_RELATIVE_PATH):
        result = subprocess.run(
            ["git", "show", f"{baseline_commit}:{repository_path}"],
            cwd=repository_root, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=TOOL_QUERY_TIMEOUT_SECONDS)
        live_path = repository_root / repository_path
        live_sha256 = _sha256(live_path)
        committed_sha256 = hashlib.sha256(result.stdout).hexdigest()
        if result.returncode != 0 or committed_sha256 != live_sha256:
            raise ValueError(
                f"{repository_path} is not the baseline commit content")
        verified.append({
            "repository_path": repository_path,
            "sha256": live_sha256,
        })
    return verified


def _snapshot_identity(files):
    identity = {
        "schema_version": SCHEMA_VERSION,
        "identity_algorithm": "sha256_canonical_json_v1",
        "files": [{
            "repository_path": item["repository_path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        } for item in files],
    }
    serialized = json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest(), identity


def freeze_source_snapshot(output_dir, repository_root, baseline_commit):
    snapshots = output_dir / "snapshots"
    source_paths = (SOURCE_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
    files = []
    for repository_path in source_paths:
        source = repository_root / repository_path
        destination = snapshots / Path(repository_path).name
        _copy_exclusive(source, destination)
        source_hash = _sha256(source)
        if _sha256(destination) != source_hash:
            raise ValueError("frozen source copy differs from repository bytes")
        files.append({
            "repository_path": repository_path,
            "snapshot_path": destination.relative_to(output_dir).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": source_hash,
        })
    snapshot_sha256, identity = _snapshot_identity(files)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": snapshot_sha256,
        "identity": identity,
        "files": files,
    }
    manifest_path = snapshots / "manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    return {
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": snapshot_sha256,
        "manifest_path": manifest_path.relative_to(output_dir).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "files": files,
    }


def _parse_launch_triplet(value, label):
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string triplet")
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"{label} must have three axes")
    parsed = [_positive_int(part, label) for part in parts]
    return parsed


def freeze_target_launch_evidence(output_dir, repository_root):
    dataset_path = repository_root / TARGET_DATASET_RELATIVE_PATH
    dataset_sha256 = _sha256(dataset_path)
    if dataset_sha256 != TARGET_DATASET_BLOB_SHA256:
        raise ValueError("canonical C500 FlashAttention blob identity changed")
    dataset = json.loads(
        zstandard.ZstdDecompressor().decompress(dataset_path.read_bytes()))
    case = dataset[TARGET_CASE_ID]
    attributes = case["op_info"]["Attr"]
    kernel = case["kernels"][TARGET_KERNEL_INDEX]
    metadata = kernel["Metadata"]
    launch = kernel["Sections"]["Launch Statistics"]["Header"]
    kernel_name = metadata["Kernel Name"]
    grid = _parse_launch_triplet(metadata["Grid Size"], "target grid")
    block = _parse_launch_triplet(metadata["Block Size"], "target block")
    registers = launch["Registers Per Thread [register/thread]"]
    dynamic_shared = launch[
        "Dynamic Shared Memory Per Block [byte/block]"]
    static_shared = launch["Static Shared Memory Per Block [byte/block]"]
    if (attributes["B"] != 4 or attributes["Q_LEN"] != 512
            or attributes["KV_LEN"] != 512 or attributes["HQ"] != 32
            or attributes["HKV"] != 8 or attributes["D"] != 128
            or attributes["SP"] != 1 or attributes["causal"] is not True
            or not isinstance(kernel_name, str)
            or "Flash_fwd_kernel_traits<128, 128, 64, 4" not in kernel_name
            or grid != [32, 4, 4] or block != [256, 1, 1]
            or registers != TARGET_REGISTERS_PER_THREAD
            or dynamic_shared != DYNAMIC_SHARED_BYTES
            or static_shared != 0):
        raise ValueError("frozen C500 target launch contract changed")
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "canonical_dataset_path": TARGET_DATASET_RELATIVE_PATH,
        "canonical_dataset_blob_sha256": dataset_sha256,
        "case_id": TARGET_CASE_ID,
        "kernel_index": TARGET_KERNEL_INDEX,
        "kernel_variant": "flash_fwd_kernel_q_tile_128",
        "kernel_name": kernel_name,
        "workload": {
            "batch": attributes["B"],
            "query_tokens": attributes["Q_LEN"],
            "kv_tokens": attributes["KV_LEN"],
            "query_heads": attributes["HQ"],
            "kv_heads": attributes["HKV"],
            "head_dim": attributes["D"],
            "sequence_parallel": attributes["SP"],
            "causal": attributes["causal"],
        },
        "head_dim": attributes["D"],
        "q_tile_tokens": 128,
        "grid": grid,
        "block": block,
        "total_ctas": math.prod(grid),
        "registers_per_thread": registers,
        "dynamic_shared_bytes": dynamic_shared,
        "static_shared_bytes": static_shared,
        "evidence_kind": "non_duration_launch_provenance",
        "duration_fields_read": False,
        "duration_fields_consumed": False,
        "scope": (
            "only the exact 238-register flash_fwd Q128-tile variant; "
            "not applicable to 216-, 218-, or 235-register variants"),
        "excluded_register_variants": [216, 218, 235],
    }
    path = output_dir / "target_launch_evidence.json"
    _write_json_exclusive(path, evidence)
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": _sha256(path),
        "evidence": evidence,
    }


def _artifact_manifest_payload(
        output_dir, baseline_commit, source_snapshot,
        target_launch_evidence, excluded_paths):
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact inventory contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative in excluded_paths:
            continue
        files.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot[
            "source_snapshot_sha256"],
        "source_snapshot_manifest_sha256": source_snapshot[
            "manifest_sha256"],
        "canonical_flash_attn_blob_sha256": TARGET_DATASET_BLOB_SHA256,
        "target_launch_evidence_sha256": target_launch_evidence["sha256"],
        "target_launch_duration_fields_consumed": False,
        "excluded_paths": sorted(excluded_paths),
        "files": files,
    }


def audit_source(path=None):
    if path is None:
        path = _repository_root() / SOURCE_RELATIVE_PATH
    source = path.read_text(encoding="utf-8")
    required = (
        "kBlockThreads = 256",
        "kDynamicSharedBytes = 32 * 1024",
        "kTargetRegistersPerThread = 238",
        "kRegisterPressureScalars = 224",
        "__launch_bounds__(kBlockThreads, 1)",
        "register_pressure(seed)",
        "atomicAdd(&counters->active, 1U)",
        "atomicMax(&counters->peak, active)",
        "atomicSub(&counters->active, 1U)",
        "clock64() - start < hold_cycles",
        "mcFuncGetAttributes(",
        "mcOccupancyMaxActiveBlocksPerMultiprocessor(",
        "mcRuntimeGetVersion(&runtime_version)",
        "mcDriverGetVersion(&driver_version)",
        "observed.geometry_mismatch != 0U",
        "observed.entered != static_cast<unsigned int>(total_ctas)",
        "observed.finished != static_cast<unsigned int>(total_ctas)",
    )
    if any(fragment not in source for fragment in required):
        raise ValueError("CTA admission source contract changed")
    forbidden = (
        "flash_attn", "paged_attention", "kernel_dataset/",
        "evaluation/", "mcEventElapsedTime", "mcProfilerStart",
        "mcProfilerStop",
    )
    if any(fragment in source for fragment in forbidden):
        raise ValueError("CTA admission source contains excluded workload logic")
    if "grid-wide barrier" not in source or "cannot deadlock" not in source:
        raise ValueError("CTA admission nonblocking protocol is undocumented")
    return path


def build_plan():
    rows = []
    base = list(GEOMETRIES)
    for trial in range(1, TRIALS + 1):
        ordered = list(base)
        random.Random(SCHEDULE_SEED + 7919 * trial).shuffle(ordered)
        if trial % 2 == 0:
            ordered.reverse()
        for geometry_id, grid_x, grid_y, grid_z, rank in ordered:
            order_position = len(rows)
            rows.append({
                "sample_id": f"t{trial:02d}_o{order_position:03d}_{geometry_id}",
                "trial": trial,
                "order_position": order_position,
                "geometry_id": geometry_id,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "grid_z": grid_z,
                "grid_rank": rank,
                "total_ctas": grid_x * grid_y * grid_z,
            })
    return tuple(rows)


def plan_text(plan):
    return "".join(
        " ".join(str(row[field]) for field in PLAN_FIELDS) + "\n"
        for row in plan
    )


def _positive_int(value, label):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if parsed <= 0 or str(parsed) != str(value):
        raise ValueError(f"{label} must be a canonical positive integer")
    return parsed


def _nonnegative_int(value, label):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if parsed < 0 or str(parsed) != str(value):
        raise ValueError(f"{label} must be a canonical non-negative integer")
    return parsed


def parse_capture(path, plan, device_id):
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    if len(rows) != len(plan) + 1:
        raise ValueError("CTA admission output row count is incomplete")
    metadata = rows[0]
    if len(metadata) != 19 or metadata[:4] != [
            "metadata", str(SCHEMA_VERSION), HARDWARE_ID, "cta_admission"]:
        raise ValueError("CTA admission metadata schema is invalid")
    observed_device = _nonnegative_int(metadata[4], "device_id")
    ap_count = _positive_int(metadata[6], "AP count")
    wave_size = _positive_int(metadata[7], "wave size")
    l2_bytes = _positive_int(metadata[8], "L2 bytes")
    block_threads = _positive_int(metadata[9], "block threads")
    dynamic_shared = _positive_int(metadata[10], "dynamic shared bytes")
    target_registers = _positive_int(metadata[11], "target registers")
    actual_registers = _positive_int(metadata[12], "actual registers")
    static_shared = _nonnegative_int(metadata[13], "static shared bytes")
    resident_blocks = _positive_int(metadata[14], "resident blocks per AP")
    hold_cycles = _positive_int(metadata[15], "admission hold cycles")
    pressure_scalars = _positive_int(metadata[16], "pressure scalars")
    runtime_version = _positive_int(metadata[17], "MACA runtime version")
    driver_version = _positive_int(metadata[18], "MACA driver version")
    if (observed_device != device_id or metadata[5] != "MetaX C500"
            or ap_count != AP_COUNT or wave_size != WAVE_SIZE
            or l2_bytes != L2_BYTES or block_threads != BLOCK_THREADS
            or dynamic_shared != DYNAMIC_SHARED_BYTES
            or target_registers != TARGET_REGISTERS_PER_THREAD
            or static_shared != 0 or hold_cycles != ADMISSION_HOLD_CYCLES
            or pressure_scalars != REGISTER_PRESSURE_SCALARS):
        raise ValueError("CTA admission device/resource contract changed")

    samples = []
    for output, expected in zip(rows[1:], plan):
        if len(output) != 13 or output[0] != "sample":
            raise ValueError("CTA admission sample schema is invalid")
        prefix = (
            expected["sample_id"], str(expected["trial"]),
            str(expected["order_position"]), expected["geometry_id"],
            str(expected["grid_x"]), str(expected["grid_y"]),
            str(expected["grid_z"]), str(expected["total_ctas"]),
        )
        if tuple(output[1:9]) != prefix:
            raise ValueError("CTA admission output differs from frozen plan")
        peak = _positive_int(output[9], "peak active CTAs")
        entered = _positive_int(output[10], "entered CTAs")
        finished = _positive_int(output[11], "finished CTAs")
        try:
            checksum = float(output[12])
        except ValueError as error:
            raise ValueError("CTA admission checksum is invalid") from error
        maximum_resident = min(
            expected["total_ctas"], resident_blocks * ap_count)
        if (entered != expected["total_ctas"]
                or finished != expected["total_ctas"]
                or peak > maximum_resident
                or not math.isfinite(checksum) or checksum <= 0.0):
            raise ValueError("CTA admission completion/admission invariant failed")
        samples.append({
            "sample_id": expected["sample_id"],
            "trial": expected["trial"],
            "order_position": expected["order_position"],
            "geometry_id": expected["geometry_id"],
            "grid_rank": expected["grid_rank"],
            "grid": [
                expected["grid_x"], expected["grid_y"],
                expected["grid_z"],
            ],
            "total_ctas": expected["total_ctas"],
            "peak_active_ctas": peak,
            "entered_ctas": entered,
            "finished_ctas": finished,
            "sink_checksum": checksum,
        })
    return {
        "device": {
            "hardware_id": HARDWARE_ID,
            "physical_device_id": observed_device,
            "device_name": metadata[5],
            "ap_count": ap_count,
            "wave_size": wave_size,
            "l2_bytes": l2_bytes,
            "runtime_version": runtime_version,
            "driver_version": driver_version,
        },
        "resources": {
            "block_threads": block_threads,
            "dynamic_shared_bytes": dynamic_shared,
            "static_shared_bytes": static_shared,
            "target_registers_per_thread": target_registers,
            "actual_registers_per_thread": actual_registers,
            "register_delta": actual_registers - target_registers,
            "exact_register_match": actual_registers == target_registers,
            "resident_blocks_per_ap": resident_blocks,
            "occupancy_upper_bound_ctas": resident_blocks * ap_count,
            "register_pressure_scalars": pressure_scalars,
        },
        "protocol": {
            "admission_hold_cycles": hold_cycles,
            "grid_wide_barrier": False,
            "deadlock_by_unscheduled_cta": False,
            "duration_recorded_or_consumed": False,
        },
        "samples": samples,
    }


def summarize(parsed):
    grouped = {}
    for geometry_id, grid_x, grid_y, grid_z, rank in GEOMETRIES:
        values = [
            sample["peak_active_ctas"] for sample in parsed["samples"]
            if sample["geometry_id"] == geometry_id
        ]
        if len(values) != TRIALS:
            raise ValueError("CTA admission geometry trial set is incomplete")
        grouped[geometry_id] = {
            "grid": [grid_x, grid_y, grid_z],
            "grid_rank": rank,
            "total_ctas": grid_x * grid_y * grid_z,
            "peak_active_ctas": values,
            "median_peak_active_ctas": statistics.median(values),
            "minimum_peak_active_ctas": min(values),
            "maximum_peak_active_ctas": max(values),
        }
    saturation = {
        geometry_id: summary["median_peak_active_ctas"]
        for geometry_id, summary in grouped.items()
        if summary["total_ctas"] == 512
    }
    rank_invariant = len(set(saturation.values())) == 1
    return {
        "geometries": grouped,
        "same_512_cta_rank_comparison": {
            "median_peak_active_ctas": saturation,
            "exactly_rank_invariant": rank_invariant,
            "maximum_median_delta": max(saturation.values())
            - min(saturation.values()),
        },
        "resource_match": {
            "status": "exact" if parsed["resources"][
                "exact_register_match"] else "mismatch",
            "target_registers_per_thread": TARGET_REGISTERS_PER_THREAD,
            "actual_registers_per_thread": parsed["resources"][
                "actual_registers_per_thread"],
            "register_delta": parsed["resources"]["register_delta"],
        },
        "candidate_eligible": (
            parsed["resources"]["exact_register_match"] and rank_invariant),
    }


class CommandExecutionError(RuntimeError):
    def __init__(self, message, execution):
        super().__init__(message)
        self.execution = execution


def _process_group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(process, expected_pid, expected_pgid):
    if process.pid != expected_pid or expected_pgid != expected_pid:
        raise RuntimeError("refusing inexact process-group cleanup")
    term_sent = False
    kill_sent = False
    if _process_group_exists(expected_pgid):
        try:
            observed_pgid = os.getpgid(expected_pid)
        except ProcessLookupError:
            observed_pgid = expected_pgid
        if observed_pgid != expected_pgid:
            raise RuntimeError("process group identity changed before cleanup")
        try:
            os.killpg(expected_pgid, signal.SIGTERM)
        except ProcessLookupError:
            term_sent = False
        else:
            term_sent = True
    deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS
    while _process_group_exists(expected_pgid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    if _process_group_exists(expected_pgid):
        try:
            os.killpg(expected_pgid, signal.SIGKILL)
        except ProcessLookupError:
            kill_sent = False
        else:
            kill_sent = True
    process.wait(timeout=TOOL_QUERY_TIMEOUT_SECONDS)
    deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS
    while _process_group_exists(expected_pgid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    if _process_group_exists(expected_pgid):
        raise RuntimeError("command process group is not quiescent")
    return {
        "pid": expected_pid,
        "pgid": expected_pgid,
        "sigterm_sent": term_sent,
        "sigkill_sent": kill_sent,
        "pgid_quiescent": True,
    }


def _run_checked(argv, stdout_path, stderr_path, cwd, timeout_seconds):
    if (not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool) or timeout_seconds <= 0):
        raise ValueError("command timeout must be positive")
    process = None
    returncode = None
    timed_out = False
    cleanup = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(
                argv, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True)
            pid = process.pid
            pgid = os.getpgid(pid)
            if pgid != pid:
                process.kill()
                process.wait(timeout=TOOL_QUERY_TIMEOUT_SECONDS)
                raise RuntimeError("command did not enter its own process group")
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            if process is not None:
                cleanup = _terminate_process_group(
                    process, process.pid, process.pid)
    execution = {
        "argv": list(argv),
        "timeout_seconds": timeout_seconds,
        "returncode": process.returncode if process is not None else None,
        "timed_out": timed_out,
        "cleanup": cleanup,
    }
    if timed_out:
        raise CommandExecutionError(
            f"command timed out after {timeout_seconds} seconds; "
            f"stdout={stdout_path}; stderr={stderr_path}", execution)
    if returncode != 0:
        raise CommandExecutionError(
            f"command failed with exit code {returncode}; "
            f"stdout={stdout_path}; stderr={stderr_path}", execution)
    return execution


def _parse_idle_outputs(usage, hbm, process, device_id, label):
    usage_devices = list(MX_SMI_DEVICE_PATTERN.finditer(usage))
    hbm_devices = list(MX_SMI_DEVICE_PATTERN.finditer(hbm))
    utilization = MX_SMI_GPU_USAGE_PATTERN.findall(usage)
    hbm_throughput = MX_SMI_HBM_PATTERN.findall(hbm)
    if (len(usage_devices) != 1 or len(hbm_devices) != 1
            or int(usage_devices[0].group("device")) != device_id
            or int(hbm_devices[0].group("device")) != device_id
            or usage_devices[0].group("bus").lower()
            != hbm_devices[0].group("bus").lower()
            or len(utilization) != 1 or float(utilization[0]) != 0.0
            or len(hbm_throughput) != 1
            or not math.isfinite(float(hbm_throughput[0]))
            or float(hbm_throughput[0]) < 0.0
            or float(hbm_throughput[0])
            > MAX_IDLE_HBM_THROUGHPUT_MBYTES_PER_SECOND
            or len(MX_SMI_NO_PROCESS_PATTERN.findall(process)) != 1):
        raise RuntimeError(f"C500 {label} check found a busy or wrong device")
    return {
        "device_id": device_id,
        "device_bus_id": usage_devices[0].group("bus").lower(),
        "gpu_utilization_pct": 0.0,
        "hbm_throughput_mbytes_per_second": float(hbm_throughput[0]),
        "maximum_idle_hbm_throughput_mbytes_per_second": (
            MAX_IDLE_HBM_THROUGHPUT_MBYTES_PER_SECOND),
        "processes": [],
    }


def _idle_snapshot(output_dir, label, device_id, repository_root):
    usage_stdout = output_dir / f"{label}-usage.stdout.txt"
    usage_stderr = output_dir / f"{label}-usage.stderr.txt"
    process_stdout = output_dir / f"{label}-process.stdout.txt"
    process_stderr = output_dir / f"{label}-process.stderr.txt"
    hbm_stdout = output_dir / f"{label}-hbm.stdout.txt"
    hbm_stderr = output_dir / f"{label}-hbm.stderr.txt"
    usage_execution = _run_checked(
        [str(MX_SMI_PATH), "--show-usage", "-i", str(device_id)],
        usage_stdout, usage_stderr, repository_root,
        TOOL_QUERY_TIMEOUT_SECONDS)
    hbm_execution = _run_checked(
        [str(MX_SMI_PATH), "--show-hbm-bandwidth", "-i", str(device_id)],
        hbm_stdout, hbm_stderr, repository_root,
        TOOL_QUERY_TIMEOUT_SECONDS)
    process_execution = _run_checked(
        [str(MX_SMI_PATH), "--show-process", "-i", str(device_id)],
        process_stdout, process_stderr, repository_root,
        TOOL_QUERY_TIMEOUT_SECONDS)
    usage = usage_stdout.read_text(encoding="utf-8")
    hbm = hbm_stdout.read_text(encoding="utf-8")
    process = process_stdout.read_text(encoding="utf-8")
    parsed = _parse_idle_outputs(usage, hbm, process, device_id, label)
    parsed["commands"] = {
        "usage": usage_execution,
        "hbm": hbm_execution,
        "process": process_execution,
    }
    parsed["raw"] = {
        "usage_stdout_path": usage_stdout.relative_to(output_dir).as_posix(),
        "usage_stdout_sha256": _sha256(usage_stdout),
        "usage_stderr_path": usage_stderr.relative_to(output_dir).as_posix(),
        "usage_stderr_sha256": _sha256(usage_stderr),
        "hbm_stdout_path": hbm_stdout.relative_to(output_dir).as_posix(),
        "hbm_stdout_sha256": _sha256(hbm_stdout),
        "hbm_stderr_path": hbm_stderr.relative_to(output_dir).as_posix(),
        "hbm_stderr_sha256": _sha256(hbm_stderr),
        "process_stdout_path": process_stdout.relative_to(
            output_dir).as_posix(),
        "process_stdout_sha256": _sha256(process_stdout),
        "process_stderr_path": process_stderr.relative_to(
            output_dir).as_posix(),
        "process_stderr_sha256": _sha256(process_stderr),
    }
    return parsed


def _capture_tool_version(tool, label, output_dir, repository_root):
    stdout_path = output_dir / f"{label}-version.stdout.txt"
    stderr_path = output_dir / f"{label}-version.stderr.txt"
    execution = _run_checked(
        [str(tool), "--version"], stdout_path, stderr_path,
        repository_root, TOOL_QUERY_TIMEOUT_SECONDS)
    version = stdout_path.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"{label} version output is empty")
    return {
        "path": str(tool),
        "executable_sha256": _sha256(tool),
        "version": version,
        "execution": execution,
        "stdout_path": stdout_path.relative_to(output_dir).as_posix(),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_path": stderr_path.relative_to(output_dir).as_posix(),
        "stderr_sha256": _sha256(stderr_path),
    }


def _run_capture_with_postflight(capture_action, postflight_action):
    capture_execution = None
    postflight = None
    try:
        capture_execution = capture_action()
    finally:
        postflight = postflight_action()
    return capture_execution, postflight


def capture(args):
    repository_root = _repository_root()
    source_path = audit_source(repository_root / SOURCE_RELATIVE_PATH)
    source_sha256 = _sha256(source_path)
    if source_sha256 != args.expected_source_sha256:
        raise ValueError("CTA admission source SHA-256 differs from deployment")
    baseline_commit = _git_commit(repository_root)
    if baseline_commit != args.baseline_commit:
        raise ValueError("CTA admission baseline commit differs from deployment")
    baseline_source_files = _verify_baseline_source_files(
        repository_root, baseline_commit)
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run-id is invalid")
    if args.device_id < 0:
        raise ValueError("device-id must be non-negative")
    output_root = repository_root / OUTPUT_RELATIVE_ROOT
    output_dir = _require_contained(
        output_root / args.run_id, output_root, "output directory")
    output_dir.mkdir(parents=True, exist_ok=False)

    source_snapshot = freeze_source_snapshot(
        output_dir, repository_root, baseline_commit)
    target_launch_evidence = freeze_target_launch_evidence(
        output_dir, repository_root)
    if source_snapshot["files"][0]["sha256"] != source_sha256:
        raise ValueError("source snapshot is not bound to expected source")
    compiler_version = _capture_tool_version(
        MXCC_PATH, "mxcc", output_dir, repository_root)
    driver_tool_version = _capture_tool_version(
        MX_SMI_PATH, "mx-smi", output_dir, repository_root)

    plan = build_plan()
    plan_path = output_dir / "plan.txt"
    plan_path.write_text(plan_text(plan), encoding="utf-8")
    compile_stdout = output_dir / "compile.stdout.txt"
    compile_stderr = output_dir / "compile.stderr.txt"
    binary_path = output_dir / "cta_admission"
    frozen_source_path = output_dir / source_snapshot["files"][0][
        "snapshot_path"]
    compile_argv = [
        str(MXCC_PATH), "-x", "maca", "-offload-arch=native",
        "--maca-path=/opt/maca", "-O3", str(frozen_source_path),
        "-o", str(binary_path),
    ]
    compile_execution = _run_checked(
        compile_argv, compile_stdout, compile_stderr, repository_root,
        COMPILE_TIMEOUT_SECONDS)
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise FileNotFoundError("mxcc did not produce an executable")

    preflight = _idle_snapshot(
        output_dir, "preflight", args.device_id, repository_root)
    capture_stdout = output_dir / "capture.stdout.csv"
    capture_stderr = output_dir / "capture.stderr.txt"
    capture_argv = [
        str(binary_path), "--device-id", str(args.device_id),
        "--plan", str(plan_path),
    ]
    capture_execution, postflight = _run_capture_with_postflight(
        lambda: _run_checked(
            capture_argv, capture_stdout, capture_stderr,
            repository_root, CAPTURE_TIMEOUT_SECONDS),
        lambda: _idle_snapshot(
            output_dir, "postflight", args.device_id, repository_root),
    )
    if preflight["device_bus_id"] != postflight["device_bus_id"]:
        raise ValueError("C500 device identity changed during capture")

    parsed = parse_capture(capture_stdout, plan, args.device_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "baseline_commit": baseline_commit,
        "baseline_source_files": baseline_source_files,
        "source_snapshot": source_snapshot,
        "target_launch_evidence": {
            "path": target_launch_evidence["path"],
            "sha256": target_launch_evidence["sha256"],
            "canonical_flash_attn_blob_sha256": (
                TARGET_DATASET_BLOB_SHA256),
            "duration_fields_consumed": False,
        },
        "source_relative_path": SOURCE_RELATIVE_PATH,
        "source_sha256": source_sha256,
        "runner_relative_path": RUNNER_RELATIVE_PATH,
        "runner_sha256": source_snapshot["files"][1]["sha256"],
        "software_versions": {
            "maca_runtime_version": parsed["device"]["runtime_version"],
            "maca_driver_version": parsed["device"]["driver_version"],
            "compiler": compiler_version,
            "driver_tool": driver_tool_version,
        },
        "build": {
            "argv": compile_argv,
            "execution": compile_execution,
            "source_snapshot_path": source_snapshot["files"][0][
                "snapshot_path"],
            "source_snapshot_file_sha256": source_sha256,
            "compiler_path": str(MXCC_PATH),
            "compiler_sha256": _sha256(MXCC_PATH),
            "stdout_path": compile_stdout.relative_to(output_dir).as_posix(),
            "stdout_sha256": _sha256(compile_stdout),
            "stderr_path": compile_stderr.relative_to(output_dir).as_posix(),
            "stderr_sha256": _sha256(compile_stderr),
            "binary_path": binary_path.relative_to(output_dir).as_posix(),
            "binary_sha256": _sha256(binary_path),
        },
        "capture": {
            "argv": capture_argv,
            "execution": capture_execution,
            "stdout_path": capture_stdout.relative_to(output_dir).as_posix(),
            "stdout_sha256": _sha256(capture_stdout),
            "stderr_path": capture_stderr.relative_to(output_dir).as_posix(),
            "stderr_sha256": _sha256(capture_stderr),
        },
        "plan_sha256": _sha256(plan_path),
        "preflight": preflight,
        "postflight": postflight,
        "measurement": parsed,
        "summary": summarize(parsed),
        "production_operator_duration_consumed": False,
        "mcprofiler_instrumented_duration_consumed": False,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    result_path = output_dir / "result.json"
    manifest = _artifact_manifest_payload(
        output_dir, baseline_commit, source_snapshot,
        target_launch_evidence, {manifest_path.name, result_path.name})
    _write_json_exclusive(manifest_path, manifest)
    payload["artifact_manifest_path"] = manifest_path.relative_to(
        output_dir).as_posix()
    payload["artifact_manifest_sha256"] = _sha256(manifest_path)
    _write_json_exclusive(result_path, payload)
    return result_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--json", action="store_true")
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--device-id", type=int, required=True)
    capture_parser.add_argument("--run-id", required=True)
    capture_parser.add_argument("--baseline-commit", required=True)
    capture_parser.add_argument(
        "--expected-source-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_plan()
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print(plan_text(plan), end="")
        return 0
    if args.command == "capture":
        if COMMIT_PATTERN.fullmatch(args.baseline_commit) is None:
            raise ValueError("baseline-commit must be lowercase 40-hex")
        if SHA256_PATTERN.fullmatch(args.expected_source_sha256) is None:
            raise ValueError("expected-source-sha256 must be lowercase SHA-256")
        print(capture(args))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
