#!/usr/bin/env python3
"""Validate and decode CANN 8.5.1 AddRMSNorm launch-only captures."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


EXPECTED_CASES = {
    "T1_H16384": 1,
    "T64_H16384": 64,
    "T256_H16384": 256,
    "T1024_H16384": 1024,
}
EXPECTED_EXECUTIONS = 3
EXPECTED_HIDDEN = 16384
EXPECTED_KERNEL = "AddRmsNorm_3_high_performance_31"
EXPECTED_FIELDS = (
    ("num_row", "uint32_t", 0),
    ("num_col", "uint32_t", 4),
    ("block_factor", "uint32_t", 8),
    ("row_factor", "uint32_t", 12),
    ("ub_factor", "uint32_t", 16),
    ("epsilon", "float", 20),
    ("avg_factor", "float", 24),
    ("num_col_align", "uint32_t", 28),
    ("last_block_factor", "uint32_t", 32),
    ("row_loop", "uint32_t", 36),
    ("last_block_row_loop", "uint32_t", 40),
    ("row_tail", "uint32_t", 44),
    ("last_block_row_tail", "uint32_t", 48),
    ("mul_loop_fp32", "uint32_t", 52),
    ("mul_tail_fp32", "uint32_t", 56),
    ("dst_rep_stride_fp32", "uint32_t", 60),
    ("mul_loop_fp16", "uint32_t", 64),
    ("mul_tail_fp16", "uint32_t", 68),
    ("dst_rep_stride_fp16", "uint32_t", 72),
    ("is_performance", "uint32_t", 76),
)
EXPECTED_TILING_BYTES = 80
EXPECTED_PLACEHOLDER = {"address_offset": 56, "data_offset": 80}
EXPECTED_ARGS_BYTES = 192
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
STRUCT_CODES = {"float": "f", "uint32_t": "I"}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_uint(value, context, *, positive=False):
    if not value.isdecimal():
        raise ValueError(f"{context} must be an unsigned decimal integer")
    parsed = int(value)
    if positive and parsed <= 0:
        raise ValueError(f"{context} must be positive")
    return parsed


def _metadata(path):
    values = {}
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or "=" not in line:
            raise ValueError(f"invalid metadata at {path}:{line_number}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"duplicate metadata key at {path}:{line_number}")
        values[key] = value
    return values


def parse_reflection(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 3 + len(EXPECTED_FIELDS):
        raise ValueError("AddRMSNorm reflection line count changed")
    if lines[0] != "SCHEMA\t1":
        raise ValueError("AddRMSNorm reflection schema must be 1")
    if lines[1].split("\t") != [
            "ROOT", "AddRmsNorm", "AddRMSNormTilingData", "80", "1", "20"]:
        raise ValueError("AddRMSNorm reflection root changed")
    if lines[2].split("\t") != [
            "STRUCT", "$", "AddRMSNormTilingData", "0", "80", "20"]:
        raise ValueError("AddRMSNorm reflected struct changed")
    for line, (name, dtype, offset) in zip(lines[3:], EXPECTED_FIELDS):
        if line.split("\t") != [
                "FIELD", name, dtype, str(offset), "1", "4"]:
            raise ValueError(f"AddRMSNorm reflected field changed: {name}")
    return {"path": str(path), "sha256": _sha256(path)}


def _decode_payload(payload):
    if len(payload) != EXPECTED_TILING_BYTES:
        raise ValueError("AddRMSNorm tiling payload must be 80 bytes")
    decoded = {}
    for name, dtype, offset in EXPECTED_FIELDS:
        code = STRUCT_CODES[dtype]
        decoded[name] = struct.unpack_from("<" + code, payload, offset)[0]
    return decoded


def _parse_case(case_directory, rows):
    metadata = [_metadata(path) for path in sorted(case_directory.glob("*.meta"))]
    workspace = [item for item in metadata if item["record_type"] == "workspace"]
    launches = [item for item in metadata if item["record_type"] == "launch_args"]
    if len(workspace) != EXPECTED_EXECUTIONS or len(launches) != EXPECTED_EXECUTIONS:
        raise ValueError(f"{case_directory.name} must contain three execute/launch pairs")
    decoded_runs = []
    payload_hashes = []
    process_ids = set()
    block_dims = set()
    for execute, launch in zip(workspace, launches):
        if set(execute) != {
                "schema_version", "record_type", "sequence", "api", "pid",
                "tid", "workspace_size", "execute_stream"}:
            raise ValueError("AddRMSNorm workspace metadata fields changed")
        if set(launch) != {
                "schema_version", "record_type", "sequence", "api",
                "execute_api", "pid", "tid", "stub_identity_kind",
                "stub_address", "stub_runtime_name_status",
                "stub_runtime_function_name", "stub_elf_membership",
                "stub_elf_object_path", "stub_elf_object_base",
                "stub_elf_object_offset", "stub_elf_symbol",
                "stub_elf_symbol_address", "stub_elf_symbol_offset",
                "block_dim", "args_size", "args_file", "placeholder_count",
                "placeholder_entry_size", "placeholder_file",
                "execute_stream", "launch_stream"}:
            raise ValueError("AddRMSNorm launch metadata fields changed")
        if execute["schema_version"] != "1" or launch["schema_version"] != "2":
            raise ValueError("AddRMSNorm capture schema changed")
        if execute["api"] != "aclnnAddRmsNorm" or launch["execute_api"] != "aclnnAddRmsNorm":
            raise ValueError("capture is outside aclnnAddRmsNorm")
        if launch["api"] != "aclrtLaunchKernelWithHostArgs":
            raise ValueError("AddRMSNorm launch API changed")
        if launch["stub_identity_kind"] != "runtime_function_name" or launch["stub_runtime_name_status"] != "0":
            raise ValueError("AddRMSNorm kernel lacks runtime name identity")
        if launch["stub_runtime_function_name"] != EXPECTED_KERNEL:
            raise ValueError("AddRMSNorm did not select BF16 Split-D key 31")
        if _parse_uint(execute["sequence"], "execute sequence", positive=True) + 1 != _parse_uint(launch["sequence"], "launch sequence", positive=True):
            raise ValueError("AddRMSNorm execute/launch sequence is not adjacent")
        for field in ("pid", "tid", "execute_stream"):
            if execute[field] != launch[field]:
                raise ValueError(f"AddRMSNorm execute/launch {field} differs")
        if launch["execute_stream"] != launch["launch_stream"]:
            raise ValueError("AddRMSNorm execute/launch stream differs")
        if launch["args_size"] != str(EXPECTED_ARGS_BYTES):
            raise ValueError("AddRMSNorm host argument size changed")
        if launch["placeholder_count"] != "1" or launch["placeholder_entry_size"] != "8":
            raise ValueError("AddRMSNorm placeholder contract changed")
        args_path = case_directory / launch["args_file"]
        placeholder_path = case_directory / launch["placeholder_file"]
        args = args_path.read_bytes()
        if len(args) != EXPECTED_ARGS_BYTES:
            raise ValueError("AddRMSNorm host argument file size changed")
        placeholder_bytes = placeholder_path.read_bytes()
        if len(placeholder_bytes) != 8:
            raise ValueError("AddRMSNorm placeholder file size changed")
        address_offset, data_offset = struct.unpack("<II", placeholder_bytes)
        if {"address_offset": address_offset, "data_offset": data_offset} != EXPECTED_PLACEHOLDER:
            raise ValueError("AddRMSNorm placeholder offsets changed")
        payload = args[data_offset:data_offset + EXPECTED_TILING_BYTES]
        decoded_runs.append(_decode_payload(payload))
        payload_hashes.append(hashlib.sha256(payload).hexdigest())
        process_ids.add(_parse_uint(launch["pid"], "launch pid", positive=True))
        block_dims.add(_parse_uint(launch["block_dim"], "block dim", positive=True))
    if len(process_ids) != 1 or len(block_dims) != 1:
        raise ValueError("AddRMSNorm process or block_dim varies across repetitions")
    if any(item != decoded_runs[0] for item in decoded_runs[1:]):
        raise ValueError("AddRMSNorm tiling payload varies across repetitions")
    if len(set(payload_hashes)) != 1:
        raise ValueError("AddRMSNorm serialized payload varies across repetitions")
    decoded = decoded_runs[0]
    if decoded["num_row"] != rows or decoded["num_col"] != EXPECTED_HIDDEN:
        raise ValueError("AddRMSNorm decoded shape differs from probe shape")
    integer_tiling = {
        name: decoded[name] for name, dtype, _ in EXPECTED_FIELDS
        if dtype == "uint32_t"
    }
    return {
        "rows": rows,
        "hidden": EXPECTED_HIDDEN,
        "dtype": "bf16",
        "tiling_key": 31,
        "kernel_symbol": EXPECTED_KERNEL,
        "launch_api": "aclrtLaunchKernelWithHostArgs",
        "launch_geometry": {
            "core_type": "AIV",
            "block_dim": next(iter(block_dims)),
        },
        "tiling": integer_tiling,
        "tiling_payload_bytes": EXPECTED_TILING_BYTES,
        "tiling_payload_sha256": payload_hashes[0],
        "host_args_bytes": EXPECTED_ARGS_BYTES,
        "placeholder": EXPECTED_PLACEHOLDER,
        "repetitions": EXPECTED_EXECUTIONS,
        "capture_process_id": next(iter(process_ids)),
    }


def _source_identity(cann_root):
    relative_paths = (
        "opp/version.info",
        "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_nn.so",
        "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/add_rms_norm/add_rms_norm.cpp",
        "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/add_rms_norm/add_rms_norm_split_d.h",
        "opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/rms_norm/rms_norm_base.h",
        "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_nn/add_rms_norm/AddRmsNorm_3_high_performance.o",
        "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_nn/add_rms_norm/AddRmsNorm_3_high_performance.json",
    )
    files = {}
    for relative in relative_paths:
        path = cann_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required CANN identity file is missing: {path}")
        files[relative] = _sha256(path)
    if (cann_root / "opp/version.info").read_text(encoding="utf-8").splitlines()[0] != "Version=8.5.1":
        raise ValueError("AddRMSNorm launch probe requires CANN 8.5.1")
    return {"cann_root": str(cann_root), "files_sha256": files}


def build_report(raw_root, reflection_path, cann_root, baseline_commit,
                 owner, host_device_id, container_device_id):
    if COMMIT_PATTERN.fullmatch(baseline_commit) is None:
        raise ValueError("baseline commit must be 40 lowercase hex characters")
    reflection = parse_reflection(reflection_path)
    cases = {
        case_key: _parse_case(raw_root / case_key, rows)
        for case_key, rows in EXPECTED_CASES.items()
    }
    return {
        "schema_version": 1,
        "artifact_type": "ascend_910b_add_rmsnorm_splitd_launch_facts",
        "hardware_id": "ascend_910b",
        "software_revision": "CANN 8.5.1",
        "production_or_evaluation_duration_consumed": False,
        "target_kernel_submitted": False,
        "baseline_commit": baseline_commit,
        "capture_owner": owner,
        "capture_command": (
            f"ASCEND_RT_VISIBLE_DEVICES={host_device_id} "
            "LD_PRELOAD=<built add_rmsnorm launch interposer> python "
            "add_rmsnorm_launch_application.py "
            f"--output-root {raw_root} --device-id {container_device_id}"),
        "raw_output_path": str(raw_root),
        "device_binding": {
            "host_device_id": host_device_id,
            "container_device_id": container_device_id,
            "environment": f"ASCEND_RT_VISIBLE_DEVICES={host_device_id}",
        },
        "workload_contract": {
            "rows": list(EXPECTED_CASES.values()),
            "hidden": EXPECTED_HIDDEN,
            "dtype": "bf16",
            "epsilon": "1e-6",
            "repetitions_per_case": EXPECTED_EXECUTIONS,
        },
        "reflection": {
            "registration_key": "AddRmsNorm",
            "class_name": "AddRMSNormTilingData",
            "data_size_bytes": EXPECTED_TILING_BYTES,
            "sha256": reflection["sha256"],
        },
        "source_and_binary_identity": _source_identity(cann_root),
        "cases": cases,
        "limitations": [
            "launch-only interposition suppresses target kernel submission",
            "facts cover only BF16 H=16384 at T=1,64,256,1024",
            "no duration, counter, utilization, or evaluation error is captured or consumed",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reflection-tsv", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--host-device-id", type=int, required=True)
    parser.add_argument("--container-device-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        args.raw_root.resolve(), args.reflection_tsv.resolve(),
        args.cann_root.resolve(), args.baseline_commit, args.owner,
        args.host_device_id, args.container_device_id)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
