"""Attach duration-free CANN GEMM launches to an Ascend Kernel Dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping


_CAPTURE_FIELDS = {
    "artifact_type", "baseline_commit", "build_and_abi_audit",
    "cann_abi_provenance", "cann_version", "cube_math_type", "device_id",
    "device_kernel_submitted", "duration_consumed", "execute_api_called",
    "full_input_operation_count", "hardware_id", "input_rank",
    "kernel_dataset", "limitations", "mode", "operation_count",
    "operations", "operator", "runner_version", "schema_version",
    "source_provenance",
}
_OPERATION_FIELDS = {
    "b", "capture", "capture_sha256", "cube_math_type", "dtype",
    "input_rank", "k", "logical_shape", "m", "n", "operation_id",
    "workload_keys",
}
_INNER_CAPTURE_FIELDS = {
    "batch_size", "capture_storage", "cube_math_type",
    "device_kernel_submitted", "dry_launch", "dtype", "duration_consumed",
    "execute_api_called", "executor_lifecycle", "executor_tiling",
    "input_rank", "layout", "operation_id", "operator",
    "process_peak_rss_kib", "schema_version", "shape", "workload_keys",
    "workspace_size_bytes",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    return value


def _exact_mapping(
        value: object, fields: set[str], context: str) -> Mapping[str, object]:
    mapping = _mapping(value, context)
    if set(mapping) != fields:
        raise ValueError(f"{context} fields differ")
    return mapping


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def attach_gemm_launch_observations(
        dataset: Mapping[str, object], capture: Mapping[str, object], *,
        capture_artifact_sha256: str) -> dict[str, object]:
    """Return a dataset copy carrying each measured operation's raw payload."""

    _exact_mapping(capture, _CAPTURE_FIELDS, "Ascend GEMM launch capture")
    if (_SHA256.fullmatch(capture_artifact_sha256) is None
            or capture["schema_version"] != 1
            or capture["artifact_type"]
            != "ascend_910b_gemm_kernel_dataset_dry_launch_probe"
            or capture["hardware_id"] != "ascend_910b"
            or capture["cann_version"] != "8.5.1"
            or capture["mode"] != "kernel_dataset_probe"
            or capture["operator"] != "aclnnMatmul"
            or capture["input_rank"] != 2
            or capture["duration_consumed"] is not False
            or capture["device_kernel_submitted"] is not False
            or capture["execute_api_called"] is not True):
        raise ValueError("Ascend GEMM launch capture contract differs")
    baseline = capture["baseline_commit"]
    if not isinstance(baseline, str) or _COMMIT.fullmatch(baseline) is None:
        raise ValueError("Ascend GEMM launch capture baseline is not a commit")
    operations = capture["operations"]
    if not isinstance(operations, list):
        raise TypeError("Ascend GEMM launch capture operations must be a list")
    if (capture["operation_count"] != len(operations)
            or capture["full_input_operation_count"] != len(operations)
            or len(operations) != len(dataset)):
        raise ValueError("Ascend GEMM launch capture coverage differs")

    result = copy.deepcopy(dict(dataset))
    seen = set()
    for index, raw_operation in enumerate(operations):
        operation = _exact_mapping(
            raw_operation, _OPERATION_FIELDS,
            f"Ascend GEMM launch operation[{index}]")
        operation_id = operation["operation_id"]
        if (not isinstance(operation_id, str) or not operation_id
                or operation_id in seen or operation_id not in result):
            raise ValueError("Ascend GEMM launch operation identity differs")
        seen.add(operation_id)
        inner = _exact_mapping(
            operation["capture"], _INNER_CAPTURE_FIELDS,
            f"Ascend GEMM launch operation {operation_id} capture")
        capture_sha256 = operation["capture_sha256"]
        if (not isinstance(capture_sha256, str)
                or _SHA256.fullmatch(capture_sha256) is None
                or _sha256_bytes(_canonical_json_bytes(inner))
                != capture_sha256):
            raise ValueError(
                f"Ascend GEMM launch operation {operation_id} SHA256 changed")
        if (inner["duration_consumed"] is not False
                or inner["device_kernel_submitted"] is not False
                or inner["execute_api_called"] is not True
                or inner["input_rank"] != 2
                or inner["operation_id"] != operation_id):
            raise ValueError(
                f"Ascend GEMM launch operation {operation_id} contract differs")
        entry = _mapping(result[operation_id], f"Dataset {operation_id}")
        attr = _mapping(
            _mapping(entry["op_info"], f"Dataset {operation_id}.op_info")["Attr"],
            f"Dataset {operation_id}.Attr")
        logical_shape = {
            "b": attr["B"], "m": attr["M"], "n": attr["N"], "k": attr["K"],
        }
        if operation["logical_shape"] != logical_shape:
            raise ValueError(
                f"Ascend GEMM launch operation {operation_id} shape differs")
        kernels = entry["kernels"]
        if not isinstance(kernels, list) or len(kernels) != 1:
            raise ValueError(
                f"Ascend GEMM Dataset {operation_id} must contain one kernel")
        kernel = _mapping(kernels[0], f"Dataset {operation_id}.kernel")
        metadata = _mapping(
            kernel["Metadata"], f"Dataset {operation_id}.Metadata")
        ascend = _mapping(kernel["Ascend"], f"Dataset {operation_id}.Ascend")
        if "gemm_launch" in ascend:
            raise ValueError(
                f"Ascend GEMM Dataset {operation_id} already has gemm_launch")
        dry_launch = _mapping(
            inner["dry_launch"], f"Ascend GEMM launch {operation_id}.dry_launch")
        launch = _mapping(
            dry_launch["launch"], f"Ascend GEMM launch {operation_id}.launch")
        tiling = _mapping(
            inner["executor_tiling"],
            f"Ascend GEMM launch {operation_id}.executor_tiling")
        symbol = launch["kernel_stub"]
        block_dim = launch["block_dim"]
        observed_symbol = metadata["Kernel Name"]
        if (observed_symbol not in (symbol, f"{symbol}_mix_aic")
                or block_dim != ascend["block_dim"]):
            raise ValueError(
                f"Ascend GEMM launch {operation_id} disagrees with profiler")
        payload_hex = tiling["serialized_hex"]
        payload_sha256 = tiling["sha256"]
        if not isinstance(payload_hex, str) or not isinstance(payload_sha256, str):
            raise TypeError(
                f"Ascend GEMM launch {operation_id} payload fields differ")
        try:
            payload = bytes.fromhex(payload_hex)
        except ValueError as error:
            raise ValueError(
                f"Ascend GEMM launch {operation_id} payload is not hexadecimal") \
                from error
        if (_SHA256.fullmatch(payload_sha256) is None
                or _sha256_bytes(payload) != payload_sha256):
            raise ValueError(
                f"Ascend GEMM launch {operation_id} payload SHA256 changed")
        ascend["gemm_launch"] = {
            "schema_version": 1,
            "software_revision": "cann_8.5.1",
            "input_rank": 2,
            "kernel_family": launch["kernel_family"],
            "kernel_symbol": symbol,
            "block_dim": block_dim,
            "tiling_payload_hex": payload_hex,
            "tiling_payload_sha256": payload_sha256,
            "capture_baseline_commit": baseline,
            "capture_artifact_sha256": capture_artifact_sha256,
            "capture_sha256": capture_sha256,
            "duration_consumed": False,
            "device_kernel_submitted": False,
        }
    if seen != set(result):
        raise ValueError("Ascend GEMM launch capture does not cover the Dataset")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach a duration-free CANN launch capture to GEMM data")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset_path = Path(args.dataset).resolve()
    capture_path = Path(args.capture).resolve()
    output_path = Path(args.output).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    capture_bytes = capture_path.read_bytes()
    capture = json.loads(capture_bytes)
    result = attach_gemm_launch_observations(
        dataset, capture,
        capture_artifact_sha256=_sha256_bytes(capture_bytes))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", delete=False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
