#!/usr/bin/env python3
"""Observe synthetic NPU INT32->INT64 dispatch, never an operator rate.

Run under msprof for raw launch/tiling facts, then project only explicitly
allowlisted non-duration columns. The original profiler files stay unchanged.
No measured whole-Cast duration is eligible as a Forward parameter.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import sys


ELEMENT_COUNTS = (1, 127, 128, 129, 2048, 32768)
PATTERN = (
    -(2 ** 31), -(2 ** 31) + 1, -65537, -65536, -1, 0, 1,
    65535, 65536, 65537, 2 ** 31 - 2, 2 ** 31 - 1)
# Exact spellings emitted by known msprof op_summary exporters. Unknown
# columns are ignored, never interpreted as a timing or arbitrary launch key.
DISCRETE_COLUMNS = (
    "Op Name", "OP Name", "Op Type", "OP Type", "Task Type",
    "Block Dim", "Block Dim.", "Mix Block Dim", "Tiling Key",
    "Input Shapes", "Input Data Types", "Input Formats",
    "Output Shapes", "Output Data Types", "Output Formats",
    "Model ID", "Task ID", "Stream ID", "Device_id", "Device ID",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    if path.exists():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _values(elements):
    return [PATTERN[index % len(PATTERN)] for index in range(elements)]


def _project(path, output):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("op_summary CSV has no header")
        fields = tuple(name for name in reader.fieldnames
                       if name in DISCRETE_COLUMNS)
        if "Block Dim" not in fields and "Block Dim." not in fields:
            raise ValueError("raw CSV lacks a recognized Block Dim column")
        if not any(name in fields for name in ("Op Name", "OP Name")):
            raise ValueError("raw CSV lacks a recognized Op Name column")
        rows = []
        for raw in reader:
            rows.append({name: raw[name] for name in fields})
    if not rows:
        raise ValueError("op_summary CSV contains no observations")
    _write(output, {
        "schema_version": 1,
        "scope": "synthetic_operator_discrete_observations_only",
        "raw_file": str(path.resolve()),
        "raw_sha256": _sha256(path),
        "columns": fields,
        "rows": rows,
        "duration_consumed": False,
        "full_operator_rate_eligible": False,
        "actual_serving_dispatch_verified": False,
        "production_binding_verified": False,
    })


def _project_host_args(path, compile_info_path, output):
    """Decode only the inline variables described by the selected object.

    The public placeholder identifies the inline payload, while compileInfo
    supplies its field order and 64-bit mode. Pointer slots and the trailing
    runtime marker are deliberately not interpreted as tiling variables.
    These are observations for the six synthetic shapes, never a formula for
    other shapes or an instruction-rate measurement.
    """
    metadata = json.loads(compile_info_path.read_text())
    info = metadata["compile_info"]
    key = str(metadata["runtime_tiling_key"])
    if info["_int64_mode"] is not True:
        raise ValueError("host-argument projection requires explicit int64 compileInfo")
    fields = info["_vars"][key]
    if fields != ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]:
        raise ValueError("unknown Cast inline variable layout")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 2 * len(ELEMENT_COUNTS):
        raise ValueError("expected exactly two synthetic launches per element count")
    projected = []
    for index, row in enumerate(rows):
        if (row["api"] != "aclrtLaunchKernelWithHostArgs"
                or row["kernel_name"] != metadata["runtime_kernel_function"]
                or row["target"] is not True or row["duration_recorded"] is not False):
            raise ValueError("host capture is not the exact selected non-duration Cast")
        payload = bytes.fromhex(row["host_args_hex"])
        if len(payload) != row["args_size"] or len(payload) > 4096:
            raise ValueError("invalid captured host argument extent")
        if len(row["placeholders"]) != 1:
            raise ValueError("Cast requires one inline tiling placeholder")
        placeholder, = row["placeholders"]
        address, data = placeholder["addr_offset"], placeholder["data_offset"]
        if (type(address) is not int or type(data) is not int
                or address < 0 or address + 8 > len(payload)
                or data < address + 8 or data + 8 * len(fields) > len(payload)):
            raise ValueError("inline tiling placeholder is outside its captured extent")
        values = struct.unpack_from("<qqq", payload, data)
        elements = ELEMENT_COUNTS[index // 2]
        dimension, block_factor, ub_factor = values
        blocks = row["block_dim"]
        if (dimension != elements or min(values) <= 0 or type(blocks) is not int
                or blocks != (dimension + block_factor - 1) // block_factor
                or ub_factor > block_factor):
            raise ValueError("captured Cast dimensions or partition do not match the synthetic inputs")
        projected.append({"elements": elements, "launch": "warmup" if index % 2 == 0 else "target",
                          "block_dim": blocks, "tiling_variables": dict(zip(fields, values)),
                          "baseline": row["baseline"]})
    for index in range(0, len(projected), 2):
        if (projected[index]["tiling_variables"] != projected[index + 1]["tiling_variables"]
                or projected[index]["block_dim"] != projected[index + 1]["block_dim"]):
            raise ValueError("warmup and target disagree on the discrete Cast partition")
    _write(output, {"schema_version": 1,
        "scope": "synthetic_selected_Cast_inline_tiling_observation_only",
        "raw_file": str(path.resolve()), "raw_sha256": _sha256(path),
        "compile_info_file": str(compile_info_path.resolve()),
        "compile_info_sha256": _sha256(compile_info_path),
        "variable_encoding": "little_endian_signed_int64_from_explicit_compileInfo",
        "rows": projected, "duration_consumed": False,
        "unobserved_shape_rule_derived": False, "selected_ISA_bound": False,
        "full_operator_rate_eligible": False, "production_binding_verified": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-reference-only", action="store_true")
    parser.add_argument("--project-op-summary", type=Path)
    parser.add_argument("--project-host-args", type=Path)
    parser.add_argument("--compile-info-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--expected-torch-version")
    parser.add_argument("--expected-torch-npu-version")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    if args.cpu_reference_only:
        for elements in ELEMENT_COUNTS:
            values = _values(elements)
            if len(values) != elements or any(
                    value < -(2 ** 31) or value >= 2 ** 31
                    for value in values):
                raise AssertionError("invalid signed INT32 synthetic inputs")
            # Signed widening preserves every exact mathematical integer;
            # the device execution below independently checks all outputs.
            if any(int(value) != value for value in values):
                raise AssertionError("integer widening reference mismatch")
        print(json.dumps({
            "scope": "cpu_synthetic_reference_only",
            "elements_checked": sum(ELEMENT_COUNTS),
            "input_dtype": "int32", "output_dtype": "int64",
            "device_execution": False,
            "duration_consumed": False,
        }, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required for capture or projection")
    if args.project_op_summary is not None:
        _project(args.project_op_summary, args.output)
        return
    if args.project_host_args is not None:
        if args.compile_info_json is None:
            parser.error("host-argument projection requires --compile-info-json")
        _project_host_args(args.project_host_args, args.compile_info_json, args.output)
        return
    if (args.device_id is None or args.device_id < 0
            or args.expected_torch_version is None
            or args.expected_torch_npu_version is None
            or args.baseline is None):
        parser.error("execution needs --device-id, --baseline and both exact "
                     "--expected-*-version arguments")
    if len(args.baseline) != 40 or any(
            character not in "0123456789abcdef" for character in args.baseline):
        parser.error("--baseline must be a 40-character lowercase commit")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if importlib.metadata.version("torch") != args.expected_torch_version:
        raise RuntimeError("installed Torch version does not match request")
    if importlib.metadata.version("torch-npu") != args.expected_torch_npu_version:
        raise RuntimeError("installed Torch-NPU version does not match request")
    # Device selection precedes imports/initialization. The hardware executor
    # must first establish exclusivity and idle status for this physical card.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    import torch
    import torch_npu

    torch_npu.npu.set_device(0)
    observations = []
    for elements in ELEMENT_COUNTS:
        values = _values(elements)
        source_cpu = torch.tensor(values, dtype=torch.int32)
        source = source_cpu.to("npu:0")
        # Prewarm outside the named target. There is no timing interval here.
        source.to(torch.int64)
        torch_npu.npu.synchronize()
        with torch.autograd.profiler.record_function(
                f"independent_int32_int64_elements_{elements}"):
            destination = source.to(torch.int64)
            torch_npu.npu.synchronize()
        actual = destination.cpu()
        if (destination.dtype != torch.int64
                or not destination.is_contiguous()
                or actual.tolist() != values):
            raise RuntimeError(f"signed widening failed for {elements} elements")
        observations.append({
            "elements": elements,
            "input_dtype": "int32", "output_dtype": "int64",
            "input_contiguous": source.is_contiguous(),
            "output_contiguous": destination.is_contiguous(),
            "all_elements_validated": True,
        })
    _write(args.output, {
        "schema_version": 1,
        "scope": "synthetic_signed_integer_cast_only",
        "baseline": args.baseline,
        "runner_sha256": _sha256(Path(__file__)),
        "python": sys.version,
        "torch_version": str(torch.__version__),
        "torch_npu_version": importlib.metadata.version("torch-npu"),
        "physical_device_id": args.device_id,
        "observations": observations,
        "profile_raw_retained_separately": True,
        "duration_consumed": False,
        "full_operator_rate_eligible": False,
        "actual_serving_dispatch_verified": False,
        "production_binding_verified": False,
    })


if __name__ == "__main__":
    main()
