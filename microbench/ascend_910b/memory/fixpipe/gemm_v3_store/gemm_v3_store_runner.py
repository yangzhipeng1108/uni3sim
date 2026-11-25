#!/usr/bin/env python3
"""Build and run the frozen MatMulV3-65536 BF16 store probe."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


CELLS = (
    ("bf16", 0, 128, 256, 1),
    ("bf16", 0, 128, 128, 1),
    ("bf16", 0, 64, 256, 1),
    ("fp32", 1, 128, 128, 1),
    ("bf16", 0, 128, 256, 24),
    ("bf16", 0, 128, 128, 24),
    ("bf16", 0, 64, 256, 24),
    ("fp32", 1, 128, 128, 24),
)
DTYPE_BYTES = {"bf16": 2, "fp32": 4}
SHORT_INVOCATIONS = 32
LONG_INVOCATIONS = 96
SOURCE_FILES = (
    "CMakeLists.txt",
    "gemm_v3_store_contract.h",
    "gemm_v3_store_kernel.cpp",
    "gemm_v3_store_main.cpp",
    "gemm_v3_store_report.py",
    "gemm_v3_store_runner.py",
)
CANN_RELATIVE_PATHS = {
    "matmul_v3_kernel": (
        "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/"
        "mat_mul_v3/op_kernel/mat_mul_v3.cpp"
    ),
    "matmul_v3_base": (
        "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/3rd/"
        "mat_mul_v3/op_kernel/mat_mul_base_kernel.h"
    ),
    "matmul_copy_out": (
        "aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_out/"
        "copy_cube_out_fixpipe.h"
    ),
    "matmul_fixpipe_params": (
        "aarch64-linux/asc/impl/adv_api/detail/matmul/stage/copy_cube_out/"
        "copy_cube_out_utils.h"
    ),
    "fixpipe_impl": (
        "aarch64-linux/asc/impl/basic_api/dav_c220/"
        "kernel_operator_fixpipe_v2_impl.h"
    ),
    "kernel_metadata": (
        "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_nn/"
        "mat_mul_v3/MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_"
        "high_performance.json"
    ),
    "kernel_binary": (
        "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_nn/"
        "mat_mul_v3/MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_"
        "high_performance.o"
    ),
}
CANN_SHA256 = {
    "matmul_v3_kernel": "78b3bd1eca00607a9372b31b032b6eafc008b967d417f48f5de0ebafe4cf23c4",
    "matmul_v3_base": "3212d6d92c8be9124630b3c8f504a245e656fe926bc336ebb9876c459c64db6d",
    "matmul_copy_out": "fd4e3fe634ed82893f8d58a6ca49016e61700267d8c6e3c421db4578746d4774",
    "matmul_fixpipe_params": "4394b3c733709600f7dd0feedb01c75253e1964e1364050938861ac202e61b3f",
    "fixpipe_impl": "bf77b939a3c91df743f0e11c162b429a7bc871252b48c349527912b715bc0be9",
    "kernel_metadata": "0d252989ae68724145b139d747dc49d9416d8a929cf422d0790531a7904008a7",
    "kernel_binary": "53f6b25ef79582e2e49f80ec9d68df1ef58f05856d4b9ba04d0fd2ce39d883b9",
}
SOURCE_MARKERS = {
    "matmul_v3_base": ("mm_.Iterate();", "mm_.GetTensorC("),
    "matmul_copy_out": (
        "FixpipeAdaptor fixpipe(baseWidth, baseHeight",
        "fixpipe.template FixpipeOut<T>(dst, co1Local);",
    ),
    "matmul_fixpipe_params": (
        "QuantMode_t::F322BF16",
        "Fixpipe<DstT, SrcT, CFG_ROW_MAJOR>(dst, colLocal, params_);",
    ),
    "fixpipe_impl": (
        "PipeBarrier<PIPE_FIX>();",
        "QuantMode_t::F322BF16",
        "copy_matrix_cc_to_gm",
    ),
}
RAW_FIELDS = (
    "run_id", "trial", "order_direction", "order_position", "process_pid",
    "dtype", "m", "n", "block_dim", "output_bytes", "l0c_bytes",
    "short_invocations", "long_invocations", "short_cycles", "long_cycles",
    "validation",
)
PROGRAM_FIELDS = RAW_FIELDS[5:]
HEX40 = re.compile(r"[0-9a-f]{40}")
RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,96}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(root):
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        payload = (root / name).read_bytes()
        digest.update(name.encode("utf-8") + b"\0" + payload)
    return digest.hexdigest()


def _run(command, environment=None, timeout=120):
    result = subprocess.run(
        command, check=True, capture_output=True, text=True,
        env=environment, timeout=timeout,
    )
    return result.stdout


def _validate_cann(cann_root):
    identities = {}
    for label, relative in CANN_RELATIVE_PATHS.items():
        path = (cann_root / relative).resolve()
        if _sha256(path) != CANN_SHA256[label]:
            raise ValueError(f"CANN identity drifted for {label}")
        identities[label] = {"path": str(path), "sha256": CANN_SHA256[label]}
    for label, markers in SOURCE_MARKERS.items():
        source = Path(identities[label]["path"]).read_text(encoding="utf-8")
        if any(marker not in source for marker in markers):
            raise ValueError(f"CANN source semantics drifted for {label}")
    metadata = json.loads(Path(identities["kernel_metadata"]["path"]).read_text())
    matching = [
        entry for entry in metadata["kernelList"]
        if entry["tilingKey"] == 65536
    ]
    if len(matching) != 1 or matching[0] != {
        "tilingKey": 65536,
        "kernelType": "MIX_AIC",
        "taskRation": "1:0",
        "crossCoreSync": 0,
        "kernelName": (
            "MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_"
            "high_performance_65536"
        ),
    }:
        raise ValueError("MatMulV3 key 65536 is not the frozen AIC-only binary")
    return identities


def _parse_idle(text):
    patterns = {
        "aicore_percent": r"Aicore Usage Rate\(%\)\s*:\s*([0-9.]+)",
        "aivector_percent": r"Aivector Usage Rate\(%\)\s*:\s*([0-9.]+)",
        "hbm_bandwidth_percent": r"HBM Bandwidth Usage Rate\(%\)\s*:\s*([0-9.]+)",
    }
    result = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if len(matches) != 1:
            raise ValueError(f"npu-smi did not expose one {name}")
        value = float(matches[0])
        if value != 0.0:
            raise RuntimeError(f"selected device is busy: {name}={value}")
        result[name] = value
    return result


def _build(source, build, cann_root):
    if build.exists():
        raise FileExistsError(f"fresh build directory required: {build}")
    cmake = Path(shutil.which("cmake")).resolve()
    configure = [
        str(cmake), "-S", str(source.resolve()), "-B", str(build.resolve()),
        "-DCMAKE_BUILD_TYPE=Release", f"-DASCEND_HOME_PATH={cann_root.resolve()}",
        f"-DASCEND_CANN_PACKAGE_PATH={cann_root.resolve()}",
        "-DSOC_VERSION=Ascend910B2",
    ]
    compile_command = [
        str(cmake), "--build", str(build.resolve()), "--target",
        "gemm_v3_store", "--parallel", "1",
    ]
    _run(configure, timeout=180)
    _run(compile_command, timeout=600)
    binary = (build / "gemm_v3_store").resolve()
    archive = (build / "lib" / "libgemm_v3_store_kernel.a").resolve()
    members = _run([str(Path(shutil.which("ar")).resolve()), "t", str(archive)])
    member_names = [line for line in members.splitlines() if line]
    if not any(name.endswith("gemm_v3_store_kernel.cpp.o") for name in member_names):
        raise ValueError("kernel archive lacks gemm_v3_store_kernel.cpp.o")
    return {
        "configure_argv": configure,
        "build_argv": compile_command,
        "binary": str(binary),
        "binary_sha256": _sha256(binary),
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "archive_members": member_names,
    }


def _cell_command(binary, cells):
    command = [str(binary), "--batch"]
    for _dtype, dtype_id, m, n, block_dim in cells:
        command.extend((str(dtype_id), str(m), str(n), str(block_dim)))
    return command


def _parse_program_output(text, expected_cells):
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames) != PROGRAM_FIELDS:
        raise ValueError("benchmark output schema drifted")
    rows = list(reader)
    if len(rows) != len(expected_cells):
        raise ValueError("benchmark output row count drifted")
    for row, cell in zip(rows, expected_cells):
        dtype, _dtype_id, m, n, block_dim = cell
        expected_bytes = m * n * DTYPE_BYTES[dtype]
        if (
            row["dtype"] != dtype
            or int(row["m"]) != m
            or int(row["n"]) != n
            or int(row["block_dim"]) != block_dim
            or int(row["output_bytes"]) != expected_bytes
            or int(row["l0c_bytes"]) != m * n * 4
            or int(row["short_invocations"]) != SHORT_INVOCATIONS
            or int(row["long_invocations"]) != LONG_INVOCATIONS
            or row["validation"] != "exact"
        ):
            raise ValueError("benchmark result violated the frozen cell contract")
        for field in ("short_cycles", "long_cycles"):
            values = [int(value) for value in row[field].split(";")]
            if len(values) != block_dim or any(value <= 0 for value in values):
                raise ValueError(f"invalid {field} vector")
        if any(
            long <= short for short, long in zip(
                (int(value) for value in row["short_cycles"].split(";")),
                (int(value) for value in row["long_cycles"].split(";")),
            )
        ):
            raise ValueError("long interval must exceed short interval")
    return rows


def _run_campaign(args, source):
    if RUN_ID.fullmatch(args.run_id) is None:
        raise ValueError("run-id must use 1-96 safe characters")
    if HEX40.fullmatch(args.source_commit) is None:
        raise ValueError("source-commit must be 40 lowercase hex characters")
    snapshot = _snapshot(source)
    if snapshot != args.expected_source_snapshot_sha256:
        raise ValueError("source snapshot does not match preregistered identity")
    identities = _validate_cann(args.cann_root)
    smi = Path(shutil.which("npu-smi")).resolve()
    idle_command = [str(smi), "info", "-t", "usages", "-i", str(args.device_id)]
    idle_text = _run(idle_command, timeout=30)
    idle = _parse_idle(idle_text)
    build = _build(source, args.build_directory, args.cann_root)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    environment["ASCEND_DEVICE_ID"] = "0"
    rows = []
    commands = []
    for trial in range(1, args.trials + 1):
        if trial % 2 == 1:
            direction = "forward"
            cells = CELLS
        else:
            direction = "reverse"
            cells = tuple(reversed(CELLS))
        command = _cell_command(Path(build["binary"]), cells)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment,
        )
        stdout, stderr = process.communicate(timeout=args.timeout_seconds)
        if process.returncode != 0:
            raise RuntimeError(
                f"benchmark failed with {process.returncode}: {stderr.strip()}")
        parsed = _parse_program_output(stdout, cells)
        for position, row in enumerate(parsed):
            rows.append({
                "run_id": args.run_id,
                "trial": str(trial),
                "order_direction": direction,
                "order_position": str(position),
                "process_pid": str(process.pid),
                **row,
            })
        commands.append({
            "trial": trial,
            "order_direction": direction,
            "pid": process.pid,
            "argv": command,
            "stderr": stderr,
        })
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "schema": "ascend_910b_gemm_v3_store_provenance_v1",
        "run_id": args.run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner": "ascend_gemm_v3_store_probe",
        "hardware": "ascend_910b",
        "physical_device_id": args.device_id,
        "logical_device_id": 0,
        "baseline_commit": args.source_commit,
        "source_snapshot_sha256": snapshot,
        "source_files": {name: _sha256(source / name) for name in SOURCE_FILES},
        "cann_version": "8.5.1",
        "cann_identities": identities,
        "v3_tiling_key": 65536,
        "v3_kernel_type": "AIC_ONLY",
        "v3_tile": {"m": 128, "n": 256, "k": 64},
        "measurement_scope": (
            "Independent direct L0C-to-GM Fixpipe service only; BF16 cast, "
            "public pre-issue PIPE_FIX barrier, and one final completion barrier"
        ),
        "production_or_evaluation_duration_consumed": False,
        "idle_command": idle_command,
        "idle_telemetry": idle,
        "idle_telemetry_sha256": hashlib.sha256(
            idle_text.encode("utf-8")).hexdigest(),
        "build": build,
        "commands": commands,
        "trials": args.trials,
        "cells": [list(cell) for cell in CELLS],
        "raw_output": str(args.raw_output.resolve()),
        "raw_sha256": _sha256(args.raw_output),
        "raw_rows": len(rows),
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance_output.open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("snapshot")
    run = subparsers.add_parser("run")
    run.add_argument("--build-directory", required=True, type=Path)
    run.add_argument("--cann-root", required=True, type=Path)
    run.add_argument("--device-id", required=True, type=int)
    run.add_argument("--run-id", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--expected-source-snapshot-sha256", required=True)
    run.add_argument("--raw-output", required=True, type=Path)
    run.add_argument("--provenance-output", required=True, type=Path)
    run.add_argument("--trials", required=True, type=int)
    run.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.command == "snapshot":
        print(_snapshot(source))
    elif args.command == "run":
        if args.trials != 3:
            raise ValueError("the frozen campaign requires exactly three trials")
        _run_campaign(args, source)
    else:
        raise AssertionError(f"unsupported command {args.command}")


if __name__ == "__main__":
    main()
