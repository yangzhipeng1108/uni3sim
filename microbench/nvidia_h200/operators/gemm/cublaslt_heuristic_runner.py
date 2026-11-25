#!/usr/bin/env python3
"""Build and run allocation-free cuBLASLt heuristic queries on NVIDIA H200.

H200 owner copy of microbench/nvidia_a800/operators/gemm/
cublaslt_heuristic_runner.py.  The measured mechanism is unchanged: for every
physical GEMM operation the cuBLASLt heuristic list is queried under a 32 MiB
workspace preference, rank 0 is dry-launched with every production launch
replaced by a one-thread no-op before submission, and CUPTI activity proves
that only the no-op executed.  No matmul is executed and no time is measured.

Hardware/software coupling changes (and nothing else):
  * device check ``NVIDIA H200`` / compute capability 9.0 instead of A800/8.0;
    idle requirement is utilization 0 % and memory.used <= 16 MiB (an idle
    H200 on this driver reports 4 MiB, the A800 reported 0 MiB);
  * cuBLASLt 12.9 from <workspace>/tools/cuda-12.9 (A800: 12.2);
  * devices are the owned GPUs 0 and 3 (one query process per GPU, the
    operation list is split round-robin exactly like the A800 runner);
  * the query emits enum ids; tile / stages / cluster names and dimensions are
    mapped here from the cublasLt.h the binary was compiled against, because
    the 12.9 enums (hundreds of tiles) outgrew the hand table of the A800 copy;
  * the launch record carries the cluster dimension and launch-attribute ids
    of Hopper ``cuLaunchKernelEx`` launches;
  * the operation domain is not rebuilt from model configs on the remote host
    (no repository there): it is the exact physical-operation set of the A800
    launch-selection CSV (``domain`` subcommand, run locally), which is the
    ShareGPT workload domain plus the A800 kernel-dataset GEMM launches;
  * output root ``profiler_output/microbench/nvidia_h200`` locally, and the
    remote working directory <workspace>/h200_microbench/wp0_launch.

Subcommands:
  domain   (local)  A800 CSV -> operation list + selection keys (JSON).
  capture  (remote) compile the query, run the sweep on the given GPUs, write
                    the full raw candidate (JSON lines, gzip) and a compact
                    rank-0 launch-fact file (JSON lines, gzip).
Standard library only for ``capture`` (the remote system python3 has no
third-party packages).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Mapping, NamedTuple, Sequence

RUNNER = Path(__file__).resolve()
SOURCE = RUNNER.with_name("cublaslt_heuristic_query.cpp")
HARDWARE_ID = "nvidia_h200"
DEVICE_NAME_PREFIX = "NVIDIA H200"
COMPUTE_CAPABILITY = "9.0"
IDLE_MAX_MEMORY_MIB = 16
HEURISTIC_RESULT_CAPACITY = 64
POSTFLIGHT_IDLE_ATTEMPTS = 31
POSTFLIGHT_IDLE_INTERVAL_SECONDS = 1.0
DEFAULT_CUDA_ROOT = (
    Path(os.environ["CUDA_HOME"])
    if "CUDA_HOME" in os.environ else Path("/usr/local/cuda")
)
ALGORITHM_FIELDS = frozenset({
    "rank", "algorithm_id", "tile_id", "stages_id", "cluster_shape_id",
    "inner_shape_id", "split_k", "reduction_scheme", "cta_swizzle",
    "custom_option", "workspace_bytes", "waves_count",
})
LAUNCH_FIELDS = {
    "api", "symbol", "grid", "block", "shared_memory_bytes", "cluster",
    "launch_attribute_ids", "cta_raster_log_group", "cta_raster_serpentine",
}
LAUNCH_APIS = {
    "cuLaunchKernel", "cuLaunchKernel_ptsz",
    "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz",
}
DEVICE_EXECUTION_FIELDS = {
    "evidence_kind", "evidence_version", "activity_record_count",
    "executed_symbols", "all_replacements_observed", "dropped_record_count",
}
DRY_LAUNCH_FIELDS = {
    "attempted", "replacement_kernel_submitted", "replacement_launch_count",
    "launches", "device_execution",
}


class DeviceBusyError(ValueError):
    """Selected device telemetry does not yet prove an idle device."""


class Operation(NamedTuple):
    m: int
    n: int
    k: int
    dtype: str
    has_bias: bool

    def key(self) -> str:
        return json.dumps([
            self.m, self.n, self.k, self.dtype, self.has_bias,
        ], separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# --------------------------------------------------------------------------
# cublasLt.h enum tables (static semantics of the compiled library header)
# --------------------------------------------------------------------------

def cublaslt_enum_tables(header: Path) -> dict[str, dict[int, object]]:
    text = header.read_text()
    tiles = {0: None}
    for a, b, value in re.findall(
            r"CUBLASLT_MATMUL_TILE_(\d+)x(\d+)\s*=\s*(\d+)", text):
        tiles[int(value)] = (int(a), int(b))
    stages = {0: None}
    for k, depth, value in re.findall(
            r"CUBLASLT_MATMUL_STAGES_(\d+)x(\d+|AUTO)\s*=\s*(\d+)", text):
        stages[int(value)] = f"{k}x{depth}"
    clusters = {0: "auto"}
    for x, y, z, value in re.findall(
            r"CUBLASLT_CLUSTER_SHAPE_(\d+)x(\d+)x(\d+)\s*=\s*(\d+)", text):
        clusters[int(value)] = (int(x), int(y), int(z))
    if len(tiles) < 100 or len(stages) < 30 or len(clusters) < 40:
        raise ValueError(f"{header} enum tables look incomplete")
    return {"tile": tiles, "stages": stages, "cluster": clusters}


def enrich_algorithm(
        algorithm: Mapping[str, object],
        tables: Mapping[str, Mapping[int, object]], context: str,
        ) -> dict[str, object]:
    tile_id = algorithm["tile_id"]
    stages_id = algorithm["stages_id"]
    cluster_id = algorithm["cluster_shape_id"]
    for name, table, value in (
            ("tile", tables["tile"], tile_id),
            ("stages", tables["stages"], stages_id),
            ("cluster", tables["cluster"], cluster_id)):
        if value not in table:
            raise ValueError(f"{context} has unknown {name} enum id {value}")
    tile = tables["tile"][tile_id]
    stages = tables["stages"][stages_id]
    cluster = tables["cluster"][cluster_id]
    enriched = dict(algorithm)
    enriched["tile"] = "undefined" if tile is None else f"{tile[0]}x{tile[1]}"
    # Same convention as the A800 query: the cuBLASLt tile is stated for the
    # column-major problem, whose first extent is the row-major N.
    enriched["logical_tile_m"] = 0 if tile is None else tile[1]
    enriched["logical_tile_n"] = 0 if tile is None else tile[0]
    enriched["stages"] = "undefined" if stages is None else stages
    enriched["cluster_shape"] = (
        cluster if isinstance(cluster, str)
        else "x".join(str(item) for item in cluster))
    return enriched


# --------------------------------------------------------------------------
# domain (local)
# --------------------------------------------------------------------------

def _domain(args: argparse.Namespace) -> int:
    import csv
    import io
    import zstandard

    csv_bytes = args.a800_csv.read_bytes()
    manifest_bytes = args.a800_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    text = zstandard.ZstdDecompressor().decompress(csv_bytes).decode()
    if _sha256_bytes(text.encode()) != manifest["csv_sha256"]:
        raise ValueError("A800 CSV does not match its manifest csv_sha256")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    key_fields = manifest["selection_key_fields"]
    if header[:len(key_fields)] != key_fields:
        raise ValueError("A800 CSV header does not start with the key fields")
    selection_keys = []
    operations = set()
    for row in reader:
        key = row[:len(key_fields)]
        (dtype, variant, b, m, n, k, a_layout, b_layout, c_layout,
         trans_a, trans_b, epilogue, has_bias) = key
        if (a_layout, b_layout, c_layout, trans_a, trans_b) != (
                "row_major", "row_major", "row_major", "false", "true"):
            raise ValueError(f"unsupported layout in domain row {key}")
        if epilogue != ("bias" if has_bias == "true" else "none"):
            raise ValueError(f"unsupported epilogue in domain row {key}")
        if variant not in ("gemm", "gemv"):
            raise ValueError(f"unsupported variant in domain row {key}")
        operation = Operation(
            int(b) * int(m), int(n), int(k), dtype, has_bias == "true")
        selection_keys.append([key, operation.key()])
        operations.add(operation)
    if len(selection_keys) != manifest["row_count"]:
        raise ValueError("A800 CSV row count differs from its manifest")
    payload = {
        "schema_version": 1,
        "artifact_type": "nvidia_h200_gemm_launch_domain",
        "source": "launch_data/nvidia_a800/gemm_launch_selection.csv.zst",
        "source_csv_sha256": manifest["csv_sha256"],
        "source_csv_compressed_sha256": _sha256_bytes(csv_bytes),
        "source_manifest_sha256": _sha256_bytes(manifest_bytes),
        "source_domain_sha256": manifest["domain_sha256"],
        "selection_key_fields": key_fields,
        "selection_key_count": len(selection_keys),
        "operation_count": len(operations),
        "operations": [list(op) for op in sorted(operations)],
        "selection_keys": selection_keys,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({
        "output": str(args.output), "selection_keys": len(selection_keys),
        "operations": len(operations)}))
    return 0


# --------------------------------------------------------------------------
# capture (remote)
# --------------------------------------------------------------------------

def _compile(executable: Path, cuda_root: Path, source: Path) -> None:
    include_dir = cuda_root / "include"
    library_dir = cuda_root / "lib64"
    if not include_dir.is_dir() or not library_dir.is_dir():
        raise FileNotFoundError(f"CUDA include/lib64 missing below {cuda_root}")
    subprocess.run(
        [
            "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            f"-I{include_dir}", str(source), f"-L{library_dir}",
            f"-Wl,-rpath,{library_dir}", "-lcublasLt", "-lcudart", "-lcupti",
            "-lcuda", "-o", str(executable),
        ],
        check=True,
    )


def _batch_query_input(
        operations: Sequence[Operation], max_workspace_bytes: int) -> str:
    return "".join(
        f"{operation.m} {operation.n} {operation.k} {operation.dtype} "
        f"{1 if operation.has_bias else 0} {max_workspace_bytes}\n"
        for operation in operations
    )


def _int_field(value: object, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (
            1 if positive else 0):
        raise ValueError(f"{context} must be a {'positive' if positive else 'non-negative'} integer")
    return value


def validate_query_result(
        value: object, operation: Operation, max_workspace_bytes: int,
        tables: Mapping[str, Mapping[int, object]], context: str,
        ) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    result = value
    expected_fields = {
        "schema_version", "hardware_id", "device_name",
        "compute_capability", "cublaslt_version", "operation",
        "max_workspace_bytes", "matmul_executed",
        "production_kernel_submitted", "dry_launch", "algorithms",
    }
    if set(result) != expected_fields:
        raise ValueError(f"{context} fields differ: {sorted(result)}")
    if result["schema_version"] != 2 or result["hardware_id"] != HARDWARE_ID:
        raise ValueError(f"{context} has unexpected schema/hardware")
    if not str(result["device_name"]).startswith(DEVICE_NAME_PREFIX):
        raise ValueError(f"{context} did not run on an {DEVICE_NAME_PREFIX}")
    if result["compute_capability"] != COMPUTE_CAPABILITY:
        raise ValueError(f"{context} has unexpected compute capability")
    _int_field(result["cublaslt_version"], f"{context}.cublaslt_version",
               positive=True)
    if result["operation"] != {
            "m": operation.m, "n": operation.n, "k": operation.k,
            "dtype": operation.dtype, "layout": "row_major",
            "trans_a": False, "trans_b": True,
            "has_bias": operation.has_bias}:
        raise ValueError(f"{context} operation does not match its request")
    if result["max_workspace_bytes"] != max_workspace_bytes:
        raise ValueError(f"{context} workspace limit differs")
    if (result["matmul_executed"] is not False
            or result["production_kernel_submitted"] is not False):
        raise ValueError(f"{context} executed a production kernel")
    algorithms = result["algorithms"]
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError(f"{context} contains no heuristic algorithms")
    if len(algorithms) >= HEURISTIC_RESULT_CAPACITY:
        raise ValueError(f"{context} reached the heuristic capture capacity")
    enriched = []
    for index, algorithm in enumerate(algorithms):
        actx = f"{context} algorithm {index}"
        if not isinstance(algorithm, dict) or set(algorithm) != ALGORITHM_FIELDS:
            raise ValueError(f"{actx} has unexpected fields")
        for field in ALGORITHM_FIELDS - {"waves_count"}:
            _int_field(algorithm[field], f"{actx}.{field}")
        if algorithm["rank"] != index:
            raise ValueError(f"{actx} ranks are not contiguous")
        if algorithm["workspace_bytes"] > max_workspace_bytes:
            raise ValueError(f"{actx} exceeds the workspace limit")
        waves = algorithm["waves_count"]
        if (isinstance(waves, bool) or not isinstance(waves, (int, float))
                or not math.isfinite(waves) or waves < 0):
            raise ValueError(f"{actx} has invalid waves_count")
        enriched.append(enrich_algorithm(algorithm, tables, actx))
    result = dict(result)
    result["algorithms"] = enriched
    dry = result["dry_launch"]
    if not isinstance(dry, dict) or set(dry) != DRY_LAUNCH_FIELDS:
        raise ValueError(f"{context} dry_launch fields differ")
    if dry["attempted"] is not True or dry["replacement_kernel_submitted"] is not True:
        raise ValueError(f"{context} did not dry-launch rank-0")
    launches = dry["launches"]
    if not isinstance(launches, list) or dry["replacement_launch_count"] != len(launches):
        raise ValueError(f"{context} did not replace every captured launch")
    for index, launch in enumerate(launches):
        lctx = f"{context} launch {index}"
        if not isinstance(launch, dict) or set(launch) != LAUNCH_FIELDS:
            raise ValueError(f"{lctx} fields differ")
        if launch["api"] not in LAUNCH_APIS:
            raise ValueError(f"{lctx} used an unknown API")
        if not launch["symbol"] or "midlayer_noop" in launch["symbol"]:
            raise ValueError(f"{lctx} has an invalid symbol")
        for field in ("grid", "block", "cluster"):
            dims = launch[field]
            if (not isinstance(dims, list) or len(dims) != 3
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           or item <= 0 for item in dims)):
                raise ValueError(f"{lctx} has invalid {field}")
        _int_field(launch["shared_memory_bytes"], f"{lctx}.shared_memory_bytes")
    evidence = dry["device_execution"]
    if (not isinstance(evidence, dict) or set(evidence) != DEVICE_EXECUTION_FIELDS
            or evidence["evidence_kind"] != "cupti_activity_kernel"
            or evidence["evidence_version"] != 1
            or evidence["all_replacements_observed"] is not True
            or evidence["dropped_record_count"] != 0
            or evidence["activity_record_count"] != len(launches)
            or evidence["executed_symbols"] != ["midlayer_noop"] * len(launches)):
        raise ValueError(f"{context} lacks affirmative CUPTI replacement evidence")
    rank0 = enriched[0]
    reducers = [launch for launch in launches
                if "splitKreduce_kernel" in launch["symbol"]]
    if rank0["reduction_scheme"] in (0, 1) and (len(launches) != 1 or reducers):
        raise ValueError(f"{context} direct/in-place execution must have one main launch")
    if rank0["reduction_scheme"] in (2, 4) and (
            len(launches) != 2 or len(reducers) != 1
            or launches[-1] is not reducers[0]):
        raise ValueError(f"{context} separate-step split must have one trailing reducer")
    return result


def _nvidia_smi_snapshot() -> dict[str, dict[str, object]]:
    completed = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    snapshot = {}
    for line in completed.stdout.splitlines():
        index, uuid, name, utilization, memory = (
            field.strip() for field in line.split(","))
        snapshot[index] = {
            "uuid": uuid, "name": name,
            "utilization_gpu_percent": int(utilization),
            "memory_used_mib": int(memory),
        }
    return snapshot


def _validate_device_snapshot(snapshot, devices, phase) -> None:
    for device in devices:
        record = snapshot[device]
        if not str(record["name"]).startswith(DEVICE_NAME_PREFIX):
            raise ValueError(f"GPU {device} is not an {DEVICE_NAME_PREFIX}")
        if (record["utilization_gpu_percent"] != 0
                or record["memory_used_mib"] > IDLE_MAX_MEMORY_MIB):
            raise DeviceBusyError(
                f"GPU {device} is not idle at {phase}: {record}")


def _wait_for_idle_snapshot(devices, phase):
    snapshots = []
    for attempt in range(POSTFLIGHT_IDLE_ATTEMPTS):
        snapshot = _nvidia_smi_snapshot()
        snapshots.append({device: snapshot[device] for device in devices})
        try:
            _validate_device_snapshot(snapshot, devices, phase)
        except DeviceBusyError:
            if attempt + 1 == POSTFLIGHT_IDLE_ATTEMPTS:
                raise
            time.sleep(POSTFLIGHT_IDLE_INTERVAL_SECONDS)
        else:
            return snapshot, snapshots
    raise AssertionError("idle telemetry loop exhausted without a result")


def _run_partition(executable, device, operations, max_workspace_bytes,
                   tables, raw_path, chunk):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    results = []
    with gzip.open(raw_path, "wt") as raw:
        for start in range(0, len(operations), chunk):
            part = operations[start:start + chunk]
            completed = subprocess.run(
                [str(executable)],
                input=_batch_query_input(part, max_workspace_bytes),
                capture_output=True, text=True, env=environment)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"query failed on GPU {device} in chunk at {start}: "
                    f"{completed.stderr[-2000:]}")
            lines = completed.stdout.splitlines()
            if len(lines) != len(part):
                raise ValueError(
                    f"GPU {device} returned {len(lines)} results for "
                    f"{len(part)} operations")
            for operation, line in zip(part, lines):
                raw.write(line + "\n")
                result = validate_query_result(
                    json.loads(line), operation, max_workspace_bytes, tables,
                    f"GPU {device} operation {operation.key()}")
                results.append((operation, result))
            print(json.dumps({"device": device,
                              "done": start + len(part),
                              "total": len(operations)}), flush=True)
    return results


def _capture(args: argparse.Namespace) -> int:
    domain = json.loads(args.domain.read_text())
    if domain["artifact_type"] != "nvidia_h200_gemm_launch_domain":
        raise ValueError("--domain is not an H200 GEMM launch domain")
    operations = tuple(sorted(
        Operation(m, n, k, dtype, has_bias)
        for m, n, k, dtype, has_bias in domain["operations"]))
    if len(operations) != domain["operation_count"]:
        raise ValueError("domain operation list is inconsistent")
    if args.limit is not None:
        step = max(1, len(operations) // args.limit)
        operations = operations[::step][:args.limit]
    devices = tuple(args.devices.split(","))
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    tables = cublaslt_enum_tables(args.cuda_root / "include" / "cublasLt.h")
    source_bytes = SOURCE.read_bytes()
    runner_bytes = RUNNER.read_bytes()
    executable = out / "cublaslt_heuristic_query"
    source_snapshot = out / SOURCE.name
    source_snapshot.write_bytes(source_bytes)
    _compile(executable, args.cuda_root, source_snapshot)
    preflight, _ = _wait_for_idle_snapshot(devices, "preflight")
    partitions = tuple(operations[index::len(devices)]
                       for index in range(len(devices)))
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(
                    _run_partition, executable, device, partition,
                    args.max_workspace_bytes, tables,
                    out / f"raw_results_gpu{device}.jsonl.gz", args.chunk)
                for device, partition in zip(devices, partitions)]
            queried = [(device, operation, result)
                       for device, future in zip(devices, futures)
                       for operation, result in future.result()]
    finally:
        postflight, attempts = _wait_for_idle_snapshot(devices, "postflight")
    versions = {result["cublaslt_version"] for _, _, result in queried}
    names = {result["device_name"] for _, _, result in queried}
    if len(versions) != 1:
        raise ValueError(f"devices reported different cuBLASLt versions {versions}")
    compact_path = out / "rank0_launch_facts.jsonl.gz"
    with gzip.open(compact_path, "wt") as compact:
        for device, operation, result in sorted(
                queried, key=lambda item: item[1]):
            compact.write(json.dumps({
                "operation_key": operation.key(),
                "device": device,
                "algorithm_count": len(result["algorithms"]),
                "rank0": result["algorithms"][0],
                "launches": result["dry_launch"]["launches"],
                "device_execution": {
                    key: result["dry_launch"]["device_execution"][key]
                    for key in ("evidence_kind", "activity_record_count",
                                "dropped_record_count",
                                "all_replacements_observed")},
            }, separators=(",", ":"), sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "artifact_type": "nvidia_h200_gemm_launch_candidate",
        "hardware_id": HARDWARE_ID,
        "runner_sha256": _sha256_bytes(runner_bytes),
        "query_source_sha256": _sha256_bytes(source_bytes),
        "domain_sha256": _sha256_bytes(args.domain.read_bytes()),
        "source_domain_csv_sha256": domain["source_csv_sha256"],
        "cuda_root": str(args.cuda_root),
        "cublaslt_version": versions.pop(),
        "device_names": sorted(names),
        "requested_devices": list(devices),
        "device_telemetry": {
            "idle_requirement": {"utilization_gpu_percent": 0,
                                 "memory_used_mib_max": IDLE_MAX_MEMORY_MIB},
            "preflight": {d: preflight[d] for d in devices},
            "postflight": {d: postflight[d] for d in devices},
            "postflight_attempts": attempts,
        },
        "max_workspace_bytes": args.max_workspace_bytes,
        "selector_contract": "cublaslt_first_feasible_rank_0",
        "duration_consumed": False,
        "matmul_executed": False,
        "production_kernel_submitted": False,
        "operation_count": len(queried),
        "operations_per_device": {
            d: sum(1 for dev, _, _ in queried if dev == d) for d in devices},
        "wall_seconds_of_sweep_not_a_kernel_time": round(
            time.time() - started, 1),
        "compact_file": compact_path.name,
        "compact_sha256": _sha256_bytes(compact_path.read_bytes()),
        "raw_files": sorted(p.name for p in out.glob("raw_results_gpu*.jsonl.gz")),
    }
    (out / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(out), "operations": len(queried)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    domain = sub.add_parser("domain")
    domain.add_argument("--a800-csv", type=Path, required=True)
    domain.add_argument("--a800-manifest", type=Path, required=True)
    domain.add_argument("--output", type=Path, required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--domain", type=Path, required=True)
    capture.add_argument("--devices", required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA_ROOT)
    capture.add_argument("--max-workspace-bytes", type=int,
                         default=32 * 1024 * 1024)
    capture.add_argument("--chunk", type=int, default=2000)
    capture.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "domain":
        return _domain(args)
    return _capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
