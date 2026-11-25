#!/usr/bin/env python3
"""Build and run allocation-free cuBLASLt heuristic queries."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
REPOSITORY_ROOT = RUNNER.parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.launch_data import (  # noqa: E402
    gemm_workload_candidates,
)

from src.kernel.launch_config.gemm_workload import (  # noqa: E402
    GEMM_WORKLOAD_KEY_FIELDS,
    validate_gemm_workload_key,
)


OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / "nvidia_a800"
)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SUPPORTED_POSITIONS = frozenset({
    "qkv_proj", "gate_up_proj", "down_proj", "o_proj", "router",
    "lm_head",
})
ALGORITHM_FIELDS = frozenset({
    "rank", "algorithm_id", "tile_id", "tile", "logical_tile_m",
    "logical_tile_n", "stages_id", "stages", "inner_shape_id", "split_k",
    "reduction_scheme", "cta_swizzle", "custom_option", "workspace_bytes",
    "waves_count",
})
ALGORITHM_INTEGER_FIELDS = (
    "rank", "algorithm_id", "tile_id", "logical_tile_m", "logical_tile_n",
    "stages_id", "inner_shape_id", "split_k", "reduction_scheme",
    "cta_swizzle", "custom_option", "workspace_bytes",
)
DRY_LAUNCH_FIELDS = {
    "attempted", "replacement_kernel_submitted", "replacement_launch_count",
    "launches", "device_execution",
}
LAUNCH_FIELDS = {
    "api", "symbol", "grid", "block", "shared_memory_bytes",
    "cta_raster_log_group", "cta_raster_serpentine",
}
LAUNCH_APIS = {
    "cuLaunchKernel", "cuLaunchKernel_ptsz",
    "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz",
}
POSTFLIGHT_IDLE_ATTEMPTS = 31
POSTFLIGHT_IDLE_INTERVAL_SECONDS = 1.0
HEURISTIC_RESULT_CAPACITY = 64
DEVICE_EXECUTION_FIELDS = {
    "evidence_kind", "evidence_version", "activity_record_count",
    "executed_symbols", "all_replacements_observed", "dropped_record_count",
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


KERNEL_DATASET_GEMM = (
    REPOSITORY_ROOT / "kernel_dataset" / "nvidia_a800" / "gemm.json.zst"
)


def kernel_dataset_gemm_operations() -> tuple[tuple[Operation, ...], str]:
    """Return the GEMM launches the canonical A800 kernel dataset measures.

    Kernel evaluation prices exactly the launches this dataset captured, and
    their CTA raster map is a heuristic fact of the same selector the ShareGPT
    domain queries.  Deriving the operations from the dataset file keeps the
    extra capture domain content-addressed by that file.
    """

    import zstandard

    payload = KERNEL_DATASET_GEMM.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    entries = json.loads(zstandard.ZstdDecompressor().decompress(payload))
    operations = set()
    for case_id, entry in sorted(entries.items()):
        info = entry["op_info"]
        if info["Type"] != "GEMM":
            raise ValueError(
                f"kernel dataset case {case_id!r} is not a GEMM operation")
        attributes = info["Attr"]
        dtype = attributes["dtype"]
        if dtype not in ("bf16", "fp16"):
            raise ValueError(
                f"kernel dataset case {case_id!r} has an unsupported dtype")
        operations.add(Operation(
            attributes["B"] * attributes["M"], attributes["N"],
            attributes["K"], dtype, False))
    if not operations:
        raise ValueError("kernel dataset declares no GEMM operation")
    return tuple(sorted(operations)), digest


class WorkloadInput(NamedTuple):
    manifest: Mapping[str, object]
    manifest_bytes: bytes
    workloads: tuple[tuple[str, Operation], ...]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _compile(
        executable: Path, cuda_root: Path, source: Path = SOURCE) -> None:
    include_dir = cuda_root / "targets" / "x86_64-linux" / "include"
    library_dir = cuda_root / "targets" / "x86_64-linux" / "lib"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not include_dir.is_dir() or not library_dir.is_dir():
        raise FileNotFoundError(
            f"CUDA include/library directories do not exist below {cuda_root}"
        )
    subprocess.run(
        [
            "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            f"-I{include_dir}", str(source), f"-L{library_dir}",
            f"-Wl,-rpath,{library_dir}", "-lcublasLt", "-lcudart", "-lcupti",
            "-lcuda", "-o", str(executable),
        ],
        check=True,
    )


def _query_command(
        executable: Path, operation: Operation,
        max_workspace_bytes: int) -> list[str]:
    return [
        str(executable), str(operation.m), str(operation.n), str(operation.k),
        operation.dtype, "1" if operation.has_bias else "0",
        str(max_workspace_bytes),
    ]


def _batch_query_input(
        operations: Sequence[Operation], max_workspace_bytes: int) -> str:
    return "".join(
        f"{operation.m} {operation.n} {operation.k} {operation.dtype} "
        f"{1 if operation.has_bias else 0} {max_workspace_bytes}\n"
        for operation in operations
    )


def _validate_query_result(
        value: object, operation: Operation, max_workspace_bytes: int,
        context: str) -> Mapping[str, object]:
    result = _mapping(value, context)
    expected_fields = {
        "schema_version", "hardware_id", "device_name",
        "compute_capability", "cublaslt_version", "operation",
        "max_workspace_bytes", "matmul_executed",
        "production_kernel_submitted", "dry_launch", "algorithms",
    }
    if set(result) != expected_fields:
        raise ValueError(
            f"{context} fields differ: expected={sorted(expected_fields)}, "
            f"actual={sorted(result)}")
    if result["schema_version"] != 2:
        raise ValueError(f"{context} has unsupported schema_version")
    if result["hardware_id"] != "nvidia_a800":
        raise ValueError(f"{context} has unexpected hardware_id")
    device_name = result["device_name"]
    if not isinstance(device_name, str) or not device_name.startswith(
            "NVIDIA A800"):
        raise ValueError(f"{context} did not run on an NVIDIA A800")
    if result["compute_capability"] != "8.0":
        raise ValueError(f"{context} has unexpected compute capability")
    version = result["cublaslt_version"]
    if (isinstance(version, bool)
            or not isinstance(version, int) or version <= 0):
        raise ValueError(f"{context} has invalid cuBLASLt version")
    expected_operation = {
        "m": operation.m,
        "n": operation.n,
        "k": operation.k,
        "dtype": operation.dtype,
        "layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "has_bias": operation.has_bias,
    }
    if result["operation"] != expected_operation:
        raise ValueError(f"{context} operation does not match its request")
    if result["max_workspace_bytes"] != max_workspace_bytes:
        raise ValueError(
            f"{context} workspace limit does not match its request")
    if result["matmul_executed"] is not False:
        raise ValueError(f"{context} reports that a matmul was executed")
    if result["production_kernel_submitted"] is not False:
        raise ValueError(f"{context} submitted a production kernel")
    algorithms = _sequence(result["algorithms"], f"{context} algorithms")
    if not algorithms:
        raise ValueError(f"{context} contains no heuristic algorithms")
    if len(algorithms) >= HEURISTIC_RESULT_CAPACITY:
        raise ValueError(
            f"{context} reached the {HEURISTIC_RESULT_CAPACITY}-entry "
            "heuristic capture capacity")
    for index, value in enumerate(algorithms):
        algorithm = _mapping(value, f"{context} algorithm {index}")
        if set(algorithm) != ALGORITHM_FIELDS:
            raise ValueError(
                f"{context} algorithm {index} has unexpected fields")
        for field in ALGORITHM_INTEGER_FIELDS:
            field_value = algorithm[field]
            if (isinstance(field_value, bool)
                    or not isinstance(field_value, int) or field_value < 0):
                raise ValueError(
                    f"{context} algorithm {index} has invalid {field}")
        if algorithm["rank"] != index:
            raise ValueError(
                f"{context} algorithm ranks are not contiguous")
        if algorithm["workspace_bytes"] > max_workspace_bytes:
            raise ValueError(
                f"{context} algorithm {index} exceeds the workspace limit")
        for field in ("tile", "stages"):
            if not isinstance(algorithm[field], str) or not algorithm[field]:
                raise ValueError(
                    f"{context} algorithm {index} has invalid {field}")
        waves_count = algorithm["waves_count"]
        if (isinstance(waves_count, bool)
                or not isinstance(waves_count, (int, float))
                or not math.isfinite(waves_count) or waves_count < 0):
            raise ValueError(
                f"{context} algorithm {index} has invalid waves_count")
    rank0 = _mapping(algorithms[0], f"{context} algorithm 0")
    dry_launch = _mapping(result["dry_launch"], f"{context} dry_launch")
    if set(dry_launch) != DRY_LAUNCH_FIELDS:
        raise ValueError(f"{context} dry_launch has unexpected fields")
    attempted = dry_launch["attempted"]
    replacement_submitted = dry_launch["replacement_kernel_submitted"]
    if not isinstance(attempted, bool) or not isinstance(
            replacement_submitted, bool):
        raise TypeError(f"{context} dry-launch flags must be boolean")
    replacement_count = dry_launch["replacement_launch_count"]
    if (isinstance(replacement_count, bool)
            or not isinstance(replacement_count, int)
            or replacement_count < 0):
        raise ValueError(f"{context} has invalid replacement launch count")
    launches = _sequence(dry_launch["launches"], f"{context} launches")
    if attempted is not True or replacement_submitted is not True:
        raise ValueError(f"{context} did not dry-launch rank-0")
    if replacement_count != len(launches):
        raise ValueError(f"{context} did not replace every captured launch")
    for launch_index, value in enumerate(launches):
        launch = _mapping(value, f"{context} launch {launch_index}")
        if set(launch) != LAUNCH_FIELDS:
            raise ValueError(f"{context} launch {launch_index} fields differ")
        for field in ("api", "symbol"):
            if not isinstance(launch[field], str) or not launch[field]:
                raise ValueError(
                    f"{context} launch {launch_index} has invalid {field}")
        if launch["api"] not in LAUNCH_APIS:
            raise ValueError(
                f"{context} launch {launch_index} used an unknown API")
        if "midlayer_noop" in launch["symbol"]:
            raise ValueError(f"{context} recorded its replacement kernel")
        for field in ("grid", "block"):
            dimensions = _sequence(
                launch[field], f"{context} launch {launch_index} {field}")
            if (len(dimensions) != 3
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           or item <= 0 for item in dimensions)):
                raise ValueError(
                    f"{context} launch {launch_index} has invalid {field}")
        shared_memory = launch["shared_memory_bytes"]
        if (isinstance(shared_memory, bool)
                or not isinstance(shared_memory, int)
                or shared_memory < 0):
            raise ValueError(
                f"{context} launch {launch_index} has invalid shared memory")
    device_execution = _mapping(
        dry_launch["device_execution"], f"{context} device_execution")
    if set(device_execution) != DEVICE_EXECUTION_FIELDS:
        raise ValueError(f"{context} device_execution fields differ")
    if (device_execution["evidence_kind"] != "cupti_activity_kernel"
            or device_execution["evidence_version"] != 1
            or device_execution["all_replacements_observed"] is not True):
        raise ValueError(
            f"{context} lacks affirmative CUPTI replacement evidence")
    activity_count = device_execution["activity_record_count"]
    dropped_count = device_execution["dropped_record_count"]
    if (isinstance(activity_count, bool)
            or not isinstance(activity_count, int) or activity_count < 0
            or isinstance(dropped_count, bool)
            or not isinstance(dropped_count, int) or dropped_count != 0):
        raise ValueError(f"{context} has incomplete CUPTI activity counts")
    executed_symbols = _sequence(
        device_execution["executed_symbols"],
        f"{context} executed_symbols")
    if (activity_count != len(executed_symbols)
            or activity_count != replacement_count
            or any(symbol != "midlayer_noop" for symbol in executed_symbols)):
        raise ValueError(
            f"{context} CUPTI activity does not prove all replacements")
    reducers = [
        launch for launch in launches
        if "splitKreduce_kernel" in launch["symbol"]
    ]
    if (rank0["reduction_scheme"] in (0, 1)
            and (len(launches) != 1 or reducers)):
        raise ValueError(
            f"{context} direct/in-place execution must have one main launch")
    if (rank0["reduction_scheme"] in (2, 4)
            and (len(launches) != 2 or len(reducers) != 1
                 or launches[-1] is not reducers[0])):
        raise ValueError(
            f"{context} separate-step split must have one trailing reducer")
    return result


def _load_workloads(path: Path, models_root: Path) -> WorkloadInput:
    manifest_bytes = path.read_bytes()
    manifest = _mapping(json.loads(manifest_bytes), path.name)
    if manifest["schema_version"] != 2:
        raise ValueError("workload manifest has unsupported schema_version")
    if manifest["artifact_type"] != "gemm_workload_candidate":
        raise ValueError("input is not a gemm_workload_candidate")
    if manifest["duration_consumed"] is not False:
        raise ValueError("workload manifest consumed duration evidence")
    source_config = manifest["source_config"]
    if not isinstance(source_config, str) or not source_config:
        raise ValueError("workload manifest has invalid source_config")
    expected_manifest = gemm_workload_candidates.build_gemm_workload_candidate(
        REPOSITORY_ROOT, REPOSITORY_ROOT / source_config,
        models_root=models_root)
    if manifest != expected_manifest:
        raise ValueError(
            "workload manifest does not match its current source config, "
            "ShareGPT inventory, scheduler contract, and model configs")
    fields = _sequence(
        manifest["workload_key_fields"], "workload_key_fields")
    if tuple(fields) != GEMM_WORKLOAD_KEY_FIELDS:
        raise ValueError(
            "workload_key_fields do not match the 16-field contract")
    hardware_domains = _mapping(
        manifest["hardware_domains"], "hardware_domains")
    hardware_domain = _mapping(
        hardware_domains["nvidia_a800"], "hardware_domains.nvidia_a800")
    if hardware_domain["status"] != "exact":
        raise ValueError("nvidia_a800 workload domain is not exact")
    keys = _sequence(manifest["workload_keys"], "workload_keys")
    if not keys:
        raise ValueError("workload_keys is empty")
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("every workload key must be a string")
    if list(keys) != sorted(keys):
        raise ValueError("workload keys must be sorted")
    if len(set(keys)) != len(keys):
        raise ValueError("workload keys must be unique")
    if hardware_domain["semantic_key_count"] != len(keys):
        raise ValueError("nvidia_a800 semantic key count is inconsistent")

    workloads = []
    for index, key in enumerate(keys):
        context = f"workload_keys[{index}]"
        validate_gemm_workload_key(key)
        values = json.loads(key)
        (dtype, variant, batch, m, n, k, a_layout, b_layout, c_layout,
         trans_a, trans_b, epilogue, has_bias, position,
         tensor_parallel_size, expert_parallel_size) = values
        if batch != 1:
            raise ValueError(f"{context} has unsupported batch size {batch}")
        if variant not in ("gemm", "gemv"):
            raise ValueError(f"{context} has unsupported variant {variant!r}")
        if dtype not in ("bf16", "fp16"):
            raise ValueError(f"{context} has unsupported dtype {dtype!r}")
        if (a_layout, b_layout, c_layout) != (
                "row_major", "row_major", "row_major"):
            raise ValueError(f"{context} has unsupported matrix layout")
        if trans_a is not False or trans_b is not True:
            raise ValueError(f"{context} has unsupported transpose semantics")
        if epilogue not in ("none", "bias") or has_bias != (
                epilogue == "bias"):
            raise ValueError(f"{context} has unsupported epilogue semantics")
        if position not in SUPPORTED_POSITIONS:
            raise ValueError(
                f"{context} has unsupported position {position!r}")
        if variant == "gemv" and (m != 1 or position != "lm_head"):
            raise ValueError(f"{context} has unsupported gemv routing")
        if variant == "gemm" and m == 1 and position == "lm_head":
            raise ValueError(f"{context} bypasses the lm_head gemv routing")
        if (isinstance(tensor_parallel_size, bool)
                or not isinstance(tensor_parallel_size, int)
                or tensor_parallel_size <= 0
                or isinstance(expert_parallel_size, bool)
                or not isinstance(expert_parallel_size, int)
                or expert_parallel_size <= 0):
            raise ValueError(f"{context} has invalid parallelism")
        workloads.append((key, Operation(m, n, k, dtype, has_bias)))
    return WorkloadInput(manifest, manifest_bytes, tuple(workloads))


def _parse_devices(values: Sequence[str]) -> tuple[str, ...]:
    devices = []
    for value in values:
        devices.extend(value.split(","))
    if not devices or any(not device or device.strip() != device
                          for device in devices):
        raise ValueError("--devices must contain non-empty device identifiers")
    if any(not device.isdigit() or str(int(device)) != device
           for device in devices):
        raise ValueError("--devices must contain canonical GPU indexes")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    return tuple(devices)


def _nvidia_smi_snapshot() -> dict[str, dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True)
    snapshot = {}
    for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 5:
            raise ValueError(
                f"nvidia-smi line {line_number} must contain five fields")
        index, uuid, name, utilization_text, memory_text = fields
        if not index.isdigit() or str(int(index)) != index:
            raise ValueError(
                f"nvidia-smi line {line_number} has invalid GPU index")
        if index in snapshot:
            raise ValueError(f"nvidia-smi repeats GPU index {index}")
        if not uuid or not name:
            raise ValueError(
                f"nvidia-smi line {line_number} has empty identity fields")
        try:
            utilization = int(utilization_text)
            memory_used_mib = int(memory_text)
        except ValueError as error:
            raise ValueError(
                f"nvidia-smi line {line_number} has non-integer telemetry") \
                from error
        if utilization < 0 or memory_used_mib < 0:
            raise ValueError(
                f"nvidia-smi line {line_number} has negative telemetry")
        snapshot[index] = {
            "uuid": uuid,
            "name": name,
            "utilization_gpu_percent": utilization,
            "memory_used_mib": memory_used_mib,
        }
    if not snapshot:
        raise ValueError("nvidia-smi returned no GPUs")
    return snapshot


def _validate_device_snapshot(
        snapshot: Mapping[str, Mapping[str, object]],
        devices: Sequence[str], phase: str) -> None:
    for device in devices:
        if device not in snapshot:
            raise ValueError(
                f"selected GPU index {device} is absent from {phase} snapshot")
        record = snapshot[device]
        if not str(record["name"]).startswith("NVIDIA A800"):
            raise ValueError(
                f"selected GPU index {device} is not an NVIDIA A800")
        if (record["utilization_gpu_percent"] != 0
                or record["memory_used_mib"] != 0):
            raise DeviceBusyError(
                f"selected GPU index {device} is not idle at {phase}: "
                f"utilization.gpu={record['utilization_gpu_percent']}%, "
                f"memory.used={record['memory_used_mib']} MiB")


def _wait_for_idle_snapshot(
        devices: Sequence[str], phase: str, *,
        attempts: int = POSTFLIGHT_IDLE_ATTEMPTS,
        interval_seconds: float = POSTFLIGHT_IDLE_INTERVAL_SECONDS,
        ) -> tuple[dict[str, dict[str, object]],
                   tuple[dict[str, dict[str, object]], ...]]:
    if attempts <= 0:
        raise ValueError("idle telemetry attempts must be positive")
    if interval_seconds < 0:
        raise ValueError("idle telemetry interval must be non-negative")
    snapshots = []
    for attempt in range(attempts):
        snapshot = _nvidia_smi_snapshot()
        snapshots.append(snapshot)
        try:
            _validate_device_snapshot(snapshot, devices, phase)
        except DeviceBusyError:
            if attempt + 1 == attempts:
                raise
            time.sleep(interval_seconds)
        else:
            return snapshot, tuple(snapshots)
    raise AssertionError("idle telemetry loop exhausted without a result")


def _repository_baseline() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True)
    baseline = head.stdout.strip()
    if COMMIT_PATTERN.fullmatch(baseline) is None:
        raise ValueError(
            "git rev-parse HEAD did not return a 40-hex commit SHA")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
    if status.stdout:
        raise ValueError(
            "tracked checkout must be clean for a batch capture: "
            f"{status.stdout!r}")
    return baseline


def _require_unchanged(path: Path, expected: bytes, context: str) -> None:
    if path.read_bytes() != expected:
        raise ValueError(f"{context} changed during the batch capture")


def _run_batch_partition(
        executable: Path, device: str, operations: Sequence[Operation],
        max_workspace_bytes: int,
        ) -> tuple[tuple[Operation, Mapping[str, object]], ...]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    try:
        completed = subprocess.run(
            [str(executable)],
            input=_batch_query_input(operations, max_workspace_bytes),
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"cuBLASLt batch query failed on device {device!r}: "
            f"{error.stderr}") from error
    lines = completed.stdout.splitlines()
    if len(lines) != len(operations):
        raise ValueError(
            f"cuBLASLt batch query on device {device!r} returned "
            f"{len(lines)} results for {len(operations)} operations")
    results = []
    for operation, line in zip(operations, lines):
        try:
            raw_result = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"cuBLASLt query on device {device!r} returned invalid JSON "
                f"for operation {operation.key()}") from error
        result = _validate_query_result(
            raw_result, operation, max_workspace_bytes,
            f"device {device!r} operation {operation.key()}")
        results.append((operation, result))
    return tuple(results)


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.",
            suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink()


def _run_batch(
        args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.m is not None or args.n is not None or args.k is not None:
        parser.error("--m, --n, and --k cannot be used in batch mode")
    if args.output is None or args.devices is None:
        parser.error("batch mode requires --output and --devices")
    if args.output.exists():
        raise FileExistsError(args.output)
    output = args.output.resolve()
    try:
        output.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"batch output must be below {OUTPUT_ROOT.resolve()}") from error
    if output == OUTPUT_ROOT.resolve() or output.suffix != ".json":
        raise ValueError(
            "batch output must be a JSON file below the output root")

    baseline = _repository_baseline()
    runner_bytes = RUNNER.read_bytes()
    query_source_bytes = SOURCE.read_bytes()
    manifest_path = args.workload_manifest.resolve()
    workload_input = _load_workloads(manifest_path, args.models_root)
    workloads = workload_input.workloads
    devices = _parse_devices(args.devices)
    kernel_dataset_operations, kernel_dataset_sha256 = (
        kernel_dataset_gemm_operations())
    operations = tuple(sorted(
        {operation for _, operation in workloads}
        | set(kernel_dataset_operations)))
    if len(devices) > len(operations):
        raise ValueError("--devices cannot outnumber unique operations")

    with tempfile.TemporaryDirectory(
            prefix="midlayer-cublaslt-query-") as temporary:
        executable = Path(temporary) / "cublaslt_heuristic_query"
        source_snapshot = Path(temporary) / SOURCE.name
        source_snapshot.write_bytes(query_source_bytes)
        _compile(executable, args.cuda_root, source_snapshot)
        preflight = _nvidia_smi_snapshot()
        _validate_device_snapshot(preflight, devices, "preflight")
        partitions = tuple(
            operations[index::len(devices)] for index in range(len(devices)))
        try:
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                futures = [
                    executor.submit(
                        _run_batch_partition, executable, device, partition,
                        args.max_workspace_bytes)
                    for device, partition in zip(devices, partitions)
                ]
                queried = [
                    (device, operation, result)
                    for device, future in zip(devices, futures)
                    for operation, result in future.result()
                ]
        finally:
            postflight, postflight_attempts = _wait_for_idle_snapshot(
                devices, "postflight")

    post_baseline = _repository_baseline()
    if post_baseline != baseline:
        raise ValueError("repository HEAD changed during the batch capture")
    _require_unchanged(RUNNER, runner_bytes, "runner source")
    _require_unchanged(SOURCE, query_source_bytes, "query source")
    _require_unchanged(
        manifest_path, workload_input.manifest_bytes, "workload manifest")

    operation_results = {
        operation.key(): result for _, operation, result in queried
    }
    operation_devices = {
        operation.key(): device for device, operation, _ in queried
    }
    device_identities: dict[str, dict[str, object]] = {}
    versions = set()
    for device, _, result in queried:
        if (preflight[device]["uuid"] != postflight[device]["uuid"]
                or preflight[device]["name"] != postflight[device]["name"]):
            raise ValueError(
                f"device {device!r} identity changed between snapshots")
        if result["device_name"] != preflight[device]["name"]:
            raise ValueError(
                f"device {device!r} CUDA and nvidia-smi names disagree")
        identity = {
            "uuid": preflight[device]["uuid"],
            "device_name": preflight[device]["name"],
            "compute_capability": result["compute_capability"],
        }
        if (device in device_identities
                and device_identities[device] != identity):
            raise ValueError(
                f"device {device!r} identity changed during capture")
        device_identities[device] = identity
        versions.add(result["cublaslt_version"])
    if len(versions) != 1:
        raise ValueError(
            "queried devices reported different cuBLASLt versions")

    candidate = {
        "schema_version": 2,
        "artifact_type": "nvidia_a800_gemm_launch_candidate",
        "hardware_id": "nvidia_a800",
        "baseline_commit": baseline,
        "runner_sha256": _sha256_bytes(runner_bytes),
        "query_source_sha256": _sha256_bytes(query_source_bytes),
        "workload_manifest_sha256": _sha256_bytes(
            workload_input.manifest_bytes),
        "workload_manifest": workload_input.manifest,
        "workload_key_fields": list(GEMM_WORKLOAD_KEY_FIELDS),
        "cublaslt_version": next(iter(versions)),
        "requested_devices": list(devices),
        "device_identities": device_identities,
        "device_telemetry": {
            "command": [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            "idle_requirement": {
                "utilization_gpu_percent": 0,
                "memory_used_mib": 0,
            },
            "preflight": {
                device: preflight[device] for device in devices
            },
            "postflight": {
                device: postflight[device] for device in devices
            },
            "postflight_attempts": [
                {device: snapshot[device] for device in devices}
                for snapshot in postflight_attempts
            ],
        },
        "max_workspace_bytes": args.max_workspace_bytes,
        "selector_contract": "cublaslt_first_feasible_rank_0",
        "duration_consumed": False,
        "matmul_executed": False,
        "production_kernel_submitted": False,
        "workload_key_count": len(workloads),
        "operation_count": len(operations),
        "kernel_dataset_sha256": kernel_dataset_sha256,
        "kernel_dataset_operation_keys": [
            operation.key() for operation in kernel_dataset_operations
        ],
        "workloads": {
            key: {
                "operation_key": operation.key(),
                "result_key": operation.key(),
            }
            for key, operation in workloads
        },
        "operations": {
            key: {
                "device": operation_devices[key],
                "result": operation_results[key],
            }
            for key in sorted(operation_results)
        },
        "limitations": [
            "cuBLASLt public heuristic API does not expose final CUDA grid, "
            "block, registers, or dynamic shared memory",
            "the complete ordered heuristic list is captured without "
            "submitting a production matrix-multiplication kernel; reaching "
            "the guarded result capacity fails instead of truncating",
            "every rank-0 launch is intercepted before submission and each "
            "production kernel is replaced by a one-thread no-op",
        ],
    }
    _write_new_json(output, candidate)
    print(json.dumps({
        "output": str(output),
        "workloads": len(workloads),
        "operations": len(operations),
    }, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--has-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--max-workspace-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument(
        "--cuda-root", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--workload-manifest", type=Path)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--devices", nargs="+")
    args = parser.parse_args()
    if args.max_workspace_bytes <= 0:
        parser.error("--max-workspace-bytes must be positive")

    if args.workload_manifest is not None:
        if args.models_root is None:
            parser.error("batch mode requires --models-root")
        return _run_batch(args, parser)
    if (args.output is not None or args.devices is not None
            or args.models_root is not None):
        parser.error(
            "--output, --devices, and --models-root require "
            "--workload-manifest")
    if args.m is None or args.n is None or args.k is None:
        parser.error("single-shape mode requires --m, --n, and --k")
    for name in ("m", "n", "k"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")

    operation = Operation(args.m, args.n, args.k, args.dtype, args.has_bias)
    with tempfile.TemporaryDirectory(
            prefix="midlayer-cublaslt-query-") as temporary:
        executable = Path(temporary) / "cublaslt_heuristic_query"
        _compile(executable, args.cuda_root)
        subprocess.run(
            _query_command(executable, operation, args.max_workspace_bytes),
            check=True,
            env=os.environ.copy(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
