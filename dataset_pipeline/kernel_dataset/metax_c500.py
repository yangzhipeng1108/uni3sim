"""C500 dataset generation: profile kernels on MetaX C500 via mcTracer + mcProfiler.

Replaces NCU-based profiling for C500. Uses mcTracer to capture kernel name,
duration, grid/block dimensions, registers, shared memory. Uses mcProfiler
(when available) to capture detailed hardware metrics: pipe utilization,
memory bytes, cache hit rates, occupancy, DRAM throughput.

Usage (inside Docker container):
    python scripts/kernel_dataset/metax_c500/build_dataset.py --config configs/dataset/metax_c500.yaml --kernel gemm --stage profile
    python scripts/kernel_dataset/metax_c500/build_dataset.py --config configs/dataset/metax_c500.yaml --kernel gemm --stage aggregate
    python scripts/kernel_dataset/metax_c500/build_dataset.py --config configs/dataset/metax_c500.yaml --kernel gemm --stage aggregate --candidate-only

Stage 'profile' runs each kernel shape under mcTracer and mcProfiler.
Stage 'aggregate' parses mcTracer + mcProfiler JSONs into per-kernel NCU-format
data. ``--candidate-only`` keeps the result under profiler_output for audit;
without it, a complete dataset is published to
kernel_dataset/<hardware>/<kernel>.json.zst. Eval merges per-kernel documents in memory
(load_kernel_dataset) — no separate merge stage.

Use --skip-profiler for an explicit mcTracer-only capture pass. Aggregate remains
fail-loud and will reject cases whose required mcProfiler observations are absent.
mcProfiler raw outputs are preserved in per-shape directories; re-running 'aggregate'
re-parses from these files without re-profiling.
"""

import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager

_MX_SMI_PATH = "/opt/mxdriver/bin/mx-smi"
_MX_SMI_DEVICE_PATTERN = re.compile(
    r"^\s*GPU#(?P<device>\d+)\s+MXC500\s+"
    r"(?P<bus>[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\s*$",
    re.IGNORECASE | re.MULTILINE)
_MX_SMI_USAGE_PATTERN = re.compile(
    r"^\s*\|?\s*GPU\s*(?:\||:)\s*(\d+(?:\.\d+)?)\s*%\s*\|?\s*$",
    re.IGNORECASE | re.MULTILINE)
_MX_SMI_HBM_PATTERN = re.compile(
    r"^\s*\|?\s*throughput\s*(?:\||:)\s*"
    r"(-?\d+(?:\.\d+)?)\s*MBytes/s\s*\|?\s*$",
    re.IGNORECASE | re.MULTILINE)
_MX_SMI_NO_PROCESS_PATTERN = re.compile(
    r"^[ \t]*\|?[ \t]*"
    r"(?:No process(?: found)?|No running processes(?: found)?)\.?"
    r"[ \t]*\|?[ \t]*$", re.IGNORECASE | re.MULTILINE)
# ``mx-smi --show-hbm-bandwidth`` has reported 1 MBytes/s on
# an otherwise idle C500 (0% engines and no device process).  Keep the process
# and engine checks exact, but admit that observed telemetry floor so a
# genuinely idle card is not rejected before mcProfiler starts.
_MX_SMI_IDLE_HBM_MAX_MBYTES_PER_S = 1.0

from dataset_pipeline.kernel_dataset.config import (
    REPO_ROOT as PROJECT_ROOT,
    load_kernel_workload_config as _load_kernel_config,
    get_kernel_trace_dir,
    file_sha256 as _file_sha256,
    load_pipeline_config,
    get_kernel_pipeline_settings,
    get_hardware_config_path,
    get_kernel_config_path,
    get_kernel_script_path,
    get_profiler_output_dir,
    get_kernel_dataset_path,
    parse_kernel_list,
    require_mapping,
    require_non_negative_int,
    require_non_empty_list,
    require_non_empty_string,
    require_environment_directory,
    require_profiler_executable,
    require_positive_int,
    require_positive_number,
    current_python_executable,
)

from dataset_pipeline.kernel_dataset.dataset_writer import (
    build_kernel_evaluation_timing,
    publish_complete_dataset,
    write_complete_candidate,
)
from dataset_pipeline.kernel_dataset.workload_shapes import (
    build_fused_moe_shapes as build_common_fused_moe_shapes,
    build_flash_attn_shapes as build_common_flash_attn_shapes,
    build_vector_shapes as build_common_vector_shapes,
    expand_kernel_shapes,
)
from src.kernel.launch_config.fused_moe_tiling import derive_fused_moe_tiling
from src.kernel.launch_config.tiling_spec import (
    flash_attention_packed_varlen_scheduled_workgroups,
    gemm_logical_workgroups,
    parse_complete_gemm_tile,
    parse_flash_attention_tile,
)
from src.kernel.dataset_storage import load_kernel_dataset_document


def _parse_grid_dimensions(value):
    if not isinstance(value, str):
        raise TypeError(
            f"Metadata.Grid Size must be a string, got {type(value).__name__}"
        )
    match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*", value)
    if match is None:
        raise ValueError(
            f"Metadata.Grid Size must contain three dimensions: {value!r}"
        )
    dimensions = tuple(int(group) for group in match.groups())
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(
            f"Metadata.Grid Size dimensions must be positive: {value!r}"
        )
    return dimensions


# ─── Kernel shape builders ───────────────────────────────────

def build_flash_attn_shapes(kernel_cfg):
    result = []
    for shape in build_common_flash_attn_shapes(kernel_cfg):
        attributes = dict(shape["attr"])
        if attributes["kv_layout"] == "mla":
            # FlashMLA derives a device tensor through get_mla_metadata; there
            # is no caller-provided scalar split request on this backend.
            attributes["SP_REQUESTED"] = None
        result.append({**shape, "attr": attributes})
    return result


def build_fused_moe_shapes(kernel_cfg):
    return build_common_fused_moe_shapes(
        kernel_cfg,
        argument_names={
            "tokens": "--num_tokens", "experts": "--num_experts",
            "top_k": "--top_k", "hidden": "--hidden_size",
            "intermediate": "--intermediate_size",
        },
        include_layer=False,
    )
def build_vector_shapes(kernel_cfg):
    return build_common_vector_shapes(kernel_cfg)


def build_shapes(kernel, kernel_cfg):
    return expand_kernel_shapes(
        kernel, kernel_cfg,
        {
            "flash_attn": build_flash_attn_shapes,
            "fused_moe": build_fused_moe_shapes,
            "vector": build_vector_shapes,
        },
        unknown_kernel_message="Unsupported kernel: {kernel}",
    )


# ─── mcTracer-based profiling ────────────────────────────────

def _write_runner_script(kernel, shape, script_path, profiler_wait=0,
                         use_marker=False):
    """Write a self-contained Python script that runs the kernel once.

    profiler_wait: seconds to sleep after GPU init, giving mcProfiler's
    profiler_server time to discover the mcpti RPC port. Required because
    Python+torch startup is slow; without this the profiler misses the
    connection window. Set to 0 for mcTracer-only runs.
    """
    attr = shape["attr"]
    wait_line = f"time.sleep({profiler_wait})" if profiler_wait > 0 else ""
    marker_start = "torch.cuda.profiler.start()" if use_marker else ""
    marker_stop = "torch.cuda.profiler.stop()" if use_marker else ""

    if kernel == "gemm":
        dtype = attr["dtype"]
        if dtype == "bf16":
            torch_dtype = "torch.bfloat16"
        elif dtype in ("fp16", "float16"):
            torch_dtype = "torch.float16"
        else:
            raise ValueError(f"unsupported MetaX GEMM dtype: {dtype!r}")
        code = f"""
import time, torch
B, SL, K, N = {attr['B']}, {attr['M']}, {attr['K']}, {attr['N']}
x = torch.randn(B, SL, K, device='cuda', dtype={torch_dtype})
w = torch.nn.Linear(K, N, bias=False).cuda().to({torch_dtype})
torch.cuda.synchronize()
{wait_line}
with torch.no_grad():
    y = w(x)
torch.cuda.synchronize()
time.sleep(0.2)
{marker_start}
with torch.no_grad():
    y = w(x)
torch.cuda.synchronize()
{marker_stop}
"""
    else:
        raise ValueError(
            f"kernel {kernel!r} requires a profiler-executable runner")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)


def _runner_command(kernel, shape, script_path, python_bin, profiler_wait,
                    hardware_runner_path, use_marker=False):
    """Materialize one profiler command without importing a hardware runner."""

    if kernel == "gemm":
        _write_runner_script(
            kernel, shape, script_path, profiler_wait=profiler_wait,
            use_marker=use_marker)
        return [python_bin, "-u", script_path]
    elif kernel in ("flash_attn", "fused_moe", "vector"):
        if hardware_runner_path is None:
            raise ValueError(
                f"{kernel} requires an explicit hardware runner")
        command = [
            python_bin,
            "-u",
            os.path.abspath(hardware_runner_path),
            *shape["script_args"],
        ]
        if kernel == "flash_attn":
            requested_splits = shape["attr"]["SP_REQUESTED"]
            if requested_splits is not None:
                command += ["--num_splits", str(requested_splits)]
            command += [
                "--split-metadata-path",
                script_path.replace(".py", "_split.json")]
        elif kernel == "fused_moe":
            command += [
                "--tiling_path",
                script_path.replace(".py", "_tiling.json"),
            ]
        elif kernel == "vector":
            pass
        else:
            raise ValueError(f"unsupported MetaX runner kernel: {kernel!r}")
        command += ["--profiler_wait", str(profiler_wait)]
        if use_marker:
            command.append("--use_marker")
        return command
    else:
        raise ValueError(f"unsupported MetaX kernel: {kernel!r}")


_KERNEL_LAUNCH_APIS = frozenset(("mcLaunchKernel", "mcModuleLaunchKernel"))


def _parse_trace_json(trace_path):
    """Extract exactly the final marker-window call from one mcTracer shard.

    mcTracer records CPU runtime calls and GPU kernel execution on separate
    timelines.  The GPU event timestamp cannot define the marker window: a
    queued kernel may start after the final ``mcDeviceSynchronize`` CPU call
    has begun.  Instead, correlate every GPU kernel's ``args.co_id`` with its
    CPU ``mcLaunchKernel``/``mcModuleLaunchKernel`` event and select launches
    issued after the penultimate device synchronization completed and before
    the final device synchronization began.  This is the runner's one real
    profiled call; precompile and warmup calls are outside that interval.
    """
    try:
        with open(trace_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"invalid mcTracer capture: {trace_path}") from error

    if isinstance(data, list):
        events = data
    elif isinstance(data, dict):
        events = data["traceEvents"]
    else:
        raise TypeError(f"mcTracer root must be a list or object: {trace_path}")

    if not isinstance(events, list):
        raise TypeError(f"mcTracer traceEvents must be a list: {trace_path}")

    kernel_events = []
    launch_timestamps = {}
    device_synchronizations = []
    for event_index, ev in enumerate(events):
        require_mapping(ev, f"mcTracer event[{event_index}]")
        if "name" in ev and ev["name"] == "mcDeviceSynchronize":
            timestamp = require_positive_number(
                ev, "ts", f"mcTracer event[{event_index}] synchronize")
            duration = require_positive_number(
                ev, "dur", f"mcTracer event[{event_index}] synchronize")
            device_synchronizations.append((timestamp, duration, event_index))

        if "name" in ev and ev["name"] in _KERNEL_LAUNCH_APIS:
            if "args" not in ev:
                raise KeyError(
                    f"mcTracer event[{event_index}] {ev['name']} has no args")
            launch_args = require_mapping(
                ev["args"], f"mcTracer event[{event_index}].args")
            correlation_id = require_positive_int(
                launch_args, "co_id",
                f"mcTracer event[{event_index}] {ev['name']}")
            launch_timestamp = require_positive_number(
                ev, "ts", f"mcTracer event[{event_index}] {ev['name']}")
            if correlation_id in launch_timestamps:
                raise ValueError(
                    "mcTracer contains duplicate CPU kernel-launch correlation "
                    f"id {correlation_id} in {trace_path}")
            launch_timestamps[correlation_id] = launch_timestamp

        if "cat" not in ev or ev["cat"] != "0":
            continue
        if "args" not in ev:
            continue
        args = require_mapping(
            ev["args"], f"mcTracer event[{event_index}].args")
        if "grid" not in args:
            continue

        name = require_non_empty_string(
            ev, "name", f"mcTracer event[{event_index}]")
        grid = require_mapping(args["grid"], f"mcTracer {name}.grid")
        block = require_mapping(args["block"], f"mcTracer {name}.block")
        mem = require_mapping(args["mem"], f"mcTracer {name}.mem")
        grid_dimensions = tuple(
            require_positive_int(grid, axis, f"mcTracer {name}.grid")
            for axis in ("x", "y", "z"))
        block_dimensions = tuple(
            require_positive_int(block, axis, f"mcTracer {name}.block")
            for axis in ("x", "y", "z"))
        if "dur" not in ev:
            raise KeyError(f"mcTracer event {name} is missing duration")
        duration = ev["dur"]
        correlation_id = require_positive_int(
            args, "co_id", f"mcTracer event {name}")

        kernel_events.append({
            "name": name,
            "grid_x": grid_dimensions[0],
            "grid_y": grid_dimensions[1],
            "grid_z": grid_dimensions[2],
            "block_size": math.prod(block_dimensions),
            "regs_per_thread": require_positive_int(
                mem, "registers_per_thread", f"mcTracer {name}.mem"),
            "dynamic_smem": require_non_negative_int(
                mem, "dynamic_shared", f"mcTracer {name}.mem"),
            "static_smem": require_non_negative_int(
                mem, "static_shared", f"mcTracer {name}.mem"),
            "dur_ns": duration,
            "_co_id": correlation_id,
            "_event_index": event_index,
        })

    if not kernel_events:
        raise ValueError(f"mcTracer capture contains no kernel events: {trace_path}")
    if len(device_synchronizations) < 2:
        raise ValueError(
            "mcTracer marker-window extraction requires at least two "
            f"mcDeviceSynchronize events: {trace_path}")
    device_synchronizations.sort(key=lambda item: (item[0], item[2]))
    penultimate_sync = device_synchronizations[-2]
    final_sync = device_synchronizations[-1]
    marker_start = penultimate_sync[0] + penultimate_sync[1]
    marker_end = final_sync[0]
    if marker_start > marker_end:
        raise ValueError(
            "mcTracer marker synchronization boundaries overlap in "
            f"{trace_path}: {marker_start} > {marker_end}")
    late_launches = sorted(
        (correlation_id, timestamp)
        for correlation_id, timestamp in launch_timestamps.items()
        if timestamp >= marker_end
    )
    if late_launches:
        raise ValueError(
            "mcTracer final mcDeviceSynchronize is not terminal: CPU kernel "
            f"launches follow its start in {trace_path}: {late_launches[:8]}")

    selected = []
    for kernel_event in kernel_events:
        correlation_id = kernel_event["_co_id"]
        if correlation_id not in launch_timestamps:
            raise ValueError(
                "mcTracer GPU kernel has no correlated CPU kernel launch: "
                f"co_id={correlation_id}, kernel={kernel_event['name']!r}, "
                f"trace={trace_path}")
        launch_timestamp = launch_timestamps[correlation_id]
        if marker_start <= launch_timestamp < marker_end:
            selected.append((launch_timestamp, kernel_event["_event_index"],
                             kernel_event))
    if not selected:
        raise ValueError(
            "mcTracer final marker window contains no correlated kernel "
            f"launches: {trace_path}")

    result = []
    for _, _, kernel_event in sorted(selected):
        record = dict(kernel_event)
        del record["_co_id"]
        del record["_event_index"]
        record["dur_ns"] = require_positive_number(
            record, "dur_ns", f"mcTracer event {record['name']}")
        result.append(record)
    return result


_GRAPH_LAUNCH_API = "mcGraphLaunch"
_GRAPH_SYNCHRONIZATION_APIS = frozenset((
    "mcDeviceSynchronize", "mcEventSynchronize"))
_GRAPH_RUNTIME_APIS = (
    _KERNEL_LAUNCH_APIS
    | {_GRAPH_LAUNCH_API}
    | _GRAPH_SYNCHRONIZATION_APIS)
_GRAPH_BATCH_MANIFEST_SCHEMA_VERSION = 1
_GRAPH_BATCH_CHILD_PREFIX = "MIDlayer C500 trace batch child: "


def execute_graph_trace_batch_child(arguments):
    """Report this process identity, then exec one hardware runner."""

    if len(arguments) < 3 or arguments[1] != "--":
        raise ValueError(
            "trace batch child requires CASE_ID -- RUNNER [ARGS ...]")
    case_id = arguments[0]
    if not case_id:
        raise ValueError("trace batch child case ID must not be empty")
    command = arguments[2:]
    if not command:
        raise ValueError("trace batch child runner command must not be empty")
    print(
        _GRAPH_BATCH_CHILD_PREFIX + json.dumps({
            "case_id": case_id,
            "child_pid": os.getpid(),
        }, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    os.execvpe(command[0], command, os.environ.copy())


def wrap_graph_trace_batch_plan(plan, wrapper_path):
    wrapped = dict(plan)
    wrapped["command"] = [
        current_python_executable(),
        os.path.abspath(os.fspath(wrapper_path)),
        "--trace-batch-child",
        plan["case_id"],
        "--",
        *plan["command"],
    ]
    return wrapped


def _parse_graph_trace_batch_child(stdout, expected_case_id):
    records = []
    for line in stdout.splitlines():
        if line.startswith(_GRAPH_BATCH_CHILD_PREFIX):
            records.append(json.loads(
                line[len(_GRAPH_BATCH_CHILD_PREFIX):]))
    if len(records) != 1:
        raise ValueError(
            f"case {expected_case_id} emitted {len(records)} trace batch "
            "child PID records; expected exactly one")
    record = records[0]
    if set(record) != {"case_id", "child_pid"}:
        raise ValueError(
            f"case {expected_case_id} emitted an invalid child PID record")
    if record["case_id"] != expected_case_id:
        raise ValueError(
            f"trace batch child case mismatch: expected={expected_case_id!r}, "
            f"actual={record['case_id']!r}")
    child_pid = record["child_pid"]
    if isinstance(child_pid, bool) or not isinstance(child_pid, int):
        raise TypeError("trace batch child PID must be an integer")
    if child_pid <= 0:
        raise ValueError("trace batch child PID must be positive")
    return child_pid


def _graph_trace_launch_plan(measurement):
    repetition_counts = measurement["repetition_counts"]
    if (not isinstance(repetition_counts, list)
            or len(repetition_counts) != 2):
        raise TypeError("graph measurement repetition_counts must be a pair")
    lower_count, higher_count = repetition_counts
    sample_count = measurement["sample_count"]
    if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count <= 0):
        raise ValueError("graph measurement sample_count must be positive")
    plan = []

    def append(role, repetition_count, sample_index):
        plan.append({
            "role": role,
            "repetition_count": repetition_count,
            "sample_index": sample_index,
        })

    mode = measurement["mode"]
    if mode == "external_events_around_cuda_graph_replay":
        for repetition_count in (lower_count, higher_count):
            append("graph_warmup", repetition_count, None)
            append("event_prelude", repetition_count, None)
            for sample_index in range(sample_count):
                append("measured_sample", repetition_count, sample_index)
    elif mode == "paired_differential_events_around_cuda_graph_replay":
        append("graph_warmup", lower_count, None)
        append("graph_warmup", higher_count, None)
        append("event_prelude", lower_count, None)
        append("event_prelude", higher_count, None)
        samples = measurement["samples"]
        if not isinstance(samples, list) or len(samples) != sample_count:
            raise ValueError(
                "differential graph measurement sample list is incomplete")
        for sample_index, sample in enumerate(samples):
            if sample["sample_index"] != sample_index:
                raise ValueError(
                    "differential graph measurement sample order mismatch")
            replay_order = sample["replay_order"]
            if replay_order == "lower_then_higher":
                ordered_counts = (lower_count, higher_count)
            elif replay_order == "higher_then_lower":
                ordered_counts = (higher_count, lower_count)
            else:
                raise ValueError(
                    f"unsupported differential replay order: {replay_order!r}")
            for repetition_count in ordered_counts:
                append("measured_sample", repetition_count, sample_index)
    else:
        raise ValueError(f"unsupported graph measurement mode: {mode!r}")
    return plan


def _write_graph_trace_batch_manifest(path, payload):
    output = os.path.realpath(path)
    output_root = os.path.realpath(os.path.join(
        PROJECT_ROOT, "profiler_output/kernel_dataset/metax_c500"))
    if os.path.commonpath((output_root, output)) != output_root:
        raise ValueError(
            "trace batch manifest must stay under "
            "profiler_output/kernel_dataset/metax_c500")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=os.path.dirname(output),
                prefix=".trace-batch-manifest-", delete=False) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2, ensure_ascii=False,
                      allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def prepare_graph_trace_batch_manifest(
        manifest_path, resume, dry_run, baseline_commit,
        source_snapshot_sha256, family, plans):
    if manifest_path is None:
        return None, None
    if resume:
        raise ValueError(
            "trace batch capture cannot resume: skipped cases would refer to "
            "child PIDs from an earlier mcTracer invocation")
    output = os.path.realpath(os.fspath(manifest_path))
    output_root = os.path.realpath(os.path.join(
        PROJECT_ROOT, "profiler_output/kernel_dataset/metax_c500"))
    if os.path.commonpath((output_root, output)) != output_root:
        raise ValueError(
            "trace batch manifest must stay under "
            "profiler_output/kernel_dataset/metax_c500")
    if os.path.exists(output):
        raise FileExistsError(f"trace batch manifest already exists: {output}")
    payload = {
        "schema_version": _GRAPH_BATCH_MANIFEST_SCHEMA_VERSION,
        "hardware_id": "metax_c500",
        "profiler": "mcTracer",
        "capture_mode": "single_mctracer_invocation_multiple_child_cases",
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "family": family,
        "expected_case_count": len(plans),
        "selected_case_ids": [plan["case_id"] for plan in plans],
        "complete": False,
        "cases": [],
    }
    if not dry_run:
        _write_graph_trace_batch_manifest(output, payload)
    return output, payload


def record_graph_trace_batch_case(
        manifest_path, manifest, ordinal, plan, artifact_path, artifact,
        child_stdout):
    child_pid = _parse_graph_trace_batch_child(
        child_stdout, plan["case_id"])
    if any(case["child_pid"] == child_pid for case in manifest["cases"]):
        raise ValueError(f"trace batch reused child PID {child_pid}")
    repository_root = os.path.realpath(PROJECT_ROOT)
    event_artifact_root = os.path.realpath(os.path.join(
        repository_root, "experiment_logs/kernel_golden"))
    artifact_file = os.path.realpath(artifact_path)
    if os.path.commonpath((event_artifact_root, artifact_file)) != (
            event_artifact_root):
        raise ValueError(
            "trace batch Event artifact is outside "
            "experiment_logs/kernel_golden")
    manifest["cases"].append({
        "ordinal": ordinal,
        "case_id": plan["case_id"],
        "child_pid": child_pid,
        "canonical_attributes_sha256": plan[
            "canonical_attributes_sha256"],
        "runner_sha256": _file_sha256(plan["runner_path"]),
        "event_artifact_path": os.path.relpath(
            artifact_file, repository_root),
        "event_artifact_sha256": _file_sha256(artifact_file),
        "graph_launch_plan": _graph_trace_launch_plan(
            artifact["measurement"]),
    })
    _write_graph_trace_batch_manifest(manifest_path, manifest)


def complete_graph_trace_batch_manifest(
        manifest_path, manifest, expected_case_count):
    if manifest_path is None:
        if manifest is not None:
            raise AssertionError("trace batch manifest state is inconsistent")
        return
    if len(manifest["cases"]) != expected_case_count:
        raise AssertionError("trace batch manifest is incomplete")
    manifest["complete"] = True
    _write_graph_trace_batch_manifest(manifest_path, manifest)


def _require_graph_trace_hex(value, length, context):
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    if re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(
            f"{context} must contain exactly {length} lowercase hex digits")
    return value


def _load_graph_trace_batch_manifest(manifest_path):
    try:
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            f"invalid C500 graph trace batch manifest: {manifest_path}"
        ) from error
    require_mapping(manifest, "C500 graph trace batch manifest")
    if manifest["schema_version"] != _GRAPH_BATCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported C500 graph trace batch manifest schema: "
            f"{manifest['schema_version']!r}")
    if manifest["hardware_id"] != "metax_c500":
        raise ValueError("graph trace batch manifest hardware must be metax_c500")
    if manifest["profiler"] != "mcTracer":
        raise ValueError("graph trace batch manifest profiler must be mcTracer")
    if manifest["capture_mode"] != (
            "single_mctracer_invocation_multiple_child_cases"):
        raise ValueError("graph trace batch manifest capture mode is invalid")
    baseline_commit = _require_graph_trace_hex(
        manifest["baseline_commit"], 40,
        "graph trace batch baseline_commit")
    source_snapshot_sha256 = _require_graph_trace_hex(
        manifest["source_snapshot_sha256"], 64,
        "graph trace batch source_snapshot_sha256")
    family = require_non_empty_string(
        manifest, "family", "C500 graph trace batch manifest")
    if manifest["complete"] is not True:
        raise ValueError("graph trace batch manifest is incomplete")
    cases = require_non_empty_list(
        manifest, "cases", "C500 graph trace batch manifest")
    expected_case_count = require_positive_int(
        manifest, "expected_case_count", "C500 graph trace batch manifest")
    if len(cases) != expected_case_count:
        raise ValueError(
            "graph trace batch manifest case count mismatch: "
            f"expected={expected_case_count}, actual={len(cases)}")
    selected_case_ids = require_non_empty_list(
        manifest, "selected_case_ids", "C500 graph trace batch manifest")
    normalized = []
    seen_case_ids = set()
    seen_child_pids = set()
    for ordinal, raw_case in enumerate(cases):
        case = require_mapping(
            raw_case, f"C500 graph trace batch case[{ordinal}]")
        if require_non_negative_int(
                case, "ordinal", f"graph trace batch case[{ordinal}]") != ordinal:
            raise ValueError("graph trace batch case ordinals are not contiguous")
        case_id = require_non_empty_string(
            case, "case_id", f"graph trace batch case[{ordinal}]")
        child_pid = require_positive_int(
            case, "child_pid", f"graph trace batch case {case_id}")
        canonical_attributes_sha256 = _require_graph_trace_hex(
            case["canonical_attributes_sha256"], 64,
            f"graph trace batch case {case_id} canonical attributes SHA-256")
        runner_sha256 = _require_graph_trace_hex(
            case["runner_sha256"], 64,
            f"graph trace batch case {case_id} runner SHA-256")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate graph trace batch case ID: {case_id}")
        if child_pid in seen_child_pids:
            raise ValueError(f"duplicate graph trace batch child PID: {child_pid}")
        seen_case_ids.add(case_id)
        seen_child_pids.add(child_pid)
        launch_plan = require_non_empty_list(
            case, "graph_launch_plan", f"graph trace batch case {case_id}")
        normalized_plan = []
        for launch_ordinal, raw_launch in enumerate(launch_plan):
            launch = require_mapping(
                raw_launch,
                f"graph trace batch case {case_id} launch[{launch_ordinal}]")
            role = require_non_empty_string(
                launch, "role",
                f"graph trace batch case {case_id} launch[{launch_ordinal}]")
            if role not in {"graph_warmup", "event_prelude", "measured_sample"}:
                raise ValueError(
                    f"invalid graph launch role for {case_id}: {role!r}")
            repetition_count = require_positive_int(
                launch, "repetition_count",
                f"graph trace batch case {case_id} launch[{launch_ordinal}]")
            sample_index = launch["sample_index"]
            if role == "measured_sample":
                if (isinstance(sample_index, bool)
                        or not isinstance(sample_index, int)
                        or sample_index < 0):
                    raise ValueError(
                        f"measured graph launch sample index is invalid for "
                        f"{case_id}")
            elif sample_index is not None:
                raise ValueError(
                    f"non-sample graph launch has a sample index for {case_id}")
            normalized_plan.append({
                "role": role,
                "repetition_count": repetition_count,
                "sample_index": sample_index,
            })
        artifact_relative_path = require_non_empty_string(
            case, "event_artifact_path", f"graph trace batch case {case_id}")
        repository_root = os.path.realpath(PROJECT_ROOT)
        event_artifact_root = os.path.realpath(os.path.join(
            repository_root, "experiment_logs/kernel_golden"))
        artifact_path = os.path.realpath(os.path.join(
            repository_root, artifact_relative_path))
        try:
            common_path = os.path.commonpath(
                (event_artifact_root, artifact_path))
        except ValueError as error:
            raise ValueError(
                f"event artifact path is outside the repository: {case_id}"
            ) from error
        if common_path != event_artifact_root:
            raise ValueError(
                f"event artifact path is outside the repository: {case_id}")
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(
                f"event artifact is missing for graph trace case {case_id}: "
                f"{artifact_path}")
        artifact_sha256 = require_non_empty_string(
            case, "event_artifact_sha256",
            f"graph trace batch case {case_id}")
        _require_graph_trace_hex(
            artifact_sha256, 64,
            f"graph trace batch case {case_id} Event artifact SHA-256")
        if _file_sha256(artifact_path) != artifact_sha256:
            raise ValueError(
                f"event artifact SHA-256 mismatch for graph trace case "
                f"{case_id}")
        try:
            with open(artifact_path, encoding="utf-8") as stream:
                event_artifact = json.load(stream)
        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError(
                f"invalid Event artifact for graph trace case {case_id}: "
                f"{artifact_path}") from error
        require_mapping(event_artifact, f"Event artifact for {case_id}")
        for field, expected in (
                ("status", "measured"),
                ("case_id", case_id),
                ("family", family),
                ("hardware_id", "metax_c500"),
                ("baseline_commit", baseline_commit),
                ("source_snapshot_sha256", source_snapshot_sha256),
                ("runner_sha256", runner_sha256),
                ("canonical_attributes_sha256",
                 canonical_attributes_sha256)):
            if event_artifact[field] != expected:
                raise ValueError(
                    f"Event artifact {field} mismatch for graph trace case "
                    f"{case_id}: expected={expected!r}, "
                    f"actual={event_artifact[field]!r}")
        measurement = require_mapping(
            event_artifact["measurement"],
            f"Event artifact measurement for {case_id}")
        if _graph_trace_launch_plan(measurement) != normalized_plan:
            raise ValueError(
                f"Event artifact measurement disagrees with graph launch "
                f"plan for {case_id}")
        normalized.append({
            "ordinal": ordinal,
            "case_id": case_id,
            "child_pid": child_pid,
            "canonical_attributes_sha256": canonical_attributes_sha256,
            "runner_sha256": runner_sha256,
            "event_artifact_path": artifact_relative_path,
            "event_artifact_sha256": artifact_sha256,
            "graph_launch_plan": normalized_plan,
        })
    if selected_case_ids != [case["case_id"] for case in normalized]:
        raise ValueError(
            "graph trace batch selected case order disagrees with case records")
    return {
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "family": family,
        "cases": normalized,
    }


def _graph_trace_kernel_event(event, event_index, trace_path):
    args = require_mapping(
        event["args"], f"mcTracer event[{event_index}].args")
    name = require_non_empty_string(
        event, "name", f"mcTracer event[{event_index}]")
    grid = require_mapping(args["grid"], f"mcTracer {name}.grid")
    block = require_mapping(args["block"], f"mcTracer {name}.block")
    mem = require_mapping(args["mem"], f"mcTracer {name}.mem")
    duration = require_positive_number(
        event, "dur", f"mcTracer graph kernel {name}")
    return {
        "name": name,
        "grid_x": require_positive_int(grid, "x", f"mcTracer {name}.grid"),
        "grid_y": require_positive_int(grid, "y", f"mcTracer {name}.grid"),
        "grid_z": require_positive_int(grid, "z", f"mcTracer {name}.grid"),
        "block_size": math.prod(
            require_positive_int(block, axis, f"mcTracer {name}.block")
            for axis in ("x", "y", "z")),
        "regs_per_thread": require_positive_int(
            mem, "registers_per_thread", f"mcTracer {name}.mem"),
        "dynamic_smem": require_non_negative_int(
            mem, "dynamic_shared", f"mcTracer {name}.mem"),
        "static_smem": require_non_negative_int(
            mem, "static_shared", f"mcTracer {name}.mem"),
        "dur_ns": duration,
        "_timestamp": require_positive_number(
            event, "ts", f"mcTracer graph kernel {name}"),
        "_co_id": require_positive_int(
            args, "co_id", f"mcTracer graph kernel {name}"),
        "_event_index": event_index,
        "_trace_path": trace_path,
    }


_GRAPH_MEASURED_REPETITION_POLICY_PER_CASE_HIGH = "per-case-high"
_GRAPH_MEASURED_REPETITION_POLICY_DERIVATION = (
    "maximum_repetition_count_from_validated_case_graph_launch_plan")


def _graph_trace_measurement_selection(
        measured_repetition_count=None, measured_repetition_policy=None):
    if (measured_repetition_count is not None
            and measured_repetition_policy is not None):
        raise ValueError(
            "C500 Graph measured repetition count and policy are mutually "
            "exclusive")
    if measured_repetition_count is not None:
        if (isinstance(measured_repetition_count, bool)
                or not isinstance(measured_repetition_count, int)
                or measured_repetition_count <= 0):
            raise ValueError(
                "C500 Graph measured repetition count must be a positive "
                "integer")
        return {
            "role": "measured_sample",
            "repetition_count": measured_repetition_count,
        }
    if measured_repetition_policy is None:
        return None
    if measured_repetition_policy != (
            _GRAPH_MEASURED_REPETITION_POLICY_PER_CASE_HIGH):
        raise ValueError(
            "unsupported C500 Graph measured repetition policy: "
            f"{measured_repetition_policy!r}")
    return {
        "role": "measured_sample",
        "policy": _GRAPH_MEASURED_REPETITION_POLICY_PER_CASE_HIGH,
        "derivation": _GRAPH_MEASURED_REPETITION_POLICY_DERIVATION,
    }


def _graph_trace_case_measured_repetition_count(
        launch_plan, selection, case_id):
    if selection is None:
        return None
    if "repetition_count" in selection:
        return selection["repetition_count"]
    if selection["policy"] != _GRAPH_MEASURED_REPETITION_POLICY_PER_CASE_HIGH:
        raise ValueError(
            "unsupported C500 Graph measured repetition policy: "
            f"{selection['policy']!r}")
    measured_counts = [
        launch["repetition_count"]
        for launch in launch_plan
        if launch["role"] == "measured_sample"
    ]
    if not measured_counts:
        raise ValueError(
            "C500 Graph case has no measured repetition count in its "
            f"validated graph launch plan: {case_id}")
    return max(measured_counts)


def _parse_graph_trace_batch(
        trace_files, manifest_path, *, measured_repetition_count=None,
        measured_repetition_policy=None):
    """Split one multi-child mcTracer invocation into strict Graph cases."""

    if not trace_files:
        raise ValueError("C500 graph trace batch file list must not be empty")
    measurement_selection = _graph_trace_measurement_selection(
        measured_repetition_count, measured_repetition_policy)
    manifest = _load_graph_trace_batch_manifest(manifest_path)
    cases = manifest["cases"]
    declared_pids = {case["child_pid"] for case in cases}
    runtime_pids = set()
    graph_launches = defaultdict(list)
    graph_launch_by_correlation = defaultdict(list)
    synchronizations = defaultdict(list)
    gpu_kernels = []
    process_trace_files = defaultdict(set)
    for trace_path in sorted(trace_files):
        try:
            with open(trace_path, encoding="utf-8") as stream:
                payload = json.load(stream)
        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError(
                f"invalid mcTracer graph batch capture: {trace_path}"
            ) from error
        if isinstance(payload, list):
            events = payload
        elif isinstance(payload, dict):
            events = payload["traceEvents"]
        else:
            raise TypeError(
                f"mcTracer graph batch root must be a list or object: "
                f"{trace_path}")
        if not isinstance(events, list):
            raise TypeError(
                f"mcTracer graph batch traceEvents must be a list: "
                f"{trace_path}")
        for event_index, event in enumerate(events):
            require_mapping(event, f"mcTracer event[{event_index}]")
            if "name" in event and event["name"] in _GRAPH_RUNTIME_APIS:
                if "args" not in event:
                    raise KeyError(
                        f"mcTracer event[{event_index}] {event['name']} "
                        "has no args")
                args = require_mapping(
                    event["args"], f"mcTracer event[{event_index}].args")
                process_id = require_positive_int(
                    args, "process_id",
                    f"mcTracer event[{event_index}] {event['name']}")
                runtime_pids.add(process_id)
                process_trace_files[process_id].add(trace_path)
                if event["name"] == _GRAPH_LAUNCH_API:
                    correlation_id = require_positive_int(
                        args, "co_id",
                        f"mcTracer event[{event_index}] mcGraphLaunch")
                    timestamp = require_positive_number(
                        event, "ts",
                        f"mcTracer event[{event_index}] mcGraphLaunch")
                    duration = require_positive_number(
                        event, "dur",
                        f"mcTracer event[{event_index}] mcGraphLaunch")
                    launch = {
                        "process_id": process_id,
                        "co_id": correlation_id,
                        "timestamp": timestamp,
                        "duration": duration,
                        "event_index": event_index,
                        "trace_path": trace_path,
                    }
                    graph_launches[process_id].append(launch)
                    graph_launch_by_correlation[
                        (os.path.realpath(trace_path), correlation_id)
                    ].append(launch)
                elif event["name"] in _GRAPH_SYNCHRONIZATION_APIS:
                    synchronizations[process_id].append({
                        "timestamp": require_positive_number(
                            event, "ts",
                            f"mcTracer event[{event_index}] synchronize"),
                        "duration": require_positive_number(
                            event, "dur",
                            f"mcTracer event[{event_index}] synchronize"),
                        "event_index": event_index,
                        "trace_path": trace_path,
                    })
            if ("cat" in event and event["cat"] == "0"
                    and "args" in event):
                args = require_mapping(
                    event["args"], f"mcTracer event[{event_index}].args")
                if "grid" in args:
                    gpu_kernels.append(_graph_trace_kernel_event(
                        event, event_index, trace_path))

    undeclared_pids = runtime_pids - declared_pids
    if undeclared_pids:
        raise ValueError(
            "mcTracer graph batch contains undeclared child PIDs: "
            f"{sorted(undeclared_pids)}")
    missing_pids = declared_pids - runtime_pids
    if missing_pids:
        raise ValueError(
            "mcTracer graph batch is missing declared child PIDs: "
            f"{sorted(missing_pids)}")
    for process_id, paths in process_trace_files.items():
        if len(paths) != 1:
            raise ValueError(
                f"mcTracer graph batch child PID {process_id} appears in "
                f"multiple trace files: {sorted(paths)}")

    kernels_by_launch = defaultdict(list)
    for kernel in gpu_kernels:
        candidates = graph_launch_by_correlation[
            (os.path.realpath(kernel["_trace_path"]), kernel["_co_id"])]
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(
                "mcTracer graph kernel correlation ID is ambiguous within "
                f"one trace: trace={kernel['_trace_path']!r}, "
                f"co_id={kernel['_co_id']}")
        launch = candidates[0]
        kernels_by_launch[(launch["process_id"], launch["co_id"])].append(
            kernel)

    results = {}
    previous_case_start = None
    for case in cases:
        case_id = case["case_id"]
        process_id = case["child_pid"]
        launches = sorted(
            graph_launches[process_id],
            key=lambda launch: (
                launch["timestamp"], launch["event_index"]))
        launch_plan = case["graph_launch_plan"]
        selected_repetition_count = (
            _graph_trace_case_measured_repetition_count(
                launch_plan, measurement_selection, case_id))
        if len(launches) != len(launch_plan):
            raise ValueError(
                f"mcTracer graph launch count mismatch for {case_id}: "
                f"expected={len(launch_plan)}, actual={len(launches)}")
        case_start = launches[0]["timestamp"]
        if previous_case_start is not None and case_start <= previous_case_start:
            raise ValueError(
                "mcTracer graph child execution order disagrees with the "
                f"manifest at case {case_id}")
        previous_case_start = case_start
        boundaries = sorted(
            synchronizations[process_id],
            key=lambda boundary: (
                boundary["timestamp"], boundary["event_index"]))
        if not boundaries:
            raise ValueError(
                f"mcTracer graph case has no synchronization boundaries: "
                f"{case_id}")
        if not any(
                boundary["timestamp"] + boundary["duration"]
                <= launches[0]["timestamp"] for boundary in boundaries):
            raise ValueError(
                f"mcTracer graph case has no completed pre-launch "
                f"synchronization boundary: {case_id}")
        expected_sequence = None
        measured_observations = None
        measured_launch_count = 0
        for launch_ordinal, (launch, planned) in enumerate(zip(
                launches, launch_plan, strict=True)):
            next_timestamp = (
                launches[launch_ordinal + 1]["timestamp"]
                if launch_ordinal + 1 < len(launches) else None)
            following_boundaries = [
                boundary for boundary in boundaries
                if boundary["timestamp"] >= (
                    launch["timestamp"] + launch["duration"])
                and (next_timestamp is None
                     or boundary["timestamp"] < next_timestamp)
            ]
            if not following_boundaries:
                raise ValueError(
                    f"mcTracer graph launch {launch_ordinal} for {case_id} "
                    "has no unique following synchronization boundary")
            launch_kernels = sorted(
                kernels_by_launch[(process_id, launch["co_id"])],
                key=lambda kernel: (
                    kernel["_timestamp"], kernel["_event_index"]))
            if not launch_kernels:
                raise ValueError(
                    f"mcTracer graph launch {launch_ordinal} for {case_id} "
                    "has no correlated GPU kernels")
            repetition_count = planned["repetition_count"]
            if len(launch_kernels) % repetition_count != 0:
                raise ValueError(
                    f"mcTracer graph launch {launch_ordinal} for {case_id} "
                    "cannot be divided by its declared repetition count")
            kernels_per_call = len(launch_kernels) // repetition_count
            chunks = [
                launch_kernels[offset:offset + kernels_per_call]
                for offset in range(0, len(launch_kernels), kernels_per_call)
            ]
            sequence = tuple(
                (kernel["name"], _trace_launch_signature(kernel))
                for kernel in chunks[0])
            if any(tuple(
                    (kernel["name"], _trace_launch_signature(kernel))
                    for kernel in chunk) != sequence for chunk in chunks[1:]):
                raise ValueError(
                    f"mcTracer graph launch {launch_ordinal} for {case_id} "
                    "has a non-repeating per-call kernel sequence")
            if expected_sequence is None:
                expected_sequence = sequence
                measured_observations = [
                    [] for _ in range(kernels_per_call)]
            elif sequence != expected_sequence:
                raise ValueError(
                    f"mcTracer graph launches disagree on the complete "
                    f"per-call kernel sequence for {case_id}")
            if (planned["role"] == "measured_sample"
                    and (selected_repetition_count is None
                         or repetition_count == selected_repetition_count)):
                measured_launch_count += 1
                for chunk in chunks:
                    for kernel_ordinal, kernel in enumerate(chunk):
                        measured_observations[kernel_ordinal].append(kernel)
        if expected_sequence is None or measured_observations is None:
            raise AssertionError("graph launch sequence was not initialized")
        if measured_launch_count == 0:
            raise ValueError(
                f"mcTracer graph case has no measured samples: {case_id}")
        kernel_events = []
        observation_counts = []
        for observations in measured_observations:
            if not observations:
                raise ValueError(
                    f"mcTracer graph case has an empty measured kernel "
                    f"ordinal: {case_id}")
            representative = dict(observations[0])
            for internal_key in (
                    "_co_id", "_event_index", "_timestamp", "_trace_path"):
                del representative[internal_key]
            representative["dur_ns"] = statistics.median(
                observation["dur_ns"] for observation in observations)
            kernel_events.append(representative)
            observation_counts.append(len(observations))
        result = {
            "child_pid": process_id,
            "canonical_attributes_sha256": case[
                "canonical_attributes_sha256"],
            "runner_sha256": case["runner_sha256"],
            "event_artifact_path": case["event_artifact_path"],
            "event_artifact_sha256": case["event_artifact_sha256"],
            "measured_graph_launch_count": measured_launch_count,
            "per_kernel_observation_count": observation_counts,
            "kernel_events": kernel_events,
        }
        if measured_repetition_policy is not None:
            result["selected_measured_repetition_count"] = (
                selected_repetition_count)
        results[case_id] = result
    return results


def _write_graph_trace_batch_candidate(
        cfg, kernel, trace_dir, manifest_path, output_path,
        *, measured_repetition_count=None, measured_repetition_policy=None):
    """Parse one complete Graph mcTracer batch into an auditable candidate."""

    repository_root = os.path.realpath(PROJECT_ROOT)
    profiler_root = os.path.realpath(get_profiler_output_dir(cfg))
    trace_root = os.path.realpath(trace_dir)
    manifest = os.path.realpath(manifest_path)
    output = os.path.realpath(output_path)
    candidate_root = os.path.abspath(os.path.join(profiler_root, "candidate"))
    resolved_candidate_root = os.path.realpath(candidate_root)
    if (os.path.islink(candidate_root)
            or resolved_candidate_root != candidate_root):
        raise ValueError(
            "graph trace batch candidate root must not be a symlink or "
            f"escape its profiler owner: {candidate_root}")
    for path, label, required_root in (
            (trace_root, "graph trace directory", profiler_root),
            (manifest, "graph trace batch manifest", profiler_root),
            (output, "graph trace batch candidate", candidate_root)):
        try:
            common_path = os.path.commonpath((required_root, path))
        except ValueError as error:
            raise ValueError(f"{label} is outside {required_root}") from error
        if common_path != required_root:
            raise ValueError(f"{label} is outside {required_root}")
    if not os.path.isdir(trace_root):
        raise FileNotFoundError(
            f"C500 graph trace batch directory is missing: {trace_root}")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(
            f"C500 graph trace batch manifest is missing: {manifest}")
    if not output.endswith(".json"):
        raise ValueError("graph trace batch candidate must be a JSON file")
    trace_files = sorted(os.path.realpath(path) for path in glob.glob(
        os.path.join(trace_root, "tracer_out*.json")))
    if not trace_files:
        raise FileNotFoundError(
            f"C500 graph trace batch has no tracer_out JSON: {trace_root}")
    if len(trace_files) != len(set(trace_files)):
        raise ValueError("C500 graph trace batch contains duplicate trace inputs")
    for trace_path in trace_files:
        if os.path.commonpath((trace_root, trace_path)) != trace_root:
            raise ValueError(
                f"graph trace input escapes its trace directory: {trace_path}")
    if output == manifest or output in trace_files:
        raise ValueError(
            "graph trace batch candidate output conflicts with an input")
    validated_manifest = _load_graph_trace_batch_manifest(manifest)
    if validated_manifest["family"] != kernel:
        raise ValueError(
            "graph trace batch family mismatch: "
            f"expected={kernel!r}, "
            f"actual={validated_manifest['family']!r}")
    measurement_selection = _graph_trace_measurement_selection(
        measured_repetition_count, measured_repetition_policy)
    parsed_cases = _parse_graph_trace_batch(
        trace_files, manifest,
        measured_repetition_count=measured_repetition_count,
        measured_repetition_policy=measured_repetition_policy)
    payload = {
        "schema_version": (
            3 if measured_repetition_policy is not None
            else 2 if measurement_selection is not None
            else 1),
        "status": "candidate",
        "hardware_id": "metax_c500",
        "family": kernel,
        "baseline_commit": validated_manifest["baseline_commit"],
        "source_snapshot_sha256": validated_manifest[
            "source_snapshot_sha256"],
        "timing_source": "mcTracer_graph_kernel_duration",
        "event_role": "cross_validation_only",
        "trace_batch_manifest": {
            "path": os.path.relpath(manifest, repository_root),
            "sha256": _file_sha256(manifest),
        },
        "trace_files": [{
            "path": os.path.relpath(path, repository_root),
            "sha256": _file_sha256(path),
        } for path in trace_files],
        "case_count": len(parsed_cases),
        "cases": parsed_cases,
    }
    if measurement_selection is not None:
        payload["measurement_selection"] = measurement_selection
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=os.path.dirname(output),
                prefix=".graph-trace-batch-candidate-",
                delete=False) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2, ensure_ascii=False,
                      allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output


_GRAPH_TRACE_BATCH_CANDIDATE_KEYS_V1 = frozenset((
    "schema_version", "status", "hardware_id", "family",
    "baseline_commit", "source_snapshot_sha256", "timing_source",
    "event_role", "trace_batch_manifest", "trace_files", "case_count",
    "cases",
))

_GRAPH_TRACE_BATCH_CANDIDATE_KEYS_V2 = (
    _GRAPH_TRACE_BATCH_CANDIDATE_KEYS_V1 | {"measurement_selection"})
_GRAPH_TRACE_BATCH_CANDIDATE_KEYS_V3 = _GRAPH_TRACE_BATCH_CANDIDATE_KEYS_V2


def _load_graph_trace_measurement_selection(payload):
    selection = require_mapping(
        payload["measurement_selection"],
        "C500 graph trace batch measurement selection")
    if set(selection) != {"role", "repetition_count"}:
        raise ValueError(
            "C500 graph trace batch measurement selection has invalid keys")
    if selection["role"] != "measured_sample":
        raise ValueError(
            "C500 graph trace batch measurement selection role must be "
            "measured_sample")
    return require_positive_int(
        selection, "repetition_count",
        "C500 graph trace batch measurement selection")


def _load_graph_trace_measurement_policy(payload):
    selection = require_mapping(
        payload["measurement_selection"],
        "C500 graph trace batch measurement selection")
    if set(selection) != {"role", "policy", "derivation"}:
        raise ValueError(
            "C500 graph trace batch measurement selection has invalid keys")
    if selection["role"] != "measured_sample":
        raise ValueError(
            "C500 graph trace batch measurement selection role must be "
            "measured_sample")
    policy = require_non_empty_string(
        selection, "policy",
        "C500 graph trace batch measurement selection")
    if policy != _GRAPH_MEASURED_REPETITION_POLICY_PER_CASE_HIGH:
        raise ValueError(
            "unsupported C500 Graph measured repetition policy: "
            f"{policy!r}")
    if selection["derivation"] != (
            _GRAPH_MEASURED_REPETITION_POLICY_DERIVATION):
        raise ValueError(
            "C500 Graph measured repetition policy derivation mismatch")
    return policy


def _canonical_attributes_sha256(attributes):
    require_mapping(attributes, "canonical kernel attributes")
    encoded = json.dumps(
        attributes, separators=(",", ":"), sort_keys=True,
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_graph_launch_sequence(canonical_entry, case_id):
    """Return the canonical kernel name and complete launch sequence."""

    kernels = require_non_empty_list(
        canonical_entry, "kernels", f"C500 canonical entry {case_id}")
    sequence = []
    for ordinal, raw_kernel in enumerate(kernels):
        kernel = require_mapping(
            raw_kernel, f"C500 canonical kernel {case_id}[{ordinal}]")
        metadata = require_mapping(
            kernel["Metadata"],
            f"C500 canonical kernel metadata {case_id}[{ordinal}]")
        name = require_non_empty_string(
            metadata, "Kernel Name",
            f"C500 canonical kernel {case_id}[{ordinal}]")
        grid = _parse_grid_dimensions(metadata["Grid Size"])
        block = _parse_grid_dimensions(metadata["Block Size"])
        launch_statistics = require_mapping(
            require_mapping(
                kernel["Sections"],
                f"C500 canonical kernel sections {case_id}[{ordinal}]")[
                    "Launch Statistics"],
            f"C500 canonical kernel launch statistics {case_id}[{ordinal}]")
        launch_header = require_mapping(
            launch_statistics["Header"],
            f"C500 canonical kernel launch header {case_id}[{ordinal}]")
        sequence.append((name, (
            grid[0], grid[1], grid[2], math.prod(block),
            require_positive_int(
                launch_header, "Registers Per Thread [register/thread]",
                f"C500 canonical kernel {case_id}[{ordinal}]"),
            require_non_negative_int(
                launch_header,
                "Dynamic Shared Memory Per Block [byte/block]",
                f"C500 canonical kernel {case_id}[{ordinal}]"),
            require_non_negative_int(
                launch_header,
                "Static Shared Memory Per Block [byte/block]",
                f"C500 canonical kernel {case_id}[{ordinal}]"),
        )))
    return tuple(sequence)


def _graph_candidate_launch_sequence(case, case_id):
    events = require_non_empty_list(
        case, "kernel_events", f"C500 graph trace candidate {case_id}")
    sequence = []
    for ordinal, raw_event in enumerate(events):
        event = require_mapping(
            raw_event, f"C500 graph trace candidate kernel {case_id}[{ordinal}]")
        sequence.append((
            require_non_empty_string(
                event, "name",
                f"C500 graph trace candidate kernel {case_id}[{ordinal}]"),
            (
                require_positive_int(
                    event, "grid_x",
                    f"C500 graph trace candidate kernel {case_id}[{ordinal}]"),
                require_positive_int(
                    event, "grid_y",
                    f"C500 graph trace candidate kernel {case_id}[{ordinal}]"),
                require_positive_int(
                    event, "grid_z",
                    f"C500 graph trace candidate kernel {case_id}[{ordinal}]"),
                require_positive_int(
                    event, "block_size",
                    f"C500 graph trace candidate kernel {case_id}[{ordinal}]"),
                require_positive_int(
                    event, "regs_per_thread",
                    f"C500 graph trace candidate kernel {case_id}[{ordinal}]"),
                require_non_negative_int(
                    event, "dynamic_smem",
                    f"C500 graph trace candidate kernel {case_id}[{ordinal}]"),
                require_non_negative_int(
                    event, "static_smem",
                    f"C500 graph trace candidate kernel {case_id}[{ordinal}]"),
            )))
    return tuple(sequence)


def _resolve_graph_trace_candidate_reference(
        repository_root, profiler_root, relative_path, label):
    if not isinstance(relative_path, str) or not relative_path:
        raise TypeError(f"{label} path must be a non-empty string")
    if os.path.isabs(relative_path):
        raise ValueError(f"{label} path must be repository-relative")
    resolved = os.path.realpath(os.path.join(repository_root, relative_path))
    try:
        common_path = os.path.commonpath((profiler_root, resolved))
    except ValueError as error:
        raise ValueError(f"{label} path is outside the profiler owner") from error
    if common_path != profiler_root:
        raise ValueError(f"{label} path is outside the profiler owner")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _load_graph_trace_batch_candidate(
        cfg, kernel, candidate_path):
    """Re-validate one persisted batch candidate from its raw evidence."""

    repository_root = os.path.realpath(PROJECT_ROOT)
    profiler_root = os.path.realpath(get_profiler_output_dir(cfg))
    candidate_root = os.path.abspath(os.path.join(profiler_root, "candidate"))
    resolved_candidate_root = os.path.realpath(candidate_root)
    if (os.path.islink(candidate_root)
            or resolved_candidate_root != candidate_root):
        raise ValueError(
            "graph trace batch candidate root must not be a symlink or "
            f"escape its profiler owner: {candidate_root}")
    resolved_candidate = os.path.realpath(candidate_path)
    try:
        common_path = os.path.commonpath(
            (resolved_candidate_root, resolved_candidate))
    except ValueError as error:
        raise ValueError(
            "graph trace batch candidate is outside its candidate owner") from error
    if common_path != resolved_candidate_root:
        raise ValueError(
            "graph trace batch candidate is outside its candidate owner")
    if not resolved_candidate.endswith(".json"):
        raise ValueError("graph trace batch candidate must be a JSON file")
    try:
        with open(resolved_candidate, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            f"invalid C500 graph trace batch candidate: {resolved_candidate}") from error
    require_mapping(payload, "C500 graph trace batch candidate")
    schema_version = payload["schema_version"]
    if schema_version == 1:
        expected_keys = _GRAPH_TRACE_BATCH_CANDIDATE_KEYS_V1
        measured_repetition_count = None
        measured_repetition_policy = None
    elif schema_version == 2:
        expected_keys = _GRAPH_TRACE_BATCH_CANDIDATE_KEYS_V2
        measured_repetition_count = _load_graph_trace_measurement_selection(
            payload)
        measured_repetition_policy = None
    elif schema_version == 3:
        expected_keys = _GRAPH_TRACE_BATCH_CANDIDATE_KEYS_V3
        measured_repetition_count = None
        measured_repetition_policy = _load_graph_trace_measurement_policy(
            payload)
    else:
        raise ValueError(
            "unsupported C500 graph trace batch candidate schema: "
            f"{schema_version!r}")
    if set(payload) != expected_keys:
        raise ValueError(
            "C500 graph trace batch candidate has unexpected schema keys: "
            f"{sorted(set(payload) ^ expected_keys)}")
    for field, expected in (
            ("status", "candidate"),
            ("hardware_id", "metax_c500"),
            ("family", kernel),
            ("timing_source", "mcTracer_graph_kernel_duration"),
            ("event_role", "cross_validation_only")):
        if payload[field] != expected:
            raise ValueError(
                f"C500 graph trace batch candidate {field} mismatch: "
                f"expected={expected!r}, actual={payload[field]!r}")
    baseline_commit = _require_graph_trace_hex(
        payload["baseline_commit"], 40,
        "graph trace batch candidate baseline_commit")
    source_snapshot_sha256 = _require_graph_trace_hex(
        payload["source_snapshot_sha256"], 64,
        "graph trace batch candidate source_snapshot_sha256")
    manifest_record = require_mapping(
        payload["trace_batch_manifest"],
        "graph trace batch candidate manifest record")
    if set(manifest_record) != {"path", "sha256"}:
        raise ValueError("graph trace batch candidate manifest record is invalid")
    manifest_path = _resolve_graph_trace_candidate_reference(
        repository_root, profiler_root, manifest_record["path"],
        "graph trace batch manifest")
    manifest_sha256 = _require_graph_trace_hex(
        manifest_record["sha256"], 64,
        "graph trace batch candidate manifest SHA-256")
    if _file_sha256(manifest_path) != manifest_sha256:
        raise ValueError(
            "graph trace batch candidate manifest SHA-256 mismatch: "
            f"{manifest_path}")
    trace_records = require_non_empty_list(
        payload, "trace_files", "graph trace batch candidate")
    trace_paths = []
    seen_trace_paths = set()
    for index, raw_record in enumerate(trace_records):
        record = require_mapping(
            raw_record, f"graph trace batch candidate trace[{index}]")
        if set(record) != {"path", "sha256"}:
            raise ValueError(
                "graph trace batch candidate trace record is invalid: "
                f"index={index}")
        trace_path = _resolve_graph_trace_candidate_reference(
            repository_root, profiler_root, record["path"],
            f"graph trace batch trace[{index}]")
        if trace_path in seen_trace_paths:
            raise ValueError(
                "graph trace batch candidate has duplicate trace references")
        seen_trace_paths.add(trace_path)
        trace_sha256 = _require_graph_trace_hex(
            record["sha256"], 64,
            f"graph trace batch candidate trace[{index}] SHA-256")
        if _file_sha256(trace_path) != trace_sha256:
            raise ValueError(
                "graph trace batch candidate trace SHA-256 mismatch: "
                f"{trace_path}")
        trace_paths.append(trace_path)
    manifest = _load_graph_trace_batch_manifest(manifest_path)
    for field, expected in (
            ("baseline_commit", baseline_commit),
            ("source_snapshot_sha256", source_snapshot_sha256),
            ("family", kernel)):
        if manifest[field] != expected:
            raise ValueError(
                f"graph trace batch manifest {field} disagrees with its "
                f"candidate: {resolved_candidate}")
    parsed_cases = _parse_graph_trace_batch(
        tuple(trace_paths), manifest_path,
        measured_repetition_count=measured_repetition_count,
        measured_repetition_policy=measured_repetition_policy)
    candidate_cases = require_mapping(
        payload["cases"], "graph trace batch candidate cases")
    if candidate_cases != parsed_cases:
        raise ValueError(
            "graph trace batch candidate cases disagree with re-parsed raw "
            f"evidence: {resolved_candidate}")
    if payload["case_count"] != len(parsed_cases):
        raise ValueError(
            "graph trace batch candidate case count disagrees with raw "
            f"evidence: {resolved_candidate}")
    return {
        "path": resolved_candidate,
        "sha256": _file_sha256(resolved_candidate),
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "measurement_selection": _graph_trace_measurement_selection(
            measured_repetition_count, measured_repetition_policy),
        "cases": parsed_cases,
    }


def _write_graph_trace_family_candidate(
        cfg, kernel, batch_candidate_paths, output_path):
    """Merge complete, independently re-validated Graph batches by case ID.

    Each input remains tied to its own mcTracer manifest, Event artifacts, and
    raw trace files.  This function does not average or otherwise combine
    observations across batches: it only accepts an exact, non-overlapping
    partition of the current canonical case IDs.
    """

    if (not isinstance(batch_candidate_paths, list)
            or not batch_candidate_paths):
        raise ValueError("graph trace family candidate needs non-empty batches")
    repository_root = os.path.realpath(PROJECT_ROOT)
    profiler_root = os.path.realpath(get_profiler_output_dir(cfg))
    candidate_root = os.path.abspath(os.path.join(profiler_root, "candidate"))
    resolved_candidate_root = os.path.realpath(candidate_root)
    if (os.path.islink(candidate_root)
            or resolved_candidate_root != candidate_root):
        raise ValueError(
            "graph trace family candidate root must not be a symlink or "
            f"escape its profiler owner: {candidate_root}")
    output = os.path.realpath(output_path)
    try:
        output_common_path = os.path.commonpath(
            (resolved_candidate_root, output))
    except ValueError as error:
        raise ValueError(
            "graph trace family candidate output is outside its candidate "
            "owner") from error
    if output_common_path != resolved_candidate_root:
        raise ValueError(
            "graph trace family candidate output is outside its candidate "
            "owner")
    if not output.endswith(".json"):
        raise ValueError("graph trace family candidate output must be a JSON file")
    loaded_batches = []
    seen_batch_paths = set()
    for raw_path in batch_candidate_paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise TypeError("graph trace batch candidate path must be a string")
        batch = _load_graph_trace_batch_candidate(cfg, kernel, raw_path)
        if batch["path"] == output:
            raise ValueError(
                "graph trace family candidate output conflicts with an input")
        if batch["path"] in seen_batch_paths:
            raise ValueError("graph trace family candidate repeats an input")
        seen_batch_paths.add(batch["path"])
        loaded_batches.append(batch)
    loaded_batches.sort(key=lambda batch: batch["path"])
    baseline_commit = loaded_batches[0]["baseline_commit"]
    source_snapshot_sha256 = loaded_batches[0]["source_snapshot_sha256"]
    measurement_selection = loaded_batches[0]["measurement_selection"]
    merged_cases = {}
    for batch in loaded_batches:
        for field, expected in (
                ("baseline_commit", baseline_commit),
                ("source_snapshot_sha256", source_snapshot_sha256),
                ("measurement_selection", measurement_selection)):
            if batch[field] != expected:
                raise ValueError(
                    "graph trace family candidate mixes incompatible "
                    f"{field} values")
        for case_id, case in batch["cases"].items():
            if case_id in merged_cases:
                raise ValueError(
                    "graph trace family candidate has duplicate case ID: "
                    f"{case_id}")
            merged_cases[case_id] = case
    dataset_path = os.path.realpath(get_kernel_dataset_path(cfg, kernel))
    try:
        dataset_common_path = os.path.commonpath(
            (repository_root, dataset_path))
    except ValueError as error:
        raise ValueError("C500 canonical dataset is outside the repository") from error
    if dataset_common_path != repository_root:
        raise ValueError("C500 canonical dataset is outside the repository")
    try:
        canonical_dataset = load_kernel_dataset_document(dataset_path)
    except (ValueError, OSError) as error:
        raise RuntimeError(
            f"invalid C500 canonical dataset: {dataset_path}") from error
    require_mapping(canonical_dataset, "C500 canonical dataset")
    if not canonical_dataset:
        raise ValueError("C500 canonical dataset must not be empty")
    expected_case_ids = set(canonical_dataset)
    actual_case_ids = set(merged_cases)
    missing_case_ids = sorted(expected_case_ids - actual_case_ids)
    unexpected_case_ids = sorted(actual_case_ids - expected_case_ids)
    if missing_case_ids or unexpected_case_ids:
        raise ValueError(
            "graph trace family candidate is not an exact canonical case "
            f"partition: missing={missing_case_ids[:8]}, "
            f"unexpected={unexpected_case_ids[:8]}")
    for case_id, case in merged_cases.items():
        canonical_entry = require_mapping(
            canonical_dataset[case_id],
            f"C500 canonical entry {case_id}")
        canonical_op_info = require_mapping(
            canonical_entry["op_info"],
            f"C500 canonical op_info {case_id}")
        canonical_attributes = require_mapping(
            canonical_op_info["Attr"],
            f"C500 canonical Attr {case_id}")
        candidate_attributes_sha256 = _require_graph_trace_hex(
            case["canonical_attributes_sha256"], 64,
            f"C500 graph trace candidate Attr SHA-256 {case_id}")
        if _canonical_attributes_sha256(canonical_attributes) != (
                candidate_attributes_sha256):
            raise ValueError(
                "graph trace family candidate canonical Attr mismatch: "
                f"case={case_id}")
        canonical_sequence = _canonical_graph_launch_sequence(
            canonical_entry, case_id)
        candidate_sequence = _graph_candidate_launch_sequence(case, case_id)
        if canonical_sequence != candidate_sequence:
            raise ValueError(
                "graph trace family candidate canonical launch mismatch: "
                f"case={case_id}")
    ordered_cases = {
        case_id: merged_cases[case_id]
        for case_id in canonical_dataset
    }
    payload = {
        "schema_version": (
            3 if (measurement_selection is not None
                  and "policy" in measurement_selection)
            else 2 if measurement_selection is not None
            else 1),
        "status": "candidate",
        "hardware_id": "metax_c500",
        "family": kernel,
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "timing_source": "mcTracer_graph_kernel_duration",
        "event_role": "cross_validation_only",
        "canonical_dataset": {
            "path": os.path.relpath(dataset_path, repository_root),
            "sha256": _file_sha256(dataset_path),
            "case_count": len(canonical_dataset),
        },
        "batch_candidates": [{
            "path": os.path.relpath(batch["path"], repository_root),
            "sha256": batch["sha256"],
        } for batch in loaded_batches],
        "case_count": len(ordered_cases),
        "cases": ordered_cases,
    }
    if measurement_selection is not None:
        payload["measurement_selection"] = measurement_selection
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=os.path.dirname(output),
                prefix=".graph-trace-family-candidate-",
                delete=False) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2, ensure_ascii=False,
                      allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output


def _observed_flash_attn_splits(shape, trace_dir):
    attributes = shape["attr"]
    if attributes["Q_LEN"] > 1:
        return 1
    if attributes["kv_layout"] == "mla":
        candidates = [
            os.path.join(
                os.path.dirname(trace_dir), "flash_attn_split_metadata",
                f"{shape['filename']}.json"),
            os.path.join(
                trace_dir, f"_runner_prof_{shape['filename']}_split.json"),
            os.path.join(
                trace_dir, f"_runner_{shape['filename']}_split.json"),
        ]
        metadata_path = next(
            (path for path in candidates if os.path.isfile(path)), None)
        if metadata_path is None:
            raise FileNotFoundError(
                f"MetaX MLA split metadata is missing for "
                f"{shape['filename']}")
        with open(metadata_path, encoding="utf-8") as stream:
            metadata = json.load(stream)
        values = metadata["executed_splits"]
        if (not isinstance(values, list) or not values
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in values)):
            raise ValueError(
                f"invalid MetaX MLA executed_splits: {values!r}")
        distinct = set(values)
        if len(distinct) != 1:
            raise ValueError(
                "MetaX MLA produced per-request split counts that cannot be "
                f"represented by scalar SP: {values!r}")
        return next(iter(distinct))

    requested = attributes["SP_REQUESTED"]
    if requested is None:
        raise ValueError(
            "non-MLA MetaX FlashAttention requires SP_REQUESTED")
    # MetaX expresses batch, KV-head, and split work over different grid axes
    # for different implementations. SP therefore cannot be reconstructed
    # from the launch grid. The non-MLA serving API receives this scalar
    # directly, so it is the executed split contract.
    return requested


def stage_flash_attn_metadata(cfg, dry_run=False, start_idx=0, end_idx=0):
    """Collect FlashMLA's executed split metadata without hardware counters."""

    kernel = "flash_attn"
    kernel_cfg = _load_kernel_config(
        get_kernel_config_path(cfg, kernel), kernel)
    shapes = [
        shape for shape in build_flash_attn_shapes(kernel_cfg)
        if shape["attr"]["kv_layout"] == "mla"
    ]
    if start_idx > 0:
        end = end_idx if end_idx > 0 else len(shapes)
        shapes = shapes[start_idx - 1:end]
    elif end_idx > 0:
        raise ValueError("end_idx requires a positive start_idx")
    output_dir = os.path.join(
        get_profiler_output_dir(cfg), "flash_attn_split_metadata")
    runner = os.path.abspath(get_kernel_script_path(cfg, kernel))
    python_bin = current_python_executable()
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
    for index, shape in enumerate(shapes, 1):
        metadata_path = os.path.join(
            output_dir, f"{shape['filename']}.json")
        command = [
            python_bin, "-u", runner, *shape["script_args"],
            "--split-metadata-path", metadata_path,
            "--metadata-only",
        ]
        if dry_run:
            print(f"[{index}/{len(shapes)}] DRY-RUN: {shlex.join(command)}")
            continue
        if os.path.isfile(metadata_path):
            print(f"[{index}/{len(shapes)}] SKIP: {shape['filename']}")
            continue
        print(f"[{index}/{len(shapes)}] metadata: {shape['filename']}")
        subprocess.run(command, check=True)


# ─── mcProfiler parsing ───────────────────────────────────────

def _to_num(v):
    """Convert one mcProfiler numeric field and reject malformed data.

    Handles mcProfiler text formats: "90.19%", "54,966,701,952.0byte",
    "16,720,290.72(Kcycles)", "1,925.44", plain numbers.
    """
    if v is None or (isinstance(v, str) and not v.strip()):
        raise ValueError("mcProfiler numeric field is empty")
    if isinstance(v, bool):
        raise TypeError("mcProfiler numeric field cannot be boolean")
    if isinstance(v, (int, float)):
        result = float(v)
    elif isinstance(v, str):
        s = v.strip()
        s = s.replace(",", "")
        for suffix in ("%", "byte", "bytes"):
            if s.endswith(suffix):
                s = s[:-len(suffix)]
        s = s.split("(")[0].strip()
        try:
            result = float(s)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"invalid mcProfiler numeric value: {v!r}") from error
    else:
        raise TypeError(
            f"invalid mcProfiler numeric type: {type(v).__name__}")
    if not math.isfinite(result):
        raise ValueError(f"mcProfiler numeric field must be finite: {v!r}")
    return result


_OPTIONAL_UNAVAILABLE_MCPROFILER_SCALARS = {
    "Constant SL1 Hit Rate": "can not find export value @sl1_hit_rate",
    "Memory Data Flow Chart": (
        "cannot draw from data: can not find export value @sl1_hit_rate"),
}


def _parse_mcprofiler_scalar(name, value):
    """Parse one scalar while preserving one explicit tool limitation.

    MetaX mcProfiler can emit an unavailable sentinel for the optional
    constant-SL1 diagnostic.  Preserve that exact observation as ``None`` in
    ``MetaX.all_scalars``.  Every other scalar remains subject to strict
    numeric parsing, including required metrics and other error strings.
    """

    if (name in _OPTIONAL_UNAVAILABLE_MCPROFILER_SCALARS
            and value == _OPTIONAL_UNAVAILABLE_MCPROFILER_SCALARS[name]):
        return None
    return _to_num(value)


def _required_metric(metrics, name, alias=None):
    if name in metrics:
        return metrics[name]
    if alias is not None and alias in metrics:
        return metrics[alias]
    if alias is None:
        raise KeyError(f"mcProfiler report is missing metric {name!r}")
    raise KeyError(
        f"mcProfiler report is missing metrics {name!r} and {alias!r}")


def _percentage(value, label):
    number = _to_num(value)
    if not 0.0 <= number <= 100.0:
        raise ValueError(
            f"mcProfiler {label} must be within [0, 100], got {number}")
    return number




_CXXFILT_PATH = None
_CANONICAL_KERNEL_NAMES = {}


def _perkernel_file_stem(path):
    """Return the kernel-name part of ``<index>_<name>.txt.json``."""

    filename = os.path.basename(path)
    match = re.fullmatch(r"\d+_(.+)\.txt\.json", filename)
    if match is None:
        raise ValueError(
            f"invalid mcProfiler per-kernel filename: {filename!r}")
    return match.group(1)


def _perkernel_file_ordinal(path):
    """Return the global launch ordinal from a layer-1 filename."""

    filename = os.path.basename(path)
    match = re.fullmatch(r"(\d+)_.+\.txt\.json", filename)
    if match is None:
        raise ValueError(
            f"invalid mcProfiler per-kernel filename: {filename!r}")
    return int(match.group(1))


def _is_period_rollup(filename):
    if not filename.endswith(".txt.json"):
        return False
    stem = filename[:-len(".txt.json")]
    stem = re.sub(r"^\d+_", "", stem, count=1)
    return stem.startswith("period")


def _allk_real_files(allk_dir):
    if not os.path.isdir(allk_dir):
        raise FileNotFoundError(
            f"layer-1 mcProfiler directory is missing: {allk_dir}")
    result = []
    for filename in os.listdir(allk_dir):
        if filename == "report.txt.json" or not filename.endswith(".txt.json"):
            continue
        if _is_period_rollup(filename):
            continue
        path = os.path.join(allk_dir, filename)
        _perkernel_file_stem(path)
        result.append(path)
    result.sort(key=lambda path: (_perkernel_file_ordinal(path), path))
    ordinals = [_perkernel_file_ordinal(path) for path in result]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError(
            "layer-1 mcProfiler files contain duplicate global launch "
            f"ordinals under {allk_dir}: {ordinals}")
    return result


def _mangled_source_names(stem):
    """Extract ordered Itanium source identifiers from a truncated symbol."""

    identifiers = []
    index = 0
    while index < len(stem):
        token = stem[index:]
        if stem[index] == "L":
            match = re.match(r"L[a-z]-?[0-9]*E?", token)
            if match is not None:
                index += match.end()
                continue
        if stem[index] == "S":
            match = re.match(r"S[0-9A-Za-z]*_", token)
            if match is not None:
                index += match.end()
                continue
        if stem[index] == "T":
            match = re.match(r"T[0-9]*_", token)
            if match is not None:
                index += match.end()
                continue
        length_match = re.match(r"[0-9]+", token)
        if length_match is not None:
            length = int(length_match.group())
            start = index + length_match.end()
            identifier = stem[start:start + length]
            if len(identifier) < length:
                break
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
                identifiers.append(identifier)
                index = start + length
                continue
        index += 1
    return identifiers


def _is_ordered_subsequence(needle, haystack):
    iterator = iter(haystack)
    return all(any(item == candidate for candidate in iterator)
               for item in needle)


def _truncated_stem_matches_trace(stem, trace_name):
    identifiers = _mangled_source_names(stem)
    if not identifiers:
        return False
    trace_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", trace_name)
    return _is_ordered_subsequence(identifiers, trace_tokens)


def _demangled_prefix_matches(stem_name, trace_name):
    if stem_name == trace_name:
        return True
    stem_open = stem_name.find("(")
    trace_open = trace_name.find("(")
    if stem_open < 0 or trace_open < 0:
        return False
    if stem_name[:stem_open] != trace_name[:trace_open]:
        return False
    stem_arguments = stem_name[stem_open + 1:].rstrip().rstrip(")").rstrip()
    trace_arguments = trace_name[trace_open + 1:].rstrip().rstrip(")").rstrip()
    if not trace_arguments.startswith(stem_arguments):
        return False
    boundary = trace_arguments[len(stem_arguments):len(stem_arguments) + 1]
    return boundary == "" or boundary in (",", ")", ">", " ", "<")


def _canonical_kernel_name(name):
    global _CXXFILT_PATH

    if name in _CANONICAL_KERNEL_NAMES:
        return _CANONICAL_KERNEL_NAMES[name]
    canonical = name
    if name.startswith("_Z"):
        if _CXXFILT_PATH is None:
            _CXXFILT_PATH = shutil.which("c++filt")
            if _CXXFILT_PATH is None:
                raise FileNotFoundError(
                    "c++filt is required to attribute MetaX layer-1 files")
        try:
            result = subprocess.run(
                [_CXXFILT_PATH, name], capture_output=True, text=True,
                timeout=10, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"c++filt failed for MetaX kernel {name!r}") from error
        canonical = result.stdout.strip()
        if not canonical:
            raise ValueError(f"c++filt returned an empty name for {name!r}")
    canonical = re.sub(r"^\s*void\s+", "", canonical)
    canonical = re.sub(r"\s+", " ", canonical).strip()
    _CANONICAL_KERNEL_NAMES[name] = canonical
    return canonical


def _matching_allk_files(allk_dir, trace_name):
    """Resolve exactly one trace kernel to its layer-1 invocation files."""

    target = _canonical_kernel_name(trace_name)
    exact = []
    mangled_prefix = []
    demangled_prefix = []
    for path in _allk_real_files(allk_dir):
        stem = _perkernel_file_stem(path)
        canonical_stem = _canonical_kernel_name(stem)
        if canonical_stem == target:
            exact.append(path)
        elif canonical_stem.startswith("_Z"):
            if _truncated_stem_matches_trace(stem, trace_name):
                mangled_prefix.append(path)
        elif _demangled_prefix_matches(canonical_stem, target):
            demangled_prefix.append(path)
    if exact:
        return tuple(exact)
    for label, candidates in (
            ("truncated mangled", mangled_prefix),
            ("truncated demangled", demangled_prefix)):
        if not candidates:
            continue
        distinct = {
            _perkernel_file_stem(path) for path in candidates
        }
        if len(distinct) != 1:
            raise ValueError(
                f"ambiguous {label} MetaX kernel match for {trace_name!r}: "
                f"{sorted(distinct)}")
        return tuple(candidates)
    return ()


def _average_perkernel_files(invocation_files, source):
    """Parse exactly one isolated layer-1 invocation report.

    The historical implementation averaged all files sharing a kernel name.
    That loses legitimate repeated same-name launches and can mix different
    positions in one op.  Keep the existing helper name for its private callers,
    but reject every multi-file input so each trace ordinal retains its own
    counter observation.
    """

    if source != "per_invocation":
        raise ValueError(f"unsupported MetaX profiler source: {source!r}")
    if not invocation_files:
        raise ValueError("per-kernel invocation file list must not be empty")
    if len(invocation_files) != 1:
        raise ValueError(
            "each MetaX layer-1 observation requires exactly one invocation "
            "file; averaging files by kernel name is forbidden")
    scalar_sums = defaultdict(float)
    roofline_sums = defaultdict(float)
    expected_scalars = None
    expected_roofline = None
    charts = None
    roofline_crash = None
    for path in invocation_files:
        with open(path, encoding="utf-8") as stream:
            report = json.load(stream)
        require_mapping(report, f"mcProfiler report {path}")
        scalars = {}
        report_charts = {}
        for section, entries in report.items():
            if not isinstance(entries, list):
                continue
            for entry_index, entry in enumerate(entries):
                context = f"mcProfiler {section}[{entry_index}]"
                require_mapping(entry, context)
                name = require_non_empty_string(entry, "name", context)
                value = entry["value"]
                if name == "RoofLine":
                    continue
                if isinstance(value, dict):
                    if "data" in value:
                        chart = value["data"]
                    else:
                        chart = value
                    if not isinstance(chart, dict):
                        raise TypeError(f"{context}.value.data must be a mapping")
                    report_charts[name] = chart
                else:
                    scalars[name] = _parse_mcprofiler_scalar(name, value)
        if not scalars:
            raise ValueError(f"mcProfiler report contains no scalars: {path}")
        if expected_scalars is None:
            expected_scalars = set(scalars)
        elif set(scalars) != expected_scalars:
            raise ValueError(
                f"mcProfiler invocation scalar fields disagree: {path}")
        if charts is None:
            charts = report_charts

        internal = report["Internal"]
        if not isinstance(internal, list):
            raise TypeError(f"mcProfiler Internal must be a list: {path}")
        roofline_entries = []
        for entry_index, entry in enumerate(internal):
            context = f"mcProfiler Internal[{entry_index}]"
            require_mapping(entry, context)
            name = require_non_empty_string(entry, "name", context)
            if name == "RoofLine":
                roofline_entries.append(entry)
        if len(roofline_entries) != 1:
            raise ValueError(
                f"mcProfiler report requires one RoofLine entry: {path}")
        roofline_value = roofline_entries[0]["value"]
        current_crash = isinstance(roofline_value, str)
        if current_crash and not (
                "cannot draw from data" in roofline_value.lower()
                and "division by zero" in roofline_value.lower()):
            raise ValueError(
                "unrecognized mcProfiler RoofLine failure: "
                f"{roofline_value!r}")
        if roofline_crash is None:
            roofline_crash = current_crash
        elif roofline_crash != current_crash:
            raise ValueError(
                "mcProfiler invocations disagree on RoofLine availability")
        roofline = {}
        if not current_crash:
            require_mapping(roofline_value, "mcProfiler RoofLine value")
            processed = require_mapping(
                roofline_value["processed_data"],
                "mcProfiler RoofLine processed_data")
            data = require_mapping(
                roofline_value["data"], "mcProfiler RoofLine data")
            for name, value in processed.items():
                roofline[name] = _to_num(value)
            for name, value in data.items():
                if isinstance(value, (int, float, str)):
                    roofline[name] = _to_num(value)
        if expected_roofline is None:
            expected_roofline = set(roofline)
        elif set(roofline) != expected_roofline:
            raise ValueError(
                f"mcProfiler invocation RoofLine fields disagree: {path}")
        for name, value in scalars.items():
            if value is None:
                scalar_sums[name] = None
            else:
                scalar_sums[name] += value
        for name, value in roofline.items():
            roofline_sums[name] += value

    count = len(invocation_files)
    averages = {
        name: None if value is None else value / count
        for name, value in scalar_sums.items()
    }
    roofline_average = {
        name: value / count for name, value in roofline_sums.items()
    }
    if roofline_crash:
        during = 0.0
        case_flops = 0.0
        case_bandwidth = 0.0
        max_bandwidth = 0.0
        case_intensity = 0.0
    else:
        during = _required_metric(roofline_average, "during")
        case_flops = _required_metric(roofline_average, "case_flops")
        case_bandwidth = _required_metric(
            roofline_average, "case_bandwith")
        max_bandwidth = _required_metric(
            roofline_average, "MAX_Bandwith")
        case_intensity = _required_metric(roofline_average, "case_I")
    return {
        "total_cycles": _required_metric(averages, "Total Cycles"),
        "ap_active_cycles": _required_metric(averages, "AP active cycles"),
        "during": during,
        "ap_mma_duty_pct": _required_metric(
            averages, "AP MMA Duty ratio"),
        "ap_mte_duty_pct": _required_metric(
            averages, "AP MTE Duty ratio"),
        "ap_ste_duty_pct": _required_metric(
            averages, "AP STE Duty ratio"),
        "vls_duty_pct": _required_metric(averages, "VLS Duty ratio"),
        "l2c_duty_pct": _required_metric(averages, "L2C Duty ratio"),
        "vl1_hit_rate_pct": _required_metric(averages, "VL1 Hit Rate"),
        "l2c_hit_rate_pct": _required_metric(averages, "L2C Hit Rate"),
        "global_memory_read_bytes": _required_metric(
            averages, "Global Memory Read bytes"),
        "global_memory_write_bytes": _required_metric(
            averages, "Global Memory Write bytes"),
        "achieved_waves": _required_metric(
            averages, "Achieved waves", "WAVES"),
        "dispatched_waves": _required_metric(
            averages, "Dispatched waves", "WAVES"),
        "shared_memory_efficiency_pct": _required_metric(
            averages, "shared memory access efficiency"),
        "avg_conflict_cycles_per_inst": _required_metric(
            averages, "average conflict cycles per instruction"),
        "real_ipc": _required_metric(averages, "instruction per cycle"),
        "roofline_case_flops": case_flops,
        "roofline_case_bandwidth": case_bandwidth,
        "roofline_max_bandwidth": max_bandwidth,
        "roofline_case_I": case_intensity,
        "all_scalars": averages,
        "charts": {} if charts is None else charts,
        "roofline": roofline_average,
        "roofline_crash": bool(roofline_crash),
        "n_invocations": count,
        "source": source,
    }


def _build_moe_tiling(tiling, attr, ncu_kernels):
    """Build the moe_tiling dict for a FusedMoE case.

    The standalone runner applies one shared BM/BN/BK/GSM configuration to w1
    and w2, but deliberately gives the projections distinct runtime names so
    layer-1 counters never blend them. Re-derive both tilings from that shared
    configuration and require each recorded grid to match its projection.
    """
    if "w1" not in tiling:
        raise ValueError(f"tiling.json missing w1 config: {tiling}")

    w1 = require_mapping(tiling["w1"], "fused_moe tiling.w1")
    BM = require_positive_int(w1, "BLOCK_SIZE_M", "fused_moe tiling.w1")
    BN = require_positive_int(w1, "BLOCK_SIZE_N", "fused_moe tiling.w1")
    BK = require_positive_int(w1, "BLOCK_SIZE_K", "fused_moe tiling.w1")
    GSM = require_positive_int(w1, "GROUP_SIZE_M", "fused_moe tiling.w1")

    moe_tiling = derive_fused_moe_tiling(
        tokens=require_positive_int(attr, "TOK", "FUSEDMOE Attr"),
        experts=require_positive_int(attr, "E", "FUSEDMOE Attr"),
        topk=require_positive_int(attr, "TOPK", "FUSEDMOE Attr"),
        hidden=require_positive_int(attr, "H", "FUSEDMOE Attr"),
        intermediate=require_positive_int(attr, "INT", "FUSEDMOE Attr"),
        tile_m=BM, tile_n=BN, tile_k=BK, group_size_m=GSM)

    expected = (
        ("fused_moe_kernel", "w1"),
        ("fused_moe_kernel_w2", "w2"),
    )
    for kernel_name, projection in expected:
        observed = {
            _parse_grid_dimensions(record["Metadata"]["Grid Size"])[0]
            for record in ncu_kernels
            if record["Metadata"]["Kernel Name"] == kernel_name
        }
        if len(observed) != 1:
            raise ValueError(
                f"FusedMoE {projection} requires exactly one observed grid for "
                f"{kernel_name!r}, got {sorted(observed)}")
        observed_grid = next(iter(observed))
        predicted_grid = moe_tiling[projection]["grid"]
        if observed_grid != predicted_grid:
            raise ValueError(
                f"FusedMoE {projection} grid mismatch: "
                f"predicted={predicted_grid}, observed={observed_grid}; "
                f"TOK={attr['TOK']} E={attr['E']} H={attr['H']} "
                f"INT={attr['INT']}")
        moe_tiling[projection]["real_grid"] = observed_grid

    return moe_tiling


def _allowed_profiled_workgroups(shape, kernel_event):
    """Return independently derived valid counter-side work counts."""

    require_mapping(shape, "kernel shape")
    require_mapping(kernel_event, "mcTracer kernel")
    name = require_non_empty_string(
        kernel_event, "name", "mcTracer kernel")
    grid = tuple(
        require_positive_int(
            kernel_event, axis, f"mcTracer kernel {name}")
        for axis in ("grid_x", "grid_y", "grid_z")
    )
    physical_workgroups = math.prod(grid)
    shape_type = require_non_empty_string(shape, "type", "kernel shape")
    if shape_type == "GEMM":
        core_name = name[len("void "):] if name.startswith("void ") else name
        if not core_name.startswith("mcblas__"):
            return frozenset((physical_workgroups,))
        parsed_tile = parse_complete_gemm_tile(name)
        if parsed_tile is None:
            raise ValueError(
                "C500 mcBLAS GEMM kernel has no parseable logical tile: "
                f"{name!r}")
        tile_m, tile_n, _, _ = parsed_tile
        attributes = require_mapping(shape["attr"], "GEMM Attr")
        logical_workgroups = gemm_logical_workgroups(
            batch=require_positive_int(attributes, "B", "GEMM Attr"),
            m=require_positive_int(attributes, "M", "GEMM Attr"),
            n=require_positive_int(attributes, "N", "GEMM Attr"),
            tile_m=tile_m,
            tile_n=tile_n,
            split_k=grid[2],
        )
        # mcProfiler reports physical launched workers for direct/padded
        # mcBLAS grids, but useful logical tiles for persistent/swizzled grids
        # whose excess workers exit without work.  Both counts are derived
        # independently from the trace and workload; no counter value selects
        # an empirical fallback.
        return frozenset((physical_workgroups, logical_workgroups))
    if shape_type in {
            "FUSEDMOE", "NORM", "ROTARY", "ACTIVATION", "ELEMENTWISE",
            "KV_CACHE"}:
        return frozenset((physical_workgroups,))
    if shape_type != "FLASHATTN":
        raise ValueError(f"unsupported C500 kernel shape type: {shape_type!r}")

    attributes = require_mapping(shape["attr"], "FLASHATTN Attr")
    layout = require_non_empty_string(
        attributes, "kv_layout", "FLASHATTN Attr")
    if layout in {"contiguous", "paged", "paged_varlen", "mla"}:
        return frozenset((physical_workgroups,))
    if layout != "chunked_prefill":
        raise ValueError(f"unsupported FLASHATTN kv_layout: {layout!r}")
    query_length = require_positive_int(
        attributes, "Q_LEN", "FLASHATTN Attr")
    requested_splits = require_positive_int(
        attributes, "SP_REQUESTED", "FLASHATTN Attr")
    if query_length <= 1:
        return frozenset((physical_workgroups,))
    core_name = name[len("void "):] if name.startswith("void ") else name
    if not core_name.startswith("flash_fwd_splitkv_kernel<"):
        return frozenset((physical_workgroups,))
    if parse_flash_attention_tile(name) != (64, 64):
        return frozenset((physical_workgroups,))
    block_size = require_positive_int(
        kernel_event, "block_size", f"mcTracer kernel {name}")
    if block_size != 256:
        return frozenset((physical_workgroups,))

    return frozenset((flash_attention_packed_varlen_scheduled_workgroups(
        batch=require_positive_int(attributes, "B", "FLASHATTN Attr"),
        q_len=query_length,
        tile_q=64,
        query_heads=require_positive_int(
            attributes, "HQ", "FLASHATTN Attr"),
        query_heads_per_workgroup=1,
        splits=requested_splits,
    ),))


def _kernel_to_ncu_format(
        k, sm_freq_hz, dram_freq_hz, max_regs_per_unit, profiler_data,
        allowed_profiled_workgroups):
    """Convert one trace launch plus its own isolated layer-1 counters."""

    require_mapping(k, "mcTracer kernel")
    require_mapping(profiler_data, "mcProfiler per-kernel observation")
    name = require_non_empty_string(k, "name", "mcTracer kernel")
    grid = tuple(
        require_positive_int(k, axis, f"mcTracer kernel {name}")
        for axis in ("grid_x", "grid_y", "grid_z")
    )
    profiler_scalars = require_mapping(
        profiler_data["all_scalars"], "mcProfiler all_scalars")
    profiled_workgroups = _to_num(_required_metric(
        profiler_scalars, "WORKGROUPS"))
    if not isinstance(allowed_profiled_workgroups, frozenset):
        raise TypeError(
            "allowed_profiled_workgroups must be a frozenset")
    if not allowed_profiled_workgroups:
        raise ValueError("allowed_profiled_workgroups must not be empty")
    for allowed_workgroups in allowed_profiled_workgroups:
        require_positive_int(
            {"allowed_workgroups": allowed_workgroups},
            "allowed_workgroups", "C500 launch contract")
    if not any(
            abs(profiled_workgroups - allowed_workgroups) <= 0.1
            for allowed_workgroups in allowed_profiled_workgroups):
        raise ValueError(
            "mcTracer launch and mcProfiler counters describe different "
            f"workgroup counts for kernel {name!r}: "
            f"allowed={sorted(allowed_profiled_workgroups)}, "
            f"physical_grid={math.prod(grid)}, "
            f"mcProfiler={profiled_workgroups}")
    block_size = require_positive_int(
        k, "block_size", f"mcTracer kernel {name}")
    registers_per_thread = require_positive_int(
        k, "regs_per_thread", f"mcTracer kernel {name}")
    dynamic_smem = require_non_negative_int(
        k, "dynamic_smem", f"mcTracer kernel {name}")
    static_smem = require_non_negative_int(
        k, "static_smem", f"mcTracer kernel {name}")
    if (isinstance(max_regs_per_unit, bool)
            or not isinstance(max_regs_per_unit, int)
            or max_regs_per_unit <= 0):
        raise TypeError("max_regs_per_unit must be a positive integer")
    register_footprint = registers_per_thread * block_size
    register_block_limit = max_regs_per_unit // register_footprint
    if register_block_limit <= 0:
        raise ValueError(f"C500 kernel {name!r} exceeds register capacity")

    profiler_duration = _to_num(profiler_data["during"])
    if profiler_duration < 0:
        raise ValueError("mcProfiler duration cannot be negative")
    roofline_crash = profiler_data["roofline_crash"]
    if not isinstance(roofline_crash, bool):
        raise TypeError("mcProfiler roofline_crash must be boolean")
    if profiler_duration <= 0 and not roofline_crash:
        raise ValueError(
            f"mcProfiler duration is missing for C500 kernel {name!r}")

    # mcProfiler layer-1 obtains counters by replaying/pausing the target
    # kernel.  Its ``during`` therefore measures the instrumented replay and
    # can be tens to thousands of times larger than the uninstrumented launch.
    # mcTracer observes the normal runner execution and is the only canonical
    # wall-time source.  Preserve ``during`` below as profiler provenance, but
    # never use it as the evaluation target or to derive elapsed cycles.
    duration_ns = require_positive_number(
        k, "dur_ns", f"mcTracer kernel {name}")
    duration_source = "mcTracer"

    elapsed_cycles = duration_ns * sm_freq_hz / 1e9
    active_cycles = _to_num(profiler_data["ap_active_cycles"])
    if active_cycles < 0:
        raise ValueError("mcProfiler AP active cycles cannot be negative")
    dram_read = _to_num(profiler_data["global_memory_read_bytes"])
    dram_write = _to_num(profiler_data["global_memory_write_bytes"])
    if dram_read < 0 or dram_write < 0:
        raise ValueError("mcProfiler DRAM byte counters cannot be negative")
    roofline = require_mapping(
        profiler_data["roofline"], "mcProfiler RoofLine")
    if roofline_crash:
        l2_read = 0.0
    else:
        l2_read = _to_num(_required_metric(roofline, "l2c_memacs"))
        if l2_read < 0:
            raise ValueError("mcProfiler L2 read bytes cannot be negative")
    l2_write = 0.0
    duty_percentages = {
        name: _percentage(profiler_data[field], name)
        for field, name in (
            ("ap_mma_duty_pct", "AP MMA Duty ratio"),
            ("ap_mte_duty_pct", "AP MTE Duty ratio"),
            ("ap_ste_duty_pct", "AP STE Duty ratio"),
            ("vls_duty_pct", "VLS Duty ratio"),
            ("l2c_duty_pct", "L2C Duty ratio"),
        )
    }
    l1_hit_rate = _percentage(
        profiler_data["vl1_hit_rate_pct"], "VL1 Hit Rate")
    l2_hit_rate = _percentage(
        profiler_data["l2c_hit_rate_pct"], "L2C Hit Rate")
    _percentage(
        profiler_data["shared_memory_efficiency_pct"],
        "shared memory access efficiency")
    achieved_waves = _to_num(profiler_data["achieved_waves"])
    dispatched_waves = _to_num(profiler_data["dispatched_waves"])
    if achieved_waves < 0 or dispatched_waves < 0:
        raise ValueError("mcProfiler wave counts cannot be negative")
    achieved_occupancy = (
        achieved_waves / dispatched_waves * 100.0
        if achieved_waves > 0 and dispatched_waves > 0 else 0.0
    )
    max_bandwidth = _to_num(profiler_data["roofline_max_bandwidth"])
    case_bandwidth = _to_num(profiler_data["roofline_case_bandwidth"])
    if max_bandwidth < 0 or case_bandwidth < 0:
        raise ValueError("mcProfiler RoofLine bandwidth cannot be negative")
    total_dram = dram_read + dram_write
    if max_bandwidth > 0 and total_dram > 0:
        usage = case_bandwidth / max_bandwidth
        read_fraction = dram_read / total_dram
        dram_read_bw_pct = usage * read_fraction * 100.0
        dram_write_bw_pct = usage * (1.0 - read_fraction) * 100.0
    else:
        dram_read_bw_pct = 0.0
        dram_write_bw_pct = 0.0

    source = profiler_data["source"]
    if source != "per_invocation":
        raise ValueError(f"invalid MetaX isolation source: {source!r}")
    invocation_count = require_positive_int(
        profiler_data, "n_invocations", "mcProfiler observation")
    metax_payload = {
        "all_scalars": profiler_scalars,
        "charts": require_mapping(
            profiler_data["charts"], "mcProfiler charts"),
        "roofline": roofline,
        "during": profiler_duration,
        "n_invocations": invocation_count,
        "source": source,
        "dur_source": duration_source,
    }
    if roofline_crash:
        metax_payload["tool_limitation"] = (
            "mcProfiler roofline crashed on sub-us kernel; duration from "
            "mcTracer, L2/memory counters may be zero or inaccurate"
        )

    return {
        "Metadata": {
            "Kernel Name": name,
            "Grid Size": ",".join(str(value) for value in grid),
            "Block Size": f"{block_size},1,1",
            "Device": "MetaX C500",
            "CC": "8.0",
            "pk_source": source,
            "pk_n_invocations": invocation_count,
        },
        "Sections": {
            "GPU Speed Of Light Throughput": {
                "Header": {
                    "Duration [nsecond]": duration_ns,
                    "SM Frequency [cycle/second]": sm_freq_hz,
                    "DRAM Frequency [cycle/second]": dram_freq_hz,
                    "Elapsed Cycles [cycle]": elapsed_cycles,
                    "SM Active Cycles [cycle]": active_cycles,
                }
            },
            "Compute Workload Analysis": {
                "Pipe Utilization (% of active cycles)": {
                    "Tensor (All) [%]": duty_percentages[
                        "AP MMA Duty ratio"],
                    "FMA [%]": duty_percentages["AP MTE Duty ratio"],
                    "ALU [%]": duty_percentages["AP STE Duty ratio"],
                },
                "Pipe Utilization (% of peak instructions executed)": {
                    "XU [%]": 0.0,
                },
            },
            "Memory Workload Analysis Chart": {
                "Memory Chart": {
                    "l1tex__m_xbar2l1tex_read_bytes.sum [byte]": l2_read,
                    "l1tex__m_l1tex2xbar_write_bytes.sum [byte]": l2_write,
                    "dram__bytes_read.sum [byte]": dram_read,
                    "dram__bytes_write.sum [byte]": dram_write,
                    "l1tex__t_sector_hit_rate.pct [%]": l1_hit_rate,
                    "lts__t_sector_hit_rate.pct [%]": l2_hit_rate,
                    "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum [byte]": 0.0,
                },
            },
            "Launch Statistics": {
                "Header": {
                    "Block Size": block_size,
                    "Registers Per Thread [register/thread]": registers_per_thread,
                    "Dynamic Shared Memory Per Block [byte/block]": dynamic_smem,
                    "Static Shared Memory Per Block [byte/block]": static_smem,
                    "Driver Shared Memory Per Block [byte/block]": 0,
                    "Achieved Occupancy [%]": achieved_occupancy,
                    "Theoretical Occupancy [%]": 0.0,
                },
            },
            "Occupancy": {
                "Header": {
                    "Block Limit Registers [block]": register_block_limit,
                },
            },
            "Memory Workload Analysis Tables": {
                "Device Memory": {
                    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed [%]": dram_read_bw_pct,
                    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed [%]": dram_write_bw_pct,
                },
            },
        },
        "MetaX": metax_payload,
    }


# ─── Stage implementations ───────────────────────────────────

def _profiler_group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_profiler_group(process, expected_pid, expected_pgid):
    """Terminate and prove quiescence of exactly one owned process group."""

    if process.pid != expected_pid or expected_pgid != expected_pid:
        raise RuntimeError("refusing inexact mcProfiler process-group cleanup")
    if _profiler_group_exists(expected_pgid):
        try:
            observed_pgid = os.getpgid(expected_pid)
        except ProcessLookupError:
            observed_pgid = expected_pgid
        if observed_pgid != expected_pgid:
            raise RuntimeError("mcProfiler process group identity changed")
        try:
            os.killpg(expected_pgid, signal.SIGTERM)
        except ProcessLookupError as error:
            if _profiler_group_exists(expected_pgid):
                raise RuntimeError(
                    "mcProfiler group disappeared before cleanup") from error
    deadline = time.monotonic() + 5.0
    while _profiler_group_exists(expected_pgid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _profiler_group_exists(expected_pgid):
        try:
            os.killpg(expected_pgid, signal.SIGKILL)
        except ProcessLookupError as error:
            if _profiler_group_exists(expected_pgid):
                raise RuntimeError(
                    "mcProfiler group disappeared before SIGKILL") from error
    process.wait(timeout=30)
    deadline = time.monotonic() + 5.0
    while _profiler_group_exists(expected_pgid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _profiler_group_exists(expected_pgid):
        raise RuntimeError("mcProfiler process group is not quiescent")


def _mcprofiler_lock_path():
    if "MIDLAYER_C500_MCPROFILER_LOCK" not in os.environ:
        raise ValueError(
            "MIDLAYER_C500_MCPROFILER_LOCK is required for mcProfiler execution"
        )
    return os.environ["MIDLAYER_C500_MCPROFILER_LOCK"]


@contextmanager
def _host_global_mcprofiler_lock():
    lock_path = _mcprofiler_lock_path()
    lock_parent = os.path.dirname(lock_path)
    os.makedirs(lock_parent, exist_ok=True)
    with open(lock_path, "a+b") as lock_stream:
        try:
            fcntl.flock(
                lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("host-global mcProfiler lock is held") from error
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _require_idle_c500(device_id, log_path, env, cwd):
    query_specs = (
        ("usage", "--show-usage", _MX_SMI_USAGE_PATTERN),
        ("hbm", "--show-hbm-bandwidth", _MX_SMI_HBM_PATTERN),
        ("process", "--show-process", _MX_SMI_NO_PROCESS_PATTERN),
    )
    for name, option, pattern in query_specs:
        result = subprocess.run(
            [_MX_SMI_PATH, option, "-i", str(device_id)],
            capture_output=True, text=True, timeout=10, check=False,
            env=env, cwd=cwd)
        raw_path = f"{log_path}.preflight_{name}.stdout.txt"
        error_path = f"{log_path}.preflight_{name}.stderr.txt"
        with open(raw_path, "x", encoding="utf-8") as stream:
            stream.write(result.stdout)
        with open(error_path, "x", encoding="utf-8") as stream:
            stream.write(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"mx-smi {name} preflight failed")
        if name == "usage":
            devices = list(_MX_SMI_DEVICE_PATTERN.finditer(result.stdout))
            values = pattern.findall(result.stdout)
            if (len(devices) != 1
                    or int(devices[0].group("device")) != device_id
                    or len(values) != 1
                    or float(values[0]) != 0.0):
                raise RuntimeError("C500 usage preflight is not idle")
        elif name == "hbm":
            devices = list(_MX_SMI_DEVICE_PATTERN.finditer(result.stdout))
            values = pattern.findall(result.stdout)
            if (len(devices) != 1
                    or int(devices[0].group("device")) != device_id
                    or len(values) != 1
                    or not 0.0 <= float(values[0])
                    <= _MX_SMI_IDLE_HBM_MAX_MBYTES_PER_S):
                raise RuntimeError("C500 hbm preflight is not idle")
        elif name == "process":
            if len(pattern.findall(result.stdout)) != 1:
                raise RuntimeError("C500 process preflight is not idle")
        else:
            raise ValueError(f"unknown C500 preflight query: {name!r}")


def _run_profiler(command, log_path, timeout, env, cwd, device_id):
    """Run mcProfiler in an isolated process group and reap its server."""

    with _host_global_mcprofiler_lock():
        _require_idle_c500(device_id, log_path, env, cwd)
        process = None
        with open(log_path, "x", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    env=env, cwd=cwd, start_new_session=True)
                pid = process.pid
                pgid = pid
                if os.getpgid(pid) != pgid:
                    process.kill()
                    process.wait(timeout=30)
                    raise RuntimeError(
                        "mcProfiler did not enter its own process group")
                return_code = process.wait(timeout=timeout)
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(
                    f"mcProfiler timed out after {timeout}s; log={log_path}"
                ) from error
            finally:
                if process is not None:
                    _terminate_profiler_group(
                        process, process.pid, process.pid)


_PROFILER_WAIT_SECONDS = {
    # mcBLAS initialization can take roughly 30 seconds before the device RPC
    # port remains discoverable.  A two-second wait produced a real kernel
    # execution but no mcProfiler report (``cannot connect to server``).
    "gemm": 30,
    "flash_attn": 15,
    "fused_moe": 5,
    "vector": 5,
}
_LAYER1_MAX_ATTEMPTS = 3


def _trace_launch_signature(kernel_event):
    return tuple(
        kernel_event[name] for name in (
            "grid_x", "grid_y", "grid_z", "block_size",
            "regs_per_thread", "dynamic_smem", "static_smem",
        )
    )


def _union_trace_kernel_events(trace_files):
    """Merge repeated observations of one deterministic mcTracer launch.

    Every non-empty ``tracer_out`` in a shape directory is evidence for the
    same runner invocation contract.  It may repeat the observation (for
    example after a resumable capture), but it may not contribute a different
    kernel sequence.  Taking a name-wise union of incompatible executions
    fabricates a multi-kernel observation and can bind one layer-1 counter
    payload to mutually exclusive launch variants.
    """

    if not trace_files:
        raise ValueError("mcTracer shard list must not be empty")
    parsed_shards = []
    expected_sequence = None
    expected_path = None
    for trace_path in sorted(trace_files):
        kernel_events = _parse_trace_json(trace_path)
        sequence = tuple(
            (kernel_event["name"], _trace_launch_signature(kernel_event))
            for kernel_event in kernel_events
        )
        if expected_sequence is None:
            expected_sequence = sequence
            expected_path = trace_path
        elif sequence != expected_sequence:
            raise ValueError(
                "mcTracer shards disagree on the complete kernel launch "
                f"sequence: {expected_path!r} != {trace_path!r}")
        parsed_shards.append(kernel_events)

    result = []
    if expected_sequence is None:
        raise AssertionError("mcTracer shard sequence was not initialized")
    for ordinal in range(len(expected_sequence)):
        observations = [
            kernel_events[ordinal] for kernel_events in parsed_shards
        ]
        representative = dict(observations[0])
        representative["dur_ns"] = statistics.median(
            observation["dur_ns"] for observation in observations)
        result.append(representative)
    if not result:
        raise ValueError("mcTracer shards contain no kernel launches")
    return result


def _whole_shape_report_exists(profiler_out):
    return any(os.path.isfile(os.path.join(profiler_out, filename)) for filename in (
        "report_dumped_result.json", "report.txt.json",
        "mcprofiler_report_dumped.json", "mcprofiler_report.txt.json",
    ))


def _layer1_capture_state(allk_dir):
    """Return the resumable layer-1 state without accepting rollups."""

    if not os.path.isdir(allk_dir):
        return "missing", 0
    real_files = _allk_real_files(allk_dir)
    if real_files:
        return "per_invocation", len(real_files)
    report_path = os.path.join(allk_dir, "report.txt.json")
    if os.path.isfile(report_path):
        return "report_only", 0
    return "missing", 0


def _captured_subprocess_tail(value):
    """Normalize one bounded diagnostic stream from a failed tool process."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif not isinstance(value, str):
        value = str(value)
    return value[-2000:]


def stage_profile(cfg, kernel, dry_run=False, max_tasks=0, skip_profiler=False,
                  start_idx=0, end_idx=0, port=50123):
    """Capture mcTracer launch metadata and layer-1 mcProfiler counters."""
    kernel_config_path = get_kernel_config_path(cfg, kernel)
    kernel_cfg = _load_kernel_config(kernel_config_path, kernel)

    shapes = build_shapes(kernel, kernel_cfg)
    if max_tasks < 0:
        raise ValueError("max_tasks cannot be negative")
    if max_tasks > 0:
        shapes = shapes[:max_tasks]
    if start_idx > 0:
        end = end_idx if end_idx > 0 else len(shapes)
        if end < start_idx:
            raise ValueError("end_idx cannot be smaller than start_idx")
        shapes = shapes[start_idx - 1:end]
    elif end_idx > 0:
        raise ValueError("end_idx requires a positive start_idx")

    if port <= 0 or port > 65535:
        raise ValueError(f"mcProfiler port is invalid: {port}")

    trace_dir = get_kernel_trace_dir(cfg, kernel)
    if not dry_run:
        os.makedirs(trace_dir, exist_ok=True)

    pipeline_settings = get_kernel_pipeline_settings(cfg)
    python_bin = current_python_executable()
    mc_tracer_bin = require_profiler_executable(
        "mcTracer", "MCTRACER_BIN")
    gpu_device_id = pipeline_settings["gpu_device_id"]
    physical_device_id = 0 if gpu_device_id is None else gpu_device_id
    runner_env = os.environ.copy()
    if gpu_device_id is not None:
        runner_env["MACA_VISIBLE_DEVICES"] = str(gpu_device_id)
    if kernel in ("flash_attn", "fused_moe", "vector"):
        hardware_runner_path = get_kernel_script_path(cfg, kernel)
    elif kernel == "gemm":
        hardware_runner_path = None
    else:
        raise ValueError(f"unsupported MetaX kernel: {kernel!r}")

    print(f">>> Stage: Profile [{kernel}], {len(shapes)} shapes, output: {trace_dir}")
    if skip_profiler:
        print("    (mcProfiler SKIPPED — mcTracer only)")
    else:
        mcprofiler_bin = require_profiler_executable(
            "mcProfiler", "MCPROFILER_BIN")
        print(f"    mcProfiler: {mcprofiler_bin}")

    for i, shape in enumerate(shapes, 1):
        trace_subdir = os.path.join(trace_dir, shape["filename"])

        if dry_run:
            print(f"  [{i}/{len(shapes)}] DRY-RUN: {shape['filename']}")
            continue

        os.makedirs(trace_subdir, exist_ok=True)

        # ── mcTracer (no profiler_wait — fast startup OK) ──
        runner_script = os.path.join(trace_dir, f"_runner_{shape['filename']}.py")
        runner_command = _runner_command(
            kernel, shape, runner_script, python_bin, 0,
            hardware_runner_path, use_marker=True)

        existing_traces = glob.glob(os.path.join(trace_subdir, "tracer_out*.json"))
        if existing_traces:
            print(f"  [{i}/{len(shapes)}] mcTracer: {shape['filename']} ... SKIP (trace exists)")
        else:
            print(f"  [{i}/{len(shapes)}] mcTracer: {shape['filename']} ... ", end="", flush=True)

            # mcTracer treats ``--odname`` as a folder name relative to its
            # process working directory even when the supplied value begins
            # with ``/``.  Passing an absolute path therefore makes the tool
            # concatenate ``cwd`` and that absolute-looking value.  Keep the
            # output contract explicit: run from the per-kernel trace root and
            # give mcTracer only the case-directory basename.
            cmd = [
                mc_tracer_bin, "--odname",
                os.path.basename(trace_subdir), *runner_command,
            ]
            t0 = time.time()
            try:
                result = subprocess.run(
                    cmd, cwd=os.path.dirname(trace_subdir),
                    capture_output=True, text=True,
                    timeout=300, check=True, env=runner_env)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"mcTracer failed for {shape['filename']} with exit "
                    f"code {error.returncode}; "
                    f"stdout={_captured_subprocess_tail(error.stdout)!r}; "
                    f"stderr={_captured_subprocess_tail(error.stderr)!r}"
                ) from error
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(
                    f"mcTracer timed out for {shape['filename']} after "
                    f"{error.timeout} seconds; "
                    f"stdout={_captured_subprocess_tail(error.stdout)!r}; "
                    f"stderr={_captured_subprocess_tail(error.stderr)!r}"
                ) from error
            elapsed = time.time() - t0

            existing_traces = glob.glob(os.path.join(trace_subdir, "tracer_out*.json"))
            if existing_traces:
                print(f"OK ({elapsed:.1f}s)")
            else:
                raise RuntimeError(
                    f"mcTracer produced no trace for {shape['filename']}; "
                    f"stdout={_captured_subprocess_tail(result.stdout)!r}; "
                    f"stderr={_captured_subprocess_tail(result.stderr)!r}")

        if skip_profiler:
            continue

        _union_trace_kernel_events(existing_traces)
        profiler_out = os.path.abspath(
            os.path.join(trace_subdir, "mcprofiler_out"))
        runner_script_prof = os.path.abspath(os.path.join(
            trace_dir, f"_runner_prof_{shape['filename']}.py"))
        wait_sec = _PROFILER_WAIT_SECONDS[kernel]
        profiler_runner_command = _runner_command(
            kernel, shape, runner_script_prof, python_bin, wait_sec,
            hardware_runner_path, use_marker=True)

        # MACA_ETC_PATH belongs to the profiler server environment only. The
        # profiled child receives MACA_PATH and the runtime library path.
        maca_etc_path = require_environment_directory("MACA_ETC_PATH")
        maca_path = require_environment_directory("MACA_PATH")
        configured_maca_path = os.environ["MACA_PATH"]
        if configured_maca_path.endswith(os.sep):
            raise ValueError("MACA_PATH must not end with a path separator")
        profiler_cmdline = (
            f"export MACA_PATH={shlex.quote(maca_path)}; "
            "export LD_LIBRARY_PATH=/opt/maca/lib/:$LD_LIBRARY_PATH; "
            f"{shlex.join(profiler_runner_command)}"
        )
        profiler_env = dict(runner_env)
        profiler_env["MACA_PATH"] = maca_path
        profiler_env["MACA_ETC_PATH"] = maca_etc_path

        if _whole_shape_report_exists(profiler_out):
            print(
                f"  [{i}/{len(shapes)}] mcProfiler: "
                f"{shape['filename']} ... SKIP (report exists)")
        else:
            print(
                f"  [{i}/{len(shapes)}] mcProfiler: "
                f"{shape['filename']} ... ", end="", flush=True)
            whole_command = [
                mcprofiler_bin, "perf_exec",
                "--cmdline", profiler_cmdline,
                "--casename", shape["filename"],
                "--output", profiler_out,
                "--host", "127.0.0.1",
                "--port", str(port),
                "--single-pass",
            ]
            started = time.time()
            whole_log = os.path.join(
                trace_subdir, "mcprofiler_whole_shape.log")
            _run_profiler(
                whole_command, whole_log, 900, profiler_env, trace_subdir,
                physical_device_id)
            if not _whole_shape_report_exists(profiler_out):
                raise RuntimeError(
                    "mcProfiler produced no whole-shape report for "
                    f"{shape['filename']}; log={whole_log}")
            print(f"OK ({time.time() - started:.1f}s)")

        allk_out = os.path.join(profiler_out, "_allk")
        state, file_count = _layer1_capture_state(allk_out)
        if state == "per_invocation":
            print(
                f"  [{i}/{len(shapes)}] layer-1: {shape['filename']} ... "
                f"SKIP ({state}, {file_count} file(s))")
            continue

        layer1_command = [
            mcprofiler_bin, "perf_exec",
            "--cmdline", profiler_cmdline,
            "--casename", shape["filename"],
            "--output", allk_out,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--custom",
            "--per-kernel",
        ]
        failures = []
        for attempt in range(1, _LAYER1_MAX_ATTEMPTS + 1):
            print(
                f"  [{i}/{len(shapes)}] layer-1: {shape['filename']} "
                f"attempt {attempt}/{_LAYER1_MAX_ATTEMPTS} ... ",
                end="", flush=True)
            layer1_log = os.path.join(
                trace_subdir, f"mcprofiler_layer1_attempt_{attempt}.log")
            started = time.time()
            try:
                _run_profiler(
                    layer1_command, layer1_log, 2400,
                    profiler_env, trace_subdir, physical_device_id)
            except (OSError, subprocess.SubprocessError, TimeoutError) as error:
                failures.append(f"attempt {attempt}: {error}")
                print(f"FAILED ({time.time() - started:.1f}s)")
            else:
                state, file_count = _layer1_capture_state(allk_out)
                if state == "per_invocation":
                    print(
                        f"OK ({time.time() - started:.1f}s, {state}, "
                        f"{file_count} file(s))")
                    break
                failures.append(
                    f"attempt {attempt}: layer-1 state={state}, "
                    f"log={layer1_log}")
                print(f"FAILED ({time.time() - started:.1f}s, {state})")
        else:
            raise RuntimeError(
                "mcProfiler layer-1 capture failed for "
                f"{shape['filename']}: {'; '.join(failures)}")

    print(f"\n=== Profile [{kernel}] done ===")


def _join_layer1_observations(allk_dir, kernel_events, shape_name):
    """Pair trace and layer-1 observations one-to-one in global launch order."""

    if not isinstance(kernel_events, list) or not kernel_events:
        raise TypeError("kernel_events must be a non-empty list")
    if not os.path.isdir(allk_dir):
        raise FileNotFoundError(
            f"layer-1 output is missing for {shape_name}: {allk_dir}")
    allk_files = _allk_real_files(allk_dir)
    if not allk_files:
        raise FileNotFoundError(
            "layer-1 produced no numbered per-invocation files for "
            f"{shape_name}; whole-shape reports are archival only")

    recapture_contract = (
        "recapture this shape with one shape and exactly one profiled call "
        "per marker window"
    )
    if len(allk_files) != len(kernel_events):
        raise ValueError(
            "mcTracer and layer-1 invocation counts cannot be paired "
            f"uniquely for {shape_name}: trace={len(kernel_events)}, "
            f"layer1={len(allk_files)}; {recapture_contract}")

    matches_by_event = []
    trace_identities_by_file = defaultdict(set)
    for kernel_event in kernel_events:
        matched_files = _matching_allk_files(
            allk_dir, kernel_event["name"])
        if not matched_files:
            raise ValueError(
                "layer-1 invocation identity cannot be matched to mcTracer "
                f"for {shape_name}: kernel={kernel_event['name']!r}; "
                f"{recapture_contract}")
        matches_by_event.append(matched_files)
        trace_identity = _canonical_kernel_name(kernel_event["name"])
        for matched_file in matched_files:
            trace_identities_by_file[matched_file].add(trace_identity)
    ambiguous_files = {
        path: sorted(identities)
        for path, identities in trace_identities_by_file.items()
        if len(identities) > 1
    }
    if ambiguous_files:
        raise ValueError(
            "layer-1 truncated kernel identities are ambiguous across trace "
            f"invocations for {shape_name}: {ambiguous_files}; "
            f"{recapture_contract}")

    joined = []
    for ordinal, (kernel_event, allk_file, matched_files) in enumerate(zip(
            kernel_events, allk_files, matches_by_event), 1):
        if allk_file not in matched_files:
            raise ValueError(
                "mcTracer and layer-1 global launch order disagree for "
                f"{shape_name} at ordinal {ordinal}: "
                f"trace={kernel_event['name']!r}, "
                f"layer1={_perkernel_file_stem(allk_file)!r}; "
                f"{recapture_contract}")
        joined.append((
            kernel_event,
            _average_perkernel_files((allk_file,), "per_invocation"),
        ))
    return joined, []


def stage_aggregate(cfg, kernel, *, candidate_only=False):
    """Join trace launches to isolated layer-1 counters.

    ``candidate_only`` writes the completeness-gated result under
    ``profiler_output`` without mutating the protected canonical dataset.
    """
    if not isinstance(candidate_only, bool):
        raise TypeError("candidate_only must be boolean")
    kernel_config_path = get_kernel_config_path(cfg, kernel)
    kernel_cfg = _load_kernel_config(kernel_config_path, kernel)

    # Clock frequencies come from the hw config (single source of truth), not
    # hardcoded. mcTracer does not report per-kernel frequency, so these device
    # MAX frequencies are used to back out elapsed_cycles from durations.
    from src.kernel.hardware_spec import HWSpec
    hw_path = get_hardware_config_path(cfg)
    hw = HWSpec(hw_path)
    sm_freq_hz = hw.nominal_sm_freq_hz
    dram_freq_hz = hw.nominal_dram_freq_hz
    print(
        f"\033[93m[WARNING] Using device MAX clocks from {hw_path} "
        f"(sm={sm_freq_hz/1e9:.4g} GHz, dram={dram_freq_hz/1e9:.4g} GHz) to "
        f"derive elapsed_cycles. mcTracer does not report actual per-kernel "
        f"frequencies, so these are NOMINAL PEAKS — real run frequencies may be "
        f"lower under DVFS. elapsed_cycles is an upper-bound approximation.\033[0m")

    shapes = build_shapes(kernel, kernel_cfg)
    trace_dir = get_kernel_trace_dir(cfg, kernel)

    print(f"\n>>> Stage: Aggregate [{kernel}], parsing traces from {trace_dir}")

    dataset = {}
    out_of_scope = []

    for shape in shapes:
        trace_subdir = os.path.join(trace_dir, shape["filename"])
        trace_files = glob.glob(os.path.join(trace_subdir, "tracer_out*.json"))
        if not trace_files:
            raise FileNotFoundError(
                f"no mcTracer capture for {shape['filename']}")

        kernel_events = _union_trace_kernel_events(trace_files)

        profiler_dir = os.path.join(trace_subdir, "mcprofiler_out")
        # Whole-shape reports remain raw archival diagnostics.  Canonical
        # records consume only the ordinal-matched isolated layer-1 files.
        allk_dir = os.path.join(profiler_dir, "_allk")
        joined, shape_out_of_scope = _join_layer1_observations(
            allk_dir, kernel_events, shape["filename"])
        out_of_scope.extend(shape_out_of_scope)
        for item in shape_out_of_scope:
            print(
                f"  OUT-OF-SCOPE: {item['shape']} :: "
                f"{item['kernel'][:80]} (trace-only)")
        ncu_kernels = []
        for kernel_event, pk_data in joined:
            ncu_kernels.append(_kernel_to_ncu_format(
                kernel_event, sm_freq_hz, dram_freq_hz,
                hw.max_regs_per_unit, pk_data,
                _allowed_profiled_workgroups(shape, kernel_event)))

        timing_durations_ns = [
            _to_num(kernel_record["Sections"]
                    ["GPU Speed Of Light Throughput"]
                    ["Header"]["Duration [nsecond]"])
            for kernel_record in ncu_kernels
        ]
        total_dur_ns = math.fsum(timing_durations_ns)

        attributes = dict(shape["attr"])
        if kernel == "flash_attn":
            attributes["SP"] = _observed_flash_attn_splits(
                shape, trace_dir)
        entry = {
            "op_info": {
                "Type": shape["type"],
                "Attr": attributes,
                "evaluation": build_kernel_evaluation_timing(
                    "metax_c500",
                    ncu_kernels,
                    timing_durations_ns,
                ),
            },
            "kernels": ncu_kernels,
        }
        dataset[shape["filename"]] = entry

        # Inject MOE tiling info if available
        if kernel == "fused_moe":
            tiling_path = os.path.join(
                trace_dir, f"_runner_prof_{shape['filename']}_tiling.json")
            if not os.path.exists(tiling_path):
                alternate_path = os.path.join(
                    trace_dir, f"_runner_{shape['filename']}_tiling.json")
                if os.path.exists(alternate_path):
                    tiling_path = alternate_path
                else:
                    raise FileNotFoundError(
                        f"fused_moe tiling data is missing for "
                        f"{shape['filename']}")
            with open(tiling_path, encoding="utf-8") as f:
                tiling = json.load(f)
            attr = shape["attr"]
            moe_tiling = _build_moe_tiling(tiling, attr, ncu_kernels)
            dataset[shape["filename"]]["op_info"]["Attr"]["moe_tiling"] = moe_tiling
            raw_tok_per_expert = tiling["tok_per_expert"]
            if not isinstance(raw_tok_per_expert, list):
                raise TypeError("tiling.tok_per_expert must be a list")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   for value in raw_tok_per_expert):
                raise TypeError(
                    "tiling.tok_per_expert must contain integers")
            if any(value < 0 for value in raw_tok_per_expert):
                raise ValueError(
                    "tiling.tok_per_expert cannot contain negative values")
            tok_per_expert = list(raw_tok_per_expert)
            experts = require_positive_int(attr, "E", "FUSEDMOE Attr")
            tokens = require_positive_int(attr, "TOK", "FUSEDMOE Attr")
            topk = require_positive_int(attr, "TOPK", "FUSEDMOE Attr")
            if len(tok_per_expert) != experts:
                raise ValueError(
                    f"tok_per_expert length mismatch for {shape['filename']}")
            if sum(tok_per_expert) != tokens * topk:
                raise ValueError(
                    f"tok_per_expert sum mismatch for {shape['filename']}")
            dataset[shape["filename"]]["op_info"]["Attr"][
                "tok_per_expert"] = tok_per_expert

        print(
            f"  OK: {shape['filename']} ({len(ncu_kernels)} isolated "
            f"kernels, {total_dur_ns:.0f} ns)")

    dataset_path = get_kernel_dataset_path(cfg, kernel)
    candidate_dir = os.path.join(get_profiler_output_dir(cfg), "candidate")
    candidate_path = os.path.join(candidate_dir, f"{kernel}.json")
    isolation_report_path = os.path.join(
        candidate_dir, f"_isolation_report_{kernel}.json")
    os.makedirs(candidate_dir, exist_ok=True)
    with open(isolation_report_path, "w", encoding="utf-8") as stream:
        json.dump({
            "kernel": kernel,
            "out_of_scope": out_of_scope,
        }, stream, indent=2, ensure_ascii=False, allow_nan=False)
    expected_keys = [shape["filename"] for shape in shapes]
    if candidate_only:
        write_complete_candidate(dataset, expected_keys, candidate_path)
        output_path = candidate_path
    else:
        publish_complete_dataset(
            dataset, expected_keys, candidate_path, dataset_path)
        output_path = dataset_path
    print(
        f"\n[Success] {kernel} dataset: {output_path} "
        f"({len(dataset)} entries)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="C500 kernel dataset generation via mcTracer + mcProfiler")
    parser.add_argument("--config", default="configs/dataset/metax_c500.yaml")
    parser.add_argument(
        "--kernel", required=True,
        help="Comma-separated: gemm,flash_attn,fused_moe,vector")
    parser.add_argument(
        "--stage", required=True,
        choices=["profile", "aggregate", "metadata", "graph-trace-batch",
                 "graph-trace-family"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--candidate-only", action="store_true",
        help=("aggregate into profiler_output without publishing the "
              "protected canonical dataset"))
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument(
        "--skip-profiler", action="store_true",
        help=("Capture mcTracer only; aggregate will reject cases until their "
              "required mcProfiler observations exist"))
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start shape index (1-based, for serial resume)")
    parser.add_argument("--end-idx", type=int, default=0,
                        help="End shape index (inclusive, for serial resume)")
    parser.add_argument("--port", type=int, default=50123,
                        help=("mcProfiler server port; changing ports does not "
                              "make layer-1 profiling safe to parallelize"))
    parser.add_argument("--graph-trace-dir")
    parser.add_argument("--graph-trace-manifest")
    parser.add_argument("--graph-candidate-output")
    graph_measurement_group = parser.add_mutually_exclusive_group()
    graph_measurement_group.add_argument(
        "--graph-measured-repetition-count", type=int,
        help=("Select only measured Graph launches with this replay count "
              "when deriving mcTracer kernel duration"))
    graph_measurement_group.add_argument(
        "--graph-measured-repetition-policy",
        choices=[_GRAPH_MEASURED_REPETITION_POLICY_PER_CASE_HIGH],
        help=("Select measured Graph launches per case; per-case-high uses "
              "the largest measured replay count in each validated manifest "
              "launch plan"))
    parser.add_argument(
        "--graph-batch-candidate", action="append",
        help=("Existing strict Graph batch candidate to re-validate and merge; "
              "repeat once per non-overlapping batch"))
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    kernels = parse_kernel_list(cfg, args.kernel)

    if args.stage == "profile":
        if args.candidate_only:
            parser.error("--candidate-only is only valid with --stage aggregate")
        if (args.graph_trace_dir is not None
                or args.graph_trace_manifest is not None
                or args.graph_candidate_output is not None
                or args.graph_batch_candidate is not None
                or args.graph_measured_repetition_count is not None
                or args.graph_measured_repetition_policy is not None):
            parser.error(
                "graph trace options require a graph-trace stage")
        for kernel in kernels:
            stage_profile(cfg, kernel, dry_run=args.dry_run,
                         max_tasks=args.max_tasks,
                         skip_profiler=args.skip_profiler,
                         start_idx=args.start_idx,
                         end_idx=args.end_idx,
                         port=args.port)
    elif args.stage == "metadata":
        if args.candidate_only:
            parser.error("--candidate-only is only valid with --stage aggregate")
        if kernels != ["flash_attn"]:
            parser.error("metadata stage requires --kernel flash_attn")
        if args.max_tasks or args.skip_profiler or args.port != 50123:
            parser.error(
                "--max-tasks, --skip-profiler, and --port are not valid with "
                "--stage metadata")
        if (args.graph_trace_dir is not None
                or args.graph_trace_manifest is not None
                or args.graph_candidate_output is not None
                or args.graph_batch_candidate is not None
                or args.graph_measured_repetition_count is not None
                or args.graph_measured_repetition_policy is not None):
            parser.error(
                "graph trace options require a graph-trace stage")
        stage_flash_attn_metadata(
            cfg, dry_run=args.dry_run,
            start_idx=args.start_idx, end_idx=args.end_idx)
    elif args.stage == "aggregate":
        if args.dry_run:
            # --dry-run only applies to profile; aggregate writes the real
            # kernel_dataset/<hw>/<kernel>.json.zst and would clobber existing
            # data if run. Refuse to avoid silent overwrite.
            parser.error("--dry-run is only valid with --stage profile")
        if (args.max_tasks or args.skip_profiler or args.start_idx
                or args.end_idx or args.port != 50123):
            parser.error(
                "--max-tasks, --skip-profiler, --start-idx, --end-idx, "
                "and --port are only valid with --stage profile")
        if (args.graph_trace_dir is not None
                or args.graph_trace_manifest is not None
                or args.graph_candidate_output is not None
                or args.graph_batch_candidate is not None
                or args.graph_measured_repetition_count is not None
                or args.graph_measured_repetition_policy is not None):
            parser.error(
                "graph trace options require a graph-trace stage")
        for kernel in kernels:
            stage_aggregate(
                cfg, kernel, candidate_only=args.candidate_only)
    elif args.stage == "graph-trace-batch":
        if args.dry_run or args.candidate_only:
            parser.error(
                "--dry-run and --candidate-only are not valid with "
                "--stage graph-trace-batch")
        if (args.max_tasks or args.skip_profiler or args.start_idx
                or args.end_idx or args.port != 50123):
            parser.error(
                "profile range, profiler, and port options are not valid "
                "with --stage graph-trace-batch")
        if len(kernels) != 1:
            parser.error("graph-trace-batch requires exactly one kernel")
        if (args.graph_trace_dir is None
                or args.graph_trace_manifest is None
                or args.graph_candidate_output is None):
            parser.error(
                "graph-trace-batch requires --graph-trace-dir, "
                "--graph-trace-manifest, and --graph-candidate-output")
        if args.graph_batch_candidate is not None:
            parser.error(
                "--graph-batch-candidate requires "
                "--stage graph-trace-family")
        output = _write_graph_trace_batch_candidate(
            cfg, kernels[0], args.graph_trace_dir,
            args.graph_trace_manifest, args.graph_candidate_output,
            measured_repetition_count=args.graph_measured_repetition_count,
            measured_repetition_policy=args.graph_measured_repetition_policy)
        print(f"[Success] C500 graph trace batch candidate: {output}")
    elif args.stage == "graph-trace-family":
        if args.dry_run or args.candidate_only:
            parser.error(
                "--dry-run and --candidate-only are not valid with "
                "--stage graph-trace-family")
        if (args.max_tasks or args.skip_profiler or args.start_idx
                or args.end_idx or args.port != 50123):
            parser.error(
                "profile range, profiler, and port options are not valid "
                "with --stage graph-trace-family")
        if len(kernels) != 1:
            parser.error("graph-trace-family requires exactly one kernel")
        if (args.graph_trace_dir is not None
                or args.graph_trace_manifest is not None
                or args.graph_measured_repetition_count is not None
                or args.graph_measured_repetition_policy is not None):
            parser.error(
                "--graph-trace-dir and --graph-trace-manifest require "
                "--stage graph-trace-batch")
        if args.graph_candidate_output is None:
            parser.error(
                "graph-trace-family requires --graph-candidate-output")
        if args.graph_batch_candidate is None:
            parser.error(
                "graph-trace-family requires --graph-batch-candidate")
        output = _write_graph_trace_family_candidate(
            cfg, kernels[0], args.graph_batch_candidate,
            args.graph_candidate_output)
        print(f"[Success] C500 graph trace family candidate: {output}")
    else:
        raise ValueError(f"unsupported dataset stage: {args.stage!r}")
