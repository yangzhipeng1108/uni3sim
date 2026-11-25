"""vLLM server lifecycle for ShareGPT ground-truth capture: launch
plans, environment scrubbing, graph/async log validation, and
process-group termination."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from evaluation.serving.experiments.experiment_plan import git_code_identity
from src.serving.request_gen import load_canonical_requests

from evaluation.serving.experiments.ground_truth_case_contract import (
    ROOT,
    sha256_file,
)


PROFILER_ENVIRONMENT_KEYS = (
    "CUDA_INJECTION64_PATH",
    "CUDA_TRACE_OUTPUT",
    "CUDA_TRACE_REQUIRE_PROFILER",
    "LD_PRELOAD",
    "NSYS_NVTX_PROFILER_REGISTER_ONLY",
    "VLLM_CUSTOM_SCOPES_FOR_PROFILING",
    "VLLM_NVTX_SCOPES_FOR_PROFILING",
    "VLLM_TORCH_PROFILER_DIR",
    "VLLM_TRACE_FUNCTION",
)

GROUND_TRUTH_SCHEDULER_POLICY = "fcfs"

EXECUTION_ENVIRONMENT_PREFIXES = (
    "ASCEND_",
    "ATB_",
    "CUDA_",
    "HCCL_",
    "MACA_",
    "NCCL_",
    "TASK_QUEUE_",
    "TORCH_",
    "VLLM_",
)
_ASCEND_STEP_TIMESTAMP_ENV = "MIDLAYER_910B_MSPROF_STEP_TIMESTAMP"
_ASCEND_MSPROFTX_STEP_MARK_ENV = "MIDLAYER_910B_MSPROFTX_STEP_MARK"
C500_DEVICE_EVENT_ENVIRONMENT_KEYS = (
    "MIDLAYER_C500_DEVICE_EVENT_JSONL",
    "MIDLAYER_C500_EVENT_POST_SOURCE_SHA256",
    "MIDLAYER_C500_EVENT_PATCH_SHA256",
    "MIDLAYER_C500_EVENT_SOURCE_PATH",
)


_SERVER_TERMINATION_TIMEOUT_SECONDS = 30.0
_PROCESS_GROUP_POLL_INTERVAL_SECONDS = 0.05
_SCHEDULER_TRACE_BOOTSTRAP = (
    ROOT
    / "dataset_pipeline"
    / "serving_dataset"
    / "instrumentation_bootstrap"
)


def _c500_device_event_environment(
    hardware_id: str, runtime: dict[str, Any], case: dict[str, Any]
) -> dict[str, str]:
    present = {
        key for key in C500_DEVICE_EVENT_ENVIRONMENT_KEYS if key in os.environ
    }
    if not present:
        return {}
    if present != set(C500_DEVICE_EVENT_ENVIRONMENT_KEYS):
        missing = sorted(set(C500_DEVICE_EVENT_ENVIRONMENT_KEYS) - present)
        raise ValueError(
            "C500 device Event environment is incomplete; missing "
            + ", ".join(missing)
        )
    if hardware_id != "metax_c500" or runtime["runtime_version"] != "0.24.0":
        raise ValueError("C500 device Event instrumentation requires MetaX vLLM 0.24.0")
    parallel_sizes = (
        runtime["tensor_parallel_size"],
        runtime["pipeline_parallel_size"],
        runtime["expert_parallel_size"],
    )
    if parallel_sizes != (1, 1, 1):
        raise ValueError("C500 device Event instrumentation requires TP1/PP1/EP1")

    environment = {
        key: os.environ[key] for key in C500_DEVICE_EVENT_ENVIRONMENT_KEYS
    }
    if any(not value for value in environment.values()):
        raise ValueError("C500 device Event environment values must be non-empty")
    for key in (
        "MIDLAYER_C500_EVENT_POST_SOURCE_SHA256",
        "MIDLAYER_C500_EVENT_PATCH_SHA256",
    ):
        digest = environment[key]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{key} must be a lowercase SHA256 digest")

    output_path = Path(environment["MIDLAYER_C500_DEVICE_EVENT_JSONL"])
    if not output_path.is_absolute():
        raise ValueError("MIDLAYER_C500_DEVICE_EVENT_JSONL must be absolute")
    if output_path.exists():
        raise ValueError(f"refusing to overwrite device Event output: {output_path}")
    if not output_path.parent.is_dir():
        raise ValueError("C500 device Event output parent must already exist")

    source_path = Path(environment["MIDLAYER_C500_EVENT_SOURCE_PATH"])
    if not source_path.is_absolute() or not source_path.is_file():
        raise ValueError("MIDLAYER_C500_EVENT_SOURCE_PATH must be an existing file")
    source_path = source_path.resolve()
    expected_suffix = Path("vllm/v1/worker/gpu_model_runner.py")
    if Path(*source_path.parts[-4:]) != expected_suffix:
        raise ValueError("C500 device Event source path has an unexpected suffix")
    if sha256_file(source_path) != environment[
        "MIDLAYER_C500_EVENT_POST_SOURCE_SHA256"
    ]:
        raise ValueError("C500 device Event source SHA256 differs")
    environment["MIDLAYER_C500_EVENT_SOURCE_PATH"] = str(source_path)

    request_stream = load_canonical_requests(case["requests_path"])
    requests = request_stream.requests
    if len(requests) != case["requests_row_count"]:
        raise ValueError("requests.csv row count disagrees with case manifest")
    if any(request.output_tokens < 1 for request in requests):
        raise ValueError("ground-truth requests must generate at least one token")
    if any(request.prefix_hit_tokens != 0 for request in requests):
        raise ValueError(
            "C500 device Event instrumentation requires zero prefix-hit tokens"
        )
    expected_scheduled_tokens = sum(
        request.input_tokens + request.output_tokens - 1
        for request in requests
    )
    environment[
        "MIDLAYER_C500_DEVICE_EVENT_EXPECTED_SCHEDULED_TOKENS"
    ] = str(expected_scheduled_tokens)

    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        raise ValueError("C500 device Event instrumentation requires PYTHONPATH")
    python_paths = tuple(
        Path(entry).resolve()
        for entry in os.environ["PYTHONPATH"].split(os.pathsep)
        if entry
    )
    overlay_root = source_path.parents[3]
    if overlay_root not in python_paths:
        raise ValueError("PYTHONPATH does not select the C500 device Event overlay")
    environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return environment


# Whether the backend can offload weights above the GPU: the CUDA backends
# have no CPU-offload flag, so their plan passes 0; the MACA and CANN backends
# take no such argument at all, so their plan passes None.
_CPU_OFFLOAD_GB = {
    "nvidia_a800": 0,
    "nvidia_h200": 0,
    "metax_c500": None,
    "ascend_910b": None,
}


# The launch record each backend needs.  The stored key is the *device
# visibility* environment variable for that backend; the rest is the argv and
# resolved plan fields the hand-written branches set.
#
# ``by_runtime_version`` overrides fields for a backend that ships more than
# one release (H200's 0.11.0 vs 0.30.0 renamed the compilation "level" key
# and added --cudagraph-metrics; MetaX's 0.8.5 is the v0 engine).  A backend
# with ``by_runtime_version`` set requires its runtime version to be listed.
_LAUNCH_RECORD_DEFAULTS = {
    "explicit_non_eager_argv": ["--no-enforce-eager"],
    "model_load_argv": [],
    "safetensors_load_strategy": None,
    "safetensors_prefetch_num_threads": None,
    "cpu_offload_argv": [],
}

_SERVER_LAUNCH_RECORDS = {
    "nvidia_a800": {
        "visibility_environment_key": "CUDA_VISIBLE_DEVICES",
        "engine_environment": {"VLLM_USE_V1": "1"},
        "compilation_argv": [
            "--compilation-config",
            json.dumps(
                {"level": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"},
                separators=(",", ":"),
            ),
        ],
        "engine_generation": "v1",
        "compilation_level": 3,
        "graph_mode": "full_and_piecewise_cuda_graph",
        "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
        "async_scheduling": True,
        "scheduling_argv": ["--async-scheduling"],
        "cpu_offload_argv": ["--cpu-offload-gb", "0"],
        "by_runtime_version": None,
        "unsupported_runtime_message": None,
    },
    "nvidia_h200": {
        "visibility_environment_key": "CUDA_VISIBLE_DEVICES",
        # Only the 0.11.0 release sets VLLM_USE_V1; 0.30.0 does not, so this
        # is empty here and the 0.11.0 override below adds it.
        "engine_environment": {},
        "compilation_argv": None,
        "engine_generation": "v1",
        "compilation_level": 3,
        "graph_mode": "full_and_piecewise_cuda_graph",
        "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
        "async_scheduling": True,
        "scheduling_argv": None,
        "cpu_offload_argv": ["--cpu-offload-gb", "0"],
        "by_runtime_version": {
            # Identical to the canonical A800 0.11.0 production contract.
            "0.11.0": {
                "engine_environment": {"VLLM_USE_V1": "1"},
                "compilation_argv": [
                    "--compilation-config",
                    json.dumps(
                        {"level": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"},
                        separators=(",", ":"),
                    ),
                ],
                "scheduling_argv": ["--async-scheduling"],
            },
            # 0.30.0 renamed compilation "level" to "mode" and reports
            # per-step CUDA Graph statistics only with --cudagraph-metrics.
            "0.30.0": {
                "compilation_argv": [
                    "--compilation-config",
                    json.dumps(
                        {"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"},
                        separators=(",", ":"),
                    ),
                ],
                "scheduling_argv": [
                    "--async-scheduling", "--cudagraph-metrics"],
            },
        },
        "unsupported_runtime_message": (
            "unsupported H200 runtime_version: {runtime_version!r}"),
    },
    "metax_c500": {
        "visibility_environment_key": "MACA_VISIBLE_DEVICES",
        "engine_environment": {},
        "compilation_argv": None,
        "engine_generation": None,
        "compilation_level": None,
        "graph_mode": None,
        "configured_cudagraph_mode": None,
        "async_scheduling": None,
        "scheduling_argv": None,
        "cpu_offload_argv": None,
        "by_runtime_version": {
            "0.8.5": {
                "engine_environment": {"VLLM_USE_V1": "0"},
                "explicit_non_eager_argv": [],
                "compilation_argv": [],
                "cpu_offload_argv": [],
                "engine_generation": "v0",
                "compilation_level": 0,
                "graph_mode": "cuda_graph_hybrid",
                "configured_cudagraph_mode": None,
                "async_scheduling": False,
                "scheduling_argv": [],
            },
            "0.24.0": {
                "engine_environment": {
                    "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE": "0",
                    "USE_VLLM_TRITON_EXPERT": "0",
                },
                "compilation_argv": [
                    "--compilation-config",
                    json.dumps({"mode": 3}, separators=(",", ":")),
                ],
                "model_load_argv": [
                    "--safetensors-load-strategy",
                    "prefetch",
                    "--safetensors-prefetch-num-threads",
                    "1",
                ],
                "safetensors_load_strategy": "prefetch",
                "safetensors_prefetch_num_threads": 1,
                "engine_generation": "v1",
                "compilation_level": 3,
                "graph_mode": "piecewise_cuda_graph",
                "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
                "async_scheduling": True,
                "scheduling_argv": [
                    "--async-scheduling", "--cudagraph-metrics"],
                "cpu_offload_argv": [],
            },
        },
        "unsupported_runtime_message": (
            "unsupported MetaX production ground-truth runtime_version: "
            "{runtime_version!r}"),
    },
    "ascend_910b": {
        "visibility_environment_key": "ASCEND_RT_VISIBLE_DEVICES",
        "engine_environment": {"VLLM_USE_V1": "1"},
        "compilation_argv": [
            "--compilation-config",
            json.dumps(
                {
                    "cudagraph_mode": "FULL",
                    "max_cudagraph_capture_size": 512,
                },
                separators=(",", ":"),
            ),
        ],
        "engine_generation": "v1",
        "compilation_level": None,
        "graph_mode": "acl_graph_full",
        "configured_cudagraph_mode": "FULL",
        "async_scheduling": True,
        "scheduling_argv": ["--async-scheduling", "--cudagraph-metrics"],
        "cpu_offload_argv": [],
        "by_runtime_version": None,
        "unsupported_runtime_message": None,
    },
}


def _launch_record(hardware_id: str, runtime_version: str) -> dict[str, Any]:
    """Return the resolved launch fields ``hardware_id`` needs at its release."""

    if hardware_id not in _SERVER_LAUNCH_RECORDS:
        raise ValueError(
            f"unsupported server hardware_id: {hardware_id!r}")
    record = _SERVER_LAUNCH_RECORDS[hardware_id]
    resolved = dict(_LAUNCH_RECORD_DEFAULTS)
    for key, value in record.items():
        if key not in ("by_runtime_version", "unsupported_runtime_message"):
            resolved[key] = value
    by_runtime_version = record["by_runtime_version"]
    if by_runtime_version is not None:
        if runtime_version not in by_runtime_version:
            raise ValueError(record["unsupported_runtime_message"].format(
                runtime_version=runtime_version))
        resolved.update(by_runtime_version[runtime_version])
    return resolved


def _server_plan(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    runtime = case["runtime"]
    hardware_id = case["identity"]["hardware_id"]
    devices = _validate_devices(args.devices, runtime)
    visible_devices = ",".join(str(device) for device in devices)
    environment = {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith(EXECUTION_ENVIRONMENT_PREFIXES)
        and key not in {"CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES"}
    }
    ascend_step_boundary_names = (
        _ASCEND_STEP_TIMESTAMP_ENV,
        _ASCEND_MSPROFTX_STEP_MARK_ENV,
    )
    enabled_ascend_boundaries = [
        name for name in ascend_step_boundary_names if name in os.environ
    ]
    if len(enabled_ascend_boundaries) > 1:
        raise ValueError(
            "910B profiler step boundary mechanisms are mutually exclusive"
        )
    for name in enabled_ascend_boundaries:
        if hardware_id != "ascend_910b":
            raise ValueError(f"{name} is supported only on ascend_910b")
        if os.environ[name] != "1":
            raise ValueError(f"{name} must be exactly '1'")
        environment[name] = "1"
    launch_record = _launch_record(hardware_id, runtime["runtime_version"])
    environment[launch_record["visibility_environment_key"]] = visible_devices
    environment.update(launch_record["engine_environment"])
    explicit_non_eager_argv = launch_record["explicit_non_eager_argv"]
    compilation_argv = launch_record["compilation_argv"]
    model_load_argv = launch_record["model_load_argv"]
    safetensors_load_strategy = launch_record["safetensors_load_strategy"]
    safetensors_prefetch_num_threads = launch_record[
        "safetensors_prefetch_num_threads"]
    engine_generation = launch_record["engine_generation"]
    compilation_level = launch_record["compilation_level"]
    graph_mode = launch_record["graph_mode"]
    configured_cudagraph_mode = launch_record["configured_cudagraph_mode"]
    async_scheduling = launch_record["async_scheduling"]
    scheduling_argv = launch_record["scheduling_argv"]
    cpu_offload_argv = launch_record["cpu_offload_argv"]
    device_event_environment = _c500_device_event_environment(
        hardware_id, runtime, case
    )
    environment.update(device_event_environment)
    shutdown_timeout_seconds = 30 if device_event_environment else 0
    shutdown_argv = (
        ["--shutdown-timeout", str(shutdown_timeout_seconds)]
        if shutdown_timeout_seconds
        else []
    )
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    scheduler_trace_output = None
    scheduler_request_add_output = None
    counter_only_output = None
    if (
        "scheduler_trace_output" in vars(args)
        and args.scheduler_trace_output is not None
    ):
        if engine_generation != "v1":
            raise ValueError(
                "Scheduler trace instrumentation currently requires vLLM V1"
            )
        scheduler_trace_output = Path(args.scheduler_trace_output).resolve()
        if scheduler_trace_output.exists():
            raise ValueError(
                "refusing to overwrite Scheduler trace output: "
                f"{scheduler_trace_output}"
            )
        if _uses_worker_dispatch_trace(hardware_id, runtime):
            for rank in range(runtime["tensor_parallel_size"]):
                for path in (
                    Path(f"{scheduler_trace_output}.rank-{rank}.jsonl"),
                    Path(
                        f"{scheduler_trace_output}.compilation-counter.rank-{rank}.jsonl"
                    ),
                ):
                    if path.exists():
                        raise ValueError(
                            "refusing to overwrite A800 Scheduler trace sidecar: "
                            f"{path}"
                        )
    if (
        "scheduler_request_add_output" in vars(args)
        and args.scheduler_request_add_output is not None
    ):
        if scheduler_trace_output is None:
            raise ValueError(
                "Scheduler request-add timing requires Scheduler tracing"
            )
        scheduler_request_add_output = Path(
            args.scheduler_request_add_output
        ).resolve()
        if scheduler_request_add_output.exists():
            raise ValueError(
                "refusing to overwrite Scheduler request-add output: "
                f"{scheduler_request_add_output}"
            )
        stage_sidecars = tuple(
            scheduler_request_add_output.parent.glob(
                f"{scheduler_request_add_output.name}.stages.*.jsonl"
            )
        )
        if stage_sidecars:
            raise ValueError(
                "refusing to reuse Scheduler request-stage sidecars: "
                + ", ".join(str(path) for path in stage_sidecars)
            )
    if (
        "a800_compilation_counter_output" in vars(args)
        and args.a800_compilation_counter_output is not None
    ):
        if hardware_id != "nvidia_a800":
            raise ValueError(
                "A800 compilation-counter instrumentation requires nvidia_a800"
            )
        if scheduler_trace_output is not None:
            raise ValueError(
                "A800 counter-only instrumentation cannot accompany Scheduler trace"
            )
        counter_only_output = Path(
            args.a800_compilation_counter_output
        ).resolve()
        if counter_only_output.exists():
            raise ValueError(
                "refusing to overwrite A800 counter-only output: "
                f"{counter_only_output}"
            )
        for rank in range(runtime["tensor_parallel_size"]):
            path = Path(
                f"{counter_only_output}.compilation-counter.rank-{rank}.jsonl"
            )
            if path.exists():
                raise ValueError(
                    "refusing to overwrite A800 counter-only sidecar: "
                    f"{path}"
                )
    instrumentation_output = (
        scheduler_trace_output
        if scheduler_trace_output is not None
        else counter_only_output
    )
    if instrumentation_output is not None:
        python_paths = [str(_SCHEDULER_TRACE_BOOTSTRAP), str(ROOT)]
        if "PYTHONPATH" in os.environ and os.environ["PYTHONPATH"]:
            python_paths.append(os.environ["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        environment["MIDLAYER_SCHEDULER_TRACE_SPAWN_WRAPPER"] = str(
            _SCHEDULER_TRACE_BOOTSTRAP / "spawn_with_scheduler_trace.sh"
        )
        environment["MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHON"] = sys.executable
        environment["MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHONPATH"] = os.pathsep.join(
            (str(_SCHEDULER_TRACE_BOOTSTRAP), str(ROOT))
        )
        if scheduler_trace_output is not None:
            source_identity = runtime["source_commit"]
            if source_identity is None:
                source_identity = git_code_identity(ROOT)["git_commit"]
            if not isinstance(source_identity, str) or not source_identity:
                raise ValueError(
                    "Scheduler trace requires a non-empty source identity")
            environment["MIDLAYER_SCHEDULER_TRACE_OUTPUT"] = str(
                scheduler_trace_output
            )
            environment["MIDLAYER_SCHEDULER_TRACE_BOOTSTRAP_MARKER"] = str(
                Path(f"{scheduler_trace_output}.bootstrap.jsonl")
            )
            environment["MIDLAYER_SCHEDULER_TRACE_REQUEST_COUNT"] = str(
                case["requests_row_count"]
            )
            environment["MIDLAYER_SCHEDULER_TRACE_SOURCE_IDENTITY"] = (
                source_identity
            )
            if scheduler_request_add_output is not None:
                environment["MIDLAYER_SCHEDULER_REQUEST_ADD_OUTPUT"] = str(
                    scheduler_request_add_output
                )
            if configured_cudagraph_mode is not None:
                environment[
                    "MIDLAYER_SCHEDULER_TRACE_CONFIGURED_CUDAGRAPH_MODE"
                ] = configured_cudagraph_mode
                environment["MIDLAYER_SCHEDULER_TRACE_MAX_IN_FLIGHT"] = str(
                    2 if async_scheduling else 1
                )
        else:
            environment["MIDLAYER_A800_COUNTER_ONLY_OUTPUT"] = str(
                counter_only_output
            )
        # All V1 Scheduler trace installations wrap the executor whenever a
        # configured Graph mode is present.  The bootstrap must therefore know
        # which platform-specific UniProcExecutor implementation to import;
        # this is not an A800-only contract.
        environment["MIDLAYER_SCHEDULER_TRACE_HARDWARE_ID"] = hardware_id
        environment["MIDLAYER_SCHEDULER_TRACE_WORKER_RANK_COUNT"] = str(
            runtime["tensor_parallel_size"]
        )
    if runtime["expert_parallel_size"] == 1:
        expert_parallel_argv = []
    elif runtime["expert_parallel_size"] == 2:
        expert_parallel_argv = ["--enable-expert-parallel"]
    else:
        raise AssertionError("case contract allowed an unsupported EP size")
    executor_argv: list[str] = []
    if hardware_id == "ascend_910b" and runtime["runtime_version"] == "0.18.0":
        # vLLM 0.18 defaults TP1 to UniProcExecutor. The accepted 910B trace
        # may instead declare MultiprocExecutor; preserve that source fact
        # when replaying its canonical request schedule.
        executor_class = runtime["scheduler_pipeline_depth_evidence"][
            "executor_class"
        ]
        executor_backends = {
            "vllm.v1.executor.multiproc_executor.MultiprocExecutor": "mp",
            "vllm.v1.executor.uniproc_executor.UniProcExecutor": "uni",
        }
        if executor_class not in executor_backends:
            raise ValueError(
                f"unsupported 910B vLLM 0.18 executor: {executor_class!r}"
            )
        executor_argv = [
            "--distributed-executor-backend", executor_backends[executor_class]
        ]
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(Path(args.model).resolve(strict=True)),
        "--served-model-name",
        case["identity"]["model_id"],
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        runtime["dtype"],
        "--tensor-parallel-size",
        str(runtime["tensor_parallel_size"]),
        "--pipeline-parallel-size",
        str(runtime["pipeline_parallel_size"]),
        *executor_argv,
        *expert_parallel_argv,
        *model_load_argv,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--scheduling-policy",
        GROUND_TRUTH_SCHEDULER_POLICY,
        *scheduling_argv,
        "--enable-prefix-caching",
        *cpu_offload_argv,
        "--block-size",
        str(args.kv_block_size_tokens),
        *explicit_non_eager_argv,
        "--seed",
        str(args.seed),
        *shutdown_argv,
        *compilation_argv,
    ]
    if hardware_id not in _CPU_OFFLOAD_GB:
        raise ValueError(f"unsupported server hardware_id: {hardware_id!r}")
    cpu_offload_gb = _CPU_OFFLOAD_GB[hardware_id]
    if scheduler_trace_output is None:
        scheduler_instrumentation_version = None
    elif _uses_worker_dispatch_trace(hardware_id, runtime):
        scheduler_instrumentation_version = 4
    elif configured_cudagraph_mode is not None:
        scheduler_instrumentation_version = 2
    elif hardware_id in ("metax_c500", "ascend_910b"):
        scheduler_instrumentation_version = 1
    else:
        raise ValueError(f"unsupported scheduler trace hardware: {hardware_id!r}")
    diagnostic_nsys = (
        os.environ["MIDLAYER_A800_NSYS_STEP_NVTX"]
        if "MIDLAYER_A800_NSYS_STEP_NVTX" in os.environ else None)
    graph_trace = (
        os.environ["MIDLAYER_A800_NSYS_GRAPH_TRACE"]
        if "MIDLAYER_A800_NSYS_GRAPH_TRACE" in os.environ else "node")
    if graph_trace not in ("node", "graph"):
        raise ValueError("A800 Nsight Graph trace must be 'node' or 'graph'")
    if diagnostic_nsys != "1" and "MIDLAYER_A800_NSYS_GRAPH_TRACE" in os.environ:
        raise ValueError("A800 Nsight Graph trace mode requires step diagnostics")
    if diagnostic_nsys == "1":
        if hardware_id != "nvidia_a800":
            raise ValueError("A800 Nsight step diagnostics require nvidia_a800")
        output_text = (
            os.environ["MIDLAYER_A800_NSYS_OUTPUT"]
            if "MIDLAYER_A800_NSYS_OUTPUT" in os.environ else None)
        if output_text is None:
            raise RuntimeError(
                "MIDLAYER_A800_NSYS_OUTPUT is required for A800 Nsight diagnostics"
            )
        output_path = Path(output_text)
        if not output_path.is_absolute():
            raise ValueError("A800 Nsight output path must be absolute")
        output_path = output_path.resolve()
        if any(path.exists() for path in (
                output_path.with_suffix(".nsys-rep"),
                output_path.with_suffix(".sqlite"))):
            raise FileExistsError("refusing to overwrite A800 Nsight diagnostic output")
        nsys_binary = (
            os.environ["NSYS_BIN"] if "NSYS_BIN" in os.environ else "nsys")
        nsys = shutil.which(nsys_binary)
        if nsys is None:
            raise FileNotFoundError("A800 Nsight diagnostic requires nsys")
        argv = [
            nsys, "profile", "--force-overwrite=true", "--trace=cuda,nvtx,osrt",
            f"--cuda-graph-trace={graph_trace}", "--trace-fork-before-exec=true",
            "--sample=none", "--output", str(output_path), *argv,
        ]
    return {
        "argv": argv,
        "python_executable": sys.executable,
        "environment": environment,
        "hardware_id": hardware_id,
        "devices": list(devices),
        "endpoint": {
            "base_url": f"http://127.0.0.1:{args.port}",
            "health_url": f"http://127.0.0.1:{args.port}/v1/models",
        },
        "engine": {
            "dtype": runtime["dtype"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
            "pipeline_parallel_size": runtime["pipeline_parallel_size"],
            "expert_parallel_size": runtime["expert_parallel_size"],
            "safetensors_load_strategy": safetensors_load_strategy,
            "safetensors_prefetch_num_threads": safetensors_prefetch_num_threads,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "scheduler_policy": GROUND_TRUTH_SCHEDULER_POLICY,
            "enable_prefix_caching": True,
            "cpu_offload_gb": cpu_offload_gb,
            "kv_block_size_tokens": args.kv_block_size_tokens,
            "enforce_eager": False,
            "vllm_engine_generation": engine_generation,
            "runtime_version": runtime["runtime_version"],
            "compilation_level": compilation_level,
            "graph_mode": graph_mode,
            "configured_cudagraph_mode": configured_cudagraph_mode,
            "cudagraph_metrics": "--cudagraph-metrics" in scheduling_argv,
            "async_scheduling": async_scheduling,
            # This is the source-defined executor setting selected by the
            # explicit ``--async-scheduling`` argv above.  Accepted capture
            # evidence still reads the live Executor property separately;
            # this field only makes the planned depth contract explicit.
            "max_concurrent_batches": 2 if async_scheduling else 1,
            "seed": args.seed,
            "shutdown_timeout_seconds": shutdown_timeout_seconds,
        },
        "profiler": {
            "requested": diagnostic_nsys == "1",
            "forbidden_environment_keys": list(PROFILER_ENVIRONMENT_KEYS),
        },
        "scheduler_trace": {
            "enabled": scheduler_trace_output is not None,
            "output_path": (
                None
                if scheduler_trace_output is None
                else str(scheduler_trace_output)
            ),
            "instrumentation_id": (
                None
                if scheduler_trace_output is None
                else "midlayer-vllm-scheduler-trace"
            ),
            "instrumentation_version": scheduler_instrumentation_version,
            "buffering": (
                None
                if scheduler_trace_output is None
                else "in_memory_append_then_single_write"
            ),
            "worker_rank_count": (
                None
                if scheduler_trace_output is None
                or not _uses_worker_dispatch_trace(hardware_id, runtime)
                else runtime["tensor_parallel_size"]
            ),
            "request_add_output_path": (
                None
                if scheduler_request_add_output is None
                else str(scheduler_request_add_output)
            ),
            "request_stage_output_pattern": (
                None
                if scheduler_request_add_output is None
                else f"{scheduler_request_add_output}.stages.*.jsonl"
            ),
        },
        "compilation_counter_instrumentation": {
            "enabled": instrumentation_output is not None
            and hardware_id == "nvidia_a800",
            "mode": (
                None
                if instrumentation_output is None
                else (
                    "scheduler_trace" if scheduler_trace_output is not None
                    else "counter_only"
                )
            ),
            "output_path": (
                None
                if instrumentation_output is None
                else str(instrumentation_output)
            ),
            "worker_rank_count": (
                None
                if instrumentation_output is None
                or hardware_id != "nvidia_a800"
                else runtime["tensor_parallel_size"]
            ),
        },
        "max_dispatch_lag_ms": args.max_dispatch_lag_ms,
        "arrival_probe_per_1000_tokens_ms": (
            args.arrival_probe_per_1000_tokens_ms),
    }


def _uses_worker_dispatch_trace(
    hardware_id: str, runtime: dict[str, Any]
) -> bool:
    """Whether the Scheduler trace observes graphs through worker dispatch.

    Mirrors ``vllm_scheduler_trace._uses_worker_dispatch_observations``: the
    vLLM 0.11.0 runtimes (A800, H200 0.11.0) have no per-step graph statistics.
    """

    if hardware_id == "nvidia_a800":
        return True
    if hardware_id == "nvidia_h200":
        return runtime["runtime_version"] == "0.11.0"
    if hardware_id in ("metax_c500", "ascend_910b"):
        return False
    raise ValueError(f"unsupported Scheduler trace hardware: {hardware_id!r}")


def _validate_devices(
    devices: Sequence[int], runtime: dict[str, Any]
) -> tuple[int, ...]:
    if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes)):
        raise TypeError("devices must be a sequence of integer device indices")
    normalized = tuple(devices)
    if not normalized:
        raise ValueError("at least one device must be selected")
    if any(isinstance(device, bool) or not isinstance(device, int) for device in normalized):
        raise TypeError("device indices must be integers")
    if any(device < 0 for device in normalized):
        raise ValueError("device indices must be non-negative")
    if len(set(normalized)) != len(normalized):
        raise ValueError("device indices must be unique")
    expected = (
        runtime["tensor_parallel_size"] * runtime["pipeline_parallel_size"]
    )
    if len(normalized) != expected:
        raise ValueError(
            "device count must equal tensor_parallel_size * "
            f"pipeline_parallel_size: expected {expected}, got {len(normalized)}"
        )
    return normalized


def _validate_no_profiler_environment() -> None:
    diagnostic_nsys = (
        os.environ["MIDLAYER_A800_NSYS_STEP_NVTX"]
        if "MIDLAYER_A800_NSYS_STEP_NVTX" in os.environ else None)
    if diagnostic_nsys not in (None, "1"):
        raise ValueError(
            "MIDLAYER_A800_NSYS_STEP_NVTX must be exactly '1' when set"
        )
    nsys_injection_keys = {"CUDA_INJECTION64_PATH", "LD_PRELOAD"}
    c500_graph_launch_context = (
        os.environ["MIDLAYER_C500_GRAPH_LAUNCH_CONTEXT"]
        if "MIDLAYER_C500_GRAPH_LAUNCH_CONTEXT" in os.environ else None
    )
    c500_graph_launch_log = (
        os.environ["MIDLAYER_C500_GRAPH_LAUNCH_LOG"]
        if "MIDLAYER_C500_GRAPH_LAUNCH_LOG" in os.environ else None
    )
    c500_graph_launch_library = (
        os.environ["MIDLAYER_C500_GRAPH_LAUNCH_INTERPOSER"]
        if "MIDLAYER_C500_GRAPH_LAUNCH_INTERPOSER" in os.environ else None
    )
    c500_graph_launch_injection = (
        c500_graph_launch_context is not None
        or c500_graph_launch_log is not None
        or c500_graph_launch_library is not None
    )
    if c500_graph_launch_injection:
        if (
            not c500_graph_launch_context
            or not c500_graph_launch_log
            or not c500_graph_launch_library
        ):
            raise ValueError("C500 Graph launch diagnostic environment is incomplete")
        if not all(
            value.startswith("/")
            for value in (
                c500_graph_launch_context,
                c500_graph_launch_log,
                c500_graph_launch_library,
            )
        ):
            raise ValueError("C500 Graph launch diagnostic paths must be absolute")
        if ("LD_PRELOAD" not in os.environ
                or os.environ["LD_PRELOAD"] != c500_graph_launch_library):
            raise ValueError(
                "C500 Graph launch diagnostic LD_PRELOAD must name its declared interposer"
            )
    active = [
        key
        for key in PROFILER_ENVIRONMENT_KEYS
        if key in os.environ and os.environ[key]
        and not (diagnostic_nsys == "1" and key in nsys_injection_keys)
        and not (c500_graph_launch_injection and key == "LD_PRELOAD")
    ]
    if active:
        raise RuntimeError(
            "production ground-truth run refuses profiler/injection "
            f"environment variables: {active}"
        )


def _assert_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as error:
            raise RuntimeError(
                f"refusing to reuse occupied server endpoint {host}:{port}"
            ) from error


def _wait_for_server(
    process: subprocess.Popen,
    health_url: str,
    *,
    expected_model_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    last_probe_error = "server has not accepted a readiness connection"
    while True:
        attempts += 1
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"vLLM server exited before readiness with code {returncode}"
            )
        try:
            with urllib_request.urlopen(health_url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    payload = json.loads(response.read())
                    model_ids = [entry["id"] for entry in payload["data"]]
                    if model_ids == [expected_model_id]:
                        if process.poll() is not None:
                            raise RuntimeError(
                                "owned vLLM server exited during readiness"
                            )
                        return {
                            "attempts": attempts,
                            "elapsed_s": time.monotonic() - started,
                            "model_ids": model_ids,
                        }
                    last_probe_error = (
                        "server model IDs disagree: "
                        f"{model_ids!r} != {[expected_model_id]!r}"
                    )
        except (OSError, urllib_error.URLError, json.JSONDecodeError) as error:
            last_probe_error = f"{type(error).__name__}: {error}"
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"vLLM server readiness timed out after {elapsed:.1f}s; "
                f"last probe error: {last_probe_error}"
            )
        time.sleep(poll_interval_seconds)


# The markers each backend's vLLM release prints when it really captured the
# production CUDA/ACL graph.  Selection is data.
#
# ``single_runtime`` is the marker set for a backend with one release, which
# is selected without consulting the engine generation.
#
# ``by_generation`` is the marker set per engine generation for a backend
# with more than one; an unlisted generation is a caller error.  The message
# differs per backend because the hand-written chain produced two of them:
# MetaX named the generation, while a generation the A800/H200 chain did not
# recognise fell through to the unsupported-hardware message.
_GRAPH_LOG_MARKERS = {
    "nvidia_a800": {
        "single_runtime": (
            "Initializing a V1 LLM engine",
            "torch.compile takes",
            "Graph capturing finished",
        ),
        "by_generation": None,
        "by_generation_error": None,
        "runtime_replay_marker": None,
    },
    "nvidia_h200": {
        "single_runtime": None,
        "by_generation": {
            "v1_0.11.0": (
                "Initializing a V1 LLM engine",
                "torch.compile takes",
                "Graph capturing finished",
            ),
            "v1_0.30.0": (
                "Initializing a V1 LLM engine",
                "torch.compile took",
                "Graph capturing finished",
            ),
        },
        # An unrecognised H200 generation fell through the
        # ``elif hardware_id == "nvidia_h200"`` branch to the generic error.
        "by_generation_error": (
            "unsupported graph-log hardware_id: {hardware_id!r}"),
        "runtime_replay_marker": None,
    },
    "metax_c500": {
        "single_runtime": None,
        "by_generation": {
            "v0": (
                "Capturing CUDA graph shapes",
                "Graph capturing finished",
            ),
            "v1": (
                "Initializing a V1 LLM engine",
                "Capturing CUDA graphs (",
                "Graph capturing finished",
            ),
        },
        "by_generation_error": (
            "MetaX graph-log validation requires a supported "
            "engine_generation, got {engine_generation!r}"),
        "runtime_replay_marker": None,
    },
    "ascend_910b": {
        "single_runtime": (
            "FULL_DECODE_ONLY compilation enabled on NPU",
        ),
        "by_generation": None,
        "by_generation_error": None,
        # The NPU backend replays its aclgraph; the CUDA ones do not.
        "runtime_replay_marker": "Replaying aclgraph",
    },
}


def _graph_log_markers(
    hardware_id: str, engine_generation: str | None, require_runtime_replay: bool,
) -> tuple[str, ...]:
    """Return the markers ``hardware_id``/``engine_generation`` must print."""

    if hardware_id not in _GRAPH_LOG_MARKERS:
        raise ValueError(
            f"unsupported graph-log hardware_id: {hardware_id!r}"
        )
    scope = _GRAPH_LOG_MARKERS[hardware_id]
    single_runtime = scope["single_runtime"]
    if single_runtime is not None:
        required_markers = single_runtime
    else:
        by_generation = scope["by_generation"]
        if engine_generation not in by_generation:
            raise ValueError(scope["by_generation_error"].format(
                hardware_id=hardware_id,
                engine_generation=engine_generation,
            ))
        required_markers = by_generation[engine_generation]
    if require_runtime_replay:
        replay_marker = scope["runtime_replay_marker"]
        if replay_marker is not None:
            required_markers += (replay_marker,)
    return required_markers


def _validate_graph_server_logs(
    stdout_path: Path,
    stderr_path: Path,
    *,
    hardware_id: str,
    engine_generation: str | None = None,
    require_runtime_replay: bool = False,
) -> dict[str, Any]:
    text = stdout_path.read_text(errors="replace") + "\n" + stderr_path.read_text(
        errors="replace"
    )
    required_markers = _graph_log_markers(
        hardware_id, engine_generation, require_runtime_replay)
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise RuntimeError(
            "vLLM server logs do not prove production Graph execution; "
            f"missing markers: {missing}"
        )
    return {
        "status": "validated",
        "required_markers": list(required_markers),
    }


def _validate_a800_async_scheduling_logs(
    stdout_path: Path, stderr_path: Path,
) -> dict[str, Any]:
    text = stdout_path.read_text(errors="replace") + "\n" + stderr_path.read_text(
        errors="replace"
    )
    required_markers = (
        "'async_scheduling': True",
        "mp-based distributed executor backend for async scheduling",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "Batch queue is enabled with size 2",
    )
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise RuntimeError(
        "vLLM logs do not prove the A800 async Scheduler runtime: "
            f"missing markers: {missing}"
        )
    return {
        "status": "validated",
        "required_markers": list(required_markers),
    }


def _requires_a800_async_scheduling_validation(
        hardware_id: str, async_scheduling: bool) -> bool:
    """Return whether the A800-specific executor-log contract applies."""

    if not isinstance(hardware_id, str):
        raise TypeError("hardware_id must be a string")
    if not isinstance(async_scheduling, bool):
        raise TypeError("async_scheduling must be a bool")
    return hardware_id == "nvidia_a800" and async_scheduling


def _graph_log_engine_generation(server_plan: dict[str, Any]) -> str:
    """Graph-log marker key; H200 markers differ between its two runtimes."""

    engine_generation = server_plan["engine"]["vllm_engine_generation"]
    if server_plan["hardware_id"] != "nvidia_h200":
        return engine_generation
    return f"{engine_generation}_{server_plan['engine']['runtime_version']}"


def _live_process_group_member_pids(process_group_id: int) -> list[int]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise RuntimeError("process-group quiescence requires Linux /proc")
    live_member_pids = []
    for process_path in proc_root.iterdir():
        if not process_path.name.isdecimal():
            continue
        stat_path = process_path / "stat"
        try:
            stat = stat_path.read_bytes()
        except FileNotFoundError:
            stat = None
        except OSError as error:
            raise RuntimeError(
                f"failed to inspect process state at {stat_path}"
            ) from error
        if stat is None:
            continue
        command_end = stat.rfind(b")")
        if command_end < 0:
            raise RuntimeError(
                f"invalid Linux process stat record at {stat_path}"
            )
        fields = stat[command_end + 1 :].split()
        if len(fields) < 3:
            raise RuntimeError(
                f"invalid Linux process stat record at {stat_path}"
            )
        try:
            member_process_group_id = int(fields[2])
        except ValueError as error:
            raise RuntimeError(f"invalid process group in {stat_path}") from error
        process_state = fields[0]
        if (
            member_process_group_id == process_group_id
            and process_state not in (b"Z", b"X", b"x")
        ):
            live_member_pids.append(int(process_path.name))
    return sorted(live_member_pids)


def _wait_for_process_group_quiescence(
    process_group_id: int,
    *,
    timeout_seconds: float,
) -> list[int]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        live_member_pids = _live_process_group_member_pids(process_group_id)
        if not live_member_pids:
            return []
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            return live_member_pids
        time.sleep(
            min(_PROCESS_GROUP_POLL_INTERVAL_SECONDS, remaining_seconds)
        )


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        if _live_process_group_member_pids(process_group_id):
            raise RuntimeError(
                "process group disappeared during signaling but still has live members: "
                f"pgid={process_group_id}"
            )


def _terminate_process_group(process_group_id: int) -> None:
    if _live_process_group_member_pids(process_group_id):
        _signal_process_group(process_group_id, signal.SIGTERM)
        live_member_pids = _wait_for_process_group_quiescence(
            process_group_id,
            timeout_seconds=_SERVER_TERMINATION_TIMEOUT_SECONDS,
        )
        if live_member_pids:
            _signal_process_group(process_group_id, signal.SIGKILL)
            live_member_pids = _wait_for_process_group_quiescence(
                process_group_id,
                timeout_seconds=_SERVER_TERMINATION_TIMEOUT_SECONDS,
            )
            if live_member_pids:
                raise RuntimeError(
                    "vLLM server process group remained live after SIGKILL: "
                    f"pgid={process_group_id}, pids={live_member_pids}"
                )


def _terminate_server(process: subprocess.Popen) -> int:
    process_group_id = process.pid
    _terminate_process_group(process_group_id)
    if process.returncode is None:
        try:
            process.wait(timeout=_SERVER_TERMINATION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "vLLM server process group is quiescent but its leader did not "
                f"produce a return code: pid={process.pid}"
            ) from error
    if process.returncode is None:
        raise RuntimeError("vLLM server termination did not produce a return code")
    return process.returncode


def _preflight_scheduler_trace(
    server_plan: dict[str, Any], server_environment: dict[str, str]
) -> None:
    if not server_plan["scheduler_trace"]["enabled"]:
        return
    completed = subprocess.run(
        [
            server_plan["python_executable"],
            "-c",
            (
                "from dataset_pipeline.serving_dataset.vllm_scheduler_trace "
                "import install_from_environment; "
                "assert install_from_environment()"
            ),
        ],
        cwd=str(ROOT),
        env=server_environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0 or "Error in sitecustomize" in completed.stderr:
        raise RuntimeError(
            "Scheduler trace instrumentation preflight failed:\n"
            + completed.stderr.strip()
        )


def _vllm_server_process_group_for_port(
    port: int,
    *,
    proc_root: Path = Path("/proc"),
) -> int:
    """Resolve one live vLLM serve process group for an exclusive port.

    Nsight's fork-before-exec mode detaches the target server from the profiler
    launcher group. The runner reserves its port before launch, so the port is
    a fail-closed ownership key for both post-run worker signals and cleanup.
    """

    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise ValueError("vLLM server port must be between 1 and 65535")
    if not proc_root.is_dir():
        raise RuntimeError("vLLM server process discovery requires Linux /proc")
    expected_port = str(port)
    candidates: list[tuple[int, int]] = []
    for process_path in proc_root.iterdir():
        if not process_path.name.isdecimal():
            continue
        try:
            cmdline = (process_path / "cmdline").read_bytes().split(b"\0")
            stat = (process_path / "stat").read_bytes()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError(
                f"failed to inspect vLLM process at {process_path}"
            ) from error
        argv = [argument.decode("utf-8", errors="surrogateescape")
                for argument in cmdline if argument]
        if (
            not argv
            or not Path(argv[0]).name.startswith("python")
            or "vllm.entrypoints.cli.main" not in argv
            or "serve" not in argv
            or "--port" not in argv
        ):
            continue
        port_index = argv.index("--port")
        if port_index + 1 >= len(argv) or argv[port_index + 1] != expected_port:
            continue
        command_end = stat.rfind(b")")
        if command_end < 0:
            raise RuntimeError(f"invalid Linux process stat record at {process_path}")
        fields = stat[command_end + 1 :].split()
        if len(fields) < 3:
            raise RuntimeError(f"invalid process group in {process_path}")
        if fields[0] in (b"Z", b"X", b"x"):
            continue
        try:
            process_group_id = int(fields[2])
        except ValueError as error:
            raise RuntimeError(
                f"invalid process group in {process_path}"
            ) from error
        candidates.append((int(process_path.name), process_group_id))
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one live vLLM server for reserved port "
            f"{port}, found {candidates!r}"
        )
    return candidates[0][1]


def _wait_for_scheduler_trace_output(trace_path: str | Path) -> None:
    """Wait for the EngineCore's post-benchmark atomic trace publication."""

    path = Path(trace_path)
    deadline = time.monotonic() + 30.0
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Scheduler trace was not published within 30 seconds after "
                f"the formal benchmark: {path}"
            )
        time.sleep(0.05)
