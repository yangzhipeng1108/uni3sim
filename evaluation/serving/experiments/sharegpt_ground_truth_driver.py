"""CLI driver for ShareGPT ground-truth capture: argument parsing,
the capture run, and result writing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from evaluation.serving.experiments.ground_truth_case_contract import (
    ROOT,
    REQUEST_TEMPLATE_HARDWARE_BY_TARGET,
    H200_RUNTIME_VERSIONS,
    sha256_file,
    _retarget_case_contract,
    _load_case_contract,
)
from evaluation.serving.experiments.ground_truth_server_lifecycle import (
    _server_plan,
    _validate_no_profiler_environment,
    _assert_port_available,
    _wait_for_server,
    _validate_graph_server_logs,
    _validate_a800_async_scheduling_logs,
    _requires_a800_async_scheduling_validation,
    _graph_log_engine_generation,
    _terminate_process_group,
    _terminate_server,
    _preflight_scheduler_trace,
    _vllm_server_process_group_for_port,
    _wait_for_scheduler_trace_output,
)
from evaluation.serving.experiments.ground_truth_http_benchmark import (
    build_ground_truth,
)
from evaluation.serving.experiments.ground_truth_scheduler_trace import (
    _scheduler_trace_summary,
    _validate_benchmark_log_window,
    _capture_a800_post_benchmark_counters,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a canonical ShareGPT schedule against vLLM"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--suite-manifest", required=True)
    parser.add_argument("--devices", required=True, nargs="+", type=int)
    parser.add_argument("--port", default=18080, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.8, type=float)
    parser.add_argument("--max-model-len", default=8192, type=int)
    parser.add_argument("--max-num-batched-tokens", default=8192, type=int)
    parser.add_argument("--max-num-seqs", default=256, type=int)
    parser.add_argument("--kv-block-size-tokens", required=True, type=int)
    parser.add_argument("--max-dispatch-lag-ms", default=100.0, type=float)
    parser.add_argument(
        "--arrival-probe-per-1000-tokens-ms",
        default=0.0,
        type=float,
        help=(
            "controlled experiment only: delay each request's submission by "
            "this many ms per 1000 prompt tokens, which perturbs the relative "
            "spacing between arrivals; zero, the default, replays the "
            "canonical schedule unchanged"),
    )
    parser.add_argument("--server-wait-seconds", default=600.0, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scheduler-trace-output")
    parser.add_argument("--scheduler-request-add-output")
    parser.add_argument(
        "--diagnostic-request-count", type=int,
        help="replay this many leading canonical requests as a diagnostic run",
    )
    parser.add_argument(
        "--diagnostic-output-token-cap", type=int,
        help="cap each diagnostic request at 4–8 output tokens; requires a diagnostic prefix",
    )
    parser.add_argument("--a800-compilation-counter-output")
    parser.add_argument(
        "--target-hardware-id",
        choices=tuple(REQUEST_TEMPLATE_HARDWARE_BY_TARGET),
        help=(
            "run on this hardware while replaying the --case-manifest request "
            "file and arrival schedule of its template hardware"),
    )
    parser.add_argument(
        "--runtime-version",
        choices=H200_RUNTIME_VERSIONS,
        help="installed vLLM release; required with --target-hardware-id",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.target_hardware_id is None) != (args.runtime_version is None):
        raise ValueError(
            "--target-hardware-id and --runtime-version must be given together")
    if any(device < 0 for device in args.devices):
        raise ValueError("devices must be non-negative")
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    for name in (
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "kv_block_size_tokens",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_dispatch_lag_ms <= 0:
        raise ValueError("max_dispatch_lag_ms must be positive")
    if args.server_wait_seconds <= 0:
        raise ValueError("server_wait_seconds must be positive")
    if args.diagnostic_request_count is not None and args.diagnostic_request_count < 8:
        raise ValueError("diagnostic request count must be at least eight for warmup")
    if args.diagnostic_output_token_cap is not None:
        if args.diagnostic_request_count is None:
            raise ValueError("diagnostic output cap requires diagnostic request count")
        if not 4 <= args.diagnostic_output_token_cap <= 8:
            raise ValueError("diagnostic output cap must be between four and eight")
    output_path = Path(args.output)
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing output: {output_path}")
    if args.scheduler_trace_output is not None:
        scheduler_trace_path = Path(args.scheduler_trace_output)
        if scheduler_trace_path.exists():
            raise ValueError(
                "refusing to overwrite existing Scheduler trace: "
                f"{scheduler_trace_path}"
            )
        if scheduler_trace_path.resolve() == output_path.resolve():
            raise ValueError("Scheduler trace output must differ from result output")
    if args.scheduler_request_add_output is not None:
        if args.scheduler_trace_output is None:
            raise ValueError(
                "--scheduler-request-add-output requires --scheduler-trace-output"
            )
        request_add_path = Path(args.scheduler_request_add_output)
        if request_add_path.exists():
            raise ValueError(
                "refusing to overwrite Scheduler request-add output: "
                f"{request_add_path}"
            )
        stage_sidecars = tuple(
            request_add_path.parent.glob(
                f"{request_add_path.name}.stages.*.jsonl"
            )
        )
        if stage_sidecars:
            raise ValueError(
                "refusing to overwrite Scheduler request-stage sidecars: "
                + ", ".join(str(path) for path in stage_sidecars)
            )
        if request_add_path.resolve() == output_path.resolve():
            raise ValueError(
                "Scheduler request-add output must differ from result output"
            )
    if args.a800_compilation_counter_output is not None:
        counter_output_path = Path(args.a800_compilation_counter_output)
        if counter_output_path.exists():
            raise ValueError(
                "refusing to overwrite existing A800 compilation-counter output: "
                f"{counter_output_path}"
            )
        if counter_output_path.resolve() == output_path.resolve():
            raise ValueError(
                "A800 compilation-counter output must differ from result output"
            )
        if args.scheduler_trace_output is not None:
            raise ValueError(
                "A800 counter-only instrumentation cannot accompany Scheduler trace"
            )
    return args


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _metric_summary(result: dict[str, Any]) -> dict[str, Any] | None:
    metrics = result["metrics"]
    if metrics is None:
        return None
    return {
        key: metrics[key]
        for key in (
            "duration_s",
            "completed_requests",
            "request_throughput_per_s",
            "output_token_throughput_per_s",
            "max_dispatch_lag_ms",
            "ttft",
            "tpot",
            "e2e",
        )
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    case = _load_case_contract(
        args.case_manifest,
        model_path=args.model,
        suite_manifest_path=args.suite_manifest,
    )
    if args.target_hardware_id is not None:
        case = _retarget_case_contract(
            case,
            target_hardware_id=args.target_hardware_id,
            target_runtime_version=args.runtime_version,
        )
    if (args.diagnostic_request_count is not None
            and args.diagnostic_request_count >= case["requests_row_count"]):
        raise ValueError("diagnostic request count must be below canonical count")
    if (args.diagnostic_request_count is not None
            and case["identity"]["hardware_id"] != "nvidia_a800"):
        raise ValueError("diagnostic request prefix currently supports A800 only")
    server_plan = _server_plan(args, case)
    if (args.diagnostic_request_count is not None
            and server_plan["scheduler_trace"]["enabled"]):
        # The plugin closes its trace after this many formal requests.  The
        # canonical file identity remains bound to the full request stream.
        server_plan["environment"]["MIDLAYER_SCHEDULER_TRACE_REQUEST_COUNT"] = (
            str(args.diagnostic_request_count))
    result = build_ground_truth(
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        case=case,
        server_plan=server_plan,
        dry_run=True,
        diagnostic_request_count=args.diagnostic_request_count,
        diagnostic_output_token_cap=args.diagnostic_output_token_cap,
    )
    output_path = Path(args.output)
    if args.dry_run:
        _write_result(output_path, result)
        return result

    _validate_no_profiler_environment()
    _assert_port_available("127.0.0.1", args.port)
    logs_dir = output_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "vllm_server.stdout.log"
    stderr_path = logs_dir / "vllm_server.stderr.log"
    server_environment = dict(os.environ)
    server_environment.update(server_plan["environment"])
    _preflight_scheduler_trace(server_plan, server_environment)
    process = None
    server_process_group_id = None
    stdout_handle = stdout_path.open("w")
    stderr_handle = stderr_path.open("w")
    result["status"] = "starting_server"
    result["server"]["runtime"] = {
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    _write_result(output_path, result)
    try:
        process = subprocess.Popen(
            server_plan["argv"],
            cwd=str(ROOT),
            env=server_environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        result["server"]["runtime"]["pid"] = process.pid
        result["server"]["runtime"]["readiness"] = _wait_for_server(
            process,
            server_plan["endpoint"]["health_url"],
            expected_model_id=case["identity"]["model_id"],
            timeout_seconds=args.server_wait_seconds,
        )
        server_process_group_id = _vllm_server_process_group_for_port(args.port)
        result["server"]["runtime"]["server_process_group_id"] = (
            server_process_group_id
        )
        stdout_handle.flush()
        stderr_handle.flush()
        result["server"]["runtime"]["graph_validation"] = (
            _validate_graph_server_logs(
                stdout_path,
                stderr_path,
                hardware_id=server_plan["hardware_id"],
                engine_generation=_graph_log_engine_generation(server_plan),
            )
        )
        if _requires_a800_async_scheduling_validation(
                server_plan["hardware_id"],
                server_plan["engine"]["async_scheduling"]) or (
                    server_plan["hardware_id"] == "nvidia_h200"
                    and server_plan["engine"]["runtime_version"] == "0.11.0"):
            # H200 0.11.0 runs the A800 runtime release; its executor logs
            # carry the same async-scheduling markers.
            result["server"]["runtime"]["async_scheduling_validation"] = (
                _validate_a800_async_scheduling_logs(
                    stdout_path,
                    stderr_path,
                )
            )
        _write_result(output_path, result)
        benchmark_log_offsets: list[tuple[Path, int]] = []

        def mark_benchmark_boundary() -> None:
            stdout_handle.flush()
            stderr_handle.flush()
            benchmark_log_offsets.extend((
                (stdout_path, stdout_path.stat().st_size),
                (stderr_path, stderr_path.stat().st_size),
            ))

        measured = build_ground_truth(
            model=args.model,
            dataset=args.dataset,
            seed=args.seed,
            case=case,
            server_plan=server_plan,
            dry_run=False,
            diagnostic_request_count=args.diagnostic_request_count,
            diagnostic_output_token_cap=args.diagnostic_output_token_cap,
            benchmark_boundary_callback=mark_benchmark_boundary,
        )
        measured["server"]["runtime"] = result["server"]["runtime"]
        result = measured
        stdout_handle.flush()
        stderr_handle.flush()
        if len(benchmark_log_offsets) != 2:
            raise RuntimeError("formal benchmark log boundary was not recorded")
        result["server"]["runtime"]["benchmark_log_window"] = (
            _validate_benchmark_log_window(benchmark_log_offsets)
        )
        scheduler_trace = server_plan["scheduler_trace"]
        counter_instrumentation = server_plan[
            "compilation_counter_instrumentation"
        ]
        if counter_instrumentation["enabled"]:
            result["server"]["runtime"]["compilation_counters"] = (
                _capture_a800_post_benchmark_counters(
                    server_process_group_id,
                    counter_instrumentation,
                )
            )
        if scheduler_trace["enabled"]:
            if server_plan["hardware_id"] == "metax_c500":
                max_result_consumption_gap_ns = 1_000_000_000
            elif server_plan["hardware_id"] in (
                "nvidia_a800", "nvidia_h200", "ascend_910b"
            ):
                max_result_consumption_gap_ns = None
            else:
                raise ValueError(
                    "unsupported Scheduler trace hardware: "
                    f"{server_plan['hardware_id']!r}"
                )
            _wait_for_scheduler_trace_output(scheduler_trace["output_path"])
            result["scheduler_trace"] = _scheduler_trace_summary(
                scheduler_trace["output_path"],
                (args.diagnostic_request_count if args.diagnostic_request_count is not None
                 else case["requests_row_count"]),
                max_result_consumption_gap_ns=max_result_consumption_gap_ns,
            )
            benchmark_clock = result["metrics"]["benchmark_clock"]
            scheduler_clock = result["scheduler_trace"]["clock"]
            if (
                benchmark_clock["clock_domain"]
                != scheduler_clock["clock_domain"]
                or benchmark_clock["host_boot_id"]
                != scheduler_clock["host_boot_id"]
            ):
                raise RuntimeError(
                    "benchmark and Scheduler trace do not share one host clock"
                )
            result["server"]["runtime"]["scheduler_pipeline_depth"] = (
                result["scheduler_trace"]["scheduler_pipeline_depth"]
            )
            result["server"]["runtime"][
                "scheduler_pipeline_depth_evidence"
            ] = result["scheduler_trace"]["scheduler_pipeline_depth_evidence"]
            if (
                server_plan["hardware_id"] == "metax_c500"
                and result["scheduler_trace"]["max_in_flight_step_count"] != 2
            ):
                raise RuntimeError(
                    "C500 formal benchmark never reached async depth=2"
                )
        else:
            result["scheduler_trace"] = None
        if server_plan["hardware_id"] == "ascend_910b":
            stdout_handle.flush()
            stderr_handle.flush()
            result["server"]["runtime"]["graph_validation"] = (
                _validate_graph_server_logs(
                    stdout_path,
                    stderr_path,
                    hardware_id=server_plan["hardware_id"],
                    engine_generation=server_plan["engine"][
                        "vllm_engine_generation"
                    ],
                    require_runtime_replay=True,
                )
            )
        max_dispatch_lag_ms = result["metrics"]["max_dispatch_lag_ms"]
        if max_dispatch_lag_ms > server_plan["max_dispatch_lag_ms"]:
            raise RuntimeError(
                "canonical arrival replay exceeded dispatch-lag limit: "
                f"{max_dispatch_lag_ms:.3f} ms > "
                f"{server_plan['max_dispatch_lag_ms']:.3f} ms"
            )
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        if process is not None:
            exited_before_cleanup = process.poll() is not None
            if (
                server_process_group_id is not None
                and server_process_group_id != process.pid
            ):
                _terminate_process_group(server_process_group_id)
            termination_returncode = _terminate_server(process)
            result["server"]["runtime"]["termination_returncode"] = (
                termination_returncode
            )
            result["server"]["runtime"]["exited_before_cleanup"] = (
                exited_before_cleanup
            )
            if exited_before_cleanup and result["status"] == "success":
                result["status"] = "failed"
                result["failure"] = {
                    "type": "ServerExitedEarly",
                    "message": (
                        "vLLM server exited before runner-managed cleanup with "
                        f"code {termination_returncode}"
                    ),
                }
        stdout_handle.close()
        stderr_handle.close()
        if stdout_path.exists():
            result["server"]["runtime"]["stdout_sha256"] = sha256_file(
                stdout_path
            )
        if stderr_path.exists():
            result["server"]["runtime"]["stderr_sha256"] = sha256_file(
                stderr_path
            )
        _write_result(output_path, result)
    if result["status"] != "success":
        raise RuntimeError(result["failure"]["message"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = _run(args)
    print(json.dumps({
        "status": result["status"],
        "output": args.output,
        "metrics": _metric_summary(result),
    }, indent=2, sort_keys=True))
    return 0
