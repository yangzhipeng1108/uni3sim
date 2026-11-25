"""Replay one canonical ShareGPT request schedule against a vLLM server.

This module is the compatibility re-export layer for the historical
import surface; the implementation lives in the sibling modules of
this package.
"""

from evaluation.serving.experiments.experiment_plan import git_code_identity

from src.serving.request_gen import load_canonical_requests

from src.serving.request_gen.request_stream import request_stream_sha256

from evaluation.serving.experiments.ground_truth_case_contract import (
    ROOT,
    SUPPORTED_HARDWARE_IDS,
    REQUEST_TEMPLATE_HARDWARE_BY_TARGET,
    H200_RUNTIME_VERSIONS,
    sha256_file,
    _installed_vllm_release,
    _vllm_source_identity,
    _retarget_case_contract,
    _PRODUCTION_DTYPES,
    _load_case_contract,
)

from evaluation.serving.experiments.ground_truth_server_lifecycle import (
    PROFILER_ENVIRONMENT_KEYS,
    GROUND_TRUTH_SCHEDULER_POLICY,
    EXECUTION_ENVIRONMENT_PREFIXES,
    _ASCEND_STEP_TIMESTAMP_ENV,
    _ASCEND_MSPROFTX_STEP_MARK_ENV,
    C500_DEVICE_EVENT_ENVIRONMENT_KEYS,
    _SERVER_TERMINATION_TIMEOUT_SECONDS,
    _PROCESS_GROUP_POLL_INTERVAL_SECONDS,
    _SCHEDULER_TRACE_BOOTSTRAP,
    _c500_device_event_environment,
    _CPU_OFFLOAD_GB,
    _LAUNCH_RECORD_DEFAULTS,
    _SERVER_LAUNCH_RECORDS,
    _launch_record,
    _server_plan,
    _uses_worker_dispatch_trace,
    _validate_devices,
    _validate_no_profiler_environment,
    _assert_port_available,
    _wait_for_server,
    _GRAPH_LOG_MARKERS,
    _graph_log_markers,
    _validate_graph_server_logs,
    _validate_a800_async_scheduling_logs,
    _requires_a800_async_scheduling_validation,
    _graph_log_engine_generation,
    _live_process_group_member_pids,
    _wait_for_process_group_quiescence,
    _signal_process_group,
    _terminate_process_group,
    _terminate_server,
    _preflight_scheduler_trace,
    _vllm_server_process_group_for_port,
    _wait_for_scheduler_trace_output,
)

from evaluation.serving.experiments.ground_truth_http_benchmark import (
    _sample_sharegpt,
    _prefix_cache_reuse_evidence,
    _percentile,
    _distribution,
    CLIENT_FD_TABLE_RESERVE,
    _reserve_client_fd_table,
    _run_http_benchmark,
    build_ground_truth,
)

from evaluation.serving.experiments.ground_truth_scheduler_trace import (
    _A800_COMPILATION_COUNTER_FIELDS,
    _scheduler_runtime_depth_metadata,
    _scheduler_clock_metadata,
    _scheduler_trace_summary,
    _VLLM_011_MULTIPROC_EXECUTOR_CLASS,
    _worker_dispatch_reply_rank,
    _a800_scheduler_trace_summary,
    _validate_a800_rank_observation_for_summary,
    _validate_a800_graph_key,
    _validate_benchmark_log_window,
    _capture_a800_post_benchmark_counters,
    _validate_a800_compilation_counters,
)

from evaluation.serving.experiments.sharegpt_ground_truth_driver import (
    _parse_args,
    _write_result,
    _metric_summary,
    _run,
    main,
)
