"""Strict device-Event sampling for kernel-golden evaluation runners."""

from __future__ import annotations

from evaluation.json_shape import file_sha256 as _file_sha256

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping


LEGACY_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION = 2
CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION = 3
BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION = 4
LEGACY_GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION = 1
SYNCHRONIZED_SWEEP_GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION = 2
GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION = 3
DIFFERENTIAL_GRAPH_REPLAY_EVENT_MODE = (
    "paired_differential_events_around_cuda_graph_replay")
DIFFERENTIAL_NPU_GRAPH_REPLAY_EVENT_MODE = (
    "paired_differential_events_around_npu_graph_replay")
EVENT_CASE_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KERNEL_GOLDEN_OUTPUT_ROOT = REPOSITORY_ROOT / "experiment_logs/kernel_golden"


@dataclass(frozen=True)
class EventRunOptions:
    output: Path
    case_id: str
    baseline_commit: str
    source_snapshot_sha256: str
    timer_sanity_sha256: str
    canonical_attributes_sha256: str
    warmup_iterations: int
    sample_count: int
    cache_eviction_mib: int
    telemetry_gpu_index: int | None = None


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def prepare_bf16_cache_eviction_sweep(
        *, torch_module: Any, device: str, cache_eviction_mib: int,
        sweep_operation: str = "add",
        ) -> tuple[Callable[[], Any] | None, int]:
    """Allocate one retained contiguous BF16 buffer and sweep every element.

    ``add`` is the established default used by device-Event measurements.
    Timeline profiler captures may request ``fill`` when the target is itself
    an elementwise Add or Mul: the cold-cache sweep must not share the target
    runtime symbol, otherwise a name-filtered Timeline summary can mix the
    sweep duration into the target-kernel duration.
    """

    if isinstance(cache_eviction_mib, bool) or not isinstance(
            cache_eviction_mib, int):
        raise TypeError("cache_eviction_mib must be an integer")
    if cache_eviction_mib < 0:
        raise ValueError("cache_eviction_mib must be nonnegative")
    if not isinstance(device, str) or not device:
        raise TypeError("device must be a non-empty string")
    if sweep_operation not in ("add", "fill"):
        raise ValueError(
            "cache eviction sweep_operation must be 'add' or 'fill'")
    if cache_eviction_mib == 0:
        return None, 0
    for name in ("bfloat16", "zeros"):
        if not hasattr(torch_module, name):
            raise TypeError(f"torch module has no {name}")
    requested_bytes = cache_eviction_mib * 1024 * 1024
    tensor = torch_module.zeros(
        requested_bytes // 2,
        dtype=torch_module.bfloat16,
        device=device,
    )
    actual_bytes = tensor.numel() * tensor.element_size()
    if actual_bytes != requested_bytes:
        raise RuntimeError(
            "BF16 cache eviction allocation size mismatch: "
            f"requested={requested_bytes}, actual={actual_bytes}")

    def sweep() -> Any:
        if sweep_operation == "add":
            return tensor.add_(1)
        if sweep_operation == "fill":
            return tensor.fill_(1)
        raise AssertionError(
            f"unhandled cache eviction sweep operation: {sweep_operation!r}")

    return sweep, actual_bytes


def validate_cache_eviction_mib(
        cache_eviction_mib: int, *, minimum_nonzero_mib: int,
        hardware_id: str) -> None:
    """Require an enabled hardware sweep to be at least eight L2 capacities."""

    for value, name in (
            (cache_eviction_mib, "cache_eviction_mib"),
            (minimum_nonzero_mib, "minimum_nonzero_mib")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if cache_eviction_mib < 0:
        raise ValueError("cache_eviction_mib must be nonnegative")
    if minimum_nonzero_mib <= 0:
        raise ValueError("minimum_nonzero_mib must be positive")
    if not isinstance(hardware_id, str) or not hardware_id:
        raise TypeError("hardware_id must be a non-empty string")
    if 0 < cache_eviction_mib < minimum_nonzero_mib:
        raise ValueError(
            f"{hardware_id} cache eviction must be 0 or at least "
            f"{minimum_nonzero_mib} MiB (8x L2)")


def _percentile(sorted_values: tuple[float, ...], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    position = percentile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def summarize_event_samples(samples_ns: list[float]) -> Mapping[str, float]:
    """Summarize independent positive device-Event samples in nanoseconds."""

    if not isinstance(samples_ns, list) or not samples_ns:
        raise TypeError("samples_ns must be a non-empty list")
    normalized = []
    for index, value in enumerate(samples_ns):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"event sample {index} must be numeric")
        sample = float(value)
        if not math.isfinite(sample) or sample <= 0.0:
            raise ValueError(
                f"event sample {index} must be finite and positive")
        normalized.append(sample)
    ordered = tuple(sorted(normalized))
    median_ns = statistics.median(ordered)
    standard_deviation_ns = statistics.pstdev(ordered)
    return {
        "median_ns": median_ns,
        "p10_ns": _percentile(ordered, 0.10),
        "p90_ns": _percentile(ordered, 0.90),
        "standard_deviation_ns": standard_deviation_ns,
        "coefficient_of_variation": standard_deviation_ns / median_ns,
    }


def add_event_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the closed set of optional Event-mode arguments to one runner."""

    parser.add_argument("--event-output")
    parser.add_argument("--event-case-id")
    parser.add_argument("--event-baseline-commit")
    parser.add_argument("--event-source-snapshot-sha256")
    parser.add_argument("--event-timer-sanity-sha256")
    parser.add_argument("--event-canonical-attributes-sha256")
    parser.add_argument("--event-warmup", type=int)
    parser.add_argument("--event-samples", type=int)
    parser.add_argument("--event-cache-eviction-mib", type=int, default=0)
    parser.add_argument("--event-telemetry-gpu", type=int)


def require_lower_hex(value: str, length: int, name: str) -> str:
    if (not isinstance(value, str) or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(
            f"{name} must be a lowercase {length}-digit hexadecimal string")
    return value


def parse_event_run_options(args: argparse.Namespace) -> EventRunOptions | None:
    """Normalize Event CLI options, rejecting incomplete mixed modes."""

    names = (
        "event_output", "event_case_id", "event_baseline_commit",
        "event_source_snapshot_sha256", "event_timer_sanity_sha256",
        "event_canonical_attributes_sha256", "event_warmup", "event_samples",
    )
    values = [getattr(args, name) for name in names]
    if all(value is None for value in values):
        if args.event_cache_eviction_mib != 0:
            raise ValueError(
                "event_cache_eviction_mib requires complete Event options")
        return None
    missing = [name for name, value in zip(names, values, strict=True)
               if value is None]
    if missing:
        raise ValueError(
            f"Event mode requires every Event option; missing={missing}")
    if not isinstance(args.event_case_id, str) or not args.event_case_id:
        raise ValueError("event_case_id must be a non-empty string")
    if args.event_cache_eviction_mib < 0:
        raise ValueError("event_cache_eviction_mib must be nonnegative")
    telemetry_gpu_index = args.event_telemetry_gpu
    if telemetry_gpu_index is not None:
        if telemetry_gpu_index < 0:
            raise ValueError("event_telemetry_gpu must be nonnegative")
    output = Path(args.event_output).resolve()
    try:
        output.relative_to(KERNEL_GOLDEN_OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            "Event output must stay under experiment_logs/kernel_golden: "
            f"{output}") from error
    return EventRunOptions(
        output=output,
        case_id=args.event_case_id,
        baseline_commit=require_lower_hex(
            args.event_baseline_commit, 40, "event_baseline_commit"),
        source_snapshot_sha256=require_lower_hex(
            args.event_source_snapshot_sha256, 64,
            "event_source_snapshot_sha256"),
        timer_sanity_sha256=require_lower_hex(
            args.event_timer_sanity_sha256, 64,
            "event_timer_sanity_sha256"),
        canonical_attributes_sha256=require_lower_hex(
            args.event_canonical_attributes_sha256, 64,
            "event_canonical_attributes_sha256"),
        warmup_iterations=_positive_integer(
            args.event_warmup, "event_warmup"),
        sample_count=_positive_integer(args.event_samples, "event_samples"),
        cache_eviction_mib=args.event_cache_eviction_mib,
        telemetry_gpu_index=telemetry_gpu_index,
    )


def build_nvidia_smi_telemetry_observer(
        physical_gpu_index: int) -> Callable[[], Mapping[str, Any]]:
    """Return a fail-closed A800 telemetry reader for retained Event samples."""

    if (isinstance(physical_gpu_index, bool)
            or not isinstance(physical_gpu_index, int)
            or physical_gpu_index < 0):
        raise ValueError("physical_gpu_index must be a nonnegative integer")
    fields = (
        "index", "clocks.sm", "clocks.mem", "pstate", "power.draw",
        "temperature.gpu", "clocks_event_reasons.active",
    )

    def observe() -> Mapping[str, Any]:
        completed = subprocess.run(
            (
                "nvidia-smi", f"--id={physical_gpu_index}",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "nvidia-smi telemetry query failed: "
                f"gpu={physical_gpu_index}, stderr={completed.stderr!r}")
        rows = [line.strip() for line in completed.stdout.splitlines()
                if line.strip()]
        if len(rows) != 1:
            raise RuntimeError(
                "nvidia-smi telemetry query must return exactly one row: "
                f"gpu={physical_gpu_index}, rows={rows!r}")
        values = [value.strip() for value in rows[0].split(",")]
        if len(values) != len(fields):
            raise RuntimeError(
                "nvidia-smi telemetry field count mismatch: "
                f"gpu={physical_gpu_index}, row={rows[0]!r}")
        index_text, sm_text, memory_text, pstate, power_text, temperature_text, throttle = values
        try:
            index = int(index_text)
            sm_mhz = int(sm_text)
            memory_mhz = int(memory_text)
            power_w = float(power_text)
            temperature_c = int(temperature_text)
        except ValueError as error:
            raise RuntimeError(
                "nvidia-smi telemetry has an invalid numeric field: "
                f"gpu={physical_gpu_index}, row={rows[0]!r}") from error
        if index != physical_gpu_index:
            raise RuntimeError(
                "nvidia-smi telemetry returned the wrong GPU: "
                f"expected={physical_gpu_index}, actual={index}")
        if (sm_mhz <= 0 or memory_mhz <= 0 or power_w < 0.0
                or temperature_c < 0 or not pstate or not throttle):
            raise RuntimeError(
                "nvidia-smi telemetry contains an invalid state: "
                f"gpu={physical_gpu_index}, row={rows[0]!r}")
        return {
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "physical_gpu_index": index,
            "sm_clock_mhz": sm_mhz,
            "memory_clock_mhz": memory_mhz,
            "pstate": pstate,
            "power_w": power_w,
            "temperature_c": temperature_c,
            "clocks_event_reasons_active": throttle,
        }

    return observe


def file_sha256(path: Path) -> str:
    return _file_sha256(path)


def write_kernel_golden_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a JSON artifact below the kernel-golden log root."""

    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("kernel-golden payload must be a non-empty mapping")
    output = path.resolve()
    try:
        output.relative_to(KERNEL_GOLDEN_OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            "kernel-golden output must stay under "
            f"experiment_logs/kernel_golden: {output}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent,
                prefix=".kernel-golden-", delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(
                dict(payload), stream, indent=2, ensure_ascii=False,
                allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def write_event_case_record(
        *, options: EventRunOptions, hardware_id: str, family: str,
        runner_path: Path, timer: Mapping[str, Any],
        stream_contract: Mapping[str, Any],
        allocation_contract: Mapping[str, Any], workload: Mapping[str, Any],
        environment: Mapping[str, Any], measurement: Mapping[str, Any]) -> Path:
    """Atomically write one measured case under the required log root."""

    for name, value in (
            ("hardware_id", hardware_id), ("family", family)):
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty string")
    for name, value in (
            ("timer", timer), ("stream_contract", stream_contract),
            ("allocation_contract", allocation_contract),
            ("workload", workload), ("environment", environment),
            ("measurement", measurement)):
        if not isinstance(value, Mapping) or not value:
            raise TypeError(f"{name} must be a non-empty mapping")
    resolved_runner = runner_path.resolve()
    if not resolved_runner.is_file():
        raise FileNotFoundError(f"runner does not exist: {resolved_runner}")
    payload = {
        "schema_version": EVENT_CASE_SCHEMA_VERSION,
        "status": "measured",
        "unsupported_reason": None,
        "hardware_id": hardware_id,
        "family": family,
        "case_id": options.case_id,
        "baseline_commit": options.baseline_commit,
        "source_snapshot_sha256": options.source_snapshot_sha256,
        "runner_path": str(resolved_runner.relative_to(REPOSITORY_ROOT)),
        "runner_sha256": file_sha256(resolved_runner),
        "timer_sanity_sha256": options.timer_sanity_sha256,
        "canonical_attributes_sha256": options.canonical_attributes_sha256,
        "timer": dict(timer),
        "stream_contract": dict(stream_contract),
        "allocation_contract": dict(allocation_contract),
        "workload": dict(workload),
        "environment": dict(environment),
        "measurement": dict(measurement),
    }
    return write_kernel_golden_json(options.output, payload)


def write_unsupported_event_case_record(
        *, options: EventRunOptions, hardware_id: str, family: str,
        runner_path: Path, reason: str, workload: Mapping[str, Any],
        environment: Mapping[str, Any]) -> Path:
    """Write one explicit unsupported result with normalized optional fields."""

    if not isinstance(reason, str) or not reason:
        raise TypeError("unsupported reason must be a non-empty string")
    for name, value in (
            ("hardware_id", hardware_id), ("family", family)):
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty string")
    for name, value in (("workload", workload), ("environment", environment)):
        if not isinstance(value, Mapping) or not value:
            raise TypeError(f"{name} must be a non-empty mapping")
    resolved_runner = runner_path.resolve()
    payload = {
        "schema_version": EVENT_CASE_SCHEMA_VERSION,
        "status": "unsupported",
        "unsupported_reason": reason,
        "hardware_id": hardware_id,
        "family": family,
        "case_id": options.case_id,
        "baseline_commit": options.baseline_commit,
        "source_snapshot_sha256": options.source_snapshot_sha256,
        "runner_path": str(resolved_runner.relative_to(REPOSITORY_ROOT)),
        "runner_sha256": file_sha256(resolved_runner),
        "timer_sanity_sha256": options.timer_sanity_sha256,
        "canonical_attributes_sha256": options.canonical_attributes_sha256,
        "timer": None,
        "stream_contract": None,
        "allocation_contract": None,
        "workload": dict(workload),
        "environment": dict(environment),
        "measurement": None,
    }
    return write_kernel_golden_json(options.output, payload)


def measure_current_stream_events(
        operation: Callable[[], Any], *, device_module: Any,
        warmup_iterations: int, sample_count: int,
        join_operation_streams: Callable[[], None] | None = None,
        cache_eviction_operation: Callable[[], Any] | None = None,
        cache_eviction_bytes: int = 0,
        pre_event_operation: Callable[[], Any] | None = None,
        operation_repetitions_per_sample: int = 1,
        telemetry_observer: Callable[[], Mapping[str, Any]] | None = None,
        ) -> Mapping[str, Any]:
    """Measure a fixed count of exact OP calls with current-stream Events.

    The callable must launch exactly one complete OP.  Each retained sample
    contains the device duration per public call.  A count greater than one is
    for a steady-state, no-host-gap timing protocol: every call still executes
    inside the same current-stream Event interval and the raw aggregate duration
    and normalization count are retained.  An optional pre-Event operation is
    submitted on that same stream immediately before each Event start and is
    excluded from every retained interval. Hardware runners remain responsible
    for proving distinct-input semantics when batching calls.
    """

    if not callable(operation):
        raise TypeError("operation must be callable")
    warmup_iterations = _positive_integer(
        warmup_iterations, "warmup_iterations")
    sample_count = _positive_integer(sample_count, "sample_count")
    operation_repetitions_per_sample = _positive_integer(
        operation_repetitions_per_sample,
        "operation_repetitions_per_sample")
    for name in ("Event", "current_stream", "synchronize"):
        if not hasattr(device_module, name):
            raise TypeError(f"device module has no {name}")
    if join_operation_streams is not None and not callable(
            join_operation_streams):
        raise TypeError("join_operation_streams must be callable or None")
    if pre_event_operation is not None and not callable(pre_event_operation):
        raise TypeError("pre_event_operation must be callable or None")
    if telemetry_observer is not None and not callable(telemetry_observer):
        raise TypeError("telemetry_observer must be callable or None")
    if cache_eviction_operation is None:
        if cache_eviction_bytes != 0:
            raise ValueError(
                "cache_eviction_bytes requires cache_eviction_operation")
    else:
        if not callable(cache_eviction_operation):
            raise TypeError("cache_eviction_operation must be callable")
        cache_eviction_bytes = _positive_integer(
            cache_eviction_bytes, "cache_eviction_bytes")

    event_stream = device_module.current_stream()

    def run_operations():
        results = []
        for _ in range(operation_repetitions_per_sample):
            results.append(operation())
        return results

    device_module.synchronize()
    warmup_results = None
    warmup_eviction_result = None
    warmup_pre_event_result = None
    for _ in range(warmup_iterations):
        if cache_eviction_operation is not None:
            warmup_eviction_result = cache_eviction_operation()
        if pre_event_operation is not None:
            warmup_pre_event_result = pre_event_operation()
        warmup_results = run_operations()
    device_module.synchronize()
    del warmup_results, warmup_eviction_result, warmup_pre_event_result

    prelude_start = device_module.Event(enable_timing=True)
    prelude_end = device_module.Event(enable_timing=True)
    prelude_eviction_result = None
    prelude_pre_event_result = None
    if cache_eviction_operation is not None:
        prelude_eviction_result = cache_eviction_operation()
    if pre_event_operation is not None:
        prelude_pre_event_result = pre_event_operation()
    prelude_start.record(event_stream)
    prelude_results = run_operations()
    if join_operation_streams is not None:
        join_operation_streams()
    prelude_end.record(event_stream)
    prelude_end.synchronize()
    device_module.synchronize()
    prelude_elapsed_ns = float(
        prelude_start.elapsed_time(prelude_end)) * 1_000_000.0
    if not math.isfinite(prelude_elapsed_ns) or prelude_elapsed_ns <= 0.0:
        raise RuntimeError(
            "device Event prelude is not positive: "
            f"{prelude_elapsed_ns!r} ns")
    del (
        prelude_results, prelude_eviction_result, prelude_pre_event_result,
        prelude_start, prelude_end)

    samples = []
    for sample_index in range(sample_count):
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        eviction_result = None
        pre_event_result = None
        if cache_eviction_operation is not None:
            eviction_result = cache_eviction_operation()
        if pre_event_operation is not None:
            pre_event_result = pre_event_operation()
        telemetry_before_event = None
        if telemetry_observer is not None:
            telemetry_before_event = telemetry_observer()
            if not isinstance(telemetry_before_event, Mapping):
                raise TypeError("telemetry_observer must return a mapping")
        host_interval_start_ns = time.monotonic_ns()
        start.record(event_stream)
        host_submission_start_ns = time.monotonic_ns()
        results = run_operations()
        host_submission_end_ns = time.monotonic_ns()
        if join_operation_streams is not None:
            join_operation_streams()
        end.record(event_stream)
        end.synchronize()
        device_module.synchronize()
        host_interval_end_ns = time.monotonic_ns()
        elapsed_ms = float(start.elapsed_time(end))
        aggregate_elapsed_ns = elapsed_ms * 1_000_000.0
        if (not math.isfinite(aggregate_elapsed_ns)
                or aggregate_elapsed_ns <= 0.0):
            raise RuntimeError(
                f"device Event sample {sample_index} is not positive: "
                f"{aggregate_elapsed_ns!r} ns")
        host_interval_ns = host_interval_end_ns - host_interval_start_ns
        if aggregate_elapsed_ns > host_interval_ns:
            raise RuntimeError(
                "device Event duration exceeds its synchronized host interval: "
                f"event={aggregate_elapsed_ns} ns, host={host_interval_ns} ns")
        sample = {
            "sample_index": sample_index,
            "event_duration_ns": (
                aggregate_elapsed_ns / operation_repetitions_per_sample),
            "host_submission_ns": (
                host_submission_end_ns - host_submission_start_ns),
            "host_synchronized_interval_ns": host_interval_ns,
        }
        if telemetry_before_event is not None:
            sample["telemetry_before_event"] = dict(telemetry_before_event)
        if operation_repetitions_per_sample > 1:
            sample["aggregate_event_duration_ns"] = aggregate_elapsed_ns
            sample["operation_repetition_count"] = (
                operation_repetitions_per_sample)
        samples.append(sample)
        del results, eviction_result, pre_event_result

    durations = [sample["event_duration_ns"] for sample in samples]
    result = {
        "schema_version": (
            BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION
            if operation_repetitions_per_sample > 1
            else CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION),
        "warmup_iterations": warmup_iterations,
        "event_prelude_iterations": 1,
        "sample_count": sample_count,
        "cache_eviction_requested_bytes": cache_eviction_bytes,
        "cache_eviction_bytes_before_sample": cache_eviction_bytes,
        "samples": samples,
        "statistics": summarize_event_samples(durations),
    }
    if telemetry_observer is not None:
        result["telemetry_protocol"] = (
            "nvidia_smi_query_after_cache_eviction_before_start_event")
    if operation_repetitions_per_sample > 1:
        result["operation_repetitions_per_sample"] = (
            operation_repetitions_per_sample)
    return result


def measure_graph_replay_events(
        operation: Callable[[], Any], *, device_module: Any,
        warmup_iterations: int, sample_count: int,
        repetition_counts: tuple[int, int],
        cache_eviction_operation: Callable[[], Any] | None = None,
        cache_eviction_bytes: int = 0) -> Mapping[str, Any]:
    """Measure per-call GPU time from externally bracketed graph replays.

    Each graph contains repeated executions of the exact callable.  Events are
    deliberately outside capture: MetaX C500 does not return reliable elapsed
    times for Events captured inside a graph.  Dividing one replay interval by
    its exact repetition count amortizes the fixed short-interval Event cost and
    removes Python submission gaps from the measured device interval.  An
    optional cache sweep is queued on the same stream before each start Event,
    outside both graph capture and the timed interval; operations repeated
    inside one replay can still reuse cache state.
    """

    if not callable(operation):
        raise TypeError("operation must be callable")
    return _measure_graph_replay_events(
        lambda _index: operation(),
        operation_cycle_length=1,
        device_module=device_module,
        warmup_iterations=warmup_iterations,
        sample_count=sample_count,
        repetition_counts=repetition_counts,
        cache_eviction_operation=cache_eviction_operation,
        cache_eviction_bytes=cache_eviction_bytes,
    )


def measure_rotating_graph_replay_events(
        operations: tuple[Callable[[], Any], ...], *, device_module: Any,
        warmup_iterations: int, sample_count: int,
        repetition_counts: tuple[int, int],
        cache_eviction_operation: Callable[[], Any] | None = None,
        cache_eviction_bytes: int = 0) -> Mapping[str, Any]:
    """Measure a graph that rotates across distinct prepared buffer sets."""

    if (not isinstance(operations, tuple) or len(operations) < 2
            or any(not callable(operation) for operation in operations)):
        raise TypeError(
            "operations must be a tuple containing at least two callables")
    return _measure_graph_replay_events(
        lambda index: operations[index % len(operations)](),
        operation_cycle_length=len(operations),
        device_module=device_module,
        warmup_iterations=warmup_iterations,
        sample_count=sample_count,
        repetition_counts=repetition_counts,
        cache_eviction_operation=cache_eviction_operation,
        cache_eviction_bytes=cache_eviction_bytes,
    )


def measure_differential_rotating_graph_replay_events(
        operations: tuple[Callable[[], Any], ...], *, device_module: Any,
        warmup_iterations: int, sample_count: int,
        repetition_counts: tuple[int, int],
        per_call_distinct_input_bytes: int,
        minimum_distinct_input_bytes: int,
        hardware_l2_bytes: int,
        cache_eviction_operation: Callable[[], Any] | None = None,
        cache_eviction_bytes: int = 0,
        pre_event_operation: Callable[[], Any] | None = None,
        ) -> Mapping[str, Any]:
    """Measure paired replay differences over unique indexed inputs.

    Both graphs use a unique operation for every captured call.  Each replay
    receives its own same-stream cache sweep before the start Event.  Sampling
    alternates lower/higher order so slow drift is not confounded with graph
    size, and the reported OP duration is the positive paired difference
    ``(T_high - T_low) / (N_high - N_low)``.  Raw replay durations remain in
    the artifact as diagnostics; no CPU interval contributes to the result.
    """

    if (not isinstance(operations, tuple) or len(operations) < 2
            or any(not callable(operation) for operation in operations)):
        raise TypeError(
            "operations must be a tuple containing at least two callables")
    for value, name in (
            (per_call_distinct_input_bytes,
             "per_call_distinct_input_bytes"),
            (minimum_distinct_input_bytes,
             "minimum_distinct_input_bytes"),
            (hardware_l2_bytes, "hardware_l2_bytes")):
        _positive_integer(value, name)
    if minimum_distinct_input_bytes < 8 * hardware_l2_bytes:
        raise ValueError(
            "minimum_distinct_input_bytes must be at least 8x hardware L2")
    if minimum_distinct_input_bytes % hardware_l2_bytes != 0:
        raise ValueError(
            "minimum_distinct_input_bytes must be an integer multiple of "
            "hardware L2")
    if (not isinstance(repetition_counts, tuple)
            or len(repetition_counts) != 2):
        raise TypeError("repetition_counts must be a pair of integers")
    lower_count, higher_count = tuple(
        _positive_integer(value, f"repetition_counts[{index}]")
        for index, value in enumerate(repetition_counts))
    if lower_count >= higher_count:
        raise ValueError("repetition_counts must be strictly increasing")
    expected_lower_count = math.ceil(
        minimum_distinct_input_bytes / per_call_distinct_input_bytes)
    if lower_count != expected_lower_count or higher_count != 2 * lower_count:
        raise ValueError(
            "differential repetition counts must use "
            "lower=ceil(minimum/per_call) and higher=2*lower")
    if len(operations) < higher_count:
        raise ValueError(
            "differential graph replay requires a unique operation for every "
            "higher-graph call")
    lower_distinct_input_bytes = lower_count * per_call_distinct_input_bytes
    higher_distinct_input_bytes = higher_count * per_call_distinct_input_bytes
    delta_distinct_input_bytes = (
        higher_count - lower_count) * per_call_distinct_input_bytes
    if (lower_distinct_input_bytes < minimum_distinct_input_bytes
            or delta_distinct_input_bytes < minimum_distinct_input_bytes):
        raise AssertionError("differential distinct-input threshold was not met")
    for name in (
            "CUDAGraph", "Event", "current_stream", "graph", "synchronize"):
        if not hasattr(device_module, name):
            raise TypeError(f"device module has no {name}")
    warmup_iterations = _positive_integer(
        warmup_iterations, "warmup_iterations")
    sample_count = _positive_integer(sample_count, "sample_count")
    if cache_eviction_operation is None:
        if cache_eviction_bytes != 0:
            raise ValueError(
                "cache_eviction_bytes requires cache_eviction_operation")
    else:
        if not callable(cache_eviction_operation):
            raise TypeError("cache_eviction_operation must be callable")
        cache_eviction_bytes = _positive_integer(
            cache_eviction_bytes, "cache_eviction_bytes")
    if pre_event_operation is not None and not callable(pre_event_operation):
        raise TypeError("pre_event_operation must be callable or None")

    device_module.synchronize()
    warmup_result = None
    warmup_eviction_result = None
    warmup_pre_event_result = None
    for index in range(warmup_iterations):
        if cache_eviction_operation is not None:
            warmup_eviction_result = cache_eviction_operation()
        if pre_event_operation is not None:
            warmup_pre_event_result = pre_event_operation()
        warmup_result = operations[index % len(operations)]()
    device_module.synchronize()
    del warmup_result, warmup_eviction_result, warmup_pre_event_result

    graphs = []
    capture_results = []
    for repetition_count in (lower_count, higher_count):
        graph = device_module.CUDAGraph()
        final_capture_result = None
        with device_module.graph(graph):
            for index in range(repetition_count):
                final_capture_result = operations[index]()
        device_module.synchronize()
        graph_warmup_eviction_result = None
        graph_warmup_pre_event_result = None
        if cache_eviction_operation is not None:
            graph_warmup_eviction_result = cache_eviction_operation()
        if pre_event_operation is not None:
            graph_warmup_pre_event_result = pre_event_operation()
        graph.replay()
        device_module.synchronize()
        del graph_warmup_eviction_result, graph_warmup_pre_event_result
        graphs.append(graph)
        capture_results.append(final_capture_result)

    event_stream = device_module.current_stream()

    def measure_one_replay(graph: Any) -> Mapping[str, Any]:
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        eviction_result = None
        pre_event_result = None
        if cache_eviction_operation is not None:
            eviction_result = cache_eviction_operation()
        if pre_event_operation is not None:
            pre_event_result = pre_event_operation()
        host_interval_start_ns = time.monotonic_ns()
        start.record(event_stream)
        host_submission_start_ns = time.monotonic_ns()
        graph.replay()
        host_submission_end_ns = time.monotonic_ns()
        end.record(event_stream)
        end.synchronize()
        host_interval_end_ns = time.monotonic_ns()
        replay_duration_ns = float(start.elapsed_time(end)) * 1_000_000.0
        if not math.isfinite(replay_duration_ns) or replay_duration_ns <= 0.0:
            raise RuntimeError(
                f"graph replay Event duration is not positive: "
                f"{replay_duration_ns!r} ns")
        host_interval_ns = host_interval_end_ns - host_interval_start_ns
        if replay_duration_ns > host_interval_ns:
            raise RuntimeError(
                "graph replay Event duration exceeds its synchronized host "
                f"interval: event={replay_duration_ns} ns, "
                f"host={host_interval_ns} ns")
        del eviction_result, pre_event_result
        return {
            "replay_duration_ns": replay_duration_ns,
            "host_submission_ns": (
                host_submission_end_ns - host_submission_start_ns),
            "host_synchronized_interval_ns": host_interval_ns,
        }

    prelude_durations = []
    for graph in graphs:
        prelude_durations.append(measure_one_replay(graph)[
            "replay_duration_ns"])

    paired_samples = []
    raw_samples = ([], [])
    repetition_delta = higher_count - lower_count
    graph_indices = {"lower": 0, "higher": 1}
    for sample_index in range(sample_count):
        replay_order = (
            ("lower", "higher") if sample_index % 2 == 0
            else ("higher", "lower"))
        measurements = {}
        for label in replay_order:
            measurements[label] = measure_one_replay(
                graphs[graph_indices[label]])
        lower_replay_duration_ns = measurements["lower"][
            "replay_duration_ns"]
        higher_replay_duration_ns = measurements["higher"][
            "replay_duration_ns"]
        event_duration_ns = (
            (higher_replay_duration_ns - lower_replay_duration_ns)
            / repetition_delta)
        if not math.isfinite(event_duration_ns) or event_duration_ns <= 0.0:
            raise RuntimeError(
                "paired differential Event sample is not positive: "
                f"sample={sample_index}, lower={lower_replay_duration_ns} ns, "
                f"higher={higher_replay_duration_ns} ns")
        paired_samples.append({
            "sample_index": sample_index,
            "event_duration_ns": event_duration_ns,
            "replay_order": "_then_".join(replay_order),
            "lower_replay_duration_ns": lower_replay_duration_ns,
            "higher_replay_duration_ns": higher_replay_duration_ns,
            "repetition_delta": repetition_delta,
        })
        for graph_index, label in enumerate(("lower", "higher")):
            replay = measurements[label]
            repetition_count = (lower_count, higher_count)[graph_index]
            raw_samples[graph_index].append({
                "sample_index": sample_index,
                "event_duration_ns": (
                    replay["replay_duration_ns"] / repetition_count),
                "replay_duration_ns": replay["replay_duration_ns"],
                "repetition_count": repetition_count,
                "host_submission_ns": replay["host_submission_ns"],
                "host_synchronized_interval_ns": replay[
                    "host_synchronized_interval_ns"],
                "replay_order_position": replay_order.index(label),
            })

    repetition_runs = []
    for repetition_count, samples in zip(
            (lower_count, higher_count), raw_samples, strict=True):
        durations = [sample["event_duration_ns"] for sample in samples]
        repetition_runs.append({
            "repetition_count": repetition_count,
            "sample_count": sample_count,
            "samples": samples,
            "statistics": summarize_event_samples(durations),
        })
    differential_durations = [
        sample["event_duration_ns"] for sample in paired_samples]
    distinct_input_contract = {
        "hardware_l2_bytes": hardware_l2_bytes,
        "required_minimum_l2_multiple": 8,
        "configured_minimum_l2_multiple": (
            minimum_distinct_input_bytes // hardware_l2_bytes),
        "configured_minimum_bytes": minimum_distinct_input_bytes,
        "per_call_bytes": per_call_distinct_input_bytes,
        "lower_replay_bytes": lower_distinct_input_bytes,
        "higher_replay_bytes": higher_distinct_input_bytes,
        "delta_replay_bytes": delta_distinct_input_bytes,
        "repetition_selection_rule": (
            "lower=ceil(configured_minimum_bytes/per_call_bytes);"
            "higher=2*lower"),
    }
    result = {
        "schema_version": GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION,
        "mode": DIFFERENTIAL_GRAPH_REPLAY_EVENT_MODE,
        "warmup_iterations": warmup_iterations,
        "event_prelude_iterations_per_repetition": 1,
        "event_prelude_replay_durations_ns": prelude_durations,
        "sample_count": sample_count,
        "repetition_counts": [lower_count, higher_count],
        "operation_cycle_length": len(operations),
        "cache_eviction_requested_bytes": cache_eviction_bytes,
        "cache_eviction_bytes_before_replay": cache_eviction_bytes,
        "pre_event_operation_executions_per_replay": (
            1 if pre_event_operation is not None else 0),
        "distinct_input_contract": distinct_input_contract,
        "repetition_runs": repetition_runs,
        "samples": paired_samples,
        "statistics": summarize_event_samples(differential_durations),
        "estimator": "(T_high-T_low)/(N_high-N_low)",
        "sample_pairing": (
            "alternating_lower_then_higher_and_higher_then_lower"),
    }
    del capture_results, graphs
    device_module.synchronize()
    return result


def _collect_tensor_storages(value: Any) -> dict[tuple[str, int], int]:
    """Return unique device storages recursively reachable from one value."""

    if isinstance(value, Mapping):
        storages: dict[tuple[str, int], int] = {}
        for child in value.values():
            storages.update(_collect_tensor_storages(child))
        return storages
    if isinstance(value, (tuple, list)):
        storages = {}
        for child in value:
            storages.update(_collect_tensor_storages(child))
        return storages
    if not hasattr(value, "untyped_storage"):
        return {}
    storage = value.untyped_storage()
    key = (str(value.device), int(storage.data_ptr()))
    return {key: int(storage.nbytes())}


def measure_differential_rotating_npu_graph_replay_events(
        operations: tuple[Callable[[], Any], ...], *,
        operation_storage_groups: tuple[tuple[Any, ...], ...],
        device_module: Any, warmup_iterations: int, sample_count: int,
        hardware_l2_bytes: int, shared_nonoverlapping_output_bytes: int = 0,
        cache_eviction_operation: Callable[[], Any] | None = None,
        cache_eviction_bytes: int = 0,
        repetition_counts: tuple[int, int] = (16, 256),
        repetition_selection_rule: str = (
            "fixed_16_256_unique_preallocated_calls"),
        ) -> Mapping[str, Any]:
    """Measure paired NPUGraph replays over unique preallocated calls."""

    if (not isinstance(repetition_counts, tuple)
            or len(repetition_counts) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in repetition_counts)):
        raise TypeError(
            "NPUGraph repetition_counts must contain two positive integers")
    lower_count, higher_count = repetition_counts
    if lower_count >= higher_count:
        raise ValueError(
            "NPUGraph repetition_counts must be strictly increasing")
    if (not isinstance(repetition_selection_rule, str)
            or not repetition_selection_rule):
        raise TypeError(
            "NPUGraph repetition_selection_rule must be non-empty")
    if (not isinstance(operations, tuple) or len(operations) != higher_count
            or any(not callable(operation) for operation in operations)):
        raise TypeError(
            "NPUGraph measurement requires one operation per higher replay "
            f"position: expected={higher_count}")
    if (not isinstance(operation_storage_groups, tuple)
            or len(operation_storage_groups) != higher_count
            or any(not isinstance(group, tuple) or not group
                   for group in operation_storage_groups)):
        raise TypeError(
            "NPUGraph measurement requires one non-empty storage group per "
            f"higher replay position: expected={higher_count}")
    _positive_integer(hardware_l2_bytes, "hardware_l2_bytes")
    if (isinstance(shared_nonoverlapping_output_bytes, bool)
            or not isinstance(shared_nonoverlapping_output_bytes, int)
            or shared_nonoverlapping_output_bytes < 0):
        raise TypeError(
            "shared_nonoverlapping_output_bytes must be a nonnegative integer")
    required_sweep_bytes = 2 * 1024 * 1024 * 1024
    if cache_eviction_bytes != required_sweep_bytes:
        raise ValueError("Ascend NPUGraph measurement requires a 2 GiB sweep")
    if cache_eviction_bytes < 8 * hardware_l2_bytes:
        raise ValueError("NPUGraph cache sweep must be at least 8x hardware L2")
    if not callable(cache_eviction_operation):
        raise TypeError("NPUGraph cache_eviction_operation must be callable")
    for name in ("NPUGraph", "Event", "current_stream", "graph", "synchronize"):
        if not hasattr(device_module, name):
            raise TypeError(f"device module has no {name}")
    warmup_iterations = _positive_integer(
        warmup_iterations, "warmup_iterations")
    sample_count = _positive_integer(sample_count, "sample_count")

    input_storages: dict[tuple[str, int], int] = {}
    prior_group_storages: set[tuple[str, int]] = set()
    for index, group in enumerate(operation_storage_groups):
        group_storages = _collect_tensor_storages(group)
        if not group_storages:
            raise ValueError(f"operation storage group {index} has no tensors")
        repeated = prior_group_storages.intersection(group_storages)
        if repeated:
            raise ValueError(
                f"operation storage group {index} reuses earlier addresses")
        prior_group_storages.update(group_storages)
        input_storages.update(group_storages)

    device_module.synchronize()
    warmup_result = None
    warmup_eviction_result = None
    for index in range(warmup_iterations):
        warmup_eviction_result = cache_eviction_operation()
        warmup_result = operations[index % higher_count]()
    device_module.synchronize()
    del warmup_result, warmup_eviction_result

    graphs = []
    capture_results = []
    for repetition_count in (lower_count, higher_count):
        graph = device_module.NPUGraph()
        graph_results = []
        with device_module.graph(graph):
            for index in range(repetition_count):
                graph_results.append(operations[index]())
        device_module.synchronize()
        graph_warmup_eviction_result = cache_eviction_operation()
        graph.replay()
        device_module.synchronize()
        del graph_warmup_eviction_result
        graphs.append(graph)
        capture_results.append(tuple(graph_results))

    output_storages: dict[tuple[str, int], int] = {}
    for index, result in enumerate(capture_results[1]):
        call_storages = dict(_collect_tensor_storages(
            operation_storage_groups[index]))
        call_storages.update(_collect_tensor_storages(result))
        repeated = set(call_storages).intersection(
            key for prior_index in range(index)
            for key in _collect_tensor_storages(
                (operation_storage_groups[prior_index],
                 capture_results[1][prior_index])))
        if repeated:
            raise ValueError(
                f"captured NPUGraph call {index} reuses earlier addresses")
        output_storages.update(_collect_tensor_storages(result))

    event_stream = device_module.current_stream()

    def measure_one_replay(graph: Any) -> Mapping[str, Any]:
        start = device_module.Event(enable_timing=True)
        end = device_module.Event(enable_timing=True)
        eviction_result = cache_eviction_operation()
        host_interval_start_ns = time.monotonic_ns()
        start.record(event_stream)
        host_submission_start_ns = time.monotonic_ns()
        graph.replay()
        host_submission_end_ns = time.monotonic_ns()
        end.record(event_stream)
        end.synchronize()
        host_interval_end_ns = time.monotonic_ns()
        replay_duration_ns = float(start.elapsed_time(end)) * 1_000_000.0
        if not math.isfinite(replay_duration_ns) or replay_duration_ns <= 0.0:
            raise RuntimeError(
                "NPUGraph replay Event duration is not positive: "
                f"{replay_duration_ns!r} ns")
        host_interval_ns = host_interval_end_ns - host_interval_start_ns
        if replay_duration_ns > host_interval_ns:
            raise RuntimeError(
                "NPUGraph replay Event duration exceeds its synchronized "
                f"host interval: event={replay_duration_ns} ns, "
                f"host={host_interval_ns} ns")
        del eviction_result
        return {
            "replay_duration_ns": replay_duration_ns,
            "host_submission_ns": host_submission_end_ns - host_submission_start_ns,
            "host_synchronized_interval_ns": host_interval_ns,
        }

    prelude_durations = [
        measure_one_replay(graph)["replay_duration_ns"] for graph in graphs]
    paired_samples = []
    raw_samples = ([], [])
    repetition_delta = higher_count - lower_count
    for sample_index in range(sample_count):
        replay_order = (("lower", "higher") if sample_index % 2 == 0
                        else ("higher", "lower"))
        graph_index = {"lower": 0, "higher": 1}
        measured = {
            label: measure_one_replay(graphs[graph_index[label]])
            for label in replay_order
        }
        lower_ns = measured["lower"]["replay_duration_ns"]
        higher_ns = measured["higher"]["replay_duration_ns"]
        duration_ns = (higher_ns - lower_ns) / repetition_delta
        if not math.isfinite(duration_ns) or duration_ns <= 0.0:
            raise RuntimeError(
                "paired differential NPUGraph sample is not positive: "
                f"sample={sample_index}, lower={lower_ns}, higher={higher_ns}")
        paired_samples.append({
            "sample_index": sample_index,
            "event_duration_ns": duration_ns,
            "replay_order": "_then_".join(replay_order),
            "lower_replay_duration_ns": lower_ns,
            "higher_replay_duration_ns": higher_ns,
            "repetition_delta": repetition_delta,
        })
        for index, label in enumerate(("lower", "higher")):
            replay = measured[label]
            count = (lower_count, higher_count)[index]
            raw_samples[index].append({
                "sample_index": sample_index,
                "event_duration_ns": replay["replay_duration_ns"] / count,
                "replay_duration_ns": replay["replay_duration_ns"],
                "repetition_count": count,
                "host_submission_ns": replay["host_submission_ns"],
                "host_synchronized_interval_ns": replay[
                    "host_synchronized_interval_ns"],
                "replay_order_position": replay_order.index(label),
            })

    repetition_runs = []
    for repetition_count, samples in zip(
            (lower_count, higher_count), raw_samples, strict=True):
        durations = [sample["event_duration_ns"] for sample in samples]
        repetition_runs.append({
            "repetition_count": repetition_count,
            "sample_count": sample_count,
            "samples": samples,
            "statistics": summarize_event_samples(durations),
        })
    output_only_storages = {
        key: size for key, size in output_storages.items()
        if key not in input_storages
    }
    preallocated_distinct_bytes = sum(input_storages.values())
    graph_output_distinct_bytes = sum(output_only_storages.values())
    rotation_total_distinct_bytes = (
        preallocated_distinct_bytes + graph_output_distinct_bytes
        + shared_nonoverlapping_output_bytes)
    l2_multiple = cache_eviction_bytes / hardware_l2_bytes
    contract = {
        "graph_backend": "torch.npu.NPUGraph",
        "hardware_l2_bytes": hardware_l2_bytes,
        "cache_eviction_bytes": cache_eviction_bytes,
        "cache_eviction_l2_multiple": l2_multiple,
        "cache_eviction_l2_multiple_rounded_2dp": round(l2_multiple, 2),
        "lower_operation_count": lower_count,
        "higher_operation_count": higher_count,
        "operation_count_delta": repetition_delta,
        "preallocated_operation_count": higher_count,
        "preallocated_distinct_bytes": preallocated_distinct_bytes,
        "graph_output_distinct_bytes": graph_output_distinct_bytes,
        "shared_nonoverlapping_output_bytes": shared_nonoverlapping_output_bytes,
        "rotation_total_distinct_bytes": rotation_total_distinct_bytes,
        "addresses_unique_within_each_graph": True,
        "rotating_set_may_be_smaller_than_l2": True,
        "selection_rule": repetition_selection_rule,
        "cache_policy": (
            "same_stream_2_gib_bf16_sweep_before_every_graph_replay"),
    }
    durations = [sample["event_duration_ns"] for sample in paired_samples]
    result = {
        "schema_version": GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION,
        "mode": DIFFERENTIAL_NPU_GRAPH_REPLAY_EVENT_MODE,
        "warmup_iterations": warmup_iterations,
        "event_prelude_iterations_per_repetition": 1,
        "event_prelude_replay_durations_ns": prelude_durations,
        "sample_count": sample_count,
        "repetition_counts": [lower_count, higher_count],
        "operation_cycle_length": higher_count,
        "cache_eviction_requested_bytes": cache_eviction_bytes,
        "cache_eviction_bytes_before_replay": cache_eviction_bytes,
        "rotating_buffer_contract": contract,
        "repetition_runs": repetition_runs,
        "samples": paired_samples,
        "statistics": summarize_event_samples(durations),
        "estimator": "(T_high-T_low)/(N_high-N_low)",
        "sample_pairing": (
            "alternating_lower_then_higher_and_higher_then_lower"),
    }
    del capture_results, graphs
    device_module.synchronize()
    return result


def _measure_graph_replay_events(
        indexed_operation: Callable[[int], Any], *,
        operation_cycle_length: int, device_module: Any,
        warmup_iterations: int, sample_count: int,
        repetition_counts: tuple[int, int],
        cache_eviction_operation: Callable[[], Any] | None,
        cache_eviction_bytes: int) -> Mapping[str, Any]:
    warmup_iterations = _positive_integer(
        warmup_iterations, "warmup_iterations")
    sample_count = _positive_integer(sample_count, "sample_count")
    if (not isinstance(repetition_counts, tuple)
            or len(repetition_counts) != 2):
        raise TypeError("repetition_counts must be a pair of integers")
    normalized_counts = tuple(
        _positive_integer(value, f"repetition_counts[{index}]")
        for index, value in enumerate(repetition_counts))
    if normalized_counts[0] >= normalized_counts[1]:
        raise ValueError("repetition_counts must be strictly increasing")
    for name in (
            "CUDAGraph", "Event", "current_stream", "graph", "synchronize"):
        if not hasattr(device_module, name):
            raise TypeError(f"device module has no {name}")
    if cache_eviction_operation is None:
        if cache_eviction_bytes != 0:
            raise ValueError(
                "cache_eviction_bytes requires cache_eviction_operation")
    else:
        if not callable(cache_eviction_operation):
            raise TypeError("cache_eviction_operation must be callable")
        cache_eviction_bytes = _positive_integer(
            cache_eviction_bytes, "cache_eviction_bytes")

    device_module.synchronize()
    warmup_result = None
    warmup_eviction_result = None
    for index in range(warmup_iterations):
        if cache_eviction_operation is not None:
            warmup_eviction_result = cache_eviction_operation()
        warmup_result = indexed_operation(index)
    device_module.synchronize()
    del warmup_result, warmup_eviction_result

    runs = []
    for repetition_count in normalized_counts:
        graph = device_module.CUDAGraph()
        final_capture_result = None
        with device_module.graph(graph):
            for index in range(repetition_count):
                final_capture_result = indexed_operation(index)
        device_module.synchronize()
        graph.replay()
        device_module.synchronize()

        event_stream = device_module.current_stream()
        prelude_start = device_module.Event(enable_timing=True)
        prelude_end = device_module.Event(enable_timing=True)
        prelude_eviction_result = None
        if cache_eviction_operation is not None:
            prelude_eviction_result = cache_eviction_operation()
        prelude_start.record(event_stream)
        graph.replay()
        prelude_end.record(event_stream)
        prelude_end.synchronize()
        prelude_elapsed_ns = float(
            prelude_start.elapsed_time(prelude_end)) * 1_000_000.0
        if not math.isfinite(prelude_elapsed_ns) or prelude_elapsed_ns <= 0.0:
            raise RuntimeError(
                "graph Event prelude is not positive: "
                f"{prelude_elapsed_ns!r} ns")
        del prelude_start, prelude_end, prelude_eviction_result

        samples = []
        for sample_index in range(sample_count):
            start = device_module.Event(enable_timing=True)
            end = device_module.Event(enable_timing=True)
            eviction_result = None
            if cache_eviction_operation is not None:
                eviction_result = cache_eviction_operation()
            host_interval_start_ns = time.monotonic_ns()
            start.record(event_stream)
            host_submission_start_ns = time.monotonic_ns()
            graph.replay()
            host_submission_end_ns = time.monotonic_ns()
            end.record(event_stream)
            end.synchronize()
            host_interval_end_ns = time.monotonic_ns()
            replay_duration_ns = float(start.elapsed_time(end)) * 1_000_000.0
            per_call_duration_ns = replay_duration_ns / repetition_count
            if (not math.isfinite(per_call_duration_ns)
                    or per_call_duration_ns <= 0.0):
                raise RuntimeError(
                    f"graph Event sample {sample_index} is not positive: "
                    f"{per_call_duration_ns!r} ns")
            host_interval_ns = host_interval_end_ns - host_interval_start_ns
            if replay_duration_ns > host_interval_ns:
                raise RuntimeError(
                    "graph Event duration exceeds its synchronized host "
                    f"interval: event={replay_duration_ns} ns, "
                    f"host={host_interval_ns} ns")
            samples.append({
                "sample_index": sample_index,
                "event_duration_ns": per_call_duration_ns,
                "replay_duration_ns": replay_duration_ns,
                "repetition_count": repetition_count,
                "host_submission_ns": (
                    host_submission_end_ns - host_submission_start_ns),
                "host_synchronized_interval_ns": host_interval_ns,
            })
            del eviction_result
        durations = [row["event_duration_ns"] for row in samples]
        runs.append({
            "repetition_count": repetition_count,
            "sample_count": sample_count,
            "samples": samples,
            "statistics": summarize_event_samples(durations),
        })
        del final_capture_result, graph
        device_module.synchronize()

    lower_median_ns = runs[0]["statistics"]["median_ns"]
    selected_median_ns = runs[1]["statistics"]["median_ns"]
    disagreement_ns = abs(selected_median_ns - lower_median_ns)
    tolerance_ns = max(500.0, 0.05 * max(
        lower_median_ns, selected_median_ns))
    converged = disagreement_ns <= tolerance_ns
    return {
        "schema_version": GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION,
        "mode": "external_events_around_cuda_graph_replay",
        "warmup_iterations": warmup_iterations,
        "event_prelude_iterations_per_repetition": 1,
        "sample_count": sample_count,
        "repetition_counts": list(normalized_counts),
        "operation_cycle_length": operation_cycle_length,
        "cache_eviction_requested_bytes": cache_eviction_bytes,
        "cache_eviction_bytes_before_replay": cache_eviction_bytes,
        "repetition_runs": runs,
        "samples": runs[1]["samples"],
        "statistics": runs[1]["statistics"],
        "convergence": {
            "status": "converged" if converged else "nonconverged",
            "absolute_difference_ns": disagreement_ns,
            "tolerance_ns": tolerance_ns,
            "relative_difference": (
                disagreement_ns / max(lower_median_ns, selected_median_ns)),
        },
    }
