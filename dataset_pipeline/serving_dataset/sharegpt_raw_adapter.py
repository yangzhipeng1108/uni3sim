"""CPU-only normalization of retained ShareGPT Serving captures.

The adapter reads immutable raw capture directories and produces only the
normalized in-memory payload accepted by :mod:`compact_candidate_writer`.
It never imports Serving or evaluation code and never consumes historical
prediction, comparison, LUT, or regenerated-trace artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left, bisect_right
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Callable, Mapping, NamedTuple, Sequence

from dataset_pipeline.conversion.mctracer_chrome import (
    McTracerKernelEvent,
    McTracerMemcpyEvent,
    McTracerMemsetEvent,
    iter_mctracer_device_events,
)

from .compact_candidate_writer import (
    AvailabilityPayload as WriterAvailabilityPayload,
    CompactCasePayload,
)
from .compact_case import (
    CANONICAL_REQUEST_COLUMNS,
    SCHEMA_VERSION,
    CanonicalRequestRow,
    raw_source_set_sha256,
    validate_converter_source_commit,
)

from dataset_pipeline.serving_dataset.compact_case import (
    _SCHEDULER_FIELD_PROVENANCE,
)

from dataset_pipeline.serving_dataset.json_shape import (
    compact_mapping as _mapping,
    compact_sequence as _sequence,
    reject_duplicate_json_keys,
)


_CASE_NAME = re.compile(
    r"ShareGPT-(?P<model>Qwen3-4B|Qwen3-8B|Llama-3\.1-8B)_TP1_n128_qps(?P<qps>4|16|64)"
)


class _RawCaptureForm(NamedTuple):
    """How one hardware's retained ShareGPT raw capture is laid out.

    ``request_contract`` is ``"arrival_schedule"`` when the case keeps the
    workload generator's ``arrival_schedule.json`` and the plugin sidecar maps
    requests explicitly (``vllm_req_N``), or ``"capture_log"`` when no arrival
    schedule is retained: serving metrics keep the arrival sequence, requests
    are mapped by ordinal and ``capture.log`` records the runtime.
    ``profiler`` names the device profiler archive the case retains:
    ``"nsys"``, rank-aware ``"nsys_ranked"``, ``"mctracer"`` or ``"msprof"``.
    """

    request_contract: str
    profiler: str


_RAW_CAPTURE_FORMS = {
    "nvidia_a800": _RawCaptureForm("arrival_schedule", "nsys"),
    "nvidia_h200": _RawCaptureForm("arrival_schedule", "nsys_ranked"),
    "metax_c500": _RawCaptureForm("arrival_schedule", "mctracer"),
    "ascend_910b": _RawCaptureForm("capture_log", "msprof"),
}
_SUPPORTED_HARDWARE = tuple(_RAW_CAPTURE_FORMS)


def _raw_capture_form(hardware_id: str, failure: str) -> _RawCaptureForm:
    if hardware_id not in _RAW_CAPTURE_FORMS:
        raise AssertionError(failure)
    return _RAW_CAPTURE_FORMS[hardware_id]
_H200_NVTX_STEP = re.compile(r"midlayer-op-(?P<kind>step|sample):(?P<iter>[0-9]+)")
_DENSE_MODELS = ("Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B")
_RAW_MODEL_BASENAME_BY_MODEL_ID = {
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-8B": "Qwen3-8B",
    "Llama-3.1-8B": "Llama-3.1-8B-Instruct",
    "Qwen3-32B": "Qwen3-32B",
    "Qwen3-30B-A3B": "Qwen3-30B-A3B",
    "Llama-3-70B": "Meta-Llama-3-70B-Instruct",
    "Llama-3.1-70B": "Llama-3.1-70B-Instruct",
}
_REQUEST_ID = re.compile(r"req_(?P<index>[0-9]+)")
_ASCEND_ENGINE_RUNTIME = re.compile(
    r"Initializing a V1 LLM engine \(v(?P<version>[^)]+)\) with config: "
    r"[^\r\n]*?enable_prefix_caching=(?P<prefix>True|False)(?:,|\))"
)
_ASCEND_BLOCK_SIZE_HINT = re.compile(
    r"\bBlock size is set to (?P<tokens>\S+) "
    r"if prefix cache or chunked prefill is enabled\."
)
_ASCEND_WORKER_RANK = re.compile(
    r"\(Worker(?:_TP[0-9]+)? pid=(?P<pid>[0-9]+)\).*?"
    r"world_size=(?P<world_size>[0-9]+) rank=(?P<rank>[0-9]+) "
    r"local_rank=(?P<local_rank>[0-9]+)"
)
_ASCEND_FORWARD_TERMINAL_KERNEL = "aclnnEmbedding_GatherV2AiCore_GatherV2"
_ASCEND_GEOMETRY_FREE_TASKS = {
    (
        "aclnnBincount_BincountAiCpu_Bincount",
        "Bincount",
        "AI_CPU",
    ),
}
_WARMUP_PREFIX = "warmup"
_CONVERTER_IMPLEMENTATION = "midlayer.sharegpt_raw_adapter.v1"
_MOE_MODEL_TOPOLOGY = {
    "Qwen3-30B-A3B": (128, 8, 48),
}
_C500_SAMPLER_TERMINAL_FUNCTION = (
    "at::native::InputPerOutputContinuousReduceKernel<"
)
_C500_SAMPLER_TERMINAL_OPERATION = "at::native::ArgMaxOps<float>"
_C500_SAMPLER_COPY_FUNCTION = "void at::native::unrolled_elementwise_kernel<"
_C500_SAMPLER_COPY_INT = (
    "direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() "
    "const::{lambda()#3}::operator()() const::{lambda(int)#1}"
)
_C500_SAMPLER_COPY_LONG = (
    "direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() "
    "const::{lambda()#4}::operator()() const::{lambda(long)#1}"
)
_C500_SAMPLER_COPY_DTOH = "memcpy DTOH(device,pinned)"
_C500_LEGACY_MEMORY_EVENT_NAMES = frozenset(
    {
        "memset",
        "memcpy HTOD(pinned,device)",
        "memcpy DTOH(device,pinned)",
        "memcpy DTOH(device,pageable)",
        "memcpy DTOD(device,device)",
    }
)
_C500_LEGACY_CONVERTER = {
    "implementation_id": "scripts/serve_predict/build_eval_dataset.py",
    "source_commit": "6304b3824f4d8f85a4c1c0f467e36d43b21c9354",
    "source_sha256": "c9a51a501a3da1f46e1d10493fba069ed66f0ad9cb22fdc1663edb65e8839ea1",
}
_C500_REBUILD_CONVERTER = {
    "implementation_id": "scripts/serve_predict/rebuild_nsys_full_from_mctracer.py",
    "source_commit": "19fd21c1ca555461a66f042d25f252d7cd87998e",
    "source_sha256": "8c95ac7824a5cad75f2704381b9f158a1fb937a0c42ab20f19851aee87638f84",
}
_C500_WORKER_PID = re.compile(
    r"Worker_TP(?P<tp>[0-9]+)(?:_EP(?P<ep>[0-9]+))? pid=(?P<pid>[0-9]+)"
)
_C500_TRACER_OUTPUT_PID = re.compile(r"mctracer-(?P<pid>[0-9]+)\.json")


@dataclass(frozen=True)
class AvailabilityPayload:
    """Internal compatibility carrier used by the raw parsers below."""

    status: str
    coverage: str | None
    reason: str | None
    payload: object | None
    source_archive_ids: tuple[str, ...] | None


@dataclass(frozen=True)
class _ShareGPTCaseContract:
    hardware_id: str
    model_id: str
    target_qps: int
    tensor_parallel_size: int
    expert_parallel_size: int
    pipeline_parallel_size: int
    dtype: str | None
    case_id: str
    raw_directory_name: str


@dataclass(frozen=True)
class _AscendRankProfile:
    rank: int
    local_rank: int
    device_id: int
    database: Path
    info_path: Path
    recovery_paths: tuple[Path, Path, Path, Path] | None


@dataclass(frozen=True)
class _C500ProjectionEvent:
    name: str
    timestamp_ns: int
    duration_ns: int | None
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_memory_bytes: int
    registers_per_thread: int | None
    stream: int | None


@dataclass(frozen=True)
class _C500ProjectionScan:
    top_level_count: int
    top_level_digest: str
    entry_count_by_rank: Mapping[int, int]
    digest_by_rank: Mapping[int, str]
    bottleneck_rank: int | None
    duration_present: bool


class _StreamingJsonReader:
    """Small bounded-memory reader for arrays of large JSON objects."""

    def __init__(self, path: Path, offset: int = 0, *, chunk_bytes: int = 1 << 20) -> None:
        self._path = path
        self._stream = path.open(encoding="utf-8")
        self._stream.seek(offset)
        if chunk_bytes <= 0:
            raise ValueError("streaming JSON chunk size must be positive")
        self._chunk_bytes = chunk_bytes
        self._buffer = ""
        self._position = 0
        self._eof = False
        self._decoder = json.JSONDecoder(
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"C500 projection contains invalid constant {value}")
            ),
        )

    def __enter__(self) -> _StreamingJsonReader:
        return self

    def __exit__(self, *_: object) -> None:
        self._stream.close()

    def _refill(self) -> None:
        if self._position:
            self._buffer = self._buffer[self._position:]
            self._position = 0
        chunk = self._stream.read(self._chunk_bytes)
        if chunk:
            self._buffer += chunk
        else:
            self._eof = True

    def _ensure(self) -> None:
        while self._position >= len(self._buffer) and not self._eof:
            self._refill()

    def skip_whitespace(self) -> None:
        while True:
            self._ensure()
            while (
                self._position < len(self._buffer)
                and self._buffer[self._position] in " \t\r\n"
            ):
                self._position += 1
            if self._position < len(self._buffer) or self._eof:
                return

    def peek(self) -> str:
        self.skip_whitespace()
        self._ensure()
        if self._position >= len(self._buffer):
            raise ValueError(f"truncated C500 projection JSON: {self._path}")
        return self._buffer[self._position]

    def consume(self, token: str) -> None:
        if self.peek() != token:
            raise ValueError(
                f"C500 projection expected {token!r} at byte-stream position"
            )
        self._position += 1

    def value(self, description: str) -> object:
        self.skip_whitespace()
        while True:
            try:
                value, consumed = self._decoder.raw_decode(
                    self._buffer, self._position
                )
            except json.JSONDecodeError as error:
                if self._eof:
                    raise ValueError(
                        f"invalid or truncated {description} in {self._path}: {error}"
                    ) from error
                if len(self._buffer) - self._position > 16 << 20:
                    raise ValueError(
                        f"{description} exceeds the bounded streaming record size"
                    ) from error
                self._refill()
            else:
                self._position = consumed
                return value

    def require_eof(self) -> None:
        self.skip_whitespace()
        self._ensure()
        if self._position != len(self._buffer) or not self._eof:
            if not self._eof:
                self._refill()
                self.skip_whitespace()
            if self._position != len(self._buffer):
                raise ValueError(
                    f"C500 projection contains trailing JSON content: {self._path}"
                )


def _ascend_profile_source_specs(
    profile: _AscendRankProfile,
) -> tuple[tuple[Path, str, str, None], ...]:
    specs = [
        (
            profile.database,
            f"device-profiler-rank{profile.rank}",
            "msprof",
            None,
        ),
        (
            profile.info_path,
            f"profiler-metadata-rank{profile.rank}",
            "msprof",
            None,
        ),
    ]
    if profile.recovery_paths is not None:
        archive_names = (
            "intermediate-device-tasks",
            "intermediate-ge-task-info",
            "intermediate-host-runtime",
            "intermediate-device-clock",
        )
        specs.extend(
            (
                path,
                f"{archive_name}-rank{profile.rank}",
                "msprof",
                None,
            )
            for path, archive_name in zip(
                profile.recovery_paths, archive_names, strict=True
            )
        )
    return tuple(specs)


_SECOND_PHASE_A800_CONTRACTS = {
    ("Qwen3-32B", 1): _ShareGPTCaseContract(
        "nvidia_a800", "Qwen3-32B", 1, 2, 1, 1, "float16",
        "tp2_ep1_qps1", "ShareGPT-Qwen3-32B_TP2_n128_qps1",
    ),
    ("Qwen3-32B", 4): _ShareGPTCaseContract(
        "nvidia_a800", "Qwen3-32B", 4, 2, 1, 1, "bfloat16",
        "tp2_ep1_qps4", "ShareGPT-Qwen3-32B_TP2_n128_qps4",
    ),
    ("Qwen3-32B", 16): _ShareGPTCaseContract(
        "nvidia_a800", "Qwen3-32B", 16, 2, 1, 1, "float16",
        "tp2_ep1_qps16", "ShareGPT-Qwen3-32B_TP2_n128_qps16",
    ),
    ("Qwen3-30B-A3B", 1): _ShareGPTCaseContract(
        "nvidia_a800", "Qwen3-30B-A3B", 1, 2, 2, 1, "bfloat16",
        "tp2_ep2_qps1", "ShareGPT-Qwen3-30B-A3B_EP2_n128_qps1",
    ),
    ("Qwen3-30B-A3B", 4): _ShareGPTCaseContract(
        "nvidia_a800", "Qwen3-30B-A3B", 4, 2, 2, 1, "bfloat16",
        "tp2_ep2_qps4", "ShareGPT-Qwen3-30B-A3B_EP2_n128_qps4",
    ),
    ("Qwen3-30B-A3B", 16): _ShareGPTCaseContract(
        "nvidia_a800", "Qwen3-30B-A3B", 16, 2, 2, 1, "bfloat16",
        "tp2_ep2_qps16", "ShareGPT-Qwen3-30B-A3B_EP2_n128_qps16",
    ),
}


def build_sharegpt_compact_payload(
    raw_case_dir: str | Path,
    *,
    hardware_id: str,
    model_id: str,
    target_qps: int,
    converter_source_commit: str,
) -> CompactCasePayload:
    """Normalize one frozen first-phase raw case without writing any files."""

    validate_converter_source_commit(converter_source_commit)
    case_dir = Path(raw_case_dir)
    _validate_case_identity(case_dir, hardware_id, model_id, target_qps)
    return _build_sharegpt_payload(
        case_dir,
        hardware_id=hardware_id,
        model_id=model_id,
        target_qps=target_qps,
        tensor_parallel_size=1,
        expert_parallel_size=1,
        pipeline_parallel_size=1,
        expected_dtype=None,
        case_id=f"tp1_qps{target_qps}",
        converter_source_commit=converter_source_commit,
    )


def build_sharegpt_case_compact_payload(
    raw_case_dir: str | Path,
    *,
    hardware_id: str,
    raw_model_id: str,
    target_qps: int,
    tensor_parallel_size: int,
    expert_parallel_size: int,
    case_id: str,
    publish_metrics: bool,
    converter_source_commit: str,
) -> CompactCasePayload:
    """Normalize one explicitly frozen matrix cell used by the 51-case builder."""

    validate_converter_source_commit(converter_source_commit)
    case_dir = Path(raw_case_dir)
    if not case_dir.is_dir():
        raise ValueError(f"raw ShareGPT case is not a directory: {case_dir}")
    if raw_model_id not in (
        "Qwen3-30B-A3B",
        "Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B", "Qwen3-32B",
        "Llama-3-70B", "Llama-3.1-70B",
    ):
        raise ValueError(f"unsupported raw model identity: {raw_model_id!r}")
    payload = _build_sharegpt_payload(
        case_dir,
        hardware_id=hardware_id,
        model_id=raw_model_id,
        target_qps=target_qps,
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
        pipeline_parallel_size=1,
        expected_dtype=None,
        case_id=case_id,
        converter_source_commit=converter_source_commit,
    )
    if publish_metrics:
        return payload
    return CompactCasePayload(
        identity=payload.identity,
        runtime=payload.runtime,
        requests=payload.requests,
        request_field_provenance=payload.request_field_provenance,
        cached_hit_value_evidence=payload.cached_hit_value_evidence,
        metrics=WriterAvailabilityPayload(
            "unavailable",
            "no official non-profiler serving ground truth is available",
            None,
        ),
        alignment_steps=payload.alignment_steps,
        raw_sources=payload.raw_sources,
        provenance_gaps=payload.provenance_gaps,
        converter=payload.converter,
    )


def build_second_phase_a800_sharegpt_compact_payload(
    raw_case_dir: str | Path,
    *,
    model_id: str,
    target_qps: int,
    converter_source_commit: str,
) -> CompactCasePayload:
    """Normalize one of the six frozen second-phase A800 captures."""

    validate_converter_source_commit(converter_source_commit)
    contract = _second_phase_a800_contract(model_id, target_qps)
    case_dir = Path(raw_case_dir)
    _validate_second_phase_case_identity(case_dir, contract)
    return _build_sharegpt_payload(
        case_dir,
        hardware_id=contract.hardware_id,
        model_id=contract.model_id,
        target_qps=contract.target_qps,
        tensor_parallel_size=contract.tensor_parallel_size,
        expert_parallel_size=contract.expert_parallel_size,
        pipeline_parallel_size=contract.pipeline_parallel_size,
        expected_dtype=contract.dtype,
        case_id=contract.case_id,
        converter_source_commit=converter_source_commit,
    )


def inspect_sharegpt_raw_source_set(
    raw_case_dir: str | Path,
    *,
    hardware_id: str,
    model_id: str,
    target_qps: int,
) -> tuple[tuple[Mapping[str, object], ...], str]:
    """Return the exact source-file inventory and digest for user confirmation."""

    case_dir = Path(raw_case_dir)
    _validate_case_identity(case_dir, hardware_id, model_id, target_qps)
    return _inspect_sharegpt_raw_source_set(case_dir, hardware_id)


def inspect_second_phase_a800_sharegpt_raw_source_set(
    raw_case_dir: str | Path,
    *,
    model_id: str,
    target_qps: int,
) -> tuple[tuple[Mapping[str, object], ...], str]:
    """Return the exact source inventory for one second-phase A800 capture."""

    contract = _second_phase_a800_contract(model_id, target_qps)
    case_dir = Path(raw_case_dir)
    _validate_second_phase_case_identity(case_dir, contract)
    return _inspect_sharegpt_raw_source_set(case_dir, contract.hardware_id)


def _inspect_sharegpt_raw_source_set(
    case_dir: Path,
    hardware_id: str,
) -> tuple[tuple[Mapping[str, object], ...], str]:
    source_specs: list[tuple[Path, str, str | None, str | None]] = [
        (case_dir / "meta.json", "case-metadata", "capture-metadata", None),
        (
            case_dir / "serving_metrics.json",
            "serving-metrics",
            "serving-runtime",
            None,
        ),
        (
            case_dir / "plugin_trace.json_scheduler.jsonl",
            "scheduler",
            "vllm-scheduler-trace",
            None,
        ),
        (
            case_dir / "plugin_trace.json_provenance.json",
            "trace-provenance",
            "vllm-operator-trace",
            None,
        ),
    ]
    operator_paths = tuple(sorted(case_dir.glob("plugin_trace.json_operators.jsonl.rank*")))
    if not operator_paths:
        raise FileNotFoundError("missing required per-rank operator traces")
    for path in operator_paths:
        rank = _operator_rank(path)
        source_specs.append((path, f"operators-rank{rank}", "vllm-operator-trace", None))
    form = _raw_capture_form(
        hardware_id,
        f"unsupported hardware after validation: {hardware_id}")
    if form.request_contract == "arrival_schedule":
        source_specs.insert(
            1,
            (
                case_dir / "arrival_schedule.json",
                "arrival-schedule",
                "workload-generator",
                None,
            ),
        )
    if form.profiler in ("nsys", "nsys_ranked"):
        profiler_path = case_dir / "nsys_profile.sqlite"
        if not profiler_path.is_file():
            raise FileNotFoundError(f"missing required raw source: {profiler_path}")
        source_specs.append(
            (profiler_path, "device-profiler", "nsys", _nsys_version(profiler_path))
        )
    elif form.profiler == "mctracer":
        source_specs.extend(_c500_profiler_source_specs(case_dir))
    elif form.profiler == "msprof":
        capture_path = case_dir / "capture.log"
        runtime_version, _, _ = _ascend_capture_runtime(capture_path)
        profiles = _ascend_rank_profiles(
            case_dir,
            tuple(_operator_rank(path) for path in operator_paths),
        )
        source_specs.append((capture_path, "runtime-log", "vllm", runtime_version))
        for profile in profiles:
            source_specs.extend(_ascend_profile_source_specs(profile))
    else:
        raise AssertionError(f"unsupported profiler hardware: {hardware_id}")
    sources = []
    for path, archive_id, tool, tool_version in source_specs:
        if not path.is_file():
            raise FileNotFoundError(f"missing required raw source: {path}")
        sources.append(
            _raw_source(case_dir, path, archive_id, tool, tool_version)
        )
    result = tuple(sources)
    return result, raw_source_set_sha256(result)


def _build_sharegpt_payload(
    case_dir: Path,
    *,
    hardware_id: str,
    model_id: str,
    target_qps: int,
    tensor_parallel_size: int,
    expert_parallel_size: int,
    pipeline_parallel_size: int,
    expected_dtype: str | None,
    case_id: str,
    converter_source_commit: str,
) -> CompactCasePayload:
    """Build a normalized payload from internally cross-validated raw evidence."""

    validate_converter_source_commit(converter_source_commit)

    meta_path = case_dir / "meta.json"
    arrival_path = case_dir / "arrival_schedule.json"
    metrics_path = case_dir / "serving_metrics.json"
    scheduler_path = case_dir / "plugin_trace.json_scheduler.jsonl"
    operator_paths = tuple(sorted(case_dir.glob("plugin_trace.json_operators.jsonl.rank*")))
    if not operator_paths:
        raise FileNotFoundError("missing required per-rank operator traces")
    operator_ranks = tuple(_operator_rank(path) for path in operator_paths)
    if operator_ranks != tuple(range(tensor_parallel_size)):
        raise ValueError(
            "operator traces must contain exactly one contiguous file for every TP rank"
        )
    provenance_path = case_dir / "plugin_trace.json_provenance.json"
    required_paths = [
        meta_path,
        metrics_path,
        scheduler_path,
        provenance_path,
    ]
    required_paths.extend(operator_paths)
    form = _raw_capture_form(
        hardware_id,
        f"unsupported hardware after validation: {hardware_id}")
    if form.request_contract == "arrival_schedule":
        required_paths.append(arrival_path)
    elif form.request_contract == "capture_log":
        required_paths.append(case_dir / "capture.log")
    else:
        raise AssertionError(f"unsupported hardware after validation: {hardware_id}")
    for required in required_paths:
        if not required.is_file():
            raise FileNotFoundError(f"missing required raw source: {required}")

    meta = _json_object(meta_path, "case metadata")
    metrics = _json_object(metrics_path, "Serving metrics")
    raw_scheduler = _jsonl_objects(scheduler_path, "Scheduler sidecar")
    provenance = _json_object(provenance_path, "trace provenance")

    _validate_meta(
        meta,
        model_id,
        target_qps,
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
    )
    metric_rows = _metric_rows(metrics)
    if form.request_contract == "capture_log":
        arrival_times = _metric_arrival_times(metric_rows)
        arrival_source_id = "serving-metrics"
        arrival_reason = (
            "the retained 910B raw has no arrival_schedule.json; serving metrics "
            "retain the requested arrival sequence"
        )
    elif form.request_contract == "arrival_schedule":
        arrival = _json_object(arrival_path, "arrival schedule")
        arrival_times = _arrival_times(arrival, target_qps, len(metric_rows))
        _validate_metric_arrivals(metric_rows, arrival_times)
        arrival_source_id = "arrival-schedule"
        arrival_reason = None
    else:
        raise AssertionError(f"unsupported hardware after validation: {hardware_id}")

    kv_block_size_tokens = None
    if form.request_contract == "arrival_schedule":
        request_mapping, input_tokens, runtime = _explicit_request_contract(
            metric_rows,
            raw_scheduler,
            provenance,
            meta,
            model_id,
            target_qps,
            tensor_parallel_size,
            expert_parallel_size,
            pipeline_parallel_size,
            expected_dtype,
        )
        source_mapping_kind = "explicit"
        source_namespace = "vllm_req_N"
        mapping_reason = None
        input_provenance_kind = "requested"
        input_source_id = "trace-provenance"
    elif form.request_contract == "capture_log":
        runtime_version, prefix_cache_enabled, kv_block_size_tokens = (
            _ascend_capture_runtime(case_dir / "capture.log")
        )
        request_mapping, input_tokens, runtime = _ordinal_request_contract(
            metric_rows,
            raw_scheduler,
            provenance,
            meta,
            model_id,
            target_qps,
            tensor_parallel_size,
            expert_parallel_size,
            pipeline_parallel_size,
            expected_dtype,
        )
        source_mapping_kind = "ordinal_derived"
        source_namespace = "opaque_scheduler_request_id"
        mapping_reason = (
            "raw metrics and Scheduler use different ID namespaces; mapping is the "
            "validated metrics physical order to Scheduler first-appearance order"
        )
        input_provenance_kind = "requested"
        input_source_id = "trace-provenance"
        runtime["runtime_version"] = runtime_version
        runtime["capture_enable_prefix_caching"] = prefix_cache_enabled
    else:
        raise AssertionError(f"unsupported hardware after validation: {hardware_id}")

    output_tokens = tuple(
        _integer(row["n_output_tokens"], "metrics n_output_tokens", minimum=0)
        for row in metric_rows
    )
    requests = tuple(
        CanonicalRequestRow(arrival_s * 1000.0, input_count, output_count, 0)
        for arrival_s, input_count, output_count in zip(
            arrival_times,
            input_tokens,
            output_tokens,
            strict=True,
        )
    )
    scheduler_records, source_step_to_compact, step_phases = _normalize_scheduler(
        raw_scheduler,
        request_mapping,
        input_tokens,
    )
    operator_records_list = []
    for operator_path, rank in zip(operator_paths, operator_ranks, strict=True):
        records, _ = _normalize_operators(
            operator_path,
            source_step_to_compact,
            step_phases,
            runtime["dtype"],
            scheduler_records,
            rank=rank,
        )
        operator_records_list.extend(records)
    operator_records = tuple(operator_records_list)

    raw_sources = [
        _raw_source(case_dir, meta_path, "case-metadata", "capture-metadata", None),
        _raw_source(case_dir, metrics_path, "serving-metrics", "serving-runtime", None),
        _raw_source(case_dir, scheduler_path, "scheduler", "vllm-scheduler-trace", None),
        _raw_source(
            case_dir,
            provenance_path,
            "trace-provenance",
            "vllm-operator-trace",
            None,
        ),
    ]
    for operator_path, rank in zip(operator_paths, operator_ranks, strict=True):
        raw_sources.append(
            _raw_source(
                case_dir,
                operator_path,
                f"operators-rank{rank}",
                "vllm-operator-trace",
                None,
            )
        )
    if form.request_contract == "arrival_schedule":
        cached_hit_evidence_source_ids = (
            "scheduler",
            "trace-provenance",
            "serving-metrics",
        )
        raw_sources.insert(
            1,
            _raw_source(
                case_dir,
                arrival_path,
                "arrival-schedule",
                "workload-generator",
                None,
            )
        )
    elif form.request_contract == "capture_log":
        cached_hit_evidence_source_ids = (
            "scheduler",
            "trace-provenance",
            "serving-metrics",
            "runtime-log",
        )
        capture_path = case_dir / "capture.log"
        runtime_version, _, _ = _ascend_capture_runtime(capture_path)
        raw_sources.append(
            _raw_source(
                case_dir,
                capture_path,
                "runtime-log",
                "vllm",
                runtime_version,
            )
        )
    else:
        raise AssertionError(f"unsupported hardware after validation: {hardware_id}")

    profiler_sources, launches, timing = _normalize_profiler(
        case_dir,
        hardware_id,
        raw_scheduler,
        scheduler_records,
        expected_ranks=tuple(range(tensor_parallel_size)),
        operator_records=operator_records,
    )
    raw_sources.extend(profiler_sources)
    raw_source_ids = {source["archive_id"] for source in raw_sources}
    if len(raw_source_ids) != len(raw_sources):
        raise ValueError("raw source archive IDs must be unique")
    actual_source_set_sha256 = raw_source_set_sha256(raw_sources)

    normalized_metrics = _normalize_metrics(metrics, metric_rows, requests)
    gaps = _provenance_gaps(runtime, raw_sources)
    expected_ranks = tuple(range(tensor_parallel_size))
    if model_id in _MOE_MODEL_TOPOLOGY:
        moe_num_experts, moe_topk, moe_num_layers = _MOE_MODEL_TOPOLOGY[model_id]
    else:
        moe_num_experts, moe_topk, moe_num_layers = None, None, None
    alignment_steps = _compact_alignment_steps(
        scheduler_records,
        operator_records,
        launches,
        expected_ranks,
        actual_source_set_sha256,
        moe_num_experts=moe_num_experts,
        moe_topk=moe_topk,
        moe_num_layers=moe_num_layers,
    )
    publish_metrics = model_id not in ("Llama-3-70B", "Llama-3.1-70B")
    return CompactCasePayload(
        identity={
            "source_kind": "sharegpt",
            "case_id": case_id,
            "hardware_id": hardware_id,
            "model_id": _canonical_model_id(model_id),
            "raw_model_id": model_id,
            "model_revision": None,
            "tokenizer_revision": None,
        },
        runtime=runtime,
        requests=requests,
        request_field_provenance={
            CANONICAL_REQUEST_COLUMNS[0]: {
                "kind": "requested",
                "source_archive_id": arrival_source_id,
                "reason": arrival_reason,
            },
            CANONICAL_REQUEST_COLUMNS[1]: {
                "kind": input_provenance_kind,
                "source_archive_id": input_source_id,
                "reason": None,
            },
            CANONICAL_REQUEST_COLUMNS[2]: {
                "kind": "observed_actual",
                "source_archive_id": "serving-metrics",
                "reason": None,
            },
            CANONICAL_REQUEST_COLUMNS[3]: {
                "kind": "scheduler_equivalent_representative",
                "source_archive_id": None,
                "reason": None,
            },
        },
        cached_hit_value_evidence={
            "kind": "scheduler_equivalent_representative",
            "logical_prefix_hit_observability": "unavailable",
            "representative_value": 0,
            "net_prefix_work_reduction_tokens": 0,
            "derivation": "full_scheduler_token_conservation",
            "source_archive_ids": cached_hit_evidence_source_ids,
            "raw_source_set_sha256": actual_source_set_sha256,
        },
        metrics=(
            WriterAvailabilityPayload("available", None, normalized_metrics)
            if publish_metrics
            else WriterAvailabilityPayload(
                "unavailable",
                "70B raw metrics were captured with a profiler and are not official ground truth",
                None,
            )
        ),
        alignment_steps=alignment_steps,
        raw_sources=tuple(raw_sources),
        provenance_gaps=tuple(gap["field"] + ": " + gap["reason"] for gap in gaps),
        converter={
            "implementation_id": _CONVERTER_IMPLEMENTATION,
            "source_commit": converter_source_commit,
            "selection_rule": "middle_complete_work_bearing_step_v1",
        },
    )


def _operator_rank(path: Path) -> int:
    match = re.search(r"\.rank(?P<rank>[0-9]+)$", path.name)
    if match is None:
        raise ValueError(f"operator trace has no rank suffix: {path.name}")
    return int(match.group("rank"))


def _canonical_model_id(raw_model_id: str) -> str:
    values = {
        "Qwen3-4B": "qwen3-4b",
        "Qwen3-8B": "qwen3-8b",
        "Llama-3.1-8B": "llama-3.1-8b",
        "Qwen3-30B-A3B": "qwen3-30b-a3b",
        "Qwen3-32B": "qwen3-32b",
        "Llama-3-70B": "llama-3-70b",
        "Llama-3.1-70B": "llama-3.1-70b",
    }
    if raw_model_id not in values:
        raise ValueError(f"unsupported raw model identity: {raw_model_id!r}")
    return values[raw_model_id]


def _middle_complete_step_by_phase(
    scheduler_records: Sequence[Mapping[str, object]],
    operator_step_ranks: set[tuple[str, int]],
    launch_step_ranks: set[tuple[str, int]],
    expected_ranks: tuple[int, ...],
) -> Mapping[str, Mapping[str, object] | None]:
    """Apply the one canonical middle-complete-step selection rule."""

    observed: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    eligible: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for execution_order, scheduler in enumerate(scheduler_records):
        scheduled_requests = _object_sequence(
            scheduler["scheduled_requests"], "scheduled_requests"
        )
        if not scheduled_requests:
            continue
        phase = _scheduler_step_phase(scheduler)
        observed[phase].append(scheduler)
        step_id = _nonempty_string(scheduler["step_id"], "Scheduler step_id")
        scheduled_request_idxs = {
            request["request_idx"] for request in scheduled_requests
        }
        finished_requests_are_self_contained = set(
            _sequence(scheduler["finished_request_idxs"], "finished_request_idxs")
        ).issubset(scheduled_request_idxs)
        if finished_requests_are_self_contained and all(
            (step_id, rank) in operator_step_ranks
            and (step_id, rank) in launch_step_ranks
            for rank in expected_ranks
        ):
            eligible[phase].append(
                {"execution_order": execution_order, "scheduler": scheduler}
            )
    selected = {}
    for phase in ("prefill", "decode", "mixed"):
        if not observed[phase]:
            selected[phase] = None
            continue
        candidates = eligible[phase]
        if not candidates:
            raise ValueError(
                f"observed phase {phase} has no complete all-rank operator/launch step"
            )
        selected[phase] = candidates[(len(candidates) - 1) // 2]
    return selected


def _compact_alignment_steps(
    scheduler_records: Sequence[Mapping[str, object]],
    operator_records: Sequence[Mapping[str, object]],
    launch_availability: AvailabilityPayload,
    expected_ranks: tuple[int, ...],
    source_set_sha256: str,
    *,
    moe_num_experts: int | None = None,
    moe_topk: int | None = None,
    moe_num_layers: int | None = None,
) -> Mapping[str, object]:
    """Select one middle complete step per observed phase without retaining time."""

    if launch_availability.status != "available":
        raise ValueError(f"real hardware launch evidence unavailable: {launch_availability.reason}")
    raw_launches = _sequence(launch_availability.payload, "launch payload")
    all_operator_signatures: dict[str, Mapping[str, object]] = {}
    operator_events: defaultdict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for record in operator_records:
        if record["record_type"] == "signature":
            attrs = {
                "shape": record["shape"],
                "dtype": record["dtype"],
                "variant": record["variant"],
                **dict(record["attrs"]),
            }
            signature = {"operation": record["operator_kind"], "attrs": attrs}
            all_operator_signatures[_canonical_sha256(signature)] = signature
        elif record["record_type"] == "event":
            continue
        else:
            raise ValueError("operator record_type must be signature or event")
    old_to_new_operator = {}
    for record in operator_records:
        if record["record_type"] == "signature":
            attrs = {
                "shape": record["shape"],
                "dtype": record["dtype"],
                "variant": record["variant"],
                **dict(record["attrs"]),
            }
            signature = {"operation": record["operator_kind"], "attrs": attrs}
            old_to_new_operator[record["signature_id"]] = _canonical_sha256(signature)
        elif record["record_type"] == "event":
            operator_events[(record["step_id"], record["rank"])].append(
                {
                    "order": record["sequence_idx"],
                    "signature_id": old_to_new_operator[record["signature_id"]],
                    "layer": record["layer_idx"],
                }
            )
        else:
            raise ValueError("operator record_type must be signature or event")

    all_launch_signatures: dict[str, Mapping[str, object]] = {}
    old_to_new_launch = {}
    launch_events: defaultdict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for record in raw_launches:
        if record["record_type"] != "signature":
            continue
        if record["geometry_kind"] == "cuda_dim3":
            geometry = {
                "kind": "cuda_dim3",
                "grid": record["grid"],
                "block": record["block"],
                "shared_memory_bytes": record["shared_memory_bytes"],
                "registers_per_thread": record["registers_per_thread"],
            }
        elif record["geometry_kind"] == "ascend_block_dim":
            geometry = {
                "kind": "ascend_block_dim",
                "block_dim": record["ascend_block_dim"],
                "mix_block_dim": record["ascend_mix_block_dim"],
            }
        elif record["geometry_kind"] == "unavailable":
            geometry = {"kind": "unavailable"}
        else:
            raise ValueError("real launch has unavailable or unsupported geometry")
        signature = {
            "kernel_name": record["kernel_name"],
            "geometry": geometry,
            "stream": record["stream"],
            "device": record["device"],
            "op_family": record["kernel_family"],
        }
        identifier = _canonical_sha256(signature)
        all_launch_signatures[identifier] = signature
        old_to_new_launch[record["signature_id"]] = identifier
    for record in raw_launches:
        if record["record_type"] == "event":
            launch_events[(record["step_id"], record["rank"])].append(
                {
                    "order": record["sequence_idx"],
                    "signature_id": old_to_new_launch[record["signature_id"]],
                }
            )

    selected_by_phase = _middle_complete_step_by_phase(
        scheduler_records,
        {
            (step_id, rank)
            for step_id, rank in operator_events
            if operator_events[(step_id, rank)]
        },
        {
            (step_id, rank)
            for step_id, rank in launch_events
            if launch_events[(step_id, rank)]
        },
        expected_ranks,
    )

    phases = {}
    selected_operator_signatures = {}
    selected_launch_signature_ids = set()
    source_ids = ["scheduler", "device-profiler"] + [
        f"operators-rank{rank}" for rank in expected_ranks
    ]
    if launch_availability.source_archive_ids is not None:
        source_ids = ["scheduler", *[f"operators-rank{rank}" for rank in expected_ranks], *launch_availability.source_archive_ids]
    source_ids = list(dict.fromkeys(source_ids))
    for phase in ("prefill", "decode", "mixed"):
        selected = selected_by_phase[phase]
        if selected is None:
            phases[phase] = {"status": "not_observed", "sample": None}
            continue
        scheduler = selected["scheduler"]
        step_id = scheduler["step_id"]
        operator_groups = []
        launch_groups = []
        operator_hashes = {}
        launch_hashes = {}
        for rank in expected_ranks:
            raw_operators = sorted(
                operator_events[(step_id, rank)], key=lambda item: item["order"]
            )
            operators = []
            for event in raw_operators:
                signature = _normalize_selected_operator_signature(
                    all_operator_signatures[event["signature_id"]],
                    moe_num_experts=moe_num_experts,
                    moe_topk=moe_topk,
                )
                signature_id = _canonical_sha256(signature)
                selected_operator_signatures[signature_id] = signature
                operators.append({**event, "signature_id": signature_id})
            launches = sorted(launch_events[(step_id, rank)], key=lambda item: item["order"])
            if [item["order"] for item in operators] != list(range(len(operators))):
                raise ValueError("operator order is not contiguous")
            if [item["order"] for item in launches] != list(range(len(launches))):
                raise ValueError("launch order is not contiguous")
            operator_groups.append({"rank": rank, "events": operators})
            launch_groups.append({"rank": rank, "events": launches})
            selected_launch_signature_ids.update(
                event["signature_id"] for event in launches
            )
            operator_hashes[str(rank)] = _canonical_sha256(operators)
            launch_hashes[str(rank)] = _canonical_sha256(launches)
        _validate_selected_moe_routing(
            operator_groups,
            selected_operator_signatures,
            expected_ranks=expected_ranks,
            moe_num_experts=moe_num_experts,
            moe_topk=moe_topk,
            moe_num_layers=moe_num_layers,
        )
        scheduled_requests = []
        for request in scheduler["scheduled_requests"]:
            scheduled_requests.append({
                "request_id": f"request-{request['request_idx']:012d}",
                "phase": request["phase"],
                "input_tokens": request["input_tokens"],
                "output_tokens": request["output_tokens"],
                "computed_tokens": request["computed_tokens"],
                "recomputed_tokens": request["recomputed_tokens"],
                "prefix_replay_tokens": request["prefix_replay_tokens"],
                "resumed_from_preemption": False,
                "trace_prefix_hit_tokens": 0,
                "scheduler_prefix_hit_tokens": 0,
                "prefix_scheduler_delta_tokens": 0,
                "raw_model_output_tokens": 0,
                "mtp_proposed_tokens": 0,
                "mtp_verified_tokens": 0,
                "mtp_accepted_tokens": 0,
                "mtp_rejected_tokens": 0,
                "mtp_proposer_tokens": 0,
                "mtp_next_proposed_tokens": 0,
                "mtp_proposer_mode": "disabled",
                "mtp_acceptance_source_kind": "disabled",
                "mtp_acceptance_source_id": "disabled",
            })
        scheduled_ids = {request["request_id"] for request in scheduled_requests}
        finished_ids = tuple(
            f"request-{request_idx:012d}" for request_idx in scheduler["finished_request_idxs"]
        )
        if not set(finished_ids).issubset(scheduled_ids):
            raise ValueError("finished Scheduler requests must execute in the selected step")
        normalized_scheduler = {
            "step_id": step_id,
            "pool_name": "main",
            "requests": scheduled_requests,
            "finished_request_ids": list(finished_ids),
            "preempted_request_ids": [],
            "attention_kv_layout": None,
        }
        phases[phase] = {
            "status": "available",
            "sample": {
                "source_step_id": step_id,
                "source_execution_order": selected["execution_order"],
                "scheduler": normalized_scheduler,
                "scheduler_field_provenance": _SCHEDULER_FIELD_PROVENANCE,
                "rank_ids": list(expected_ranks),
                "operators": operator_groups,
                "launches": launch_groups,
                "association": "step_order_only",
                "coverage": "complete",
                "provenance": {
                    "source_archive_ids": source_ids,
                    "raw_source_set_sha256": source_set_sha256,
                },
                "sequence_sha256": {
                    "operators": operator_hashes,
                    "launches": launch_hashes,
                },
            },
        }
    missing_launch_signatures = selected_launch_signature_ids - set(all_launch_signatures)
    if missing_launch_signatures:
        raise ValueError("selected launch event references a missing signature")
    operator_signatures = {
        identifier: selected_operator_signatures[identifier]
        for identifier in sorted(selected_operator_signatures)
    }
    launch_signatures = {
        identifier: all_launch_signatures[identifier]
        for identifier in sorted(selected_launch_signature_ids)
    }
    return {
        "operator_signatures": operator_signatures,
        "launch_signatures": launch_signatures,
        "phases": phases,
    }


def _normalize_selected_operator_signature(
    signature: Mapping[str, object],
    *,
    moe_num_experts: int | None,
    moe_topk: int | None,
) -> Mapping[str, object]:
    """Canonicalize selected MoE histograms using the frozen model topology."""

    if signature["operation"] != "moe":
        return signature
    if moe_num_experts is None or moe_topk is None:
        raise ValueError("selected MoE alignment step has no model topology")
    _integer(moe_num_experts, "MoE model num_experts", minimum=1)
    _integer(moe_topk, "MoE model topk", minimum=1)
    attrs = dict(_mapping(signature["attrs"], "selected MoE attrs"))
    shape = dict(_mapping(attrs["shape"], "selected MoE shape"))
    if "tok_per_expert" not in shape:
        raise ValueError(
            "selected MoE alignment step lacks tok_per_expert routing evidence"
        )
    histogram = _sequence(
        shape["tok_per_expert"], "selected MoE tok_per_expert"
    )
    if not histogram:
        raise ValueError("selected MoE tok_per_expert cannot be empty")
    counts = [
        _integer(value, "selected MoE tok_per_expert value", minimum=0)
        for value in histogram
    ]
    if len(counts) > moe_num_experts:
        raise ValueError(
            "selected MoE tok_per_expert exceeds the model expert count"
        )
    if "batch" in shape and "seq_len" in shape:
        tokens = (
            _integer(shape["batch"], "selected MoE batch", minimum=1)
            * _integer(shape["seq_len"], "selected MoE seq_len", minimum=1)
        )
    elif "tokens" in shape:
        tokens = _integer(shape["tokens"], "selected MoE tokens", minimum=1)
    else:
        raise ValueError("selected MoE shape has no token count")
    expected_assignments = tokens * moe_topk
    if sum(counts) != expected_assignments:
        raise ValueError(
            "selected MoE tok_per_expert does not conserve tokens * topk"
        )
    shape["tok_per_expert"] = counts + [0] * (moe_num_experts - len(counts))
    attrs["shape"] = shape
    return {"operation": "moe", "attrs": attrs}


def _validate_selected_moe_routing(
    operator_groups: Sequence[Mapping[str, object]],
    signatures: Mapping[str, Mapping[str, object]],
    *,
    expected_ranks: tuple[int, ...],
    moe_num_experts: int | None,
    moe_topk: int | None,
    moe_num_layers: int | None,
) -> None:
    """Require complete, replicated global routing evidence for selected steps."""

    if moe_num_experts is None:
        if moe_topk is not None or moe_num_layers is not None:
            raise ValueError("partial MoE model topology is invalid")
        return
    if moe_topk is None or moe_num_layers is None:
        raise ValueError("partial MoE model topology is invalid")
    expected_layers = set(range(
        _integer(moe_num_layers, "MoE model layer count", minimum=1)
    ))
    by_rank = {}
    for group in operator_groups:
        rank = group["rank"]
        layers = {}
        for event in group["events"]:
            signature = signatures[event["signature_id"]]
            if signature["operation"] != "moe":
                continue
            layer = event["layer"]
            if layer is None:
                raise ValueError("selected MoE operator has no layer index")
            shape = _mapping(signature["attrs"]["shape"], "selected MoE shape")
            histogram = tuple(shape["tok_per_expert"])
            if len(histogram) != moe_num_experts:
                raise ValueError("normalized MoE histogram width is inconsistent")
            if layer in layers:
                raise ValueError("selected MoE layer occurs more than once")
            layers[layer] = histogram
        if set(layers) != expected_layers:
            raise ValueError(
                "selected MoE routing must cover every model layer exactly"
            )
        by_rank[rank] = layers
    if set(by_rank) != set(expected_ranks):
        raise ValueError("selected MoE routing does not cover every expected rank")
    first = by_rank[expected_ranks[0]]
    if any(by_rank[rank] != first for rank in expected_ranks[1:]):
        raise ValueError("selected MoE routing histograms disagree across ranks")


def _validate_case_identity(
    case_dir: Path,
    hardware_id: str,
    model_id: str,
    target_qps: int,
) -> None:
    if hardware_id not in _SUPPORTED_HARDWARE:
        raise ValueError(f"unsupported first-phase hardware_id: {hardware_id!r}")
    if model_id not in _DENSE_MODELS:
        raise ValueError(f"unsupported first-phase model_id: {model_id!r}")
    if target_qps not in (4, 16, 64):
        raise ValueError(f"unsupported first-phase target_qps: {target_qps!r}")
    if not case_dir.is_dir():
        raise ValueError(f"raw ShareGPT case is not a directory: {case_dir}")
    match = _CASE_NAME.fullmatch(case_dir.name)
    if match is None:
        raise ValueError(f"raw case directory has non-canonical identity: {case_dir.name}")
    if match.group("model") != model_id or int(match.group("qps")) != target_qps:
        raise ValueError("raw case directory identity disagrees with requested model/QPS")


def _second_phase_a800_contract(
    model_id: str,
    target_qps: int,
) -> _ShareGPTCaseContract:
    key = (model_id, target_qps)
    if key not in _SECOND_PHASE_A800_CONTRACTS:
        raise ValueError(
            "unsupported second-phase A800 ShareGPT model/QPS: "
            f"{model_id!r}/{target_qps!r}"
        )
    return _SECOND_PHASE_A800_CONTRACTS[key]


def _validate_second_phase_case_identity(
    case_dir: Path,
    contract: _ShareGPTCaseContract,
) -> None:
    if not case_dir.is_dir():
        raise ValueError(f"raw ShareGPT case is not a directory: {case_dir}")
    if case_dir.name != contract.raw_directory_name:
        raise ValueError(
            "raw case directory identity disagrees with the second-phase case contract"
        )
    meta = _json_object(case_dir / "meta.json", "case metadata")
    provenance = _json_object(
        case_dir / "plugin_trace.json_provenance.json",
        "trace provenance",
    )
    _validate_meta(
        meta,
        contract.model_id,
        contract.target_qps,
        tensor_parallel_size=contract.tensor_parallel_size,
        expert_parallel_size=contract.expert_parallel_size,
    )
    if _nonempty_string(meta["case_id"], "meta case_id") != contract.raw_directory_name:
        raise ValueError("meta case_id disagrees with the second-phase case contract")
    if _nonempty_string(meta["workload"], "meta workload") != "sharegpt":
        raise ValueError("second-phase case metadata must identify a ShareGPT workload")
    _runtime_contract(
        meta,
        provenance,
        contract.model_id,
        contract.target_qps,
        128,
        expected_tensor_parallel_size=contract.tensor_parallel_size,
        expected_expert_parallel_size=contract.expert_parallel_size,
        expected_pipeline_parallel_size=contract.pipeline_parallel_size,
        expected_dtype=contract.dtype,
    )


def _validate_meta(
    meta: Mapping[str, object],
    model_id: str,
    target_qps: int,
    *,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
) -> None:
    _validate_raw_model_identity(meta["model"], model_id, "meta model")
    if _integer(meta["num_requests"], "meta num_requests", minimum=1) != 128:
        raise ValueError("ShareGPT case must contain exactly 128 requests")
    if _number(meta["qps"], "meta qps", positive=True) != float(target_qps):
        raise ValueError("meta qps disagrees with requested target QPS")
    actual_tp = _integer(
        meta["tensor_parallel_size"], "meta tensor parallel", minimum=1
    )
    if actual_tp != tensor_parallel_size:
        raise ValueError(
            f"ShareGPT case must use TP{tensor_parallel_size}, got TP{actual_tp}"
        )
    actual_ep = _integer(
        meta["expert_parallel_size"], "meta expert parallel", minimum=1
    )
    if actual_ep != expert_parallel_size:
        raise ValueError(
            f"ShareGPT case must use EP{expert_parallel_size}, got EP{actual_ep}"
        )


def _metric_rows(metrics: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    rows = _object_sequence(metrics["per_request"], "metrics per_request")
    if _integer(metrics["num_requests"], "metrics num_requests", minimum=1) != len(rows):
        raise ValueError("metrics num_requests disagrees with per_request length")
    if len(rows) != 128:
        raise ValueError("ShareGPT metrics must contain exactly 128 requests")
    return rows


def _arrival_times(
    arrival: Mapping[str, object],
    target_qps: int,
    request_count: int,
) -> tuple[float, ...]:
    values = _sequence(arrival["arrival_times_s"], "arrival_times_s")
    if len(values) != request_count:
        raise ValueError("arrival schedule length disagrees with metrics")
    if _integer(arrival["num_requests"], "arrival num_requests", minimum=1) != request_count:
        raise ValueError("arrival num_requests disagrees with metrics")
    if _number(arrival["qps"], "arrival qps", positive=True) != float(target_qps):
        raise ValueError("arrival qps disagrees with requested target QPS")
    result = tuple(
        _number(value, f"arrival_times_s[{index}]", positive=False)
        for index, value in enumerate(values)
    )
    if any(later < earlier for earlier, later in zip(result, result[1:])):
        raise ValueError("arrival schedule must be non-decreasing")
    return result


def _validate_metric_arrivals(
    metric_rows: Sequence[Mapping[str, object]],
    arrival_times: Sequence[float],
) -> None:
    for index, (row, expected) in enumerate(zip(metric_rows, arrival_times, strict=True)):
        actual = _number(row["arrival_s"], f"metrics arrival_s[{index}]", positive=False)
        if actual != expected:
            raise ValueError(
                f"metrics arrival_s disagrees with arrival schedule at request index {index}"
            )


def _metric_arrival_times(
    metric_rows: Sequence[Mapping[str, object]],
) -> tuple[float, ...]:
    """Read the requested arrival sequence retained by the Serving runtime."""

    result = tuple(
        _number(row["arrival_s"], f"metrics arrival_s[{index}]", positive=False)
        for index, row in enumerate(metric_rows)
    )
    if any(later < earlier for earlier, later in zip(result, result[1:])):
        raise ValueError("metrics arrival sequence must be non-decreasing")
    return result


def _explicit_request_contract(
    metric_rows: Sequence[Mapping[str, object]],
    scheduler: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
    meta: Mapping[str, object],
    model_id: str,
    target_qps: int,
    tensor_parallel_size: int,
    expert_parallel_size: int,
    pipeline_parallel_size: int,
    expected_dtype: str | None,
) -> tuple[dict[str, int], tuple[int, ...], dict[str, object]]:
    request_count = len(metric_rows)
    prompt_lengths = _sequence(provenance["prompt_lengths"], "provenance prompt_lengths")
    output_lengths = _sequence(provenance["output_lengths"], "provenance output_lengths")
    if len(prompt_lengths) != request_count or len(output_lengths) != request_count:
        raise ValueError("trace provenance request lengths disagree with metrics")
    mapping: dict[str, int] = {}
    for index, row in enumerate(metric_rows):
        source_id = _nonempty_string(row["request_id"], f"metrics request_id[{index}]")
        match = _REQUEST_ID.fullmatch(source_id)
        if match is None or int(match.group("index")) != index:
            raise ValueError("explicit metrics request IDs must be physical-order req_N")
        mapping[source_id] = index
        metric_output = _integer(
            row["n_output_tokens"], f"metrics output[{index}]", minimum=0
        )
        if _integer(output_lengths[index], "provenance output length", minimum=0) != metric_output:
            raise ValueError("trace provenance output lengths disagree with metrics")
    _validate_scheduler_request_set(scheduler, set(mapping))
    inputs = tuple(
        _integer(value, f"prompt_lengths[{index}]", minimum=1)
        for index, value in enumerate(prompt_lengths)
    )
    _validate_full_scheduler_token_conservation(
        scheduler, mapping, inputs, metric_rows
    )
    runtime = _runtime_contract(
        meta,
        provenance,
        model_id,
        target_qps,
        len(metric_rows),
        expected_tensor_parallel_size=tensor_parallel_size,
        expected_expert_parallel_size=expert_parallel_size,
        expected_pipeline_parallel_size=pipeline_parallel_size,
        expected_dtype=expected_dtype,
    )
    return mapping, inputs, runtime


def _ordinal_request_contract(
    metric_rows: Sequence[Mapping[str, object]],
    scheduler: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
    meta: Mapping[str, object],
    model_id: str,
    target_qps: int,
    tensor_parallel_size: int | None = None,
    expert_parallel_size: int | None = None,
    pipeline_parallel_size: int | None = None,
    expected_dtype: str | None = None,
) -> tuple[dict[str, int], tuple[int, ...], dict[str, object]]:
    first_appearance: list[str] = []
    seen = set()
    for record in scheduler:
        scheduled = _mapping(record["scheduled"], "Scheduler scheduled")
        for source_id, token_value in scheduled.items():
            _nonempty_string(source_id, "Scheduler request ID")
            _integer(token_value, "Scheduler scheduled tokens", minimum=0)
            if source_id.startswith(_WARMUP_PREFIX):
                continue
            if source_id not in seen:
                seen.add(source_id)
                first_appearance.append(source_id)
    if len(first_appearance) != len(metric_rows):
        raise ValueError(
            "ordinal-derived mapping requires Scheduler first-appearance count "
            "to equal metrics request count"
        )
    mapping = {source_id: index for index, source_id in enumerate(first_appearance)}
    prompt_lengths = _sequence(provenance["prompt_lengths"], "provenance prompt_lengths")
    output_lengths = _sequence(provenance["output_lengths"], "provenance output_lengths")
    if len(prompt_lengths) != len(metric_rows) or len(output_lengths) != len(metric_rows):
        raise ValueError("trace provenance request lengths disagree with metrics")
    inputs = tuple(
        _integer(value, f"prompt_lengths[{index}]", minimum=1)
        for index, value in enumerate(prompt_lengths)
    )
    for index, output_length in enumerate(output_lengths):
        metric_output = _integer(
            metric_rows[index]["n_output_tokens"],
            f"metrics output[{index}]",
            minimum=0,
        )
        if _integer(
            output_length,
            f"provenance output length[{index}]",
            minimum=0,
        ) != metric_output:
            raise ValueError("trace provenance output lengths disagree with metrics")
    _validate_scheduler_request_set(scheduler, set(mapping))
    _validate_full_scheduler_token_conservation(
        scheduler, mapping, inputs, metric_rows
    )
    runtime = _runtime_contract(
        meta,
        provenance,
        model_id,
        target_qps,
        len(metric_rows),
        expected_tensor_parallel_size=tensor_parallel_size,
        expected_expert_parallel_size=expert_parallel_size,
        expected_pipeline_parallel_size=pipeline_parallel_size,
        expected_dtype=expected_dtype,
    )
    return mapping, inputs, runtime


def _validate_scheduler_request_set(
    scheduler: Sequence[Mapping[str, object]],
    expected: set[str],
) -> None:
    observed = set()
    finished = set()
    for record in scheduler:
        scheduled = _mapping(record["scheduled"], "Scheduler scheduled")
        observed.update(
            source_id
            for source_id in scheduled
            if not source_id.startswith(_WARMUP_PREFIX)
        )
        finished_values = _sequence(record["finished"], "Scheduler finished")
        finished.update(
            _nonempty_string(value, "Scheduler finished request ID")
            for value in finished_values
            if not str(value).startswith(_WARMUP_PREFIX)
        )
    if observed != expected:
        raise ValueError("Scheduler request set disagrees with normalized request mapping")
    if finished != expected:
        raise ValueError("Scheduler finished-request set is not complete")


def _validate_full_scheduler_token_conservation(
    scheduler: Sequence[Mapping[str, object]],
    mapping: Mapping[str, int],
    input_tokens: Sequence[int],
    metric_rows: Sequence[Mapping[str, object]],
) -> None:
    totals: defaultdict[str, int] = defaultdict(int)
    for record in scheduler:
        scheduled = _mapping(record["scheduled"], "Scheduler scheduled")
        for source_id, token_value in scheduled.items():
            if source_id in mapping:
                totals[source_id] += _integer(
                    token_value, "Scheduler scheduled tokens", minimum=0
                )
    for source_id, request_idx in mapping.items():
        output_tokens = _integer(
            metric_rows[request_idx]["n_output_tokens"],
            "metrics output tokens",
            minimum=0,
        )
        expected = input_tokens[request_idx] + output_tokens - 1
        if totals[source_id] != expected:
            raise ValueError(
                "full Scheduler token conservation failed: cumulative scheduled "
                "tokens do not equal prompt + actual output - 1"
            )


def _runtime_contract(
    meta: Mapping[str, object],
    provenance: Mapping[str, object] | None,
    model_id: str,
    target_qps: int,
    request_count: int,
    *,
    expected_tensor_parallel_size: int | None = None,
    expected_expert_parallel_size: int | None = None,
    expected_pipeline_parallel_size: int | None = None,
    expected_dtype: str | None = None,
) -> dict[str, object]:
    tensor_parallel_size = _integer(
        meta["tensor_parallel_size"], "meta tensor_parallel_size", minimum=1
    )
    expert_parallel_size = _integer(
        meta["expert_parallel_size"], "meta expert_parallel_size", minimum=1
    )
    if (
        expected_tensor_parallel_size is not None
        and tensor_parallel_size != expected_tensor_parallel_size
    ):
        raise ValueError("metadata TP disagrees with the requested case contract")
    if (
        expected_expert_parallel_size is not None
        and expert_parallel_size != expected_expert_parallel_size
    ):
        raise ValueError("metadata EP disagrees with the requested case contract")
    dtype = None
    pipeline_parallel_size = None
    if provenance is not None:
        _validate_raw_model_identity(
            provenance["model"],
            model_id,
            "trace provenance model",
        )
        if _number(provenance["qps"], "provenance qps", positive=True) != float(target_qps):
            raise ValueError("trace provenance qps disagrees with case QPS")
        if (
            _integer(provenance["num_requests"], "provenance requests", minimum=1)
            != request_count
        ):
            raise ValueError("trace provenance request count disagrees with metrics")
        if _integer(provenance["tensor_parallel_size"], "provenance TP", minimum=1) != tensor_parallel_size:
            raise ValueError("trace provenance TP disagrees with metadata")
        if _integer(provenance["expert_parallel_size"], "provenance EP", minimum=1) != expert_parallel_size:
            raise ValueError("trace provenance EP disagrees with metadata")
        pipeline_parallel_size = _integer(
            provenance["pipeline_parallel_size"], "provenance PP", minimum=1
        )
        if (
            expected_pipeline_parallel_size is not None
            and pipeline_parallel_size != expected_pipeline_parallel_size
        ):
            raise ValueError("provenance PP disagrees with the requested case contract")
        engine_kwargs = _mapping(provenance["engine_kwargs"], "provenance engine_kwargs")
        dtype = _nonempty_string(engine_kwargs["dtype"], "provenance dtype")
        if expected_dtype is not None:
            if expected_tensor_parallel_size is None:
                raise AssertionError("strict dtype contract requires an expected TP")
            engine_tp = _integer(
                engine_kwargs["tensor_parallel_size"],
                "provenance engine tensor_parallel_size",
                minimum=1,
            )
            if engine_tp != tensor_parallel_size:
                raise ValueError("provenance engine TP disagrees with metadata")
            engine_pp = _integer(
                engine_kwargs["pipeline_parallel_size"],
                "provenance engine pipeline_parallel_size",
                minimum=1,
            )
            if engine_pp != pipeline_parallel_size:
                raise ValueError("provenance engine PP disagrees with provenance")
            if (
                expected_pipeline_parallel_size is not None
                and engine_pp != expected_pipeline_parallel_size
            ):
                raise ValueError("provenance engine PP disagrees with the requested case contract")
        if expected_dtype is not None and dtype != expected_dtype:
            raise ValueError(
                "provenance dtype disagrees with the requested case contract"
            )
        if expected_dtype is not None:
            if expected_expert_parallel_size is None:
                raise AssertionError("strict dtype contract requires an expected EP")
            if expected_expert_parallel_size > 1:
                if engine_kwargs["enable_expert_parallel"] is not True:
                    raise ValueError("TP2+EP2 case must enable engine expert parallelism")
            elif "enable_expert_parallel" in engine_kwargs:
                if engine_kwargs["enable_expert_parallel"] is not False:
                    raise ValueError("EP1 case cannot enable engine expert parallelism")
    return {
        "dtype": dtype,
        "deployment": None,
        "tensor_parallel_size": tensor_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "target_qps": float(target_qps),
        "seed": None,
        "scheduler_config_sha256": None,
        "runtime_version": None,
        "source_commit": None,
        "capture_enable_prefix_caching": None,
    }


def _validate_raw_model_identity(
    raw_model: object,
    model_id: str,
    description: str,
) -> None:
    if model_id not in _RAW_MODEL_BASENAME_BY_MODEL_ID:
        raise ValueError(f"unsupported first-phase model_id: {model_id!r}")
    actual_basename = Path(_nonempty_string(raw_model, description)).name
    expected_basename = _RAW_MODEL_BASENAME_BY_MODEL_ID[model_id]
    if actual_basename != expected_basename:
        raise ValueError(f"{description} path disagrees with case model")


def _normalize_scheduler(
    raw_records: Sequence[Mapping[str, object]],
    request_mapping: Mapping[str, int],
    input_tokens: Sequence[int],
) -> tuple[tuple[Mapping[str, object], ...], dict[object, str], dict[str, str]]:
    normalized = []
    source_step_to_compact: dict[object, str] = {}
    step_phases: dict[str, str] = {}
    cumulative = [0 for _ in input_tokens]
    first_timestamp = None
    seen_iterations = set()
    previous_timestamp = None
    for sequence_idx, raw in enumerate(raw_records):
        source_iter = raw["iter_id"]
        if isinstance(source_iter, bool) or not isinstance(source_iter, (int, str)):
            raise TypeError("Scheduler iter_id must be an integer or string")
        if source_iter in seen_iterations:
            raise ValueError("Scheduler iter_id values must be unique")
        seen_iterations.add(source_iter)
        timestamp = _integer(raw["ts_us"], "Scheduler ts_us", minimum=0)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("Scheduler timestamps must strictly increase")
        previous_timestamp = timestamp
        if first_timestamp is None:
            first_timestamp = timestamp
        scheduled = _mapping(raw["scheduled"], "Scheduler scheduled")
        scheduled_requests = []
        phases = set()
        for source_id, token_value in scheduled.items():
            tokens = _integer(token_value, "Scheduler scheduled tokens", minimum=0)
            if source_id.startswith(_WARMUP_PREFIX):
                continue
            if tokens == 0:
                raise ValueError("work-bearing Scheduler request cannot schedule zero tokens")
            if source_id not in request_mapping:
                raise ValueError("Scheduler references an unmapped request ID")
            request_idx = request_mapping[source_id]
            computed_before = cumulative[request_idx]
            phase = "prefill" if computed_before < input_tokens[request_idx] else "decode"
            if phase == "prefill":
                remaining_input = input_tokens[request_idx] - computed_before
                if tokens > remaining_input:
                    raise ValueError("Scheduler step crosses the prefill/decode boundary")
                scheduled_input_tokens = tokens
                scheduled_output_tokens = 0
            elif phase == "decode":
                scheduled_input_tokens = 0
                scheduled_output_tokens = tokens
            else:
                raise AssertionError(f"unsupported derived request phase: {phase!r}")
            cumulative[request_idx] += tokens
            phases.add(phase)
            scheduled_requests.append(
                {
                    "request_idx": request_idx,
                    "scheduled_tokens": tokens,
                    "phase": phase,
                    "input_tokens": scheduled_input_tokens,
                    "output_tokens": scheduled_output_tokens,
                    "computed_tokens": computed_before,
                    "recomputed_tokens": 0,
                    "prefix_replay_tokens": 0,
                }
            )
        finished_values = _sequence(raw["finished"], "Scheduler finished")
        finished = []
        for value in finished_values:
            source_id = _nonempty_string(value, "Scheduler finished request ID")
            if source_id.startswith(_WARMUP_PREFIX):
                continue
            if source_id not in request_mapping:
                raise ValueError("Scheduler finished list references an unmapped request ID")
            finished.append(request_mapping[source_id])
        source_total = _integer(raw["total_tokens"], "Scheduler total_tokens", minimum=0)
        if source_total != sum(
            _integer(value, "Scheduler scheduled value", minimum=0)
            for value in scheduled.values()
        ):
            raise ValueError("Scheduler total_tokens disagrees with scheduled values")
        step_id = f"step-{sequence_idx:012d}"
        if source_iter in source_step_to_compact:
            raise AssertionError("duplicate Scheduler source step after uniqueness check")
        source_step_to_compact[source_iter] = step_id
        phase = "mixed" if len(phases) > 1 else next(iter(phases), "mixed")
        step_phases[step_id] = phase
        normalized.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": f"scheduler-{sequence_idx:012d}",
                "step_id": step_id,
                "source_iter_id": source_iter,
                "scheduler_time_us": timestamp - first_timestamp,
                "pool": None,
                "total_scheduled_tokens": sum(
                    request["scheduled_tokens"] for request in scheduled_requests
                ),
                "scheduled_requests": scheduled_requests,
                "finished_request_idxs": finished,
                "preempted_request_idxs": None,
                "field_availability": {
                    "source_iter_id": "available",
                    "scheduler_time_us": "available",
                    "pool": "unavailable",
                    "total_scheduled_tokens": "available",
                    "finished_request_idxs": "available",
                    "preempted_request_idxs": "unavailable",
                    "scheduled_requests.computed_tokens": "unavailable",
                    "scheduled_requests.recomputed_tokens": "unavailable",
                    "scheduled_requests.prefix_replay_tokens": "unavailable",
                },
            }
        )
    if any(total <= 0 for total in cumulative):
        raise ValueError("every normalized request must appear in Scheduler records")
    return tuple(normalized), source_step_to_compact, step_phases


def _normalize_operators(
    path: Path,
    source_step_to_compact: Mapping[object, str],
    step_phases: Mapping[str, str],
    runtime_dtype: object,
    scheduler_records: Sequence[Mapping[str, object]],
    *,
    rank: int = 0,
) -> tuple[tuple[Mapping[str, object], ...], str]:
    """Normalize logical operators while deferring selected-step checks."""
    dtype = (
        None
        if runtime_dtype is None
        else _nonempty_string(runtime_dtype, "operator runtime dtype")
    )
    signatures: dict[str, Mapping[str, object]] = {}
    events = []
    next_sequence: dict[tuple[int, str], int] = {}
    covered_steps = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"operator JSONL contains a blank line at {line_number}")
            raw = _mapping(_json_loads(line, f"operator line {line_number}"), "operator record")
            base_keys = {"step_id", "op_idx", "op_type", "position", "layer_idx", "attrs"}
            if set(raw) not in (base_keys, base_keys | {"ts_us"}):
                raise ValueError(
                    f"operator line {line_number} has unsupported keys: {tuple(raw)}"
                )
            if "ts_us" in raw:
                _number(raw["ts_us"], "operator ts_us", positive=False)
            source_step = raw["step_id"]
            if source_step not in source_step_to_compact:
                raise ValueError("operator record references an unknown Scheduler source step")
            step_id = source_step_to_compact[source_step]
            scheduler_record = scheduler_records[int(step_id.split("-")[1])]
            if not scheduler_record["scheduled_requests"]:
                continue
            sequence_key = (rank, step_id)
            if sequence_key not in next_sequence:
                next_sequence[sequence_key] = 0
            sequence_idx = next_sequence[sequence_key]
            raw_op_idx = _integer(raw["op_idx"], "operator op_idx", minimum=0)
            if raw_op_idx != sequence_idx:
                raise ValueError("operator op_idx must be contiguous within each Scheduler step")
            next_sequence[sequence_key] = sequence_idx + 1
            op_type = _nonempty_string(raw["op_type"], "operator op_type")
            position = _nonempty_or_null(raw["position"], "operator position")
            attrs = dict(_mapping(raw["attrs"], "operator attrs"))
            raw_layer = _integer(raw["layer_idx"], "operator layer_idx", minimum=-1)
            layer_idx = raw_layer if raw_layer >= 0 else None
            signature_payload = {
                "operator_kind": op_type,
                "shape": attrs,
                "dtype": dtype,
                "variant": position,
                "attrs": {},
            }
            signature_id = _canonical_sha256(signature_payload)
            signatures[signature_id] = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "signature",
                "signature_id": signature_id,
                **signature_payload,
            }
            event_id = f"operator-r{rank}-s{step_id[5:]}-n{sequence_idx:08d}"
            events.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "event",
                    "operator_id": event_id,
                    "signature_id": signature_id,
                    "rank": rank,
                    "step_id": step_id,
                    "phase": step_phases[step_id],
                    "layer_idx": layer_idx,
                    "sequence_idx": sequence_idx,
                    "request_idxs": [],
                    "link_status": "step_only",
                }
            )
            covered_steps.add(step_id)
    if not events:
        raise ValueError("operator trace contains no work-bearing records")
    work_steps = {
        record["step_id"]
        for record in scheduler_records
        if record["scheduled_requests"]
    }
    coverage = "complete" if work_steps.issubset(covered_steps) else "partial"
    signature_records = tuple(signatures[key] for key in sorted(signatures))
    return signature_records + tuple(events), coverage


def _normalize_metrics(
    metrics: Mapping[str, object],
    metric_rows: Sequence[Mapping[str, object]],
    requests: Sequence[CanonicalRequestRow],
) -> dict[str, object]:
    wall_seconds = _number(metrics["wall_s"], "metrics wall_s", positive=True)
    throughput = _number(
        metrics["throughput_rps"], "metrics throughput_rps", positive=False
    )
    per_request = []
    for request_idx, (raw, request) in enumerate(zip(metric_rows, requests, strict=True)):
        output_count = _integer(
            raw["n_output_tokens"], "metrics n_output_tokens", minimum=0
        )
        ttft = _optional_number(raw["ttft_ms"], "metrics ttft_ms")
        tpot = _optional_number(raw["tpot_ms"], "metrics tpot_ms")
        e2e = _number(raw["e2e_ms"], "metrics e2e_ms", positive=False)
        if output_count == 0:
            ttft = None
            tpot = None
        elif output_count == 1:
            if ttft is None:
                raise ValueError("one-output request requires measured TTFT")
            tpot = 0.0
        elif ttft is None or tpot is None:
            raise ValueError("multi-output request requires measured TTFT and TPOT")
        per_request.append(
            {
                "request_idx": request_idx,
                "actual_output_tokens": output_count,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "e2e_latency_ms": e2e,
            }
        )
    output_total = sum(request.output_tokens for request in requests)
    token_total = sum(
        request.input_tokens + request.output_tokens for request in requests
    )
    return {
        "metric_provenance": {
            "source_kind": "serving_runtime",
            "source_archive_id": "serving-metrics",
            "source_sha256": _sha256_file(Path(metrics["__source_path"])),
            "profiler_attached": False,
        },
        "per_request": per_request,
        "summary": {
            "request_count": len(per_request),
            "request_throughput_per_s": throughput,
            "output_token_throughput_per_s": output_total / wall_seconds,
            "total_token_throughput_per_s": token_total / wall_seconds,
        },
    }


def _ascend_rank_profiles(
    case_dir: Path,
    expected_ranks: tuple[int, ...],
) -> tuple[_AscendRankProfile, ...]:
    """Bind one msprof database to every rank using captured worker identity."""

    if expected_ranks != tuple(range(len(expected_ranks))) or not expected_ranks:
        raise ValueError("910B expected ranks must be non-empty and contiguous")
    capture_path = case_dir / "capture.log"
    text = capture_path.read_text(encoding="utf-8")
    worker_by_pid = {}
    for match in _ASCEND_WORKER_RANK.finditer(text):
        pid = int(match.group("pid"))
        identity = (
            int(match.group("world_size")),
            int(match.group("rank")),
            int(match.group("local_rank")),
        )
        if pid in worker_by_pid and worker_by_pid[pid] != identity:
            raise ValueError("910B capture maps one worker PID to conflicting ranks")
        worker_by_pid[pid] = identity
    databases = tuple(sorted(case_dir.glob("msprof/PROF_*/msprof_*.db")))
    if len(databases) != len(expected_ranks):
        raise ValueError("910B case must contain exactly one msprof database per rank")
    has_worker_rank_evidence = bool(worker_by_pid)
    if has_worker_rank_evidence:
        if len(worker_by_pid) != len(expected_ranks):
            raise ValueError("910B capture must identify exactly one worker PID per rank")
        if {value[0] for value in worker_by_pid.values()} != {len(expected_ranks)}:
            raise ValueError("910B capture world_size disagrees with expected ranks")
        if {value[1] for value in worker_by_pid.values()} != set(expected_ranks):
            raise ValueError("910B capture does not identify every expected global rank")
        if {value[2] for value in worker_by_pid.values()} != set(expected_ranks):
            raise ValueError("910B capture does not identify every expected local rank")
    elif expected_ranks != (0,):
        raise ValueError("multi-rank 910B capture contains no worker PID-to-rank evidence")
    profiles_by_rank = {}
    for database in databases:
        profile_dir = database.parent
        info_paths = tuple(
            path
            for path in sorted(profile_dir.glob("device_*/info.json.*"))
            if not path.name.endswith(".done")
        )
        if len(info_paths) != 1:
            raise ValueError("each 910B msprof database requires exactly one device info")
        info_path = info_paths[0]
        info = _json_object(info_path, "910B PROF info")
        _nonempty_string(info["version"], "910B PROF schema version")
        pid_text = _nonempty_string(info["pid"], "910B PROF worker PID")
        if not pid_text.isdecimal():
            raise ValueError("910B PROF worker PID must be a decimal integer")
        pid = int(pid_text)
        if has_worker_rank_evidence:
            if pid not in worker_by_pid:
                raise ValueError("910B PROF worker PID has no capture rank evidence")
            world_size, rank, local_rank = worker_by_pid[pid]
            if world_size != len(expected_ranks):
                raise AssertionError("910B worker world size was validated")
        else:
            rank = 0
            local_rank = 0
        device_info = _sequence(info["DeviceInfo"], "910B PROF DeviceInfo")
        if len(device_info) != 1:
            raise ValueError("910B PROF info must identify exactly one device")
        device_id = _integer(
            _mapping(device_info[0], "910B PROF device")["id"],
            "910B PROF device ID",
            minimum=0,
        )
        if info_path.parent.name != f"device_{device_id}":
            raise ValueError("910B PROF device directory disagrees with device ID")
        if info_path.name != f"info.json.{device_id}":
            raise ValueError("910B PROF info filename disagrees with device ID")
        if info["devices"] != str(device_id):
            raise ValueError("910B PROF devices field disagrees with device ID")
        with sqlite3.connect(
            f"file:{database}?mode=ro&immutable=1", uri=True
        ) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            compute_count = (
                connection.execute(
                    "SELECT COUNT(*) FROM COMPUTE_TASK_INFO"
                ).fetchone()[0]
                if "COMPUTE_TASK_INFO" in tables
                else None
            )
            task_count = (
                connection.execute("SELECT COUNT(*) FROM TASK").fetchone()[0]
                if "TASK" in tables
                else None
            )
        if {"TASK", "COMPUTE_TASK_INFO", "STRING_IDS"}.issubset(tables):
            if task_count is None or task_count <= 0:
                raise ValueError("910B native msprof TASK table is empty")
            recovery_paths = None
        elif (
            "COMPUTE_TASK_INFO" in tables
            and compute_count is not None
            and compute_count > 0
            and "STRING_IDS" not in tables
            and ("TASK" not in tables or task_count == 0)
        ):
            recovery_paths = (
                info_path.parent / "sqlite" / "ascend_task.db",
                profile_dir / "host" / "sqlite" / "ge_info.db",
                profile_dir / "host" / "sqlite" / "runtime.db",
                info_path.parent / f"start_info.{device_id}",
            )
            missing = tuple(path for path in recovery_paths if not path.is_file())
            if missing:
                raise FileNotFoundError(
                    f"910B intermediate recovery is missing raw sources: {missing}"
                )
        else:
            raise ValueError(
                "910B msprof database has an unsupported per-invocation table shape"
            )
        if rank in profiles_by_rank:
            raise ValueError("910B case contains duplicate msprof evidence for one rank")
        profiles_by_rank[rank] = _AscendRankProfile(
            rank,
            local_rank,
            device_id,
            database,
            info_path,
            recovery_paths,
        )
    if set(profiles_by_rank) != set(expected_ranks):
        raise ValueError("910B msprof evidence does not cover every expected rank")
    return tuple(profiles_by_rank[rank] for rank in expected_ranks)


def _require_sqlite_columns(
    connection: sqlite3.Connection,
    table: str,
    required: frozenset[str],
    description: str,
) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if table not in tables:
        raise ValueError(f"{description} is missing table {table}")
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    missing = required - columns
    if missing:
        raise ValueError(f"{description} {table} is missing columns {sorted(missing)}")


def _ascend_clock_offset_ns(start_info_path: Path) -> int:
    payload = _json_object(start_info_path, "910B device start_info")
    monotonic = _nonempty_string(
        payload["clockMonotonicRaw"], "910B clockMonotonicRaw"
    )
    epoch_us = _nonempty_string(
        payload["collectionTimeBegin"], "910B collectionTimeBegin"
    )
    if not monotonic.isdecimal() or not epoch_us.isdecimal():
        raise ValueError("910B start_info clocks must be decimal integers")
    offset_ns = int(epoch_us) * 1000 - int(monotonic)
    if offset_ns <= 0:
        raise ValueError("910B start_info produces a non-positive clock offset")
    return offset_ns


def _rebuild_ascend_task_database(
    profile: _AscendRankProfile,
    output_path: Path,
) -> None:
    if profile.recovery_paths is None:
        raise AssertionError("native 910B profiles do not require recovery")
    ascend_task_path, ge_info_path, runtime_path, start_info_path = (
        profile.recovery_paths
    )
    clock_offset_ns = _ascend_clock_offset_ns(start_info_path)
    with sqlite3.connect(
        f"file:{profile.database}?mode=ro&immutable=1", uri=True
    ) as consolidated:
        expected_count = consolidated.execute(
            "SELECT COUNT(*) FROM COMPUTE_TASK_INFO"
        ).fetchone()[0]
    if expected_count <= 0:
        raise ValueError("910B recovery requires non-empty COMPUTE_TASK_INFO evidence")

    ge = sqlite3.connect(f"file:{ge_info_path}?mode=ro&immutable=1", uri=True)
    runtime = sqlite3.connect(f"file:{runtime_path}?mode=ro&immutable=1", uri=True)
    device = sqlite3.connect(
        f"file:{ascend_task_path}?mode=ro&immutable=1", uri=True
    )
    recovered = sqlite3.connect(output_path)
    try:
        _require_sqlite_columns(
            ge,
            "TaskInfo",
            frozenset(
                {
                    "op_name", "stream_id", "task_id", "block_dim",
                    "mix_block_dim", "task_type", "op_type", "thread_id",
                    "timestamp",
                }
            ),
            "910B ge_info",
        )
        _require_sqlite_columns(
            runtime,
            "HostTask",
            frozenset(
                {
                    "stream_id", "task_id", "device_id", "timestamp",
                    "connection_id", "thread_id",
                }
            ),
            "910B runtime",
        )
        _require_sqlite_columns(
            device,
            "AscendTask",
            frozenset(
                {
                    "stream_id", "task_id", "start_time", "duration",
                    "device_task_type", "connection_id",
                }
            ),
            "910B ascend_task",
        )
        source_count = ge.execute("SELECT COUNT(*) FROM TaskInfo").fetchone()[0]
        if source_count != expected_count:
            raise ValueError(
                "910B TaskInfo count disagrees with retained COMPUTE_TASK_INFO count"
            )
        recovered.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE BINDING(
                globalTaskId INTEGER PRIMARY KEY,
                connectionId INTEGER NOT NULL UNIQUE,
                streamId INTEGER NOT NULL,
                taskId INTEGER NOT NULL,
                name TEXT NOT NULL,
                blockDim INTEGER NOT NULL,
                mixBlockDim INTEGER,
                taskType TEXT NOT NULL,
                opType TEXT NOT NULL
            );
            """
        )
        task_rows = ge.execute(
            "SELECT rowid, op_name, stream_id, task_id, block_dim, mix_block_dim, "
            "task_type, op_type, thread_id, timestamp FROM TaskInfo "
            "ORDER BY timestamp, rowid"
        )
        host_rows = iter(
            runtime.execute(
                "SELECT rowid, stream_id, task_id, device_id, timestamp, "
                "connection_id, thread_id FROM HostTask ORDER BY timestamp, rowid"
            )
        )
        host_row = next(host_rows, None)
        host_timestamp = None
        host_group = []
        previous_task_timestamp = None
        inserted = 0
        batch = []
        for row in task_rows:
            timestamp = _integer(row[9], "910B TaskInfo timestamp", minimum=0)
            if previous_task_timestamp is not None and timestamp < previous_task_timestamp:
                raise ValueError("910B TaskInfo timestamps are not monotonic")
            previous_task_timestamp = timestamp
            while host_timestamp is None or host_timestamp < timestamp:
                if host_row is None:
                    raise ValueError("910B HostTask ends before TaskInfo")
                host_timestamp = _integer(
                    host_row[4], "910B HostTask timestamp", minimum=0
                )
                host_group = []
                while host_row is not None and host_row[4] == host_timestamp:
                    host_group.append(host_row)
                    host_row = next(host_rows, None)
            if host_timestamp != timestamp:
                raise ValueError("910B TaskInfo has no exact HostTask timestamp")
            stream_id = _integer(row[2], "910B TaskInfo stream_id", minimum=0)
            task_id = _integer(row[3], "910B TaskInfo task_id", minimum=0)
            thread_id = _integer(row[8], "910B TaskInfo thread_id", minimum=0)
            matches = tuple(
                item
                for item in host_group
                if item[1] == stream_id
                and item[2] == task_id
                and item[6] == thread_id
            )
            bindings = {(item[5], item[3]) for item in matches}
            if len(bindings) != 1:
                raise ValueError(
                    "910B TaskInfo does not have one unique HostTask launch binding"
                )
            connection_id, device_id = next(iter(bindings))
            if device_id != profile.device_id:
                raise ValueError("910B HostTask device disagrees with PROF metadata")
            name = _nonempty_string(row[1], "910B TaskInfo op_name")
            block_dim = _integer(row[4], "910B TaskInfo block_dim", minimum=0)
            mix_block_dim = (
                None
                if row[5] is None
                else _integer(row[5], "910B TaskInfo mix_block_dim", minimum=0)
            )
            task_type = _nonempty_string(row[6], "910B TaskInfo task_type")
            op_type = _nonempty_string(row[7], "910B TaskInfo op_type")
            if task_type not in {
                "AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "AI_CPU",
                "DSA_SQE",
            }:
                raise ValueError(f"unsupported 910B TaskInfo task_type: {task_type!r}")
            batch.append(
                (
                    row[0], connection_id, stream_id, task_id, name, block_dim,
                    mix_block_dim, task_type, op_type,
                )
            )
            if len(batch) == 10000:
                recovered.executemany(
                    "INSERT INTO BINDING VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
                )
                inserted += len(batch)
                batch.clear()
        if batch:
            recovered.executemany(
                "INSERT INTO BINDING VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
            )
            inserted += len(batch)
        if inserted != expected_count:
            raise ValueError("910B recovery did not bind every TaskInfo row")
        recovered.execute(
            "ATTACH DATABASE ? AS device_raw", (str(ascend_task_path),)
        )
        invalid_device_times = recovered.execute(
            "SELECT COUNT(*) FROM device_raw.AscendTask WHERE start_time>=0 "
            "AND duration<0"
        ).fetchone()[0]
        if invalid_device_times:
            raise ValueError("910B AscendTask contains negative duration")
        recovered.executescript(
            """
            CREATE TABLE CANDIDATE AS
            SELECT b.globalTaskId, a.start_time, a.duration, b.streamId, b.taskId,
                   b.name, b.blockDim, b.mixBlockDim, b.taskType, b.opType,
                   CASE
                     WHEN b.taskType='AI_VECTOR_CORE' AND a.device_task_type='AI_CORE' THEN 0
                     WHEN b.taskType=a.device_task_type THEN 0
                     WHEN b.taskType='MIX_AIC' AND a.device_task_type='MIX_AIV' THEN 1
                     ELSE 99
                   END AS priority
            FROM device_raw.AscendTask a
            JOIN BINDING b ON b.connectionId=a.connection_id
                          AND b.streamId=a.stream_id AND b.taskId=a.task_id
            WHERE a.start_time>=0;
            CREATE INDEX candidate_global_task ON CANDIDATE(globalTaskId, priority);
            """
        )
        missing_or_ambiguous = recovered.execute(
            "SELECT COUNT(*) FROM BINDING b WHERE "
            "(SELECT COUNT(*) FROM CANDIDATE c WHERE c.globalTaskId=b.globalTaskId "
            "AND c.priority=(SELECT MIN(c2.priority) FROM CANDIDATE c2 "
            "WHERE c2.globalTaskId=b.globalTaskId) AND c.priority<99) != 1"
        ).fetchone()[0]
        if missing_or_ambiguous:
            raise ValueError(
                "910B intermediate task sources do not form a complete 1:1 join"
            )
        recovered.executescript(
            """
            CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY, value TEXT NOT NULL UNIQUE);
            INSERT INTO STRING_IDS(value)
            SELECT value FROM (
                SELECT name AS value FROM BINDING
                UNION SELECT taskType AS value FROM BINDING
                UNION SELECT opType AS value FROM BINDING
            ) ORDER BY value;
            CREATE TABLE TASK(
                startNs INTEGER NOT NULL, endNs INTEGER NOT NULL,
                deviceId INTEGER NOT NULL, globalTaskId INTEGER NOT NULL UNIQUE,
                streamId INTEGER NOT NULL, taskId INTEGER NOT NULL
            );
            CREATE TABLE COMPUTE_TASK_INFO(
                name INTEGER NOT NULL, globalTaskId INTEGER PRIMARY KEY,
                blockDim INTEGER NOT NULL, mixBlockDim INTEGER,
                opType INTEGER NOT NULL, taskType INTEGER NOT NULL
            );
            """
        )
        recovered.execute(
            "INSERT INTO TASK SELECT c.start_time+?, c.start_time+c.duration+?, ?, "
            "c.globalTaskId, c.streamId, c.taskId FROM CANDIDATE c WHERE c.priority<99 "
            "AND c.priority=(SELECT MIN(c2.priority) FROM CANDIDATE c2 "
            "WHERE c2.globalTaskId=c.globalTaskId)",
            (clock_offset_ns, clock_offset_ns, profile.device_id),
        )
        recovered.execute(
            "INSERT INTO COMPUTE_TASK_INFO SELECT sn.id, b.globalTaskId, b.blockDim, "
            "b.mixBlockDim, so.id, st.id FROM BINDING b "
            "JOIN STRING_IDS sn ON sn.value=b.name "
            "JOIN STRING_IDS so ON so.value=b.opType "
            "JOIN STRING_IDS st ON st.value=b.taskType"
        )
        task_count = recovered.execute("SELECT COUNT(*) FROM TASK").fetchone()[0]
        compute_count = recovered.execute(
            "SELECT COUNT(*) FROM COMPUTE_TASK_INFO"
        ).fetchone()[0]
        if task_count != expected_count or compute_count != expected_count:
            raise ValueError("910B recovery row-count validation failed")
        recovered.executescript(
            "DROP TABLE CANDIDATE; DROP TABLE BINDING; DETACH DATABASE device_raw;"
        )
        recovered.commit()
    finally:
        recovered.close()
        device.close()
        runtime.close()
        ge.close()


@contextmanager
def _prepared_ascend_profile(profile: _AscendRankProfile):
    if profile.recovery_paths is None:
        yield profile
        return
    with tempfile.TemporaryDirectory(prefix="midlayer-ascend-recovery-") as directory:
        database = Path(directory) / "recovered.db"
        _rebuild_ascend_task_database(profile, database)
        yield replace(profile, database=database)


def _normalize_profiler(
    case_dir: Path,
    hardware_id: str,
    raw_scheduler: Sequence[Mapping[str, object]],
    scheduler: Sequence[Mapping[str, object]],
    *,
    expected_ranks: tuple[int, ...] | None = None,
    operator_records: Sequence[Mapping[str, object]] | None = None,
) -> tuple[
    tuple[Mapping[str, object], ...],
    AvailabilityPayload,
    AvailabilityPayload,
]:
    form = _raw_capture_form(
        hardware_id, f"unsupported profiler hardware: {hardware_id}")
    if form.profiler == "nsys":
        sqlite_path = case_dir / "nsys_profile.sqlite"
        if not sqlite_path.is_file():
            raise FileNotFoundError(f"missing whitelisted A800 nsys SQLite: {sqlite_path}")
        version = _nsys_version(sqlite_path)
        launches, timing = _normalize_a800_profiler(sqlite_path, scheduler)
        sources = (
            _raw_source(case_dir, sqlite_path, "device-profiler", "nsys", version),
        )
    elif form.profiler == "nsys_ranked":
        sqlite_path = case_dir / "nsys_profile.sqlite"
        if not sqlite_path.is_file():
            raise FileNotFoundError(f"missing H200 nsys SQLite: {sqlite_path}")
        if expected_ranks is None:
            raise ValueError("H200 profiler normalization requires expected ranks")
        version = _nsys_version(sqlite_path)
        launches, timing = _normalize_h200_profiler(
            sqlite_path, raw_scheduler, scheduler, expected_ranks)
        sources = (
            _raw_source(case_dir, sqlite_path, "device-profiler", "nsys", version),
        )
    elif form.profiler == "mctracer":
        source_specs = _c500_profiler_source_specs(case_dir)
        if source_specs[0][1] == "device-profiler":
            trace_path = source_specs[0][0]
            sync_path = source_specs[1][0]
            launches, timing = _normalize_c500_profiler(
                trace_path,
                sync_path,
                scheduler,
            )
        elif source_specs[0][1] == "device-launch-projection":
            if expected_ranks is None:
                raise ValueError("C500 projection normalization requires expected ranks")
            if operator_records is None:
                raise ValueError("C500 projection normalization requires operator records")
            projection_path = source_specs[0][0]
            sync_paths = tuple(
                path
                for path, archive_id, _tool, _version in source_specs
                if archive_id.startswith("profiler-step-boundaries-rank")
            )
            launches, timing = _normalize_c500_projection(
                case_dir,
                projection_path,
                sync_paths,
                raw_scheduler,
                scheduler,
                expected_ranks,
                operator_records,
            )
        else:
            raise AssertionError("unknown C500 profiler source kind")
        sources = tuple(
            _raw_source(case_dir, path, archive_id, tool, tool_version)
            for path, archive_id, tool, tool_version in source_specs
        )
    elif form.profiler == "msprof":
        if expected_ranks is None:
            raise ValueError("910B profiler normalization requires expected ranks")
        if operator_records is None:
            raise ValueError("910B profiler normalization requires operator records")
        profiles = _ascend_rank_profiles(case_dir, expected_ranks)
        with ExitStack() as stack:
            prepared_profiles = tuple(
                stack.enter_context(_prepared_ascend_profile(profile))
                for profile in profiles
            )
            for profile in prepared_profiles:
                _validate_ascend_task_geometry(profile.database)
            launches, timing = _normalize_ascend_profiler(
                prepared_profiles, raw_scheduler, scheduler, operator_records
            )
        sources = tuple(
            source
            for profile in profiles
            for path, archive_id, tool, tool_version in _ascend_profile_source_specs(
                profile
            )
            for source in (
                _raw_source(case_dir, path, archive_id, tool, tool_version),
            )
        )
    else:
        raise AssertionError(f"unsupported profiler hardware: {hardware_id}")
    return sources, launches, timing


def _normalize_a800_profiler(
    sqlite_path: Path,
    scheduler: Sequence[Mapping[str, object]],
) -> tuple[AvailabilityPayload, AvailabilityPayload]:
    work_steps = _work_step_ids(scheduler)
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as connection:
        devices = _a800_rank_devices(connection)
        if not devices:
            return _unavailable_profiler("A800 Nsys contains no device kernel rows")
        phase_by_step = {
            record["step_id"]: _scheduler_step_phase(record)
            for record in scheduler
        }
        all_launch_records = []
        all_timing_records = []
        covered = set()
        for rank, device_id in enumerate(devices):
            first_model = connection.execute(
                "SELECT MIN(k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.demangledName=s.id WHERE k.deviceId=? AND "
                "(s.value LIKE '%cross_device_reduce%' OR "
                "s.value LIKE '%rms_norm%' OR "
                "s.value LIKE '%fused_add_rms_norm%')",
                (device_id,),
            ).fetchone()
            if first_model is None or first_model[0] is None:
                return _unavailable_profiler("A800 Nsys rank has no collective or norm model-entry kernel")
            anchor_ends = _a800_sampler_tail_anchor_ends(
                connection, device_id, len(work_steps)
            )
            windows = []
            lower = _integer(first_model[0], "A800 first model timestamp", minimum=0)
            for step_id, upper in zip(work_steps, anchor_ends, strict=True):
                if upper <= lower:
                    raise ValueError("A800 sampler-tail boundaries are not increasing")
                windows.append((lower, upper, step_id, True, phase_by_step[step_id]))
                lower = upper
            kernel_rows = connection.execute(
                "SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.staticSharedMemory+k.dynamicSharedMemory, k.registersPerThread, k.streamId, k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.deviceId=? AND k.start>=? AND k.start<? ORDER BY k.start",
                (device_id, windows[0][0], windows[-1][1]),
            ).fetchall()
            launch_records, timing_records, rank_covered = _cuda_profiler_records(
                kernel_rows, (), windows, kernel_family="cuda_kernel",
                clock_domain="nsys_device_timestamp", time_origin="capture_start", rank=rank,
            )
            all_launch_records.extend(launch_records)
            all_timing_records.extend(timing_records)
            covered.update(rank_covered)
    return _profiler_availability(
        tuple(all_launch_records),
        tuple(all_timing_records),
        work_steps,
        covered,
        launch_reason=None,
        timing_reason=(
            "device intervals use capture-relative nsys time; raw has no measured "
            "cross-clock error for Scheduler-to-forward or Host interval attribution"
        ),
    )


def _normalize_h200_profiler(
    sqlite_path: Path,
    raw_scheduler: Sequence[Mapping[str, object]],
    scheduler: Sequence[Mapping[str, object]],
    expected_ranks: tuple[int, ...],
) -> tuple[AvailabilityPayload, AvailabilityPayload]:
    """Attribute H200 kernels to Scheduler steps through NVTX launch ranges.

    The H200 capture pushes ``midlayer-op-step:<iter>`` around every formal
    model-runner forward and ``midlayer-op-sample:<iter>`` around the Model
    Runner V2 sampler.  A kernel belongs to the step whose range contains the
    host runtime call that launched it (same thread, correlation ID within the
    same process), so no sampler-kernel heuristic is needed.  Device windows
    are then the contiguous spans from each work step's first kernel start.
    """

    if len(raw_scheduler) != len(scheduler):
        raise ValueError("raw and normalized Scheduler record counts disagree")
    step_by_iter = {}
    for raw, normalized in zip(raw_scheduler, scheduler, strict=True):
        step_by_iter[_integer(raw["iter_id"], "Scheduler iter_id", minimum=0)] = (
            normalized["step_id"])
    work_steps = _work_step_ids(scheduler)
    phase_by_step = {
        record["step_id"]: _scheduler_step_phase(record) for record in scheduler
    }
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as connection:
        ranges = connection.execute(
            "SELECT e.start, e.end, e.globalTid, COALESCE(e.text, s.value) "
            "FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id "
            "WHERE e.end IS NOT NULL AND COALESCE(e.text, s.value) LIKE 'midlayer-op-%'"
        ).fetchall()
        by_thread: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for start_ns, end_ns, global_tid, text in ranges:
            match = _H200_NVTX_STEP.fullmatch(_nonempty_string(text, "H200 NVTX text"))
            if match is None:
                raise ValueError(f"unexpected H200 NVTX range: {text!r}")
            by_thread[_integer(global_tid, "H200 NVTX globalTid", minimum=0)].append((
                _integer(start_ns, "H200 NVTX start", minimum=0),
                _integer(end_ns, "H200 NVTX end", minimum=0),
                int(match.group("iter")),
            ))
        if not by_thread:
            return _unavailable_profiler("H200 Nsys contains no midlayer step ranges")
        for values in by_thread.values():
            values.sort()
            if any(b[0] < a[1] for a, b in zip(values, values[1:])):
                raise ValueError("H200 NVTX step ranges overlap on one thread")
        launch_rows = connection.execute(
            "SELECT r.start, r.globalTid, k.start, k.end, s.value, k.gridX, k.gridY, "
            "k.gridZ, k.blockX, k.blockY, k.blockZ, "
            "k.staticSharedMemory+k.dynamicSharedMemory, k.registersPerThread, "
            "k.streamId, k.deviceId "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId "
            "AND k.globalPid = (r.globalTid & -16777216) "
            "JOIN StringIds s ON k.demangledName = s.id "
            "WHERE r.globalTid IN (%s) ORDER BY k.start"
            % ",".join(str(value) for value in sorted(by_thread))
        ).fetchall()
    starts_by_thread = {
        thread: [value[0] for value in values] for thread, values in by_thread.items()
    }
    kernels_by_device: defaultdict[int, list[tuple[str, Sequence[object]]]] = (
        defaultdict(list))
    for row in launch_rows:
        call_ns = _integer(row[0], "H200 runtime call start", minimum=0)
        thread = _integer(row[1], "H200 runtime globalTid", minimum=0)
        values = by_thread[thread]
        position = bisect_right(starts_by_thread[thread], call_ns) - 1
        if position < 0 or call_ns >= values[position][1]:
            continue
        iter_id = values[position][2]
        if iter_id not in step_by_iter:
            raise ValueError("H200 NVTX range names an unknown Scheduler iteration")
        device = _integer(row[14], "H200 kernel deviceId", minimum=0)
        kernels_by_device[device].append((step_by_iter[iter_id], row[2:]))
    devices = tuple(sorted(kernels_by_device))
    if devices != expected_ranks:
        raise ValueError(
            f"H200 attributed kernel devices {devices} disagree with ranks {expected_ranks}")
    all_launch_records = []
    all_timing_records = []
    covered: set[str] = set()
    work_step_set = set(work_steps)
    for rank in expected_ranks:
        by_step: defaultdict[str, list[Sequence[object]]] = defaultdict(list)
        for step_id, kernel in kernels_by_device[rank]:
            if step_id in work_step_set:
                by_step[step_id].append(kernel)
        ordered_steps = [step for step in work_steps if step in by_step]
        windows = []
        for index, step_id in enumerate(ordered_steps):
            lower = min(int(kernel[0]) for kernel in by_step[step_id])
            last_start = max(int(kernel[0]) for kernel in by_step[step_id])
            if index + 1 < len(ordered_steps):
                upper = min(int(kernel[0]) for kernel in by_step[ordered_steps[index + 1]])
            else:
                upper = max(int(kernel[1]) for kernel in by_step[step_id]) + 1
            if last_start >= upper:
                raise ValueError("H200 device kernels of consecutive steps overlap")
            windows.append((lower, upper, step_id, True, phase_by_step[step_id]))
        kernel_rows = sorted(
            (kernel for step_id in ordered_steps for kernel in by_step[step_id]),
            key=lambda kernel: int(kernel[0]),
        )
        launch_records, timing_records, rank_covered = _cuda_profiler_records(
            kernel_rows, (), windows, kernel_family="cuda_kernel",
            clock_domain="nsys_device_timestamp", time_origin="capture_start",
            rank=rank,
        )
        all_launch_records.extend(launch_records)
        all_timing_records.extend(timing_records)
        covered.update(rank_covered)
    return _profiler_availability(
        tuple(all_launch_records),
        tuple(all_timing_records),
        work_steps,
        covered,
        launch_reason=None,
        timing_reason=(
            "device intervals use capture-relative nsys time; kernels are "
            "attributed to Scheduler steps through NVTX launch ranges"
        ),
    )


def _a800_sampler_tail_anchor_ends(
    connection: sqlite3.Connection,
    device_id: int,
    work_step_count: int,
) -> tuple[int, ...]:
    """Return the unique direct-copy sampler tail ending every work step."""

    rows = connection.execute(
        "SELECT s.value, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName=s.id "
        "WHERE k.deviceId=? AND s.value LIKE '%direct_copy_kernel_cuda%' "
        "ORDER BY k.start",
        (device_id,),
    ).fetchall()
    by_name: defaultdict[str, list[int]] = defaultdict(list)
    for name, end_ns in rows:
        by_name[_nonempty_string(name, "A800 direct-copy kernel name")].append(
            _integer(end_ns, "A800 direct-copy end", minimum=1)
        )
    candidates = {
        name: tuple(ends)
        for name, ends in by_name.items()
        if len(ends) == work_step_count
    }
    if not candidates:
        raise ValueError(
            "A800 capture has no direct-copy signature occurring exactly once per work step"
        )
    latest = []
    for name, ends in candidates.items():
        if all(
            all(left >= right for left, right in zip(ends, other, strict=True))
            for other_name, other in candidates.items()
            if other_name != name
        ):
            latest.append((name, ends))
    if len(latest) != 1:
        raise ValueError(
            "A800 direct-copy evidence does not identify one unique sampler-tail signature"
        )
    ends = latest[0][1]
    if any(later <= earlier for earlier, later in zip(ends, ends[1:])):
        raise ValueError("A800 sampler-tail direct-copy anchors must strictly increase")
    return ends


def _a800_rank_devices(connection: sqlite3.Connection) -> tuple[int, ...]:
    """Map worker-rank order to CUDA devices through launch correlation ownership."""

    ownership: dict[int, tuple[int, int]] = {}
    rows = connection.execute(
        "SELECT r.globalTid, k.deviceId, COUNT(*) "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
        "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId "
        "WHERE s.value LIKE 'cudaLaunchKernel%' GROUP BY r.globalTid, k.deviceId"
    ).fetchall()
    for global_tid, device_id, count in rows:
        thread = _integer(global_tid, "A800 launch globalTid", minimum=0)
        device = _integer(device_id, "A800 launch deviceId", minimum=0)
        launches = _integer(count, "A800 correlated launch count", minimum=1)
        if thread not in ownership or launches > ownership[thread][1]:
            ownership[thread] = (device, launches)
    if not ownership:
        raise ValueError("A800 Nsys has no correlated cudaLaunchKernel ownership")
    devices = tuple(ownership[thread][0] for thread in sorted(ownership))
    if len(devices) != len(set(devices)):
        raise ValueError("A800 worker threads do not map one-to-one onto devices")
    captured_devices = tuple(
        row[0] for row in connection.execute(
            "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId"
        )
    )
    if set(devices) != set(captured_devices):
        raise ValueError("A800 correlated worker ownership does not cover every device")
    return devices


def _normalize_c500_profiler(
    trace_path: Path,
    sync_path: Path,
    scheduler: Sequence[Mapping[str, object]],
) -> tuple[AvailabilityPayload, AvailabilityPayload]:
    windows = _c500_sync_windows(sync_path, scheduler)
    sync_ends = [window[1] for window in windows]
    signatures: dict[str, Mapping[str, object]] = {}
    launches = []
    timing = []
    covered = set()
    launch_sequence: defaultdict[tuple[int, str], int] = defaultdict(int)
    interval_sequence: defaultdict[tuple[int, str], int] = defaultdict(int)
    ordered_events = sorted(
        enumerate(iter_mctracer_device_events(trace_path)),
        key=lambda indexed_event: (indexed_event[1].ts_ns, indexed_event[0]),
    )
    for raw_index, event in ordered_events:
        window_index = bisect_right(sync_ends, event.ts_ns)
        if window_index >= len(windows):
            continue
        _, upper, step_id, has_work, phase = windows[window_index]
        if event.ts_ns + event.dur_ns > upper:
            raise ValueError("C500 mcTracer device event crosses a sync boundary")
        if not has_work:
            continue
        rank = event.device_id
        interval_index = interval_sequence[(rank, step_id)]
        interval_sequence[(rank, step_id)] += 1
        launch_id = None
        if isinstance(event, McTracerKernelEvent):
            signature_payload = {
                "kernel_name": event.name,
                "kernel_family": "metax_kernel",
                "geometry_kind": "cuda_dim3",
                "grid": list(event.grid),
                "block": list(event.block),
                "ascend_block_dim": None,
                "ascend_mix_block_dim": None,
                "shared_memory_bytes": (
                    event.static_shared_memory_bytes
                    + event.dynamic_shared_memory_bytes
                ),
                "registers_per_thread": event.registers_per_thread,
                "tile": None,
                "split": None,
                "stream": None,
                "device": event.device_id,
            }
            launch_index = launch_sequence[(rank, step_id)]
            launch_sequence[(rank, step_id)] += 1
            launch_id = _append_launch(
                signatures,
                launches,
                signature_payload,
                step_id,
                launch_index,
                rank=rank,
            )
            interval_kind = "device_kernel"
        elif isinstance(event, (McTracerMemcpyEvent, McTracerMemsetEvent)):
            interval_kind = "device_copy"
        else:
            raise AssertionError(f"unknown typed mcTracer event: {type(event)!r}")
        timing.append(
            _timing_record(
                step_id,
                phase,
                interval_index,
                interval_kind,
                event.ts_ns,
                event.ts_ns + event.dur_ns,
                launch_id,
                "mctracer_device_timestamp",
                "capture_start",
                f"mctracer-{event.event_type}-{raw_index}",
                rank=rank,
            )
        )
        covered.add(step_id)
    launch_records = tuple(signatures[key] for key in sorted(signatures)) + tuple(
        launches
    )
    prefix_reason = (
        "retained C500 mcTracer synchronization boundaries prove only a Scheduler "
        "prefix; warmup and idle buckets are excluded"
    )
    return _profiler_availability(
        launch_records,
        tuple(timing),
        _work_step_ids(scheduler),
        covered,
        launch_reason=prefix_reason,
        timing_reason=prefix_reason,
        source_archive_ids=("device-profiler", "profiler-step-boundaries"),
    )


def _normalize_c500_projection(
    case_dir: Path,
    projection_path: Path,
    sync_paths: tuple[Path, ...],
    raw_scheduler: Sequence[Mapping[str, object]],
    scheduler: Sequence[Mapping[str, object]],
    expected_ranks: tuple[int, ...],
    operator_records: Sequence[Mapping[str, object]],
) -> tuple[AvailabilityPayload, AvailabilityPayload]:
    """Recover launch facts from an explicitly identified mcTracer projection."""

    if expected_ranks != tuple(range(len(expected_ranks))) or not expected_ranks:
        raise ValueError("C500 expected ranks must be non-empty and contiguous")
    _validate_c500_projection_rank_provenance(
        case_dir, expected_ranks, sync_paths
    )
    work_steps = _work_step_ids(scheduler)
    terminal_keys_by_rank: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    sampler_evidence_by_rank: defaultdict[
        int, list[tuple[int, int, str]]
    ] = defaultdict(list)

    def collect_terminal(
        rank: int, source_index: int, event: _C500ProjectionEvent
    ) -> None:
        if (
            _c500_projection_event_kind(event) == "device_kernel"
            and _is_c500_sampler_argmax(event.name)
        ):
            terminal_keys_by_rank[rank].append((event.timestamp_ns, source_index))
        if (
            _is_c500_sampler_argmax(event.name)
            or _is_c500_sampler_copy_int(event.name)
            or _is_c500_sampler_copy_long(event.name)
            or event.name == _C500_SAMPLER_COPY_DTOH
        ):
            sampler_evidence_by_rank[rank].append(
                (event.timestamp_ns, source_index, event.name)
            )

    initial_scan = _scan_c500_projection_events(
        projection_path, expected_ranks, collect_terminal
    )
    for rank in expected_ranks:
        terminal_keys_by_rank[rank].sort()
        timestamps = [key[0] for key in terminal_keys_by_rank[rank]]
        if any(
            later <= earlier
            for earlier, later in zip(timestamps, timestamps[1:])
        ):
            raise ValueError(
                f"C500 rank {rank} sampler terminals must strictly increase"
            )
    if sync_paths:
        _validate_c500_projection_layout(initial_scan, expects_rebuild=True)
        windows_by_rank = _c500_projection_sync_windows(
            sync_paths,
            raw_scheduler,
            scheduler,
            expected_ranks,
            terminal_keys_by_rank,
        )
        source_archive_ids = (
            "device-launch-projection",
            "mctracer-capture-log",
            *(f"profiler-step-boundaries-rank{rank}" for rank in expected_ranks),
        )
        window_reason = (
            "each rank independently groups its synchronization intervals until "
            "one semantic sampler ArgMax terminal closes the next work step"
        )
        starts_by_rank = {
            rank: tuple(int(window[0]) for window in windows_by_rank[rank])
            for rank in expected_ranks
        }

        def event_step(
            rank: int, source_index: int, event: _C500ProjectionEvent
        ) -> str | None:
            del source_index
            window = _c500_projection_event_window(
                event, starts_by_rank[rank], windows_by_rank[rank]
            )
            if window is None or not window[3]:
                return None
            return str(window[2])
    else:
        _validate_c500_projection_layout(initial_scan, expects_rebuild=False)
        source_archive_ids = (
            "device-launch-projection",
            "mctracer-capture-log",
        )
        window_reason = (
            "each rank is independently segmented by the semantic sampler "
            "ArgMax terminal; Scheduler timestamps are not used"
        )

        for rank in expected_ranks:
            terminals = _c500_legacy_sampler_terminal_keys(
                sampler_evidence_by_rank[rank], rank
            )
            terminal_keys_by_rank[rank] = list(terminals)
            if len(terminals) != len(work_steps):
                raise ValueError(
                    f"C500 rank {rank} sampler terminal count {len(terminals)} "
                    f"does not match {len(work_steps)} work-bearing Scheduler steps"
                )
            timestamps = [key[0] for key in terminals]
            if any(
                later <= earlier
                for earlier, later in zip(timestamps, timestamps[1:])
            ):
                raise ValueError(
                    f"C500 rank {rank} sampler terminals must strictly increase"
                )
        def event_step(
            rank: int, source_index: int, event: _C500ProjectionEvent
        ) -> str | None:
            ordinal = bisect_left(
                terminal_keys_by_rank[rank],
                (event.timestamp_ns, source_index),
            )
            if ordinal >= len(work_steps):
                return None
            return work_steps[ordinal]
    presence = {
        (step_id, rank)
        for step_id in work_steps
        for rank in expected_ranks
    }
    operator_step_ranks = {
        (record["step_id"], record["rank"])
        for record in operator_records
        if record["record_type"] == "event"
    }
    selected_by_phase = _middle_complete_step_by_phase(
        scheduler,
        operator_step_ranks,
        presence,
        expected_ranks,
    )
    selected_step_ids = {
        value["scheduler"]["step_id"]
        for value in selected_by_phase.values()
        if value is not None
    }
    signatures: dict[str, Mapping[str, object]] = {}
    launches: list[Mapping[str, object]] = []
    launch_sequence: defaultdict[tuple[int, str], int] = defaultdict(int)
    retained_events: defaultdict[
        tuple[int, str], list[tuple[int, int, _C500ProjectionEvent, str]]
    ] = defaultdict(list)
    real_launch_count: defaultdict[tuple[int, str], int] = defaultdict(int)

    def collect_event(
        rank: int, source_index: int, event: _C500ProjectionEvent
    ) -> None:
        step_id = event_step(rank, source_index, event)
        if step_id is None:
            return
        event_kind = _c500_projection_event_kind(event)
        if event_kind == "device_kernel":
            real_launch_count[(rank, step_id)] += 1
        if step_id not in selected_step_ids:
            return
        retained_events[(rank, step_id)].append(
            (event.timestamp_ns, source_index, event, event_kind)
        )

    final_scan = _scan_c500_projection_events(
        projection_path, expected_ranks, collect_event
    )
    if final_scan != initial_scan:
        raise ValueError("C500 projection changed between validation passes")
    if any(
        real_launch_count[(rank, step_id)] <= 0
        for rank in expected_ranks
        for step_id in work_steps
    ):
        raise ValueError(
            "C500 structural step segmentation produced a work step without a real kernel launch"
        )
    for rank in expected_ranks:
        for step_id in work_steps:
            if step_id not in selected_step_ids:
                continue
            for _timestamp, source_index, event, event_kind in sorted(
                retained_events[(rank, step_id)], key=lambda item: (item[0], item[1])
            ):
                if event_kind == "device_kernel":
                    signature_payload = {
                        "kernel_name": event.name,
                        "kernel_family": "metax_legacy_launch_projection",
                        "geometry_kind": "unavailable",
                        "grid": None,
                        "block": None,
                        "ascend_block_dim": None,
                        "ascend_mix_block_dim": None,
                        "shared_memory_bytes": None,
                        "registers_per_thread": None,
                        "tile": None,
                        "split": None,
                        "stream": None,
                        "device": None,
                    }
                    launch_index = launch_sequence[(rank, step_id)]
                    launch_sequence[(rank, step_id)] += 1
                    _append_launch(
                        signatures,
                        launches,
                        signature_payload,
                        step_id,
                        launch_index,
                        rank=rank,
                    )
    selected_presence = {
        (record["step_id"], record["rank"])
        for record in launches
        if record["record_type"] == "event"
    }
    expected_presence = {
        (step_id, rank)
        for step_id in selected_step_ids
        for rank in expected_ranks
    }
    if selected_presence != expected_presence:
        raise ValueError(
            "C500 legacy projection does not retain every selected step on every rank"
        )
    launch_records = tuple(signatures[key] for key in sorted(signatures)) + tuple(
        launches
    )
    launch_reason = (
        "kernel names and event order come from the retained mcTracer-derived "
        "legacy projection; historical converters could synthesize missing grid, "
        "block, shared-memory, device and duration fields, so those fields are not "
        "published; "
        f"{window_reason}; only the canonical representative steps are retained; "
        "rank association follows the version-pinned converter grouping and is "
        "recorded as inferred rather than native observation"
    )
    launch_availability = AvailabilityPayload(
        "available",
        "partial",
        launch_reason,
        launch_records,
        source_archive_ids,
    )
    timing_availability = AvailabilityPayload(
        "unavailable",
        None,
        (
            "historical projection converters could synthesize missing durations; "
            "native mcTracer device events are unavailable"
        ),
        None,
        source_archive_ids,
    )
    return launch_availability, timing_availability


def _validate_c500_projection_layout(
    scan: _C500ProjectionScan, *, expects_rebuild: bool
) -> None:
    if expects_rebuild:
        if scan.top_level_count != 0 or scan.bottleneck_rank is not None:
            raise ValueError(
                "C500 rebuilt projection must have an empty redundant kernel array and no bottleneck rank"
            )
    elif scan.top_level_count <= 0 or scan.bottleneck_rank is None:
        raise ValueError(
            "C500 legacy projection must retain the bottleneck-rank redundant kernel sequence"
        )


def _c500_projection_event_kind(event: _C500ProjectionEvent) -> str:
    if event.name in _C500_LEGACY_MEMORY_EVENT_NAMES:
        return "device_copy"
    lowered = event.name.strip().lower()
    if lowered.startswith("memcpy") or lowered.startswith("memset"):
        raise ValueError(
            f"unsupported C500 legacy memory event name: {event.name!r}"
        )
    return "device_kernel"


def _is_c500_sampler_argmax(name: str) -> bool:
    return (
        name.startswith("std::enable_if<")
        and _C500_SAMPLER_TERMINAL_FUNCTION in name
        and _C500_SAMPLER_TERMINAL_OPERATION in name
    )


def _is_c500_sampler_copy_int(name: str) -> bool:
    return (
        name.startswith(_C500_SAMPLER_COPY_FUNCTION)
        and _C500_SAMPLER_COPY_INT in name
    )


def _is_c500_sampler_copy_long(name: str) -> bool:
    return (
        name.startswith(_C500_SAMPLER_COPY_FUNCTION)
        and _C500_SAMPLER_COPY_LONG in name
    )


def _c500_legacy_sampler_terminal_keys(
    evidence: Sequence[tuple[int, int, str]], rank: int
) -> tuple[tuple[int, int], ...]:
    ordered = sorted(evidence, key=lambda item: (item[0], item[1]))
    terminals = []
    for index, (_timestamp, _source_index, name) in enumerate(ordered):
        if not _is_c500_sampler_argmax(name):
            continue
        if index + 3 >= len(ordered):
            raise ValueError(
                f"C500 rank {rank} sampler ArgMax has no complete launch/copy trailer"
            )
        copy_int = ordered[index + 1][2]
        copy_long = ordered[index + 2][2]
        dtoh = ordered[index + 3]
        source_indices = tuple(item[1] for item in ordered[index:index + 4])
        if source_indices != tuple(range(source_indices[0], source_indices[0] + 4)):
            raise ValueError(
                f"C500 rank {rank} sampler trailer is not consecutive in the raw event sequence"
            )
        if (
            not _is_c500_sampler_copy_int(copy_int)
            or not _is_c500_sampler_copy_long(copy_long)
            or dtoh[2] != _C500_SAMPLER_COPY_DTOH
        ):
            raise ValueError(
                f"C500 rank {rank} sampler ArgMax does not have the proven fixed trailer"
            )
        terminals.append((dtoh[0], dtoh[1]))
    return tuple(terminals)


def _c500_projection_event_window(
    event: _C500ProjectionEvent,
    starts: Sequence[int],
    windows: Sequence[Sequence[object]],
) -> Sequence[object] | None:
    window = _event_window(event.timestamp_ns, starts, windows)
    if window is None:
        return None
    upper = window[1]
    if (
        event.duration_ns is not None
        and upper is not None
        and event.timestamp_ns + event.duration_ns > upper
    ):
        raise ValueError(
            "C500 projected device event crosses its structural step boundary"
        )
    return window


def _c500_projection_sync_windows(
    sync_paths: tuple[Path, ...],
    raw_scheduler: Sequence[Mapping[str, object]],
    scheduler: Sequence[Mapping[str, object]],
    expected_ranks: tuple[int, ...],
    terminal_keys_by_rank: Mapping[int, Sequence[tuple[int, int]]],
) -> Mapping[int, tuple[tuple[int, int, str, bool, str], ...]]:
    if len(raw_scheduler) != len(scheduler):
        raise ValueError("raw and normalized Scheduler record counts disagree")
    by_rank = {_c500_sync_rank(path): path for path in sync_paths}
    if set(by_rank) != set(expected_ranks) or len(by_rank) != len(sync_paths):
        raise ValueError(
            "C500 projection sync sidecars must cover every expected rank exactly once"
        )
    intervals_by_rank = {
        rank: _c500_sync_intervals(by_rank[rank]) for rank in expected_ranks
    }
    scheduler_times = tuple(
        _integer(record["ts_us"], "Scheduler ts_us", minimum=0) * 1000
        for record in raw_scheduler
    )
    if any(
        later <= earlier
        for earlier, later in zip(scheduler_times, scheduler_times[1:])
    ):
        raise ValueError("Scheduler timestamps must strictly increase")
    work_records = tuple(
        record for record in scheduler if record["scheduled_requests"]
    )
    terminal_bucket_shapes = {}
    result = {}
    for rank in expected_ranks:
        intervals = intervals_by_rank[rank]
        rank_ends = [interval[1] for interval in intervals]
        warmup_count = bisect_left(rank_ends, scheduler_times[0])
        if warmup_count != 4:
            raise ValueError(
                f"C500 rank {rank} does not retain exactly four pre-Scheduler warmups"
            )
        terminals = terminal_keys_by_rank[rank]
        terminal_timestamps = [key[0] for key in terminals]
        bucket_counts = []
        previous_end_ns = -1
        for _start_ns, end_ns in intervals:
            lower = bisect_right(terminal_timestamps, previous_end_ns)
            upper = bisect_right(terminal_timestamps, end_ns)
            bucket_counts.append(upper - lower)
            previous_end_ns = end_ns
        if sum(bucket_counts[:warmup_count]) <= 0:
            raise ValueError(
                f"C500 rank {rank} warmup sync prefix contains no sampler terminals"
            )
        windows = []
        group_start_ns = intervals[warmup_count - 1][1]
        terminal_count = 0
        for bucket_index in range(warmup_count, len(intervals)):
            terminal_count += bucket_counts[bucket_index]
            if terminal_count > 1:
                raise ValueError(
                    f"C500 rank {rank} sync group contains multiple sampler terminals"
                )
            if terminal_count == 0:
                continue
            work_ordinal = len(windows)
            if work_ordinal >= len(work_records):
                raise ValueError(
                    f"C500 rank {rank} retains sampler terminals after all Scheduler work steps"
                )
            normalized = work_records[work_ordinal]
            end_ns = intervals[bucket_index][1]
            windows.append(
                (
                    group_start_ns,
                    end_ns,
                    _nonempty_string(normalized["step_id"], "Scheduler step_id"),
                    True,
                    _scheduler_step_phase(normalized),
                )
            )
            group_start_ns = end_ns
            terminal_count = 0
        if terminal_count != 0 or len(windows) != len(work_records):
            raise ValueError(
                f"C500 rank {rank} sync/terminal groups do not close every work-bearing Scheduler step"
            )
        result[rank] = tuple(windows)
        terminal_bucket_shapes[rank] = tuple(bucket_counts)
    reference_shape = terminal_bucket_shapes[expected_ranks[0]]
    for rank in expected_ranks[1:]:
        if terminal_bucket_shapes[rank] != reference_shape:
            raise ValueError(
                "C500 rank-local sync/terminal structure disagrees across TP ranks"
            )
    return result


def _c500_sync_intervals(path: Path) -> tuple[tuple[int, int], ...]:
    raw_intervals = _sequence(
        _json_loads(path.read_text(encoding="utf-8"), "C500 sync boundaries"),
        "C500 sync boundaries",
    )
    if not raw_intervals:
        raise ValueError("C500 structural sync sidecar is empty")
    intervals = []
    previous_end = -1
    for index, value in enumerate(raw_intervals):
        pair = _sequence(value, f"C500 sync boundary {index}")
        if len(pair) != 2:
            raise ValueError("C500 sync boundary must contain start and end")
        start = _integer(pair[0], "C500 sync start", minimum=0)
        end = _integer(pair[1], "C500 sync end", minimum=0)
        if end <= start:
            raise ValueError("C500 sync end must follow sync start")
        if start < previous_end or end <= previous_end:
            raise ValueError("C500 sync boundaries must be strictly ordered")
        intervals.append((start, end))
        previous_end = end
    return tuple(intervals)


def _scan_c500_projection_events(
    path: Path,
    expected_ranks: tuple[int, ...],
    consume_event: Callable[[int, int, _C500ProjectionEvent], None],
    *,
    chunk_bytes: int = 1 << 20,
) -> _C500ProjectionScan:
    """Fully parse and cross-check one retained legacy projection."""

    seen_ranks = set()
    entry_count_by_rank = {}
    digest_by_rank = {}
    rank_duration_presence = set()
    top_duration_presence = set()
    top_level_count = None
    top_level_digest = None
    bottleneck_rank = None
    with _StreamingJsonReader(path, chunk_bytes=chunk_bytes) as reader:
        reader.consume("{")
        seen_top_level = set()
        first_field = True
        while reader.peek() != "}":
            if not first_field:
                reader.consume(",")
            first_field = False
            key = reader.value("C500 projection top-level field")
            if not isinstance(key, str):
                raise ValueError("C500 projection top-level keys must be strings")
            if key in seen_top_level:
                raise ValueError(f"C500 projection contains duplicate top-level key {key!r}")
            seen_top_level.add(key)
            reader.consume(":")
            if key == "kernels":
                top_level_count, top_level_digest = _scan_c500_projection_array(
                    reader,
                    rank=-1,
                    consume_event=lambda _rank, _index, _event: None,
                    duration_presence=top_duration_presence,
                )
            elif key == "per_rank":
                reader.consume("{")
                first_rank = True
                while reader.peek() != "}":
                    if not first_rank:
                        reader.consume(",")
                    first_rank = False
                    rank_text = reader.value("C500 projection rank key")
                    if not isinstance(rank_text, str) or not rank_text.isdigit():
                        raise ValueError("C500 projection rank keys must be decimal strings")
                    rank = int(rank_text)
                    if rank in seen_ranks:
                        raise ValueError("C500 projection contains a duplicate rank")
                    seen_ranks.add(rank)
                    reader.consume(":")
                    count, digest = _scan_c500_projection_rank(
                        reader,
                        rank,
                        consume_event,
                        rank_duration_presence,
                    )
                    entry_count_by_rank[rank] = count
                    digest_by_rank[rank] = digest
                reader.consume("}")
            elif key == "bottleneck_rank":
                raw_bottleneck = reader.value("C500 projection bottleneck rank")
                if not isinstance(raw_bottleneck, str) or not raw_bottleneck.isdigit():
                    raise ValueError("C500 projection bottleneck_rank must be a rank string")
                bottleneck_rank = int(raw_bottleneck)
            else:
                raise ValueError(
                    f"C500 projection has unsupported top-level field {key!r}"
                )
        reader.consume("}")
        reader.require_eof()
    if "kernels" not in seen_top_level or "per_rank" not in seen_top_level:
        raise ValueError("C500 projection requires top-level kernels and per_rank fields")
    if top_level_count is None or top_level_digest is None:
        raise AssertionError("C500 projection top-level kernel scan was not recorded")
    if seen_ranks != set(expected_ranks):
        raise ValueError(
            "C500 projection ranks must exactly match the expected TP ranks"
        )
    if any(entry_count_by_rank[rank] <= 0 for rank in expected_ranks):
        raise ValueError("C500 projection contains an empty rank event array")
    if bottleneck_rank is not None and bottleneck_rank not in seen_ranks:
        raise ValueError("C500 projection bottleneck_rank is not a retained rank")
    if len(rank_duration_presence) != 1:
        raise ValueError(
            "C500 projection must retain duration for either every entry or no entry"
        )
    if top_level_count:
        if bottleneck_rank is None:
            raise ValueError(
                "C500 non-empty redundant kernels require a bottleneck_rank"
            )
        if (
            top_level_count != entry_count_by_rank[bottleneck_rank]
            or top_level_digest != digest_by_rank[bottleneck_rank]
        ):
            raise ValueError(
                "C500 redundant kernels disagree with the declared bottleneck rank"
            )
        if top_duration_presence != rank_duration_presence:
            raise ValueError(
                "C500 redundant kernels disagree with per-rank duration presence"
            )
    elif bottleneck_rank is not None:
        raise ValueError("C500 empty redundant kernels cannot declare a bottleneck_rank")
    return _C500ProjectionScan(
        top_level_count=top_level_count,
        top_level_digest=top_level_digest,
        entry_count_by_rank=dict(entry_count_by_rank),
        digest_by_rank=dict(digest_by_rank),
        bottleneck_rank=bottleneck_rank,
        duration_present=True in rank_duration_presence,
    )


def _scan_c500_projection_rank(
    reader: _StreamingJsonReader,
    rank: int,
    consume_event: Callable[[int, int, _C500ProjectionEvent], None],
    duration_presence: set[bool],
) -> tuple[int, str]:
    nested = reader.peek() == "{"
    if nested:
        reader.consume("{")
        key = reader.value(f"C500 rank {rank} field")
        if key != "kernels":
            raise ValueError("C500 nested rank payload must contain only kernels")
        reader.consume(":")
    count, digest = _scan_c500_projection_array(
        reader,
        rank=rank,
        consume_event=consume_event,
        duration_presence=duration_presence,
    )
    if nested:
        reader.consume("}")
    return count, digest


def _scan_c500_projection_array(
    reader: _StreamingJsonReader,
    *,
    rank: int,
    consume_event: Callable[[int, int, _C500ProjectionEvent], None],
    duration_presence: set[bool],
) -> tuple[int, str]:
    reader.consume("[")
    entry_index = 0
    digest = hashlib.sha256()
    while reader.peek() != "]":
        if entry_index:
            reader.consume(",")
        raw = _mapping(
            reader.value(f"C500 rank {rank} entry {entry_index}"),
            f"C500 rank {rank} entry {entry_index}",
        )
        event = _c500_projection_event(raw, rank, entry_index)
        duration_presence.add(event.duration_ns is not None)
        encoded = json.dumps(
            {
                "name": event.name,
                "timestamp_ns": event.timestamp_ns,
                "duration_ns": event.duration_ns,
                "grid": event.grid,
                "block": event.block,
                "shared_memory_bytes": event.shared_memory_bytes,
                "registers_per_thread": event.registers_per_thread,
                "stream": event.stream,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        consume_event(rank, entry_index, event)
        entry_index += 1
    reader.consume("]")
    return entry_index, digest.hexdigest()


def _c500_projection_event(
    raw: Mapping[str, object], rank: int, entry_index: int
) -> _C500ProjectionEvent:
    base_keys = {"name", "ts_ns", "grid", "block", "smem"}
    optional_keys = {
        "dur_ns",
        "static_smem",
        "dynamic_smem",
        "registers_per_thread",
        "stream",
    }
    if not base_keys.issubset(raw) or set(raw) - base_keys - optional_keys:
        raise ValueError(
            f"C500 projection rank {rank} entry {entry_index} has unsupported keys"
        )
    if ("static_smem" in raw) != ("dynamic_smem" in raw):
        raise ValueError(
            "C500 projection must retain static and dynamic shared memory together"
        )
    shared_memory = _integer(raw["smem"], "C500 projected smem", minimum=0)
    if "static_smem" in raw:
        static = _integer(
            raw["static_smem"], "C500 projected static_smem", minimum=0
        )
        dynamic = _integer(
            raw["dynamic_smem"], "C500 projected dynamic_smem", minimum=0
        )
        if shared_memory != static + dynamic:
            raise ValueError(
                "C500 projected shared-memory total disagrees with its components"
            )
    duration = None
    if "dur_ns" in raw:
        duration = _integer(
            raw["dur_ns"], "C500 projected duration", minimum=0
        )
    registers = None
    if "registers_per_thread" in raw:
        registers = _integer(
            raw["registers_per_thread"],
            "C500 projected registers_per_thread",
            minimum=0,
        )
    stream = None
    if "stream" in raw:
        stream = _integer(raw["stream"], "C500 projected stream", minimum=0)
    name = _nonempty_string(raw["name"], "C500 projected event name")
    if name == "unknown":
        raise ValueError("C500 projection contains a converter placeholder kernel name")
    timestamp_ns = _integer(
        raw["ts_ns"], "C500 projected event timestamp", minimum=0
    )
    if timestamp_ns == 0:
        raise ValueError("C500 projection contains a converter placeholder timestamp")
    return _C500ProjectionEvent(
        name=name,
        timestamp_ns=timestamp_ns,
        duration_ns=duration,
        grid=tuple(_dim3_values(raw["grid"], "C500 projected grid")),
        block=tuple(_dim3_values(raw["block"], "C500 projected block")),
        shared_memory_bytes=shared_memory,
        registers_per_thread=registers,
        stream=stream,
    )


def _normalize_ascend_profiler(
    profiles: Sequence[_AscendRankProfile],
    raw_scheduler: Sequence[Mapping[str, object]],
    scheduler: Sequence[Mapping[str, object]],
    operator_records: Sequence[Mapping[str, object]],
) -> tuple[AvailabilityPayload, AvailabilityPayload]:
    work_steps = _ascend_work_step_order(scheduler, operator_records, profiles)
    first_work_start_ns = _ascend_first_work_scheduler_time_ns(
        raw_scheduler, scheduler
    )
    terminal_rows_by_rank = {
        profile.rank: _ascend_forward_terminals(profile.database)
        for profile in profiles
    }
    anchor_and_terminals = {}
    for profile in profiles:
        terminals = terminal_rows_by_rank[profile.rank]
        before_work = tuple(
            terminal for terminal in terminals if terminal[1] < first_work_start_ns
        )
        if not before_work:
            raise ValueError(
                "910B msprof has no completed forward terminal before the first "
                "non-warmup Scheduler step"
            )
        anchor = before_work[-1]
        work_terminals = tuple(
            terminal for terminal in terminals if terminal[0] > anchor[1]
        )
        if not work_terminals:
            raise ValueError("910B msprof contains no post-warmup forward terminals")
        anchor_and_terminals[profile.rank] = (anchor, work_terminals)
    common_count = min(
        len(value[1]) for value in anchor_and_terminals.values()
    )
    common_count = min(common_count, len(work_steps))
    if common_count <= 0:
        raise ValueError("910B msprof has no all-rank forward prefix")
    first_identities = {
        (value[1][0][2], value[1][0][3])
        for value in anchor_and_terminals.values()
    }
    if len(first_identities) != 1:
        raise ValueError(
            "910B ranks disagree on the first post-warmup forward terminal"
        )
    selected_steps = _ascend_representative_step_ids(
        scheduler, work_steps[:common_count]
    )
    signatures: dict[str, Mapping[str, object]] = {}
    launches = []
    timing = []
    scheduler_by_step = {record["step_id"]: record for record in scheduler}
    covered_by_rank: defaultdict[int, set[str]] = defaultdict(set)
    launch_sequence: defaultdict[tuple[int, str], int] = defaultdict(int)
    interval_sequence: defaultdict[tuple[int, str], int] = defaultdict(int)
    for profile in profiles:
        anchor, terminals = anchor_and_terminals[profile.rank]
        selected_terminals = terminals[:common_count]
        terminal_ends = [terminal[1] for terminal in selected_terminals]
        terminal_counts: defaultdict[str, int] = defaultdict(int)
        task_counts: defaultdict[str, int] = defaultdict(int)
        with sqlite3.connect(
            f"file:{profile.database}?mode=ro&immutable=1", uri=True
        ) as connection:
            string_ids = {
                row[0]: row[1]
                for row in connection.execute("SELECT id, value FROM STRING_IDS")
            }
            device_ids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT deviceId FROM TASK ORDER BY deviceId"
                )
            )
            if device_ids != (profile.device_id,):
                raise ValueError(
                    "910B msprof TASK device does not match its PROF metadata"
                )
            rows = connection.execute(
                "SELECT t.startNs, t.endNs, c.name, c.opType, c.taskType, "
                "c.blockDim, c.mixBlockDim, t.deviceId, t.globalTaskId, "
                "t.streamId, t.taskId FROM TASK t "
                "JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId "
                "ORDER BY t.startNs, t.globalTaskId"
            )
            for raw_index, row in enumerate(rows):
                start_ns = _integer(row[0], "910B task startNs", minimum=0)
                end_ns = _integer(row[1], "910B task endNs", minimum=0)
                if end_ns < start_ns:
                    raise ValueError("910B task endNs precedes startNs")
                if start_ns <= anchor[1]:
                    continue
                terminal_index = bisect_left(terminal_ends, start_ns)
                if terminal_index >= common_count:
                    continue
                if end_ns > terminal_ends[terminal_index]:
                    raise ValueError(
                        "910B task crosses a proven forward terminal boundary"
                    )
                if row[7] != profile.device_id:
                    raise ValueError(
                        "910B task device changes within a rank profile"
                    )
                rank = profile.rank
                step_id = work_steps[terminal_index]
                if step_id not in selected_steps:
                    continue
                phase = _scheduler_step_phase(scheduler_by_step[step_id])
                if row[2] not in string_ids:
                    raise ValueError(
                        "910B compute task name references unknown STRING_IDS ID"
                    )
                name = _nonempty_string(string_ids[row[2]], "910B task name")
                if row[3] not in string_ids:
                    raise ValueError(
                        "910B compute task opType references unknown STRING_IDS ID"
                    )
                op_type = _nonempty_string(
                    string_ids[row[3]], "910B task opType"
                )
                if row[4] not in string_ids:
                    raise ValueError(
                        "910B compute task taskType references unknown STRING_IDS ID"
                    )
                task_type = _nonempty_string(
                    string_ids[row[4]], "910B task taskType"
                )
                block_dim = _integer(row[5], "910B blockDim", minimum=0)
                mix_block_dim = (
                    None
                    if row[6] is None
                    else _integer(row[6], "910B mixBlockDim", minimum=0)
                )
                task_counts[step_id] += 1
                is_terminal = name == _ASCEND_FORWARD_TERMINAL_KERNEL
                if is_terminal:
                    terminal_counts[step_id] += 1
                if block_dim == 0:
                    if (name, op_type, task_type) not in _ASCEND_GEOMETRY_FREE_TASKS:
                        raise ValueError(
                            "910B accelerator task has unavailable blockDim: "
                            f"rank={rank}, step={step_id}, name={name!r}, "
                            f"opType={op_type!r}, taskType={task_type!r}"
                        )
                    signature_payload = {
                        "kernel_name": name,
                        "kernel_family": "AI_CPU/Bincount",
                        "geometry_kind": "ascend_block_dim",
                        "grid": None,
                        "block": None,
                        "ascend_block_dim": 0,
                        "ascend_mix_block_dim": mix_block_dim,
                        "shared_memory_bytes": None,
                        "registers_per_thread": None,
                        "tile": None,
                        "split": None,
                        "stream": None,
                        "device": profile.device_id,
                    }
                    launch_id = _append_launch(
                        signatures,
                        launches,
                        signature_payload,
                        step_id,
                        launch_sequence[(rank, step_id)],
                        rank=rank,
                    )
                    launch_sequence[(rank, step_id)] += 1
                    sequence_idx = interval_sequence[(rank, step_id)]
                    interval_sequence[(rank, step_id)] += 1
                    timing.append(
                        _timing_record(
                            step_id,
                            phase,
                            sequence_idx,
                            "device_aicpu_task",
                            start_ns,
                            end_ns,
                            launch_id,
                            "msprof_host_clock",
                            "scheduler_clock_epoch",
                            f"msprof-rank{rank}-task-{raw_index}",
                            rank=rank,
                            source_archive_id=f"device-profiler-rank{rank}",
                        )
                    )
                    continue
                signature_payload = {
                    "kernel_name": name,
                    "kernel_family": op_type,
                    "geometry_kind": "ascend_block_dim",
                    "grid": None,
                    "block": None,
                    "ascend_block_dim": block_dim,
                    "ascend_mix_block_dim": mix_block_dim,
                    "shared_memory_bytes": None,
                    "registers_per_thread": None,
                    "tile": None,
                    "split": None,
                    "stream": None,
                    "device": profile.device_id,
                }
                launch_id = _append_launch(
                    signatures,
                    launches,
                    signature_payload,
                    step_id,
                    launch_sequence[(rank, step_id)],
                    rank=rank,
                )
                launch_sequence[(rank, step_id)] += 1
                sequence_idx = interval_sequence[(rank, step_id)]
                interval_sequence[(rank, step_id)] += 1
                timing.append(
                    _timing_record(
                        step_id,
                        phase,
                        sequence_idx,
                        "device_kernel",
                        start_ns,
                        end_ns,
                        launch_id,
                        "msprof_host_clock",
                        "scheduler_clock_epoch",
                        f"msprof-rank{rank}-task-{raw_index}",
                        rank=rank,
                        source_archive_id=f"device-profiler-rank{rank}",
                    )
                )
                covered_by_rank[rank].add(step_id)
        for step_id in selected_steps:
            if task_counts[step_id] <= 0:
                raise ValueError("910B forward bucket contains no device tasks")
            if terminal_counts[step_id] != 1:
                raise ValueError(
                    "910B forward bucket must contain exactly one terminal task"
                )
    selected_signature_ids = {record["signature_id"] for record in launches}
    launch_records = tuple(
        signatures[key] for key in sorted(selected_signature_ids)
    ) + tuple(launches)
    covered = {
        step_id
        for step_id in work_steps
        if all(step_id in covered_by_rank[profile.rank] for profile in profiles)
    }
    reason = None
    if selected_steps != set(work_steps):
        reason = (
            "compact alignment retains one complete all-rank representative per "
            "observed phase from the prefix bounded by proven forward-terminal tasks"
        )
    return _profiler_availability(
        launch_records,
        tuple(timing),
        work_steps,
        covered,
        launch_reason=reason,
        timing_reason=(
            "msprof capture ends before the workload and does not provide complete "
            "Host/Scheduler interval attribution"
        ),
        source_archive_ids=tuple(
            f"device-profiler-rank{profile.rank}" for profile in profiles
        ),
    )


def _ascend_work_step_order(
    scheduler: Sequence[Mapping[str, object]],
    operator_records: Sequence[Mapping[str, object]],
    profiles: Sequence[_AscendRankProfile],
) -> tuple[str, ...]:
    work_steps = _work_step_ids(scheduler)
    operator_steps_by_rank: defaultdict[int, list[str]] = defaultdict(list)
    for record in operator_records:
        if record["record_type"] == "signature":
            continue
        if record["record_type"] != "event":
            raise ValueError("operator record_type must be signature or event")
        rank = _integer(record["rank"], "operator rank", minimum=0)
        step_id = _nonempty_string(record["step_id"], "operator step_id")
        if not operator_steps_by_rank[rank] or operator_steps_by_rank[rank][-1] != step_id:
            if step_id in operator_steps_by_rank[rank]:
                raise ValueError("910B operator steps must form contiguous groups")
            operator_steps_by_rank[rank].append(step_id)
    for profile in profiles:
        if tuple(operator_steps_by_rank[profile.rank]) != work_steps:
            raise ValueError(
                "910B operator step order disagrees with work-bearing Scheduler order"
            )
    return work_steps


def _ascend_representative_step_ids(
    scheduler: Sequence[Mapping[str, object]],
    covered_work_steps: Sequence[str],
) -> frozenset[str]:
    covered = set(covered_work_steps)
    candidates: defaultdict[str, list[str]] = defaultdict(list)
    observed_phases = set()
    for record in scheduler:
        if not record["scheduled_requests"]:
            continue
        phase = _scheduler_step_phase(record)
        observed_phases.add(phase)
        scheduled_request_idxs = {
            request["request_idx"] for request in record["scheduled_requests"]
        }
        if (
            record["step_id"] in covered
            and set(record["finished_request_idxs"]).issubset(scheduled_request_idxs)
        ):
            candidates[phase].append(record["step_id"])
    selected = set()
    for phase in observed_phases:
        phase_candidates = candidates[phase]
        if not phase_candidates:
            raise ValueError(
                f"910B captured forward prefix has no complete {phase} step"
            )
        selected.add(phase_candidates[(len(phase_candidates) - 1) // 2])
    return frozenset(selected)


def _ascend_first_work_scheduler_time_ns(
    raw_scheduler: Sequence[Mapping[str, object]],
    scheduler: Sequence[Mapping[str, object]],
) -> int:
    if len(raw_scheduler) != len(scheduler):
        raise ValueError("raw and normalized Scheduler record counts disagree")
    for raw, normalized in zip(raw_scheduler, scheduler, strict=True):
        if normalized["scheduled_requests"]:
            return _integer(raw["ts_us"], "Scheduler ts_us", minimum=0) * 1000
    raise ValueError("910B Scheduler contains no non-warmup work step")


def _ascend_forward_terminals(
    database: Path,
) -> tuple[tuple[int, int, int, int], ...]:
    with sqlite3.connect(
        f"file:{database}?mode=ro&immutable=1", uri=True
    ) as connection:
        rows = connection.execute(
            "SELECT t.startNs, t.endNs, t.streamId, t.taskId FROM TASK t "
            "JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId "
            "JOIN STRING_IDS s ON c.name=s.id WHERE s.value=? "
            "ORDER BY t.startNs, t.globalTaskId",
            (_ASCEND_FORWARD_TERMINAL_KERNEL,),
        ).fetchall()
    terminals = []
    previous_start = None
    for row in rows:
        start_ns = _integer(row[0], "910B terminal startNs", minimum=0)
        end_ns = _integer(row[1], "910B terminal endNs", minimum=0)
        stream_id = _integer(row[2], "910B terminal streamId", minimum=0)
        task_id = _integer(row[3], "910B terminal taskId", minimum=0)
        if end_ns < start_ns:
            raise ValueError("910B terminal endNs precedes startNs")
        if previous_start is not None and start_ns <= previous_start:
            raise ValueError("910B forward terminal timestamps must strictly increase")
        previous_start = start_ns
        terminals.append((start_ns, end_ns, stream_id, task_id))
    if not terminals:
        raise ValueError("910B msprof contains no forward terminal task")
    return tuple(terminals)


def _cuda_profiler_records(
    kernel_rows: Sequence[Sequence[object]],
    copy_rows: Sequence[Sequence[object]],
    windows: Sequence[tuple[int, int, str, bool, str]],
    *,
    kernel_family: str,
    clock_domain: str,
    time_origin: str,
    rank: int = 0,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    set[str],
]:
    signatures: dict[str, Mapping[str, object]] = {}
    launches = []
    timing = []
    covered = set()
    launch_sequence: defaultdict[str, int] = defaultdict(int)
    interval_sequence: defaultdict[str, int] = defaultdict(int)
    starts = [window[0] for window in windows]
    for raw_index, row in enumerate(kernel_rows):
        start_ns = _integer(row[0], "CUDA kernel start", minimum=0)
        end_ns = _integer(row[1], "CUDA kernel end", minimum=0)
        window = _event_window(start_ns, starts, windows)
        if window is None:
            continue
        step_id = window[2]
        signature_payload = {
            "kernel_name": _nonempty_string(row[2], "CUDA kernel name"),
            "kernel_family": kernel_family,
            "geometry_kind": "cuda_dim3",
            "grid": [_integer(value, "CUDA grid", minimum=1) for value in row[3:6]],
            "block": [_integer(value, "CUDA block", minimum=1) for value in row[6:9]],
            "ascend_block_dim": None,
            "ascend_mix_block_dim": None,
            "shared_memory_bytes": _integer(row[9], "CUDA shared memory", minimum=0),
            "registers_per_thread": _integer(row[10], "CUDA registers", minimum=0),
            "tile": None,
            "split": None,
            "stream": row[11] if len(row) > 11 else None,
            "device": row[12] if len(row) > 12 else rank,
        }
        launch_id = _append_launch(
            signatures,
            launches,
            signature_payload,
            step_id,
            launch_sequence[step_id],
            rank=rank,
        )
        launch_sequence[step_id] += 1
        sequence_idx = interval_sequence[step_id]
        interval_sequence[step_id] += 1
        timing.append(
            _timing_record(
                step_id,
                window[4],
                sequence_idx,
                "device_kernel",
                start_ns,
                end_ns,
                launch_id,
                clock_domain,
                time_origin,
                f"kernel-row-{raw_index}",
                rank=rank,
            )
        )
        covered.add(step_id)
    for raw_index, row in enumerate(copy_rows):
        start_ns = _integer(row[0], "CUDA memcpy start", minimum=0)
        end_ns = _integer(row[1], "CUDA memcpy end", minimum=0)
        window = _event_window(start_ns, starts, windows)
        if window is None:
            continue
        step_id = window[2]
        sequence_idx = interval_sequence[step_id]
        interval_sequence[step_id] += 1
        timing.append(
            _timing_record(
                step_id,
                window[4],
                sequence_idx,
                "device_copy",
                start_ns,
                end_ns,
                None,
                clock_domain,
                time_origin,
                f"memcpy-row-{raw_index}",
            )
        )
    timing.sort(key=lambda record: (record["start_ns"], record["interval_id"]))
    return (
        tuple(signatures[key] for key in sorted(signatures)) + tuple(launches),
        tuple(timing),
        covered,
    )


def _scheduler_clock_windows(
    raw_scheduler: Sequence[Mapping[str, object]],
    scheduler: Sequence[Mapping[str, object]],
) -> list[tuple[int, int | None, str, bool, str]]:
    if len(raw_scheduler) != len(scheduler):
        raise ValueError("raw and normalized Scheduler record counts disagree")
    windows = []
    for index, (raw, normalized) in enumerate(zip(raw_scheduler, scheduler, strict=True)):
        lower = _integer(raw["ts_us"], "Scheduler ts_us", minimum=0) * 1000
        upper = None
        if index + 1 < len(raw_scheduler):
            upper = (
                _integer(
                    raw_scheduler[index + 1]["ts_us"],
                    "Scheduler ts_us",
                    minimum=0,
                )
                * 1000
            )
        phases = {request["phase"] for request in normalized["scheduled_requests"]}
        phase = "mixed" if len(phases) != 1 else next(iter(phases))
        windows.append(
            (
                lower,
                upper,
                normalized["step_id"],
                bool(normalized["scheduled_requests"]),
                phase,
            )
        )
    return windows


def _event_window(
    timestamp_ns: int,
    starts: Sequence[int],
    windows: Sequence[Sequence[object]],
) -> Sequence[object] | None:
    position = bisect_right(starts, timestamp_ns) - 1
    if position < 0:
        return None
    window = windows[position]
    upper = window[1]
    if upper is not None and timestamp_ns >= upper:
        return None
    return window


def _work_step_ids(scheduler: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(
        _nonempty_string(record["step_id"], "Scheduler step_id")
        for record in scheduler
        if _sequence(record["scheduled_requests"], "scheduled_requests")
    )


def _scheduler_step_phase(record: Mapping[str, object]) -> str:
    phases = {
        request["phase"]
        for request in _object_sequence(
            record["scheduled_requests"], "scheduled_requests"
        )
    }
    return "mixed" if len(phases) != 1 else next(iter(phases))


def _append_launch(
    signatures: dict[str, Mapping[str, object]],
    launches: list[Mapping[str, object]],
    signature_payload: Mapping[str, object],
    step_id: str,
    sequence_idx: int,
    *,
    rank: int = 0,
) -> str:
    signature_id = _canonical_sha256(signature_payload)
    signatures[signature_id] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "signature",
        "signature_id": signature_id,
        **signature_payload,
    }
    rank_prefix = "" if rank == 0 else f"r{rank}-"
    launch_id = f"launch-{rank_prefix}s{step_id[5:]}-n{sequence_idx:08d}"
    launches.append(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "event",
            "launch_id": launch_id,
            "signature_id": signature_id,
            "rank": rank,
            "step_id": step_id,
            "sequence_idx": sequence_idx,
            "operator_ids": [],
            "link_status": "step_order_only",
        }
    )
    return launch_id


def _timing_record(
    step_id: str,
    phase: str,
    sequence_idx: int,
    interval_kind: str,
    start_ns: int,
    end_ns: int,
    launch_id: str | None,
    clock_domain: str,
    time_origin: str,
    source_id: str,
    *,
    rank: int = 0,
    source_archive_id: str = "device-profiler",
) -> dict[str, object]:
    rank_prefix = "" if rank == 0 else f"r{rank}-"
    return {
        "schema_version": SCHEMA_VERSION,
        "interval_id": f"timing-{rank_prefix}s{step_id[5:]}-n{sequence_idx:08d}",
        "interval_kind": interval_kind,
        "rank": rank,
        "step_id": step_id,
        "launch_id": launch_id,
        "phase": phase,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        "clock_domain": clock_domain,
        "time_origin": time_origin,
        "source_archive_id": source_archive_id,
        "correlation": {
            "status": "exact",
            "source_id": source_id,
            "mapping_source": None,
            "mapping_error_ns": None,
        },
    }


def _profiler_availability(
    launches: Sequence[Mapping[str, object]],
    timing: Sequence[Mapping[str, object]],
    work_steps: Sequence[str],
    covered_steps: set[str],
    *,
    launch_reason: str | None,
    timing_reason: str | None,
    source_archive_ids: tuple[str, ...] = ("device-profiler",),
) -> tuple[AvailabilityPayload, AvailabilityPayload]:
    if not launches or not timing:
        return _unavailable_profiler("profiler source contains no in-scope device events")
    all_steps_covered = set(work_steps).issubset(covered_steps)
    if not set(work_steps).issubset(covered_steps):
        missing_reason = "profiler source does not cover every work-bearing Scheduler step"
        if launch_reason is None:
            launch_reason = missing_reason
        else:
            launch_reason = f"{launch_reason}; {missing_reason}"
        if timing_reason is None:
            timing_reason = missing_reason
        else:
            timing_reason = f"{timing_reason}; {missing_reason}"
    launch_coverage = (
        "complete" if all_steps_covered and launch_reason is None else "partial"
    )
    timing_coverage = (
        "complete" if all_steps_covered and timing_reason is None else "partial"
    )
    return (
        AvailabilityPayload(
            "available",
            launch_coverage,
            launch_reason,
            tuple(launches),
            source_archive_ids,
        ),
        AvailabilityPayload(
            "available",
            timing_coverage,
            timing_reason,
            tuple(timing),
            source_archive_ids,
        ),
    )


def _unavailable_profiler(
    reason: str,
) -> tuple[AvailabilityPayload, AvailabilityPayload]:
    return (
        AvailabilityPayload("unavailable", None, reason, None, None),
        AvailabilityPayload("unavailable", None, reason, None, None),
    )


def _dim3_values(value: object, description: str) -> list[int]:
    values = _sequence(value, description)
    if len(values) != 3:
        raise ValueError(f"{description} must have exactly three dimensions")
    return [_integer(item, description, minimum=1) for item in values]


def _nsys_version(sqlite_path: Path) -> str:
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as connection:
        # Nsight Systems 2023.x exports EXPORT_META_DATA; 2025.x renamed the
        # table to META_DATA_EXPORT with the same name/value rows.
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('EXPORT_META_DATA', 'META_DATA_EXPORT')")
        }
        if len(tables) != 1:
            raise ValueError(f"nsys SQLite export metadata tables are ambiguous: {tables}")
        table = tables.pop()
        row = connection.execute(
            f"SELECT value FROM {table} WHERE name='EXPORT_PRODUCT_VERSION'"
        ).fetchone()
    if row is None:
        raise ValueError("A800 nsys SQLite has no EXPORT_PRODUCT_VERSION")
    return _nonempty_string(row[0], "nsys product version")


def _ascend_capture_runtime(path: Path) -> tuple[str, bool, int | None]:
    if not path.is_file():
        raise FileNotFoundError(f"missing 910B capture log: {path}")
    text = path.read_text(encoding="utf-8")
    matches = tuple(_ASCEND_ENGINE_RUNTIME.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            "910B capture log must contain exactly one EngineCore runtime config"
        )
    match = matches[0]
    version = _nonempty_string(match.group("version"), "910B vLLM version")
    prefix_value = match.group("prefix")
    if prefix_value == "True":
        prefix_enabled = True
    elif prefix_value == "False":
        prefix_enabled = False
    else:
        raise AssertionError(f"unknown parsed prefix-cache value: {prefix_value!r}")
    block_size_matches = tuple(_ASCEND_BLOCK_SIZE_HINT.finditer(text))
    kv_block_size_tokens = None
    parsed_block_sizes = set()
    for block_size_match in block_size_matches:
        raw_block_size = block_size_match.group("tokens")
        try:
            parsed_block_size = int(raw_block_size)
        except ValueError as error:
            raise ValueError(
                "910B capture Block size hint must contain an integer"
            ) from error
        if parsed_block_size <= 0:
            raise ValueError("910B capture Block size hint must be positive")
        parsed_block_sizes.add(parsed_block_size)
    if len(parsed_block_sizes) > 1:
        raise ValueError("910B capture has conflicting Block size hints")
    if parsed_block_sizes:
        kv_block_size_tokens = next(iter(parsed_block_sizes))
    return version, prefix_enabled, kv_block_size_tokens


def _validate_ascend_task_geometry(database: Path) -> None:
    with sqlite3.connect(
        f"file:{database}?mode=ro&immutable=1", uri=True
    ) as connection:
        required = {"TASK", "COMPUTE_TASK_INFO", "STRING_IDS"}
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not required.issubset(tables):
            raise ValueError("910B msprof database is missing per-invocation task tables")
        row = connection.execute(
            "SELECT COUNT(*), SUM(CASE WHEN c.blockDim=0 THEN 1 ELSE 0 END) "
            "FROM TASK t JOIN COMPUTE_TASK_INFO c "
            "ON t.globalTaskId=c.globalTaskId"
        ).fetchone()
        string_ids = {
            string_id: value
            for string_id, value in connection.execute("SELECT id, value FROM STRING_IDS")
        }
        task_strings = connection.execute(
            "SELECT c.name, c.opType FROM TASK t JOIN COMPUTE_TASK_INFO c "
            "ON t.globalTaskId=c.globalTaskId"
        ).fetchall()
    if row is None or row[0] <= 0:
        raise ValueError("910B msprof database contains no compute task events")
    if row[1] is None or row[1] < 0:
        raise AssertionError("invalid zero-blockDim count")
    for name_id, op_type_id in task_strings:
        if name_id not in string_ids:
            raise ValueError("910B compute task name references unknown STRING_IDS ID")
        _nonempty_string(string_ids[name_id], "910B task name")
        if op_type_id not in string_ids:
            raise ValueError("910B compute task opType references unknown STRING_IDS ID")
        _nonempty_string(string_ids[op_type_id], "910B task opType")


def _c500_profiler_source_specs(
    case_dir: Path,
) -> tuple[tuple[Path, str, str, str | None], ...]:
    """Resolve one explicit C500 evidence kind without fallback after errors."""

    trace_dir = case_dir / "mctracer_out"
    if trace_dir.exists():
        trace_path, sync_path = _c500_profiler_paths(case_dir)
        return (
            (trace_path, "device-profiler", "mcTracer", None),
            (
                sync_path,
                "profiler-step-boundaries",
                "mcTracer",
                None,
            ),
        )
    projection_path = case_dir / "nsys_kernels_full.json"
    if not projection_path.is_file():
        raise FileNotFoundError(
            "missing C500 native mcTracer directory and explicit legacy projection; "
            f"case: {case_dir}"
        )
    sync_paths = tuple(sorted(case_dir.glob(".mctracer_sync_ends_*.json")))
    converter = _c500_projection_converter(sync_paths)
    converter_version = (
        f"git:{converter['source_commit']};sha256:{converter['source_sha256']}"
    )
    return (
        (
            projection_path,
            "device-launch-projection",
            converter["implementation_id"],
            converter_version,
        ),
        (
            case_dir / "mctracer_stdout.log",
            "mctracer-capture-log",
            "mcTracer",
            None,
        ),
        *tuple(
            (
                path,
                f"profiler-step-boundaries-rank{_c500_sync_rank(path)}",
                "mcTracer",
                None,
            )
            for path in sync_paths
        ),
    )


def _c500_projection_converter(
    sync_paths: Sequence[Path],
) -> Mapping[str, str]:
    if sync_paths:
        return _C500_REBUILD_CONVERTER
    return _C500_LEGACY_CONVERTER


def _validate_c500_projection_rank_provenance(
    case_dir: Path,
    expected_ranks: tuple[int, ...],
    sync_paths: Sequence[Path],
) -> None:
    stdout_path = case_dir / "mctracer_stdout.log"
    if not stdout_path.is_file():
        raise FileNotFoundError(
            f"missing C500 mcTracer capture evidence: {stdout_path}"
        )
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    pids_by_rank: defaultdict[int, set[int]] = defaultdict(set)
    ep_by_rank: defaultdict[int, set[int | None]] = defaultdict(set)
    for match in _C500_WORKER_PID.finditer(text):
        rank = int(match.group("tp"))
        pids_by_rank[rank].add(int(match.group("pid")))
        ep_by_rank[rank].add(
            None if match.group("ep") is None else int(match.group("ep"))
        )
    if set(pids_by_rank) != set(expected_ranks) or any(
        len(pids_by_rank[rank]) != 1 for rank in expected_ranks
    ):
        raise ValueError(
            "C500 capture log does not identify exactly one worker PID for every TP rank"
        )
    worker_pid_by_rank = {
        rank: next(iter(pids_by_rank[rank])) for rank in expected_ranks
    }
    if len(set(worker_pid_by_rank.values())) != len(expected_ranks):
        raise ValueError("C500 capture log reuses one worker PID across TP ranks")
    for rank in expected_ranks:
        if ep_by_rank[rank] not in ({None}, {rank}):
            raise ValueError(
                f"C500 worker TP rank {rank} has inconsistent EP-rank evidence"
            )
    output_pids = {
        int(match.group("pid")) for match in _C500_TRACER_OUTPUT_PID.finditer(text)
    }
    if not set(worker_pid_by_rank.values()).issubset(output_pids):
        raise ValueError(
            "C500 capture log does not bind every worker PID to an mcTracer output"
        )
    meta = _json_object(case_dir / "meta.json", "C500 case metadata")
    gpu_ids = tuple(
        _integer(value, "C500 meta gpu_id", minimum=0)
        for value in _sequence(meta["gpu_ids"], "C500 meta gpu_ids")
    )
    if len(gpu_ids) != len(expected_ranks) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(
            "C500 meta gpu_ids must identify one unique device for every TP rank"
        )
    worker_names_by_rank = {
        rank: f"mctracer-{worker_pid_by_rank[rank]}.json"
        for rank in expected_ranks
    }
    if not sync_paths:
        sorted_worker_names = sorted(worker_names_by_rank.values())
        if sorted_worker_names != [
            worker_names_by_rank[rank] for rank in expected_ranks
        ]:
            raise ValueError(
                "C500 legacy converter filename ordering disagrees with TP rank evidence"
            )
        return
    trace_dir = case_dir / "mctracer_out"
    if {_c500_sync_rank(path) for path in sync_paths} != set(expected_ranks):
        raise ValueError(
            "C500 rebuild sync sidecars do not retain every device-rank identity"
        )
    missing = [
        worker_names_by_rank[rank]
        for rank in expected_ranks
        if not (trace_dir / worker_names_by_rank[rank]).is_file()
    ]
    if missing:
        if len(missing) != len(expected_ranks):
            raise ValueError(
                "C500 rebuild retains only a partial native mcTracer worker set"
            )
        return
    for rank in expected_ranks:
        device_counts: defaultdict[int, int] = defaultdict(int)
        trace_path = trace_dir / worker_names_by_rank[rank]
        for event in iter_mctracer_device_events(trace_path):
            device_counts[event.device_id] += 1
        if not device_counts:
            raise ValueError(
                f"C500 native mcTracer worker trace for rank {rank} has no device events"
            )
        dominant_count = max(device_counts.values())
        dominant_devices = [
            device for device, count in device_counts.items()
            if count == dominant_count
        ]
        if dominant_devices != [gpu_ids[rank]]:
            raise ValueError(
                f"C500 rebuild device identity disagrees with TP rank {rank}: "
                f"dominant devices {dominant_devices}"
            )


def _c500_sync_rank(path: Path) -> int:
    match = re.fullmatch(r"\.mctracer_sync_ends_(?P<rank>[0-9]+)\.json", path.name)
    if match is None:
        raise ValueError(f"invalid C500 rank sync filename: {path.name}")
    return int(match.group("rank"))


def _c500_profiler_paths(case_dir: Path) -> tuple[Path, Path]:
    trace_dir = case_dir / "mctracer_out"
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"missing C500 native mcTracer directory: {trace_dir}")
    trace_paths = tuple(sorted(
        (*trace_dir.glob("mctracer-*.json"), *trace_dir.glob("tracer_out-*.json"))
    ))
    if not trace_paths:
        raise ValueError(
            "C500 case contains no native mctracer_out trace "
            "(mctracer-*.json or tracer_out-*.json)"
        )
    device_trace_paths = []
    for trace_path in trace_paths:
        events = iter_mctracer_device_events(trace_path)
        try:
            first_device_event = next(events, None)
        finally:
            events.close()
        if first_device_event is not None:
            device_trace_paths.append(trace_path)
    if len(device_trace_paths) != 1:
        raise ValueError(
            "C500 case must contain exactly one mcTracer process file with typed "
            "device events"
        )
    sync_path = case_dir / ".mctracer_sync_ends_0.json"
    if not sync_path.is_file():
        raise FileNotFoundError(f"missing C500 structural step boundaries: {sync_path}")
    return device_trace_paths[0], sync_path


def _c500_sync_windows(
    path: Path,
    scheduler: Sequence[Mapping[str, object]],
) -> tuple[tuple[int, int, str, bool, str], ...]:
    raw_syncs = _sequence(
        _json_loads(path.read_text(encoding="utf-8"), "C500 sync boundaries"),
        "C500 sync boundaries",
    )
    if not raw_syncs:
        raise ValueError("C500 structural sync sidecar is empty")
    if len(raw_syncs) > len(scheduler):
        raise ValueError("C500 structural sync count exceeds Scheduler records")
    windows = []
    previous_end = 0
    for index, raw_sync in enumerate(raw_syncs):
        pair = _sequence(raw_sync, f"C500 sync boundary {index}")
        if len(pair) != 2:
            raise ValueError("C500 sync boundary must contain start and end")
        start = _integer(pair[0], "C500 sync start", minimum=0)
        end = _integer(pair[1], "C500 sync end", minimum=0)
        if end <= start:
            raise ValueError("C500 sync end must follow sync start")
        if start < previous_end:
            raise ValueError("C500 sync boundaries must not overlap")
        scheduler_record = scheduler[index]
        scheduled_requests = _sequence(
            scheduler_record["scheduled_requests"],
            f"Scheduler record {index} scheduled_requests",
        )
        if index == 0 and scheduled_requests:
            raise ValueError(
                "C500 first structural sync must isolate the non-work warmup step"
            )
        windows.append(
            (
                previous_end,
                end,
                _nonempty_string(
                    scheduler_record["step_id"],
                    f"Scheduler record {index} step_id",
                ),
                bool(scheduled_requests),
                (
                    _scheduler_step_phase(scheduler_record)
                    if scheduled_requests
                    else "mixed"
                ),
            )
        )
        previous_end = end
    return tuple(windows)


def _raw_source(
    case_dir: Path,
    path: Path,
    archive_id: str,
    tool: str | None,
    tool_version: str | None,
) -> dict[str, object]:
    try:
        relative = path.relative_to(case_dir).as_posix()
    except ValueError as error:
        raise ValueError("raw source must be contained by its case directory") from error
    return {
        "archive_id": archive_id,
        "role": archive_id,
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "tool": tool,
        "tool_version": tool_version,
    }


def _provenance_gaps(
    runtime: Mapping[str, object],
    raw_sources: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    gaps = [
        {
            "field": "identity.model_revision",
            "reason": "historical raw does not record an immutable model revision",
        },
        {
            "field": "identity.tokenizer_revision",
            "reason": "historical raw does not record an immutable tokenizer revision",
        },
    ]
    runtime_reasons = {
        "dtype": "historical raw does not record effective runtime dtype",
        "deployment": "historical raw does not record a versioned deployment identity",
        "tensor_parallel_size": "historical raw does not record tensor parallel size",
        "expert_parallel_size": "historical raw does not record expert parallel size",
        "pipeline_parallel_size": "historical raw does not record pipeline parallel size",
        "target_qps": "historical raw does not record target QPS",
        "seed": "historical raw does not identify one runtime seed contract",
        "scheduler_config_sha256": "historical raw has no canonical Scheduler config digest",
        "runtime_version": "historical raw does not record runtime version",
        "source_commit": "historical raw does not record source commit",
        "capture_enable_prefix_caching": (
            "historical raw does not record whether prefix caching was enabled"
        ),
    }
    for key, value in runtime.items():
        if value is None:
            gaps.append({"field": f"runtime.{key}", "reason": runtime_reasons[key]})
    for source in raw_sources:
        archive_id = source["archive_id"]
        if source["tool"] is None:
            gaps.append(
                {
                    "field": f"raw_sources.{archive_id}.tool",
                    "reason": "historical raw does not identify the producing tool",
                }
            )
        if source["tool_version"] is None:
            gaps.append(
                {
                    "field": f"raw_sources.{archive_id}.tool_version",
                    "reason": "historical raw does not record the producing tool version",
                }
            )
    has_c500_projection = any(
        source["archive_id"] == "device-launch-projection" for source in raw_sources
    )
    has_native_c500_mctracer = any(
        source["archive_id"] == "device-profiler"
        and source["tool"] == "mcTracer"
        for source in raw_sources
    )
    if has_native_c500_mctracer and not has_c500_projection:
        gaps.append(
            {
                "field": "alignment.launch_signatures.stream",
                "reason": (
                    "native mcTracer normalization does not preserve stream identity, "
                    "so stream is published as null"
                ),
            }
        )
    if has_c500_projection:
        rebuild_projection = any(
            str(source["archive_id"]).startswith("profiler-step-boundaries-rank")
            for source in raw_sources
        )
        rank_rule = (
            "device-derived rank grouping"
            if rebuild_projection
            else "worker-filename-order rank grouping"
        )
        gaps.extend(
            (
                {
                    "field": "raw_sources.native_mctracer_device_events",
                    "reason": (
                        "the retained legacy projection has no immutable native "
                        "mcTracer worker-event files; capture and converter identity "
                        "are retained separately"
                    ),
                },
                {
                    "field": "alignment.launch_signatures.geometry",
                    "reason": (
                        "historical projection converters supplied defaults when "
                        "native grid, block or shared-memory fields were absent; "
                        "the native worker events are unavailable, so geometry is "
                        "published as unavailable"
                    ),
                },
                {
                    "field": "alignment.launch_signatures.device",
                    "reason": (
                        "historical projection converters did not preserve a "
                        "per-event device identity, so device is published as null"
                    ),
                },
                {
                    "field": "alignment.launch_events.rank",
                    "reason": (
                        "projection rank association is reconstructed from the "
                        f"version-pinned converter {rank_rule} and worker PID log; "
                        "native worker files are unavailable"
                    ),
                },
                {
                    "field": "alignment.device_timing",
                    "reason": (
                        "historical projection converters could supply zero when a "
                        "native duration was absent, so projection timing is not published"
                    ),
                },
                {
                    "field": "alignment.launch_signatures.stream",
                    "reason": (
                        "the retained mcTracer-derived legacy projection does not "
                        "preserve stream identity"
                    ),
                },
                {
                    "field": "alignment.launch_signatures.geometry.registers_per_thread",
                    "reason": (
                        "the retained mcTracer-derived legacy projection does not "
                        "preserve register allocation"
                    ),
                },
            )
        )
    return gaps


def _json_object(path: Path, description: str) -> dict[str, object]:
    value = _json_loads(path.read_text(encoding="utf-8"), description)
    result = dict(_mapping(value, description))
    result["__source_path"] = str(path)
    return result


def _jsonl_objects(path: Path, description: str) -> tuple[Mapping[str, object], ...]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{description} contains a blank line at {line_number}")
            records.append(
                _mapping(
                    _json_loads(line, f"{description} line {line_number}"),
                    f"{description} line {line_number}",
                )
            )
    if not records:
        raise ValueError(f"{description} is empty")
    return tuple(records)


def _json_loads(text: str, description: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{description} contains invalid constant {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description}: {error}") from error


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    return reject_duplicate_json_keys(pairs)






def _object_sequence(
    value: object,
    description: str,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _mapping(item, f"{description}[{index}]")
        for index, item in enumerate(_sequence(value, description))
    )


def _exact_keys(
    value: Mapping[str, object],
    expected: Sequence[str],
    description: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(
            f"{description} keys must exactly equal {tuple(expected)}, got {tuple(value)}"
        )


def _integer(value: object, description: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer")
    if value < minimum:
        raise ValueError(f"{description} must be >= {minimum}")
    return value


def _number(value: object, description: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{description} must be {qualifier}")
    return result


def _optional_number(value: object, description: str) -> float | None:
    if value is None:
        return None
    return _number(value, description, positive=False)


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{description} must be a non-empty string")
    return value


def _nonempty_or_null(value: object, description: str) -> str | None:
    if value == "":
        return None
    return _nonempty_string(value, description)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
