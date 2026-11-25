"""Reconcile a complete Scheduler/Graph Serving recapture suite."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Mapping

from .compact_case import ALIGNMENT_FILENAME, load_compact_case

from dataset_pipeline.serving_dataset.json_shape import (
    compact_mapping as _mapping,
    file_sha256,
)


_CANARY_RELATIVE_CASE = Path("qwen3-8b/tp1_qps4")
_OFFLOAD_FLAGS = (
    "--cpu-offload-gb",
    "--kv-offloading-size",
    "--kv-offloading-backend",
)
_LOG_TIMESTAMP = re.compile(
    r"\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL) "
    r"(?P<timestamp>[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})\b"
)


def reconcile_scheduler_capture_suite(
    *,
    canonical_hardware_root: str | Path,
    run_root: str | Path,
    candidate_root: str | Path,
    canary_run: str | Path,
    expected_case_count: int,
    baseline_commit: str,
    execution_snapshot: str,
    container_name: str,
    container_image_id: str,
    container_image: str,
) -> Mapping[str, object]:
    """Validate all cases and return a compact machine-readable report."""

    canonical_root = Path(canonical_hardware_root).resolve(strict=True)
    runs = Path(run_root).resolve(strict=True)
    candidates = Path(candidate_root).resolve(strict=True)
    canary = Path(canary_run).resolve(strict=True)
    case_dirs = sorted(path.parent for path in canonical_root.glob("*/*/manifest.json"))
    if len(case_dirs) != expected_case_count:
        raise ValueError(
            f"expected {expected_case_count} canonical cases, found {len(case_dirs)}"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", baseline_commit):
        raise ValueError("baseline_commit must be a full lowercase Git commit")
    results = []
    source_diff_hashes = set()
    runtime_versions = set()
    for canonical_dir in case_dirs:
        relative_case = canonical_dir.relative_to(canonical_root)
        run_dir = canary if relative_case == _CANARY_RELATIVE_CASE else runs / relative_case
        candidate_dir = candidates / canonical_root.name / relative_case
        canonical = load_compact_case(canonical_dir)
        candidate = load_compact_case(candidate_dir)
        if candidate.scheduler_steps is None:
            raise ValueError(f"candidate has no Scheduler trace: {relative_case}")
        for filename in ("requests.csv", ALIGNMENT_FILENAME):
            if (candidate_dir / filename).read_bytes() != (canonical_dir / filename).read_bytes():
                raise ValueError(f"{relative_case} did not byte-reuse {filename}")
        ground_truth_path = run_dir / "ground_truth.json"
        ground_truth = _json_object(ground_truth_path, "ground-truth result")
        if ground_truth["status"] != "success":
            raise ValueError(f"ground-truth run failed: {relative_case}")
        metrics = _mapping(ground_truth["metrics"], "ground-truth metrics")
        provenance = _mapping(ground_truth["provenance"], "ground-truth provenance")
        if provenance["git_commit"] != baseline_commit:
            raise ValueError(f"baseline commit mismatch: {relative_case}")
        source_diff_hashes.add(provenance["git_diff_hash"])
        if metrics["completed_requests"] != len(canonical.requests):
            raise ValueError(f"request completion mismatch: {relative_case}")
        if metrics["max_dispatch_lag_ms"] > 100.0:
            raise ValueError(f"dispatch lag exceeds 100 ms: {relative_case}")
        workload = _mapping(ground_truth["workload"], "ground-truth workload")
        prefix_evidence = _mapping(
            workload["prefix_cache_reuse_evidence"],
            "prefix-cache reuse evidence",
        )
        if (
            prefix_evidence["enable_prefix_caching"] is not True
            or prefix_evidence["shared_prefix_block_pair_count"] != 0
            or prefix_evidence["max_shared_prefix_blocks"] != 0
            or prefix_evidence["effective_reuse_representative_tokens"] != 0
        ):
            raise ValueError(f"prefix-cache reuse evidence failed: {relative_case}")
        canonical_manifest = _mapping(canonical.manifest, "canonical manifest")
        candidate_runtime = _mapping(
            candidate.manifest["runtime"], "candidate runtime")
        configured_depth = candidate_runtime["scheduler_pipeline_depth"]
        if (isinstance(configured_depth, bool)
                or not isinstance(configured_depth, int)
                or configured_depth < 1):
            raise ValueError(
                f"candidate depth is invalid: {relative_case}")
        canonical_requests = _mapping(canonical_manifest["requests"], "canonical requests")
        if workload["requests_sha256"] != canonical_requests["sha256"]:
            raise ValueError(f"request stream hash mismatch: {relative_case}")
        if workload["case_manifest_sha256"] != _sha256(canonical_dir / "manifest.json"):
            raise ValueError(f"case manifest hash mismatch: {relative_case}")
        server = _mapping(ground_truth["server"], "ground-truth server")
        runtime_versions.add(server["client_vllm_version"])
        engine = _mapping(server["engine"], "ground-truth engine")
        if (
            engine["dtype"] != "bfloat16"
            or engine["enable_prefix_caching"] is not True
            or engine["async_scheduling"] is not True
            or engine["configured_cudagraph_mode"] != "FULL"
            or engine["graph_mode"] != "acl_graph_full"
        ):
            raise ValueError(f"runtime configuration mismatch: {relative_case}")
        argv = server["argv"]
        explicit_offload_flags = [
            argument for argument in argv
            if any(
                argument == flag or argument.startswith(f"{flag}=")
                for flag in _OFFLOAD_FLAGS
            )
        ]
        if explicit_offload_flags:
            raise ValueError(f"offload was enabled: {relative_case}")
        if _mapping(server["profiler"], "ground-truth profiler")["requested"] is not False:
            raise ValueError(f"profiler was attached: {relative_case}")
        runtime = _mapping(server["runtime"], "ground-truth runtime")
        log_paths = (
            Path(runtime["stdout_log"]), Path(runtime["stderr_log"])
        )
        log_start, log_end = _log_time_window(log_paths)
        stdout_text = log_paths[0].read_text(errors="replace")
        moe_runtime_marker = "fused_moe.py" in stdout_text
        if relative_case.parts[0] == "qwen3-30b-a3b" and not moe_runtime_marker:
            raise ValueError(f"MoE runtime marker is missing: {relative_case}")
        log_window = _mapping(runtime["benchmark_log_window"], "benchmark log window")
        if log_window["status"] != "validated" or log_window["jit_warning_count"] != 0:
            raise ValueError(f"benchmark window contains JIT: {relative_case}")
        async_validation = _mapping(
            runtime["async_scheduling_validation"], "async validation"
        )
        if (
            async_validation["status"] != "validated"
        ):
            raise ValueError(f"async depth validation failed: {relative_case}")
        if (runtime["scheduler_pipeline_depth"] != configured_depth
                or runtime["scheduler_pipeline_depth_evidence"]
                != candidate_runtime["scheduler_pipeline_depth_evidence"]):
            raise ValueError(
                f"live Executor depth evidence disagrees with candidate: "
                f"{relative_case}")
        rows = candidate.scheduler_steps
        if any(row["configured_cudagraph_mode"] != "FULL" for row in rows):
            raise ValueError(f"configured Graph mode is not FULL: {relative_case}")
        if any(row["capture_limit_tokens"] != 512 for row in rows):
            raise ValueError(f"capture limit is not 512: {relative_case}")
        max_in_flight = max(row["in_flight_step_count"] for row in rows)
        if max_in_flight > configured_depth:
            raise ValueError(
                f"trace occupancy exceeds configured depth: {relative_case}")
        expected_tokens = sum(
            row.input_tokens - row.cached_hit_tokens
            + max(row.output_tokens - 1, 0)
            for row in canonical.requests
        )
        if rows[-1]["cumulative_scheduled_tokens"] != expected_tokens:
            raise ValueError(f"suite token conservation failed: {relative_case}")
        graph_counts = Counter(row["runtime_cudagraph_mode"] for row in rows)
        fallback_counts = Counter(
            row["graph_fallback_reason"]
            for row in rows
            if row["runtime_cudagraph_mode"] == "NONE"
        )
        scheduler_summary = _mapping(
            ground_truth["scheduler_trace"], "ground-truth Scheduler summary"
        )
        if dict(graph_counts) != {
            key: value
            for key, value in scheduler_summary["runtime_cudagraph_mode_counts"].items()
            if value
        }:
            raise ValueError(f"Graph mode summary mismatch: {relative_case}")
        warmup = _mapping(scheduler_summary["warmup"], "warmup summary")
        warmup_request_ids = warmup["request_ids"]
        if (
            warmup["max_scheduled_request_count"] < 8
            or warmup["mixed_prefill_decode_step_count"] < 1
            or len(warmup_request_ids) != len(set(warmup_request_ids))
            or any("warmup-" not in request_id for request_id in warmup_request_ids)
        ):
            raise ValueError(f"warmup coverage failed: {relative_case}")
        results.append({
            "case": str(relative_case),
            "completed_requests": metrics["completed_requests"],
            "ttft_mean_ms": metrics["ttft"]["mean_ms"],
            "tpot_mean_ms": metrics["tpot"]["mean_ms"],
            "e2e_mean_ms": metrics["e2e"]["mean_ms"],
            "request_throughput_per_s": metrics["request_throughput_per_s"],
            "output_token_throughput_per_s": metrics["output_token_throughput_per_s"],
            "max_dispatch_lag_ms": metrics["max_dispatch_lag_ms"],
            "scheduler_step_count": len(rows),
            "scheduled_tokens": expected_tokens,
            "longest_device_execution_ms": max(
                row["device_execution_end_ns"] - row["device_execution_start_ns"]
                for row in rows
            ) / 1_000_000.0,
            "runtime_cudagraph_mode_counts": dict(sorted(graph_counts.items())),
            "graph_fallback_counts": dict(sorted(fallback_counts.items())),
            "padding_tokens_total": sum(row["padding_tokens"] for row in rows),
            "padding_tokens_max_step": max(row["padding_tokens"] for row in rows),
            "max_in_flight_step_count": max_in_flight,
            "scheduler_pipeline_depth": configured_depth,
            "warmup_max_scheduled_request_count": warmup[
                "max_scheduled_request_count"
            ],
            "warmup_mixed_prefill_decode_step_count": warmup[
                "mixed_prefill_decode_step_count"
            ],
            "jit_warning_count": log_window["jit_warning_count"],
            "dtype": engine["dtype"],
            "prefix_cache_enabled": engine["enable_prefix_caching"],
            "prefix_cache_shared_block_pairs": prefix_evidence[
                "shared_prefix_block_pair_count"
            ],
            "prefix_cache_effective_reuse_tokens": prefix_evidence[
                "effective_reuse_representative_tokens"
            ],
            "explicit_offload_flags": explicit_offload_flags,
            "profiler_attached": False,
            "requests_sha256": canonical_requests["sha256"],
            "alignment_steps_sha256": _mapping(
                canonical_manifest["alignment_steps"], "canonical alignment"
            )["sha256"],
            "ground_truth_sha256": _sha256(ground_truth_path),
            "scheduler_trace_sha256": scheduler_summary["sha256"],
            "candidate_manifest_sha256": _sha256(candidate_dir / "manifest.json"),
            "devices": server["devices"],
            "runner_argv": provenance["argv"],
            "server_argv": server["argv"],
            "server_pid": runtime["pid"],
            "log_start_local": log_start,
            "log_end_local": log_end,
            "log_timezone": "Asia/Shanghai",
            "stdout_sha256": runtime["stdout_sha256"],
            "stderr_sha256": runtime["stderr_sha256"],
            "moe_runtime_marker": moe_runtime_marker,
        })
    return {
        "schema_version": 1,
        "status": "passed",
        "hardware_id": canonical_root.name,
        "baseline_commit": baseline_commit,
        "execution_snapshot": execution_snapshot,
        "container": {
            "name": container_name,
            "image_id": container_image_id,
            "image": container_image,
        },
        "runtime_versions": sorted(runtime_versions),
        "source_snapshot_diff_hashes": sorted(source_diff_hashes),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_case_count": expected_case_count,
        "completed_case_count": len(results),
        "cases": results,
    }


def _log_time_window(paths: tuple[Path, Path]) -> tuple[str, str]:
    timestamps = []
    for path in paths:
        for match in _LOG_TIMESTAMP.finditer(path.read_text(errors="replace")):
            timestamps.append(match.group("timestamp"))
    if not timestamps:
        raise ValueError("server logs contain no parseable timestamps")
    return min(timestamps), max(timestamps)


def _json_object(path: Path, description: str) -> Mapping[str, object]:
    value = json.loads(path.read_text())
    return _mapping(value, description)




def _sha256(path: Path) -> str:
    return file_sha256(path)
