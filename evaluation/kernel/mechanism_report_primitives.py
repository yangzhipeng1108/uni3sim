"""Row primitives for kernel mechanism reports: metric summaries,
comparison classification, task/kernel ordinals, CSV and result
identity helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

from evaluation.kernel.mechanism_capture import (
    BUCKET_SCHEMA_VERSION,
    COMPARISON_STATUSES,
    PHYSICAL_BUCKET_DISCRETE_CONTROLS,
    SUPPORT_LEVELS,
    TASK_AGGREGATION_CONTRACT,
    canonical_capture_comparison_evidence,
    sha256,
)


def metric_summary(pairs: Sequence[tuple[float, float]]) -> Mapping[str, Any]:
    if not pairs:
        raise ValueError("metric summary requires at least one pair")
    predicted = np.asarray([pair[0] for pair in pairs], dtype=float)
    observed = np.asarray([pair[1] for pair in pairs], dtype=float)
    if (not np.all(np.isfinite(predicted))
            or not np.all(np.isfinite(observed))):
        raise ValueError("latency metrics require finite values")
    if np.any(predicted <= 0) or np.any(observed <= 0):
        raise ValueError("latency metrics require positive values")
    signed = (predicted - observed) / observed * 100.0
    absolute = np.abs(signed)
    return {
        "case_count": len(pairs),
        "mape_pct": float(np.mean(absolute)),
        "mdape_pct": float(np.percentile(absolute, 50)),
        "p50_ape_pct": float(np.percentile(absolute, 50)),
        "p90_ape_pct": float(np.percentile(absolute, 90)),
        "p99_ape_pct": float(np.percentile(absolute, 99)),
        "max_ape_pct": float(np.max(absolute)),
        "weighted_ape_pct": float(
            np.sum(np.abs(predicted - observed)) / np.sum(observed) * 100.0),
        "mean_signed_error_pct": float(np.mean(signed)),
        "median_signed_error_pct": float(np.percentile(signed, 50)),
    }


def format_percentage(value: float | None, *, signed: bool = False) -> str:
    """Format an optional percentage without turning missing evidence into zero."""

    if value is None:
        return "n/a"
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("percentage display value must be finite")
    if signed:
        return f"{numeric:+.1f}%"
    return f"{numeric:.1f}%"


def result_scope_title(result_scope: str) -> str:
    if result_scope == "real_kernel":
        return "Real kernel invocation"
    elif result_scope == "aggregate_task":
        return "Complete task"
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")


def classify_comparison_status(row: Mapping[str, Any]) -> str:
    status = row["comparison_status"]
    if status not in COMPARISON_STATUSES:
        raise ValueError(f"unknown accuracy comparison status: {status!r}")
    reason = row["comparison_reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("accuracy comparison reason must be non-empty")
    return status


def observed_duration_ns(
        row: Mapping[str, Any], result_scope: str) -> float:
    if result_scope == "real_kernel":
        return float(row["actual_ns"])
    elif result_scope == "aggregate_task":
        return float(row["observed_ns"])
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")


def kernel_ordinal_for_scope(
        row: Mapping[str, Any], result_scope: str) -> int | None:
    if result_scope == "real_kernel":
        ordinal = row["kernel_ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("real-kernel ordinal must be an integer")
        if ordinal < 0:
            raise ValueError("real-kernel ordinal must be non-negative")
        return ordinal
    elif result_scope == "aggregate_task":
        return None
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")


def kernel_ordinal_sort_value(
        row: Mapping[str, Any], result_scope: str) -> int:
    ordinal = kernel_ordinal_for_scope(row, result_scope)
    if ordinal is None:
        return -1
    return ordinal


def kernel_ordinal_suffix(
        row: Mapping[str, Any], result_scope: str) -> str:
    ordinal = kernel_ordinal_for_scope(row, result_scope)
    if ordinal is None:
        return ""
    return f" k{ordinal}"


def plot_case_id(case_id: str, width: int = 34) -> str:
    """Wrap a case identifier at underscores without changing its text."""

    if not isinstance(case_id, str) or not case_id:
        raise ValueError("plot case identifier must be a non-empty string")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("plot case identifier width must be positive")
    tokens = case_id.split("_")
    lines = []
    current = tokens[0]
    for token in tokens[1:]:
        candidate = f"{current}_{token}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = token
    lines.append(current)
    return "\n".join(lines)


def support_level(kernels: Sequence[Mapping[str, Any]]) -> str:
    ranks = {level: rank for rank, level in enumerate(SUPPORT_LEVELS)}
    levels = [
        kernel["mechanisms"]["forward"]["support_level"]
        for kernel in kernels
    ]
    unknown = set(levels) - set(ranks)
    if unknown:
        raise ValueError(f"unknown Forward support levels: {sorted(unknown)}")
    return max(levels, key=lambda value: ranks[value])


def kernel_launch_class(mechanism: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("hardware_id", "kernel_kind", "variant", "dtype"):
        value = mechanism[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"physical bucket requires non-empty {name}: {value!r}")
    launch = mechanism["observed_launch"]
    if not isinstance(launch, Mapping):
        raise TypeError("physical bucket launch must be a mapping")
    grid = launch["grid"]
    block = launch["block"]
    if (not isinstance(grid, list) or len(grid) != 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in grid)):
        raise ValueError("physical bucket requires a positive 3-D launch grid")
    if (not isinstance(block, list) or len(block) != 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in block)):
        raise ValueError("physical bucket requires a positive 3-D launch block")
    for name in ("algorithm", "execution_variant"):
        value = launch[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"physical bucket requires non-empty launch {name}: "
                f"{value!r}")
    cluster = launch["cluster"]
    if (cluster is not None
            and (not isinstance(cluster, list) or len(cluster) != 3
                 or any(isinstance(value, bool)
                        or not isinstance(value, int) or value <= 0
                        for value in cluster))):
        raise ValueError(
            "physical bucket launch cluster must be null or a positive 3-D "
            "list")
    source_program_type = launch["source_program_type"]
    if (source_program_type is not None
            and (not isinstance(source_program_type, str)
                 or not source_program_type.strip())):
        raise ValueError(
            "physical bucket source_program_type must be null or a non-empty "
            "string")
    tiling = launch["tiling"]
    if tiling is not None:
        if not isinstance(tiling, Mapping):
            raise TypeError("physical bucket tiling must be a mapping or null")
        expected_tiling_fields = {
            "tile_shape", "pipeline_depth", "split_k", "num_kv_splits",
        }
        if set(tiling) != expected_tiling_fields:
            raise ValueError(
                "physical bucket tiling fields must be exactly "
                f"{sorted(expected_tiling_fields)}")
        tile_shape = tiling["tile_shape"]
        if (not isinstance(tile_shape, Mapping)
                or any(not isinstance(name, str) or not name
                       for name in tile_shape)
                or any(isinstance(value, bool)
                       or not isinstance(value, int) or value <= 0
                       for value in tile_shape.values())):
            raise ValueError(
                "physical bucket tile_shape must be an explicit mapping of "
                "positive integer dimensions; an empty mapping denotes a "
                "source launch with no tile-shape fields")
        for name in ("pipeline_depth", "split_k", "num_kv_splits"):
            value = tiling[name]
            if (value is not None
                    and (isinstance(value, bool)
                         or not isinstance(value, int) or value <= 0)):
                raise ValueError(
                    f"physical bucket tiling {name} must be null or a "
                    f"positive integer: {value!r}")
    parameters = launch["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("physical bucket launch parameters must be a mapping")
    controls = {}
    for name in PHYSICAL_BUCKET_DISCRETE_CONTROLS:
        if name in parameters:
            controls[name] = parameters[name]
        else:
            controls[name] = None
    for name, value in controls.items():
        if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not float(value).is_integer()):
            raise ValueError(
                f"physical bucket control {name} must be a finite integer "
                f"or null: {value!r}")
    return {
        "hardware_id": mechanism["hardware_id"],
        "kernel_kind": mechanism["kernel_kind"],
        "variant": mechanism["variant"],
        "dtype": mechanism["dtype"],
        "execution_variant": launch["execution_variant"],
        "algorithm": launch["algorithm"],
        "grid_rank": max(
            1, sum(dimension > 1 for dimension in grid)),
        "block": block,
        "cluster": cluster,
        "source_program_type": source_program_type,
        "tiling": tiling,
        "discrete_controls": controls,
    }


def task_execution_signature(task: Mapping[str, Any]) -> str:
    signature = []
    for kernel in task["kernels"]:
        signature.append(kernel_launch_class(kernel["mechanisms"]))
    if not signature:
        raise ValueError("task execution signature requires at least one kernel")
    return json.dumps(signature, separators=(",", ":"), sort_keys=True)


def physical_bucket(
        *, hardware_id: str, family: str, metadata: Mapping[str, Any],
        result_scope: str, execution_signature: str) -> Mapping[str, Any]:
    if result_scope not in ("real_kernel", "aggregate_task"):
        raise ValueError(f"unknown result scope: {result_scope!r}")
    dimensions = {
        "schema": BUCKET_SCHEMA_VERSION,
        "result_scope": result_scope,
        "hardware_id": hardware_id,
        "execution_signature": execution_signature,
    }
    if family == "FlashAttention":
        layout = metadata["attention_layout"]
        if not isinstance(layout, str) or not layout:
            raise ValueError("FlashAttention bucket requires a layout")
        dimensions["layout"] = layout
    elif family in ("GEMM", "FusedMoE", "Vector"):
        pass
    else:
        raise ValueError(f"unsupported physical-bucket family: {family!r}")
    if result_scope == "real_kernel":
        bucket_basis = "static_kernel_launch_class"
    elif result_scope == "aggregate_task":
        bucket_basis = "ordered_static_launch_graph"
    else:
        raise AssertionError(f"unhandled result scope: {result_scope!r}")
    bucket_dimensions = json.dumps(
        dimensions, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(bucket_dimensions.encode("utf-8")).hexdigest()
    bucket_id = f"{BUCKET_SCHEMA_VERSION}:{result_scope}:{digest}"
    return {
        "bucket_schema": BUCKET_SCHEMA_VERSION,
        "physical_bucket_id": bucket_id,
        "physical_bucket_short_id": digest[:12],
        "bucket_dimensions": bucket_dimensions,
        "bucket_basis": bucket_basis,
    }


def task_axis_summary(
        metadata: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    result = {
        "gemm_m": None,
        "gemm_n": None,
        "gemm_k": None,
        "query_length": None,
        "kv_length": None,
        "executed_attention_splits": None,
        "attention_layout": None,
        "moe_experts": None,
        "moe_topk": None,
        "moe_max_tokens_per_expert": None,
        "vector_tokens": None,
    }
    if family == "GEMM":
        result["gemm_m"] = metadata["gemm_m"]
        result["gemm_n"] = metadata["gemm_n"]
        result["gemm_k"] = metadata["gemm_k"]
    elif family == "FlashAttention":
        result["query_length"] = metadata["query_length"]
        result["kv_length"] = metadata["kv_length"]
        result["executed_attention_splits"] = (
            metadata["executed_attention_splits"])
        result["attention_layout"] = metadata["attention_layout"]
    elif family == "FusedMoE":
        result["moe_experts"] = metadata["moe_experts"]
        result["moe_topk"] = metadata["moe_topk"]
        result["moe_max_tokens_per_expert"] = (
            metadata["moe_max_tokens_per_expert"])
    elif family == "Vector":
        result["vector_tokens"] = metadata["vector_tokens"]
    else:
        raise ValueError(f"unsupported report family: {family!r}")
    return result


def task_rows(documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        comparison_evidence = canonical_capture_comparison_evidence(
            hardware_id)
        for task in document["tasks"]:
            predicted_ns, observed_ns = task["forward_pair"]
            family = task["family"]
            metadata = task["metadata"]
            variant_signature = "/".join(
                kernel["mechanisms"]["variant"]
                for kernel in task["kernels"]
            )
            execution_signature = task_execution_signature(task)
            bucket = physical_bucket(
                hardware_id=hardware_id,
                family=family,
                metadata=metadata,
                result_scope="aggregate_task",
                execution_signature=execution_signature,
            )
            rows.append({
                "result_scope": "aggregate_task",
                "hardware_id": hardware_id,
                "family": family,
                "case_id": task["case_id"],
                "predicted_ns": predicted_ns,
                "observed_ns": observed_ns,
                "timing_source": metadata["actual_source"],
                "signed_error_pct": (
                    (predicted_ns - observed_ns) / observed_ns * 100.0),
                "absolute_error_pct": (
                    abs(predicted_ns - observed_ns) / observed_ns * 100.0),
                "absolute_duration_error_ns": abs(
                    predicted_ns - observed_ns),
                **comparison_evidence,
                "support_level": support_level(task["kernels"]),
                "kernel_count": len(task["kernels"]),
                "aggregation_contract": TASK_AGGREGATION_CONTRACT,
                "observed_timing_granularity": metadata[
                    "actual_granularity"],
                "variant_signature": variant_signature,
                "task_execution_signature": execution_signature,
                "scale": metadata["eqm"],
                "max_derived_launch_rounds": metadata["max_launch_waves"],
                **task_axis_summary(metadata, family),
                **bucket,
            })
    return rows


def error_direction(predicted_ns: float, observed_ns: float) -> str:
    if predicted_ns < observed_ns:
        return "under"
    if predicted_ns > observed_ns:
        return "over"
    return "exact"


def memory_path_bytes(
        analytical: Mapping[str, Any], *, access_kind: str,
        resource_role: str) -> float:
    """Sum one directional resource interface from the Forward path ledger."""

    if access_kind not in ("read", "write"):
        raise ValueError(
            f"memory-path access kind must be read or write: {access_kind!r}")
    paths = analytical["memory_paths"]
    if not isinstance(paths, list):
        raise TypeError("analytical memory_paths must be a list")
    total = 0.0
    for path in paths:
        path_kind = path["access_kind"]
        if path_kind not in ("read", "write"):
            raise ValueError(
                f"unknown analytical memory-path access kind: {path_kind!r}")
        nodes = path["nodes"]
        if not isinstance(nodes, list):
            raise TypeError("analytical memory-path nodes must be a list")
        if path_kind != access_kind:
            continue
        for node in nodes:
            if node["resource_role"] != resource_role:
                continue
            byte_count = float(node["bytes"])
            if not math.isfinite(byte_count) or byte_count < 0:
                raise ValueError(
                    "analytical memory-path bytes must be finite and "
                    f"non-negative: {byte_count!r}")
            total += byte_count
    return total


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields_: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields_)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields_})


def read_result_csv(path: Path, required_fields: Sequence[str]) -> list[Mapping[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"comparison result is missing: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"comparison result has no header: {path}")
        missing = sorted(set(required_fields) - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"comparison result {path} is missing fields: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"comparison result has no rows: {path}")
    return rows


def verify_manifest_artifact(
        results_dir: Path,
        manifest: Mapping[str, Any],
        relative_path: str) -> None:
    """Fail closed when a comparison input is not bound by its manifest."""

    artifact_hashes = manifest["artifact_sha256"]
    if not isinstance(artifact_hashes, Mapping):
        raise TypeError("before-results artifact_sha256 must be a mapping")
    if relative_path not in artifact_hashes:
        raise ValueError(
            "before-results manifest does not bind comparison artifact: "
            f"{relative_path}")
    expected = artifact_hashes[relative_path]
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(
            "before-results manifest has an invalid SHA256 for "
            f"{relative_path}: {expected!r}")
    path = results_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"comparison result is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            "before-results comparison artifact SHA256 mismatch for "
            f"{relative_path}: expected {expected}, got {actual}")


def result_join_key(
        row: Mapping[str, Any], result_scope: str) -> tuple[Any, ...]:
    if result_scope == "real_kernel":
        ordinal = int(row["kernel_ordinal"])
        if ordinal < 0:
            raise ValueError("comparison kernel ordinal must be non-negative")
        return (
            result_scope,
            row["hardware_id"],
            row["family"],
            row["op_type"],
            row["case_id"],
            ordinal,
        )
    elif result_scope == "aggregate_task":
        return (
            result_scope,
            row["hardware_id"],
            row["family"],
            row["case_id"],
        )
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")


def index_result_rows(
        rows: Sequence[Mapping[str, Any]],
        result_scope: str) -> Mapping[tuple[Any, ...], Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        if "result_scope" in row and row["result_scope"] != result_scope:
            raise ValueError(
                f"comparison row scope {row['result_scope']!r} does not "
                f"match {result_scope!r}")
        key = result_join_key(row, result_scope)
        if key in indexed:
            raise ValueError(f"duplicate comparison row identity: {key!r}")
        indexed[key] = row
    return indexed


def require_same_result_identity(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        fields_: Sequence[str],
        key: tuple[Any, ...]) -> None:
    for field in fields_:
        before_value = before[field]
        after_value = after[field]
        if str(before_value) != str(after_value):
            raise ValueError(
                f"comparison identity {key!r} disagrees on {field}: "
                f"{before_value!r} != {after_value!r}")


def comparison_model_identity(
        row: Mapping[str, Any], result_scope: str) -> str:
    if result_scope == "real_kernel":
        fields_ = (
            "kernel_kind", "variant", "launch_algorithm",
            "launch_grid_size", "launch_block_size",
            "launch_execution_variant", "launch_tile_shape",
            "launch_num_kv_splits",
        )
    elif result_scope == "aggregate_task":
        fields_ = ("kernel_count", "task_execution_signature")
    else:
        raise ValueError(f"unknown result scope: {result_scope!r}")
    return json.dumps(
        {field: str(row[field]) for field in fields_},
        separators=(",", ":"),
        sort_keys=True,
    )


def positive_csv_float(
        row: Mapping[str, Any], field: str, key: tuple[Any, ...]) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"comparison identity {key!r} has invalid {field}: {value!r}")
    return value


def physical_bucket_definitions() -> Mapping[str, Any]:
    return {
        "schema_version": BUCKET_SCHEMA_VERSION,
        "classification_boundary": (
            "workload and observed discrete launch facts only; prediction, "
            "duration, counters, and errors are forbidden inputs"),
        "result_scopes": {
            "real_kernel": (
                "one profiler kernel invocation with its own duration"),
            "aggregate_task": (
                "one canonical dataset task; may contain multiple kernels"),
        },
        "task_execution_signature": (
            "ordered full static kernel launch classes; duplicates are "
            f"preserved; task aggregation is {TASK_AGGREGATION_CONTRACT}"),
        "task_composition_context": (
            "task family is report context, not a bucket dimension; "
            "FlashAttention includes layout at both result scopes"),
        "kernel_launch_class": {
            "dimensions": [
                "hardware_id", "kernel_kind", "variant", "dtype",
                "execution_variant", "algorithm", "grid_rank", "block",
                "cluster", "source_program_type", "tiling",
                "discrete_controls",
            ],
            "excluded_shape_dependent_fields": [
                "grid extents", "grid_size", "padded_rows", "operation counts",
                "traffic byte counts", "worker operation counts",
            ],
            "excluded_observation_identity_fields": [
                "runtime_symbol",
            ],
            "discrete_controls": list(PHYSICAL_BUCKET_DISCRETE_CONTROLS),
        },
        "family_semantics": {
            "GEMM": [],
            "FlashAttention": ["attention_layout"],
            "FusedMoE": [],
            "Vector": [],
        },
        "small_sample_policy": (
            "never merge buckets using observed errors; report row_count and "
            "retain singleton buckets as evidence_low_n"),
        "diagnostic_only": {
            "derived_launch_rounds": (
                "ceil(grid_size/execution_units); not a profiler-observed "
                "wave counter and never a primary bucket dimension"),
        },
    }


def component_ratio_summary(
        rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    positive = [
        row for row in rows
        if row["predicted"] > 0 and row["observed"] > 0
    ]
    zero_mismatch = sum(
        (row["predicted"] == 0) != (row["observed"] == 0)
        for row in rows)
    if positive:
        ratios = np.asarray([
            row["predicted"] / row["observed"] for row in positive],
            dtype=float,
        )
        absolute = np.abs(ratios - 1.0) * 100.0
        median_ratio = float(np.median(ratios))
        p10_ratio = float(np.percentile(ratios, 10))
        p90_ratio = float(np.percentile(ratios, 90))
        mape_pct = float(np.mean(absolute))
    else:
        median_ratio = None
        p10_ratio = None
        p90_ratio = None
        mape_pct = None
    return {
        "positive_pair_count": len(positive),
        "zero_mismatch_count": zero_mismatch,
        "median_predicted_over_observed": median_ratio,
        "p10_predicted_over_observed": p10_ratio,
        "p90_predicted_over_observed": p90_ratio,
        "mape_pct": mape_pct,
    }
