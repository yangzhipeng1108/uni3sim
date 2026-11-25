"""Production Trace sweep report entry points: point validation,
aggregate report rendering, and sweep plot writing."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path

from evaluation.serving.metrics.scheduler_progress import (
    validate_scheduler_progress_manifest,
)
from evaluation.serving.simulation.execution_artifact import performance_timing_status
from src.serving.backend.vllm import normalize_mtp_scenario_manifest

from evaluation.json_shape import require_object as _mapping

from evaluation.serving.metrics.production_trace_report_shape import (
    _SWEEP_FIELDS,
    _AGGREGATE_SWEEP_FIELDS,
    _AGGREGATE_RESULT_FIELDS,
    _AGGREGATE_RESULT_TIMING_FIELDS,
    _POINT_FIELDS,
    _SUMMARY_METRIC_FIELDS,
    _REPLAY_FIELDS,
    _EXPECTED_FIXED_STATUS,
    _ARRIVAL_TRANSFORM,
    _sequence,
    _exact_fields,
    _required_fields,
    _sha256,
    _positive_number,
    _nonnegative_number,
    _nonnegative_integer,
    _canonical_json_bytes,
    _code,
)
from evaluation.serving.metrics.production_trace_execution_validation import (
    _validate_execution_and_host_timing,
    _validate_source_trace,
    _validate_execution_provenance,
    _validate_execution_artifact,
)
from evaluation.serving.metrics.production_trace_result_validation import (
    _validate_latency_distributions,
    _validate_slo_goodput,
    _validate_system_resources,
    _validate_host_kv_resource_binding,
    _validate_critical_path,
    _validate_sweep_analysis,
)
from evaluation.serving.metrics.production_trace_report_rendering import (
    _render_aggregate_point,
    _render_timing_scope,
    _render_mtp_scenario,
    _render_execution_provenance,
    _render_sweep_analysis,
    _render_point,
)


def render_production_trace_sweep_report(
    payload: Mapping[str, object],
) -> str:
    """Validate a production-Trace sweep v2 and render it as Markdown.

    This is a read-only report consumer.  It does not recompute predictions or
    metric values, and it preserves the validated timing-support status.
    """

    sweep = _mapping(payload, "production Trace sweep")
    if ("schema_version" in sweep
            and sweep["schema_version"] == 3):
        return _render_aggregate_production_trace_sweep_report(sweep)
    _exact_fields(sweep, _SWEEP_FIELDS, "production Trace sweep")
    if sweep["schema_version"] != 2:
        raise ValueError("production Trace sweep schema_version must be 2")
    if sweep["kind"] != "production_trace_arrival_rate_sweep":
        raise ValueError(
            "production Trace report requires kind "
            "'production_trace_arrival_rate_sweep'")
    if sweep["interpretation"] != "successful_path_performance_only":
        raise ValueError(
            "production Trace sweep interpretation must be "
            "'successful_path_performance_only'")
    if sweep["scale_order"] != "caller_supplied":
        raise ValueError("production Trace sweep scale_order must be caller_supplied")
    if sweep["arrival_transform"] != _ARRIVAL_TRANSFORM:
        raise ValueError("production Trace sweep arrival transform is unsupported")

    source_trace = _validate_source_trace(sweep["source_trace"])
    execution_provenance = _validate_execution_provenance(
        sweep["execution_provenance"])
    points = _sequence(sweep["points"], "production Trace sweep points")
    if not points:
        raise ValueError("production Trace sweep must contain at least one point")
    point_count = _nonnegative_integer(
        sweep["point_count"], "production Trace sweep point_count")
    if point_count != len(points):
        raise ValueError("production Trace sweep point_count disagrees with points")
    scales = tuple(
        _positive_number(scale, "production Trace arrival-rate scale")
        for scale in _sequence(
            sweep["arrival_rate_scales"],
            "production Trace arrival-rate scales",
        )
    )
    if len(scales) != len(points):
        raise ValueError(
            "production Trace arrival-rate scales disagree with point_count")
    if len(scales) != len(set(scales)):
        raise ValueError("production Trace arrival-rate scales must be unique")

    validated_points = tuple(
        _validate_point(point, index, scales[index], source_trace,
                        execution_provenance)
        for index, point in enumerate(points)
    )
    timing_status = execution_provenance["performance_timing_status"]
    if any(
            point["result"]["performance_timing_status"] != timing_status
            for point in validated_points):
        raise ValueError(
            "production Trace sweep points disagree on performance timing "
            "status")
    mtp_scenario = validated_points[0]["result"]["mtp_scenario"]
    if any(
            point["result"]["mtp_scenario"] != mtp_scenario
            for point in validated_points[1:]):
        raise ValueError(
            "production Trace sweep points disagree on the MTP scenario")
    _validate_sweep_analysis(sweep, validated_points)

    lines = [
        "# Production Trace Serving Sweep Report",
        "",
        "## Scope and validation status",
        "",
        *_render_timing_scope(timing_status),
        "- Scheduler control plane: `native_pinned_vllm`.",
        "- Trace replay: `validated_conservation`.",
        *_render_mtp_scenario(mtp_scenario),
        "- Interpretation: successful-path performance only; failed source "
        "requests are not replayed.",
        "- This report does not claim device-runtime performance validation.",
        "",
        "## Reproducibility identity",
        "",
        f"- Source path: {_code(source_trace['source_path'])}",
        f"- Source SHA256: `{source_trace['source_sha256']}`",
        "- Normalized source request-stream SHA256: "
        f"`{source_trace['normalized_request_stream_sha256']}`",
    ]
    lines.extend(_render_execution_provenance(execution_provenance))
    lines.extend(_render_sweep_analysis(sweep))
    for index, point in enumerate(validated_points, start=1):
        lines.extend(_render_point(index, point))
    return "\n".join(lines) + "\n"


def write_production_trace_sweep_plots(
    output_dir: str | Path,
    payload: Mapping[str, object],
) -> Mapping[str, Path]:
    """Write maintained throughput, latency, and simulation-speed plots."""

    render_production_trace_sweep_report(payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    sweep = _mapping(payload, "production Trace sweep")
    points = tuple(
        _mapping(point, f"production Trace plot point {index}")
        for index, point in enumerate(
            _sequence(sweep["points"], "production Trace plot points"))
    )
    scales = tuple(
        _positive_number(
            point["arrival_rate_scale"],
            "production Trace plot arrival-rate scale",
        )
        for point in points
    )
    metrics = tuple(
        _mapping(point["metrics"], "production Trace plot metrics")
        for point in points
    )
    results = tuple(
        _mapping(point["result"], "production Trace plot result")
        for point in points
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Production Trace plot generation requires matplotlib") from error

    paths = {
        "throughput": destination / "production_trace_throughput.png",
        "ttft_p0_p100": destination / "production_trace_ttft_p0_p100.png",
        "tpot_p0_p100": destination / "production_trace_tpot_p0_p100.png",
        "simulation_speed": destination / "production_trace_simulation_speed.png",
        "interactivity_tgs": destination / "production_trace_interactivity_tgs.png",
    }
    positions = tuple(range(len(points)))
    labels = tuple(f"{scale:g}×" for scale in scales)

    figure, axis = plt.subplots(figsize=(max(8.0, len(points) * 1.2), 5.5))
    width = 0.36
    request_values = tuple(
        _nonnegative_number(
            row["request_throughput_per_s"], "request throughput")
        for row in metrics
    )
    token_values = tuple(
        _nonnegative_number(
            row["output_token_throughput_per_s"], "output-token throughput")
        for row in metrics
    )
    axis.bar(
        [position - width / 2 for position in positions],
        request_values,
        width=width,
        label="Requests/s",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        token_values,
        width=width,
        label="Output tokens/s",
    )
    axis.set_title("Production Trace throughput by arrival-rate scale")
    axis.set_xlabel("Arrival-rate scale")
    axis.set_ylabel("Throughput per second")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["throughput"], dpi=160)
    plt.close(figure)

    for metric_name, path_key, title in (
        ("ttft_s", "ttft_p0_p100", "TTFT P0–P100"),
        ("tpot_s", "tpot_p0_p100", "TPOT P0–P100"),
    ):
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        for scale, row in zip(scales, metrics, strict=True):
            distributions = _mapping(
                row["latency_distributions"], "latency distributions")
            distribution = _mapping(
                distributions[metric_name], f"{metric_name} distribution")
            if distribution["availability"] != "available":
                continue
            quantiles = _mapping(
                distribution["quantiles"], f"{metric_name} quantiles")
            axis.plot(
                _sequence(quantiles["percentiles"], "latency percentiles"),
                _sequence(quantiles["values_s"], "latency quantile values"),
                label=f"{scale:g}× arrival rate",
                linewidth=1.8,
            )
        axis.set_title(title)
        axis.set_xlabel("Percentile")
        axis.set_ylabel("Seconds")
        axis.set_xlim(0, 100)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(paths[path_key], dpi=160)
        plt.close(figure)

    simulation_wall = tuple(
        _nonnegative_number(
            result["simulation_wall_time_s"], "simulation wall time")
        for result in results
    )
    virtual_time = tuple(
        _nonnegative_number(
            result["schedule_end_time_s"], "virtual schedule end time")
        for result in results
    )
    figure, axis = plt.subplots(figsize=(max(8.0, len(points) * 1.2), 5.5))
    bars = axis.bar(positions, simulation_wall, color="#4C78A8")
    axis.bar_label(
        bars,
        labels=[f"{value:.2f}s" for value in simulation_wall],
        padding=3,
    )
    for position, wall, virtual in zip(
        positions, simulation_wall, virtual_time, strict=True
    ):
        ratio = math.inf if virtual == 0.0 else wall / virtual
        label = "n/a" if not math.isfinite(ratio) else f"{ratio:.2f} wall/virtual"
        axis.annotate(
            label,
            (position, wall),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axis.set_title("Production Trace simulation wall time")
    axis.set_xlabel("Arrival-rate scale")
    axis.set_ylabel("Wall seconds")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths["simulation_speed"], dpi=160)
    plt.close(figure)

    # The trade-off contract is persisted in the sweep JSON, so this plot
    # consumes only the validated P99 TPOT/interactivity and TGS values.
    tradeoff = _mapping(sweep["interactivity_tgs"], "Interactivity-TGS analysis")
    if tradeoff["availability"] != "unavailable_no_metric_contract":
        points_tradeoff = tuple(
            _mapping(point, "Interactivity-TGS plot point")
            for point in _sequence(tradeoff["points"], "Interactivity-TGS plot points")
        )
        frontier_tradeoff = tuple(
            _mapping(point, "Interactivity-TGS frontier point")
            for point in _sequence(tradeoff["pareto_frontier"], "Interactivity-TGS frontier")
        )
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        x_values = tuple(_nonnegative_number(point["interactivity_value_s"], "interactivity") for point in points_tradeoff)
        y_values = tuple(_nonnegative_number(point["tgs_value"], "TGS") for point in points_tradeoff)
        labels_tradeoff = tuple(_positive_number(point["arrival_rate_scale"], "arrival-rate scale") for point in points_tradeoff)
        axis.scatter(x_values, y_values, s=55, label="Sweep points")
        for x_value, y_value, label in zip(x_values, y_values, labels_tradeoff, strict=True):
            axis.annotate(f"{label:g}×", (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
        if frontier_tradeoff:
            frontier_xy = tuple(sorted(
                (_nonnegative_number(point["interactivity_value_s"], "frontier interactivity"), _nonnegative_number(point["tgs_value"], "frontier TGS"))
                for point in frontier_tradeoff
            ))
            axis.plot(tuple(item[0] for item in frontier_xy), tuple(item[1] for item in frontier_xy), color="#E45756", marker="o", linewidth=2, label="Pareto frontier")
        axis.set_title("Interactivity (P99 TPOT) vs TGS")
        axis.set_xlabel("P99 TPOT (seconds)")
        axis.set_ylabel("Output-token TGS (tokens/s)")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(paths["interactivity_tgs"], dpi=160)
        plt.close(figure)
    return paths


def _render_aggregate_production_trace_sweep_report(
    sweep: Mapping[str, object],
) -> str:
    """Validate and render the request-safe persisted sweep schema."""

    _exact_fields(
        sweep, _AGGREGATE_SWEEP_FIELDS, "aggregate production Trace sweep")
    if sweep["kind"] != "production_trace_arrival_rate_sweep":
        raise ValueError("aggregate production Trace sweep kind is invalid")
    persistence = _mapping(
        sweep["persistence"], "aggregate persistence contract")
    _exact_fields(
        persistence,
        {"schema_version", "projection", "request_level_content"},
        "aggregate persistence contract",
    )
    if persistence != {
            "schema_version": 1,
            "projection": "aggregate_only_production_trace_v1",
            "request_level_content": "omitted",
    }:
        raise ValueError("aggregate persistence contract is not canonical")
    if sweep["interpretation"] != "successful_path_performance_only":
        raise ValueError("aggregate production Trace interpretation is invalid")
    if sweep["scale_order"] != "caller_supplied":
        raise ValueError("aggregate production Trace scale order is invalid")
    if sweep["arrival_transform"] != _ARRIVAL_TRANSFORM:
        raise ValueError("aggregate production Trace arrival transform is invalid")
    source_trace = _validate_source_trace(sweep["source_trace"])
    execution_provenance = _validate_execution_provenance(
        sweep["execution_provenance"])
    points = _sequence(sweep["points"], "aggregate production Trace points")
    scales = tuple(
        _positive_number(scale, "aggregate production Trace scale")
        for scale in _sequence(
            sweep["arrival_rate_scales"],
            "aggregate production Trace scales",
        )
    )
    if (not points
            or len(points) != len(scales)
            or sweep["point_count"] != len(points)
            or len(scales) != len(set(scales))):
        raise ValueError("aggregate production Trace point cardinality is invalid")
    validated_points = tuple(
        _validate_aggregate_point(
            point,
            index,
            scales[index],
            source_trace,
            execution_provenance,
        )
        for index, point in enumerate(points)
    )
    _validate_sweep_analysis(sweep, validated_points)
    timing_status = execution_provenance["performance_timing_status"]
    mtp_scenario = validated_points[0]["result"]["mtp_scenario"]
    if any(
            point["result"]["mtp_scenario"] != mtp_scenario
            for point in validated_points[1:]):
        raise ValueError(
            "aggregate production Trace points disagree on MTP scenario")
    lines = [
        "# Production Trace Serving Sweep Report",
        "",
        "## Scope and validation status",
        "",
        *_render_timing_scope(timing_status),
        "- Persistence: **aggregate-only**; request-level arrivals, metrics, "
        "routes, transfers, Scheduler steps, and MTP outcomes are omitted.",
        "- Scheduler control plane: `native_pinned_vllm`.",
        "- Trace replay: `validated_conservation`.",
        *_render_mtp_scenario(mtp_scenario),
        "- Interpretation: successful-path performance only.",
        "",
        "## Reproducibility identity",
        "",
        f"- Source SHA256: `{source_trace['source_sha256']}`",
        "- Normalized source request-stream SHA256: "
        f"`{source_trace['normalized_request_stream_sha256']}`",
    ]
    lines.extend(_render_execution_provenance(execution_provenance))
    lines.extend(_render_sweep_analysis(sweep))
    for index, point in enumerate(validated_points, start=1):
        lines.extend(_render_aggregate_point(index, point))
    return "\n".join(lines) + "\n"


def _validate_aggregate_point(
    value: object,
    index: int,
    expected_scale: float,
    source_trace: Mapping[str, object],
    execution_provenance: Mapping[str, object],
) -> Mapping[str, object]:
    label = f"aggregate production Trace point {index}"
    point = _mapping(value, label)
    _exact_fields(point, _POINT_FIELDS, label)
    scale = _positive_number(point["arrival_rate_scale"], f"{label} scale")
    if scale != expected_scale:
        raise ValueError(f"{label} scale disagrees with sweep scales")
    stream_sha256 = _sha256(
        point["request_stream_sha256"], f"{label} request stream SHA256")
    status = _mapping(point["validation_status"], f"{label} status")
    _exact_fields(
        status,
        set(_EXPECTED_FIXED_STATUS) | {"performance_timing"},
        f"{label} status",
    )
    if any(status[name] != expected for name, expected in (
            _EXPECTED_FIXED_STATUS.items())):
        raise ValueError(f"{label} fixed validation status is invalid")
    summary_metrics = _mapping(point["metrics"], f"{label} summary metrics")
    _exact_fields(
        summary_metrics, _SUMMARY_METRIC_FIELDS, f"{label} summary metrics")
    result = _mapping(point["result"], f"{label} result")
    _exact_fields(result, _AGGREGATE_RESULT_FIELDS | _AGGREGATE_RESULT_TIMING_FIELDS,
        f"{label} result")
    if (result["schema_version"] != 1
            or result["kind"] != "production_trace_simulation_aggregate"
            or result["version"] != 3
            or result["backend"] != "vllm"):
        raise ValueError(f"{label} aggregate result identity is invalid")
    result_persistence = _mapping(
        result["persistence"], f"{label} result persistence")
    _exact_fields(
        result_persistence,
        {"projection", "request_level_content", "full_in_memory_result_sha256"},
        f"{label} result persistence",
    )
    if (result_persistence["projection"]
            != "aggregate_only_production_trace_v1"
            or result_persistence["request_level_content"] != "omitted"):
        raise ValueError(f"{label} aggregate projection is invalid")
    _sha256(
        result_persistence["full_in_memory_result_sha256"],
        f"{label} in-memory result SHA256",
    )
    expected_timing = performance_timing_status(
        result["launch_policy"],
        tuple(_sequence(result["hardware_ids"], f"{label} hardware IDs")),
    )
    if (result["performance_timing_status"] != expected_timing
            or status["performance_timing"] != expected_timing
            or execution_provenance["performance_timing_status"]
            != expected_timing
            or result["execution_provenance"] != execution_provenance):
        raise ValueError(f"{label} timing or execution provenance is invalid")
    _validate_execution_and_host_timing(result, label)
    request_source = _mapping(
        result["request_source"], f"{label} request source")
    _exact_fields(
        request_source,
        {"schema_version", "kind", "trace", "replay",
         "zero_output_contract", "prefix_alignment"},
        f"{label} request source",
    )
    if (request_source["kind"] != "production_trace"
            or request_source["trace"] != source_trace):
        raise ValueError(f"{label} source Trace identity is invalid")
    replay = _mapping(request_source["replay"], f"{label} replay")
    _exact_fields(replay, _REPLAY_FIELDS, f"{label} replay")
    if (replay["request_stream_sha256"] != stream_sha256
            or replay["arrival_rate_scale"] != scale
            or replay["kind"] != "arrival_rate_scale"):
        raise ValueError(f"{label} replay identity is invalid")
    alignment = _mapping(
        request_source["prefix_alignment"], f"{label} prefix alignment")
    _exact_fields(
        alignment,
        {"schema_version", "owner", "vllm_version", "rule",
         "kv_block_size_tokens", "aggregate",
         "request_level_entries_sha256", "request_level_entries"},
        f"{label} prefix alignment",
    )
    if alignment["request_level_entries"] != "omitted":
        raise ValueError(f"{label} retained request-level prefix entries")
    _sha256(
        alignment["request_level_entries_sha256"],
        f"{label} prefix-entry SHA256",
    )
    aggregate_metrics = _mapping(result["metrics"], f"{label} metrics")
    _exact_fields(
        aggregate_metrics,
        _SUMMARY_METRIC_FIELDS | {
            "system_resources_sha256", "critical_path_sha256",
            "request_level_metrics",
        },
        f"{label} metrics",
    )
    if aggregate_metrics["request_level_metrics"] != "omitted":
        raise ValueError(f"{label} retained request-level metrics")
    for name in _SUMMARY_METRIC_FIELDS:
        if aggregate_metrics[name] != summary_metrics[name]:
            raise ValueError(f"{label} summary metric {name} disagrees")
    for name in (
            "request_throughput_per_s",
            "output_token_throughput_per_s",
            "token_throughput_per_s"):
        _nonnegative_number(aggregate_metrics[name], f"{label} {name}")
    _nonnegative_number(
        result["simulation_wall_time_s"], f"{label} simulation wall time")
    _validate_latency_distributions(
        aggregate_metrics["latency_distributions"], label)
    _validate_slo_goodput(aggregate_metrics["slo_goodput"], label)
    _sha256(aggregate_metrics["system_resources_sha256"],
            f"{label} system-resource SHA256")
    _sha256(aggregate_metrics["critical_path_sha256"],
            f"{label} critical-path SHA256")
    artifact = _mapping(result["execution_artifact"], f"{label} artifact")
    _exact_fields(
        artifact,
        {"schema_version", "hash_method", "performance_timing_status",
         "counts", "artifact_sha256", "request_level_artifact"},
        f"{label} artifact",
    )
    if artifact["request_level_artifact"] != "omitted":
        raise ValueError(f"{label} retained request-level execution artifact")
    artifact_sha256 = _sha256(
        artifact["artifact_sha256"], f"{label} artifact SHA256")
    scheduler_progress = validate_scheduler_progress_manifest(
        result["scheduler_progress"])
    if scheduler_progress["summary"]["step_count"] != result["num_steps"]:
        raise ValueError(f"{label} Scheduler progress step count disagrees")
    main_chain = _mapping(result["main_chain_status"], f"{label} main chain")
    _exact_fields(
        main_chain, {"status", "chain", "execution_artifact_sha256"},
        f"{label} main chain",
    )
    if (main_chain["status"] != "executed"
            or main_chain["execution_artifact_sha256"] != artifact_sha256):
        raise ValueError(f"{label} main-chain attestation is invalid")
    binding = _mapping(
        result["workload_assumption_binding"], f"{label} binding")
    _exact_fields(
        binding,
        {"schema_version", "request_stream_sha256",
         "expert_routing_policy_sha256", "mtp_acceptance_policy_sha256",
         "composite_sha256", "independence_contract"},
        f"{label} binding",
    )
    if binding["request_stream_sha256"] != stream_sha256:
        raise ValueError(f"{label} workload binding request hash disagrees")
    components = {
        "request_stream_sha256": binding["request_stream_sha256"],
        "expert_routing_policy_sha256": (
            binding["expert_routing_policy_sha256"]),
        "mtp_acceptance_policy_sha256": (
            binding["mtp_acceptance_policy_sha256"]),
    }
    expected_composite = hashlib.sha256(
        _canonical_json_bytes(components)).hexdigest()
    if binding["composite_sha256"] != expected_composite:
        raise ValueError(f"{label} composite workload binding is invalid")
    try:
        scenario = normalize_mtp_scenario_manifest(result["mtp_scenario"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} MTP scenario is invalid: {error}") from error
    if _canonical_json_bytes(scenario) != _canonical_json_bytes(
            result["mtp_scenario"]):
        raise ValueError(f"{label} MTP scenario is not canonical")
    return point


def _validate_point(
    value: object,
    index: int,
    expected_scale: float,
    source_trace: Mapping[str, object],
    execution_provenance: Mapping[str, object],
) -> Mapping[str, object]:
    label = f"production Trace sweep point {index}"
    point = _mapping(value, label)
    _exact_fields(point, _POINT_FIELDS, label)
    scale = _positive_number(point["arrival_rate_scale"], f"{label} scale")
    if scale != expected_scale:
        raise ValueError(f"{label} scale disagrees with arrival_rate_scales")
    request_stream_sha256 = _sha256(
        point["request_stream_sha256"], f"{label} request stream SHA256")

    summary_metrics = _mapping(point["metrics"], f"{label} summary metrics")
    _exact_fields(summary_metrics, _SUMMARY_METRIC_FIELDS,
                  f"{label} summary metrics")
    result = _mapping(point["result"], f"{label} result")
    _required_fields(
        result,
        {
            "version",
            "backend",
            "execution_provenance",
            "execution_artifact",
            "request_source",
            "mtp_scenario",
            "metrics",
            "launch_policy",
            "hardware_ids",
            "performance_timing_status",
            "execution_mode",
            "serving_timing_mode",
            "host_timing",
        },
        f"{label} result",
    )
    if result["version"] != 3:
        raise ValueError(f"{label} result version must be 3")
    if result["backend"] != "vllm":
        raise ValueError(f"{label} result backend must be vllm")
    expected_timing_status = performance_timing_status(
        result["launch_policy"],
        tuple(_sequence(result["hardware_ids"], f"{label} hardware IDs")),
    )
    if result["performance_timing_status"] != expected_timing_status:
        raise ValueError(
            f"{label} performance timing status disagrees with launch policy "
            "and hardware IDs")
    status = _mapping(point["validation_status"], f"{label} validation status")
    _exact_fields(
        status,
        set(_EXPECTED_FIXED_STATUS) | {"performance_timing"},
        f"{label} validation status",
    )
    for field, expected in _EXPECTED_FIXED_STATUS.items():
        if status[field] != expected:
            raise ValueError(
                f"{label} validation status has invalid {field}")
    if status["performance_timing"] != expected_timing_status:
        raise ValueError(
            f"{label} validation status disagrees with result timing status")
    if result["execution_provenance"] != execution_provenance:
        raise ValueError(f"{label} execution provenance disagrees with sweep")
    if execution_provenance["performance_timing_status"] != (
            expected_timing_status):
        raise ValueError(
            f"{label} execution provenance timing status disagrees with "
            "result")
    _validate_execution_and_host_timing(result, label)
    try:
        mtp_scenario = normalize_mtp_scenario_manifest(
            result["mtp_scenario"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} MTP scenario is invalid: {error}") from error
    if _canonical_json_bytes(mtp_scenario) != _canonical_json_bytes(
            result["mtp_scenario"]):
        raise ValueError(f"{label} MTP scenario is not canonical")

    request_source = _mapping(result["request_source"], f"{label} request source")
    _required_fields(
        request_source, {"kind", "trace", "replay"}, f"{label} request source")
    if request_source["kind"] != "production_trace":
        raise ValueError(f"{label} request source must be production_trace")
    if request_source["trace"] != source_trace:
        raise ValueError(f"{label} source Trace provenance disagrees with sweep")
    replay = _mapping(request_source["replay"], f"{label} replay")
    _exact_fields(replay, _REPLAY_FIELDS, f"{label} replay")
    if replay["schema_version"] != 1:
        raise ValueError(f"{label} replay schema_version must be 1")
    if replay["kind"] != "arrival_rate_scale":
        raise ValueError(f"{label} replay kind must be arrival_rate_scale")
    if replay["arrival_transform"] != _ARRIVAL_TRANSFORM:
        raise ValueError(f"{label} replay arrival transform is unsupported")
    if replay["interpretation"] != "successful_path_performance_only":
        raise ValueError(f"{label} replay interpretation is unsupported")
    if _positive_number(replay["arrival_rate_scale"], f"{label} replay scale") != scale:
        raise ValueError(f"{label} replay scale disagrees with point")
    replay_sha256 = _sha256(
        replay["request_stream_sha256"], f"{label} replay request stream SHA256")
    if replay_sha256 != request_stream_sha256:
        raise ValueError(f"{label} request stream SHA256 disagrees with replay")

    result_metrics = _mapping(result["metrics"], f"{label} result metrics")
    _required_fields(
        result_metrics,
        _SUMMARY_METRIC_FIELDS | {"system_resources", "critical_path"},
        f"{label} result metrics",
    )
    for metric_name in _SUMMARY_METRIC_FIELDS:
        if summary_metrics[metric_name] != result_metrics[metric_name]:
            raise ValueError(
                f"{label} summary metrics disagree with nested result: "
                f"{metric_name}")
    for metric_name in (
            "request_throughput_per_s",
            "output_token_throughput_per_s",
            "token_throughput_per_s"):
        _nonnegative_number(result_metrics[metric_name], f"{label} {metric_name}")
    _validate_latency_distributions(result_metrics["latency_distributions"], label)
    _validate_slo_goodput(result_metrics["slo_goodput"], label)
    _validate_system_resources(result_metrics["system_resources"], label)
    _validate_critical_path(result_metrics["critical_path"], label)
    _validate_execution_artifact(
        result["execution_artifact"], label, expected_timing_status)
    _validate_host_kv_resource_binding(
        result["execution_artifact"],
        result_metrics["system_resources"],
        label,
    )
    return point
