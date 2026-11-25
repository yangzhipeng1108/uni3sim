"""Markdown rendering for validated Production Trace sweep points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from evaluation.serving.simulation.execution_artifact import (
    PROJECTED_PERFORMANCE_TIMING_STATUS,
    UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
)

from evaluation.json_shape import require_object as _mapping

from evaluation.serving.metrics.production_trace_report_shape import (
    _sequence,
    _required_fields,
    _sha256,
    _format,
    _inline_json,
    _code,
    _markdown_cell,
    _table_code,
)


def _render_aggregate_point(
    index: int,
    point: Mapping[str, object],
) -> list[str]:
    result = _mapping(point["result"], "aggregate point result")
    metrics = _mapping(result["metrics"], "aggregate point metrics")
    artifact = _mapping(result["execution_artifact"], "aggregate artifact")
    lines = [
        f"## Sweep point {index}: arrival-rate scale "
        f"{_format(point['arrival_rate_scale'])}",
        "",
        f"- Request-stream SHA256: `{point['request_stream_sha256']}`",
        f"- Execution artifact SHA256: `{artifact['artifact_sha256']}`",
        f"- Scheduler steps: `{result['num_steps']}`; kernel calls: "
        f"`{result['num_kernel_calls']}`; transfers: "
        f"`{result['num_transfers']}`.",
        *_render_timing_scope(result["performance_timing_status"]),
        "",
        "### Throughput",
        "",
        "| Metric | Value per second |",
        "|---|---:|",
        f"| Requests | {_format(metrics['request_throughput_per_s'])} |",
        "| Output tokens | "
        f"{_format(metrics['output_token_throughput_per_s'])} |",
        "| Input + output tokens | "
        f"{_format(metrics['token_throughput_per_s'])} |",
        "",
    ]
    lines.extend(_render_latency_distributions(metrics["latency_distributions"]))
    lines.extend(_render_slo(metrics["slo_goodput"]))
    return lines


def _render_timing_scope(timing_status: object) -> list[str]:
    if timing_status == UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS:
        return [
            "> [!WARNING]",
            "> **UNSUPPORTED / DEVELOPMENT ONLY:** this run used A800 "
            "analytical launch. Do not use its timings for publication "
            "performance claims, accuracy validation, hardware comparison, "
            "or DSE conclusions.",
            "",
            "- Timing: **Unsupported / development-only** "
            f"(`{UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS}`).",
        ]
    if timing_status == PROJECTED_PERFORMANCE_TIMING_STATUS:
        return [
            "- Timing: **Projected** "
            f"(`{PROJECTED_PERFORMANCE_TIMING_STATUS}`).",
        ]
    raise AssertionError("validated performance timing status is unknown")


def _render_mtp_scenario(scenario: Mapping[str, object]) -> list[str]:
    policy = scenario["acceptance_policy"]
    if policy == "disabled":
        return ["- MTP acceptance: disabled."]
    if policy == "fixed_count":
        return [
            "- MTP acceptance: **Projected explicit assumption** "
            f"(`fixed_count`, accepted "
            f"`{scenario['fixed_accepted_tokens']}`/"
            f"`{scenario['num_speculative_tokens']}`, source "
            f"{_code(scenario['acceptance_source_id'])})."
        ]
    if policy == "maximum_entropy":
        assumption = _mapping(
            scenario["acceptance_assumption"],
            "MTP maximum-entropy acceptance assumption",
        )
        return [
            "- MTP acceptance: **Projected explicit assumption** "
            "(`maximum_entropy`, finite support "
            f"`[0,{assumption['proposal_depth']}]`, mean rate "
            f"`{assumption['mean_acceptance_rate']}`, seed "
            f"`{assumption['root_seed']}`, source "
            f"{_code(assumption['source_id'])}).",
            "- MTP assumption policy SHA256: "
            f"`{assumption['policy_descriptor_sha256']}`; duration "
            "observations: none.",
        ]
    if policy == "replay":
        replay = _mapping(
            scenario["acceptance_replay"], "MTP acceptance replay provenance")
        lines = [
            "- MTP acceptance: **Replayed external observation** "
            f"(`replay`, source {_code(scenario['acceptance_source_id'])}).",
            "- MTP replay artifact: "
            f"schema `{replay['schema_version']}`, config "
            f"{_code(replay['config_path'])}, SHA256 "
            f"`{replay['config_sha256']}`.",
            "- MTP replay producer: "
            f"{_code(replay['producer'])}; vLLM "
            f"{_code(replay['vllm_version'])} at "
            f"`{replay['vllm_source_commit']}`; package SHA256 "
            f"`{replay['vllm_package_manifest_sha256']}`.",
            "- MTP replay model config SHA256: "
            f"`{replay['model_config_sha256']}`.",
            "- MTP replay scope: pool "
            f"{_code(replay['pool_name'])}, role "
            f"{_code(replay['backend_role'])}, speculative tokens "
            f"`{replay['num_speculative_tokens']}`, Scheduler steps "
            f"`{replay['step_count']}`.",
        ]
        if replay["schema_version"] == 1:
            lines.append(
                "- MTP replay request-stream SHA256: "
                f"`{replay['request_stream_sha256']}`.")
        elif replay["schema_version"] == 2:
            lines.extend((
                "- MTP replay adapter: artifact schema "
                f"`{replay['artifact_schema_version']}`, contract "
                f"{_code(replay['adapter_contract'])}.",
                "- MTP replay ingress request-stream SHA256: "
                f"`{replay['ingress_request_stream_sha256']}`.",
                "- MTP replay decode-admission SHA256: "
                f"`{replay['decode_admission_contract_sha256']}`.",
            ))
        else:
            raise AssertionError(
                "validated MTP replay has an unknown schema version")
        return lines
    raise AssertionError("validated MTP scenario has an unknown policy")


def _render_execution_provenance(provenance: Mapping[str, object]) -> list[str]:
    repository = _mapping(provenance["repository"], "repository provenance")
    hardware_ids = _sequence(provenance["loaded_hardware_ids"], "loaded hardware IDs")
    configs = _mapping(provenance["hardware_configs"], "hardware configs")
    lines = [
        f"- Git commit observed at run end: `{repository['git_commit']}`",
        "- Exact imported-code snapshot binding: unbound; this is not a frozen execution snapshot.",
        f"- Worktree state: {_code(repository['worktree_state'])}",
        f"- Worktree diff hash: `{repository['git_diff_hash']}`",
        "- Loaded hardware IDs: "
        + ", ".join(_code(value) for value in hardware_ids),
        f"- Prediction method: {_code(provenance['prediction_method'])}",
        "- Hardware config identities:",
    ]
    if provenance["schema_version"] == 4:
        lines[2:2] = [
            f"- Git commit observed at run start: `{repository['run_start_identity']['git_commit']}`",
            f"- Repository identity changed between sampled boundaries: {repository['identity_changed_between_boundaries']}",
            "- Observation scope: entry and exit only; transient changes between observations are not excluded.",
        ]
    for role in sorted(configs):
        config = configs[role]
        if config is None:
            lines.append(f"  - {_code(role)}: not applicable")
        elif isinstance(config, Mapping):
            if "sha256" not in config or "path" not in config:
                raise ValueError(
                    f"hardware config provenance {role!r} lacks path or SHA256")
            _sha256(config["sha256"], f"hardware config {role} SHA256")
            lines.append(
                f"  - {_code(role)}: "
                f"{_code(config['path'])}; SHA256 "
                f"`{config['sha256']}`")
        elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
            if not config:
                lines.append(f"  - {_code(role)}: none")
            for index, item in enumerate(config):
                additional = _mapping(
                    item, f"additional hardware config {index}")
                _required_fields(
                    additional, {"path", "sha256"},
                    f"additional hardware config {index}")
                _sha256(
                    additional["sha256"],
                    f"additional hardware config {index} SHA256",
                )
                lines.append(
                    f"  - {_code(f'{role}[{index}]')}: "
                    f"{_code(additional['path'])}; SHA256 "
                    f"`{additional['sha256']}`")
        else:
            raise ValueError(f"hardware config provenance {role!r} is invalid")
    launch_inputs = _mapping(provenance["launch_inputs"], "launch inputs")
    launch_data = _mapping(launch_inputs["launch_data"], "launch data")
    lines.extend([
        f"- Launch policy: {_code(launch_inputs['launch_policy'])}",
        "- Launch-data identities:",
    ])
    for hardware_id in sorted(launch_data):
        launch = _mapping(launch_data[hardware_id], "launch-data identity")
        _required_fields(
            launch, {"path", "tree_sha256", "file_count"},
            f"launch data {hardware_id}")
        _sha256(launch["tree_sha256"], f"launch data {hardware_id} tree SHA256")
        lines.append(
            f"  - {_code(hardware_id)}: "
            f"{_code(launch['path'])}; tree SHA256 "
            f"`{launch['tree_sha256']}`; files `{launch['file_count']}`")
    lines.extend([
        "- Runtime inputs: " + _inline_json(provenance["runtime_inputs"]),
        "- Microbenchmark inputs: "
        + _inline_json(provenance["microbenchmark_inputs"]),
        "- External predictor inputs: "
        + _inline_json(provenance["external_predictor_inputs"]),
        "",
    ])
    return lines


def _render_sweep_analysis(sweep: Mapping[str, object]) -> list[str]:
    tradeoff = _mapping(sweep["interactivity_tgs"], "Interactivity-TGS analysis")
    peak = _mapping(sweep["peak_selection"], "peak selection")
    saturation = _mapping(sweep["saturation_analysis"], "saturation analysis")
    lines = ["## Interactivity–TGS analysis", ""]
    if tradeoff["availability"] == "unavailable_no_metric_contract":
        lines.extend([
            "Unavailable: no metric contract was supplied; no "
            "Interactivity–TGS metric default applied.",
            "",
            "Peak and saturation are also unavailable because a sampled "
            "trade-off metric was not defined.",
            "",
        ])
        return lines

    contract = _mapping(tradeoff["contract"], "Interactivity-TGS contract")
    lines.extend([
        f"- Availability: `{tradeoff['availability']}`",
        f"- Contract: {_code(contract['contract_id'])}",
        f"- Contract source SHA256: `{contract['source_sha256']}`",
        "- Definition: "
        f"P{contract['interactivity_percentile']} "
        f"{_code(contract['interactivity_metric'])} versus "
        f"{_code(contract['tgs_metric'])}.",
        "",
        "| Scale | Interactivity (s) | TGS | Request throughput/s |",
        "|---:|---:|---:|---:|",
    ])
    for point in _sequence(tradeoff["points"], "Interactivity-TGS points"):
        row = _mapping(point, "Interactivity-TGS point")
        _required_fields(
            row,
            {"arrival_rate_scale", "interactivity_value_s", "tgs_value",
             "request_throughput_per_s"},
            "Interactivity-TGS point",
        )
        lines.append(
            f"| {_format(row['arrival_rate_scale'])} "
            f"| {_format(row['interactivity_value_s'])} "
            f"| {_format(row['tgs_value'])} "
            f"| {_format(row['request_throughput_per_s'])} |")
    lines.extend(["", "Pareto frontier:", ""])
    frontier = _sequence(tradeoff["pareto_frontier"], "Interactivity-TGS frontier")
    if frontier:
        for point in frontier:
            row = _mapping(point, "Interactivity-TGS frontier point")
            lines.append(
                f"- scale `{_format(row['arrival_rate_scale'])}`: "
                f"interactivity `{_format(row['interactivity_value_s'])}` s, "
                f"TGS `{_format(row['tgs_value'])}`")
    else:
        lines.append("- No eligible Pareto point.")
    lines.extend(["", "### Observed sampled peak", ""])
    lines.append(f"- Availability: `{peak['availability']}`")
    if "selection_scope" in peak:
        lines.append(f"- Scope: {_code(peak['selection_scope'])}")
    if "point" in peak and peak["point"] is not None:
        peak_point = _mapping(peak["point"], "sampled peak point")
        lines.append(
            f"- Point: scale `{_format(peak_point['arrival_rate_scale'])}`, "
            f"TGS `{_format(peak_point['tgs_value'])}`.")
    lines.extend(["", "### Saturation", ""])
    lines.extend([
        f"- Availability: `{saturation['availability']}`",
        "- Scope: sampled arrival-rate scales only.",
        "",
        "| From scale | To scale | Marginal TGS gain ratio | Classification |",
        "|---:|---:|---:|---|",
    ])
    for interval in _sequence(saturation["intervals"], "saturation intervals"):
        row = _mapping(interval, "saturation interval")
        lines.append(
            f"| {_format(row['from_arrival_rate_scale'])} "
            f"| {_format(row['to_arrival_rate_scale'])} "
            f"| {_format(row['marginal_tgs_gain_ratio'])} "
            f"| {_table_code(row['classification'])} |")
    lines.append("")
    return lines


def _render_point(index: int, point: Mapping[str, object]) -> list[str]:
    result = _mapping(point["result"], "point result")
    metrics = _mapping(result["metrics"], "point result metrics")
    artifact = _mapping(result["execution_artifact"], "execution artifact")
    lines = [
        f"## Sweep point {index}: arrival-rate scale "
        f"{_format(point['arrival_rate_scale'])}",
        "",
        f"- Request-stream SHA256: `{point['request_stream_sha256']}`",
        f"- Execution artifact SHA256: `{artifact['artifact_sha256']}`",
        *_render_timing_scope(result["performance_timing_status"]),
        "",
        "### Throughput",
        "",
        "| Metric | Value per second |",
        "|---|---:|",
        "| Requests | "
        f"{_format(metrics['request_throughput_per_s'])} |",
        "| Output tokens | "
        f"{_format(metrics['output_token_throughput_per_s'])} |",
        "| Input + output tokens | "
        f"{_format(metrics['token_throughput_per_s'])} |",
        "",
    ]
    lines.extend(_render_latency_distributions(metrics["latency_distributions"]))
    lines.extend(_render_slo(metrics["slo_goodput"]))
    lines.extend(_render_resources(metrics["system_resources"]))
    lines.extend(_render_critical_path(metrics["critical_path"]))
    return lines


def _render_latency_distributions(value: object) -> list[str]:
    distributions = _mapping(value, "latency distributions")
    lines = ["### Latency CDF and P0–P100", ""]
    labels = {
        "ttft_s": "TTFT",
        "tpot_s": "TPOT",
        "latency_s": "E2E latency",
    }
    for metric_name in ("ttft_s", "tpot_s", "latency_s"):
        distribution = _mapping(distributions[metric_name], metric_name)
        lines.extend([
            f"#### {labels[metric_name]}",
            "",
            f"- Availability: `{distribution['availability']}`",
            f"- Eligible requests: {distribution['eligible_request_count']}",
            f"- Excluded requests: {distribution['excluded_request_count']} "
            f"({_code(distribution['exclusion_reason'])})",
            "",
        ])
        if distribution["availability"] != "available":
            continue
        quantiles = _mapping(distribution["quantiles"], "latency quantiles")
        percentiles = _sequence(quantiles["percentiles"], "latency percentiles")
        values = _sequence(quantiles["values_s"], "latency quantile values")
        lines.extend([
            "Quantiles (seconds):",
            "",
            ", ".join(
                f"P{percentile}={_format(metric_value)}"
                for percentile, metric_value in zip(percentiles, values)
            ),
            "",
            "Empirical CDF:",
            "",
            "| Latency (s) | Cumulative probability |",
            "|---:|---:|",
        ])
        cdf = _mapping(distribution["empirical_cdf"], "latency CDF")
        for latency, probability in zip(
                _sequence(cdf["values_s"], "latency CDF values"),
                _sequence(cdf["cumulative_probabilities"],
                          "latency CDF probabilities")):
            lines.append(f"| {_format(latency)} | {_format(probability)} |")
        lines.append("")
    return lines


def _render_slo(value: object) -> list[str]:
    slo = _mapping(value, "SLO goodput")
    lines = ["### SLO goodput", ""]
    if slo["availability"] == "unavailable_no_slo_contract":
        lines.extend([
            "Unavailable: no SLO contract was supplied; no SLO threshold "
            "default applied.",
            "",
        ])
        return lines
    contract = _mapping(slo["contract"], "SLO contract")
    lines.extend([
        f"- Contract: {_code(contract['contract_id'])}",
        f"- Contract source SHA256: `{contract['source_sha256']}`",
        f"- TTFT threshold: `{_format(contract['ttft_threshold_s'])}` s",
        f"- TPOT threshold: `{_format(contract['tpot_threshold_s'])}` s",
        f"- Request SLO goodput: `{_format(slo['request_goodput_per_s'])}` /s",
        f"- Token SLO goodput: `{_format(slo['token_goodput_per_s'])}` /s",
        "- Population: " + _inline_json(slo["population"]),
        "- Violations: " + _inline_json(slo["violation_counts"]),
        "",
    ])
    return lines


def _render_resources(value: object) -> list[str]:
    resources = _mapping(value, "system resources")
    window = _mapping(resources["observation_window"], "observation window")
    lines = [
        "### Resource utilization and queues",
        "",
        f"Observation window: `{_format(window['duration_s'])}` s "
        f"({_code(window['basis'])}).",
        "",
        "Worker pools:",
        "",
        "| Pool | Kernel intervals | Busy (s) | Idle (s) | Utilization |",
        "|---|---:|---:|---:|---:|",
    ]
    worker_pools = _mapping(resources["worker_pool_utilization"], "worker pools")
    for pool_name in sorted(worker_pools):
        record = _mapping(worker_pools[pool_name], f"worker pool {pool_name}")
        lines.append(
            f"| {_table_code(pool_name)} "
            f"| {_format(record['kernel_interval_count'])} "
            f"| {_format(record['busy_time_s'])} "
            f"| {_format(record['idle_time_s'])} "
            f"| {_format(record['utilization'])} |")
    lines.extend(["", "Transfer links:", ""])
    links = _mapping(resources["link_utilization"], "link utilization")
    if not links:
        lines.append("- None.")
    for resource_id in sorted(links):
        lines.append(
            f"- {_code(resource_id)}: "
            + _inline_json(links[resource_id]))
    lines.extend(["", "Communication resources:", ""])
    communication = _mapping(
        resources["communication_resource_utilization"],
        "communication resource utilization",
    )
    if not communication:
        lines.append("- None.")
    for resource_id in sorted(communication):
        lines.append(
            f"- {_code(resource_id)}: "
            + _inline_json(communication[resource_id]))
    if resources["schema_version"] == 5:
        lines.extend(["", "Host-KV rank-local links:", ""])
        host_kv_links = _mapping(
            resources["host_kv_link_utilization"],
            "Host-KV link utilization",
        )
        if not host_kv_links:
            lines.append("- None.")
        for resource_id in sorted(host_kv_links):
            lines.append(
                f"- {_code(resource_id)}: "
                + _inline_json(host_kv_links[resource_id]))
    router = _mapping(resources["router_queue"], "router queue")
    scheduler = _mapping(resources["scheduler_queue"], "scheduler queue")
    lines.extend([
        "",
        "Router queue:",
        "",
        f"- Availability: {_code(router['availability'])}",
        f"- Overall: {_inline_json(router['overall'])}",
        f"- By stage: {_inline_json(router['by_stage'])}",
    ])
    if "dispatch_service" in router:
        lines.append(
            f"- Dispatch service: {_inline_json(router['dispatch_service'])}")
    lines.extend([
        "",
        "Scheduler queue:",
        "",
        f"- Availability: {_code(scheduler['availability'])}",
        f"- Overall: {_inline_json(scheduler['overall'])}",
        f"- By stage: {_inline_json(scheduler['by_stage'])}",
        f"- By pool: {_inline_json(scheduler['by_pool'])}",
        "",
    ])
    return lines


def _render_critical_path(value: object) -> list[str]:
    path = _mapping(value, "critical path")
    semantics = _mapping(path["semantics"], "critical-path semantics")
    completion = _mapping(path["system_completion"], "system completion")
    lines = ["### Modeled critical path", "", "Semantics:", ""]
    for name in sorted(semantics):
        lines.append(
            f"- {_code(name)}: {_code(semantics[name])}")
    lines.extend([
        "",
        "Step paths:",
        "",
        "| Step | Pool | Duration (s) | Unique | Dominant category | Contributions |",
        "|---|---|---:|---|---|---|",
    ])
    for step_value in _sequence(path["steps"], "critical-path steps"):
        step = _mapping(step_value, "critical-path step")
        lines.append(
            f"| {_table_code(step['step_id'])} "
            f"| {_table_code(step['pool_name'])} "
            f"| {_format(step['duration_s'])} "
            f"| {_format(step['canonical_path_unique'])} "
            f"| {_table_code(step['dominant_category'])} "
            f"| {_markdown_cell(_inline_json(step['contribution_s_by_category']))} |")
    lines.extend([
        "",
        "Request paths:",
        "",
        "| Request | Attribution | TTFT duration (s) | TTFT dominant "
        "| E2E duration (s) | E2E dominant |",
        "|---|---|---:|---|---:|---|",
    ])
    requests = _mapping(path["requests"], "critical-path requests")
    for request_id in sorted(requests):
        request = _mapping(requests[request_id], "critical-path request")
        ttft = _mapping(request["ttft"], "request TTFT path")
        e2e = _mapping(request["e2e"], "request E2E path")
        lines.append(
            f"| {_table_code(request_id)} "
            f"| {_table_code(request['attribution'])} "
            f"| {_format(ttft['duration_s'])} "
            f"| {_table_code(ttft['dominant_category'])} "
            f"| {_format(e2e['duration_s'])} "
            f"| {_table_code(e2e['dominant_category'])} |")
    lines.extend([
        "",
        f"System completion: `{_format(completion['time_s'])}` s.",
        "",
        "| Terminal request | Path start (s) | Duration (s) "
        "| Dominant category | Contributions |",
        "|---|---:|---:|---|---|",
    ])
    terminal_paths = _sequence(
        completion["terminal_observation_paths"], "terminal observation paths")
    for terminal_path in terminal_paths:
        row = _mapping(terminal_path, "terminal observation path")
        lines.append(
            f"| {_table_code(row['request_id'])} "
            f"| {_format(row['path_start_time_s'])} "
            f"| {_format(row['path_duration_s'])} "
            f"| {_table_code(row['dominant_category'])} "
            f"| {_markdown_cell(_inline_json(row['contribution_s_by_category']))} |")
    lines.append("")
    return lines
