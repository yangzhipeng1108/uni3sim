"""Request identity and metric normalization at the external baseline boundary."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean

from evaluation.serving.metrics.latency_distribution import linear_percentile


def finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def native_workload(path: Path) -> list[dict]:
    requests = []
    for index, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        for key in ("input_toks", "output_toks", "arrival_time_ns"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if row["input_toks"] == 0 or row["output_toks"] == 0:
            raise ValueError("native flat baseline requires positive input/output lengths")
        requests.append({"request_id": str(index), "arrival_time_ns": row["arrival_time_ns"],
                         "input_toks": row["input_toks"], "output_toks": row["output_toks"]})
    if not requests:
        raise ValueError("workload must not be empty")
    return requests


def normalize_native_csv(path: Path, expected: list[dict], model: str,
                         *, instance_ids: tuple[int, ...] = (0,)) -> list[dict]:
    """Reject incomplete/changed requests, and convert native nanoseconds to seconds."""
    ids = [row["request_id"] for row in expected]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("expected request identities must be unique and nonempty")
    results = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"instance id", "request id", "model", "input", "output", "arrival", "end_time",
                    "latency", "queuing_delay", "TTFT", "TPOT", "ITL"}
        if reader.fieldnames is None or set(reader.fieldnames) != required or len(reader.fieldnames) != len(required):
            raise ValueError("invalid native result columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("invalid native result row width")
            native_id = int(row["request id"])
            if native_id < 0 or native_id >= len(expected):
                raise ValueError("unknown native request ID")
            request = expected[native_id]
            request_id = request["request_id"]
            if request_id in results:
                raise ValueError("duplicate native request ID")
            if row["model"] != model or int(row["instance id"]) not in instance_ids:
                raise ValueError("native model/instance identity mismatch")
            for column, field in (("input", "input_toks"), ("output", "output_toks"), ("arrival", "arrival_time_ns")):
                if int(row[column]) != request[field]:
                    raise ValueError(f"native request {request_id} changed {field}")
            times = {key: finite_nonnegative(float(row[key]), key) for key in
                     ("arrival", "end_time", "latency", "queuing_delay", "TTFT", "TPOT")}
            if abs(times["end_time"] - times["arrival"] - times["latency"]) > 1:
                raise ValueError("inconsistent native completion time")
            if times["TTFT"] > times["latency"]:
                raise ValueError("native first token follows completion")
            itl = json.loads(row["ITL"])
            if not isinstance(itl, list) or len(itl) != request["output_toks"] - 1:
                raise ValueError("native ITL count differs from output-token gaps")
            itl_s = [finite_nonnegative(value, "ITL") / 1e9 for value in itl]
            tpot = ((times["latency"] - times["TTFT"]) / (request["output_toks"] - 1)
                    if request["output_toks"] > 1 else 0.0)
            if abs(times["TPOT"] - tpot) > 1:
                raise ValueError("native TPOT has an inconsistent token denominator")
            results[request_id] = {"request_id": request_id, "native_request_id": native_id,
                "instance_id": int(row["instance id"]),
                "arrival_time_s": times["arrival"] / 1e9, "end_time_s": times["end_time"] / 1e9,
                "input_tokens": request["input_toks"], "output_tokens": request["output_toks"],
                "ttft_s": times["TTFT"] / 1e9, "tpot_s": tpot / 1e9,
                "latency_s": times["latency"] / 1e9, "itl_s": itl_s,
                "native_tpot_s": times["TPOT"] / 1e9}
    if set(results) != set(ids):
        raise ValueError(f"incomplete native output: {len(results)}/{len(ids)} requests")
    return [results[request_id] for request_id in ids]


def distribution(values: list[float]) -> dict:
    ordered = sorted(finite_nonnegative(v, "metric") for v in values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None}
    return {"count": len(ordered), "mean": fmean(ordered),
            "p50": linear_percentile(ordered, 50), "p90": linear_percentile(ordered, 90),
            "p99": linear_percentile(ordered, 99)}


def summarize(requests: list[dict]) -> dict:
    if not requests:
        raise ValueError("cannot summarize an empty run")
    end = max(row["end_time_s"] for row in requests)
    if end <= 0:
        raise ValueError("completion window must be positive")
    return {"request_count": len(requests),
            "ttft_s": distribution([row["ttft_s"] for row in requests]),
            "tpot_s": distribution([row["tpot_s"] for row in requests]),
            "latency_s": distribution([row["latency_s"] for row in requests]),
            "itl_s": distribution([v for row in requests for v in row["itl_s"]]),
            "throughput_window": "workload time zero to last completion, matching Midlayer",
            "throughput_window_s": end,
            "request_throughput_per_s": len(requests) / end,
            "output_token_throughput_per_s": sum(row["output_tokens"] for row in requests) / end,
            "tpot_denominator": "output_tokens - 1; one output token yields zero"}


def bundled_truth(path: Path, expected: list[dict]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != len(expected):
        raise ValueError("bundled truth request count mismatch")
    indexed = {row["request_id"]: row for row in rows}
    if set(indexed) != {f"bench-{i}" for i in range(len(expected))}:
        raise ValueError("bundled truth has missing, duplicate or unknown request IDs")
    origin = min(row["queued_ts"] for row in rows)
    result = []
    for index, request in enumerate(expected):
        row = indexed[f"bench-{index}"]
        if row["input_toks"] != request["input_toks"] or row["output_toks"] != request["output_toks"]:
            raise ValueError("bundled truth lengths differ from workload")
        ttft = finite_nonnegative(row["first_token_ts"] - row["queued_ts"], "truth TTFT")
        latency = finite_nonnegative(row["last_token_ts"] - row["queued_ts"], "truth E2E")
        if latency < ttft:
            raise ValueError("truth completion precedes first token")
        result.append({"request_id": request["request_id"], "input_tokens": row["input_toks"],
            "output_tokens": row["output_toks"], "ttft_s": ttft, "latency_s": latency,
            "tpot_s": (latency - ttft) / (row["output_toks"] - 1) if row["output_toks"] > 1 else 0.0,
            "arrival_time_s": row["queued_ts"] - origin,
            "end_time_s": row["last_token_ts"] - origin, "itl_s": []})
    return result


def compare_requests(prediction: list[dict], reference: list[dict]) -> dict:
    if [r["request_id"] for r in prediction] != [r["request_id"] for r in reference]:
        raise ValueError("comparison request identities/order differ")
    result = {}
    for field in ("ttft_s", "tpot_s", "latency_s"):
        errors = [abs(p[field] - r[field]) for p, r in zip(prediction, reference)]
        relative = [abs(p[field] - r[field]) / r[field] * 100 for p, r in zip(prediction, reference) if r[field] > 0]
        pred, ref = distribution([p[field] for p in prediction]), distribution([r[field] for r in reference])
        result[field] = {"prediction": pred, "reference": ref, "mae_s": fmean(errors),
                         "mape_percent": fmean(relative) if relative else None,
                         "mape_eligible": len(relative), "mape_zero_reference_excluded": len(errors) - len(relative),
                         "mean_relative_error_percent": (pred["mean"] / ref["mean"] - 1) * 100 if ref["mean"] > 0 else None}
    def observed_window(rows):
        duration = max(r["end_time_s"] for r in rows) - min(r["arrival_time_s"] for r in rows)
        return {"window_s": duration, "requests_per_s": len(rows) / duration,
                "output_tokens_per_s": sum(r["output_tokens"] for r in rows) / duration}
    result["throughput"] = {"window": "first arrival to last completion, evaluated independently for each run",
                            "prediction": observed_window(prediction), "reference": observed_window(reference)}
    return result


def summarize_target_scope(cases: list[dict], statuses: dict[str, str],
                           model_ids: list[str], case_suffix: str) -> dict:
    """Keep active failures in their denominator and deferred history outside it."""
    if (not isinstance(statuses, dict) or not isinstance(model_ids, list) or not model_ids
            or any(not isinstance(model, str) or not model for model in model_ids)
            or not isinstance(case_suffix, str) or not case_suffix
            or any(status not in ('active', 'deferred_by_user') for status in statuses.values())
            or 'active' not in statuses.values() or len(set(model_ids)) != len(model_ids)):
        raise ValueError('invalid target scope')
    expected = {f'{hardware}/{model}/{case_suffix}' for hardware in statuses for model in model_ids}
    ids = [entry['case'] for entry in cases]
    if len(set(ids)) != len(ids) or set(ids) != expected:
        raise ValueError('target scope changes the historical case denominator')
    active = [entry for entry in cases if statuses[entry['case'].split('/')[0]] == 'active']
    deferred = [entry['case'] for entry in cases if statuses[entry['case'].split('/')[0]] == 'deferred_by_user']
    return {'current_expected_cases': len(active),
            'current_completed_simulations': sum(entry['status'] == 'completed' for entry in active),
            'current_status_counts': {status: sum(entry['status'] == status for entry in active)
                                      for status in sorted({entry['status'] for entry in active})},
            'deferred_by_user': deferred, 'historical_expected_cases': len(cases),
            'historical_completed_simulations': sum(entry['status'] == 'completed' for entry in cases)}


def generate_report(runtime: Path, output: Path) -> dict:
    from evaluation.serving.experiments.llmservingsim import (
        REPOSITORY_ROOT, SOURCE, read_json, write_json, target_hardware_statuses,
    )
    import yaml

    source = read_json(SOURCE)
    cases = []
    for case in source["bundled_cases"]:
        directory = output / "bundled" / case.replace("/", "-")
        manifest = read_json(directory / "run.json")
        entry = {"case": case, "status": manifest["status"], "exit_code": manifest["exit_code"],
                 "request_count": manifest["request_count"], "output": str(directory.resolve())}
        if manifest["status"] == "completed":
            native = read_json(directory / "normalized.json")
            expected = native_workload(runtime / manifest["meta"]["dataset_path"])
            example = runtime / "bench/examples" / case
            model = manifest["config"]["nodes"][0]["instances"][0]["model_name"]
            instance_ids = tuple(range(sum(node["num_instances"] for node in manifest["config"]["nodes"])))
            historical = normalize_native_csv(example / "outputs/sim.csv", expected, model, instance_ids=instance_ids)
            truth = bundled_truth(example / "vllm/requests.jsonl", expected)
            write_json(directory / "vllm_normalized.json", {"requests": truth, "summary": summarize(truth)})
            profile_bundles = []
            for node in manifest["config"]["nodes"]:
                for instance in node["instances"]:
                    profile_root = runtime / "profiler/perf" / instance["hardware"] / instance["model_name"]
                    for meta_path in sorted(profile_root.glob("*/meta.yaml")):
                        profile_bundles.append({"path": str(meta_path), "metadata": yaml.safe_load(meta_path.read_text())})
            log_lines = (directory / "stdout.log").read_text().splitlines()
            warnings = [line for line in log_lines if "warn" in line.lower() or "extrapolat" in line.lower()]
            comparison = {"historical_sim_reproduction": compare_requests(native["requests"], historical),
                          "vllm_accuracy": compare_requests(native["requests"], truth),
                          "summary": native["summary"], "truth_summary": summarize(truth),
                          "model_identity": {"simulator": model, "recorded_vllm": manifest["meta"]["model"]},
                          "truth_itl": "unavailable: bundled vLLM records first/last token only",
                          "arrival_semantics": "workload planned arrival versus recorded vLLM queued_ts; frontend/engine delays can differ",
                          "profile_coverage": "native interpolation/extrapolation unchanged; runtime warnings retained, per-lookup counts not instrumented",
                          "profile_bundles": profile_bundles,
                          "native_warnings": {"stdout": warnings, "stderr": (directory / "stderr.log").read_text()}}
            write_json(directory / "comparison.json", comparison)
            entry["comparison"] = comparison
        cases.append(entry)
    report = {"goal_status": "checkpoint_pending_main_acceptance", "bundled_cases": cases,
              "target_cases": "not prepared",
              "native_compute_modes": [
                  {"compute": "LUT", "network": "analytical", "status": "available"},
                  {"compute": "pure_analytical", "network": None, "status": "unavailable",
                   "source_binding": str(SOURCE),
                   "evidence": ["serving/__main__.py: --network-backend selects network only",
                                "serving/core/trace_generator.py: _load_perf_db, _lookup_dense, _lookup_attention",
                                "docs/docs/getting-started/overview.mdx: Profiled compute, configurable memory"]}]}
    target_path = output / "targets.json"
    if target_path.is_file():
        config = read_json(REPOSITORY_ROOT / 'configs/baselines/llmservingsim/targets.json')
        scope = target_hardware_statuses(REPOSITORY_ROOT)
        report["target_cases"] = read_json(target_path)
        for entry in report["target_cases"]["cases"]:
            hardware, model, _ = entry["case"].split("/")
            entry['scope_status'] = scope[hardware]
            directory = output / "targets" / hardware / model
            latest = read_json(directory / 'case.json')
            if latest['case'] != entry['case'] or latest['request_count'] != entry['request_count']:
                raise ValueError('target case state changes the preserved denominator')
            entry['status'] = latest['status']
            entry['llmservingsim_prediction'] = latest['llmservingsim_prediction']
            entry['comparability_gaps'] = latest['comparability_gaps']
            if latest['status'] == 'profile_geometry_mismatch':
                entry['profile_validation'] = latest['profile_validation']
            state_path = directory / "midlayer-request-metrics/run.json"
            if state_path.is_file():
                state = read_json(state_path)
                entry["midlayer_status"] = state["status"]
                if state["status"] == "executed" and state["exit_code"] == 0:
                    prediction = read_json(state_path.parent / "prediction.json")
                    mapping = read_json(directory / "request_mapping.json")["requests"]
                    truth_payload = read_json(Path(entry["preserved_inputs"]) / "metrics.json")
                    per_request = truth_payload["per_request"]
                    if len(per_request) != len(mapping) or len(prediction["requests"]) != len(mapping):
                        raise ValueError("target prediction/truth request count differs from canonical")
                    truth_by_id = {r["request_idx"]: r for r in per_request}
                    if set(truth_by_id) != set(range(len(mapping))):
                        raise ValueError("target truth request identities mismatch")
                    expected_ids = {f"request-{i:012d}" for i in range(len(mapping))}
                    if set(prediction["requests"]) != expected_ids:
                        raise ValueError("Midlayer request identities mismatch")
                    pred_rows, truth_rows = [], []
                    for index, request in enumerate(mapping):
                        metric = prediction["requests"][f"request-{index:012d}"]
                        truth_metric = truth_by_id[index]
                        if truth_metric["actual_output_tokens"] != request["output_tokens"]:
                            raise ValueError("target observed output-token count differs from workload")
                        base = {"request_id": request["request_id"], "input_tokens": request["input_tokens"],
                                "output_tokens": request["output_tokens"], "arrival_time_s": request["arrival_time_ms"] / 1000,
                                "itl_s": []}
                        pred_rows.append({**base, "ttft_s": metric["ttft_s"], "tpot_s": metric["tpot_s"],
                                          "latency_s": metric["latency_s"],
                                          "end_time_s": base["arrival_time_s"] + metric["latency_s"]})
                        truth_rows.append({**base, "ttft_s": truth_metric["ttft_ms"] / 1000,
                                           "tpot_s": truth_metric["tpot_ms"] / 1000,
                                           "latency_s": truth_metric["e2e_latency_ms"] / 1000,
                                           "end_time_s": base["arrival_time_s"] + truth_metric["e2e_latency_ms"] / 1000})
                    comparison = {"case": entry["case"], "midlayer_vs_truth": compare_requests(pred_rows, truth_rows),
                                  "midlayer_summary": summarize(pred_rows), "truth_summary": summarize(truth_rows),
                                  "truth_throughput_limit": "completion reconstructed from canonical arrival plus observed E2E; no absolute original run origin",
                                  "performance_timing_status": prediction["aggregate"]["performance_timing_status"],
                                  "llmservingsim_status": entry["status"], "llmservingsim_vs_truth": None}
                    if entry['status'] == 'completed':
                        native_manifest = Path(entry['llmservingsim_prediction']).parent / 'run.json'
                        native_state = read_json(native_manifest)
                        if native_state['status'] != 'completed' or native_state['exit_code'] != 0:
                            raise ValueError('target completion lacks a successful native run')
                        native = read_json(Path(entry['llmservingsim_prediction']))
                        comparison['llmservingsim_vs_truth'] = compare_requests(native['requests'], truth_rows)
                        comparison['llmservingsim_vs_midlayer'] = compare_requests(native['requests'], pred_rows)
                        comparison['llmservingsim_summary'] = native['summary']
                        comparison['native_run_manifest'] = str(native_manifest)
                        comparison['native_compute_mode'] = 'LUT'
                        comparison['native_network_mode'] = 'analytical'
                        if (native_state['native_prefix_hit_prompt_tokens'] != 0
                                or not native_state['capacity']['equal_block_bytes']
                                or not native_state['capacity']['equal_block_count']):
                            raise ValueError('completed target lacks matching cache-work/KV capacity evidence')
                        comparison['input_capacity_cache_evidence'] = {
                            'request_count': native_state['request_count'],
                            'native_prefix_hit_prompt_tokens': native_state['native_prefix_hit_prompt_tokens'],
                            'cache_representation': native_state['cache_representation'],
                            'capacity': native_state['capacity'],
                            'unused_resources': native_state['unused_resources']}
                        comparison['profile_evidence'] = {
                            'capture': native_state['profile_capture'],
                            'hardware_capture': native_state['hardware_capture'],
                            'metadata': native_state['profile_metadata'],
                            'tables': native_state['profile_tables'],
                            'truth_runtime': entry['canonical_runtime'],
                            'limitations': entry['comparability_gaps']}
                        comparison['native_warnings'] = {
                            'stdout': [line for line in (native_manifest.parent / 'stdout.log').read_text().splitlines()
                                       if 'warn' in line.lower() or 'extrapolat' in line.lower()],
                            'stderr': (native_manifest.parent / 'stderr.log').read_text()}
                        if hardware == 'ascend_910b':
                            comparison['collection_adapter'] = native_state['collection_adapter']
                    write_json(directory / "comparison.json", comparison)
                    write_json(directory / "normalized_comparison_inputs.json", {"midlayer": pred_rows, "truth": truth_rows})
                    entry["midlayer_comparison"] = comparison
        report['target_cases']['completed_simulations'] = sum(row['status'] == 'completed' for row in report['target_cases']['cases'])
        report['target_cases']['missing_profiles'] = sum(row['status'] == 'missing_profile' for row in report['target_cases']['cases'])
        report['target_scope'] = summarize_target_scope(report['target_cases']['cases'], scope,
            [model['model_id'] for model in config['models']], config['case_id'])
    capacity_path = output / "capacity-comparison.json"
    if capacity_path.is_file():
        report["static_capacity_comparison"] = read_json(capacity_path)
    write_json(output / "comparison.json", report)
    lines = ["# LLMServingSim baseline checkpoint", "", "Goal incomplete — checkpoint, pending primary-agent acceptance. Current scope and completion status are listed below; user-deferred items do not count as success.", "",
             "## Bundled CPU frontend → Chakra → ASTRA-Sim runs", "",
             "Reproduction uses historical sim.csv; accuracy uses recorded vLLM requests. Native outputs are generated by fresh CPU runs.", "",
             "| Case | Requests | Status | Mean TTFT (s) | Mean TPOT (ms) | Mean E2E (s) | TTFT / TPOT / E2E mean error vs vLLM (%) |",
             "|---|---:|---|---:|---:|---:|---|"]
    for entry in cases:
        if entry["status"] == "completed":
            comparison = entry["comparison"]
            s = comparison["summary"]
            errors = [comparison["vllm_accuracy"][f]["mean_relative_error_percent"] for f in ("ttft_s", "tpot_s", "latency_s")]
            lines.append(f"| {entry['case']} | {entry['request_count']} | completed | {s['ttft_s']['mean']:.6f} | {s['tpot_s']['mean'] * 1000:.6f} | {s['latency_s']['mean']:.6f} | " + " / ".join(f"{e:+.3f}" for e in errors) + " |")
        else:
            lines.append(f"| {entry['case']} | {entry['request_count']} | {entry['status']} | — | — | — | — |")
    lines.extend(["", "Each case directory contains run.json (exact command/settings/exit), sim.csv, normalized.json, comparison.json and stdout/stderr. comparison.json includes mean/P50/P90/P99, request MAE/MAPE and the full request denominator. TPOT divides by output_tokens−1; native integer nanoseconds are preserved separately. Throughput comparison uses first arrival to last completion; normalized Midlayer-style throughput additionally uses workload zero to final completion.", "", "Native interpolation/extrapolation is retained. No per-lookup coverage counters were added to third-party code; warnings and static profile limits are retained. vLLM records contain no full ITL trace, so ITL error is unavailable.", "", "## Target matrix", ""])
    if isinstance(report["target_cases"], dict):
        current = report['target_scope']
        lines.extend([f"Current scope: {current['current_completed_simulations']}/{current['current_expected_cases']} target simulations completed. {len(current['deferred_by_user'])} historical cases are deferred by the user and excluded from this round's denominator. The preserved historical matrix contains {current['historical_expected_cases']} cases.", '',
                      '### Current cases', ''])
        lines.extend(["| Case | Requests | LLMServingSim | Midlayer | LLMServingSim TTFT / TPOT / E2E mean error (%) | Midlayer TTFT / TPOT / E2E mean error (%) |", "|---|---:|---|---|---|---|"])
        for entry in report["target_cases"]["cases"]:
            if entry['scope_status'] == 'deferred_by_user':
                continue
            midlayer = entry["midlayer_status"] if "midlayer_status" in entry else "not run"
            if entry["midlayer_comparison"] is None:
                errors = "—"
            else:
                metric = entry["midlayer_comparison"]["midlayer_vs_truth"]
                errors = " / ".join(f"{metric[field]['mean_relative_error_percent']:+.3f}" for field in ("ttft_s", "tpot_s", "latency_s"))
            native_errors = '—'
            if entry['midlayer_comparison'] is not None and entry['midlayer_comparison']['llmservingsim_vs_truth'] is not None:
                metric = entry['midlayer_comparison']['llmservingsim_vs_truth']
                native_errors = ' / '.join(f"{metric[field]['mean_relative_error_percent']:+.3f}" for field in ('ttft_s','tpot_s','latency_s'))
            lines.append(f"| {entry['case']} | {entry['request_count']} | {entry['status']} | {midlayer} | {native_errors} | {errors} |")
        lines.extend(["", "Every active case remains in this round's denominator, including failures, missing profiles and OOM. Native interpolation/extrapolation is retained; per-lookup counts are unavailable. comparison.json keeps active status counts and all historical cases. See target manifests for original settings, preserved inputs and comparability limitations.", '',
                      '### User-deferred history', '',
                      '| Case | Scope | Historical native state | Historical Midlayer state | Historical Midlayer TTFT / TPOT / E2E mean error (%) |',
                      '|---|---|---|---|---|'])
        for entry in report['target_cases']['cases']:
            if entry['scope_status'] != 'deferred_by_user':
                continue
            errors = '—'
            if entry['midlayer_comparison'] is not None:
                metric = entry['midlayer_comparison']['midlayer_vs_truth']
                errors = ' / '.join(f"{metric[field]['mean_relative_error_percent']:+.3f}" for field in ('ttft_s', 'tpot_s', 'latency_s'))
            midlayer = entry['midlayer_status'] if 'midlayer_status' in entry else 'not run'
            lines.append(f"| {entry['case']} | deferred_by_user | {entry['status']} | {midlayer} | {errors} |")
        lines.extend(['', 'H200 experiments are deferred by explicit user instruction. Historical inputs and Midlayer results are preserved; no H200 capture, simulation, SSH probe or entry request is part of the current work. Deferred cases are neither successes nor active failures.'])
        for entry in report['target_cases']['cases']:
            if entry['status'] == 'profile_geometry_mismatch':
                geometry = entry['profile_validation']
                lines.extend(['', f"{entry['case']}: requested block size {geometry['requested']['block_size']}, captured native block size {geometry['resolved']['block_size']}. The profile is retained, the original target is unchanged, and no target simulation was executed. See the case's profile-validation.json and the raw capture meta.yaml. The installed Ascend dense backend declares only [128] in get_supported_kernel_block_sizes(); source evidence is in profiler_output/serving/llmservingsim-20261016/910b-block-geometry-evidence.txt. A matching runtime or bundle is required."])
            elif (entry['status'] == 'completed' and entry['native_hardware'] == 'ascend_910b'
                  and entry['midlayer_comparison'] is not None
                  and entry['midlayer_comparison']['collection_adapter'] is not None):
                lines.extend(['', f"{entry['case']}: actual block16 profiles use a collector-private capability adapter. Same-runtime FIA and paged-attention correctness controls at block16/block128, plus mixed causal TND controls, are retained in the raw campaign. Only block sizing policy and capability advertisement changed; native attention/cache device implementations and native lookup are unchanged. The old block128 profiles remain separate. The run manifest records the evidence binding and adapter scope."])
        lines.extend(['', '## Current input, cache and profile evidence', '',
                      'Canonical request IDs, arrival times, input/output lengths and request counts are validated before comparing each completed target. Each target run.json contains the exact native command, input mapping, software environment, source binding and profile identities. Prefix allocation stays enabled: unique synthetic token IDs instantiate the canonical scheduler-equivalent zero net prefix-work representative. Logical hit counts and original prompt tokens remain unobserved. The native output confirms zero prompt work reduction for these representatives; this is not an observed-zero hit claim.', '',
                      '| Case | Requests | Collector / truth vLLM | Actual block / context | Dense / sequence / attention rows | Native prefix-work reduction | Evidence |',
                      '|---|---:|---|---|---|---:|---|'])
        measured_ranges = []
        for entry in report['target_cases']['cases']:
            if entry['scope_status'] != 'active' or entry['status'] != 'completed':
                continue
            comparison = entry['midlayer_comparison']
            proof = comparison['profile_evidence']
            meta = proof['metadata']
            resolved = meta['engine_resolved']['per_tp']['1']
            rows = ' / '.join(str(proof['tables'][group]['rows']) for group in ('dense', 'per_sequence', 'attention'))
            prefix = comparison['input_capacity_cache_evidence']['native_prefix_hit_prompt_tokens']
            manifest = comparison['native_run_manifest']
            lines.append(f"| {entry['case']} | {entry['request_count']} | {meta['vllm_version']} / {entry['canonical_runtime']['runtime_version']} | {resolved['block_size']} / {resolved['max_model_len']} | {rows} | {prefix} | [run.json]({manifest}) · [raw]({proof['capture']}) |")
            ranges = '; '.join(f"{group}: {json.dumps(proof['tables'][group]['measured_ranges'], sort_keys=True)}" for group in ('dense', 'per_sequence', 'attention'))
            measured_ranges.extend(['', f"{entry['case']} measured coordinate ranges: {ranges}."])
        lines.extend(measured_ranges)
        lines.extend(['', 'Profiles are BF16/TP1 one-layer dummy-weight eager executions with initialized dummy query/KV tensors. The full target simulation uses the original full model configuration and all requests; trained-weight end-to-end collection is not claimed. Profiles disable prefix sharing to measure explicit query/KV work, while the target simulator keeps the evidenced prefix-allocation policy enabled. Equal runtime release numbers do not prove identical plugin builds, graph execution, host scheduling or memory residency. Raw capture manifests and target comparability_gaps retain these differences. The measured ranges above are not a promise that every target lookup is directly sampled; native interpolation/extrapolation and its warnings are retained without per-lookup counters.'])
        if "static_capacity_comparison" in report:
            lines.extend(["", "## CPU static KV capacity check", "",
                          "Native MemoryModel was instantiated on CPU for every target, without loading latency profiles or running a target simulation. The same nominal memory-utilization gives different KV capacity. The explicit conversion below preserves the KV byte budget from the bound runtime allocation manifest: native utilization = (native weight bytes + specified available KV bytes) / physical HBM bytes. No duration or accuracy error enters this conversion; no target config was changed.", "",
                          "| Case | Scope | Specified KV blocks | Native blocks at nominal utilization | Native blocks after budget conversion | Converted native utilization |",
                          "|---|---|---:|---:|---:|---:|"])
            for row in report["static_capacity_comparison"]["cases"]:
                status = scope[row['case'].split('/')[0]]
                lines.append(f"| {row['case']} | {status} | {row['midlayer_blocks']} | {row['native_blocks_at_same_utilization']} | {row['native_matched_blocks']} | {row['native_utilization_for_same_kv_budget']:.9f} |")
            lines.extend(["", "Block byte sizes and counts match in the retained CPU checks, including all current cases; H200 rows are historical checks reused without new experiments. capacity-comparison.json retains the original policy, available bytes, native and Midlayer weight counts, and environment. This closes logical KV allocation, not full runtime residency: native weight accounting differs, and host timing, scheduler, graph execution and plugin builds are not asserted equivalent."])
        lines.extend(["", "## Profile provenance and collection limits", "",
                      "Current matrix: nvidia_a800, metax_c500, ascend_910b × Qwen3-4B / Llama-3.1-8B-Instruct, BF16, TP1. H200's two historical cases are deferred by the user. Each current bundle has hardware metadata and model/engine/profiler provenance plus dense, attention and per-sequence layerwise tables. Requested bounds are 8192 batched tokens, 256 sequences, 8192 context tokens; actual measured ranges and request length ranges are retained separately. Existing RTX tables and operator-level durations cannot substitute.", "",
                      "Both models on A800, C500 and Ascend completed real BF16 TP1 one-layer dummy-weight coverage: prefill/decode/mixed each bound 100% of module device time. Raw run manifests, commands, failures and logs are under profiler_output/serving/llmservingsim-20261016; generated execution-copy adapters are under this report directory's profiler-adapters. No model weights were downloaded. Native uniform-only profiles retain their skip-skew limitation; the attention geometric grid factor is eight. Earlier factor-two candidates and failed captures remain separate or were explicitly resumed with native measurement-identity checks; no cross-hardware/version timings are mixed. Ascend retains native token points 1..15 then fixed powers of two to bound CANN parsing cost; all grid changes preserve requested domain limits and native lookup/interpolation. Grid density was chosen from collection cost, without comparison errors.", "",
                      "A800 uses the Torch2.8/vLLM0.11 collection adapter. Linked CUPTI CPU records establish logical-to-OS thread ownership because ordinary operator events can report resource0; unique launch ownership and native device intervals remain required. The assigned GPU UUID and physical-to-visible mapping are recorded in the capture manifest. NVIDIA hardware metadata uses UUID-bound NVML specification queries and the native DDR bandwidth derivation; no interconnect benchmark is claimed for the single-card TP1 target.", "",
                      "Llama base versus the existing Instruct config differs only in eos_token_id and transformers_version at this pinned revision; all other config fields match. Target profiles use the matching Instruct config. Length-only inputs and synthetic token identities do not reproduce tokenizer behavior or original prompt sharing. This is a conditional comparison under the disclosed input, KV-budget and cache-work representative; exact runtime equivalence is not claimed."])
    else:
        lines.append("Target inputs not yet prepared.")
    lines.extend(["", "## Native compute modes", "",
                  "| Compute | Network | Result |",
                  "|---|---|---|",
                  "| LUT (profile lookup with native interpolation/extrapolation) | analytical | Five bundled settings completed; target matrix above |",
                  "| Pure analytical compute | — | unavailable in the bound native frontend for these dense Transformer models |", "",
                  "Source evidence at the revision bound by configs/baselines/llmservingsim/source.json: serving/__main__.py exposes --network-backend analytical/ns3; serving/core/trace_generator.py unconditionally loads _load_perf_db and uses _lookup_dense/_lookup_attention for ordinary Transformer compute. docs/docs/getting-started/overview.mdx describes 'Profiled compute, configurable memory'. Analytical network and PIM-specific paths do not provide a pure analytical compute alternative for the target models. No analytical compute result is fabricated."])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report
