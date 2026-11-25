#!/usr/bin/env python3
"""Publish the WP6 H200 FA2 dependency references from derived reports.

Inputs are the small derived files copied back from the H200 run
directories (report.json, manifest.json, build/build_manifest.json); the
bulky raw CSVs stay on the remote host and are identified by the SHA256
recorded in each capture manifest. Raw paths are relative to the capture
checkout. No value is computed here beyond
re-labelling the report parameters into the H200 hardware-config field
names used by the A800 configuration (``h200_yaml_values``).

Every reference keeps all keys of its A800 counterpart (the A800 reference
JSON where one exists, otherwise the A800 report.json schema produced by the
same reporter) and adds exactly two keys: ``h200_provenance`` (paths,
SHA256s, remote raw directory, clock-gate statistics, accepted/rejected
cells) and ``h200_yaml_values``.
"""

import argparse
import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
H200_ROOT = REPOSITORY_ROOT / "microbench" / "nvidia_h200"
REFERENCE_DIR = H200_ROOT / "references"
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output" / "microbench" / "nvidia_h200"
MEASURED_SM_CLOCK_MHZ = 1980
FA = "operators/flash_attn"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must hold a JSON object")
    return value


def _files(*relative):
    return {
        name: {"path": f"microbench/nvidia_h200/{name}",
               "sha256": _sha256(H200_ROOT / name)}
        for name in relative
    }


def _clock(gate):
    return {
        key: gate[key] for key in (
            "logger", "target_sm_clock_mhz", "total_samples",
            "busy_samples", "busy_sm_clock_mhz_min",
            "busy_sm_clock_mhz_max", "busy_sm_clock_mhz_mean")
    }


def _gate_counts(gate):
    passes = gate["passes"]
    if passes[-1]["rejected"] != 0:
        raise ValueError("a retained cell failed the clock gate")
    return {
        "passes": len(passes),
        "accepted_cells": passes[0]["cells"],
        "rejected_cell_runs_rerun": sum(item["rejected"] for item in passes),
        "rejected_cells": [
            cell for item in passes for cell in item["rejected_cells"]],
    }


def _provenance(run_dir, files, manifest_gate, *, raw_relative, raw_sha256,
                baseline_commit, extra=None):
    provenance = {
        "hardware_id": "nvidia_h200",
        "device": 'NVIDIA H200 SXM (GPU 2)',
        "files": files,
        "git_commit_of_probe_sources": baseline_commit,
        "run_id": run_dir.name,
        "local_derived_dir": str(run_dir.relative_to(REPOSITORY_ROOT)),
        "remote_raw_path": f"{run_dir.relative_to(REPOSITORY_ROOT)}/"
                           f"{raw_relative}",
        "raw_sha256": raw_sha256,
        "compiler": "CUDA 12.2 nvcc -arch=sm_90 -cudart static "
                    "(same toolchain as the A800 sm_80 captures)",
        "clock": _clock(manifest_gate),
        "clock_gate": _gate_counts(manifest_gate),
        "measured_sm_clock_mhz_for_cycle_ns_conversion": MEASURED_SM_CLOCK_MHZ,
        "vllm_runtime_note": (
            "these probes model the FA2 (sm80 source) program unchanged; "
            "the H200 vLLM 0.24 deployment runs FA3 for MLA attention"),
        "production_or_evaluation_duration_consumed": False,
    }
    if extra is not None:
        provenance.update(extra)
    return provenance


def _write(name, value):
    path = REFERENCE_DIR / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def scalar(run_dir, resident_dir):
    report = _load(run_dir / "report.json")
    manifest = _load(run_dir / "manifest.json")
    if report["hardware_id"] != "nvidia_h200":
        raise ValueError("scalar report is not an H200 report")
    resident = _load(resident_dir / "report.json")
    field = {
        "fp32_max_quad4": "fp32_max_quad4_cycles",
        "fp32_local_sum": "fp32_local_sum_cycles",
        "fp32_scale_exp2_sum": "fp32_scale_exp2_sum_cycles",
        "cta_barrier": "cta_barrier_cycles",
        "fp32_quad4_sum": "fp32_quad4_sum_cycles",
        "integer_column_predicate": "integer_column_predicate_cycles",
        "fp32_to_bf16_conversion": "fp32_to_bf16_conversion_cycles",
    }
    entries = {}
    for parameter in report["parameters"]:
        key = (parameter["lane_local_rows_per_thread"],
               parameter["score_values_per_lane_local_row"])
        if key not in entries:
            entries[key] = {
                "lane_local_rows_per_thread": key[0],
                "score_values_per_lane_local_row": key[1],
            }
        entries[key][field[parameter["mode_name"]]] = parameter[
            "cycles_per_128_thread_cta_stage"]
    factors = {}
    for bound in resident["contention_span_upper_bounds"]:
        key = (bound["lane_local_rows_per_thread"],
               bound["score_values_per_lane_local_row"])
        if key not in factors:
            factors[key] = {"resident_ctas_per_sm": 2}
        factors[key][bound["mode_name"]] = bound[
            "per_cta_demand_factor_upper_bound"]
    # The A800 configuration attaches the optional two-resident factors to
    # the (4, 16) fragment only; the same attachment is kept for H200.
    entries[(4, 16)]["two_resident_demand_factors"] = factors[(4, 16)]
    reference = dict(report)
    reference["h200_provenance"] = _provenance(
        run_dir, _files(
            f"{FA}/fa2_scalar_dependency.cu",
            f"{FA}/fa2_scalar_dependency_runner.py",
            f"{FA}/fa2_scalar_dependency_report.py"),
        manifest["clock_gate"], raw_relative="raw.csv",
        raw_sha256=manifest["raw_sha256"],
        baseline_commit=manifest["baseline_commit"],
        extra={"report_sha256": _sha256(run_dir / "report.json"),
               "binary_sha256": manifest["binary_sha256"]})
    reference["h200_yaml_values"] = {
        "yaml_key": "register_reduction_dependency_cycles",
        "software_revision": "cuda_12.2_sm90",
        "entries": [entries[key] for key in sorted(entries)],
        "two_resident_demand_factors_all_geometries": [
            {"lane_local_rows_per_thread": key[0],
             "score_values_per_lane_local_row": key[1], **factors[key]}
            for key in sorted(factors)],
    }
    return _write("fa2_scalar_dependency_sm90.json", reference)


def resident(run_dir):
    report = _load(run_dir / "report.json")
    manifest = _load(run_dir / "manifest.json")
    if report["status"] != "pass" or report["hardware_id"] != "nvidia_h200":
        raise ValueError("resident report did not pass on H200")
    bounds = [
        {key: bound[key] for key in (
            "lane_local_rows_per_thread", "score_values_per_lane_local_row",
            "mode_name", "maximum_grid_span_ratio_216_to_108",
            "per_cta_demand_factor_upper_bound", "cross_trial_ratio_spread")}
        for bound in report["contention_span_upper_bounds"]]
    reference = {
        "schema": "nvidia_h200_fa2_resident_dependency_overlap_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": manifest["device"]["name"],
        "compute_capability": "sm_90",
        "evidence_kind": (
            "independent_two_resident_cta_scalar_dependency_microbenchmark"),
        "source_capture": str(run_dir.relative_to(REPOSITORY_ROOT)),
        "source_baseline_commit": manifest["baseline_commit"],
        "physical_device_index": manifest["device"]["index"],
        "device_uuid": manifest["device"]["uuid"],
        "timer_domains": [
            "cuda_event_grid_span", "clock64_same_cta_envelope_cycles"],
        "resource_contract": {
            "threads_per_cta": 128,
            "dynamic_shared_bytes_per_cta": 77824,
            "resident_ctas_per_sm": 2,
            "physical_sm_count": 132,
            "one_resident_grid": 132,
            "two_resident_grid": 264,
        },
        "source_geometries": [
            {"lane_local_rows_per_thread": rows,
             "score_values_per_lane_local_row": values}
            for rows, values in manifest["geometries"]],
        "overlap_classes": ["cta_barrier", "fp32_quad4_sum"],
        "contention_span_upper_bounds": bounds,
        "maximum_overlap_grid_span_ratio_216_to_108": report[
            "maximum_overlap_grid_span_ratio"],
        "maximum_overlap_cta_cycle_ratio_216_to_108": report[
            "maximum_overlap_cta_cycle_ratio"],
        "maximum_overlap_cross_trial_ratio_spread": report[
            "maximum_overlap_cross_trial_ratio_spread"],
        "publication_gates": report["gates"],
        "artifact_sha256": {
            "source": _sha256(
                H200_ROOT / FA / "fa2_resident_dependency_overlap.cu"),
            "runner": _sha256(
                H200_ROOT / FA / "fa2_resident_dependency_overlap_runner.py"),
            "reporter": _sha256(
                H200_ROOT / FA / "fa2_resident_dependency_overlap_report.py"),
            "capture_manifest": _sha256(run_dir / "manifest.json"),
            "raw": manifest["raw_sha256"],
            "report": _sha256(run_dir / "report.json"),
            "binary": manifest["binary_sha256"],
        },
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "For the three captured source geometries with exactly two "
            "resident 128-thread CTAs per SM on H200, each measured "
            "dependency has a finite conservative per-CTA demand-factor "
            "upper bound derived from the maximum 264/132 aggregate "
            "grid-span ratio across three trials. CTA barriers and FP32 quad "
            "reductions additionally satisfy the frozen near-complete-overlap "
            "gates. Keys named *_216_to_108 keep the A800 schema names and "
            "hold the H200 264/132 ratios."),
        "excluded_conclusions": [
            "a complete production FlashAttention duration",
            "resident counts other than two",
            "source geometries outside the three captured entries",
            "CUDA admission topology",
        ],
    }
    reference["h200_provenance"] = _provenance(
        run_dir, _files(
            f"{FA}/fa2_resident_dependency_overlap.cu",
            f"{FA}/fa2_scalar_dependency.cu",
            f"{FA}/fa2_resident_dependency_overlap_runner.py",
            f"{FA}/fa2_resident_dependency_overlap_report.py"),
        manifest["clock_gate"], raw_relative="raw.csv",
        raw_sha256=manifest["raw_sha256"],
        baseline_commit=manifest["baseline_commit"],
        extra={"key_name_note": (
            "*_216_to_108 keys hold span(264)/span(132) on H200")})
    reference["h200_yaml_values"] = {
        "yaml_key": (
            "register_reduction_dependency_cycles.entries"
            "[lane_local_rows_per_thread=4,score_values_per_lane_local_row="
            "16].two_resident_demand_factors"),
        "two_resident_demand_factors": {
            "resident_ctas_per_sm": 2,
            **{bound["mode_name"]: bound["per_cta_demand_factor_upper_bound"]
               for bound in bounds
               if (bound["lane_local_rows_per_thread"],
                   bound["score_values_per_lane_local_row"]) == (4, 16)},
        },
    }
    return _write("fa2_resident_dependency_overlap_sm90.json", reference)


def spill(run_dir, specialization):
    report = _load(run_dir / "report.json")
    if (report["hardware_id"] != "nvidia_h200"
            or report["specialization"] != specialization):
        raise ValueError("spill report identity changed")
    gate = report["provenance"]["clock_gate"]
    rejected = [item for item in gate["passes"] if not item["accepted"]]
    reference = dict(report)
    yaml_key = {
        "d64_chunked": "fa2_local_spill_dependency_cycles",
        "d128_causal": "fa2_dense_d128_causal_local_spill_dependency_cycles",
    }[specialization]
    measurement = report["measurement"]
    reference["h200_provenance"] = {
        "hardware_id": "nvidia_h200",
        "device": 'NVIDIA H200 SXM (GPU 2)',
        "files": _files(
            "memory/l1/fa2_local_spill_round.cu",
            "memory/l1/fa2_local_spill_round_runner.py"),
        "git_commit_of_probe_sources": report["baseline_commit"],
        "run_id": run_dir.name,
        "local_derived_dir": str(run_dir.relative_to(REPOSITORY_ROOT)),
        "remote_raw_path": f"{run_dir.relative_to(REPOSITORY_ROOT)}/"
                           "{local,control}.csv",
        "raw_sha256": {
            "local": measurement["local_raw_sha256"],
            "control": measurement["control_raw_sha256"]},
        "report_sha256": _sha256(run_dir / "report.json"),
        "compiler": "CUDA 12.2 nvcc -arch=sm_90 -cudart static "
                    "-Xptxas=-O0 (same toolchain as the A800 sm_80 capture)",
        "clock": _clock(gate),
        "clock_gate": {
            "captures": len(gate["passes"]),
            "accepted_captures": len(gate["passes"]) - len(rejected),
            "rejected_captures_rerun": len(rejected),
            "rejected": rejected,
        },
        "measured_sm_clock_mhz_for_cycle_ns_conversion": MEASURED_SM_CLOCK_MHZ,
        "production_static_contract_status": report[
            "production_static_contract"]["status"],
        "vllm_runtime_note": (
            "these probes model the FA2 (sm80 source) program unchanged; "
            "the H200 vLLM 0.24 deployment runs FA3 for MLA attention"),
        "production_or_evaluation_duration_consumed": False,
    }
    reference["h200_yaml_values"] = {
        "yaml_key": yaml_key,
        "software_revision": "flash_attn_2_8_3_source_cuda_12_2_sm90_probe",
        "spill_round_cycles": measurement[
            "one_round_local_minus_register_control_cycles"],
        "stack_bytes_per_thread": report["probe_static_contract"][
            "stack_bytes_per_thread"],
        "scalar_store_sites": report["probe_static_contract"][
            "scalar_store_sites"],
        "scalar_load_sites": report["probe_static_contract"][
            "scalar_load_sites"],
    }
    return _write(
        f"fa2_local_spill_round_{specialization}_sm90.json", reference)


def _combine_yaml(report):
    reduction = {}
    for entry in report["reduction_entries"]:
        width = entry["allreduce_width"]
        if width not in reduction:
            reduction[width] = {"allreduce_width": width}
        name = {"fp32_max_allreduce": "fp32_max_allreduce_cycles",
                "fp32_sum_allreduce": "fp32_sum_allreduce_cycles"}[
                    entry["mechanism"]]
        reduction[width][name] = entry["cycles_per_128_thread_cta_stage"]
    conversion = [
        {"block_m": entry["block_m"], "head_dim": entry["head_dim"],
         "output_values_per_lane": entry["output_values_per_lane"],
         "fp32_to_bf16_output_conversion_cycles": entry[
             "cycles_per_128_thread_cta_stage"]}
        for entry in report["conversion_entries"]]
    names = {
        "slse_shared_store_barrier_transpose_load":
            "slse_publish_transpose_cycles",
        "post_max_expf_local_sum": "post_max_exp_local_sum_cycles",
        "post_sum_logf": "post_sum_log_cycles",
        "scale_expf_shared_store_barrier": "scale_exp_smem_barrier_cycles",
    }
    fragments = {}
    for entry in report["source_fragment_entries"]:
        if entry["mechanism"] not in names:
            continue
        key = (entry["block_m"], entry["head_dim"], entry["num_splits"],
               entry["max_splits"])
        if key not in fragments:
            fragments[key] = {
                "block_m": key[0], "head_dim": key[1],
                "num_splits": key[2], "max_splits": key[3]}
        fragments[key][names[entry["mechanism"]]] = entry[
            "cycles_per_128_thread_cta_stage"]
    order = sorted(fragments, key=lambda key: (-key[0], key[2]))
    return (
        [reduction[width] for width in sorted(reduction)], conversion,
        [fragments[key] for key in order])


def combine(a800_dir, extension_dir):
    reports = {}
    manifests = {}
    for name, run_dir in (("a800_source", a800_dir),
                          ("h200_heuristic_extension", extension_dir)):
        reports[name] = _load(run_dir / "report.json")
        manifests[name] = _load(run_dir / "manifest.json")
        if (reports[name]["hardware_id"] != "nvidia_h200"
                or manifests[name]["fragment_set"] != name):
            raise ValueError(f"combine report {name} identity changed")
    reduction, conversion, fragments = _combine_yaml(reports["a800_source"])
    _, _, extension_fragments = _combine_yaml(
        reports["h200_heuristic_extension"])
    extension_reduction, extension_conversion, _ = _combine_yaml(
        reports["h200_heuristic_extension"])
    reference = dict(reports["a800_source"])
    reference["h200_provenance"] = _provenance(
        a800_dir, _files(
            f"{FA}/fa2_combine_dependency.cu",
            f"{FA}/fa2_combine_dependency_runner.py",
            f"{FA}/fa2_combine_dependency_report.py"),
        manifests["a800_source"]["clock_gate"], raw_relative="raw.csv",
        raw_sha256=manifests["a800_source"]["raw_sha256"],
        baseline_commit=manifests["a800_source"]["baseline_commit"],
        extra={
            "report_sha256": _sha256(a800_dir / "report.json"),
            "binary_sha256": manifests["a800_source"]["binary_sha256"],
            "heuristic_extension_capture": {
                "run_id": extension_dir.name,
                "remote_raw_path":         f"{extension_dir.relative_to(REPOSITORY_ROOT)}"
                                   "/raw.csv",
                "raw_sha256": manifests["h200_heuristic_extension"][
                    "raw_sha256"],
                "report_sha256": _sha256(extension_dir / "report.json"),
                "clock": _clock(
                    manifests["h200_heuristic_extension"]["clock_gate"]),
                "clock_gate": _gate_counts(
                    manifests["h200_heuristic_extension"]["clock_gate"]),
                "source_fragment_entries": reports[
                    "h200_heuristic_extension"]["source_fragment_entries"],
                "reduction_entries_independent_repeat": reports[
                    "h200_heuristic_extension"]["reduction_entries"],
                "conversion_entries_independent_repeat": reports[
                    "h200_heuristic_extension"]["conversion_entries"],
            },
            "split_heuristic_coverage": (
                "FA2 num_splits_heuristic with 132 SMs (launch slots 264, "
                "max_splits 128) can choose every split count 1..128. "
                "Measured: the unchanged A800 table plus D128/kBlockM=4 "
                "splits 2..32 and D64/kBlockM=8 splits 2..16. Split counts "
                "above those bounds (D128 33..128, D64 17..128) are not "
                "representable by the unchanged probe (static MaxSplits <= 32 "
                "and a grid*128-float LSE buffer for mode 9); extending the "
                "probe would change the measured program beyond hardware "
                "coupling."),
        })
    reference["h200_yaml_values"] = {
        "yaml_key": "fa2_combine_dependency_cycles",
        "software_revision": "cuda_12.2_sm90",
        "reduction_entries": reduction,
        "conversion_entries": conversion,
        "source_fragment_entries": fragments,
        "heuristic_extension_source_fragment_entries": extension_fragments,
        "independent_repeat_reduction_entries": extension_reduction,
        "independent_repeat_conversion_entries": extension_conversion,
    }
    return _write("fa2_combine_dependency_sm90.json", reference)


def oaccum(run_dir):
    report = _load(run_dir / "report.json")
    manifest = _load(run_dir / "manifest.json")
    if report["hardware_id"] != "nvidia_h200":
        raise ValueError("oaccum report is not an H200 report")
    reference = dict(report)
    reference["h200_provenance"] = _provenance(
        run_dir, _files(
            "memory/lsu/fa2_oaccum_ldg128_load_use.cu",
            "memory/lsu/fa2_oaccum_ldg128_load_use_runner.py",
            "memory/lsu/fa2_oaccum_ldg128_load_use_report.py"),
        manifest["clock_gate"], raw_relative="native/raw.csv",
        raw_sha256=manifest["raw_sha256"],
        baseline_commit=manifest["baseline_commit"],
        extra={"report_sha256": _sha256(run_dir / "report.json"),
               "binary_sha256": manifest["binary_sha256"],
               "ncu_audit_status": report["ncu_cache_traffic_audit"][
                   "status"]})
    reference["h200_yaml_values"] = {
        "yaml_key": "fa2_oaccum_load_use_cycles",
        "software_revision": "cuda_12.2_sm90",
        "cache_entries": [
            {key: entry[key] for key in entry
             if key in ("cache_tier", "model", "first_load_ready_cycles",
                        "additional_load_interval_cycles",
                        "load_to_operands_ready_cycles")}
            for entry in report["cache_entries"]],
    }
    return _write("fa2_oaccum_ldg128_load_use_sm90.json", reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--resident", type=Path, required=True)
    parser.add_argument("--spill-d64", type=Path, required=True)
    parser.add_argument("--spill-d128", type=Path, required=True)
    parser.add_argument("--combine-a800", type=Path, required=True)
    parser.add_argument("--combine-extension", type=Path, required=True)
    parser.add_argument("--oaccum", type=Path, required=True)
    arguments = parser.parse_args()
    runs = {name: value.resolve() for name, value in vars(arguments).items()}
    for run_dir in runs.values():
        if OUTPUT_ROOT.resolve() not in run_dir.parents:
            raise ValueError(f"{run_dir} is not an H200 run directory")
    written = [
        scalar(runs["scalar"], runs["resident"]),
        resident(runs["resident"]),
        spill(runs["spill_d64"], "d64_chunked"),
        spill(runs["spill_d128"], "d128_causal"),
        combine(runs["combine_a800"], runs["combine_extension"]),
        oaccum(runs["oaccum"]),
    ]
    print(json.dumps([str(path.relative_to(REPOSITORY_ROOT))
                      for path in written], indent=2))


if __name__ == "__main__":
    main()
