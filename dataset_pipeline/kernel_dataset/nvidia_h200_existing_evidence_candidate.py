"""Assemble a review-only H200 dataset from pinned, already captured evidence.

This never writes the canonical dataset.  The explicit helper source is the
dataset owner's existing NCU CSV parser and sidecar schema implementation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from typing import Any

import zstandard

from dataset_pipeline.kernel_dataset.nvidia_h200 import (
    H200_FAMILIES, _entry, _flash_attn_split_attributes,
    load_campaign_plan, select_case_timing, window_vllm_stack,
)
from dataset_pipeline.profiler.nvidia_h200_ncu_plan import (
    normalized_profiler_kernel_name)
from src.kernel.dataset_storage import (
    KERNEL_DATASET_SUFFIX, compress_kernel_dataset_content,
    load_kernel_dataset_document,
)


_MISSING_NSYS_SOURCES = (
    ("FUSEDMOE_TOK16384_E8_TOPK2_H6144_INT16384", "fused_moe",
     "missing15_retry_20261031", "missing15_fused_moe_b01__v030"),
    ("FUSEDMOE_TOK32768_E8_TOPK2_H6144_INT16384", "fused_moe",
     "missing9_retry3_20261101", "missing9_fused_moe_b03__v030"),
    ("GEMM_B128_M1_K28672_N8192", "gemm",
     "missing6_retry_20261103", "missing6_gemm_06__v030"),
)
NEW_NSYS = {
    case_id: (family, tuple(
        f"{campaign}/batches/{batch}/runs/{batch}/cases/{case_id}/"
        f"window_{index:02d}.launch.json" for index in (1, 2, 3)))
    for case_id, family, campaign, batch in _MISSING_NSYS_SOURCES
}
DIAGNOSTIC_NCU_CASE = "FUSEDMOE_TOK32768_E256_TOPK8_H7168_INT2048"
PROVISIONAL_NCU_BATCHES = {
    "GEMM_B8_M4096_K28672_N8192": "ncu_directed7_r1_06_20261102",
    "GEMM_B8_M4096_K8192_N28672": "ncu_quality_retry_gemm_20261102",
}
PROVISIONAL_NCU_DIAGNOSTIC = "FUSEDMOE_TOK32768_E8_TOPK2_H6144_INT16384"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(path: Path, root: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(root)), "size": path.stat().st_size,
            "sha256": _sha(path)}


def _helper(path: Path):
    spec = importlib.util.spec_from_file_location(
        "h200_existing_publication_helper", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load H200 publication helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _historical_review_plan(campaign: Path, batch: str, review: dict) -> Path:
    """Locate the immutable captured plan without re-running admission rules."""
    name = "ncu_pilot_01" if batch == "pilot_01" else batch
    candidates = (campaign / "plans" / f"{name}.json",
                  campaign / "plans" / "candidates" / f"{name}.json")
    matches = [path for path in candidates if path.is_file() and
               _sha(path) == review["plan_sha256"]]
    if len(matches) != 1:
        raise ValueError(f"captured NCU plan pin differs: {batch}")
    plan = json.loads(matches[0].read_text())
    if plan["batch_name"] != name:
        raise ValueError(f"captured NCU batch differs: {batch}")
    return matches[0]


def _source_review(data: Path, path: Path) -> tuple[Path, dict[str, dict]]:
    relative = path.relative_to(data)
    campaign = data / relative.parts[0]
    if relative.parts[0] == "ncu_1027_20261031":
        batch = relative.parts[2]
        review_path = campaign / f"{batch}_review.json"
    elif relative.parts[0] == "missing15_retry_20261031":
        review_path = campaign / "capture_review.json"
    elif relative.parts[0] == "missing6_retry_20261103":
        review_path = campaign / "reviews" / f"{relative.parts[2]}.json"
    elif relative.parts[0] == "missing9_retry3_20261101":
        review_path = campaign / f"{relative.parts[2]}_review.json"
    else:
        raise ValueError(f"unexpected NSYS source campaign: {path}")
    review = json.loads(review_path.read_text())
    return review_path, {pin["path"]: pin for pin in review["raw_files"]}


def _check_review_pin(data: Path, path: Path, pins: dict[str, dict]) -> dict:
    campaign = data / path.relative_to(data).parts[0]
    relative = str(path.relative_to(campaign))
    if relative not in pins:
        raise ValueError(f"source review does not pin {path}")
    actual = _pin(path, campaign)
    expected = pins[relative]
    if (actual["size"] != expected["size"]
            or actual["sha256"] != expected["sha256"]):
        raise ValueError(f"source review pin changed: {path}")
    return actual


def _new_nsys_entry(data: Path, case_id: str, family: str,
                    relatives: tuple[str, ...]) -> tuple[dict, dict]:
    source = Path(relatives[0])
    source_plan_path = data / source.parts[0] / "plans" / f"{source.parts[2]}.json"
    source_plan = load_campaign_plan(source_plan_path)
    matches = [item for item in source_plan["plans"]
               if item["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"NSYS source plan is not unique: {case_id}")
    plan = matches[0]
    windows = []
    provenance = []
    signatures = []
    environments = []
    for relative in relatives:
        path = data / relative
        review_path, pins = _source_review(data, path)
        files = [path, path.with_name(path.name.replace(
            ".launch.json", ".telemetry.json")), path.with_name(
                path.name.replace(".launch.json", ".nsys-rep"))]
        checked = [_check_review_pin(data, file, pins) for file in files]
        window = json.loads(path.read_text())
        if (window["case_id"] != case_id or window["family"] != family
                or window["marker"] != plan["marker"]
                or window["runner_args"] != plan["runner_args"]
                or window["runner_relpath"] != plan["runner_relpath"]):
            raise ValueError(f"NSYS workload differs: {path}")
        if window["telemetry"] != json.loads(files[1].read_text()):
            raise ValueError(f"NSYS telemetry differs: {path}")
        signatures.append([(item["name"], item["grid"], item["block"])
                           for item in window["launches"]])
        environments.append(window["telemetry"]["environment"])
        windows.append(window)
        provenance.append({"source_review": _pin(review_path, data),
                           "raw_files": checked,
                           "clock_before_mhz": window["telemetry"][
                               "clock_before_nvtx"]["sm_clock_mhz"],
                           "clock_after_mhz": window["telemetry"][
                               "clock_after_nvtx"]["sm_clock_mhz"],
                           "total_duration_ns": sum(item["duration_ns"]
                                                    for item in window["launches"])})
    if (len(windows) != 3 or signatures[0] != signatures[1]
            or signatures[0] != signatures[2]
            or environments[0] != environments[1]
            or environments[0] != environments[2]):
        raise ValueError(f"NSYS identity differs across windows: {case_id}")
    order = sorted(range(3), key=lambda index: windows[index]["telemetry"][
        "nvtx_push_epoch_ns"])
    windows = [windows[index] for index in order]
    provenance = [provenance[index] for index in order]
    normalized = [{**window, "window": index}
                  for index, window in enumerate(windows, 1)]
    totals = [item["total_duration_ns"] for item in provenance]
    spread = (max(totals) - min(totals)) / max(totals)
    strict_clock = all(
        p["clock_before_mhz"] >= 1980 and p["clock_after_mhz"] >= 1980
        for p in provenance)
    quality_class = ("strict" if strict_clock and spread <= 0.05 else
                     "provisional_relaxed_clock_or_spread_20261103")
    selected_total = statistics.median(totals)
    selected_index = totals.index(selected_total)
    decision = {"status": "pass", "retained": normalized,
                "retained_windows": [1, 2, 3], "window_totals_ns": totals,
                "maximum_relative_difference": spread,
                "selected_window": selected_index + 1,
                "selected": normalized[selected_index]}
    if family == "flash_attn":
        attributes = _flash_attn_split_attributes(
            normalized, int(plan["runner_args"][
                plan["runner_args"].index("--batch") + 1]))
        if attributes is None:
            raise ValueError(f"FA3 split evidence is missing: {case_id}")
        decision["fa3_split"] = attributes
    entry = _entry(family, plan, decision,
                   "existing_evidence_candidate_20261103",
                   window_vllm_stack(normalized[selected_index]))
    entry["h200_capture"]["quality_class"] = quality_class
    if entry["op_info"]["evaluation"]["timing"]["total_duration_ns"] != (
            sum(item["duration_ns"] for item in normalized[
                selected_index]["launches"])):
        raise ValueError(f"NSYS selected duration differs: {case_id}")
    return entry, {"family": family, "case_id": case_id,
                   "quality_rule": quality_class,
                   "maximum_relative_difference": spread,
                   "source_plan": _pin(source_plan_path, data),
                   "selected_sources": provenance}


def _ncu_sidecars(data: Path, source: Path, case: dict, entry: dict,
                  app_id: str, helper, *, diagnostic: bool,
                  pilot: bool = False,
                  provisional_clock: bool = False) -> list[dict]:
    case_id = case["case_id"]
    command = json.loads((source / "ncu_command.json").read_text())
    validation = (None if pilot else json.loads(
        (source / "validation.json").read_text()))
    telemetry = json.loads((source / "ncu_telemetry.json").read_text())
    if (command["returncode"] != 0
            or telemetry["clock_accepted"] is not (not provisional_clock)
            or (validation is not None and not provisional_clock
                and validation["kernel_count"] != len(entry["kernels"]))):
        raise ValueError(f"NCU validation is incomplete: {case_id}")
    if pilot:
        transfer = json.loads((source / "transfer_status.json").read_text())
        if transfer["case_id"] != case_id or transfer["status"] != "complete":
            raise ValueError(f"NCU pilot transfer differs: {case_id}")
    elif diagnostic:
        expected_status = ("diagnostic_clock_rejected" if provisional_clock
                           else "diagnostic_clock_accepted")
        if validation["status"] != expected_status:
            raise ValueError(f"NCU diagnostic clock differs: {case_id}")
    else:
        transfer = json.loads((source / "transfer_status.json").read_text())
        expected_status = "ncu_clock_rejected" if provisional_clock else "complete"
        if (validation["status"] != expected_status
                or transfer["case_id"] != case_id
                or transfer["status"] != expected_status):
            raise ValueError(f"NCU transfer or validation differs: {case_id}")
    versions = command["identity"]["versions"]
    if any(versions[key] != value for key, value in (
            ("torch", "2.13.0+cu129"), ("torch_cuda", "12.9"),
            ("vllm", "0.30.0+cu129"))):
        raise ValueError(f"NCU software identity differs: {case_id}")
    if not command["identity"]["gpu"].startswith("NVIDIA H200,"):
        raise ValueError(f"NCU GPU identity differs: {case_id}")
    plan = case["plan"]
    if json.loads((source / "preflight_plan.json").read_text())["plans"] != [plan]:
        raise ValueError(f"NCU preflight plan differs: {case_id}")
    metric_names = case["traffic_metrics"] + case["service_metrics"]
    units = {name: helper._METRIC_UNITS[name] for name in metric_names}
    if set(metric_names) not in (helper._BASE_METRICS,
                                 frozenset(helper._METRIC_UNITS)):
        raise ValueError(f"NCU metrics differ: {case_id}")
    argv = command["command"]
    runner_indices = [index for index, value in enumerate(argv)
                      if value.endswith(plan["runner_relpath"])]
    if (argv[argv.index("--nvtx-include") + 1] != plan["marker"] + "/"
            or argv[argv.index("--metrics") + 1] != ",".join(metric_names)
            or len(runner_indices) != 1
            or argv[runner_indices[0] + 1:
                    runner_indices[0] + 1 + len(plan["runner_args"])] !=
            plan["runner_args"]):
        raise ValueError(f"NCU command differs: {case_id}")
    preflight = json.loads((source / "preflight/cases" / case_id /
                            "window_01.launch.json").read_text())
    if (type(preflight["telemetry"]["clock_accepted"]) is not bool
            or (not provisional_clock
                and preflight["telemetry"]["clock_accepted"] is not True)):
        raise ValueError(f"NCU preflight clock differs: {case_id}")
    observed = [[item["name"], tuple(item["grid"]), tuple(item["block"])]
                for item in preflight["launches"]]
    expected = [[item["Metadata"]["Kernel Name"],
                 helper._triplet(item["Metadata"]["Grid Size"]),
                 helper._triplet(item["Metadata"]["Block Size"])]
                for item in entry["kernels"]]
    if observed != expected:
        raise ValueError(f"NCU preflight launch differs: {case_id}")
    report = source / "profile.ncu-rep.zst"
    if not zstandard.ZstdDecompressor().decompress(
            report.read_bytes(), max_output_size=100_000_000):
        raise ValueError(f"NCU report is empty: {case_id}")
    csv_path = source / "ncu_stdout.csv"
    rows = helper._ncu_rows(csv_path, len(entry["kernels"]), units,
                            plan["marker"])
    attached = []
    for index, (row, kernel) in enumerate(zip(
            rows, entry["kernels"], strict=True)):
        meta = kernel["Metadata"]
        if (normalized_profiler_kernel_name(row["Kernel Name"]) !=
                normalized_profiler_kernel_name(meta["Kernel Name"])
                or helper._triplet(row["Grid Size"]) !=
                helper._triplet(meta["Grid Size"])
                or helper._triplet(row["Block Size"]) !=
                helper._triplet(meta["Block Size"])):
            raise ValueError(f"NCU/NSYS kernel join differs: {case_id}/{index}")
        metrics = {}
        for metric in metric_names:
            value = (int(row[metric]) if metric in helper._INTEGER_METRICS
                     else float(row[metric]))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"NCU metric is invalid: {case_id}/{metric}")
            if units[metric] == "%" and value > 100:
                raise ValueError(f"NCU percentage is invalid: {case_id}/{metric}")
            metrics[metric] = value
        if "sm__cycles_active.avg" in metrics and not (
                0 < metrics["sm__cycles_active.avg"] <=
                metrics["sm__cycles_elapsed.avg"]):
            raise ValueError(f"NCU cycles are invalid: {case_id}/{index}")
        attached.append({
            "Schema Version": 1, "Source": "ncu_raw_page", "Metrics": metrics,
            "Units": units,
            "Provenance": {
                "campaign": ("ncu_clock_diagnostic_20261102" if diagnostic
                             else "ncu_1027_20261031"),
                "case_id": case_id, "kernel_index": index,
                "ncu_kernel_name": row["Kernel Name"],
                "csv_sha256": _sha(csv_path), "report_sha256": _sha(report),
                "report_name": report.name, "app_id": app_id,
                "ncu_version": command["identity"]["profiler_versions"]["ncu"],
            },
        })
    return attached


def build_candidate(*, root: Path, canonical_dir: Path,
                    helper_path: Path, output_dir: Path) -> dict:
    """Materialize complete candidate files only inside a new output dir."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    data = root / "profiler_output/kernel_dataset/nvidia_h200"
    ncu = data / "ncu_1027_20261031"
    helper = _helper(helper_path)
    canonical = {}
    base_sha = {}
    for family in H200_FAMILIES:
        path = canonical_dir / f"{family}{KERNEL_DATASET_SUFFIX}"
        base_sha[family] = _sha(path)
        canonical[family] = load_kernel_dataset_document(path)
    merged = copy.deepcopy(canonical)
    additions = {}
    for case_id, (family, relatives) in NEW_NSYS.items():
        if case_id in merged[family]:
            raise ValueError(f"NSYS addition already exists: {case_id}")
        entry, proof = _new_nsys_entry(data, case_id, family, relatives)
        merged[family][case_id] = entry
        additions[case_id] = proof

    accepted: dict[str, tuple[Path, dict]] = {}
    for review_path in sorted(ncu.glob("*_review.json")):
        review = json.loads(review_path.read_text())
        if "statuses" not in review:
            continue
        for case_id, status in review["statuses"].items():
            if status["status"] != "complete":
                continue
            if case_id in accepted:
                old_review = accepted[case_id][1]
                stopped = (review["app"]["stopped_at"] if "app" in review
                           else review["app_stopped_at"])
                old_stopped = (old_review["app"]["stopped_at"] if "app" in old_review
                               else old_review["app_stopped_at"])
                if stopped <= old_stopped:
                    continue
            accepted[case_id] = (review_path, review)
    a800 = root / "kernel_dataset/nvidia_a800"
    if not a800.exists():
        a800 = canonical_dir.parent / "nvidia_a800"
    target = set()
    for family in H200_FAMILIES:
        target |= set(load_kernel_dataset_document(
            a800 / f"{family}{KERNEL_DATASET_SUFFIX}"))
    if set(accepted) - target:
        raise ValueError("accepted NCU review is outside A800 target")
    existing_ncu_target = {
        case_id for family in H200_FAMILIES
        for case_id, entry in canonical[family].items()
        if case_id in target and all(
            "NCU Counters" in kernel["Sections"]
            for kernel in entry["kernels"])}
    attached = {}
    held = {}
    for case_id, (review_path, review) in sorted(accepted.items()):
        batch = review_path.name.removesuffix("_review.json")
        plan_path = _historical_review_plan(ncu, batch, review)
        plan = json.loads(plan_path.read_text())
        matches = [item for item in plan["cases"]
                   if item["case_id"] == case_id]
        if len(matches) != 1:
            raise ValueError(f"NCU plan case is not unique: {case_id}")
        case = {**matches[0], "traffic_metrics": plan["traffic_metrics"],
                "service_metrics": plan["service_metrics"]}
        family = case["family"]
        if case_id not in merged[family]:
            held[case_id] = {"reason": "NSYS case is still unpublished",
                             "source_review": _pin(review_path, root)}
            continue
        entry = merged[family][case_id]
        if all("NCU Counters" in item["Sections"] for item in entry["kernels"]):
            continue
        if any("NCU Counters" in item["Sections"] for item in entry["kernels"]):
            raise ValueError(f"NCU sidecar is partial: {case_id}")
        source = (ncu / "batches" /
                  ("ncu_pilot_01" if batch == "pilot_01" else batch) /
                  "ncu_raw" / case_id)
        pins = {item["path"]: item for item in review["raw_files"]}
        required = ("ncu_stdout.csv", "profile.ncu-rep.zst", "ncu_command.json",
                    "ncu_telemetry.json", "transfer_status.json",
                    f"preflight/cases/{case_id}/window_01.launch.json")
        if batch != "pilot_01":
            required += ("validation.json",)
        raw_pins = [_check_review_pin(data, source / name, pins)
                    for name in required]
        if json.loads((source / "ncu_command.json").read_text())[
                "plan_sha256"] != _sha(plan_path):
            raise ValueError(f"NCU captured plan differs: {case_id}")
        app_id = (review["app"]["app_id"] if "app" in review
                  else review["app_id"])
        sidecars = _ncu_sidecars(
            data, source, case, entry, app_id, helper,
            diagnostic=False, pilot=batch == "pilot_01")
        for kernel, sidecar in zip(entry["kernels"], sidecars, strict=True):
            kernel["Sections"]["NCU Counters"] = sidecar
        attached[case_id] = {"family": family,
                             "source_review": _pin(review_path, root),
                             "source_plan": _pin(plan_path, root),
                             "raw_pins": raw_pins,
                             "kernel_count": len(sidecars)}

    # Earlier publication may already contain the verified E256 sidecar.
    if DIAGNOSTIC_NCU_CASE not in existing_ncu_target:
        # The E256 diagnostic was clock-qualified and passed the ordinary NCU
        # validator.  It stayed raw-only solely because its NSYS was unavailable.
        cid = DIAGNOSTIC_NCU_CASE
        family = "fused_moe"
        if cid in accepted or cid not in merged[family]:
            raise ValueError("E256 diagnostic promotion domain changed")
        diag = data / "ncu_clock_diagnostic_20261102/attempt_1_capture"
        raw_path = diag / "raw.json"
        raw = json.loads(raw_path.read_text())
        validation_path = (data / "ncu_clock_diagnostic_20261102/"
                           "standard_validation_candidate.json")
        validation = json.loads(validation_path.read_text())
        matched = [item for item in validation["cases"] if item["case_id"] == cid]
        if (len(matched) != 1
                or matched[0]["ordinary_ncu_validator_result"]["status"] != "complete"
                or raw["status"] != "diagnostic_clock_accepted"):
            raise ValueError("E256 diagnostic validator differs")
        source_plan_path = root / raw["source_plan"]["path"]
        source_plan = json.loads(source_plan_path.read_text())
        plans = [item for item in source_plan["cases"] if item["case_id"] == cid]
        if len(plans) != 1:
            raise ValueError("E256 NCU source plan differs")
        case = {**plans[0], "traffic_metrics": source_plan["traffic_metrics"],
                "service_metrics": source_plan["service_metrics"]}
        source = diag / "batches/ncu_clock_diagnostic_attempt_1_capture/ncu_raw" / cid
        raw_pins = []
        for item in raw["files"]:
            file = diag / item["path"]
            if file.stat().st_size != item["size"] or _sha(file) != item["sha256"]:
                raise ValueError(f"E256 diagnostic raw pin differs: {file}")
            raw_pins.append(item)
        receipt = json.loads((diag / "receipt.json").read_text())
        sidecars = _ncu_sidecars(
            data, source, case, merged[family][cid], receipt["app_id"], helper,
            diagnostic=True)
        for kernel, sidecar in zip(merged[family][cid]["kernels"], sidecars,
                                   strict=True):
            if "NCU Counters" in kernel["Sections"]:
                raise ValueError("E256 already has NCU counters")
            kernel["Sections"]["NCU Counters"] = sidecar
        attached[cid] = {"family": family, "quality_class":
                         "diagnostic_clock_accepted_owner_promotion_candidate",
                         "source_review": _pin(validation_path, root),
                         "source_plan": _pin(source_plan_path, root),
                         "raw_manifest": _pin(raw_path, root),
                         "raw_pins": raw_pins, "kernel_count": len(sidecars)}

    # The remaining three cases have full counter output but their recorded
    # post-range clocks missed the former 1980-MHz gate. Preserve that fact
    # while applying the same workload, launch and numeric checks as above.
    for case_id, batch in PROVISIONAL_NCU_BATCHES.items():
        review_path = ncu / f"{batch}_review.json"
        review = json.loads(review_path.read_text())
        if review["statuses"][case_id]["status"] != "ncu_clock_rejected":
            raise ValueError(f"provisional NCU review status changed: {case_id}")
        plan_path = _historical_review_plan(ncu, batch, review)
        plan = json.loads(plan_path.read_text())
        matches = [item for item in plan["cases"] if item["case_id"] == case_id]
        if len(matches) != 1 or matches[0]["family"] != "gemm":
            raise ValueError(f"provisional NCU workload changed: {case_id}")
        case = {**matches[0], "traffic_metrics": plan["traffic_metrics"],
                "service_metrics": plan["service_metrics"]}
        source = ncu / "batches" / batch / "ncu_raw" / case_id
        pins = {item["path"]: item for item in review["raw_files"]}
        names = ("ncu_stdout.csv", "profile.ncu-rep.zst", "ncu_command.json",
                 "ncu_telemetry.json", "validation.json", "transfer_status.json",
                 f"preflight/cases/{case_id}/window_01.launch.json")
        raw_pins = [_check_review_pin(data, source / name, pins)
                    for name in names]
        if json.loads((source / "ncu_command.json").read_text())[
                "plan_sha256"] != _sha(plan_path):
            raise ValueError(f"provisional NCU captured plan changed: {case_id}")
        entry = merged["gemm"][case_id]
        sidecars = _ncu_sidecars(
            data, source, case, entry, review["app"]["app_id"], helper,
            diagnostic=False, provisional_clock=True)
        for kernel, sidecar in zip(entry["kernels"], sidecars, strict=True):
            if "NCU Counters" in kernel["Sections"]:
                raise ValueError(f"provisional NCU already exists: {case_id}")
            kernel["Sections"]["NCU Counters"] = sidecar
        telemetry = json.loads((source / "ncu_telemetry.json").read_text())
        attached[case_id] = {
            "family": "gemm", "quality_class": "provisional_relaxed_clock_20261103",
            "clock_before_mhz": telemetry["clock_before_nvtx"]["sm_clock_mhz"],
            "clock_after_mhz": telemetry["clock_after_nvtx"]["sm_clock_mhz"],
            "source_review": _pin(review_path, root),
            "source_plan": _pin(plan_path, root), "raw_pins": raw_pins,
            "kernel_count": len(sidecars)}

    case_id = PROVISIONAL_NCU_DIAGNOSTIC
    diag = data / "ncu_clock_diagnostic_20261102/attempt_2_capture"
    raw_path = diag / "raw.json"
    raw = json.loads(raw_path.read_text())
    receipt_path = diag / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (raw["case_id"] != case_id
            or raw["status"] != "diagnostic_clock_rejected"
            or receipt["raw"] != _pin(raw_path, diag)):
        raise ValueError("E8 provisional NCU diagnostic identity changed")
    raw_pins = []
    for item in raw["files"]:
        file = diag / item["path"]
        if file.stat().st_size != item["size"] or _sha(file) != item["sha256"]:
            raise ValueError(f"E8 provisional NCU raw pin changed: {file}")
        raw_pins.append(item)
    source_plan_path = root / raw["source_plan"]["path"]
    if _pin(source_plan_path, root) != raw["source_plan"]:
        raise ValueError("E8 provisional NCU source plan changed")
    source_plan = json.loads(source_plan_path.read_text())
    matches = [item for item in source_plan["cases"]
               if item["case_id"] == case_id]
    if len(matches) != 1 or matches[0]["family"] != "fused_moe":
        raise ValueError("E8 provisional NCU workload changed")
    case = {**matches[0], "traffic_metrics": source_plan["traffic_metrics"],
            "service_metrics": source_plan["service_metrics"]}
    source = diag / "batches/ncu_clock_diagnostic_attempt_2_capture/ncu_raw" / case_id
    entry = merged["fused_moe"][case_id]
    sidecars = _ncu_sidecars(
        data, source, case, entry, receipt["app_id"], helper,
        diagnostic=True, provisional_clock=True)
    for kernel, sidecar in zip(entry["kernels"], sidecars, strict=True):
        if "NCU Counters" in kernel["Sections"]:
            raise ValueError("E8 provisional NCU already exists")
        kernel["Sections"]["NCU Counters"] = sidecar
    telemetry = json.loads((source / "ncu_telemetry.json").read_text())
    attached[case_id] = {
        "family": "fused_moe", "quality_class": "provisional_relaxed_clock_20261103",
        "clock_before_mhz": telemetry["clock_before_nvtx"]["sm_clock_mhz"],
        "clock_after_mhz": telemetry["clock_after_nvtx"]["sm_clock_mhz"],
        "source_review": _pin(receipt_path, root),
        "source_plan": _pin(source_plan_path, root),
        "raw_manifest": _pin(raw_path, root), "raw_pins": raw_pins,
        "kernel_count": len(sidecars)}

    # Preserve every canonical timing and entry. Only counters are added to
    # existing records; the three new case IDs carry new NSYS timing.
    for family in H200_FAMILIES:
        for case_id, before in canonical[family].items():
            after = merged[family][case_id]
            if before["op_info"] != after["op_info"]:
                raise ValueError(f"existing NSYS timing changed: {case_id}")
            for old_kernel, new_kernel in zip(before["kernels"],
                                               after["kernels"], strict=True):
                old_sections = old_kernel["Sections"]
                if ({key: value for key, value in new_kernel["Sections"].items()
                     if key != "NCU Counters"} !=
                        {key: value for key, value in old_sections.items()
                         if key != "NCU Counters"}):
                    raise ValueError(f"existing kernel changed: {case_id}")
        json.dumps(merged[family], ensure_ascii=False, allow_nan=False)
    paired_target = {
        case_id for family in H200_FAMILIES
        for case_id, entry in merged[family].items()
        if case_id in target and all(
            "NCU Counters" in kernel["Sections"]
            for kernel in entry["kernels"])}
    timed_target = target & {
        case_id for family in H200_FAMILIES for case_id in merged[family]}
    if timed_target != target:
        raise ValueError(f"H200 target timing remains incomplete: {sorted(target - timed_target)}")
    output_dir.mkdir(parents=True)
    candidate_sha = {}
    case_counts = {}
    for family in H200_FAMILIES:
        path = output_dir / f"{family}{KERNEL_DATASET_SUFFIX}"
        encoded = json.dumps(
            {case_id: merged[family][case_id] for case_id in sorted(merged[family])},
            indent=2, ensure_ascii=False, allow_nan=False).encode()
        with path.open("xb") as stream:
            stream.write(compress_kernel_dataset_content(encoded))
        candidate_sha[family] = _sha(path)
        case_counts[family] = len(merged[family])
    manifest = {"schema_version": 1,
                "kind": "h200_existing_evidence_four_family_candidate",
                "status": "worker_candidate_not_published",
                "canonical_dir": str(canonical_dir),
                "canonical_family_sha256": base_sha,
                "publication_helper_source": _pin(helper_path, root)
                if helper_path.is_relative_to(root) else {
                    "path": str(helper_path), "size": helper_path.stat().st_size,
                    "sha256": _sha(helper_path)},
                "case_counts": case_counts,
                "candidate_family_sha256": candidate_sha,
                "new_nsys_sources": additions,
                "ncu_attached_sources": attached,
                "ncu_held_for_missing_nsys": held,
                "ncu_unpaired_target_cases": sorted(target - paired_target),
                "ncu_provisional_clock_cases": sorted(
                    case_id for case_id, proof in attached.items()
                    if "quality_class" in proof and proof["quality_class"] ==
                    "provisional_relaxed_clock_20261103"),
                "nsys_relaxed_included": sorted(
                    case_id for case_id, proof in additions.items()
                    if proof["quality_rule"] != "strict"),
                "target_case_count": len(target),
                "target_timing_count": len(timed_target),
                "ncu_target_available_count": len(paired_target),
                "ncu_target_paired_count": len(paired_target),
                "ncu_attached_case_count": len(attached),
                "ncu_held_case_count": len(held),
                "no_modal_execution": True}
    path = output_dir / "manifest.json"
    with path.open("x") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False,
                  sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest
