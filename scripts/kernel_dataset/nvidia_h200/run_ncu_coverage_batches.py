"""Run explicitly named H200 NCU batches one at a time under a spend cap."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.profiler.nvidia_h200_ncu_plan import (  # noqa: E402
    admitted_batch_plan,
    expected_launch_signature,
    validate_missing15_local_case,
    verify_missing15_local_evidence,
    verify_quality_retry_local_evidence,
)

CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_1027_20261031"
RUNNER = ROOT / "scripts/kernel_dataset/nvidia_h200/ncu_coverage_modal_runner.py"
NSYS_CAMPAIGN = (ROOT / "profiler_output/kernel_dataset/nvidia_h200/"
                 "missing6_retry_20261103")
NSYS_RUNNER = ROOT / "scripts/kernel_dataset/nvidia_h200/missing39_modal_runner.py"
NSYS_REVIEW_RUNNER = (ROOT / "scripts/kernel_dataset/nvidia_h200/"
                      "review_missing6_nsys.py")
NSYS_REVIEW_PYTHON = Path(sys.executable)
LOGS = ROOT / "experiment_logs/h200_accuracy_20261028/ncu_batch_runs"
LEDGER = CAMPAIGN / "task_spend_ledger.json"
PROBE_RUNNERS = {
    "scheduler_phase": (
        ROOT / "microbench/nvidia_h200/runtime/scheduler_host/modal_async_scheduler_phase_runner.py",
        ROOT / "profiler_output/microbench/nvidia_h200/modal_async_scheduler_phase_20261102"),
    "scheduler_phase_mixed": (
        ROOT / "microbench/nvidia_h200/runtime/scheduler_host/modal_async_scheduler_phase_mixed_runner.py",
        ROOT / "profiler_output/microbench/nvidia_h200/modal_async_scheduler_phase_20261102"),
    "fa3_barrier": (
        ROOT / "microbench/nvidia_h200/operators/flash_attn/fa3_wg_scheduler_barrier_modal_runner.py",
        ROOT / "profiler_output/microbench/nvidia_h200/fa3_wg_scheduler_barrier_20261102"),
    "gemm_descriptor": (
        ROOT / "microbench/nvidia_h200/operators/gemm/cublas_descriptor_modal_runner.py",
        ROOT / "profiler_output/microbench/nvidia_h200/cublas_descriptor_20261102"),
    "request_path_030": (
        ROOT / "microbench/nvidia_h200/runtime/host_request_path/modal_vllm030_render_runner.py",
        ROOT / "profiler_output/microbench/nvidia_h200/vllm030_request_path_20261102"),
    "ncu_clock_diagnostic": (
        ROOT / "scripts/kernel_dataset/nvidia_h200/ncu_coverage_modal_runner.py",
        ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_clock_diagnostic_20261102"),
    "w2_panel_footprint": (
        ROOT / "microbench/nvidia_h200/compute/tensor_core/routed_panel_wgmma_footprint_modal_runner.py",
        ROOT / "profiler_output/microbench/nvidia_h200/routed_panel_wgmma_footprint_20261102"),
    "w2_panel_grouped": (
        ROOT / "microbench/nvidia_h200/compute/tensor_core/routed_panel_wgmma_grouped_modal_runner.py",
        ROOT / "profiler_output/microbench/nvidia_h200/routed_panel_wgmma_grouped_20261102"),
    "w2_panel_grouped_timing": (
        ROOT / "microbench/nvidia_h200/compute/tensor_core/routed_panel_wgmma_grouped_timing_modal_runner.py",
        ROOT / "profiler_output/microbench/nvidia_h200/routed_panel_wgmma_grouped_timing_20261102"),
    "w2_triton_dependency": (
        ROOT / "microbench/nvidia_h200/compute/tensor_core/routed_panel_triton_dependency_modal_runner.py",
        ROOT / "profiler_output/microbench/nvidia_h200/routed_panel_triton_dependency_20261102"),
}
PROBE_APP_NAMES = {
    "scheduler_phase": "midlayer-h200-async-scheduler-phase-20261102",
    "scheduler_phase_mixed": "midlayer-h200-async-scheduler-phase-20261102",
    "fa3_barrier": "midlayer-h200-fa3-wg-scheduler-barrier-20261102",
    "gemm_descriptor": "midlayer-h200-gemm-descriptor-20261102",
    "request_path_030": "midlayer-h200-vllm030-request-path-20261102",
    "ncu_clock_diagnostic": "midlayer-h200-ncu-coverage-20261031",
    "w2_panel_footprint": "midlayer-h200-routed-panel-wgmma-footprint-20261102",
    "w2_panel_grouped": "midlayer-h200-routed-panel-wgmma-grouped-20261102",
    "w2_panel_grouped_timing": "midlayer-h200-routed-panel-wgmma-grouped-timing-20261102",
    "w2_triton_dependency": "midlayer-h200-routed-panel-triton-dependency-20261102",
}
PROBE_GATE_ENV = "MIDLAYER_H200_PAID_PROBE_CONTROLLER"
CONTROLLER_TIMEOUT_SECONDS = Decimal(900)


def _modal_json(modal_cli: Path, *arguments: str):
    result = subprocess.run(
        [str(modal_cli), *arguments], cwd=ROOT, check=True,
        capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


_CAMPAIGN_APP_NAMES = {
    "midlayer-h200-missing15-nsys-20261031",
    "midlayer-h200-ncu-coverage-20261031",
    "midlayer-h200-async-scheduler-phase-20261102",
    "midlayer-h200-fa3-wg-scheduler-barrier-20261102",
    "midlayer-h200-gemm-descriptor-20261102",
    "midlayer-h200-vllm030-request-path-20261102",
    "midlayer-h200-routed-panel-wgmma-footprint-20261102",
    "midlayer-h200-routed-panel-wgmma-grouped-20261102",
    "midlayer-h200-routed-panel-wgmma-grouped-timing-20261102",
    "midlayer-h200-routed-panel-triton-dependency-20261102",
}


def _historical_app_ids() -> set[str]:
    missing = json.loads((ROOT / "profiler_output/kernel_dataset/nvidia_h200/"
                          "missing15_retry_20261031/capture_review.json").read_text())
    ids = {batch["app_id"] for batch in missing["batches"]}
    for path in CAMPAIGN.glob("*_review.json"):
        review = json.loads(path.read_text())
        app = review["app"] if "app" in review else review
        ids.add(app["app_id"])
    for campaign in ("missing12_retry2_20261101", "missing9_retry3_20261101"):
        retry = ROOT / "profiler_output/kernel_dataset/nvidia_h200" / campaign
        for path in retry.glob("*_review.json"):
            ids.add(json.loads(path.read_text())["app_id"])
    isolated = (ROOT / "profiler_output/kernel_dataset/nvidia_h200/"
                "missing6_gemm_isolated_20261102")
    if isolated.exists():
        receipt_path = isolated / "app_receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"isolated NSYS campaign lacks app receipt: {receipt_path}")
        app_id = json.loads(receipt_path.read_text())["app"]["app_id"]
        if re.fullmatch(r"ap-[A-Za-z0-9]+", app_id) is None:
            raise ValueError(f"invalid isolated NSYS app ID: {app_id}")
        ids.add(app_id)
    for path in LOGS.glob("*.log"):
        ids.update(re.findall(r"/((?:ap-)[A-Za-z0-9]+)", path.read_text()))
    for path in (CAMPAIGN / "batches").glob("*/failed_transfer_receipt.json"):
        app = json.loads(path.read_text())["app"]
        if app is not None:
            ids.add(app["app_id"])
    for path in (NSYS_CAMPAIGN / "receipts").glob("*.json"):
        app = json.loads(path.read_text())["app"]
        if app is not None:
            ids.add(app["app_id"])
    phase_attempts = (ROOT / "profiler_output/microbench/nvidia_h200/"
                      "modal_async_scheduler_phase_20261102")
    for attempt in phase_attempts.glob("attempt_*"):
        if not attempt.is_dir():
            raise ValueError(f"Scheduler phase attempt is not a directory: {attempt}")
        receipt = attempt / "receipt.json"
        if not receipt.is_file():
            raise FileNotFoundError(
                f"Scheduler phase attempt lacks an app receipt: {attempt}")
        ids.add(json.loads(receipt.read_text())["app_id"])
    barrier = (ROOT / "profiler_output/microbench/nvidia_h200/"
               "fa3_wg_scheduler_barrier_20261102")
    first = barrier / "attempt_01_failure_receipt.json"
    if barrier.exists():
        if not first.is_file():
            raise FileNotFoundError(f"FA3 probe lacks first app receipt: {first}")
        ids.add(json.loads(first.read_text())["app_id"])
    for attempt in barrier.glob("attempt_*"):
        if not attempt.is_dir():
            continue
        receipt = attempt / "receipt.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"FA3 probe attempt lacks app receipt: {attempt}")
        ids.add(json.loads(receipt.read_text())["app_id"])
    descriptor = (ROOT / "profiler_output/microbench/nvidia_h200/"
                  "cublas_descriptor_20261102")
    for attempt in descriptor.glob("attempt_*"):
        if not attempt.is_dir():
            raise ValueError(f"GEMM descriptor attempt is not a directory: {attempt}")
        receipt = attempt / "receipt.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"GEMM descriptor attempt lacks receipt: {attempt}")
        ids.add(json.loads(receipt.read_text())["app_id"])
    request_path = (ROOT / "profiler_output/microbench/nvidia_h200/"
                    "vllm030_request_path_20261102")
    for attempt in request_path.glob("attempt_*"):
        if not attempt.is_dir():
            raise ValueError(f"request-path attempt is not a directory: {attempt}")
        receipt = attempt / "receipt.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"request-path attempt lacks receipt: {attempt}")
        ids.add(json.loads(receipt.read_text())["app_id"])
    clock_diagnostic = (ROOT / "profiler_output/kernel_dataset/nvidia_h200/"
                        "ncu_clock_diagnostic_20261102")
    for attempt in clock_diagnostic.glob("attempt_*"):
        if not attempt.is_dir():
            raise ValueError(f"NCU diagnostic attempt is not a directory: {attempt}")
        receipt = attempt / "receipt.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"NCU diagnostic attempt lacks receipt: {attempt}")
        ids.add(json.loads(receipt.read_text())["app_id"])
    for label, campaign_name in (
            ("W2 footprint", "routed_panel_wgmma_footprint_20261102"),
            ("W2 grouped", "routed_panel_wgmma_grouped_20261102"),
            ("W2 grouped timing", "routed_panel_wgmma_grouped_timing_20261102")):
        campaign = (ROOT / "profiler_output/microbench/nvidia_h200/"
                    / campaign_name)
        for attempt in campaign.glob("attempt_*"):
            if not attempt.is_dir():
                raise ValueError(f"{label} attempt is not a directory: {attempt}")
            receipt = attempt / "receipt.json"
            if not receipt.is_file():
                raise FileNotFoundError(f"{label} attempt lacks receipt: {attempt}")
            ids.add(json.loads(receipt.read_text())["app_id"])
    dependency = (ROOT / "profiler_output/microbench/nvidia_h200/"
                  "routed_panel_triton_dependency_20261102")
    for attempt in dependency.glob("attempt_*"):
        if not attempt.is_dir():
            raise ValueError(f"W2 Triton dependency attempt is not a directory: {attempt}")
        receipt = attempt / "receipt.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"W2 Triton dependency attempt lacks receipt: {attempt}")
        ids.add(json.loads(receipt.read_text())["app_id"])
    return ids


def _pinned_file(parent: Path, pin: dict) -> None:
    path = parent / pin["path"]
    if (not path.is_file() or path.stat().st_size != pin["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]):
        raise ValueError(f"probe evidence pin differs: {path}")


def _verify_isolated_nsys_attempt(ledger: dict) -> None:
    campaign = (ROOT / "profiler_output/kernel_dataset/nvidia_h200/"
                "missing6_gemm_isolated_20261102")
    if not campaign.exists():
        return
    receipt = json.loads((campaign / "app_receipt.json").read_text())
    review = json.loads((campaign / "capture_review.json").read_text())
    manifest = json.loads((campaign / "plan_manifest.json").read_text())
    app = receipt["app"]
    app_id = app["app_id"]
    if (review["app_id"] != app_id or receipt["returncode"] != 0
            or review["case_status"] != "reproducibility_exceeds_limit"
            or review["canonical_publishable"] is not False
            or receipt["gpu"] != "H200" or receipt["gpu_count"] != 1
            or receipt["max_attempts"] != 1
            or int(receipt["after"]["app_count"])
            < int(receipt["before"]["app_count"]) + 1):
        raise ValueError("isolated NSYS receipt or review identity differs")
    batch = receipt["batch"]
    entry = manifest["batches"][batch]
    plan = campaign / "plans" / Path(entry["plan_path"]).name
    if (receipt["plan_sha256"] != entry["plan_sha256"]
            or review["plan_sha256"] != entry["plan_sha256"]
            or hashlib.sha256(plan.read_bytes()).hexdigest() != entry["plan_sha256"]
            or receipt["manifest_sha256"] != review["manifest_sha256"]
            or receipt["manifest_sha256"] != hashlib.sha256(
                (campaign / "plan_manifest.json").read_bytes()).hexdigest()):
        raise ValueError("isolated NSYS plan or manifest pin differs")
    _pinned_file(campaign, {"path": "modal_v2.log",
                            "sha256": receipt["modal_log_sha256"],
                            "size": receipt["modal_log_size"]})
    pins = review["raw_files"]
    paths = [pin["path"] for pin in pins]
    actual = {str(path.relative_to(campaign)) for path in
              (campaign / "batches").rglob("*") if path.is_file()}
    if (len(paths) != len(set(paths)) or set(paths) != actual
            or any(Path(path).is_absolute() or ".." in Path(path).parts
                   or not Path(path).parts or Path(path).parts[0] != "batches"
                   for path in paths)):
        raise ValueError("isolated NSYS raw pin set differs")
    for pin in pins:
        _pinned_file(campaign, pin)
    started = datetime.fromisoformat(app["created_at"])
    stopped = datetime.fromisoformat(app["stopped_at"])
    if (Decimal(str((stopped - started).total_seconds())) !=
            Decimal(str(app["conservative_app_lifecycle_gpu_seconds"]))):
        raise ValueError("isolated NSYS app lifecycle differs")
    if app_id in ledger["apps"]:
        old = ledger["apps"][app_id]
        if (old["created_at"] != app["created_at"]
                or old["stopped_at"] != app["stopped_at"]
                or old["description"] != "midlayer-h200-missing15-nsys-20261031"):
            raise ValueError("isolated NSYS ledger identity differs")


def _verify_phase_attempts(ledger: dict) -> None:
    """Bind each existing phase receipt to local Modal identity and pinned raw."""

    root = (ROOT / "profiler_output/microbench/nvidia_h200/"
            "modal_async_scheduler_phase_20261102")
    for attempt in root.glob("attempt_*"):
        if not attempt.is_dir():
            raise ValueError(f"Scheduler phase attempt is not a directory: {attempt}")
        receipt = json.loads((attempt / "receipt.json").read_text())
        info_path = attempt / "app_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Scheduler phase attempt lacks app info: {attempt}")
        info = json.loads(info_path.read_text())
        app_id = receipt["app_id"]
        lifecycle = info["lifecycle"]
        if (info["app_id"] != app_id
                or info["description"] != "midlayer-h200-async-scheduler-phase-20261102"
                or lifecycle["state"] != "stopped"
                or lifecycle["stopped_at"] is None
                or info["functions"]["capture"]["summary"]["gpu_config"]
                != [{"count": 1, "gpu_type": "H200"}]):
            raise ValueError(f"Scheduler phase app identity differs: {attempt}")
        for key in ("raw", "log", "app_info"):
            if key in receipt:
                _pinned_file(attempt, receipt[key])
        if "app" in receipt:
            for key in ("description", "created_at", "stopped_at", "lifecycle_seconds"):
                value = (str((datetime.fromisoformat(lifecycle["stopped_at"])
                              - datetime.fromisoformat(lifecycle["created_at"])).total_seconds())
                         if key == "lifecycle_seconds" else
                         (info["description"] if key == "description" else lifecycle[key]))
                if receipt["app"][key] != value:
                    raise ValueError(f"Scheduler phase receipt differs: {attempt} {key}")
        for key in ("created_at", "stopped_at"):
            if key in receipt and receipt[key] != lifecycle[key]:
                raise ValueError(f"Scheduler phase receipt differs: {attempt} {key}")
        if app_id in ledger["apps"]:
            old = ledger["apps"][app_id]
            for key in ("created_at", "stopped_at"):
                if old[key] != lifecycle[key]:
                    raise ValueError(f"Scheduler phase ledger differs: {attempt} {key}")
            if old["description"] != info["description"]:
                raise ValueError(f"Scheduler phase ledger description differs: {attempt}")
    barrier = (ROOT / "profiler_output/microbench/nvidia_h200/"
               "fa3_wg_scheduler_barrier_20261102")
    if barrier.exists():
        evidence = [(barrier, barrier / "attempt_01_failure_receipt.json")]
        evidence.extend((path, path / "receipt.json") for path in barrier.glob("attempt_*")
                        if path.is_dir())
        for parent, receipt_path in evidence:
            receipt = json.loads(receipt_path.read_text())
            info_path = parent / "app_info.json"
            if not info_path.is_file():
                raise FileNotFoundError(f"FA3 probe lacks app info: {parent}")
            info = json.loads(info_path.read_text())
            app_id = receipt["app_id"]
            lifecycle = info["lifecycle"]
            if (info["app_id"] != app_id
                    or info["description"] != "midlayer-h200-fa3-wg-scheduler-barrier-20261102"
                    or lifecycle["state"] != "stopped"
                    or lifecycle["stopped_at"] is None
                    or info["functions"]["capture"]["summary"]["gpu_config"]
                    != [{"count": 1, "gpu_type": "H200"}]):
                raise ValueError(f"FA3 probe app identity differs: {parent}")
            for key in ("raw", "log", "app_info"):
                if key in receipt:
                    _pinned_file(parent, receipt[key])
            for filename, size_key, sha_key in (("raw.json", "raw_size", "raw_sha256"),
                                                 ("app_info.json", "app_info_size", "app_info_sha256")):
                if size_key in receipt:
                    path = parent / filename
                    if (not path.is_file() or path.stat().st_size != receipt[size_key]
                            or hashlib.sha256(path.read_bytes()).hexdigest()
                            != receipt[sha_key]):
                        raise ValueError(f"FA3 probe evidence pin differs: {path}")
            if app_id in ledger["apps"]:
                old = ledger["apps"][app_id]
                if (old["created_at"] != lifecycle["created_at"]
                        or old["stopped_at"] != lifecycle["stopped_at"]
                        or old["description"] != info["description"]):
                    raise ValueError(f"FA3 probe ledger differs: {parent}")
    descriptor = (ROOT / "profiler_output/microbench/nvidia_h200/"
                  "cublas_descriptor_20261102")
    for attempt in descriptor.glob("attempt_*"):
        receipt = json.loads((attempt / "receipt.json").read_text())
        info = json.loads((attempt / "app_info.json").read_text())
        lifecycle = info["lifecycle"]
        app_id = receipt["app_id"]
        if (info["app_id"] != app_id
                or info["description"] != PROBE_APP_NAMES["gemm_descriptor"]
                or lifecycle["state"] != "stopped"
                or lifecycle["stopped_at"] is None
                or info["functions"]["capture"]["summary"]["gpu_config"]
                != [{"count": 1, "gpu_type": "H200"}]):
            raise ValueError(f"GEMM descriptor app identity differs: {attempt}")
        for key in ("raw", "log", "app_info"):
            if key in receipt:
                _pinned_file(attempt, receipt[key])
        if receipt["modal_returncode"] == 0 and "raw" not in receipt:
            raise ValueError(f"successful GEMM descriptor probe lacks raw: {attempt}")
        if "raw" in receipt:
            raw = json.loads((attempt / "raw.json").read_text())
            if (raw["kind"] != "h200_cublas_descriptor_probe_raw"
                    or raw["shapes_mkn"] != [[768, 4096, 2560],
                                               [768, 9728, 2560]]
                    or len(raw["files"]) != 1
                    or raw["files"][0]["path"] != "descriptor.nsys-rep.zst"):
                raise ValueError(f"GEMM descriptor raw identity differs: {attempt}")
            _pinned_file(attempt, raw["files"][0])
        if app_id in ledger["apps"]:
            old = ledger["apps"][app_id]
            if (old["created_at"] != lifecycle["created_at"]
                    or old["stopped_at"] != lifecycle["stopped_at"]
                    or old["description"] != info["description"]):
                raise ValueError(f"GEMM descriptor ledger differs: {attempt}")
    request_path = (ROOT / "profiler_output/microbench/nvidia_h200/"
                    "vllm030_request_path_20261102")
    for attempt in request_path.glob("attempt_*"):
        receipt = json.loads((attempt / "receipt.json").read_text())
        info = json.loads((attempt / "app_info.json").read_text())
        lifecycle = info["lifecycle"]
        app_id = receipt["app_id"]
        if (receipt["probe_kind"] != "request_path_030"
                or info["app_id"] != app_id
                or info["description"] != PROBE_APP_NAMES["request_path_030"]
                or lifecycle["state"] != "stopped"
                or lifecycle["stopped_at"] is None
                or info["functions"]["capture"]["summary"]["gpu_config"]
                != [{"count": 1, "gpu_type": "H200"}]):
            raise ValueError(f"request-path app identity differs: {attempt}")
        for key in ("raw", "log", "app_info"):
            if key in receipt:
                _pinned_file(attempt, receipt[key])
        if receipt["modal_returncode"] == 0:
            if "raw" not in receipt:
                raise ValueError(f"successful request-path probe lacks raw: {attempt}")
            raw = json.loads((attempt / "raw.json").read_text())
            measurement = raw["measurement"]
            input_manifest = request_path / "source_inputs/inputs_manifest.json"
            if (raw["status"] != "complete"
                    or raw["command_returncode"] != 0
                    or not raw["gpu_query"]["stdout"].startswith("NVIDIA H200,")
                    or measurement["kind"] != "vllm030_host_only_completion_renderer_service"
                    or measurement["software"]["vllm"].split("+", 1)[0] != "0.30.0"
                    or measurement["model_weights_loaded"] is not False
                    or measurement["production_or_evaluation_duration_consumed"] is not False
                    or measurement["local_source_preflight"] is not False
                    or measurement["renderer"]["class"] != "HfRenderer"
                    or [row["prompt_tokens"] for row in measurement["rows"]]
                    != [32, 128, 729, 2048]
                    or not input_manifest.is_file()
                    or hashlib.sha256(input_manifest.read_bytes()).hexdigest()
                    != raw["input_manifest_sha256"]
                    or measurement["input_manifest_sha256"]
                    != raw["input_manifest_sha256"]):
                raise ValueError(f"request-path raw identity differs: {attempt}")
        if app_id in ledger["apps"]:
            old = ledger["apps"][app_id]
            if (old["created_at"] != lifecycle["created_at"]
                    or old["stopped_at"] != lifecycle["stopped_at"]
                    or old["description"] != info["description"]):
                raise ValueError(f"request-path ledger differs: {attempt}")
    clock_diagnostic = (ROOT / "profiler_output/kernel_dataset/nvidia_h200/"
                        "ncu_clock_diagnostic_20261102")
    for attempt in clock_diagnostic.glob("attempt_*"):
        receipt = json.loads((attempt / "receipt.json").read_text())
        info = json.loads((attempt / "app_info.json").read_text())
        lifecycle = info["lifecycle"]
        app_id = receipt["app_id"]
        if (receipt["probe_kind"] != "ncu_clock_diagnostic"
                or info["app_id"] != app_id
                or info["description"] != PROBE_APP_NAMES["ncu_clock_diagnostic"]
                or lifecycle["state"] != "stopped"
                or lifecycle["stopped_at"] is None
                or info["functions"]["capture"]["summary"]["gpu_config"]
                != [{"count": 1, "gpu_type": "H200"}]):
            raise ValueError(f"NCU diagnostic app identity differs: {attempt}")
        for key in ("raw", "log", "app_info"):
            if key in receipt:
                _pinned_file(attempt, receipt[key])
        if receipt["modal_returncode"] == 0 and "raw" not in receipt:
            raise ValueError(f"successful NCU diagnostic lacks raw: {attempt}")
        if "raw" in receipt:
            raw = json.loads((attempt / "raw.json").read_text())
            if raw["schema_version"] == 1 and attempt.name == "attempt_1_capture":
                plan_path = clock_diagnostic / "plan.json"
            elif (raw["schema_version"] == 2
                  and raw["plan_path"] == f"plans/{attempt.name}.json"):
                plan_path = clock_diagnostic / raw["plan_path"]
            else:
                raise ValueError(f"NCU diagnostic plan identity differs: {attempt}")
            if (raw["kind"] != "h200_ncu_clock_diagnostic_raw_only"
                    or raw["quality_accepted"] is not False
                    or raw["production_duration_consumed"] is not False
                    or raw["status"] not in (
                        "diagnostic_clock_rejected", "diagnostic_clock_accepted")
                    or hashlib.sha256(plan_path.read_bytes()).hexdigest()
                    != raw["plan_sha256"]):
                raise ValueError(f"NCU diagnostic raw identity differs: {attempt}")
            for pin in raw["files"]:
                _pinned_file(attempt, pin)
        if app_id in ledger["apps"]:
            old = ledger["apps"][app_id]
            if (old["created_at"] != lifecycle["created_at"]
                    or old["stopped_at"] != lifecycle["stopped_at"]
                    or old["description"] != info["description"]):
                raise ValueError(f"NCU diagnostic ledger differs: {attempt}")
    for probe_kind, campaign_name, boundary in (
            ("w2_panel_footprint", "routed_panel_wgmma_footprint_20261102",
             "short_k_large_backing_gather_cp_async_wgmma_no_moe_output"),
            ("w2_panel_grouped", "routed_panel_wgmma_grouped_20261102",
             "four_k_rounds_source_grouped_cp_async_wgmma_no_moe_output"),
            ("w2_panel_grouped_timing", "routed_panel_wgmma_grouped_timing_20261102",
             "four_eight_k_rounds_source_grouped_cta_phase_timing_no_moe_output")):
        campaign = (ROOT / "profiler_output/microbench/nvidia_h200/"
                    / campaign_name)
        for attempt in campaign.glob("attempt_*"):
            receipt = json.loads((attempt / "receipt.json").read_text())
            info = json.loads((attempt / "app_info.json").read_text())
            lifecycle = info["lifecycle"]
            app_id = receipt["app_id"]
            if (receipt["probe_kind"] != probe_kind
                    or info["app_id"] != app_id
                    or info["description"] != PROBE_APP_NAMES[probe_kind]
                    or lifecycle["state"] != "stopped"
                    or lifecycle["stopped_at"] is None
                    or info["functions"]["capture"]["summary"]["gpu_config"]
                    != [{"count": 1, "gpu_type": "H200"}]):
                raise ValueError(f"W2 panel app identity differs: {attempt}")
            for key in ("raw", "log", "app_info"):
                if key in receipt:
                    _pinned_file(attempt, receipt[key])
            if "raw" in receipt:
                raw = json.loads((attempt / "raw.json").read_text())
                if raw["measurement_boundary"] != boundary:
                    raise ValueError(f"W2 panel raw identity differs: {attempt}")
                if raw["status"] == "complete":
                    if (raw["sass_boundary"]["valid"] is not True
                            or raw["execution"]["exit_code"] != 0
                            or set(raw["counters"]) != {
                                "gather_copy", "gather_plus_wgmma"}
                            or any(row["exit_code"] != 0
                                   for row in raw["counters"].values())):
                        raise ValueError(f"W2 panel complete raw differs: {attempt}")
                elif raw["status"] not in (
                        "environment_failed", "generation_failed", "build_failed",
                        "sass_failed", "sass_boundary_failed", "execution_failed",
                        "counter_failed"):
                    raise ValueError(f"W2 panel failure status differs: {attempt}")
            if app_id in ledger["apps"]:
                old = ledger["apps"][app_id]
                if (old["created_at"] != lifecycle["created_at"]
                        or old["stopped_at"] != lifecycle["stopped_at"]
                        or old["description"] != info["description"]):
                    raise ValueError(f"W2 panel ledger differs: {attempt}")
    dependency = (ROOT / "profiler_output/microbench/nvidia_h200/"
                  "routed_panel_triton_dependency_20261102")
    for attempt in dependency.glob("attempt_*"):
        receipt = json.loads((attempt / "receipt.json").read_text())
        info = json.loads((attempt / "app_info.json").read_text())
        lifecycle = info["lifecycle"]
        app_id = receipt["app_id"]
        if (receipt["probe_kind"] != "w2_triton_dependency"
                or info["app_id"] != app_id
                or info["description"] != PROBE_APP_NAMES["w2_triton_dependency"]
                or lifecycle["state"] != "stopped"
                or lifecycle["stopped_at"] is None
                or receipt["created_at"] != lifecycle["created_at"]
                or receipt["stopped_at"] != lifecycle["stopped_at"]
                or info["functions"]["capture"]["summary"]["gpu_config"]
                != [{"count": 1, "gpu_type": "H200"}]):
            raise ValueError(f"W2 Triton dependency app identity differs: {attempt}")
        for key in ("log", "app_info"):
            _pinned_file(attempt, receipt[key])
        if "raw" in receipt:
            _pinned_file(attempt, receipt["raw"])
            raw = json.loads((attempt / "raw.json").read_text())
            if (raw["measurement_boundary"] !=
                    "synthetic_grouped_bf16_kloop_triton371_no_moe_output"
                    or raw["source_config"]["num_warps"] != 8
                    or raw["source_config"]["num_stages"] != 4
                    or raw["source_config"]["swap_ab"] is not False
                    or raw["source_config"]["use_td"] is not False
                    or raw["status"] not in (
                        "environment_failed", "capture_failed",
                        "compile_mismatch", "counter_failed", "complete")):
                raise ValueError(f"W2 Triton dependency raw identity differs: {attempt}")
            if raw["status"] in ("compile_mismatch", "counter_failed", "complete"):
                if (set(raw["compiled"]) != {"runtime", "4", "8", "12"}
                        or raw["software"]["triton"] != "3.7.1"
                        or raw["local_source_sha256"] != raw["source_sha256"]):
                    raise ValueError(f"W2 Triton dependency compiler evidence differs: {attempt}")
            elif ("source_sha256" in raw and
                  raw["local_source_sha256"] != raw["source_sha256"]):
                raise ValueError(f"W2 Triton dependency source identity differs: {attempt}")
        elif receipt["modal_returncode"] == 0:
            raise ValueError(f"successful W2 Triton dependency lacks raw: {attempt}")
        if app_id in ledger["apps"]:
            old = ledger["apps"][app_id]
            if (old["created_at"] != lifecycle["created_at"]
                    or old["stopped_at"] != lifecycle["stopped_at"]
                    or old["description"] != info["description"]):
                raise ValueError(f"W2 Triton dependency ledger differs: {attempt}")


def _write_ledger(ledger: dict) -> None:
    target = LEDGER.with_suffix(".json.tmp")
    with target.open("x") as stream:
        json.dump(ledger, stream, indent=2, sort_keys=True)
        stream.write("\n")
    target.replace(LEDGER)


def _record_failed_batch(name: str, plan_path: Path, app: dict | None,
                         before: dict, after: dict, returncode: int,
                         output: str) -> Path:
    """Pin any returned raw and the paid app after a broken client transfer."""

    batch = CAMPAIGN / "batches" / name
    batch.mkdir(parents=True, exist_ok=True)
    receipt_path = batch / "failed_transfer_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    files = []
    for path in sorted(batch.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(batch)),
                          "size": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    log = LOGS / f"{name}.log"
    receipt = {
        "schema_version": 1,
        "kind": "h200_ncu_failed_transfer_receipt",
        "batch_name": name,
        "case_ids": [case["case_id"] for case in
                     json.loads(plan_path.read_text())["cases"]],
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "app": app,
        "modal_returncode": returncode,
        "failure_reason": ("client_connection_reset" if
                           "ConnectionResetError" in output else
                           "modal_process_nonzero"),
        "clock_evidence": "only_returned_files; missing telemetry is unknown",
        "partial_raw_files": files,
        "modal_log": {"path": str(log.relative_to(ROOT)),
                      "size": log.stat().st_size,
                      "sha256": hashlib.sha256(log.read_bytes()).hexdigest()},
        "task_budget_charge_before_usd": str(before["budget_charge_usd"]),
        "task_budget_charge_after_usd": str(after["budget_charge_usd"]),
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt_path


def _verify_failed_batch_receipts(ledger: dict) -> None:
    for receipt_path in (CAMPAIGN / "batches").glob(
            "*/failed_transfer_receipt.json"):
        batch = receipt_path.parent
        receipt = json.loads(receipt_path.read_text())
        name = batch.name
        plan_path = admitted_batch_plan(CAMPAIGN, name)
        if (receipt["schema_version"] != 1
                or receipt["kind"] != "h200_ncu_failed_transfer_receipt"
                or receipt["batch_name"] != name
                or receipt["plan_sha256"]
                != hashlib.sha256(plan_path.read_bytes()).hexdigest()
                or receipt["case_ids"] != [case["case_id"] for case in
                                           json.loads(plan_path.read_text())["cases"]]
                or receipt["modal_returncode"] == 0):
            raise ValueError(f"failed NCU batch receipt identity differs: {name}")
        for item in receipt["partial_raw_files"]:
            _pinned_file(batch, item)
        _pinned_file(ROOT, receipt["modal_log"])
        app = receipt["app"]
        if app is not None and app["app_id"] in ledger["apps"]:
            entry = ledger["apps"][app["app_id"]]
            if (entry["created_at"] != app["created_at"]
                    or entry["stopped_at"] != app["stopped_at"]
                    or Decimal(entry["lifecycle_seconds"]) != Decimal(str(
                        app["conservative_app_lifecycle_gpu_seconds"]))):
                raise ValueError(f"failed NCU app ledger differs: {name}")


def _verify_missing6_nsys_receipts(ledger: dict) -> None:
    manifest_path = NSYS_CAMPAIGN / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["batch_order"]:
        receipt_path = NSYS_CAMPAIGN / "receipts" / f"{name}.json"
        if ((NSYS_CAMPAIGN / "batches" / name).exists()
                or (LOGS / f"{name}.log").exists()) and not receipt_path.is_file():
            raise FileNotFoundError(
                f"attempted missing6 NSYS batch lacks receipt: {name}")
    for receipt_path in (NSYS_CAMPAIGN / "receipts").glob("*.json"):
        receipt = json.loads(receipt_path.read_text())
        name = receipt_path.stem
        batch = manifest["batches"][name]
        plan_path = NSYS_CAMPAIGN / "plans" / f"{name}.json"
        if (receipt["schema_version"] != 1
                or receipt["kind"] != "h200_missing6_nsys_attempt_receipt"
                or receipt["batch_name"] != name
                or receipt["plan_sha256"] != batch["plan_sha256"]
                or hashlib.sha256(plan_path.read_bytes()).hexdigest()
                != batch["plan_sha256"]
                or receipt["manifest_sha256"]
                != hashlib.sha256(manifest_path.read_bytes()).hexdigest()):
            raise ValueError(f"missing6 NSYS receipt identity differs: {name}")
        _pinned_file(ROOT, receipt["modal_log"])
        _pinned_file(NSYS_CAMPAIGN, receipt["review_pin"])
        for pin in receipt["raw_files"]:
            _pinned_file(NSYS_CAMPAIGN, pin)
        app = receipt["app"]
        if app is not None and app["app_id"] in ledger["apps"]:
            old = ledger["apps"][app["app_id"]]
            if (old["created_at"] != app["created_at"]
                    or old["stopped_at"] != app["stopped_at"]
                    or Decimal(old["lifecycle_seconds"]) != Decimal(str(
                        app["conservative_app_lifecycle_gpu_seconds"]))):
                raise ValueError(f"missing6 NSYS app ledger differs: {name}")


def _task_cost(modal_cli: Path, *, start_date: date,
               gpu_hourly_usd: Decimal, overhead_factor: Decimal,
               initialize: bool = False) -> dict:
    if not LEDGER.is_file() and not initialize:
        raise FileNotFoundError(f"task spend ledger missing: {LEDGER}")
    ledger = ({"schema_version": 3, "start_date": start_date.isoformat(),
               "apps": {}} if initialize else json.loads(LEDGER.read_text()))
    if initialize and LEDGER.exists():
        raise FileExistsError(LEDGER)
    if ledger["schema_version"] != 3 or ledger["start_date"] != start_date.isoformat():
        raise ValueError("task spend ledger schema or billing start date differs")
    _verify_phase_attempts(ledger)
    _verify_failed_batch_receipts(ledger)
    _verify_missing6_nsys_receipts(ledger)
    _verify_isolated_nsys_attempt(ledger)
    known = _historical_app_ids() | set(ledger["apps"])
    apps = _modal_json(modal_cli, "app", "list", "--limit", "0", "--json")
    all_ids = [app["app_id"] for app in apps]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Modal app listing contains duplicate IDs")
    by_id = {app["app_id"]: app for app in apps}
    verified_info_ids = set()
    for app_id in sorted(known - by_id.keys()):
        if app_id in ledger["apps"] and ledger["apps"][app_id]["verified_via_info"] is True:
            old = ledger["apps"][app_id]
            by_id[app_id] = {"app_id": app_id, "description": old["description"],
                             "state": "stopped", "created_at": old["created_at"],
                             "stopped_at": old["stopped_at"]}
            continue
        info = _modal_json(modal_cli, "app", "info", app_id, "--json")
        if info["app_id"] != app_id:
            raise ValueError(f"Modal app info differs from requested ID: {app_id}")
        lifecycle = info["lifecycle"]
        by_id[app_id] = {"app_id": app_id, "description": info["description"],
                         "state": lifecycle["state"],
                         "created_at": lifecycle["created_at"],
                         "stopped_at": lifecycle["stopped_at"]}
        verified_info_ids.add(app_id)
    campaign_apps = {app["app_id"]: app for app in apps
                     if app["description"] in _CAMPAIGN_APP_NAMES}
    campaign_apps.update({app_id: by_id[app_id] for app_id in known - set(campaign_apps)
                          if by_id[app_id]["description"] in _CAMPAIGN_APP_NAMES})
    if not known <= campaign_apps.keys():
        raise ValueError("historical campaign app description differs")
    for app_id, app in campaign_apps.items():
        if (app["state"] != "stopped" or app["stopped_at"] is None
                or ("tasks" in app and app["tasks"] != "0")):
            raise ValueError(f"campaign app still active: {app_id}")
        elapsed = (datetime.fromisoformat(app["stopped_at"])
                   - datetime.fromisoformat(app["created_at"]))
        seconds = Decimal(str(elapsed.total_seconds()))
        if seconds < 0:
            raise ValueError(f"negative app lifecycle: {app_id}")
        entry = {"description": app["description"],
                 "created_at": app["created_at"], "stopped_at": app["stopped_at"],
                 "lifecycle_seconds": str(seconds),
                 "max_billed_usd": "0", "max_lifecycle_estimate_usd": "0",
                 "verified_via_info": app_id in verified_info_ids}
        if app_id in ledger["apps"]:
            old = ledger["apps"][app_id]
            for key in ("description", "created_at", "stopped_at", "lifecycle_seconds"):
                if old[key] != entry[key]:
                    raise ValueError(f"app lifecycle changed: {app_id} {key}")
            entry["max_billed_usd"] = old["max_billed_usd"]
            entry["max_lifecycle_estimate_usd"] = old["max_lifecycle_estimate_usd"]
            entry["verified_via_info"] = (old["verified_via_info"]
                                          or app_id in verified_info_ids)
        ledger["apps"][app_id] = entry
    local_today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    end_date = local_today + timedelta(days=1)
    rows = _modal_json(
        modal_cli, "billing", "report", "--start", start_date.isoformat(),
        "--end", end_date.isoformat(), "--resolution", "h", "--tz",
        "Asia/Shanghai", "--show-resources", "--json")
    billed = {app_id: Decimal(0) for app_id in ledger["apps"]}
    for row in rows:
        if row["description"] not in _CAMPAIGN_APP_NAMES:
            continue
        app_id = row["object_id"]
        if app_id not in billed:
            raise ValueError(f"billed campaign app absent from complete listing: {app_id}")
        if row["description"] != ledger["apps"][app_id]["description"]:
            raise ValueError(f"billed app description differs: {app_id}")
        amount = Decimal(row["cost"])
        if not amount.is_finite() or amount < 0:
            raise ValueError(f"invalid billed amount: {app_id}")
        billed[app_id] += amount
    for app_id, amount in billed.items():
        entry = ledger["apps"][app_id]
        entry["max_billed_usd"] = str(max(Decimal(entry["max_billed_usd"]), amount))
        estimate = (Decimal(entry["lifecycle_seconds"]) * gpu_hourly_usd
                    * overhead_factor / Decimal(3600))
        entry["max_lifecycle_estimate_usd"] = str(max(
            Decimal(entry["max_lifecycle_estimate_usd"]), estimate))
    _write_ledger(ledger)
    elapsed_seconds = sum((Decimal(entry["lifecycle_seconds"])
                           for entry in ledger["apps"].values()), Decimal(0))
    bill = sum((Decimal(entry["max_billed_usd"])
                for entry in ledger["apps"].values()), Decimal(0))
    estimated = sum((Decimal(entry["max_lifecycle_estimate_usd"])
                     for entry in ledger["apps"].values()), Decimal(0))
    charge = sum((max(Decimal(entry["max_billed_usd"]),
                      Decimal(entry["max_lifecycle_estimate_usd"]))
                  for entry in ledger["apps"].values()), Decimal(0))
    return {"bill_usd": bill, "lifecycle_estimate_usd": estimated,
            "budget_charge_usd": charge, "app_count": len(ledger["apps"]),
            "app_lifecycle_seconds": elapsed_seconds}


def _app_receipt(modal_cli: Path, output: str) -> dict:
    app_ids = re.findall(r"/((?:ap-)[A-Za-z0-9]+)", output)
    if len(set(app_ids)) != 1:
        raise ValueError(f"expected one Modal app ID, found {app_ids}")
    deadline = time.monotonic() + 30
    while True:
        apps = _modal_json(modal_cli, "app", "list", "--limit", "0", "--json")
        matches = [app for app in apps if app["app_id"] == app_ids[0]]
        if len(matches) != 1:
            raise ValueError(f"Modal app missing from recent list: {app_ids[0]}")
        app = matches[0]
        if app["state"] == "stopped" and app["tasks"] == "0":
            break
        if time.monotonic() >= deadline:
            raise ValueError(f"Modal app not fully stopped: {app}")
        time.sleep(2)
    start = datetime.fromisoformat(app["created_at"])
    end = datetime.fromisoformat(app["stopped_at"])
    return {"app_id": app["app_id"], "created_at": app["created_at"],
            "stopped_at": app["stopped_at"],
            "conservative_app_lifecycle_gpu_seconds": (end - start).total_seconds()}


def _settled_task_cost(modal_cli: Path, *, completed_app_id: str,
                       start_date: date, gpu_hourly_usd: Decimal,
                       overhead_factor: Decimal,
                       timeout_seconds: float = 45) -> dict:
    """Wait for an already stopped app to leave Modal's stale active listing."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return _task_cost(
                modal_cli, start_date=start_date,
                gpu_hourly_usd=gpu_hourly_usd,
                overhead_factor=overhead_factor)
        except ValueError as error:
            if str(error) != f"campaign app still active: {completed_app_id}":
                raise
            info = _modal_json(modal_cli, "app", "info", completed_app_id,
                               "--json")
            if info["app_id"] != completed_app_id:
                raise ValueError("completed Modal app info ID differs") from error
            lifecycle = info["lifecycle"]
            if time.monotonic() >= deadline:
                raise ValueError(
                    f"completed Modal app did not settle in listing: "
                    f"{completed_app_id}; info_state={lifecycle['state']}") from error
            time.sleep(2)


def _review_batch(name: str, plan_path: Path, app: dict,
                  before: dict, after: dict) -> dict:
    batch = CAMPAIGN / "batches" / name
    plan = json.loads(plan_path.read_text())
    batch_status = json.loads((batch / "batch_status.json").read_text())
    cases = {case["case_id"]: case for case in plan["cases"]}
    case_ids = list(cases)
    statuses = {}
    for case_id in case_ids:
        status_path = batch / "ncu_raw" / case_id / "validation.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
        elif case_id in batch_status["case_statuses"]:
            recorded = batch_status["case_statuses"][case_id]
            if recorded not in {"preflight_runner_failed", "preflight_window_missing",
                                "preflight_clock_rejected", "preflight_cluster_invalid",
                                "launch_mismatch"}:
                raise ValueError(f"batch {name} lacks validation for {case_id}: {recorded}")
            status = {"status": recorded}
        else:
            status = {"status": "not_attempted"}
        if (case_id in batch_status["case_statuses"]
                and status["status"] != batch_status["case_statuses"][case_id]):
            raise ValueError(f"batch {name} disagrees on {case_id} status")
        if status["status"] == "complete":
            if batch_status["case_statuses"][case_id] != "complete":
                raise ValueError(f"batch {name} lacks complete transfer status for {case_id}")
            target = status_path.parent
            required = ("ncu_command.json", "ncu_stdout.csv", "ncu_telemetry.json",
                        "profile.ncu-rep.zst", "transfer_status.json",
                        f"preflight/cases/{case_id}/window_01.launch.json")
            if (not status_path.is_file()
                    or any(not (target / item).is_file() for item in required)
                    or type(status["kernel_count"]) is not int
                    or status["kernel_count"] != len(expected_launch_signature(cases[case_id]))):
                raise ValueError(f"batch {name} incomplete validation for {case_id}")
            command = json.loads((target / "ncu_command.json").read_text())
            transfer = json.loads((target / "transfer_status.json").read_text())
            telemetry = json.loads((target / "ncu_telemetry.json").read_text())
            if (command["returncode"] != 0
                    or command["plan_sha256"] != hashlib.sha256(plan_path.read_bytes()).hexdigest()
                    or transfer["case_id"] != case_id or transfer["status"] != "complete"
                    or telemetry["clock_accepted"] is not True):
                raise ValueError(f"batch {name} invalid complete evidence for {case_id}")
            if "reference_kind" in cases[case_id]:
                validate_missing15_local_case(
                    cases[case_id], target,
                    plan["traffic_metrics"] + plan["service_metrics"])
        statuses[case_id] = status
    raw_files = [{"path": str(path.relative_to(CAMPAIGN)),
                  "size": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(batch.rglob("*")) if path.is_file()]
    receipt = {
        "schema_version": 1, "kind": "h200_ncu_candidate_batch_review",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "app": app, "batch_kind": batch_status["kind"],
        "task_bill_before_usd": str(before["bill_usd"]),
        "task_bill_after_usd": str(after["bill_usd"]),
        "task_budget_charge_after_usd": str(after["budget_charge_usd"]),
        "statuses": statuses,
        "quality_accepted_case_count": sum(
            status["status"] == "complete" for status in statuses.values()),
        "raw_files": raw_files,
    }
    review_path = CAMPAIGN / f"{name}_review.json"
    with review_path.open("x") as target:
        json.dump(receipt, target, indent=2, sort_keys=True)
        target.write("\n")
    return receipt


def _pin(path: Path, parent: Path) -> dict:
    return {"path": str(path.relative_to(parent)), "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _run_probe(modal_cli: Path, *, kind: str, attempt: str, before: dict,
               start_date: date, gpu_hourly_usd: Decimal,
               overhead_factor: Decimal) -> dict:
    runner, campaign = PROBE_RUNNERS[kind]
    if re.fullmatch(r"attempt_[A-Za-z0-9_]+", attempt) is None:
        raise ValueError(f"invalid probe attempt name: {attempt}")
    target = campaign / attempt
    if target.exists():
        raise FileExistsError(f"probe attempt already exists: {target}")
    environment = dict(os.environ)
    environment[PROBE_GATE_ENV] = "1"
    runner_entrypoint = (str(runner) + "::main"
                         if kind == "w2_triton_dependency" else str(runner))
    try:
        result = subprocess.run(
            [str(modal_cli), "run", runner_entrypoint,
             "--output-dir", str(target)],
            cwd=ROOT, env=environment, capture_output=True, text=True,
            timeout=900)
        output = result.stdout + result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr
        output = (stdout or "") + (stderr or "") + "\ncontroller timeout after 900s\n"
        returncode = -1
    target.mkdir(parents=True, exist_ok=True)
    log = target / "modal.log"
    with log.open("x") as stream:
        stream.write(output)
    app_ids = re.findall(r"/((?:ap-)[A-Za-z0-9]+)", output)
    if len(set(app_ids)) != 1:
        raise ValueError(f"probe log has no unique app ID; blocked pending reconciliation: {log}")
    app = _app_receipt(modal_cli, output)
    info = _modal_json(modal_cli, "app", "info", app["app_id"], "--json")
    lifecycle = info["lifecycle"]
    if (info["app_id"] != app["app_id"] or lifecycle["state"] != "stopped"
            or lifecycle["created_at"] != app["created_at"]
            or lifecycle["stopped_at"] != app["stopped_at"]
            or info["description"] != PROBE_APP_NAMES[kind]
            or info["functions"]["capture"]["summary"]["gpu_config"]
            != [{"count": 1, "gpu_type": "H200"}]):
        raise ValueError(f"probe Modal app info differs; blocked: {target}")
    info_path = target / "app_info.json"
    with info_path.open("x") as stream:
        json.dump(info, stream, indent=2, sort_keys=True)
        stream.write("\n")
    receipt = {"schema_version": 1, "kind": "h200_paid_probe_attempt",
               "probe_kind": kind, "app_id": app["app_id"],
               "created_at": app["created_at"], "stopped_at": app["stopped_at"],
               "modal_returncode": returncode,
               "log": _pin(log, target), "app_info": _pin(info_path, target)}
    raw = target / "raw.json"
    if raw.is_file():
        receipt["raw"] = _pin(raw, target)
    receipt_path = target / "receipt.json"
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    after = _settled_task_cost(
        modal_cli, completed_app_id=app["app_id"], start_date=start_date,
        gpu_hourly_usd=gpu_hourly_usd, overhead_factor=overhead_factor)
    receipt["task_bill_before_usd"] = str(before["bill_usd"])
    receipt["task_bill_after_usd"] = str(after["bill_usd"])
    receipt["task_budget_charge_after_usd"] = str(after["budget_charge_usd"])
    with receipt_path.open("w") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if returncode != 0:
        raise RuntimeError(f"Modal probe failed; app and evidence retained: {target}")
    if kind in ("w2_panel_footprint", "w2_panel_grouped",
                "w2_panel_grouped_timing"):
        if "raw" not in receipt or json.loads(raw.read_text())["status"] != "complete":
            raise RuntimeError(f"W2 panel probe incomplete; app retained: {target}")
    if kind == "w2_triton_dependency":
        if "raw" not in receipt or json.loads(raw.read_text())["status"] not in (
                "compile_mismatch", "complete"):
            raise RuntimeError(f"W2 Triton dependency probe incomplete; app retained: {target}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    work = parser.add_mutually_exclusive_group(required=True)
    work.add_argument("--batch", action="append")
    work.add_argument("--nsys-batch", action="append")
    work.add_argument("--probe-kind", choices=sorted(PROBE_RUNNERS))
    parser.add_argument("--probe-attempt")
    parser.add_argument("--billing-start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--budget-usd", type=Decimal, required=True)
    parser.add_argument("--reserve-usd", type=Decimal, required=True)
    parser.add_argument("--gpu-hourly-usd", type=Decimal, required=True)
    parser.add_argument("--overhead-factor", type=Decimal, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch is not None and len(set(args.batch)) != len(args.batch):
        raise ValueError("batch names must be unique")
    if args.nsys_batch is not None and len(set(args.nsys_batch)) != len(args.nsys_batch):
        raise ValueError("NSYS batch names must be unique")
    if (args.probe_kind is None) != (args.probe_attempt is None):
        raise ValueError("probe kind and attempt must be provided together")
    if (args.budget_usd <= 0 or args.budget_usd > Decimal(50)
            or args.reserve_usd <= 0
            or args.gpu_hourly_usd <= 0 or args.overhead_factor < 1):
        raise ValueError("budget must be within the active US$50 authorization")
    if (args.reserve_usd < CONTROLLER_TIMEOUT_SECONDS
            * args.gpu_hourly_usd * args.overhead_factor / Decimal(3600)):
        raise ValueError("reserve is below the 900-second lifecycle ceiling")
    if args.batch is not None:
        for name in args.batch:
            admitted_batch_plan(CAMPAIGN, name)
            if name.startswith("ncu_missing15_"):
                verify_missing15_local_evidence(CAMPAIGN, name)
            if name.startswith("ncu_quality_retry_"):
                verify_quality_retry_local_evidence(CAMPAIGN, name)
            if (CAMPAIGN / "batches" / name).exists():
                raise FileExistsError(f"NCU batch already has raw: {name}")
    elif args.nsys_batch is not None:
        manifest_path = NSYS_CAMPAIGN / "plan_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        evidence_path = NSYS_CAMPAIGN / "source_evidence.json"
        if (manifest["status"] != "prepared_not_executed"
                or manifest["resource_plan"]["gpu"] != "H200"
                or manifest["resource_plan"]["gpu_count"] != 1
                or hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                != manifest["source_evidence_sha256"]):
            raise ValueError("missing6 NSYS campaign identity differs")
        for name in args.nsys_batch:
            if name not in manifest["batches"]:
                raise ValueError(f"unknown missing6 NSYS batch: {name}")
            plan_path = NSYS_CAMPAIGN / "plans" / f"{name}.json"
            if (hashlib.sha256(plan_path.read_bytes()).hexdigest()
                    != manifest["batches"][name]["plan_sha256"]):
                raise ValueError(f"missing6 NSYS plan changed: {name}")
            if ((NSYS_CAMPAIGN / "batches" / name).exists()
                    or (NSYS_CAMPAIGN / "receipts" / f"{name}.json").exists()):
                raise FileExistsError(f"missing6 NSYS batch already attempted: {name}")
    else:
        if re.fullmatch(r"attempt_[A-Za-z0-9_]+", args.probe_attempt) is None:
            raise ValueError("invalid probe attempt name")
        _, campaign = PROBE_RUNNERS[args.probe_kind]
        if (campaign / args.probe_attempt).exists():
            raise FileExistsError(f"probe attempt already exists: {args.probe_attempt}")
    modal_cli = Path(sys.executable).with_name("modal")
    if not modal_cli.is_file():
        raise FileNotFoundError(modal_cli)
    current = _task_cost(
        modal_cli, start_date=args.billing_start_date,
        gpu_hourly_usd=args.gpu_hourly_usd,
        overhead_factor=args.overhead_factor)
    if args.dry_run:
        print(json.dumps({"batches": args.batch, "probe_kind": args.probe_kind,
                          "nsys_batches": args.nsys_batch,
                          "probe_attempt": args.probe_attempt,
                          "task_bill_usd": str(current["bill_usd"]),
                          "budget_charge_usd": str(current["budget_charge_usd"]),
                          "app_count": current["app_count"],
                          "budget_usd": str(args.budget_usd),
                          "reserve_usd": str(args.reserve_usd)}))
        return
    if args.probe_kind is not None:
        if current["budget_charge_usd"] + args.reserve_usd > args.budget_usd:
            raise ValueError("probe budget reserve exceeds authorized total")
        receipt = _run_probe(
            modal_cli, kind=args.probe_kind, attempt=args.probe_attempt,
            before=current, start_date=args.billing_start_date,
            gpu_hourly_usd=args.gpu_hourly_usd,
            overhead_factor=args.overhead_factor)
        print(json.dumps({"probe_kind": args.probe_kind,
                          "attempt": args.probe_attempt,
                          "app_id": receipt["app_id"],
                          "budget_charge_usd": receipt["task_budget_charge_after_usd"]}))
        return
    if args.nsys_batch is not None:
        LOGS.mkdir(parents=True, exist_ok=True)
        (NSYS_CAMPAIGN / "receipts").mkdir(exist_ok=True)
        (NSYS_CAMPAIGN / "reviews").mkdir(exist_ok=True)
        manifest_path = NSYS_CAMPAIGN / "plan_manifest.json"
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        for name in args.nsys_batch:
            before = _task_cost(
                modal_cli, start_date=args.billing_start_date,
                gpu_hourly_usd=args.gpu_hourly_usd,
                overhead_factor=args.overhead_factor)
            if before["budget_charge_usd"] + args.reserve_usd > args.budget_usd:
                print(json.dumps({"stopped_before_batch": name,
                                  "reason": "budget_reserve"}))
                return
            environment = dict(os.environ)
            environment["MIDLAY_H200_NSYS_CAMPAIGN"] = NSYS_CAMPAIGN.name
            run = subprocess.run(
                [str(modal_cli), "run", f"{NSYS_RUNNER}::main",
                 "--batch-name", name], cwd=ROOT, env=environment,
                capture_output=True, text=True, timeout=900)
            output = run.stdout + run.stderr
            log_path = LOGS / f"{name}.log"
            with log_path.open("x") as stream:
                stream.write(output)
            app_ids = re.findall(r"/((?:ap-)[A-Za-z0-9]+)", output)
            app = _app_receipt(modal_cli, output) if app_ids else None
            after = _settled_task_cost(
                modal_cli, completed_app_id=app["app_id"] if app is not None else "",
                start_date=args.billing_start_date,
                gpu_hourly_usd=args.gpu_hourly_usd,
                overhead_factor=args.overhead_factor)
            review_path = NSYS_CAMPAIGN / "reviews" / f"{name}.json"
            subprocess.run(
                [str(NSYS_REVIEW_PYTHON), str(NSYS_REVIEW_RUNNER),
                 "--campaign", str(NSYS_CAMPAIGN), "--batch", name,
                 "--modal-returncode", str(run.returncode),
                 "--output", str(review_path)],
                cwd=ROOT, check=True, timeout=120)
            review = json.loads(review_path.read_text())
            receipt = {
                "schema_version": 1,
                "kind": "h200_missing6_nsys_attempt_receipt",
                "batch_name": name, "app": app,
                "modal_returncode": run.returncode,
                "plan_sha256": json.loads(manifest_path.read_text())[
                    "batches"][name]["plan_sha256"],
                "manifest_sha256": manifest_sha,
                "modal_log": {"path": str(log_path.relative_to(ROOT)),
                              "size": log_path.stat().st_size,
                              "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()},
                "review_pin": {"path": str(review_path.relative_to(NSYS_CAMPAIGN)),
                               "size": review_path.stat().st_size,
                               "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest()},
                "raw_files": review["raw_files"],
                "task_budget_charge_before_usd": str(before["budget_charge_usd"]),
                "task_budget_charge_after_usd": str(after["budget_charge_usd"]),
            }
            receipt_path = NSYS_CAMPAIGN / "receipts" / f"{name}.json"
            with receipt_path.open("x") as stream:
                json.dump(receipt, stream, indent=2, sort_keys=True)
                stream.write("\n")
            print(json.dumps({"batch": name, "case_id": review["case_id"],
                              "quality_accepted": review["quality_accepted"],
                              "status": review["status"],
                              "app_gpu_seconds": (app[
                                  "conservative_app_lifecycle_gpu_seconds"]
                                  if app is not None else None),
                              "budget_charge_usd": str(after["budget_charge_usd"])}),
                  flush=True)
            if run.returncode != 0 or app is None:
                raise RuntimeError(f"missing6 NSYS batch failed: {receipt_path}")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    for name in args.batch:
        plan_path = admitted_batch_plan(CAMPAIGN, name)
        before = _task_cost(
            modal_cli, start_date=args.billing_start_date,
            gpu_hourly_usd=args.gpu_hourly_usd,
            overhead_factor=args.overhead_factor)
        if before["budget_charge_usd"] + args.reserve_usd > args.budget_usd:
            print(json.dumps({"stopped_before_batch": name,
                              "reason": "budget_reserve"}))
            return
        result = subprocess.run(
            [str(modal_cli), "run", str(RUNNER), "--batch-name", name],
            cwd=ROOT, capture_output=True, text=True, timeout=900)
        output = result.stdout + result.stderr
        with (LOGS / f"{name}.log").open("x") as target:
            target.write(output)
        app_ids = re.findall(r"/((?:ap-)[A-Za-z0-9]+)", output)
        app = _app_receipt(modal_cli, output) if app_ids else None
        after = _settled_task_cost(
            modal_cli, completed_app_id=app["app_id"] if app is not None else "",
            start_date=args.billing_start_date,
            gpu_hourly_usd=args.gpu_hourly_usd,
            overhead_factor=args.overhead_factor)
        if result.returncode != 0:
            receipt_path = _record_failed_batch(
                name, plan_path, app, before, after, result.returncode, output)
            raise RuntimeError(f"Modal batch {name} failed; see {receipt_path}")
        if app is None:
            raise ValueError(f"Modal batch {name} has no app receipt")
        review = _review_batch(name, plan_path,
                               app, before, after)
        accepted = review["quality_accepted_case_count"]
        total = len(review["statuses"])
        print(json.dumps({"batch": name, "accepted": accepted, "total": total,
                          "app_gpu_seconds": app["conservative_app_lifecycle_gpu_seconds"],
                          "task_bill_usd": str(after["bill_usd"]),
                          "budget_charge_usd": str(after["budget_charge_usd"])}),
              flush=True)
        if review["batch_kind"] != "complete":
            print(json.dumps({"stopped_after_batch": name,
                              "reason": "batch_partial"}))
            return
        if accepted * 4 < total * 3:
            print(json.dumps({"stopped_after_batch": name,
                              "reason": "quality_below_75_percent"}))
            return


if __name__ == "__main__":
    main()
