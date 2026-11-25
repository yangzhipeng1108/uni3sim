"""Prepare six source-identical H200 NSYS retries from pinned failed cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dataset_pipeline.kernel_dataset.nvidia_h200 import (
    load_campaign_plan, load_case_failures, load_case_windows,
    select_case_timing,
)


CAMPAIGN_NAME = "missing6_retry_20261103"
DIAGNOSTIC = (
    "experiment_logs/h200_accuracy_20261028/"
    "remaining6_nsys_clock_screen_20261102.json")
SOURCE_NAME = "missing15_retry_20261031"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_missing6_retry(root: Path) -> Path:
    source = root / "profiler_output/kernel_dataset/nvidia_h200" / SOURCE_NAME
    target = root / "profiler_output/kernel_dataset/nvidia_h200" / CAMPAIGN_NAME
    diagnostic_path = root / DIAGNOSTIC
    diagnostic = json.loads(diagnostic_path.read_text())
    rows = diagnostic["rows"]
    if (diagnostic["kind"] != "h200_remaining6_nsys_pre_nvtx_settle_screen"
            or len(rows) != 6
            or len({row["case_id"] for row in rows}) != 6):
        raise ValueError("missing6 source diagnostic differs")
    old_manifest_path = source / "plan_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text())
    if (old_manifest["status"] != "prepared_not_executed"
            or len(old_manifest["batches"]) != 4):
        raise ValueError("missing15 source campaign differs")
    review_path = source / "capture_review.json"
    review = json.loads(review_path.read_text())
    statuses = {case_id: status for batch in review["batches"]
                for case_id, status in batch["case_statuses"].items()}
    if not set(row["case_id"] for row in rows) <= set(statuses):
        raise ValueError("missing6 case lacks pinned prior review")
    for row in rows:
        if statuses[row["case_id"]] == "pass":
            raise ValueError("missing6 case already passed first review")
    if target.exists():
        raise FileExistsError(target)

    batches = {}
    plans = {}
    source_entries = []
    for index, row in enumerate(rows, start=1):
        case_id = row["case_id"]
        matching = [(name, batch) for name, batch in old_manifest["batches"].items()
                    if case_id in batch["case_ids"]]
        if len(matching) != 1:
            raise ValueError(f"missing6 source plan is ambiguous: {case_id}")
        old_name, old_batch = matching[0]
        old_plan_path = root / old_batch["plan_path"]
        if _sha256(old_plan_path) != old_batch["plan_sha256"]:
            raise ValueError(f"missing6 old plan changed: {case_id}")
        old_plan = load_campaign_plan(old_plan_path)
        case_plans = [plan for plan in old_plan["plans"]
                      if plan["case_id"] == case_id]
        if len(case_plans) != 1:
            raise ValueError(f"missing6 old workload differs: {case_id}")
        family = old_batch["family"]
        name = f"missing6_{family}_{index:02d}__v030"
        plan = {"schema_version": 1, "hardware_id": "nvidia_h200",
                "campaign": CAMPAIGN_NAME, "case_count": 1,
                "plans": case_plans}
        encoded = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
        plans[name] = encoded
        batches[name] = {
            "family": family, "case_ids": [case_id],
            "plan_path": str((target / "plans" / f"{name}.json").relative_to(root)),
            "plan_sha256": hashlib.sha256(encoded).hexdigest(),
            "expected_launch_signatures": {
                case_id: old_batch["expected_launch_signatures"][case_id]},
        }
        source_entries.append({"case_id": case_id, "old_batch": old_name,
                               "old_plan_sha256": old_batch["plan_sha256"],
                               "old_review_status": statuses[case_id],
                               "diagnostic_review_sha256": row["review_sha256"]})

    target.mkdir(parents=True)
    (target / "plans").mkdir()
    for name, encoded in plans.items():
        (target / "plans" / f"{name}.json").write_bytes(encoded)
    evidence = {
        "schema_version": 1, "kind": "h200_missing6_nsys_retry_source",
        "source_manifest_sha256": _sha256(old_manifest_path),
        "source_review_sha256": _sha256(review_path),
        "diagnostic_sha256": _sha256(diagnostic_path),
        "cases": source_entries,
    }
    evidence_path = target / "source_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "status": "prepared_not_executed",
        "purpose": "Six source-identical NSYS quality retries; worker raw candidates only",
        "source_campaign": str(source),
        "source_manifest_sha256": _sha256(old_manifest_path),
        "source_evidence_sha256": _sha256(evidence_path),
        "canonical_base_dev": old_manifest["canonical_base_dev"],
        "retention": "pinned-raw; preserve all successful and failed windows",
        "resource_plan": {
            "gpu": "H200", "gpu_count": 1, "per_batch_timeout_s": 600,
            "max_attempts_per_case": 1, "reserve_usd": 1.40,
            "budget_usd": 50, "no_model_weights": True,
            "windows_per_case": 3, "min_sm_clock_mhz": 1980,
            "max_spread": 0.05,
        },
        "batch_order": list(batches), "batches": batches,
    }
    path = target / "plan_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def review_missing6_attempt(campaign: Path, batch_name: str,
                            modal_returncode: int) -> dict:
    """Pin every returned file and apply the unchanged three-window gate."""

    manifest = json.loads((campaign / "plan_manifest.json").read_text())
    batch = manifest["batches"][batch_name]
    case_id = batch["case_ids"][0]
    plan_path = campaign / "plans" / f"{batch_name}.json"
    if _sha256(plan_path) != batch["plan_sha256"]:
        raise ValueError("missing6 NSYS plan changed")
    plan = load_campaign_plan(plan_path)["plans"][0]
    if (plan["case_id"] != case_id or plan["family"] != batch["family"]):
        raise ValueError("missing6 NSYS plan case differs")
    runs = campaign / "batches" / batch_name / "runs"
    files = []
    if runs.is_dir():
        for path in sorted(runs.rglob("*")):
            if path.is_file():
                files.append({"path": str(path.relative_to(campaign)),
                              "size": path.stat().st_size,
                              "sha256": _sha256(path)})
    identity_path = runs / "identity.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text())
        if (not identity["gpu"].startswith("NVIDIA H200,")
                or identity["versions"]["vllm"] != "0.30.0+cu129"
                or identity["versions"]["torch"] != "2.13.0+cu129"
                or identity["versions"]["torch_cuda"] != "12.9"):
            raise ValueError("missing6 NSYS GPU/software identity differs")
    transfer_path = runs / "transfer_status.json"
    transfer = (json.loads(transfer_path.read_text())
                if transfer_path.is_file() else None)
    case_dir = runs / batch_name / "cases" / case_id
    windows = load_case_windows(case_dir) if case_dir.is_dir() else []
    failures = load_case_failures(case_dir) if case_dir.is_dir() else []
    expected = batch["expected_launch_signatures"][case_id]["signature"]
    for window in windows:
        if (window["case_id"] != case_id or window["family"] != plan["family"]
                or window["marker"] != plan["marker"]
                or window["runner_args"] != plan["runner_args"]
                or [[item["name"], item["grid"], item["block"]]
                    for item in window["launches"]] != expected):
            raise ValueError(f"missing6 NSYS workload or launch differs: {case_id}")
    failed_ids = [failure["window"] for failure in failures]
    decision = (select_case_timing(case_id, windows, failed_ids)
                if windows or failed_ids else None)
    status = decision["status"] if decision is not None else "not_captured"
    quality_accepted = (modal_returncode == 0 and transfer is not None
                        and transfer["kind"] == "complete"
                        and identity_path.is_file() and status == "pass")
    return {
        "schema_version": 1, "kind": "h200_missing6_nsys_attempt_review",
        "batch_name": batch_name, "case_id": case_id,
        "source_plan_sha256": batch["plan_sha256"],
        "status": status, "quality_accepted": quality_accepted,
        "transfer_status": transfer, "raw_files": files,
        "retained_windows": (decision["retained_windows"]
                             if decision is not None and status == "pass" else []),
        "maximum_relative_difference": (
            decision["maximum_relative_difference"]
            if decision is not None and status == "pass" else None),
    }
