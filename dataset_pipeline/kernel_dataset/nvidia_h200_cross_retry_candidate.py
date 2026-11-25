"""Review existing H200 NSYS retry windows without changing their raw files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dataset_pipeline.kernel_dataset.nvidia_h200 import (
    _entry,
    select_case_timing,
    window_vllm_stack,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "profiler_output/kernel_dataset/nvidia_h200"
LEDGER = DATA / "ncu_1027_20261031/task_spend_ledger.json"
CAMPAIGNS = (
    "missing12_retry2_20261101",
    "missing9_retry3_20261101",
    "missing8_onecase_retry_20261102",
)
CASES = (
    "FUSEDMOE_TOK32768_E8_TOPK2_H4096_INT14336",
    "GEMM_B8_M4096_K8192_N28672",
)
OUTPUT = DATA / "missing8_onecase_retry_20261102/cross_retry_candidate.json"


def _pin(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _checked_pin(path: Path, campaign: Path, pins: dict[str, dict]) -> dict:
    rel = str(path.relative_to(campaign))
    if rel not in pins:
        raise ValueError(f"source review does not pin {path}")
    pin = _pin(path)
    if pin["size"] != pins[rel]["size"] or pin["sha256"] != pins[rel]["sha256"]:
        raise ValueError(f"source review pin differs: {path}")
    return pin


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    if ledger["schema_version"] != 3:
        raise ValueError("task spend ledger schema differs")
    result = {"kind": "h200_nsys_cross_retry_candidate", "schema_version": 1,
              "status": "worker_candidate_not_published",
              "source_ledger": _pin(LEDGER), "cases": {}}
    for case_id in CASES:
        records = []
        case_plan = None
        for campaign_name in CAMPAIGNS:
            campaign = DATA / campaign_name
            manifest_path = campaign / "plan_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for batch_name, batch in manifest["batches"].items():
                if case_id not in batch["case_ids"]:
                    continue
                plan_path = ROOT / batch["plan_path"]
                plan_pin = _pin(plan_path)
                if plan_pin["sha256"] != batch["plan_sha256"]:
                    raise ValueError(f"source plan pin differs: {plan_path}")
                plans = json.loads(plan_path.read_text())["plans"]
                matches = [plan for plan in plans if plan["case_id"] == case_id]
                if len(matches) != 1:
                    raise ValueError(f"source case plan differs: {plan_path}")
                if case_plan is None:
                    case_plan = matches[0]
                elif case_plan != matches[0]:
                    raise ValueError(f"cross retry workload differs: {plan_path}")
                review_path = (campaign / ("capture_review.json" if
                               campaign_name == "missing8_onecase_retry_20261102"
                               else batch_name + "_review.json"))
                review = json.loads(review_path.read_text())
                app_id = review["app_id"]
                if app_id not in ledger["apps"]:
                    raise ValueError(f"source app missing from task ledger: {app_id}")
                if ledger["apps"][app_id]["stopped_at"] is None:
                    raise ValueError(f"source app has no stop time: {app_id}")
                pins = {item["path"]: item for item in review["raw_files"]}
                source = campaign / "batches" / batch_name / "runs"
                identity_path = source / "identity.json"
                identity_pin = _checked_pin(identity_path, campaign, pins)
                identity = json.loads(identity_path.read_text())
                if not identity["gpu"].startswith("NVIDIA H200"):
                    raise ValueError(f"source GPU differs: {identity_path}")
                case_dir = source / batch_name / "cases" / case_id
                for path in sorted(case_dir.glob("window_*.launch.json")):
                    window = json.loads(path.read_text())
                    if (window["marker"] != case_plan["marker"]
                            or window["runner_relpath"] != case_plan["runner_relpath"]
                            or window["runner_args"] != case_plan["runner_args"]
                            or [[item["name"], item["grid"], item["block"]]
                                for item in window["launches"]]
                            != batch["expected_launch_signatures"][case_id]["signature"]):
                        raise ValueError(f"source launch/workload differs: {path}")
                    stem = path.name.removesuffix(".launch.json")
                    evidence = [_checked_pin(case_dir / (stem + suffix),
                                             campaign, pins)
                                for suffix in (".launch.json", ".telemetry.json",
                                               ".nsys-rep")]
                    telemetry = window["telemetry"]
                    if telemetry != json.loads(
                            (case_dir / (stem + ".telemetry.json")).read_text()):
                        raise ValueError(f"source telemetry differs: {path}")
                    records.append({"window": window, "evidence": evidence,
                                    "source_review": _pin(review_path),
                                    "identity": identity,
                                    "identity_pin": identity_pin,
                                    "app_id": app_id,
                                    "source_window": window["window"]})
        if not records:
            raise ValueError(f"no source windows: {case_id}")
        records.sort(key=lambda item: item["window"]["telemetry"][
            "nvtx_push_epoch_ns"])
        windows = []
        for number, record in enumerate(records, start=1):
            # Logical IDs encode retry chronology only. Raw files stay intact.
            window = dict(record["window"])
            window["window"] = number
            windows.append(window)
        decision = select_case_timing(case_id, windows, [])
        if decision["status"] != "pass":
            raise ValueError(f"cross retry candidate does not pass: {case_id}: "
                             f"{decision['status']}")
        retained = [records[number - 1] for number in decision["retained_windows"]]
        if any(record["identity"] != retained[0]["identity"]
               for record in retained[1:]):
            raise ValueError(f"cross retry software identity differs: {case_id}")
        result["cases"][case_id] = {
            "status": "candidate_quality_pass_cross_retry",
            "source_case_plan": case_plan,
            "candidate_case_entry": _entry(
                case_plan["family"], case_plan, decision,
                "cross_retry_candidate_20261102",
                window_vllm_stack(decision["selected"])),
            "window_count": len(records),
            "retained": [{"logical_window": number,
                          "original_window": record["source_window"],
                          "app_id": record["app_id"],
                          "identity": record["identity_pin"],
                          "source_review": record["source_review"],
                          "evidence": record["evidence"]}
                         for number, record in zip(
                             decision["retained_windows"], retained)],
            "window_totals_ns": decision["window_totals_ns"],
            "maximum_relative_difference": decision[
                "maximum_relative_difference"],
            "selected_logical_window": decision["selected_window"],
        }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
