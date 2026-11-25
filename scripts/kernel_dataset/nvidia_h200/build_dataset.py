#!/usr/bin/env python3
"""CLI for NVIDIA H200 kernel-dataset candidates (plans / assemble / check / publish).

``plans`` builds H200 runner plans from the A800 canonical case list.
``assemble`` turns kcampaign plan runs (``<root>/plans/<plan>.json`` +
``<root>/runs/<plan>/``) into one candidate per family under
``<output-root>/<plan>/``.  ``publish`` merges the evaluation-supported
families of an assembled root into ``kernel_dataset/nvidia_h200/`` and
writes the publication record next to the candidates.  ``check`` loads a candidate with the canonical
loader and validates every entry against the ``nvidia_h200`` kernel
evaluation timing contract.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200 import (  # noqa: E402
    assemble_campaign_plan,
    build_nvidia_h200_plans,
    publish_campaign_candidates,
)

DEFAULT_OUTPUT_ROOT = (
    ROOT / "profiler_output/kernel_dataset/nvidia_h200/kcampaign/candidates")


def _check(path):
    from evaluation.kernel.dataset import parse_kernel_evaluation_timing
    from src.kernel.dataset_storage import load_kernel_dataset_document

    document = load_kernel_dataset_document(path)
    op_types = {}
    for entry in document.values():
        parse_kernel_evaluation_timing(entry, entry["kernels"], "nvidia_h200")
        op_type = entry["op_info"]["Type"]
        op_types[op_type] = op_types[op_type] + 1 if op_type in op_types else 1
    print(json.dumps({
        "path": str(path), "cases": len(document), "op_types": op_types,
        "load_kernel_dataset_document": "ok",
        "timing_under_nvidia_h200_contract": "ok",
    }))
    return 0


def _discover_plans(kcampaign_root):
    """Every ``plans/<plan>.json`` of the root; runs are resolved per plan."""

    plans = sorted(path.name[:-len(".json")]
                   for path in (kcampaign_root / "plans").glob("*.json"))
    if not plans:
        raise ValueError(f"{kcampaign_root}/plans holds no plan JSON")
    return plans


def _status_table(summary):
    statuses = sorted({status for row in summary
                       for status in row["status_counts"]} - {"pass"})
    lines = [
        "| Plan | Family | Stack | Evaluated | Plan cases | Pass | Excluded | "
        + " | ".join(statuses) + " |",
        "|---|---|---|---|---:|---:|---:|" + "---:|" * len(statuses),
    ]
    for row in summary:
        counts = row["status_counts"]
        excluded = row["plan_cases"] - row["measured"]
        lines.append(
            f"| {row['plan']} | {row['family']} | {row['vllm_stack']} | "
            f"{'yes' if row['evaluation_supported'] else 'no'} | "
            f"{row['plan_cases']} | {row['measured']} | {excluded} | "
            + " | ".join(str(counts[status]) if status in counts else "0"
                         for status in statuses) + " |")
    return lines


def _assemble(kcampaign_root, plan_names, output_root):
    summary = []
    for plan_name in plan_names:
        results_root = kcampaign_root / "runs" / plan_name
        if not (results_root / "_provenance.json").is_file():
            raise ValueError(
                f"plan {plan_name} has no run provenance under {results_root}")
        manifests = assemble_campaign_plan(
            plan_path=kcampaign_root / "plans" / f"{plan_name}.json",
            results_root=results_root,
            output_dir=output_root / plan_name)
        for family, manifest in manifests.items():
            if manifest["measured_case_count"] + sum(
                    count for status, count in
                    manifest["case_status_counts"].items()
                    if status != "pass") != manifest["plan_case_count"]:
                raise AssertionError(
                    f"{plan_name}/{family} case accounting is incomplete")
            summary.append({
                "plan": plan_name, "family": family,
                "vllm_stack": manifest["vllm_stack"],
                "evaluation_supported": manifest["evaluation_supported"],
                "plan_cases": manifest["plan_case_count"],
                "measured": manifest["measured_case_count"],
                "status_counts": manifest["case_status_counts"],
                "candidate": manifest["candidate_path"],
            })
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "assembly_summary.json").write_text(
        json.dumps({"kcampaign_root": str(kcampaign_root.resolve()),
                    "plans": plan_names, "rows": summary}, indent=1),
        encoding="utf-8")
    (output_root / "assembly_summary.md").write_text(
        "\n".join(_status_table(summary)) + "\n", encoding="utf-8")
    print("\n".join(_status_table(summary)))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",
                        choices=("plans", "assemble", "check", "publish"))
    parser.add_argument("--family", help="plans: A800 canonical family")
    parser.add_argument(
        "--a800-config", type=Path,
        default=ROOT / "configs/dataset/nvidia_a800.yaml")
    parser.add_argument("--output", type=Path, help="plans: output JSON")
    parser.add_argument(
        "--kcampaign-root", type=Path,
        help="assemble: directory holding plans/<plan>.json and runs/<plan>/")
    parser.add_argument("--plan", nargs="+", help="assemble: plan names")
    parser.add_argument(
        "--all-plans", action="store_true",
        help="assemble: every plans/<plan>.json under --kcampaign-root")
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate", type=Path,
                        help="check: candidate .json.zst")
    parser.add_argument(
        "--candidate-root", type=Path,
        help="publish: assembled root holding <plan>/<family>.manifest.json")
    args = parser.parse_args()
    if args.action == "plans":
        if args.output is None or args.family is None:
            raise ValueError("plans requires --family and --output")
        payload = build_nvidia_h200_plans(
            a800_config_path=args.a800_config, family=args.family)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(json.dumps({"family": args.family,
                          "case_count": payload["case_count"],
                          "output": str(args.output)}))
        return 0
    if args.action == "check":
        if args.candidate is None:
            raise ValueError("check requires --candidate")
        return _check(args.candidate)
    if args.action == "publish":
        if args.candidate_root is None:
            raise ValueError("publish requires --candidate-root")
        record = publish_campaign_candidates(
            candidate_root=args.candidate_root,
            publication_dir=args.candidate_root / "publication",
            canonical_dir=ROOT / "kernel_dataset" / "nvidia_h200")
        print(json.dumps({
            "published_case_counts": record["published_case_counts"],
            "deduplicated_cases": len(record["deduplicated_cases"]),
            "canonical_sha256": record["canonical_sha256"]}, indent=1))
        return 0
    if args.kcampaign_root is None or bool(args.plan) == args.all_plans:
        raise ValueError(
            "assemble requires --kcampaign-root and exactly one of --plan "
            "or --all-plans")
    plans = (_discover_plans(args.kcampaign_root) if args.all_plans
             else args.plan)
    return _assemble(args.kcampaign_root, plans, args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
