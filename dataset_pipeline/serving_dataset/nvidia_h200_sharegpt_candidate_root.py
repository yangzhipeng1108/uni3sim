#!/usr/bin/env python3
"""Build one H200 ShareGPT candidate dataset root for one vLLM version.

Every case of the 15-case H200 matrix is converted by
``build_nvidia_h200_sharegpt_candidate.py`` (compact schema v7) and copied into ``<output-root>/vllm_<version>/nvidia_h200/<model>/
<case>/``, the layout the Serving evaluators read as ``--dataset-root``.
Output never goes to ``serving_dataset/``.

Each H200 case carries its own five-metric trace-off/trace-on observer result.
A failed gate remains provisional; another case cannot replace its evidence.

``--alignment-stage-root`` reuses already validated stage-1 cases
(``<root>/<model>/<case>/alignment_stage``) through hard links; otherwise the
raw ``raw/ShareGPT-*`` tree of each case is converted again.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

from dataset_pipeline.serving_dataset.compact_case import load_compact_case  # noqa: E402

BUILDER = ROOT / "scripts/evaluation/serving/build_nvidia_h200_sharegpt_candidate.py"
HARDWARE_ID = "nvidia_h200"
CASES = (
    "qwen3-4b/tp1_qps4", "qwen3-4b/tp1_qps16", "qwen3-4b/tp1_qps64",
    "qwen3-8b/tp1_qps4", "qwen3-8b/tp1_qps16", "qwen3-8b/tp1_qps64",
    "llama-3.1-8b/tp1_qps4", "llama-3.1-8b/tp1_qps16", "llama-3.1-8b/tp1_qps64",
    "qwen3-32b/tp2_ep1_qps1", "qwen3-32b/tp2_ep1_qps4", "qwen3-32b/tp2_ep1_qps16",
    "qwen3-30b-a3b/tp2_ep2_qps1", "qwen3-30b-a3b/tp2_ep2_qps4",
    "qwen3-30b-a3b/tp2_ep2_qps16",
)
_VERSIONS = ("0.11.0", "0.30.0")
_CANDIDATE_FILES = (
    "manifest.json", "requests.csv", "metrics.json",
    "scheduler_steps.jsonl.zst", "alignment_steps.json.zst",
)


def _own_gate(case_dir: Path) -> dict[str, object]:
    gate = json.loads(
        (case_dir / "trace_on/scheduler_trace_observer_effect.json").read_text())
    if gate["kind"] != "scheduler_trace_observer_effect":
        raise ValueError(f"unexpected observer-effect record: {case_dir}")
    if not isinstance(gate["passed"], bool):
        raise TypeError(f"observer-effect passed must be boolean: {case_dir}")
    return gate


def _source_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()


def _build_case(args, version: str, case: str,
                commit: str) -> dict[str, object]:
    model_id, case_id = case.split("/")
    raw_case = args.raw_root / version / HARDWARE_ID / model_id / case_id
    work = args.output_root / "work" / f"vllm_{version}" / model_id / case_id
    if work.exists():
        raise FileExistsError(f"refusing to overwrite conversion work: {work}")
    work.mkdir(parents=True)
    command = [
        sys.executable, str(BUILDER), "--case-dir", str(raw_case),
        "--work-root", str(work), "--converter-source-commit", commit,
    ]
    if args.alignment_stage_root is not None:
        source = args.alignment_stage_root / f"vllm_{version}" / model_id / case_id / "alignment_stage"
        subprocess.run(("cp", "-al", str(source.resolve(strict=True)),
                        str(work / "alignment_stage")), check=True)
        command.append("--reuse-alignment-stage")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with (work / "stdout.log").open("w") as stdout, (work / "stderr.log").open("w") as stderr:
        returncode = subprocess.call(command, stdout=stdout, stderr=stderr,
                                     env=environment, cwd=ROOT)
    (work / "command.json").write_text(json.dumps(command, indent=1) + "\n")
    if returncode != 0:
        raise RuntimeError(f"{version} {case} conversion failed rc={returncode}; see {work}")
    decoder = json.JSONDecoder()
    text = (work / "stdout.log").read_text()
    summary = None
    for index in range(len(text)):
        if text.startswith('{\n  "candidate"', index):
            summary, _ = decoder.raw_decode(text[index:])
    if summary is None:
        raise RuntimeError(f"{version} {case} printed no candidate summary")
    source = Path(summary["candidate"])
    destination = (args.output_root / f"vllm_{version}" / HARDWARE_ID
                   / model_id / case_id)
    destination.mkdir(parents=True)
    for name in _CANDIDATE_FILES:
        shutil.copy2(source / name, destination / name)
    load_compact_case(destination)
    return {"case": case, "candidate": str(destination), **summary["summary"]}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vllm-version", choices=_VERSIONS, required=True)
    parser.add_argument(
        "--raw-root", type=Path, required=True,
        help="capture mirror holding <version>/nvidia_h200/<model>/<case>/")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alignment-stage-root", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    args.raw_root = args.raw_root.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    if ROOT / "serving_dataset" in (args.output_root, *args.output_root.parents):
        raise ValueError("H200 candidates must not be written into serving_dataset/")
    version = args.vllm_version
    dataset_root = args.output_root / f"vllm_{version}"
    if dataset_root.exists():
        raise FileExistsError(f"refusing to overwrite candidate root: {dataset_root}")
    gates = {
        case: _own_gate(args.raw_root / version / HARDWARE_ID / case)
        for case in CASES
    }
    commit = _source_commit()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            case: pool.submit(
                _build_case, args, version, case, commit)
            for case in CASES
        }
        rows = [futures[case].result() for case in CASES]
    for row in rows:
        row["own_gate"] = {
            "passed": gates[row["case"]]["passed"],
            "metric_delta_percent": gates[row["case"]]["metric_delta_percent"],
        }
    status = {
        "schema_version": 1,
        "hardware_id": HARDWARE_ID,
        "vllm_version": version,
        "dataset_root": str(dataset_root),
        "converter_source_commit": commit,
        "observer_evidence": "measured_for_each_case",
        "cases": rows,
    }
    (args.output_root / f"status_vllm_{version}.json").write_text(
        json.dumps(status, indent=1, sort_keys=True) + "\n")
    lines = [
        f"# H200 ShareGPT candidates, vLLM {version}",
        "",
        f"Dataset root `{dataset_root}`; converter commit `{commit}`; "
        "each case keeps its own observer-effect result.",
        "",
        "| case | gate | gate source | own gate passed | own d_ttft % | own d_tpot % "
        "| requests | metric rows | steps | cons | finished cover all | depth |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        gate = row["observer_effect"]
        own = row["own_gate"]
        lines.append(
            f"| {row['case']} | {gate['status']} | {gate['validation_case_id']} "
            f"| {own['passed']} | {own['metric_delta_percent']['ttft']:.2f} "
            f"| {own['metric_delta_percent']['tpot']:.2f} | {row['request_rows']} "
            f"| {row['metric_rows']} | {row['scheduler_step_rows']} "
            f"| {row['scheduled_minus_input_plus_output_minus_one']} "
            f"| {row['finished_requests_cover_all']} | {row['scheduler_pipeline_depth']} |")
    (args.output_root / f"status_vllm_{version}.md").write_text("\n".join(lines) + "\n")
    print(args.output_root / f"status_vllm_{version}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
