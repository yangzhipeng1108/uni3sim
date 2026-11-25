#!/usr/bin/env python3
"""Convert one H200 ShareGPT capture into a compact Serving case candidate.

Input is one case directory written by
``scripts/experiments/run_nvidia_h200_sharegpt_capture.py``.  Conversion uses
the same two converters as the canonical A800 cases, unchanged in schema:

1. ``sharegpt_raw_adapter`` normalizes ``raw/ShareGPT-*`` (operators,
   Scheduler sidecar, Nsight) into a compact case with ``alignment_steps``;
2. ``scheduler_capture_candidate`` (H200 compact schema v7)
   binds the no-profiler trace-on ground truth, its Scheduler trace and the
   measured observer-effect gate, producing ``requests.csv``, ``metrics.json``,
   ``scheduler_steps.jsonl.zst`` and ``manifest.json``.

Candidates are written below ``--work-root`` (each stage under its own
``profiler_output/serving_dataset_compaction``) and never into
``serving_dataset/``.  A consistency summary is printed at the end.

The trace-on run defaults to ``<case>/trace_on/``.  A trace-on repeated with
``run_nvidia_h200_sharegpt_capture.py rerun-trace-on --tag <tag>`` lives in
``<case>/trace_on_<tag>/`` next to its record ``case_capture.trace_on_<tag>.json``
and is selected explicitly with ``--trace-on-dir-name trace_on_<tag>``.  Each
H200 candidate uses its own observer gate from that run.  The selected directory
names are recorded in ``manifest.json`` (``capture.run_id``,
``scheduler_provenance.capture_run_id`` and each accepted ``raw_sources``
``relative_path``) and echoed in the printed summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.compact_candidate_writer import (  # noqa: E402
    write_compact_case_candidate,
)
from dataset_pipeline.serving_dataset.compact_case import load_compact_case  # noqa: E402
from dataset_pipeline.serving_dataset.scheduler_capture_candidate import (  # noqa: E402
    build_scheduler_capture_candidate,
)
from dataset_pipeline.serving_dataset.sharegpt_raw_adapter import (  # noqa: E402
    build_sharegpt_case_compact_payload,
)

_MODELS = {
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "qwen3-32b": "Qwen3-32B",
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
}

_DEFAULT_TRACE_ON_DIR_NAME = "trace_on"
# ``rerun-trace-on --tag`` accepts letters, digits and underscores.
_TRACE_ON_DIR_NAME = re.compile(r"trace_on(?:_[A-Za-z0-9_]+)?")
_TRACE_ON_RUN_FILES = ("ground_truth.json", "scheduler_steps.raw.jsonl")
_OBSERVER_GATE_FILE = "scheduler_trace_observer_effect.json"
_ACCEPTED_SOURCE_IDS = (
    "accepted-ground-truth-v3",
    "accepted-scheduler-trace-v3",
    "accepted-scheduler-runtime-depth-v1",
    "scheduler-observer-effect-v1",
)


def _trace_on_dir(case_dir: Path, name: str, required_files: tuple[str, ...]) -> Path:
    """Return a validated trace-on directory of one captured case.

    ``trace_on`` is the capture's own run.  ``trace_on_<tag>`` must be a
    finished ``rerun-trace-on`` of the same case and runtime, as recorded in
    ``case_capture.trace_on_<tag>.json``.
    """

    if _TRACE_ON_DIR_NAME.fullmatch(name) is None:
        raise ValueError(
            f"trace-on directory name must be 'trace_on' or 'trace_on_<tag>' with "
            f"a letters/digits/underscore tag, got {name!r}")
    directory = case_dir / name
    if not directory.is_dir():
        raise FileNotFoundError(f"trace-on directory does not exist: {directory}")
    for filename in required_files:
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"trace-on directory lacks {filename}: {directory}")
    if name != _DEFAULT_TRACE_ON_DIR_NAME:
        original = json.loads((case_dir / "case_capture.json").read_text())
        record_path = case_dir / f"case_capture.{name}.json"
        if not record_path.is_file():
            raise FileNotFoundError(f"rerun trace-on record does not exist: {record_path}")
        record = json.loads(record_path.read_text())
        expected = {
            "schema_version": 1,
            "hardware_id": original["hardware_id"],
            "case": original["case"],
            "runtime_version": original["runtime_version"],
            "rerun_of": _DEFAULT_TRACE_ON_DIR_NAME,
            "rerun_dir": name,
        }
        observed = {key: record[key] for key in expected}
        if observed != expected:
            raise ValueError(
                f"{record_path} does not record a rerun of this case into {name}: "
                f"expected {expected}, got {observed}")
        if "ended" not in record:
            raise ValueError(f"{record_path} has no 'ended' time; the rerun did not finish")
    return directory


def _case_contract(case_dir: Path) -> dict[str, object]:
    record = json.loads((case_dir / "case_capture.json").read_text())
    if record["hardware_id"] != "nvidia_h200":
        raise ValueError("case capture is not an H200 capture")
    model_id, case_id = record["case"].split("/")
    raw_model_id = _MODELS[model_id]
    if case_id.startswith("tp1_qps"):
        tp, ep, qps = 1, 1, int(case_id[len("tp1_qps"):])
        raw_name = f"ShareGPT-{raw_model_id}_TP1_n128_qps{qps}"
    elif case_id.startswith("tp2_ep1_qps"):
        tp, ep, qps = 2, 1, int(case_id[len("tp2_ep1_qps"):])
        raw_name = f"ShareGPT-{raw_model_id}_TP2_n128_qps{qps}"
    elif case_id.startswith("tp2_ep2_qps"):
        tp, ep, qps = 2, 2, int(case_id[len("tp2_ep2_qps"):])
        raw_name = f"ShareGPT-{raw_model_id}_EP2_n128_qps{qps}"
    else:
        raise ValueError(f"unsupported H200 case id: {case_id!r}")
    return {
        "model_id": model_id, "case_id": case_id, "raw_model_id": raw_model_id,
        "tp": tp, "ep": ep, "qps": qps, "raw_dir": case_dir / "raw" / raw_name,
    }


def _candidate_root(work_root: Path, stage: str) -> Path:
    root = work_root / stage / "profiler_output" / "serving_dataset_compaction"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_alignment_stage(
    args: argparse.Namespace,
    contract: dict[str, object],
    alignment_root: Path,
    alignment_case: Path,
) -> None:
    payload = build_sharegpt_case_compact_payload(
        contract["raw_dir"],
        hardware_id="nvidia_h200",
        raw_model_id=contract["raw_model_id"],
        target_qps=contract["qps"],
        tensor_parallel_size=contract["tp"],
        expert_parallel_size=contract["ep"],
        case_id=contract["case_id"],
        publish_metrics=True,
        converter_source_commit=args.converter_source_commit,
    )
    alignment_case.parent.mkdir(parents=True, exist_ok=True)
    write_compact_case_candidate(alignment_case, payload, candidate_root=alignment_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--converter-source-commit", required=True)
    parser.add_argument(
        "--observer-representative-case-dir", type=Path,
        help=(
            "inherit the trace-off/trace-on gate from this H200 case of the same "
            "runtime version (the A800/C500/910B canonical cases inherit theirs "
            "from one representative case); default: this case's own gate"),
    )
    parser.add_argument(
        "--trace-on-dir-name", default=_DEFAULT_TRACE_ON_DIR_NAME,
        help=(
            "trace-on run directory inside --case-dir: 'trace_on' (default) or a "
            "finished 'rerun-trace-on --tag <tag>' output 'trace_on_<tag>'; with the "
            "case's own gate its scheduler_trace_observer_effect.json is used"),
    )
    parser.add_argument(
        "--observer-representative-trace-on-dir-name",
        help=(
            "directory inside --observer-representative-case-dir holding the "
            "representative gate scheduler_trace_observer_effect.json; "
            "default 'trace_on'"),
    )
    parser.add_argument(
        "--reuse-alignment-stage", action="store_true",
        help="reuse an existing, validated stage-1 case under --work-root")
    args = parser.parse_args()
    case_dir = args.case_dir.resolve(strict=True)
    work_root = args.work_root.resolve()
    if ROOT / "serving_dataset" in (work_root, *work_root.parents):
        raise ValueError("H200 candidates must not be written into serving_dataset/")
    contract = _case_contract(case_dir)
    own_gate = args.observer_representative_case_dir is None
    if not own_gate:
        raise ValueError("H200 candidates require their own observer-effect evidence")
    if own_gate and args.observer_representative_trace_on_dir_name is not None:
        raise ValueError(
            "--observer-representative-trace-on-dir-name requires "
            "--observer-representative-case-dir")
    trace_on = _trace_on_dir(
        case_dir, args.trace_on_dir_name,
        _TRACE_ON_RUN_FILES + ((_OBSERVER_GATE_FILE,) if own_gate else ()))
    if own_gate:
        observer_path = trace_on / _OBSERVER_GATE_FILE
        observer_status = "measured_for_this_case"
    else:
        representative = args.observer_representative_case_dir.resolve(strict=True)
        own = json.loads((case_dir / "case_capture.json").read_text())
        other = json.loads((representative / "case_capture.json").read_text())
        if (other["hardware_id"], other["runtime_version"]) != (
                own["hardware_id"], own["runtime_version"]):
            raise ValueError("representative observer case must share hardware and runtime")
        representative_name = (
            _DEFAULT_TRACE_ON_DIR_NAME
            if args.observer_representative_trace_on_dir_name is None
            else args.observer_representative_trace_on_dir_name)
        observer_path = _trace_on_dir(
            representative, representative_name, (_OBSERVER_GATE_FILE,)
        ) / _OBSERVER_GATE_FILE
        observer_status = "inherited_from_representative_case"
    alignment_root = _candidate_root(work_root, "alignment_stage")
    alignment_case = alignment_root / "nvidia_h200" / contract["model_id"] / contract["case_id"]
    if args.reuse_alignment_stage:
        load_compact_case(alignment_case)
    else:
        _write_alignment_stage(args, contract, alignment_root, alignment_case)
    # Raw consistency is reported before the observer-effect gate so a case
    # whose gate fails still shows whether its evidence itself is sound.
    print(json.dumps({"raw_trace_on_consistency": _raw_consistency(
        alignment_case, trace_on)}, indent=2, sort_keys=True), flush=True)
    final_root = _candidate_root(work_root, "candidate")
    output = build_scheduler_capture_candidate(
        canonical_case_dir=alignment_case,
        trace_on_result_path=trace_on / "ground_truth.json",
        scheduler_trace_path=trace_on / "scheduler_steps.raw.jsonl",
        observer_effect_path=observer_path,
        candidate_root=final_root,
        converter_source_commit=args.converter_source_commit,
        observer_validation_status=observer_status,
    )
    load_compact_case(output)
    print(json.dumps({"candidate": str(output), "summary": _summary(output)},
                     indent=2, sort_keys=True))
    return 0


def _raw_consistency(alignment_case: Path, trace_on: Path) -> dict[str, object]:
    import hashlib

    from dataset_pipeline.serving_dataset.scheduler_capture_candidate import (
        _accepted_requests_and_metrics,
        _remapped_scheduler_steps,
        _runtime_depth_metadata,
    )

    result_path = trace_on / "ground_truth.json"
    trace_path = trace_on / "scheduler_steps.raw.jsonl"
    result = json.loads(result_path.read_text())
    requests, old_to_new, metrics = _accepted_requests_and_metrics(
        result, hashlib.sha256(result_path.read_bytes()).hexdigest(),
        alignment_case / "requests.csv")
    steps = _remapped_scheduler_steps(trace_path, old_to_new)
    depth, _ = _runtime_depth_metadata(Path(f"{trace_path}.runtime.json"), steps)
    scheduled: Counter[int] = Counter()
    for step in steps:
        for work in step["scheduled_requests"]:
            scheduled[work["request_idx"]] += work["scheduled_tokens"]
    observer_path = trace_on / "scheduler_trace_observer_effect.json"
    return {
        "request_rows": len(requests),
        "metric_rows": len(metrics["per_request"]),
        "scheduler_step_rows": len(steps),
        # Synchronous scheduling schedules input+output-1 tokens per request;
        # vLLM 0.11.0 async scheduling (A800 canonical cases included)
        # schedules one extra decode token, i.e. a histogram of {1: 128}.
        "scheduled_minus_input_plus_output_minus_one": dict(sorted(Counter(
            scheduled[index] - (row.input_tokens + row.output_tokens - 1)
            for index, row in enumerate(requests)).items())),
        "finished_requests_cover_all": sorted(
            index for step in steps for index in step["finished_request_indexes"]
        ) == list(range(len(requests))),
        "scheduler_pipeline_depth": depth["scheduler_pipeline_depth"],
        "observer_effect": (
            json.loads(observer_path.read_text())["metric_delta_percent"]
            if observer_path.is_file() else "missing"),
    }


def _summary(output: Path) -> dict[str, object]:
    import csv

    from dataset_pipeline.serving_dataset.compressed_artifact import (
        read_compressed_canonical_bytes,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    with (output / "requests.csv").open(newline="") as stream:
        requests = list(csv.DictReader(stream))
    steps_entry = manifest["scheduler_steps"]
    steps = [
        json.loads(line) for line in read_compressed_canonical_bytes(
            output / steps_entry["path"],
            compressed_sha256=steps_entry["compressed_sha256"],
            content_sha256=steps_entry["sha256"],
            label="scheduler_steps",
        ).decode().splitlines()
    ]
    alignment_entry = manifest["alignment_steps"]
    alignment = json.loads(read_compressed_canonical_bytes(
        output / alignment_entry["path"],
        compressed_sha256=alignment_entry["compressed_sha256"],
        content_sha256=alignment_entry["sha256"],
        label="alignment_steps",
    ))
    scheduled: Counter[int] = Counter()
    for step in steps:
        for work in step["scheduled_requests"]:
            scheduled[work["request_idx"]] += work["scheduled_tokens"]
    conservation = dict(sorted(Counter(
        scheduled[index]
        - (int(row["In Token [tokens]"]) + int(row["Out Token [tokens]"]) - 1)
        for index, row in enumerate(requests)
    ).items()))
    finished = sorted(
        index for step in steps for index in step["finished_request_indexes"])
    return {
        "request_rows": len(requests),
        "metric_rows": len(metrics["per_request"]),
        "scheduler_step_rows": len(steps),
        "scheduled_minus_input_plus_output_minus_one": conservation,
        "finished_requests_cover_all": finished == list(range(len(requests))),
        "alignment_phase_status": {
            phase: entry["status"] for phase, entry in alignment["phases"].items()
        },
        "manifest_alignment_status": alignment_entry["status"],
        "observer_effect": manifest["capture"]["observer_effect_validation"],
        "runtime_version": manifest["runtime"]["runtime_version"],
        "scheduler_pipeline_depth": manifest["runtime"]["scheduler_pipeline_depth"],
        "capture_run_id": manifest["capture"]["run_id"],
        "accepted_source_relative_paths": {
            source["archive_id"]: source["relative_path"]
            for source in manifest["raw_sources"]
            if source["archive_id"] in _ACCEPTED_SOURCE_IDS
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
