"""Validate and publish accepted H200 missing-case NSYS and NCU evidence.

The original per-family canonical datasets remain the base.  NSYS candidates
only add case IDs absent from that base.  NCU adds non-duration sidecars to
three existing kernel records after an exact ordinal, geometry and normalized
demangled-name join.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any

import zstandard

from dataset_pipeline.kernel_dataset.dataset_writer import publish_complete_dataset
from dataset_pipeline.kernel_dataset.nvidia_h200 import (
    H200_FAMILIES, assemble_campaign_plan, h200_timing_marker)
from src.kernel.dataset_storage import (
    KERNEL_DATASET_SUFFIX, load_kernel_dataset_document,
    read_kernel_dataset_content)


_METRIC_UNITS = {
    "lts__t_sectors_op_read.sum": "sector",
    "lts__t_sectors_op_write.sum": "sector",
    "dram__bytes_read.sum": "byte",
    "dram__bytes_write.sum": "byte",
    "smsp__warps_eligible.avg.per_cycle_active": "warp",
    "smsp__warps_active.avg.per_cycle_active": "warp",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "%",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "%",
    "sm__cycles_active.avg": "cycle",
    "sm__cycles_elapsed.avg": "cycle",
}
_BASE_METRICS = frozenset(_METRIC_UNITS) - {
    "sm__cycles_active.avg", "sm__cycles_elapsed.avg"}
_INTEGER_METRICS = frozenset(name for name in _METRIC_UNITS
                             if name.startswith(("lts__", "dram__")))
_NCU_NVTX_COLUMN = (
    "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg")
_NCU_RANGE_PATTERN = re.compile(
    r"H200_KERNEL::[A-Za-z0-9_]+::[A-Za-z0-9_]+")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_raw_manifest(root: Path) -> str:
    """Verify every pinned raw byte before creating any publication output."""
    manifest_path = root / "raw_sha256.json"
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "files"}
            or manifest["schema_version"] != 1
            or not isinstance(manifest["files"], list)
            or not manifest["files"]):
        raise ValueError("H200 raw SHA manifest is invalid")
    pinned = set()
    for item in manifest["files"]:
        if not isinstance(item, dict) or set(item) != {
                "path", "size_bytes", "sha256"}:
            raise ValueError("H200 raw SHA entry has invalid fields")
        name = item["path"]
        if not isinstance(name, str):
            raise ValueError("H200 raw SHA entry path is invalid")
        relative = PurePosixPath(name)
        if (relative.is_absolute() or not relative.parts
                or str(relative) != name
                or any(part in (".", "..") for part in relative.parts)
                or relative.parts[0] == "publication"
                or name == "raw_sha256.json"
                or name in pinned):
            raise ValueError(f"H200 raw SHA path is unsafe or repeated: {name}")
        pinned.add(name)
        size = item["size_bytes"]
        digest = item["sha256"]
        if (type(size) is not int or size < 0
                or not isinstance(digest, str)
                or _SHA256_PATTERN.fullmatch(digest) is None):
            raise ValueError(f"H200 raw SHA metadata is invalid: {name}")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"H200 pinned raw file is missing or linked: {name}")
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ValueError(f"H200 pinned raw file changed: {name}")
    present = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "raw_sha256.json"
        and path.relative_to(root).parts[0] != "publication"}
    if present != pinned:
        raise ValueError(
            "H200 raw SHA manifest does not cover the campaign: "
            f"missing={sorted(present - pinned)[:5]}, "
            f"stale={sorted(pinned - present)[:5]}")
    return _sha256(manifest_path)


def _triplet(value: str) -> tuple[int, int, int]:
    result = ast.literal_eval(value)
    if (not isinstance(result, tuple) or len(result) != 3
            or any(type(item) is not int or item <= 0 for item in result)):
        raise ValueError(f"invalid launch dimensions: {value!r}")
    return result


def _demangled_name(name: str) -> str:
    """Erase only type and namespace spellings that NSYS/NCU print differently."""
    for token in ("(int)", "(bool)", "(unsigned long)"):
        name = name.replace(token, "")
    return (name.replace("cutlass::arch::Sm90", "cutlass::Sm90")
            .replace("at::native::", "at::"))


def _load_complete_candidate(
        root: Path, batch_name: str, batch: dict[str, Any],
        scratch: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    plan_path = root / "plans" / Path(batch["plan_path"]).name
    if _sha256(plan_path) != batch["plan_sha256"]:
        raise ValueError(f"{batch_name} plan SHA mismatch")
    result_root = root / "batches" / batch_name / "runs" / plan_path.stem
    status_path = root / "batches" / batch_name / "runs" / "transfer_status.json"
    if json.loads(status_path.read_text()) != {"kind": "complete"}:
        raise ValueError(f"{batch_name} transfer is incomplete")
    assembled = assemble_campaign_plan(
        plan_path=plan_path, results_root=result_root,
        output_dir=scratch / batch_name)
    family = batch["family"]
    manifest = assembled[family]
    original_dir = root / "batches" / batch_name / "candidates"
    original_manifest = json.loads(
        (original_dir / f"{family}.manifest.json").read_text())
    if (manifest["case_status_counts"] != original_manifest["case_status_counts"]
            or manifest["measured_case_count"] !=
            original_manifest["measured_case_count"]):
        raise ValueError(f"{batch_name} candidate disagrees with raw reassembly")
    candidate_path = scratch / batch_name / f"{family}{KERNEL_DATASET_SUFFIX}"
    original_path = original_dir / f"{family}{KERNEL_DATASET_SUFFIX}"
    # The captured candidate remains byte-pinned by raw_sha256.json.  A
    # rebuilt zstd frame can differ across compressor versions even when the
    # uncompressed JSON bytes are identical; compare those exact bytes here.
    if (read_kernel_dataset_content(candidate_path)
            != read_kernel_dataset_content(original_path)):
        raise ValueError(f"{batch_name} candidate content disagrees with raw")
    document = load_kernel_dataset_document(candidate_path)
    if (manifest["vllm_stack"] != "vllm030"
            or manifest["evaluation_supported"] is not True
            or set(document) - set(batch["case_ids"])):
        raise ValueError(f"{batch_name} has an invalid stack or case set")
    if not manifest["run_provenance"]:
        raise ValueError(f"{batch_name} has no run provenance")
    for provenance in manifest["run_provenance"]:
        versions = provenance["software_identity"]
        if (versions["vllm"] != "0.30.0+cu129"
                or versions["torch"] != "2.13.0+cu129"
                or versions["torch_cuda"] != "12.9"):
            raise ValueError(f"{batch_name} run software differs from old H200")
    for case_id, entry in document.items():
        signature = [[item["Metadata"]["Kernel Name"],
                      list(_triplet(item["Metadata"]["Grid Size"])),
                      list(_triplet(item["Metadata"]["Block Size"]))]
                     for item in entry["kernels"]]
        if signature != batch["expected_launch_signatures"][case_id]["signature"]:
            raise ValueError(f"{batch_name}/{case_id} launch changed")
    return family, document, {
        "batch": batch_name, "family": family, "case_ids": sorted(document),
        "candidate_sha256": _sha256(original_path),
        "manifest_sha256": _sha256(original_dir / f"{family}.manifest.json"),
    }


def _ncu_rows(
        source: Path, expected_count: int,
        metric_units: dict[str, str],
        expected_marker: str) -> list[dict]:
    text = source.read_text()
    if "No kernels were profiled" in text:
        raise ValueError(f"{source} contains no profiled kernels")
    reader = csv.DictReader(
        io.StringIO("".join(line for line in text.splitlines(keepends=True)
                            if line.startswith('"'))))
    headers = reader.fieldnames
    if (headers is None or len(headers) != len(set(headers))
            or _NCU_NVTX_COLUMN not in headers
            or any(headers.count(metric) != 1 for metric in metric_units)):
        raise ValueError(f"{source} has missing or duplicate NCU columns")
    records = list(reader)
    if len(records) != expected_count + 1:
        raise ValueError(f"{source} has {len(records) - 1} NCU rows; "
                         f"expected {expected_count}")
    units, *rows = records
    for metric, unit in metric_units.items():
        if metric not in units or units[metric] != unit:
            raise ValueError(f"{source} metric {metric} unit changed")
    if [int(row["ID"]) for row in rows] != list(range(expected_count)):
        raise ValueError(f"{source} kernel ordinals changed")
    for ordinal, row in enumerate(rows):
        ranges = _NCU_RANGE_PATTERN.findall(row[_NCU_NVTX_COLUMN])
        if ranges != [expected_marker]:
            raise ValueError(
                f"{source} kernel {ordinal} NVTX range differs from "
                f"{expected_marker}: {ranges}")
    return rows


def _ncu_sidecar(
        root: Path, specification: dict[str, Any],
        entry: dict[str, Any]) -> list[dict[str, Any]]:
    case_id = specification["case_id"]
    source = root / "ncu_raw" / specification["attempt"] / case_id
    if json.loads((source / "transfer_status.json").read_text()) != {
            "kind": "complete"}:
        raise ValueError(f"{case_id} NCU transfer is incomplete")
    command = json.loads((source / "ncu_command.json").read_text())
    if command["returncode"] != 0:
        raise ValueError(f"{case_id} NCU failed")
    ncu_plan = json.loads((root / specification["plan_name"]).read_text())
    captured_plan = source / "metric_plan.json"
    if captured_plan.is_file() and json.loads(
            captured_plan.read_text()) != ncu_plan:
        raise ValueError(f"{case_id} captured NCU metric plan changed")
    metric_names = ncu_plan["traffic_metrics"] + ncu_plan["service_metrics"]
    if (set(metric_names) not in
            (_BASE_METRICS, frozenset(_METRIC_UNITS))
            or len(metric_names) != len(set(metric_names))):
        raise ValueError(f"{case_id} NCU metric plan is unsupported")
    metric_units = {name: _METRIC_UNITS[name] for name in metric_names}
    planned = [item for item in ncu_plan["cases"]
               if item["case_id"] == case_id]
    if len(planned) != 1 or planned[0]["family"] != specification["family"]:
        raise ValueError(f"{case_id} is absent from the pinned NCU plan")
    expected_command = planned[0]["plan"]
    if expected_command["marker"] != h200_timing_marker(
            specification["family"], case_id):
        raise ValueError(f"{case_id} NCU plan marker differs from workload")
    preflight_plan = json.loads((source / "preflight_plan.json").read_text())
    if preflight_plan["plans"] != [expected_command]:
        raise ValueError(f"{case_id} NCU preflight workload changed")
    argv = command["command"]
    marker = expected_command["marker"]
    if (argv[argv.index("--nvtx-include") + 1] != marker + "/"
            or argv[argv.index("--metrics") + 1] != ",".join(metric_names)):
        raise ValueError(f"{case_id} NCU range or metric list changed")
    runner = expected_command["runner_relpath"]
    runner_index = next((index for index, value in enumerate(argv)
                         if value.endswith(runner)), None)
    if runner_index is None or argv[runner_index + 1:
                                    runner_index + 1 + len(
                                        expected_command["runner_args"])] != (
                                            expected_command["runner_args"]):
        raise ValueError(f"{case_id} NCU runner arguments changed")
    if (command["identity"]["versions"]["vllm"] != "0.30.0+cu129"
            or command["identity"]["versions"]["torch"] != "2.13.0+cu129"
            or command["identity"]["versions"]["torch_cuda"] != "12.9"
            or not command["identity"]["gpu"].startswith("NVIDIA H200,")):
        raise ValueError(f"{case_id} NCU hardware or stack changed")
    profile = source / specification["report_name"]
    if profile.suffix == ".zst":
        report = zstandard.ZstdDecompressor().decompress(
            profile.read_bytes(), max_output_size=100_000_000)
        if len(report) == 0:
            raise ValueError(f"{profile} is empty")
    elif profile.suffix == ".ncu-rep":
        if profile.stat().st_size == 0:
            raise ValueError(f"{profile} is empty")
    else:
        raise ValueError(f"unexpected NCU report suffix: {profile}")
    csv_path = source / "ncu_stdout.csv"
    rows = _ncu_rows(
        csv_path, len(entry["kernels"]), metric_units, marker)
    preflight = source / "preflight" / "cases" / case_id / "window_01.launch.json"
    observed = json.loads(preflight.read_text())
    preflight_signature = [
        [item["name"], tuple(item["grid"]), tuple(item["block"])]
        for item in observed["launches"]]
    canonical_signature = [
        [item["Metadata"]["Kernel Name"],
         _triplet(item["Metadata"]["Grid Size"]),
         _triplet(item["Metadata"]["Block Size"])]
        for item in entry["kernels"]]
    if (not observed["telemetry"]["clock_accepted"]
            or preflight_signature != canonical_signature):
        raise ValueError(f"{case_id} NCU preflight differs from canonical")
    attached = []
    for index, (row, kernel) in enumerate(zip(
            rows, entry["kernels"], strict=True)):
        meta = kernel["Metadata"]
        if (_demangled_name(row["Kernel Name"]) !=
                _demangled_name(meta["Kernel Name"])
                or _triplet(row["Grid Size"]) != _triplet(meta["Grid Size"])
                or _triplet(row["Block Size"]) != _triplet(meta["Block Size"])):
            raise ValueError(f"{case_id} NCU/NSYS kernel {index} mismatch")
        values = {}
        for metric in metric_names:
            raw_value = row[metric]
            if metric in _INTEGER_METRICS:
                value = int(raw_value)
            else:
                value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{case_id}/{metric} counter is invalid")
            if metric_units[metric] == "%" and value > 100:
                raise ValueError(f"{case_id}/{metric} percentage exceeds 100")
            values[metric] = value
        if "sm__cycles_active.avg" in values:
            if not 0 < values["sm__cycles_active.avg"] <= (
                    values["sm__cycles_elapsed.avg"]):
                raise ValueError(
                    f"{case_id} NCU active/elapsed cycles are invalid")
        attached.append({
            "Schema Version": 1,
            "Source": "ncu_raw_page",
            "Metrics": values,
            "Units": metric_units,
            "Provenance": {
                "campaign": root.name,
                "case_id": case_id,
                "kernel_index": index,
                "ncu_kernel_name": row["Kernel Name"],
                "csv_sha256": _sha256(csv_path),
                "report_sha256": _sha256(profile),
                "report_name": profile.name,
                "app_id": specification["app_id"],
                "ncu_version": command["identity"]["profiler_versions"]["ncu"],
            },
        })
    return attached


def publish_missing_h200_evidence(
        *, campaign_root: Path, canonical_dir: Path,
        publication_dir: Path) -> dict[str, Any]:
    """Validate all sources before writing any canonical family."""
    raw_manifest_sha256 = _validate_raw_manifest(campaign_root)
    plan = json.loads((campaign_root / "plan_manifest.json").read_text())
    selection = json.loads((campaign_root / "publication_plan.json").read_text())
    if selection["schema_version"] != 1:
        raise ValueError("unknown H200 publication plan version")
    if len(selection["batches"]) != len(set(selection["batches"])):
        raise ValueError("H200 publication plan repeats a batch")
    usage = json.loads((campaign_root / "modal_usage.json").read_text())
    if (usage["conservative_gpu_seconds_used"] > usage["authorization_gpu_seconds"]
            or not usage["apps"]):
        raise ValueError("Modal usage record exceeds authorization or is incomplete")
    ncu_app_ids = {item["app_id"] for item in usage["apps"]
                   if item["purpose"] == "midlayer-h200-ncu-subset-20261030"}
    if any(item["app_id"] not in ncu_app_ids
           for item in selection["ncu_cases"]):
        raise ValueError("NCU publication references an unknown Modal app")
    publication_dir.mkdir(parents=True, exist_ok=True)
    scratch = publication_dir / "reassembled"
    canonical = {}
    for family in H200_FAMILIES:
        path = canonical_dir / f"{family}{KERNEL_DATASET_SUFFIX}"
        if _sha256(path) != plan["families"][family]["h200_canonical_sha256"]:
            raise ValueError(f"{family} canonical base changed")
        canonical[family] = load_kernel_dataset_document(path)
    additions = {family: {} for family in H200_FAMILIES}
    sources = []
    for batch_name in selection["batches"]:
        if batch_name not in plan["batches"]:
            raise ValueError(f"unknown selected batch {batch_name}")
        family, document, source = _load_complete_candidate(
            campaign_root, batch_name, plan["batches"][batch_name], scratch)
        duplicates = set(document) & (set(canonical[family])
                                      | set(additions[family]))
        if duplicates:
            raise ValueError(f"{family} candidate collides with canonical: "
                             f"{sorted(duplicates)}")
        if set(document) - set(plan["families"][family]["case_ids"]):
            raise ValueError(f"{family} candidate is outside the 39-case plan")
        additions[family].update({
            case_id: {"op_info": entry["op_info"], "kernels": entry["kernels"]}
            for case_id, entry in document.items()})
        sources.append(source)
    if sum(map(len, additions.values())) != 24:
        raise ValueError("accepted missing-case count changed")
    ncu_sources = []
    for specification in selection["ncu_cases"]:
        family = specification["family"]
        case_id = specification["case_id"]
        if family not in canonical or case_id not in canonical[family]:
            raise ValueError(f"NCU case {case_id} is not existing canonical")
        entry = canonical[family][case_id]
        sidecars = _ncu_sidecar(campaign_root, specification, entry)
        for kernel, sidecar in zip(entry["kernels"], sidecars, strict=True):
            if "NCU Counters" in kernel["Sections"]:
                raise ValueError(f"{case_id} already has NCU counters")
            kernel["Sections"]["NCU Counters"] = sidecar
        ncu_sources.append({"family": family, "case_id": case_id,
                            "app_id": specification["app_id"],
                            "kernel_count": len(sidecars)})
    if len(ncu_sources) != 3:
        raise ValueError("expected three NCU sidecar cases")
    merged = {}
    for family in H200_FAMILIES:
        merged[family] = {**canonical[family], **additions[family]}
    # A late non-finite value must not leave earlier family files published.
    for family in H200_FAMILIES:
        json.dumps(merged[family], ensure_ascii=False, allow_nan=False)
    changed_families = {
        family for family in H200_FAMILIES
        if additions[family] or any(item["family"] == family
                                    for item in ncu_sources)}
    record = {
        "schema_version": 1,
        "artifact_kind": "nvidia_h200_missing39_partial_publication",
        "campaign": campaign_root.name,
        "accepted_added_cases": {family: sorted(additions[family])
                                 for family in H200_FAMILIES},
        "unaccepted_case_ids": {
            family: sorted(set(plan["families"][family]["case_ids"])
                           - set(additions[family]))
            for family in H200_FAMILIES},
        "ncu_sources": ncu_sources, "candidate_sources": sources,
        "modal_usage_sha256": _sha256(campaign_root / "modal_usage.json"),
        "raw_manifest_sha256": raw_manifest_sha256,
    }
    for family in H200_FAMILIES:
        if family not in changed_families:
            continue
        path = canonical_dir / f"{family}{KERNEL_DATASET_SUFFIX}"
        publish_complete_dataset(
            merged[family], tuple(sorted(merged[family])),
            str(publication_dir / f"{family}.json"), str(path))
    record["canonical_sha256"] = {
        family: _sha256(canonical_dir / f"{family}{KERNEL_DATASET_SUFFIX}")
        for family in H200_FAMILIES}
    (publication_dir / "publication.json").write_text(
        json.dumps(record, indent=2) + "\n")
    return record
