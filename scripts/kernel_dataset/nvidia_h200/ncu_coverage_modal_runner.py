"""Capture pinned non-duration NCU evidence for one bounded H200 batch."""

from __future__ import annotations

import hashlib
import importlib.metadata
import csv
import json
import math
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import os
import re

import modal

_MODULE_PATH = Path(__file__).resolve()
_REMOTE_MODULE = (len(_MODULE_PATH.parents) < 4
                  or Path("/opt/midlayer/scripts").is_dir())
ROOT = Path("/opt/midlayer") if _REMOTE_MODULE else _MODULE_PATH.parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.profiler.nvidia_h200_ncu_plan import (  # noqa: E402
    MISSING15_REFERENCE_KIND,
    admitted_batch_plan,
    expected_launch_signature,
    normalized_profiler_kernel_name,
    preflight_cluster_facts,
)

CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_1027_20261031"
DIAGNOSTIC = (ROOT / "profiler_output/kernel_dataset/nvidia_h200/"
              "ncu_clock_diagnostic_20261102")
PROBE_GATE_ENV = "MIDLAYER_H200_PAID_PROBE_CONTROLLER"

if _REMOTE_MODULE:
    IMAGE = modal.Image.debian_slim()
else:
    IMAGE = (
        modal.Image.from_registry(
            "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
        .entrypoint([])
        .apt_install("nsight-systems-2025.6.3", "cuda-nsight-compute-12-9")
        .run_commands(
            "python -m pip install --no-cache-dir "
            "'torch==2.13.0+cu129' 'vllm==0.30.0+cu129' "
            "'triton==3.7.1' 'flashinfer-python==0.6.18.post1' "
            "pandas tqdm zstandard "
            "--extra-index-url https://download.pytorch.org/whl/cu129 "
            "--extra-index-url https://wheels.vllm.ai/0.30.0/cu129/"
        )
        .run_commands(
            "python -m pip install --no-cache-dir --force-reinstall --no-deps "
            "'torchcodec==0.16.0+cu129' "
            "--index-url https://download.pytorch.org/whl/cu129"
        )
        .add_local_dir(ROOT / "scripts", remote_path="/opt/midlayer/scripts")
        .add_local_dir(ROOT / "dataset_pipeline",
                       remote_path="/opt/midlayer/dataset_pipeline")
        .add_local_dir(ROOT / "evaluation", remote_path="/opt/midlayer/evaluation")
        .add_local_dir(ROOT / "src", remote_path="/opt/midlayer/src")
        .add_local_dir(CAMPAIGN / "plans", remote_path="/opt/midlayer/plans")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stream_files(root: Path, sent: set[str]):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "gpu.lock":
            continue
        relative = str(path.relative_to(root))
        if relative in sent:
            continue
        sent.add(relative)
        yield {"kind": "file_start", "path": relative,
               "size": path.stat().st_size, "sha256": _sha256(path)}
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                yield {"kind": "file_chunk", "path": relative,
                       "data": chunk}
        yield {"kind": "file_end", "path": relative}


def _check_identity():
    import torch

    versions = {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "vllm": importlib.metadata.version("vllm"),
        "triton": importlib.metadata.version("triton"),
        "flashinfer_python": importlib.metadata.version("flashinfer-python"),
        "cublas": importlib.metadata.version("nvidia-cublas-cu12"),
    }
    if versions != {
            "torch": "2.13.0+cu129", "torch_cuda": "12.9",
            "vllm": "0.30.0+cu129", "triton": "3.7.1",
            "flashinfer_python": "0.6.18.post1", "cublas": "12.9.1.4"}:
        raise ValueError(f"software differs from old campaign: {versions}")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version",
         "--format=csv,noheader"], check=True, capture_output=True,
        text=True, timeout=20).stdout.strip()
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError(f"expected one H200: {gpu}")
    tools = {}
    for name in ("nsys", "ncu"):
        path = shutil.which(name)
        if path is None:
            raise ValueError(f"missing profiler: {name}")
        tools[name] = subprocess.run(
            [path, "--version"], check=True, capture_output=True,
            text=True, timeout=20).stdout.strip()
    return {"gpu": gpu, "versions": versions, "profiler_versions": tools}


def _verify_launch(case: dict, window_path: Path, *, diagnostic_only: bool = False) -> None:
    window = json.loads(window_path.read_text())
    if not diagnostic_only and not window["telemetry"]["clock_accepted"]:
        raise ValueError(f"{case['case_id']} preflight clock rejected")
    actual = [[launch["name"], launch["grid"], launch["block"]]
              for launch in window["launches"]]
    expected = expected_launch_signature(case)
    if actual != expected:
        raise ValueError(f"{case['case_id']} launch differs from pinned reference")
    if "reference_kind" in case:
        if case["reference_kind"] != MISSING15_REFERENCE_KIND:
            raise ValueError("unknown NCU launch reference")
        old = case["reference_launch_signature"]
        clusters = json.loads((window_path.parent.parent.parent.parent /
                               "cluster_preflight.json").read_text())
        if len(window["launches"]) != len(old) or len(clusters["launches"]) != len(old):
            raise ValueError("missing15 preflight kernel count differs")
        for observed, prior, fact in zip(window["launches"], old,
                                         clusters["launches"]):
            if (observed["dynamic_shared_memory_bytes"]
                    != prior["dynamic_shared_memory_bytes"]
                    or fact["name"] != prior["name"]
                    or fact["grid"] != prior["grid"]
                    or fact["block"] != prior["block"]
                    or fact["dynamic_shared_memory_bytes"]
                    != prior["dynamic_shared_memory_bytes"]
                    or (prior["cluster"] is not None
                        and fact["cluster"] != prior["cluster"])):
                raise ValueError("missing15 preflight differs from pinned old NSYS")


def _validate_ncu_capture(case: dict, metrics: list[str], target: Path,
                          *, diagnostic_only: bool = False) -> dict:
    telemetry = json.loads((target / "ncu_telemetry.json").read_text())
    if not diagnostic_only and not telemetry["clock_accepted"]:
        return {"status": "ncu_clock_rejected"}
    with (target / "ncu_stdout.csv").open(newline="") as source:
        rows = list(csv.reader(source))
    header_indices = [index for index, row in enumerate(rows)
                      if "Kernel Name" in row and all(metric in row for metric in metrics)]
    if len(header_indices) != 1:
        return {"status": "ncu_metrics_invalid", "reason": "metric_header"}
    header_index = header_indices[0]
    header = rows[header_index]
    observations = [dict(zip(header, row)) for row in rows[header_index + 2:]
                    if len(row) == len(header)]
    expected_launches = expected_launch_signature(case)
    if len(observations) != len(expected_launches):
        return {"status": "ncu_metrics_invalid", "reason": "kernel_count"}
    cluster_facts = None
    if "reference_kind" in case:
        if case["reference_kind"] != MISSING15_REFERENCE_KIND:
            raise ValueError("unknown NCU launch reference")
        cluster_path = target / "cluster_preflight.json"
        if not cluster_path.is_file():
            return {"status": "ncu_launch_mismatch", "reason": "cluster_evidence_missing"}
        cluster_facts = json.loads(cluster_path.read_text())["launches"]
        if len(cluster_facts) != len(expected_launches):
            return {"status": "ncu_launch_mismatch", "reason": "cluster_count"}
    for ordinal, (observation, expected) in enumerate(zip(
            observations, expected_launches)):
        grid = [int(part.strip()) for part in observation["Grid Size"].strip("()").split(",")]
        block = [int(part.strip()) for part in observation["Block Size"].strip("()").split(",")]
        if grid != expected[1] or block != expected[2]:
            return {"status": "ncu_metrics_invalid", "reason": "launch_geometry"}
        if cluster_facts is not None:
            fact = cluster_facts[ordinal]
            cluster_keys = [f"launch__cluster_dim_{axis}" for axis in "xyz"]
            shared_key = "launch__shared_mem_per_block_dynamic"
            if any(key not in observation for key in (*cluster_keys, shared_key)):
                return {"status": "ncu_launch_mismatch", "reason": "cluster_fields"}
            try:
                cluster = [int(observation[key]) for key in cluster_keys]
                dynamic_shared = int(observation[shared_key])
            except ValueError:
                return {"status": "ncu_launch_mismatch", "reason": "cluster_fields"}
            if (normalized_profiler_kernel_name(observation["Kernel Name"])
                    != normalized_profiler_kernel_name(fact["name"])
                    or cluster != fact["cluster"]
                    or dynamic_shared != fact["dynamic_shared_memory_bytes"]):
                return {"status": "ncu_launch_mismatch", "reason": "kernel_identity"}
        if not any(case["plan"]["marker"] in observation[field]
                   for field in header if "Range" in field):
            return {"status": "ncu_metrics_invalid", "reason": "nvtx_range"}
        values = {}
        for metric in metrics:
            try:
                value = float(observation[metric])
            except ValueError:
                return {"status": "ncu_metrics_invalid", "reason": metric}
            if not math.isfinite(value) or value < 0:
                return {"status": "ncu_metrics_invalid", "reason": metric}
            values[metric] = value
        active = values["sm__cycles_active.avg"]
        elapsed = values["sm__cycles_elapsed.avg"]
        if elapsed <= 0 or active > elapsed:
            return {"status": "ncu_metrics_invalid", "reason": "cycles"}
    result = {"status": (
        "diagnostic_clock_rejected" if diagnostic_only and not telemetry["clock_accepted"]
        else "diagnostic_clock_accepted" if diagnostic_only else "complete"),
        "kernel_count": len(observations)}
    if cluster_facts is not None:
        result["same_run_cluster_verified"] = True
    return result


APP = modal.App("midlayer-h200-ncu-coverage-20261031")
REMOTE = Path("/opt/midlayer")


@APP.function(gpu="H200", image=IMAGE, timeout=650,
              max_containers=1, single_use_containers=True)
def capture(batch: dict, plan_sha256: str):
    identity = _check_identity()
    diagnostic_only = (batch["diagnostic_only"] is True
                       if "diagnostic_only" in batch else False)
    source_counters = (batch["source_counters"] is True
                       if "source_counters" in batch else False)
    if source_counters and not diagnostic_only:
        raise ValueError("source counters require a raw-only diagnostic plan")
    yield {"kind": "identity", "data": identity,
           "plan_sha256": plan_sha256}
    deadline = time.monotonic() + 600
    with tempfile.TemporaryDirectory(prefix="midlayer-ncu-coverage-", dir="/tmp") as tmp:
        root = Path(tmp)
        sent = set()
        for case in batch["cases"]:
            case_id = case["case_id"]
            if time.monotonic() + 270 > deadline:
                yield {"kind": "partial", "reason": "batch_time_limit",
                       "next_case_id": case_id}
                return
            target = root / "batches" / batch["batch_name"] / "ncu_raw" / case_id
            target.mkdir(parents=True)
            plan_path = target / "preflight_plan.json"
            plan_path.write_text(json.dumps({
                "schema_version": 1, "hardware_id": "nvidia_h200",
                "campaign": "ncu_preflight_" + case_id,
                "case_count": 1, "plans": [case["plan"]],
            }, indent=2) + "\n")
            preflight = target / "preflight"
            preflight_cmd = [
                sys.executable,
                str(REMOTE / "scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py"),
                "--plans", str(plan_path), "--output-dir", str(preflight),
                "--window-ids", "1", "--concurrency", "1",
                "--physical-gpu", "0", "--nsys", shutil.which("nsys"),
                "--python", sys.executable, "--gpu-lock", str(root / "gpu.lock"),
            ]
            missing15 = "reference_kind" in case
            if missing15:
                if case["reference_kind"] != MISSING15_REFERENCE_KIND:
                    raise ValueError("unknown NCU launch reference")
            preflight_run = subprocess.run(
                preflight_cmd, capture_output=True, text=True, timeout=90)
            (target / "preflight_command.json").write_text(json.dumps({
                "command": preflight_cmd, "returncode": preflight_run.returncode,
                "stdout": preflight_run.stdout, "stderr": preflight_run.stderr,
            }, indent=2) + "\n")
            window = preflight / "cases" / case_id / "window_01.launch.json"
            cluster_failure = None
            if missing15:
                sqlite_path = window.with_name("window_01.sqlite")
                report_path = window.with_name("window_01.nsys-rep")
                if preflight_run.returncode == 0 and report_path.is_file():
                    export_cmd = [shutil.which("nsys"), "export", "--type=sqlite",
                                  "-o", str(sqlite_path), str(report_path)]
                    export = subprocess.run(export_cmd, capture_output=True,
                                            text=True, timeout=90)
                    (target / "sqlite_export_command.json").write_text(json.dumps({
                        "command": export_cmd, "returncode": export.returncode,
                        "stdout": export.stdout, "stderr": export.stderr,
                    }, indent=2) + "\n")
                if (preflight_run.returncode == 0 and window.is_file()
                        and sqlite_path.is_file() and report_path.is_file()):
                    try:
                        facts = preflight_cluster_facts(sqlite_path, window)
                        (target / "cluster_preflight.json").write_text(json.dumps({
                            "schema_version": 1,
                            "kind": "same_run_nsys_cluster_facts",
                            "case_id": case_id,
                            "nsys_report_sha256": _sha256(report_path),
                            "launches": facts,
                        }, indent=2) + "\n")
                    except (ValueError, sqlite3.DatabaseError) as error:
                        cluster_failure = str(error)
                        (target / "cluster_preflight_failure.json").write_text(
                            json.dumps({"reason": cluster_failure}, indent=2) + "\n")
                elif preflight_run.returncode == 0:
                    cluster_failure = "preflight NSYS SQLite or launch is missing"
                    (target / "cluster_preflight_failure.json").write_text(
                        json.dumps({"reason": cluster_failure}, indent=2) + "\n")
                if sqlite_path.is_file():
                    if (preflight_run.returncode == 0 and window.is_file()
                            and report_path.is_file()):
                        import zstandard
                        compressed = sqlite_path.with_suffix(".sqlite.zst")
                        with sqlite_path.open("rb") as source, compressed.open("wb") as dest:
                            zstandard.ZstdCompressor(level=3).copy_stream(source, dest)
                        cluster_path = target / "cluster_preflight.json"
                        if cluster_path.is_file():
                            cluster_payload = json.loads(cluster_path.read_text())
                            cluster_payload["sqlite_zst_sha256"] = _sha256(compressed)
                            cluster_payload["sqlite_zst_size"] = compressed.stat().st_size
                            cluster_path.write_text(json.dumps(cluster_payload, indent=2) + "\n")
                    sqlite_path.unlink()  # Derived from retained .nsys-rep.
            yield from _stream_files(root, sent)
            if preflight_run.returncode != 0:
                yield {"kind": "case_status", "case_id": case_id,
                       "status": "preflight_runner_failed"}
                continue
            if not window.is_file():
                yield {"kind": "case_status", "case_id": case_id,
                       "status": "preflight_window_missing"}
                continue
            if (not diagnostic_only
                    and not json.loads(window.read_text())["telemetry"]["clock_accepted"]):
                yield {"kind": "case_status", "case_id": case_id,
                       "status": "preflight_clock_rejected"}
                continue
            if cluster_failure is not None:
                yield {"kind": "case_status", "case_id": case_id,
                       "status": "preflight_cluster_invalid"}
                continue
            try:
                _verify_launch(case, window, diagnostic_only=diagnostic_only)
            except ValueError:
                yield {"kind": "case_status", "case_id": case_id,
                       "status": "launch_mismatch"}
                continue
            marker = case["plan"]["marker"]
            metrics = batch["traffic_metrics"] + batch["service_metrics"]
            ncu_cmd = [
                shutil.which("ncu"), "--nvtx", "--nvtx-include", marker + "/",
                "--clock-control", "none", "--cache-control", "none",
            ]
            if source_counters:
                ncu_cmd.extend(["--section", "SourceCounters"])
            ncu_cmd.extend([
                "--csv", "--page", "raw", "--metrics", ",".join(metrics),
                "--export", str(target / "profile"), sys.executable,
                str(REMOTE / case["plan"]["runner_relpath"]),
                *case["plan"]["runner_args"],
                "--h200-gpu-lock", str(root / "gpu.lock"),
                "--h200-telemetry-output", str(target / "ncu_telemetry.json"),
                "--h200-physical-gpu", "0",
            ])
            result = subprocess.run(
                ncu_cmd, capture_output=True, timeout=180)
            (target / "ncu_stdout.csv").write_bytes(result.stdout)
            (target / "ncu_stderr.txt").write_bytes(result.stderr)
            (target / "ncu_command.json").write_text(json.dumps({
                "command": ncu_cmd, "returncode": result.returncode,
                "identity": identity, "plan_sha256": plan_sha256,
            }, indent=2) + "\n")
            report = target / "profile.ncu-rep"
            if result.returncode == 0 and report.is_file():
                import zstandard
                compressed = target / "profile.ncu-rep.zst"
                with report.open("rb") as source, compressed.open("wb") as dest:
                    zstandard.ZstdCompressor(level=3).copy_stream(source, dest)
                report.unlink()
            if result.returncode == 0 and (target / "profile.ncu-rep.zst").is_file():
                validation = _validate_ncu_capture(
                    case, metrics, target, diagnostic_only=diagnostic_only)
            else:
                validation = {"status": "ncu_failed"}
            (target / "validation.json").write_text(
                json.dumps(validation, indent=2) + "\n")
            yield from _stream_files(root, sent)
            yield {"kind": "case_status", "case_id": case_id,
                   "status": validation["status"]}
        yield {"kind": "complete"}


def _diagnostic_capture(output_dir: str) -> None:
    """Collect source-matched counters as raw-only diagnostic evidence."""

    if os.environ[PROBE_GATE_ENV] != "1":
        raise PermissionError("paid NCU diagnostic requires the spend controller")
    target = Path(output_dir).resolve()
    if (target.parent != DIAGNOSTIC.resolve()
            or re.fullmatch(r"attempt_[A-Za-z0-9_]+", target.name) is None
            or target.exists()):
        raise ValueError("NCU diagnostic output must be a new registered attempt")
    legacy_plan = target.name == "attempt_1_capture"
    plan_path = (DIAGNOSTIC / "plan.json" if legacy_plan else
                 DIAGNOSTIC / "plans" / f"{target.name}.json")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    admitted_cases = {
        "FUSEDMOE_TOK32768_E256_TOPK8_H7168_INT2048",
        "FUSEDMOE_TOK32768_E8_TOPK2_H6144_INT16384",
        "GEMM_B8_M4096_K28672_N8192",
        "FLASHATTN_PAGED_PREFILL_B128_S2048_HQ32_HKV8_D128_C1",
        "FLASHATTN_PAGED_PREFILL_B128_S4096_HQ32_HKV8_D128_C1",
        "FLASHATTN_PREFILL_B16_S4096_HQ32_HKV8_D128_C1",
        "FUSEDMOE_TOK32768_E128_TOPK16_H2048_INT768",
    }
    if plan["schema_version"] == 1:
        source_counters = False
    elif plan["schema_version"] == 2:
        source_counters = plan["source_counters"] is True
        if (not source_counters
                or plan["case_id"] != "FUSEDMOE_TOK32768_E128_TOPK16_H2048_INT768"):
            raise ValueError("NCU SourceCounters diagnostic domain differs")
    else:
        raise ValueError("unknown NCU diagnostic plan schema")
    if (plan["kind"] != "h200_ncu_clock_diagnostic_raw_only"
            or plan["status"] != "prepared_not_executed"
            or plan["case_id"] not in admitted_cases):
        raise ValueError("NCU diagnostic plan identity differs")
    source_pin = plan["source_plan"]
    source_path = ROOT / source_pin["path"]
    if (not source_path.resolve().is_relative_to(
            (CAMPAIGN / "plans/candidates").resolve())
            or not source_path.is_file()
            or source_path.stat().st_size != source_pin["size"]
            or _sha256(source_path) != source_pin["sha256"]):
        raise ValueError("NCU diagnostic source plan pin differs")
    source = json.loads(source_path.read_text())
    matches = [case for case in source["cases"]
               if case["case_id"] == plan["case_id"]]
    if len(matches) != 1:
        raise ValueError("NCU diagnostic case is absent or duplicated")
    batch = {"schema_version": 1,
             "batch_name": "ncu_clock_diagnostic_" + target.name,
             "diagnostic_only": True, "cases": matches,
             "traffic_metrics": source["traffic_metrics"],
             "service_metrics": source["service_metrics"]}
    if source_counters:
        batch["source_counters"] = True
    open_files = {}
    statuses = {}
    complete = False
    for event in capture.remote_gen(batch, _sha256(plan_path)):
        kind = event["kind"]
        if kind == "identity":
            target.mkdir(parents=True)
            (target / "identity.json").write_text(json.dumps(event, indent=2) + "\n")
        elif kind == "file_start":
            relative = Path(event["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("NCU diagnostic remote path is invalid")
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            open_files[event["path"]] = (
                path.open("xb"), event["sha256"], event["size"])
        elif kind == "file_chunk":
            open_files[event["path"]][0].write(event["data"])
        elif kind == "file_end":
            handle, digest, size = open_files.pop(event["path"])
            handle.close()
            path = target / event["path"]
            if path.stat().st_size != size or _sha256(path) != digest:
                raise ValueError("NCU diagnostic transfer pin differs")
        elif kind == "case_status":
            if event["case_id"] != plan["case_id"] or statuses:
                raise ValueError("NCU diagnostic status coverage differs")
            statuses[event["case_id"]] = event["status"]
        elif kind == "complete":
            complete = True
        elif kind == "partial":
            raise ValueError("NCU diagnostic exceeded its batch time limit")
        else:
            raise ValueError(f"unknown NCU diagnostic event: {kind}")
    if open_files or not complete or set(statuses) != {plan["case_id"]}:
        raise ValueError("NCU diagnostic transfer incomplete")
    files = [{"path": str(path.relative_to(target)),
              "size": path.stat().st_size, "sha256": _sha256(path)}
             for path in sorted(target.rglob("*")) if path.is_file()]
    raw = {"schema_version": 1 if legacy_plan else 2,
           "kind": "h200_ncu_clock_diagnostic_raw_only",
           "case_id": plan["case_id"], "status": statuses[plan["case_id"]],
           "plan_sha256": _sha256(plan_path),
           "source_plan": source_pin, "files": files,
           "quality_accepted": False,
           "production_duration_consumed": False}
    if not legacy_plan:
        raw["plan_path"] = str(plan_path.relative_to(DIAGNOSTIC))
    if source_counters:
        raw["source_counters"] = True
    (target / "raw.json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_id": plan["case_id"], "status": raw["status"],
                      "raw_path": str(target / "raw.json")}), flush=True)
    if raw["status"] not in ("diagnostic_clock_rejected",
                             "diagnostic_clock_accepted"):
        raise RuntimeError("NCU diagnostic did not obtain valid counters")


@APP.local_entrypoint()
def main(batch_name: str | None = None, output_dir: str | None = None) -> None:
    if output_dir is not None:
        if batch_name is not None:
            raise ValueError("choose batch or NCU diagnostic output")
        _diagnostic_capture(output_dir)
        return
    if batch_name is None:
        raise ValueError("NCU batch name is required")
    plan_path = admitted_batch_plan(CAMPAIGN, batch_name)
    batch = json.loads(plan_path.read_text())
    output = CAMPAIGN / "batches" / batch_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite NCU raw: {output}")
    open_files = {}
    statuses = {}
    for event in capture.remote_gen(batch, _sha256(plan_path)):
        kind = event["kind"]
        if kind == "identity":
            output.mkdir(parents=True)
            (output / "identity.json").write_text(
                json.dumps(event, indent=2) + "\n")
        elif kind == "file_start":
            relative = Path(event["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"invalid remote path: {relative}")
            target = CAMPAIGN / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            open_files[event["path"]] = (
                target.open("xb"), event["sha256"], event["size"])
        elif kind == "file_chunk":
            open_files[event["path"]][0].write(event["data"])
        elif kind == "file_end":
            handle, digest, size = open_files.pop(event["path"])
            handle.close()
            target = CAMPAIGN / event["path"]
            if target.stat().st_size != size or _sha256(target) != digest:
                raise ValueError(f"NCU raw transfer hash mismatch: {target}")
        elif kind == "case_status":
            statuses[event["case_id"]] = event["status"]
            path = output / "ncu_raw" / event["case_id"] / "transfer_status.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(event, indent=2) + "\n")
            print(json.dumps(event))
        elif kind in ("partial", "complete"):
            (output / "batch_status.json").write_text(json.dumps({
                **event, "case_statuses": statuses,
            }, indent=2) + "\n")
            print(json.dumps(event))
        else:
            raise ValueError(f"unknown NCU event: {kind}")
    if open_files:
        raise ValueError(f"incomplete NCU raw transfer: {sorted(open_files)}")
