"""Run the prepared 39-case H200 NSYS plan on one Modal GPU.

The image must reproduce the CUDA 12.9 vLLM 0.30 stack before any timed
capture.  A first window of every case is checked against the old launch
identity before the remaining two windows run.  Raw files are streamed home;
this runner never publishes a canonical dataset.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import modal


_MODULE_PATH = Path(__file__).resolve()
_REMOTE_MODULE = (len(_MODULE_PATH.parents) < 4
                  or Path("/opt/midlayer/scripts").is_dir())
ROOT = Path("/opt/midlayer") if _REMOTE_MODULE else _MODULE_PATH.parents[3]
CAMPAIGN_NAME = (os.environ["MIDLAY_H200_NSYS_CAMPAIGN"]
                 if "MIDLAY_H200_NSYS_CAMPAIGN" in os.environ
                 else "missing15_retry_20261031")
if not re.fullmatch(r"[a-z0-9_]+", CAMPAIGN_NAME):
    raise ValueError(f"invalid H200 NSYS campaign name: {CAMPAIGN_NAME}")
CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200" / CAMPAIGN_NAME
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
APP = modal.App("midlayer-h200-missing15-nsys-20261031")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stream_files(root: Path, sent: set[str]):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "gpu.lock":
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
    if versions["torch"] != "2.13.0+cu129" or versions["torch_cuda"] != "12.9":
        raise ValueError(f"Torch/CUDA differs from old campaign: {versions}")
    if versions["vllm"] != "0.30.0+cu129":
        raise ValueError(f"vLLM differs from old campaign: {versions}")
    if versions["triton"] != "3.7.1" or versions["flashinfer_python"] != "0.6.18.post1":
        raise ValueError(f"Triton/FlashInfer differs from old campaign: {versions}")
    if versions["cublas"] != "12.9.1.4":
        raise ValueError(f"cuBLAS differs from old campaign: {versions}")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version",
         "--format=csv,noheader"], check=True, capture_output=True,
        text=True, timeout=20).stdout.strip()
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError(f"expected one H200: {gpu}")
    nsys = shutil.which("nsys")
    ncu = shutil.which("ncu")
    if nsys is None or ncu is None:
        raise ValueError("NSYS and NCU must both be installed")
    tools = {}
    for name, path in (("nsys", nsys), ("ncu", ncu)):
        tools[name] = subprocess.run(
            [path, "--version"], check=True, capture_output=True,
            text=True, timeout=20).stdout.strip()
    return {"gpu": gpu, "versions": versions, "profiler_versions": tools}


def _signature(window: dict) -> list:
    return [[launch["name"], launch["grid"], launch["block"]]
            for launch in window["launches"]]


@APP.function(gpu="H200", image=IMAGE, timeout=600,
              max_containers=1, single_use_containers=True)
def capture(manifest: dict, batch_name: str):
    identity = _check_identity()
    yield {"kind": "identity", "data": identity}
    remote = Path("/opt/midlayer")
    sys.path.insert(0, str(remote))
    from dataset_pipeline.kernel_dataset.nvidia_h200 import (
        load_case_failures, load_case_windows, select_case_timing)

    with tempfile.TemporaryDirectory(prefix="midlayer-missing39-", dir="/tmp") as tmp:
        output = Path(tmp) / "batches" / batch_name
        output.mkdir(parents=True)
        lock = output / "gpu.lock"
        sent = set()
        for family in (manifest["batches"][batch_name]["family"],):
            entry = manifest["batches"][batch_name]
            plan = remote / "plans" / Path(entry["plan_path"]).name
            if _sha256(plan) != entry["plan_sha256"]:
                raise ValueError(f"plan hash mismatch: {family}")
            result_root = output / "runs" / plan.stem
            result_root.mkdir(parents=True)
            provenance = [{
                "plan_sha256": entry["plan_sha256"],
                "physical_gpu": 0,
                "gpu_identity": identity["gpu"],
                "software_identity": identity["versions"],
                "profiler_versions": identity["profiler_versions"],
                "runner_sha256": _sha256(remote / "scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py"),
            }]
            (result_root / "_provenance.json").write_text(
                json.dumps(provenance, indent=2) + "\n")
            first_case = entry["case_ids"][0]
            smoke_file = output / f"{family}_smoke_id.txt"
            smoke_file.write_text(first_case + "\n")
            for window_ids, case_filter in (("1", smoke_file), ("1", None),
                                            ("2,3", None)):
                command = [
                    sys.executable,
                    str(remote / "scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py"),
                    "--plans", str(plan), "--output-dir", str(result_root),
                    "--window-ids", window_ids, "--concurrency", "1",
                    "--physical-gpu", "0", "--nsys", shutil.which("nsys"),
                    "--python", sys.executable, "--gpu-lock", str(lock),
                ]
                if case_filter is not None:
                    command.extend(["--case-ids", str(case_filter)])
                run = subprocess.run(command, capture_output=True, text=True,
                                     timeout=5000)
                label = ("1_smoke" if case_filter == smoke_file
                         else window_ids.replace(",", "_"))
                log = result_root / f"window_pass_{label}.json"
                log.write_text(json.dumps({
                    "command": command, "returncode": run.returncode,
                    "stdout": run.stdout, "stderr": run.stderr,
                }, indent=2) + "\n")
                yield from _stream_files(Path(tmp), sent)
                if run.returncode != 0:
                    yield {"kind": "stopped", "reason": "runner_failed",
                           "family": family, "window_ids": window_ids}
                    return
                if case_filter == smoke_file:
                    smoke = result_root / "cases" / first_case / "window_01.launch.json"
                    if not smoke.is_file():
                        yield {"kind": "stopped", "reason": "smoke_window_missing",
                               "family": family}
                        return
                    first = json.loads(smoke.read_text())
                    if not first["telemetry"]["clock_accepted"]:
                        yield {"kind": "stopped", "reason": "smoke_clock_rejected",
                               "family": family}
                        return
                    if _signature(first) != entry["expected_launch_signatures"][
                            first_case]["signature"]:
                        yield {"kind": "stopped", "reason": "smoke_launch_mismatch",
                               "family": family}
                        return
                elif window_ids == "1":
                    mismatches = []
                    for case_id, expected in entry["expected_launch_signatures"].items():
                        path = result_root / "cases" / case_id / "window_01.launch.json"
                        if not path.is_file() or _signature(json.loads(path.read_text())) != expected["signature"]:
                            mismatches.append(case_id)
                    if mismatches:
                        yield {"kind": "stopped", "reason": "launch_mismatch",
                               "family": family, "case_ids": mismatches}
                        return
                elif window_ids == "2,3":
                    missing = [
                        case_id for case_id in entry["case_ids"]
                        if any(not (result_root / "cases" / case_id /
                                    f"window_{window:02d}.launch.json").is_file()
                               for window in (2, 3))]
                    if missing:
                        yield {"kind": "stopped", "reason": "incomplete_windows",
                               "family": family, "case_ids": missing}
                        return
            statuses = {}
            for case_id in entry["case_ids"]:
                case_dir = result_root / "cases" / case_id
                windows = load_case_windows(case_dir)
                failures = load_case_failures(case_dir)
                failed_ids = [item["window"] for item in failures]
                statuses[case_id] = select_case_timing(
                    case_id, windows, failed_ids)["status"]
            yield {"kind": "family_status", "family": family,
                   "case_statuses": statuses}
        yield {"kind": "complete"}


@APP.local_entrypoint()
def main(batch_name: str) -> None:
    manifest_path = CAMPAIGN / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "prepared_not_executed":
        raise ValueError("only the prepared campaign can be started")
    if batch_name not in manifest["batches"]:
        raise ValueError(f"unknown prepared batch: {batch_name}")
    output = CAMPAIGN / "batches" / batch_name / "runs"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite raw: {output}")
    output.mkdir(parents=True)
    open_files = {}
    for event in capture.remote_gen(manifest, batch_name):
        kind = event["kind"]
        if kind == "identity":
            (output / "identity.json").write_text(
                json.dumps(event["data"], indent=2) + "\n")
        elif kind == "file_start":
            relative = Path(event["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"invalid remote path: {relative}")
            target = CAMPAIGN / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            open_files[event["path"]] = (target.open("xb"), event["sha256"],
                                          event["size"])
        elif kind == "file_chunk":
            open_files[event["path"]][0].write(event["data"])
        elif kind == "file_end":
            handle, digest, size = open_files.pop(event["path"])
            handle.close()
            target = CAMPAIGN / event["path"]
            if target.stat().st_size != size or _sha256(target) != digest:
                raise ValueError(f"transferred raw failed hash check: {target}")
        elif kind == "family_status":
            (output / f"{event['family']}_status.json").write_text(
                json.dumps(event, indent=2) + "\n")
            print(json.dumps({"family": event["family"],
                              "statuses": event["case_statuses"]}))
        elif kind in ("stopped", "complete"):
            (output / "transfer_status.json").write_text(
                json.dumps(event, indent=2) + "\n")
            print(json.dumps(event))
        else:
            raise ValueError(f"unknown remote event: {kind}")
    if open_files:
        raise ValueError(f"incomplete raw transfer: {sorted(open_files)}")


@APP.function(gpu="H200", image=IMAGE, timeout=600,
              max_containers=1, single_use_containers=True)
def clock_probe(manifest: dict, batch_name: str, case_id: str,
                probe_name: str):
    """Probe same-run GPU clock metrics without changing timing publication."""
    identity = _check_identity()
    yield {"kind": "identity", "data": identity}
    entry = manifest["batches"][batch_name]
    if case_id not in entry["case_ids"]:
        raise ValueError(f"case is not in prepared batch: {case_id}")
    remote = Path("/opt/midlayer")
    plan = remote / "plans" / Path(entry["plan_path"]).name
    if _sha256(plan) != entry["plan_sha256"]:
        raise ValueError("clock probe plan SHA mismatch")
    with tempfile.TemporaryDirectory(prefix="midlayer-clock-probe-",
                                     dir="/tmp") as tmp:
        output = Path(tmp) / probe_name
        output.mkdir()
        help_result = subprocess.run(
            [shutil.which("nsys"), "profile", "--gpu-metrics-devices=help"],
            capture_output=True, text=True, timeout=20)
        help_record = {
            "command": ["nsys", "profile", "--gpu-metrics-devices=help"],
            "returncode": help_result.returncode,
            "stdout": help_result.stdout,
            "stderr": help_result.stderr,
            "identity": identity,
            "plan_sha256": entry["plan_sha256"],
            "case_id": case_id,
        }
        (output / "gpu_metrics_help.json").write_text(
            json.dumps(help_record, indent=2) + "\n")
        help_text = (help_result.stdout + help_result.stderr).lower()
        if (help_result.returncode != 0 or "not available" in help_text
                or "not supported" in help_text):
            yield from _stream_files(Path(tmp), set())
            yield {"kind": "stopped", "reason": "gpu_metrics_unavailable"}
            return
        case_filter = output / "case_id.txt"
        case_filter.write_text(case_id + "\n")
        result_root = output / "run"
        command = [
            sys.executable,
            str(remote / "scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py"),
            "--plans", str(plan), "--output-dir", str(result_root),
            "--window-ids", "1", "--case-ids", str(case_filter),
            "--concurrency", "1", "--physical-gpu", "0",
            "--nsys", shutil.which("nsys"), "--python", sys.executable,
            "--gpu-lock", str(output / "gpu.lock"),
            "--gpu-metrics-devices", "cuda-visible", "--retain-sqlite",
        ]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=500)
        (output / "capture_command.json").write_text(json.dumps({
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "runner_sha256": _sha256(remote / "scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py"),
        }, indent=2) + "\n")
        launch = result_root / "cases" / case_id / "window_01.launch.json"
        status = "capture_failed"
        if completed.returncode == 0 and launch.is_file():
            actual = _signature(json.loads(launch.read_text()))
            expected = entry["expected_launch_signatures"][case_id]["signature"]
            status = ("capture_complete_unqualified" if actual == expected
                      else "launch_mismatch")
        yield from _stream_files(Path(tmp), set())
        yield {"kind": status, "case_id": case_id}


@APP.local_entrypoint()
def clock_probe_main(batch_name: str, case_id: str,
                     probe_name: str = "clock_probe_20261102") -> None:
    if not re.fullmatch(r"clock_probe_[a-z0-9_]+", probe_name):
        raise ValueError(f"invalid clock probe name: {probe_name}")
    output = CAMPAIGN / probe_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite clock probe: {output}")
    output.mkdir()
    manifest = json.loads((CAMPAIGN / "plan_manifest.json").read_text())
    open_files = {}
    for event in clock_probe.remote_gen(manifest, batch_name, case_id,
                                        probe_name):
        kind = event["kind"]
        if kind == "identity":
            (output / "identity.json").write_text(
                json.dumps(event["data"], indent=2) + "\n")
        elif kind == "file_start":
            relative = Path(event["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"invalid remote path: {relative}")
            target = CAMPAIGN / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            open_files[event["path"]] = (target.open("xb"), event["sha256"],
                                          event["size"])
        elif kind == "file_chunk":
            open_files[event["path"]][0].write(event["data"])
        elif kind == "file_end":
            handle, digest, size = open_files.pop(event["path"])
            handle.close()
            target = CAMPAIGN / event["path"]
            if target.stat().st_size != size or _sha256(target) != digest:
                raise ValueError(f"clock probe transfer hash mismatch: {target}")
        elif kind in ("stopped", "capture_failed", "launch_mismatch",
                      "capture_complete_unqualified"):
            (output / "transfer_status.json").write_text(
                json.dumps(event, indent=2) + "\n")
            print(json.dumps(event))
        else:
            raise ValueError(f"unknown clock probe event: {kind}")
    if open_files:
        raise ValueError(f"incomplete clock probe transfer: {sorted(open_files)}")
