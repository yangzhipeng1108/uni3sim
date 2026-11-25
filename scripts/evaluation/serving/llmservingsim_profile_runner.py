#!/usr/bin/env python3
"""Run an explicitly owned native layerwise capture in an isolated execution copy."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dataset_pipeline.serving_dataset.llmservingsim_profile import (
    require_capture_owner,
    validate_resume_identity,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--owner", type=require_capture_owner, required=True,
                        help="explicit owner of this hardware task; must match any resumed capture")
    parser.add_argument("--hardware", choices=("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b"), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--device-uuid", help="expected physical NVIDIA GPU UUID for an explicit reservation")
    parser.add_argument("--model", choices=("Qwen/Qwen3-4B", "meta-llama/Llama-3.1-8B-Instruct"))
    parser.add_argument("--stage", choices=("coverage", "profile", "slice", "hardware"), required=True)
    parser.add_argument("--group", choices=("dense", "per_sequence", "attention"))
    parser.add_argument("--vllm-011", action="store_true")
    parser.add_argument("--skip-skew", action="store_true", help="native uniform-only pilot; report missing skew coverage")
    parser.add_argument("--attention-grid-factor", type=float, choices=(2.0, 4.0, 8.0), help="native geometric grid density; does not change its domain limits")
    parser.add_argument("--resume-capture", type=Path, help="reuse measured native checkpoints from this task's prior capture")
    parser.add_argument("--collector-baseline", help="exact deployed Midlayer collector commit")
    parser.add_argument("--block16-evidence", type=Path, help="explicit Ascend native-operator capability evidence")
    args = parser.parse_args()
    if args.model is None and args.stage != "hardware":
        parser.error("--model is required for model collection")
    if args.block16_evidence is not None and args.hardware != "ascend_910b":
        parser.error("--block16-evidence applies only to Ascend collection")
    if args.device_uuid is not None:
        if args.hardware not in ("nvidia_a800", "nvidia_h200"):
            parser.error("--device-uuid applies only to NVIDIA collection")
        if re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", args.device_uuid) is None:
            parser.error("--device-uuid must be a physical NVIDIA GPU UUID")
    root = Path(__file__).resolve().parents[3]
    target_config = json.loads((root / 'configs/baselines/llmservingsim/targets.json').read_text())
    scope = target_config['hardware_status']
    hardware_ids = target_config['hardware_ids']
    if (not isinstance(scope, dict) or not isinstance(hardware_ids, list)
            or any(not isinstance(hardware, str) for hardware in hardware_ids)
            or len(set(hardware_ids)) != len(hardware_ids) or set(scope) != set(hardware_ids)
            or any(status not in ('active', 'deferred_by_user') for status in scope.values())
            or 'active' not in scope.values()):
        parser.error('invalid target hardware scope')
    if scope[args.hardware] == 'deferred_by_user':
        parser.error('hardware deferred by user; collection is disabled')
    runtime, output = args.runtime.resolve(), args.raw_output.resolve()
    if not output.is_relative_to(root / "profiler_output"):
        raise ValueError("raw capture must be written under this execution checkout's profiler_output")
    if args.resume_capture is not None:
        previous = args.resume_capture.resolve()
        if not previous.is_relative_to(root / "profiler_output") or previous == output:
            raise ValueError("resume source must be a different task raw capture")
        recorded = json.loads((previous / "run.json").read_text())
        validate_resume_identity(recorded, args.owner, args.hardware)
        if args.model not in recorded["command"] or recorded["status"] == "running":
            raise ValueError("resume requires a stopped same-model capture")
        if args.hardware == 'ascend_910b':
            prior = recorded['block16_evidence']
            prior_digest = None if prior is None else prior['sha256']
            current_digest = None if args.block16_evidence is None else hashlib.sha256(
                args.block16_evidence.read_bytes()).hexdigest()
            if prior_digest != current_digest:
                raise ValueError('resume cannot change the Ascend geometry capability evidence')
    device_identity = None
    if args.hardware in ("nvidia_a800", "nvidia_h200"):
        idle = subprocess.check_output(["nvidia-smi", "-i", str(args.device),
            "--query-gpu=name,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
        name, uuid, _, utilization = (value.strip() for value in idle.split(","))
        expected_model = {"nvidia_a800": "A800", "nvidia_h200": "H200"}[args.hardware]
        if re.search(rf"\b{expected_model}\b", name) is None:
            raise RuntimeError(f"selected device is not {args.hardware}: {name}")
        if args.device_uuid is not None and uuid != args.device_uuid:
            raise RuntimeError(f"selected physical GPU UUID differs from reservation: {uuid}")
        processes = subprocess.check_output(["nvidia-smi", "-i", str(args.device),
            "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if int(utilization) != 0 or processes:
            raise RuntimeError(f"selected {args.hardware} is busy: {idle}; processes={processes}")
        device_identity = {"physical_device": args.device, "uuid": uuid,
                           "cuda_visible_devices": str(args.device), "process_device": 0}
    elif args.hardware == "metax_c500":
        idle = subprocess.check_output(["mx-smi"], text=True)
        rows = re.findall(rf"^\|\s*{args.device}\s+MetaX C500\s*\|\s*{args.device}\s+.*?\|\s*(\d+)%", idle, re.MULTILINE)
        processes = re.findall(rf"^\|\s*{args.device}\s+(\d+)\s+\S+.*\|$", idle, re.MULTILINE)
        if rows != ["0"] or processes:
            raise RuntimeError("C500 layerwise collection requires an idle selected card with no device process")
    elif args.hardware == "ascend_910b":
        idle = subprocess.check_output(["npu-smi", "info"], text=True)
        if f"No running processes found in NPU {args.device}" not in idle:
            raise RuntimeError("selected Ascend card has a running process")
    else:
        raise ValueError("unknown capture hardware")
    output.mkdir(parents=True, exist_ok=False)
    resumed_files = None
    if args.resume_capture is not None:
        source_perf = previous / "perf"
        resumed_files = {str(path.relative_to(previous)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_perf.rglob("*")) if path.is_file()}
        shutil.copytree(source_perf, output / "perf")
    if args.vllm_011:
        from dataset_pipeline.serving_dataset.llmservingsim_profile import prepare_vllm_011_profiler
        prepare_vllm_011_profiler(runtime, root / "configs/baselines/llmservingsim/source.json",
            root / "experiment_logs/evaluation/llmservingsim/profiler-adapters")
    command = [str(args.python.absolute()), "-m", "profiler", args.stage, args.model,
        "--hardware", args.hardware, "--tp", "1", "--dtype", "bfloat16", "--kv-cache-dtype", "auto",
        "--num-hidden-layers", "1", "--max-num-batched-tokens", "8192", "--max-num-seqs", "256",
        "--max-model-len", "8192", "--block-size", "16", "--gpu-memory-utilization", "0.8",
        "--out-root", str(output / "perf"), "--verbose"]
    if args.stage == "hardware":
        if args.model is not None or args.group is not None or args.skip_skew or args.attention_grid_factor is not None:
            parser.error("hardware acquisition does not accept model/sweep options")
        command = [str(args.python.absolute()), "-m", "profiler", "hardware",
            "--hardware", args.hardware, "--npus", "2", "--out", str(output / "perf")]
        if args.hardware in ("nvidia_a800", "nvidia_h200"):
            command = [str(args.python.absolute()), "-c",
                "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                "from dataset_pipeline.serving_dataset.llmservingsim_profile import write_nvidia_hardware_metadata; "
                "write_nvidia_hardware_metadata(sys.argv[2], sys.argv[3], Path(sys.argv[4]))",
                str(root), args.hardware, device_identity["uuid"], str(output)]
    if args.stage == "slice":
        if args.group is None:
            parser.error("slice requires --group")
        command.extend(("--tp-refresh", "1", "--group", args.group))
    elif args.group is not None:
        parser.error("--group requires slice")
    if args.skip_skew:
        command.append("--skip-skew")
    if args.attention_grid_factor is not None:
        for option in ("--attention-chunk-factor", "--attention-kv-factor", "--attention-n-factor"):
            command.extend((option, str(args.attention_grid_factor)))
    rpc = tempfile.mkdtemp(prefix="llmsim-rpc-")
    temporary = Path(rpc) / "tmp"
    temporary.mkdir()
    cache = root / "experiment_logs/evaluation/llmservingsim/profiler-cache" / output.name
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": str(args.device), "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "VLLM_RPC_BASE_PATH": rpc,
        "LLMSIM_RAW_OUTPUT": str(output),
        "TMPDIR": str(temporary), "HF_HOME": str(output / "hf-cache"),
        "TRITON_CACHE_DIR": str(cache / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TORCH_EXTENSIONS_DIR": str(cache / "extensions"), "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "PYTHONDONTWRITEBYTECODE": "1"})
    if args.hardware == "ascend_910b":
        environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
        if args.block16_evidence is not None:
            environment["LLMSIM_BLOCK16_EVIDENCE"] = str(args.block16_evidence.resolve())
    if args.collector_baseline is None:
        baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    elif re.fullmatch("[0-9a-f]{40}", args.collector_baseline):
        baseline = args.collector_baseline
    else:
        raise ValueError("collector baseline must be an exact commit")
    (output / "tmp").mkdir()
    state = {"owner": args.owner, "hardware": args.hardware,
        "device": args.device, "device_identity": device_identity, "baseline": baseline,
        "block16_evidence": None if args.block16_evidence is None else {
            "path": str(args.block16_evidence.resolve()),
            "sha256": hashlib.sha256(args.block16_evidence.read_bytes()).hexdigest()},
        "resumed_from": None if args.resume_capture is None else str(args.resume_capture.resolve()),
        "resumed_files": resumed_files,
        "model_config": None if args.model is None else json.loads(
            (runtime / "configs/model" / f"{args.model}.json").read_text()),
        "collector_sources": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), root / "dataset_pipeline/serving_dataset/llmservingsim_profile.py")
            if path.is_file()},
        "profiler_sources": {str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((runtime / "profiler").rglob("*"))
            if path.is_file() and path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts},
        "command": command, "cwd": str(runtime), "output": str(output), "preflight": idle,
        "pid": None, "runner_pid": os.getpid(), "status": "registered", "exit_code": None,
        "started_unix_s": time.time(), "rpc_tmp": rpc}
    manifest = output / "run.json"
    manifest.write_text(json.dumps(state, indent=2) + "\n")
    process = None

    def interrupted(signum, frame):
        raise InterruptedError(f"capture runner received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (output / "capture.log").open("w") as log:
            process = subprocess.Popen(command, cwd=runtime, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state.update({"pid": process.pid, "status": "running"})
            manifest.write_text(json.dumps(state, indent=2) + "\n")
            code = process.wait()
        state.update({"exit_code": code, "status": "completed" if code == 0 else "failed",
                      "finished_unix_s": time.time()})
        manifest.write_text(json.dumps(state, indent=2) + "\n")
        return code
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        state.update({"status": "interrupted", "error": repr(exc),
                      "finished_unix_s": time.time()})
        manifest.write_text(json.dumps(state, indent=2) + "\n")
        raise
    finally:
        shutil.rmtree(rpc)


if __name__ == "__main__":
    raise SystemExit(main())
