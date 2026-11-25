"""Paired random-fraction versus exact-power Tensor activity controls."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import modal

IMAGE = modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
if modal.is_local():
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parents[3]
    SOURCES = (HERE / "shared_tensor_activity.cu", HERE / "shared_tensor_activity_gen.py",
        ROOT / "microbench/nvidia_h200/operators/flash_attn/fa3_wgmma_service.cu")
    for source in SOURCES:
        IMAGE = IMAGE.add_local_file(source, remote_path="/tmp/tensor_mantissa/" + source.name)
APP = modal.App("midlayer-h200-tensor-mantissa-20261107")

@APP.function(gpu="H200", image=IMAGE, timeout=300, retries=0, single_use_containers=True)
def capture(sustained: bool):
    started = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit,clocks.sm,temperature.gpu", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    results = []
    with tempfile.TemporaryDirectory(prefix="midlayer-tensor-mantissa-") as directory:
        subprocess.run(["python", "/tmp/tensor_mantissa/shared_tensor_activity_gen.py",
            "/tmp/tensor_mantissa/fa3_wgmma_service.cu", directory + "/shared_tensor_activity_helper.cuh"],
            check=True, timeout=30)
        for extended in (False, True):
            command = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                "-DMANTISSA_ACTIVITY_CONTROLS", "-Xptxas=-v", "-I" + directory,
                "/tmp/tensor_mantissa/shared_tensor_activity.cu", "-o", directory + "/probe"]
            if extended:
                command.insert(1, "-DEXTENDED_CONTROLS")
            if sustained:
                command.insert(1, "-DSUSTAINED_DEVICE_ACTIVITY")
            environment = dict(os.environ)
            removed = []
            for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
                if name in environment:
                    del environment[name]
                    removed.append(name)
            build = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=90)
            result = {"extended_controls": extended, "command": command,
                "sustained_device_warmup": sustained,
                "compiler_flag_environment_removed": removed, "build_exit_code": build.returncode,
                "build_stdout": build.stdout, "build_stderr": build.stderr}
            results.append(result)
            if build.returncode != 0:
                break
            result["binary_sha256"] = hashlib.sha256(Path(directory + "/probe").read_bytes()).hexdigest()
            run = subprocess.run([directory + "/probe"], capture_output=True, text=True, timeout=180)
            result.update({"run_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
            result["gpu_after"] = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit,clocks.sm,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=20).stdout
            if run.returncode != 0:
                break
    return {"gpu": gpu, "captures": results, "function_wall_s": time.monotonic() - started}

@APP.local_entrypoint()
def main(sustained: bool = False, matched_replica: bool = False):
    if matched_replica and not sustained:
        raise ValueError("matched replica requires the original sustained protocol")
    campaign = ("shared_tensor_sustained_replica_20261107" if matched_replica
        else "shared_tensor_sustained_20261107" if sustained else "shared_tensor_mantissa_20261107")
    target = ROOT / "profiler_output/microbench/nvidia_h200" / campaign
    if target.exists() and {p.name for p in target.iterdir()} != {"preflight.json"}:
        raise ValueError("refuse to overwrite captured evidence")
    result = capture.remote(sustained)
    target.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for index, row in enumerate(result["captures"]):
        row.update({"gpu": result["gpu"], "sources": sources, "runner_sha256": runner_sha,
            "production_duration_consumed": False})
        name = ("base_raw.json", "extended_raw.json")[index]
        (target / name).write_text(json.dumps(row, indent=2) + "\n")
    (target / "capture_record.json").write_text(json.dumps({"gpu": result["gpu"],
        "function_wall_s": result["function_wall_s"], "subcaptures": len(result["captures"]),
        "source_sha256": sources, "runner_sha256": runner_sha}, indent=2) + "\n")
    snapshot = target / "source"
    snapshot.mkdir()
    for source in (*SOURCES, Path(__file__)):
        if source.name == "fa3_wgmma_service.cu":
            (snapshot / (source.name + ".gz")).write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        else:
            (snapshot / source.name).write_bytes(source.read_bytes())
    print(json.dumps({"build_exit_codes": [r["build_exit_code"] for r in result["captures"]],
        "function_wall_s": result["function_wall_s"]}))
