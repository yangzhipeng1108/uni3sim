"""Independent operand-distribution activity pilot; no GEMM execution."""
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
        IMAGE = IMAGE.add_local_file(source, remote_path="/tmp/tensor_distribution/" + source.name)
APP = modal.App("midlayer-h200-tensor-distribution-pilot-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=300, retries=0, single_use_containers=True)
def capture(steady: bool):
    started = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    with tempfile.TemporaryDirectory(prefix="midlayer-tensor-distribution-") as directory:
        subprocess.run(["python", "/tmp/tensor_distribution/shared_tensor_activity_gen.py",
            "/tmp/tensor_distribution/fa3_wgmma_service.cu", directory + "/shared_tensor_activity_helper.cuh"],
            check=True, timeout=30)
        command = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
            "-DOPERAND_DISTRIBUTION_CONTROLS", "-DDISTRIBUTION_PILOT_CONTROLS", "-Xptxas=-v", "-I" + directory,
            "/tmp/tensor_distribution/shared_tensor_activity.cu", "-o", directory + "/probe"]
        if steady:
            command.append("-DDISTRIBUTION_POINT_WARMUP")
        environment = dict(os.environ)
        removed = []
        for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
            if name in environment:
                del environment[name]
                removed.append(name)
        build = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=90)
        result = {"gpu": gpu, "command": command, "point_warmup_s": 2 if steady else 0,
            "compiler_flag_environment_removed": removed,
            "build_exit_code": build.returncode, "build_stdout": build.stdout, "build_stderr": build.stderr}
        if build.returncode == 0:
            result["binary_sha256"] = hashlib.sha256(Path(directory + "/probe").read_bytes()).hexdigest()
            run = subprocess.run([directory + "/probe"], capture_output=True, text=True, timeout=260)
            result.update({"run_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
        result["function_wall_s"] = time.monotonic() - started
    return result


@APP.local_entrypoint()
def main(steady: bool = False):
    campaign = "shared_tensor_distribution_steady_20261107" if steady else "shared_tensor_distribution_pilot_20261107"
    target = ROOT / "profiler_output/microbench/nvidia_h200" / campaign
    if target.exists():
        raise ValueError("refuse to overwrite captured evidence")
    result = capture.remote(steady)
    result["sources"] = {str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
        for source in SOURCES}
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"] = False
    target.mkdir(parents=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot = target / "source"
    snapshot.mkdir()
    for source in (*SOURCES, Path(__file__)):
        if source.name == "fa3_wgmma_service.cu":
            (snapshot / (source.name + ".gz")).write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        else:
            (snapshot / source.name).write_bytes(source.read_bytes())
    print(json.dumps({"build_exit_code": result["build_exit_code"], "function_wall_s": result["function_wall_s"]}))
