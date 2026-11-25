"""Finite on-chip Tensor activity after same-kernel ALU-only high-clock prelude."""
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
    SOURCE = HERE / "tensor_activity_transition.cu"
    GENERATOR = HERE / "shared_tensor_activity_gen.py"
    DEPENDENCY = ROOT / "microbench/nvidia_h200/operators/flash_attn/fa3_wgmma_service.cu"
    IMAGE = IMAGE.add_local_file(SOURCE, remote_path="/tmp/tensor_activity_transition.cu")
    IMAGE = IMAGE.add_local_file(DEPENDENCY, remote_path="/tmp/fa3_wgmma_service.cu")
if modal.is_local():
    IMAGE = IMAGE.add_local_file(GENERATOR, remote_path="/tmp/shared_tensor_activity_gen.py")
APP = modal.App("midlayer-h200-tensor-activity-transition-20261107")


@APP.function(gpu="H200", image=IMAGE, memory=1024, timeout=300, retries=0, single_use_containers=True)
def capture():
    started = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    environment = dict(os.environ)
    removed = []
    for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
        if name in environment:
            del environment[name]
            removed.append(name)
    captures = []
    with tempfile.TemporaryDirectory(prefix="midlayer-tensor-transition-") as directory:
        subprocess.run(["python", "/tmp/shared_tensor_activity_gen.py", "/tmp/fa3_wgmma_service.cu", directory + "/shared_tensor_activity_helper.cuh"], check=True, timeout=20)
        for mode in ("finite_activity",):
            command = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v",
                "-I" + directory, "/tmp/tensor_activity_transition.cu", "-o", directory + "/probe"]
            build = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
            result = {"mode": mode, "build_command": command,
                "build_exit_code": build.returncode, "build_stdout": build.stdout, "build_stderr": build.stderr}
            if build.returncode == 0:
                result["binary_sha256"] = hashlib.sha256(Path(directory + "/probe").read_bytes()).hexdigest()
                dump = subprocess.run(["cuobjdump", "--dump-sass", directory + "/probe"],
                    capture_output=True, text=True, check=True, timeout=20)
                result["sass_stdout"] = dump.stdout
                result["sass_stderr"] = dump.stderr
                run = subprocess.run([directory + "/probe"], capture_output=True, text=True, env=environment, timeout=60)
                result.update({"run_exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr})
            captures.append(result)
            if build.returncode != 0 or result["run_exit_code"] != 0:
                break
    return {"gpu": gpu, "compiler_flag_environment_removed": removed,
        "captures": captures, "function_wall_s": time.monotonic() - started}


@APP.local_entrypoint()
def main():
    target = ROOT / "profiler_output/microbench/nvidia_h200/tensor_activity_transition_20261107"
    if target.exists() and {p.name for p in target.iterdir()} != {"preflight.json", "local_build.log.gz"}:
        raise ValueError("refuse to overwrite captured evidence")
    result = capture.remote()
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"] = False
    result["generator_sha256"] = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    result["dependency_sha256"] = hashlib.sha256(DEPENDENCY.read_bytes()).hexdigest()
    target.mkdir(parents=True, exist_ok=True)
    for item in result["captures"]:
        if "stdout" in item:
            payload = item["stdout"].encode()
            (target / "timings.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
            item["stdout_sha256"] = hashlib.sha256(payload).hexdigest()
            item["stdout_path"] = "timings.jsonl.gz"
            del item["stdout"]
        if "sass_stdout" in item:
            data = item["sass_stdout"].encode()
            filename = "transition.sass.txt.gz"
            (target / filename).write_bytes(gzip.compress(data, mtime=0))
            item["sass_stdout_sha256"] = hashlib.sha256(data).hexdigest()
            item["sass_path"] = filename
            del item["sass_stdout"]
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot = target / "source"
    snapshot.mkdir()
    for source in (SOURCE, DEPENDENCY, GENERATOR, Path(__file__)):
        if source.name == "fa3_wgmma_service.cu":
            (snapshot / (source.name + ".gz")).write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        else:
            (snapshot / source.name).write_bytes(source.read_bytes())
    print(json.dumps({"capture_count": len(result["captures"]), "function_wall_s": result["function_wall_s"]}))
