"""Capture an independent 96-thread CTA SiLU register service on H200."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

import modal


SOURCE = Path(__file__).with_name("silu_96_cta_service.cu")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12"
).add_local_file(SOURCE, remote_path="/tmp/silu_96_cta_service.cu")
APP = modal.App("midlayer-h200-silu-96-cta-service-20261029")


@APP.function(gpu="H200", image=IMAGE, timeout=600, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    task_dir = Path("/tmp/midlayer-silu-96-cta-service-20261029")
    task_dir.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,driver_version",
             "--format=csv,noheader"], check=True, capture_output=True,
            text=True, timeout=30).stdout.strip()
        nvcc = subprocess.run(["nvcc", "--version"], check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip()
        binary = task_dir / "silu_96_cta_service"
        build = subprocess.run(
            ["nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-Xptxas=-v",
             "-o", str(binary), "/tmp/silu_96_cta_service.cu"],
            capture_output=True, text=True, timeout=120)
        result = {"started_utc": started, "gpu_identity": gpu,
                  "nvcc_version": nvcc, "build_exit_code": build.returncode,
                  "build_stdout": build.stdout, "build_stderr": build.stderr}
        if build.returncode == 0:
            sass = subprocess.run(["cuobjdump", "--dump-sass", str(binary)],
                                  capture_output=True, text=True, timeout=60)
            result["sass_exit_code"] = sass.returncode
            result["sass_stdout"] = sass.stdout
            result["sass_stderr"] = sass.stderr
            result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
            runs = []
            for _ in range(2):
                run = subprocess.run([str(binary)], capture_output=True,
                                     text=True, timeout=180)
                runs.append({"exit_code": run.returncode, "stdout": run.stdout,
                             "stderr": run.stderr})
                if run.returncode:
                    break
            result["runs"] = runs
        result["ended_utc"] = datetime.now(timezone.utc).isoformat()
        result["gpu_wall_s"] = time.monotonic() - start
        return result
    finally:
        shutil.rmtree(task_dir)


@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    parent = root / "profiler_output/microbench/nvidia_h200/silu_96_cta_service_20261029"
    target = Path(output_dir).resolve()
    if target != parent.resolve() or target.exists():
        raise ValueError("output must be the new registered SiLU service campaign directory")
    result = capture.remote()
    target.mkdir(parents=True)
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = "independent_register_silu_96_cta_service_only"
    (target / "raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(target), "gpu_wall_s": result["gpu_wall_s"],
                      "build_exit_code": result["build_exit_code"]}, sort_keys=True))
