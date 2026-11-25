"""Run the independent H200 16-byte two-read/one-write stream on Modal."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

import modal


SOURCE = Path(__file__).with_name("vec16_two_read_one_write.cu")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12"
).add_local_file(SOURCE, remote_path="/tmp/vec16_two_read_one_write.cu")
APP = modal.App("midlayer-h200-vec16-mixed-dram-20261029")


@APP.function(gpu="H200", image=IMAGE, timeout=600, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    task_dir = Path("/tmp/midlayer-vec16-mixed-dram-20261029")
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
        binary = task_dir / "vec16_two_read_one_write"
        build = subprocess.run(
            ["nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-Xptxas=-v",
             "-o", str(binary), "/tmp/vec16_two_read_one_write.cu"],
            capture_output=True, text=True, timeout=120)
        if build.returncode:
            return {"started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
                    "gpu_wall_s": time.monotonic() - start, "gpu_identity": gpu,
                    "nvcc_version": nvcc, "build_exit_code": build.returncode,
                    "build_stdout": build.stdout, "build_stderr": build.stderr}
        results = []
        for _ in range(2):
            run = subprocess.run([str(binary)], capture_output=True,
                                 text=True, timeout=180)
            results.append({"exit_code": run.returncode, "stdout": run.stdout,
                            "stderr": run.stderr})
            if run.returncode:
                break
        return {"started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_wall_s": time.monotonic() - start, "gpu_identity": gpu,
                "nvcc_version": nvcc, "build_exit_code": 0,
                "build_stdout": build.stdout, "build_stderr": build.stderr,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "runs": results}
    finally:
        shutil.rmtree(task_dir)


@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    parent = root / "profiler_output/microbench/nvidia_h200/vec16_mixed_dram_20261029"
    target = Path(output_dir).resolve()
    if target != parent.resolve() or target.exists():
        raise ValueError("output must be the new registered vec16 campaign directory")
    result = capture.remote()
    target.mkdir(parents=True)
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = "independent_16byte_2read_1write_dram_stream_only"
    (target / "raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(target), "gpu_wall_s": result["gpu_wall_s"],
                      "build_exit_code": result["build_exit_code"]}, sort_keys=True))
