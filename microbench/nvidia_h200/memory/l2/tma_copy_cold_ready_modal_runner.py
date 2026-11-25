"""Run an independent cold and repeated TMA copy-ready probe on one H200."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
TASK = Path("/tmp/midlayer-fa-tma-copy-ready-20261107")
SOURCE = HERE / "tma_copy_cold_ready.cu"
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).apt_install("cuda-nsight-compute-12-6").add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
APP = modal.App("midlayer-fa-tma-copy-ready-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=240, retries=0,
              single_use_containers=True)
def capture():
    started = time.monotonic()
    result = {}
    try:
        commands = {
            "identity": ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm",
                         "--format=csv,noheader"],
            "compiler": ["nvcc", "--version"],
            "build": ["nvcc", "-O3", "-arch=sm_90", "-Xptxas=-v",
                      str(TASK / SOURCE.name), "-L/usr/local/cuda/lib64/stubs", "-lcuda", "-o", str(TASK / "probe")],
            "sass": ["cuobjdump", "--dump-sass", str(TASK / "probe")],
            "measurement": [str(TASK / "probe")],
            # 3 preceding grids * 70 launches + 5 byte widths * 10:
            # profile the paired cold/warm 32-KiB, 132-CTA copies only.
            "counter": ["ncu", "--csv", "--page", "raw", "--cache-control", "none",
                        "--clock-control", "none", "--kernel-name", "regex:ready",
                        "--launch-skip", "260", "--launch-count", "2", "--metrics",
                        "lts__t_sectors_op_read.sum,dram__bytes_read.sum", str(TASK / "probe")],
            "final_clock": ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
        }
        for name, command in commands.items():
            process = subprocess.run(command, capture_output=True, text=True, timeout=90)
            result[name] = {"command": command, "exit_code": process.returncode,
                            "stdout": process.stdout, "stderr": process.stderr}
            if process.returncode:
                break
        return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started
        shutil.rmtree(TASK, ignore_errors=True)


@APP.local_entrypoint()
def main(output_dir: str):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "raw.json").exists():
        raise ValueError("capture already exists")
    result = capture.remote()
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print({"wall_s": result["gpu_function_wall_s"], "measurement_exit_code": result["measurement"]["exit_code"] if "measurement" in result else None})
