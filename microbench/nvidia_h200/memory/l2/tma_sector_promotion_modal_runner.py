"""Capture independent TMA-promotion traffic counters on one H200."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent
SOURCES = (HERE / "tma_sector_promotion.cu",
    ROOT / "microbench/nvidia_h200/operators/gemm/fp8_tma_operand_stream.cu")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12"
).apt_install("cuda-nsight-compute-12-9")
for source in SOURCES:
    IMAGE = IMAGE.add_local_file(source, remote_path="/tmp/promotion/" + source.name)
APP = modal.App("midlayer-h200-tma-promotion-20261107")

@APP.function(gpu="H200", image=IMAGE, timeout=300, retries=0,
              max_containers=1, single_use_containers=True)
def capture():
    start = time.monotonic()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    with tempfile.TemporaryDirectory(prefix="midlayer-tma-promotion-") as directory:
        command = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
            "-I/tmp/promotion", "/tmp/promotion/tma_sector_promotion.cu", "-o", directory + "/probe"]
        build = subprocess.run(command, capture_output=True, text=True, timeout=90)
        result = {"gpu": gpu, "build_command": command, "build_exit_code": build.returncode,
            "build_stdout": build.stdout, "build_stderr": build.stderr, "records": []}
        if build.returncode == 0:
            result["binary_sha256"] = hashlib.sha256(Path(directory + "/probe").read_bytes()).hexdigest()
            for promotion in (0, 128, 256):
                for rounds in (8, 128):
                    args = ["ncu", "--csv", "--page", "raw", "--cache-control", "none",
                        "--clock-control", "none", "--kernel-name", "regex:tma_stream_kernel",
                        "--launch-skip", "2", "--launch-count", "1", "--metrics",
                        "lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum",
                        directory + "/probe", str(promotion), str(rounds)]
                    run = subprocess.run(args, capture_output=True, text=True, timeout=25)
                    result["records"].append({"promotion_bytes": promotion, "rounds": rounds,
                        "command": args, "exit_code": run.returncode,
                        "stdout": run.stdout, "stderr": run.stderr})
                    if run.returncode != 0:
                        result["function_wall_s"] = time.monotonic() - start
                        return result
        result["function_wall_s"] = time.monotonic() - start
        return result

@APP.local_entrypoint()
def main():
    target = ROOT / "profiler_output/microbench/nvidia_h200/tma_l2_promotion_20261107"
    if (target / "raw.json").exists():
        raise ValueError("refuse to overwrite captured evidence")
    result = capture.remote()
    result["sources"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in SOURCES}
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["production_duration_consumed"] = False
    target.mkdir(parents=True, exist_ok=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"build_exit_code": result["build_exit_code"],
        "exit_codes": [r["exit_code"] for r in result["records"]],
        "function_wall_s": result["function_wall_s"]}))
