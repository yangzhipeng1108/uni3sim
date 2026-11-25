"""Measure TMA operand address sharing, with no matrix or production times."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import modal

IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12"
).apt_install("cuda-nsight-compute-12-9")
if modal.is_local():
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parents[3]
    SOURCE = HERE / "tma_queue_service.cu"
    IMAGE = IMAGE.add_local_file(SOURCE, remote_path="/tmp/tma_queue_service.cu")
APP = modal.App("midlayer-h200-tma-sharing-queue-20261108")


@APP.function(gpu="H200", image=IMAGE, timeout=300, retries=0,
              single_use_containers=True)
def capture():
    started = time.monotonic()
    gpu = subprocess.run([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit",
        "--format=csv,noheader"], capture_output=True, text=True,
        check=True, timeout=20).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    environment = dict(os.environ)
    for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
        if name in environment:
            del environment[name]
    result = {"gpu": gpu, "captures": [], "counter_captures": []}
    with tempfile.TemporaryDirectory(prefix="midlayer-tma-sharing-") as directory:
        for counter in (False, True):
            for mask in (0, 1, 2, 3):
                binary = str(Path(directory) / f"probe_{mask}_{int(counter)}")
                command = [
                    "nvcc", "-O3", "-std=c++17",
                    "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v",
                    f"-DTMA_SHARED_OPERAND_MASK={mask}",
                    f"-DTMA_COUNTER_CAPTURE={int(counter)}",
                    "/tmp/tma_queue_service.cu", "-o", binary]
                build = subprocess.run(command, capture_output=True, text=True,
                    env=environment, timeout=30)
                item = {"shared_operand_mask": mask, "build_command": command,
                    "build_exit_code": build.returncode,
                    "build_stdout": build.stdout, "build_stderr": build.stderr,
                    "run_exit_code": None, "stdout": "", "stderr": ""}
                key = "counter_captures" if counter else "captures"
                result[key].append(item)
                if build.returncode != 0:
                    result["function_wall_s"] = time.monotonic() - started
                    return result
                item["binary_sha256"] = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
                dump = subprocess.run(["cuobjdump", "--dump-sass", binary],
                    capture_output=True, text=True, check=True, timeout=15)
                item["sass_stdout"] = dump.stdout
                if counter:
                    run_command = [
                        "ncu", "--csv", "--page", "raw", "--cache-control", "none",
                        "--clock-control", "none", "--kernel-name", "regex:queue_service",
                        "--launch-skip", "2", "--launch-count", "1", "--metrics",
                        "lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum",
                        binary, "4096", "4096", "4", "132", "1", "512"]
                else:
                    run_command = [binary]
                run = subprocess.run(run_command, capture_output=True, text=True,
                    env=environment, timeout=40)
                item.update({"run_command": run_command, "run_exit_code": run.returncode,
                    "stdout": run.stdout, "stderr": run.stderr})
                if run.returncode != 0:
                    result["function_wall_s"] = time.monotonic() - started
                    return result
    result["function_wall_s"] = time.monotonic() - started
    return result


@APP.local_entrypoint()
def main():
    target = ROOT / "profiler_output/microbench/nvidia_h200/tma_sharing_queue_20261108"
    if (target / "raw.json").exists():
        raise ValueError("refuse to overwrite captured evidence")
    target.mkdir(parents=True, exist_ok=True)
    snapshot = target / "source"
    snapshot.mkdir(exist_ok=True)
    for source in (SOURCE, Path(__file__)):
        destination = snapshot / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError("source snapshot conflict")
        destination.write_bytes(source.read_bytes())
    result = capture.remote()
    result["source_sha256"] = hashlib.sha256((snapshot / SOURCE.name).read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256((snapshot / Path(__file__).name).read_bytes()).hexdigest()
    result["production_duration_consumed"] = False
    for kind in ("captures", "counter_captures"):
        for item in result[kind]:
            if "sass_stdout" in item:
                data = item["sass_stdout"].encode()
                filename = f"{kind}_{item['shared_operand_mask']}.sass.txt.gz"
                (target / filename).write_bytes(gzip.compress(data, mtime=0))
                item["sass_stdout_sha256"] = hashlib.sha256(data).hexdigest()
                item["sass_path"] = filename
                del item["sass_stdout"]
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"capture_count": len(result["captures"]),
        "counter_capture_count": len(result["counter_captures"]),
        "function_wall_s": result["function_wall_s"]}))
