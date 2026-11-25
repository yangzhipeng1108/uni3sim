"""Compare finite TMA streams after disjoint-copy versus ALU activity."""
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
    SOURCE = HERE / "tma_queue_service.cu"
    IMAGE = IMAGE.add_local_file(SOURCE, remote_path="/tmp/tma_queue_service.cu")
APP = modal.App("midlayer-h200-tma-activity-transition-20261108")


@APP.function(gpu="H200", image=IMAGE, timeout=150, retries=0,
              single_use_containers=True)
def capture():
    start = time.monotonic()
    query = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem",
        "--format=csv,noheader"]
    gpu = subprocess.run(query, capture_output=True, text=True, check=True, timeout=15).stdout
    if not gpu.startswith("NVIDIA H200,"):
        raise ValueError("one H200 required")
    environment = dict(os.environ)
    for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
        if name in environment:
            del environment[name]
    result = {"gpu_before": gpu, "captures": []}
    with tempfile.TemporaryDirectory(prefix="midlayer-tma-activity-") as directory:
        for mode in (1, 2):
            binary = str(Path(directory) / f"probe_{mode}")
            command = ["nvcc", "-O3", "-std=c++17",
                "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v",
                f"-DTMA_ACTIVITY_PRELUDE={mode}", "/tmp/tma_queue_service.cu", "-o", binary]
            build = subprocess.run(command, capture_output=True, text=True,
                env=environment, timeout=30)
            item = {"activity_prelude": mode, "build_command": command,
                "build_exit_code": build.returncode, "build_stdout": build.stdout,
                "build_stderr": build.stderr, "run_exit_code": None,
                "stdout": "", "stderr": ""}
            result["captures"].append(item)
            if build.returncode != 0:
                break
            item["binary_sha256"] = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
            dump = subprocess.run(["cuobjdump", "--dump-sass", binary],
                capture_output=True, text=True, check=True, timeout=15)
            item["sass_stdout"] = dump.stdout
            run = subprocess.run([binary], capture_output=True, text=True,
                env=environment, timeout=40)
            item.update({"run_exit_code": run.returncode,
                "stdout": run.stdout, "stderr": run.stderr})
            if run.returncode != 0:
                break
    result["gpu_after"] = subprocess.run(query, capture_output=True,
        text=True, check=True, timeout=15).stdout
    result["function_wall_s"] = time.monotonic() - start
    return result


@APP.local_entrypoint()
def main(campaign_name: str = "tma_activity_transition_20261108"):
    if campaign_name not in ("tma_activity_transition_20261108",
            "tma_activity_transition_full_20261108",
            "tma_activity_transition_wide_20261108"):
        raise ValueError("activity campaign name outside the owned scope")
    target = ROOT / "profiler_output/microbench/nvidia_h200" / campaign_name
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
    for item in result["captures"]:
        if "sass_stdout" in item:
            data = item["sass_stdout"].encode()
            filename = f"prelude{item['activity_prelude']}.sass.txt.gz"
            (target / filename).write_bytes(gzip.compress(data, mtime=0))
            item["sass_stdout_sha256"] = hashlib.sha256(data).hexdigest()
            item["sass_path"] = filename
            del item["sass_stdout"]
    (target / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"capture_count": len(result["captures"]),
        "function_wall_s": result["function_wall_s"]}))
