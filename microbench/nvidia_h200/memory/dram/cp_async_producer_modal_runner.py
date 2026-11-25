"""One independently budgeted H200 asynchronous-copy capacity screen."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import modal

HERE = Path(__file__).resolve()
REMOTE = HERE.parent == Path("/root")
SOURCE = Path("/tmp/midlayer_cp_async_producer.cu") if REMOTE else HERE.with_name("cp_async_producer_capacity.cu")
ROOT = Path("/tmp") if REMOTE else HERE.parents[4]
CAMPAIGN = ROOT / "profiler_output/microbench/nvidia_h200/serving_accuracy_cp_async_20261107"
IMAGE = modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11").add_local_file(
    SOURCE, remote_path="/tmp/midlayer_cp_async_producer.cu")
APP = modal.App("midlayer-h200-serving-cp-async-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=540, retries=0, max_containers=1, single_use_containers=True)
def measure(digest: str) -> dict:
    started = time.monotonic()
    source = Path("/tmp/midlayer_cp_async_producer.cu")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("uploaded independent source changed")
    identity = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                              check=True, capture_output=True, text=True).stdout
    if "H200" not in identity:
        raise ValueError("authorized H200 is required")
    try:
        with tempfile.TemporaryDirectory(prefix="midlayer_cp_async_", dir="/tmp") as temporary:
            binary = Path(temporary) / "probe"
            runs = []
            for command in (["nvcc", "-std=c++17", "-O3", "-arch=sm_90", str(source), "-o", str(binary)], [str(binary)]):
                result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
                runs.append({"command": command, "exit_code": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr})
                if result.returncode != 0:
                    return {"status": "failed", "identity": identity, "source_sha256": digest,
                            "runs": runs, "gpu_function_wall_s": time.monotonic() - started}
            return {"status": "complete", "identity": identity, "source_sha256": digest,
                    "runs": runs, "gpu_function_wall_s": time.monotonic() - started}
    finally:
        source.unlink()


@APP.local_entrypoint()
def main():
    if CAMPAIGN.exists():
        raise FileExistsError(CAMPAIGN)
    started = time.monotonic()
    result = measure.remote(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = hashlib.sha256(HERE.read_bytes()).hexdigest()
    result["budget_scope"] = "Serving independent US$25; <=10min1xH200/0.167GPU-hours/US$1.20; no retry; cumulative reservations<=US$4.80"
    CAMPAIGN.mkdir(parents=True)
    (CAMPAIGN / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(CAMPAIGN), "status": result["status"],
                      "gpu_function_wall_s": result["gpu_function_wall_s"]}))
