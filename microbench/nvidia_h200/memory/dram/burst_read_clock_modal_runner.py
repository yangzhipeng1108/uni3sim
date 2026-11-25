"""One explicitly budgeted H200 memory-duty-cycle diagnostic, no model."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import modal

HERE = Path(__file__).resolve()
REMOTE = HERE.parent == Path("/root")
ROOT = Path("/tmp") if REMOTE else HERE.parents[4]
SOURCE = Path("/tmp/midlayer_burst_read_clock.cu") if REMOTE else HERE.with_name("burst_read_clock.cu")
CAMPAIGN = ROOT / "profiler_output/microbench/nvidia_h200/serving_accuracy_burst_memory_20261107"
IMAGE = modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11").add_local_file(
    SOURCE, remote_path="/tmp/midlayer_burst_read_clock.cu")
APP = modal.App("midlayer-h200-serving-accuracy-burst-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=540, retries=0, max_containers=1)
def measure(source_sha256: str) -> dict:
    started = time.monotonic()
    source = Path("/tmp/midlayer_burst_read_clock.cu")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("uploaded independent probe source identity changed")
    identity = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.mem",
        "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout
    if "H200" not in identity:
        raise ValueError("the authorized H200 is required")
    try:
        with tempfile.TemporaryDirectory(prefix="midlayer_burst_", dir="/tmp") as temporary:
            binary = str(Path(temporary) / "burst_read_clock")
            build = subprocess.run(["nvcc", "-std=c++17", "-O3", "-arch=sm_90", str(source), "-o", binary],
                check=False, capture_output=True, text=True, timeout=120)
            if build.returncode:
                return {"status": "compile_failed", "gpu_identity": identity,
                    "compile_stdout": build.stdout, "compile_stderr": build.stderr,
                    "gpu_function_wall_s": time.monotonic() - started}
            run = subprocess.run([binary], check=False, capture_output=True, text=True, timeout=240)
            return {"status": "complete" if run.returncode == 0 else "probe_failed",
                "gpu_identity": identity, "exit_code": run.returncode,
                "stdout": run.stdout, "stderr": run.stderr,
                "source_sha256": source_sha256,
                "measurement_boundary": "cuda_events_around_one_graph_read_stream_and_in_kernel_clock64_globaltimer",
                "production_or_evaluation_duration_consumed": False,
                "gpu_function_wall_s": time.monotonic() - started}
    finally:
        source.unlink()


@APP.local_entrypoint()
def main():
    if CAMPAIGN.exists():
        raise FileExistsError(CAMPAIGN)
    started = time.monotonic()
    result = measure.remote(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["budget_scope"] = "Serving independent US$25; this run max10min1xH200, estimate<=US$1.20; no retry"
    CAMPAIGN.mkdir(parents=True)
    (CAMPAIGN / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(CAMPAIGN / "raw.json"), "status": result["status"],
        "gpu_function_wall_s": result["gpu_function_wall_s"]}))
