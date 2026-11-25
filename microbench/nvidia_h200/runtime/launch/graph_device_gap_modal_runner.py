"""One authorized H200 independent launch-gap capture, without retries."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import modal

HERE = Path(__file__).resolve()
REMOTE = HERE.parent == Path("/root")
SOURCE = Path("/tmp/midlayer_graph_device_gap.cu") if REMOTE else HERE.with_name("graph_device_gap.cu")
ROOT = Path("/tmp") if REMOTE else HERE.parents[4]
CAMPAIGN = ROOT / "profiler_output/microbench/nvidia_h200/serving_accuracy_graph_gap_20261107"
IMAGE = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("cuda-nsight-systems-12-9")
         .add_local_file(SOURCE, remote_path="/tmp/midlayer_graph_device_gap.cu"))
APP = modal.App("midlayer-h200-serving-graph-gap-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=540, retries=0, max_containers=1, single_use_containers=True)
def capture(digest: str) -> dict:
    started = time.monotonic()
    source = Path("/tmp/midlayer_graph_device_gap.cu")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("uploaded independent probe changed")
    identity = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                              check=True, capture_output=True, text=True).stdout
    if "H200" not in identity:
        raise ValueError("authorized H200 is required")
    try:
        with tempfile.TemporaryDirectory(prefix="midlayer_graph_gap_", dir="/tmp") as temporary:
            root = Path(temporary)
            binary = root / "probe"
            commands = [["nvcc", "-std=c++17", "-O3", "-arch=sm_90", str(source), "-o", str(binary)],
                        [str(binary)],
                        ["nsys", "profile", "--trace=cuda", "--cuda-graph-trace=node", "--sample=none",
                         "--cpuctxsw=none", "--force-overwrite=true", "-o", str(root / "gap"), str(binary)]]
            runs = []
            for command in commands:
                process = subprocess.run(command, check=False, capture_output=True, text=True, timeout=150)
                runs.append({"command": command, "exit_code": process.returncode,
                             "stdout": process.stdout, "stderr": process.stderr})
                if process.returncode != 0:
                    return {"status": "failed", "identity": identity, "source_sha256": digest,
                            "runs": runs, "gpu_function_wall_s": time.monotonic() - started}
            report = root / "gap.nsys-rep"
            return {"status": "complete", "identity": identity, "source_sha256": digest,
                    "runs": runs, "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                    "report": report.read_bytes(), "gpu_function_wall_s": time.monotonic() - started}
    finally:
        source.unlink()


@APP.local_entrypoint()
def main():
    if CAMPAIGN.exists():
        raise FileExistsError(CAMPAIGN)
    started = time.monotonic()
    result = capture.remote(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    CAMPAIGN.mkdir(parents=True)
    if "report" in result:
        report = result.pop("report")
        if hashlib.sha256(report).hexdigest() != result["report_sha256"]:
            raise ValueError("returned raw report changed")
        (CAMPAIGN / "gap.nsys-rep").write_bytes(report)
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = hashlib.sha256(HERE.read_bytes()).hexdigest()
    result["budget_scope"] = "Serving independent US$25; <=10min1xH200/0.167GPU-hours/US$1.20; no retry; cumulative reservations<=US$3.60"
    (CAMPAIGN / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(CAMPAIGN), "status": result["status"],
                      "gpu_function_wall_s": result["gpu_function_wall_s"]}))
