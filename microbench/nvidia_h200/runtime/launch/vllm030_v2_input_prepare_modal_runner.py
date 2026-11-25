"""One budgeted H200 measurement of independent V2 input preparation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import modal

HERE = Path(__file__).resolve()
REMOTE = HERE.parent == Path("/root")
SOURCE = Path("/tmp/midlayer_v2_input_prepare.py") if REMOTE else HERE.with_name("vllm030_v2_input_prepare_runner.py")
ROOT = Path("/tmp") if REMOTE else HERE.parents[4]
CAMPAIGN = ROOT / "profiler_output/microbench/nvidia_h200/serving_accuracy_v2_input_prepare_20261107"
IMAGE = (modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12")
         .entrypoint([]).add_local_file(SOURCE, remote_path="/tmp/midlayer_v2_input_prepare.py"))
APP = modal.App("midlayer-h200-serving-v2-input-prepare-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=540, retries=0,
              max_containers=1, single_use_containers=True)
def measure(digest: str) -> dict:
    started = time.monotonic()
    source = Path("/tmp/midlayer_v2_input_prepare.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("uploaded probe changed")
    identity = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                              check=True, capture_output=True, text=True).stdout
    if "H200" not in identity:
        raise ValueError("authorized H200 required")
    try:
        with tempfile.TemporaryDirectory(prefix="midlayer_v2_prepare_", dir="/tmp") as temporary:
            work = Path(temporary)
            output = work / "raw.json"
            environment = dict(os.environ)
            environment["TRITON_CACHE_DIR"] = str(work / "triton")
            locate = subprocess.run(["/usr/bin/python3", "-c",
                                     "import pathlib,vllm; print(pathlib.Path(vllm.__file__).parent.parent)"],
                                    check=True, capture_output=True, text=True)
            command = ["/usr/bin/python3", str(source), "--source-root", locate.stdout.strip().splitlines()[-1],
                       "--device", "cuda", "--samples", "64", "--output", str(output)]
            completed = subprocess.run(command, cwd=work, env=environment, check=False,
                                       capture_output=True, text=True, timeout=420)
            result = {"status": "complete" if completed.returncode == 0 and output.is_file() else "probe_failed",
                      "gpu_identity": identity, "command": command, "exit_code": completed.returncode,
                      "stdout": completed.stdout, "stderr": completed.stderr,
                      "source_sha256": digest,
                      "raw": json.loads(output.read_text()) if output.is_file() else None,
                      "gpu_function_wall_s": time.monotonic() - started}
            return result
    finally:
        source.unlink()


@APP.local_entrypoint()
def main():
    if (CAMPAIGN / "raw.json").exists():
        raise FileExistsError(CAMPAIGN / "raw.json")
    started = time.monotonic()
    result = measure.remote(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = hashlib.sha256(HERE.read_bytes()).hexdigest()
    result["budget_scope"] = ("Serving US$25; one1xH200 reservation<=10min/0.167GPU-hours/US$1.20; "
                              "cumulative<=US$8.40; no automatic retry or persistent storage; invoice unavailable")
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(CAMPAIGN / "raw.json"), "status": result["status"],
                      "gpu_function_wall_s": result["gpu_function_wall_s"]}))
