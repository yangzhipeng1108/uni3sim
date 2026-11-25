"""Check whether this authorized H200 Modal host matches the source CPU."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import modal


APP = modal.App("midlayer-h200-modal-host-identity-20261029")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")


@APP.function(gpu="H200", image=IMAGE, timeout=90, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    started = time.monotonic()
    fields = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if not line.strip():
            break
        if ":" in line:
            name, value = line.split(":", 1)
            if name.strip() in ("vendor_id", "cpu family", "model", "model name",
                                "stepping", "cpu MHz"):
                fields[name.strip()] = value.strip()
    gpu = subprocess.run([
        "nvidia-smi", "--query-gpu=name,driver_version",
        "--format=csv,noheader"], check=True, capture_output=True,
        text=True, timeout=20)
    return {"cpu_first_processor": fields, "machine": platform.machine(),
            "kernel_release": platform.release(),
            "gpu_identity": gpu.stdout.strip(),
            "gpu_function_wall_s": time.monotonic() - started}


@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    target = Path(output_dir).resolve()
    parent = root / "profiler_output/microbench/nvidia_h200/modal_host_identity_20261029"
    if target != parent.resolve() or (target.exists() and (target / "raw.json").exists()):
        raise ValueError("output must be the registered Modal Host identity campaign")
    started = time.monotonic()
    result = capture.remote()
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["measurement_boundary"] = "environment_identity_only_no_duration_parameter"
    target.mkdir(parents=True, exist_ok=True)
    (target / "raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(target),
                      "cpu_model": result["cpu_first_processor"]["model name"],
                      "client_wall_s": result["client_wall_s"]}, sort_keys=True))
