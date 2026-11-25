"""Run the no-weight vLLM 0.30 Host renderer probe on the authorized Modal H200."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import modal


SOURCE = Path(__file__).resolve()
_REMOTE_MODULE = len(SOURCE.parents) < 5
ROOT = Path("/opt/midlayer") if _REMOTE_MODULE else SOURCE.parents[4]
CAMPAIGN = (ROOT / "profiler_output/microbench/nvidia_h200/"
            "vllm030_request_path_20261102")
if _REMOTE_MODULE:
    IMAGE = modal.Image.debian_slim()
else:
    IMAGE = (
        modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12")
        .entrypoint([])
        .add_local_file(
            ROOT / "microbench/nvidia_h200/runtime/host_request_path/"
            "vllm030_render_service.py",
            remote_path="/opt/midlayer/vllm030_render_service.py")
        .add_local_dir(CAMPAIGN / "source_inputs",
                       remote_path="/opt/midlayer/source_inputs")
    )
APP = modal.App("midlayer-h200-vllm030-request-path-20261102")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@APP.function(gpu="H200", image=IMAGE, timeout=300, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    root = Path("/opt/midlayer")
    helper = root / "vllm030_render_service.py"
    inputs = root / "source_inputs"
    result = {
        "status": "started", "hardware_id": "nvidia_h200",
        "measurement_boundary": "vllm030_HfRenderer_render_cmpl_async_host_only",
        "helper_sha256": _sha256(helper),
        "input_manifest_sha256": _sha256(inputs / "inputs_manifest.json"),
    }
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    result["gpu_query"] = {"returncode": gpu.returncode,
                           "stdout": gpu.stdout, "stderr": gpu.stderr}
    if gpu.returncode != 0 or not gpu.stdout.startswith("NVIDIA H200,"):
        result["status"] = "gpu_identity_failed"
        return result
    with tempfile.TemporaryDirectory(prefix="midlayer-v030-render-",
                                     dir="/tmp") as directory:
        output = Path(directory) / "raw.json"
        command = [
            "/usr/bin/python3", str(helper),
            "--input-dir", str(inputs), "--output", str(output),
            "--samples", "128", "--warmup", "16",
        ]
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=180)
        result["command"] = command
        result["command_returncode"] = process.returncode
        result["command_stdout"] = process.stdout[-4000:]
        result["command_stderr"] = process.stderr[-4000:]
        if process.returncode != 0 or not output.is_file():
            result["status"] = "measurement_failed"
            return result
        raw = json.loads(output.read_text())
        if (raw["software"]["vllm"].split("+", 1)[0] != "0.30.0"
                or raw["model_weights_loaded"] is not False
                or raw["production_or_evaluation_duration_consumed"] is not False
                or raw["local_source_preflight"] is not False
                or len(raw["rows"]) != 4):
            result["status"] = "measurement_identity_failed"
            return result
        result["measurement"] = raw
        result["status"] = "complete"
        return result


@APP.local_entrypoint()
def main(output_dir: str) -> None:
    if ("MIDLAYER_H200_PAID_PROBE_CONTROLLER" not in os.environ
            or os.environ["MIDLAYER_H200_PAID_PROBE_CONTROLLER"] != "1"):
        raise ValueError("paid request-path probe requires the budget controller")
    target = Path(output_dir).resolve()
    if (target.parent != CAMPAIGN.resolve()
            or re.fullmatch(r"attempt_[A-Za-z0-9_]+", target.name) is None):
        raise ValueError("output must be a registered request-path attempt")
    if target.exists():
        raise FileExistsError(target)
    result = capture.remote()
    target.mkdir(parents=True)
    result["wrapper_sha256"] = _sha256(SOURCE)
    raw = target / "raw.json"
    raw.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "raw_path": str(raw)}))
    if result["status"] != "complete":
        raise RuntimeError(f"request-path probe failed: {result['status']}")
