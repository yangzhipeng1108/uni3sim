"""One budgeted H200 request-state/metadata Host mechanism capture."""
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
SOURCE = Path("/tmp/vllm030_v2_state_metadata_runner.py") if REMOTE else HERE.with_name("vllm030_v2_state_metadata_runner.py")
FIXTURE = Path("/tmp/vllm030_v2_input_prepare_runner.py") if REMOTE else HERE.with_name("vllm030_v2_input_prepare_runner.py")
ROOT = Path("/tmp") if REMOTE else HERE.parents[4]
CAMPAIGN = ROOT / "profiler_output/microbench/nvidia_h200/serving_accuracy_v2_state_metadata_20261108"
KEY = "1107-serving-v2-state-metadata-20261108-01"
IMAGE = (modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12").entrypoint([])
    .add_local_file(SOURCE, remote_path="/tmp/vllm030_v2_state_metadata_runner.py")
    .add_local_file(FIXTURE, remote_path="/tmp/vllm030_v2_input_prepare_runner.py"))
APP = modal.App("midlayer-h200-serving-v2-state-metadata-20261108")


@APP.function(gpu="H200", image=IMAGE, cpu=2, memory=4096, timeout=480, retries=0,
    max_containers=1, single_use_containers=True)
def measure(expected_shas: list[str]) -> dict:
    started = time.monotonic()
    try:
        if [hashlib.sha256(path.read_bytes()).hexdigest() for path in (SOURCE, FIXTURE)] != expected_shas:
            raise ValueError("uploaded source/fixture changed")
        identity = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True)
        if "H200" not in identity:
            raise ValueError("authorized H200 required")
        with tempfile.TemporaryDirectory(prefix="midlayer_v2_state_metadata_", dir="/tmp") as directory:
            work = Path(directory)
            output = work / "raw.json"
            environment = dict(os.environ)
            environment["VLLM_FLASH_ATTN_VERSION"] = "3"
            for name, subdir in (("CUDA_CACHE_PATH", "cuda-cache"), ("TRITON_CACHE_DIR", "triton"),
                    ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("XDG_CACHE_HOME", "cache"), ("TMPDIR", "temporary")):
                path = work / subdir
                path.mkdir()
                environment[name] = str(path)
            locate = subprocess.check_output(["/usr/bin/python3", "-c",
                "import pathlib,vllm; print(pathlib.Path(vllm.__file__).parent.parent)"], text=True)
            command = ["/usr/bin/python3", str(SOURCE), "--source-root", locate.strip().splitlines()[-1], "--output", str(output)]
            result = subprocess.run(command, cwd=work, env=environment, capture_output=True,
                text=True, check=False, timeout=420)
            return {"status": "complete" if result.returncode == 0 and output.is_file() else "probe_failed",
                "gpu_identity": identity, "command": command, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "source_sha256": expected_shas,
                "raw": json.loads(output.read_text()) if output.is_file() else None,
                "gpu_function_wall_s": time.monotonic() - started}
    finally:
        for path in (SOURCE, FIXTURE):
            if path.exists():
                path.unlink()


@APP.local_entrypoint()
def main(attempt: str = ""):
    if attempt not in ("", "attempt_2"):
        raise ValueError("only the original call and one diagnosed retry are declared")
    output = CAMPAIGN / attempt
    if (output / "raw.json").exists():
        raise FileExistsError(output / "raw.json")
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    result = measure.remote([hashlib.sha256(path.read_bytes()).hexdigest() for path in (SOURCE, FIXTURE)])
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = hashlib.sha256(HERE.read_bytes()).hexdigest()
    result["budget_key"] = KEY if not attempt else "1107-serving-v2-state-metadata-20261108-02"
    result["conservative_hold_usd"] = 1.10
    result["authorized_gpu_bound"] = {"model": "H200", "cards": 1, "seconds": 600, "gpu_hours": 1 / 6,
        "automatic_retries": 0, "persistent_storage": False}
    (output / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(output / "raw.json"), "status": result["status"],
        "gpu_function_wall_s": result["gpu_function_wall_s"]}))
