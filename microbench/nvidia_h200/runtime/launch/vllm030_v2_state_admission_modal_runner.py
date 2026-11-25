"""One H200 invocation, two independent pure request-state process rotations."""
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
NAMES = ("vllm030_v2_state_admission_runner.py", "vllm030_v2_state_metadata_runner.py", "vllm030_v2_input_prepare_runner.py")
SOURCES = tuple((Path("/tmp") if REMOTE else HERE.parent) / name for name in NAMES)
ROOT = Path("/tmp") if REMOTE else HERE.parents[4]
CAMPAIGN = ROOT / "profiler_output/microbench/nvidia_h200/serving_accuracy_v2_state_admission_20261108"
KEY = "1107-serving-v2-admission-20261108-01"
IMAGE = modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12").entrypoint([])
for source in SOURCES:
    IMAGE = IMAGE.add_local_file(source, remote_path="/tmp/" + source.name)
APP = modal.App("midlayer-h200-serving-v2-state-admission-20261108")


@APP.function(gpu="H200", image=IMAGE, cpu=2, memory=4096, timeout=480, retries=0,
    max_containers=1, single_use_containers=True)
def measure(digests: list[str]) -> dict:
    started = time.monotonic()
    runs = []
    try:
        if [hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES] != digests:
            raise ValueError("uploaded source/fixture changed")
        identity = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True)
        if "H200" not in identity:
            raise ValueError("authorized H200 required")
        with tempfile.TemporaryDirectory(prefix="midlayer_v2_admission_", dir="/tmp") as directory:
            work = Path(directory)
            environment = dict(os.environ)
            for name, subdir in (("CUDA_CACHE_PATH", "cuda-cache"), ("TRITON_CACHE_DIR", "triton"),
                    ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("XDG_CACHE_HOME", "cache"), ("TMPDIR", "temporary")):
                path = work / subdir
                path.mkdir()
                environment[name] = str(path)
            locate = subprocess.check_output(["/usr/bin/python3", "-c",
                "import pathlib,vllm; print(pathlib.Path(vllm.__file__).parent.parent)"], text=True)
            for rotation in (0, 1):
                output = work / ("rotation" + str(rotation) + ".json")
                command = ["/usr/bin/python3", str(SOURCES[0]), "--source-root", locate.strip().splitlines()[-1],
                    "--rotation", str(rotation), "--output", str(output)]
                completed = subprocess.run(command, cwd=work, env=environment, capture_output=True,
                    text=True, check=False, timeout=180)
                runs.append({"command": command, "exit_code": completed.returncode,
                    "stdout": completed.stdout, "stderr": completed.stderr,
                    "raw": json.loads(output.read_text()) if output.is_file() else None})
                if completed.returncode != 0:
                    break
            complete = len(runs) == 2 and all(row["exit_code"] == 0 and row["raw"]["status"] == "complete" for row in runs)
            return {"status": "complete" if complete else "probe_failed", "gpu_identity": identity,
                "source_sha256": digests, "runs": runs, "gpu_function_wall_s": time.monotonic() - started}
    finally:
        for source in SOURCES:
            if source.exists():
                source.unlink()


@APP.local_entrypoint()
def main():
    if (CAMPAIGN / "raw.json").exists():
        raise FileExistsError(CAMPAIGN / "raw.json")
    started = time.monotonic()
    result = measure.remote([hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES])
    result.update(client_wall_s=time.monotonic() - started, budget_key=KEY,
        conservative_hold_usd=1.10, upper_gpu_hours=1 / 6, automatic_retries=0, volume=False,
        modal_runner_sha256=hashlib.sha256(HERE.read_bytes()).hexdigest())
    (CAMPAIGN / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(CAMPAIGN / "raw.json"), "status": result["status"],
        "gpu_function_wall_s": result["gpu_function_wall_s"]}))
