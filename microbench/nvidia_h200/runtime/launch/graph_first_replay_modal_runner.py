"""One budgeted H200 run of a standalone default-Graph first-replay mechanism."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

import modal

HERE = Path(__file__).resolve()
REMOTE = HERE.parent == Path("/root")
SOURCE = Path("/tmp/midlayer_graph_first_replay.py") if REMOTE else HERE.with_name("graph_first_replay_runner.py")
ROOT = Path("/tmp") if REMOTE else HERE.parents[4]
CAMPAIGN = ROOT / "profiler_output/microbench/nvidia_h200/serving_accuracy_graph_first_replay_20261107"
IMAGE = (modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12").entrypoint([])
         .add_local_file(SOURCE, remote_path="/tmp/midlayer_graph_first_replay.py"))
APP = modal.App("midlayer-h200-serving-graph-first-replay-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=540, retries=0, max_containers=1, single_use_containers=True)
def measure(digest: str) -> dict:
    started = time.monotonic()
    source = Path("/tmp/midlayer_graph_first_replay.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("uploaded mechanism changed")
    identity = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                              check=True, capture_output=True, text=True).stdout
    if "H200" not in identity:
        raise ValueError("authorized H200 required")
    try:
        with tempfile.TemporaryDirectory(prefix="midlayer_graph_first_replay_", dir="/tmp") as temporary:
            work = Path(temporary)
            output = work / "raw.json"
            environment = dict(os.environ)
            environment["TRITON_CACHE_DIR"] = str(work / "triton")
            command = ["/usr/bin/python3", str(source), "--output", str(output)]
            completed = subprocess.run(command, cwd=work, env=environment,
                                       check=False, capture_output=True, text=True, timeout=420)
            raw = output.read_bytes() if output.is_file() else b""
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(work.rglob("*")):
                    if path.is_file() and (path.parent == work / "compiler" or path.suffix == ".json" and path != output):
                        archive.write(path, path.relative_to(work).as_posix())
            return {"status": "complete" if completed.returncode == 0 and raw else "probe_failed",
                    "identity": identity, "command": command, "exit_code": completed.returncode,
                    "stdout": completed.stdout, "stderr": completed.stderr, "source_sha256": digest,
                    "raw_gzip": gzip.compress(raw, mtime=0), "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "artifact_bundle": bundle.getvalue(), "gpu_function_wall_s": time.monotonic() - started}
    finally:
        source.unlink()


@APP.local_entrypoint()
def main():
    if (CAMPAIGN / "execution.json").exists():
        raise FileExistsError(CAMPAIGN / "execution.json")
    started = time.monotonic()
    result = measure.remote(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    raw, artifacts = result.pop("raw_gzip"), result.pop("artifact_bundle")
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = hashlib.sha256(HERE.read_bytes()).hexdigest()
    result["raw_gzip_sha256"] = hashlib.sha256(raw).hexdigest()
    result["compiler_artifacts_sha256"] = hashlib.sha256(artifacts).hexdigest()
    result["budget_scope"] = "Serving US$25;1xH200<=10min/0.167GPU-hours/US$1.20; cumulative<=US$10.80; no retry/persistent storage; invoice unavailable"
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / "raw.json.gz").write_bytes(raw)
    (CAMPAIGN / "compiler_artifacts.zip").write_bytes(artifacts)
    (CAMPAIGN / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(CAMPAIGN), "status": result["status"],
                      "gpu_function_wall_s": result["gpu_function_wall_s"]}))
