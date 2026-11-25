"""One authorized1xH200 reference/detailed Serving boundary campaign."""
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
ROOT = Path("/tmp") if REMOTE else HERE.parents[2]
PROBE = (Path("/tmp/midlayer_boundary_probe.py") if REMOTE else
         ROOT / "dataset_pipeline/profiler/nvidia_h200_serving_boundary_probe.py")
HOOKS = (Path("/tmp/midlayer_boundary_hooks.py") if REMOTE else
         ROOT / "dataset_pipeline/profiler/nvidia_h200_serving_boundary_hooks.py")
CAMPAIGN = ROOT / "profiler_output/serving/nvidia_h200/serving_accuracy_qwen4_boundary_20261107"
GEOMETRY = Path("/tmp/midlayer_boundary_geometry.json") if REMOTE else CAMPAIGN / "requests_geometry.json"
IMAGE = (modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12").entrypoint([])
    .add_local_file(PROBE, remote_path="/tmp/midlayer_boundary_probe.py")
    .add_local_file(HOOKS, remote_path="/tmp/midlayer_boundary_hooks.py")
    .add_local_file(GEOMETRY, remote_path="/tmp/midlayer_boundary_geometry.json"))
APP = modal.App("midlayer-h200-serving-qwen4-boundary-20261107")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@APP.function(gpu="H200", cpu=8, memory=32768, image=IMAGE, timeout=2640,
              retries=0, max_containers=1, single_use_containers=True)
def capture(hashes, commit):
    started = time.monotonic()
    sources = (Path("/tmp/midlayer_boundary_probe.py"), Path("/tmp/midlayer_boundary_hooks.py"),
               Path("/tmp/midlayer_boundary_geometry.json"))
    if [sha(path) for path in sources] != hashes:
        raise ValueError("uploaded frozen collector/input differs")
    identity = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout
    if "H200" not in identity:
        raise ValueError("authorized H200 required")
    try:
        with tempfile.TemporaryDirectory(prefix="midlayer_serving_boundary_", dir="/tmp") as temporary:
            work = Path(temporary)
            environment = dict(os.environ)
            environment["HF_HOME"] = str(work / "hf_home")
            environment["VLLM_NO_USAGE_STATS"] = "1"
            command = ["/usr/bin/python3", str(sources[0]), "--hooks", str(sources[1]),
                "--geometry", str(sources[2]), "--output", str(work / "candidate")]
            timed_out = False
            with (work / "collector.log").open("w") as log:
                process = subprocess.Popen(command, cwd=work, env=environment, stdout=log,
                    stderr=subprocess.STDOUT)
                try:
                    process.wait(timeout=2460)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    import psutil
                    children = psutil.Process(process.pid).children(recursive=True)
                    for child in reversed(children):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    process.terminate()
                    _, alive = psutil.wait_procs(children, timeout=20)
                    for child in alive:
                        child.kill()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=10)
            bundle = io.BytesIO()
            retained = []
            with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(work.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(work)
                    if "hf" in relative.parts or "compile_cache" in relative.parts or "hf_home" in relative.parts:
                        continue
                    if path.suffix in (".log", ".json", ".bin"):
                        archive.write(path, relative.as_posix())
                        retained.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                                         "sha256": sha(path)})
            receipt_path = work / "candidate/receipt.json"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
            complete = (process.returncode == 0 and receipt is not None and len(receipt["runs"]) == 2
                        and all(run["completed_requests"] == 128 for run in receipt["runs"]))
            return {"status": "complete_candidate" if complete else "capture_failed", "identity": identity,
                "source_commit": commit, "source_sha256": hashes, "command": command,
                "exit_code": process.returncode, "timed_out": timed_out,
                "gpu_function_wall_s": time.monotonic() - started,
                "files": retained, "artifact_bundle": bundle.getvalue(),
                "weight_and_compiler_caches_retained": False}
    finally:
        for source in sources:
            source.unlink()


@APP.local_entrypoint()
def main():
    if (CAMPAIGN / "execution.json").exists():
        raise FileExistsError("single campaign attempt already exists")
    started = time.monotonic()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = capture.remote([sha(path) for path in (PROBE, HOOKS, GEOMETRY)], commit)
    payload = result.pop("artifact_bundle")
    result.update(client_wall_s=time.monotonic() - started,
        bundle_sha256=hashlib.sha256(payload).hexdigest(), runner_sha256=sha(HERE),
        billing="Serving1xH200<=45min/0.75GPU-hours/US$5.40;cumulative<=US$16.20/25;no automatic retry/persistent storage;invoice unavailable")
    (CAMPAIGN / "evidence.zip").write_bytes(payload)
    (CAMPAIGN / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "path": str(CAMPAIGN),
        "gpu_function_wall_s": result["gpu_function_wall_s"]}), flush=True)
