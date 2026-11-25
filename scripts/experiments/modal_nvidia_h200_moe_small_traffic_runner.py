"""One budgeted H200 capture of non-duration BF16 W1/W2 counters."""
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
SOURCE = Path("/tmp/midlayer_moe_small_traffic.py") if REMOTE else HERE.with_name("nvidia_h200_moe_small_traffic_runner.py")
ROOT = Path("/tmp") if REMOTE else HERE.parents[2]
CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/serving_accuracy_moe_small_traffic_20261107"
IMAGE = (modal.Image.from_registry("vllm/vllm-openai:v0.30.0", add_python="3.12")
    .entrypoint([]).apt_install("cuda-nsight-compute-12-9")
    .add_local_file(SOURCE, remote_path="/tmp/midlayer_moe_small_traffic.py"))
APP = modal.App("midlayer-h200-serving-moe-traffic-20261107")


@APP.function(gpu="H200", image=IMAGE, timeout=540, retries=0, max_containers=1, single_use_containers=True)
def capture(digest: str) -> dict:
    started = time.monotonic()
    source = Path("/tmp/midlayer_moe_small_traffic.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("uploaded fixture source changed")
    identity = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout
    if "H200" not in identity:
        raise ValueError("authorized H200 required")
    result = {"status": "failed", "identity": identity, "source_sha256": digest, "runs": [], "artifact_bundle": b""}
    try:
        with tempfile.TemporaryDirectory(prefix="midlayer_moe_traffic_", dir="/tmp") as temporary:
            work = Path(temporary)
            environment = dict(os.environ)
            environment["TRITON_CACHE_DIR"] = str(work / "triton")
            ncu_paths = tuple(path for path in Path("/opt/nvidia/nsight-compute").glob("*/ncu") if path.is_file())
            if len(ncu_paths) != 1:
                result["error"] = "expected exactly one installed Nsight Compute binary"
                return result
            for tokens in (1, 8, 32):
                output = work / f"fixture_t{tokens}.json"
                command = [str(ncu_paths[0]), "--csv", "--page", "raw", "--cache-control", "none", "--clock-control", "none",
                    "--nvtx", "--nvtx-include", "counter_frame/", "--kernel-name", "regex:fused_moe_kernel",
                    "--launch-count", "2", "--metrics", "lts__t_sectors_op_read.sum,dram__bytes_read.sum",
                    "/usr/bin/python3", str(source), "--tokens", str(tokens), "--output", str(output)]
                try:
                    completed = subprocess.run(command, cwd=work, env=environment,
                        check=False, capture_output=True, text=True, timeout=150)
                except subprocess.TimeoutExpired as error:
                    result["runs"].append({"tokens": tokens, "status": "timed_out", "command": command, "error": str(error)})
                    break
                result["runs"].append({"tokens": tokens, "command": command, "exit_code": completed.returncode,
                    "stdout": completed.stdout, "stderr": completed.stderr,
                    "fixture": json.loads(output.read_text()) if output.is_file() else None})
                if completed.returncode != 0 or not output.is_file():
                    break
            else:
                result["status"] = "captured_commands"
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted((work / "triton").rglob("*")):
                    if path.is_file() and path.suffix in (".json", ".ptx"):
                        archive.write(path, path.relative_to(work).as_posix())
            result["artifact_bundle"] = bundle.getvalue()
    finally:
        source.unlink()
        result["gpu_function_wall_s"] = time.monotonic() - started
    return result


@APP.local_entrypoint()
def main():
    if (CAMPAIGN / "raw.json").exists():
        raise FileExistsError(CAMPAIGN)
    started = time.monotonic()
    result = capture.remote(hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    artifact = result.pop("artifact_bundle")
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = hashlib.sha256(HERE.read_bytes()).hexdigest()
    result["budget_scope"] = "Serving US$25; one necessary tool-path repair retry; reservation<=10min1xH200/0.167GPU-hours/US$1.20; cumulative<=US$7.20; no automatic retry or persistent storage"
    result["counter_bundle_sha256"] = hashlib.sha256(artifact).hexdigest()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / "compiler_artifacts.zip").write_bytes(artifact)
    (CAMPAIGN / "raw.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(CAMPAIGN), "status": result["status"], "gpu_function_wall_s": result["gpu_function_wall_s"]}))
