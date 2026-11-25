"""Measure one Modal H200 Host's launch and runtime-prepare mechanisms.

The Modal CPU SKU may be hidden by gVisor. Results remain candidate evidence
unless the CPU and software environment match the pinned original capture.
"""

import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import modal


SOURCE = Path(__file__).resolve()
REMOTE = SOURCE.parent == Path("/root")
ROOT = Path("/tmp") if REMOTE else SOURCE.parents[4]
LAUNCH = (Path("/tmp/midlayer_host_launch_runner.py") if REMOTE else
          ROOT / "microbench/nvidia_h200/runtime/launch/torch_eager_graph_runner.py")
PREPARE = (Path("/tmp/midlayer_host_prepare_runner.py") if REMOTE else
           ROOT / "microbench/nvidia_a800/runtime/launch/host_runtime_prepare_runner.py")
IMAGE = modal.Image.from_registry(
    "vllm/vllm-openai:v0.30.0", add_python="3.12"
).entrypoint([]).add_local_file(
    LAUNCH, remote_path="/tmp/midlayer_host_launch_runner.py"
).add_local_file(
    PREPARE, remote_path="/tmp/midlayer_host_prepare_runner.py"
)
APP = modal.App("midlayer-h200-host-comparison-20261029")
PYTHON = "/usr/bin/python3"
CAMPAIGN = ROOT / "profiler_output/microbench/nvidia_h200/modal_host_comparison_20261029"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture(command: list[str], timeout: int) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=timeout, check=False)
        return {"command": command, "exit_code": result.returncode,
                "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:],
                "wall_s": time.monotonic() - started}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "exit_code": None,
                "error": f"timeout after {timeout}s: {error}",
                "wall_s": time.monotonic() - started}


@APP.function(gpu="H200", image=IMAGE, timeout=660, retries=0,
              max_containers=1, single_use_containers=True)
def measure(baseline_commit: str, launch_source_sha256: str,
            prepare_source_sha256: str) -> dict:
    started = time.monotonic()
    launch = Path("/tmp/midlayer_host_launch_runner.py")
    prepare = Path("/tmp/midlayer_host_prepare_runner.py")
    if _sha256(launch) != launch_source_sha256 or _sha256(prepare) != prepare_source_sha256:
        raise RuntimeError("uploaded probe source differs from the local snapshot")
    identity = _capture([
        PYTHON, "-c", "import json, pathlib, platform, torch, vllm; "
        "print(json.dumps({'cpuinfo': pathlib.Path('/proc/cpuinfo').read_text().split('\\n\\n',1)[0], "
        "'kernel': platform.release(), 'torch': torch.__version__, "
        "'cuda': torch.version.cuda, 'vllm': vllm.__version__}))"], 45)
    gpu = _capture(["nvidia-smi", "--query-gpu=name,driver_version",
                    "--format=csv,noheader"], 20)
    payload = {"measurement_boundary": "independent_host_mechanisms_no_model",
               "identity": identity, "gpu": gpu,
               "launch_source_sha256": launch_source_sha256,
               "prepare_source_sha256": prepare_source_sha256,
               "baseline_commit": baseline_commit, "launch_runs": [],
               "runtime_prepare": None}
    if identity["exit_code"] != 0 or gpu["exit_code"] != 0:
        payload["status"] = "identity_failed"
        payload["gpu_function_wall_s"] = time.monotonic() - started
        return payload
    with tempfile.TemporaryDirectory(prefix="midlayer_h200_host_", dir="/tmp") as temp:
        owner = Path(temp) / "profiler_output/microbench/nvidia_h200"
        owner.mkdir(parents=True)
        for index in range(3):
            output = owner / f"launch_{index + 1}"
            run = _capture([PYTHON, str(launch), "--output-dir", str(output),
                            "--device-id", "0", "--baseline-commit", baseline_commit,
                            "--expected-source-sha256", launch_source_sha256], 120)
            if run["exit_code"] == 0:
                run["raw"] = json.loads((output / "manifest.json").read_text())
            payload["launch_runs"].append(run)
            if run["exit_code"] != 0:
                payload["status"] = "launch_failed"
                payload["gpu_function_wall_s"] = time.monotonic() - started
                return payload
        eager_us = statistics.median(
            run["raw"]["summary"]["eager:512"]["median"] / 1000
            for run in payload["launch_runs"])
        output = owner / "runtime_prepare"
        run = _capture([PYTHON, str(prepare), "--hardware-id", "nvidia_h200",
                        "--device-id", "0", "--eager-enqueue-us", str(eager_us),
                        "--expected-vllm-version", "0.30.0", "--trials", "5",
                        "--output-dir", str(output)], 420)
        if run["exit_code"] == 0:
            run["raw"] = json.loads((output / "summary.json").read_text())
        payload["runtime_prepare"] = run
        payload["status"] = "complete" if run["exit_code"] == 0 else "runtime_prepare_failed"
    payload["gpu_function_wall_s"] = time.monotonic() - started
    return payload


@APP.local_entrypoint()
def main() -> None:
    if CAMPAIGN.exists():
        raise FileExistsError(CAMPAIGN)
    baseline_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True, cwd=ROOT).stdout.strip()
    started = time.monotonic()
    result = measure.remote(baseline_commit, _sha256(LAUNCH), _sha256(PREPARE))
    result["client_wall_s"] = time.monotonic() - started
    result["runner_sha256"] = _sha256(Path(__file__).resolve())
    CAMPAIGN.mkdir(parents=True)
    (CAMPAIGN / "raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(CAMPAIGN / "raw.json"),
                      "status": result["status"],
                      "gpu_function_wall_s": result["gpu_function_wall_s"]}))


if __name__ == "__main__":
    main()
