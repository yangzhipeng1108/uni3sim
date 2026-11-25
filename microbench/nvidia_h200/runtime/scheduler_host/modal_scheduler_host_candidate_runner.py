"""Measure one Modal H200 Host as a candidate, without relabeling Xeon 8558.

First run prepare_modal_scheduler_bundle.py locally. The bundle contains a
clean, tagged vLLM source checkout and committed Midlayer code, but no model
weights or dataset. The independent asynchronous Scheduler CPU samples describe
this Modal H200 container. The earlier launch comparison is a separate capture.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

import modal


SOURCE = Path(__file__).resolve()
BUNDLE = Path("/tmp/midlayer-h200-modal-scheduler-bundle.tar.gz")
REMOTE_BUNDLE = "/tmp/midlayer-h200-modal-scheduler-bundle.tar.gz"
IMAGE = (modal.Image.from_registry(
    "vllm/vllm-openai:v0.30.0", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential")
    .add_local_file(BUNDLE, remote_path=REMOTE_BUNDLE))
APP = modal.App("midlayer-h200-async-scheduler-candidate")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(argv: list[str], *, cwd: Path, timeout: int,
         env: dict[str, str] | None = None) -> dict:
    started = time.monotonic()
    try:
        process = subprocess.run(
            argv, cwd=cwd, env=env, check=False, capture_output=True,
            text=True, timeout=timeout)
        return {"command": argv, "exit_code": process.returncode,
                "stdout": process.stdout[-4000:],
                "stderr": process.stderr[-4000:],
                "wall_s": time.monotonic() - started}
    except subprocess.TimeoutExpired as error:
        return {"command": argv, "exit_code": None,
                "error": f"timeout after {timeout}s: {error}",
                "wall_s": time.monotonic() - started}


@APP.function(gpu="H200", image=IMAGE, timeout=480, retries=0,
              max_containers=1, single_use_containers=True)
def capture(expected_bundle_sha256: str) -> dict:
    started = time.monotonic()
    result = {"status": "started", "measurement_boundary":
              "modal_h200_container_independent_vllm030_async_scheduler_cpu"}
    try:
        bundle = Path(REMOTE_BUNDLE)
        if _sha256(bundle) != expected_bundle_sha256:
            result["status"] = "bundle_hash_failed"
            return result
        with tempfile.TemporaryDirectory(prefix="midlayer-h200-scheduler-",
                                         dir="/tmp") as temporary:
            task = Path(temporary)
            with tarfile.open(bundle, mode="r:gz") as reader:
                reader.extractall(task, filter="data")
            manifest = json.loads((task / "bundle_manifest.json").read_text())
            result["bundle_manifest"] = manifest
            source = task / "vllm_source"
            repo = task / "midlayer_repo"
            checks = (
                (["git", "rev-parse", "HEAD"], manifest["source_commit"]),
                (["git", "describe", "--tags", "--exact-match", "HEAD"],
                 "v0.30.0"),
                (["git", "status", "--porcelain", "--untracked-files=all"],
                 ""),
            )
            result["source_checks"] = []
            for command, expected in checks:
                row = _run(command, cwd=source, timeout=20)
                result["source_checks"].append(row)
                if row["exit_code"] != 0 or row["stdout"].strip() != expected:
                    result["status"] = "source_identity_failed"
                    return result
            for relative, expected in manifest["source_hashes"].items():
                if _sha256(repo / relative) != expected:
                    result["status"] = "repo_source_hash_failed"
                    return result
            (repo / "build").mkdir()
            result["lru_build"] = _run([
                "cc", "-O3", "-shared", "-fPIC", "-o",
                str(repo / "build/lru_sim.so"),
                str(repo / "src/kernel/csrc/lru_sim.c")],
                cwd=repo, timeout=60)
            if result["lru_build"]["exit_code"] != 0:
                result["status"] = "lru_build_failed"
                return result
            result["lru_binary_sha256"] = _sha256(repo / "build/lru_sim.so")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo)
            result["identity"] = _run([
                "/usr/bin/python3", "-c",
                "import json,pathlib,platform,torch,vllm; "
                "print(json.dumps({'cpuinfo':pathlib.Path('/proc/cpuinfo').read_text().split('\\n\\n',1)[0],"
                "'kernel':platform.release(),'torch':torch.__version__,"
                "'cuda':torch.version.cuda,'vllm':vllm.__version__}))"],
                cwd=task, timeout=45, env=env)
            result["gpu"] = _run([
                "nvidia-smi", "--query-gpu=name,driver_version",
                "--format=csv,noheader"], cwd=task, timeout=20)
            if (result["identity"]["exit_code"] != 0
                    or result["gpu"]["exit_code"] != 0
                    or not result["gpu"]["stdout"].startswith("NVIDIA H200,")):
                result["status"] = "environment_failed"
                return result
            result["scheduler_runs"] = []
            for prompt_tokens in (32, 256):
                for max_requests in (1, 8, 32):
                    output = task / f"scheduler_{prompt_tokens}_{max_requests}.json"
                    row = _run([
                        "/usr/bin/python3", str(repo / "scripts/measure_vllm_scheduler_service.py"),
                        "--hardware-id", "nvidia_h200", "--vllm-version", "0.30.0",
                        "--vllm030-python", "/usr/bin/python3", "--vllm030-source",
                        str(source), "--model-config", str(repo / "configs/model/qwen3-4b.json"),
                        "--output", str(output), "--sample-count", "64",
                        "--max-batched-tokens", "8192", "--max-requests", str(max_requests),
                        "--num-kv-blocks", "8192", "--max-model-len", "4096",
                        "--prompt-tokens", str(prompt_tokens), "--output-tokens", "8",
                        "--async-scheduling"], cwd=repo, timeout=55, env=env)
                    row["prompt_tokens"] = prompt_tokens
                    row["max_requests"] = max_requests
                    if row["exit_code"] == 0:
                        row["raw"] = json.loads(output.read_text())
                    result["scheduler_runs"].append(row)
                    if row["exit_code"] != 0:
                        result["status"] = "scheduler_failed"
                        return result
            result["status"] = "complete"
            return result
    finally:
        result["gpu_function_wall_s"] = time.monotonic() - started


@APP.local_entrypoint()
def main() -> None:
    local_root = SOURCE.parents[4]
    target = local_root / (
        "profiler_output/microbench/nvidia_h200/"
        "modal_async_scheduler_candidate_20261029/attempt_4_capture")
    if target.exists():
        raise FileExistsError(target)
    if not BUNDLE.is_file():
        raise FileNotFoundError(
            "run prepare_modal_scheduler_bundle.py before Modal capture")
    started = time.monotonic()
    result = capture.remote(_sha256(BUNDLE))
    result["client_wall_s"] = time.monotonic() - started
    result["bundle_sha256"] = _sha256(BUNDLE)
    result["runner_sha256"] = _sha256(SOURCE)
    target.mkdir(parents=True)
    (target / "raw.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(target / "raw.json"),
                      "status": result["status"],
                      "gpu_function_wall_s": result["gpu_function_wall_s"]}))


if __name__ == "__main__":
    main()
