"""Capture additional exact-geometry mixed-stream capacities."""
import hashlib
import json
from pathlib import Path
import shutil
import importlib.util
import subprocess
import sys
import time
import modal

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "vec16_mixed_stream_capacity_extended.cu"
ANALYSIS = HERE / "vec16_mixed_stream_capacity_extended_analysis.py"
PLAN = HERE.parents[1] / "references/vec16_mixed_stream_capacity_extended_plan_sm90.json" if modal.is_local() else Path("/tmp/vec16-mixed-stream-extended/plan.json")
TASK = Path("/tmp/vec16-mixed-stream-extended")
_ANALYSIS_PATH = ANALYSIS if modal.is_local() else TASK / ANALYSIS.name
_SPEC = importlib.util.spec_from_file_location("vec16_mixed_stream_capacity_extended_analysis", _ANALYSIS_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
analyze = _MODULE.analyze
IMAGE = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
         .add_local_file(SOURCE, remote_path=str(TASK / SOURCE.name))
         .add_local_file(ANALYSIS, remote_path=str(TASK / ANALYSIS.name))
         .add_local_file(PLAN, remote_path=str(TASK / "plan.json")))
APP = modal.App("midlayer-vec16-mixed-stream-extended-20261109")


def run(command, limit=180):
    result = subprocess.run(command, cwd=TASK, capture_output=True, text=True, timeout=limit)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


@APP.function(gpu="H200", image=IMAGE, timeout=180, retries=0, max_containers=1, single_use_containers=True)
def capture(plan):
    start = time.monotonic()
    out = {"status": "started", "plan": plan}
    try:
        for name, command in (
            ("gpu", ["nvidia-smi", "--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version", "--format=csv,noheader"]),
            ("build", ["nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-Xptxas=-v", SOURCE.name, "-o", "probe"]),
            ("sass", ["cuobjdump", "--dump-sass", "probe"]),
        ):
            out[name] = run(command)
            if out[name]["exit_code"]:
                out["status"] = name + "_failed"
                return out
        out["runs"] = []
        for process in range(2):
            measurement = run([str(TASK / "probe")])
            out["runs"].append({"process": process, **measurement})
            if measurement["exit_code"]:
                out["status"] = "measurement_failed"
                return out
        out["status"] = "complete"
        return out
    finally:
        out["gpu_function_wall_s"] = time.monotonic() - start
        shutil.rmtree(TASK)
        out["remote_task_directory_removed"] = not TASK.exists()


@APP.local_entrypoint()
def main(output_dir: str, budget_key: str, budget_tool: str):
    root = HERE.parents[3]
    target = Path(output_dir).resolve()
    tool = Path(budget_tool)
    if target != (root / "profiler_output/microbench/nvidia_h200/vec16_mixed_stream_capacity_extended_20261109/attempt_1").resolve() or target.exists():
        raise ValueError("new registered attempt_1 required")
    if not tool.is_file() or not budget_key:
        raise ValueError("atomic shared reservation required")
    target.mkdir(parents=True)
    plan = json.loads(PLAN.read_text())
    meta = {"app_id": APP.app_id, "budget_key": budget_key,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest()}
    subprocess.run([sys.executable, str(tool), "bind", "--key", budget_key, "--app-id", APP.app_id], check=True)
    (target / "launch.json").write_text(json.dumps(meta, indent=2) + "\n")
    start = time.monotonic()
    try:
        out = capture.remote(plan)
        out.update(meta)
        out["client_wall_s"] = time.monotonic() - start
        (target / "raw.json").write_text(json.dumps(out, indent=2) + "\n")
        if out["status"] == "complete":
            out["analysis"] = analyze(out)
        elif out["status"] in ("gpu_failed", "build_failed", "sass_failed", "measurement_failed"):
            out["analysis"] = None
        else:
            raise ValueError("unknown extended mixed-stream capture status")
        (target / "raw.json").write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({"status": out["status"], "app_id": APP.app_id,
                          "analysis": out["analysis"]}))
    except Exception as exc:
        (target / "client_failure.json").write_text(json.dumps(dict(
            **meta, error_type=type(exc).__name__, error=str(exc)), indent=2) + "\n")
        raise
