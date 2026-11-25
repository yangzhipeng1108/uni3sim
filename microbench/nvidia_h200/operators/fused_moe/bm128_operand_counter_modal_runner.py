"""Capture L2/DRAM counters for the independent BM128 operand probe."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import modal


SOURCE = Path(__file__).with_name("bm128_operand_service.cu")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
).apt_install("cuda-nsight-compute-12-6")
if SOURCE.is_file():
    IMAGE = IMAGE.add_local_file(
        SOURCE, remote_path="/tmp/bm128_operand_service.cu")
APP = modal.App("midlayer-h200-bm128-operand-counters-20261029")


@APP.function(gpu="H200", image=IMAGE, timeout=600, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    import shutil
    from datetime import datetime, timezone

    task_dir = Path("/tmp/midlayer-bm128-operand-counters-20261029")
    task_dir.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,driver_version",
             "--format=csv,noheader"], check=True, capture_output=True,
            text=True, timeout=30).stdout.strip()
        ncu = subprocess.run(["ncu", "--version"], check=True,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        binary = task_dir / "bm128_operand_service"
        subprocess.run(
            ["nvcc", "-O2", "-arch=sm_90", "-std=c++17",
             "/tmp/bm128_operand_service.cu", "-o", str(binary)],
            check=True, capture_output=True, text=True, timeout=180)
        records = []
        for case in ("l2", "stream"):
            command = [
                "ncu", "--csv", "--page", "raw", "--cache-control", "none",
                "--clock-control", "none",
                "--kernel-name", "regex:operand_read", "--launch-skip", "30",
                "--launch-count", "1", "--metrics",
                "lts__t_sectors_op_read.sum,dram__bytes_read.sum",
                str(binary), case,
            ]
            completed = subprocess.run(command, capture_output=True,
                                       text=True, timeout=240)
            records.append({"case": case, "command": command,
                            "exit_code": completed.returncode,
                            "stdout": completed.stdout,
                            "stderr": completed.stderr})
            if completed.returncode != 0:
                break
        return {"started_utc": started,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_wall_s": time.monotonic() - start,
                "gpu_identity": gpu, "ncu_version": ncu,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "records": records}
    finally:
        shutil.rmtree(task_dir)


@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    parent = root / "profiler_output/microbench/nvidia_h200/bm128_operand_service_20261029"
    target = Path(output_dir).resolve()
    if target.parent != parent.resolve() or target.exists():
        raise ValueError("counter output must be a new child of the registered probe directory")
    result = capture.remote()
    target.mkdir(parents=True)
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["baseline_commit"] = "ef73f6a476b8184d1cf5faad3c5c5a86dc9d8049"
    result["measurement_boundary"] = "independent_operand_read_only_no_mma_or_moe"
    (target / "counter_attempt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(target), "gpu_wall_s": result["gpu_wall_s"],
                      "exit_codes": [row["exit_code"] for row in result["records"]]},
                     indent=2, sort_keys=True))
