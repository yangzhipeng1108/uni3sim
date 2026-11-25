"""Run the independent BM128 operand service probe on one Modal H200."""

import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import time

import modal


SOURCE = Path(__file__).with_name("bm128_operand_service.cu")
IMAGE = modal.Image.from_registry(
    "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
)
if SOURCE.is_file():
    IMAGE = IMAGE.add_local_file(
        SOURCE, remote_path="/tmp/bm128_operand_service.cu")
APP = modal.App("midlayer-h200-bm128-operand-20261029")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@APP.function(gpu="H200", image=IMAGE, timeout=780, retries=0,
              max_containers=1, single_use_containers=True)
def capture() -> dict:
    import shutil
    from datetime import datetime, timezone

    task_dir = Path("/tmp/midlayer-bm128-operand-20261029")
    task_dir.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    monotonic_start = time.monotonic()
    try:
        gpu_before = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,driver_version",
             "--format=csv,noheader"], check=True, capture_output=True,
            text=True, timeout=30).stdout.strip()
        nvcc = subprocess.run(["nvcc", "--version"], check=True,
                              capture_output=True, text=True, timeout=30).stdout
        binary = task_dir / "bm128_operand_service"
        build = subprocess.run(
            ["nvcc", "-O2", "-arch=sm_90", "-std=c++17",
             "/tmp/bm128_operand_service.cu", "-o", str(binary)],
            check=True, capture_output=True, text=True, timeout=180)
        outputs = []
        for case in ("l2", "stream"):
            result = subprocess.run([str(binary), case], check=True,
                                    capture_output=True, text=True,
                                    timeout=180)
            outputs.append({"case": case, "csv": result.stdout,
                            "validation": result.stderr.strip()})
        gpu_after = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,driver_version",
             "--format=csv,noheader"], check=True, capture_output=True,
            text=True, timeout=30).stdout.strip()
        return {"started_utc": started,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_wall_s": time.monotonic() - monotonic_start,
                "gpu_before": gpu_before, "gpu_after": gpu_after,
                "nvcc": nvcc, "build_stderr": build.stderr,
                "binary_sha256": _sha256(binary.read_bytes()),
                "outputs": outputs}
    finally:
        shutil.rmtree(task_dir)


def _validated_rows(result: dict) -> list[dict]:
    expected = {"l2": (32, 16 * 2**20, 32 * 2**20),
                "stream": (128, 16 * 2**20, 128 * 2**20)}
    if set(result) != {"started_utc", "ended_utc", "gpu_wall_s",
                       "gpu_before", "gpu_after", "nvcc", "build_stderr",
                       "binary_sha256", "outputs"}:
        raise ValueError("remote capture returned an unexpected contract")
    if len(result["outputs"]) != 2:
        raise ValueError("remote capture omitted a footprint")
    rows = []
    for output in result["outputs"]:
        case = output["case"]
        if case not in expected:
            raise ValueError("unknown footprint")
        if not output["validation"].startswith(f"validated {case}"):
            raise ValueError("probe output was not validated")
        case_rows = list(csv.DictReader(io.StringIO(output["csv"])))
        if len(case_rows) != 7:
            raise ValueError("probe must return seven samples per footprint")
        tiles, activation_bytes, weight_bytes = expected[case]
        for index, row in enumerate(case_rows):
            if (row["case"] != ("l2_48mib_unique" if case == "l2"
                                else "stream_144mib_unique")
                    or int(row["sample"]) != index
                    or int(row["weight_tiles"]) != tiles
                    or int(row["activation_bytes"]) != activation_bytes
                    or int(row["weight_bytes"]) != weight_bytes
                    or int(row["logical_bytes"]) != 4096 * 2048 * 384 * 2
                    or float(row["milliseconds"]) <= 0
                    or float(row["logical_gb_s"]) <= 0):
                raise ValueError("probe sample violates its fixed geometry")
            rows.append(row)
    return rows


@APP.local_entrypoint()
def main(output_dir: str):
    root = Path(__file__).resolve().parents[4]
    output_root = root / "profiler_output/microbench/nvidia_h200"
    target = Path(output_dir).resolve()
    if target.exists() or output_root.resolve() not in target.parents:
        raise ValueError("new output directory must be under H200 profiler root")
    source_bytes = SOURCE.read_bytes()
    runner_bytes = Path(__file__).read_bytes()
    result = capture.remote()
    rows = _validated_rows(result)
    target.mkdir(parents=True)
    raw = target / "raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for case in ("l2_48mib_unique", "stream_144mib_unique"):
        values = [float(row["logical_gb_s"]) for row in rows
                  if row["case"] == case]
        summary[case] = {"median_logical_gb_s": statistics.median(values),
                         "minimum_logical_gb_s": min(values),
                         "maximum_logical_gb_s": max(values)}
    manifest = {
        "schema": "h200_bm128_operand_service_candidate_v1",
        "baseline_commit": "ef73f6a476b8184d1cf5faad3c5c5a86dc9d8049",
        "hardware_id": "nvidia_h200", "gpu_count": 1,
        "task_gpu_hour_cap": 1.5,
        "capture_gpu_hour_upper_bound": 0.25,
        "actual_gpu_hours": result["gpu_wall_s"] / 3600,
        "source_path": str(SOURCE.relative_to(root)),
        "source_sha256": _sha256(source_bytes),
        "runner_path": str(Path(__file__).relative_to(root)),
        "runner_sha256": _sha256(runner_bytes),
        "raw_sha256": _sha256(raw.read_bytes()),
        "production_or_evaluation_duration_consumed": False,
        "measurement_boundary": "independent_operand_read_only_no_mma_or_moe",
        "result": {key: value for key, value in result.items()
                   if key != "outputs"},
        "validations": [output["validation"] for output in result["outputs"]],
        "summary": summary,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"output_dir": str(target), "summary": summary,
                      "actual_gpu_hours": manifest["actual_gpu_hours"]},
                     indent=2, sort_keys=True))
