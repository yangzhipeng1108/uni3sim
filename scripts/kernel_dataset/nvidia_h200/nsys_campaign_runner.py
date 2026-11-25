"""Remote NSYS campaign for the H200 kernel dataset (A800 canonical protocol).

Runs, for every planned case and every window, exactly the A800 canonical
timing command shape::

    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --export=sqlite --force-overwrite=true --output <stem> <runner ...>

with the H200 owner-copy runner.  Changes versus the A800
``step_run_nsys_timing`` loop, all hardware/host coupling:

* CUDA_VISIBLE_DEVICES pins the owned H200 GPU; runners take an exclusive
  GPU lock before their first CUDA call, so ``--concurrency`` processes may
  import Python/vLLM and run NSYS start-up/export in parallel while exactly
  one process uses the GPU.
* a continuous ``nvidia-smi -lms 50`` clock log covers the whole campaign;
  every window also records SM clocks right before/after its NVTX range.
* each SQLite is reduced on the host to a compact launch JSON (A800
  ``load_nsys_timing_sqlite`` for timing identity plus registers/shared
  memory launch facts) and then deleted; the ``.nsys-rep`` report is kept.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_a800 import (  # noqa: E402
    _NSYS_PROFILE_ARGS,
    load_nsys_timing_sqlite,
)


def _launch_facts(sqlite_path, launches):
    connection = sqlite3.connect(str(sqlite_path))
    try:
        facts = []
        for launch in launches:
            rows = connection.execute(
                """
                SELECT registersPerThread, staticSharedMemory,
                       dynamicSharedMemory, localMemoryPerThread,
                       sharedMemoryExecuted, streamId
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE start = ? AND gridId = ?
                """,
                (launch["start"], launch["grid_id"]),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(
                    f"ambiguous kernel launch-fact row count {len(rows)}")
            row = rows[0]
            facts.append({
                "name": launch["name"],
                "grid": list(launch["grid"]),
                "block": list(launch["block"]),
                "duration_ns": launch["duration_ns"],
                "start_ns": launch["start"],
                "device": launch["device"],
                "registers_per_thread": int(row[0]),
                "static_shared_memory_bytes": int(row[1]),
                "dynamic_shared_memory_bytes": int(row[2]),
                "local_memory_per_thread_bytes": int(row[3]),
                "shared_memory_executed_bytes": int(row[4]),
                "stream_id": int(row[5]),
            })
        return facts
    finally:
        connection.close()


def _run_window(plan, window, args, output_root, lock_path, print_lock):
    case_id = plan["case_id"]
    case_dir = output_root / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    stem = case_dir / f"window_{window:02d}"
    result_path = Path(str(stem) + ".launch.json")
    if result_path.is_file():
        return case_id, window, "skipped"
    telemetry_path = Path(str(stem) + ".telemetry.json")
    report_path = Path(str(stem) + ".nsys-rep")
    sqlite_path = Path(str(stem) + ".sqlite")
    for stale in (telemetry_path, report_path, sqlite_path):
        if stale.exists():
            stale.unlink()
    profile_args = list(_NSYS_PROFILE_ARGS)
    if args.gpu_metrics_devices is not None:
        profile_args.append(f"--gpu-metrics-devices={args.gpu_metrics_devices}")
    command = [
        args.nsys, *profile_args, "--output", str(stem),
        args.python, str(ROOT / plan["runner_relpath"]),
        *plan["runner_args"],
        "--h200-gpu-lock", str(lock_path),
        "--h200-telemetry-output", str(telemetry_path),
        "--h200-physical-gpu", str(args.physical_gpu),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    started = time.time()
    completed = subprocess.run(
        command, text=True, capture_output=True, env=environment)
    elapsed = time.time() - started
    if completed.returncode != 0 or not sqlite_path.is_file():
        failure = {
            "case_id": case_id, "window": window,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-3000:],
            "stderr_tail": completed.stderr[-6000:],
        }
        Path(str(stem) + ".failure.json").write_text(
            json.dumps(failure, indent=1), encoding="utf-8")
        with print_lock:
            print(json.dumps({"case_id": case_id, "window": window,
                              "status": "failed",
                              "returncode": completed.returncode}),
                  flush=True)
        return case_id, window, "failed"
    launches = load_nsys_timing_sqlite(str(sqlite_path), plan["marker"])
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "family": plan["family"],
        "window": window,
        "marker": plan["marker"],
        "runner_relpath": plan["runner_relpath"],
        "runner_args": plan["runner_args"],
        "nsys_arguments": profile_args,
        "launches": _launch_facts(sqlite_path, launches),
        "telemetry": telemetry,
        "wall_seconds": elapsed,
    }
    temporary = Path(str(result_path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(temporary, result_path)
    if not args.retain_sqlite:
        sqlite_path.unlink()
    with print_lock:
        print(json.dumps({
            "case_id": case_id, "window": window, "status": "measured",
            "total_ns": sum(item["duration_ns"]
                            for item in payload["launches"]),
            "kernels": len(payload["launches"]),
            "clock_before": telemetry["clock_before_nvtx"]["sm_clock_mhz"],
            "clock_after": telemetry["clock_after_nvtx"]["sm_clock_mhz"],
            "seconds": round(elapsed, 1)}), flush=True)
    return case_id, window, "measured"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--window-ids", type=str, default=None,
                        help="comma list overriding 1..windows (retries)")
    parser.add_argument("--case-ids", type=str, default=None,
                        help="optional file with one case id per line")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--nsys", required=True)
    parser.add_argument("--gpu-metrics-devices", choices=("cuda-visible",),
                        default=None,
                        help="diagnostic capture of GPU Metrics in the same report")
    parser.add_argument("--retain-sqlite", action="store_true",
                        help="retain diagnostic metric tables beside the report")
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True,
                        help="one lock file shared by every campaign on the GPU")
    args = parser.parse_args()
    plans = json.loads(args.plans.read_text(encoding="utf-8"))["plans"]
    if args.case_ids is not None:
        wanted = [line.strip() for line in Path(args.case_ids).read_text(
            encoding="utf-8").splitlines() if line.strip()]
        index = {plan["case_id"]: plan for plan in plans}
        plans = [index[case_id] for case_id in wanted]
    if args.window_ids is None:
        windows = list(range(1, args.windows + 1))
    else:
        windows = [int(value) for value in args.window_ids.split(",")]
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.gpu_lock
    lock_path.touch()
    clock_log = output_root / f"clock_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    logger = subprocess.Popen(
        ["nvidia-smi", "-i", str(args.physical_gpu),
         "--query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,"
         "power.draw",
         "--format=csv,noheader,nounits", "-lms", "50"],
        stdout=clock_log.open("w"), stderr=subprocess.DEVNULL)
    print_lock = threading.Lock()
    jobs = [(plan, window) for window in windows for plan in plans]
    counts = {"measured": 0, "skipped": 0, "failed": 0}
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(_run_window, plan, window, args, output_root,
                            lock_path, print_lock)
                for plan, window in jobs]
            for future in futures:
                _, _, status = future.result()
                counts[status] += 1
    finally:
        logger.terminate()
        logger.wait()
    print(json.dumps({"done": True, **counts}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
