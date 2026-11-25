#!/usr/bin/env python3
"""Measure H200 L2 partition service cycles via multi-SM GPU pointer chase.

Independent mechanism: GPU pointer-chase through L2-resident array from N
co-resident CTAs (one per SM).  Each CTA's threads chase pointers through
the same L2 lines, creating partition contention.  Per-request latency
increases with N following the M/D/1 queue formula.

Method: Triton kernel where each CTA does steps dependent 64-bit loads.
The chase is GPU-side (no CPU sync).  CUDA events time the full chase;
dividing by steps gives per-request latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _pointer_chase_kernel(chase_ptr, out_ptr, steps: tl.constexpr,
                          BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Each CTA starts at a different offset in the chase array
    idx = (pid * 9973 % (BLOCK * 64)).to(tl.int64)
    acc = tl.zeros((), dtype=tl.int64)
    for _ in range(steps):
        # Dependent load: next index comes from current load
        val = tl.load(chase_ptr + idx)
        idx = (val % (BLOCK * 64)).to(tl.int64)
        acc += val
    tl.store(out_ptr + pid, acc)


def _run_chase(chase_array, active_sms, steps, device, trials=30):
    n = chase_array.shape[0]
    out = torch.zeros(active_sms, dtype=torch.int64, device=device)
    BLOCK = 1024
    grid = (active_sms,)

    # Warmup
    _pointer_chase_kernel[grid](chase_array, out, steps=steps, BLOCK=BLOCK)
    torch.cuda.synchronize(device)

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _pointer_chase_kernel[grid](chase_array, out, steps=steps, BLOCK=BLOCK)
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ns = start.elapsed_time(end) * 1e6
        per_request_ns = elapsed_ns / steps
        times.append(per_request_ns)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    physical = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    physical_index = [line.split(",")[0].strip() for line in physical.splitlines()
                      if line.split(",")[1].strip().endswith(uuid)]
    if len(physical_index) != 1:
        raise RuntimeError("cannot map torch device to nvidia-smi")

    clock_path = args.output + ".clock.csv"
    clock_file = open(clock_path, "w", encoding="utf-8")
    sampler = subprocess.Popen(
        ["nvidia-smi", "-i", physical_index[0],
         "--query-gpu=clocks.sm,utilization.gpu",
         "--format=csv,noheader,nounits", "-lms", "50"], stdout=clock_file)

    samples = []
    try:
        # Build L2-resident chase array (~10MB of int64)
        n_elements = 1_250_000  # 10MB
        chase_array = torch.zeros(n_elements, dtype=torch.int64, device=device)
        # Build a cycle that visits different L2 lines
        step_size = 4096 // 8  # 4KB apart (different 128B cache lines)
        indices = [(i * step_size) % n_elements for i in range(n_elements // step_size)]
        for i in range(len(indices) - 1):
            chase_array[indices[i]] = indices[i + 1]
        chase_array[indices[-1]] = indices[0]
        torch.cuda.synchronize(device)

        for active_sms in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                med_ns = _run_chase(
                    chase_array, active_sms, args.steps, device, args.trials)
                samples.append({
                    "active_sms": active_sms,
                    "median_per_request_ns": med_ns,
                    "steps": args.steps,
                    "trials": args.trials,
                })
                print(f"  active_sms={active_sms:3d}: {med_ns:.1f} ns/request")
            except Exception as error:
                samples.append({
                    "active_sms": active_sms,
                    "error": str(error)[:400],
                })
            torch.cuda.synchronize(device)
    finally:
        time.sleep(0.2)
        sampler.terminate()
        sampler.wait()
        clock_file.close()

    with open(clock_path, encoding="utf-8") as handle:
        busy = [int(line.split(",")[0]) for line in handle
                if line.strip() and int(line.split(",")[1]) > 0]

    document = {
        "schema": "nvidia_h200_l2_partition_service_raw_v1",
        "hardware_id": "nvidia_h200",
        "physical_gpu_index": int(physical_index[0]),
        "busy_clock_samples": len(busy),
        "busy_clock_mhz_min": min(busy) if busy else None,
        "busy_clock_mhz_max": max(busy) if busy else None,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timer_domain": "cuda_event_gpu_kernel",
        "mechanism": "l2_pointer_chase_multi_sm_contention_triton",
        "target_operator_used": False,
        "production_or_evaluation_duration_consumed": False,
        "samples": samples,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1, sort_keys=True)
    print(json.dumps({"samples": len(samples)}, indent=1))


if __name__ == "__main__":
    main()
