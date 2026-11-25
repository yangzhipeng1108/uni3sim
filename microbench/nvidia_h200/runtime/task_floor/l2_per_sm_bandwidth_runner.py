#!/usr/bin/env python3
"""Measure H200 per-SM L2 read bandwidth via grid sweep.

Triton streaming read kernel: each CTA reads a disjoint L2-resident
buffer with vectorized loads.  Per-SM bandwidth is derived from
throughput at each grid size.
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
def _streaming_read_kernel(data_ptr, out_ptr, n_elements: tl.constexpr,
                           BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Each CTA reads its own slice
    start = pid * n_elements
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for offset in range(0, n_elements, BLOCK):
        ptrs = data_ptr + start + offset + tl.arange(0, BLOCK)
        vals = tl.load(ptrs, cache_modifier=".cg").to(tl.float32)
        acc += vals
    tl.store(out_ptr + pid, tl.sum(acc))


def _measure_bandwidth(grid: int, elements_per_cta: int, device: torch.device,
                       trials: int = 20) -> float:
    data = torch.randn(grid * elements_per_cta, dtype=torch.bfloat16, device=device)
    out = torch.zeros(grid, dtype=torch.float32, device=device)
    BLOCK = 4096

    # Warmup
    _streaming_read_kernel[(grid,)](data, out, n_elements=elements_per_cta, BLOCK=BLOCK, num_warps=16)
    torch.cuda.synchronize(device)

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _streaming_read_kernel[(grid,)](data, out, n_elements=elements_per_cta, BLOCK=BLOCK, num_warps=16)
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ns = start.elapsed_time(end) * 1e6
        total_bytes = grid * elements_per_cta * 2  # bf16
        bw_gbps = total_bytes / elapsed_ns  # bytes/ns = GB/s
        times.append(bw_gbps)

    del data, out
    torch.cuda.empty_cache()
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=20)
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
        elements_per_cta = 131072  # 256KB per CTA  # 32KB per CTA
        for grid in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                bw = _measure_bandwidth(grid, elements_per_cta, device, args.trials)
                samples.append({
                    "grid": grid,
                    "bandwidth_gbps": bw,
                    "elements_per_cta": elements_per_cta,
                    "trials": args.trials,
                })
                print(f"  grid={grid:3d}: {bw:8.1f} GB/s ({bw/grid:6.1f} GB/s/SM)")
            except Exception as error:
                samples.append({
                    "grid": grid,
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
        "schema": "nvidia_h200_l2_per_sm_bandwidth_raw_v1",
        "hardware_id": "nvidia_h200",
        "physical_gpu_index": int(physical_index[0]),
        "busy_clock_samples": len(busy),
        "busy_clock_mhz_min": min(busy) if busy else None,
        "busy_clock_mhz_max": max(busy) if busy else None,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timer_domain": "cuda_event_gpu_kernel",
        "mechanism": "l2_streaming_read_grid_sweep_triton",
        "target_operator_used": False,
        "production_or_evaluation_duration_consumed": False,
        "samples": samples,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1, sort_keys=True)
    print(json.dumps({"samples": len(samples)}, indent=1))


if __name__ == "__main__":
    main()
