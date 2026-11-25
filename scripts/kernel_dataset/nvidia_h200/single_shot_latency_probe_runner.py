"""Single-shot occupancy-latency probe for H200.

Measures actual kernel duration (not loop per-iteration time) for
a single-shot elementwise ADD kernel at varying warp occupancy.
This gives the true latency exposure for production-style single-shot launches.
"""
import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Single-shot elementwise ADD: C = A + B
// Each thread: 2 LDG.128 (independent) + FADD + 1 STG.128
// Grid/block controlled externally to set occupancy

__global__ void __launch_bounds__(1024) single_shot_add(
    const float4* __restrict__ bufA, const float4* __restrict__ bufB,
    float4* __restrict__ out, int n_vec4)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_vec4) {
        float4 a = bufA[tid];
        float4 b = bufB[tid];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        out[tid] = c;
    }
}

// Single-shot with NO data load (just to measure launch floor)
__global__ void __launch_bounds__(1024) single_shot_empty(
    float4* __restrict__ out, int n_vec4)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_vec4) {
        float4 v = make_float4(tid, tid, tid, tid);
        out[tid] = v;
    }
}

// Store-only single-shot
__global__ void __launch_bounds__(1024) single_shot_store(
    float4* __restrict__ out, int n_vec4)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_vec4) {
        float4 v = make_float4(tid, tid, tid, tid);
        out[tid] = v;
    }
}

// Load-only single-shot
__global__ void __launch_bounds__(1024) single_shot_load(
    const float4* __restrict__ bufA, float4* __restrict__ out, int n_vec4)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n_vec4) {
        float4 a = bufA[tid];
        float acc = a.x + a.y + a.z + a.w;
        if (acc == -1.0f) out[tid] = a;  // never true, prevents optimization
    }
}

extern "C" int run_single_shot(
    int nblocks, int nthreads, int n_vec4,
    float* out_kernel_ns, float* out_empty_ns, float* out_store_ns, float* out_load_ns)
{
    size_t buf_bytes = (size_t)n_vec4 * 16;
    if (buf_bytes < 1024) buf_bytes = 1024;
    if (buf_bytes > 64*1024*1024) buf_bytes = 64*1024*1024;

    float4 *bufA, *bufB, *out;
    cudaMalloc(&bufA, buf_bytes);
    cudaMalloc(&bufB, buf_bytes);
    cudaMalloc(&out, buf_bytes);
    cudaMemset(bufA, 1, buf_bytes);
    cudaMemset(bufB, 2, buf_bytes);

    dim3 grid(nblocks), block(nthreads);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int w = 0; w < 5; w++) {
        single_shot_add<<<grid, block>>>(bufA, bufB, out, n_vec4);
        single_shot_empty<<<grid, block>>>(out, n_vec4);
        single_shot_store<<<grid, block>>>(out, n_vec4);
        single_shot_load<<<grid, block>>>(bufA, out, n_vec4);
    }
    cudaDeviceSynchronize();

    // Measure ADD
    float total_ms = 0;
    int reps = 50;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(start);
        single_shot_add<<<grid, block>>>(bufA, bufB, out, n_vec4);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    *out_kernel_ns = total_ms * 1e6f / reps;

    // Measure empty
    total_ms = 0;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(start);
        single_shot_empty<<<grid, block>>>(out, n_vec4);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    *out_empty_ns = total_ms * 1e6f / reps;

    // Measure store-only
    total_ms = 0;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(start);
        single_shot_store<<<grid, block>>>(out, n_vec4);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    *out_store_ns = total_ms * 1e6f / reps;

    // Measure load-only
    total_ms = 0;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(start);
        single_shot_load<<<grid, block>>>(bufA, out, n_vec4);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    *out_load_ns = total_ms * 1e6f / reps;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(out);
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sm-count", type=int, default=132)
    parser.add_argument("--smsp-per-sm", type=int, default=4)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "single_shot.cu"
        lib = Path(tmpdir) / "single_shot.so"
        src.write_text(CUDA_SRC)

        # Compile with nvcc
        cmd = [
            "nvcc", "-shared", "-Xcompiler", "-fPIC",
            "-arch=sm_90a",
            "-o", str(lib), str(src),
            "-lcudart", "-lcuda"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"nvcc failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)

        # Load library
        cuda_lib = ctypes.CDLL(str(lib))
        cuda_lib.run_single_shot.restype = ctypes.c_int
        cuda_lib.run_single_shot.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
        ]

        # Test configurations: vary grid and block to control occupancy
        # For H200: 132 SMs, 4 SMSPs per SM, max 32 warps per SM (8 per SMSP)
        # Occupancy = (grid * block/32) / (132 * 32) = warps_per_smsp / 8

        configs = []
        # Match production elementwise shapes: grid = ceil(elements/1024), block=128
        # elements = tok * hidden, grid = ceil(elements / (128 * 8)) for vec8
        # Actually for vectorized_elementwise_kernel<8>: each thread processes 8 elements
        # So grid = ceil(elements / (block * 8))

        # T=1, H=2048: elements=2048, grid=ceil(2048/(128*8))=2, block=128
        # T=8, H=2048: elements=16384, grid=16
        # T=16, H=2048: elements=32768, grid=32
        # T=64, H=2048: elements=131072, grid=128
        # T=256, H=2048: elements=524288, grid=512
        # T=1024, H=2048: elements=2097152, grid=2048

        # For single-shot probe, use n_vec4 = elements / 8 (float4 = 8 bf16)
        # But we want to match the grid/block of production
        # n_vec4 = grid * block (each thread does one vec4 = 8 elements)

        test_cases = [
            # (label, grid, block, n_vec4)
            ("T1_H2048", 2, 128, 2*128),
            ("T2_H2048", 4, 128, 4*128),
            ("T4_H2048", 8, 128, 8*128),
            ("T8_H2048", 16, 128, 16*128),
            ("T16_H2048", 32, 128, 32*128),
            ("T32_H2048", 64, 128, 64*128),
            ("T64_H2048", 128, 128, 128*128),
            ("T128_H2048", 256, 128, 256*128),
            ("T256_H2048", 512, 128, 512*128),
            ("T512_H2048", 1024, 128, 1024*128),
            ("T1024_H2048", 2048, 128, 2048*128),
        ]

        results = []
        for label, grid, block, n_vec4 in test_cases:
            kernel_ns = ctypes.c_float(0)
            empty_ns = ctypes.c_float(0)
            store_ns = ctypes.c_float(0)
            load_ns = ctypes.c_float(0)

            ret = cuda_lib.run_single_shot(
                grid, block, n_vec4,
                ctypes.byref(kernel_ns), ctypes.byref(empty_ns),
                ctypes.byref(store_ns), ctypes.byref(load_ns)
            )

            if ret == 0:
                warps_per_cta = block // 32
                total_warps = grid * warps_per_cta
                sms_used = min(grid, args.sm_count)
                warps_per_sm = total_warps / sms_used
                warps_per_smsp = warps_per_sm / args.smsp_per_sm

                results.append({
                    "label": label,
                    "grid": grid,
                    "block": block,
                    "n_vec4": n_vec4,
                    "elements": n_vec4 * 8,
                    "warps_per_smsp": round(warps_per_smsp, 2),
                    "add_kernel_ns": round(kernel_ns.value, 1),
                    "empty_kernel_ns": round(empty_ns.value, 1),
                    "store_only_ns": round(store_ns.value, 1),
                    "load_only_ns": round(load_ns.value, 1),
                    "latency_exposure_ns": round(kernel_ns.value - empty_ns.value, 1),
                })

        output = {
            "probe_type": "single_shot_occupancy_latency",
            "gpu": "H200",
            "method": "nvcc",
            "measurements": results,
            "status": "complete",
        }
        output_path.write_text(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
