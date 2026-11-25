"""Independent occupancy-latency probe: L2 latency vs warp occupancy.

Microbenchmark with closed physical units (cycles, nanoseconds).
Measures exposed latency of N independent LDG.128 loads at controlled
warp-per-SMSP occupancy. No production operator timing; this is a pure
hardware latency measurement for model evidence.

Design:
- One CTA per SM (132 CTAs for H200 SM count)
- Each CTA has `nwarps` warps, each warp doing the same independent loads
- Occupancy per SMSP = nwarps/4 (H200 has 4 SMSP per SM)
- Load pattern: each warp issues N independent LDG.E.128 from L2-resident buffer
- Measure kernel duration with cudaEvent; convert to ns via clock calibration
- Expected latency model: single load ~L2 latency; at low occupancy, N loads
  expose N*L2; at high occupancy, warps hide each other's latency
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Each warp: N_INDEP independent 16B loads from L2-resident buffer, then add-reduce
// N_INDEP=1: one load per thread (serial per thread)
// N_INDEP=2: two independent loads per thread (both issue before first use)
// N_INDEP=4: four independent loads per thread

template <int N_INDEP>
__global__ void __launch_bounds__(256) occupancy_latency_kernel(
    const float4* __restrict__ buf, float4* __restrict__ out, int iters)
{
    float4 vals[N_INDEP];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Prime L2
    for (int i = 0; i < N_INDEP; i++) {
        vals[i] = buf[(tid + i * stride) & 0x3FFFF];
    }

    // Timed section: repeated independent loads
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int i = 0; i < N_INDEP; i++) {
            vals[i] = buf[(tid + it * stride + i * stride * 7) & 0x3FFFF];
        }
        // Prevent compiler from removing loads
        float acc = 0.f;
        #pragma unroll
        for (int i = 0; i < N_INDEP; i++) {
            acc += vals[i].x + vals[i].y + vals[i].z + vals[i].w;
        }
        if (acc == -1.f) out[tid & 1023] = make_float4(acc, acc, acc, acc);
    }
}

// Store-only kernel to measure store path latency
__global__ void __launch_bounds__(256) store_only_kernel(
    float4* __restrict__ out, int iters)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float4 v = make_float4(tid, tid+1, tid+2, tid+3);
    for (int it = 0; it < iters; it++) {
        out[(tid + it * blockDim.x * gridDim.x) & 0x3FFFFF] = v;
    }
}

// Load+store kernel matching elementwise add pattern: 2 LDG + 1 STG
__global__ void __launch_bounds__(256) add_pattern_kernel(
    const float4* __restrict__ bufA, const float4* __restrict__ bufB,
    float4* __restrict__ out, int iters)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int it = 0; it < iters; it++) {
        float4 a = bufA[(tid + it * stride) & 0x3FFFF];
        float4 b = bufB[(tid + it * stride) & 0x3FFFF];
        float4 c;
        c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
        out[(tid + it * stride) & 0x3FFFFF] = c;
    }
}

extern "C" int run_probe(int nwarps, int n_indep, int iters, int nblocks,
                         float* out_ns, int* out_blocks, int* out_threads)
{
    // H200: 132 SMs, 4 SMSP per SM
    // nwarps per CTA; nblocks CTAs; total warps = nwarps * nblocks
    // warps per SM = nwarps * nblocks / 132
    // warps per SMSP = warps_per_SM / 4

    // 4MB buffer per input = L2 resident (60MB L2)
    size_t buf_bytes = 4 * 1024 * 1024;
    float4 *bufA, *bufB, *out;
    cudaMalloc(&bufA, buf_bytes);
    cudaMalloc(&bufB, buf_bytes);
    cudaMalloc(&out, 64 * 1024 * 1024);

    // Initialize
    cudaMemset(bufA, 1, buf_bytes);
    cudaMemset(bufB, 2, buf_bytes);

    int threads = nwarps * 32;
    dim3 grid(nblocks), block(threads);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int w = 0; w < 3; w++) {
        if (n_indep == 1)
            occupancy_latency_kernel<1><<<grid, block>>>(bufA, out, iters);
        else if (n_indep == 2)
            occupancy_latency_kernel<2><<<grid, block>>>(bufA, out, iters);
        else
            occupancy_latency_kernel<4><<<grid, block>>>(bufA, out, iters);
    }
    cudaDeviceSynchronize();

    // Timed
    cudaEventRecord(start);
    for (int r = 0; r < 10; r++) {
        if (n_indep == 1)
            occupancy_latency_kernel<1><<<grid, block>>>(bufA, out, iters);
        else if (n_indep == 2)
            occupancy_latency_kernel<2><<<grid, block>>>(bufA, out, iters);
        else
            occupancy_latency_kernel<4><<<grid, block>>>(bufA, out, iters);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *out_ns = ms * 1e6f / 10; // ns per kernel

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(out);

    *out_blocks = nblocks;
    *out_threads = threads;
    return 0;
}

extern "C" int run_store_probe(int nwarps, int iters, int nblocks, float* out_ns)
{
    size_t out_bytes = 64 * 1024 * 1024;
    float4 *out;
    cudaMalloc(&out, out_bytes);
    cudaMemset(out, 0, out_bytes);

    int threads = nwarps * 32;
    dim3 grid(nblocks), block(threads);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int w = 0; w < 3; w++)
        store_only_kernel<<<grid, block>>>(out, iters);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int r = 0; r < 10; r++)
        store_only_kernel<<<grid, block>>>(out, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *out_ns = ms * 1e6f / 10;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(out);
    return 0;
}

extern "C" int run_add_probe(int nwarps, int iters, int nblocks, float* out_ns)
{
    size_t buf_bytes = 4 * 1024 * 1024;
    float4 *bufA, *bufB, *out;
    cudaMalloc(&bufA, buf_bytes);
    cudaMalloc(&bufB, buf_bytes);
    cudaMalloc(&out, 64 * 1024 * 1024);
    cudaMemset(bufA, 1, buf_bytes);
    cudaMemset(bufB, 2, buf_bytes);

    int threads = nwarps * 32;
    dim3 grid(nblocks), block(threads);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int w = 0; w < 3; w++)
        add_pattern_kernel<<<grid, block>>>(bufA, bufB, out, iters);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int r = 0; r < 10; r++)
        add_pattern_kernel<<<grid, block>>>(bufA, bufB, out, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *out_ns = ms * 1e6f / 10;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(bufA); cudaFree(bufB); cudaFree(out);
    return 0;
}
"""


def compile_cuda(output_dir):
    """Compile CUDA source to shared library."""
    src = Path(output_dir) / "probe.cu"
    lib = Path(output_dir) / "probe.so"
    src.write_text(CUDA_SRC)

    # Try nvcc first
    nvcc = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
    if nvcc.returncode == 0:
        cmd = [
            "nvcc", "-shared", "-Xcompiler", "-fPIC",
            "-arch=sm_90a",  # H200 is sm_90a
            "-o", str(lib), str(src),
            "-lcudart", "-lcuda"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return str(lib), "nvcc"

    # Fallback: use torch cpp_extension (compiles with CUDA)
    try:
        import torch
        from torch.utils.cpp_extension import load

        # Write to a temp file
        cpp_src = src.read_text()

        lib = load(
            name="occupancy_probe",
            sources=[str(src)],
            extra_cuda_cflags=["-arch=sm_90a", "--use_fast_math"],
            build_directory=output_dir,
            verbose=False,
        )
        return str(lib), "torch_cpp_extension"
    except Exception as e:
        return None, f"compile_failed: {e}, nvcc_err: {nvcc.stderr if nvcc.returncode != 0 else 'n/a'}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--sm-count", type=int, default=132, help="H200 SM count")
    parser.add_argument("--sm-per-smsp", type=int, default=4, help="SMSP per SM")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load and run via ctypes
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        lib_path, method = compile_cuda(tmpdir)

        results = {
            "probe_type": "occupancy_latency",
            "gpu": "H200",
            "method": method,
            "sm_count": args.sm_count,
            "smsp_per_sm": args.sm_per_smsp,
            "measurements": []
        }

        if lib_path and method != "compile_failed":
            import ctypes
            lib = ctypes.CDLL(lib_path)

            lib.run_probe.restype = ctypes.c_int
            lib.run_probe.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int)
            ]

            lib.run_store_probe.restype = ctypes.c_int
            lib.run_store_probe.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float)
            ]

            lib.run_add_probe.restype = ctypes.c_int
            lib.run_add_probe.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float)
            ]

            # Configurations: (nwarps_per_cta, n_indep_loads, iters, nblocks)
            # Occupancy = nwarps_per_cta * nblocks / (sm_count * smsp_per_sm)
            configs = [
                # Low occupancy: 1 warp/SMSP = 4 warps/SM
                # 132 CTAs of 4 warps = 528 warps / 132 SMs = 4 warps/SM = 1/SMSP
                (4, 1, 1000, 132),   # 1 warp/SMSP, 1 load
                (4, 2, 1000, 132),   # 1 warp/SMSP, 2 loads
                (4, 4, 1000, 132),   # 1 warp/SMSP, 4 loads
                # Medium: 2 warps/SMSP = 8 warps/SM
                (8, 1, 1000, 132),   # 2 warp/SMSP, 1 load
                (8, 2, 1000, 132),   # 2 warp/SMSP, 2 loads
                (8, 4, 1000, 132),   # 2 warp/SMSP, 4 loads
                # High: 8 warps/SMSP = 32 warps/SM
                (32, 1, 1000, 132),  # 8 warp/SMSP, 1 load
                (32, 2, 1000, 132),  # 8 warp/SMSP, 2 loads
                (32, 4, 1000, 132),  # 8 warp/SMSP, 4 loads
                # Very high: 16 warps/SMSP = 64 warps/SM (max)
                (64, 1, 1000, 66),   # 16 warps/SMSP (2 CTAs of 64 warps)
                (64, 2, 1000, 66),   # 16 warps/SMSP, 2 loads
                (64, 4, 1000, 66),   # 16 warps/SMSP, 4 loads
            ]

            iters = 1000
            for nwarps, n_indep, iters, nblocks in configs:
                out_ns = ctypes.c_float(0)
                out_blocks = ctypes.c_int(0)
                out_threads = ctypes.c_int(0)

                ret = lib.run_probe(nwarps, n_indep, iters, nblocks,
                                    ctypes.byref(out_ns),
                                    ctypes.byref(out_blocks),
                                    ctypes.byref(out_threads))
                if ret == 0:
                    total_warps = nwarps * nblocks
                    warps_per_sm = total_warps / args.sm_count
                    warps_per_smsp = warps_per_sm / args.sm_per_smsp
                    results["measurements"].append({
                        "nwarps_per_cta": nwarps,
                        "n_indep_loads": n_indep,
                        "iters": iters,
                        "nblocks": nblocks,
                        "total_warps": total_warps,
                        "warps_per_sm": round(warps_per_sm, 2),
                        "warps_per_smsp": round(warps_per_smsp, 2),
                        "kernel_ns": round(out_ns.value, 1),
                        "ns_per_iter": round(out_ns.value / iters, 2),
                    })

            # Store-only probes
            store_configs = [
                (4, 1000, 132),   # 1 warp/SMSP
                (8, 1000, 132),   # 2 warp/SMSP
                (32, 1000, 132),  # 8 warp/SMSP
                (64, 1000, 66),   # 16 warp/SMSP
            ]
            for nwarps, iters, nblocks in store_configs:
                out_ns = ctypes.c_float(0)
                ret = lib.run_store_probe(nwarps, iters, nblocks,
                                          ctypes.byref(out_ns))
                if ret == 0:
                    total_warps = nwarps * nblocks
                    warps_per_smsp = total_warps / args.sm_count / args.sm_per_smsp
                    results["measurements"].append({
                        "probe": "store_only",
                        "nwarps_per_cta": nwarps,
                        "nblocks": nblocks,
                        "warps_per_smsp": round(warps_per_smsp, 2),
                        "kernel_ns": round(out_ns.value, 1),
                        "ns_per_iter": round(out_ns.value / iters, 2),
                    })

            # Add pattern probes (2 LDG + 1 STG, matching elementwise add)
            add_configs = [
                (4, 1000, 132),   # 1 warp/SMSP
                (8, 1000, 132),   # 2 warp/SMSP
                (32, 1000, 132),  # 8 warp/SMSP
                (64, 1000, 66),   # 16 warp/SMSP
            ]
            for nwarps, iters, nblocks in add_configs:
                out_ns = ctypes.c_float(0)
                ret = lib.run_add_probe(nwarps, iters, nblocks,
                                        ctypes.byref(out_ns))
                if ret == 0:
                    total_warps = nwarps * nblocks
                    warps_per_smsp = total_warps / args.sm_count / args.sm_per_smsp
                    results["measurements"].append({
                        "probe": "add_pattern_2ldg_1stg",
                        "nwarps_per_cta": nwarps,
                        "nblocks": nblocks,
                        "warps_per_smsp": round(warps_per_smsp, 2),
                        "kernel_ns": round(out_ns.value, 1),
                        "ns_per_iter": round(out_ns.value / iters, 2),
                    })

            results["status"] = "complete"
        else:
            results["status"] = "compile_failed"
            results["error"] = method

        output_path.write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
