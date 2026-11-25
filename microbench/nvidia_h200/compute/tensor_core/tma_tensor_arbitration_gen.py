"""Reuse primitive helpers for an independent copy/Tensor contention probe."""
from pathlib import Path
import sys

if __name__=='__main__':
    tma,math,output=map(Path,sys.argv[1:])
    copy=tma.read_text();matrix=math.read_text()
    prefix=copy[:copy.index('__global__ void marker_kernel')]
    descriptor=copy[copy.index('static PFN_cuTensorMapEncodeTiled_v12000 encode_tiled()'):copy.index('int main(')]
    helpers=matrix[matrix.index('__device__ __forceinline__ uint64_t make_desc'):matrix.index('__device__ __forceinline__ void wgmma_ss_n64_kmajor')]
    request=matrix[matrix.index('__device__ __forceinline__ void wgmma_ss_n128_kmajor'):matrix.index('__device__ __forceinline__ void wgmma_rs_n128_mnmajor')]
    output.write_text(prefix+descriptor+helpers+request.replace('.bf16.bf16','.f16.f16'))
