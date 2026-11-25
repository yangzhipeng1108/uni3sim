// Compile-only source-size semantics. No clocks, polling or matrix workload.
#include <cuda_runtime.h>

template<int SourceBytes>
__device__ __forceinline__ void copy_size(const char* p,unsigned* q) {
    __shared__ __align__(16) char data[32*16];
    char* lane=data+threadIdx.x*16;
    unsigned dst=unsigned(__cvta_generic_to_shared(lane));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 ::"r"(dst),"l"(p+threadIdx.x*16),"n"(SourceBytes):"memory");
    asm volatile("cp.async.commit_group;":::"memory");
    asm volatile("cp.async.wait_group 0;":::"memory");
    q[threadIdx.x]=*reinterpret_cast<volatile unsigned*>(lane);
}

extern "C" __global__ __launch_bounds__(32) void zero_fill(const char* p,unsigned* q) {
    copy_size<0>(p,q);
}
extern "C" __global__ __launch_bounds__(32) void full_copy(const char* p,unsigned* q) {
    copy_size<16>(p,q);
}
