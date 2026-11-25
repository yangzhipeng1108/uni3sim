// Untimed compiler binding only: two copy groups per shared operand window.
// No host runner, device clocks, polling, matrix work, routing or output body.
#include <cuda_runtime.h>

template<int Slots>
__device__ __forceinline__ void group_window(const char* input, unsigned* output) {
    __shared__ __align__(16) char storage[32*Slots*32];
    char* lane_storage=storage+threadIdx.x*Slots*32;
    #pragma unroll
    for (int s=0;s<Slots;++s) {
        unsigned a=unsigned(__cvta_generic_to_shared(lane_storage+s*32));
        unsigned b=a+16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     ::"r"(a),"l"(input+s*32):"memory");
        asm volatile("cp.async.commit_group;":::"memory");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     ::"r"(b),"l"(input+s*32+16):"memory");
        asm volatile("cp.async.commit_group;":::"memory");
    }
    asm volatile("cp.async.wait_group %0;"::"n"(2*(Slots-1)):"memory");
    __syncthreads();
    // Consume only the oldest window; a later wait drains the newer groups.
    unsigned a=*reinterpret_cast<volatile unsigned*>(lane_storage);
    unsigned b=*reinterpret_cast<volatile unsigned*>(lane_storage+16);
    asm volatile("cp.async.wait_group 0;":::"memory");
    output[threadIdx.x]=a^b;
}

#define BIND(S) extern "C" __global__ __launch_bounds__(32) void window_##S(const char* p,unsigned* q) { group_window<S>(p,q); }
BIND(1)
BIND(2)
BIND(3)
BIND(4)
