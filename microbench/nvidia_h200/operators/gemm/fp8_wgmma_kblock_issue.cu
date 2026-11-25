// H200 FP8 WGMMA k-block issue mechanism probe (shared-memory operands only).
//
// Measured mechanism: the Hopper tensor-pipe service of one BLOCK_K = 128
// FP8 k-block as a warp-group issues it: per "wave" the warp-group fences its
// accumulators, issues four dependent wgmma.mma_async m64nNk32 e4m3 x e4m3 ->
// f32 instructions (K = 4 x 32) against 128-B-swizzled shared-memory
// descriptors, commits the group and waits for it (wait_group 0).  With
// ``promote`` = 1 every thread then applies one FFMA per accumulator into a
// second register array (the per-128-channel FP32 promotion structure of
// block-scaled FP8 kernels).  ``waves`` repeats the wave inside one k-block,
// ``warpgroups`` math warp-groups run concurrently in the CTA and one CTA runs
// on every SM (grid = 132).  Operands are constant shared memory: no global
// load, TMA, mbarrier or production tensor is involved.
//
// Output (stdout): one line per configuration with the median over warp-groups
// of clock64 cycles per k-block and of globaltimer ns per k-block.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a fp8_wgmma_kblock_issue.cu -o probe
// Usage: probe <kblocks>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
  exit(1); } } while (0)

template <int N> struct Wgmma;
template <> struct Wgmma<16> {
  static constexpr int kAccum = 8;
  __device__ __forceinline__ static void mma(uint64_t da, uint64_t db, float* d, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(da), "l"(db), "r"(scale_d));
  }
};

template <> struct Wgmma<32> {
  static constexpr int kAccum = 16;
  __device__ __forceinline__ static void mma(uint64_t da, uint64_t db, float* d, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(da), "l"(db), "r"(scale_d));
  }
};

template <> struct Wgmma<48> {
  static constexpr int kAccum = 24;
  __device__ __forceinline__ static void mma(uint64_t da, uint64_t db, float* d, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %26, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, %24, %25, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
        : "l"(da), "l"(db), "r"(scale_d));
  }
};

template <> struct Wgmma<64> {
  static constexpr int kAccum = 32;
  __device__ __forceinline__ static void mma(uint64_t da, uint64_t db, float* d, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(scale_d));
  }
};

template <> struct Wgmma<80> {
  static constexpr int kAccum = 40;
  __device__ __forceinline__ static void mma(uint64_t da, uint64_t db, float* d, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %42, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39}, %40, %41, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
        : "l"(da), "l"(db), "r"(scale_d));
  }
};

template <> struct Wgmma<112> {
  static constexpr int kAccum = 56;
  __device__ __forceinline__ static void mma(uint64_t da, uint64_t db, float* d, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %58, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n112k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55}, %56, %57, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
        : "l"(da), "l"(db), "r"(scale_d));
  }
};

template <> struct Wgmma<128> {
  static constexpr int kAccum = 64;
  __device__ __forceinline__ static void mma(uint64_t da, uint64_t db, float* d, int scale_d) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(da), "l"(db), "r"(scale_d));
  }
};

__device__ __forceinline__ uint64_t smem_desc(const void* p) {
  // K-major, 128-byte swizzle (layout type 1), stride byte offset 1024 B.
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((addr >> 4) & 0x3FFF);
  desc |= static_cast<uint64_t>(1) << 16;
  desc |= static_cast<uint64_t>(1024 >> 4) << 32;
  desc |= static_cast<uint64_t>(1) << 62;
  return desc;
}

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

template <int N, int WAVES, int PROMOTE>
__global__ void __launch_bounds__(256, 1)
wgmma_kernel(int kblocks, long long* cycles, unsigned long long* nanos,
             float* sink) {
  extern __shared__ __align__(1024) uint8_t smem[];
  constexpr int kA = 64 * 128;           // one m64 x k128 FP8 tile
  constexpr int kB = N * 128;            // one nN x k128 FP8 tile
  const int wg = threadIdx.x / 128;
  for (int i = threadIdx.x; i < (2 * WAVES * kA + kB) / 4; i += blockDim.x)
    reinterpret_cast<uint32_t*>(smem)[i] = 0x38383838u;  // FP8 ~0.5
  __syncthreads();
  const uint8_t* a_base = smem + wg * WAVES * kA;
  const uint8_t* b_base = smem + 2 * WAVES * kA;
  float acc[Wgmma<N>::kAccum];
  float fin[Wgmma<N>::kAccum];
#pragma unroll
  for (int i = 0; i < Wgmma<N>::kAccum; ++i) { acc[i] = 0.f; fin[i] = 0.f; }
  const float scale = 1.0f + threadIdx.x * 1e-7f;
  __syncwarp();
  asm volatile("bar.sync 1, %0;" :: "r"(static_cast<int>(blockDim.x)));
  const long long c0 = clock64();
  const uint64_t t0 = globaltimer();
  for (int kb = 0; kb < kblocks; ++kb) {
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
#pragma unroll
      for (int i = 0; i < Wgmma<N>::kAccum; ++i)
        asm volatile("" : "+f"(acc[i]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        Wgmma<N>::mma(smem_desc(a_base + w * kA + k * 32),
                      smem_desc(b_base + k * 32), acc, k);
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#pragma unroll
      for (int i = 0; i < Wgmma<N>::kAccum; ++i)
        asm volatile("" : "+f"(acc[i]) :: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      if (PROMOTE) {
#pragma unroll
        for (int i = 0; i < Wgmma<N>::kAccum; ++i) fin[i] += scale * acc[i];
      }
    }
  }
  const uint64_t t1 = globaltimer();
  const long long c1 = clock64();
  float total = 0.f;
#pragma unroll
  for (int i = 0; i < Wgmma<N>::kAccum; ++i) total += fin[i] + acc[i];
  if (total == 1234.5f) sink[blockIdx.x] = total;
  if (threadIdx.x % 128 == 0) {
    const int slot = blockIdx.x * 2 + wg;
    cycles[slot] = c1 - c0;
    nanos[slot] = t1 - t0;
  }
}

template <int N, int WAVES, int PROMOTE>
void run(int warpgroups, int kblocks, long long* d_cycles,
         unsigned long long* d_nanos, float* d_sink) {
  constexpr int smem = 2 * WAVES * 64 * 128 + N * 128;
  auto kernel = wgmma_kernel<N, WAVES, PROMOTE>;
  CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem));
  const int grid = 132;
  std::vector<double> best_cycles, best_ns;
  for (int rep = 0; rep < 5; ++rep) {
    CHECK(cudaMemset(d_cycles, 0, sizeof(long long) * grid * 2));
    kernel<<<grid, 128 * warpgroups, smem>>>(kblocks, d_cycles, d_nanos, d_sink);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<long long> c(grid * 2);
    std::vector<unsigned long long> t(grid * 2);
    CHECK(cudaMemcpy(c.data(), d_cycles, sizeof(long long) * grid * 2,
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(t.data(), d_nanos, sizeof(unsigned long long) * grid * 2,
                     cudaMemcpyDeviceToHost));
    std::vector<double> cv, tv;
    for (int b = 0; b < grid; ++b)
      for (int w = 0; w < warpgroups; ++w) {
        cv.push_back(static_cast<double>(c[b * 2 + w]) / kblocks);
        tv.push_back(static_cast<double>(t[b * 2 + w]) / kblocks);
      }
    std::sort(cv.begin(), cv.end());
    std::sort(tv.begin(), tv.end());
    if (rep > 0) {  // rep 0 is the warm-up launch
      best_cycles.push_back(cv[cv.size() / 2]);
      best_ns.push_back(tv[tv.size() / 2]);
    }
  }
  std::sort(best_cycles.begin(), best_cycles.end());
  std::sort(best_ns.begin(), best_ns.end());
  printf("n %d waves %d promote %d warpgroups %d kblocks %d "
         "cycles_per_kblock %.3f ns_per_kblock %.3f\n",
         N, WAVES, PROMOTE, warpgroups, kblocks,
         best_cycles[best_cycles.size() / 2], best_ns[best_ns.size() / 2]);
  fflush(stdout);
}

template <int N>
void run_all(int kblocks, long long* c, unsigned long long* t, float* s) {
  for (int warpgroups = 1; warpgroups <= 2; ++warpgroups) {
    run<N, 1, 0>(warpgroups, kblocks, c, t, s);
    run<N, 1, 1>(warpgroups, kblocks, c, t, s);
    run<N, 2, 0>(warpgroups, kblocks, c, t, s);
    run<N, 2, 1>(warpgroups, kblocks, c, t, s);
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <kblocks>\n", argv[0]);
    return 2;
  }
  const int kblocks = atoi(argv[1]);
  long long* c = nullptr;
  unsigned long long* t = nullptr;
  float* s = nullptr;
  CHECK(cudaMalloc(&c, sizeof(long long) * 264));
  CHECK(cudaMalloc(&t, sizeof(unsigned long long) * 264));
  CHECK(cudaMalloc(&s, sizeof(float) * 132));
  run_all<16>(kblocks, c, t, s);
  run_all<32>(kblocks, c, t, s);
  run_all<48>(kblocks, c, t, s);
  run_all<64>(kblocks, c, t, s);
  run_all<80>(kblocks, c, t, s);
  run_all<112>(kblocks, c, t, s);
  run_all<128>(kblocks, c, t, s);
  return 0;
}
