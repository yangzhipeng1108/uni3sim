// One corrected compile-gate attempt for the pinned vLLM 0.30 FA3 Q729 GQA.
// Q arrives by 5D TMA; paged K/V arrive by two-stage cp.async.  The probe
// keeps one CTA and a QK-next/PV-current asynchronous window, with copy-only,
// math-only, and combined timing controls.  It omits causal masks, complete
// softmax, persistent task selection and Attention output.
// CUDA 12.2+ sm_90a required.  Usage: probe MODE (0 copy, 1 math, 2 combined).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(2); } } while (0)

constexpr int kPanelBytes = 32768;
constexpr int kSmemBytes = 166912;
constexpr int kGrid = 132;
constexpr int kStages = 2;
constexpr int kQBase = 0;
constexpr int kKBase = kPanelBytes;
constexpr int kVBase = kKBase + kStages * kPanelBytes;

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint64_t make_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = uint64_t((addr >> 4) & 0x3fff);
  d |= uint64_t((lbo >> 4) & 0x3fff) << 16;
  d |= uint64_t((sbo >> 4) & 0x3fff) << 32;
  d |= uint64_t(1) << 62;  // 128-byte swizzle
  return d;
}
__device__ __forceinline__ void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wg_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}
__device__ __forceinline__ void wg_wait1() {
  asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
}
__device__ __forceinline__ void wait_bar(uint32_t address) {
  asm volatile("{ .reg .pred done; WAIT_%=: "
               "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], 0; "
               "@!done bra WAIT_%=; }" :: "r"(address) : "memory");
}
__device__ __forceinline__ void tma_q(uint32_t address,
    const CUtensorMap* map, uint32_t barrier) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global."
               "mbarrier::complete_tx::bytes "
               "[%0], [%1, {%3, %4, %5, %6, %7}], [%2];" ::
               "r"(address), "l"(map), "r"(barrier), "r"(0), "r"(0),
               "r"(0), "r"(0), "r"(0) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t address, const void* source) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
               "r"(address), "l"(source) : "memory");
}
__device__ __forceinline__ void cp_async_arrive(uint32_t barrier) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::
               "r"(barrier) : "memory");
}
__device__ __forceinline__ void arrive_one() {
  asm volatile("bar.arrive 1, 256;" ::: "memory");
}
__device__ __forceinline__ void arrive_two() {
  asm volatile("bar.arrive 2, 256;" ::: "memory");
}
__device__ __forceinline__ void wait_one() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}
__device__ __forceinline__ void wait_two() {
  asm volatile("bar.sync 2, 256;" ::: "memory");
}

__device__ __forceinline__ void wgmma_ss8(float (&d)[64], uint64_t const (&a)[8], uint64_t const (&b)[8]) {
    asm volatile(
        "{\n.reg .pred p0,p1;\nsetp.ne.b32 p0, 0, 0;\nsetp.ne.b32 p1, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p0, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %66, %67, p1, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %68, %69, p1, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %70, %71, p1, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %72, %73, p1, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %74, %75, p1, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %76, %77, p1, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %78, %79, p1, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(a[0]), "l"(b[0]), "l"(a[1]), "l"(b[1]), "l"(a[2]), "l"(b[2]), "l"(a[3]), "l"(b[3]), "l"(a[4]), "l"(b[4]), "l"(a[5]), "l"(b[5]), "l"(a[6]), "l"(b[6]), "l"(a[7]), "l"(b[7]) : "memory");
}

__device__ __forceinline__ void wgmma_rs8(float (&d)[64], uint32_t const (&a)[4], uint64_t const (&b)[8]) {
    asm volatile(
        "{\n.reg .pred p0,p1;\nsetp.ne.b32 p0, 0, 0;\nsetp.ne.b32 p1, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p0, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %69, p1, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %70, p1, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %71, p1, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %72, p1, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %73, p1, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %74, p1, 1, 1, 1;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %75, p1, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]), "l"(b[6]), "l"(b[7]) : "memory");
}

__device__ __forceinline__ void local_fragment_prob(float (&score)[64], uint32_t (&prob)[4]) {
  float maximum = score[0];
  #pragma unroll 1
  for (int i = 1; i < 64; ++i) maximum = fmaxf(maximum, score[i]);
  float sum = 0.f;
  #pragma unroll 1
  for (int i = 0; i < 64; ++i) {
    score[i] = exp2f((score[i] - maximum) * 1.4426950408889634f);
    sum += score[i];
  }
  float inv = 1.f / sum;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint16_t a = __bfloat16_as_ushort(__float2bfloat16(score[2*i] * inv));
    uint16_t b = __bfloat16_as_ushort(__float2bfloat16(score[2*i+1] * inv));
    prob[i] = uint32_t(a) | (uint32_t(b) << 16);
  }
}
__device__ __forceinline__ void qk_desc(uint32_t base, int wg, int stage,
    uint64_t (&a)[8], uint64_t (&b)[8]) {
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    a[k] = make_desc(base + kQBase + wg * 16384 + (k % 4) * 32, 16, 1024);
    b[k] = make_desc(base + kKBase + stage * kPanelBytes
                     + (k % 4) * 32 + (k / 4) * 16384, 16, 1024);
  }
}
__device__ __forceinline__ void pv_desc(uint32_t base, int stage,
    uint64_t (&b)[8]) {
  #pragma unroll
  for (int k = 0; k < 8; ++k)
    b[k] = make_desc(base + kVBase + stage * kPanelBytes + k * 2048, 8192, 1024);
}
__global__ void __launch_bounds__(384, 1) probe(
    const __grid_constant__ CUtensorMap q_map,
    const uint8_t* __restrict__ k_source,
    const uint8_t* __restrict__ v_source,
    const int* __restrict__ page_table,
    int mode, float* checksum, unsigned long long* ticks) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ __align__(8) unsigned long long q_bar;
  __shared__ __align__(8) unsigned long long kv_bar[4];
  int tid = threadIdx.x;
  int wg = tid / 128;
  uint32_t base = smem_addr(smem);
  if (tid == 0) {
    uint32_t q = smem_addr(&q_bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(q) : "memory");
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t bar = smem_addr(&kv_bar[i]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" :: "r"(bar) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) {
    uint32_t q = smem_addr(&q_bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                 "r"(q), "r"(kPanelBytes) : "memory");
    tma_q(base + kQBase, &q_map, q);
  }
  if (wg == 2) {
    int lane = tid - 256;
    #pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      int page = page_table[blockIdx.x * kStages + stage];
      const uint8_t* k = k_source + page * kPanelBytes;
      const uint8_t* v = v_source + page * kPanelBytes;
      #pragma unroll
      for (int chunk = 0; chunk < 16; ++chunk) {
        int byte = lane * 16 + chunk * 2048;
        cp_async16(base + kKBase + stage * kPanelBytes + byte, k + byte);
        cp_async16(base + kVBase + stage * kPanelBytes + byte, v + byte);
      }
      cp_async_arrive(smem_addr(&kv_bar[2 * stage]));
      cp_async_arrive(smem_addr(&kv_bar[2 * stage + 1]));
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
  } else {
    // mode 0: copy-only, mode 1: math after both stages are ready,
    // mode 2: mixed admission (second-stage copy may overlap first QK).
    unsigned long long begin = clock64();
    wait_bar(smem_addr(&q_bar));
    wait_bar(smem_addr(&kv_bar[0]));
    wait_bar(smem_addr(&kv_bar[1]));
    if (mode == 1) {
      wait_bar(smem_addr(&kv_bar[2]));
      wait_bar(smem_addr(&kv_bar[3]));
      begin = clock64();
    }
    if (mode == 0) {
      wait_bar(smem_addr(&kv_bar[2]));
      wait_bar(smem_addr(&kv_bar[3]));
      if ((tid & 127) == 0)
        checksum[blockIdx.x * 2 + wg] = float(smem[kQBase] + smem[kKBase]
                                                 + smem[kVBase]);
    } else {
      float score0[64], score1[64], value0[64], value1[64];
      #pragma unroll
      for (int i = 0; i < 64; ++i) {
        score0[i] = 0.f; score1[i] = 0.f;
        value0[i] = 0.f; value1[i] = 0.f;
      }
      uint64_t qa[8], kb[8], vb[8];
      uint32_t prob0[4], prob1[4];
      wg_fence();
      qk_desc(base, wg, 0, qa, kb);
      wgmma_ss8(score0, qa, kb);
      wg_commit();
      wg_wait0();  // first tile must provide probabilities to following PV
      local_fragment_prob(score0, prob0);
      if (mode == 2) {
        wait_bar(smem_addr(&kv_bar[2]));
        wait_bar(smem_addr(&kv_bar[3]));
      }
      if (wg == 0) arrive_one();
      if (wg == 0) wait_one(); else wait_two();
      qk_desc(base, wg, 1, qa, kb);
      wgmma_ss8(score1, qa, kb);
      wg_commit();
      pv_desc(base, 0, vb);
      wgmma_rs8(value0, prob0, vb);
      wg_commit();
      if (wg == 0) arrive_two(); else arrive_one();
      wg_wait1();  // QK-next done; PV-current may still be in flight
      local_fragment_prob(score1, prob1);
      pv_desc(base, 1, vb);
      wgmma_rs8(value1, prob1, vb);
      wg_commit();
      wg_wait0();
      if ((tid & 127) == 0)
        checksum[blockIdx.x * 2 + wg] = value0[0] + value1[0];
    }
    if ((tid & 127) == 0) {
      unsigned long long stop = clock64();
      ticks[(blockIdx.x * 2 + wg) * 2] = begin;
      ticks[(blockIdx.x * 2 + wg) * 2 + 1] = stop;
    }
  }
}

CUtensorMap make_map(uint8_t* source) {
  void* address = nullptr;
  cudaDriverEntryPointQueryResult status;
  CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &address,
                                cudaEnableDefault, &status));
  if (address == nullptr || status != cudaDriverEntryPointSuccess) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled unavailable\n");
    std::exit(2);
  }
  auto encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
  CUtensorMap map;
  cuuint64_t dims[5] = {128, 256, 1, 1, 1};
  cuuint64_t strides[4] = {128, kPanelBytes, kPanelBytes, kPanelBytes};
  cuuint32_t box[5] = {128, 256, 1, 1, 1};
  cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
  CUresult result = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, source,
                           dims, strides, box, element_strides,
                           CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", int(result));
    std::exit(2);
  }
  return map;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s MODE (0 copy, 1 math, 2 combined)\n", argv[0]);
    return 1;
  }
  int mode = std::atoi(argv[1]);
  if (mode < 0 || mode > 2) return 2;
  std::mt19937 generator(42);
  std::vector<uint16_t> host(5 * kPanelBytes / 2);
  for (auto& item : host) item = uint16_t(0x3f00u + (generator() % 32u));
  uint8_t *source = nullptr;
  int *pages = nullptr;
  float* output = nullptr;
  unsigned long long* ticks = nullptr;
  CHECK(cudaMalloc(&source, host.size() * sizeof(uint16_t)));
  CHECK(cudaMemcpy(source, host.data(), host.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice));
  std::vector<int> host_pages(kGrid * kStages);
  for (int i = 0; i < kGrid; ++i) {
    host_pages[i * kStages] = 0;
    host_pages[i * kStages + 1] = 1;
  }
  CHECK(cudaMalloc(&pages, host_pages.size() * sizeof(int)));
  CHECK(cudaMemcpy(pages, host_pages.data(), host_pages.size() * sizeof(int),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&output, kGrid * 2 * sizeof(float)));
  CHECK(cudaMalloc(&ticks, kGrid * 4 * sizeof(unsigned long long)));
  CUtensorMap q = make_map(source);
  CHECK(cudaFuncSetAttribute(probe,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  probe<<<kGrid, 384, kSmemBytes>>>(q,
      source + kPanelBytes, source + 3 * kPanelBytes,
      pages, mode, output, ticks);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<float> sums(kGrid * 2);
  CHECK(cudaMemcpy(sums.data(), output, sums.size() * sizeof(float),
                   cudaMemcpyDeviceToHost));
  std::vector<unsigned long long> times(kGrid * 4);
  CHECK(cudaMemcpy(times.data(), ticks, times.size() * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost));
  double checksum = 0;
  for (float value : sums) {
    if (!std::isfinite(value)) return 3;
    checksum += value;
  }
  unsigned long long total_cycles = 0;
  for (int i = 0; i < kGrid * 2; ++i)
    total_cycles += times[2 * i + 1] - times[2 * i];
  std::printf("{\"mode\":%d,\"checksum\":%.12g,\"mean_cta_cycles\":%.3f}\n",
              mode, checksum, double(total_cycles) / (kGrid * 2));
  CHECK(cudaFree(ticks));
  CHECK(cudaFree(pages));
  CHECK(cudaFree(output));
  CHECK(cudaFree(source));
  return 0;
}
