/*
 * MetaX C500 CTA dispatch-order probe.
 *
 * Every CTA of a memory-bound streaming kernel records the global realtime
 * at its first and last instruction and its hardware location register.
 * Argument 3/4 (slow_mod, slow_res) doubles the work of one pid residue
 * class.  If CTAs were bound to per-cluster queues by pid, that class would
 * fall progressively behind; under one global dynamic dispatcher it only
 * carries a constant lag.  Mechanism evidence only: no duration here may
 * price a Forward path.
 *
 * Build: /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *   --maca-path=/opt/maca -O3 -o cta_dispatch_order cta_dispatch_order.cpp
 * Run:   ./cta_dispatch_order <grid> <steps> [slow_mod slow_res]
 */

#include <mc_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#define CHECK(c) do{mcError_t e=(c); if(e!=mcSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,mcGetErrorString(e));exit(1);}}while(0)
template <int R> __device__ __forceinline__ unsigned rd() { return __builtin_mxc_get_hwreg(R); }
__global__ void body(const uint4* __restrict__ src, int steps, int stride_vec, unsigned long long* rec, uint4* sink, int slow_mod, int slow_res) {
  const int pid = blockIdx.x, tid = threadIdx.x;
  if (slow_mod > 0 && pid % slow_mod == slow_res) steps *= 2;
  unsigned long long t0 = 0;
  if (tid == 0) t0 = __builtin_mxc_get_realtime();
  uint4 acc = make_uint4(0,0,0,0);
  const uint4* base = src + (size_t)pid * stride_vec;
  for (int s = 0; s < steps; s++) { uint4 v = base[(size_t)s * 256 + tid]; acc.x ^= v.x; acc.y += v.y; }
  if (acc.x == 0xdeadbeef) sink[pid] = acc;
  __syncthreads();
  if (tid == 0) {
    unsigned long long t1 = __builtin_mxc_get_realtime();
    rec[pid*3+0] = t0; rec[pid*3+1] = t1; rec[pid*3+2] = rd<4 | (31<<11)>();
  }
}
int main(int argc, char** argv) {
  int grid = argc > 1 ? atoi(argv[1]) : 46080;
  int steps = argc > 2 ? atoi(argv[2]) : 64;
  // each CTA streams steps x 4 KiB from its own window; windows overlap by
  // stride so neighbouring CTAs share nothing
  int stride_vec = steps * 256;
  size_t nvec = (size_t)grid * stride_vec + (size_t)steps * 256 * 2 + 256;
  uint4* src; CHECK(mcMalloc(&src, nvec * 16)); CHECK(mcMemset(src, 1, nvec*16));
  uint4* sink; CHECK(mcMalloc(&sink, grid*16));
  unsigned long long* rec; CHECK(mcMalloc(&rec, (size_t)grid*3*8));
  int slow_mod = argc > 3 ? atoi(argv[3]) : 0; int slow_res = argc > 4 ? atoi(argv[4]) : 0;
  body<<<grid,256>>>(src, steps, stride_vec, rec, sink, slow_mod, slow_res); CHECK(mcDeviceSynchronize());
  body<<<grid,256>>>(src, steps, stride_vec, rec, sink, slow_mod, slow_res); CHECK(mcDeviceSynchronize());
  std::vector<unsigned long long> h((size_t)grid*3); CHECK(mcMemcpy(h.data(), rec, h.size()*8, mcMemcpyDeviceToHost));
  for (int p = 0; p < grid; p++) printf("%d %llu %llu %llu\n", p, h[p*3], h[p*3+1], h[p*3+2]);
}
