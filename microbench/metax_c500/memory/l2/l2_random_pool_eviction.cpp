/*
 * MetaX C500 L2 random-pool eviction probe (timing, cross-AP reload).
 *
 * Block 0 warms a target line and then reads a pool of N random lines from a
 * 2-GiB buffer; blocks on other APs time one dependent reload of the target.
 * The fraction of targets evicted against N is the retention of a single
 * line against random intervening traffic; the empty-pool reload also shows
 * whether a line one AP brought in hits for another AP.  Mechanism evidence
 * only: the reload latency classifies hit vs miss and no duration here may
 * price a Forward path.
 *
 * Build: /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *   --maca-path=/opt/maca -O3 -o l2_random_pool_eviction l2_random_pool_eviction.cpp
 */

#include <mc_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#define CHECK(c) do{mcError_t e=(c); if(e!=mcSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,mcGetErrorString(e));exit(1);}}while(0)
static const int kWords = 16;
template <int R> __device__ __forceinline__ unsigned rd() { return __builtin_mxc_get_hwreg(R); }

__global__ void evict_test(const uint64_t* __restrict__ bufc, const long long* __restrict__ list,
                           int n, long long target, volatile unsigned* flag,
                           unsigned long long* out, unsigned* ids) {
  volatile const uint64_t* buf = bufc;
  __shared__ uint64_t acc_s[256];
  const int tid = threadIdx.x;
  if (tid == 0) ids[blockIdx.x] = rd<4 | (31 << 11)>();
  if (blockIdx.x == 0) {
    uint64_t acc = 0;
    if (tid == 0) acc += buf[target * kWords];
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x) acc += buf[list[i] * kWords];
    acc_s[tid] = acc;
    __syncthreads();
    if (tid == 0) { __threadfence(); *flag = 1u + (unsigned)(acc_s[7] & 0); __threadfence(); }
    return;
  }
  if (tid == 0) {
    unsigned f = 0; int spins = 0;
    while ((f = *flag) == 0u) { spins++; }
    unsigned long long t0 = clock64();
    uint64_t v = buf[target * kWords + (f & 0x10000u ? 1 : 0)];
    uint64_t v2 = buf[target * kWords + 2 + (v & 1)];
    unsigned long long t1 = clock64() + ((v2 + 1) >> 63);
    out[blockIdx.x] = t1 - t0;
  }
}

static uint64_t* g_buf; static long long* g_list; static unsigned long long* g_out; static unsigned* g_flag; static unsigned* g_ids;
static int g_timers = 4;
// returns median over timers x reps of reload latency
static unsigned long long run(const std::vector<long long>& lst, long long target, int reps, std::vector<unsigned>* ids_out=nullptr) {
  CHECK(mcMemcpy(g_list, lst.data(), std::max<size_t>(1,lst.size())*8, mcMemcpyHostToDevice));
  std::vector<unsigned long long> all;
  for (int r = 0; r < reps; r++) {
    CHECK(mcMemset(g_flag, 0, 4));
    evict_test<<<1 + g_timers, 256>>>(g_buf, g_list, (int)lst.size(), target, g_flag, g_out, g_ids);
    CHECK(mcDeviceSynchronize());
    std::vector<unsigned long long> h(1 + g_timers);
    CHECK(mcMemcpy(h.data(), g_out, h.size()*8, mcMemcpyDeviceToHost));
    for (int t = 1; t <= g_timers; t++) all.push_back(h[t]);
    if (ids_out && r == 0) { ids_out->resize(1+g_timers); CHECK(mcMemcpy(ids_out->data(), g_ids, (1+g_timers)*4, mcMemcpyDeviceToHost)); }
  }
  std::sort(all.begin(), all.end());
  return all[all.size()/2];
}
int main(int argc, char** argv) {
  const long long buf_lines = (1LL<<31)/128; // 2 GiB
  CHECK(mcMalloc(&g_buf, buf_lines*128)); CHECK(mcMemset(g_buf, 0, buf_lines*128));
  CHECK(mcMalloc(&g_list, 8LL<<20)); CHECK(mcMalloc(&g_out, 64*8)); CHECK(mcMalloc(&g_flag, 64)); CHECK(mcMalloc(&g_ids, 64*4));
  std::mt19937_64 rng(12345);
  long long target = 777777;
  std::vector<unsigned> ids;
  std::vector<long long> none = {target + 100};
  unsigned long long h0 = run(none, target, 5, &ids);
  printf("empty pool: %llu  ids:", h0); for (auto i : ids) printf(" %08x", i); printf("\n");
  for (double mib : {1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0, 32.0}) {
    long long n = (long long)(mib*1024*1024/128);
    std::vector<long long> l(n); for (auto& x : l) { do { x = rng() % buf_lines; } while (x == target); }
    // several independent targets to get an eviction probability
    int ev = 0; const int T = 16;
    for (int t = 0; t < T; t++) { long long tg = rng() % buf_lines; if (run(l, tg, 3) > 600) ev++; }
    printf("random pool %5.1f MiB: median-lat(target0)=%llu  evicted %d/%d targets\n", mib, run(l, target, 3), ev, T);
  }
  return 0;
}
