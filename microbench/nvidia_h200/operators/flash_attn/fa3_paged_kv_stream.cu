// H200 (sm_90) paged-KV cp.async two-stage operand stream probe.
//
// Mechanism: the operand service of the FlashAttention-3 MLA decode producer
// (vllm-flash-attention 803020a8, hopper/paged_kv.h PagedKVManager with
// kBlockN=64, kHeadDim=64, kHeadDimV=512, NumProducerThreads=128,
// KV_Same_Iter=true; kStages=2 in flash_fwd_launch_template.h).  One
// producer warpgroup (128 threads) streams KV blocks of 64 cache rows from a
// paged MLA cache (row = 576 BF16 = 1152 B: 512 latent + 64 rope channels;
// page_size rows per page) into a two-stage shared-memory ring:
//   per block and thread: one page-table int load (8 threads share one row
//   group via __shfl_sync), 4 x 16-B cp.async.cg for K (rope part) and
//   32 x 16-B cp.async.cg for V (latent part), then
//   cp.async.mbarrier.arrive.noinc on the stage's full barrier.
// Two consumer warpgroups wait on the full barrier, hold the stage for
// HOLD_CYCLES (a synthetic, compute-free occupancy of the stage), and
// release it.  No MMA, softmax or epilogue is executed: the probe measures
// only how fast a finite two-deep ring of this load pattern is served from
// DRAM (the cache is flushed from L2 before every trial) as a function of
// the number of concurrently streaming SMs and of the stage hold time.
//
// usage: fa3_paged_kv_stream GRID BLOCKS HOLD_CYCLES PAGE_SIZE TRIALS
// prints: grid,blocks,hold,page_size,trial,cta,first_full_cycles,
//         steady_ii_cycles,total_cycles,kernel_ns
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

constexpr int kBlockN = 64;
constexpr int kRowElems = 576;               // 512 latent + 64 rope
constexpr int kKBytes = kBlockN * 64 * 2;    // 8 KiB rope part
constexpr int kVBytes = kBlockN * 512 * 2;   // 64 KiB latent part
constexpr int kStageBytes = kKBytes + kVBytes;
constexpr int kStages = 2;
constexpr int kSmemBytes = kStages * kStageBytes + 1024;
constexpr int kMaxBlocks = 256;

__device__ __forceinline__ void mbar_init(uint32_t addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, int parity) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" :: "r"(addr), "r"(parity) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void cp_async_arrive_noinc(uint32_t addr) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem) : "memory");
}

__global__ void __launch_bounds__(384, 1)
stream(const uint16_t* __restrict__ cache, const int* __restrict__ page_table,
       int blocks, int hold, int page_size, int pages_per_cta,
       unsigned long long* __restrict__ out) {
    extern __shared__ __align__(1024) unsigned char smem[];
    __shared__ __align__(8) unsigned long long bars[2 * kStages];
    const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const uint32_t bar_base = static_cast<uint32_t>(__cvta_generic_to_shared(bars));
    auto full_bar = [&](int s) { return bar_base + 8u * s; };
    auto empty_bar = [&](int s) { return bar_base + 8u * (kStages + s); };
    if (threadIdx.x == 0) {
        for (int s = 0; s < kStages; ++s) { mbar_init(full_bar(s), 128); mbar_init(empty_bar(s), 8); }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    const int* my_pages = page_table + static_cast<long long>(blockIdx.x) * pages_per_cta;
    unsigned long long* my_out = out + static_cast<long long>(blockIdx.x) * (2 * kMaxBlocks + 2);
    if (threadIdx.x < 128) {
        const int t = threadIdx.x;
        const int row_in_pass = t / 8;      // 16 rows per pass, 8 threads x 16 B per 128-B row chunk
        const int chunk = t % 8;
        if (t == 0) my_out[0] = clock64();
        for (int b = 0; b < blocks; ++b) {
            const int s = b % kStages;
            const int use = b / kStages;
            if (b >= kStages) mbar_wait(empty_bar(s), (use - 1) & 1);
            // Page-table lookup: lane (row group leader + i) loads the page of pass i's row.
            const int lane = t % 32;
            const int group_leader = lane & ~7;
            int my_page = 0;
            if (chunk < 4) {
                const int row = b * kBlockN + chunk * 16 + row_in_pass;
                my_page = my_pages[row / page_size];
            }
            const uint32_t k_smem = smem_base + s * kStageBytes;
            const uint32_t v_smem = k_smem + kKBytes;
            #pragma unroll
            for (int pass = 0; pass < 4; ++pass) {
                const int page = __shfl_sync(0xffffffffu, my_page, group_leader + pass);
                const int row = b * kBlockN + pass * 16 + row_in_pass;
                const long long token = static_cast<long long>(page) * page_size + row % page_size;
                const uint16_t* row_ptr = cache + token * kRowElems;
                const int smem_row = pass * 16 + row_in_pass;
                cp_async16(k_smem + (smem_row * 64 + chunk * 8) * 2, row_ptr + 512 + chunk * 8);
                #pragma unroll
                for (int c = 0; c < 8; ++c)
                    cp_async16(v_smem + (smem_row * 512 + c * 64 + chunk * 8) * 2,
                               row_ptr + c * 64 + chunk * 8);
            }
            cp_async_arrive_noinc(full_bar(s));
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
    } else {
        const int ct = threadIdx.x - 128;
        for (int b = 0; b < blocks; ++b) {
            const int s = b % kStages;
            mbar_wait(full_bar(s), (b / kStages) & 1);
            const unsigned long long seen = clock64();
            if (ct == 0) my_out[1 + b] = seen;
            if (hold > 0) { while (clock64() - seen < static_cast<unsigned long long>(hold)) { } }
            __syncwarp();
            if (ct % 32 == 0) mbar_arrive(empty_bar(s));
            if (ct == 0) my_out[1 + kMaxBlocks + b] = clock64();
        }
    }
}

__global__ void flush(uint4* buf, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        buf[i] = make_uint4(i, i, i, i);
}

int main(int argc, char** argv) {
    if (argc != 6) { std::fprintf(stderr, "usage: %s GRID BLOCKS HOLD PAGE_SIZE TRIALS\n", argv[0]); return 1; }
    const int grid = std::atoi(argv[1]), blocks = std::atoi(argv[2]), hold = std::atoi(argv[3]);
    const int page_size = std::atoi(argv[4]), trials = std::atoi(argv[5]);
    if (grid < 1 || blocks < 3 || blocks > kMaxBlocks || hold < 0 || page_size < 1 ||
        (kBlockN % page_size != 0 && page_size % kBlockN != 0) || trials < 1) {
        std::fprintf(stderr, "bad args\n"); return 1;
    }
    const int rows_per_cta = blocks * kBlockN;
    const int pages_per_cta = (rows_per_cta + page_size - 1) / page_size;
    const long long total_pages = static_cast<long long>(pages_per_cta) * grid;
    // Pages are allocated in request order (a fresh vLLM block pool hands out
    // ascending block ids); every CTA streams its own request's pages.
    std::vector<int> table(total_pages);
    for (long long i = 0; i < total_pages; ++i) table[i] = static_cast<int>(i);
    uint16_t* cache = nullptr; int* d_table = nullptr; unsigned long long* out = nullptr; uint4* scratch = nullptr;
    const long long cache_elems = total_pages * page_size * static_cast<long long>(kRowElems);
    CHECK(cudaMalloc(&cache, cache_elems * 2));
    CHECK(cudaMemset(cache, 0x3c, cache_elems * 2));
    CHECK(cudaMalloc(&d_table, total_pages * sizeof(int)));
    CHECK(cudaMemcpy(d_table, table.data(), total_pages * sizeof(int), cudaMemcpyHostToDevice));
    const long long out_len = static_cast<long long>(grid) * (2 * kMaxBlocks + 2);
    CHECK(cudaMalloc(&out, out_len * sizeof(unsigned long long)));
    const long long scratch_vec = (256ll << 20) / 16;  // 256 MiB > 60 MiB L2
    CHECK(cudaMalloc(&scratch, scratch_vec * 16));
    CHECK(cudaFuncSetAttribute(stream, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    std::printf("grid,blocks,hold,page_size,trial,cta,first_full_cycles,steady_ii_cycles,total_cycles,kernel_ns\n");
    std::vector<unsigned long long> host(out_len);
    for (int trial = -1; trial < trials; ++trial) {  // trial -1 is an unrecorded warm-up
        flush<<<528, 256>>>(scratch, scratch_vec);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(e0));
        stream<<<grid, 384, kSmemBytes>>>(cache, d_table, blocks, hold, page_size, pages_per_cta, out);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (trial < 0) continue;
        CHECK(cudaMemcpy(host.data(), out, out_len * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        for (int c = 0; c < grid; ++c) {
            const unsigned long long* r = host.data() + static_cast<long long>(c) * (2 * kMaxBlocks + 2);
            const unsigned long long t0 = r[0];
            const unsigned long long first = r[1] - t0;
            const double steady = double(r[blocks] - r[2]) / double(blocks - 2);
            const unsigned long long total = r[1 + kMaxBlocks + blocks - 1] - t0;
            std::printf("%d,%d,%d,%d,%d,%d,%llu,%.2f,%llu,%.0f\n", grid, blocks, hold, page_size, trial, c,
                        first, steady, total, ms * 1e6);
        }
    }
    return 0;
}
