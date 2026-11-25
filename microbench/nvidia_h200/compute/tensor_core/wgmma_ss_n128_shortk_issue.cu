// Independent SM90 BF16 SS m64n128 short-K issue/wait probe. Two CTAs per SM with 98304 B shared each. Shared operands are
// initialized once; there is no global operand stream, routing, or GEMM output.
// The runner generates the helper from the pinned FA3 descriptor ABI. Four
// WGMMA instructions share one inline PTX block, so ptxas sees one group.
#include "wgmma_ss_helper.cuh"

#include <algorithm>
#include <initializer_list>

constexpr int kProbeSmemBytes = 98304;
constexpr int kProbeThreads = 256;

template <int WAIT_GROUP>
__global__ void __launch_bounds__(256, 2) wait_probe(
        int groups, int repeats, unsigned long long* timestamps, float* sink) {
    extern __shared__ __align__(1024) unsigned char smem[];
    for (int offset = threadIdx.x * 4; offset < kProbeSmemBytes;
         offset += blockDim.x * 4) {
        *reinterpret_cast<unsigned*>(smem + offset) = 0x3f803f80u;
    }
    __syncthreads();
    const int wg = threadIdx.x / 128;
    const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    float d[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) d[i] = 0.0f;
    const unsigned long long start = clock64();
    wg_fence();
    for (int r = 0; r < repeats; ++r) {
        for (int group = 0; group < groups; ++group) {
            uint64_t a[4], b[4];
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int k = (group * 4 + step) % 32;
                const uint32_t a_offset = (k % 4) * 32 + (k / 4) * 4096;
                const uint32_t b_offset = (k % 4) * 32 + (k / 4) * 4096;
                a[step] = make_desc(base + a_offset, 16, 1024);
                b[step] = make_desc(base + 49152 + b_offset, 16, 1024);
            }
            if constexpr (WAIT_GROUP == 0)
                wgmma_ss_n128_group_wait0(d, a, b, (r | group) != 0);
            else
                wgmma_ss_n128_group_wait1(d, a, b, (r | group) != 0);
        }
    }
    wg_wait0();
    const unsigned long long stop = clock64();
    float check = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; ++i) check += d[i];
    if (threadIdx.x % 128 == 0) {
        const int index = (blockIdx.x * 2 + wg) * 2;
        timestamps[index] = start;
        timestamps[index + 1] = stop;
        sink[blockIdx.x * 2 + wg] = check;
    }
}

void run_probe(int sm_count, int wait_group, int groups, int repeats,
               unsigned long long* times, float* sink) {
    const int grid = sm_count * 16;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    std::vector<unsigned long long> host_times(static_cast<size_t>(grid) * 4);
    std::printf("{\"wait_group\":%d,\"groups\":%d,\"repeats\":%d,"
                "\"grid\":%d,\"samples_ms\":[", wait_group, groups, repeats, grid);
    for (int sample = -2; sample < 7; ++sample) {
        CHECK(cudaEventRecord(start));
        if (wait_group == 0) {
            wait_probe<0><<<grid, kProbeThreads, kProbeSmemBytes>>>(groups, repeats, times, sink);
        } else {
            wait_probe<1><<<grid, kProbeThreads, kProbeSmemBytes>>>(groups, repeats, times, sink);
        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaGetLastError());
        if (sample >= 0) {
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            std::printf("%s%.9f", sample ? "," : "", ms);
        }
    }
    std::printf("],\"cta_cycles\":[");
    CHECK(cudaMemcpy(host_times.data(), times, host_times.size() * sizeof(unsigned long long),
                     cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> cycles;
    cycles.reserve(static_cast<size_t>(grid) * 2);
    for (int cta = 0; cta < grid * 2; ++cta) {
        const auto begin = host_times[cta * 2];
        const auto end = host_times[cta * 2 + 1];
        if (end <= begin) {
            std::fprintf(stderr, "nonpositive CTA clock interval\n");
            std::exit(3);
        }
        cycles.push_back(end - begin);
    }
    std::sort(cycles.begin(), cycles.end());
    std::printf("%llu,%llu,%llu]}\n", cycles[cycles.size() / 10],
                cycles[cycles.size() / 2], cycles[cycles.size() * 9 / 10]);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

int main() {
    cudaDeviceProp props{};
    CHECK(cudaGetDeviceProperties(&props, 0));
    if (props.major != 9 || props.multiProcessorCount != 132) {
        std::fprintf(stderr, "probe requires one 132-SM H200\n");
        return 2;
    }
    for (int mode : {0, 1}) {
        int occupancy = 0;
        if (mode == 0) {
            CHECK(cudaFuncSetAttribute(wait_probe<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kProbeSmemBytes));
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, wait_probe<0>,
                  kProbeThreads, kProbeSmemBytes));
        } else {
            CHECK(cudaFuncSetAttribute(wait_probe<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kProbeSmemBytes));
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, wait_probe<1>,
                  kProbeThreads, kProbeSmemBytes));
        }
        if (occupancy != 2) {
            std::fprintf(stderr, "expected two resident CTAs per SM\n");
            return 4;
        }
    }
    const int grid = props.multiProcessorCount * 16;
    unsigned long long* times = nullptr;
    float* sink = nullptr;
    CHECK(cudaMalloc(&times, static_cast<size_t>(grid) * 4 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&sink, static_cast<size_t>(grid) * 2 * sizeof(float)));
    std::printf("{\"device\":\"%s\",\"sm_count\":%d,\"shared_bytes\":%d,"
                "\"threads\":%d,\"resident_ctas_per_sm\":2}\n",
                props.name, props.multiProcessorCount, kProbeSmemBytes, kProbeThreads);
    for (int groups : {4, 12}) {
        for (int mode : {0, 1}) run_probe(props.multiProcessorCount, mode, groups, 8, times, sink);
    }
    CHECK(cudaFree(sink));
    CHECK(cudaFree(times));
    return 0;
}
