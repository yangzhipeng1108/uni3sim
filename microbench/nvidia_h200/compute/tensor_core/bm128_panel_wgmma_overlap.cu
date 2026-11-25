// Independent SM90 four-slot cp.async / SS WGMMA overlap probe.
// A 48-KiB panel is 128x64 and 256x64 BF16. This omits routing, GEMM output,
// expert weights, and all production duration. It measures a mechanism only.
#include "wgmma_ss_helper.cuh"

#include <algorithm>
#include <vector>

constexpr int kThreads = 256;
constexpr int kSmCount = 132;
constexpr int kGrid = 16 * kSmCount;
constexpr int kPanelBytes = (128 + 256) * 64 * 2;
constexpr int kStages = 4;
constexpr int kSharedBytes = kStages * kPanelBytes;
constexpr int kRounds = 256;
constexpr int kBackingBytes = 48 * 1024 * 1024;
constexpr int kBackingPanels = kBackingBytes / kPanelBytes;
static_assert(kPanelBytes == 49152 && kSharedBytes == 196608);

__device__ __forceinline__ void issue_panel(
        unsigned char* destination, const unsigned char* source) {
    for (int vector = threadIdx.x; vector < kPanelBytes / 16;
         vector += blockDim.x) {
        const auto shared_address = static_cast<unsigned>(
            __cvta_generic_to_shared(destination + vector * 16));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                     "r"(shared_address), "l"(source + vector * 16) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int MODE>
__global__ __launch_bounds__(kThreads, 1) void panel_probe(
        const unsigned char* __restrict__ input,
        unsigned long long* __restrict__ timestamps,
        float* __restrict__ sink,
        unsigned* __restrict__ checksums) {
    extern __shared__ __align__(1024) unsigned char shared[];
    const int wg = threadIdx.x / 128;
    float d[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) d[i] = 0.0f;
    if constexpr (MODE == 1) {
        for (int offset = threadIdx.x * 4; offset < kSharedBytes;
             offset += blockDim.x * 4) {
            *reinterpret_cast<unsigned*>(shared + offset) = 0x3f803f80u;
        }
        __syncthreads();
    } else {
        // Three prefetched panels leave one slot free for the next copy.
        // wait_group 1 retires the previous WGMMA before that slot is reused.
        for (int stage = 0; stage < kStages - 1; ++stage) {
            const int panel = (blockIdx.x * 17 + stage) % kBackingPanels;
            issue_panel(shared + stage * kPanelBytes,
                        input + panel * kPanelBytes);
        }
    }
    unsigned checksum = 0;
    const auto begin = clock64();
    for (int round = 0; round < kRounds; ++round) {
        const int slot = round % kStages;
        if constexpr (MODE != 1) {
            if (round < kRounds - (kStages - 1)) {
                asm volatile("cp.async.wait_group 2;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            __syncthreads();
        }
        const auto* panel = shared + slot * kPanelBytes;
        checksum += reinterpret_cast<const volatile unsigned*>(panel)[threadIdx.x];
        if constexpr (MODE != 0) {
            wg_fence();
            const auto base = static_cast<unsigned>(__cvta_generic_to_shared(panel));
            uint64_t a[4], b[4];
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                a[step] = make_desc(base + step * 32, 16, 1024);
                b[step] = make_desc(base + 16384 + step * 32, 16, 1024);
            }
            wgmma_ss_n256_group_wait1(d, a, b, round != 0);
            // wait_group 1 keeps only this round's group outstanding.
            // The next cp.async targets the slot consumed one round ago.
        }
        if constexpr (MODE != 1) {
            __syncthreads();
            const int next = round + kStages - 1;
            if (next < kRounds) {
                const int panel_index = (blockIdx.x * 17 + next) % kBackingPanels;
                issue_panel(shared + (next % kStages) * kPanelBytes,
                            input + panel_index * kPanelBytes);
            }
        }
    }
    if constexpr (MODE != 0) wg_wait0();
    const auto end = clock64();
    float value = 0.0f;
    if constexpr (MODE != 0) {
#pragma unroll
        for (int i = 0; i < 128; ++i) value += d[i];
    }
    if (threadIdx.x % 128 == 0) {
        const int index = blockIdx.x * 2 + wg;
        timestamps[index * 2] = begin;
        timestamps[index * 2 + 1] = end;
        sink[index] = value;
        checksums[index] = checksum;
    }
}

template <int MODE>
void run_mode(const char* name, const unsigned char* input,
              unsigned long long* timestamps, float* sink,
              unsigned* checksums) {
    CHECK(cudaFuncSetAttribute(panel_probe<MODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
    int occupancy = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, panel_probe<MODE>, kThreads, kSharedBytes));
    if (occupancy != 1) {
        std::fprintf(stderr, "expected one CTA per SM: %d\n", occupancy);
        std::exit(3);
    }
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    for (int sample = -2; sample < 7; ++sample) {
        CHECK(cudaEventRecord(start));
        panel_probe<MODE><<<kGrid, kThreads, kSharedBytes>>>(
            input, timestamps, sink, checksums);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaGetLastError());
        if (sample >= 0) {
            float milliseconds = 0;
            CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            samples.push_back(milliseconds);
        }
    }
    std::sort(samples.begin(), samples.end());
    std::vector<unsigned long long> host_times(kGrid * 4);
    CHECK(cudaMemcpy(host_times.data(), timestamps,
        host_times.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> cycles;
    for (int index = 0; index < kGrid * 2; ++index) {
        const auto begin = host_times[index * 2];
        const auto end = host_times[index * 2 + 1];
        if (end <= begin) std::exit(4);
        cycles.push_back(end - begin);
    }
    std::sort(cycles.begin(), cycles.end());
    unsigned checksum = 0;
    CHECK(cudaMemcpy(&checksum, checksums, sizeof(checksum), cudaMemcpyDeviceToHost));
    float value = 0;
    CHECK(cudaMemcpy(&value, sink, sizeof(value), cudaMemcpyDeviceToHost));
    std::printf("{\"mode\":\"%s\",\"grid\":%d,\"rounds\":%d,"
                "\"shared_bytes\":%d,\"checksum\":%u,\"sink\":%.9g,"
                "\"cta_cycles_p10_p50_p90\":[%llu,%llu,%llu],"
                "\"samples_ms\":[", name, kGrid, kRounds, kSharedBytes,
                checksum, value,
                cycles[cycles.size() / 10], cycles[cycles.size() / 2],
                cycles[cycles.size() * 9 / 10]);
    for (int i = 0; i < 7; ++i)
        std::printf("%s%.9f", i ? "," : "", samples[i]);
    std::printf("]}\n");
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

int main() {
    cudaDeviceProp props{};
    CHECK(cudaGetDeviceProperties(&props, 0));
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices));
    if (devices != 1 || std::strcmp(props.name, "NVIDIA H200") != 0 ||
        props.major != 9 || props.multiProcessorCount != kSmCount) {
        std::fprintf(stderr, "requires one 132-SM H200\n");
        return 2;
    }
    unsigned char* input = nullptr;
    unsigned long long* timestamps = nullptr;
    float* sink = nullptr;
    unsigned* checksums = nullptr;
    CHECK(cudaMalloc(&input, kBackingBytes));
    CHECK(cudaMalloc(&timestamps, kGrid * 4 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&sink, kGrid * 2 * sizeof(float)));
    CHECK(cudaMalloc(&checksums, kGrid * 2 * sizeof(unsigned)));
    CHECK(cudaMemset(input, 0x3f, kBackingBytes));
    std::printf("{\"device\":\"%s\",\"sm_count\":%d,"
                "\"backing_bytes\":%d,\"logical_panel_bytes\":%llu}\n",
                props.name, props.multiProcessorCount, kBackingBytes,
                static_cast<unsigned long long>(kGrid) * kRounds * kPanelBytes);
    run_mode<0>("copy_only", input, timestamps, sink, checksums);
    run_mode<1>("wgmma_only", input, timestamps, sink, checksums);
    run_mode<2>("copy_plus_wgmma", input, timestamps, sink, checksums);
    CHECK(cudaFree(checksums));
    CHECK(cudaFree(sink));
    CHECK(cudaFree(timestamps));
    CHECK(cudaFree(input));
    return 0;
}
