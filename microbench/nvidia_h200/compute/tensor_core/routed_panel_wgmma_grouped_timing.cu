// Independent 4/8-round panel/WGMMA timing probe with source-ordered M/N reuse.
// The target W2 has twelve K rounds. This does no router, activation, result
// store, or full GEMM; phase timings remain mechanism evidence only.
#include "wgmma_ss_helper.cuh"

#include <algorithm>
#include <vector>

constexpr int kThreads = 256;
constexpr int kSmCount = 132;
constexpr int kTokens = 524288;
constexpr int kExperts = 128;
constexpr int kK = 768;
constexpr int kN = 2048;
constexpr int kBM = 128;
constexpr int kBN = 256;
constexpr int kBK = 64;
constexpr int kMTiles = 4223;
constexpr int kNTiles = kN / kBN;
constexpr int kGroupM = 16;
constexpr int kGrid = kMTiles * kNTiles;
constexpr int kStages = 4;
constexpr int kPanelBytes = (kBM + kBN) * kBK * 2;
constexpr int kSharedBytes = kStages * kPanelBytes;
static_assert(kPanelBytes == 49152 && kSharedBytes == 196608);

__device__ __forceinline__ void cp_vector(
        unsigned char* destination, const unsigned char* source,
        int source_bytes) {
    const unsigned address = static_cast<unsigned>(
        __cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(address), "l"(source), "r"(source_bytes) : "memory");
}

__device__ __forceinline__ void issue_panel(
        unsigned char* panel, const unsigned char* activation,
        const unsigned char* weights, const int* rows, int round) {
    const int group_width = kGroupM * kNTiles;
    const int group = blockIdx.x / group_width;
    const int first_m = group * kGroupM;
    const int group_m = min(kMTiles - first_m, kGroupM);
    const int group_pid = blockIdx.x % group_width;
    const int pid_m = first_m + group_pid % group_m;
    const int n_tile = group_pid / group_m;
    const int expert = min(pid_m / 33, kExperts - 1);
    for (int vector = threadIdx.x; vector < kBM * kBK * 2 / 16;
         vector += kThreads) {
        const int row_position = pid_m * kBM + vector / (kBK * 2 / 16);
        const int row = rows[row_position];
        const int k_vector = vector % (kBK * 2 / 16);
        const std::size_t offset = std::size_t(row) * kK * 2 +
            round * kBK * 2 + k_vector * 16;
        cp_vector(panel + vector * 16, activation + offset,
                  row_position < kTokens ? 16 : 0);
    }
    for (int vector = threadIdx.x; vector < kBN * kBK * 2 / 16;
         vector += kThreads) {
        const int n_row = n_tile * kBN + vector / (kBK * 2 / 16);
        const int k_vector = vector % (kBK * 2 / 16);
        const std::size_t offset =
            (std::size_t(expert) * kN + n_row) * kK * 2 +
            round * kBK * 2 + k_vector * 16;
        cp_vector(panel + kBM * kBK * 2 + vector * 16,
                  weights + offset, 16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int MODE, int ROUNDS>
__global__ __launch_bounds__(kThreads, 1) void routed_panel_probe(
        const unsigned char* __restrict__ activation,
        const unsigned char* __restrict__ weights,
        const int* __restrict__ rows,
        unsigned long long* __restrict__ timestamps,
        float* __restrict__ sink,
        unsigned* __restrict__ checksums) {
    extern __shared__ __align__(1024) unsigned char shared[];
    const int wg = threadIdx.x / 128;
    const auto begin = clock64();
    float d[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) d[i] = 0.0f;
    if constexpr (MODE == 1) {
        for (int offset = threadIdx.x * 4; offset < kSharedBytes;
             offset += blockDim.x * 4)
            *reinterpret_cast<unsigned*>(shared + offset) = 0x3f803f80u;
        __syncthreads();
    } else {
        for (int stage = 0; stage < kStages - 1; ++stage)
            issue_panel(shared + stage * kPanelBytes,
                        activation, weights, rows, stage);
    }
    unsigned checksum = 0;
    const auto after_prefill_issue = clock64();
    for (int round = 0; round < ROUNDS; ++round) {
        const int slot = round % kStages;
        if constexpr (MODE != 1) {
            if (round < ROUNDS - (kStages - 1))
                asm volatile("cp.async.wait_group 2;" ::: "memory");
            else
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
        const auto* panel = shared + slot * kPanelBytes;
        checksum += reinterpret_cast<const volatile unsigned*>(panel)[threadIdx.x];
        if constexpr (MODE != 0) {
            wg_fence();
            const auto base = static_cast<unsigned>(
                __cvta_generic_to_shared(panel));
            uint64_t a[4], b[4];
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                a[step] = make_desc(base + step * 32, 16, 1024);
                b[step] = make_desc(base + 16384 + step * 32, 16, 1024);
            }
            wgmma_ss_n256_group_wait1(d, a, b, round != 0);
        }
        if constexpr (MODE != 1) {
            __syncthreads();
            const int next = round + kStages - 1;
            if (next < ROUNDS)
                issue_panel(shared + (next % kStages) * kPanelBytes,
                            activation, weights, rows, next);
        }
    }
    if constexpr (MODE != 0) wg_wait0();
    const auto after_loop_wait = clock64();
    float value = 0.0f;
    if constexpr (MODE != 0) {
#pragma unroll
        for (int i = 0; i < 128; ++i) value += d[i];
    }
    if (threadIdx.x % 128 == 0) {
        const int index = blockIdx.x * 2 + wg;
        timestamps[index * 3] = begin;
        timestamps[index * 3 + 1] = after_prefill_issue;
        timestamps[index * 3 + 2] = after_loop_wait;
        sink[index] = value;
        checksums[index] = checksum;
    }
}

template <int MODE, int ROUNDS>
void run_mode(const char* name,
              const unsigned char* activation, const unsigned char* weights,
              const int* rows, unsigned long long* timestamps,
              float* sink, unsigned* checksums, bool single) {
    CHECK(cudaFuncSetAttribute(routed_panel_probe<MODE, ROUNDS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
    int occupancy = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, routed_panel_probe<MODE, ROUNDS>, kThreads, kSharedBytes));
    if (occupancy != 1) {
        std::fprintf(stderr, "expected one resident CTA/SM, got %d\n", occupancy);
        std::exit(3);
    }
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    const int first = single ? 0 : -2;
    const int last = single ? 1 : 7;
    for (int sample = first; sample < last; ++sample) {
        CHECK(cudaEventRecord(start));
        routed_panel_probe<MODE, ROUNDS><<<kGrid, kThreads, kSharedBytes>>>(
            activation, weights, rows, timestamps, sink, checksums);
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
    std::vector<unsigned long long> host_times(kGrid * 6);
    CHECK(cudaMemcpy(host_times.data(), timestamps,
        host_times.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> prefill_cycles, loop_cycles, full_cycles;
    for (int index = 0; index < kGrid * 2; ++index) {
        const auto start = host_times[index * 3];
        const auto middle = host_times[index * 3 + 1];
        const auto end = host_times[index * 3 + 2];
        if (middle <= start || end <= middle) std::exit(4);
        prefill_cycles.push_back(middle - start);
        loop_cycles.push_back(end - middle);
        full_cycles.push_back(end - start);
    }
    std::sort(prefill_cycles.begin(), prefill_cycles.end());
    std::sort(loop_cycles.begin(), loop_cycles.end());
    std::sort(full_cycles.begin(), full_cycles.end());
    unsigned checksum = 0;
    CHECK(cudaMemcpy(&checksum, checksums, sizeof(checksum), cudaMemcpyDeviceToHost));
    float value = 0;
    CHECK(cudaMemcpy(&value, sink, sizeof(value), cudaMemcpyDeviceToHost));
    std::printf("{\"mode\":\"%s\",\"rounds\":%d,\"grid\":%d,"
                "\"shared_bytes\":%d,\"checksum\":%u,\"sink\":%.9g,"
                "\"cta_prefill_issue_cycles_p10_p50_p90\":[%llu,%llu,%llu],"
                "\"cta_loop_wait_cycles_p10_p50_p90\":[%llu,%llu,%llu],"
                "\"cta_full_cycles_p10_p50_p90\":[%llu,%llu,%llu],"
                "\"samples_ms\":[", name, ROUNDS, kGrid, kSharedBytes,
                checksum, value,
                prefill_cycles[prefill_cycles.size() / 10],
                prefill_cycles[prefill_cycles.size() / 2],
                prefill_cycles[prefill_cycles.size() * 9 / 10],
                loop_cycles[loop_cycles.size() / 10],
                loop_cycles[loop_cycles.size() / 2],
                loop_cycles[loop_cycles.size() * 9 / 10],
                full_cycles[full_cycles.size() / 10],
                full_cycles[full_cycles.size() / 2],
                full_cycles[full_cycles.size() * 9 / 10]);
    for (std::size_t i = 0; i < samples.size(); ++i)
        std::printf("%s%.9f", i ? "," : "", samples[i]);
    std::printf("]}\n");
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

template <int ROUNDS>
void run_round_set(const unsigned char* activation, const unsigned char* weights,
                   const int* rows, unsigned long long* timestamps,
                   float* sink, unsigned* checksums, bool single,
                   int single_mode) {
    if (!single || single_mode == 0)
        run_mode<0, ROUNDS>("gather_copy", activation, weights, rows,
                            timestamps, sink, checksums, single);
    if (!single || single_mode == 1)
        run_mode<1, ROUNDS>("wgmma_only", activation, weights, rows,
                            timestamps, sink, checksums, single);
    if (!single || single_mode == 2)
        run_mode<2, ROUNDS>("gather_plus_wgmma", activation, weights, rows,
                            timestamps, sink, checksums, single);
}

int main(int argc, char** argv) {
    const bool single = argc == 4 && std::strcmp(argv[1], "--single") == 0;
    if (argc != 1 && !single) return 2;
    const int single_mode = single ? std::atoi(argv[2]) : -1;
    const int single_rounds = single ? std::atoi(argv[3]) : 0;
    if (single && (single_mode < 0 || single_mode > 2 ||
                   (single_rounds != 4 && single_rounds != 8))) return 2;
    cudaDeviceProp props{};
    CHECK(cudaGetDeviceProperties(&props, 0));
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices));
    if (devices != 1 || std::strcmp(props.name, "NVIDIA H200") != 0 ||
        props.major != 9 || props.multiProcessorCount != kSmCount) return 2;
    std::vector<int> host_rows(kMTiles * kBM);
    for (std::size_t i = 0; i < host_rows.size(); ++i)
        host_rows[i] = int((i * 1103515245ull + 12345ull) % kTokens);
    unsigned char *activation = nullptr, *weights = nullptr;
    int* rows = nullptr;
    unsigned long long* timestamps = nullptr;
    float* sink = nullptr;
    unsigned* checksums = nullptr;
    CHECK(cudaMalloc(&activation, std::size_t(kTokens) * kK * 2));
    CHECK(cudaMalloc(&weights, std::size_t(kExperts) * kN * kK * 2));
    CHECK(cudaMalloc(&rows, host_rows.size() * sizeof(int)));
    CHECK(cudaMalloc(&timestamps, kGrid * 6 * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&sink, kGrid * 2 * sizeof(float)));
    CHECK(cudaMalloc(&checksums, kGrid * 2 * sizeof(unsigned)));
    CHECK(cudaMemset(activation, 0x3f, std::size_t(kTokens) * kK * 2));
    CHECK(cudaMemset(weights, 0x3f, std::size_t(kExperts) * kN * kK * 2));
    CHECK(cudaMemcpy(rows, host_rows.data(), host_rows.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
    std::printf("{\"device\":\"%s\",\"sm_count\":%d,"
                "\"synthetic_rows\":%zu,\"k_full_rounds\":%d}\n",
                props.name, props.multiProcessorCount, host_rows.size(),
                kK / kBK);
    if (!single || single_rounds == 4)
        run_round_set<4>(activation, weights, rows, timestamps, sink,
                         checksums, single, single_mode);
    if (!single || single_rounds == 8)
        run_round_set<8>(activation, weights, rows, timestamps, sink,
                         checksums, single, single_mode);
    CHECK(cudaFree(checksums));
    CHECK(cudaFree(sink));
    CHECK(cudaFree(timestamps));
    CHECK(cudaFree(rows));
    CHECK(cudaFree(weights));
    CHECK(cudaFree(activation));
    return 0;
}
