// Reuse the existing isolated CUB/division/rsqrt/publication device function.
// The renamed legacy host entry is retained verbatim, but never executed.
#define main baseline_block_reduce_host_entry
#include "block_reduce_latency.cu"
#undef main
#include <map>

__global__ void warm_small_collective_clock() {
    unsigned long long tick = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick));
    const unsigned long long stop = tick + 250000000ULL;
    while (tick < stop) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick));
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s THREADS GRID SAMPLES HIDDEN\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int threads = parse_positive_integer(argv[1], "threads");
    const int grid = parse_positive_integer(argv[2], "grid");
    const int samples = parse_positive_integer(argv[3], "samples");
    const int hidden = parse_positive_integer(argv[4], "hidden");
    if (!((threads == 16 && hidden == 128) || (threads == 256 && hidden == 4096))) {
        std::fprintf(stderr, "expected H128/block16 or H4096/block256 control\n");
        return EXIT_FAILURE;
    }
    int device = -1;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.minor != 0 || properties.multiProcessorCount != 132) {
        std::fprintf(stderr, "expected132-SM sm90 device\n");
        return EXIT_FAILURE;
    }
    cudaFuncAttributes resource{};
    int resident = 0;
    CHECK_CUDA(cudaFuncGetAttributes(&resource, reduce_rsqrt_broadcast));
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
        reduce_rsqrt_broadcast, threads, 0));
    std::fprintf(stderr, "RESOURCE {\"registers_per_thread\":%d,\"static_shared_bytes\":%zu,"
        "\"maximum_threads\":%d,\"resident_ctas_per_sm\":%d,\"threads\":%d,\"hidden\":%d}\n",
        resource.numRegs, resource.sharedSizeBytes, resource.maxThreadsPerBlock, resident, threads, hidden);

    const std::size_t records = static_cast<std::size_t>(samples) * grid;
    std::uint64_t *begin_device = nullptr, *end_device = nullptr;
    float* value_device = nullptr;
    unsigned int* sm_device = nullptr;
    CHECK_CUDA(cudaMalloc(&begin_device, records * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&end_device, records * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(&value_device, records * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sm_device, records * sizeof(unsigned int)));
    warm_small_collective_clock<<<132, 128>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int warm = 0; warm < 64; ++warm) {
        reduce_rsqrt_broadcast<<<grid, threads>>>(nullptr, 0, nullptr, nullptr,
            kEpsilon, grid, hidden, begin_device, end_device, value_device, sm_device, 0);
    }
    for (int sample = 0; sample < samples; ++sample) {
        reduce_rsqrt_broadcast<<<grid, threads>>>(nullptr, 0, nullptr, nullptr,
            kEpsilon, grid, hidden, begin_device, end_device, value_device, sm_device, sample);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<std::uint64_t> begins(records), ends(records);
    std::vector<float> values(records);
    std::vector<unsigned int> sms(records);
    CHECK_CUDA(cudaMemcpy(begins.data(), begin_device, records * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ends.data(), end_device, records * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(values.data(), value_device, records * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sms.data(), sm_device, records * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    double total = 0;
    for (int thread = 0; thread < threads; ++thread) total += 1.0 + (thread & 7) * 0.125;
    const float expected = static_cast<float>(1.0 / std::sqrt(total / hidden + kEpsilon));
    for (std::size_t index = 0; index < records; ++index) {
        if (ends[index] <= begins[index] || !std::isfinite(values[index])
                || std::abs(values[index] - expected) > 2e-6f) {
            std::fprintf(stderr, "invalid timer/numeric record%zu\n", index);
            return EXIT_FAILURE;
        }
    }
    std::printf("sample,smid,ctas,median_block_cycles,min_begin,max_end,span_cycles,hidden,threads,grid,numerical_validation\n");
    for (int sample = 0; sample < samples; ++sample) {
        std::map<unsigned int, std::vector<std::size_t>> groups;
        for (int block = 0; block < grid; ++block) {
            const std::size_t index = static_cast<std::size_t>(sample) * grid + block;
            groups[sms[index]].push_back(index);
        }
        if (groups.size() > 132) return EXIT_FAILURE;
        for (const auto& group : groups) {
            auto first = std::numeric_limits<std::uint64_t>::max();
            std::uint64_t last = 0;
            std::vector<std::uint64_t> body;
            for (auto index : group.second) {
                first = std::min(first, begins[index]);
                last = std::max(last, ends[index]);
                body.push_back(ends[index] - begins[index]);
            }
            std::printf("%d,%u,%zu,%llu,%llu,%llu,%llu,%d,%d,%d,pass\n", sample,
                group.first, group.second.size(), static_cast<unsigned long long>(median(body)),
                static_cast<unsigned long long>(first), static_cast<unsigned long long>(last),
                static_cast<unsigned long long>(last - first), hidden, threads, grid);
        }
    }
    CHECK_CUDA(cudaFree(begin_device));
    CHECK_CUDA(cudaFree(end_device));
    CHECK_CUDA(cudaFree(value_device));
    CHECK_CUDA(cudaFree(sm_device));
    return EXIT_SUCCESS;
}
