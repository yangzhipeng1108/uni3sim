/*
 * MetaX C500 vector copy load/compute overlap benchmark.
 *
 * The Forward model prices the source-bound vectorized KV copy as a strictly
 * serial chain: submit, then the global read, then the read's first-request
 * latency, then the integer address/control work, then the store.  That
 * summation is only correct if an AP cannot advance one CTA's address
 * arithmetic while another CTA's loads are still in flight.  Whether it can
 * is the question this benchmark answers, and it answers it without
 * consuming any production duration.
 *
 * The kernel is the copy's own shape and nothing else: every thread performs
 * one aligned 16-byte load, a variable number of dependent integer
 * operations on the loaded words, and one aligned 16-byte store.  Traffic is
 * held exactly constant while the integer work is swept, so the duration
 * curve separates the two hypotheses outright:
 *
 *   serial      duration rises from the first added operation, with slope
 *               equal to the integer throughput
 *   overlapped  duration stays flat while the integer work fits inside the
 *               memory time, then rises once it exceeds it
 *
 * The grid is swept across the resident window so the answer is reported
 * both where one CTA per AP is resident and where the queue must refill,
 * which is the regime the production launch occupies.
 *
 * There is no production operator, no KV cache, no paging and no attention
 * arithmetic: the reported quantity is a launch duration against an integer
 * operation count at fixed traffic.
 *
 * Build:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 vector_copy_compute_overlap.cpp \
 *       -o vector_copy_compute_overlap
 */

#include <mc_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

/* The source-bound KV copy launches 512 threads and one aligned vec8 pack
 * per active thread; this probe matches both. */
constexpr int kThreads = 512;

/* ``dependent`` selects which of the two integer chains is measured.  A
 * dependent chain consumes the loaded value and therefore cannot start until
 * the load returns; an independent chain is derived from the CTA's own index
 * arithmetic and can be issued while the load is still in flight.  The source
 * KV copy's address/control work is of the second kind, so the two must be
 * distinguished before either is placed on a serial critical path.
 *
 * The two chains execute the *identical* instruction sequence and differ
 * only in their seed -- the loaded word against the flat index.  An earlier
 * version gave them different bodies, which confounded "cheaper" with
 * "better hidden" and settled nothing; matching the instruction counts is
 * what makes the comparison mean anything. */
__global__ void copy_with_integer_work(const uint4* __restrict__ input,
                                       uint4* __restrict__ output,
                                       long long packs, int integer_rounds,
                                       int dependent) {
    const long long stride =
        static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long index = blockIdx.x * blockDim.x + threadIdx.x;
         index < packs; index += stride) {
        uint4 value = input[index];
        /* Same instruction sequence either way; only the seed differs.  The
         * dependent seed is the loaded word, so the chain cannot begin until
         * the load returns.  The independent seed is the flat index, exactly
         * like the source's slot and page address arithmetic, so the chain
         * is issuable while the load is still in flight. */
        unsigned int chain = dependent
            ? value.x
            : static_cast<unsigned int>(index);
        for (int round = 0; round < integer_rounds; ++round) {
            chain = chain * 1664525U + 1013904223U;
            chain ^= chain >> 13;
        }
        /* Folded in at the end so neither chain can be deleted. */
        value.x += chain & 1U;
        output[index] = value;
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
                     "usage: %s <trials> <warmup_seconds> <grid_csv> "
                     "<integer_rounds_csv>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int trials = std::atoi(argv[1]);
    if (trials < 1 || trials > 64) {
        std::fprintf(stderr, "trials must be within 1..64\n");
        return EXIT_FAILURE;
    }
    const int warmup_seconds = std::atoi(argv[2]);
    if (warmup_seconds < 0 || warmup_seconds > 600) {
        std::fprintf(stderr, "warmup_seconds must be within 0..600\n");
        return EXIT_FAILURE;
    }
    std::vector<int> grids;
    for (const char* cursor = argv[3]; *cursor != '\0';) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value <= 0 || value > 2000000) {
            std::fprintf(stderr, "grids must be within 1..2000000\n");
            return EXIT_FAILURE;
        }
        grids.push_back(static_cast<int>(value));
        cursor = (*end == ',') ? end + 1 : end;
    }
    std::vector<int> rounds;
    for (const char* cursor = argv[4]; *cursor != '\0';) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value < 0 || value > 4096) {
            std::fprintf(stderr, "integer rounds must be within 0..4096\n");
            return EXIT_FAILURE;
        }
        rounds.push_back(static_cast<int>(value));
        cursor = (*end == ',') ? end + 1 : end;
    }

    mcDeviceProp_t properties{};
    CHECK(mcGetDeviceProperties(&properties, 0));
    std::printf("device,%s\n", properties.name);
    std::printf("multiprocessors,%d\n", properties.multiProcessorCount);
    std::printf("threads_per_cta,%d\n", kThreads);
    std::printf("grid_ctas,integer_rounds,dependent,packs,trial_index,"
                "duration_ms\n");

    /* One aligned vec8 pack per thread of the largest grid, so traffic is
     * identical for every integer-round setting at a given grid. */
    const int max_grid = *std::max_element(grids.begin(), grids.end());
    const long long max_packs =
        static_cast<long long>(max_grid) * kThreads;
    uint4* input = nullptr;
    uint4* output = nullptr;
    CHECK(mcMalloc(&input, sizeof(uint4) * max_packs));
    CHECK(mcMalloc(&output, sizeof(uint4) * max_packs));
    CHECK(mcMemset(input, 0x5A, sizeof(uint4) * max_packs));
    CHECK(mcMemset(output, 0x00, sizeof(uint4) * max_packs));

    mcEvent_t start;
    mcEvent_t stop;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&stop));
    const auto warmup_deadline =
        std::chrono::steady_clock::now()
        + std::chrono::seconds(warmup_seconds);
    while (std::chrono::steady_clock::now() < warmup_deadline) {
        for (int burst = 0; burst < 32; ++burst) {
            copy_with_integer_work<<<max_grid, kThreads>>>(
                input, output, max_packs, 8, 1);
        }
        CHECK(mcDeviceSynchronize());
    }
    /* Trial-outer so every (grid, rounds) cell is sampled once per round and
     * sees the same clock state; a cell-outer sweep measures this device's
     * clock ramp instead. */
    for (int trial = 0; trial < trials; ++trial) {
        for (const int grid : grids) {
            const long long packs =
                static_cast<long long>(grid) * kThreads;
            for (const int integer_rounds : rounds) {
                for (int dependent = 1; dependent >= 0; --dependent) {
                    CHECK(mcDeviceSynchronize());
                    CHECK(mcEventRecord(start, 0));
                    copy_with_integer_work<<<grid, kThreads>>>(
                        input, output, packs, integer_rounds, dependent);
                    CHECK(mcEventRecord(stop, 0));
                    CHECK(mcEventSynchronize(stop));
                    float milliseconds = 0.0f;
                    CHECK(mcEventElapsedTime(&milliseconds, start, stop));
                    std::printf("%d,%d,%d,%lld,%d,%.6f\n", grid,
                                integer_rounds, dependent, packs, trial,
                                static_cast<double>(milliseconds));
                }
            }
        }
    }
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(stop));
    CHECK(mcFree(input));
    CHECK(mcFree(output));
    return EXIT_SUCCESS;
}
