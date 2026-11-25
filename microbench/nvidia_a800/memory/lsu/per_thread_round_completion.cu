/*
 * A800 CTA-wide dependent-round completion microbenchmark.
 *
 * One round is the unit a source loop body executes: every one of ``N``
 * threads issues exactly one global load, and the consumer of that load is
 * what the next round's address depends on.  Within a round the loads are
 * independent and fully coalesced; between rounds they are strictly
 * serialized.  The measured quantity is therefore the completion time of one
 * CTA-wide round, which is what a per-round pipeline stage must be priced
 * against.
 *
 * The access width ``W`` is swept over 2, 4, 8 and 16 bytes per thread with
 * the round's instruction count held fixed at one load per thread, so a
 * byte-proportional per-round cost and a request-latency-floored per-round
 * cost make opposite predictions:
 *
 *   byte-proportional : t(16 B) == 8 * t(2 B)
 *   latency-floored   : t(16 B) ~= t(2 B) while the round stays below the
 *                       SM's byte capacity
 *
 * Two residency regimes are declared.  ``l2`` walks a 16 MiB window -- larger
 * than the 192 KiB per-SM L1, smaller than the 40 MiB device L2 -- and is
 * primed by an untimed walk.  ``l1`` walks a 64 KiB window that a single CTA
 * retains in its own L1, and ``mixed`` gives one stream each -- the shape of
 * a second source pass that rereads its own row while pulling a shared row
 * it has not touched.  Loads use the default cache policy and the same weak
 * ``LDG.E`` encoding the production source compiles to.
 *
 * Timing reads ``%globaltimer`` inside the CTA, so a dynamic SM clock cannot
 * corrupt the reported nanoseconds.  No production kernel, tensor or duration
 * is consumed: the probe loads a zero-filled buffer and reports only its own
 * elapsed time.
 *
 * Build:
 *   /usr/local/cuda-12.2/bin/nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo \
 *     -o per_thread_round_completion per_thread_round_completion.cu
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                           \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                     __FILE__, __LINE__, cudaGetErrorString(error));      \
        std::exit(EXIT_FAILURE);                                          \
    }                                                                     \
} while (0)

constexpr std::size_t kL2WindowBytes = 16ULL * 1024 * 1024;
constexpr std::size_t kL1WindowBytes = 64ULL * 1024;
constexpr std::size_t kAllocationBytes = 64ULL * 1024 * 1024;
constexpr int kRounds = 512;

__device__ __forceinline__ unsigned long long read_globaltimer() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// One dependent load per thread at the declared width.  These are ordinary
// typed C++ loads, so ``ptxas`` emits the same weak ``LDG.E`` encoding the
// production source compiles to.  An earlier revision of this probe used
// inline ``ld.global.ca`` asm, which emits ``LDG.E.STRONG.SM``; that scoped
// encoding orders the two loads of a round against each other and made a
// two-stream round cost the sum rather than the maximum of its requests.
// The returned integer is zero for the zero-filled buffer, but ``ptxas``
// cannot prove that, so it keeps the next round's address dependent on this
// round's data.
template <int Width>
struct RoundWord;

template <> struct RoundWord<2> {
    using type = unsigned short;
    __device__ static int reduce(type v) { return static_cast<int>(v); }
};
template <> struct RoundWord<4> {
    using type = unsigned int;
    __device__ static int reduce(type v) { return static_cast<int>(v); }
};
template <> struct RoundWord<8> {
    using type = uint2;
    __device__ static int reduce(type v) {
        return static_cast<int>(v.x | v.y);
    }
};
template <> struct RoundWord<16> {
    using type = uint4;
    __device__ static int reduce(type v) {
        return static_cast<int>(v.x | v.y | v.z | v.w);
    }
};

// ``rms_norm_kernel`` declares its input and weight ``const __restrict__``
// and the production binary issues ``LDG.E.U16.CONSTANT`` for them, so a
// plain typed read through a ``__restrict__`` pointer reproduces the
// production encoding exactly.  ``fused_add_rms_norm_kernel`` updates input
// and residual in place and issues plain ``LDG.E.128`` for those two
// streams; nvcc only withholds the read-only path when the kernel also
// stores through the pointer, so that encoding needs a round that carries
// its store as well and is outside this probe's declared domain.
template <int Width>
__device__ __forceinline__ typename RoundWord<Width>::type load_word(
        const char* address) {
    using Word = typename RoundWord<Width>::type;
    return *reinterpret_cast<const Word*>(address);
}

template <int Width>
__device__ __forceinline__ void store_word(
        char* address, typename RoundWord<Width>::type value) {
    using Word = typename RoundWord<Width>::type;
    *reinterpret_cast<Word*>(address) = value;
}

// ``rounds`` CTA-wide rounds over a window of ``window_bytes``.  Round ``r``
// places thread ``t`` at element ``r * blockDim.x + t`` of the window, so the
// round is coalesced and consecutive rounds touch disjoint lines -- the same
// grid-stride shape a source loop body has.  ``carry`` threads the previous
// round's loaded value into the next round's address.
template <int Width, int Streams, int Stores>
__global__ void dependent_rounds(char* __restrict__ buffer_a,
                                 const char* __restrict__ buffer_b,
                                 unsigned int mask_a,
                                 unsigned int mask_b,
                                 int prime_rounds,
                                 int rounds,
                                 unsigned long long* elapsed_ns,
                                 int* sink) {
    // ``element_mask`` is ``window_elements - 1`` for a power-of-two window,
    // so the round's address arithmetic is one add and one and -- no integer
    // division is allowed inside the timed loop.
    using Word = typename RoundWord<Width>::type;
    const unsigned int threads = blockDim.x;
    unsigned int cursor = threadIdx.x;
    int carry = 0;
    // Prime the complete window of every declared stream before timing, so
    // each timed round re-reads lines this CTA has already installed.
    #pragma unroll 1
    for (int round = 0; round < prime_rounds; ++round) {
        const unsigned int base = cursor + static_cast<unsigned int>(carry);
        const std::size_t offset_a =
            static_cast<std::size_t>(base & mask_a) * Width;
        const Word word_a = load_word<Width>(buffer_a + offset_a);
        carry = RoundWord<Width>::reduce(word_a);
        if (Streams == 2) {
            const Word word_b = load_word<Width>(
                buffer_b + static_cast<std::size_t>(base & mask_b) * Width);
            carry |= RoundWord<Width>::reduce(word_b);
            if (Stores == 1) {
                store_word<Width>(buffer_a + offset_a, word_b);
            }
        }
        cursor += threads;
    }
    __syncthreads();
    const unsigned long long start = read_globaltimer();
    #pragma unroll 1
    for (int round = 0; round < rounds; ++round) {
        // Both stream addresses depend only on the previous round, so the
        // two requests of one round may overlap exactly as the source
        // program's two independent loads do.
        const unsigned int base = cursor + static_cast<unsigned int>(carry);
        const std::size_t offset_a =
            static_cast<std::size_t>(base & mask_a) * Width;
        const Word word_a = load_word<Width>(buffer_a + offset_a);
        const int a = RoundWord<Width>::reduce(word_a);
        int b = 0;
        if (Streams == 2) {
            const Word word_b = load_word<Width>(
                buffer_b + static_cast<std::size_t>(base & mask_b) * Width);
            b = RoundWord<Width>::reduce(word_b);
            if (Stores == 1) {
                // Storing the other stream's word keeps the buffer zero while
                // denying ptxas the "stores back what it read" proof.
                store_word<Width>(buffer_a + offset_a, word_b);
            }
        }
        carry = a | b;
        cursor += threads;
    }
    const unsigned long long end = read_globaltimer();
    if (threadIdx.x == 0) {
        *elapsed_ns = end - start;
    }
    if (carry != 0) {
        // Never taken for the zero-filled buffer; keeps every load live.
        sink[threadIdx.x] = carry;
    }
}

using Launcher = void (*)(char*, const char*, unsigned int,
                          unsigned int, int, int, unsigned long long*,
                          int*, int, int);

template <int Width, int Streams, int Stores>
void launch(char* buffer_a, const char* buffer_b, unsigned int mask_a,
            unsigned int mask_b, int prime_rounds, int rounds,
            unsigned long long* elapsed_ns, int* sink, int threads,
            int grid) {
    dependent_rounds<Width, Streams, Stores><<<grid, threads>>>(
        buffer_a, buffer_b, mask_a, mask_b, prime_rounds, rounds,
        elapsed_ns, sink);
}

template <int Width>
Launcher select_streams(int streams, int stores) {
    if (streams == 1) {
        return stores == 0 ? launch<Width, 1, 0> : launch<Width, 1, 1>;
    }
    return stores == 0 ? launch<Width, 2, 0> : launch<Width, 2, 1>;
}

Launcher select(int width, int streams, int stores) {
    if (width == 2) return select_streams<2>(streams, stores);
    if (width == 4) return select_streams<4>(streams, stores);
    if (width == 8) return select_streams<8>(streams, stores);
    if (width == 16) return select_streams<16>(streams, stores);
    std::fprintf(stderr, "undeclared width %d\n", width);
    std::exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
    const int widths[] = {2, 4, 8, 16};
    // Every width a production NORM source round actually issues, so no
    // production round has to be interpolated between measured cells:
    // rms_norm (scalar, 2 B) issues 512 and 1024; fused_add_rms_norm
    // (vec8, 16 B) issues 64, 128, 256, 320, 512, 640 and 1024.  32 is kept
    // as a single-warp control.
    const int thread_counts[] = {32, 64, 128, 256, 320, 512, 640, 1024};
    // ``mixed`` gives stream A the L1 window and stream B the L2 window: the
    // shape of a second source pass that rereads its own row while pulling a
    // shared row it has not touched.
    const char* residencies[] = {"l2", "l1", "mixed"};
    const int stream_counts[] = {1, 2};
    const int store_counts[] = {0, 1};
    const int trials = 5;

    if (argc == 2 && std::strcmp(argv[1], "--describe") == 0) {
        std::printf("probe=per_thread_round_completion arch=sm_80\n");
        std::printf("rounds=%d trials=%d grid=1 prime=full_window_untimed\n",
                    kRounds, trials);
        std::printf("l2_window_bytes=%zu l1_window_bytes=%zu\n",
                    kL2WindowBytes, kL1WindowBytes);
        std::printf("widths_bytes=2,4,8,16 streams=1,2 load_encoding=weak_ldg_e\n");
        std::printf("residencies=l2,l1,mixed\n");
        std::printf("threads_per_cta=32,64,128,256,320,512,640,1024\n");
        return 0;
    }

    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (static_cast<std::size_t>(properties.l2CacheSize) <= kL2WindowBytes) {
        std::fprintf(stderr, "L2 capacity %d must exceed window %zu\n",
                     properties.l2CacheSize, kL2WindowBytes);
        return EXIT_FAILURE;
    }

    // Stream A is written when the round carries a store, so it is declared
    // non-const throughout; stream B is read-only.
    char* buffer_a = nullptr;
    char* buffer_b = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer_a, kAllocationBytes));
    CUDA_CHECK(cudaMalloc(&buffer_b, kAllocationBytes));
    CUDA_CHECK(cudaMemset(buffer_a, 0, kAllocationBytes));
    CUDA_CHECK(cudaMemset(buffer_b, 0, kAllocationBytes));
    unsigned long long* elapsed_ns = nullptr;
    CUDA_CHECK(cudaMalloc(&elapsed_ns, sizeof(unsigned long long)));
    int* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, 1024 * sizeof(int)));

    std::printf("residency,streams,stores,width_bytes,threads,rounds,"
                "prime_rounds,timed_wraps_a,trial,elapsed_ns,round_ns,"
                "load_bytes_per_round,store_bytes_per_round\n");

    // Trial-first: every trial walks the complete cell list once, so a
    // contention burst from another tenant on the shared L2 perturbs all
    // cells alike instead of corrupting whichever cell held the device at
    // that moment.  The per-cell spread the reference gate checks is then a
    // real stability statistic rather than an accident of ordering.
    for (int trial = 0; trial < trials; ++trial)
    for (const char* residency : residencies) {
        const bool mixed = std::strcmp(residency, "mixed") == 0;
        const std::size_t window_a =
            (std::strcmp(residency, "l2") == 0) ? kL2WindowBytes
                                                : kL1WindowBytes;
        const std::size_t window_b =
            (std::strcmp(residency, "l1") == 0) ? kL1WindowBytes
                                                : kL2WindowBytes;
        for (int streams : stream_counts) {
            if (mixed && streams == 1) {
                // ``mixed`` is only defined for two streams.
                continue;
            }
            for (int stores : store_counts) {
                if (stores == 1 && streams == 1) {
                    // A store is only declared for the two-stream round: the
                    // stored value comes from the other stream.
                    continue;
                }
                for (int width : widths) {
                    const std::size_t elements_a =
                        window_a / static_cast<std::size_t>(width);
                    const std::size_t elements_b =
                        window_b / static_cast<std::size_t>(width);
                    if ((elements_a & (elements_a - 1)) != 0
                            || (elements_b & (elements_b - 1)) != 0) {
                        std::fprintf(stderr,
                                     "windows must be powers of two\n");
                        return EXIT_FAILURE;
                    }
                    const unsigned int mask_a =
                        static_cast<unsigned int>(elements_a - 1);
                    const unsigned int mask_b =
                        static_cast<unsigned int>(elements_b - 1);
                    for (int threads : thread_counts) {
                        Launcher launcher = select(width, streams, stores);
                        const std::size_t widest =
                            std::max(elements_a, elements_b);
                        const int prime_rounds = static_cast<int>(
                            (widest + threads - 1) / threads);
                        const double wraps_a =
                            static_cast<double>(kRounds)
                            * threads / elements_a;
                        {
                            launcher(buffer_a, buffer_b, mask_a, mask_b,
                                     prime_rounds, kRounds, elapsed_ns,
                                     sink, threads, 1);
                            CUDA_CHECK(cudaDeviceSynchronize());
                            unsigned long long host_ns = 0;
                            CUDA_CHECK(cudaMemcpy(&host_ns, elapsed_ns,
                                                  sizeof(host_ns),
                                                  cudaMemcpyDeviceToHost));
                            const double round_ns =
                                static_cast<double>(host_ns) / kRounds;
                            const double bytes_per_round =
                                static_cast<double>(threads) * width
                                * streams;
                            std::printf(
                                "%s,%d,%d,%d,%d,%d,%d,%.2f,%d,%llu,%.3f,"
                                "%.0f,%.0f\n",
                                residency, streams, stores, width, threads,
                                kRounds, prime_rounds, wraps_a, trial,
                                host_ns, round_ns, bytes_per_round,
                                static_cast<double>(threads) * width
                                * stores);
                        }
                    }
                }
            }
        }
    }

    CUDA_CHECK(cudaFree(sink));
    CUDA_CHECK(cudaFree(elapsed_ns));
    CUDA_CHECK(cudaFree(buffer_b));
    CUDA_CHECK(cudaFree(buffer_a));
    return 0;
}
