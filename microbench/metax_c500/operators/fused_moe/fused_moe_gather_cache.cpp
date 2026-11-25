/*
 * MetaX C500 grouped-gather cache-residency microbenchmark.
 *
 * This is not a FusedMoE implementation.  It removes routing, activation and
 * matrix output and retains only the cache mechanism that production grouped
 * GEMM needs:
 * random activation rows are reread by N-fanout CTAs while an optional,
 * separately addressed weight-like stream competes for the shared 8-MiB L2.
 * The launch uses the production BM/BK/BN/GROUP_SIZE_M signatures, but every
 * cache input remains independently controllable.
 * An optional register-only BF16 MMA stream supplies the production tile's
 * eight wave-level 16x16x16 instructions per K step.  Comparing the exact
 * same address stream with MMA disabled/enabled isolates whether matrix issue
 * changes cache residency; it does not execute a complete FusedMoE kernel.
 *
 * The BM64 production launch has a frozen two-request L2 multiplicity for
 * both operands.  The scalar cache probe already reproduces A's measured
 * two-request behavior, while its contiguous B loop otherwise produces only
 * one request stream.  POLLUTER_LOAD_MULTIPLICITY therefore issues the second
 * B stream explicitly from the same unique panels.  The report checks the
 * resulting L2 bytes against the frozen A/B request contract before any
 * mechanism parameter can be released.
 *
 * One process executes exactly one configuration.  Normal event timing is
 * outside the mcProfiler marker.  In profile mode the marker contains exactly
 * one kernel invocation so L2-source bytes, downstream bytes and hit counters
 * have an unambiguous identity.  mcProfiler replay duration is diagnostic and
 * must never be used as a production timing target.
 * A pre-marker flush is also diagnostic only: mcProfiler may replay the
 * marked kernel without replaying that flush, so it cannot prove cold-state
 * or compulsory traffic.
 *
 * Build in the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch=native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       -o fused_moe_gather_cache fused_moe_gather_cache.cpp
 */

#include <mc_runtime.h>

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unistd.h>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

constexpr int kBlockThreads = 256;
constexpr int kElementBytes = sizeof(std::uint16_t);
constexpr int kIndexBytes = sizeof(std::int32_t);
constexpr int kMinimumDynamicSharedBytes = 16 * 1024;
constexpr int kMaximumDynamicSharedBytes = 64 * 1024;
constexpr std::size_t kCacheFlushBytes = 64ULL * 1024ULL * 1024ULL;
constexpr int kCacheFlushBlocks = 832;
constexpr std::uint32_t kPermutationMultiplier = 2654435761U;
constexpr std::uint32_t kPermutationOffset = 2246822519U;

struct Arguments {
    const char* signature;
    int k;
    int source_rows;
    int group_count;
    int n_fanout;
    int weight_share_groups;
    int dynamic_shared_bytes;
    int include_polluter;
    int polluter_load_multiplicity;
    int include_mma;
    const char* route_pattern;
    int timed_iterations;
    int attach_wait_seconds;
    int profile;
    const char* profile_setup;
};

static long long checked_product(
        long long left, long long right, const char* label) {
    if (left <= 0 || right <= 0
            || left > std::numeric_limits<long long>::max() / right) {
        std::fprintf(stderr, "%s product is invalid or overflows\n", label);
        std::exit(EXIT_FAILURE);
    }
    return left * right;
}

static int parse_positive_int(const char* text, const char* label) {
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0'
            || value <= 0 || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a positive integer: %s\n",
                     label, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

static int parse_binary_flag(const char* text, const char* label) {
    if (std::strcmp(text, "0") == 0) return 0;
    if (std::strcmp(text, "1") == 0) return 1;
    std::fprintf(stderr, "%s must be 0 or 1\n", label);
    std::exit(EXIT_FAILURE);
}

static bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static Arguments parse_arguments(int argc, char** argv) {
    if (argc != 16) {
        std::fprintf(
            stderr,
            "usage: %s SIGNATURE K SOURCE_ROWS GROUP_COUNT N_FANOUT "
            "WEIGHT_SHARE_GROUPS DYNAMIC_SHARED_BYTES INCLUDE_POLLUTER "
            "POLLUTER_LOAD_MULTIPLICITY INCLUDE_MMA ROUTE_PATTERN TIMED_ITERATIONS "
            "ATTACH_WAIT_SECONDS PROFILE PROFILE_SETUP\n",
            argv[0]);
        std::exit(EXIT_FAILURE);
    }
    Arguments arguments = {
        argv[1],
        parse_positive_int(argv[2], "K"),
        parse_positive_int(argv[3], "SOURCE_ROWS"),
        parse_positive_int(argv[4], "GROUP_COUNT"),
        parse_positive_int(argv[5], "N_FANOUT"),
        parse_positive_int(argv[6], "WEIGHT_SHARE_GROUPS"),
        parse_positive_int(argv[7], "DYNAMIC_SHARED_BYTES"),
        parse_binary_flag(argv[8], "INCLUDE_POLLUTER"),
        parse_positive_int(argv[9], "POLLUTER_LOAD_MULTIPLICITY"),
        parse_binary_flag(argv[10], "INCLUDE_MMA"),
        argv[11],
        parse_positive_int(argv[12], "TIMED_ITERATIONS"),
        parse_positive_int(argv[13], "ATTACH_WAIT_SECONDS"),
        parse_binary_flag(argv[14], "PROFILE"),
        argv[15],
    };
    if (!is_power_of_two(arguments.source_rows)) {
        std::fprintf(stderr, "SOURCE_ROWS must be a power of two\n");
        std::exit(EXIT_FAILURE);
    }
    if (std::strcmp(arguments.route_pattern, "periodic") != 0
            && std::strcmp(
                arguments.route_pattern, "expert_contiguous") != 0) {
        std::fprintf(
            stderr, "ROUTE_PATTERN must be periodic or expert_contiguous\n");
        std::exit(EXIT_FAILURE);
    }
    if (arguments.dynamic_shared_bytes < kMinimumDynamicSharedBytes
            || arguments.dynamic_shared_bytes > kMaximumDynamicSharedBytes
            || arguments.dynamic_shared_bytes % kMinimumDynamicSharedBytes) {
        std::fprintf(
            stderr,
            "DYNAMIC_SHARED_BYTES must be 16384, 32768, 49152, or 65536\n");
        std::exit(EXIT_FAILURE);
    }
    if (std::strcmp(arguments.profile_setup, "prewarm") != 0
            && std::strcmp(
                arguments.profile_setup,
                "prelaunch_flush_diagnostic") != 0) {
        std::fprintf(
            stderr,
            "PROFILE_SETUP must be prewarm or prelaunch_flush_diagnostic\n");
        std::exit(EXIT_FAILURE);
    }
    return arguments;
}

typedef __NATIVE_VECTOR__(2, int32_t) v2i;
typedef __NATIVE_VECTOR__(4, float) v4f;

__device__ __forceinline__ v4f mma_bf16(v2i a, v2i b, v4f c) {
    return __builtin_mxc_mma_16x16x16bf16(a, b, c);
}

__global__ void c500_cache_flush(
        const std::uint16_t* __restrict__ flush_data,
        std::size_t element_count,
        std::uint32_t* __restrict__ flush_sink) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t stride = (
        static_cast<std::size_t>(gridDim.x) * blockDim.x);
    std::uint32_t checksum = static_cast<std::uint32_t>(threadIdx.x + 1);
    for (std::size_t index = thread; index < element_count; index += stride) {
        checksum += static_cast<std::uint32_t>(flush_data[index]);
    }
    flush_sink[thread] = checksum;
}

template<int BM, int BK, int BN, int GROUP_SIZE_M>
__global__ void c500_grouped_gather_cache(
        const std::uint16_t* __restrict__ activation,
        const std::int32_t* __restrict__ row_indices,
        const std::uint16_t* __restrict__ polluter,
        int k, int n_fanout, int weight_share_groups,
        int include_polluter, int polluter_load_multiplicity, int include_mma,
        std::uint32_t* __restrict__ sink) {
    extern __shared__ unsigned char shared_storage[];
    std::int32_t* shared_rows =
        reinterpret_cast<std::int32_t*>(shared_storage);
    std::uint16_t* activation_tile =
        reinterpret_cast<std::uint16_t*>(shared_rows + BM);
    std::uint16_t* polluter_tile = activation_tile + BM * BK;

    int pid = static_cast<int>(blockIdx.x);
    int ctas_per_group = GROUP_SIZE_M * n_fanout;
    int group = pid / ctas_per_group;
    int within_group = pid % ctas_per_group;
    int pid_m = within_group % GROUP_SIZE_M;
    int pid_n = within_group / GROUP_SIZE_M;
    int route_start = (group * GROUP_SIZE_M + pid_m) * BM;
    int tid = static_cast<int>(threadIdx.x);
    if (tid < BM) shared_rows[tid] = row_indices[route_start + tid];
    __syncthreads();

    std::uint32_t checksum = static_cast<std::uint32_t>(tid + 1);
    v2i mma_a = {0x3b803c00, 0x3a803b00};
    v2i mma_b = {0x3b003a80, 0x3c003b80};
    v4f accumulator0 = {0.125f, 0.25f, 0.5f, 1.0f};
    v4f accumulator1 = {0.25f, 0.5f, 1.0f, 0.125f};
    v4f accumulator2 = {0.5f, 1.0f, 0.125f, 0.25f};
    v4f accumulator3 = {1.0f, 0.125f, 0.25f, 0.5f};
    int k_steps = k / BK;
    int weight_group = group / weight_share_groups;
    long long weight_panel = (
        static_cast<long long>(weight_group) * n_fanout + pid_n);
    long long weight_panel_elements =
        static_cast<long long>(k) * BN;
    for (int k_step = 0; k_step < k_steps; ++k_step) {
        int k_offset = k_step * BK;
        for (int index = tid; index < BM * BK; index += blockDim.x) {
            int row = index / BK;
            int column = index % BK;
            int source_row = shared_rows[row];
            std::uint16_t value = activation[
                static_cast<long long>(source_row) * k
                + k_offset + column];
            activation_tile[index] = value;
            checksum += static_cast<std::uint32_t>(value);
        }
        __syncthreads();
        if (include_polluter) {
            for (int replay = 0;
                 replay < polluter_load_multiplicity; ++replay) {
                for (int index = tid; index < BK * BN; index += blockDim.x) {
                    int permuted_index = (index + replay * 64) % (BK * BN);
                    int row = permuted_index / BN;
                    int column = permuted_index % BN;
                    std::uint16_t value = polluter[
                        weight_panel * weight_panel_elements
                        + static_cast<long long>(k_offset + row) * BN
                        + column];
                    polluter_tile[permuted_index] = value;
                    checksum += static_cast<std::uint32_t>(value);
                }
                __syncthreads();
            }
        }
        if (include_mma) {
            accumulator0 = mma_bf16(mma_a, mma_b, accumulator0);
            accumulator1 = mma_bf16(mma_a, mma_b, accumulator1);
            accumulator2 = mma_bf16(mma_a, mma_b, accumulator2);
            accumulator3 = mma_bf16(mma_a, mma_b, accumulator3);
            accumulator0 = mma_bf16(mma_a, mma_b, accumulator0);
            accumulator1 = mma_bf16(mma_a, mma_b, accumulator1);
            accumulator2 = mma_bf16(mma_a, mma_b, accumulator2);
            accumulator3 = mma_bf16(mma_a, mma_b, accumulator3);
        }
        checksum += activation_tile[tid % (BM * BK)];
        if (include_polluter) {
            checksum += polluter_tile[tid % (BK * BN)];
        }
        __syncthreads();
    }
    if (include_mma) {
        checksum += static_cast<std::uint32_t>(
            accumulator0.x + accumulator1.y
            + accumulator2.z + accumulator3.w);
    }
    sink[static_cast<long long>(pid) * blockDim.x + tid] = checksum;
}

template<int BM, int BK, int BN, int GROUP_SIZE_M>
static void run_signature(const Arguments& arguments) {
    if (arguments.k % BK) {
        std::fprintf(stderr, "K=%d must be divisible by BK=%d\n",
                     arguments.k, BK);
        std::exit(EXIT_FAILURE);
    }
    if (arguments.group_count % arguments.weight_share_groups) {
        std::fprintf(
            stderr,
            "GROUP_COUNT must be divisible by WEIGHT_SHARE_GROUPS\n");
        std::exit(EXIT_FAILURE);
    }
    constexpr int required_shared_bytes = (
        BM * kIndexBytes + (BM * BK + BK * BN) * kElementBytes);
    if (arguments.dynamic_shared_bytes < required_shared_bytes) {
        std::fprintf(
            stderr,
            "dynamic shared bytes %d are below the required %d\n",
            arguments.dynamic_shared_bytes, required_shared_bytes);
        std::exit(EXIT_FAILURE);
    }

    long long route_rows = checked_product(
        checked_product(arguments.group_count, GROUP_SIZE_M, "route rows"),
        BM, "route rows");
    long long grid = checked_product(
        checked_product(arguments.group_count, GROUP_SIZE_M, "grid"),
        arguments.n_fanout, "grid");
    if (grid > INT_MAX) {
        std::fprintf(stderr, "grid exceeds the one-dimensional launch limit\n");
        std::exit(EXIT_FAILURE);
    }
    long long activation_elements = checked_product(
        arguments.source_rows, arguments.k, "activation elements");
    long long weight_groups = (
        arguments.group_count / arguments.weight_share_groups);
    long long polluter_panels = checked_product(
        weight_groups, arguments.n_fanout, "polluter panels");
    long long polluter_elements = arguments.include_polluter
        ? checked_product(
            checked_product(
                polluter_panels, arguments.k, "polluter elements"),
            BN, "polluter elements")
        : 1;
    long long sink_elements = checked_product(
        grid, kBlockThreads, "sink elements");

    std::uint16_t* device_activation = nullptr;
    std::uint16_t* device_polluter = nullptr;
    std::int32_t* device_indices = nullptr;
    std::uint32_t* device_sink = nullptr;
    CHECK(mcMalloc(
        &device_activation,
        static_cast<std::size_t>(activation_elements) * kElementBytes));
    CHECK(mcMalloc(
        &device_polluter,
        static_cast<std::size_t>(polluter_elements) * kElementBytes));
    CHECK(mcMalloc(
        &device_indices,
        static_cast<std::size_t>(route_rows) * kIndexBytes));
    CHECK(mcMalloc(
        &device_sink,
        static_cast<std::size_t>(sink_elements) * sizeof(std::uint32_t)));
    CHECK(mcMemset(
        device_activation, 1,
        static_cast<std::size_t>(activation_elements) * kElementBytes));
    CHECK(mcMemset(
        device_polluter, 1,
        static_cast<std::size_t>(polluter_elements) * kElementBytes));
    CHECK(mcMemset(
        device_sink, 0,
        static_cast<std::size_t>(sink_elements) * sizeof(std::uint32_t)));

    std::int32_t* host_indices = static_cast<std::int32_t*>(
        std::malloc(static_cast<std::size_t>(route_rows) * kIndexBytes));
    if (host_indices == nullptr) {
        std::fprintf(stderr, "host row-index allocation failed\n");
        std::exit(EXIT_FAILURE);
    }
    std::uint32_t source_mask =
        static_cast<std::uint32_t>(arguments.source_rows - 1);
    if (std::strcmp(arguments.route_pattern, "periodic") == 0) {
        for (long long route = 0; route < route_rows; ++route) {
            std::uint32_t permuted = (
                static_cast<std::uint32_t>(route) * kPermutationMultiplier
                + kPermutationOffset) & source_mask;
            host_indices[route] = static_cast<std::int32_t>(permuted);
        }
    } else {
        long long rows_per_expert = checked_product(
            GROUP_SIZE_M, BM, "rows per expert group");
        rows_per_expert = checked_product(
            rows_per_expert, arguments.weight_share_groups,
            "rows per expert");
        if (rows_per_expert > arguments.source_rows
                || arguments.source_rows % rows_per_expert
                || !is_power_of_two(static_cast<int>(rows_per_expert))) {
            std::fprintf(
                stderr,
                "expert_contiguous requires a power-of-two rows-per-expert "
                "that divides SOURCE_ROWS\n");
            std::exit(EXIT_FAILURE);
        }
        const std::uint32_t rows_per_expert_mask =
            static_cast<std::uint32_t>(rows_per_expert - 1);
        const long long source_partitions =
            arguments.source_rows / rows_per_expert;
        for (long long route = 0; route < route_rows; ++route) {
            const long long expert = route / rows_per_expert;
            const std::uint32_t local = static_cast<std::uint32_t>(
                route % rows_per_expert);
            const std::uint32_t permuted_local = (
                local * kPermutationMultiplier + kPermutationOffset)
                & rows_per_expert_mask;
            const long long source_partition = expert % source_partitions;
            host_indices[route] = static_cast<std::int32_t>(
                source_partition * rows_per_expert + permuted_local);
        }
    }
    CHECK(mcMemcpy(
        device_indices, host_indices,
        static_cast<std::size_t>(route_rows) * kIndexBytes,
        mcMemcpyHostToDevice));

    auto launch = [&]() {
        c500_grouped_gather_cache<BM, BK, BN, GROUP_SIZE_M>
            <<<static_cast<int>(grid), kBlockThreads,
               arguments.dynamic_shared_bytes>>>(
                device_activation, device_indices, device_polluter,
                arguments.k, arguments.n_fanout,
                arguments.weight_share_groups,
                arguments.include_polluter,
                arguments.polluter_load_multiplicity,
                arguments.include_mma, device_sink);
        CHECK(mcGetLastError());
    };

    const void* kernel = reinterpret_cast<const void*>(
        c500_grouped_gather_cache<BM, BK, BN, GROUP_SIZE_M>);
    mcFuncAttributes attributes;
    CHECK(mcFuncGetAttributes(&attributes, kernel));
    int resident_blocks = 0;
    CHECK(mcOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks, kernel, kBlockThreads,
        arguments.dynamic_shared_bytes));
    const bool requires_one_resident_cta = (
        std::strcmp(arguments.signature, "bm32_bk128_bn64_gsm1") == 0);
    if (requires_one_resident_cta && resident_blocks != 1) {
        std::fprintf(
            stderr,
            "resource contract requires one resident CTA/AP, got %d "
            "(registers=%d static_shared=%zu dynamic_shared=%d)\n",
            resident_blocks, attributes.numRegs,
            static_cast<std::size_t>(attributes.sharedSizeBytes),
            arguments.dynamic_shared_bytes);
        std::exit(EXIT_FAILURE);
    }
    std::fprintf(
        stderr,
        "resource,signature=%s,registers=%d,static_shared=%zu,"
        "dynamic_shared=%d,resident_ctas_per_ap=%d\n",
        arguments.signature, attributes.numRegs,
        static_cast<std::size_t>(attributes.sharedSizeBytes),
        arguments.dynamic_shared_bytes, resident_blocks);

    for (int warmup = 0; warmup < 3; ++warmup) launch();
    CHECK(mcDeviceSynchronize());
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));
    CHECK(mcEventRecord(start));
    for (int iteration = 0;
         iteration < arguments.timed_iterations; ++iteration) {
        launch();
    }
    CHECK(mcEventRecord(end));
    CHECK(mcDeviceSynchronize());
    float total_ms = 0.0F;
    CHECK(mcEventElapsedTime(&total_ms, start, end));
    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    if (!(total_ms > 0.0F)) {
        std::fprintf(stderr, "event timing must be positive\n");
        std::exit(EXIT_FAILURE);
    }

    std::uint16_t* device_flush_data = nullptr;
    std::uint32_t* device_flush_sink = nullptr;
    if (arguments.profile
            && std::strcmp(
                arguments.profile_setup,
                "prelaunch_flush_diagnostic") == 0) {
        constexpr std::size_t flush_elements = (
            kCacheFlushBytes / sizeof(std::uint16_t));
        constexpr std::size_t flush_sink_elements = (
            static_cast<std::size_t>(kCacheFlushBlocks) * kBlockThreads);
        CHECK(mcMalloc(&device_flush_data, kCacheFlushBytes));
        CHECK(mcMalloc(
            &device_flush_sink,
            flush_sink_elements * sizeof(std::uint32_t)));
        CHECK(mcMemset(device_flush_data, 1, kCacheFlushBytes));
        CHECK(mcMemset(
            device_flush_sink, 0,
            flush_sink_elements * sizeof(std::uint32_t)));
    }

    if (arguments.profile) {
        sleep(static_cast<unsigned int>(arguments.attach_wait_seconds));
        if (std::strcmp(arguments.profile_setup, "prewarm") == 0) {
            launch();
            CHECK(mcDeviceSynchronize());
        } else {
            constexpr std::size_t flush_elements = (
                kCacheFlushBytes / sizeof(std::uint16_t));
            c500_cache_flush<<<kCacheFlushBlocks, kBlockThreads>>>(
                device_flush_data, flush_elements, device_flush_sink);
            CHECK(mcGetLastError());
            CHECK(mcDeviceSynchronize());
        }
        CHECK(mcProfilerStart());
        launch();
        CHECK(mcDeviceSynchronize());
        CHECK(mcProfilerStop());
    }

    std::uint32_t first_sink = 0;
    std::uint32_t last_sink = 0;
    CHECK(mcMemcpy(
        &first_sink, device_sink, sizeof(first_sink), mcMemcpyDeviceToHost));
    CHECK(mcMemcpy(
        &last_sink, device_sink + sink_elements - 1,
        sizeof(last_sink), mcMemcpyDeviceToHost));
    if (first_sink == 0 || last_sink == 0) {
        std::fprintf(stderr, "sink validation failed\n");
        std::exit(EXIT_FAILURE);
    }

    long long logical_a_read_bytes = checked_product(
        checked_product(grid, BM, "logical A bytes"),
        checked_product(arguments.k, kElementBytes, "logical A bytes"),
        "logical A bytes");
    long long logical_b_base_read_bytes = arguments.include_polluter
        ? checked_product(
            checked_product(grid, arguments.k, "logical B bytes"),
            BN * kElementBytes, "logical B bytes")
        : 0;
    long long logical_b_read_bytes = arguments.include_polluter
        ? checked_product(
            logical_b_base_read_bytes,
            arguments.polluter_load_multiplicity,
            "logical B bytes including multiplicity")
        : 0;
    long long unique_a_rows = route_rows < arguments.source_rows
        ? route_rows : arguments.source_rows;
    long long unique_a_bytes = checked_product(
        checked_product(unique_a_rows, arguments.k, "unique A bytes"),
        kElementBytes, "unique A bytes");
    long long unique_b_bytes = arguments.include_polluter
        ? checked_product(
            checked_product(polluter_panels, arguments.k, "unique B bytes"),
            BN * kElementBytes, "unique B bytes")
        : 0;
    long long index_read_bytes = checked_product(
        checked_product(grid, BM, "index read bytes"),
        kIndexBytes, "index read bytes");
    long long sink_write_bytes = checked_product(
        sink_elements, static_cast<long long>(sizeof(std::uint32_t)),
        "sink write bytes");
    double per_iteration_ms = (
        static_cast<double>(total_ms) / arguments.timed_iterations);

    std::printf(
        "metadata,5,metax_c500,grouped_gather_cache,"
        "production_or_evaluation_duration_consumed=0\n");
    std::printf(
        "sample_header,signature,bm,bk,bn,group_size_m,k,source_rows,"
        "group_count,n_fanout,weight_share_groups,grid,block,"
        "dynamic_shared_bytes,include_polluter,polluter_load_multiplicity,"
        "include_mma,route_pattern,timed_iterations,event_cache_state,profile_setup,"
        "logical_a_read_bytes,logical_b_base_read_bytes,logical_b_read_bytes,"
        "unique_a_bytes,unique_b_bytes,index_read_bytes,sink_write_bytes,"
        "event_ms_per_iteration,first_sink,last_sink\n");
    std::printf(
        "sample,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%d,%d,%d,%d,%d,%s,%d,"
        "steady_state,%s,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.9f,%08x,%08x\n",
        arguments.signature, BM, BK, BN, GROUP_SIZE_M, arguments.k,
        arguments.source_rows, arguments.group_count, arguments.n_fanout,
        arguments.weight_share_groups, grid, kBlockThreads,
        arguments.dynamic_shared_bytes, arguments.include_polluter,
        arguments.polluter_load_multiplicity, arguments.include_mma,
        arguments.route_pattern, arguments.timed_iterations,
        arguments.profile_setup, logical_a_read_bytes,
        logical_b_base_read_bytes, logical_b_read_bytes,
        unique_a_bytes, unique_b_bytes,
        index_read_bytes, sink_write_bytes, per_iteration_ms,
        first_sink, last_sink);

    std::free(host_indices);
    CHECK(mcFree(device_activation));
    CHECK(mcFree(device_polluter));
    CHECK(mcFree(device_indices));
    CHECK(mcFree(device_sink));
    if (device_flush_data != nullptr) CHECK(mcFree(device_flush_data));
    if (device_flush_sink != nullptr) CHECK(mcFree(device_flush_sink));
}

int main(int argc, char** argv) {
    Arguments arguments = parse_arguments(argc, argv);
    if (std::strcmp(arguments.signature, "bm16_bk64_bn32_gsm1") == 0) {
        if (arguments.polluter_load_multiplicity != 1) {
            std::fprintf(stderr, "BM16 polluter multiplicity must be one\n");
            return EXIT_FAILURE;
        }
        run_signature<16, 64, 32, 1>(arguments);
    } else if (
            std::strcmp(arguments.signature, "bm32_bk128_bn64_gsm1") == 0) {
        // The deployed vLLM-metax BF16 default issues one operand stream for
        // the BM32 program.  This mechanism probe keeps that static fact;
        // it does not replay the complete FusedMoE operation.
        if (arguments.polluter_load_multiplicity != 1) {
            std::fprintf(stderr, "BM32 polluter multiplicity must be one\n");
            return EXIT_FAILURE;
        }
        run_signature<32, 128, 64, 1>(arguments);
    } else if (
            std::strcmp(arguments.signature, "bm64_bk32_bn64_gsm8") == 0) {
        if (arguments.polluter_load_multiplicity != 2) {
            std::fprintf(stderr, "BM64 polluter multiplicity must be two\n");
            return EXIT_FAILURE;
        }
        run_signature<64, 32, 64, 8>(arguments);
    } else {
        std::fprintf(stderr, "unknown signature: %s\n", arguments.signature);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
