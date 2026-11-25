/*
 * MetaX C500 register-only and memory-coupled frequency benchmark.
 *
 * Each block brackets the same register-resident instruction stream with the
 * AP-local clock64 counter and the 100-MHz get_realtime counter. Their ratio
 * measures the XCORE clock during that block without sampling production
 * kernels. Four instruction mixes distinguish a generic issue response from
 * SFU-specific behavior; silu_mix has 240 add and 16 EXP/RCP instructions per
 * 256-instruction repeat, matching only the instruction-class proportion of
 * the audited SiLU program rather than executing the production operator.
 * Three additional bodies add exactly two coalesced float4 loads and one
 * coalesced float4 store per thread.  They test whether an otherwise identical
 * short Vector program enters a different XCORE frequency state when coupled
 * to the global-memory path.  No production operator or production duration is
 * used.
 */

#include <mc_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));         \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kExpectedApCount = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kExpectedBlockSize = 512;
constexpr int kMaximumGrid = 13312;
constexpr int kPrewarmGrid = 416;
constexpr int kPrewarmProgramRepeats = 256;
constexpr int kPrewarmLaunches = 8;
constexpr int kSelfWarmupLaunches = 3;
constexpr unsigned long long kGlobalTimerHz = 100000000ULL;

enum class Pattern : int {
    kAdd = 0,
    kExp = 1,
    kRcp = 2,
    kSiluMix = 3,
    kMemoryOnly = 4,
    kMemoryAdd = 5,
    kMemorySiluMix = 6,
};

struct PlanRow {
    std::string sample_id;
    std::string condition;
    Pattern pattern;
    int trial;
    int order_position;
    int grid;
    int program_repeats;
    int idle_ms;
    int prewarm;
};

struct RegisterState {
    float x0;
    float x1;
    float x2;
    float x3;
    float x4;
    float x5;
    float x6;
    float x7;
};

__device__ __forceinline__ float dependent_add(float value) {
    float output;
    const float addend = 0.0000152587890625f;
    asm volatile("add_f32 %0, %1, %2"
                 : "=r"(output) : "r"(value), "r"(addend));
    return output;
}

__device__ __forceinline__ float dependent_exp(float value) {
    float output;
    asm volatile("exp_f32 %0, neg(%1)"
                 : "=r"(output) : "r"(value));
    return output;
}

__device__ __forceinline__ float dependent_rcp(float value) {
    float output;
    asm volatile("rcp_f32 %0, %1" : "=r"(output) : "r"(value));
    return output;
}

__device__ __forceinline__ void add_round(RegisterState* state) {
    state->x0 = dependent_add(state->x0);
    state->x1 = dependent_add(state->x1);
    state->x2 = dependent_add(state->x2);
    state->x3 = dependent_add(state->x3);
    state->x4 = dependent_add(state->x4);
    state->x5 = dependent_add(state->x5);
    state->x6 = dependent_add(state->x6);
    state->x7 = dependent_add(state->x7);
}

__device__ __forceinline__ void exp_round(RegisterState* state) {
    state->x0 = dependent_exp(state->x0);
    state->x1 = dependent_exp(state->x1);
    state->x2 = dependent_exp(state->x2);
    state->x3 = dependent_exp(state->x3);
    state->x4 = dependent_exp(state->x4);
    state->x5 = dependent_exp(state->x5);
    state->x6 = dependent_exp(state->x6);
    state->x7 = dependent_exp(state->x7);
}

__device__ __forceinline__ void rcp_round(RegisterState* state) {
    state->x0 = dependent_rcp(state->x0);
    state->x1 = dependent_rcp(state->x1);
    state->x2 = dependent_rcp(state->x2);
    state->x3 = dependent_rcp(state->x3);
    state->x4 = dependent_rcp(state->x4);
    state->x5 = dependent_rcp(state->x5);
    state->x6 = dependent_rcp(state->x6);
    state->x7 = dependent_rcp(state->x7);
}

template <int Remaining>
struct AddRounds {
    __device__ __forceinline__ static void run(RegisterState* state) {
        add_round(state);
        AddRounds<Remaining - 1>::run(state);
    }
};

template <>
struct AddRounds<0> {
    __device__ __forceinline__ static void run(RegisterState*) {}
};

template <Pattern pattern>
struct Program;

template <>
struct Program<Pattern::kAdd> {
    __device__ __forceinline__ static void run(RegisterState* state) {
        AddRounds<32>::run(state);
    }
};

template <>
struct Program<Pattern::kExp> {
    __device__ __forceinline__ static void run(RegisterState* state) {
        #pragma unroll 32
        for (int round = 0; round < 32; ++round) {
            exp_round(state);
        }
    }
};

template <>
struct Program<Pattern::kRcp> {
    __device__ __forceinline__ static void run(RegisterState* state) {
        #pragma unroll 32
        for (int round = 0; round < 32; ++round) {
            rcp_round(state);
        }
    }
};

template <>
struct Program<Pattern::kSiluMix> {
    __device__ __forceinline__ static void run(RegisterState* state) {
        AddRounds<15>::run(state);
        exp_round(state);
        AddRounds<15>::run(state);
        rcp_round(state);
    }
};

template <Pattern pattern>
__global__ void register_burst_kernel(
        unsigned long long* clock_begin,
        unsigned long long* clock_end,
        unsigned long long* realtime_begin,
        unsigned long long* realtime_end,
        int program_repeats) {
    RegisterState state {
        1.0009765625f, 1.0019531250f, 1.0029296875f, 1.0039062500f,
        1.0048828125f, 1.0058593750f, 1.0068359375f, 1.0078125000f,
    };
    asm volatile("" : : : "memory");
    const unsigned long long global_start = __builtin_mxc_get_realtime();
    const unsigned long long local_start = clock64();
    #pragma unroll 1
    for (int repeat = 0; repeat < program_repeats; ++repeat) {
        Program<pattern>::run(&state);
    }
    asm volatile("" : : "r"(state.x0), "r"(state.x1), "r"(state.x2),
                 "r"(state.x3), "r"(state.x4), "r"(state.x5),
                 "r"(state.x6), "r"(state.x7) : "memory");
    const unsigned long long local_end = clock64();
    const unsigned long long global_end = __builtin_mxc_get_realtime();
    if (threadIdx.x == 0) {
        const int block = blockIdx.x;
        clock_begin[block] = local_start;
        clock_end[block] = local_end;
        realtime_begin[block] = global_start;
        realtime_end[block] = global_end;
    }
}

template <Pattern pattern>
__device__ __forceinline__ void run_memory_compute(
        RegisterState* state, int program_repeats) {
    #pragma unroll 1
    for (int repeat = 0; repeat < program_repeats; ++repeat) {
        if constexpr (pattern == Pattern::kMemoryOnly) {
            asm volatile("" : : "r"(state->x0), "r"(state->x1),
                         "r"(state->x2), "r"(state->x3),
                         "r"(state->x4), "r"(state->x5),
                         "r"(state->x6), "r"(state->x7) : "memory");
        } else if constexpr (pattern == Pattern::kMemoryAdd) {
            Program<Pattern::kAdd>::run(state);
        } else if constexpr (pattern == Pattern::kMemorySiluMix) {
            Program<Pattern::kSiluMix>::run(state);
        }
    }
}

template <Pattern pattern>
__global__ void memory_burst_kernel(
        const float4* source_a,
        const float4* source_b,
        float4* destination,
        unsigned long long* clock_begin,
        unsigned long long* clock_end,
        unsigned long long* realtime_begin,
        unsigned long long* realtime_end,
        int program_repeats) {
    const long long index = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    asm volatile("" : : : "memory");
    const unsigned long long global_start = __builtin_mxc_get_realtime();
    const unsigned long long local_start = clock64();
    const float4 left = source_a[index];
    const float4 right = source_b[index];
    RegisterState state {
        left.x + right.x, left.y + right.y,
        left.z + right.z, left.w + right.w,
        left.x - right.x, left.y - right.y,
        left.z - right.z, left.w - right.w,
    };
    run_memory_compute<pattern>(&state, program_repeats);
    destination[index] = make_float4(
        state.x0 + state.x4, state.x1 + state.x5,
        state.x2 + state.x6, state.x3 + state.x7);
    asm volatile("" : : : "memory");
    const unsigned long long local_end = clock64();
    const unsigned long long global_end = __builtin_mxc_get_realtime();
    if (threadIdx.x == 0) {
        const int block = blockIdx.x;
        clock_begin[block] = local_start;
        clock_end[block] = local_end;
        realtime_begin[block] = global_start;
        realtime_end[block] = global_end;
    }
}

Pattern parse_pattern(const std::string& text) {
    if (text == "add") {
        return Pattern::kAdd;
    }
    if (text == "exp") {
        return Pattern::kExp;
    }
    if (text == "rcp") {
        return Pattern::kRcp;
    }
    if (text == "silu_mix") {
        return Pattern::kSiluMix;
    }
    if (text == "memory_only") {
        return Pattern::kMemoryOnly;
    }
    if (text == "memory_add") {
        return Pattern::kMemoryAdd;
    }
    if (text == "memory_silu_mix") {
        return Pattern::kMemorySiluMix;
    }
    std::fprintf(stderr, "invalid pattern: %s\n", text.c_str());
    std::exit(EXIT_FAILURE);
}

const char* pattern_name(Pattern pattern) {
    if (pattern == Pattern::kAdd) {
        return "add";
    }
    if (pattern == Pattern::kExp) {
        return "exp";
    }
    if (pattern == Pattern::kRcp) {
        return "rcp";
    }
    if (pattern == Pattern::kSiluMix) {
        return "silu_mix";
    }
    if (pattern == Pattern::kMemoryOnly) {
        return "memory_only";
    }
    if (pattern == Pattern::kMemoryAdd) {
        return "memory_add";
    }
    if (pattern == Pattern::kMemorySiluMix) {
        return "memory_silu_mix";
    }
    std::fprintf(stderr, "unknown pattern\n");
    std::exit(EXIT_FAILURE);
}

int instructions_per_wave(Pattern pattern, int program_repeats) {
    if (pattern == Pattern::kAdd || pattern == Pattern::kExp
            || pattern == Pattern::kRcp || pattern == Pattern::kSiluMix) {
        return 256 * program_repeats;
    }
    if (pattern == Pattern::kMemoryOnly) {
        return 3;
    }
    if (pattern == Pattern::kMemoryAdd
            || pattern == Pattern::kMemorySiluMix) {
        return 3 + 256 * program_repeats;
    }
    std::fprintf(stderr, "unknown pattern\n");
    std::exit(EXIT_FAILURE);
}

int parse_nonnegative_int(const char* text, const char* label) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 2147483647L) {
        std::fprintf(stderr, "%s must be a non-negative integer\n", label);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

void validate_row(const PlanRow& row) {
    if (row.sample_id.empty() || row.sample_id.size() > 96
            || row.sample_id.find(',') != std::string::npos) {
        std::fprintf(stderr, "invalid sample_id\n");
        std::exit(EXIT_FAILURE);
    }
    if (row.condition != "cold" && row.condition != "idle100"
            && row.condition != "warm" && row.condition != "warm3self") {
        std::fprintf(stderr, "invalid condition: %s\n", row.condition.c_str());
        std::exit(EXIT_FAILURE);
    }
    if (row.trial <= 0 || row.order_position < 0
            || (row.grid != 104 && row.grid != 416 && row.grid != 1664
                && row.grid != 3328 && row.grid != 6656
                && row.grid != 9984 && row.grid != 13312)
            || row.program_repeats <= 0
            || (row.idle_ms != 0 && row.idle_ms != 100
                && row.idle_ms != 2000)
            || (row.prewarm != 0 && row.prewarm != 1
                && row.prewarm != 2)) {
        std::fprintf(stderr, "invalid plan row: %s\n", row.sample_id.c_str());
        std::exit(EXIT_FAILURE);
    }
    int expected_prewarm = 0;
    if (row.condition == "warm") {
        expected_prewarm = 1;
    } else if (row.condition == "warm3self") {
        expected_prewarm = 2;
    }
    if (row.prewarm != expected_prewarm) {
        std::fprintf(stderr, "condition/prewarm mismatch\n");
        std::exit(EXIT_FAILURE);
    }
    int expected_idle_ms = 0;
    if (row.condition == "cold") {
        expected_idle_ms = 2000;
    } else if (row.condition == "idle100") {
        expected_idle_ms = 100;
    }
    if (row.idle_ms != expected_idle_ms) {
        std::fprintf(stderr, "condition/idle mismatch\n");
        std::exit(EXIT_FAILURE);
    }
}

std::vector<PlanRow> read_plan() {
    std::vector<PlanRow> rows;
    PlanRow row;
    std::string pattern;
    while (std::cin >> row.sample_id >> row.condition >> pattern
                    >> row.trial >> row.order_position >> row.grid
                    >> row.program_repeats >> row.idle_ms >> row.prewarm) {
        row.pattern = parse_pattern(pattern);
        validate_row(row);
        rows.push_back(row);
    }
    if (!std::cin.eof() || rows.empty()) {
        std::fprintf(stderr, "short-burst plan is empty or malformed\n");
        std::exit(EXIT_FAILURE);
    }
    return rows;
}

long long monotonic_raw_ns() {
    timespec timestamp {};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
        std::perror("clock_gettime");
        std::exit(EXIT_FAILURE);
    }
    return (static_cast<long long>(timestamp.tv_sec) * 1000000000LL
            + timestamp.tv_nsec);
}

template <typename T>
double median_sorted(const std::vector<T>& values) {
    const std::size_t size = values.size();
    if (size % 2 == 1) {
        return static_cast<double>(values[size / 2]);
    }
    return 0.5 * static_cast<double>(values[size / 2 - 1] + values[size / 2]);
}

template <Pattern pattern>
void launch_typed(
        int grid, int program_repeats,
        const float4* source_a,
        const float4* source_b,
        float4* destination,
        unsigned long long* clock_begin,
        unsigned long long* clock_end,
        unsigned long long* realtime_begin,
        unsigned long long* realtime_end) {
    if constexpr (pattern == Pattern::kMemoryOnly
            || pattern == Pattern::kMemoryAdd
            || pattern == Pattern::kMemorySiluMix) {
        memory_burst_kernel<pattern><<<grid, kExpectedBlockSize>>>(
            source_a, source_b, destination,
            clock_begin, clock_end, realtime_begin, realtime_end,
            program_repeats);
    } else {
        register_burst_kernel<pattern><<<grid, kExpectedBlockSize>>>(
            clock_begin, clock_end, realtime_begin, realtime_end,
            program_repeats);
    }
    CHECK(mcGetLastError());
}

void launch(
        Pattern pattern, int grid, int program_repeats,
        const float4* source_a,
        const float4* source_b,
        float4* destination,
        unsigned long long* clock_begin,
        unsigned long long* clock_end,
        unsigned long long* realtime_begin,
        unsigned long long* realtime_end) {
    if (pattern == Pattern::kAdd) {
        launch_typed<Pattern::kAdd>(
            grid, program_repeats, source_a, source_b, destination,
            clock_begin, clock_end,
            realtime_begin, realtime_end);
    } else if (pattern == Pattern::kExp) {
        launch_typed<Pattern::kExp>(
            grid, program_repeats, source_a, source_b, destination,
            clock_begin, clock_end,
            realtime_begin, realtime_end);
    } else if (pattern == Pattern::kRcp) {
        launch_typed<Pattern::kRcp>(
            grid, program_repeats, source_a, source_b, destination,
            clock_begin, clock_end,
            realtime_begin, realtime_end);
    } else if (pattern == Pattern::kSiluMix) {
        launch_typed<Pattern::kSiluMix>(
            grid, program_repeats, source_a, source_b, destination,
            clock_begin, clock_end,
            realtime_begin, realtime_end);
    } else if (pattern == Pattern::kMemoryOnly) {
        launch_typed<Pattern::kMemoryOnly>(
            grid, program_repeats, source_a, source_b, destination,
            clock_begin, clock_end,
            realtime_begin, realtime_end);
    } else if (pattern == Pattern::kMemoryAdd) {
        launch_typed<Pattern::kMemoryAdd>(
            grid, program_repeats, source_a, source_b, destination,
            clock_begin, clock_end,
            realtime_begin, realtime_end);
    } else if (pattern == Pattern::kMemorySiluMix) {
        launch_typed<Pattern::kMemorySiluMix>(
            grid, program_repeats, source_a, source_b, destination,
            clock_begin, clock_end,
            realtime_begin, realtime_end);
    } else {
        std::fprintf(stderr, "unknown pattern\n");
        std::exit(EXIT_FAILURE);
    }
}

struct TimingSummary {
    unsigned long long clock_min;
    double clock_median;
    unsigned long long clock_max;
    unsigned long long realtime_min;
    double realtime_median;
    unsigned long long realtime_max;
    unsigned long long grid_realtime_span;
    double frequency_min_mhz;
    double frequency_median_mhz;
    double frequency_max_mhz;
    double frequency_first_decile_mhz;
    double frequency_last_decile_mhz;
};

TimingSummary summarize_timing(
        int grid,
        const std::vector<unsigned long long>& clock_begin,
        const std::vector<unsigned long long>& clock_end,
        const std::vector<unsigned long long>& realtime_begin,
        const std::vector<unsigned long long>& realtime_end) {
    std::vector<unsigned long long> clock_spans;
    std::vector<unsigned long long> realtime_spans;
    std::vector<double> frequencies;
    std::vector<std::pair<unsigned long long, double>> ordered_frequencies;
    clock_spans.reserve(grid);
    realtime_spans.reserve(grid);
    frequencies.reserve(grid);
    ordered_frequencies.reserve(grid);
    unsigned long long earliest = std::numeric_limits<unsigned long long>::max();
    unsigned long long latest = 0;
    for (int block = 0; block < grid; ++block) {
        if (clock_end[block] <= clock_begin[block]
                || realtime_end[block] <= realtime_begin[block]) {
            std::fprintf(stderr, "invalid device timer interval at block %d\n", block);
            std::exit(EXIT_FAILURE);
        }
        const unsigned long long clock_span = (
            clock_end[block] - clock_begin[block]);
        const unsigned long long realtime_span = (
            realtime_end[block] - realtime_begin[block]);
        const double frequency_mhz = (
            100.0 * static_cast<double>(clock_span)
            / static_cast<double>(realtime_span));
        if (!std::isfinite(frequency_mhz) || frequency_mhz <= 0.0) {
            std::fprintf(stderr, "invalid derived frequency at block %d\n", block);
            std::exit(EXIT_FAILURE);
        }
        clock_spans.push_back(clock_span);
        realtime_spans.push_back(realtime_span);
        frequencies.push_back(frequency_mhz);
        ordered_frequencies.emplace_back(realtime_begin[block], frequency_mhz);
        earliest = std::min(earliest, realtime_begin[block]);
        latest = std::max(latest, realtime_end[block]);
    }
    std::sort(clock_spans.begin(), clock_spans.end());
    std::sort(realtime_spans.begin(), realtime_spans.end());
    std::sort(frequencies.begin(), frequencies.end());
    std::sort(ordered_frequencies.begin(), ordered_frequencies.end());
    const std::size_t decile_count = std::max<std::size_t>(
        1, ordered_frequencies.size() / 10);
    std::vector<double> first_decile;
    std::vector<double> last_decile;
    first_decile.reserve(decile_count);
    last_decile.reserve(decile_count);
    for (std::size_t index = 0; index < decile_count; ++index) {
        first_decile.push_back(ordered_frequencies[index].second);
        last_decile.push_back(
            ordered_frequencies[ordered_frequencies.size()
                                - decile_count + index].second);
    }
    std::sort(first_decile.begin(), first_decile.end());
    std::sort(last_decile.begin(), last_decile.end());
    return {
        clock_spans.front(), median_sorted(clock_spans), clock_spans.back(),
        realtime_spans.front(), median_sorted(realtime_spans),
        realtime_spans.back(), latest - earliest,
        frequencies.front(), median_sorted(frequencies), frequencies.back(),
        median_sorted(first_decile), median_sorted(last_decile),
    };
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s DEVICE_ID < plan.txt\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device_id = parse_nonnegative_int(argv[1], "device-id");
    CHECK(mcSetDevice(device_id));

    mcDeviceProp_t properties {};
    CHECK(mcGetDeviceProperties(&properties, device_id));
    if (properties.multiProcessorCount != kExpectedApCount
            || properties.warpSize != kExpectedWaveSize
            || properties.clockRate <= 0) {
        std::fprintf(stderr, "unexpected C500 device properties\n");
        return EXIT_FAILURE;
    }
    char pci_bus_id[32] = {};
    CHECK(mcDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_id));
    for (char* character = pci_bus_id; *character != '\0'; ++character) {
        *character = static_cast<char>(std::tolower(
            static_cast<unsigned char>(*character)));
    }
    if (std::strchr(properties.name, ',') != nullptr
            || std::strchr(pci_bus_id, ',') != nullptr) {
        std::fprintf(stderr, "device identity is not safe for CSV\n");
        return EXIT_FAILURE;
    }

    const std::vector<PlanRow> plan = read_plan();
    unsigned long long* device_clock_begin = nullptr;
    unsigned long long* device_clock_end = nullptr;
    unsigned long long* device_realtime_begin = nullptr;
    unsigned long long* device_realtime_end = nullptr;
    float4* device_source_a = nullptr;
    float4* device_source_b = nullptr;
    float4* device_destination = nullptr;
    const std::size_t bytes = kMaximumGrid * sizeof(unsigned long long);
    const std::size_t vector_bytes = (
        static_cast<std::size_t>(kMaximumGrid)
        * kExpectedBlockSize * sizeof(float4));
    CHECK(mcMalloc(reinterpret_cast<void**>(&device_clock_begin), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&device_clock_end), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&device_realtime_begin), bytes));
    CHECK(mcMalloc(reinterpret_cast<void**>(&device_realtime_end), bytes));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&device_source_a), vector_bytes));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&device_source_b), vector_bytes));
    CHECK(mcMalloc(
        reinterpret_cast<void**>(&device_destination), vector_bytes));
    CHECK(mcMemset(device_source_a, 0x3f, vector_bytes));
    CHECK(mcMemset(device_source_b, 0x3e, vector_bytes));
    CHECK(mcMemset(device_destination, 0, vector_bytes));
    std::vector<unsigned long long> host_clock_begin(kMaximumGrid);
    std::vector<unsigned long long> host_clock_end(kMaximumGrid);
    std::vector<unsigned long long> host_realtime_begin(kMaximumGrid);
    std::vector<unsigned long long> host_realtime_end(kMaximumGrid);
    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));

    const Pattern patterns[] = {
        Pattern::kAdd, Pattern::kExp, Pattern::kRcp, Pattern::kSiluMix,
        Pattern::kMemoryOnly, Pattern::kMemoryAdd,
        Pattern::kMemorySiluMix};
    for (Pattern pattern : patterns) {
        launch(pattern, 1, 1,
               device_source_a, device_source_b, device_destination,
               device_clock_begin, device_clock_end,
               device_realtime_begin, device_realtime_end);
    }
    CHECK(mcDeviceSynchronize());

    std::printf(
        "metadata,3,%s,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%llu\n",
        properties.name, device_id, pci_bus_id, properties.multiProcessorCount,
        properties.warpSize, properties.clockRate, kExpectedBlockSize,
        kPrewarmGrid, kPrewarmProgramRepeats, kPrewarmLaunches,
        kSelfWarmupLaunches, kGlobalTimerHz);
    std::printf(
        "sample_header,sample_id,condition,pattern,trial,order_position,grid,"
        "block,program_repeats,instructions_per_wave,idle_ms,prewarm,"
        "host_start_ns,host_end_ns,event_ns,clock64_min,clock64_median,"
        "clock64_max,realtime_min_ns,realtime_median_ns,realtime_max_ns,"
        "grid_realtime_span_ns,frequency_min_mhz,frequency_median_mhz,"
        "frequency_max_mhz,frequency_first_decile_mhz,"
        "frequency_last_decile_mhz\n");
    std::fflush(stdout);

    for (const PlanRow& row : plan) {
        if (row.prewarm == 1) {
            for (int index = 0; index < kPrewarmLaunches; ++index) {
                launch(Pattern::kAdd, kPrewarmGrid, kPrewarmProgramRepeats,
                       device_source_a, device_source_b, device_destination,
                       device_clock_begin, device_clock_end,
                       device_realtime_begin, device_realtime_end);
            }
            CHECK(mcDeviceSynchronize());
        } else if (row.prewarm == 2) {
            for (int index = 0; index < kSelfWarmupLaunches; ++index) {
                launch(row.pattern, row.grid, row.program_repeats,
                       device_source_a, device_source_b, device_destination,
                       device_clock_begin, device_clock_end,
                       device_realtime_begin, device_realtime_end);
            }
            CHECK(mcDeviceSynchronize());
        }
        if (row.idle_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(row.idle_ms));
        }

        const long long host_start_ns = monotonic_raw_ns();
        CHECK(mcEventRecord(start));
        launch(row.pattern, row.grid, row.program_repeats,
               device_source_a, device_source_b, device_destination,
               device_clock_begin, device_clock_end,
               device_realtime_begin, device_realtime_end);
        CHECK(mcEventRecord(end));
        CHECK(mcDeviceSynchronize());
        const long long host_end_ns = monotonic_raw_ns();
        float elapsed_ms = 0.0f;
        CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
        CHECK(mcMemcpy(host_clock_begin.data(), device_clock_begin,
                       row.grid * sizeof(unsigned long long),
                       mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(host_clock_end.data(), device_clock_end,
                       row.grid * sizeof(unsigned long long),
                       mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(host_realtime_begin.data(), device_realtime_begin,
                       row.grid * sizeof(unsigned long long),
                       mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(host_realtime_end.data(), device_realtime_end,
                       row.grid * sizeof(unsigned long long),
                       mcMemcpyDeviceToHost));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
            std::fprintf(stderr, "invalid timed sample: %s\n",
                         row.sample_id.c_str());
            return EXIT_FAILURE;
        }
        const TimingSummary summary = summarize_timing(
            row.grid, host_clock_begin, host_clock_end,
            host_realtime_begin, host_realtime_end);
        std::printf(
            "sample,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%lld,%.9f,"
            "%llu,%.3f,%llu,%llu,%.3f,%llu,%llu,%.6f,%.6f,%.6f,%.6f,"
            "%.6f\n",
            row.sample_id.c_str(), row.condition.c_str(),
            pattern_name(row.pattern), row.trial, row.order_position,
            row.grid, kExpectedBlockSize, row.program_repeats,
            instructions_per_wave(row.pattern, row.program_repeats),
            row.idle_ms, row.prewarm, host_start_ns, host_end_ns,
            static_cast<double>(elapsed_ms) * 1000000.0,
            summary.clock_min, summary.clock_median, summary.clock_max,
            summary.realtime_min, summary.realtime_median,
            summary.realtime_max, summary.grid_realtime_span,
            summary.frequency_min_mhz, summary.frequency_median_mhz,
            summary.frequency_max_mhz,
            summary.frequency_first_decile_mhz,
            summary.frequency_last_decile_mhz);
        std::fflush(stdout);
    }

    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    CHECK(mcFree(device_clock_begin));
    CHECK(mcFree(device_clock_end));
    CHECK(mcFree(device_realtime_begin));
    CHECK(mcFree(device_realtime_end));
    CHECK(mcFree(device_source_a));
    CHECK(mcFree(device_source_b));
    CHECK(mcFree(device_destination));
    return EXIT_SUCCESS;
}
