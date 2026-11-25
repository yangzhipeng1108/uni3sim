/*
 * NVIDIA A800 DRAM read/write-direction transition microbenchmark.
 *
 * This is a matched-sequence mechanism test, not a production operator.  A
 * trial launches four pure streaming phases per repetition.  Every phase has
 * the same byte footprint and launch geometry, and every launch receives a
 * distinct, non-overlapping window.  Read and write windows also live in
 * separate allocations.  RRWW, WWRR, RWRW, and WRWR vary only their program
 * order; RRRR and WWWW are no-transition startup controls.
 *
 * The streaming operation is a coalesced 128-bit ld.global.cg (L1 bypass) or
 * st.global.wb (explicit write-back policy).  The timed read kernel has no
 * store or global output: volatile inline loads feed a register-only trap
 * invariant.
 * Read-buffer and deterministic write correctness are checked by independent
 * validation kernels after timing.
 *
 * `program_*_transitions` below count adjacent R/W labels issued by this
 * program.  They are not evidence of controller-visible direction changes.
 * NCU duration metrics are forbidden for this benchmark.  NCU may only supply
 * non-duration counter and SASS evidence for traffic, cache policy, and any
 * eventual claim about controller-visible behavior.
 *
 * Build:
 *   nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo \
 *       -o direction_transition direction_transition.cu
 *
 * Run the declared sweep, one case, or a host-only CLI/config check:
 *   CUDA_VISIBLE_DEVICES=3 ./direction_transition
 *   CUDA_VISIBLE_DEVICES=3 ./direction_transition \
 *       --single RWRW 64 2 216 ab
 *   ./direction_transition --describe
 */

#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                     __FILE__, __LINE__, cudaGetErrorString(error));        \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kVectorBytes = 16;
constexpr int kBlockSize = 256;
constexpr int kTrials = 7;
constexpr int kPhasesPerSequence = 4;
constexpr int kExpectedSms = 108;
constexpr int kExpectedWarpSize = 32;
constexpr int kExpectedL2Bytes = 40 * 1024 * 1024;
constexpr std::uint32_t kArenaASalt = 0x243f6a88U;
constexpr std::uint32_t kArenaBSalt = 0x85a308d3U;
constexpr std::uint32_t kWriteSalt = 0x9e3779b9U;
constexpr std::uint32_t kTokenSeed = 0xd1b54a35U;
constexpr std::uint32_t kReadFoldInvariant = 0x6a09e667U;
constexpr std::uint8_t kSentinelByte = 0xa5U;
static_assert(sizeof(uint4) == kVectorBytes, "uint4 must be 16 bytes");

enum class Direction : unsigned char {
    kRead,
    kWrite,
};

struct SequenceSpec {
    const char* name;
    Direction phases[kPhasesPerSequence];
};

constexpr SequenceSpec kSequences[] = {
    {"RRWW", {Direction::kRead, Direction::kRead,
               Direction::kWrite, Direction::kWrite}},
    {"WWRR", {Direction::kWrite, Direction::kWrite,
               Direction::kRead, Direction::kRead}},
    {"RWRW", {Direction::kRead, Direction::kWrite,
               Direction::kRead, Direction::kWrite}},
    {"WRWR", {Direction::kWrite, Direction::kRead,
               Direction::kWrite, Direction::kRead}},
    {"RRRR", {Direction::kRead, Direction::kRead,
               Direction::kRead, Direction::kRead}},
    {"WWWW", {Direction::kWrite, Direction::kWrite,
               Direction::kWrite, Direction::kWrite}},
};
constexpr int kSequenceCount = (
    static_cast<int>(sizeof(kSequences) / sizeof(kSequences[0])));
static_assert(kSequenceCount == 6, "sequence schedule expects six labels");

constexpr int count_pattern_direction(
        const SequenceSpec& sequence, Direction direction) {
    int count = 0;
    for (Direction phase : sequence.phases) {
        if (phase == direction) {
            ++count;
        }
    }
    return count;
}
static_assert(
    count_pattern_direction(kSequences[0], Direction::kRead) == 2
    && count_pattern_direction(kSequences[1], Direction::kRead) == 2
    && count_pattern_direction(kSequences[2], Direction::kRead) == 2
    && count_pattern_direction(kSequences[3], Direction::kRead) == 2
    && count_pattern_direction(kSequences[0], Direction::kWrite) == 2
    && count_pattern_direction(kSequences[1], Direction::kWrite) == 2
    && count_pattern_direction(kSequences[2], Direction::kWrite) == 2
    && count_pattern_direction(kSequences[3], Direction::kWrite) == 2,
    "mixed patterns must have identical direction-local occurrence counts");

// Compile-time cyclic counterbalancing: over trials 0..5 every sequence takes
// every order slot once.  Trial 6 repeats the declared trial-0 order.  This
// order is fixed independently of all measurements.
constexpr int kSequenceOrder[kTrials][kSequenceCount] = {
    {0, 1, 2, 3, 4, 5},
    {1, 2, 3, 4, 5, 0},
    {2, 3, 4, 5, 0, 1},
    {3, 4, 5, 0, 1, 2},
    {4, 5, 0, 1, 2, 3},
    {5, 0, 1, 2, 3, 4},
    {0, 1, 2, 3, 4, 5},
};

constexpr bool sequence_order_is_valid() {
    for (int trial = 0; trial < kTrials; ++trial) {
        for (int slot = 0; slot < kSequenceCount; ++slot) {
            int candidate = kSequenceOrder[trial][slot];
            if (candidate < 0 || candidate >= kSequenceCount) {
                return false;
            }
            for (int prior = 0; prior < slot; ++prior) {
                if (kSequenceOrder[trial][prior] == candidate) {
                    return false;
                }
            }
        }
    }
    return true;
}
static_assert(sequence_order_is_valid(),
              "each trial schedule row must be a sequence permutation");

enum class AddressRoles {
    kAReadBWrite,
    kBReadAWrite,
};

struct RoleSpec {
    AddressRoles roles;
    const char* cli_name;
    const char* output_name;
};

constexpr RoleSpec kRoleSpecs[] = {
    {AddressRoles::kAReadBWrite, "ab", "A_read_B_write"},
    {AddressRoles::kBReadAWrite, "ba", "B_read_A_write"},
};

struct TransitionCounts {
    int read_to_write;
    int write_to_read;
};

struct CaseConfig {
    const SequenceSpec* sequence;
    int q_mib;
    int repetitions;
    int grid;
    const RoleSpec* role;
};

static int parse_positive_integer(const char* text, const char* field) {
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value <= 0 || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a positive integer, got %s\n",
                     field, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

template <std::size_t N>
static bool contains(const int (&values)[N], int value) {
    for (int candidate : values) {
        if (candidate == value) {
            return true;
        }
    }
    return false;
}

static const SequenceSpec& parse_sequence(const char* text) {
    for (const SequenceSpec& sequence : kSequences) {
        if (std::strcmp(text, sequence.name) == 0) {
            return sequence;
        }
    }
    std::fprintf(
        stderr,
        "sequence must be one of RRWW,WWRR,RWRW,WRWR,RRRR,WWWW; got %s\n",
        text);
    std::exit(EXIT_FAILURE);
}

static const RoleSpec& parse_roles(const char* text) {
    for (const RoleSpec& role : kRoleSpecs) {
        if (std::strcmp(text, role.cli_name) == 0) {
            return role;
        }
    }
    std::fprintf(stderr, "address roles must be ab or ba; got %s\n", text);
    std::exit(EXIT_FAILURE);
}

static CaseConfig parse_single_case(char** argv) {
    constexpr int q_values[] = {64, 256};
    constexpr int repetition_values[] = {1, 2, 4};
    constexpr int grid_values[] = {108, 216, 432};
    CaseConfig config = {
        &parse_sequence(argv[2]),
        parse_positive_integer(argv[3], "q_mib"),
        parse_positive_integer(argv[4], "M"),
        parse_positive_integer(argv[5], "grid"),
        &parse_roles(argv[6]),
    };
    if (!contains(q_values, config.q_mib)) {
        std::fprintf(stderr, "q_mib must be 64 or 256; got %d\n",
                     config.q_mib);
        std::exit(EXIT_FAILURE);
    }
    if (!contains(repetition_values, config.repetitions)) {
        std::fprintf(stderr, "M must be 1, 2, or 4; got %d\n",
                     config.repetitions);
        std::exit(EXIT_FAILURE);
    }
    if (!contains(grid_values, config.grid)) {
        std::fprintf(stderr, "grid must be 108, 216, or 432; got %d\n",
                     config.grid);
        std::exit(EXIT_FAILURE);
    }
    return config;
}

static void describe(const char* program) {
    std::printf(
        "usage: %s [--single SEQUENCE Q_MIB M GRID ADDRESS_ROLES]\n"
        "       %s --describe\n"
        "SEQUENCE=RRWW|WWRR|RWRW|WRWR|RRRR|WWWW\n"
        "Q_MIB=64|256 M=1|2|4 GRID=108|216|432 "
        "ADDRESS_ROLES=ab|ba\n",
        program, program);
}

__host__ __device__ __forceinline__ std::uint32_t mix_bits(
        std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__host__ __device__ __forceinline__ uint4 make_value(
        std::uint32_t index, std::uint32_t salt) {
    std::uint32_t x = mix_bits(index ^ salt);
    std::uint32_t y = mix_bits(x ^ 0x68bc21ebU);
    std::uint32_t z = mix_bits(y ^ 0x02e5be93U);
    std::uint32_t w = mix_bits(z ^ 0x967a889bU);
    return make_uint4(x, y, z, w);
}

__host__ __device__ __forceinline__ uint4 make_read_value(
        std::uint32_t index, std::uint32_t salt) {
    uint4 value = make_value(index, salt);
    value.w = value.x ^ value.y ^ value.z ^ kReadFoldInvariant;
    return value;
}

__device__ __forceinline__ uint4 load_global_cg(const uint4* address) {
    uint4 value;
    asm volatile(
        "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(address)
        : "memory");
    return value;
}

__device__ __forceinline__ void store_global_wb(
        uint4* address, const uint4& value) {
    asm volatile(
        "st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
        :: "l"(address), "r"(value.x), "r"(value.y),
           "r"(value.z), "r"(value.w)
        : "memory");
}

__global__ void initialize_read_window(
        uint4* window, std::size_t element_count,
        std::size_t arena_element_offset, std::uint32_t arena_salt) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t stride = (
        static_cast<std::size_t>(blockDim.x) * gridDim.x);
    for (std::size_t index = thread; index < element_count; index += stride) {
        window[index] = make_read_value(
            static_cast<std::uint32_t>(arena_element_offset + index),
            arena_salt);
    }
}

extern "C" __global__ void a800_direction_read_phase(
        const uint4* source, std::size_t element_count) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t stride = (
        static_cast<std::size_t>(blockDim.x) * gridDim.x);
    for (std::size_t index = thread; index < element_count; index += stride) {
        uint4 value = load_global_cg(source + index);
        std::uint32_t observed_fold = (
            value.x ^ value.y ^ value.z ^ value.w);
        // The data-dependent trap is a register-only observable consume.  It
        // prevents ptxas from deleting the volatile load without adding a
        // store or global output.  The independently initialized read value
        // guarantees this path is not taken for a correct buffer.
        if (observed_fold != kReadFoldInvariant) {
            asm volatile("trap;" ::: "memory");
        }
    }
}

extern "C" __global__ void a800_direction_write_phase(
        uint4* destination, std::size_t element_count,
        std::uint32_t phase_salt) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t stride = (
        static_cast<std::size_t>(blockDim.x) * gridDim.x);
    for (std::size_t index = thread; index < element_count; index += stride) {
        uint4 value = make_value(static_cast<std::uint32_t>(index), phase_salt);
        store_global_wb(destination + index, value);
    }
}

__global__ void validate_read_window(
        const uint4* source, std::size_t element_count,
        std::size_t arena_element_offset, std::uint32_t arena_salt,
        unsigned long long* errors) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t stride = (
        static_cast<std::size_t>(blockDim.x) * gridDim.x);
    unsigned long long local_errors = 0;
    for (std::size_t index = thread; index < element_count; index += stride) {
        uint4 expected = make_read_value(
            static_cast<std::uint32_t>(arena_element_offset + index),
            arena_salt);
        uint4 observed = source[index];
        local_errors += (
            observed.x != expected.x || observed.y != expected.y
            || observed.z != expected.z || observed.w != expected.w);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

__global__ void validate_write_window(
        const uint4* destination, std::size_t element_count,
        std::uint32_t phase_salt, unsigned long long* errors) {
    std::size_t thread = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t stride = (
        static_cast<std::size_t>(blockDim.x) * gridDim.x);
    unsigned long long local_errors = 0;
    for (std::size_t index = thread; index < element_count; index += stride) {
        uint4 expected = make_value(static_cast<std::uint32_t>(index), phase_salt);
        uint4 observed = destination[index];
        local_errors += (
            observed.x != expected.x || observed.y != expected.y
            || observed.z != expected.z || observed.w != expected.w);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

static std::uint32_t direction_token(
        Direction direction, int occurrence, AddressRoles roles) {
    std::uint32_t role_bit = (
        roles == AddressRoles::kAReadBWrite ? 0U : 0x80000000U);
    std::uint32_t direction_bit = 0;
    if (direction == Direction::kRead) {
        direction_bit = 0U;
    } else if (direction == Direction::kWrite) {
        direction_bit = 0x40000000U;
    } else {
        std::fprintf(stderr, "unknown token direction\n");
        std::exit(EXIT_FAILURE);
    }
    return mix_bits(
        static_cast<std::uint32_t>(occurrence)
        ^ role_bit ^ direction_bit ^ kTokenSeed);
}

static Direction direction_at(const SequenceSpec& sequence, int ordinal) {
    return sequence.phases[ordinal % kPhasesPerSequence];
}

static TransitionCounts count_program_transitions(
        const SequenceSpec& sequence, int repetitions) {
    TransitionCounts counts = {0, 0};
    const int launches = kPhasesPerSequence * repetitions;
    for (int ordinal = 1; ordinal < launches; ++ordinal) {
        Direction previous = direction_at(sequence, ordinal - 1);
        Direction current = direction_at(sequence, ordinal);
        if (previous == Direction::kRead && current == Direction::kWrite) {
            ++counts.read_to_write;
        } else if (previous == Direction::kWrite
                   && current == Direction::kRead) {
            ++counts.write_to_read;
        } else if (previous == current) {
            continue;
        } else {
            std::fprintf(stderr, "unknown direction transition\n");
            std::exit(EXIT_FAILURE);
        }
    }
    return counts;
}

static void select_arenas(
        AddressRoles roles, uint4* arena_a, uint4* arena_b,
        uint4** read_arena, uint4** write_arena,
        std::uint32_t* read_arena_salt) {
    if (roles == AddressRoles::kAReadBWrite) {
        *read_arena = arena_a;
        *write_arena = arena_b;
        *read_arena_salt = kArenaASalt;
    } else if (roles == AddressRoles::kBReadAWrite) {
        *read_arena = arena_b;
        *write_arena = arena_a;
        *read_arena_salt = kArenaBSalt;
    } else {
        std::fprintf(stderr, "unknown address-role assignment\n");
        std::exit(EXIT_FAILURE);
    }
}

static std::uint64_t measure_sequence(
        const CaseConfig& config, std::size_t elements_per_window,
        uint4* read_arena, uint4* write_arena) {
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    const int launches = kPhasesPerSequence * config.repetitions;
    int read_occurrence = 0;
    int write_occurrence = 0;
    for (int ordinal = 0; ordinal < launches; ++ordinal) {
        // Address and value selection depend only on the direction-local
        // occurrence, never on sequence ordinal or sequence identity.
        if (direction_at(*config.sequence, ordinal) == Direction::kRead) {
            std::size_t offset = (
                static_cast<std::size_t>(read_occurrence)
                * elements_per_window);
            a800_direction_read_phase<<<config.grid, kBlockSize>>>(
                read_arena + offset, elements_per_window);
            ++read_occurrence;
        } else if (direction_at(*config.sequence, ordinal)
                   == Direction::kWrite) {
            std::size_t offset = (
                static_cast<std::size_t>(write_occurrence)
                * elements_per_window);
            std::uint32_t token = direction_token(
                Direction::kWrite, write_occurrence, config.role->roles);
            a800_direction_write_phase<<<config.grid, kBlockSize>>>(
                write_arena + offset, elements_per_window,
                kWriteSalt ^ token);
            ++write_occurrence;
        } else {
            std::fprintf(stderr, "unknown phase direction\n");
            std::exit(EXIT_FAILURE);
        }
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0F) {
        std::fprintf(stderr, "invalid event time %.9g ms\n", elapsed_ms);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<std::uint64_t>(
        std::llround(static_cast<double>(elapsed_ms) * 1.0e6));
}

static unsigned long long validate_sequence(
        const CaseConfig& config, std::size_t elements_per_window,
        const uint4* read_arena, const uint4* write_arena,
        std::uint32_t read_arena_salt, unsigned long long* device_errors) {
    CUDA_CHECK(cudaMemset(device_errors, 0, sizeof(unsigned long long)));
    const int validation_grid = kExpectedSms * 4;
    const int launches = kPhasesPerSequence * config.repetitions;
    int read_occurrence = 0;
    int write_occurrence = 0;
    for (int ordinal = 0; ordinal < launches; ++ordinal) {
        if (direction_at(*config.sequence, ordinal) == Direction::kRead) {
            std::size_t offset = (
                static_cast<std::size_t>(read_occurrence)
                * elements_per_window);
            validate_read_window<<<validation_grid, kBlockSize>>>(
                read_arena + offset, elements_per_window,
                offset, read_arena_salt, device_errors);
            ++read_occurrence;
        } else if (direction_at(*config.sequence, ordinal)
                   == Direction::kWrite) {
            std::size_t offset = (
                static_cast<std::size_t>(write_occurrence)
                * elements_per_window);
            std::uint32_t token = direction_token(
                Direction::kWrite, write_occurrence, config.role->roles);
            validate_write_window<<<validation_grid, kBlockSize>>>(
                write_arena + offset, elements_per_window,
                kWriteSalt ^ token, device_errors);
            ++write_occurrence;
        } else {
            std::fprintf(stderr, "unknown validation direction\n");
            std::exit(EXIT_FAILURE);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long host_errors = 0;
    CUDA_CHECK(cudaMemcpy(
        &host_errors, device_errors, sizeof(host_errors),
        cudaMemcpyDeviceToHost));
    return host_errors;
}

static void prepare_trial(
        const CaseConfig& config, std::size_t elements_per_window,
        uint4* read_arena, uint4* write_arena,
        std::uint32_t read_arena_salt) {
    const int launches = kPhasesPerSequence * config.repetitions;
    const std::size_t active_elements = (
        static_cast<std::size_t>(launches) * elements_per_window);
    // Prepare every ordinal slot in both roles, even when the sequence does
    // not consume that slot in that role.  Preparation work and ordering are
    // therefore identical across all six sequence labels.
    initialize_read_window<<<kExpectedSms * 4, kBlockSize>>>(
        read_arena, active_elements, 0, read_arena_salt);
    CUDA_CHECK(cudaMemset(
        write_arena, kSentinelByte, active_elements * sizeof(uint4)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void warm_up_phase_kernels(uint4* arena_a, uint4* arena_b) {
    constexpr std::size_t warmup_bytes = 64ULL * kMiB;
    constexpr std::size_t warmup_elements = warmup_bytes / sizeof(uint4);
    constexpr int warmup_grid = kExpectedSms;
    initialize_read_window<<<kExpectedSms * 4, kBlockSize>>>(
        arena_a, warmup_elements, 0, kArenaASalt);
    CUDA_CHECK(cudaMemset(arena_b, kSentinelByte, warmup_bytes));
    a800_direction_read_phase<<<warmup_grid, kBlockSize>>>(
        arena_a, warmup_elements);
    a800_direction_write_phase<<<warmup_grid, kBlockSize>>>(
        arena_b, warmup_elements,
        kWriteSalt ^ direction_token(
            Direction::kWrite, 0, AddressRoles::kAReadBWrite));
    initialize_read_window<<<kExpectedSms * 4, kBlockSize>>>(
        arena_b, warmup_elements, 0, kArenaBSalt);
    CUDA_CHECK(cudaMemset(arena_a, kSentinelByte, warmup_bytes));
    a800_direction_read_phase<<<warmup_grid, kBlockSize>>>(
        arena_b, warmup_elements);
    a800_direction_write_phase<<<warmup_grid, kBlockSize>>>(
        arena_a, warmup_elements,
        kWriteSalt ^ direction_token(
            Direction::kWrite, 0, AddressRoles::kBReadAWrite));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void run_trial(
        const CaseConfig& config, int trial, int sequence_order_slot,
        uint4* arena_a, uint4* arena_b,
        unsigned long long* device_errors) {
    const std::size_t q_bytes = (
        static_cast<std::size_t>(config.q_mib) * kMiB);
    const std::size_t elements_per_window = q_bytes / sizeof(uint4);
    if (q_bytes <= static_cast<std::size_t>(kExpectedL2Bytes)
            || elements_per_window * sizeof(uint4) != q_bytes) {
        std::fprintf(stderr, "invalid phase footprint %zu bytes\n", q_bytes);
        std::exit(EXIT_FAILURE);
    }
    uint4* read_arena = nullptr;
    uint4* write_arena = nullptr;
    std::uint32_t read_arena_salt = 0;
    select_arenas(
        config.role->roles, arena_a, arena_b,
        &read_arena, &write_arena, &read_arena_salt);

    int read_launches = 0;
    int write_launches = 0;
    const int launches = kPhasesPerSequence * config.repetitions;
    for (int ordinal = 0; ordinal < launches; ++ordinal) {
        if (direction_at(*config.sequence, ordinal) == Direction::kRead) {
            ++read_launches;
        } else if (direction_at(*config.sequence, ordinal)
                   == Direction::kWrite) {
            ++write_launches;
        } else {
            std::fprintf(stderr, "unknown accounting direction\n");
            std::exit(EXIT_FAILURE);
        }
    }
    const std::size_t logical_read_bytes = (
        static_cast<std::size_t>(read_launches) * q_bytes);
    const std::size_t logical_write_bytes = (
        static_cast<std::size_t>(write_launches) * q_bytes);
    TransitionCounts transitions = count_program_transitions(
        *config.sequence, config.repetitions);

    prepare_trial(
        config, elements_per_window, read_arena, write_arena,
        read_arena_salt);
    std::uint64_t event_ns = measure_sequence(
        config, elements_per_window, read_arena, write_arena);
    unsigned long long errors = validate_sequence(
        config, elements_per_window, read_arena, write_arena,
        read_arena_salt, device_errors);
    const char* validation = errors == 0 ? "pass" : "fail";
    std::printf(
        "%s,%d,%d,%d,%s,%d,%d,%d,%" PRIu64 ",%zu,%zu,%d,%d,%d,%s\n",
        config.sequence->name, config.q_mib, config.repetitions,
        config.grid, config.role->output_name, trial, sequence_order_slot,
        launches, event_ns, logical_read_bytes, logical_write_bytes,
        transitions.read_to_write, transitions.write_to_read,
        transitions.read_to_write + transitions.write_to_read,
        validation);
    std::fflush(stdout);
    if (errors != 0) {
        std::fprintf(
            stderr,
            "validation failed sequence=%s q_mib=%d M=%d grid=%d "
            "address_roles=%s trial=%d errors=%llu\n",
            config.sequence->name, config.q_mib, config.repetitions,
            config.grid, config.role->output_name, trial, errors);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    bool single_case = false;
    CaseConfig selected = {nullptr, 0, 0, 0, nullptr};
    if (argc == 1) {
        single_case = false;
    } else if (argc == 2 && std::strcmp(argv[1], "--describe") == 0) {
        describe(argv[0]);
        return EXIT_SUCCESS;
    } else if (argc == 7 && std::strcmp(argv[1], "--single") == 0) {
        single_case = true;
        selected = parse_single_case(argv);
    } else {
        describe(argv[0]);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    constexpr std::size_t minimum_global_bytes = 79ULL * 1024 * 1024 * 1024;
    constexpr std::size_t maximum_global_bytes = 80ULL * 1024 * 1024 * 1024;
    if (std::strstr(properties.name, "A800") == nullptr
            || properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != kExpectedSms
            || properties.warpSize != kExpectedWarpSize
            || properties.l2CacheSize != kExpectedL2Bytes
            || properties.totalGlobalMem < minimum_global_bytes
            || properties.totalGlobalMem > maximum_global_bytes) {
        std::fprintf(
            stderr,
            "unsupported device name=%s cc=%d.%d sms=%d warp=%d l2=%d "
            "global=%zu\n",
            properties.name, properties.major, properties.minor,
            properties.multiProcessorCount, properties.warpSize,
            properties.l2CacheSize, properties.totalGlobalMem);
        return EXIT_FAILURE;
    }

    constexpr int q_values[] = {64, 256};
    constexpr int repetition_values[] = {1, 2, 4};
    constexpr int grid_values[] = {108, 216, 432};
    constexpr int max_q_mib = 256;
    constexpr int max_repetitions = 4;
    constexpr int max_launches = kPhasesPerSequence * max_repetitions;
    const std::size_t arena_bytes = (
        static_cast<std::size_t>(max_q_mib) * kMiB * max_launches);

    uint4* arena_a = nullptr;
    uint4* arena_b = nullptr;
    unsigned long long* device_errors = nullptr;
    CUDA_CHECK(cudaMalloc(&arena_a, arena_bytes));
    CUDA_CHECK(cudaMalloc(&arena_b, arena_bytes));
    CUDA_CHECK(cudaMalloc(&device_errors, sizeof(unsigned long long)));
    warm_up_phase_kernels(arena_a, arena_b);

    std::fprintf(
        stderr,
        "device=%s cc=%d.%d sms=%d warp=%d l2_bytes=%d global_bytes=%zu "
        "arena_bytes_each=%zu trials=%d block=%d schedule=cyclic_fixed\n",
        properties.name, properties.major, properties.minor,
        properties.multiProcessorCount, properties.warpSize,
        properties.l2CacheSize, properties.totalGlobalMem,
        arena_bytes, kTrials, kBlockSize);
    std::printf(
        "sequence,q_mib,M,grid,address_roles,trial,sequence_order_slot,"
        "phase_launches,event_ns,"
        "logical_read_bytes,logical_write_bytes,program_read_to_write,"
        "program_write_to_read,program_transition_count,buffer_validation\n");

    if (single_case) {
        for (int trial = 0; trial < kTrials; ++trial) {
            run_trial(
                selected, trial, 0, arena_a, arena_b, device_errors);
        }
    } else {
        for (int trial = 0; trial < kTrials; ++trial) {
            for (int q_mib : q_values) {
                for (int repetitions : repetition_values) {
                    for (int grid : grid_values) {
                        for (const RoleSpec& role : kRoleSpecs) {
                            for (int slot = 0;
                                 slot < kSequenceCount; ++slot) {
                                const SequenceSpec& sequence = kSequences[
                                    kSequenceOrder[trial][slot]];
                                CaseConfig config = {
                                    &sequence, q_mib, repetitions,
                                    grid, &role};
                                run_trial(
                                    config, trial, slot, arena_a, arena_b,
                                    device_errors);
                            }
                        }
                    }
                }
            }
        }
    }

    CUDA_CHECK(cudaFree(arena_a));
    CUDA_CHECK(cudaFree(arena_b));
    CUDA_CHECK(cudaFree(device_errors));
    std::fprintf(stderr, "all_validation=pass\n");
    return EXIT_SUCCESS;
}
