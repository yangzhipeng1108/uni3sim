/*
 * MetaX C500 issued two-phase direction-transition microbenchmark.
 *
 * This benchmark emits matched raw event observations for four epochs:
 * RW|R-precondition, WR|W-precondition, RR|R-precondition, and
 * WW|W-precondition.  A synchronized precondition launch precedes the start
 * event.  Exactly two phase launches occur between the start/end events;
 * initialization, synchronization, warmup, and correctness validation are
 * outside that interval.
 *
 * Stable extern-C symbols R_b and W_b exist for b={1,4,8}.  Their outer loop
 * is explicitly not unrolled and their inner body is statically unrolled:
 * R_b first loads b slot-major float4 values into independent registers,
 * then uses a register-only reduction and unreachable trap guard with no
 * global-memory side effect; W_b
 * first generates b address/iteration-dependent float4 values in registers,
 * then stores the b slot-major values.  Within each static slot, adjacent
 * lanes access adjacent float4 values.  Timed R_b is load-only: it has no
 * sink, global store, or intended helper call.  Exact mxobjdump evidence from
 * the collected binary is the final gate for b consecutive ldg_b128/stg_b128
 * devc instructions, absence of spills/helper calls, and W_b load absence.
 * Source structure or another ISA's disassembly is not that evidence.
 *
 * Four disjoint 256-MiB read allocations and four disjoint 256-MiB write
 * allocations exceed the reported 8-MiB L2.  B={128,256} MiB, so even the
 * b=8/grid=1664 case assigns at least one slot item to every launched thread.
 * Trial/sequence/phase rotations select buffer roles and, for B=128 MiB, one
 * of two disjoint offsets.  Every sequence/phase slot covers the same joint
 * (buffer role, offset) set instead of merely balancing the two marginals or
 * assigning a fixed physical region to a sequence.  Read initialization
 * and W_b values are nonconstant deterministic functions of absolute logical
 * address (and W iteration identity), allowing event-external validators to
 * detect source corruption and address permutation.
 *
 * Capture contract:
 *   - Do not use mcTracer or mcProfiler.  Use ordinary MACA events only.
 *   - Record the explicit run_id, exact source commit and SHA-256, binary
 *     SHA-256, mxcc version and flags, full mxobjdump SHA-256, device/runtime
 *     identity, and the command line beside the CSV.
 *   - Before and after capture record mx-smi isolation, clocks, temperature,
 *     power, and memory telemetry.  Reject a run with contention or clock
 *     state changes; never silently normalize it.
 *   - Audit every R_b/W_b symbol in the exact disassembly before interpreting
 *     event data.  The program does not perform this audit or infer a result.
 *
 * This independent mechanism benchmark does not reproduce a production
 * operator.  It can measure at most issued two-phase event excess for the
 * exact audited MACA lowering.  It cannot establish physical DRAM turnaround,
 * transactions, request credits, or undocumented controller behavior.  Grid
 * is only the declared launch size; it is not interpreted as a resident
 * producer count, concurrency level, or request-credit count.
 * Subsequent audit may form matched references such as
 *
 *   T_RW|R - (T_RR|R + T_WW|W) / 2
 *   T_WR|W - (T_RR|R + T_WW|W) / 2
 *
 * but this program emits raw quartets and does not make that inference.
 *
 * Build inside the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       -o direction_transition direction_transition.cpp
 *
 * Run the complete sweep or one (b, B, grid) configuration:
 *   MACA_VISIBLE_DEVICES=0 ./direction_transition --run-id RUN_ID
 *   MACA_VISIBLE_DEVICES=0 ./direction_transition --run-id RUN_ID \
 *       --single 4 256 416
 */

#include <mc_runtime.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do {                                                   \
    mcError_t error = (call);                                              \
    if (error != mcSuccess) {                                              \
        std::fprintf(stderr, "MACA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, mcGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kGiB = 1024ULL * kMiB;
constexpr std::size_t kAllocationBytes = 256ULL * kMiB;
constexpr std::size_t kExpectedL2Bytes = 8ULL * kMiB;
constexpr int kBlockSize = 256;
constexpr int kExpectedAps = 104;
constexpr int kExpectedWaveSize = 64;
constexpr int kTrials = 8;
constexpr int kOrderSize = 4;
constexpr int kBufferRoleCount = 4;
constexpr int kValidationGrid = kExpectedAps * 8;
constexpr std::uint32_t kReadSalt = 0x31415926U;
constexpr std::uint32_t kWriteSalt = 0x9e3779b9U;
constexpr int kWriteSentinelByte = 0xa5;
constexpr int kBursts[] = {1, 4, 8};
constexpr int kFootprintMiB[] = {128, 256};
constexpr int kGrids[] = {
    kExpectedAps, kExpectedAps * 4, kExpectedAps * 16,
};

static_assert(sizeof(float4) == 16, "float4 must be 16 bytes");
static_assert(alignof(float4) == 16, "float4 must be 16-byte aligned");
static_assert(kAllocationBytes > kExpectedL2Bytes,
              "every phase allocation must exceed L2");
static_assert(kAllocationBytes % 256 == 0,
              "allocation and phase offsets must preserve 256B alignment");
static_assert(kBlockSize % 16 == 0,
              "blocks must contain complete 16-lane subgroups");

enum class Direction {
    kRead,
    kWrite,
};

struct SequenceSpec {
    const char* name;
    Direction precondition;
    Direction first;
    Direction second;
};

constexpr SequenceSpec kSequences[] = {
    {"RW", Direction::kRead, Direction::kRead, Direction::kWrite},
    {"WR", Direction::kWrite, Direction::kWrite, Direction::kRead},
    {"RR", Direction::kRead, Direction::kRead, Direction::kRead},
    {"WW", Direction::kWrite, Direction::kWrite, Direction::kWrite},
};

// Forward rotations followed by reverse rotations.  Across eight trials each
// sequence occupies each order position twice, while both traversal directions
// are represented.
constexpr int kTrialOrder[kTrials][kOrderSize] = {
    {0, 1, 2, 3},
    {1, 2, 3, 0},
    {2, 3, 0, 1},
    {3, 0, 1, 2},
    {0, 3, 2, 1},
    {1, 0, 3, 2},
    {2, 1, 0, 3},
    {3, 2, 1, 0},
};

// The first four trials rotate every role at one parity-derived 128-MiB
// offset; the second four revisit every role at the complementary offset.
// Therefore every sequence/phase slot covers all eight joint (role, offset)
// regions once when two offsets exist, or every role twice for a 256-MiB
// phase.  Sequence and phase shifts only permute that common joint set.
constexpr int kTrialRoleRotation[kTrials] = {0, 1, 2, 3, 0, 1, 2, 3};
constexpr int kTrialOffsetBank[kTrials] = {0, 0, 0, 0, 1, 1, 1, 1};

struct BufferPool {
    float4* read[kBufferRoleCount];
    float4* write[kBufferRoleCount];
};

struct PhaseBinding {
    Direction direction;
    int buffer_role;
    std::size_t offset_bytes;
    std::uint32_t iteration;
};

struct EpochBindings {
    PhaseBinding precondition;
    PhaseBinding first;
    PhaseBinding second;
};

static int parse_positive_integer(const char* text, const char* name) {
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0'
            || value <= 0 || value > INT_MAX) {
        std::fprintf(stderr, "%s must be a positive integer, got %s\n",
                     name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

template <std::size_t N>
static bool contains(const int (&values)[N], int candidate) {
    for (int value : values) {
        if (value == candidate) {
            return true;
        }
    }
    return false;
}

static void validate_run_id(const char* run_id) {
    std::size_t length = std::strlen(run_id);
    if (length == 0 || length > 96) {
        std::fprintf(stderr, "run_id length must be in [1,96]\n");
        std::exit(EXIT_FAILURE);
    }
    for (std::size_t index = 0; index < length; ++index) {
        char value = run_id[index];
        bool valid = (
            (value >= 'a' && value <= 'z')
            || (value >= 'A' && value <= 'Z')
            || (value >= '0' && value <= '9')
            || value == '.' || value == '_' || value == '-');
        if (!valid) {
            std::fprintf(stderr,
                         "run_id contains unsupported character at %zu\n",
                         index);
            std::exit(EXIT_FAILURE);
        }
    }
}

__host__ __device__ __forceinline__ std::uint32_t mix_bits(
        std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__host__ __device__ __forceinline__ float finite_float(
        std::uint32_t bits) {
    union {
        std::uint32_t bits;
        float value;
    } converted;
    converted.bits = 0x3f000000U | (bits & 0x007fffffU);
    return converted.value;
}

__host__ __device__ __forceinline__ float4 make_pattern(
        std::uint32_t absolute_index, std::uint32_t salt) {
    std::uint32_t x = mix_bits(absolute_index ^ salt);
    std::uint32_t y = mix_bits(x ^ 0x68bc21ebU);
    std::uint32_t z = mix_bits(y ^ 0x02e5be93U);
    std::uint32_t w = mix_bits(z ^ 0x967a889bU);
    return make_float4(
        finite_float(x), finite_float(y), finite_float(z), finite_float(w));
}

template <int Burst>
__device__ __forceinline__ void read_phase_body(
        const float4* source, long long segment_vectors) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long thread_count = (
        static_cast<long long>(gridDim.x) * blockDim.x);
    float4 values[Burst];
    #pragma unroll 1
    for (long long index_in_segment = thread;
         index_in_segment < segment_vectors;
         index_in_segment += thread_count) {
        // Slot-major layout: slot owns one segment_vectors-long contiguous
        // region.  For a fixed static slot, adjacent lanes differ by exactly
        // one float4 (16 bytes).  A 16-lane subgroup begins at a global thread
        // multiple of 16; segment bases are required to be 128B-aligned, so
        // every subgroup's first address is also 128B-aligned.
        #pragma unroll
        for (int slot = 0; slot < Burst; ++slot) {
            long long index = (
                static_cast<long long>(slot) * segment_vectors
                + index_in_segment);
            values[slot] = source[index];
        }
        // Keep all loaded registers live past a compiler memory barrier, then
        // consume them only after the complete static load group.  Initialized
        // values are finite and positive, so the trap guard is unreachable for
        // valid input.  It gives the backend a real register dependency without
        // introducing a global store.  Exact disassembly must still reject
        // spills, helper calls, or reordered/non-vector loads.
        asm volatile("" ::: "memory");
        float consume = 0.0f;
        #pragma unroll
        for (int slot = 0; slot < Burst; ++slot) {
            consume += values[slot].x;
            consume += values[slot].y;
            consume += values[slot].z;
            consume += values[slot].w;
        }
        if (consume < 0.0f) {
            __builtin_trap();
        }
    }
}

template <int Burst>
__device__ __forceinline__ void write_phase_body(
        float4* destination, long long segment_vectors,
        long long base_vector, std::uint32_t iteration) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long thread_count = (
        static_cast<long long>(gridDim.x) * blockDim.x);
    float4 values[Burst];
    #pragma unroll 1
    for (long long index_in_segment = thread;
         index_in_segment < segment_vectors;
         index_in_segment += thread_count) {
        #pragma unroll
        for (int slot = 0; slot < Burst; ++slot) {
            long long index = (
                static_cast<long long>(slot) * segment_vectors
                + index_in_segment);
            long long absolute_index = base_vector + index;
            values[slot] = make_pattern(
                static_cast<std::uint32_t>(absolute_index),
                kWriteSalt ^ iteration);
        }
        asm volatile("" ::: "memory");
        #pragma unroll
        for (int slot = 0; slot < Burst; ++slot) {
            long long index = (
                static_cast<long long>(slot) * segment_vectors
                + index_in_segment);
            destination[index] = values[slot];
        }
    }
}

// Only these six stable symbols may be launched inside a timed phase pair.
extern "C" __global__ void R_1(
        const float4* source, long long segment_vectors) {
    read_phase_body<1>(source, segment_vectors);
}

extern "C" __global__ void R_4(
        const float4* source, long long segment_vectors) {
    read_phase_body<4>(source, segment_vectors);
}

extern "C" __global__ void R_8(
        const float4* source, long long segment_vectors) {
    read_phase_body<8>(source, segment_vectors);
}

extern "C" __global__ void W_1(
        float4* destination, long long segment_vectors,
        long long base_vector, std::uint32_t iteration) {
    write_phase_body<1>(
        destination, segment_vectors, base_vector, iteration);
}

extern "C" __global__ void W_4(
        float4* destination, long long segment_vectors,
        long long base_vector, std::uint32_t iteration) {
    write_phase_body<4>(
        destination, segment_vectors, base_vector, iteration);
}

extern "C" __global__ void W_8(
        float4* destination, long long segment_vectors,
        long long base_vector, std::uint32_t iteration) {
    write_phase_body<8>(
        destination, segment_vectors, base_vector, iteration);
}

__global__ void initialize_read_source(
        float4* buffer, long long vector_count, std::uint32_t salt) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(gridDim.x) * blockDim.x);
    for (long long index = thread;
         index < vector_count; index += stride) {
        buffer[index] = make_pattern(
            static_cast<std::uint32_t>(index), salt);
    }
}

__global__ void validate_read_source(
        const float4* source, long long vector_count,
        long long base_vector, std::uint32_t salt,
        unsigned long long* errors) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(gridDim.x) * blockDim.x);
    unsigned long long local_errors = 0;
    for (long long index = thread;
         index < vector_count; index += stride) {
        float4 observed = source[index];
        float4 expected = make_pattern(
            static_cast<std::uint32_t>(base_vector + index), salt);
        local_errors += (
            observed.x != expected.x || observed.y != expected.y
            || observed.z != expected.z || observed.w != expected.w);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

__global__ void validate_write_destination(
        const float4* destination, long long vector_count,
        long long base_vector, std::uint32_t iteration,
        unsigned long long* errors) {
    long long thread = (
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    long long stride = (
        static_cast<long long>(gridDim.x) * blockDim.x);
    unsigned long long local_errors = 0;
    for (long long index = thread;
         index < vector_count; index += stride) {
        float4 observed = destination[index];
        float4 expected = make_pattern(
            static_cast<std::uint32_t>(base_vector + index),
            kWriteSalt ^ iteration);
        local_errors += (
            observed.x != expected.x || observed.y != expected.y
            || observed.z != expected.z || observed.w != expected.w);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

static const char* direction_name(Direction direction) {
    if (direction == Direction::kRead) {
        return "R";
    } else if (direction == Direction::kWrite) {
        return "W";
    }
    std::fprintf(stderr, "unknown direction\n");
    std::exit(EXIT_FAILURE);
}

static int binding_role(int trial, int sequence_index, int slot) {
    return (
        kTrialRoleRotation[trial] + sequence_index + slot)
        % kBufferRoleCount;
}

static int binding_offset_index(
        int trial, int sequence_index, int slot, int offset_count) {
    int role = binding_role(trial, sequence_index, slot);
    return (role + kTrialOffsetBank[trial]) % offset_count;
}

static void validate_trial_order() {
    int position_counts[kOrderSize][kOrderSize] = {};
    for (int trial = 0; trial < kTrials; ++trial) {
        if (kTrialRoleRotation[trial] < 0
                || kTrialRoleRotation[trial] >= kBufferRoleCount
                || kTrialOffsetBank[trial] < 0
                || kTrialOffsetBank[trial] >= 2) {
            std::fprintf(stderr, "invalid trial address rotation %d\n", trial);
            std::exit(EXIT_FAILURE);
        }
        bool seen[kOrderSize] = {};
        for (int position = 0; position < kOrderSize; ++position) {
            int sequence = kTrialOrder[trial][position];
            if (sequence < 0 || sequence >= kOrderSize || seen[sequence]) {
                std::fprintf(stderr,
                             "invalid trial order trial=%d position=%d\n",
                             trial, position);
                std::exit(EXIT_FAILURE);
            }
            seen[sequence] = true;
            ++position_counts[position][sequence];
        }
    }
    for (int position = 0; position < kOrderSize; ++position) {
        for (int sequence = 0; sequence < kOrderSize; ++sequence) {
            if (position_counts[position][sequence] != 2) {
                std::fprintf(
                    stderr,
                    "unbalanced order position=%d sequence=%d count=%d\n",
                    position, sequence, position_counts[position][sequence]);
                std::exit(EXIT_FAILURE);
            }
        }
    }
    const int offset_counts[] = {1, 2};
    for (int offset_count : offset_counts) {
        int expected_count = kTrials / (kBufferRoleCount * offset_count);
        if (expected_count * kBufferRoleCount * offset_count != kTrials) {
            std::fprintf(stderr,
                         "trial count cannot balance joint address regions\n");
            std::exit(EXIT_FAILURE);
        }
        for (int sequence = 0; sequence < kOrderSize; ++sequence) {
            for (int slot = 0; slot < 3; ++slot) {
                int joint_counts[kBufferRoleCount][2] = {};
                for (int trial = 0; trial < kTrials; ++trial) {
                    int role = binding_role(trial, sequence, slot);
                    int offset = binding_offset_index(
                        trial, sequence, slot, offset_count);
                    ++joint_counts[role][offset];
                }
                for (int role = 0; role < kBufferRoleCount; ++role) {
                    for (int offset = 0; offset < offset_count; ++offset) {
                        if (joint_counts[role][offset] != expected_count) {
                            std::fprintf(
                                stderr,
                                "unbalanced joint address sequence=%d "
                                "slot=%d role=%d offset=%d count=%d\n",
                                sequence, slot, role, offset,
                                joint_counts[role][offset]);
                            std::exit(EXIT_FAILURE);
                        }
                    }
                }
            }
        }
    }
}

static std::uint32_t read_role_salt(int role) {
    return kReadSalt ^ (
        static_cast<std::uint32_t>(role + 1) * 0x85ebca6bU);
}

static std::uint32_t phase_iteration(
        int configuration_id, int quartet_id,
        int order_position, int phase_slot) {
    return (
        (static_cast<std::uint32_t>(configuration_id + 1) << 24)
        ^ (static_cast<std::uint32_t>(quartet_id + 1) << 12)
        ^ (static_cast<std::uint32_t>(order_position + 1) << 4)
        ^ static_cast<std::uint32_t>(phase_slot + 1));
}

static EpochBindings make_bindings(
        const SequenceSpec& sequence, int sequence_index,
        int trial, int configuration_id, int quartet_id,
        int order_position, std::size_t phase_bytes) {
    const Direction directions[] = {
        sequence.precondition, sequence.first, sequence.second,
    };
    PhaseBinding phases[3];
    int offset_count = static_cast<int>(kAllocationBytes / phase_bytes);
    if (offset_count != 1 && offset_count != 2) {
        std::fprintf(stderr, "unsupported offset count %d\n", offset_count);
        std::exit(EXIT_FAILURE);
    }
    for (int slot = 0; slot < 3; ++slot) {
        int role = binding_role(trial, sequence_index, slot);
        int offset_index = binding_offset_index(
            trial, sequence_index, slot, offset_count);
        phases[slot] = PhaseBinding{
            directions[slot], role,
            static_cast<std::size_t>(offset_index) * phase_bytes,
            phase_iteration(
                configuration_id, quartet_id, order_position, slot),
        };
    }
    return EpochBindings{phases[0], phases[1], phases[2]};
}

static float4* phase_pointer(
        const BufferPool& buffers, const PhaseBinding& binding) {
    float4* allocation = nullptr;
    if (binding.direction == Direction::kRead) {
        allocation = buffers.read[binding.buffer_role];
    } else if (binding.direction == Direction::kWrite) {
        allocation = buffers.write[binding.buffer_role];
    } else {
        std::fprintf(stderr, "unknown phase pointer direction\n");
        std::exit(EXIT_FAILURE);
    }
    return allocation + binding.offset_bytes / sizeof(float4);
}

static void launch_read(
        int burst, int grid, const float4* source,
        long long segment_vectors) {
    if (burst == 1) {
        R_1<<<grid, kBlockSize>>>(source, segment_vectors);
    } else if (burst == 4) {
        R_4<<<grid, kBlockSize>>>(source, segment_vectors);
    } else if (burst == 8) {
        R_8<<<grid, kBlockSize>>>(source, segment_vectors);
    } else {
        std::fprintf(stderr, "unsupported read burst %d\n", burst);
        std::exit(EXIT_FAILURE);
    }
}

static void launch_write(
        int burst, int grid, float4* destination,
        long long segment_vectors, long long base_vector,
        std::uint32_t iteration) {
    if (burst == 1) {
        W_1<<<grid, kBlockSize>>>(
            destination, segment_vectors, base_vector, iteration);
    } else if (burst == 4) {
        W_4<<<grid, kBlockSize>>>(
            destination, segment_vectors, base_vector, iteration);
    } else if (burst == 8) {
        W_8<<<grid, kBlockSize>>>(
            destination, segment_vectors, base_vector, iteration);
    } else {
        std::fprintf(stderr, "unsupported write burst %d\n", burst);
        std::exit(EXIT_FAILURE);
    }
}

static void launch_phase(
        const BufferPool& buffers, const PhaseBinding& binding,
        int burst, int grid, long long segment_vectors) {
    float4* pointer = phase_pointer(buffers, binding);
    long long base_vector = static_cast<long long>(
        binding.offset_bytes / sizeof(float4));
    if (binding.direction == Direction::kRead) {
        launch_read(burst, grid, pointer, segment_vectors);
    } else if (binding.direction == Direction::kWrite) {
        launch_write(
            burst, grid, pointer, segment_vectors,
            base_vector, binding.iteration);
    } else {
        std::fprintf(stderr, "unknown launch direction\n");
        std::exit(EXIT_FAILURE);
    }
}

static void collect_validation_errors(
        unsigned long long* device_errors, const char* label) {
    CHECK(mcDeviceSynchronize());
    unsigned long long errors = 0;
    CHECK(mcMemcpy(
        &errors, device_errors, sizeof(errors), mcMemcpyDeviceToHost));
    if (errors != 0) {
        std::fprintf(stderr,
                     "%s validation found %llu incorrect vectors\n",
                     label, errors);
        std::exit(EXIT_FAILURE);
    }
}

static void validate_phase(
        const BufferPool& buffers, const PhaseBinding& binding,
        std::size_t phase_bytes, unsigned long long* device_errors) {
    long long vector_count = static_cast<long long>(
        phase_bytes / sizeof(float4));
    long long base_vector = static_cast<long long>(
        binding.offset_bytes / sizeof(float4));
    CHECK(mcMemset(device_errors, 0, sizeof(unsigned long long)));
    float4* pointer = phase_pointer(buffers, binding);
    if (binding.direction == Direction::kRead) {
        validate_read_source<<<kValidationGrid, kBlockSize>>>(
            pointer, vector_count, base_vector,
            read_role_salt(binding.buffer_role), device_errors);
        collect_validation_errors(device_errors, "read source");
    } else if (binding.direction == Direction::kWrite) {
        validate_write_destination<<<kValidationGrid, kBlockSize>>>(
            pointer, vector_count, base_vector,
            binding.iteration, device_errors);
        collect_validation_errors(device_errors, "write destination");
    } else {
        std::fprintf(stderr, "unknown validation direction\n");
        std::exit(EXIT_FAILURE);
    }
}

static void prepare_epoch(
        const BufferPool& buffers, const EpochBindings& bindings,
        std::size_t phase_bytes) {
    const PhaseBinding phases[] = {
        bindings.precondition, bindings.first, bindings.second,
    };
    for (const PhaseBinding& phase : phases) {
        if (phase.direction == Direction::kWrite) {
            CHECK(mcMemset(
                phase_pointer(buffers, phase),
                kWriteSentinelByte, phase_bytes));
        } else if (phase.direction != Direction::kRead) {
            std::fprintf(stderr, "unknown preparation direction\n");
            std::exit(EXIT_FAILURE);
        }
    }
}

static long long validated_segment_vectors(
        std::size_t phase_bytes, int burst, int grid) {
    long long vector_count = static_cast<long long>(
        phase_bytes / sizeof(float4));
    if (vector_count % burst != 0) {
        std::fprintf(stderr, "phase vector count is not burst-aligned\n");
        std::exit(EXIT_FAILURE);
    }
    long long segment_vectors = vector_count / burst;
    long long active_threads = static_cast<long long>(grid) * kBlockSize;
    if (segment_vectors < active_threads
            || (segment_vectors * sizeof(float4)) % 128 != 0) {
        std::fprintf(
            stderr,
            "invalid slot geometry segment_vectors=%lld threads=%lld\n",
            segment_vectors, active_threads);
        std::exit(EXIT_FAILURE);
    }
    return segment_vectors;
}

static double run_epoch(
        const BufferPool& buffers, const EpochBindings& bindings,
        int burst, int grid, std::size_t phase_bytes,
        mcEvent_t start, mcEvent_t end,
        unsigned long long* device_errors) {
    long long segment_vectors = validated_segment_vectors(
        phase_bytes, burst, grid);
    prepare_epoch(buffers, bindings, phase_bytes);

    // Preamble and synchronization are outside the timed event.
    launch_phase(
        buffers, bindings.precondition, burst, grid, segment_vectors);
    CHECK(mcDeviceSynchronize());

    CHECK(mcEventRecord(start));
    // Timed contract: exactly these two phase launches are in the event.
    launch_phase(buffers, bindings.first, burst, grid, segment_vectors);
    launch_phase(buffers, bindings.second, burst, grid, segment_vectors);
    CHECK(mcEventRecord(end));

    CHECK(mcDeviceSynchronize());
    float elapsed_ms = 0.0f;
    CHECK(mcEventElapsedTime(&elapsed_ms, start, end));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
        std::fprintf(stderr, "invalid event time %.9g ms\n", elapsed_ms);
        std::exit(EXIT_FAILURE);
    }

    validate_phase(
        buffers, bindings.precondition, phase_bytes, device_errors);
    validate_phase(buffers, bindings.first, phase_bytes, device_errors);
    validate_phase(buffers, bindings.second, phase_bytes, device_errors);
    return static_cast<double>(elapsed_ms) * 1.0e6;
}

static PhaseBinding warmup_binding(
        Direction direction, int role, std::size_t offset_bytes,
        std::uint32_t iteration) {
    return PhaseBinding{direction, role, offset_bytes, iteration};
}

static void warmup_all_symbols(
        const BufferPool& buffers, int configuration_id,
        int grid, std::size_t phase_bytes,
        unsigned long long* device_errors) {
    long long vector_count = static_cast<long long>(
        phase_bytes / sizeof(float4));
    int offset_count = static_cast<int>(kAllocationBytes / phase_bytes);
    PhaseBinding reads[] = {
        warmup_binding(Direction::kRead, 0, 0, 0),
        warmup_binding(
            Direction::kRead, 1,
            static_cast<std::size_t>(1 % offset_count) * phase_bytes, 0),
        warmup_binding(
            Direction::kRead, 2,
            static_cast<std::size_t>(2 % offset_count) * phase_bytes, 0),
    };
    PhaseBinding writes[] = {
        warmup_binding(
            Direction::kWrite, 0, 0,
            phase_iteration(configuration_id, 0, 0, 0)),
        warmup_binding(
            Direction::kWrite, 1,
            static_cast<std::size_t>(1 % offset_count) * phase_bytes,
            phase_iteration(configuration_id, 0, 0, 1)),
        warmup_binding(
            Direction::kWrite, 2,
            static_cast<std::size_t>(2 % offset_count) * phase_bytes,
            phase_iteration(configuration_id, 0, 0, 2)),
    };
    for (const PhaseBinding& binding : writes) {
        CHECK(mcMemset(
            phase_pointer(buffers, binding),
            kWriteSentinelByte, phase_bytes));
    }
    launch_phase(buffers, reads[0], 1, grid, vector_count);
    launch_phase(buffers, reads[1], 4, grid, vector_count / 4);
    launch_phase(buffers, reads[2], 8, grid, vector_count / 8);
    launch_phase(buffers, writes[0], 1, grid, vector_count);
    launch_phase(buffers, writes[1], 4, grid, vector_count / 4);
    launch_phase(buffers, writes[2], 8, grid, vector_count / 8);
    CHECK(mcDeviceSynchronize());
    for (const PhaseBinding& binding : reads) {
        validate_phase(buffers, binding, phase_bytes, device_errors);
    }
    for (const PhaseBinding& binding : writes) {
        validate_phase(buffers, binding, phase_bytes, device_errors);
    }
}

static int configuration_id_for(int burst, int footprint_mib, int grid) {
    int burst_index = -1;
    int footprint_index = -1;
    int grid_index = -1;
    for (int index = 0; index < 3; ++index) {
        if (kBursts[index] == burst) {
            burst_index = index;
        }
        if (kGrids[index] == grid) {
            grid_index = index;
        }
    }
    for (int index = 0; index < 2; ++index) {
        if (kFootprintMiB[index] == footprint_mib) {
            footprint_index = index;
        }
    }
    if (burst_index < 0 || footprint_index < 0 || grid_index < 0) {
        std::fprintf(stderr, "unsupported configuration identity\n");
        std::exit(EXIT_FAILURE);
    }
    return burst_index * 6 + footprint_index * 3 + grid_index;
}

static void run_configuration(
        const char* run_id, int burst, int footprint_mib, int grid,
        const BufferPool& buffers, mcEvent_t start, mcEvent_t end,
        unsigned long long* device_errors) {
    std::size_t phase_bytes = static_cast<std::size_t>(footprint_mib) * kMiB;
    long long vector_count = static_cast<long long>(
        phase_bytes / sizeof(float4));
    if (phase_bytes > kAllocationBytes || phase_bytes <= kExpectedL2Bytes
            || static_cast<std::size_t>(vector_count) * sizeof(float4)
                != phase_bytes
            || vector_count % burst != 0) {
        std::fprintf(stderr, "invalid phase footprint %zu\n", phase_bytes);
        std::exit(EXIT_FAILURE);
    }
    int configuration_id = configuration_id_for(
        burst, footprint_mib, grid);
    // Warmup launches every symbol, so validate the all-symbol geometry before
    // the first configuration launch rather than only the selected b.
    for (int warmup_burst : kBursts) {
        validated_segment_vectors(phase_bytes, warmup_burst, grid);
    }
    warmup_all_symbols(
        buffers, configuration_id, grid, phase_bytes, device_errors);

    for (int trial = 0; trial < kTrials; ++trial) {
        int quartet_id = configuration_id * kTrials + trial;
        for (int order_position = 0;
             order_position < kOrderSize; ++order_position) {
            int sequence_index = kTrialOrder[trial][order_position];
            const SequenceSpec& sequence = kSequences[sequence_index];
            EpochBindings bindings = make_bindings(
                sequence, sequence_index, trial, configuration_id,
                quartet_id, order_position, phase_bytes);
            double event_ns = run_epoch(
                buffers, bindings, burst, grid, phase_bytes,
                start, end, device_errors);
            unsigned long long logical_read_bytes = 0;
            unsigned long long logical_write_bytes = 0;
            const Direction timed[] = {
                sequence.first, sequence.second,
            };
            for (Direction direction : timed) {
                if (direction == Direction::kRead) {
                    logical_read_bytes += phase_bytes;
                } else if (direction == Direction::kWrite) {
                    logical_write_bytes += phase_bytes;
                } else {
                    std::fprintf(stderr, "unknown timed direction\n");
                    std::exit(EXIT_FAILURE);
                }
            }
            std::printf(
                "%s,%d,%d,%d,%d,%zu,%d,%d,%s,%s,%.3f,%llu,%llu,0,"
                "%d,%zu,%d,%zu,%d,%zu,pass\n",
                run_id, configuration_id, quartet_id, order_position,
                burst, phase_bytes, grid, kBlockSize,
                direction_name(sequence.precondition), sequence.name,
                event_ns, logical_read_bytes, logical_write_bytes,
                bindings.precondition.buffer_role,
                bindings.precondition.offset_bytes,
                bindings.first.buffer_role, bindings.first.offset_bytes,
                bindings.second.buffer_role, bindings.second.offset_bytes);
            std::fflush(stdout);
        }
    }
}

static void allocate_buffer(float4** buffer) {
    CHECK(mcMalloc(buffer, kAllocationBytes));
    std::uintptr_t address = reinterpret_cast<std::uintptr_t>(*buffer);
    if (address % 256 != 0) {
        std::fprintf(stderr,
                     "device allocation is not 256B aligned: %p\n",
                     static_cast<void*>(*buffer));
        std::exit(EXIT_FAILURE);
    }
}

static void allocate_buffers(BufferPool* buffers) {
    for (int role = 0; role < kBufferRoleCount; ++role) {
        allocate_buffer(&buffers->read[role]);
        allocate_buffer(&buffers->write[role]);
    }
    const float4* all[kBufferRoleCount * 2];
    for (int role = 0; role < kBufferRoleCount; ++role) {
        all[role] = buffers->read[role];
        all[kBufferRoleCount + role] = buffers->write[role];
    }
    for (int left = 0; left < kBufferRoleCount * 2; ++left) {
        for (int right = left + 1;
             right < kBufferRoleCount * 2; ++right) {
            if (all[left] == all[right]) {
                std::fprintf(stderr,
                             "phase allocations %d and %d alias\n",
                             left, right);
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

static void initialize_and_validate_read_buffers(
        const BufferPool& buffers,
        unsigned long long* device_errors) {
    long long vector_count = static_cast<long long>(
        kAllocationBytes / sizeof(float4));
    for (int role = 0; role < kBufferRoleCount; ++role) {
        initialize_read_source<<<kValidationGrid, kBlockSize>>>(
            buffers.read[role], vector_count, read_role_salt(role));
    }
    CHECK(mcDeviceSynchronize());
    for (int role = 0; role < kBufferRoleCount; ++role) {
        PhaseBinding binding{
            Direction::kRead, role, 0, 0,
        };
        validate_phase(
            buffers, binding, kAllocationBytes, device_errors);
    }
}

static void free_buffers(const BufferPool& buffers) {
    for (int role = 0; role < kBufferRoleCount; ++role) {
        CHECK(mcFree(buffers.read[role]));
        CHECK(mcFree(buffers.write[role]));
    }
}

int main(int argc, char** argv) {
    const char* run_id = nullptr;
    bool single = false;
    int selected_burst = 0;
    int selected_footprint_mib = 0;
    int selected_grid = 0;
    if (argc == 3 && std::strcmp(argv[1], "--run-id") == 0) {
        run_id = argv[2];
    } else if (argc == 7
            && std::strcmp(argv[1], "--run-id") == 0
            && std::strcmp(argv[3], "--single") == 0) {
        run_id = argv[2];
        single = true;
        selected_burst = parse_positive_integer(argv[4], "b");
        selected_footprint_mib = parse_positive_integer(argv[5], "B_MiB");
        selected_grid = parse_positive_integer(argv[6], "grid");
        if (!contains(kBursts, selected_burst)) {
            std::fprintf(stderr, "b must be one of 1,4,8\n");
            return EXIT_FAILURE;
        }
        if (!contains(kFootprintMiB, selected_footprint_mib)) {
            std::fprintf(stderr, "B_MiB must be one of 128,256\n");
            return EXIT_FAILURE;
        }
        if (!contains(kGrids, selected_grid)) {
            std::fprintf(stderr, "grid must be one of 104,416,1664\n");
            return EXIT_FAILURE;
        }
    } else {
        std::fprintf(
            stderr,
            "usage: %s --run-id RUN_ID [--single b B_MiB grid]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    validate_run_id(run_id);
    validate_trial_order();

    mcDeviceProp_t properties;
    CHECK(mcGetDeviceProperties(&properties, 0));
    constexpr std::size_t minimum_global_bytes = 63ULL * kGiB;
    constexpr std::size_t maximum_global_bytes = 65ULL * kGiB;
    if (std::strstr(properties.name, "C500") == nullptr
            || properties.multiProcessorCount != kExpectedAps
            || properties.warpSize != kExpectedWaveSize
            || properties.l2CacheSize != kExpectedL2Bytes
            || properties.totalGlobalMem < minimum_global_bytes
            || properties.totalGlobalMem > maximum_global_bytes
            || properties.clockRate <= 0 || properties.memoryClockRate <= 0) {
        std::fprintf(
            stderr,
            "unsupported device name=%s aps=%d wave=%d l2=%d global=%zu "
            "clock_khz=%d memory_clock_khz=%d\n",
            properties.name, properties.multiProcessorCount,
            properties.warpSize, properties.l2CacheSize,
            properties.totalGlobalMem, properties.clockRate,
            properties.memoryClockRate);
        return EXIT_FAILURE;
    }

    BufferPool buffers{};
    allocate_buffers(&buffers);
    unsigned long long* device_errors = nullptr;
    CHECK(mcMalloc(&device_errors, sizeof(unsigned long long)));
    initialize_and_validate_read_buffers(buffers, device_errors);

    mcEvent_t start;
    mcEvent_t end;
    CHECK(mcEventCreate(&start));
    CHECK(mcEventCreate(&end));

    std::fprintf(
        stderr,
        "run_id=%s device=%s aps=%d wave=%d l2_bytes=%d global_bytes=%zu "
        "clock_khz=%d memory_clock_khz=%d allocation_bytes=%zu roles=%d "
        "block=%d trials=%d\n",
        run_id, properties.name, properties.multiProcessorCount,
        properties.warpSize, properties.l2CacheSize,
        properties.totalGlobalMem, properties.clockRate,
        properties.memoryClockRate, kAllocationBytes,
        kBufferRoleCount, kBlockSize, kTrials);
    std::printf(
        "run_id,configuration_id,quartet_id,order_position,b,"
        "phase_footprint_bytes,grid,block,precondition,sequence,event_ns,"
        "logical_read_bytes,logical_write_bytes,epilogue_bytes,"
        "precondition_buffer_role,precondition_offset_bytes,"
        "first_buffer_role,first_offset_bytes,second_buffer_role,"
        "second_offset_bytes,validation\n");

    if (single) {
        run_configuration(
            run_id, selected_burst, selected_footprint_mib, selected_grid,
            buffers, start, end, device_errors);
    } else {
        for (int burst : kBursts) {
            for (int footprint_mib : kFootprintMiB) {
                for (int grid : kGrids) {
                    run_configuration(
                        run_id, burst, footprint_mib, grid,
                        buffers, start, end, device_errors);
                }
            }
        }
    }

    CHECK(mcEventDestroy(start));
    CHECK(mcEventDestroy(end));
    CHECK(mcFree(device_errors));
    free_buffers(buffers);
    std::fprintf(stderr, "validation=pass\n");
    return EXIT_SUCCESS;
}
