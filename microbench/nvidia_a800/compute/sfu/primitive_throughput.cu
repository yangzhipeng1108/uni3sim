/*
 * NVIDIA A800 register-resident primitive latency/issue benchmark.
 *
 * Both latency and throughput kernels use the per-SM clock64 cycle domain.
 * Every thread records its local interval and every CTA records its SM ID
 * outside the timed window.  Throughput is aggregated from a separate
 * minimum-start / maximum-end interval on each SM; clock64 absolute values
 * are never compared across SMs.  The runner independently queries target
 * and control occupancy and supplies a common cooperative grid that this
 * binary validates against each specialization.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>


#define CHECK_CUDA(call) do {                                              \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                     __FILE__, __LINE__, cudaGetErrorString(error));        \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)


namespace cg = cooperative_groups;

constexpr int kWarpSize = 32;
constexpr int kSmCount = 108;
constexpr int kSubcoresPerSm = 4;
constexpr int kThroughputUnroll = 128;
constexpr int kWarmups = 5;
constexpr const char* kExpectedDeviceName = "NVIDIA A800-SXM4-80GB";
constexpr const char* kRawHeader =
    "trial,mode,primitive,clock_domain,device_name,device_uuid,pci_bus_id,"
    "device_property_clock_khz,sm_count,grid_blocks,block_threads,"
    "warps_per_cta,residency_cell,paired_resident_ctas_per_sm,"
    "specialization_occupancy_limit_ctas_per_sm,chains,unroll,"
    "timed_opcode_count_per_thread,issued_warp_opcodes,"
    "sm_clock_start_cycle,sm_clock_end_cycle,elapsed_sm_cycles,"
    "active_sm_count,min_local_elapsed_sm_cycles,"
    "max_local_elapsed_sm_cycles,sum_local_elapsed_sm_cycles,"
    "lane_ops_per_subcore_cycle,fma_flops_per_subcore_cycle,"
    "warp_instructions_per_subcore_cycle";
constexpr const char* kAbiLine =
    "abi,primitive_throughput,6,trial,mode,primitive,clock_domain,"
    "device_name,device_uuid,pci_bus_id,device_property_clock_khz,sm_count,"
    "grid_blocks,block_threads,warps_per_cta,residency_cell,"
    "paired_resident_ctas_per_sm,"
    "specialization_occupancy_limit_ctas_per_sm,chains,unroll,"
    "timed_opcode_count_per_thread,issued_warp_opcodes,"
    "sm_clock_start_cycle,sm_clock_end_cycle,elapsed_sm_cycles,"
    "active_sm_count,min_local_elapsed_sm_cycles,"
    "max_local_elapsed_sm_cycles,sum_local_elapsed_sm_cycles,"
    "lane_ops_per_subcore_cycle,fma_flops_per_subcore_cycle,"
    "warp_instructions_per_subcore_cycle";


enum Primitive {
    FMA_F32 = 0,
    ADD_F32 = 1,
    EX2_APPROX_F32 = 2,
    RCP_APPROX_F32 = 3,
    RSQRT_APPROX_F32 = 4,
    CONTROL_XOR_B32 = 5,
    MUL_F32 = 6,
};


template <Primitive primitive>
struct Instruction;


template <>
struct Instruction<FMA_F32> {
    template <int step>
    __device__ __forceinline__ static float dependent(float value) {
        float output;
        asm volatile(
            "fma.rn.f32 %0, %1, 0f3f000100, %1;"
            : "=f"(output) : "f"(value));
        return output;
    }
    template <int step>
    __device__ __forceinline__ static void independent(
        float& value, float) {
        asm volatile(
            "fma.rn.f32 %0, %0, 0f3f000100, %0;"
            : "+f"(value));
    }
};


template <>
struct Instruction<ADD_F32> {
    template <int step>
    __device__ __forceinline__ static float dependent(float value) {
        float output;
        asm volatile(
            "add.rn.f32 %0, %1, %2;"
            : "=f"(output) : "f"(value), "f"(0.0000152587890625f));
        return output;
    }
    template <int step>
    __device__ __forceinline__ static void independent(
            float& value, float) {
        asm volatile(
            "add.rn.f32 %0, %0, %1;"
            : "+f"(value) : "f"(0.0000152587890625f));
    }
};


template <>
struct Instruction<MUL_F32> {
    template <int step>
    __device__ __forceinline__ static float dependent(float value) {
        float output;
        asm volatile(
            "mul.rn.f32 %0, %1, 0f3f000100;"
            : "=f"(output) : "f"(value));
        return output;
    }
};


template <>
struct Instruction<EX2_APPROX_F32> {
    template <int step>
    __device__ __forceinline__ static void independent(
            float& value, float) {
        asm volatile(
            "ex2.approx.ftz.f32 %0, %0;"
            : "+f"(value));
    }
};


template <>
struct Instruction<RCP_APPROX_F32> {
    template <int step>
    __device__ __forceinline__ static float dependent(float value) {
        float output;
        asm volatile(
            "rcp.approx.ftz.f32 %0, %1;"
            : "=f"(output) : "f"(value));
        return output;
    }
    template <int step>
    __device__ __forceinline__ static void independent(
            float& value, float) {
        asm volatile(
            "rcp.approx.ftz.f32 %0, %0;"
            : "+f"(value));
    }
};


template <>
struct Instruction<RSQRT_APPROX_F32> {
    template <int step>
    __device__ __forceinline__ static float dependent(float value) {
        float output;
        asm volatile(
            "rsqrt.approx.ftz.f32 %0, %1;"
            : "=f"(output) : "f"(value));
        return output;
    }
    template <int step>
    __device__ __forceinline__ static void independent(
            float& value, float) {
        asm volatile(
            "rsqrt.approx.ftz.f32 %0, %0;"
            : "+f"(value));
    }
};


template <>
struct Instruction<CONTROL_XOR_B32> {
    template <int step>
    __device__ __forceinline__ static float dependent(float value) {
        float output;
        asm volatile(
            "xor.b32 %0, %1, %2;"
            : "=f"(output)
            : "f"(value), "n"(0x9e3779b9u ^ (step * 0x01010101u)));
        return output;
    }
    template <int step>
    __device__ __forceinline__ static void independent(
            float& value, float salt) {
        asm volatile(
            "lop3.b32 %0, %0, %1, %2, 0xe8;"
            : "+f"(value)
            : "f"(salt),
              "n"(0x9e3779b9u ^ (step * 0x01010101u)));
    }
};


template <Primitive primitive, int unroll, std::size_t... steps>
__device__ __forceinline__ void apply_dependent_observable(
        float (&values)[unroll + 1], std::index_sequence<steps...>) {
    ((values[steps + 1] =
          Instruction<primitive>::template dependent<steps>(values[steps])),
     ...);
}


template <Primitive primitive, int chains, std::size_t step>
__device__ __forceinline__ void apply_throughput_step(
        float (&values)[chains], const float (&salts)[chains]) {
    constexpr int chain = step % chains;
    Instruction<primitive>::template independent<step>(
        values[chain], salts[chain]);
}


template <Primitive primitive, int chains, std::size_t... steps>
__device__ __forceinline__ void apply_throughput_chains(
        float (&values)[chains], const float (&salts)[chains],
        std::index_sequence<steps...>) {
    (apply_throughput_step<primitive, chains, steps>(values, salts), ...);
}


__device__ __forceinline__ std::uint64_t read_clock64() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value));
    return value;
}


__device__ __forceinline__ std::uint64_t read_clock64_after(
        float dependency) {
    std::uint64_t value;
    std::uint32_t dependency_bits = __float_as_uint(dependency);
    asm volatile(
        "{ .reg .u32 endpoint_xor;\n"
        "  .reg .pred endpoint_ready;\n"
        "  xor.b32 endpoint_xor, %1, 0x6a09e667;\n"
        "  setp.ne.u32 endpoint_ready, %1, endpoint_xor;\n"
        "  @!endpoint_ready exit;\n"
        "  mov.u64 %0, %%clock64;\n"
        "}"
        : "=l"(value) : "r"(dependency_bits));
    return value;
}


__device__ __forceinline__ std::uint32_t read_smid() {
    std::uint32_t value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}


template <Primitive primitive, int unroll>
__global__ void dependent_kernel(
        std::uint64_t* thread_starts, std::uint64_t* thread_ends,
        std::uint32_t* block_start_smids, std::uint32_t* block_end_smids,
        const float* inputs, float* observations) {
    static_assert(
        unroll == 4 || unroll == 8 || unroll == 16 || unroll == 32,
        "unsupported dependent unroll");
    std::size_t thread_offset = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    float values[unroll + 1];
    values[0] = inputs[thread_offset];
    if (threadIdx.x == 0) {
        block_start_smids[blockIdx.x] = read_smid();
    }
    cg::this_grid().sync();
    std::uint64_t start = read_clock64();
    apply_dependent_observable<primitive, unroll>(
        values, std::make_index_sequence<unroll>{});
    std::uint64_t end = read_clock64_after(values[unroll]);
    thread_starts[thread_offset] = start;
    thread_ends[thread_offset] = end;
    std::size_t observation_offset = thread_offset * unroll;
    #pragma unroll
    for (int step = 0; step < unroll; ++step) {
        observations[observation_offset + step] = values[step + 1];
    }
    if (threadIdx.x == 0) {
        block_end_smids[blockIdx.x] = read_smid();
    }
}


template <Primitive primitive, int chains>
__global__ void throughput_kernel(
        std::uint64_t* thread_starts, std::uint64_t* thread_ends,
        std::uint32_t* block_start_smids, std::uint32_t* block_end_smids,
        const float* inputs, float* observations) {
    static_assert(chains == 1 || chains == 2 || chains == 4
                  || chains == 8 || chains == 16,
                  "unsupported independent chain count");
    std::size_t thread_offset = (
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
    std::size_t observation_offset = thread_offset * chains;
    float values[chains];
    float salts[chains];
    #pragma unroll
    for (int chain = 0; chain < chains; ++chain) {
        std::size_t input_offset = thread_offset * 2 * chains;
        values[chain] = inputs[input_offset + chain];
        salts[chain] = inputs[input_offset + chains + chain];
    }
    if (threadIdx.x == 0) {
        block_start_smids[blockIdx.x] = read_smid();
    }
    cg::this_grid().sync();
    std::uint64_t start = read_clock64();
    apply_throughput_chains<primitive, chains>(
        values, salts,
        std::make_index_sequence<kThroughputUnroll * chains>{});
    std::uint64_t end = read_clock64();
    thread_starts[thread_offset] = start;
    thread_ends[thread_offset] = end;
    #pragma unroll
    for (int chain = 0; chain < chains; ++chain) {
        observations[observation_offset + chain] = values[chain];
    }
    if (threadIdx.x == 0) {
        block_end_smids[blockIdx.x] = read_smid();
    }
}


struct Arguments {
    const char* action;
    Primitive primitive;
    const char* primitive_text;
    const char* mode;
    const char* residency_cell;
    std::string expected_device_uuid;
    std::string expected_pci_bus_id;
    int unroll;
    int chains;
    int warps;
    int trials;
    int grid_blocks;
    int paired_resident_ctas_per_sm;
};


[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(EXIT_FAILURE);
}


int parse_positive_integer(const char* text, const char* name) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 1048576) {
        std::fprintf(stderr, "%s must be a positive bounded integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}


Primitive parse_primitive(const char* text) {
    if (std::strcmp(text, "fma_f32") == 0) return FMA_F32;
    if (std::strcmp(text, "add_f32") == 0) return ADD_F32;
    if (std::strcmp(text, "ex2_approx_f32") == 0) return EX2_APPROX_F32;
    if (std::strcmp(text, "rcp_approx_f32") == 0) return RCP_APPROX_F32;
    if (std::strcmp(text, "rsqrt_approx_f32") == 0) return RSQRT_APPROX_F32;
    if (std::strcmp(text, "control_xor_b32") == 0) return CONTROL_XOR_B32;
    if (std::strcmp(text, "mul_f32") == 0) return MUL_F32;
    fail("--primitive has an unknown value");
}


Arguments parse_arguments(int argc, char** argv) {
    if (argc != 25) {
        fail(
            "usage: primitive_throughput --action occupancy|run "
            "--primitive NAME --mode MODE --unroll N --chains N --warps N "
            "--residency one|max --trials N --expected-device-uuid UUID "
            "--expected-pci-bus-id PCI --grid-blocks N "
            "--paired-resident-ctas-per-sm N");
    }
    const char* action = nullptr;
    const char* primitive_text = nullptr;
    const char* mode = nullptr;
    const char* residency_cell = nullptr;
    const char* expected_device_uuid = nullptr;
    const char* expected_pci_bus_id = nullptr;
    int unroll = 0;
    int chains = 0;
    int warps = 0;
    int trials = 0;
    int grid_blocks = 0;
    int paired_resident = 0;
    for (int index = 1; index < argc; index += 2) {
        const char* option = argv[index];
        const char* value = argv[index + 1];
        if (std::strcmp(option, "--action") == 0 && action == nullptr) {
            action = value;
        } else if (std::strcmp(option, "--primitive") == 0
                   && primitive_text == nullptr) {
            primitive_text = value;
        } else if (std::strcmp(option, "--mode") == 0 && mode == nullptr) {
            mode = value;
        } else if (std::strcmp(option, "--unroll") == 0 && unroll == 0) {
            unroll = parse_positive_integer(value, "--unroll");
        } else if (std::strcmp(option, "--chains") == 0 && chains == 0) {
            chains = parse_positive_integer(value, "--chains");
        } else if (std::strcmp(option, "--warps") == 0 && warps == 0) {
            warps = parse_positive_integer(value, "--warps");
        } else if (std::strcmp(option, "--residency") == 0
                   && residency_cell == nullptr) {
            residency_cell = value;
        } else if (std::strcmp(option, "--trials") == 0 && trials == 0) {
            trials = parse_positive_integer(value, "--trials");
        } else if (std::strcmp(option, "--expected-device-uuid") == 0
                   && expected_device_uuid == nullptr) {
            expected_device_uuid = value;
        } else if (std::strcmp(option, "--expected-pci-bus-id") == 0
                   && expected_pci_bus_id == nullptr) {
            expected_pci_bus_id = value;
        } else if (std::strcmp(option, "--grid-blocks") == 0
                   && grid_blocks == 0) {
            grid_blocks = parse_positive_integer(value, "--grid-blocks");
        } else if (std::strcmp(option, "--paired-resident-ctas-per-sm") == 0
                   && paired_resident == 0) {
            paired_resident = parse_positive_integer(
                value, "--paired-resident-ctas-per-sm");
        } else {
            fail("unknown or duplicate command-line option");
        }
    }
    if (action == nullptr || primitive_text == nullptr || mode == nullptr
            || residency_cell == nullptr || expected_device_uuid == nullptr
            || expected_pci_bus_id == nullptr) {
        fail("all command-line options are required exactly once");
    }
    if (std::strcmp(action, "occupancy") != 0
            && std::strcmp(action, "run") != 0) {
        fail("--action must be occupancy or run");
    }
    if (std::strcmp(mode, "latency") != 0
            && std::strcmp(mode, "throughput") != 0) {
        fail("--mode must be latency or throughput");
    }
    if (std::strcmp(residency_cell, "one") != 0
            && std::strcmp(residency_cell, "max") != 0) {
        fail("--residency must be one or max");
    }
    if (chains != 1 && chains != 2 && chains != 4
            && chains != 8 && chains != 16) {
        fail("--chains must be one of 1, 2, 4, 8, 16");
    }
    if (warps != 1 && warps != 4 && warps != 8
            && warps != 16 && warps != 32) {
        fail("--warps must be one of 1, 4, 8, 16, 32");
    }
    if (std::strcmp(mode, "latency") == 0) {
        if (unroll != 4 && unroll != 8 && unroll != 16 && unroll != 32) {
            fail("latency --unroll must be one of 4, 8, 16, 32");
        }
        if (chains != 1 || warps != 1
                || std::strcmp(residency_cell, "one") != 0
                || grid_blocks != 1 || paired_resident != 1) {
            fail("latency cells require chains=1, warps=1, residency=one, "
                 "grid=1, paired residency=1");
        }
        if (std::strcmp(primitive_text, "ex2_approx_f32") == 0) {
            fail("EX2 dependent latency is not_identifiable with a finite "
                 "pure recurrence");
        }
        if (std::strcmp(primitive_text, "rcp_approx_f32") == 0) {
            fail("RCP dependent latency is not_identifiable after CUDA "
                 "ptxas reciprocal-pair collapse");
        }
    } else {
        if (unroll != kThroughputUnroll) {
            fail("throughput --unroll must be 128");
        }
        if (std::strcmp(primitive_text, "mul_f32") == 0) {
            fail("mul_f32 is a latency-only replacement mechanism");
        }
        if (std::strcmp(primitive_text, "ex2_approx_f32") == 0) {
            fail("EX2 throughput is not_identifiable without a finite pure "
                 "recurrence operand class");
        }
        if (std::strcmp(primitive_text, "rcp_approx_f32") == 0) {
            fail("RCP throughput is not_identifiable after CUDA ptxas "
                 "reciprocal-chain collapse");
        }
    }
    return Arguments{
        action, parse_primitive(primitive_text), primitive_text, mode,
        residency_cell, expected_device_uuid, expected_pci_bus_id, unroll,
        chains, warps, trials, grid_blocks, paired_resident,
    };
}


std::string uuid_string(const cudaUUID_t& uuid) {
    char text[41];
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(
        uuid.bytes);
    int written = std::snprintf(
        text, sizeof(text),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
        "%02x%02x%02x%02x%02x%02x",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5],
        bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
        bytes[12], bytes[13], bytes[14], bytes[15]);
    if (written != 40) fail("CUDA UUID formatting failed");
    return std::string(text);
}


bool same_pci_identity(const std::string& left, const std::string& right) {
    unsigned int left_domain, left_bus, left_device, left_function;
    unsigned int right_domain, right_bus, right_device, right_function;
    int left_length = 0;
    int right_length = 0;
    int left_fields = std::sscanf(
        left.c_str(), "%x:%x:%x.%x%n", &left_domain, &left_bus,
        &left_device, &left_function, &left_length);
    int right_fields = std::sscanf(
        right.c_str(), "%x:%x:%x.%x%n", &right_domain, &right_bus,
        &right_device, &right_function, &right_length);
    return left_fields == 4 && right_fields == 4
        && left_length == static_cast<int>(left.size())
        && right_length == static_cast<int>(right.size())
        && left_domain == right_domain && left_bus == right_bus
        && left_device == right_device && left_function == right_function;
}


double lane_operations_per_instruction(Primitive primitive) {
    switch (primitive) {
        case FMA_F32: return 2.0;
        case ADD_F32:
        case EX2_APPROX_F32:
        case RCP_APPROX_F32:
        case RSQRT_APPROX_F32:
        case CONTROL_XOR_B32: return 1.0;
        case MUL_F32: return 1.0;
        default: fail("unreachable primitive operation scale");
    }
}


template <typename Kernel>
int occupancy_limit(Kernel kernel, int block_threads) {
    int occupancy = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, block_threads, 0));
    if (occupancy <= 0) fail("kernel has no legal cooperative residency");
    return occupancy;
}


template <typename Kernel>
void launch_cooperative(
        Kernel kernel, int grid_blocks, int block_threads,
        std::uint64_t* starts, std::uint64_t* ends,
        std::uint32_t* start_smids, std::uint32_t* end_smids,
        const float* inputs, float* observations) {
    void* kernel_arguments[] = {
        &starts, &ends, &start_smids, &end_smids, &inputs, &observations};
    CHECK_CUDA(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid_blocks),
        dim3(block_threads), kernel_arguments, 0, nullptr));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}


struct SmCycleAggregate {
    std::size_t active_sm_count;
    std::uint64_t minimum_elapsed_cycles;
    std::uint64_t maximum_elapsed_cycles;
    std::uint64_t sum_elapsed_cycles;
    std::uint64_t single_sm_start_cycle;
    std::uint64_t single_sm_end_cycle;
};


SmCycleAggregate aggregate_sm_cycles(
        const std::vector<std::uint64_t>& starts,
        const std::vector<std::uint64_t>& ends,
        const std::vector<std::uint32_t>& start_smids,
        const std::vector<std::uint32_t>& end_smids,
        int block_threads, int expected_active_sms,
        int expected_blocks_per_sm) {
    if (starts.size() != ends.size()
            || starts.size() != start_smids.size()
                * static_cast<std::size_t>(block_threads)
            || start_smids.size() != end_smids.size()) {
        fail("host timing and SMID buffers have inconsistent sizes");
    }
    std::map<std::uint32_t, std::vector<std::size_t>> blocks_by_sm;
    for (std::size_t block = 0; block < start_smids.size(); ++block) {
        if (start_smids[block] != end_smids[block]) {
            fail("CTA changed SMID across the measured clock64 interval");
        }
        blocks_by_sm[start_smids[block]].push_back(block);
    }
    if (blocks_by_sm.size()
            != static_cast<std::size_t>(expected_active_sms)) {
        fail("cooperative grid did not cover the expected distinct SMIDs");
    }
    std::uint64_t minimum_elapsed = (
        std::numeric_limits<std::uint64_t>::max());
    std::uint64_t maximum_elapsed = 0;
    std::uint64_t sum_elapsed = 0;
    std::uint64_t single_start = 0;
    std::uint64_t single_end = 0;
    for (const auto& entry : blocks_by_sm) {
        const std::vector<std::size_t>& blocks = entry.second;
        if (blocks.size()
                != static_cast<std::size_t>(expected_blocks_per_sm)) {
            fail("cooperative grid CTA count per SMID changed");
        }
        std::uint64_t local_start = (
            std::numeric_limits<std::uint64_t>::max());
        std::uint64_t local_end = 0;
        for (std::size_t block : blocks) {
            std::size_t begin = block * block_threads;
            std::size_t finish = begin + block_threads;
            for (std::size_t thread = begin; thread < finish; ++thread) {
                local_start = std::min(local_start, starts[thread]);
                local_end = std::max(local_end, ends[thread]);
            }
        }
        if (local_start == 0 || local_end <= local_start) {
            fail("clock64 produced an invalid local-SM interval");
        }
        std::uint64_t elapsed = local_end - local_start;
        minimum_elapsed = std::min(minimum_elapsed, elapsed);
        maximum_elapsed = std::max(maximum_elapsed, elapsed);
        if (sum_elapsed > std::numeric_limits<std::uint64_t>::max() - elapsed) {
            fail("sum of local-SM clock64 intervals overflowed");
        }
        sum_elapsed += elapsed;
        if (blocks_by_sm.size() == 1) {
            single_start = local_start;
            single_end = local_end;
        }
    }
    return SmCycleAggregate{
        blocks_by_sm.size(), minimum_elapsed, maximum_elapsed, sum_elapsed,
        single_start, single_end};
}


template <typename Kernel>
void run_kernel(
        Kernel kernel, const Arguments& arguments,
        const cudaDeviceProp& properties, const std::string& device_uuid,
        const std::string& pci_bus_id, int occupancy, int opcode_count) {
    int block_threads = arguments.warps * kWarpSize;
    int maximum_grid = occupancy * properties.multiProcessorCount;
    if (arguments.paired_resident_ctas_per_sm > occupancy) {
        fail("paired residency exceeds this specialization occupancy");
    }
    int expected_grid = 1;
    if (std::strcmp(arguments.mode, "throughput") == 0) {
        expected_grid = (
            arguments.paired_resident_ctas_per_sm
            * properties.multiProcessorCount);
    }
    if (arguments.grid_blocks != expected_grid
            || arguments.grid_blocks > maximum_grid) {
        fail("requested grid is not the proven paired cooperative grid");
    }
    std::size_t thread_elements = (
        static_cast<std::size_t>(arguments.grid_blocks) * block_threads);
    bool throughput = std::strcmp(arguments.mode, "throughput") == 0;
    std::size_t input_elements = thread_elements * (
        throughput ? 2 * arguments.chains : 1);
    std::size_t observation_elements = thread_elements * (
        throughput ? arguments.chains : opcode_count);
    std::uint64_t* starts_device = nullptr;
    std::uint64_t* ends_device = nullptr;
    std::uint32_t* start_smids_device = nullptr;
    std::uint32_t* end_smids_device = nullptr;
    float* inputs_device = nullptr;
    float* observations_device = nullptr;
    CHECK_CUDA(cudaMalloc(
        &starts_device,
        thread_elements * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(
        &ends_device,
        thread_elements * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaMalloc(
        &start_smids_device,
        arguments.grid_blocks * sizeof(std::uint32_t)));
    CHECK_CUDA(cudaMalloc(
        &end_smids_device,
        arguments.grid_blocks * sizeof(std::uint32_t)));
    CHECK_CUDA(cudaMalloc(
        &inputs_device, input_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(
        &observations_device, observation_elements * sizeof(float)));
    std::vector<float> inputs(input_elements);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        inputs[index] = 0.75f + static_cast<float>(index % 17) * 0.015625f;
    }
    CHECK_CUDA(cudaMemcpy(
        inputs_device, inputs.data(), inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    std::fprintf(stderr, "telemetry_ready\n");
    std::fflush(stderr);
    int telemetry_release = std::fgetc(stdin);
    if (telemetry_release != 'G') {
        fail("runner did not complete overlapping live telemetry handshake");
    }
    auto launch = [&]() {
        launch_cooperative(
            kernel, arguments.grid_blocks, block_threads,
            starts_device, ends_device, start_smids_device, end_smids_device,
            inputs_device, observations_device);
    };
    for (int warmup = 0; warmup < kWarmups; ++warmup) launch();
    std::vector<std::uint64_t> starts(thread_elements);
    std::vector<std::uint64_t> ends(thread_elements);
    std::vector<std::uint32_t> start_smids(arguments.grid_blocks);
    std::vector<std::uint32_t> end_smids(arguments.grid_blocks);
    std::printf("%s\n", kRawHeader);
    for (int trial = 1; trial <= arguments.trials; ++trial) {
        launch();
        CHECK_CUDA(cudaMemcpy(
            starts.data(), starts_device,
            starts.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            ends.data(), ends_device,
            ends.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            start_smids.data(), start_smids_device,
            start_smids.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(
            end_smids.data(), end_smids_device,
            end_smids.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost));
        int expected_active_sms = throughput
            ? properties.multiProcessorCount : 1;
        int expected_blocks_per_sm = throughput
            ? arguments.paired_resident_ctas_per_sm : 1;
        SmCycleAggregate aggregate = aggregate_sm_cycles(
            starts, ends, start_smids, end_smids, block_threads,
            expected_active_sms, expected_blocks_per_sm);
        std::uint64_t opcodes_per_thread = opcode_count;
        std::uint64_t issued_warp_opcodes = (
            static_cast<std::uint64_t>(arguments.grid_blocks)
            * arguments.warps * opcodes_per_thread);
        std::printf(
            "%d,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%s,%d,%d,"
            "%d,%d,%llu,%llu,",
            trial, arguments.mode, arguments.primitive_text,
            "clock64",
            properties.name, device_uuid.c_str(), pci_bus_id.c_str(),
            properties.clockRate, properties.multiProcessorCount,
            arguments.grid_blocks, block_threads, arguments.warps,
            arguments.residency_cell,
            arguments.paired_resident_ctas_per_sm, occupancy,
            arguments.chains, arguments.unroll,
            static_cast<unsigned long long>(opcodes_per_thread),
            static_cast<unsigned long long>(issued_warp_opcodes));
        if (!throughput) {
            std::printf(
                "%llu,%llu,%llu,1,not_applicable,not_applicable,"
                "not_applicable,not_applicable,not_applicable,"
                "not_applicable\n",
                static_cast<unsigned long long>(
                    aggregate.single_sm_start_cycle),
                static_cast<unsigned long long>(
                    aggregate.single_sm_end_cycle),
                static_cast<unsigned long long>(
                    aggregate.sum_elapsed_cycles));
        } else {
            std::printf(
                "not_applicable,not_applicable,not_applicable,%zu,"
                "%llu,%llu,%llu,",
                aggregate.active_sm_count,
                static_cast<unsigned long long>(
                    aggregate.minimum_elapsed_cycles),
                static_cast<unsigned long long>(
                    aggregate.maximum_elapsed_cycles),
                static_cast<unsigned long long>(
                    aggregate.sum_elapsed_cycles));
            double subcore_cycles = (
                static_cast<double>(aggregate.sum_elapsed_cycles)
                * kSubcoresPerSm);
            double lane_rate = (
                issued_warp_opcodes * kWarpSize
                * lane_operations_per_instruction(arguments.primitive)
                / subcore_cycles);
            double warp_rate = issued_warp_opcodes / subcore_cycles;
            std::printf("%.12f,", lane_rate);
            if (arguments.primitive == FMA_F32) {
                std::printf("%.12f,", lane_rate);
            } else {
                std::printf("not_applicable,");
            }
            std::printf("%.12f\n", warp_rate);
        }
    }
    CHECK_CUDA(cudaFree(observations_device));
    CHECK_CUDA(cudaFree(inputs_device));
    CHECK_CUDA(cudaFree(end_smids_device));
    CHECK_CUDA(cudaFree(start_smids_device));
    CHECK_CUDA(cudaFree(ends_device));
    CHECK_CUDA(cudaFree(starts_device));
}


template <Primitive primitive, int unroll>
void execute_dependent(
        const Arguments& arguments, const cudaDeviceProp& properties,
        const std::string& device_uuid, const std::string& pci_bus_id) {
    auto kernel = dependent_kernel<primitive, unroll>;
    int occupancy = occupancy_limit(
        kernel, arguments.warps * kWarpSize);
    if (std::strcmp(arguments.action, "occupancy") == 0) {
        std::printf(
            "occupancy,latency,%s,%d,1,1,%d,%d,%d\n",
            arguments.primitive_text, unroll, occupancy,
            properties.multiProcessorCount,
            occupancy * properties.multiProcessorCount);
        return;
    }
    run_kernel(
        kernel, arguments, properties, device_uuid, pci_bus_id,
        occupancy, unroll);
}


template <Primitive primitive, int chains>
void execute_throughput(
        const Arguments& arguments, const cudaDeviceProp& properties,
        const std::string& device_uuid, const std::string& pci_bus_id) {
    auto kernel = throughput_kernel<primitive, chains>;
    int occupancy = occupancy_limit(
        kernel, arguments.warps * kWarpSize);
    if (std::strcmp(arguments.action, "occupancy") == 0) {
        std::printf(
            "occupancy,throughput,%s,%d,%d,%d,%d,%d,%d\n",
            arguments.primitive_text, kThroughputUnroll, chains,
            arguments.warps, occupancy, properties.multiProcessorCount,
            occupancy * properties.multiProcessorCount);
        return;
    }
    run_kernel(
        kernel, arguments, properties, device_uuid, pci_bus_id,
        occupancy, kThroughputUnroll * chains);
}


template <Primitive primitive>
void dispatch_dependent_unroll(
        const Arguments& arguments, const cudaDeviceProp& properties,
        const std::string& device_uuid, const std::string& pci_bus_id) {
    switch (arguments.unroll) {
        case 4:
            execute_dependent<primitive, 4>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case 8:
            execute_dependent<primitive, 8>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case 16:
            execute_dependent<primitive, 16>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case 32:
            execute_dependent<primitive, 32>(
                arguments, properties, device_uuid, pci_bus_id); return;
        default: fail("unreachable dependent unroll dispatch");
    }
}


template <Primitive primitive>
void dispatch_throughput_chains(
        const Arguments& arguments, const cudaDeviceProp& properties,
        const std::string& device_uuid, const std::string& pci_bus_id) {
    switch (arguments.chains) {
        case 1:
            execute_throughput<primitive, 1>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case 2:
            execute_throughput<primitive, 2>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case 4:
            execute_throughput<primitive, 4>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case 8:
            execute_throughput<primitive, 8>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case 16:
            execute_throughput<primitive, 16>(
                arguments, properties, device_uuid, pci_bus_id); return;
        default: fail("unreachable throughput chain dispatch");
    }
}


void dispatch(
        const Arguments& arguments, const cudaDeviceProp& properties,
        const std::string& device_uuid, const std::string& pci_bus_id) {
    if (std::strcmp(arguments.mode, "latency") == 0) {
        switch (arguments.primitive) {
            case FMA_F32:
                dispatch_dependent_unroll<FMA_F32>(
                    arguments, properties, device_uuid, pci_bus_id); return;
            case ADD_F32:
                dispatch_dependent_unroll<ADD_F32>(
                    arguments, properties, device_uuid, pci_bus_id); return;
            case RCP_APPROX_F32:
                fail("RCP dependent latency is not_identifiable after "
                     "CUDA ptxas reciprocal-pair collapse");
            case MUL_F32:
                dispatch_dependent_unroll<MUL_F32>(
                    arguments, properties, device_uuid, pci_bus_id); return;
            case RSQRT_APPROX_F32:
                dispatch_dependent_unroll<RSQRT_APPROX_F32>(
                    arguments, properties, device_uuid, pci_bus_id); return;
            case CONTROL_XOR_B32:
                dispatch_dependent_unroll<CONTROL_XOR_B32>(
                    arguments, properties, device_uuid, pci_bus_id); return;
            case EX2_APPROX_F32:
                fail("EX2 dependent latency is not_identifiable");
            default: fail("unreachable latency primitive dispatch");
        }
    }
    switch (arguments.primitive) {
        case FMA_F32:
            dispatch_throughput_chains<FMA_F32>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case ADD_F32:
            dispatch_throughput_chains<ADD_F32>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case EX2_APPROX_F32:
            fail("EX2 throughput is not_identifiable without a finite pure "
                 "recurrence operand class");
        case RCP_APPROX_F32:
            fail("RCP throughput is not_identifiable after CUDA ptxas "
                 "reciprocal-chain collapse");
        case RSQRT_APPROX_F32:
            dispatch_throughput_chains<RSQRT_APPROX_F32>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case CONTROL_XOR_B32:
            dispatch_throughput_chains<CONTROL_XOR_B32>(
                arguments, properties, device_uuid, pci_bus_id); return;
        case MUL_F32:
            fail("MUL_F32 is a latency-only replacement mechanism");
        default: fail("unreachable throughput primitive dispatch");
    }
}


int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--describe-abi") == 0) {
        std::printf("%s\n", kAbiLine);
        return EXIT_SUCCESS;
    }
    Arguments arguments = parse_arguments(argc, argv);
    int device = -1;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (std::strcmp(properties.name, kExpectedDeviceName) != 0
            || properties.major != 8 || properties.minor != 0
            || properties.warpSize != kWarpSize
            || properties.multiProcessorCount != kSmCount
            || properties.cooperativeLaunch == 0) {
        std::fprintf(
            stderr,
            "expected %s sm_80/warp32/108SM/cooperative, got "
            "%s sm_%d%d/warp%d/%dSM/cooperative=%d\n",
            kExpectedDeviceName, properties.name, properties.major,
            properties.minor, properties.warpSize,
            properties.multiProcessorCount, properties.cooperativeLaunch);
        return EXIT_FAILURE;
    }
    std::string device_uuid = uuid_string(properties.uuid);
    if (device_uuid != arguments.expected_device_uuid) {
        std::fprintf(
            stderr, "device UUID mismatch: expected %s, got %s\n",
            arguments.expected_device_uuid.c_str(), device_uuid.c_str());
        return EXIT_FAILURE;
    }
    char pci_text[32];
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci_text, sizeof(pci_text), device));
    std::string pci_bus_id(pci_text);
    if (!same_pci_identity(pci_bus_id, arguments.expected_pci_bus_id)) {
        std::fprintf(
            stderr, "PCI identity mismatch: expected %s, got %s\n",
            arguments.expected_pci_bus_id.c_str(), pci_bus_id.c_str());
        return EXIT_FAILURE;
    }
    dispatch(arguments, properties, device_uuid, pci_bus_id);
    return EXIT_SUCCESS;
}
