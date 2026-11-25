/* Independent SM80 BM64xBN64xBK32 two-slot operand-residency probe.
 *
 * This is a tile mechanism, not a GEMM or MoE workload.  Five CTA-equivalent
 * register residency, four warps, and two 8-KiB A/B shared-memory slots match
 * the statically inspected production W1 envelope.  The same instruction
 * stream is measured with a 32-MiB circular backing set (L2 hit service) and
 * a 270-MiB single-pass backing set (DRAM miss service).
 */
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK_CUDA(call) do {                                                \
    const cudaError_t error = (call);                                        \
    if (error != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(error));          \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

constexpr int kThreads = 128;
constexpr int kGrid = 540;
constexpr int kMaximumGrid = 896;
constexpr int kRounds = 256;
constexpr int kOperandBytes = 64 * 32 * 2 + 32 * 64 * 2;
constexpr int kSlots = 2;
constexpr int kSharedBytes = kSlots * kOperandBytes;
constexpr int kHitBackingSlots = (32 * 1024 * 1024) / kOperandBytes;
constexpr int kMissBackingSlots = kGrid * kRounds;
constexpr int kGroupedPanelsPerStream = (kGrid + 7) / 8;
constexpr int kGroupedStreams = 16;
constexpr int kGroupedBackingSlots =
    kGroupedStreams * kGroupedPanelsPerStream * kRounds;
constexpr int kAResidentBytes = 32 * 1024 * 1024;
constexpr int kMaximumBRowBytes = 7168 * 2;
constexpr std::size_t kMaximumBBackingBytes =
    static_cast<std::size_t>(kGroupedStreams) * kRounds * 32
    * kMaximumBRowBytes;

static_assert(kOperandBytes == 8192, "A+B tile bytes changed");
static_assert(kSharedBytes == 16384, "two-slot shared footprint changed");

__device__ __forceinline__ void cp_async_16(
        void* shared_destination, const void* global_source) {
    const std::uint32_t shared_address = static_cast<std::uint32_t>(
        __cvta_generic_to_shared(shared_destination));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(shared_address), "l"(global_source) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
}

__device__ __forceinline__ void keep_registers_live(
        float (&values)[40]) {
    asm volatile(
        "" :
        "+f"(values[0]), "+f"(values[1]), "+f"(values[2]),
        "+f"(values[3]), "+f"(values[4]), "+f"(values[5]),
        "+f"(values[6]), "+f"(values[7]), "+f"(values[8]),
        "+f"(values[9]), "+f"(values[10]), "+f"(values[11]),
        "+f"(values[12]), "+f"(values[13]), "+f"(values[14]),
        "+f"(values[15]), "+f"(values[16]), "+f"(values[17]),
        "+f"(values[18]), "+f"(values[19]), "+f"(values[20]),
        "+f"(values[21]), "+f"(values[22]), "+f"(values[23]),
        "+f"(values[24]), "+f"(values[25]), "+f"(values[26]),
        "+f"(values[27]), "+f"(values[28]), "+f"(values[29]),
        "+f"(values[30]), "+f"(values[31]), "+f"(values[32]),
        "+f"(values[33]), "+f"(values[34]), "+f"(values[35]),
        "+f"(values[36]), "+f"(values[37]), "+f"(values[38]),
        "+f"(values[39]));
}

__device__ __forceinline__ void keep_extra_registers_live(
        float (&values)[32]) {
    asm volatile(
        "" :
        "+f"(values[0]), "+f"(values[1]), "+f"(values[2]),
        "+f"(values[3]), "+f"(values[4]), "+f"(values[5]),
        "+f"(values[6]), "+f"(values[7]), "+f"(values[8]),
        "+f"(values[9]), "+f"(values[10]), "+f"(values[11]),
        "+f"(values[12]), "+f"(values[13]), "+f"(values[14]),
        "+f"(values[15]), "+f"(values[16]), "+f"(values[17]),
        "+f"(values[18]), "+f"(values[19]), "+f"(values[20]),
        "+f"(values[21]), "+f"(values[22]), "+f"(values[23]),
        "+f"(values[24]), "+f"(values[25]), "+f"(values[26]),
        "+f"(values[27]), "+f"(values[28]), "+f"(values[29]),
        "+f"(values[30]), "+f"(values[31]));
}

__device__ __forceinline__ void issue_copy(
        std::uint8_t* shared_slot, const std::uint8_t* global_slot) {
    constexpr int kVectors = kOperandBytes / 16;
    for (int vector = threadIdx.x; vector < kVectors;
         vector += blockDim.x) {
        cp_async_16(
            shared_slot + vector * 16,
            global_slot + vector * 16);
    }
    cp_async_commit();
}

template<int Rounds>
__device__ __forceinline__ void issue_strided_copy(
        std::uint8_t* shared_slot,
        const std::uint8_t* a_resident,
        const std::uint8_t* b_streams,
        int stream, int panel, int round, int b_row_bytes) {
    constexpr int kMatrixBytes = kOperandBytes / 2;
    constexpr int kVectorsPerMatrix = kMatrixBytes / 16;
    const std::size_t a_tile =
        (static_cast<std::size_t>(blockIdx.x) * Rounds + round)
        % (kAResidentBytes / kMatrixBytes);
    for (int vector = threadIdx.x; vector < kVectorsPerMatrix;
         vector += blockDim.x) {
        cp_async_16(
            shared_slot + vector * 16,
            a_resident + a_tile * kMatrixBytes + vector * 16);
        const int row = vector / 8;
        const int column_vector = vector % 8;
        const std::size_t b_stream_bytes =
            static_cast<std::size_t>(Rounds) * 32 * b_row_bytes;
        const std::size_t b_offset =
            static_cast<std::size_t>(stream) * b_stream_bytes
            + static_cast<std::size_t>(round * 32 + row) * b_row_bytes
            + static_cast<std::size_t>(panel) * 128
            + column_vector * 16;
        cp_async_16(
            shared_slot + kMatrixBytes + vector * 16,
            b_streams + b_offset);
    }
    cp_async_commit();
}

__device__ __forceinline__ void ldmatrix_x4(
        std::uint32_t (&fragment)[4], const void* shared_pointer) {
    const std::uint32_t shared_address = static_cast<std::uint32_t>(
        __cvta_generic_to_shared(shared_pointer));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment[0]), "=r"(fragment[1]),
          "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(shared_address));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
        std::uint32_t (&fragment)[2], const void* shared_pointer) {
    const std::uint32_t shared_address = static_cast<std::uint32_t>(
        __cvta_generic_to_shared(shared_pointer));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(shared_address));
}

__device__ __forceinline__ void mma_bf16_m16n8k16(
        float (&accumulator)[4], const std::uint32_t (&a)[4],
        const std::uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(accumulator[0]), "+f"(accumulator[1]),
          "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void issue_hmma_round(
        const std::uint8_t* shared_slot, float (&accumulators)[8][4]) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp >> 1;
    const int warp_n = warp & 1;
#pragma unroll
    for (int k_phase = 0; k_phase < 2; ++k_phase) {
        std::uint32_t a_fragments[2][4];
        const int a_column = k_phase * 16 + ((lane >> 4) * 8);
#pragma unroll
        for (int m_instruction = 0; m_instruction < 2; ++m_instruction) {
            const int a_row =
                warp_m * 32 + m_instruction * 16 + (lane & 15);
            const std::uint16_t* a =
                reinterpret_cast<const std::uint16_t*>(shared_slot)
                + a_row * 32 + a_column;
            ldmatrix_x4(a_fragments[m_instruction], a);
        }
#pragma unroll
        for (int n_instruction = 0; n_instruction < 4; ++n_instruction) {
            std::uint32_t b_fragment[2];
            const int b_row = k_phase * 16 + (lane & 15);
            const int b_column = warp_n * 32 + n_instruction * 8;
            const std::uint16_t* b =
                reinterpret_cast<const std::uint16_t*>(
                    shared_slot + kOperandBytes / 2)
                + b_row * 64 + b_column;
            for (int m_instruction = 0; m_instruction < 2;
                 ++m_instruction) {
                mma_bf16_m16n8k16(
                    accumulators[m_instruction * 4 + n_instruction],
                    a_fragments[m_instruction], b_fragment);
            }
        }
    }
}

__global__ __launch_bounds__(kThreads, 5)
void sm80_bf16_bm64_operand_residency(
        const std::uint8_t* __restrict__ operands,
        std::uint32_t* __restrict__ checksums,
        int backing_slots) {
    extern __shared__ __align__(16) std::uint8_t shared_tiles[];
    float live[40];
    float extra_live[32];
#pragma unroll
    for (int index = 0; index < 40; ++index) {
        live[index] = static_cast<float>(threadIdx.x + index + 1);
    }
#pragma unroll
    for (int index = 0; index < 32; ++index) {
        extra_live[index] = static_cast<float>(threadIdx.x + index + 41);
    }
    std::uint32_t checksum = 0;
    auto global_slot = [&](int round) {
        const std::size_t linear =
            static_cast<std::size_t>(blockIdx.x) * kRounds + round;
        return operands + (linear % backing_slots) * kOperandBytes;
    };
    issue_copy(shared_tiles, global_slot(0));
    cp_async_wait_all();
    __syncthreads();
    for (int round = 0; round < kRounds; ++round) {
        const int slot = round & 1;
        std::uint8_t* current = shared_tiles + slot * kOperandBytes;
        if (round + 1 < kRounds) {
            issue_copy(
                shared_tiles + ((round + 1) & 1) * kOperandBytes,
                global_slot(round + 1));
        }
        const volatile std::uint32_t* words =
            reinterpret_cast<const volatile std::uint32_t*>(current);
        checksum ^= words[threadIdx.x];
        checksum ^= words[kOperandBytes / 4 - 1 - threadIdx.x];
        keep_registers_live(live);
        keep_extra_registers_live(extra_live);
        if (round + 1 < kRounds) {
            cp_async_wait_all();
            __syncthreads();
        }
    }
    keep_registers_live(live);
    keep_extra_registers_live(extra_live);
    float live_sum = 0.0f;
#pragma unroll
    for (int index = 0; index < 40; ++index) live_sum += live[index];
#pragma unroll
    for (int index = 0; index < 32; ++index) live_sum += extra_live[index];
    checksums[static_cast<std::size_t>(blockIdx.x) * blockDim.x
              + threadIdx.x] = checksum ^ __float_as_uint(live_sum);
}

template<int Rounds>
__global__ __launch_bounds__(kThreads, 5)
void sm80_bf16_bm64_dram_miss_pipeline(
        const std::uint8_t* __restrict__ a_resident,
        const std::uint8_t* __restrict__ b_streams,
        float* __restrict__ checksums,
        int fanout,
        int b_row_bytes,
        int n_tiles) {
    extern __shared__ __align__(16) std::uint8_t shared_tiles[];
    float accumulators[8][4];
    float residency_live[40];
#pragma unroll
    for (int output = 0; output < 8; ++output) {
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            accumulators[output][word] =
                static_cast<float>((threadIdx.x & 31) + output + word + 1)
                * 0.0009765625f;
        }
    }
#pragma unroll
    for (int index = 0; index < 40; ++index) {
        residency_live[index] =
            static_cast<float>(threadIdx.x + index + 1);
    }
    const int group_span = 8 * n_tiles;
    const int within_group = blockIdx.x % group_span;
    const int group_lane = within_group % 8;
    const int group = blockIdx.x / group_span;
    const int panel = within_group / 8;
    const int stream = group * 8 + group_lane / fanout;
    issue_strided_copy<Rounds>(
        shared_tiles, a_resident, b_streams,
        stream, panel, 0, b_row_bytes);
    cp_async_wait_all();
    __syncthreads();
    for (int round = 0; round < Rounds; ++round) {
        const int slot = round & 1;
        std::uint8_t* current = shared_tiles + slot * kOperandBytes;
        if (round + 1 < Rounds) {
            issue_strided_copy<Rounds>(
                shared_tiles + ((round + 1) & 1) * kOperandBytes,
                a_resident, b_streams, stream, panel, round + 1,
                b_row_bytes);
        }
        issue_hmma_round(current, accumulators);
        keep_registers_live(residency_live);
        if (round + 1 < Rounds) {
            cp_async_wait_all();
            __syncthreads();
        }
    }
    float checksum = 0.0f;
#pragma unroll
    for (int output = 0; output < 8; ++output) {
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            checksum += accumulators[output][word];
        }
    }
#pragma unroll
    for (int index = 0; index < 40; ++index) {
        checksum += residency_live[index] * 0.000001f;
    }
    checksums[static_cast<std::size_t>(blockIdx.x) * blockDim.x
              + threadIdx.x] = checksum;
}

static void launch(
        const std::uint8_t* operands, std::uint32_t* checksums,
        int backing_slots) {
    sm80_bf16_bm64_operand_residency<<<
        kGrid, kThreads, kSharedBytes>>>(operands, checksums, backing_slots);
}

static void launch_dram_pipeline(
        const std::uint8_t* a_resident, const std::uint8_t* b_streams,
        float* checksums, int fanout, int b_row_bytes, int n_tiles,
        int grid, int rounds) {
    if (rounds == 2048 / 32) {
        sm80_bf16_bm64_dram_miss_pipeline<2048 / 32><<<
            grid, kThreads, kSharedBytes>>>(
                a_resident, b_streams, checksums, fanout, b_row_bytes,
                n_tiles);
    } else if (rounds == 2880 / 32) {
        sm80_bf16_bm64_dram_miss_pipeline<2880 / 32><<<
            grid, kThreads, kSharedBytes>>>(
                a_resident, b_streams, checksums, fanout, b_row_bytes,
                n_tiles);
    } else if (rounds == kRounds) {
        sm80_bf16_bm64_dram_miss_pipeline<kRounds><<<
            grid, kThreads, kSharedBytes>>>(
                a_resident, b_streams, checksums, fanout, b_row_bytes,
                n_tiles);
    } else {
        std::fprintf(stderr, "unsupported pipeline rounds: %d\n", rounds);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s DEVICE TRIALS\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int device = std::stoi(argv[1]);
    const int trials = std::stoi(argv[2]);
    if (device < 0 || trials < 3 || trials > 11) {
        std::fprintf(stderr, "device >= 0 and 3 <= trials <= 11 required\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 8 || properties.minor != 0
            || properties.multiProcessorCount != 108) {
        std::fprintf(stderr, "probe requires the 108-SM A800 sm80 target\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        sm80_bf16_bm64_operand_residency,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
    CHECK_CUDA(cudaFuncSetAttribute(
        sm80_bf16_bm64_dram_miss_pipeline<kRounds>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
    cudaFuncAttributes attributes{};
    CHECK_CUDA(cudaFuncGetAttributes(
        &attributes, sm80_bf16_bm64_operand_residency));
    int resident_ctas = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_ctas, sm80_bf16_bm64_operand_residency,
        kThreads, kSharedBytes));
    cudaFuncAttributes pipeline_attributes{};
    CHECK_CUDA(cudaFuncGetAttributes(
        &pipeline_attributes,
        sm80_bf16_bm64_dram_miss_pipeline<kRounds>));
    int pipeline_resident_ctas = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &pipeline_resident_ctas,
        sm80_bf16_bm64_dram_miss_pipeline<kRounds>,
        kThreads, kSharedBytes));
    std::printf(
        "ATTRIBUTE,%d,%zu,%zu,%d,%d,%d\n",
        attributes.numRegs, attributes.sharedSizeBytes,
        attributes.localSizeBytes, attributes.maxThreadsPerBlock,
        attributes.binaryVersion, resident_ctas);
    std::printf(
        "PIPELINE_ATTRIBUTE,%d,%zu,%zu,%d,%d,%d\n",
        pipeline_attributes.numRegs, pipeline_attributes.sharedSizeBytes,
        pipeline_attributes.localSizeBytes,
        pipeline_attributes.maxThreadsPerBlock,
        pipeline_attributes.binaryVersion, pipeline_resident_ctas);
    std::printf(
        "DATA,trial,residency,backing_bytes,logical_bytes,event_duration_ms,checksum\n");

    std::uint8_t* operands = nullptr;
    std::uint8_t* b_streams = nullptr;
    std::uint32_t* checksums = nullptr;
    float* pipeline_checksums = nullptr;
    CHECK_CUDA(cudaMalloc(
        &operands, static_cast<std::size_t>(kGroupedBackingSlots)
            * kOperandBytes));
    CHECK_CUDA(cudaMalloc(&b_streams, kMaximumBBackingBytes));
    CHECK_CUDA(cudaMalloc(
        &checksums, static_cast<std::size_t>(kGrid) * kThreads
            * sizeof(std::uint32_t)));
    CHECK_CUDA(cudaMalloc(
        &pipeline_checksums, static_cast<std::size_t>(kMaximumGrid) * kThreads
            * sizeof(float)));
    CHECK_CUDA(cudaMemset(
        operands, 0,
        static_cast<std::size_t>(kGroupedBackingSlots) * kOperandBytes));
    CHECK_CUDA(cudaMemset(b_streams, 0, kMaximumBBackingBytes));
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    constexpr int backing_slots[] = {kHitBackingSlots, kMissBackingSlots};
    constexpr const char* names[] = {"l2_resident", "dram_stream"};
    for (int case_index = 0; case_index < 2; ++case_index) {
        for (int warmup = 0; warmup < 3; ++warmup) {
            launch(operands, checksums, backing_slots[case_index]);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        for (int trial = 0; trial < trials; ++trial) {
            CHECK_CUDA(cudaEventRecord(start));
            launch(operands, checksums, backing_slots[case_index]);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float duration_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start, stop));
            std::uint32_t checksum = 0;
            CHECK_CUDA(cudaMemcpy(
                &checksum, checksums, sizeof(checksum),
                cudaMemcpyDeviceToHost));
            const std::size_t backing_bytes =
                static_cast<std::size_t>(backing_slots[case_index])
                * kOperandBytes;
            const std::size_t logical_bytes =
                static_cast<std::size_t>(kGrid) * kRounds * kOperandBytes;
            std::printf(
                "DATA,%d,%s,%zu,%zu,%.9f,%u\n",
                trial, names[case_index], backing_bytes, logical_bytes,
                duration_ms, checksum);
        }
    }
    struct PipelineCase {
        const char* name;
        int fanout;
        int b_row_bytes;
        int n_tiles;
        int grid;
        int rounds;
    };
    constexpr PipelineCase pipeline_cases[] = {
        {"grouped_fanout_1_n4096", 1, 4096 * 2, 64, kGrid, kRounds},
        {"grouped_fanout_1_n7168", 1, 7168 * 2, 112, kGrid, kRounds},
        {"grouped_fanout_8_n6144", 8, 6144 * 2, 96, kGrid, kRounds},
        {"source_group_multiwave_k2048_n7168_fanout1", 1, 7168 * 2,
         112, 8 * 112, 2048 / 32},
        {"source_group_multiwave_k2048_n7168_fanout2", 2, 7168 * 2,
         112, 8 * 112, 2048 / 32},
        {"source_group_multiwave_k2048_n7168_fanout4", 4, 7168 * 2,
         112, 8 * 112, 2048 / 32},
        {"source_group_multiwave_k2048_n7168", 8, 7168 * 2, 112,
         8 * 112, 2048 / 32},
        {"source_group_multiwave_k2880_n5760_fanout1", 1, 5760 * 2,
         90, 8 * 90, 2880 / 32},
        {"source_group_multiwave_k2880_n5760_fanout2", 2, 5760 * 2,
         90, 8 * 90, 2880 / 32},
        {"source_group_multiwave_k2880_n5760_fanout4", 4, 5760 * 2,
         90, 8 * 90, 2880 / 32},
        {"source_group_multiwave_k2880_n5760", 8, 5760 * 2, 90,
         8 * 90, 2880 / 32},
    };
    for (const PipelineCase& pipeline_case : pipeline_cases) {
        for (int warmup = 0; warmup < 3; ++warmup) {
            launch_dram_pipeline(
                operands, b_streams, pipeline_checksums,
                pipeline_case.fanout, pipeline_case.b_row_bytes,
                pipeline_case.n_tiles, pipeline_case.grid,
                pipeline_case.rounds);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        for (int trial = 0; trial < trials; ++trial) {
            CHECK_CUDA(cudaEventRecord(start));
            launch_dram_pipeline(
                operands, b_streams, pipeline_checksums,
                pipeline_case.fanout, pipeline_case.b_row_bytes,
                pipeline_case.n_tiles, pipeline_case.grid,
                pipeline_case.rounds);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float duration_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&duration_ms, start, stop));
            float checksum = 0.0f;
            CHECK_CUDA(cudaMemcpy(
                &checksum, pipeline_checksums, sizeof(checksum),
                cudaMemcpyDeviceToHost));
            const std::size_t logical_bytes =
                static_cast<std::size_t>(pipeline_case.grid)
                * pipeline_case.rounds * kOperandBytes;
            std::printf(
                "PIPELINE_DATA,%d,%s,%zu,%zu,%.9f,%.9g\n",
                trial, pipeline_case.name,
                kAResidentBytes + kMaximumBBackingBytes,
                logical_bytes, duration_ms, checksum);
        }
    }
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaFree(checksums));
    CHECK_CUDA(cudaFree(pipeline_checksums));
    CHECK_CUDA(cudaFree(b_streams));
    CHECK_CUDA(cudaFree(operands));
    return EXIT_SUCCESS;
}
