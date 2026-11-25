#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * H200 owner copy of microbench/nvidia_a800/compute/sfu/
 * silu_nonrecurrent_envelope.cu.  Kernel body (source expression
 * x / (1 + expf(-x)) on non-recurrent BF16 inputs, arithmetic control), widths,
 * variants, trials, cyclic schedule and output fields are unchanged.  Hardware
 * coupling only:
 *   - kSmCount 108 -> 132 and device check "NVIDIA H200", cc 9.0;
 *   - grids {108, 216, 256, 1024} -> {132, 264, 313, 1056}: the full-SM,
 *     two-resident and eight-CTA/SM calibration grids scale by 132/108, and the
 *     held-out 256 (2.37 CTAs/SM on A800: min 2, max 3 per SM) becomes
 *     313 = round(256 * 132 / 108) (2.37 CTAs/SM: min 2, max 3 per SM);
 *   - kMaximumGrid 1024 -> 1056 for the output/timer buffers;
 *   - built for sm_90 (was sm_80).
 *
 * Register-resident A800 SiLU math-envelope probe (original description).
 *
 * The target uses the source expression x / (1 + expf(-x)).  Every source
 * input is reconstructed from the thread and loop indices; no SiLU result is
 * an input to a later SiLU evaluation.  The control preserves the same loop,
 * tail predicate, BF16-to-FP32 conversion, accumulation, and single final
 * store but has no exponential or reciprocal instruction.
 *
 * The seven widths are the complete production intermediate-width domain,
 * but neither this binary nor its report consumes a production duration.
 */

namespace {

constexpr int kThreads = 1024;
constexpr int kSmCount = 132;
constexpr int kTrials = 8;
constexpr int kIndependentRuns = 3;
constexpr int kClockWarmupLaunches = 512;
constexpr int kWidths[] = {6144, 9728, 12288, 14336, 25600, 28672, 53248};
constexpr int kGrids[] = {132, 264, 313, 1056};
constexpr int kVariantCount = 2;
constexpr int kCellCount = (
    sizeof(kWidths) / sizeof(kWidths[0])
    * sizeof(kGrids) / sizeof(kGrids[0]) * kVariantCount);


void CHECK_CUDA(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}


enum class Variant {
  kArithmeticControl,
  kSourceSilu,
};


struct Cell {
  Variant variant;
  int grid;
  int width;
};


const char* variant_name(Variant variant) {
  switch (variant) {
    case Variant::kArithmeticControl:
      return "arithmetic_control";
    case Variant::kSourceSilu:
      return "source_silu";
    default:
      throw std::runtime_error("unknown benchmark variant");
  }
}


__device__ __forceinline__ unsigned int read_smid() {
  unsigned int value;
  asm volatile("mov.u32 %0, %smid;" : "=r"(value));
  return value;
}


__device__ __forceinline__ __nv_bfloat16 independent_input(
    int linear_index) {
  const std::uint16_t bits = static_cast<std::uint16_t>(
      0x3e00U + (static_cast<unsigned int>(linear_index) & 0x01ffU));
  return __ushort_as_bfloat16(bits);
}


__device__ __forceinline__ float source_silu(__nv_bfloat16 input) {
  const float value = __bfloat162float(input);
  return value / (1.0F + expf(-value));
}


template <Variant variant>
__global__ __launch_bounds__(kThreads, 2)
void silu_nonrecurrent_envelope(
    int width,
    float* __restrict__ output,
    std::uint64_t* __restrict__ starts,
    std::uint64_t* __restrict__ ends,
    unsigned int* __restrict__ smids) {
  float accumulator = 0.0F;
  const int rounds = (width + blockDim.x - 1) / blockDim.x;

  __syncthreads();
  if (threadIdx.x == 0) {
    smids[blockIdx.x] = read_smid();
    starts[blockIdx.x] = clock64();
  }
  __syncthreads();

#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const int index = round * blockDim.x + threadIdx.x;
    if (index < width) {
      const __nv_bfloat16 input = independent_input(index);
      if constexpr (variant == Variant::kSourceSilu) {
        accumulator += source_silu(input);
      } else {
        const float value = __bfloat162float(input);
        accumulator = fmaf(value, 0.125F, accumulator);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    ends[blockIdx.x] = clock64();
  }
  __syncthreads();
  output[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x]
      = accumulator;
}


std::vector<Cell> base_cells() {
  std::vector<Cell> cells;
  const Variant variants[] = {
      Variant::kArithmeticControl, Variant::kSourceSilu};
  for (const int width : kWidths) {
    for (const int grid : kGrids) {
      for (const Variant variant : variants) {
        cells.push_back({variant, grid, width});
      }
    }
  }
  if (static_cast<int>(cells.size()) != kCellCount) {
    throw std::runtime_error("cell matrix does not match kCellCount");
  }
  return cells;
}


int cell_index_for(int trial, int order_position, int run_rotation) {
  return (order_position + 7 * trial + 19 * run_rotation) % kCellCount;
}


void validate_schedule(const std::vector<Cell>& cells, int run_rotation) {
  int variant_positions[kVariantCount][kVariantCount] = {};
  for (int trial = 0; trial < kTrials; ++trial) {
    bool seen[kCellCount] = {};
    for (int position = 0; position < kCellCount; ++position) {
      const int cell_index = cell_index_for(trial, position, run_rotation);
      if (cell_index < 0 || cell_index >= kCellCount || seen[cell_index]) {
        throw std::runtime_error("cyclic schedule is not a permutation");
      }
      seen[cell_index] = true;
      ++variant_positions[static_cast<int>(cells[cell_index].variant)]
          [position % kVariantCount];
    }
  }
  for (int variant = 0; variant < kVariantCount; ++variant) {
    if (variant_positions[variant][0] != variant_positions[variant][1]) {
      throw std::runtime_error("variant positions are not balanced");
    }
  }
}


void validate_run_id(const char* run_id) {
  const std::size_t length = std::strlen(run_id);
  if (length == 0 || length > 96) {
    throw std::runtime_error("run_id length must be in [1,96]");
  }
  for (std::size_t index = 0; index < length; ++index) {
    const char value = run_id[index];
    const bool valid = (
        (value >= 'a' && value <= 'z')
        || (value >= 'A' && value <= 'Z')
        || (value >= '0' && value <= '9')
        || value == '.' || value == '_' || value == '-');
    if (!valid) {
      throw std::runtime_error("run_id contains an unsupported character");
    }
  }
}


template <Variant variant>
void launch_cell(
    const Cell& cell, float* output, std::uint64_t* starts,
    std::uint64_t* ends, unsigned int* smids) {
  silu_nonrecurrent_envelope<variant><<<cell.grid, kThreads>>>(
      cell.width, output, starts, ends, smids);
}


void run_cell(
    const Cell& cell, float* output, std::uint64_t* starts_device,
    std::uint64_t* ends_device, unsigned int* smids_device,
    std::vector<std::uint64_t>* starts,
    std::vector<std::uint64_t>* ends,
    std::vector<unsigned int>* smids) {
  if (cell.variant == Variant::kSourceSilu) {
    launch_cell<Variant::kSourceSilu>(
        cell, output, starts_device, ends_device, smids_device);
  } else {
    launch_cell<Variant::kArithmeticControl>(
        cell, output, starts_device, ends_device, smids_device);
  }
  CHECK_CUDA(cudaGetLastError(), "launch benchmark cell");
  CHECK_CUDA(cudaDeviceSynchronize(), "synchronize benchmark cell");
  CHECK_CUDA(cudaMemcpy(
      starts->data(), starts_device,
      cell.grid * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
      "copy start clocks");
  CHECK_CUDA(cudaMemcpy(
      ends->data(), ends_device,
      cell.grid * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
      "copy end clocks");
  CHECK_CUDA(cudaMemcpy(
      smids->data(), smids_device,
      cell.grid * sizeof(unsigned int), cudaMemcpyDeviceToHost),
      "copy SM IDs");
}

}  // namespace


int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::runtime_error(
          "usage: silu_nonrecurrent_envelope DEVICE_INDEX RUN_ID ROTATION");
    }
    const int device = std::stoi(argv[1]);
    const char* run_id = argv[2];
    const int rotation = std::stoi(argv[3]);
    validate_run_id(run_id);
    if (device < 0 || rotation < 0 || rotation >= kIndependentRuns) {
      throw std::runtime_error("device must be nonnegative and rotation in [0,2]");
    }
    const std::vector<Cell> cells = base_cells();
    validate_schedule(cells, rotation);

    CHECK_CUDA(cudaSetDevice(device), "select device");
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device), "query device");
    if (std::string(properties.name) != "NVIDIA H200"
        || properties.major != 9 || properties.minor != 0
        || properties.multiProcessorCount != kSmCount) {
      throw std::runtime_error(
          "benchmark requires a 132-SM NVIDIA H200");
    }
    int control_residency = 0;
    int source_silu_residency = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &control_residency,
        silu_nonrecurrent_envelope<Variant::kArithmeticControl>,
        kThreads, 0), "query arithmetic-control occupancy");
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &source_silu_residency,
        silu_nonrecurrent_envelope<Variant::kSourceSilu>,
        kThreads, 0), "query source-SiLU occupancy");
    if (control_residency != 2 || source_silu_residency != 2) {
      throw std::runtime_error("both probe variants must admit two CTAs per SM");
    }

    constexpr int kMaximumGrid = 1056;
    constexpr std::size_t kOutputCount = (
        static_cast<std::size_t>(kMaximumGrid) * kThreads);
    float* output = nullptr;
    std::uint64_t* starts_device = nullptr;
    std::uint64_t* ends_device = nullptr;
    unsigned int* smids_device = nullptr;
    CHECK_CUDA(cudaMalloc(&output, kOutputCount * sizeof(float)), "allocate output");
    CHECK_CUDA(cudaMalloc(
        &starts_device, kMaximumGrid * sizeof(std::uint64_t)),
        "allocate start clocks");
    CHECK_CUDA(cudaMalloc(
        &ends_device, kMaximumGrid * sizeof(std::uint64_t)),
        "allocate end clocks");
    CHECK_CUDA(cudaMalloc(
        &smids_device, kMaximumGrid * sizeof(unsigned int)),
        "allocate SM IDs");

    for (int warmup = 0; warmup < kClockWarmupLaunches; ++warmup) {
      silu_nonrecurrent_envelope<Variant::kSourceSilu>
          <<<kSmCount, kThreads>>>(
              6144, output, starts_device, ends_device, smids_device);
    }
    CHECK_CUDA(cudaGetLastError(), "launch clock warmup");
    CHECK_CUDA(cudaDeviceSynchronize(), "synchronize clock warmup");

    std::vector<std::uint64_t> starts(kMaximumGrid);
    std::vector<std::uint64_t> ends(kMaximumGrid);
    std::vector<unsigned int> smids(kMaximumGrid);
    std::printf("meta,device_index,%d\n", device);
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,compute_capability,%d.%d\n", properties.major, properties.minor);
    std::printf("meta,sm_count,%d\n", properties.multiProcessorCount);
    std::printf("meta,threads_per_cta,%d\n", kThreads);
    std::printf("meta,resident_cta_limit_per_sm,2\n");
    std::printf("meta,run_id,%s\n", run_id);
    std::printf("meta,run_rotation,%d\n", rotation);
    std::printf("meta,schedule,silu_nonrecurrent_cyclic_v1\n");
    std::printf("meta,cycle_unit,sm_clock_cycles\n");
    std::printf("meta,work_unit,scalar_silu_evaluations\n");
    std::printf("meta,production_or_evaluation_duration_consumed,false\n");
    std::printf(
        "raw,run_id,trial,order_position,cell_index,variant,grid_ctas,"
        "intermediate_elements_per_cta,loop_rounds_per_cta,"
        "scalar_evaluations,block_threads,active_sms,minimum_ctas_per_sm,"
        "maximum_ctas_per_sm,maximum_sm_span_cycles,median_cta_cycles,checksum\n");

    for (int trial = 0; trial < kTrials; ++trial) {
      for (int position = 0; position < kCellCount; ++position) {
        const int cell_index = cell_index_for(trial, position, rotation);
        const Cell& cell = cells[cell_index];
        run_cell(
            cell, output, starts_device, ends_device, smids_device,
            &starts, &ends, &smids);
        std::uint64_t sm_start[kSmCount];
        std::uint64_t sm_end[kSmCount] = {};
        int ctas_per_sm[kSmCount] = {};
        std::fill_n(sm_start, kSmCount, UINT64_MAX);
        std::vector<std::uint64_t> cta_cycles;
        cta_cycles.reserve(cell.grid);
        for (int block = 0; block < cell.grid; ++block) {
          if (smids[block] >= kSmCount || ends[block] <= starts[block]) {
            throw std::runtime_error("invalid block timing record");
          }
          const int sm = static_cast<int>(smids[block]);
          sm_start[sm] = std::min(sm_start[sm], starts[block]);
          sm_end[sm] = std::max(sm_end[sm], ends[block]);
          ++ctas_per_sm[sm];
          cta_cycles.push_back(ends[block] - starts[block]);
        }
        std::uint64_t maximum_span = 0;
        int active_sms = 0;
        int minimum_ctas = cell.grid;
        int maximum_ctas = 0;
        for (int sm = 0; sm < kSmCount; ++sm) {
          if (ctas_per_sm[sm] == 0) {
            continue;
          }
          ++active_sms;
          minimum_ctas = std::min(minimum_ctas, ctas_per_sm[sm]);
          maximum_ctas = std::max(maximum_ctas, ctas_per_sm[sm]);
          maximum_span = std::max(maximum_span, sm_end[sm] - sm_start[sm]);
        }
        std::sort(cta_cycles.begin(), cta_cycles.end());
        const std::uint64_t median_cta_cycles = cta_cycles[cta_cycles.size() / 2];
        float checksum = 0.0F;
        CHECK_CUDA(cudaMemcpy(
            &checksum, output, sizeof(float), cudaMemcpyDeviceToHost),
            "copy checksum");
        std::printf(
            "raw,%s,%d,%d,%d,%s,%d,%d,%d,%lld,%d,%d,%d,%d,%llu,%llu,%.9g\n",
            run_id, trial + 1, position, cell_index,
            variant_name(cell.variant), cell.grid, cell.width,
            (cell.width + kThreads - 1) / kThreads,
            static_cast<long long>(cell.grid) * cell.width, kThreads,
            active_sms, minimum_ctas, maximum_ctas,
            static_cast<unsigned long long>(maximum_span),
            static_cast<unsigned long long>(median_cta_cycles), checksum);
      }
    }

    CHECK_CUDA(cudaFree(smids_device), "free SM IDs");
    CHECK_CUDA(cudaFree(ends_device), "free end clocks");
    CHECK_CUDA(cudaFree(starts_device), "free start clocks");
    CHECK_CUDA(cudaFree(output), "free output");
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return EXIT_FAILURE;
  }
}
