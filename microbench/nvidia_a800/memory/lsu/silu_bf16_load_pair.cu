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
 * Isolate the source-side request overlap of the two adjacent BF16 loads in
 * the A800 SiLU kernel.  The control, single-load, and paired-load variants
 * share the same 1024-thread indexing loop; grids 108, 216, and 432 represent
 * one resident CTA, one two-CTA cohort, and two complete two-CTA cohorts per
 * SM respectively.
 *
 * Build:
 *   /usr/local/cuda-12.2/bin/nvcc -O3 -std=c++17 -arch=sm_80 \
 *     -Xptxas=-v silu_bf16_load_pair.cu -o silu_bf16_load_pair
 * Run on an idle A800.  Rotations 0, 1, and 2 are separate processes:
 *   CUDA_VISIBLE_DEVICES=0 ./silu_bf16_load_pair 0 RUN_ID ROTATION
 */


namespace {

constexpr int kThreads = 1024;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerCta = kThreads / kWarpSize;
constexpr int kSmCount = 108;
constexpr std::size_t kEvictionBytes = 96ULL * 1024ULL * 1024ULL;
constexpr int kWarmups = 3;
constexpr int kClockWarmupLaunches = 2048;
constexpr int kEventSamplesPerTrial = 8;
constexpr int kTrials = 9;
constexpr int kWidths[] = {6144, 9728, 12288, 14336, 25600, 28672, 53248};
constexpr int kGrids[] = {kSmCount, 2 * kSmCount, 4 * kSmCount};
constexpr int kVariantCount = 3;
constexpr int kCellCount = (
    sizeof(kWidths) / sizeof(kWidths[0])
    * sizeof(kGrids) / sizeof(kGrids[0]) * kVariantCount);
constexpr int kIndependentRuns = 3;


void CHECK_CUDA(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}


__device__ __forceinline__ std::uint32_t warp_xor(std::uint32_t value) {
  #pragma unroll
  for (int delta = kWarpSize / 2; delta > 0; delta /= 2) {
    value ^= __shfl_xor_sync(0xffffffffU, value, delta);
  }
  return value;
}


extern "C" __global__ __launch_bounds__(kThreads, 2)
void silu_load_control(
    std::uint32_t* sink, int elements_per_cta) {
  std::uint32_t checksum = static_cast<std::uint32_t>(threadIdx.x + 1);
  for (int index = threadIdx.x; index < elements_per_cta;
       index += blockDim.x) {
    checksum ^= static_cast<std::uint32_t>(index + 1);
  }
  checksum = warp_xor(checksum);
  if ((threadIdx.x & (kWarpSize - 1)) == 0) {
    sink[static_cast<std::size_t>(blockIdx.x) * kWarpsPerCta
         + threadIdx.x / kWarpSize] = checksum;
  }
}


extern "C" __global__ __launch_bounds__(kThreads, 2)
void silu_single_bf16_load(
    const __nv_bfloat16* input, std::uint32_t* sink, int elements_per_cta) {
  const std::size_t cta_base = (
      static_cast<std::size_t>(blockIdx.x) * 2 * elements_per_cta);
  std::uint32_t checksum = static_cast<std::uint32_t>(threadIdx.x + 1);
  for (int index = threadIdx.x; index < elements_per_cta;
       index += blockDim.x) {
    const __nv_bfloat16 x = __ldg(input + cta_base + index);
    checksum ^= static_cast<std::uint32_t>(__bfloat16_as_ushort(x));
  }
  checksum = warp_xor(checksum);
  if ((threadIdx.x & (kWarpSize - 1)) == 0) {
    sink[static_cast<std::size_t>(blockIdx.x) * kWarpsPerCta
         + threadIdx.x / kWarpSize] = checksum;
  }
}


extern "C" __global__ __launch_bounds__(kThreads, 2)
void silu_paired_bf16_load(
    const __nv_bfloat16* input, std::uint32_t* sink, int elements_per_cta) {
  const std::size_t cta_base = (
      static_cast<std::size_t>(blockIdx.x) * 2 * elements_per_cta);
  const __nv_bfloat16* x_ptr = input + cta_base;
  const __nv_bfloat16* y_ptr = x_ptr + elements_per_cta;
  std::uint32_t x_checksum = static_cast<std::uint32_t>(threadIdx.x + 1);
  std::uint32_t y_checksum = static_cast<std::uint32_t>(threadIdx.x + 3);
  for (int index = threadIdx.x; index < elements_per_cta;
       index += blockDim.x) {
    const __nv_bfloat16 x = __ldg(x_ptr + index);
    const __nv_bfloat16 y = __ldg(y_ptr + index);
    x_checksum ^= static_cast<std::uint32_t>(__bfloat16_as_ushort(x));
    y_checksum ^= static_cast<std::uint32_t>(__bfloat16_as_ushort(y));
  }
  std::uint32_t checksum = warp_xor(x_checksum ^ y_checksum);
  if ((threadIdx.x & (kWarpSize - 1)) == 0) {
    sink[static_cast<std::size_t>(blockIdx.x) * kWarpsPerCta
         + threadIdx.x / kWarpSize] = checksum;
  }
}


enum class Variant {
  kControl,
  kSingle,
  kPaired,
};


struct Cell {
  Variant variant;
  int grid;
  int width;
};


const char* variant_name(Variant variant) {
  switch (variant) {
    case Variant::kControl:
      return "control";
    case Variant::kSingle:
      return "single_bf16_load";
    case Variant::kPaired:
      return "paired_bf16_load";
    default:
      throw std::runtime_error("unknown benchmark variant");
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


std::vector<Cell> base_cells() {
  std::vector<Cell> cells;
  const Variant variants[] = {
      Variant::kControl, Variant::kSingle, Variant::kPaired};
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
  return (
      order_position + 7 * trial
      + 20 * run_rotation) % kCellCount;
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
      const int variant = static_cast<int>(cells[cell_index].variant);
      ++variant_positions[variant][position % kVariantCount];
    }
  }
  for (int variant = 0; variant < kVariantCount; ++variant) {
    for (int position = 0; position < kVariantCount; ++position) {
      if (variant_positions[variant][position] != kTrials * kCellCount
              / (kVariantCount * kVariantCount)) {
        throw std::runtime_error("variant positions are not balanced");
      }
    }
  }
}


float run_once(
    Variant variant, const __nv_bfloat16* input, std::uint32_t* sink,
    unsigned char* eviction, int grid, int width, cudaEvent_t start,
    cudaEvent_t stop) {
  CHECK_CUDA(cudaMemset(eviction, 0xa5, kEvictionBytes), "evict L2");
  CHECK_CUDA(cudaDeviceSynchronize(), "synchronize eviction");
  CHECK_CUDA(cudaEventRecord(start), "record start");
  switch (variant) {
    case Variant::kControl:
      silu_load_control<<<grid, kThreads>>>(sink, width);
      break;
    case Variant::kSingle:
      silu_single_bf16_load<<<grid, kThreads>>>(input, sink, width);
      break;
    case Variant::kPaired:
      silu_paired_bf16_load<<<grid, kThreads>>>(input, sink, width);
      break;
    default:
      throw std::runtime_error("unknown benchmark variant");
  }
  CHECK_CUDA(cudaGetLastError(), "launch benchmark kernel");
  CHECK_CUDA(cudaEventRecord(stop), "record stop");
  CHECK_CUDA(cudaEventSynchronize(stop), "synchronize benchmark kernel");
  float elapsed_ms = 0.0F;
  CHECK_CUDA(
      cudaEventElapsedTime(&elapsed_ms, start, stop), "read event duration");
  return elapsed_ms * 1.0e6F;
}


}  // namespace


int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::runtime_error(
          "usage: silu_bf16_load_pair DEVICE_INDEX RUN_ID ROTATION");
    }
    const int device = std::stoi(argv[1]);
    const char* run_id = argv[2];
    const int run_rotation = std::stoi(argv[3]);
    validate_run_id(run_id);
    if (run_rotation < 0 || run_rotation >= kIndependentRuns) {
      throw std::runtime_error("rotation must be 0, 1, or 2");
    }
    const std::vector<Cell> cells = base_cells();
    validate_schedule(cells, run_rotation);
    CHECK_CUDA(cudaSetDevice(device), "select device");
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device), "query device");
    if (properties.major != 8 || properties.minor != 0
        || properties.multiProcessorCount != kSmCount) {
      throw std::runtime_error(
          "benchmark requires the 108-SM compute-capability-8.0 A800");
    }

    constexpr int kMaximumWidth = kWidths[sizeof(kWidths) / sizeof(kWidths[0]) - 1];
    constexpr int kMaximumGrid = kGrids[sizeof(kGrids) / sizeof(kGrids[0]) - 1];
    const std::size_t input_elements = (
        static_cast<std::size_t>(kMaximumGrid) * 2 * kMaximumWidth);
    const std::size_t sink_elements = (
        static_cast<std::size_t>(kMaximumGrid) * kWarpsPerCta);
    __nv_bfloat16* input = nullptr;
    std::uint32_t* sink = nullptr;
    unsigned char* eviction = nullptr;
    CHECK_CUDA(
        cudaMalloc(&input, input_elements * sizeof(*input)), "allocate input");
    CHECK_CUDA(
        cudaMalloc(&sink, sink_elements * sizeof(*sink)), "allocate sink");
    CHECK_CUDA(cudaMalloc(&eviction, kEvictionBytes), "allocate eviction");
    CHECK_CUDA(
        cudaMemset(input, 0x3f, input_elements * sizeof(*input)),
        "initialize input");
    CHECK_CUDA(
        cudaMemset(sink, 0, sink_elements * sizeof(*sink)),
        "initialize sink");

    for (int warmup = 0; warmup < kClockWarmupLaunches; ++warmup) {
      silu_paired_bf16_load<<<kMaximumGrid, kThreads>>>(
          input, sink, kMaximumWidth);
    }
    CHECK_CUDA(cudaGetLastError(), "launch clock warmup");
    CHECK_CUDA(cudaDeviceSynchronize(), "synchronize clock warmup");

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start), "create start event");
    CHECK_CUDA(cudaEventCreate(&stop), "create stop event");

    std::printf("meta,device_index,%d\n", device);
    std::printf("meta,device_name,%s\n", properties.name);
    std::printf("meta,compute_capability,%d.%d\n", properties.major, properties.minor);
    std::printf("meta,sm_count,%d\n", properties.multiProcessorCount);
    std::printf("meta,threads_per_cta,%d\n", kThreads);
    std::printf("meta,eviction_bytes,%zu\n", kEvictionBytes);
    std::printf("meta,clock_warmup_launches,%d\n", kClockWarmupLaunches);
    std::printf("meta,event_samples_per_trial,%d\n", kEventSamplesPerTrial);
    std::printf("meta,run_id,%s\n", run_id);
    std::printf("meta,run_rotation,%d\n", run_rotation);
    std::printf("meta,schedule,silu_bf16_load_pair_cyclic_v1\n");
    std::printf("meta,production_or_evaluation_duration_consumed,false\n");
    std::printf("meta,production_parameter_published,false\n");
    std::printf(
        "raw,run_id,trial,order_position,cell_index,variant,grid,"
        "elements_per_cta,read_bytes_per_cta,duration_ns\n");
    for (int trial = 0; trial < kTrials; ++trial) {
      for (int position = 0; position < kCellCount; ++position) {
        const int cell_index = cell_index_for(
            trial, position, run_rotation);
        const Cell& cell = cells[cell_index];
        for (int warmup = 0; warmup < kWarmups; ++warmup) {
          static_cast<void>(run_once(
              cell.variant, input, sink, eviction, cell.grid, cell.width,
              start, stop));
        }
        double duration_ns = 0.0;
        for (int sample = 0; sample < kEventSamplesPerTrial; ++sample) {
          duration_ns += static_cast<double>(run_once(
              cell.variant, input, sink, eviction, cell.grid, cell.width,
              start, stop));
        }
        duration_ns /= kEventSamplesPerTrial;
        const std::size_t read_bytes = (
            cell.variant == Variant::kControl ? 0 :
            static_cast<std::size_t>(cell.width) * sizeof(*input)
            * (cell.variant == Variant::kPaired ? 2 : 1));
        std::printf(
            "raw,%s,%d,%d,%d,%s,%d,%d,%zu,%.3f\n",
            run_id, trial + 1, position, cell_index,
            variant_name(cell.variant), cell.grid, cell.width, read_bytes,
            duration_ns);
      }
    }

    CHECK_CUDA(cudaEventDestroy(stop), "destroy stop event");
    CHECK_CUDA(cudaEventDestroy(start), "destroy start event");
    CHECK_CUDA(cudaFree(eviction), "free eviction");
    CHECK_CUDA(cudaFree(sink), "free sink");
    CHECK_CUDA(cudaFree(input), "free input");
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return EXIT_FAILURE;
  }
}
