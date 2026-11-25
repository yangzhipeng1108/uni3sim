/*
 * H200 owner copy of microbench/nvidia_a800/memory/l1/rmsnorm_gamma_residency.cu.
 * Kernel body (two private input streams, barrier, input + cached gamma stream,
 * store), cases, grid 1024 / block 1024, launch and native CSV line are
 * unchanged.  The probe has no SM-count or device-name coupling; it is built for
 * sm_90.
 *
 * Measurement-tool coupling: the A800 runner read the L1/L2 counters with
 * Nsight Compute (ncu --launch-count 1 --metrics ...), which is not installed on
 * the H200 host.  When compiled with -DH200_CUPTI_RANGE_PROFILER the single
 * probe launch is wrapped in the CUPTI 12.9 Range Profiler (auto range, kernel
 * replay) collecting the same five metric names, and one extra line
 * "metrics,<name>=<value>,..." is printed before the native CSV line.  The
 * kernel, its launch and the native output are identical in both builds.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(H200_CUPTI_RANGE_PROFILER)
#include <cuda.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>
#include <vector>

#define CHECK_CUPTI(call)                                                   \
  do {                                                                      \
    const CUptiResult status_ = (call);                                     \
    if (status_ != CUPTI_SUCCESS) {                                         \
      const char* message_ = nullptr;                                       \
      cuptiGetResultString(status_, &message_);                             \
      std::fprintf(stderr, "%s failed: %s\n", #call, message_);             \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

namespace cupti_range {

const char* kMetrics[] = {
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld.sum",
    "lts__t_requests_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
};
constexpr size_t kMetricCount = sizeof(kMetrics) / sizeof(kMetrics[0]);

struct Session {
  CUpti_Profiler_Host_Object* host = nullptr;
  CUpti_RangeProfiler_Object* range = nullptr;
  std::vector<uint8_t> config;
  std::vector<uint8_t> counter_data;
};

Session begin() {
  Session session;
  CUpti_Profiler_Initialize_Params initialize{
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CHECK_CUPTI(cuptiProfilerInitialize(&initialize));
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr) {
    std::fprintf(stderr, "no current CUDA context\n");
    std::exit(1);
  }
  CUpti_Device_GetChipName_Params chip{CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  chip.deviceIndex = 0;
  CHECK_CUPTI(cuptiDeviceGetChipName(&chip));
  CUpti_Profiler_GetCounterAvailability_Params availability{
      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  availability.ctx = context;
  CHECK_CUPTI(cuptiProfilerGetCounterAvailability(&availability));
  std::vector<uint8_t> availability_image(availability.counterAvailabilityImageSize);
  availability.pCounterAvailabilityImage = availability_image.data();
  CHECK_CUPTI(cuptiProfilerGetCounterAvailability(&availability));

  CUpti_Profiler_Host_Initialize_Params host{
      CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
  host.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
  host.pChipName = chip.pChipName;
  host.pCounterAvailabilityImage = availability_image.data();
  CHECK_CUPTI(cuptiProfilerHostInitialize(&host));
  session.host = host.pHostObject;
  CUpti_Profiler_Host_ConfigAddMetrics_Params add{
      CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
  add.pHostObject = session.host;
  add.ppMetricNames = kMetrics;
  add.numMetrics = kMetricCount;
  CHECK_CUPTI(cuptiProfilerHostConfigAddMetrics(&add));
  CUpti_Profiler_Host_GetConfigImageSize_Params config_size{
      CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
  config_size.pHostObject = session.host;
  CHECK_CUPTI(cuptiProfilerHostGetConfigImageSize(&config_size));
  session.config.resize(config_size.configImageSize);
  CUpti_Profiler_Host_GetConfigImage_Params config{
      CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
  config.pHostObject = session.host;
  config.configImageSize = session.config.size();
  config.pConfigImage = session.config.data();
  CHECK_CUPTI(cuptiProfilerHostGetConfigImage(&config));
  CUpti_Profiler_Host_GetNumOfPasses_Params passes{
      CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
  passes.configImageSize = session.config.size();
  passes.pConfigImage = session.config.data();
  CHECK_CUPTI(cuptiProfilerHostGetNumOfPasses(&passes));

  CUpti_RangeProfiler_Enable_Params enable{
      CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
  enable.ctx = context;
  CHECK_CUPTI(cuptiRangeProfilerEnable(&enable));
  session.range = enable.pRangeProfilerObject;
  CUpti_RangeProfiler_GetCounterDataSize_Params data_size{
      CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
  data_size.pRangeProfilerObject = session.range;
  data_size.pMetricNames = kMetrics;
  data_size.numMetrics = kMetricCount;
  data_size.maxNumOfRanges = 1;
  data_size.maxNumRangeTreeNodes = 1;
  CHECK_CUPTI(cuptiRangeProfilerGetCounterDataSize(&data_size));
  session.counter_data.resize(data_size.counterDataSize);
  CUpti_RangeProfiler_CounterDataImage_Initialize_Params data{
      CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  data.pRangeProfilerObject = session.range;
  data.counterDataSize = session.counter_data.size();
  data.pCounterData = session.counter_data.data();
  CHECK_CUPTI(cuptiRangeProfilerCounterDataImageInitialize(&data));
  CUpti_RangeProfiler_SetConfig_Params set{
      CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
  set.pRangeProfilerObject = session.range;
  set.configSize = session.config.size();
  set.pConfig = session.config.data();
  set.counterDataImageSize = session.counter_data.size();
  set.pCounterDataImage = session.counter_data.data();
  set.range = CUPTI_AutoRange;
  set.replayMode = CUPTI_KernelReplay;
  set.maxRangesPerPass = 1;
  set.numNestingLevels = 1;
  set.minNestingLevel = 1;
  set.passIndex = 0;
  set.targetNestingLevel = 0;
  CHECK_CUPTI(cuptiRangeProfilerSetConfig(&set));
  std::fprintf(stderr, "cupti chip=%s passes=%zu\n", chip.pChipName,
               passes.numOfPasses);
  CUpti_RangeProfiler_Start_Params start{
      CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
  start.pRangeProfilerObject = session.range;
  CHECK_CUPTI(cuptiRangeProfilerStart(&start));
  return session;
}

void end_and_print(Session& session) {
  CUpti_RangeProfiler_Stop_Params stop{CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
  stop.pRangeProfilerObject = session.range;
  CHECK_CUPTI(cuptiRangeProfilerStop(&stop));
  if (!stop.isAllPassSubmitted) {
    std::fprintf(stderr, "kernel replay did not submit every pass\n");
    std::exit(1);
  }
  CUpti_RangeProfiler_DecodeData_Params decode{
      CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
  decode.pRangeProfilerObject = session.range;
  CHECK_CUPTI(cuptiRangeProfilerDecodeData(&decode));
  if (decode.numOfRangeDropped != 0) {
    std::fprintf(stderr, "range profiler dropped %zu ranges\n",
                 decode.numOfRangeDropped);
    std::exit(1);
  }
  CUpti_RangeProfiler_GetCounterDataInfo_Params info{
      CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
  info.pCounterDataImage = session.counter_data.data();
  info.counterDataImageSize = session.counter_data.size();
  CHECK_CUPTI(cuptiRangeProfilerGetCounterDataInfo(&info));
  if (info.numTotalRanges != 1) {
    std::fprintf(stderr, "expected one profiled kernel range, got %zu\n",
                 info.numTotalRanges);
    std::exit(1);
  }
  double values[kMetricCount] = {};
  CUpti_Profiler_Host_EvaluateToGpuValues_Params evaluate{
      CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
  evaluate.pHostObject = session.host;
  evaluate.pCounterDataImage = session.counter_data.data();
  evaluate.counterDataImageSize = session.counter_data.size();
  evaluate.rangeIndex = 0;
  evaluate.ppMetricNames = kMetrics;
  evaluate.numMetrics = kMetricCount;
  evaluate.pMetricValues = values;
  CHECK_CUPTI(cuptiProfilerHostEvaluateToGpuValues(&evaluate));
  std::printf("metrics");
  for (size_t index = 0; index < kMetricCount; ++index) {
    std::printf(",%s=%.0f", kMetrics[index], values[index]);
  }
  std::printf("\n");
  CUpti_RangeProfiler_Disable_Params disable{
      CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
  disable.pRangeProfilerObject = session.range;
  CHECK_CUPTI(cuptiRangeProfilerDisable(&disable));
  CUpti_Profiler_Host_Deinitialize_Params host{
      CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
  host.pHostObject = session.host;
  CHECK_CUPTI(cuptiProfilerHostDeinitialize(&host));
  CUpti_Profiler_DeInitialize_Params deinitialize{
      CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
  CHECK_CUPTI(cuptiProfilerDeInitialize(&deinitialize));
}

}  // namespace cupti_range
#endif

namespace {

constexpr int kGrid = 1024;
constexpr int kBlock = 1024;
constexpr int kStaticSharedBytes = 168;

struct Case {
  const char* name;
  int hidden;
  bool private_gamma;
};

constexpr Case kCases[] = {
    {"shared_gamma_h2048", 2048, false},
    {"shared_gamma_h8192", 8192, false},
    {"shared_gamma_h16384", 16384, false},
    {"private_gamma_h16384_control", 16384, true},
    {"private_gamma_h2048_control", 2048, true},
    {"private_gamma_h8192_control", 8192, true},
};

__device__ __forceinline__ uint16_t load_cached_bf16(const uint16_t* address) {
  uint16_t value;
  asm volatile("ld.global.ca.u16 %0, [%1];" : "=h"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ float bf16_to_float(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

extern "C" __global__ __launch_bounds__(kBlock, 2)
void rmsnorm_gamma_residency_probe(const uint16_t* input_a,
                                   const uint16_t* input_b,
                                   const uint16_t* gamma,
                                   uint16_t* output,
                                   int hidden,
                                   int private_gamma) {
  __shared__ volatile unsigned char occupancy_anchor[kStaticSharedBytes];
  const size_t row = static_cast<size_t>(blockIdx.x) * hidden;
  float acc_a = 0.0f;
  float acc_b = 0.0f;

  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    acc_a += bf16_to_float(load_cached_bf16(input_a + row + column));
  }
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    acc_b += bf16_to_float(load_cached_bf16(input_b + row + column));
  }
  if (threadIdx.x == 0) {
    occupancy_anchor[kStaticSharedBytes - 1] =
        static_cast<unsigned char>(__float_as_uint(acc_a + acc_b));
  }
  __syncthreads();

  uint32_t register_anchor_0;
  uint32_t register_anchor_1;
  uint32_t register_anchor_2;
  uint32_t register_anchor_3;
  uint32_t register_anchor_4;
  uint32_t register_anchor_5;
  uint32_t register_anchor_6;
  uint32_t register_anchor_7;
  uint32_t register_anchor_8;
  uint32_t register_anchor_9;
  uint32_t register_anchor_10;
  uint32_t register_anchor_11;
  uint32_t register_anchor_12;
  uint32_t register_anchor_13;
  uint32_t register_anchor_14;
  uint32_t register_anchor_15;
  uint32_t register_anchor_16;
  uint32_t register_anchor_17;
  uint32_t register_anchor_18;
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_0) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_1) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_2) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_3) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_4) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_5) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_6) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_7) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_8) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_9) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_10) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_11) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_12) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_13) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_14) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_15) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_16) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_17) : "r"(threadIdx.x));
  asm volatile("mov.u32 %0, %%clock; xor.b32 %0, %0, %1;"
               : "=&r"(register_anchor_18) : "r"(threadIdx.x));
  asm volatile(
      "add.u32 %0, %0, %1;\n\t"
      "add.u32 %0, %0, %2;\n\t"
      "add.u32 %0, %0, %3;\n\t"
      "add.u32 %0, %0, %4;\n\t"
      "add.u32 %0, %0, %5;\n\t"
      "add.u32 %0, %0, %6;\n\t"
      "add.u32 %0, %0, %7;"
      "add.u32 %0, %0, %8;\n\t"
      "add.u32 %0, %0, %9;\n\t"
      "add.u32 %0, %0, %10;\n\t"
      "add.u32 %0, %0, %11;\n\t"
      "add.u32 %0, %0, %12;\n\t"
      "add.u32 %0, %0, %13;\n\t"
      "add.u32 %0, %0, %14;\n\t"
      "add.u32 %0, %0, %15;\n\t"
      "add.u32 %0, %0, %16;\n\t"
      "add.u32 %0, %0, %17;\n\t"
      "add.u32 %0, %0, %18;"
      : "+r"(register_anchor_0)
      : "r"(register_anchor_1), "r"(register_anchor_2),
        "r"(register_anchor_3), "r"(register_anchor_4),
        "r"(register_anchor_5), "r"(register_anchor_6),
        "r"(register_anchor_7), "r"(register_anchor_8),
        "r"(register_anchor_9), "r"(register_anchor_10),
        "r"(register_anchor_11), "r"(register_anchor_12),
        "r"(register_anchor_13), "r"(register_anchor_14),
        "r"(register_anchor_15), "r"(register_anchor_16),
        "r"(register_anchor_17), "r"(register_anchor_18));

  const size_t gamma_row = private_gamma ? row : 0;
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float input =
        bf16_to_float(load_cached_bf16(input_a + row + column));
    const float weight =
        bf16_to_float(load_cached_bf16(gamma + gamma_row + column));
    const float value = input * weight + acc_a * 0.0f + acc_b * 0.0f +
                        static_cast<float>(register_anchor_0) * 0.0f +
                        static_cast<float>(occupancy_anchor[kStaticSharedBytes - 1]) * 0.0f;
    output[row + column] = static_cast<uint16_t>(__float_as_uint(value) >> 16);
  }
}

void CHECK_CUDA(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

size_t gamma_elements(const Case& probe_case) {
  return static_cast<size_t>(probe_case.hidden) *
         (probe_case.private_gamma ? kGrid : 1);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s CASE_ID\n", argv[0]);
    return 2;
  }
  char* end = nullptr;
  const long case_id = std::strtol(argv[1], &end, 10);
  if (*argv[1] == '\0' || *end != '\0' || case_id < 0 ||
      case_id >= static_cast<long>(sizeof(kCases) / sizeof(kCases[0]))) {
    std::fprintf(stderr, "CASE_ID must be in [0, 5]\n");
    return 2;
  }
  const Case& probe_case = kCases[case_id];
  cudaDeviceProp properties{};
  CHECK_CUDA(cudaGetDeviceProperties(&properties, 0),
             "cudaGetDeviceProperties");
  int resident_ctas_per_sm = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &resident_ctas_per_sm, rmsnorm_gamma_residency_probe,
                 kBlock, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  const size_t input_elements = static_cast<size_t>(kGrid) * probe_case.hidden;
  const size_t input_bytes = input_elements * sizeof(uint16_t);
  const size_t gamma_bytes = gamma_elements(probe_case) * sizeof(uint16_t);
  uint16_t* input_a = nullptr;
  uint16_t* input_b = nullptr;
  uint16_t* gamma = nullptr;
  uint16_t* output = nullptr;
  CHECK_CUDA(cudaMalloc(&input_a, input_bytes), "cudaMalloc input_a");
  CHECK_CUDA(cudaMalloc(&input_b, input_bytes), "cudaMalloc input_b");
  CHECK_CUDA(cudaMalloc(&gamma, gamma_bytes), "cudaMalloc gamma");
  CHECK_CUDA(cudaMalloc(&output, input_bytes), "cudaMalloc output");
  CHECK_CUDA(cudaMemset(input_a, 0x3f, input_bytes), "cudaMemset input_a");
  CHECK_CUDA(cudaMemset(input_b, 0x40, input_bytes), "cudaMemset input_b");
  CHECK_CUDA(cudaMemset(gamma, 0x3f, gamma_bytes), "cudaMemset gamma");
  CHECK_CUDA(cudaMemset(output, 0, input_bytes), "cudaMemset output");
  CHECK_CUDA(cudaDeviceSynchronize(), "pre-launch sync");
#if defined(H200_CUPTI_RANGE_PROFILER)
  cupti_range::Session session = cupti_range::begin();
#endif
  rmsnorm_gamma_residency_probe<<<kGrid, kBlock>>>(
      input_a, input_b, gamma, output, probe_case.hidden,
      probe_case.private_gamma ? 1 : 0);
  CHECK_CUDA(cudaGetLastError(), "rmsnorm_gamma_residency_probe launch");
  CHECK_CUDA(cudaDeviceSynchronize(), "rmsnorm_gamma_residency_probe sync");
#if defined(H200_CUPTI_RANGE_PROFILER)
  cupti_range::end_and_print(session);
#endif
  uint16_t checksum = 0;
  CHECK_CUDA(cudaMemcpy(&checksum, output, sizeof(checksum),
                        cudaMemcpyDeviceToHost), "cudaMemcpy checksum");
  const size_t private_input_read_bytes = 3 * input_bytes;
  const size_t logical_gamma_read_bytes = input_bytes;
  const size_t logical_read_bytes =
      private_input_read_bytes + logical_gamma_read_bytes;
  std::printf("%s,%s,%d,%d,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,%u\n",
              properties.name, probe_case.name,
              probe_case.hidden, kGrid, kBlock, input_bytes, input_bytes,
              gamma_bytes, input_bytes, private_input_read_bytes,
              logical_gamma_read_bytes, logical_read_bytes,
              resident_ctas_per_sm, static_cast<unsigned>(checksum));
  CHECK_CUDA(cudaFree(output), "cudaFree output");
  CHECK_CUDA(cudaFree(gamma), "cudaFree gamma");
  CHECK_CUDA(cudaFree(input_b), "cudaFree input_b");
  CHECK_CUDA(cudaFree(input_a), "cudaFree input_a");
  return 0;
}
