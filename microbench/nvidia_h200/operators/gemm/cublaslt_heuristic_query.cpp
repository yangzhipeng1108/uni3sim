/* NVIDIA H200 (sm_90) owner copy of
 * microbench/nvidia_a800/operators/gemm/cublaslt_heuristic_query.cpp.
 *
 * The heuristic query, the rank-0 dry launch (every production launch is
 * replaced by a one-thread no-op before submission) and the CUPTI activity
 * proof are unchanged.  Hardware/software coupling changes only:
 *   1. hardware_id "nvidia_h200"; built against cuBLASLt 12.9
 *      (<workspace>/tools/cuda-12.9) instead of 12.2.
 *   2. The replacement no-op PTX targets sm_90 (PTX ISA 7.8).
 *   3. Hopper cuBLASLt (nvjet / sm90 CUTLASS) kernels launch through
 *      cuLaunchKernelEx with launch attributes (thread-block cluster
 *      dimension, cluster scheduling policy, programmatic serialization).
 *      The A800 copy rejected any attribute; this copy records every attribute
 *      id and the cluster dimension as launch facts and still submits the
 *      attribute-free no-op in its place.
 *   4. The cuBLASLt 12.9 tile/stages/cluster enums grew far beyond the 12.2
 *      tables, so the query emits the raw enum ids plus the cluster-shape id
 *      (CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID); the H200 runner maps every id
 *      to its name and dimensions by parsing the cublasLt.h it compiled with.
 *   5. The CTA raster word is read only for the ampere_*gemm* family whose
 *      sm_80 disassembly defines it (unchanged rule); no sm_90 kernel family
 *      has a verified raster-word layout, so none is read for them.
 */
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <cupti_activity.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kHeuristicResultCapacity = 64;
constexpr size_t kActivityBufferBytes = 32 * 1024;
constexpr size_t kActivityBufferAlignment = 8;

struct LaunchFact {
  std::string api;
  std::string symbol;
  unsigned int grid_x;
  unsigned int grid_y;
  unsigned int grid_z;
  unsigned int block_x;
  unsigned int block_y;
  unsigned int block_z;
  unsigned int shared_memory_bytes;
  unsigned int cluster_x;
  unsigned int cluster_y;
  unsigned int cluster_z;
  std::vector<unsigned int> attribute_ids;
  bool has_cta_raster;
  unsigned int cta_raster_log_group;
  unsigned int cta_raster_serpentine;
};

/* The sm_80 cuBLASLt GEMM prologue reads its CTA-to-tile raster program from
 * one kernel parameter word.  Bits 8-11 hold the log2 width of the grouped
 * blockIdx.y band that the kernel walks M-fastest, and bit 12 enables the
 * serpentine reversal of the N direction on alternate bands.  Kernel
 * parameters start at c[0x0][0x160] on this architecture, so the word the
 * prologue reads from c[0x0][0x1e4] is at parameter offset 0x84.  The read is
 * confined to the kernel family whose disassembly defines that layout.
 */
constexpr size_t kCtaRasterParameterOffset = 0x84;
constexpr size_t kCtaRasterParameterEnd = kCtaRasterParameterOffset + 4;

bool SymbolDeclaresCtaRaster(const char* symbol) {
  if (symbol == nullptr) return false;
  const std::string name(symbol);
  return name.rfind("ampere_", 0) == 0 && name.find("gemm") != std::string::npos;
}

CUpti_SubscriberHandle subscriber;
CUmodule noop_module = nullptr;
CUfunction noop_function = nullptr;
std::atomic<bool> capture_active{false};
std::vector<LaunchFact> captured_launches;
int replacement_launch_count = 0;
std::string launch_capture_failure;

std::atomic<bool> activity_capture_active{false};
std::mutex activity_mutex;
std::vector<std::string> activity_kernel_symbols;
size_t activity_dropped_record_count = 0;
bool activity_callback_failed = false;
bool activity_unknown_record = false;

const char* kNoopPtx = R"ptx(
.version 7.8
.target sm_90
.address_size 64
.visible .entry midlayer_noop() { ret; }
)ptx";

void CUBLAS_CHECK(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void CUDA_CHECK(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             cudaGetErrorString(status));
  }
}

void DRIVER_CHECK(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with CUDA driver status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void CUPTI_CHECK(CUptiResult status, const char* operation) {
  if (status != CUPTI_SUCCESS) {
    const char* message = nullptr;
    cuptiGetResultString(status, &message);
    throw std::runtime_error(std::string(operation) + " failed: " + message);
  }
}

void RecordLaunch(const CUpti_CallbackData* data,
                  unsigned int grid_x, unsigned int grid_y,
                  unsigned int grid_z, unsigned int block_x,
                  unsigned int block_y, unsigned int block_z,
                  unsigned int shared_memory_bytes,
                  void** kernel_parameters,
                  const CUlaunchConfig* config) {
  bool has_raster = false;
  unsigned int log_group = 0;
  unsigned int serpentine = 0;
  if (SymbolDeclaresCtaRaster(data->symbolName)) {
    if (kernel_parameters == nullptr || kernel_parameters[0] == nullptr) {
      launch_capture_failure =
          "a raster-programmed GEMM launch exposed no kernel parameters";
    } else {
      unsigned int word = 0;
      std::memcpy(&word,
                  static_cast<const unsigned char*>(kernel_parameters[0])
                      + kCtaRasterParameterOffset,
                  kCtaRasterParameterEnd - kCtaRasterParameterOffset);
      has_raster = true;
      log_group = (word >> 8) & 0xfu;
      serpentine = (word >> 12) & 0x1u;
    }
  }
  unsigned int cluster_x = 1;
  unsigned int cluster_y = 1;
  unsigned int cluster_z = 1;
  std::vector<unsigned int> attribute_ids;
  if (config != nullptr) {
    for (unsigned int index = 0; index < config->numAttrs; ++index) {
      const CUlaunchAttribute& attribute = config->attrs[index];
      attribute_ids.push_back(static_cast<unsigned int>(attribute.id));
      if (attribute.id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
        cluster_x = attribute.value.clusterDim.x;
        cluster_y = attribute.value.clusterDim.y;
        cluster_z = attribute.value.clusterDim.z;
      }
    }
  }
  captured_launches.push_back({
      data->functionName,
      data->symbolName == nullptr ? "unknown" : data->symbolName,
      grid_x, grid_y, grid_z, block_x, block_y, block_z,
      shared_memory_bytes, cluster_x, cluster_y, cluster_z, attribute_ids,
      has_raster, log_group, serpentine,
  });
}

void ReplaceStandardLaunch(cuLaunchKernel_params* params) {
  params->f = noop_function;
  params->gridDimX = params->gridDimY = params->gridDimZ = 1;
  params->blockDimX = params->blockDimY = params->blockDimZ = 1;
  params->sharedMemBytes = 0;
  params->kernelParams = nullptr;
  params->extra = nullptr;
  ++replacement_launch_count;
}

void ReplaceStandardLaunch(cuLaunchKernel_ptsz_params* params) {
  params->f = noop_function;
  params->gridDimX = params->gridDimY = params->gridDimZ = 1;
  params->blockDimX = params->blockDimY = params->blockDimZ = 1;
  params->sharedMemBytes = 0;
  params->kernelParams = nullptr;
  params->extra = nullptr;
  ++replacement_launch_count;
}

template <typename Parameters>
const CUlaunchConfig* ValidateExtendedLaunch(Parameters* params) {
  if (params->config == nullptr) {
    std::fprintf(stderr, "cuLaunchKernelEx supplied a null launch config\n");
    std::abort();
  }
  if (params->config->numAttrs > 0 && params->config->attrs == nullptr) {
    launch_capture_failure =
        "cuLaunchKernelEx declared attributes without an attribute array";
  }
  return params->config;
}

template <typename Parameters>
void ReplaceExtendedLaunch(Parameters* params) {
  static thread_local CUlaunchConfig replacement;
  replacement = *params->config;
  replacement.gridDimX = replacement.gridDimY = replacement.gridDimZ = 1;
  replacement.blockDimX = replacement.blockDimY = replacement.blockDimZ = 1;
  replacement.sharedMemBytes = 0;
  replacement.attrs = nullptr;
  replacement.numAttrs = 0;
  params->config = &replacement;
  params->f = noop_function;
  params->kernelParams = nullptr;
  params->extra = nullptr;
  ++replacement_launch_count;
}

void CUPTIAPI LaunchCallback(
    void*, CUpti_CallbackDomain domain, CUpti_CallbackId callback_id,
    const void* raw_data) {
  if (!capture_active || domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
  const auto* data = static_cast<const CUpti_CallbackData*>(raw_data);
  if (data->callbackSite != CUPTI_API_ENTER) return;
  if (callback_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
    auto* params = const_cast<cuLaunchKernel_params*>(
        static_cast<const cuLaunchKernel_params*>(data->functionParams));
    RecordLaunch(data, params->gridDimX, params->gridDimY, params->gridDimZ,
                 params->blockDimX, params->blockDimY, params->blockDimZ,
                 params->sharedMemBytes, params->kernelParams, nullptr);
    ReplaceStandardLaunch(params);
  } else if (callback_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
    auto* params = const_cast<cuLaunchKernel_ptsz_params*>(
        static_cast<const cuLaunchKernel_ptsz_params*>(data->functionParams));
    RecordLaunch(data, params->gridDimX, params->gridDimY, params->gridDimZ,
                 params->blockDimX, params->blockDimY, params->blockDimZ,
                 params->sharedMemBytes, params->kernelParams, nullptr);
    ReplaceStandardLaunch(params);
  } else if (callback_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
    auto* params = const_cast<cuLaunchKernelEx_params*>(
        static_cast<const cuLaunchKernelEx_params*>(data->functionParams));
    const CUlaunchConfig* config = ValidateExtendedLaunch(params);
    RecordLaunch(data, config->gridDimX, config->gridDimY, config->gridDimZ,
                 config->blockDimX, config->blockDimY, config->blockDimZ,
                 config->sharedMemBytes, params->kernelParams, config);
    ReplaceExtendedLaunch(params);
  } else if (callback_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz) {
    auto* params = const_cast<cuLaunchKernelEx_ptsz_params*>(
        static_cast<const cuLaunchKernelEx_ptsz_params*>(data->functionParams));
    const CUlaunchConfig* config = ValidateExtendedLaunch(params);
    RecordLaunch(data, config->gridDimX, config->gridDimY, config->gridDimZ,
                 config->blockDimX, config->blockDimY, config->blockDimZ,
                 config->sharedMemBytes, params->kernelParams, config);
    ReplaceExtendedLaunch(params);
  } else if (data->functionName != nullptr
             && std::string(data->functionName).find("cuLaunch") == 0) {
    std::fprintf(stderr, "unsupported driver launch API during dry launch: %s\n",
                 data->functionName);
    std::abort();
  }
}

void CUPTIAPI ActivityBufferRequested(
    uint8_t** buffer, size_t* size, size_t* max_records) {
  void* allocation = nullptr;
  if (posix_memalign(
          &allocation, kActivityBufferAlignment, kActivityBufferBytes) != 0) {
    std::lock_guard<std::mutex> lock(activity_mutex);
    activity_callback_failed = true;
    *buffer = nullptr;
    *size = 0;
    *max_records = 0;
    return;
  }
  *buffer = static_cast<uint8_t*>(allocation);
  *size = kActivityBufferBytes;
  *max_records = 0;
}

void CUPTIAPI ActivityBufferCompleted(
    CUcontext context, uint32_t stream_id, uint8_t* buffer,
    size_t, size_t valid_size) {
  try {
    if (activity_capture_active.load()) {
      CUpti_Activity* record = nullptr;
      while (true) {
        const CUptiResult status =
            cuptiActivityGetNextRecord(buffer, valid_size, &record);
        if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        if (status != CUPTI_SUCCESS) {
          std::lock_guard<std::mutex> lock(activity_mutex);
          activity_callback_failed = true;
          break;
        }
        std::lock_guard<std::mutex> lock(activity_mutex);
        if (record->kind != CUPTI_ACTIVITY_KIND_KERNEL
            && record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
          activity_unknown_record = true;
          continue;
        }
        const auto* kernel =
            reinterpret_cast<const CUpti_ActivityKernel9*>(record);
        if (kernel->name == nullptr) {
          activity_callback_failed = true;
        } else {
          activity_kernel_symbols.emplace_back(kernel->name);
        }
      }
      size_t dropped = 0;
      const CUptiResult dropped_status = cuptiActivityGetNumDroppedRecords(
          context, stream_id, &dropped);
      std::lock_guard<std::mutex> lock(activity_mutex);
      if (dropped_status != CUPTI_SUCCESS) {
        activity_callback_failed = true;
      } else {
        activity_dropped_record_count += dropped;
      }
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(activity_mutex);
    activity_callback_failed = true;
  }
  std::free(buffer);
}

void InitializeDryLaunchCapture() {
  DRIVER_CHECK(cuModuleLoadData(&noop_module, kNoopPtx),
               "load replacement no-op module");
  DRIVER_CHECK(cuModuleGetFunction(
                   &noop_function, noop_module, "midlayer_noop"),
               "get replacement no-op function");
  CUPTI_CHECK(cuptiSubscribe(&subscriber, LaunchCallback, nullptr),
              "subscribe launch callback");
  CUPTI_CHECK(cuptiEnableDomain(
                  1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API),
              "enable driver callbacks");
  CUPTI_CHECK(cuptiActivityRegisterCallbacks(
                  ActivityBufferRequested, ActivityBufferCompleted),
              "register activity callbacks");
  CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
              "enable concurrent-kernel activity");
}

void FinalizeDryLaunchCapture() {
  CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
              "flush final activity records");
  CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
              "disable concurrent-kernel activity");
  CUPTI_CHECK(cuptiUnsubscribe(subscriber), "unsubscribe launch callback");
  DRIVER_CHECK(cuModuleUnload(noop_module), "unload replacement no-op module");
}

std::string JsonString(const std::string& value) {
  std::ostringstream stream;
  stream << '"';
  for (const char character : value) {
    if (character == '"' || character == '\\') stream << '\\';
    stream << character;
  }
  stream << '"';
  return stream.str();
}

class DryBuffers {
 public:
  DryBuffers() {
    CUDA_CHECK(cudaMalloc(&a_, 256), "allocate dry A pointer");
    CUDA_CHECK(cudaMalloc(&b_, 256), "allocate dry B pointer");
    CUDA_CHECK(cudaMalloc(&c_, 256), "allocate dry C pointer");
  }

  ~DryBuffers() noexcept(false) {
    if (workspace_ != nullptr) {
      CUDA_CHECK(cudaFree(workspace_), "release dry workspace");
    }
    CUDA_CHECK(cudaFree(c_), "release dry C pointer");
    CUDA_CHECK(cudaFree(b_), "release dry B pointer");
    CUDA_CHECK(cudaFree(a_), "release dry A pointer");
  }

  void* a() const { return a_; }
  void* b() const { return b_; }
  void* c() const { return c_; }
  void* workspace() const { return workspace_; }

  void EnsureWorkspace(size_t bytes) {
    if (bytes <= workspace_bytes_) return;
    if (workspace_ != nullptr) {
      CUDA_CHECK(cudaFree(workspace_), "release old dry workspace");
    }
    CUDA_CHECK(cudaMalloc(&workspace_, bytes), "allocate dry workspace");
    workspace_bytes_ = bytes;
  }

 private:
  void* a_ = nullptr;
  void* b_ = nullptr;
  void* c_ = nullptr;
  void* workspace_ = nullptr;
  size_t workspace_bytes_ = 0;
};

template <typename T>
T AlgoAttribute(const cublasLtMatmulAlgo_t& algo,
                cublasLtMatmulAlgoConfigAttributes_t attribute) {
  T value{};
  size_t written = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
                  &algo, attribute, &value, sizeof(value), &written),
              "cublasLtMatmulAlgoConfigGetAttribute");
  if (written != sizeof(value)) {
    throw std::runtime_error("cuBLASLt returned an unexpected attribute size");
  }
  return value;
}

int64_t PositiveInteger(const char* text, const char* name) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be a positive integer");
  }
  return value;
}

int BinaryInteger(const char* text, const char* name) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || (value != 0 && value != 1)) {
    throw std::invalid_argument(std::string(name) + " must be zero or one");
  }
  return static_cast<int>(value);
}

struct OperationRequest {
  int64_t m;
  int64_t n;
  int64_t k;
  std::string dtype_name;
  int has_bias;
  size_t max_workspace;
};

OperationRequest ParseRequest(const std::string& m_text,
                              const std::string& n_text,
                              const std::string& k_text,
                              const std::string& dtype_name,
                              const std::string& has_bias_text,
                              const std::string& max_workspace_text) {
  if (dtype_name != "bf16" && dtype_name != "fp16") {
    throw std::invalid_argument("DTYPE must be bf16 or fp16");
  }
  return {
      PositiveInteger(m_text.c_str(), "M"),
      PositiveInteger(n_text.c_str(), "N"),
      PositiveInteger(k_text.c_str(), "K"),
      dtype_name,
      BinaryInteger(has_bias_text.c_str(), "HAS_BIAS"),
      static_cast<size_t>(PositiveInteger(
          max_workspace_text.c_str(), "MAX_WORKSPACE_BYTES")),
  };
}

OperationRequest ParseRequestLine(const std::string& line) {
  std::istringstream stream(line);
  std::string m_text;
  std::string n_text;
  std::string k_text;
  std::string dtype_name;
  std::string has_bias_text;
  std::string max_workspace_text;
  std::string trailing;
  if (!(stream >> m_text >> n_text >> k_text >> dtype_name >> has_bias_text >>
        max_workspace_text) || (stream >> trailing)) {
    throw std::invalid_argument(
        "batch input line must contain M N K DTYPE HAS_BIAS "
        "MAX_WORKSPACE_BYTES");
  }
  return ParseRequest(m_text, n_text, k_text, dtype_name, has_bias_text,
                      max_workspace_text);
}

void Query(cublasLtHandle_t handle, const cudaDeviceProp& properties,
           const OperationRequest& request, DryBuffers& buffers) {
  const cudaDataType_t dtype =
      request.dtype_name == "bf16" ? CUDA_R_16BF : CUDA_R_16F;

  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(
                   &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
               "cublasLtMatmulDescCreate");
    const cublasOperation_t trans_a = CUBLAS_OP_N;
    const cublasOperation_t trans_b = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
                    operation, CUBLASLT_MATMUL_DESC_TRANSA,
                    &trans_a, sizeof(trans_a)),
                "set TRANSA");
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
                    operation, CUBLASLT_MATMUL_DESC_TRANSB,
                    &trans_b, sizeof(trans_b)),
                "set TRANSB");
  if (request.has_bias != 0) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
                     operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                     &epilogue, sizeof(epilogue)),
                 "set EPILOGUE");
    void* bias = buffers.b();
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
                     operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                     &bias, sizeof(bias)),
                 "set BIAS_POINTER");
  }

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
                  &a_layout, dtype, request.m, request.k, request.k),
                "create A layout");
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
                  &b_layout, dtype, request.n, request.k, request.k),
                "create B layout");
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
                  &c_layout, dtype, request.m, request.n, request.n),
                "create C layout");
    const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    for (cublasLtMatrixLayout_t layout : {a_layout, b_layout, c_layout}) {
      CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
                      layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                      &row_order, sizeof(row_order)),
                  "set row-major layout");
    }
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference),
                "cublasLtMatmulPreferenceCreate");
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                   preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                   &request.max_workspace, sizeof(request.max_workspace)),
                "set MAX_WORKSPACE_BYTES");

    // Retain the complete ordered list while preserving the vendor rank-0
    // selection.  Reaching the guarded capacity fails instead of silently
    // truncating.  Split-K launch geometry is captured below with every
    // production launch replaced by a one-thread no-op before submission.
    std::vector<cublasLtMatmulHeuristicResult_t> results(
        kHeuristicResultCapacity);
    int count = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
                    handle, operation, a_layout, b_layout, c_layout, c_layout,
                    preference, static_cast<int>(results.size()),
                    results.data(), &count),
                "cublasLtMatmulAlgoGetHeuristic");
    if (count <= 0) {
      throw std::runtime_error("cuBLASLt heuristic returned no algorithms");
    }
    if (count == kHeuristicResultCapacity) {
      throw std::runtime_error(
          "cuBLASLt heuristic list reached the capture capacity");
    }

    const int32_t rank0_split_k = AlgoAttribute<int32_t>(
        results[0].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
    const uint32_t rank0_reduction = AlgoAttribute<uint32_t>(
        results[0].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
    if (rank0_split_k <= 0) {
      throw std::runtime_error("cuBLASLt rank-0 split-K must be positive");
    }
    if ((rank0_split_k == 1 && rank0_reduction != 0)
        || (rank0_split_k > 1
            && rank0_reduction != CUBLASLT_REDUCTION_SCHEME_INPLACE
            && rank0_reduction != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE
            && rank0_reduction != CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE)) {
      throw std::runtime_error(
          "cuBLASLt rank-0 split/reduction contract is inconsistent");
    }
    captured_launches.clear();
    replacement_launch_count = 0;
    launch_capture_failure.clear();
    buffers.EnsureWorkspace(request.max_workspace);
    CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
                "flush activity records before query");
    {
      std::lock_guard<std::mutex> lock(activity_mutex);
      activity_kernel_symbols.clear();
      activity_dropped_record_count = 0;
      activity_callback_failed = false;
      activity_unknown_record = false;
    }
    activity_capture_active.store(true);
    capture_active.store(true);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasLtMatmul(
        handle, operation, &alpha, buffers.a(), a_layout,
        buffers.b(), b_layout, &beta, buffers.c(), c_layout,
        buffers.c(), c_layout, &results[0].algo, buffers.workspace(),
        request.max_workspace, nullptr);
    capture_active.store(false);
    CUBLAS_CHECK(status, "dry cublasLtMatmul");
    CUDA_CHECK(cudaDeviceSynchronize(), "synchronize replacement no-ops");
    CUPTI_CHECK(cuptiActivityFlushAll(0),
                "flush completed replacement-kernel activities");
    activity_capture_active.store(false);
    if (!launch_capture_failure.empty()) {
      throw std::runtime_error(launch_capture_failure);
    }
    if (captured_launches.empty()
        || replacement_launch_count
               != static_cast<int>(captured_launches.size())) {
      throw std::runtime_error(
          "dry launch did not replace every captured production launch");
    }
    int reducer_count = 0;
    for (const LaunchFact& launch : captured_launches) {
      if (launch.symbol == "unknown"
          || launch.symbol.find("midlayer_noop") != std::string::npos) {
        throw std::runtime_error(
            "dry launch captured an unknown or replacement symbol");
      }
      if (launch.symbol.find("splitKreduce_kernel") != std::string::npos) {
        ++reducer_count;
      }
    }
    std::vector<std::string> executed_symbols;
    size_t dropped_record_count = 0;
    bool callback_failed = false;
    bool unknown_record = false;
    {
      std::lock_guard<std::mutex> lock(activity_mutex);
      executed_symbols = activity_kernel_symbols;
      dropped_record_count = activity_dropped_record_count;
      callback_failed = activity_callback_failed;
      unknown_record = activity_unknown_record;
    }
    if (callback_failed || unknown_record || dropped_record_count != 0) {
      throw std::runtime_error(
          "CUPTI activity evidence is incomplete or contains unknown records");
    }
    if (executed_symbols.size()
        != static_cast<size_t>(replacement_launch_count)) {
      throw std::runtime_error(
          "CUPTI activity count differs from replaced launch count");
    }
    for (const std::string& symbol : executed_symbols) {
      if (symbol != "midlayer_noop") {
        throw std::runtime_error(
            "CUPTI activity observed a non-replacement kernel: " + symbol);
      }
    }
    // CUBLASLT_ALGO_CONFIG_SPLITK_NUM is the vendor algorithm parameter, not
    // a promise that the launched main kernel uses the same grid.z.  With a
    // separate reduction cuBLASLt can materialize fewer workspace partials
    // than the configured split count (for example splitK=12, grid.z=11).
    // Retain both source facts independently; the captured launch geometry is
    // authoritative for the physical launch.
    if (rank0_reduction == 0
        || rank0_reduction == CUBLASLT_REDUCTION_SCHEME_INPLACE) {
      if (captured_launches.size() != 1 || reducer_count != 0) {
        throw std::runtime_error(
            "rank-0 direct/in-place execution must launch one main kernel");
      }
    } else if (captured_launches.size() != 2 || reducer_count != 1
               || captured_launches.back().symbol.find(
                      "splitKreduce_kernel") == std::string::npos) {
      throw std::runtime_error(
          "separate-step split-K must launch one main then one reducer");
    }

    std::cout << "{\"schema_version\":2,\"hardware_id\":\"nvidia_h200\",";
    std::cout << "\"device_name\":\"" << properties.name << "\",";
    std::cout << "\"compute_capability\":\"" << properties.major << "."
              << properties.minor << "\",";
    std::cout << "\"cublaslt_version\":" << cublasLtGetVersion() << ",";
    std::cout << "\"operation\":{\"m\":" << request.m << ",\"n\":"
              << request.n << ",\"k\":" << request.k << ",\"dtype\":\""
              << request.dtype_name
              << "\",";
    std::cout << "\"layout\":\"row_major\",\"trans_a\":false,";
    std::cout << "\"trans_b\":true,\"has_bias\":"
              << (request.has_bias != 0 ? "true" : "false") << "},";
    std::cout << "\"max_workspace_bytes\":" << request.max_workspace << ",";
    std::cout << "\"matmul_executed\":false,";
    std::cout << "\"production_kernel_submitted\":false,";
    std::cout << "\"dry_launch\":{";
    std::cout << "\"attempted\":true,";
    std::cout << "\"replacement_kernel_submitted\":true,";
    std::cout << "\"replacement_launch_count\":"
              << replacement_launch_count << ",\"launches\":[";
    for (size_t index = 0; index < captured_launches.size(); ++index) {
      if (index != 0) std::cout << ',';
      const LaunchFact& launch = captured_launches[index];
      std::cout << "{\"api\":" << JsonString(launch.api)
                << ",\"symbol\":" << JsonString(launch.symbol)
                << ",\"grid\":[" << launch.grid_x << ',' << launch.grid_y
                << ',' << launch.grid_z << "],\"block\":["
                << launch.block_x << ',' << launch.block_y << ','
                << launch.block_z << "],\"shared_memory_bytes\":"
                << launch.shared_memory_bytes
                << ",\"cluster\":[" << launch.cluster_x << ','
                << launch.cluster_y << ',' << launch.cluster_z
                << "],\"launch_attribute_ids\":[";
      for (size_t a = 0; a < launch.attribute_ids.size(); ++a) {
        if (a != 0) std::cout << ',';
        std::cout << launch.attribute_ids[a];
      }
      std::cout << "],\"cta_raster_log_group\":"
                << (launch.has_cta_raster
                        ? std::to_string(launch.cta_raster_log_group)
                        : std::string("null"))
                << ",\"cta_raster_serpentine\":"
                << (launch.has_cta_raster
                        ? std::to_string(launch.cta_raster_serpentine)
                        : std::string("null"))
                << '}';
    }
    std::cout << "],\"device_execution\":{";
    std::cout << "\"evidence_kind\":\"cupti_activity_kernel\",";
    std::cout << "\"evidence_version\":1,";
    std::cout << "\"activity_record_count\":"
              << executed_symbols.size() << ",\"executed_symbols\":[";
    for (size_t index = 0; index < executed_symbols.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << JsonString(executed_symbols[index]);
    }
    std::cout << "],\"all_replacements_observed\":true,";
    std::cout << "\"dropped_record_count\":" << dropped_record_count;
    std::cout << "}},\"algorithms\":[";
    for (int index = 0; index < count; ++index) {
      if (index != 0) std::cout << ',';
      const auto& result = results[index];
      const int32_t algo_id = AlgoAttribute<int32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_ID);
      const uint32_t tile_id = AlgoAttribute<uint32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
      const int32_t split_k = AlgoAttribute<int32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
      const uint32_t reduction = AlgoAttribute<uint32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
      const uint32_t swizzle = AlgoAttribute<uint32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
      const uint32_t custom = AlgoAttribute<uint32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
      const uint32_t stages_id = AlgoAttribute<uint32_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
      const uint16_t inner_shape = AlgoAttribute<uint16_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID);
      const uint16_t cluster_shape = AlgoAttribute<uint16_t>(
          result.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
      std::cout << "{\"rank\":" << index << ",\"algorithm_id\":" << algo_id
                << ",\"tile_id\":" << tile_id
                << ",\"stages_id\":" << stages_id
                << ",\"cluster_shape_id\":" << cluster_shape
                << ",\"inner_shape_id\":" << inner_shape
                << ",\"split_k\":" << split_k
                << ",\"reduction_scheme\":" << reduction
                << ",\"cta_swizzle\":" << swizzle
                << ",\"custom_option\":" << custom
                << ",\"workspace_bytes\":" << result.workspaceSize
                << ",\"waves_count\":" << result.wavesCount << '}';
    }
    std::cout << "]}\n";

    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference),
                "cublasLtMatmulPreferenceDestroy");
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_layout), "destroy C layout");
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_layout), "destroy B layout");
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_layout), "destroy A layout");
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation), "destroy operation");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 1 && argc != 7) {
      throw std::invalid_argument(
          "usage: cublaslt_heuristic_query [M N K DTYPE HAS_BIAS "
          "MAX_WORKSPACE_BYTES]; with no arguments, read requests from stdin");
    }
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0),
               "cudaGetDeviceProperties");
    DryBuffers buffers;
    InitializeDryLaunchCapture();
    cublasLtHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasLtCreate(&handle), "cublasLtCreate");
    if (argc == 7) {
      Query(handle, properties, ParseRequest(
          argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]), buffers);
    } else {
      std::string line;
      int request_count = 0;
      while (std::getline(std::cin, line)) {
        if (line.empty()) {
          throw std::invalid_argument("batch input cannot contain empty lines");
        }
        Query(handle, properties, ParseRequestLine(line), buffers);
        ++request_count;
      }
      if (request_count == 0) {
        throw std::invalid_argument("batch input must contain at least one request");
      }
    }
    CUBLAS_CHECK(cublasLtDestroy(handle), "cublasLtDestroy");
    FinalizeDryLaunchCapture();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
