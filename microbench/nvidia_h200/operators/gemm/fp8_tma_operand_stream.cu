// H200 TMA operand-ring streaming mechanism probe (no math, no scaling).
//
// Measured mechanism: one persistent CTA per SM runs the Hopper
// producer/consumer shared-memory ring that TMA-fed kernels use.  A single
// elected producer thread waits for a free stage, posts the stage's expected
// byte count on its full mbarrier and issues 2D cp.async.bulk.tensor loads of
//   * one "streamed" box  (rows_stream x 128 B, K-major, 128-B swizzle) whose
//     rows are private to the CTA and rotate through a buffer several times
//     larger than L2, so every launch reads them from DRAM, and
//   * one optional "shared" box (rows_shared x 128 B) that every CTA reads
//     from the same small matrix (L2 resident after the first touch).
// Consumer warps (the math warp-groups of the launch geometry) only wait on
// the full barrier and release the stage (one arrive per warp on the empty
// barrier).  No MMA, no scale arithmetic and no production data layout
// beyond the box geometry is executed, so the kernel duration measures the
// launch shell (kblocks = 0), the ring fill latency and the per-stage TMA
// ingress service as a function of active CTAs, ring depth and stage bytes.
// An optional epilogue stores rows_store x cols_store BF16 bytes per CTA with
// a bulk shared->global copy and waits for it (store completion service).
//
// Each configuration is preceded by one marker launch whose grid.x equals
// 1000 + configuration index so the nsys GPU trace can be split.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a fp8_tma_operand_stream.cu -o probe
// Usage: probe <config_file> <launches_per_config>
//   config_file lines: grid active_ctas threads stages rows_stream
//                      rows_shared kblocks cluster store_bytes smem_bytes
//   CTAs with blockIdx.x >= active_ctas initialise the barriers, synchronise
//   and exit (the persistent-scheduler idle CTAs of a one-wave launch).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
  exit(1); } } while (0)

constexpr int kBoxInnerBytes = 128;      // BLOCK_K = 128 FP8 elements
constexpr int kMaxStages = 16;

struct Config {
  int grid, active_ctas, threads, stages, rows_stream, rows_shared, kblocks,
      cluster, store_bytes, smem_bytes;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
               "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
               "r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::
               "r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" :: "r"(smem_u32(bar)), "r"(parity));
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map,
                                            uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];" ::
      "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}

__global__ void marker_kernel() {}

__global__ void __launch_bounds__(384, 1)
tma_stream_kernel(const __grid_constant__ CUtensorMap stream_map,
                  const __grid_constant__ CUtensorMap shared_map,
                  uint8_t* store_out, int stages, int rows_stream,
                  int rows_shared, int kblocks, int row_base, int store_bytes,
                  int math_threads, int active_ctas) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int stream_stage_bytes = rows_stream * kBoxInnerBytes;
  const int shared_stage_bytes = rows_shared * kBoxInnerBytes;
  uint8_t* stream_buf = smem;
  uint8_t* shared_buf = smem + stages * stream_stage_bytes;
  uint8_t* store_buf = shared_buf + stages * shared_stage_bytes;
  uint64_t* bars = reinterpret_cast<uint64_t*>(
      store_buf + ((store_bytes + 127) / 128) * 128);
  uint64_t* full = bars;
  uint64_t* empty = bars + kMaxStages;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int math_warps = math_threads / 32;
  if (warp == math_warps + 1 && lane == 0) {
    for (int s = 0; s < stages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], math_warps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (static_cast<int>(blockIdx.x) >= active_ctas) return;
  if (warp >= math_warps) {
    if (warp == math_warps + 2 && lane == 0) {
      const int row = row_base + blockIdx.x * rows_stream;
      int stage = 0;
      uint32_t phase = 0;
      for (int kb = 0; kb < kblocks; ++kb) {
        mbar_wait(&empty[stage], phase ^ 1);
        mbar_expect_tx(&full[stage], stream_stage_bytes + shared_stage_bytes);
        tma_load_2d(stream_buf + stage * stream_stage_bytes, &stream_map,
                    &full[stage], kb * kBoxInnerBytes, row);
        if (rows_shared > 0) {
          tma_load_2d(shared_buf + stage * shared_stage_bytes, &shared_map,
                      &full[stage], kb * kBoxInnerBytes, 0);
        }
        stage = stage == stages - 1 ? 0 : stage + 1;
        phase ^= stage == 0;
      }
    }
  } else {
    int stage = 0;
    uint32_t phase = 0;
    uint32_t sink = 0;
    for (int kb = 0; kb < kblocks; ++kb) {
      mbar_wait(&full[stage], phase);
      sink += stream_buf[stage * stream_stage_bytes + threadIdx.x];
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[stage]);
      stage = stage == stages - 1 ? 0 : stage + 1;
      phase ^= stage == 0;
    }
    if (store_bytes > 0) {
      for (int i = threadIdx.x; i < store_bytes; i += math_threads)
        store_buf[i] = static_cast<uint8_t>(sink + i);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync 1, %0;" :: "r"(math_threads));
      if (threadIdx.x == 0) {
        asm volatile(
            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
            "l"(store_out + static_cast<size_t>(blockIdx.x) * store_bytes),
            "r"(smem_u32(store_buf)), "r"(store_bytes) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      }
    } else if (sink == 0xFFFFFFFFu) {
      store_out[blockIdx.x] = 1;
    }
  }
}

static PFN_cuTensorMapEncodeTiled_v12000 encode_tiled() {
  void* fn = nullptr;
  cudaDriverEntryPointQueryResult status;
  CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn,
                                cudaEnableDefault, &status));
  if (fn == nullptr || status != cudaDriverEntryPointSuccess) {
    fprintf(stderr, "cuTensorMapEncodeTiled unavailable\n");
    exit(1);
  }
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
}

static CUtensorMap make_map(void* base, uint64_t rows, uint64_t row_bytes,
                            uint32_t box_rows) {
  CUtensorMap map;
  cuuint64_t dims[2] = {row_bytes, rows};
  cuuint64_t strides[1] = {row_bytes};
  cuuint32_t box[2] = {static_cast<cuuint32_t>(kBoxInnerBytes),
                       box_rows == 0 ? 1u : box_rows};
  cuuint32_t elem[2] = {1, 1};
  CUresult r = encode_tiled()(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, elem,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "cuTensorMapEncodeTiled failed %d\n", static_cast<int>(r));
    exit(1);
  }
  return map;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <config_file> <launches_per_config>\n", argv[0]);
    return 2;
  }
  std::vector<Config> configs;
  std::ifstream in(argv[1]);
  Config c;
  while (in >> c.grid >> c.active_ctas >> c.threads >> c.stages >> c.rows_stream >>
         c.rows_shared >> c.kblocks >> c.cluster >> c.store_bytes >>
         c.smem_bytes) {
    configs.push_back(c);
  }
  const int launches = atoi(argv[2]);
  // Rotating DRAM operand: 512 MiB of 7168-byte (K=7168 FP8) rows.
  const uint64_t row_bytes = 7168;
  const uint64_t stream_rows = (512ull << 20) / row_bytes;
  uint8_t* stream_mem = nullptr;
  uint8_t* shared_mem = nullptr;
  uint8_t* store_mem = nullptr;
  CHECK(cudaMalloc(&stream_mem, stream_rows * row_bytes));
  CHECK(cudaMalloc(&shared_mem, 256 * row_bytes));
  CHECK(cudaMalloc(&store_mem, 132ull * 65536));
  CHECK(cudaMemset(stream_mem, 1, stream_rows * row_bytes));
  CHECK(cudaMemset(shared_mem, 1, 256 * row_bytes));
  CHECK(cudaFuncSetAttribute(tma_stream_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             232448));
  CHECK(cudaFuncSetAttribute(tma_stream_kernel,
                             cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  // The streamed-row cursor advances across every launch of every
  // configuration, so a box is not revisited before the cursor wraps the
  // 512-MiB buffer (> 10x the 50-MiB L2).
  uint64_t next_row = 0;
  for (size_t index = 0; index < configs.size(); ++index) {
    const Config& cfg = configs[index];
    const int math_threads = cfg.threads - 128;
    const int needed = cfg.stages * (cfg.rows_stream + cfg.rows_shared) *
                           kBoxInnerBytes +
                       ((cfg.store_bytes + 127) / 128) * 128 +
                       2 * kMaxStages * 8;
    if (cfg.smem_bytes < needed || cfg.stages > kMaxStages ||
        cfg.kblocks * kBoxInnerBytes > static_cast<int>(row_bytes) ||
        cfg.active_ctas > cfg.grid || cfg.grid % cfg.cluster != 0) {
      fprintf(stderr, "invalid config %zu\n", index);
      return 3;
    }
    CUtensorMap stream_map = make_map(stream_mem, stream_rows, row_bytes,
                                      cfg.rows_stream);
    CUtensorMap shared_map = make_map(shared_mem, 256, row_bytes,
                                      cfg.rows_shared);
    const uint64_t rows_per_launch =
        static_cast<uint64_t>(cfg.active_ctas) * cfg.rows_stream;
    marker_kernel<<<1000 + static_cast<int>(index), 32>>>();
    CHECK(cudaGetLastError());
    for (int launch = 0; launch < launches; ++launch) {
      if (next_row + rows_per_launch > stream_rows) next_row = 0;
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(cfg.grid, 1, 1);
      lc.blockDim = dim3(cfg.threads, 1, 1);
      lc.dynamicSmemBytes = cfg.smem_bytes;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = cfg.cluster;
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      lc.attrs = attr;
      lc.numAttrs = 1;
      CHECK(cudaLaunchKernelEx(&lc, tma_stream_kernel, stream_map, shared_map,
                               store_mem, cfg.stages, cfg.rows_stream,
                               cfg.rows_shared, cfg.kblocks,
                               static_cast<int>(next_row), cfg.store_bytes,
                               math_threads, cfg.active_ctas));
      next_row += rows_per_launch;
    }
    CHECK(cudaDeviceSynchronize());
    printf("config %zu grid %d active %d threads %d stages %d rows_stream %d "
           "rows_shared %d kblocks %d cluster %d store %d smem %d\n",
           index, cfg.grid, cfg.active_ctas, cfg.threads, cfg.stages, cfg.rows_stream,
           cfg.rows_shared, cfg.kblocks, cfg.cluster, cfg.store_bytes,
           cfg.smem_bytes);
  }
  CHECK(cudaDeviceSynchronize());
  return 0;
}
