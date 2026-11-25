/*
 * Independent SM80 operand interleave traffic probe.
 *
 * A block reads A and B panels in a cuBLASLt-style grouped CTA order and
 * stages them in shared memory. There is no matrix multiply, output matrix,
 * production operator, or duration estimator. NCU counters determine whether
 * repeated operand panels remain resident as CTA cohorts increase.
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int kTileM = 128;
constexpr int kTileN = 256;
constexpr int kTileK = 64;
constexpr int kK = 2048;
constexpr int kThreads = 256;
constexpr int kPanelAVectors = kTileM * kTileK * 2 / 16;
constexpr int kPanelBVectors = kTileN * kTileK * 2 / 16;
constexpr int kSharedBytes = 98304;

static void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(2);
    }
}

__device__ __forceinline__ void grouped_tile(int& m_tile, int& n_tile) {
    const int row = static_cast<int>(blockIdx.y);
    const int column = static_cast<int>(blockIdx.x);
    int width = 16;
    bool residual_three = false;
    while (width > 1) {
        const int mask = width - 1;
        if ((row | mask) < static_cast<int>(gridDim.y)) break;
        if (mask == 3 && (row | mask) == static_cast<int>(gridDim.y)) {
            residual_three = true;
            break;
        }
        width >>= 1;
    }
    const int mask = width - 1;
    const int linear = (row & mask) * static_cast<int>(gridDim.x) + column;
    if (residual_three) {
        n_tile = linear / 3;
        m_tile = (row & ~mask) + linear % 3;
    } else {
        n_tile = linear / width;
        m_tile = (row & ~mask) + linear % width;
    }
    if ((row & 16) == 16) n_tile = static_cast<int>(gridDim.x) - 1 - n_tile;
}

__device__ __forceinline__ void stage_vector(
        uint4* destination, const uint4* source, bool async_copy) {
    if (async_copy) {
        const uint32_t shared_address =
            static_cast<uint32_t>(__cvta_generic_to_shared(destination));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(shared_address), "l"(source));
    } else {
        *destination = __ldcg(source);
    }
}

__global__ void operand_interleave_probe(const uint4* __restrict__ a,
                                         const uint4* __restrict__ b,
                                         unsigned int* __restrict__ checksums,
                                         unsigned int* __restrict__ tile_ids,
                                         bool include_a, bool async_copy) {
    extern __shared__ uint4 panels[];
    uint4* shared_a = panels;
    uint4* shared_b = panels + kPanelAVectors;
    const int lane = static_cast<int>(threadIdx.x);
    int m_tile, n_tile;
    grouped_tile(m_tile, n_tile);
    unsigned int checksum = 0;
    for (int round = 0; round < kK / kTileK; ++round) {
        if (include_a) {
            for (int vector = lane; vector < kPanelAVectors; vector += kThreads) {
                const int row = vector / 8;
                const int column = vector % 8;
                const int global_vector =
                    (m_tile * kTileM + row) * (kK / 8)
                    + round * (kTileK / 8) + column;
                stage_vector(shared_a + vector, a + global_vector, async_copy);
            }
        }
        for (int vector = lane; vector < kPanelBVectors; vector += kThreads) {
            const int row = vector / 8;
            const int column = vector % 8;
            const int global_vector =
                (n_tile * kTileN + row) * (kK / 8)
                + round * (kTileK / 8) + column;
            stage_vector(shared_b + vector, b + global_vector, async_copy);
        }
        if (async_copy) {
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;");
        }
        __syncthreads();
        for (int vector = lane; vector < kPanelBVectors; vector += kThreads) {
            checksum += shared_b[vector].x;
        }
        if (include_a) {
            for (int vector = lane; vector < kPanelAVectors; vector += kThreads) {
                checksum += shared_a[vector].x;
            }
        }
        __syncthreads();
    }
    checksums[blockIdx.y * gridDim.x + blockIdx.x] = checksum;
    tile_ids[blockIdx.y * gridDim.x + blockIdx.x] =
        (static_cast<unsigned int>(m_tile) << 16) | static_cast<unsigned int>(n_tile);
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <m_tiles:8|32> <n_tiles:32|64|96> <include_a:0|1> <async_copy:0|1>\n", argv[0]);
        return 2;
    }
    const int m_tiles = std::atoi(argv[1]);
    const int n_tiles = std::atoi(argv[2]);
    const int include_a = std::atoi(argv[3]);
    const int async_copy = std::atoi(argv[4]);
    if ((m_tiles != 8 && m_tiles != 32) || (include_a != 0 && include_a != 1)
            || (n_tiles != 32 && n_tiles != 64 && n_tiles != 96)
            || (async_copy != 0 && async_copy != 1)) {
        std::fprintf(stderr, "invalid mechanism cell\n");
        return 2;
    }
    uint4 *a = nullptr, *b = nullptr;
    unsigned int *checksums = nullptr, *tile_ids = nullptr;
    const size_t a_bytes = size_t(m_tiles) * kTileM * kK * 2;
    const size_t b_bytes = size_t(n_tiles) * kTileN * kK * 2;
    checked(cudaMalloc(&a, a_bytes), "cudaMalloc(A)");
    checked(cudaMalloc(&b, b_bytes), "cudaMalloc(B)");
    checked(cudaMalloc(&checksums, size_t(m_tiles) * n_tiles * sizeof(unsigned int)),
            "cudaMalloc(checksums)");
    checked(cudaMalloc(&tile_ids, size_t(m_tiles) * n_tiles * sizeof(unsigned int)),
            "cudaMalloc(tile_ids)");
    checked(cudaMemset(a, 1, a_bytes), "cudaMemset(A)");
    checked(cudaMemset(b, 1, b_bytes), "cudaMemset(B)");
    checked(cudaFuncSetAttribute(operand_interleave_probe,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes),
            "cudaFuncSetAttribute(shared)");
    operand_interleave_probe<<<dim3(n_tiles, m_tiles, 1), kThreads, kSharedBytes>>>(
        a, b, checksums, tile_ids, include_a != 0, async_copy != 0);
    checked(cudaGetLastError(), "operand_interleave_probe launch");
    checked(cudaDeviceSynchronize(), "operand_interleave_probe completion");
    std::vector<unsigned int> observed(size_t(m_tiles) * n_tiles);
    checked(cudaMemcpy(observed.data(), tile_ids,
            observed.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost),
            "cudaMemcpy(tile_ids)");
    std::vector<unsigned char> covered(observed.size(), 0);
    for (unsigned int encoded : observed) {
        const unsigned int m_tile = encoded >> 16;
        const unsigned int n_tile = encoded & 0xffffU;
        if (m_tile >= static_cast<unsigned int>(m_tiles)
                || n_tile >= static_cast<unsigned int>(n_tiles)
                || covered[size_t(m_tile) * n_tiles + n_tile] != 0) {
            std::fprintf(stderr, "CTA raster is not a one-to-one tile map\n");
            return 3;
        }
        covered[size_t(m_tile) * n_tiles + n_tile] = 1;
    }
    std::printf("m_tiles=%d,n_tiles=%d,k=%d,include_a=%d,async_copy=%d,grid_ctas=%d,"
                "b_request_bytes=%zu,a_request_bytes=%zu\n",
                m_tiles, n_tiles, kK, include_a, async_copy, m_tiles * n_tiles,
                size_t(m_tiles) * b_bytes,
                include_a ? size_t(m_tiles) * n_tiles * kTileM * kK * 2 : 0);
    checked(cudaFree(checksums), "cudaFree(checksums)");
    checked(cudaFree(tile_ids), "cudaFree(tile_ids)");
    checked(cudaFree(b), "cudaFree(B)");
    checked(cudaFree(a), "cudaFree(A)");
    return 0;
}
