// NVIDIA H200 CUDA-graph dependent-node gap versus kernel launch properties.
//
// Mechanism probe (no production kernel, input or duration).  The trivial-node
// gap probe (cuda_graph_node_gap.cu, --switch) measures 64-96 ns between
// dependent graph nodes, far below the ~0.45 us seen between production
// kernels of a replayed decode graph.  Production kernels differ from the
// trivial node in launch properties, not in work: large kernel parameter
// blocks (TMA descriptors, __grid_constant__ structs), thread-block clusters,
// cooperative launch, high register counts, large dynamic shared memory.  This
// probe gives each property its own kernel with the same trivial body and
// measures the gap in front of it.
//
// Timing: every node records, with %globaltimer, the start of its first CTA
// (atomicMin) and the end of its last CTA (atomicMax) into a per-node slot;
// the gap in front of node i+1 is start(i+1) - end(i) inside one replay.
// A graph is a chain of NODES nodes alternating the trivial kernel (even) and
// the variant kernel (odd); per variant, GRID CTAs x 128 threads, REPLAYS
// replays after 3 warm-up replays.  Output CSV: variant,grid,replay,node,
// gap_before_ns,node_span_ns,variant_registers (even nodes: the gap after the
// variant node, in front of the next trivial node).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -cudart static -o launch_property_node_gap launch_property_node_gap.cu
// Usage: launch_property_node_gap <nodes> <replays> <grid>
// (nodes up to 4096: a decode graph replays ~2000 dependent nodes)

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) do {                                              \
    const cudaError_t error = (call);                                      \
    if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(error));       \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace {

constexpr int kThreads = 128;
constexpr int kMaxNodes = 4096;
constexpr int kBigParamBytes = 4000;

struct Slot {
    unsigned long long start;
    unsigned long long end;
};

struct BigParam {
    unsigned char bytes[kBigParamBytes];
};

__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void mark_start(Slot* slot) {
    if (threadIdx.x == 0) atomicMin(&slot->start, now_ns());
}

__device__ __forceinline__ void mark_end(Slot* slot, unsigned int* sink,
                                         unsigned int value) {
    __syncthreads();
    if (threadIdx.x == 0) {
        sink[blockIdx.x] += value;
        __threadfence();
        atomicMax(&slot->end, now_ns());
    }
}

__global__ void __launch_bounds__(kThreads) trivial(Slot* slot,
                                                    unsigned int* sink) {
    mark_start(slot);
    mark_end(slot, sink, 1U);
}

__global__ void __launch_bounds__(kThreads) big_param(
        Slot* slot, unsigned int* sink, const __grid_constant__ BigParam p) {
    mark_start(slot);
    mark_end(slot, sink, p.bytes[threadIdx.x % kBigParamBytes]);
}

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(kThreads)
cluster2(Slot* slot, unsigned int* sink) {
    mark_start(slot);
    mark_end(slot, sink, 1U);
}

__global__ void __launch_bounds__(kThreads) cooperative(Slot* slot,
                                                        unsigned int* sink) {
    mark_start(slot);
    mark_end(slot, sink, 1U);
}

// Register pressure: 160 live accumulators with a data-dependent update keep
// the compiler from folding them (the count is reported per run).
__global__ void __launch_bounds__(kThreads) high_registers(
        Slot* slot, unsigned int* sink, unsigned int seed) {
    mark_start(slot);
    float acc[160];
#pragma unroll
    for (int i = 0; i < 160; ++i) acc[i] = seed * (i + 1) * 1e-3f;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int i = 0; i < 160; ++i) acc[i] = acc[i] * acc[(i + 1) % 160] + 1.0f;
    }
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < 160; ++i) total += acc[i];
    mark_end(slot, sink, total > 1e30f ? 2U : 1U);
}

// Store volume: every CTA writes 16 KiB (8 x 128 threads x 16 B) before its
// end mark, so the successor's gap includes completing those stores.
__global__ void __launch_bounds__(kThreads) store_heavy(Slot* slot,
                                                        unsigned int* sink,
                                                        uint4* out) {
    mark_start(slot);
    uint4* base = out + static_cast<size_t>(blockIdx.x) * 8 * kThreads;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        base[i * kThreads + threadIdx.x] =
            make_uint4(threadIdx.x, blockIdx.x, i, 1U);
    }
    mark_end(slot, sink, 1U);
}

// Programmatic dependent launch, as SGLang's sgl-kernel / flashinfer /
// DeepGEMM launches: the node may start before its predecessor completes and
// waits in griddepcontrol.wait; it releases its own dependents early.
__global__ void __launch_bounds__(kThreads) pdl_node(Slot* slot,
                                                     unsigned int* sink) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    mark_start(slot);
    asm volatile("griddepcontrol.launch_dependents;");
    mark_end(slot, sink, 1U);
}

__global__ void __launch_bounds__(kThreads) big_smem(Slot* slot,
                                                     unsigned int* sink) {
    extern __shared__ unsigned int scratch[];
    mark_start(slot);
    scratch[threadIdx.x] = threadIdx.x;
    mark_end(slot, sink, scratch[kThreads - 1 - threadIdx.x]);
}

enum class Variant { kTrivial, kBigParam, kCluster2, kCooperative,
                     kHighRegisters, kBigSmem, kStoreHeavy, kPdl };

const char* name(Variant v) {
    switch (v) {
        case Variant::kTrivial: return "trivial";
        case Variant::kBigParam: return "big_param_4000B";
        case Variant::kCluster2: return "cluster2";
        case Variant::kCooperative: return "cooperative";
        case Variant::kHighRegisters: return "high_registers";
        case Variant::kBigSmem: return "big_smem_200KB";
        case Variant::kStoreHeavy: return "store_16KB_per_cta";
        case Variant::kPdl: return "pdl";
    }
    return "?";
}

constexpr int kBigSmemBytes = 200 * 1024;

uint4* g_store_out = nullptr;

void launch(Variant v, int grid, Slot* slot, unsigned int* sink,
            const BigParam& p, cudaStream_t stream) {
    switch (v) {
        case Variant::kTrivial:
            trivial<<<grid, kThreads, 0, stream>>>(slot, sink);
            break;
        case Variant::kBigParam:
            big_param<<<grid, kThreads, 0, stream>>>(slot, sink, p);
            break;
        case Variant::kCluster2:
            cluster2<<<grid, kThreads, 0, stream>>>(slot, sink);
            break;
        case Variant::kCooperative: {
            void* args[] = {&slot, &sink};
            CHECK_CUDA(cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(cooperative), dim3(grid),
                dim3(kThreads), args, 0, stream));
            return;
        }
        case Variant::kHighRegisters:
            high_registers<<<grid, kThreads, 0, stream>>>(slot, sink, 3U);
            break;
        case Variant::kBigSmem:
            big_smem<<<grid, kThreads, kBigSmemBytes, stream>>>(slot, sink);
            break;
        case Variant::kStoreHeavy:
            store_heavy<<<grid, kThreads, 0, stream>>>(slot, sink,
                                                       g_store_out);
            break;
        case Variant::kPdl: {
            cudaLaunchConfig_t config = {};
            config.gridDim = dim3(grid);
            config.blockDim = dim3(kThreads);
            config.stream = stream;
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            config.attrs = attribute;
            config.numAttrs = 1;
            CHECK_CUDA(cudaLaunchKernelEx(&config, pdl_node, slot, sink));
            return;
        }
    }
    CHECK_CUDA(cudaGetLastError());
}

int registers_of(Variant v) {
    cudaFuncAttributes a;
    const void* f = nullptr;
    switch (v) {
        case Variant::kTrivial: f = reinterpret_cast<const void*>(trivial); break;
        case Variant::kBigParam: f = reinterpret_cast<const void*>(big_param); break;
        case Variant::kCluster2: f = reinterpret_cast<const void*>(cluster2); break;
        case Variant::kCooperative: f = reinterpret_cast<const void*>(cooperative); break;
        case Variant::kHighRegisters: f = reinterpret_cast<const void*>(high_registers); break;
        case Variant::kBigSmem: f = reinterpret_cast<const void*>(big_smem); break;
        case Variant::kStoreHeavy: f = reinterpret_cast<const void*>(store_heavy); break;
        case Variant::kPdl: f = reinterpret_cast<const void*>(pdl_node); break;
    }
    CHECK_CUDA(cudaFuncGetAttributes(&a, f));
    return a.numRegs;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <nodes> <replays> <grid>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int nodes = std::atoi(argv[1]);
    const int replays = std::atoi(argv[2]);
    const int grid = std::atoi(argv[3]);
    if (nodes < 4 || nodes > kMaxNodes || nodes % 2 || replays < 1
            || grid < 2 || grid % 2 || grid > 132) {
        std::fprintf(stderr, "nodes 4..4096 even, replays >= 1, grid 2..132 even\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaFuncSetAttribute(
        big_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, kBigSmemBytes));
    Slot* slots = nullptr;
    unsigned int* sink = nullptr;
    CHECK_CUDA(cudaMalloc(&slots, sizeof(Slot) * kMaxNodes));
    CHECK_CUDA(cudaMalloc(&sink, sizeof(unsigned int) * 1024));
    CHECK_CUDA(cudaMemset(sink, 0, sizeof(unsigned int) * 1024));
    CHECK_CUDA(cudaMalloc(&g_store_out, sizeof(uint4) * 8 * kThreads * 132));
    BigParam p;
    for (int i = 0; i < kBigParamBytes; ++i) p.bytes[i] = static_cast<unsigned char>(i);
    std::vector<Slot> reset(kMaxNodes, Slot{~0ULL, 0ULL});
    std::vector<Slot> host(kMaxNodes);
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::printf("variant,grid,replay,node,gap_before_ns,node_span_ns,variant_registers\n");
    for (Variant v : {Variant::kTrivial, Variant::kBigParam, Variant::kCluster2,
                      Variant::kCooperative, Variant::kHighRegisters,
                      Variant::kBigSmem, Variant::kStoreHeavy, Variant::kPdl}) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int node = 0; node < nodes; ++node) {
            launch(node % 2 ? v : Variant::kTrivial, grid, slots + node, sink,
                   p, stream);
        }
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
        const int regs = registers_of(v);
        for (int replay = -3; replay < replays; ++replay) {
            CHECK_CUDA(cudaMemcpyAsync(slots, reset.data(), sizeof(Slot) * nodes,
                                       cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaGraphLaunch(exec, stream));
            CHECK_CUDA(cudaMemcpyAsync(host.data(), slots, sizeof(Slot) * nodes,
                                       cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            if (replay < 0) continue;
            for (int node = 1; node < nodes; ++node) {
                std::printf("%s,%d,%d,%d,%lld,%lld,%d\n",
                            name(node % 2 ? v : Variant::kTrivial), grid, replay,
                            node,
                            static_cast<long long>(host[node].start)
                                - static_cast<long long>(host[node - 1].end),
                            static_cast<long long>(host[node].end)
                                - static_cast<long long>(host[node].start),
                            node % 2 ? regs : registers_of(Variant::kTrivial));
            }
        }
        CHECK_CUDA(cudaGraphExecDestroy(exec));
        CHECK_CUDA(cudaGraphDestroy(graph));
    }
    CHECK_CUDA(cudaFree(slots));
    CHECK_CUDA(cudaFree(sink));
    return EXIT_SUCCESS;
}
