/*
 * FusedMoE A-gather BW sweep: buffer size (L2-resident vs >L2), BM, and
 * production-width transaction granularity.  The original FP32/BK=32 sweep
 * issues one exact 128-byte cache line per gathered row.  The two frozen BF16
 * launch families use BM16/BK64 (128 logical bytes per row) and BM64/BK32
 * (64 logical bytes per row), so the uint16_t storage mode independently tests
 * both signatures and whether half-line overfetch changes delivered logical
 * bandwidth.  No operator target duration or profiler counter is consumed.
 *
 * Build in the prescribed C500 container:
 *   /opt/maca/mxgpu_llvm/bin/mxcc -x maca -offload-arch native \
 *       --maca-path=/opt/maca -O3 -std=c++17 \
 *       -o fused_moe_gather_bw_sweep fused_moe_gather_bw_sweep.cu
 */
#include <mc_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MC_CHECK(call) do {                                               \
    mcError_t error = (call);                                             \
    if (error != mcSuccess) {                                             \
        fprintf(stderr, "MACA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, mcGetErrorString(error));             \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

template<typename T, int BM, int BK>
__global__ void gather_kernel(const T* __restrict__ act,
                               const int* __restrict__ row_idx,
                               int K, int EM, int n_k_groups, float* __restrict__ out)
{
    __shared__ T a_tile[BM][BK];
    int pid_m = blockIdx.x, tid = threadIdx.x, m_start = pid_m * BM;
    if (m_start >= EM) return;
    float acc = 0.f;
    for (int g = 0; g < n_k_groups; g++) {
        int kk = g * BK;
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int r = i / BK, c = i % BK;
            int src = row_idx[m_start + r];
            a_tile[r][c] = act[(size_t)src * K + (kk + c)];
        }
        __syncthreads();
        for (int i = 0; i < BK; i++) {
            acc += static_cast<float>(a_tile[tid % BM][i]);
        }
        __syncthreads();
    }
    if (tid == 0) out[pid_m] = acc;
}

static void fill_indices(int* indices, int TOK, int EM, bool scattered) {
    int* permutation = static_cast<int*>(malloc((size_t)TOK * sizeof(int)));
    if (permutation == NULL) {
        fprintf(stderr, "host permutation allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < TOK; i++) permutation[i] = i;
    if (scattered) {
        uint32_t state = 0x9e3779b9u;
        for (int i = TOK - 1; i > 0; i--) {
            state = state * 1664525u + 1013904223u;
            int j = static_cast<int>(state % static_cast<uint32_t>(i + 1));
            int temporary = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = temporary;
        }
    }
    for (int i = 0; i < EM; i++) {
        int replica = i / TOK;
        int position = (i + replica * 131) % TOK;
        indices[i] = permutation[position];
    }
    free(permutation);
}

template<typename T, int BM, int BK>
static double run_case(int TOK, int K, int topk, bool scattered,
                       int routed_rows = 0) {
    if (K <= 0 || K % BK != 0) {
        fprintf(stderr, "K=%d must be positive and divisible by BK=%d\n", K, BK);
        exit(EXIT_FAILURE);
    }
    int EM = routed_rows > 0 ? routed_rows : TOK * topk;
    if (TOK <= 0 || topk <= 0 || EM <= 0) {
        fprintf(stderr,
                "TOK, topk, and routed-row count must be positive\n");
        exit(EXIT_FAILURE);
    }
    int mt = (EM + BM - 1) / BM;
    int n_kg = K / BK;
    size_t act_bytes = (size_t)TOK * K * sizeof(T);
    size_t idx_bytes = (size_t)EM * sizeof(int);
    T *d_act;
    float *d_out;
    int *d_idx;
    MC_CHECK(mcMalloc(&d_act, act_bytes));
    MC_CHECK(mcMemset(d_act, 1, act_bytes));
    MC_CHECK(mcMalloc(&d_out, mt*sizeof(float)));
    MC_CHECK(mcMemset(d_out, 0, mt*sizeof(float)));
    MC_CHECK(mcMalloc(&d_idx, idx_bytes));
    int *h_idx = (int*)malloc(idx_bytes);
    if (h_idx == NULL) {
        fprintf(stderr, "host allocation failed for %zu bytes\n", idx_bytes);
        exit(EXIT_FAILURE);
    }
    fill_indices(h_idx, TOK, EM, scattered);
    MC_CHECK(mcMemcpy(d_idx, h_idx, idx_bytes, mcMemcpyHostToDevice));
    int block=256;
    gather_kernel<T,BM,BK><<<mt,block>>>(d_act,d_idx,K,EM,n_kg,d_out);
    MC_CHECK(mcDeviceSynchronize());
    int iters=5; mcEvent_t s,e;
    MC_CHECK(mcEventCreate(&s));
    MC_CHECK(mcEventCreate(&e));
    MC_CHECK(mcEventRecord(s));
    for(int it=0;it<iters;it++) {
        gather_kernel<T,BM,BK><<<mt,block>>>(
            d_act,d_idx,K,EM,n_kg,d_out);
    }
    MC_CHECK(mcEventRecord(e));
    MC_CHECK(mcDeviceSynchronize());
    float ms;
    MC_CHECK(mcEventElapsedTime(&ms,s,e));
    ms/=iters;
    double traffic=(double)EM*K*sizeof(T);
    double bw=traffic/(ms/1000.0)/1e9;
    MC_CHECK(mcEventDestroy(s));
    MC_CHECK(mcEventDestroy(e));
    MC_CHECK(mcFree(d_act));
    MC_CHECK(mcFree(d_out));
    MC_CHECK(mcFree(d_idx));
    free(h_idx);
    return bw;
}

template<int BM, int BK>
__global__ void gather_load_u16_kernel(
        const uint16_t* __restrict__ act,
        const int* __restrict__ row_idx,
        int K, int EM, int n_k_groups, uint32_t* __restrict__ out) {
    // Volatile shared writes make every gathered global load observable to the
    // compiler without adding a scalar uint16->float reduction over the whole
    // tile.  One checksum read per K group is enough to prove consumption;
    // the two barriers match the producer/reuse boundary of the tiled path.
    __shared__ volatile uint16_t a_tile[BM][BK];
    int pid_m = blockIdx.x;
    int tid = threadIdx.x;
    int m_start = pid_m * BM;
    if (m_start >= EM) return;
    uint32_t checksum = 0;
    for (int g = 0; g < n_k_groups; g++) {
        int kk = g * BK;
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int column = i % BK;
            int source = row_idx[m_start + row];
            a_tile[row][column] =
                act[(size_t)source * K + (kk + column)];
        }
        __syncthreads();
        if (tid == 0) checksum += a_tile[g % BM][g % BK];
        __syncthreads();
    }
    if (tid == 0) out[pid_m] = checksum;
}

template<int BM, int BK>
static double run_load_case_u16(int TOK, int K, int topk, bool scattered,
                                int routed_rows = 0) {
    if (K <= 0 || K % BK != 0 || TOK <= 0 || topk <= 0) {
        fprintf(stderr,
                "TOK/topk/K must be positive and K divisible by BK=%d\n", BK);
        exit(EXIT_FAILURE);
    }
    int EM = routed_rows > 0 ? routed_rows : TOK * topk;
    if (EM <= 0) {
        fprintf(stderr, "routed-row count must be positive\n");
        exit(EXIT_FAILURE);
    }
    int mt = (EM + BM - 1) / BM;
    int n_k_groups = K / BK;
    size_t activation_bytes = (size_t)TOK * K * sizeof(uint16_t);
    size_t index_bytes = (size_t)EM * sizeof(int);
    uint16_t* d_activation;
    uint32_t* d_output;
    int* d_indices;
    MC_CHECK(mcMalloc(&d_activation, activation_bytes));
    MC_CHECK(mcMemset(d_activation, 1, activation_bytes));
    MC_CHECK(mcMalloc(&d_output, (size_t)mt * sizeof(uint32_t)));
    MC_CHECK(mcMemset(d_output, 0, (size_t)mt * sizeof(uint32_t)));
    MC_CHECK(mcMalloc(&d_indices, index_bytes));
    int* indices = static_cast<int*>(malloc(index_bytes));
    if (indices == NULL) {
        fprintf(stderr, "host allocation failed for %zu bytes\n", index_bytes);
        exit(EXIT_FAILURE);
    }
    fill_indices(indices, TOK, EM, scattered);
    MC_CHECK(mcMemcpy(
        d_indices, indices, index_bytes, mcMemcpyHostToDevice));
    constexpr int BLOCK = 256;
    gather_load_u16_kernel<BM,BK><<<mt,BLOCK>>>(
        d_activation,d_indices,K,EM,n_k_groups,d_output);
    MC_CHECK(mcDeviceSynchronize());
    constexpr int ITERATIONS = 5;
    mcEvent_t start, end;
    MC_CHECK(mcEventCreate(&start));
    MC_CHECK(mcEventCreate(&end));
    MC_CHECK(mcEventRecord(start));
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        gather_load_u16_kernel<BM,BK><<<mt,BLOCK>>>(
            d_activation,d_indices,K,EM,n_k_groups,d_output);
    }
    MC_CHECK(mcEventRecord(end));
    MC_CHECK(mcDeviceSynchronize());
    float milliseconds;
    MC_CHECK(mcEventElapsedTime(&milliseconds,start,end));
    milliseconds /= ITERATIONS;
    if (!(milliseconds > 0.0f)) {
        fprintf(stderr, "non-positive event duration: %.9f ms\n", milliseconds);
        exit(EXIT_FAILURE);
    }
    double logical_bytes = (double)EM*K*sizeof(uint16_t);
    double bandwidth = logical_bytes/(milliseconds/1000.0)/1e9;
    uint32_t* output = static_cast<uint32_t*>(
        malloc((size_t)mt * sizeof(uint32_t)));
    if (output == NULL) {
        fprintf(stderr, "host allocation failed for validation output\n");
        exit(EXIT_FAILURE);
    }
    MC_CHECK(mcMemcpy(
        output, d_output, (size_t)mt * sizeof(uint32_t),
        mcMemcpyDeviceToHost));
    const uint32_t expected = 0x0101u * (uint32_t)n_k_groups;
    for (int block = 0; block < mt; ++block) {
        if (output[block] != expected) {
            fprintf(stderr,
                    "validation failed at block %d: got=%u expected=%u\n",
                    block, output[block], expected);
            exit(EXIT_FAILURE);
        }
    }
    free(output);
    MC_CHECK(mcEventDestroy(start));
    MC_CHECK(mcEventDestroy(end));
    MC_CHECK(mcFree(d_activation));
    MC_CHECK(mcFree(d_output));
    MC_CHECK(mcFree(d_indices));
    free(indices);
    return bandwidth;
}

static void run_legacy_f32() {
    printf("=== MoE A-gather BW sweep (C500) ===\n");
    printf("streaming L2 peak ~3240 GB/s, DRAM peak ~1800 GB/s\n\n");
    printf("%-22s %10s %10s %12s %10s\n","config","act_MB","BM","BW(GB/s)","%stream");
    printf("----------------------------------------------------------\n");
    int K=7168, topk=8;
    // sweep buffer size with BM=64 (L2=8MB): 2MB..2048MB
    int toks[]={256, 1024, 4096, 16384, 32768, 65536, 131072};
    for(int t:toks){
        double bw=run_case<float,64,32>(t,K,topk,true);
        printf("%-22s %10.1f %10d %12.0f %9.1f%%\n","BM64_bufsweep",(double)t*K*4/1e6,64,bw,bw/3240*100);
    }
    printf("\n--- BM sweep (TOK=32768, large buffer) ---\n");
    double b16=run_case<float,16,32>(32768,K,topk,true);
    double b32=run_case<float,32,32>(32768,K,topk,true);
    double b64=run_case<float,64,32>(32768,K,topk,true);
    double b128=run_case<float,128,32>(32768,K,topk,true);
    printf("BM=16:  %.0f GB/s (%.1f%%)\n",b16,b16/3240*100);
    printf("BM=32:  %.0f GB/s (%.1f%%)\n",b32,b32/3240*100);
    printf("BM=64:  %.0f GB/s (%.1f%%)\n",b64,b64/3240*100);
    printf("BM=128: %.0f GB/s (%.1f%%)\n",b128,b128/3240*100);
    printf("linear-fit k = frac/BM:  k16=%.5f k32=%.5f k64=%.5f\n",b16/3240/16,b32/3240/32,b64/3240/64);
}

static void run_bf16_width_transaction_sweep() {
    constexpr int TOK = 8192;
    constexpr int K = 2048;
    constexpr int TOPK = 8;
    printf("=== MoE BF16-width gather transaction sweep (C500) ===\n");
    printf("uint16_t storage isolates 2-byte load transactions; no arithmetic "
           "result is interpreted. activation=%.1f MiB\n",
           (double)TOK * K * sizeof(uint16_t) / (1024.0 * 1024.0));
    printf("%-12s %8s %12s %12s\n",
           "index", "row_B", "BW(GB/s)", "vs_BK64");
    for (int mode = 0; mode < 2; mode++) {
        bool scattered = mode != 0;
        const char* name = scattered ? "scattered" : "contiguous";
        double b32 = run_load_case_u16<64,32>(TOK,K,TOPK,scattered);
        double b64 = run_load_case_u16<64,64>(TOK,K,TOPK,scattered);
        printf("%-12s %8d %12.1f %11.3f\n", name, 64, b32, b32/b64);
        printf("%-12s %8d %12.1f %11.3f\n", name, 128, b64, 1.0);
    }
}

template<int BM, int BK>
static void run_bf16_capacity_signature(const char* signature) {
    constexpr int K = 2048;
    constexpr int TOPK = 8;
    constexpr int ROUTED_ROWS = 262144;
    int tokens[] = {512, 1024, 2048, 4096, 8192, 16384, 32768};
    printf("signature=%s K=%d routed_rows=%d BM=%d BK=%d row_bytes=%zu "
           "L2=8MiB; grid is fixed while unique activation footprint "
           "changes\n",
           signature, K, ROUTED_ROWS, BM, BK, sizeof(uint16_t) * BK);
    printf("%16s %8s %12s %12s\n",
           "signature", "TOK", "act_MiB", "BW(GB/s)");
    for (int TOK : tokens) {
        double bw = run_load_case_u16<BM,BK>(
            TOK,K,TOPK,true,ROUTED_ROWS);
        double act_mib = (double)TOK*K*sizeof(uint16_t)/(1024.0*1024.0);
        printf("%16s %8d %12.1f %12.1f\n",
               signature, TOK, act_mib, bw);
    }
}

static void run_bf16_width_capacity_sweep() {
    printf("=== MoE BF16-width gather capacity sweep (C500) ===\n");
    // These are the two fixed launch families selected by the production
    // MetaX Fused-MoE package.  They are mechanism signatures, not shape bins.
    run_bf16_capacity_signature<16,64>("bm16_bk64");
    run_bf16_capacity_signature<64,32>("bm64_bk32");
}

int main(int argc, char** argv){
    if (argc != 2) {
        fprintf(stderr,
                "usage: %s legacy_f32|transaction_bf16|capacity_bf16|all\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "legacy_f32") == 0) {
        run_legacy_f32();
    } else if (strcmp(argv[1], "transaction_bf16") == 0) {
        run_bf16_width_transaction_sweep();
    } else if (strcmp(argv[1], "capacity_bf16") == 0) {
        run_bf16_width_capacity_sweep();
    } else if (strcmp(argv[1], "all") == 0) {
        run_legacy_f32();
        run_bf16_width_transaction_sweep();
        run_bf16_width_capacity_sweep();
    } else {
        fprintf(stderr, "unknown sweep mode: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    return 0;
}
