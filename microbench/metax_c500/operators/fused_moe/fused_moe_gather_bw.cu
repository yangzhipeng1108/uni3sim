/*
 * Clean FusedMoE A-operand gather BW microbench (C500).
 *
 * Isolates the INDIRECT GATHER that is MoE's A operand: each CTA gathers BM
 * scattered rows of BK floats from a large (>L2) activation buffer via an
 * index array (sorted_token_ids-style). No weight/output, no compute beyond
 * a sum to defeat dead-load elimination. Mirrors the A-load phase of
 * fused_moe_kernel's mainloop.
 *
 * Compare delivered BW to streaming L2 peak (~3240 GB/s) and to the model's
 * assumption that A streams at L2 peak.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(error));           \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

#define BM 64
#define BK 32

__global__ void gather_kernel(const float* __restrict__ act,
                               const int*   __restrict__ row_idx,
                               int K, int EM, int n_k_groups, float* __restrict__ out)
{
    __shared__ float a_tile[BM][BK];
    int pid_m = blockIdx.x;
    int tid = threadIdx.x;
    int m_start = pid_m * BM;
    if (m_start >= EM) return;
    float acc = 0.f;
    for (int g = 0; g < n_k_groups; g++) {
        int kk = g * BK;
        // gather BM rows via row_idx (INDIRECT), each BK floats
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int r = i / BK, c = i % BK;
            int src = row_idx[m_start + r];
            a_tile[r][c] = act[(size_t)src * K + (kk + c)];
        }
        __syncthreads();
        for (int i = 0; i < BK; i++) acc += a_tile[tid % BM][i];
        __syncthreads();
    }
    if (tid == 0) out[pid_m] = acc;
}

int main() {
    printf("=== MoE A-gather BW (C500)  BM=%d BK=%d ===\n\n", BM, BK);
    int K = 7168;
    int TOK = 32768;
    int topk = 8;
    int EM = TOK * topk;      // 262144 sorted tokens
    int mt = (EM + BM - 1) / BM;  // 4096 CTAs
    int n_k_groups = K / BK;     // 224 k-steps

    size_t act_bytes = (size_t)TOK * K * sizeof(float);  // 940 MB >> 8MB L2
    size_t idx_bytes = (size_t)EM * sizeof(int);
    printf("activation buffer: %.0f MB (>> 8MB L2)\n", (double)act_bytes/1e6);
    printf("CTAs (mt) = %d, k_groups = %d, block = 256\n\n", mt, n_k_groups);

    float *d_act, *d_out; int *d_idx;
    CUDA_CHECK(cudaMalloc(&d_act, act_bytes));
    CUDA_CHECK(cudaMemset(d_act, 1, act_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, mt*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, mt*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx, idx_bytes));
    int *h_idx = (int*)malloc(idx_bytes);
    if (h_idx == NULL) {
        fprintf(stderr, "host allocation failed for %zu bytes\n", idx_bytes);
        return EXIT_FAILURE;
    }
    // sorted-token gather: indices are a permutation of [0,TOK), tiled to EM.
    // Make it scattered (random permutation) to mimic real routing.
    for (int i = 0; i < EM; i++) h_idx[i] = i % TOK;
    for (int i = EM - 1; i > 0; i--) { int j = h_idx[i] % (i+1); int t=h_idx[i]; h_idx[i]=h_idx[j]; h_idx[j]=t; }
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx, idx_bytes, cudaMemcpyHostToDevice));

    int block = 256;
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    gather_kernel<<<mt, block>>>(d_act, d_idx, K, EM, n_k_groups, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int iters = 5;
    CUDA_CHECK(cudaEventRecord(s));
    for (int it = 0; it < iters; it++)
        gather_kernel<<<mt, block>>>(d_act, d_idx, K, EM, n_k_groups, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    ms /= iters;

    // A traffic per iter = EM * K * 4 bytes (each sorted token reads K floats, gather)
    double traffic = (double)EM * K * 4;
    double bw = traffic / (ms/1000.0) / 1e9;
    printf("A-gather: %.2f ms/iter, A traffic = %.1f GB, BW = %.0f GB/s\n", ms, traffic/1e9, bw);
    printf("streaming L2 peak ~3240 GB/s => gather is %.1f%% of streaming peak\n", bw/3240*100);
    printf("(if <<100%%, MoE A-operand is gather-bound — the unmodeled penalty)\n");
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaFree(d_act));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_idx));
    free(h_idx);
    return 0;
}
