// Build the shared ingress program with its dependent Cube consumer.  The
// probe intentionally omits FixPipe and output storage, so it measures a
// pipeline mechanism rather than reproducing the complete GEMM operator.
#define GEMM_ENABLE_CUBE_CONSUMER 1
#define GEMM_INGRESS_KERNEL_NAME aic_mte2_gemm_pipeline_completion
#include "aic_mte2_gemm_ingress_completion_kernel.cpp"
