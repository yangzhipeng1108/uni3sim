// Build the compact-region ingress program with its dependent Cube consumer.
// The probe intentionally omits FixPipe and output storage, so it isolates
// the finite source/compute pipeline instead of reproducing a complete GEMM.
#define GEMM_REGION_ENABLE_CUBE_CONSUMER 1
#define GEMM_REGION_INGRESS_KERNEL_NAME \
    aic_mte2_gemm_region_pipeline_completion
#include "aic_mte2_gemm_region_ingress_completion_kernel.cpp"
