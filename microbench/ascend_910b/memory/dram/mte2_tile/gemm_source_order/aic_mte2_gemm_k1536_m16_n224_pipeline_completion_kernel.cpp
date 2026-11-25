#define GEMM_SOURCE_A_ROWS 16
#define GEMM_SOURCE_A_COLUMNS 1536
#define GEMM_SOURCE_B_COLUMNS 256
#define GEMM_SOURCE_MAXIMUM_B_ROWS 224
#define GEMM_ENABLE_CUBE_CONSUMER 1
#define GEMM_INGRESS_KERNEL_NAME aic_mte2_gemm_pipeline_completion
#include "aic_mte2_gemm_ingress_completion_kernel.cpp"
