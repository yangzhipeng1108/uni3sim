// Compile the shared source-order probe for the normalized 128x112x128
// MatMulV2 program.  MTE2 groups two base-K tiles into each 256-column A/B
// descriptor; the runtime b_rows=112 argument supplies the B extent.
#define GEMM_SOURCE_A_ROWS 128
#define GEMM_SOURCE_A_COLUMNS 256
#define GEMM_SOURCE_B_COLUMNS 256
#include "aic_mte2_gemm_source_order_kernel.cpp"
