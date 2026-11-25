cmake_minimum_required(VERSION 3.16)
project(aic_mte2_gemm_source_order)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend910B2")
endif()
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
endif()
include(${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(aic_mte2_gemm_source_order_kernel STATIC
    aic_mte2_gemm_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_ingress_completion_kernel STATIC
    aic_mte2_gemm_ingress_completion_kernel.cpp)
ascendc_library(aic_mte2_gemm_pipeline_completion_kernel STATIC
    aic_mte2_gemm_pipeline_completion_kernel.cpp)
add_executable(aic_mte2_gemm_source_order
    aic_mte2_gemm_source_order_main.cpp)
target_compile_definitions(aic_mte2_gemm_source_order PRIVATE
    GEMM_ENABLE_CUBE_COMPLETION=1 GEMM_ENABLE_DIRTY_PREAMBLE=1)
target_include_directories(aic_mte2_gemm_source_order PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_source_order PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_source_order PRIVATE
    aic_mte2_gemm_source_order_kernel
    aic_mte2_gemm_ingress_completion_kernel
    aic_mte2_gemm_pipeline_completion_kernel
    aiv_l2_dirty_preamble_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_k1536_n5120_source_order_kernel STATIC
    aic_mte2_gemm_k1536_n5120_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_k1536_n5120_ingress_completion_kernel STATIC
    aic_mte2_gemm_k1536_n5120_ingress_completion_kernel.cpp)
add_executable(aic_mte2_gemm_k1536_n5120_source_order
    aic_mte2_gemm_k1536_n5120_source_order_main.cpp)
target_include_directories(aic_mte2_gemm_k1536_n5120_source_order PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_k1536_n5120_source_order PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_k1536_n5120_source_order PRIVATE
    aic_mte2_gemm_k1536_n5120_source_order_kernel
    aic_mte2_gemm_k1536_n5120_ingress_completion_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_k14336_source_order_kernel STATIC
    aic_mte2_gemm_k14336_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_k14336_ingress_completion_kernel STATIC
    aic_mte2_gemm_k14336_ingress_completion_kernel.cpp)
add_executable(aic_mte2_gemm_k14336_source_order
    aic_mte2_gemm_k14336_source_order_main.cpp)
target_include_directories(aic_mte2_gemm_k14336_source_order PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_k14336_source_order PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_k14336_source_order PRIVATE
    aic_mte2_gemm_k14336_source_order_kernel
    aic_mte2_gemm_k14336_ingress_completion_kernel
    aiv_l2_dirty_preamble_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_k13824_m16_n112_source_order_kernel STATIC
    aic_mte2_gemm_k13824_m16_n112_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_k13824_m16_n112_ingress_completion_kernel STATIC
    aic_mte2_gemm_k13824_m16_n112_ingress_completion_kernel.cpp)
add_executable(aic_mte2_gemm_k13824_m16_n112_source_order
    aic_mte2_gemm_k13824_m16_n112_source_order_main.cpp)
target_include_directories(aic_mte2_gemm_k13824_m16_n112_source_order PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_k13824_m16_n112_source_order PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_k13824_m16_n112_source_order PRIVATE
    aic_mte2_gemm_k13824_m16_n112_source_order_kernel
    aic_mte2_gemm_k13824_m16_n112_ingress_completion_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_k1536_m16_n224_source_order_kernel STATIC
    aic_mte2_gemm_k1536_m16_n224_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_k1536_m16_n224_ingress_completion_kernel STATIC
    aic_mte2_gemm_k1536_m16_n224_ingress_completion_kernel.cpp)
ascendc_library(aic_mte2_gemm_k1536_m16_n224_pipeline_completion_kernel STATIC
    aic_mte2_gemm_k1536_m16_n224_pipeline_completion_kernel.cpp)
add_executable(aic_mte2_gemm_k1536_m16_n224_source_order
    aic_mte2_gemm_k1536_m16_n224_source_order_main.cpp)
target_compile_definitions(
    aic_mte2_gemm_k1536_m16_n224_source_order PRIVATE
    GEMM_ENABLE_CUBE_COMPLETION=1)
target_include_directories(
    aic_mte2_gemm_k1536_m16_n224_source_order PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(
    aic_mte2_gemm_k1536_m16_n224_source_order PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(
    aic_mte2_gemm_k1536_m16_n224_source_order PRIVATE
    aic_mte2_gemm_k1536_m16_n224_source_order_kernel
    aic_mte2_gemm_k1536_m16_n224_ingress_completion_kernel
    aic_mte2_gemm_k1536_m16_n224_pipeline_completion_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_k14336_m128_n128_source_order_kernel STATIC
    aic_mte2_gemm_k14336_m128_n128_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_k14336_m128_n128_ingress_completion_kernel STATIC
    aic_mte2_gemm_k14336_m128_n128_ingress_completion_kernel.cpp)
ascendc_library(aiv_l2_dirty_preamble_kernel STATIC
    aiv_l2_dirty_preamble_kernel.cpp)
add_executable(aic_mte2_gemm_k14336_m128_n128_source_order
    aic_mte2_gemm_k14336_m128_n128_source_order_main.cpp)
target_include_directories(aic_mte2_gemm_k14336_m128_n128_source_order PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_k14336_m128_n128_source_order PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_k14336_m128_n128_source_order PRIVATE
    aic_mte2_gemm_k14336_m128_n128_source_order_kernel
    aic_mte2_gemm_k14336_m128_n128_ingress_completion_kernel
    aiv_l2_dirty_preamble_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_m256_n112_source_order_kernel STATIC
    aic_mte2_gemm_m256_n112_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_m256_n112_ingress_completion_kernel STATIC
    aic_mte2_gemm_m256_n112_ingress_completion_kernel.cpp)
add_executable(aic_mte2_gemm_m256_n112_source_order
    aic_mte2_gemm_m256_n112_source_order_main.cpp)
target_include_directories(aic_mte2_gemm_m256_n112_source_order PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_m256_n112_source_order PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_m256_n112_source_order PRIVATE
    aic_mte2_gemm_m256_n112_source_order_kernel
    aic_mte2_gemm_m256_n112_ingress_completion_kernel
    aiv_l2_dirty_preamble_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_m2220_source_order_kernel STATIC
    aic_mte2_gemm_m2220_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_m2220_ingress_completion_kernel STATIC
    aic_mte2_gemm_m2220_ingress_completion_kernel.cpp)
add_executable(aic_mte2_gemm_m2220_source_order
    aic_mte2_gemm_m2220_source_order_main.cpp)
target_include_directories(aic_mte2_gemm_m2220_source_order PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_m2220_source_order PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_m2220_source_order PRIVATE
    aic_mte2_gemm_m2220_source_order_kernel
    aic_mte2_gemm_m2220_ingress_completion_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_m64_source_order_kernel STATIC
    aic_mte2_gemm_m64_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_m64_ingress_completion_kernel STATIC
    aic_mte2_gemm_m64_ingress_completion_kernel.cpp)
add_executable(aic_mte2_gemm_m64_source_order
    aic_mte2_gemm_m64_source_order_main.cpp)
target_include_directories(aic_mte2_gemm_m64_source_order PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_m64_source_order PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_m64_source_order PRIVATE
    aic_mte2_gemm_m64_source_order_kernel
    aic_mte2_gemm_m64_ingress_completion_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
add_executable(aic_mte2_gemm_region
    aic_mte2_gemm_region_main.cpp)
ascendc_library(aic_mte2_gemm_region_source_order_kernel STATIC
    aic_mte2_gemm_region_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_region_ingress_completion_kernel STATIC
    aic_mte2_gemm_region_ingress_completion_kernel.cpp)
ascendc_library(aic_mte2_gemm_region_pipeline_completion_kernel STATIC
    aic_mte2_gemm_region_pipeline_completion_kernel.cpp)
target_include_directories(aic_mte2_gemm_region PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_region PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_region PRIVATE
    aic_mte2_gemm_region_source_order_kernel
    aic_mte2_gemm_region_ingress_completion_kernel
    aic_mte2_gemm_region_pipeline_completion_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

# This is a separate source program, rather than a parameterized rebuild of
# aic_mte2_gemm_region: the checked-in A8/B4 reference keeps its own exact
# source snapshot.  The A4/B4 probe covers the V3 K=13824 program's 24-AIC
# region schedule and two-slot A1/B1 credit lifetime.
add_executable(aic_mte2_gemm_region_a4b4
    aic_mte2_gemm_region_a4b4_main.cpp)
ascendc_library(aic_mte2_gemm_region_a4b4_source_order_kernel STATIC
    aic_mte2_gemm_region_a4b4_source_order_kernel.cpp)
ascendc_library(aic_mte2_gemm_region_a4b4_ingress_completion_kernel STATIC
    aic_mte2_gemm_region_a4b4_ingress_completion_kernel.cpp)
target_include_directories(aic_mte2_gemm_region_a4b4 PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_region_a4b4 PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_region_a4b4 PRIVATE
    aic_mte2_gemm_region_a4b4_source_order_kernel
    aic_mte2_gemm_region_a4b4_ingress_completion_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_full_a_b_stream_kernel STATIC
    aic_mte2_gemm_full_a_b_stream_kernel.cpp)
add_executable(aic_mte2_gemm_full_a_b_stream
    aic_mte2_gemm_full_a_b_stream_main.cpp)
target_include_directories(aic_mte2_gemm_full_a_b_stream PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_full_a_b_stream PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_full_a_b_stream PRIVATE
    aic_mte2_gemm_full_a_b_stream_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_a1_b20_stream_kernel STATIC
    aic_mte2_gemm_a1_b20_stream_kernel.cpp)
add_executable(aic_mte2_gemm_a1_b20_stream
    aic_mte2_gemm_a1_b20_stream_main.cpp)
target_include_directories(aic_mte2_gemm_a1_b20_stream PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_a1_b20_stream PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_a1_b20_stream PRIVATE
    aic_mte2_gemm_a1_b20_stream_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_full_a_b_stream_n8192_kernel STATIC
    aic_mte2_gemm_full_a_b_stream_n8192_kernel.cpp)
add_executable(aic_mte2_gemm_full_a_b_stream_n8192
    aic_mte2_gemm_full_a_b_stream_n8192_main.cpp)
target_include_directories(aic_mte2_gemm_full_a_b_stream_n8192 PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_full_a_b_stream_n8192 PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_full_a_b_stream_n8192 PRIVATE
    aic_mte2_gemm_full_a_b_stream_n8192_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
ascendc_library(aic_mte2_gemm_v2_warm_n8192_kernel STATIC
    aic_mte2_gemm_v2_warm_n8192_kernel.cpp)
add_executable(aic_mte2_gemm_v2_warm_n8192
    aic_mte2_gemm_v2_warm_n8192_main.cpp)
target_include_directories(aic_mte2_gemm_v2_warm_n8192 PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_v2_warm_n8192 PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_v2_warm_n8192 PRIVATE
    aic_mte2_gemm_v2_warm_n8192_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

# This source program isolates the full-A / five-N-tile ingress topology of
# the MatMulV2 16x96x128 K=4096 launch.  It deliberately excludes MTE1,
# Cube, FixPipe and output work, so its timing is an MTE2 queue mechanism
# rather than an operator-duration proxy.
ascendc_library(aic_mte2_matmul_v2_full_a_k4096_n96_kernel STATIC
    aic_mte2_matmul_v2_full_a_k4096_n96_kernel.cpp)
add_executable(aic_mte2_matmul_v2_full_a_k4096_n96
    aic_mte2_matmul_v2_full_a_k4096_n96_main.cpp)
target_include_directories(aic_mte2_matmul_v2_full_a_k4096_n96 PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_matmul_v2_full_a_k4096_n96 PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_matmul_v2_full_a_k4096_n96 PRIVATE
    aic_mte2_matmul_v2_full_a_k4096_n96_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(aic_mte2_gemm_topology_stream_kernel STATIC
    aic_mte2_gemm_topology_stream_kernel.cpp)
add_executable(aic_mte2_gemm_topology_stream
    aic_mte2_gemm_topology_stream_main.cpp)
target_include_directories(aic_mte2_gemm_topology_stream PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_topology_stream PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_topology_stream PRIVATE
    aic_mte2_gemm_topology_stream_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(aic_mte2_gemm_credit_stream_kernel STATIC
    aic_mte2_gemm_credit_stream_kernel.cpp)
add_executable(aic_mte2_gemm_credit_stream
    aic_mte2_gemm_credit_stream_main.cpp)
target_include_directories(aic_mte2_gemm_credit_stream PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_gemm_credit_stream PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_gemm_credit_stream PRIVATE
    aic_mte2_gemm_credit_stream_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
