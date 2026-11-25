cmake_minimum_required(VERSION 3.16)
project(norm_reduce_bench)

set(CMAKE_CXX_STANDARD 17)
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

ascendc_library(norm_reduce_kernel STATIC norm_reduce_kernel.cpp)
ascendc_library(rmsnorm_full_kernel STATIC rmsnorm_full_kernel.cpp)
ascendc_library(
    whole_reduce_latency_kernel STATIC whole_reduce_latency_kernel.cpp)
ascendc_library(
    reduce_sum_latency_kernel STATIC reduce_sum_latency_kernel.cpp)
ascendc_library(
    add_rmsnorm_input_pair_kernel STATIC
    add_rmsnorm_input_pair_kernel.cpp)
ascendc_library(
    rmsnorm_key0_gamma_input_sequence_kernel STATIC
    rmsnorm_key0_gamma_input_sequence_kernel.cpp)

add_executable(norm_reduce_bench norm_reduce_main.cpp)
target_include_directories(norm_reduce_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(norm_reduce_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(norm_reduce_bench PRIVATE norm_reduce_kernel ascendcl)

add_executable(rmsnorm_full_bench rmsnorm_full_main.cpp)
target_include_directories(rmsnorm_full_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(rmsnorm_full_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(rmsnorm_full_bench PRIVATE rmsnorm_full_kernel ascendcl)

add_executable(
    whole_reduce_latency_bench whole_reduce_latency_main.cpp)
target_include_directories(whole_reduce_latency_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(whole_reduce_latency_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    whole_reduce_latency_bench PRIVATE whole_reduce_latency_kernel ascendcl)

add_executable(reduce_sum_latency_bench reduce_sum_latency_main.cpp)
target_include_directories(reduce_sum_latency_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(reduce_sum_latency_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    reduce_sum_latency_bench PRIVATE reduce_sum_latency_kernel ascendcl)

add_executable(
    add_rmsnorm_input_pair_bench add_rmsnorm_input_pair_main.cpp)
target_include_directories(add_rmsnorm_input_pair_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(add_rmsnorm_input_pair_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    add_rmsnorm_input_pair_bench PRIVATE
    add_rmsnorm_input_pair_kernel ascendcl)

add_executable(
    rmsnorm_key0_gamma_input_sequence_bench
    rmsnorm_key0_gamma_input_sequence_main.cpp)
target_include_directories(rmsnorm_key0_gamma_input_sequence_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(rmsnorm_key0_gamma_input_sequence_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    rmsnorm_key0_gamma_input_sequence_bench PRIVATE
    rmsnorm_key0_gamma_input_sequence_kernel ascendcl)
