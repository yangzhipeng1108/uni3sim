cmake_minimum_required(VERSION 3.16)
project(swiglu_input_pair_completion_bench)

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

ascendc_library(
    swiglu_input_pair_completion_kernel STATIC
    swiglu_input_pair_completion_kernel.cpp)

add_executable(
    swiglu_input_pair_completion_bench
    swiglu_input_pair_completion_main.cpp)
target_include_directories(swiglu_input_pair_completion_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(swiglu_input_pair_completion_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    swiglu_input_pair_completion_bench PRIVATE
    swiglu_input_pair_completion_kernel ascendcl)

ascendc_library(
    swiglu_input_pair_geometry_kernel STATIC
    swiglu_input_pair_geometry_kernel.cpp)

add_executable(
    swiglu_input_pair_geometry_bench
    swiglu_input_pair_geometry_main.cpp)
target_include_directories(swiglu_input_pair_geometry_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(swiglu_input_pair_geometry_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    swiglu_input_pair_geometry_bench PRIVATE
    swiglu_input_pair_geometry_kernel ascendcl)

# A high-repetition, paired 4/8 KiB probe.  Unlike the grid sweep this keeps
# admission and queue construction fixed while measuring both byte counts in
# opposite in-kernel orders.
ascendc_library(
    swiglu_input_pair_repeat_kernel STATIC
    swiglu_input_pair_repeat_kernel.cpp)

add_executable(
    swiglu_input_pair_repeat_bench
    swiglu_input_pair_repeat_main.cpp)
target_include_directories(swiglu_input_pair_repeat_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(swiglu_input_pair_repeat_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    swiglu_input_pair_repeat_bench PRIVATE
    swiglu_input_pair_repeat_kernel ascendcl)

# This keeps a q2 A/B CopyIn primitive resident while separating its first
# and second source-loop visits.  Both visits have an order-exchanged
# PIPE_ALL-only control, so the resulting evidence is a queue mechanism rather
# than a production SwiGLU duration.
ascendc_library(
    swiglu_input_pair_loop_edges_kernel STATIC
    swiglu_input_pair_loop_edges_kernel.cpp)

add_executable(
    swiglu_input_pair_loop_edges_bench
    swiglu_input_pair_loop_edges_main.cpp)
target_include_directories(swiglu_input_pair_loop_edges_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(swiglu_input_pair_loop_edges_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    swiglu_input_pair_loop_edges_bench PRIVATE
    swiglu_input_pair_loop_edges_kernel ascendcl)
