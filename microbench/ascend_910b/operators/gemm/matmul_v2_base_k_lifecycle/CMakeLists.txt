cmake_minimum_required(VERSION 3.16)
project(matmul_v2_base_k_lifecycle)

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

ascendc_library(matmul_v2_base_k_lifecycle_kernel STATIC
    matmul_v2_base_k_lifecycle_kernel.cpp)
add_executable(matmul_v2_base_k_lifecycle
    matmul_v2_base_k_lifecycle_main.cpp)
target_include_directories(matmul_v2_base_k_lifecycle PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(matmul_v2_base_k_lifecycle PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(matmul_v2_base_k_lifecycle PRIVATE
    matmul_v2_base_k_lifecycle_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(matmul_v2_output_tile_lifecycle_kernel STATIC
    matmul_v2_output_tile_lifecycle_kernel.cpp)
add_executable(matmul_v2_output_tile_lifecycle
    matmul_v2_output_tile_lifecycle_main.cpp)
target_include_directories(matmul_v2_output_tile_lifecycle PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(matmul_v2_output_tile_lifecycle PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(matmul_v2_output_tile_lifecycle PRIVATE
    matmul_v2_output_tile_lifecycle_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(matmul_v2_narrow_stage_ladder_kernel STATIC
    matmul_v2_narrow_stage_ladder_kernel.cpp)
add_executable(matmul_v2_narrow_stage_ladder
    matmul_v2_narrow_stage_ladder_main.cpp)
target_include_directories(matmul_v2_narrow_stage_ladder PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(matmul_v2_narrow_stage_ladder PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(matmul_v2_narrow_stage_ladder PRIVATE
    matmul_v2_narrow_stage_ladder_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
