cmake_minimum_required(VERSION 3.16)
project(dram_latency)

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

# AscendC kernel cmake package (provides ascendc_library → aclrtlaunch_* host stubs)
include(${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

# Compile the AIV kernel into a static lib; generates aclrtlaunch_dram_latency stub.
ascendc_library(dram_latency_kernel STATIC dram_latency_kernel.cpp)

add_executable(dram_latency main.cpp)
target_include_directories(dram_latency PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(dram_latency PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(dram_latency PRIVATE dram_latency_kernel ascendcl)

# The paged FIA block-table lookup executes on AIC. Keep its scalar-memory
# service separate from the otherwise identical AIV mechanism above.
ascendc_library(aic_dram_latency_kernel STATIC aic_dram_latency_kernel.cpp)

add_executable(aic_dram_latency main.cpp)
target_include_directories(aic_dram_latency PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_dram_latency PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aic_dram_latency PRIVATE
    aic_dram_latency_kernel ascendcl)

ascendc_library(first_get_value_kernel STATIC first_get_value_kernel.cpp)

add_executable(first_get_value first_get_value_main.cpp)
target_include_directories(first_get_value PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(first_get_value PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(first_get_value PRIVATE first_get_value_kernel ascendcl)

ascendc_library(first_mte_completion_kernel STATIC
    first_mte_completion_kernel.cpp)

add_executable(first_mte_completion first_mte_completion_main.cpp)
target_include_directories(first_mte_completion PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(first_mte_completion PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(first_mte_completion PRIVATE
    first_mte_completion_kernel ascendcl)
