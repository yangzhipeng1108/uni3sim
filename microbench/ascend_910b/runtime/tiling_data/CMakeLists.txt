cmake_minimum_required(VERSION 3.16)
project(tiling_data_materialization)

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

ascendc_library(tiling_data_materialization_kernel STATIC
    tiling_data_materialization_kernel.cpp)

add_executable(tiling_data_materialization_bench
    tiling_data_materialization_main.cpp)
target_include_directories(tiling_data_materialization_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(tiling_data_materialization_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(tiling_data_materialization_bench PRIVATE
    tiling_data_materialization_kernel ascendcl)

ascendc_library(fia_tiling_data_materialization_kernel STATIC
    fia_tiling_data_materialization_kernel.cpp)

add_executable(fia_tiling_data_materialization_bench
    fia_tiling_data_materialization_main.cpp)
target_include_directories(fia_tiling_data_materialization_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(fia_tiling_data_materialization_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(fia_tiling_data_materialization_bench PRIVATE
    fia_tiling_data_materialization_kernel ascendcl)
