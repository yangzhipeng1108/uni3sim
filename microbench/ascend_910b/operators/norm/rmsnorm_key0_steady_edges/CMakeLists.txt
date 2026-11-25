cmake_minimum_required(VERSION 3.16)
project(rmsnorm_key0_steady_edges_bench)

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
    rmsnorm_key0_steady_edges_kernel STATIC
    rmsnorm_key0_steady_edges_kernel.cpp)

add_executable(
    rmsnorm_key0_steady_edges_bench
    rmsnorm_key0_steady_edges_main.cpp)
target_include_directories(rmsnorm_key0_steady_edges_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(rmsnorm_key0_steady_edges_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    rmsnorm_key0_steady_edges_bench PRIVATE
    rmsnorm_key0_steady_edges_kernel ascendcl)
