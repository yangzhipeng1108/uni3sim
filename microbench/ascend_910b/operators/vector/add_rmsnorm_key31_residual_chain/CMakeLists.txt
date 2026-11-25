cmake_minimum_required(VERSION 3.16)
project(add_rmsnorm_key31_residual_chain)

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
    add_rmsnorm_key31_residual_chain_kernel STATIC
    add_rmsnorm_key31_residual_chain_kernel.cpp)

add_executable(
    add_rmsnorm_key31_residual_chain
    add_rmsnorm_key31_residual_chain_main.cpp)
target_include_directories(add_rmsnorm_key31_residual_chain PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(add_rmsnorm_key31_residual_chain PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(add_rmsnorm_key31_residual_chain PRIVATE
    add_rmsnorm_key31_residual_chain_kernel ascendcl)
