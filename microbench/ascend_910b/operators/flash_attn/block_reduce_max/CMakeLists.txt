cmake_minimum_required(VERSION 3.16)
project(block_reduce_max_service)

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

ascendc_library(
    block_reduce_max_service_kernel STATIC block_reduce_max_kernel.cpp)
target_include_directories(
    block_reduce_max_service_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(block_reduce_max_service block_reduce_max_main.cpp)
target_include_directories(block_reduce_max_service PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(block_reduce_max_service PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    block_reduce_max_service PRIVATE
    block_reduce_max_service_kernel ascendcl)
