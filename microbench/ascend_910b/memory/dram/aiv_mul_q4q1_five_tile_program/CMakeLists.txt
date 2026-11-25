cmake_minimum_required(VERSION 3.16)
project(aiv_mul_q4q1_five_tile_program)

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
    aiv_mul_q4q1_five_tile_program_kernel STATIC
    aiv_mul_q4q1_five_tile_program_kernel.cpp)

add_executable(
    aiv_mul_q4q1_five_tile_program
    aiv_mul_q4q1_five_tile_program_main.cpp)
target_include_directories(aiv_mul_q4q1_five_tile_program PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_mul_q4q1_five_tile_program PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_mul_q4q1_five_tile_program PRIVATE
    aiv_mul_q4q1_five_tile_program_kernel ascendcl)
