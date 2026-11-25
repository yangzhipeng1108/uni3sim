cmake_minimum_required(VERSION 3.16)
project(grouped_matmul_prelude_control)

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
set(GROUPED_MATMUL_PRELUDE_LOGICAL_N "256" CACHE STRING
    "Compile-time logical N for the source prelude probe")
set(GROUPED_MATMUL_PRELUDE_TILE_M "16" CACHE STRING
    "Compile-time M program for the source prelude probe")
if(NOT GROUPED_MATMUL_PRELUDE_LOGICAL_N STREQUAL "64"
        AND NOT GROUPED_MATMUL_PRELUDE_LOGICAL_N STREQUAL "128"
        AND NOT GROUPED_MATMUL_PRELUDE_LOGICAL_N STREQUAL "256")
    message(FATAL_ERROR "GROUPED_MATMUL_PRELUDE_LOGICAL_N must be 64, 128, or 256")
endif()
if(NOT GROUPED_MATMUL_PRELUDE_TILE_M STREQUAL "16"
        AND NOT GROUPED_MATMUL_PRELUDE_TILE_M STREQUAL "32")
    message(FATAL_ERROR "GROUPED_MATMUL_PRELUDE_TILE_M must be 16 or 32")
endif()
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
endif()

include(${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(grouped_matmul_prelude_control_kernel STATIC
    grouped_matmul_prelude_control_kernel.cpp)
# AscendC compiles AIC/AIV code in child projects.  Its dedicated helper is
# required here; normal target definitions only reach the host target.
ascendc_compile_definitions(grouped_matmul_prelude_control_kernel PRIVATE
    GROUPED_MATMUL_PRELUDE_LOGICAL_N=${GROUPED_MATMUL_PRELUDE_LOGICAL_N}
    GROUPED_MATMUL_PRELUDE_TILE_M=${GROUPED_MATMUL_PRELUDE_TILE_M})
target_compile_definitions(grouped_matmul_prelude_control_kernel PRIVATE
    GROUPED_MATMUL_PRELUDE_LOGICAL_N=${GROUPED_MATMUL_PRELUDE_LOGICAL_N}
    GROUPED_MATMUL_PRELUDE_TILE_M=${GROUPED_MATMUL_PRELUDE_TILE_M})
add_executable(grouped_matmul_prelude_control
    grouped_matmul_prelude_control_main.cpp)
target_compile_definitions(grouped_matmul_prelude_control PRIVATE
    GROUPED_MATMUL_PRELUDE_LOGICAL_N=${GROUPED_MATMUL_PRELUDE_LOGICAL_N}
    GROUPED_MATMUL_PRELUDE_TILE_M=${GROUPED_MATMUL_PRELUDE_TILE_M})
target_include_directories(grouped_matmul_prelude_control PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(grouped_matmul_prelude_control PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(grouped_matmul_prelude_control PRIVATE
    grouped_matmul_prelude_control_kernel ascendcl runtime profapi mmpa
    c_sec stdc++ pthread dl)
