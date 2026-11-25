cmake_minimum_required(VERSION 3.16)
project(fia_program_span_rt)
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
ascendc_library(fia_program_span_rt_kernel STATIC fia_program_span_rt_kernel.cpp)
add_executable(fia_program_span_rt fia_program_span_rt_main.cpp)
target_include_directories(fia_program_span_rt PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(fia_program_span_rt PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(fia_program_span_rt PRIVATE fia_program_span_rt_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
