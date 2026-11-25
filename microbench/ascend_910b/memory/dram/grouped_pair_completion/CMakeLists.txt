cmake_minimum_required(VERSION 3.16)
project(grouped_pair_completion)
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
if(NOT DEFINED GROUPED_PAIR_BASE_M)
    set(GROUPED_PAIR_BASE_M 16)
endif()
if(NOT DEFINED GROUPED_PAIR_ACTIVE_M)
    set(GROUPED_PAIR_ACTIVE_M ${GROUPED_PAIR_BASE_M})
endif()
if(NOT DEFINED GROUPED_PAIR_B_ROW_STRIDE)
    set(GROUPED_PAIR_B_ROW_STRIDE 256)
endif()
include(${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

ascendc_library(grouped_pair_completion_kernel STATIC
    grouped_pair_completion_kernel.cpp)
target_compile_definitions(grouped_pair_completion_kernel PRIVATE
    GROUPED_PAIR_BASE_M=${GROUPED_PAIR_BASE_M}
    GROUPED_PAIR_ACTIVE_M=${GROUPED_PAIR_ACTIVE_M}
    GROUPED_PAIR_B_ROW_STRIDE=${GROUPED_PAIR_B_ROW_STRIDE})
add_executable(grouped_pair_completion
    grouped_pair_completion_main.cpp)
target_compile_definitions(grouped_pair_completion PRIVATE
    GROUPED_PAIR_BASE_M=${GROUPED_PAIR_BASE_M}
    GROUPED_PAIR_ACTIVE_M=${GROUPED_PAIR_ACTIVE_M}
    GROUPED_PAIR_B_ROW_STRIDE=${GROUPED_PAIR_B_ROW_STRIDE})
target_include_directories(grouped_pair_completion PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(grouped_pair_completion PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(grouped_pair_completion PRIVATE
    grouped_pair_completion_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
