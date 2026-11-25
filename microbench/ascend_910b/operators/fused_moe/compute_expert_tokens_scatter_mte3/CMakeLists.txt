cmake_minimum_required(VERSION 3.16)
project(compute_expert_tokens_scatter_mte3)
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
ascendc_library(compute_expert_tokens_scatter_mte3_kernel STATIC
    compute_expert_tokens_scatter_mte3_kernel.cpp)
ascendc_library(compute_expert_tokens_sync_phase_kernel STATIC
    compute_expert_tokens_sync_phase_kernel.cpp)
ascendc_library(compute_expert_tokens_ss_init_sync_phase_kernel STATIC
    compute_expert_tokens_ss_init_sync_phase_kernel.cpp)
add_executable(compute_expert_tokens_scatter_mte3
    compute_expert_tokens_scatter_mte3_main.cpp)
target_include_directories(compute_expert_tokens_scatter_mte3 PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(compute_expert_tokens_scatter_mte3 PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(compute_expert_tokens_scatter_mte3 PRIVATE
    compute_expert_tokens_scatter_mte3_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
add_executable(compute_expert_tokens_sync_phase
    compute_expert_tokens_sync_phase_main.cpp)
target_include_directories(compute_expert_tokens_sync_phase PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(compute_expert_tokens_sync_phase PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(compute_expert_tokens_sync_phase PRIVATE
    compute_expert_tokens_sync_phase_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
add_executable(compute_expert_tokens_ss_init_sync_phase
    compute_expert_tokens_ss_init_sync_phase_main.cpp)
target_include_directories(compute_expert_tokens_ss_init_sync_phase PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(compute_expert_tokens_ss_init_sync_phase PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(compute_expert_tokens_ss_init_sync_phase PRIVATE
    compute_expert_tokens_ss_init_sync_phase_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
