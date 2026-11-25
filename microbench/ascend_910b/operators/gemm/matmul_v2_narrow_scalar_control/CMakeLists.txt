cmake_minimum_required(VERSION 3.16)
project(matmul_v2_narrow_scalar_control)

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
ascendc_library(matmul_v2_narrow_scalar_control_kernel STATIC
    matmul_v2_narrow_scalar_control_kernel.cpp)
add_executable(matmul_v2_narrow_scalar_control
    matmul_v2_narrow_scalar_control_main.cpp)
target_include_directories(matmul_v2_narrow_scalar_control PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(matmul_v2_narrow_scalar_control PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(matmul_v2_narrow_scalar_control PRIVATE
    matmul_v2_narrow_scalar_control_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

# Keep the full-A 16x176x64 program separate from the 16x224x64 narrow
# control.  Its A1 lifetime spans the entire K extent, so sharing a build
# would make the descriptor topology ambiguous.
ascendc_library(matmul_v2_full_a_scalar_control_kernel STATIC
    matmul_v2_full_a_scalar_control_kernel.cpp)
add_executable(matmul_v2_full_a_scalar_control
    matmul_v2_full_a_scalar_control_main.cpp)
target_include_directories(matmul_v2_full_a_scalar_control PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(matmul_v2_full_a_scalar_control PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(matmul_v2_full_a_scalar_control PRIVATE
    matmul_v2_full_a_scalar_control_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

# The 23-AIC 16x256x64 control retains the distinct five-local-N-tile loop
# and is intentionally not folded into the one-N 16x176x64 control.
ascendc_library(matmul_v2_full_a_five_n_scalar_control_kernel STATIC
    matmul_v2_full_a_five_n_scalar_control_kernel.cpp)
add_executable(matmul_v2_full_a_five_n_scalar_control
    matmul_v2_full_a_five_n_scalar_control_main.cpp)
target_include_directories(matmul_v2_full_a_five_n_scalar_control PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(matmul_v2_full_a_five_n_scalar_control PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(matmul_v2_full_a_five_n_scalar_control PRIVATE
    matmul_v2_full_a_five_n_scalar_control_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
