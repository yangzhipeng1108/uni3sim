cmake_minimum_required(VERSION 3.16)
project(mte2_tile)
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
ascendc_library(mte2_tile_kernel STATIC mte2_tile_kernel.cpp)
add_executable(mte2_tile mte2_main.cpp)
target_include_directories(mte2_tile PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(mte2_tile PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(mte2_tile PRIVATE mte2_tile_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(aic_mte2_nd2nz_kernel STATIC
    nd2nz/aic_mte2_nd2nz_kernel.cpp)
add_executable(aic_mte2_nd2nz nd2nz/aic_mte2_nd2nz_main.cpp)
target_include_directories(aic_mte2_nd2nz PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_nd2nz PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_nd2nz PRIVATE aic_mte2_nd2nz_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(aic_mte2_grouped_pair_kernel STATIC
    grouped_pair/aic_mte2_grouped_pair_kernel.cpp)
add_executable(aic_mte2_grouped_pair
    grouped_pair/aic_mte2_grouped_pair_main.cpp)
target_include_directories(aic_mte2_grouped_pair PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_grouped_pair PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_grouped_pair PRIVATE
    aic_mte2_grouped_pair_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(aic_mte2_l1_program_kernel STATIC
    l1_program/aic_mte2_l1_program_kernel.cpp)
add_executable(aic_mte2_l1_program
    l1_program/aic_mte2_l1_program_main.cpp)
target_include_directories(aic_mte2_l1_program PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_l1_program PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_l1_program PRIVATE
    aic_mte2_l1_program_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(aic_mte2_legacy_l1_pair_kernel STATIC
    l1_program/aic_mte2_legacy_l1_pair_kernel.cpp)
add_executable(aic_mte2_legacy_l1_pair
    l1_program/aic_mte2_legacy_l1_pair_main.cpp)
target_include_directories(aic_mte2_legacy_l1_pair PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_legacy_l1_pair PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_legacy_l1_pair PRIVATE
    aic_mte2_legacy_l1_pair_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(aic_mte2_nd2nz_stride_kernel STATIC
    nd2nz/aic_mte2_nd2nz_stride_kernel.cpp)
add_executable(aic_mte2_nd2nz_stride
    nd2nz/aic_mte2_nd2nz_stride_main.cpp)
target_include_directories(aic_mte2_nd2nz_stride PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_nd2nz_stride PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_nd2nz_stride PRIVATE
    aic_mte2_nd2nz_stride_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)

ascendc_library(aic_mte2_small_m_program_kernel STATIC
    l1_program/aic_mte2_small_m_program_kernel.cpp)
add_executable(aic_mte2_small_m_program
    l1_program/aic_mte2_small_m_program_main.cpp)
target_include_directories(aic_mte2_small_m_program PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aic_mte2_small_m_program PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(aic_mte2_small_m_program PRIVATE
    aic_mte2_small_m_program_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
