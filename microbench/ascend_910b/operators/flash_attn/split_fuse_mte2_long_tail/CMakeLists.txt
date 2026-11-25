cmake_minimum_required(VERSION 3.16)
project(split_fuse_mte2_interleaved)

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
ascendc_library(split_fuse_mte2_interleaved_kernel STATIC kernel.cpp)
add_executable(split_fuse_mte2_interleaved main.cpp)
target_include_directories(split_fuse_mte2_interleaved PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(split_fuse_mte2_interleaved PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(split_fuse_mte2_interleaved PRIVATE
    split_fuse_mte2_interleaved_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
add_executable(split_fuse_mte2_interleaved_tail main.cpp)
target_compile_definitions(split_fuse_mte2_interleaved_tail PRIVATE
    MIDLAYER_TIMED_LAUNCHES=48)
target_include_directories(split_fuse_mte2_interleaved_tail PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(split_fuse_mte2_interleaved_tail PRIVATE
    ${ASCEND_HOME_PATH}/lib64 ${ASCEND_HOME_PATH}/runtime/lib64
    /usr/local/Ascend/cann-8.5.1/lib64)
target_link_libraries(split_fuse_mte2_interleaved_tail PRIVATE
    split_fuse_mte2_interleaved_kernel
    ascendcl runtime profapi mmpa c_sec stdc++ pthread dl)
