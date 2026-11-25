cmake_minimum_required(VERSION 3.16)
project(single_tbuf_lifecycle)

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

ascendc_library(single_tbuf_lifecycle_kernel STATIC single_tbuf_lifecycle_kernel.cpp)

add_executable(single_tbuf_lifecycle single_tbuf_lifecycle_main.cpp)
target_include_directories(single_tbuf_lifecycle PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(single_tbuf_lifecycle PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(single_tbuf_lifecycle PRIVATE single_tbuf_lifecycle_kernel ascendcl)
