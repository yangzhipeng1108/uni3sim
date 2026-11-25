cmake_minimum_required(VERSION 3.16)
project(fia_brcb_service)

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

ascendc_library(fia_brcb_service_kernel STATIC fia_brcb_kernel.cpp)
add_executable(fia_brcb_service fia_brcb_main.cpp)
target_include_directories(fia_brcb_service PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(fia_brcb_service PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(
    fia_brcb_service PRIVATE fia_brcb_service_kernel ascendcl)
