cmake_minimum_required(VERSION 3.16)
project(empty_launch_bench)

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

ascendc_library(empty_aic_kernel STATIC empty_aic_kernel.cpp)
ascendc_library(empty_aiv_kernel STATIC empty_aiv_kernel.cpp)
ascendc_library(nonempty_aiv_kernel STATIC nonempty_aiv_kernel.cpp)
ascendc_library(nonempty_mix_aiv_kernel STATIC nonempty_mix_aiv_kernel.cpp)
ascendc_library(aiv_worker_start_kernel STATIC aiv_worker_start_kernel.cpp)
ascendc_library(aiv_memory_task_floor_kernel STATIC
    aiv_memory_task_floor_kernel.cpp)
ascendc_library(aiv_norm_control_task_floor_kernel STATIC
    aiv_norm_control_task_floor_kernel.cpp)
ascendc_library(aiv_text_footprint_1_kernel STATIC
    aiv_text_footprint_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_1_kernel PRIVATE TEXT_BODY=1)
ascendc_library(aiv_text_footprint_32_kernel STATIC
    aiv_text_footprint_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_32_kernel PRIVATE TEXT_BODY=32)
ascendc_library(aiv_text_footprint_96_kernel STATIC
    aiv_text_footprint_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_96_kernel PRIVATE TEXT_BODY=96)
ascendc_library(aiv_text_footprint_192_kernel STATIC
    aiv_text_footprint_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_192_kernel PRIVATE TEXT_BODY=192)
ascendc_library(aiv_text_footprint_384_kernel STATIC
    aiv_text_footprint_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_384_kernel PRIVATE TEXT_BODY=384)
ascendc_library(aiv_text_footprint_768_kernel STATIC
    aiv_text_footprint_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_768_kernel PRIVATE TEXT_BODY=768)
ascendc_library(aiv_text_footprint_alt_1_kernel STATIC
    aiv_text_footprint_alt_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_alt_1_kernel PRIVATE ALT_TEXT_BODY=1)
ascendc_library(aiv_text_footprint_alt_96_kernel STATIC
    aiv_text_footprint_alt_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_alt_96_kernel PRIVATE ALT_TEXT_BODY=96)
ascendc_library(aiv_text_footprint_alt_192_kernel STATIC
    aiv_text_footprint_alt_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_alt_192_kernel PRIVATE ALT_TEXT_BODY=192)
ascendc_library(aiv_text_footprint_alt_384_kernel STATIC
    aiv_text_footprint_alt_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_alt_384_kernel PRIVATE ALT_TEXT_BODY=384)
ascendc_library(aiv_text_footprint_alt_768_kernel STATIC
    aiv_text_footprint_alt_kernel.cpp)
ascendc_compile_definitions(aiv_text_footprint_alt_768_kernel PRIVATE ALT_TEXT_BODY=768)
ascendc_library(empty_mixed_kernel STATIC empty_mixed_kernel.cpp)
ascendc_library(empty_mix_aic_kernel STATIC empty_mix_aic_kernel.cpp)

add_executable(empty_aic_bench empty_aic_main.cpp)
target_include_directories(empty_aic_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(empty_aic_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(empty_aic_bench PRIVATE empty_aic_kernel ascendcl)

add_executable(empty_aiv_bench empty_aiv_main.cpp)
target_include_directories(empty_aiv_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(empty_aiv_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(empty_aiv_bench PRIVATE empty_aiv_kernel ascendcl)

add_executable(nonempty_aiv_bench nonempty_aiv_main.cpp)
target_include_directories(nonempty_aiv_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(nonempty_aiv_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(nonempty_aiv_bench PRIVATE nonempty_aiv_kernel ascendcl)

add_executable(nonempty_mix_aiv_bench nonempty_mix_aiv_main.cpp)
target_include_directories(nonempty_mix_aiv_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(nonempty_mix_aiv_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(nonempty_mix_aiv_bench PRIVATE
    nonempty_mix_aiv_kernel ascendcl)

add_executable(aiv_worker_start_bench aiv_worker_start_main.cpp)
target_include_directories(aiv_worker_start_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_worker_start_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_worker_start_bench PRIVATE
    aiv_worker_start_kernel ascendcl)

add_executable(aiv_memory_task_floor_bench aiv_memory_task_floor_main.cpp)
target_include_directories(aiv_memory_task_floor_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_memory_task_floor_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_memory_task_floor_bench PRIVATE
    aiv_memory_task_floor_kernel ascendcl)

add_executable(aiv_norm_control_task_floor_bench
    aiv_norm_control_task_floor_main.cpp)
target_include_directories(aiv_norm_control_task_floor_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_norm_control_task_floor_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_norm_control_task_floor_bench PRIVATE
    aiv_norm_control_task_floor_kernel ascendcl)

add_executable(empty_mixed_bench empty_mixed_main.cpp)
target_include_directories(empty_mixed_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(empty_mixed_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(empty_mixed_bench PRIVATE empty_mixed_kernel ascendcl)

add_executable(empty_mix_aic_bench empty_mix_aic_main.cpp)
target_include_directories(empty_mix_aic_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(empty_mix_aic_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(empty_mix_aic_bench PRIVATE empty_mix_aic_kernel ascendcl)

add_executable(aiv_text_footprint_1_bench aiv_text_footprint_main.cpp)
target_include_directories(aiv_text_footprint_1_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_text_footprint_1_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_text_footprint_1_bench PRIVATE
    aiv_text_footprint_1_kernel ascendcl)

add_executable(aiv_text_footprint_32_bench aiv_text_footprint_main.cpp)
target_include_directories(aiv_text_footprint_32_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_text_footprint_32_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_text_footprint_32_bench PRIVATE
    aiv_text_footprint_32_kernel ascendcl)

add_executable(aiv_text_footprint_96_bench aiv_text_footprint_main.cpp)
target_include_directories(aiv_text_footprint_96_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_text_footprint_96_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_text_footprint_96_bench PRIVATE
    aiv_text_footprint_96_kernel ascendcl)

add_executable(aiv_text_footprint_192_bench aiv_text_footprint_main.cpp)
target_include_directories(aiv_text_footprint_192_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_text_footprint_192_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_text_footprint_192_bench PRIVATE
    aiv_text_footprint_192_kernel ascendcl)

add_executable(aiv_text_footprint_384_bench aiv_text_footprint_main.cpp)
target_include_directories(aiv_text_footprint_384_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_text_footprint_384_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_text_footprint_384_bench PRIVATE
    aiv_text_footprint_384_kernel ascendcl)

add_executable(aiv_text_footprint_768_bench aiv_text_footprint_main.cpp)
target_include_directories(aiv_text_footprint_768_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_text_footprint_768_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_text_footprint_768_bench PRIVATE
    aiv_text_footprint_768_kernel ascendcl)

add_executable(aiv_icache_evict_1_bench aiv_icache_evict_main.cpp)
target_include_directories(aiv_icache_evict_1_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_icache_evict_1_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_icache_evict_1_bench PRIVATE
    aiv_text_footprint_1_kernel aiv_text_footprint_alt_1_kernel ascendcl)

add_executable(aiv_icache_evict_96_bench aiv_icache_evict_main.cpp)
target_include_directories(aiv_icache_evict_96_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_icache_evict_96_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_icache_evict_96_bench PRIVATE
    aiv_text_footprint_96_kernel aiv_text_footprint_alt_96_kernel ascendcl)

add_executable(aiv_icache_evict_192_bench aiv_icache_evict_main.cpp)
target_include_directories(aiv_icache_evict_192_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_icache_evict_192_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_icache_evict_192_bench PRIVATE
    aiv_text_footprint_192_kernel aiv_text_footprint_alt_192_kernel ascendcl)

add_executable(aiv_icache_evict_384_bench aiv_icache_evict_main.cpp)
target_include_directories(aiv_icache_evict_384_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_icache_evict_384_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_icache_evict_384_bench PRIVATE
    aiv_text_footprint_384_kernel aiv_text_footprint_alt_384_kernel ascendcl)

add_executable(aiv_icache_evict_768_bench aiv_icache_evict_main.cpp)
target_include_directories(aiv_icache_evict_768_bench PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include)
target_link_directories(aiv_icache_evict_768_bench PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64)
target_link_libraries(aiv_icache_evict_768_bench PRIVATE
    aiv_text_footprint_768_kernel aiv_text_footprint_alt_768_kernel ascendcl)
