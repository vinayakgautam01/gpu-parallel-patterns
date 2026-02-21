cmake_minimum_required(VERSION 3.20)
project(gpu_parallel_patterns LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/cuda_warnings.cmake)
include(cmake/sanitizers.cmake)

# Shared utilities library (compiled once, linked by all pattern binaries)
add_library(gpp_common
    src/common/checks.cu
    src/common/timers.cu
    src/common/cli.cpp
    src/common/rng.cpp
)
target_include_directories(gpp_common PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Smoke test
add_executable(cuda_smoke src/cuda_smoke.cu)
target_link_libraries(cuda_smoke PRIVATE gpp_common)

# ---------------------------------------------------------------------------
# Convolution pattern
# ---------------------------------------------------------------------------
add_executable(conv_test_cpu
    src/patterns/convolution/test_cpu_ref.cpp
)
target_include_directories(conv_test_cpu PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/convolution
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(conv_test_cpu PRIVATE gpp_common)

# TODO: add pattern executable targets (reduce, scan, histogram, etc.)
