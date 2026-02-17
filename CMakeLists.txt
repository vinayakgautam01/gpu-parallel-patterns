cmake_minimum_required(VERSION 3.24)
project(gpu-parallel-patterns LANGUAGES CXX CUDA)

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

include_directories(${CMAKE_SOURCE_DIR}/include)

# Shared utilities library (compiled once, linked by all pattern binaries)
add_library(gpp_common
    src/common/checks.cu
    src/common/timers.cu
    src/common/cli.cpp
    src/common/rng.cpp
)

# TODO: add pattern executable targets (reduce, scan, histogram, etc.)
