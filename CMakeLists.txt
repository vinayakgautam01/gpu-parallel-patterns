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

# GPU kernel implementations — compiled once, linked by test + bench
add_library(conv_kernels
    src/patterns/convolution/baseline.cu
    src/patterns/convolution/opt1_const_mem.cu
    src/patterns/convolution/opt2_tiled.cu
    src/patterns/convolution/opt3_cached_halo.cu
    src/patterns/convolution/opt4_separable.cu
    src/patterns/convolution/dispatch.cu
)
target_include_directories(conv_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/src/patterns/convolution
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(conv_kernels PUBLIC gpp_common)

# CPU-only correctness test (no GPU required)
add_executable(conv_cpu_test
    src/patterns/convolution/cpu_ref_test.cpp
)
target_include_directories(conv_cpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/convolution
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(conv_cpu_test PRIVATE gpp_common)

# GPU correctness test — validates all variants against conv2d_cpu_ref.
add_executable(conv_gpu_test
    src/patterns/convolution/gpu_test.cu
)
target_include_directories(conv_gpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/convolution
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(conv_gpu_test PRIVATE conv_kernels gpp_common)

# Benchmark binary — picked up by scripts/bench.sh (*_bench glob).
add_executable(conv_bench
    src/patterns/convolution/bench.cu
)
target_include_directories(conv_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/convolution
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(conv_bench PRIVATE conv_kernels gpp_common)

# ---------------------------------------------------------------------------
# Stencil pattern (7-point 3D)
# ---------------------------------------------------------------------------

# GPU kernel implementations — compiled once, linked by test + bench
add_library(stencil_kernels
    src/patterns/stencil/baseline.cu
    src/patterns/stencil/opt1_shared_halo.cu
    src/patterns/stencil/opt2_thread_coarsening.cu
    src/patterns/stencil/opt3_register_tiling.cu
    src/patterns/stencil/dispatch.cu
)
target_include_directories(stencil_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/src/patterns/stencil
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(stencil_kernels PUBLIC gpp_common)

# CPU-only correctness test (no GPU required)
add_executable(stencil_cpu_test
    src/patterns/stencil/cpu_ref_test.cpp
)
target_include_directories(stencil_cpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/stencil
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(stencil_cpu_test PRIVATE gpp_common)

# GPU correctness test — validates all variants against stencil3d_cpu_ref.
add_executable(stencil_gpu_test
    src/patterns/stencil/gpu_test.cu
)
target_include_directories(stencil_gpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/stencil
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(stencil_gpu_test PRIVATE stencil_kernels gpp_common)

# Benchmark binary — picked up by scripts/bench.sh (*_bench glob).
add_executable(stencil_bench
    src/patterns/stencil/bench.cu
)
target_include_directories(stencil_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/stencil
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(stencil_bench PRIVATE stencil_kernels gpp_common)

# ---------------------------------------------------------------------------
# Histogram pattern
# ---------------------------------------------------------------------------

# GPU kernel implementations — compiled once, linked by test + bench
add_library(hist_kernels
    src/patterns/histogram/baseline_global_atomics.cu
    src/patterns/histogram/opt1_block_shared_hist.cu
    src/patterns/histogram/opt2_coarsen_contiguous.cu
    src/patterns/histogram/opt3_coarsen_interleaved.cu
    src/patterns/histogram/opt4_aggregation.cu
    src/patterns/histogram/dispatch.cu
)
target_include_directories(hist_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/src/patterns/histogram
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(hist_kernels PUBLIC gpp_common)

# CPU-only correctness test (no GPU required)
add_executable(hist_cpu_test
    src/patterns/histogram/cpu_ref_test.cpp
)
target_include_directories(hist_cpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/histogram
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(hist_cpu_test PRIVATE gpp_common)

# GPU correctness test — validates all variants against histogram_cpu_ref.
add_executable(hist_gpu_test
    src/patterns/histogram/gpu_test.cu
)
target_include_directories(hist_gpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/histogram
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(hist_gpu_test PRIVATE hist_kernels gpp_common)

# Benchmark binary — picked up by scripts/bench.sh (*_bench glob).
add_executable(hist_bench
    src/patterns/histogram/bench.cu
)
target_include_directories(hist_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/histogram
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(hist_bench PRIVATE hist_kernels gpp_common)

# TODO: add pattern executable targets (reduce, scan, etc.)
