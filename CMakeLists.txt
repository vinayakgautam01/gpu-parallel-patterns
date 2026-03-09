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

# CPU timing binary for two-pass benchmark scripts.
add_executable(hist_cpu_timing
    src/patterns/histogram/cpu_bench.cpp
)
target_include_directories(hist_cpu_timing PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/histogram
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(hist_cpu_timing PRIVATE gpp_common)

# ---------------------------------------------------------------------------
# Reduce pattern (sum reduction)
# ---------------------------------------------------------------------------

# GPU kernel implementations — compiled once, linked by test + bench
add_library(reduce_kernels
    src/patterns/reduce/baseline.cu
    src/patterns/reduce/opt1_contiguous_threads.cu
    src/patterns/reduce/opt2_coarsened.cu
    src/patterns/reduce/opt3_warp_shuffle.cu
    src/patterns/reduce/dispatch.cu
)
target_include_directories(reduce_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/src/patterns/reduce
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(reduce_kernels PUBLIC gpp_common)

# CPU-only correctness test (no GPU required)
add_executable(reduce_cpu_test
    src/patterns/reduce/cpu_ref_test.cpp
)
target_include_directories(reduce_cpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/reduce
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(reduce_cpu_test PRIVATE gpp_common)

# GPU correctness test — validates all variants against reduce_sum_cpu_ref.
add_executable(reduce_gpu_test
    src/patterns/reduce/gpu_test.cu
)
target_include_directories(reduce_gpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/reduce
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(reduce_gpu_test PRIVATE reduce_kernels gpp_common)

# Benchmark binary — picked up by scripts/bench.sh (*_bench glob).
add_executable(reduce_bench
    src/patterns/reduce/bench.cu
)
target_include_directories(reduce_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/reduce
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(reduce_bench PRIVATE reduce_kernels gpp_common)

# ---------------------------------------------------------------------------
# Scan pattern (inclusive prefix sum)
# ---------------------------------------------------------------------------

# GPU kernel implementations — compiled once, linked by test + bench
add_library(scan_kernels
    src/patterns/scan/baseline.cu
    src/patterns/scan/opt1_kogge_stone.cu
    src/patterns/scan/opt2_brent_kung.cu
    src/patterns/scan/opt3_coarsened.cu
    src/patterns/scan/opt4_single_pass.cu
    src/patterns/scan/add_block_prefix.cu
    src/patterns/scan/dispatch.cu
)
target_include_directories(scan_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/src/patterns/scan
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(scan_kernels PUBLIC gpp_common)

# CPU-only correctness test (no GPU required)
add_executable(scan_cpu_test
    src/patterns/scan/cpu_ref_test.cpp
)
target_include_directories(scan_cpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/scan
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(scan_cpu_test PRIVATE gpp_common)

# GPU correctness test — validates all variants against inclusive_scan_cpu_ref.
add_executable(scan_gpu_test
    src/patterns/scan/gpu_test.cu
)
target_include_directories(scan_gpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/scan
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(scan_gpu_test PRIVATE scan_kernels gpp_common)

# Benchmark binary — picked up by scripts/bench.sh (*_bench glob).
add_executable(scan_bench
    src/patterns/scan/bench.cu
)
target_include_directories(scan_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/scan
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(scan_bench PRIVATE scan_kernels gpp_common)

# ---------------------------------------------------------------------------
# Merge pattern
# ---------------------------------------------------------------------------

# GPU kernel implementations — compiled once, linked by test + bench
add_library(merge_kernels
    src/patterns/merge/baseline.cu
    src/patterns/merge/opt1_tiled.cu
    src/patterns/merge/opt2_circular_buf.cu
    src/patterns/merge/dispatch.cu
)
target_include_directories(merge_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/src/patterns/merge
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(merge_kernels PUBLIC gpp_common)

# CPU-only correctness test (no GPU required)
add_executable(merge_cpu_test
    src/patterns/merge/cpu_ref_test.cpp
)
target_include_directories(merge_cpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/merge
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(merge_cpu_test PRIVATE gpp_common)

# GPU correctness test — validates all variants against merge_cpu_ref.
add_executable(merge_gpu_test
    src/patterns/merge/gpu_test.cu
)
target_include_directories(merge_gpu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/merge
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(merge_gpu_test PRIVATE merge_kernels gpp_common)

# Benchmark binary — picked up by scripts/bench.sh (*_bench glob).
add_executable(merge_bench
    src/patterns/merge/bench.cu
)
target_include_directories(merge_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/src/patterns/merge
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(merge_bench PRIVATE merge_kernels gpp_common)
