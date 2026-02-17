# CUDA + host compiler warning flags.
# Applied globally to all targets via add_compile_options.

# Host compiler warnings (g++ / clang++) — forwarded by nvcc via -Xcompiler
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wshadow>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wconversion>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-parameter>"
)

# Pure C++ files get the same warnings directly
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wextra>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wshadow>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wconversion>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>"
)

# NVCC-specific: warn about cross-execution-space calls (host calling device, etc.)
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:--Wreorder>"
)
