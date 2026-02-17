# Address + UndefinedBehavior sanitizer support (host code only).
# Enabled only for Debug builds:  ./scripts/build.sh Debug
#
# Usage on Colab:
#   !bash gpu-parallel-patterns/scripts/build.sh Debug
#   !./gpu-parallel-patterns/build/bin/reduce_test    ← ASan catches memory bugs

option(ENABLE_SANITIZERS "Enable ASan + UBSan for Debug builds" ON)

if(ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Apply to C++ (host) code only — sanitizers don't work on GPU code
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>"
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=address,undefined>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-omit-frame-pointer>"
    )
    add_link_options(-fsanitize=address,undefined)

    message(STATUS "Sanitizers enabled (ASan + UBSan)")
endif()
