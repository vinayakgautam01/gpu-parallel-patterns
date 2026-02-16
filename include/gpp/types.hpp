#pragma once

#include <cstdint>

namespace gpp {

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

using u8  = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

// ---------------------------------------------------------------------------
// Kernel variant selector
// ---------------------------------------------------------------------------

enum class Variant {
    Baseline,
    Opt1,
    Opt2,
};

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------

struct BenchConfig {
    Variant variant = Variant::Baseline;
    int     size    = 1 << 20;   // number of elements
    int     iters   = 100;       // timed iterations
    int     warmup  = 10;        // warm-up iterations (not timed)
    bool    verify  = false;     // compare GPU output against CPU reference
};

}  // namespace gpp
