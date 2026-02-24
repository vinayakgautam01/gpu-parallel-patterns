#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpp/types.hpp"

namespace gpp {

struct CliArgs {
    BenchConfig bench;
    int         width  = 0;     // for 2-D patterns (convolution, stencil)
    int         height = 0;
    int         radius = 1;     // filter radius R; pattern-specific max enforced by caller
    uint32_t    seed   = 42;
};

/// Parse argv into CliArgs. Unrecognised flags are silently skipped.
///
/// Usage:
///   int main(int argc, char** argv) {
///       auto args = gpp::parse_cli(argc, argv);
///       ...
///   }
inline CliArgs parse_cli(int argc, char** argv) {
    CliArgs args;

    for (int i = 1; i < argc; ++i) {
        auto match = [&](const char* flag) { return std::strcmp(argv[i], flag) == 0; };
        auto next  = [&]() -> const char* {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };

        if (match("--variant") || match("-v")) {
            const char* v = next();
            if (!v) continue;
            if (std::strcmp(v, "baseline") == 0) args.bench.variant = Variant::Baseline;
            else if (std::strcmp(v, "opt1") == 0) args.bench.variant = Variant::Opt1;
            else if (std::strcmp(v, "opt2") == 0) args.bench.variant = Variant::Opt2;
            else if (std::strcmp(v, "opt3") == 0) args.bench.variant = Variant::Opt3;
            else if (std::strcmp(v, "opt4") == 0) args.bench.variant = Variant::Opt4;
            else {
                std::fprintf(stderr, "error: unknown variant '%s'\n", v);
                std::exit(1);
            }
        } else if (match("--n") || match("-n")) {
            const char* v = next();
            if (v) args.bench.size = std::atoi(v);
        } else if (match("--R") || match("--radius")) {
            const char* v = next();
            if (v) args.radius = std::atoi(v);
        } else if (match("--w")) {
            const char* v = next();
            if (v) args.width = std::atoi(v);
        } else if (match("--h")) {
            const char* v = next();
            if (v) args.height = std::atoi(v);
        } else if (match("--iters")) {
            const char* v = next();
            if (v) args.bench.iters = std::atoi(v);
        } else if (match("--warmup")) {
            const char* v = next();
            if (v) args.bench.warmup = std::atoi(v);
        } else if (match("--seed")) {
            const char* v = next();
            if (v) args.seed = static_cast<uint32_t>(std::atoi(v));
        } else if (match("--verify")) {
            args.bench.verify = true;
        } else if (match("--help")) {
            std::printf(
                "Usage: <binary> [options]\n"
                "  --variant, -v  baseline|opt1|opt2|opt3|opt4  (default: baseline)\n"
                "  --n, -n        number of elements  (default: %d)\n"
                "  --R, --radius  filter radius        (default: 1)\n"
                "  --w            width  (2-D patterns)\n"
                "  --h            height (2-D patterns)\n"
                "  --iters        timed iterations     (default: %d)\n"
                "  --warmup       warm-up iterations   (default: %d)\n"
                "  --seed         RNG seed             (default: %u)\n"
                "  --verify       compare GPU vs CPU\n"
                "  --help         show this message\n",
                args.bench.size, args.bench.iters, args.bench.warmup, args.seed);
            std::exit(0);
        }
    }

    // For 2-D patterns: if width/height set, derive size from them.
    if (args.width > 0 && args.height > 0) {
        args.bench.size = args.width * args.height;
    }

    return args;
}

}  // namespace gpp
