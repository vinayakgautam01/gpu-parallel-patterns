#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "gpp/common/compare.hpp"
#include "gpp/common/rng.hpp"
#include "cpu_ref.hpp"

#define REQUIRE(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "REQUIRE failed: %s  at %s:%d\n", \
                         #cond, __FILE__, __LINE__); \
            std::exit(1); \
        } \
    } while (0)

// -------------------------------------------------------------------------
// Independent slow reference used only to cross-check cpu_ref.
// Accumulates in double to reduce rounding error.
// -------------------------------------------------------------------------
static void stencil3d_slow_ref(const std::vector<float>& in,
                                std::vector<float>& out,
                                int nx, int ny, int nz,
                                const gpp::stencil::Weights7& w) {
    const int slab = nx * ny;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int idx = z * slab + y * nx + x;

                if (x >= 1 && x < nx - 1 &&
                    y >= 1 && y < ny - 1 &&
                    z >= 1 && z < nz - 1) {
                    double acc = static_cast<double>(w.c) * static_cast<double>(in[idx])
                        + static_cast<double>(w.xn) * static_cast<double>(in[idx - 1])
                        + static_cast<double>(w.xp) * static_cast<double>(in[idx + 1])
                        + static_cast<double>(w.yn) * static_cast<double>(in[idx - nx])
                        + static_cast<double>(w.yp) * static_cast<double>(in[idx + nx])
                        + static_cast<double>(w.zn) * static_cast<double>(in[idx - slab])
                        + static_cast<double>(w.zp) * static_cast<double>(in[idx + slab]);
                    out[idx] = static_cast<float>(acc);
                } else {
                    out[idx] = in[idx];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Identity stencil: center=1, all neighbors=0 → output == input.
static void test_identity_stencil() {
    const int nx = 5, ny = 4, nz = 3;
    const int n = nx * ny * nz;

    std::vector<float> in(n);
    gpp::fill_range(in);

    gpp::stencil::Weights7 w{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> out(n, -1.0f);
    gpp::stencil::stencil3d_cpu_ref(in.data(), out.data(), nx, ny, nz, w);

    auto cmp = gpp::compare_arrays_float(in.data(), out.data(), n, 0.0f, 0.0f);
    gpp::print_compare(cmp, "identity_stencil");
    REQUIRE(cmp.ok);
}

// All-ones input, uniform weight=1 → boundary voxels pass through (=1),
// interior voxels get center + 6 neighbors = 7.
static void test_ones_neighbor_counts() {
    const int nx = 4, ny = 4, nz = 4;
    const int n = nx * ny * nz;

    std::vector<float> in(n, 1.0f);
    gpp::stencil::Weights7 w{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(n, 0.0f);
    gpp::stencil::stencil3d_cpu_ref(in.data(), out.data(), nx, ny, nz, w);

    auto at = [&](int x, int y, int z) { return out[z * ny * nx + y * nx + x]; };

    // corner (0,0,0): boundary → passthrough = 1
    REQUIRE(std::fabs(at(0, 0, 0) - 1.0f) < 1e-6f);
    // edge (1,0,0): boundary → passthrough = 1
    REQUIRE(std::fabs(at(1, 0, 0) - 1.0f) < 1e-6f);
    // face-interior (1,1,0): boundary → passthrough = 1
    REQUIRE(std::fabs(at(1, 1, 0) - 1.0f) < 1e-6f);
    // interior (1,1,1): center + 6 neighbors = 7
    REQUIRE(std::fabs(at(1, 1, 1) - 7.0f) < 1e-6f);
    // interior (2,2,2): center + 6 neighbors = 7
    REQUIRE(std::fabs(at(2, 2, 2) - 7.0f) < 1e-6f);

    std::fprintf(stderr, "[ones_neighbor_counts] PASS\n");
}

// Known interior point: verify by hand computation.
static void test_known_interior_point() {
    const int nx = 3, ny = 3, nz = 3;
    const int n = nx * ny * nz;

    std::vector<float> in(n);
    gpp::fill_range(in);

    // center = (1,1,1) → idx = 1*9 + 1*3 + 1 = 13
    // xn=(1,1,0)=12, xp=(1,1,2)=14, yn=(1,0,1)=10, yp=(1,2,1)=16,
    // zn=(0,1,1)=4,  zp=(2,1,1)=22
    gpp::stencil::Weights7 w{2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(n, 0.0f);
    gpp::stencil::stencil3d_cpu_ref(in.data(), out.data(), nx, ny, nz, w);

    // expected = 2*13 + 12 + 14 + 10 + 16 + 4 + 22 = 26 + 78 = 104
    float got = out[13];
    std::fprintf(stderr, "[known_interior_point] center=%.1f (expected 104.0)\n", got);
    REQUIRE(std::fabs(got - 104.0f) < 1e-4f);
}

// Random cross-check: cpu_ref must match slow_ref within float tolerance.
static void test_random_crosscheck() {
    for (int t = 0; t < 25; ++t) {
        const int nx = 5 + (t % 7);
        const int ny = 4 + (t % 5);
        const int nz = 3 + (t % 4);
        const int n = nx * ny * nz;

        std::vector<float> in(n);
        gpp::fill_random_float(in, 100 + static_cast<uint32_t>(t), -1.0f, 1.0f);

        // Randomise weights per iteration.
        std::vector<float> wv(7);
        gpp::fill_random_float(wv, 200 + static_cast<uint32_t>(t), -2.0f, 2.0f);
        gpp::stencil::Weights7 w{wv[0], wv[1], wv[2], wv[3], wv[4], wv[5], wv[6]};

        std::vector<float> out_ref(n, 0.0f);
        std::vector<float> out_slow(n, 0.0f);

        gpp::stencil::stencil3d_cpu_ref(in.data(), out_ref.data(), nx, ny, nz, w);
        stencil3d_slow_ref(in, out_slow, nx, ny, nz, w);

        auto cmp = gpp::compare_arrays_float(out_slow.data(), out_ref.data(), n,
                                              1e-5f, 1e-5f);
        if (!cmp.ok) {
            gpp::print_compare(cmp, "random_crosscheck");
            std::fprintf(stderr, "  nx=%d ny=%d nz=%d\n", nx, ny, nz);
            std::exit(1);
        }
    }
    std::fprintf(stderr, "[random_crosscheck] PASS (25 cases)\n");
}

int main() {
    test_identity_stencil();
    test_ones_neighbor_counts();
    test_known_interior_point();
    test_random_crosscheck();
    std::puts("PASS stencil_cpu_ref_test");
    return 0;
}
