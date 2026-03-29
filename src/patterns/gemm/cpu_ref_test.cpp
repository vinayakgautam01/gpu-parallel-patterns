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
static void matmul_slow_ref(const std::vector<float>& A,
                            const std::vector<float>& B,
                            std::vector<float>& C,
                            int I, int J, int K) {
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int j = 0; j < J; ++j)
                acc += static_cast<double>(A[i * J + j]) *
                       static_cast<double>(B[j * K + k]);
            C[i * K + k] = static_cast<float>(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Identity matrix: A * I = A.
static void test_identity() {
    const int N = 5;
    std::vector<float> A(N * N);
    gpp::fill_range(A);

    std::vector<float> eye(N * N, 0.0f);
    for (int i = 0; i < N; ++i) eye[i * N + i] = 1.0f;

    std::vector<float> C(N * N, -1.0f);
    gpp::gemm::matmul_cpu_ref(A.data(), eye.data(), C.data(), N, N, N);

    auto cmp = gpp::compare_arrays_float(A.data(), C.data(), N * N, 0.0f, 0.0f);
    gpp::print_compare(cmp, "identity");
    REQUIRE(cmp.ok);
}

// Known 2x3 * 3x2: hand-computed result.
//   A = [1 2 3; 4 5 6]   B = [7 8; 9 10; 11 12]
//   C = [1*7+2*9+3*11  1*8+2*10+3*12;  4*7+5*9+6*11  4*8+5*10+6*12]
//     = [58 64; 139 154]
static void test_known_2x3_3x2() {
    std::vector<float> A = {1,2,3, 4,5,6};
    std::vector<float> B = {7,8, 9,10, 11,12};
    std::vector<float> C(2 * 2, 0.0f);

    gpp::gemm::matmul_cpu_ref(A.data(), B.data(), C.data(), 2, 3, 2);

    REQUIRE(std::fabs(C[0] -  58.0f) < 1e-6f);
    REQUIRE(std::fabs(C[1] -  64.0f) < 1e-6f);
    REQUIRE(std::fabs(C[2] - 139.0f) < 1e-6f);
    REQUIRE(std::fabs(C[3] - 154.0f) < 1e-6f);

    std::fprintf(stderr, "[known_2x3_3x2] PASS\n");
}

// 1x1 degenerate case.
static void test_1x1() {
    std::vector<float> A = {3.0f};
    std::vector<float> B = {5.0f};
    std::vector<float> C(1, 0.0f);

    gpp::gemm::matmul_cpu_ref(A.data(), B.data(), C.data(), 1, 1, 1);
    REQUIRE(std::fabs(C[0] - 15.0f) < 1e-6f);
    std::fprintf(stderr, "[1x1] PASS\n");
}

// Random cross-check: cpu_ref must match slow_ref (double) within float tolerance.
static void test_random_crosscheck() {
    struct Case { int I, J, K; };
    const Case cases[] = {
        {4, 4, 4}, {7, 3, 5}, {1, 10, 1}, {10, 1, 10},
        {16, 16, 16}, {17, 13, 11}, {32, 64, 32},
        {64, 128, 64}, {100, 100, 100},
    };

    for (int t = 0; t < static_cast<int>(sizeof(cases)/sizeof(cases[0])); ++t) {
        const auto& c = cases[t];

        std::vector<float> A(c.I * c.J), B(c.J * c.K);
        gpp::fill_random_float(A, 100 + static_cast<uint32_t>(t), -1.0f, 1.0f);
        gpp::fill_random_float(B, 200 + static_cast<uint32_t>(t), -1.0f, 1.0f);

        std::vector<float> out_ref(c.I * c.K, 0.0f);
        std::vector<float> out_slow(c.I * c.K, 0.0f);

        gpp::gemm::matmul_cpu_ref(A.data(), B.data(), out_ref.data(), c.I, c.J, c.K);
        matmul_slow_ref(A, B, out_slow, c.I, c.J, c.K);

        auto cmp = gpp::compare_arrays_float(out_slow.data(), out_ref.data(),
                                             c.I * c.K, 1e-4f, 1e-4f);
        if (!cmp.ok) {
            gpp::print_compare(cmp, "random_crosscheck");
            std::fprintf(stderr, "  I=%d J=%d K=%d\n", c.I, c.J, c.K);
            std::exit(1);
        }
    }
    std::fprintf(stderr, "[random_crosscheck] PASS (%d cases)\n",
                 static_cast<int>(sizeof(cases)/sizeof(cases[0])));
}

int main() {
    test_identity();
    test_known_2x3_3x2();
    test_1x1();
    test_random_crosscheck();
    std::puts("PASS gemm_cpu_ref_test");
    return 0;
}
