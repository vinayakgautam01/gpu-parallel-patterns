#pragma once

namespace gpp::gemm {

/// CPU reference matrix multiplication: C = A * B
///
/// Parameters
///   A  — host input, I x J, row-major
///   B  — host input, J x K, row-major
///   C  — host output, I x K, row-major
///   I  — rows of A (and C)
///   J  — cols of A / rows of B (contraction dimension)
///   K  — cols of B (and C)
inline void matmul_cpu_ref(const float* A, const float* B, float* C,
                           int I, int J, int K) {
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            float acc = 0.0f;
            for (int j = 0; j < J; ++j)
                acc += A[i * J + j] * B[j * K + k];
            C[i * K + k] = acc;
        }
    }
}

}  // namespace gpp::gemm
