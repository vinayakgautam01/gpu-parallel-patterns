#pragma once

namespace gpp::hist {

constexpr int NUM_BINS = 7;

/// CPU reference histogram over lowercase ASCII letters.
///
/// Each character in [a-z] is mapped to bin = (ch - 'a') / 4, giving 7 bins
/// (bin 0 = a-d, bin 1 = e-h, ..., bin 6 = y-z). Non-lowercase characters
/// are silently ignored.
///
/// Parameters
///   data   — input character buffer
///   length — number of characters to process
///   histo  — output histogram array, must have at least NUM_BINS elements
///            (caller is responsible for zeroing before the call)
inline void histogram_cpu_ref(const char* data, unsigned int length,
                              unsigned int* histo) {
    for (unsigned int i = 0; i < length; ++i) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26)
            histo[pos / 4]++;
    }
}

}  // namespace gpp::hist
