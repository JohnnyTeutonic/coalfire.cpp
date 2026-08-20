// SWA deterministic receipt (STUDIO_PLAN section 12, queued 2026-07-31).
//
// One MultiHeadAttention instance, window_size = 4, T = 64. Perturb the
// input at controlled distances from a probe row and measure the change
// in that row's output. No training, no seeds, no noise:
//
//   probe = last row (t = 63):
//     A. perturb t = 20  (FAR outside the +/-2 window)  -> must NOT change
//     B. perturb t = 62  (inside window, past)          -> MUST change
//   probe = middle row (t = 30):
//     C. perturb t = 31  (inside window, FUTURE)        -> must not change
//        IF causality is combined with the symmetric window mask; a change
//        here means the window leaks future tokens in training.
//   Control: window OFF, perturb t = 20, probe t = 63   -> MUST change
//            (full attention sees everything; also proves probe validity).
#include <cmath>
#include <cstdio>
#include <random>

#include "../include/attention.hpp"
#include "../include/config.hpp"

namespace {
Matrix rand_input(size_t T, size_t d, unsigned seed) {
    Matrix x(T, d);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    for (size_t i = 0; i < T; ++i)
        for (size_t j = 0; j < d; ++j) x(i, j) = u(rng);
    return x;
}

double row_delta(const Matrix& a, const Matrix& b, size_t row) {
    double s = 0;
    for (size_t j = 0; j < a.cols(); ++j) {
        const double d = double(a(row, j)) - b(row, j);
        s += d * d;
    }
    return std::sqrt(s);
}
}  // namespace

int main() {
    const size_t T = 64, hidden = 64, heads = 4, head_dim = 16;
    const AttentionMask mask = AttentionMask::create_causal_mask(T);
    Matrix x = rand_input(T, hidden, 7);

    auto run_case = [&](bool use_window, size_t window, size_t perturb_row,
                        size_t probe_row) {
        MultiHeadAttention attn(hidden, heads, head_dim, /*dropout=*/0.0f,
                                /*flash=*/false, /*rope=*/false,
                                use_window, window,
                                /*gqa=*/false, heads, T, /*fp16=*/false,
                                /*fused=*/false);
        Matrix base = attn.forward_batched(x, mask, 1, T);
        Matrix x2 = x;
        for (size_t j = 0; j < hidden; ++j) x2(perturb_row, j) += 10.0f;
        Matrix pert = attn.forward_batched(x2, mask, 1, T);
        return row_delta(base, pert, probe_row);
    };

    const double dA = run_case(true, 4, 20, 63);   // far, windowed
    const double dB = run_case(true, 4, 62, 63);   // near-past, windowed
    const double dC = run_case(true, 4, 31, 30);   // future-in-window
    const double dCtl = run_case(false, 4, 20, 63);// far, NO window

    std::printf("A far-outside-window   delta = %.3e (want ~0)\n", dA);
    std::printf("B inside-window-past   delta = %.3e (want > 0)\n", dB);
    std::printf("C future-inside-window delta = %.3e (want ~0 if causal combines)\n", dC);
    std::printf("D control full-attn    delta = %.3e (want > 0)\n", dCtl);

    bool window_binds = dA < 1e-5 && dB > 1e-3 && dCtl > 1e-3;
    bool causal_ok = dC < 1e-5;
    std::printf("\nwindow binds in forward_batched: %s\n",
                window_binds ? "YES" : "NO");
    std::printf("causal mask combines with window: %s\n",
                causal_ok ? "YES" : "NO -- FUTURE LEAK");
    return (window_binds && causal_ok) ? 0 : 1;
}
