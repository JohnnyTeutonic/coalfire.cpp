// Grouped-query attention in MultiHeadAttention (3 Nov 2026), on the CPU
// training path (forward_batched + the exact backward):
//
//  1. forward against an independent float64 reference written here from the
//     definition (adjacent-pairing RoPE, causal softmax, query head h reading
//     KV head h / (num_heads / num_kv_heads)), for GQA (4 heads, 2 and 1 KV
//     heads) and for multi-head attention (4 KV heads);
//  2. the analytic gradients of every weight, bias and the input against
//     central differences of the scalar loss L = sum(out .* G);
//  3. a save/load round trip reproduces the forward bit for bit.
//
// Prints the maximum errors; exit 1 when any exceeds its bound.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "attention.hpp"
#include "config.hpp"

namespace {

constexpr size_t kHidden = 16, kHeads = 4, kBatch = 2, kSeq = 5, kMaxSeq = 8;

float val(uint32_t seed, size_t i) {  // a deterministic value in [-0.5, 0.5)
    uint32_t h = seed * 0x9E3779B1u ^ static_cast<uint32_t>(i) * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return static_cast<float>(static_cast<int32_t>(h >> 8) - (1 << 23)) / static_cast<float>(1 << 24);
}

void fill(Matrix& m, uint32_t seed, float scale) {
    for (size_t i = 0; i < m.rows() * m.cols(); ++i) m.data()[i] = scale * val(seed, i);
}
void fill(Vector& v, uint32_t seed, float scale) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = scale * val(seed, i);
}

std::unique_ptr<MultiHeadAttention> make(size_t kv_heads) {
    auto ap = std::make_unique<MultiHeadAttention>(kHidden, kHeads, kHidden / kHeads, 0.0f, /*flash*/ false, /*rope*/ true,
                         /*sliding*/ false, 0, /*gqa*/ kv_heads != kHeads, kv_heads, kMaxSeq,
                         /*fp16*/ false, /*fused*/ false);
    auto& p = ap->parameters();
    fill(p.query_weights, 11, 0.6f);
    fill(p.key_weights, 12, 0.6f);
    fill(p.value_weights, 13, 0.6f);
    fill(p.output_weights, 14, 0.6f);
    fill(p.query_bias, 15, 0.2f);
    fill(p.key_bias, 16, 0.2f);
    fill(p.value_bias, 17, 0.2f);
    fill(p.output_bias, 18, 0.2f);
    return ap;
}

// float64 reference forward from the definition.
std::vector<double> reference(const MultiHeadAttention& a, const Matrix& X, size_t kv_heads) {
    const auto& p = a.parameters();
    const size_t hd = kHidden / kHeads, kvd = kv_heads * hd, group = kHeads / kv_heads, T = X.rows();
    auto proj = [&](const Matrix& W, const Vector& b, size_t cols) {
        std::vector<double> y(T * cols);
        for (size_t r = 0; r < T; ++r)
            for (size_t c = 0; c < cols; ++c) {
                double s = b[c];
                for (size_t i = 0; i < kHidden; ++i) s += double(X(r, i)) * W(i, c);
                y[r * cols + c] = s;
            }
        return y;
    };
    auto rope = [&](std::vector<double>& m, size_t cols) {
        for (size_t r = 0; r < T; ++r) {
            const double pos = double(r % kSeq);
            for (size_t h = 0; h < cols / hd; ++h)
                for (size_t i = 0; i < hd / 2; ++i) {
                    const double th = pos * std::pow(10000.0, -2.0 * double(i) / double(hd));
                    double& x0 = m[r * cols + h * hd + 2 * i];
                    double& x1 = m[r * cols + h * hd + 2 * i + 1];
                    const double a0 = x0, a1 = x1;
                    x0 = a0 * std::cos(th) - a1 * std::sin(th);
                    x1 = a0 * std::sin(th) + a1 * std::cos(th);
                }
        }
    };
    std::vector<double> q = proj(p.query_weights, p.query_bias, kHidden);
    std::vector<double> k = proj(p.key_weights, p.key_bias, kvd);
    std::vector<double> v = proj(p.value_weights, p.value_bias, kvd);
    rope(q, kHidden);
    rope(k, kvd);
    std::vector<double> att(T * kHidden, 0.0);
    for (size_t b = 0; b < kBatch; ++b)
        for (size_t h = 0; h < kHeads; ++h) {
            const size_t g = h / group;
            for (size_t i = 0; i < kSeq; ++i) {
                std::vector<double> s(i + 1);
                double mx = -1e300, z = 0.0;
                for (size_t j = 0; j <= i; ++j) {
                    double d = 0.0;
                    for (size_t e = 0; e < hd; ++e)
                        d += q[(b * kSeq + i) * kHidden + h * hd + e] * k[(b * kSeq + j) * kvd + g * hd + e];
                    s[j] = d / std::sqrt(double(hd));
                    mx = std::max(mx, s[j]);
                }
                for (auto& x : s) z += (x = std::exp(x - mx));
                for (size_t e = 0; e < hd; ++e) {
                    double o = 0.0;
                    for (size_t j = 0; j <= i; ++j) o += s[j] / z * v[(b * kSeq + j) * kvd + g * hd + e];
                    att[(b * kSeq + i) * kHidden + h * hd + e] = o;
                }
            }
        }
    std::vector<double> out(T * kHidden);
    for (size_t r = 0; r < T; ++r)
        for (size_t c = 0; c < kHidden; ++c) {
            double s = p.output_bias[c];
            for (size_t i = 0; i < kHidden; ++i) s += att[r * kHidden + i] * p.output_weights(i, c);
            out[r * kHidden + c] = s;
        }
    return out;
}

double loss(MultiHeadAttention& a, const Matrix& X, const Matrix& G) {
    Matrix out = a.forward_batched(X, AttentionMask(), kBatch, kSeq);
    double s = 0.0;
    for (size_t i = 0; i < out.rows() * out.cols(); ++i) s += double(out.data()[i]) * G.data()[i];
    return s;
}

int failures = 0;
void report(const char* what, double err, double bound) {
    const bool ok = std::isfinite(err) && err <= bound;
    std::printf("  %-44s %.3e  (bound %.0e)  %s\n", what, err, bound, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

void run(size_t kv_heads) {
    std::printf("num_heads %zu, num_kv_heads %zu\n", kHeads, kv_heads);
    std::unique_ptr<MultiHeadAttention> owner = make(kv_heads);
    MultiHeadAttention& a = *owner;
    const size_t kvd = kv_heads * (kHidden / kHeads);
    const auto& p0 = a.parameters();
    if (a.kv_heads() != kv_heads || p0.key_weights.cols() != kvd || p0.value_weights.cols() != kvd ||
        p0.key_bias.size() != kvd || p0.value_bias.size() != kvd) {
        std::printf("  K/V shape: %zu x %zu, expected %zu x %zu  FAIL\n", p0.key_weights.rows(),
                    p0.key_weights.cols(), kHidden, kvd);
        ++failures;
        return;
    }
    Matrix X(kBatch * kSeq, kHidden), G(kBatch * kSeq, kHidden);
    fill(X, 21, 1.0f);
    fill(G, 22, 1.0f);

    // 1. forward against the float64 reference
    Matrix out = a.forward_batched(X, AttentionMask(), kBatch, kSeq);
    const std::vector<double> ref = reference(a, X, kv_heads);
    double fwd = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) fwd = std::max(fwd, std::fabs(out.data()[i] - ref[i]));
    report("forward max |coalfire - float64 reference|", fwd, 1e-5);
    {  // the single-sequence forward() (generation path), causal mask, against forward_batched
        Matrix X1(kSeq, kHidden);
        for (size_t i = 0; i < kSeq * kHidden; ++i) X1.data()[i] = X.data()[i];
        AttentionMask causal;
        causal.mask = Matrix(kSeq, kSeq, 0.0f);
        for (size_t i = 0; i < kSeq; ++i)
            for (size_t j = i + 1; j < kSeq; ++j) causal.mask(i, j) = -1e9f;
        const Matrix o_single = a.forward(X1, causal, std::nullopt);
        const Matrix o_batched = a.forward_batched(X1, AttentionMask(), 1, kSeq);
        double d = 0.0;
        for (size_t i = 0; i < kSeq * kHidden; ++i)
            d = std::max(d, double(std::fabs(o_single.data()[i] - o_batched.data()[i])));
        report("forward() max |single-sequence - batched|", d, 1e-5);
    }

    // 2. analytic gradients (backward averages weight/bias grads over rows; no
    //    clipping at this threshold) against central differences
    TransformerConfig cfg;
    cfg.gradient_clip_threshold = 1e30f;
    a.forward_batched(X, AttentionMask(), kBatch, kSeq);
    const Matrix dX = a.backward(G, X, Matrix(), cfg);
    const auto grads = a.param_gradients();
    const double rows = double(X.rows());
    const float eps = 4e-3f;
    // normwise: max |analytic - numeric| / max |numeric| over the tensor
    auto check = [&](const char* what, size_t n, const std::function<float&(size_t)>& at,
                     const std::function<double(size_t)>& analytic) {
        double worst = 0.0, scale = 0.0;
        for (size_t i = 0; i < n; ++i) {
            float& w = at(i);
            const float w0 = w;
            w = w0 + eps;
            const double lp = loss(a, X, G);
            w = w0 - eps;
            const double lm = loss(a, X, G);
            w = w0;
            const double num = (lp - lm) / (2.0 * double(eps));
            worst = std::max(worst, std::fabs(analytic(i) - num));
            scale = std::max(scale, std::fabs(num));
        }
        report(what, worst / std::max(scale, 1e-12), 2e-3);
    };
    auto& p = a.parameters();
    auto mat = [&](const char* what, Matrix& W, const Matrix& dW) {
        check(what, W.rows() * W.cols(), [&](size_t i) -> float& { return W.data()[i]; },
              [&](size_t i) { return rows * dW.data()[i]; });
    };
    auto vec = [&](const char* what, Vector& b, const Vector& db) {
        check(what, b.size(), [&](size_t i) -> float& { return b[i]; }, [&](size_t i) { return rows * db[i]; });
    };
    mat("dL/dW_q   rel. error", p.query_weights, grads.query_grad);
    mat("dL/dW_k   rel. error", p.key_weights, grads.key_grad);
    mat("dL/dW_v   rel. error", p.value_weights, grads.value_grad);
    mat("dL/dW_o   rel. error", p.output_weights, grads.output_grad);
    vec("dL/db_q   rel. error", p.query_bias, grads.query_bias_grad);
    vec("dL/db_k   rel. error", p.key_bias, grads.key_bias_grad);
    vec("dL/db_v   rel. error", p.value_bias, grads.value_bias_grad);
    vec("dL/db_o   rel. error", p.output_bias, grads.output_bias_grad);
    Matrix Xp = X;
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < Xp.rows() * Xp.cols(); ++i) {
        const float x0 = Xp.data()[i];
        Xp.data()[i] = x0 + eps;
        const double lp = loss(a, Xp, G);
        Xp.data()[i] = x0 - eps;
        const double lm = loss(a, Xp, G);
        Xp.data()[i] = x0;
        const double num = (lp - lm) / (2.0 * double(eps));
        worst = std::max(worst, std::fabs(dX.data()[i] - num));
        scale = std::max(scale, std::fabs(num));
    }
    report("dL/dX     rel. error", worst / std::max(scale, 1e-12), 2e-3);

    // 3. save/load round trip
    std::stringstream ss;
    a.save(ss);
    auto b = MultiHeadAttention::load(ss, cfg);
    const Matrix o1 = a.forward_batched(X, AttentionMask(), kBatch, kSeq);
    const Matrix o2 = b->forward_batched(X, AttentionMask(), kBatch, kSeq);
    size_t differ = (b->kv_heads() == kv_heads) ? 0 : 1;
    for (size_t i = 0; i < o1.rows() * o1.cols(); ++i) differ += o1.data()[i] != o2.data()[i];
    report("save/load: output values that differ", double(differ), 0.0);
}

}  // namespace

int main() {
    run(2);  // GQA, groups of two query heads
    run(1);  // GQA, one KV head (multi-query)
    run(4);  // multi-head attention
    if (failures) {
        std::printf("\n[FAIL] %d checks failed\n", failures);
        return 1;
    }
    std::printf("\n[PASS] grouped-query attention: forward, gradients and save/load\n");
    return 0;
}
