#pragma once
// Exports write the llama architecture, which has no FFN biases (GGUF llama, HF LlamaForCausalLM with mlp_bias
// false). A llama-mode model whose FFN biases are not exactly zero computes something no export can describe:
// the CPU forward adds them, the exported file and the CUDA SwiGLU do not. Until 3 Nov 2026 the weight
// initialiser set them to a frozen 0.01 in llama mode, so a CPU-trained model's export silently differed from it
// (paperkiln docs/receipts/c2_coalfire_parity_20261103). Refuse such an export instead of writing a different model.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include "embeddings.hpp"
#include "feed_forward.hpp"
#include "lm_head.hpp"
#include "transformer.hpp"

namespace export_guard {

inline bool llama_ffn_biases_zero(const Transformer& transformer, const std::string& what) {
    if (transformer.getConfig().use_biases) return true;  // a biased architecture: not the llama export contract
    float worst = 0.0f;
    for (const auto& layer : transformer.getLayers()) {
        const FeedForward* ff = layer->getFeedForward();
        if (!ff) continue;
        const auto& p = ff->parameters();
        for (const FloatVector* b : {&p.gate_proj_bias, &p.up_proj_bias, &p.down_proj_bias})
            for (size_t i = 0; i < b->size(); ++i) worst = std::max(worst, std::fabs((*b)[i]));
    }
    if (worst == 0.0f) return true;
    std::cerr << what << ": refusing to export a llama-mode model whose FFN biases are not zero (max |bias| " << worst
              << "). The llama export has no FFN biases, so the file would not be this model. This happens to"
                 " checkpoints trained on the CPU before the 3 Nov 2026 initialiser fix; retrain, or zero the"
                 " biases knowingly." << std::endl;
    return false;
}

// A coalfire model always has its own trained LanguageModelHead; nothing ties it to the token embedding. An
// exporter's tie_word_embeddings omits the head, and a consumer then uses the embedding as the head, which is a
// different model unless the two are already the same. Allow the flag only when tying is lossless: the head weight
// [hidden, vocab] is exactly the transposed embedding [vocab, hidden] and the head bias is zero.
inline bool tied_head_lossless(const Transformer& transformer, bool tie_word_embeddings, const std::string& what) {
    if (!tie_word_embeddings) return true;
    const LanguageModelHead* head = transformer.getLMHead();
    if (!head) return true;  // no head to drop
    const TokenEmbedding* embedding = transformer.getTokenEmbedding();
    std::string why;
    if (!embedding) {
        why = "the model has no token embedding to tie the head to";
    } else {
        const Matrix& w = head->getWeights();
        const Matrix& e = embedding->getWeights();
        if (w.rows() != e.cols() || w.cols() != e.rows()) {
            why = "the head is [" + std::to_string(w.rows()) + ", " + std::to_string(w.cols()) +
                  "] and the embedding [" + std::to_string(e.rows()) + ", " + std::to_string(e.cols()) +
                  "], not its transpose";
        } else {
            size_t differ = 0;
            float worst = 0.0f;
            for (size_t h = 0; h < w.rows(); ++h)
                for (size_t v = 0; v < w.cols(); ++v)
                    if (w(h, v) != e(v, h)) {
                        ++differ;
                        worst = std::max(worst, std::fabs(w(h, v) - e(v, h)));
                    }
            float worst_bias = 0.0f;
            const Vector& b = head->get_bias();
            for (size_t i = 0; i < b.size(); ++i) worst_bias = std::max(worst_bias, std::fabs(b[i]));
            if (differ)
                why = std::to_string(differ) + " of " + std::to_string(w.rows() * w.cols()) +
                      " head weights differ from the transposed embedding (max |difference| " +
                      std::to_string(worst) + ")";
            else if (worst_bias != 0.0f)
                why = "the head bias is not zero (max |bias| " + std::to_string(worst_bias) + ")";
        }
    }
    if (why.empty()) return true;
    std::cerr << what << ": refusing tie_word_embeddings: " << why << ". The model has its own trained lm_head; a"
              << " tied export omits it, and a consumer would use the embedding as the head, which is not this model."
                 " Export without tie_word_embeddings." << std::endl;
    return false;
}

// The number of K/V heads the model's attention tensors hold: num_heads for full multi-head attention, num_kv_heads
// under grouped-query attention (K and V [hidden, num_kv_heads * head_dim]; query head h reads KV head
// h / (num_heads / num_kv_heads), the grouping GGUF llama and HF Llama use). Read from the tensors, not the config,
// so an export always declares what it writes. Returns 0 (refuse the export) when the layers disagree or the K/V
// width is not a whole number of heads dividing num_heads.
inline size_t kv_head_count(const Transformer& transformer, const std::string& what) {
    const auto& cfg = transformer.getConfig();
    const size_t n_heads = cfg.num_heads;
    const size_t head_dim = n_heads ? cfg.hidden_size / n_heads : 0;
    size_t kv = 0;
    std::string why;
    for (const auto& layer : transformer.getLayers()) {
        const MultiHeadAttention* attn = layer->getAttention();
        if (!attn) continue;
        const size_t k_cols = attn->parameters().key_weights.cols();
        const size_t v_cols = attn->parameters().value_weights.cols();
        if (head_dim == 0 || k_cols != v_cols || k_cols % head_dim != 0 || k_cols == 0 ||
            n_heads % (k_cols / head_dim) != 0) {
            why = "K/V widths " + std::to_string(k_cols) + "/" + std::to_string(v_cols) +
                  " are not a whole number of heads (head_dim " + std::to_string(head_dim) + ") dividing " +
                  std::to_string(n_heads) + " heads";
            break;
        }
        if (kv != 0 && kv != k_cols / head_dim) {
            why = "the layers hold different numbers of K/V heads";
            break;
        }
        kv = k_cols / head_dim;
    }
    if (why.empty()) return kv ? kv : n_heads;
    std::cerr << what << ": refusing to export: " << why << "." << std::endl;
    return 0;
}

}  // namespace export_guard
