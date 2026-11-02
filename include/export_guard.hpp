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

}  // namespace export_guard
