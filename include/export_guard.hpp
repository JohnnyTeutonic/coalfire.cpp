#pragma once
// Exports write the llama architecture, which has no FFN biases (GGUF llama, HF LlamaForCausalLM with mlp_bias
// false). A llama-mode model whose FFN biases are not exactly zero computes something no export can describe:
// the CPU forward adds them, the exported file and the CUDA SwiGLU do not. Until 3 Nov 2026 the weight
// initialiser set them to a frozen 0.01 in llama mode, so a CPU-trained model's export silently differed from it
// (paperkiln docs/receipts/c2_coalfire_parity_20261103). Refuse such an export instead of writing a different model.
#include <cmath>
#include <iostream>
#include <string>

#include "feed_forward.hpp"
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

}  // namespace export_guard
