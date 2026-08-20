#include "../include/transformer.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>

TransformerLayer::TransformerLayer(const TransformerConfig& config_, size_t idx)
    : config(config_), layer_idx(idx) {
    // Initialize components
    std::cout << "Initializing TransformerLayer " << idx << " with config:" << std::endl;
    std::cout << "- use_gqa: " << (config.use_gqa ? "true" : "false") << std::endl;
    std::cout << "- num_kv_heads: " << config.num_kv_heads << std::endl;
    std::cout << "- hidden_size: " << config.hidden_size << std::endl;
    std::cout << "- intermediate_size: " << config.intermediate_size << std::endl;
    std::cout << "- layer_norm_epsilon: " << config.layer_norm_epsilon << std::endl;
    std::cout << "- dropout_rate: " << config.dropout_rate << std::endl;

    self_attention = std::make_unique<MultiHeadAttention>(
        config.hidden_size, config.num_heads, config.head_dim, config.dropout_rate,
        config.use_flash_attention, config.use_rope, config.use_sliding_window, config.window_size,
        config.use_gqa, config.num_kv_heads, config.max_seq_length, config.use_fp16,
        true);  // use_fused_attention = true for GPU acceleration
        // ⚠️ The fused kernel (fused_attention_kernels.cu) implements NO
        // RoPE — it must never carry a RoPE-family model's forward. It is
        // reachable only via single-sequence forward() without kv_cache;
        // training uses forward_batched (which applies RoPE on host).
        // Also note MultiHeadAttention::load does not restore this flag.

    // use_rms_norm selects RMSNorm (no mean subtraction, no beta) to match
    // llama.cpp-family inference engines; false gives classic LayerNorm.
    // (Decoupled from the legacy llama_mode flag by arch::ArchitectureSpec.)
    attention_ln = std::make_unique<LayerNorm>(config.hidden_size, config.layer_norm_epsilon,
                                               config.use_rms_norm);
    // FFN sublayer: dense FeedForward, or MixtureOfExperts when the config
    // enables it. Everything downstream already branched on moe_layer
    // (backward in transformer.cpp:131, update_parameters/save/load in
    // transformer.hpp) -- construction and the forward call sites were the
    // only missing wires (found + closed 2026-07-31).
    if (config.moe.enabled) {
        std::cout << "- MoE: " << config.moe.num_experts << " experts, top-"
                  << config.moe.top_k << ", aux "
                  << config.moe.aux_loss_coefficient << std::endl;
        moe_layer = std::make_unique<MixtureOfExperts>(
            config.moe.num_experts, config.moe.top_k, config.hidden_size,
            config.intermediate_size, config.moe.aux_loss_coefficient);
    } else {
        feed_forward = std::make_unique<FeedForward>(config.hidden_size,
                                                     config.intermediate_size);
    }
    ffn_ln = std::make_unique<LayerNorm>(config.hidden_size, config.layer_norm_epsilon,
                                         config.use_rms_norm);

    // Initialize dropout layers
    attention_dropout = std::make_unique<Dropout>(config.dropout_rate);
    ffn_dropout = std::make_unique<Dropout>(config.dropout_rate);
}

Matrix TransformerLayer::forward(const Matrix& input, const AttentionMask& mask,
                                 const std::optional<KVCache>& kv_cache) {
    // Layer norm before attention
    Matrix normalized = attention_ln->forward(input);
    
    // Cache the normalized input for attention backward pass
    std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(attn_key, normalized);

    // Self attention with dropout
    Matrix attention_output = self_attention->forward(normalized, mask, kv_cache);
    if (training) {
        attention_output = attention_dropout->forward(attention_output);
    }
    
    // First residual connection
    Matrix residual = attention_output + input;
    
    // Feed forward network with dropout
    // Cache the residual before ffn layer norm for backward pass
    std::string residual_key = "residual_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(residual_key, residual);
    
    Matrix ffn_normalized = ffn_ln->forward(residual);
    
    // Cache normalized FFN input for backward pass
    std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(ffn_key, ffn_normalized);
    
    Matrix ffn_output = moe_layer ? moe_layer->forward(ffn_normalized)
                                  : feed_forward->forward(ffn_normalized);
    if (training) {
        ffn_output = ffn_dropout->forward(ffn_output);
    }
    
    // Second residual connection
    Matrix output = ffn_output + residual;
    
    return output;
}

// Batched forward - uses O(batch × seq²) attention
Matrix TransformerLayer::forward_batched(const Matrix& input, const AttentionMask& mask,
                                         size_t batch_size, size_t seq_len) {
    const bool phase_timing = (std::getenv("TCPP_PHASE_TIMING") != nullptr);
    auto lt0 = std::chrono::high_resolution_clock::now();

    // Layer norm before attention
    Matrix normalized = attention_ln->forward(input);

    // Cache the normalized input for attention backward pass
    std::string attn_key = "attn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(attn_key, normalized);
    auto lt1 = std::chrono::high_resolution_clock::now();

    // BATCHED self attention - O(batch × seq²) instead of O((batch×seq)²)
    Matrix attention_output = self_attention->forward_batched(normalized, mask, batch_size, seq_len);
    if (training) {
        attention_output = attention_dropout->forward(attention_output);
    }
    auto lt2 = std::chrono::high_resolution_clock::now();

    // First residual connection
    Matrix residual = attention_output + input;

    // Feed forward network with dropout
    std::string residual_key = "residual_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(residual_key, residual);

    Matrix ffn_normalized = ffn_ln->forward(residual);

    std::string ffn_key = "ffn_norm_" + std::to_string(layer_idx);
    GradientCheckpoint::cache_activation(ffn_key, ffn_normalized);
    auto lt3 = std::chrono::high_resolution_clock::now();

    Matrix ffn_output = moe_layer ? moe_layer->forward(ffn_normalized)
                                  : feed_forward->forward(ffn_normalized);
    if (training) {
        ffn_output = ffn_dropout->forward(ffn_output);
    }

    // Second residual connection
    Matrix output = ffn_output + residual;

    // Sub-op activation trace (TCPP_LAYER_TRACE): CPU and CUDA builds
    // produce ~200x different layer outputs on identical weights; the first
    // diverging sub-op L2 between builds names the broken op.
    if (std::getenv("TCPP_LAYER_TRACE") != nullptr) {
        auto l2 = [](const Matrix& m) {
            double s = 0.0;
            for (size_t i = 0; i < m.rows(); ++i)
                for (size_t j = 0; j < m.cols(); ++j)
                    s += double(m(i, j)) * m(i, j);
            return std::sqrt(s);
        };
        std::cout << "[SUBOP] L" << layer_idx
                  << " in=" << l2(input)
                  << " ln1=" << l2(normalized)
                  << " attn=" << l2(attention_output)
                  << " res1=" << l2(residual)
                  << " ln2=" << l2(ffn_normalized)
                  << " ffn=" << l2(ffn_output)
                  << " out=" << l2(output) << std::endl;
    }

    if (phase_timing) {
        auto lt4 = std::chrono::high_resolution_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
        };
        std::cout << "[LAYER_TIMING] L" << layer_idx
                  << " ln1+cache=" << ms(lt0, lt1)
                  << "ms attn=" << ms(lt1, lt2)
                  << "ms ln2+cache=" << ms(lt2, lt3)
                  << "ms ffn=" << ms(lt3, lt4) << "ms" << std::endl;
    }

    return output;
}