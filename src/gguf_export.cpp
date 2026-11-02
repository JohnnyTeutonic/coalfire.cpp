/**
 * @file gguf_export.cpp
 * @brief Implementation of GGUF exporter for transformer_cpp models
 */

#include "../include/gguf_export.hpp"
#include "../include/transformer.hpp"
#include "../include/tiktoken_tokenizer.hpp"
#include "../include/attention.hpp"
#include "../include/feed_forward.hpp"
#include "../include/layer_norm.hpp"
#include "../include/embeddings.hpp"
#include "../include/lm_head.hpp"

#include <iostream>
#include <cstring>
#include <algorithm>
#include <memory>

namespace gguf_export {

// ============================================================================
// Utility functions
// ============================================================================

std::string convert_tensor_name(const std::string& internal_name, int layer_idx) {
    // Map transformer_cpp names to GGUF/llama.cpp convention
    // GGUF uses: blk.{n}.{component}.weight
    
    if (internal_name == "token_embedding" || internal_name == "embed_tokens") {
        return "token_embd.weight";
    }
    if (internal_name == "lm_head" || internal_name == "output") {
        return "output.weight";
    }
    if (internal_name == "final_norm" || internal_name == "norm") {
        return "output_norm.weight";
    }
    
    // Layer-specific tensors
    std::string prefix = "blk." + std::to_string(layer_idx) + ".";
    
    if (internal_name == "q_proj" || internal_name == "wq") {
        return prefix + "attn_q.weight";
    }
    if (internal_name == "k_proj" || internal_name == "wk") {
        return prefix + "attn_k.weight";
    }
    if (internal_name == "v_proj" || internal_name == "wv") {
        return prefix + "attn_v.weight";
    }
    if (internal_name == "o_proj" || internal_name == "wo") {
        return prefix + "attn_output.weight";
    }
    if (internal_name == "gate_proj" || internal_name == "w1") {
        return prefix + "ffn_gate.weight";
    }
    if (internal_name == "up_proj" || internal_name == "w3") {
        return prefix + "ffn_up.weight";
    }
    if (internal_name == "down_proj" || internal_name == "w2") {
        return prefix + "ffn_down.weight";
    }
    if (internal_name == "attention_norm" || internal_name == "input_layernorm") {
        return prefix + "attn_norm.weight";
    }
    if (internal_name == "ffn_norm" || internal_name == "post_attention_layernorm") {
        return prefix + "ffn_norm.weight";
    }
    
    // Fallback: return as-is with layer prefix if applicable
    if (layer_idx >= 0) {
        return prefix + internal_name + ".weight";
    }
    return internal_name + ".weight";
}

// The byte writer (GGUFWriter) is the shared header gguf_writer.hpp.

// ============================================================================
// Main export function
// ============================================================================

bool export_to_gguf(
    const Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::string& output_path,
    const GGUFExportConfig& config
) {
    std::cout << "GGUF Export: Starting export to " << output_path << std::endl;

    GGUFWriter writer(output_path);

    // Get model config
    const auto& model_config = transformer.getConfig();
    const size_t head_dim = model_config.hidden_size / model_config.num_heads;

    // ========== Write metadata ==========

    // General metadata
    writer.write_metadata_string("general.architecture", config.architecture);
    writer.write_metadata_string("general.name", config.model_name);
    writer.write_metadata_uint32("general.alignment", GGUF_DEFAULT_ALIGNMENT);

    // Model architecture metadata (llama format)
    std::string arch = config.architecture;
    writer.write_metadata_uint32(arch + ".context_length", model_config.max_seq_length);
    writer.write_metadata_uint32(arch + ".embedding_length", model_config.hidden_size);
    writer.write_metadata_uint32(arch + ".block_count", model_config.num_layers);
    writer.write_metadata_uint32(arch + ".feed_forward_length", model_config.intermediate_size);
    writer.write_metadata_uint32(arch + ".attention.head_count", model_config.num_heads);
    writer.write_metadata_uint32(arch + ".attention.head_count_kv", model_config.num_kv_heads);
    writer.write_metadata_float32(arch + ".attention.layer_norm_rms_epsilon", model_config.layer_norm_epsilon);
    writer.write_metadata_uint32(arch + ".vocab_size", model_config.vocab_size);

    // RoPE metadata
    if (model_config.use_rope) {
        writer.write_metadata_float32(arch + ".rope.freq_base", 10000.0f);
        writer.write_metadata_uint32(arch + ".rope.dimension_count",
                                     static_cast<uint32_t>(head_dim));
    }

    // Tokenizer metadata. transformer_cpp's tokenizer is WORD-LEVEL
    // (lowercased whole-word lookup with <unk> fallback), not BPE or
    // SentencePiece. "word" is a custom model string; tinyllama.cpp carries a
    // matching word-level tokenizer mode.
    if (config.include_tokenizer) {
        writer.write_metadata_string("tokenizer.ggml.model", "word");

        std::vector<std::string> tokens;
        std::vector<float> scores;

        // Export only the vocabulary the MODEL was trained with. The
        // tokenizer's vocab can exceed the model's (the trainer caps
        // vocab_size); ids >= model vocab were mapped to <unk> during
        // training, so exporting them would desynchronize ids from rows
        // of the embedding matrix.
        size_t vocab_size = std::min(tokenizer.vocab_size(),
                                     model_config.vocab_size);
        tokens.reserve(vocab_size);
        scores.reserve(vocab_size);

        for (size_t i = 0; i < vocab_size; ++i) {
            std::string token = tokenizer.decode({static_cast<int>(i)});
            tokens.push_back(token);
            scores.push_back(0.0f);
        }

        writer.write_metadata_string_array("tokenizer.ggml.tokens", tokens);
        writer.write_metadata_float_array("tokenizer.ggml.scores", scores);

        // Special tokens. Word-level vocab layout: <unk>=0, '|' (document
        // separator)=1, real words from id 2. bos/eos point at the separator:
        // WORD_LEVEL encode never injects them, and stopping at a document
        // boundary is right. (bos=2/eos=3 previously branded "the"/"and" as
        // specials: decode dropped them and generation stopped at the first
        // "and".) If the corpus carried an explicit end-of-document token
        // (TinyStories-Instruct's <|endoftext|>), prefer it for EOS so
        // generation stops at story boundaries.
        uint32_t eos_id = 1;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (tokens[i] == "<|endoftext|>") {
                eos_id = static_cast<uint32_t>(i);
                break;
            }
        }
        writer.write_metadata_uint32("tokenizer.ggml.bos_token_id", 1);
        writer.write_metadata_uint32("tokenizer.ggml.eos_token_id", eos_id);
        writer.write_metadata_uint32("tokenizer.ggml.unknown_token_id", 0);
        writer.write_metadata_uint32("tokenizer.ggml.padding_token_id", 0);
    }

    // ========== Collect and write tensors ==========

    // The weights are written as the F32 they are trained in. Any other type
    // would label float bytes as that type (an unreadable file), so refuse it
    // before anything is written.
    GGMLType weight_type = config.weight_type;
    if (weight_type != GGMLType::F32) {
        std::cerr << "GGUF Export Error: weight_type " << static_cast<uint32_t>(weight_type)
                  << " is not supported; export_to_gguf writes F32 only" << std::endl;
        return false;
    }

    // Keep-alive storage for transposed copies; must outlive finalize().
    std::vector<std::unique_ptr<std::vector<float>>> owned;

    // transformer_cpp stores projection weights row-major [in, out] and
    // computes y = x @ W. llama.cpp-family engines (tinyllama.cpp) store
    // row-major [out, in] and compute y[o] = sum_i W[o][i] x[i]. So every
    // projection is transposed on export. GGUF dims are written fastest-first:
    // ne = {in_dim, out_dim}.
    auto add_transposed = [&](const Matrix& w, const std::string& name) {
        auto buf = std::make_unique<std::vector<float>>(w.rows() * w.cols());
        const size_t R = w.rows(), C = w.cols();
        for (size_t i = 0; i < R; ++i) {
            for (size_t j = 0; j < C; ++j) {
                (*buf)[j * R + i] = w(i, j);  // [out=C][in=R]
            }
        }
        TensorExportInfo info;
        info.name = name;
        info.shape = {static_cast<uint64_t>(R), static_cast<uint64_t>(C)};  // ne0=in, ne1=out
        info.type = weight_type;
        info.data = buf->data();
        info.num_elements = R * C;
        owned.push_back(std::move(buf));
        writer.add_tensor(info);
    };

    // 1D norm weights: gamma stored as [1 x hidden]
    auto add_norm = [&](const LayerNorm* norm, const std::string& name) {
        if (!norm) return;
        const Matrix& gamma = norm->get_gamma();
        TensorExportInfo info;
        info.name = name;
        info.shape = {static_cast<uint64_t>(gamma.cols())};
        info.type = weight_type;
        info.data = gamma.data();
        info.num_elements = gamma.cols();
        writer.add_tensor(info);
    };

    // Token embeddings: stored [vocab, hidden] row-major, which is already
    // the llama layout (ne = {hidden, vocab}). Export the first
    // model_config.vocab_size rows only, matching the exported vocab.
    const auto* embedding = transformer.getTokenEmbedding();
    if (embedding) {
        const Matrix& emb_weights = embedding->getWeights();
        TensorExportInfo emb_info;
        emb_info.name = "token_embd.weight";
        emb_info.shape = {static_cast<uint64_t>(emb_weights.cols()),
                          static_cast<uint64_t>(emb_weights.rows())};
        emb_info.type = weight_type;
        emb_info.data = emb_weights.data();
        emb_info.num_elements = emb_weights.rows() * emb_weights.cols();
        writer.add_tensor(emb_info);
    }

    // Transformer layers
    const auto& layers = transformer.getLayers();
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto& layer = layers[layer_idx];
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";

        const auto* attention = layer->getAttention();
        if (attention) {
            const auto& ap = attention->parameters();
            add_transposed(ap.query_weights, prefix + "attn_q.weight");
            add_transposed(ap.key_weights, prefix + "attn_k.weight");
            add_transposed(ap.value_weights, prefix + "attn_v.weight");
            add_transposed(ap.output_weights, prefix + "attn_output.weight");
        }

        add_norm(layer->getLayerNorm(), prefix + "attn_norm.weight");

        const auto* ffn = layer->getFeedForward();
        if (ffn) {
            const auto& fp = ffn->parameters();
            add_transposed(fp.gate_proj_weights, prefix + "ffn_gate.weight");
            add_transposed(fp.up_proj_weights, prefix + "ffn_up.weight");
            add_transposed(fp.down_proj_weights, prefix + "ffn_down.weight");
        }

        add_norm(layer->getFfnLayerNorm(), prefix + "ffn_norm.weight");
    }

    // Final layer norm
    add_norm(transformer.getFinalLayerNorm(), "output_norm.weight");

    // LM Head: stored [hidden, vocab] (logits = h @ W) -> transpose to
    // llama's output.weight layout [vocab, hidden] (ne = {hidden, vocab})
    const auto* lm_head = transformer.getLMHead();
    if (lm_head && !config.tie_word_embeddings) {
        add_transposed(lm_head->getWeights(), "output.weight");
    }

    // Finalize and write (owned transposed buffers stay alive until here)
    return writer.finalize();
}

} // namespace gguf_export
