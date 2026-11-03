/**
 * @file safetensors_export.cpp
 * @brief Safetensors exporter. Mirrors the tensor enumeration of
 *        gguf_export.cpp exactly (same names, same [out,in] transposition),
 *        so a model exports identically under both formats.
 */

#include "../include/safetensors_export.hpp"
#include "../include/export_guard.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "../include/transformer.hpp"
#include "../include/tiktoken_tokenizer.hpp"
#include "../include/layer_norm.hpp"

namespace safetensors_export {

namespace {

struct STTensor {
    std::string name;
    std::vector<uint64_t> shape;
    const float* data = nullptr;                     // non-owning view, or...
    std::unique_ptr<std::vector<float>> owned;       // ...owning (transposed copies)
    size_t num_elements = 0;

    const float* ptr() const { return owned ? owned->data() : data; }
};

}  // namespace

bool export_to_safetensors(
    const Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::string& output_path,
    const SafetensorsExportConfig& config
) {
    std::cout << "Safetensors Export: Starting export to " << output_path << std::endl;
    if (!export_guard::llama_ffn_biases_zero(transformer, "Safetensors export")) return false;
    if (!export_guard::tied_head_lossless(transformer, config.tie_word_embeddings, "Safetensors export"))
        return false;
    const size_t n_kv_heads = export_guard::kv_head_count(transformer, "Safetensors export");
    if (n_kv_heads == 0) return false;

    const auto& model_config = transformer.getConfig();
    std::vector<STTensor> tensors;

    // Projection weights are stored row-major [in, out] (y = x @ W); export
    // transposed to [out, in], matching the GGUF exporter and torch's
    // nn.Linear convention.
    auto add_transposed = [&](const Matrix& w, const std::string& name) {
        STTensor t;
        t.name = name;
        const size_t R = w.rows(), C = w.cols();
        t.owned = std::make_unique<std::vector<float>>(R * C);
        for (size_t i = 0; i < R; ++i) {
            for (size_t j = 0; j < C; ++j) {
                (*t.owned)[j * R + i] = w(i, j);     // [out=C][in=R]
            }
        }
        t.shape = {static_cast<uint64_t>(C), static_cast<uint64_t>(R)};
        t.num_elements = R * C;
        tensors.push_back(std::move(t));
    };

    auto add_norm = [&](const LayerNorm* norm, const std::string& name) {
        if (!norm) return;
        const Matrix& gamma = norm->get_gamma();     // [1 x hidden]
        STTensor t;
        t.name = name;
        t.shape = {static_cast<uint64_t>(gamma.cols())};
        t.data = gamma.data();
        t.num_elements = gamma.cols();
        tensors.push_back(std::move(t));
    };

    // Token embeddings: stored [vocab, hidden] row-major; export the first
    // model vocab_size rows only (ids beyond the model vocab were <unk> in
    // training), exactly as the GGUF exporter does.
    const auto* embedding = transformer.getTokenEmbedding();
    if (embedding) {
        const Matrix& emb = embedding->getWeights();
        STTensor t;
        t.name = "token_embd.weight";
        const size_t vocab = std::min(static_cast<size_t>(emb.rows()),
                                      model_config.vocab_size);
        t.shape = {static_cast<uint64_t>(vocab), static_cast<uint64_t>(emb.cols())};
        t.data = emb.data();
        t.num_elements = vocab * emb.cols();
        tensors.push_back(std::move(t));
    }

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

    add_norm(transformer.getFinalLayerNorm(), "output_norm.weight");

    // LM head: stored [hidden, vocab] -> [vocab, hidden]
    const auto* lm_head = transformer.getLMHead();
    if (lm_head && !config.tie_word_embeddings) {
        add_transposed(lm_head->getWeights(), "output.weight");
    }

    // ---- Build the JSON header (offsets are into the data section) ----
    nlohmann::json header;
    header["__metadata__"] = {
        {"format", "transformer_cpp"},
        {"model_name", config.model_name},
        {"architecture", "llama"},
        {"vocab_size", std::to_string(model_config.vocab_size)},
        {"hidden_size", std::to_string(model_config.hidden_size)},
        {"num_layers", std::to_string(model_config.num_layers)},
        {"num_heads", std::to_string(model_config.num_heads)},
        {"intermediate_size", std::to_string(model_config.intermediate_size)},
        {"max_seq_length", std::to_string(model_config.max_seq_length)},
        {"layer_norm_epsilon", std::to_string(model_config.layer_norm_epsilon)},
        {"use_rope", model_config.use_rope ? "true" : "false"},
        {"tokenizer", "word_level"},
    };
    // Grouped-query attention: attn_k / attn_v are [kv_dim, hidden]. Recorded
    // only then, so a multi-head model's header is what it always was.
    if (n_kv_heads != model_config.num_heads)
        header["__metadata__"]["num_kv_heads"] = std::to_string(n_kv_heads);
    uint64_t offset = 0;
    for (const auto& t : tensors) {
        const uint64_t nbytes = static_cast<uint64_t>(t.num_elements) * sizeof(float);
        header[t.name] = {
            {"dtype", "F32"},
            {"shape", t.shape},
            {"data_offsets", {offset, offset + nbytes}},
        };
        offset += nbytes;
    }
    std::string header_str = header.dump();
    // Pad the header with spaces to 8-byte alignment (spec recommendation).
    while (header_str.size() % 8 != 0) header_str.push_back(' ');

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Safetensors Export: cannot open " << output_path << std::endl;
        return false;
    }
    const uint64_t header_len = header_str.size();
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    for (const auto& t : tensors) {
        out.write(reinterpret_cast<const char*>(t.ptr()),
                  static_cast<std::streamsize>(t.num_elements * sizeof(float)));
    }
    out.close();
    if (!out) {
        std::cerr << "Safetensors Export: write failed for " << output_path << std::endl;
        return false;
    }
    std::cout << "Safetensors Export: wrote " << tensors.size() << " tensors ("
              << (offset / (1024.0 * 1024.0)) << " MB data) to "
              << output_path << std::endl;

    // ---- Vocabulary sidecar (safetensors carries no tokenizer) ----
    if (config.write_vocab_sidecar) {
        const size_t vocab_size = std::min(tokenizer.vocab_size(),
                                           model_config.vocab_size);
        nlohmann::json vocab;
        vocab["tokenizer"] = "word_level";
        vocab["special_tokens"] = {{"pad", 0}, {"unk", 1}, {"bos", 2}, {"eos", 3}};
        nlohmann::json id_to_token = nlohmann::json::array();
        for (size_t i = 0; i < vocab_size; ++i) {
            id_to_token.push_back(tokenizer.decode({static_cast<int>(i)}));
        }
        vocab["id_to_token"] = std::move(id_to_token);
        std::ofstream vf(output_path + ".vocab.json");
        vf << vocab.dump();
        std::cout << "Safetensors Export: vocab sidecar (" << vocab_size
                  << " tokens) -> " << output_path << ".vocab.json" << std::endl;
    }
    return true;
}

// ===========================================================================
// HuggingFace LlamaForCausalLM export (with the RoPE permute)
// ===========================================================================

bool export_to_hf(
    const Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::string& out_dir,
    const SafetensorsExportConfig& config
) {
    if (!export_guard::llama_ffn_biases_zero(transformer, "HF export")) return false;
    if (!export_guard::tied_head_lossless(transformer, config.tie_word_embeddings, "HF export")) return false;
    const size_t n_kv_heads = export_guard::kv_head_count(transformer, "HF export");
    if (n_kv_heads == 0) return false;
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    const auto& mc = transformer.getConfig();
    const size_t H = mc.hidden_size;
    const size_t n_heads = mc.num_heads;
    const size_t head_dim = H / n_heads;
    if (head_dim % 2 != 0) {
        std::cerr << "HF export: head_dim must be even for RoPE permute\n";
        return false;
    }

    std::vector<STTensor> tensors;

    // Plain [in,out] -> [out,in] transpose (no permute): for v/o/ffn/embeddings.
    auto add_T = [&](const Matrix& w, const std::string& name) {
        STTensor t; t.name = name;
        const size_t R = w.rows(), C = w.cols();
        t.owned = std::make_unique<std::vector<float>>(R * C);
        for (size_t i = 0; i < R; ++i)
            for (size_t j = 0; j < C; ++j) (*t.owned)[j * R + i] = w(i, j);
        t.shape = {(uint64_t)C, (uint64_t)R};
        t.num_elements = R * C;
        tensors.push_back(std::move(t));
    };

    // q/k: transpose to [out,in] THEN permute the out-rows per head from this
    // codebase's interleaved order [0,1,2,3,...] to HF's split-half order
    // [0,2,4,...,1,3,5,...], so HF rotate_half reproduces interleaved RoPE.
    // k_proj under GQA has C = n_kv_heads * head_dim rows: permute its heads.
    auto add_qk_permuted = [&](const Matrix& w, const std::string& name) {
        STTensor t; t.name = name;
        const size_t R = w.rows(), C = w.cols();     // R=in(hidden), C=out(heads*head_dim)
        t.owned = std::make_unique<std::vector<float>>(R * C);
        for (size_t h = 0; h < C / head_dim; ++h) {
            for (size_t d = 0; d < head_dim; ++d) {
                // HF row d within the head maps to this codebase's src row:
                //   d in [0,head_dim/2) -> 2*d      (even/"cos" lanes)
                //   d in [head_dim/2, head_dim) -> 2*(d-head_dim/2)+1 (odd)
                size_t src_d = (d < head_dim / 2) ? (2 * d)
                                                  : (2 * (d - head_dim / 2) + 1);
                size_t dst_row = h * head_dim + d;
                size_t src_col = h * head_dim + src_d;   // out index in w's cols
                for (size_t i = 0; i < R; ++i)
                    (*t.owned)[dst_row * R + i] = w(i, src_col);
            }
        }
        t.shape = {(uint64_t)C, (uint64_t)R};
        t.num_elements = R * C;
        tensors.push_back(std::move(t));
    };

    auto add_norm = [&](const LayerNorm* norm, const std::string& name) {
        if (!norm) return;
        const Matrix& g = norm->get_gamma();
        STTensor t; t.name = name;
        t.shape = {(uint64_t)g.cols()};
        t.data = g.data(); t.num_elements = g.cols();
        tensors.push_back(std::move(t));
    };

    const auto* embedding = transformer.getTokenEmbedding();
    size_t vocab = mc.vocab_size;
    if (embedding) {
        const Matrix& emb = embedding->getWeights();
        vocab = std::min(static_cast<size_t>(emb.rows()), mc.vocab_size);
        STTensor t; t.name = "model.embed_tokens.weight";
        t.shape = {(uint64_t)vocab, (uint64_t)emb.cols()};
        t.data = emb.data(); t.num_elements = vocab * emb.cols();
        tensors.push_back(std::move(t));
    }

    const auto& layers = transformer.getLayers();
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];
        std::string p = "model.layers." + std::to_string(l) + ".";
        if (const auto* attn = layer->getAttention()) {
            const auto& ap = attn->parameters();
            add_qk_permuted(ap.query_weights, p + "self_attn.q_proj.weight");
            add_qk_permuted(ap.key_weights,   p + "self_attn.k_proj.weight");
            add_T(ap.value_weights,           p + "self_attn.v_proj.weight");
            add_T(ap.output_weights,          p + "self_attn.o_proj.weight");
        }
        add_norm(layer->getLayerNorm(),    p + "input_layernorm.weight");
        if (const auto* ffn = layer->getFeedForward()) {
            const auto& fp = ffn->parameters();
            add_T(fp.gate_proj_weights, p + "mlp.gate_proj.weight");
            add_T(fp.up_proj_weights,   p + "mlp.up_proj.weight");
            add_T(fp.down_proj_weights, p + "mlp.down_proj.weight");
        }
        add_norm(layer->getFfnLayerNorm(), p + "post_attention_layernorm.weight");
    }
    add_norm(transformer.getFinalLayerNorm(), "model.norm.weight");
    const auto* lm_head = transformer.getLMHead();
    if (lm_head && !config.tie_word_embeddings)
        add_T(lm_head->getWeights(), "lm_head.weight");

    // ---- write model.safetensors ----
    nlohmann::json header;
    header["__metadata__"] = {{"format", "pt"}};
    uint64_t offset = 0;
    for (const auto& t : tensors) {
        const uint64_t nbytes = (uint64_t)t.num_elements * sizeof(float);
        header[t.name] = {{"dtype", "F32"}, {"shape", t.shape},
                          {"data_offsets", {offset, offset + nbytes}}};
        offset += nbytes;
    }
    std::string hs = header.dump();
    while (hs.size() % 8 != 0) hs.push_back(' ');
    std::ofstream out(out_dir + "/model.safetensors", std::ios::binary);
    if (!out.is_open()) { std::cerr << "HF export: cannot open model.safetensors\n"; return false; }
    const uint64_t hlen = hs.size();
    out.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
    out.write(hs.data(), (std::streamsize)hs.size());
    for (const auto& t : tensors)
        out.write(reinterpret_cast<const char*>(t.ptr()),
                  (std::streamsize)(t.num_elements * sizeof(float)));
    out.close();
    if (!out) { std::cerr << "HF export: write failed\n"; return false; }

    // ---- config.json (LlamaConfig) ----
    nlohmann::json cfg = {
        {"architectures", {"LlamaForCausalLM"}},
        {"model_type", "llama"},
        {"hidden_size", H},
        {"intermediate_size", mc.intermediate_size},
        {"num_hidden_layers", layers.size()},
        {"num_attention_heads", n_heads},
        {"num_key_value_heads", n_kv_heads},
        {"vocab_size", vocab},
        {"max_position_embeddings", mc.max_seq_length},
        {"rms_norm_eps", mc.layer_norm_epsilon},
        {"rope_theta", 10000.0},
        {"hidden_act", "silu"},
        {"tie_word_embeddings", config.tie_word_embeddings},
        {"torch_dtype", "float32"},
        {"bos_token_id", 2}, {"eos_token_id", 3}, {"pad_token_id", 0},
    };
    std::ofstream cf(out_dir + "/config.json"); cf << cfg.dump(2); cf.close();

    // ---- vocab + README (word-level tokenizer is non-standard for HF) ----
    nlohmann::json vj;
    const size_t vsz = std::min(tokenizer.vocab_size(), mc.vocab_size);
    for (size_t i = 0; i < vsz; ++i) vj[tokenizer.decode({(int)i})] = i;
    std::ofstream vf(out_dir + "/vocab.json"); vf << vj.dump(); vf.close();
    std::ofstream rd(out_dir + "/README.md");
    rd << "# transformer_cpp HF export\n\n"
          "`config.json` + `model.safetensors` load as `LlamaForCausalLM` in\n"
          "HuggingFace Transformers (`AutoModelForCausalLM.from_pretrained(dir)`).\n"
          "Q/K weights carry the RoPE permute (interleaved -> rotate_half), so\n"
          "generation matches this repo's own inference.\n\n"
          "TOKENIZER: this model is WORD-LEVEL (whitespace split, `vocab.json`\n"
          "id map), not SentencePiece/BPE. HF's `LlamaTokenizer` will not match\n"
          "it; wrap the vocab in a `PreTrainedTokenizerFast`/`WordLevel` model\n"
          "for text I/O, or feed token ids directly.\n";
    rd.close();

    std::cout << "HF export: " << tensors.size() << " tensors + config.json + vocab.json -> "
              << out_dir << "  (" << offset / (1024.0 * 1024.0) << " MB)" << std::endl;
    return true;
}

}  // namespace safetensors_export
