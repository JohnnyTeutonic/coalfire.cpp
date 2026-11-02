// export_to_gguf with a byte-level BPE vocabulary (ROADMAP C5).
//
// A tiny llama model on tests/fixtures/bpe/small.gguf's vocabulary (paperkiln
// tools/bpe.py) is exported and checked:
//  - its tokenizer.ggml.* entries, as raw key/type/value bytes in file order,
//    equal tests/fixtures/bpe/paperkiln_tokenizer_kv.bin: the same entries cut
//    from the GGUF paperkiln's mtstudio exports for a model on that vocabulary
//    (paperkiln 7546f51d, `mtstudio run` with data.vocab = small.gguf);
//  - the vocabulary reads back from the export (model, tokens, merges, no pre
//    or token types) and encodes as the vocabulary does;
//  - with include_tokenizer off, the BPE and word-level exports of the same
//    model are the same bytes: only the tokenizer block differs;
//  - a vocabulary whose model is not "bpe", one without merges and one whose
//    size is not the model's are refused, and nothing is written.
//
//   bpe_export_test FIXTURE_DIR
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "architecture.hpp"
#include "byte_bpe_tokenizer.hpp"
#include "config.hpp"
#include "embeddings.hpp"
#include "gguf_export.hpp"
#include "layer_norm.hpp"
#include "lm_head.hpp"
#include "tiktoken_tokenizer.hpp"
#include "transformer.hpp"

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool exists(const std::string& path) { return static_cast<bool>(std::ifstream(path, std::ios::binary)); }

// The raw bytes (key, type, value) of every metadata entry whose key starts
// with prefix, concatenated in file order.
std::string kv_bytes(const std::string& buf, const std::string& prefix) {
    size_t p = 0;
    auto need = [&](size_t n) {
        if (p + n > buf.size()) throw std::runtime_error("truncated GGUF");
    };
    auto u32 = [&]() { need(4); uint32_t v; std::memcpy(&v, buf.data() + p, 4); p += 4; return v; };
    auto u64 = [&]() { need(8); uint64_t v; std::memcpy(&v, buf.data() + p, 8); p += 8; return v; };
    std::function<void(uint32_t)> skip = [&](uint32_t t) {
        static const size_t sz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (t == 8) {
            const uint64_t n = u64();
            need(n);
            p += n;
        } else if (t == 9) {
            const uint32_t et = u32();
            const uint64_t n = u64();
            for (uint64_t i = 0; i < n; ++i) skip(et);
        } else if (t < 13) {
            need(sz[t]);
            p += sz[t];
        } else {
            throw std::runtime_error("unknown GGUF type");
        }
    };
    if (u32() != 0x46554747u) throw std::runtime_error("not GGUF");
    u32();
    u64();
    const uint64_t nkv = u64();
    std::string out;
    for (uint64_t i = 0; i < nkv; ++i) {
        const size_t start = p;
        const uint64_t n = u64();
        need(n);
        const std::string key = buf.substr(p, n);
        p += n;
        skip(u32());
        if (key.compare(0, prefix.size(), prefix) == 0) out += buf.substr(start, p - start);
    }
    return out;
}

// An exact float in [-0.5, 0.5) from an integer hash (as gguf_export_test).
float val(uint32_t seed, size_t i) {
    uint32_t h = seed * 0x9E3779B1u ^ static_cast<uint32_t>(i) * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return static_cast<float>(static_cast<int32_t>(h >> 8) - (1 << 23)) / static_cast<float>(1 << 24);
}

uint32_t g_seed = 1;
void fill(Matrix& m) {
    const uint32_t s = g_seed++;
    float* p = m.data();
    for (size_t i = 0; i < m.rows() * m.cols(); ++i) p[i] = val(s, i);
}

std::unique_ptr<Transformer> model(std::shared_ptr<TiktokenTokenizer> tok) {
    TransformerConfig cfg;
    cfg.hidden_size = 8;
    cfg.num_heads = 2;
    cfg.num_layers = 2;
    cfg.intermediate_size = 12;
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.max_seq_length = 16;
    cfg.layer_norm_epsilon = 1e-5f;
    cfg.use_flash_attention = false;
    cfg.use_gqa = false;
    cfg.num_kv_heads = cfg.num_heads;
    arch::ArchitectureSpec spec = arch::ArchitectureSpec::from_family("llama");
    spec.validate();
    spec.apply(cfg);
    transformer_runtime::llama_no_bias = !cfg.use_biases;
    auto t = std::make_unique<Transformer>(cfg, tok);
    g_seed = 300;
    fill(const_cast<TokenEmbedding*>(t->getTokenEmbedding())->get_weights());
    for (auto& layer : t->getLayers()) {
        auto& ap = layer->getAttention()->parameters();
        fill(ap.query_weights);
        fill(ap.key_weights);
        fill(ap.value_weights);
        fill(ap.output_weights);
        fill(layer->getLayerNorm()->get_gamma_mut());
        auto& fp = layer->getFeedForward()->parameters();
        fill(fp.gate_proj_weights);
        fill(fp.up_proj_weights);
        fill(fp.down_proj_weights);
        fill(const_cast<LayerNorm*>(layer->getFfnLayerNorm())->get_gamma_mut());
    }
    fill(const_cast<LayerNorm*>(t->getFinalLayerNorm())->get_gamma_mut());
    fill(t->get_lm_head()->get_weights());
    return t;
}

std::string tmp_dir() {
    const char* t = std::getenv("TMPDIR");
#ifdef _WIN32
    if (!t) t = std::getenv("TEMP");
#endif
    return t ? std::string(t) : std::string("/tmp");
}

int failed = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-64s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("usage: bpe_export_test FIXTURE_DIR\n");
        return 2;
    }
    const std::string dir = argv[1];
    const std::string tmp = tmp_dir() + "/bpe_export_test_";
    try {
        const byte_bpe::GgufTokenizer vocab = byte_bpe::read_gguf_tokenizer(dir + "/small.gguf");
        auto mirror = std::make_shared<TiktokenTokenizer>();
        mirror->load_vocabulary_from_tokens(vocab.tokens);
        check(mirror->vocab_size() == vocab.tokens.size() && mirror->decode({300}) == vocab.tokens[300],
              "the word-level tokenizer's id-indexed copy of the vocabulary");
        auto llama = model(mirror);
        check(llama->getConfig().vocab_size == vocab.tokens.size(), "the model's vocabulary is the BPE vocabulary");

        const std::string bpe_path = tmp + "bpe.gguf";
        std::remove(bpe_path.c_str());
        check(gguf_export::export_to_gguf(*llama, vocab, bpe_path), "export_to_gguf(BPE) succeeds");
        const std::string file = read_file(bpe_path);

        const std::string want = read_file(dir + "/paperkiln_tokenizer_kv.bin");
        const std::string got = kv_bytes(file, "tokenizer.ggml.");
        check(!want.empty() && got == want,
              "tokenizer.ggml.* bytes equal paperkiln's export (" + std::to_string(got.size()) + " bytes)");

        const byte_bpe::GgufTokenizer back = byte_bpe::read_gguf_tokenizer(bpe_path);
        check(back.model == "bpe" && back.tokens == vocab.tokens && back.merges == vocab.merges && back.pre.empty() &&
                  back.token_type.empty(),
              "the vocabulary reads back from the export");
        const byte_bpe::Tokenizer a = byte_bpe::load_gguf(dir + "/small.gguf"), b = byte_bpe::load_gguf(bpe_path);
        const std::string text = "Once upon a time, Lily had 12 apples.\n<|user|> hi<|assistant|> ok<|endoftext|>";
        check(!a.encode(text).empty() && a.encode(text) == b.encode(text) && b.decode(b.encode(text)) == text,
              "the export encodes as the vocabulary does");
        std::remove(bpe_path.c_str());

        gguf_export::GGUFExportConfig no_tok;
        no_tok.include_tokenizer = false;
        const std::string p1 = tmp + "notok_bpe.gguf", p2 = tmp + "notok_word.gguf";
        const bool ok1 = gguf_export::export_to_gguf(*llama, vocab, p1, no_tok);
        const bool ok2 = gguf_export::export_to_gguf(*llama, *mirror, p2, no_tok);
        const std::string f1 = read_file(p1), f2 = read_file(p2);
        check(ok1 && ok2 && !f1.empty() && f1 == f2, "without the tokenizer, the BPE and word-level exports are equal");
        std::remove(p1.c_str());
        std::remove(p2.c_str());

        auto refused = [&](byte_bpe::GgufTokenizer v, const std::string& what) {
            const std::string p = tmp + "refused.gguf";
            std::remove(p.c_str());
            const bool ok = gguf_export::export_to_gguf(*llama, v, p);
            check(!ok && !exists(p), what);
            std::remove(p.c_str());
        };
        byte_bpe::GgufTokenizer v = vocab;
        v.model = "word";
        refused(v, "a vocabulary whose model is not \"bpe\" is refused");
        v = vocab;
        v.merges.clear();
        refused(v, "a vocabulary without merges is refused");
        v = vocab;
        v.tokens.push_back("extra");
        refused(v, "a vocabulary larger than the model's is refused");
    } catch (const std::exception& e) {
        std::printf("  error: %s\n", e.what());
        ++failed;
    }
    if (failed) {
        std::printf("\n[FAIL] %d BPE export checks\n", failed);
        return 1;
    }
    std::printf("\n[PASS] BPE GGUF export matches paperkiln's tokenizer block\n");
    return 0;
}
