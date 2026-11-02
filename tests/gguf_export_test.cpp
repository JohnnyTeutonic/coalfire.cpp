// GGUF byte identity for export_to_gguf, every option, hashed.
//
// Each case builds a tiny llama-family Transformer, overwrites every weight
// with an exact dyadic float from an integer hash (so the bytes do not depend
// on the initialiser, the compiler or -march), exports it, and compares the
// file's SHA-256 with the value recorded below. The hashes were recorded from
// the exporter as it stood before its byte writer moved into the shared
// header include/gguf_writer.hpp (3 Nov 2026), so a pass means every file is
// the bytes it was. A mismatch is a change in what coalfire writes: fix the
// writer, never the table, unless the format change is the intended one.
//
//   gguf_export_test          check every case
//   gguf_export_test --print  print "name sha256" for every case
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "architecture.hpp"
#include "config.hpp"
#include "embeddings.hpp"
#include "gguf_export.hpp"
#include "layer_norm.hpp"
#include "lm_head.hpp"
#include "tiktoken_tokenizer.hpp"
#include "transformer.hpp"

namespace {

// ---- SHA-256 (FIPS 180-4), for the hashes below ----
std::string sha256_hex(const std::string& msg) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    std::string m = msg;
    const uint64_t bits = static_cast<uint64_t>(msg.size()) * 8;
    m.push_back(static_cast<char>(0x80));
    while (m.size() % 64 != 56) m.push_back('\0');
    for (int i = 7; i >= 0; --i) m.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const auto* p = reinterpret_cast<const unsigned char*>(m.data() + off + 4 * i);
            w[i] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    }
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 64);
}

std::string file_sha256(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "missing";
    std::ostringstream ss;
    ss << f.rdbuf();
    return sha256_hex(ss.str());
}

// An exact float in [-0.5, 0.5): a 24-bit integer hash over 2^24.
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

std::string tmp_dir() {
    const char* t = std::getenv("TMPDIR");
#ifdef _WIN32
    if (!t) t = std::getenv("TEMP");
#endif
    return t ? std::string(t) : std::string("/tmp");
}

std::shared_ptr<TiktokenTokenizer> vocab(const std::string& path, const std::vector<std::string>& words) {
    {
        std::ofstream f(path);
        for (const auto& w : words) f << w << "\n";
    }
    auto tok = std::make_shared<TiktokenTokenizer>();
    tok->load_vocabulary_from_id_list(path);
    std::remove(path.c_str());
    return tok;
}

std::unique_ptr<Transformer> model(const std::string& family, std::shared_ptr<TiktokenTokenizer> tok,
                                   uint32_t seed) {
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
    arch::ArchitectureSpec spec = arch::ArchitectureSpec::from_family(family);
    spec.validate();
    spec.apply(cfg);
    transformer_runtime::llama_no_bias = !cfg.use_biases;
    auto t = std::make_unique<Transformer>(cfg, tok);

    g_seed = seed;
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

// The recorded SHA-256 of every case (see the header comment). "refused":
// export_to_gguf must return false and write nothing.
const std::map<std::string, std::string> kWant = {
    {"llama_default", "618d57cb9147d003be9b0236886d527631bb714f2facea51295eaffd76901edf"},
    {"llama_no_tokenizer", "da1a64e4615930c5b7b4ac26e2590db18386753c0987361bd72a7db24dab6a76"},
    {"llama_tied", "bfcf3327c3ca16a33a28436004daa2c87e583901ab1f23e3ce384ce4a818805f"},
    {"llama_name_arch", "12dc3c843252bcef96708ef31f210b39e4e1bed9e1d867e6657919978286718a"},
    {"llama_eos_endoftext", "71a4f3b58d0e40bda96691187389df22a91b9a5911cba23f3b5d216d17ba9e7d"},
    // only the model's 12 ids are exported, so the file is llama_default's
    {"llama_tokenizer_larger_than_model", "618d57cb9147d003be9b0236886d527631bb714f2facea51295eaffd76901edf"},
    {"vanilla_no_rope", "f981b87bd5230b91cdbab2c1fcbdcd919bb1c2d3b761393fdb6fd0494f418b74"},
    {"vanilla_tied_no_tokenizer", "15fad6377f16e2a4277816d8c01cc3c7ffd60fa07f48548fe0cad1aeed798546"},
    {"llama_weight_f16", "refused"},
    {"llama_weight_bf16", "refused"},
    {"llama_weight_q8_0", "refused"},
    {"llama_weight_q4_0", "refused"},
};

}  // namespace

int main(int argc, char** argv) {
    const bool print = argc > 1 && std::string(argv[1]) == "--print";
    const std::string tmp = tmp_dir();
    if (sha256_hex("abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ||
        sha256_hex(std::string(1000, 'a')) != "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3") {
        std::printf("[FAIL] the test's SHA-256 is wrong\n");
        return 1;
    }

    std::vector<std::string> words = {"<unk>", "|", "the", "and", "a", "cat", "sat", "on", "mat", "dog", "ran", "."};
    auto tok = vocab(tmp + "/gguf_export_test_vocab.txt", words);
    std::vector<std::string> words_eos = words;
    words_eos[7] = "<|endoftext|>";
    auto tok_eos = vocab(tmp + "/gguf_export_test_vocab.txt", words_eos);
    std::vector<std::string> words_big = words;
    for (int i = 0; i < 9; ++i) words_big.push_back("extra" + std::to_string(i));
    auto tok_big = vocab(tmp + "/gguf_export_test_vocab.txt", words_big);

    auto llama = model("llama", tok, 100);
    auto vanilla = model("vanilla", tok, 200);  // LayerNorm, no RoPE: no rope.* keys

    using Cfg = gguf_export::GGUFExportConfig;
    struct Case {
        std::string name;
        const Transformer* t;
        const TiktokenTokenizer* tok;
        std::function<void(Cfg&)> set;
    };
    std::vector<Case> cases = {
        {"llama_default", llama.get(), tok.get(), [](Cfg&) {}},
        {"llama_no_tokenizer", llama.get(), tok.get(), [](Cfg& c) { c.include_tokenizer = false; }},
        {"llama_tied", llama.get(), tok.get(), [](Cfg& c) { c.tie_word_embeddings = true; }},
        {"llama_name_arch", llama.get(), tok.get(),
         [](Cfg& c) {
             c.model_name = "transformer_cpp_wikitext";
             c.architecture = "fixture_arch";
         }},
        {"llama_eos_endoftext", llama.get(), tok_eos.get(), [](Cfg&) {}},
        {"llama_tokenizer_larger_than_model", llama.get(), tok_big.get(), [](Cfg&) {}},
        {"vanilla_no_rope", vanilla.get(), tok.get(), [](Cfg&) {}},
        {"vanilla_tied_no_tokenizer", vanilla.get(), tok.get(),
         [](Cfg& c) {
             c.tie_word_embeddings = true;
             c.include_tokenizer = false;
         }},
        // export_to_gguf never converted the F32 weights: these types wrote
        // float bytes under another type's tag (an invalid file), so they are
        // refused.
        {"llama_weight_f16", llama.get(), tok.get(), [](Cfg& c) { c.weight_type = gguf_export::GGMLType::F16; }},
        {"llama_weight_bf16", llama.get(), tok.get(), [](Cfg& c) { c.weight_type = gguf_export::GGMLType::BF16; }},
        {"llama_weight_q8_0", llama.get(), tok.get(), [](Cfg& c) { c.weight_type = gguf_export::GGMLType::Q8_0; }},
        {"llama_weight_q4_0", llama.get(), tok.get(), [](Cfg& c) { c.weight_type = gguf_export::GGMLType::Q4_0; }},
    };

    int failed = 0;
    for (const auto& c : cases) {
        const std::string path = tmp + "/gguf_export_test_" + c.name + ".gguf";
        std::remove(path.c_str());
        Cfg cfg;
        c.set(cfg);
        const bool ok = gguf_export::export_to_gguf(*c.t, *c.tok, path, cfg);
        std::string got = ok ? file_sha256(path) : "refused";
        if (!ok && file_sha256(path) != "missing") got = "refused-but-wrote";
        std::remove(path.c_str());
        if (print) {
            std::printf("%s %s\n", c.name.c_str(), got.c_str());
            continue;
        }
        auto it = kWant.find(c.name);
        if (it == kWant.end()) {
            std::printf("  %-36s %s  (no recorded hash)\n", c.name.c_str(), got.c_str());
            ++failed;
        } else if (it->second != got) {
            std::printf("  %-36s %s  MISMATCH, recorded %s\n", c.name.c_str(), got.c_str(), it->second.c_str());
            ++failed;
        } else {
            std::printf("  %-36s ok\n", c.name.c_str());
        }
    }
    if (print) return 0;
    if (kWant.size() != cases.size()) {
        std::printf("recorded %zu hashes for %zu cases\n", kWant.size(), cases.size());
        ++failed;
    }
    if (failed) {
        std::printf("\n[FAIL] %d GGUF exports differ from their recorded bytes\n", failed);
        return 1;
    }
    std::printf("\n[PASS] %zu GGUF exports byte-identical to their recorded hashes\n", cases.size());
    return 0;
}
