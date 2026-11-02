// byte_bpe_tokenizer.hpp against paperkiln's tokenizer (ROADMAP C5).
//
// tests/fixtures/bpe/ holds two vocabularies trained by paperkiln's
// reference trainer (tools/bpe.py at paperkiln 7546f51d): small.gguf (395
// tokens, 135 merges, on tests/test_bpe.py's text) and big.gguf (2,048 tokens,
// 1,788 merges, on paperkiln's README.md, ROADMAP.md, docs/ECOSYSTEM.md and
// CHANGELOG.md). ids_<vocab>.txt has one case per line: the text's bytes in
// hex (empty for the empty string), a tab, and the ids paperkiln's C++
// tokenizer (include/microtorch/bpe_tokenizer.hpp at the same commit, before
// it moved here) gave it; tools/bpe.py gave the same ids. The cases are edge
// cases (empty, whitespace runs, digits, apostrophes, multibyte and invalid
// UTF-8, NUL, the specials, partial specials) and a sample of those
// documents' lines.
//
// Each case must encode to the recorded ids and decode back to its bytes, and
// its pretokenization must partition it. A mismatch means coalfire and
// paperkiln no longer agree: fix the tokenizer, never the fixture.
//
//   bpe_tokenizer_test FIXTURE_DIR
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "byte_bpe_tokenizer.hpp"
#include "gguf_writer.hpp"

namespace {

std::string unhex(const std::string& h) {
    std::string s;
    for (size_t i = 0; i + 1 < h.size(); i += 2) s.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
    return s;
}

int check_vocab(const std::string& dir, const std::string& name, size_t want_tokens, size_t want_merges) {
    const std::string gguf = dir + "/" + name + ".gguf";
    const byte_bpe::GgufTokenizer g = byte_bpe::read_gguf_tokenizer(gguf);
    const byte_bpe::Tokenizer tok = byte_bpe::load_gguf(gguf);
    int failed = 0;
    if (g.model != "bpe" || g.tokens.size() != want_tokens || g.merges.size() != want_merges || tok.size() != want_tokens) {
        std::printf("  %s: model '%s', %zu tokens, %zu merges; want bpe, %zu, %zu\n", name.c_str(), g.model.c_str(),
                    g.tokens.size(), g.merges.size(), want_tokens, want_merges);
        ++failed;
    }
    // The layout every paperkiln vocabulary has: the specials, the 256 bytes, then the merges.
    for (size_t i = 0; i < byte_bpe::specials().size(); ++i)
        if (tok.tokens()[i] != byte_bpe::specials()[i]) ++failed;
    for (int b = 0; b < 256; ++b)
        if (tok.tokens()[byte_bpe::specials().size() + b] != std::string(1, static_cast<char>(b))) ++failed;

    std::ifstream f(dir + "/ids_" + name + ".txt");
    if (!f) {
        std::printf("  %s: no ids_%s.txt\n", name.c_str(), name.c_str());
        return failed + 1;
    }
    std::string line;
    size_t cases = 0, ids_total = 0;
    while (std::getline(f, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            std::printf("  %s: malformed line %zu\n", name.c_str(), cases + 1);
            ++failed;
            continue;
        }
        const std::string text = unhex(line.substr(0, tab));
        std::vector<int> want;
        std::istringstream ws(line.substr(tab + 1));
        for (int id; ws >> id;) want.push_back(id);
        const std::vector<int> got = tok.encode(text);
        ++cases;
        ids_total += want.size();
        if (got != want) {
            std::printf("  %s case %zu (%s): %zu ids, recorded %zu\n", name.c_str(), cases,
                        line.substr(0, std::min<size_t>(tab, 40)).c_str(), got.size(), want.size());
            ++failed;
        }
        if (tok.decode(got) != text) {
            std::printf("  %s case %zu: decode(encode(s)) != s\n", name.c_str(), cases);
            ++failed;
        }
        std::string joined;
        for (const auto& [chunk, special] : byte_bpe::Tokenizer::pretokenize(text)) joined += chunk;
        if (joined != text) {
            std::printf("  %s case %zu: the chunks do not concatenate to the text\n", name.c_str(), cases);
            ++failed;
        }
    }
    std::printf("  %-6s %zu cases, %zu ids: %s\n", name.c_str(), cases, ids_total, failed ? "FAIL" : "ok");
    return failed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("usage: bpe_tokenizer_test FIXTURE_DIR\n");
        return 2;
    }
    int failed = 0;
    try {
        failed += check_vocab(argv[1], "small", 395, 135);
        failed += check_vocab(argv[1], "big", 2048, 1788);
        // A vocabulary that is not "bpe" is refused.
        const char* t = std::getenv("TMPDIR");
        const std::string word = std::string(t ? t : "/tmp") + "/bpe_tokenizer_test_word.gguf";
        {
            gguf_export::GGUFWriter w(word);
            w.write_metadata_string("tokenizer.ggml.model", "word");
            w.write_metadata_string_array("tokenizer.ggml.tokens", {"<unk>", "|", "the"});
            if (!w.finalize()) ++failed;
        }
        bool refused = false;
        try {
            byte_bpe::load_gguf(word);
        } catch (const std::exception&) {
            refused = true;
        }
        std::remove(word.c_str());
        if (!refused) {
            std::printf("  a word-level vocabulary was not refused\n");
            ++failed;
        }
    } catch (const std::exception& e) {
        std::printf("  error: %s\n", e.what());
        ++failed;
    }
    if (failed) {
        std::printf("\n[FAIL] %d byte-level BPE checks differ from paperkiln's\n", failed);
        return 1;
    }
    std::printf("\n[PASS] byte-level BPE ids identical to paperkiln's\n");
    return 0;
}
