#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class TiktokenTokenizer {
private:
    std::unordered_map<std::string, int> word_to_id_;
    std::unordered_map<int, std::string> id_to_word_;
    bool initialized_ = false;
    bool id_indexed_ = false;  // set by load_vocabulary_from_tokens only
    static constexpr int UNK_ID = 0;
    static constexpr const char* UNK_TOKEN = "<unk>";
    static constexpr int SEP_ID = 1;  // Add separator token ID
    static constexpr const char* SEP_TOKEN = "|";  // Use | as separator token
    
public:
    // Core tokenization methods
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    
    // Text preprocessing
    static std::string preprocess_text(const std::string& text);
    
    // Vocabulary management
    void build_vocabulary_from_file(const std::string& filepath);
    void build_vocabulary_from_plain_text(const std::string& filepath);  // For WikiText-style data
    // One word per line, line number = token id. The frequency-sorted builder
    // above has NO tie-break (platform-dependent ids for equal-frequency
    // words), so cross-machine reproduction of a training run's vocab must go
    // through the id list embedded in its exported GGUF
    // (scripts/extract_gguf_vocab.py).
    void load_vocabulary_from_id_list(const std::string& filepath);
    // An id-indexed copy of another tokenizer's vocabulary (train --bpe-vocab:
    // the byte-level BPE tokens of byte_bpe_tokenizer.hpp), for the model's
    // wiring: vocab_size() is tokens.size() and decode({id}) is tokens[id],
    // even when two ids share a string. encode() stays the word-level lookup;
    // the data is encoded by the BPE tokenizer, never by this one.
    void load_vocabulary_from_tokens(const std::vector<std::string>& tokens);
    size_t vocab_size() const { return id_indexed_ ? id_to_word_.size() : word_to_id_.size(); }
    bool is_initialized() const { return initialized_; }
    
    // Token type checking
    bool is_verb(const std::string& token) const;
    bool is_adjective(const std::string& token) const;
    bool is_noun(const std::string& token) const;
    bool is_special_token(int token_id) const;
    
    // Special token getters
    int get_unk_token_id() const { return UNK_ID; }
    int get_sep_token_id() const { return SEP_ID; }
    
    // Debugging helpers
    void print_vocabulary() const;
    
private:
    static std::vector<std::string> split_into_words(const std::string& text);
}; 