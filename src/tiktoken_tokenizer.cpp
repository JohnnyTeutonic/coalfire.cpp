#include "../include/tiktoken_tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <nlohmann/json.hpp>

std::vector<std::string> TiktokenTokenizer::split_into_words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }
    return words;
}

std::string TiktokenTokenizer::preprocess_text(const std::string& text) {
    return text;  // No preprocessing needed for simple word tokenization
}

void TiktokenTokenizer::build_vocabulary_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    // Reset vocabulary
    word_to_id_.clear();
    id_to_word_.clear();
    
    // Add special tokens first
    word_to_id_[UNK_TOKEN] = UNK_ID;
    id_to_word_[UNK_ID] = UNK_TOKEN;
    word_to_id_[SEP_TOKEN] = SEP_ID;  // Add separator token
    id_to_word_[SEP_ID] = SEP_TOKEN;  // Add separator token
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Find the separator (|, *, or #)
        size_t sep_pos = line.find_first_of("|*#");
        if (sep_pos == std::string::npos) continue;
        
        // Split into context and target
        std::string context = line.substr(0, sep_pos);
        std::string target = line.substr(sep_pos + 1);
        
        // Process context words
        auto context_words = split_into_words(context);
        for (const auto& word : context_words) {
            if (word_to_id_.find(word) == word_to_id_.end()) {
                int new_id = word_to_id_.size();
                word_to_id_[word] = new_id;
                id_to_word_[new_id] = word;
            }
        }
        
        // Process target word(s)
        auto target_words = split_into_words(target);
        for (const auto& word : target_words) {
            if (word_to_id_.find(word) == word_to_id_.end()) {
                int new_id = word_to_id_.size();
                word_to_id_[word] = new_id;
                id_to_word_[new_id] = word;
            }
        }
    }
    
    initialized_ = true;
    size_t actual_vocab_size = word_to_id_.size();
    std::cout << "Built vocabulary with " << actual_vocab_size << " tokens" << std::endl;
}

void TiktokenTokenizer::build_vocabulary_from_plain_text(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    // Track word frequencies - but stop early if too many unique words
    std::unordered_map<std::string, size_t> word_freq;
    std::string line;
    size_t line_count = 0;
    const size_t MAX_UNIQUE_WORDS = 100000;  // Stop counting after 100k unique words
    
    std::cout << "Counting word frequencies (max " << MAX_UNIQUE_WORDS << " unique words)..." << std::endl;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        line_count++;
        
        // Count word frequencies (lowercase for consistency)
        auto words = split_into_words(line);
        for (const auto& word : words) {
            std::string lower_word = word;
            std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
            word_freq[lower_word]++;
        }
        
        // Stop if vocabulary gets too large (memory protection)
        if (word_freq.size() > MAX_UNIQUE_WORDS) {
            std::cout << "Reached " << MAX_UNIQUE_WORDS << " unique words at line " 
                      << line_count << ", stopping count..." << std::endl;
            break;
        }
        
        // Progress indicator
        if (line_count % 10000 == 0) {
            std::cout << "Processed " << line_count << " lines, " 
                      << word_freq.size() << " unique words..." << std::endl;
        }
    }
    
    std::cout << "Found " << word_freq.size() << " unique words in " << line_count << " lines" << std::endl;
    std::cout << "Sorting by frequency..." << std::endl;
    
    // Sort by frequency (descending)
    std::vector<std::pair<std::string, size_t>> freq_vec(word_freq.begin(), word_freq.end());
    std::sort(freq_vec.begin(), freq_vec.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "Sorting complete, building vocabulary..." << std::endl;
    
    // Build vocabulary with top 20k words (must match model vocab_size)
    const size_t MAX_VOCAB_SIZE = 20000;
    
    // If vocabulary is empty, add special tokens
    if (word_to_id_.empty()) {
        word_to_id_[UNK_TOKEN] = UNK_ID;
        id_to_word_[UNK_ID] = UNK_TOKEN;
        word_to_id_[SEP_TOKEN] = SEP_ID;
        id_to_word_[SEP_ID] = SEP_TOKEN;
    }
    
    // Add top frequent words.
    // Cap on TOTAL vocabulary size (not per-call), so building from multiple
    // files still respects the global limit instead of accumulating past it.
    size_t added = 0;
    for (const auto& [word, freq] : freq_vec) {
        if (word_to_id_.size() >= MAX_VOCAB_SIZE) break;
        
        if (word_to_id_.find(word) == word_to_id_.end()) {
            int new_id = word_to_id_.size();
            word_to_id_[word] = new_id;
            id_to_word_[new_id] = word;
            added++;
        }
    }
    
    initialized_ = true;
    std::cout << "Built vocabulary with top " << added << " most frequent tokens" << std::endl;
    std::cout << "Total vocabulary size: " << word_to_id_.size() << std::endl;
}

void TiktokenTokenizer::load_vocabulary_from_id_list(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open vocab id list: " + filepath);
    }
    word_to_id_.clear();
    id_to_word_.clear();
    std::string line;
    int id = 0;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        word_to_id_[line] = id;
        id_to_word_[id] = line;
        ++id;
    }
    initialized_ = true;
    std::cout << "Loaded vocabulary from id list: " << id << " tokens" << std::endl;
}

void TiktokenTokenizer::load_vocabulary_from_tokens(const std::vector<std::string>& tokens) {
    word_to_id_.clear();
    id_to_word_.clear();
    for (size_t i = 0; i < tokens.size(); ++i) {
        word_to_id_.emplace(tokens[i], static_cast<int>(i));  // a repeated string keeps its first id
        id_to_word_[static_cast<int>(i)] = tokens[i];
    }
    id_indexed_ = true;
    initialized_ = true;
}

std::vector<int> TiktokenTokenizer::encode(const std::string& text) const {
    if (!initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    std::vector<int> tokens;
    auto words = split_into_words(text);
    
    for (const auto& word : words) {
        // Lowercase for consistency with vocabulary
        std::string lower_word = word;
        std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
        
        auto it = word_to_id_.find(lower_word);
        if (it != word_to_id_.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(UNK_ID);
        }
    }
    
    return tokens;
}

std::string TiktokenTokenizer::decode(const std::vector<int>& tokens) const {
    if (!initialized_) {
        throw std::runtime_error("Tokenizer not initialized");
    }
    
    std::string result;
    bool first = true;
    
    for (int token_id : tokens) {
        auto it = id_to_word_.find(token_id);
        if (!first) {
            result += " ";
        }
        if (it != id_to_word_.end()) {
            result += it->second;
        } else {
            result += UNK_TOKEN;
        }
        first = false;
    }
    
    return result;
}

void TiktokenTokenizer::print_vocabulary() const {
    std::cout << "Vocabulary (" << word_to_id_.size() << " words):" << std::endl;
    for (const auto& [word, id] : word_to_id_) {
        std::cout << id << ": '" << word << "'" << std::endl;
    }
}

bool TiktokenTokenizer::is_special_token(int token_id) const {
    return token_id == UNK_ID || token_id == SEP_ID;  // Updated to include separator token
}

bool TiktokenTokenizer::is_verb(const std::string& token) const {
    // Common verb endings
    const std::vector<std::string> verb_endings = {
        "ing", "ed", "ate", "ize", "ify", "ise", "ect",
        "ent", "age", "ute", "end", "ish", "ade", "ine"
    };

    // Common verbs
    const std::unordered_set<std::string> common_verbs = {
        "go", "do", "make", "take", "come", "see", "get",
        "know", "find", "give", "tell", "work", "call", "try"
    };

    // Check if it's a common verb
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    if (common_verbs.find(lower_token) != common_verbs.end()) {
        return true;
    }

    // Check for verb endings
    for (const auto& ending : verb_endings) {
        if (lower_token.length() > ending.length() && 
            lower_token.substr(lower_token.length() - ending.length()) == ending) {
            return true;
        }
    }

    return false;
}

bool TiktokenTokenizer::is_adjective(const std::string& token) const {
    // Common adjective endings
    const std::vector<std::string> adj_endings = {
        "ful", "ous", "ible", "able", "al", "ive", "less",
        "ish", "like", "ic", "ian", "en", "ent", "ant"
    };

    // Common adjectives
    const std::unordered_set<std::string> common_adjectives = {
        "good", "bad", "new", "old", "high", "low", "big",
        "small", "large", "little", "long", "short", "great"
    };

    // Check if it's a common adjective
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    if (common_adjectives.find(lower_token) != common_adjectives.end()) {
        return true;
    }

    // Check for adjective endings
    for (const auto& ending : adj_endings) {
        if (lower_token.length() > ending.length() && 
            lower_token.substr(lower_token.length() - ending.length()) == ending) {
            return true;
        }
    }

    return false;
}

bool TiktokenTokenizer::is_noun(const std::string& token) const {
    // Common noun endings
    const std::vector<std::string> noun_endings = {
        "tion", "sion", "ment", "ness", "ity", "ship", "dom",
        "ence", "ance", "ist", "er", "or", "ism", "ing"
    };

    // Common nouns
    const std::unordered_set<std::string> common_nouns = {
        "time", "person", "year", "way", "day", "thing", "man",
        "world", "life", "hand", "part", "child", "eye", "place"
    };

    // Check if it's a common noun
    std::string lower_token = token;
    std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
    if (common_nouns.find(lower_token) != common_nouns.end()) {
        return true;
    }

    // Check for noun endings
    for (const auto& ending : noun_endings) {
        if (lower_token.length() > ending.length() && 
            lower_token.substr(lower_token.length() - ending.length()) == ending) {
            return true;
        }
    }

    return false;
}