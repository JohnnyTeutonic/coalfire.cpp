#pragma once

/**
 * @file gguf_export.hpp
 * @brief GGUF (GPT-Generated Unified Format) exporter for transformer_cpp models
 *
 * Exports trained models to GGUF format for inference with tinyllama.cpp
 * and other GGUF-compatible inference engines. The bytes are written by the
 * shared, standalone writer in gguf_writer.hpp (GGUFWriter, GGMLType,
 * TensorExportInfo and the GGUF constants), which paperkiln's exporters use
 * too.
 */

#include <string>

#include "gguf_writer.hpp"

// Forward declarations
class Transformer;
class TiktokenTokenizer;
namespace byte_bpe {
struct GgufTokenizer;
}

namespace gguf_export {

/**
 * @brief Configuration for GGUF export
 */
struct GGUFExportConfig {
    std::string model_name = "transformer_cpp_model";
    std::string architecture = "llama";  // For compatibility with llama.cpp ecosystem
    // F32 only: the exporter writes the trained F32 weights unconverted, and
    // export_to_gguf refuses any other type (they would be float bytes under
    // another type's tag).
    GGMLType weight_type = GGMLType::F32;
    bool include_tokenizer = true;
    bool tie_word_embeddings = false;  // Whether lm_head shares weights with embeddings
};

/**
 * @brief Export a Transformer model to GGUF format
 *
 * @param transformer The trained transformer model
 * @param tokenizer The tokenizer (for vocabulary export)
 * @param output_path Path for the output .gguf file
 * @param config Export configuration
 * @return true if export succeeded
 */
bool export_to_gguf(
    const Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::string& output_path,
    const GGUFExportConfig& config = GGUFExportConfig()
);

/**
 * @brief Export a Transformer trained on a byte-level BPE vocabulary
 *
 * The same file as the word-level export except the tokenizer block, which
 * is paperkiln's for a "bpe" vocabulary (its src/gguf.cpp write_tokenizer):
 * tokenizer.ggml.model "bpe", pre when the vocabulary has one, the tokens,
 * the merges, add_bos/add_eos false when pre is set, zero scores, and
 * bos = 1, eos = the id of <|endoftext|> (else 1), unknown = padding = 0.
 * Refuses a vocabulary whose model is not "bpe", one without merges, and one
 * whose size is not the model's vocabulary size.
 *
 * @param vocab The vocabulary as read by byte_bpe::read_gguf_tokenizer
 *              (a tools/bpe.py vocabulary GGUF or a BPE model export)
 */
bool export_to_gguf(
    const Transformer& transformer,
    const byte_bpe::GgufTokenizer& vocab,
    const std::string& output_path,
    const GGUFExportConfig& config = GGUFExportConfig()
);

/**
 * @brief Convert tensor name from transformer_cpp convention to GGUF/llama convention
 *
 * transformer_cpp: layer_0_attention_q_proj
 * GGUF:           blk.0.attn_q.weight
 */
std::string convert_tensor_name(const std::string& internal_name, int layer_idx = -1);

} // namespace gguf_export
