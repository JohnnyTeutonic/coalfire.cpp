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
 * @brief Convert tensor name from transformer_cpp convention to GGUF/llama convention
 *
 * transformer_cpp: layer_0_attention_q_proj
 * GGUF:           blk.0.attn_q.weight
 */
std::string convert_tensor_name(const std::string& internal_name, int layer_idx = -1);

} // namespace gguf_export
