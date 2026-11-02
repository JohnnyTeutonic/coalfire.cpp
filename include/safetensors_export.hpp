#pragma once

/**
 * @file safetensors_export.hpp
 * @brief Export a trained Transformer to the safetensors format.
 *
 * Safetensors layout: 8-byte little-endian u64 header length, then a JSON
 * header mapping tensor names to {dtype, shape, data_offsets}, then the raw
 * tensor bytes (offsets relative to the start of the data section).
 *
 * Tensor naming and layout follow the GGUF exporter exactly (llama.cpp-style
 * names; projection weights transposed to [out, in]) so the two exporters
 * describe the same model identically. Model hyperparameters are recorded in
 * the header's __metadata__ block, and the word-level vocabulary is written
 * to a sidecar <path>.vocab.json so the export is loadable end to end.
 */

#include <string>

class Transformer;
class TiktokenTokenizer;

namespace safetensors_export {

struct SafetensorsExportConfig {
    std::string model_name = "transformer_cpp";
    // Omit the head tensor (as GGUFExportConfig's flag does). Refused, writing
    // nothing, unless the model's head is exactly the transposed embedding with
    // a zero bias (export_guard::tied_head_lossless).
    bool tie_word_embeddings = false;
    bool write_vocab_sidecar = true;
};

bool export_to_safetensors(
    const Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::string& output_path,
    const SafetensorsExportConfig& config = SafetensorsExportConfig());

// HuggingFace-compatible export: writes <out_dir>/model.safetensors with HF
// LlamaForCausalLM tensor names, <out_dir>/config.json (LlamaConfig), and a
// vocab sidecar. Crucially, applies the RoPE weight PERMUTE to q_proj/k_proj:
// this codebase (like llama.cpp/GGUF) rotates ADJACENT dimension pairs
// ("interleaved"), while HF Transformers rotates split halves ("rotate_half")
// and stores Q/K permuted so the two agree. Without the permute an HF load
// runs but produces garbage — the classic undocumented safetensors<->GGUF
// RoPE gotcha. out_dir is created if absent.
bool export_to_hf(
    const Transformer& transformer,
    const TiktokenTokenizer& tokenizer,
    const std::string& out_dir,
    const SafetensorsExportConfig& config = SafetensorsExportConfig());

}  // namespace safetensors_export
