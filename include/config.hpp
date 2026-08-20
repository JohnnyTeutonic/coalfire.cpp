#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// Forward declarations of configuration structures
struct BeamSearchConfig {
    bool use_beam_search = true;
    size_t beam_size = 4;
    size_t beams_per_group = 4;
    size_t num_groups = 3;
    float length_penalty = 1.5f;
    float temperature = 2.5f;
    float top_p = 0.98f;
    size_t max_length = 4;
    float initial_temperature = 3.0f;
    float initial_noise_scale = 0.8f;
    float diversity_strength = 4.0f;
    size_t top_k = 100;
    float token_noise_scale = 0.1f;
};

struct TokenizerConfig {
    bool use_subword = true;
    std::string model_path;
    std::vector<std::string> special_tokens;
};

struct TokenPredictionConfig {
    float temperature = 1.0f;
    int top_k = 5;
    float top_p = 0.9f;
    float frequency_penalty = 0.1f;
    float presence_penalty = 0.0f;
    float min_token_prob = 0.05f;
    
    struct CategoryBonus {
        float verb = 0.2f;
        float adjective = 0.2f;
        float noun = 0.3f;
    } category_bonus;
};

struct PathConfig {
    std::string save_directory;
    std::string model_name;
    size_t checkpoint_frequency;
};

/**
 * @brief Configuration class for transformer model architecture and training settings.
 * 
 * This class encapsulates all configuration parameters needed to define and train
 * a transformer model, including architectural choices, optimization settings,
 * and various hyperparameters. The configuration is divided into several categories:
 * - Model architecture parameters
 * - Attention mechanism settings
 * - Training and optimization parameters
 * - File paths and checkpointing
 * - Generation and beam search settings
 */
namespace transformer_runtime {
// Set once at startup, before model construction, when training in LLaMA
// mode. Modules whose update path has no config access (attention, FFN)
// consult this to keep biases frozen at zero, so the exported GGUF (which
// carries no bias tensors) reproduces training math exactly.
extern bool llama_no_bias;

// Adam moment coefficients, set once at startup from config/CLI. Four
// modules (embeddings, lm_head, feed_forward, attention) run their own
// in-module Adam updates and hardcoded 0.9/0.999 until 2026-07-31; they
// now read these. The same runtime-global pattern as llama_no_bias --
// threading through every update_parameters signature was the churn this
// avoids.
extern float adam_beta1;
extern float adam_beta2;
}

struct TransformerConfig {
    // Model parameters
    size_t num_layers;
    size_t num_heads;
    size_t hidden_size;
    size_t intermediate_size;
    size_t head_dim;
    size_t max_seq_length;
    size_t vocab_size = 0;  // Initialize to 0 to detect if not properly set

    // Training parameters
    size_t batch_size = 32;
    size_t min_batch_size = 1;
    size_t max_batch_size = 128;
    size_t num_epochs = 10;
    float initial_lr = 1e-4f;
    float dropout_rate = 0.1f;
    float weight_decay = 0.0f;  // Default to 0 - no weight decay
    
    // Early stopping parameters (0 = disabled; consumed by train_wikitext's
    // validation loop as of 2026-07-30 — previously parsed but only ever
    // read by the dead main.cpp trainer)
    size_t early_stopping_patience = 0;
    float early_stopping_threshold = 0.0f;
    
    // Cross validation parameters
    size_t num_folds = 5;
    
    // Learning rate parameters
    float peak_lr;
    size_t warmup_steps;
    float decay_factor;
    
    // Optimization parameters
    float gradient_clip_threshold = 1.0f;
    float layer_norm_epsilon = 1e-5f;  // Prevent division by zero in LayerNorm
    size_t gradient_accumulation_steps = 1;
    bool use_gradient_checkpointing;
    bool use_fp16;
    size_t memory_pool_size;

    // Attention parameters
    bool use_flash_attention;
    bool use_rope;

    // LLaMA-compatible training mode: RMSNorm instead of LayerNorm, RoPE
    // instead of additive sinusoidal position embeddings, and all biases
    // frozen at zero. Required for lossless GGUF export to llama.cpp-family
    // inference engines (tinyllama.cpp), whose "llama" architecture computes
    // exactly this math.
    // NOTE: kept as the derived legacy flag; the primitives are now selected
    // independently by arch::ArchitectureSpec via the fields below.
    bool llama_mode = false;

    // Decoupled architecture primitives (set via arch::ArchitectureSpec).
    bool use_rms_norm = false;   // RMSNorm (true) vs classic LayerNorm (false)
    bool use_biases = true;      // false freezes all biases at zero (llama)

    bool use_sliding_window;
    size_t window_size;
    bool use_gqa;
    size_t num_kv_heads;  // Add number of key-value heads for GQA

    // Paths
    PathConfig paths;

    // Component configs
    TokenizerConfig tokenizer;
    BeamSearchConfig beam_search;
    TokenPredictionConfig token_prediction;

    // Checkpoint loading
    bool load_from_checkpoint = false;
    std::string checkpoint_to_load;

    // Post-training export. format: "gguf" | "safetensors" | "both" | "none".
    // path is the base output path (format extension appended when missing).
    // Explicit CLI flags (--export-gguf / --export-safetensors) take
    // precedence over this block.
    std::string export_format = "none";
    std::string export_path = "";

    // LoRA fine-tuning (see lora.hpp): freeze the base model, train rank-r
    // adapters on the attention/FFN projections. Pair with --resume to
    // fine-tune a trained checkpoint. Exports carry merged weights.
    bool lora_enabled = false;
    size_t lora_rank = 8;
    float lora_alpha = 16.0f;

    // Optimizer settings
    bool use_momentum = false;
    bool use_adam = false;
    float momentum = 0.9f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    struct MoEConfig {
        bool enabled = false;
        size_t num_experts = 8;
        size_t top_k = 2;
        float aux_loss_coefficient = 0.01f;
    };
    MoEConfig moe;

    // Add method to update vocab size
    void update_vocab_size(size_t new_vocab_size) {
        if (new_vocab_size == 0) {
            throw std::runtime_error("Cannot set vocabulary size to 0");
        }
        vocab_size = new_vocab_size;
    }

    /**
     * @brief Constructs a transformer configuration with default values.
     * @param max_seq_length Maximum sequence length (default: 512)
     * @param hidden_size Dimension of hidden states (default: 768)
     * @param num_layers Number of transformer layers (default: 12)
     * @param num_heads Number of attention heads (default: 12)
     * @param samples_per_iteration Number of samples to process per iteration (default: 32)
     * @param num_epochs Number of training epochs (default: 10)
     */
    TransformerConfig(size_t max_seq_length = 512,
                     size_t hidden_size = 768, size_t num_layers = 12, size_t num_heads = 12,
                     size_t samples_per_iteration = 32, size_t num_epochs = 10);

    /**
     * @brief Compares two configurations for inequality.
     * @param other Configuration to compare against
     * @return true if configurations differ, false otherwise
     */
    bool operator!=(const TransformerConfig& other) const;

    /**
     * @brief Loads configuration from a JSON file.
     */
    void load_from_json(const std::string& path);
};

// JSON serialization declarations
void to_json(nlohmann::json& j, const TokenizerConfig& t);
void from_json(const nlohmann::json& j, TokenizerConfig& t);

void to_json(nlohmann::json& j, const BeamSearchConfig& b);
void from_json(const nlohmann::json& j, BeamSearchConfig& b);

void to_json(nlohmann::json& j, const TokenPredictionConfig& t);
void from_json(const nlohmann::json& j, TokenPredictionConfig& t);

void to_json(nlohmann::json& j, const TransformerConfig& t);
void from_json(const nlohmann::json& j, TransformerConfig& t);

#endif // CONFIG_HPP