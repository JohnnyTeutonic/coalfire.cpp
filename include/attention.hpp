#pragma once
#include "vector.hpp"
#include "matrix.hpp"
#include "cache.hpp"
#include "components.hpp"
#include "tensor.hpp"
#include "config.hpp"
#include "lora.hpp"
#include "tiktoken_tokenizer.hpp"
#ifdef USE_CUDA
#include "../include/cuda/cuda_utils.cuh"
#include "../include/cuda/matrix_ops.cuh"
#endif
#include <optional>
#include <memory>
#include <vector>
#include <random>
#include <string>
using FloatVector = Vector;

// Utility functions for gradient computation
float compute_grad_norm(const Matrix& grad);
size_t count_params(const Matrix& param);

/**
 * @brief Class representing attention masks used in transformer attention mechanisms.
 * 
 * Attention masks are used to control which tokens can attend to which other tokens.
 * This class provides functionality for both causal (autoregressive) masking and
 * padding masking for variable length sequences.
 */
class AttentionMask {
  public:
    Matrix mask;  ///< The actual mask matrix where 0 indicates masked positions

    /**
     * @brief Creates a causal (triangular) mask for autoregressive attention.
     * @param size The sequence length to create the mask for
     * @return An AttentionMask object with a causal mask
     */
    static AttentionMask create_causal_mask(size_t size);

    /**
     * @brief Creates a padding mask for variable length sequences.
     * @param lengths Vector of actual sequence lengths
     * @param max_len Maximum sequence length to pad to
     * @return An AttentionMask object with a padding mask
     */
    static AttentionMask create_padding_mask(const std::vector<int>& lengths, size_t max_len);

    /**
     * @brief Creates a separator-aware attention mask
     * @param tokens Input token sequence
     * @param size Sequence length
     * @return An AttentionMask object that respects separator boundaries
     */
    static AttentionMask create_separator_mask(const std::vector<int>& tokens, size_t size);

    /**
     * @brief Adjusts attention based on separator types
     * @param tokens Input token sequence
     * @return Modified attention mask that respects separator semantics
     */
    void apply_separator_rules(const std::vector<int>& tokens);

    /**
     * @brief Find the position of the last separator in the sequence
     * @param tokens Input token sequence
     * @return Position of the last separator, or -1 if none found
     */
    static int find_separator_position(const std::vector<int>& tokens);

    AttentionMask() = default;
    explicit operator bool() const { return has_mask_; }
    const Matrix& value() const { return mask_; }

    // Constructor taking a mask matrix
    explicit AttentionMask(const Matrix& mask) : mask_(mask), has_mask_(true) {}

    // Add is_masked method
    bool is_masked(size_t i, size_t j) const {
        return mask.empty() ? false : mask(i, j) == 0.0f;
    }

  private:
    Matrix mask_;
    bool has_mask_ = false;
    
    // Helper method to identify separator type
    static char get_separator_type(int token) {
        // Convert token to string to check for separator characters
        std::string token_str = std::to_string(token);
        if (token_str.find('|') != std::string::npos) return '|';
        if (token_str.find('#') != std::string::npos) return '#';
        if (token_str.find('*') != std::string::npos) return '*';
        return '\0';
    }
};

class KVCache;

/**
 * @brief Implementation of Multi-Head Attention mechanism with various optimizations.
 * 
 * This class implements the core attention mechanism used in transformers, with support for:
 * - Multi-head attention with separate Q/K/V projections
 * - Grouped Query Attention (GQA)
 * - Rotary Position Embeddings (RoPE)
 * - Flash Attention optimization
 * - Sliding window attention
 * - Key-Value caching for efficient inference
 */
class MultiHeadAttention {
  public:
    virtual ~MultiHeadAttention() = default;
    MultiHeadAttention() = default;

    /**
     * @brief Constructs a multi-head attention layer with the specified parameters.
     * @param hidden_size_ Size of the input and output tensors
     * @param num_heads_ Number of attention heads
     * @param head_dim_ Dimension of each attention head
     * @param dropout_prob_ Dropout probability
     * @param use_flash_ Whether to use Flash Attention optimization
     * @param use_rope_ Whether to use rotary position embeddings
     * @param use_sliding_window_ Whether to use sliding window attention
     * @param window_size_ Size of the sliding window
     * @param use_gqa_ Whether to use grouped query attention
     * @param num_kv_heads_ Number of key/value heads for GQA
     * @param max_seq_length_ Maximum sequence length supported
     * @param use_fp16_ Whether to use fp16 for computation
     */
    MultiHeadAttention(size_t hidden_size_, size_t num_heads_, size_t head_dim_,
                       float dropout_prob_, bool use_flash_, bool use_rope_,
                       bool use_sliding_window_, size_t window_size_, bool use_gqa_,
                       size_t num_kv_heads_, size_t max_seq_length_, bool use_fp16,
                       bool use_fused_attention = true);

    /**
     * @brief Performs the forward pass of the attention mechanism.
     * @param x Input tensor of shape [batch_size, seq_len, hidden_size]
     * @param mask Attention mask to prevent attending to certain positions
     * @param kv_cache Optional cache of key and value projections for efficient inference
     * @return Output tensor of shape [batch_size, seq_len, hidden_size]
     */
    Matrix forward(const Matrix& x, const AttentionMask& mask,
                   const std::optional<KVCache>& kv_cache = std::nullopt);

    /**
     * @brief Batched forward pass with explicit batch/sequence dimensions.
     * Computes attention per-sequence: O(batch × seq²) instead of O((batch×seq)²)
     * @param x Input tensor [batch_size * seq_len, hidden_size]
     * @param mask Attention mask (block diagonal for batched sequences)
     * @param batch_size Number of sequences in batch
     * @param seq_len Length of each sequence
     * @return Output tensor [batch_size * seq_len, hidden_size]
     */
    Matrix forward_batched(const Matrix& x, const AttentionMask& mask,
                          size_t batch_size, size_t seq_len);

    /**
     * @brief Performs the backward pass to compute gradients.
     * @param grad_output Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @param target Target distribution for loss computation
     * @param config Configuration object containing gradient clipping parameters
     * @return Gradient with respect to the input
     */
    Matrix backward(const Matrix& grad_output, const Matrix& input, 
                   const Matrix& target, const TransformerConfig& config);

    /**
     * @brief Legacy backward pass without configuration.
     * @param grad_output Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @param target Target distribution for loss computation
     * @return Gradient with respect to the input
     */
    Matrix backward(const Matrix& grad_output, const Matrix& input, 
                   const Matrix& target = Matrix()) {
        // Create a default config with standard clipping threshold
        TransformerConfig default_config;
        default_config.gradient_clip_threshold = 5.0f;  // Use default value
        return backward(grad_output, input, target, default_config);
    }

    /**
     * @brief CUDA-accelerated version of the backward pass.
     * @param grad Gradient of the loss with respect to the output
     * @param input Original input tensor
     * @return Gradient with respect to the input
     */
    Matrix backward_cuda(const Matrix& grad, const Matrix& input) const;

    /**
     * @brief Saves the attention layer parameters to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads attention layer parameters from a stream.
     * @param is Input stream to load from
     * @param config Configuration object
     * @return Unique pointer to the loaded attention layer
     */
    static std::unique_ptr<MultiHeadAttention> load(std::istream& is, const TransformerConfig& config);

    /**
     * @brief Gets references to all trainable weight matrices.
     * @return Vector of references to weight matrices
     */
    std::vector<std::reference_wrapper<Matrix>> get_weights() {
        return {
            std::ref(params_.query_weights),
            std::ref(params_.key_weights),
            std::ref(params_.value_weights),
            std::ref(params_.output_weights)
        };
    }

    friend class Transformer;

    FloatVector& getQueryBias() { return params_.query_bias; }
    FloatVector& getKeyBias() { return params_.key_bias; }
    FloatVector& getValueBias() { return params_.value_bias; }
    FloatVector& getOutputBias() { return params_.output_bias; }

    MultiHeadAttention(const MultiHeadAttention& other)
        : params_(other.params_), grads_(other.grads_),
          num_heads(other.num_heads), head_dim(other.head_dim),
          hidden_size(other.hidden_size), dropout_prob(other.dropout_prob),
          use_rope(other.use_rope), use_flash(other.use_flash),
          use_sliding_window(other.use_sliding_window),
          window_size(other.window_size), use_gqa(other.use_gqa),
          num_kv_heads(other.num_kv_heads), max_seq_length(other.max_seq_length),
          use_fp16_(other.use_fp16_) {}

    MultiHeadAttention& operator=(const MultiHeadAttention& other) {
        if (this != &other) {
            params_ = other.params_;
            grads_ = other.grads_;
            num_heads = other.num_heads;
            head_dim = other.head_dim;
            hidden_size = other.hidden_size;
            dropout_prob = other.dropout_prob;
            use_rope = other.use_rope;
            use_flash = other.use_flash;
            use_sliding_window = other.use_sliding_window;
            window_size = other.window_size;
            use_gqa = other.use_gqa;
            num_kv_heads = other.num_kv_heads;
            max_seq_length = other.max_seq_length;
            use_fp16_ = other.use_fp16_;
        }
        return *this;
    }

    // Parameter structure to hold all weights and biases
    struct Parameters {
        Matrix query_weights;
        Matrix key_weights;
        Matrix value_weights;
        Matrix output_weights;
        FloatVector query_bias;
        FloatVector key_bias;
        FloatVector value_bias;
        FloatVector output_bias;
    };

    // Gradient structure to hold all gradients
    struct Gradients {
        Matrix query_grad;      // Gradient for query weights
        Matrix key_grad;        // Gradient for key weights
        Matrix value_grad;      // Gradient for value weights
        Matrix output_grad;     // Gradient for output weights
        FloatVector query_bias_grad;    // Gradient for query bias
        FloatVector key_bias_grad;      // Gradient for key bias
        FloatVector value_bias_grad;    // Gradient for value bias
        FloatVector output_bias_grad;   // Gradient for output bias
    };
    
    // Adam optimizer state (first + second moments, shared step counter).
    // Was first-moment-only SGD-momentum until 2026-07-13: with the LM head
    // on Adam and the core on SGD@2e-4, the core stayed near init (the
    // loss-6.0 plateau).
    struct MomentumState {
        Matrix query_m;
        Matrix key_m;
        Matrix value_m;
        Matrix output_m;
        Matrix query_v;
        Matrix key_v;
        Matrix value_v;
        Matrix output_v;
        FloatVector query_bias_m;
        FloatVector key_bias_m;
        FloatVector value_bias_m;
        FloatVector output_bias_m;
        FloatVector query_bias_v;
        FloatVector key_bias_v;
        FloatVector value_bias_v;
        FloatVector output_bias_v;
        size_t t = 0;
        bool initialized = false;
    };

    Parameters& parameters() { return params_; }
    Gradients& param_gradients() { return grads_; }
    const Parameters& parameters() const { return params_; }
    const Gradients& param_gradients() const { return grads_; }

    /**
     * @brief Computes attention scores between query and key matrices.
     * 
     * Implements the scaled dot-product attention mechanism:
     * scores = softmax(Q * K^T / sqrt(head_dim))
     * 
     * @param Q Query matrix [batch_size * num_heads, seq_len, head_dim]
     * @param K Key matrix [batch_size * num_kv_heads, seq_len, head_dim]
     * @param mask Attention mask to prevent attending to certain positions
     * @return Attention scores [batch_size * num_heads, seq_len, seq_len]
     */
    Matrix compute_attention_scores(const Matrix& Q, const Matrix& K, const AttentionMask& mask);

    /**
     * @brief Initialize the attention weights and biases
     * 
     * Initializes projection matrices with Xavier/Glorot initialization
     * and biases with small non-zero values for better training stability.
     */
    void initialize_weights();

    // Update accessor methods to use Parameters structure
    Matrix& get_query_weights() { return params_.query_weights; }
    Matrix& get_key_weights() { return params_.key_weights; }
    Matrix& get_value_weights() { return params_.value_weights; }
    Matrix& get_output_weights() { return params_.output_weights; }
    
    // Add const versions
    const Matrix& get_query_weights() const { return params_.query_weights; }
    const Matrix& get_key_weights() const { return params_.key_weights; }
    const Matrix& get_value_weights() const { return params_.value_weights; }
    const Matrix& get_output_weights() const { return params_.output_weights; }
    
    // GGUF export aliases
    const Matrix& getQueryWeights() const { return params_.query_weights; }
    const Matrix& getKeyWeights() const { return params_.key_weights; }
    const Matrix& getValueWeights() const { return params_.value_weights; }
    const Matrix& getOutputWeights() const { return params_.output_weights; }

    // Add the direct matrix version of forward
    Matrix forward(const Matrix& input, const Matrix& attention_mask);

    void reset_state() {
        // Clear all cached computations
        cached_keys = Matrix();
        cached_values = Matrix();
        cached_attention_weights = Matrix();
        cached_query_layer = Matrix();
        cached_key_layer = Matrix();
        cached_value_layer = Matrix();
    }

    /**
     * @brief Creates a causal attention mask.
     * @param seq_length Length of the sequence
     * @return Matrix containing the causal mask
     */
    Matrix create_causal_mask(size_t seq_length) const;
    
    /**
     * @brief Performs forward pass of multi-head attention.
     * @param input Input tensor
     * @param attention_mask Optional attention mask
     * @return Output tensor after attention
     */
    Matrix forward(const Matrix& input, const AttentionMask& attention_mask);

    void set_training(bool mode) { training = mode; }

    // Add the missing update_parameters declaration
    void update_parameters(float learning_rate);

    // Separator token handling
    const std::vector<std::string> separators = {"|", "#", "*"};  // All possible separators
    size_t find_separator_position(const std::vector<int>& tokens) const {
        if (!tokenizer_) {
            throw std::runtime_error("Tokenizer not set in MultiHeadAttention");
        }
        
        if (tokens.empty()) {
            return 0;
        }
        
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string token_text = tokenizer_->decode({tokens[i]});
            if (std::find(separators.begin(), separators.end(), token_text) != separators.end()) {
                return i;
            }
        }
        return tokens.size();
    }

    // Create a separator-aware attention mask
    AttentionMask create_separator_mask(const std::vector<int>& tokens) const {
        if (tokens.empty()) {
            return AttentionMask(Matrix(0, 0));  // Return empty mask for empty tokens
        }
        
        size_t seq_len = tokens.size();
        size_t sep_pos = find_separator_position(tokens);
        
        // Create mask matrix
        Matrix mask_matrix(seq_len, seq_len, 0.0f);
        
        // For positions before separator: can attend to all previous tokens
        // For positions after separator: can only attend to tokens before separator
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                if (i < sep_pos) {
                    // Before separator: standard causal masking
                    mask_matrix(i, j) = (j <= i) ? 1.0f : 0.0f;
                } else {
                    // After separator: can only attend to tokens before separator
                    mask_matrix(i, j) = (j < sep_pos) ? 1.0f : 0.0f;
                }
            }
        }
        
        return AttentionMask(mask_matrix);
    }
    
    // Adjust positional encodings based on separator position
    Matrix adjust_position_encodings(const Matrix& pos_enc, const std::vector<int>& tokens) const {
        Matrix adjusted = pos_enc;
        size_t sep_pos = find_separator_position(tokens);
        
        // For tokens after separator, use positions relative to separator
        for (size_t i = sep_pos + 1; i < tokens.size(); i++) {
            for (size_t j = 0; j < pos_enc.cols(); j++) {
                // Reset position to be relative to separator
                size_t relative_pos = i - sep_pos;
                adjusted(i, j) = pos_enc(relative_pos, j);
            }
        }
        
        return adjusted;
    }

    // Update tokenizer methods
    void set_tokenizer(const std::shared_ptr<TiktokenTokenizer>& tokenizer) {
        if (!tokenizer) {
            throw std::runtime_error("Cannot set null tokenizer");
        }
        tokenizer_ = tokenizer;
    }

    std::shared_ptr<TiktokenTokenizer> get_tokenizer() const {
        return tokenizer_;
    }

    /**
     * @brief Resets the key-value cache used for attention
     */
    void reset_cache() {
        key_cache = Matrix();    // Reset to empty matrix
        value_cache = Matrix();  // Reset to empty matrix
        cache_position = 0;      // Reset position counter
    }

  private:
    Parameters params_;
    Gradients grads_;
    MomentumState momentum_;  // For stable training with momentum
    // LoRA adapters (active only when lora::settings().enabled)
    lora::LoRAAdapter lora_q_, lora_k_, lora_v_, lora_o_;
    
    // Configuration parameters
    size_t num_heads;         ///< Number of attention heads
    size_t head_dim;         ///< Dimension of each attention head
    size_t hidden_size;       ///< Size of input and output tensors
    float dropout_prob;       ///< Dropout probability
    bool use_rope;           ///< Whether to use rotary position embeddings
    bool use_flash;          ///< Whether to use Flash Attention
    bool use_sliding_window; ///< Whether to use sliding window attention
    size_t window_size;      ///< Size of attention window
    bool use_gqa;           ///< Whether to use grouped query attention
    size_t num_kv_heads;     ///< Number of key/value heads for GQA
    size_t max_seq_length;   ///< Maximum sequence length supported
    bool use_fp16_;         ///< Whether to use FP16 computation
    bool use_fused_attention_; ///< Whether to use fused attention kernels

    // Helper methods for attention computation
    void print_matrix_stats(const std::string& name, const Matrix& mat) const;
    Matrix softmax_with_temperature(const Matrix& input, float temperature) const;
    Vector softmax_with_temperature(const Vector& input, float temperature) const;
    void apply_stable_softmax(Matrix& x) const;

    // Cache variables for training (backward pass)
    Matrix cached_keys;              ///< Cached key projections
    Matrix cached_values;            ///< Cached value projections
    Matrix cached_attention_weights; ///< Cached attention scores
    Matrix cached_query_layer;       ///< Cached query layer output
    Matrix cached_key_layer;         ///< Cached key layer output
    Matrix cached_value_layer;       ///< Cached value layer output
    Matrix cached_attn_output;       ///< Cached attention output BEFORE output projection (critical for backward!)

    // Batched-forward geometry, cached for the exact backward pass.
    // cached_batched_valid_ is true only when forward_batched populated the
    // caches in its [batch*seq x hidden] layout; backward falls back to the
    // legacy approximate path otherwise.
    size_t cached_batch_size_ = 0;
    size_t cached_seq_len_ = 0;
    bool cached_batched_valid_ = false;

    // Static RoPE cache shared across all instances
    static Matrix cos_cached;
    static Matrix sin_cached;
    static bool rope_cache_initialized;

    // Private helper methods
    Vector apply_rope(const Vector& x, size_t position) const;

    /// Apply RoPE rotation in-place to every row of a [batch*seq x heads*head_dim]
    /// matrix (row r -> position r % seq_len). inverse=true applies the
    /// transpose rotation (rotate by -theta), used in the backward pass.
    /// heads = 0 means num_heads; K at KV width passes kv_heads().
    void rotate_rows_rope(Matrix& m, size_t seq_len, bool inverse, size_t heads = 0) const;

  public:
    // ---- Grouped-query attention (GQA) ----
    // With use_gqa and 0 < num_kv_heads < num_heads (num_heads a multiple of
    // num_kv_heads), K and V project to kv_dim() = num_kv_heads * head_size
    // columns, and query head h reads KV head h / (num_heads / num_kv_heads),
    // the contiguous grouping HF Llama and llama.cpp use. Otherwise attention
    // is full multi-head: kv_heads() == num_heads, kv_dim() == hidden_size.
    size_t kv_heads() const {
        return (use_gqa && num_kv_heads > 0 && num_kv_heads < num_heads) ? num_kv_heads : num_heads;
    }
    size_t kv_dim() const { return kv_heads() * (hidden_size / num_heads); }
    bool gqa_active() const { return kv_heads() != num_heads; }

  private:
    /// [rows x kv_dim] -> [rows x hidden]: column h*hd + d of the result is
    /// column (h / group)*hd + d of kv. Returns kv unchanged when !gqa_active().
    Matrix expand_kv_heads(const Matrix& kv) const;
    /// The adjoint of expand_kv_heads: [rows x hidden] -> [rows x kv_dim],
    /// summing each KV head's group of query-head columns (h ascending).
    Matrix reduce_kv_heads(const Matrix& full) const;
    Matrix flash_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                           const AttentionMask& mask) const;
    Matrix standard_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                              const AttentionMask& mask);
    Tensor reshape_for_attention(const Matrix& x, size_t batch_size, size_t num_heads,
                                 size_t seq_len, size_t head_size) const;

    // Change the inline definition to just a declaration
    Matrix reshape_from_attention(const Tensor& x, size_t seq_len, size_t hidden_size) const;

    Tensor compute_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                             const AttentionMask& mask, size_t batch_size, size_t num_heads,
                             size_t seq_len, size_t head_size);

    void validate_dimensions(const Matrix& grad_output, const Matrix& input,
                             const Matrix& target_dist) const {
        if (grad_output.cols() != hidden_size) {
            throw std::runtime_error("grad_output.cols (" + std::to_string(grad_output.cols()) +
                                     ") != hidden_size (" + std::to_string(hidden_size) + ")");
        }
        if (input.cols() != hidden_size) {
            throw std::runtime_error("input.cols (" + std::to_string(input.cols()) +
                                     ") != hidden_size (" + std::to_string(hidden_size) + ")");
        }
    }

    // Update method declarations to match implementations
    Matrix compute_query_gradients(const Matrix& grad, const Matrix& input);
    Matrix compute_key_gradients(const Matrix& grad, const Matrix& input);
    Matrix compute_value_gradients(const Matrix& grad, const Matrix& input);
    Matrix combine_gradients(const Matrix& d_query, const Matrix& d_key, const Matrix& d_value);
    Matrix compute_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                           const AttentionMask& mask) const;

#ifdef USE_CUDA
    // Fused attention kernel method
    Matrix forward_fused(const Matrix& input, const AttentionMask& mask);
#endif

    // Helper method for safe matrix multiplication
    Matrix safe_matmul(const Matrix& A, const Matrix& B) {
        if (A.cols() != B.rows()) {
            throw std::runtime_error("Matrix multiplication dimension mismatch: " +
                                     std::to_string(A.cols()) + " != " + std::to_string(B.rows()));
        }
        return matmul(A, B);
    }

    void apply_mask(Matrix& scores, const Matrix& mask) const {
        std::cout << "Applying mask - scores shape: " << scores.rows() << "x" << scores.cols()
                  << ", mask shape: " << mask.rows() << "x" << mask.cols() << std::endl;

        if (scores.rows() != mask.rows() || scores.cols() != mask.cols()) {
            throw std::runtime_error(
                "Mask dimensions don't match attention scores: scores(" +
                std::to_string(scores.rows()) + "," + std::to_string(scores.cols()) + ") != mask(" +
                std::to_string(mask.rows()) + "," + std::to_string(mask.cols()) + ")");
        }

        for (size_t i = 0; i < scores.rows(); i++) {
            for (size_t j = 0; j < scores.cols(); j++) {
                if (mask(i, j) == 0.0f) {
                    scores(i, j) = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    // Move implementation to source file
    // void apply_stable_softmax(Matrix& x) const;  // Remove this duplicate declaration

    // Add these new methods to handle Tensors directly
    void apply_mask(Tensor& scores, const Matrix& mask) const {
        Matrix scores_mat = scores.to_matrix();
        apply_mask(scores_mat, mask);
        scores = Tensor(scores_mat,
                        {scores.dims()[0], scores.dims()[1], scores.dims()[2], scores.dims()[3]});
    }

    void apply_stable_softmax(Tensor& scores) const {
        Matrix scores_mat = scores.to_matrix();
        apply_stable_softmax(scores_mat);
        scores = Tensor(scores_mat,
                        {scores.dims()[0], scores.dims()[1], scores.dims()[2], scores.dims()[3]});
    }

    // RoPE helper functions
    /**
     * @brief Applies rotary position embeddings to Q/K matrices.
     * 
     * RoPE improves the model's ability to capture positional information
     * by encoding positions through rotation in vector space.
     * 
     * @param matrix Input Q or K matrix to apply RoPE to
     * @param offset Position offset for cached generation
     */
    void apply_rotary_embeddings(Matrix& matrix, size_t offset = 0);

    /**
     * @brief Initializes or updates RoPE caches.
     * @param max_seq_len Length of sequence to cache embeddings for
     * @param dim_idx Dimension index for cached generation
     */
    void initialize_rope_cache(size_t max_seq_len, size_t dim_idx);
    float get_cos_cached(size_t pos, size_t dim_idx) const;
    float get_sin_cached(size_t pos, size_t dim_idx) const;

    // Helper method for computing outer product
    Matrix outer_product(const Vector& a, const Vector& b) const {
        Matrix result(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                result(i, j) = a[i] * b[j];
            }
        }
        return result;
    }

    // Add static initialization method
    static void initialize_static_rope_cache(size_t max_seq_len, size_t dim, size_t num_heads);

    // Add missing member variables
    bool training = false;
    std::mt19937 gen{std::random_device{}()};  // Random number generator

    // Add helper methods
    Matrix softmax(const Matrix& input) const {
        Matrix output = input;
        for (size_t i = 0; i < input.rows(); i++) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < input.cols(); j++) {
                max_val = std::max(max_val, input(i, j));
            }
            
            float sum_exp = 0.0f;
            for (size_t j = 0; j < input.cols(); j++) {
                output(i, j) = std::exp(input(i, j) - max_val);
                sum_exp += output(i, j);
            }
            
            for (size_t j = 0; j < input.cols(); j++) {
                output(i, j) /= sum_exp;
            }
        }
        return output;
    }

    Matrix elementwise_multiply(const Matrix& a, const Matrix& b) const {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw std::runtime_error("Matrix dimensions must match for elementwise multiplication");
        }
        Matrix result(a.rows(), a.cols());
        for (size_t i = 0; i < a.rows(); i++) {
            for (size_t j = 0; j < a.cols(); j++) {
                result(i, j) = a(i, j) * b(i, j);
            }
        }
        return result;
    }

    Matrix reshape(const Matrix& input, size_t batch_size, size_t num_heads, 
                  size_t seq_length, size_t head_dim) const {
        if (input.rows() * input.cols() != batch_size * num_heads * seq_length * head_dim) {
            throw std::runtime_error("Invalid dimensions for reshape");
        }
        Matrix result(batch_size * num_heads * seq_length, head_dim);
        // Copy data with proper reshaping
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t h = 0; h < num_heads; h++) {
                for (size_t s = 0; s < seq_length; s++) {
                    for (size_t d = 0; d < head_dim; d++) {
                        size_t new_idx = (b * num_heads * seq_length + h * seq_length + s);
                        size_t old_idx = b * seq_length + s;
                        result(new_idx, d) = input(old_idx, h * head_dim + d);
                    }
                }
            }
        }
        return result;
    }

    std::shared_ptr<TiktokenTokenizer> tokenizer_;  // Tokenizer member
    
    // Helper method for adding noise to vectors
    void add_random_noise(Vector& vec, std::mt19937& gen) const {
        std::normal_distribution<float> noise_dist(0.0f, 0.1f);
        for (size_t i = 0; i < vec.size(); i++) {
            vec[i] += noise_dist(gen);
        }
    }

    // Add cache-related members
    Matrix key_cache;                  ///< Cache for key projections
    Matrix value_cache;                ///< Cache for value projections
    size_t cache_position = 0;         ///< Current position in the cache
};

// Add sliding window attention
class SlidingWindowAttention : public MultiHeadAttention {
  private:
    size_t window_size;
    bool use_local_attention;
    size_t head_dim;  // Add head dimension member

    /**
     * @brief Process attention for a single window
     * @param Q Query matrix for current position
     * @param K Key matrix for local window
     * @param V Value matrix for local window
     * @param output Output matrix to update
     * @param pos Current position in sequence
     * @param window_start Start position of current window
     */
    void process_attention_window(const Matrix& Q, const Matrix& K, const Matrix& V,
                                Matrix& output, size_t pos, size_t window_start);

  public:
    /**
     * @brief Construct a sliding window attention layer
     * @param window_size_ Size of the attention window
     * @param head_dim_ Dimension of each attention head
     * @param use_local_attention_ Whether to use local attention only
     */
    explicit SlidingWindowAttention(size_t window_size_ = 512, size_t head_dim_ = 64,
                                  bool use_local_attention_ = true)
        : MultiHeadAttention(), window_size(window_size_), head_dim(head_dim_),
          use_local_attention(use_local_attention_) {}

    /**
     * @brief Compute attention with local sliding windows
     * @param Q Query matrix
     * @param K Key matrix
     * @param V Value matrix
     * @return Output matrix after local attention
     */
    Matrix compute_local_attention(const Matrix& Q, const Matrix& K, const Matrix& V);
};

// Add sparse attention
class SparseAttention : public MultiHeadAttention {
  private:
    std::vector<std::pair<int, int>> attention_patterns;
    float sparsity_threshold;

    Matrix compute_sparse_attention(const Matrix& Q, const Matrix& K, const Matrix& V) {
        // Implement sparse attention using custom patterns
        return Matrix();
    }
};