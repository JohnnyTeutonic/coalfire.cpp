#include "../include/attention.hpp"
#include "../include/repro_reduce.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/cuda/cuda_utils.cuh"
#include "../include/cuda/cuda_launch.cuh"
#include "../include/cuda/attention_ops.cuh"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/fused_attention_kernels.cuh"
#endif
#include <cstdlib>
#include "../include/gqa.hpp"
#include "../include/performance_metrics.hpp"
#include "../include/transformer.hpp"
#include "../include/config.hpp"
#include "../include/half_precision.hpp"
#include "../include/scope_logger.hpp"
#include "../include/gradient_diagnostics.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

// Disable verbose logging for production builds (MASSIVELY speeds up training)
// Uncomment the line below if you need debug output:
// #define DEBUG_ATTENTION_LOGGING

// Null stream that discards all output (MSVC-compatible)
struct NullStream {
    template<typename T>
    NullStream& operator<<(const T&) { return *this; }
    NullStream& operator<<(std::ostream& (*)(std::ostream&)) { return *this; }
};
static NullStream nullstream;

#ifndef DEBUG_ATTENTION_LOGGING
    // Redirect all debug logging to no-op
    #define DEBUG_LOG(x) do {} while(0)
    #define DEBUG_COUT nullstream
    #define DEBUG_CERR nullstream
#else
    #define DEBUG_LOG(x) x
    #define DEBUG_COUT std::cout
    #define DEBUG_CERR std::cerr
#endif

extern PerformanceMetrics metrics;

// Initialize static members
Matrix MultiHeadAttention::cos_cached;
Matrix MultiHeadAttention::sin_cached;
bool MultiHeadAttention::rope_cache_initialized = false;

void MultiHeadAttention::initialize_static_rope_cache(size_t max_seq_len, size_t dim, size_t num_heads) {
    if (rope_cache_initialized) {
        return;  // Cache already initialized
    }

    DEBUG_COUT << "Initializing static RoPE cache:" << std::endl;
    DEBUG_COUT << "- max_seq_len: " << max_seq_len << std::endl;
    DEBUG_COUT << "- dim: " << dim << std::endl;
    DEBUG_COUT << "- num_heads: " << num_heads << std::endl;

    cos_cached = Matrix(max_seq_len, dim * num_heads);
    sin_cached = Matrix(max_seq_len, dim * num_heads);

    DEBUG_COUT << "Created cache matrices:" << std::endl;
    DEBUG_COUT << "- cos_cached: " << cos_cached.rows() << "x" << cos_cached.cols() << std::endl;
    DEBUG_COUT << "- sin_cached: " << sin_cached.rows() << "x" << sin_cached.cols() << std::endl;

    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t i = 0; i < dim; i++) {
                size_t idx = h * dim + i;
                float theta = std::pow(10000.0f, -2.0f * i / dim);
                cos_cached(pos, idx) = std::cos(pos * theta);
                sin_cached(pos, idx) = std::sin(pos * theta);
            }
        }
    }

    rope_cache_initialized = true;
}

MultiHeadAttention::MultiHeadAttention(size_t hidden_size_, size_t num_heads_, size_t head_dim_,
                                     float dropout_prob_, bool use_flash_, bool use_rope_,
                                     bool use_sliding_window_, size_t window_size_, bool use_gqa_,
                                     size_t num_kv_heads_, size_t max_seq_length_, bool use_fp16,
                                     bool use_fused_attention)
    : num_heads(num_heads_), head_dim(head_dim_), hidden_size(hidden_size_),
      dropout_prob(dropout_prob_), use_flash(use_flash_), use_rope(use_rope_),
      use_sliding_window(use_sliding_window_), window_size(window_size_),
      use_gqa(use_gqa_), num_kv_heads(num_kv_heads_), max_seq_length(max_seq_length_),
      use_fp16_(use_fp16), use_fused_attention_(use_fused_attention) {

    DEBUG_LOG(SCOPE_LOG());
    DEBUG_COUT << "\n=== MultiHeadAttention::constructor START ===" << std::endl;

    // Initialize matrices with correct dimensions
    params_.query_weights = Matrix(hidden_size_, hidden_size_);
    params_.key_weights = Matrix(hidden_size_, hidden_size_);
    params_.value_weights = Matrix(hidden_size_, hidden_size_);
    params_.output_weights = Matrix(hidden_size_, hidden_size_);

    // Initialize bias vectors - size should match output dimension of weights
    params_.query_bias = FloatVector(hidden_size_);  // Match query_weights.cols()
    params_.key_bias = FloatVector(hidden_size_);    // Match key_weights.cols()
    params_.value_bias = FloatVector(hidden_size_);  // Match value_weights.cols()
    params_.output_bias = FloatVector(hidden_size_);

    // Initialize gradients
    grads_.query_grad = Matrix(hidden_size_, hidden_size_);
    grads_.key_grad = Matrix(hidden_size_, hidden_size_);
    grads_.value_grad = Matrix(hidden_size_, hidden_size_);
    grads_.output_grad = Matrix(hidden_size_, hidden_size_);
    grads_.query_bias_grad = FloatVector(hidden_size_);  // Match query_bias
    grads_.key_bias_grad = FloatVector(hidden_size_);    // Match key_bias
    grads_.value_bias_grad = FloatVector(hidden_size_);  // Match value_bias
    grads_.output_bias_grad = FloatVector(hidden_size_);

    // Print configuration
    DEBUG_COUT << "Configuration:" << std::endl;
    DEBUG_COUT << "- Hidden size: " << hidden_size << std::endl;
    DEBUG_COUT << "- Number of heads: " << num_heads << std::endl;
    DEBUG_COUT << "- Head dimension: " << head_dim << std::endl;
    DEBUG_COUT << "- Dropout probability: " << dropout_prob << std::endl;
    DEBUG_COUT << "- Use flash attention: " << std::boolalpha << use_flash << std::endl;
    DEBUG_COUT << "- Use RoPE: " << use_rope << std::endl;
    DEBUG_COUT << "- Use sliding window: " << use_sliding_window << std::endl;
    DEBUG_COUT << "- Window size: " << window_size << std::endl;
    DEBUG_COUT << "- Use GQA: " << use_gqa << std::endl;
    DEBUG_COUT << "- Number of KV heads: " << num_kv_heads << std::endl;

    // Validate input dimensions
    DEBUG_COUT << "\nValidating dimensions..." << std::endl;
    if (hidden_size == 0 || num_heads == 0 || head_dim == 0) {
        throw std::runtime_error("Invalid dimensions: hidden_size=" + std::to_string(hidden_size) +
                               ", num_heads=" + std::to_string(num_heads) +
                               ", head_dim=" + std::to_string(head_dim));
    }

    if (hidden_size % num_heads != 0) {
        throw std::runtime_error("hidden_size must be divisible by num_heads");
    }
    DEBUG_COUT << "Dimension validation passed" << std::endl;

    // Initialize weights with Xavier/Glorot initialization
    DEBUG_COUT << "\nInitializing weights..." << std::endl;
    initialize_weights();

    // Initialize RoPE cache if needed and not already initialized
    if (use_rope) {
        initialize_static_rope_cache(max_seq_length, head_dim, num_heads);
    }

    DEBUG_COUT << "=== MultiHeadAttention::constructor END ===\n" << std::endl;
}

Vector MultiHeadAttention::apply_rope(const Vector& x, size_t position) const {
    Vector result = x;
    // Apply rotary position embeddings
    for (size_t i = 0; i < x.size(); i += 2) {
        if (i + 1 >= x.size()) {
            DEBUG_COUT << "Breaking at i=" << i << " (odd size)" << std::endl;
            break;
        }

        float x_i = x[i];
        float x_i1 = x[i + 1];

        // Each pair of elements belongs to a specific head and position within that head
        size_t pair_idx = i / 2; // Index of the current pair
        size_t head_idx =
            pair_idx / (head_dim / 2); // Which head (using half head_dim since we process pairs)
        size_t dim_idx = pair_idx % (head_dim / 2); // Position within head (using half head_dim)
        size_t cache_idx = head_idx * head_dim + dim_idx; // Correct: direct mapping to cache

        try {
            float cos_theta = get_cos_cached(position, cache_idx);
            float sin_theta = get_sin_cached(position, cache_idx);

            result[i] = x_i * cos_theta - x_i1 * sin_theta;
            result[i + 1] = x_i * sin_theta + x_i1 * cos_theta;
        } catch (const std::exception& e) {
            DEBUG_COUT << "Error in RoPE application:" << std::endl;
            DEBUG_COUT << "- Error message: " << e.what() << std::endl;
            DEBUG_COUT << "- Current indices: pos=" << position << ", cache_idx=" << cache_idx
                      << ", i=" << i << std::endl;
            throw;
        }
    }

    return result;
}

Matrix MultiHeadAttention::flash_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                         const AttentionMask& mask) const {
    DEBUG_LOG(SCOPE_LOG());
    DEBUG_COUT << "=== MultiHeadAttention::flash_attention START ===\n";
    
    const size_t seq_len = Q.rows();
    const size_t head_dim = Q.cols();
    Matrix O(seq_len, head_dim, 0.0f);
    std::vector<float> m(seq_len, -std::numeric_limits<float>::infinity());
    std::vector<float> L(seq_len, 0.0f);
    
    const float scale = 1.0f / std::sqrt(head_dim);  // Scaling factor for better numerical stability
    const float attn_clip = 5.0f;  // Clip attention scores
    
    // Process in blocks for better cache efficiency
    const size_t block_size = 64;  // Adjust based on cache size
    for (size_t kr = 0; kr < seq_len; kr += block_size) {
        const size_t k_end = std::min(kr + block_size, seq_len);
        
        // Compute attention scores for this block (MSVC: collapse ignored, loop vars must be signed int)
        Matrix S(seq_len, k_end - kr);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < static_cast<int>(seq_len); i++) {
            for (int j = static_cast<int>(kr); j < static_cast<int>(k_end); j++) {
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    score += Q(i, d) * K(j, d);
                }
                score *= scale;
                
                // Apply mask if needed
                if (mask.is_masked(i, j)) {
                    score = -std::numeric_limits<float>::infinity();
                }
                
                // Clip attention scores for stability
                score = std::max(-attn_clip, std::min(attn_clip, score));
                S(i, j - kr) = score;
            }
        }
        
        // Update output with numerically stable softmax (MSVC: loop vars must be signed int)
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(seq_len); i++) {
            float mi = m[i];
            float li = L[i];
            
            // Find max for numerical stability
            for (int j = 0; j < static_cast<int>(S.cols()); j++) {
                mi = std::max(mi, S(i, j));
            }
            
            // Compute softmax with improved stability
            std::vector<float> exp_scores(S.cols());
            float sum_exp = 0.0f;
            
            for (int j = 0; j < static_cast<int>(S.cols()); j++) {
                float e = std::exp(S(i, j) - mi);
                exp_scores[j] = e;
                sum_exp += e;
            }
            
            // Update output with normalized attention scores
            for (int j = 0; j < static_cast<int>(S.cols()); j++) {
                float attn_prob = exp_scores[j] / (sum_exp + 1e-6f);  // Add small epsilon
                for (int d = 0; d < static_cast<int>(head_dim); d++) {
                    O(i, d) += attn_prob * V(j + kr, d);
                }
            }
            
            m[i] = mi;
            L[i] = sum_exp;
        }
    }
    
    // Final normalization of output (MSVC: collapse ignored, loop vars must be signed int)
    const float output_clip = 3.0f;  // Clip output values
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < static_cast<int>(seq_len); i++) {
        for (int d = 0; d < static_cast<int>(head_dim); d++) {
            O(i, d) = std::max(-output_clip, std::min(output_clip, O(i, d)));
        }
    }
    
    DEBUG_COUT << "=== MultiHeadAttention::flash_attention END ===\n";
    return O;
}

Matrix MultiHeadAttention::forward(const Matrix& input, const AttentionMask& mask,
                                 const std::optional<KVCache>& kv_cache) {
    size_t batch_size = input.rows();
    size_t seq_length = batch_size;  // Each row represents a token in the sequence
        
    // Verify dimensions
    if (input.cols() != hidden_size) {
        throw std::runtime_error("Input dimension mismatch: expected hidden_size=" + 
                               std::to_string(hidden_size) + ", got " + 
                               std::to_string(input.cols()));
    }

#ifdef USE_CUDA
    // Use fused attention kernel for better performance when available
    if (use_fused_attention_ && !kv_cache) {
        return forward_fused(input, mask);
    }
#endif

    // Project input to Q, K, V using matmul
    Matrix Q = matmul(input, params_.query_weights);
    Matrix K = matmul(input, params_.key_weights);
    Matrix V = matmul(input, params_.value_weights);
    // Add biases
    for (size_t i = 0; i < Q.rows(); i++) {
        for (size_t j = 0; j < Q.cols(); j++) {
            Q(i, j) += params_.query_bias[j];
            K(i, j) += params_.key_bias[j];
            V(i, j) += params_.value_bias[j];
        }
    }
    
    // CRITICAL: Cache Q, K, V for backward pass
    cached_query_layer = Q;
    cached_key_layer = K;
    cached_value_layer = V;

    // Debug Q,K,V statistics before attention
    DEBUG_COUT << "\nAttention component statistics before attention:" << std::endl;
    print_matrix_stats("Query", Q);
    print_matrix_stats("Key", K);
    print_matrix_stats("Value", V);
        
    // If KV cache is provided, use cached values
    if (kv_cache && !kv_cache->empty()) {
        K = kv_cache->key_cache;
        V = kv_cache->value_cache;
    }
    
    // ========== PER-HEAD ATTENTION ==========
    // Compute attention for each head independently to avoid O((seq*heads)^2) memory
    // This allows batch_size * seq_len up to sqrt(1B) / heads ≈ 4000 positions per head
    const size_t head_size = hidden_size / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    
    // Output matrix [seq_length x hidden_size]
    Matrix output(seq_length, hidden_size);
    
    // Store attention weights for backward: [num_heads * seq_length x seq_length]
    // Each block of seq_length rows is one head's attention weights
    cached_attention_weights = Matrix(num_heads * seq_length, seq_length);
    
    // Get attention mask if present
    bool has_mask = !mask.mask.empty();
    const Matrix& attention_mask = mask.mask;
    
    // Process each head independently - keeps attention scores at [seq x seq]
    #pragma omp parallel for
    for (int h = 0; h < static_cast<int>(num_heads); h++) {
        // Extract Q_h, K_h, V_h for this head: [seq_length x head_size]
        Matrix Q_h(seq_length, head_size);
        Matrix K_h(seq_length, head_size);
        Matrix V_h(seq_length, head_size);
        
        // Extract head slices from Q, K, V [seq_length x hidden_size]
        // Q[s, h*head_size : (h+1)*head_size] -> Q_h[s, :]
        for (size_t s = 0; s < seq_length; s++) {
            for (size_t d = 0; d < head_size; d++) {
                Q_h(s, d) = Q(s, h * head_size + d);
                K_h(s, d) = K(s, h * head_size + d);
                V_h(s, d) = V(s, h * head_size + d);
            }
        }
        
        // Apply RoPE if enabled (per-sequence position)
        if (use_rope) {
            for (size_t s = 0; s < seq_length; s++) {
                // Note: For batched input, position should wrap per sequence
                // TODO: Track actual positions per sequence in batch
                Vector q_row = Q_h.row(s);
                Vector k_row = K_h.row(s);
                q_row = apply_rope(q_row, s);  // Using flat position for now
                k_row = apply_rope(k_row, s);
                for (size_t d = 0; d < head_size; d++) {
                    Q_h(s, d) = q_row[d];
                    K_h(s, d) = k_row[d];
                }
            }
        }
        
        // Compute attention scores: Q_h @ K_h^T -> [seq_length x seq_length]
        Matrix K_h_T = K_h.transpose();
        Matrix scores_h = matmul(Q_h, K_h_T);
        scores_h *= scale;
        
        // Apply attention mask (same mask for all heads)
        if (has_mask) {
            for (size_t i = 0; i < seq_length; i++) {
                for (size_t j = 0; j < seq_length; j++) {
                    // Mask value < -1e8 or == 0 means "don't attend"
                    if (attention_mask(i, j) < -1e8f) {
                        scores_h(i, j) = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
        
        // Apply softmax per row (numerically stable)
        for (size_t i = 0; i < seq_length; i++) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < seq_length; j++) {
                max_val = std::max(max_val, scores_h(i, j));
            }
            // Handle all-masked rows (max_val is -inf)
            if (max_val == -std::numeric_limits<float>::infinity()) {
                // Uniform distribution over all positions (avoid NaN)
                for (size_t j = 0; j < seq_length; j++) {
                    scores_h(i, j) = 1.0f / seq_length;
                }
            } else {
                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_length; j++) {
                    scores_h(i, j) = std::exp(scores_h(i, j) - max_val);
                    sum_exp += scores_h(i, j);
                }
                float inv_sum = 1.0f / (sum_exp + 1e-10f);
                for (size_t j = 0; j < seq_length; j++) {
                    scores_h(i, j) *= inv_sum;
                }
            }
        }
        
        // Store attention weights for this head (for backward pass)
        for (size_t i = 0; i < seq_length; i++) {
            for (size_t j = 0; j < seq_length; j++) {
                cached_attention_weights(h * seq_length + i, j) = scores_h(i, j);
            }
        }
        
        // Compute attention output: scores_h @ V_h -> [seq_length x head_size]
        Matrix attn_h = matmul(scores_h, V_h);
        
        // Write to output (thread-safe: each head writes to different columns)
        for (size_t s = 0; s < seq_length; s++) {
            for (size_t d = 0; d < head_size; d++) {
                output(s, h * head_size + d) = attn_h(s, d);
            }
        }
    }
    
    // CRITICAL: Cache attention output BEFORE projection for backward pass
    cached_attn_output = output;
    
    // Project output
    Matrix final_output = matmul(output, params_.output_weights);
    
    // Add output bias
    for (size_t i = 0; i < final_output.rows(); i++) {
        for (size_t j = 0; j < final_output.cols(); j++) {
            final_output(i, j) += params_.output_bias[j];
        }
    }
    
    return final_output;
}

// ========== BATCHED FORWARD: O(batch × seq²) instead of O((batch×seq)²) ==========
// Optimized: Do QKV projections in ONE batch, then per-sequence attention
void MultiHeadAttention::rotate_rows_rope(Matrix& m, size_t seq_len, bool inverse) const {
    // Adjacent-pairing RoPE, matching apply_rope() and tinyllama.cpp's GGUF
    // path: within each head, pair (2i, 2i+1) rotates by theta_i = pos *
    // 10000^(-2i/head_dim). The static cache stores cos/sin at
    // (pos, head*head_dim + i) for i in [0, head_dim/2).
    const float sign = inverse ? -1.0f : 1.0f;
    #pragma omp parallel for
    for (int r = 0; r < static_cast<int>(m.rows()); r++) {
        const size_t pos = static_cast<size_t>(r) % seq_len;
        for (size_t h = 0; h < num_heads; h++) {
            const size_t head_offset = h * head_dim;
            for (size_t i = 0; i < head_dim / 2; i++) {
                const size_t cache_idx = h * head_dim + i;
                const float cos_t = cos_cached(pos, cache_idx);
                const float sin_t = sign * sin_cached(pos, cache_idx);
                const size_t i0 = head_offset + 2 * i;
                const size_t i1 = head_offset + 2 * i + 1;
                const float x0 = m(r, i0);
                const float x1 = m(r, i1);
                m(r, i0) = x0 * cos_t - x1 * sin_t;
                m(r, i1) = x0 * sin_t + x1 * cos_t;
            }
        }
    }
}

Matrix MultiHeadAttention::forward_batched(const Matrix& input, const AttentionMask& mask,
                                           size_t batch_size, size_t seq_len) {
    const size_t total_positions = batch_size * seq_len;
    if (input.rows() != total_positions) {
        throw std::runtime_error("forward_batched: input rows mismatch");
    }
    
    const size_t head_size = hidden_size / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    
    // ========== STEP 1: BATCH QKV PROJECTIONS (3 big matmuls instead of 3*batch small ones) ==========
    // Q, K, V: [total_positions x hidden_size]
    Matrix Q = matmul(input, params_.query_weights);
    Matrix K = matmul(input, params_.key_weights);
    Matrix V = matmul(input, params_.value_weights);
    
    // Add biases (parallelized)
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_positions); i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            Q(i, j) += params_.query_bias[j];
            K(i, j) += params_.key_bias[j];
            V(i, j) += params_.value_bias[j];
        }
    }

    // Rotary position embeddings (LLaMA-style, adjacent pairing).
    // Applied post-projection; the caches hold the ROTATED Q/K, which is what
    // the exact backward needs for the score-path gradients (the weight-path
    // gradients un-rotate first).
    if (use_rope) {
        rotate_rows_rope(Q, seq_len, /*inverse=*/false);
        rotate_rows_rope(K, seq_len, /*inverse=*/false);
    }

    // Cache for backward pass
    cached_query_layer = Q;
    cached_key_layer = K;
    cached_value_layer = V;
    cached_batch_size_ = batch_size;
    cached_seq_len_ = seq_len;
    cached_batched_valid_ = true;
    
    // ========== STEP 2: BATCHED ATTENTION ON GPU ==========
    // Uses cuBLAS batched GEMM - replaces 512 CPU matmuls with GPU calls
    Matrix attn_out(total_positions, hidden_size);
    cached_attention_weights = Matrix(batch_size * num_heads * seq_len, seq_len);
    
#ifdef USE_CUDA
    // The CUDA batched kernel implements causal masking only; running a
    // sliding-window model through it would silently train full-attention
    // (the exact defect just fixed on the CPU path). Fail loudly until the
    // kernel gains the window term.
    if (use_sliding_window) {
        throw std::runtime_error(
            "sliding-window attention is not implemented in the CUDA "
            "batched path (2026-07-31); use the CPU build or extend "
            "batched_attention_forward");
    }
    // Call CUDA batched attention.
    // RESOLVED (2026-07-23): the previously-suspected CUDA-vs-CPU forward
    // divergence was not real — it was a test harness reconstructing the
    // model without its architecture flags (additive PE instead of RoPE).
    // With the correct config, the CPU forward is bit-identical to the
    // inference engine (golden-batch parity PASSES). See CHAT_EXPERIMENTS.md
    // Finding 6 (retracted).
    cuda::batched_attention_forward(
        Q.data(),
        K.data(),
        V.data(),
        attn_out.data(),
        cached_attention_weights.data(),
        static_cast<int>(batch_size),
        static_cast<int>(seq_len),
        static_cast<int>(num_heads),
        static_cast<int>(head_size),
        scale
    );
#else
    // CPU fallback: process each (batch, head) combination
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < static_cast<int>(batch_size); b++) {
        for (int h = 0; h < static_cast<int>(num_heads); h++) {
            size_t batch_offset = b * seq_len;
            size_t head_offset = h * head_size;
            
            // Attention scores: Q_h @ K_h^T
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < head_size; k++) {
                        sum += Q(batch_offset + i, head_offset + k) * K(batch_offset + j, head_offset + k);
                    }
                    size_t cache_idx = (b * num_heads + h) * seq_len + i;
                    // Causal AND sliding-window mask. The window uses the
                    // same half-window semantics as the single-sequence
                    // path (attention.cpp:1306) so train and infer agree;
                    // combined with causality the visible span is
                    // [i - window_size/2, i]. Before 2026-07-31 the window
                    // was NOT applied here at all -- the swa_check probe
                    // showed far-outside-window perturbations leaking into
                    // the training path (STUDIO_PLAN section 12).
                    const bool masked_out =
                        (j > i) ||
                        (use_sliding_window && (i - j) > window_size / 2);
                    cached_attention_weights(cache_idx, j) =
                        masked_out ? -1e30f : sum * scale;
                }
            }
            
            // Softmax per row
            for (size_t i = 0; i < seq_len; i++) {
                size_t cache_idx = (b * num_heads + h) * seq_len + i;
                float max_val = -1e30f;
                for (size_t j = 0; j < seq_len; j++) {
                    max_val = std::max(max_val, cached_attention_weights(cache_idx, j));
                }
                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    cached_attention_weights(cache_idx, j) = std::exp(cached_attention_weights(cache_idx, j) - max_val);
                    sum_exp += cached_attention_weights(cache_idx, j);
                }
                float inv_sum = 1.0f / (sum_exp + 1e-10f);
                for (size_t j = 0; j < seq_len; j++) {
                    cached_attention_weights(cache_idx, j) *= inv_sum;
                }
            }
            
            // Attention output: scores @ V_h
            for (size_t i = 0; i < seq_len; i++) {
                size_t cache_idx = (b * num_heads + h) * seq_len + i;
                for (size_t d = 0; d < head_size; d++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_len; j++) {
                        sum += cached_attention_weights(cache_idx, j) * V(batch_offset + j, head_offset + d);
                    }
                    attn_out(batch_offset + i, head_offset + d) = sum;
                }
            }
        }
    }
#endif
    
    // Cache attention output
    cached_attn_output = attn_out;
    
    // ========== STEP 3: BATCH OUTPUT PROJECTION (1 big matmul) ==========
    Matrix output = matmul(attn_out, params_.output_weights);
    
    // Add output bias
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_positions); i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            output(i, j) += params_.output_bias[j];
        }
    }

    if (std::getenv("TCPP_ATTN_TRACE") != nullptr) {
        static int cc = 0;
        auto p0 = [&](const Matrix& m) {
            double s = 0.0;
            for (size_t j = 0; j < hidden_size; ++j) { float v = m(0, j); s += double(v) * v; }
            return std::sqrt(s);
        };
        fprintf(stderr, "[ATTN] call=%d in[0..3]=%g %g %g %g V_pos0=%g attnout_pos0=%g out_pos0=%g\n",
                cc++, input(0,0), input(0,1), input(0,2), input(0,3),
                p0(V), p0(attn_out), p0(output));
    }

    return output;
}

Matrix MultiHeadAttention::compute_attention_scores(const Matrix& Q, const Matrix& K, const AttentionMask& mask) {
    DEBUG_LOG(SCOPE_LOG());
    // For attention scores, we want (seq_len x seq_len)
    Matrix scores(Q.rows(), K.rows());  // Changed from K.cols() to K.rows()
    #ifdef USE_CUDA
    cuda::matmul(Q, K.transpose(), scores);
    #else
    scores = matmul(Q, K.transpose());
    #endif

    // Scale scores with careful handling of numerical stability
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Clip extremely large values to prevent overflow
    const float max_score = 100.0f; // Prevent exp overflow

    for (size_t i = 0; i < scores.rows(); ++i) {
        // Find max for numerical stability in softmax
        float row_max = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < scores.cols(); ++j) {
            scores(i, j) *= scale;
            
            // Apply mask if needed
            if (mask.is_masked(i, j)) {
                scores(i, j) = -std::numeric_limits<float>::infinity();
            } else {
                scores(i, j) = std::min(scores(i, j), max_score);
            }
            row_max = std::max(row_max, scores(i, j));
        }

        // Compute softmax with improved numerical stability
        float sum_exp = 0.0f;
        for (size_t j = 0; j < scores.cols(); ++j) {
            scores(i, j) = std::exp(scores(i, j) - row_max);
            sum_exp += scores(i, j);
        }

        // Normalize with careful handling of small values
        const float eps = 1e-6f;
        if (sum_exp < eps)
            sum_exp = eps;

        for (size_t j = 0; j < scores.cols(); ++j) {
            scores(i, j) /= sum_exp;
        }
    }

    return scores;
}

Matrix MultiHeadAttention::backward(const Matrix& grad_output, const Matrix& input, const Matrix& target, const TransformerConfig& config) {
    const float clip_threshold = config.gradient_clip_threshold;
    try {
        const float eps = 1e-6f;
        
        // Verify cached attention output is available
        if (cached_attn_output.empty()) {
            throw std::runtime_error("MultiHeadAttention::backward called without cached_attn_output - forward() must be called first!");
        }
        
        // Deterministic gradient norm for clipping (gate-2, thread-count-invariant);
        // see repro_reduce.hpp.
        float grad_norm = std::sqrt(repro_sumsq(grad_output.data(), grad_output.size()));
        
        // Proper gradient clipping (no sqrt - direct scaling)
        float scale = (grad_norm > clip_threshold) ? (clip_threshold / (grad_norm + eps)) : 1.0f;
        
        // Scale gradients
        Matrix scaled_grad = grad_output;
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(scaled_grad.rows()); i++) {
            for (int j = 0; j < static_cast<int>(scaled_grad.cols()); j++) {
                scaled_grad(i, j) *= scale;
            }
        }
        
        size_t batch_size = input.rows();
        const float grad_scale = 1.0f / static_cast<float>(batch_size);  // For averaging gradients
        Matrix input_t = input.transpose();

        // Set by the exact-backward branch below; when valid, it replaces the
        // approximate V-path input gradient at the end of this function.
        Matrix exact_d_input;
        bool have_exact_d_input = false;
        
        // ========== OUTPUT PROJECTION GRADIENTS ==========
        // Forward: final_output = attn_out @ W_o + b_o
        // dW_o = attn_out^T @ d_final_output  (MUST use cached_attn_output, NOT input!)
        Matrix attn_out_t = cached_attn_output.transpose();
        Matrix output_weight_grad = matmul(attn_out_t, scaled_grad);
        output_weight_grad *= grad_scale;  // Average over batch, not sum!
        
        // Bias gradient: average over batch
        Vector output_bias_grad = Vector(params_.output_bias.size(), 0.0f);
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_bias_grad.size(); ++j) {
                output_bias_grad[j] += scaled_grad(i, j) * grad_scale;
            }
        }
        
        // ========== BACKPROP THROUGH OUTPUT PROJECTION ==========
        // d_attn_out = scaled_grad @ W_o^T
        Matrix output_weights_t = params_.output_weights.transpose();
        Matrix d_attn_out = matmul(scaled_grad, output_weights_t);
        
        // ========== Q/K/V PROJECTION GRADIENTS ==========
        // Now properly implemented using cached Q, K, V, and attention weights
        
        // Check if we have cached values (should always be true if forward was called)
        if (cached_query_layer.empty() || cached_key_layer.empty() || 
            cached_value_layer.empty() || cached_attention_weights.empty()) {
            // Fallback to zero gradients if caches not available
            DEBUG_COUT << "WARNING: Attention caches not available, using zero Q/K/V gradients" << std::endl;
            Matrix query_weight_grad(params_.query_weights.rows(), params_.query_weights.cols(), 0.0f);
            Matrix key_weight_grad(params_.key_weights.rows(), params_.key_weights.cols(), 0.0f);
            Matrix value_weight_grad(params_.value_weights.rows(), params_.value_weights.cols(), 0.0f);
            Vector query_bias_grad = Vector(params_.query_bias.size(), 0.0f);
            Vector key_bias_grad = Vector(params_.key_bias.size(), 0.0f);
            Vector value_bias_grad = Vector(params_.value_bias.size(), 0.0f);
            
            param_gradients().query_grad = query_weight_grad;
            param_gradients().key_grad = key_weight_grad;
            param_gradients().value_grad = value_weight_grad;
            param_gradients().query_bias_grad = query_bias_grad;
            param_gradients().key_bias_grad = key_bias_grad;
            param_gradients().value_bias_grad = value_bias_grad;
        } else if (cached_batched_valid_) {
            // ========== EXACT ATTENTION BACKWARD (batched layout) ==========
            // Forward (per batch b, head h):
            //   S = softmax(mask(Q_h K_h^T * scale)); out_h = S @ V_h
            // Backward:
            //   dV_h = S^T @ dOut_h
            //   dP   = dOut_h @ V_h^T
            //   dS   = S .* (dP - rowsum(dP .* S))        [softmax backward]
            //   dQ_h = dS @ K_h * scale ; dK_h = dS^T @ Q_h * scale
            // The causal mask is respected implicitly: masked S entries are 0.
            // Q/K in the caches are POST-RoPE; gradients are un-rotated before
            // the weight-gradient matmuls.
            const size_t S_len = cached_seq_len_;
            const size_t B = cached_batch_size_;
            const size_t H = num_heads;
            const size_t hd = hidden_size / num_heads;
            const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

            Matrix dQ(input.rows(), hidden_size, 0.0f);
            Matrix dK(input.rows(), hidden_size, 0.0f);
            Matrix dV(input.rows(), hidden_size, 0.0f);

            // The exact backward is four batched GEMMs plus a row-wise softmax
            // backward. The scalar CPU loops below are mathematically identical
            // but ~40x slower per step at the GPU training config (they were
            // the July-2026 training-speed regression); on CUDA builds the
            // cuBLAS path replaces them. Set TCPP_BACKWARD_PARITY=1 to run
            // both and report the max deviation (gradient parity check).
#ifdef USE_CUDA
            cuda::batched_attention_backward(
                d_attn_out.data(), cached_attention_weights.data(),
                cached_query_layer.data(), cached_key_layer.data(),
                cached_value_layer.data(),
                dQ.data(), dK.data(), dV.data(),
                static_cast<int>(B), static_cast<int>(S_len),
                static_cast<int>(H), static_cast<int>(hd), scale);
            const bool run_cpu_exact_backward =
                (std::getenv("TCPP_BACKWARD_PARITY") != nullptr);
            Matrix dQ_ref, dK_ref, dV_ref;
            if (run_cpu_exact_backward) {
                dQ_ref = Matrix(input.rows(), hidden_size, 0.0f);
                dK_ref = Matrix(input.rows(), hidden_size, 0.0f);
                dV_ref = Matrix(input.rows(), hidden_size, 0.0f);
            }
            Matrix& dQ_cpu = run_cpu_exact_backward ? dQ_ref : dQ;
            Matrix& dK_cpu = run_cpu_exact_backward ? dK_ref : dK;
            Matrix& dV_cpu = run_cpu_exact_backward ? dV_ref : dV;
            if (run_cpu_exact_backward) {
#else
            const bool run_cpu_exact_backward = true;
            Matrix& dQ_cpu = dQ;
            Matrix& dK_cpu = dK;
            Matrix& dV_cpu = dV;
            {
#endif
            #pragma omp parallel for collapse(2)
            for (int b = 0; b < static_cast<int>(B); b++) {
                for (int h = 0; h < static_cast<int>(H); h++) {
                    const size_t bo = static_cast<size_t>(b) * S_len;
                    const size_t ho = static_cast<size_t>(h) * hd;
                    const size_t p_base = (static_cast<size_t>(b) * H + h) * S_len;

                    std::vector<float> dP(S_len * S_len, 0.0f);
                    std::vector<float> dS(S_len * S_len, 0.0f);

                    // dP = dOut_h @ V_h^T; dV_h = S^T @ dOut_h
                    for (size_t i = 0; i < S_len; i++) {
                        for (size_t j = 0; j < S_len; j++) {
                            float sum = 0.0f;
                            for (size_t d = 0; d < hd; d++) {
                                sum += d_attn_out(bo + i, ho + d) *
                                       cached_value_layer(bo + j, ho + d);
                            }
                            dP[i * S_len + j] = sum;
                        }
                    }

                    // Softmax backward per row
                    for (size_t i = 0; i < S_len; i++) {
                        float dot = 0.0f;
                        for (size_t j = 0; j < S_len; j++) {
                            dot += dP[i * S_len + j] *
                                   cached_attention_weights(p_base + i, j);
                        }
                        for (size_t j = 0; j < S_len; j++) {
                            dS[i * S_len + j] =
                                cached_attention_weights(p_base + i, j) *
                                (dP[i * S_len + j] - dot);
                        }
                    }

                    // dQ_h = dS @ K_h * scale; dK_h = dS^T @ Q_h * scale;
                    // dV_h = S^T @ dOut_h
                    for (size_t i = 0; i < S_len; i++) {
                        for (size_t j = 0; j < S_len; j++) {
                            const float ds = dS[i * S_len + j] * scale;
                            const float p = cached_attention_weights(p_base + i, j);
                            for (size_t d = 0; d < hd; d++) {
                                dQ_cpu(bo + i, ho + d) += ds * cached_key_layer(bo + j, ho + d);
                                dK_cpu(bo + j, ho + d) += ds * cached_query_layer(bo + i, ho + d);
                                dV_cpu(bo + j, ho + d) += p * d_attn_out(bo + i, ho + d);
                            }
                        }
                    }
                }
            }
            }
#ifdef USE_CUDA
            if (run_cpu_exact_backward) {
                float max_diff = 0.0f;
                for (size_t i = 0; i < dQ.rows(); ++i) {
                    for (size_t j = 0; j < dQ.cols(); ++j) {
                        max_diff = std::max(max_diff, std::fabs(dQ(i, j) - dQ_ref(i, j)));
                        max_diff = std::max(max_diff, std::fabs(dK(i, j) - dK_ref(i, j)));
                        max_diff = std::max(max_diff, std::fabs(dV(i, j) - dV_ref(i, j)));
                    }
                }
                std::cout << "[BACKWARD_PARITY] max |gpu - cpu| gradient deviation: "
                          << max_diff << std::endl;
            }
#endif

            // Un-rotate: gradients w.r.t. pre-RoPE projections (rotation is
            // orthogonal, so the backward is rotation by -theta)
            if (use_rope) {
                rotate_rows_rope(dQ, S_len, /*inverse=*/true);
                rotate_rows_rope(dK, S_len, /*inverse=*/true);
            }

            // Weight gradients: input^T @ dProj, averaged over rows
            Matrix input_t_local = input.transpose();
            Matrix query_weight_grad = matmul(input_t_local, dQ);
            query_weight_grad *= grad_scale;
            Matrix key_weight_grad = matmul(input_t_local, dK);
            key_weight_grad *= grad_scale;
            Matrix value_weight_grad = matmul(input_t_local, dV);
            value_weight_grad *= grad_scale;

            // Bias gradients: column sums, averaged over rows
            Vector query_bias_grad = Vector(params_.query_bias.size(), 0.0f);
            Vector key_bias_grad = Vector(params_.key_bias.size(), 0.0f);
            Vector value_bias_grad = Vector(params_.value_bias.size(), 0.0f);
            for (size_t i = 0; i < input.rows(); ++i) {
                for (size_t j = 0; j < hidden_size; ++j) {
                    query_bias_grad[j] += dQ(i, j) * grad_scale;
                    key_bias_grad[j] += dK(i, j) * grad_scale;
                    value_bias_grad[j] += dV(i, j) * grad_scale;
                }
            }

            param_gradients().query_grad = query_weight_grad;
            param_gradients().key_grad = key_weight_grad;
            param_gradients().value_grad = value_weight_grad;
            param_gradients().query_bias_grad = query_bias_grad;
            param_gradients().key_bias_grad = key_bias_grad;
            param_gradients().value_bias_grad = value_bias_grad;

            // Exact input gradient: sum over all three projection paths
            exact_d_input = matmul(dQ, params_.query_weights.transpose());
            exact_d_input += matmul(dK, params_.key_weights.transpose());
            exact_d_input += matmul(dV, params_.value_weights.transpose());
            have_exact_d_input = true;
        } else {
            // Simplified backward through attention (avoids multi-head reshape complexity)
            // Key insight: Even approximate gradients are better than zero
            // 
            // For attention: output = softmax(Q @ K^T / sqrt(d)) @ V
            // The dominant gradient path is through V (most direct)
            // We use a simplified approximation for Q, K gradients
            
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            
            // Step 1: Gradient w.r.t. V (simplified: treat attention as weighted average)
            // Approximate: d_V ≈ d_attn_out (attention weights sum to 1 per row)
            Matrix d_V = d_attn_out;
            
            // Step 2: Gradient w.r.t. Q and K (simplified approximation)
            // The Q/K gradients are smaller in magnitude but help the model learn
            // Use the gradient flowing through V scaled down
            Matrix d_Q = d_attn_out;
            d_Q *= (scale * 0.5f);  // Scale down Q gradient
            
            Matrix d_K = d_attn_out;
            d_K *= (scale * 0.5f);  // Scale down K gradient
            
            // Step 3: Weight gradients (input^T @ d_projection, normalized by batch)
            Matrix input_t_local = input.transpose();
            Matrix query_weight_grad = matmul(input_t_local, d_Q);
            query_weight_grad *= grad_scale;  // Average over batch
            Matrix key_weight_grad = matmul(input_t_local, d_K);
            key_weight_grad *= grad_scale;  // Average over batch
            Matrix value_weight_grad = matmul(input_t_local, d_V);
            value_weight_grad *= grad_scale;  // Average over batch
            
            // Step 4: Bias gradients (average over batch)
            Vector query_bias_grad = Vector(params_.query_bias.size(), 0.0f);
            Vector key_bias_grad = Vector(params_.key_bias.size(), 0.0f);
            Vector value_bias_grad = Vector(params_.value_bias.size(), 0.0f);
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < query_bias_grad.size(); ++j) {
                    query_bias_grad[j] += d_Q(i, j) * grad_scale;
                    key_bias_grad[j] += d_K(i, j) * grad_scale;
                    value_bias_grad[j] += d_V(i, j) * grad_scale;
                }
            }
            
            // Apply gradient clipping to Q/K/V gradients
            auto clip_matrix = [clip_threshold, eps](Matrix& grad) {
                float norm_sq = 0.0f;
                for (size_t i = 0; i < grad.rows(); ++i) {
                    for (size_t j = 0; j < grad.cols(); ++j) {
                        norm_sq += grad(i, j) * grad(i, j);
                    }
                }
                float norm = std::sqrt(norm_sq);
                if (norm > clip_threshold) {
                    float clip_scale = clip_threshold / (norm + eps);
                    for (size_t i = 0; i < grad.rows(); ++i) {
                        for (size_t j = 0; j < grad.cols(); ++j) {
                            grad(i, j) *= clip_scale;
                        }
                    }
                }
            };
            
            clip_matrix(query_weight_grad);
            clip_matrix(key_weight_grad);
            clip_matrix(value_weight_grad);
            
            param_gradients().query_grad = query_weight_grad;
            param_gradients().key_grad = key_weight_grad;
            param_gradients().value_grad = value_weight_grad;
            param_gradients().query_bias_grad = query_bias_grad;
            param_gradients().key_bias_grad = key_bias_grad;
            param_gradients().value_bias_grad = value_bias_grad;
        }
        
        // Store output gradients (always computed)
        param_gradients().output_grad = output_weight_grad;
        param_gradients().output_bias_grad = output_bias_grad;
        
        // ========== INPUT GRADIENT FOR BACKWARD FLOW ==========
        if (have_exact_d_input) {
            DEBUG_COUT << "=== MultiHeadAttention::backward END (exact) ===\n" << std::endl;
            return exact_d_input;
        }

        // Legacy approximation (non-batched path only): value projection path
        Matrix value_weights_t = params_.value_weights.transpose();
        Matrix d_input = matmul(d_attn_out, value_weights_t);

        DEBUG_COUT << "=== MultiHeadAttention::backward END ===\n" << std::endl;
        return d_input;
        
    } catch (const std::exception& e) {
        DEBUG_CERR << "\nError in MultiHeadAttention::backward: " << e.what() << std::endl;
        throw;
    }
}

Matrix MultiHeadAttention::compute_query_gradients(const Matrix& grad, const Matrix& input) {
    DEBUG_COUT << "\n=== compute_query_gradients dimensions ===" << std::endl;
    DEBUG_COUT << "grad: " << grad.rows() << "x" << grad.cols() << std::endl;
    DEBUG_COUT << "input: " << input.rows() << "x" << input.cols() << std::endl;
    DEBUG_COUT << "query_weights: " << params_.query_weights.rows() << "x" << params_.query_weights.cols() << std::endl;

    Matrix d_query(grad.rows(), grad.cols());
    DEBUG_COUT << "d_query (pre-matmul): " << d_query.rows() << "x" << d_query.cols() << std::endl;

    #ifdef USE_CUDA
    cuda::matmul(grad, params_.query_weights.transpose(), d_query);
    #else
    d_query = matmul(grad, params_.query_weights.transpose());
    #endif

    DEBUG_COUT << "d_query (final): " << d_query.rows() << "x" << d_query.cols() << std::endl;
    return d_query;
}

Matrix MultiHeadAttention::compute_key_gradients(const Matrix& grad, const Matrix& input) {
    DEBUG_COUT << "\n=== compute_key_gradients dimensions ===" << std::endl;
    DEBUG_COUT << "grad: " << grad.rows() << "x" << grad.cols() << std::endl;
    DEBUG_COUT << "input: " << input.rows() << "x" << input.cols() << std::endl;
    DEBUG_COUT << "key_weights: " << params_.key_weights.rows() << "x" << params_.key_weights.cols() << std::endl;

    Matrix d_key(grad.rows(), grad.cols());
    DEBUG_COUT << "d_key (pre-matmul): " << d_key.rows() << "x" << d_key.cols() << std::endl;

    #ifdef USE_CUDA
    cuda::matmul(grad, params_.key_weights.transpose(), d_key);
    #else
    d_key = matmul(grad, params_.key_weights.transpose());
    #endif

    DEBUG_COUT << "d_key (final): " << d_key.rows() << "x" << d_key.cols() << std::endl;
    return d_key;
}

Matrix MultiHeadAttention::compute_value_gradients(const Matrix& grad, const Matrix& input) {
    DEBUG_COUT << "\n=== compute_value_gradients dimensions ===" << std::endl;
    DEBUG_COUT << "grad: " << grad.rows() << "x" << grad.cols() << std::endl;
    DEBUG_COUT << "input: " << input.rows() << "x" << input.cols() << std::endl;
    DEBUG_COUT << "value_weights: " << params_.value_weights.rows() << "x" << params_.value_weights.cols() << std::endl;

    Matrix d_value(grad.rows(), grad.cols());
    DEBUG_COUT << "d_value (pre-matmul): " << d_value.rows() << "x" << d_value.cols() << std::endl;

    #ifdef USE_CUDA
    cuda::matmul(grad, params_.value_weights.transpose(), d_value);
    #else
    d_value = matmul(grad, params_.value_weights.transpose());
    #endif

    DEBUG_COUT << "d_value (final): " << d_value.rows() << "x" << d_value.cols() << std::endl;
    return d_value;
}

Matrix MultiHeadAttention::combine_gradients(const Matrix& d_query, const Matrix& d_key, const Matrix& d_value) {
    DEBUG_COUT << "\n=== combine_gradients dimensions ===" << std::endl;
    DEBUG_COUT << "d_query: " << d_query.rows() << "x" << d_query.cols() << std::endl;
    DEBUG_COUT << "d_key: " << d_key.rows() << "x" << d_key.cols() << std::endl;
    DEBUG_COUT << "d_value: " << d_value.rows() << "x" << d_value.cols() << std::endl;
    
    // Validate dimensions before combining
    if (d_query.rows() != d_key.rows() || d_query.cols() != d_key.cols() ||
        d_key.rows() != d_value.rows() || d_key.cols() != d_value.cols()) {
        DEBUG_CERR << "Dimension mismatch in combine_gradients:" << std::endl;
        DEBUG_CERR << "d_query: " << d_query.rows() << "x" << d_query.cols() << std::endl;
        DEBUG_CERR << "d_key: " << d_key.rows() << "x" << d_key.cols() << std::endl;
        DEBUG_CERR << "d_value: " << d_value.rows() << "x" << d_value.cols() << std::endl;
        throw std::runtime_error("Gradient dimensions must match for combination");
    }
    
    Matrix combined = d_query;
    DEBUG_COUT << "combined (pre-addition): " << combined.rows() << "x" << combined.cols() << std::endl;
    
    combined += d_key;
    DEBUG_COUT << "combined (after d_key): " << combined.rows() << "x" << combined.cols() << std::endl;
    
    combined += d_value;
    DEBUG_COUT << "combined (final): " << combined.rows() << "x" << combined.cols() << std::endl;
    
    return combined;
}

void MultiHeadAttention::initialize_weights() {
    // Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (hidden_size + head_dim));
    
    // Initialize projection matrices using parameter accessors
    params_.query_weights.initialize_random(scale);
    params_.key_weights.initialize_random(scale);
    params_.value_weights.initialize_random(scale);
    params_.output_weights.initialize_random(scale);
    
    // Initialize bias vectors. LLaMA mode: exactly zero and frozen (the
    // llama architecture has no attention biases; zero + frozen makes the
    // bias-add a no-op so exported weights reproduce training math).
    const float bias_init = transformer_runtime::llama_no_bias ? 0.0f : 0.01f;
    params_.query_bias.initialize_constant(bias_init);
    params_.key_bias.initialize_constant(bias_init);
    params_.value_bias.initialize_constant(bias_init);
    params_.output_bias.initialize_constant(bias_init);
}

float compute_grad_norm(const Matrix& grad) {
    // Deterministic (thread-count-invariant) grad norm (gate-2); see repro_reduce.hpp.
    return std::sqrt(repro_sumsq(grad.data(), grad.rows() * grad.cols()));
}

size_t count_params(const Matrix& param) {
    return param.rows() * param.cols();
}

float MultiHeadAttention::get_cos_cached(size_t pos, size_t dim_idx) const {
    if (pos >= cos_cached.rows() || dim_idx >= cos_cached.cols()) {
        throw std::runtime_error("RoPE cache access out of bounds: pos=" + std::to_string(pos) +
                                 ", dim=" + std::to_string(dim_idx));
    }
    return cos_cached(pos, dim_idx);
}

float MultiHeadAttention::get_sin_cached(size_t pos, size_t dim_idx) const {
    if (pos >= sin_cached.rows() || dim_idx >= sin_cached.cols()) {
        throw std::runtime_error("RoPE cache access out of bounds: pos=" + std::to_string(pos) +
                                 ", dim=" + std::to_string(dim_idx));
    }
    return sin_cached(pos, dim_idx);
}

void MultiHeadAttention::apply_stable_softmax(Matrix& x) const {
#ifdef USE_CUDA
    // Use CUDA softmax for GPU acceleration
    cuda::softmax(x);
#else
    // CPU implementation: Process each row separately for proper attention distribution
    for (size_t row = 0; row < x.rows(); row++) {
        // Find max value in this row for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t col = 0; col < x.cols(); col++) {
            max_val = std::max(max_val, x(row, col));
        }

        // Subtract max value and compute exp for this row
        float row_sum = 0.0f;
        for (size_t col = 0; col < x.cols(); col++) {
            // Subtract max value before exp for numerical stability
            x(row, col) = std::exp(x(row, col) - max_val);
            row_sum += x(row, col);
        }

        // Check for numerical instability in this row
        if (row_sum < 1e-10) {
            DEBUG_CERR << "WARNING: Row " << row << " has near-zero softmax sum\n";
            // Fall back to uniform attention for this row only
            float uniform_val = 1.0f / x.cols();
            for (size_t col = 0; col < x.cols(); col++) {
                x(row, col) = uniform_val;
            }
            continue;
        }

        // Normalize this row
        for (size_t col = 0; col < x.cols(); col++) {
            x(row, col) /= row_sum;
        }
    }
#endif

    // Validate results (only in debug mode to avoid overhead)
#ifdef DEBUG_ATTENTION_LOGGING
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < x.rows(); i++) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < x.cols(); j++) {
            min_val = std::min(min_val, x(i, j));
            max_val = std::max(max_val, x(i, j));
            row_sum += x(i, j);
        }
        if (std::abs(row_sum - 1.0f) > 1e-6) {
            DEBUG_CERR << "WARNING: Row " << i << " softmax sum = " << row_sum << "\n";
        }
    }
    
    DEBUG_COUT << "Softmax output statistics:\n"
              << "Min: " << min_val << "\n"
              << "Max: " << max_val << "\n";
#endif
}

Tensor MultiHeadAttention::reshape_for_attention(const Matrix& x, size_t batch_size,
                                                 size_t num_heads, size_t seq_len,
                                                 size_t head_size) const {
    // Create a 4D tensor with shape [batch_size, num_heads, seq_len, head_size]
    Tensor reshaped(batch_size, num_heads, seq_len, head_size);

    // Copy and reshape the data
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t d = 0; d < head_size; ++d) {
                    // Correct indexing for the input matrix
                    size_t flat_idx = s * x.cols() + h * head_size + d;
                    reshaped.at(b, h, s, d) = x.data()[flat_idx];
                }
            }
        }
    }
    return reshaped;
}

Matrix MultiHeadAttention::reshape_from_attention(const Tensor& x, size_t batch_size,
                                                  size_t hidden_size) const {
    DEBUG_COUT << "=== reshape_from_attention START ===" << std::endl;

    // Get dimensions from tensor
    const auto& dims = x.dims();
    size_t seq_len = dims[2]; // Third dimension is sequence length

    // Output should have shape (batch_size * seq_len, hidden_size)
    Matrix reshaped(batch_size * seq_len, hidden_size);

    // Reshape from [batch_size, num_heads, seq_len, head_dim] to [batch_size * seq_len,
    // hidden_size]
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t d = 0; d < head_dim; ++d) {
                    // Calculate output position
                    size_t out_row = b * seq_len + s;
                    size_t out_col = h * head_dim + d;

                    // Get value from tensor
                    reshaped(out_row, out_col) = x.at(b, h, s, d);
                }
            }
        }
    }

    DEBUG_COUT << "Reshaped output dimensions: " << reshaped.rows() << "x" << reshaped.cols()
              << std::endl;
    DEBUG_COUT << "=== reshape_from_attention END ===" << std::endl;

    return reshaped;
}

Matrix MultiHeadAttention::compute_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                          const AttentionMask& mask) const {
    // Validate input dimensions
    if (Q.cols() != K.cols() || K.cols() != V.cols()) {
        throw std::runtime_error("Q, K, V dimension mismatch");
    }

    size_t seq_len = Q.rows();
    size_t head_size = Q.cols() / num_heads;

    // Reshape maintaining [seq_len, hidden_size] as the basic shape
    Tensor Q_reshaped = reshape_for_attention(Q, 1, num_heads, seq_len, head_size);
    Tensor K_reshaped = reshape_for_attention(K, 1, num_heads, seq_len, head_size);
    Tensor V_reshaped = reshape_for_attention(V, 1, num_heads, seq_len, head_size);

    // Convert to matrices for computation while preserving effective dimensions
    Matrix Q_mat = Q_reshaped.to_matrix(); // [num_heads * seq_len, head_size]
    Matrix K_mat = K_reshaped.to_matrix(); // [num_heads * seq_len, head_size]
    Matrix V_mat = V_reshaped.to_matrix(); // [num_heads * seq_len, head_size]

    // Compute attention scores
    #ifdef USE_CUDA
    Matrix scores(Q_mat.rows(), K_mat.cols());
    cuda::matmul(Q_mat, K_mat.transpose(), scores);
    #else
    Matrix scores = matmul(Q_mat, K_mat.transpose());
    #endif

    // Scale scores
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    scores *= scale;

    // Apply sliding window attention if enabled
    if (use_sliding_window) {
        const size_t half_window = window_size / 2;
        DEBUG_COUT << "Applying sliding window attention with window size " << window_size << std::endl;
        
        // MSVC: collapse ignored, loop vars must be signed int
        #pragma omp parallel for collapse(2)
        for (int head = 0; head < static_cast<int>(num_heads); head++) {
            for (int i = 0; i < static_cast<int>(seq_len); i++) {
                size_t row_idx = head * seq_len + i;
                
                // Calculate window boundaries
                size_t window_start = (i >= half_window) ? i - half_window : 0;
                size_t window_end = std::min(i + half_window + 1, seq_len);
                
                // Mask out everything outside the window
                for (size_t j = 0; j < seq_len; j++) {
                    size_t col_idx = j;
                    if (j < window_start || j >= window_end) {
                        scores(row_idx, col_idx) = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
    }

    // Apply attention mask if provided
    for (size_t i = 0; i < scores.rows(); i++) {
        for (size_t j = 0; j < scores.cols(); j++) {
            if (mask.is_masked(i % seq_len, j % seq_len)) {
                scores(i, j) = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // Apply softmax with improved numerical stability
    apply_stable_softmax(scores);

    // Compute attention output
    Matrix attention;
    #ifdef USE_CUDA
    attention = Matrix(scores.rows(), V.cols());
    cuda::matmul(scores, V, attention);
    #else
    attention = matmul(scores, V);
    #endif

    // Reshape back to original dimensions
    std::vector<unsigned long> dims = {
        static_cast<unsigned long>(1), static_cast<unsigned long>(num_heads),
        static_cast<unsigned long>(seq_len), static_cast<unsigned long>(head_size)};
    return reshape_from_attention(Tensor(attention, dims), seq_len, hidden_size);
}

Matrix MultiHeadAttention::create_causal_mask(size_t seq_length) const {
    Matrix mask(seq_length, seq_length, 0.0f);
    // Create lower triangular matrix (1s below diagonal, 0s above)
    for (size_t i = 0; i < seq_length; i++) {
        for (size_t j = 0; j <= i; j++) {
            mask(i, j) = 1.0f;
        }
    }
    return mask;
}

AttentionMask AttentionMask::create_padding_mask(const std::vector<int>& lengths, size_t max_len) {
    AttentionMask mask;
    size_t batch_size = lengths.size();
    mask.mask = Matrix(max_len, max_len, 0.0f);

    // Create padding mask
    for (size_t i = 0; i < max_len; ++i) {
        for (size_t j = 0; j < max_len; ++j) {
            // Allow attention up to the sequence length
            mask.mask(i, j) = (i < lengths[0] && j < lengths[0]) ? 1.0f : 0.0f;
        }
    }
    return mask;
}

Tensor MultiHeadAttention::compute_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                             const AttentionMask& mask, size_t batch_size,
                                             size_t num_heads, size_t seq_len, size_t head_dim) {
    DEBUG_COUT << "=== compute_attention START ===" << std::endl;

    // Validate input dimensions
    DEBUG_COUT << "Validating dimensions..." << std::endl;
    DEBUG_COUT << "Expected dimensions:" << std::endl;
    DEBUG_COUT << "- batch_size: " << batch_size << std::endl;
    DEBUG_COUT << "- num_heads: " << num_heads << std::endl;
    DEBUG_COUT << "- seq_len: " << seq_len << std::endl;
    DEBUG_COUT << "- head_dim: " << head_dim << std::endl;

    size_t expected_rows = batch_size * num_heads * seq_len;
    size_t expected_cols = head_dim;

    // Dimension validation...
    if (Q.rows() != expected_rows || Q.cols() != expected_cols) {
        throw std::runtime_error("Q dimensions mismatch");
    }
    if (K.rows() != expected_rows || K.cols() != expected_cols) {
        throw std::runtime_error("K dimensions mismatch");
    }
    if (V.rows() != expected_rows || V.cols() != expected_cols) {
        throw std::runtime_error("V dimensions mismatch");
    }

    // Initialize output matrix
    Matrix output(Q.rows(), V.cols(), 0.0f);

    // Block size for processing (adjust based on available memory)
    const size_t BLOCK_SIZE = 1024; // Process 1024 rows at a time
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Process attention in blocks
    for (size_t start_idx = 0; start_idx < Q.rows(); start_idx += BLOCK_SIZE) {
        size_t end_idx = std::min(start_idx + BLOCK_SIZE, Q.rows());
        size_t current_block_size = end_idx - start_idx;

        DEBUG_COUT << "Processing block " << start_idx / BLOCK_SIZE + 1 << " of "
                  << (Q.rows() + BLOCK_SIZE - 1) / BLOCK_SIZE << std::endl;

        // Extract block of Q
        Matrix Q_block(current_block_size, Q.cols());
        for (size_t i = 0; i < current_block_size; ++i) {
            for (size_t j = 0; j < Q.cols(); ++j) {
                Q_block(i, j) = Q(start_idx + i, j);
            }
        }

        // Compute scores for this block
        Matrix scores(Q_block.rows(), K.cols());
        #ifdef USE_CUDA
        cuda::matmul(Q_block, K.transpose(), scores);
        #else
        scores = matmul(Q_block, K.transpose());
        #endif
        scores *= scale_factor;

        // Apply mask for this block if provided
        if (!mask.mask.empty()) {
            for (size_t i = 0; i < current_block_size; ++i) {
                for (size_t j = 0; j < K.rows(); ++j) {
                    // Calculate original indices for masking
                    size_t orig_i = start_idx + i;
                    size_t batch_idx_i = orig_i / (num_heads * seq_len);
                    size_t head_idx_i = (orig_i % (num_heads * seq_len)) / seq_len;
                    size_t seq_idx_i = orig_i % seq_len;

                    size_t batch_idx_j = j / (num_heads * seq_len);
                    size_t head_idx_j = (j % (num_heads * seq_len)) / seq_len;
                    size_t seq_idx_j = j % seq_len;

                    // Apply mask only within same batch and head
                    if (batch_idx_i == batch_idx_j && head_idx_i == head_idx_j) {
                        if (mask.mask(seq_idx_i, seq_idx_j) == 0.0f) {
                            scores(i, j) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
        }

        // Apply softmax row-wise
        for (size_t i = 0; i < current_block_size; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < scores.cols(); ++j) {
                max_val = std::max(max_val, scores(i, j));
            }

            float sum = 0.0f;
            for (size_t j = 0; j < scores.cols(); ++j) {
                scores(i, j) = std::exp(scores(i, j) - max_val);
                sum += scores(i, j);
            }

            for (size_t j = 0; j < scores.cols(); ++j) {
                scores(i, j) /= sum;
            }
        }

        // Compute output for this block
        Matrix block_output;
        #ifdef USE_CUDA
        block_output = Matrix(scores.rows(), V.cols());
        cuda::matmul(scores, V, block_output);
        #else
        block_output = matmul(scores, V);
        #endif

        // Add block output to final output
        for (size_t i = 0; i < current_block_size; ++i) {
            for (size_t j = 0; j < V.cols(); ++j) {
                output(start_idx + i, j) = block_output(i, j);
            }
        }
    }

    // Create tensor with proper dimensions
    std::vector<unsigned long> dims = {
        static_cast<unsigned long>(batch_size), static_cast<unsigned long>(num_heads),
        static_cast<unsigned long>(seq_len), static_cast<unsigned long>(head_dim)};

    DEBUG_COUT << "=== compute_attention END ===" << std::endl;
    return Tensor(output, dims);
}

void MultiHeadAttention::save(std::ostream& os) const {
    DEBUG_LOG(SCOPE_LOG());
    DEBUG_COUT << "\n=== MultiHeadAttention::save START ===" << std::endl;

    // Save dimensions and configuration
    DEBUG_COUT << "Saving configuration..." << std::endl;
    DEBUG_COUT << "- Number of heads: " << num_heads << std::endl;
    DEBUG_COUT << "- Head dimension: " << head_dim << std::endl;
    os.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    os.write(reinterpret_cast<const char*>(&head_dim), sizeof(head_dim));
    os.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    os.write(reinterpret_cast<const char*>(&dropout_prob), sizeof(dropout_prob));
    os.write(reinterpret_cast<const char*>(&use_rope), sizeof(use_rope));
    os.write(reinterpret_cast<const char*>(&use_flash), sizeof(use_flash));
    os.write(reinterpret_cast<const char*>(&use_sliding_window), sizeof(use_sliding_window));
    os.write(reinterpret_cast<const char*>(&window_size), sizeof(window_size));
    os.write(reinterpret_cast<const char*>(&use_gqa), sizeof(use_gqa));
    os.write(reinterpret_cast<const char*>(&num_kv_heads), sizeof(num_kv_heads));
    os.write(reinterpret_cast<const char*>(&max_seq_length), sizeof(max_seq_length));
    os.write(reinterpret_cast<const char*>(&use_fp16_), sizeof(use_fp16_));

    // Save weight matrices
    DEBUG_COUT << "Saving weight matrices..." << std::endl;
    params_.query_weights.save(os);
    params_.key_weights.save(os);
    params_.value_weights.save(os);
    params_.output_weights.save(os);

    // Save bias vectors
    DEBUG_COUT << "Saving bias vectors..." << std::endl;
    os.write(reinterpret_cast<const char*>(params_.query_bias.data()), params_.query_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.key_bias.data()), params_.key_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.value_bias.data()), params_.value_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.output_bias.data()), params_.output_bias.size() * sizeof(float));

    DEBUG_COUT << "=== MultiHeadAttention::save END ===\n" << std::endl;
}

std::unique_ptr<MultiHeadAttention> MultiHeadAttention::load(std::istream& is, const TransformerConfig& config) {
    DEBUG_LOG(SCOPE_LOG());
    DEBUG_COUT << "\n=== MultiHeadAttention::load START ===" << std::endl;

    // Read configuration
    size_t num_heads, head_dim, hidden_size;
    float dropout_prob;
    bool use_rope, use_flash, use_sliding_window, use_gqa, use_fp16;
    size_t window_size, num_kv_heads, max_seq_length;

    is.read(reinterpret_cast<char*>(&num_heads), sizeof(num_heads));
    is.read(reinterpret_cast<char*>(&head_dim), sizeof(head_dim));
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    is.read(reinterpret_cast<char*>(&dropout_prob), sizeof(dropout_prob));
    is.read(reinterpret_cast<char*>(&use_rope), sizeof(use_rope));
    is.read(reinterpret_cast<char*>(&use_flash), sizeof(use_flash));
    is.read(reinterpret_cast<char*>(&use_sliding_window), sizeof(use_sliding_window));
    is.read(reinterpret_cast<char*>(&window_size), sizeof(window_size));
    is.read(reinterpret_cast<char*>(&use_gqa), sizeof(use_gqa));
    is.read(reinterpret_cast<char*>(&num_kv_heads), sizeof(num_kv_heads));
    is.read(reinterpret_cast<char*>(&max_seq_length), sizeof(max_seq_length));
    is.read(reinterpret_cast<char*>(&use_fp16), sizeof(use_fp16));

    DEBUG_COUT << "Loaded configuration:" << std::endl;
    DEBUG_COUT << "- Number of heads: " << num_heads << std::endl;
    DEBUG_COUT << "- Head dimension: " << head_dim << std::endl;
    DEBUG_COUT << "- Hidden size: " << hidden_size << std::endl;

    // Create attention instance
    auto attention = std::make_unique<MultiHeadAttention>(
        hidden_size, num_heads, head_dim,
        dropout_prob, use_flash, use_rope,
        use_sliding_window, window_size,
        use_gqa, num_kv_heads,
        max_seq_length, use_fp16
    );

    // Load weight matrices
    DEBUG_COUT << "Loading weight matrices..." << std::endl;
    attention->params_.query_weights = Matrix::load(is);
    attention->params_.key_weights = Matrix::load(is);
    attention->params_.value_weights = Matrix::load(is);
    attention->params_.output_weights = Matrix::load(is);

    // Load bias vectors
    DEBUG_COUT << "Loading bias vectors..." << std::endl;
    is.read(reinterpret_cast<char*>(attention->params_.query_bias.data()), attention->params_.query_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(attention->params_.key_bias.data()), attention->params_.key_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(attention->params_.value_bias.data()), attention->params_.value_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(attention->params_.output_bias.data()), attention->params_.output_bias.size() * sizeof(float));

    DEBUG_COUT << "=== MultiHeadAttention::load END ===\n" << std::endl;
    return attention;
}

void MultiHeadAttention::update_parameters(float learning_rate) {
    DEBUG_LOG(SCOPE_LOG());
    
    // Adam (matches the LM head's optimizer; SGD-momentum at Adam-scale LRs
    // left these weights near init — the 2026-07-13 loss-6.0 plateau).
    const float beta = transformer_runtime::adam_beta1;
    const float beta2 = transformer_runtime::adam_beta2;
    const float adam_eps = 1e-8f;
    const float clip_threshold = 1.0f;

    // Initialize optimizer state on first call
    if (!momentum_.initialized) {
        momentum_.query_m = Matrix(params_.query_weights.rows(), params_.query_weights.cols(), 0.0f);
        momentum_.key_m = Matrix(params_.key_weights.rows(), params_.key_weights.cols(), 0.0f);
        momentum_.value_m = Matrix(params_.value_weights.rows(), params_.value_weights.cols(), 0.0f);
        momentum_.output_m = Matrix(params_.output_weights.rows(), params_.output_weights.cols(), 0.0f);
        momentum_.query_v = Matrix(params_.query_weights.rows(), params_.query_weights.cols(), 0.0f);
        momentum_.key_v = Matrix(params_.key_weights.rows(), params_.key_weights.cols(), 0.0f);
        momentum_.value_v = Matrix(params_.value_weights.rows(), params_.value_weights.cols(), 0.0f);
        momentum_.output_v = Matrix(params_.output_weights.rows(), params_.output_weights.cols(), 0.0f);
        momentum_.query_bias_m = FloatVector(params_.query_bias.size(), 0.0f);
        momentum_.key_bias_m = FloatVector(params_.key_bias.size(), 0.0f);
        momentum_.value_bias_m = FloatVector(params_.value_bias.size(), 0.0f);
        momentum_.output_bias_m = FloatVector(params_.output_bias.size(), 0.0f);
        momentum_.query_bias_v = FloatVector(params_.query_bias.size(), 0.0f);
        momentum_.key_bias_v = FloatVector(params_.key_bias.size(), 0.0f);
        momentum_.value_bias_v = FloatVector(params_.value_bias.size(), 0.0f);
        momentum_.output_bias_v = FloatVector(params_.output_bias.size(), 0.0f);
        momentum_.initialized = true;
    }
    momentum_.t += 1;
    const float bc1 = 1.0f - std::pow(beta, static_cast<float>(momentum_.t));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(momentum_.t));

    // Helper: Adam update with per-tensor clipping for matrices
    auto momentum_update = [&](Matrix& param, const Matrix& grad, Matrix& m, Matrix& v) {
        // Compute gradient norm for clipping
        float grad_norm_sq = 0.0f;
        for (size_t i = 0; i < grad.rows(); ++i) {
            for (size_t j = 0; j < grad.cols(); ++j) {
                grad_norm_sq += grad(i, j) * grad(i, j);
            }
        }
        float grad_norm = std::sqrt(grad_norm_sq);
        float clip_scale = (grad_norm > clip_threshold) ? (clip_threshold / (grad_norm + 1e-8f)) : 1.0f;

        for (size_t i = 0; i < param.rows(); ++i) {
            for (size_t j = 0; j < param.cols(); ++j) {
                float g = grad(i, j) * clip_scale;
                m(i, j) = beta * m(i, j) + (1.0f - beta) * g;
                v(i, j) = beta2 * v(i, j) + (1.0f - beta2) * g * g;
                param(i, j) -= learning_rate * (m(i, j) / bc1)
                               / (std::sqrt(v(i, j) / bc2) + adam_eps);
            }
        }
    };

    // Helper: Adam update for bias vectors
    auto momentum_update_bias = [&](FloatVector& param, const FloatVector& grad,
                                    FloatVector& m, FloatVector& v) {
        float grad_norm_sq = 0.0f;
        for (size_t i = 0; i < grad.size(); ++i) {
            grad_norm_sq += grad[i] * grad[i];
        }
        float grad_norm = std::sqrt(grad_norm_sq);
        float clip_scale = (grad_norm > clip_threshold) ? (clip_threshold / (grad_norm + 1e-8f)) : 1.0f;

        for (size_t i = 0; i < param.size(); ++i) {
            float g = grad[i] * clip_scale;
            m[i] = beta * m[i] + (1.0f - beta) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            param[i] -= learning_rate * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + adam_eps);
        }
    };
    
    // Log gradients before update
    GRAD_LOG_MATRIX("attention_query_grad", grads_.query_grad);
    GRAD_LOG_MATRIX("attention_output_grad", grads_.output_grad);
    
    // Update weight matrices with momentum. Under LoRA fine-tuning the base
    // weights are frozen: the gradient is projected onto the rank-r adapters
    // and the composed weight is refreshed (see lora.hpp).
    if (lora::settings().enabled) {
        if (!lora_q_.attached()) {
            const auto& s = lora::settings();
            lora_q_.attach(params_.query_weights, s.rank, s.alpha, 101);
            lora_k_.attach(params_.key_weights, s.rank, s.alpha, 102);
            lora_v_.attach(params_.value_weights, s.rank, s.alpha, 103);
            lora_o_.attach(params_.output_weights, s.rank, s.alpha, 104);
        }
        lora_q_.update(params_.query_weights, grads_.query_grad, learning_rate);
        lora_k_.update(params_.key_weights, grads_.key_grad, learning_rate);
        lora_v_.update(params_.value_weights, grads_.value_grad, learning_rate);
        lora_o_.update(params_.output_weights, grads_.output_grad, learning_rate);
    } else {
        momentum_update(params_.query_weights, grads_.query_grad, momentum_.query_m, momentum_.query_v);
        momentum_update(params_.key_weights, grads_.key_grad, momentum_.key_m, momentum_.key_v);
        momentum_update(params_.value_weights, grads_.value_grad, momentum_.value_m, momentum_.value_v);
        momentum_update(params_.output_weights, grads_.output_grad, momentum_.output_m, momentum_.output_v);
    }
    
    // Log momentum state
    GRAD_LOG_MATRIX("attention_output_momentum", momentum_.output_m);
    
    // Update bias vectors with momentum (frozen at zero in LLaMA mode -
    // the llama architecture has no attention biases)
    if (!transformer_runtime::llama_no_bias) {
        momentum_update_bias(params_.query_bias, grads_.query_bias_grad, momentum_.query_bias_m, momentum_.query_bias_v);
        momentum_update_bias(params_.key_bias, grads_.key_bias_grad, momentum_.key_bias_m, momentum_.key_bias_v);
        momentum_update_bias(params_.value_bias, grads_.value_bias_grad, momentum_.value_bias_m, momentum_.value_bias_v);
        momentum_update_bias(params_.output_bias, grads_.output_bias_grad, momentum_.output_bias_m, momentum_.output_bias_v);
    }
    
    // Zero out gradients for next iteration
    grads_.query_grad.initialize_constant(0.0f);
    grads_.key_grad.initialize_constant(0.0f);
    grads_.value_grad.initialize_constant(0.0f);
    grads_.output_grad.initialize_constant(0.0f);
    grads_.query_bias_grad.initialize_constant(0.0f);
    grads_.key_bias_grad.initialize_constant(0.0f);
    grads_.value_bias_grad.initialize_constant(0.0f);
    grads_.output_bias_grad.initialize_constant(0.0f);
}

// Add helper function for matrix statistics
void MultiHeadAttention::print_matrix_stats(const std::string& name, const Matrix& mat) const {
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    float sum_val = 0.0f;
    size_t nonzero = 0;
    
    for (size_t i = 0; i < mat.rows(); i++) {
        for (size_t j = 0; j < mat.cols(); j++) {
            float val = mat(i, j);
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum_val += val;
            if (std::abs(val) > 1e-6) nonzero++;
        }
    }
    
    float mean = sum_val / (mat.rows() * mat.cols());
    
    DEBUG_COUT << name << " stats:"
              << "\n  Shape: " << mat.rows() << "x" << mat.cols()
              << "\n  Min: " << min_val
              << "\n  Max: " << max_val
              << "\n  Mean: " << mean
              << "\n  Nonzero: " << nonzero << "/" << (mat.rows() * mat.cols())
              << "\n  Range: " << (max_val - min_val) << std::endl;
}

// Add Vector version of softmax_with_temperature
Vector MultiHeadAttention::softmax_with_temperature(const Vector& input, float temperature) const {
    Vector output(input.size());
    float max_val = -std::numeric_limits<float>::infinity();
    
    // Find max value for numerical stability
    for (size_t i = 0; i < input.size(); i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::exp((input[i] - max_val) / temperature);
        sum_exp += output[i];
    }
    
    // Normalize
    for (size_t i = 0; i < output.size(); i++) {
        output[i] /= sum_exp;
    }
    
    return output;
}

// Keep the existing Matrix version
Matrix MultiHeadAttention::softmax_with_temperature(const Matrix& input, float temperature) const {
    Matrix output = input;
    for (size_t i = 0; i < input.rows(); i++) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < input.cols(); j++) {
            max_val = std::max(max_val, input(i, j));
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < input.cols(); j++) {
            output(i, j) = std::exp((input(i, j) - max_val) / temperature);
            sum_exp += output(i, j);
        }
        
        for (size_t j = 0; j < input.cols(); j++) {
            output(i, j) /= sum_exp;
        }
    }
    return output;
}

// Add implementation of static method for AttentionMask
AttentionMask AttentionMask::create_causal_mask(size_t size) {
    Matrix mask_matrix(size, size, 0.0f);
    // Create lower triangular matrix (1s below diagonal, 0s above)
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j <= i; j++) {
            mask_matrix(i, j) = 1.0f;
        }
    }
    return AttentionMask(mask_matrix);
}

AttentionMask AttentionMask::create_separator_mask(const std::vector<int>& tokens, size_t size) {
    Matrix mask_matrix(size, size, 1.0f);  // Start with full attention
    
    // Track separator positions and types
    std::vector<std::pair<size_t, char>> separators;
    for (size_t i = 0; i < tokens.size(); i++) {
        char sep_type = get_separator_type(tokens[i]);
        if (sep_type != '\0') {
            separators.push_back({i, sep_type});
        }
    }
    
    // Apply separator-specific attention rules
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            // Find the separators that bound these positions
            char i_type = '\0', j_type = '\0';
            for (const auto& [pos, type] : separators) {
                if (pos <= i) i_type = type;
                if (pos <= j) j_type = type;
            }
            
            // Apply attention rules based on separator types
            if (i_type != '\0' && j_type != '\0') {
                // Same type separators can attend to each other
                if (i_type != j_type) {
                    mask_matrix(i, j) = 0.0f;  // Prevent cross-type attention
                }
                
                // Special rules for different separator types
                switch (i_type) {
                    case '#':  // Verb phrases
                        // Verbs can attend to general context but not adjectives
                        if (j_type == '*') mask_matrix(i, j) = 0.0f;
                        break;
                    case '*':  // Adjective phrases
                        // Adjectives can attend to general context but not verbs
                        if (j_type == '#') mask_matrix(i, j) = 0.0f;
                        break;
                    case '|':  // General phrases
                        // General phrases can attend to everything
                        break;
                }
            }
        }
    }
    
    return AttentionMask(mask_matrix);
}

void AttentionMask::apply_separator_rules(const std::vector<int>& tokens) {
    if (!has_mask_ || mask_.empty()) return;
    
    size_t size = mask_.rows();
    
    // Find all separators and their types
    std::vector<std::pair<size_t, char>> separators;
    for (size_t i = 0; i < tokens.size(); i++) {
        char sep_type = get_separator_type(tokens[i]);
        if (sep_type != '\0') {
            separators.push_back({i, sep_type});
        }
    }
    
    // Apply attention modifications based on separator types
    for (size_t i = 0; i < size; i++) {
        // Find the current context type
        char current_type = '\0';
        for (const auto& [pos, type] : separators) {
            if (pos <= i) current_type = type;
        }
        
        if (current_type != '\0') {
            for (size_t j = 0; j < size; j++) {
                // Find the target context type
                char target_type = '\0';
                for (const auto& [pos, type] : separators) {
                    if (pos <= j) target_type = type;
                }
                
                // Apply type-specific attention rules
                if (target_type != '\0') {
                    float attention_weight = 1.0f;
                    
                    // Adjust attention weights based on separator types
                    switch (current_type) {
                        case '#':  // Verb phrases
                            if (target_type == '*') attention_weight = 0.0f;  // Verbs don't attend to adjectives
                            else if (target_type == '|') attention_weight = 0.8f;  // Reduced attention to general context
                            break;
                            
                        case '*':  // Adjective phrases
                            if (target_type == '#') attention_weight = 0.0f;  // Adjectives don't attend to verbs
                            else if (target_type == '|') attention_weight = 0.8f;  // Reduced attention to general context
                            break;
                            
                        case '|':  // General phrases
                            // General phrases can attend to everything but with varying weights
                            if (target_type == '#') attention_weight = 0.9f;  // Strong attention to verbs
                            else if (target_type == '*') attention_weight = 0.9f;  // Strong attention to adjectives
                            break;
                    }
                    
                    mask_(i, j) *= attention_weight;
                }
            }
        }
    }
}

int AttentionMask::find_separator_position(const std::vector<int>& tokens) {
    // Search from the end to find the last separator
    for (int i = static_cast<int>(tokens.size()) - 1; i >= 0; i--) {
        char sep_type = get_separator_type(tokens[i]);
        if (sep_type != '\0') {
            return i;
        }
    }
    return -1;
}

#ifdef USE_CUDA
Matrix MultiHeadAttention::forward_fused(const Matrix& input, const AttentionMask& mask) {
    // Input: [total_seq, hidden_size] where total_seq could be batch_size * seq_len
    size_t total_seq = input.rows();
    
    // For fused kernel, we need to know batch_size and seq_len separately
    // The mask dimensions tell us seq_len (if provided), otherwise assume single sequence
    size_t seq_len = total_seq;  // Default: treat as single sequence
    size_t batch_size = 1;
    
    // If we have a block-diagonal mask, infer batch_size from mask structure
    // For now, use simple heuristic: if total_seq > 512, assume batching
    if (total_seq > 512) {
        // Assume seq_len = 128 (common value) or infer from mask
        seq_len = 128;  // TODO: Make this configurable
        batch_size = total_seq / seq_len;
        if (batch_size * seq_len != total_seq) {
            // Fall back to single sequence if doesn't divide evenly
            seq_len = total_seq;
            batch_size = 1;
        }
    }
    
    // Create combined QKV weights matrix [hidden_size, 3*hidden_size]
    Matrix qkv_weights(hidden_size, 3 * hidden_size);
    FloatVector qkv_bias(3 * hidden_size);
    
    // Combine Q, K, V weights and biases
    for (size_t i = 0; i < hidden_size; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            qkv_weights(i, j) = params_.query_weights(i, j);
            qkv_weights(i, j + hidden_size) = params_.key_weights(i, j);
            qkv_weights(i, j + 2 * hidden_size) = params_.value_weights(i, j);
        }
        qkv_bias[i] = params_.query_bias[i];
        qkv_bias[i + hidden_size] = params_.key_bias[i];
        qkv_bias[i + 2 * hidden_size] = params_.value_bias[i];
    }
    
    // Prepare output matrix
    Matrix output(total_seq, hidden_size);
    
    // Launch fused attention kernel (handles causal masking internally)
    cuda::launch_fused_attention_kernel(
        input.data(),
        qkv_weights.data(),
        qkv_bias.data(),
        params_.output_weights.data(),
        output.data(),
        nullptr,  // Causal mask is handled inside kernel
        static_cast<int>(batch_size),
        static_cast<int>(seq_len),
        static_cast<int>(hidden_size),
        static_cast<int>(num_heads)
    );
    
    // Add output bias
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(output.rows()); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output(i, j) += params_.output_bias[j];
        }
    }
    
    // Cache for backward pass (simplified - compute Q, K, V on CPU for gradient caching)
    cached_query_layer = matmul(input, params_.query_weights);
    cached_key_layer = matmul(input, params_.key_weights);
    cached_value_layer = matmul(input, params_.value_weights);
    cached_attn_output = output;  // Pre-bias output
    cached_attention_weights = Matrix(num_heads * total_seq, total_seq);  // Placeholder
    
    return output;
}
#endif

