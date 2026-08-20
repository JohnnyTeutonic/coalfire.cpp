#include "../include/lm_head.hpp"
#include "../include/config.hpp"
#include "../include/lora.hpp"
#include "../include/token_constants.hpp"
#include "../include/cuda/matrix_ops.cuh"  // Add CUDA matrix operations header
#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
#include "../include/cuda/loss_kernels.cuh"  // Device-resident projection + loss
#endif
#include "../include/scope_logger.hpp"  // Add scope logger header
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <random>
#include <deque>
#include <cassert>

// Add minimum active tokens constant
constexpr size_t MIN_ACTIVE_TOKENS = 1000;  // Reasonable default value

// Only include CUDA headers if CUDA is available
#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

LanguageModelHead::LanguageModelHead(size_t hidden_size, size_t vocab_size)
    : hidden_size_(hidden_size), vocab_size_(vocab_size), 
      projection(hidden_size, vocab_size),  // [hidden_size x vocab_size]
      bias(vocab_size, 0.0f),
      token_frequencies(vocab_size, 0.0f),
      pruning_threshold(1e-6f),
      active_tokens(vocab_size, 1),
      training_steps(0),
      is_training_(false),
      m_proj(hidden_size, vocab_size, 0.0f),  // [hidden_size x vocab_size]
      v_proj(hidden_size, vocab_size, 0.0f),  // [hidden_size x vocab_size]
      m_bias(vocab_size, 0.0f),
      v_bias(vocab_size, 0.0f),
      t(0),
      beta1(transformer_runtime::adam_beta1),
      beta2(transformer_runtime::adam_beta2),
      eps(1e-8f),
      current_lr(0.001f),
      min_lr(0.0001f),
      max_lr(0.01f),
      lr_decay(0.99f) {
    
    SCOPE_LOG();
    
    std::cout << "Initializing LM head with:"
              << "\n- Hidden size: " << hidden_size
              << "\n- Vocabulary size: " << vocab_size << std::endl;
    
    // Initialize projection matrix with Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (hidden_size + vocab_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    // Initialize projection weights
    for (size_t i = 0; i < projection.rows(); i++) {
        for (size_t j = 0; j < projection.cols(); j++) {
            projection(i, j) = dist(gen);
        }
    }
}

Matrix LanguageModelHead::forward(const Matrix& hidden_states, bool training) {
    if (hidden_states.cols() != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch in LM head");
    }
    
    // Cache hidden states for backward pass
    hidden_states_ = hidden_states;
    
#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
    if (training) {
        // DEVICE-RESIDENT TRAINING PATH:
        // Weights (projection/bias) live on the GPU and are updated there by the
        // Adam step in backward_pass_cuda; they are uploaded once and synced back
        // to host only for eval/checkpointing. Project hidden -> logits directly
        // into the persistent device logits buffer (bias added on device), so the
        // 655MB logits never round-trip to host. The training loop computes loss
        // from the device logits.
        cuda::lmhead_ensure_weights(
            projection.data(),
            bias.data(),
            static_cast<int>(hidden_size_),
            static_cast<int>(vocab_size_)
        );
        cuda::project_logits_to_device(
            hidden_states.data(),
            static_cast<int>(hidden_states.rows()),
            static_cast<int>(hidden_size_),
            static_cast<int>(vocab_size_)
        );
        // Logits live on device; return empty placeholder (training loop reads
        // dimensions from the model config, not from this matrix).
        return Matrix();
    }
#endif

    // Project hidden states to vocabulary space: [seq_len x hidden] @ [hidden x vocab] = [seq_len x vocab]
    Matrix logits(hidden_states.rows(), vocab_size_);
    
#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
    // Use CUDA cuBLAS for GPU acceleration (eval / inference path)
    cuda::matmul(hidden_states, projection, logits);
#else
    // CPU fallback
    logits = matmul(hidden_states, projection);
#endif
    
    // Add bias using OpenMP for speed
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(logits.rows()); ++i) {
        for (size_t j = 0; j < logits.cols(); ++j) {
            logits(i, j) += bias[j];
        }
    }
    
    return logits;
}

Matrix LanguageModelHead::project_to_vocab(const Matrix& hidden_states) {
    SCOPE_LOG();
    this->hidden_states = hidden_states;
    size_t total_size = hidden_states.rows();
    size_t hidden_dim = hidden_states.cols();

    if (hidden_dim != hidden_size_) {
        throw std::runtime_error("Hidden dimension mismatch: " + std::to_string(hidden_dim) +
                               " != " + std::to_string(hidden_size_));
    }
    
    return forward(hidden_states, false);
}

Matrix LanguageModelHead::backward(const Matrix& grad_output, const Matrix& target_distribution) {
    SCOPE_LOG();
    // Verify input dimensions
    if (grad_output.cols() != vocab_size_) {
        throw std::runtime_error("Gradient output dimension mismatch in backward pass. Expected vocab_size: " + 
                               std::to_string(vocab_size_) + ", got: " + std::to_string(grad_output.cols()));
    }

    // Use backward_pass which already has Adam optimization
    Matrix grad_hidden = backward_pass(grad_output, hidden_states_);
    
    return grad_hidden;
}

void LanguageModelHead::backward_linear(const Matrix& grad_output) {
    // Use backward_pass which already has Adam optimization
    backward_pass(grad_output, hidden_states_);
}

void LanguageModelHead::update_learning_rate(float current_loss) {
    SCOPE_LOG();
    // Add loss to history
    loss_history.push_back(current_loss);
    if (loss_history.size() > LOSS_HISTORY_SIZE) {
        loss_history.pop_front();
    }
    
    // Only adjust learning rate if we have enough history
    if (loss_history.size() >= 2) {
        float avg_recent_loss = 0.0f;
        float avg_old_loss = 0.0f;
        size_t recent_count = loss_history.size() / 2;
        
        // Calculate average of recent and older losses
        for (size_t i = 0; i < loss_history.size(); i++) {
            if (i >= loss_history.size() - recent_count) {
                avg_recent_loss += loss_history[i];
            } else {
                avg_old_loss += loss_history[i];
            }
        }
        avg_recent_loss /= recent_count;
        avg_old_loss /= (loss_history.size() - recent_count);
        
        // Adjust learning rate based on loss trend
        if (avg_recent_loss < avg_old_loss) {
            // Loss is decreasing, increase learning rate slightly
            current_lr = std::min(max_lr, current_lr * lr_growth);
        } else {
            // Loss is increasing or stagnant, decrease learning rate
            current_lr = std::max(min_lr, current_lr * lr_decay);
        }
    }
    
    prev_loss = current_loss;
}

void LanguageModelHead::update_token_frequencies(const std::vector<int>& tokens) {
    SCOPE_LOG();
    // Reset frequencies periodically to prevent over-accumulation (MSVC: loop vars must be signed int)
    if (training_steps % 1000 == 0) {  // Reset every 1000 steps
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(token_frequencies.size()); i++) {
            token_frequencies[i] = 0.0f;
        }
    }
    
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(tokens.size()); i++) {
        int token = tokens[i];
        if (token >= 0 && static_cast<size_t>(token) < vocab_size_) {
            #pragma omp atomic
            token_frequencies[token] += 1.0f;
        }
    }
    training_steps++;
    
    // Normalize frequencies to prevent extreme values
    if (!token_frequencies.empty()) {
        float max_freq = *std::max_element(token_frequencies.begin(), token_frequencies.end());
        if (max_freq > 0) {
            // MSVC: loop vars must be signed int
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(token_frequencies.size()); i++) {
                token_frequencies[i] /= max_freq;  // Normalize to [0,1] range
            }
        }
    }
}

void LanguageModelHead::update_active_tokens() {
    SCOPE_LOG();
    const float decay = 0.99f;
    
    // Parallelize frequency decay (MSVC: loop vars must be signed int)
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(vocab_size_); i++) {
        token_frequencies[i] *= decay;
    }
    
    size_t active_count = 0;
    active_token_indices.clear();
    
    // Use vector of pairs to avoid multiple passes
    std::vector<std::pair<float, size_t>> freq_pairs(vocab_size_);
    
    // MSVC: loop vars must be signed int
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(vocab_size_); i++) {
        freq_pairs[i] = {token_frequencies[i], i};
    }
    
    // Partial sort only what we need
    std::partial_sort(freq_pairs.begin(), 
                     freq_pairs.begin() + std::min(MIN_ACTIVE_TOKENS, vocab_size_),
                     freq_pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Reset active tokens
    std::fill(active_tokens.begin(), active_tokens.end(), 0);
    active_token_indices.clear();
    active_token_indices.reserve(std::min(MIN_ACTIVE_TOKENS, vocab_size_));
    
    // Set active tokens based on sorted frequencies
    for (size_t i = 0; i < std::min(MIN_ACTIVE_TOKENS, vocab_size_); i++) {
        size_t idx = freq_pairs[i].second;
        active_tokens[idx] = 1;
        active_token_indices.push_back(idx);
    }
}

#ifdef __CUDACC__
// GPU kernel for FP16 conversion (only compiled with nvcc, not MSVC)
__global__ void convert_projection_to_fp16_kernel(
    half* output, const float* input, const unsigned char* active_tokens,
    size_t hidden_size, size_t vocab_size) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size * vocab_size && active_tokens[idx / hidden_size]) {
        output[idx] = __float2half(input[idx]);
    }
}
#endif

LanguageModelHead::~LanguageModelHead() {
#ifdef USE_CUDA
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
    }
    if (d_projection) cudaFree(d_projection);
    if (d_bias) cudaFree(d_bias);
    if (d_projection_fp16) cudaFree(d_projection_fp16);
    if (d_hidden_states_fp16) cudaFree(d_hidden_states_fp16);
    if (d_output_fp16) cudaFree(d_output_fp16);
    if (d_output) cudaFree(d_output);
    if (d_active_tokens) cudaFree(d_active_tokens);
    if (d_active_token_indices) cudaFree(d_active_token_indices);
    if (compute_stream) cudaStreamDestroy(compute_stream);
#endif
}

void LanguageModelHead::set_training(bool training_mode) {
    SCOPE_LOG();
    is_training_ = training_mode;
}

Matrix LanguageModelHead::backward_pass(const Matrix& grad_output, const Matrix& hidden_states) {
    SCOPE_LOG();  // Add scope logging
    
    // Verify input dimensions
    if (grad_output.cols() != vocab_size_) {
        throw std::runtime_error("Gradient output dimension mismatch in backward pass. Expected vocab_size: " + 
                               std::to_string(vocab_size_) + ", got: " + std::to_string(grad_output.cols()));
    }
    if (hidden_states.cols() != hidden_size_) {
        throw std::runtime_error("Hidden states dimension mismatch in backward pass. Expected hidden_size: " + 
                               std::to_string(hidden_size_) + ", got: " + std::to_string(hidden_states.cols()));
    }

    // Compute gradients for projection matrix (normalized by batch size)
    // hidden_states.T: [hidden_size x batch_size]
    // grad_output: [batch_size x vocab_size]
    // grad_proj: [hidden_size x vocab_size]
    const float batch_size = static_cast<float>(grad_output.rows());
    const float grad_scale = 1.0f / batch_size;
    
    Matrix hidden_states_t = hidden_states.transpose();
    Matrix grad_proj = matmul(hidden_states_t, grad_output);
    grad_proj *= grad_scale;  // Average over batch, not sum!
    
    // Compute bias gradients (normalized)
    Vector grad_bias = grad_output.row_sum();
    for (size_t i = 0; i < grad_bias.size(); ++i) {
        grad_bias[i] *= grad_scale;
    }
    
    // Update parameters using Adam optimizer
    t++;  // Increment time step
    
    // Constants for gradient clipping and stability
    const float clip_threshold = 5.0f;
    const float max_allowed_value = 100.0f;
    const float scale_factor = std::sqrt(1.0f / hidden_size_);
    const float max_update = 0.05f * scale_factor;
    
    bool has_unstable_update = false;
    // MSVC: collapse ignored, loop vars must be signed int
    #pragma omp parallel for collapse(2) reduction(|:has_unstable_update)
    for (int i = 0; i < static_cast<int>(grad_proj.rows()); ++i) {
        for (int j = 0; j < static_cast<int>(grad_proj.cols()); ++j) {
            try {
                if (!std::isfinite(grad_proj(i, j))) {
                    std::cout << "Non-finite gradient at (" << i << "," << j << ")" << std::endl;
                    continue;
                }
                
                // Verify matrix access is in bounds
                if (i >= m_proj.rows() || j >= m_proj.cols()) {
                    std::cout << "Out of bounds access attempt on m_proj: (" << i << "," << j 
                              << ") with dims " << m_proj.rows() << "x" << m_proj.cols() << std::endl;
                    continue;
                }
                if (i >= v_proj.rows() || j >= v_proj.cols()) {
                    std::cout << "Out of bounds access attempt on v_proj: (" << i << "," << j 
                              << ") with dims " << v_proj.rows() << "x" << v_proj.cols() << std::endl;
                    continue;
                }
                if (i >= projection.rows() || j >= projection.cols()) {
                    std::cout << "Out of bounds access attempt on projection: (" << i << "," << j 
                              << ") with dims " << projection.rows() << "x" << projection.cols() << std::endl;
                    continue;
                }

                float clipped_grad = grad_proj(i, j);
                if (std::abs(clipped_grad) > clip_threshold) {
                    clipped_grad *= clip_threshold / std::abs(clipped_grad);
                }
                
                // Adam optimizer update
                float new_m = beta1 * m_proj(i, j) + (1.0f - beta1) * clipped_grad;
                float new_v = beta2 * v_proj(i, j) + (1.0f - beta2) * clipped_grad * clipped_grad;
                
                // Bias correction
                float m_hat = new_m / (1.0f - std::pow(beta1, t));
                float v_hat = new_v / (1.0f - std::pow(beta2, t));
                
                // Store updated momentum and velocity
                m_proj(i, j) = new_m;
                v_proj(i, j) = new_v;
                
                // Compute update with learning rate and bias correction
                float update = current_lr * m_hat / (std::sqrt(v_hat) + eps);
                
                // Clip update magnitude
                if (std::abs(update) > max_update) {
                    update *= max_update / std::abs(update);
                }
                
                // Apply update to projection matrix
                float new_value = projection(i, j) - update;
                
                // Clip parameter value
                if (std::abs(new_value) > max_allowed_value) {
                    new_value = std::copysign(max_allowed_value, new_value);
                    has_unstable_update = true;
                }
                
                projection(i, j) = new_value;
                
            } catch (const std::exception& e) {
                std::cout << "Error in parameter update: " << e.what() << std::endl;
                has_unstable_update = true;
            }
        }
    }
    
    // Compute gradient with respect to input
    // grad_output: [batch_size x vocab_size]
    // projection.T: [vocab_size x hidden_size]
    Matrix projection_t = projection.transpose();
    Matrix grad_input = matmul(grad_output, projection_t);
    
    // Verify output dimensions
    if (grad_input.cols() != hidden_size_) {
        throw std::runtime_error("Output gradient dimension mismatch. Expected hidden_size: " + 
                               std::to_string(hidden_size_) + ", got: " + std::to_string(grad_input.cols()));
    }
    
    // Gradient logging disabled for performance
    
    return grad_input;
}

#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
void LanguageModelHead::backward_pass_cuda(float* d_grad_logits, int num_positions, Matrix& grad_hidden_out) {
    SCOPE_LOG();
    // Device-resident LM head backward + Adam update. Consumes the gradient that
    // is already on the GPU (no 655MB D2H), updates the resident projection
    // weights in place, and returns grad_hidden (~16MB) for the rest of the
    // backward pass. Uses the same Adam hyperparameters as the CPU path; the
    // bias is intentionally not updated (matches backward_pass()).
    t++;  // Match CPU: increment time step before the update.

    if (grad_hidden_out.rows() != num_positions ||
        grad_hidden_out.cols() != hidden_size_) {
        grad_hidden_out = Matrix(num_positions, hidden_size_);
    }

    // Under LoRA fine-tuning the LM head is frozen: lr=0 leaves the resident
    // weights untouched while grad_hidden still flows to the layers below.
    const float effective_lr = lora::settings().enabled ? 0.0f : current_lr;
    cuda::lmhead_backward_device(
        d_grad_logits,
        num_positions,
        static_cast<int>(hidden_size_),
        static_cast<int>(vocab_size_),
        beta1, beta2, eps, effective_lr,
        static_cast<int>(t),
        grad_hidden_out.data()
    );
}
#endif

void LanguageModelHead::sync_weights_from_device() {
#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
    // Pull the resident device weights back into the host matrices so that
    // eval/inference and checkpoint saving observe the latest values.
    cuda::lmhead_sync_weights_to_host(projection.data(), bias.data());
#endif
}

void LanguageModelHead::invalidate_device_weights() {
#if defined(USE_CUDA) && defined(CUDA_AVAILABLE)
    cuda::lmhead_invalidate_weights();
#endif
}

void LanguageModelHead::bias_completion_format(Matrix& logits) {
    if (!tokenizer) {
        return;  // Skip biasing if tokenizer is not set
    }

    // Get special token IDs from tokenizer
    const int sep_token_id = tokenizer->get_sep_token_id();
    
    // Get the last predicted token
    int last_token = -1;  // You'll need to track this
    
    // After separator token, boost probability of tokens that commonly start completions
    if (last_token == sep_token_id) {
        // Boost tokens that typically start completions (e.g., space token)
        // This helps enforce the format where completions start with a space
        const float boost_factor = 2.0f;
        for (size_t i = 0; i < logits.rows(); i++) {
            std::string token = tokenizer->decode({static_cast<int>(i)});
            if (!token.empty() && token[0] == ' ') {
                logits.data()[i] *= boost_factor;
            }
        }
    }
}

Vector LanguageModelHead::sample_next_token(const Matrix& logits, const std::string& input_str, float temperature) {
    std::cout << "\n=== Starting LanguageModelHead::sample_next_token ===" << std::endl;
    std::cout << "Input matrix shape: " << logits.rows() << "x" << logits.cols() << std::endl;
    std::cout << "Input context: '" << input_str << "'" << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    
    // Initialize vocabulary if needed
    if (!vocabulary_initialized) {
        std::cout << "Initializing vocabulary..." << std::endl;
        ensure_vocabulary_initialized();
    }
    
    // Get logits for last token
    std::cout << "Extracting last row of logits..." << std::endl;
    Vector token_logits = logits.row(logits.rows() - 1);
    std::cout << "Token logits size: " << token_logits.size() << std::endl;
    
    // Print initial logits statistics
    float min_logit = std::numeric_limits<float>::max();
    float max_logit = -std::numeric_limits<float>::max();
    float sum_logits = 0.0f;
    for (size_t i = 0; i < token_logits.size(); i++) {
        min_logit = std::min(min_logit, token_logits[i]);
        max_logit = std::max(max_logit, token_logits[i]);
        sum_logits += token_logits[i];
    }
    std::cout << "Initial logits statistics:" << std::endl;
    std::cout << "  Min: " << min_logit << std::endl;
    std::cout << "  Max: " << max_logit << std::endl;
    std::cout << "  Mean: " << sum_logits / token_logits.size() << std::endl;
    
    // Store original logits for debugging
    Vector original_logits = token_logits;
    
    // Apply context-aware adjustments before temperature scaling
    if (tokenizer) {
        std::cout << "Applying context-aware adjustments..." << std::endl;
        Matrix logits_matrix(1, token_logits.size());
        for (size_t i = 0; i < token_logits.size(); i++) {
            logits_matrix(0, i) = token_logits[i];
        }
        bias_completion_format(logits_matrix);
        token_logits = logits_matrix.row(0);
    }
    
    // Apply temperature scaling after adjustments
    std::cout << "Applying temperature scaling (T=" << temperature << ")..." << std::endl;
    float effective_temp = std::max(temperature, 0.1f); // Prevent division by zero
    for (size_t i = 0; i < token_logits.size(); i++) {
        token_logits[i] /= effective_temp;
    }
    
    // Convert to probabilities with softmax
    std::cout << "Converting to probabilities..." << std::endl;
    Vector probabilities = softmax(token_logits);
    
    // Print probability statistics
    float min_prob = 1.0f;
    float max_prob = 0.0f;
    float sum_prob = 0.0f;
    size_t non_zero_probs = 0;
    for (size_t i = 0; i < probabilities.size(); i++) {
        if (probabilities[i] > 0.0f) {
            min_prob = std::min(min_prob, probabilities[i]);
            max_prob = std::max(max_prob, probabilities[i]);
            sum_prob += probabilities[i];
            non_zero_probs++;
        }
    }
    std::cout << "Probability statistics:" << std::endl;
    std::cout << "  Min (non-zero): " << min_prob << std::endl;
    std::cout << "  Max: " << max_prob << std::endl;
    std::cout << "  Sum: " << sum_prob << std::endl;
    std::cout << "  Non-zero probabilities: " << non_zero_probs << "/" << probabilities.size() << std::endl;
    
    // Debug: Print top 5 tokens before sampling
    std::cout << "\nGathering top 5 tokens..." << std::endl;
    std::vector<std::pair<float, size_t>> top_tokens;
    top_tokens.reserve(probabilities.size());
    for (size_t i = 0; i < probabilities.size(); i++) {
        top_tokens.push_back({probabilities[i], i});
    }
    
    std::cout << "Sorting tokens..." << std::endl;
    std::partial_sort(top_tokens.begin(), 
                     top_tokens.begin() + std::min(size_t(5), top_tokens.size()),
                     top_tokens.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "\nTop 5 tokens before sampling:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), top_tokens.size()); i++) {
        std::string token_text = get_token_text(top_tokens[i].second);
        std::cout << i + 1 << ". '" << token_text << "' "
                 << "(ID: " << top_tokens[i].second 
                 << ", prob: " << top_tokens[i].first 
                 << ", logit: " << original_logits[top_tokens[i].second] << ")" << std::endl;
    }
    
    // Input-dependent seeding
    std::cout << "\nPreparing random sampling..." << std::endl;
    std::random_device rd;
    std::seed_seq seed{
        rd(),
        static_cast<unsigned>(std::hash<std::string>{}(input_str)),
        static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())
    };
    std::mt19937 gen(seed);
    
    // Get valid token indices (filter out very low probability tokens)
    std::cout << "Filtering valid tokens..." << std::endl;
    std::vector<size_t> valid_indices;
    valid_indices.reserve(probabilities.size());
    const float prob_threshold = 1e-6f;
    
    for (size_t i = 0; i < probabilities.size(); i++) {
        if (probabilities[i] > prob_threshold) {
            valid_indices.push_back(i);
        }
    }
    std::cout << "Valid tokens after filtering: " << valid_indices.size() << std::endl;
    
    if (valid_indices.empty()) {
        std::cout << "No valid tokens found, including all tokens..." << std::endl;
        valid_indices.resize(probabilities.size());
        std::iota(valid_indices.begin(), valid_indices.end(), 0);
    }
    
    // Create distribution for sampling
    std::cout << "Creating sampling distribution..." << std::endl;
    std::vector<float> valid_probs;
    valid_probs.reserve(valid_indices.size());
    float sum_probs = 0.0f;
    
    for (size_t idx : valid_indices) {
        valid_probs.push_back(probabilities[idx]);
        sum_probs += probabilities[idx];
    }
    
    // Renormalize probabilities
    std::cout << "Renormalizing probabilities..." << std::endl;
    if (sum_probs > 0.0f) {
        for (float& prob : valid_probs) {
            prob /= sum_probs;
        }
    }
    
    // Sample token using the normalized distribution
    std::cout << "Sampling token..." << std::endl;
    std::discrete_distribution<> dist(valid_probs.begin(), valid_probs.end());
    size_t sampled_idx = valid_indices[dist(gen)];
    
    // Create one-hot vector
    std::cout << "Creating one-hot vector result..." << std::endl;
    Vector result(probabilities.size(), 0.0f);
    result[sampled_idx] = 1.0f;
    
    // Debug output
    std::cout << "\nSampling details:" << std::endl;
    std::cout << "Input context: '" << input_str << "'" << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    std::cout << "Valid tokens: " << valid_indices.size() << "/" << probabilities.size() << std::endl;
    std::cout << "Sampled token: " << sampled_idx << " ('" << get_token_text(sampled_idx) << "')" << std::endl;
    std::cout << "Original logit: " << original_logits[sampled_idx] << std::endl;
    std::cout << "Final probability: " << probabilities[sampled_idx] << std::endl;
    
    std::cout << "=== LanguageModelHead::sample_next_token complete ===\n" << std::endl;
    return result;
}

void LanguageModelHead::set_tokenizer(std::shared_ptr<TiktokenTokenizer> tok) {
    std::cout << "Setting tokenizer..." << std::endl;
    if (!tok) {
        std::cerr << "Warning: Null tokenizer provided!" << std::endl;
        return;
    }
    tokenizer = tok;
    std::cout << "Tokenizer set successfully. Vocab size: " << tok->vocab_size() << std::endl;
    
    // Reset vocabulary cache when tokenizer changes
    vocabulary_initialized = false;
    vocabulary_cache.clear();
}

void LanguageModelHead::ensure_vocabulary_initialized() {
    std::cout << "Checking vocabulary initialization..." << std::endl;
    
    if (!tokenizer) {
        std::cerr << "Error: Cannot initialize vocabulary - tokenizer not set!" << std::endl;
        return;
    }
    
    if (!vocabulary_initialized) {
        std::cout << "Initializing vocabulary with size " << vocab_size_ << "..." << std::endl;
        vocabulary_cache.clear();
        vocabulary_cache.reserve(vocab_size_);
        
        for (size_t i = 0; i < vocab_size_; i++) {
            try {
                std::string token = tokenizer->decode({static_cast<int>(i)});
                vocabulary_cache.push_back(token);
                if (i < 10 || i > vocab_size_ - 10) {  // Print first and last 10 tokens
                    std::cout << "Token " << i << ": '" << token << "'" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error decoding token " << i << ": " << e.what() << std::endl;
                vocabulary_cache.push_back("");  // Add empty string for failed tokens
            }
        }
        
        vocabulary_initialized = true;
        std::cout << "Vocabulary initialization complete. Size: " << vocabulary_cache.size() << std::endl;
    }
}
