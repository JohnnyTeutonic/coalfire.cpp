#include "../include/feed_forward.hpp"
#ifdef USE_CUDA
#include "../include/cuda/cuda_matrix.hpp"
#include "../include/cuda/swiglu_kernels.cuh"
#include "../include/cuda/cuda_check.cuh"
#endif
#include "../include/config.hpp"
#include "../include/scope_logger.hpp"
#include "../include/gradient_diagnostics.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FeedForward::FeedForward(size_t hidden_size, size_t intermediate_size, float dropout)
    : dropout_prob(dropout) {
    SCOPE_LOG();
    // Initialize matrices with correct dimensions
    params_.gate_proj_weights = Matrix(hidden_size, intermediate_size);
    params_.up_proj_weights = Matrix(hidden_size, intermediate_size);
    params_.down_proj_weights = Matrix(intermediate_size, hidden_size);

    params_.gate_proj_bias = FloatVector(intermediate_size);
    params_.up_proj_bias = FloatVector(intermediate_size);
    params_.down_proj_bias = FloatVector(hidden_size);
    
    // Initialize gradients with same dimensions
    grads_.gate_proj_grad = Matrix(hidden_size, intermediate_size);
    grads_.up_proj_grad = Matrix(hidden_size, intermediate_size);
    grads_.down_proj_grad = Matrix(intermediate_size, hidden_size);

    grads_.gate_proj_bias_grad = FloatVector(intermediate_size);
    grads_.up_proj_bias_grad = FloatVector(intermediate_size);
    grads_.down_proj_bias_grad = FloatVector(hidden_size);

    // Initialize weights with Xavier/Glorot initialization
    initialize_weights();

    // Initialize gradients to zero
    grads_.gate_proj_grad.initialize_constant(0.0f);
    grads_.up_proj_grad.initialize_constant(0.0f);
    grads_.down_proj_grad.initialize_constant(0.0f);
    grads_.gate_proj_bias_grad.initialize_constant(0.0f);
    grads_.up_proj_bias_grad.initialize_constant(0.0f);
    grads_.down_proj_bias_grad.initialize_constant(0.0f);
}

// Helper for Swish activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float swish(float x) {
    return x * sigmoid(x);
}

inline float swish_derivative(float x) {
    float sig_x = sigmoid(x);
    return sig_x + x * sig_x * (1.0f - sig_x);
}


Matrix FeedForward::forward(const Matrix& input) {
    SCOPE_LOG();
    try {
#ifdef USE_CUDA
        // Convert input to CudaMatrix
        cuda::CudaMatrix d_input(input);

        // Ensure model parameters are on the GPU
        if (!params_gpu_) {
            params_gpu_ = std::make_unique<CudaParameters>();
            params_gpu_->gate_proj_weights = cuda::CudaMatrix(params_.gate_proj_weights);
            params_gpu_->up_proj_weights = cuda::CudaMatrix(params_.up_proj_weights);
            params_gpu_->down_proj_weights = cuda::CudaMatrix(params_.down_proj_weights);
        }
        
        // Allocate intermediate and output buffers on GPU
        cuda::CudaMatrix d_gated_linear_output(input.rows(), params_gpu_->gate_proj_weights.cols());
        cuda::CudaMatrix d_up_proj_output(input.rows(), params_gpu_->up_proj_weights.cols());
        cuda::CudaMatrix d_output(input.rows(), params_gpu_->down_proj_weights.cols());

        // Launch SwiGLU forward kernel
        cuda::kernels::swiglu_forward_kernel_launcher(
            d_input,
            params_gpu_->gate_proj_weights,
            params_gpu_->up_proj_weights,
            params_gpu_->down_proj_weights,
            d_gated_linear_output,
            d_up_proj_output,
            d_output
        );

        // Cache intermediate results for backward pass
        gated_linear_output_cache_gpu_ = std::move(d_gated_linear_output);
        up_proj_output_cache_gpu_ = std::move(d_up_proj_output);
        input_cache_gpu_ = std::move(d_input);
        
        // Copy result back to CPU
        return d_output.to_matrix();
#else
        // CPU implementation
        input_cache_ = input;

        // Gate and Up projections. Biases are added here; in llama mode they
        // are frozen at zero (use_biases=false) so this is a no-op, and in
        // vanilla mode they are the trained biases. (An earlier edit removed
        // these on a since-retracted misdiagnosis — see Finding 6 correction;
        // the real CPU/engine divergence was an unset-architecture config in
        // the golden-batch test, not the FFN biases.)
        Matrix gate = matmul(input, params_.gate_proj_weights);
        gate.add_bias(params_.gate_proj_bias);

        Matrix up = matmul(input, params_.up_proj_weights);
        up.add_bias(params_.up_proj_bias);

        // Swish activation on the gate (MSVC: loop vars must be signed int)
        Matrix gate_swish = gate;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < static_cast<int>(gate_swish.rows()); ++i) {
            for (int j = 0; j < static_cast<int>(gate_swish.cols()); ++j) {
                gate_swish(i, j) = swish(gate_swish(i, j));
            }
        }
        
        // Element-wise multiplication (MSVC: loop vars must be signed int)
        Matrix gated = gate_swish; // Re-use memory
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < static_cast<int>(gated.rows()); ++i) {
            for (int j = 0; j < static_cast<int>(gated.cols()); ++j) {
                gated(i, j) *= up(i, j);
            }
        }
        
        // Cache for backward pass
        intermediate_cache_ = gated;
        gate_cache_ = gate;
        up_cache_ = up;

        // Down projection (bias frozen at zero in llama mode)
        Matrix output = matmul(gated, params_.down_proj_weights);
        output.add_bias(params_.down_proj_bias);

        return output;
#endif

    } catch (const std::exception& e) {
        throw std::runtime_error("FeedForward forward failed: " + std::string(e.what()));
    }
}

void FeedForward::save(std::ostream& os) const {
    size_t hidden_size = params_.down_proj_weights.cols();
    size_t intermediate_size = params_.gate_proj_weights.cols();
    
    os.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    os.write(reinterpret_cast<const char*>(&intermediate_size), sizeof(intermediate_size));
    
    // Save weights
    params_.gate_proj_weights.save(os);
    params_.up_proj_weights.save(os);
    params_.down_proj_weights.save(os);
    
    // Save biases
    os.write(reinterpret_cast<const char*>(params_.gate_proj_bias.data()), params_.gate_proj_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.up_proj_bias.data()), params_.up_proj_bias.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(params_.down_proj_bias.data()), params_.down_proj_bias.size() * sizeof(float));
}

std::unique_ptr<FeedForward> FeedForward::load(std::istream& is) {
    size_t hidden_size, intermediate_size;
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    is.read(reinterpret_cast<char*>(&intermediate_size), sizeof(intermediate_size));
    
    auto ffn = std::make_unique<FeedForward>(hidden_size, intermediate_size);
    
    // Load weights
    ffn->params_.gate_proj_weights = Matrix::load(is);
    ffn->params_.up_proj_weights = Matrix::load(is);
    ffn->params_.down_proj_weights = Matrix::load(is);
    
    // Load biases
    is.read(reinterpret_cast<char*>(ffn->params_.gate_proj_bias.data()), ffn->params_.gate_proj_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->params_.up_proj_bias.data()), ffn->params_.up_proj_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(ffn->params_.down_proj_bias.data()), ffn->params_.down_proj_bias.size() * sizeof(float));
    
    return ffn;
}

Matrix FeedForward::backward(const Matrix& grad_output, const Matrix& original_input) {
    SCOPE_LOG();
    try {
#ifdef USE_CUDA
        // Convert grad_output to CudaMatrix
        cuda::CudaMatrix d_grad_output(grad_output);

        // Allocate gradient buffers on GPU
        cuda::CudaMatrix d_grad_input(original_input.rows(), original_input.cols());
        if (!grads_gpu_) {
            grads_gpu_ = std::make_unique<CudaGradients>();
            grads_gpu_->gate_proj_grad = cuda::CudaMatrix(grads_.gate_proj_grad.rows(), grads_.gate_proj_grad.cols());
            grads_gpu_->up_proj_grad = cuda::CudaMatrix(grads_.up_proj_grad.rows(), grads_.up_proj_grad.cols());
            grads_gpu_->down_proj_grad = cuda::CudaMatrix(grads_.down_proj_grad.rows(), grads_.down_proj_grad.cols());
        }


        // Launch SwiGLU backward kernel
        cuda::kernels::swiglu_backward_kernel_launcher(
            d_grad_output,
            input_cache_gpu_,
            params_gpu_->gate_proj_weights,
            params_gpu_->up_proj_weights,
            params_gpu_->down_proj_weights,
            gated_linear_output_cache_gpu_,
            up_proj_output_cache_gpu_,
            d_grad_input,
            grads_gpu_->gate_proj_grad,
            grads_gpu_->up_proj_grad,
            grads_gpu_->down_proj_grad
        );

        // Copy gradients back to CPU
        grads_.gate_proj_grad = grads_gpu_->gate_proj_grad.to_matrix();
        grads_.up_proj_grad = grads_gpu_->up_proj_grad.to_matrix();
        grads_.down_proj_grad = grads_gpu_->down_proj_grad.to_matrix();
        
        return d_grad_input.to_matrix();
#else
        // CPU implementation
        // Get batch size for gradient normalization (critical for stable training!)
        const float batch_size = static_cast<float>(grad_output.rows());
        const float grad_scale = 1.0f / batch_size;
        
        // Step 1: Gradient w.r.t. the down projection (normalized by batch size)
        grads_.down_proj_grad = matmul(intermediate_cache_.transpose(), grad_output);
        grads_.down_proj_grad *= grad_scale;  // Average over batch, not sum!
        grads_.down_proj_bias_grad = grad_output.column_sum();
        for (size_t i = 0; i < grads_.down_proj_bias_grad.size(); ++i) {
            grads_.down_proj_bias_grad[i] *= grad_scale;
        }

        // Step 2: Backpropagate through the down projection
        Matrix d_gated = matmul(grad_output, params_.down_proj_weights.transpose());

        // Step 3: Backpropagate through the element-wise multiplication
        // d_up = d_gated * swish(gate)
        // d_gate_swish = d_gated * up
        Matrix d_up = d_gated;
        Matrix gate_swish = gate_cache_;
        gate_swish.apply_swish(); // swish(gate)

        // MSVC: loop vars must be signed int
        #pragma omp parallel for collapse(2)
        for(int i=0; i<static_cast<int>(d_up.rows()); ++i){
            for(int j=0; j<static_cast<int>(d_up.cols()); ++j){
                d_up(i,j) *= gate_swish(i,j);
            }
        }

        Matrix d_gate_swish = d_gated;
        // MSVC: loop vars must be signed int
        #pragma omp parallel for collapse(2)
        for(int i=0; i<static_cast<int>(d_gate_swish.rows()); ++i){
            for(int j=0; j<static_cast<int>(d_gate_swish.cols()); ++j){
                d_gate_swish(i,j) *= up_cache_(i,j);
            }
        }

        // Step 4: Backpropagate through the Swish activation (MSVC: loop vars must be signed int)
        Matrix d_gate = d_gate_swish;
        #pragma omp parallel for collapse(2)
        for(int i=0; i<static_cast<int>(d_gate.rows()); ++i){
            for(int j=0; j<static_cast<int>(d_gate.cols()); ++j){
                d_gate(i,j) *= swish_derivative(gate_cache_(i,j));
            }
        }

        // Step 5: Gradients for up and gate projections (normalized by batch size)
        grads_.up_proj_grad = matmul(input_cache_.transpose(), d_up);
        grads_.up_proj_grad *= grad_scale;  // Average over batch, not sum!
        grads_.up_proj_bias_grad = d_up.column_sum();
        for (size_t i = 0; i < grads_.up_proj_bias_grad.size(); ++i) {
            grads_.up_proj_bias_grad[i] *= grad_scale;
        }

        grads_.gate_proj_grad = matmul(input_cache_.transpose(), d_gate);
        grads_.gate_proj_grad *= grad_scale;  // Average over batch, not sum!
        grads_.gate_proj_bias_grad = d_gate.column_sum();
        for (size_t i = 0; i < grads_.gate_proj_bias_grad.size(); ++i) {
            grads_.gate_proj_bias_grad[i] *= grad_scale;
        }

        // Step 6: Gradient with respect to the input
        Matrix d_input = matmul(d_gate, params_.gate_proj_weights.transpose());
        d_input += matmul(d_up, params_.up_proj_weights.transpose());

        return d_input;
#endif

    } catch (const std::exception& e) {
        std::cerr << "\nError in FeedForward::backward: " << e.what() << std::endl;
        throw;
    }
}

void FeedForward::update_parameters(float learning_rate) {
    SCOPE_LOG();
    
    // Adam (matches the LM head's optimizer; see attention.cpp note)
    const float beta = transformer_runtime::adam_beta1;
    const float beta2 = transformer_runtime::adam_beta2;
    const float adam_eps = 1e-8f;
    const float clip_threshold = 1.0f;

    // Initialize optimizer state on first call
    if (!momentum_.initialized) {
        momentum_.gate_proj_m = Matrix(params_.gate_proj_weights.rows(), params_.gate_proj_weights.cols(), 0.0f);
        momentum_.up_proj_m = Matrix(params_.up_proj_weights.rows(), params_.up_proj_weights.cols(), 0.0f);
        momentum_.down_proj_m = Matrix(params_.down_proj_weights.rows(), params_.down_proj_weights.cols(), 0.0f);
        momentum_.gate_proj_v = Matrix(params_.gate_proj_weights.rows(), params_.gate_proj_weights.cols(), 0.0f);
        momentum_.up_proj_v = Matrix(params_.up_proj_weights.rows(), params_.up_proj_weights.cols(), 0.0f);
        momentum_.down_proj_v = Matrix(params_.down_proj_weights.rows(), params_.down_proj_weights.cols(), 0.0f);
        momentum_.gate_proj_bias_m = FloatVector(params_.gate_proj_bias.size(), 0.0f);
        momentum_.up_proj_bias_m = FloatVector(params_.up_proj_bias.size(), 0.0f);
        momentum_.down_proj_bias_m = FloatVector(params_.down_proj_bias.size(), 0.0f);
        momentum_.gate_proj_bias_v = FloatVector(params_.gate_proj_bias.size(), 0.0f);
        momentum_.up_proj_bias_v = FloatVector(params_.up_proj_bias.size(), 0.0f);
        momentum_.down_proj_bias_v = FloatVector(params_.down_proj_bias.size(), 0.0f);
        momentum_.initialized = true;
    }
    momentum_.t += 1;
    const float bc1 = 1.0f - std::pow(beta, static_cast<float>(momentum_.t));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(momentum_.t));

    // Helper: Adam update with per-tensor clipping for matrices
    auto momentum_update = [&](Matrix& param, const Matrix& grad, Matrix& m, Matrix& v) {
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
    GRAD_LOG_MATRIX("ffn_gate_proj_grad", grads_.gate_proj_grad);
    GRAD_LOG_MATRIX("ffn_down_proj_grad", grads_.down_proj_grad);
    
    // Update projections with momentum (biases frozen at zero in LLaMA mode -
    // the llama FFN has no biases). Under LoRA fine-tuning the base weights
    // freeze and the gradient is projected onto rank-r adapters (lora.hpp).
    if (lora::settings().enabled) {
        if (!lora_gate_.attached()) {
            const auto& s = lora::settings();
            lora_gate_.attach(params_.gate_proj_weights, s.rank, s.alpha, 201);
            lora_up_.attach(params_.up_proj_weights, s.rank, s.alpha, 202);
            lora_down_.attach(params_.down_proj_weights, s.rank, s.alpha, 203);
        }
        lora_gate_.update(params_.gate_proj_weights, grads_.gate_proj_grad, learning_rate);
        lora_up_.update(params_.up_proj_weights, grads_.up_proj_grad, learning_rate);
        lora_down_.update(params_.down_proj_weights, grads_.down_proj_grad, learning_rate);
    } else {
        momentum_update(params_.gate_proj_weights, grads_.gate_proj_grad, momentum_.gate_proj_m, momentum_.gate_proj_v);
        momentum_update(params_.up_proj_weights, grads_.up_proj_grad, momentum_.up_proj_m, momentum_.up_proj_v);
        momentum_update(params_.down_proj_weights, grads_.down_proj_grad, momentum_.down_proj_m, momentum_.down_proj_v);
    }
    if (!transformer_runtime::llama_no_bias) {
        momentum_update_bias(params_.gate_proj_bias, grads_.gate_proj_bias_grad, momentum_.gate_proj_bias_m, momentum_.gate_proj_bias_v);
        momentum_update_bias(params_.up_proj_bias, grads_.up_proj_bias_grad, momentum_.up_proj_bias_m, momentum_.up_proj_bias_v);
        momentum_update_bias(params_.down_proj_bias, grads_.down_proj_bias_grad, momentum_.down_proj_bias_m, momentum_.down_proj_bias_v);
    }
    
    // Log momentum state
    GRAD_LOG_MATRIX("ffn_down_proj_momentum", momentum_.down_proj_m);
    
    // Reset gradients
    grads_.gate_proj_grad.initialize_constant(0.0f);
    grads_.gate_proj_bias_grad.initialize_constant(0.0f);
    grads_.up_proj_grad.initialize_constant(0.0f);
    grads_.up_proj_bias_grad.initialize_constant(0.0f);
    grads_.down_proj_grad.initialize_constant(0.0f);
    grads_.down_proj_bias_grad.initialize_constant(0.0f);

#ifdef USE_CUDA
    // ROOT CAUSE OF THE 2026-07 "mushy GPU models" saga: forward() caches
    // the weights on the GPU once (if (!params_gpu_)) and never re-uploads,
    // so every forward after step 1 used the FROZEN INITIAL FFN while these
    // host weights drifted under Adam updates whose effects never reached
    // the loss. Invalidate the device copy after every host update so the
    // next forward re-uploads the weights that were actually trained.
    params_gpu_.reset();
#endif
}

void FeedForward::initialize_weights() {
    // Get dimensions
    size_t hidden_size = params_.gate_proj_weights.rows();
    size_t intermediate_size = params_.gate_proj_weights.cols();

    // Initialize with Xavier/Glorot initialization
    float gate_limit = std::sqrt(6.0f / (hidden_size + intermediate_size));
    params_.gate_proj_weights.initialize_random(gate_limit);
    
    float up_limit = std::sqrt(6.0f / (hidden_size + intermediate_size));
    params_.up_proj_weights.initialize_random(up_limit);

    float down_limit = std::sqrt(6.0f / (intermediate_size + hidden_size));
    params_.down_proj_weights.initialize_random(down_limit);

    // Initialize biases. LLaMA mode: exactly zero and frozen (the llama FFN
    // has no biases; zero + frozen makes the bias-add a no-op).
    const float bias_init = transformer_runtime::llama_no_bias ? 0.0f : 0.01f;
    params_.gate_proj_bias.initialize_constant(bias_init);
    params_.up_proj_bias.initialize_constant(bias_init);
    params_.down_proj_bias.initialize_constant(bias_init);
}