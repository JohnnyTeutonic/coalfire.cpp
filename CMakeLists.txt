cmake_minimum_required(VERSION 3.18)

# Compatibility for CMake 4.x with older dependencies
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum policy version for dependencies")
endif()

# Option to enable CUDA
option(CUDA_ENABLED "Enable CUDA support" OFF)

if(CUDA_ENABLED)
    # CRITICAL: Set CUDA flags BEFORE project() call for compiler identification
    # VS 2026 is not officially supported by CUDA 13.0, so we need to allow it
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "CUDA compiler flags" FORCE)
    
    # Also set for C++ compiler to allow CUDA headers in .cpp files
    if(MSVC)
        # Force-include compatibility header before anything else
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FI\"${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_compat.h\"" CACHE STRING "C++ compiler flags" FORCE)
        # Add compile definitions to disable CUDA host compiler check
        add_compile_definitions(__NV_NO_HOST_COMPILER_CHECK _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH USE_CUDA CUDA_AVAILABLE)
    endif()
    
    project(transformer_cpp LANGUAGES CXX CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # CUDA architectures. Respect an explicit -DCMAKE_CUDA_ARCHITECTURES=NN
    # (required for cross-GPU builds); otherwise detect the local GPU
    # ("native", CMake >= 3.24) and fall back to a multi-arch set covering
    # the GPUs we train on:
    #   75 = T4 (Colab)   80 = A100 (Colab)   86 = Ampere consumer
    #   89 = L4 (Colab) / RTX 4060 (local)
    # NOTE: never FORCE a single arch here — sm_89 binaries cannot run on
    # T4/A100, which silently broke Colab builds before.
    include(CheckLanguage)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")
        endif()
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    message(STATUS "CUDA enabled - version ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    project(transformer_cpp LANGUAGES CXX)
    message(STATUS "CUDA disabled - CPU-only build")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent
include(FetchContent)

# Find Threads
find_package(Threads REQUIRED)

# Find OpenSSL (optional for minimal build)
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
endif()

# Fetch nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Core source files (COMPLETE BUILD)
set(CORE_SOURCES
    # Core infrastructure
    src/config.cpp
    src/architecture.cpp
    src/lora.cpp
    src/logger.cpp
    src/scope_logger.cpp
    src/components.cpp  # Contains ACTUAL Matrix implementations!
    src/matrix.cpp
    src/matmul_optimized.cpp  # Optimized matmul (12-18x speedup)
    src/tensor.cpp
    src/utils.cpp
    src/vector_ops.cpp
    
    # Model components
    src/embeddings.cpp
    src/layer_norm.cpp
    src/feed_forward.cpp
    src/attention.cpp
    src/transformer_layer.cpp
    src/transformer.cpp
    src/lm_head.cpp
    src/router.cpp
    src/moe.cpp
    src/gqa.cpp
    
    # Training components
    src/vocabulary.cpp
    src/tiktoken_tokenizer.cpp
    src/gradient_checkpoint.cpp
    # src/gradient_accumulator.cpp  # Excluded: missing logger/half_precision dependencies
    src/cache.cpp
    
    # Quantization
    src/quantization.cpp
    src/quantization_api.cpp
    
    # Training infrastructure (EXCLUDED: PyTorch dependencies)
    # src/training/gradient_manager.cpp
    # src/training/learning_rate_scheduler.cpp
    # src/training/loss_tracker.cpp
    # src/training/training_monitor.cpp
    # src/training/training_state_manager.cpp
    
    # Utilities
    src/model_saver.cpp
    src/gguf_export.cpp
    src/safetensors_export.cpp
    # src/performance_metrics.cpp
    # src/debugging_monitoring.cpp
)

# Add CUDA sources if enabled
if(CUDA_ENABLED)
    list(APPEND CORE_SOURCES
        src/cuda/cuda_matrix.cu
        src/cuda/swiglu_kernels.cu
        src/cuda/quantization_kernels.cu
        src/cuda/matrix_ops.cu
        src/cuda/fused_attention_kernels.cu
        src/cuda/moe_kernel.cu
        src/cuda/router_kernel.cu
        src/cuda/layernorm_cuda.cu
        src/cuda/loss_kernels.cu
        src/cuda/attention_ops.cu
    )
    # Set CUDA language for .cu files only (not .cpp files)
    # lm_head.cpp removed - it has CPU fallback and doesn't need NVCC
    set_source_files_properties(
        src/cuda/cuda_matrix.cu
        src/cuda/swiglu_kernels.cu
        src/cuda/quantization_kernels.cu
        src/cuda/matrix_ops.cu
        src/cuda/fused_attention_kernels.cu
        src/cuda/moe_kernel.cu
        src/cuda/router_kernel.cu
        src/cuda/layernorm_cuda.cu
        src/cuda/loss_kernels.cu
        src/cuda/attention_ops.cu
        PROPERTIES LANGUAGE CUDA
    )
endif()

add_library(transformer_core STATIC ${CORE_SOURCES})

target_include_directories(transformer_core 
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(transformer_core 
    PUBLIC
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Enable OpenMP for optimized matmul (multi-threading)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(transformer_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for optimized matmul")
else()
    message(WARNING "OpenMP not found - matmul will use single thread (still faster)")
endif()

# Enable CPU optimizations (AVX2/SIMD) - CXX only, not CUDA
if(MSVC)
    # MSVC flags for Windows - only apply to C++ files, not CUDA
    target_compile_options(transformer_core PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/O2>              # Maximum optimization
        $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>       # Enable AVX2 SIMD
        $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>         # Fast floating-point math
    )
    message(STATUS "MSVC optimizations enabled: /O2 /arch:AVX2 /fp:fast (CXX only)")
else()
    # GCC/Clang flags for Linux/macOS
    target_compile_options(transformer_core PRIVATE 
        -O3              # Maximum optimization
        -march=native    # Use all available CPU instructions
        -ffast-math      # Fast floating-point math
    )
    message(STATUS "GCC/Clang optimizations enabled: -O3 -march=native")
endif()

# Link CUDA libraries if enabled
if(CUDA_ENABLED)
    target_link_libraries(transformer_core PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_definitions(transformer_core PUBLIC 
        USE_CUDA
        CUDA_AVAILABLE  # Legacy macro used in some parts of the code
        __NV_NO_HOST_COMPILER_CHECK  # Disable CUDA host compiler version check
        _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH  # Allow VS 2026 with CUDA 13
    )
    # Only define __CUDACC__ for CUDA files, not C++ files
    target_compile_definitions(transformer_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDACC__>
    )
    
    # CUDA 12.6 bug workaround: suppress constexpr redeclaration errors
    # This is NVIDIA bug #4229173 - safe to suppress as math functions work correctly
    # Use compiler-specific flags: GCC/Clang vs MSVC
    if(MSVC)
        target_compile_options(transformer_core PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/W3>  # Warning level 3 for MSVC
            $<$<COMPILE_LANGUAGE:CXX>:/D__NV_NO_HOST_COMPILER_CHECK>
            $<$<COMPILE_LANGUAGE:CXX>:/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>  # Pass to host compiler
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/D__NV_NO_HOST_COMPILER_CHECK>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>
        )
    else()
        target_compile_options(transformer_core PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-error -fpermissive>
        )
    endif()
endif()

# Main executable commented out - depends on PyTorch
# (transformer_minimal / src/main.cpp deleted 2026-07-31: dead code that
# referenced a config shape which no longer exists; `train` is the trainer.)

# Simple training test executable (CPU-only, production ready)
add_executable(train_simple src/train_simple.cpp)
target_link_libraries(train_simple PRIVATE transformer_core)
target_include_directories(train_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# THE central trainer (TinyStories, WikiText, chat/instruct — word-level
# corpora generally; renamed from train_wikitext 2026-07-31 to reflect that).
# SWA deterministic receipt: perturbation probes proving the sliding
# window binds in forward_batched and combines with causality.
add_executable(swa_check src/swa_check.cpp)
target_link_libraries(swa_check PRIVATE transformer_core)
target_include_directories(swa_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(train src/train.cpp)
target_link_libraries(train PRIVATE transformer_core)
target_include_directories(train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(golden_batch_test src/golden_batch_test.cpp)
target_link_libraries(golden_batch_test PRIVATE transformer_core)
target_include_directories(golden_batch_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(matmul_selftest src/matmul_selftest.cpp)
target_link_libraries(matmul_selftest PRIVATE transformer_core)
target_include_directories(matmul_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Gate-2 brick: order-invariant (reproducible) summation. Standalone, no deps.
add_executable(repro_sum_demo src/repro_sum_demo.cpp)

# Proof-of-learning challenge-response: honest-vs-lying-node trust demo. No deps.
add_executable(pol_demo src/pol_demo.cpp)

# Gate-2 brick: thread-count-invariant reproducible reduction.
add_executable(repro_reduce_demo src/repro_reduce_demo.cpp)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(repro_reduce_demo PRIVATE OpenMP::OpenMP_CXX)
endif()

message(STATUS "")
message(STATUS "================================")
if(CUDA_ENABLED)
    message(STATUS "CUDA-Enabled Build:")
    message(STATUS "  - Core transformer with GPU acceleration")
    message(STATUS "  - Per-channel quantization kernels")
    message(STATUS "  - CUDA ${CUDAToolkit_VERSION}")
else()
    message(STATUS "CPU-Only Build:")
    message(STATUS "  - Core transformer (no GPU)")
endif()
message(STATUS "================================")
message(STATUS "")

