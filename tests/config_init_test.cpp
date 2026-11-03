// Every scalar field of a constructed TransformerConfig is initialised. Built with
// -ftrivial-auto-var-init=pattern, GCC and Clang fill uninitialised automatic storage with 0xFE
// bytes, so a field that neither the constructor nor a default member initialiser sets still
// holds the pattern after construction. Until 3 Nov 2026, twelve fields (use_fp16, use_rope,
// window_size, ...) did: layer construction read them and checkpoint saving wrote them.
// A new scalar field belongs in the list below.
#include <cstdio>
#include <cstring>

#include "config.hpp"

namespace {

int failures = 0;
int checked = 0;

template <class T>
void expect_set(const char* name, const T& v) {
    ++checked;
    unsigned char b[sizeof(T)];
    std::memcpy(b, &v, sizeof(T));
    bool pattern = true;
    for (unsigned char x : b) pattern = pattern && x == 0xFE;
    if (pattern) {
        std::printf("FAILED: TransformerConfig::%s is uninitialised\n", name);
        ++failures;
    }
}

}  // namespace

#define FIELD(f) expect_set(#f, c.f)

int main() {
    TransformerConfig c(64, 128, 2, 4);
    FIELD(num_layers); FIELD(num_heads); FIELD(hidden_size); FIELD(intermediate_size);
    FIELD(head_dim); FIELD(max_seq_length); FIELD(vocab_size); FIELD(batch_size);
    FIELD(min_batch_size); FIELD(max_batch_size); FIELD(num_epochs); FIELD(initial_lr);
    FIELD(dropout_rate); FIELD(weight_decay); FIELD(early_stopping_patience);
    FIELD(early_stopping_threshold); FIELD(num_folds); FIELD(peak_lr); FIELD(warmup_steps);
    FIELD(decay_factor); FIELD(gradient_clip_threshold); FIELD(layer_norm_epsilon);
    FIELD(gradient_accumulation_steps); FIELD(use_gradient_checkpointing); FIELD(use_fp16);
    FIELD(memory_pool_size); FIELD(use_flash_attention); FIELD(use_rope); FIELD(llama_mode);
    FIELD(use_rms_norm); FIELD(use_biases); FIELD(use_sliding_window); FIELD(window_size);
    FIELD(use_gqa); FIELD(num_kv_heads); FIELD(paths.checkpoint_frequency);
    FIELD(tokenizer.use_subword); FIELD(beam_search.beam_size); FIELD(load_from_checkpoint);
    FIELD(lora_enabled); FIELD(lora_rank); FIELD(lora_alpha); FIELD(use_momentum);
    FIELD(use_adam); FIELD(momentum); FIELD(beta1); FIELD(beta2); FIELD(epsilon);
    FIELD(moe.enabled); FIELD(moe.num_experts); FIELD(moe.top_k); FIELD(moe.aux_loss_coefficient);
    if (failures == 0) std::printf("config_init: all %d checked fields initialised\n", checked);
    return failures == 0 ? 0 : 1;
}
