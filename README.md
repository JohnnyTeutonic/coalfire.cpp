# coalfire.cpp — Transformer Training in C++/CUDA

*Formerly `transformer` — renamed 2026-08-01 (GitHub redirects the old
URL). coalfire.cpp is the original trainer engine of the paperkiln
family, alongside [paperkiln](https://github.com/JohnnyTeutonic/paperkiln)
(the training studio, formerly microtorch) and
[ember.cpp](https://github.com/JohnnyTeutonic/ember.cpp) (the inference
half, formerly tinyllama.cpp).*

A pure C++ implementation of transformer training with CUDA acceleration.

> ## RESOLVED (2026-07-19) — CUDA FeedForward used a frozen stale weight cache
>
> `FeedForward::forward` (CUDA path) uploaded its weights to the GPU **once,
> at the first forward, and never re-uploaded** (`if (!params_gpu_)`), while
> Adam updates wrote the host copies. Consequence: on CUDA builds the FFN
> never trained — models learned around a frozen random FFN — and the host
> FFN weights drifted under gradients whose effects never reached the loss.
> Checkpoints/exports faithfully saved that drifted, never-executed FFN,
> which explodes (~245x activation blowup) under any honest forward
> (CPU build, NumPy, tinyllama.cpp). This is why every GPU-trained model
> generated degenerate text while CPU-trained toys were always coherent.
> FIXED by invalidating `params_gpu_` after every host update
> (`FeedForward::update_parameters`).
>
> Remaining hygiene from the same investigation: the fused attention kernel
> (`fused_attention_kernels.cu`) implements NO RoPE and must never carry a
> RoPE-family model (it is reachable only from single-sequence `forward()`
> without kv-cache, and contaminated eval/probe measurements);
> `MultiHeadAttention::load` does not restore `use_fused_attention`; the MoE
> router (`router.cpp`) has the same lazy-cache pattern (unused path).
> Checkpoint format 3 serializes LayerNorm eps/rms (format-2 restores
> silently swapped RMSNorm for LayerNorm).

## What Works (Verified)

- **End-to-end pipeline (from-scratch, no external ML frameworks)** — models trained here, exported to GGUF/safetensors, and served by a separate from-scratch inference engine ([tinyllama.cpp](https://github.com/JohnnyTeutonic/tinyllama.cpp)) for working chat generation. The training and inference code are independent implementations, so successful interop verifies both the trainer's export and the loader's parsing. Argmax parity between the two stacks was verified 2026-07-13 (4/4 fixed prompts).
- **Dialogue chat milestone (2026-07-20)** — a 4-layer/256-hidden model (`--mid` preset) trained on synthetic dialogue holds question→answer semantics and serves interactive chat through the engine's web UI, including multi-turn history. The experimental record — what worked, what failed, and why — lives in [CHAT_EXPERIMENTS.md](CHAT_EXPERIMENTS.md).
- **Core transformer training** — TinyStories / TinyStories-Instruct / dialogue corpora on Colab T4s (WikiText-2 for the original smoke tests); resumable checkpoints (format 3: full-model serialization, payload hashes, atomic writes, restore-fidelity probe at resume).
- **CUDA kernels** — fused softmax+cross-entropy loss, batched training path; fused attention, SwiGLU, MoE routing (see hygiene notes above for the fused-attention caveat)
- **Standard optimizations** — FP16 mixed precision, gradient accumulation, RoPE, GQA

## What Exists But Is Not Operational

The codebase contains extensive distributed training infrastructure that has been **implemented but not tested in production**:

- P2P networking with PBFT consensus
- Kademlia DHT peer discovery
- Byzantine fault detection
- RLHF/PPO training
- Web annotation interface
- Distributed checkpointing

**These features are aspirational.** The distributed code compiles but has not been validated at scale.

## Known Data Pitfall (found 2026-07-21)

The word-level tokenizer caps its vocabulary at 20k while the `--mid`/`--tiny`
model presets cap the embedding table at 5k. The CUDA embedding gather was
unchecked, so any token id past the model cap silently read out-of-bounds GPU
memory as its embedding. Fixed twice over: the trainer now remaps out-of-range
ids to UNK (reporting the OOV rate as a `[VOCAB]` log line) and
`TokenEmbedding::forward_cuda` bounds-checks before the gather. Models trained
on open-vocabulary corpora before this date carried corrupted tail-word
embeddings; see CHAT_EXPERIMENTS.md Finding 5 before comparing against them.

## Installation

### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CUDA 11.0+ with compatible GPU (optional but recommended)
- CMake 3.16+

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Windows

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

## Usage

The main training entry point is `train_wikitext` (name is historical — it
drives all corpora):

```bash
./train_wikitext --mid --seq-len 256 --doc-aligned \
    --steps 12000 --export-gguf model.gguf
```

Key flags:

- `--mid` / `--tiny` — model preset (4L/256h chat-capable tier vs 2L/128h
  round-trip smoke tier)
- `--seq-len N` — context length
- `--doc-aligned` — bin-pack whole documents instead of prose windowing.
  Required for dialogue/instruct data: the windowing loader starts 97% of
  windows mid-document and destroys prompt→response conditioning
  (CHAT_EXPERIMENTS.md Finding 3)
- `--assistant-loss` — mask the loss on user-turn tokens and role markers so
  every update optimizes P(answer | question) directly; prompts still shape
  representations through the answer loss
- `--export-gguf / --export-safetensors PATH` — engine-ready exports
- `--resume PATH` — resume from a format-3 checkpoint; the restored model
  must reproduce its recorded loss (batched-forward probe) or the run aborts
  (`TCPP_FORCE_RESUME=1` overrides)

Dialogue/instruct corpora are prepared with `scripts/prepare_chat_data.py`
(TinyStories-Instruct re-aggregation, DailyDialog, synthetic TinyChat);
`scripts/eval_tinychat_generalisation.py` scores served models on train-form
vs unseen-paraphrase probes with an intent confusion matrix.

## Transformer Features

### Attention
- Multi-Head Attention with GQA support
- Rotary Position Embeddings (RoPE)
- Flash Attention memory optimization
- Key-Value Cache

### Architecture
- Layer Normalization
- Feed Forward Networks with SwiGLU
- Mixture of Experts (MoE)
- Dropout and residual connections

### Training
- FP16 mixed precision
- Gradient accumulation and clipping
- Batch processing
- Checkpoint save/load

### CUDA Kernels
- Fused attention kernel
- SwiGLU activation kernel
- MoE router kernel

## Limitations

- Small models (0.4M–6M core parameters); coherence requires capacity-matched
  data (CHAT_EXPERIMENTS.md Finding 2) — open-domain corpora produce register
  without semantics at this scale
- Distributed features are not operational
- Curated GGUF exports are kept out of the repo (see `.gitignore`); training
  outputs land under `colab_out_*/`

## License

MIT License
