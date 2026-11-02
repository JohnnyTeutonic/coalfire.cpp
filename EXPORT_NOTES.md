# GGUF export notes (transformer_cpp → tinyllama.cpp)

## 2026-07-13: word-salad root cause and fixes

Symptom: a trained model whose weights demonstrably predicted sensible text
(verified by an independent NumPy forward pass over the safetensors twin)
generated rare-word salad in tinyllama.cpp — always drawn from the same ~40
tail-of-vocab tokens.

Three independent defects:

1. **Data-section alignment (the fatal one, in `gguf_export.cpp`).** The GGUF
   spec starts the tensor-data section at `align_up(header_end,
   general.alignment=32)`; tensor offsets are relative to that aligned start.
   The writer padded *between* tensors but never padded the *start* of the
   data section, so spec-conformant readers (tinyllama.cpp mmap) read every
   tensor shifted by `align_up(header_end,32) − header_end` bytes (11 bytes
   for the 2026-07-12 model). Result: garbage weights → NaN logits → the
   sampler's top-k selected a fixed arbitrary set of tied candidates.
   Fixed in `GGUFWriter::finalize()` (the shared writer
   `include/gguf_writer.hpp` since 3 Nov 2026; `tests/gguf_writer_test.cpp`
   checks the padding); `general.alignment=32` is now written
   explicitly.

2. **Special-token ids (`gguf_export.cpp`).** The word-level vocab layout is
   `<unk>`=0, `|` (document separator)=1, real words from id 2 — but the
   exporter hardcoded bos=2/eos=3/unk=1, branding "the" and "and" as
   BOS/EOS. tinyllama.cpp then dropped every "the"/"and" in decode and
   stopped generation at the first "and". Now: bos=eos=1 (`|`; never
   injected by WORD_LEVEL encode, and stopping at a document boundary is
   correct), unk=pad=0.

3. **Q/A prompt wrapping (tinyllama.cpp `api.cpp`).** `generate()` defaulted
   to wrapping prompts in `"Q: … \nA:"`. For word-level raw LMs those wrapper
   tokens are OOV noise that shift every position. WORD_LEVEL models now take
   the prompt verbatim (chat formatting such as `user: … assistant:` is the
   caller's contract).

Verification: `tinyllama.cpp` argmax (top_k=1) matches an independent NumPy
reimplementation of the trainer's forward pass (pre-norm RMS, adjacent-pair
RoPE, SwiGLU) exactly, 4/4 prompts.

## Repairing GGUFs exported before the fix

`scripts/repair_gguf.py in.gguf out.gguf` — inserts the missing alignment
padding (offsets stay valid: they are relative to the data-section start)
and patches the special-token ids. No tensor bytes change.

## Known interop quirk (harmless for tinyllama.cpp)

The writer emits tensor dims outer-first (e.g. `token_embd = [vocab,
hidden]`), whereas ggml convention is innermost-first. tinyllama.cpp only
uses the element product, so this doesn't matter there, and llama.cpp cannot
load these models anyway (`tokenizer.ggml.model = "word"` is not a llama.cpp
tokenizer). Worth fixing if that ever changes.
