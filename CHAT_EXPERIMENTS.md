# Chat Experiments — findings ledger (v5–v8, 2026-07-19 → )

Companion to EXPORT_NOTES.md (export/engine bugs) and PERF_NOTES.md
(throughput). This file records the *scientific* findings from the chat
model campaign — several are publishable-shape observations in their own
right, and two feed directly into `AI_ML/choquet_transformer` (the
repetition-hunger of binding) and future scaling work.

## Model ledger

| model | config | data | key number | verdict |
|---|---|---|---|---|
| chat5 (v5) | 2L/128h/seq128 | TinyStories-Instruct (fragmented) | loss 3.04 | first coherent GPU model; stories, weak conditioning |
| chat6 (v6) | 2L/128h/seq128 | DailyDialog | loss 3.61 | dialogue register, wandering content |
| chat7a | 4L/256h/seq256 | DailyDialog+Empathetic (prose-windowed) | loss ~4.1 | topic-following multi-turn; QA semantics absent |
| chat7b | 4L/256h/seq256 | Instruct (aggregation-fixed) | loss 3.25 | fluent narrative arcs; zero conditioning (~0.2 epochs) |
| micro-bind | 2L/128h, CPU | 1.5k TinyChat pairs, 60 epochs | ppl 1.4 | PERFECT 6/6 dialogue-act binding (greedy) |
| chat8b @1.3ep | 4L/256h/seq256 | 150k TinyChat (doc-aligned) | loss 0.35 | partial binding 4/6; extended to ~4 epochs |
| chat8b @3ep | 4L/256h/seq256 | 150k TinyChat x3 (doc-aligned) | loss 0.35; gen eval below | intent-wise binding; bound intents GENERALISE |
| chat8a @18k | 4L/256h/seq256 | Instruct aggregated, ~1.5 epochs | loss ~2.8-3.0 | fluent; ZERO conditioning (dragon test failed) |
| chat8c | 4L/256h/seq256 | tinychatmix 8000 steps | loss 4.28 | mix dilutes binding — CONFOUNDED by F5 |

## chat8a dragon test (2026-07-21, greedy, raw prompt) — repetition test CLOSED

18,000 steps (~1.5 epochs over 2.28M aggregated instruct examples):
"words: dragon , dark , brave story:" and summary-conditioning probes all
produce fluent stories that ignore the fields; greedy collapses to one
corpus-head opening ("once upon a time, there was a little girl named
lily..."). vs chat7b at 0.2 epochs: same zero conditioning, better
fluency. CONCLUSION (with Finding 1): epochs alone cannot rescue
conditioning when the pattern inventory itself is combinatorial — the
words: field draws from thousands of combinations, so 1.5 or even 10
epochs of the corpus is <<1 exposure per pattern. Binding budgets must
count exposures per semantic equivalence class (Jonathan's framing),
and the instruct corpus has ~none. TinyChat's inventory is small enough
to repeat; that is WHY 8b binds. (OOV caveat F5 applies to 8a's tail
words but probe words are high-frequency; the conclusion stands.)

## chat8b generalisation eval (2026-07-21, greedy, raw prompt, engine rebuilt)

Result-matrix row (eval_tinychat_generalisation.py, 20 probes):
- Tier A (train-form): answer-type 6/11, semantic-slot 6/11
- Tier B (unseen paraphrase): answer-type 4/9, semantic-slot 3/9
- Sampled@0.7 pass: A 6/11, B 4/9 — no greedy/sampled gap.
- Control floor (chat5 story model): slot 0/20.

Reading: binding at ~3 epochs is INTENT-WISE, and it transfers: tier B
tracks tier A per intent (wellbeing/food/day_activity generalise to unseen
surface forms — semantic abstraction, not template memorisation), while
drink_offer/hobby/pet fail in BOTH tiers. What is learned is learned as a
relation; what is missing is missing everywhere. Refines Finding 1: more
epochs should convert intents wholesale, not polish surfaces.

ANOMALY (open, feeds golden-batch suite): failures are mostly intent-
correct but function-word-dropped ("no thank milk ."). [TOPK] logging
(new TINYLLAMA_TOPK_DEBUG=1) shows the dropped word losing by 0.11 logits
(5.27 vs 5.15) at a corpus-deterministic position, while neighbouring
positions win by 12+. Ruled out: repetition penalty (identical at 1.0),
prompt wrapping (word-level path is raw), stale binary (rebuilt — NB the
CLI binary had been 9 days / 3 fix-generations stale and silently killed
F32 models via the pre-fix layer-clearing path; quantized models masked
it). Remaining suspects: genuine underlearning of those intents vs
trainer/engine logit divergence at specific positions — exactly what the
golden-batch GGUF-parity test decides. Engine got --rep-penalty,
--raw-prompt, and [TOPK] debug flags out of the investigation.

## Finding 1 — Binding is repetition-hungry and is learned last

Register, fluency, and corpus CE saturate long before conditional binding
(question-type -> answer-type; words: -> story content) emerges. Evidence,
three independent setups:
- chat7b: 937k-seq corpus, each conditioning pattern seen <=1x -> fluent,
  zero conditioning.
- micro-bind vs chat8b: identical data distribution; 60 epochs -> 6/6
  perfect binding; 1.3 epochs -> 4/6 with class confusions.
- choq2 cell 1 (33.6M, different task family): CE saturates by step ~1500,
  probe accuracy leaves chance only after ~3000.
Practical rule: for binding-type capabilities at small scale, budget
EPOCHS OVER THE PATTERN INVENTORY, not tokens. A dataset too large to
repeat is a dataset too large to bind.

## Finding 2 — Capacity-matched data unlocks coherence (TinyStories thesis
## transfers to dialogue)

A 13M model on open-domain dialogue (DailyDialog, 20k-cap vocab) learns
the *manifold* (turn-taking, register, reply shapes) but not the
*conditional map* — greedy decoding collapses to one generic reply across
question classes. The same architecture on a 145-word synthetic dialogue
world (TinyChat, gen in scripts/prepare_chat_data.py) binds
question->answer semantics. Coherence at small scale is a data-bandwidth
match problem before it is a parameter-count problem. Escalation path:
grow the world's vocabulary with capacity (curriculum), not ahead of it.

## Finding 3 — Prose windowing destroys dialogue conditioning signal

The stream-packing loader (built for WikiText/TinyStories) applied to
dialogue: 97% of training windows began MID-dialogue (often mid-sentence),
2.3 dialogues per window with attention crossing eos, 50%-overlap stride
re-exposing every dialogue in differently-truncated form (audit
2026-07-20, DailyDialog). Responses train without their prompts; sparse QA
signal drowns. Fix: --doc-aligned loader (whole-document bin-packing,
turn-boundary splits, no overlap, PAD ignore-index). Pairing itself was
never misaligned (post-eos token = "user:" in 2932/2932 windows) — the
harm was windowing, not shift.

## Finding 5 — Vocab-cap mismatch = silent OOV embedding corruption
## (found 2026-07-21 by the tokenizer audit; affects EVERY lane except pure
## TinyChat — chat5, chat6, chat7a, chat7b, chat8a, chat8c; log-verified
## for 8a/8c: "Total vocabulary size: 20000" + "Vocab size: 5000")

The tokenizer caps its vocabulary at 20k; the --mid/--tiny presets cap the
MODEL at 5k. Nothing reconciled the two, and the CUDA embedding gather
(token_embedding_cuda.cu) is unchecked: every token with id >= 5000 read
out-of-bounds GPU memory as its embedding vector — silent garbage, run
after run. Ids are frequency-ranked, so the corruption hit exactly the
open-domain tail (TinyChat's 148 words all rank low and were untouched;
that is why 8b was clean). CONSEQUENCE FOR FINDINGS: the mix-dilution
result (8c) and the open-domain incoherence attributions (chat6/7a inside
Finding 2) are CONFOUNDED — part of "capacity mismatch" may be literal
input corruption on tail words. The clean TinyChat results stand. Fix
(2026-07-21): trainer remaps OOV ids to UNK with a reported rate
([VOCAB] line), and forward_cuda bounds-checks before the gather. Any
future mixing/breadth experiment must rerun post-fix before comparison.
Moral (same family as Finding 4): an unchecked device gather is another
lying metric — plausible outputs, corrupt inputs.

## Roadmap adopted 2026-07-21 (Jonathan's review, recorded verbatim in spirit)

Sequence: (1) evaluate 8a/8b GENERALISATION, not just the six probes —
unseen paraphrases + held-out slot combinations distinguish abstraction
from memorisation; result matrix per run: {train probes, paraphrases, new
combos}. (2) assistant-only loss (--assistant-loss, implemented
2026-07-21: role-marker state machine masks user-turn/marker targets via
the ignore-index; arms to compare: full CE vs assistant-only vs ~20%
prompt loss [weighted variant TODO — needs kernel scaling]). (3) vocab as
architecture: audit numbers below; TIE embeddings (deferred into the
resident-tensor refactor — one authoritative weight representation, host
tensors as snapshots, version-numbered if dual state is unavoidable).
(4) hidden-state intent probe at the assistant boundary (linear decode of
prompt intent from the pre-first-answer-token vector; decodable-but-wrong
=> head/objective/decoding problem, not-decodable => representation never
formed). (5) TinyChat++ as a controlled world grown one axis at a time
(intents / lexicon / surface / state), with COMBINATION holdouts; track
exposures per semantic equivalence class, not corpus epochs. (6) golden-
batch differential test suite (CPU-vs-CUDA one-step state equality is the
strongest test; save->load->identical loss; GGUF round-trip logits;
resume->identical next update). (7) resident-tensor refactor, then
6L/512h flagship at the SHORTEST sufficient context (pilot seq 128 vs 256
against the empirical dialogue-length distribution; compare at equal
PATTERN EXPOSURES, not tokens). (8) no more compute on DailyDialog
through current capacity.

Vocab audit numbers (mid preset, hidden 256, untied): at V=5000,
embedding+head = 2.56M of 5.73M total = 45% of parameters serving
vocabulary, most rows nearly untrained under Zipf sparsity; at V=148
(TinyChat) = 2.3%. Tying input/output embeddings halves the vocab share
(~22% freed at V=5000) and couples lexical representation to prediction.

## Finding 6 — RETRACTED (2026-07-23): there was no CPU/CUDA divergence;
## the golden-batch harness was reconstructing the model architecture wrong

RESOLUTION: with the correct architecture applied, the trainer CPU forward
is **bit-identical to the inference engine** — golden-batch parity PASSES on
all four probes (max|delta logit| ~1e-4, argmax match, top-5 overlap 5/5).
There is no CPU-vs-CUDA forward bug and no bug in attention.cpp.

Root cause of the false alarm: `TransformerConfig::use_rope` (and
`use_rms_norm`, `use_biases`) have no defaults and are NOT serialized in the
format-3 checkpoint. golden_batch_test's make_config reconstructed the config
without applying the "llama" family preset, so the CPU forward used **additive
sinusoidal positional encoding instead of RoPE** (plus LayerNorm instead of
RMSNorm, plus live biases). At pos 0 the additive PE's cos(0)=1 on odd indices
gave the tell-tale [small, ~1.4, small, ~1.4] input pattern; the model, trained
RoPE-only, then produced garbage. Applying `arch::ArchitectureSpec::
from_family("llama")` in the test restored bit-exact parity.

Two real fixes came out of it (net positive): (1) golden_batch_test now applies
the family preset; (2) the format-3 checkpoint now serializes use_rope/
use_rms_norm/use_biases so this class of bug — a config reconstructed without
the architecture — cannot recur (same class as the format-3 LayerNorm eps/rms
fix). The earlier "FFN fictional-bias" sub-finding was part of the same
misdiagnosis and has been reverted: in llama mode the FFN biases are frozen at
zero, so the CPU bias-add is a no-op, and removing it would have silently
broken vanilla-family models.

LESSON (the durable one, ironically vindicating the suite's epigraph): "a lying
metric mimics every other failure mode" — including a lying TEST. The
golden-batch suite worked exactly as intended; it was the harness's model
reconstruction that lied, and only a differential test against an independent
implementation (the engine) exposed it. Verify the test's own assumptions
before trusting a divergence it reports.

## Finding 6 (original, WRONG — kept for the method record) — CPU-build
## forward diverges from CUDA forward (golden-batch suite, first run, 2026-07-22)

The golden-batch suite (src/golden_batch_test.cpp + TINYLLAMA_LOGITS_DUMP +
scripts/compare_golden_logits.py) was built to decide the 8b near-tie
anomaly and instead caught something bigger on its first run: chat8b's
checkpoint — which passes the CUDA resume probe on the VM at its recorded
loss 0.348 — measures **batched CE 10.03 on the CPU build** of the same
code, and its CPU forward predicts a near-constant token ('water')
regardless of context, while the inference engine serves the same weights
correctly. Same weights, three stacks: CUDA trainer ~0.35, engine coherent,
CPU trainer broken. Divergence is present at sequence position 0 with a
single-token prompt, so RoPE and causal masking are exonerated; the defect
is in the CPU per-position pipeline (embedding/norm/FFN/head path). This is
(a) the two-year-old "major discrepancies between CPU and GPU builds"
Jonathan fought when first writing tinyllama.cpp, now reproducible in one
command, and (b) a candidate root cause for the unresolved XCHECK ~2x-loss
anomaly IF the CUDA build's host-side eval path shares the broken CPU code.
Localization is a [SUBOP]-style differential trace between the trainer CPU
forward and the engine CPU forward on the same input — both stacks run on
this laptop; no GPU needed for the first bisection.

Bisection state (2026-07-22, in progress):
- FFN biases were FICTIONAL parameters: the CUDA SwiGLU kernel computes a
  bias-free FFN in every configuration, while use_biases defaulted on, so
  biases were initialized at 0.01 and Adam-updated along gradients for a
  function never computed; GGUF export has no FFN-bias slots (why the
  engine stayed clean). FIXED: CPU forward is now bias-free to match the
  trained function. Necessary but NOT sufficient — parity still fails with
  nearly identical deltas.
- LOCALIZED to ATTENTION (2026-07-22, TINYLLAMA_LAYER_TRACE vs
  TCPP_LAYER_TRACE on the same single token, pos=0): at position 0
  attention is exactly o_proj(v_proj(x)) — softmax over one position is 1,
  RoPE cannot matter — and the two stacks disagree 6x: trainer-CPU
  attn L2 16.78 vs engine 97.31 (engine is the trusted side; it serves
  this model coherently). Everything else matches: ln1 16.34/16.15,
  ln2 31.78/32.20, and the "FFN blowup" was a red herring (engine ffn
  L2 263 — ~300 is this model's normal magnitude; RMSNorm's scale
  invariance hid the attention error from downstream L2s). Suspect: the
  trainer CPU attention Q/K/V/O matmul convention (x*W vs W*x / layout)
  differs from the CUDA kernel convention the weights were trained in —
  the same disease class as the fictional FFN biases. NEXT: diff
  trainer-CPU attention matvec semantics against the CUDA attention
  kernel's, starting at v_proj/o_proj.
- VOID RESULT, do not reuse: a ckpt-vs-VM GGUF tensor diff run overnight
  showed "39/39 tensors differ" — meaningless, because train_wikitext
  --max-steps 0 means UNLIMITED (the "export-only" run trained ~6 CPU-hours
  through the broken CPU forward before exporting). The proper
  ckpt->GGUF fidelity leg is a golden_batch_test --export-gguf flag that
  calls the exporter directly with no training loop (TODO).

En route the suite also (1) PASSED save->reload->forward bit-identity
(after fixing the test to set_training(false) — dropout was live at
inference, cross-process max|dlogit| 1.68 on identical weights), and
(2) exposed that vocabulary construction has NO sort tie-break, so
equal-frequency words get platform-dependent ids: a checkpoint's vocab is
only reproducible from its own exported GGUF (scripts/extract_gguf_vocab.py
+ golden_batch_test --vocab-list). Do NOT change the sort to fix this while
any lane is mid-run; it reorders every existing model's vocabulary.

## Finding 4 — (engineering, cross-ref) Silent train/deploy divergence

The FeedForward stale-GPU-cache bug (README, resolved 2026-07-19) meant
every pre-fix GPU model trained around a frozen random FFN. Post-fix, loss
improved from a lying ~6.5-plateau to an honest 3.0 and exports became
servable. Moral for the paper-adjacent record: verify train==deploy
function equality with differential activation traces before interpreting
any quality result; "the metric was lying" outranks most modeling
hypotheses in prior probability.

## Finding 5 — tcpp9a cannot survive a Colab reclaim (2026-07-27)

**The v9a lane will never finish as configured.** Observed 08:36 today, after
the supervisor relaunched it following a reclaim:

```
[RESUME PROBE] measured=8 batched=7 checkpoint-recorded=0.6 (1754 positions)
[RESUME PROBE] RESTORED MODEL DOES NOT MATCH ITS CHECKPOINT
  (measured 7.75525 vs recorded 0.5577). Discarding the restore and
  training from scratch instead of training a corrupt model.
--- Epoch 1/40 ---
```

The guard in `src/train_wikitext.cpp:1442-1459` is behaving **correctly**. It
judges on the batched measurement, deliberately, because single-sequence forward
routes through the fused no-RoPE kernel that neither training nor deployment
uses and false-alarms on good restores. Batched read 7 against a recorded 0.5577,
a gap of ~6.4 nats, far past the `+2.0` tolerance. That is not a false alarm:
the checkpoint round-trip is genuinely lossy, and the guard saved a 34-hour run
from training a corrupt model.

**Why this is terminal for the lane.** One epoch is ~51 minutes at ~2.8s/step
over 1033 batches, and the run is 40 epochs, so it needs roughly **34 GPU-hours
without interruption**. Colab T4 sessions do not last that long. Every reclaim
therefore returns to `Epoch 1/40`, and the lane resets forever. It had reached
step 970/1033 before this reset; that work is gone.

**Two cosmetic things that make the log misleading**, worth fixing when the real
bug is:
- the failure message prints `probe_loss` (7.75525) but the DECISION used
  `probe_decision` (the batched value). Reading the message alone suggests the
  single-sequence path was the judge, which is exactly the confusion the comment
  above it was written to prevent.
- default ostream precision renders the summary as `measured=8 batched=7
  checkpoint-recorded=0.6`, so all three look like integers-ish and the real
  gap is hard to read. `std::setprecision` on that line would help.

**Before relaunching this lane, one of:**
1. fix the checkpoint save/load fidelity (this is the real bug, and it is the
   same disease class as Finding 4 and the GGUF alignment-padding bug: a
   serialisation path that silently disagrees with the training path);
2. or reduce the run so one epoch-set fits inside a session and treat each
   session as a completed unit;
3. or set `TCPP_FORCE_RESUME` **only** after proving the restore is faithful.
   Do NOT set it to make the lane "work" — it would train the corrupt model the
   guard just caught.

Until then the lane burns a T4 slot for a result that cannot arrive.
