"""VM-side driver: build transformer_cpp and train on TinyStories on a Colab GPU.

Expects /content/transformer_cpp_src.zip (made by scripts/make_colab_src_zip.py,
uploaded via the colab CLI). Then:
  1. unzip source -> /content/tcpp
  2. cmake -DCUDA_ENABLED=ON (CMAKE_CUDA_ARCHITECTURES=native detects T4/A100/L4)
  3. prepare TinyStories text files (datasets lib)
  4. run train with the requested steps/export format

Usage on the VM (via `colab exec -f` or a notebook cell):
    python colab_train_tinystories.py [--steps 5000] [--export both]

Artifacts land in /content/tcpp/run/: checkpoints, model.gguf and/or
model.safetensors (+ .vocab.json), training logs.
"""
import argparse
import os
import subprocess
import sys


def sh(cmd, **kw):
    print(f"+ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", **kw)
    if r.returncode != 0:
        print(f"FAILED (rc={r.returncode}): {cmd}", flush=True)
        sys.exit(r.returncode)


ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--export", choices=["gguf", "safetensors", "both"], default="gguf")
ap.add_argument("--lr", type=float, default=None,
                help="learning rate override (trainer default 2e-4 is Adam-scale; "
                     "the SGD-momentum+clip update wants ~10-50x more)")
ap.add_argument("--cosine-decay", action="store_true",
                help="cosine LR decay to 10%% over max_steps (constant-LR runs "
                     "destabilized late: loss 3.4->5.3 in epoch 2, 2026-07-13)")
ap.add_argument("--stories-per-file", type=int, default=300_000)
ap.add_argument("--tiny", action="store_true",
                help="2L/128h config: steps fast under the current per-op "
                     "copy architecture (see PERF_NOTES.md); the 512 config "
                     "awaits the resident-tensor refactor")
ap.add_argument("--mid", action="store_true",
                help="4L/256h config (~8x tiny core): chat-capable, T4-trainable")
ap.add_argument("--seq-len", type=int, default=None,
                help="context length in tokens (default 128; 256 for chat)")
ap.add_argument("--doc-aligned", action="store_true",
                help="bin-pack whole documents into windows (dialogue/instruct)")
ap.add_argument("--assistant-loss", action="store_true",
                help="mask loss on user-turn tokens and role markers")
ap.add_argument("--dataset", choices=["plain", "instruct", "dailydialog",
                                      "empathetic", "dialogmix", "tinychat",
                                      "tinychatmix", "dolly"],
                default="plain",
                help="plain=TinyStories; others via prepare_chat_data.py "
                     "(chat-formatted, user:/assistant: turns)")
a = ap.parse_args()

ROOT = "/content/tcpp"
os.makedirs(ROOT, exist_ok=True)
sh(f"cd /content && unzip -oq transformer_cpp_src.zip -d {ROOT}")

# Build (native arch: nvcc detects the attached GPU — T4=75, A100=80, L4=89)
sh(f"cd {ROOT} && cmake -B build -DCUDA_ENABLED=ON -DCMAKE_BUILD_TYPE=Release "
   f"-DCMAKE_CUDA_ARCHITECTURES=native")
sh(f"cd {ROOT} && cmake --build build --target train --parallel $(nproc)")

# Data
sh("pip install -q datasets")
if a.dataset == "plain":
    data_dir = "data/tinystories-txt"
    sh(f"cd {ROOT} && python scripts/prepare_tinystories.py "
       f"--out {data_dir} --stories-per-file {a.stories_per_file}")
else:
    hf_name = {"instruct": "tinystories-instruct", "dailydialog": "dailydialog",
               "empathetic": "empathetic", "dialogmix": "dialogmix",
               "tinychat": "tinychat", "tinychatmix": "tinychatmix",
               "dolly": "dolly"}[a.dataset]
    data_dir = f"data/{hf_name}-txt"
    sh(f"cd {ROOT} && python scripts/prepare_chat_data.py "
       f"--dataset {hf_name} --out {data_dir}")

# Train (run/ is the working dir so checkpoints and logs collect there)
run_dir = f"{ROOT}/run"
os.makedirs(run_dir, exist_ok=True)

# Reclaim robustness: continuously mirror the newest on-VM checkpoint to a
# fixed path the supervisor can relay out; if the supervisor pushed a relayed
# checkpoint back (fresh VM after a reclaim), resume from it.
# Mirror atomically (cp to tmp + mv): a plain cp of the live file can relay a
# partial checkpoint if it races the trainer's save or the supervisor's pull.
sh("nohup bash -c 'while true; do "
   f"latest=$(ls -t {run_dir}/checkpoints/*.ckpt 2>/dev/null | head -1); "
   "[ -n \"$latest\" ] && cp -f \"$latest\" /content/.latest_tcpp.tmp "
   "&& mv -f /content/.latest_tcpp.tmp /content/latest_tcpp.ckpt; "
   "sleep 120; done' >/dev/null 2>&1 &")
resume = ""
if os.path.exists("/content/restore_tcpp.ckpt"):
    resume = "--resume /content/restore_tcpp.ckpt "
    print("[driver] resuming from relayed checkpoint", flush=True)

exports = []
if a.export in ("gguf", "both"):
    exports.append("--export-gguf model.gguf")
if a.export in ("safetensors", "both"):
    exports.append("--export-safetensors model.safetensors")
sh(f"cd {run_dir} && ../build/train ../{data_dir} "
   f"--epochs {a.epochs} --max-steps {a.steps} --batch-size {a.batch_size} "
   + (f"--lr {a.lr} " if a.lr else "")
   + ("--cosine-decay " if a.cosine_decay else "")
   + ("--tiny " if a.tiny else "")
   + ("--mid " if a.mid else "")
   + (f"--seq-len {a.seq_len} " if a.seq_len else "")
   + ("--doc-aligned " if a.doc_aligned else "")
   + ("--assistant-loss " if a.assistant_loss else "")
   + resume + " ".join(exports))

print("COLAB_TRAIN_COMPLETE", flush=True)
sh(f"ls -la {run_dir}")
