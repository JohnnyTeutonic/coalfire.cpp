#!/usr/bin/env python3
"""Proof-of-learning on the REAL transformer trainer (not the stand-in loop).

Prover builds a chain of N real training-step states -- each a resumable
checkpoint whose metadata carries payload_fnv64 = the state commitment. A
verifier re-executes a randomly chosen step from the committed prior state,
with the committed per-step seed, and checks the commitment reproduces. An
honest prover always verifies; a node that shortcut a step (committed a state
it didn't actually compute) is caught, having done O(challenges) verifier work
rather than re-running the whole chain.

Each transition is
  train --resume S_{k-1} --max-steps 1 --pol-save dir_k
with TCPP_SEED=BASE+k, single-thread -- reproducible by gate-1 determinism
(include/rng_seed.hpp). This is the demonstration in src/pol_demo.cpp, now on
the actual model.
"""
import json, os, random, shutil, struct, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = f"{REPO}/build_wsl/train"
DATA = os.environ.get("POL_DATA", "/tmp/tinychat-txt")
S0 = os.environ.get("POL_S0",
                    "/mnt/c/ml_artifacts/transformer_cpp/colab_out_chat8b/latest_tcpp.ckpt")
WORK = "/tmp/pol_transformer"
BASE, N, CHALLENGES = 1000, 4, 3


def commit(ckpt):
    d = open(ckpt, "rb").read()
    n = struct.unpack_from("<Q", d, 0)[0]
    return json.loads(d[8:8 + n])["payload_fnv64"]


def run_step(prev_ckpt, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, TCPP_SEED=str(seed), TCPP_FORCE_RESUME="1", OMP_NUM_THREADS="1")
    with open(f"{out_dir}/log", "w") as log:
        subprocess.run([TRAINER, DATA, "--mid", "--seq-len", "256", "--doc-aligned",
                        "--resume", prev_ckpt, "--max-steps", "1", "--pol-save", out_dir],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return f"{out_dir}/pol_checkpoint_0.ckpt"


def main():
    random.seed(0)  # reproducible demo; challenges are still "randomly" chosen
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)

    print(f"Prover: building a chain of {N} REAL transformer training steps\n")
    states, commits = [S0], [commit(S0)]
    print(f"  S0 (given)        commit={commits[0]}")
    for k in range(1, N + 1):
        ck = run_step(states[k - 1], BASE + k, f"{WORK}/s{k}")
        states.append(ck); commits.append(commit(ck))
        print(f"  step {k} seed={BASE + k}  commit={commits[k]}")

    print(f"\nVerifier: {CHALLENGES} random challenges (re-run one step each, cheap)")
    ok = True
    for _ in range(CHALLENGES):
        k = random.randint(1, N)
        rc = commit(run_step(states[k - 1], BASE + k, f"{WORK}/v{k}_{random.randrange(10**6)}"))
        m = rc == commits[k]; ok &= m
        print(f"  step {k}: recomputed {rc} vs committed {commits[k]} -> {'MATCH' if m else 'MISMATCH'}")
    print(f"  HONEST PROVER -> {'PASSED' if ok else 'FAILED'}")

    j = random.randint(1, N)
    faked = commits[j - 1]  # liar shortcut step j: commits the prior state as if it worked
    print(f"\nLying node: shortcuts step {j}, commits {faked} (the prior state) instead of doing the work")
    rc = commit(run_step(states[j - 1], BASE + j, f"{WORK}/liar"))  # verifier honestly re-runs step j
    caught = rc != faked
    print(f"  verifier re-executes step {j}: gets {rc}, liar claimed {faked} -> {'CAUGHT' if caught else 'slipped'}")
    print("\nProof-of-learning on the real transformer: honest work verifies, shortcut work is caught.")
    sys.exit(0 if (ok and caught) else 1)


if __name__ == "__main__":
    main()
