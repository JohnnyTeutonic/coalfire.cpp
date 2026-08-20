"""Prepare chat/instruction datasets in transformer_cpp's data contract.

Same contract as prepare_tinystories.py (train-0/1.txt, validation.txt,
*-small.txt), with dialogue turns flattened into "user: ... assistant: ..."
lines — turn markers become ordinary words for the word-level tokenizer.

Datasets (choose with --dataset; vocabulary-simplicity ranked for the 20k cap):
  tinystories-instruct  roneneldan/TinyStories-Instruct (instruction variant;
                        same simple vocabulary as TinyStories — best fit)
  dailydialog           daily_dialog (13k everyday multi-turn dialogues)
  dolly                 databricks/databricks-dolly-15k (real instructions;
                        richer vocabulary — expect more <unk> at 20k cap)

    python scripts/prepare_chat_data.py --dataset tinystories-instruct
"""
import argparse
import os
import random


def tinychat_rows(n_dialogues=150_000, seed=7):
    """Synthetic dialogue with CONSISTENT question->answer semantics and a
    deliberately tiny vocabulary (~1.5k words).

    Rationale (2026-07-20): the mid model trained on DailyDialog mastered
    conversational REGISTER but not answer-type pairing ("how are you?" ->
    "no , we will be fine") — open-domain vocabulary/topic breadth exceeds
    13M params. TinyStories' thesis applied to dialogue: restrict the world
    to match capacity and coherence becomes learnable. Templates are
    stochastic (slots, paraphrases, turn counts) so the model must bind
    meaning, not memorize strings.
    """
    rng = random.Random(seed)
    foods = ["pizza", "pasta", "rice", "soup", "salad", "eggs", "bread",
             "cheese", "apples", "fish", "chicken", "pancakes", "noodles"]
    drinks = ["tea", "coffee", "juice", "water", "milk"]
    places = ["the park", "the beach", "the market", "the library",
              "the museum", "the garden", "the city", "the lake"]
    activities = ["reading", "cooking", "running", "painting", "swimming",
                  "gardening", "playing chess", "watching movies", "hiking"]
    weather = ["sunny", "rainy", "cold", "warm", "windy", "cloudy"]
    feelings_good = ["great", "very good", "happy", "fine", "wonderful"]
    feelings_bad = ["tired", "a little sad", "busy", "sleepy", "not so good"]
    days = ["today", "yesterday", "this morning", "last night", "this week"]
    pets = ["dog", "cat", "bird", "rabbit"]

    def greeting(rng):
        g = rng.choice(["hi !", "hello !", "good morning !", "hey there !"])
        r = rng.choice(["hi ! nice to see you .", "hello ! how are you ?",
                        "good morning ! it is nice to see you ."])
        return [("user", g), ("assistant", r)]

    def qa_pair(rng):
        kind = rng.randrange(8)
        if kind == 0:
            f = rng.choice(feelings_good) if rng.random() < 0.7 else rng.choice(feelings_bad)
            return [("user", rng.choice(["how are you ?", "how do you feel " + rng.choice(days) + " ?"])),
                    ("assistant", rng.choice(["i am " + f + " , thank you .",
                                              "i feel " + f + " " + rng.choice(days) + " ."]))]
        if kind == 1:
            food = rng.choice(foods)
            return [("user", rng.choice(["what do you like to eat ?", "what is your favorite food ?"])),
                    ("assistant", rng.choice(["i like " + food + " very much .",
                                              "my favorite food is " + food + " ."]))]
        if kind == 2:
            d = rng.choice(drinks)
            return [("user", rng.choice(["would you like some " + d + " ?", "do you want " + d + " or " + rng.choice(drinks) + " ?"])),
                    ("assistant", rng.choice(["yes please , i would love some " + d + " .",
                                              "no thank you , i just had some " + rng.choice(drinks) + " ."]))]
        if kind == 3:
            p = rng.choice(places)
            a = rng.choice(activities)
            return [("user", rng.choice(["where did you go " + rng.choice(days) + " ?", "what did you do " + rng.choice(days) + " ?"])),
                    ("assistant", rng.choice(["i went to " + p + " .", "i spent the day " + a + " .",
                                              "i went to " + p + " and enjoyed " + a + " ."]))]
        if kind == 4:
            w = rng.choice(weather)
            return [("user", "how is the weather " + rng.choice(days) + " ?"),
                    ("assistant", rng.choice(["it is very " + w + " .", "it was " + w + " all day ."]))]
        if kind == 5:
            a = rng.choice(activities)
            return [("user", rng.choice(["what is your hobby ?", "what do you like to do ?"])),
                    ("assistant", rng.choice(["i really enjoy " + a + " .", "my hobby is " + a + " ."]))]
        if kind == 6:
            pet = rng.choice(pets)
            return [("user", rng.choice(["do you have a pet ?", "do you like animals ?"])),
                    ("assistant", rng.choice(["yes , i have a little " + pet + " .",
                                              "yes , my " + pet + " is very sweet ."]))]
        p = rng.choice(places)
        return [("user", rng.choice(["do you want to go to " + p + " with me ?", "shall we visit " + p + " ?"])),
                ("assistant", rng.choice(["yes , that sounds lovely .", "sure , i would like that .",
                                          "sorry , i am too busy " + rng.choice(days) + " ."]))]

    def farewell(rng):
        return [("user", rng.choice(["i have to go now . goodbye !", "see you later !"])),
                ("assistant", rng.choice(["goodbye ! have a nice day .", "see you soon ! take care ."]))]

    for _ in range(n_dialogues):
        turns = greeting(rng)
        for _ in range(rng.randrange(1, 4)):
            turns += qa_pair(rng)
        if rng.random() < 0.5:
            turns += farewell(rng)
        yield " ".join(f"{who}: {text}" for who, text in turns)


def rows(dataset):
    from datasets import load_dataset
    if dataset == "tinystories-instruct":
        # The HF dataset serves examples LINE-split: each "Words:"/"Summary:"
        # field, each story paragraph, and literal "<|endoftext|>" delimiter
        # lines arrive as separate rows. Trained as-is (found 2026-07-19),
        # the fields and their stories never share a context window, so
        # words-/summary-conditioning cannot be learned. Re-aggregate into
        # one row per example, flushing on the delimiter lines.
        buf = []
        for ex in load_dataset("roneneldan/TinyStories-Instruct", split="train")["text"]:
            piece = " ".join(str(ex).split())
            if piece == "<|endoftext|>" or not piece:
                if buf:
                    yield " ".join(buf)
                    buf = []
                continue
            buf.append(piece)
        if buf:
            yield " ".join(buf)
    elif dataset == "dailydialog":
        # daily_dialog ships a legacy loading script that modern `datasets`
        # refuses ("Dataset scripts are no longer supported", 2026-07-19).
        # Read the hub's auto-converted parquet branch instead — no script.
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        repo = "li2017dailydialog/daily_dialog"
        files = [f for f in api.list_repo_files(repo, repo_type="dataset",
                                                revision="refs/convert/parquet")
                 if f.endswith(".parquet") and "train" in f]
        if not files:
            raise RuntimeError(f"no parquet shards found for {repo}")
        for fname in sorted(files):
            local = hf_hub_download(repo, fname, repo_type="dataset",
                                    revision="refs/convert/parquet")
            for ex in pd.read_parquet(local)["dialog"]:
                turns = []
                for i, utt in enumerate(ex):
                    who = "user" if i % 2 == 0 else "assistant"
                    turns.append(f"{who}: {' '.join(str(utt).split())}")
                yield " ".join(turns)
    elif dataset == "tinychat":
        yield from tinychat_rows()
    elif dataset == "tinychatmix":
        # Simple-vocab synthetic backbone + natural DailyDialog variety
        yield from tinychat_rows(n_dialogues=100_000)
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        repo = "li2017dailydialog/daily_dialog"
        files = [f for f in api.list_repo_files(repo, repo_type="dataset",
                                                revision="refs/convert/parquet")
                 if f.endswith(".parquet") and "train" in f]
        for fname in sorted(files):
            local = hf_hub_download(repo, fname, repo_type="dataset",
                                    revision="refs/convert/parquet")
            for ex in pd.read_parquet(local)["dialog"]:
                turns = []
                for i, utt in enumerate(ex):
                    who = "user" if i % 2 == 0 else "assistant"
                    turns.append(f"{who}: {' '.join(str(utt).split())}")
                yield " ".join(turns)
    elif dataset in ("empathetic", "dialogmix"):
        def parquet_rows(repo, split="train"):
            import pandas as pd
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()
            files = [f for f in api.list_repo_files(repo, repo_type="dataset",
                                                    revision="refs/convert/parquet")
                     if f.endswith(".parquet") and split in f]
            for fname in sorted(files):
                local = hf_hub_download(repo, fname, repo_type="dataset",
                                        revision="refs/convert/parquet")
                yield pd.read_parquet(local)

        def dailydialog_rows():
            for df in parquet_rows("li2017dailydialog/daily_dialog"):
                for ex in df["dialog"]:
                    turns = []
                    for i, utt in enumerate(ex):
                        who = "user" if i % 2 == 0 else "assistant"
                        turns.append(f"{who}: {' '.join(str(utt).split())}")
                    yield " ".join(turns)

        def empathetic_rows():
            # empathetic_dialogues stores one UTTERANCE per row; group by
            # conv_id, order by utterance_idx, undo the "_comma_" escaping.
            for df in parquet_rows("empathetic_dialogues"):
                for _, conv in df.sort_values("utterance_idx").groupby("conv_id"):
                    turns = []
                    for i, utt in enumerate(conv["utterance"]):
                        text = " ".join(str(utt).replace("_comma_", ",").split())
                        who = "user" if i % 2 == 0 else "assistant"
                        turns.append(f"{who}: {text}")
                    if len(turns) >= 2:
                        yield " ".join(turns)

        if dataset == "empathetic":
            yield from empathetic_rows()
        else:  # dialogmix: everyday transactional + emotional register
            yield from dailydialog_rows()
            yield from empathetic_rows()
    elif dataset == "dolly":
        for ex in load_dataset("databricks/databricks-dolly-15k", split="train"):
            ctx = " ".join(str(ex.get("context", "")).split())
            q = " ".join(str(ex["instruction"]).split())
            a = " ".join(str(ex["response"]).split())
            yield (f"user: {q} {ctx} assistant: {a}").strip()
    else:
        raise SystemExit(f"unknown dataset {dataset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tinystories-instruct",
                    choices=["tinystories-instruct", "dailydialog", "empathetic",
                             "dialogmix", "tinychat", "tinychatmix", "dolly"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-rows", type=int, default=600_000)
    ap.add_argument("--small-rows", type=int, default=5_000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()
    out = args.out or f"data/{args.dataset}-txt"
    os.makedirs(out, exist_ok=True)

    # <|endoftext|> after every dialogue: the trainer's packing loader
    # concatenates lines, so without it dialogues bleed into each other with
    # no learnable boundary. The GGUF exporter prefers this token as EOS, so
    # the served model stops at the end of its reply.
    lines = []
    for i, line in enumerate(rows(args.dataset)):
        if line:
            lines.append(line + " <|endoftext|>")
        if len(lines) >= args.max_rows:
            break
    n_val = max(50, int(len(lines) * args.val_frac))
    val, train = lines[:n_val], lines[n_val:]
    half = len(train) // 2

    def write(name, chunk):
        p = os.path.join(out, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk) + "\n")
        print(f"  {name}: {len(chunk)} rows, {os.path.getsize(p)/1e6:.1f} MB")

    write("train-0.txt", train[:half])
    write("train-1.txt", train[half:])
    write("train-0-small.txt", train[:args.small_rows])
    write("train-1-small.txt", train[args.small_rows:2 * args.small_rows])
    write("validation.txt", val)
    print(f"done -> {out}  (train with: train ../{out})")


if __name__ == "__main__":
    main()
