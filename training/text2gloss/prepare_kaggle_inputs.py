"""Build the folder to upload to Kaggle as a dataset.

    python prepare_kaggle_inputs.py --how2sign ../../data/how2sign_text/how2sign_realigned_train.csv

Writes kaggle_inputs/ with:
    glosses.json        every sign in the dictionary
    core_glosses.json   the 300 signs used most in the Claude-written pairs
    fewshot.jsonl       8 Claude-written pairs shown to the open model as examples
    claude_pairs.jsonl  all Claude-written pairs, for training
    sentences.txt       How2Sign sentences that have no gloss yet
    *.py                the scripts the notebook runs

The folder derives from How2Sign (CC BY-NC 4.0): keep the Kaggle dataset PRIVATE.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--how2sign", required=True, help="how2sign_realigned_train.csv")
    parser.add_argument("--pairs", default=str(HERE / "pairs.jsonl"))
    parser.add_argument("--glosses", default=str(HERE / "../../data/asl_citizen_glosses.json"))
    parser.add_argument("--out", default=str(HERE / "kaggle_inputs"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    glosses = sorted(json.loads(Path(args.glosses).read_text()))
    counts = Counter(t for p in pairs for t in p["gloss"].split() if not t.startswith("fs-"))
    core = [g for g, _ in counts.most_common(300)]

    # Examples: short enough to keep prompts small, at most one fingerspelled word, no repeats of the same opening.
    rng = random.Random(3)
    pool = [p for p in pairs if 5 <= len(p["english"].split()) <= 14 and p["gloss"].count("fs-") <= 1
            and len(p["gloss"].split()) >= 4]
    rng.shuffle(pool)
    fewshot, openings = [], set()
    for p in pool:
        opening = p["english"].split()[0].lower()
        if opening not in openings:
            fewshot.append(p)
            openings.add(opening)
        if len(fewshot) == 8:
            break

    used = {p["english"].lower() for p in pairs}
    sentences, seen = [], set(used)
    with open(args.how2sign) as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            s = " ".join((row["SENTENCE"] or "").split())
            if 4 <= len(s.split()) <= 18 and s.isascii() and s.lower() not in seen:
                seen.add(s.lower())
                sentences.append(s)
    rng.shuffle(sentences)

    (out / "glosses.json").write_text(json.dumps(glosses))
    (out / "core_glosses.json").write_text(json.dumps(core))
    (out / "fewshot.jsonl").write_text("\n".join(json.dumps(p) for p in fewshot) + "\n")
    (out / "claude_pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    (out / "sentences.txt").write_text("\n".join(sentences) + "\n")
    for script in ("kaggle_generate.py", "train.py", "export_onnx.py"):
        shutil.copy(HERE / script, out / script)
    print(f"{out}: {len(glosses)} signs, {len(core)} core, {len(fewshot)} examples, "
          f"{len(pairs)} Claude pairs, {len(sentences)} sentences to gloss")


if __name__ == "__main__":
    main()
