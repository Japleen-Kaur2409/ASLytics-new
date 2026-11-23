"""Make English -> ASL gloss pairs with an open model on a free Kaggle GPU.

    python kaggle_generate.py --inputs <folder> --out pairs_kaggle.jsonl

`<folder>` is the one written by prepare_kaggle_inputs.py. A 7B model cannot
hold the whole 2,731-sign dictionary in every prompt at a useful speed, so each
sentence gets a shortlist: the most common signs plus any sign whose name
matches a word in the sentence. Every output is then checked against the FULL
dictionary. Near-misses are repaired (WANT -> WANT1, I -> ME), the model gets one
retry naming the words that were not signs, and anything still invalid is dropped.

Safe to stop and re-run: finished sentences are skipped.

Expect lower quality than the Claude-written pairs. Treat this as bulk data to
mix with them, and compare models trained with and without it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

MODEL = "Qwen/Qwen2.5-7B-Instruct"

RULES = """You translate English sentences into American Sign Language gloss for a signing avatar.

Every word of your answer must be either a sign from the allowed list, copied exactly (including any trailing digit), or fs-WORD for a name or term that must be fingerspelled. Keep fingerspelling rare.

Use ASL structure, not English word order: time first, then topic, then comment. Leave out articles, forms of "to be" and filler words. Put question words at the end and NOT after the verb.

Many plain English words are NOT signs here. There is no I (use ME), no IT or THIS alone (use THIS/IT), no SHE or HER (use HE), no DO, no IS/ARE. Many signs only exist with a digit, such as WANT1, CAN1, WHAT1, THEY1: copy them from the list. If a word is not in the list, rephrase with signs that are, or fingerspell it.

Answer with the gloss only, in capitals, on one line."""


def base_name(gloss: str) -> str:
    """APPLE2 -> APPLE, THIS/IT -> THISIT. Used to match signs to English words."""
    return re.sub(r"[^A-Z]", "", gloss)


def word_forms(word: str) -> set[str]:
    forms = {word}
    for ending, replacements in (("ies", ["y"]), ("es", [""]), ("s", [""]), ("ing", ["", "e"]), ("ed", ["", "e"]), ("ly", [""])):
        if word.endswith(ending) and len(word) > len(ending) + 1:
            forms.update(word[: -len(ending)] + r for r in replacements)
    forms.update(f[:-1] for f in list(forms) if len(f) > 2 and f[-1] == f[-2])  # planned -> plann -> plan
    return {f.upper() for f in forms}


class Shortlister:
    def __init__(self, glosses: list[str], core: list[str]) -> None:
        self.core = list(core)
        self.by_base: dict[str, list[str]] = {}
        for gloss in glosses:
            self.by_base.setdefault(base_name(gloss), []).append(gloss)

    def shortlist(self, sentence: str) -> list[str]:
        chosen = set(self.core)
        for word in re.findall(r"[a-z]+", sentence.lower()):
            for form in word_forms(word):
                chosen.update(self.by_base.get(form, []))
        return sorted(chosen)


def build_messages(sentence: str, allowed: list[str], examples: list[dict]) -> list[dict]:
    shots = "\n".join(f"English: {e['english']}\nGloss: {e['gloss']}" for e in examples)
    user = f"Allowed signs:\n{' '.join(allowed)}\n\nExamples:\n{shots}\n\nEnglish: {sentence}\nGloss:"
    return [{"role": "system", "content": RULES}, {"role": "user", "content": user}]


ALIASES = {"I": "ME", "IT": "THIS/IT", "THIS": "THIS/IT", "SHE": "HE", "HER": "HE", "HIM": "HE",
           "THING": "THINGS", "HAND": "HANDS", "EYE": "EYES", "SAY": "TELL", "LOOK": "LOOKAT"}
# Words ASL does not sign separately; safe to drop if the model writes them anyway.
DROPPABLE = {"A", "AN", "THE", "IS", "ARE", "AM", "WAS", "WERE", "BE", "TO", "OF", "DO", "DOES", "DID", "SO", "JUST"}


class Repairer:
    """Turns a model answer into a performable gloss, fixing near-misses."""

    def __init__(self, glosses: list[str]) -> None:
        self.vocabulary = set(glosses)
        self.variants: dict[str, str] = {}
        for gloss in sorted(glosses, key=lambda g: (len(g), g)):  # APPLE before APPLE2, WANT1 before WANT2
            self.variants.setdefault(base_name(gloss), gloss)

    def repair(self, answer: str) -> tuple[str | None, list[str]]:
        """(gloss, []) when every token could be made valid, else (None, the tokens that could not)."""
        lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
        if not lines:
            return None, []
        first = re.sub(r"^(gloss|asl gloss)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
        out, invalid = [], []
        for raw in first.replace(",", " ").replace(";", " ").split():
            token = raw.strip(".!?\"'()")
            if not token:
                continue
            if token.lower().startswith("fs-"):
                word = re.sub(r"[^A-Za-z]", "", token[3:])
                if word:
                    out.append("fs-" + word.upper())
                continue
            token = token.upper()
            if token in self.vocabulary:
                out.append(token)
            elif token in ALIASES and ALIASES[token] in self.vocabulary:
                out.append(ALIASES[token])
            elif base_name(token) in self.variants:  # WANT -> WANT1
                out.append(self.variants[base_name(token)])
            elif token in DROPPABLE:
                continue
            else:
                invalid.append(token)
        if invalid or not out:
            return None, invalid
        return " ".join(out), []


def load_model(name: str) -> Callable[[list[list[dict]]], list[str]]:
    """Returns a function that answers a batch of chats."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict = {"device_map": "auto"}
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()

    def answer(chats: list[list[dict]]) -> list[str]:
        prompts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)

    return answer


def run(inputs: Path, out: Path, answer: Callable[[list[list[dict]]], list[str]],
        batch_size: int = 8, limit: int | None = None) -> dict:
    glosses = json.loads((inputs / "glosses.json").read_text())
    repairer = Repairer(glosses)
    shortlister = Shortlister(glosses, json.loads((inputs / "core_glosses.json").read_text()))
    examples = [json.loads(line) for line in (inputs / "fewshot.jsonl").read_text().splitlines() if line.strip()]
    sentences = [s for s in (inputs / "sentences.txt").read_text().splitlines() if s.strip()]
    if limit:
        sentences = sentences[:limit]

    done: set[str] = set()
    if out.exists():
        done = {json.loads(line)["english"] for line in out.read_text().splitlines() if line.strip()}
    todo = [s for s in sentences if s not in done]
    kept = dropped = retried = 0
    with out.open("a") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            chats = [build_messages(s, shortlister.shortlist(s), examples) for s in batch]
            raws = answer(chats)
            results = [repairer.repair(raw) for raw in raws]

            # One more try for the failures, telling the model exactly which words were not signs.
            again = [i for i, (gloss, invalid) in enumerate(results) if gloss is None and invalid]
            if again:
                retried += len(again)
                follow_ups = [chats[i] + [
                    {"role": "assistant", "content": raws[i].strip()},
                    {"role": "user", "content": "These are not in the allowed list: " + " ".join(results[i][1])
                     + ". Rewrite the gloss using only allowed signs, or fs-WORD for those words. Gloss only."},
                ] for i in again]
                for i, raw in zip(again, answer(follow_ups)):
                    gloss, invalid = repairer.repair(raw)
                    if gloss is not None:
                        results[i], raws[i] = (gloss, []), raw
                    else:
                        results[i] = (None, invalid or results[i][1])

            for sentence, raw, (gloss, invalid) in zip(batch, raws, results):
                # Rejected sentences are recorded too, so a re-run does not retry them.
                f.write(json.dumps({"english": sentence, "gloss": gloss or "", "ok": gloss is not None,
                                    "raw": raw.strip()[:300], "invalid": invalid}) + "\n")
                kept += gloss is not None
                dropped += gloss is None
            f.flush()
            if (start // batch_size) % 25 == 0:
                print(f"{start + len(batch)}/{len(todo)} done, {kept} kept, {dropped} dropped", flush=True)
    return {"kept": kept, "dropped": dropped, "retried": retried, "skipped": len(sentences) - len(todo)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", default="pairs_kaggle.jsonl")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(run(Path(args.inputs), Path(args.out), load_model(args.model), args.batch_size, args.limit))


if __name__ == "__main__":
    main()
