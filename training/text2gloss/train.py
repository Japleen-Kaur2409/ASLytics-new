"""Fine-tune t5-small to translate English into ASL gloss.

    python train.py --pairs pairs.jsonl --out model/

Runs on a Kaggle or Colab GPU in well under an hour for ~50k pairs; on CPU only
for smoke tests. Reports exact-match and token F1 on a held-out 5%.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PREFIX = "translate English to ASL gloss: "
MAX_SOURCE, MAX_TARGET = 96, 64


class Pairs(Dataset):
    def __init__(self, pairs: list[dict], tokenizer) -> None:
        self.pairs, self.tokenizer = pairs, tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        pair = self.pairs[i]
        item = self.tokenizer(PREFIX + pair["english"], max_length=MAX_SOURCE, truncation=True)
        item["labels"] = self.tokenizer(text_target=pair["gloss"], max_length=MAX_TARGET, truncation=True)["input_ids"]
        return item


def token_f1(predicted: str, reference: str) -> float:
    p, r = predicted.split(), reference.split()
    if not p or not r:
        return float(p == r)
    remaining, overlap = list(r), 0
    for token in p:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--extra", help="more pairs, used for training only; the held-out set always comes from --pairs")
    parser.add_argument("--out", default="model")
    parser.add_argument("--base", default="t5-small")
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    pairs = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    random.Random(0).shuffle(pairs)
    n_val = max(1, len(pairs) // 20)
    val, train = pairs[:n_val], pairs[n_val:]
    if args.extra:
        held_out = {p["english"] for p in val}
        extra = [json.loads(line) for line in Path(args.extra).read_text().splitlines() if line.strip()]
        train += [p for p in extra if p.get("ok", True) and p["gloss"] and p["english"] not in held_out]
        random.Random(1).shuffle(train)
    print(f"{len(train)} training pairs, {len(val)} held out")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)

    def metrics(prediction) -> dict:
        labels = np.where(prediction.label_ids < 0, tokenizer.pad_token_id, prediction.label_ids)
        predictions = np.where(prediction.predictions < 0, tokenizer.pad_token_id, prediction.predictions)
        decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {
            "exact_match": float(np.mean([d.strip() == r.strip() for d, r in zip(decoded, references)])),
            "token_f1": float(np.mean([token_f1(d, r) for d, r in zip(decoded, references)])),
        }

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=f"{args.out}/checkpoints",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="no",
            predict_with_generate=True,
            generation_max_length=MAX_TARGET,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to=[],
        ),
        train_dataset=Pairs(train, tokenizer),
        eval_dataset=Pairs(val, tokenizer),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=metrics,
    )
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    model.eval()
    for pair in val[:8]:
        ids = tokenizer(PREFIX + pair["english"], return_tensors="pt").to(model.device)
        out = tokenizer.decode(model.generate(**ids, max_length=MAX_TARGET)[0], skip_special_tokens=True)
        print(f"{pair['english']}\n  expected: {pair['gloss']}\n  got:      {out}")


if __name__ == "__main__":
    main()
