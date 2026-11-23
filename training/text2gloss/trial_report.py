"""Summarise a kaggle_generate.py output file: python trial_report.py trial.jsonl"""
import collections
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print(sum(r["ok"] for r in rows), "of", len(rows), "kept")
invalid = collections.Counter(t for r in rows for t in r.get("invalid", []))
print("words that were not signs:", invalid.most_common(30), "\n")
for r in rows[: int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print(r["english"])
    print("   ->", r["gloss"] if r["ok"] else f"(dropped) model wrote: {r.get('raw', '')}")
