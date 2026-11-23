"""Make English -> ASL gloss training pairs with Claude, constrained to our sign dictionary.

    python generate_pairs.py submit  --sentences sentences.txt --glosses ../../data/asl_citizen_glosses.json
    python generate_pairs.py collect --out pairs.jsonl

`submit` sends the sentences through the Message Batches API (half price,
usually done within an hour) and records the batch id in batch_state.json.
`collect` waits for it, validates every gloss against the dictionary and
writes one {"english", "gloss"} object per line.

Needs Anthropic credentials (ANTHROPIC_API_KEY, or an `ant auth login` profile).

These glosses are machine-written. They have not been checked by an ASL signer
and should be treated as a starting point, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

MODEL = "claude-opus-5-5"
SENTENCES_PER_REQUEST = 20
STATE = Path("batch_state.json")

INSTRUCTIONS = """You translate English sentences into American Sign Language gloss for a signing avatar.

The avatar can only perform the signs in the dictionary below, so every gloss you write must be one of two things:
- a dictionary entry, copied exactly as listed (including any trailing digit, which marks a variant of the sign), or
- fs-WORD, meaning the word is fingerspelled. Use this for names, brands, technical terms and anything the dictionary cannot express. Keep it rare: prefer rephrasing with dictionary signs when that keeps the meaning.

Write real ASL structure rather than English word order:
- Put time expressions first, then the topic, then the comment.
- Leave out articles, forms of "to be", and other words ASL does not sign separately.
- Put wh-question signs at the end of the sentence. Put negation after the verb it negates.
- Express plurals, tense and intensity with separate signs (MANY, FINISH, WILL, VERY) only when the meaning needs them.
- When a dictionary entry has numbered variants and you cannot tell which fits, use the unnumbered or lowest-numbered one.

Captions are often fragments or contain filler. Translate what is there. If a sentence has nothing signable, return an empty gloss for it.

Return one result per input sentence, with the same index, as a space-separated gloss string.

Dictionary:
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "gloss": {"type": "string"}},
                "required": ["index", "gloss"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def load_sentences(path: Path) -> list[str]:
    seen, out = set(), []
    for line in path.read_text().splitlines():
        s = " ".join(line.split())
        if 3 <= len(s) <= 300 and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def submit(args: argparse.Namespace) -> None:
    if STATE.exists():
        raise SystemExit(f"{STATE} already exists. Run `collect`, or delete it to start a new batch.")
    glosses = sorted(json.loads(Path(args.glosses).read_text()))
    sentences = load_sentences(Path(args.sentences))
    if args.limit:
        sentences = sentences[: args.limit]
    # The dictionary is identical in every request, so it is cached after the first.
    system = [{"type": "text", "text": INSTRUCTIONS + " ".join(glosses), "cache_control": {"type": "ephemeral"}}]

    requests = []
    for start in range(0, len(sentences), SENTENCES_PER_REQUEST):
        chunk = sentences[start:start + SENTENCES_PER_REQUEST]
        numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(chunk))
        requests.append(Request(
            custom_id=f"chunk-{start}",
            params=MessageCreateParamsNonStreaming(
                model=args.model,
                max_tokens=16000,
                system=system,
                output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": numbered}],
            ),
        ))

    batch = anthropic.Anthropic().messages.batches.create(requests=requests)
    STATE.write_text(json.dumps({"batch_id": batch.id, "sentences": sentences, "glosses": glosses}))
    print(f"Submitted {len(sentences)} sentences in {len(requests)} requests. Batch {batch.id}")


def clean(gloss: str, vocabulary: set[str]) -> tuple[str, list[str]]:
    """Keep dictionary glosses and fs- tokens; report anything else."""
    kept, rejected = [], []
    for token in gloss.split():
        if token in vocabulary:
            kept.append(token)
        elif token.lower().startswith("fs-") and token[3:].isalpha():
            kept.append("fs-" + token[3:].upper())
        else:
            rejected.append(token)
    return " ".join(kept), rejected


def collect(args: argparse.Namespace) -> None:
    state = json.loads(STATE.read_text())
    sentences, vocabulary = state["sentences"], set(state["glosses"])
    client = anthropic.Anthropic()
    while True:
        batch = client.messages.batches.retrieve(state["batch_id"])
        if batch.processing_status == "ended":
            break
        print(f"{batch.processing_status}: {batch.request_counts.processing} requests still processing")
        time.sleep(60)

    pairs, failed, invalid = [], 0, 0
    for result in client.messages.batches.results(state["batch_id"]):  # arrives in any order
        start = int(result.custom_id.removeprefix("chunk-"))
        if result.result.type != "succeeded":
            failed += 1
            continue
        message = result.result.message
        if message.stop_reason != "end_turn":  # refusal or max_tokens: the JSON may be missing or cut off
            failed += 1
            continue
        text = next((b.text for b in message.content if b.type == "text"), "")
        for item in json.loads(text)["results"]:
            index = start + item["index"]
            if not 0 <= item["index"] < SENTENCES_PER_REQUEST or index >= len(sentences):
                continue
            gloss, rejected = clean(item["gloss"], vocabulary)
            if rejected:  # the model used a sign we cannot perform: drop the pair rather than teach a gap
                invalid += 1
                continue
            if gloss:
                pairs.append({"english": sentences[index], "gloss": gloss})

    with Path(args.out).open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"{len(pairs)} pairs written to {args.out}. "
          f"{failed} requests failed, {invalid} pairs dropped for out-of-dictionary signs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("submit")
    p.add_argument("--sentences", required=True, help="text file, one English sentence per line")
    p.add_argument("--glosses", required=True, help="JSON list of dictionary glosses")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--limit", type=int, help="only the first N sentences (try 200 first)")
    p.set_defaults(func=submit)
    p = sub.add_parser("collect")
    p.add_argument("--out", default="pairs.jsonl")
    p.set_defaults(func=collect)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
