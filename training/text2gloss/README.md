# English → ASL gloss model

Replaces the word-lookup placeholder in `extension/src/gloss.ts` with a small trained translator.

| Step | Where | Command |
|---|---|---|
| 1. Collect English sentences | anywhere | one sentence per line in `sentences.txt` (captions, How2Sign English text, etc.) |
| 2. Generate pairs | anywhere with Anthropic credentials | `python generate_pairs.py submit --sentences sentences.txt --glosses ../../data/asl_citizen_glosses.json --limit 200`, then `python generate_pairs.py collect` |
| 3. Train | Kaggle / Colab GPU | `python train.py --pairs pairs.jsonl --out model/` |
| 4. Export | same machine | `python export_onnx.py --model model/ --out onnx_model/` |

Setup: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

Notes:

- Try step 2 with `--limit 200` first and read the output before spending on the full set. The default model is `claude-opus-5-5`; pass `--model` to change it.
- Output glosses are either exact dictionary entries or `fs-WORD` (fingerspelled). Pairs that use any other sign are dropped. The dictionary has no alphabet yet, so the extension cannot perform `fs-` tokens.
- The pairs are machine-written and unreviewed. Have an ASL signer check a sample before trusting metrics computed against them.
- The quantised export is about 150 MB across three files (95 MB if the browser loads only the encoder and one decoder).
- Not done yet: running the exported model inside the extension.

Verified so far only by a smoke test: 600 template sentences, 4 epochs on CPU, export, and inference from the quantised ONNX files.
