# English → ASL gloss model

Replaces the word-lookup placeholder in `extension/src/gloss.ts` with a small trained translator.

Two ways to get training pairs, used together:

- **Claude-written pairs** (`pairs.jsonl`): 2,400 How2Sign sentences glossed against the full dictionary. Best quality, small. Made either in a Claude Code session or with `generate_pairs.py` (needs API credentials).
- **Open-model pairs on Kaggle** (`kaggle_gloss.ipynb`): free and larger, lower quality. The notebook also trains the model with and without them and scores both on the same held-out Claude pairs.

| Step | Where | Command |
|---|---|---|
| 1. Build the Kaggle upload | Mac | `python prepare_kaggle_inputs.py --how2sign ../../data/how2sign_text/how2sign_realigned_train.csv` |
| 2. Upload `kaggle_inputs/` as a **private** Kaggle dataset | kaggle.com | New Dataset → upload folder |
| 3. Generate pairs and train | Kaggle, GPU T4 x2, Internet on | upload and run `kaggle_gloss.ipynb` |
| 4. Export for the browser | Mac | unzip `models.zip`, then `python export_onnx.py --model model_with_kaggle --out onnx_model/` |

Setup: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

Notes:

- Try step 2 with `--limit 200` first and read the output before spending on the full set. The default model is `claude-opus-5-5`; pass `--model` to change it.
- Output glosses are either exact dictionary entries or `fs-WORD` (fingerspelled). Pairs that use any other sign are dropped. The dictionary has no alphabet yet, so the extension cannot perform `fs-` tokens.
- The pairs are machine-written and unreviewed. Have an ASL signer check a sample before trusting metrics computed against them.
- The quantised export is about 150 MB across three files (95 MB if the browser loads only the encoder and one decoder).
- Not done yet: running the exported model inside the extension.

Verified so far: a smoke test of train and export (600 template sentences, 4 epochs on CPU), and the Kaggle generator's shortlisting, validation and resume logic with a stand-in model. The notebook itself has not been run on Kaggle, and no model has been trained on the real pairs yet (training does not fit on an 8 GB Mac alongside other work).
