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
- ## Putting a model in the extension

The browser library wants a merged decoder, which `export_onnx.py` does not produce:

```bash
optimum-cli export onnx --model model_7200 --task text2text-generation-with-past onnx_merged
```

Then quantise `encoder_model.onnx` and `decoder_model_merged.onnx` (same `ORTQuantizer` call as in `export_onnx.py`) into `extension/public/models/gloss/onnx/`, and copy `config.json`, `generation_config.json` and the tokenizer files next to it. About 77 MB in total.

Quantisation costs some accuracy: the int8 browser model wrote TASTE for "apples" where the int8 model checked in Python wrote APPLE1. Not yet measured on the held-out set.

## Results so far

Scored on the same 120 held-out Claude-written sentences (token F1 against the reference gloss):

| Training pairs | Token F1 | Exact match |
|---|---|---|
| 2,400 | 0.61 | 0.8% |
| 7,200 | 0.70 | 13.3% |

These measure agreement with Claude-written glosses, which no ASL signer has reviewed.

The open-model (Qwen2.5-7B) trial kept 21 of 40 sentences after repair and retry, and what it kept was mostly English word order, so the full run was not done.

Note: models trained on Kaggle are saved by a newer `transformers` whose tokenizer config the pinned local version cannot read. Before exporting, replace the tokenizer files with `AutoTokenizer.from_pretrained("t5-small").save_pretrained(<model dir>)`; fine-tuning does not change the tokenizer.

## Putting a model in the extension

The browser library wants a merged decoder, which `export_onnx.py` does not produce:

```bash
optimum-cli export onnx --model model_7200 --task text2text-generation-with-past onnx_merged
```

Then quantise `encoder_model.onnx` and `decoder_model_merged.onnx` (same `ORTQuantizer` call as in `export_onnx.py`) into `extension/public/models/gloss/onnx/`, and copy `config.json`, `generation_config.json` and the tokenizer files next to it. About 77 MB in total.

Quantisation costs some accuracy: the int8 browser model wrote TASTE for "apples" where the int8 model checked in Python wrote APPLE1. Not yet measured on the held-out set.
