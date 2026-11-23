# ASLytics

A Chrome extension that shows an ASL-signing 3D avatar beside captioned YouTube videos. Research prototype.

Two systems share one dataset, one pose format and one avatar:

1. **Dictionary baseline:** English → gloss model, then real recorded motion per sign, stitched.
2. **End-to-end model:** English → pose tokens, trained on sentence-level signing.

Both run inside the extension on the viewer's machine. Nothing here is a substitute for a human interpreter, and no accuracy claim is made until ASL-fluent (ideally Deaf) reviewers have rated the output.

## Status

| Milestone | State |
|---|---|
| 1. Dataset pipeline (`pipeline/`) | Code done. Sign dictionary build in progress; How2Sign build not yet run. |
| 2. Extension, avatar, dictionary baseline | Overlay, captions, stitching, 3D stick figure, and the trained English → gloss model running in the background worker (verified in a browser demo page, about 0.2 s per sentence; not yet run as an installed extension). Word lookup is the fallback until a caption is glossed. No VRM avatar, no fingerspelling. |
| 3. End-to-end model | Not started |
| 4. Evaluation | Not started |

## Pipeline

```bash
cd pipeline
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

| Command | What it does |
|---|---|
| `aslytics lexicon --per-sign 2` | Builds the sign dictionary from ASL Citizen. Streams clips out of the 43 GB archive one at a time, keeps only poses. Resumable. About 4 s per clip on an M2 Air. |
| `aslytics how2sign --root <dir> --split train` | Builds the sentence corpus from local How2Sign clips (35 GB; run on Kaggle). Resumable. |
| `aslytics stitch HELLO NAME APPLE` | Stitches dictionary signs into one clip and renders a preview. |
| `aslytics preview <clip.npz>` | Renders a clip as a strip of skeleton frames. |
| `aslytics captions <youtube url>` | Prints timed captions. |

Pose format: 102 points per frame (8 body, 2×21 hand, 52 face), 15 fps, centred on the shoulders with shoulder width 1, float16 `.npz`. Defined in `pipeline/aslytics/landmarks.py`.

## Extension

```bash
cd pipeline && .venv/bin/aslytics export-lexicon     # writes extension/public/lexicon/
cd ../extension && npm install && npm test && npm run build
```

The gloss model must be in `extension/public/models/gloss/` before building (config and tokenizer files plus `onnx/encoder_model_quantized.onnx` and `onnx/decoder_model_merged_quantized.onnx`). See `training/text2gloss/README.md`. Without it the extension still works, using word lookup only.

Load it in Chrome: open `chrome://extensions`, switch on Developer mode, choose "Load unpacked" and pick `extension/dist`. Open a captioned YouTube video; the overlay appears bottom right.

`npm run dev` serves a demo page at http://localhost:5178 for working on stitching and rendering without YouTube.

Captions: YouTube only serves a caption track to the player's own tokenised request, so the extension briefly switches captions on, re-reads the request the player made, and restores the setting. If that fails it signs whatever caption text is on screen.

The dictionary pack in `extension/public/lexicon/` is derived from ASL Citizen. Do not publish a build that contains it.

## Data and licences

Everything under `data/` is gitignored and must not be redistributed.

| Dataset | Used for | Licence |
|---|---|---|
| ASL Citizen (Microsoft) | Sign dictionary, 2,731 signs | Microsoft Research License: non-commercial research only, no redistribution of the data or modifications |
| How2Sign | Sentence corpus, ~80 h | CC BY-NC 4.0 |

A public or commercial release needs different data.

## Compute

Development on an M2 MacBook Air (8 GB). Large extraction and all training run on free Kaggle/Colab GPUs.
