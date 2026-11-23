"""Pack the sign dictionary into two files the browser extension loads.

    lexicon.json  {"fps", "points", "scale", "signs": {GLOSS: [first_frame, n_frames]}}
    lexicon.bin   int16 little-endian, frames x points x 3, value = coordinate * scale

Clips are trimmed to the signing itself, so the extension only has to join them.
The pack is derived from ASL Citizen: keep it local, do not publish it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import landmarks as lm
from .posefile import load_clip
from .stitch import signing_bounds

SCALE = 1000
EXPORT_MARGIN = 2


def read_log(data_dir: Path) -> pd.DataFrame:
    """Read the build log directly so a pack can be made while the build is still running."""
    df = pd.read_json(data_dir / "lexicon.jsonl", lines=True, dtype={"signer": str, "gloss": str})
    return df[df["ok"]].drop_duplicates("gloss", keep="first")


def export_lexicon(data_dir: str | Path, out_dir: str | Path) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signs: dict[str, list[int]] = {}
    chunks: list[np.ndarray] = []
    offset = 0
    for row in read_log(data_dir).sort_values("gloss").itertuples():
        clip = load_clip(data_dir / row.pose_path)
        lo, hi = signing_bounds(clip, margin=EXPORT_MARGIN)
        frames = np.clip(np.round(clip.xyz[lo:hi] * SCALE), -32768, 32767).astype("<i2")
        if len(frames) < 3:
            continue
        signs[row.gloss] = [offset, len(frames)]
        chunks.append(frames)
        offset += len(frames)
    index = {"fps": lm.FPS, "points": lm.NUM_POINTS, "scale": SCALE, "signs": signs}
    (out_dir / "lexicon.json").write_text(json.dumps(index, separators=(",", ":")))
    blob = np.concatenate(chunks).tobytes() if chunks else b""
    (out_dir / "lexicon.bin").write_bytes(blob)
    return {"signs": len(signs), "frames": offset, "megabytes": len(blob) / 1e6}
