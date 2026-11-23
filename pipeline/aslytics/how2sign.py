"""Build the sentence corpus (English sentence <-> signing) from How2Sign.

How2Sign: ~80 h of ASL by 11 signers, CC BY-NC 4.0 (research only).
The frontal sentence clips are 31 GB (train) + 1.7 GB (val) + 2.2 GB (test) on
Google Drive, so run this where the clips are (a Kaggle notebook), not on a laptop.

Expected input, as unpacked by the official download_how2sign.sh:
    <root>/sentence_level/<split>/rgb_front/raw_videos/<SENTENCE_NAME>.mp4
    <root>/sentence_level/<split>/text/en/raw_text/re_aligned/how2sign_realigned_<split>.csv

Each clip is one English sentence. The clips are cut at the original timestamps;
the "re-aligned" CSV has better sentence text/timing but we only use its text here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .extract_pose import extract_clip
from .manifest import ManifestLog
from .normalize import normalize, passes_quality, quality
from .posefile import save_clip

LICENCE = "CC BY-NC 4.0"
REQUIRED = ["SENTENCE_NAME", "SENTENCE"]


def find_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} under {root}")
    return matches[0]


def load_sentences(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="\t", dtype=str)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns {missing}; found {list(df.columns)}. "
            "Check the file is the tab-separated How2Sign sentence CSV."
        )
    return df.dropna(subset=REQUIRED).reset_index(drop=True)


def signer_of(sentence_name: str) -> str:
    """Clip names look like '<video>-<signer id>-rgb_front' followed by a sentence number."""
    parts = sentence_name.rsplit("-", 2)
    return parts[1] if len(parts) == 3 else ""


def build_corpus(root: str | Path, data_dir: str | Path, split: str, limit: int | None = None) -> pd.DataFrame:
    """Extract sentence poses into data_dir/how2sign/<split>/. Safe to stop and re-run."""
    root, data_dir = Path(root), Path(data_dir)
    sentences = load_sentences(find_file(root, f"how2sign_realigned_{split}.csv"))
    videos = {p.stem: p for p in root.rglob("*.mp4") if "rgb_front" in str(p)}
    if not videos:
        raise FileNotFoundError(f"No rgb_front .mp4 clips found under {root}")
    log = ManifestLog(data_dir, f"how2sign_{split}")
    if limit:
        sentences = sentences.head(limit)

    missing = 0
    for row in tqdm(sentences.itertuples(), total=len(sentences), desc=f"how2sign {split}"):
        clip_id = row.SENTENCE_NAME
        if clip_id in log.done:
            continue
        video = videos.get(clip_id)
        if video is None:
            missing += 1
            continue
        record = {
            "clip_id": clip_id, "source": "how2sign", "licence": LICENCE,
            "split": split, "kind": "sentence", "text": row.SENTENCE.strip(), "gloss": "",
            "signer": signer_of(clip_id), "n_frames": 0, "pose_path": "",
            "body_rate": 0.0, "hand_rate": 0.0, "face_rate": 0.0, "ok": False,
        }
        try:
            clip = extract_clip(video)
            q = quality(clip)
            record.update(q, n_frames=len(clip))
            if len(clip) >= 5 and passes_quality(q):
                rel = Path("how2sign") / split / f"{clip_id}.npz"
                save_clip(normalize(clip), data_dir / rel)
                record.update(pose_path=str(rel), ok=True)
        except Exception as exc:
            tqdm.write(f"skip {clip_id}: {exc}")
        log.append(record)
    if missing:
        print(f"{missing} sentences had no matching clip under {root}")
    return log.compact()
