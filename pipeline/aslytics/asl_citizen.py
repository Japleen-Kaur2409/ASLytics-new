"""Build the sign dictionary (gloss -> pose clips) from ASL Citizen.

ASL Citizen: 83,399 videos of 2,731 isolated signs by 52 Deaf/HoH signers.
Microsoft Research License: non-commercial research only, no redistribution of
the data or of modifications to it. So the poses we derive stay in data/ and
out of git, and this project stays a research prototype while it uses them.

We only need a few clean examples per sign, so clips are pulled one at a time
out of the 43 GB archive with range requests; the videos are never kept.
"""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .extract_pose import extract_clip
from .httpzip import open_remote_zip
from .manifest import ManifestLog
from .normalize import normalize, passes_quality, quality
from .posefile import save_clip

ARCHIVE_URL = ("https://download.microsoft.com/download/b/8/8/"
               "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip")
LICENCE = "Microsoft Research License (non-commercial, no redistribution)"
SEED_SIGNER = "P52"  # the dataset's model signer; her recordings are the reference ones
CSV_COLUMNS = ["Participant ID", "Video file", "Gloss", "ASL-LEX Code"]


def load_splits(archive: zipfile.ZipFile) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        raw = archive.read(f"ASL_Citizen/splits/{split}.csv")
        df = pd.read_csv(io.BytesIO(raw), dtype=str)
        if list(df.columns) != CSV_COLUMNS:
            raise ValueError(f"Unexpected columns in {split}.csv: {list(df.columns)}")
        df.columns = ["signer", "video_file", "gloss", "asl_lex_code"]
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def choose_exemplars(df: pd.DataFrame, per_sign: int = 2) -> pd.DataFrame:
    """Up to `per_sign` videos per gloss: the seed signer first, then others in a fixed order."""
    df = df.copy()
    df["_rank"] = (df["signer"] != SEED_SIGNER).astype(int)
    df = df.sort_values(["gloss", "_rank", "signer", "video_file"])
    return df.groupby("gloss", sort=True).head(per_sign).drop(columns="_rank").reset_index(drop=True)


def safe_name(gloss: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", gloss).strip("_").upper()


def build_lexicon(
    data_dir: str | Path,
    per_sign: int = 2,
    limit: int | None = None,
    glosses: list[str] | None = None,
) -> pd.DataFrame:
    """Extract dictionary poses into data_dir/lexicon/. Safe to stop and re-run."""
    data_dir = Path(data_dir)
    log = ManifestLog(data_dir, "lexicon")
    archive = open_remote_zip(ARCHIVE_URL)
    chosen = choose_exemplars(load_splits(archive), per_sign)
    if glosses:
        chosen = chosen[chosen["gloss"].isin(glosses)]
    if limit:
        chosen = chosen.head(limit)

    counters: dict[str, int] = {}
    for row in tqdm(chosen.itertuples(), total=len(chosen), desc="lexicon"):
        n = counters.get(row.gloss, 0)
        counters[row.gloss] = n + 1
        clip_id = f"{safe_name(row.gloss)}__{n}"
        if clip_id in log.done:
            continue
        record = {
            "clip_id": clip_id, "source": "asl_citizen", "licence": LICENCE,
            "split": row.split, "kind": "sign", "text": "", "gloss": row.gloss,
            "signer": row.signer, "n_frames": 0, "pose_path": "",
            "body_rate": 0.0, "hand_rate": 0.0, "face_rate": 0.0, "ok": False,
        }
        try:
            video = archive.read(f"ASL_Citizen/videos/{row.video_file}")
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                tmp.write(video)
                tmp.flush()
                clip = extract_clip(tmp.name)
            q = quality(clip)
            record.update(q, n_frames=len(clip))
            if len(clip) >= 5 and passes_quality(q):
                rel = Path("lexicon") / f"{clip_id}.npz"
                save_clip(normalize(clip), data_dir / rel)
                record.update(pose_path=str(rel), ok=True)
        except Exception as exc:  # one bad video must not stop a multi-hour run
            tqdm.write(f"skip {row.video_file}: {exc}")
        log.append(record)
    return log.compact()
