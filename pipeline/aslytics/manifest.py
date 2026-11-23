"""The dataset index: one row per pose clip, stored as parquet.

Rows are appended to a .jsonl log while processing (so a crashed or timed-out
run can resume) and compacted to parquet at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COLUMNS = [
    "clip_id",     # unique, also the pose file stem
    "source",      # asl_citizen | how2sign
    "licence",
    "split",       # train | val | test
    "kind",        # sign (isolated) | sentence
    "text",        # English sentence, empty for isolated signs
    "gloss",       # sign label, empty for sentences
    "signer",
    "n_frames",
    "pose_path",   # relative to the data directory
    "body_rate", "hand_rate", "face_rate",
    "ok",          # passed the quality gate
]


class ManifestLog:
    def __init__(self, data_dir: str | Path, name: str) -> None:
        self.data_dir = Path(data_dir)
        self.log_path = self.data_dir / f"{name}.jsonl"
        self.parquet_path = self.data_dir / f"{name}.parquet"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if self.log_path.exists():
            with self.log_path.open() as f:
                self.done = {json.loads(line)["clip_id"] for line in f if line.strip()}

    def append(self, row: dict) -> None:
        missing = set(COLUMNS) - set(row)
        if missing:
            raise ValueError(f"Manifest row is missing {sorted(missing)}")
        with self.log_path.open("a") as f:
            f.write(json.dumps({c: row[c] for c in COLUMNS}) + "\n")
        self.done.add(row["clip_id"])

    def compact(self) -> pd.DataFrame:
        if self.log_path.exists():
            df = pd.read_json(self.log_path, lines=True, dtype={"signer": str, "gloss": str})
        else:
            df = pd.DataFrame(columns=COLUMNS)
        df = df.drop_duplicates("clip_id", keep="last").reset_index(drop=True)
        df.to_parquet(self.parquet_path, index=False)
        return df


def load_manifest(data_dir: str | Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{name}.parquet")
