"""Reading and writing clips in the canonical layout (compressed .npz)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import landmarks as lm


@dataclass
class Clip:
    """A pose sequence.

    xyz:  (T, 102, 3) float32. Missing components are NaN before `fill_missing`.
    mask: (T, 4) bool. Whether body / left hand / right hand / face was detected.
    """

    xyz: np.ndarray
    mask: np.ndarray
    fps: float = lm.FPS

    def __post_init__(self) -> None:
        if self.xyz.ndim != 3 or self.xyz.shape[1:] != (lm.NUM_POINTS, 3):
            raise ValueError(f"xyz must be (T, {lm.NUM_POINTS}, 3), got {self.xyz.shape}")
        if self.mask.shape != (len(self.xyz), len(lm.COMPONENTS)):
            raise ValueError(f"mask must be (T, {len(lm.COMPONENTS)}), got {self.mask.shape}")

    def __len__(self) -> int:
        return len(self.xyz)

    @property
    def duration(self) -> float:
        return len(self) / self.fps

    def slice(self, start: int, stop: int) -> "Clip":
        return Clip(self.xyz[start:stop].copy(), self.mask[start:stop].copy(), self.fps)


def save_clip(clip: Clip, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=np.nan_to_num(clip.xyz, nan=0.0).astype(np.float16),
        mask=clip.mask.astype(bool),
        fps=np.float32(clip.fps),
    )


def load_clip(path: str | Path) -> Clip:
    with np.load(path) as data:
        return Clip(data["xyz"].astype(np.float32), data["mask"].astype(bool), float(data["fps"]))
