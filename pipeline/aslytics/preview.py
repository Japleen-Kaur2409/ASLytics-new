"""Draw a pose clip as a strip of skeleton frames, for eyeballing dataset quality."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import landmarks as lm
from .posefile import Clip

BODY_BONES = [
    (lm.L_SHOULDER, lm.R_SHOULDER), (lm.L_SHOULDER, lm.L_ELBOW), (lm.L_ELBOW, lm.L_WRIST),
    (lm.R_SHOULDER, lm.R_ELBOW), (lm.R_ELBOW, lm.R_WRIST),
    (lm.L_SHOULDER, lm.L_HIP), (lm.R_SHOULDER, lm.R_HIP), (lm.L_HIP, lm.R_HIP),
]
_FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
HAND_BONES = [(a, b) for finger in _FINGERS for a, b in zip(finger, finger[1:])]


def save_preview(clip: Clip, path: str | Path, frames: int = 8, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = np.linspace(0, len(clip) - 1, min(frames, len(clip))).astype(int)
    fig, axes = plt.subplots(1, len(picks), figsize=(2.2 * len(picks), 3), squeeze=False)
    for ax, t in zip(axes[0], picks):
        p = clip.xyz[t]
        for a, b in BODY_BONES:
            ax.plot(p[[a, b], 0], p[[a, b], 1], color="0.35", lw=2)
        for col, part, colour in ((1, lm.LEFT_HAND, "tab:blue"), (2, lm.RIGHT_HAND, "tab:red")):
            hand = p[part]
            # Dashed when the hand was not detected in this frame and is interpolated.
            style = "-" if clip.mask[t, col] else ":"
            for a, b in HAND_BONES:
                ax.plot(hand[[a, b], 0], hand[[a, b], 1], style, color=colour, lw=1)
        ax.scatter(p[lm.FACE, 0], p[lm.FACE, 1], s=2, color="tab:green")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(1.8, -1.4)  # image y points down
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{t / clip.fps:.1f}s", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
