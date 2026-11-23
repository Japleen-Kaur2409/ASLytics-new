"""Join isolated-sign clips into one sequence (the dictionary baseline).

Same method as the old ASLytics prototype (itself from sign-language-processing's
spoken-to-signed-translation, MIT): trim each sign to where the hands are up,
cut neighbouring signs at their closest frames, bridge the gap by interpolation,
then lightly smooth everything except the face.

Works on normalised clips. `extension/src/stitch.ts` mirrors this file.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

from . import landmarks as lm
from .posefile import Clip

TRIM_MARGIN = 1        # frames kept either side of the active part
EDGE_MARGIN = 3        # extra frames kept at the start and end of the sentence
JOIN_WINDOW = 0.3     # seconds searched at each end for the best join
TRANSITION = 0.2       # seconds of interpolated movement between signs


def signing_bounds(clip: Clip, margin: int = TRIM_MARGIN) -> tuple[int, int]:
    """First and last frame of actual signing, plus a margin.

    A frame counts when a wrist is above its elbow (image y points down) and
    that hand was really tracked. Either test alone lets resting frames through.
    """
    left = (clip.xyz[:, lm.L_WRIST, 1] < clip.xyz[:, lm.L_ELBOW, 1]) & clip.mask[:, 1]
    right = (clip.xyz[:, lm.R_WRIST, 1] < clip.xyz[:, lm.R_ELBOW, 1]) & clip.mask[:, 2]
    up = left | right
    if not up.any():
        return 0, len(clip)
    first = int(np.argmax(up))
    last = len(clip) - int(np.argmax(up[::-1]))
    return max(0, first - margin), min(len(clip), last + margin)


def best_join(a: Clip, b: Clip) -> tuple[int, int]:
    """(last frame to keep from a, exclusive; first frame to keep from b)."""
    na = max(1, math.ceil(min(JOIN_WINDOW * a.fps, len(a) * JOIN_WINDOW)))
    nb = max(1, math.ceil(min(JOIN_WINDOW * b.fps, len(b) * JOIN_WINDOW)))
    # The face barely moves between signs, so match on body and hands only.
    tail = a.xyz[len(a) - na:, : lm.FACE.start].reshape(na, -1)
    head = b.xyz[:nb, : lm.FACE.start].reshape(nb, -1)
    i, j = np.unravel_index(np.argmin(cdist(tail, head)), (na, nb))
    return len(a) - na + int(i) + 1, int(j)


def stitch(clips: list[Clip], trim: bool = True) -> Clip:
    if not clips:
        raise ValueError("No clips to stitch")
    fps = clips[0].fps
    if any(abs(c.fps - fps) > 1e-6 for c in clips):
        raise ValueError("All clips must share one frame rate")
    if any(len(c) == 0 for c in clips):
        raise ValueError("Cannot stitch an empty clip")

    if trim:
        last = len(clips) - 1
        trimmed = []
        for i, c in enumerate(clips):
            lo, hi = signing_bounds(c)
            # Leave a little more lead-in and lead-out at the ends of the sentence.
            if i == 0:
                lo = max(0, lo - EDGE_MARGIN)
            if i == last:
                hi = min(len(c), hi + EDGE_MARGIN)
            trimmed.append(c.slice(lo, hi))
        clips = trimmed
    if len(clips) == 1:
        return clips[0]

    starts = [0] * len(clips)
    ends = [len(c) for c in clips]
    for i in range(len(clips) - 1):
        ends[i], starts[i + 1] = best_join(clips[i], clips[i + 1])
        ends[i] = max(ends[i], starts[i] + 1)

    gap = max(1, int(round(TRANSITION * fps)))
    blend = np.linspace(0, 1, gap + 2, dtype=np.float32)[1:-1, None, None]
    xyz_parts, mask_parts = [], []
    for i, c in enumerate(clips):
        xyz_parts.append(c.xyz[starts[i]:ends[i]])
        mask_parts.append(c.mask[starts[i]:ends[i]])
        if i < len(clips) - 1:
            a = c.xyz[ends[i] - 1]
            b = clips[i + 1].xyz[starts[i + 1]]
            xyz_parts.append(a * (1 - blend) + b * blend)
            mask_parts.append(np.repeat(c.mask[ends[i] - 1][None], gap, axis=0))
    xyz = np.concatenate(xyz_parts).astype(np.float32)
    mask = np.concatenate(mask_parts)

    if len(xyz) >= 3:
        body_hands = slice(0, lm.FACE.start)
        xyz[:, body_hands] = savgol_filter(xyz[:, body_hands], 3, 1, axis=0)
    return Clip(xyz, mask, fps)
