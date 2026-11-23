"""Make clips comparable across signers and cameras, and score their quality."""

from __future__ import annotations

import numpy as np

from . import landmarks as lm
from .posefile import Clip


def _interp_gaps(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Linearly fill frames where `present` is False. values: (T, N, 3)."""
    if present.all():
        return values
    out = values.copy()
    if not present.any():
        out[:] = 0.0
        return out
    t = np.arange(len(values))
    known = t[present]
    flat = out.reshape(len(values), -1)
    for col in range(flat.shape[1]):
        flat[~present, col] = np.interp(t[~present], known, flat[present, col])
    return flat.reshape(values.shape)


def fill_missing(clip: Clip) -> Clip:
    """Fill undetected components. The mask keeps the truth.

    Body and face are interpolated over time. A hand that was not detected
    (usually because it is resting out of frame) keeps its nearest known shape
    but is attached to the body's wrist, so it never floats away from the arm.
    """
    xyz = clip.xyz.copy()
    xyz[:, lm.BODY] = _interp_gaps(xyz[:, lm.BODY], clip.mask[:, 0])
    xyz[:, lm.FACE] = _interp_gaps(xyz[:, lm.FACE], clip.mask[:, 3])
    for col, hand, wrist in ((1, lm.LEFT_HAND, lm.L_WRIST), (2, lm.RIGHT_HAND, lm.R_WRIST)):
        seen = clip.mask[:, col]
        if seen.all():
            continue
        shape = xyz[:, hand] - xyz[:, hand.start][:, None]  # relative to the hand's own wrist
        shape = _interp_gaps(shape, seen)
        xyz[~seen, hand] = xyz[~seen, wrist][:, None] + shape[~seen]
    return Clip(xyz, clip.mask.copy(), clip.fps)


def normalize(clip: Clip) -> Clip:
    """Centre on the mid-shoulder point and scale shoulder width to 1.

    One centre and one scale per clip (medians), so real body movement is kept.

    MediaPipe reports hand and face depth relative to the wrist / head, not the
    body, so each hand's z is re-anchored onto its body wrist and the face's z
    onto the shoulders. x and y are already in a shared image frame.
    """
    if not clip.mask[:, 0].any():
        raise ValueError("No body detected in clip; cannot normalise")
    clip = fill_missing(clip)
    xyz = clip.xyz.copy()

    for hand, wrist in ((lm.LEFT_HAND, lm.L_WRIST), (lm.RIGHT_HAND, lm.R_WRIST)):
        xyz[:, hand, 2] += xyz[:, wrist, 2][:, None] - xyz[:, hand.start, 2][:, None]
    shoulder_z = (xyz[:, lm.L_SHOULDER, 2] + xyz[:, lm.R_SHOULDER, 2]) / 2
    xyz[:, lm.FACE, 2] += shoulder_z[:, None] - xyz[:, lm.FACE, 2].mean(axis=1, keepdims=True)

    left, right = xyz[:, lm.L_SHOULDER], xyz[:, lm.R_SHOULDER]
    centre = np.median((left + right) / 2, axis=0)
    width = float(np.median(np.linalg.norm((left - right)[:, :2], axis=1)))
    if width < 1e-6:
        raise ValueError("Degenerate shoulder width; cannot normalise")
    return Clip((xyz - centre) / width, clip.mask, clip.fps)


def quality(clip: Clip) -> dict[str, float]:
    """Detection rates used to filter bad clips out of the dataset."""
    if len(clip) == 0:
        return {"body_rate": 0.0, "hand_rate": 0.0, "face_rate": 0.0}
    # Hands only matter while someone is signing. Isolated-sign clips are mostly
    # the signer at rest, so measure hand tracking on frames where a wrist is
    # raised above its elbow (image y points down; NaN compares False).
    with np.errstate(invalid="ignore"):
        raised = (clip.xyz[:, lm.L_WRIST, 1] < clip.xyz[:, lm.L_ELBOW, 1]) | (
            clip.xyz[:, lm.R_WRIST, 1] < clip.xyz[:, lm.R_ELBOW, 1]
        )
    tracked = clip.mask[:, 1] | clip.mask[:, 2]
    return {
        "body_rate": float(clip.mask[:, 0].mean()),
        "hand_rate": float(tracked[raised].mean()) if raised.any() else 0.0,
        "face_rate": float(clip.mask[:, 3].mean()),
    }


def passes_quality(q: dict[str, float], min_body: float = 0.9, min_hand: float = 0.7) -> bool:
    return q["body_rate"] >= min_body and q["hand_rate"] >= min_hand
