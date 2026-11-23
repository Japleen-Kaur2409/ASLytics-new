"""Video -> canonical pose clip, using MediaPipe Holistic.

Frames are sampled down to `landmarks.FPS` while decoding, so we never run the
detector on frames we would throw away.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import landmarks as lm
from .posefile import Clip


def _points(landmark_list, indices, aspect: float) -> np.ndarray:
    """MediaPipe landmarks -> (N, 3) with x scaled so x and y share one unit (image height)."""
    all_points = landmark_list.landmark
    out = np.empty((len(indices), 3), dtype=np.float32)
    for row, i in enumerate(indices):
        p = all_points[i]
        out[row] = (p.x * aspect, p.y, p.z * aspect)
    return out


def extract_clip(
    video_path: str | Path,
    start: float | None = None,
    end: float | None = None,
    model_complexity: int = 1,
) -> Clip:
    """Run Holistic over a video (optionally only between `start` and `end` seconds)."""
    import mediapipe as mp  # heavy import, keep it local

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect = width / height if height else 1.0

    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    t0 = start or 0.0
    hand_idx = list(range(lm.HAND_POINTS))

    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    next_sample = t0
    frame_no = 0
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        refine_face_landmarks=False,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = t0 + frame_no / src_fps
            frame_no += 1
            if end is not None and t >= end:
                break
            if t + 1e-6 < next_sample:
                continue
            next_sample += 1.0 / lm.FPS

            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            xyz = np.full((lm.NUM_POINTS, 3), np.nan, dtype=np.float32)
            mask = np.zeros(len(lm.COMPONENTS), dtype=bool)
            if result.pose_landmarks:
                xyz[lm.BODY] = _points(result.pose_landmarks, lm.BODY_MP_INDICES, aspect)
                mask[0] = True
            if result.left_hand_landmarks:
                xyz[lm.LEFT_HAND] = _points(result.left_hand_landmarks, hand_idx, aspect)
                mask[1] = True
            if result.right_hand_landmarks:
                xyz[lm.RIGHT_HAND] = _points(result.right_hand_landmarks, hand_idx, aspect)
                mask[2] = True
            if result.face_landmarks:
                xyz[lm.FACE] = _points(result.face_landmarks, lm.FACE_MP_INDICES, aspect)
                mask[3] = True
            frames.append(xyz)
            masks.append(mask)
    cap.release()

    if not frames:
        return Clip(
            np.zeros((0, lm.NUM_POINTS, 3), np.float32),
            np.zeros((0, len(lm.COMPONENTS)), bool),
        )
    return Clip(np.stack(frames), np.stack(masks))
