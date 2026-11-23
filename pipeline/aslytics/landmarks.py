"""The canonical ASLytics pose layout.

Every clip in the dataset, every model input/output and the browser avatar use
this one layout: 102 points per frame, each (x, y, z).

    [0:8]     upper body   (shoulders, elbows, wrists, hips)
    [8:29]    left hand    (21 MediaPipe hand landmarks)
    [29:50]   right hand   (21 MediaPipe hand landmarks)
    [50:102]  face subset  (lips, eyebrows, eyes, nose tip, chin)

`extension/src/landmarks.ts` must stay in sync with this file.
"""

from __future__ import annotations

FPS = 15

# Indices into MediaPipe's 33 pose landmarks.
BODY_NAMES = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
]
BODY_MP_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]

HAND_POINTS = 21

# Indices into MediaPipe's 468-point face mesh.
FACE_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                   291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FACE_LIPS_INNER = [78, 81, 13, 311, 308, 402, 14, 178]
FACE_BROW_LEFT = [70, 63, 105, 66, 107]
FACE_BROW_RIGHT = [336, 296, 334, 293, 300]
FACE_EYE_LEFT = [33, 160, 158, 133, 153, 144]
FACE_EYE_RIGHT = [362, 385, 387, 263, 373, 380]
FACE_NOSE_CHIN = [1, 152]
FACE_MP_INDICES = (FACE_LIPS_OUTER + FACE_LIPS_INNER + FACE_BROW_LEFT + FACE_BROW_RIGHT
                   + FACE_EYE_LEFT + FACE_EYE_RIGHT + FACE_NOSE_CHIN)

BODY = slice(0, 8)
LEFT_HAND = slice(8, 29)
RIGHT_HAND = slice(29, 50)
FACE = slice(50, 50 + len(FACE_MP_INDICES))
NUM_POINTS = FACE.stop

# Order of the per-frame visibility mask columns.
COMPONENTS = ["body", "left_hand", "right_hand", "face"]
COMPONENT_SLICES = [BODY, LEFT_HAND, RIGHT_HAND, FACE]

# Handy single-point indices in the canonical layout.
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = range(8)

assert NUM_POINTS == 102
