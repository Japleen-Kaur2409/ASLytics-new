// The canonical ASLytics pose layout. Must match pipeline/aslytics/landmarks.py.
//   [0,8)    upper body: shoulders, elbows, wrists, hips (left then right)
//   [8,29)   left hand, 21 points
//   [29,50)  right hand, 21 points
//   [50,102) face subset

export const FPS = 15;
export const NUM_POINTS = 102;
export const FRAME_SIZE = NUM_POINTS * 3;

export const L_SHOULDER = 0, R_SHOULDER = 1, L_ELBOW = 2, R_ELBOW = 3;
export const L_HIP = 6, R_HIP = 7;
export const LEFT_HAND = 8, RIGHT_HAND = 29, FACE = 50;
export const HAND_POINTS = 21;

/** A pose sequence: `n` frames of FRAME_SIZE floats (x, y, z per point; y points down). */
export interface Sequence {
  frames: Float32Array;
  n: number;
}

const FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
const HAND_BONES: [number, number][] = [
  ...FINGERS.flatMap((f) => f.slice(1).map((b, i) => [f[i], b] as [number, number])),
  [5, 9], [9, 13], [13, 17],
];

/** Pairs of point indices to draw as bones. Forearms end at the hand's own wrist so arm and hand stay joined. */
export const BONES: [number, number][] = [
  [L_SHOULDER, R_SHOULDER],
  [L_SHOULDER, L_ELBOW], [L_ELBOW, LEFT_HAND],
  [R_SHOULDER, R_ELBOW], [R_ELBOW, RIGHT_HAND],
  [L_SHOULDER, L_HIP], [R_SHOULDER, R_HIP], [L_HIP, R_HIP],
  ...HAND_BONES.map(([a, b]) => [LEFT_HAND + a, LEFT_HAND + b] as [number, number]),
  ...HAND_BONES.map(([a, b]) => [RIGHT_HAND + a, RIGHT_HAND + b] as [number, number]),
];
export const BODY_BONE_COUNT = 8;
