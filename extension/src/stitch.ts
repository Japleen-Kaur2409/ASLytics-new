// Join dictionary signs into one sequence. Mirrors pipeline/aslytics/stitch.py,
// minus trimming: the dictionary pack is already trimmed to the signing itself.

import { FACE, FPS, FRAME_SIZE, type Sequence } from "./landmarks";

const JOIN_WINDOW = 0.3; // seconds searched at each end for the best join
const TRANSITION = 0.2;  // seconds of interpolated movement between signs
const MATCH_SIZE = FACE * 3; // match on body and hands only; the face barely moves

/** [last frame to keep from a (exclusive), first frame to keep from b]. */
export function bestJoin(a: Sequence, b: Sequence): [number, number] {
  const na = Math.max(1, Math.ceil(Math.min(JOIN_WINDOW * FPS, a.n * JOIN_WINDOW)));
  const nb = Math.max(1, Math.ceil(Math.min(JOIN_WINDOW * FPS, b.n * JOIN_WINDOW)));
  let best = Infinity, bestI = a.n - 1, bestJ = 0;
  for (let i = a.n - na; i < a.n; i++) {
    for (let j = 0; j < nb; j++) {
      let d = 0;
      const oa = i * FRAME_SIZE, ob = j * FRAME_SIZE;
      for (let k = 0; k < MATCH_SIZE; k++) {
        const diff = a.frames[oa + k] - b.frames[ob + k];
        d += diff * diff;
      }
      if (d < best) { best = d; bestI = i; bestJ = j; }
    }
  }
  return [bestI + 1, bestJ];
}

/** Three-frame linear smoothing of body and hands (same as scipy's savgol_filter(x, 3, 1)). */
function smooth(frames: Float32Array, n: number): void {
  if (n < 3) return;
  const src = frames.slice();
  const at = (t: number, k: number) => src[t * FRAME_SIZE + k];
  for (let k = 0; k < MATCH_SIZE; k++) {
    frames[k] = (5 * at(0, k) + 2 * at(1, k) - at(2, k)) / 6;
    for (let t = 1; t < n - 1; t++) frames[t * FRAME_SIZE + k] = (at(t - 1, k) + at(t, k) + at(t + 1, k)) / 3;
    frames[(n - 1) * FRAME_SIZE + k] = (5 * at(n - 1, k) + 2 * at(n - 2, k) - at(n - 3, k)) / 6;
  }
}

export function stitch(clips: Sequence[]): Sequence {
  if (clips.length === 0) throw new Error("No clips to stitch");
  if (clips.some((c) => c.n === 0)) throw new Error("Cannot stitch an empty clip");
  if (clips.length === 1) return clips[0];

  const starts = clips.map(() => 0);
  const ends = clips.map((c) => c.n);
  for (let i = 0; i < clips.length - 1; i++) {
    const [end, nextStart] = bestJoin(clips[i], clips[i + 1]);
    ends[i] = Math.max(end, starts[i] + 1);
    starts[i + 1] = nextStart;
  }

  const gap = Math.max(1, Math.round(TRANSITION * FPS));
  const total = clips.reduce((sum, _, i) => sum + ends[i] - starts[i], 0) + gap * (clips.length - 1);
  const out = new Float32Array(total * FRAME_SIZE);
  let cursor = 0;
  clips.forEach((clip, i) => {
    const part = clip.frames.subarray(starts[i] * FRAME_SIZE, ends[i] * FRAME_SIZE);
    out.set(part, cursor);
    cursor += part.length;
    if (i === clips.length - 1) return;
    const a = clip.frames.subarray((ends[i] - 1) * FRAME_SIZE, ends[i] * FRAME_SIZE);
    const next = clips[i + 1];
    const b = next.frames.subarray(starts[i + 1] * FRAME_SIZE, (starts[i + 1] + 1) * FRAME_SIZE);
    for (let g = 1; g <= gap; g++) {
      const w = g / (gap + 1);
      for (let k = 0; k < FRAME_SIZE; k++) out[cursor + k] = a[k] * (1 - w) + b[k] * w;
      cursor += FRAME_SIZE;
    }
  });
  smooth(out, total);
  return { frames: out, n: total };
}
