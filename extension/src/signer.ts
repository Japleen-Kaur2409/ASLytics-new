// Text -> pose sequence, and where in that sequence playback should be.

import { FPS, type Sequence } from "./landmarks";
import type { Lexicon } from "./lexicon";
import { WordLookup, type GlossResult } from "./gloss";
import { stitch } from "./stitch";

const MAX_SPEEDUP = 1.8;

export interface Signed extends GlossResult {
  sequence: Sequence | null;
}

export class Signer {
  private lookup: WordLookup;
  private cache = new Map<string, Signed>();

  constructor(private lexicon: Lexicon) {
    this.lookup = new WordLookup(lexicon.glosses);
  }

  sign(text: string): Signed {
    let result = this.cache.get(text);
    if (!result) {
      const gloss = this.lookup.toGloss(text);
      const clips = gloss.glosses.map((g) => this.lexicon.get(g));
      result = { ...gloss, sequence: clips.length ? stitch(clips) : null };
      this.cache.set(text, result);
    }
    return result;
  }
}

/**
 * Frame position for a caption that has been on screen for `elapsed` seconds.
 * Signing plays at natural speed; if it would overrun the caption it speeds up,
 * to at most MAX_SPEEDUP, and holds the last frame once finished.
 */
export function framePosition(seq: Sequence, elapsed: number, captionDuration: number): number {
  const natural = seq.n / FPS;
  const speed = Math.min(MAX_SPEEDUP, Math.max(1, natural / Math.max(captionDuration, 0.1)));
  return Math.min(seq.n - 1, Math.max(0, elapsed * FPS * speed));
}
