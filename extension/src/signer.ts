// Text -> pose sequence, and where in that sequence playback should be.

import { FPS, type Sequence } from "./landmarks";
import type { Lexicon } from "./lexicon";
import { GlossRepairer, WordLookup, type GlossResult } from "./gloss";
import { stitch } from "./stitch";

const MAX_SPEEDUP = 1.8;

export interface Signed extends GlossResult {
  sequence: Sequence | null;
  /** "model" when the trained gloss model produced this, "lookup" for the word-by-word fallback. */
  source: "model" | "lookup";
}

export class Signer {
  private lookup: WordLookup;
  private repairer: GlossRepairer;
  private cache = new Map<string, Signed>();

  constructor(private lexicon: Lexicon) {
    this.lookup = new WordLookup(lexicon.glosses);
    this.repairer = new GlossRepairer(lexicon.glosses);
  }

  private build(gloss: GlossResult, source: Signed["source"]): Signed {
    const clips = gloss.glosses.map((g) => this.lexicon.get(g));
    return { ...gloss, source, sequence: clips.length ? stitch(clips) : null };
  }

  /** Record the trained model's gloss for a caption; it replaces the fallback from then on. */
  setModelGloss(text: string, modelOutput: string): void {
    this.cache.set(text, this.build(this.repairer.repair(modelOutput), "model"));
  }

  hasModelGloss(text: string): boolean {
    return this.cache.get(text)?.source === "model";
  }

  sign(text: string): Signed {
    let result = this.cache.get(text);
    if (!result) {
      result = this.build(this.lookup.toGloss(text), "lookup");
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
