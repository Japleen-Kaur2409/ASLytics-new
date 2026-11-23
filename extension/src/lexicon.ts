import { FRAME_SIZE, NUM_POINTS, type Sequence } from "./landmarks";

interface LexiconIndex {
  fps: number;
  points: number;
  scale: number;
  signs: Record<string, [number, number]>;
}

/** The sign dictionary: gloss -> recorded pose clip. Built by `aslytics export-lexicon`. */
export class Lexicon {
  private constructor(private index: LexiconIndex, private data: Int16Array) {}

  static async load(baseUrl: string): Promise<Lexicon> {
    const [indexRes, binRes] = await Promise.all([fetch(`${baseUrl}lexicon.json`), fetch(`${baseUrl}lexicon.bin`)]);
    if (!indexRes.ok || !binRes.ok) throw new Error("Sign dictionary files are missing");
    const index = (await indexRes.json()) as LexiconIndex;
    if (index.points !== NUM_POINTS) throw new Error(`Dictionary has ${index.points} points, expected ${NUM_POINTS}`);
    return new Lexicon(index, new Int16Array(await binRes.arrayBuffer()));
  }

  static fromData(index: LexiconIndex, data: Int16Array): Lexicon {
    return new Lexicon(index, data);
  }

  get glosses(): string[] {
    return Object.keys(this.index.signs);
  }

  has(gloss: string): boolean {
    return gloss in this.index.signs;
  }

  get(gloss: string): Sequence {
    const entry = this.index.signs[gloss];
    if (!entry) throw new Error(`Not in the dictionary: ${gloss}`);
    const [first, n] = entry;
    const frames = new Float32Array(n * FRAME_SIZE);
    const start = first * FRAME_SIZE;
    for (let i = 0; i < frames.length; i++) frames[i] = this.data[start + i] / this.index.scale;
    return { frames, n };
  }
}
