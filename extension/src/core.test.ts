import { describe, expect, it } from "vitest";
import { parseJson3, trackUrlFor } from "./captions";
import { GlossRepairer, WordLookup } from "./gloss";
import { FRAME_SIZE, type Sequence } from "./landmarks";
import { framePosition } from "./signer";
import { bestJoin, stitch } from "./stitch";

/** A clip whose every coordinate ramps linearly from `from` to `to`. */
function ramp(n: number, from: number, to: number): Sequence {
  const frames = new Float32Array(n * FRAME_SIZE);
  for (let t = 0; t < n; t++) frames.fill(from + ((to - from) * t) / (n - 1), t * FRAME_SIZE, (t + 1) * FRAME_SIZE);
  return { frames, n };
}

describe("stitch", () => {
  it("joins at the closest frames", () => {
    // a ends at 1.0; b starts at 0.8 and passes through 1.0 on its second frame.
    expect(bestJoin(ramp(10, 0, 1), ramp(10, 0.8, 2.6))).toEqual([10, 1]);
  });

  it("bridges clips without a jump", () => {
    const out = stitch([ramp(20, 0, 1), ramp(20, 2, 3)]);
    expect(out.n).toBe(20 + 3 + 20);
    let maxStep = 0;
    for (let t = 1; t < out.n; t++) maxStep = Math.max(maxStep, Math.abs(out.frames[t * FRAME_SIZE] - out.frames[(t - 1) * FRAME_SIZE]));
    expect(maxStep).toBeLessThan(0.3); // the raw gap between the clips is 1.0
    expect(out.frames.some(Number.isNaN)).toBe(false);
  });

  it("returns a single clip unchanged and rejects none", () => {
    const clip = ramp(5, 0, 1);
    expect(stitch([clip])).toBe(clip);
    expect(() => stitch([])).toThrow();
  });
});

describe("word lookup (placeholder gloss step)", () => {
  const lookup = new WordLookup(["HELLO", "NAME", "APPLE", "APPLE2", "SOCCER2", "MY", "ME", "WALK", "NOT"]);

  it("maps words, drops function words and reports missing ones", () => {
    const result = lookup.toGloss("Hello, my name is Zorblax!");
    expect(result.glosses).toEqual(["HELLO", "MY", "NAME"]);
    expect(result.missing).toEqual(["zorblax"]);
  });

  it("handles variants, endings, pronouns and contractions", () => {
    expect(lookup.toGloss("apples soccer walking").glosses).toEqual(["APPLE", "SOCCER2", "WALK"]);
    expect(lookup.toGloss("I don't walk").glosses).toEqual(["ME", "NOT", "WALK"]);
    expect(new WordLookup(["PLAN", "RUN"]).toGloss("planned running").glosses).toEqual(["PLAN", "RUN"]);
  });
});

describe("gloss model output repair", () => {
  const repairer = new GlossRepairer(["HELLO", "MY", "NAME", "APPLE", "APPLE2", "WANT1", "WANT2", "ME", "THIS/IT", "DONTLIKE", "BICYCLE"]);

  it("keeps exact signs and fixes near misses", () => {
    expect(repairer.repair("ME DONTLIKE APPLE1").glosses).toEqual(["ME", "DONTLIKE", "APPLE"]);
    expect(repairer.repair("I want this").glosses).toEqual(["ME", "WANT1", "THIS/IT"]);
  });

  it("reports fingerspelled and unknown words as missing", () => {
    const result = repairer.repair("HELLO MY NAME fs-SAM Bike");
    expect(result.glosses).toEqual(["HELLO", "MY", "NAME"]);
    expect(result.missing).toEqual(["sam", "bike"]);
    expect(repairer.repair("").glosses).toEqual([]);
  });
});

describe("captions", () => {
  it("picks the newest caption request for this video, ignoring ads", () => {
    const urls = [
      "https://www.youtube.com/api/timedtext?v=VIDEO&lang=en&pot=1",
      "https://www.youtube.com/api/timedtext?v=VIDEO&lang=en&pot=2",
      "https://www.youtube.com/api/timedtext?v=ADVERT&lang=en&pot=3",
    ];
    expect(trackUrlFor("VIDEO", urls)?.searchParams.get("pot")).toBe("2");
    expect(trackUrlFor("OTHER", urls)).toBeNull();
  });

  it("parses json3 events and trims overlaps", () => {
    const captions = parseJson3({
      events: [
        { tStartMs: 0, dDurationMs: 5000, segs: [{ utf8: "[Music]" }] },
        { tStartMs: 1000, dDurationMs: 4000, segs: [{ utf8: "hello " }, { utf8: "world" }] },
        { tStartMs: 3000, dDurationMs: 2000, segs: [{ utf8: "again" }] },
        { tStartMs: 4000 },
      ],
    });
    expect(captions).toEqual([
      { start: 1, end: 3, text: "hello world" },
      { start: 3, end: 5, text: "again" },
    ]);
  });
});

describe("framePosition", () => {
  const seq = ramp(30, 0, 1); // 2 seconds at 15 fps
  it("plays at natural speed when the caption is long enough", () => {
    expect(framePosition(seq, 1, 4)).toBe(15);
    expect(framePosition(seq, 3.5, 4)).toBe(29); // holds the last frame
  });
  it("speeds up, within a limit, when the caption is short", () => {
    expect(framePosition(seq, 0.5, 1.25)).toBeCloseTo(12); // 1.6x
    expect(framePosition(seq, 0.5, 0.5)).toBeCloseTo(13.5); // capped at 1.8x
  });
});
