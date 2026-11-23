// PLACEHOLDER English -> gloss step.
//
// This looks English words up in the sign dictionary one by one, in English word
// order. That is signed English at best, not ASL: it has no ASL grammar, no
// fingerspelling, and silently drops any word without a dictionary entry.
// It exists so the rest of the extension can be built and tested, and is to be
// replaced by the trained English -> gloss model (Milestone 2, step 2).

const CONTRACTIONS: Record<string, string> = {
  "it's": "it is", "you're": "you are", "i'm": "i am", "don't": "do not", "can't": "can not",
  "they're": "they are", "i'll": "i will", "you'll": "you will", "we'll": "we will",
  "won't": "will not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
  "weren't": "were not", "doesn't": "does not", "didn't": "did not", "that's": "that is",
  "there's": "there is", "what's": "what is", "let's": "let us", "we're": "we are",
  "i've": "i have", "you've": "you have", "we've": "we have", "he's": "he is", "she's": "she is",
};

// Words ASL normally does not sign separately.
const DROPPED = new Set(["a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being", "to", "of", "do", "does", "did"]);

const PRONOUNS: Record<string, string> = { i: "ME", my: "MY", me: "ME", you: "YOU", your: "YOUR", we: "WE", us: "WE" };

/** Dictionary glosses to try for one English word, most likely first. */
function candidates(word: string): string[] {
  const out = [word];
  if (PRONOUNS[word]) out.unshift(PRONOUNS[word].toLowerCase());
  if (word.endsWith("ies")) out.push(word.slice(0, -3) + "y");
  if (word.endsWith("es")) out.push(word.slice(0, -2));
  if (word.endsWith("s")) out.push(word.slice(0, -1));
  if (word.endsWith("ing")) out.push(word.slice(0, -3), word.slice(0, -3) + "e");
  if (word.endsWith("ed")) out.push(word.slice(0, -2), word.slice(0, -1));
  // planned -> plann -> plan, running -> runn -> run
  for (const w of [...out]) if (/([a-z])\1$/.test(w)) out.push(w.slice(0, -1));
  return out.map((w) => w.toUpperCase());
}

export interface GlossResult {
  glosses: string[];
  /** English words that had no dictionary entry and were dropped. */
  missing: string[];
}

export class WordLookup {
  private byName = new Map<string, string>();

  constructor(dictionaryGlosses: string[]) {
    // ASL Citizen names variants APPLE, APPLE2... Prefer the unnumbered one, else the lowest number.
    for (const gloss of [...dictionaryGlosses].sort()) {
      const base = gloss.replace(/[^A-Z]/g, "");
      if (base && !this.byName.has(base)) this.byName.set(base, gloss);
    }
    for (const gloss of dictionaryGlosses) {
      if (/^[A-Z]+$/.test(gloss)) this.byName.set(gloss, gloss);
    }
  }

  toGloss(text: string): GlossResult {
    const words = text
      .toLowerCase()
      .replace(/’/g, "'")
      .split(/\s+/)
      .flatMap((w) => (CONTRACTIONS[w.replace(/[^a-z']/g, "")] ?? w).split(" "))
      .map((w) => w.replace(/[^a-z]/g, ""))
      .filter((w) => w && !DROPPED.has(w));
    const glosses: string[] = [];
    const missing: string[] = [];
    for (const word of words) {
      const hit = candidates(word).map((c) => this.byName.get(c)).find(Boolean);
      if (hit) glosses.push(hit);
      else missing.push(word);
    }
    return { glosses, missing };
  }
}
