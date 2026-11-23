// Timed captions for the video being watched, read from YouTube inside the page.

export interface Caption {
  start: number;
  end: number;
  text: string;
}

export function cleanText(text: string): string {
  return text
    .replace(/\[[^\]]*\]|\([^)]*\)/g, " ")
    .replace(/>>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

interface Json3 {
  events?: { tStartMs?: number; dDurationMs?: number; segs?: { utf8?: string }[] }[];
}

export function parseJson3(data: Json3): Caption[] {
  const captions: Caption[] = [];
  for (const event of data.events ?? []) {
    if (!event.segs || event.tStartMs === undefined) continue;
    const text = cleanText(event.segs.map((s) => s.utf8 ?? "").join(""));
    if (!text) continue;
    const start = event.tStartMs / 1000;
    captions.push({ start, end: start + (event.dDurationMs ?? 2000) / 1000, text });
  }
  captions.sort((a, b) => a.start - b.start);
  // Auto-generated captions overlap as they scroll; end each one where the next begins.
  for (let i = 0; i < captions.length - 1; i++) {
    captions[i].end = Math.min(captions[i].end, captions[i + 1].start);
  }
  return captions.filter((c) => c.end - c.start > 0.05);
}

// YouTube only serves a caption track to requests carrying a token that the
// player generates, so a plain fetch of the track comes back empty. Instead we
// watch for the request the player itself makes when captions are switched on,
// and re-read that same URL as JSON.

const seenTrackUrls: string[] = [];

function remember(entries: PerformanceEntry[]): void {
  for (const entry of entries) {
    if (entry.name.includes("/api/timedtext") && !seenTrackUrls.includes(entry.name)) seenTrackUrls.push(entry.name);
  }
}

/** Start recording the player's caption requests. Call once, as early as possible. */
export function watchCaptionRequests(): void {
  remember(performance.getEntriesByType("resource"));
  // An observer still sees requests after the page's resource timing buffer has filled up.
  new PerformanceObserver((list) => remember(list.getEntries())).observe({ type: "resource", buffered: true });
}

/** The newest caption request the player made for this video (not for an ad). */
export function trackUrlFor(videoId: string, urls: string[] = seenTrackUrls): URL | null {
  for (let i = urls.length - 1; i >= 0; i--) {
    const url = new URL(urls[i]);
    if (url.searchParams.get("v") === videoId) return url;
  }
  return null;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const adPlaying = () => document.querySelector(".ad-showing") !== null;

/**
 * Fetch the whole caption track up front. Briefly switches captions on if they
 * are off, so the player requests the track, then restores the viewer's setting.
 * `cancelled` lets the caller abandon the wait when the viewer leaves the video.
 */
export async function fetchCaptions(videoId: string, cancelled: () => boolean = () => false): Promise<Caption[]> {
  let url = trackUrlFor(videoId);
  if (!url) {
    for (let i = 0; i < 180 && adPlaying() && !cancelled(); i++) await sleep(1000);
    const button = document.querySelector<HTMLElement>(".ytp-subtitles-button");
    if (!button || button.getAttribute("aria-disabled") === "true") throw new Error("This video has no captions");
    const wasOff = button.getAttribute("aria-pressed") !== "true";
    if (!wasOff) {
      // Captions were already on, so the request happened before we were watching
      // (YouTube clears the page's resource log). Off then on makes the player ask again.
      button.click();
      await sleep(300);
    }
    button.click();
    for (let i = 0; i < 40 && !url && !cancelled(); i++) {
      await sleep(250);
      url = trackUrlFor(videoId);
    }
    if (wasOff && button.getAttribute("aria-pressed") === "true") button.click();
  }
  if (cancelled()) throw new Error("Cancelled");
  if (!url) throw new Error("Could not read this video's captions");
  const language = url.searchParams.get("tlang") ?? url.searchParams.get("lang") ?? "";
  if (!language.startsWith("en")) throw new Error("Captions are not in English");
  url.searchParams.set("fmt", "json3");
  const res = await fetch(url, { credentials: "include" });
  const body = await res.text();
  if (!res.ok || !body.trim()) throw new Error("YouTube did not return the caption track");
  const captions = parseJson3(JSON.parse(body) as Json3);
  if (captions.length === 0) throw new Error("The caption track was empty");
  return captions;
}

/** Fallback: the caption text currently shown on the player (needs captions switched on). */
export function visibleCaptionText(): string {
  const segments = document.querySelectorAll<HTMLElement>(".ytp-caption-segment");
  return cleanText(Array.from(segments, (s) => s.textContent ?? "").join(" "));
}
