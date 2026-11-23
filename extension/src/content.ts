// Content script: adds the signing overlay to YouTube watch pages.

import { fetchCaptions, visibleCaptionText, watchCaptionRequests, type Caption } from "./captions";
import type { Sequence } from "./landmarks";
import { Lexicon } from "./lexicon";
import { Renderer } from "./renderer";
import { framePosition, Signer, type Signed } from "./signer";

const WIDTH = 240;
// Shrink the figure on short windows so the overlay never runs off the top.
const HEIGHT = Math.max(140, Math.min(280, window.innerHeight - 230));

let signer: Signer | null = null;
let session: Session | null = null;

function videoIdFromLocation(): string | null {
  return location.pathname === "/watch" ? new URLSearchParams(location.search).get("v") : null;
}

/** Index of the caption covering time t, or -1. Captions are sorted by start. */
export function captionAt(captions: Caption[], t: number): number {
  let lo = 0, hi = captions.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (captions[mid].start <= t) lo = mid + 1;
    else hi = mid - 1;
  }
  return hi >= 0 && t < captions[hi].end ? hi : -1;
}

class Session {
  private root = document.createElement("div");
  private status = document.createElement("div");
  private glossLine = document.createElement("div");
  private renderer = new Renderer(WIDTH, HEIGHT);
  private captions: Caption[] | null = null;
  private live = { text: "", since: 0 };
  private rest: Sequence | null = null;
  private raf = 0;
  private stopped = false;

  constructor(private videoId: string, private signerRef: Signer) {
    this.root.id = "aslytics-overlay";
    this.root.style.cssText =
      `position:fixed;right:24px;bottom:96px;width:${WIDTH}px;z-index:2147483000;` +
      "background:rgba(250,250,250,.94);border-radius:12px;box-shadow:0 4px 18px rgba(0,0,0,.35);" +
      "font:12px/1.4 system-ui,sans-serif;color:#222;overflow:hidden";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;justify-content:space-between;align-items:center;padding:6px 10px;background:#2d3436;color:#fff";
    header.textContent = "ASLytics · experimental";
    const close = document.createElement("button");
    close.textContent = "×";
    close.setAttribute("aria-label", "Close ASLytics overlay");
    close.style.cssText = "background:none;border:0;color:#fff;font-size:16px;cursor:pointer;padding:0 2px";
    close.onclick = () => this.stop();
    header.append(close);
    this.glossLine.style.cssText = "padding:4px 10px;min-height:17px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
    this.status.style.cssText = "padding:0 10px 8px;color:#666;font-size:11px";
    this.root.append(header, this.renderer.canvas, this.glossLine, this.status);
    document.body.append(this.root);
    void this.start();
  }

  private async start(): Promise<void> {
    this.status.textContent = "Loading captions…";
    try {
      this.captions = await fetchCaptions(this.videoId, () => this.stopped);
      this.status.textContent = "Word-by-word signs, not full ASL";
    } catch (err) {
      this.status.textContent = `${(err as Error).message}. Turn on captions (CC) to sign what is shown.`;
    }
    if (!this.stopped) this.raf = requestAnimationFrame(this.tick);
  }

  private tick = (): void => {
    if (this.stopped) return;
    this.raf = requestAnimationFrame(this.tick);
    const video = document.querySelector<HTMLVideoElement>("video.html5-main-video");
    if (!video) return;

    let signed: Signed | null = null;
    let elapsed = 0, duration = 1;
    if (this.captions) {
      const i = captionAt(this.captions, video.currentTime);
      if (i >= 0) {
        const caption = this.captions[i];
        signed = this.signerRef.sign(caption.text);
        elapsed = video.currentTime - caption.start;
        duration = caption.end - caption.start;
      }
    } else {
      // Fallback: sign whatever caption text the player is showing right now.
      const text = visibleCaptionText();
      if (text !== this.live.text) this.live = { text, since: video.currentTime };
      if (text) {
        signed = this.signerRef.sign(text);
        elapsed = Math.max(0, video.currentTime - this.live.since);
        duration = 3;
      }
    }

    if (signed?.sequence) {
      this.rest = signed.sequence;
      this.glossLine.textContent = signed.glosses.join(" ");
      this.renderer.draw(signed.sequence, framePosition(signed.sequence, elapsed, duration));
    } else {
      this.glossLine.textContent = "";
      // Between captions, hold the end of the last sign rather than vanish.
      if (this.rest) this.renderer.draw(this.rest, this.rest.n - 1);
    }
  };

  stop(): void {
    this.stopped = true;
    cancelAnimationFrame(this.raf);
    this.renderer.dispose();
    this.root.remove();
  }
}

async function onNavigate(): Promise<void> {
  session?.stop();
  session = null;
  const videoId = videoIdFromLocation();
  if (!videoId) return;
  try {
    signer ??= new Signer(await Lexicon.load(chrome.runtime.getURL("lexicon/")));
  } catch (err) {
    console.warn("[ASLytics]", err);
    return;
  }
  if (videoIdFromLocation() === videoId) session = new Session(videoId, signer);
}

if (typeof chrome !== "undefined" && chrome.runtime?.id) {
  watchCaptionRequests();
  // YouTube is a single-page app: this event fires on every in-site navigation.
  document.addEventListener("yt-navigate-finish", () => void onNavigate());
  void onNavigate();
}
