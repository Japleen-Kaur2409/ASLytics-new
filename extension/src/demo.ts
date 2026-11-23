// Development page: type text, watch it signed on a loop. Not part of the extension build.

import { FPS } from "./landmarks";
import { Lexicon } from "./lexicon";
import { Renderer } from "./renderer";
import { Signer, type Signed } from "./signer";

const input = document.querySelector<HTMLInputElement>("#text")!;
const info = document.querySelector<HTMLDivElement>("#info")!;
const renderer = new Renderer(360, 420);
document.querySelector("#stage")!.append(renderer.canvas);

const lexicon = await Lexicon.load("/lexicon/");
const signer = new Signer(lexicon);
let current: Signed = signer.sign(input.value);
let startedAt = performance.now();

function update(): void {
  current = signer.sign(input.value);
  startedAt = performance.now();
  const seconds = current.sequence ? (current.sequence.n / FPS).toFixed(1) : "0";
  info.textContent =
    `${lexicon.glosses.length} signs in the dictionary. Signing: ${current.glosses.join(" ") || "(nothing)"} (${seconds}s)` +
    (current.missing.length ? `. Not in the dictionary: ${current.missing.join(", ")}` : "");
}
input.addEventListener("change", update);
update();

// Lets automated checks freeze the animation at an exact frame.
const frozen = new URLSearchParams(location.search).get("frame");

function tick(): void {
  requestAnimationFrame(tick);
  if (!current.sequence) return;
  const seq = current.sequence;
  const loop = seq.n / FPS + 0.8; // pause briefly on the last frame, then repeat
  const elapsed = ((performance.now() - startedAt) / 1000) % loop;
  renderer.draw(seq, frozen !== null ? Number(frozen) : Math.min(seq.n - 1, elapsed * FPS));
}
tick();
