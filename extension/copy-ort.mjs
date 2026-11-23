// Copies onnxruntime's WebAssembly runtime into public/ort so the extension never loads code from a CDN.
import { cpSync, mkdirSync, readdirSync } from "node:fs";

const from = "node_modules/onnxruntime-web/dist";
mkdirSync("public/ort", { recursive: true });
for (const file of readdirSync(from)) {
  if (file.startsWith("ort-wasm")) cpSync(`${from}/${file}`, `public/ort/${file}`);
}
