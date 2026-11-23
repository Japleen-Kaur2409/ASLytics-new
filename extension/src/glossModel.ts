// Runs the trained English -> ASL gloss model (fine-tuned t5-small, int8 ONNX) in the browser.
// Used by the extension's background worker and by the demo page.

import { env, pipeline } from "@huggingface/transformers";

const PREFIX = "translate English to ASL gloss: "; // must match training/text2gloss/train.py

export type Glosser = (texts: string[]) => Promise<string[]>;

/**
 * @param modelsUrl folder containing `gloss/` (config, tokenizer, onnx/)
 * @param ortUrl    folder containing onnxruntime's ort-wasm-* files
 */
export async function loadGlossModel(modelsUrl: string, ortUrl: string): Promise<Glosser> {
  // Everything is bundled with the extension: never fetch models or runtime code from the network.
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = modelsUrl;
  env.useBrowserCache = false;
  const wasm = env.backends.onnx.wasm!;
  wasm.wasmPaths = ortUrl;
  wasm.numThreads = 1; // threads need cross-origin isolation, which an extension worker does not have

  const translate = await pipeline("text2text-generation", "gloss", { dtype: "q8", device: "wasm" });
  return async (texts) => {
    if (texts.length === 0) return [];
    const out = (await translate(texts.map((t) => PREFIX + t), { max_new_tokens: 64 })) as { generated_text: string }[];
    return out.map((o) => o.generated_text);
  };
}
