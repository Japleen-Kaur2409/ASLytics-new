// Extension background worker: owns the gloss model so it is loaded once, not per tab.

import { loadGlossModel, type Glosser } from "./glossModel";

let glosser: Promise<Glosser> | null = null;

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "aslytics-gloss" || !Array.isArray(message.texts)) return false;
  glosser ??= loadGlossModel(chrome.runtime.getURL("models/"), chrome.runtime.getURL("ort/"));
  glosser
    .then((run) => run(message.texts as string[]))
    .then((glosses) => sendResponse({ glosses }))
    .catch((err) => {
      glosser = null; // allow a retry on the next request
      sendResponse({ error: String(err) });
    });
  return true; // keep the channel open for the async reply
});
