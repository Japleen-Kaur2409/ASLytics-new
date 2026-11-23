import { createReadStream, existsSync } from "node:fs";
import { join } from "node:path";
import { defineConfig, type Plugin } from "vite";

// The dev server refuses to let page code import JavaScript from public/, but
// onnxruntime loads its runtime (public/ort/*.mjs) exactly that way. Serve those
// files as-is in dev. The built extension serves them statically and needs none of this.
const serveOrtRaw: Plugin = {
  name: "serve-ort-raw",
  configureServer(server) {
    server.middlewares.use("/ort", (req, res, next) => {
      const name = (req.url ?? "").split("?")[0].replace(/^\/+/, "");
      const file = join("public/ort", name);
      if (!/^[\w.-]+$/.test(name) || !existsSync(file)) return next();
      res.setHeader("Content-Type", name.endsWith(".wasm") ? "application/wasm" : "text/javascript");
      createReadStream(file).pipe(res);
    });
  },
};

// The content script must be one self-contained file: Chrome content scripts cannot import modules.
export default defineConfig({
  plugins: [serveOrtRaw],
  // `npm run preview` serves dist/ so the built script can be tried on a page without installing it.
  preview: { cors: true },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    lib: { entry: "src/content.ts", formats: ["iife"], name: "ASLytics", fileName: () => "content.js" },
  },
});
