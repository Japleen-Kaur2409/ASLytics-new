import { defineConfig } from "vite";

// The content script must be one self-contained file: Chrome content scripts cannot import modules.
export default defineConfig({
  // `npm run preview` serves dist/ so the built script can be tried on a page without installing it.
  preview: { cors: true },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    lib: { entry: "src/content.ts", formats: ["iife"], name: "ASLytics", fileName: () => "content.js" },
  },
});
