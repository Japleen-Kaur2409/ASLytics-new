import { defineConfig } from "vite";

// Second build: the background service worker, as an ES module next to content.js.
export default defineConfig({
  publicDir: false,
  build: {
    outDir: "dist",
    emptyOutDir: false,
    lib: { entry: "src/background.ts", formats: ["es"], fileName: () => "background.js" },
  },
});
