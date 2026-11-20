import { defineConfig } from "vitest/config";

// The page calls /api/...; in development Vite passes those calls on to the Python API
// (uvicorn on port 8000), so the browser never needs a cross-origin request.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", rewrite: (path) => path.replace(/^\/api/, "") },
    },
  },
  test: { environment: "jsdom" },
});
