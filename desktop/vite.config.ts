import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [
    react(),
    {
      name: "development-refresh-csp",
      apply: "serve",
      transformIndexHtml(html) {
        return html.replace(
          "script-src 'self'",
          "script-src 'self' 'unsafe-inline'",
        );
      },
    },
  ],
  base: "./",
  server: { host: "127.0.0.1", strictPort: true, port: 5178 },
});
