import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 127.0.0.1, not the default. The API is on 127.0.0.1:8000, and a
    // browser treats localhost and 127.0.0.1 as different SITES - so a
    // page served from localhost:5173 makes cross-site fetches, and the
    // SameSite=Lax session cookie is withheld from every one of them.
    // Sign-in appeared to work (that is a top-level navigation, which
    // Lax does allow) and then every /auth/me came back signed-out.
    // Binding here also pins IPv4: left to itself Vite bound [::1] only
    // on this machine, so 127.0.0.1:5173 refused outright.
    host: "127.0.0.1",
    port: 5173,
    // Everything under /api is forwarded to the backend, so the browser
    // only ever talks to one origin. That removes CORS from development
    // entirely, and - the reason it is here - makes the session cookie
    // an ordinary same-origin cookie instead of a cross-site one that
    // SameSite=Lax refuses to send.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
    // Fail loudly if 5173 is taken, instead of silently moving to 5174.
    // The backend's CORS list only allows 5173, so a silent port change
    // breaks every request with an opaque "OPTIONS ... 400 Bad Request"
    // preflight failure that looks like the backend is down. Better to
    // refuse to start and say why.
    strictPort: true,
  },
})
