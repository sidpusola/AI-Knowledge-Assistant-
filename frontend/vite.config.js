import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Fail loudly if 5173 is taken, instead of silently moving to 5174.
    // The backend's CORS list only allows 5173, so a silent port change
    // breaks every request with an opaque "OPTIONS ... 400 Bad Request"
    // preflight failure that looks like the backend is down. Better to
    // refuse to start and say why.
    strictPort: true,
  },
})
