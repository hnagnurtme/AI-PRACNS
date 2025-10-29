import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Use a relative base so production assets are referenced relatively.
// This makes `dist/index.html` usable when opened directly (file://)
// or when the site is served from a subpath. Dev server is unaffected.
export default defineConfig( {
    base: './',
    plugins: [ react() ],
    define: {
        global: "window",
    },
} )
