/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx,html}",
  ],
  theme: {
    extend: {
      colors: {
        // Deep Space Background
        cosmic: {
          black: '#0a0a0f',
          navy: '#0f172a',
          dark: '#1e1b4b',
          deeper: '#070710',
        },
        // Nebula Accent Colors
        nebula: {
          purple: '#7c3aed',
          pink: '#db2777',
          cyan: '#06b6d4',
          blue: '#3b82f6',
          violet: '#8b5cf6',
        },
        // Star Accent Colors
        star: {
          gold: '#fbbf24',
          white: '#f8fafc',
          silver: '#94a3b8',
        },
      },
      backgroundImage: {
        // Nebula Gradients
        'nebula-gradient': 'linear-gradient(135deg, #7c3aed 0%, #db2777 50%, #06b6d4 100%)',
        'cosmic-gradient': 'linear-gradient(to bottom right, #0a0a0f 0%, #0f172a 50%, #1e1b4b 100%)',
        'space-glow': 'radial-gradient(ellipse at center, rgba(124, 58, 237, 0.15) 0%, transparent 70%)',
      },
      boxShadow: {
        'nebula': '0 0 30px rgba(124, 58, 237, 0.3)',
        'nebula-lg': '0 0 50px rgba(124, 58, 237, 0.4)',
        'cyan-glow': '0 0 20px rgba(6, 182, 212, 0.4)',
        'pink-glow': '0 0 20px rgba(219, 39, 119, 0.4)',
        'star': '0 0 10px rgba(251, 191, 36, 0.5)',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'twinkle': 'twinkle 3s ease-in-out infinite',
        'nebula-flow': 'nebula-flow 8s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1', boxShadow: '0 0 20px rgba(124, 58, 237, 0.4)' },
          '50%': { opacity: '0.8', boxShadow: '0 0 40px rgba(124, 58, 237, 0.6)' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        'twinkle': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
        'nebula-flow': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
};
