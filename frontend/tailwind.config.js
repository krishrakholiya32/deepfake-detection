/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        df: {
          bg:      '#0a0c10',
          surface: '#11151c',
          border:  '#222a35',
          real:    '#00ff99',
          fake:    '#ff4b4b',
          accent:  '#4a9eff',
          muted:   '#8ab4c8',
        },
      },
    },
  },
  plugins: [],
}
