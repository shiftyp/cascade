/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'cascade-blue': '#0066CC',
        'cascade-dark': '#1a1a2e',
        'cascade-accent': '#00D4FF',
        'warning-red': '#FF4444',
        'warning-yellow': '#FFD700',
        'success-green': '#00C851',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}