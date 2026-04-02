import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./data/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        muted: "var(--muted)",
        line: "var(--line)",
        panel: "var(--panel)",
        panelStrong: "var(--panel-strong)",
        accent: "var(--accent)",
        accentSoft: "var(--accent-soft)",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(140, 204, 255, 0.08), 0 24px 80px rgba(7, 16, 28, 0.45)",
      },
      borderRadius: {
        "2xl": "1.5rem",
        "3xl": "2rem"
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(148,163,184,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.08) 1px, transparent 1px)"
      },
      fontFamily: {
        sans: ["var(--font-manrope)"],
        mono: ["var(--font-mono)"],
        display: ["var(--font-display)"],
      }
    }
  },
  plugins: []
};

export default config;
