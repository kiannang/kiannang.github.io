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
      fontFamily: {
        // Lucida Grande was the Mac OS X system font in 2001.
        sans: ['"Lucida Grande"', '"Lucida Sans Unicode"', '"Lucida Sans"', "Geneva", "Verdana", "sans-serif"],
        // Stand-in for Apple Garamond, the "Think different." era headline face.
        display: ["var(--font-garamond)", "Garamond", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};

export default config;
