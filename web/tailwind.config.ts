import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        void: "#030814",
        field: "#071426",
        cyan: "#31D9FF",
        violet: "#8F5BFF",
        amber: "#FFB84D",
        star: "#F4F7FB",
        glass: "rgba(9, 21, 42, 0.68)"
      },
      boxShadow: {
        glowCyan: "0 0 42px rgba(49, 217, 255, 0.24)",
        glowAmber: "0 0 34px rgba(255, 184, 77, 0.22)"
      },
      fontFamily: {
        display: ["Iowan Old Style", "Palatino", "Georgia", "serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["SFMono-Regular", "Menlo", "Consolas", "monospace"]
      }
    }
  },
  plugins: []
};

export default config;
