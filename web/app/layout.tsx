import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SkyPulse",
  description:
    "A cinematic public layer for exploring LSST-associated ANTARES alert-analysis data processed on RSP."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
