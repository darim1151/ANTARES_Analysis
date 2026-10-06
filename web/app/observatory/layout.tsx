import type { Metadata, Viewport } from "next";
import "./observatory.css";

export const metadata: Metadata = {
  title: "Unified Scientific Observatory · First Light",
  description:
    "One scientific workspace over broker-native ANTARES and Fink domains: time, sky, population, native records and provenance against a pinned basis."
};

export const viewport: Viewport = {
  themeColor: "#0b0b0a",
  colorScheme: "dark"
};

export default function ObservatoryLayout({ children }: { children: React.ReactNode }) {
  return children;
}
