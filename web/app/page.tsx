import SkyPulseExperience from "@/components/SkyPulseExperience";
import densityData from "../public/data/density_tiles.json";
import lightcurveData from "../public/data/lightcurve_samples.json";
import manifestData from "../public/data/public_manifest.json";
import summaryData from "../public/data/public_summary.json";
import pointsData from "../public/data/sky_points.json";
import candidatesData from "../public/data/top_candidates.json";
import type {
  DensityTilesFile,
  LightcurveSamplesFile,
  PublicManifest,
  PublicSummary,
  SkyPointsFile,
  TopCandidatesFile
} from "@/types/skypulse";

export default function Page() {
  return (
    <SkyPulseExperience
      manifest={manifestData as PublicManifest}
      summary={summaryData as PublicSummary}
      pointsFile={pointsData as SkyPointsFile}
      densityFile={densityData as DensityTilesFile}
      candidatesFile={candidatesData as TopCandidatesFile}
      lightcurveFile={lightcurveData as LightcurveSamplesFile}
    />
  );
}
