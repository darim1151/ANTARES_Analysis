export const sourceIdentity =
  "ANTARES alert-analysis data processed on Rubin Science Platform storage";

export const forbiddenClaims = [
  "Rubin live feed",
  "official Rubin result",
  "direct Rubin catalog query",
  "real-time LSST stream",
  "classified transient"
];

export const glossary = [
  {
    term: "Locus",
    publicTerm: "Sky object",
    expanded: "One ANTARES object or sky position, often connected to many alerts.",
    avoid: "Do not call it a confirmed astrophysical class.",
    tooltip: "A sky object tracked by ANTARES at one position."
  },
  {
    term: "Alert",
    publicTerm: "Change notice",
    expanded: "A broker notice that something about a sky object changed or was detected.",
    avoid: "Do not imply every alert is a discovery.",
    tooltip: "A notice that the sky changed at this position."
  },
  {
    term: "Lightcurve",
    publicTerm: "Brightness over time",
    expanded: "A sequence of magnitude measurements for one sky object.",
    avoid: "Do not describe it as a photo or spectrum.",
    tooltip: "How bright the object appeared across repeated observations."
  },
  {
    term: "Magnitude",
    publicTerm: "Brightness",
    expanded: "Astronomical brightness where smaller numbers are brighter.",
    avoid: "Do not map larger magnitude to brighter UI emphasis.",
    tooltip: "Lower magnitude means brighter."
  },
  {
    term: "MJD",
    publicTerm: "Astronomer's time stamp",
    expanded: "Modified Julian Date, the time coordinate used by ANTARES alert data.",
    avoid: "Do not expose raw MJD without a calendar date nearby.",
    tooltip: "A standard astronomy time stamp."
  },
  {
    term: "Cumulative history",
    publicTerm: "Previously processed nights",
    expanded: "Saved nightly ANTARES/RSP outputs strictly before the comparison night.",
    avoid: "Do not include the selected night in its own history.",
    tooltip: "The saved sky memory from earlier processed nights."
  },
  {
    term: "LSST-associated ANTARES data",
    publicTerm: "Rubin/LSST-associated objects indexed through ANTARES",
    expanded: "ANTARES loci with LSST DIA-object or Solar-System identifiers.",
    avoid: "Do not say direct Rubin Butler, TAP, or production query.",
    tooltip: "Objects associated with Rubin/LSST identifiers in ANTARES."
  }
];
