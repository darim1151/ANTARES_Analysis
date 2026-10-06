const intFormat = new Intl.NumberFormat("en-US");

export function fmtInt(value: number): string {
  return intFormat.format(value);
}

/** Compact significant-figure formatting for scientific readouts. */
export function fmtNum(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e5 || abs < 1e-3) return value.toExponential(Math.max(0, digits - 1)).replace("e+", "e");
  if (Number.isInteger(value) && abs < 1e5) return fmtInt(value);
  return Number(value.toPrecision(digits)).toString();
}

export function fmtFixed(value: number | null | undefined, decimals: number): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(decimals);
}

export function fmtSigned(value: number, decimals: number): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(decimals)}`;
}

export function fmtPercent(fraction: number, decimals = 0): string {
  if (!Number.isFinite(fraction)) return "—";
  return `${(fraction * 100).toFixed(decimals)}%`;
}

export function fmtBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} kB`;
  return `${bytes} B`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "Feb 25" for a YYYY-MM-DD UTC date label. */
export function shortDate(date: string): string {
  return `${MONTHS[Number(date.slice(5, 7)) - 1]} ${Number(date.slice(8, 10))}`;
}

/** Inclusive display of a half-open [start, stop) UTC date interval. */
export function dateSpan(start: string, stopExclusive: string, lastInclusive: string): string {
  return start === lastInclusive ? `${start} UTC` : `${start} → ${lastInclusive} UTC`;
}

export function shortHash(hex: string | null | undefined, n = 10): string {
  return hex ? hex.slice(0, n) : "—";
}

/** RA in degrees to sexagesimal hours. */
export function raHms(ra: number): string {
  const h = ra / 15;
  const hh = Math.floor(h);
  const m = (h - hh) * 60;
  const mm = Math.floor(m);
  const ss = (m - mm) * 60;
  return `${String(hh).padStart(2, "0")}ʰ${String(mm).padStart(2, "0")}ᵐ${ss.toFixed(1).padStart(4, "0")}ˢ`;
}

export function decDms(dec: number): string {
  const sign = dec < 0 ? "−" : "+";
  const a = Math.abs(dec);
  const d = Math.floor(a);
  const m = (a - d) * 60;
  const mm = Math.floor(m);
  const ss = (m - mm) * 60;
  return `${sign}${String(d).padStart(2, "0")}°${String(mm).padStart(2, "0")}′${ss.toFixed(0).padStart(2, "0")}″`;
}

const SUPERSCRIPT: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };

/** Tick label for log axes: exact decades as 10ⁿ outside [0.01, 1000], otherwise plain. */
export function fmtLogTick(value: number): string {
  const exponent = Math.log10(value);
  const isDecade = Math.abs(exponent - Math.round(exponent)) < 1e-9;
  if (isDecade && (value >= 1e4 || value < 0.01)) {
    return `10${String(Math.round(exponent)).replace(/./g, (c) => SUPERSCRIPT[c] ?? c)}`;
  }
  return fmtNum(value, 3);
}
