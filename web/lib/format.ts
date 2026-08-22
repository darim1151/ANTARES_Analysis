export function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatMagnitude(value: number) {
  return `${value.toFixed(2)} mag`;
}

export function shortDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC"
  }).format(new Date(`${value}T00:00:00Z`));
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
