/**
 * UTC date labels (YYYY-MM-DD) as half-open [00:00, 24:00) UTC bins.
 *
 * These are calendar bins on the UTC axis only. They are not observing nights
 * and they never convert between time scales: every domain adapter assigns its
 * own entities to UTC dates using its declared scale (see TimeSemantics).
 */

const DAY_MS = 86_400_000;
const DATE_RE = /^(\d{4})-(\d{2})-(\d{2})$/;

export function isUtcDate(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const match = DATE_RE.exec(value);
  if (!match) return false;
  const parsed = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
  return parsed.toISOString().slice(0, 10) === value;
}

export function utcDateToMs(date: string): number {
  const match = DATE_RE.exec(date);
  if (!match) throw new RangeError(`not a UTC date: ${date}`);
  return Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
}

export function msToUtcDate(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

export function addDays(date: string, days: number): string {
  return msToUtcDate(utcDateToMs(date) + days * DAY_MS);
}

/** Dates in the half-open interval [start, stop). */
export function utcDateRange(start: string, stop: string): string[] {
  const out: string[] = [];
  for (let ms = utcDateToMs(start); ms < utcDateToMs(stop); ms += DAY_MS) out.push(msToUtcDate(ms));
  return out;
}

export function daysBetween(start: string, stop: string): number {
  return Math.round((utcDateToMs(stop) - utcDateToMs(start)) / DAY_MS);
}

/** Half-open membership, lexicographic on ISO dates. */
export function inUtcInterval(date: string, start: string, stop: string): boolean {
  return date >= start && date < stop;
}

/**
 * UTC date of an MJD that is already on the UTC scale. Only valid when the
 * producing domain declares its MJD scale as UTC; never apply to TAI values.
 */
export function utcMjdToUtcDate(mjdUtc: number): string {
  return msToUtcDate(Math.floor(mjdUtc - 40587) * DAY_MS);
}
