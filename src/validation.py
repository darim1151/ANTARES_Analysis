"""
Data integrity / LSST-origin validation suite.

Every check writes one of three tags:
    PASS - assertion holds
    WARN - non-critical (data probably still usable)
    FAIL - integrity issue that could corrupt downstream plots/conclusions

The whole suite is self-contained: it never raises. The final verdict
line lists every failure so a script consuming this output can `grep
'^  FAIL'` for a machine-readable summary.

Tests rationale:
    1. Did we get any data at all?          (FAIL on empty)
    2. Are all returned MJDs inside the queried window? (FAIL on out-of-band)
    3. Are RA/Dec physically valid?         (WARN - bad coords are rare)
    4. Are observations in the LSST era?    (WARN - some pre-LSST is OK)
    5. Are alert mags in the analysis window? (FAIL if <50%)
    6. Are locus_ids unique within a range? (WARN - dupes are rare)
    7. Is the random sampling actually random? (FAIL if huge overlap on
       disjoint MJD ranges - that signals random_score ordering broke)
    8. Sanity-check the MJDmin <= MJDmax rule (FAIL if a "valid" range
       silently slipped through with min > max)
"""

import pandas as pd

from .config import LSST_START_MJD
from .lightcurves import LC_FID, LC_MAG


# Tolerance for floating-point MJD comparisons. ANTARES rounds to the
# millisecond (~1e-8 day) but pandas/parquet round-trips can introduce
# 1e-9 noise. 0.02 day (~30 minutes) is loose enough to never flag a
# float artefact and tight enough to catch true off-window data.
MJD_TOL = 0.02
MJD_COL = 'newest_alert_observation_time'


def _run_check(name, ok, msg='', warn=False, failures=None, warnings=None):
    """
    Print one PASS/WARN/FAIL line and append failures to the right list.

    Returns the boolean `ok` so the caller can chain conditions.
    """
    tag = 'PASS' if ok else ('WARN' if warn else 'FAIL')
    line = f'  [{tag}]  {name}'
    if not ok and msg:
        line += f'\n          -> {msg}'
    print(line)
    if not ok:
        # Route to the right bucket so the final verdict can distinguish
        # "everything passed with some warnings" from "actual failures".
        (warnings if warn else failures).append(name)
    return ok


def run_validation_suite(df1, df2, df1_alerts, df2_alerts,
                         range1_valid, range2_valid,
                         label1, label2,
                         mjd1_min, mjd1_max, mjd2_min, mjd2_max,
                         mag_min=15.0, mag_max=25.0):
    """
    Run all eight validation tests and print a final verdict.

    Returns `(failures, warnings)` - lists of failed/warned check names so
    a script could decide programmatically whether to gate on the result.
    """
    failures = []
    warnings = []

    # Bind the failure / warning lists into the per-check helper to avoid
    # passing them on every call. functools.partial would also work but
    # this keeps the call sites short and self-explanatory.
    def check(name, ok, msg='', warn=False):
        return _run_check(name, ok, msg=msg, warn=warn,
                          failures=failures, warnings=warnings)

    print('=' * 62)
    print('  ANTARES / LSST DATA VALIDATION')
    print('=' * 62)

    # ------------------------------------------------------------------
    # 1. Did we get any data?
    # ------------------------------------------------------------------
    # Empty results from a VALID query indicate either an outage, a
    # network problem, or an MJD window that genuinely contains nothing.
    print('\n[1] Data retrieval')
    for df, valid, lbl in [(df1, range1_valid, label1), (df2, range2_valid, label2)]:
        if not valid:
            print(f'  [SKIP]  {lbl} - range intentionally invalid, skipped by design.')
            continue
        check(f'{lbl}: at least 1 locus retrieved  (got {len(df)})',
              not df.empty,
              'Query returned 0 rows - check ANTARES connectivity and MJD range.')

    # ------------------------------------------------------------------
    # 2. MJD compliance.
    # ------------------------------------------------------------------
    # A returned MJD outside the queried window means our query body is
    # wrong or ANTARES is mis-indexing - either way, the comparison is
    # invalid because the two ranges may no longer be disjoint.
    print('\n[2] MJD compliance  (newest_alert_observation_time within queried window)')
    for df, valid, lbl, lo, hi in [
        (df1, range1_valid, label1, mjd1_min, mjd1_max),
        (df2, range2_valid, label2, mjd2_min, mjd2_max),
    ]:
        if not valid or df.empty or MJD_COL not in df.columns:
            continue
        mjds = df[MJD_COL].dropna()
        below = (mjds < lo - MJD_TOL).sum()
        above = (mjds > hi + MJD_TOL).sum()
        check(
            f'{lbl}: all {len(mjds)} MJDs within [{lo:.1f}, {hi:.1f}]',
            below == 0 and above == 0,
            f'{below} below window, {above} above.  '
            f'Actual: {mjds.min():.3f}-{mjds.max():.3f}',
        )
        print(f'          MJD actual range: {mjds.min():.3f} - {mjds.max():.3f}')

    # ------------------------------------------------------------------
    # 3. Coordinate validity (WARN - RA/Dec are very rarely off).
    # ------------------------------------------------------------------
    print('\n[3] Coordinate validity  (0 <= RA < 360 deg,  -90 <= Dec <= 90 deg)')
    for df, valid, lbl in [(df1, range1_valid, label1), (df2, range2_valid, label2)]:
        if not valid or df.empty:
            continue
        ra_bad = ~df['ra'].between(0, 360, inclusive='left')
        dec_bad = ~df['dec'].between(-90, 90)
        check(f'{lbl}: RA  valid  ({ra_bad.sum()} bad)', ra_bad.sum() == 0,
              f'RA range: {df["ra"].min():.3f}-{df["ra"].max():.3f}', warn=True)
        check(f'{lbl}: Dec valid  ({dec_bad.sum()} bad)', dec_bad.sum() == 0,
              f'Dec range: {df["dec"].min():.3f}-{df["dec"].max():.3f}', warn=True)

    # ------------------------------------------------------------------
    # 4. LSST-era verification (WARN).
    # ------------------------------------------------------------------
    # Some pre-LSST objects can legitimately appear because ANTARES also
    # reprocesses ZTF data. We only WARN if more than 5% predate the
    # nominal LSST start - that hints the query mis-targeted ZTF
    # archival objects rather than fresh LSST alerts.
    print(f'\n[4] LSST-era verification  (obs time >= MJD {LSST_START_MJD})')
    for df, valid, lbl in [(df1, range1_valid, label1), (df2, range2_valid, label2)]:
        if not valid or df.empty or MJD_COL not in df.columns:
            continue
        mjds = df[MJD_COL].dropna()
        pre = (mjds < LSST_START_MJD).sum()
        frac = pre / len(mjds) if len(mjds) else 0.0
        check(
            f'{lbl}: >=95% of loci in LSST era  ({pre} pre-LSST of {len(mjds)})',
            frac < 0.05,
            f'{frac:.1%} of loci pre-date MJD {LSST_START_MJD} (may be ZTF objects).',
            warn=True,
        )

    # Informational: print any LSST-/Rubin-specific properties we found
    # so the user knows whether genuine LSST schema is in play.
    lsst_cols = sorted(set(
        c for c in list(df1.columns) + list(df2.columns)
        if 'lsst' in c.lower() or 'rubin' in c.lower()
    ))
    if lsst_cols:
        print(f'  [INFO]  LSST-specific property columns: {lsst_cols}')
    else:
        print('  [INFO]  No lsst_*/rubin_* columns found; data is ZTF-origin '
              'processed during the LSST era.')
    combined = pd.concat([df1, df2], ignore_index=True)
    if not combined.empty and 'tags' in combined.columns:
        ddf = combined['tags'].str.contains('in_LSSTDDF', na=False).sum()
        if ddf:
            print(f'  [INFO]  {ddf} loci carry the "in_LSSTDDF" tag.')

    # ------------------------------------------------------------------
    # 5. Magnitude sanity.
    # ------------------------------------------------------------------
    # If <50% of alerts fall in [15, 25] mag, the magnitude histograms
    # will be largely empty - either lightcurves failed to load properly
    # or the underlying data is dominated by saturated/noise alerts.
    print(f'\n[5] Magnitude sanity  (some alerts in {mag_min}-{mag_max} mag window)')
    for df_a, valid, lbl in [
        (df1_alerts, range1_valid, label1),
        (df2_alerts, range2_valid, label2),
    ]:
        if not valid:
            continue
        if df_a.empty or LC_MAG not in df_a.columns:
            check(f'{lbl}: alert-level magnitudes available', False,
                  'No alert data (LOAD_LIGHTCURVES=False or fetch failed).', warn=True)
            continue
        mags = df_a[LC_MAG].dropna()
        in_w = mags.between(mag_min, mag_max).sum()
        frac = in_w / len(mags) if len(mags) else 0.0
        check(
            f'{lbl}: >=50% of alerts in [{mag_min}, {mag_max}] mag  ({frac:.1%})',
            frac >= 0.5,
            f'Only {in_w}/{len(mags)} alerts in window. '
            f'Full: {mags.min():.2f}-{mags.max():.2f}',
        )
        print(f'          mag range: {mags.min():.2f}-{mags.max():.2f},  '
              f'{in_w:,}/{len(mags):,} in window')

    # ------------------------------------------------------------------
    # 6. Unique locus IDs within each range.
    # ------------------------------------------------------------------
    # Duplicates within a single query are very rare (would indicate an
    # ANTARES indexing issue). We WARN rather than FAIL because a few
    # dupes don't materially distort plots.
    print('\n[6] Unique locus IDs within each range  (no duplicate rows)')
    for df, valid, lbl in [(df1, range1_valid, label1), (df2, range2_valid, label2)]:
        if not valid or df.empty or 'locus_id' not in df.columns:
            continue
        dup = len(df) - df['locus_id'].nunique()
        check(f'{lbl}: no duplicate locus_ids  ({dup} duplicates)', dup == 0,
              f'{dup} duplicate rows detected.', warn=True)

    # ------------------------------------------------------------------
    # 7. Cross-range overlap.
    # ------------------------------------------------------------------
    # CRITICAL test for randomisation quality: when MJD windows are
    # disjoint, the two samples should share essentially no loci. If
    # they share a lot, random_score silently fell back to newest-first
    # ordering and BOTH queries returned the same recently-active
    # objects - making the entire comparison trivially identical.
    print('\n[7] Range overlap  (checks randomisation quality)')
    ranges_disjoint = (mjd1_min >= mjd2_max) or (mjd2_min >= mjd1_max)
    if not df1.empty and not df2.empty and 'locus_id' in df1.columns:
        ids1, ids2 = set(df1['locus_id']), set(df2['locus_id'])
        overlap = ids1 & ids2
        pct = len(overlap) / min(len(ids1), len(ids2)) * 100
        print(f'  [INFO]  {len(overlap):,} loci appear in both ranges '
              f'({pct:.1f}% of the smaller range).')

        if ranges_disjoint:
            # WARN at 5-20%, FAIL above 20%. The thresholds are empirical:
            # disjoint MJD windows + working randomisation typically yield
            # <1% overlap from incidental hash collisions, never 5%+.
            check(
                f'Overlap < 5% given non-overlapping MJD windows  (got {pct:.1f}%)',
                pct < 5.0,
                f'High overlap ({pct:.1f}%) despite non-overlapping MJD ranges.  '
                'This strongly suggests random_score is not supported by the '
                'ANTARES ES cluster and both queries returned the same '
                '"newest-first" objects.',
                warn=(pct < 20),
            )
        else:
            print(f'  [INFO]  MJD windows overlap so shared loci are expected.')
            if mjd1_min >= mjd2_min and mjd1_max <= mjd2_max:
                print('          Range 1 is fully contained within Range 2.  '
                      'Consider setting MJD2_MAX = MJD1_MIN for a non-overlapping comparison.')
    else:
        print('  [INFO]  Cannot compute overlap (one or both DataFrames empty).')

    # ------------------------------------------------------------------
    # 8. The "core project rule" guard.
    # ------------------------------------------------------------------
    # If RANGE_VALID is True despite MJDmin > MJDmax, our config-time
    # validation lied to us - that's a code bug, not a data issue, hence
    # FAIL.
    print('\n[8] MJDmin > MJDmax guard  (core project rule)')
    for lbl, lo, hi, valid in [
        (label1, mjd1_min, mjd1_max, range1_valid),
        (label2, mjd2_min, mjd2_max, range2_valid),
    ]:
        if lo > hi:
            check(f'{lbl}: correctly skipped (MJDmin {lo} > MJDmax {hi})',
                  not valid, 'RANGE_VALID flag is True despite invalid MJD order!')
        else:
            check(f'{lbl}: MJDmin <= MJDmax  ({lo} <= {hi})', True)

    # ------------------------------------------------------------------
    # Final verdict.
    # ------------------------------------------------------------------
    print('\n' + '=' * 62)
    if failures:
        print(f'  RESULT: {len(failures)} FAILURE(s),  {len(warnings)} warning(s)')
        for f in failures:
            print(f'    FAIL  {f}')
    else:
        w = f'  ({len(warnings)} warning(s))' if warnings else ''
        print(f'  RESULT: All critical tests PASSED.{w}')
    print('=' * 62)

    return failures, warnings
