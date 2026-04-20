"""
Human-readable summary statistics for the loci + alerts of one MJD range.

This module exists purely to print things. It computes nothing the
analysis depends on. We separate it from `validation.py` because the goal
here is "give me a quick eyeball check that the data looks sane", whereas
validation is "assert that integrity rules hold".
"""

import pandas as pd

from .lightcurves import LC_FID, LC_MAG, FILTER_INFO


def print_range_summary(label, df_loci, df_alerts):
    """
    Print a one-page-ish summary of a single MJD range.

    Each branch checks whether the column exists before reading it - the
    ANTARES `properties` schema is not fixed across loci, so a column we
    rely on may simply be absent in some samples. Skipping silently keeps
    the summary clean rather than littering it with KeyErrors.
    """
    bar = "=" * 55
    if df_loci.empty:
        # No data is a legitimate state (e.g. a deliberately invalid
        # range). Print a placeholder so the output is still aligned.
        print(f"\n{bar}\n  {label}: no data loaded\n{bar}")
        return

    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)

    # Sky coverage - useful for spotting queries that accidentally
    # restricted to a small declination band.
    ra_ok = df_loci['ra'].dropna()
    dec_ok = df_loci['dec'].dropna()
    print(f"  Loci loaded        : {len(df_loci):>6,}")
    if len(ra_ok):
        print(f"  RA  range          : {ra_ok.min():.2f} deg - {ra_ok.max():.2f} deg")
        print(f"  Dec range          : {dec_ok.min():.2f} deg - {dec_ok.max():.2f} deg")

    # Locus-level summary properties. These come from the ANTARES
    # `properties` block, which is why we tolerate missing columns.
    for col, name in [
        ('newest_alert_observation_time', 'Newest obs (MJD)'),
        ('brightest_alert_magnitude',     'Brightest mag   '),
        ('num_mag_values',                'Obs count       '),
    ]:
        if col in df_loci.columns:
            s = df_loci[col].dropna()
            if len(s):
                print(f"  {name}: {s.min():.2f} - {s.max():.2f}  (median {s.median():.2f})")

    # Alert-level summary - only meaningful when lightcurves were loaded.
    if not df_alerts.empty:
        print(f"  Alert rows         : {len(df_alerts):>6,}")
        if LC_MAG in df_alerts.columns:
            mag = df_alerts[LC_MAG].dropna()
            print(f"  Alert mag range    : {mag.min():.2f} - {mag.max():.2f}  (median {mag.median():.2f})")
        if LC_FID in df_alerts.columns:
            # Per-filter counts let the user see at a glance whether one
            # band dominates the sample (it often does for ZTF: r >> g >> i).
            counts = df_alerts[LC_FID].value_counts().sort_index()
            band_str = "  ".join(
                f"{FILTER_INFO.get(int(fid), (fid, '?'))[0]}:{n}"
                for fid, n in counts.items() if not pd.isna(fid)
            )
            print(f"  Alerts by filter   : {band_str}")
    else:
        print(f"  Alert rows         : (lightcurves not loaded)")

    # Total column count is a quick "did the schema change?" indicator
    # without dumping the whole list every time.
    print(f"  Locus columns      : {len(df_loci.columns)} total")


def print_all_summaries(label1, df1, df1_alerts, label2, df2, df2_alerts):
    """Convenience wrapper that prints the summary header + both ranges."""
    print("DATA SUMMARY")
    print_range_summary(label1, df1, df1_alerts)
    print_range_summary(label2, df2, df2_alerts)
