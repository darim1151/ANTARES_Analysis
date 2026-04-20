"""
All matplotlib figure functions used by the notebook.

Each function takes the already-fetched DataFrames + range labels and
draws ONE complete figure. They are written to be called from a notebook
cell (they call `plt.show()` at the end), but each one returns its
`Figure` so an external script could save the figures to disk if needed.

Visual conventions kept consistent across the three plots:
  - Range 1 ("Last Night")    -> cyan / blue / solid lines
  - Range 2 ("LSST History")  -> orange / dashed lines
  - Filter colours: g=green, r=red, i=orange (set in lightcurves.FILTER_INFO)
  - Dark background (assumed set globally by the notebook via plt.style.use)
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from .lightcurves import LC_FID, LC_MAG, FILTER_INFO


# ---------------------------------------------------------------------------
# Common per-range visual styling.
# ---------------------------------------------------------------------------
# Pulled out as module constants so a future user changing one plot's
# colour can find every other place the same colour is reused.
SKY_STYLE = {
    # (color, marker, alpha, point_size)
    1: ('#00bcd4', 'o', 0.75, 10),   # cyan circles - foregrounded
    2: ('#ff7043', '^', 0.30,  6),   # orange triangles - backgrounded
}

DENSITY_CMAP = {
    1: 'Blues',
    2: 'Oranges',
}

# Solid for Range 1, dashed for Range 2, with slightly different line weights
# so the solid line is visually dominant when ranges overlap.
MAG_LS = {1: '-',  2: '--'}
MAG_LW = {1: 2.2, 2: 1.6}

FALLBACK_MAG_COLORS = {1: '#00bcd4', 2: '#ff7043'}


def plot_sky_aitoff(df1, df2, range1_valid, range2_valid,
                    label1, label2, mjd1_min, mjd2_max):
    """
    Plot 1: All-sky distribution on an Aitoff (equal-area) projection.

    Why Aitoff?
        It is the standard equal-area whole-sky projection used in
        astronomy. It minimises area distortion (so eyeballed densities
        are honest) at the cost of mild shape distortion near the edges -
        exactly the right tradeoff for surveying where objects are.

    Why we wrap RA into [-pi, pi]?
        matplotlib's Aitoff projection expects radians in [-pi, pi] with
        0 at the centre and RA increasing leftwards. ANTARES gives RA in
        degrees in [0, 360). Subtracting 2*pi from values >pi flips them
        into the negative half-axis the projection expects.
    """
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='aitoff')

    plotted = False
    for rnum, (df, valid, label) in enumerate(
        [(df1, range1_valid, label1), (df2, range2_valid, label2)], start=1
    ):
        color, marker, alpha, sz = SKY_STYLE[rnum]
        if not valid or df.empty:
            continue
        coords = df[['ra', 'dec']].dropna()
        if coords.empty:
            continue

        # Convert to radians and shift the RA into [-pi, pi] for Aitoff.
        ra_rad = np.radians(coords['ra'].values)
        ra_rad = np.where(ra_rad > np.pi, ra_rad - 2 * np.pi, ra_rad)
        dec_rad = np.radians(coords['dec'].values)

        ax.scatter(
            ra_rad, dec_rad,
            s=sz, c=color, marker=marker, alpha=alpha, edgecolors='none',
            label=f'{label}  ({len(coords):,} loci)',
            # zorder bumps Range 1 on top of Range 2 so the foreground
            # snapshot isn't hidden behind the historical cloud.
            zorder=rnum,
        )
        plotted = True

    if not plotted:
        # Centre an explicit notice so a viewer who only sees the figure
        # (not the printed log) understands why the canvas is blank.
        ax.text(0, 0,
                'No valid data to display\n(MJD ranges invalid or queries returned 0 loci)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='#cccccc')

    # Build a date-stamped subtitle from the MJD bounds. This makes a
    # saved PNG self-documenting - future-you can read the dates without
    # re-opening the notebook.
    t1_str = Time(mjd1_min, format='mjd').iso[:10] if range1_valid else '--'
    t2_str = Time(mjd2_max, format='mjd').iso[:10] if range2_valid else '--'
    ax.set_title(
        f'Sky Distribution - ANTARES / LSST\n'
        f'{label1}: {t1_str}  |  {label2}: through {t2_str}',
        fontsize=13, fontweight='bold', pad=22,
    )
    ax.grid(True, alpha=0.15, color='white')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.25,
              facecolor='#1e1e1e', edgecolor='#555555')
    # Standard equatorial Aitoff RA labels (RA decreasing left-to-right).
    ax.set_xticklabels(
        ['14h', '16h', '18h', '20h', '22h', '0h', '2h', '4h', '6h', '8h', '10h'],
        alpha=0.55, fontsize=9,
    )
    plt.tight_layout()
    plt.show()
    return fig


def plot_sky_density(df1, df2, range1_valid, range2_valid,
                     label1, label2, n_bins=80):
    """
    Plot 2: Side-by-side RA/Dec 2-D histograms.

    Aitoff (Plot 1) is great for showing WHERE objects are; a 2-D
    histogram is better at showing HOW MANY are in each pixel. Using a
    log colour scale (LogNorm) is essential because survey footprints
    are extremely non-uniform - a few high-density tiles otherwise wash
    out everything else into a single dark colour.

    Note we do NOT plot focal-plane / camera coordinates here:
    `ztf_xpos`, `ztf_ypos`, `ztf_ccdid` etc. are not exposed via the
    ANTARES lightcurve schema.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for rnum, (df, valid, lbl) in enumerate(
        [(df1, range1_valid, label1), (df2, range2_valid, label2)], start=1
    ):
        ax = axes[rnum - 1]
        cmap = DENSITY_CMAP[rnum]

        if not valid or df.empty or 'ra' not in df.columns:
            # Keep the axes consistent with the populated case so the
            # composite figure still looks intentional when one range is
            # missing.
            ax.text(0.5, 0.5, f'No data for  {lbl}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='#aaaaaa')
            ax.set_title(lbl, fontsize=12, fontweight='bold')
            ax.set_facecolor('#111111')
            continue

        coords = df[['ra', 'dec']].dropna()
        # vmin=1 in LogNorm avoids log(0) for empty bins; pixels with
        # zero count are simply not drawn.
        h, xe, ye, img = ax.hist2d(
            coords['ra'].values, coords['dec'].values,
            bins=n_bins,
            cmap=cmap,
            norm=mcolors.LogNorm(vmin=1),
        )
        plt.colorbar(img, ax=ax, label='Locus count  (log scale)',
                     pad=0.02, fraction=0.04)
        ax.set_xlabel('RA  (deg)', fontsize=10)
        ax.set_ylabel('Dec  (deg)', fontsize=10)
        ax.set_title(f'{lbl}  ({len(coords):,} loci)', fontsize=12, fontweight='bold')

    fig.suptitle('Sky Density Distribution - ANTARES / LSST',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig


def plot_magnitude_histograms(df1, df2, df1_alerts, df2_alerts,
                              range1_valid, range2_valid,
                              label1, label2,
                              mag_min=15.0, mag_max=25.0, n_bins=40):
    """
    Plot 3: Per-filter magnitude histograms on a log y-axis.

    Two data paths:
      - PRIMARY: split alert-level `ztf_magpsf` by `ztf_fid` (g/r/i).
        This needs lightcurves to have been loaded.
      - FALLBACK: if no alert data, use the locus-level
        `brightest_alert_magnitude` from each range. Less informative
        (one value per locus, no filter split) but still useful.

    Why log y-axis?
        Magnitude distributions span several orders of magnitude
        between the bright (rare) and faint (numerous) ends. Linear y
        compresses the bright tail into invisibility. The lower bound of
        0.5 keeps log scale honest by clipping empty bins (which would
        otherwise plot as -inf).

    Why a 15-25 mag window?
        Tighter than ZTF's full saturation/limiting range, but covers
        essentially every real LSST/ZTF alert. Anything outside this
        window is overwhelmingly artefacts or saturated stars.
    """
    bins = np.linspace(mag_min, mag_max, n_bins + 1)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Choose the data path. We prefer per-alert because it gives the
    # filter breakdown - that is the whole point of running the
    # lightcurve fetch.
    has_alert_mags = (
        (not df1_alerts.empty and LC_MAG in df1_alerts.columns) or
        (not df2_alerts.empty and LC_MAG in df2_alerts.columns)
    )

    if has_alert_mags:
        for rnum, (df_a, valid, label) in enumerate(
            [(df1_alerts, range1_valid, label1),
             (df2_alerts, range2_valid, label2)],
            start=1,
        ):
            if not valid or df_a.empty or LC_MAG not in df_a.columns:
                continue
            ls = MAG_LS[rnum]
            lw = MAG_LW[rnum]
            for fid, (band, color) in FILTER_INFO.items():
                if LC_FID not in df_a.columns:
                    continue
                # Restrict to the visible mag window AND to alerts in
                # this filter band. `between(..., inclusive='both')`
                # matches the bin edges so nothing is silently dropped
                # at the boundaries.
                mask = (
                    (df_a[LC_FID] == fid) &
                    df_a[LC_MAG].between(mag_min, mag_max, inclusive='both')
                )
                vals = df_a.loc[mask, LC_MAG].dropna()
                if vals.empty:
                    continue
                ax.hist(
                    vals, bins=bins,
                    histtype='step', color=color,
                    linestyle=ls, linewidth=lw,
                    alpha=0.90,
                    label=f'{band}-band  [{label}]  (n={len(vals):,})',
                )
    else:
        # Fallback path: one curve per range using locus brightest mag.
        mag_col = 'brightest_alert_magnitude'
        for rnum, (df, valid, label) in enumerate(
            [(df1, range1_valid, label1), (df2, range2_valid, label2)],
            start=1,
        ):
            if not valid or df.empty or mag_col not in df.columns:
                continue
            vals = df[mag_col].dropna()
            vals = vals[vals.between(mag_min, mag_max)]
            if vals.empty:
                continue
            ax.hist(
                vals, bins=bins,
                histtype='step', color=FALLBACK_MAG_COLORS[rnum],
                linestyle=MAG_LS[rnum], linewidth=MAG_LW[rnum],
                alpha=0.90,
                label=f'{label}  brightest mag  (n={len(vals):,})',
            )
        # Tell the viewer why they're seeing only two curves instead of
        # the usual six (3 filters x 2 ranges).
        ax.text(
            0.99, 0.99,
            'Using locus-level brightest_alert_magnitude\n'
            '(set LOAD_LIGHTCURVES = True for per-filter breakdown)',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=9, color='#aaaaaa',
        )

    ax.set_yscale('log')
    ax.set_xlim(mag_min, mag_max)
    ax.set_ylim(bottom=0.5)        # avoid log(0) for empty bins
    ax.set_xlabel('Magnitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alert count  (log scale)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Magnitude Distributions - {label1} (solid)  vs  {label2} (dashed)',
        fontsize=13, fontweight='bold',
    )
    ax.grid(True, alpha=0.2)
    # Vertical guide lines mark the analysis window edges; matches the
    # 15/25 mag x-limits but is drawn explicitly so it survives any
    # future change to the limits.
    ax.axvline(mag_min, color='white', linewidth=0.6, linestyle=':', alpha=0.35)
    ax.axvline(mag_max, color='white', linewidth=0.6, linestyle=':', alpha=0.35)
    ax.legend(fontsize=9, framealpha=0.25, facecolor='#1e1e1e',
              edgecolor='#555555', loc='upper left', ncol=2)
    plt.tight_layout()
    plt.show()
    return fig
