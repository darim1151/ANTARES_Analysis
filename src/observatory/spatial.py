"""Isolated deterministic equal-solid-angle native backend, independent of V1."""

import math

from .model import AntaresSkyProduct, SkyCell


def build_sky(loci, provenance_refs, *, ra_bins=72, sin_dec_bins=36):
    for count in (ra_bins, sin_dec_bins):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4096:
            raise ValueError("Spatial bin counts must be integers in [1, 4096]")
    counts = {}
    excluded = {}
    for locus in loci:
        snapshot = locus.snapshots[-1]
        if snapshot.coordinate_state != "VALID":
            state = snapshot.coordinate_state
            excluded[state] = excluded.get(state, 0) + 1
            continue
        ra_index = min(ra_bins - 1, int((snapshot.ra_deg % 360.0) / 360.0 * ra_bins))
        sin_dec = math.sin(math.radians(snapshot.dec_deg))
        dec_index = min(sin_dec_bins - 1, int((sin_dec + 1.0) / 2.0 * sin_dec_bins))
        key = (dec_index, ra_index)
        counts[key] = counts.get(key, 0) + 1
    cells = tuple(SkyCell(
        cell_id=f"{dec_index}:{ra_index}",
        ra_min_deg=ra_index * 360.0 / ra_bins,
        ra_max_deg=(ra_index + 1) * 360.0 / ra_bins,
        sin_dec_min=-1.0 + dec_index * 2.0 / sin_dec_bins,
        sin_dec_max=-1.0 + (dec_index + 1) * 2.0 / sin_dec_bins,
        locus_count=count,
    ) for (dec_index, ra_index), count in sorted(counts.items()))
    return AntaresSkyProduct(cells, ra_bins, sin_dec_bins, len(loci),
                             sum(counts.values()), tuple(sorted(excluded.items())), provenance_refs)
