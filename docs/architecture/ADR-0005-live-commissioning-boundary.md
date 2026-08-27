# ADR-0005: Live ANTARES commissioning boundary

Status: accepted for Phase 6 qualification

## Decision

Phase 6 introduces exactly one new authority: `LIVE_ANTARES_READ`. It is sealed
to the exact Arnor canary run root, exact committed release, and UTC night
2026-06-27. It has no production data path and no publication, reconciliation,
or cache method. `night ingest` remains a pre-provider refusal.

The live query extracts the minimum reusable logic from the accepted
`probe_first_time_ra_dec` historical path. It reuses the LSST identifier
filter, `src.history.prepare_loci`, `prepare_alerts`, and `validation_summary`.
The requested interval is half-open MJD `[61218.0, 61219.0)`. Every time tile
uses `gte`/`lt`; RA covers `[0, 360)` and declination covers `[-90, 90]`, with
only the terminal +90-degree declination edge inclusive. No tag, sample target,
random seed, sort, cache, or parallel parent shard is allowed.

## Completeness contract

The initial partition is 48 30-minute time bins by 24 RA bins by 6 declination
bins: 6,912 tiles. An accepted tile must return fewer than the 50-row probe
threshold and its pinned `antares-client` iterator must end normally. A 50-row
probe is discarded and split along the largest normalized width using floors of
30 seconds, 0.05 degrees RA, and 0.05 degrees declination. Saturation at all
floors, a transport exception after bounded retry, malformed data, or a coverage
gap is `INCOMPLETE` or `FAILED`, never successful zero. Accepted tile frames are
concatenated in traversal order and duplicate `locus_id` values use historical
`keep="last"` semantics; raw and deduplicated identities are evidenced.

Fetch completion requires every query locus ID to have a successful full
`get_by_id` result. Requested, completed, failed, retry, lightcurve, and alert
counts are recorded. Any missing locus is non-publishable.

## Operational boundary

The Phase 5 journal, lock, staging, validator, and pre-commit reproof are reused.
The commissioning coordinator never calls `PublicationTransaction.publish` and
never begins reconciliation. A fresh process reopens the candidate while its
run-local writer lock is held, validates external query/fetch evidence and the
journal, and proves the production target absent. Production is hashed before
and after the operation; the retained candidate and evidence remain
non-authoritative beneath `work/canary/<RUN_ID>`.

## Authentication and load policy

ANTARES search is public in `antares-client` 1.14.0; streaming credentials are
not used. Credential-bearing or non-official API URLs are refused. Whole-tile
query attempts are capped at two, per-object fetch attempts at three, full-locus
fetches at four workers, and API calls at the pinned 60-second timeout. The
extractor constants, accepted cache-version identity
`probe50_time_ra_dec_v1`, actual cache absence, and all bounds are recorded in
release evidence.
