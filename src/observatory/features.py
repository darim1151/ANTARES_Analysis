"""Registry grounded in saved columns, with no invented broker estimators."""

import math
from numbers import Real

from .model import AntaresFeatureDefinition, FeatureValue


BROKER_FEATURES = (
    "feature_chi2_magn_r", "feature_standard_deviation_magn_r",
    *(f"feature_weighted_mean_magn_{band}" for band in "ugrizy"),
)
COLOR_FEATURES = ("color_g_minus_i", "color_u_minus_g", "color_g_minus_r", "color_r_minus_i")
SAVED_COUNT = "saved_alert_source_rows"


def numeric_value(value, *, present=True):
    if not present:
        return None, "ABSENT_COLUMN"
    if value is None:
        return None, "NULL"
    # Pandas/numpy null sentinels are intentionally not converted to science.
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None, "NULL"
    if isinstance(value, bool) or not isinstance(value, Real):
        return None, "MALFORMED"
    number = float(value)
    if not math.isfinite(number):
        return None, "NONFINITE"
    return number, None


def row_features(row, alert_count):
    values = [FeatureValue(name, *numeric_value(row.get(name), present=name in row))
              for name in BROKER_FEATURES]
    values.append(FeatureValue(SAVED_COUNT, alert_count, None))
    return tuple(values)


def feature_registry(snapshots, evidence_refs):
    records = tuple(snapshots)
    definitions = []
    for name in (*BROKER_FEATURES, SAVED_COUNT, *COLOR_FEATURES):
        values = [value for row in records for value in row.features if value.feature_ref == name]
        finite = sum(value.value is not None for value in values)
        missing = {}
        for value in values:
            if value.missingness:
                missing[value.missingness] = missing.get(value.missingness, 0) + 1
        color = name in COLOR_FEATURES
        count = name == SAVED_COUNT
        if color:
            missing = {"UNDEFINED_ESTIMATOR": len(records)} if records else {}
        state = "AVAILABLE" if count and records else (
            "PARTIALLY_QUALIFIED" if finite and not color else "UNAVAILABLE"
        )
        definitions.append(AntaresFeatureDefinition(
            name=name, namespace="antares.adapter" if count or color else "antares.broker",
            entity_level="locus_snapshot", dtype="int64" if count else "float64",
            unit="row" if count else None,
            band_label=None if count or color else name.rsplit("_", 1)[-1],
            passband_semantics="not applicable" if count else "UNESTABLISHED; suffix is a native label",
            definition=("Count of saved alert/history source rows linked to this locus in this partition"
                        if count else "Unavailable: no qualified color estimator" if color else
                        "Unmodified finite numeric value from the identically named saved broker column"),
            estimator_semantics=("Exact row count; repeats across partitions are not unique observations"
                                 if count else
                                 "Legacy diagnostic subtracts weighted means; calibration, simultaneity and broker estimators unestablished"
                                 if color else
                                 "Repository reads this feature but contains no generating estimator; weights, normalization, calibration and input selection unestablished"),
            qualification_state=state, source_rows=len(records),
            column_present_rows=0 if color else len(values) - missing.get("ABSENT_COLUMN", 0),
            finite_rows=finite, missing_rows=len(records) - finite,
            missingness_counts=tuple(sorted(missing.items())),
            laboratory_role="selection_count" if count else "unavailable_color" if color else "raw_native_feature",
            evidence_refs=evidence_refs,
        ))
    return tuple(sorted(definitions, key=lambda item: item.name))
