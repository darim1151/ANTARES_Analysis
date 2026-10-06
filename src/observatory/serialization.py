"""Canonical native serialization and the sole, fail-closed shared boundary."""

import hashlib
import json
import math
from dataclasses import fields, is_dataclass


class SharedContractUnbound(RuntimeError):
    pass


def _native(value):
    if is_dataclass(value):
        return {field.name: _native(getattr(value, field.name)) for field in fields(value)}
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValueError("Unsafe JSON integer; identifiers must be encoded as strings")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite native value")
        return value
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _native(item) for key, item in value.items()}
    raise TypeError(f"Unsupported native JSON type: {type(value).__name__}")


def canonical_bytes(value):
    return (json.dumps(_native(value), sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def native_sha256(product):
    return hashlib.sha256(canonical_bytes(product)).hexdigest()


# A proposal describing responsibilities only, not common field definitions.
NATIVE_TO_COMMON_PROPOSAL = (
    ("manifest", "product kind, derivation, native version, verified owner identity"),
    ("basis", "native basis and source-generation evidence"),
    ("capabilities", "qualified capabilities and selection metadata"),
    ("time", "independent authority, availability, content and time semantics"),
    ("sky", "native backend metadata, cell geometry, exclusions and density"),
    ("entities", "native loci and preserved per-night snapshots"),
    ("features", "definitions, qualifications, missingness and raw native values"),
    ("provenance", "content-addressed input identities and evidence references"),
)


def serialize_shared_bundle(product, *, owner_identity=None):
    """No identity supplied by a caller can substitute for a verified owner contract.

    Binding requires a reviewed owner schema/version/content hash and tests in a
    later gate. Native serialization remains independently usable meanwhile.
    """
    raise SharedContractUnbound(
        "UNBOUND: exact owner Bundle V1 schema, version and hash are not verified"
    )
