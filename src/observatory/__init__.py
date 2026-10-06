"""Read-only ANTARES scientific products, independent of any shared UI schema."""

from .adapter import build_metadata_fixture, build_native_product
from .serialization import canonical_bytes, native_sha256, serialize_shared_bundle

__all__ = ["build_metadata_fixture", "build_native_product", "canonical_bytes",
           "native_sha256", "serialize_shared_bundle"]
