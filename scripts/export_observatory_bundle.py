#!/usr/bin/env python3
"""Export canonical ANTARES-native offline products. Shared Bundle V1 is UNBOUND."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.observatory import (  # noqa: E402
    build_metadata_fixture, build_native_product, canonical_bytes,
    native_sha256, serialize_shared_bundle,
)


def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def export(args):
    """Build fully before writing; the output must be outside every source root."""
    output = args.output.expanduser().resolve()
    source_roots = [path.expanduser().resolve() for path in
                    (args.offline_root, args.journal_root) if path is not None]
    if any(_inside(output, root) for root in source_roots):
        raise ValueError("Output must be outside authoritative/source and journal roots")
    if args.sentinel_path and output == args.sentinel_path.expanduser().resolve():
        raise ValueError("Output cannot replace supplied Sentinel evidence")
    if _inside(output, Path("/astro/store")):
        raise ValueError("This offline gate cannot write production paths")
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        code_state = "DIRTY" if dirty else "CLEAN"
    except (subprocess.CalledProcessError, OSError):
        raise ValueError("A repository code revision is required for this offline export")
    if args.metadata_fixture:
        product = build_metadata_fixture(code_sha=code_sha, code_state=code_state)
    else:
        if not all((args.journal_root, args.start, args.end, args.input_kind)):
            raise ValueError("Offline science export requires journal root, start, end and input kind")
        if any(_inside(root, Path("/astro/store")) for root in source_roots):
            raise ValueError("This gate reads offline copies only; production execution is forbidden")
        product = build_native_product(args.offline_root, args.journal_root, start=args.start,
            end=args.end, input_kind=args.input_kind, code_sha=code_sha, code_state=code_state,
            sentinel_path=args.sentinel_path, max_loci=args.max_loci)
    payload = serialize_shared_bundle(product) if args.format == "shared" else canonical_bytes(product)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Replace the named output with a new inode; hardlinks cannot mutate inputs.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".observatory-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return native_sha256(product)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata-fixture", action="store_true")
    source.add_argument("--offline-root", type=Path)
    parser.add_argument("--journal-root", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--input-kind", choices=("SYNTHETIC_AUTHORITY_FIXTURE", "SAVED_SCIENCE_SNAPSHOT"))
    parser.add_argument("--sentinel-path", type=Path)
    parser.add_argument("--max-loci", type=int, default=10000)
    parser.add_argument("--format", choices=("native", "shared"), default="native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        digest = export(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"Native SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
