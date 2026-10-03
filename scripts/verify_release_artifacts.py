#!/usr/bin/env python3
"""Verify layered locks and their Linux wheelhouse closure."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, Mapping, Tuple


LOCKED_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)"
)
SHA256_HASH = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})(?:\s|\\|$)")
NORMALIZED_NAME = re.compile(r"[-_.]+")


def normalize_name(name: str) -> str:
    """Return the PEP 503 normalized distribution name."""
    return NORMALIZED_NAME.sub("-", name).lower()


def locked_versions(path: Path) -> Dict[str, str]:
    """Read exact, SHA-256-hashed pins from a pip-compile lock."""
    versions: Dict[str, str] = {}
    current_name = None
    current_line = None
    current_has_hash = False

    def finish_requirement() -> None:
        nonlocal current_name, current_line, current_has_hash
        if current_name is not None and not current_has_hash:
            raise ValueError(
                f"{path}:{current_line}: {current_name} has no SHA-256 hash"
            )
        current_name = None
        current_line = None
        current_has_hash = False

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        match = LOCKED_REQUIREMENT.match(stripped)
        if match is not None:
            finish_requirement()
            name = normalize_name(match.group(1))
            version = match.group(2)
            previous = versions.setdefault(name, version)
            if previous != version:
                raise ValueError(
                    f"{path}:{line_number}: conflicting pins for {name}: "
                    f"{previous!r} and {version!r}"
                )
            current_name = name
            current_line = line_number
            current_has_hash = SHA256_HASH.search(stripped) is not None
        elif current_name is not None:
            current_has_hash = current_has_hash or SHA256_HASH.search(stripped) is not None
        elif stripped and not stripped.startswith(("#", "--")):
            raise ValueError(
                f"{path}:{line_number}: requirement is not an exact name==version pin"
            )

        if current_name is not None and not stripped.endswith("\\"):
            finish_requirement()

    finish_requirement()
    if not versions:
        raise ValueError(f"{path}: no exact package pins found")
    return versions


def verify_lock_layer(
    production: Mapping[str, str], test: Mapping[str, str]
) -> None:
    """Require the test lock to contain the exact production closure."""
    missing = sorted(set(production) - set(test))
    mismatched = sorted(
        name
        for name, version in production.items()
        if name in test and test[name] != version
    )
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing from test lock: {', '.join(missing)}")
        if mismatched:
            rendered = ", ".join(
                f"{name} ({production[name]} != {test[name]})"
                for name in mismatched
            )
            details.append(f"version mismatch: {rendered}")
        raise ValueError("test lock does not preserve production closure; " + "; ".join(details))


def wheel_inventory(directory: Path) -> Dict[str, Tuple[str, Path, str]]:
    """Map normalized names to version, path, and SHA-256 for one wheelhouse."""
    inventory: Dict[str, Tuple[str, Path, str]] = {}
    unexpected = sorted(path.name for path in directory.iterdir() if path.suffix != ".whl")
    if unexpected:
        raise ValueError(
            f"{directory}: wheelhouse contains non-wheel files: {unexpected}"
        )
    for path in sorted(directory.glob("*.whl")):
        parts = path.name[:-4].split("-")
        if len(parts) < 5:
            raise ValueError(f"{path}: invalid wheel filename")
        name = normalize_name(parts[0])
        version = parts[1]
        if name in inventory:
            raise ValueError(f"{directory}: multiple wheels found for {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory[name] = (version, path, digest)
    if not inventory:
        raise ValueError(f"{directory}: no wheels found")
    return inventory


def verify_wheelhouse(
    lock: Mapping[str, str],
    wheels: Mapping[str, Tuple[str, Path, str]],
    label: str,
) -> None:
    """Require one exactly versioned wheel for every and only every lock pin."""
    missing = sorted(set(lock) - set(wheels))
    extra = sorted(set(wheels) - set(lock))
    mismatched = sorted(
        name
        for name, version in lock.items()
        if name in wheels and wheels[name][0] != version
    )
    if missing or extra or mismatched:
        details = []
        if missing:
            details.append(f"missing wheels: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected wheels: {', '.join(extra)}")
        if mismatched:
            rendered = ", ".join(
                f"{name} ({lock[name]} != {wheels[name][0]})"
                for name in mismatched
            )
            details.append(f"version mismatch: {rendered}")
        raise ValueError(f"{label} wheelhouse mismatch; " + "; ".join(details))


def verify_shared_wheels(
    production: Mapping[str, Tuple[str, Path, str]],
    test: Mapping[str, Tuple[str, Path, str]],
) -> None:
    """Require byte-identical wheels for the shared production closure."""
    mismatched = []
    for name, (version, path, digest) in production.items():
        test_version, test_path, test_digest = test[name]
        if version != test_version or path.name != test_path.name or digest != test_digest:
            mismatched.append(name)
    if mismatched:
        raise ValueError(
            "production/test wheelhouses disagree for: " + ", ".join(sorted(mismatched))
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("production_lock", type=Path)
    parser.add_argument("test_lock", type=Path)
    parser.add_argument("production_wheelhouse", type=Path)
    parser.add_argument("test_wheelhouse", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    production_lock = locked_versions(args.production_lock)
    test_lock = locked_versions(args.test_lock)
    verify_lock_layer(production_lock, test_lock)

    production_wheels = wheel_inventory(args.production_wheelhouse)
    test_wheels = wheel_inventory(args.test_wheelhouse)
    verify_wheelhouse(production_lock, production_wheels, "production")
    verify_wheelhouse(test_lock, test_wheels, "test")
    verify_shared_wheels(production_wheels, test_wheels)

    print(
        "Release closure verified: "
        f"{len(production_lock)} production pins, "
        f"{len(test_lock)} test pins, byte-identical shared wheels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
