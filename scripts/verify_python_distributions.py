#!/usr/bin/env python3
"""Verify reproducible wheels and timestamp-independent sdist contents."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Dict, Tuple


FileRecord = Tuple[int, int, str]
ROOT_FILES = {"PKG-INFO", "README.md", "pyproject.toml", "setup.cfg"}
EGG_INFO_FILES = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "entry_points.txt",
    "requires.txt",
    "top_level.txt",
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def relative_member_name(member_name: str, root_name: str) -> str:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe sdist member path: {member_name!r}")
    if path.parts[0] != root_name:
        raise ValueError(
            f"sdist member {member_name!r} is outside the single root {root_name!r}"
        )
    relative = PurePosixPath(*path.parts[1:]).as_posix()
    if relative == ".":
        return ""
    if unicodedata.normalize("NFC", relative) != relative:
        raise ValueError(f"sdist member path is not NFC-normalized: {relative!r}")
    return relative


def allowed_sdist_file(relative: str) -> bool:
    if relative in ROOT_FILES:
        return True
    if relative.startswith("antares_analysis.egg-info/"):
        return relative.removeprefix("antares_analysis.egg-info/") in EGG_INFO_FILES
    if relative.startswith("src/"):
        return relative.endswith(".py") and "/__pycache__/" not in relative
    if relative.startswith("tests/"):
        return relative.endswith(".py") and "/__pycache__/" not in relative
    return False


def sdist_inventory(path: Path, source_root: Path) -> Dict[str, FileRecord]:
    """Return stable file metadata while rejecting unsafe or unexpected members."""
    inventory: Dict[str, FileRecord] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members if member.name}
        if len(roots) != 1:
            raise ValueError(f"{path}: expected one archive root, found {sorted(roots)}")
        root_name = next(iter(roots))
        for member in members:
            relative = relative_member_name(member.name, root_name)
            if not relative or member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"{path}: non-regular sdist member: {member.name}")
            if not allowed_sdist_file(relative):
                raise ValueError(f"{path}: unexpected sdist file: {relative}")
            if relative in inventory:
                raise ValueError(f"{path}: duplicate sdist member: {relative}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"{path}: cannot read sdist member: {relative}")
            content = extracted.read()
            inventory[relative] = (
                member.mode & 0o777,
                len(content),
                sha256_bytes(content),
            )

    expected_sources = {
        source.relative_to(source_root).as_posix(): source
        for source in sorted((source_root / "src").rglob("*.py"))
        if "__pycache__" not in source.parts
    }
    expected_sources.update(
        {
            source.relative_to(source_root).as_posix(): source
            for source in sorted((source_root / "tests").rglob("*.py"))
            if "__pycache__" not in source.parts
        }
    )
    expected_sources["pyproject.toml"] = source_root / "pyproject.toml"
    expected_sources["README.md"] = source_root / "README.md"
    for relative, source in expected_sources.items():
        if relative not in inventory:
            raise ValueError(f"{path}: missing source file: {relative}")
        actual_digest = inventory[relative][2]
        expected_digest = sha256_bytes(source.read_bytes())
        if actual_digest != expected_digest:
            raise ValueError(f"{path}: archived source differs from checkout: {relative}")
    return inventory


def verify_distributions(
    wheel_a: Path,
    wheel_b: Path,
    sdist_a: Path,
    sdist_b: Path,
    source_root: Path,
) -> None:
    wheel_a_digest = sha256_bytes(wheel_a.read_bytes())
    wheel_b_digest = sha256_bytes(wheel_b.read_bytes())
    if wheel_a.name != wheel_b.name or wheel_a_digest != wheel_b_digest:
        raise ValueError("project wheels are not byte-identical")

    first = sdist_inventory(sdist_a, source_root)
    second = sdist_inventory(sdist_b, source_root)
    if sdist_a.name != sdist_b.name or first != second:
        differing = sorted(set(first) ^ set(second))
        differing.extend(
            name for name in sorted(set(first) & set(second)) if first[name] != second[name]
        )
        raise ValueError(
            "project sdists are not content-equivalent: " + ", ".join(differing)
        )

    print(
        "Python distributions verified: byte-identical wheels and "
        f"{len(first)} content-equivalent, safe sdist files."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-a", type=Path, required=True)
    parser.add_argument("--wheel-b", type=Path, required=True)
    parser.add_argument("--sdist-a", type=Path, required=True)
    parser.add_argument("--sdist-b", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    verify_distributions(
        args.wheel_a,
        args.wheel_b,
        args.sdist_a,
        args.sdist_b,
        args.source_root.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
