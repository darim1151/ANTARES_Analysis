"""Read-only Jupyter notebook discovery and command rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from src.cli_profiles import StorageProfile, shell_command


@dataclass(frozen=True)
class NotebookSpec:
    alias: str
    filename: str
    purpose: str

    def as_dict(self, repo_root: Path) -> dict[str, object]:
        values = asdict(self)
        path = repo_root / "notebooks" / self.filename
        values.update({"path": str(path), "exists": path.is_file()})
        return values


NOTEBOOKS = (
    NotebookSpec(
        "setup",
        "rsp_setup.ipynb",
        "Environment, storage, dependency, and ANTARES connectivity checks.",
    ),
    NotebookSpec(
        "historical-backfill",
        "historical_backfill.ipynb",
        "Build or extend historical nightly partitions and cumulative indexes.",
    ),
    NotebookSpec(
        "last-night",
        "alerts_time_comparison.ipynb",
        "Compare the latest completed ANTARES/LSST night with saved history.",
    ),
)


def _candidate_roots(cwd: Path) -> Iterable[Path]:
    yield cwd
    yield from cwd.parents
    package_root = Path(__file__).resolve().parents[1]
    if package_root != cwd and package_root not in cwd.parents:
        yield package_root


def discover_repo_root(explicit: Optional[Path] = None, cwd: Optional[Path] = None) -> Path:
    """Find a checkout containing the project metadata and notebook directory."""
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = list(_candidate_roots((cwd or Path.cwd()).resolve()))
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "notebooks"
        ).is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise ValueError(
        "Could not find an ANTARES Analysis checkout with pyproject.toml and "
        f"notebooks/. Searched: {searched}"
    )


def notebook_spec(alias: str) -> NotebookSpec:
    normalized = alias.strip().lower()
    for spec in NOTEBOOKS:
        if spec.alias == normalized:
            return spec
    choices = ", ".join(spec.alias for spec in NOTEBOOKS)
    raise ValueError(f"Unknown notebook {alias!r}; expected one of: {choices}.")


def render_jupyter_command(
    profile: StorageProfile,
    repo_root: Path,
    alias: str,
    launcher: str = "jupyter",
) -> str:
    """Render a safe Jupyter Lab command; do not execute it."""
    spec = notebook_spec(alias)
    notebook_path = repo_root / "notebooks" / spec.filename
    if not notebook_path.is_file():
        raise ValueError(f"Notebook is missing: {notebook_path}")
    return shell_command(profile, [launcher, "lab", str(notebook_path)])
