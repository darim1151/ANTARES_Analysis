"""Server-coordinated authority locking for canonical ANTARES data reads.

The NFS-resident lock serializes complete logical reads against publication.
It is deliberately separate from the durable publication gate and journal:
the kernel lock is concurrency state, while those files are crash/recovery
evidence.

Readers never create the production lock.  A deployment/publisher must
provision it beneath the publication control root before canonical reads are
enabled.  Temporary test roots are the sole exception and are provisioned
locally so ordinary fixtures remain self-contained.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


AUTHORITY_LOCK_NAME = "publication-authority.flock"
CANONICAL_PRODUCTION_ROOT = Path("/astro/store/shire/ANTARES/data")
CANONICAL_CONTROL_ROOT = Path(
    "/astro/store/shire/ANTARES/work/publication/control"
)


class AuthorityLockError(RuntimeError):
    """The authority lock contract could not be established safely."""


class AuthorityLockUnavailable(AuthorityLockError):
    """Another reader/writer currently prevents the requested lock mode."""


class AuthorityLockOrderError(AuthorityLockError):
    """A nested lock request would invert the authority-lock order."""


class AuthorityLockUnsupported(AuthorityLockError):
    """The filesystem does not support the required kernel lock protocol."""


@dataclass
class _ProcessLockState:
    shared_descriptor: Optional[int] = None
    shared_holders: int = 0
    exclusive_descriptor: Optional[int] = None
    exclusive_owner: Optional[int] = None


_GUARD = threading.RLock()
_STATES: dict[str, _ProcessLockState] = {}
_LOCAL = threading.local()
_PROCESS_ID = os.getpid()


def _after_fork_child() -> None:
    """Drop inherited bookkeeping and descriptors in a forked child."""
    global _PROCESS_ID
    for state in _STATES.values():
        for descriptor in (state.shared_descriptor, state.exclusive_descriptor):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
    _STATES.clear()
    _LOCAL.held = {}
    _PROCESS_ID = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _thread_holds() -> dict[str, list[object]]:
    held = getattr(_LOCAL, "held", None)
    if held is None:
        held = {}
        _LOCAL.held = held
    return held


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolved_directory(path: Path, label: str) -> Path:
    lexical = Path(path).expanduser()
    if lexical.is_symlink():
        raise AuthorityLockError(f"{label} must be an existing real directory: {lexical}")
    if lexical.is_dir():
        return lexical.resolve(strict=True)
    if lexical.exists():
        raise AuthorityLockError(f"{label} must be an existing real directory: {lexical}")
    # Empty-snapshot and missing-root diagnostics are valid only in temporary
    # fixtures.  Anchor their lock on the nearest existing parent filesystem;
    # no production/non-temporary absence is accepted.
    unresolved = lexical.resolve(strict=False)
    temporary = Path(tempfile.gettempdir()).resolve(strict=True)
    if unresolved != temporary and _inside(unresolved, temporary):
        parent = unresolved.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if parent.is_dir() and not parent.is_symlink():
            return unresolved
    raise AuthorityLockError(f"{label} must be an existing real directory: {lexical}")


def _data_device(path: Path) -> int:
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.is_symlink() or not candidate.is_dir():
        raise AuthorityLockError(
            f"Authoritative data root has no safe filesystem anchor: {path}"
        )
    return os.stat(candidate).st_dev


def authority_lock_path(
    data_root: Path,
    *,
    explicit_path: Optional[Path] = None,
) -> Path:
    """Return the one lock path bound to ``data_root`` without creating it."""
    root = _resolved_directory(Path(data_root), "Authoritative data root")
    requested: Optional[Path] = None
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute() or ".." in path.parts:
            raise AuthorityLockError("An explicit authority lock path must be absolute and canonical.")
        requested = Path(os.path.abspath(path))

    if root == CANONICAL_PRODUCTION_ROOT:
        candidate = CANONICAL_CONTROL_ROOT / "locks" / AUTHORITY_LOCK_NAME

    # Synthetic publication roots use the same sibling control tree as the
    # publisher.  This is also the exact topology bound by the production
    # publication contract.
    elif root.name == "published" and (root.parent / "control").is_dir():
        candidate = root.parent / "control" / "locks" / AUTHORITY_LOCK_NAME
    else:
        temporary = Path(tempfile.gettempdir()).resolve(strict=True)
        if root != temporary and _inside(root, temporary):
            identity = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:20]
            candidate = (
                root.parent
                / ".antares-authority-locks"
                / f"{identity}-{AUTHORITY_LOCK_NAME}"
            )
        else:
            configured = os.environ.get("ANTARES_AUTHORITY_LOCK_PATH")
            if not configured:
                raise AuthorityLockError(
                    "No authority lock is configured for this non-temporary data root."
                )
            path = Path(configured).expanduser()
            if not path.is_absolute() or ".." in path.parts:
                raise AuthorityLockError(
                    "ANTARES_AUTHORITY_LOCK_PATH must be absolute and canonical."
                )
            candidate = Path(os.path.abspath(path))
            if not _inside(candidate, root.parent):
                raise AuthorityLockError(
                    "Configured authority lock leaves the trusted data-root parent."
                )

    if requested is not None and requested != candidate:
        raise AuthorityLockError(
            f"Explicit authority lock differs from the trusted lock path: {candidate}"
        )
    return candidate


def _temporary_lock(path: Path, data_root: Path) -> bool:
    temporary = Path(tempfile.gettempdir()).resolve(strict=True)
    root = Path(data_root).expanduser().resolve(strict=False)
    return root != temporary and _inside(root, temporary) and _inside(path, root.parent)


def _prepare_parent(path: Path, data_root: Path, *, create: bool) -> Path:
    parent = path.parent
    if not parent.exists():
        if not create and not _temporary_lock(path, data_root):
            raise AuthorityLockError(
                f"Authority lock parent is absent; deployment must provision it: {parent}"
            )
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise AuthorityLockError(f"Authority lock parent is unsafe: {parent}")
    resolved = parent.resolve(strict=True)
    if resolved != Path(os.path.abspath(parent)):
        raise AuthorityLockError(f"Authority lock parent is not canonical: {parent}")
    root = _resolved_directory(Path(data_root), "Authoritative data root")
    if os.stat(resolved).st_dev != _data_device(root):
        raise AuthorityLockError(
            "Authority lock and authoritative data are not on the same filesystem."
        )
    return resolved


def _open_lock(path: Path, data_root: Path, mode: str, *, create: bool) -> int:
    parent = _prepare_parent(path, data_root, create=create)
    if path.is_symlink():
        raise AuthorityLockError(f"Authority lock must not be a symlink: {path}")
    flags = os.O_RDONLY if mode == "shared" else os.O_RDWR
    if create or _temporary_lock(path, data_root):
        flags |= os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileNotFoundError as exc:
        raise AuthorityLockError(
            f"Authority lock is absent; deployment must provision it: {path}"
        ) from exc
    except OSError as exc:
        raise AuthorityLockError(f"Could not open authority lock {path}: {exc}") from exc
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise AuthorityLockError(f"Authority lock is not a regular file: {path}")
        named = os.stat(path, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (observed.st_dev, observed.st_ino):
            raise AuthorityLockError("Authority lock identity changed during acquisition.")
        if observed.st_dev != _data_device(Path(data_root).resolve(strict=False)):
            raise AuthorityLockError(
                "Authority lock and authoritative data are not on the same filesystem."
            )
        if path.parent.resolve(strict=True) != parent:
            raise AuthorityLockError("Authority lock parent changed during acquisition.")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _flock(descriptor: int, operation: int, path: Path) -> None:
    try:
        fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
    except BlockingIOError:
        raise
    except OSError as exc:
        if exc.errno in {
            getattr(errno, "ENOTSUP", -1),
            getattr(errno, "EOPNOTSUPP", -1),
            errno.EINVAL,
            errno.ENOSYS,
        }:
            raise AuthorityLockUnsupported(
                f"The filesystem does not support the authority lock protocol: {path}"
            ) from exc
        raise AuthorityLockError(f"Authority lock operation failed for {path}: {exc}") from exc


class AuthorityLock:
    """Reentrant process/thread-aware shared or exclusive authority lock."""

    def __init__(
        self,
        data_root: Path,
        mode: str,
        *,
        lock_path: Optional[Path] = None,
        wait_seconds: float = 0.0,
        poll_seconds: float = 0.05,
        create: bool = False,
    ) -> None:
        if mode not in {"shared", "exclusive"}:
            raise ValueError("Authority lock mode must be shared or exclusive.")
        if wait_seconds < 0 or poll_seconds <= 0:
            raise ValueError("Authority lock wait must be non-negative and poll positive.")
        self.data_root = _resolved_directory(Path(data_root), "Authoritative data root")
        self.path = authority_lock_path(self.data_root, explicit_path=lock_path)
        self.mode = mode
        self.wait_seconds = float(wait_seconds)
        self.poll_seconds = float(poll_seconds)
        self.create = bool(create)
        self._entered = False

    def __enter__(self) -> "AuthorityLock":
        global _PROCESS_ID
        if os.getpid() != _PROCESS_ID:
            _after_fork_child()
        key = str(self.path)
        held = _thread_holds()
        with _GUARD:
            existing = held.get(key)
            if existing is not None:
                existing_mode = str(existing[0])
                if existing_mode == "shared" and self.mode == "exclusive":
                    raise AuthorityLockOrderError(
                        "Cannot upgrade a nested shared authority lock to exclusive."
                    )
                if existing_mode == "exclusive" and self.mode == "exclusive":
                    raise AuthorityLockUnavailable(
                        "A nested publisher cannot reuse an exclusive authority lock."
                    )
                existing[1] = int(existing[1]) + 1
                self._entered = True
                return self

        deadline = time.monotonic() + self.wait_seconds
        while True:
            descriptor: Optional[int] = None
            blocked = False
            with _GUARD:
                state = _STATES.setdefault(key, _ProcessLockState())
                if self.mode == "shared":
                    if state.exclusive_descriptor is not None:
                        blocked = True
                    elif state.shared_descriptor is not None:
                        state.shared_holders += 1
                        held[key] = ["shared", 1]
                        self._entered = True
                        return self
                    else:
                        descriptor = _open_lock(
                            self.path, self.data_root, "shared", create=self.create
                        )
                        try:
                            _flock(descriptor, fcntl.LOCK_SH, self.path)
                        except BlockingIOError:
                            os.close(descriptor)
                            blocked = True
                        else:
                            state.shared_descriptor = descriptor
                            state.shared_holders = 1
                            held[key] = ["shared", 1]
                            self._entered = True
                            return self
                else:
                    if state.exclusive_descriptor is not None or state.shared_holders:
                        blocked = True
                    else:
                        descriptor = _open_lock(
                            self.path, self.data_root, "exclusive", create=self.create
                        )
                        try:
                            _flock(descriptor, fcntl.LOCK_EX, self.path)
                        except BlockingIOError:
                            os.close(descriptor)
                            blocked = True
                        else:
                            state.exclusive_descriptor = descriptor
                            state.exclusive_owner = threading.get_ident()
                            held[key] = ["exclusive", 1]
                            self._entered = True
                            return self
            if not blocked:
                raise AuthorityLockError("Authority lock acquisition failed unexpectedly.")
            if time.monotonic() >= deadline:
                raise AuthorityLockUnavailable(
                    f"The {self.mode} authority lock is busy: {self.path}"
                )
            time.sleep(self.poll_seconds)

    def __exit__(self, *exc: object) -> None:
        if not self._entered:
            return
        key = str(self.path)
        held = _thread_holds()
        with _GUARD:
            record = held.get(key)
            if record is None:
                raise AuthorityLockError("Authority lock bookkeeping was lost.")
            record[1] = int(record[1]) - 1
            if int(record[1]):
                return
            mode = str(record[0])
            del held[key]
            state = _STATES[key]
            if mode == "shared":
                state.shared_holders -= 1
                if state.shared_holders == 0:
                    descriptor = state.shared_descriptor
                    state.shared_descriptor = None
                    if descriptor is not None:
                        try:
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                        finally:
                            os.close(descriptor)
            else:
                descriptor = state.exclusive_descriptor
                state.exclusive_descriptor = None
                state.exclusive_owner = None
                if descriptor is not None:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    finally:
                        os.close(descriptor)
            if not state.shared_holders and state.exclusive_descriptor is None:
                _STATES.pop(key, None)
        self._entered = False


@contextmanager
def shared_authority_lock(
    data_root: Path,
    *,
    lock_path: Optional[Path] = None,
    wait_seconds: float = 30.0,
    poll_seconds: float = 0.05,
) -> Iterator[AuthorityLock]:
    with AuthorityLock(
        data_root,
        "shared",
        lock_path=lock_path,
        wait_seconds=wait_seconds,
        poll_seconds=poll_seconds,
    ) as lock:
        yield lock


@contextmanager
def exclusive_authority_lock(
    data_root: Path,
    *,
    lock_path: Optional[Path] = None,
    wait_seconds: float = 0.0,
    poll_seconds: float = 0.05,
    create: bool = True,
) -> Iterator[AuthorityLock]:
    with AuthorityLock(
        data_root,
        "exclusive",
        lock_path=lock_path,
        wait_seconds=wait_seconds,
        poll_seconds=poll_seconds,
        create=create,
    ) as lock:
        yield lock


__all__ = [
    "AUTHORITY_LOCK_NAME",
    "AuthorityLock",
    "AuthorityLockError",
    "AuthorityLockOrderError",
    "AuthorityLockUnavailable",
    "AuthorityLockUnsupported",
    "authority_lock_path",
    "exclusive_authority_lock",
    "shared_authority_lock",
]
