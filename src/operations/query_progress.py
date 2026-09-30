"""Run-local adaptive query journal; never scientific or publication authority.

Each committed event is an atomic, fsynced, hash-chained file. Orphan temporary
files are ignored. A missing/reordered/modified committed event fails closed.
The extractor independently replays and validates every traversal decision.
"""
from __future__ import annotations

import fcntl
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping
import uuid

from .query_checkpoint import (
    QueryCheckpointError, _DIRECTORY_FLAGS, _canonical_json_bytes,
    _canonical_run_root, _decode_checked, _encode_value, _open_directory_at,
    _parse_canonical_object, _read_regular_at, _sha256, _write_new_file_at,
)

SCHEMA = "v3.adaptive-query-progress.v1"


class QueryProgress:
    """Exclusive per-night progress access with exact caller-derived identity."""

    def __init__(self, run_root: Path, run_id: str, identity: Mapping[str, Any]):
        self.root = _canonical_run_root(run_root, run_id)
        self.identity = json.loads(_canonical_json_bytes(dict(identity)))
        self.identity_document = {"schema_version": SCHEMA, "authoritative": False, "identity": self.identity}
        self.identity_sha256 = _sha256(_canonical_json_bytes(self.identity_document))
        self.fd = None
        self.lock_fd = None
        self.events = []
        self.previous = "0" * 64
        self.head = None

    def __enter__(self):
        root_fd = os.open(str(self.root), _DIRECTORY_FLAGS)
        parent_fd = None
        try:
            for name in ("checkpoints",):
                try:
                    os.mkdir(name, 0o700, dir_fd=root_fd)
                except FileExistsError:
                    pass
            parent_fd = _open_directory_at(root_fd, "checkpoints", "Query checkpoints")
            try:
                os.mkdir("query-progress-v1", 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            self.fd = _open_directory_at(parent_fd, "query-progress-v1", "Query progress")
            for descriptor in (parent_fd, self.fd):
                if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
                    raise QueryCheckpointError("Query-progress directories must have mode 0700.")
            os.fsync(root_fd)
            os.fsync(parent_fd)
            self.lock_fd = os.open("LOCK", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
            if not stat.S_ISREG(os.fstat(self.lock_fd).st_mode):
                raise QueryCheckpointError("Query-progress lock is not a regular file.")
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            expected = self.identity_document
            if "identity.json" not in os.listdir(self.fd):
                if any(name.startswith("event-") or name == "HEAD.json" for name in os.listdir(self.fd)):
                    raise QueryCheckpointError("Query-progress identity is missing.")
                self._install("identity.json", _canonical_json_bytes(expected))
            observed = _parse_canonical_object(_read_regular_at(self.fd, "identity.json"), "Query progress identity")
            if observed != expected:
                raise QueryCheckpointError("Query-progress identity/ref mismatch.")
            entries = os.listdir(self.fd)
            if any(name not in {"identity.json", "HEAD.json", "LOCK"} and not name.startswith(("event-", ".tmp-")) for name in entries):
                raise QueryCheckpointError("Unexpected query-progress entry.")
            names = sorted(name for name in entries if name.startswith("event-"))
            head = None
            if "HEAD.json" in entries:
                head = _parse_canonical_object(_read_regular_at(self.fd, "HEAD.json"), "Query progress head")
                if (set(head) != {"count", "last_sha256", "identity_sha256"}
                        or type(head["count"]) is not int or not 0 <= head["count"] <= len(names)
                        or head["identity_sha256"] != self.identity_sha256):
                    raise QueryCheckpointError("Query-progress head/ref is corrupt or truncated.")
            elif names:
                raise QueryCheckpointError("Committed query-progress head is missing.")
            hashes = [self.previous]
            for index, name in enumerate(names):
                if name != f"event-{index:08d}.json":
                    raise QueryCheckpointError("Query-progress sequence is incomplete.")
                raw = _read_regular_at(self.fd, name)
                envelope = _parse_canonical_object(raw, "Query progress event")
                payload = envelope.get("payload")
                if (set(envelope) != {"payload", "sha256"} or not isinstance(payload, dict)
                        or payload.get("previous_sha256") != self.previous
                        or payload.get("sequence") != index
                        or payload.get("identity_sha256") != self.identity_sha256
                        or envelope["sha256"] != _sha256(_canonical_json_bytes(payload))):
                    raise QueryCheckpointError("Query-progress event integrity/ref mismatch.")
                self.previous = _sha256(raw)
                hashes.append(self.previous)
                self.events.append(payload["event"])
            if head is not None and hashes[head["count"]] != head["last_sha256"]:
                raise QueryCheckpointError("Query-progress head integrity differs.")
            self._commit_head()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise
        finally:
            if parent_fd is not None:
                os.close(parent_fd)
            os.close(root_fd)

    def _install(self, name, raw):
        temporary = f".tmp-{uuid.uuid4().hex}"
        _write_new_file_at(self.fd, temporary, raw)
        # Exclusive link is the commit boundary; never replace durable evidence.
        os.link(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
        os.fsync(self.fd)
        os.unlink(temporary, dir_fd=self.fd)
        os.fsync(self.fd)

    def commit(self, event):
        payload = {"sequence": len(self.events), "previous_sha256": self.previous,
                   "identity_sha256": self.identity_sha256, "event": event}
        raw = _canonical_json_bytes({"payload": payload, "sha256": _sha256(_canonical_json_bytes(payload))})
        self._install(f"event-{len(self.events):08d}.json", raw)
        self.previous = _sha256(raw)
        self.events.append(event)
        self._commit_head()

    def _commit_head(self):
        head = {"count": len(self.events), "last_sha256": self.previous,
                "identity_sha256": self.identity_sha256}
        temporary = f".tmp-{uuid.uuid4().hex}"
        _write_new_file_at(self.fd, temporary, _canonical_json_bytes(head))
        os.rename(temporary, "HEAD.json", src_dir_fd=self.fd, dst_dir_fd=self.fd)
        os.fsync(self.fd)

    def __exit__(self, *_args):
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def encode_records(records):
    # Top-level insertion order determines pandas column order. Nested mapping
    # key order remains semantically irrelevant, using the accepted value codec.
    return [[list(record), [_encode_value(value) for value in record.values()]] for record in records]


def decode_records(records):
    if not isinstance(records, list):
        raise QueryCheckpointError("Query-progress records must be an ordered list.")
    decoded = []
    for record in records:
        if (not isinstance(record, list) or len(record) != 2
                or not isinstance(record[0], list) or not isinstance(record[1], list)
                or len(record[0]) != len(record[1]) or len(set(record[0])) != len(record[0])
                or any(not isinstance(key, str) for key in record[0])):
            raise QueryCheckpointError("Query-progress record columns are invalid.")
        decoded.append(dict(zip(record[0], [_decode_checked(value) for value in record[1]])))
    if encode_records(decoded) != records:
        raise QueryCheckpointError("Query-progress record semantic round trip differs.")
    return decoded
