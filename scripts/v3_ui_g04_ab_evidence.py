#!/usr/bin/env python3
"""Bounded V3-UI-G0.4A/B evidence consumer. No command runs by default.

Local qualification uses temporary fixtures through the functions below.
The CLI has no alternate production roots, shortened observation interval,
longer deadline, retry, or write capability. See the accompanying runbook.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import selectors
import shutil
import socket
import stat
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASELINE = "812c545e14693cdce7ff7458f1d2b50b0804dcd8"
VERSION = "0.4.7"
SOURCE_DIGEST = "16499236e3d878b6fd6885d4d84db8e112616b063ccd5872606d20ccafa0590c"
PROJECT_DIGEST = "49fca3968093e5cbff0a879901b5f70c1ed796403499ed1b093f2f48eb4a7681"
AUTHORITY = Path("/astro/store/shire/ANTARES")
CONTROL_SECRETS = Path("/astro/users/mdarim/antares-control")
RELEASE = Path("/astro/users/mdarim/opt/antares-analysis/releases") / BASELINE
EXPECTED_UID = 1533564  # V3_G5_MANUAL_RANGE_RUNBOOK.md, scope paragraph.
OBSERVATION_SECONDS = 60
LOCK_WAIT_SECONDS = 5
HARD_SECONDS = 10
MAX_ITEMS = 4096
MAX_EVENTS_BYTES = 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_JOURNAL_TOTAL_BYTES = 64 * 1024 * 1024
MAX_PACKET_BYTES = 8 * 1024 * 1024
SCHEMA = "v3-ui-g04-ab-evidence.v1"
FORBIDDEN_KEYS = frozenset({
    "nonce", "control_token", "control_token_sha256", "control_approval_sha256",
    "approval_hmac", "authorization", "authorization_body", "authorization_sha256",
    "production_capability", "token", "secret", "token_file_content",
    "secret_file_content", "approval", "command_line", "argv",
})
ACTIVE_STAGES = frozenset({
    "QUERYING", "QUERY_COMPLETE", "FETCHING", "FETCH_COMPLETE", "CANDIDATE_BUILDING",
    "CANDIDATE_VALIDATED", "WAITING_FOR_PUBLICATION", "PUBLISHING",
})
STAGES = ACTIVE_STAGES | {"PLANNED", "PUBLISHED", "BLOCKED", "RECONCILIATION_REQUIRED"}
EVENTS = frozenset({
    "stage", "metrics", "failure", "retry", "query_reused", "query_uncommitted_preserved",
    "candidate_superseded", "acquisition_adopted",
})
SUMMARY_COLUMNS = (
    "date_utc", "status", "target_loci", "actual_loci", "alert_rows", "lsst_dia_count",
    "lsst_ss_count", "ztf_object_id_count", "parallel_shards", "chunk_count",
    "split_count", "saturated_chunk_count", "append_ready", "started_at_utc",
    "finished_at_utc",
)
INDEX_COLUMNS = ("night_date_utc", "ingested_at_utc")
CLASSIFICATION_FIELDS = (
    "state", "finalized", "legacy", "transaction_id", "journal_outcome", "reason",
    "resulting_production_fingerprint",
)
JOURNAL_NAME = re.compile(r"^v3pub-(\d{4}-\d{2}-\d{2})-[0-9a-f]{12}-a[1-9][0-9]{0,5}\.json$")


class Refuse(RuntimeError):
    """A fixed, non-secret refusal code; never format source exceptions."""


def inside(path, root):
    return path == root or root in path.parents


def real_path(path, *, missing=False):
    """lstat every component before resolving; never follow a symlink."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise Refuse("NONCANONICAL_PATH")
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor /= component
        try:
            observed = cursor.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise Refuse("MISSING_PATH")
        if stat.S_ISLNK(observed.st_mode):
            raise Refuse("SYMLINK_PATH")
        if cursor != path and not stat.S_ISDIR(observed.st_mode):
            raise Refuse("UNEXPECTED_FILE_TYPE")
    if path.resolve(strict=False) != path:
        raise Refuse("ALIASED_PATH")
    return path


def metadata(path, *, optional=False, directory=False):
    path = real_path(path, missing=optional)
    try:
        observed = path.lstat()
    except FileNotFoundError:
        if optional:
            return {"path": str(path), "presence": "ABSENT"}
        raise Refuse("MISSING_PATH")
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(observed.st_mode):
        raise Refuse("UNEXPECTED_FILE_TYPE")
    return {"path": str(path), "presence": "PRESENT", "bytes": observed.st_size,
            "mode": f"{stat.S_IMODE(observed.st_mode):04o}", "uid": observed.st_uid,
            "gid": observed.st_gid, "links": observed.st_nlink, "inode": observed.st_ino,
            "mtime_ns": observed.st_mtime_ns}


def entries(path, *, optional=False):
    info = metadata(path, optional=optional, directory=True)
    if info["presence"] == "ABSENT":
        return []
    result = []
    with os.scandir(path) as handle:
        for entry in handle:
            if len(result) >= MAX_ITEMS:
                raise Refuse("INVENTORY_LIMIT")
            result.append(Path(entry.path))
    return sorted(result)


def read_small(path, limit=MAX_JSON_BYTES, *, expected_uid=None):
    metadata(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as handle:
        observed = os.fstat(handle.fileno())
        if not stat.S_ISREG(observed.st_mode):
            raise Refuse("UNEXPECTED_FILE_TYPE")
        if expected_uid is not None and observed.st_uid != expected_uid:
            raise Refuse("PROCESS_OWNER_CHANGED")
        payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise Refuse("READ_LIMIT")
    return payload


def utc(value):
    if not isinstance(value, str) or len(value) > 64:
        raise Refuse("INVALID_TIMESTAMP")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise Refuse("INVALID_TIMESTAMP")
    if parsed.tzinfo is None:
        raise Refuse("INVALID_TIMESTAMP")
    return parsed.astimezone(timezone.utc)


def night(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise Refuse("INVALID_NIGHT")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise Refuse("INVALID_NIGHT")
    return value


def scan_forbidden(value):
    """Scan keys recursively; never print the rejected key or its value."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized in FORBIDDEN_KEYS:
                raise Refuse("FORBIDDEN_EVIDENCE_KEY")
            scan_forbidden(item)
    elif isinstance(value, list):
        for item in value:
            scan_forbidden(item)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise Refuse("NONPLAIN_EVIDENCE")


def output_root(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise Refuse("NONCANONICAL_PATH")
    # Reject lexically BEFORE lstat: even qualification must never probe $A.
    for protected in (AUTHORITY, CONTROL_SECRETS):
        if inside(path, protected) or inside(protected, path):
            raise Refuse("PROTECTED_OUTPUT_ROOT")
    path = real_path(path, missing=True)
    if path.exists():
        observed = path.lstat()
        if not stat.S_ISDIR(observed.st_mode) or observed.st_uid != os.geteuid():
            raise Refuse("UNSAFE_OUTPUT_ROOT")
        if stat.S_IMODE(observed.st_mode) & 0o077:
            raise Refuse("OUTPUT_ROOT_NOT_PRIVATE")
    return path


def write_evidence(root, tier, document):
    root = output_root(root)
    scan_forbidden(document)
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    if len(payload) > MAX_PACKET_BYTES:
        raise Refuse("PACKET_LIMIT")
    # No global tmpdir or replace of an existing artifact. All files are new.
    metadata(root.parent, directory=True)
    root.mkdir(mode=0o700, exist_ok=True)
    output_root(root)
    path = root / f"evidence-{tier}-{uuid.uuid4().hex}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    # Re-read only the artifact just generated, then recursively scan its keys.
    scan_forbidden(json.loads(read_small(path, MAX_PACKET_BYTES)))
    return path


def source_digest(directory):
    digest = hashlib.sha256()
    paths = sorted(directory.rglob("*.py"))
    if len(paths) > MAX_ITEMS:
        raise Refuse("SOURCE_LIMIT")
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0"
                      + hashlib.sha256(read_small(path)).digest())
    return digest.hexdigest()


def git(root, *arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True,
                            timeout=5, text=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    if result.returncode or len(result.stdout) > 65536:
        raise Refuse("REPOSITORY_IDENTITY")
    return result.stdout.strip()


def verify_repository(root):
    root = real_path(root)
    if git(root, "rev-parse", "--show-toplevel") != str(root):
        raise Refuse("REPOSITORY_IDENTITY")
    remote = git(root, "remote", "get-url", "origin")
    if remote not in {"https://github.com/darim1151/ANTARES_Analysis.git",
                      "git@github.com:darim1151/ANTARES_Analysis.git"}:
        raise Refuse("REPOSITORY_IDENTITY")
    git(root, "merge-base", "--is-ancestor", BASELINE, "HEAD")
    if (source_digest(root / "src") != SOURCE_DIGEST or
            hashlib.sha256(read_small(root / "pyproject.toml")).hexdigest() != PROJECT_DIGEST):
        raise Refuse("RELEASE_SOURCE_IDENTITY")
    return {"repository": "darim1151/ANTARES_Analysis", "source_sha": BASELINE,
            "review_head": git(root, "rev-parse", "HEAD"), "release": VERSION,
            "source_digest": SOURCE_DIGEST}


@dataclass(frozen=True)
class Layout:
    authority: Path
    data: Path
    survey: Path
    journals: Path
    lock: Path
    evidence: Path
    ranges: Path
    gate: Path
    index: Path
    summary: Path

    def partition(self, day):
        return self.survey / "nightly" / Path(night(day).replace("-", "/"))

    def plain(self):
        return {key: str(value) for key, value in self.__dict__.items()}


def static_layout(root):
    """Evaluate ONLY Path/string/join constants, without importing the package."""
    tree = ast.parse(read_small(root / "src/operations/storage.py"))
    values = {}

    def constant(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
            if len(node.args) == 1 and not node.keywords:
                return Path(constant(node.args[0]))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return constant(node.left) / constant(node.right)
        raise Refuse("STATIC_PATH_CONTRADICTION")

    names = {"PRODUCTION_AUTHORITY_ROOT", "PRODUCTION_DATA_ROOT", "PRODUCTION_PUBLICATION_ROOT",
             "PRODUCTION_CONTROL_ROOT", "PRODUCTION_EVIDENCE_ROOT", "RANGE_WORK_PARENT"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in names:
                values[key] = constant(node.value)
    history = ast.parse(read_small(root / "src/history.py"))
    functions = {node.name: node for node in history.body if isinstance(node, ast.FunctionDef)}
    survey_assignment = next(node for node in functions["survey_data_root"].body if isinstance(node, ast.Assign))
    # The pinned code says Path(data_root) / "data". Preserve that extra component.
    if ast.dump(survey_assignment.value) != ast.dump(ast.parse('Path(data_root) / "data"', mode="eval").body):
        raise Refuse("STATIC_PATH_CONTRADICTION")
    gate = next(node.value.value for node in history.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "PUBLICATION_GATE_NAME" for target in node.targets))
    config = ast.parse(read_small(root / "src/config.py"))
    subdir = next(node.value.args[1].body.value for node in config.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "HISTORY_DATA_SUBDIR" for target in node.targets))
    authority_tree = ast.parse(read_small(root / "src/authority.py"))
    lock_name = next(node.value.value for node in authority_tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "AUTHORITY_LOCK_NAME" for target in node.targets))
    data = values["PRODUCTION_DATA_ROOT"]
    survey = data / "data" / subdir
    control = values["PRODUCTION_CONTROL_ROOT"]
    if values["PRODUCTION_AUTHORITY_ROOT"] != AUTHORITY:
        raise Refuse("STATIC_PATH_CONTRADICTION")
    return Layout(AUTHORITY, data, survey, control / "journals", control / "locks" / lock_name,
                  values["PRODUCTION_EVIDENCE_ROOT"], values["RANGE_WORK_PARENT"],
                  survey / gate, survey / "cumulative/loci_index.parquet",
                  survey / "cumulative/nightly_summary.parquet")


class ReadBoundary:
    """Script-specific audit guard, before imports or production access.

    Native worker reads/writes additionally use Landlock; native networking is
    denied by seccomp. Parent uses stdlib metadata and explicit read-only opens.
    """
    def __init__(self, layout, output=None, *, worker=False, plan=False):
        self.layout, self.output, self.worker, self.plan = layout, output, worker, plan

    def check_path(self, value, writing=False):
        if isinstance(value, int):  # fdopen of descriptors we opened ourselves.
            return
        path = Path(os.fsdecode(value))
        if not path.is_absolute():
            path = Path.cwd() / path
        if ".." in path.parts:
            raise Refuse("PATH_TRAVERSAL")
        lower_parts = {part.lower() for part in path.parts}
        if (inside(path, CONTROL_SECRETS) or "production-bindings" in lower_parts
                or lower_parts & {"authorizations", "tokens", "approvals"}
                or path.name.lower() in {"authorization.json", "approval.json"}
                or path.suffix.lower() == ".token"):
            raise Refuse("CONTROL_READ_DENIED")
        if writing:
            if self.worker or self.output is None or not inside(path, self.output):
                raise Refuse("WRITE_DENIED")
        if inside(path, self.layout.authority):
            if writing or self.plan:
                raise Refuse("PRODUCTION_ACCESS_DENIED")
            real_path(path, missing=True)
            if path == self.layout.gate:
                raise Refuse("GATE_CONTENT_DENIED")
            allowed = (path in {self.layout.index, self.layout.summary, self.layout.lock}
                       or path == self.layout.journals
                       or (inside(path, self.layout.journals) and path.suffix == ".json")
                       or (inside(path, self.layout.survey / "nightly") and path.name == "manifest.json"))
            event = inside(path, self.layout.ranges) and path.name == "events.jsonl"
            if not event and not (self.worker and allowed):
                raise Refuse("PRODUCTION_CONTENT_DENIED")

    def __call__(self, event, args):
        if event == "open":
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            self.check_path(args[0], writing)
        elif event.startswith("socket.") and event != "socket.gethostname":
            raise Refuse("NETWORK_DENIED")
        elif event == "import" and str(args[0]).split(".")[0] == "antares_client":
            raise Refuse("CLIENT_IMPORT_DENIED")
        elif event in {"os.listdir", "os.scandir"} and not isinstance(args[0], int):
            path = Path(os.path.abspath(os.fsdecode(args[0])))
            if inside(path, CONTROL_SECRETS) or "production-bindings" in path.parts:
                raise Refuse("CONTROL_READ_DENIED")
            if self.plan and inside(path, self.layout.authority):
                raise Refuse("PRODUCTION_ACCESS_DENIED")
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate", "os.utime"}:
            self.check_path(args[0], True)
        elif event in {"os.rename", "os.link", "os.symlink"}:
            raise Refuse("MUTATION_DENIED")
        elif event == "subprocess.Popen":
            executable, arguments = args[0], args[1]
            if self.worker:
                raise Refuse("SUBPROCESS_DENIED")
            name = Path(executable).name
            if executable == "/usr/bin/python3" and arguments[1:] == ["-I", "-B", "--version"]:
                return
            if name not in {"git", "ps", Path(sys.executable).name}:
                raise Refuse("SUBPROCESS_DENIED")
            if name == Path(sys.executable).name and arguments[1:4] != ["-I", "-B", "-c"]:
                raise Refuse("SUBPROCESS_DENIED")


def generation(layout):
    # Same dimensions/order as history.authority_generation; metadata only.
    result = []
    for path in (layout.index, layout.summary):
        item = metadata(path, optional=True)
        result.append(None if item["presence"] == "ABSENT" else
                      [item["inode"], item["bytes"], item["mtime_ns"]])
    return result


def gate_clear(layout):
    # A final-component symlink is a present gate, including a dangling link.
    real_path(layout.gate.parent, missing=True)
    try:
        layout.gate.lstat()
    except FileNotFoundError:
        return True
    return False


def event_tail(path, run_id, now):
    info = metadata(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as handle:
        start = max(0, info["bytes"] - MAX_EVENTS_BYTES)
        handle.seek(start)
        raw = handle.read(MAX_EVENTS_BYTES + 1)
    if len(raw) > MAX_EVENTS_BYTES or metadata(path) != info or (raw and not raw.endswith(b"\n")):
        raise Refuse("EVENT_LOG_MOVING_OR_INCOMPLETE")
    if start:
        raw = raw.partition(b"\n")[2]  # drop partial first line.
    latest, last_stage, recent, oldest = {}, None, False, None
    for line in raw.splitlines():
        if len(line) > 16384:
            raise Refuse("EVENT_LINE_LIMIT")
        item = json.loads(line)
        if not isinstance(item, dict):
            raise Refuse("MALFORMED_EVENT")
        stamp = utc(item.get("utc"))
        if oldest is None:
            oldest = stamp
        if stamp > now + timedelta(seconds=60):
            raise Refuse("EVENT_CLOCK_UNCERTAIN")
        stage = item.get("stage")
        if stage is not None:
            if stage not in STAGES:
                raise Refuse("UNKNOWN_STAGE")
            last_stage = stage
        if stage in ACTIVE_STAGES and stamp >= now - timedelta(minutes=30):
            recent = True
        event = item.get("event")
        latest = {"run_id": run_id, "most_recent_event": event if event in EVENTS else "UNKNOWN",
                  "most_recent_stage": last_stage, "timestamp": stamp.isoformat()}
    if start and (oldest is None or oldest > now - timedelta(minutes=30)):
        raise Refuse("EVENT_WINDOW_INCOMPLETE")
    return latest or {"run_id": run_id, "most_recent_event": None,
                      "most_recent_stage": None, "timestamp": None}, recent


def range_activity(layout, now):
    records, recent = [], False
    for run in entries(layout.ranges, optional=True):
        metadata(run, directory=True)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run.name):
            raise Refuse("UNSAFE_RUN_ID")
        # Exact release topology: run/nights/night-YYYY-MM-DD/events.jsonl.
        for workspace in entries(run / "nights", optional=True):
            metadata(workspace, directory=True)
            if not workspace.name.startswith("night-"):
                raise Refuse("UNEXPECTED_WORKSPACE")
            night(workspace.name[6:])
            path = workspace / "events.jsonl"
            if metadata(path, optional=True)["presence"] == "ABSENT":
                continue
            record, busy = event_tail(path, run.name, now)
            records.append(record)
            if len(records) > MAX_ITEMS:
                raise Refuse("INVENTORY_LIMIT")
            recent |= busy
    return records, recent


def controller_classification(command):
    """Match the actual module/import-form runbook entry and conservative wrappers."""
    if "src.operations.production_range" in command or re.search(r"(?:/|\s)production_range\.py(?:\s|$)", command):
        if re.search(r"(?:^|\s)(execute|recover)(?:\s|$)", command) or "--execute-authorized-range" in command:
            return "production_range.execute_or_recover"
        if " plan " in f" {command} " or " inspect " in f" {command} ":
            return None
        return "production_range.ambiguous_controller"
    if "BackfillController" in command or "ProductionRangePublisher" in command:
        return "range_controller.python_wrapper"
    if str(AUTHORITY / "work/backfill") in command and ("--resume" in command or "--execute-authorized-range" in command):
        return "range_controller.work_root_wrapper"
    if "src.operations.production_canary" in command:
        return "production_canary.possible_publisher"
    return None


def live_controllers(rows=None):
    if rows is None:
        if not Path("/proc").is_dir():
            raise Refuse("PROCESS_DISCOVERY_UNAVAILABLE")
        rows = []
        for entry in entries(Path("/proc")):
            if not entry.name.isdigit():
                continue
            try:
                owner = entry.lstat().st_uid
                if owner != os.geteuid() or int(entry.name) == os.getpid():
                    continue
                # Ownership is checked before reading cmdline. Never inspect another UID.
                command = read_small(entry / "cmdline", 65536, expected_uid=owner).replace(b"\0", b" ").decode("utf-8", "replace")
            except FileNotFoundError:
                continue  # process exited between ownership check and read.
            except Refuse as error:
                if str(error) == "MISSING_PATH":
                    continue
                raise
            rows.append(f"{owner} {entry.name} {command}")
    matches = []
    if len(rows) > MAX_ITEMS:
        raise Refuse("PROCESS_LIMIT")
    for row in rows:
        fields = row.strip().split(None, 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            raise Refuse("PROCESS_DISCOVERY_UNAVAILABLE")
        if int(fields[0]) != os.geteuid() or int(fields[1]) == os.getpid():
            continue
        classification = controller_classification(fields[2])
        if classification:
            matches.append({"controller_pid": int(fields[1]), "classification": classification})
    return {"active_controller_present": bool(matches), "controllers": matches}


def ap0(layout, *, sleep=time.sleep, clock=lambda: datetime.now(timezone.utc), controllers=live_controllers):
    def activity():
        try:
            records, busy = range_activity(layout, clock())
            return records, busy, None
        except Refuse as error:
            if str(error) in {"SYMLINK_PATH", "UNEXPECTED_FILE_TYPE", "ALIASED_PATH"}:
                raise  # packet requires STOP on unsafe files.
            return [], True, str(error)

    def processes():
        try:
            return controllers(), None
        except (OSError, Refuse) as error:
            return {"active_controller_present": None, "controllers": []}, type(error).__name__

    first = generation(layout)
    clear = gate_clear(layout)
    early_records, early_busy, early_activity_error = activity()
    early_live, early_process_error = processes()
    started = time.monotonic()
    sleep(OBSERVATION_SECONDS)
    elapsed = time.monotonic() - started
    second = generation(layout)
    clear = clear and gate_clear(layout)
    records, busy, activity_error = activity()
    live, process_error = processes()
    flags = {"gate_clear": clear,
             "generation_stable": first == second and all(item is not None for item in first),
             "no_live_controller": early_live["active_controller_present"] is False and live["active_controller_present"] is False,
             "no_recent_nonterminal_activity": not (early_busy or busy)}
    return {**flags, "tier_b_safe_to_attempt": all(flags.values()),
            "observation_seconds": elapsed, "first_generation": first, "second_generation": second,
            "initial_range_activity": early_records, "range_activity": records,
            "initial_controller_detection": early_live, "controller_detection": live,
            "activity_discovery_refusal": early_activity_error or activity_error,
            "process_discovery_refusal": early_process_error or process_error}


def mount_facts(lines):
    matches = []
    for line in lines:
        left, separator, right = line.partition(" - ")
        fields, extra = left.split(), right.split()
        if not separator or len(fields) < 6 or len(extra) < 3:
            continue
        point = fields[4].replace(r"\040", " ").replace(r"\134", "\\")
        if inside(AUTHORITY, Path(point)) and extra[0] in {"nfs", "nfs4"}:
            options = fields[5].split(",") + extra[2].split(",")
            selected = sorted({value for value in options if value in {
                "hard", "soft", "noac", "ac", "nocto", "cto", "sync", "async"} or
                value.split("=", 1)[0] in {"vers", "nfsvers", "minorversion", "proto", "mountproto",
                                         "actimeo", "acregmin", "acregmax", "acdirmin", "acdirmax",
                                         "lookupcache", "local_lock", "nconnect"}})
            matches.append({"mount_point": point, "source": extra[1],
                            "filesystem_type": extra[0], "nfs_options": selected})
    return max(matches, key=lambda item: len(item["mount_point"])) if matches else "UNKNOWN"


def host_text(path, limit=65536):
    """Fixed host pseudo-files may have zero stat size. Never open arbitrary input."""
    try:
        return read_small(Path(path), limit).decode().strip()
    except (OSError, Refuse, UnicodeError):
        return "UNKNOWN"


def ap1():
    hostname = socket.gethostname()  # local kernel call, no DNS/NSS lookup.
    fqdn = hostname if "." in hostname else "UNKNOWN"
    for line in host_text("/etc/hosts").splitlines():
        names = line.partition("#")[0].split()[1:]
        if hostname in names:
            fqdn = next((name for name in names if "." in name and name != "localhost.localdomain"), fqdn)
    return {"hostname": hostname, "hostname_fqdn_local": fqdn, "uid": os.getuid(),
            "effective_uid": os.geteuid(), "gid": os.getgid(), "groups": os.getgroups(),
            "kernel_release": platform.release(), "expected_lock_uid": EXPECTED_UID,
            "tier_b_uid_eligible": os.getuid() == os.geteuid() == EXPECTED_UID,
            "shire_mount": mount_facts(host_text("/proc/self/mountinfo", MAX_JSON_BYTES).splitlines())}


def ap2():
    username = next((line.split(":")[0] for line in host_text("/etc/passwd").splitlines()
                     if len(line.split(":")) >= 4 and line.split(":")[2] == str(os.geteuid())), None)
    cgroup = host_text("/proc/self/cgroup")
    relative = next((line[3:] for line in cgroup.splitlines() if line.startswith("0::")), None)
    controllers = "UNKNOWN"
    subtree = writable = "UNKNOWN"
    if relative and ".." not in Path(relative).parts:
        scope = Path("/sys/fs/cgroup") / relative.lstrip("/")
        controllers = host_text(str(scope / "cgroup.controllers"))
        subtree = host_text(str(scope / "cgroup.subtree_control"))
        writable = os.access(scope / "cgroup.procs", os.W_OK)
    abi = landlock_abi()
    return {"lsms": host_text("/sys/kernel/security/lsm"),
            "landlock_abi": abi, "landlock_plausibility": "EXPOSED" if isinstance(abi, int) and abi >= 1 else "UNKNOWN",
            "bubblewrap": shutil.which("bwrap") or "ABSENT",
            "unprivileged_userns_clone": host_text("/proc/sys/kernel/unprivileged_userns_clone"),
            "max_user_namespaces": host_text("/proc/sys/user/max_user_namespaces"),
            "delegated_cgroup_controllers": controllers,
            "cgroup_subtree_control": subtree, "cgroup_procs_writable": writable,
            "user_linger_file_present": Path("/var/lib/systemd/linger", username).exists() if username else "UNKNOWN",
            "cron_executable": shutil.which("crontab") or "ABSENT"}


def ap3(layout):
    journals = entries(layout.journals, optional=True)
    journal_metadata = [metadata(path) for path in journals]
    v3_dates = {night(match.group(1)) for path in journals if (match := JOURNAL_NAME.fullmatch(path.name))}
    nights, manifests = [], []
    for year in entries(layout.survey / "nightly"):
        metadata(year, directory=True)
        if not re.fullmatch(r"\d{4}", year.name):
            raise Refuse("UNEXPECTED_PARTITION")
        for month in entries(year):
            metadata(month, directory=True)
            if not re.fullmatch(r"\d{2}", month.name):
                raise Refuse("UNEXPECTED_PARTITION")
            for day in entries(month):
                metadata(day, directory=True)
                stamp = night(f"{year.name}-{month.name}-{day.name}")
                # Inspect EVERY immediate partition entry for symlink/nonregular types.
                for path in entries(day):
                    metadata(path)
                manifest = metadata(day / "manifest.json", optional=True)
                if manifest["presence"] == "PRESENT":
                    nights.append(stamp)
                    manifests.append({"night": stamp, **manifest})
                if len(nights) > MAX_ITEMS:
                    raise Refuse("INVENTORY_LIMIT")
    nights.sort()
    candidates = sorted(set(nights) & v3_dates)
    legacy = next((day for day in nights if day < "2026-06-27" and day not in v3_dates), None)
    newest = candidates[-1] if candidates else None
    selected = sorted(set(candidates[-3:] + ["2026-06-27"] + ([legacy] if legacy else [])))
    partitions = {day: {name: metadata(layout.partition(day) / name, optional=True)
                        for name in ("manifest.json", "loci.parquet", "alerts.parquet")} for day in selected}
    terminals = []
    if newest:
        terminals = [metadata(path) for path in entries(layout.evidence / "publications" / newest, optional=True)]
    return {"directory_night_count": len(nights), "directory_tail": nights[-1] if nights else None,
            "authority_status": "UNVERIFIED_METADATA_ONLY", "physical_nights": nights,
            "v3_candidates_from_journal_names": candidates, "newest_v3_candidate": newest,
            "legacy_candidate": legacy, "legacy_confirmation": "REQUIRES_RELEASE_CLASSIFIER",
            "partitions": partitions, "journal_count": len(journals),
            "journal_total_bytes": sum(item["bytes"] for item in journal_metadata),
            "terminal_records": terminals, "cumulative": {"loci_index": metadata(layout.index),
            "nightly_summary": metadata(layout.summary)},
            "manifest_inventory": manifests,
            "manifest_total_bytes": sum(item["bytes"] for item in manifests),
            "p2_x1_first_measured_manifest_sizes": [{"night": day, **partitions[day]["manifest.json"]} for day in selected]}


def ap4(output):
    memory = host_text("/proc/meminfo")
    match = re.search(r"^MemTotal:\s+(\d+) kB$", memory, re.M)
    anchor = output
    while not anchor.exists():
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    try:
        result = subprocess.run(["/usr/bin/python3", "-I", "-B", "--version"], capture_output=True,
                                text=True, timeout=5)
        system_version = result.stdout.strip() if result.returncode == 0 and re.fullmatch(
            r"Python \d+\.\d+\.\d+", result.stdout.strip()) else "UNKNOWN"
    except (OSError, subprocess.TimeoutExpired):
        system_version = "UNKNOWN"
    # Interpreter identity is inspected in this process, never via executable input.
    return {"cpus": os.cpu_count(), "memory_bytes": int(match.group(1)) * 1024 if match else "UNKNOWN",
            "disk_candidates": [{"path": str(anchor), "total_bytes": usage.total, "free_bytes": usage.free}],
            "console_store_decision": "UNKNOWN", "invoked_python": platform.python_version(),
            "invoked_python_path": sys.executable,
            "system_python_path": "/usr/bin/python3",
            "system_python_version": system_version,
            "immutable_release_python_version": platform.python_version() if Path(sys.prefix) == RELEASE / "venv" else "UNKNOWN"}


POLICY_QUESTIONS = (
    "May a user process listen on a mode-0600 Unix domain socket on Arnor?",
    "Does sshd permit stream-local forwarding?",
    "Is Arnor multi-user in normal operation?",
    "May the user run a persistent user timer?",
    "If not, may cron run bounded exporter/deriver jobs?",
    "Are there site restrictions relevant to bubblewrap/user namespaces?",
)


def tier_a(layout, output):
    return {"AP-1": ap1(), "AP-2": ap2(), "AP-3": ap3(layout), "AP-4": ap4(output),
            "AP-5": [{"question": question, "answer": "UNKNOWN"} for question in POLICY_QUESTIONS]}


def sha256_file(path):
    metadata(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise Refuse("UNEXPECTED_FILE_TYPE")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise Refuse("HASH_GENERATION_DRIFT")
    return digest.hexdigest()


def parquet_rows(path, columns):
    """All Arrow objects and descriptors die here, inside the authority hold."""
    import pyarrow.parquet as pq
    metadata(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as handle, pq.ParquetFile(handle, memory_map=False) as parquet:
        names = parquet.schema_arrow.names
        if len(names) != len(set(names)):
            raise Refuse("DUPLICATE_COLUMNS")
        present = [column for column in columns if column in names]
        rows = []
        if columns == SUMMARY_COLUMNS:
            if parquet.metadata.num_rows > MAX_ITEMS:
                raise Refuse("SUMMARY_ROW_LIMIT")
            for batch in parquet.iter_batches(columns=present, batch_size=4096, use_threads=False):
                rows.extend(batch.to_pylist())
            return {"columns": {column: "PRESENT" if column in present else "ABSENT" for column in columns},
                    "unexpected_column_count": len(set(names) - set(columns)),
                    "rows": sanitize_summary(rows)}
        aggregates = {}
        invalid = 0
        if parquet.metadata.num_rows > 20_000_000:
            raise Refuse("INDEX_ROW_LIMIT")
        if "night_date_utc" in present:
            for batch in parquet.iter_batches(columns=present, batch_size=32768, use_threads=False):
                for row in batch.to_pylist():
                    day = night(row["night_date_utc"])
                    value = row.get("ingested_at_utc")
                    stamp = None
                    if value is not None:
                        try:
                            stamp = utc(value).isoformat()
                        except Refuse:
                            invalid += 1
                    entry = aggregates.setdefault(day, {"row_count": 0, "min_ingested_at_utc": None,
                                                         "max_ingested_at_utc": None})
                    if len(aggregates) > MAX_ITEMS:
                        raise Refuse("INDEX_NIGHT_LIMIT")
                    entry["row_count"] += 1
                    if stamp:
                        entry["min_ingested_at_utc"] = min(entry["min_ingested_at_utc"] or stamp, stamp)
                        entry["max_ingested_at_utc"] = max(entry["max_ingested_at_utc"] or stamp, stamp)
        return {"columns": {column: "PRESENT" if column in present else "ABSENT" for column in columns},
                "physical_row_count": parquet.metadata.num_rows, "nights": aggregates,
                "invalid_ingestion_timestamp_count": invalid,
                "aggregation_status": "COMPLETE" if all(column in present for column in columns) else "INCOMPLETE"}


def sanitize_summary(rows):
    result = []
    for row in rows:
        clean = {}
        for key in SUMMARY_COLUMNS:
            if key not in row:
                continue  # absence is in columns; never fill an absent source field.
            value = row[key]
            if value is None:
                clean[key] = None
            elif key == "date_utc":
                clean[key] = night(value)
            elif key in {"started_at_utc", "finished_at_utc"}:
                try:
                    clean[key] = utc(value).isoformat()
                except Refuse:
                    clean[key] = "INVALID"
            elif key == "status":
                clean[key] = value if value in {"complete", "under_target", "saturated_unresolved", "invalid"} else "UNKNOWN"
            elif key == "append_ready":
                clean[key] = value if type(value) is bool else "INVALID"
            else:
                # Nullable numeric pandas columns can be stored as Arrow double.
                # An integral finite double represents the original count exactly.
                if type(value) is int and value >= 0:
                    clean[key] = value
                elif type(value) is float and math.isfinite(value) and 0 <= value <= 2 ** 53 and value.is_integer():
                    clean[key] = int(value)
                else:
                    clean[key] = "INVALID"
        result.append(clean)
    return result


def classification_fields(value):
    result = {key: value.get(key) for key in CLASSIFICATION_FIELDS if key in value}
    if result.get("state") not in {"COMPLETE", "NOT_COMMITTED", "CONTRADICTION", "RECONCILIATION_REQUIRED"}:
        raise Refuse("UNKNOWN_CLASSIFICATION")
    for key in ("finalized", "legacy"):
        if type(result.get(key)) is not bool:
            raise Refuse("INVALID_CLASSIFICATION")
    tx = result.get("transaction_id")
    if tx is not None and not JOURNAL_NAME.fullmatch(tx + ".json"):
        raise Refuse("INVALID_TRANSACTION_ID")
    fingerprint = result.get("resulting_production_fingerprint")
    if fingerprint is not None and not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise Refuse("INVALID_FINGERPRINT")
    if result.get("reason") is not None and result["reason"] not in {
        "gate_without_journal", "multiple_authorities", "manifest_differs_from_journal",
        "transition_without_gate", "unsafe_partition", "authority_without_journal", "reserved_target_without_gate"}:
        raise Refuse("UNKNOWN_CLASSIFICATION_REASON")
    if result.get("journal_outcome") is not None and result["journal_outcome"] not in {
        "ACTIVE", "UNPUBLISHED_FAILURE", "PUBLISHED", "PUBLISHED_DURABILITY_UNCERTAIN",
        "PUBLISHED_RECONCILIATION_REQUIRED", "COMPLETE"}:
        raise Refuse("UNKNOWN_JOURNAL_OUTCOME")
    return result


def descriptor_targets():
    """Linux runtime check; local macOS tests additionally instrument fd opens."""
    root = Path("/proc/self/fd")
    if not root.is_dir():
        return None
    targets = {}
    for name in os.listdir(root):
        try:
            targets[int(name)] = os.readlink(root / name)
        except FileNotFoundError:
            pass  # enumeration's own already-closed directory fd.
    return targets


def assert_descriptors_closed(layout, before):
    after = descriptor_targets()
    if after is None:
        return "UNAVAILABLE_LOCAL_PLATFORM"
    remaining = [target for target in after.values() if target.startswith(str(layout.authority) + "/")
                 and target != str(layout.lock)]
    if remaining or (before is not None and set(after) - set(before)):
        raise Refuse("DESCRIPTOR_LEAK")
    return "PASS"


def ap6_locked(layout, representatives, physical_nights, legacy_candidates, *, history, classify):
    """Only release authority semantics; no publisher/controller objects."""
    result = {}
    with history.authoritative_read(layout.data, wait_seconds=LOCK_WAIT_SECONDS) as token:
        before = descriptor_targets()
        classifications, hashes = {}, {}
        for day in representatives:
            value = classification_fields(classify(layout.data, layout.journals, day))
            classifications[day] = value
            if value["state"] != "COMPLETE" or not value["finalized"]:
                raise Refuse("UNFINALIZED_OR_UNPUBLISHED_NIGHT")
            hashes[day] = sha256_file(layout.partition(day) / "manifest.json")
        if not classifications[representatives[-1]]["legacy"]:
            raise Refuse("LEGACY_CANDIDATE_NOT_LEGACY")
        summary_hash = sha256_file(layout.summary)
        summary = parquet_rows(layout.summary, SUMMARY_COLUMNS)
        index = parquet_rows(layout.index, INDEX_COLUMNS)
        classes = Counter()
        for row in summary["rows"]:
            if row.get("date_utc") not in legacy_candidates:
                continue
            a, b = row.get("started_at_utc"), row.get("finished_at_utc")
            category = "ABSENT" if any(column not in row for column in ("started_at_utc", "finished_at_utc")) else (
                "NULL" if a is None or b is None else "INVALID" if "INVALID" in {a, b} else
                "ORDERED" if utc(a) <= utc(b) else "REVERSED")
            classes[category] += 1
        absent = sorted(set(physical_nights) - set(index["nights"])) if index["aggregation_status"] == "COMPLETE" else None
        result = {"classifications": classifications, "manifest_sha256": hashes,
                  "representative_roles": {"june_27": representatives[0], "newest_v3": representatives[1],
                                           "legacy": representatives[2]},
                  "nightly_summary_sha256": summary_hash, "nightly_summary": summary,
                  "loci_index_generation": list(token[0]) if token[0] else None, "loci_index": index,
                  "legacy_metadata_candidate_time_evidence_classes": dict(classes),
                  "legacy_distribution_scope": "SUMMARY_OF_METADATA_CANDIDATES; THREE_NIGHT_CLASSIFICATION_SAMPLE",
                  "partition_candidates_without_index_rows": absent,
                  "confirmed_published_representatives_without_index_rows":
                      [day for day in representatives if absent is not None and day in absent],
                  "zero_row_interpretation": "ABSENCE_FROM_INDEX_ALONE_DOES_NOT_PROVE_EXCLUSION",
                  "acquisition_time_reliability": "COMPARE_SUMMARY_TIMES_WITH_INDEX_MIN_MAX; POLICY_UNKNOWN",
                  "descriptor_check_before_release": assert_descriptors_closed(layout, before)}
        scan_forbidden(result)
    # Nothing is serialized unless the context's generation/gate exit check passes.
    return result


def verify_installed_release():
    from importlib import metadata as distributions
    from importlib.util import find_spec
    if (Path(sys.prefix) != RELEASE / "venv" or sys.version_info[:3] != (3, 11, 16)
            or distributions.version("antares-analysis") != VERSION):
        raise Refuse("IMMUTABLE_RELEASE_IDENTITY")
    for name, expected in (("RELEASE_SHA", BASELINE), ("PACKAGE_VERSION", VERSION)):
        if read_small(RELEASE / name, 256).decode("ascii").strip() != expected:
            raise Refuse("IMMUTABLE_RELEASE_IDENTITY")
    module_root = Path(find_spec("src").origin).parent
    real_path(module_root)
    if not inside(module_root, RELEASE / "venv") or source_digest(module_root) != SOURCE_DIGEST:
        raise Refuse("INSTALLED_SOURCE_IDENTITY")
    return module_root


def landlock_abi():
    if sys.platform != "linux" or platform.machine() != "x86_64":
        return "UNKNOWN"
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(444, 0, 0, 1)
    return result if result >= 1 else "UNAVAILABLE"


def kernel_read_boundary(read_paths, *, directory_only=()):
    """Worker-only Linux/x86_64 Landlock read allow-list and socket syscall denial.

    No namespaces, filesystem changes, helper binary, or fallback. Unsupported
    kernels refuse before any production content read or lock acquisition.
    """
    import ctypes
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise Refuse("KERNEL_BOUNDARY_UNAVAILABLE")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(444, 0, 0, 1)  # landlock_create_ruleset(VERSION)
    if abi < 3:  # Older ABIs cannot prohibit native truncate(2).
        raise Refuse("KERNEL_BOUNDARY_UNAVAILABLE")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    # ABI1 access bits 0..12; ABI2 REFER, ABI3 TRUNCATE. Deny every write bit.
    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    rules = Ruleset(handled)
    descriptor = libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if descriptor < 0:
        raise Refuse("KERNEL_BOUNDARY_UNAVAILABLE")
    try:
        for path in sorted(set(read_paths) | set(directory_only)):
            path = real_path(path)
            fd = os.open(path, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                allowed = (1 << 3) if path in directory_only else (
                    (1 << 2) | ((1 << 3) if path.is_dir() else 0))  # READ_FILE, READ_DIR
                rule = PathRule(allowed, fd)
                if libc.syscall(445, descriptor, 1, ctypes.byref(rule), 0) != 0:
                    raise Refuse("KERNEL_BOUNDARY_UNAVAILABLE")
            finally:
                os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, descriptor, 0):
            raise Refuse("KERNEL_BOUNDARY_UNAVAILABLE")
    finally:
        os.close(descriptor)

    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(Filter))]

    # seccomp_data.arch then nr. Refuse other ABIs (including x32).
    deny = 0x00050000 | 1  # SECCOMP_RET_ERRNO | EPERM
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E), (0x06, 0, 0, deny),
                    (0x20, 0, 0, 0), (0x35, 0, 1, 0x40000000), (0x06, 0, 0, deny)]
    # socket through getsockopt, socketpair, plus sendmmsg/recvmmsg/io_uring.
    for syscall in list(range(41, 56)) + [57, 58, 59, 288, 299, 307, 322, 425, 426, 427]:
        instructions.extend([(0x15, 0, 1, syscall), (0x06, 0, 0, deny)])
    # No subprocess can inherit flock. Native pthread creation remains allowed.
    instructions.extend([(0x15, 0, 1, 435), (0x06, 0, 0, 0x00050000 | 38),  # clone3 => ENOSYS
                         (0x15, 0, 3, 56), (0x20, 0, 0, 16),  # clone flags, args[0]
                         (0x45, 1, 0, 0x00010000), (0x06, 0, 0, deny),  # require CLONE_THREAD
                         (0x20, 0, 0, 0)])
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    array = (Filter * len(instructions))(*(Filter(*row) for row in instructions))
    program = Program(len(array), array)
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise Refuse("KERNEL_BOUNDARY_UNAVAILABLE")
    return {"landlock_abi": abi, "native_socket_syscalls": "DENIED", "native_filesystem_writes": "DENIED",
            "native_child_processes": "DENIED"}


def send_worker(value):
    scan_forbidden(value)
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    if len(raw) > MAX_PACKET_BYTES:
        raise Refuse("PACKET_LIMIT")
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def worker_main():
    sys.dont_write_bytecode = True
    request = json.loads(sys.stdin.buffer.read(MAX_JSON_BYTES + 1))
    layout = Layout(**{key: Path(value) for key, value in request["layout"].items()})
    sys.addaudithook(ReadBoundary(layout, worker=True))
    try:
        module_root = verify_installed_release()
        for key in list(os.environ):
            if key.startswith("ANTARES_"):
                del os.environ[key]
        # Static import paths only. Production files are individually allow-listed.
        readable = [Path(path) for path in sys.path if path and Path(path).is_dir()]
        readable += [Path(sys.base_prefix), RELEASE / "venv", Path(f"/proc/{os.getpid()}/fd")]
        for path in ("/usr", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/localtime", "/dev/null", "/dev/urandom"):
            candidate = Path(path)
            if candidate.exists():
                # System library aliases are resolved only for kernel rules, never production.
                readable.append(candidate.resolve())
        for path in entries(layout.journals):
            if metadata(path)["bytes"] > MAX_JSON_BYTES:
                raise Refuse("JOURNAL_SIZE_LIMIT")
            if path.suffix != ".json":
                raise Refuse("UNEXPECTED_JOURNAL")
            readable.append(path)
        if sum(metadata(path)["bytes"] for path in readable if inside(path, layout.journals)) > MAX_JOURNAL_TOTAL_BYTES:
            raise Refuse("JOURNAL_TOTAL_SIZE_LIMIT")
        readable += [layout.lock, layout.index, layout.summary]
        for day in request["representatives"]:
            manifest = layout.partition(day) / "manifest.json"
            if metadata(manifest)["bytes"] > MAX_JSON_BYTES:
                raise Refuse("MANIFEST_SIZE_LIMIT")
            readable.append(manifest)
        # Journal.load opens its parent dir_fd. Listing/opening that directory
        # does not grant content access to newly appearing or unlisted files.
        boundary = kernel_read_boundary(readable, directory_only=(layout.journals,))
        from src import authority, history
        from src.operations.publication import classify_night_authority
        if (Path(history.__file__).parent != module_root or history.publication_gate_path(layout.data) != layout.gate
                or history.cumulative_paths(layout.data)["loci_index"] != layout.index
                or authority.authority_lock_path(layout.data) != layout.lock):
            raise Refuse("RELEASE_PATH_CONTRADICTION")
        measured = {}
        original = authority.fcntl.flock

        def timed_flock(fd, operation):
            started = time.monotonic_ns()
            original(fd, operation)
            if operation & authority.fcntl.LOCK_SH:
                measured["acquired_ns"] = started
                send_worker({"phase": "lock_acquired", "at_ns": started})
            elif operation == authority.fcntl.LOCK_UN:
                measured["released_ns"] = time.monotonic_ns()
                send_worker({"phase": "lock_released", "at_ns": measured["released_ns"]})

        authority.fcntl.flock = timed_flock
        try:
            result = ap6_locked(layout, request["representatives"], request["physical_nights"],
                                request["legacy_candidates"], history=history, classify=classify_night_authority)
        finally:
            authority.fcntl.flock = original
        result["hold_seconds"] = (measured["released_ns"] - measured["acquired_ns"]) / 1e9
        result["hold_measurement"] = "MONOTONIC_FLOCK_SYSCALL_ENVELOPE"
        result["kernel_boundary"] = boundary
        send_worker({"phase": "result", "status": "PASS", "evidence": result})
    except BaseException as error:
        if type(error.__cause__).__name__ == "AuthorityLockUnavailable":
            code = "LOCK_ACQUISITION_TIMEOUT"
        else:
            code = str(error) if isinstance(error, Refuse) else type(error).__name__
        send_worker({"phase": "result", "status": "REFUSE", "refusal": code})


def supervise(command, request, *, hard_seconds=HARD_SECONDS):
    """Outer process owns deadline; child alone holds flock; never retries."""
    if not 0 < hard_seconds <= HARD_SECONDS:
        raise Refuse("INVALID_DEADLINE")
    started = time.monotonic()
    acquired = released = None
    buffer, received, packet = b"", 0, None
    outgoing = json.dumps(request, allow_nan=False).encode()
    if len(outgoing) > MAX_JSON_BYTES:
        raise Refuse("WORKER_REQUEST_LIMIT")
    offset = 0
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, close_fds=True)
    def timeout_result():
        kill_at = time.monotonic()
        child.kill()
        child.wait()
        return {"status": "HOLD_TIMEOUT", "worker_killed": True,
                "worker_returncode": child.returncode,
                "worker_lifetime_seconds": time.monotonic() - started,
                "kill_reap_margin_seconds": time.monotonic() - kill_at,
                "hold_seconds": (released - acquired) / 1e9 if released and acquired else None,
                "hold_upper_bound_seconds": (time.monotonic_ns() - acquired) / 1e9 if acquired else None,
                "hold_measurement": "EXACT_ENVELOPE_IF_RELEASED; OTHERWISE_KILL_REAP_UPPER_BOUND",
                "hard_deadline_seconds": hard_seconds}

    try:
        # Nonblocking input: even a child that never reads its request is killed.
        os.set_blocking(child.stdin.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            selector.register(child.stdin, selectors.EVENT_WRITE)
            while selector.get_map():
                remaining = hard_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    return timeout_result()  # SIGKILL, then reap; never retries.
                ready = selector.select(min(remaining, 0.1))
                for key, _ in ready:
                    if key.fileobj is child.stdin:
                        offset += os.write(child.stdin.fileno(), outgoing[offset:offset + 65536])
                        if offset == len(outgoing):
                            selector.unregister(child.stdin)
                            child.stdin.close()
                        continue
                    raw = os.read(key.fileobj.fileno(), 65536)
                    if not raw:
                        selector.unregister(key.fileobj)
                        continue
                    received += len(raw)
                    if received > MAX_PACKET_BYTES:
                        raise Refuse("WORKER_PACKET_LIMIT")
                    buffer += raw
                    while b"\n" in buffer:
                        line, _, buffer = buffer.partition(b"\n")
                        value = json.loads(line)
                        scan_forbidden(value)
                        phase = value.get("phase")
                        if phase in {"lock_acquired", "lock_released"}:
                            if set(value) != {"phase", "at_ns"} or type(value["at_ns"]) is not int:
                                raise Refuse("INVALID_WORKER_MESSAGE")
                            if phase == "lock_acquired":
                                if acquired is not None:
                                    raise Refuse("INVALID_WORKER_MESSAGE")
                                acquired = value["at_ns"]
                            else:
                                if acquired is None or released is not None or value["at_ns"] < acquired:
                                    raise Refuse("INVALID_WORKER_MESSAGE")
                                released = value["at_ns"]
                        elif phase == "result" and packet is None:
                            packet = value
                        else:
                            raise Refuse("INVALID_WORKER_MESSAGE")
        remaining = hard_seconds - (time.monotonic() - started)
        try:
            child.wait(timeout=max(0, remaining))
        except subprocess.TimeoutExpired:
            return timeout_result()
        if buffer or child.returncode or packet is None:
            raise Refuse("WORKER_FAILED")
        packet.pop("phase")
        packet["worker_lifetime_seconds"] = time.monotonic() - started
        packet["hard_deadline_seconds"] = hard_seconds
        if acquired is not None and released is not None:
            packet["hold_seconds"] = (released - acquired) / 1e9
            packet["hold_measurement"] = "MONOTONIC_FLOCK_SYSCALL_ENVELOPE"
        return packet
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        for handle in (child.stdin, child.stdout):
            if handle and not handle.closed:
                handle.close()


def run_tier_b(layout, discovery, safety, acknowledged, *, supervisor=supervise):
    if not acknowledged:
        raise Refuse("CONTROL_ACKNOWLEDGEMENT_REQUIRED")
    if safety.get("tier_b_safe_to_attempt") is not True or not all(safety.get(key) is True for key in (
            "gate_clear", "generation_stable", "no_live_controller", "no_recent_nonterminal_activity")):
        return {"status": "REFUSE", "refusal": "AP0_UNSAFE"}
    if discovery["AP-1"]["tier_b_uid_eligible"] is not True:
        return {"status": "INELIGIBLE", "refusal": "WRONG_LOCK_UID"}
    inventory = discovery["AP-3"]
    newest, legacy = inventory["newest_v3_candidate"], inventory["legacy_candidate"]
    if not newest or not legacy:
        return {"status": "REFUSE", "refusal": "MISSING_REPRESENTATIVE_NIGHT"}
    representatives = ["2026-06-27", newest, legacy]  # roles may share the first two paths.
    request = {"layout": layout.plain(), "representatives": representatives,
               "physical_nights": inventory["physical_nights"],
               "legacy_candidates": [day for day in inventory["physical_nights"]
                                     if day < "2026-06-27" and day not in inventory["v3_candidates_from_journal_names"]]}
    script = str(Path(__file__).resolve())
    command = [sys.executable, "-I", "-B", "-c", "import runpy,sys; runpy.run_path(sys.argv[1], run_name='__worker__')", script]
    return supervisor(command, request)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    tiers = result.add_mutually_exclusive_group(required=True)
    tiers.add_argument("--plan", action="store_true", help="static paths and code identity only")
    tiers.add_argument("--tier-a", action="store_true", help="AP-0 then AP-1 through AP-5 discovery")
    tiers.add_argument("--tier-b", action="store_true", help="fresh discovery and AP-0 then supervised AP-6")
    result.add_argument("--output-dir", required=True, help="absolute private evidence directory outside protected roots")
    result.add_argument("--control-authorized-tier-b", action="store_true", help="accidental-use barrier; requires separate Control authorization")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.control_authorized_tier_b and not args.tier_b:
        raise Refuse("ACKNOWLEDGEMENT_REQUIRES_TIER_B")
    if args.tier_b and not args.control_authorized_tier_b:
        raise Refuse("CONTROL_ACKNOWLEDGEMENT_REQUIRED")
    output = output_root(args.output_dir)
    root = Path(__file__).resolve().parent.parent
    identity = verify_repository(root)
    layout = static_layout(root)
    sys.dont_write_bytecode = True
    sys.addaudithook(ReadBoundary(layout, output, plan=args.plan))
    document = {"schema": SCHEMA, "identity": identity, "paths": layout.plain(),
                "observed_at_utc": datetime.now(timezone.utc).isoformat()}
    if args.plan:
        tier = "plan"
        document["status"] = "NOT_EXECUTED"
        document["limits"] = {"observation_seconds": OBSERVATION_SECONDS, "lock_wait_seconds": LOCK_WAIT_SECONDS,
                              "hard_worker_seconds": HARD_SECONDS, "inventory_items": MAX_ITEMS,
                              "event_tail_bytes": MAX_EVENTS_BYTES}
        document["policy_questions"] = [{"question": question, "answer": "UNKNOWN"} for question in POLICY_QUESTIONS]
    else:
        tier = "b" if args.tier_b else "a"
        document["AP-0-before-A"] = ap0(layout)
        discovery = tier_a(layout, output)
        document.update(discovery)
        if args.tier_b:
            safety = ap0(layout)  # fresh final gate, no discovery in between it and the worker.
            document["AP-0-before-B"] = safety
            document["AP-6"] = run_tier_b(layout, discovery, safety, args.control_authorized_tier_b)
    path = write_evidence(output, tier, document)
    print(str(path))
    return 0 if not args.tier_b or document["AP-6"]["status"] == "PASS" else 1


if __name__ == "__worker__":
    worker_main()
elif __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        code = str(error) if isinstance(error, Refuse) else type(error).__name__
        print(json.dumps({"status": "REFUSE", "refusal": code}), file=sys.stderr)
        raise SystemExit(1)
