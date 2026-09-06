"""One-way, unpublished recovery of the sealed June 27 acquisition.

There is deliberately no provider, live-read capability, query callback,
fetch callback, publication transaction, or checkpoint writer in this path.
The source binding remains 0.4.1; only the consuming qualification is 0.4.2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

CONTRACT = "phase6f.offline-recovery.0.4.1-to-0.4.2.v1"
SOURCE_VERSION = "0.4.1"
CONSUMER_VERSION = "0.4.2"
SOURCE_SHA = "6f68e7b3955bbda08d5d6c5e2319d26cd7d4e829"
SOURCE_RUN_ID = "phase6d-live-0.4.1-6f68e7b-20260627-20260903T201059Z"
CANARY_ROOT = Path("/astro/store/shire/ANTARES/work/canary")
SOURCE_ROOT = CANARY_ROOT / SOURCE_RUN_ID
RELEASES_ROOT = Path("/astro/users/mdarim/opt/antares-analysis/releases")
DATA_ROOT = Path("/astro/store/shire/ANTARES/data")
CACHE_ROOT = Path("/astro/store/shire/ANTARES/cache")
NIGHT = "2026-06-27"
MJD = (61218.0, 61219.0)
OBJECTS = 331786
ALERTS = 509431
SEGMENTS = 1297
QUERY_ID = "fb92b26d77eb4baae26a2550cea3c8dbc7d60ef585c1843b80733207831c0054"
QUERY_POLICY = "1f828f93fcb88729690c15bab9af09423841b18051e7136a8bb5a5a210476d05"
CONFIGURATION = "f5de205b56706aeed3f8f58280ae4bd9fa483656b82e41b5d701faff60711e95"
FETCH_POLICY = "bfe06131cbd0a6ce86f7649ccb96bda6e1d5292e7f7546396960108da9cb5442"
QUERY_CONTRACT = "e9927901681136d3129a07b476119e1f13e199caa6c90e3c27ff569d0270ee1c"
QUERY_ORDER = "634e2846d5602145be7e223861631e47fdba99fb9f984d8e51dcd31841d3465d"
FETCH_ID = "4df4086c49b698141ed73d0796ad658a42e20c12b4306a1e97752cc4caa240af"
PRODUCTION_SENTINEL = "90673ce201b8bc7b439ffc58ddee8750728dc07e35d9004c053e18b5e4aaf2a3"
PRIOR_HASH = "f75196d18690e610ab6e79231b244c3fddca396a68eea08dd2d0408e91d8b587"
SUMMARY_HASH = "85c5fac9c242fa2e7993155036ada649336b0affe8ffc8843d2c5733ea765114"
TIMEOUT_SECONDS = 14400
PINNED = {
    "antares-client": "1.14.0", "astropy": "6.0.1", "matplotlib": "3.9.4",
    "numpy": "1.26.4", "pandas": "2.3.3", "pyarrow": "21.0.0",
}


class OfflineRecoveryError(RuntimeError):
    """A recovery identity, integrity, or confinement gate failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OfflineRecoveryError(message)


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _real(path: Path) -> Path:
    _require(path.is_absolute() and path.resolve(strict=True) == path, "Symlink or path alias refused.")
    return path


def _read(path: Path) -> Mapping[str, Any]:
    _real(path)
    _require(stat.S_ISREG(path.lstat().st_mode), "Evidence must be a regular file.")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON key refused.")
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    _require(isinstance(value, dict), "Evidence must be a JSON object.")
    return value


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def source_identity(root: Path) -> Mapping[str, Any]:
    """Hash all preserved bytes and stable metadata, excluding access times."""
    _real(root)
    entries = []
    for path in [root, *sorted(root.rglob("*"))]:
        before = path.lstat()
        _require(stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode), "Source contains a link or special file.")
        _real(path)
        digest = _hash(path) if stat.S_ISREG(before.st_mode) else None
        after = path.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        _require(identity(before) == identity(after), "Source changed while hashing.")
        entries.append([str(path.relative_to(root)), *identity(before), digest])
    return {"sha256": _digest(_json(entries)), "entries": len(entries), "inventory": entries}


def validate_release_pair(source_version: str, source_sha: str, consumer_version: str, consumer_sha: str) -> None:
    _require(
        (source_version, source_sha, consumer_version) == (SOURCE_VERSION, SOURCE_SHA, CONSUMER_VERSION)
        and re.fullmatch(r"[0-9a-f]{40}", consumer_sha) is not None
        and consumer_sha != SOURCE_SHA,
        "Only the exact 0.4.1 source to installed 0.4.2 recovery is authorized.",
    )


def release_environment(consumer_sha: str) -> Mapping[str, Any]:
    """Check the immutable release markers, wheel bytes, and loaded origins."""
    import zipfile
    import importlib.util
    _require(sys.dont_write_bytecode and os.environ.get("PIP_NO_INDEX") == "1" and not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME"), "Offline release environment is not sanitized.")
    version = metadata.version("antares-analysis")
    validate_release_pair(SOURCE_VERSION, SOURCE_SHA, version, consumer_sha)
    _require(sys.version_info[:3] == (3, 11, 16), "Arnor requires CPython 3.11.16.")
    root = _real(RELEASES_ROOT / consumer_sha)
    _require(root.stat().st_mode & 0o222 == 0, "Consuming release is not frozen.")
    source_release = _real(RELEASES_ROOT / SOURCE_SHA)
    for release, sha, expected_version in ((root, consumer_sha, CONSUMER_VERSION), (source_release, SOURCE_SHA, SOURCE_VERSION)):
        _require((release / "RELEASE_SHA").read_text().strip() == sha, "Release SHA marker differs.")
        _require((release / "PACKAGE_VERSION").read_text().strip() == expected_version, "Release version marker differs.")
    _require(Path(os.path.abspath(sys.executable)).parent == root / "venv/bin", "Executable is outside the consuming release.")
    _require(Path(sys.prefix) == root / "venv", "Python prefix is outside the consuming release.")
    wheels = list((root / "artifacts").glob("*.whl"))
    _require(len(wheels) == 1, "Exactly one retained release wheel is required.")
    wheel = _real(wheels[0])
    wheel_hash = _hash(wheel)
    _require(wheel_hash == (root / "WHEEL_SHA256").read_text().strip(), "Release wheel hash differs.")
    distribution = metadata.distribution("antares-analysis")
    package_root = Path(distribution.locate_file("src")).resolve(strict=True)
    _require(root in package_root.parents, "Distribution origin escapes the release.")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("src/") and name.endswith(".py"):
                installed = _real(Path(distribution.locate_file(name)))
                _require(root in installed.parents and installed.read_bytes() == archive.read(name), "Installed code differs from retained wheel.")
                _require(installed.stat().st_mode & 0o222 == 0, "Installed release code is writable.")
    dependency_origins = {}
    for name in ("numpy", "pandas", "pyarrow", "astropy", "matplotlib"):
        spec = importlib.util.find_spec(name)
        _require(spec is not None and spec.origin is not None, "Scientific import origin is missing.")
        origin = _real(Path(spec.origin))
        _require(root in origin.parents, "Scientific dependency origin escapes the release.")
        dependency_origins[name] = str(origin)
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == "src" or name.startswith("src."):
            origin = getattr(module, "__file__", None)
            if origin:
                path = _real(Path(origin))
                _require(package_root == path.parent or package_root in path.parents, "Loaded module comes from a checkout.")
                origins[name] = str(path)
    versions = {name: metadata.version(name) for name in PINNED}
    _require(versions == PINNED, "Pinned scientific dependencies differ.")
    _require(not any(name == "antares_client" or name.startswith("antares_client.") for name in sys.modules), "ANTARES client was initialized.")
    return {"release_sha": consumer_sha, "version": version, "python": sys.version.split()[0], "executable": sys.executable, "wheel_sha256": wheel_hash, "origins": origins, "dependency_origins": dependency_origins, "versions": versions}


def process_identity(root: Path) -> Mapping[str, Any]:
    _require(os.getcwd() == "/", "Detached recovery cwd must be /.")
    _require(os.environ.get("TMPDIR") == str(root / "tmp"), "Temporary path is outside recovery.")
    descriptors = {}
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
            info = Path("/proc/self/fdinfo", entry.name).read_text()
        except FileNotFoundError:
            continue
        flags = int(next(line.split()[1] for line in info.splitlines() if line.startswith("flags:")), 8)
        if flags & os.O_ACCMODE:
            path = Path(target)
            _require(path.is_absolute() and root in path.parents and path.resolve(strict=True) == path, "Unsafe inherited write descriptor.")
        descriptors[entry.name] = {"target": target, "flags": flags}
    _require(descriptors.get("0", {}).get("target") == "/dev/null", "Recovery stdin must be /dev/null.")
    return {"pid": os.getpid(), "ppid": os.getppid(), "session": os.getsid(0), "process_group": os.getpgrp(), "cwd": "/", "descriptors": descriptors}


def validate_source_metadata(query_manifest: Mapping[str, Any], fetch_manifest: Mapping[str, Any]) -> None:
    bindings = query_manifest.get("bindings", {})
    _require(query_manifest.get("schema_version") == "phase6.query-result-checkpoint.v1", "Query schema differs.")
    _require(query_manifest.get("content_integrity_sha256") == QUERY_ID, "Query identity differs.")
    expected = {"run_id": SOURCE_RUN_ID, "release_sha": SOURCE_SHA, "configuration_hash": CONFIGURATION, "target_date_utc": NIGHT, "provider_name": "live-antares", "provider_scenario": "commissioning-v1", "query_policy_sha256": QUERY_POLICY}
    _require(all(bindings.get(key) == value for key, value in expected.items()), "Query originating binding differs.")
    _require(_digest(_json(bindings.get("query_policy"))) == QUERY_POLICY, "Query policy differs.")
    request = query_manifest.get("scientific_request", {})
    _require(request.get("date_utc") == NIGHT and request.get("mjd_min") == MJD[0] and request.get("mjd_max") == MJD[1] and request.get("lsst_only") is True and request.get("query_tag") is None and request.get("target_loci") is None, "Night, interval, or query policy differs.")
    expected_fetch = {"run_id": SOURCE_RUN_ID, "release_sha": SOURCE_SHA, "configuration_sha256": CONFIGURATION, "target_date_utc": NIGHT, "mjd_min": MJD[0], "mjd_max": MJD[1], "provider_name": "live-antares", "provider_scenario": "commissioning-v1", "provider_policy_sha256": FETCH_POLICY, "query_contract_sha256": QUERY_CONTRACT, "query_identity_sha256": QUERY_ID, "query_locus_order_sha256": QUERY_ORDER, "expected_objects": OBJECTS, "segment_size": 256}
    _require(fetch_manifest.get("schema_version") == "phase6.segmented-fetch-checkpoint.v1" and fetch_manifest.get("binding") == expected_fetch and fetch_manifest.get("checkpoint_identity_sha256") == FETCH_ID, "Fetch originating binding differs.")


def checkpoint_layout() -> None:
    parent = _real(SOURCE_ROOT / "checkpoints")
    _require({entry.name for entry in parent.iterdir()} == {"query-result", "live-fetch-v1"}, "Unexpected or partial checkpoint state.")
    temporary = _real(parent / "live-fetch-v1/tmp")
    _require(not any(temporary.iterdir()), "Temporary fetch residue is present.")


def production_snapshot() -> Mapping[str, Any]:
    from .commissioning import capture_production_sentinel, establish_target_eligibility
    from ..history import cumulative_paths
    sentinel = capture_production_sentinel(DATA_ROOT, CACHE_ROOT)
    _require(sentinel["fingerprint_sha256"] == PRODUCTION_SENTINEL and sentinel["target_absent"] and sentinel["cache_absent"] and sentinel["transaction_artifacts"] == [], "Production baseline, target absence, or cache absence differs.")
    _require(sentinel["durable_file_count"] == 324 and sentinel["durable_bytes"] == 1141241743, "Production inventory differs.")
    paths = cumulative_paths(DATA_ROOT)
    _require(_hash(paths["loci_index"]) == PRIOR_HASH and _hash(paths["nightly_summary"]) == SUMMARY_HASH, "Cumulative identity differs.")
    eligibility = establish_target_eligibility(DATA_ROOT)
    _require(eligibility["passed"] and eligibility["authoritative_manifest_count"] == 90 and eligibility["total_loci"] == 993218 and eligibility["total_alerts"] == 13579707, "Production science baseline differs.")
    return {"sentinel": sentinel, "eligibility": eligibility}


def _root(run_id: str) -> Path:
    _require(re.fullmatch(r"phase6f-recovery-0[.]4[.]2-[A-Za-z0-9._-]+", run_id) is not None, "Unsafe recovery run id.")
    _require(run_id != SOURCE_RUN_ID, "Recovery cannot use the source root.")
    return _real(CANARY_ROOT) / run_id


class OfflineGuard:
    """Process-lifetime network and filesystem confinement using audit events."""

    def __init__(self, destination: Path, *, writable: bool = True):
        self.destination = _real(destination)
        self.writable = writable
        self.counts = {"network_attempts": 0, "client_import_attempts": 0, "query_callbacks": 0, "fetch_callbacks": 0, "provider_initializations": 0, "outside_write_attempts": 0, "forbidden_process_attempts": 0}
        self.allowed_subprocess = None

    def _write_path(self, value) -> None:
        if isinstance(value, int):
            raise OfflineRecoveryError("Descriptor-based filesystem mutation refused.")
        path = Path(os.fsdecode(value)).absolute()
        resolved = path.resolve(strict=False)
        if not self.writable or (resolved != self.destination and self.destination not in resolved.parents) or resolved != path:
            self.counts["outside_write_attempts"] += 1
            raise OfflineRecoveryError("Write outside the exact qualification root refused.")

    def audit(self, event, args) -> None:
        if event in {"socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.sendto"}:
            self.counts["network_attempts"] += 1
            raise OfflineRecoveryError("Network access refused during offline recovery.")
        if event == "import" and (args[0] == "antares_client" or args[0].startswith("antares_client.")):
            self.counts["client_import_attempts"] += 1
            raise OfflineRecoveryError("ANTARES client import refused.")
        if event == "subprocess.Popen":
            if self.allowed_subprocess is None or list(args[1]) != self.allowed_subprocess:
                self.counts["forbidden_process_attempts"] += 1
                raise OfflineRecoveryError("Unexpected child process refused.")
            self.allowed_subprocess = None
        if event in {"os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty"}:
            self.counts["forbidden_process_attempts"] += 1
            raise OfflineRecoveryError("Unexpected process creation refused.")
        if event == "open":
            path, mode, flags = args
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                if isinstance(path, int):
                    # fdopen receives only descriptors already opened through this hook.
                    return
                self._write_path(path)
            elif not isinstance(path, int):
                lexical = Path(os.fsdecode(path)).absolute()
                if SOURCE_ROOT == lexical or SOURCE_ROOT in lexical.parents:
                    _require(lexical.resolve(strict=True) == lexical, "Source read follows a symlink.")
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown", "os.truncate", "os.utime"}:
            self._write_path(args[0])
        if event in {"os.rename", "os.link", "os.symlink"}:
            for path in args[:2]:
                self._write_path(path)
            if event in {"os.link", "os.symlink"}:
                raise OfflineRecoveryError("Links are unavailable during recovery.")

    def install(self) -> None:
        from .live_antares import LiveAntaresProvider
        # These process-local circuit breakers also measure attempted entry.
        # Installing them constructs no provider and grants no live capability.
        def denied(counter):
            def refuse(*args, **kwargs):
                self.counts[counter] += 1
                raise OfflineRecoveryError("Live entry point refused during recovery.")
            return refuse
        for name, counter in (
            ("__init__", "provider_initializations"),
            ("query", "query_callbacks"),
            ("fetch", "fetch_callbacks"),
            ("fetch_resumable", "fetch_callbacks"),
            ("fetch_night", "fetch_callbacks"),
            ("_load_client", "client_import_attempts"),
        ):
            setattr(LiveAntaresProvider, name, denied(counter))
        sys.addaudithook(self.audit)


def prepare(run_id: str, consumer_sha: str) -> Mapping[str, Any]:
    environment = release_environment(consumer_sha)
    _require(os.uname().nodename.split(".")[0] == "arnor" and os.getuid() == SOURCE_ROOT.stat().st_uid, "Arnor source ownership gate failed.")
    root = _root(run_id)
    _require(not root.exists() and not root.is_symlink(), "Recovery destination already exists.")
    production = production_snapshot()
    checkpoint_layout()
    query_manifest = _read(SOURCE_ROOT / "checkpoints/query-result/manifest.json")
    fetch_manifest = _read(SOURCE_ROOT / "checkpoints/live-fetch-v1/checkpoint.json")
    validate_source_metadata(query_manifest, fetch_manifest)
    source = source_identity(SOURCE_ROOT)
    resources = os.statvfs(CANARY_ROOT)
    memory = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in Path("/proc/meminfo").read_text().splitlines()}
    _require(resources.f_bavail * resources.f_frsize >= 20 * 1024**3 and resources.f_favail >= 10000, "Insufficient recovery disk space or inodes.")
    _require(int(memory["MemAvailable"].split()[0]) * 1024 >= 16 * 1024**3, "Insufficient recovery memory.")
    root.mkdir(mode=0o700)
    for name in ("logs", "status", "evidence", "candidate", "tmp", "launcher"):
        (root / name).mkdir(mode=0o700)
    binding = {"schema_version": CONTRACT, "run_id": run_id, "run_root": str(root), "source_root": str(SOURCE_ROOT), "source_version": SOURCE_VERSION, "source_sha": SOURCE_SHA, "consumer_version": CONSUMER_VERSION, "consumer_sha": consumer_sha, "night": NIGHT, "mjd": list(MJD), "query_identity": QUERY_ID, "fetch_identity": FETCH_ID, "authoritative": False, "publishable": False, "publication_authorized": False, "timeout_seconds": TIMEOUT_SECONDS, "term_grace_seconds": 60, "source_identity": source["sha256"], "production_sentinel": PRODUCTION_SENTINEL, "environment": environment, "resources": {"free_bytes": resources.f_bavail * resources.f_frsize, "free_inodes": resources.f_favail, "memory": memory}}
    _write_new(root / "evidence/source-before.json", _json(source))
    _write_new(root / "evidence/production-before.json", _json(production))
    _write_new(root / "binding.json", _json(binding))
    _write_new(root / "binding.sha256", (_hash(root / "binding.json") + "\n").encode())
    return binding


def _binding(root: Path, consumer_sha: str) -> Mapping[str, Any]:
    _real(root)
    _require(root == _root(root.name) and stat.S_IMODE(root.stat().st_mode) == 0o700 and root.stat().st_uid == os.getuid(), "Recovery root identity differs.")
    binding = _read(root / "binding.json")
    _require(_hash(root / "binding.json") == (root / "binding.sha256").read_text().strip(), "Recovery binding seal differs.")
    expected = {"schema_version": CONTRACT, "run_id": root.name, "run_root": str(root), "source_root": str(SOURCE_ROOT), "source_version": SOURCE_VERSION, "source_sha": SOURCE_SHA, "consumer_version": CONSUMER_VERSION, "consumer_sha": consumer_sha, "night": NIGHT, "mjd": list(MJD), "query_identity": QUERY_ID, "fetch_identity": FETCH_ID, "authoritative": False, "publishable": False, "publication_authorized": False, "timeout_seconds": TIMEOUT_SECONDS, "production_sentinel": PRODUCTION_SENTINEL}
    _require(all(binding.get(key) == value for key, value in expected.items()), "Recovery contract binding differs.")
    return binding


def reconstruct(root: Path, consumer_sha: str) -> Mapping[str, Any]:
    """Claim a prepared destination once, reopen all source data, and qualify."""
    import pandas as pd
    from .. import history
    from .fetch_checkpoint import FetchCheckpointBinding, SegmentedFetchCheckpoint
    from .query_checkpoint import QueryResultCheckpointBindings, load_query_result_checkpoint
    from .science import FetchStageEvidence, NightScienceRequest, NightScienceResult, ProviderOutcome, _result_evidence, build_night_artifacts, reopen_and_validate_artifacts
    environment = release_environment(consumer_sha)
    binding = _binding(root, consumer_sha)
    process = process_identity(root)
    _require(not any((root / "candidate").iterdir()), "Candidate destination is not empty.")
    _write_new(root / "status/worker-claim.json", _json({"consumer_sha": consumer_sha, **process}))
    guard = OfflineGuard(root)
    guard.install()
    final = {"schema_version": CONTRACT, "run_id": root.name, "success": False, "authoritative": False, "publishable": False, "publication_attempted": False, "environment": environment}
    stage = "source_reproof"
    def progress(name, **details):
        nonlocal stage
        stage = name
        with (root / "logs/progress.jsonl").open("ab") as stream:
            stream.write(_json({"stage": name, "utc": datetime.now(timezone.utc).isoformat(), **details}))
            stream.flush()
            os.fsync(stream.fileno())
    try:
        progress(stage)
        source_before = source_identity(SOURCE_ROOT)
        _require(source_before["sha256"] == binding["source_identity"], "Preserved source changed since preparation.")
        production_snapshot()
        checkpoint_layout()
        query_manifest = _read(SOURCE_ROOT / "checkpoints/query-result/manifest.json")
        fetch_manifest = _read(SOURCE_ROOT / "checkpoints/live-fetch-v1/checkpoint.json")
        validate_source_metadata(query_manifest, fetch_manifest)
        prior = history.load_cumulative_loci_index(DATA_ROOT, before_mjd=MJD[0], before_date=NIGHT)
        prior_ids = tuple(prior["locus_id"].dropna().astype(str).tolist())
        request = NightScienceRequest(NIGHT, *MJD, target_loci=None, range_label="ANTARES commissioning 2026-06-27", prior_locus_ids=prior_ids)
        values = dict(query_manifest["bindings"])
        values.pop("query_policy_sha256")
        progress("query_checkpoint_reopen")
        loaded = load_query_result_checkpoint(SOURCE_ROOT, request, QueryResultCheckpointBindings(**values))
        _require(loaded.integrity_sha256 == QUERY_ID and len(loaded.query_result.loci) == OBJECTS, "Reopened query identity differs.")
        query_result = loaded.query_result
        ids = query_result.loci["locus_id"].astype(str).tolist()
        checkpoint = SegmentedFetchCheckpoint.open_read_only(SOURCE_ROOT, FetchCheckpointBinding(**fetch_manifest["binding"]))
        progress("fetch_checkpoint_validation")
        completion = checkpoint.inspect_complete(ids)
        _require((completion.segment_count, completion.completed_objects, completion.alert_rows, completion.reused_segments, completion.fetched_segments) == (SEGMENTS, OBJECTS, ALERTS, SEGMENTS, 0), "Completed acquisition totals differ.")
        frames = []
        with_rows = 0
        for index, segment in enumerate(checkpoint.iter_segments(ids)):
            frames.append(segment.alerts)
            with_rows += sum(int(value["alert_rows"]) > 0 for value in segment.objects)
            if index % 100 == 0:
                progress("fetch_checkpoint_reopen", segments_reopened=index + 1, segments_total=SEGMENTS)
        raw_alerts = pd.concat(frames, ignore_index=True, sort=False)
        del frames
        _require(len(raw_alerts) == ALERTS, "Reopened alert count differs.")
        progress("science_preparation")
        loci = history.prepare_loci(query_result.loci, NIGHT, *MJD, query_result.evidence.details["request_completed_at_utc"], source_query_mode="probe_first_time_ra_dec")
        alerts = history.prepare_alerts(raw_alerts, NIGHT, request.range_label)
        details = {"completion_classification": "COMPLETE_NONZERO", "requested_objects": OBJECTS, "completed_objects": OBJECTS, "failed_objects": 0, "failed_object_identity_sha256": _digest(b""), "failure_exception_types": [], "retry_exception_types": list(completion.retry_exception_types), "retry_count": completion.retry_count, "lightcurves_with_rows": with_rows, "lightcurves_empty": OBJECTS - with_rows, "full_locus_history_requests": OBJECTS, "full_locus_history_completed": OBJECTS, "alert_rows": ALERTS, "max_workers": 4, "effective_workers": min(4, OBJECTS), "max_in_flight_futures": min(OBJECTS, min(4, OBJECTS) * 4), "max_attempts_per_object": 3, "cache_used": False, "secret_material_recorded": False, "checkpoint": completion.as_dict(), "offline_recovery": {"contract": CONTRACT, "new_history_requests": 0, "source_root": str(SOURCE_ROOT)}}
        fetch = FetchStageEvidence(True, False, len(loci), len(alerts), (), details)
        evidence = _result_evidence(query_result.evidence, fetch)
        validation = history.validation_summary(loci, alerts, mjd_min=MJD[0], mjd_max=MJD[1], prior_locus_ids=request.prior_locus_ids, lsst_only=True, query_completed=True, query_fetch_clean=evidence.clean, mjd_upper_exclusive=True)
        _require(validation.get("append_ready") is True, "Scientific validation failed.")
        result = NightScienceResult(request, query_result.provider_name, query_result.scenario, ProviderOutcome.SUCCESS, query_result, loci, alerts, fetch, validation, (), evidence)
        progress("artifact_build")
        artifacts = build_night_artifacts(result)
        manifest = json.loads(artifacts["manifest.json"])
        manifest.update({"authoritative": False, "publishable": False, "publication_authorized": False, "offline_recovery_contract": CONTRACT})
        artifacts["manifest.json"] = _json(manifest)
        progress("artifact_reopen")
        reopen_and_validate_artifacts(artifacts, expected=result)
        artifact_hashes = {}
        for name, payload in artifacts.items():
            _write_new(root / "candidate" / name, payload)
            artifact_hashes[name] = {"sha256": _digest(payload), "bytes": len(payload)}
        _write_new(root / "evidence/artifacts.json", _json(artifact_hashes))
        progress("independent_reopen")
        command = [sys.executable, "-m", "src.operations.offline_recovery", "audit", "--run-id", root.name, "--release-sha", consumer_sha]
        guard.allowed_subprocess = command
        with open("/dev/null", "rb") as null_input:
            child = subprocess.run(command, stdin=null_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/", timeout=1800, check=False)
        _write_new(root / "logs/independent-reopen.stderr", child.stderr)
        _require(child.returncode == 0, "Independent artifact reopen failed.")
        reopened = json.loads(child.stdout)
        _require(reopened["success"] and reopened["artifacts"] == artifact_hashes, "Independent artifact evidence differs.")
        _write_new(root / "evidence/independent-reopen.json", _json(reopened))
        final.update({"success": True, "status": "RECOVERY_COMPLETE_UNPUBLISHED", "query_checkpoint_reused": True, "fetch_checkpoint": completion.as_dict(), "validation": validation, "artifacts": artifact_hashes})
    except Exception as error:
        final.update({"success": False, "status": "BLOCKED", "stage": stage, "error_type": type(error).__name__, "error_code": getattr(error, "code", type(error).__name__)})
    finally:
        try:
            source_after = source_identity(SOURCE_ROOT)
            _write_new(root / "evidence/source-after.json", _json(source_after))
            final["source_before_sha256"] = binding["source_identity"]
            final["source_after_sha256"] = source_after["sha256"]
            _require(source_after["sha256"] == binding["source_identity"], "Source mutation detected.")
        except Exception as error:
            final.update({"success": False, "status": "BLOCKED", "source_invariant_error_type": type(error).__name__})
        try:
            after = production_snapshot()
            _write_new(root / "evidence/production-after.json", _json(after))
            final["production_before_sha256"] = PRODUCTION_SENTINEL
            final["production_after_sha256"] = after["sentinel"]["fingerprint_sha256"]
        except Exception as error:
            final.update({"success": False, "status": "BLOCKED", "production_invariant_error_type": type(error).__name__})
        final["callback_and_network_counts"] = dict(guard.counts)
        if any(guard.counts.values()):
            final.update({"success": False, "status": "BLOCKED"})
        final["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        final["evidence_inventory"] = {
            str(path.relative_to(root)): {"sha256": _hash(_real(path)), "bytes": path.stat().st_size}
            for path in sorted((root / "evidence").iterdir()) if path.is_file()
        }
        final["binding_sha256"] = _hash(root / "binding.json")
        _write_new(root / "status/RECOVERY_FINAL.json", _json(final))
    return final


def audit(root: Path, consumer_sha: str) -> Mapping[str, Any]:
    from .science import reopen_and_validate_artifacts
    environment = release_environment(consumer_sha)
    _binding(root, consumer_sha)
    guard = OfflineGuard(root, writable=False)
    guard.install()
    expected = _read(root / "evidence/artifacts.json")
    _require(set(expected) == {"loci.parquet", "alerts.parquet", "manifest.json"}, "Artifact inventory differs.")
    _require({p.name for p in (root / "candidate").iterdir()} == set(expected), "Unexpected candidate entries.")
    payloads = {name: _real(root / "candidate" / name).read_bytes() for name in expected}
    observed = {name: {"bytes": len(payload), "sha256": _digest(payload)} for name, payload in payloads.items()}
    _require(observed == expected, "Artifact hashes differ.")
    reopened = reopen_and_validate_artifacts(payloads)
    _require(len(reopened.loci) == OBJECTS and len(reopened.alerts) == ALERTS, "Independent scientific counts differ.")
    manifest = json.loads(payloads["manifest.json"])
    _require(manifest.get("authoritative") is False and manifest.get("publishable") is False and manifest.get("publication_authorized") is False and manifest.get("offline_recovery_contract") == CONTRACT, "Qualification publication boundary differs.")
    return {"schema_version": CONTRACT, "success": True, "artifacts": observed, "validation": manifest["validation"], "loci": len(reopened.loci), "alerts": len(reopened.alerts), "environment": environment, "callback_and_network_counts": guard.counts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args(argv)
    try:
        root = _root(args.run_id)
        if args.action == "prepare":
            result = prepare(args.run_id, args.release_sha)
        elif args.action == "run":
            result = reconstruct(root, args.release_sha)
        else:
            result = audit(root, args.release_sha)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("success", True) else 1
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error_type": type(error).__name__, "error_code": getattr(error, "code", type(error).__name__)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
