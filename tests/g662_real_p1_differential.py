"""Read-only qualification helper. Invoke only after local qualification PASS.

Runs under immutable A Python. Optional --candidate adds a private source copy;
no installation, capability issuance, query, fetch, controller, or publication.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import stat
import sys
import time

SHA = "812c545e14693cdce7ff7458f1d2b50b0804dcd8"
RELEASE = Path("/astro/users/mdarim/opt/antares-analysis/releases") / SHA
SOURCE = Path("/astro/store/shire/ANTARES/work/backfill/g66-live-2026-07-07_2026-07-13-v1")
FILES = ("live_antares.py", "science.py", "backfill.py", "production_range.py")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def snapshot(root):
    rows = []
    for path in [root] + sorted(root.rglob("*")):
        observed = path.lstat()
        assert not stat.S_ISLNK(observed.st_mode), "Source symlink"
        row = {"path": "." if path == root else path.relative_to(root).as_posix(),
               "mode": stat.S_IMODE(observed.st_mode), "uid": observed.st_uid, "gid": observed.st_gid,
               "inode": observed.st_ino, "device": observed.st_dev, "mtime_ns": observed.st_mtime_ns,
               "size": observed.st_size, "type": "directory" if stat.S_ISDIR(observed.st_mode) else "file"}
        stable = lambda value: (value.st_mode, value.st_uid, value.st_gid, value.st_ino,
                               value.st_dev, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if stat.S_ISREG(observed.st_mode):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            row["sha256"] = digest.hexdigest()
        else:
            assert stat.S_ISDIR(observed.st_mode), "Source special file"
        assert stable(observed) == stable(path.lstat()), "Source changed during inventory"
        rows.append(row)
    return {"algorithm": "g66.source_tree_sha256.v1.content_metadata_no_atime",
            "sha256": hashlib.sha256(canonical(rows)).hexdigest(),
            "entries": len(rows), "files": sum(row["type"] == "file" for row in rows),
            "bytes": sum(row["size"] for row in rows if row["type"] == "file"), "inventory": rows}


def selection(request, filter_value):
    # Independent interpretation of already proved immutable P1 semantics.
    return {"schema_version": "v3.qualified-scientific-selection.v1", "date_utc": request.date_utc,
        "time": {"field": "properties.newest_alert_observation_time", "mjd_min": request.mjd_min,
                 "mjd_max": request.mjd_max, "lower": "inclusive", "upper": "exclusive", "timezone": "UTC"},
        "spatial": {"ra_field": "ra", "dec_field": "dec", "units": "degrees",
            "ra_min": 0.0, "ra_max": 360.0, "ra_lower": "inclusive", "ra_upper": "exclusive",
            "dec_min": -90.0, "dec_max": 90.0, "dec_lower": "inclusive", "dec_upper": "inclusive_at_90_only"},
        "lsst_filter": filter_value, "query_tag": None, "lsst_only": True, "target_loci": None, "prior_free": True,
        "normalization": "locus_to_record-properties-overlay;string-strip-nonblank-locus-id;tile-membership",
        "deduplication": {"key": "locus_id", "keep": "last", "scope": "accepted_tiles"},
        "input_order": "lower-child-first;within-leaf-qualified-service-order;keep-last;reset-index",
        "equivalence": "selection-semantics-only-not-service-snapshot"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--compare")
    parser.add_argument("--scratch", required=True)
    args = parser.parse_args()
    assert args.day in {"2026-07-%02d" % n for n in range(8,13)}
    assert Path(sys.prefix) == RELEASE / "venv", "Wrong immutable verifier Python"
    assert (RELEASE / "RELEASE_SHA").read_text().strip() == SHA
    assert (RELEASE / "PACKAGE_VERSION").read_text().strip() == "0.4.7"
    assert (RELEASE / "WHEEL_SHA256").read_text().strip() == "d81e1f754c6ed781f47fc4cab3d92e1966ef7fcdc6f39d6d23566bf52ecb3440"
    assert args.candidate is None or args.compare, "B requires A comparison"
    sys.dont_write_bytecode = True
    scratch = Path(args.scratch).resolve(strict=True)
    assert scratch.parent == Path("/astro/users/mdarim/tmp") and scratch.name.startswith("g662b-")
    # -B forbids writes but still permits reading installed bytecode. Force
    # imports to search an absent private prefix so the hashed source executes.
    cache_prefix = scratch / ("no-pyc-" + ("B" if args.candidate else "A") + "-" + args.day)
    assert not os.path.lexists(cache_prefix), "Bytecode prefix must be absent"
    sys.pycache_prefix = str(cache_prefix)
    blocked_writes = []
    def protected_path(value):
        if isinstance(value, int):
            value = os.readlink("/proc/self/fd/%d" % value)
        path = Path(os.fsdecode(value))
        if not path.is_absolute():
            raise AssertionError("AMBIGUOUS_RELATIVE_WRITE")
        resolved = path.resolve(strict=False)
        if resolved != scratch and scratch not in resolved.parents:
            blocked_writes.append(str(resolved))
            raise AssertionError("FORBIDDEN_FILESYSTEM_WRITE")
    def readonly_audit(event, values):
        if event == "open":
            path, mode, flags = values
            if (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                protected_path(path)
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime", "os.truncate"}:
            protected_path(values[0])
        elif event in {"os.rename", "os.link", "os.symlink"}:
            protected_path(values[0]); protected_path(values[1])
        elif event in {"subprocess.Popen", "os.system", "os.exec", "os.fork"}:
            raise AssertionError("FORBIDDEN_CHILD_PROCESS")
    sys.addaudithook(readonly_audit)
    if args.candidate:
        candidate = Path(args.candidate).resolve(strict=True)
        assert candidate != RELEASE and RELEASE not in candidate.parents
        sys.path.insert(0, str(candidate))
    import requests
    attempts = []
    def refuse(*_args, **_kwargs):
        attempts.append("network-or-live-callback")
        raise AssertionError("FORBIDDEN_NETWORK_CALLBACK")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    requests.sessions.Session.request = refuse
    client_search = importlib.import_module("antares_client.search")
    client_search.search = client_search.get_by_id = refuse
    from src.operations import backfill as B, production_range as R, science as S, query_checkpoint as Q
    all_identities = json.loads(Path(args.identities).read_bytes())
    side = "B" if args.candidate else "A"
    identities = all_identities[side]
    actual = {}
    for name in FILES:
        module = importlib.import_module("src.operations." + name[:-3])
        actual[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert actual == identities, "Verifier code identity differs"
    import src
    source_package = Path(src.__file__).parent
    tree_identities = all_identities[side + "_source_tree"]
    assert all(hashlib.sha256((source_package / name).read_bytes()).hexdigest() == expected
               for name, expected in tree_identities.items()), "Full verifier source tree differs"
    root = SOURCE / "nights" / ("night-" + args.day)
    assert root.resolve(strict=True) == root and B._adoptable_source_root(root, args.day)
    original = json.loads(Path(args.historical).read_bytes())[args.day]
    reference = json.loads(Path(args.compare).read_bytes()) if args.compare else None
    helper_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if reference:
        assert (reference["event"] == "PASS" and reference["verifier"] == "A"
                and reference["date_utc"] == args.day and reference["forced_source_import"] is True
                and reference["qualification_helper_sha256"] == helper_sha256
                and reference["implementation_sha256"] == all_identities["A"]
                and reference["network_callback_attempts"] == reference["blocked_write_attempts"] == 0)
    started = time.monotonic()
    before = snapshot(root)
    assert before == original["source_tree_before"], "Source differs from G6.6.1 inventory"
    if reference:
        assert before == reference["source_tree_after"], "Source changed between A and B"
    print(json.dumps({"event":"begin","verifier":"B" if args.candidate else "A","date_utc":args.day}), file=sys.stderr, flush=True)
    adapter = R.LiveRangeAdapter(SOURCE, SHA, refuse)
    saved = B.describe_saved_acquisition(root, args.day, adapter, 256)
    assert saved.entry == original["entry"], "Historical adoption entry differs"
    result = saved.loaded.query_result
    frame = result.loci
    details = dict(result.evidence.details)
    replay = S._phase6_replay_trace(details["tile_trace"], saved.request.mjd_min, saved.request.mjd_max)
    contract = adapter.scientific_contract(saved.request)
    frame_manifest = saved.loaded.manifest["frame"]
    columns, dtypes = list(frame.columns), [str(dtype) for dtype in frame.dtypes]
    records, semantic = hashlib.sha256(), hashlib.sha256()
    semantic.update(Q._semantic_header(columns, dtypes))
    for ordinal, values in enumerate(frame.itertuples(index=False, name=None)):
        line = Q._record_line(ordinal, values)
        records.update(line)
        semantic.update(line)
    assert records.hexdigest() == frame_manifest["records_sha256"]
    assert semantic.hexdigest() == frame_manifest["semantic_sha256"]
    order = hashlib.sha256("".join(str(value)+"\n" for value in frame.locus_id).encode()).hexdigest()
    assert order == details["locus_order_sha256"]
    trace = hashlib.sha256(canonical({"tiles":details["tile_trace"]})).hexdigest()
    assert trace == details["tile_trace_sha256"]
    descriptor = selection(saved.request, contract["lsst_filter"])
    if args.candidate:
        assert canonical(saved.selection_descriptor) == canonical(descriptor)
    compared = {"entry": dict(saved.entry), "request": B._request_document(saved.request),
        "selection": descriptor, "scientific_contract": contract,
        "frame": {"columns":columns,"dtypes":dtypes,"row_count":len(frame),
                  "records_sha256":records.hexdigest(),"semantic_sha256":semantic.hexdigest(),
                  "locus_order_sha256":order},
        "partition": replay, "tile_trace_sha256":trace,
        "query_evidence_sha256":saved.loaded.query_evidence_sha256,
        "query_integrity_sha256":saved.loaded.integrity_sha256,
        "query_evidence":saved.loaded.query_evidence,
        "fetch_binding":saved.binding.as_dict(), "fetch_completion":saved.completion.as_dict()}
    if reference:
        assert canonical(compared) == canonical(reference["compared"]), "A/B differential mismatch"
    after = snapshot(root)
    assert before == after and not attempts and not blocked_writes, "Source mutation or attempted network/write"
    assert not os.path.lexists(cache_prefix), "Bytecode prefix was unexpectedly created"
    output = {"event":"PASS","verifier":"B" if args.candidate else "A","date_utc":args.day,
        "implementation_sha256":actual, "compared":compared,
        "source_tree_before":before,"source_tree_after":after,
        "source_unchanged":True,"network_callback_attempts":0,
        "blocked_write_attempts":0,"write_guard":"audit-hook; private scratch only; relative writes refuse",
        "forced_source_import":True,"absent_bytecode_prefix":str(cache_prefix),
        "qualification_helper_sha256":helper_sha256,
        "verification_seconds":round(time.monotonic()-started,3)}
    print(json.dumps(output, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
