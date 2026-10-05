"""G6.6.3D process-isolated P2 transport boundary (offline; loopback only).

RB1: a G6.6.3A page thread blocked in name resolution or in urllib3's chunked
trailer loop could not be stopped.  Each P2 page's blocking HTTP exchange now
runs in a dedicated transport child process that the parent can always kill.
These tests qualify that boundary against real child processes:

* hard deadlines kill a child blocked before any socket exists and a child
  stuck reading trailers, prove it dead (reaped), and the bytes stop;
* the child reports exactly the body and charset ``requests`` derived, so the
  process paginator equals the in-process paginator and the pinned client;
* every IPC violation, early end of stream, crash, oversized frame or startup
  failure is a typed, terminal refusal that ends the P2 night incomplete;
* one child per night query, started lazily, never restarted after a failure;
* P1 never constructs, starts or references the transport;
* the child imports only the standard library and ``requests``.

Every HTTP exchange reaches a loopback server only.  ``G663D_PERF_OUT`` names
an optional JSON file for the measured overhead.
"""

import ast
import contextlib
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path
from unittest import mock

import v3_fixtures as F
from test_operations_phase6 import FakeLocus
from test_g663_transport_guard import (
    BASE, CANARY, ES_KEY, FIRST_TILE, GUARD, LISTING, NIGHT, RELEASE, ROOT, SORT, ArnorNight,
    FakeAntares, Loopback, OfflineTestCase, child_hook, drain, guarded, limits, listing_resource,
    listing_schema, pinned_search, process_guarded, send_json, tile_loci, transport_children)
from src.operations import live_antares as L, p2_transport as T, production_range as R, science as S
from src.operations import backfill as B
from src.operations import query_checkpoint as Q
from src.operations.science import ArtifactValidationError, ProviderOutcome, QueryInterruptedError

BODY = L._build_tile_query(FIRST_TILE)
SOURCE = ROOT / "src" / "operations" / "p2_transport.py"
EMPTY = {"data": [], "links": {"next": None}}


def close_quietly(child):
    with contextlib.suppress(T.TransportFailure):
        child.close()


def raw_transport(test, transport_limits=None, **hook_args):
    """One transport child (not yet started) with an explicit test hook."""
    child = T.P2ProcessTransport(L._p2_child_config(transport_limits or CANARY.transport, BASE),
                                 hook=child_hook(**hook_args))
    test.addCleanup(close_quietly, child)
    return child


def child_listing(test, transport_limits=None):
    """The production paginator over a child that uses the installed service hook."""
    transport_limits = transport_limits or CANARY.transport
    child = T.P2ProcessTransport(L._p2_child_config(transport_limits, BASE))
    test.addCleanup(close_quietly, child)
    return L._GuardedListing(transport_limits, BASE, child, listing_schema), child


def alive(pid):
    """True while ``pid`` exists and has not exited (a zombie has exited)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def rss_kib(pid, key="VmRSS"):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return None


def empty_listing(handler):
    send_json(handler, EMPTY)


class ProcessTestCase(OfflineTestCase):
    def serve(self, handle):
        server = Loopback(handle)
        self.addCleanup(server.close)
        return server

    def assertDead(self, child, *, spawns=1):
        """Reaped, gone from the process table, FAILED for good, never restarted."""
        record = child.exit_record
        self.assertEqual(child.state, T.FAILED)
        self.assertTrue(record["reaped"], record)
        self.assertIsNotNone(record["returncode"])
        with self.assertRaises(ProcessLookupError):
            os.kill(record["pid"], 0)
        self.assertEqual(transport_children(), [])
        with self.assertRaises(T.TransportFailure) as caught:
            child.fetch(LISTING, {"sort": SORT}, 1 << 20, "page_bytes", 30)
        self.assertEqual((caught.exception.kind, caught.exception.code), ("unavailable", T.FAILED))
        self.assertIs(type(L._p2_transport_error(caught.exception)), L.P2TransportUnavailableError)
        self.assertEqual(child.stats["spawns"], spawns)
        return record


class PreSocketTerminationTests(ProcessTestCase):
    """RB1 (A): a child blocked inside name resolution, before any socket, is killed."""

    def run_blocked(self, **hook):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        marker = Path(folder.name) / "marker"
        server = self.serve(empty_listing)
        listing = process_guarded(self, limits(iterator_deadline_seconds=1), server.port,
                                  block_resolver=True, marker=str(marker), **hook)
        started = time.monotonic()
        records, error = drain(listing.search(BODY))
        elapsed = time.monotonic() - started
        self.assertIs(type(error), L.P2DeadlineError)
        self.assertEqual(records, [])
        self.assertEqual(marker.read_text(), "resolver\n")  # it was blocked in the resolver
        self.assertEqual(server.paths, [])  # no connection, no request ever reached the service
        return listing._transport, elapsed

    def test_resolver_block_is_terminated_and_reaped(self):
        child, elapsed = self.run_blocked()
        record = self.assertDead(child)
        self.assertEqual((record["returncode"], record["escalated_to_sigkill"]), (-signal.SIGTERM, False))
        self.assertLess(elapsed, 2.0)

    def test_resolver_block_ignoring_sigterm_escalates_to_sigkill(self):
        child, elapsed = self.run_blocked(ignore_sigterm=True)
        record = self.assertDead(child)
        grace = limits().terminate_grace_seconds
        self.assertEqual((record["returncode"], record["escalated_to_sigkill"]), (-signal.SIGKILL, True))
        self.assertGreaterEqual(record["seconds"], grace)
        self.assertLess(record["seconds"], grace + limits().kill_join_seconds)
        self.assertLess(elapsed, 1.0 + grace + 1.5)


class TrailerTerminationTests(ProcessTestCase):
    """RB1 (B): a child stuck in urllib3's chunked-trailer loop is killed; the bytes stop."""

    def setUp(self):
        super().setUp()
        self.sent, self.ended = 0, None

    def trailers(self, handler):
        payload = json.dumps(EMPTY).encode()
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "application/vnd.api+json")
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            handler.wfile.write(b"%x\r\n%s\r\n0\r\n" % (len(payload), payload))
            handler.wfile.flush()
            for n in range(1_000_000):  # trailer lines, each well inside the read timeout, forever
                line = b"X-Trailer-%d: %s\r\n" % (n, b"t" * 64)
                handler.wfile.write(line)
                handler.wfile.flush()
                self.sent += len(line)
                time.sleep(.01)
        except OSError:
            self.ended = time.monotonic()
        handler.close_connection = True

    def test_unterminated_trailers_are_killed_and_reading_stops(self):
        server = self.serve(self.trailers)
        listing = process_guarded(self, limits(iterator_deadline_seconds=1), server.port)
        started = time.monotonic()
        records, error = drain(listing.search(BODY))
        killed = time.monotonic()
        at_kill = self.sent
        self.assertIs(type(error), L.P2DeadlineError)
        self.assertLess(killed - started, 2.0)
        self.assertEqual(records, [])  # the complete body never became a page
        self.assertGreater(at_kill, 0)  # the child was consuming trailers when it was killed
        record = self.assertDead(listing._transport)
        self.assertEqual(record["returncode"], -signal.SIGTERM)
        deadline = time.monotonic() + 3.0
        while self.ended is None and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertIsNotNone(self.ended, "the service kept writing to a dead reader")
        self.assertLess(self.ended - killed, 1.0)
        settled = self.sent
        time.sleep(1.0)
        one_second = self.sent
        time.sleep(1.0)
        self.assertEqual((one_second, self.sent), (settled, settled))
        self.assertLess(settled - at_kill, 4096)  # at most a few lines raced the kill


class ProcessEquivalenceTests(OfflineTestCase):
    """Process paginator == in-process paginator == pinned client, request for request."""

    def three_ways(self, service, *, stop_at=None):
        with service.installed():
            pinned, pinned_error = drain(pinned_search(BODY), stop_at=stop_at)
            pinned_urls = list(service.urls)
            service.calls.clear()
            in_process, in_process_error = drain(guarded().search(BODY), stop_at=stop_at)
            in_process_urls = list(service.urls)
            service.calls.clear()
            listing, child = child_listing(self)
            isolated, isolated_error = drain(listing.search(BODY), stop_at=stop_at)
            isolated_calls = list(service.calls)
            pages = service.pages
            listing.close()
        self.assertEqual((pinned_error, in_process_error, isolated_error), (None, None, None))
        self.assertEqual(isolated, pinned)
        self.assertEqual(isolated, in_process)
        self.assertEqual([call["url"] for call in isolated_calls], pinned_urls)
        self.assertEqual(in_process_urls, pinned_urls)
        for call in isolated_calls:  # the client's scalar 60 s, split exactly; streamed body
            self.assertEqual((call["timeout"], call["stream"]), ((60, 60), True))
        self.assertEqual((child.stats["spawns"], child.stats["requests"]), (1, pages))
        self.assertEqual((child.state, child.exit_record["returncode"]), (T.CLOSED, 0))
        self.assertEqual(transport_children(), [])
        return isolated, isolated_calls

    def test_single_page_equivalence(self):
        for count, terminal in ((0, "null"), (7, "null"), (7, "missing"), (7, "no-links"), (10, "null")):
            with self.subTest(count=count, terminal=terminal):
                loci = tile_loci(FIRST_TILE, count)
                records, calls = self.three_ways(FakeAntares(loci, page_size=10, terminal=terminal))
                self.assertEqual(len(calls), 1)
                self.assertEqual([r["locus_id"] for r in records], [l.locus_id for l in loci])

    def test_multi_page_equivalence(self):
        loci = tile_loci(FIRST_TILE, 130)
        for note in loci[::7]:  # payload variety: unicode, nested values, exact binary64
            note.properties = {**note.properties, "note": "ä→✓", "nested": {"f": 1e300 / 3}}
        for kwargs, pages in (({"page_size": 10}, 13), ({"page_size": 10, "full_page_next": True}, 14),
                              ({"page_size": 9, "gaps": {18, 54}}, 17), ({"page_size": 25, "redirect_offsets": {0, 50}}, 8),
                              ({"page_size": 64, "next_style": "port443"}, 3), ({"page_size": 64, "next_style": "upper"}, 3)):
            with self.subTest(**{k: sorted(v) if isinstance(v, set) else v for k, v in kwargs.items()}):
                records, calls = self.three_ways(FakeAntares(loci, **kwargs))
                self.assertEqual(len(calls), pages)
                self.assertEqual([r["locus_id"] for r in records], [l.locus_id for l in loci])

    def test_declared_charsets_decode_identically(self):
        loci = tile_loci(FIRST_TILE, 3)
        for loc in loci:
            loc.properties = {**loc.properties, "note": "Grüße ✓"}
        document = {"data": [listing_resource(l) for l in loci], "links": {"next": None}}
        for charset, encoding in (("utf-8", "utf-8"), ("utf-16", "utf-16"), (None, "utf-8")):
            with self.subTest(charset=charset):
                content_type = "application/vnd.api+json" + (f"; charset={charset}" if charset else "")
                spec = {"body": json.dumps(document, ensure_ascii=False).encode(encoding),
                        "headers": {"Content-Type": content_type}}
                records, _calls = self.three_ways(FakeAntares(script=lambda s, r, i, spec=spec: dict(spec)))
                self.assertEqual([r["locus_id"] for r in records], [l.locus_id for l in loci])

    def test_probe_of_50_rows_orders_and_stops_identically(self):
        for count, page_size, pages in ((60, 10, 5), (60, 7, 8), (51, 1, 50), (120, 64, 1)):
            with self.subTest(count=count, page_size=page_size):
                loci = tile_loci(FIRST_TILE, count, prefix=f"P{count}")
                loci.reverse()  # service order is not identifier order
                records, calls = self.three_ways(FakeAntares(loci, page_size=page_size), stop_at=50)
                self.assertEqual([r["locus_id"] for r in records], [l.locus_id for l in loci[:50]])
                self.assertEqual(len(calls), pages)
        records, calls = self.three_ways(FakeAntares(tile_loci(FIRST_TILE, 49), page_size=1, full_page_next=True),
                                         stop_at=50)
        self.assertEqual((len(records), len(calls)), (49, 50))

    def test_one_child_serves_every_page_of_every_search(self):
        tiles = L._make_initial_tiles(61218.0, 61219.0)[:3]
        loci = [locus for tile in tiles for locus in tile_loci(tile, 23, prefix=f"S{tile['ra_min']:g}")]
        service = FakeAntares(loci, page_size=10)
        with service.installed():
            listing, child = child_listing(self)
            pids = set()
            for tile in tiles:
                records, error = drain(listing.search(L._build_tile_query(tile)))
                self.assertIsNone(error)
                self.assertEqual(len(records), 23)
                pids.add(child.stats["pid"])
                self.assertEqual(transport_children(), [child.stats["pid"]])
            listing.close()
        self.assertEqual((len(pids), child.stats["spawns"], child.stats["requests"]), (1, 1, 9))
        self.assertEqual(transport_children(), [])


class LifecycleTests(ProcessTestCase):
    """NOT_STARTED -> RUNNING -> FAILED or CLOSED; lazy start; no restart; no orphan."""

    def test_lazy_start_persistence_and_clean_shutdown(self):
        server = self.serve(empty_listing)
        child = raw_transport(self, port=server.port)
        self.assertEqual((child.state, child.stats["spawns"], transport_children()), (T.NOT_STARTED, 0, []))
        first = child.fetch(LISTING, {"sort": SORT}, 1 << 20, "page_bytes", 30)
        pid = child.stats["pid"]
        self.assertEqual(first, (json.dumps(EMPTY).encode(), None))
        self.assertEqual((child.state, transport_children()), (T.RUNNING, [pid]))
        self.assertEqual(os.getsid(pid), pid)  # its own session: terminal signals never reach it
        for _ in range(3):
            self.assertEqual(child.fetch(LISTING, None, 1 << 20, "page_bytes", 30), first)
        self.assertEqual((child.stats["spawns"], child.stats["pid"], child.stats["requests"]), (1, pid, 4))
        started = time.monotonic()
        record = child.close()
        took = time.monotonic() - started
        self.assertEqual((child.state, record["returncode"], record["reaped"], record["escalated_to_sigkill"]),
                         (T.CLOSED, 0, True, False))
        self.assertLess(took, 1.0)
        self.assertEqual(transport_children(), [])
        self.assertIs(child.close(), record)  # idempotent
        with self.assertRaises(T.TransportFailure) as caught:
            child.fetch(LISTING, None, 1 << 20, "page_bytes", 30)
        self.assertEqual((caught.exception.kind, caught.exception.code), ("unavailable", T.CLOSED))
        self.assertEqual(child.stats["spawns"], 1)

    def test_close_before_start_spawns_nothing(self):
        child = raw_transport(self)
        self.assertIsNone(child.close())
        self.assertEqual((child.state, child.stats["spawns"], transport_children()), (T.CLOSED, 0, []))

    def test_child_runs_only_the_bound_file_with_no_inherited_descriptors(self):
        server = self.serve(empty_listing)
        child = raw_transport(self, port=server.port)
        child.fetch(LISTING, None, 1 << 20, "page_bytes", 30)
        pid = child.stats["pid"]
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
        flags = subprocess._args_from_interpreter_flags()
        self.assertEqual([part.decode() for part in argv[:len(flags) + 4]],
                         [sys.executable, *flags, "-c", T._BOOTSTRAP, str(SOURCE)])
        targets = sorted(os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd"))
        self.assertEqual(sum(target.startswith("pipe:") for target in targets), 2)  # command, result
        self.assertTrue(all(target == "/dev/null" or target.startswith("pipe:") for target in targets), targets)

    def test_concurrent_use_is_refused(self):
        server = self.serve(lambda handler: send_json(handler, EMPTY, delay=1.0))
        child = raw_transport(self, port=server.port)
        outcome = {}
        worker = threading.Thread(target=lambda: outcome.setdefault(
            "first", child.fetch(LISTING, None, 1 << 20, "page_bytes", 30)))
        worker.start()
        time.sleep(.3)
        with self.assertRaises(T.TransportFailure) as caught:
            child.fetch(LISTING, None, 1 << 20, "page_bytes", 30)
        worker.join()
        self.assertEqual((caught.exception.kind, caught.exception.code), ("unavailable", "concurrent_use"))
        self.assertEqual(outcome["first"][0], json.dumps(EMPTY).encode())
        self.assertEqual((child.state, child.stats["requests"]), (T.RUNNING, 1))

    def test_an_abandoned_transport_kills_its_child(self):
        server = self.serve(empty_listing)
        child = T.P2ProcessTransport(L._p2_child_config(CANARY.transport, BASE), hook=child_hook(server.port))
        child.fetch(LISTING, None, 1 << 20, "page_bytes", 30)
        pid, reference = child.stats["pid"], weakref.ref(child)
        del child
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(alive(pid))
        self.assertEqual(transport_children(), [])

    def test_an_orphaned_child_exits_when_its_parent_dies(self):
        server = self.serve(empty_listing)
        script = (
            "import os, sys\n"
            f"sys.path[:0] = [{str(ROOT)!r}]\n"
            "from src.operations import live_antares as L, p2_transport as T\n"
            "profile = L.g663_canary_p2_profile()\n"
            "child = T.P2ProcessTransport(L._p2_child_config(profile.transport, L.OFFICIAL_API_BASE_URL),\n"
            f"    hook={child_hook(server.port)!r})\n"
            "child.fetch(L.OFFICIAL_API_BASE_URL + 'loci', None, 1 << 20, 'page_bytes', 30)\n"
            "print(child.stats['pid'], flush=True)\n"
            "os._exit(0)  # no close, no finalizer: the child is orphaned\n")
        done = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        pid = int(done.stdout.split()[0])
        deadline = time.monotonic() + 5.0
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertFalse(alive(pid), "an orphaned transport child kept running")


class ChildFailureTests(ProcessTestCase):
    """Crashes, IPC violations, early EOF and oversized frames: typed, killed, terminal."""

    def broken(self, mode, *, after=0, transport_limits=None, expect="protocol", **extra):
        server = self.serve(empty_listing)
        child = raw_transport(self, transport_limits, port=server.port, misbehave=mode, after=after, **extra)
        for _ in range(after):
            self.assertEqual(child.fetch(LISTING, None, 1 << 20, "page_bytes", 30)[0], json.dumps(EMPTY).encode())
        with self.assertRaises(T.TransportFailure) as caught:
            child.fetch(LISTING, None, 1 << 20, "page_bytes", 30)
        self.assertEqual(caught.exception.kind, expect)
        return child, caught.exception

    def test_child_crash_and_signal_death(self):
        for mode, returncode in (("crash", 9), ("segfault", -signal.SIGSEGV)):
            for after in (0, 2):
                with self.subTest(mode=mode, after=after):
                    child, failure = self.broken(mode, after=after)
                    self.assertIs(type(L._p2_transport_error(failure)), L.P2TransportProtocolError)
                    self.assertEqual(self.assertDead(child)["returncode"], returncode)

    def test_malformed_foreign_and_out_of_order_frames(self):
        for mode in ("malformed", "bad_json", "unknown_kind", "extra_key", "unknown_code", "foreign_id",
                     "seq_gap", "complete_mismatch", "error_after_result"):
            with self.subTest(mode=mode):
                child, failure = self.broken(mode)
                self.assertIs(type(L._p2_transport_error(failure)), L.P2TransportProtocolError)
                record = self.assertDead(child)
                self.assertEqual((record["returncode"], record["escalated_to_sigkill"]), (-signal.SIGTERM, False))

    def test_end_of_stream_before_complete(self):
        for mode, returncode in (("eof", -signal.SIGTERM), ("partial_then_exit", 0)):
            with self.subTest(mode=mode):
                child, failure = self.broken(mode)
                self.assertIs(type(L._p2_transport_error(failure)), L.P2TransportProtocolError)
                self.assertEqual(self.assertDead(child)["returncode"], returncode)

    def test_oversized_frames_and_over_budget_bodies(self):
        for mode in ("oversized_header", "oversized_chunk", "over_budget"):
            with self.subTest(mode=mode):
                child, failure = self.broken(mode)
                self.assertIs(type(L._p2_transport_error(failure)), L.P2TransportProtocolError)
                self.assertEqual(self.assertDead(child)["returncode"], -signal.SIGTERM)

    def test_request_over_the_ipc_bound_is_refused_before_any_child(self):
        child = raw_transport(self, limits(ipc_max_header_bytes=4096))
        for url, budget, overflow in ((LISTING + "?p=" + "x" * 4096, 1 << 20, "page_bytes"),
                                      (LISTING, 0, "page_bytes"), (LISTING, 1, "teapot"), (None, 1, "page_bytes")):
            with self.subTest(budget=budget, overflow=overflow, url=None if url is None else len(url)):
                with self.assertRaises(T.TransportFailure) as caught:
                    child.fetch(url, None, budget, overflow, 30)
                self.assertEqual(caught.exception.kind, "request_bound")
                self.assertIs(type(L._p2_transport_error(caught.exception)), L.P2TransportProtocolError)
        self.assertEqual((child.state, child.stats["spawns"], transport_children()), (T.NOT_STARTED, 0, []))

    def test_reply_header_over_the_ipc_bound_is_a_typed_refusal(self):
        def handle(handler):
            payload = json.dumps(EMPTY).encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/vnd.api+json; charset=" + "x" * 5000)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
        server = self.serve(handle)
        child = raw_transport(self, limits(ipc_max_header_bytes=4096), port=server.port)
        with self.assertRaises(T.TransportFailure) as caught:
            child.fetch(LISTING, None, 1 << 20, "page_bytes", 30)
        self.assertEqual((caught.exception.kind, caught.exception.code), ("remote", "header_bound"))
        self.assertIs(type(L._p2_transport_error(caught.exception)), L.P2TransportProcessError)
        self.assertEqual(child.state, T.RUNNING)  # the healthy child reported it; the night still fails

    def test_unrecognised_remote_failures_are_terminal(self):
        child, failure = self.broken("unrecognised_exception", expect="remote")
        self.assertEqual((failure.code, failure.remote_type), ("exception", "builtins.OSError"))
        error = L._p2_transport_error(failure)
        self.assertIs(type(error), L.P2TransportProcessError)
        self.assertFalse(L._retryable_query_error(error))
        self.assertIn(L._exception_type(error), L._P2_TRANSPORT_GUARD_TYPES)
        self.assertEqual(child.state, T.RUNNING)

    def test_only_whitelisted_request_errors_cross_and_they_carry_no_text(self):
        known = L._p2_remote_request_errors()
        for name, kind in sorted(known.items()):
            with self.subTest(name=name):
                error = L._p2_transport_error(T.TransportFailure("remote", "exception", name))
                self.assertIs(type(error), kind)
                self.assertEqual(error.args, ())
        for name in ("test.Reset", "requests.exceptions.JSONDecodeError", "builtins.ValueError", None):
            with self.subTest(name=name):
                error = L._p2_transport_error(T.TransportFailure("remote", "exception", name))
                self.assertIs(type(error), L.P2TransportProcessError)
        for kind, error_type in (("deadline", L.P2DeadlineError), ("startup", L.P2TransportStartupError),
                                 ("protocol", L.P2TransportProtocolError), ("unkillable", L.P2TransportUnkillableError),
                                 ("unavailable", L.P2TransportUnavailableError), ("teapot", L.P2TransportProcessError)):
            self.assertIs(type(L._p2_transport_error(T.TransportFailure(kind))), error_type)


class DeadlineDuringTransferTests(ProcessTestCase):
    """The parent's deadline also bounds the IPC transfer itself."""

    def test_stalled_frame_and_endless_chunks_are_killed_at_the_deadline(self):
        for mode in ("stall_mid_frame", "slow_chunks"):
            with self.subTest(mode=mode):
                server = self.serve(empty_listing)
                listing = process_guarded(self, limits(iterator_deadline_seconds=1), server.port, misbehave=mode)
                started = time.monotonic()
                records, error = drain(listing.search(BODY))
                self.assertLess(time.monotonic() - started, 2.0)
                self.assertIs(type(error), L.P2DeadlineError)
                self.assertEqual(records, [])
                self.assertEqual(self.assertDead(listing._transport)["returncode"], -signal.SIGTERM)


class StartupFailureTests(ProcessTestCase):
    """The child must start and prove its identity within its bound, or it is killed."""

    def failed_start(self, transport_limits=None, remaining=600, **hook):
        child = raw_transport(self, transport_limits, **hook)
        started = time.monotonic()
        with self.assertRaises(T.TransportFailure) as caught:
            child.fetch(LISTING, None, 1 << 20, "page_bytes", remaining)
        return child, caught.exception, time.monotonic() - started

    def test_startup_hang_is_bounded_and_killed(self):
        child, failure, took = self.failed_start(limits(child_startup_seconds=1), startup="hang")
        self.assertEqual(failure.kind, "startup")
        self.assertIs(type(L._p2_transport_error(failure)), L.P2TransportStartupError)
        self.assertLess(took, 2.0)
        self.assertEqual(self.assertDead(child)["returncode"], -signal.SIGTERM)

    def test_the_iterator_deadline_binds_during_startup(self):
        child, failure, took = self.failed_start(remaining=1, startup="hang")
        self.assertEqual(failure.kind, "deadline")
        self.assertIs(type(L._p2_transport_error(failure)), L.P2DeadlineError)
        self.assertLess(took, 2.0)
        self.assertDead(child)

    def test_child_exit_or_science_import_or_identity_mismatch_refuses_startup(self):
        for name, hook, patch, returncode in (
                ("exit", {"startup": "exit"}, None, 17),
                ("imports-src", {"startup": "import_src"}, None, -signal.SIGTERM),
                ("identity", {}, mock.patch.object(T, "implementation_sha256", lambda: "0" * 64), -signal.SIGTERM)):
            with self.subTest(name):
                with patch or contextlib.nullcontext():
                    child, failure, _took = self.failed_start(**hook)
                self.assertEqual(failure.kind, "startup")
                self.assertIs(type(L._p2_transport_error(failure)), L.P2TransportStartupError)
                self.assertEqual(self.assertDead(child)["returncode"], returncode)

    def test_spawn_failure_is_a_startup_refusal(self):
        with mock.patch.object(T.subprocess, "Popen", side_effect=OSError("no executable")):
            child, failure, _took = self.failed_start()
        self.assertEqual(failure.kind, "startup")
        self.assertEqual((child.state, child.exit_record["pid"], child.stats["spawns"]), (T.FAILED, None, 0))
        self.assertEqual(transport_children(), [])


class ProcessFailClosedNightTests(OfflineTestCase):
    """Process failures inside whole P2 nights: incomplete, no science, replayable offline."""

    first = tile_loci(FIRST_TILE, 25)

    def night(self, child, exception_type, rows, requests_made, transport_limits=None, patches=()):
        profile = CANARY if transport_limits is None else L.P2ProofProfile(**{
            **{k: getattr(CANARY, k) for k in ("max_depth", "max_nodes_per_root", "max_nodes_per_night",
                                               "max_search_attempts", "crash_reserve", "max_event_bytes")},
            "transport": transport_limits})
        night = ArnorNight(self)
        adapter = R.LiveRangeAdapter(night.root, RELEASE, None, proof_profile=profile)
        request = adapter.acquisition_request(NIGHT)
        service = FakeAntares(self.first, page_size=10)
        created = []

        class Recording(T.P2ProcessTransport):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)
        with contextlib.ExitStack() as stack:
            stack.enter_context(service.installed(**child))
            stack.enter_context(mock.patch.object(T, "P2ProcessTransport", Recording))
            for patch in patches:
                stack.enter_context(patch)
            result = night.provider(profile).query(request)
        self.assertEqual(len(service.calls), requests_made)
        # One child for the night, never restarted.  (Loading the client for its
        # identity builds a paginator too; its transport never starts.)
        started = [transport for transport in created if transport.stats["spawns"]]
        self.assertEqual([transport.stats["spawns"] for transport in started], [1])
        self.assertTrue(all(transport.state == T.NOT_STARTED for transport in created if transport not in started))
        self.assertIn(started[0].state, (T.FAILED, T.CLOSED))
        self.assertTrue(started[0].exit_record["reaped"] or exception_type.endswith("UnkillableError"))
        self.assertEqual(transport_children(), [])
        self.assertFalse(result.clean)
        self.assertEqual(result.outcome, ProviderOutcome.QUERY_INTERRUPTION)
        self.assertEqual([issue.code for issue in result.evidence.errors], ["p2_transport_guard"])
        details = result.evidence.details
        self.assertEqual((details["completion_classification"], details["terminal_evidence"]),
                         ("INCOMPLETE", "p2_transport_guard"))
        events = details["p2_events"]
        outcomes = [event["p2_event"] for event in events if event["p2_event"]["kind"] == "outcome"]
        self.assertEqual(len(outcomes), 1)
        decision = outcomes[0]["decision"]
        self.assertEqual((outcomes[0]["node"]["id"], decision["status"], decision["exception_type"],
                          decision["validated_rows"], decision["discarded_rows"], decision["reason"]),
                         ("i00000", "attempt_error", exception_type, rows, rows, "p2_transport_guard"))
        self.assertTrue(result.loci.empty)
        with self.assertRaises(QueryInterruptedError):
            result.require_completed()
        with self.assertRaises(ArtifactValidationError):
            S.validate_p2_query_result(request, result, profile)
        bindings = B.query_checkpoint_bindings(night.run_id, RELEASE, B.configuration_sha256(RELEASE, adapter, 256),
                                               adapter, request)
        with self.assertRaises(Q.QueryCheckpointError):
            Q.seal_query_result_checkpoint(night.root, result, bindings)
        # The journal alone replays the refusal: no request, no child.
        service.calls.clear()
        with service.installed(), mock.patch.object(T.P2ProcessTransport, "_start",
                                                    side_effect=AssertionError("replay started a child")):
            replayed = adapter.replay_query_journal(request, events)
        self.assertEqual(service.calls, [])
        for key in ("p2_events", "p2_budget", "terminal_evidence", "retry_exception_types", "search_request_count"):
            self.assertEqual(replayed.evidence.details[key], details[key], key)
        self.assertFalse(replayed.clean)
        return started[0]

    def test_every_process_failure_ends_the_night_incomplete(self):
        deadline = limits(iterator_deadline_seconds=1)

        def unprovable(process, grace, join):
            return {**real_stop(process, grace, join), "reaped": False}
        real_stop = T._stop_process
        for name, child, exception_type, rows, requests_made, transport_limits, patches in (
                ("resolver-deadline", {"block_resolver": True}, "P2DeadlineError", 0, 0, deadline, ()),
                ("ipc-stall-deadline", {"misbehave": "stall_mid_frame"}, "P2DeadlineError", 0, 0, deadline, ()),
                ("crash", {"misbehave": "crash"}, "P2TransportProtocolError", 0, 0, None, ()),
                ("crash-after-a-valid-page", {"misbehave": "crash", "after": 1}, "P2TransportProtocolError",
                 10, 1, None, ()),
                ("malformed-ipc", {"misbehave": "malformed"}, "P2TransportProtocolError", 0, 0, None, ()),
                ("eof-before-complete", {"misbehave": "partial_then_exit"}, "P2TransportProtocolError", 0, 0, None, ()),
                ("oversized-ipc", {"misbehave": "oversized_chunk"}, "P2TransportProtocolError", 0, 0, None, ()),
                ("startup-exit", {"startup": "exit"}, "P2TransportStartupError", 0, 0, None, ()),
                ("unrecognised-remote-failure", {"misbehave": "unrecognised_exception"}, "P2TransportProcessError",
                 0, 0, None, ()),
                ("death-not-provable", {"block_resolver": True}, "P2TransportUnkillableError", 0, 0, deadline,
                 (mock.patch.object(T, "_stop_process", unprovable),))):
            with self.subTest(name):
                self.night(child, GUARD + exception_type, rows, requests_made, transport_limits, patches)

    def test_a_child_that_cannot_be_proven_dead_at_close_fails_the_query(self):
        # A healthy child reports a terminal failure, so the night ends with the child
        # still running; closing it must then prove it dead, or the query itself fails.
        real_stop = T._stop_process
        night = ArnorNight(self)
        service = FakeAntares(self.first, page_size=10)
        with service.installed(misbehave="unrecognised_exception"), mock.patch.object(
                T, "_stop_process", lambda *a: {**real_stop(*a), "reaped": False}):
            with self.assertRaises(L.P2TransportUnkillableError):
                night.provider().query(night.request)
        self.assertEqual(transport_children(), [])  # (it was in fact reaped; only the proof was withheld)


class P1ProcessFirewallTests(OfflineTestCase):
    """P1 never constructs, starts or references the process transport (G6.6.3D P1 firewall)."""

    def test_p1_night_never_reaches_the_transport_or_spawns_a_process(self):
        night = ArnorNight(self)
        loci = tile_loci(FIRST_TILE, 25) + [FakeLocus(f"P1-{n}", mjd=61218.6, ra=30.0 + 40 * n, dec=-10.0 + 9 * n)
                                            for n in range(3)]
        service = FakeAntares(loci, page_size=10)
        refuse = AssertionError("P1 reached the P2 process transport")
        with service.installed(), \
                mock.patch.object(T, "P2ProcessTransport", side_effect=refuse), \
                mock.patch.object(T.subprocess, "Popen", side_effect=refuse), \
                mock.patch.object(T, "fetch_page", side_effect=refuse):
            p1 = L.LiveAntaresProvider(night.capability, sleeper=lambda _: None, clock=F.fixed_clock,
                                       monotonic=lambda: 0.0)
            identity = p1.client_identity()
            result = p1.query(night.request)
            fetched = p1.fetch(night.request, result)
        self.assertTrue(result.clean, result.evidence.errors)
        self.assertTrue(fetched.fetch_evidence.clean)
        self.assertEqual(identity["pagination_contract"], "jsonapi-links-next-until-null")
        self.assertNotIn("transport_sha256", identity)
        self.assertEqual(len(result.loci), len(loci))
        self.assertEqual(transport_children(), [])

    def test_only_p2_definitions_reference_the_transport_module(self):
        source = (ROOT / "src" / "operations" / "live_antares.py").read_text()
        users = set()
        for node in ast.parse(source).body:
            if any(isinstance(n, ast.Name) and n.id == "_p2_transport" for n in ast.walk(node)):
                users.add(node.name)
        self.assertEqual(users, {"P2TransportLimits", "_GuardedListing", "_load_p2_client"})
        provider = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                        and n.name == "LiveAntaresProvider")
        names = {n.id for n in ast.walk(provider) if isinstance(n, ast.Name)}
        self.assertFalse(names & {"_p2_transport", "_GuardedListing", "P2ProcessTransport"})
        self.assertIn("_load_p2_client", names)  # reached only from the P2-only branch (G663FirewallTests)


class ChildBoundaryTests(unittest.TestCase):
    """The child's code can reach nothing but the standard library and ``requests``."""

    def test_transport_module_imports_and_opens_only_what_it_must(self):
        source = SOURCE.read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "relative import: the src package")
                imported.add(node.module)
        self.assertEqual(imported, {"__future__", "hashlib", "json", "os", "re", "selectors", "signal", "struct",
                                    "subprocess", "sys", "threading", "time", "weakref", "urllib.parse", "requests",
                                    "importlib.util"})
        identifiers = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                       | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})
        self.assertFalse({"pickle", "marshal", "shelve", "multiprocessing", "copyreg", "eval"}
                         & (imported | identifiers))
        opened = [ast.get_source_segment(source, n) for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"]
        self.assertEqual(opened, ['open(os.path.abspath(__file__), "rb")'])
        self.assertIsNone(T.CHILD_TEST_HOOK)  # no production path ever sets the test seam
        self.assertNotIn("CHILD_TEST_HOOK", (ROOT / "src" / "operations" / "live_antares.py").read_text())

    def test_transport_identity_binds_the_child_bytes(self):
        digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        self.assertEqual(T.implementation_sha256(), digest)
        self.assertEqual(CANARY.transport.as_dict()["child_implementation_sha256"], digest)
        self.assertEqual(CANARY.transport.as_dict()["ipc"], T.PROTOCOL)
        with mock.patch.object(T, "implementation_sha256", lambda: "0" * 64):
            with self.assertRaises(RuntimeError):
                L.g663_canary_p2_profile()


class SelectionDescriptorIdentityTests(unittest.TestCase):
    """The SCIENTIFIC SELECTION descriptor of Jul07/Jul13 is unchanged by G6.6.3D."""

    # Computed from b359087 (G6.6.3A packet), before any G6.6.3D edit.
    EXPECTED = {
        "2026-07-07": ("e5cc9cd76e158a827a17e25b6b9b3aa5cd31e9a9a0b1bb1ec2b166487be567fb",
                       "e1f8855443d7b1e25cf2d22f8552cb4a2950adba604fac49e7a4e46aff741504"),
        "2026-07-13": ("744f6134a58166ea2a6d909608da10746d1f102f4a85311ecd7e7d5454be8ba4",
                       "6d9bd10a37b373ad4f90263e79fab354f045d8efa68c91c8c2b09010adab5f31"),
    }

    def test_selection_descriptor_and_p1_contract_are_unchanged(self):
        for day, (selection, p1_contract) in self.EXPECTED.items():
            with self.subTest(day=day):
                request = R.LiveRangeAdapter(ROOT, RELEASE, None).acquisition_request(day)
                descriptor = B.selection_descriptor_for_request(request)
                self.assertEqual(B.selection_descriptor_identity(descriptor), {
                    "selection_descriptor_schema": "v3.qualified-scientific-selection.v1",
                    "selection_descriptor_sha256": selection})
                self.assertEqual(B.selection_descriptor_for_request(
                    R.LiveRangeAdapter(ROOT, RELEASE, None, proof_profile=CANARY).acquisition_request(day)), descriptor)
                self.assertEqual(L._sha256_json(L.scientific_contract_for_profile(request)), p1_contract)


class ProcessOverheadMeasurement(OfflineTestCase):
    """Measured cost of the boundary (recorded in the G6.6.3D packet; sanity-bounded here)."""

    def test_measure_startup_round_trip_shutdown_kill_memory_and_process_count(self):
        loci = tile_loci(FIRST_TILE, 10)
        document = json.dumps({"data": [listing_resource(l) for l in loci], "links": {"next": None}}).encode()

        def handle(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "application/vnd.api+json")
            handler.send_header("Content-Length", str(len(document)))
            handler.end_headers()
            handler.wfile.write(document)
        server = Loopback(handle)
        self.addCleanup(server.close)
        pages, measured = 300, {"python": sys.version.split()[0], "platform": sys.platform,
                                "start_method": "subprocess.Popen(fork_exec, start_new_session)"}
        in_process = guarded(session_factory=server.session)
        started = time.perf_counter()
        for _ in range(pages):
            drain(in_process.search(BODY))
        measured["in_process_ms_per_page"] = round((time.perf_counter() - started) * 1000 / pages, 3)
        listing = process_guarded(self, None, server.port)
        child = listing._transport
        drain(listing.search(BODY))  # lazy start
        measured["startup_seconds"] = child.stats["startup_seconds"]
        pid = child.stats["pid"]
        started = time.perf_counter()
        for _ in range(pages):
            records, error = drain(listing.search(BODY))
            self.assertIsNone(error)
        measured["process_ms_per_page"] = round((time.perf_counter() - started) * 1000 / pages, 3)
        measured["round_trip_overhead_ms_per_page"] = round(
            measured["process_ms_per_page"] - measured["in_process_ms_per_page"], 3)
        measured["children_during_pages"] = len(transport_children())
        measured["persistence"] = {"spawns": child.stats["spawns"], "requests": child.stats["requests"],
                                   "one_pid": child.stats["pid"] == pid}
        measured["child_rss_kib"], measured["child_peak_rss_kib"] = rss_kib(pid), rss_kib(pid, "VmHWM")
        started = time.perf_counter()
        record = listing._transport.close()
        measured["shutdown_seconds"] = round(time.perf_counter() - started, 6)
        measured["shutdown_returncode"] = record["returncode"]
        kills = {}
        for name, hook in (("sigterm", {}), ("sigterm_ignored", {"ignore_sigterm": True})):
            blocked = process_guarded(self, limits(iterator_deadline_seconds=1), server.port, block_resolver=True, **hook)
            drain(blocked.search(BODY))
            kills[name] = {k: blocked._transport.exit_record[k] for k in ("returncode", "escalated_to_sigkill", "seconds")}
        measured["kill"] = kills
        self.assertEqual(measured["children_during_pages"], 1)
        self.assertEqual(measured["persistence"], {"spawns": 1, "requests": pages + 1, "one_pid": True})
        self.assertLess(measured["startup_seconds"], CANARY.transport.child_startup_seconds)
        self.assertEqual(measured["shutdown_returncode"], 0)
        self.assertLess(kills["sigterm"]["seconds"], CANARY.transport.terminate_grace_seconds)
        self.assertLess(kills["sigterm_ignored"]["seconds"],
                        CANARY.transport.terminate_grace_seconds + CANARY.transport.kill_join_seconds)
        self.assertEqual(transport_children(), [])
        output = os.environ.get("G663D_PERF_OUT")
        if output:
            Path(output).write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
        print("\nG6.6.3D overhead: " + json.dumps(measured, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
