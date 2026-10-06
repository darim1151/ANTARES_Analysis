"""G6.6.3A guarded P2 transport qualification (offline; no ANTARES request).

G6.6.3D: each page's blocking HTTP exchange now runs in a dedicated transport
child process.  The HTTP guard rules below run the child's own page function
in-process (``InProcessPages``) through ``requests``' adapter seam; every
whole-night, socket and deadline test runs the real child, which reaches the
same offline service over loopback (``ServiceServer``).  The process boundary
itself is qualified in ``test_g663d_process_transport``.

The pinned antares-client 1.14.0 ``_list_all_resources`` follows ``links.next``
inside one ``next()`` with no page, byte, time, cycle, scheme or origin bound,
and ``requests.get`` follows redirects anywhere.  An explicit P2 profile with
exact ``P2TransportLimits`` pages through the provider-owned
``_GuardedListing`` instead.  These tests challenge that paginator against the
*real* pinned client running over the same deterministic offline service:

* Every HTTP exchange goes through ``requests``' own transport-adapter seam
  (``HTTPAdapter.send``), answered by an in-process JSON:API service, or through
  a loopback HTTP server for real socket timing.  A socket guard refuses any
  non-loopback connection or name lookup in every test.
* Valid page sequences must yield exactly the pinned client's loci, in order,
  after exactly the same HTTP requests.
* Every refusal must leave the P2 node incomplete: no sealed query, no
  accepted rows, no completeness, and deterministic journal evidence.
"""

import ast
import contextlib
import copy
import gzip
import hashlib
import http.server
import io
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.response import HTTPResponse

import v3_fixtures as F
from test_operations_phase6 import FakeLocus, _body_matches_locus
from test_g662_p2_proof import dense_loci, floor_tile, test_profile
from src import query
from src.operations import backfill as B, live_antares as L, production_range as R, science as S
from src.operations import p2_transport as T
from src.operations import query_checkpoint as Q
from src.operations.science import ArtifactValidationError, ProviderOutcome, QueryInterruptedError

ROOT = Path(__file__).resolve().parents[1]
G662B = "8f4ee9bcf4a8be64cfabfa94af50ebb2fc35bc47"
BASE = L.OFFICIAL_API_BASE_URL
LISTING = BASE + "loci"
HOST = "api.antares.noirlab.edu"
ES_KEY = "elasticsearch_query[locus_listing]"
SORT = "-properties.newest_alert_observation_time"
NIGHT = "2026-06-27"
RELEASE = "a" * 40
GUARD = "src.operations.live_antares."
CANARY = L.g663_canary_p2_profile()
FIRST_TILE = L._make_initial_tiles(61218.0, 61219.0)[0]
CHILD_HOOKS = str(ROOT / "tests" / "g663d_child_hooks.py")


def listing_schema():
    from antares_client._api.schemas import _LocusListingSchema
    return _LocusListingSchema(many=True, partial=True)


def limits(**changes):
    values = dict(max_pages=64, max_consecutive_empty_pages=2, max_page_bytes=16_777_216,
                  max_iterator_bytes=67_108_864, iterator_deadline_seconds=600,
                  connect_timeout_seconds=60, read_timeout_seconds=60, max_redirects=2,
                  ipc_max_header_bytes=65_536, ipc_max_chunk_bytes=1_048_576,
                  child_startup_seconds=30, child_shutdown_seconds=5,
                  terminate_grace_seconds=2, kill_join_seconds=5)
    values.update(changes)
    return L.P2TransportLimits(**values)


class InProcessPages:
    """The transport child's own page function and failure encoding, without the process.

    Used only to drive the HTTP guard rules through ``requests``' adapter seam;
    the process boundary that replaces it in production is qualified separately.
    """

    def __init__(self, transport, session_factory):
        self.config = L._p2_child_config(transport, BASE)
        self.session_factory = session_factory

    def fetch(self, url, params, budget, overflow, remaining):
        try:
            return T.fetch_page(self.session_factory, self.config, url, params, budget, overflow)
        except Exception as exc:
            failure = T.classify_failure(exc)
            raise T.TransportFailure("remote", failure["code"], failure["type"], failure["status"]) from exc


def guarded(transport=None, session_factory=requests.Session):
    transport = transport or CANARY.transport
    return L._GuardedListing(transport, BASE, InProcessPages(transport, session_factory), listing_schema)


def child_hook(port=None, **args):
    return {"path": CHILD_HOOKS, "name": "configure", "args": {"port": port, **args}}


def process_guarded(test, transport=None, port=None, **hook_args):
    """The production paginator over a real transport child routed to a loopback port."""
    transport = transport or CANARY.transport
    child = T.P2ProcessTransport(L._p2_child_config(transport, BASE), hook=child_hook(port, **hook_args))
    test.addCleanup(child.close)
    return L._GuardedListing(transport, BASE, child, listing_schema)


def transport_children():
    """Live (or unreaped) child processes of this test process: transport children."""
    me, children = os.getpid(), []
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if entry.name.isdigit():
                with contextlib.suppress(OSError, IndexError, ValueError):
                    if int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1]) == me:
                        children.append(int(entry.name))
        return sorted(children)
    inspector = subprocess.Popen(["ps", "-A", "-o", "pid=,ppid="], stdout=subprocess.PIPE, text=True)
    listing, _ = inspector.communicate()
    return sorted(int(pid) for pid, ppid in (line.split() for line in listing.splitlines())
                  if int(ppid) == me and int(pid) not in (me, inspector.pid))


def listing_resource(locus):
    return {"type": "locus_listing", "id": locus.locus_id,
            "attributes": {"ra": locus.ra, "dec": locus.dec, "htm16": 0,
                           "properties": locus.properties, "tags": list(locus.tags), "catalogs": []}}


def locus_document(locus):
    frame = (locus.lightcurve if isinstance(locus.lightcurve, pd.DataFrame)
             else pd.DataFrame(columns=["mjd", "ztf_magpsf", "ztf_sigmapsf", "ztf_fid"]))
    return {"data": {"type": "locus", "id": locus.locus_id,
                     "attributes": {"ra": locus.ra, "dec": locus.dec, "htm16": 0,
                                    "properties": locus.properties, "tags": list(locus.tags),
                                    "catalogs": [], "lightcurve": frame.to_csv(index=False)}}}


def tile_loci(tile, count, prefix="T", *, time_fraction=.5):
    """``count`` distinct LSST loci strictly inside ``tile`` (service order = list order)."""
    loci = []
    for n in range(count):
        mjd = tile["mjd_min"] + (tile["mjd_max"] - tile["mjd_min"]) * time_fraction
        ra = tile["ra_min"] + (tile["ra_max"] - tile["ra_min"]) * (n + .5) / max(count, 1)
        dec = tile["dec_min"] + (tile["dec_max"] - tile["dec_min"]) * .5
        loci.append(FakeLocus(f"{prefix}-{n:04d}", mjd=mjd, ra=ra, dec=dec))
    return loci


def respond(adapter, request, *, status=200, json_body=None, body=b"", headers=None, stream=None):
    if json_body is not None:
        body = json.dumps(json_body).encode()
    raw = HTTPResponse(body=stream if stream is not None else io.BytesIO(body),
                       headers={"Content-Type": "application/vnd.api+json", **(headers or {})},
                       status=status, preload_content=False, decode_content=True,
                       enforce_content_length=False, request_method="GET", request_url=request.url)
    return adapter.build_response(request, raw)


class FakeAntares:
    """Deterministic offline JSON:API service seen through ``HTTPAdapter.send``.

    Listings filter fixture loci with the exact tile-query predicate, keep list
    order as service order and paginate with absolute same-origin
    ``page[offset]`` links, like the evidence of the preserved production
    journals (``links.next`` followed at 10-row page boundaries).
    """

    def __init__(self, loci=(), *, page_size=10, terminal="null", full_page_next=False,
                 next_style="absolute", gaps=(), redirect_offsets=(), script=None):
        self.loci = list(loci)
        self.page_size, self.terminal, self.full_page_next = page_size, terminal, full_page_next
        self.next_style, self.gaps, self.redirect_offsets = next_style, set(gaps), set(redirect_offsets)
        self.script = script
        self.calls = []
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def installed(self, **child):
        """In-process HTTP (P1, fetch) through the adapter seam; the P2 transport
        child reaches the same service and call log over loopback.  ``child``
        are extra child hook arguments (for example a misbehaviour)."""
        service = self

        def send(adapter, request, **kwargs):
            return service.send(adapter, request, **kwargs)
        server = ServiceServer(self)
        try:
            with mock.patch.object(HTTPAdapter, "send", send), \
                    mock.patch.object(T, "CHILD_TEST_HOOK", server.hook(**child)):
                yield self
        finally:
            server.close()
        server.assert_no_violations()

    def record(self, url, timeout, stream):
        with self.lock:
            self.calls.append({"url": url, "timeout": timeout, "stream": stream})
            return len(self.calls) - 1

    def spec(self, request, index):
        parts = urlsplit(request.url)
        if parts.scheme != "https" or parts.hostname != HOST or parts.port not in (None, 443):
            # Foreign targets are answered (an empty, terminal listing) so a test can
            # prove that the pinned client *did* go there and the guard did not.
            return {"json_body": {"data": [], "links": {"next": None}}}
        spec = self.script(self, request, index) if self.script is not None else None
        return spec if spec is not None else self.default(request.url)

    def send(self, adapter, request, **kwargs):
        index = self.record(request.url, kwargs.get("timeout"), kwargs.get("stream"))
        response = respond(adapter, request, **self.spec(request, index))
        self.calls[index]["status"] = response.status_code
        return response

    @property
    def urls(self):
        return [call["url"] for call in self.calls]

    @property
    def pages(self):
        """Logical pages: exchanges that were not redirects."""
        return sum(call.get("status") not in (301, 302, 303, 307, 308) for call in self.calls)

    def default(self, url):
        parts = urlsplit(url)
        if parts.path == "/v1/loci":
            return self.listing(url)
        if parts.path.startswith("/v1/loci/"):
            identity = parts.path.rsplit("/", 1)[1]
            for locus in self.loci:
                if locus.locus_id == identity:
                    return {"json_body": locus_document(locus)}
        return {"status": 404, "json_body": {"errors": [{"status": "404"}]}}

    def next_url(self, raw_query, offset, *, gap=False):
        pairs = [("sort", SORT), (ES_KEY, raw_query), ("page[offset]", str(offset))]
        if gap:
            pairs.append(("gap", "1"))
        base = {"absolute": LISTING, "port443": f"https://{HOST}:443/v1/loci",
                "upper": f"HTTPS://{HOST.upper()}/v1/loci"}[self.next_style]
        return base + "?" + urlencode(pairs)

    def listing(self, url):
        params = parse_qs(urlsplit(url).query, keep_blank_values=True)
        raw_query = params[ES_KEY][0]
        offset = int(params.get("page[offset]", ["0"])[0])
        if offset in self.redirect_offsets and "hop" not in params:
            return {"status": 307, "headers": {"Location": "/v1/loci?" + urlsplit(url).query + "&hop=1"}}
        if "gap" in params:
            return {"json_body": {"data": [], "links": {"next": self.next_url(raw_query, offset)}}}
        body = json.loads(raw_query)
        matched = [locus for locus in self.loci if _body_matches_locus(body, locus)]
        page = matched[offset:offset + self.page_size]
        more = (offset + self.page_size < len(matched)
                or (self.full_page_next and len(page) == self.page_size))
        following = offset + self.page_size
        target = self.next_url(raw_query, following, gap=following in self.gaps) if more else None
        document = {"data": [listing_resource(locus) for locus in page], "meta": {"count": len(matched)}}
        if target is not None or self.terminal == "null":
            document["links"] = {"self": url, "next": target}
        elif self.terminal == "missing":
            document["links"] = {"self": url}
        return {"json_body": document}


class ServiceServer:
    """A loopback HTTP/1.1 front for one ``FakeAntares``, used only by transport children.

    The child's test adapter sends the official URL and its send arguments in
    headers, so the service sees exactly the request the child prepared and
    records it in the same call log as in-process traffic.  A script that
    raises makes the child's adapter raise the same ``requests`` exception.
    """

    def __init__(self, service):
        outer = self
        self.service = service
        folder = tempfile.mkdtemp(prefix="g663d-child-")
        self.violations = Path(folder) / "violations.log"

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                outer.handle(self)

            def log_message(self, *args):
                pass
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def hook(self, **args):
        return child_hook(self.port, violations=str(self.violations), **args)

    def handle(self, handler):
        url = handler.headers.get("X-G663-Original-URL")
        sent = json.loads(handler.headers.get("X-G663-Send") or "{}")
        timeout = sent.get("timeout")
        index = self.service.record(url, tuple(timeout) if isinstance(timeout, list) else timeout,
                                    sent.get("stream"))
        try:
            spec = self.service.spec(types.SimpleNamespace(url=url), index)
        except Exception as exc:
            spec = {"status": 599, "headers": {"X-G663-Raise": f"{type(exc).__module__}.{type(exc).__name__}"}}
        status = spec.get("status", 200)
        headers = {"Content-Type": "application/vnd.api+json", **(spec.get("headers") or {})}
        stream = spec.get("stream")
        body = json.dumps(spec["json_body"]).encode() if spec.get("json_body") is not None else spec.get("body", b"")
        try:
            handler.send_response(status)
            for key, value in headers.items():
                handler.send_header(key, value)
            if "Content-Length" not in headers and stream is None:
                handler.send_header("Content-Length", str(len(body)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            if stream is None:
                handler.wfile.write(body)
            else:
                for chunk in iter(lambda: stream.read(65536), b""):
                    handler.wfile.write(chunk)
        except OSError:  # the child already refused or abandoned this page
            pass
        handler.close_connection = True
        if status != 599:
            self.service.calls[index]["status"] = status

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def assert_no_violations(self):
        recorded = self.violations.read_text() if self.violations.exists() else ""
        if recorded:
            raise AssertionError(f"a transport child attempted network access: {recorded}")


class NoNetwork:
    """Refuse every non-loopback connection and name lookup; record attempts."""

    LOOPBACK = {"127.0.0.1", "localhost"}

    def __init__(self):
        self.attempts = []
        self.stack = contextlib.ExitStack()

    def __enter__(self):
        real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
        real_getaddrinfo = socket.getaddrinfo

        def host_of(address):
            return address[0] if isinstance(address, tuple) else address

        def connect(sock, address):
            if host_of(address) not in self.LOOPBACK:
                self.attempts.append(address)
                raise AssertionError(f"network access refused: {address!r}")
            return real_connect(sock, address)

        def connect_ex(sock, address):
            if host_of(address) not in self.LOOPBACK:
                self.attempts.append(address)
                raise AssertionError(f"network access refused: {address!r}")
            return real_connect_ex(sock, address)

        def getaddrinfo(host, *args, **kwargs):
            if host not in self.LOOPBACK:
                self.attempts.append(host)
                raise AssertionError(f"name lookup refused: {host!r}")
            return real_getaddrinfo(host, *args, **kwargs)

        self.stack.enter_context(mock.patch.object(socket.socket, "connect", connect))
        self.stack.enter_context(mock.patch.object(socket.socket, "connect_ex", connect_ex))
        self.stack.enter_context(mock.patch.object(socket, "getaddrinfo", getaddrinfo))
        return self

    def __exit__(self, *exc):
        self.stack.close()
        return False


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        self.network = NoNetwork().__enter__()
        self.addCleanup(self.network.__exit__, None, None, None)

    def tearDown(self):
        self.assertEqual(self.network.attempts, [], "a test attempted a non-loopback connection")

    def assertOfficial(self, urls):
        for url in urls:
            parts = urlsplit(url)
            self.assertEqual((parts.scheme, parts.hostname), ("https", HOST), url)
            self.assertIn(parts.port, (None, 443), url)
            self.assertTrue(parts.path.startswith("/v1/"), url)


def drain(iterable, *, stop_at=None):
    """Consume like P2: up to ``stop_at`` loci, then close; return records and error."""
    records, error, iterator = [], None, None
    try:
        iterator = iter(iterable)
        while stop_at is None or len(records) < stop_at:
            try:
                locus = next(iterator)
            except StopIteration:
                break
            records.append(query.locus_to_record(locus))
    except Exception as exc:  # the typed refusal is the result under test
        error = exc
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    return records, error


def pinned_search(body):
    from antares_client.search import search
    return search(body)


class TransportEquivalenceTests(OfflineTestCase):
    """Valid sequences: guarded loci, order and HTTP requests equal the pinned client's."""

    def compare(self, service, body, *, stop_at=None):
        with service.installed():
            expected, expected_error = drain(pinned_search(body), stop_at=stop_at)
            pinned_urls = list(service.urls)
            service.calls.clear()
            observed, observed_error = drain(guarded().search(body), stop_at=stop_at)
            guarded_calls = list(service.calls)
        self.assertIsNone(expected_error)
        self.assertIsNone(observed_error)
        self.assertEqual(observed, expected)
        self.assertEqual(guarded_calls[0]["url"], pinned_urls[0])
        self.assertEqual([(urlsplit(call["url"]).path, urlsplit(call["url"]).query) for call in guarded_calls],
                         [(urlsplit(url).path, urlsplit(url).query) for url in pinned_urls])
        self.assertOfficial([call["url"] for call in guarded_calls])
        self.assertOfficial(pinned_urls)
        for call in guarded_calls:  # the client's scalar 60 s, split exactly; streamed body
            self.assertEqual(call["timeout"], (60, 60))
            self.assertTrue(call["stream"])
        return observed, pinned_urls

    def test_one_page_multiple_pages_and_terminal_shapes(self):
        body = L._build_tile_query(FIRST_TILE)
        for count, page_size, terminal, full_next, pages in (
                (0, 10, "null", False, 1), (7, 10, "null", False, 1), (7, 10, "missing", False, 1),
                (7, 10, "no-links", False, 1), (35, 10, "null", False, 4), (40, 10, "null", True, 5),
                (40, 10, "missing", True, 5), (40, 10, "null", False, 4)):
            with self.subTest(count=count, terminal=terminal, full_next=full_next):
                loci = tile_loci(FIRST_TILE, count)
                records, urls = self.compare(FakeAntares(loci, page_size=page_size, terminal=terminal,
                                                         full_page_next=full_next), body)
                self.assertEqual([r["locus_id"] for r in records], [l.locus_id for l in loci])
                self.assertEqual(len(urls), pages)

    def test_empty_intermediate_pages_same_origin_forms_and_redirects(self):
        body = L._build_tile_query(FIRST_TILE)
        loci = tile_loci(FIRST_TILE, 45)
        for kwargs in ({"gaps": {10, 30}}, {"next_style": "port443"}, {"next_style": "upper"},
                       {"redirect_offsets": {0, 20}}, {"gaps": {20}, "redirect_offsets": {10}},
                       {"full_page_next": True, "page_size": 9, "gaps": {18}}):
            with self.subTest(**{k: sorted(v) if isinstance(v, set) else v for k, v in kwargs.items()}):
                records, _urls = self.compare(FakeAntares(loci, **kwargs), body)
                self.assertEqual([r["locus_id"] for r in records], [l.locus_id for l in loci])

    def test_p2_consumption_stops_at_50_after_the_same_pages(self):
        body = L._build_tile_query(FIRST_TILE)
        for count, page_size, pages in ((60, 10, 5), (60, 7, 8), (51, 1, 50), (49, 1, 49), (200, 64, 1)):
            with self.subTest(count=count, page_size=page_size):
                service = FakeAntares(tile_loci(FIRST_TILE, count), page_size=page_size)
                records, urls = self.compare(service, body, stop_at=50)
                self.assertEqual(len(records), min(count, 50))
                self.assertEqual(len(urls), pages)
        # Page size 1 with a terminal empty page: 49 rows, 50 pages, still inside 64.
        service = FakeAntares(tile_loci(FIRST_TILE, 49), page_size=1, full_page_next=True)
        records, urls = self.compare(service, body, stop_at=50)
        self.assertEqual((len(records), len(urls)), (49, 50))
        self.assertLessEqual(len(urls), CANARY.transport.max_pages)

    def test_randomized_sequences_match_exactly_or_refuse_only_beyond_profile(self):
        rng = random.Random(663)
        body = L._build_tile_query(FIRST_TILE)
        branches = {"equal": 0, "refused": 0}
        self.addCleanup(lambda: self.assertTrue(all(branches.values()), branches))
        for trial in range(60):
            count = rng.randrange(0, 130)
            page_size = rng.choice((1, 2, 3, 7, 10, 25, 50, 100))
            loci = tile_loci(FIRST_TILE, count, prefix=f"R{trial}")
            rng.shuffle(loci)  # service order is not identifier order
            for locus in loci:  # payload variety: unicode, nested values, exact binary64
                locus.properties = {**locus.properties, "note": rng.choice(("ä→✓", "", "x" * rng.randrange(50))),
                                    "vector": [rng.random() for _ in range(rng.randrange(3))],
                                    "nested": {"k": rng.randrange(10 ** 12), "f": rng.uniform(-1e300, 1e300)}}
            offsets = range(page_size, count, page_size)
            gaps = {offset for offset in offsets if rng.random() < .15}
            hops = {offset for offset in range(0, max(count, 1), page_size) if rng.random() < .15}
            service = FakeAntares(loci, page_size=page_size, terminal=rng.choice(("null", "missing", "no-links")),
                                  full_page_next=rng.random() < .3, gaps=gaps, redirect_offsets=hops,
                                  next_style=rng.choice(("absolute", "port443", "upper")))
            with self.subTest(trial=trial, count=count, page_size=page_size):
                # P2 consumption never needs more pages than 50 rows plus empty ones.
                records, urls = self.compare(service, body, stop_at=50)
                self.assertEqual([r["locus_id"] for r in records], [l.locus_id for l in loci][:50])
                # Full drain: exact equality while the page bound holds; beyond it, refusal.
                with service.installed():
                    service.calls.clear()
                    expected, _ = drain(pinned_search(body))
                    pages = service.pages
                    observed, error = drain(guarded().search(body))
                if pages <= CANARY.transport.max_pages:
                    self.assertIsNone(error)
                    self.assertEqual(observed, expected)
                    branches["equal"] += 1
                else:
                    self.assertIsInstance(error, L.P2PageLimitError)
                    self.assertEqual(observed, expected[:len(observed)])
                    branches["refused"] += 1

    def test_records_are_the_clients_own_locus_objects(self):
        service = FakeAntares(tile_loci(FIRST_TILE, 3))
        with service.installed():
            loci = list(guarded().search(L._build_tile_query(FIRST_TILE)))
        from antares_client.models import Locus
        self.assertTrue(all(type(locus) is Locus for locus in loci))
        self.assertEqual([locus.locus_id for locus in loci], ["T-0000", "T-0001", "T-0002"])


MISSING = object()


def page(loci=(), next=MISSING, *, links=MISSING):
    document = {"data": [listing_resource(locus) for locus in loci]}
    if links is not MISSING:
        document["links"] = links
    elif next is not MISSING:
        document["links"] = {"next": next}
    return {"json_body": document}


def link(key, base=LISTING):
    return f"{base}?p={key}"


def scripted(pages):
    """``pages[key]`` answers ``?p=key``; key ``None`` answers the first request."""
    def script(service, request, index):
        key = parse_qs(urlsplit(request.url).query).get("p", [None])[0]
        spec = pages[key]
        return spec(request.url) if callable(spec) else spec
    return script


class CountingStream(io.RawIOBase):
    """A response body that counts what the transport actually read."""

    def __init__(self, total, fill=b" ", prefix=b""):
        self.total, self.fill, self.prefix, self.read_bytes = total, fill, prefix, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        remaining = self.total - self.read_bytes
        if remaining <= 0:
            return 0
        size = min(len(buffer), remaining)
        produced = bytearray()
        if self.read_bytes < len(self.prefix):
            produced += self.prefix[self.read_bytes:self.read_bytes + size]
        produced += self.fill * (size - len(produced))
        buffer[:size] = produced[:size]
        self.read_bytes += size
        return size


def page_threads():
    """G6.6.3A's page threads no longer exist: no thread may ever run a page."""
    return [thread for thread in threading.enumerate() if thread.name == "antares-p2-page"]


class TransportGuardTests(OfflineTestCase):
    """Every refusal is typed, bounded, never follows the refused target, never completes."""

    body = L._build_tile_query(FIRST_TILE)

    def run_guarded(self, service, transport=None, stop_at=None):
        with service.installed():
            records, error = drain(guarded(transport).search(self.body), stop_at=stop_at)
        self.assertOfficial(service.urls)
        self.assertEqual(page_threads(), [])
        return records, error

    def run_pinned(self, service, stop_at=None):
        with service.installed():
            return drain(pinned_search(self.body), stop_at=stop_at)

    def assertRefused(self, error, kind, *, retryable=False):
        self.assertIs(type(error), kind, repr(error))
        self.assertEqual(L._retryable_query_error(error), retryable)
        if issubclass(kind, L.P2TransportGuardError):
            self.assertIn(L._exception_type(error), L._P2_TRANSPORT_GUARD_TYPES)

    def test_pinned_client_is_unbounded_where_the_guard_refuses_cycles_and_empty_runs(self):
        loop = {None: page(next=link("a")), "a": page(next=link("a"))}
        two = {None: page(next=link("a")), "a": page(next=link("b")), "b": page(next=link("a"))}
        advancing = {None: page(next=link(1)), **{str(n): page(next=link(n + 1)) for n in range(1, 40)}}
        for name, pages, kind, requests_made in (
                ("self-loop", loop, L.P2ContinuationCycleError, 2),
                ("two-page-cycle", two, L.P2EmptyPageLimitError, 3),
                ("ever-advancing-empty", advancing, L.P2EmptyPageLimitError, 3)):
            with self.subTest(name):
                stop = FakeAntares(script=scripted(pages))
                class SafetyStop(RuntimeError):
                    pass
                inner = stop.script
                stop.script = lambda service, request, index: (
                    (_ for _ in ()).throw(SafetyStop()) if index >= 30 else inner(service, request, index))
                records, error = self.run_pinned(stop)
                self.assertIsInstance(error, SafetyStop)  # 30 HTTP pages inside a single next()
                self.assertEqual((records, len(stop.calls)), ([], 31))
                service = FakeAntares(script=scripted(pages))
                records, error = self.run_guarded(service)
                self.assertRefused(error, kind)
                self.assertEqual((records, len(service.calls)), ([], requests_made))

    def test_cycles_are_detected_across_pages_and_url_spellings(self):
        one = tile_loci(FIRST_TILE, 6)
        for name, pages, requests_made, rows in (
                ("first-page-self", {None: lambda url: page(one[:1], next=url)}, 1, 1),
                ("two-page", {None: page(one[:1], next=link("a")), "a": page(one[1:2], next=link("b")),
                              "b": page(one[2:3], next=link("a"))}, 3, 3),
                ("longer", {None: page(one[:1], next=link("a")), "a": page(one[1:2], next=link("b")),
                            "b": page(one[2:3], next=link("c")), "c": page(one[3:4], next=link("d")),
                            "d": page(one[4:5], next=link("b"))}, 5, 5),
                ("port-443-spelling", {None: page(one[:1], next=link("a")),
                                       "a": page(one[1:2], next=link("a", f"https://{HOST}:443/v1/loci"))}, 2, 2),
                ("host-case-spelling", {None: page(one[:1], next=link("a")),
                                        "a": page(one[1:2], next=link("a", f"https://{HOST.upper()}/v1/loci"))}, 2, 2),
        ):
            with self.subTest(name):
                service = FakeAntares(script=scripted(pages))
                records, error = self.run_guarded(service, limits(max_consecutive_empty_pages=1000))
                self.assertRefused(error, L.P2ContinuationCycleError)
                self.assertEqual((len(records), len(service.calls)), (rows, requests_made))

    def test_empty_page_runs_are_bounded_but_a_terminal_empty_page_is_natural_exhaustion(self):
        loci = tile_loci(FIRST_TILE, 3)
        allowed = {None: page(loci[:1], next=link("e1")), "e1": page(next=link("e2")),
                   "e2": page(next=link("x")), "x": page(loci[1:2], next=link("t")), "t": page(next=None)}
        records, error = self.run_guarded(FakeAntares(script=scripted(allowed)))
        self.assertIsNone(error)
        self.assertEqual([r["locus_id"] for r in records], ["T-0000", "T-0001"])
        refused = {None: page(loci[:1], next=link("e1")), "e1": page(next=link("e2")),
                   "e2": page(next=link("e3")), "e3": page(next=link("x")), "x": page(loci[1:2], next=None)}
        service = FakeAntares(script=scripted(refused))
        records, error = self.run_guarded(service)
        self.assertRefused(error, L.P2EmptyPageLimitError)
        self.assertEqual((len(records), len(service.calls)), (1, 4))
        records, error = self.run_guarded(FakeAntares(script=scripted(refused)),
                                          limits(max_consecutive_empty_pages=0))
        self.assertRefused(error, L.P2EmptyPageLimitError)

    def test_page_ceiling_is_a_refusal_at_the_exact_bound(self):
        loci = tile_loci(FIRST_TILE, 120)
        advancing = {None: page(loci[:1], next=link(1)),
                     **{str(n): page(loci[n:n + 1], next=link(n + 1)) for n in range(1, 120)}}
        service = FakeAntares(script=scripted(advancing))
        records, error = self.run_guarded(service, limits(max_pages=5))
        self.assertRefused(error, L.P2PageLimitError)
        self.assertEqual((len(records), len(service.calls)), (5, 5))
        # Frozen profile: rows interleaved with the maximum empty run reach 64 pages, < 50 rows.
        pattern = {None: page(loci[:1], next=link(1))}
        for n in range(1, 200):
            pattern[str(n)] = page(loci[n // 3:n // 3 + 1] if n % 3 == 0 else (), next=link(n + 1))
        service = FakeAntares(script=scripted(pattern))
        records, error = self.run_guarded(service, stop_at=50)
        self.assertRefused(error, L.P2PageLimitError)
        self.assertEqual(len(service.calls), CANARY.transport.max_pages)
        self.assertLess(len(records), 50)

    def test_page_and_iterator_byte_ceilings_at_frozen_values(self):
        ceiling = CANARY.transport.max_page_bytes
        valid = json.dumps({"data": [], "links": {"next": None}}).encode()
        for declared, kind in ((ceiling, None), (ceiling + 1, L.P2PageBytesError)):
            with self.subTest(declared=declared):
                stream = CountingStream(len(valid), prefix=valid)
                service = FakeAntares(script=lambda s, r, i, stream=stream, declared=declared: {
                    "stream": stream, "headers": {"Content-Length": str(declared)}})
                records, error = self.run_guarded(service)
                if kind is None:
                    self.assertIsNone(error)
                else:
                    self.assertRefused(error, kind)
                    self.assertEqual(stream.read_bytes, 0)  # refused before reading the body
        for size, kind in ((ceiling, None), (ceiling + 1, L.P2PageBytesError)):
            with self.subTest(streamed=size):
                stream = CountingStream(size, prefix=valid)
                service = FakeAntares(script=lambda s, r, i, stream=stream: {"stream": stream})
                records, error = self.run_guarded(service)
                if kind is None:
                    self.assertIsNone(error)
                else:
                    self.assertRefused(error, kind)
                    self.assertLessEqual(stream.read_bytes, ceiling + T.STREAM_CHUNK_BYTES)
        one = tile_loci(FIRST_TILE, 5)
        def padded(loci, target):
            raw = json.dumps({"data": [listing_resource(l) for l in loci], "links": {"next": target}}).encode()
            return {"body": raw + b" " * (ceiling - 1 - len(raw))}
        pages = {None: padded(one[:1], link(1)), **{str(n): padded(one[n:n + 1], link(n + 1)) for n in range(1, 5)}}
        service = FakeAntares(script=scripted(pages))
        records, error = self.run_guarded(service)
        self.assertRefused(error, L.P2IteratorBytesError)
        self.assertEqual((len(records), len(service.calls)), (4, 5))

    def test_decoded_bytes_are_counted_so_compression_cannot_bypass_the_ceiling(self):
        bomb = gzip.compress(b" " * 40_000_000)
        self.assertLess(len(bomb), 100_000)
        service = FakeAntares(script=lambda s, r, i: {"body": bomb, "headers": {
            "Content-Encoding": "gzip", "Content-Length": str(len(bomb))}})
        records, error = self.run_guarded(service, limits(max_page_bytes=1 << 20))
        self.assertRefused(error, L.P2PageBytesError)

    def test_iterator_deadline_before_a_further_page(self):
        clock = {"now": 1000.0}
        fake_time = types.SimpleNamespace(monotonic=lambda: clock["now"])
        loci = tile_loci(FIRST_TILE, 3)
        for elapsed, kind in ((599.999, None), (600.0, L.P2DeadlineError)):
            with self.subTest(elapsed=elapsed):
                clock["now"] = 1000.0
                def advance(url, elapsed=elapsed):
                    clock["now"] += elapsed
                    return page(loci[:1], next=link("b"))
                service = FakeAntares(script=scripted({None: advance, "b": page(loci[1:2], next=None)}))
                with mock.patch.object(L, "time", fake_time):
                    records, error = self.run_guarded(service)
                if kind is None:
                    self.assertIsNone(error)
                    self.assertEqual(len(service.calls), 2)
                else:
                    self.assertRefused(error, kind)
                    self.assertEqual((len(records), len(service.calls)), (1, 1))

    def test_timeouts_are_passed_exactly(self):
        service = FakeAntares()
        self.run_guarded(service, limits(connect_timeout_seconds=5, read_timeout_seconds=7))
        self.assertEqual([call["timeout"] for call in service.calls], [(5, 7)])
        service = FakeAntares()
        self.run_guarded(service)
        self.assertEqual([call["timeout"] for call in service.calls], [(60, 60)])

    def test_malformed_json_and_jsonapi_fail_exactly_like_the_pinned_client(self):
        bad_type = {"data": [{"type": "wrong", "id": "x", "attributes": {}}]}
        missing_id = {"data": [{"type": "locus_listing", "attributes": {"ra": 1.0}}]}
        for name, spec, retryable in (
                ("html-gateway-page", {"body": b"<html>502 Bad Gateway</html>"}, True),
                ("truncated-json", {"body": b'{"data": [{"type": "locus_listing"'}, True),
                ("empty-body", {"body": b""}, True),
                ("top-level-array", {"json_body": []}, True),
                ("null-data", {"json_body": {"data": None}}, True),
                ("missing-data", {"json_body": {"meta": {}}}, True),
                ("wrong-resource-type", {"json_body": bad_type}, False),
                ("resource-without-id", {"json_body": missing_id}, False)):
            with self.subTest(name):
                pinned = FakeAntares(script=lambda s, r, i, spec=spec: copy.deepcopy(spec))
                _records, expected = self.run_pinned(pinned)
                service = FakeAntares(script=lambda s, r, i, spec=spec: copy.deepcopy(spec))
                records, error = self.run_guarded(service)
                self.assertIsNotNone(expected)
                self.assertIs(type(error), type(expected))
                self.assertEqual(L._retryable_query_error(error), retryable)
                self.assertEqual((records, len(service.calls)), ([], 1))

    def test_links_member_that_is_not_an_object_is_refused(self):
        loci = tile_loci(FIRST_TILE, 2)
        for links in (None, "next", [], 7):
            with self.subTest(links=links):
                service = FakeAntares(script=lambda s, r, i, links=links: page(loci, links=links))
                records, error = self.run_guarded(service)
                self.assertRefused(error, L.P2MalformedContinuationError)
                self.assertEqual(len(records), 2)  # yielded before refusal; P2 discards them
                _records, expected = self.run_pinned(FakeAntares(
                    script=lambda s, r, i, links=links: page(loci, links=links)))
                self.assertIsInstance(expected, AttributeError)  # the client cannot continue either

    def test_continuation_vectors(self):
        loci = tile_loci(FIRST_TILE, 2)
        query = "?p=2"
        cases = [
            # relative forms: the pinned client cannot request them (MissingSchema / InvalidURL)
            ("/v1/loci" + query, L.P2RelativeContinuationError, "relative"),
            ("loci" + query, L.P2RelativeContinuationError, "relative"),
            (query, L.P2RelativeContinuationError, "relative"),
            (f"//{HOST}/v1/loci" + query, L.P2RelativeContinuationError, "relative"),
            # insecure or foreign: the pinned client would follow these
            (f"http://{HOST}:8080/v1/loci" + query, L.P2CrossOriginContinuationError, "followed"),
            (f"ftp://{HOST}/v1/loci" + query, L.P2InsecureContinuationError, None),
            ("https://evil.example/v1/loci" + query, L.P2CrossOriginContinuationError, "followed"),
            (f"https://{HOST}.evil.example/v1/loci" + query, L.P2CrossOriginContinuationError, "followed"),
            (f"https://{HOST}:8443/v1/loci" + query, L.P2CrossOriginContinuationError, "followed"),
            (f"https://{HOST}./v1/loci" + query, L.P2CrossOriginContinuationError, None),
            ("https:/v1/loci" + query, L.P2CrossOriginContinuationError, None),
            # credentials, fragments, unparseable or non-plain strings
            (f"https://user@{HOST}/v1/loci" + query, L.P2MalformedContinuationError, None),
            (f"https://user:pw@{HOST}/v1/loci" + query, L.P2MalformedContinuationError, None),
            (f"https://evil.example@{HOST}/v1/loci" + query, L.P2MalformedContinuationError, None),
            (f"https://{HOST}/v1/loci" + query + "#frag", L.P2MalformedContinuationError, None),
            (f"https://{HOST}:99999/v1/loci" + query, L.P2MalformedContinuationError, None),
            (f"https://{HOST}:x/v1/loci" + query, L.P2MalformedContinuationError, None),
            (f" https://{HOST}/v1/loci" + query, L.P2MalformedContinuationError, None),
            (f"https://{HOST}/v1/loci?p=a b", L.P2MalformedContinuationError, None),
            (f"https://{HOST}/v1/loci\t" + query, L.P2MalformedContinuationError, None),
            (f"https://{HOST}/v1/lo\nci" + query, L.P2MalformedContinuationError, None),
            (f"https://{HOST}/v1/loci?p=\x7f", L.P2MalformedContinuationError, None),
            (f"https://{HOST}/v1/loci?p=é", L.P2MalformedContinuationError, None),
            (f"https://{HOST}\\@evil.example/v1/loci" + query, L.P2MalformedContinuationError, None),
            (f"https://evil.example\\.{HOST}/v1/loci" + query, L.P2MalformedContinuationError, None),
            (f"https://{HOST}/v1/loci\\..\\alerts" + query, L.P2MalformedContinuationError, None),
            ("", L.P2MalformedContinuationError, None),
            (7, L.P2MalformedContinuationError, None),
            (True, L.P2MalformedContinuationError, None),
            ({"href": LISTING + query}, L.P2MalformedContinuationError, None),
            ([LISTING + query], L.P2MalformedContinuationError, None),
            # another resource of the same origin
            (f"https://{HOST}/v1/alerts" + query, L.P2ContinuationPathError, None),
            (f"https://{HOST}/v1/loci/" + query, L.P2ContinuationPathError, None),
            (f"https://{HOST}/v2/loci" + query, L.P2ContinuationPathError, None),
            (f"https://{HOST}/v1/./loci" + query, L.P2ContinuationPathError, None),
            (f"https://{HOST}/v1/x/../loci" + query, L.P2ContinuationPathError, None),
            (f"https://{HOST}/v1/loci/statistics" + query, L.P2ContinuationPathError, None),
        ]
        for target, kind, pinned_behaviour in cases:
            with self.subTest(target=target):
                service = FakeAntares(script=scripted({None: page(loci, next=target), "2": page(next=None)}))
                records, error = self.run_guarded(service)
                self.assertRefused(error, kind)
                self.assertEqual(len(records), 2)
                self.assertEqual(len(service.calls), 1)  # the refused target is never requested
                if pinned_behaviour == "relative":
                    _records, expected = self.run_pinned(FakeAntares(
                        script=scripted({None: page(loci, next=target), "2": page(next=None)})))
                    self.assertIsInstance(expected, (requests.exceptions.MissingSchema,
                                                     requests.exceptions.InvalidURL))
                    self.assertFalse(L._retryable_query_error(expected))
                elif pinned_behaviour == "followed":
                    pinned = FakeAntares(script=scripted({None: page(loci, next=target), "2": page(next=None)}))
                    self.run_pinned(pinned)
                    self.assertEqual(len(pinned.calls), 2)
                    self.assertNotEqual(urlsplit(pinned.urls[1])[:2], ("https", HOST))
        # Every continuation the guard accepts is the same origin to requests/urllib3 too.
        from urllib3.util import parse_url
        accepted = guarded()
        for target in (LISTING + query, f"https://{HOST}:443/v1/loci" + query,
                       f"HTTPS://{HOST.upper()}/v1/loci" + query, LISTING + "?p=%2F..%2F&x=%40evil.example"):
            accepted._validated_continuation(target)
            prepared = parse_url(requests.Request("GET", target).prepare().url)
            self.assertEqual((prepared.scheme, prepared.host, prepared.port in (None, 443), prepared.path),
                             ("https", HOST, True, "/v1/loci"))

    def test_redirect_vectors(self):
        loci = tile_loci(FIRST_TILE, 2)
        refused = [
            (302, f"http://{HOST}/v1/loci?p=r", "followed"),
            (301, "https://evil.example/v1/loci?p=r", "followed"),
            (307, f"https://user:pw@{HOST}/v1/loci?p=r", None),
            (308, f"https://{HOST}:8443/v1/loci?p=r", "followed"),
            (302, "/other?p=r", None),
            (302, "/v2/loci?p=r", None),
            (303, f"https://{HOST}/v1/loci?p=r#f", None),
            (302, f"https://{HOST}/v1/lo ci?p=r", None),
            (302, f"https://{HOST}\\@evil.example/v1/loci?p=r", None),
            (302, " /v1/loci?p=r", None),
            (302, "", None),
            (302, None, None),
            (302, "/v1/loci?p=é", None),
            (302, "/v1/loci?p=Ã©", None),
            (302, "/v1/loci?p=r\x7f", None),
        ]
        for status, location, pinned_behaviour in refused:
            with self.subTest(status=status, location=location):
                headers = {} if location is None else {"Location": location}
                pages = {None: {"status": status, "headers": headers, "body": b""}, "r": page(loci, next=None)}
                service = FakeAntares(script=scripted(pages))
                records, error = self.run_guarded(service)
                self.assertRefused(error, L.P2UnsafeRedirectError)
                self.assertEqual((records, len(service.calls)), ([], 1))
                if pinned_behaviour == "followed":
                    pinned = FakeAntares(script=scripted(pages))
                    self.run_pinned(pinned)
                    self.assertNotEqual(urlsplit(pinned.urls[1])[:2], ("https", HOST))
        for status in (301, 302, 303, 307, 308):
            with self.subTest(followed=status):
                pages = {None: {"status": status, "headers": {"Location": "/v1/loci?p=r"}},
                         "r": page(loci, next=None)}
                records, error = self.run_guarded(FakeAntares(script=scripted(pages)))
                self.assertIsNone(error)
                self.assertEqual(len(records), 2)
        loop = {None: {"status": 302, "headers": {"Location": "/v1/loci?p=loop"}},
                "loop": {"status": 302, "headers": {"Location": "/v1/loci?p=loop"}}}
        service = FakeAntares(script=scripted(loop))
        records, error = self.run_guarded(service)
        self.assertRefused(error, L.P2RedirectLimitError)
        self.assertEqual(len(service.calls), CANARY.transport.max_redirects + 1)
        service = FakeAntares(script=scripted(loop))
        records, error = self.run_guarded(service, limits(max_redirects=0))
        self.assertRefused(error, L.P2RedirectLimitError)
        self.assertEqual(len(service.calls), 1)

    def test_redirect_bodies_are_never_read(self):
        loci = tile_loci(FIRST_TILE, 2)
        for location, kind in (("/v1/loci?p=r", None), ("https://evil.example/v1/loci?p=r", L.P2UnsafeRedirectError)):
            with self.subTest(location=location):
                stream = CountingStream(5_000_000)
                pages = {None: {"status": 302, "headers": {"Location": location}, "stream": stream},
                         "r": page(loci, next=None)}
                records, error = self.run_guarded(FakeAntares(script=scripted(pages)))
                self.assertIs(type(error), kind) if kind else self.assertIsNone(error)
                self.assertEqual(stream.read_bytes, 0)
        # requests.Session.send pre-computes Response.next even with allow_redirects=False
        # and so reads a redirect's entire body; the pinned client reads it too.
        stream = CountingStream(5_000_000)
        pages = {None: {"status": 302, "headers": {"Location": "/v1/loci?p=r"}, "stream": stream},
                 "r": page(loci, next=None)}
        with FakeAntares(script=scripted(pages)).installed():
            requests.Session().get(LISTING, allow_redirects=False, stream=True).close()
        self.assertEqual(stream.read_bytes, 5_000_000)

    def test_status_classification_never_reads_error_bodies(self):
        for status, kind, retryable in ((204, L.P2UnexpectedStatusError, False),
                                        (203, L.P2UnexpectedStatusError, False),
                                        (206, L.P2UnexpectedStatusError, False),
                                        (304, L.P2UnexpectedStatusError, False),
                                        (400, L.P2TransportHTTPError, True), (404, L.P2TransportHTTPError, True),
                                        (429, L.P2TransportHTTPError, True), (500, L.P2TransportHTTPError, True),
                                        (502, L.P2TransportHTTPError, True), (503, L.P2TransportHTTPError, True)):
            with self.subTest(status=status):
                stream = CountingStream(100, prefix=b'{"data": []}')
                service = FakeAntares(script=lambda s, r, i, status=status, stream=stream: {
                    "status": status, "stream": stream})
                records, error = self.run_guarded(service)
                self.assertRefused(error, kind, retryable=retryable)
                self.assertEqual(stream.read_bytes, 0)
                if status >= 400:  # the client also fails, retryably, on these statuses
                    _records, expected = self.run_pinned(FakeAntares(
                        script=lambda s, r, i, status=status: {"status": status, "body": b"<html/>"}))
                    self.assertTrue(L._retryable_query_error(expected))

    def test_failures_after_earlier_valid_pages_propagate_after_the_valid_rows(self):
        loci = tile_loci(FIRST_TILE, 30)
        valid = {None: page(loci[:10], next=link(2)), "2": page(loci[10:20], next=link(3))}

        def raise_reset(url):
            # G6.6.3D: only whitelisted requests exception types cross the process
            # boundary (by exact name); anything else is a terminal process error.
            raise requests.exceptions.ConnectionError("connection reset by peer")
        for name, third, kind, retryable in (
                ("http-503", {"status": 503, "body": b"busy"}, L.P2TransportHTTPError, True),
                ("non-json", {"body": b"<html>502</html>"}, requests.exceptions.JSONDecodeError, True),
                ("connection-reset", raise_reset, requests.exceptions.ConnectionError, True),
                ("insecure-next", page(loci[20:30], next=f"ftp://{HOST}/v1/loci?p=4"),
                 L.P2InsecureContinuationError, False),
                ("unexpected-206", {"status": 206, "body": b"{}"}, L.P2UnexpectedStatusError, False)):
            with self.subTest(name):
                service = FakeAntares(script=scripted({**valid, "3": third}))
                records, error = self.run_guarded(service)
                self.assertIsInstance(error, kind)
                self.assertEqual(L._retryable_query_error(error), retryable)
                self.assertEqual(len(records), 30 if name == "insecure-next" else 20)
                self.assertEqual(len(service.calls), 3)

    def test_consumer_stop_closes_without_further_requests(self):
        service = FakeAntares(tile_loci(FIRST_TILE, 80), page_size=10)
        with service.installed():
            iterator = guarded().search(self.body)
            first = [next(iterator) for _ in range(25)]
            self.assertEqual(len(service.calls), 3)
            iterator.close()
            with self.assertRaises(StopIteration):
                next(iterator)
        self.assertEqual((len(first), len(service.calls)), (25, 3))
        self.assertEqual(page_threads(), [])

    def test_limits_are_exact_finite_integers(self):
        for name, (low, high) in L._P2_TRANSPORT_BOUNDS.items():
            for value in (low - 1, high + 1, float(low), str(low), None, True, False):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        limits(**{name: value})
            limits(**{name: low})
            limits(**{name: high})
        pages = InProcessPages(CANARY.transport, requests.Session)
        with self.assertRaises(ValueError):
            L._GuardedListing(CANARY.transport.as_dict(), BASE, pages, listing_schema)
        with self.assertRaises(ValueError):  # a bare session factory is no page transport
            L._GuardedListing(CANARY.transport, BASE, requests.Session, listing_schema)
        with self.assertRaises(RuntimeError):
            L._GuardedListing(CANARY.transport, "https://evil.example/v1/", pages, listing_schema)
        with self.assertRaises(ValueError):
            L.P2ProofProfile(**{**{k: getattr(CANARY, k) for k in (
                "max_depth", "max_nodes_per_root", "max_nodes_per_night", "max_search_attempts",
                "crash_reserve", "max_event_bytes")}, "transport": CANARY.transport.as_dict()})


class LoopbackAdapter(HTTPAdapter):
    """Route the official origin to a loopback server over real sockets."""

    def __init__(self, port):
        super().__init__()
        self.port = port

    def send(self, request, **kwargs):
        original = request.url
        parts = urlsplit(original)
        if (parts.scheme, parts.hostname) != ("https", HOST):
            raise AssertionError(f"loopback adapter refused {original!r}")
        request.url = urlunsplit(("http", f"127.0.0.1:{self.port}", parts.path, parts.query, ""))
        try:
            response = super().send(request, **{**kwargs, "proxies": {}})
        finally:
            request.url = original
        response.url = original
        return response


class Loopback:
    """A real HTTP/1.1 server on 127.0.0.1 whose GET behaviour a test scripts."""

    def __init__(self, handle):
        outer = self
        self.paths = []

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                outer.paths.append(self.path)
                handle(self)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def session(self):
        session = requests.Session()
        session.mount("https://", LoopbackAdapter(self.port))
        return session


def send_json(handler, document, *, status=200, delay=0.0):
    time.sleep(delay)
    payload = json.dumps(document).encode()
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/vnd.api+json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
    except OSError:  # the client already abandoned this page
        handler.close_connection = True


def offset_of(handler):
    return int(parse_qs(urlsplit(handler.path).query).get("page[offset]", ["0"])[0])


def official_next(handler, offset):
    query = parse_qs(urlsplit(handler.path).query)
    return LISTING + "?" + urlencode([("sort", SORT), (ES_KEY, query[ES_KEY][0]), ("page[offset]", str(offset))])


class LoopbackSocketTests(OfflineTestCase):
    """Timeouts, deadlines and resets on real sockets (127.0.0.1 only), through the
    real transport child: a deadline kills it, and only a reaped status proves it dead."""

    body = L._build_tile_query(FIRST_TILE)
    loci = tile_loci(FIRST_TILE, 30)

    def serve(self, handle):
        server = Loopback(handle)
        self.addCleanup(server.close)
        return server

    def listing_page(self, handler, *, delay=0.0, size=10):
        offset = offset_of(handler)
        more = offset + size < len(self.loci)
        send_json(handler, {"data": [listing_resource(l) for l in self.loci[offset:offset + size]],
                            "links": {"next": official_next(handler, offset + size) if more else None}}, delay=delay)

    def assertKilled(self, listing, *, spawns=1):
        child = listing._transport
        self.assertEqual(child.state, T.FAILED)
        self.assertTrue(child.exit_record["reaped"], child.exit_record)
        self.assertIsNotNone(child.exit_record["returncode"])
        self.assertEqual(child.stats["spawns"], spawns)
        self.assertEqual(transport_children(), [])
        with self.assertRaises(ProcessLookupError):  # the pid no longer exists at all
            os.kill(child.exit_record["pid"], 0)
        # Never restarted: the failed transport refuses every further page.
        _records, error = drain(listing.search(self.body))
        self.assertIs(type(error), L.P2TransportUnavailableError)
        self.assertEqual(child.stats["spawns"], spawns)

    def test_socket_equivalence_with_same_origin_relative_redirect(self):
        def handle(handler):
            query = parse_qs(urlsplit(handler.path).query)
            if offset_of(handler) == 10 and "hop" not in query:
                handler.send_response(302)
                handler.send_header("Location", "/v1/loci?" + urlsplit(handler.path).query + "&hop=1")
                handler.send_header("Content-Length", "0")
                handler.end_headers()
                return
            self.listing_page(handler)
        server = self.serve(handle)
        listing = process_guarded(self, None, server.port)
        records, error = drain(listing.search(self.body))
        guarded_paths = list(server.paths)
        server.paths.clear()

        class LoopbackSession(requests.Session):
            def __init__(inner):
                super().__init__()
                inner.mount("https://", LoopbackAdapter(server.port))
        with mock.patch.object(requests.sessions, "Session", LoopbackSession):
            expected, expected_error = drain(pinned_search(self.body))
        self.assertIsNone(error)
        self.assertIsNone(expected_error)
        self.assertEqual(records, expected)
        self.assertEqual(len(records), 30)
        self.assertEqual(guarded_paths, server.paths)
        self.assertEqual(len(guarded_paths), 4)
        self.assertEqual((listing._transport.state, listing._transport.stats["spawns"]), (T.RUNNING, 1))
        self.assertEqual(listing._transport.stats["requests"], 3)  # one child, three pages
        listing.close()
        self.assertEqual((listing._transport.state, listing._transport.exit_record["returncode"]), (T.CLOSED, 0))
        self.assertEqual(transport_children(), [])

    def test_read_timeout_is_a_transport_failure(self):
        server = self.serve(lambda handler: self.listing_page(handler, delay=2.5))
        started = time.monotonic()
        listing = process_guarded(self, limits(read_timeout_seconds=1), server.port)
        records, error = drain(listing.search(self.body))
        self.assertLess(time.monotonic() - started, 2.4)
        self.assertIs(type(error), requests.exceptions.ReadTimeout)  # the exact type crosses
        self.assertTrue(L._retryable_query_error(error))
        self.assertEqual(records, [])
        self.assertEqual(listing._transport.state, T.RUNNING)  # a healthy child reported it

    def test_iterator_deadline_kills_a_stalled_page_and_the_child_is_reaped(self):
        server = self.serve(lambda handler: self.listing_page(handler, delay=3.0))
        started = time.monotonic()
        listing = process_guarded(self, limits(iterator_deadline_seconds=1), server.port)
        records, error = drain(listing.search(self.body))
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIs(type(error), L.P2DeadlineError)
        self.assertEqual(records, [])
        self.assertFalse(listing._transport.exit_record["escalated_to_sigkill"])
        self.assertKilled(listing)

    def test_slow_drip_body_defeats_read_timeout_but_not_the_iterator_deadline(self):
        state = {"sent": 0, "ended": None}

        def drip(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "application/vnd.api+json")
            handler.send_header("Content-Length", "100000")
            handler.end_headers()
            while True:
                try:
                    handler.wfile.write(b" ")
                    handler.wfile.flush()
                except OSError:
                    state["ended"] = time.monotonic()
                    break
                state["sent"] += 1
                time.sleep(.1)
            handler.close_connection = True
        server = self.serve(drip)
        started = time.monotonic()
        listing = process_guarded(self, limits(iterator_deadline_seconds=1, read_timeout_seconds=60), server.port)
        records, error = drain(listing.search(self.body))
        killed = time.monotonic()
        self.assertLess(killed - started, 2.0)
        self.assertIs(type(error), L.P2DeadlineError)
        self.assertEqual(records, [])
        self.assertKilled(listing)
        # The killed child's socket is gone: the drip stops by itself, nothing is read.
        deadline = time.monotonic() + 5.0
        while state["ended"] is None and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertIsNotNone(state["ended"])
        self.assertLess(state["ended"] - killed, 2.0)
        settled = state["sent"]
        time.sleep(.5)
        self.assertEqual(state["sent"], settled)

    def test_deadline_across_individually_fast_pages(self):
        self.loci = tile_loci(FIRST_TILE, 300)
        server = self.serve(lambda handler: self.listing_page(handler, delay=.4, size=1))
        listing = process_guarded(self, limits(iterator_deadline_seconds=1), server.port)
        records, error = drain(listing.search(self.body))
        self.assertIs(type(error), L.P2DeadlineError)
        self.assertLessEqual(len(server.paths), 4)  # each page alone is far inside its timeouts
        self.assertTrue(1 <= len(records) <= len(server.paths))
        self.assertKilled(listing)

    def test_connection_reset_after_earlier_valid_pages(self):
        def handle(handler):
            if offset_of(handler) < 20:
                return self.listing_page(handler)
            handler.send_response(200)
            handler.send_header("Content-Type", "application/vnd.api+json")
            handler.send_header("Content-Length", "5000")
            handler.end_headers()
            handler.wfile.write(b'{"data": [')
            handler.wfile.flush()
            handler.close_connection = True
            handler.connection.shutdown(socket.SHUT_RDWR)
        server = self.serve(handle)
        listing = process_guarded(self, None, server.port)
        records, error = drain(listing.search(self.body))
        self.assertIsInstance(error, (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError))
        self.assertTrue(L._retryable_query_error(error))
        self.assertEqual(len(records), 20)
        self.assertEqual(len(server.paths), 3)
        self.assertEqual(listing._transport.state, T.RUNNING)


FIRST_QUERY = json.dumps(L._build_tile_query(FIRST_TILE))


def first_tile_script(pages):
    """Pathological pages for the first canonical tile only; every other search is normal."""
    def script(service, request, index):
        params = parse_qs(urlsplit(request.url).query)
        if "p" in params:
            key = params["p"][0]
        elif params.get(ES_KEY, [None])[0] == FIRST_QUERY:
            key = None
        else:
            return None
        spec = pages[key]
        return spec(service, request, index) if callable(spec) else copy.deepcopy(spec)
    return script


class ArnorNight:
    """A sealed arnor-commissioning LIVE_ANTARES_READ below a temporary canary parent."""

    def __init__(self, test, night=NIGHT):
        folder = tempfile.TemporaryDirectory()
        test.addCleanup(folder.cleanup)
        self.parent = Path(folder.name).resolve() / "canary"
        self.parent.mkdir(mode=0o700)
        self.run_id = f"g663-test-{night}"
        self.root = self.parent / self.run_id
        self.root.mkdir(mode=0o700)
        for module in (L, B.storage):
            patcher = mock.patch.object(module, "ARNOR_CANARY_ROOT", self.parent)
            patcher.start()
            test.addCleanup(patcher.stop)
        self.capability = L.LiveAntaresReadCapability.for_arnor_commissioning(
            self.root, run_id=self.run_id, target_date_utc=night, release_sha=RELEASE,
            authority=L.LIVE_ANTARES_READ, hostname="arnor")
        self.adapter = R.LiveRangeAdapter(self.root, RELEASE, None, proof_profile=CANARY)
        self.request = self.adapter.acquisition_request(night)

    def provider(self, profile=CANARY):
        return L.LiveAntaresProvider(self.capability, proof_profile=profile, sleeper=lambda _: None,
                                     clock=F.fixed_clock, monotonic=lambda: 0.0)

    def bindings(self):
        configuration = B.configuration_sha256(RELEASE, self.adapter, 256)
        return B.query_checkpoint_bindings(self.run_id, RELEASE, configuration, self.adapter, self.request)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def guarded_identity(transport=None):
    return {"distribution": "antares-client", "version": "1.14.0", "api_base_url": BASE,
            "api_timeout_seconds": 60, "authentication": "public-search-no-credentials",
            "pagination_contract": "p2-guarded-jsonapi-links-next-v2",
            "transport_sha256": L._sha256_json((transport or CANARY.transport).as_dict())}


class GuardedP2NightTests(OfflineTestCase):
    """Whole P2 nights through the real guarded wiring on an Arnor capability."""

    def query(self, night, service, profile=CANARY):
        provider = night.provider(profile)
        with service.installed():
            result = provider.query(night.request)
        self.assertOfficial(service.urls)
        self.assertEqual(page_threads(), [])
        self.assertEqual(transport_children(), [])  # the night's child ended with its query
        return provider, result

    def test_floor_saturated_night_completes_identically_to_qualified_callbacks(self):
        night = ArnorNight(self)
        loci = dense_loci()
        service = FakeAntares(loci, page_size=10)
        provider, result = self.query(night, service)
        self.assertTrue(result.clean, result.evidence.errors)
        replay = S.validate_p2_query_result(night.request, result, CANARY)
        details = result.evidence.details
        self.assertEqual(details["client"], guarded_identity())
        self.assertEqual(details["client"]["transport_sha256"], L.G663_CANARY_TRANSPORT_SHA256)
        self.assertEqual(details["capability_environment"], "arnor-commissioning")
        self.assertEqual(details["p2_budget"]["secondary_nodes"], 3)
        self.assertEqual(result.loci.locus_id.tolist(), [locus.locus_id for locus in loci])
        self.assertTrue(replay["frame"].equals(result.loci))
        self.assertTrue(all(call["timeout"] == (60, 60) and call["stream"] for call in service.calls))
        # The G6.6.2B-qualified callback path (same profile, same service order) makes
        # exactly the same decisions: identical journal events and science.
        capability = F.mock_read_capability(night.root, night.run_id, NIGHT, RELEASE)
        callbacks = L.LiveAntaresProvider(capability, proof_profile=CANARY, sleeper=lambda _: None,
            clock=F.fixed_clock, monotonic=lambda: 0.0,
            search_fn=lambda body: [l for l in loci if _body_matches_locus(body, l)],
            get_by_id_fn=lambda _: None, connectivity_fn=lambda: [])
        expected = callbacks.query(night.request)
        self.assertEqual(canonical_hash(details["p2_events"]), canonical_hash(expected.evidence.details["p2_events"]))
        pd.testing.assert_frame_equal(result.loci, expected.loci)
        self.assertEqual(details["search_request_count"], 6912 + 2 * details["split_count"])
        # Fetch is the unchanged pinned get_by_id path; the night builds and reopens as science.
        with service.installed():
            fetched = provider.fetch(night.request, result)
        self.assertTrue(fetched.fetch_evidence.clean)
        self.assertEqual(sum("/v1/loci/" in url for url in service.urls), len(loci))
        with mock.patch.object(S, "QUALIFIED_P2_PROFILES", (CANARY,)):
            artifacts = S.build_night_artifacts(fetched)
            S.reopen_and_validate_artifacts(artifacts, expected=fetched)
            manifest = json.loads(artifacts["manifest.json"])
            for mutation in (
                    lambda m: m["query_evidence"]["details"]["client"].update(transport_sha256="0" * 64),
                    lambda m: m["query_evidence"]["details"]["client"].update(
                        pagination_contract="jsonapi-links-next-until-null"),
                    lambda m: m["query_evidence"]["details"].update(capability_environment="local-mock")):
                with self.subTest(mutation=mutation):
                    changed = copy.deepcopy(manifest)
                    mutation(changed)
                    with self.assertRaises(ArtifactValidationError):
                        S.reopen_and_validate_artifacts({**artifacts, "manifest.json": json.dumps(changed).encode()})
        with mock.patch.object(S, "QUALIFIED_P2_PROFILES", ()):
            with self.assertRaises(ArtifactValidationError):
                S.reopen_and_validate_artifacts(artifacts)

    def test_non_floor_night_equals_p1_pinned_client_request_for_request(self):
        night = ArnorNight(self)
        anchor = (61218.1, 258.02, -23.36)
        loci = [FakeLocus(f"N-{n:03d}", mjd=anchor[0] + (n % 7) * 1e-3, ra=anchor[1] + (n % 11) * .17,
                          dec=anchor[2] + (n % 13) * .11) for n in range(140)]
        loci += [FakeLocus("DUP", ra=10.0), FakeLocus("A", ra=10.0), FakeLocus("DUP", ra=250.0)]
        service = FakeAntares(loci, page_size=10)
        _p2, guarded_result = self.query(night, service)
        guarded_urls = list(service.urls)
        service.calls.clear()
        p1 = L.LiveAntaresProvider(night.capability, sleeper=lambda _: None, clock=F.fixed_clock,
                                   monotonic=lambda: 0.0)
        with service.installed():
            pinned_result = p1.query(night.request)
        self.assertTrue(guarded_result.clean)
        self.assertTrue(pinned_result.clean)
        self.assertEqual(guarded_urls, service.urls)  # the same HTTP requests, in the same order
        pd.testing.assert_frame_equal(guarded_result.loci, pinned_result.loci)
        for key in ("tile_trace", "tile_trace_sha256", "locus_order_sha256", "split_count",
                    "accepted_tile_count", "deduplication", "raw_returned_loci", "returned_loci"):
            self.assertEqual(guarded_result.evidence.details[key], pinned_result.evidence.details[key], key)
        self.assertEqual(guarded_result.evidence.details["p2_budget"]["secondary_nodes"], 0)
        self.assertGreater(guarded_result.evidence.details["split_count"], 0)
        self.assertEqual(pinned_result.evidence.details["client"]["pagination_contract"],
                         "jsonapi-links-next-until-null")
        self.assertEqual(guarded_result.loci[guarded_result.loci.locus_id == "DUP"].ra.tolist(), [250.0])

    def test_transient_failure_after_valid_pages_retries_the_whole_node(self):
        night = ArnorNight(self)
        first = tile_loci(FIRST_TILE, 25)
        state = {"failed": 0}

        def third(service, request, index):
            if state["failed"] == 0:
                state["failed"] += 1
                return {"status": 502, "body": b"<html>502 Bad Gateway</html>"}
            return page(first[20:], next=None)
        pages = {None: page(first[:10], next=link(2)), "2": page(first[10:20], next=link(3)), "3": third}
        provider, result = self.query(night, FakeAntares(first, script=first_tile_script(pages)))
        clean_provider, clean = self.query(ArnorNight(self), FakeAntares(first, page_size=10))
        self.assertTrue(result.clean, result.evidence.errors)
        S.validate_p2_query_result(night.request, result, CANARY)
        details = result.evidence.details
        self.assertEqual((details["retry_count"], details["partial_rows_discarded"]), (1, 20))
        self.assertEqual(details["retry_exception_types"], [GUARD + "P2TransportHTTPError"])
        pd.testing.assert_frame_equal(result.loci, clean.loci)
        self.assertEqual(details["locus_order_sha256"], clean.evidence.details["locus_order_sha256"])


def pages_of(loci, keys):
    """Chain ``keys`` (None first) with one locus slice each."""
    pages = {}
    for position, (key, rows) in enumerate(zip(keys, loci)):
        following = keys[position + 1] if position + 1 < len(keys) else None
        pages[key] = page(rows, next=link(following) if following is not None else None)
    return pages


class FailClosedMatrixTests(OfflineTestCase):
    """Every transport refusal or failure ends the P2 night incomplete, provably."""

    first = tile_loci(FIRST_TILE, 25)

    def cases(self):
        f = self.first
        ceiling = CANARY.transport.max_page_bytes

        def padded(rows, target):
            raw = json.dumps({"data": [listing_resource(l) for l in rows], "links": {"next": target}}).encode()
            return {"body": raw + b" " * (ceiling - 1 - len(raw))}

        def raising(error):
            def script(service, request, index):
                raise error
            return script
        advancing_pattern = {None: page(f[:1], next=link(1))}
        for n in range(1, 120):
            advancing_pattern[str(n)] = page(f[n // 3:n // 3 + 1] if n % 3 == 0 else (), next=link(n + 1))
        guard, retry, malformed = "p2_transport_guard", "p2_retry_exhausted", "p2_malformed"
        return [
            # name, pages, exception type, reason, validated rows, attempts, HTTP requests
            # G6.6.4-R3: retryable failures now use all 4 attempts; guard/malformed stay at 1.
            ("self-loop", {None: lambda s, r, i: page(f[:10], next=r.url)},
             GUARD + "P2ContinuationCycleError", guard, 10, 1, 1),
            ("two-page-cycle", {None: page(f[:10], next=link("a")), "a": page(f[10:20], next=link("b")),
                                "b": page(f[20:25], next=link("a"))}, GUARD + "P2ContinuationCycleError", guard, 25, 1, 3),
            ("longer-cycle", {**pages_of([f[0:5], f[5:10], f[10:15], f[15:20]], [None, "a", "b", "c"]),
                              "c": page(f[15:20], next=link("a"))}, GUARD + "P2ContinuationCycleError", guard, 20, 1, 4),
            ("ever-advancing-empty", {None: page(next=link(1)), **{str(n): page(next=link(n + 1)) for n in range(1, 9)}},
             GUARD + "P2EmptyPageLimitError", guard, 0, 1, 3),
            ("page-ceiling", advancing_pattern, GUARD + "P2PageLimitError", guard, 22, 1, 64),
            ("page-byte-ceiling", {None: {"body": b"{}", "headers": {"Content-Length": str(ceiling + 1)}}},
             GUARD + "P2PageBytesError", guard, 0, 1, 1),
            ("iterator-byte-ceiling", {None: padded(f[0:1], link(1)), **{str(n): padded(f[n:n + 1], link(n + 1))
                                                                       for n in range(1, 5)}},
             GUARD + "P2IteratorBytesError", guard, 4, 1, 5),
            ("read-timeout", {None: raising(requests.exceptions.ReadTimeout("read timed out"))},
             "requests.exceptions.ReadTimeout", retry, 0, 4, 4),
            ("connect-timeout", {None: raising(requests.exceptions.ConnectTimeout("connect timed out"))},
             "requests.exceptions.ConnectTimeout", retry, 0, 4, 4),
            ("malformed-json", {None: {"body": b"<html>gateway</html>"}},
             "requests.exceptions.JSONDecodeError", retry, 0, 4, 4),
            ("malformed-jsonapi-type", {None: {"json_body": {"data": [{"type": "wrong", "id": "x", "attributes": {}}]}}},
             "marshmallow_jsonapi.exceptions.IncorrectTypeError", malformed, 0, 1, 1),
            ("malformed-jsonapi-null-data", {None: {"json_body": {"data": None}}},
             "marshmallow.exceptions.ValidationError", retry, 0, 4, 4),
            ("malformed-continuation", {None: page(f[:10], next=7)},
             GUARD + "P2MalformedContinuationError", guard, 10, 1, 1),
            ("relative-continuation", {None: page(f[:10], next="/v1/loci?p=2")},
             GUARD + "P2RelativeContinuationError", guard, 10, 1, 1),
            ("unsupported-continuation-scheme", {None: page(f[:10], next=f"ftp://{HOST}/v1/loci?p=2")},
             GUARD + "P2InsecureContinuationError", guard, 10, 1, 1),
            ("wrong-origin", {None: page(f[:10], next="https://evil.example/v1/loci?p=2")},
             GUARD + "P2CrossOriginContinuationError", guard, 10, 1, 1),
            ("unsafe-redirect", {None: {"status": 302, "headers": {"Location": "https://evil.example/v1/loci"}}},
             GUARD + "P2UnsafeRedirectError", guard, 0, 1, 1),
            ("redirect-loop", {None: {"status": 302, "headers": {"Location": "/v1/loci?p=loop"}},
                               "loop": {"status": 302, "headers": {"Location": "/v1/loci?p=loop"}}},
             GUARD + "P2RedirectLimitError", guard, 0, 1, 3),
            ("http-503-after-valid-pages", {**pages_of([f[:10], f[10:20]], [None, "2"]), "2": page(f[10:20], next=link(3)),
                                            "3": {"status": 503, "body": b"busy"}},
             GUARD + "P2TransportHTTPError", retry, 20, 4, 12),
            ("reset-after-valid-pages", {None: page(f[:10], next=link(2)), "2": page(f[10:20], next=link(3)),
                                         "3": raising(requests.exceptions.ConnectionError("reset"))},
             "requests.exceptions.ConnectionError", retry, 20, 4, 12),
            ("unexpected-status-after-valid-pages", {None: page(f[:10], next=link(2)), "2": page(f[10:20], next=link(3)),
                                                     "3": {"status": 206, "body": b"{}"}},
             GUARD + "P2UnexpectedStatusError", guard, 20, 1, 3),
        ]

    def test_every_refusal_is_incomplete_deterministic_and_never_science(self):
        for name, pages, exception_type, reason, rows, attempts, requests_made in self.cases():
            with self.subTest(name):
                self.check(name, pages, exception_type, reason, rows, attempts, requests_made)

    def test_iterator_deadline_ends_the_night(self):
        clock = {"now": 0.0}

        def slow_first(service, request, index):
            clock["now"] += CANARY.transport.iterator_deadline_seconds
            return page(self.first[:10], next=link(2))
        with mock.patch.object(L, "time", types.SimpleNamespace(monotonic=lambda: clock["now"])):
            self.check("deadline", {None: slow_first, "2": page(self.first[10:], next=None)},
                       GUARD + "P2DeadlineError", "p2_transport_guard", 10, 1, 1)

    def check(self, name, pages, exception_type, reason, rows, attempts, requests_made):
        night = ArnorNight(self)
        service = FakeAntares(self.first, script=first_tile_script(pages))
        provider = night.provider()
        with service.installed():
            result = provider.query(night.request)
        self.assertOfficial(service.urls)
        self.assertEqual(len(service.calls), requests_made)
        self.assertEqual(page_threads(), [])
        self.assertEqual(transport_children(), [])
        # 1. incomplete, never complete: typed terminal evidence
        self.assertFalse(result.clean)
        self.assertEqual(result.outcome, ProviderOutcome.QUERY_INTERRUPTION)
        self.assertEqual((result.evidence.completed, result.evidence.partial), (False, True))
        self.assertEqual([issue.code for issue in result.evidence.errors], [reason])
        details = result.evidence.details
        self.assertEqual(details["completion_classification"], "INCOMPLETE")
        self.assertEqual((details["coverage_complete"], details["iterator_exhausted"],
                          details["terminal_evidence"]), (False, False, reason))
        self.assertEqual(details["client"], guarded_identity())
        events = details["p2_events"]
        outcomes = [event["p2_event"] for event in events if event["p2_event"]["kind"] == "outcome"]
        self.assertEqual(len(outcomes), attempts)
        for outcome in outcomes:
            decision = outcome["decision"]
            self.assertEqual(outcome["node"]["id"], "i00000")
            self.assertEqual((decision["status"], decision["exception_type"], decision["validated_rows"],
                              decision["discarded_rows"], decision["children"]),
                             ("attempt_error", exception_type, rows, rows, []))
        self.assertEqual(outcomes[-1]["decision"]["reason"], reason)
        from src.operations.query_progress import decode_records
        self.assertTrue(all(decode_records(event["records"]) == [] for event in events))
        # 2. no partial science: nothing accepted, nothing sealable, nothing fetchable
        self.assertTrue(result.loci.empty)
        self.assertEqual((details["accepted_tile_count"], details["returned_loci"]), (0, 0))
        with self.assertRaises(QueryInterruptedError):
            result.require_completed()
        with self.assertRaises(ArtifactValidationError):
            S.validate_p2_query_result(night.request, result, CANARY)
        with self.assertRaises(ArtifactValidationError):
            S._p2_replay_events(events, night.request.mjd_min, night.request.mjd_max, CANARY, 2)
        with self.assertRaises(Q.QueryCheckpointError):
            Q.seal_query_result_checkpoint(night.root, result, night.bindings())
        self.assertFalse((night.root / "checkpoints").exists())
        service.calls.clear()
        with service.installed():
            fetched = provider.fetch(night.request, result)
        self.assertFalse(fetched.fetch_evidence.completed)
        self.assertEqual(service.calls, [])
        # 3. deterministic: the journal alone replays the same refusal, without any request
        with service.installed():
            replayed = night.adapter.replay_query_journal(night.request, events)
        self.assertEqual(service.calls, [])
        for key in ("p2_events", "p2_budget", "tile_trace", "terminal_evidence", "retry_exception_types",
                    "partial_rows_discarded", "search_request_count"):
            self.assertEqual(replayed.evidence.details[key], details[key], key)
        self.assertEqual([issue.code for issue in replayed.evidence.errors], [reason])
        self.assertFalse(replayed.clean)


CANARY_CANONICAL = (
    '{"acceptance":"natural-exhaustion-below-50","axis_ties":["time","ra","dec"],"crash_reserve":4,'
    '"grammar":"v3.p2-query-decisions.v1","max_depth":18,"max_event_bytes":33554432,'
    '"max_nodes_per_night":4095,"max_nodes_per_root":511,"max_search_attempts":250000,'
    '"midpoint":"binary64-(lower+upper)/2","primary":{"cache_version":"probe50_time_ra_dec_v1","dec_bins":6,'
    '"min_dec_degrees":0.05,"min_ra_degrees":0.05,"min_time_seconds":30.0,"name":"probe_first_time_ra_dec",'
    '"probe_limit":50,"probe_threshold":50,"ra_bins":24,"time_bin_minutes":30},'
    '"schema_version":"v3.p2-proof-profile.v2",'
    '"transport":{"child_implementation_sha256":"2e661e528899c6a6ac02eb6af38efbcc23ad9e5cad2e083963c9f9a24cea5bee",'
    '"child_shutdown_seconds":5,"child_startup_seconds":30,"connect_timeout_seconds":60,'
    '"continuation":"absolute-http-default-port-or-https-default-port;canonical-trusted-https-same-listing-raw-query;prepared-page-never-repeated",'
    '"ipc":"g663d.p2-transport-ipc.v1","ipc_max_chunk_bytes":1048576,"ipc_max_header_bytes":65536,'
    '"ipc_wait":"remaining-iterator-deadline",'
    '"isolation":"dedicated-child-process-per-night-query;parent-owned-deadline;sigterm-grace-sigkill-join-reaped;never-restarted-after-failure",'
    '"iterator_deadline_seconds":600,"kill_join_seconds":5,"limit_semantics":"refusal-never-completeness",'
    '"max_consecutive_empty_pages":2,"max_iterator_bytes":67108864,"max_page_bytes":16777216,"max_pages":64,'
    '"max_redirects":2,"pagination":"p2-guarded-jsonapi-links-next-v2","read_timeout_seconds":60,'
    '"redirects":"same-origin-https-api-prefix-validated-before-request",'
    '"schema_version":"v3.p2-transport-limits.v2","terminate_grace_seconds":2,'
    '"termination":"complete-page-then-links-next-null-or-missing"},"traversal":"lower-child-first",'
    '"trigger":"saturated_primary_floor"}')
CANARY_PROFILE_SHA256 = "39b0ff54bcbb5be3d9c627365dcb3cf5b6ef59cd266842575ab7febd9441dec3"
CANARY_TRANSPORT_SHA256 = "85f278a495a455fbb652561ce9a147f092f7f58510e980322350afa2f6c0e716"


def baseline_module(sha, path, alias):
    import sys
    payload = subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=ROOT)
    module = types.ModuleType(alias)
    module.__package__ = "src.operations"
    module.__file__ = str(ROOT / path)
    sys.modules[alias] = module
    exec(compile(payload, f"<{sha[:7]}:{path}>", "exec"), module.__dict__)
    return module


class FrozenProfileTests(OfflineTestCase):
    """The one canary identity: literal bytes, digests, and refusal to drift."""

    def test_canonical_bytes_and_digests_are_frozen(self):
        profile = L.g663_canary_p2_profile()
        self.assertEqual(L._canonical_json(profile.as_dict()).decode(), CANARY_CANONICAL)
        self.assertEqual(hashlib.sha256(CANARY_CANONICAL.encode()).hexdigest(), CANARY_PROFILE_SHA256)
        self.assertEqual(L.G663_CANARY_P2_PROFILE_SHA256, CANARY_PROFILE_SHA256)
        self.assertEqual(L._sha256_json(profile.transport.as_dict()), CANARY_TRANSPORT_SHA256)
        self.assertEqual(L.G663_CANARY_TRANSPORT_SHA256, CANARY_TRANSPORT_SHA256)
        self.assertEqual(S._phase6_json_hash(profile.as_dict()), CANARY_PROFILE_SHA256)  # science's own hashing
        self.assertEqual(profile, L.g663_canary_p2_profile())
        method = profile.extraction_method()
        self.assertEqual((method["name"], method["cache_version"]), ("hierarchical_completeness_p2", "guarded_p2_v1"))
        self.assertEqual(method["proof_profile"], profile.as_dict())
        for constant in ("G663_CANARY_P2_PROFILE_SHA256", "G663_CANARY_TRANSPORT_SHA256"):
            with mock.patch.object(L, constant, "0" * 64):
                with self.assertRaises(RuntimeError):
                    L.g663_canary_p2_profile()

    def test_every_journal_event_and_contract_binds_the_frozen_profile(self):
        night = ArnorNight(self)
        provider = night.provider()
        self.assertEqual(provider.execution_policy()["extraction_method"]["proof_profile"]["transport"],
                         CANARY.transport.as_dict())
        contract = provider.scientific_contract(night.request)
        self.assertEqual(contract["extraction_method"], CANARY.extraction_method())
        state = L._P2State(night.request, CANARY, L._make_initial_tiles(61218.0, 61219.0), 2)
        self.assertEqual(state.base("reserve")["profile_sha256"], CANARY_PROFILE_SHA256)
        # Jul07/Jul13 identities under the frozen profile.  G6.6.3D changed them (the profile
        # now binds the process boundary); G6.6.3A recorded e1e28e86... and 08ab9c77....
        digests = {}
        for day in ("2026-07-07", "2026-07-13"):
            request = R.LiveRangeAdapter(ROOT, RELEASE, None).acquisition_request(day)
            p2 = L.scientific_contract_for_profile(request, CANARY)
            p1 = L.scientific_contract_for_profile(request)
            self.assertEqual({k: v for k, v in p2.items() if k != "extraction_method"},
                             {k: v for k, v in p1.items() if k != "extraction_method"})
            self.assertNotIn("extraction_method", B.selection_descriptor_for_request(request))
            digests[day] = L._sha256_json(p2)
        self.assertEqual(digests, {
            "2026-07-07": "bacf32185b1eff128c705befaba4d34359a60012388a2708b392a18bac0a4e48",
            "2026-07-13": "aee7ec99d1cc3fc54cf8bf3f270bfebff53a5cacb8e2104f3aca8aa16f1e1891"})

    def test_offline_g662b_profile_identity_is_unchanged(self):
        old = baseline_module(G662B, "src/operations/live_antares.py", "src.operations._g663_g662b_provider")
        values = dict(max_depth=4, max_nodes_per_root=31, max_nodes_per_night=128,
                      max_search_attempts=16000, crash_reserve=3, max_event_bytes=1000000)
        self.assertEqual(L.P2ProofProfile(**values).as_dict(), old.P2ProofProfile(**values).as_dict())
        self.assertEqual(L.P2ProofProfile(**values).extraction_method(), old.P2ProofProfile(**values).extraction_method())
        self.assertEqual(L.extraction_method_contract(), old.extraction_method_contract())

    def test_no_registry_trusts_the_canary_profile(self):
        self.assertEqual(S.QUALIFIED_P2_PROFILES, ())
        self.assertEqual([profile.name for profile in B.QUALIFIED_SOURCE_PROFILES], ["historical-P1"])
        self.assertIsNone(B.QUALIFIED_SOURCE_PROFILES[0].proof_profile)
        self.assertEqual(B.QUALIFIED_SOURCE_PROFILES[0].provider_sha256s, B.ADOPTABLE_SOURCE_PROVIDERS)
        self.assertNotIn(hashlib.sha256((ROOT / "src/operations/live_antares.py").read_bytes()).hexdigest(),
                         B.ADOPTABLE_SOURCE_PROVIDERS)

    def test_depth_18_cells_never_collapse_at_domain_extremes(self):
        tiles = L._make_initial_tiles(61228.0, 61229.0)  # 2026-07-07
        extremes = [tiles[0], tiles[-1], next(t for t in tiles if t["ra_max"] == 360.0),
                    next(t for t in reversed(tiles) if t["dec_min"] == -90.0)]
        for start in extremes:
            for side in (0, 1):
                tile = start
                while L._split_tile(tile):
                    tile = L._split_tile(tile)[side]
                widths = ((tile["mjd_max"] - tile["mjd_min"]) * 86400, tile["ra_max"] - tile["ra_min"])
                self.assertAlmostEqual(widths[0], 28.125, places=5)
                self.assertEqual(widths[1], .029296875)
                for _depth in range(CANARY.max_depth):
                    _axis, _middle, children = L._p2_split(tile)
                    tile = children[side]
                self.assertAlmostEqual((tile["ra_max"] - tile["ra_min"]) * 3600, 1.6479, places=3)
                self.assertAlmostEqual((tile["mjd_max"] - tile["mjd_min"]) * 86400, .4395, places=3)


class GuardedWiringTests(OfflineTestCase):
    """Only a sealed Arnor capability with a guarded profile reaches the paginator."""

    def test_capability_and_profile_gating(self):
        night = ArnorNight(self)
        service = FakeAntares()
        with service.installed():
            search, get_by_id, connectivity = night.provider()._load_client()
            self.assertIs(type(search.__self__), L._GuardedListing)
            self.assertIs(search.__self__.limits, CANARY.transport)
            # G6.6.3D: the paginator's pages go through a process transport that
            # starts lazily; loading the client starts nothing.
            child = search.__self__._transport
            self.assertIs(type(child), T.P2ProcessTransport)
            self.assertEqual((child.state, child.stats["spawns"]), (T.NOT_STARTED, 0))
            self.assertEqual(child._config, L._p2_child_config(CANARY.transport, BASE))
            self.assertEqual(transport_children(), [])
            from antares_client import search as client
            self.assertIs(get_by_id, client.get_by_id)
            self.assertIs(connectivity, client.get_available_tags)
            # P1 on the same capability: the pinned client's own unbounded search, unchanged.
            p1 = L.LiveAntaresProvider(night.capability)
            self.assertIs(p1._load_client()[0], client.search)
            self.assertEqual(p1.client_identity()["pagination_contract"], "jsonapi-links-next-until-null")
            self.assertEqual(night.provider().client_identity(), guarded_identity())
            with self.assertRaises(L.LiveCapabilityError):  # the offline G6.6.2B profile never searches live
                night.provider(test_profile())._load_client()
            local = F.mock_read_capability(night.root, night.run_id, NIGHT, RELEASE)
            with self.assertRaises(L.LiveCapabilityError):
                L.LiveAntaresProvider(local, proof_profile=CANARY)._load_client()
            forged = night.provider()
            forged.capability = types.SimpleNamespace(environment="arnor-commissioning")
            with self.assertRaises(L.LiveCapabilityError):
                forged._load_client()
            with self.assertRaises(L.LiveCapabilityError):
                L.LiveAntaresProvider(night.capability, proof_profile=CANARY, search_fn=lambda _: [],
                                      get_by_id_fn=lambda _: None, connectivity_fn=lambda: [])
        self.assertEqual(service.calls, [])

    def test_pinned_client_configuration_drift_refuses(self):
        night = ArnorNight(self)
        from antares_client.config import config
        for key, value in (("ANTARES_API_BASE_URL", "https://evil.example/v1/"),
                           ("ANTARES_API_BASE_URL", f"http://{HOST}/v1/"), ("API_TIMEOUT", 61)):
            with self.subTest(key=key, value=value), mock.patch.dict(config, {key: value}):
                with self.assertRaises(RuntimeError):
                    night.provider()._load_client()
        with mock.patch.object(L.metadata, "version", lambda name: "1.14.1"):
            with self.assertRaises(RuntimeError):
                night.provider()._load_client()

    def test_science_qualifies_the_guarded_identity_only_for_its_exact_profile(self):
        night = ArnorNight(self)
        service = FakeAntares(dense_loci(), page_size=10)
        with service.installed():
            result = night.provider().query(night.request)
        S.validate_p2_query_result(night.request, result, CANARY)
        other = L.P2ProofProfile(**{**{k: getattr(CANARY, k) for k in (
            "max_depth", "max_nodes_per_root", "max_nodes_per_night", "max_search_attempts",
            "crash_reserve", "max_event_bytes")}, "transport": limits(max_pages=65)})
        for profile in (other, test_profile()):
            with self.subTest(profile=profile):
                with self.assertRaises(ArtifactValidationError):
                    S.validate_p2_query_result(night.request, result, profile)
        for edit in ({"capability_environment": "local-mock"},
                     {"client": {**guarded_identity(), "transport_sha256": "0" * 64}},
                     {"client": {**guarded_identity(), "pagination_contract": "jsonapi-links-next-until-null"}},
                     {"client": {k: v for k, v in guarded_identity().items() if k != "transport_sha256"}}):
            with self.subTest(edit=sorted(edit)):
                forged = copy.copy(result)
                object.__setattr__(forged, "evidence", type(result.evidence)(
                    result.evidence.completed, result.evidence.partial, result.evidence.returned_loci,
                    result.evidence.errors, {**result.evidence.details, **edit}))
                with self.assertRaises(ArtifactValidationError):
                    S.validate_p2_query_result(night.request, forged, CANARY)


class G663FirewallTests(unittest.TestCase):
    """P1 runtime paths are byte-for-byte the accepted G6.6.2B candidate's."""

    G663_LIVE_NEW = {
        "P2TransportGuardError", "P2PageLimitError", "P2EmptyPageLimitError", "P2PageBytesError",
        "P2IteratorBytesError", "P2DeadlineError", "P2MalformedContinuationError",
        "P2RelativeContinuationError", "P2InsecureContinuationError", "P2CrossOriginContinuationError",
        "P2ContinuationPathError", "P2ContinuationCycleError", "P2UnsafeRedirectError",
        "P2RedirectLimitError", "P2UnexpectedStatusError", "P2TransportHTTPError", "P2TransportLimits",
        "_GuardedListing", "_p2_client_identity", "_load_p2_client", "g663_canary_p2_profile",
        # G6.6.3D process boundary (P2 only)
        "P2TransportProcessError", "P2TransportStartupError", "P2TransportProtocolError",
        "P2TransportUnavailableError", "P2TransportUnkillableError", "_p2_remote_request_errors",
        "_p2_transport_error", "_p2_child_config", "_canonical_p2_continuation"}

    @staticmethod
    def definitions(source):
        tree = ast.parse(source)
        named = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                named[node.name] = ast.get_source_segment(source, node)
            elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
                for target in node.targets:
                    named["=" + target.id] = ast.get_source_segment(source, node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                named["import:" + ast.get_source_segment(source, node)] = ast.get_source_segment(source, node)
        return tree, named

    def baseline(self, path):
        return subprocess.check_output(["git", "show", f"{G662B}:{path}"], cwd=ROOT).decode()

    def test_only_the_p2_runtime_files_differ_from_g662b(self):
        changed = subprocess.check_output(["git", "diff", "--name-only", G662B, "--", "src"], cwd=ROOT).decode().split()
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", "src"],
                                            cwd=ROOT).decode().split()
        self.assertEqual(sorted(changed + untracked), ["src/operations/live_antares.py", "src/operations/p2_transport.py",
                                                       "src/operations/science.py"])
        for path in ("src/operations/publication.py", "src/operations/transaction.py", "src/operations/storage.py",
                     "src/operations/commissioning.py", "src/operations/query_checkpoint.py",
                     "src/operations/query_progress.py", "src/operations/fetch_checkpoint.py", "src/history.py",
                     "src/query.py", "src/operations/backfill.py", "src/operations/production_range.py"):
            self.assertEqual((ROOT / path).read_text(), self.baseline(path), path)

    def test_live_antares_p1_definitions_are_unchanged(self):
        old_source, new_source = self.baseline("src/operations/live_antares.py"), (ROOT / "src/operations/live_antares.py").read_text()
        _old_tree, old = self.definitions(old_source)
        new_tree, new = self.definitions(new_source)
        changed = {name for name in old if name in new and old[name] != new[name]}
        removed = set(old) - set(new)
        added = set(new) - set(old)
        self.assertEqual(removed, {"import:from urllib.parse import urlsplit, urlunsplit"})
        self.assertEqual(changed, {"LiveAntaresProvider", "P2ProofProfile", "_P2State", "_run_p2_query"})
        self.assertEqual({name for name in added if not name.startswith(("=", "import:"))}, self.G663_LIVE_NEW)
        self.assertEqual({name for name in added if name.startswith("import:")},
                         {"import:from . import p2_transport as _p2_transport",
                          "import:from urllib.parse import urljoin, urlsplit, urlunsplit"})
        # G6.6.3D: the P2 reducer only gained a try/finally that ends the night's child.
        self.assertIn(
            "    finally:\n"
            "        # G6.6.3D: the night's one transport child ends with its query.  A child\n"
            "        # that cannot be proven dead raises here; nothing continues past it.\n"
            "        listing = getattr(search, \"__self__\", None)\n"
            "        if isinstance(listing, _GuardedListing):\n"
            "            listing.close()\n", new["_run_p2_query"])
        reverted = new["_run_p2_query"].replace(
            "    finally:\n"
            "        # G6.6.3D: the night's one transport child ends with its query.  A child\n"
            "        # that cannot be proven dead raises here; nothing continues past it.\n"
            "        listing = getattr(search, \"__self__\", None)\n"
            "        if isinstance(listing, _GuardedListing):\n"
            "            listing.close()\n", "")
        body_start = reverted.index("    try:\n        while state.frontier")
        body_end = reverted.index("\n\n    raw = pd.DataFrame(")
        body = reverted[body_start + len("    try:\n"):body_end]
        dedented = "\n".join(line[4:] if line.strip() else line for line in body.splitlines())
        # G6.6.4-R3: exactly three P2-only edits (frozen 4-attempt/5 s policy and its evidence record).
        r3_edits = (
            ("P2_MAX_QUERY_ATTEMPTS)\n    events = list(", "provider.max_query_attempts)\n    events = list("),
            ("provider.sleeper(P2_RETRY_DELAY_SECONDS * state", "provider.sleeper(provider.retry_delay_seconds * state"),
            ('        "p2_retry_policy": {"max_attempts_per_tile": P2_MAX_QUERY_ATTEMPTS, '
             '"backoff_seconds_per_attempt": P2_RETRY_DELAY_SECONDS},\n', ""),
        )
        r3_body = reverted[:body_start] + dedented + reverted[body_end:]
        for edited, original in r3_edits:
            self.assertEqual(r3_body.count(edited), 1, edited)
            r3_body = r3_body.replace(edited, original)
        self.assertEqual(r3_body, old["_run_p2_query"])
        self.assertTrue(all(name[1:].startswith(("P2_", "_P2_", "G663_")) for name in added if name.startswith("=")))
        # Inside the provider class only _load_client differs, and only by its P2 branch.
        def methods(source):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "LiveAntaresProvider")
            return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}, source
        (old_methods, _), (new_methods, _) = methods(old_source), methods(new_source)
        self.assertEqual(set(old_methods), set(new_methods))
        for name in old_methods:
            same = ast.get_source_segment(old_source, old_methods[name]) == ast.get_source_segment(new_source, new_methods[name])
            self.assertEqual(same, name != "_load_client", name)
        old_body, new_body = old_methods["_load_client"].body, new_methods["_load_client"].body
        self.assertEqual([ast.get_source_segment(old_source, n) for n in old_body[1:]],
                         [ast.get_source_segment(new_source, n) for n in new_body[1:]])
        for body, source in ((old_body, old_source), (new_body, new_source)):  # the P2-only guard
            self.assertIn('getattr(self, "proof_profile", None) is not None', ast.get_source_segment(source, body[0].test))
        # Inside the P2 reducer only the terminal-reason classification of outcome() differs.
        def p2_methods(source):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "_P2State")
            return {n.name: ast.get_source_segment(source, n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        old_state, new_state = p2_methods(old_source), p2_methods(new_source)
        self.assertEqual({n for n in old_state if old_state[n] != new_state[n]}, {"outcome"})
        self.assertEqual(new_state["outcome"].replace(
            '("p2_retry_exhausted" if retryable else\n'
            '                                      "p2_transport_guard" if exception_type in _P2_TRANSPORT_GUARD_TYPES\n'
            '                                      else "p2_malformed")',
            '"p2_retry_exhausted" if retryable else "p2_malformed"'), old_state["outcome"])

    def test_science_p1_paths_are_unchanged(self):
        old_source, new_source = self.baseline("src/operations/science.py"), (ROOT / "src/operations/science.py").read_text()
        _t, old = self.definitions(old_source)
        _t, new = self.definitions(new_source)
        # G6.6.4-R3: one new P2-only constant; its use changes the P2 replay and its two callers.
        self.assertEqual(set(new) - set(old), {"=_P2_MAX_QUERY_ATTEMPTS"})
        self.assertEqual(set(old), set(new) - {"=_P2_MAX_QUERY_ATTEMPTS"})
        self.assertEqual({name for name in old if old[name] != new[name]},
                         {"validate_p2_query_result", "_validate_phase6_manifest_evidence", "_p2_replay_events"})
        self.assertEqual(new["=_P2_MAX_QUERY_ATTEMPTS"], "_P2_MAX_QUERY_ATTEMPTS = 4")
        self.assertEqual(new["_p2_replay_events"].replace("<= _P2_MAX_QUERY_ATTEMPTS:", "<= 2:"), old["_p2_replay_events"])
        self.assertEqual(new["validate_p2_query_result"].count("_P2_MAX_QUERY_ATTEMPTS"), 1)
        new["_validate_phase6_manifest_evidence"] = new["_validate_phase6_manifest_evidence"].replace(
            "expected_profile, _P2_MAX_QUERY_ATTEMPTS)", "expected_profile, max_query_attempts)")
        self.assertIn("    replay = _p2_replay_events(details.get(\"p2_events\"), request.mjd_min, request.mjd_max,\n"
                      "                               profile, _P2_MAX_QUERY_ATTEMPTS)", new["validate_p2_query_result"])
        block = re.search(r"\n    # G6\.6\.3A: a trusted guarded-transport P2 profile.*?\n        \}\n", new["_validate_phase6_manifest_evidence"], re.S)
        self.assertIsNotNone(block)
        self.assertIn("if (expected_profile is not None", block.group(0))
        self.assertEqual(new["_validate_phase6_manifest_evidence"].replace(block.group(0), "\n"),
                         old["_validate_phase6_manifest_evidence"])


DOC = ROOT / "docs" / "operations" / "V3_G663_TRANSPORT_CANARY_QUALIFICATION.md"


def runbook_script(marker):
    match = re.search(r"<<'EOF'\n(# " + re.escape(marker) + r" v\d+ .*?\n)EOF\n", DOC.read_text(), re.S)
    if match is None:
        raise AssertionError(f"runbook script {marker} is missing")
    return match.group(1)


def night_loci(night, *, extra=True):
    """The dense floor fixture and a few ordinary loci, moved to ``night``, with histories."""
    shift = L.night_mjd_interval(night)[0] - 61218.0
    loci = dense_loci()
    if extra:
        loci += [FakeLocus(f"E-{n}", mjd=61218.6, ra=30.0 + 40 * n, dec=-10.0 + 9 * n) for n in range(5)]
    for locus in loci:
        locus.locus_id = f"{night}-{locus.locus_id}"
        locus.properties["survey"] = {"lsst": {"dia_object_id": f"DIA-{locus.locus_id}"}}
        locus.properties["newest_alert_observation_time"] += shift
        mjd = locus.properties["newest_alert_observation_time"]
        locus.lightcurve = pd.DataFrame({"mjd": [mjd - .5, mjd], "ztf_magpsf": [20.5, 20.0],
                                         "ztf_sigmapsf": [.1, .1], "ztf_fid": [1, 2]})
    return loci


class CanaryRunbookDryRunTests(OfflineTestCase):
    """Sections 7.2/7.3 of the qualification document, executed verbatim and offline."""

    def setUp(self):
        super().setUp()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.parent = Path(folder.name).resolve() / "canary"
        self.parent.mkdir(mode=0o700)
        for module in (L, B.storage):
            patcher = mock.patch.object(module, "ARNOR_CANARY_ROOT", self.parent)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(socket, "gethostname", lambda: "arnor.astro.washington.edu")
        patcher.start()
        self.addCleanup(patcher.stop)

    def environment(self, night):
        return {"NIGHT": night, "RUN_ID": f"g664-p2-{night}-v2", "RELEASE_SHA": "6" * 40,
                "PROVIDER_SHA256": hashlib.sha256((ROOT / "src/operations/live_antares.py").read_bytes()).hexdigest(),
                "P2_TRANSPORT_SHA256": hashlib.sha256((ROOT / "src/operations/p2_transport.py").read_bytes()).hexdigest(),
                "PROFILE_SHA256": CANARY_PROFILE_SHA256, "TRANSPORT_SHA256": CANARY_TRANSPORT_SHA256}

    def run_script(self, marker, environment):
        import os
        output = io.StringIO()
        with mock.patch.dict(os.environ, environment), contextlib.redirect_stdout(output):
            try:
                exec(compile(runbook_script(marker), f"<runbook:{marker}>", "exec"), {"__name__": "__main__"})
            finally:
                self.output = output.getvalue()
        return self.output

    @staticmethod
    def documents(output):
        decoder, position, documents = json.JSONDecoder(), 0, []
        while position < len(output.strip()):
            document, position = decoder.raw_decode(output, position)
            documents.append(document)
            while position < len(output) and output[position].isspace():
                position += 1
        return documents

    def test_runbook_identities_are_the_committed_bytes(self):
        text = DOC.read_text()
        exported = dict(re.findall(
            r"^export (PROVIDER_SHA256|P2_TRANSPORT_SHA256|SCIENCE_SHA256|PROFILE_SHA256|TRANSPORT_SHA256)=([0-9a-f]{64})$",
            text, re.M))
        self.assertEqual(exported, {
            "PROVIDER_SHA256": hashlib.sha256((ROOT / "src/operations/live_antares.py").read_bytes()).hexdigest(),
            "P2_TRANSPORT_SHA256": hashlib.sha256((ROOT / "src/operations/p2_transport.py").read_bytes()).hexdigest(),
            "SCIENCE_SHA256": hashlib.sha256((ROOT / "src/operations/science.py").read_bytes()).hexdigest(),
            "PROFILE_SHA256": L.G663_CANARY_P2_PROFILE_SHA256, "TRANSPORT_SHA256": L.G663_CANARY_TRANSPORT_SHA256})
        listed = re.findall(r"^(src/operations/\S+\.py)\s+sha256 ([0-9a-f]{64})$", text, re.M)
        self.assertEqual(sorted(path for path, _digest in listed), [
            "src/operations/live_antares.py", "src/operations/p2_transport.py", "src/operations/science.py"])
        for path, digest in listed:
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        self.assertIn(f"child implementation (p2_transport.py) sha256 {T.implementation_sha256()}", text)
        self.assertIn(f"G663_CANARY_P2_PROFILE_SHA256 = {CANARY_PROFILE_SHA256}", text)
        self.assertIn(f"G663_CANARY_TRANSPORT_SHA256 = {CANARY_TRANSPORT_SHA256}", text)
        for night, digest in (("2026-07-07", "bacf32185b1eff128c705befaba4d34359a60012388a2708b392a18bac0a4e48"),
                              ("2026-07-13", "aee7ec99d1cc3fc54cf8bf3f270bfebff53a5cacb8e2104f3aca8aa16f1e1891")):
            self.assertIn(f"{night} scientific contract sha256 = {digest}", text)

    def test_transport_child_preflight_starts_and_stops_one_production_child(self):
        environment = self.environment("2026-07-07")
        self.assertIsNone(T.CHILD_TEST_HOOK)  # the production child: no test hook at all
        report = self.documents(self.run_script("g663-child-preflight", environment))[-1]
        self.assertEqual((report["stage"], report["child_implementation_sha256"]),
                         ("child-preflight", environment["P2_TRANSPORT_SHA256"]))
        self.assertEqual((report["exit"]["returncode"], report["exit"]["reaped"]), (0, True))
        self.assertLess(report["startup_seconds"], CANARY.transport.child_startup_seconds)
        self.assertEqual(transport_children(), [])
        with self.assertRaises(AssertionError):  # other bytes refuse before any child starts
            self.run_script("g663-child-preflight", {**environment, "P2_TRANSPORT_SHA256": "0" * 64})

    def test_jul07_acquisition_resume_and_network_free_verification(self):
        night = "2026-07-07"
        environment = self.environment(night)
        root = self.parent / environment["RUN_ID"]
        root.mkdir(mode=0o700)
        loci = night_loci(night)
        service = FakeAntares(loci, page_size=10)
        with service.installed():
            output = self.run_script("g663-canary-acquire", environment)
        query, acquired = self.documents(output)
        self.assertEqual((query["stage"], query["clean"], query["errors"]), ("query", True, []))
        self.assertEqual(query["client"], guarded_identity())
        self.assertEqual(acquired["terminal_evidence"], "natural-exhaustion-below-50")
        self.assertEqual(acquired["p2_budget"]["secondary_nodes"], 3)
        self.assertEqual((acquired["loci"], acquired["fetch"]["alert_rows"]), (len(loci), 2 * len(loci)))
        self.assertEqual(acquired["query_contract_sha256"], "bacf32185b1eff128c705befaba4d34359a60012388a2708b392a18bac0a4e48")
        self.assertOfficial(service.urls)
        # An identical re-run resumes: the sealed query and complete fetch are reused, no requests.
        service.calls.clear()
        with service.installed():
            resumed = self.documents(self.run_script("g663-canary-acquire", environment))
        self.assertEqual(service.calls, [])
        self.assertEqual(resumed[-1]["query_integrity_sha256"], acquired["query_integrity_sha256"])
        # Verification: no service at all, root unchanged, the adoption identity of the sealed evidence.
        refusing = FakeAntares(script=lambda *_: (_ for _ in ()).throw(AssertionError("HTTP during verification")))
        with refusing.installed(), mock.patch.object(socket, "if_nameindex", lambda: [(1, "lo")]):
            verified = self.documents(self.run_script("g663-canary-verify", environment))[-1]
        self.assertEqual(refusing.calls, [])
        self.assertEqual(verified["stage"], "verified")
        entry = verified["adoption_entry"]
        self.assertEqual((entry["loci"], entry["alert_rows"], entry["source_run_id"]),
                         (len(loci), 2 * len(loci), environment["RUN_ID"]))
        self.assertEqual(entry["query_integrity_sha256"], acquired["query_integrity_sha256"])
        self.assertEqual(entry["source_provider_implementation_sha256"], environment["PROVIDER_SHA256"])
        self.assertEqual(verified["selection_descriptor"], B.selection_descriptor_identity(
            B.selection_descriptor_for_request(R.LiveRangeAdapter(ROOT, RELEASE, None).acquisition_request(night))))
        self.assertEqual(verified["client"]["transport_sha256"], CANARY_TRANSPORT_SHA256)
        self.assertEqual((verified["loci"], verified["alert_rows"]), (len(loci), 2 * len(loci)))
        # Without the in-process verification profile the P2 canary is not trusted by anything.
        with self.assertRaises(B.BackfillRefused):
            B.describe_saved_acquisition(root, night, R.LiveRangeAdapter(ROOT, RELEASE, None), 256)

    def test_jul13_refusal_seals_nothing_replays_offline_and_cannot_verify(self):
        night = "2026-07-13"
        environment = self.environment(night)
        root = self.parent / environment["RUN_ID"]
        root.mkdir(mode=0o700)
        tile = L._make_initial_tiles(*L.night_mjd_interval(night))[0]
        first = tile_loci(tile, 15, prefix=f"{night}-F")
        tile_query = json.dumps(L._build_tile_query(tile))

        def script(service, request, index):
            if parse_qs(urlsplit(request.url).query).get(ES_KEY, [None])[0] == tile_query:
                return page(first[:10], next=f"ftp://{HOST}/v1/loci?p=2")
            return None
        service = FakeAntares(night_loci(night) + first, script=script)
        with service.installed(), self.assertRaises(QueryInterruptedError):
            self.run_script("g663-canary-acquire", environment)
        query = self.documents(self.output)[0]
        self.assertEqual((query["clean"], query["errors"], query["terminal_evidence"]),
                         (False, ["p2_transport_guard"], "p2_transport_guard"))
        self.assertEqual(query["completion_classification"], "INCOMPLETE")
        self.assertFalse((root / "checkpoints" / "query-result").exists())
        self.assertEqual(len(service.calls), 1)
        service.calls.clear()
        with service.installed(), self.assertRaises(QueryInterruptedError):
            self.run_script("g663-canary-acquire", environment)
        self.assertEqual(service.calls, [])  # the journal alone reproduces the refusal
        self.assertEqual(self.documents(self.output)[0]["errors"], ["p2_transport_guard"])
        with mock.patch.object(socket, "if_nameindex", lambda: [(1, "lo")]), self.assertRaises(B.BackfillRefused):
            self.run_script("g663-canary-verify", environment)
