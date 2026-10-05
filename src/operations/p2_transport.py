"""G6.6.3D process-isolated P2 transport boundary.

The guarded P2 paginator (``live_antares._GuardedListing``) stays in the
scientific process and owns pagination, continuation validation, decoding,
deadlines and every completeness decision.  Only the blocking part of one HTTP
page exchange runs here, in one dedicated child process per P2 night query:
name resolution, connect, TLS, redirects inside the page and the bounded body
read.  A thread cannot be stopped while it is blocked in a resolver or in
urllib3's chunked-trailer loop; a process can always be killed, so the
parent's deadline is a hard bound.

The parent starts the child lazily on the first page, reuses it for every
page of every search of that night, and kills it (SIGTERM, a short grace,
SIGKILL, a bounded join) at a deadline or on any IPC violation.  Only a reaped
exit status counts as proof that the child is dead.  A failed or closed
transport is never restarted.

The child runs only this file, loaded by path, never the ``src`` package:
standard library and ``requests`` only, so it cannot reach science,
checkpoints, journals, storage, transactions or publication.  It returns the
exact body bytes and the charset ``requests`` derived for them, or a typed
failure.  The IPC frames carry JSON headers of primitive values plus raw
bytes; nothing is pickled.  The SHA-256 of this file is part of the transport
identity, and the child proves at startup that it executed exactly those
bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import signal
import struct
import subprocess
import sys
import threading
import time
import weakref
from urllib.parse import urljoin, urlsplit

PROTOCOL = "g663d.p2-transport-ipc.v1"
STREAM_CHUNK_BYTES = 8192
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
REFUSAL_CODES = frozenset({"page_bytes", "iterator_bytes", "unsafe_redirect", "redirect_limit",
                           "unexpected_status", "http_status"})
ERROR_CODES = REFUSAL_CODES | {"exception", "header_bound"}
CONFIG_KEYS = frozenset({"host", "service_prefix", "max_redirects", "connect_timeout_seconds",
                         "read_timeout_seconds", "ipc_max_header_bytes", "ipc_max_chunk_bytes",
                         "child_startup_seconds", "child_shutdown_seconds", "terminate_grace_seconds",
                         "kill_join_seconds"})
NOT_STARTED, RUNNING, FAILED, CLOSED = "NOT_STARTED", "RUNNING", "FAILED", "CLOSED"

# Test-only seam: a {"path", "name", "args"} hook the child loads before READY.
# None in every production path; nothing in the provider ever sets it.
CHILD_TEST_HOOK = None

_MAGIC = b"G6D1"
_PREFIX = struct.Struct(">4sII")
_INIT_HEADER_BYTES = 65_536
_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXIT_PROTOCOL, _EXIT_STARTUP, _EXIT_ORPHANED = 3, 4, 5
_implementation = []

# Executed with ``python -c``: no working-directory import path, then exactly
# this file's bytes as a private module that is not part of the src package.
_BOOTSTRAP = (
    "import sys\n"
    "sys.path[:] = [entry for entry in sys.path if entry]\n"
    "import hashlib, types\n"
    "path = sys.argv[1]\n"
    "with open(path, 'rb') as handle:\n"
    "    source = handle.read()\n"
    "module = types.ModuleType('_antares_p2_transport_child')\n"
    "module.__file__ = path\n"
    "sys.modules[module.__name__] = module\n"
    "exec(compile(source, path, 'exec'), module.__dict__)\n"
    "module._child_main(int(sys.argv[2]), int(sys.argv[3]), hashlib.sha256(source).hexdigest())\n"
)


def implementation_sha256():
    """SHA-256 of this file: what every child must prove it executed."""
    if not _implementation:
        with open(os.path.abspath(__file__), "rb") as handle:
            _implementation.append(hashlib.sha256(handle.read()).hexdigest())
    return _implementation[0]


def plain_url(value):
    """One printable-ASCII URL: no whitespace, control character or backslash.

    URL parsers disagree on these (``urlsplit`` drops tabs and keeps ``\\`` in
    the authority; urllib3 splits on ``\\``), so they are refused before any
    origin decision rather than interpreted.
    """
    return (isinstance(value, str) and bool(value) and value.isascii() and "\\" not in value
            and not any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value))


class Refusal(Exception):
    """A transport rule refused one page; the parent maps ``code`` to its typed error."""

    def __init__(self, code, status=None):
        super().__init__(code)
        self.code, self.status = code, status


class TransportFailure(Exception):
    """Parent-side failure of one page exchange; carries no body, header or URL.

    ``kind`` is ``remote`` (the healthy child reported ``code``), or
    ``deadline``, ``startup``, ``protocol``, ``unkillable``, ``unavailable``
    or ``request_bound`` (the parent refused or ended the exchange itself).
    """

    def __init__(self, kind, code=None, remote_type=None, status=None, exit_record=None):
        super().__init__(kind if code is None else f"{kind}:{code}")
        self.kind, self.code, self.remote_type, self.status = kind, code, remote_type, status
        self.exit_record = exit_record


def classify_failure(exc):
    """The typed, text-free description of one failed page exchange."""
    if isinstance(exc, Refusal) and exc.code in REFUSAL_CODES:
        return {"code": exc.code, "type": None, "status": exc.status}
    name = f"{type(exc).__module__}.{type(exc).__name__}"
    return {"code": "exception", "type": name if _TYPE_NAME.match(name) else None, "status": None}


def _send(session, url, params, config):
    """One HTTP exchange through the session's own transport adapter.

    ``Session.send`` pre-computes ``Response.next`` even with
    ``allow_redirects=False``, which reads a redirect response's whole body
    without any ceiling.  The request and environment settings here are
    exactly those ``Session.request`` derives for ``requests.get``; only that
    redirect bookkeeping is skipped, so every body this page reads passes
    through the bounded reader in :func:`fetch_page`.
    """
    import requests

    prepared = session.prepare_request(requests.Request("GET", url, params=params or {}))
    settings = session.merge_environment_settings(prepared.url, {}, True, None, None)
    return session.get_adapter(prepared.url).send(
        prepared, stream=True, verify=settings["verify"], cert=settings["cert"],
        proxies=settings["proxies"],
        timeout=(config["connect_timeout_seconds"], config["read_timeout_seconds"]))


def _validated_redirect(response, requested, config):
    location = response.headers.get("Location")
    if not plain_url(location):
        raise Refusal("unsafe_redirect")
    base = response.url if isinstance(response.url, str) and response.url else requested
    target = urljoin(base, location)
    try:
        parts = urlsplit(target)
        port, host = parts.port, parts.hostname
    except ValueError as exc:
        raise Refusal("unsafe_redirect") from exc
    if (parts.scheme != "https" or host != config["host"] or port not in (None, 443)
            or parts.username is not None or parts.password is not None or parts.fragment
            or not parts.path.startswith(config["service_prefix"])):
        raise Refusal("unsafe_redirect")
    return target


def fetch_page(session_factory, config, url, params, budget, overflow):
    """One bounded HTTP page: ``(body bytes, requests' charset)`` or a refusal.

    Each page uses a fresh session, as ``requests.get`` does.  Redirects are
    validated before they are requested and their bodies are never read;
    error statuses are classified without reading the body; decoded bytes
    are counted, so compression cannot bypass ``budget``.
    """
    session = session_factory()
    try:
        for _hop in range(config["max_redirects"] + 1):
            response = _send(session, url, params, config)
            try:
                status = int(response.status_code)
                if status in REDIRECT_STATUSES:
                    url, params = _validated_redirect(response, url, config), None
                    continue
                if status >= 400:
                    raise Refusal("http_status", status)
                if status != 200:
                    raise Refusal("unexpected_status", status)
                declared = response.headers.get("Content-Length")
                if (isinstance(declared, str) and declared.isascii() and declared.isdigit()
                        and int(declared) > budget):
                    raise Refusal(overflow)
                body = bytearray()
                for chunk in response.iter_content(STREAM_CHUNK_BYTES):
                    body += chunk
                    if len(body) > budget:
                        raise Refusal(overflow)
                return bytes(body), response.encoding
            finally:
                response.close()
        raise Refusal("redirect_limit")
    finally:
        session.close()


def _encode_header(header):
    return json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _frame(header, payload=b""):
    raw = _encode_header(header)
    return _PREFIX.pack(_MAGIC, len(raw), len(payload)) + raw + payload


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _decode_header(raw):
    header = json.loads(raw.decode("ascii"), parse_constant=_reject_constant)
    if not isinstance(header, dict) or not isinstance(header.get("kind"), str):
        raise ValueError("IPC header is not a typed object")
    return header


def _count(value, low, high):
    return type(value) is int and low <= value <= high


def _valid_config(config):
    return (isinstance(config, dict) and set(config) == CONFIG_KEYS
            and isinstance(config["host"], str) and bool(config["host"])
            and isinstance(config["service_prefix"], str) and config["service_prefix"].startswith("/")
            and all(_count(config[key], 0 if key == "max_redirects" else 1, 1 << 30)
                    for key in CONFIG_KEYS - {"host", "service_prefix"}))


class _Abort(Exception):
    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


def _stop_process(process, grace, join):
    """SIGTERM, ``grace`` seconds, SIGKILL, ``join`` seconds; only a reaped status is proof."""
    started, escalated = time.monotonic(), False
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(grace)
        except subprocess.TimeoutExpired:
            escalated = True
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(join)
            except subprocess.TimeoutExpired:
                pass
    return {"pid": process.pid, "returncode": process.returncode,
            "reaped": process.returncode is not None, "escalated_to_sigkill": escalated,
            "seconds": round(time.monotonic() - started, 6)}


def _close_descriptors(handle):
    for key in ("command", "result"):
        descriptor = handle.pop(key, None)
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _release(handle):
    """Finalizer (garbage collection or interpreter exit): never leave a child behind."""
    process = handle.get("process")
    if process is not None and process.poll() is None:
        try:
            process.kill()
            process.wait(5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    _close_descriptors(handle)


class P2ProcessTransport:
    """Parent-owned lifecycle of one P2 transport child (one per night query).

    States: ``NOT_STARTED`` -> ``RUNNING`` -> ``FAILED`` or ``CLOSED``.  The
    child starts on the first page.  A typed failure the healthy child reports
    leaves it ``RUNNING`` (the qualified per-node retries may reuse it).  A
    deadline, IPC violation, early EOF or child death kills it and leaves the
    transport ``FAILED`` for good: there is no restart.
    """

    def __init__(self, config, *, hook=None):
        if not _valid_config(config):
            raise ValueError("P2 transport configuration is invalid.")
        self._config = dict(config)
        self._hook = CHILD_TEST_HOOK if hook is None else hook
        self._lock = threading.Lock()
        self._handle = {}
        self._finalizer = weakref.finalize(self, _release, self._handle)
        self._buffer = bytearray()
        self._next_id = 0
        self._ready = False
        self._readable = self._writable = None
        self.state = NOT_STARTED
        self.exit_record = None
        self.stats = {"spawns": 0, "pid": None, "startup_seconds": None, "requests": 0}

    # -- public -------------------------------------------------------------
    def fetch(self, url, params, budget, overflow, remaining):
        """One page within ``remaining`` seconds: ``(body, encoding)`` or ``TransportFailure``."""
        if not self._lock.acquire(blocking=False):
            raise TransportFailure("unavailable", "concurrent_use")
        try:
            return self._fetch(url, params, budget, overflow, remaining)
        finally:
            self._lock.release()

    def close(self):
        """Cooperative shutdown, escalating to terminate/kill; returns the exit record."""
        with self._lock:
            if self.state == RUNNING:
                deadline = time.monotonic() + self._config["child_shutdown_seconds"]
                try:
                    self._write(_frame({"kind": "SHUTDOWN"}), deadline, "shutdown")
                except _Abort:
                    pass
                process = self._handle["process"]
                try:
                    process.wait(max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
                self._finish(CLOSED)
                if not self.exit_record["reaped"]:
                    self.state = FAILED
                    raise TransportFailure("unkillable", exit_record=self.exit_record)
            elif self.state == NOT_STARTED:
                self.state = CLOSED
            return self.exit_record

    # -- lifecycle ----------------------------------------------------------
    def _fetch(self, url, params, budget, overflow, remaining):
        if self.state not in (NOT_STARTED, RUNNING):
            raise TransportFailure("unavailable", self.state)
        deadline = time.monotonic() + max(0.0, float(remaining))
        request_id = self._next_id + 1
        pairs = None if params is None else [[str(k), str(v)] for k, v in dict(params).items()]
        header = {"kind": "REQUEST", "id": request_id, "url": url, "params": pairs,
                  "budget": budget, "overflow": overflow}
        if (not isinstance(url, str) or overflow not in ("page_bytes", "iterator_bytes")
                or not _count(budget, 1, 1 << 32)
                or len(_encode_header(header)) > self._config["ipc_max_header_bytes"]):
            raise TransportFailure("request_bound")
        in_flight = False
        try:
            if self.state == NOT_STARTED:
                self._start(deadline)
            self._next_id = request_id
            self.stats["requests"] += 1
            in_flight = True
            self._write(_frame(header), deadline, "deadline")
            body, chunks = bytearray(), 0
            while True:
                reply, payload = self._read(deadline, "deadline")
                kind = reply["kind"]
                if not _count(reply.get("id"), request_id, request_id):
                    raise _Abort("protocol")  # a frame for another request
                if kind == "RESULT" and set(reply) == {"kind", "id", "seq"}:
                    if (not _count(reply["seq"], chunks, chunks) or not payload
                            or len(body) + len(payload) > budget):
                        raise _Abort("protocol")
                    body += payload
                    chunks += 1
                elif kind == "COMPLETE" and set(reply) == {"kind", "id", "chunks", "length", "sha256",
                                                          "encoding"}:
                    if (payload or not _count(reply["chunks"], chunks, chunks)
                            or not _count(reply["length"], len(body), len(body))
                            or reply["sha256"] != hashlib.sha256(body).hexdigest()
                            or not (reply["encoding"] is None or isinstance(reply["encoding"], str))):
                        raise _Abort("protocol")
                    in_flight = False
                    return bytes(body), reply["encoding"]
                elif kind == "ERROR" and set(reply) == {"kind", "id", "code", "type", "status"}:
                    if (payload or chunks or reply["code"] not in ERROR_CODES
                            or not (reply["type"] is None or (isinstance(reply["type"], str)
                                                              and _TYPE_NAME.match(reply["type"])))
                            or not (reply["status"] is None or _count(reply["status"], 100, 999))):
                        raise _Abort("protocol")
                    in_flight = False
                    raise TransportFailure("remote", reply["code"], reply["type"], reply["status"])
                else:
                    raise _Abort("protocol")  # unexpected kind, shape or ordering
        except _Abort as abort:
            self._finish(FAILED)
            kind = "unkillable" if not self.exit_record["reaped"] else abort.kind
            raise TransportFailure(kind, exit_record=self.exit_record) from None
        except TransportFailure:
            raise
        except BaseException:
            # Interrupted mid-exchange or mid-startup (for example KeyboardInterrupt):
            # the child's state is unknown, so it is killed and never reused.
            if self.state == RUNNING and (in_flight or not self._ready):
                self._finish(FAILED)
            raise

    def _start(self, deadline):
        startup_deadline = time.monotonic() + self._config["child_startup_seconds"]
        timeout_kind = "deadline" if deadline <= startup_deadline else "startup"
        deadline = min(deadline, startup_deadline)
        started = time.monotonic()
        self.state = RUNNING  # from here on, every failure kills and reaps what exists
        descriptors = []
        try:
            for _pipe in range(2):
                descriptors.extend(os.pipe())
        except OSError:
            for descriptor in descriptors:
                os.close(descriptor)
            raise _Abort("startup") from None
        command_read, command_write, result_read, result_write = descriptors
        self._handle.update(command=command_write, result=result_read)
        try:
            flags = getattr(subprocess, "_args_from_interpreter_flags", lambda: [])()
            self._handle["process"] = subprocess.Popen(
                [sys.executable, *flags, "-c", _BOOTSTRAP, os.path.abspath(__file__),
                 str(command_read), str(result_write)],
                pass_fds=(command_read, result_write), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
                start_new_session=True)
        except (OSError, ValueError, subprocess.SubprocessError):
            raise _Abort("startup") from None
        finally:
            os.close(command_read)
            os.close(result_write)
        process = self._handle["process"]
        self.stats.update(spawns=self.stats["spawns"] + 1, pid=process.pid)
        try:
            os.set_blocking(command_write, False)
            os.set_blocking(result_read, False)
            self._readable = selectors.DefaultSelector()
            self._readable.register(result_read, selectors.EVENT_READ)
            self._writable = selectors.DefaultSelector()
            self._writable.register(command_write, selectors.EVENT_WRITE)
        except (OSError, ValueError):
            raise _Abort("startup") from None
        init = {"kind": "INIT", "protocol": PROTOCOL, "config": self._config, "hook": self._hook}
        try:
            self._write(_frame(init), deadline, timeout_kind)
            ready, payload = self._read(deadline, timeout_kind)
        except _Abort as abort:
            raise _Abort(abort.kind if abort.kind == "deadline" else "startup") from None
        if (payload or set(ready) != {"kind", "protocol", "pid", "implementation_sha256",
                                       "antares_modules"}
                or ready["kind"] != "READY" or ready["protocol"] != PROTOCOL
                or not _count(ready["pid"], process.pid, process.pid)
                or ready["implementation_sha256"] != implementation_sha256()
                or ready["antares_modules"] != []):
            raise _Abort("startup")
        self._ready = True
        self.stats["startup_seconds"] = round(time.monotonic() - started, 6)

    def _finish(self, state):
        process = self._handle.get("process")
        if process is not None:
            self.exit_record = _stop_process(process, self._config["terminate_grace_seconds"],
                                             self._config["kill_join_seconds"])
        else:
            self.exit_record = {"pid": None, "returncode": None, "reaped": True,
                                "escalated_to_sigkill": False, "seconds": 0.0}
        _close_descriptors(self._handle)
        for selector in (self._readable, self._writable):
            if selector is not None:
                selector.close()
        self._readable = self._writable = None
        self.state = state

    # -- framing ------------------------------------------------------------
    def _write(self, data, deadline, timeout_kind):
        view = memoryview(data)
        while view:
            try:
                view = view[os.write(self._handle["command"], view):]
                continue
            except BlockingIOError:
                pass
            except OSError:
                raise _Abort("protocol") from None
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._writable.select(remaining):
                raise _Abort(timeout_kind)

    def _fill(self, size, deadline, timeout_kind):
        while len(self._buffer) < size:
            try:
                chunk = os.read(self._handle["result"], size - len(self._buffer))
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._readable.select(remaining):
                    raise _Abort(timeout_kind) from None
                continue
            except OSError:
                raise _Abort("protocol") from None
            if not chunk:  # EOF before the terminal frame: the child closed or died
                raise _Abort("protocol")
            self._buffer += chunk

    def _read(self, deadline, timeout_kind):
        if time.monotonic() >= deadline:
            raise _Abort(timeout_kind)
        self._fill(_PREFIX.size, deadline, timeout_kind)
        magic, header_size, payload_size = _PREFIX.unpack(bytes(self._buffer[:_PREFIX.size]))
        if (magic != _MAGIC or not 2 <= header_size <= self._config["ipc_max_header_bytes"]
                or payload_size > self._config["ipc_max_chunk_bytes"]):
            raise _Abort("protocol")
        end = _PREFIX.size + header_size + payload_size
        self._fill(end, deadline, timeout_kind)
        raw = bytes(self._buffer[_PREFIX.size:_PREFIX.size + header_size])
        payload = bytes(self._buffer[_PREFIX.size + header_size:end])
        del self._buffer[:end]
        try:
            return _decode_header(raw), payload
        except (UnicodeDecodeError, ValueError):
            raise _Abort("protocol") from None


# ---------------------------------------------------------------------------
# Child process.  Everything below runs only in the child.
# ---------------------------------------------------------------------------

def _child_read_exact(descriptor, size):
    data = bytearray()
    while len(data) < size:
        chunk = os.read(descriptor, size - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return bytes(data)


def _child_read(descriptor, header_limit):
    magic, header_size, payload_size = _PREFIX.unpack(_child_read_exact(descriptor, _PREFIX.size))
    if magic != _MAGIC or not 2 <= header_size <= header_limit or payload_size:
        raise ValueError("IPC frame is invalid")
    return _decode_header(_child_read_exact(descriptor, header_size))


def _child_send(descriptor, header, payload=b""):
    view = memoryview(_frame(header, payload))
    while view:
        view = view[os.write(descriptor, view):]


def _child_hook(hook, namespace):
    """Load a test hook (tests only); it may return a session factory."""
    import importlib.util

    if (not isinstance(hook, dict) or set(hook) != {"path", "name", "args"}
            or not isinstance(hook["path"], str) or not isinstance(hook["name"], str)):
        raise ValueError("test hook is invalid")
    spec = importlib.util.spec_from_file_location("_antares_p2_transport_test_hook", hook["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, hook["name"])(namespace, hook["args"])


def _child_serve(request, session_factory, config, result):
    request_id = request["id"]
    try:
        params = None if request["params"] is None else dict(request["params"])
        body, encoding = fetch_page(session_factory, config, request["url"], params,
                                    request["budget"], request["overflow"])
    except Exception as exc:  # every failure becomes one typed, text-free ERROR frame
        _child_send(result, {"kind": "ERROR", "id": request_id, **classify_failure(exc)})
        return
    size = config["ipc_max_chunk_bytes"]
    chunks = [body[offset:offset + size] for offset in range(0, len(body), size)]
    complete = {"kind": "COMPLETE", "id": request_id, "chunks": len(chunks), "length": len(body),
                "sha256": hashlib.sha256(body).hexdigest(), "encoding": encoding}
    if len(_encode_header(complete)) > config["ipc_max_header_bytes"]:
        _child_send(result, {"kind": "ERROR", "id": request_id, "code": "header_bound",
                             "type": None, "status": None})
        return
    for sequence, chunk in enumerate(chunks):
        _child_send(result, {"kind": "RESULT", "id": request_id, "seq": sequence}, chunk)
    _child_send(result, complete)


def _valid_request(request, expected_id):
    pairs = request.get("params")
    return (set(request) == {"kind", "id", "url", "params", "budget", "overflow"}
            and request["id"] == expected_id and isinstance(request["url"], str)
            and _count(request["budget"], 1, 1 << 32)
            and request["overflow"] in ("page_bytes", "iterator_bytes")
            and (pairs is None or (isinstance(pairs, list) and all(
                isinstance(pair, list) and len(pair) == 2
                and all(isinstance(item, str) for item in pair) for pair in pairs))))


def _child_main(command, result, executed_sha256):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # only the parent decides this lifetime
    parent = os.getppid()

    def orphan_watchdog():
        while True:
            time.sleep(0.25)
            if os.getppid() != parent:
                os._exit(_EXIT_ORPHANED)
    threading.Thread(target=orphan_watchdog, name="p2-transport-orphan-watchdog", daemon=True).start()
    try:
        init = _child_read(command, _INIT_HEADER_BYTES)
        if (set(init) != {"kind", "protocol", "config", "hook"} or init["kind"] != "INIT"
                or init["protocol"] != PROTOCOL or not _valid_config(init["config"])):
            os._exit(_EXIT_PROTOCOL)
        config = init["config"]
        import requests

        session_factory = requests.Session
        if init["hook"] is not None:
            session_factory = _child_hook(init["hook"], globals()) or session_factory
    except BaseException:
        os._exit(_EXIT_STARTUP)
    try:
        _child_send(result, {"kind": "READY", "protocol": PROTOCOL, "pid": os.getpid(),
                             "implementation_sha256": executed_sha256,
                             "antares_modules": sorted(name for name in sys.modules
                                                       if name == "src" or name.startswith("src."))})
        expected_id = 1
        while True:
            try:
                request = _child_read(command, config["ipc_max_header_bytes"])
            except EOFError:
                os._exit(0)
            if request["kind"] == "SHUTDOWN" and set(request) == {"kind"}:
                os._exit(0)
            if request["kind"] != "REQUEST" or not _valid_request(request, expected_id):
                os._exit(_EXIT_PROTOCOL)
            _child_serve(request, session_factory, config, result)
            expected_id += 1
    except BaseException:
        os._exit(_EXIT_PROTOCOL)
