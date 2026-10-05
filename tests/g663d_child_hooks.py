"""Test hooks loaded *inside* the G6.6.3D transport child (by path, tests only).

Standard library and ``requests`` only: the child never imports the ``src``
package.  ``configure`` installs a network guard, optionally routes the
official ANTARES origin to a loopback test service, and can make the child
block or misbehave so the parent's deadline, kill and IPC validation can be
proved against a real process.
"""

import json
import os
import signal
import socket
import sys
import time

OFFICIAL = ("https", "api.antares.noirlab.edu")


def _record(path, message):
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def _network_guard(violations):
    """Refuse every non-loopback connection or name lookup made by this child."""
    loopback = {"127.0.0.1", "localhost", "::1"}

    def audit(event, args):
        if event == "socket.connect":
            address = args[1]
            if isinstance(address, tuple) and address[0] not in loopback:
                _record(violations, f"connect {address!r}")
                raise PermissionError("network access refused in the transport child")
        elif event == "socket.getaddrinfo" and args[0] not in loopback:
            _record(violations, f"getaddrinfo {args[0]!r}")
            raise PermissionError("name lookup refused in the transport child")
    sys.addaudithook(audit)


def _blocking_call():
    """A C-level blocking read nothing will ever satisfy (a resolver-like stall)."""
    read_end, _write_end = os.pipe()
    _blocking_call.keep = _write_end
    os.read(read_end, 1)


def _loopback_session_factory(port):
    import requests
    from requests import exceptions
    from requests.adapters import HTTPAdapter
    from urllib.parse import urlsplit, urlunsplit

    class Loopback(HTTPAdapter):
        def send(self, request, **kwargs):
            original = request.url
            parts = urlsplit(original)
            if (parts.scheme.lower(), (parts.hostname or "").lower()) != OFFICIAL:
                raise AssertionError(f"transport child left the official origin: {original!r}")
            request.url = urlunsplit(("http", f"127.0.0.1:{port}", parts.path, parts.query, ""))
            request.headers["X-G663-Original-URL"] = original
            timeout = kwargs.get("timeout")
            request.headers["X-G663-Send"] = json.dumps(
                {"timeout": list(timeout) if isinstance(timeout, tuple) else timeout,
                 "stream": kwargs.get("stream")})
            try:
                response = super().send(request, **{**kwargs, "proxies": {}})
            finally:
                request.url = original
                del request.headers["X-G663-Original-URL"]
                del request.headers["X-G663-Send"]
            response.url = original
            raised = response.headers.get("X-G663-Raise")
            if raised:
                response.close()
                module, _, name = raised.rpartition(".")
                error = getattr(exceptions, name) if module == "requests.exceptions" else None
                if error is None:
                    raise AssertionError(f"unsupported raised type {raised!r}")
                raise error("raised by the offline test service")
            return response

    class Refuse(HTTPAdapter):
        def send(self, request, **kwargs):
            raise AssertionError(f"transport child attempted {request.url!r}")

    def factory():
        session = requests.Session()
        session.mount("https://", Loopback())
        session.mount("http://", Refuse())
        return session
    return factory


def configure(namespace, args):
    """Entry point named by the parent's test hook."""
    _network_guard(args.get("violations"))
    if args.get("ignore_sigterm"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if args.get("block_resolver"):
        def blocked_resolver(*_args, **_kwargs):
            _record(args.get("marker"), "resolver")  # entered: no socket exists yet
            _blocking_call()
        socket.getaddrinfo = blocked_resolver
    if args.get("startup") == "hang":
        _blocking_call()
    if args.get("startup") == "import_src":
        sys.modules["src"] = type(sys)("src")  # what importing the ANTARES package would show
    if args.get("startup") == "exit":
        os._exit(17)
    misbehaviour = args.get("misbehave")
    if misbehaviour:
        _misbehave(namespace, misbehaviour, args)
    if args.get("port") is not None:
        return _loopback_session_factory(args["port"])
    return None


def _misbehave(namespace, mode, args):
    """Replace the child's request handler with one IPC violation (from request ``after``)."""
    serve, send, frame = namespace["_child_serve"], namespace["_child_send"], namespace["_frame"]
    honest = int(args.get("after", 0))
    state = {"served": 0}

    def raw(descriptor, data):
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view):]

    def misbehaving(request, session_factory, config, result):
        state["served"] += 1
        if state["served"] <= honest:
            return serve(request, session_factory, config, result)
        rid = request["id"]
        if mode == "crash":
            os._exit(9)
        if mode == "segfault":
            os.kill(os.getpid(), signal.SIGSEGV)
        if mode == "malformed":
            raw(result, b"this is not a frame at all")
        elif mode == "bad_json":
            raw(result, b"G6D1" + (9).to_bytes(4, "big") + (0).to_bytes(4, "big") + b"{not json")
        elif mode == "eof":
            os.close(result)
        elif mode == "oversized_header":
            raw(result, b"G6D1" + (config["ipc_max_header_bytes"] + 1).to_bytes(4, "big") + (0).to_bytes(4, "big"))
        elif mode == "oversized_chunk":
            raw(result, b"G6D1" + (20).to_bytes(4, "big") + (config["ipc_max_chunk_bytes"] + 1).to_bytes(4, "big"))
        elif mode == "over_budget":
            for seq in range(2):
                send(result, {"kind": "RESULT", "id": rid, "seq": seq}, b" " * request["budget"])
        elif mode == "foreign_id":
            send(result, {"kind": "ERROR", "id": rid + 1, "code": "http_status", "type": None, "status": 503})
        elif mode == "seq_gap":
            send(result, {"kind": "RESULT", "id": rid, "seq": 1}, b"{}")
        elif mode == "complete_mismatch":
            send(result, {"kind": "RESULT", "id": rid, "seq": 0}, b"{}")
            send(result, {"kind": "COMPLETE", "id": rid, "chunks": 1, "length": 2, "sha256": "0" * 64,
                          "encoding": None})
        elif mode == "error_after_result":
            send(result, {"kind": "RESULT", "id": rid, "seq": 0}, b"{}")
            send(result, {"kind": "ERROR", "id": rid, "code": "http_status", "type": None, "status": 503})
        elif mode == "unknown_kind":
            send(result, {"kind": "BOGUS", "id": rid})
        elif mode == "extra_key":
            send(result, {"kind": "ERROR", "id": rid, "code": "http_status", "type": None, "status": 503,
                          "detail": "x"})
        elif mode == "unknown_code":
            send(result, {"kind": "ERROR", "id": rid, "code": "teapot", "type": None, "status": None})
        elif mode == "unrecognised_exception":  # a well-formed ERROR; the child stays healthy
            send(result, {"kind": "ERROR", "id": rid, "code": "exception", "type": "builtins.OSError",
                          "status": None})
            return
        elif mode == "partial_then_exit":
            send(result, {"kind": "RESULT", "id": rid, "seq": 0}, b"{}")
            os._exit(0)
        elif mode == "stall_mid_frame":
            data = frame({"kind": "RESULT", "id": rid, "seq": 0}, b"{" * 100)
            raw(result, data[:len(data) // 2])
        elif mode == "slow_chunks":
            for seq in range(10_000):
                send(result, {"kind": "RESULT", "id": rid, "seq": seq}, b" ")
                time.sleep(0.05)
        _blocking_call()
    namespace["_child_serve"] = misbehaving
