"""Subprocess entry point that dies by SIGKILL at one named publication boundary.

``SIGKILL`` bypasses ``except``/``finally``/``atexit``, so the parent test sees
exactly the durable state a hard process death leaves behind.  Exit code 3
means the boundary was never reached.
"""

import json
import os
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import v3_fixtures as F  # noqa: E402
from src.operations.publication import (  # noqa: E402
    NightPublisher,
    load_authorization,
    load_backfill_candidate,
)
from src.operations.storage import SyntheticWriteCapability  # noqa: E402


def killer(boundary, occurrence=1, date_utc=None):
    seen = {"count": 0}

    def hook(point, details):
        if point != boundary:
            return
        if date_utc is not None and _hook_date(details) not in (None, date_utc):
            return
        seen["count"] += 1
        if seen["count"] == occurrence:
            os.kill(os.getpid(), signal.SIGKILL)

    return hook


def _hook_date(details):
    transaction = str(details.get("transaction_id") or "")
    return transaction[6:16] if transaction.startswith("v3pub-") else None


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    capability = SyntheticWriteCapability.for_local_run_root(
        Path(spec["run_root"]), spec["run_id"]
    )
    hook = killer(spec["boundary"], spec.get("occurrence", 1), spec.get("date_utc"))
    if spec["mode"] == "publish":
        publisher = F.publisher_for(capability, fault_hook=hook)
        publisher.publish(
            load_backfill_candidate(Path(spec["night_root"])),
            load_authorization(Path(spec["authorization"])),
        )
        return 3
    if spec["mode"] == "hold-lock":
        from src.operations.publication import PublicationAuthorityLock

        with PublicationAuthorityLock(capability):
            Path(spec["ready"]).write_text("held\n")
            signal.pause()
        return 3
    if spec["mode"] == "hold-shared-lock":
        from src import history

        with history.authority_read_lock(capability.published_root):
            Path(spec["ready"]).write_text("held\n")
            signal.pause()
        return 3
    from src.operations.backfill import (
        BackfillController,
        BackfillSettings,
        RangePublicationAuthorization,
    )

    publisher = NightPublisher(
        capability,
        publisher_release_sha=F.PUBLISHER_RELEASE,
        cache_root=capability.root / "absent-cache",
        mountinfo_lines=F.mountinfo_for(capability.published_root),
        fault_hook=hook,
    )
    authority = RangePublicationAuthorization(
        **json.loads(Path(spec["range_authorization"]).read_text())
    )
    BackfillController(
        capability,
        F.SyntheticBackfillAdapter(spec["loci"]),
        release_sha=F.CANDIDATE_RELEASE,
        read_capability_factory=F.mock_read_capability,
        settings=BackfillSettings(segment_size=4),
        publisher=publisher,
        range_authorization=authority,
        prior_free_attestations=F.fixture_prior_free_attestations(),
    ).run(spec["start"], spec["end"], resume=spec.get("resume", False))
    return 3


if __name__ == "__main__":
    sys.exit(main())
