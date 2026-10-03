"""G6.4B: Control-bound adoption of saved acquisitions into range publication.

Saved acquisitions are produced here by the real acquisition code in the two
shapes that exist on Arnor: a canary run root (G6.3 v0.4.6 style) and another
range's night root (v0.4.5 week-test style), each from an older release. An
adoption range must publish them in order through the unchanged range
publication machinery with no ANTARES call and no write to the source, and
must fail closed on any evidence, identity or predecessor difference.
"""
import contextlib
import hashlib
import io
import json
import os
import socket
import unittest
import uuid
from pathlib import Path
from unittest import mock

import v3_fixtures as F
import test_production_range_g5 as G5
from src.operations import backfill as B, publication as P, production_range as R
from src.operations import storage as S
from src.operations.fetch_checkpoint import SegmentedFetchCheckpoint
from src.operations.query_checkpoint import load_query_result_checkpoint, seal_query_result_checkpoint
from src.operations.storage import RangeWorkCapability

OLD_RELEASE = "7" * 40      # v0.4.5-style range acquisition
CANARY_RELEASE = "6" * 40   # v0.4.6-style canary acquisition
DAYS = ("2026-06-28", "2026-06-29", "2026-06-30")
# Configurations of the real saved acquisitions, re-proved on Arnor in G6.4B.
REAL_SOURCE_CONFIGURATIONS = {
    "7211b5c2bce60edd24ca92309d33f0e81b412821": "8e5111a55e4999aca869c04ffb818d1ded1fd1a736f325f62fa6f2cad0abcba7",
    "b808f285bf7375c50c85cb170035f957b038ddb8": "0a56ba386b39eb206d4b9f9548904ae3e68eaa37f36fe271330f7129bce96954",
}


def tree_digest(root):
    entries = []
    for directory, names, files in os.walk(root):
        names.sort()
        for name in sorted(names + files):
            path = Path(directory) / name
            observed = os.lstat(path)
            entries.append([str(path.relative_to(root)), observed.st_mode,
                            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None])
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def flip_last_byte(path):
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    os.chmod(path, 0o600)
    path.write_bytes(bytes(payload))


class SavedAcquisitionAdoptionTests(unittest.TestCase):
    controller = G5.ProductionAuthorityTests.controller

    def setUp(self):
        G5.ProductionAuthorityTests.setUp(self)
        self.canary = self.parent / "canary"
        self.canary.mkdir(mode=0o700)
        self.stack.enter_context(mock.patch.multiple(
            S, RANGE_WORK_PARENT=self.work_root.parent, ARNOR_CANARY_ROOT=self.canary))

    # -- saved acquisitions made by the real acquisition code ----------------

    def range_source(self, day, run="g5-old-range"):
        root = self.work_root.parent / run
        root.mkdir(mode=0o700, exist_ok=True)
        controller = B.BackfillController(
            None, R.LiveRangeAdapter(root, OLD_RELEASE, F.mock_read_capability,
                                     provider_factory=self.adapter.provider_factory),
            release_sha=OLD_RELEASE, work_capability=RangeWorkCapability.for_local(root, root.name),
            publication_roots=self.roots, read_capability_factory=F.mock_read_capability,
            settings=self.settings)
        self.assertTrue(controller.acquire(day)["ok"])
        return root / "nights" / f"night-{day}"

    def canary_source(self, day, tag="a", segment_size=None):
        segment_size = segment_size or self.settings.segment_size
        root = self.canary / f"g63-canary-test-{tag}-{day}"
        root.mkdir(mode=0o700)
        capability = F.mock_read_capability(root, root.name, day, CANARY_RELEASE)
        provider = self.adapter.provider_factory(capability)
        adapter = R.LiveRangeAdapter(root, CANARY_RELEASE, None)
        request = adapter.acquisition_request(day)
        B._write_json_new(root / "request.json", B._request_document(request))
        configuration = B.configuration_sha256(CANARY_RELEASE, adapter, segment_size)
        bindings = B.query_checkpoint_bindings(root.name, CANARY_RELEASE, configuration, adapter, request)
        result = provider.query_resumable(request, bindings).require_completed()
        seal_query_result_checkpoint(root, result, bindings)
        loaded = load_query_result_checkpoint(root, request, bindings)
        binding = B.fetch_checkpoint_binding(root.name, CANARY_RELEASE, configuration, adapter, request,
                                             loaded, segment_size)
        SegmentedFetchCheckpoint.open(capability, binding).fetch_missing(
            B.BackfillController._ordered_ids(loaded), lambda ids: provider.fetch_segment(request, ids))
        return root

    def adoption_authority(self, start, end, sources, edit=None):
        binding = self.controller().range_binding(start, end)
        adopted = {day: B.describe_saved_acquisition(root, day, self.adapter, self.settings.segment_size).entry
                   for day, root in sources.items()}
        if edit is not None:
            adopted = edit(json.loads(json.dumps(adopted)))
        initial = P.production_binding_from_sentinel(self.observer.sentinel(start))
        scope = B.RangePublicationAuthorization(
            start, end, B._previous(start), {key: initial[key] for key in B._INITIAL_SENTINEL_FIELDS},
            candidate_release_sha=G5.RELEASE, publisher_release_sha=F.PUBLISHER_RELEASE,
            authorized_by="Control", authorized_at_utc=F.AUTHORIZED_AT, expires_at_utc=F.EXPIRES_AT,
            adopted_acquisitions=adopted, **binding)
        authority = R.ProductionRangeAuthorization(
            scope, str(self.work_root), "arnor.fixture", os.geteuid(),
            hashlib.sha256(self.token.encode()).hexdigest(), R.implementation_identity(),
            publisher_wheel_sha256="e" * 64, segment_size=2)
        path = G5.control_approval(authority, self.token, self.parent / "control" / f"approval-{uuid.uuid4().hex}.json")
        approval = R.ControlRangeApproval.load(path, work_root=self.work_root)
        publisher = R.ProductionRangePublisher(authority, control_token=self.token, approval=approval)
        publisher.mountinfo_lines = F.mountinfo_for(self.capability.published_root)
        return authority, publisher

    def no_network(self):
        refuse = AssertionError("network access during adoption")
        stack = mock.patch.multiple(socket.socket, connect=mock.Mock(side_effect=refuse),
                                    connect_ex=mock.Mock(side_effect=refuse))
        return stack

    def published(self):
        return P.authoritative_nights(self.capability.published_root)

    # -- tests ----------------------------------------------------------------

    def test_saved_canary_and_range_acquisitions_publish_in_order_without_antares(self):
        sources = {DAYS[0]: self.canary_source(DAYS[0]), DAYS[1]: self.range_source(DAYS[1]),
                   DAYS[2]: self.canary_source(DAYS[2])}
        before = {day: tree_digest(root) for day, root in sources.items()}
        authority, publisher = self.adoption_authority(DAYS[0], DAYS[2], sources)
        self.queried.clear()
        self.fetched.clear()
        controller = self.controller(publisher, authority.scope)
        with self.no_network():
            result = controller.run(DAYS[0], DAYS[2])
        self.assertEqual([night["stage"] for night in result["nights"]], ["PUBLISHED"] * 3, result)
        self.assertEqual((self.queried, self.fetched), ([], []))
        self.assertEqual({day: tree_digest(root) for day, root in sources.items()}, before)
        self.assertEqual(self.published()[-3:], DAYS)
        published_at = []
        for index, day in enumerate(DAYS):
            workspace = controller.workspace(day)
            record = json.loads((workspace.candidate / "candidate-record.json").read_text())
            provenance = record["provenance"]
            self.assertEqual(provenance["acquisition_source"], authority.scope.adopted_acquisitions[day])
            self.assertEqual(provenance["range_authorization_sha256"], authority.scope.digest)
            self.assertEqual(provenance["query_identity"],
                             authority.scope.adopted_acquisitions[day]["query_integrity_sha256"])
            self.assertFalse((workspace.root / "checkpoints").exists())
            self.assertEqual(json.loads(workspace.adoption.read_text()), authority.scope.adopted_acquisitions[day])
            # Rebuilt against the current predecessor state, never the source's.
            self.assertEqual(provenance["prior_locus_count"], 6 + 3 * index)
            self.assertEqual(record["prior_locus_identity_sha256"], controller._current_prior_identity(day))
            published_at += [event["utc"] for event in workspace.read_events()
                             if event.get("stage") == "PUBLISHED"]
        self.assertEqual(published_at, sorted(published_at))
        # A replayed run publishes nothing twice.
        journals = len(P.load_transaction_journals(self.capability.journal_root))
        again = controller.run(DAYS[0], DAYS[2], resume=True)
        self.assertEqual([night["stage"] for night in again["nights"]], ["PUBLISHED"] * 3)
        self.assertEqual(len(P.load_transaction_journals(self.capability.journal_root)), journals)

    def test_mixed_range_queries_only_unadopted_nights(self):
        sources = {DAYS[0]: self.range_source(DAYS[0])}
        authority, publisher = self.adoption_authority(DAYS[0], DAYS[1], sources)
        self.queried.clear()
        result = self.controller(publisher, authority.scope).run(DAYS[0], DAYS[1])
        self.assertEqual([night["stage"] for night in result["nights"]], ["PUBLISHED"] * 2, result)
        self.assertEqual(set(self.queried), {DAYS[1]})
        live = json.loads((self.work_root / "nights" / f"night-{DAYS[1]}" / "candidate" /
                           "candidate-record.json").read_text())
        self.assertNotIn("acquisition_source", live["provenance"])

    def test_tampered_saved_evidence_fails_closed_before_any_effect(self):
        def blob(root):
            flip_last_byte(sorted((root / "checkpoints/live-fetch-v1/blobs").iterdir())[0])

        def records(root):
            flip_last_byte(sorted((root / "checkpoints/query-result").glob("records-*"))[0])

        def request(root):
            path = root / "request.json"
            document = json.loads(path.read_text())
            document["range_label"] = "tampered"
            os.chmod(path, 0o600)
            path.write_text(json.dumps(document))

        def journal_head(root):
            path = root / "checkpoints/query-progress-v2/HEAD.json"
            head = json.loads(path.read_text())
            head["count"] -= 1
            os.chmod(path, 0o600)
            path.write_text(json.dumps(head, sort_keys=True, separators=(",", ":")))

        for tamper in (blob, records, request, journal_head):
            with self.subTest(tamper=tamper.__name__):
                root = self.canary_source(DAYS[0], tag=tamper.__name__)
                authority, publisher = self.adoption_authority(DAYS[0], DAYS[0], {DAYS[0]: root})
                tamper(root)
                with self.assertRaises(B.BackfillRefused):
                    self.controller(publisher, authority.scope).run(DAYS[0], DAYS[0])
                self.assertEqual(self.published()[-1], "2026-06-27")
                self.assertFalse((self.work_root / "nights").exists())

    def test_tamper_after_adoption_is_refused_at_construction(self):
        root = self.canary_source(DAYS[0])
        authority, publisher = self.adoption_authority(DAYS[0], DAYS[0], {DAYS[0]: root})
        controller = self.controller(publisher, authority.scope)
        self.assertTrue(controller.acquire(DAYS[0])["ok"])
        flip_last_byte(sorted((root / "checkpoints/live-fetch-v1/blobs").iterdir())[0])
        built = controller.construct(DAYS[0], ())
        self.assertFalse(built["ok"])
        self.assertEqual(controller.night_state(DAYS[0])["stage"], "BLOCKED")
        self.assertFalse(controller.workspace(DAYS[0]).candidate.exists())

    def test_wrong_release_configuration_run_id_provider_or_location_fails_closed(self):
        root = self.canary_source(DAYS[0])

        def entry_edit(field, value):
            def edit(adopted):
                adopted[DAYS[0]][field] = value
                return adopted
            return edit

        for field, value in (("source_release_sha", OLD_RELEASE),
                             ("source_configuration_sha256", "f" * 64),
                             ("fetch_completion_sha256", "e" * 64)):
            with self.subTest(field=field):
                authority, publisher = self.adoption_authority(DAYS[0], DAYS[0], {DAYS[0]: root},
                                                               edit=entry_edit(field, value))
                with self.assertRaises(B.BackfillRefused):
                    self.controller(publisher, authority.scope).run(DAYS[0], DAYS[0])
        with self.assertRaises(B.BackfillRefused):
            self.adoption_authority(DAYS[0], DAYS[0], {DAYS[0]: root},
                                    edit=entry_edit("source_provider_implementation_sha256", "c" * 64))
        with self.assertRaises(B.BackfillRefused):  # a different segment size is another configuration
            B.describe_saved_acquisition(root, DAYS[0], self.adapter, self.settings.segment_size + 1)
        renamed = root.with_name(root.name + "-renamed")
        os.rename(root, renamed)
        with self.assertRaises(B.BackfillRefused):  # the sealed run id no longer matches its root
            B.describe_saved_acquisition(renamed, DAYS[0], self.adapter, self.settings.segment_size)
        elsewhere = self.parent / "elsewhere"
        elsewhere.mkdir(mode=0o700)
        os.rename(renamed, elsewhere / root.name)
        with self.assertRaises(B.BackfillRefused):  # not a canary or range night location
            B.describe_saved_acquisition(elsewhere / root.name, DAYS[0], self.adapter, self.settings.segment_size)
        with self.assertRaises(B.BackfillRefused):  # another night's acquisition
            B.describe_saved_acquisition(self.range_source(DAYS[0]), DAYS[1], self.adapter, self.settings.segment_size)

    def test_wrong_predecessor_publishes_nothing(self):
        authority, publisher = self.adoption_authority(DAYS[1], DAYS[1], {DAYS[1]: self.canary_source(DAYS[1])})
        night = self.controller(publisher, authority.scope).run(DAYS[1], DAYS[1])["nights"][0]
        self.assertEqual(night["stage"], "BLOCKED", night)
        self.assertFalse(night["blocked"]["retryable"], night)
        self.assertEqual(self.published()[-1], "2026-06-27")

    def test_interrupted_rebuild_and_publication_resume_exactly_once(self):
        sources = {DAYS[0]: self.canary_source(DAYS[0]), DAYS[1]: self.range_source(DAYS[1])}
        authority, publisher = self.adoption_authority(DAYS[0], DAYS[1], sources)
        for point in ("before_construct", "before_publish"):
            with self.subTest(point=point):
                controller = self.controller(publisher, authority.scope)

                def die(name, details, point=point):
                    if name == point and details.get("date_utc") == DAYS[1]:
                        raise SystemExit(f"process death at {point}")

                controller.event_hook = die
                with self.assertRaises(SystemExit):
                    controller.run(DAYS[0], DAYS[1], resume=point != "before_construct")
                self.assertEqual(self.published()[-1], DAYS[0])
        resumed = self.controller(publisher, authority.scope).run(DAYS[0], DAYS[1], resume=True)
        self.assertEqual([night["stage"] for night in resumed["nights"]], ["PUBLISHED"] * 2, resumed)
        self.assertEqual(self.published()[-2:], DAYS[:2])
        complete = [journal for journal in P.load_transaction_journals(self.capability.journal_root)
                    if journal.snapshot.published]
        nights = [P._metadata(journal).get("target_utc_night") for journal in complete]
        self.assertEqual(sorted(day for day in nights if day in DAYS), list(DAYS[:2]))

    def test_ranges_without_adoption_keep_their_accepted_digest(self):
        scope = self.authority.scope
        self.assertEqual(scope.adopted_acquisitions, {})
        legacy = {name: getattr(scope, name) for name in B._RANGE_FIELDS}
        self.assertEqual(scope.as_dict(), json.loads(B._canonical(legacy)))
        self.assertNotIn("adopted_acquisitions", self.controller().range_binding(DAYS[0], DAYS[1]))

    def test_plan_binds_adoption_and_execute_refuses_ad_hoc_adoption(self):
        root = self.canary_source(DAYS[0], segment_size=256)  # the operator range segment size
        args = ["--start", DAYS[0], "--end", DAYS[1], "--work-root", str(self.work_root),
                "--candidate-release", G5.RELEASE, "--adopt", f"{DAYS[0]}={root}"]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            R.main(["plan", *args])
        plan = json.loads(printed.getvalue())
        self.assertEqual(plan["execution"], "NOT EXECUTED")
        expected = B.describe_saved_acquisition(root, DAYS[0], self.adapter, 256).entry
        self.assertEqual(plan["range_binding"]["adopted_acquisitions"], {DAYS[0]: expected})
        with self.assertRaises(B.BackfillRefused):
            R.main(["plan", *args, "--adopt", f"{DAYS[1]}={root}"])  # wrong night for that source
        with self.assertRaises(B.BackfillRefused):
            R.main(["execute", *args, "--execute-authorized-range", "--authorization", "a.json",
                    "--publisher-wheel-sha256", "e" * 64, "--control-token-file", "t",
                    "--control-approval", "a"])

    def test_real_saved_acquisition_identities(self):
        adapter = R.LiveRangeAdapter(self.work_root, G5.RELEASE, None)
        for release, configuration in REAL_SOURCE_CONFIGURATIONS.items():
            self.assertEqual(B.configuration_sha256(release, adapter, 256), configuration)
        self.assertEqual(B.ADOPTABLE_SOURCE_PROVIDERS, {
            "f22578a51ca65a3cf41d7fc8690fc026ccc0a16aefb300fc9c488cfb99c04199",
            "afe11a1b0846ed20293d503b393d477f3b4309fcfa195b13320170ed4e18d14c"})


if __name__ == "__main__":
    unittest.main()
