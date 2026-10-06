"""Offline scientific/authority/determinism regression tests for the native adapter."""

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import v3_fixtures as F
from src import history
from src.observatory import (
    build_metadata_fixture, build_native_product, canonical_bytes, native_sha256,
    serialize_shared_bundle,
)
from src.observatory.adapter import NativeReadRefused, identifier_json
from src.observatory.features import BROKER_FEATURES, numeric_value
from src.observatory.model import BASELINE_SHA
from src.observatory.serialization import SharedContractUnbound
from src.observatory.spatial import build_sky
from src.operations import publication as P
from src.operations.science import SyntheticScienceProvider, build_night_artifacts


NIGHT = "2026-07-01"
NEXT = "2026-07-02"
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_observatory_bundle", ROOT / "scripts/export_observatory_bundle.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class ObservatoryFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)
        self.cap = F.make_capability(self.tmp)
        self.root = self.cap.published_root
        self.journals = self.cap.journal_root
        self.write(NIGHT)
        # Existing locking permits provisioning only in temporary fixtures.
        with history.authority_read_lock(self.root):
            pass

    def write(self, night, zero=False):
        artifacts = F.synthetic_artifacts(night)
        if zero:
            result = SyntheticScienceProvider("success_zero").fetch_night(F.night_request(night))
            artifacts = build_night_artifacts(result)
        return F.write_night(self.root, night, artifacts)

    def change(self, update_loci=None, update_manifest=None):
        paths = history.nightly_paths(self.root, NIGHT)
        manifest = json.loads(paths["manifest"].read_bytes())
        if update_loci:
            loci = pd.read_parquet(paths["loci"])
            update_loci(loci)
            loci.to_parquet(paths["loci"], index=False)
            payload = paths["loci"].read_bytes()
            manifest["artifacts"]["loci.parquet"] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        if update_manifest:
            update_manifest(manifest)
        paths["manifest"].write_bytes(F.manifest_bytes(manifest))

    def build(self, **kwargs):
        options = dict(start=NIGHT, end=NIGHT, code_sha=BASELINE_SHA,
                       input_kind="SYNTHETIC_AUTHORITY_FIXTURE")
        options.update(kwargs)
        return build_native_product(self.root, self.journals, **options)

    def test_legacy_complete_retains_partial_authority_qualification(self):
        product = self.build()
        night = product.time.nights[0]
        self.assertEqual(night.qualification.authority_state, "COMPLETE")
        self.assertEqual(night.qualification.scientific_qualification, "PARTIALLY_QUALIFIED")
        self.assertEqual(night.qualification.content_state, "COMPLETE_WITH_ROWS")
        self.assertEqual((night.loci_rows, night.alert_source_rows), (2, 4))

    def test_zero_missing_and_outside_are_distinct(self):
        shutil.rmtree(history.nightly_dir(self.root, NIGHT))
        self.write(NIGHT, zero=True)
        product = self.build(end=NEXT, ribbon_start="2026-06-30", ribbon_end="2026-07-03")
        states = product.time.nights
        self.assertEqual(states[0].qualification.view_state, "OUTSIDE_BASIS_COVERAGE")
        self.assertEqual(states[1].qualification.content_state, "ZERO_ROW_NIGHT")
        self.assertEqual(states[1].alert_source_rows, 0)
        self.assertEqual(states[2].qualification.content_state, "MISSING")
        self.assertIsNone(states[2].loci_rows)
        self.assertEqual(states[2].qualification.authority_state, "NOT_COMMITTED")

    def test_missing_product_never_means_zero(self):
        history.nightly_paths(self.root, NIGHT)["alerts"].unlink()
        product = self.build()
        self.assertEqual(product.time.nights[0].qualification.content_state, "MISSING")
        self.assertIsNone(product.time.nights[0].alert_source_rows)
        self.assertEqual(product.loci, ())

    def test_v3_manifest_without_journal_is_contradiction(self):
        self.change(update_manifest=lambda item: item.update(authority={"state": "authoritative"}))
        product = self.build()
        self.assertEqual(product.time.nights[0].qualification.authority_state, "CONTRADICTION")
        self.assertFalse(product.loci)

    def test_synthetic_source_cannot_be_relabelled_real(self):
        with self.assertRaisesRegex(NativeReadRefused, "Synthetic"):
            self.build(input_kind="SAVED_SCIENCE_SNAPSHOT")

    def test_deterministic_bytes_hashes_and_no_source_mutation(self):
        before = tree_bytes(self.cap.root)
        first, second = self.build(), self.build()
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual(native_sha256(first), hashlib.sha256(canonical_bytes(first)).hexdigest())
        self.assertEqual(before, tree_bytes(self.cap.root))
        with self.assertRaises(FrozenInstanceError):
            first.product_kind = "science"

    def test_row_order_is_canonical_and_input_hash_detects_change(self):
        first = self.build()
        self.change(update_loci=lambda frame: frame.sort_values("locus_id", ascending=False, inplace=True))
        second = self.build()
        self.assertEqual(first.loci, second.loci)
        self.assertNotEqual(first.basis.generation_sha256, second.basis.generation_sha256)

    def test_locus_membership_snapshots_survive_later_changes(self):
        self.write(NEXT)
        self.change(update_loci=lambda frame: frame.__setitem__("tags", ["old_tag", "old_tag"]))
        product = self.build(end=NEXT)
        locus = product.loci[0]
        self.assertEqual(locus.membership_nights, (NIGHT, NEXT))
        self.assertEqual(locus.snapshots[0].tags, ("old_tag",))
        self.assertEqual(locus.snapshots[1].tags, ("lsst", "synthetic"))
        self.assertEqual(product.sky.input_loci, 2)
        self.assertEqual(locus.snapshots[0].saved_alert_source_rows, 2)

    def test_multisurvey_large_identifier_and_nested_multiplicity_preserved(self):
        big = 2**63 - 1
        def update(frame):
            frame["dia_object_id"] = pd.Series([big, big - 1], dtype="int64")
            frame["survey"] = [{"lsst": {"ss_object_id": [str(big), str(big)]}, "ztf": {"id": "ZTF001"}}] * 2
        self.change(update_loci=update)
        snapshot = self.build().loci[0].snapshots[0]
        values = {(item.namespace, item.source_field): json.loads(item.value_json) for item in snapshot.external_identifiers}
        self.assertEqual(values[("lsst.dia_object_id", "dia_object_id")], str(big))
        self.assertEqual(values[("lsst.ss_object_id", "survey.lsst.ss_object_id (nested)")], [str(big), str(big)])
        self.assertEqual(values[("ztf.object_id", "survey.ztf.id (nested)")], "ZTF001")

    def test_missing_tags_and_identifier_content_do_not_imply_known_absence(self):
        self.change(update_loci=lambda frame: frame.__setitem__("tags", [None, ""]))
        product = self.build()
        self.assertEqual(product.loci[0].snapshots[0].tag_state, "MISSING")
        self.assertEqual(product.loci[1].snapshots[0].tag_state, "KNOWN_EMPTY")
        identifier = next(item for item in product.loci[0].snapshots[0].external_identifiers if item.source_field == "ztf_object_id")
        self.assertEqual(identifier.availability, "MISSING")

    def test_lossy_float_id_rejects_partition(self):
        self.change(update_loci=lambda frame: frame.__setitem__("dia_object_id", [float(2**63), 1.0]))
        product = self.build()
        self.assertFalse(product.loci)
        self.assertEqual(product.time.nights[0].qualification.reason, "LOSSY_OR_MALFORMED_IDENTIFIER")

    def test_feature_definitions_remain_partial_and_colors_unavailable(self):
        self.change(update_loci=lambda frame: frame.__setitem__("feature_weighted_mean_magn_g", [20.0, 21.0]))
        definitions = {item.name: item for item in self.build().capabilities.features}
        mean = definitions["feature_weighted_mean_magn_g"]
        self.assertEqual(mean.qualification_state, "PARTIALLY_QUALIFIED")
        self.assertIsNone(mean.unit)
        self.assertIn("no generating estimator", mean.estimator_semantics)
        self.assertEqual(mean.finite_rows, 2)
        self.assertEqual(definitions["color_g_minus_r"].qualification_state, "UNAVAILABLE")
        self.assertEqual(definitions["saved_alert_source_rows"].qualification_state, "AVAILABLE")

    def test_feature_missingness_absent_null_nonfinite_and_malformed(self):
        self.change(update_loci=lambda frame: frame.__setitem__(BROKER_FEATURES[0], [1.0, np.inf]))
        product = self.build()
        definitions = {item.name: item for item in product.capabilities.features}
        chi = definitions[BROKER_FEATURES[0]]
        self.assertEqual((chi.finite_rows, chi.missing_rows), (1, 1))
        self.assertEqual(chi.missingness_counts, (("NONFINITE", 1),))
        self.assertEqual(definitions[BROKER_FEATURES[1]].missingness_counts, (("ABSENT_COLUMN", 2),))
        self.assertEqual(numeric_value(None), (None, "NULL"))
        self.assertEqual(numeric_value("1"), (None, "MALFORMED"))
        self.assertEqual(numeric_value(True), (None, "MALFORMED"))
        json.loads(canonical_bytes(product))

    def test_time_conventions_do_not_invent_observation_scale(self):
        product = self.build()
        self.assertEqual(product.time.observation_time_scale, "UNESTABLISHED")
        self.assertIn("not alerts observed", product.time.alert_count_definition)
        self.assertIsNone(product.time.nights[0].publication_started_at_utc)
        self.assertIsNone(product.time.nights[0].fetch_completed_at_utc)
        self.assertIsNotNone(product.time.nights[0].build_finished_at_utc)
        self.assertEqual(product.time.nights[0].mjd_upper_bound, "inclusive")
        provenance = next(item for item in product.provenance if item.identity == "semantics:time")
        self.assertIn("leap-second", provenance.evidence_json)
        self.assertNotEqual(product.loci[0].snapshots[0].newest_alert_observation_mjd,
                            product.time.nights[0].mjd_min)

    def test_wrong_calendar_or_membership_night_is_rejected(self):
        self.change(update_manifest=lambda item: item.update(mjd_min=item["mjd_min"] - 1))
        self.assertFalse(self.build().loci)
        self.assertEqual(self.build().time.nights[0].qualification.reason, "MJD_CALENDAR_NIGHT_MISMATCH")

    def test_membership_mismatch_and_incomplete_acquisition_are_rejected(self):
        self.change(update_loci=lambda frame: frame.__setitem__("night_date_utc", NEXT))
        self.assertEqual(self.build().time.nights[0].qualification.reason, "MEMBERSHIP_NIGHT_MISMATCH")
        self.change(update_loci=lambda frame: frame.__setitem__("night_date_utc", NIGHT),
                    update_manifest=lambda item: item["query_fetch_evidence"].update(fetch_completed=False))
        self.assertEqual(self.build().time.nights[0].qualification.reason, "INCOMPLETE_QUERY_FETCH_EVIDENCE")

    def test_zero_requires_native_proof_not_only_empty_files(self):
        shutil.rmtree(history.nightly_dir(self.root, NIGHT))
        self.write(NIGHT, zero=True)
        self.change(update_manifest=lambda item: item.update(chunk_count=0))
        product = self.build()
        self.assertEqual(product.time.nights[0].qualification.availability, "UNAVAILABLE")
        self.assertIsNone(product.time.nights[0].loci_rows)

    def test_locus_integer_ids_are_strings_and_keep_alert_links(self):
        paths = history.nightly_paths(self.root, NIGHT)
        big = 2**63 - 1
        alerts = pd.read_parquet(paths["alerts"])
        alerts["locus_id"] = pd.Series([big, big, big - 1, big - 1], dtype="int64")
        alerts.to_parquet(paths["alerts"], index=False)
        payload = paths["alerts"].read_bytes()
        self.change(update_loci=lambda frame: frame.__setitem__("locus_id", pd.Series([big, big - 1], dtype="int64")),
            update_manifest=lambda item: item["artifacts"].update({"alerts.parquet": {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}}))
        product = self.build()
        self.assertEqual({locus.locus_id for locus in product.loci}, {str(big), str(big - 1)})
        self.assertTrue(all(locus.snapshots[0].saved_alert_source_rows == 2 for locus in product.loci))

    def test_generation_guard_detects_writer_bypassing_lock(self):
        tokens = iter([("before",), ("before",), ("before",), ("changed",)])
        with mock.patch("src.history.authority_generation", side_effect=lambda root: next(tokens)):
            with self.assertRaises(history.PublicationInProgress):
                self.build()

    def test_invalid_coordinates_accounted_without_placeholder_positions(self):
        self.change(update_loci=lambda frame: frame.__setitem__("ra", [np.inf, -1.0]))
        product = self.build()
        self.assertEqual(product.sky.included_loci, 0)
        self.assertEqual(sum(count for _, count in product.sky.excluded_coordinates), 2)
        self.assertIsNone(product.loci[0].snapshots[0].ra_deg)
        self.assertEqual(product.sky.cells, ())

    def test_spatial_geometry_equal_area_and_pole_boundaries(self):
        self.change(update_loci=lambda frame: (frame.__setitem__("ra", [0.0, 359.99]), frame.__setitem__("dec", [-90.0, 90.0])))
        product = self.build()
        self.assertEqual(product.sky.included_loci, 2)
        self.assertEqual(sum(cell.locus_count for cell in product.sky.cells), 2)
        areas = [(cell.ra_max_deg - cell.ra_min_deg) * (cell.sin_dec_max - cell.sin_dec_min) for cell in product.sky.cells]
        self.assertAlmostEqual(areas[0], areas[1])
        self.assertIn("footprint unavailable", product.sky.interpretation)
        with self.assertRaises(ValueError):
            build_sky(product.loci, (), ra_bins=0)

    def test_input_artifact_mismatch_rejects_content(self):
        paths = history.nightly_paths(self.root, NIGHT)
        frame = pd.read_parquet(paths["loci"])
        frame["tags"] = "changed"
        frame.to_parquet(paths["loci"], index=False)
        self.assertEqual(self.build().time.nights[0].qualification.reason, "ARTIFACT_HASH_MISMATCH")

    def test_entity_bound_refuses_truncation(self):
        with self.assertRaisesRegex(NativeReadRefused, "bound exceeded"):
            self.build(max_loci=1)

    def test_provenance_references_resolve_and_sentinel_is_identity_only(self):
        sentinel = self.tmp / "sentinel.json"
        sentinel.write_text('{"durable_fingerprint_sha256":"supplied"}')
        product = self.build(sentinel_path=sentinel)
        provenance = {item.identity for item in product.provenance}
        artifact_ids = {item.identity for item in product.basis.input_artifacts}
        self.assertTrue(set(product.time.provenance_refs) <= provenance)
        self.assertTrue(set(product.sky.provenance_refs) <= provenance)
        for record in product.provenance:
            self.assertTrue(set(record.artifact_refs) <= artifact_ids)
        for definition in product.capabilities.features:
            self.assertTrue(set(definition.evidence_refs) <= provenance)
        self.assertIn(product.basis.sentinel_artifact_ref, artifact_ids)
        self.assertEqual(product.basis.sentinel_qualification, "NOT_REQUALIFIED_BY_ADAPTER")

    def test_metadata_fixture_has_no_fake_science_and_shared_boundary_fails_closed(self):
        product = build_metadata_fixture(code_sha=BASELINE_SHA)
        self.assertEqual(product.product_kind, "METADATA / CONTRACT FIXTURE")
        self.assertEqual(product.derivation, "DERIVED / NON-AUTHORITATIVE")
        self.assertFalse(product.loci)
        self.assertFalse(product.time.nights)
        self.assertFalse(product.sky.cells)
        self.assertTrue(all(item.qualification_state == "UNAVAILABLE" for item in product.capabilities.features))
        for identity in (None, {"schema": "Bundle V1", "hash": "claimed"}):
            with self.assertRaises(SharedContractUnbound):
                serialize_shared_bundle(product, owner_identity=identity)

    def test_serializer_rejects_invalid_json_and_unsafe_numbers(self):
        from fractions import Fraction
        for invalid in (np.inf, float("nan"), 2**63, {1: "nonstring-key"}, object()):
            with self.assertRaises((ValueError, TypeError)):
                canonical_bytes(invalid)
        self.assertEqual(identifier_json(["0001", 2**63, None]), ["0001", str(2**63), None])
        with self.assertRaises(NativeReadRefused):
            identifier_json(Fraction(7000000000000000001, 10**18))

    def test_no_live_functions_invoked(self):
        with mock.patch("src.query.query_range", side_effect=AssertionError("live query")), mock.patch("src.lightcurves.load_lightcurves", side_effect=AssertionError("fetch")):
            self.assertEqual(len(self.build().loci), 2)

    def test_symlink_inputs_refused(self):
        link = self.tmp / "link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(NativeReadRefused):
            build_native_product(link, self.journals, start=NIGHT, end=NIGHT,
                input_kind="SYNTHETIC_AUTHORITY_FIXTURE", code_sha=BASELINE_SHA)

    def test_export_protects_source_and_fails_before_shared_output(self):
        output = self.tmp / "native.json"
        self.assertEqual(EXPORT.main(["--metadata-fixture", "--output", str(output)]), 0)
        self.assertEqual(json.loads(output.read_bytes())["loci"], [])
        failed = self.tmp / "shared.json"
        self.assertEqual(EXPORT.main(["--metadata-fixture", "--format", "shared", "--output", str(failed)]), 1)
        self.assertFalse(failed.exists())
        before = tree_bytes(self.root)
        self.assertEqual(EXPORT.main(["--offline-root", str(self.root), "--journal-root", str(self.journals), "--output", str(history.nightly_paths(self.root, NIGHT)["manifest"])]), 1)
        self.assertEqual(before, tree_bytes(self.root))

    def test_hardlinked_output_cannot_mutate_source(self):
        import os
        source = history.nightly_paths(self.root, NIGHT)["manifest"]
        before = source.read_bytes()
        output = self.tmp / "hardlink.json"
        os.link(source, output)
        self.assertEqual(EXPORT.main(["--metadata-fixture", "--output", str(output)]), 0)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(json.loads(output.read_bytes())["loci"], [])


class ObservatoryV3AuthorityTests(unittest.TestCase):
    def setUp(self):
        # Reuse the accepted V3 fixture, including Sentinel and publication journals.
        from test_publication_authority_v3 import AuthorityFixture
        self.fixture = AuthorityFixture("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def build(self):
        fixture = self.fixture
        return build_native_product(fixture.root, fixture.capability.journal_root,
            start=NIGHT, end=NIGHT, code_sha=BASELINE_SHA, input_kind="SYNTHETIC_AUTHORITY_FIXTURE")

    def test_uncommitted_pending_candidate_has_no_rows(self):
        product = self.build()
        self.assertEqual(product.time.nights[0].qualification.authority_state, "NOT_COMMITTED")
        self.assertFalse(product.loci)

    def test_published_finalized_v3_has_complete_evidence(self):
        fixture = self.fixture
        fixture.publisher.publish(fixture.candidate, fixture.authorization)
        product = self.build()
        self.assertEqual(product.time.nights[0].qualification.authority_state, "COMPLETE")
        self.assertEqual(product.time.nights[0].qualification.scientific_qualification, "QUALIFIED_SAVED_CONTENT")
        self.assertEqual(len(product.loci), 2)
        self.assertIsNotNone(product.time.nights[0].publication_started_at_utc)
        self.assertTrue(any(item.identity.startswith("authority:journal:") for item in product.basis.input_artifacts))

    def test_reconciliation_gate_never_exposes_rows(self):
        from test_publication_authority_v3 import _FailAt
        fixture = self.fixture
        publisher = F.publisher_for(fixture.capability, fault_hook=_FailAt("after_authority_gate"))
        outcome = publisher.publish(fixture.candidate, fixture.authorization)
        self.assertFalse(outcome.success)
        before = tree_bytes(fixture.capability.root)
        product = self.build()
        self.assertEqual(product.time.nights[0].qualification.authority_state, "RECONCILIATION_REQUIRED")
        self.assertFalse(product.loci)
        self.assertIsNone(product.time.nights[0].loci_rows)
        self.assertEqual(before, tree_bytes(fixture.capability.root))

    def test_pending_pre_gate_is_not_promoted(self):
        import signal
        from test_publication_authority_v3 import run_child
        fixture = self.fixture
        completed = run_child({"mode": "publish", "run_root": str(fixture.capability.root),
            "run_id": fixture.capability.run_id, "night_root": str(fixture.night_root),
            "authorization": str(fixture.authorization_path), "boundary": "before_authority_gate"}, fixture.tmp)
        self.assertEqual(completed.returncode, -signal.SIGKILL)
        product = self.build()
        self.assertEqual(product.time.nights[0].qualification.view_state, "NOT_COMMITTED")
        self.assertFalse(product.loci)

    def test_other_complete_nights_are_unavailable_under_global_gate(self):
        from test_publication_authority_v3 import _FailAt
        fixture = self.fixture
        F.publisher_for(fixture.capability, fault_hook=_FailAt("after_authority_gate")).publish(fixture.candidate, fixture.authorization)
        product = build_native_product(fixture.root, fixture.capability.journal_root,
            start="2026-06-29", end=NIGHT, code_sha=BASELINE_SHA,
            input_kind="SYNTHETIC_AUTHORITY_FIXTURE")
        self.assertEqual(product.time.nights[0].qualification.authority_state, "COMPLETE")
        self.assertEqual(product.time.nights[0].qualification.reason, "GLOBAL_PUBLICATION_GATE")
        self.assertFalse(product.loci)


if __name__ == "__main__":
    unittest.main()
