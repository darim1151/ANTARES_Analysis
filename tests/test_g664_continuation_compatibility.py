"""G6.6.4 R2 official continuation compatibility, entirely offline.

The observed service continuation has an absolute HTTP spelling for the same
listing.  Its representation is upgraded before the transport sees it.  These
tests bind that narrow compatibility behavior to HTTPS request destinations,
unchanged query bytes, finite-page science equivalence, and terminal refusal
for ambiguous or hostile representations.  The unchanged process transport's
existing qualification is reused through the established offline harness.
"""

import os
import unittest
from urllib.parse import urlsplit

import requests

import test_g663_transport_guard as H
from src.operations import live_antares as L


class HttpContinuationService(H.FakeAntares):
    """The existing JSON:API fixture with the live service's HTTP next spelling."""

    def next_url(self, raw_query, offset, *, gap=False):
        canonical = super().next_url(raw_query, offset, gap=gap)
        return "http://" + canonical.removeprefix("https://")


class CanonicalContinuationTests(H.OfflineTestCase):
    def canonical(self, target):
        return L._canonical_p2_continuation(target, host=H.HOST, listing_path="/v1/loci")

    def test_observed_representation_changes_only_the_trusted_https_authority(self):
        # Preserve ordering, duplicate names, empty values, pluses, and reserved
        # escapes.  Parsing/re-encoding parameters would change these bytes.
        raw_query = (
            "sort=-properties.newest_alert_observation_time&"
            "elasticsearch_query%5Blocus_listing%5D=%7B%22query%22%3A%7B%7D%7D&"
            "page%5Boffset%5D=10&duplicate=first&duplicate=last&empty=&"
            "value=a+b%2Bc%2Fd%26e%3Df%25&opaque=%40evil.example"
        )
        for authority in (
                "http://" + H.HOST, "http://" + H.HOST + ":80",
                "HTTP://" + H.HOST.upper() + ":80",
                "https://" + H.HOST, "https://" + H.HOST + ":443",
                "HTTPS://" + H.HOST.upper() + ":443"):
            with self.subTest(authority=authority):
                target = authority + "/v1/loci?" + raw_query
                canonical = self.canonical(target)
                self.assertEqual(canonical, H.LISTING + "?" + raw_query)
                self.assertEqual(urlsplit(canonical).query, urlsplit(target).query)
                prepared = requests.Request("GET", canonical).prepare().url
                parts = urlsplit(prepared)
                self.assertEqual((parts.scheme, parts.netloc, parts.path),
                                 ("https", H.HOST, "/v1/loci"))
                self.assertEqual(parts.query, raw_query)

    def test_hostile_or_ambiguous_continuations_are_typed_terminal_refusals(self):
        malformed = L.P2MalformedContinuationError
        cross_origin = L.P2CrossOriginContinuationError
        path_error = L.P2ContinuationPathError
        relative = L.P2RelativeContinuationError
        insecure = L.P2InsecureContinuationError
        cases = [
            (None, malformed), (False, malformed), (7, malformed), ([], malformed),
            ({"href": H.LISTING}, malformed), ("", malformed),
            ("/v1/loci?p=2", relative), ("?p=2", relative),
            ("//" + H.HOST + "/v1/loci?p=2", relative),
            ("ftp://" + H.HOST + "/v1/loci?p=2", insecure),
            ("file://" + H.HOST + "/v1/loci?p=2", insecure),
            ("https://user@" + H.HOST + "/v1/loci?p=2", malformed),
            ("http://user:password@" + H.HOST + "/v1/loci?p=2", malformed),
            ("https://@" + H.HOST + "/v1/loci?p=2", malformed),
            ("http://evil.example@" + H.HOST + "/v1/loci?p=2", malformed),
            (H.LISTING + "?p=2#", malformed),
            (H.LISTING + "?p=2#fragment", malformed),
            (" " + H.LISTING + "?p=2", malformed),
            (H.LISTING + "?p=a b", malformed),
            (H.LISTING + "?p=\t", malformed),
            (H.LISTING + "?p=\n", malformed),
            (H.LISTING + "?p=\r", malformed),
            (H.LISTING + "?p=\x00", malformed),
            (H.LISTING + "?p=\x7f", malformed),
            (H.LISTING + "?p=é", malformed),
            ("http://" + H.HOST + "\\@evil.example/v1/loci?p=2", malformed),
            ("http://" + H.HOST + ":x/v1/loci?p=2", malformed),
            ("https://" + H.HOST + ":99999/v1/loci?p=2", malformed),
            ("http://evil.example/v1/loci?p=2", cross_origin),
            ("https://" + H.HOST + ".evil.example/v1/loci?p=2", cross_origin),
            ("https://" + H.HOST + "./v1/loci?p=2", cross_origin),
            ("https://%61pi.antares.noirlab.edu/v1/loci?p=2", cross_origin),
            ("https://[::1]/v1/loci?p=2", cross_origin),
            ("https:///v1/loci?p=2", cross_origin),
            ("http://" + H.HOST + ":443/v1/loci?p=2", cross_origin),
            ("http://" + H.HOST + ":81/v1/loci?p=2", cross_origin),
            ("https://" + H.HOST + ":80/v1/loci?p=2", cross_origin),
            ("https://" + H.HOST + ":8443/v1/loci?p=2", cross_origin),
            ("http://" + H.HOST + ":/v1/loci?p=2", cross_origin),
            ("http://" + H.HOST + ":080/v1/loci?p=2", cross_origin),
            ("https://" + H.HOST + ":0443/v1/loci?p=2", cross_origin),
            ("https://" + H.HOST + ":000443/v1/loci?p=2", cross_origin),
            ("http://" + H.HOST + "/v1/alerts?p=2", path_error),
            (H.LISTING + "/?p=2", path_error),
            ("http://" + H.HOST + "/v1/./loci?p=2", path_error),
            ("http://" + H.HOST + "/v1/x/../loci?p=2", path_error),
            ("http://" + H.HOST + "/v1/%6Coci?p=2", path_error),
            ("http://" + H.HOST + "/v1%2Floci?p=2", path_error),
            (H.LISTING + ";parameter?p=2", path_error),
        ]
        for target, kind in cases:
            with self.subTest(target=target):
                with self.assertRaises(kind) as caught:
                    self.canonical(target)
                self.assertIs(type(caught.exception), kind)
                self.assertFalse(L._retryable_query_error(caught.exception))


class CompatibilityPaginationTests(H.OfflineTestCase):
    body = L._build_tile_query(H.FIRST_TILE)
    first = H.FailClosedMatrixTests.first

    def run_guarded(self, service, *, stop_at=None):
        with service.installed():
            listing = H.guarded()
            try:
                rows, error = H.drain(listing.search(self.body), stop_at=stop_at)
            finally:
                listing.close()
        self.assertOfficial(service.urls)
        return rows, error

    def test_http_links_have_exact_finite_page_and_50_row_equivalence(self):
        for count, page_size, full_next, stop_at, expected_pages in (
                (7, 10, False, None, 1), (35, 10, False, None, 4),
                (40, 10, True, None, 5), (49, 7, True, None, 8),
                (60, 10, False, 50, 5), (51, 7, False, 50, 8),
                (50, 10, True, 50, 5)):
            with self.subTest(count=count, page_size=page_size, stop_at=stop_at):
                loci = list(reversed(H.tile_loci(H.FIRST_TILE, count)))
                expected_service = H.FakeAntares(loci, page_size=page_size,
                                                full_page_next=full_next)
                with expected_service.installed():
                    expected, baseline_error = H.drain(H.pinned_search(self.body), stop_at=stop_at)
                self.assertIsNone(baseline_error)
                upgraded_service = HttpContinuationService(loci, page_size=page_size,
                                                           full_page_next=full_next)
                observed, error = self.run_guarded(upgraded_service, stop_at=stop_at)
                self.assertIsNone(error)
                self.assertEqual(observed, expected)
                self.assertEqual(upgraded_service.urls, expected_service.urls)
                self.assertEqual(upgraded_service.pages, expected_pages)
                self.assertEqual([r["locus_id"] for r in observed],
                                 [l.locus_id for l in loci][:stop_at])
                # Includes exact initial sort/query parameters and all continuations.
                self.assertEqual(upgraded_service.urls[0], expected_service.urls[0])
                for call in upgraded_service.calls:
                    self.assertEqual(call["timeout"], (60, 60))
                    self.assertTrue(call["stream"])

    def test_observed_upgrade_runs_through_the_unchanged_process_transport(self):
        loci = H.tile_loci(H.FIRST_TILE, 25)
        service = HttpContinuationService(loci, page_size=10)
        with service.installed():
            # installed() selects the existing child hook for this offline service.
            from src.operations import p2_transport as T
            child = T.P2ProcessTransport(L._p2_child_config(H.CANARY.transport, H.BASE),
                                         hook=T.CHILD_TEST_HOOK)
            listing = L._GuardedListing(H.CANARY.transport, H.BASE, child, H.listing_schema)
            try:
                observed, error = H.drain(listing.search(self.body))
            finally:
                listing.close()
        self.assertIsNone(error)
        self.assertEqual([r["locus_id"] for r in observed], [l.locus_id for l in loci])
        self.assertEqual(len(service.calls), 3)
        self.assertOfficial(service.urls)
        self.assertEqual(child.state, T.CLOSED)
        self.assertEqual(child.stats["spawns"], 1)
        self.assertEqual(child.exit_record["returncode"], 0)
        self.assertTrue(child.exit_record["reaped"])
        with self.assertRaises(ProcessLookupError):
            os.kill(child.exit_record["pid"], 0)

    def test_cycles_share_the_actual_prepared_https_page_identity(self):
        loci = H.tile_loci(H.FIRST_TILE, 2)
        aliases = [
            "http://" + H.HOST + "/v1/loci?p=a",
            "http://" + H.HOST + ":80/v1/loci?p=a",
            "HTTP://" + H.HOST.upper() + ":80/v1/loci?p=a",
            "HTTPS://" + H.HOST.upper() + ":443/v1/loci?p=a",
            "http://" + H.HOST + "/v1/loci?%70=%61",
        ]
        for alias in aliases:
            with self.subTest(alias=alias):
                service = H.FakeAntares(script=H.scripted({
                    None: H.page(loci[:1], next=H.LISTING + "?p=a"),
                    "a": H.page(loci[1:], next=alias),
                }))
                observed, error = self.run_guarded(service)
                self.assertIs(type(error), L.P2ContinuationCycleError)
                self.assertFalse(L._retryable_query_error(error))
                self.assertEqual((len(observed), len(service.calls)), (2, 2))

    def test_first_page_cycle_alias_is_refused_without_a_second_request(self):
        locus = H.tile_loci(H.FIRST_TILE, 1)

        def repeat_first(service, request, index):
            parts = urlsplit(request.url)
            alias_query = parts.query.replace("sort=", "%73ort=", 1)
            return H.page(locus, next="http://" + H.HOST.upper() + ":80"
                          + parts.path + "?" + alias_query)

        service = H.FakeAntares(script=repeat_first)
        observed, error = self.run_guarded(service)
        self.assertIs(type(error), L.P2ContinuationCycleError)
        self.assertEqual((len(observed), len(service.calls)), (1, 1))

    def test_failure_after_an_upgraded_page_remains_incomplete_and_unsealable(self):
        # Exercise the established whole-night fail-closed assertions, including
        # deterministic journal replay, discarded rows, no fetch, and no science.
        first = H.tile_loci(H.FIRST_TILE, 25)
        H.FailClosedMatrixTests.check(
            self, "r2-http-upgrade-then-empty-fragment", {
                None: H.page(first[:10], next="http://" + H.HOST + "/v1/loci?p=2"),
                "2": H.page(first[10:20], next=H.LISTING + "?p=3#"),
            }, H.GUARD + "P2MalformedContinuationError", "p2_transport_guard", 20, 1, 2)


if __name__ == "__main__":
    unittest.main()
