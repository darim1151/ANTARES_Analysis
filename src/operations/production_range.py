"""Control-bound range work and one-shot publication derivation.

Planning is offline. Execution requires an explicit switch, an exact persisted
authorization, and a secret token file. This module never provisions roots.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Mapping

from .backfill import (
    BackfillController, BackfillRefused, BackfillSettings, RangePublicationAuthorization,
    _canonical, _dates, _previous, _utc_now, acquisition_identity,
    require_prior_free_acquisition, inspect_backfill,
    RANGE_PRIOR_FREE_ACQUISITION_ATTESTATIONS,
)
from .live_antares import (
    LIVE_ANTARES_READ, LiveAntaresReadCapability, LiveAntaresProvider,
    _LIVE_READ_TOKEN, _real_directory, _scientific_query_contract, night_mjd_interval,
)
from .publication import (
    AuthorityState, NightPublisher, ProductionPublicationBinding, PublicationRefused,
    _is_hex64, _parse_utc, _read_json, _write_json_new, classify_night_authority,
    issue_production_publication_capability, production_authority_lock_identity,
    production_binding_from_sentinel, summary_source_manifest,
)
from .science import NightScienceRequest
from .production_canary import _verify_release
from .storage import (
    PublicationRoots, RangeWorkCapability, RANGE_WORK_PARENT, PRODUCTION_AUTHORITY_ROOT,
    PRODUCTION_DATA_ROOT, PRODUCTION_STAGE_ROOT, PRODUCTION_CONTROL_ROOT,
    PRODUCTION_EVIDENCE_ROOT,
)

SCHEMA = "v3.control-production-range.v1"
SENTINEL_CACHE = PRODUCTION_AUTHORITY_ROOT / "cache"
SEGMENT_CACHE = PRODUCTION_AUTHORITY_ROOT / "work" / "segment-cache"
MAX_PRODUCTION_RANGE_NIGHTS = 31


def implementation_identity():
    directory = Path(__file__).parent.parent
    # Bind delegated normalization, validation, authority and writer code as
    # well as the provider itself. Installed wheels contain this same source.
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*.py")) if "__pycache__" not in path.parts}


@dataclasses.dataclass(frozen=True)
class ProductionRangeAuthorization:
    """No write capability: exact Control range, work namespace and code binding."""

    scope: RangePublicationAuthorization
    work_root: str
    hostname: str
    service_uid: int
    control_token_sha256: str
    implementation_sha256: Mapping[str, str]
    publisher_wheel_sha256: str
    segment_size: int = 256
    schema_version: str = SCHEMA

    def __post_init__(self):
        dates = _dates(self.scope.start_date_utc, self.scope.end_date_utc)
        root = Path(self.work_root)
        if (self.schema_version != SCHEMA or len(dates) > MAX_PRODUCTION_RANGE_NIGHTS
                or root.parent != RANGE_WORK_PARENT or root.name in {".", ".."}
                or str(root) != self.work_root
                or not isinstance(self.hostname, str) or not self.hostname.startswith("arnor.")
                or type(self.service_uid) is not int or self.service_uid <= 0
                or not _is_hex64(self.control_token_sha256)
                or not _is_hex64(self.publisher_wheel_sha256)
                or self.scope.cache_root is not None
                or self.scope.initial_sentinel["canonical_root"] != str(PRODUCTION_DATA_ROOT)
                or dict(self.implementation_sha256) != implementation_identity()):
            raise BackfillRefused("Production range authorization has unsafe roots, identity or semantics.")
        BackfillSettings(segment_size=self.segment_size)
        adapter = LiveRangeAdapter(self.work_root, self.scope.candidate_release_sha, None)
        identity = acquisition_identity(adapter)
        science = hashlib.sha256(_canonical([
            adapter.scientific_contract(adapter.acquisition_request(day)) for day in dates])).hexdigest()
        configuration = hashlib.sha256(_canonical({
            "release_sha": self.scope.candidate_release_sha, "provider": adapter.provider_name,
            "scenario": adapter.scenario, "execution_policy": dict(adapter.execution_policy()),
            "segment_size": self.segment_size})).hexdigest()
        attestation = require_prior_free_acquisition(identity, RANGE_PRIOR_FREE_ACQUISITION_ATTESTATIONS,
                            science_contract_sha256=science, configuration_sha256=configuration)
        if (self.scope.acquisition_identity != identity or self.scope.science_contract_sha256 != science
                or self.scope.configuration_sha256 != configuration
                or self.scope.prior_free_attestation_sha256 != hashlib.sha256(_canonical(dict(attestation))).hexdigest()):
            raise BackfillRefused("Production range does not bind the supported live science/implementation/configuration.")

    def as_dict(self):
        return {"schema_version": self.schema_version, "scope": self.scope.as_dict(),
                "work_root": self.work_root, "hostname": self.hostname,
                "service_uid": self.service_uid, "control_token_sha256": self.control_token_sha256,
                "implementation_sha256": dict(self.implementation_sha256), "segment_size": self.segment_size,
                "publisher_wheel_sha256": self.publisher_wheel_sha256}

    @property
    def digest(self):
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()

    @classmethod
    def from_dict(cls, document):
        data = dict(document)
        data["scope"] = RangePublicationAuthorization(**data["scope"])
        return cls(**data)

    def verify(self, *, control_token=None, hostname=None, uid=None, allow_expired=False):
        # Repeat code and authority checks at every acquisition/issuance boundary.
        self.__post_init__()
        if (hostname or socket.getfqdn()) != self.hostname or (os.geteuid() if uid is None else uid) != self.service_uid:
            raise BackfillRefused("Production range belongs to another host/service UID.")
        if not allow_expired and _utc_now() >= _parse_utc(self.scope.expires_at_utc, "expires_at_utc"):
            raise BackfillRefused("Production range authorization expired.")
        if control_token is not None and (not _is_hex64(control_token) or not hmac.compare_digest(
                hashlib.sha256(control_token.encode("ascii")).hexdigest(), self.control_token_sha256)):
            raise BackfillRefused("Production range Control token differs.")
        _verify_release(self.scope.publisher_release_sha, self.publisher_wheel_sha256)
        if Path(sys.prefix).resolve() not in Path(__file__).resolve().parents:
            raise BackfillRefused("Range module must load from the exact immutable release venv.")


def range_read_capability(authority, run_root, run_id, night, release_sha, *, control_token):
    if type(authority) is not ProductionRangeAuthorization or control_token is None:
        raise BackfillRefused("Live range reads require explicit Control authority and token.")
    authority.verify(control_token=control_token)
    expected = Path(authority.work_root) / "nights" / f"night-{night}"
    if (night not in _dates(authority.scope.start_date_utc, authority.scope.end_date_utc)
            or Path(run_root) != expected or expected.resolve(strict=True) != expected
            or run_id != expected.name or release_sha != authority.scope.candidate_release_sha):
        raise BackfillRefused("Live read is outside its exact authorized night/work namespace.")
    return LiveAntaresReadCapability(_real_directory(expected, "Range night"), run_id,
                                    night, release_sha, "arnor-commissioning", _LIVE_READ_TOKEN)


class LiveRangeAdapter:
    """Per-night provider instances; the accepted extractor remains sequential."""

    provider_name = LiveAntaresProvider.provider_name
    scenario = LiveAntaresProvider.scenario
    provider_module = "src.operations.live_antares"

    def __init__(self, work_root, release_sha, read_factory, *, provider_factory=LiveAntaresProvider):
        self.work_root = Path(work_root)
        self.release_sha = release_sha
        self.read_factory = read_factory
        self.provider_factory = provider_factory

    def acquisition_request(self, night):
        minimum, maximum = night_mjd_interval(night)
        return NightScienceRequest(night, minimum, maximum, target_loci=None,
                                   range_label=f"ANTARES LSST {night}")

    def scientific_contract(self, request):
        if request != dataclasses.replace(self.acquisition_request(request.date_utc), prior_locus_ids=request.prior_locus_ids):
            raise BackfillRefused("Live range request differs from the exact UTC scientific contract.")
        return _scientific_query_contract(request)

    def execution_policy(self):
        # Same bounded defaults as the existing live provider, without issuing a read.
        provider = object.__new__(LiveAntaresProvider)
        provider.max_query_attempts = 2
        provider.max_fetch_attempts = 3
        provider.max_fetch_workers = 4
        provider.retry_delay_seconds = 0.5
        return provider.execution_policy()

    def _provider(self, request):
        run_id = f"night-{request.date_utc}"
        root = self.work_root / "nights" / run_id
        capability = self.read_factory(root, run_id, request.date_utc, self.release_sha)
        if capability.environment != "local-mock" and self.provider_factory is not LiveAntaresProvider:
            raise BackfillRefused("Provider injection is restricted to local qualification.")
        provider = self.provider_factory(capability)
        if provider.execution_policy() != self.execution_policy():
            raise BackfillRefused("Live range execution configuration drifted.")
        return provider

    def query_resumable(self, request, bindings, *, event_hook=None):
        return self._provider(request).query_resumable(request, bindings, event_hook=event_hook)

    def fetch_segment(self, request, locus_ids):
        return self._provider(request).fetch_segment(request, locus_ids)

    def construct_checkpoint(self, request, query_result, checkpoint):
        # Existing fetch_resumable reopens all segments, prepares and validates;
        # a complete checkpoint calls no service callback.
        root = self.work_root / "nights" / f"night-{request.date_utc}"
        # Construction needs no new live authority. Use an internal provider
        # carrying the same previously authorized night identity; no client is loaded.
        capability = LiveAntaresReadCapability(root, root.name, request.date_utc,
                    self.release_sha, "arnor-commissioning", _LIVE_READ_TOKEN)
        provider = LiveAntaresProvider(capability)
        return provider.fetch_resumable(request, query_result, checkpoint)


def qualify_range_night(authority, binding, authorization, candidate, *, resume=False):
    if type(authority) is not ProductionRangeAuthorization:
        raise BackfillRefused("Exact production range authority required.")
    scope = authority.scope
    expected = Path(authority.work_root) / "nights" / f"night-{binding.night_utc}"
    if (binding.range_authorization_sha256 != authority.digest
            or binding.night_utc not in _dates(scope.start_date_utc, scope.end_date_utc)
            or binding.candidate_root != str(expected)
            or expected.resolve(strict=True) != expected
            or authorization.authorized_by != f"range:{scope.digest}"
            or binding.control_token_sha256 != authority.control_token_sha256
            or binding.hostname != authority.hostname or binding.service_uid != authority.service_uid
            or binding.candidate_release_sha != scope.candidate_release_sha
            or binding.publisher_release_sha != scope.publisher_release_sha
            or binding.sentinel_cache_path != str(SENTINEL_CACHE)
            or binding.segment_cache_root != str(SEGMENT_CACHE)
            or binding.expires_at_utc != scope.expires_at_utc):
        raise BackfillRefused("One-shot night does not bind its exact range/candidate/roots.")
    expected_contract = _scientific_query_contract(NightScienceRequest(
        binding.night_utc, *night_mjd_interval(binding.night_utc), target_loci=None))
    if (candidate.provenance.get("range_authorization_sha256") != scope.digest
            or candidate.provenance.get("configuration_sha256") != scope.configuration_sha256
            or candidate.provenance.get("night_query_contract_sha256") != hashlib.sha256(_canonical(expected_contract)).hexdigest()):
        raise BackfillRefused("Candidate acquisition evidence belongs to another range/scientific request.")
    authority.__post_init__()
    # Even a later completed acquisition cannot bypass a blocked predecessor.
    predecessor = classify_night_authority(PRODUCTION_DATA_ROOT,
                          PRODUCTION_CONTROL_ROOT / "journals", binding.predecessor_night_utc)
    if predecessor["state"] != AuthorityState.COMPLETE.value or not predecessor.get("finalized"):
        raise PublicationRefused("predecessor_gap", "Range predecessor is not verified COMPLETE.")
    if not resume:
        fingerprint = predecessor.get("resulting_production_fingerprint")
        if fingerprint != binding.sentinel_fingerprint_sha256:
            raise PublicationRefused("sentinel_drift", "Range predecessor/Sentinel fingerprint differs.")
        if binding.night_utc == scope.start_date_utc:
            initial = scope.initial_sentinel
            if (binding.sentinel_fingerprint_sha256 != initial["durable_fingerprint_sha256"]
                    or binding.manifest_count != initial["manifest_count"]
                    or dict(binding.mount_binding) != initial["mount_binding"]
                    or dict(binding.cumulative_baseline_sha256) != initial["cumulative_sha256"]):
                raise PublicationRefused("sentinel_drift", "Range initial Sentinel differs.")


class ProductionRangePublisher:
    """Read-only planning plus a freshly derived single-night publisher per call."""

    roots = PublicationRoots()
    capability = roots  # Observation compatibility, never consumed by a writer.

    def __init__(self, authority, *, control_token):
        self.authority = authority
        self.control_token = control_token
        self.publisher_release_sha = authority.scope.publisher_release_sha
        self.data_root = self.roots.published_root
        self.cache_root = SENTINEL_CACHE
        self.mountinfo_lines = None
        self.clock = _utc_now

    sentinel = NightPublisher.sentinel
    authorization_inputs = NightPublisher.authorization_inputs
    authority_state = NightPublisher.authority_state
    _journals = NightPublisher._journals
    pending_authorization = NightPublisher.pending_authorization

    def publish(self, candidate, authorization, *, lock_wait_seconds=120.0):
        self.authority.verify(control_token=self.control_token, allow_expired=self.control_token is None)
        directory = Path(self.authority.work_root) / "production-bindings"
        path = directory / f"{candidate.date_utc}-{authorization.digest}.json"
        sentinel_path = path.with_suffix(".sentinel.json")
        if path.exists():
            binding = ProductionPublicationBinding(**_read_json(path))
        else:
            if self.control_token is None:
                raise BackfillRefused("Recovery requires an existing exact production binding.")
            inputs = self.authorization_inputs(candidate)
            production = inputs["production"]
            binding = ProductionPublicationBinding(
                hostname=self.authority.hostname, service_uid=self.authority.service_uid,
                production_root=str(self.data_root), stage_root=str(PRODUCTION_STAGE_ROOT),
                control_root=str(PRODUCTION_CONTROL_ROOT), evidence_root=str(PRODUCTION_EVIDENCE_ROOT),
                sentinel_cache_path=str(SENTINEL_CACHE), segment_cache_root=str(SEGMENT_CACHE),
                mount_binding=production["mount_binding"],
                sentinel_fingerprint_sha256=production["durable_fingerprint_sha256"],
                manifest_count=production["manifest_count"], authority_lock=production_authority_lock_identity(),
                night_utc=candidate.date_utc, predecessor_night_utc=authorization.predecessor_date_utc,
                candidate_root=str(candidate.candidate_dir.parent),
                candidate_record_sha256=candidate.record_sha256,
                candidate_binding_sha256=candidate.provenance["binding_sha256"],
                candidate_provenance_sha256=candidate.provenance_sha256,
                artifact_sha256=candidate.artifact_sha256(), publisher_release_sha=self.publisher_release_sha,
                candidate_release_sha=candidate.release_sha,
                cumulative_baseline_sha256=production["cumulative_sha256"],
                expected_cumulative_sha256=inputs["expected_cumulative_sha256"],
                expected_cumulative_schema_sha256={key: hashlib.sha256(_canonical(value)).hexdigest()
                    for key, value in inputs["cumulative_plan"]["arrow_schemas"].items()},
                authorization_sha256=authorization.digest, control_token_sha256=self.authority.control_token_sha256,
                nonce=authorization.nonce, expires_at_utc=authorization.expires_at_utc,
                range_authorization_sha256=self.authority.digest,
            )
            directory.mkdir(mode=0o700, exist_ok=True)
            if not sentinel_path.exists():
                _write_json_new(sentinel_path, inputs["sentinel"])
            _write_json_new(path, binding.as_dict())
        from .publication import _matching_resume_evidence
        recovering = _matching_resume_evidence(binding, authorization)
        capability = issue_production_publication_capability(
            binding, authorization, candidate, control_token=None if recovering else self.control_token,
            sentinel=_read_json(sentinel_path), range_authorization=self.authority,
        )
        return NightPublisher(capability, publisher_release_sha=self.publisher_release_sha,
                              cache_root=self.cache_root, mountinfo_lines=self.mountinfo_lines,
                              clock=self.clock).publish(candidate, authorization,
                                            lock_wait_seconds=lock_wait_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "inspect", "execute", "recover"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--candidate-release", required=True)
    parser.add_argument("--authorization")
    parser.add_argument("--publisher-wheel-sha256")
    parser.add_argument("--control-token-file")
    parser.add_argument("--execute-authorized-range", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--acquisition-concurrency", type=int, default=2)
    args = parser.parse_args(argv)
    settings = BackfillSettings(acquisition_concurrency=args.acquisition_concurrency)
    dates = _dates(args.start, args.end)
    if len(dates) > MAX_PRODUCTION_RANGE_NIGHTS or Path(args.work_root).parent != RANGE_WORK_PARENT:
        raise BackfillRefused("Operator range/work root is outside production bounds.")
    adapter = LiveRangeAdapter(args.work_root, args.candidate_release, None)
    if args.operation == "plan":
        science = hashlib.sha256(_canonical([adapter.scientific_contract(adapter.acquisition_request(day)) for day in dates])).hexdigest()
        configuration = hashlib.sha256(_canonical({"release_sha": args.candidate_release,
            "provider": adapter.provider_name, "scenario": adapter.scenario,
            "execution_policy": dict(adapter.execution_policy()), "segment_size": settings.segment_size})).hexdigest()
        identity = acquisition_identity(adapter)
        attestation = require_prior_free_acquisition(identity, RANGE_PRIOR_FREE_ACQUISITION_ATTESTATIONS,
                        science_contract_sha256=science, configuration_sha256=configuration)
        document = {"execution": "NOT EXECUTED", "nights": [dataclasses.asdict(adapter.acquisition_request(day)) for day in dates],
                    "work_root": args.work_root, "publication_concurrency": 1,
                    "implementation_sha256": implementation_identity(), "acquisition_identity": identity,
                    "range_binding": {"science_contract_sha256": science, "configuration_sha256": configuration,
                        "acquisition_identity": identity, "prior_free_attestation_sha256": hashlib.sha256(_canonical(dict(attestation))).hexdigest(),
                        "cache_root": None, "publication_concurrency": 1},
                    "requires": "Control authorization with initial COMPLETE predecessor/Sentinel, exact range binding and token"}
    elif args.operation == "inspect":
        document = inspect_backfill(Path(args.work_root), PRODUCTION_DATA_ROOT, args.start, args.end,
                    work_root=Path(args.work_root), journal_root=PRODUCTION_CONTROL_ROOT / "journals",
                    evidence_root=PRODUCTION_EVIDENCE_ROOT)
    else:
        recovery_only = args.operation == "recover"
        if (not args.execute_authorized_range or not args.authorization or not args.publisher_wheel_sha256
                or (not recovery_only and not args.control_token_file)
                or (recovery_only and (not args.resume or args.control_token_file))):
            raise BackfillRefused("Execution requires explicit --execute-authorized-range, authorization and Control token file.")
        authority = ProductionRangeAuthorization.from_dict(_read_json(Path(args.authorization)))
        if args.publisher_wheel_sha256 != authority.publisher_wheel_sha256:
            raise BackfillRefused("Operator wheel identity differs from Control authorization.")
        token = Path(args.control_token_file).read_text().strip() if args.control_token_file else None
        authority.verify(control_token=token, allow_expired=recovery_only)
        if args.work_root != authority.work_root or args.candidate_release != authority.scope.candidate_release_sha:
            raise BackfillRefused("Operator arguments differ from Control authorization.")
        read_factory = lambda root, run_id, night, release: range_read_capability(
            authority, root, run_id, night, release, control_token=token)
        adapter.read_factory = None if recovery_only else read_factory
        work = RangeWorkCapability.for_arnor(Path(args.work_root), Path(args.work_root).name)
        settings = dataclasses.replace(settings, segment_size=authority.segment_size)
        controller = BackfillController(None, adapter, release_sha=args.candidate_release,
            work_capability=work, publication_roots=PublicationRoots(), read_capability_factory=None if recovery_only else read_factory,
            settings=settings, publisher=ProductionRangePublisher(authority, control_token=token),
            range_authorization=authority.scope)
        document = controller.run(args.start, args.end, resume=args.resume, recovery_only=recovery_only)
    print(json.dumps(document, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
