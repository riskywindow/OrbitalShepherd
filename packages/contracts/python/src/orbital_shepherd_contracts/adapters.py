from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from orbital_shepherd_contracts.constants import (
    LEGACY_REPLAY_EVENT_ALIASES,
    LEGACY_REPLAY_PAYLOAD_FIELD_ALIASES,
    PHASE0_SCHEMA_VERSION,
    PHASE1_COMPILER_VERSION,
    PHASE1_SCHEMA_VERSION,
)
from orbital_shepherd_contracts.models import (
    DownlinkWindow,
    ObservationOpportunity,
    ReplayEvent,
    ScenarioBundle,
    ScenarioManifest,
)
from orbital_shepherd_core import canonical_json_dumps, format_utc_timestamp, sha256_fingerprint


def manifest_id_from_bundle_id(bundle_id: str) -> str:
    if bundle_id.startswith("sb:"):
        return f"sm:{bundle_id[3:]}"
    return f"sm:{bundle_id}"


def bundle_id_from_manifest_id(manifest_id: str) -> str:
    if manifest_id.startswith("sm:"):
        return f"sb:{manifest_id[3:]}"
    return f"sb:{manifest_id}"


def to_canonical_manifest(document: Mapping[str, Any]) -> ScenarioManifest:
    normalized = deepcopy(dict(document))
    if "manifest_id" not in normalized:
        bundle_id = str(normalized.get("bundle_id", "legacy-scenario"))
        normalized["manifest_id"] = manifest_id_from_bundle_id(bundle_id)
    normalized.pop("bundle_id", None)
    _rewrite_schema_versions(normalized)
    normalized["schema_version"] = PHASE1_SCHEMA_VERSION
    return ScenarioManifest.model_validate(normalized)


def compile_scenario_bundle(
    manifest: ScenarioManifest | Mapping[str, Any],
    *,
    compiled_at: datetime | None = None,
    compiler_version: str = PHASE1_COMPILER_VERSION,
    bundle_id: str | None = None,
    observation_opportunities: Sequence[Mapping[str, Any] | ObservationOpportunity] = (),
    downlink_windows: Sequence[Mapping[str, Any] | DownlinkWindow] = (),
) -> ScenarioBundle:
    canonical_manifest = (
        manifest if isinstance(manifest, ScenarioManifest) else to_canonical_manifest(manifest)
    )
    compiled_at_value = compiled_at or datetime.now(UTC)
    manifest_document = canonical_manifest.model_dump(mode="json")
    compiled_bundle = {
        **manifest_document,
        "bundle_id": bundle_id or bundle_id_from_manifest_id(canonical_manifest.manifest_id),
        "compilation": {
            "source_manifest_id": canonical_manifest.manifest_id,
            "source_manifest_schema_version": canonical_manifest.schema_version,
            "source_manifest_sha256": sha256_fingerprint(canonical_json_dumps(manifest_document)),
            "compiled_at_utc": format_utc_timestamp(compiled_at_value),
            "compiler_version": compiler_version,
        },
        "observation_opportunities": [
            _coerce_phase1_nested_contract("ObservationOpportunity", item)
            for item in observation_opportunities
        ],
        "downlink_windows": [
            _coerce_phase1_nested_contract("DownlinkWindow", item) for item in downlink_windows
        ],
    }
    compiled_bundle["bundle_fingerprint"] = scenario_bundle_fingerprint(compiled_bundle)
    return ScenarioBundle.model_validate(compiled_bundle)


def scenario_bundle_fingerprint(document: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(document))
    normalized.pop("bundle_fingerprint", None)
    return sha256_fingerprint(canonical_json_dumps(normalized))


def normalize_replay_event(event: Mapping[str, Any]) -> ReplayEvent:
    normalized = deepcopy(dict(event))
    normalized["schema_version"] = PHASE1_SCHEMA_VERSION
    event_type = str(normalized.get("event_type"))
    canonical_event_type = LEGACY_REPLAY_EVENT_ALIASES.get(event_type, event_type)
    normalized["event_type"] = canonical_event_type
    payload = deepcopy(dict(normalized.get("payload", {})))
    for legacy_name, canonical_name in LEGACY_REPLAY_PAYLOAD_FIELD_ALIASES.get(
        canonical_event_type, {}
    ).items():
        if legacy_name in payload and canonical_name not in payload:
            payload[canonical_name] = payload.pop(legacy_name)
    if canonical_event_type == "scenario_bundle_loaded" and "manifest_id" not in payload:
        bundle_id = payload.get("bundle_id")
        if isinstance(bundle_id, str):
            payload["manifest_id"] = manifest_id_from_bundle_id(bundle_id)
    normalized["payload"] = payload
    return ReplayEvent.model_validate(normalized)


def normalize_replay_events(events: Iterable[Mapping[str, Any]]) -> list[ReplayEvent]:
    normalized_events = [normalize_replay_event(event) for event in events]
    previous_event_index = -1
    previous_sim_tick = -1
    for event in normalized_events:
        if event.event_index <= previous_event_index:
            raise ValueError("event_index must be strictly increasing within a replay")
        if event.sim_tick < previous_sim_tick:
            raise ValueError("sim_tick must be monotonically non-decreasing within a replay")
        previous_event_index = event.event_index
        previous_sim_tick = event.sim_tick
    return normalized_events


def _rewrite_schema_versions(node: Any) -> Any:
    if isinstance(node, dict):
        if "schema_version" in node:
            node["schema_version"] = PHASE1_SCHEMA_VERSION
        for value in node.values():
            _rewrite_schema_versions(value)
    elif isinstance(node, list):
        for item in node:
            _rewrite_schema_versions(item)
    return node


def _coerce_phase1_nested_contract(kind: str, value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, ObservationOpportunity | DownlinkWindow):
        return value.model_dump(mode="json", exclude_none=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{kind} value must be a mapping or a Phase 1 model instance")
    normalized = deepcopy(dict(value))
    _rewrite_schema_versions(normalized)
    if normalized.get("schema_version") == PHASE0_SCHEMA_VERSION:
        normalized["schema_version"] = PHASE1_SCHEMA_VERSION
    return normalized
