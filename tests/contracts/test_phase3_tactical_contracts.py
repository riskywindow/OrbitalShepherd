from __future__ import annotations

from orbital_shepherd_contracts import (
    CANONICAL_TACTICAL_REPLAY_EVENT_TYPES,
    load_json,
    load_ndjson,
    phase1_examples_dir,
    validate_all_contract_examples,
    validate_region_bundle,
    validate_region_manifest,
    validate_spatial_ingest_manifest,
    validate_tactical_activation,
    validate_tactical_metrics_summary,
    validate_tactical_replay_events,
    validate_tactical_scenario_bundle,
    validate_tactical_scenario_manifest,
)


def test_validate_all_contract_examples_includes_phase3_examples() -> None:
    results = validate_all_contract_examples()

    assert {result.path.name for result in results} >= {
        "phase3_spatial_ingest_manifest.json",
        "phase3_region_manifest.json",
        "phase3_region_bundle.json",
        "phase3_tactical_activation.json",
        "phase3_tactical_scenario_manifest.json",
        "phase3_tactical_scenario_bundle.json",
        "phase3_tactical_replay.ndjson",
        "phase3_tactical_metrics_summary.json",
    }


def test_phase3_bridge_preserves_incident_packet_handoff() -> None:
    activation = validate_tactical_activation(
        load_json(phase1_examples_dir() / "phase3_tactical_activation.json")
    )
    manifest = validate_tactical_scenario_manifest(
        load_json(phase1_examples_dir() / "phase3_tactical_scenario_manifest.json")
    )
    bundle = validate_tactical_scenario_bundle(
        load_json(phase1_examples_dir() / "phase3_tactical_scenario_bundle.json")
    )

    assert activation.incident_packet.packet_id == activation.incident_packet_id
    assert manifest.incident_packet.packet_id == activation.incident_packet_id
    assert bundle.incident_packet.packet_id == activation.incident_packet_id
    assert bundle.region_bundle_id == activation.region_bundle_id


def test_phase3_region_artifacts_validate_cleanly() -> None:
    spatial_ingest = validate_spatial_ingest_manifest(
        load_json(phase1_examples_dir() / "phase3_spatial_ingest_manifest.json")
    )
    region_manifest = validate_region_manifest(
        load_json(phase1_examples_dir() / "phase3_region_manifest.json")
    )
    region_bundle = validate_region_bundle(
        load_json(phase1_examples_dir() / "phase3_region_bundle.json")
    )

    assert region_manifest.spatial_ingest_ids == [spatial_ingest.spatial_ingest_id]
    assert region_bundle.region_manifest_id == region_manifest.region_manifest_id
    assert region_bundle.spatial_ingests[0].spatial_ingest_id == spatial_ingest.spatial_ingest_id


def test_phase3_tactical_replay_uses_canonical_taxonomy() -> None:
    tactical_replay = validate_tactical_replay_events(
        load_ndjson(phase1_examples_dir() / "phase3_tactical_replay.ndjson")
    )

    assert tuple(event.event_type for event in tactical_replay) == (
        CANONICAL_TACTICAL_REPLAY_EVENT_TYPES
    )


def test_phase3_tactical_metrics_summary_matches_bundle() -> None:
    bundle = validate_tactical_scenario_bundle(
        load_json(phase1_examples_dir() / "phase3_tactical_scenario_bundle.json")
    )
    summary = validate_tactical_metrics_summary(
        load_json(phase1_examples_dir() / "phase3_tactical_metrics_summary.json")
    )

    assert summary.tactical_bundle_id == bundle.tactical_bundle_id
    assert summary.incident_packet_id == bundle.incident_packet_id
    assert summary.arrived_unit_count <= summary.dispatched_unit_count
