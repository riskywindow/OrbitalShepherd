from __future__ import annotations

from datetime import UTC, datetime

from orbital_shepherd_contracts import (
    bundle_id_from_manifest_id,
    compile_scenario_bundle,
    load_json,
    load_ndjson,
    normalize_replay_events,
    phase0_examples_dir,
    phase1_examples_dir,
    to_canonical_manifest,
    validate_all_contract_examples,
)


def test_validate_all_contract_examples() -> None:
    results = validate_all_contract_examples()

    assert {result.path.name for result in results} >= {
        "sample_scenario.json",
        "sample_replay.ndjson",
        "phase1_scenario_manifest.json",
        "phase1_scenario_bundle.json",
        "phase1_replay.ndjson",
    }


def test_legacy_sample_scenario_compiles_to_canonical_bundle() -> None:
    legacy_document = load_json(phase0_examples_dir() / "sample_scenario.json")

    manifest = to_canonical_manifest(legacy_document)
    bundle = compile_scenario_bundle(
        manifest,
        compiled_at=datetime(2026, 4, 9, 12, 0, tzinfo=UTC),
    )

    assert manifest.manifest_id == "sm:osbench-v01:cloud-trap:seed-42"
    assert bundle.bundle_id == bundle_id_from_manifest_id(manifest.manifest_id)
    assert bundle.compilation.source_manifest_id == manifest.manifest_id
    assert bundle.compilation.compiler_version == "orbital-shepherd-contracts/1.0.0"


def test_legacy_replay_is_normalized_to_phase1_event_names() -> None:
    legacy_events = load_ndjson(phase0_examples_dir() / "sample_replay.ndjson")

    normalized_events = normalize_replay_events(legacy_events)

    assert normalized_events[0].event_type == "scenario_bundle_loaded"
    assert normalized_events[2].event_type == "candidate_set_materialized"
    assert normalized_events[4].event_type == "observation_executed"
    assert normalized_events[0].payload["manifest_id"] == "sm:osbench-v01:cloud-trap:seed-42"


def test_phase1_examples_use_canonical_event_names() -> None:
    canonical_events = load_ndjson(phase1_examples_dir() / "phase1_replay.ndjson")

    assert {event["event_type"] for event in canonical_events} >= {
        "scenario_bundle_loaded",
        "candidate_set_materialized",
        "observation_executed",
        "downlink_executed",
        "incident_packet_emitted",
    }
