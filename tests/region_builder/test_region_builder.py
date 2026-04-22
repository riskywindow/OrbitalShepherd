from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from orbital_shepherd_contracts import load_json, validate_region_bundle, validate_region_manifest
from orbital_shepherd_core import canonical_json_dumps
from orbital_shepherd_geo_artifacts import export_region_bundle
from orbital_shepherd_region_builder import RegionBuilderConfig, compile_manifest_to_bundle

FIXTURE_MANIFEST_PATH = Path("data/regions/manifests/fixture_micro_region.json")
FIXTURE_BUNDLE_PATH = Path("data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json")


def test_fixture_manifest_and_bundle_validate() -> None:
    validate_region_manifest(load_json(FIXTURE_MANIFEST_PATH))
    validate_region_bundle(load_json(FIXTURE_BUNDLE_PATH))


def test_fixture_bundle_fingerprint_is_deterministic() -> None:
    manifest = load_json(FIXTURE_MANIFEST_PATH)

    first = compile_manifest_to_bundle(manifest, config=RegionBuilderConfig())
    second = compile_manifest_to_bundle(manifest, config=RegionBuilderConfig())

    assert first.bundle_fingerprint == second.bundle_fingerprint
    assert canonical_json_dumps(
        first.model_dump(mode="json", exclude_none=True)
    ) == canonical_json_dumps(
        second.model_dump(mode="json", exclude_none=True),
    )


def test_fixture_bundle_counts_are_stable() -> None:
    bundle = compile_manifest_to_bundle(
        load_json(FIXTURE_MANIFEST_PATH),
        config=RegionBuilderConfig(),
    )

    assert bundle.traversable_node_count == 8
    assert bundle.traversable_edge_count == 12
    assert len(bundle.road_nodes) == 8
    assert len(bundle.road_edges) == 12


def test_h3_cover_generation_is_deterministic() -> None:
    manifest = load_json(FIXTURE_MANIFEST_PATH)
    manifest["h3_cover"]["explicit_cell_ids"] = [
        "8928d54250bffff",
        "8928308280fffff",
        "8928341aec3ffff",
        "8928308280fffff",
    ]

    bundle = compile_manifest_to_bundle(manifest, config=RegionBuilderConfig())

    assert bundle.h3_cover.cell_ids == [
        "8928308280fffff",
        "8928341aec3ffff",
        "8928d54250bffff",
    ]
    assert bundle.h3_cover.cell_count == 3


def test_checked_in_fixture_bundle_matches_recompile() -> None:
    expected = load_json(FIXTURE_BUNDLE_PATH)
    actual = compile_manifest_to_bundle(
        load_json(FIXTURE_MANIFEST_PATH),
        config=RegionBuilderConfig(),
    )

    assert actual.bundle_fingerprint == expected["bundle_fingerprint"]
    assert canonical_json_dumps(
        actual.model_dump(mode="json", exclude_none=True)
    ) == canonical_json_dumps(
        expected,
    )


def test_region_exports_materialize_geojson_and_geoparquet(tmp_path: Path) -> None:
    bundle = compile_manifest_to_bundle(
        load_json(FIXTURE_MANIFEST_PATH),
        config=RegionBuilderConfig(),
    )

    outputs = export_region_bundle(bundle, output_dir=tmp_path)

    assert outputs["geojson:road_nodes"].exists()
    assert outputs["geojson:road_edges"].exists()
    assert outputs["geoparquet:road_nodes"].exists()
    assert outputs["geoparquet:road_edges"].exists()

    road_nodes_geojson = json.loads(outputs["geojson:road_nodes"].read_text(encoding="utf-8"))
    road_edges_geojson = json.loads(outputs["geojson:road_edges"].read_text(encoding="utf-8"))
    assert len(road_nodes_geojson["features"]) == 8
    assert len(road_edges_geojson["features"]) == 12

    road_nodes_table = pq.read_table(outputs["geoparquet:road_nodes"])
    road_edges_table = pq.read_table(outputs["geoparquet:road_edges"])
    assert road_nodes_table.num_rows == 8
    assert road_edges_table.num_rows == 12
