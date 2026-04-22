from __future__ import annotations

import json
import tempfile
from pathlib import Path

from orbital_shepherd_ephemeris import CelesTrakClient
from orbital_shepherd_ephemeris.cli import main as ephemeris_main

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RECORDS_PATH = (
    REPO_ROOT / "tests/fixtures/ephemeris/celestrak_demo_records.json"
)
RAW_SNAPSHOT_PATH = (
    REPO_ROOT
    / "data/fixtures/ephemeris/raw/raw:celestrak:demo-phase1:2026-04-01t00-00-00z--f7955187d221.json"
)
RAW_SNAPSHOT_RELATIVE = Path(
    "data/fixtures/ephemeris/raw/raw:celestrak:demo-phase1:2026-04-01t00-00-00z--f7955187d221.json"
)
COMPILED_BUNDLE_PATH = (
    REPO_ROOT
    / "data/fixtures/ephemeris/compiled/eph--demo-phase1--raw-celestrak-demo-phase1-2026-04-01t00-00-00z.json"
)


def test_celestrak_snapshot_compiles_to_the_golden_orbit_asset_bundle() -> None:
    client = CelesTrakClient()

    snapshot = client.load_snapshot(RAW_SNAPSHOT_PATH)
    bundle = client.compile_orbit_assets(snapshot, raw_snapshot_path=RAW_SNAPSHOT_RELATIVE)
    golden = json.loads(COMPILED_BUNDLE_PATH.read_text(encoding="utf-8"))

    assert snapshot.raw_snapshot_sha256 == "f7955187d22190802a9a60cbc03396679eb65c6364a819d5a7000f9726b545b9"
    assert [asset.satellite_id for asset in bundle.assets] == [
        "sat:norad-25544:iss--zarya",
        "sat:norad-40697:sentinel-2a",
        "sat:norad-43013:noaa-20",
    ]
    assert [asset.asset_fingerprint for asset in bundle.assets] == [
        "196362fa8b3fb7c47bf5b1710d771d8d1e2ccfd280612731d853187cac4af4a2",
        "82af5f3ed335baedce6354d833d14fa69a2709b94f5e87ebd7fbee1957da3e48",
        "c0a033b1960ba15b40ea5b9e015fa82a8d91724943c8167dd4d818dc859cad9a",
    ]
    assert bundle.bundle_fingerprint == "b8c95b7c343afc77161d7163484ce33929a71906a9c1a015e830475249282564"
    assert bundle.model_dump(mode="json") == golden


def test_fetch_cli_creates_a_deterministic_raw_snapshot_from_fixture_records() -> None:
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
        output_dir = Path(temp_dir) / "raw"

        exit_code = ephemeris_main(
            [
                "fetch-ephemeris",
                "--group",
                "demo-phase1",
                "--input",
                str(SOURCE_RECORDS_PATH),
                "--output-dir",
                str(output_dir),
            ]
        )

        assert exit_code == 0
        produced_files = sorted(output_dir.glob("*.json"))
        assert [path.name for path in produced_files] == [RAW_SNAPSHOT_PATH.name]
        assert json.loads(produced_files[0].read_text(encoding="utf-8")) == json.loads(
            RAW_SNAPSHOT_PATH.read_text(encoding="utf-8")
        )


def test_compile_cli_reproduces_the_golden_bundle_from_the_raw_fixture() -> None:
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
        output_path = Path(temp_dir) / "bundle.json"

        exit_code = ephemeris_main(
            [
                "compile-orbit-assets",
                str(RAW_SNAPSHOT_RELATIVE),
                "--output",
                str(output_path),
            ]
        )

        assert exit_code == 0
        assert json.loads(output_path.read_text(encoding="utf-8")) == json.loads(
            COMPILED_BUNDLE_PATH.read_text(encoding="utf-8")
        )
