from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from orbital_shepherd_core import canonical_json_dumps
from orbital_shepherd_ephemeris.celestrak import CelesTrakClient
from orbital_shepherd_ephemeris.paths import ephemeris_compiled_dir, ephemeris_raw_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbital-shepherd-ephemeris")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch-ephemeris",
        help="Fetch a CelesTrak GP/OMM snapshot or rewrap a local JSON payload into an immutable raw snapshot.",
    )
    fetch_parser.add_argument("--group", required=True, help="CelesTrak group name such as active or stations.")
    fetch_parser.add_argument(
        "--input",
        type=Path,
        help="Optional local JSON file containing a raw CelesTrak GP JSON array for fixture-backed use.",
    )
    fetch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ephemeris_raw_dir(),
        help="Directory where the immutable raw snapshot should be written.",
    )
    fetch_parser.add_argument(
        "--fetched-at",
        default=None,
        help="Optional UTC timestamp to use for the snapshot metadata. Defaults to now for live fetches.",
    )
    fetch_parser.set_defaults(handler=_handle_fetch_ephemeris)

    compile_parser = subparsers.add_parser(
        "compile-orbit-assets",
        help="Compile a raw CelesTrak snapshot into the normalized OrbitAssetBundle consumed by Phase 1.",
    )
    compile_parser.add_argument("input", type=Path, help="Path to a raw ephemeris snapshot JSON file.")
    compile_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path for the compiled orbit-asset bundle.",
    )
    compile_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ephemeris_compiled_dir(),
        help="Directory for compiled assets when --output is not supplied.",
    )
    compile_parser.set_defaults(handler=_handle_compile_orbit_assets)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_fetch_ephemeris(args: argparse.Namespace) -> int:
    client = CelesTrakClient()
    if args.input is not None:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--input must point to a JSON array of CelesTrak OMM/GP records")
        fetched_at = (
            datetime.fromisoformat(args.fetched_at.replace("Z", "+00:00"))
            if args.fetched_at
            else _infer_fixture_timestamp(payload)
        )
        snapshot = client.build_snapshot(
            payload,
            catalog_group=args.group,
            source_mode="fixture",
            fetched_at=fetched_at,
        )
    else:
        fetched_at = (
            datetime.fromisoformat(args.fetched_at.replace("Z", "+00:00"))
            if args.fetched_at
            else datetime.now(UTC)
        )
        snapshot = client.fetch_group(args.group, fetched_at=fetched_at)
    output_path = client.persist_raw_snapshot(snapshot, args.output_dir)
    print(output_path)
    return 0


def _handle_compile_orbit_assets(args: argparse.Namespace) -> int:
    client = CelesTrakClient()
    bundle = client.compile_orbit_assets(args.input, raw_snapshot_path=args.input)
    output_path = args.output or args.output_dir / f"{bundle.bundle_id.replace(':', '--')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json_dumps(bundle.model_dump(mode="json")) + "\n", encoding="utf-8")
    print(output_path)
    return 0


def _infer_fixture_timestamp(payload: list[object]) -> datetime:
    if not payload:
        raise ValueError("fixture payload must contain at least one record")
    first = payload[0]
    if not isinstance(first, dict):
        raise ValueError("fixture payload must contain JSON objects")
    epoch = first.get("EPOCH")
    if not isinstance(epoch, str):
        raise ValueError("fixture payload must include EPOCH fields or pass --fetched-at")
    return datetime.fromisoformat(epoch.replace("Z", "+00:00")).astimezone(UTC)
