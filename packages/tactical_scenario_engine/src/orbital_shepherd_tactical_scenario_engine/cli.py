from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from orbital_shepherd_contracts import load_json
from orbital_shepherd_core import canonical_json_dumps
from orbital_shepherd_tactical_scenario_engine.catalog import FAMILY_SPECS
from orbital_shepherd_tactical_scenario_engine.compiler import (
    build_scenario_pack,
    compile_incident_packet_to_activation,
    compile_incident_packet_to_bundle,
    compile_manifest_to_bundle,
    inspect_bundle,
    validate_scenario_pack,
)
from orbital_shepherd_tactical_scenario_engine.config import TacticalScenarioEngineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbital-shepherd-tactical-scenario-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-scenario-pack",
        help="Build the deterministic Phase 3 tactical scenario pack.",
    )
    build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=TacticalScenarioEngineConfig().scenario_dir,
    )
    build_parser.add_argument(
        "--region-bundle",
        type=Path,
        default=TacticalScenarioEngineConfig().default_region_bundle_path,
        help="Compiled RegionBundle used by the built-in tactical recipes.",
    )
    build_parser.set_defaults(handler=_handle_build_scenario_pack)

    validate_parser = subparsers.add_parser(
        "validate-scenario-pack",
        help=(
            "Validate schema, fingerprint, and deterministic rebuild equivalence "
            "for the tactical pack."
        ),
    )
    validate_parser.add_argument(
        "--input-dir",
        type=Path,
        default=TacticalScenarioEngineConfig().scenario_dir,
    )
    validate_parser.add_argument(
        "--region-bundle",
        type=Path,
        default=TacticalScenarioEngineConfig().default_region_bundle_path,
        help="Compiled RegionBundle used by the built-in tactical recipes.",
    )
    validate_parser.set_defaults(handler=_handle_validate_scenario_pack)

    activation_parser = subparsers.add_parser(
        "compile-activation",
        help=(
            "Compile an IncidentPacket into a TacticalActivation with deterministic "
            "region resolution."
        ),
    )
    activation_parser.add_argument("incident_packet", type=Path)
    activation_parser.add_argument("output", type=Path)
    activation_parser.add_argument("--region-bundle", type=Path, default=None)
    activation_parser.add_argument("--region-catalog", type=Path, action="append", default=[])
    activation_parser.add_argument(
        "--incident-geometry",
        type=Path,
        default=None,
        help="Optional TacticalIncidentGeometry JSON override.",
    )
    activation_parser.set_defaults(handler=_handle_compile_activation)

    manifest_parser = subparsers.add_parser(
        "compile-manifest",
        help="Compile a TacticalScenarioManifest into a TacticalScenarioBundle.",
    )
    manifest_parser.add_argument("manifest", type=Path)
    manifest_parser.add_argument("output", type=Path)
    manifest_parser.add_argument("--region-bundle", type=Path, default=None)
    manifest_parser.set_defaults(handler=_handle_compile_manifest)

    packet_parser = subparsers.add_parser(
        "compile-incident-packet",
        help=(
            "Compile an IncidentPacket directly into a TacticalScenarioBundle through "
            "TacticalActivation and TacticalScenarioManifest."
        ),
    )
    packet_parser.add_argument("incident_packet", type=Path)
    packet_parser.add_argument("output", type=Path)
    packet_parser.add_argument(
        "--scenario-family",
        required=True,
        choices=tuple(FAMILY_SPECS),
    )
    packet_parser.add_argument("--seed", type=int, required=True)
    packet_parser.add_argument("--region-bundle", type=Path, default=None)
    packet_parser.add_argument("--region-catalog", type=Path, action="append", default=[])
    packet_parser.add_argument(
        "--incident-geometry",
        type=Path,
        default=None,
        help="Optional TacticalIncidentGeometry JSON override.",
    )
    packet_parser.set_defaults(handler=_handle_compile_incident_packet)

    inspect_parser = subparsers.add_parser(
        "inspect-bundle",
        help="Inspect a compiled TacticalScenarioBundle and print a compact JSON summary.",
    )
    inspect_parser.add_argument("bundle", type=Path)
    inspect_parser.set_defaults(handler=_handle_inspect_bundle)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_build_scenario_pack(args: argparse.Namespace) -> int:
    config = TacticalScenarioEngineConfig(
        scenario_dir=args.output_dir,
        default_region_bundle_path=args.region_bundle,
        region_bundle_catalog=(args.region_bundle,),
    )
    records = build_scenario_pack(config=config, output_dir=args.output_dir)
    for record in records:
        print(
            f"{record.tactical_bundle_id} {record.scenario_family} {record.bundle_fingerprint}"
        )
    print(f"built {len(records)} tactical scenario bundle(s)")
    return 0


def _handle_validate_scenario_pack(args: argparse.Namespace) -> int:
    config = TacticalScenarioEngineConfig(
        scenario_dir=args.input_dir,
        default_region_bundle_path=args.region_bundle,
        region_bundle_catalog=(args.region_bundle,),
    )
    records = validate_scenario_pack(config=config, input_dir=args.input_dir)
    for record in records:
        print(
            f"validated {record.tactical_bundle_id} "
            f"{record.scenario_family} {record.bundle_fingerprint}"
        )
    print(f"validated {len(records)} tactical scenario bundle(s)")
    return 0


def _handle_compile_activation(args: argparse.Namespace) -> int:
    packet = load_json(args.incident_packet)
    activation = compile_incident_packet_to_activation(
        packet,
        region_bundle=args.region_bundle,
        catalog_sources=args.region_catalog,
        incident_geometry_override=_load_optional_json(args.incident_geometry),
    )
    args.output.write_text(
        canonical_json_dumps(activation.model_dump(mode="json", exclude_none=True)) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


def _handle_compile_manifest(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    bundle = compile_manifest_to_bundle(manifest, region_bundle=args.region_bundle)
    args.output.write_text(
        canonical_json_dumps(bundle.model_dump(mode="json", exclude_none=True)) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


def _handle_compile_incident_packet(args: argparse.Namespace) -> int:
    packet = load_json(args.incident_packet)
    bundle = compile_incident_packet_to_bundle(
        packet,
        scenario_family=args.scenario_family,
        simulation_seed=args.seed,
        region_bundle=args.region_bundle,
        catalog_sources=args.region_catalog,
        incident_geometry_override=_load_optional_json(args.incident_geometry),
    )
    args.output.write_text(
        canonical_json_dumps(bundle.model_dump(mode="json", exclude_none=True)) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


def _handle_inspect_bundle(args: argparse.Namespace) -> int:
    print(canonical_json_dumps(inspect_bundle(args.bundle)))
    return 0


def _load_optional_json(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return cast(dict[str, object], load_json(path))
