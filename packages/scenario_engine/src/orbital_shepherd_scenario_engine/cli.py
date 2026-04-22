from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from orbital_shepherd_contracts import load_json
from orbital_shepherd_scenario_engine.catalog import builtin_phase2_recipes
from orbital_shepherd_scenario_engine.compiler import (
    build_scenario_pack,
    compile_manifest_to_bundle,
    validate_scenario_pack,
)
from orbital_shepherd_scenario_engine.config import ScenarioEngineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbital-shepherd-scenario-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-scenario-pack",
        help="Build the deterministic Phase 1 wildfire scenario pack under data/scenarios.",
    )
    build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ScenarioEngineConfig().scenario_dir,
        help="Destination directory for compiled scenario bundles.",
    )
    build_parser.add_argument(
        "--recipe-set",
        choices=("phase1", "phase2"),
        default="phase1",
        help="Built-in deterministic recipe set to compile.",
    )
    build_parser.set_defaults(handler=_handle_build_scenario_pack)

    validate_parser = subparsers.add_parser(
        "validate-scenario-pack",
        help=(
            "Validate schema, fingerprint, and deterministic rebuild equivalence "
            "for the scenario pack."
        ),
    )
    validate_parser.add_argument(
        "--input-dir",
        type=Path,
        default=ScenarioEngineConfig().scenario_dir,
        help="Directory containing compiled scenario bundles.",
    )
    validate_parser.add_argument(
        "--recipe-set",
        choices=("phase1", "phase2"),
        default="phase1",
        help="Built-in deterministic recipe set expected in the pack.",
    )
    validate_parser.set_defaults(handler=_handle_validate_scenario_pack)

    compile_parser = subparsers.add_parser(
        "compile-manifest",
        help=(
            "Compile a canonical ScenarioManifest into a deterministic "
            "ScenarioBundle using the Phase 1 engine."
        ),
    )
    compile_parser.add_argument("input", type=Path)
    compile_parser.add_argument("output", type=Path)
    compile_parser.set_defaults(handler=_handle_compile_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_build_scenario_pack(args: argparse.Namespace) -> int:
    if args.recipe_set == "phase2":
        benchmark_id = "osbench-phase2-foundation-v1"
        recipes = builtin_phase2_recipes(benchmark_id)
    else:
        benchmark_id = ScenarioEngineConfig().benchmark_id
        recipes = None
    records = build_scenario_pack(
        engine_config=ScenarioEngineConfig(
            benchmark_id=benchmark_id,
            scenario_dir=args.output_dir,
        ),
        output_dir=args.output_dir,
        recipes=recipes,
    )
    for record in records:
        print(f"{record.bundle_id} {record.scenario_family} {record.bundle_fingerprint}")
    print(f"built {len(records)} scenario bundle(s)")
    return 0


def _handle_validate_scenario_pack(args: argparse.Namespace) -> int:
    if args.recipe_set == "phase2":
        benchmark_id = "osbench-phase2-foundation-v1"
        recipes = builtin_phase2_recipes(benchmark_id)
    else:
        benchmark_id = ScenarioEngineConfig().benchmark_id
        recipes = None
    records = validate_scenario_pack(
        engine_config=ScenarioEngineConfig(
            benchmark_id=benchmark_id,
            scenario_dir=args.input_dir,
        ),
        input_dir=args.input_dir,
        recipes=recipes,
    )
    for record in records:
        print(f"validated {record.bundle_id} {record.scenario_family} {record.bundle_fingerprint}")
    print(f"validated {len(records)} scenario bundle(s)")
    return 0


def _handle_compile_manifest(args: argparse.Namespace) -> int:
    manifest_document = load_json(args.input)
    bundle = compile_manifest_to_bundle(manifest_document)
    args.output.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(args.output)
    return 0
