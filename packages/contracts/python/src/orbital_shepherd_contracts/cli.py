from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from orbital_shepherd_contracts.adapters import compile_scenario_bundle, to_canonical_manifest
from orbital_shepherd_contracts.validation import load_json, validate_all_contract_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbital-shepherd-contracts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate legacy and canonical contract examples and fixture artifacts."
    )
    validate_parser.set_defaults(handler=_handle_validate)

    compile_parser = subparsers.add_parser(
        "compile-manifest", help="Compile a scenario manifest into a canonical scenario bundle."
    )
    compile_parser.add_argument("input", type=Path)
    compile_parser.add_argument("output", type=Path)
    compile_parser.add_argument(
        "--compiled-at",
        default=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        help="RFC 3339 UTC timestamp for bundle compilation metadata.",
    )
    compile_parser.set_defaults(handler=_handle_compile_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_validate(_: argparse.Namespace) -> int:
    results = validate_all_contract_examples()
    for result in results:
        print(f"validated {result.artifact_type}: {result.path} ({result.details})")
    print(f"validated {len(results)} artifact(s)")
    return 0


def _handle_compile_manifest(args: argparse.Namespace) -> int:
    source_document = load_json(args.input)
    manifest = to_canonical_manifest(source_document)
    compiled_at = datetime.fromisoformat(args.compiled_at.replace("Z", "+00:00"))
    bundle = compile_scenario_bundle(manifest, compiled_at=compiled_at)
    args.output.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0
