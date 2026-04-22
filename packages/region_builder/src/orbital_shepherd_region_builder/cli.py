from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from orbital_shepherd_region_builder.compiler import compile_manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbital-shepherd-region-builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile-manifest",
        help="Compile a canonical RegionManifest into a deterministic RegionBundle.",
    )
    compile_parser.add_argument("input", type=Path)
    compile_parser.add_argument("output", type=Path)
    compile_parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional output directory for GeoJSON and GeoParquet layer exports.",
    )
    compile_parser.set_defaults(handler=_handle_compile_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_compile_manifest(args: argparse.Namespace) -> int:
    record = compile_manifest_path(
        args.input,
        output_path=args.output,
        export_dir=args.export_dir,
    )
    print(f"{record.bundle_id} {record.bundle_fingerprint} {record.bundle_path}")
    return 0
