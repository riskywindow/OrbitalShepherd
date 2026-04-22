from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from orbital_shepherd_benchmark.config import BenchmarkConfig
from orbital_shepherd_benchmark.planners import planner_descriptions
from orbital_shepherd_benchmark.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbital-shepherd-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-planners", help="List available benchmark planners.")
    list_parser.set_defaults(handler=_handle_list_planners)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one or more planners over a local Phase 1 scenario pack.",
    )
    run_parser.add_argument(
        "--planner",
        action="append",
        dest="planners",
        default=None,
        help="Planner id to run. Repeat to run multiple planners.",
    )
    run_parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=BenchmarkConfig().scenario_dir,
        help="Directory containing compiled scenario bundles.",
    )
    run_parser.add_argument(
        "--report-dir",
        type=Path,
        default=BenchmarkConfig().report_dir,
        help="Directory to write benchmark summaries.",
    )
    run_parser.add_argument(
        "--replay-dir",
        type=Path,
        default=BenchmarkConfig().replay_dir,
        help="Directory to write episode replay artifacts.",
    )
    run_parser.add_argument(
        "--family",
        action="append",
        dest="families",
        default=[],
        help="Scenario family filter. Repeat to include multiple families.",
    )
    run_parser.add_argument(
        "--bundle-id",
        action="append",
        dest="bundle_ids",
        default=[],
        help="Exact bundle id filter. Repeat to include multiple bundles.",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of matching bundles to run per planner.",
    )
    run_parser.add_argument(
        "--run-id",
        default=None,
        help="Stable artifact folder name for the benchmark run.",
    )
    run_parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip planner_summary.md output.",
    )
    run_parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip planner_summary.csv output.",
    )
    run_parser.set_defaults(handler=_handle_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_list_planners(_: argparse.Namespace) -> int:
    for planner_id, description in sorted(planner_descriptions().items()):
        print(f"{planner_id}\t{description}")
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    config = BenchmarkConfig(
        scenario_dir=args.scenario_dir,
        replay_dir=args.replay_dir,
        report_dir=args.report_dir,
        planner_ids=tuple(args.planners or BenchmarkConfig().planner_ids),
        write_markdown=not args.no_markdown,
        write_csv=not args.no_csv,
    )
    result = run_benchmark(
        config=config,
        planner_ids=args.planners,
        scenario_families=args.families,
        bundle_ids=args.bundle_ids,
        limit=args.limit,
        run_id=args.run_id,
    )
    print(result.report_dir / "summary.json")
    print(result.replay_dir)
    for planner_id in result.planners:
        summary = result.planner_summaries[planner_id]
        print(
            f"{planner_id} mission_utility_mean={summary['mission_utility_mean']:.6f} "
            f"uovc_mean={summary['useful_observation_value_captured_mean']:.6f} "
            f"missed_urgent_mean={summary['missed_urgent_incident_rate_mean']:.6f}"
        )
    return 0
