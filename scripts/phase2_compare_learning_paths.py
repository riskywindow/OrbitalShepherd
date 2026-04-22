from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import install_repo_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase2-compare-learning-paths",
        description="Summarize BC-only, PPO scratch, and BC->PPO run artifacts.",
    )
    parser.add_argument("--bc-run-id", required=True, help="Behavior cloning run id.")
    parser.add_argument(
        "--ppo-scratch-run-id",
        required=True,
        help="PPO scratch run id.",
    )
    parser.add_argument(
        "--ppo-warmstart-run-id",
        required=True,
        help="PPO BC-warm-start run id.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/training/manifests"),
        help="Root directory containing run manifests and checkpoint indices.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("data/training/reports"),
        help="Root directory containing per-run report directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the JSON summary to.",
    )
    return parser


def main() -> int:
    install_repo_sources()
    parser = build_parser()
    args = parser.parse_args()
    comparison = {
        "bc": _summarize_run(
            run_id=args.bc_run_id,
            manifest_root=args.manifest_root,
            report_root=args.report_root,
        ),
        "ppo_scratch": _summarize_run(
            run_id=args.ppo_scratch_run_id,
            manifest_root=args.manifest_root,
            report_root=args.report_root,
        ),
        "ppo_warmstart": _summarize_run(
            run_id=args.ppo_warmstart_run_id,
            manifest_root=args.manifest_root,
            report_root=args.report_root,
        ),
    }
    payload = json.dumps(comparison, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)
        return 0
    print(payload)
    return 0


def _summarize_run(*, run_id: str, manifest_root: Path, report_root: Path) -> dict[str, object]:
    run_manifest_path = _find_latest_run_manifest(run_id=run_id, manifest_root=manifest_root)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_dir_name = run_manifest_path.parent.name
    checkpoint_manifests = sorted(run_manifest_path.parent.glob("checkpoint_*.json"))
    latest_checkpoint = (
        json.loads(checkpoint_manifests[-1].read_text(encoding="utf-8"))
        if checkpoint_manifests
        else None
    )
    report_dir = report_root / run_dir_name
    metrics_path = report_dir / "metrics.jsonl"
    summary_path = report_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else None
    )
    last_metrics = _read_last_metrics_row(metrics_path) if metrics_path.exists() else None
    return {
        "run_id": run_id,
        "run_manifest_path": str(run_manifest_path),
        "report_dir": str(report_dir),
        "latest_checkpoint_manifest_path": (
            str(checkpoint_manifests[-1]) if checkpoint_manifests else None
        ),
        "latest_checkpoint_id": latest_checkpoint["checkpoint_id"] if latest_checkpoint else None,
        "latest_checkpoint_step": latest_checkpoint["global_step"] if latest_checkpoint else None,
        "initialization": run_manifest.get("initialization"),
        "warm_start_checkpoint_id": run_manifest.get("warm_start_checkpoint_id"),
        "summary": summary,
        "last_metrics": last_metrics,
    }


def _find_latest_run_manifest(*, run_id: str, manifest_root: Path) -> Path:
    matches: list[tuple[str, Path]] = []
    for path in manifest_root.glob("*/run_manifest.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("run_id") == run_id:
            started_at = str(document.get("started_at_utc", ""))
            matches.append((started_at, path))
    if not matches:
        raise FileNotFoundError(f"no run manifest found for run_id={run_id} in {manifest_root}")
    matches.sort()
    return matches[-1][1]


def _read_last_metrics_row(path: Path) -> dict[str, object] | None:
    last_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        return None
    return json.loads(last_line)


if __name__ == "__main__":
    raise SystemExit(main())
