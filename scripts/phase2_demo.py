from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from _bootstrap import REPO_ROOT, install_repo_sources

DEFAULT_BASELINE_ID = "urgency_greedy"
DEFAULT_CHECKPOINT_RUN_ID = "ppo:phase2-online-from-bc-smoke-v1"


@dataclass(frozen=True, slots=True)
class Phase2DemoArtifacts:
    bundle_id: str
    baseline_id: str
    model_key: str
    checkpoint_id: str
    episode_id: str
    replay_path: Path
    defaults_path: Path
    report_summary_path: Path | None

    def ui_query(self) -> str:
        return (
            f"?scenario={self.bundle_id}"
            f"&baseline={self.baseline_id}"
            f"&episode={self.episode_id}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare evaluation-backed Phase 2 demo artifacts and optional local UI serving.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Run a trained checkpoint through the API service and write Phase 2 demo defaults.",
    )
    _add_prepare_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=_handle_prepare)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a resolved checkpoint against the committed Phase 2 baseline set.",
    )
    evaluate_parser.add_argument("--checkpoint-manifest", type=Path, default=None)
    evaluate_parser.add_argument("--checkpoint-run-id", default=DEFAULT_CHECKPOINT_RUN_ID)
    evaluate_parser.add_argument("--policy-label", default="ppo_from_bc_smoke")
    evaluate_parser.add_argument("--run-id", default="evalrun--phase2-heldout-smoke-v1")
    evaluate_parser.add_argument("--limit-bundles-per-split", type=int, default=1)
    evaluate_parser.set_defaults(handler=_handle_evaluate)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Prepare Phase 2 demo defaults, then start the API and built web app.",
    )
    _add_prepare_arguments(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--api-port", default=8000, type=int)
    serve_parser.add_argument("--web-port", default=3000, type=int)
    serve_parser.set_defaults(handler=_handle_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def prepare_phase2_demo_artifacts(
    *,
    checkpoint_manifest: Path | None,
    checkpoint_run_id: str,
    bundle_id: str | None,
    baseline_id: str,
) -> Phase2DemoArtifacts:
    install_repo_sources()

    from orbital_shepherd_api.models import ModelRunRequest
    from orbital_shepherd_api.service import Phase1ApiService
    from orbital_shepherd_api.settings import ApiSettings
    from orbital_shepherd_core import canonical_json_dumps

    settings = ApiSettings()
    service = Phase1ApiService(settings)
    service.model_registry.refresh()
    manifest_record = _resolve_checkpoint_manifest(
        manifest_root=settings.training_manifest_dir,
        checkpoint_manifest=checkpoint_manifest,
        checkpoint_run_id=checkpoint_run_id,
    )
    try:
        entry = service.model_registry.get_policy(manifest_record.checkpoint_id)
    except KeyError:
        entry = service.model_registry.register_checkpoint(
            manifest_path=manifest_record.manifest_path,
            model_key=manifest_record.checkpoint_id,
        )

    resolved_bundle_id = _resolve_bundle_id(
        service=service,
        preferred_bundle_id=bundle_id,
        preferred_bundle_ids=manifest_record.source_bundle_ids,
    )
    bundle = service.get_scenario(resolved_bundle_id)
    model_run = service.run_model(
        entry.model_key,
        ModelRunRequest(
            bundle_id=resolved_bundle_id,
            simulation_seed=bundle.simulation_seed,
            include_inference_traces=True,
        ),
    )
    if model_run.episode_id is None or model_run.replay_path is None:
        raise RuntimeError("trained-policy demo run did not produce a replayable episode")

    report_summary_path = _resolve_latest_report_summary(settings.training_report_dir)
    defaults_path = settings.phase2_demo_defaults_path
    defaults_payload = {
        "bundle_id": resolved_bundle_id,
        "baseline_id": baseline_id,
        "episode_id": model_run.episode_id,
        "benchmark_run_id": report_summary_path.parent.name if report_summary_path else None,
        "benchmark_summary_path": str(report_summary_path) if report_summary_path else None,
        "replay_path": model_run.replay_path,
    }
    defaults_path.parent.mkdir(parents=True, exist_ok=True)
    defaults_path.write_text(canonical_json_dumps(defaults_payload) + "\n", encoding="utf-8")
    return Phase2DemoArtifacts(
        bundle_id=resolved_bundle_id,
        baseline_id=baseline_id,
        model_key=entry.model_key,
        checkpoint_id=entry.checkpoint_id,
        episode_id=model_run.episode_id,
        replay_path=Path(model_run.replay_path),
        defaults_path=defaults_path,
        report_summary_path=report_summary_path,
    )


def _handle_prepare(args: argparse.Namespace) -> int:
    artifacts = prepare_phase2_demo_artifacts(
        checkpoint_manifest=args.checkpoint_manifest,
        checkpoint_run_id=args.checkpoint_run_id,
        bundle_id=args.bundle_id,
        baseline_id=args.baseline_id,
    )
    print(artifacts.defaults_path)
    if artifacts.report_summary_path is not None:
        print(artifacts.report_summary_path)
    print(artifacts.replay_path)
    print(f"http://127.0.0.1:3000/{artifacts.ui_query()}")
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    install_repo_sources()

    from orbital_shepherd_api.settings import ApiSettings
    from orbital_shepherd_training.evaluation import run_phase2_evaluation

    settings = ApiSettings()
    manifest_record = _resolve_checkpoint_manifest(
        manifest_root=settings.training_manifest_dir,
        checkpoint_manifest=args.checkpoint_manifest,
        checkpoint_run_id=args.checkpoint_run_id,
    )
    summary = run_phase2_evaluation(
        evaluation_config=REPO_ROOT / "training" / "configs" / "evaluation" / "phase2_eval.yaml",
        training_pack=settings.training_manifest_dir / "phase2-training-pack-manifest.json",
        split_registry=REPO_ROOT / "training" / "configs" / "curriculum" / "phase2_splits.yaml",
        checkpoint_manifests=(manifest_record.manifest_path,),
        policy_labels=(args.policy_label,),
        planner_ids=(
            "random_valid_action",
            "urgency_greedy",
            "value_density_greedy",
            "ortools_receding_horizon",
        ),
        splits=("val", "test", "ood"),
        limit_bundles_per_split=args.limit_bundles_per_split,
        run_id=args.run_id,
    )
    print(summary.report_dir)
    print(summary.summary_path)
    if summary.markdown_path is not None:
        print(summary.markdown_path)
    print(summary.run_manifest_path)
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    artifacts = prepare_phase2_demo_artifacts(
        checkpoint_manifest=args.checkpoint_manifest,
        checkpoint_run_id=args.checkpoint_run_id,
        bundle_id=args.bundle_id,
        baseline_id=args.baseline_id,
    )
    _ensure_web_dependencies()
    _run_web_build()

    env = dict(os.environ)
    env.setdefault("COREPACK_HOME", "/tmp/corepack")
    api_command = [
        sys.executable,
        "scripts/run_phase1_api.py",
        "--host",
        args.host,
        "--port",
        str(args.api_port),
    ]
    web_command = [
        "pnpm",
        "--dir",
        "apps/web",
        "preview",
        "--host",
        args.host,
        "--port",
        str(args.web_port),
    ]
    api_process = subprocess.Popen(api_command, cwd=REPO_ROOT, env=env)
    web_process = subprocess.Popen(web_command, cwd=REPO_ROOT, env=env)
    try:
        _wait_for_url(f"http://{args.host}:{args.api_port}/v1/health")
        _wait_for_url(f"http://{args.host}:{args.web_port}/")
        print(f"API docs: http://{args.host}:{args.api_port}/docs")
        print(f"UI: http://{args.host}:{args.web_port}/")
        print(f"Demo URL: http://{args.host}:{args.web_port}/{artifacts.ui_query()}")
        if artifacts.report_summary_path is not None:
            print(f"Evaluation summary: {artifacts.report_summary_path}")
        print("Press Ctrl-C to stop both servers.")
        try:
            _wait_for_processes(api_process, web_process)
        except KeyboardInterrupt:
            return 0
    finally:
        _terminate_process(web_process)
        _terminate_process(api_process)
    return 0


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-manifest", type=Path, default=None)
    parser.add_argument("--checkpoint-run-id", default=DEFAULT_CHECKPOINT_RUN_ID)
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--baseline-id", default=DEFAULT_BASELINE_ID)


def _resolve_checkpoint_manifest(
    *,
    manifest_root: Path,
    checkpoint_manifest: Path | None,
    checkpoint_run_id: str,
):
    from orbital_shepherd_policy_models import PolicyCheckpointManifestRecord

    if checkpoint_manifest is not None:
        return PolicyCheckpointManifestRecord.load(checkpoint_manifest)
    candidates: list[PolicyCheckpointManifestRecord] = []
    for path in sorted(manifest_root.glob("*/checkpoint_*.json")):
        record = PolicyCheckpointManifestRecord.load(path)
        if record.run_id == checkpoint_run_id:
            candidates.append(record)
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint manifests found for run_id={checkpoint_run_id!r} in {manifest_root}"
        )
    return max(
        candidates,
        key=lambda item: (item.global_step, item.created_at_utc, item.checkpoint_id),
    )


def _resolve_bundle_id(
    *,
    service: "Phase1ApiService",
    preferred_bundle_id: str | None,
    preferred_bundle_ids: Sequence[str],
) -> str:
    scenarios = service.list_scenarios()
    if not scenarios:
        raise RuntimeError("no scenarios are available for the Phase 2 demo")
    known_bundle_ids = {scenario.bundle_id for scenario in scenarios}
    if preferred_bundle_id and preferred_bundle_id in known_bundle_ids:
        return preferred_bundle_id
    for bundle_id in preferred_bundle_ids:
        if bundle_id in known_bundle_ids:
            return bundle_id
    phase2_scenarios = [
        scenario.bundle_id
        for scenario in scenarios
        if scenario.benchmark_id == "osbench-phase2-foundation-v1"
    ]
    if phase2_scenarios:
        return phase2_scenarios[0]
    return scenarios[0].bundle_id


def _resolve_latest_report_summary(training_report_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for path in sorted(training_report_dir.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("episodes"), list):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _ensure_web_dependencies() -> None:
    if not (REPO_ROOT / "apps" / "web" / "node_modules").exists():
        raise SystemExit(
            "apps/web/node_modules is missing; run "
            "`COREPACK_HOME=/tmp/corepack pnpm install` first."
        )


def _run_web_build() -> None:
    env = dict(os.environ)
    env.setdefault("COREPACK_HOME", "/tmp/corepack")
    subprocess.run(
        ["/bin/zsh", "-lc", "COREPACK_HOME=/tmp/corepack pnpm --dir apps/web build"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


def _wait_for_url(url: str, *, attempts: int = 60, delay_seconds: float = 0.5) -> None:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url) as response:
                if response.status < 500:
                    return
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(delay_seconds)
    raise RuntimeError(f"timed out waiting for {url}") from last_error


def _wait_for_processes(*processes: subprocess.Popen[bytes]) -> None:
    while True:
        for process in processes:
            return_code = process.poll()
            if return_code is not None:
                raise SystemExit(return_code)
        time.sleep(0.5)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
