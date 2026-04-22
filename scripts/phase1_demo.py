from __future__ import annotations

import argparse
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
DEFAULT_BENCHMARK_RUN_ID = "phase1-pack-v1"
DEFAULT_BUNDLE_ID = "sb:osbench-phase1-pack-v1:cloud_trap:seed-301"


@dataclass(frozen=True, slots=True)
class DemoArtifacts:
    bundle_id: str
    baseline_id: str
    episode_id: str
    benchmark_run_id: str
    benchmark_summary_path: Path
    replay_path: Path
    defaults_path: Path

    def ui_query(self) -> str:
        return (
            f"?scenario={self.bundle_id}"
            f"&baseline={self.baseline_id}"
            f"&episode={self.episode_id}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and serve the Orbital Shepherd Phase 1 demo stack.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Build scenarios, run the benchmark pack, and materialize a default demo episode.",
    )
    prepare_parser.add_argument("--bundle-id", default=DEFAULT_BUNDLE_ID)
    prepare_parser.add_argument("--baseline-id", default=DEFAULT_BASELINE_ID)
    prepare_parser.add_argument("--benchmark-run-id", default=DEFAULT_BENCHMARK_RUN_ID)
    prepare_parser.set_defaults(handler=_handle_prepare)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Prepare the demo artifacts, then start the API and web app.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--api-port", default=8000, type=int)
    serve_parser.add_argument("--web-port", default=3000, type=int)
    serve_parser.add_argument("--bundle-id", default=DEFAULT_BUNDLE_ID)
    serve_parser.add_argument("--baseline-id", default=DEFAULT_BASELINE_ID)
    serve_parser.add_argument("--benchmark-run-id", default=DEFAULT_BENCHMARK_RUN_ID)
    serve_parser.set_defaults(handler=_handle_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def prepare_demo_artifacts(
    *,
    bundle_id: str,
    baseline_id: str,
    benchmark_run_id: str,
) -> DemoArtifacts:
    install_repo_sources()
    from orbital_shepherd_api.models import BaselineRunRequest
    from orbital_shepherd_api.service import Phase1ApiService
    from orbital_shepherd_api.settings import ApiSettings
    from orbital_shepherd_benchmark import BenchmarkConfig, run_benchmark
    from orbital_shepherd_core import canonical_json_dumps
    from orbital_shepherd_scenario_engine import (
        ScenarioEngineConfig,
        build_scenario_pack,
        validate_scenario_pack,
    )

    scenario_config = ScenarioEngineConfig()
    build_scenario_pack(engine_config=scenario_config, output_dir=scenario_config.scenario_dir)
    validate_scenario_pack(engine_config=scenario_config, input_dir=scenario_config.scenario_dir)

    benchmark_result = run_benchmark(
        config=BenchmarkConfig(
            benchmark_id=scenario_config.benchmark_id,
            scenario_dir=scenario_config.scenario_dir,
        ),
        run_id=benchmark_run_id,
    )

    service = Phase1ApiService(ApiSettings())
    resolved_bundle_id = _resolve_demo_bundle_id(service, preferred_bundle_id=bundle_id)
    bundle = service.get_scenario(resolved_bundle_id)
    baseline_run = service.run_baseline(
        baseline_id,
        BaselineRunRequest(
            bundle_id=resolved_bundle_id,
            simulation_seed=bundle.simulation_seed,
        ),
    )
    if baseline_run.episode_id is None or baseline_run.replay_path is None:
        raise RuntimeError("demo baseline run did not produce a replay episode")

    defaults_path = service.settings.demo_defaults_path
    defaults_payload = {
        "bundle_id": resolved_bundle_id,
        "baseline_id": baseline_id,
        "episode_id": baseline_run.episode_id,
        "benchmark_run_id": benchmark_result.run_id,
        "benchmark_summary_path": str(benchmark_result.report_dir / "summary.json"),
        "replay_path": baseline_run.replay_path,
    }
    defaults_path.parent.mkdir(parents=True, exist_ok=True)
    defaults_path.write_text(canonical_json_dumps(defaults_payload) + "\n", encoding="utf-8")

    return DemoArtifacts(
        bundle_id=resolved_bundle_id,
        baseline_id=baseline_id,
        episode_id=baseline_run.episode_id,
        benchmark_run_id=benchmark_result.run_id,
        benchmark_summary_path=benchmark_result.report_dir / "summary.json",
        replay_path=Path(baseline_run.replay_path),
        defaults_path=defaults_path,
    )


def _handle_prepare(args: argparse.Namespace) -> int:
    artifacts = prepare_demo_artifacts(
        bundle_id=args.bundle_id,
        baseline_id=args.baseline_id,
        benchmark_run_id=args.benchmark_run_id,
    )
    print(artifacts.defaults_path)
    print(artifacts.benchmark_summary_path)
    print(artifacts.replay_path)
    print(f"http://127.0.0.1:3000/{artifacts.ui_query()}")
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    artifacts = prepare_demo_artifacts(
        bundle_id=args.bundle_id,
        baseline_id=args.baseline_id,
        benchmark_run_id=args.benchmark_run_id,
    )
    _ensure_web_dependencies()
    _run_web_build()

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

    env = dict(os.environ)
    env.setdefault("COREPACK_HOME", "/tmp/corepack")

    api_process = subprocess.Popen(api_command, cwd=REPO_ROOT, env=env)
    web_process = subprocess.Popen(web_command, cwd=REPO_ROOT, env=env)
    try:
        _wait_for_url(f"http://{args.host}:{args.api_port}/v1/health")
        _wait_for_url(f"http://{args.host}:{args.web_port}/")
        print(f"API docs: http://{args.host}:{args.api_port}/docs")
        print(f"UI: http://{args.host}:{args.web_port}/")
        print(f"Demo URL: http://{args.host}:{args.web_port}/{artifacts.ui_query()}")
        print(f"Benchmark summary: {artifacts.benchmark_summary_path}")
        print("Press Ctrl-C to stop both servers.")
        try:
            _wait_for_processes(api_process, web_process)
        except KeyboardInterrupt:
            return 0
    finally:
        _terminate_process(web_process)
        _terminate_process(api_process)
    return 0


def _resolve_demo_bundle_id(
    service: "Phase1ApiService",
    *,
    preferred_bundle_id: str,
) -> str:
    scenarios = service.list_scenarios()
    if not scenarios:
        raise RuntimeError("no scenarios are available for the Phase 1 demo")
    known_bundle_ids = {scenario.bundle_id for scenario in scenarios}
    if preferred_bundle_id in known_bundle_ids:
        return preferred_bundle_id
    cloud_trap = [
        scenario.bundle_id for scenario in scenarios if scenario.scenario_family == "cloud_trap"
    ]
    if cloud_trap:
        return cloud_trap[0]
    return scenarios[0].bundle_id


def _ensure_web_dependencies() -> None:
    if not (REPO_ROOT / "apps/web/node_modules").exists():
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
