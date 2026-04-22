from __future__ import annotations

import sys

from _bootstrap import install_repo_sources

def main() -> int:
    install_repo_sources()
    from orbital_shepherd_scenario_engine.cli import main as scenario_main

    return scenario_main(["build-scenario-pack", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
