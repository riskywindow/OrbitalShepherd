from __future__ import annotations

import sys

from _bootstrap import install_repo_sources

def main() -> int:
    install_repo_sources()
    from orbital_shepherd_ephemeris.cli import main as ephemeris_main

    return ephemeris_main(["fetch-ephemeris", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
