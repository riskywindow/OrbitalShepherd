from __future__ import annotations

import sys

from _bootstrap import install_repo_sources


def main() -> int:
    install_repo_sources()
    from orbital_shepherd_region_builder.cli import main as region_main

    return region_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
