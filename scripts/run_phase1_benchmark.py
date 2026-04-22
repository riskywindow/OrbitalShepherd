from __future__ import annotations

from _bootstrap import install_repo_sources

def main() -> int:
    install_repo_sources()
    from orbital_shepherd_benchmark.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(main())
