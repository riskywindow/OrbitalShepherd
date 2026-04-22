from __future__ import annotations

from _bootstrap import install_repo_sources

def main() -> int | None:
    install_repo_sources()
    from orbital_shepherd_api.main import main as api_main

    return api_main()


if __name__ == "__main__":
    raise SystemExit(main())
