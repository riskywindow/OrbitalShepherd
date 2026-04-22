from __future__ import annotations

from _bootstrap import install_repo_sources


def main() -> int:
    install_repo_sources()
    from orbital_shepherd_training.cli import main as training_main

    return training_main()


if __name__ == "__main__":
    raise SystemExit(main())
