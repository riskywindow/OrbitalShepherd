from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def ephemeris_fixture_dir() -> Path:
    return repo_root() / "data" / "fixtures" / "ephemeris"


def ephemeris_raw_dir() -> Path:
    return ephemeris_fixture_dir() / "raw"


def ephemeris_compiled_dir() -> Path:
    return ephemeris_fixture_dir() / "compiled"
