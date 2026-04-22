from __future__ import annotations

from datetime import UTC, datetime

import pytest

from orbital_shepherd_core import (
    canonical_json_dumps,
    format_utc_timestamp,
    parse_utc_timestamp,
    seeded_rng,
    sha256_fingerprint,
    stable_id,
    stable_token,
)


def test_canonical_json_dumps_is_sorted_and_compact() -> None:
    payload = {"beta": 2, "alpha": 1}
    assert canonical_json_dumps(payload) == '{"alpha":1,"beta":2}'


def test_sha256_fingerprint_is_order_independent_for_mappings() -> None:
    left = {"alpha": 1, "beta": [1, 2, 3]}
    right = {"beta": [1, 2, 3], "alpha": 1}
    assert sha256_fingerprint(left) == sha256_fingerprint(right)


def test_parse_and_format_utc_timestamp_round_trip() -> None:
    parsed = parse_utc_timestamp("2026-08-14T00:00:00Z")
    assert parsed == datetime(2026, 8, 14, 0, 0, tzinfo=UTC)
    assert format_utc_timestamp(parsed) == "2026-08-14T00:00:00Z"


def test_parse_utc_timestamp_rejects_non_utc_offsets() -> None:
    with pytest.raises(ValueError, match="UTC"):
        parse_utc_timestamp("2026-08-14T00:00:00-05:00")


def test_seeded_rng_is_repeatable_for_string_seed() -> None:
    first = seeded_rng("cloud-trap:seed-42")
    second = seeded_rng("cloud-trap:seed-42")
    assert [first.random() for _ in range(3)] == [second.random() for _ in range(3)]


def test_stable_id_and_token_are_deterministic() -> None:
    identifier = stable_id("ep", "OSBench V01", "Cloud Trap", "Seed 42")
    assert identifier == "ep:osbench-v01:cloud-trap:seed-42"
    assert stable_token({"episode": identifier}, length=10) == stable_token(
        {"episode": identifier},
        length=10,
    )
