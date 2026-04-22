from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from orbital_shepherd_core import sha256_fingerprint
from orbital_shepherd_ephemeris import (
    DeterministicKeplerPropagationBackend,
    VisibilityTarget,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILED_BUNDLE_PATH = (
    REPO_ROOT
    / "data/fixtures/ephemeris/compiled/eph--demo-phase1--raw-celestrak-demo-phase1-2026-04-01t00-00-00z.json"
)
ISS_SATELLITE_ID = "sat:norad-25544:iss--zarya"


def test_deterministic_kepler_backend_emits_repeatable_state_samples() -> None:
    backend = DeterministicKeplerPropagationBackend()

    first = backend.sample_satellite_states(
        COMPILED_BUNDLE_PATH,
        start_time_utc=datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
        end_time_utc=datetime(2026, 4, 1, 0, 10, tzinfo=UTC),
        step_seconds=300,
        satellite_ids=[ISS_SATELLITE_ID],
    )
    second = backend.sample_satellite_states(
        COMPILED_BUNDLE_PATH,
        start_time_utc=datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
        end_time_utc=datetime(2026, 4, 1, 0, 10, tzinfo=UTC),
        step_seconds=300,
        satellite_ids=[ISS_SATELLITE_ID],
    )

    projection = [
        {
            "timestamp_utc": sample.timestamp_utc.isoformat().replace("+00:00", "Z"),
            "latitude_deg": round(sample.latitude_deg, 6),
            "longitude_deg": round(sample.longitude_deg, 6),
            "altitude_m": round(sample.altitude_m, 3),
        }
        for sample in first
    ]

    assert [sample.model_dump(mode="json") for sample in first] == [
        sample.model_dump(mode="json") for sample in second
    ]
    assert projection == [
        {
            "timestamp_utc": "2026-04-01T00:00:00Z",
            "latitude_deg": 37.949141,
            "longitude_deg": 128.972853,
            "altitude_m": 424745.895,
        },
        {
            "timestamp_utc": "2026-04-01T00:05:00Z",
            "latitude_deg": 24.685007,
            "longitude_deg": 144.359176,
            "altitude_m": 421784.07,
        },
        {
            "timestamp_utc": "2026-04-01T00:10:00Z",
            "latitude_deg": 9.931691,
            "longitude_deg": 156.382069,
            "altitude_m": 420232.987,
        },
    ]
    assert sha256_fingerprint(projection) == "df64fc100bb41fe2c339d7d684e6187ba3f915356802828b6e5f70a2c0713584"


def test_deterministic_kepler_backend_computes_coarse_visibility_windows() -> None:
    backend = DeterministicKeplerPropagationBackend()
    windows = backend.compute_visibility(
        COMPILED_BUNDLE_PATH,
        targets=[
            VisibilityTarget(
                target_id="gs:wallops",
                target_kind="ground_station",
                location={"lat": 37.9402, "lon": -75.4664, "alt_m": 3.0},
                min_elevation_deg=5.0,
            ),
            VisibilityTarget(
                target_id="tc:socal",
                target_kind="target_cell",
                location={"lat": 34.2523, "lon": -118.5134, "alt_m": 0.0},
                max_off_nadir_deg=55.0,
            ),
        ],
        start_time_utc=datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
        end_time_utc=datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
        step_seconds=300,
        satellite_ids=[ISS_SATELLITE_ID],
    )

    projection = [
        {
            "target_id": window.target_id,
            "start_time_utc": window.start_time_utc.isoformat().replace("+00:00", "Z"),
            "end_time_utc": window.end_time_utc.isoformat().replace("+00:00", "Z"),
            "peak_elevation_deg": round(window.peak_elevation_deg, 6),
            "closest_approach_km": round(window.closest_approach_km, 3),
            "minimum_off_nadir_deg": round(window.minimum_off_nadir_deg or 0.0, 6),
        }
        for window in windows
    ]

    assert projection == [
        {
            "target_id": "gs:wallops",
            "start_time_utc": "2026-04-01T05:50:00Z",
            "end_time_utc": "2026-04-01T05:50:00Z",
            "peak_elevation_deg": 26.105095,
            "closest_approach_km": 858.719,
            "minimum_off_nadir_deg": 57.25229,
        },
        {
            "target_id": "gs:wallops",
            "start_time_utc": "2026-04-01T07:25:00Z",
            "end_time_utc": "2026-04-01T07:30:00Z",
            "peak_elevation_deg": 18.855959,
            "closest_approach_km": 1073.804,
            "minimum_off_nadir_deg": 62.558693,
        },
        {
            "target_id": "gs:wallops",
            "start_time_utc": "2026-04-01T09:05:00Z",
            "end_time_utc": "2026-04-01T09:05:00Z",
            "peak_elevation_deg": 9.992925,
            "closest_approach_km": 1511.752,
            "minimum_off_nadir_deg": 67.525414,
        },
        {
            "target_id": "tc:socal",
            "start_time_utc": "2026-04-01T08:55:00Z",
            "end_time_utc": "2026-04-01T08:55:00Z",
            "peak_elevation_deg": 43.812597,
            "closest_approach_km": 588.997,
            "minimum_off_nadir_deg": 42.43659,
        },
    ]
    assert sha256_fingerprint(projection) == "018d9771cb305a8fba159053e64dd48e00461951ba762e1d3f9dafe8e8502ee9"
