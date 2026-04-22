from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from orbital_shepherd_ephemeris.models import (
    CartesianVector,
    GroundLocation,
    OrbitAsset,
    OrbitAssetBundle,
    SatelliteStateSample,
    VisibilityTarget,
    VisibilityWindow,
)

_MU_EARTH_M3_PER_S2 = 3.986004418e14
_EARTH_ROTATION_RATE_RAD_PER_S = 7.2921150e-5
_WGS84_A_M = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class PropagationBackend(ABC):
    @abstractmethod
    def load_orbit_asset_snapshot(
        self, source: OrbitAssetBundle | Mapping[str, Any] | Path | str
    ) -> OrbitAssetBundle:
        raise NotImplementedError

    @abstractmethod
    def sample_satellite_states(
        self,
        source: OrbitAssetBundle | Mapping[str, Any] | Path | str,
        *,
        start_time_utc: datetime,
        end_time_utc: datetime,
        step_seconds: int = 60,
        satellite_ids: Sequence[str] | None = None,
    ) -> list[SatelliteStateSample]:
        raise NotImplementedError

    @abstractmethod
    def compute_visibility(
        self,
        source: OrbitAssetBundle | Mapping[str, Any] | Path | str,
        *,
        targets: Sequence[VisibilityTarget],
        start_time_utc: datetime,
        end_time_utc: datetime,
        step_seconds: int = 60,
        satellite_ids: Sequence[str] | None = None,
    ) -> list[VisibilityWindow]:
        raise NotImplementedError


class DeterministicKeplerPropagationBackend(PropagationBackend):
    def load_orbit_asset_snapshot(
        self, source: OrbitAssetBundle | Mapping[str, Any] | Path | str
    ) -> OrbitAssetBundle:
        if isinstance(source, OrbitAssetBundle):
            return source
        if isinstance(source, Path):
            payload = json.loads(source.read_text(encoding="utf-8"))
        elif isinstance(source, str):
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            payload = dict(source)
        return OrbitAssetBundle.model_validate(payload)

    def sample_satellite_states(
        self,
        source: OrbitAssetBundle | Mapping[str, Any] | Path | str,
        *,
        start_time_utc: datetime,
        end_time_utc: datetime,
        step_seconds: int = 60,
        satellite_ids: Sequence[str] | None = None,
    ) -> list[SatelliteStateSample]:
        bundle = self.load_orbit_asset_snapshot(source)
        step = _normalize_step(step_seconds)
        _validate_window(start_time_utc, end_time_utc)
        assets = _select_assets(bundle.assets, satellite_ids)
        samples: list[SatelliteStateSample] = []
        for asset in assets:
            current = start_time_utc
            while current <= end_time_utc:
                samples.append(_sample_asset_state(asset, current))
                current += step
        return samples

    def compute_visibility(
        self,
        source: OrbitAssetBundle | Mapping[str, Any] | Path | str,
        *,
        targets: Sequence[VisibilityTarget],
        start_time_utc: datetime,
        end_time_utc: datetime,
        step_seconds: int = 60,
        satellite_ids: Sequence[str] | None = None,
    ) -> list[VisibilityWindow]:
        bundle = self.load_orbit_asset_snapshot(source)
        step = _normalize_step(step_seconds)
        _validate_window(start_time_utc, end_time_utc)
        assets = _select_assets(bundle.assets, satellite_ids)
        ordered_targets = sorted(targets, key=lambda item: item.target_id)
        windows: list[VisibilityWindow] = []

        for asset in assets:
            for target in ordered_targets:
                current_window: dict[str, Any] | None = None
                current = start_time_utc
                while current <= end_time_utc:
                    state = _sample_asset_state(asset, current)
                    metrics = _compute_visibility_metrics(state.position_ecef_m, target.location)
                    qualifies = _qualifies(metrics, target)
                    if qualifies:
                        if current_window is None:
                            current_window = {
                                "start_time_utc": current,
                                "end_time_utc": current,
                                "sample_count": 1,
                                "peak_elevation_deg": metrics["elevation_deg"],
                                "closest_approach_km": metrics["slant_range_m"] / 1000.0,
                                "minimum_off_nadir_deg": metrics["off_nadir_deg"],
                            }
                        else:
                            current_window["end_time_utc"] = current
                            current_window["sample_count"] += 1
                            current_window["peak_elevation_deg"] = max(
                                current_window["peak_elevation_deg"], metrics["elevation_deg"]
                            )
                            current_window["closest_approach_km"] = min(
                                current_window["closest_approach_km"],
                                metrics["slant_range_m"] / 1000.0,
                            )
                            current_window["minimum_off_nadir_deg"] = min(
                                current_window["minimum_off_nadir_deg"],
                                metrics["off_nadir_deg"],
                            )
                    elif current_window is not None:
                        windows.append(
                            VisibilityWindow(
                                satellite_id=asset.satellite_id,
                                target_id=target.target_id,
                                target_kind=target.target_kind,
                                **current_window,
                            )
                        )
                        current_window = None
                    current += step

                if current_window is not None:
                    windows.append(
                        VisibilityWindow(
                            satellite_id=asset.satellite_id,
                            target_id=target.target_id,
                            target_kind=target.target_kind,
                            **current_window,
                        )
                    )

        return sorted(
            windows, key=lambda item: (item.satellite_id, item.target_id, item.start_time_utc)
        )


def _normalize_step(step_seconds: int) -> timedelta:
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    return timedelta(seconds=step_seconds)


def _validate_window(start_time_utc: datetime, end_time_utc: datetime) -> None:
    for value in (start_time_utc, end_time_utc):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("sampling windows must use UTC timestamps")
    if end_time_utc < start_time_utc:
        raise ValueError("end_time_utc must be greater than or equal to start_time_utc")


def _select_assets(assets: Sequence[OrbitAsset], satellite_ids: Sequence[str] | None) -> list[OrbitAsset]:
    if satellite_ids is None:
        return sorted(assets, key=lambda item: item.satellite_id)
    requested = set(satellite_ids)
    selected = [asset for asset in assets if asset.satellite_id in requested]
    if len(selected) != len(requested):
        missing = sorted(requested - {asset.satellite_id for asset in selected})
        raise ValueError(f"unknown satellite IDs requested: {', '.join(missing)}")
    return sorted(selected, key=lambda item: item.satellite_id)


def _sample_asset_state(asset: OrbitAsset, when: datetime) -> SatelliteStateSample:
    position_eci, velocity_eci = _keplerian_state_vectors(asset, when)
    gmst = _gmst_radians(when)
    position_ecef = _eci_to_ecef(position_eci, gmst)
    velocity_ecef = _eci_velocity_to_ecef(position_eci, velocity_eci, gmst)
    latitude_deg, longitude_deg, altitude_m = _ecef_to_geodetic(position_ecef)
    return SatelliteStateSample(
        satellite_id=asset.satellite_id,
        timestamp_utc=when,
        position_eci_m=CartesianVector(x=position_eci[0], y=position_eci[1], z=position_eci[2]),
        velocity_eci_mps=CartesianVector(x=velocity_eci[0], y=velocity_eci[1], z=velocity_eci[2]),
        position_ecef_m=CartesianVector(
            x=position_ecef[0],
            y=position_ecef[1],
            z=position_ecef[2],
        ),
        velocity_ecef_mps=CartesianVector(
            x=velocity_ecef[0],
            y=velocity_ecef[1],
            z=velocity_ecef[2],
        ),
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        altitude_m=altitude_m,
    )


def _keplerian_state_vectors(asset: OrbitAsset, when: datetime) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    orbit = asset.orbit
    delta_seconds = (when - orbit.epoch_utc).total_seconds()
    mean_motion_rad_per_s = orbit.mean_motion_rev_per_day * 2.0 * math.pi / 86400.0
    semi_major_axis_m = (_MU_EARTH_M3_PER_S2 / mean_motion_rad_per_s**2) ** (1.0 / 3.0)
    eccentricity = orbit.eccentricity
    mean_anomaly = math.radians(orbit.mean_anomaly_deg) + (mean_motion_rad_per_s * delta_seconds)
    eccentric_anomaly = _solve_kepler(mean_anomaly, eccentricity)
    radius_m = semi_major_axis_m * (1.0 - eccentricity * math.cos(eccentric_anomaly))
    sqrt_one_minus_e2 = math.sqrt(1.0 - eccentricity**2)

    x_orb = semi_major_axis_m * (math.cos(eccentric_anomaly) - eccentricity)
    y_orb = semi_major_axis_m * sqrt_one_minus_e2 * math.sin(eccentric_anomaly)
    orbital_speed_factor = math.sqrt(_MU_EARTH_M3_PER_S2 * semi_major_axis_m) / radius_m
    vx_orb = -orbital_speed_factor * math.sin(eccentric_anomaly)
    vy_orb = orbital_speed_factor * sqrt_one_minus_e2 * math.cos(eccentric_anomaly)

    return _rotate_orbital_plane(
        x_orb,
        y_orb,
        vx_orb,
        vy_orb,
        raan_deg=orbit.raan_deg,
        inclination_deg=orbit.inclination_deg,
        arg_perigee_deg=orbit.arg_perigee_deg,
    )


def _solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    wrapped = math.fmod(mean_anomaly, 2.0 * math.pi)
    if wrapped < 0:
        wrapped += 2.0 * math.pi
    estimate = wrapped if eccentricity < 0.8 else math.pi
    for _ in range(12):
        numerator = estimate - eccentricity * math.sin(estimate) - wrapped
        denominator = 1.0 - eccentricity * math.cos(estimate)
        estimate -= numerator / denominator
    return estimate


def _rotate_orbital_plane(
    x_orb: float,
    y_orb: float,
    vx_orb: float,
    vy_orb: float,
    *,
    raan_deg: float,
    inclination_deg: float,
    arg_perigee_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    raan = math.radians(raan_deg)
    inclination = math.radians(inclination_deg)
    arg_perigee = math.radians(arg_perigee_deg)
    cos_raan = math.cos(raan)
    sin_raan = math.sin(raan)
    cos_inc = math.cos(inclination)
    sin_inc = math.sin(inclination)
    cos_arg = math.cos(arg_perigee)
    sin_arg = math.sin(arg_perigee)

    x_peri = cos_arg * x_orb - sin_arg * y_orb
    y_peri = sin_arg * x_orb + cos_arg * y_orb
    vx_peri = cos_arg * vx_orb - sin_arg * vy_orb
    vy_peri = sin_arg * vx_orb + cos_arg * vy_orb

    x_temp = x_peri
    y_temp = cos_inc * y_peri
    z_temp = sin_inc * y_peri
    vx_temp = vx_peri
    vy_temp = cos_inc * vy_peri
    vz_temp = sin_inc * vy_peri

    x = cos_raan * x_temp - sin_raan * y_temp
    y = sin_raan * x_temp + cos_raan * y_temp
    z = z_temp
    vx = cos_raan * vx_temp - sin_raan * vy_temp
    vy = sin_raan * vx_temp + cos_raan * vy_temp
    vz = vz_temp
    return (x, y, z), (vx, vy, vz)


def _gmst_radians(when: datetime) -> float:
    jd = (when - _UNIX_EPOCH).total_seconds() / 86400.0 + 2440587.5
    centuries = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * centuries**2
        - (centuries**3) / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


def _eci_to_ecef(vector: tuple[float, float, float], gmst: float) -> tuple[float, float, float]:
    cos_gmst = math.cos(gmst)
    sin_gmst = math.sin(gmst)
    x, y, z = vector
    return (
        cos_gmst * x + sin_gmst * y,
        -sin_gmst * x + cos_gmst * y,
        z,
    )


def _eci_velocity_to_ecef(
    position_eci: tuple[float, float, float],
    velocity_eci: tuple[float, float, float],
    gmst: float,
) -> tuple[float, float, float]:
    rotated_velocity = _eci_to_ecef(velocity_eci, gmst)
    position_ecef = _eci_to_ecef(position_eci, gmst)
    omega_cross_r = (
        -_EARTH_ROTATION_RATE_RAD_PER_S * position_ecef[1],
        _EARTH_ROTATION_RATE_RAD_PER_S * position_ecef[0],
        0.0,
    )
    return (
        rotated_velocity[0] - omega_cross_r[0],
        rotated_velocity[1] - omega_cross_r[1],
        rotated_velocity[2] - omega_cross_r[2],
    )


def _ecef_to_geodetic(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vector
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    alt = 0.0
    for _ in range(6):
        sin_lat = math.sin(lat)
        normal = _WGS84_A_M / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        alt = p / max(math.cos(lat), 1e-12) - normal
        lat = math.atan2(z, p * (1.0 - _WGS84_E2 * normal / (normal + alt)))
    return math.degrees(lat), _wrap_longitude(math.degrees(lon)), alt


def _geodetic_to_ecef(location: GroundLocation) -> tuple[float, float, float]:
    lat = math.radians(location.lat)
    lon = math.radians(location.lon)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    normal = _WGS84_A_M / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (normal + location.alt_m) * cos_lat * math.cos(lon)
    y = (normal + location.alt_m) * cos_lat * math.sin(lon)
    z = (normal * (1.0 - _WGS84_E2) + location.alt_m) * sin_lat
    return x, y, z


def _compute_visibility_metrics(
    position_ecef: CartesianVector, location: GroundLocation
) -> dict[str, float]:
    satellite = (position_ecef.x, position_ecef.y, position_ecef.z)
    target = _geodetic_to_ecef(location)
    rho = (
        satellite[0] - target[0],
        satellite[1] - target[1],
        satellite[2] - target[2],
    )
    slant_range_m = math.sqrt(rho[0] ** 2 + rho[1] ** 2 + rho[2] ** 2)

    lat = math.radians(location.lat)
    lon = math.radians(location.lon)
    up = (
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    )
    rho_unit = (rho[0] / slant_range_m, rho[1] / slant_range_m, rho[2] / slant_range_m)
    elevation_deg = math.degrees(math.asin(_clamp(_dot(rho_unit, up), -1.0, 1.0)))

    sat_norm = math.sqrt(satellite[0] ** 2 + satellite[1] ** 2 + satellite[2] ** 2)
    nadir = (-satellite[0] / sat_norm, -satellite[1] / sat_norm, -satellite[2] / sat_norm)
    to_target = (
        (target[0] - satellite[0]) / slant_range_m,
        (target[1] - satellite[1]) / slant_range_m,
        (target[2] - satellite[2]) / slant_range_m,
    )
    off_nadir_deg = math.degrees(math.acos(_clamp(_dot(nadir, to_target), -1.0, 1.0)))
    return {
        "elevation_deg": elevation_deg,
        "off_nadir_deg": off_nadir_deg,
        "slant_range_m": slant_range_m,
    }


def _qualifies(metrics: Mapping[str, float], target: VisibilityTarget) -> bool:
    min_elevation = target.min_elevation_deg if target.min_elevation_deg is not None else 0.0
    if metrics["elevation_deg"] < min_elevation:
        return False
    if target.max_off_nadir_deg is not None and metrics["off_nadir_deg"] > target.max_off_nadir_deg:
        return False
    return True


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_longitude(value: float) -> float:
    wrapped = ((value + 180.0) % 360.0) - 180.0
    return 180.0 if wrapped == -180.0 else wrapped
