from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from orbital_shepherd_ephemeris.constants import EPHEMERIS_SCHEMA_VERSION

NamespacedId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._:-]{2,127}$")]
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
HexDigest = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
SchemaVersion = Literal["1.0.0"]
SourceMode = Literal["fixture", "live"]
VisibilityTargetKind = Literal["ground_station", "target_cell", "ground_point"]


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return value.astimezone(UTC)


class EphemerisModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EphemerisContractModel(EphemerisModel):
    schema_version: SchemaVersion = EPHEMERIS_SCHEMA_VERSION


class RawSnapshotQuery(EphemerisModel):
    group: NonEmptyString
    format: NonEmptyString = "json"


class CelesTrakRawSnapshot(EphemerisContractModel):
    snapshot_id: NonEmptyString
    provider: Literal["celestrak"] = "celestrak"
    source_mode: SourceMode
    catalog_group: NonEmptyString
    fetched_at_utc: datetime
    query: RawSnapshotQuery
    records: list[dict[str, Any]] = Field(min_length=1)
    raw_snapshot_sha256: HexDigest

    _validate_fetched = field_validator("fetched_at_utc")(_ensure_utc)


class OrbitMeanElements(EphemerisModel):
    epoch_utc: datetime
    mean_motion_rev_per_day: float = Field(gt=0)
    inclination_deg: float = Field(ge=0, le=180)
    raan_deg: float = Field(ge=0, lt=360)
    eccentricity: float = Field(ge=0, lt=1)
    arg_perigee_deg: float = Field(ge=0, lt=360)
    mean_anomaly_deg: float = Field(ge=0, lt=360)
    bstar: float | None = None
    mean_motion_dot_rev_per_day2: float | None = None
    mean_motion_ddot_rev_per_day3: float | None = None
    rev_at_epoch: int | None = Field(default=None, ge=0)
    element_set_no: int | None = Field(default=None, ge=0)

    _validate_epoch = field_validator("epoch_utc")(_ensure_utc)


class OrbitAssetSource(EphemerisModel):
    provider: Literal["celestrak"] = "celestrak"
    source_mode: SourceMode
    catalog_group: NonEmptyString
    snapshot_id: NonEmptyString
    raw_record_index: int = Field(ge=0)
    raw_record_sha256: HexDigest


class OrbitAsset(EphemerisContractModel):
    satellite_id: NamespacedId
    norad_catalog_id: int = Field(ge=1)
    name: NonEmptyString
    international_designator: str | None = None
    object_type: str | None = None
    classification_type: str | None = None
    orbit: OrbitMeanElements
    source: OrbitAssetSource
    asset_fingerprint: HexDigest


class OrbitAssetBundleSource(EphemerisModel):
    provider: Literal["celestrak"] = "celestrak"
    source_mode: SourceMode
    catalog_group: NonEmptyString
    snapshot_id: NonEmptyString
    fetched_at_utc: datetime
    raw_snapshot_sha256: HexDigest
    raw_snapshot_path: str | None = None

    _validate_fetched = field_validator("fetched_at_utc")(_ensure_utc)


class OrbitAssetBundleCompilation(EphemerisModel):
    compiled_at_utc: datetime
    compiler_version: NonEmptyString

    _validate_compiled_at = field_validator("compiled_at_utc")(_ensure_utc)


class OrbitAssetBundle(EphemerisContractModel):
    bundle_id: NamespacedId
    source: OrbitAssetBundleSource
    compilation: OrbitAssetBundleCompilation
    assets: list[OrbitAsset] = Field(min_length=1)
    bundle_fingerprint: HexDigest


class GroundLocation(EphemerisModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt_m: float = 0.0


class VisibilityTarget(EphemerisModel):
    target_id: NamespacedId
    target_kind: VisibilityTargetKind
    location: GroundLocation
    min_elevation_deg: float | None = Field(default=None, ge=0, le=90)
    max_off_nadir_deg: float | None = Field(default=None, ge=0, le=90)


class CartesianVector(EphemerisModel):
    x: float
    y: float
    z: float


class SatelliteStateSample(EphemerisModel):
    satellite_id: NamespacedId
    timestamp_utc: datetime
    position_eci_m: CartesianVector
    velocity_eci_mps: CartesianVector
    position_ecef_m: CartesianVector
    velocity_ecef_mps: CartesianVector
    latitude_deg: float = Field(ge=-90, le=90)
    longitude_deg: float = Field(ge=-180, le=180)
    altitude_m: float

    _validate_timestamp = field_validator("timestamp_utc")(_ensure_utc)


class VisibilityWindow(EphemerisModel):
    satellite_id: NamespacedId
    target_id: NamespacedId
    target_kind: VisibilityTargetKind
    start_time_utc: datetime
    end_time_utc: datetime
    sample_count: int = Field(ge=1)
    peak_elevation_deg: float
    closest_approach_km: float = Field(ge=0)
    minimum_off_nadir_deg: float | None = Field(default=None, ge=0, le=180)

    _validate_start = field_validator("start_time_utc")(_ensure_utc)
    _validate_end = field_validator("end_time_utc")(_ensure_utc)
