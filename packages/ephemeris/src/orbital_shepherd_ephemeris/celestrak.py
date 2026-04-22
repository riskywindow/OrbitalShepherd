from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import parse, request

from orbital_shepherd_core import canonical_json_dumps, format_utc_timestamp, sha256_fingerprint, stable_id
from orbital_shepherd_ephemeris.constants import (
    CELESTRAK_GP_URL,
    CELESTRAK_PROVIDER,
    EPHEMERIS_COMPILER_VERSION,
    EPHEMERIS_SCHEMA_VERSION,
)
from orbital_shepherd_ephemeris.models import (
    CelesTrakRawSnapshot,
    OrbitAsset,
    OrbitAssetBundle,
    OrbitAssetBundleCompilation,
    OrbitAssetBundleSource,
    OrbitAssetSource,
    OrbitMeanElements,
    RawSnapshotQuery,
    SourceMode,
)


class CelesTrakClient:
    def __init__(self, *, base_url: str = CELESTRAK_GP_URL) -> None:
        self.base_url = base_url

    def fetch_group(self, group: str, *, fetched_at: datetime | None = None) -> CelesTrakRawSnapshot:
        fetched_at_value = fetched_at or datetime.now(UTC)
        params = parse.urlencode({"GROUP": group, "FORMAT": "json"})
        url = f"{self.base_url}?{params}"
        with request.urlopen(url, timeout=30) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise ValueError("expected a JSON array from CelesTrak GP endpoint")
        return self.build_snapshot(
            payload,
            catalog_group=group,
            source_mode="live",
            fetched_at=fetched_at_value,
        )

    def build_snapshot(
        self,
        raw_records: Sequence[Mapping[str, Any]],
        *,
        catalog_group: str,
        source_mode: SourceMode,
        fetched_at: datetime,
    ) -> CelesTrakRawSnapshot:
        normalized_records = [dict(record) for record in raw_records]
        snapshot_id = _snapshot_id(catalog_group, fetched_at)
        snapshot_document = {
            "schema_version": EPHEMERIS_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "provider": CELESTRAK_PROVIDER,
            "source_mode": source_mode,
            "catalog_group": catalog_group,
            "fetched_at_utc": format_utc_timestamp(fetched_at),
            "query": {"group": catalog_group, "format": "json"},
            "records": normalized_records,
        }
        raw_snapshot_sha256 = sha256_fingerprint(snapshot_document)
        return CelesTrakRawSnapshot.model_validate(
            {**snapshot_document, "raw_snapshot_sha256": raw_snapshot_sha256}
        )

    def load_snapshot(self, source: Path | str | Mapping[str, Any]) -> CelesTrakRawSnapshot:
        if isinstance(source, Path):
            payload = json.loads(source.read_text(encoding="utf-8"))
        elif isinstance(source, str):
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            payload = dict(source)
        snapshot = CelesTrakRawSnapshot.model_validate(payload)
        expected = sha256_fingerprint(snapshot.model_dump(mode="json", exclude={"raw_snapshot_sha256"}))
        if snapshot.raw_snapshot_sha256 != expected:
            raise ValueError("raw snapshot fingerprint does not match payload")
        return snapshot

    def persist_raw_snapshot(self, snapshot: CelesTrakRawSnapshot, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{snapshot.snapshot_id}--{snapshot.raw_snapshot_sha256[:12]}.json"
        path = directory / filename
        content = canonical_json_dumps(snapshot.model_dump(mode="json"))
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != content:
                raise ValueError(f"refusing to overwrite non-identical raw snapshot: {path}")
            return path
        path.write_text(f"{content}\n", encoding="utf-8")
        return path

    def compile_orbit_assets(
        self,
        source: CelesTrakRawSnapshot | Path | str | Mapping[str, Any],
        *,
        raw_snapshot_path: Path | None = None,
    ) -> OrbitAssetBundle:
        snapshot = source if isinstance(source, CelesTrakRawSnapshot) else self.load_snapshot(source)
        assets = [
            _build_orbit_asset(snapshot, record, index)
            for index, record in sorted(
                enumerate(snapshot.records),
                key=lambda item: (
                    int(_get_required(item[1], "NORAD_CAT_ID", "norad_cat_id")),
                    str(_get_optional(item[1], "OBJECT_NAME", "object_name") or ""),
                ),
            )
        ]
        if len({asset.norad_catalog_id for asset in assets}) != len(assets):
            raise ValueError("raw snapshot contains duplicate NORAD catalog IDs")

        source_block = OrbitAssetBundleSource.model_validate(
            {
                "provider": snapshot.provider,
                "source_mode": snapshot.source_mode,
                "catalog_group": snapshot.catalog_group,
                "snapshot_id": snapshot.snapshot_id,
                "fetched_at_utc": snapshot.fetched_at_utc,
                "raw_snapshot_sha256": snapshot.raw_snapshot_sha256,
                "raw_snapshot_path": str(raw_snapshot_path) if raw_snapshot_path is not None else None,
            }
        )
        compilation = OrbitAssetBundleCompilation(
            compiled_at_utc=snapshot.fetched_at_utc,
            compiler_version=EPHEMERIS_COMPILER_VERSION,
        )
        bundle_id = stable_id("eph", snapshot.catalog_group, snapshot.snapshot_id)
        bundle_document = {
            "schema_version": snapshot.schema_version,
            "bundle_id": bundle_id,
            "source": source_block.model_dump(mode="json"),
            "compilation": compilation.model_dump(mode="json"),
            "assets": [asset.model_dump(mode="json") for asset in assets],
        }
        bundle_fingerprint = sha256_fingerprint(bundle_document)
        return OrbitAssetBundle.model_validate(
            {**bundle_document, "bundle_fingerprint": bundle_fingerprint}
        )


def _snapshot_id(catalog_group: str, fetched_at: datetime) -> str:
    timestamp = format_utc_timestamp(fetched_at).replace(":", "-")
    return stable_id("raw", CELESTRAK_PROVIDER, catalog_group, timestamp)


def _build_orbit_asset(
    snapshot: CelesTrakRawSnapshot, record: Mapping[str, Any], raw_record_index: int
) -> OrbitAsset:
    norad_catalog_id = _to_int(_get_required(record, "NORAD_CAT_ID", "norad_cat_id"))
    name = _normalize_name(str(_get_required(record, "OBJECT_NAME", "object_name")))
    satellite_id = stable_id("sat", f"norad-{norad_catalog_id}", name)
    raw_record_sha256 = sha256_fingerprint(record)
    orbit = OrbitMeanElements(
        epoch_utc=_to_datetime(_get_required(record, "EPOCH", "epoch")),
        mean_motion_rev_per_day=_to_float(
            _get_required(record, "MEAN_MOTION", "mean_motion")
        ),
        inclination_deg=_to_float(_get_required(record, "INCLINATION", "inclination")),
        raan_deg=_wrap_degrees(
            _to_float(_get_required(record, "RA_OF_ASC_NODE", "raan", "ra_of_asc_node"))
        ),
        eccentricity=_to_float(_get_required(record, "ECCENTRICITY", "eccentricity")),
        arg_perigee_deg=_wrap_degrees(
            _to_float(_get_required(record, "ARG_OF_PERICENTER", "arg_of_pericenter"))
        ),
        mean_anomaly_deg=_wrap_degrees(
            _to_float(_get_required(record, "MEAN_ANOMALY", "mean_anomaly"))
        ),
        bstar=_to_optional_float(_get_optional(record, "BSTAR", "bstar")),
        mean_motion_dot_rev_per_day2=_to_optional_float(
            _get_optional(record, "MEAN_MOTION_DOT", "mean_motion_dot")
        ),
        mean_motion_ddot_rev_per_day3=_to_optional_float(
            _get_optional(record, "MEAN_MOTION_DDOT", "mean_motion_ddot")
        ),
        rev_at_epoch=_to_optional_int(_get_optional(record, "REV_AT_EPOCH", "rev_at_epoch")),
        element_set_no=_to_optional_int(
            _get_optional(record, "ELEMENT_SET_NO", "element_set_no")
        ),
    )
    asset_document = {
        "schema_version": snapshot.schema_version,
        "satellite_id": satellite_id,
        "norad_catalog_id": norad_catalog_id,
        "name": name,
        "international_designator": _to_optional_string(
            _get_optional(record, "OBJECT_ID", "object_id")
        ),
        "object_type": _to_optional_string(_get_optional(record, "OBJECT_TYPE", "object_type")),
        "classification_type": _to_optional_string(
            _get_optional(record, "CLASSIFICATION_TYPE", "classification_type")
        ),
        "orbit": orbit.model_dump(mode="json"),
        "source": OrbitAssetSource(
            provider=snapshot.provider,
            source_mode=snapshot.source_mode,
            catalog_group=snapshot.catalog_group,
            snapshot_id=snapshot.snapshot_id,
            raw_record_index=raw_record_index,
            raw_record_sha256=raw_record_sha256,
        ).model_dump(mode="json"),
    }
    asset_fingerprint = sha256_fingerprint(asset_document)
    return OrbitAsset.model_validate({**asset_document, "asset_fingerprint": asset_fingerprint})


def _get_required(record: Mapping[str, Any], *keys: str) -> Any:
    value = _get_optional(record, *keys)
    if value is None:
        joined = ", ".join(keys)
        raise ValueError(f"missing required CelesTrak field: {joined}")
    return value


def _get_optional(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _normalize_name(value: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("satellite name must not be blank")
    return normalized


def _to_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError("datetime fields must be strings")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("epoch timestamps must be UTC")
    return parsed.astimezone(UTC)


def _to_float(value: Any) -> float:
    return float(value)


def _to_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_int(value: Any) -> int:
    return int(value)


def _to_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _to_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _wrap_degrees(value: float) -> float:
    wrapped = value % 360.0
    if wrapped == 360.0:
        return 0.0
    return wrapped
