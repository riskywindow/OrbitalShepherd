from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any, Protocol, cast

from orbital_shepherd_contracts.models import (
    Incident,
    IncidentState,
    PriorityClass,
    TargetCell,
    Wgs84Point,
)
from orbital_shepherd_core import seeded_rng, stable_id, stable_token
from orbital_shepherd_scenario_engine.catalog import CELL_CATALOG, FAMILY_SPECS, ScenarioRecipe


@dataclass(frozen=True, slots=True)
class TargetDemandPoint:
    point_id: str
    h3_cell: str
    lat: float
    lon: float
    region_name: str
    base_demand: float
    priority_class: PriorityClass


@dataclass(frozen=True, slots=True)
class IncidentTemplate:
    template_id: str
    target_h3_cell: str
    ignition_offset_minutes: int
    urgency_score: float
    confidence: float
    estimated_area_ha: float
    state: IncidentState


@dataclass(frozen=True, slots=True)
class CloudRiskSegment:
    profile_id: str
    segment_start_hour: int
    segment_end_hour: int
    cloud_risk: float


class TargetDemandAdapter(Protocol):
    def build_target_cells(self, recipe: ScenarioRecipe) -> list[TargetCell]:
        raise NotImplementedError


class IncidentAdapter(Protocol):
    def build_incidents(
        self, recipe: ScenarioRecipe, target_cells: Sequence[TargetCell]
    ) -> list[Incident]:
        raise NotImplementedError


class WeatherAdapter(Protocol):
    def cloud_risk(self, target_cell: TargetCell, when: datetime) -> float:
        raise NotImplementedError


class FixtureTargetDemandAdapter:
    def __init__(self, fixture_dir: Path) -> None:
        self.fixture_dir = fixture_dir

    def build_target_cells(self, recipe: ScenarioRecipe) -> list[TargetCell]:
        selected = [
            point
            for point_id in recipe.fixture_target_point_ids
            for point in [_fixture_demand_points(self.fixture_dir).get(point_id)]
            if point is not None
        ]
        if len(selected) != len(recipe.fixture_target_point_ids):
            missing = sorted(
                set(recipe.fixture_target_point_ids) - {point.point_id for point in selected}
            )
            raise ValueError(f"missing fixture demand point(s): {', '.join(missing)}")
        return aggregate_target_cells(selected)


class SyntheticTargetDemandAdapter:
    def build_target_cells(self, recipe: ScenarioRecipe) -> list[TargetCell]:
        rng = seeded_rng(stable_id("demand", recipe.family, recipe.seed))
        candidate_cells = list(recipe.candidate_h3_cells) or sorted(CELL_CATALOG)
        if recipe.target_count <= 0:
            raise ValueError("synthetic target generation requires target_count > 0")
        count = min(recipe.target_count, len(candidate_cells))
        ordered = sorted(
            candidate_cells,
            key=lambda h3_cell: (
                -CELL_CATALOG[h3_cell].base_static_value,
                stable_token({"seed": recipe.seed, "cell": h3_cell}, length=8),
            ),
        )
        selected_h3_cells = ordered[:count]
        points: list[TargetDemandPoint] = []
        family_spec = FAMILY_SPECS[recipe.family]
        for h3_cell in selected_h3_cells:
            cell = CELL_CATALOG[h3_cell]
            point_count = 1 + rng.randint(0, 1)
            for point_index in range(point_count):
                jitter_lat = (rng.random() - 0.5) * 0.18
                jitter_lon = (rng.random() - 0.5) * 0.18
                base = cell.base_static_value * family_spec.demand_scale
                load = min(0.98, max(0.18, base + ((rng.random() - 0.5) * 0.12)))
                points.append(
                    TargetDemandPoint(
                        point_id=stable_id(
                            "dp", recipe.family, f"seed-{recipe.seed}", h3_cell, point_index
                        ),
                        h3_cell=h3_cell,
                        lat=cell.lat + jitter_lat,
                        lon=cell.lon + jitter_lon,
                        region_name=cell.region_name,
                        base_demand=round(load, 3),
                        priority_class=cast(PriorityClass, cell.priority_class),
                    )
                )
        return aggregate_target_cells(points)


class LiveTargetDemandAdapter:
    def build_target_cells(self, recipe: ScenarioRecipe) -> list[TargetCell]:
        raise NotImplementedError(
            "live target-demand adapter boundary is defined but not implemented "
            f"for {recipe.recipe_id}"
        )


class FixtureIncidentAdapter:
    def __init__(self, fixture_dir: Path) -> None:
        self.fixture_dir = fixture_dir

    def build_incidents(
        self, recipe: ScenarioRecipe, target_cells: Sequence[TargetCell]
    ) -> list[Incident]:
        target_id_by_h3 = {target.h3_cell: target.target_cell_id for target in target_cells}
        templates = [
            template
            for template_id in recipe.fixture_incident_template_ids
            for template in [_fixture_incident_templates(self.fixture_dir).get(template_id)]
            if template is not None
        ]
        if len(templates) != len(recipe.fixture_incident_template_ids):
            missing = sorted(
                set(recipe.fixture_incident_template_ids)
                - {template.template_id for template in templates}
            )
            raise ValueError(f"missing fixture incident template(s): {', '.join(missing)}")
        incidents = []
        for index, template in enumerate(
            sorted(templates, key=lambda item: item.ignition_offset_minutes)
        ):
            if template.target_h3_cell not in target_id_by_h3:
                raise ValueError(
                    f"fixture incident {template.template_id} targets H3 cell "
                    f"{template.target_h3_cell} that is absent from the manifest demand set"
                )
            ignition_time_utc = recipe.start_time_utc + timedelta(
                minutes=template.ignition_offset_minutes
            )
            incidents.append(
                Incident(
                    incident_id=stable_id(
                        "inc",
                        recipe.family,
                        f"seed-{recipe.seed}",
                        f"{index:03d}",
                    ),
                    incident_type="wildfire",
                    target_cell_id=target_id_by_h3[template.target_h3_cell],
                    ignition_time_utc=ignition_time_utc,
                    urgency_score=template.urgency_score,
                    confidence=template.confidence,
                    state=template.state,
                    estimated_area_ha=template.estimated_area_ha,
                )
            )
        return incidents


class SyntheticIncidentAdapter:
    def build_incidents(
        self, recipe: ScenarioRecipe, target_cells: Sequence[TargetCell]
    ) -> list[Incident]:
        if recipe.incident_count <= 0:
            raise ValueError("synthetic incident generation requires incident_count > 0")
        rng = seeded_rng(stable_id("incident", recipe.family, recipe.seed))
        family_spec = FAMILY_SPECS[recipe.family]
        ordered_targets = sorted(
            target_cells, key=lambda item: (-item.static_value, item.target_cell_id)
        )
        offsets = _synthetic_ignition_offsets(recipe, rng)
        incidents: list[Incident] = []
        for index, offset_minutes in enumerate(offsets):
            target = ordered_targets[index % len(ordered_targets)]
            urgency = min(
                0.99,
                max(
                    0.52,
                    target.static_value * 0.78
                    + family_spec.urgency_bias
                    + ((rng.random() - 0.5) * 0.08),
                ),
            )
            incidents.append(
                Incident(
                    incident_id=stable_id(
                        "inc", recipe.family, f"seed-{recipe.seed}", f"{index:03d}"
                    ),
                    incident_type="wildfire",
                    target_cell_id=target.target_cell_id,
                    ignition_time_utc=recipe.start_time_utc + timedelta(minutes=offset_minutes),
                    urgency_score=round(urgency, 3),
                    confidence=round(min(0.98, max(0.68, 0.82 + ((rng.random() - 0.5) * 0.16))), 3),
                    state="active",
                    estimated_area_ha=round(
                        60.0
                        + (target.static_value * 180.0)
                        + (index * 12.0)
                        + (rng.random() * 40.0),
                        1,
                    ),
                )
            )
        return sorted(incidents, key=lambda item: (item.ignition_time_utc, item.incident_id))


class LiveIncidentAdapter:
    def build_incidents(
        self, recipe: ScenarioRecipe, target_cells: Sequence[TargetCell]
    ) -> list[Incident]:
        raise NotImplementedError(
            f"live incident adapter boundary is defined but not implemented for {recipe.recipe_id}"
        )


class FixtureWeatherAdapter:
    def __init__(
        self,
        fixture_dir: Path,
        *,
        start_time_utc: datetime,
        weather_profile_by_target_cell: Mapping[str, str],
    ) -> None:
        self.fixture_dir = fixture_dir
        self.start_time_utc = start_time_utc
        self.weather_profile_by_target_cell = dict(weather_profile_by_target_cell)

    def cloud_risk(self, target_cell: TargetCell, when: datetime) -> float:
        profile_id = self.weather_profile_by_target_cell.get(
            target_cell.target_cell_id, "clear_morning"
        )
        elapsed_hours = (when - self.start_time_utc).total_seconds() / 3600.0
        for segment in _fixture_cloud_profiles(self.fixture_dir).get(profile_id, ()):
            if segment.segment_start_hour <= elapsed_hours < segment.segment_end_hour:
                return segment.cloud_risk
        segments = _fixture_cloud_profiles(self.fixture_dir).get(profile_id, ())
        if not segments:
            raise ValueError(f"unknown weather profile: {profile_id}")
        return segments[-1].cloud_risk


class SyntheticWeatherAdapter:
    def __init__(
        self,
        *,
        family: str,
        seed: int,
        start_time_utc: datetime,
        cloud_style: str,
        trap_relief_hour: int,
    ) -> None:
        self.family = family
        self.seed = seed
        self.start_time_utc = start_time_utc
        self.cloud_style = cloud_style
        self.trap_relief_hour = trap_relief_hour

    def cloud_risk(self, target_cell: TargetCell, when: datetime) -> float:
        elapsed_hours = (when - self.start_time_utc).total_seconds() / 3600.0
        phase = _stable_phase(self.seed, target_cell.target_cell_id)
        base_wave = 0.16 * (1.0 + math.sin((elapsed_hours / 3.2) + phase)) / 2.0
        static_term = target_cell.static_value * 0.12
        if self.cloud_style == "trap":
            trap_curve = 0.62 if elapsed_hours < self.trap_relief_hour else 0.18
            relief = max(0.0, elapsed_hours - self.trap_relief_hour) * 0.015
            return _clamp(0.16 + static_term + base_wave + trap_curve - relief, 0.05, 0.95)
        if self.cloud_style == "variable":
            volatility = 0.22 * (1.0 + math.sin((elapsed_hours / 1.6) + (phase * 1.7))) / 2.0
            return _clamp(0.18 + static_term + base_wave + volatility, 0.08, 0.9)
        return _clamp(0.12 + static_term + base_wave, 0.05, 0.7)


class LiveWeatherAdapter:
    def cloud_risk(self, target_cell: TargetCell, when: datetime) -> float:
        raise NotImplementedError(
            "live weather adapter boundary is defined but not implemented in Phase 1"
        )


def aggregate_target_cells(points: Sequence[TargetDemandPoint]) -> list[TargetCell]:
    grouped: dict[str, list[TargetDemandPoint]] = defaultdict(list)
    for point in points:
        grouped[point.h3_cell].append(point)

    target_cells: list[TargetCell] = []
    for h3_cell, group in grouped.items():
        weight_total = sum(point.base_demand for point in group)
        centroid_lat = sum(point.lat * point.base_demand for point in group) / weight_total
        centroid_lon = sum(point.lon * point.base_demand for point in group) / weight_total
        target_cells.append(
            TargetCell(
                target_cell_id=f"tc:{h3_cell}",
                h3_cell=h3_cell,
                centroid=Wgs84Point(
                    lat=round(centroid_lat, 4),
                    lon=round(centroid_lon, 4),
                ),
                region_name=group[0].region_name,
                static_value=round(min(1.0, weight_total), 3),
                priority_class=_max_priority(point.priority_class for point in group),
            )
        )
    return sorted(target_cells, key=lambda item: item.target_cell_id)


def build_weather_adapter(
    *,
    mode: str,
    fixture_dir: Path,
    family: str,
    seed: int,
    start_time_utc: datetime,
    weather_profile_by_target_cell: Mapping[str, str],
    cloud_style: str,
    trap_relief_hour: int,
) -> WeatherAdapter:
    if mode == "fixture":
        return FixtureWeatherAdapter(
            fixture_dir,
            start_time_utc=start_time_utc,
            weather_profile_by_target_cell=weather_profile_by_target_cell,
        )
    if mode == "synthetic":
        return SyntheticWeatherAdapter(
            family=family,
            seed=seed,
            start_time_utc=start_time_utc,
            cloud_style=cloud_style,
            trap_relief_hour=trap_relief_hour,
        )
    if mode == "live":
        return LiveWeatherAdapter()
    raise ValueError(f"unsupported weather adapter mode: {mode}")


def build_target_demand_adapter(mode: str, fixture_dir: Path) -> TargetDemandAdapter:
    if mode == "fixture":
        return FixtureTargetDemandAdapter(fixture_dir)
    if mode == "synthetic":
        return SyntheticTargetDemandAdapter()
    if mode == "live":
        return LiveTargetDemandAdapter()
    raise ValueError(f"unsupported target-demand adapter mode: {mode}")


def build_incident_adapter(mode: str, fixture_dir: Path) -> IncidentAdapter:
    if mode == "fixture":
        return FixtureIncidentAdapter(fixture_dir)
    if mode == "synthetic":
        return SyntheticIncidentAdapter()
    if mode == "live":
        return LiveIncidentAdapter()
    raise ValueError(f"unsupported incident adapter mode: {mode}")


def target_cell_ids_by_h3(target_cells: Sequence[TargetCell]) -> dict[str, str]:
    return {target.h3_cell: target.target_cell_id for target in target_cells}


@cache
def _fixture_demand_points(fixture_dir: Path) -> dict[str, TargetDemandPoint]:
    points: dict[str, TargetDemandPoint] = {}
    for row in _read_csv_rows(fixture_dir / "target_demand_points.csv"):
        point = TargetDemandPoint(
            point_id=row["point_id"],
            h3_cell=row["h3_cell"],
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            region_name=row["region_name"],
            base_demand=float(row["base_demand"]),
            priority_class=cast(PriorityClass, row["priority_class"]),
        )
        points[point.point_id] = point
    return points


@cache
def _fixture_incident_templates(fixture_dir: Path) -> dict[str, IncidentTemplate]:
    templates: dict[str, IncidentTemplate] = {}
    for row in _read_csv_rows(fixture_dir / "incident_replay_rows.csv"):
        template = IncidentTemplate(
            template_id=row["template_id"],
            target_h3_cell=row["target_h3_cell"],
            ignition_offset_minutes=int(row["ignition_offset_minutes"]),
            urgency_score=float(row["urgency_score"]),
            confidence=float(row["confidence"]),
            estimated_area_ha=float(row["estimated_area_ha"]),
            state=cast(IncidentState, row["state"]),
        )
        templates[template.template_id] = template
    return templates


@cache
def _fixture_cloud_profiles(fixture_dir: Path) -> dict[str, tuple[CloudRiskSegment, ...]]:
    grouped: dict[str, list[CloudRiskSegment]] = defaultdict(list)
    for row in _read_csv_rows(fixture_dir / "cloud_profiles.csv"):
        grouped[row["profile_id"]].append(
            CloudRiskSegment(
                profile_id=row["profile_id"],
                segment_start_hour=int(row["segment_start_hour"]),
                segment_end_hour=int(row["segment_end_hour"]),
                cloud_risk=float(row["cloud_risk"]),
            )
        )
    return {
        profile_id: tuple(sorted(segments, key=lambda item: item.segment_start_hour))
        for profile_id, segments in grouped.items()
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _synthetic_ignition_offsets(recipe: ScenarioRecipe, rng: Any) -> list[int]:
    horizon_minutes = recipe.horizon_hours * 60
    if recipe.family == "burst_outbreak":
        burst_window = max(45, recipe.burst_window_minutes or 75)
        burst_start = 120 + rng.randint(0, 180)
        return sorted(
            burst_start + rng.randint(0, burst_window) for _ in range(recipe.incident_count)
        )
    if recipe.family == "cloud_trap":
        lead = 90 + rng.randint(0, 120)
        return sorted(
            lead + (index * 55) + rng.randint(0, 20) for index in range(recipe.incident_count)
        )
    return sorted(
        rng.randint(90, max(120, horizon_minutes - 180)) for _ in range(recipe.incident_count)
    )


def _max_priority(priorities: Iterable[PriorityClass]) -> PriorityClass:
    ranking = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    ordered = sorted(tuple(priorities), key=lambda item: ranking[item], reverse=True)
    return ordered[0]


def _stable_phase(seed: int, target_cell_id: str) -> float:
    token = stable_token({"seed": seed, "target_cell_id": target_cell_id}, length=8)
    return (int(token, 16) / 16**8) * math.tau


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
