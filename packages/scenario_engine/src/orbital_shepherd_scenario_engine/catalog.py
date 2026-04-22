from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

AdapterMode = Literal["fixture", "synthetic", "live"]


@dataclass(frozen=True, slots=True)
class CellCatalogEntry:
    h3_cell: str
    lat: float
    lon: float
    region_name: str
    base_static_value: float
    priority_class: str


@dataclass(frozen=True, slots=True)
class FamilySpec:
    family: str
    display_name: str
    description: str
    quality_threshold: float
    cloud_block_threshold: float
    cloud_style: str
    demand_scale: float
    urgency_bias: float


@dataclass(frozen=True, slots=True)
class ScenarioRecipe:
    recipe_id: str
    benchmark_id: str
    family: str
    seed: int
    start_time_utc: datetime
    notes: str
    demand_mode: AdapterMode
    incident_mode: AdapterMode
    weather_mode: AdapterMode
    fixture_target_point_ids: tuple[str, ...] = ()
    fixture_incident_template_ids: tuple[str, ...] = ()
    fixture_weather_profiles: tuple[tuple[str, str], ...] = ()
    candidate_h3_cells: tuple[str, ...] = ()
    target_count: int = 0
    incident_count: int = 0
    burst_window_minutes: int = 0
    family_parameters: dict[str, Any] = field(default_factory=dict)
    horizon_hours: int = 24
    decision_interval_seconds: int = 60


FAMILY_SPECS: dict[str, FamilySpec] = {
    "sparse_frontier": FamilySpec(
        family="sparse_frontier",
        display_name="Sparse Frontier",
        description="Few incidents, low contention, and broad geographic dispersion.",
        quality_threshold=0.52,
        cloud_block_threshold=0.72,
        cloud_style="stable",
        demand_scale=0.92,
        urgency_bias=0.04,
    ),
    "burst_outbreak": FamilySpec(
        family="burst_outbreak",
        display_name="Burst Outbreak",
        description="Multiple ignitions arrive in a short temporal burst.",
        quality_threshold=0.5,
        cloud_block_threshold=0.76,
        cloud_style="variable",
        demand_scale=1.08,
        urgency_bias=0.1,
    ),
    "cloud_trap": FamilySpec(
        family="cloud_trap",
        display_name="Cloud Trap",
        description=(
            "Early windows are tempting but cloud-obstructed; later cleaner passes "
            "arrive with delay."
        ),
        quality_threshold=0.46,
        cloud_block_threshold=0.82,
        cloud_style="trap",
        demand_scale=1.05,
        urgency_bias=0.08,
    ),
    "downlink_crunch": FamilySpec(
        family="downlink_crunch",
        display_name="Downlink Crunch",
        description="Observation throughput exceeds nominal contact and buffer relief capacity.",
        quality_threshold=0.48,
        cloud_block_threshold=0.74,
        cloud_style="variable",
        demand_scale=1.12,
        urgency_bias=0.1,
    ),
    "station_outage": FamilySpec(
        family="station_outage",
        display_name="Station Outage",
        description="Ground-station outages and degraded contacts disrupt otherwise valid plans.",
        quality_threshold=0.47,
        cloud_block_threshold=0.76,
        cloud_style="stable",
        demand_scale=1.0,
        urgency_bias=0.09,
    ),
    "constellation_degradation": FamilySpec(
        family="constellation_degradation",
        display_name="Constellation Degradation",
        description="Reduced constellation capacity forces robustness to degraded orbital assets.",
        quality_threshold=0.45,
        cloud_block_threshold=0.78,
        cloud_style="stable",
        demand_scale=1.0,
        urgency_bias=0.08,
    ),
}


CELL_CATALOG: dict[str, CellCatalogEntry] = {
    "8928308280fffff": CellCatalogEntry(
        h3_cell="8928308280fffff",
        lat=34.2523,
        lon=-118.5134,
        region_name="Southern California",
        base_static_value=0.82,
        priority_class="critical",
    ),
    "8928341aec3ffff": CellCatalogEntry(
        h3_cell="8928341aec3ffff",
        lat=39.5361,
        lon=-121.7125,
        region_name="Northern California",
        base_static_value=0.78,
        priority_class="high",
    ),
    "8928d54250bffff": CellCatalogEntry(
        h3_cell="8928d54250bffff",
        lat=44.0582,
        lon=-121.3153,
        region_name="Central Oregon",
        base_static_value=0.68,
        priority_class="high",
    ),
    "88268cda81fffff": CellCatalogEntry(
        h3_cell="88268cda81fffff",
        lat=39.1178,
        lon=-105.3589,
        region_name="Colorado Front Range",
        base_static_value=0.58,
        priority_class="medium",
    ),
    "892f054b6d7ffff": CellCatalogEntry(
        h3_cell="892f054b6d7ffff",
        lat=56.7269,
        lon=-111.38,
        region_name="Alberta Boreal",
        base_static_value=0.74,
        priority_class="high",
    ),
    "892f5ab6e13ffff": CellCatalogEntry(
        h3_cell="892f5ab6e13ffff",
        lat=52.0614,
        lon=-122.2297,
        region_name="British Columbia Interior",
        base_static_value=0.66,
        priority_class="high",
    ),
    "8929a1d7573ffff": CellCatalogEntry(
        h3_cell="8929a1d7573ffff",
        lat=34.435,
        lon=-111.26,
        region_name="Arizona Rim",
        base_static_value=0.61,
        priority_class="high",
    ),
    "8928860e4cbffff": CellCatalogEntry(
        h3_cell="8928860e4cbffff",
        lat=44.2405,
        lon=-114.4788,
        region_name="Idaho Salmon-Challis",
        base_static_value=0.57,
        priority_class="medium",
    ),
    "89be0e35a13ffff": CellCatalogEntry(
        h3_cell="89be0e35a13ffff",
        lat=-23.512,
        lon=147.095,
        region_name="Central Queensland",
        base_static_value=0.62,
        priority_class="medium",
    ),
    "89be6d34a4fffff": CellCatalogEntry(
        h3_cell="89be6d34a4fffff",
        lat=-32.163,
        lon=149.122,
        region_name="NSW Central Ranges",
        base_static_value=0.56,
        priority_class="medium",
    ),
}


SATELLITE_PROFILES: dict[int, dict[str, Any]] = {
    25544: {
        "name": "ISS Zarya Tactical",
        "sensor": {
            "sensor_id": "sensor:iss-zarya-optical-v1",
            "swath_km": 52.0,
            "quality_nominal": 0.78,
            "max_off_nadir_deg": 45.0,
            "estimated_data_volume_mb": 640.0,
        },
        "downlink": {
            "buffer_capacity_mb": 48000.0,
            "nominal_downlink_rate_mbps": 240.0,
        },
        "constraints": {
            "max_retargets_per_orbit": 4,
            "availability": "nominal",
        },
    },
    40697: {
        "name": "Sentinel-2A Wildfire",
        "sensor": {
            "sensor_id": "sensor:sentinel-2a-optical-v1",
            "swath_km": 290.0,
            "quality_nominal": 0.92,
            "max_off_nadir_deg": 30.0,
            "estimated_data_volume_mb": 920.0,
        },
        "downlink": {
            "buffer_capacity_mb": 90000.0,
            "nominal_downlink_rate_mbps": 420.0,
        },
        "constraints": {
            "max_retargets_per_orbit": 2,
            "availability": "nominal",
        },
    },
    43013: {
        "name": "NOAA-20 Recon",
        "sensor": {
            "sensor_id": "sensor:noaa-20-recon-v1",
            "swath_km": 150.0,
            "quality_nominal": 0.86,
            "max_off_nadir_deg": 35.0,
            "estimated_data_volume_mb": 780.0,
        },
        "downlink": {
            "buffer_capacity_mb": 72000.0,
            "nominal_downlink_rate_mbps": 360.0,
        },
        "constraints": {
            "max_retargets_per_orbit": 3,
            "availability": "nominal",
        },
    },
}


GROUND_STATION_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "station_id": "gs:alaska-fairbanks",
        "name": "Fairbanks Alaska",
        "location": {"lat": 64.8378, "lon": -147.7164, "alt_m": 136.0},
        "capabilities": {
            "max_concurrent_contacts": 2,
            "downlink_rate_mbps": 700.0,
            "availability": "nominal",
        },
    },
    {
        "station_id": "gs:svalbard-longyearbyen",
        "name": "Longyearbyen Svalbard",
        "location": {"lat": 78.2232, "lon": 15.6469, "alt_m": 28.0},
        "capabilities": {
            "max_concurrent_contacts": 3,
            "downlink_rate_mbps": 760.0,
            "availability": "nominal",
        },
    },
    {
        "station_id": "gs:hawaii-kauai",
        "name": "Kauai Hawaii",
        "location": {"lat": 22.0964, "lon": -159.5261, "alt_m": 12.0},
        "capabilities": {
            "max_concurrent_contacts": 2,
            "downlink_rate_mbps": 520.0,
            "availability": "nominal",
        },
    },
    {
        "station_id": "gs:alice-springs",
        "name": "Alice Springs",
        "location": {"lat": -23.698, "lon": 133.8807, "alt_m": 545.0},
        "capabilities": {
            "max_concurrent_contacts": 2,
            "downlink_rate_mbps": 560.0,
            "availability": "nominal",
        },
    },
)


def builtin_phase1_recipes(benchmark_id: str) -> tuple[ScenarioRecipe, ...]:
    return (
        ScenarioRecipe(
            recipe_id="rcp:phase1:sparse-frontier:seed-101",
            benchmark_id=benchmark_id,
            family="sparse_frontier",
            seed=101,
            start_time_utc=datetime(2026, 4, 10, 0, 0, tzinfo=UTC),
            notes=(
                "Low-incident western frontier replay with one tempting SoCal "
                "target and long quiet intervals."
            ),
            demand_mode="fixture",
            incident_mode="fixture",
            weather_mode="fixture",
            fixture_target_point_ids=(
                "dp:socal-core",
                "dp:socal-valley",
                "dp:oregon-bend",
                "dp:alberta-fortmac",
            ),
            fixture_incident_template_ids=("fx:sparse-ca-001", "fx:sparse-or-001"),
            fixture_weather_profiles=(
                ("8928308280fffff", "clear_morning"),
                ("8928d54250bffff", "mountain_variable"),
                ("892f054b6d7ffff", "clear_morning"),
            ),
            target_count=3,
            incident_count=2,
            family_parameters={"dispersion": "wide", "contention": "low"},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase1:sparse-frontier:seed-102",
            benchmark_id=benchmark_id,
            family="sparse_frontier",
            seed=102,
            start_time_utc=datetime(2026, 4, 11, 0, 0, tzinfo=UTC),
            notes=(
                "Synthetic sparse scenario with broad Rocky Mountain dispersion "
                "and long ignition gaps."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928341aec3ffff",
                "8928d54250bffff",
                "88268cda81fffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
            ),
            target_count=4,
            incident_count=2,
            family_parameters={"dispersion": "wide", "synthetic_profile": "dry-frontier"},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase1:sparse-frontier:seed-103",
            benchmark_id=benchmark_id,
            family="sparse_frontier",
            seed=103,
            start_time_utc=datetime(2026, 4, 12, 0, 0, tzinfo=UTC),
            notes=(
                "Fixture-backed northern interior replay with quiet skies but long "
                "travel distances."
            ),
            demand_mode="fixture",
            incident_mode="fixture",
            weather_mode="fixture",
            fixture_target_point_ids=(
                "dp:bc-interior",
                "dp:colorado-front",
                "dp:idaho-salmon",
            ),
            fixture_incident_template_ids=("fx:sparse-bc-001", "fx:sparse-co-001"),
            fixture_weather_profiles=(
                ("892f5ab6e13ffff", "clear_morning"),
                ("88268cda81fffff", "mountain_variable"),
                ("8928860e4cbffff", "clear_morning"),
            ),
            target_count=3,
            incident_count=2,
            family_parameters={"dispersion": "northern", "contention": "low"},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase1:sparse-frontier:seed-104",
            benchmark_id=benchmark_id,
            family="sparse_frontier",
            seed=104,
            start_time_utc=datetime(2026, 4, 13, 0, 0, tzinfo=UTC),
            notes=(
                "Synthetic sparse scenario that spreads demand across Alberta, BC, "
                "Arizona, and Idaho."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "892f054b6d7ffff",
                "892f5ab6e13ffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
            ),
            target_count=4,
            incident_count=3,
            family_parameters={"dispersion": "continental", "synthetic_profile": "cool-frontier"},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase1:burst-outbreak:seed-201",
            benchmark_id=benchmark_id,
            family="burst_outbreak",
            seed=201,
            start_time_utc=datetime(2026, 4, 14, 0, 0, tzinfo=UTC),
            notes="Synthetic burst across the western US with a 75-minute ignition shock.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "8928d54250bffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
            ),
            target_count=5,
            incident_count=6,
            burst_window_minutes=75,
            family_parameters={"burst_profile": "rapid-west", "contention": "high"},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase1:burst-outbreak:seed-202",
            benchmark_id=benchmark_id,
            family="burst_outbreak",
            seed=202,
            start_time_utc=datetime(2026, 4, 15, 0, 0, tzinfo=UTC),
            notes=(
                "Fixture-backed outbreak with clustered west-coast ignitions and "
                "smoky variable cloud cover."
            ),
            demand_mode="fixture",
            incident_mode="fixture",
            weather_mode="fixture",
            fixture_target_point_ids=(
                "dp:norcal-foothills",
                "dp:norcal-ridge",
                "dp:oregon-bend",
                "dp:arizona-rim",
            ),
            fixture_incident_template_ids=(
                "fx:burst-west-001",
                "fx:burst-west-002",
                "fx:burst-west-003",
                "fx:burst-west-004",
                "fx:burst-west-005",
            ),
            fixture_weather_profiles=(
                ("8928341aec3ffff", "burst_smoke_mix"),
                ("8928d54250bffff", "burst_smoke_mix"),
                ("8929a1d7573ffff", "clear_morning"),
            ),
            target_count=3,
            incident_count=5,
            burst_window_minutes=80,
            family_parameters={"burst_profile": "historical-cluster", "contention": "high"},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase1:burst-outbreak:seed-203",
            benchmark_id=benchmark_id,
            family="burst_outbreak",
            seed=203,
            start_time_utc=datetime(2026, 4, 16, 0, 0, tzinfo=UTC),
            notes=(
                "Synthetic outbreak spanning Australian and North American cells "
                "to force triage under burst pressure."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "89be0e35a13ffff",
                "89be6d34a4fffff",
                "8928308280fffff",
                "8928341aec3ffff",
                "8929a1d7573ffff",
            ),
            target_count=5,
            incident_count=7,
            burst_window_minutes=90,
            family_parameters={"burst_profile": "cross-hemisphere", "contention": "high"},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase1:cloud-trap:seed-301",
            benchmark_id=benchmark_id,
            family="cloud_trap",
            seed=301,
            start_time_utc=datetime(2026, 4, 17, 0, 0, tzinfo=UTC),
            notes=(
                "Fixture-backed cloud trap over high-value California cells with "
                "early obstructed passes and later relief."
            ),
            demand_mode="fixture",
            incident_mode="fixture",
            weather_mode="fixture",
            fixture_target_point_ids=(
                "dp:socal-core",
                "dp:socal-valley",
                "dp:norcal-foothills",
                "dp:alberta-fortmac",
            ),
            fixture_incident_template_ids=("fx:cloudtrap-ca-001", "fx:cloudtrap-ca-002"),
            fixture_weather_profiles=(
                ("8928308280fffff", "cloud_trap_early"),
                ("8928341aec3ffff", "cloud_trap_stubborn"),
                ("892f054b6d7ffff", "clear_morning"),
            ),
            target_count=3,
            incident_count=2,
            family_parameters={"trap_window_hours": 12, "decision_punish_delay": True},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase1:cloud-trap:seed-302",
            benchmark_id=benchmark_id,
            family="cloud_trap",
            seed=302,
            start_time_utc=datetime(2026, 4, 18, 0, 0, tzinfo=UTC),
            notes=(
                "Synthetic cloud trap across northern forest cells with early heavy "
                "cloud and slow clearing."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "892f054b6d7ffff",
                "892f5ab6e13ffff",
                "8928d54250bffff",
                "88268cda81fffff",
            ),
            target_count=4,
            incident_count=3,
            family_parameters={"trap_window_hours": 14, "synthetic_profile": "slow-clearing"},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase1:cloud-trap:seed-303",
            benchmark_id=benchmark_id,
            family="cloud_trap",
            seed=303,
            start_time_utc=datetime(2026, 4, 19, 0, 0, tzinfo=UTC),
            notes=(
                "Fixture-backed mountain cloud trap with obscured early windows "
                "over Colorado and Arizona."
            ),
            demand_mode="fixture",
            incident_mode="fixture",
            weather_mode="fixture",
            fixture_target_point_ids=(
                "dp:colorado-front",
                "dp:arizona-rim",
                "dp:idaho-salmon",
            ),
            fixture_incident_template_ids=(
                "fx:cloudtrap-mountain-001",
                "fx:cloudtrap-mountain-002",
            ),
            fixture_weather_profiles=(
                ("88268cda81fffff", "cloud_trap_stubborn"),
                ("8929a1d7573ffff", "cloud_trap_early"),
                ("8928860e4cbffff", "mountain_variable"),
            ),
            target_count=3,
            incident_count=2,
            family_parameters={"trap_window_hours": 10, "terrain": "mountain"},
        ),
    )


def builtin_phase2_recipes(benchmark_id: str) -> tuple[ScenarioRecipe, ...]:
    return (
        ScenarioRecipe(
            recipe_id="rcp:phase2:sparse-frontier:seed-401",
            benchmark_id=benchmark_id,
            family="sparse_frontier",
            seed=401,
            start_time_utc=datetime(2026, 5, 1, 0, 0, tzinfo=UTC),
            notes="Sparse bootstrap scenario with quiet western cells and long decision gaps.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928d54250bffff",
                "88268cda81fffff",
                "892f054b6d7ffff",
            ),
            target_count=4,
            incident_count=2,
            family_parameters={"dispersion": "wide", "difficulty_tier": 1},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:sparse-frontier:seed-402",
            benchmark_id=benchmark_id,
            family="sparse_frontier",
            seed=402,
            start_time_utc=datetime(2026, 5, 2, 0, 0, tzinfo=UTC),
            notes="Low-contention bootstrap scenario spanning Alberta, Arizona, and Idaho.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "892f054b6d7ffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
                "892f5ab6e13ffff",
            ),
            target_count=4,
            incident_count=3,
            family_parameters={"dispersion": "continental", "difficulty_tier": 1},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:sparse-frontier:seed-403",
            benchmark_id=benchmark_id,
            family="sparse_frontier",
            seed=403,
            start_time_utc=datetime(2026, 5, 3, 0, 0, tzinfo=UTC),
            notes="Validation sparse frontier scenario with Rockies and California separation.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928341aec3ffff",
                "8928d54250bffff",
                "88268cda81fffff",
                "8928308280fffff",
            ),
            target_count=4,
            incident_count=3,
            family_parameters={"dispersion": "validation", "difficulty_tier": 1},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:sparse-frontier:seed-404",
            benchmark_id=benchmark_id,
            family="sparse_frontier",
            seed=404,
            start_time_utc=datetime(2026, 5, 4, 0, 0, tzinfo=UTC),
            notes=(
                "Held-out sparse frontier test scenario with quiet but "
                "geographically broad demand."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "892f054b6d7ffff",
                "892f5ab6e13ffff",
            ),
            target_count=4,
            incident_count=3,
            family_parameters={"dispersion": "heldout", "difficulty_tier": 1},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:burst-outbreak:seed-411",
            benchmark_id=benchmark_id,
            family="burst_outbreak",
            seed=411,
            start_time_utc=datetime(2026, 5, 5, 0, 0, tzinfo=UTC),
            notes="Training outbreak with dense western ignitions and strong urgency clustering.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "8928d54250bffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
            ),
            target_count=5,
            incident_count=7,
            burst_window_minutes=70,
            family_parameters={"contention": "high", "difficulty_tier": 2},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:burst-outbreak:seed-412",
            benchmark_id=benchmark_id,
            family="burst_outbreak",
            seed=412,
            start_time_utc=datetime(2026, 5, 6, 0, 0, tzinfo=UTC),
            notes=(
                "Training outbreak that spreads high urgency across both US "
                "and Australian cells."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "89be0e35a13ffff",
                "89be6d34a4fffff",
                "8928308280fffff",
                "8928341aec3ffff",
                "8929a1d7573ffff",
            ),
            target_count=5,
            incident_count=8,
            burst_window_minutes=85,
            family_parameters={"contention": "cross-hemisphere", "difficulty_tier": 2},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:burst-outbreak:seed-413",
            benchmark_id=benchmark_id,
            family="burst_outbreak",
            seed=413,
            start_time_utc=datetime(2026, 5, 7, 0, 0, tzinfo=UTC),
            notes="Validation outbreak scenario with concentrated West Coast incident bursts.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "8928d54250bffff",
                "8928860e4cbffff",
                "8929a1d7573ffff",
            ),
            target_count=5,
            incident_count=8,
            burst_window_minutes=80,
            family_parameters={"contention": "validation", "difficulty_tier": 2},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:burst-outbreak:seed-414",
            benchmark_id=benchmark_id,
            family="burst_outbreak",
            seed=414,
            start_time_utc=datetime(2026, 5, 8, 0, 0, tzinfo=UTC),
            notes="Held-out outbreak test scenario with sustained incident pressure.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "88268cda81fffff",
                "8929a1d7573ffff",
                "892f054b6d7ffff",
            ),
            target_count=5,
            incident_count=9,
            burst_window_minutes=95,
            family_parameters={"contention": "heldout", "difficulty_tier": 2},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:cloud-trap:seed-421",
            benchmark_id=benchmark_id,
            family="cloud_trap",
            seed=421,
            start_time_utc=datetime(2026, 5, 9, 0, 0, tzinfo=UTC),
            notes="Training cloud-trap scenario with slow relief on high-value western cells.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "8928d54250bffff",
                "88268cda81fffff",
            ),
            target_count=4,
            incident_count=4,
            family_parameters={"trap_window_hours": 12, "difficulty_tier": 2},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:cloud-trap:seed-422",
            benchmark_id=benchmark_id,
            family="cloud_trap",
            seed=422,
            start_time_utc=datetime(2026, 5, 10, 0, 0, tzinfo=UTC),
            notes=(
                "Training cloud-trap scenario with northern forests "
                "clearing later than expected."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "892f054b6d7ffff",
                "892f5ab6e13ffff",
                "8928d54250bffff",
                "88268cda81fffff",
            ),
            target_count=4,
            incident_count=4,
            family_parameters={"trap_window_hours": 14, "difficulty_tier": 2},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:cloud-trap:seed-423",
            benchmark_id=benchmark_id,
            family="cloud_trap",
            seed=423,
            start_time_utc=datetime(2026, 5, 11, 0, 0, tzinfo=UTC),
            notes="Validation cloud-trap scenario with delayed clean passes on mixed terrain.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
                "88268cda81fffff",
            ),
            target_count=4,
            incident_count=4,
            family_parameters={"trap_window_hours": 10, "difficulty_tier": 2},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:cloud-trap:seed-424",
            benchmark_id=benchmark_id,
            family="cloud_trap",
            seed=424,
            start_time_utc=datetime(2026, 5, 12, 0, 0, tzinfo=UTC),
            notes="Held-out cloud-trap test scenario with stubborn early cloud over California.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
            ),
            target_count=4,
            incident_count=5,
            family_parameters={"trap_window_hours": 13, "difficulty_tier": 2},
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:downlink-crunch:seed-431",
            benchmark_id=benchmark_id,
            family="downlink_crunch",
            seed=431,
            start_time_utc=datetime(2026, 5, 13, 0, 0, tzinfo=UTC),
            notes=(
                "Training downlink crunch with reduced station rates "
                "and elevated observation load."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "8928d54250bffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
            ),
            target_count=5,
            incident_count=8,
            burst_window_minutes=60,
            family_parameters={
                "difficulty_tier": 3,
                "station_downlink_rate_scale_by_id": {
                    "gs:alaska-fairbanks": 0.72,
                    "gs:hawaii-kauai": 0.68,
                    "gs:alice-springs": 0.7,
                },
                "station_availability_overrides": {
                    "gs:hawaii-kauai": "degraded",
                    "gs:alice-springs": "degraded",
                },
                "satellite_buffer_scale_by_id": {
                    "sat:norad-40697:sentinel-2a": 0.75,
                    "sat:norad-43013:noaa-20": 0.78,
                },
            },
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:downlink-crunch:seed-432",
            benchmark_id=benchmark_id,
            family="downlink_crunch",
            seed=432,
            start_time_utc=datetime(2026, 5, 14, 0, 0, tzinfo=UTC),
            notes=(
                "Training downlink crunch with high-volume observations "
                "and limited contact relief."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "892f054b6d7ffff",
                "892f5ab6e13ffff",
                "88268cda81fffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
            ),
            target_count=5,
            incident_count=8,
            burst_window_minutes=65,
            family_parameters={
                "difficulty_tier": 3,
                "station_downlink_rate_scale_by_id": {
                    "gs:svalbard-longyearbyen": 0.7,
                    "gs:alaska-fairbanks": 0.74,
                    "gs:hawaii-kauai": 0.7,
                },
                "station_availability_overrides": {
                    "gs:hawaii-kauai": "degraded",
                },
                "satellite_buffer_scale_by_id": {
                    "sat:norad-40697:sentinel-2a": 0.72,
                    "sat:norad-25544:iss--zarya": 0.8,
                },
            },
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:downlink-crunch:seed-433",
            benchmark_id=benchmark_id,
            family="downlink_crunch",
            seed=433,
            start_time_utc=datetime(2026, 5, 15, 0, 0, tzinfo=UTC),
            notes="Validation downlink crunch scenario with tight contact windows and lower rates.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "892f054b6d7ffff",
                "892f5ab6e13ffff",
                "88268cda81fffff",
            ),
            target_count=5,
            incident_count=8,
            burst_window_minutes=70,
            family_parameters={
                "difficulty_tier": 3,
                "station_downlink_rate_scale_by_id": {
                    "gs:alaska-fairbanks": 0.68,
                    "gs:svalbard-longyearbyen": 0.72,
                },
                "station_availability_overrides": {
                    "gs:alice-springs": "degraded",
                },
                "satellite_buffer_scale_by_id": {
                    "sat:norad-43013:noaa-20": 0.74,
                },
            },
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:downlink-crunch:seed-434",
            benchmark_id=benchmark_id,
            family="downlink_crunch",
            seed=434,
            start_time_utc=datetime(2026, 5, 16, 0, 0, tzinfo=UTC),
            notes=(
                "Held-out downlink crunch test scenario with broad "
                "targets and constrained downlink."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "8928d54250bffff",
                "88268cda81fffff",
                "892f054b6d7ffff",
            ),
            target_count=5,
            incident_count=9,
            burst_window_minutes=70,
            family_parameters={
                "difficulty_tier": 3,
                "station_downlink_rate_scale_by_id": {
                    "gs:hawaii-kauai": 0.66,
                    "gs:alice-springs": 0.68,
                },
                "station_availability_overrides": {
                    "gs:hawaii-kauai": "degraded",
                    "gs:alice-springs": "degraded",
                },
                "satellite_buffer_scale_by_id": {
                    "sat:norad-40697:sentinel-2a": 0.7,
                    "sat:norad-43013:noaa-20": 0.72,
                },
            },
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:station-outage:seed-441",
            benchmark_id=benchmark_id,
            family="station_outage",
            seed=441,
            start_time_utc=datetime(2026, 5, 17, 0, 0, tzinfo=UTC),
            notes="OOD station outage scenario with one station offline and another degraded.",
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "8929a1d7573ffff",
                "88268cda81fffff",
            ),
            target_count=4,
            incident_count=6,
            family_parameters={
                "difficulty_tier": 4,
                "station_availability_overrides": {
                    "gs:hawaii-kauai": "offline",
                    "gs:alice-springs": "degraded",
                },
                "station_downlink_rate_scale_by_id": {
                    "gs:alice-springs": 0.58,
                },
            },
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:station-outage:seed-442",
            benchmark_id=benchmark_id,
            family="station_outage",
            seed=442,
            start_time_utc=datetime(2026, 5, 18, 0, 0, tzinfo=UTC),
            notes=(
                "OOD station outage scenario with polar coverage loss "
                "and degraded fallback links."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "892f054b6d7ffff",
                "892f5ab6e13ffff",
                "8928d54250bffff",
                "88268cda81fffff",
            ),
            target_count=4,
            incident_count=6,
            family_parameters={
                "difficulty_tier": 4,
                "station_availability_overrides": {
                    "gs:svalbard-longyearbyen": "offline",
                    "gs:alaska-fairbanks": "degraded",
                },
                "station_downlink_rate_scale_by_id": {
                    "gs:alaska-fairbanks": 0.6,
                },
            },
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:constellation-degradation:seed-451",
            benchmark_id=benchmark_id,
            family="constellation_degradation",
            seed=451,
            start_time_utc=datetime(2026, 5, 19, 0, 0, tzinfo=UTC),
            notes=(
                "OOD degraded constellation scenario with one satellite "
                "removed and another degraded."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "8928308280fffff",
                "8928341aec3ffff",
                "8928d54250bffff",
                "8929a1d7573ffff",
                "8928860e4cbffff",
            ),
            target_count=5,
            incident_count=7,
            family_parameters={
                "difficulty_tier": 4,
                "excluded_satellite_ids": ["sat:norad-25544:iss--zarya"],
                "satellite_availability_overrides": {
                    "sat:norad-40697:sentinel-2a": "degraded",
                },
                "satellite_quality_scale_by_id": {
                    "sat:norad-40697:sentinel-2a": 0.72,
                    "sat:norad-43013:noaa-20": 0.84,
                },
                "satellite_downlink_rate_scale_by_id": {
                    "sat:norad-40697:sentinel-2a": 0.74,
                    "sat:norad-43013:noaa-20": 0.82,
                },
                "satellite_buffer_scale_by_id": {
                    "sat:norad-40697:sentinel-2a": 0.76,
                },
            },
        ),
        ScenarioRecipe(
            recipe_id="rcp:phase2:constellation-degradation:seed-452",
            benchmark_id=benchmark_id,
            family="constellation_degradation",
            seed=452,
            start_time_utc=datetime(2026, 5, 20, 0, 0, tzinfo=UTC),
            notes=(
                "OOD degraded constellation scenario with lower sensor "
                "quality and slower downlink."
            ),
            demand_mode="synthetic",
            incident_mode="synthetic",
            weather_mode="synthetic",
            candidate_h3_cells=(
                "892f054b6d7ffff",
                "892f5ab6e13ffff",
                "88268cda81fffff",
                "8928860e4cbffff",
                "8929a1d7573ffff",
            ),
            target_count=5,
            incident_count=7,
            family_parameters={
                "difficulty_tier": 4,
                "excluded_satellite_ids": ["sat:norad-43013:noaa-20"],
                "satellite_availability_overrides": {
                    "sat:norad-40697:sentinel-2a": "degraded",
                },
                "satellite_quality_scale_by_id": {
                    "sat:norad-40697:sentinel-2a": 0.75,
                    "sat:norad-25544:iss--zarya": 0.82,
                },
                "satellite_downlink_rate_scale_by_id": {
                    "sat:norad-40697:sentinel-2a": 0.7,
                    "sat:norad-25544:iss--zarya": 0.8,
                },
                "satellite_buffer_scale_by_id": {
                    "sat:norad-40697:sentinel-2a": 0.74,
                    "sat:norad-25544:iss--zarya": 0.82,
                },
            },
        ),
    )
