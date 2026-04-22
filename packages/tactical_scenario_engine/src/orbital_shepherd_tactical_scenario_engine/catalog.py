from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from orbital_shepherd_contracts import IncidentPacket
from orbital_shepherd_core import stable_id


@dataclass(frozen=True, slots=True)
class UnitTemplate:
    role: str
    unit_type: str
    travel_mode: str
    home_label: str
    personnel_count: int
    equipment_capacity: float


@dataclass(frozen=True, slots=True)
class ScoutTemplate:
    asset_type: str
    status: str
    home_label: str
    endurance_minutes: int
    sensor_focus: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TacticalFamilySpec:
    family_id: str
    display_name: str
    description: str
    planning_horizon_minutes: int
    decision_interval_seconds: int
    max_active_routes: int
    reroute_on_blockage: bool
    requested_capabilities: tuple[str, ...]
    operational_objectives: tuple[str, ...]
    base_area_ha: float
    severity_floor: float
    overlay_focus: str
    unit_templates: tuple[UnitTemplate, ...]
    scout_templates: tuple[ScoutTemplate, ...] = ()


@dataclass(frozen=True, slots=True)
class TacticalScenarioRecipe:
    recipe_id: str
    benchmark_id: str
    family_id: str
    seed: int
    packet: IncidentPacket
    region_bundle_path: Path
    incident_anchor_index: int
    activation_delay_seconds: int
    family_parameters: dict[str, Any] = field(default_factory=dict)


FIXTURE_TARGET_CELL_IDS: tuple[str, ...] = (
    "tc:8928308280fffff",
    "tc:8928341aec3ffff",
    "tc:8928d54250bffff",
)


FAMILY_SPECS: dict[str, TacticalFamilySpec] = {
    "foothill_access": TacticalFamilySpec(
        family_id="foothill_access",
        display_name="Foothill Access",
        description="Initial-attack access into steep foothill terrain with narrow approaches.",
        planning_horizon_minutes=240,
        decision_interval_seconds=300,
        max_active_routes=4,
        reroute_on_blockage=True,
        requested_capabilities=("initial_attack", "road_dispatch", "line_construction"),
        operational_objectives=(
            "secure a reliable uphill access route",
            "stage engines where water resupply remains viable",
            "keep command reachable if access roads degrade",
        ),
        base_area_ha=42.0,
        severity_floor=0.72,
        overlay_focus="temporary_penalty",
        unit_templates=(
            UnitTemplate("engine_alpha", "engine", "road", "station", 4, 1500.0),
            UnitTemplate("engine_bravo", "engine", "road", "station", 4, 1400.0),
            UnitTemplate("crew_hill", "crew", "road", "station", 6, 320.0),
            UnitTemplate("dozer_cut", "dozer", "road", "staging", 2, 1.0),
            UnitTemplate("command_post", "command", "road", "command", 3, 0.0),
            UnitTemplate("heli_lift", "helicopter", "air", "helibase", 5, 900.0),
        ),
        scout_templates=(
            ScoutTemplate("lookout", "available", "command", 360, ("slope_watch",)),
        ),
    ),
    "urban_interface": TacticalFamilySpec(
        family_id="urban_interface",
        display_name="Urban Interface",
        description="WUI structure protection with dense assets and conflicting access priorities.",
        planning_horizon_minutes=210,
        decision_interval_seconds=180,
        max_active_routes=5,
        reroute_on_blockage=True,
        requested_capabilities=("structure_protection", "road_dispatch", "air_recon"),
        operational_objectives=(
            "protect population-facing assets along the interface",
            "keep the command post outside the highest ember exposure band",
            "maintain a recon orbit over threatened structure clusters",
        ),
        base_area_ha=28.0,
        severity_floor=0.75,
        overlay_focus="risk_zone",
        unit_templates=(
            UnitTemplate("engine_alpha", "engine", "road", "station", 4, 1500.0),
            UnitTemplate("engine_bravo", "engine", "road", "station", 4, 1500.0),
            UnitTemplate("engine_charlie", "engine", "road", "staging", 4, 1400.0),
            UnitTemplate("crew_interface", "crew", "road", "station", 6, 280.0),
            UnitTemplate("crew_structure", "crew", "road", "staging", 5, 260.0),
            UnitTemplate("command_post", "command", "road", "command", 4, 0.0),
            UnitTemplate("heli_watch", "helicopter", "air", "helibase", 5, 850.0),
        ),
        scout_templates=(
            ScoutTemplate(
                "drone",
                "available",
                "command",
                95,
                ("ember_front", "structure_triage"),
            ),
        ),
    ),
    "closure_cascade": TacticalFamilySpec(
        family_id="closure_cascade",
        display_name="Closure Cascade",
        description="Sequential access closures force reroutes and create fragile fallback routes.",
        planning_horizon_minutes=300,
        decision_interval_seconds=300,
        max_active_routes=4,
        reroute_on_blockage=True,
        requested_capabilities=("road_dispatch", "detour_management", "contingency_planning"),
        operational_objectives=(
            "retain at least one viable route to the drop point",
            "avoid committing the full engine roster onto a single corridor",
            "stage dozer support where detours remain reachable",
        ),
        base_area_ha=55.0,
        severity_floor=0.7,
        overlay_focus="closure",
        unit_templates=(
            UnitTemplate("engine_alpha", "engine", "road", "station", 4, 1500.0),
            UnitTemplate("engine_bravo", "engine", "road", "station", 4, 1450.0),
            UnitTemplate("crew_detour", "crew", "road", "station", 6, 310.0),
            UnitTemplate("dozer_cut", "dozer", "road", "staging", 2, 1.0),
            UnitTemplate("command_post", "command", "road", "command", 3, 0.0),
        ),
    ),
    "depot_saturation": TacticalFamilySpec(
        family_id="depot_saturation",
        display_name="Depot Saturation",
        description="Many units share a small depot footprint, causing queueing and launch delays.",
        planning_horizon_minutes=240,
        decision_interval_seconds=240,
        max_active_routes=6,
        reroute_on_blockage=False,
        requested_capabilities=("dispatch_surging", "staging_management", "command_coordination"),
        operational_objectives=(
            "deconflict departures from the overloaded depot apron",
            "keep reserve engines staged without gridlocking the access road",
            "preserve outbound movement for heavy equipment",
        ),
        base_area_ha=34.0,
        severity_floor=0.68,
        overlay_focus="temporary_penalty",
        unit_templates=(
            UnitTemplate("engine_alpha", "engine", "road", "station", 4, 1500.0),
            UnitTemplate("engine_bravo", "engine", "road", "station", 4, 1500.0),
            UnitTemplate("engine_charlie", "engine", "road", "station", 4, 1450.0),
            UnitTemplate("engine_delta", "engine", "road", "station", 4, 1450.0),
            UnitTemplate("crew_primary", "crew", "road", "station", 6, 300.0),
            UnitTemplate("crew_reserve", "crew", "road", "staging", 5, 260.0),
            UnitTemplate("dozer_cut", "dozer", "road", "staging", 2, 1.0),
            UnitTemplate("command_post", "command", "road", "command", 4, 0.0),
            UnitTemplate("heli_watch", "helicopter", "air", "helibase", 5, 850.0),
        ),
    ),
    "smoke_corridor": TacticalFamilySpec(
        family_id="smoke_corridor",
        display_name="Smoke Corridor",
        description="Long smoke-filled access corridors degrade travel speed and recon quality.",
        planning_horizon_minutes=270,
        decision_interval_seconds=240,
        max_active_routes=5,
        reroute_on_blockage=True,
        requested_capabilities=("air_support", "road_dispatch", "visibility_management"),
        operational_objectives=(
            "keep at least one low-smoke ingress corridor open",
            "balance air support with slow-moving ground access",
            "avoid overcommitting crews into the densest smoke band",
        ),
        base_area_ha=61.0,
        severity_floor=0.77,
        overlay_focus="risk_zone",
        unit_templates=(
            UnitTemplate("engine_alpha", "engine", "road", "station", 4, 1500.0),
            UnitTemplate("engine_bravo", "engine", "road", "station", 4, 1450.0),
            UnitTemplate("crew_smoke", "crew", "road", "station", 6, 300.0),
            UnitTemplate("command_post", "command", "road", "command", 3, 0.0),
            UnitTemplate("heli_watch", "helicopter", "air", "helibase", 5, 900.0),
            UnitTemplate("tanker_line", "air_tanker", "air", "helibase", 2, 3200.0),
        ),
        scout_templates=(
            ScoutTemplate(
                "drone",
                "available",
                "command",
                75,
                ("visibility_lane", "smoke_column"),
            ),
        ),
    ),
    "drone_scout_gap": TacticalFamilySpec(
        family_id="drone_scout_gap",
        display_name="Drone Scout Gap",
        description="Ground dispatch proceeds with degraded scout coverage and stale recon.",
        planning_horizon_minutes=240,
        decision_interval_seconds=300,
        max_active_routes=4,
        reroute_on_blockage=True,
        requested_capabilities=("ground_dispatch", "manual_recon", "coverage_gap_mitigation"),
        operational_objectives=(
            "cover the blind scout sector with ground lookouts",
            "avoid committing engines beyond the last confirmed recon fix",
            "hold a reserve path until visibility is re-established",
        ),
        base_area_ha=39.0,
        severity_floor=0.69,
        overlay_focus="risk_zone",
        unit_templates=(
            UnitTemplate("engine_alpha", "engine", "road", "station", 4, 1500.0),
            UnitTemplate("engine_bravo", "engine", "road", "station", 4, 1400.0),
            UnitTemplate("crew_gap", "crew", "road", "station", 6, 290.0),
            UnitTemplate("command_post", "command", "road", "command", 3, 0.0),
            UnitTemplate("heli_watch", "helicopter", "air", "helibase", 5, 850.0),
        ),
        scout_templates=(
            ScoutTemplate(
                "drone",
                "offline",
                "command",
                60,
                ("offline_recon",),
            ),
            ScoutTemplate("lookout", "available", "command", 360, ("manual_recon",)),
        ),
    ),
}


def builtin_phase3_recipes(
    benchmark_id: str,
    *,
    region_bundle_path: Path,
) -> tuple[TacticalScenarioRecipe, ...]:
    recipes: list[TacticalScenarioRecipe] = []
    base_time = datetime(2026, 8, 14, 3, 30, tzinfo=UTC)
    family_offsets = {
        "foothill_access": 0,
        "urban_interface": 1,
        "closure_cascade": 2,
        "depot_saturation": 3,
        "smoke_corridor": 4,
        "drone_scout_gap": 5,
    }
    family_seed_bases = {
        "foothill_access": 4100,
        "urban_interface": 4200,
        "closure_cascade": 4300,
        "depot_saturation": 4400,
        "smoke_corridor": 4500,
        "drone_scout_gap": 4600,
    }
    urgency_variants = (0.0, 0.04, -0.03, 0.07)
    confidence_variants = (0.02, -0.01, 0.03, 0.0)
    value_variants = (0.03, 0.0, 0.05, 0.08)
    delay_variants = (120, 180, 240, 150)

    for family_id, spec in FAMILY_SPECS.items():
        family_offset = family_offsets[family_id]
        seed_base = family_seed_bases[family_id]
        for variant_index in range(4):
            seed = seed_base + variant_index + 1
            target_cell_id = FIXTURE_TARGET_CELL_IDS[(family_offset + variant_index) % 3]
            observation_time = base_time + timedelta(days=family_offset, minutes=variant_index * 17)
            downlink_time = observation_time + timedelta(minutes=11 + family_offset + variant_index)
            urgency = min(max(spec.severity_floor + urgency_variants[variant_index], 0.0), 0.99)
            confidence = min(max(0.8 + confidence_variants[variant_index], 0.0), 0.98)
            downstream_value = round(0.78 + value_variants[variant_index], 3)
            incident_id = stable_id("inc", benchmark_id, family_id, f"seed-{seed}")
            packet = IncidentPacket(
                packet_id=stable_id("pkt", incident_id, "0001"),
                incident_id=incident_id,
                target_cell_id=target_cell_id,
                observation_time_utc=observation_time,
                downlink_time_utc=downlink_time,
                confidence=round(confidence, 3),
                urgency_score=round(urgency, 3),
                recommended_action="dispatch_ground",
                observation_opportunity_id=stable_id(
                    "opp",
                    family_id,
                    target_cell_id.replace(":", "-"),
                    f"seed-{seed}",
                ),
                downstream_value_estimate=downstream_value,
                summary=(
                    f"{spec.display_name} packet variant {variant_index + 1} for deterministic "
                    "tactical compilation."
                ),
            )
            recipes.append(
                TacticalScenarioRecipe(
                    recipe_id=stable_id("tr", benchmark_id, family_id, f"seed-{seed}"),
                    benchmark_id=benchmark_id,
                    family_id=family_id,
                    seed=seed,
                    packet=packet,
                    region_bundle_path=region_bundle_path,
                    incident_anchor_index=(family_offset + variant_index) % 4,
                    activation_delay_seconds=delay_variants[variant_index] + family_offset * 15,
                    family_parameters={
                        "variant_index": variant_index,
                        "family_display_name": spec.display_name,
                    },
                )
            )
    return tuple(recipes)
