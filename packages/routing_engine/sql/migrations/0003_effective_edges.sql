CREATE OR REPLACE FUNCTION routing.effective_edges(
    in_region_bundle_id text,
    in_overlay_ids text[] DEFAULT ARRAY[]::text[],
    in_effective_at_utc timestamptz DEFAULT CURRENT_TIMESTAMP
) RETURNS TABLE (
    edge_index integer,
    edge_id text,
    source_vertex integer,
    target_vertex integer,
    source_node_id text,
    target_node_id text,
    distance_m double precision,
    speed_kph double precision,
    base_travel_time_seconds double precision,
    blocked boolean,
    risk_multiplier double precision,
    restriction_multiplier double precision,
    delay_seconds double precision,
    min_speed_cap_kph double precision,
    effective_cost_seconds double precision
) LANGUAGE sql STABLE AS $$
WITH active_overlays AS (
    SELECT overlay_id, overlay_kind
    FROM routing.overlay_sets
    WHERE region_bundle_id = in_region_bundle_id
      AND COALESCE(array_length(in_overlay_ids, 1), 0) > 0
      AND overlay_id = ANY(in_overlay_ids)
      AND (starts_at_utc IS NULL OR starts_at_utc <= in_effective_at_utc)
      AND (ends_at_utc IS NULL OR ends_at_utc >= in_effective_at_utc)
),
closure_effects AS (
    SELECT edge_id, bool_or(is_closed) AS blocked
    FROM routing.overlay_edge_closures
    WHERE region_bundle_id = in_region_bundle_id
      AND overlay_id IN (
          SELECT overlay_id
          FROM active_overlays
          WHERE overlay_kind = 'closure'
      )
    GROUP BY edge_id
),
risk_effects AS (
    SELECT edge_id, COALESCE(EXP(SUM(LN(cost_multiplier))), 1.0) AS risk_multiplier
    FROM routing.overlay_edge_risk_multipliers
    WHERE region_bundle_id = in_region_bundle_id
      AND overlay_id IN (
          SELECT overlay_id
          FROM active_overlays
          WHERE overlay_kind = 'risk_multiplier'
      )
    GROUP BY edge_id
),
restriction_effects AS (
    SELECT
        edge_id,
        COALESCE(EXP(SUM(LN(cost_multiplier))), 1.0) AS restriction_multiplier,
        COALESCE(SUM(delay_seconds), 0.0) AS delay_seconds,
        MIN(speed_cap_kph) FILTER (WHERE speed_cap_kph IS NOT NULL) AS min_speed_cap_kph
    FROM routing.overlay_edge_temporary_restrictions
    WHERE region_bundle_id = in_region_bundle_id
      AND overlay_id IN (
          SELECT overlay_id
          FROM active_overlays
          WHERE overlay_kind = 'temporary_restriction'
      )
    GROUP BY edge_id
)
SELECT
    edges.edge_index,
    edges.edge_id,
    edges.source_vertex,
    edges.target_vertex,
    edges.source_node_id,
    edges.target_node_id,
    edges.distance_m,
    edges.speed_kph,
    edges.base_travel_time_seconds,
    COALESCE(closure_effects.blocked, false) AS blocked,
    COALESCE(risk_effects.risk_multiplier, 1.0) AS risk_multiplier,
    COALESCE(restriction_effects.restriction_multiplier, 1.0) AS restriction_multiplier,
    COALESCE(restriction_effects.delay_seconds, 0.0) AS delay_seconds,
    restriction_effects.min_speed_cap_kph,
    CASE
        WHEN COALESCE(closure_effects.blocked, false) THEN -1.0
        ELSE ROUND(
            (
                (
                    CASE
                        WHEN restriction_effects.min_speed_cap_kph IS NULL
                            THEN edges.base_travel_time_seconds
                        ELSE (
                            edges.distance_m
                            / (
                                LEAST(edges.speed_kph, restriction_effects.min_speed_cap_kph)
                                * 1000.0
                                / 3600.0
                            )
                        ) + regions.intersection_penalty_seconds
                    END
                    + COALESCE(restriction_effects.delay_seconds, 0.0)
                )
                * COALESCE(restriction_effects.restriction_multiplier, 1.0)
                * COALESCE(risk_effects.risk_multiplier, 1.0)
            )::numeric,
            6
        )::double precision
    END AS effective_cost_seconds
FROM routing.edges AS edges
JOIN routing.regions
    ON regions.region_bundle_id = edges.region_bundle_id
LEFT JOIN closure_effects
    ON closure_effects.edge_id = edges.edge_id
LEFT JOIN risk_effects
    ON risk_effects.edge_id = edges.edge_id
LEFT JOIN restriction_effects
    ON restriction_effects.edge_id = edges.edge_id
WHERE edges.region_bundle_id = in_region_bundle_id
ORDER BY edges.edge_index;
$$;

CREATE OR REPLACE FUNCTION routing.effective_edges_pgr_sql(
    in_region_bundle_id text,
    in_overlay_ids text[] DEFAULT ARRAY[]::text[],
    in_effective_at_utc timestamptz DEFAULT CURRENT_TIMESTAMP
) RETURNS text LANGUAGE sql STABLE AS $$
SELECT format(
    $format$
    SELECT
        edge_index AS id,
        source_vertex AS source,
        target_vertex AS target,
        effective_cost_seconds AS cost,
        -1::double precision AS reverse_cost
    FROM routing.effective_edges(%L, %L::text[], %L::timestamptz)
    $format$,
    in_region_bundle_id,
    in_overlay_ids,
    in_effective_at_utc
);
$$;

