CREATE SCHEMA IF NOT EXISTS routing;

CREATE TABLE IF NOT EXISTS routing.regions (
    region_bundle_id text PRIMARY KEY,
    region_manifest_id text NOT NULL,
    region_id text NOT NULL,
    region_name text NOT NULL,
    bundle_fingerprint text NOT NULL CHECK (bundle_fingerprint ~ '^[a-f0-9]{64}$'),
    bounds geometry(Polygon, 4326) NOT NULL,
    h3_resolution integer NOT NULL,
    h3_cell_ids text[] NOT NULL DEFAULT ARRAY[]::text[],
    travel_time_defaults jsonb NOT NULL,
    default_speed_kph double precision NOT NULL CHECK (default_speed_kph > 0),
    intersection_penalty_seconds double precision NOT NULL CHECK (intersection_penalty_seconds >= 0),
    traversable_node_count integer NOT NULL CHECK (traversable_node_count >= 0),
    traversable_edge_count integer NOT NULL CHECK (traversable_edge_count >= 0),
    compilation jsonb NOT NULL,
    provenance_notes jsonb NOT NULL DEFAULT '[]'::jsonb,
    ingested_at_utc timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS routing_regions_bounds_gix
    ON routing.regions
    USING gist (bounds);

CREATE TABLE IF NOT EXISTS routing.spatial_ingests (
    region_bundle_id text NOT NULL REFERENCES routing.regions(region_bundle_id) ON DELETE CASCADE,
    spatial_ingest_id text NOT NULL,
    region_id text NOT NULL,
    source_name text NOT NULL,
    layer_type text NOT NULL,
    source_uri text NOT NULL,
    source_fingerprint text NOT NULL,
    transform_fingerprint text NOT NULL,
    ingest_time_utc timestamptz NOT NULL,
    coverage_bounds geometry(Polygon, 4326) NOT NULL,
    feature_count integer NOT NULL CHECK (feature_count >= 0),
    notes text,
    manifest jsonb NOT NULL,
    PRIMARY KEY (region_bundle_id, spatial_ingest_id)
);

CREATE INDEX IF NOT EXISTS routing_spatial_ingests_bounds_gix
    ON routing.spatial_ingests
    USING gist (coverage_bounds);

CREATE TABLE IF NOT EXISTS routing.nodes (
    region_bundle_id text NOT NULL REFERENCES routing.regions(region_bundle_id) ON DELETE CASCADE,
    vertex_id integer NOT NULL CHECK (vertex_id > 0),
    node_id text NOT NULL,
    geom geometry(Point, 4326) NOT NULL,
    PRIMARY KEY (region_bundle_id, vertex_id),
    UNIQUE (region_bundle_id, node_id)
);

CREATE INDEX IF NOT EXISTS routing_nodes_geom_gix
    ON routing.nodes
    USING gist (geom);

CREATE TABLE IF NOT EXISTS routing.edges (
    region_bundle_id text NOT NULL REFERENCES routing.regions(region_bundle_id) ON DELETE CASCADE,
    edge_index integer NOT NULL CHECK (edge_index > 0),
    edge_id text NOT NULL,
    source_node_id text NOT NULL,
    target_node_id text NOT NULL,
    source_vertex integer NOT NULL,
    target_vertex integer NOT NULL,
    source_ingest_id text NOT NULL,
    road_class text NOT NULL,
    distance_m double precision NOT NULL CHECK (distance_m > 0),
    speed_kph double precision NOT NULL CHECK (speed_kph > 0),
    base_travel_time_seconds double precision NOT NULL CHECK (base_travel_time_seconds > 0),
    oneway boolean NOT NULL,
    road_name text,
    source_edge_ref text,
    geom geometry(LineString, 4326) NOT NULL,
    PRIMARY KEY (region_bundle_id, edge_index),
    UNIQUE (region_bundle_id, edge_id),
    FOREIGN KEY (region_bundle_id, source_node_id)
        REFERENCES routing.nodes(region_bundle_id, node_id)
        ON DELETE CASCADE,
    FOREIGN KEY (region_bundle_id, target_node_id)
        REFERENCES routing.nodes(region_bundle_id, node_id)
        ON DELETE CASCADE,
    FOREIGN KEY (region_bundle_id, source_vertex)
        REFERENCES routing.nodes(region_bundle_id, vertex_id)
        ON DELETE CASCADE,
    FOREIGN KEY (region_bundle_id, target_vertex)
        REFERENCES routing.nodes(region_bundle_id, vertex_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS routing_edges_source_vertex_idx
    ON routing.edges (region_bundle_id, source_vertex);

CREATE INDEX IF NOT EXISTS routing_edges_target_vertex_idx
    ON routing.edges (region_bundle_id, target_vertex);

CREATE INDEX IF NOT EXISTS routing_edges_geom_gix
    ON routing.edges
    USING gist (geom);

CREATE TABLE IF NOT EXISTS routing.facilities (
    region_bundle_id text NOT NULL REFERENCES routing.regions(region_bundle_id) ON DELETE CASCADE,
    facility_id text NOT NULL,
    facility_name text NOT NULL,
    facility_type text NOT NULL,
    availability text NOT NULL,
    capacity_units integer NOT NULL CHECK (capacity_units >= 0),
    supported_unit_types text[] NOT NULL DEFAULT ARRAY[]::text[],
    geom geometry(Point, 4326) NOT NULL,
    nearest_node_id text NOT NULL,
    nearest_vertex_id integer NOT NULL,
    nearest_distance_m double precision NOT NULL CHECK (nearest_distance_m >= 0),
    approach_time_seconds double precision NOT NULL CHECK (approach_time_seconds >= 0),
    payload jsonb NOT NULL,
    PRIMARY KEY (region_bundle_id, facility_id),
    FOREIGN KEY (region_bundle_id, nearest_node_id)
        REFERENCES routing.nodes(region_bundle_id, node_id)
        ON DELETE CASCADE,
    FOREIGN KEY (region_bundle_id, nearest_vertex_id)
        REFERENCES routing.nodes(region_bundle_id, vertex_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS routing_facilities_geom_gix
    ON routing.facilities
    USING gist (geom);

CREATE INDEX IF NOT EXISTS routing_facilities_nearest_vertex_idx
    ON routing.facilities (region_bundle_id, nearest_vertex_id);

CREATE TABLE IF NOT EXISTS routing.asset_layers (
    region_bundle_id text NOT NULL REFERENCES routing.regions(region_bundle_id) ON DELETE CASCADE,
    asset_layer_id text NOT NULL,
    layer_name text NOT NULL,
    asset_kind text NOT NULL,
    geometry_type text NOT NULL,
    feature_count integer NOT NULL CHECK (feature_count >= 0),
    PRIMARY KEY (region_bundle_id, asset_layer_id),
    UNIQUE (region_bundle_id, layer_name)
);

CREATE TABLE IF NOT EXISTS routing.asset_features (
    region_bundle_id text NOT NULL REFERENCES routing.regions(region_bundle_id) ON DELETE CASCADE,
    asset_id text NOT NULL,
    asset_layer_id text NOT NULL,
    asset_name text NOT NULL,
    asset_kind text NOT NULL,
    geometry_type text NOT NULL,
    priority_class text,
    tags jsonb NOT NULL DEFAULT '{}'::jsonb,
    geom geometry(Geometry, 4326) NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (region_bundle_id, asset_id),
    FOREIGN KEY (region_bundle_id, asset_layer_id)
        REFERENCES routing.asset_layers(region_bundle_id, asset_layer_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS routing_asset_features_geom_gix
    ON routing.asset_features
    USING gist (geom);

CREATE TABLE IF NOT EXISTS routing.overlay_sets (
    region_bundle_id text NOT NULL REFERENCES routing.regions(region_bundle_id) ON DELETE CASCADE,
    overlay_id text NOT NULL,
    overlay_kind text NOT NULL CHECK (
        overlay_kind IN ('closure', 'risk_multiplier', 'temporary_restriction')
    ),
    overlay_name text NOT NULL,
    overlay_fingerprint text NOT NULL CHECK (overlay_fingerprint ~ '^[a-f0-9]{64}$'),
    starts_at_utc timestamptz,
    ends_at_utc timestamptz,
    notes text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at_utc timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (region_bundle_id, overlay_id),
    CHECK (ends_at_utc IS NULL OR starts_at_utc IS NULL OR ends_at_utc >= starts_at_utc)
);

CREATE INDEX IF NOT EXISTS routing_overlay_sets_active_idx
    ON routing.overlay_sets (region_bundle_id, starts_at_utc, ends_at_utc);

CREATE TABLE IF NOT EXISTS routing.overlay_edge_closures (
    region_bundle_id text NOT NULL,
    overlay_id text NOT NULL,
    edge_id text NOT NULL,
    is_closed boolean NOT NULL DEFAULT true,
    reason text,
    PRIMARY KEY (region_bundle_id, overlay_id, edge_id),
    FOREIGN KEY (region_bundle_id, overlay_id)
        REFERENCES routing.overlay_sets(region_bundle_id, overlay_id)
        ON DELETE CASCADE,
    FOREIGN KEY (region_bundle_id, edge_id)
        REFERENCES routing.edges(region_bundle_id, edge_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS routing_overlay_edge_closures_edge_idx
    ON routing.overlay_edge_closures (region_bundle_id, edge_id);

CREATE TABLE IF NOT EXISTS routing.overlay_edge_risk_multipliers (
    region_bundle_id text NOT NULL,
    overlay_id text NOT NULL,
    edge_id text NOT NULL,
    cost_multiplier double precision NOT NULL CHECK (cost_multiplier > 0),
    reason text,
    PRIMARY KEY (region_bundle_id, overlay_id, edge_id),
    FOREIGN KEY (region_bundle_id, overlay_id)
        REFERENCES routing.overlay_sets(region_bundle_id, overlay_id)
        ON DELETE CASCADE,
    FOREIGN KEY (region_bundle_id, edge_id)
        REFERENCES routing.edges(region_bundle_id, edge_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS routing_overlay_edge_risk_edge_idx
    ON routing.overlay_edge_risk_multipliers (region_bundle_id, edge_id);

CREATE TABLE IF NOT EXISTS routing.overlay_edge_temporary_restrictions (
    region_bundle_id text NOT NULL,
    overlay_id text NOT NULL,
    edge_id text NOT NULL,
    speed_cap_kph double precision CHECK (speed_cap_kph IS NULL OR speed_cap_kph > 0),
    delay_seconds double precision NOT NULL DEFAULT 0 CHECK (delay_seconds >= 0),
    cost_multiplier double precision NOT NULL DEFAULT 1 CHECK (cost_multiplier > 0),
    reason text,
    PRIMARY KEY (region_bundle_id, overlay_id, edge_id),
    FOREIGN KEY (region_bundle_id, overlay_id)
        REFERENCES routing.overlay_sets(region_bundle_id, overlay_id)
        ON DELETE CASCADE,
    FOREIGN KEY (region_bundle_id, edge_id)
        REFERENCES routing.edges(region_bundle_id, edge_id)
        ON DELETE CASCADE,
    CHECK (
        speed_cap_kph IS NOT NULL
        OR delay_seconds > 0
        OR cost_multiplier <> 1
    )
);

CREATE INDEX IF NOT EXISTS routing_overlay_edge_restrictions_edge_idx
    ON routing.overlay_edge_temporary_restrictions (region_bundle_id, edge_id);

