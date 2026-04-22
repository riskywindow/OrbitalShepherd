import { useEffect, useEffectEvent, useRef } from "react";
import {
  ArcType,
  Cartesian2,
  Cartesian3,
  Color,
  ColorMaterialProperty,
  CustomDataSource,
  CzmlDataSource,
  EllipsoidTerrainProvider,
  HorizontalOrigin,
  JulianDate,
  LabelStyle,
  VerticalOrigin,
  Viewer,
  defined,
} from "cesium";

import type { ReplayTickSnapshot } from "./replay";
import type { CzmlPacket, ScenarioBundle } from "./types";

interface GlobeViewProps {
  bundle: ScenarioBundle | null;
  czml: CzmlPacket[] | null;
  snapshot: ReplayTickSnapshot | null;
}

const TARGET_RING_RADIUS_M = 42_000;

export function GlobeView({ bundle, czml, snapshot }: GlobeViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const baseSourceRef = useRef<CustomDataSource | null>(null);
  const overlaySourceRef = useRef<CustomDataSource | null>(null);
  const satelliteSourceRef = useRef<CzmlDataSource | null>(null);

  const syncOverlayState = useEffectEvent(() => {
    const viewer = viewerRef.current;
    const overlaySource = overlaySourceRef.current;
    const satelliteSource = satelliteSourceRef.current;
    if (!viewer || !overlaySource || !bundle || !snapshot) {
      return;
    }

    const currentTime = JulianDate.fromIso8601(snapshot.simTimeUtc);
    viewer.clock.currentTime = currentTime;
    overlaySource.entities.removeAll();

    const opportunitiesById = Object.fromEntries(
      bundle.observation_opportunities.map((opportunity) => [
        opportunity.opportunity_id,
        opportunity,
      ]),
    );
    const downlinksById = Object.fromEntries(
      bundle.downlink_windows.map((window) => [window.window_id, window]),
    );
    const stationsById = Object.fromEntries(
      bundle.ground_stations.map((station) => [station.station_id, station]),
    );

    for (const targetCell of bundle.target_cells) {
      const state = snapshot.targetStates[targetCell.target_cell_id] ?? "idle";
      const style = targetStyle(state);
      overlaySource.entities.add({
        id: `target:${targetCell.target_cell_id}`,
        position: Cartesian3.fromDegrees(
          targetCell.centroid.lon,
          targetCell.centroid.lat,
          0,
        ),
        point: {
          pixelSize: style.pixelSize,
          color: style.color,
          outlineColor: style.outlineColor,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        ellipse: {
          semiMajorAxis: TARGET_RING_RADIUS_M,
          semiMinorAxis: TARGET_RING_RADIUS_M,
          material: new ColorMaterialProperty(style.color.withAlpha(0.12)),
          outline: true,
          outlineColor: style.color.withAlpha(0.8),
          height: 0,
        },
        label: {
          text: labelForTarget(targetCell.region_name ?? targetCell.target_cell_id, state),
          font: "11pt monospace",
          fillColor: style.labelColor,
          outlineColor: Color.fromCssColorString("#03070d"),
          outlineWidth: 2,
          showBackground: true,
          backgroundColor: Color.fromCssColorString("#03070d").withAlpha(0.78),
          style: LabelStyle.FILL_AND_OUTLINE,
          horizontalOrigin: HorizontalOrigin.LEFT,
          verticalOrigin: VerticalOrigin.BOTTOM,
          pixelOffset: new Cartesian2(16, -12),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    }

    for (const event of snapshot.currentObservationEvents) {
      const opportunity = opportunitiesById[String(event.payload.opportunity_id)];
      if (!opportunity || !satelliteSource) {
        continue;
      }
      const satellitePosition = resolveSatellitePosition(
        satelliteSource,
        opportunity.satellite_id,
        currentTime,
      );
      if (!satellitePosition) {
        continue;
      }
      const targetPosition = Cartesian3.fromDegrees(
        targetLongitude(bundle, opportunity.target_cell_id),
        targetLatitude(bundle, opportunity.target_cell_id),
        0,
      );
      const usable = event.payload.usable !== false;
      overlaySource.entities.add({
        id: `obs-link:${event.event_id}`,
        polyline: {
          positions: [satellitePosition, targetPosition],
          width: usable ? 3 : 2,
          material: usable
            ? Color.fromCssColorString("#ffd166")
            : Color.fromCssColorString("#ff7b72"),
          arcType: ArcType.NONE,
        },
      });
    }

    for (const event of snapshot.currentDownlinkEvents) {
      const window = downlinksById[String(event.payload.window_id)];
      if (!window || !satelliteSource) {
        continue;
      }
      const station = stationsById[window.station_id];
      if (!station) {
        continue;
      }
      const satellitePosition = resolveSatellitePosition(
        satelliteSource,
        window.satellite_id,
        currentTime,
      );
      if (!satellitePosition) {
        continue;
      }
      const stationPosition = Cartesian3.fromDegrees(
        station.location.lon,
        station.location.lat,
        station.location.alt_m,
      );
      overlaySource.entities.add({
        id: `downlink:${event.event_id}`,
        polyline: {
          positions: [satellitePosition, stationPosition],
          width: 3,
          material: Color.fromCssColorString("#52d6ff"),
          arcType: ArcType.NONE,
        },
      });
    }

    viewer.scene.requestRender();
  });

  useEffect(() => {
    if (!containerRef.current || viewerRef.current) {
      return;
    }
    const viewer = new Viewer(containerRef.current, {
      animation: false,
      baseLayerPicker: false,
      fullscreenButton: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
      navigationHelpButton: false,
      sceneModePicker: false,
      selectionIndicator: false,
      timeline: false,
      terrainProvider: new EllipsoidTerrainProvider(),
    });
    viewerRef.current = viewer;
    viewer.imageryLayers.removeAll();
    viewer.scene.backgroundColor = Color.fromCssColorString("#02050a");
    if (viewer.scene.skyAtmosphere) {
      viewer.scene.skyAtmosphere.show = false;
    }
    if (viewer.scene.sun) {
      viewer.scene.sun.show = false;
    }
    if (viewer.scene.moon) {
      viewer.scene.moon.show = false;
    }
    viewer.scene.globe.baseColor = Color.fromCssColorString("#07111e");
    viewer.scene.globe.enableLighting = true;
    viewer.scene.globe.showGroundAtmosphere = true;
    viewer.scene.globe.depthTestAgainstTerrain = false;
    viewer.scene.requestRenderMode = true;
    viewer.camera.setView({
      destination: Cartesian3.fromDegrees(-110, 28, 24_000_000),
    });

    const baseSource = new CustomDataSource("orbital-base");
    const overlaySource = new CustomDataSource("orbital-overlay");
    baseSourceRef.current = baseSource;
    overlaySourceRef.current = overlaySource;
    void viewer.dataSources.add(baseSource);
    void viewer.dataSources.add(overlaySource);

    return () => {
      satelliteSourceRef.current = null;
      baseSourceRef.current = null;
      overlaySourceRef.current = null;
      viewer.destroy();
      viewerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const viewer = viewerRef.current;
    const baseSource = baseSourceRef.current;
    if (!viewer || !baseSource) {
      return;
    }
    baseSource.entities.removeAll();
    if (!bundle) {
      viewer.scene.requestRender();
      return;
    }

    for (const station of bundle.ground_stations) {
      baseSource.entities.add({
        id: `station:${station.station_id}`,
        position: Cartesian3.fromDegrees(
          station.location.lon,
          station.location.lat,
          station.location.alt_m,
        ),
        point: {
          pixelSize: 9,
          color: Color.fromCssColorString("#8ef0ff"),
          outlineColor: Color.fromCssColorString("#031019"),
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: station.name,
          font: "11pt monospace",
          fillColor: Color.fromCssColorString("#d5fbff"),
          outlineColor: Color.fromCssColorString("#031019"),
          outlineWidth: 2,
          showBackground: true,
          backgroundColor: Color.fromCssColorString("#031019").withAlpha(0.78),
          style: LabelStyle.FILL_AND_OUTLINE,
          horizontalOrigin: HorizontalOrigin.LEFT,
          verticalOrigin: VerticalOrigin.CENTER,
          pixelOffset: new Cartesian2(14, 0),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    }
    viewer.scene.requestRender();
  }, [bundle]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) {
      return;
    }
    let cancelled = false;

    async function loadSatelliteSource(): Promise<void> {
      const activeViewer = viewerRef.current;
      if (!activeViewer) {
        return;
      }
      if (satelliteSourceRef.current) {
        await activeViewer.dataSources.remove(satelliteSourceRef.current, true);
        satelliteSourceRef.current = null;
      }
      if (!czml || czml.length === 0) {
        activeViewer.scene.requestRender();
        return;
      }
      const dataSource = new CzmlDataSource("orbital-satellites");
      await dataSource.load(czml);
      if (cancelled) {
        return;
      }
      satelliteSourceRef.current = dataSource;
      await activeViewer.dataSources.add(dataSource);
      activeViewer.clock.shouldAnimate = false;
      activeViewer.scene.requestRender();
      syncOverlayState();
    }

    void loadSatelliteSource();

    return () => {
      cancelled = true;
    };
  }, [czml, syncOverlayState]);

  useEffect(() => {
    syncOverlayState();
  }, [snapshot, bundle, syncOverlayState]);

  return <div className="globe-canvas" ref={containerRef} />;
}

function resolveSatellitePosition(
  dataSource: CzmlDataSource,
  satelliteId: string,
  currentTime: JulianDate,
): Cartesian3 | null {
  const entity = dataSource.entities.getById(satelliteId);
  const position = entity?.position?.getValue(currentTime);
  return defined(position) ? position : null;
}

function targetLongitude(bundle: ScenarioBundle, targetCellId: string): number {
  return (
    bundle.target_cells.find((targetCell) => targetCell.target_cell_id === targetCellId)
      ?.centroid.lon ?? 0
  );
}

function targetLatitude(bundle: ScenarioBundle, targetCellId: string): number {
  return (
    bundle.target_cells.find((targetCell) => targetCell.target_cell_id === targetCellId)
      ?.centroid.lat ?? 0
  );
}

function labelForTarget(name: string, state: string): string {
  if (state === "idle") {
    return name;
  }
  return `${name}\n${state.toUpperCase()}`;
}

function targetStyle(state: string): {
  color: Color;
  outlineColor: Color;
  labelColor: Color;
  pixelSize: number;
} {
  if (state === "available") {
    return {
      color: Color.fromCssColorString("#ffb347"),
      outlineColor: Color.fromCssColorString("#ffda8a"),
      labelColor: Color.fromCssColorString("#ffda8a"),
      pixelSize: 14,
    };
  }
  if (state === "selected") {
    return {
      color: Color.fromCssColorString("#ffd166"),
      outlineColor: Color.fromCssColorString("#fff3bf"),
      labelColor: Color.fromCssColorString("#fff3bf"),
      pixelSize: 15,
    };
  }
  if (state === "downlinked") {
    return {
      color: Color.fromCssColorString("#4ed9ff"),
      outlineColor: Color.fromCssColorString("#d8fbff"),
      labelColor: Color.fromCssColorString("#d8fbff"),
      pixelSize: 15,
    };
  }
  if (state === "degraded") {
    return {
      color: Color.fromCssColorString("#ff7b72"),
      outlineColor: Color.fromCssColorString("#ffd7d4"),
      labelColor: Color.fromCssColorString("#ffd7d4"),
      pixelSize: 14,
    };
  }
  if (state === "missed") {
    return {
      color: Color.fromCssColorString("#ff4d6d"),
      outlineColor: Color.fromCssColorString("#ffc0ca"),
      labelColor: Color.fromCssColorString("#ffc0ca"),
      pixelSize: 16,
    };
  }
  return {
    color: Color.fromCssColorString("#3b4f63"),
    outlineColor: Color.fromCssColorString("#7f97aa"),
    labelColor: Color.fromCssColorString("#b8cad8"),
    pixelSize: 10,
  };
}
