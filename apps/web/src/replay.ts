import type {
  DownlinkWindow,
  ObservationOpportunity,
  ReplayEvent,
  ScenarioBundle,
  TargetCell,
} from "./types";

export type TargetOverlayState =
  | "idle"
  | "available"
  | "selected"
  | "downlinked"
  | "degraded"
  | "missed";

type IncidentReplayStatus = "unseen" | "observed" | "downlinked" | "degraded" | "missed";

interface IncidentProgress {
  incidentId: string;
  targetCellId: string;
  status: IncidentReplayStatus;
  urgencyScore: number;
  hadUnusableObservation: boolean;
  lastOpportunityId?: string;
}

export interface ReplayTickSnapshot {
  simTick: number;
  simTimeUtc: string;
  availableObservationIds: string[];
  availableDownlinkIds: string[];
  selectedObservationIds: string[];
  selectedDownlinkIds: string[];
  executedObservationIds: string[];
  executedDownlinkIds: string[];
  currentEvents: ReplayEvent[];
  currentObservationEvents: ReplayEvent[];
  currentDownlinkEvents: ReplayEvent[];
  currentPacketEvents: ReplayEvent[];
  cumulativeReward: number;
  missionUtility: number | null;
  incidentCounts: Record<IncidentReplayStatus, number>;
  targetStates: Record<string, TargetOverlayState>;
}

export interface ReplayModel {
  ticks: ReplayTickSnapshot[];
  importantEvents: ReplayEvent[];
}

const IMPORTANT_EVENT_TYPES = new Set([
  "episode_started",
  "action_selected",
  "observation_executed",
  "downlink_executed",
  "incident_packet_emitted",
  "episode_ended",
]);

export function buildReplayModel(
  bundle: ScenarioBundle,
  events: ReplayEvent[],
): ReplayModel {
  const sortedEvents = [...events].sort((left, right) => left.event_index - right.event_index);
  const observationById = indexById(
    bundle.observation_opportunities,
    (opportunity) => opportunity.opportunity_id,
  );
  const downlinkById = indexById(bundle.downlink_windows, (window) => window.window_id);
  const incidentProgressById = new Map<string, IncidentProgress>(
    bundle.incidents.map((incident) => [
      incident.incident_id,
      {
        incidentId: incident.incident_id,
        targetCellId: incident.target_cell_id,
        status: "unseen",
        urgencyScore: incident.urgency_score,
        hadUnusableObservation: false,
      },
    ]),
  );

  const availableObservationIds = new Set<string>();
  const availableDownlinkIds = new Set<string>();
  const selectedObservationIds = new Set<string>();
  const selectedDownlinkIds = new Set<string>();
  const executedObservationIds = new Set<string>();
  const executedDownlinkIds = new Set<string>();
  let cumulativeReward = 0;
  let missionUtility: number | null = null;

  if (sortedEvents.length === 0) {
    return {
      ticks: [
        {
          simTick: 0,
          simTimeUtc: bundle.time_window.start_time_utc,
          availableObservationIds: [],
          availableDownlinkIds: [],
          selectedObservationIds: [],
          selectedDownlinkIds: [],
          executedObservationIds: [],
          executedDownlinkIds: [],
          currentEvents: [],
          currentObservationEvents: [],
          currentDownlinkEvents: [],
          currentPacketEvents: [],
          cumulativeReward,
          missionUtility,
          incidentCounts: countIncidentStatuses(incidentProgressById),
          targetStates: computeTargetStates(
            bundle.target_cells,
            observationById,
            availableObservationIds,
            selectedObservationIds,
            incidentProgressById,
            [],
          ),
        },
      ],
      importantEvents: [],
    };
  }

  const ticks: ReplayTickSnapshot[] = [];
  let tickEvents: ReplayEvent[] = [];
  let tickObservationEvents: ReplayEvent[] = [];
  let tickDownlinkEvents: ReplayEvent[] = [];
  let tickPacketEvents: ReplayEvent[] = [];
  let currentTick = sortedEvents[0].sim_tick;
  let currentTime = sortedEvents[0].sim_time_utc;

  function pushSnapshot(): void {
    ticks.push({
      simTick: currentTick,
      simTimeUtc: currentTime,
      availableObservationIds: sortedValues(availableObservationIds),
      availableDownlinkIds: sortedValues(availableDownlinkIds),
      selectedObservationIds: sortedValues(selectedObservationIds),
      selectedDownlinkIds: sortedValues(selectedDownlinkIds),
      executedObservationIds: sortedValues(executedObservationIds),
      executedDownlinkIds: sortedValues(executedDownlinkIds),
      currentEvents: [...tickEvents],
      currentObservationEvents: [...tickObservationEvents],
      currentDownlinkEvents: [...tickDownlinkEvents],
      currentPacketEvents: [...tickPacketEvents],
      cumulativeReward,
      missionUtility,
      incidentCounts: countIncidentStatuses(incidentProgressById),
      targetStates: computeTargetStates(
        bundle.target_cells,
        observationById,
        availableObservationIds,
        selectedObservationIds,
        incidentProgressById,
        tickEvents,
      ),
    });
  }

  for (const event of sortedEvents) {
    if (event.sim_tick !== currentTick) {
      pushSnapshot();
      tickEvents = [];
      tickObservationEvents = [];
      tickDownlinkEvents = [];
      tickPacketEvents = [];
      currentTick = event.sim_tick;
      currentTime = event.sim_time_utc;
    }
    currentTime = event.sim_time_utc;
    tickEvents.push(event);

    if (event.event_type === "observation_executed") {
      tickObservationEvents.push(event);
    }
    if (event.event_type === "downlink_executed") {
      tickDownlinkEvents.push(event);
    }
    if (event.event_type === "incident_packet_emitted") {
      tickPacketEvents.push(event);
    }

    applyEvent(
      event,
      incidentProgressById,
      observationById,
      downlinkById,
      availableObservationIds,
      availableDownlinkIds,
      selectedObservationIds,
      selectedDownlinkIds,
      executedObservationIds,
      executedDownlinkIds,
      (rewardDelta) => {
        cumulativeReward += rewardDelta;
      },
      (nextMissionUtility) => {
        missionUtility = nextMissionUtility;
      },
    );
  }

  pushSnapshot();

  return {
    ticks,
    importantEvents: sortedEvents.filter(isImportantEvent),
  };
}

function applyEvent(
  event: ReplayEvent,
  incidentProgressById: Map<string, IncidentProgress>,
  observationById: Record<string, ObservationOpportunity>,
  downlinkById: Record<string, DownlinkWindow>,
  availableObservationIds: Set<string>,
  availableDownlinkIds: Set<string>,
  selectedObservationIds: Set<string>,
  selectedDownlinkIds: Set<string>,
  executedObservationIds: Set<string>,
  executedDownlinkIds: Set<string>,
  onReward: (rewardDelta: number) => void,
  onMissionUtility: (missionUtility: number | null) => void,
): void {
  const payload = event.payload;

  if (event.event_type === "candidate_set_materialized") {
    syncSet(availableObservationIds, stringArray(payload.materialized_observation_ids));
    syncSet(availableDownlinkIds, stringArray(payload.materialized_downlink_ids));
    return;
  }

  if (event.event_type === "action_selected") {
    const actionType = stringValue(payload.action_type);
    const actionRef = stringValue(payload.action_ref);
    if (actionType === "schedule_observation" && actionRef) {
      selectedObservationIds.add(actionRef);
    }
    if (actionType === "schedule_downlink" && actionRef) {
      selectedDownlinkIds.add(actionRef);
    }
    return;
  }

  if (event.event_type === "observation_executed") {
    const opportunityId = stringValue(payload.opportunity_id);
    const usable = Boolean(payload.usable);
    if (opportunityId) {
      executedObservationIds.add(opportunityId);
      selectedObservationIds.delete(opportunityId);
      availableObservationIds.delete(opportunityId);
    }
    for (const incidentId of stringArray(payload.incident_ids)) {
      const progress = incidentProgressById.get(incidentId);
      if (!progress) {
        continue;
      }
      progress.lastOpportunityId = opportunityId ?? progress.lastOpportunityId;
      if (usable && progress.status !== "downlinked") {
        progress.status = "observed";
      }
      if (!usable) {
        progress.hadUnusableObservation = true;
      }
    }
    return;
  }

  if (event.event_type === "downlink_executed") {
    const windowId = stringValue(payload.window_id);
    if (windowId) {
      executedDownlinkIds.add(windowId);
      selectedDownlinkIds.delete(windowId);
      availableDownlinkIds.delete(windowId);
      const window = downlinkById[windowId];
      if (window) {
        for (const selectedId of [...selectedObservationIds]) {
          const opportunity = observationById[selectedId];
          if (opportunity?.satellite_id === window.satellite_id) {
            selectedObservationIds.delete(selectedId);
          }
        }
      }
    }
    return;
  }

  if (event.event_type === "incident_packet_emitted") {
    const incidentId = stringValue(payload.incident_id);
    const opportunityId = stringValue(payload.observation_opportunity_id);
    const progress = incidentId ? incidentProgressById.get(incidentId) : undefined;
    if (progress) {
      progress.status = "downlinked";
      progress.lastOpportunityId = opportunityId ?? progress.lastOpportunityId;
    }
    return;
  }

  if (event.event_type === "reward_assessed") {
    onReward(numberValue(payload.total_reward) ?? 0);
    return;
  }

  if (event.event_type === "episode_ended") {
    const nextMissionUtility = numberValue(payload.mission_utility);
    onMissionUtility(nextMissionUtility ?? null);
    for (const progress of incidentProgressById.values()) {
      if (progress.status === "downlinked") {
        continue;
      }
      if (progress.status === "observed" || progress.hadUnusableObservation) {
        progress.status = "degraded";
        continue;
      }
      progress.status = "missed";
    }
  }
}

function computeTargetStates(
  targetCells: TargetCell[],
  observationById: Record<string, ObservationOpportunity>,
  availableObservationIds: Set<string>,
  selectedObservationIds: Set<string>,
  incidentProgressById: Map<string, IncidentProgress>,
  currentEvents: ReplayEvent[],
): Record<string, TargetOverlayState> {
  const targetStates: Record<string, TargetOverlayState> = {};
  for (const targetCell of targetCells) {
    targetStates[targetCell.target_cell_id] = "idle";
  }

  for (const opportunityId of availableObservationIds) {
    const opportunity = observationById[opportunityId];
    if (!opportunity) {
      continue;
    }
    targetStates[opportunity.target_cell_id] = "available";
  }

  for (const opportunityId of selectedObservationIds) {
    const opportunity = observationById[opportunityId];
    if (!opportunity) {
      continue;
    }
    targetStates[opportunity.target_cell_id] = "selected";
  }

  for (const progress of incidentProgressById.values()) {
    const current = targetStates[progress.targetCellId] ?? "idle";
    if (progress.status === "downlinked") {
      targetStates[progress.targetCellId] = current === "missed" ? current : "downlinked";
    }
    if (progress.status === "degraded") {
      targetStates[progress.targetCellId] = "degraded";
    }
    if (progress.status === "missed") {
      targetStates[progress.targetCellId] = "missed";
    }
  }

  for (const event of currentEvents) {
    if (event.event_type !== "observation_executed") {
      continue;
    }
    const targetCellId = stringValue(event.payload.target_cell_id);
    if (!targetCellId) {
      continue;
    }
    if (event.payload.usable === false && targetStates[targetCellId] !== "missed") {
      targetStates[targetCellId] = "degraded";
    }
  }

  return targetStates;
}

function countIncidentStatuses(
  incidentProgressById: Map<string, IncidentProgress>,
): Record<IncidentReplayStatus, number> {
  const counts: Record<IncidentReplayStatus, number> = {
    unseen: 0,
    observed: 0,
    downlinked: 0,
    degraded: 0,
    missed: 0,
  };
  for (const progress of incidentProgressById.values()) {
    counts[progress.status] += 1;
  }
  return counts;
}

function isImportantEvent(event: ReplayEvent): boolean {
  if (!IMPORTANT_EVENT_TYPES.has(event.event_type)) {
    return false;
  }
  if (event.event_type === "action_selected") {
    return event.payload.action_type !== "noop";
  }
  return true;
}

function indexById<T>(values: T[], keyFor: (value: T) => string): Record<string, T> {
  const entries: Record<string, T> = {};
  for (const value of values) {
    entries[keyFor(value)] = value;
  }
  return entries;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item));
}

function sortedValues(values: Set<string>): string[] {
  return [...values].sort((left, right) => left.localeCompare(right));
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function syncSet(target: Set<string>, values: string[]): void {
  target.clear();
  for (const value of values) {
    target.add(value);
  }
}
