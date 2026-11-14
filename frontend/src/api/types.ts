/**
 * Friendly names for the generated API types (ADR 0017).
 *
 * Everything here is an alias of `schema.gen.ts`, which openapi-typescript generates from
 * `openapi.json` (`uv run poe api-docs`). Never hand-write a backend type: add it to the
 * backend models and regenerate.
 */
import type { components, paths } from "./schema.gen.ts";

type Schemas = components["schemas"];

export type ApiPaths = paths;

export type Reading = Schemas["Reading"];
export type StateChange = Schemas["StateChange"];
export type Alert = Schemas["Alert"];
export type AlertSeverity = Schemas["AlertSeverity"];
export type Prediction = Schemas["Prediction"];
export type Attribution = Schemas["Attribution"];
export type MachineState = Schemas["MachineState"];
export type MachineKind = Schemas["MachineKind"];
export type MachineView = Schemas["MachineView"];
export type DataQualityReport = Schemas["DataQualityReport"];
export type LineViewModel = Schemas["LineViewModel"];
export type RootCauseView = Schemas["RootCauseView"];
export type ReplayResponse = Schemas["ReplayResponse"];
export type Command = Schemas["Command"];
export type CommandAck = Schemas["CommandAck"];
export type CommandRequest = Schemas["CommandRequest"];

export type LineResponse = Schemas["LineResponse"];
export type LineMachine = Schemas["LineMachineModel"];
export type LineBuffer = Schemas["LineBufferModel"];
export type SensorSpec = Schemas["SensorSpecModel"];
export type Topology = Schemas["TopologyModel"];

export type MachinesResponse = Schemas["MachinesResponse"];
export type MachineStateResponse = Schemas["MachineStateResponse"];
export type HistoryResponse = Schemas["HistoryResponse"];
export type AlertsResponse = Schemas["AlertsResponse"];
export type HealthResponse = Schemas["HealthResponse"];
export type SourcesResponse = Schemas["SourcesResponse"];
export type IngestStats = Schemas["IngestStatsModel"];
export type AdapterKind = Schemas["AdapterKind"];
export type Capability = Schemas["Capability"];
export type AuthMeResponse = Schemas["AuthMeResponse"];
export type Role = Schemas["Role"];
export type CommandLimitsResponse = Schemas["CommandLimitsResponse"];
export type CommandType = Schemas["CommandType"];
export type AckStatus = Schemas["AckStatus"];
export type CommandRecord = Schemas["CommandRecord"];
export type CommandsResponse = Schemas["CommandsResponse"];
export type ReplayRangeResponse = Schemas["ReplayRangeResponse"];
export type HistoryBucket = Schemas["HistoryBucketModel"];
export type ScenarioOptionsResponse = Schemas["ScenarioOptionsResponse"];
export type ScenarioRecord = Schemas["ScenarioRecord"];
export type ScenarioPayload = Schemas["ScenarioPayload"];
export type ScenariosResponse = Schemas["ScenariosResponse"];

export type WsClientMessage = Schemas["WsClientMessage"];
export type WsServerMessage = Schemas["WsServerMessage"];
export type WsSnapshotData = Schemas["WsSnapshotData"];
export type WsSubscribeRequest = Schemas["WsSubscribeRequest"];
export type WsAuthRequest = Schemas["WsAuthRequest"];
export type Channel = NonNullable<WsSubscribeRequest["channels"]>[number];

/** Sensor fields of a Reading (the line config's sensor catalog keys). */
export const SENSOR_FIELDS = ["speed", "temp", "vib", "load", "current"] as const;
export type SensorField = (typeof SENSOR_FIELDS)[number];
