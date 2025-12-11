import type { Position as Position, Orbit, Velocity, Communication } from "./ModelTypes";

export type NodeType = 'GROUND_STATION' | 'LEO_SATELLITE' | 'MEO_SATELLITE' | 'GEO_SATELLITE';

export type WeatherType = 'CLEAR' | 'LIGHT_RAIN' | 'RAIN' | 'SNOW' | 'STORM' | 'SEVERE_STORM';

export interface NodeDTO {
  id: string;
  nodeId: string;
  nodeName: string;
  nodeType: string;
  position: Position;
  velocity: Velocity;
  orbit?: Orbit | null;
  communication: Communication;
  isOperational: boolean;
  batteryChargePercent: number;
  nodeProcessingDelayMs: number;
  packetLossRate: number;
  resourceUtilization: number;
  packetBufferCapacity: number;
  currentPacketCount: number;
  weather: string;
  lastUpdated: string;
  healthy: boolean;
  host?: string;
  port?: number;
}

export interface UpdateStatusRequest {
  isOperational?: boolean;
  batteryChargePercent?: number;
  nodeProcessingDelayMs?: number;
  packetLossRate?: number;
  resourceUtilization?: number;
  currentPacketCount?: number;
  weather?: WeatherType;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  message?: string;
}
