import type { Geo3D as Position, Orbit, Velocity } from "./ModelTypes";

export type NodeType = 'GROUND_STATION' | 'LEO_SATELLITE' | 'MEO_SATELLITE' | 'GEO_SATELLITE';

export type WeatherType = 'CLEAR' | 'LIGHT_RAIN' | 'RAIN' | 'SNOW' | 'STORM' | 'SEVERE_STORM';

export interface Communication {
    frequencyGHz?: number;
    bandwidthMHz?: number;
    transmitPowerDbW?: number;
    antennaGainDb?: number;
    beamWidthDeg?: number;
    maxRangeKm?: number;
    minElevationDeg?: number;
    ipAddress?: string;
    port?: number;
    protocol?: string;
}

export interface NodeDTO {
    id?: string; // backend id
    nodeId: string;
    nodeName: string;
    nodeType: NodeType;
    position: Position;
    orbit?: Orbit | null;
    velocity?: Velocity | null;
    communication?: Communication | null;
    isOperational: boolean;
    batteryChargePercent?: number; // 0..100
    nodeProcessingDelayMs?: number; // >= 0
    packetLossRate?: number; // 0..1
    resourceUtilization?: number; // 0..1
    packetBufferCapacity?: number; // >= 0
    currentPacketCount?: number; // >= 0
    weather?: WeatherType;
    lastUpdated?: string | number; // ISO string (spec) or epoch ms (tolerate)
    host: string;
    port?: number; // 1..65535
    healthy?: boolean; // readOnly
}

export interface UpdateStatusRequest {
    nodeName?: string;
    orbit?: Orbit | null;
    velocity?: Velocity | null;
    communication?: Communication | null;
    isOperational?: boolean;
    batteryChargePercent?: number;
    nodeProcessingDelayMs?: number;
    packetLossRate?: number;
    resourceUtilization?: number;
    packetBufferCapacity?: number;
    currentPacketCount?: number;
    weather?: WeatherType;
    host?: string;
    port?: number;
}

export interface HealthResponse {
    status: string;
    message: string;
}