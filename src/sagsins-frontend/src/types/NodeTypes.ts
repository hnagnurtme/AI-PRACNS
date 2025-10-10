import type { Geo3D, Orbit, Velocity } from "./ModelTypes";

export type NodeType = 'GROUND_STATION' | 'LEO_SATELLITE' | 'MEO_SATELLITE' | 'GEO_SATELLITE';

export type WeatherType = 'CLEAR' | 'LIGHT_RAIN' | 'RAIN' | 'SNOW' | 'STORM' | 'SEVERE_STORM';

export interface CreateNodeRequest {
    nodeId: string; 
    nodeType: NodeType;
    position: Geo3D;
    orbit?: Orbit | null;      
    velocity?: Velocity | null;
    batteryChargePercent: number;
    nodeProcessingDelayMs: number;
    packetLossRate: number;
    resourceUtilization: number;
    packetBufferCapacity: number;
    weather: WeatherType;
    host: string;
    port?: number;
    isOperational: boolean;
}

export interface UpdateNodeRequest {
    nodeType?: NodeType;
    isOperational?: boolean;
    position?: Geo3D;
    orbit?: Orbit | null;
    velocity?: Velocity | null;
    currentBandwidth?: number;
    avgLatencyMs?: number;
    packetLossRate?: number;
    currentThroughput?: number;
    nodeProcessingDelayMs?: number;
    resourceUtilization?: number;
    packetBufferLoad?: number;
    packetBufferCapacity?: number;
    batteryChargePercent?: number;
    powerLevel?: number;
    temperatureCelsius?: number;
    cpuUsagePercent?: number;
    memoryUsagePercent?: number;
    weather?: WeatherType;
    radiationLevel?: number;
    signalToNoiseRatio?: number;
    linkQuality?: number;
    host?: string;
    port?: number;
    uplinkPower?: number;
    downlinkPower?: number;
    linkLatencyMs?: number;
    lastUpdated?: number;
    statusMessage?: string;
    errorCount?: number;
    reliabilityScore?: number;
}

export interface NodeDTO {
    nodeId: string;
    nodeType: NodeType;
    position: Geo3D;
    orbit?: Orbit | null;
    velocity?: Velocity | null;
    
    operational?: boolean; // Make it optional for backward compatibility
    
    // Core fields that might exist
    batteryChargePercent?: number;
    nodeProcessingDelayMs?: number;
    packetLossRate?: number;
    resourceUtilization?: number;
    packetBufferCapacity?: number;
    currentPacketCount?: number;
    weather?: WeatherType;
    host?: string;
    port?: number;
    lastUpdated?: number;
    
    // Legacy fields for backward compatibility
    currentBandwidth?: number;
    avgLatencyMs?: number;
    currentThroughput?: number;
    powerLevel?: number;
    isOperational?: boolean;
}

export interface HealthResponse {
    status: string;
    message: string;
}

export interface DockerResponse {
    nodeId: string;
    pid: string;
    status: string;
}