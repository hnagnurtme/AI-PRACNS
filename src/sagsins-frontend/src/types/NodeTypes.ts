import type { Geo3D, Orbit, Velocity } from "./ModelTypes";

export interface CreateNodeRequest {
    nodeId: string; 
    nodeType: string;
    isOperational: boolean;
    position: Geo3D;
    orbit?: Orbit | null;      
    velocity?: Velocity | null;
    currentBandwidth: number;
    avgLatencyMs: number;
    packetLossRate: number;
}

export interface UpdateNodeRequest {
    nodeType?: string;
    isOperational?: boolean;
    position?: Geo3D;
    orbit?: Orbit | null;
    velocity?: Velocity | null;
    currentBandwidth?: number;
    avgLatencyMs?: number;
    packetLossRate?: number;
    packetBufferLoad?: number;
    currentThroughput?: number;
    resourceUtilization?: number;
    powerLevel?: number;
}

export interface NodeDTO {
    nodeId: string;
    nodeType: string;
    position: Geo3D;
    orbit?: Orbit | null;
    velocity?: Velocity | null;
    
    isHealthy: boolean; 
    isOperational: boolean;
    
    currentBandwidth: number;
    avgLatencyMs: number;
    packetLossRate: number;
    currentThroughput: number;
    resourceUtilization: number;
    powerLevel: number;
    lastUpdated: number;
}

export interface ApiResponse<T> {
    status: number;
    error: string | null;
    message: string;
    data: T;
}