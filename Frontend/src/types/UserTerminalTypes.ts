import type { Position } from './ModelTypes';

export type TerminalStatus = 'idle' | 'connected' | 'transmitting' | 'disconnected';

export type TerminalType = 'MOBILE' | 'FIXED' | 'VEHICLE' | 'AIRCRAFT';

export interface QoSRequirements {
    maxLatencyMs: number;
    minBandwidthMbps: number;
    maxLossRate: number;
    priority: number;
    serviceType?: string;
}

export interface UserTerminal {
    id: string;
    terminalId: string;
    terminalName: string;
    terminalType: TerminalType;
    position: Position;
    status: TerminalStatus;
    connectedNodeId?: string | null;
    qosRequirements: QoSRequirements;
    metadata?: {
        description?: string;
        region?: string;
        [key: string]: unknown;
    };
    lastUpdated: string;
    connectionMetrics?: {
        latencyMs?: number;
        bandwidthMbps?: number;
        packetLossRate?: number;
        signalStrength?: number;
    };
}

export interface GenerateTerminalsRequest {
    count: number;
    bounds?: {
        minLat: number;
        maxLat: number;
        minLon: number;
        maxLon: number;
    };
    region?: string;
    density?: number;
    terminalType?: TerminalType;
    qosRequirements?: Partial<QoSRequirements>;
}

export interface ConnectTerminalRequest {
    terminalId: string;
    nodeId: string;
}

export interface TerminalConnectionResult {
    terminalId: string;
    nodeId: string;
    success: boolean;
    latencyMs?: number;
    bandwidthMbps?: number;
    packetLossRate?: number;
    message?: string;
    timestamp: string;
}

export interface TerminalUpdate {
    terminalId: string;
    status?: TerminalStatus;
    connectedNodeId?: string | null;
    connectionMetrics?: UserTerminal['connectionMetrics'];
    lastUpdated: string;
}

