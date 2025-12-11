import type { NodeDTO } from './NodeTypes';
import type { UserTerminal } from './UserTerminalTypes';

export interface NetworkConnection {
    fromNodeId: string;
    toNodeId: string;
    latency: number; // ms
    bandwidth: number; // Mbps
    status: 'active' | 'inactive' | 'degraded';
    distance?: number; // km
    signalStrength?: number; // dBm
    packetLossRate?: number;
    lastUpdated?: string;
}

export interface NetworkStatistics {
    totalNodes: number;
    activeNodes: number;
    totalTerminals: number;
    connectedTerminals: number;
    activeConnections: number;
    totalConnections: number;
    totalBandwidth: number; // Mbps
    averageLatency: number; // ms
    averagePacketLoss: number;
    networkHealth: 'healthy' | 'degraded' | 'critical';
    utilizationRate: number; // percentage
}

export interface NetworkTopology {
    networkId: string;
    networkName: string;
    description?: string;
    nodes: NodeDTO[];
    terminals: UserTerminal[];
    connections: NetworkConnection[];
    statistics: NetworkStatistics;
    createdAt?: string;
    updatedAt?: string;
}

export interface TopologyUpdate {
    networkId: string;
    type: 'node' | 'terminal' | 'connection' | 'statistics';
    data: NodeDTO | UserTerminal | NetworkConnection | NetworkStatistics;
    timestamp: string;
}

export interface TopologyStatsRequest {
    networkId?: string; // If not provided, returns global stats
}

