import type { Position } from './ModelTypes';
import type { QoSRequirements } from './UserTerminalTypes';

export interface PathSegment {
    type: 'terminal' | 'node';
    id: string;
    name: string;
    position: Position;
}

export interface RoutingPath {
    source: {
        terminalId: string;
        position: Position;
    };
    destination: {
        terminalId: string;
        position: Position;
    };
    path: PathSegment[];
    totalDistance: number; // km
    estimatedLatency: number; // ms
    hops: number;
    algorithm?: 'simple' | 'dijkstra' | 'rl'; // Routing algorithm used
}

export interface SendPacketRequest {
    sourceTerminalId: string;
    destinationTerminalId: string;
    packetSize?: number; // bytes
    priority?: number; // 1-10
    serviceQos?: QoSRequirements; // QoS requirements for the packet
}

export interface Packet {
    packetId: string;
    sourceTerminalId: string;
    destinationTerminalId: string;
    packetSize: number;
    priority: number;
    serviceQos?: QoSRequirements;
    path: RoutingPath;
    status: 'sent' | 'in_transit' | 'delivered' | 'failed';
    sentAt: string;
    estimatedArrival?: string;
    actualArrival?: string;
}

export interface NodeResourceInfo {
    nodeId: string;
    nodeName: string;
    nodeType: string;
    isOperational: boolean;
    resourceUtilization: number;
    currentPacketCount: number;
    packetBufferCapacity: number;
    nodeProcessingDelayMs: number;
    packetLossRate: number;
    batteryChargePercent: number;
}

export interface AlgorithmComparison {
    sourceTerminalId: string;
    destinationTerminalId: string;
    serviceQos?: QoSRequirements;
    scenario?: string;
    algorithm1: {
        name: string;
        path: RoutingPath;
        qosMet: boolean;
    };
    algorithm2: {
        name: string;
        path: RoutingPath;
        qosMet: boolean;
    };
    comparison: {
        distanceDifference: number;
        latencyDifference: number;
        hopsDifference: number;
        bestDistance: string;
        bestLatency: string;
        bestHops: string;
    };
    nodeResources?: Record<string, NodeResourceInfo>;
    qosWarnings: string[];
    timestamp: string;
}

export interface CompareAlgorithmsRequest {
    sourceTerminalId: string;
    destinationTerminalId: string;
    serviceQos?: QoSRequirements;
    scenario?: string;
    algorithm1: 'simple' | 'dijkstra' | 'rl';
    algorithm2: 'simple' | 'dijkstra' | 'rl';
}

