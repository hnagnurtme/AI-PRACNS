// import type { NodeDTO } from './NodeTypes';
// import type { NetworkConnection } from './NetworkTopologyTypes';

export interface NodeLinkMetric {
    nodeId: string;
    nodeName: string;
    latency: number; // ms
    bandwidth: number; // Mbps
    packetLoss: number;
    signalStrength: number; // dBm
    distance: number; // km
    quality: 'excellent' | 'good' | 'fair' | 'poor';
    score: number; // 0-100
}

export interface UpcomingSatellite {
    nodeId: string;
    nodeName: string;
    nodeType: string;
    currentPosition: {
        latitude: number;
        longitude: number;
        altitude: number;
    };
    currentDistance?: number; // km - current distance from target
    estimatedArrivalTime: string; // ISO timestamp
    estimatedArrivalIn: number; // seconds
    willBeInRange: boolean;
    estimatedLatency: number; // ms when in range
    estimatedBandwidth: number; // Mbps when in range
}

export interface DegradingNode {
    nodeId: string;
    nodeName: string;
    nodeType: string;
    currentMetrics: {
        latency: number;
        packetLoss: number;
        utilization: number;
        queueRatio: number;
        battery: number;
    };
    predictedDegradationTime: string; // ISO timestamp
    predictedDegradationIn: number; // seconds
    degradationReason: string[];
    severity: 'critical' | 'warning' | 'minor';
}

export interface NodeAnalysis {
    nodeId: string;
    nodeName: string;
    bestLinks: NodeLinkMetric[];
    upcomingSatellites: UpcomingSatellite[];
    degradingNodes: DegradingNode[];
    timestamp: string;
}

