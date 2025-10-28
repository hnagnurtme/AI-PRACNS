// src/types.ts
// This file contains all shared type definitions for the dashboard.

// ----------------------------------------------
// 1. DATA STRUCTURE INTERFACES
// ----------------------------------------------
export type ServiceType = "VIDEO_STREAM" | "AUDIO_CALL" | "IMAGE_TRANSFER" | "TEXT_MESSAGE" | "FILE_TRANSFER";

export interface ServiceQoS {
    serviceType: ServiceType;
    defaultPriority?: number;
    maxLatencyMs: number;
    maxJitterMs?: number;
    minBandwidthMbps?: number;
    maxLossRate?: number;
}

export interface Position { latitude: number; longitude: number; altitude: number; }

export interface HopRecord {
    fromNodeId: string;
    toNodeId: string;
    latencyMs: number;
    timestampMs?: number;
    fromNodePosition?: Position;
    toNodePosition?: Position;
    distanceKm?: number;
    fromNodeBufferState: { queueSize: number; utilization: number; };
    routingDecisionInfo?: { algorithm: string;[ key: string ]: unknown; };
}

export interface Packet {
    packetId: string;
    sourceUserId?: string;
    destinationUserId?: string;
    isUseRL: boolean;
    serviceQoS: ServiceQoS;
    accumulatedDelayMs: number;
    dropped: boolean;
    hopRecords: HopRecord[];
    type?: string;
    timeSentFromSourceMs?: number;
    TTL?: number;
}

// ----------------------------------------------
// 2. CHART-SPECIFIC DATA INTERFACES
// ----------------------------------------------

/** Data for the KPI Bar Chart */
export interface KpiData { name: string; RL: number; NonRL: number; }

/** Data point for the Correlation Scatter Chart */
export interface CorrelationPoint { utilization: number; latencyMs: number; }

/** Data for the Latency Distribution Bar Chart */
export interface LatencyDistData { name: string; RL: number; NonRL: number; }

/** Data nodes/links for the Sankey Flow Chart */
export interface SankeyNode { name: string; }
export interface SankeyLink { source: number; target: number; value: number; }
export interface SankeyData { nodes: SankeyNode[]; links: SankeyLink[]; }

/** The final, processed data structure for the entire dashboard */
export interface ChartData {
    kpiData: KpiData[];
    correlationData: { rl: CorrelationPoint[]; nonRl: CorrelationPoint[]; };
    latencyDist: LatencyDistData[];
    sankeyDataRL: SankeyData;
    sankeyDataNonRL: SankeyData;
}

// ----------------------------------------------
// 3. INTERNAL PROCESSOR TYPES (Not exported to components)
// ----------------------------------------------
export type StatsCounter = { total: number; dropped: number; totalLatency: number; qosSuccess: number; };
export type SankeyAggregator = {
    nodes: Map<string, number>;
    links: Map<string, { source: string; target: string; value: number }>;
    nodeIndex: number;
};
export type AggregatedData = {
    stats: { rl: StatsCounter; nonRl: StatsCounter; };
    correlationData: { rl: CorrelationPoint[]; nonRl: CorrelationPoint[]; };
    latencyBins: { rl: number[]; nonRl: number[]; };
    sankey: { rl: SankeyAggregator; nonRl: SankeyAggregator; };
};

/**
 * Trạng thái khởi tạo rỗng cho các biểu đồ
 */
export const EMPTY_CHART_DATA: ChartData = {
  kpiData: [],
  correlationData: { rl: [], nonRl: [] },
  latencyDist: [],
  sankeyDataRL: { nodes: [], links: [] },
  sankeyDataNonRL: { nodes: [], links: [] }
};