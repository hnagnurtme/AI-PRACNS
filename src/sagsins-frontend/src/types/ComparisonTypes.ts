export interface Position {
    latitude: number;
    longitude: number;
    altitude: number;
}

export interface BufferState {
    queueSize: number;
    bandwidthUtilization: number; // 0.0 - 1.0
}

export interface RoutingDecisionInfo {
    algorithm: "Dijkstra" | "ReinforcementLearning";
    metric?: string;
    reward?: number;
}

export interface HopRecord {
    fromNodeId: string;
    toNodeId: string;
    latencyMs: number;
    timestampMs: number;
    distanceKm: number;
    fromNodePosition: Position | null;
    toNodePosition: Position | null;
    fromNodeBufferState: BufferState;
    routingDecisionInfo: RoutingDecisionInfo;
    scenarioType?: string;
    nodeLoadPercent?: number;
    dropReasonDetails?: string;
}

export interface QoS {
    serviceType: string;
    defaultPriority: number;
    maxLatencyMs: number;
    maxJitterMs: number;
    minBandwidthMbps: number;
    maxLossRate: number;
}

export interface AnalysisData {
    avgLatency: number;
    avgDistanceKm: number;
    routeSuccessRate: number;
    totalDistanceKm: number;
    totalLatencyMs: number;
}

export interface Packet {
    packetId: string;
    sourceUserId: string;
    destinationUserId: string;
    stationSource: string;
    stationDest: string;
    type: string;
    acknowledgedPacketId?: string | null;
    timeSentFromSourceMs: number;
    payloadDataBase64: string;
    payloadSizeByte: number;
    serviceQoS: QoS;
    currentHoldingNodeId: string;
    nextHopNodeId: string;
    pathHistory: string[];
    hopRecords: HopRecord[];
    accumulatedDelayMs: number;
    priorityLevel: number;
    maxAcceptableLatencyMs: number;
    maxAcceptableLossRate: number;
    dropped: boolean;
    dropReason?: string | null;
    analysisData: AnalysisData;
    useRL: boolean;
    ttl: number;
}

export interface ComparisonData {
    dijkstraPacket: Packet | null;
    rlPacket: Packet | null;  // ✅ Fixed: Match server response (capital P)
}

export interface NetworkBatch {
    batchId: string;
    totalPairPackets: number;
    packets: ComparisonData[];
}

export interface SimulationScenario {
    name: string;
    displayName: string;
    description: string;
}

export interface ScenarioState {
    scenario: string;
    displayName: string;
    description: string;
}

export interface NodeCongestion {
    nodeId: string;
    packetsThrough: string[];
    totalPackets: number;
    avgQueueSize: number;
    avgBandwidthUtil: number;
    avgLatency: number;
    algorithms: { dijkstra: number; rl: number };
}