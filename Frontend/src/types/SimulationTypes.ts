export type ScenarioType = 
    | 'NORMAL' 
    | 'HIGH_CONGESTION' 
    | 'LOW_BANDWIDTH' 
    | 'HIGH_LATENCY' 
    | 'NODE_FAILURE' 
    | 'NETWORK_PARTITION'
    | 'RL_OPTIMIZED_PATH'
    | 'STANDARD_ROUTING';

export interface SimulationScenario {
    name: ScenarioType;
    displayName: string;
    description: string;
    parameters?: ScenarioParameters;
}

export interface ScenarioParameters {
    congestionLevel?: number; // 0-1
    bandwidthReduction?: number; // 0-1 (percentage reduction)
    latencyMultiplier?: number; // multiplier for latency
    nodeFailureRate?: number; // 0-1 (percentage of nodes to fail)
    networkPartitionProbability?: number; // 0-1
}

export interface ScenarioState {
    scenario: ScenarioType;
    displayName: string;
    description?: string;
    parameters?: ScenarioParameters;
    startedAt?: string;
    isActive: boolean;
}

export interface SimulationMetrics {
    totalPackets: number;
    successfulPackets: number;
    failedPackets: number;
    averageLatency: number;
    averageDistance: number;
    averageHops: number;
    algorithmPerformance: {
        dijkstra: {
            avgLatency: number;
            successRate: number;
            avgHops: number;
        };
        rl: {
            avgLatency: number;
            successRate: number;
            avgHops: number;
        };
    };
    timestamp: string;
}

export interface SimulationUpdate {
    type: 'scenario_changed' | 'metrics_updated' | 'scenario_started' | 'scenario_stopped';
    scenario?: ScenarioState;
    metrics?: SimulationMetrics;
    timestamp: string;
}

