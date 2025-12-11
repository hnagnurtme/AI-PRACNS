import { AxiosError } from 'axios';
import axiosClient from '../api/axiosClient';
import type { NetworkBatch } from '../types/ComparisonTypes';

const BATCH_ENDPOINT = '/api/v1/batch';

interface ApiResponse<T> {
    status: number;
    error: string | null;
    message: string;
    data: T;
}

const handleAxiosError = (error: unknown): Error => {
    if (error instanceof AxiosError) {
        if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
            return new Error('Cannot connect to server. Please make sure the backend is running.');
        }

        if (error.response) {
            return new Error(
                `Server error: ${error.response.status} - ${error.response.data?.message || error.message}`
            );
        } else if (error.request) {
            return new Error('No response from server. Please check your connection.');
        }
    }

    if (error instanceof Error) {
        return error;
    }

    return new Error('An unexpected error occurred.');
};

export interface GenerateBatchRequest {
    pairCount?: number;
    scenario?: string;
}

export interface BatchSuggestion {
    suggestedPairCount: number;
    suggestedPairs: Array<{
        type: string;
        sourceTerminalId: string;
        sourceTerminalName: string;
        destTerminalId: string;
        destTerminalName: string;
        distance_km: number;
        reason: string;
        priority: string;
    }>;
    linkQualityPrediction?: {
        pairInfo: {
            source: string;
            destination: string;
            distance_km: number;
        };
        bestTime: string;
        bestQuality: number;
        worstTime: string;
        worstQuality: number;
        averageQuality: number;
        predictions: Array<{
            timestamp: string;
            quality: number;
            latency_ms: number;
            snr_db: number;
            satellite: string;
        }>;
    };
    nodePlacementRecommendations?: {
        currentCoverage: number;
        gapsIdentified: number;
        locations: Array<{
            rank: number;
            type: string;
            latitude: number;
            longitude: number;
            priorityScore: number;
            reason: string;
            benefits: string[];
        }>;
    };
    overloadAnalysis: {
        overloadedNodes: Array<{
            nodeName: string;
            nodeType: string;
            utilization: number;
            packetLoss: number;
            overloadScore: number;
        }>;
        atRiskNodes: number;
        recommendations: Array<{
            priority: string;
            message: string;
            suggestions: string[];
        }>;
    };
    networkHealth: {
        totalNodes: number;
        totalTerminals: number;
        overloadedNodes: number;
        overloadPercentage: number;
        availableCapacity: number;
    };
    recommendations: Array<{
        type: string;
        message: string;
        suggestions: string[];
    }>;
}

/**
 * Get batch parameter suggestions based on network analysis
 */
export const getBatchSuggestions = async (): Promise<BatchSuggestion> => {
    try {
        const response = await axiosClient.get<BatchSuggestion | ApiResponse<BatchSuggestion>>(
            `${BATCH_ENDPOINT}/suggest-batch-params`
        );

        // Handle both direct data and wrapped data
        if ('data' in response.data && response.data.data) {
            return (response.data as ApiResponse<BatchSuggestion>).data;
        }

        return response.data as BatchSuggestion;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Generate a batch of packet pairs
 */
export const generateBatch = async (request: GenerateBatchRequest = {}): Promise<NetworkBatch> => {
    try {
        const response = await axiosClient.post<NetworkBatch | ApiResponse<NetworkBatch>>(
            `${BATCH_ENDPOINT}/generate`,
            {
                pairCount: request.pairCount || 10,
                scenario: request.scenario || 'NORMAL',
            }
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<NetworkBatch>).data;
        }

        return response.data as NetworkBatch;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Get algorithm comparison history
 */
export interface ComparisonHistoryParams {
    limit?: number;
    skip?: number;
    scenario?: string;
    algorithm1?: string;
    algorithm2?: string;
}

export interface ComparisonHistoryResponse {
    total: number;
    limit: number;
    skip: number;
    count: number;
    comparisons: any[];
}

export const getComparisonsHistory = async (params: ComparisonHistoryParams = {}): Promise<ComparisonHistoryResponse> => {
    try {
        const queryParams = new URLSearchParams();
        if (params.limit) queryParams.append('limit', params.limit.toString());
        if (params.skip) queryParams.append('skip', params.skip.toString());
        if (params.scenario) queryParams.append('scenario', params.scenario);
        if (params.algorithm1) queryParams.append('algorithm1', params.algorithm1);
        if (params.algorithm2) queryParams.append('algorithm2', params.algorithm2);

        const response = await axiosClient.get<ComparisonHistoryResponse>(
            `${BATCH_ENDPOINT}/history/comparisons?${queryParams.toString()}`
        );

        return response.data;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Get aggregate statistics from comparison history
 */
export interface ComparisonStats {
    totalComparisons: number;
    avgLatencyDiff: number;
    avgDistanceDiff: number;
    avgHopsDiff: number;
    scenarios: string[];
    algorithmPairs: Array<{ alg1: string; alg2: string }>;
    pairStatistics: any[];
}

export const getComparisonsStats = async (scenario?: string): Promise<ComparisonStats> => {
    try {
        const queryParams = scenario ? `?scenario=${scenario}` : '';
        const response = await axiosClient.get<ComparisonStats>(
            `${BATCH_ENDPOINT}/history/comparisons/stats${queryParams}`
        );

        return response.data;
    } catch (error) {
        throw handleAxiosError(error);
    }
};
