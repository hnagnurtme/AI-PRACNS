import { AxiosError } from 'axios';
import axiosClient from '../api/axiosClient';
import type { NodeAnalysis } from '../types/NodeAnalysisTypes';

const NODE_ANALYSIS_ENDPOINT = '/api/v1/topology';

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

/**
 * Get node analysis (best links, upcoming satellites, degrading nodes)
 */
export const getNodeAnalysis = async (nodeId: string): Promise<NodeAnalysis> => {
    try {
        const response = await axiosClient.get<NodeAnalysis | ApiResponse<NodeAnalysis>>(
            `${NODE_ANALYSIS_ENDPOINT}/nodes/${nodeId}/analysis`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<NodeAnalysis>).data;
        }

        return response.data as NodeAnalysis;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

