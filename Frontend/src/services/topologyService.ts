import { AxiosError } from 'axios';
import axiosClient from '../api/axiosClient';
import type { NetworkTopology, NetworkStatistics, NetworkConnection } from '../types/NetworkTopologyTypes';

const TOPOLOGY_ENDPOINT = '/api/v1/topology';

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
 * Get complete network topology
 */
export const getTopology = async (): Promise<NetworkTopology> => {
    try {
        const response = await axiosClient.get<NetworkTopology | ApiResponse<NetworkTopology>>(
            TOPOLOGY_ENDPOINT
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<NetworkTopology>).data;
        }

        return response.data as NetworkTopology;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Get network statistics only
 */
export const getTopologyStatistics = async (): Promise<NetworkStatistics> => {
    try {
        const response = await axiosClient.get<NetworkStatistics | ApiResponse<NetworkStatistics>>(
            `${TOPOLOGY_ENDPOINT}/statistics`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<NetworkStatistics>).data;
        }

        return response.data as NetworkStatistics;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Get network connections only
 */
export const getTopologyConnections = async (): Promise<NetworkConnection[]> => {
    try {
        const response = await axiosClient.get<NetworkConnection[] | ApiResponse<NetworkConnection[]>>(
            `${TOPOLOGY_ENDPOINT}/connections`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<NetworkConnection[]>).data;
        }

        return response.data as NetworkConnection[];
    } catch (error) {
        throw handleAxiosError(error);
    }
};

