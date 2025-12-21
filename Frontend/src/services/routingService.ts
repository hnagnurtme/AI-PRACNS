import { AxiosError } from 'axios';
import axiosClient from '../api/axiosClient';
import type { RoutingPath, SendPacketRequest, Packet, AlgorithmComparison, CompareAlgorithmsRequest } from '../types/RoutingTypes';

const ROUTING_ENDPOINT = '/api/v1/routing';

interface ApiResponse<T> {
    status: number;
    error: string | null;
    message: string;
    data: T;
}

const handleAxiosError = ( error: unknown ): Error => {
    if ( error instanceof AxiosError ) {
        if ( error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK' ) {
            return new Error( 'Cannot connect to server. Please make sure the backend is running.' );
        }

        if ( error.response ) {
            return new Error(
                `Server error: ${ error.response.status } - ${ error.response.data?.message || error.message }`
            );
        } else if ( error.request ) {
            return new Error( 'No response from server. Please check your connection.' );
        }
    }

    if ( error instanceof Error ) {
        return error;
    }

    return new Error( 'An unexpected error occurred.' );
};

/**
 * Calculate path from source terminal to destination terminal
 */
export const calculatePath = async (
    sourceTerminalId: string,
    destinationTerminalId: string,
    algorithm: 'dijkstra' | 'rl' = 'rl'
): Promise<RoutingPath> => {
    try {
        const response = await axiosClient.post<RoutingPath | ApiResponse<RoutingPath>>(
            `${ ROUTING_ENDPOINT }/calculate-path`,
            {
                sourceTerminalId,
                destinationTerminalId,
                algorithm,
            }
        );

        if ( response.data && typeof response.data === 'object' && 'data' in response.data ) {
            return ( response.data as ApiResponse<RoutingPath> ).data;
        }

        return response.data as RoutingPath;
    } catch ( error ) {
        throw handleAxiosError( error );
    }
};

/**
 * Send a packet from source terminal to destination terminal
 */
export const sendPacket = async ( request: SendPacketRequest & { algorithm?: 'dijkstra' | 'rl' } ): Promise<Packet> => {
    try {
        const response = await axiosClient.post<Packet | ApiResponse<Packet>>(
            `${ ROUTING_ENDPOINT }/send-packet`,
            request
        );

        if ( response.data && typeof response.data === 'object' && 'data' in response.data ) {
            return ( response.data as ApiResponse<Packet> ).data;
        }

        return response.data as Packet;
    } catch ( error ) {
        throw handleAxiosError( error );
    }
};

/**
 * Compare two routing algorithms for a given source and destination terminal
 */
export const compareAlgorithms = async ( request: CompareAlgorithmsRequest ): Promise<AlgorithmComparison> => {
    try {
        const response = await axiosClient.post<AlgorithmComparison | ApiResponse<AlgorithmComparison>>(
            `${ ROUTING_ENDPOINT }/compare-algorithms`,
            request
        );

        if ( response.data && typeof response.data === 'object' && 'data' in response.data ) {
            return ( response.data as ApiResponse<AlgorithmComparison> ).data;
        }

        return response.data as AlgorithmComparison;
    } catch ( error ) {
        throw handleAxiosError( error );
    }
};

