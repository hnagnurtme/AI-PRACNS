import { AxiosError } from 'axios';
import axiosClient from '../api/axiosClient';
import type {
    UserTerminal,
    GenerateTerminalsRequest,
    ConnectTerminalRequest,
    TerminalConnectionResult,
} from '../types/UserTerminalTypes';

const TERMINALS_ENDPOINT = '/api/v1/terminals';

interface ApiResponse<T> {
    status: number;
    error: string | null;
    message: string;
    data: T;
}

const extractData = <T>(response: ApiResponse<T>): T => {
    if (response.status !== 200 || response.error) {
        throw new Error(response.message || `API call failed with status ${response.status}`);
    }
    return response.data;
};

const handleAxiosError = (error: unknown): Error => {
    if (error instanceof AxiosError) {
        if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
            return new Error('Cannot connect to server. Please make sure the backend is running on port 8080.');
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
 * Generate user terminals
 */
export const generateUserTerminals = async (
    request: GenerateTerminalsRequest
): Promise<UserTerminal[]> => {
    try {
        const response = await axiosClient.post<UserTerminal[] | ApiResponse<UserTerminal[]>>(
            `${TERMINALS_ENDPOINT}/generate`,
            request
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<UserTerminal[]>);
        }

        return response.data as UserTerminal[];
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Get all user terminals
 */
export const getUserTerminals = async (): Promise<UserTerminal[]> => {
    try {
        const response = await axiosClient.get<UserTerminal[] | ApiResponse<UserTerminal[]>>(
            TERMINALS_ENDPOINT
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<UserTerminal[]>);
        }

        return response.data as UserTerminal[];
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Get terminal by ID
 */
export const getTerminalById = async (terminalId: string): Promise<UserTerminal> => {
    try {
        const response = await axiosClient.get<UserTerminal | ApiResponse<UserTerminal>>(
            `${TERMINALS_ENDPOINT}/${terminalId}`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<UserTerminal>);
        }

        return response.data as UserTerminal;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Connect terminal to a node
 */
export const connectTerminalToNode = async (
    request: ConnectTerminalRequest
): Promise<TerminalConnectionResult> => {
    try {
        const response = await axiosClient.post<TerminalConnectionResult | ApiResponse<TerminalConnectionResult>>(
            `${TERMINALS_ENDPOINT}/${request.terminalId}/connect`,
            { nodeId: request.nodeId }
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<TerminalConnectionResult>);
        }

        return response.data as TerminalConnectionResult;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Disconnect terminal from node
 */
export const disconnectTerminal = async (terminalId: string): Promise<UserTerminal> => {
    try {
        const response = await axiosClient.post<UserTerminal | ApiResponse<UserTerminal>>(
            `${TERMINALS_ENDPOINT}/${terminalId}/disconnect`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<UserTerminal>);
        }

        return response.data as UserTerminal;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Get terminal connection result
 */
export const getTerminalConnectionResult = async (
    terminalId: string
): Promise<TerminalConnectionResult | null> => {
    try {
        const response = await axiosClient.get<TerminalConnectionResult | ApiResponse<TerminalConnectionResult>>(
            `${TERMINALS_ENDPOINT}/${terminalId}/connection-result`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return extractData(response.data as ApiResponse<TerminalConnectionResult>);
        }

        return response.data as TerminalConnectionResult;
    } catch (error) {
        // If terminal has no connection result, return null
        if (error instanceof AxiosError && error.response?.status === 404) {
            return null;
        }
        throw handleAxiosError(error);
    }
};

/**
 * Delete terminal
 */
export const deleteTerminal = async (terminalId: string): Promise<void> => {
    try {
        await axiosClient.delete(`${TERMINALS_ENDPOINT}/${terminalId}`);
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Delete all terminals
 */
export const deleteAllTerminals = async (): Promise<void> => {
    try {
        await axiosClient.delete(TERMINALS_ENDPOINT);
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Create terminal from map position (double-click)
 */
export const createTerminalFromMap = async (
    position: { latitude: number; longitude: number; altitude?: number },
    terminalType: string = 'MOBILE',
    terminalName?: string
): Promise<UserTerminal> => {
    try {
        const response = await axiosClient.post<{ success: boolean; terminal: UserTerminal; message: string }>(
            `${TERMINALS_ENDPOINT}/create`,
            {
                position,
                terminalType,
                terminalName
            }
        );

        console.log('✅ Terminal creation response:', response);

        // Backend returns 201 with success flag
        if (response.data && response.data.success && response.data.terminal) {
            return response.data.terminal;
        }

        // Fallback: if response.data is the terminal directly
        if (response.data && (response.data as any).terminalId) {
            return response.data as unknown as UserTerminal;
        }

        throw new Error(response.data?.message || 'Failed to create terminal');
    } catch (error) {
        console.error('❌ Terminal creation error:', error);
        throw handleAxiosError(error);
    }
};

