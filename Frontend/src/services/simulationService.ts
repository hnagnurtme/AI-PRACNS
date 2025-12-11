import { AxiosError } from 'axios';
import axiosClient from '../api/axiosClient';
import type { SimulationScenario, ScenarioState, ScenarioParameters, SimulationMetrics } from '../types/SimulationTypes';

const SIMULATION_ENDPOINT = '/api/v1/simulation';

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
 * Get all available simulation scenarios
 */
export const getScenarios = async (): Promise<SimulationScenario[]> => {
    try {
        const response = await axiosClient.get<SimulationScenario[] | ApiResponse<SimulationScenario[]>>(
            `${SIMULATION_ENDPOINT}/scenarios`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<SimulationScenario[]>).data;
        }

        // If backend returns array of strings, convert to SimulationScenario[]
        const data = response.data as any;
        if (Array.isArray(data) && typeof data[0] === 'string') {
            return data.map((name: string) => ({
                name: name as any,
                displayName: name.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
                description: `${name.replace(/_/g, ' ').toLowerCase()} scenario`,
            }));
        }

        return response.data as SimulationScenario[];
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Get current active scenario
 */
export const getCurrentScenario = async (): Promise<ScenarioState> => {
    try {
        const response = await axiosClient.get<ScenarioState | ApiResponse<ScenarioState>>(
            `${SIMULATION_ENDPOINT}/scenario/current`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<ScenarioState>).data;
        }

        return response.data as ScenarioState;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Set active scenario
 */
export const setScenario = async (scenarioName: string, parameters?: ScenarioParameters): Promise<ScenarioState> => {
    try {
        const response = await axiosClient.post<ScenarioState | ApiResponse<ScenarioState>>(
            `${SIMULATION_ENDPOINT}/scenario/${scenarioName}`,
            parameters ? { parameters } : {}
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<ScenarioState>).data;
        }

        return response.data as ScenarioState;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Reset scenario to NORMAL
 */
export const resetScenario = async (): Promise<ScenarioState> => {
    try {
        const response = await axiosClient.post<ScenarioState | ApiResponse<ScenarioState>>(
            `${SIMULATION_ENDPOINT}/scenario/reset`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<ScenarioState>).data;
        }

        return response.data as ScenarioState;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Get simulation metrics
 */
export const getSimulationMetrics = async (): Promise<SimulationMetrics> => {
    try {
        const response = await axiosClient.get<SimulationMetrics | ApiResponse<SimulationMetrics>>(
            `${SIMULATION_ENDPOINT}/metrics`
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<SimulationMetrics>).data;
        }

        return response.data as SimulationMetrics;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Start simulation
 */
export const startSimulation = async (scenarioName: string, parameters?: ScenarioParameters): Promise<ScenarioState> => {
    try {
        const response = await axiosClient.post<ScenarioState | ApiResponse<ScenarioState>>(
            `${SIMULATION_ENDPOINT}/start`,
            {
                scenario: scenarioName,
                parameters,
            }
        );

        if (response.data && typeof response.data === 'object' && 'data' in response.data) {
            return (response.data as ApiResponse<ScenarioState>).data;
        }

        return response.data as ScenarioState;
    } catch (error) {
        throw handleAxiosError(error);
    }
};

/**
 * Stop simulation
 */
export const stopSimulation = async (): Promise<void> => {
    try {
        await axiosClient.post(`${SIMULATION_ENDPOINT}/stop`);
    } catch (error) {
        throw handleAxiosError(error);
    }
};

