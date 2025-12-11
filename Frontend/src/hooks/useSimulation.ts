import { useState, useEffect, useCallback } from 'react';
import { getScenarios, getCurrentScenario, setScenario, resetScenario, getSimulationMetrics, startSimulation, stopSimulation } from '../services/simulationService';
import type { SimulationScenario, ScenarioState, SimulationMetrics } from '../types/SimulationTypes';

interface UseSimulationReturn {
    scenarios: SimulationScenario[];
    currentScenario: ScenarioState | null;
    metrics: SimulationMetrics | null;
    loading: boolean;
    error: Error | null;
    changeScenario: (scenarioName: string) => Promise<void>;
    resetToNormal: () => Promise<void>;
    startSim: (scenarioName: string) => Promise<void>;
    stopSim: () => Promise<void>;
    refreshMetrics: () => Promise<void>;
}

/**
 * Hook to manage simulation scenarios
 */
export const useSimulation = (): UseSimulationReturn => {
    const [scenarios, setScenarios] = useState<SimulationScenario[]>([]);
    const [currentScenario, setCurrentScenario] = useState<ScenarioState | null>(null);
    const [metrics, setMetrics] = useState<SimulationMetrics | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    // Fetch scenarios on mount
    useEffect(() => {
        const loadScenarios = async () => {
            try {
                const data = await getScenarios();
                setScenarios(data);
            } catch (err) {
                setError(err instanceof Error ? err : new Error('Failed to load scenarios'));
            }
        };
        loadScenarios();
    }, []);

    // Fetch current scenario on mount
    useEffect(() => {
        const loadCurrentScenario = async () => {
            try {
                const data = await getCurrentScenario();
                setCurrentScenario(data);
            } catch (err) {
                console.error('Failed to load current scenario:', err);
            }
        };
        loadCurrentScenario();
    }, []);

    // Refresh metrics periodically
    useEffect(() => {
        const refreshMetrics = async () => {
            try {
                const data = await getSimulationMetrics();
                setMetrics(data);
            } catch (err) {
                console.error('Failed to load metrics:', err);
            }
        };

        refreshMetrics();
        const interval = setInterval(refreshMetrics, 5000); // Refresh every 5 seconds
        return () => clearInterval(interval);
    }, []);

    const changeScenario = useCallback(async (scenarioName: string) => {
        setLoading(true);
        setError(null);
        try {
            const result = await setScenario(scenarioName);
            setCurrentScenario(result);
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to change scenario');
            setError(error);
            throw error;
        } finally {
            setLoading(false);
        }
    }, []);

    const resetToNormal = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const result = await resetScenario();
            setCurrentScenario(result);
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to reset scenario');
            setError(error);
            throw error;
        } finally {
            setLoading(false);
        }
    }, []);

    const startSim = useCallback(async (scenarioName: string) => {
        setLoading(true);
        setError(null);
        try {
            const result = await startSimulation(scenarioName);
            setCurrentScenario(result);
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to start simulation');
            setError(error);
            throw error;
        } finally {
            setLoading(false);
        }
    }, []);

    const stopSim = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            await stopSimulation();
            const result = await getCurrentScenario();
            setCurrentScenario(result);
        } catch (err) {
            const error = err instanceof Error ? err : new Error('Failed to stop simulation');
            setError(error);
            throw error;
        } finally {
            setLoading(false);
        }
    }, []);

    const refreshMetrics = useCallback(async () => {
        try {
            const data = await getSimulationMetrics();
            setMetrics(data);
        } catch (err) {
            console.error('Failed to refresh metrics:', err);
        }
    }, []);

    return {
        scenarios,
        currentScenario,
        metrics,
        loading,
        error,
        changeScenario,
        resetToNormal,
        startSim,
        stopSim,
        refreshMetrics,
    };
};

