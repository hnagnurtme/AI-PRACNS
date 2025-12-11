import { useState, useEffect } from 'react';
import { useWebSocket } from '../contexts/WebSocketContext';
import type { ScenarioState, SimulationMetrics } from '../types/SimulationTypes';

interface UseSimulationWebSocketReturn {
    currentScenario: ScenarioState | null;
    metrics: SimulationMetrics | null;
    isConnected: boolean;
    error: Error | null;
}

/**
 * Hook to subscribe to simulation updates via WebSocket
 */
export const useSimulationWebSocket = (): UseSimulationWebSocketReturn => {
    const { isConnected, subscribeToTopologyUpdates } = useWebSocket();
    const [currentScenario, _setCurrentScenario] = useState<ScenarioState | null>(null);
    const [metrics, _setMetrics] = useState<SimulationMetrics | null>(null);
    const [error, _setError] = useState<Error | null>(null);

    useEffect(() => {
        if (!isConnected) return;

        const unsubscribe = subscribeToTopologyUpdates((_update) => {
            // Check if this is a simulation update
            // Note: This depends on how backend sends simulation updates
            // For now, we'll handle it through a separate subscription if needed
            // The actual implementation depends on backend WebSocket structure
        });

        return unsubscribe;
    }, [isConnected, subscribeToTopologyUpdates]);

    return {
        currentScenario,
        metrics,
        isConnected,
        error,
    };
};

/**
 * Hook to manage simulation scenarios with WebSocket updates
 */
export const useSimulation = () => {
    const [scenarios, _setScenarios] = useState<any[]>([]);
    const [currentScenario, setCurrentScenario] = useState<ScenarioState | null>(null);
    const [metrics, setMetrics] = useState<SimulationMetrics | null>(null);
    const [loading, _setLoading] = useState(false);
    const [error, _setError] = useState<Error | null>(null);
    const { isConnected } = useWebSocket();

    // Subscribe to simulation updates via WebSocket
    useEffect(() => {
        if (!isConnected) return;

        // Note: This would need backend to send simulation updates via WebSocket
        // For now, we'll use polling or direct API calls
        // The WebSocket subscription can be added when backend supports it
    }, [isConnected]);

    return {
        scenarios,
        currentScenario,
        metrics,
        loading,
        error,
        setCurrentScenario,
        setMetrics,
    };
};

