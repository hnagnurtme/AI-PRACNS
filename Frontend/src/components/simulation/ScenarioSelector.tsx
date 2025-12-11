import React, { useState, useEffect } from 'react';
import { getScenarios, getCurrentScenario, setScenario, resetScenario } from '../../services/simulationService';
import type { SimulationScenario, ScenarioState } from '../../types/SimulationTypes';

interface ScenarioSelectorProps {
    onScenarioChange?: (scenario: ScenarioState) => void;
}

export const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({ 
    onScenarioChange
}) => {
    const [scenarios, setScenarios] = useState<SimulationScenario[]>([]);
    const [currentScenario, setCurrentScenario] = useState<ScenarioState | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Fetch available scenarios
    useEffect(() => {
        fetchScenarios();
        fetchCurrentScenario();
    }, []);

    const fetchScenarios = async () => {
        try {
            const data = await getScenarios();
            setScenarios(data);
        } catch (err) {
            console.error('Error fetching scenarios:', err);
            setError('Failed to load scenarios');
        }
    };

    const fetchCurrentScenario = async () => {
        try {
            const data = await getCurrentScenario();
            setCurrentScenario(data);
            if (onScenarioChange) {
                onScenarioChange(data);
            }
        } catch (err) {
            console.error('Error fetching current scenario:', err);
        }
    };

    const handleScenarioChange = async (scenarioName: string) => {
        setLoading(true);
        setError(null);

        try {
            const result = await setScenario(scenarioName);
            setCurrentScenario(result);
            if (onScenarioChange) {
                onScenarioChange(result);
            }
        } catch (err) {
            console.error('Error changing scenario:', err);
            setError(err instanceof Error ? err.message : 'Failed to change scenario');
        } finally {
            setLoading(false);
        }
    };

    const handleReset = async () => {
        setLoading(true);
        setError(null);

        try {
            const result = await resetScenario();
            setCurrentScenario(result);
            if (onScenarioChange) {
                onScenarioChange(result);
            }
        } catch (err) {
            console.error('Error resetting scenario:', err);
            setError(err instanceof Error ? err.message : 'Failed to reset scenario');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white rounded-lg shadow p-4 mb-6">
            <div className="flex items-center justify-between">
                <div className="flex-1">
                    <label htmlFor="scenario-select" className="block text-sm font-medium text-gray-700 mb-2">
                        Simulation Scenario
                    </label>
                    <select
                        id="scenario-select"
                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        value={currentScenario?.scenario || ''}
                        onChange={(e) => handleScenarioChange(e.target.value)}
                        disabled={loading}
                    >
                        <option value="">Select scenario...</option>
                        {scenarios.map((scenario) => (
                            <option key={scenario.name} value={scenario.name}>
                                {scenario.displayName}
                            </option>
                        ))}
                    </select>
                </div>

                <button
                    onClick={handleReset}
                    disabled={loading || currentScenario?.scenario === 'NORMAL'}
                    className="ml-4 px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                >
                    Reset to Normal
                </button>
            </div>

            {currentScenario && (
                <div className="mt-4 p-3 bg-blue-50 rounded-md border border-blue-200">
                    <div className="flex items-start">
                        <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div className="ml-3">
                            <p className="text-sm font-medium text-blue-800">
                                Current: {currentScenario.displayName}
                            </p>
                            <p className="text-sm text-blue-700 mt-1">
                                {currentScenario.description}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {error && (
                <div className="mt-4 p-3 bg-red-50 rounded-md border border-red-200">
                    <p className="text-sm text-red-800">{error}</p>
                </div>
            )}

            {loading && (
                <div className="mt-4 flex items-center justify-center">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                    <span className="ml-2 text-sm text-gray-600">Applying scenario...</span>
                </div>
            )}
        </div>
    );
};
