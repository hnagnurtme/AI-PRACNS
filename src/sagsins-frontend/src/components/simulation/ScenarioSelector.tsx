import React, { useState, useEffect } from 'react';
import type { SimulationScenario, ScenarioState } from '../../types/ComparisonTypes';

interface ScenarioSelectorProps {
    baseUrl?: string;
}

export const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({ 
    baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8080'
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
            const response = await fetch(`${baseUrl}/api/simulation/scenarios`);
            if (!response.ok) throw new Error('Failed to fetch scenarios');
            const data = await response.json();
            
            // Transform enum data to scenario objects
            const scenarioList = data.map((name: string) => ({
                name,
                displayName: name.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
                description: `${name.replace(/_/g, ' ').toLowerCase()} scenario`
            }));
            
            setScenarios(scenarioList);
        } catch (err) {
            console.error('Error fetching scenarios:', err);
            setError('Failed to load scenarios');
        }
    };

    const fetchCurrentScenario = async () => {
        try {
            const response = await fetch(`${baseUrl}/api/simulation/scenario/current`);
            if (!response.ok) throw new Error('Failed to fetch current scenario');
            const data = await response.json();
            setCurrentScenario(data);
        } catch (err) {
            console.error('Error fetching current scenario:', err);
        }
    };

    const handleScenarioChange = async (scenarioName: string) => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${baseUrl}/api/simulation/scenario/${scenarioName}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error('Failed to change scenario');
            }

            const result = await response.json();
            if (result.success) {
                await fetchCurrentScenario();
            } else {
                throw new Error(result.error || 'Failed to change scenario');
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
            const response = await fetch(`${baseUrl}/api/simulation/scenario/reset`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error('Failed to reset scenario');
            }

            const result = await response.json();
            if (result.success) {
                await fetchCurrentScenario();
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
