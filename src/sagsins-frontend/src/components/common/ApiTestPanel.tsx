// src/components/common/ApiTestPanel.tsx

import React, { useState } from 'react';
import { checkHealth } from '../../services/nodeService';
import type { HealthResponse } from '../../types/NodeTypes';

const ApiTestPanel: React.FC = () => {
    const [healthStatus, setHealthStatus] = useState<HealthResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleCheckHealth = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const health = await checkHealth();
            setHealthStatus(health);
        } catch (err) {
            setError((err as Error).message);
        } finally {
            setIsLoading(false);
        }
    };

    // Simplified: only health endpoint is available

    return (
        <div className="bg-white p-6 rounded-xl shadow-lg">
            <h3 className="text-lg font-bold mb-4 text-gray-800">🔧 API Test Panel</h3>
            
            {error && (
                <div className="mb-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded">
                    Error: {error}
                </div>
            )}

            <div className="space-y-4">
                {/* Health Check */}
                <div className="border rounded p-4">
                    <h4 className="font-semibold mb-2">Health Check</h4>
                    <button 
                        onClick={handleCheckHealth}
                        disabled={isLoading}
                        className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded disabled:opacity-50"
                    >
                        {isLoading ? 'Checking...' : 'Check Health'}
                    </button>
                    {healthStatus && (
                        <div className="mt-2 p-2 bg-gray-100 rounded">
                            <div>Status: <span className="font-semibold">{healthStatus.status}</span></div>
                            <div>Message: {healthStatus.message}</div>
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
};

export default ApiTestPanel;