// src/components/common/ApiTestPanel.tsx

import React, { useState } from 'react';
import { 
    checkHealth, 
    getDockerEntities, 
    runNodeProcess 
} from '../../services/nodeService';
import type { HealthResponse, DockerResponse } from '../../types/NodeTypes';

const ApiTestPanel: React.FC = () => {
    const [healthStatus, setHealthStatus] = useState<HealthResponse | null>(null);
    const [dockerEntities, setDockerEntities] = useState<DockerResponse[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedNodeId, setSelectedNodeId] = useState('');

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

    const handleGetDockerEntities = async (isRunning: boolean) => {
        setIsLoading(true);
        setError(null);
        try {
            const entities = await getDockerEntities(isRunning);
            setDockerEntities(entities);
        } catch (err) {
            setError((err as Error).message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleRunNodeProcess = async () => {
        if (!selectedNodeId.trim()) {
            setError('Please enter a Node ID');
            return;
        }
        
        setIsLoading(true);
        setError(null);
        try {
            await runNodeProcess(selectedNodeId);
            alert(`Successfully started process for node: ${selectedNodeId}`);
        } catch (err) {
            setError((err as Error).message);
        } finally {
            setIsLoading(false);
        }
    };

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

                {/* Docker Entities */}
                <div className="border rounded p-4">
                    <h4 className="font-semibold mb-2">Docker Entities</h4>
                    <div className="flex space-x-2 mb-2">
                        <button 
                            onClick={() => handleGetDockerEntities(true)}
                            disabled={isLoading}
                            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded disabled:opacity-50"
                        >
                            Get Running
                        </button>
                        <button 
                            onClick={() => handleGetDockerEntities(false)}
                            disabled={isLoading}
                            className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded disabled:opacity-50"
                        >
                            Get All
                        </button>
                    </div>
                    {dockerEntities.length > 0 && (
                        <div className="mt-2 space-y-1">
                            {dockerEntities.map((entity, index) => (
                                <div key={index} className="p-2 bg-gray-100 rounded text-sm">
                                    <div>Node ID: {entity.nodeId}</div>
                                    <div>PID: {entity.pid}</div>
                                    <div>Status: <span className="font-semibold">{entity.status}</span></div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Run Node Process */}
                <div className="border rounded p-4">
                    <h4 className="font-semibold mb-2">Run Node Process</h4>
                    <div className="flex space-x-2">
                        <input
                            type="text"
                            placeholder="Enter Node ID"
                            value={selectedNodeId}
                            onChange={(e) => setSelectedNodeId(e.target.value)}
                            className="flex-1 px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                        <button 
                            onClick={handleRunNodeProcess}
                            disabled={isLoading || !selectedNodeId.trim()}
                            className="bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded disabled:opacity-50"
                        >
                            {isLoading ? 'Starting...' : 'Run Process'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ApiTestPanel;