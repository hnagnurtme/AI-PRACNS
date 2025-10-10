// src/components/nodes/NodeDetailCard.tsx

import React, { useState } from 'react';

import { useNodeStore } from '../../state/nodeStore';
import { runNodeProcess } from '../../services/nodeService';
import NodeForm from './NodeForm';
import type { NodeDTO } from '../../types/NodeTypes';

interface NodeDetailCardProps {
    node: NodeDTO;
    onRefresh: () => Promise<NodeDTO[]>;
}

const NodeDetailCard: React.FC<NodeDetailCardProps> = ({ node, onRefresh }) => {
    
    const { setSelectedNode, cameraFollowMode, setCameraFollowMode, runningNodes, setNodeRunning } = useNodeStore();
    const [isEditFormOpen, setIsEditFormOpen] = useState(false);
    const [processMessage, setProcessMessage] = useState<string | null>(null);
    
    const isRunning = runningNodes.has(node.nodeId);

    const handleCameraToggle = () => {
        setCameraFollowMode(!cameraFollowMode);
    };

    const handleRunProcess = async () => {
        setNodeRunning(node.nodeId, true);
        setProcessMessage(null);
        try {
            await runNodeProcess(node.nodeId);
            setProcessMessage(`✅ Process started successfully for ${node.nodeId}`);
            
            // Keep showing RUNNING status for a few seconds after success
            setTimeout(() => {
                setNodeRunning(node.nodeId, false);
                setProcessMessage(null);
            }, 5000);
        } catch (error) {
            setProcessMessage(`❌ Failed to start process: ${(error as Error).message}`);
            setNodeRunning(node.nodeId, false);
            setTimeout(() => setProcessMessage(null), 5000);
        }
    };
    
    return (
        <>
            <div className="bg-white p-5 rounded-xl shadow-xl border border-gray-200">
                <div className="flex justify-between items-start mb-3">
                    <h3 className="text-xl font-extrabold text-gray-800 truncate">Node: {node.nodeId.substring(0, 8)}...</h3>
                    <button 
                        onClick={() => setSelectedNode(null)}
                        className="text-gray-500 hover:text-red-600 text-lg font-bold"
                    >
                        &times;
                    </button>
                </div>
                
                <p className="text-sm mb-4">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${isRunning ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {isRunning ? 'RUNNING' : 'STOPPED'}
                    </span>
                    {(node.operational !== undefined || node.isOperational !== undefined) && (
                        <span className={`ml-2 px-2 py-0.5 rounded-full text-xs font-semibold ${
                            (node.operational ?? node.isOperational) ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
                        }`}>
                            {(node.operational ?? node.isOperational) ? 'OPERATIONAL' : 'OFFLINE'}
                        </span>
                    )}
                    <span className="ml-2 text-gray-500">({node.nodeType})</span>
                    {cameraFollowMode && (
                        <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-100 text-blue-700 animate-pulse">
                            📹 CAMERA TRACKING
                        </span>
                    )}
                </p>

                <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                    <DetailItem label="Latitude" value={node.position.latitude.toFixed(4)} />
                    <DetailItem label="Longitude" value={node.position.longitude.toFixed(4)} />
                    <DetailItem label="Altitude (Km)" value={node.position.altitude.toFixed(2)} />
                    
                    {/* New API Fields */}
                    {node.batteryChargePercent !== undefined && (
                        <DetailItem label="Battery" value={`${node.batteryChargePercent.toFixed(1)}%`} />
                    )}
                    {node.nodeProcessingDelayMs !== undefined && (
                        <DetailItem label="Processing Delay" value={`${node.nodeProcessingDelayMs.toFixed(2)} ms`} />
                    )}
                    {node.packetLossRate !== undefined && (
                        <DetailItem label="Loss Rate" value={(node.packetLossRate * 100).toFixed(3) + '%'} />
                    )}
                    {node.resourceUtilization !== undefined && (
                        <DetailItem label="Resource Usage" value={`${node.resourceUtilization.toFixed(1)}%`} />
                    )}
                    {node.packetBufferCapacity !== undefined && node.currentPacketCount !== undefined && (
                        <DetailItem label="Buffer" value={`${node.currentPacketCount}/${node.packetBufferCapacity}`} />
                    )}
                    {node.weather && (
                        <DetailItem label="Weather" value={node.weather.replace(/_/g, ' ')} />
                    )}
                    {node.host && node.port && (
                        <DetailItem label="Host:Port" value={`${node.host}:${node.port}`} />
                    )}
                    
                    {/* Legacy Fields */}
                    {node.currentBandwidth !== undefined && (
                        <DetailItem label="Bandwidth" value={`${node.currentBandwidth.toFixed(2)} Mbps`} />
                    )}
                    {node.avgLatencyMs !== undefined && (
                        <DetailItem label="Latency" value={`${node.avgLatencyMs.toFixed(2)} ms`} />
                    )}
                    {node.currentThroughput !== undefined && (
                        <DetailItem label="Throughput" value={`${node.currentThroughput.toFixed(2)} Mbps`} />
                    )}
                    {node.powerLevel !== undefined && (
                        <DetailItem label="Power Level" value={`${node.powerLevel.toFixed(1)}%`} />
                    )}
                </div>

                {/* Process Status Message */}
                {processMessage && (
                    <div className={`mt-3 p-2 rounded text-sm ${
                        processMessage.includes('✅') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                    }`}>
                        {processMessage}
                    </div>
                )}

                {/* Last Updated */}
                {node.lastUpdated && (
                    <div className="mt-4 pt-3 border-t border-gray-200">
                        <h4 className="text-sm font-semibold text-gray-700 mb-2">📊 Status</h4>
                        <div className="text-xs text-gray-500">
                            Last Updated: {new Date(node.lastUpdated).toLocaleString()}
                        </div>
                    </div>
                )}

                {/* Orbital Information */}
                {(node.orbit || node.velocity) && (
                    <div className="mt-4 pt-3 border-t border-gray-200">
                        <h4 className="text-sm font-semibold text-gray-700 mb-2">🛰️ Orbital Data</h4>
                        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                            {node.orbit && (
                                <>
                                    <DetailItem label="Semi-major Axis" value={`${node.orbit.semiMajorAxisKm.toFixed(0)} km`} />
                                    <DetailItem label="Eccentricity" value={node.orbit.eccentricity.toFixed(4)} />
                                    <DetailItem label="Inclination" value={`${node.orbit.inclinationDeg.toFixed(2)}°`} />
                                    <DetailItem label="RAAN" value={`${node.orbit.raanDeg.toFixed(2)}°`} />
                                </>
                            )}
                            {node.velocity && (
                                <>
                                    <DetailItem label="Velocity X" value={`${node.velocity.velocityX.toFixed(2)} km/s`} />
                                    <DetailItem label="Velocity Y" value={`${node.velocity.velocityY.toFixed(2)} km/s`} />
                                    <DetailItem label="Velocity Z" value={`${node.velocity.velocityZ.toFixed(2)} km/s`} />
                                </>
                            )}
                        </div>
                    </div>
                )}

                {/* Action Buttons */}
                <div className="mt-5 flex flex-wrap gap-2">
                    <button 
                        onClick={handleCameraToggle}
                        className={`text-sm font-medium px-3 py-1 rounded border transition-colors ${
                            cameraFollowMode 
                                ? 'bg-green-600 text-white border-green-600 hover:bg-green-700' 
                                : 'text-green-600 border-green-200 hover:bg-green-50'
                        }`}
                    >
                        📹 {cameraFollowMode ? 'Following' : 'Follow Cam'}
                    </button>
                    
                    <button 
                        onClick={handleRunProcess}
                        disabled={isRunning}
                        className="text-sm text-orange-600 hover:text-orange-800 font-medium px-3 py-1 rounded border border-orange-200 hover:bg-orange-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isRunning ? '⏳ Starting...' : '🚀 Run Process'}
                    </button>
                    
                    <button 
                        onClick={() => setIsEditFormOpen(true)}
                        className="text-sm text-blue-600 hover:text-blue-800 font-medium px-3 py-1 rounded border border-blue-200 hover:bg-blue-50"
                    >
                        ✏️ Edit
                    </button>
                </div>
            </div>

            {/* Modal cho Form EDIT Node */}
            {isEditFormOpen && (
                <div className="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center p-4 z-50">
                    <NodeForm 
                        onClose={() => setIsEditFormOpen(false)} 
                        mode="update"
                        initialNode={node}
                        onSuccess={onRefresh}
                    />
                </div>
            )}
        </>
    );
};

// Component phụ trợ
const DetailItem: React.FC<{ label: string; value: string }> = ({ label, value }) => (
    <div>
        <dt className="text-gray-500">{label}</dt>
        <dd className="font-medium text-gray-900">{value}</dd>
    </div>
);

export default NodeDetailCard;