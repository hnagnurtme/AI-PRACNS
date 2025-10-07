// src/components/nodes/NodeDetailCard.tsx

import React from 'react';

import { useNodeStore } from '../../state/nodeStore';
import type { NodeDTO } from '../../types/NodeTypes';

interface NodeDetailCardProps {
    node: NodeDTO;
}

const NodeDetailCard: React.FC<NodeDetailCardProps> = ({ node }) => {
    
    const { setSelectedNode } = useNodeStore();
    
    return (
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
                <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${node.isHealthy ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                    {node.isHealthy ? 'HEALTHY' : 'UNHEALTHY'}
                </span>
                <span className="ml-2 text-gray-500">({node.nodeType})</span>
            </p>

            <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                <DetailItem label="Latitude" value={node.position.latitude.toFixed(4)} />
                <DetailItem label="Longitude" value={node.position.longitude.toFixed(4)} />
                <DetailItem label="Altitude (Km)" value={node.position.altitude.toFixed(2)} />
                <DetailItem label="Bandwidth" value={`${node.currentBandwidth.toFixed(2)} Mbps`} />
                <DetailItem label="Latency" value={`${node.avgLatencyMs.toFixed(2)} ms`} />
                <DetailItem label="Loss Rate" value={(node.packetLossRate * 100).toFixed(2) + '%'} />
            </div>

            {/* Action Buttons */}
            <div className="mt-5 flex justify-end space-x-2">
                <button className="text-sm text-blue-600 hover:text-blue-800">Edit</button>
                <button className="text-sm text-red-600 hover:text-red-800">Delete</button>
            </div>
        </div>
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