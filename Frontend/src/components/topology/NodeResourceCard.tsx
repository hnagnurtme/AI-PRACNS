import React from 'react';
import type { NodeDTO } from '../../types/NodeTypes';

interface NodeResourceCardProps {
    node: NodeDTO;
}

const NodeResourceCard: React.FC<NodeResourceCardProps> = ({ node }) => {
    // Calculate warning levels
    const getLatencyWarning = (latency: number) => {
        if (latency > 500) return { level: 'critical', color: 'red', text: 'Rất cao' };
        if (latency > 200) return { level: 'warning', color: 'yellow', text: 'Cao' };
        return { level: 'normal', color: 'green', text: 'Bình thường' };
    };

    const getPacketLossWarning = (lossRate: number) => {
        if (lossRate > 0.1) return { level: 'critical', color: 'red', text: 'Rất cao' };
        if (lossRate > 0.05) return { level: 'warning', color: 'yellow', text: 'Cao' };
        return { level: 'normal', color: 'green', text: 'Bình thường' };
    };

    const getQueueWarning = (current: number, capacity: number) => {
        const ratio = capacity > 0 ? (current / capacity) * 100 : 0;
        if (ratio > 90) return { level: 'critical', color: 'red', text: 'Gần đầy' };
        if (ratio > 70) return { level: 'warning', color: 'yellow', text: 'Cao' };
        return { level: 'normal', color: 'green', text: 'Bình thường' };
    };

    const getUtilizationWarning = (utilization: number) => {
        if (utilization > 90) return { level: 'critical', color: 'red', text: 'Rất cao' };
        if (utilization > 70) return { level: 'warning', color: 'yellow', text: 'Cao' };
        return { level: 'normal', color: 'green', text: 'Bình thường' };
    };

    const latencyWarning = getLatencyWarning(node.nodeProcessingDelayMs);
    const packetLossWarning = getPacketLossWarning(node.packetLossRate);
    const queueWarning = getQueueWarning(node.currentPacketCount, node.packetBufferCapacity);
    const utilizationWarning = getUtilizationWarning(node.resourceUtilization);

    const getStatusColor = (level: string) => {
        switch (level) {
            case 'critical':
                return 'bg-red-100 text-red-800 border-red-300';
            case 'warning':
                return 'bg-yellow-100 text-yellow-800 border-yellow-300';
            default:
                return 'bg-green-100 text-green-800 border-green-300';
        }
    };

    return (
        <div className="bg-white border-2 border-gray-200 rounded-lg p-4 shadow-md">
            <div className="flex justify-between items-start mb-3">
                <div>
                    <h3 className="text-lg font-bold text-gray-800">{node.nodeName}</h3>
                    <p className="text-sm text-gray-600">{node.nodeType}</p>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-semibold ${
                    node.isOperational ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                    {node.isOperational ? 'Online' : 'Offline'}
                </div>
            </div>

            <div className="space-y-3">
                {/* Processing Delay (Latency) */}
                <div className="border-l-4 border-violet-300 pl-3 bg-violet-50/30 rounded-r-lg py-2">
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-violet-900">Processing Latency</span>
                        <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${getStatusColor(latencyWarning.level)}`}>
                            {latencyWarning.level === 'critical' ? 'Critical' : latencyWarning.level === 'warning' ? 'Warning' : 'Normal'}
                        </span>
                    </div>
                    <div className="text-lg font-bold text-gray-900">{node.nodeProcessingDelayMs} ms</div>
                </div>

                {/* Packet Loss */}
                <div className="border-l-4 border-violet-300 pl-3 bg-violet-50/30 rounded-r-lg py-2">
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-violet-900">Packet Loss Rate</span>
                        <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${getStatusColor(packetLossWarning.level)}`}>
                            {packetLossWarning.level === 'critical' ? 'Critical' : packetLossWarning.level === 'warning' ? 'Warning' : 'Normal'}
                        </span>
                    </div>
                    <div className="text-lg font-bold text-gray-900">{(node.packetLossRate * 100).toFixed(2)}%</div>
                </div>

                {/* Queue Status */}
                <div className="border-l-4 border-violet-300 pl-3 bg-violet-50/30 rounded-r-lg py-2">
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-violet-900">Queue Buffer</span>
                        <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${getStatusColor(queueWarning.level)}`}>
                            {queueWarning.level === 'critical' ? 'Near Full' : queueWarning.level === 'warning' ? 'High' : 'Normal'}
                        </span>
                    </div>
                    <div className="text-lg font-bold text-gray-900">
                        {node.currentPacketCount} / {node.packetBufferCapacity}
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                        <div
                            className={`h-2 rounded-full ${
                                queueWarning.level === 'critical' ? 'bg-red-500' :
                                queueWarning.level === 'warning' ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{
                                width: `${Math.min(100, (node.currentPacketCount / node.packetBufferCapacity) * 100)}%`
                            }}
                        ></div>
                    </div>
                </div>

                {/* Resource Utilization */}
                <div className="border-l-4 border-violet-300 pl-3 bg-violet-50/30 rounded-r-lg py-2">
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-violet-900">CPU/Memory Usage</span>
                        <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${getStatusColor(utilizationWarning.level)}`}>
                            {utilizationWarning.level === 'critical' ? 'Critical' : utilizationWarning.level === 'warning' ? 'Warning' : 'Normal'}
                        </span>
                    </div>
                    <div className="text-lg font-bold text-gray-900">{node.resourceUtilization}%</div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                        <div
                            className={`h-2 rounded-full ${
                                utilizationWarning.level === 'critical' ? 'bg-red-500' :
                                utilizationWarning.level === 'warning' ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${node.resourceUtilization}%` }}
                        ></div>
                    </div>
                </div>

                {/* Battery */}
                <div className="border-l-4 border-violet-300 pl-3 bg-violet-50/30 rounded-r-lg py-2">
                    <span className="text-sm font-medium text-violet-900">Battery Level</span>
                    <div className="text-lg font-bold text-gray-900">{node.batteryChargePercent}%</div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                        <div
                            className={`h-2 rounded-full ${
                                node.batteryChargePercent > 50 ? 'bg-green-500' :
                                node.batteryChargePercent > 20 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${node.batteryChargePercent}%` }}
                        ></div>
                    </div>
                </div>

                {/* Communication Info */}
                <div className="pt-2 border-t border-gray-200">
                    <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                            <span className="text-gray-600">Max Range:</span>
                            <span className="font-semibold ml-1">{node.communication?.maxRangeKm || 0} km</span>
                        </div>
                        <div>
                            <span className="text-gray-600">Bandwidth:</span>
                            <span className="font-semibold ml-1">{(node.communication as any)?.maxBandwidthMbps || (node.communication?.bandwidthMHz ? node.communication.bandwidthMHz / 1000 : 0)} Mbps</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default NodeResourceCard;

