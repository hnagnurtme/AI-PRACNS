import React from 'react';
import type { RoutingPath } from '../../types/RoutingTypes';
import type { NodeDTO } from '../../types/NodeTypes';

interface PathDetailCardProps {
    path: RoutingPath;
    nodes: NodeDTO[];
    onClose?: () => void;
}

const PathDetailCard: React.FC<PathDetailCardProps> = ({ path, nodes, onClose }) => {
    // Get node details for each node in path
    const getNodeDetails = (nodeId: string): NodeDTO | undefined => {
        return nodes.find(n => n.nodeId === nodeId);
    };

    // Calculate total resource utilization across path
    const calculatePathMetrics = () => {
        const nodeSegments = path.path.filter(seg => seg.type === 'node');
        let totalUtilization = 0;
        let totalPacketLoss = 0;
        let totalBattery = 0;
        let maxLatency = 0;
        let nodeCount = 0;

        nodeSegments.forEach(segment => {
            const node = getNodeDetails(segment.id);
            if (node) {
                totalUtilization += node.resourceUtilization || 0;
                totalPacketLoss += node.packetLossRate || 0;
                totalBattery += node.batteryChargePercent || 100;
                maxLatency = Math.max(maxLatency, node.nodeProcessingDelayMs || 0);
                nodeCount++;
            }
        });

        return {
            avgUtilization: nodeCount > 0 ? totalUtilization / nodeCount : 0,
            avgPacketLoss: nodeCount > 0 ? totalPacketLoss / nodeCount : 0,
            avgBattery: nodeCount > 0 ? totalBattery / nodeCount : 0,
            maxLatency,
            nodeCount
        };
    };

    const metrics = calculatePathMetrics();

    const getAlgorithmName = (algo?: string) => {
        switch (algo) {
            case 'rl': return 'Reinforcement Learning';
            case 'dijkstra': return 'Dijkstra';
            case 'simple': return 'Simple';
            default: return 'Unknown';
        }
    };

    const getAlgorithmColor = (algo?: string) => {
        switch (algo) {
            case 'rl': return 'bg-purple-100 text-purple-800 border-purple-400';         // Tím gradient - RL
            case 'dijkstra': return 'bg-blue-100 text-blue-700 border-blue-400';         // Xanh dương - Dijkstra
            case 'simple': return 'bg-teal-100 text-teal-700 border-teal-400';           // Xanh lục - Simple
            default: return 'bg-gray-100 text-gray-700 border-gray-300';
        }
    };

    const getUtilizationColor = (util: number) => {
        if (util < 50) return 'text-green-600';
        if (util < 80) return 'text-yellow-600';
        return 'text-red-600';
    };

    const getBatteryColor = (battery: number) => {
        if (battery > 80) return 'text-green-600';
        if (battery > 50) return 'text-yellow-600';
        return 'text-red-600';
    };

    return (
        <div className="absolute top-20 right-4 z-30 w-96 bg-white rounded-lg shadow-2xl border border-gray-200 p-6 max-h-[80vh] overflow-y-auto">
            {/* Header */}
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold text-gray-800">📊 Path Details</h3>
                <button
                    onClick={onClose}
                    className="text-gray-500 hover:text-red-600 text-2xl font-bold transition-colors"
                    title="Close"
                >
                    &times;
                </button>
            </div>

            {/* Algorithm Badge */}
            <div className={`mb-4 px-3 py-1 rounded-lg border text-sm font-semibold inline-block ${getAlgorithmColor(path.algorithm)}`}>
                {getAlgorithmName(path.algorithm)}
            </div>

            {/* Path Overview */}
            <div className="mb-4 space-y-2">
                <div className="flex items-center gap-2 p-2 bg-emerald-50 rounded-lg border border-emerald-300">
                    <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                    <div className="flex-1">
                        <div className="text-xs text-gray-500">Source</div>
                        <div className="font-semibold text-gray-800">Terminal {path.source.terminalId}</div>
                    </div>
                </div>
                <div className="flex items-center gap-2 p-2 bg-orange-50 rounded-lg border border-orange-300">
                    <div className="w-3 h-3 rounded-full bg-orange-600"></div>
                    <div className="flex-1">
                        <div className="text-xs text-gray-500">Destination</div>
                        <div className="font-semibold text-gray-800">Terminal {path.destination.terminalId}</div>
                    </div>
                </div>
            </div>

            {/* Path Metrics */}
            <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-sm font-semibold text-blue-800 mb-2">Path Metrics</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                        <span className="text-gray-600">Distance:</span>
                        <span className="font-semibold ml-1">{path.totalDistance.toFixed(2)} km</span>
                    </div>
                    <div>
                        <span className="text-gray-600">Latency:</span>
                        <span className="font-semibold ml-1">{path.estimatedLatency.toFixed(0)} ms</span>
                    </div>
                    <div>
                        <span className="text-gray-600">Hops:</span>
                        <span className="font-semibold ml-1">{path.hops}</span>
                    </div>
                    <div>
                        <span className="text-gray-600">Nodes:</span>
                        <span className="font-semibold ml-1">{metrics.nodeCount}</span>
                    </div>
                </div>
            </div>

            {/* Resource Status */}
            <div className="mb-4">
                <div className="text-sm font-semibold text-gray-800 mb-2">Resource Status</div>
                <div className="space-y-2">
                    <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                        <span className="text-sm text-gray-700">Avg Utilization:</span>
                        <span className={`font-semibold ${getUtilizationColor(metrics.avgUtilization)}`}>
                            {metrics.avgUtilization.toFixed(1)}%
                        </span>
                    </div>
                    <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                        <span className="text-sm text-gray-700">Avg Packet Loss:</span>
                        <span className="font-semibold text-gray-800">
                            {(metrics.avgPacketLoss * 100).toFixed(2)}%
                        </span>
                    </div>
                    <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                        <span className="text-sm text-gray-700">Avg Battery:</span>
                        <span className={`font-semibold ${getBatteryColor(metrics.avgBattery)}`}>
                            {metrics.avgBattery.toFixed(1)}%
                        </span>
                    </div>
                    <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                        <span className="text-sm text-gray-700">Max Node Latency:</span>
                        <span className="font-semibold text-gray-800">
                            {metrics.maxLatency.toFixed(0)} ms
                        </span>
                    </div>
                </div>
            </div>

            {/* Path Segments */}
            <div className="mb-4">
                <div className="text-sm font-semibold text-gray-800 mb-2">Path Segments ({path.path.length})</div>
                <div className="space-y-1 max-h-48 overflow-y-auto">
                    {path.path.map((segment, index) => {
                        const node = segment.type === 'node' ? getNodeDetails(segment.id) : null;
                        return (
                            <div
                                key={index}
                                className="flex items-center gap-2 p-2 bg-gray-50 rounded text-xs"
                            >
                                <div className={`w-2 h-2 rounded-full ${
                                    segment.type === 'terminal' ? 'bg-blue-500' : 'bg-purple-500'
                                }`}></div>
                                <div className="flex-1">
                                    <div className="font-semibold text-gray-800">{segment.name}</div>
                                    {node && (
                                        <div className="text-gray-600 text-xs mt-0.5">
                                            Util: {node.resourceUtilization?.toFixed(1)}% | 
                                            Battery: {node.batteryChargePercent?.toFixed(0)}% | 
                                            Loss: {(node.packetLossRate * 100)?.toFixed(2)}%
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Quality Indicator */}
            <div className={`p-3 rounded-lg border ${
                metrics.avgUtilization < 70 && metrics.avgPacketLoss < 0.05 && metrics.avgBattery > 50
                    ? 'bg-green-50 border-green-200'
                    : metrics.avgUtilization < 85 && metrics.avgPacketLoss < 0.1
                    ? 'bg-yellow-50 border-yellow-200'
                    : 'bg-red-50 border-red-200'
            }`}>
                <div className="text-sm font-semibold mb-1">
                    {metrics.avgUtilization < 70 && metrics.avgPacketLoss < 0.05 && metrics.avgBattery > 50
                        ? '✅ Path Quality: Excellent'
                        : metrics.avgUtilization < 85 && metrics.avgPacketLoss < 0.1
                        ? '⚠️ Path Quality: Good'
                        : '❌ Path Quality: Poor'
                    }
                </div>
                <div className="text-xs text-gray-600">
                    {metrics.avgUtilization < 70 && metrics.avgPacketLoss < 0.05 && metrics.avgBattery > 50
                        ? 'All resources are in good condition'
                        : metrics.avgUtilization < 85 && metrics.avgPacketLoss < 0.1
                        ? 'Some resources may be under stress'
                        : 'Path has resource constraints'
                    }
                </div>
            </div>
        </div>
    );
};

export default PathDetailCard;

