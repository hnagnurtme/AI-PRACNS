import React from 'react';
import type { AlgorithmComparison, NodeResourceInfo } from '../../types/RoutingTypes';

interface PathNodeResourcesProps {
    comparison: AlgorithmComparison;
}

export const PathNodeResources: React.FC<PathNodeResourcesProps> = ({ comparison }) => {
    if (!comparison.nodeResources || Object.keys(comparison.nodeResources).length === 0) {
        return null;
    }

    const getWarningStatus = (node: NodeResourceInfo) => {
        const warnings: string[] = [];
        let status: 'critical' | 'warning' | 'normal' = 'normal';

        // Check latency
        if (node.nodeProcessingDelayMs > 500) {
            warnings.push('High Latency');
            status = 'critical';
        } else if (node.nodeProcessingDelayMs > 200) {
            warnings.push('Moderate Latency');
            if (status === 'normal') status = 'warning';
        }

        // Check packet loss
        if (node.packetLossRate > 0.1) {
            warnings.push('High Packet Loss');
            status = 'critical';
        } else if (node.packetLossRate > 0.05) {
            warnings.push('Moderate Packet Loss');
            if (status === 'normal') status = 'warning';
        }

        // Check queue
        const queueRatio = node.packetBufferCapacity > 0 
            ? (node.currentPacketCount / node.packetBufferCapacity) * 100 
            : 0;
        if (queueRatio > 90) {
            warnings.push('Queue Nearly Full');
            status = 'critical';
        } else if (queueRatio > 70) {
            warnings.push('High Queue');
            if (status === 'normal') status = 'warning';
        }

        // Check utilization
        if (node.resourceUtilization > 90) {
            warnings.push('High Utilization');
            status = 'critical';
        } else if (node.resourceUtilization > 70) {
            warnings.push('Moderate Utilization');
            if (status === 'normal') status = 'warning';
        }

        return { status, warnings, queueRatio };
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'critical':
                return 'bg-red-100 text-red-800 border-red-300';
            case 'warning':
                return 'bg-yellow-100 text-yellow-800 border-yellow-300';
            default:
                return 'bg-green-100 text-green-800 border-green-300';
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'critical':
                return '🔴';
            case 'warning':
                return '⚠️';
            default:
                return '✅';
        }
    };

    // Get nodes from both paths
    const path1NodeIds = comparison.algorithm1.path.path
        .filter(seg => seg.type === 'node')
        .map(seg => seg.id);
    const path2NodeIds = comparison.algorithm2.path.path
        .filter(seg => seg.type === 'node')
        .map(seg => seg.id);
    const allNodeIds = Array.from(new Set([...path1NodeIds, ...path2NodeIds]));

    const nodes = allNodeIds
        .map(nodeId => comparison.nodeResources?.[nodeId])
        .filter((node): node is NodeResourceInfo => node !== undefined);

    if (nodes.length === 0) {
        return null;
    }

    return (
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <span>📊</span>
                <span>Node Resource Statistics</span>
            </h3>
            
            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
                        <tr>
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Node Name
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Type
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Status
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Latency (ms)
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Packet Loss (%)
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Queue
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Utilization (%)
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Battery (%)
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                                Warnings
                            </th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {nodes.map((node) => {
                            const { status, warnings, queueRatio } = getWarningStatus(node);
                            const inPath1 = path1NodeIds.includes(node.nodeId);
                            const inPath2 = path2NodeIds.includes(node.nodeId);

                            return (
                                <tr
                                    key={node.nodeId}
                                    className={`hover:bg-gray-50 transition-colors ${
                                        status === 'critical' ? 'bg-red-50' :
                                        status === 'warning' ? 'bg-yellow-50' : ''
                                    }`}
                                >
                                    <td className="px-4 py-3 whitespace-nowrap">
                                        <div className="text-sm font-medium text-gray-900">{node.nodeName}</div>
                                        <div className="text-xs text-gray-500">{node.nodeId}</div>
                                        <div className="flex gap-1 mt-1">
                                            {inPath1 && (
                                                <span className="px-1.5 py-0.5 text-xs bg-blue-100 text-blue-800 rounded">
                                                    {comparison.algorithm1.name}
                                                </span>
                                            )}
                                            {inPath2 && (
                                                <span className="px-1.5 py-0.5 text-xs bg-orange-100 text-orange-800 rounded">
                                                    {comparison.algorithm2.name}
                                                </span>
                                            )}
                                        </div>
                                    </td>
                                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                                        {node.nodeType}
                                    </td>
                                    <td className="px-4 py-3 whitespace-nowrap">
                                        <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                            node.isOperational 
                                                ? 'bg-green-100 text-green-800' 
                                                : 'bg-red-100 text-red-800'
                                        }`}>
                                            {node.isOperational ? 'Online' : 'Offline'}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 whitespace-nowrap">
                                        <span className={`text-sm font-semibold ${
                                            node.nodeProcessingDelayMs > 500 ? 'text-red-600' :
                                            node.nodeProcessingDelayMs > 200 ? 'text-yellow-600' :
                                            'text-green-600'
                                        }`}>
                                            {node.nodeProcessingDelayMs.toFixed(2)}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 whitespace-nowrap">
                                        <span className={`text-sm font-semibold ${
                                            node.packetLossRate > 0.1 ? 'text-red-600' :
                                            node.packetLossRate > 0.05 ? 'text-yellow-600' :
                                            'text-green-600'
                                        }`}>
                                            {(node.packetLossRate * 100).toFixed(2)}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 whitespace-nowrap">
                                        <div className="flex items-center gap-2">
                                            <span className={`text-sm font-semibold ${
                                                queueRatio > 90 ? 'text-red-600' :
                                                queueRatio > 70 ? 'text-yellow-600' :
                                                'text-green-600'
                                            }`}>
                                                {node.currentPacketCount} / {node.packetBufferCapacity}
                                            </span>
                                            <div className="w-20 bg-gray-200 rounded-full h-2">
                                                <div
                                                    className={`h-2 rounded-full transition-all ${
                                                        queueRatio > 90 ? 'bg-red-500' :
                                                        queueRatio > 70 ? 'bg-yellow-500' : 'bg-green-500'
                                                    }`}
                                                    style={{ width: `${Math.min(100, queueRatio)}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-4 py-3 whitespace-nowrap">
                                        <div className="flex items-center gap-2">
                                            <span className={`text-sm font-semibold ${
                                                node.resourceUtilization > 90 ? 'text-red-600' :
                                                node.resourceUtilization > 70 ? 'text-yellow-600' :
                                                'text-green-600'
                                            }`}>
                                                {node.resourceUtilization.toFixed(1)}
                                            </span>
                                            <div className="w-20 bg-gray-200 rounded-full h-2">
                                                <div
                                                    className={`h-2 rounded-full transition-all ${
                                                        node.resourceUtilization > 90 ? 'bg-red-500' :
                                                        node.resourceUtilization > 70 ? 'bg-yellow-500' : 'bg-green-500'
                                                    }`}
                                                    style={{ width: `${node.resourceUtilization}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-4 py-3 whitespace-nowrap">
                                        <div className="flex items-center gap-2">
                                            <span className={`text-sm font-semibold ${
                                                node.batteryChargePercent > 50 ? 'text-green-600' :
                                                node.batteryChargePercent > 20 ? 'text-yellow-600' : 'text-red-600'
                                            }`}>
                                                {node.batteryChargePercent.toFixed(1)}
                                            </span>
                                            <div className="w-20 bg-gray-200 rounded-full h-2">
                                                <div
                                                    className={`h-2 rounded-full transition-all ${
                                                        node.batteryChargePercent > 50 ? 'bg-green-500' :
                                                        node.batteryChargePercent > 20 ? 'bg-yellow-500' : 'bg-red-500'
                                                    }`}
                                                    style={{ width: `${node.batteryChargePercent}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-4 py-3">
                                        {warnings.length > 0 ? (
                                            <div className="flex flex-col gap-1">
                                                {warnings.map((warning, idx) => (
                                                    <span
                                                        key={idx}
                                                        className={`px-2 py-0.5 rounded text-xs border ${getStatusColor(status)}`}
                                                    >
                                                        {getStatusIcon(status)} {warning}
                                                    </span>
                                                ))}
                                            </div>
                                        ) : (
                                            <span className="text-green-600 text-xs">✅ Normal</span>
                                        )}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

