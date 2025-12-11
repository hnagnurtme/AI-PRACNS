import React from 'react';
import type { NodeDTO } from '../../types/NodeTypes';

interface NodeResourceTableProps {
    nodes: NodeDTO[];
    selectedNodeIds?: Set<string>;
    onNodeClick?: (node: NodeDTO) => void;
}

const NodeResourceTable: React.FC<NodeResourceTableProps> = ({ nodes, selectedNodeIds, onNodeClick }) => {
    // Filter nodes if selection is active
    const displayNodes = selectedNodeIds && selectedNodeIds.size > 0
        ? nodes.filter(node => selectedNodeIds.has(node.nodeId))
        : nodes;

    // Get warning status
    const getWarningStatus = (node: NodeDTO) => {
        const warnings: string[] = [];
        let status: 'critical' | 'warning' | 'normal' = 'normal';

        // Check latency
        if (node.nodeProcessingDelayMs > 500) {
            warnings.push(`Critical Latency: ${node.nodeProcessingDelayMs.toFixed(0)}ms`);
            status = 'critical';
        } else if (node.nodeProcessingDelayMs > 200) {
            warnings.push(`High Latency: ${node.nodeProcessingDelayMs.toFixed(0)}ms`);
            if (status === 'normal') status = 'warning';
        }

        // Check packet loss
        if (node.packetLossRate > 0.1) {
            warnings.push(`Critical Loss: ${(node.packetLossRate * 100).toFixed(1)}%`);
            status = 'critical';
        } else if (node.packetLossRate > 0.05) {
            warnings.push(`High Loss: ${(node.packetLossRate * 100).toFixed(2)}%`);
            if (status === 'normal') status = 'warning';
        }

        // Check queue
        const queueRatio = node.packetBufferCapacity > 0 
            ? (node.currentPacketCount / node.packetBufferCapacity) * 100 
            : 0;
        if (queueRatio > 90) {
            warnings.push(`Queue Near Full: ${queueRatio.toFixed(0)}%`);
            status = 'critical';
        } else if (queueRatio > 70) {
            warnings.push(`Queue High: ${queueRatio.toFixed(0)}%`);
            if (status === 'normal') status = 'warning';
        }

        // Check utilization
        if (node.resourceUtilization > 90) {
            warnings.push(`Critical Load: ${node.resourceUtilization.toFixed(0)}%`);
            status = 'critical';
        } else if (node.resourceUtilization > 70) {
            warnings.push(`High Load: ${node.resourceUtilization.toFixed(0)}%`);
            if (status === 'normal') status = 'warning';
        }

        return { status, warnings };
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

    return (
        <div className="overflow-x-auto rounded-lg border border-violet-200">
            <table className="min-w-full bg-white">
                <thead className="bg-gradient-to-r from-violet-50 to-fuchsia-50">
                    <tr>
                        <th className="px-4 py-3 text-left text-xs font-bold text-violet-900 uppercase tracking-wider border-b border-violet-200">
                            Node Name
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-bold text-violet-900 uppercase tracking-wider border-b border-violet-200">
                            Type
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-bold text-violet-900 uppercase tracking-wider border-b border-violet-200">
                            Operational
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-bold text-violet-900 uppercase tracking-wider border-b border-violet-200">
                            Process Delay
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-bold text-violet-900 uppercase tracking-wider border-b border-violet-200">
                            Loss Rate
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-bold text-violet-900 uppercase tracking-wider border-b border-violet-200">
                            Queue Buffer
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-bold text-violet-900 uppercase tracking-wider border-b border-violet-200">
                            CPU/Memory
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-bold text-violet-900 uppercase tracking-wider border-b border-violet-200">
                            Battery
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-bold text-violet-900 uppercase tracking-wider border-b border-violet-200">
                            Health Alerts
                        </th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                    {displayNodes.length === 0 ? (
                        <tr>
                            <td colSpan={9} className="px-4 py-8 text-center text-gray-500">
                                No nodes to display
                            </td>
                        </tr>
                    ) : (
                        displayNodes.map((node) => {
                            const { status, warnings } = getWarningStatus(node);
                            const queueRatio = node.packetBufferCapacity > 0 
                                ? (node.currentPacketCount / node.packetBufferCapacity) * 100 
                                : 0;

                            return (
                                <tr
                                    key={node.nodeId}
                                    onClick={() => onNodeClick?.(node)}
                                    className={`hover:bg-gray-50 cursor-pointer transition-colors ${
                                        status === 'critical' ? 'bg-red-50' :
                                        status === 'warning' ? 'bg-yellow-50' : ''
                                    }`}
                                >
                                    <td className="px-4 py-3 text-sm font-medium text-gray-900">
                                        {node.nodeName}
                                    </td>
                                    <td className="px-4 py-3 text-sm text-gray-600">
                                        {node.nodeType}
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                        <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                            node.isOperational 
                                                ? 'bg-green-100 text-green-800' 
                                                : 'bg-red-100 text-red-800'
                                        }`}>
                                            {node.isOperational ? 'Online' : 'Offline'}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                        <span className={`font-semibold ${
                                            node.nodeProcessingDelayMs > 500 ? 'text-red-600' :
                                            node.nodeProcessingDelayMs > 200 ? 'text-yellow-600' :
                                            'text-green-600'
                                        }`}>
                                            {node.nodeProcessingDelayMs.toFixed(2)}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                        <span className={`font-semibold ${
                                            node.packetLossRate > 0.1 ? 'text-red-600' :
                                            node.packetLossRate > 0.05 ? 'text-yellow-600' :
                                            'text-green-600'
                                        }`}>
                                            {(node.packetLossRate * 100).toFixed(2)}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                        <div className="flex items-center gap-2">
                                            <span className={`font-semibold ${
                                                queueRatio > 90 ? 'text-red-600' :
                                                queueRatio > 70 ? 'text-yellow-600' :
                                                'text-green-600'
                                            }`}>
                                                {node.currentPacketCount} / {node.packetBufferCapacity}
                                            </span>
                                            <div className="w-16 bg-gray-200 rounded-full h-2">
                                                <div
                                                    className={`h-2 rounded-full ${
                                                        queueRatio > 90 ? 'bg-red-500' :
                                                        queueRatio > 70 ? 'bg-yellow-500' : 'bg-green-500'
                                                    }`}
                                                    style={{ width: `${Math.min(100, queueRatio)}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                        <div className="flex items-center gap-2">
                                            <span className={`font-semibold ${
                                                node.resourceUtilization > 90 ? 'text-red-600' :
                                                node.resourceUtilization > 70 ? 'text-yellow-600' :
                                                'text-green-600'
                                            }`}>
                                                {node.resourceUtilization.toFixed(1)}
                                            </span>
                                            <div className="w-16 bg-gray-200 rounded-full h-2">
                                                <div
                                                    className={`h-2 rounded-full ${
                                                        node.resourceUtilization > 90 ? 'bg-red-500' :
                                                        node.resourceUtilization > 70 ? 'bg-yellow-500' : 'bg-green-500'
                                                    }`}
                                                    style={{ width: `${node.resourceUtilization}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                        <div className="flex items-center gap-2">
                                            <span className={`font-semibold ${
                                                node.batteryChargePercent > 50 ? 'text-green-600' :
                                                node.batteryChargePercent > 20 ? 'text-yellow-600' : 'text-red-600'
                                            }`}>
                                                {node.batteryChargePercent.toFixed(1)}
                                            </span>
                                            <div className="w-16 bg-gray-200 rounded-full h-2">
                                                <div
                                                    className={`h-2 rounded-full ${
                                                        node.batteryChargePercent > 50 ? 'bg-green-500' :
                                                        node.batteryChargePercent > 20 ? 'bg-yellow-500' : 'bg-red-500'
                                                    }`}
                                                    style={{ width: `${node.batteryChargePercent}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                        {warnings.length > 0 ? (
                                            <div className="flex flex-col gap-1.5">
                                                {warnings.map((warning, idx) => (
                                                    <span
                                                        key={idx}
                                                        className={`px-2.5 py-1 rounded-md text-xs font-semibold border shadow-sm ${getStatusColor(status)}`}
                                                    >
                                                        {getStatusIcon(status)} {warning}
                                                    </span>
                                                ))}
                                            </div>
                                        ) : (
                                            <span className="text-green-600 text-xs font-medium bg-green-50 px-2 py-1 rounded-md">✅ All Systems Normal</span>
                                        )}
                                    </td>
                                </tr>
                            );
                        })
                    )}
                </tbody>
            </table>
        </div>
    );
};

export default NodeResourceTable;

