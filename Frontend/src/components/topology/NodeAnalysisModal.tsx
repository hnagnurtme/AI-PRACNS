import React from 'react';
import {
    Radar,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    ResponsiveContainer,
    Legend,
    Tooltip,
} from 'recharts';
import type { NodeAnalysis } from '../../types/NodeAnalysisTypes';
import type { NodeDTO } from '../../types/NodeTypes';

interface NodeAnalysisModalProps {
    node: NodeDTO | null;
    analysis: NodeAnalysis | null;
    loading: boolean;
    error: Error | null;
    onClose: () => void;
}

const NodeAnalysisModal: React.FC<NodeAnalysisModalProps> = ({
    node,
    analysis,
    loading,
    error,
    onClose,
}) => {
    // Prepare radar chart data
    const radarData = React.useMemo(() => {
        if (!node) return [];

        const queueRatio =
            node.packetBufferCapacity > 0
                ? (node.currentPacketCount / node.packetBufferCapacity) * 100
                : 0;

        return [
            {
                subject: 'Latency',
                value: Math.min(100, (200 - node.nodeProcessingDelayMs) / 2), // Invert: lower latency = higher value
                fullMark: 100,
            },
            {
                subject: 'Bandwidth',
                value: Math.min(100, node.resourceUtilization), // Utilization as bandwidth indicator
                fullMark: 100,
            },
            {
                subject: 'Reliability',
                value: Math.max(0, 100 - node.packetLossRate * 1000), // Lower loss = higher reliability
                fullMark: 100,
            },
            {
                subject: 'Queue',
                value: Math.max(0, 100 - queueRatio), // Lower queue = higher value
                fullMark: 100,
            },
            {
                subject: 'Battery',
                value: node.batteryChargePercent,
                fullMark: 100,
            },
        ];
    }, [node]);

    if (!node) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={onClose}>
            <div
                className="bg-white rounded-lg shadow-2xl w-full max-w-7xl max-h-[90vh] overflow-y-auto m-4"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="sticky top-0 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white p-6 rounded-t-lg shadow-lg">
                    <div className="flex justify-between items-center">
                        <div>
                            <h2 className="text-2xl font-bold flex items-center gap-2">
                                <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                                </svg>
                                {node.nodeName}
                            </h2>
                            <p className="text-violet-100 text-sm mt-1 font-medium">{node.nodeType} • Detailed Analysis</p>
                        </div>
                        <button
                            onClick={onClose}
                            className="text-white hover:text-gray-200 text-2xl font-bold w-10 h-10 flex items-center justify-center rounded-full hover:bg-white hover:bg-opacity-20 transition-all"
                        >
                            ×
                        </button>
                    </div>
                </div>

                {/* Node Resource Summary - Horizontal Layout */}
                <div className="bg-gradient-to-r from-violet-50 to-fuchsia-50 border-b-2 border-violet-200 p-6">
                    <h3 className="text-lg font-bold text-violet-900 mb-4 flex items-center gap-2">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        Node Resources & Performance
                    </h3>
                    <div className="grid grid-cols-5 gap-3">
                        {/* Processing Latency */}
                        <div className="bg-white rounded-xl p-4 border border-violet-200 shadow-sm">
                            <div className="text-xs font-semibold text-violet-700 mb-1 uppercase tracking-wide">Latency</div>
                            <div className="text-2xl font-bold text-gray-900">{node.nodeProcessingDelayMs.toFixed(0)}</div>
                            <div className="text-xs text-gray-500 mt-0.5">ms</div>
                            <div className={`mt-2 px-2 py-1 rounded text-xs font-semibold text-center ${
                                node.nodeProcessingDelayMs > 500 ? 'bg-red-100 text-red-700' :
                                node.nodeProcessingDelayMs > 200 ? 'bg-yellow-100 text-yellow-700' :
                                'bg-green-100 text-green-700'
                            }`}>
                                {node.nodeProcessingDelayMs > 500 ? 'Critical' : node.nodeProcessingDelayMs > 200 ? 'Warning' : 'Normal'}
                            </div>
                        </div>

                        {/* Packet Loss */}
                        <div className="bg-white rounded-xl p-4 border border-violet-200 shadow-sm">
                            <div className="text-xs font-semibold text-violet-700 mb-1 uppercase tracking-wide">Packet Loss</div>
                            <div className="text-2xl font-bold text-gray-900">{(node.packetLossRate * 100).toFixed(1)}</div>
                            <div className="text-xs text-gray-500 mt-0.5">%</div>
                            <div className={`mt-2 px-2 py-1 rounded text-xs font-semibold text-center ${
                                node.packetLossRate > 0.1 ? 'bg-red-100 text-red-700' :
                                node.packetLossRate > 0.05 ? 'bg-yellow-100 text-yellow-700' :
                                'bg-green-100 text-green-700'
                            }`}>
                                {node.packetLossRate > 0.1 ? 'Critical' : node.packetLossRate > 0.05 ? 'Warning' : 'Normal'}
                            </div>
                        </div>

                        {/* Queue Buffer */}
                        <div className="bg-white rounded-xl p-4 border border-violet-200 shadow-sm">
                            <div className="text-xs font-semibold text-violet-700 mb-1 uppercase tracking-wide">Queue Buffer</div>
                            <div className="text-2xl font-bold text-gray-900">{node.currentPacketCount}</div>
                            <div className="text-xs text-gray-500 mt-0.5">of {node.packetBufferCapacity}</div>
                            <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                                <div
                                    className={`h-2 rounded-full ${
                                        (node.currentPacketCount / node.packetBufferCapacity) > 0.9 ? 'bg-red-500' :
                                        (node.currentPacketCount / node.packetBufferCapacity) > 0.7 ? 'bg-yellow-500' : 'bg-green-500'
                                    }`}
                                    style={{ width: `${Math.min(100, (node.currentPacketCount / node.packetBufferCapacity) * 100)}%` }}
                                ></div>
                            </div>
                        </div>

                        {/* CPU/Memory */}
                        <div className="bg-white rounded-xl p-4 border border-violet-200 shadow-sm">
                            <div className="text-xs font-semibold text-violet-700 mb-1 uppercase tracking-wide">CPU/Memory</div>
                            <div className="text-2xl font-bold text-gray-900">{node.resourceUtilization.toFixed(0)}</div>
                            <div className="text-xs text-gray-500 mt-0.5">%</div>
                            <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                                <div
                                    className={`h-2 rounded-full ${
                                        node.resourceUtilization > 90 ? 'bg-red-500' :
                                        node.resourceUtilization > 70 ? 'bg-yellow-500' : 'bg-green-500'
                                    }`}
                                    style={{ width: `${node.resourceUtilization}%` }}
                                ></div>
                            </div>
                        </div>

                        {/* Battery */}
                        <div className="bg-white rounded-xl p-4 border border-violet-200 shadow-sm">
                            <div className="text-xs font-semibold text-violet-700 mb-1 uppercase tracking-wide">Battery</div>
                            <div className="text-2xl font-bold text-gray-900">{node.batteryChargePercent.toFixed(0)}</div>
                            <div className="text-xs text-gray-500 mt-0.5">%</div>
                            <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                                <div
                                    className={`h-2 rounded-full ${
                                        node.batteryChargePercent > 50 ? 'bg-green-500' :
                                        node.batteryChargePercent > 20 ? 'bg-yellow-500' : 'bg-red-500'
                                    }`}
                                    style={{ width: `${node.batteryChargePercent}%` }}
                                ></div>
                            </div>
                        </div>
                    </div>

                    {/* Additional Resource Details - Horizontal */}
                    <div className="grid grid-cols-4 gap-3 mt-4">
                        <div className="bg-white rounded-lg p-3 border border-violet-200">
                            <div className="text-xs text-violet-700 font-semibold mb-1">Bandwidth</div>
                            <div className="text-lg font-bold text-gray-900">{(node as any).bandwidthMbps?.toFixed(0) || 'N/A'} <span className="text-sm text-gray-500">Mbps</span></div>
                        </div>
                        <div className="bg-white rounded-lg p-3 border border-violet-200">
                            <div className="text-xs text-violet-700 font-semibold mb-1">Signal Strength</div>
                            <div className="text-lg font-bold text-gray-900">{(node as any).signalStrengthDbm?.toFixed(0) || 'N/A'} <span className="text-sm text-gray-500">dBm</span></div>
                        </div>
                        <div className="bg-white rounded-lg p-3 border border-violet-200">
                            <div className="text-xs text-violet-700 font-semibold mb-1">Status</div>
                            <div className={`text-sm font-bold px-2 py-1 rounded inline-block ${
                                node.isOperational ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                            }`}>
                                {node.isOperational ? 'Online' : 'Offline'}
                            </div>
                        </div>
                        <div className="bg-white rounded-lg p-3 border border-violet-200">
                            <div className="text-xs text-violet-700 font-semibold mb-1">Position</div>
                            <div className="text-xs text-gray-600">
                                Lat: {node.position.latitude.toFixed(1)}°<br/>
                                Lon: {node.position.longitude.toFixed(1)}°<br/>
                                Alt: {(node.position.altitude / 1000).toFixed(0)} km
                            </div>
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="p-6 space-y-6">
                    {loading && (
                        <div className="flex items-center justify-center py-12">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                            <span className="ml-4 text-gray-600">Loading analysis...</span>
                        </div>
                    )}

                    {error && (
                        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                            Error: {error.message}
                        </div>
                    )}

                    {!loading && !error && analysis && (
                        <>
                            {/* Radar Chart - Pentagon */}
                            <div className="bg-gradient-to-br from-slate-50 to-violet-50/30 rounded-xl p-6 border border-violet-200 shadow-sm">
                                <h3 className="text-lg font-semibold text-violet-900 mb-4 flex items-center gap-2">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                    </svg>
                                    Performance Score Breakdown
                                </h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RadarChart data={radarData}>
                                            <PolarGrid stroke="#c4b5fd" />
                                            <PolarAngleAxis dataKey="subject" tick={{ fontSize: 12, fill: '#6b21a8' }} />
                                            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10, fill: '#9333ea' }} />
                                            <Radar
                                                name="Performance"
                                                dataKey="value"
                                                stroke="#9333ea"
                                                fill="#9333ea"
                                                fillOpacity={0.5}
                                            />
                                            <Tooltip />
                                            <Legend />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Best Links - Top nodes gần nhất với link metric tốt nhất */}
                            {analysis.bestLinks.length > 0 && (
                                <div className="bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-300 rounded-xl p-5 shadow-md">
                                    <h3 className="text-lg font-bold text-green-800 mb-4 flex items-center gap-2">
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                                        </svg>
                                        Best Neighbor Links
                                        <span className="text-sm font-normal text-green-600 bg-green-100 px-2 py-0.5 rounded-full">
                                            {analysis.bestLinks.length} optimal connections
                                        </span>
                                    </h3>
                                    <div className="space-y-2 max-h-64 overflow-y-auto">
                                        {analysis.bestLinks.map((link, index) => (
                                            <div
                                                key={link.nodeId}
                                                className="bg-white rounded-lg p-4 border-2 border-green-200 hover:shadow-lg hover:border-green-300 transition-all"
                                            >
                                                <div className="flex justify-between items-start">
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-2 mb-2">
                                                            <span className="text-xs font-bold text-white bg-gradient-to-r from-green-600 to-emerald-600 px-2.5 py-1 rounded-full">
                                                                #{index + 1}
                                                            </span>
                                                            <span className="font-bold text-gray-900">{link.nodeName}</span>
                                                        </div>
                                                        <div className="flex items-center gap-3">
                                                            <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                                                                link.quality === 'excellent' ? 'bg-green-100 text-green-700 border border-green-300' :
                                                                link.quality === 'good' ? 'bg-blue-100 text-blue-700 border border-blue-300' :
                                                                link.quality === 'fair' ? 'bg-yellow-100 text-yellow-700 border border-yellow-300' : 'bg-red-100 text-red-700 border border-red-300'
                                                            }`}>
                                                                {link.quality.toUpperCase()}
                                                            </div>
                                                            <div className="text-sm">
                                                                <span className="text-gray-500">Score:</span>
                                                                <span className="font-bold text-green-600 ml-1">{link.score.toFixed(1)}/100</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="text-right space-y-1.5 ml-4 bg-gray-50 rounded-lg p-3 border border-gray-200">
                                                        <div className="font-bold text-gray-700 flex items-center justify-end gap-1">
                                                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                                                            </svg>
                                                            {link.distance.toFixed(0)} km
                                                        </div>
                                                        <div className="text-xs text-gray-600">Latency: <span className="font-semibold">{link.latency.toFixed(0)}ms</span></div>
                                                        <div className="text-xs text-gray-600">Bandwidth: <span className="font-semibold">{link.bandwidth.toFixed(0)}Mbps</span></div>
                                                        <div className="text-xs text-gray-600">Loss: <span className="font-semibold">{(link.packetLoss * 100).toFixed(2)}%</span></div>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Upcoming Satellites - Dự báo nodes chuẩn bị đến */}
                            {analysis.upcomingSatellites.length > 0 && (
                                <div className="bg-gradient-to-br from-purple-50 to-fuchsia-50 border-2 border-purple-300 rounded-xl p-5 shadow-md">
                                    <h3 className="text-lg font-bold text-purple-800 mb-4 flex items-center gap-2">
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                                        </svg>
                                        Incoming Satellites
                                        <span className="text-sm font-normal text-purple-600 bg-purple-100 px-2 py-0.5 rounded-full">
                                            {analysis.upcomingSatellites.length} approaching
                                        </span>
                                    </h3>
                                    <div className="space-y-2 max-h-64 overflow-y-auto">
                                        {analysis.upcomingSatellites.map((sat, index) => {
                                            const minutes = Math.floor(sat.estimatedArrivalIn / 60);
                                            const seconds = sat.estimatedArrivalIn % 60;
                                            const hours = Math.floor(minutes / 60);
                                            const remainingMinutes = minutes % 60;
                                            
                                            let timeDisplay = '';
                                            if (hours > 0) {
                                                timeDisplay = `${hours}h ${remainingMinutes}m`;
                                            } else if (minutes > 0) {
                                                timeDisplay = `${minutes}m ${seconds}s`;
                                            } else {
                                                timeDisplay = `${seconds}s`;
                                            }
                                            
                                            return (
                                                <div
                                                    key={sat.nodeId}
                                                    className="bg-white rounded p-3 border border-purple-200 hover:shadow-md transition-shadow"
                                                >
                                                    <div className="flex justify-between items-start">
                                                        <div className="flex-1">
                                                            <div className="flex items-center gap-2 mb-2">
                                                                <span className="text-xs font-bold text-white bg-gradient-to-r from-purple-600 to-fuchsia-600 px-2.5 py-1 rounded-full">
                                                                    #{index + 1}
                                                                </span>
                                                                <span className="font-bold text-gray-900">{sat.nodeName}</span>
                                                            </div>
                                                            <div className="text-sm text-gray-600 space-y-1">
                                                                <div className="font-medium">{sat.nodeType}</div>
                                                                <div className="flex items-center gap-1">
                                                                    {sat.willBeInRange ? (
                                                                        <span className="text-green-600 text-xs font-semibold bg-green-50 px-2 py-0.5 rounded-md">✅ Will be in range</span>
                                                                    ) : (
                                                                        <span className="text-yellow-600 text-xs font-semibold bg-yellow-50 px-2 py-0.5 rounded-md">⚠️ May be out of range</span>
                                                                    )}
                                                                </div>
                                                                {(sat as any).currentDistance && (
                                                                    <div className="text-xs text-gray-500 font-medium">
                                                                        Current distance: {(sat as any).currentDistance} km
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <div className="text-right space-y-2 ml-4 bg-purple-50 rounded-lg p-3 border border-purple-200">
                                                            <div className="font-bold text-purple-700 text-lg flex items-center justify-end gap-1">
                                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                                </svg>
                                                                ETA: {timeDisplay}
                                                            </div>
                                                            <div className="text-xs space-y-1 pt-2 border-t border-purple-200">
                                                                <div className="text-gray-600">Est. Latency: <span className="font-semibold">{sat.estimatedLatency}ms</span></div>
                                                                <div className="text-gray-600">Est. Bandwidth: <span className="font-semibold">{sat.estimatedBandwidth}Mbps</span></div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Degrading Nodes */}
                            {analysis.degradingNodes.length > 0 && (
                                <div className="bg-gradient-to-br from-red-50 to-orange-50 border-2 border-red-300 rounded-xl p-5 shadow-md">
                                    <h3 className="text-lg font-bold text-red-800 mb-4 flex items-center gap-2">
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                        </svg>
                                        Performance Degradation Alerts
                                        <span className="text-sm font-normal text-red-600 bg-red-100 px-2 py-0.5 rounded-full">
                                            {analysis.degradingNodes.length} nodes at risk
                                        </span>
                                    </h3>
                                    <div className="space-y-2 max-h-48 overflow-y-auto">
                                        {analysis.degradingNodes.map((degrading) => (
                                            <div
                                                key={degrading.nodeId}
                                                className={`bg-white rounded-lg p-4 border-2 hover:shadow-lg transition-all ${
                                                    degrading.severity === 'critical'
                                                        ? 'border-red-400 bg-red-50/50'
                                                        : degrading.severity === 'warning'
                                                        ? 'border-yellow-400 bg-yellow-50/50'
                                                        : 'border-orange-300 bg-orange-50/50'
                                                }`}
                                            >
                                                <div className="flex justify-between items-start">
                                                    <div className="flex-1">
                                                        <div className="font-bold text-gray-900 mb-2">{degrading.nodeName}</div>
                                                        <div className={`inline-block px-3 py-1 rounded-full text-xs font-bold mb-2 ${
                                                            degrading.severity === 'critical' ? 'bg-red-100 text-red-700 border border-red-300' :
                                                            degrading.severity === 'warning' ? 'bg-yellow-100 text-yellow-700 border border-yellow-300' :
                                                            'bg-orange-100 text-orange-700 border border-orange-300'
                                                        }`}>
                                                            {degrading.severity.toUpperCase()} SEVERITY
                                                        </div>
                                                        <div className="text-xs text-gray-700 font-medium">
                                                            Causes: <span className="text-gray-600">{degrading.degradationReason.join(' • ')}</span>
                                                        </div>
                                                    </div>
                                                    <div className="text-right space-y-1.5 ml-4 bg-white rounded-lg p-3 border border-gray-200">
                                                        <div className="font-bold text-red-600 text-sm">
                                                            ⏰ In {Math.floor(degrading.predictedDegradationIn / 60)} min
                                                        </div>
                                                        <div className="text-xs text-gray-600 pt-2 border-t border-gray-200 space-y-1">
                                                            <div>Latency: <span className="font-semibold">{degrading.currentMetrics.latency.toFixed(0)}ms</span></div>
                                                            <div>Load: <span className="font-semibold">{degrading.currentMetrics.utilization.toFixed(0)}%</span></div>
                                                            <div>Battery: <span className="font-semibold">{degrading.currentMetrics.battery.toFixed(0)}%</span></div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* No predictions message */}
                            {analysis.bestLinks.length === 0 &&
                                analysis.upcomingSatellites.length === 0 &&
                                analysis.degradingNodes.length === 0 && (
                                    <div className="bg-gradient-to-br from-gray-50 to-slate-50 border-2 border-gray-300 rounded-xl p-8 text-center">
                                        <svg className="w-16 h-16 text-gray-400 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                        </svg>
                                        <p className="text-gray-600 font-medium">No analysis data available for this node</p>
                                        <p className="text-gray-500 text-sm mt-1">All metrics are within normal ranges</p>
                                    </div>
                                )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default NodeAnalysisModal;

