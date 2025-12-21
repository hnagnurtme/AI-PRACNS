import React, { useMemo, useState } from 'react';
import type { NetworkBatch, ComparisonData, Packet } from '../../types/ComparisonTypes';

interface BatchComparisonLogProps {
    batch: NetworkBatch;
}

interface ComparisonMetrics {
    pairIndex: number;
    source: string;
    destination: string;
    dijkstra: {
        success: boolean;
        successRate: number;
        totalLatency: number;
        avgLatency: number;
        avgLatencyPerHop: number;
        totalDistance: number;
        avgDistance: number;
        hops: number;
        avgQueueSize: number;
        maxQueueSize: number;
        avgBandwidthUtil: number;
        maxBandwidthUtil: number;
        avgPacketLoss: number;
        maxPacketLoss: number;
        avgResourceUtil: number;
        maxResourceUtil: number;
        avgNodeLatency: number;
        maxNodeLatency: number;
        totalDelay: number;
        dropped: boolean;
        dropReason?: string;
    };
    rl: {
        success: boolean;
        successRate: number;
        totalLatency: number;
        avgLatency: number;
        avgLatencyPerHop: number;
        totalDistance: number;
        avgDistance: number;
        hops: number;
        avgQueueSize: number;
        maxQueueSize: number;
        avgBandwidthUtil: number;
        maxBandwidthUtil: number;
        avgPacketLoss: number;
        maxPacketLoss: number;
        avgResourceUtil: number;
        maxResourceUtil: number;
        avgNodeLatency: number;
        maxNodeLatency: number;
        totalDelay: number;
        dropped: boolean;
        dropReason?: string;
    };
    improvement: {
        latencyImprovement: number;
        distanceImprovement: number;
        hopsImprovement: number;
        queueImprovement: number;
        bandwidthImprovement: number;
        packetLossImprovement: number;
        resourceUtilImprovement: number;
        overallScore: number;
    };
}

export const BatchComparisonLog: React.FC<BatchComparisonLogProps> = ({ batch }) => {
    const [sortBy, setSortBy] = useState<'index' | 'latency' | 'distance' | 'success'>('index');
    const [filterDropped, setFilterDropped] = useState<'all' | 'success' | 'dropped'>('all');

    // Early return if batch is invalid
    if (!batch || !batch.packets || !Array.isArray(batch.packets) || batch.packets.length === 0) {
        return (
            <div className="bg-white rounded-xl shadow-xl border border-gray-200 p-6">
                <div className="text-center py-12 text-gray-500">
                    <p className="text-lg mb-2">No batch data available</p>
                    <p className="text-sm">Waiting for batch packets...</p>
                </div>
            </div>
        );
    }

    const calculateMetrics = (packet: Packet | null): ComparisonMetrics['dijkstra'] => {
        if (!packet) {
            return {
                success: false,
                successRate: 0,
                totalLatency: 0,
                avgLatency: 0,
                avgLatencyPerHop: 0,
                totalDistance: 0,
                avgDistance: 0,
                hops: 0,
                avgQueueSize: 0,
                maxQueueSize: 0,
                avgBandwidthUtil: 0,
                maxBandwidthUtil: 0,
                avgPacketLoss: 0,
                maxPacketLoss: 0,
                avgResourceUtil: 0,
                maxResourceUtil: 0,
                avgNodeLatency: 0,
                maxNodeLatency: 0,
                totalDelay: 0,
                dropped: true,
                dropReason: 'No packet data'
            };
        }

        const success = !packet.dropped;
        const hops = packet.hopRecords?.length || 0;
        const analysis = packet.analysisData || {};
        
        // Calculate queue metrics
        const queueSizes = (packet.hopRecords || []).map(h => h.fromNodeBufferState?.queueSize || 0);
        const avgQueueSize = queueSizes.length > 0 
            ? queueSizes.reduce((a, b) => a + b, 0) / queueSizes.length 
            : 0;
        const maxQueueSize = queueSizes.length > 0 ? Math.max(...queueSizes) : 0;

        // Calculate bandwidth utilization
        const bandwidthUtils = (packet.hopRecords || []).map(h => h.fromNodeBufferState?.bandwidthUtilization || 0);
        const avgBandwidthUtil = bandwidthUtils.length > 0
            ? bandwidthUtils.reduce((a, b) => a + b, 0) / bandwidthUtils.length
            : 0;
        const maxBandwidthUtil = bandwidthUtils.length > 0 ? Math.max(...bandwidthUtils) : 0;

        // Calculate packet loss (from hop records if available, otherwise 0)
        // TODO: Extract packet loss from hop records when available
        const avgPacketLoss = 0; // Placeholder
        const maxPacketLoss = 0; // Placeholder

        // Calculate resource utilization (from node load if available)
        const resourceUtils = (packet.hopRecords || []).map(h => (h.nodeLoadPercent || 0) / 100);
        const avgResourceUtil = resourceUtils.length > 0
            ? resourceUtils.reduce((a, b) => a + b, 0) / resourceUtils.length * 100
            : 0;
        const maxResourceUtil = resourceUtils.length > 0 ? Math.max(...resourceUtils) * 100 : 0;

        // Calculate node latency
        const nodeLatencies = (packet.hopRecords || []).map(h => h.latencyMs || 0);
        const avgNodeLatency = nodeLatencies.length > 0
            ? nodeLatencies.reduce((a, b) => a + b, 0) / nodeLatencies.length
            : 0;
        const maxNodeLatency = nodeLatencies.length > 0 ? Math.max(...nodeLatencies) : 0;

        return {
            success,
            successRate: success ? 100 : 0,
            totalLatency: analysis.totalLatencyMs || packet.accumulatedDelayMs || 0,
            avgLatency: analysis.avgLatency || 0,
            avgLatencyPerHop: hops > 0 ? (analysis.totalLatencyMs || 0) / hops : 0,
            totalDistance: analysis.totalDistanceKm || 0,
            avgDistance: analysis.avgDistanceKm || 0,
            hops,
            avgQueueSize,
            maxQueueSize,
            avgBandwidthUtil: avgBandwidthUtil * 100,
            maxBandwidthUtil: maxBandwidthUtil * 100,
            avgPacketLoss: avgPacketLoss * 100,
            maxPacketLoss: maxPacketLoss * 100,
            avgResourceUtil,
            maxResourceUtil,
            avgNodeLatency,
            maxNodeLatency,
            totalDelay: packet.accumulatedDelayMs || 0,
            dropped: packet.dropped || false,
            dropReason: packet.dropReason || undefined
        };
    };

    const comparisonData: ComparisonMetrics[] = useMemo(() => {
        return batch.packets.map((pair: ComparisonData, index: number) => {
            const dijkstra = calculateMetrics(pair.dijkstraPacket);
            const rl = calculateMetrics(pair.rlPacket);

            // Calculate improvements (RL vs Dijkstra)
            const latencyImprovement = dijkstra.totalLatency > 0
                ? ((dijkstra.totalLatency - rl.totalLatency) / dijkstra.totalLatency) * 100
                : 0;
            
            const distanceImprovement = dijkstra.totalDistance > 0
                ? ((dijkstra.totalDistance - rl.totalDistance) / dijkstra.totalDistance) * 100
                : 0;
            
            const hopsImprovement = dijkstra.hops > 0
                ? ((dijkstra.hops - rl.hops) / dijkstra.hops) * 100
                : 0;
            
            const queueImprovement = dijkstra.avgQueueSize > 0
                ? ((dijkstra.avgQueueSize - rl.avgQueueSize) / dijkstra.avgQueueSize) * 100
                : 0;
            
            const bandwidthImprovement = dijkstra.avgBandwidthUtil > 0
                ? ((dijkstra.avgBandwidthUtil - rl.avgBandwidthUtil) / dijkstra.avgBandwidthUtil) * 100
                : 0;
            
            const packetLossImprovement = dijkstra.avgPacketLoss > 0
                ? ((dijkstra.avgPacketLoss - rl.avgPacketLoss) / dijkstra.avgPacketLoss) * 100
                : 0;
            
            const resourceUtilImprovement = dijkstra.avgResourceUtil > 0
                ? ((dijkstra.avgResourceUtil - rl.avgResourceUtil) / dijkstra.avgResourceUtil) * 100
                : 0;

            // Overall score (weighted average)
            const overallScore = (
                latencyImprovement * 0.3 +
                distanceImprovement * 0.2 +
                hopsImprovement * 0.15 +
                queueImprovement * 0.1 +
                bandwidthImprovement * 0.1 +
                packetLossImprovement * 0.1 +
                resourceUtilImprovement * 0.05
            );

            return {
                pairIndex: index,
                source: pair.dijkstraPacket?.stationSource || pair.rlPacket?.stationSource || 'N/A',
                destination: pair.dijkstraPacket?.stationDest || pair.rlPacket?.stationDest || 'N/A',
                dijkstra,
                rl,
                improvement: {
                    latencyImprovement,
                    distanceImprovement,
                    hopsImprovement,
                    queueImprovement,
                    bandwidthImprovement,
                    packetLossImprovement,
                    resourceUtilImprovement,
                    overallScore
                }
            };
        });
    }, [batch]);

    const filteredAndSorted = useMemo(() => {
        let filtered = [...comparisonData];

        // Filter by dropped status
        if (filterDropped === 'success') {
            filtered = filtered.filter(m => m.dijkstra.success && m.rl.success);
        } else if (filterDropped === 'dropped') {
            filtered = filtered.filter(m => m.dijkstra.dropped || m.rl.dropped);
        }

        // Sort
        filtered.sort((a, b) => {
            switch (sortBy) {
                case 'latency':
                    return b.improvement.latencyImprovement - a.improvement.latencyImprovement;
                case 'distance':
                    return b.improvement.distanceImprovement - a.improvement.distanceImprovement;
                case 'success':
                    return (b.dijkstra.success && b.rl.success ? 1 : 0) - (a.dijkstra.success && a.rl.success ? 1 : 0);
                default:
                    return a.pairIndex - b.pairIndex;
            }
        });

        return filtered;
    }, [comparisonData, sortBy, filterDropped]);

    const formatPercentage = (value: number) => {
        const sign = value >= 0 ? '+' : '';
        return `${sign}${value.toFixed(2)}%`;
    };

    const getImprovementColor = (value: number) => {
        if (value > 10) return 'text-green-600 font-bold';
        if (value > 0) return 'text-green-500';
        if (value > -10) return 'text-yellow-500';
        return 'text-red-500';
    };

    return (
        <div className="bg-white rounded-xl shadow-xl border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-gradient-to-br from-violet-500 to-fuchsia-500 rounded-lg">
                        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                    </div>
                    <div>
                        <h3 className="text-2xl font-bold text-gray-800">Comparison Log</h3>
                        <p className="text-sm text-gray-500">Detailed metrics comparison: Dijkstra vs RL</p>
                    </div>
                </div>
                
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-gray-700">Filter:</label>
                        <select
                            value={filterDropped}
                            onChange={(e) => setFilterDropped(e.target.value as any)}
                            className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        >
                            <option value="all">All</option>
                            <option value="success">Success Only</option>
                            <option value="dropped">Dropped Only</option>
                        </select>
                    </div>
                    
                    <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-gray-700">Sort By:</label>
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value as any)}
                            className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        >
                            <option value="index">Index</option>
                            <option value="latency">Latency Improvement</option>
                            <option value="distance">Distance Improvement</option>
                            <option value="success">Success Rate</option>
                        </select>
                    </div>
                </div>
            </div>

            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gradient-to-r from-gray-50 to-gray-100 sticky top-0">
                        <tr>
                            <th rowSpan={2} className="px-3 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r border-gray-300">
                                Pair
                            </th>
                            <th rowSpan={2} className="px-3 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider border-r border-gray-300">
                                Scenario
                            </th>
                            <th colSpan={2} className="px-3 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider border-b border-gray-300">
                                Route
                            </th>
                            <th colSpan={2} className="px-3 py-3 text-center text-xs font-semibold text-blue-700 uppercase tracking-wider border-b border-gray-300 bg-blue-50">
                                Success Rate
                            </th>
                            <th colSpan={3} className="px-3 py-3 text-center text-xs font-semibold text-green-700 uppercase tracking-wider border-b border-gray-300 bg-green-50">
                                Latency (ms)
                            </th>
                            <th colSpan={2} className="px-3 py-3 text-center text-xs font-semibold text-purple-700 uppercase tracking-wider border-b border-gray-300 bg-purple-50">
                                Distance (km)
                            </th>
                            <th colSpan={2} className="px-3 py-3 text-center text-xs font-semibold text-orange-700 uppercase tracking-wider border-b border-gray-300 bg-orange-50">
                                Hops
                            </th>
                            <th colSpan={3} className="px-3 py-3 text-center text-xs font-semibold text-red-700 uppercase tracking-wider border-b border-gray-300 bg-red-50">
                                Queue
                            </th>
                            <th colSpan={3} className="px-3 py-3 text-center text-xs font-semibold text-indigo-700 uppercase tracking-wider border-b border-gray-300 bg-indigo-50">
                                Bandwidth Util (%)
                            </th>
                            <th colSpan={2} className="px-3 py-3 text-center text-xs font-semibold text-pink-700 uppercase tracking-wider border-b border-gray-300 bg-pink-50">
                                Resource Util (%)
                            </th>
                            <th colSpan={2} className="px-3 py-3 text-center text-xs font-semibold text-yellow-700 uppercase tracking-wider border-b border-gray-300 bg-yellow-50">
                                Node Latency (ms)
                            </th>
                            <th colSpan={8} className="px-3 py-3 text-center text-xs font-semibold text-teal-700 uppercase tracking-wider border-b border-gray-300 bg-teal-50">
                                Improvement vs Dijkstra (%)
                            </th>
                        </tr>
                        <tr>
                            <th className="px-2 py-2 text-xs font-medium text-gray-600">Source</th>
                            <th className="px-2 py-2 text-xs font-medium text-gray-600">Dest</th>
                            <th className="px-2 py-2 text-xs font-medium text-blue-600 bg-blue-50">Dijkstra</th>
                            <th className="px-2 py-2 text-xs font-medium text-orange-600 bg-orange-50">RL</th>
                            <th className="px-2 py-2 text-xs font-medium text-green-600 bg-green-50">Total</th>
                            <th className="px-2 py-2 text-xs font-medium text-green-600 bg-green-50">Avg</th>
                            <th className="px-2 py-2 text-xs font-medium text-green-600 bg-green-50">Per Hop</th>
                            <th className="px-2 py-2 text-xs font-medium text-purple-600 bg-purple-50">Total</th>
                            <th className="px-2 py-2 text-xs font-medium text-purple-600 bg-purple-50">Avg</th>
                            <th className="px-2 py-2 text-xs font-medium text-orange-600 bg-orange-50">Dijkstra</th>
                            <th className="px-2 py-2 text-xs font-medium text-orange-600 bg-orange-50">RL</th>
                            <th className="px-2 py-2 text-xs font-medium text-red-600 bg-red-50">Avg</th>
                            <th className="px-2 py-2 text-xs font-medium text-red-600 bg-red-50">Max</th>
                            <th className="px-2 py-2 text-xs font-medium text-red-600 bg-red-50">Improve</th>
                            <th className="px-2 py-2 text-xs font-medium text-indigo-600 bg-indigo-50">Avg</th>
                            <th className="px-2 py-2 text-xs font-medium text-indigo-600 bg-indigo-50">Max</th>
                            <th className="px-2 py-2 text-xs font-medium text-indigo-600 bg-indigo-50">Improve</th>
                            <th className="px-2 py-2 text-xs font-medium text-pink-600 bg-pink-50">Avg</th>
                            <th className="px-2 py-2 text-xs font-medium text-pink-600 bg-pink-50">Max</th>
                            <th className="px-2 py-2 text-xs font-medium text-yellow-600 bg-yellow-50">Avg</th>
                            <th className="px-2 py-2 text-xs font-medium text-yellow-600 bg-yellow-50">Max</th>
                            <th className="px-2 py-2 text-xs font-medium text-teal-600 bg-teal-50">Latency</th>
                            <th className="px-2 py-2 text-xs font-medium text-teal-600 bg-teal-50">Distance</th>
                            <th className="px-2 py-2 text-xs font-medium text-teal-600 bg-teal-50">Hops</th>
                            <th className="px-2 py-2 text-xs font-medium text-teal-600 bg-teal-50">Queue</th>
                            <th className="px-2 py-2 text-xs font-medium text-teal-600 bg-teal-50">Bandwidth</th>
                            <th className="px-2 py-2 text-xs font-medium text-teal-600 bg-teal-50">Loss</th>
                            <th className="px-2 py-2 text-xs font-medium text-teal-600 bg-teal-50">Resource</th>
                            <th className="px-2 py-2 text-xs font-medium text-teal-600 bg-teal-50">Overall</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {filteredAndSorted.map((metrics) => {
                            const hasDropped = metrics.dijkstra.dropped || metrics.rl.dropped;
                            
                            return (
                                <tr
                                    key={metrics.pairIndex}
                                    className={`hover:bg-gray-50 transition-colors ${
                                        hasDropped ? 'bg-red-50' : ''
                                    }`}
                                >
                                    <td className="px-3 py-3 whitespace-nowrap text-sm font-bold text-gray-900 border-r border-gray-200">
                                        #{metrics.pairIndex + 1}
                                    </td>
                                    <td className="px-3 py-3 whitespace-nowrap text-xs font-medium text-gray-700 border-r border-gray-200">
                                        {batch.scenario || 'NORMAL'}
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-gray-600">
                                        {metrics.source}
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-gray-600">
                                        {metrics.destination}
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center ${
                                        metrics.dijkstra.success ? 'text-green-600' : 'text-red-600'
                                    } bg-blue-50`}>
                                        {metrics.dijkstra.successRate.toFixed(1)}%
                                        {metrics.dijkstra.dropped && <span className="block text-red-600 text-xs">❌</span>}
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center ${
                                        metrics.rl.success ? 'text-green-600' : 'text-red-600'
                                    } bg-orange-50`}>
                                        {metrics.rl.successRate.toFixed(1)}%
                                        {metrics.rl.dropped && <span className="block text-red-600 text-xs">❌</span>}
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-green-50">
                                        <div className="text-gray-700">{metrics.dijkstra.totalLatency.toFixed(2)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div className="text-gray-700">{metrics.rl.totalLatency.toFixed(2)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-green-50">
                                        <div>{metrics.dijkstra.avgLatency.toFixed(2)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.avgLatency.toFixed(2)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-green-50">
                                        <div>{metrics.dijkstra.avgLatencyPerHop.toFixed(2)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.avgLatencyPerHop.toFixed(2)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-purple-50">
                                        <div>{metrics.dijkstra.totalDistance.toFixed(2)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.totalDistance.toFixed(2)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-purple-50">
                                        <div>{metrics.dijkstra.avgDistance.toFixed(2)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.avgDistance.toFixed(2)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-orange-50">
                                        {metrics.dijkstra.hops}
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-orange-50">
                                        {metrics.rl.hops}
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-red-50">
                                        <div>{metrics.dijkstra.avgQueueSize.toFixed(1)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.avgQueueSize.toFixed(1)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-red-50">
                                        <div>{metrics.dijkstra.maxQueueSize.toFixed(1)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.maxQueueSize.toFixed(1)}</div>
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-red-50 ${getImprovementColor(metrics.improvement.queueImprovement)}`}>
                                        {formatPercentage(metrics.improvement.queueImprovement)}
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-indigo-50">
                                        <div>{metrics.dijkstra.avgBandwidthUtil.toFixed(1)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.avgBandwidthUtil.toFixed(1)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-indigo-50">
                                        <div>{metrics.dijkstra.maxBandwidthUtil.toFixed(1)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.maxBandwidthUtil.toFixed(1)}</div>
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-indigo-50 ${getImprovementColor(metrics.improvement.bandwidthImprovement)}`}>
                                        {formatPercentage(metrics.improvement.bandwidthImprovement)}
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-pink-50">
                                        <div>{metrics.dijkstra.avgResourceUtil.toFixed(1)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.avgResourceUtil.toFixed(1)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-pink-50">
                                        <div>{metrics.dijkstra.maxResourceUtil.toFixed(1)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.maxResourceUtil.toFixed(1)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-yellow-50">
                                        <div>{metrics.dijkstra.avgNodeLatency.toFixed(2)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.avgNodeLatency.toFixed(2)}</div>
                                    </td>
                                    <td className="px-2 py-3 whitespace-nowrap text-xs text-center bg-yellow-50">
                                        <div>{metrics.dijkstra.maxNodeLatency.toFixed(2)}</div>
                                        <div className="text-gray-500 text-xs">vs</div>
                                        <div>{metrics.rl.maxNodeLatency.toFixed(2)}</div>
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-bold text-center bg-teal-50 ${getImprovementColor(metrics.improvement.latencyImprovement)}`}>
                                        {formatPercentage(metrics.improvement.latencyImprovement)}
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-teal-50 ${getImprovementColor(metrics.improvement.distanceImprovement)}`}>
                                        {formatPercentage(metrics.improvement.distanceImprovement)}
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-teal-50 ${getImprovementColor(metrics.improvement.hopsImprovement)}`}>
                                        {formatPercentage(metrics.improvement.hopsImprovement)}
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-teal-50 ${getImprovementColor(metrics.improvement.queueImprovement)}`}>
                                        {formatPercentage(metrics.improvement.queueImprovement)}
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-teal-50 ${getImprovementColor(metrics.improvement.bandwidthImprovement)}`}>
                                        {formatPercentage(metrics.improvement.bandwidthImprovement)}
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-teal-50 ${getImprovementColor(metrics.improvement.packetLossImprovement)}`}>
                                        {formatPercentage(metrics.improvement.packetLossImprovement)}
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-semibold text-center bg-teal-50 ${getImprovementColor(metrics.improvement.resourceUtilImprovement)}`}>
                                        {formatPercentage(metrics.improvement.resourceUtilImprovement)}
                                    </td>
                                    <td className={`px-2 py-3 whitespace-nowrap text-xs font-bold text-center bg-teal-50 ${getImprovementColor(metrics.improvement.overallScore)}`}>
                                        {formatPercentage(metrics.improvement.overallScore)}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {filteredAndSorted.length === 0 && (
                <div className="text-center py-12 text-gray-500">
                    No data matching the current filters
                </div>
            )}
        </div>
    );
};

