import React, { useMemo, useState } from "react";
import {
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ComposedChart,
    Bar,
} from "recharts";

// Types
interface Position {
    latitude: number;
    longitude: number;
    altitude: number;
}

interface BufferState {
    queueSize: number;
    bandwidthUtilization: number;
}

interface RoutingDecisionInfo {
    algorithm: "Dijkstra" | "ReinforcementLearning";
    metric?: string;
    reward?: number;
}

interface HopRecord {
    fromNodeId: string;
    toNodeId: string;
    latencyMs: number;
    timestampMs: number;
    distanceKm: number;
    fromNodePosition: Position | null;
    toNodePosition: Position | null;
    fromNodeBufferState: BufferState;
    routingDecisionInfo: RoutingDecisionInfo;
}

interface QoS {
    serviceType: string;
    defaultPriority: number;
    maxLatencyMs: number;
    maxJitterMs: number;
    minBandwidthMbps: number;
    maxLossRate: number;
}

interface AnalysisData {
    avgLatency: number;
    avgDistanceKm: number;
    routeSuccessRate: number;
    totalLatencyMs: number;
}

interface Packet {
    packetId: string;
    sourceUserId: string;
    destinationUserId: string;
    stationSource: string;
    stationDest: string;
    type: string;
    acknowledgedPacketId?: string | null;
    timeSentFromSourceMs: number;
    payloadDataBase64: string;
    payloadSizeByte: number;
    serviceQoS: QoS;
    currentHoldingNodeId: string;
    nextHopNodeId: string;
    pathHistory: string[];
    hopRecords: HopRecord[];
    accumulatedDelayMs: number;
    priorityLevel: number;
    maxAcceptableLatencyMs: number;
    maxAcceptableLossRate: number;
    dropped: boolean;
    dropReason?: string | null;
    analysisData: AnalysisData;
    isUseRL: boolean;  // ✅ Fixed: Match backend field name
    TTL: number;       // ✅ Fixed: Uppercase to match backend
}

interface ComparisonData {
    dijkstraPacket: Packet | null;
    rlPacket: Packet | null;  // ✅ Fixed: Match server response (capital P)
}

interface Props {
    data: ComparisonData;
    scenario?: string; // Optional scenario info
}

const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
            <p className="font-semibold text-gray-800 mb-2">Hop {label}</p>
            {payload.map((entry: any, index: number) => (
                <p key={index} style={{ color: entry.color }} className="text-sm">
                    {entry.name}: <span className="font-medium">{typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}</span>
                </p>
            ))}
        </div>
    );
};

export const CombinedHopMetricsChart: React.FC<Props> = ({ data, scenario }) => {
    const [visibleMetrics, setVisibleMetrics] = useState({
        latency: true,
        distance: true,
        bandwidth: true,
    });

    // ✅ NULL-SAFE: Xử lý dữ liệu cho biểu đồ
    const mergedData = useMemo(() => {
        const dijkstraHops = data?.dijkstraPacket?.hopRecords || [];
        const rlHops = data?.rlPacket?.hopRecords || [];  // ✅ Fixed: rlPacket (capital P)

        if (dijkstraHops.length === 0 && rlHops.length === 0) {
            return [];
        }

        const maxHops = Math.max(dijkstraHops.length, rlHops.length);

        return Array.from({ length: maxHops }, (_, i) => {
            const dijkstraHop = dijkstraHops[i];
            const rlHop = rlHops[i];
            
            // Kiểm tra gói có bị drop không
            const isDijkstraComplete = !data?.dijkstraPacket?.dropped || i < dijkstraHops.length;
            const isRlComplete = !data?.rlPacket?.dropped || i < rlHops.length;  // ✅ Fixed: rlPacket (capital P)

            return {
                hop: i + 1,
                
                // Latency
                dijkstraLatency: isDijkstraComplete ? (dijkstraHop?.latencyMs ?? null) : null,
                rlLatency: isRlComplete ? (rlHop?.latencyMs ?? null) : null,
                
                // Distance
                dijkstraDistance: isDijkstraComplete ? (dijkstraHop?.distanceKm ?? null) : null,
                rlDistance: isRlComplete ? (rlHop?.distanceKm ?? null) : null,
                
                // Bandwidth
                dijkstraBandwidth: isDijkstraComplete && dijkstraHop?.fromNodeBufferState?.bandwidthUtilization !== undefined
                    ? dijkstraHop.fromNodeBufferState.bandwidthUtilization * 100
                    : null,
                rlBandwidth: isRlComplete && rlHop?.fromNodeBufferState?.bandwidthUtilization !== undefined
                    ? rlHop.fromNodeBufferState.bandwidthUtilization * 100
                    : null,
            };
        });
    }, [data]);

    // ✅ NULL-SAFE: Tính toán thống kê
    const stats = useMemo(() => {
        const dijkstra = data?.dijkstraPacket;
        const rl = data?.rlPacket;  // ✅ Fixed: rlPacket (capital P)

        // Trường hợp không có data
        if (!dijkstra && !rl) {
            return {
                dijkstraTotalLatency: 0,
                rlTotalLatency: 0,
                dijkstraAvgLatency: 0,
                rlAvgLatency: 0,
                latencyImprovement: 'N/A',
                dijkstraDropped: false,
                rlDropped: false,
            };
        }

        // Trường hợp chỉ có 1 gói
        if (!dijkstra) {
            return {
                dijkstraTotalLatency: 0,
                rlTotalLatency: rl?.accumulatedDelayMs || 0,
                dijkstraAvgLatency: 0,
                rlAvgLatency: (rl?.hopRecords?.length || 0) > 0 
                    ? (rl?.accumulatedDelayMs || 0) / (rl?.hopRecords?.length || 1)  // ✅ Fixed: null-safe division
                    : 0,
                latencyImprovement: 'N/A',
                dijkstraDropped: false,
                rlDropped: rl?.dropped || false,
            };
        }

        if (!rl) {
            return {
                dijkstraTotalLatency: dijkstra?.accumulatedDelayMs || 0,
                rlTotalLatency: 0,
                dijkstraAvgLatency: (dijkstra?.hopRecords?.length || 0) > 0 
                    ? (dijkstra?.accumulatedDelayMs || 0) / dijkstra.hopRecords.length 
                    : 0,
                rlAvgLatency: 0,
                latencyImprovement: 'N/A',
                dijkstraDropped: dijkstra?.dropped || false,
                rlDropped: false,
            };
        }

        // Trường hợp có cả 2 gói
        const dijkstraTotalLatency = dijkstra.accumulatedDelayMs || 0;
        const rlTotalLatency = rl.accumulatedDelayMs || 0;

        const dijkstraAvgLatency = (dijkstra.hopRecords?.length || 0) > 0
            ? dijkstraTotalLatency / dijkstra.hopRecords.length
            : 0;
        const rlAvgLatency = (rl.hopRecords?.length || 0) > 0
            ? rlTotalLatency / rl.hopRecords.length
            : 0;

        // Tính improvement
        let latencyImprovement = 'N/A';
        const dijkstraDropped = dijkstra.dropped || false;
        const rlDropped = rl.dropped || false;

        if (!dijkstraDropped && !rlDropped && dijkstraTotalLatency > 0) {
            latencyImprovement = ((dijkstraTotalLatency - rlTotalLatency) / dijkstraTotalLatency * 100).toFixed(1);
        } else if (dijkstraDropped && !rlDropped) {
            latencyImprovement = '100.0';
        } else if (!dijkstraDropped && rlDropped) {
            latencyImprovement = 'DROP';
        }

        return {
            dijkstraTotalLatency,
            rlTotalLatency,
            dijkstraAvgLatency,
            rlAvgLatency,
            latencyImprovement,
            dijkstraDropped,
            rlDropped,
        };
    }, [data]);

    const toggleMetric = (metric: keyof typeof visibleMetrics) => {
        setVisibleMetrics(prev => ({ ...prev, [metric]: !prev[metric] }));
    };

    // ✅ NULL-SAFE: Early return khi không có data
    if (!data || (!data.dijkstraPacket && !data.rlPacket)) {  // ✅ Fixed: rlPacket (capital P)
        return (
            <div className="p-6 bg-white rounded-2xl shadow-lg col-span-2 border border-gray-200">
                <p className="text-center text-gray-500">⚠️ No packet data available</p>
            </div>
        );
    }

    // Format hiển thị latency
    const formatLatencyValue = (totalLatency: number, isDropped: boolean) => {
        return isDropped 
            ? <span className="text-red-600 font-bold">❌ DROP</span> 
            : `${totalLatency.toFixed(2)} ms`;
    };

    return (
        <div className="p-8 bg-gradient-to-br from-white via-blue-50/30 to-gray-50 rounded-2xl shadow-2xl col-span-2 border-2 border-gray-200">
            {/* Header */}
            <div className="mb-6">
                <div className="flex justify-between items-center mb-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg">
                            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                        </div>
                        <div>
                            <h2 className="text-3xl font-bold text-gray-800">
                                Route Performance Analysis by Hop
                            </h2>
                            <p className="text-sm text-gray-500 mt-1">Detailed hop-by-hop performance metrics comparison</p>
                        </div>
                    </div>
                    {scenario && scenario !== 'NORMAL' && (
                        <span className="px-4 py-2 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-lg text-sm font-semibold shadow-md">
                            🌐 {scenario.replace(/_/g, ' ')}
                        </span>
                    )}
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                    {/* Dijkstra Stats */}
                    <div className={`p-4 rounded-xl border-2 shadow-md transition-all ${
                        !data.dijkstraPacket ? 'bg-gray-50 border-gray-300' :
                        stats.dijkstraDropped ? 'bg-red-50 border-red-400' : 'bg-gradient-to-br from-blue-50 to-blue-100 border-blue-300'
                    }`}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="w-3 h-3 rounded-full bg-blue-500"></span>
                            <p className="text-sm font-semibold text-gray-700">Dijkstra Total Latency</p>
                        </div>
                        <p className="text-2xl font-bold text-blue-700 mb-1">
                            {!data.dijkstraPacket ? (
                                <span className="text-gray-400">N/A</span>
                            ) : (
                                formatLatencyValue(stats.dijkstraTotalLatency, stats.dijkstraDropped)
                            )}
                        </p>
                        {data.dijkstraPacket && !stats.dijkstraDropped && (
                            <p className="text-xs text-gray-600">
                                Avg: {(stats.dijkstraAvgLatency).toFixed(2)} ms/hop
                            </p>
                        )}
                        {data.dijkstraPacket?.dropped && (
                            <p className="text-xs text-red-600 font-medium mt-1">
                                ⚠️ {data.dijkstraPacket.dropReason || 'Dropped'}
                            </p>
                        )}
                    </div>

                    {/* RL Stats */}
                    <div className={`p-4 rounded-xl border-2 shadow-md transition-all ${
                        !data.rlPacket ? 'bg-gray-50 border-gray-300' :
                        stats.rlDropped ? 'bg-red-50 border-red-400' : 'bg-gradient-to-br from-orange-50 to-orange-100 border-orange-300'
                    }`}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="w-3 h-3 rounded-full bg-orange-500"></span>
                            <p className="text-sm font-semibold text-gray-700">RL Total Latency</p>
                        </div>
                        <p className="text-2xl font-bold text-orange-700 mb-1">
                            {!data.rlPacket ? (
                                <span className="text-gray-400">N/A</span>
                            ) : (
                                formatLatencyValue(stats.rlTotalLatency, stats.rlDropped)
                            )}
                        </p>
                        {data.rlPacket && !stats.rlDropped && (
                            <p className="text-xs text-gray-600">
                                Avg: {(stats.rlAvgLatency).toFixed(2)} ms/hop
                            </p>
                        )}
                        {data.rlPacket?.dropped && (
                            <p className="text-xs text-red-600 font-medium mt-1">
                                ⚠️ {data.rlPacket.dropReason || 'Dropped'}
                            </p>
                        )}
                    </div>

                    {/* Improvement Stats */}
                    <div className={`p-4 rounded-xl border-2 shadow-md transition-all ${
                        stats.latencyImprovement === 'N/A' ? 'bg-gray-100 border-gray-300' :
                        stats.latencyImprovement === 'DROP' ? 'bg-red-100 border-red-400' :
                        parseFloat(stats.latencyImprovement) > 0 ? 'bg-gradient-to-br from-green-50 to-green-100 border-green-300' : 'bg-gray-100 border-gray-300'
                    }`}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-lg">📈</span>
                            <p className="text-sm font-semibold text-gray-700">RL Improvement</p>
                        </div>
                        <p className={`text-2xl font-bold mb-1 ${
                            stats.latencyImprovement === 'N/A' ? 'text-gray-600' :
                            stats.latencyImprovement === 'DROP' ? 'text-red-700' :
                            parseFloat(stats.latencyImprovement) > 0 ? 'text-green-700' : 'text-gray-600'
                        }`}>
                            {stats.latencyImprovement === 'DROP' ? '❌ RL Dropped' : 
                             stats.latencyImprovement === 'N/A' ? 'N/A' :
                             `+${stats.latencyImprovement}%`}
                        </p>
                        {stats.latencyImprovement !== 'N/A' && stats.latencyImprovement !== 'DROP' && parseFloat(stats.latencyImprovement) > 0 && (
                            <p className="text-xs text-green-700 font-medium">
                                RL is {stats.latencyImprovement}% faster
                            </p>
                        )}
                    </div>
                </div>

                {/* Metric Toggle Buttons */}
                <div className="flex gap-2 flex-wrap">
                    <button
                        onClick={() => toggleMetric('latency')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            visibleMetrics.latency
                                ? 'bg-blue-500 text-white shadow-md'
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                        }`}
                    >
                        📈 Latency
                    </button>
                    <button
                        onClick={() => toggleMetric('distance')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            visibleMetrics.distance
                                ? 'bg-green-500 text-white shadow-md'
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                        }`}
                    >
                        📍 Distance
                    </button>
                    <button
                        onClick={() => toggleMetric('bandwidth')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            visibleMetrics.bandwidth
                                ? 'bg-purple-500 text-white shadow-md'
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                        }`}
                    >
                        📊 Bandwidth
                    </button>
                </div>
            </div>

            {/* Chart */}
            <div className="bg-white rounded-xl p-4 border border-gray-200 shadow-inner">
                <ResponsiveContainer width="100%" height={500}>
                <ComposedChart data={mergedData} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                        dataKey="hop"
                        label={{ value: "Hop Number", position: "insideBottom", offset: -10 }}
                        stroke="#6b7280"
                    />

                    <YAxis
                        yAxisId="left"
                        label={{
                            value: "Latency (ms) / Bandwidth (%)",
                            angle: -90,
                            position: "insideLeft",
                            style: { fill: '#6b7280' }
                        }}
                        stroke="#6b7280"
                    />

                    <YAxis
                        yAxisId="right"
                        orientation="right"
                        label={{
                            value: "Distance (km)",
                            angle: 90,
                            position: "insideRight",
                            style: { fill: '#059669' }
                        }}
                        stroke="#059669"
                    />

                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="line" />

                    {/* Latency Lines */}
                    {visibleMetrics.latency && (
                        <>
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="dijkstraLatency"
                                stroke="#2563eb"
                                name="Dijkstra Latency (ms)"
                                strokeWidth={3}
                                dot={{ fill: '#2563eb', r: 4 }}
                                activeDot={{ r: 6 }}
                                connectNulls
                            />
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="rlLatency"
                                stroke="#f97316"
                                name="RL Latency (ms)"
                                strokeWidth={3}
                                dot={{ fill: '#f97316', r: 4 }}
                                activeDot={{ r: 6 }}
                                connectNulls
                            />
                        </>
                    )}

                    {/* Distance Lines */}
                    {visibleMetrics.distance && (
                        <>
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="dijkstraDistance"
                                stroke="#059669"
                                name="Dijkstra Distance (km)"
                                strokeWidth={2}
                                strokeDasharray="5 5"
                                dot={false}
                                connectNulls
                            />
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="rlDistance"
                                stroke="#0891b2"
                                name="RL Distance (km)"
                                strokeWidth={2}
                                strokeDasharray="5 5"
                                dot={false}
                                connectNulls
                            />
                        </>
                    )}

                    {/* Bandwidth Bars */}
                    {visibleMetrics.bandwidth && (
                        <>
                            <Bar
                                yAxisId="left"
                                dataKey="dijkstraBandwidth"
                                fill="#60a5fa"
                                name="Dijkstra Bandwidth (%)"
                                opacity={0.6}
                                radius={[4, 4, 0, 0]}
                            />
                            <Bar
                                yAxisId="left"
                                dataKey="rlBandwidth"
                                fill="#fb923c"
                                name="RL Bandwidth (%)"
                                opacity={0.6}
                                radius={[4, 4, 0, 0]}
                            />
                        </>
                    )}
                </ComposedChart>
            </ResponsiveContainer>
            </div>

            {/* Footer Info */}
            <div className="mt-4 text-xs text-gray-500 text-center">
                <p>
                    Total hops: Dijkstra ({data.dijkstraPacket?.hopRecords?.length || 0}) | RL ({data.rlPacket?.hopRecords?.length || 0})
                </p>
                {(stats.dijkstraDropped || stats.rlDropped) && (
                    <p className="text-red-500 font-semibold mt-1">
                        ⚠️ Lưu ý: Tuyến {stats.dijkstraDropped ? 'Dijkstra' : ''}{stats.dijkstraDropped && stats.rlDropped ? ' và ' : ''}{stats.rlDropped ? 'RL' : ''} đã bị DROP. Dữ liệu biểu đồ dừng lại ở hop cuối cùng.
                    </p>
                )}
            </div>
        </div>
    );
};