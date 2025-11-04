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
    useRL: boolean;
    ttl: number;
}

interface ComparisonData {
    dijkstraPacket: Packet | null;
    rlPacket: Packet | null;  // ✅ Fixed: Match server response (capital P)
}

interface Props {
    data: ComparisonData;
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

export const CombinedHopMetricsChart: React.FC<Props> = ({ data }) => {
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
        <div className="p-6 bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-lg col-span-2 border border-gray-200">
            {/* Header */}
            <div className="mb-4">
                <h2 className="text-2xl font-bold text-gray-800 mb-3">
                    📊 Route Performance Analysis by Hop
                </h2>

                {/* Quick Stats */}
                <div className="grid grid-cols-3 gap-3 mb-4">
                    {/* Dijkstra Stats */}
                    <div className={`p-3 rounded-lg border ${
                        !data.dijkstraPacket ? 'bg-gray-50 border-gray-300' :
                        stats.dijkstraDropped ? 'bg-red-50 border-red-300' : 'bg-blue-50 border-blue-200'
                    }`}>
                        <p className="text-xs text-gray-600 mb-1">Dijkstra Total Latency</p>
                        <p className="text-lg font-bold text-blue-600">
                            {!data.dijkstraPacket ? (
                                <span className="text-gray-400">N/A</span>
                            ) : (
                                formatLatencyValue(stats.dijkstraTotalLatency, stats.dijkstraDropped)
                            )}
                        </p>
                        {data.dijkstraPacket?.dropped && (
                            <p className="text-xs text-red-500 font-medium mt-1">
                                Reason: {data.dijkstraPacket.dropReason || 'Unknown'}
                            </p>
                        )}
                    </div>

                    {/* RL Stats */}
                    <div className={`p-3 rounded-lg border ${
                        !data.rlPacket ? 'bg-gray-50 border-gray-300' :
                        stats.rlDropped ? 'bg-red-50 border-red-300' : 'bg-orange-50 border-orange-200'
                    }`}>
                        <p className="text-xs text-gray-600 mb-1">RL Total Latency</p>
                        <p className="text-lg font-bold text-orange-600">
                            {!data.rlPacket ? (
                                <span className="text-gray-400">N/A</span>
                            ) : (
                                formatLatencyValue(stats.rlTotalLatency, stats.rlDropped)
                            )}
                        </p>
                        {data.rlPacket?.dropped && (
                            <p className="text-xs text-red-500 font-medium mt-1">
                                Reason: {data.rlPacket.dropReason || 'Unknown'}
                            </p>
                        )}
                    </div>

                    {/* Improvement Stats */}
                    <div className={`p-3 rounded-lg border ${
                        stats.latencyImprovement === 'N/A' ? 'bg-gray-100 border-gray-200' :
                        stats.latencyImprovement === 'DROP' ? 'bg-red-100 border-red-300' :
                        parseFloat(stats.latencyImprovement) > 0 ? 'bg-green-50 border-green-200' : 'bg-gray-100 border-gray-200'
                    }`}>
                        <p className="text-xs text-gray-600 mb-1">RL Improvement (%)</p>
                        <p className={`text-lg font-bold ${
                            stats.latencyImprovement === 'N/A' ? 'text-gray-600' :
                            stats.latencyImprovement === 'DROP' ? 'text-red-700' :
                            parseFloat(stats.latencyImprovement) > 0 ? 'text-green-600' : 'text-gray-600'
                        }`}>
                            {stats.latencyImprovement === 'DROP' ? 'RL Dropped' : 
                             stats.latencyImprovement === 'N/A' ? 'N/A' :
                             `${stats.latencyImprovement}%`}
                        </p>
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
            <ResponsiveContainer width="100%" height={450}>
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