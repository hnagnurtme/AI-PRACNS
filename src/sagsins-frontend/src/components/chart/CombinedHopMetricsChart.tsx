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

// Types (Được giữ nguyên)
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
    fromNodePosition: Position;
    toNodePosition: Position;
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
    totalLatencyMs: number; // Đã thêm trường này để khớp với logic mới
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
    dijkstraPacket: Packet;
    rlpacket: Packet;
}

interface Props {
    data: ComparisonData;
}

// Custom Tooltip Component (Giữ nguyên)
const CustomTooltip = ( { active, payload, label }: any ) => {
    if ( !active || !payload || !payload.length ) return null;

    return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
            <p className="font-semibold text-gray-800 mb-2">Hop { label }</p>
            { payload.map( ( entry: any, index: number ) => (
                <p key={ index } style={ { color: entry.color } } className="text-sm">
                    { entry.name }: <span className="font-medium">{ typeof entry.value === 'number' ? entry.value.toFixed( 2 ) : entry.value }</span>
                </p>
            ) ) }
        </div>
    );
};

export const CombinedHopMetricsChart: React.FC<Props> = ( { data } ) => {
    const [ visibleMetrics, setVisibleMetrics ] = useState( {
        latency: true,
        distance: true,
        bandwidth: true,
    } );

    // ✅ Cập nhật Logic xử lý dữ liệu cho gói bị DROP
    const mergedData = useMemo( () => {
        if ( !data?.dijkstraPacket?.hopRecords || !data?.rlpacket?.hopRecords ) {
            return [];
        }

        const maxHops = Math.max(
            data.dijkstraPacket.hopRecords.length,
            data.rlpacket.hopRecords.length
        );

        // Lấy số hop thực tế (stop ở hop cuối cùng của gói bị drop)
        const dijkstraHops = data.dijkstraPacket.hopRecords;
        const rlHops = data.rlpacket.hopRecords;

        return Array.from( { length: maxHops }, ( _, i ) => {
            const dijkstraHop = dijkstraHops[ i ];
            const rlHop = rlHops[ i ];
            
            // Nếu gói bị drop, dữ liệu của nó sẽ là NULL sau hop cuối cùng
            const isDijkstraComplete = !data.dijkstraPacket.dropped || i < dijkstraHops.length;
            const isRlComplete = !data.rlpacket.dropped || i < rlHops.length;

            return {
                hop: i + 1,
                
                // Latency (chỉ hiển thị nếu gói chưa bị drop)
                dijkstraLatency: isDijkstraComplete ? ( dijkstraHop?.latencyMs ?? null ) : null,
                rlLatency: isRlComplete ? ( rlHop?.latencyMs ?? null ) : null,
                
                // Distance
                dijkstraDistance: isDijkstraComplete ? ( dijkstraHop?.distanceKm ?? null ) : null,
                rlDistance: isRlComplete ? ( rlHop?.distanceKm ?? null ) : null,
                
                // Bandwidth Utilization
                dijkstraBandwidth: isDijkstraComplete && dijkstraHop?.fromNodeBufferState?.bandwidthUtilization !== undefined
                    ? dijkstraHop.fromNodeBufferState.bandwidthUtilization * 100
                    : null,
                rlBandwidth: isRlComplete && rlHop?.fromNodeBufferState?.bandwidthUtilization !== undefined
                    ? rlHop.fromNodeBufferState.bandwidthUtilization * 100
                    : null,
            };
        } );
    }, [ data ] );

    // ✅ Cập nhật Logic tính toán Stats để hiển thị DROP/TOTAL LATENCY
    const stats = useMemo( () => {
        const dijkstra = data.dijkstraPacket;
        const rl = data.rlpacket;

        // Lấy độ trễ tổng (sử dụng accumulatedDelayMs vì nó chính xác hơn)
        const dijkstraTotalLatency = dijkstra.accumulatedDelayMs;
        const rlTotalLatency = rl.accumulatedDelayMs;

        // Tính độ trễ trung bình chỉ trên các hop thực tế
        const dijkstraAvgLatency = dijkstra.hopRecords.length > 0
            ? dijkstraTotalLatency / dijkstra.hopRecords.length
            : 0;
        const rlAvgLatency = rl.hopRecords.length > 0
            ? rlTotalLatency / rl.hopRecords.length
            : 0;

        // Tính Độ trễ so sánh (chỉ tính nếu cả hai không bị drop)
        let latencyImprovement = 'N/A';
        if (!dijkstra.dropped && !rl.dropped && dijkstraTotalLatency > 0) {
            latencyImprovement = ( ( dijkstraTotalLatency - rlTotalLatency ) / dijkstraTotalLatency * 100 ).toFixed( 1 );
        } else if (dijkstra.dropped && !rl.dropped) {
            latencyImprovement = '100.0'; // RL thành công, Dijkstra thất bại
        } else if (!dijkstra.dropped && rl.dropped) {
            latencyImprovement = 'DROP'; // RL thất bại
        } else if (dijkstra.dropped && rl.dropped) {
             latencyImprovement = 'N/A'; // Cả hai thất bại
        }

        return {
            dijkstraTotalLatency,
            rlTotalLatency,
            dijkstraAvgLatency,
            rlAvgLatency,
            latencyImprovement,
            dijkstraDropped: dijkstra.dropped,
            rlDropped: rl.dropped,
        };
    }, [ data ] );

    const toggleMetric = ( metric: keyof typeof visibleMetrics ) => {
        setVisibleMetrics( prev => ( { ...prev, [ metric ]: !prev[ metric ] } ) );
    };

    // Early return if no data
    if ( !data || !data.dijkstraPacket || !data.rlpacket ) {
        return (
            <div className="p-6 bg-white rounded-2xl shadow-lg col-span-2 border border-gray-200">
                <p className="text-center text-gray-500">No data available</p>
            </div>
        );
    }

    // Định dạng giá trị Total Latency cho hiển thị
    const formatLatencyValue = (totalLatency: number, isDropped: boolean) => {
        return isDropped 
            ? <span className="text-red-600 font-bold">❌ DROP</span> 
            : `${totalLatency.toFixed( 2 )} ms`;
    };

    return (
        <div className="p-6 bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-lg col-span-2 border border-gray-200">
            {/* Header with Stats */}
            <div className="mb-4">
                <h2 className="text-2xl font-bold text-gray-800 mb-3">
                    📊 Route Performance Analysis by Hop
                </h2>

                {/* Quick Stats (Sử dụng Total Latency và Drop Status) */}
                <div className="grid grid-cols-3 gap-3 mb-4">
                    <div className={`p-3 rounded-lg border ${stats.dijkstraDropped ? 'bg-red-50 border-red-300' : 'bg-blue-50 border-blue-200'}`}>
                        <p className="text-xs text-gray-600 mb-1">Dijkstra Total Latency</p>
                        <p className="text-lg font-bold text-blue-600">
                            {formatLatencyValue(stats.dijkstraTotalLatency, stats.dijkstraDropped)}
                        </p>
                        {stats.dijkstraDropped && <p className="text-xs text-red-500 font-medium mt-1">Reason: {data.dijkstraPacket.dropReason || 'Unknown'}</p>}
                    </div>
                    <div className={`p-3 rounded-lg border ${stats.rlDropped ? 'bg-red-50 border-red-300' : 'bg-orange-50 border-orange-200'}`}>
                        <p className="text-xs text-gray-600 mb-1">RL Total Latency</p>
                        <p className="text-lg font-bold text-orange-600">
                            {formatLatencyValue(stats.rlTotalLatency, stats.rlDropped)}
                        </p>
                         {stats.rlDropped && <p className="text-xs text-red-500 font-medium mt-1">Reason: {data.rlpacket.dropReason || 'Unknown'}</p>}
                    </div>
                    <div className={ `p-3 rounded-lg border ${ 
                        stats.latencyImprovement === 'DROP' ? 'bg-red-100 border-red-300' : 
                        parseFloat( stats.latencyImprovement ) > 0 ? 'bg-green-50 border-green-200' : 'bg-gray-100 border-gray-200'
                    }` }>
                        <p className="text-xs text-gray-600 mb-1">RL Improvement (%)</p>
                        <p className={ `text-lg font-bold ${ 
                            stats.latencyImprovement === 'DROP' ? 'text-red-700' :
                            parseFloat( stats.latencyImprovement ) > 0 ? 'text-green-600' : 'text-gray-600'
                        }` }>
                            { stats.latencyImprovement === 'DROP' ? 'RL Dropped' : `${stats.latencyImprovement}%` }
                        </p>
                    </div>
                </div>

                {/* Metric Toggle Buttons */ }
                <div className="flex gap-2 flex-wrap">
                    <button
                        onClick={ () => toggleMetric( 'latency' ) }
                        className={ `px-4 py-2 rounded-lg text-sm font-medium transition-all ${ visibleMetrics.latency
                                ? 'bg-blue-500 text-white shadow-md'
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                            }` }
                    >
                        📈 Latency
                    </button>
                    <button
                        onClick={ () => toggleMetric( 'distance' ) }
                        className={ `px-4 py-2 rounded-lg text-sm font-medium transition-all ${ visibleMetrics.distance
                                ? 'bg-green-500 text-white shadow-md'
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                            }` }
                    >
                        📍 Distance
                    </button>
                    <button
                        onClick={ () => toggleMetric( 'bandwidth' ) }
                        className={ `px-4 py-2 rounded-lg text-sm font-medium transition-all ${ visibleMetrics.bandwidth
                                ? 'bg-purple-500 text-white shadow-md'
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                            }` }
                    >
                        📊 Bandwidth
                    </button>
                </div>
            </div>

            {/* Chart */ }
            <ResponsiveContainer width="100%" height={ 450 }>
                <ComposedChart data={ mergedData } margin={ { top: 10, right: 30, left: 10, bottom: 30 } }>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                        dataKey="hop"
                        label={ { value: "Hop Number", position: "insideBottom", offset: -10 } }
                        stroke="#6b7280"
                    />

                    {/* Left Y-axis: Latency & Bandwidth */ }
                    <YAxis
                        yAxisId="left"
                        label={ {
                            value: "Latency (ms) / Bandwidth (%)",
                            angle: -90,
                            position: "insideLeft",
                            style: { fill: '#6b7280' }
                        } }
                        stroke="#6b7280"
                    />

                    {/* Right Y-axis: Distance */ }
                    <YAxis
                        yAxisId="right"
                        orientation="right"
                        label={ {
                            value: "Distance (km)",
                            angle: 90,
                            position: "insideRight",
                            style: { fill: '#059669' }
                        } }
                        stroke="#059669"
                    />

                    <Tooltip content={ <CustomTooltip /> } />
                    <Legend
                        wrapperStyle={ { paddingTop: '20px' } }
                        iconType="line"
                    />

                    {/* Latency Lines */}
                    { visibleMetrics.latency && (
                        <>
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="dijkstraLatency"
                                stroke="#2563eb"
                                name="Dijkstra Latency (ms)"
                                strokeWidth={ 3 }
                                dot={ { fill: '#2563eb', r: 4 } }
                                activeDot={ { r: 6 } }
                                connectNulls
                            />
                            <Line
                                yAxisId="left"
                                type="monotone"
                                dataKey="rlLatency"
                                stroke="#f97316"
                                name="RL Latency (ms)"
                                strokeWidth={ 3 }
                                dot={ { fill: '#f97316', r: 4 } }
                                activeDot={ { r: 6 } }
                                connectNulls
                            />
                        </>
                    ) }

                    {/* Distance Lines */ }
                    { visibleMetrics.distance && (
                        <>
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="dijkstraDistance"
                                stroke="#059669"
                                name="Dijkstra Distance (km)"
                                strokeWidth={ 2 }
                                strokeDasharray="5 5"
                                dot={ false }
                                connectNulls
                            />
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="rlDistance"
                                stroke="#0891b2"
                                name="RL Distance (km)"
                                strokeWidth={ 2 }
                                strokeDasharray="5 5"
                                dot={ false }
                                connectNulls
                            />
                        </>
                    ) }

                    {/* Bandwidth Bars */ }
                    { visibleMetrics.bandwidth && (
                        <>
                            <Bar
                                yAxisId="left"
                                dataKey="dijkstraBandwidth"
                                fill="#60a5fa"
                                name="Dijkstra Bandwidth (%)"
                                opacity={ 0.6 }
                                radius={ [ 4, 4, 0, 0 ] }
                            />
                            <Bar
                                yAxisId="left"
                                dataKey="rlBandwidth"
                                fill="#fb923c"
                                name="RL Bandwidth (%)"
                                opacity={ 0.6 }
                                radius={ [ 4, 4, 0, 0 ] }
                            />
                        </>
                    ) }
                </ComposedChart>
            </ResponsiveContainer>

            {/* Footer Info */}
            <div className="mt-4 text-xs text-gray-500 text-center">
                <p>Total hops: Dijkstra ({ data.dijkstraPacket.hopRecords?.length || 0 }) | RL ({ data.rlpacket.hopRecords?.length || 0 })</p>
                {/* Cảnh báo gói bị drop */}
                 {(stats.dijkstraDropped || stats.rlDropped) && (
                    <p className="text-red-500 font-semibold mt-1">
                        ⚠️ Lưu ý: Tuyến {stats.dijkstraDropped ? 'Dijkstra' : ''}{stats.dijkstraDropped && stats.rlDropped ? ' và ' : ''}{stats.rlDropped ? 'RL' : ''} đã bị DROP. Dữ liệu biểu đồ dừng lại ở hop cuối cùng.
                    </p>
                )}
            </div>
        </div>
    );  
};