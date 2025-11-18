import React, { useMemo, useState } from "react";

// --- TYPES (Copied from User's input) ---
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
    // Thêm các thông số tổng khác nếu có
    totalDistanceKm: number;
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
}

interface NormalizedNode {
    x: number;
    y: number;
    id: string;
    nodeId: string;
}

interface NodeData {
    nodeId: string;
    position: Position;
    bufferState: BufferState | null;
}

// --- UTILITY FUNCTIONS ---

const normalizePositions = (
    positions: Position[], 
    width: number, 
    height: number, 
    nodeIds: string[]
): NormalizedNode[] => {
    if (positions.length === 0) return [];
    
    // Filter out null/undefined positions
    const validPositions = positions.filter(p => p != null);
    if (validPositions.length === 0) return [];
    
    const lats = validPositions.map(p => p.latitude);
    const lons = validPositions.map(p => p.longitude);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);

    const rangeLat = maxLat - minLat;
    const rangeLon = maxLon - minLon;
    const padding = 60;

    if (rangeLat === 0 && rangeLon === 0) {
        return validPositions.map((p, i) => ({
            x: width / 2,
            y: height / 2,
            id: `${p.latitude}_${p.longitude}`,
            nodeId: nodeIds[i],
        }));
    }

    return validPositions.map((p, i) => ({
        // Đã thêm kiểm tra chia cho 0 để an toàn hơn
        x: padding + ((p.longitude - minLon) / (rangeLon || 1)) * (width - 2 * padding),
        y: height - padding - ((p.latitude - minLat) / (rangeLat || 1)) * (height - 2 * padding),
        id: `${p.latitude}_${p.longitude}`,
        nodeId: nodeIds[i],
    }));
};

const getLatencyColor = (latencyMs: number, maxLatency: number): string => {
    const effectiveMax = maxLatency || 200;
    const ratio = Math.min(latencyMs / effectiveMax, 1);
    
    if (ratio < 0.3) return "#10b981"; // Green
    if (ratio < 0.7) return "#facc15"; // Yellow
    return "#ef4444"; // Red
};

const getNodeSize = (utilization: number): number => {
    return 6 + utilization * 8; // Kích thước Node từ 6 đến 14
};

// --- MAIN COMPONENT ---

export const PacketRouteGraph: React.FC<Props> = ({ data }) => {
    const [hoveredNode, setHoveredNode] = useState<string | null>(null);
    const [showDijkstra, setShowDijkstra] = useState(true);
    const [showRL, setShowRL] = useState(true);
    
    const width = 900;
    const height = 550;
    
    const DIJKSTRA_COLOR = "#2563eb"; // Blue-600
    const RL_COLOR = "#f97316"; // Orange-500
    const DROP_COLOR = "#dc2626"; // Red-600

    // SỬ DỤNG useMemo để tính toán Node Data chỉ một lần
    const { uniqueNodeData, normalizedNodes } = useMemo(() => {
        const nodeDataMap = new Map<string, { position: Position, bufferState: BufferState | null }>();

        // Thêm currentHoldingNodeId (node cuối cùng trước khi drop) vào map nếu bị drop
        const allPackets = [data?.dijkstraPacket, data?.rlPacket];  // ✅ Fixed: rlPacket (capital P)

        allPackets.forEach(packet => {
            if (packet?.dropped && packet.currentHoldingNodeId) {
                 // Nếu gói bị drop, node cuối cùng (currentHoldingNodeId) không có bufferState cập nhật
                 // Nó sẽ được cập nhật ở bước hopRecords
            }
        });


        const allHopRecords = [
            ...(data?.dijkstraPacket?.hopRecords || []),
            ...(data?.rlPacket?.hopRecords || [])  // ✅ Fixed: rlPacket (capital P)
        ];

        allHopRecords.forEach(hop => {
            if (!hop) return;
            
            // Lấy dữ liệu Buffer State và Position từ FromNode (chỉ nếu position hợp lệ)
            if (hop.fromNodePosition) {
                nodeDataMap.set(hop.fromNodeId, {
                    position: hop.fromNodePosition,
                    bufferState: hop.fromNodeBufferState,
                });
            }

            // Lấy dữ liệu Position từ ToNode (chỉ nếu position hợp lệ)
            // *Lưu ý: bufferState của ToNode sẽ được giả định hoặc lấy từ FromNode tiếp theo*
            if (hop.toNodePosition && !nodeDataMap.has(hop.toNodeId)) {
                nodeDataMap.set(hop.toNodeId, {
                    position: hop.toNodePosition,
                    bufferState: { queueSize: 0, bandwidthUtilization: 0 }, // Giả định cho node nhận cuối
                });
            }
        });

        const uniqueNodeData: NodeData[] = Array.from(nodeDataMap.entries()).map(([nodeId, nodeData]) => ({
            nodeId,
            ...nodeData
        }));

        const allPositions = uniqueNodeData.map(d => d.position);
        const allNodeIds = uniqueNodeData.map(d => d.nodeId);
        
        const normalizedNodes = normalizePositions(allPositions, width, height, allNodeIds);

        return { uniqueNodeData, normalizedNodes };
    }, [data, width, height]); // Phụ thuộc vào data, width, height

    const getNormalizedNodeData = (nodeId: string) => {
        const nodeData = uniqueNodeData.find(d => d.nodeId === nodeId);
        const normalizedCoord = normalizedNodes.find(n => n.nodeId === nodeId);
        
        return { 
            coord: normalizedCoord, 
            bufferState: nodeData?.bufferState ?? null 
        };
    };

    // Hàm renderRoute đã sửa lỗi Cannot find namespace 'JSX' (đã đổi JSX.Element[] thành React.ReactElement[])
    const renderRoute = (packet: Packet, color: string, name: string, visible: boolean) => {
        if (!visible || !packet?.hopRecords) return { lines: null, dropNode: null };

        const { hopRecords, dropped } = packet;
        const lines: React.ReactElement[] = []; // ✅ Sửa lỗi 2503
        let dropNode: NormalizedNode | null = null;
        
        // Vòng lặp chỉ qua các hop thành công
        hopRecords.forEach((hop, i) => {
            if (!hop) return;

            const fromData = getNormalizedNodeData(hop.fromNodeId);
            const toData = getNormalizedNodeData(hop.toNodeId);
            
            if (!fromData.coord || !toData.coord) return;

            const hopLatencyColor = getLatencyColor(hop.latencyMs, packet.serviceQoS?.maxLatencyMs || 200);
            const strokeWidth = 2.5;

            const textX = (fromData.coord.x + toData.coord.x) / 2;
            const textY = (fromData.coord.y + toData.coord.y) / 2;
            const angle = Math.atan2(
                toData.coord.y - fromData.coord.y, 
                toData.coord.x - fromData.coord.x
            ) * (180 / Math.PI);

            // Highlight nếu node của hop này đang được hover
            const isHighlighted = hoveredNode === hop.fromNodeId || hoveredNode === hop.toNodeId;
            
            lines.push(
                <g key={`${name}-hop-${i}`} opacity={isHighlighted ? 1 : 0.8}>
                    {/* Đường (Edge) */}
                    <line
                        x1={fromData.coord.x} 
                        y1={fromData.coord.y}
                        x2={toData.coord.x} 
                        y2={toData.coord.y}
                        stroke={hopLatencyColor}
                        strokeWidth={isHighlighted ? strokeWidth + 1 : strokeWidth}
                        opacity={0.8}
                        markerEnd={`url(#arrowhead-${color.substring(1)})`}
                    />

                    {/* Hiển thị Latency (Text trên Edge) */}
                    <text
                        x={textX} 
                        y={textY} 
                        fill="#1f2937" 
                        fontSize="11" 
                        fontWeight="600" 
                        textAnchor="middle"
                        transform={`rotate(${angle}, ${textX}, ${textY})`} 
                        dy={-6}
                    >
                        {hop.latencyMs.toFixed(1)}ms
                    </text>
                    
                    {/* Hiển thị Distance (Text dưới Edge) */}
                    <text
                        x={textX} 
                        y={textY} 
                        fill="#6b7280" 
                        fontSize="9" 
                        textAnchor="middle"
                        transform={`rotate(${angle}, ${textX}, ${textY})`} 
                        dy={10}
                    >
                        {hop.distanceKm.toFixed(1)}km
                    </text>
                </g>
            );

            // Nếu đây là hop cuối cùng VÀ gói bị drop
            if (dropped && i === hopRecords.length - 1) {
                // Node cuối cùng trong hopRecords là node ngay trước khi gói bị drop (currentHoldingNodeId)
                // Ta sử dụng currentHoldingNodeId để tìm node bị drop, vì toData.coord là node tiếp theo
                const dropCoord = normalizedNodes.find(n => n.nodeId === packet.currentHoldingNodeId);
                if (dropCoord) {
                    dropNode = dropCoord;
                }
            }
        });
        
        return { lines, dropNode };
    };

    if (!data || (!data.dijkstraPacket && !data.rlPacket)) {  // ✅ Fixed: rlPacket (capital P)
        return (
            <div className="p-6 bg-white rounded-2xl shadow-lg col-span-3 border border-gray-200">
                <p className="text-center text-gray-500">No route data available</p>
            </div>
        );
    }

    const dijkstraRoute = data.dijkstraPacket ? renderRoute(data.dijkstraPacket, DIJKSTRA_COLOR, "Dijkstra", showDijkstra) : { lines: null, dropNode: null };
    const rlRoute = data.rlPacket ? renderRoute(data.rlPacket, RL_COLOR, "RL", showRL) : { lines: null, dropNode: null };  // ✅ Fixed: rlPacket (capital P)

    return (
        <div className="p-6 bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-lg col-span-3 border border-gray-200">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-800">
                    🗺️ Route Visualization & Hop Performance
                </h2>
                
                {/* Toggle Buttons */}
                <div className="flex gap-2">
                    <button
                        onClick={() => setShowDijkstra(!showDijkstra)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            showDijkstra
                                ? 'bg-blue-500 text-white shadow-md'
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                        }`}
                    >
                        Dijkstra
                    </button>
                    <button
                        onClick={() => setShowRL(!showRL)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            showRL
                                ? 'bg-orange-500 text-white shadow-md'
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                        }`}
                    >
                        RL
                    </button>
                </div>
            </div>

            <div className="flex justify-center bg-white rounded-lg p-4 border border-gray-200">
                <svg width={width} height={height} className="rounded-lg">
                    <defs>
                        {/* Markers */}
                        <marker 
                            id={`arrowhead-${DIJKSTRA_COLOR.substring(1)}`} 
                            markerWidth="10" 
                            markerHeight="7" 
                            refX="5" 
                            refY="3.5" 
                            orient="auto"
                        >
                            <polygon points="0 0, 10 3.5, 0 7" fill={DIJKSTRA_COLOR} />
                        </marker>
                        <marker 
                            id={`arrowhead-${RL_COLOR.substring(1)}`} 
                            markerWidth="10" 
                            markerHeight="7" 
                            refX="5" 
                            refY="3.5" 
                            orient="auto"
                        >
                            <polygon points="0 0, 10 3.5, 0 7" fill={RL_COLOR} />
                        </marker>
                        
                        {/* Gradient for background */}
                        <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#f9fafb" />
                            <stop offset="100%" stopColor="#f3f4f6" />
                        </linearGradient>
                    </defs>
                    
                    {/* Background */}
                    <rect width={width} height={height} fill="url(#bgGradient)" />
                    
                    {/* Routes */}
                    {dijkstraRoute.lines}
                    {rlRoute.lines}
                    
                    {/* Nodes */}
                    {uniqueNodeData.map((nodeData) => {
                        const node = normalizedNodes.find(n => n.nodeId === nodeData.nodeId);
                        
                        if (!node || !nodeData.bufferState) return null;
                        
                        const bufferState = nodeData.bufferState;
                        const radius = getNodeSize(bufferState.bandwidthUtilization);
                        const isHovered = hoveredNode === node.nodeId;
                        
                        // Kiểm tra xem node này có phải là điểm drop không
                        const isDropNode = (data.dijkstraPacket?.dropped && data.dijkstraPacket?.currentHoldingNodeId === node.nodeId) ||
                                           (data.rlPacket?.dropped && data.rlPacket?.currentHoldingNodeId === node.nodeId);  // ✅ Fixed: rlPacket (capital P)

                        return (
                            <g 
                                key={node.nodeId}
                                onMouseEnter={() => setHoveredNode(node.nodeId)}
                                onMouseLeave={() => setHoveredNode(null)}
                                style={{ cursor: 'pointer' }}
                            >
                                {/* Hover effect circle */}
                                {isHovered && (
                                    <circle 
                                        cx={node.x} 
                                        cy={node.y} 
                                        r={radius + 4} 
                                        fill="none"
                                        stroke="#3b82f6"
                                        strokeWidth={2}
                                        opacity={0.5}
                                    />
                                )}
                                
                                {/* Main node circle */}
                                <circle 
                                    cx={node.x} 
                                    cy={node.y} 
                                    r={radius} 
                                    fill={isDropNode ? DROP_COLOR : bufferState.queueSize > 0 ? "#8b5cf6" : "#1f2937"} // Đổi màu node bị DROP
                                    stroke="#ffffff" 
                                    strokeWidth={2.5}
                                    style={{ transition: 'all 0.2s' }}
                                />

                                {/* Drop Halo Effect */}
                                {isDropNode && (
                                    <circle 
                                        cx={node.x} 
                                        cy={node.y} 
                                        r={radius + 6} 
                                        fill="none" 
                                        stroke={DROP_COLOR} 
                                        strokeWidth={1.5} 
                                        opacity={0.8} 
                                        style={{ animation: 'pulse 1.5s infinite' }}
                                    />
                                )}
                                
                                {/* Node ID */}
                                <text 
                                    x={node.x} 
                                    y={node.y} 
                                    dy={radius + 14} 
                                    textAnchor="middle" 
                                    fontSize="11" 
                                    fontWeight="600"
                                    fill="#374151"
                                >
                                    {node.nodeId}
                                </text>
                                
                                {/* Buffer State */}
                                <text 
                                    x={node.x} 
                                    y={node.y} 
                                    dy={-radius - 4} 
                                    textAnchor="middle" 
                                    fontSize="9" 
                                    fontWeight="500"
                                    fill="#1f2937"
                                >
                                    Q:{bufferState.queueSize} | BW:{(bufferState.bandwidthUtilization * 100).toFixed(0)}%
                                </text>
                            </g>
                        );
                    })}
                </svg>
            </div>
            
            {/* THÊM CSS MỚI CHO HIỆU ỨNG PULSE CẢNH BÁO DROP */}
            {/* ✅ Sửa lỗi 2322: Loại bỏ 'jsx' và 'global' */}
            <style>
                {`
                @keyframes pulse {
                    0% {
                        transform: scale(1);
                        opacity: 0.8;
                    }
                    50% {
                        transform: scale(1.1);
                        opacity: 0.2;
                    }
                    100% {
                        transform: scale(1);
                        opacity: 0.8;
                    }
                }
                `}
            </style>


            {/* Legend */}
            <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-gray-200">
                <h3 className="font-semibold text-gray-800 mb-3 text-sm">📋 Metrics Legend:</h3>
                <div className="grid grid-cols-3 gap-3 text-xs text-gray-700">
                    <div className="flex items-start gap-2">
                        <span className="text-green-500 font-bold">●</span>
                        <div>
                            <strong>Edge Color:</strong> Hop Latency (Green = Low, Red = High)
                        </div>
                    </div>
                    <div className="flex items-start gap-2">
                        <span className="text-gray-800 font-bold text-base">●</span>
                        <div>
                            <strong>Node Size:</strong> Bandwidth Utilization (Larger = Higher usage)
                        </div>
                    </div>
                    <div className="flex items-start gap-2">
                        <span className="text-purple-500 font-bold text-base">●</span>
                        <div>
                            <strong>Purple Node:</strong> Queue has packets (Dark gray = Empty)
                        </div>
                    </div>
                     <div className="flex items-start gap-2">
                        <span className="text-red-600 font-bold text-base">🔴</span>
                        <div>
                            <strong>Red Node:</strong> Packet **Dropped** at this node
                        </div>
                    </div>
                    <div className="flex items-start gap-2 col-span-2">
                        <span className="font-bold">📊</span>
                        <div>
                            <strong>Edge Labels:</strong> Latency (ms) & Distance (km) (Per hop metrics)
                        </div>
                    </div>
                </div>
            </div>

            {/* THÔNG SỐ TỔNG (Overall Stats) */}
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                {/* Dijkstra Stats */}
                {data.dijkstraPacket && (
                    <>
                        <div className={`p-3 rounded-lg border ${data.dijkstraPacket.dropped ? 'bg-red-50 border-red-200' : 'bg-blue-50 border-blue-200'}`}>
                            <p className="text-xs text-gray-600 mb-1 font-medium">Dijkstra Total Latency</p>
                            <p className={`text-lg font-bold ${data.dijkstraPacket.dropped ? 'text-red-700' : 'text-blue-700'}`}>
                                {data.dijkstraPacket.dropped ? '❌ DROPPED' : (data.dijkstraPacket.accumulatedDelayMs).toFixed(2) + ' ms'}
                            </p>
                        </div>
                        <div className={`p-3 rounded-lg border ${data.dijkstraPacket.dropped ? 'bg-red-50 border-red-200' : 'bg-blue-50 border-blue-200'}`}>
                            <p className="text-xs text-gray-600 mb-1 font-medium">Dijkstra Drop Reason</p>
                            <p className="text-sm font-bold text-gray-700 truncate" title={data.dijkstraPacket.dropReason || 'N/A'}>
                                {data.dijkstraPacket.dropped ? data.dijkstraPacket.dropReason || 'Unknown Reason' : 'N/A'}
                            </p>
                        </div>
                    </>
                )}

                {/* RL Stats */}
                {data.rlPacket && (  // ✅ Fixed: rlPacket (capital P)
                    <>
                        <div className={`p-3 rounded-lg border ${data.rlPacket.dropped ? 'bg-red-50 border-red-200' : 'bg-orange-50 border-orange-200'}`}>
                            <p className="text-xs text-gray-600 mb-1 font-medium">RL Total Latency</p>
                            <p className={`text-lg font-bold ${data.rlPacket.dropped ? 'text-red-700' : 'text-orange-700'}`}>
                                {data.rlPacket.dropped ? '❌ DROPPED' : (data.rlPacket.accumulatedDelayMs).toFixed(2) + ' ms'}
                            </p>
                        </div>
                        <div className={`p-3 rounded-lg border ${data.rlPacket.dropped ? 'bg-red-50 border-red-200' : 'bg-orange-50 border-orange-200'}`}>
                            <p className="text-xs text-gray-600 mb-1 font-medium">RL Drop Reason</p>
                            <p className="text-sm font-bold text-gray-700 truncate" title={data.rlPacket.dropReason || 'N/A'}>
                                {data.rlPacket.dropped ? data.rlPacket.dropReason || 'Unknown Reason' : 'N/A'}
                            </p>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};