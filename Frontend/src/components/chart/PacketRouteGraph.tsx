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
    if ( positions.length === 0 ) return [];

    // Filter out null/undefined positions
    const validPositions = positions.filter( p => p != null );
    if ( validPositions.length === 0 ) return [];

    const lats = validPositions.map( p => p.latitude );
    const lons = validPositions.map( p => p.longitude );
    const minLat = Math.min( ...lats );
    const maxLat = Math.max( ...lats );
    const minLon = Math.min( ...lons );
    const maxLon = Math.max( ...lons );

    const rangeLat = maxLat - minLat;
    const rangeLon = maxLon - minLon;
    const padding = 100; // Increased padding for better node separation

    if ( rangeLat === 0 && rangeLon === 0 ) {
        return validPositions.map( ( p, i ) => ( {
            x: width / 2,
            y: height / 2,
            id: `${ p.latitude }_${ p.longitude }`,
            nodeId: nodeIds[ i ],
        } ) );
    }

    return validPositions.map( ( p, i ) => ( {
        // Đã thêm kiểm tra chia cho 0 để an toàn hơn
        x: padding + ( ( p.longitude - minLon ) / ( rangeLon || 1 ) ) * ( width - 2 * padding ),
        y: height - padding - ( ( p.latitude - minLat ) / ( rangeLat || 1 ) ) * ( height - 2 * padding ),
        id: `${ p.latitude }_${ p.longitude }`,
        nodeId: nodeIds[ i ],
    } ) );
};

const getLatencyColor = ( latencyMs: number, maxLatency: number ): string => {
    const effectiveMax = maxLatency || 200;
    const ratio = Math.min( latencyMs / effectiveMax, 1 );

    if ( ratio < 0.3 ) return "#10b981"; // Green
    if ( ratio < 0.7 ) return "#facc15"; // Yellow
    return "#ef4444"; // Red
};

const getNodeSize = ( utilization: number ): number => {
    return 6 + utilization * 8; // Kích thước Node từ 6 đến 14
};

// --- MAIN COMPONENT ---

export const PacketRouteGraph: React.FC<Props> = ( { data } ) => {
    const [ hoveredNode, setHoveredNode ] = useState<string | null>( null );
    const [ showDijkstra, setShowDijkstra ] = useState( true );
    const [ showRL, setShowRL ] = useState( true );

    const width = 900;
    const height = 550;

    const DIJKSTRA_COLOR = "#e11d48"; // Rose-600 - Dijkstra path
    const RL_COLOR = "#0d9488";        // Teal-600 - RL path (contrasting color)
    const DROP_COLOR = "#dc2626";      // Red-600 - Dropped packets

    // SỬ DỤNG useMemo để tính toán Node Data chỉ một lần
    const { uniqueNodeData, normalizedNodes } = useMemo( () => {
        const nodeDataMap = new Map<string, { position: Position, bufferState: BufferState | null }>();

        // Thêm currentHoldingNodeId (node cuối cùng trước khi drop) vào map nếu bị drop
        const allPackets = [ data?.dijkstraPacket, data?.rlPacket ];  // ✅ Fixed: rlPacket (capital P)

        allPackets.forEach( packet => {
            if ( packet?.dropped && packet.currentHoldingNodeId ) {
                // Nếu gói bị drop, node cuối cùng (currentHoldingNodeId) không có bufferState cập nhật
                // Nó sẽ được cập nhật ở bước hopRecords
            }
        } );


        const allHopRecords = [
            ...( data?.dijkstraPacket?.hopRecords || [] ),
            ...( data?.rlPacket?.hopRecords || [] )  // ✅ Fixed: rlPacket (capital P)
        ];

        allHopRecords.forEach( hop => {
            if ( !hop ) return;

            // Lấy dữ liệu Buffer State và Position từ FromNode (chỉ nếu position hợp lệ)
            if ( hop.fromNodePosition ) {
                nodeDataMap.set( hop.fromNodeId, {
                    position: hop.fromNodePosition,
                    bufferState: hop.fromNodeBufferState,
                } );
            }

            // Lấy dữ liệu Position từ ToNode (chỉ nếu position hợp lệ)
            // *Lưu ý: bufferState của ToNode sẽ được giả định hoặc lấy từ FromNode tiếp theo*
            if ( hop.toNodePosition && !nodeDataMap.has( hop.toNodeId ) ) {
                nodeDataMap.set( hop.toNodeId, {
                    position: hop.toNodePosition,
                    bufferState: { queueSize: 0, bandwidthUtilization: 0 }, // Giả định cho node nhận cuối
                } );
            }
        } );

        const uniqueNodeData: NodeData[] = Array.from( nodeDataMap.entries() ).map( ( [ nodeId, nodeData ] ) => ( {
            nodeId,
            ...nodeData
        } ) );

        const allPositions = uniqueNodeData.map( d => d.position );
        const allNodeIds = uniqueNodeData.map( d => d.nodeId );

        const normalizedNodes = normalizePositions( allPositions, width, height, allNodeIds );

        return { uniqueNodeData, normalizedNodes };
    }, [ data, width, height ] ); // Phụ thuộc vào data, width, height

    const getNormalizedNodeData = ( nodeId: string ) => {
        const nodeData = uniqueNodeData.find( d => d.nodeId === nodeId );
        const normalizedCoord = normalizedNodes.find( n => n.nodeId === nodeId );

        return {
            coord: normalizedCoord,
            bufferState: nodeData?.bufferState ?? null
        };
    };

    // Helper function to calculate perpendicular offset for path separation
    const getOffsetPoint = ( x: number, y: number, dx: number, dy: number, offset: number ) => {
        const length = Math.sqrt( dx * dx + dy * dy );
        if ( length === 0 ) return { x, y };
        const perpX = -dy / length * offset;
        const perpY = dx / length * offset;
        return { x: x + perpX, y: y + perpY };
    };

    // Render route with offset to avoid overlap - pathOffset separates Dijkstra from RL visually
    const renderRoute = ( packet: Packet, color: string, name: string, visible: boolean, pathOffset: number = 0 ) => {
        if ( !visible || !packet?.hopRecords ) return { lines: null, dropNode: null };

        const { hopRecords, dropped } = packet;
        const lines: React.ReactElement[] = [];
        let dropNode: NormalizedNode | null = null;

        hopRecords.forEach( ( hop, i ) => {
            if ( !hop ) return;

            const fromData = getNormalizedNodeData( hop.fromNodeId );
            const toData = getNormalizedNodeData( hop.toNodeId );

            if ( !fromData.coord || !toData.coord ) return;

            // Calculate direction and apply perpendicular offset
            const dx = toData.coord.x - fromData.coord.x;
            const dy = toData.coord.y - fromData.coord.y;
            const fromOffset = getOffsetPoint( fromData.coord.x, fromData.coord.y, dx, dy, pathOffset );
            const toOffset = getOffsetPoint( toData.coord.x, toData.coord.y, dx, dy, pathOffset );

            const strokeWidth = 2.5;

            // Use offset coordinates for text positioning
            const textX = ( fromOffset.x + toOffset.x ) / 2;
            const textY = ( fromOffset.y + toOffset.y ) / 2;

            const isHighlighted = hoveredNode === hop.fromNodeId || hoveredNode === hop.toNodeId;

            lines.push(
                <g key={ `${ name }-hop-${ i }` } opacity={ 1 }>
                    {/* Path line with offset - thicker and more visible */ }
                    <line
                        x1={ fromOffset.x }
                        y1={ fromOffset.y }
                        x2={ toOffset.x }
                        y2={ toOffset.y }
                        stroke={ color }
                        strokeWidth={ isHighlighted ? 4.5 : 3.5 }
                        opacity={ 1 }
                        markerEnd={ `url(#arrowhead-${ color.substring( 1 ) })` }
                    />

                    {/* Show latency only for first and last hops to reduce clutter */ }
                    { ( i === 0 || i === hopRecords.length - 1 ) && (
                        <text
                            x={ textX }
                            y={ textY }
                            fill={ color }
                            fontSize="9"
                            fontWeight="600"
                            textAnchor="middle"
                            dy={ pathOffset > 0 ? 16 : -12 }
                        >
                            { hop.latencyMs.toFixed( 1 ) }ms
                        </text>
                    ) }
                </g>
            );

            if ( dropped && i === hopRecords.length - 1 ) {
                const dropCoord = normalizedNodes.find( n => n.nodeId === packet.currentHoldingNodeId );
                if ( dropCoord ) {
                    dropNode = dropCoord;
                }
            }
        } );

        return { lines, dropNode };
    };

    if ( !data || ( !data.dijkstraPacket && !data.rlPacket ) ) {  // ✅ Fixed: rlPacket (capital P)
        return (
            <div className="p-6 bg-white rounded-2xl shadow-lg col-span-3 border border-gray-200">
                <p className="text-center text-gray-500">No route data available</p>
            </div>
        );
    }

    const dijkstraRoute = data.dijkstraPacket ? renderRoute( data.dijkstraPacket, DIJKSTRA_COLOR, "Dijkstra", showDijkstra, 0 ) : { lines: null, dropNode: null };
    const rlRoute = data.rlPacket ? renderRoute( data.rlPacket, RL_COLOR, "RL", showRL, 0 ) : { lines: null, dropNode: null };

    return (
        <div className="p-6 bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-lg col-span-3 border border-gray-200">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-800">
                    🗺️ Route Visualization & Hop Performance
                </h2>

                {/* Toggle Buttons */ }
                <div className="flex gap-2">
                    <button
                        onClick={ () => setShowDijkstra( !showDijkstra ) }
                        className={ `px-4 py-2 rounded-lg text-sm font-medium transition-all ${ showDijkstra
                            ? 'bg-rose-500 text-white shadow-md'
                            : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                            }` }
                    >
                        Dijkstra
                    </button>
                    <button
                        onClick={ () => setShowRL( !showRL ) }
                        className={ `px-4 py-2 rounded-lg text-sm font-medium transition-all ${ showRL
                            ? 'bg-teal-500 text-white shadow-md'
                            : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                            }` }
                    >
                        RL
                    </button>
                </div>
            </div>

            {/* 7:3 GRID: Visualization (Left, larger) | Information (Right, smaller) */ }
            <div className="grid grid-cols-10 gap-4">
                {/* LEFT COLUMN: SVG Visualization - 7 cols */ }
                <div className="col-span-7 bg-white rounded-lg p-4 border border-gray-200 flex items-center justify-center">
                    <svg width={ width } height={ height } className="rounded-lg">
                        <defs>
                            {/* Markers */ }
                            <marker
                                id={ `arrowhead-${ DIJKSTRA_COLOR.substring( 1 ) }` }
                                markerWidth="10"
                                markerHeight="7"
                                refX="5"
                                refY="3.5"
                                orient="auto"
                            >
                                <polygon points="0 0, 10 3.5, 0 7" fill={ DIJKSTRA_COLOR } />
                            </marker>
                            <marker
                                id={ `arrowhead-${ RL_COLOR.substring( 1 ) }` }
                                markerWidth="10"
                                markerHeight="7"
                                refX="5"
                                refY="3.5"
                                orient="auto"
                            >
                                <polygon points="0 0, 10 3.5, 0 7" fill={ RL_COLOR } />
                            </marker>

                            {/* Gradient for background */ }
                            <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#f9fafb" />
                                <stop offset="100%" stopColor="#f3f4f6" />
                            </linearGradient>
                        </defs>

                        {/* Background */ }
                        <rect width={ width } height={ height } fill="url(#bgGradient)" />

                        {/* Routes */ }
                        { dijkstraRoute.lines }
                        { rlRoute.lines }

                        {/* Nodes */ }
                        { uniqueNodeData.map( ( nodeData ) => {
                            const node = normalizedNodes.find( n => n.nodeId === nodeData.nodeId );

                            if ( !node || !nodeData.bufferState ) return null;

                            const bufferState = nodeData.bufferState;
                            const radius = getNodeSize( bufferState.bandwidthUtilization );
                            const isHovered = hoveredNode === node.nodeId;

                            // Kiểm tra xem node này có phải là điểm drop không
                            const isDropNode = ( data.dijkstraPacket?.dropped && data.dijkstraPacket?.currentHoldingNodeId === node.nodeId ) ||
                                ( data.rlPacket?.dropped && data.rlPacket?.currentHoldingNodeId === node.nodeId );  // ✅ Fixed: rlPacket (capital P)

                            return (
                                <g
                                    key={ node.nodeId }
                                    onMouseEnter={ () => setHoveredNode( node.nodeId ) }
                                    onMouseLeave={ () => setHoveredNode( null ) }
                                    style={ { cursor: 'pointer' } }
                                >
                                    {/* Hover effect circle */ }
                                    { isHovered && (
                                        <circle
                                            cx={ node.x }
                                            cy={ node.y }
                                            r={ radius + 4 }
                                            fill="none"
                                            stroke="#3b82f6"
                                            strokeWidth={ 2 }
                                            opacity={ 0.5 }
                                        />
                                    ) }

                                    {/* Main node circle */ }
                                    <circle
                                        cx={ node.x }
                                        cy={ node.y }
                                        r={ radius }
                                        fill={ isDropNode ? DROP_COLOR : bufferState.queueSize > 0 ? "#8b5cf6" : "#1f2937" } // Đổi màu node bị DROP
                                        stroke="#ffffff"
                                        strokeWidth={ 2.5 }
                                        style={ { transition: 'all 0.2s' } }
                                    />

                                    {/* Drop Halo Effect */ }
                                    { isDropNode && (
                                        <circle
                                            cx={ node.x }
                                            cy={ node.y }
                                            r={ radius + 6 }
                                            fill="none"
                                            stroke={ DROP_COLOR }
                                            strokeWidth={ 1.5 }
                                            opacity={ 0.8 }
                                            style={ { animation: 'pulse 1.5s infinite' } }
                                        />
                                    ) }

                                    {/* Node ID - show only on hover or for key nodes */ }
                                    { isHovered && (
                                        <>
                                            <text
                                                x={ node.x }
                                                y={ node.y }
                                                dy={ radius + 14 }
                                                textAnchor="middle"
                                                fontSize="10"
                                                fontWeight="600"
                                                fill="#374151"
                                            >
                                                { node.nodeId }
                                            </text>
                                            <text
                                                x={ node.x }
                                                y={ node.y }
                                                dy={ -radius - 4 }
                                                textAnchor="middle"
                                                fontSize="8"
                                                fontWeight="500"
                                                fill="#6b7280"
                                            >
                                                Q:{ bufferState.queueSize } | BW:{ ( bufferState.bandwidthUtilization * 100 ).toFixed( 0 ) }%
                                            </text>
                                        </>
                                    ) }
                                </g>
                            );
                        } ) }
                    </svg>
                </div>
                {/* END LEFT COLUMN */ }

                {/* RIGHT COLUMN: Compact Metrics Panel - 3 cols */ }
                <div className="col-span-3 space-y-3">
                    {/* Algorithm Comparison Table */ }
                    <div className="bg-white rounded-lg border border-gray-300 overflow-hidden">
                        <div className="bg-gradient-to-r from-rose-600 to-teal-600 px-3 py-2">
                            <h3 className="font-bold text-white text-sm uppercase tracking-wide">⚡ Performance Metrics</h3>
                        </div>

                        <div className="divide-y divide-gray-200">
                            {/* Dijkstra Row */ }
                            { data.dijkstraPacket && (
                                <div className="p-3 hover:bg-blue-50 transition-colors">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs font-bold text-blue-700 uppercase tracking-wide">🔵 Dijkstra</span>
                                        { data.dijkstraPacket.dropped && (
                                            <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs font-bold rounded-full">DROPPED</span>
                                        ) }
                                    </div>
                                    <div className="grid grid-cols-2 gap-2 text-xs">
                                        <div>
                                            <p className="text-gray-500 font-medium">Latency</p>
                                            <p className={ `font-bold ${ data.dijkstraPacket.dropped ? 'text-red-600' : 'text-rose-600' }` }>
                                                { data.dijkstraPacket.dropped ? 'N/A' : ( data.dijkstraPacket.accumulatedDelayMs ).toFixed( 1 ) + ' ms' }
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-gray-500 font-medium">Hops</p>
                                            <p className="font-bold text-gray-700">{ data.dijkstraPacket.hopRecords.length }</p>
                                        </div>
                                    </div>
                                    { data.dijkstraPacket.dropped && (
                                        <div className="mt-2 pt-2 border-t border-red-200">
                                            <p className="text-xs text-red-700 font-medium truncate" title={ data.dijkstraPacket.dropReason || 'Unknown' }>
                                                ⚠️ { data.dijkstraPacket.dropReason || 'Unknown Reason' }
                                            </p>
                                        </div>
                                    ) }
                                    {/* Path Display */ }
                                    <div className="mt-2 pt-2 border-t border-blue-200">
                                        <p className="text-xs text-gray-500 font-medium mb-1">Path:</p>
                                        <p className="text-xs text-rose-700 font-mono font-semibold break-all leading-relaxed">
                                            { data.dijkstraPacket.pathHistory.join( ' → ' ) }
                                        </p>
                                    </div>
                                </div>
                            ) }

                            {/* RL Row */ }
                            { data.rlPacket && (
                                <div className="p-3 hover:bg-teal-50 transition-colors">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs font-bold text-teal-700 uppercase tracking-wide">🟠 RL</span>
                                        { data.rlPacket.dropped && (
                                            <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs font-bold rounded-full">DROPPED</span>
                                        ) }
                                    </div>
                                    <div className="grid grid-cols-2 gap-2 text-xs">
                                        <div>
                                            <p className="text-gray-500 font-medium">Latency</p>
                                            <p className={ `font-bold ${ data.rlPacket.dropped ? 'text-red-600' : 'text-teal-600' }` }>
                                                { data.rlPacket.dropped ? 'N/A' : ( data.rlPacket.accumulatedDelayMs ).toFixed( 1 ) + ' ms' }
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-gray-500 font-medium">Hops</p>
                                            <p className="font-bold text-gray-700">{ data.rlPacket.hopRecords.length }</p>
                                        </div>
                                    </div>
                                    { data.rlPacket.dropped && (
                                        <div className="mt-2 pt-2 border-t border-red-200">
                                            <p className="text-xs text-red-700 font-medium truncate" title={ data.rlPacket.dropReason || 'Unknown' }>
                                                ⚠️ { data.rlPacket.dropReason || 'Unknown Reason' }
                                            </p>
                                        </div>
                                    ) }
                                    {/* Path Display */ }
                                    <div className="mt-2 pt-2 border-t border-teal-200">
                                        <p className="text-xs text-gray-500 font-medium mb-1">Path:</p>
                                        <p className="text-xs text-teal-700 font-mono font-semibold break-all leading-relaxed">
                                            { data.rlPacket.pathHistory.join( ' → ' ) }
                                        </p>
                                    </div>
                                </div>
                            ) }
                        </div>
                    </div>

                    {/* Quick Legend */ }
                    <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-3 border border-gray-300">
                        <h4 className="text-xs font-bold text-gray-700 uppercase tracking-wide mb-2">📊 Visual Guide</h4>
                        <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                            <div className="flex items-center gap-1.5">
                                <span className="w-3 h-3 bg-gradient-to-r from-green-400 to-red-500 rounded-full"></span>
                                <span>Edge: Latency</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <div className="flex items-center">
                                    <span className="w-2 h-2 bg-gray-800 rounded-full"></span>
                                    <span className="w-3 h-3 bg-gray-800 rounded-full"></span>
                                </div>
                                <span>Size: Bandwidth</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className="w-3 h-3 bg-purple-500 rounded-full"></span>
                                <span>Queue Active</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className="w-3 h-3 bg-red-600 rounded-full"></span>
                                <span>Packet Drop</span>
                            </div>
                        </div>
                    </div>
                </div>
                {/* END RIGHT COLUMN */ }
            </div>
            {/* END 2-COLUMN GRID */ }

            {/* THÊM CSS MỚI CHO HIỆU ỨNG PULSE CẢNH BÁO DROP */ }
            {/* ✅ Sửa lỗi 2322: Loại bỏ 'jsx' và 'global' */ }
            <style>
                { `
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

        </div>
    );
};
