// src/hooks/usePacketProcessor.ts
import { useState, useEffect, useRef } from 'react';
import type {
    Packet, ChartData, AggregatedData, SankeyAggregator,
    KpiData, LatencyDistData, SankeyData
} from '../types/type';
import { EMPTY_CHART_DATA } from '../types/type';

// ----------------------------------------------
// 1. LOGIC XỬ LÝ (Di chuyển từ dataProcessor.ts)
// ----------------------------------------------
// (Tất cả logic aggregatePacketData, formatChartData, processAllData
//  được copy/paste vào đây. Chúng ta đặt chúng bên trong file này
//  để hook này là một module độc lập.)

const LATENCY_BIN_SIZE_MS = 10;
const LATENCY_BIN_COUNT = 10;

const aggregatePacketData = ( packets: Packet[] ): AggregatedData => {
    // ... (Nội dung hàm aggregatePacketData giữ nguyên) ...
    const data: AggregatedData = {
        stats: {
            rl: { total: 0, dropped: 0, totalLatency: 0, qosSuccess: 0 },
            nonRl: { total: 0, dropped: 0, totalLatency: 0, qosSuccess: 0 }
        },
        correlationData: { rl: [], nonRl: [] },
        latencyBins: {
            rl: new Array( LATENCY_BIN_COUNT ).fill( 0 ),
            nonRl: new Array( LATENCY_BIN_COUNT ).fill( 0 )
        },
        sankey: {
            rl: { nodes: new Map(), links: new Map(), nodeIndex: 0 },
            nonRl: { nodes: new Map(), links: new Map(), nodeIndex: 0 }
        }
    };

    for ( const packet of packets ) {
        const key = packet.isUseRL ? 'rl' : 'nonRl';
        const groupStats = data.stats[ key ];
        const sankeyAgg = data.sankey[ key ];

        groupStats.total++;

        if ( packet.hopRecords && packet.hopRecords.length > 0 ) {
            let { nodeIndex } = sankeyAgg;
            for ( const hop of packet.hopRecords ) {
                const { fromNodeId: from, toNodeId: to } = hop;
                if ( !sankeyAgg.nodes.has( from ) ) sankeyAgg.nodes.set( from, nodeIndex++ );
                if ( !sankeyAgg.nodes.has( to ) ) sankeyAgg.nodes.set( to, nodeIndex++ );
                const linkKey = `${ from } -> ${ to }`;
                const link = sankeyAgg.links.get( linkKey ) ?? { source: from, target: to, value: 0 };
                link.value++;
                sankeyAgg.links.set( linkKey, link );
            }
            sankeyAgg.nodeIndex = nodeIndex;
        }

        if ( packet.dropped ) {
            groupStats.dropped++;
            continue;
        }

        groupStats.totalLatency += packet.accumulatedDelayMs;
        if ( packet.accumulatedDelayMs <= packet.serviceQoS.maxLatencyMs ) {
            groupStats.qosSuccess++;
        }

        if ( packet.hopRecords && packet.hopRecords.length > 0 ) {
            const hop = packet.hopRecords[ 0 ];
            if ( hop.fromNodeBufferState?.utilization !== undefined ) {
                data.correlationData[ key ].push( {
                    utilization: hop.fromNodeBufferState.utilization,
                    latencyMs: hop.latencyMs
                } );
            }
        }

        const binIndex = Math.min(
            Math.floor( packet.accumulatedDelayMs / LATENCY_BIN_SIZE_MS ),
            LATENCY_BIN_COUNT - 1
        );
        data.latencyBins[ key ][ binIndex ]++;
    }

    return data;
};

const formatChartData = ( data: AggregatedData ): ChartData => {
    // ... (Nội dung hàm formatChartData giữ nguyên) ...
    const { stats, correlationData, latencyBins, sankey } = data;

    const rlDelivered = ( stats.rl.total - stats.rl.dropped ) || 1;
    const nonRlDelivered = ( stats.nonRl.total - stats.nonRl.dropped ) || 1;

    const kpiData: KpiData[] = [
        { name: 'Avg. Latency (ms)', RL: parseFloat( ( stats.rl.totalLatency / rlDelivered ).toFixed( 2 ) ), NonRL: parseFloat( ( stats.nonRl.totalLatency / nonRlDelivered ).toFixed( 2 ) ) },
        { name: 'Packet Loss Rate (%)', RL: parseFloat( ( ( stats.rl.dropped / ( stats.rl.total || 1 ) ) * 100 ).toFixed( 2 ) ), NonRL: parseFloat( ( ( stats.nonRl.dropped / ( stats.nonRl.total || 1 ) ) * 100 ).toFixed( 2 ) ) }
    ];

    const latencyDist: LatencyDistData[] = latencyBins.rl.map( ( rlValue, i ) => {
        const name = ( i === LATENCY_BIN_COUNT - 1 )
            ? `${ i * LATENCY_BIN_SIZE_MS }+ ms`
            : `${ i * LATENCY_BIN_SIZE_MS }-${ ( i + 1 ) * LATENCY_BIN_SIZE_MS }ms`;
        return { name, RL: rlValue, NonRL: latencyBins.nonRl[ i ] };
    } );

    const formatSankeyData = ( aggregator: SankeyAggregator ): SankeyData => {
        const nodes = Array.from( aggregator.nodes.keys() ).map( name => ( { name } ) );
        const links = Array.from( aggregator.links.values() ).map( link => ( {
            source: aggregator.nodes.get( link.source )!,
            target: aggregator.nodes.get( link.target )!,
            value: link.value
        } ) );
        return { nodes, links };
    };

    const sankeyDataRL = formatSankeyData( sankey.rl );
    const sankeyDataNonRL = formatSankeyData( sankey.nonRl );

    return { kpiData, correlationData, latencyDist, sankeyDataRL, sankeyDataNonRL };
};

const processAllData = ( packets: Packet[] ): ChartData => {
    // ... (Nội dung hàm processAllData giữ nguyên) ...
    if ( !packets || packets.length === 0 ) {
        return EMPTY_CHART_DATA;
    }
    const aggregatedData = aggregatePacketData( packets );
    const chartData = formatChartData( aggregatedData );
    return chartData;
};


// ----------------------------------------------
// 2. WEB SOCKET HOOK (LOGIC MỚI)
// ----------------------------------------------

/**
 * Hook tùy chỉnh để kết nối WebSocket, nhận packet,
 * và xử lý chúng theo lô (batch) để cập nhật UI.
 * @param url Địa chỉ WebSocket server (ví dụ: 'ws://localhost:8080/packets')
 * @param updateIntervalMs Tần suất cập nhật UI (mili-giây).
 */
export const usePacketProcessor = ( url: string, updateIntervalMs = 1000 ) => {
    // State 1: Dữ liệu đã xử lý để vẽ biểu đồ
    const [ chartData, setChartData ] = useState<ChartData>( EMPTY_CHART_DATA );

    // State 2: Trạng thái kết nối
    const [ isConnected, setIsConnected ] = useState( false );

    // State 3: Tổng số packet đã nhận
    const [ totalPackets, setTotalPackets ] = useState( 0 );

    // Ref: Dùng để lưu trữ *TẤT CẢ* packet đã nhận.
    // Dùng ref để tránh re-render mỗi khi có packet mới.
    const allPacketsRef = useRef<Packet[]>( [] );

    // Effect 1: Quản lý kết nối WebSocket
    useEffect( () => {
        console.log( `[WS] Đang cố gắng kết nối đến ${ url }...` );
        const ws = new WebSocket( url );

        ws.onopen = () => {
            console.log( "[WS] Đã kết nối." );
            setIsConnected( true );
            allPacketsRef.current = []; // Xóa dữ liệu cũ khi kết nối mới
        };

        ws.onmessage = ( event ) => {
            try {
                const newPacket = JSON.parse( event.data ) as Packet;
                // Thêm packet mới vào ref mà không re-render
                allPacketsRef.current.push( newPacket );
            } catch ( error ) {
                console.error( "[WS] Lỗi parse JSON:", error );
            }
        };

        ws.onerror = ( error ) => {
            console.error( "[WS] Lỗi:", error );
            setIsConnected( false );
        };

        ws.onclose = () => {
            console.log( "[WS] Đã ngắt kết nối." );
            setIsConnected( false );
        };

        // Hàm dọn dẹp: Đóng kết nối khi component unmount
        return () => {
            ws.close();
        };
    }, [ url ] ); // Chỉ chạy lại effect này nếu URL thay đổi

    // Effect 2: Interval để xử lý dữ liệu theo lô
    useEffect( () => {
        const intervalId = setInterval( () => {
            // Chỉ xử lý nếu có packet mới
            const currentPacketCount = allPacketsRef.current.length;
            if ( currentPacketCount === 0 || currentPacketCount === totalPackets ) {
                return; // Không có gì mới, bỏ qua
            }

            // Xử lý *toàn bộ* danh sách packet
            // (Đây là lúc tốn CPU, nhưng chỉ 1 lần/giây)
            const newChartData = processAllData( allPacketsRef.current );

            // Cập nhật state (kích hoạt re-render)
            setChartData( newChartData );
            setTotalPackets( currentPacketCount );

        }, updateIntervalMs ); // Chạy mỗi giây

        // Hàm dọn dẹp: Dừng interval
        return () => {
            clearInterval( intervalId );
        };
    }, [ updateIntervalMs, totalPackets ] ); // Chạy lại nếu interval thay đổi

    // Trả về dữ liệu cho component
    return { chartData, isConnected, totalPackets };
};