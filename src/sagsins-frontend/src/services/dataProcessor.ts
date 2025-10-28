// src/dataProcessor.ts
// Contains all logic for generating, processing, and formatting data.

import type {
    Packet, ServiceType, ChartData, AggregatedData,
    SankeyAggregator, KpiData, LatencyDistData, SankeyData
} from '../types/type';

// ----------------------------------------------
// 1. STATIC & SIMULATION DATA
// ----------------------------------------------
export const STATIC_PACKET_DATA: Packet[] = [
    {
        "packetId": "pkt_audio_run_A_RL", "isUseRL": true, "serviceQoS": { "serviceType": "AUDIO_CALL", "maxLatencyMs": 80.0 }, "accumulatedDelayMs": 25.5, "dropped": false,
        "hopRecords": [
            { "fromNodeId": "Ground_Station_A", "toNodeId": "LEO_RL_1", "latencyMs": 10.1, "fromNodeBufferState": { "queueSize": 70, "utilization": 0.80 }, "routingDecisionInfo": { "algorithm": "DQN_Agent" } },
            { "fromNodeId": "LEO_RL_1", "toNodeId": "LEO_RL_2", "latencyMs": 5.4, "fromNodeBufferState": { "queueSize": 60, "utilization": 0.70 }, "routingDecisionInfo": { "algorithm": "DQN_Agent" } },
            { "fromNodeId": "LEO_RL_2", "toNodeId": "User_B", "latencyMs": 10.0, "fromNodeBufferState": { "queueSize": 20, "utilization": 0.30 }, "routingDecisionInfo": { "algorithm": "DQN_Agent" } }
        ]
    },
    {
        "packetId": "pkt_audio_run_A_NonRL", "isUseRL": false, "serviceQoS": { "serviceType": "AUDIO_CALL", "maxLatencyMs": 80.0 }, "accumulatedDelayMs": 110.7, "dropped": false,
        "hopRecords": [
            { "fromNodeId": "Ground_Station_A", "toNodeId": "LEO_SLOW_1", "latencyMs": 40.2, "fromNodeBufferState": { "queueSize": 10, "utilization": 0.10 }, "routingDecisionInfo": { "algorithm": "OSPF" } },
            { "fromNodeId": "LEO_SLOW_1", "toNodeId": "LEO_SLOW_2", "latencyMs": 40.5, "fromNodeBufferState": { "queueSize": 15, "utilization": 0.15 }, "routingDecisionInfo": { "algorithm": "OSPF" } },
            { "fromNodeId": "LEO_SLOW_2", "toNodeId": "User_B", "latencyMs": 30.0, "fromNodeBufferState": { "queueSize": 10, "utilization": 0.10 }, "routingDecisionInfo": { "algorithm": "OSPF" } }
        ]
    },
    {
        "packetId": "pkt_video_run_B_RL", "isUseRL": true, "serviceQoS": { "serviceType": "VIDEO_STREAM", "maxLatencyMs": 150.0 }, "accumulatedDelayMs": 40.1, "dropped": false,
        "hopRecords": [
            { "fromNodeId": "Ground_Station_A", "toNodeId": "LEO_RL_1", "latencyMs": 10.0, "fromNodeBufferState": { "queueSize": 70, "utilization": 0.80 }, "routingDecisionInfo": { "algorithm": "DQN_Agent" } },
            { "fromNodeId": "LEO_RL_1", "toNodeId": "User_C", "latencyMs": 30.1, "fromNodeBufferState": { "queueSize": 40, "utilization": 0.50 }, "routingDecisionInfo": { "algorithm": "DQN_Agent" } }
        ]
    },
    {
        "packetId": "pkt_video_run_B_NonRL", "isUseRL": false, "serviceQoS": { "serviceType": "VIDEO_STREAM", "maxLatencyMs": 150.0 }, "accumulatedDelayMs": 9999.0, "dropped": true,
        "hopRecords": [
            { "fromNodeId": "Ground_Station_A", "toNodeId": "LEO_SLOW_1", "latencyMs": 40.0, "fromNodeBufferState": { "queueSize": 10, "utilization": 0.10 }, "routingDecisionInfo": { "algorithm": "OSPF" } }
        ]
    }
];

export const DEMO_PACKET_COUNT = 1000;
const SIM_PARAMS = {
    RL_DROP_CHANCE: 0.02,
    NON_RL_DROP_CHANCE: 0.10,
    MIN_HOPS: 2,
    MAX_HOPS: 3,
    RL_BASE_LATENCY: 5,
    NON_RL_BASE_LATENCY: 20,
};

const LATENCY_BIN_SIZE_MS = 10;
const LATENCY_BIN_COUNT = 10;

// ----------------------------------------------
// 2. DATA GENERATOR
// ----------------------------------------------
export const generateDemoData = ( count = DEMO_PACKET_COUNT ): Packet[] => {
    const packets: Packet[] = [];
    const services: Array<{ type: ServiceType, maxLatency: number }> = [
        { type: "AUDIO_CALL", maxLatency: 80.0 },
        { type: "VIDEO_STREAM", maxLatency: 150.0 },
        { type: "FILE_TRANSFER", maxLatency: 2000.0 }
    ];

    for ( let i = 0; i < count; i++ ) {
        const isUseRL = Math.random() > 0.5;
        const service = services[ Math.floor( Math.random() * services.length ) ];
        const dropped = Math.random() < ( isUseRL ? SIM_PARAMS.RL_DROP_CHANCE : SIM_PARAMS.NON_RL_DROP_CHANCE );

        let accumulatedDelayMs = 0;
        const hopRecords: Packet[ 'hopRecords' ] = []; // Type safety

        const numHops = Math.floor( Math.random() * ( SIM_PARAMS.MAX_HOPS - SIM_PARAMS.MIN_HOPS + 1 ) ) + SIM_PARAMS.MIN_HOPS;
        let currentNode = "Ground_Station";

        for ( let j = 0; j < numHops; j++ ) {
            const baseNodeName = isUseRL ? "LEO_RL_" : "LEO_NONRL_";
            const nextNode = ( j === numHops - 1 ) ? "User_Terminal" : `${ baseNodeName }${ j + 1 }`;

            const hopUtil = isUseRL ? ( Math.random() * 0.5 + 0.1 ) : ( Math.random() * 0.8 + 0.1 );
            const hopLatency = ( isUseRL ? SIM_PARAMS.RL_BASE_LATENCY : SIM_PARAMS.NON_RL_BASE_LATENCY ) + ( hopUtil * 10 ) + ( Math.random() * ( isUseRL ? 2 : 10 ) );

            accumulatedDelayMs += hopLatency;

            hopRecords.push( {
                fromNodeId: currentNode,
                toNodeId: nextNode,
                latencyMs: hopLatency,
                fromNodeBufferState: { utilization: hopUtil, queueSize: Math.floor( hopUtil * 100 ) },
                routingDecisionInfo: { algorithm: isUseRL ? "DQN" : "OSPF" }
            } );
            currentNode = nextNode;
        }

        packets.push( {
            packetId: `pkt_demo_${ i }`,
            isUseRL: isUseRL,
            serviceQoS: { serviceType: service.type, maxLatencyMs: service.maxLatency },
            accumulatedDelayMs: accumulatedDelayMs,
            dropped: dropped,
            hopRecords: hopRecords
        } );
    }
    return packets;
};


// ----------------------------------------------
// 3. DATA PROCESSOR
// ----------------------------------------------

/** Step 1: Aggregate raw data from packets */
const aggregatePacketData = ( packets: Packet[] ): AggregatedData => {
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

/** Step 2: Format aggregated data for Recharts */
const formatChartData = ( data: AggregatedData ): ChartData => {
    const { stats, correlationData, latencyBins, sankey } = data;

    const rlDelivered = ( stats.rl.total - stats.rl.dropped ) || 1;
    const nonRlDelivered = ( stats.nonRl.total - stats.nonRl.dropped ) || 1;

    const kpiData: KpiData[] = [
        {
            name: 'Avg. Latency (ms)',
            RL: parseFloat( ( stats.rl.totalLatency / rlDelivered ).toFixed( 2 ) ),
            NonRL: parseFloat( ( stats.nonRl.totalLatency / nonRlDelivered ).toFixed( 2 ) )
        },
        {
            name: 'Packet Loss Rate (%)',
            RL: parseFloat( ( ( stats.rl.dropped / ( stats.rl.total || 1 ) ) * 100 ).toFixed( 2 ) ),
            NonRL: parseFloat( ( ( stats.nonRl.dropped / ( stats.nonRl.total || 1 ) ) * 100 ).toFixed( 2 ) )
        }
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

/** Main processor function to be exported */
export const processAllData = ( packets: Packet[] ): ChartData => {
    if ( !packets || packets.length === 0 ) {
        const emptySankey = { nodes: [], links: [] };
        return {
            kpiData: [],
            correlationData: { rl: [], nonRl: [] },
            latencyDist: [],
            sankeyDataRL: emptySankey,
            sankeyDataNonRL: emptySankey
        };
    }

    const aggregatedData = aggregatePacketData( packets );
    const chartData = formatChartData( aggregatedData );
    return chartData;
};