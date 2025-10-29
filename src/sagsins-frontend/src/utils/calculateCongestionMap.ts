// src/utils/calculateCongestionMap.ts

import type { NetworkBatch, NodeCongestion, Packet, HopRecord } from "../types/ComparisonTypes";

// Giả định types NodeCongestion đã được định nghĩa trong file bạn cung cấp

interface NodeAccumulator {
    nodeId: string;
    totalPackets: number;
    sumQueueSize: number;
    sumBandwidthUtil: number;
    sumLatency: number;
    hopCount: number; // Tổng số hop qua Node
    dijkstraCount: number;
    rlCount: number;
}

const initializeNodeAccumulator = (nodeId: string): NodeAccumulator => ({
    nodeId,
    totalPackets: 0,
    sumQueueSize: 0,
    sumBandwidthUtil: 0,
    sumLatency: 0,
    hopCount: 0,
    dijkstraCount: 0,
    rlCount: 0,
});

const accumulateHopData = (nodeId: string, hop: HopRecord, accumulator: Map<string, NodeAccumulator>) => {
    if (!accumulator.has(nodeId)) {
        accumulator.set(nodeId, initializeNodeAccumulator(nodeId));
    }
    const node = accumulator.get(nodeId)!;

    // Node nguồn (fromNodeId) là nơi BufferState được ghi lại
    if (nodeId === hop.fromNodeId) {
        node.sumQueueSize += hop.fromNodeBufferState.queueSize;
        node.sumBandwidthUtil += hop.fromNodeBufferState.bandwidthUtilization;
        node.hopCount += 1;
    } 
    
    // Tổng số gói tin đi qua (traffic)
    node.totalPackets += 1;
    
    // Tích lũy độ trễ (cho mỗi lần node là fromNode)
    if (nodeId === hop.fromNodeId) {
        node.sumLatency += hop.latencyMs; 
    }
};

const processPacket = (packet: Packet, accumulator: Map<string, NodeAccumulator>) => {
    const isRL = packet.useRL;

    for (const hop of packet.hopRecords) {
        // Tích lũy cho Node nguồn (fromNode)
        accumulateHopData(hop.fromNodeId, hop, accumulator);
        if (isRL) {
            accumulator.get(hop.fromNodeId)!.rlCount += 1;
        } else {
            accumulator.get(hop.fromNodeId)!.dijkstraCount += 1;
        }

        // Tích lũy cho Node đích (toNode). Gói tin cũng "đi qua" node đích.
        accumulateHopData(hop.toNodeId, hop, accumulator);
        if (isRL) {
            accumulator.get(hop.toNodeId)!.rlCount += 1;
        } else {
            accumulator.get(hop.toNodeId)!.dijkstraCount += 1;
        }
    }
};

export const calculateCongestionMap = (batches: NetworkBatch[]): NodeCongestion[] => {
    const accumulator = new Map<string, NodeAccumulator>();

    for (const batch of batches) {
        for (const comparison of batch.packets) {
            processPacket(comparison.dijkstraPacket, accumulator);
            processPacket(comparison.rlpacket, accumulator);
        }
    }

    // Chuyển đổi từ Accumulator Map sang NodeCongestion[]
    return Array.from(accumulator.values()).map(acc => {
        const avgQueueSize = acc.hopCount > 0 ? acc.sumQueueSize / acc.hopCount : 0;
        const avgBandwidthUtil = acc.hopCount > 0 ? acc.sumBandwidthUtil / acc.hopCount : 0;
        const avgLatency = acc.hopCount > 0 ? acc.sumLatency / acc.hopCount : 0; // Độ trễ trung bình của các hop xuất phát từ node này

        return {
            nodeId: acc.nodeId,
            packetsThrough: [], // Giả định không cần chi tiết ID gói tin ở đây
            totalPackets: acc.totalPackets,
            avgQueueSize: avgQueueSize,
            avgBandwidthUtil: avgBandwidthUtil,
            avgLatency: avgLatency,
            algorithms: { dijkstra: acc.dijkstraCount, rl: acc.rlCount },
        };
    });
};