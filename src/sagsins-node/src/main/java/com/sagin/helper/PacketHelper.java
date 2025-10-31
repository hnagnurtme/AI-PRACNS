package com.sagin.helper;

import com.sagin.model.HopRecord;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.routing.RouteInfo;

import java.util.HashMap;
import java.util.Map;

public class PacketHelper {

    /**
     * Cập nhật packet khi đi qua node.
     * - Giảm TTL, drop nếu TTL <= 0
     * - Cập nhật pathHistory
     * - Tạo hopRecord mới
     * - Cập nhật accumulatedDelay dựa trên node info và link latency
     * - Tính toán packet loss dựa trên node + link
     * - Kiểm tra QoS (maxAcceptableLatencyMs, maxAcceptableLossRate)
     */
    public static void updatePacketForTransit(Packet packet, NodeInfo currentNode, NodeInfo nextNode, RouteInfo routeInfo) {

        // --- Giảm TTL ---
        packet.setTTL(packet.getTTL() - 1);
        if (packet.getTTL() <= 0) {
            packet.setDropped(true);
            packet.setDropReason("TTL_EXPIRED");
            return;
        }

        // --- Cập nhật pathHistory ---
        if (packet.getPathHistory() != null) {
            packet.getPathHistory().add(nextNode.getNodeId());
        }

        // --- Tính latency dựa vào node + route ---
        double linkLatency = calculateLinkLatency(currentNode, nextNode, routeInfo);
        packet.setAccumulatedDelayMs(packet.getAccumulatedDelayMs() + linkLatency);

        // --- Tính toán packet loss chính xác ---
        double nodeLoss = currentNode.getPacketLossRate();
        double linkLoss = routeInfo.getAvgPacketLossRate();
        double combinedLossRate = 1 - (1 - nodeLoss) * (1 - linkLoss);

        if (combinedLossRate > packet.getMaxAcceptableLossRate()) {
            packet.setDropped(true);
            packet.setDropReason("LOSS_RATE_EXCEEDED");
            return;
        }

        // --- Tạo hop record ---
        HopRecord hop = new HopRecord(
                currentNode.getNodeId(),
                nextNode.getNodeId(),
                linkLatency,
                System.currentTimeMillis(),
                currentNode.getPosition(),
                nextNode.getPosition(),
                calculateDistanceKm(currentNode, nextNode),
                getBufferState(currentNode)
                , null
                
        );

        if (packet.getHopRecords() != null) {
            packet.getHopRecords().add(hop);
        }

        // --- Kiểm tra QoS: Latency ---
        if (packet.getAccumulatedDelayMs() > packet.getMaxAcceptableLatencyMs()) {
            packet.setDropped(true);
            packet.setDropReason("LATENCY_EXCEEDED");
        }
    }

    private static double calculateLinkLatency(NodeInfo from, NodeInfo to, RouteInfo route) {
        return route.getTotalLatencyMs() / Math.max(1, route.getHopCount());
    }

    private static double calculateDistanceKm(NodeInfo from, NodeInfo to) {
        return haversine(from.getPosition(), to.getPosition());
    }

    private static double haversine(com.sagin.model.Position p1, com.sagin.model.Position p2) {
        double R = 6371; // Earth radius km
        double dLat = Math.toRadians(p2.getLatitude() - p1.getLatitude());
        double dLon = Math.toRadians(p2.getLongitude() - p1.getLongitude());
        double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(Math.toRadians(p1.getLatitude())) * Math.cos(Math.toRadians(p2.getLatitude())) *
                        Math.sin(dLon / 2) * Math.sin(dLon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    private static Map<String, Object> getBufferState(NodeInfo node) {
        Map<String, Object> bufferState = new HashMap<>();
        bufferState.put("currentPacketCount", node.getCurrentPacketCount());
        bufferState.put("packetBufferCapacity", node.getPacketBufferCapacity());
        bufferState.put("resourceUtilization", node.getResourceUtilization());
        return bufferState;
    }
}
