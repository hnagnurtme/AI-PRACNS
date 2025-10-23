// package com.sagin.helper;

// import com.sagin.model.HopRecord;
// import com.sagin.model.NodeInfo;
// import com.sagin.model.Packet;
// import com.sagin.routing.RouteInfo;

// import java.util.HashMap;
// import java.util.Map;

// public class PacketHelper {

//     /**
//      * Cập nhật packet khi đi qua node.
//      * - Giảm TTL, drop nếu TTL <= 0
//      * - Cập nhật pathHistory
//      * - Tạo hopRecord mới
//      * - Cập nhật accumulatedDelay dựa trên node info và link latency
//      * - Tính toán packet loss dựa trên node + link
//      * - Kiểm tra QoS (maxAcceptableLatencyMs, maxAcceptableLossRate)
//      */
//     public static void updatePacketForTransit(Packet packet, NodeInfo currentNode, NodeInfo nextNode, RouteInfo routeInfo) {

//         // --- Giảm TTL ---
//         packet.setTTL(packet.getTTL() - 1);
//         if (packet.getTTL() <= 0) {
//             packet.setDropped(true);
//             packet.setDropReason("TTL_EXPIRED");
//             return;
//         }

//         // --- Cập nhật pathHistory ---
//         if (packet.getPathHistory() != null) {
//             packet.getPathHistory().add(nextNode.getNodeId());
//         }

//         // --- Tính latency dựa vào node + route ---
//         double linkLatency = calculateLinkLatency(currentNode, nextNode, routeInfo);
//         packet.setAccumulatedDelayMs(packet.getAccumulatedDelayMs() + linkLatency);

//         // --- Tính toán packet loss chính xác ---
//         double nodeLoss = currentNode.getPacketLossRate();
//         double linkLoss = routeInfo.getAvgPacketLossRate();
//         double combinedLossRate = 1 - (1 - nodeLoss) * (1 - linkLoss);

//         if (combinedLossRate > packet.getMaxAcceptableLossRate()) {
//             packet.setDropped(true);
//             packet.setDropReason("LOSS_RATE_EXCEEDED");
//             return;
//         }

//         // --- Tạo hop record ---
//         HopRecord hop = new HopRecord(
//                 currentNode.getNodeId(),
//                 nextNode.getNodeId(),
//                 linkLatency,
//                 System.currentTimeMillis(),
//                 currentNode.getPosition(),
//                 nextNode.getPosition(),
//                 calculateDistanceKm(currentNode, nextNode),
//                 getBufferState(currentNode),
//                 new HashMap<>() {{
//                     put("totalCost", routeInfo.getTotalCost());
//                     put("avgPacketLoss", routeInfo.getAvgPacketLossRate());
//                     put("hopCount", routeInfo.getHopCount());
//                     put("linkLatency", linkLatency);
//                     put("nodePacketLossRate", currentNode.getPacketLossRate());
//                     put("combinedLossRate", combinedLossRate);
//                 }}
//         );

//         if (packet.getHopRecords() != null) {
//             packet.getHopRecords().add(hop);
//         }

//         // --- Kiểm tra QoS: Latency ---
//         if (packet.getAccumulatedDelayMs() > packet.getMaxAcceptableLatencyMs()) {
//             packet.setDropped(true);
//             packet.setDropReason("LATENCY_EXCEEDED");
//         }
//     }

//     private static double calculateLinkLatency(NodeInfo from, NodeInfo to, RouteInfo route) {
//         return route.getTotalLatencyMs() / Math.max(1, route.getHopCount());
//     }

//     private static double calculateDistanceKm(NodeInfo from, NodeInfo to) {
//         return haversine(from.getPosition(), to.getPosition());
//     }

//     private static double haversine(com.sagin.model.Position p1, com.sagin.model.Position p2) {
//         double R = 6371; // Earth radius km
//         double dLat = Math.toRadians(p2.getLatitude() - p1.getLatitude());
//         double dLon = Math.toRadians(p2.getLongitude() - p1.getLongitude());
//         double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
//                 Math.cos(Math.toRadians(p1.getLatitude())) * Math.cos(Math.toRadians(p2.getLatitude())) *
//                         Math.sin(dLon / 2) * Math.sin(dLon / 2);
//         double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
//         return R * c;
//     }

//     private static Map<String, Object> getBufferState(NodeInfo node) {
//         Map<String, Object> bufferState = new HashMap<>();
//         bufferState.put("currentPacketCount", node.getCurrentPacketCount());
//         bufferState.put("packetBufferCapacity", node.getPacketBufferCapacity());
//         bufferState.put("resourceUtilization", node.getResourceUtilization());
//         return bufferState;
//     }
// }
package com.sagin.helper;

import com.sagin.model.*;
import com.sagin.routing.RouteInfo;

import java.util.*;

public class PacketHelper {

    private static final double SPEED_OF_SIGNAL_KM_PER_MS = 200_000.0 / 1000.0; // km/ms ~ 2e5 km/s
    private static final double WEATHER_LOSS_FACTOR = 0.001; // mỗi dB attenuation thêm 0.1% loss
    private static final double WEATHER_DELAY_FACTOR = 0.05; // mỗi dB attenuation thêm 5% delay

    public static void updatePacketForTransit(Packet packet, NodeInfo currentNode, NodeInfo nextNode, RouteInfo routeInfo) {
        if (packet == null || currentNode == null || nextNode == null) return;

        // --- TTL ---
        packet.setTTL(packet.getTTL() - 1);
        if (packet.getTTL() <= 0) {
            drop(packet, "TTL_EXPIRED", currentNode);
            return;
        }

        // --- Chuẩn bị dữ liệu ---
        if (packet.getPathHistory() == null)
            packet.setPathHistory(new ArrayList<>());
        if (packet.getHopRecords() == null)
            packet.setHopRecords(new ArrayList<>());

        packet.getPathHistory().add(nextNode.getNodeId());

        // --- 1️⃣ Propagation delay ---
        double distanceKm = calculateDistanceKm(currentNode.getPosition(), nextNode.getPosition());
        double propagationDelay = distanceKm / SPEED_OF_SIGNAL_KM_PER_MS;

        // --- 2️⃣ Transmission delay ---
        double transmissionDelay = calculateTransmissionDelay(currentNode.getCommunication(), packet.getPayloadSizeByte());

        // --- 3️⃣ Processing delay ---
        double processingDelay = calculateProcessingDelay(currentNode);

        // --- 4️⃣ Weather impact ---
        double weatherAttenuation = currentNode.getWeather().getTypicalAttenuationDb();
        double weatherDelay = (1 + WEATHER_DELAY_FACTOR * weatherAttenuation);
        double weatherLoss = WEATHER_LOSS_FACTOR * weatherAttenuation;

        // --- 5️⃣ Queuing delay ---
        double queueDelay = 0.0;
        double bufferRatio = (double) currentNode.getCurrentPacketCount() / currentNode.getPacketBufferCapacity();
        if (bufferRatio > 0.8) {
            queueDelay = (bufferRatio - 0.8) * 20.0; // ms thêm do chờ đợi buffer
        }

        // --- 6️⃣ Tổng delay ---
        double totalDelay = (propagationDelay + transmissionDelay + processingDelay + queueDelay)
                * weatherDelay;

        try {
            Thread.sleep((long) totalDelay);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        packet.setAccumulatedDelayMs(packet.getAccumulatedDelayMs() + totalDelay);

        // --- 7️⃣ Packet loss ---
        double nodeLoss = currentNode.getPacketLossRate();
        double linkLoss = routeInfo.getAvgPacketLossRate();
        double combinedLoss = 1 - (1 - nodeLoss) * (1 - linkLoss);
        combinedLoss += weatherLoss; // thêm tổn thất thời tiết

        if (Math.random() < combinedLoss) {
            drop(packet, "PACKET_LOST", currentNode);
            return;
        }

        if (combinedLoss > packet.getMaxAcceptableLossRate()) {
            drop(packet, "LOSS_RATE_EXCEEDED", currentNode);
            return;
        }

        // --- 8️⃣ Ghi lại hop record ---
        HopRecord hop = new HopRecord(
                currentNode.getNodeId(),
                nextNode.getNodeId(),
                totalDelay,
                System.currentTimeMillis(),
                currentNode.getPosition(),
                nextNode.getPosition(),
                distanceKm,
                currentNode.getBufferState(),
                Map.of(
                        "propagationDelay", propagationDelay,
                        "transmissionDelay", transmissionDelay,
                        "processingDelay", processingDelay,
                        "queueDelay", queueDelay,
                        "weatherAttenuation", weatherAttenuation,
                        "combinedLossRate", combinedLoss
                )
        );

        packet.getHopRecords().add(hop);

        // --- 9️⃣ QoS check ---
        if (packet.getAccumulatedDelayMs() > packet.getMaxAcceptableLatencyMs()) {
            drop(packet, "LATENCY_EXCEEDED", currentNode);
        }
    }

    private static void drop(Packet packet, String reason, NodeInfo node) {
        packet.setDropped(true);
        packet.setDropReason(reason);
        System.out.printf("[DROP] Packet %s at Node %s (%s): %s%n",
                packet.getPacketId() , node.getNodeId(), node.getNodeType(), reason);
    }

    private static double calculateDistanceKm(Position p1, Position p2) {
        double R = 6371;
        double dLat = Math.toRadians(p2.getLatitude() - p1.getLatitude());
        double dLon = Math.toRadians(p2.getLongitude() - p1.getLongitude());
        double a = Math.sin(dLat / 2) * Math.sin(dLat / 2)
                + Math.cos(Math.toRadians(p1.getLatitude())) * Math.cos(Math.toRadians(p2.getLatitude()))
                * Math.sin(dLon / 2) * Math.sin(dLon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    private static double calculateTransmissionDelay(Communication comm, int packetSizeBytes) {
        if (comm == null || comm.bandwidthMHz() <= 0) return 0.0;
        double bandwidthBps = comm.bandwidthMHz() * 1_000_000.0;
        return (packetSizeBytes * 8.0) / bandwidthBps * 1000.0; // ms
    }

    private static double calculateProcessingDelay(NodeInfo node) {
        double base = node.getNodeProcessingDelayMs();
        double loadFactor = 1 + (node.getResourceUtilization() * 0.5);
        return base * loadFactor;
    }
}
