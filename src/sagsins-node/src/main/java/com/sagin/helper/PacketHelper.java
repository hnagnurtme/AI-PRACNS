package com.sagin.helper;

import com.sagin.model.BufferState;
import com.sagin.model.HopRecord;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.model.RoutingDecisionInfo;
import com.sagin.model.RoutingDecisionInfo.Algorithm;
import com.sagin.routing.RouteInfo;


public class PacketHelper {

    /**
     * **BƯỚC 1 (TRƯỚC KHI GỬI)**: Cập nhật thông tin cơ bản của packet.
     * - Giảm TTL, drop nếu TTL <= 0
     * - Cập nhật pathHistory
     * 
     * ⚠️ KHÔNG TẠO HopRecord Ở ĐÂY! HopRecord sẽ được tạo SAU khi gửi thành công với delay THỰC TẾ.
     */
    public static void preparePacketForTransit(Packet packet, NodeInfo nextNode) {
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
    }

    /**
     * **BƯỚC 2 (SAU KHI GỬI THÀNH CÔNG)**: Tạo HopRecord với delay THỰC TẾ.
     * 
     * @param packet Packet đã gửi thành công
     * @param currentNode Node gửi
     * @param nextNode Node nhận (có thể null nếu đích là user)
     * @param actualDelayMs Delay THỰC TẾ vừa tính (queuing + processing + transmission + propagation)
     * @param routeInfo Thông tin routing để lấy metric (có thể null cho hop cuối đến user)
     */
    public static void createHopRecordWithActualDelay(
            Packet packet, 
            NodeInfo currentNode, 
            NodeInfo nextNode, 
            double actualDelayMs,
            RouteInfo routeInfo) {
        
        BufferState bufferState = new BufferState(
            currentNode.getPacketBufferCapacity(),
            currentNode.getCommunication().getBandwidthMHz()
        );

        RoutingDecisionInfo routingDecisionInfo = new RoutingDecisionInfo(
            packet.isUseRL() ? RoutingDecisionInfo.Algorithm.ReinforcementLearning : Algorithm.Dijkstra,
            "latency",
            actualDelayMs  // ✅ Sử dụng delay THỰC TẾ
        );
        
        // --- Xử lý nextNode (có thể null nếu gửi đến user) ---
        String nextNodeId;
        if (nextNode != null) {
            nextNodeId = nextNode.getNodeId();
        } else {
            // ✅ Kiểm tra destinationUserId trước khi dùng
            String destUserId = packet.getDestinationUserId();
            if (destUserId == null || destUserId.isBlank()) {
                throw new IllegalStateException(
                    "Cannot create HopRecord: destinationUserId is NULL for packet " + packet.getPacketId()
                );
            }
            nextNodeId = "USER:" + destUserId;
        }
        
        com.sagin.model.Position nextPosition = (nextNode != null) ? nextNode.getPosition() : null;
        
        // Tính khoảng cách (0 nếu gửi đến user vì không có position)
        double distanceKm = (nextNode != null) ? calculateDistanceKm(currentNode, nextNode) : 0.0;
        
        // --- Tạo hop record với delay THỰC TẾ ---
        HopRecord hop = new HopRecord(
                currentNode.getNodeId(),
                nextNodeId,  // ✅ Có thể là nodeId hoặc "USER:userId"
                actualDelayMs,  // ✅ Delay thực tế đã được tính toán chính xác
                System.currentTimeMillis(),
                currentNode.getPosition(),
                nextPosition,  // ✅ Có thể null nếu đích là user
                distanceKm,  // ✅ 0 nếu đích là user
                bufferState,
                routingDecisionInfo
        );

        if (packet.getHopRecords() != null) {
            packet.getHopRecords().add(hop);
        }
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

    /**
     * Tính toán AnalysisData từ danh sách HopRecords của packet.
     * Được gọi khi packet đến đích để phân tích hiệu suất tuyến đường.
     * 
     * @param packet Packet cần phân tích
     */
    public static void calculateAnalysisData(Packet packet) {
        if (packet == null || packet.getHopRecords() == null || packet.getHopRecords().isEmpty()) {
            // Không có hop records, không thể phân tích
            return;
        }

        var hopRecords = packet.getHopRecords();
        int hopCount = hopRecords.size();
        
        // Tính tổng distance và latency
        double totalDistanceKm = hopRecords.stream()
                .mapToDouble(com.sagin.model.HopRecord::distanceKm)
                .sum();
        
        double totalLatencyMs = hopRecords.stream()
                .mapToDouble(com.sagin.model.HopRecord::latencyMs)
                .sum();
        
        // Tính trung bình
        double avgLatency = totalLatencyMs / hopCount;
        double avgDistanceKm = totalDistanceKm / hopCount;
        
        // Route success rate: 1.0 nếu packet đến đích, 0.0 nếu bị drop
        double routeSuccessRate = packet.isDropped() ? 0.0 : 1.0;
        
        // ✅ Đồng bộ totalLatencyMs với accumulatedDelayMs
        // Đảm bảo tổng delay từ HopRecords khớp với delay tích lũy của packet
        if (Math.abs(totalLatencyMs - packet.getAccumulatedDelayMs()) > 0.01) {
            // Nếu có sai lệch, sử dụng giá trị từ packet (đã được cập nhật liên tục)
            totalLatencyMs = packet.getAccumulatedDelayMs();
            avgLatency = totalLatencyMs / hopCount;
        }
        
        // Tạo AnalysisData
        com.sagin.model.AnalysisData analysisData = new com.sagin.model.AnalysisData(
                avgLatency,
                avgDistanceKm,
                routeSuccessRate,
                totalDistanceKm,
                totalLatencyMs
        );
        
        packet.setAnalysisData(analysisData);
    }
}
