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
     * @param nextNode Node nhận
     * @param actualDelayMs Delay THỰC TẾ vừa tính (queuing + processing + transmission + propagation)
     * @param routeInfo Thông tin routing để lấy metric
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
        
        // --- Tạo hop record với delay THỰC TẾ ---
        HopRecord hop = new HopRecord(
                currentNode.getNodeId(),
                nextNode.getNodeId(),
                actualDelayMs,  // ✅ Delay thực tế đã được tính toán chính xác
                System.currentTimeMillis(),
                currentNode.getPosition(),
                nextNode.getPosition(),
                calculateDistanceKm(currentNode, nextNode),
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
}
