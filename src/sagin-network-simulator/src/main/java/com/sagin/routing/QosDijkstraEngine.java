package com.sagin.routing;

import com.sagin.model.LinkMetric;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.model.RoutingTable;
import java.util.Map;
import java.util.ArrayList;
import java.util.List;

// SLF4J Imports
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Triển khai RoutingEngine sử dụng thuật toán Dijkstra,
 * với chi phí (cost) là giá trị nghịch đảo của LinkMetric.linkScore (1/score).
 */
public class QosDijkstraEngine implements RoutingEngine {
    
    // Khai báo Logger
    private static final Logger logger = LoggerFactory.getLogger(QosDijkstraEngine.class);

    @Override
    public RoutingTable computeRoutes(NodeInfo currentNode, Map<String, LinkMetric> neighborMetrics) {
        // SỬ DỤNG LOGGER
        logger.info("Node {} đang chạy thuật toán QoS-Dijkstra...", currentNode.getNodeId());
        
        // 1. Khởi tạo Bảng Định tuyến
        RoutingTable newRoutingTable = new RoutingTable();
        
        // --- LOGIC MÔ PHỎNG DIJKSTRA ---
        String FINAL_DESTINATION = "USER_TERMINAL_02"; 
        String bestNextHopId = null;
        double highestScore = -1.0;

        for (Map.Entry<String, LinkMetric> entry : neighborMetrics.entrySet()) {
            // Cần đảm bảo linkScore đã được tính toán trong LinkMetric
            if (entry.getValue().getLinkScore() > highestScore) {
                highestScore = entry.getValue().getLinkScore();
                bestNextHopId = entry.getKey();
            }
        }
        
        // Nếu tìm thấy next hop, cập nhật RoutingTable
        if (bestNextHopId != null) {
            List<String> mockPath = new ArrayList<>();
            mockPath.add(currentNode.getNodeId());
            mockPath.add(bestNextHopId);
            mockPath.add(FINAL_DESTINATION);
            
            // SỬA LỖI TẠI ĐÂY: Dùng updateSingleRoute thay vì updateRoute cũ
            newRoutingTable.updateSingleRoute(FINAL_DESTINATION, bestNextHopId, mockPath);
            
            // SỬ DỤNG LOGGER
            logger.info("Định tuyến cập nhật: Next Hop tới {} là {}", FINAL_DESTINATION, bestNextHopId);
        }

        return newRoutingTable;
    }

    @Override
    public String getNextHop(Packet packet, RoutingTable routingTable) {
        String destinationId = packet.getDestinationUserId();
        
        // Tra cứu trong bảng
        String nextHop = routingTable.getNextHop(destinationId);
        
        if (nextHop == null) {
             // SỬ DỤNG LOGGER.WARN
             logger.warn("Không tìm thấy đường đi cụ thể cho {} trong bảng.", destinationId);
        }
        
        return nextHop;
    }
}