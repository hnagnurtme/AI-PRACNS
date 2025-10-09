package com.sagin.routing;

import com.sagin.model.*;
import java.util.*;

public class DijkstraRoutingEngine implements IRoutingEngine {

    /**
     * @inheritdoc
     */
    @Override
    public RoutingTable computeRoutes(
        NodeInfo sourceNode, 
        Map<String, LinkMetric> allActiveLinks,
        Map<String, NodeInfo> allNodeInfos,
        ServiceQoS targetQoS) {

        // 1. Chuẩn bị Dữ liệu và Thuật toán (Dijkstra)
        RoutingTable newRoutingTable = new RoutingTable();
        // PriorityQueue để lưu trữ các node cần thăm, ưu tiên node có chi phí thấp nhất
        PriorityQueue<RouteCalculationState> queue = new PriorityQueue<>();
        // Map lưu trữ chi phí thấp nhất tìm thấy cho mỗi node
        Map<String, Double> minCostToNode = new HashMap<>();
        // Map lưu trữ tuyến đường tốt nhất đến mỗi node
        Map<String, RouteInfo> bestRouteToNode = new HashMap<>();

        // 2. Khởi tạo trạng thái ban đầu
        String sourceId = sourceNode.getNodeId();
        minCostToNode.put(sourceId, 0.0);
        
        // Khởi tạo trạng thái (Chi phí 0.0, Latency 0.0, BW vô hạn)
        RouteCalculationState initialState = new RouteCalculationState(
            sourceId, 0.0, 0.0, Double.MAX_VALUE, 0.0, new ArrayList<>()
        );
        queue.add(initialState);

        // 3. Vòng lặp Thuật toán Dijkstra
        while (!queue.isEmpty()) {
            RouteCalculationState currentState = queue.poll();
            String currentId = currentState.nodeId;

            // Nếu chi phí hiện tại lớn hơn chi phí tốt nhất đã tìm thấy, bỏ qua (đã tìm thấy tuyến đường tốt hơn)
            if (currentState.cost > minCostToNode.getOrDefault(currentId, Double.MAX_VALUE)) {
                continue;
            }

            // Lấy các liên kết (LinkMetric) xuất phát từ currentId
            for (LinkMetric link : allActiveLinks.values()) {
                if (link.getSourceNodeId().equals(currentId) && link.isLinkActive()) {
                    
                    // Tính toán chi phí của liên kết này
                    double linkCost = calculateLinkCost(link);
                    
                    // Chi phí TÍCH LŨY
                    double newAccumulatedCost = currentState.cost + linkCost;

                    // Nếu Chi phí Tích lũy mới TỐT HƠN chi phí đã biết
                    if (newAccumulatedCost < minCostToNode.getOrDefault(link.getDestinationNodeId(), Double.MAX_VALUE)) {
                        
                        // Cập nhật các Metrics Tích lũy
                        double newTotalLatency = currentState.totalLatencyMs + link.getLatencyMs();
                        double newMinBandwidth = Math.min(currentState.minBandwidthMbps, link.getCurrentAvailableBandwidthMbps());
                        // Giả định: Tỷ lệ mất gói trung bình đơn giản
                        double newAvgLossRate = (currentState.avgLossRate * currentState.path.size() + link.getPacketLossRate()) / (currentState.path.size() + 1);

                        // Cập nhật trạng thái
                        minCostToNode.put(link.getDestinationNodeId(), newAccumulatedCost);
                        
                        // Tạo đường đi mới
                        List<String> newPath = new ArrayList<>(currentState.path);
                        newPath.add(link.getDestinationNodeId());

                        RouteCalculationState newState = new RouteCalculationState(
                            link.getDestinationNodeId(), 
                            newAccumulatedCost, 
                            newTotalLatency, 
                            newMinBandwidth,
                            newAvgLossRate,
                            newPath
                        );
                        queue.add(newState);

                        // Lưu trữ RouteInfo tốt nhất (cho đích đến hiện tại)
                        RouteInfo routeInfo = new RouteInfo(
                            // Next Hop là node thứ hai trong đường đi
                            newPath.size() > 1 ? newPath.get(1) : newPath.get(0), 
                            newPath, 
                            newAccumulatedCost, 
                            newTotalLatency, 
                            newMinBandwidth, 
                            newAvgLossRate,
                            System.currentTimeMillis()
                        );
                        bestRouteToNode.put(link.getDestinationNodeId(), routeInfo);
                    }
                }
            }
        }
        
        // 4. Hoàn thiện RoutingTable
        for (String destId : bestRouteToNode.keySet()) {
            newRoutingTable.updateSingleRoute(destId, bestRouteToNode.get(destId));
        }

        return newRoutingTable;
    }

    /**
     * Hàm tính Chi phí (Cost) của một liên kết dựa trên LinkMetric.
     * Chi phí = 1 / LinkScore.
     */
    private double calculateLinkCost(LinkMetric link) {
        // Tránh chia cho 0, Link Score > 0.001
        double score = Math.max(0.001, link.calculateLinkScore()); 
        return 1.0 / score;
    }

    /**
     * @inheritdoc
     * Triển khai đơn giản: Gọi lại computeRoutes và lấy RouteInfo từ kết quả.
     */
    @Override
    public RouteInfo findSingleRoute(
        NodeInfo sourceNode, 
        String destinationNodeId,
        Map<String, LinkMetric> allActiveLinks,
        Map<String, NodeInfo> allNodeInfos,
        ServiceQoS targetQoS) {
        
        // Tính toán toàn bộ bảng định tuyến
        RoutingTable table = computeRoutes(sourceNode, allActiveLinks, allNodeInfos, targetQoS);
        
        // Trả về RouteInfo cho đích đến cụ thể
        return table.getRouteInfo(destinationNodeId);
    }

    // --- Lớp nội bộ để lưu trữ trạng thái trong quá trình tính toán Dijkstra ---
    private static class RouteCalculationState implements Comparable<RouteCalculationState> {
        String nodeId;
        double cost;
        double totalLatencyMs;
        double minBandwidthMbps;
        double avgLossRate;
        List<String> path;

        public RouteCalculationState(String nodeId, double cost, double totalLatencyMs, double minBandwidthMbps, double avgLossRate, List<String> path) {
            this.nodeId = nodeId;
            this.cost = cost;
            this.totalLatencyMs = totalLatencyMs;
            this.minBandwidthMbps = minBandwidthMbps;
            this.avgLossRate = avgLossRate;
            this.path = path;
        }

        @Override
        public int compareTo(RouteCalculationState other) {
            return Double.compare(this.cost, other.cost);
        }
    }
}