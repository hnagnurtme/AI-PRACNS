package com.sagin.satellite.model;

import lombok.*;
import java.util.*;

/**
 * RoutingTable lưu thông tin đường đi tốt nhất từ source → destination
 * Kết hợp với LinkMetric để chọn nextHop
 */
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class RoutingTable {

    // destinationNodeId -> nextHopNodeId
    private Map<String, String> table = new HashMap<>();

    // destinationNodeId -> toàn bộ path NodeId
    private Map<String, List<String>> pathHistory = new HashMap<>();

    /**
     * Cập nhật tuyến đường từ source → destination
     */
    public void updateRoute(String destinationNodeId, String nextHopNodeId, List<String> path) {
        table.put(destinationNodeId, nextHopNodeId);
        pathHistory.put(destinationNodeId, path != null ? new ArrayList<>(path) : new ArrayList<>());
    }

    /**
     * Lấy next hop dựa trên destination
     */
    public String getNextHop(String destinationNodeId) {
        return table.get(destinationNodeId);
    }

    /**
     * Lấy toàn bộ path history đến destination
     */
    public List<String> getPath(String destinationNodeId) {
        return pathHistory.getOrDefault(destinationNodeId, new ArrayList<>());
    }
}
