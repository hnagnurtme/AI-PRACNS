package com.sagin.model;

import lombok.*;
import java.util.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class RoutingTable {
    
    // Ánh xạ: destinationNodeId -> RouteInfo (chứa nextHop, path, và chi phí)
    private Map<String, RouteInfo> routeInfoMap = new HashMap<>(); 

    /**
     * Cập nhật toàn bộ bảng định tuyến bằng Map mới.
     */
    public void updateRoute(Map<String, RouteInfo> newRouteInfoMap) {
        if (newRouteInfoMap != null) {
            // Sử dụng bản sao sâu
            this.routeInfoMap = new HashMap<>(newRouteInfoMap);
        } else {
            this.routeInfoMap = new HashMap<>();
        }
    }
    
    /**
     * Cập nhật một tuyến đường cụ thể.
     */
    public void updateSingleRoute(String destinationNodeId, RouteInfo routeInfo) {
        routeInfoMap.put(destinationNodeId, routeInfo);
    }

    /**
     * Lấy next hop dựa trên destination.
     */
    public String getNextHop(String destinationNodeId) {
        RouteInfo info = routeInfoMap.get(destinationNodeId);
        return info != null ? info.getNextHopNodeId() : null;
    }

    /**
     * Lấy thông tin chi tiết về tuyến đường (bao gồm chi phí).
     */
    public RouteInfo getRouteInfo(String destinationNodeId) {
        return routeInfoMap.get(destinationNodeId);
    }
}