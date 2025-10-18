package com.sagin.routing;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Triển khai IRoutingService.
 * Lớp này giữ một map của tất cả các Bảng Định tuyến trong mạng,
 * mỗi bảng tương ứng với một node.
 */
public class RoutingService implements IRoutingService {

    /**
     * Key: NodeId (ví dụ: "LEO-1", "GND-2")
     * Value: Bảng định tuyến của node đó.
     */
    private final Map<String, RoutingTable> nodeRoutingTables = new ConcurrentHashMap<>();

    /**
     * Lấy hoặc tạo mới (nếu chưa có) bảng định tuyến cho một node.
     * @param nodeId ID của node.
     * @return Bảng định tuyến của node đó.
     */
    private RoutingTable getTableForNode(String nodeId) {
        return nodeRoutingTables.computeIfAbsent(nodeId, k -> new RoutingTable());
    }

    @Override
    public RouteInfo getBestRoute(String currentNodeId, String destinationNodeId) {
        if (currentNodeId == null || destinationNodeId == null) {
            return null;
        }
        
        RoutingTable table = getTableForNode(currentNodeId);
        return table.getBestRoute(destinationNodeId);
    }

    @Override
    public RoutingTable getRoutingTableForNode(String nodeId) {
        return getTableForNode(nodeId);
    }

    @Override
    public void updateRoute(String forNodeId, RouteInfo newRoute) {
        if (forNodeId == null || newRoute == null) {
            return;
        }
        
        RoutingTable table = getTableForNode(forNodeId);
        table.updateRoute(newRoute);
    }
}