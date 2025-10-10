package com.sagin.routing;

import com.sagin.model.*;
import com.sagin.util.GeoUtils;
import java.util.*;

public class DijkstraRoutingEngine implements IRoutingEngine {

    /**
     * @inheritdoc
     * Dijkstra “không trọng số”: mỗi liên kết được coi là chi phí = 1.
     */
    @Override
    public RoutingTable computeRoutes(
            NodeInfo sourceNode,
            Map<String, LinkMetric> allActiveLinks,
            Map<String, NodeInfo> allNodeInfos,
            ServiceQoS targetQoS) {

        RoutingTable newRoutingTable = new RoutingTable();
        PriorityQueue<RouteCalculationState> queue = new PriorityQueue<>();
        Map<String, Integer> minHopsToNode = new HashMap<>();
        Map<String, RouteInfo> bestRouteToNode = new HashMap<>();

        String sourceId = sourceNode.getNodeId();
        minHopsToNode.put(sourceId, 0);
        queue.add(new RouteCalculationState(sourceId, 0, new ArrayList<>()));

        while (!queue.isEmpty()) {
            RouteCalculationState currentState = queue.poll();
            String currentId = currentState.nodeId;

            if (currentState.hops > minHopsToNode.getOrDefault(currentId, Integer.MAX_VALUE)) {
                continue;
            }

            NodeInfo currentNode = allNodeInfos.get(currentId);
            if (currentNode == null) continue;

            for (LinkMetric link : allActiveLinks.values()) {
                if (!link.getSourceNodeId().equals(currentId) || !link.isLinkActive()) continue;

                NodeInfo nextNode = allNodeInfos.get(link.getDestinationNodeId());
                if (nextNode == null) continue;

                // --- KIỂM TRA VISIBILITY ---
                if (!GeoUtils.checkVisibility(currentNode, nextNode)) continue;

                int newHops = currentState.hops + 1;

                if (newHops < minHopsToNode.getOrDefault(nextNode.getNodeId(), Integer.MAX_VALUE)) {

                    minHopsToNode.put(nextNode.getNodeId(), newHops);

                    List<String> newPath = new ArrayList<>(currentState.path);
                    newPath.add(nextNode.getNodeId());

                    queue.add(new RouteCalculationState(nextNode.getNodeId(), newHops, newPath));

                    RouteInfo routeInfo = new RouteInfo(
                            newPath.size() > 1 ? newPath.get(1) : newPath.get(0),
                            newPath,
                            newHops,         // Sử dụng hops làm cost
                            0.0,             // Không quan tâm latency
                            Double.MAX_VALUE, // bandwidth tối đa
                            0.0,             // packet loss
                            System.currentTimeMillis()
                    );
                    bestRouteToNode.put(nextNode.getNodeId(), routeInfo);
                }
            }
        }

        for (String destId : bestRouteToNode.keySet()) {
            newRoutingTable.updateSingleRoute(destId, bestRouteToNode.get(destId));
        }

        return newRoutingTable;
    }

    // --- Lớp nội bộ lưu trạng thái Dijkstra ---
    private static class RouteCalculationState implements Comparable<RouteCalculationState> {
        String nodeId;
        int hops; // Số bước từ source
        List<String> path;

        public RouteCalculationState(String nodeId, int hops, List<String> path) {
            this.nodeId = nodeId;
            this.hops = hops;
            this.path = path;
        }

        @Override
        public int compareTo(RouteCalculationState other) {
            return Integer.compare(this.hops, other.hops);
        }
    }
}
