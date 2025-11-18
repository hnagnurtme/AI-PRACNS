package com.sagin.routing;

import com.sagin.model.NodeInfo;
import com.sagin.service.INodeService;
import com.sagin.repository.INodeRepository;
import com.sagin.util.AppLogger;
import org.slf4j.Logger;

import java.util.*;
import java.util.concurrent.*;

/**
 * DynamicRoutingService tối ưu:
 * - Precompute routing table cho tất cả node.
 * - Scheduler update liên tục.
 * - Packet chỉ lookup bảng định tuyến đã có.
 */
public class DynamicRoutingService implements IRoutingService {

    private static final Logger logger = AppLogger.getLogger(DynamicRoutingService.class);

    private final INodeRepository nodeRepository;
    private final INodeService nodeService;

    // Map nodeId -> RoutingTable
    private final Map<String, RoutingTable> routingTables = new ConcurrentHashMap<>();

    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

    public DynamicRoutingService(INodeRepository nodeRepository, INodeService nodeService) {
        this.nodeRepository = nodeRepository;
        this.nodeService = nodeService;

        scheduler.scheduleAtFixedRate(this::updateAllRoutingTables, 0, 5, TimeUnit.SECONDS);
    }

    @Override
    public RouteInfo getBestRoute(String currentNodeId, String destinationNodeId) {
        if (currentNodeId == null || destinationNodeId == null) return null;
        RoutingTable table = routingTables.get(currentNodeId);
        return table != null ? table.getBestRoute(destinationNodeId) : null;
    }

    @Override
    public RoutingTable getRoutingTableForNode(String nodeId) {
        return routingTables.getOrDefault(nodeId, new RoutingTable());
    }

    @Override
    public void updateRoute(String forNodeId, RouteInfo newRoute) {
        routingTables.computeIfAbsent(forNodeId, k -> new RoutingTable())
                        .updateRoute(newRoute);
    }

    /**
     * Scheduler: tính toàn bộ routing table cho tất cả node
     */
    private void updateAllRoutingTables() {
        logger.debug("Updating routing tables for all nodes...");
        List<NodeInfo> allNodes = new ArrayList<>(nodeRepository.loadAllNodeConfigs().values());

        // 1 adjacency list chung cho tất cả nodes
        Map<String, List<String>> graph = buildAdjacencyList(allNodes);

        for (NodeInfo srcNode : allNodes) {
            if (!srcNode.isOperational()) {
                continue;
            }
            RoutingTable table = computeRoutingTable(srcNode.getNodeId(), graph);
            routingTables.put(srcNode.getNodeId(), table);
        }
        logger.debug("Routing tables updated for {} nodes", routingTables.size());
    }

    private Map<String, List<String>> buildAdjacencyList(List<NodeInfo> nodes) {
        Map<String, List<String>> graph = new HashMap<>();
        for (NodeInfo node : nodes) {
            if (!node.isOperational()) continue;
            List<String> visibleNeighbors = nodeService.getVisibleNodes(node, nodes).stream()
                    .filter(NodeInfo::getHealthy)
                    .map(NodeInfo::getNodeId)
                    .toList();
            graph.put(node.getNodeId(), visibleNeighbors);
        }
        return graph;
    }

    private RoutingTable computeRoutingTable(String sourceNodeId, Map<String, List<String>> graph) {
        RoutingTable table = new RoutingTable();

        // ✅ DIJKSTRA ALGORITHM with actual edge weights (distance/latency-based)
        // Maps: nodeId -> shortest cost from source
        Map<String, Double> costFromSource = new HashMap<>();
        // Maps: nodeId -> previous node in shortest path
        Map<String, String> previousNode = new HashMap<>();
        // Priority queue: (cost, nodeId)
        PriorityQueue<NodeCost> pq = new PriorityQueue<>(Comparator.comparingDouble(nc -> nc.cost));

        // Initialize distances
        costFromSource.put(sourceNodeId, 0.0);
        pq.offer(new NodeCost(sourceNodeId, 0.0));

        // Dijkstra's main loop
        while (!pq.isEmpty()) {
            NodeCost current = pq.poll();
            String currentNodeId = current.nodeId;
            double currentCost = current.cost;

            // Skip if we've already processed this node with a better cost
            if (currentCost > costFromSource.getOrDefault(currentNodeId, Double.MAX_VALUE)) {
                continue;
            }

            // Explore neighbors
            for (String neighborId : graph.getOrDefault(currentNodeId, List.of())) {
                // Calculate edge cost (distance-based + other factors)
                double edgeCost = calculateEdgeCost(currentNodeId, neighborId);
                double newCost = currentCost + edgeCost;

                // If this path is better, update it
                if (newCost < costFromSource.getOrDefault(neighborId, Double.MAX_VALUE)) {
                    costFromSource.put(neighborId, newCost);
                    previousNode.put(neighborId, currentNodeId);
                    pq.offer(new NodeCost(neighborId, newCost));
                }
            }
        }

        // Reconstruct paths for all reachable destinations
        for (String destNodeId : costFromSource.keySet()) {
            if (destNodeId.equals(sourceNodeId)) continue;

            List<String> path = reconstructPath(sourceNodeId, destNodeId, previousNode);
            if (path != null && !path.isEmpty()) {
                RouteInfo route = RouteHelper.createRouteWithCost(
                    sourceNodeId,
                    destNodeId,
                    path,
                    costFromSource.get(destNodeId),
                    nodeRepository
                );
                table.updateRoute(route);
            }
        }

        return table;
    }

    /**
     * Calculates the cost of an edge between two nodes.
     * Cost is based on distance (propagation delay) and other factors.
     */
    private double calculateEdgeCost(String fromNodeId, String toNodeId) {
        Optional<NodeInfo> fromNodeOpt = nodeRepository.getNodeInfo(fromNodeId);
        Optional<NodeInfo> toNodeOpt = nodeRepository.getNodeInfo(toNodeId);

        if (fromNodeOpt.isEmpty() || toNodeOpt.isEmpty()) {
            return Double.MAX_VALUE; // Unreachable
        }

        NodeInfo fromNode = fromNodeOpt.get();
        NodeInfo toNode = toNodeOpt.get();

        // Calculate distance using Haversine formula
        double distanceKm = calculateDistance(fromNode, toNode);

        // Propagation delay (distance / speed of light)
        // Speed of light in fiber/satellite ~= 200,000 km/s = 200 km/ms
        double propagationDelayMs = distanceKm / 200.0;

        // Bandwidth factor (lower bandwidth = higher cost)
        double bandwidthMHz = fromNode.getCommunication().getBandwidthMHz();
        double bandwidthFactor = (bandwidthMHz > 0) ? (1000.0 / bandwidthMHz) : 10.0;

        // Weather impact (if available)
        double weatherFactor = 1.0;
        if (fromNode.getWeather() != null) {
            weatherFactor = 1.0 + (fromNode.getWeather().getTypicalAttenuationDb() / 100.0);
        }

        // Total cost: propagation delay + bandwidth penalty + weather penalty
        double totalCost = propagationDelayMs + bandwidthFactor + (weatherFactor * 0.5);

        return totalCost;
    }

    /**
     * Calculates distance between two nodes using Haversine formula.
     */
    private double calculateDistance(NodeInfo from, NodeInfo to) {
        double R = 6371; // Earth radius in km
        double dLat = Math.toRadians(to.getPosition().getLatitude() - from.getPosition().getLatitude());
        double dLon = Math.toRadians(to.getPosition().getLongitude() - from.getPosition().getLongitude());
        double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(Math.toRadians(from.getPosition().getLatitude())) *
                        Math.cos(Math.toRadians(to.getPosition().getLatitude())) *
                        Math.sin(dLon / 2) * Math.sin(dLon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    /**
     * Reconstructs the path from source to destination using the previousNode map.
     */
    private List<String> reconstructPath(String source, String dest, Map<String, String> previousNode) {
        List<String> path = new ArrayList<>();
        String current = dest;

        while (current != null) {
            path.add(0, current); // Add to front
            if (current.equals(source)) break;
            current = previousNode.get(current);
        }

        // Verify path is valid
        if (path.isEmpty() || !path.get(0).equals(source)) {
            return null;
        }

        return path;
    }

    /**
     * Helper record for Dijkstra's priority queue.
     */
    private record NodeCost(String nodeId, double cost) {}
    public void forceUpdateRoutingTables() {
        logger.info("Force updating routing tables...");
        updateAllRoutingTables();
    }


    public void shutdown() {
        logger.info("Shutting down DynamicRoutingService scheduler...");
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
        logger.info("DynamicRoutingService shutdown complete");
        scheduler.shutdownNow();
    }
}
