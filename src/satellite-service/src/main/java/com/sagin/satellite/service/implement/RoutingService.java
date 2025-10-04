package com.sagin.satellite.service.implement;

import com.sagin.satellite.model.NodeInfo;
import com.sagin.satellite.model.RoutingTable;
import com.sagin.satellite.model.LinkMetric;
import com.sagin.satellite.service.IRoutingService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * RoutingService triển khai thuật toán Dijkstra để tìm đường đi tối ưu
 * dựa trên link score và khoảng cách
 */
public class RoutingService implements IRoutingService {

    private static final Logger logger = LoggerFactory.getLogger(RoutingService.class);
    private RoutingTable currentRoutingTable;

    public RoutingService() {
        this.currentRoutingTable = new RoutingTable();
    }

    @Override
    public RoutingTable calculateRoutingTable(String currentNodeId, 
                                            List<NodeInfo> networkNodes, 
                                            Map<String, LinkMetric> linkMetrics) {
        logger.info("Calculating routing table for node: {}", currentNodeId);
        
        RoutingTable routingTable = new RoutingTable();
        
        // Tính toán shortest path tới tất cả nodes khác
        for (NodeInfo targetNode : networkNodes) {
            if (!targetNode.getNodeId().equals(currentNodeId)) {
                List<String> path = findOptimalPath(currentNodeId, targetNode.getNodeId(), 
                                                  networkNodes, linkMetrics);
                if (!path.isEmpty() && path.size() > 1) {
                    String nextHop = path.get(1); // Next hop là node thứ 2 trong path
                    routingTable.updateRoute(targetNode.getNodeId(), nextHop, path);
                }
            }
        }
        
        this.currentRoutingTable = routingTable;
        logger.info("Routing table calculated with {} routes", routingTable.getTable().size());
        return routingTable;
    }

    @Override
    public String findNextHop(String sourceNodeId, String destinationNodeId) {
        if (currentRoutingTable == null) {
            logger.warn("Routing table not initialized");
            return null;
        }
        return currentRoutingTable.getNextHop(destinationNodeId);
    }

    @Override
    public List<String> findOptimalPath(String sourceNodeId, String destinationNodeId,
                                      List<NodeInfo> networkNodes,
                                      Map<String, LinkMetric> linkMetrics) {
        logger.debug("Finding optimal path from {} to {}", sourceNodeId, destinationNodeId);
        
        // Dijkstra's algorithm implementation
        Map<String, Double> distances = new HashMap<>();
        Map<String, String> predecessors = new HashMap<>();
        Set<String> visited = new HashSet<>();
        PriorityQueue<NodeDistance> pq = new PriorityQueue<>(Comparator.comparingDouble(nd -> nd.distance));
        
        // Initialize distances
        for (NodeInfo node : networkNodes) {
            distances.put(node.getNodeId(), Double.MAX_VALUE);
        }
        distances.put(sourceNodeId, 0.0);
        pq.offer(new NodeDistance(sourceNodeId, 0.0));
        
        while (!pq.isEmpty()) {
            NodeDistance current = pq.poll();
            String currentNodeId = current.nodeId;
            
            if (visited.contains(currentNodeId)) {
                continue;
            }
            visited.add(currentNodeId);
            
            if (currentNodeId.equals(destinationNodeId)) {
                break;
            }
            
            // Explore neighbors
            for (LinkMetric link : linkMetrics.values()) {
                String neighborId = null;
                double linkCost = 0.0;
                
                if (link.getSourceNodeId().equals(currentNodeId)) {
                    neighborId = link.getDestinationNodeId();
                    linkCost = calculateLinkCost(link);
                } else if (link.getDestinationNodeId().equals(currentNodeId)) {
                    neighborId = link.getSourceNodeId();
                    linkCost = calculateLinkCost(link);
                }
                
                if (neighborId != null && !visited.contains(neighborId) && link.isLinkAvailable()) {
                    double newDistance = distances.get(currentNodeId) + linkCost;
                    if (newDistance < distances.get(neighborId)) {
                        distances.put(neighborId, newDistance);
                        predecessors.put(neighborId, currentNodeId);
                        pq.offer(new NodeDistance(neighborId, newDistance));
                    }
                }
            }
        }
        
        // Reconstruct path
        List<String> path = new ArrayList<>();
        String current = destinationNodeId;
        
        while (current != null) {
            path.add(0, current);
            current = predecessors.get(current);
        }
        
        // If path doesn't start with source, no path found
        if (path.isEmpty() || !path.get(0).equals(sourceNodeId)) {
            logger.warn("No path found from {} to {}", sourceNodeId, destinationNodeId);
            return new ArrayList<>();
        }
        
        logger.debug("Found path from {} to {}: {}", sourceNodeId, destinationNodeId, path);
        return path;
    }

    @Override
    public void updateRoutingTable(RoutingTable routingTable, Map<String, LinkMetric> updatedLinkMetrics) {
        logger.info("Updating routing table with new link metrics");
        // This would trigger recalculation if significant changes detected
        // For now, we'll just log the update
        this.currentRoutingTable = routingTable;
    }

    @Override
    public boolean hasRouteTo(String sourceNodeId, String destinationNodeId) {
        if (currentRoutingTable == null) {
            return false;
        }
        return currentRoutingTable.getNextHop(destinationNodeId) != null;
    }

    /**
     * Tính toán cost của link dựa trên các metrics
     * Cost thấp hơn = link tốt hơn
     */
    private double calculateLinkCost(LinkMetric link) {
        if (!link.isLinkAvailable()) {
            return Double.MAX_VALUE;
        }
        
        // Công thức cost: kết hợp khoảng cách, latency và packet loss
        // Score cao = cost thấp
        double baseCost = link.getDistanceKm() / 1000.0; // Normalize distance
        double latencyPenalty = link.getLatencyMs() / 100.0; // Normalize latency
        double lossPenalty = link.getPacketLossRate() * 1000; // Heavy penalty for packet loss
        
        return baseCost + latencyPenalty + lossPenalty;
    }

    /**
     * Helper class cho Dijkstra's algorithm
     */
    private static class NodeDistance {
        String nodeId;
        double distance;
        
        NodeDistance(String nodeId, double distance) {
            this.nodeId = nodeId;
            this.distance = distance;
        }
    }
}