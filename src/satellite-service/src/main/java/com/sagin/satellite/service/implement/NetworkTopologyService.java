package com.sagin.satellite.service.implement;

import com.sagin.satellite.model.NodeInfo;
import com.sagin.satellite.service.INetworkTopologyService;
import com.sagin.satellite.service.ILinkManagementService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * NetworkTopologyService quản lý topology của mạng vệ tinh
 */
public class NetworkTopologyService implements INetworkTopologyService {

    private static final Logger logger = LoggerFactory.getLogger(NetworkTopologyService.class);
    
    // Thread-safe storage cho network nodes
    private final Map<String, NodeInfo> networkNodes = new ConcurrentHashMap<>();
    private final ILinkManagementService linkManagementService;
    
    // Configuration
    private static final double DEFAULT_MAX_LINK_DISTANCE = 5000.0; // 5000 km

    public NetworkTopologyService(ILinkManagementService linkManagementService) {
        this.linkManagementService = linkManagementService;
    }

    @Override
    public void registerNode(NodeInfo nodeInfo) {
        if (nodeInfo == null || nodeInfo.getNodeId() == null) {
            logger.error("Cannot register null node or node with null ID");
            return;
        }
        
        logger.info("Registering node: {}", nodeInfo.getNodeId());
        networkNodes.put(nodeInfo.getNodeId(), nodeInfo);
        
        // Tự động tạo links với các node lân cận
        establishLinksForNewNode(nodeInfo);
        
        logger.info("Node {} registered successfully. Total nodes: {}", 
                   nodeInfo.getNodeId(), networkNodes.size());
    }

    @Override
    public void unregisterNode(String nodeId) {
        if (nodeId == null) {
            logger.error("Cannot unregister node with null ID");
            return;
        }
        
        logger.info("Unregistering node: {}", nodeId);
        
        NodeInfo removedNode = networkNodes.remove(nodeId);
        if (removedNode != null) {
            // Xóa tất cả links liên quan đến node này
            removeAllLinksForNode(nodeId);
            logger.info("Node {} unregistered successfully. Remaining nodes: {}", 
                       nodeId, networkNodes.size());
        } else {
            logger.warn("Attempted to unregister non-existent node: {}", nodeId);
        }
    }

    @Override
    public void updateNode(NodeInfo nodeInfo) {
        if (nodeInfo == null || nodeInfo.getNodeId() == null) {
            logger.error("Cannot update null node or node with null ID");
            return;
        }
        
        String nodeId = nodeInfo.getNodeId();
        
        if (networkNodes.containsKey(nodeId)) {
            logger.debug("Updating node: {}", nodeId);
            networkNodes.put(nodeId, nodeInfo);
            
            // Cập nhật link distances nếu vị trí thay đổi
            updateLinksForNode(nodeInfo);
        } else {
            logger.warn("Attempted to update non-existent node: {}", nodeId);
        }
    }

    @Override
    public List<NodeInfo> getAllNodes() {
        return new ArrayList<>(networkNodes.values());
    }

    @Override
    public NodeInfo getNode(String nodeId) {
        return networkNodes.get(nodeId);
    }

    @Override
    public List<NodeInfo> findNeighborNodes(String nodeId, double maxDistance) {
        NodeInfo sourceNode = networkNodes.get(nodeId);
        if (sourceNode == null) {
            logger.warn("Source node not found: {}", nodeId);
            return new ArrayList<>();
        }
        
        List<NodeInfo> neighbors = new ArrayList<>();
        
        for (NodeInfo candidate : networkNodes.values()) {
            if (!candidate.getNodeId().equals(nodeId)) {
                double distance = sourceNode.distanceTo(candidate);
                if (distance <= maxDistance) {
                    neighbors.add(candidate);
                }
            }
        }
        
        // Sắp xếp theo khoảng cách gần nhất
        neighbors.sort((n1, n2) -> {
            double d1 = sourceNode.distanceTo(n1);
            double d2 = sourceNode.distanceTo(n2);
            return Double.compare(d1, d2);
        });
        
        logger.debug("Found {} neighbors for node {} within {}km", 
                    neighbors.size(), nodeId, maxDistance);
        
        return neighbors;
    }

    @Override
    public int updateTopology(double maxLinkDistance) {
        logger.info("Updating network topology with max link distance: {}km", maxLinkDistance);
        
        int newLinksCreated = 0;
        List<NodeInfo> nodes = getAllNodes();
        
        // Cập nhật vị trí cho link management service
        linkManagementService.updateLinkDistances(nodes);
        
        // Tạo links mới cho các node trong phạm vi
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                NodeInfo node1 = nodes.get(i);
                NodeInfo node2 = nodes.get(j);
                
                double distance = node1.distanceTo(node2);
                
                if (distance <= maxLinkDistance) {
                    // Kiểm tra xem link đã tồn tại chưa
                    if (linkManagementService.getLinkMetric(node1.getNodeId(), node2.getNodeId()) == null) {
                        linkManagementService.establishLink(
                            node1.getNodeId(), node2.getNodeId(), node1, node2);
                        newLinksCreated++;
                    }
                }
            }
        }
        
        // Cleanup các dead links
        int deadLinksRemoved = linkManagementService.cleanupDeadLinks();
        
        logger.info("Topology update completed: {} new links created, {} dead links removed", 
                   newLinksCreated, deadLinksRemoved);
        
        return newLinksCreated;
    }

    @Override
    public Map<String, Object> getNetworkSnapshot() {
        Map<String, Object> snapshot = new HashMap<>();
        
        List<NodeInfo> nodes = getAllNodes();
        
        snapshot.put("totalNodes", nodes.size());
        snapshot.put("nodes", nodes);
        snapshot.put("totalLinks", linkManagementService.getAllLinkMetrics().size());
        snapshot.put("linkMetrics", linkManagementService.getAllLinkMetrics());
        snapshot.put("networkCoverage", calculateNetworkCoverage());
        snapshot.put("isolatedNodes", findIsolatedNodes());
        snapshot.put("timestamp", System.currentTimeMillis());
        
        // Network statistics
        double avgNodesPerNode = nodes.isEmpty() ? 0 : 
            nodes.stream()
                 .mapToDouble(node -> linkManagementService.getNeighborNodes(node.getNodeId()).size())
                 .average()
                 .orElse(0.0);
        
        snapshot.put("averageConnectionsPerNode", avgNodesPerNode);
        
        return snapshot;
    }

    @Override
    public boolean isConnected(String sourceNodeId, String destinationNodeId) {
        if (!networkNodes.containsKey(sourceNodeId) || !networkNodes.containsKey(destinationNodeId)) {
            return false;
        }
        
        // Sử dụng BFS để kiểm tra connectivity
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        
        queue.offer(sourceNodeId);
        visited.add(sourceNodeId);
        
        while (!queue.isEmpty()) {
            String currentNode = queue.poll();
            
            if (currentNode.equals(destinationNodeId)) {
                return true;
            }
            
            List<String> neighbors = linkManagementService.getNeighborNodes(currentNode);
            for (String neighbor : neighbors) {
                if (!visited.contains(neighbor) && 
                    linkManagementService.isLinkAvailable(currentNode, neighbor)) {
                    visited.add(neighbor);
                    queue.offer(neighbor);
                }
            }
        }
        
        return false;
    }

    @Override
    public double calculateNetworkCoverage() {
        if (networkNodes.isEmpty()) {
            return 0.0;
        }
        
        // Đơn giản hóa: tính toán dựa trên số node có ít nhất 1 connection
        long connectedNodes = networkNodes.keySet().stream()
            .filter(nodeId -> !linkManagementService.getNeighborNodes(nodeId).isEmpty())
            .count();
        
        return (double) connectedNodes / networkNodes.size();
    }

    @Override
    public List<String> findIsolatedNodes() {
        return networkNodes.keySet().stream()
            .filter(nodeId -> linkManagementService.getNeighborNodes(nodeId).isEmpty())
            .collect(Collectors.toList());
    }

    /**
     * Thiết lập links cho node mới
     */
    private void establishLinksForNewNode(NodeInfo newNode) {
        List<NodeInfo> potentialNeighbors = findNeighborNodes(
            newNode.getNodeId(), DEFAULT_MAX_LINK_DISTANCE);
        
        for (NodeInfo neighbor : potentialNeighbors) {
            linkManagementService.establishLink(
                newNode.getNodeId(), neighbor.getNodeId(), newNode, neighbor);
        }
        
        logger.debug("Established {} links for new node {}", 
                    potentialNeighbors.size(), newNode.getNodeId());
    }

    /**
     * Xóa tất cả links của một node
     */
    private void removeAllLinksForNode(String nodeId) {
        List<String> neighbors = linkManagementService.getNeighborNodes(nodeId);
        
        for (String neighbor : neighbors) {
            linkManagementService.removeLink(nodeId, neighbor);
        }
        
        logger.debug("Removed {} links for node {}", neighbors.size(), nodeId);
    }

    /**
     * Cập nhật links cho node đã thay đổi vị trí
     */
    private void updateLinksForNode(NodeInfo updatedNode) {
        List<NodeInfo> allNodes = getAllNodes();
        linkManagementService.updateLinkDistances(allNodes);
    }
}