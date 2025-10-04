package com.sagin.satellite.controller;

import com.sagin.satellite.model.NodeInfo;
import com.sagin.satellite.model.LinkMetric;
import com.sagin.satellite.service.INetworkTopologyService;
import com.sagin.satellite.service.ILinkManagementService;
import com.sagin.satellite.service.IRoutingService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * REST Controller cho network management
 */
public class NetworkController extends BaseController {

    private static final Logger logger = LoggerFactory.getLogger(NetworkController.class);
    
    private final INetworkTopologyService topologyService;
    private final ILinkManagementService linkService;
    private final IRoutingService routingService;

    public NetworkController(INetworkTopologyService topologyService,
                           ILinkManagementService linkService,
                           IRoutingService routingService) {
        this.topologyService = topologyService;
        this.linkService = linkService;
        this.routingService = routingService;
    }

    /**
     * GET /api/network/nodes
     * Lấy danh sách tất cả nodes
     */
    public List<NodeInfo> getAllNodes() {
        logger.debug("Getting all network nodes");
        try {
            return topologyService.getAllNodes();
        } catch (Exception e) {
            logger.error("Error getting all nodes: {}", e.getMessage());
            throw new RuntimeException("Failed to get nodes", e);
        }
    }

    /**
     * GET /api/network/nodes/{nodeId}
     * Lấy thông tin node cụ thể
     */
    public NodeInfo getNode(String nodeId) {
        logger.debug("Getting node info for: {}", nodeId);
        try {
            NodeInfo node = topologyService.getNode(nodeId);
            if (node == null) {
                throw new RuntimeException("Node not found: " + nodeId);
            }
            return node;
        } catch (Exception e) {
            logger.error("Error getting node {}: {}", nodeId, e.getMessage());
            throw new RuntimeException("Failed to get node info", e);
        }
    }

    /**
     * POST /api/network/nodes
     * Đăng ký node mới
     */
    public Map<String, Object> registerNode(NodeInfo nodeInfo) {
        logger.info("Registering new node: {}", nodeInfo.getNodeId());
        try {
            topologyService.registerNode(nodeInfo);
            return Map.of(
                "success", true,
                "message", "Node registered successfully",
                "nodeId", nodeInfo.getNodeId(),
                "timestamp", System.currentTimeMillis()
            );
        } catch (Exception e) {
            logger.error("Error registering node {}: {}", nodeInfo.getNodeId(), e.getMessage());
            throw new RuntimeException("Failed to register node", e);
        }
    }

    /**
     * DELETE /api/network/nodes/{nodeId}
     * Hủy đăng ký node
     */
    public Map<String, Object> unregisterNode(String nodeId) {
        logger.info("Unregistering node: {}", nodeId);
        try {
            topologyService.unregisterNode(nodeId);
            return Map.of(
                "success", true,
                "message", "Node unregistered successfully",
                "nodeId", nodeId,
                "timestamp", System.currentTimeMillis()
            );
        } catch (Exception e) {
            logger.error("Error unregistering node {}: {}", nodeId, e.getMessage());
            throw new RuntimeException("Failed to unregister node", e);
        }
    }

    /**
     * GET /api/network/nodes/{nodeId}/neighbors
     * Lấy danh sách node kề
     */
    public Map<String, Object> getNeighbors(String nodeId, Double maxDistance) {
        logger.debug("Getting neighbors for node: {} within {}km", nodeId, maxDistance);
        try {
            double distance = maxDistance != null ? maxDistance : 5000.0; // Default 5000km
            List<NodeInfo> neighbors = topologyService.findNeighborNodes(nodeId, distance);
            
            return Map.of(
                "nodeId", nodeId,
                "maxDistance", distance,
                "neighbors", neighbors,
                "count", neighbors.size(),
                "timestamp", System.currentTimeMillis()
            );
        } catch (Exception e) {
            logger.error("Error getting neighbors for {}: {}", nodeId, e.getMessage());
            throw new RuntimeException("Failed to get neighbors", e);
        }
    }

    /**
     * GET /api/network/links
     * Lấy tất cả link metrics
     */
    public Map<String, LinkMetric> getAllLinks() {
        logger.debug("Getting all link metrics");
        try {
            return linkService.getAllLinkMetrics();
        } catch (Exception e) {
            logger.error("Error getting all links: {}", e.getMessage());
            throw new RuntimeException("Failed to get links", e);
        }
    }

    /**
     * GET /api/network/links/{sourceId}/{destinationId}
     * Lấy link metric giữa hai node
     */
    public Map<String, Object> getLink(String sourceId, String destinationId) {
        logger.debug("Getting link between {} and {}", sourceId, destinationId);
        try {
            LinkMetric link = linkService.getLinkMetric(sourceId, destinationId);
            
            if (link == null) {
                return Map.of(
                    "exists", false,
                    "sourceId", sourceId,
                    "destinationId", destinationId,
                    "timestamp", System.currentTimeMillis()
                );
            }
            
            return Map.of(
                "exists", true,
                "link", link,
                "timestamp", System.currentTimeMillis()
            );
        } catch (Exception e) {
            logger.error("Error getting link {}-{}: {}", sourceId, destinationId, e.getMessage());
            throw new RuntimeException("Failed to get link", e);
        }
    }

    /**
     * POST /api/network/topology/update
     * Cập nhật topology với khoảng cách tối đa
     */
    public Map<String, Object> updateTopology(Double maxLinkDistance) {
        logger.info("Updating network topology with max distance: {}km", maxLinkDistance);
        try {
            double distance = maxLinkDistance != null ? maxLinkDistance : 5000.0;
            int newLinks = topologyService.updateTopology(distance);
            
            return Map.of(
                "success", true,
                "newLinksCreated", newLinks,
                "maxLinkDistance", distance,
                "timestamp", System.currentTimeMillis()
            );
        } catch (Exception e) {
            logger.error("Error updating topology: {}", e.getMessage());
            throw new RuntimeException("Failed to update topology", e);
        }
    }

    /**
     * GET /api/network/route/{sourceId}/{destinationId}
     * Tìm đường đi từ source tới destination
     */
    public Map<String, Object> findRoute(String sourceId, String destinationId) {
        logger.debug("Finding route from {} to {}", sourceId, destinationId);
        try {
            List<NodeInfo> allNodes = topologyService.getAllNodes();
            Map<String, LinkMetric> linkMetrics = linkService.getAllLinkMetrics();
            
            List<String> path = routingService.findOptimalPath(
                sourceId, destinationId, allNodes, linkMetrics);
            
            String nextHop = routingService.findNextHop(sourceId, destinationId);
            
            return Map.of(
                "sourceId", sourceId,
                "destinationId", destinationId,
                "pathExists", !path.isEmpty(),
                "path", path,
                "nextHop", nextHop != null ? nextHop : "",
                "hopCount", Math.max(0, path.size() - 1),
                "timestamp", System.currentTimeMillis()
            );
        } catch (Exception e) {
            logger.error("Error finding route {}-{}: {}", sourceId, destinationId, e.getMessage());
            throw new RuntimeException("Failed to find route", e);
        }
    }

    /**
     * GET /api/network/coverage
     * Lấy thông tin độ phủ sóng mạng
     */
    public Map<String, Object> getNetworkCoverage() {
        logger.debug("Getting network coverage information");
        try {
            double coverage = topologyService.calculateNetworkCoverage();
            List<String> isolatedNodes = topologyService.findIsolatedNodes();
            
            return Map.of(
                "coverage", coverage,
                "coveragePercentage", coverage * 100,
                "isolatedNodes", isolatedNodes,
                "isolatedCount", isolatedNodes.size(),
                "timestamp", System.currentTimeMillis()
            );
        } catch (Exception e) {
            logger.error("Error getting network coverage: {}", e.getMessage());
            throw new RuntimeException("Failed to get coverage info", e);
        }
    }
}