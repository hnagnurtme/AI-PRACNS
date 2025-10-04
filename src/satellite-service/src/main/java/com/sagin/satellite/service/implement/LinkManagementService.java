package com.sagin.satellite.service.implement;

import com.sagin.satellite.model.NodeInfo;
import com.sagin.satellite.model.LinkMetric;
import com.sagin.satellite.service.ILinkManagementService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * LinkManagementService quản lý tất cả các link trong mạng vệ tinh
 */
public class LinkManagementService implements ILinkManagementService {

    private static final Logger logger = LoggerFactory.getLogger(LinkManagementService.class);
    
    // Map lưu trữ link metrics với key format: "sourceId-destinationId"
    private final Map<String, LinkMetric> linkMetrics = new ConcurrentHashMap<>();
    
    // Cache để tăng tốc độ tìm kiếm neighbor
    private final Map<String, Set<String>> neighborsCache = new ConcurrentHashMap<>();

    @Override
    public LinkMetric establishLink(String sourceNodeId, String destinationNodeId, 
                                  NodeInfo sourceNode, NodeInfo destinationNode) {
        logger.info("Establishing link between {} and {}", sourceNodeId, destinationNodeId);
        
        LinkMetric linkMetric = new LinkMetric();
        linkMetric.setSourceNodeId(sourceNodeId);
        linkMetric.setDestinationNodeId(destinationNodeId);
        linkMetric.setLinkAvailable(true);
        linkMetric.setLastUpdated(System.currentTimeMillis());
        
        // Tính khoảng cách nếu có thông tin vị trí
        if (sourceNode != null && destinationNode != null) {
            linkMetric.updateDistance(sourceNode, destinationNode);
        }
        
        // Khởi tạo metrics mặc định
        linkMetric.updateMetrics(
            100.0, // Default bandwidth 100 Mbps
            calculateBasicLatency(linkMetric.getDistanceKm()), // Latency dựa trên khoảng cách
            0.01, // Default 1% packet loss
            true
        );
        
        String linkId = generateLinkId(sourceNodeId, destinationNodeId);
        linkMetrics.put(linkId, linkMetric);
        
        // Cập nhật neighbor cache
        updateNeighborCache(sourceNodeId, destinationNodeId);
        
        logger.info("Link established: {} with distance {}km, latency {}ms", 
                   linkId, linkMetric.getDistanceKm(), linkMetric.getLatencyMs());
        
        return linkMetric;
    }

    @Override
    public void updateLinkMetrics(String linkId, double bandwidthMbps, double latencyMs, 
                                double packetLossRate, boolean isAvailable) {
        LinkMetric linkMetric = linkMetrics.get(linkId);
        if (linkMetric != null) {
            linkMetric.updateMetrics(bandwidthMbps, latencyMs, packetLossRate, isAvailable);
            logger.debug("Updated metrics for link {}: bandwidth={}Mbps, latency={}ms, loss={}%", 
                        linkId, bandwidthMbps, latencyMs, packetLossRate * 100);
        } else {
            logger.warn("Attempted to update non-existent link: {}", linkId);
        }
    }

    @Override
    public Map<String, LinkMetric> getAllLinkMetrics() {
        return new HashMap<>(linkMetrics);
    }

    @Override
    public LinkMetric getLinkMetric(String sourceNodeId, String destinationNodeId) {
        String linkId = generateLinkId(sourceNodeId, destinationNodeId);
        LinkMetric direct = linkMetrics.get(linkId);
        
        if (direct != null) {
            return direct;
        }
        
        // Thử tìm link ngược lại (bidirectional)
        String reverseLinkId = generateLinkId(destinationNodeId, sourceNodeId);
        return linkMetrics.get(reverseLinkId);
    }

    @Override
    public boolean isLinkAvailable(String sourceNodeId, String destinationNodeId) {
        LinkMetric linkMetric = getLinkMetric(sourceNodeId, destinationNodeId);
        return linkMetric != null && linkMetric.isLinkAvailable();
    }

    @Override
    public List<String> getNeighborNodes(String nodeId) {
        Set<String> neighbors = neighborsCache.get(nodeId);
        return neighbors != null ? new ArrayList<>(neighbors) : new ArrayList<>();
    }

    @Override
    public void removeLink(String sourceNodeId, String destinationNodeId) {
        String linkId = generateLinkId(sourceNodeId, destinationNodeId);
        LinkMetric removed = linkMetrics.remove(linkId);
        
        if (removed != null) {
            // Cập nhật neighbor cache
            removeFromNeighborCache(sourceNodeId, destinationNodeId);
            logger.info("Removed link: {}", linkId);
        } else {
            logger.warn("Attempted to remove non-existent link: {}", linkId);
        }
    }

    @Override
    public void updateLinkDistances(List<NodeInfo> updatedNodes) {
        logger.info("Updating link distances for {} nodes", updatedNodes.size());
        
        Map<String, NodeInfo> nodeMap = new HashMap<>();
        for (NodeInfo node : updatedNodes) {
            nodeMap.put(node.getNodeId(), node);
        }
        
        for (LinkMetric linkMetric : linkMetrics.values()) {
            NodeInfo sourceNode = nodeMap.get(linkMetric.getSourceNodeId());
            NodeInfo destNode = nodeMap.get(linkMetric.getDestinationNodeId());
            
            if (sourceNode != null && destNode != null) {
                linkMetric.updateDistance(sourceNode, destNode);
                
                // Cập nhật latency dựa trên khoảng cách mới
                double newLatency = calculateBasicLatency(linkMetric.getDistanceKm());
                linkMetric.updateMetrics(
                    linkMetric.getBandwidthMbps(),
                    newLatency,
                    linkMetric.getPacketLossRate(),
                    linkMetric.isLinkAvailable()
                );
            }
        }
        
        logger.info("Updated distances for {} links", linkMetrics.size());
    }

    @Override
    public int cleanupDeadLinks() {
        long currentTime = System.currentTimeMillis();
        long deadLinkThreshold = 30000; // 30 seconds
        
        List<String> deadLinks = new ArrayList<>();
        
        for (Map.Entry<String, LinkMetric> entry : linkMetrics.entrySet()) {
            LinkMetric linkMetric = entry.getValue();
            
            // Link được coi là chết nếu:
            // 1. Không khả dụng
            // 2. Không được cập nhật trong thời gian dài
            if (!linkMetric.isLinkAvailable() || 
                (currentTime - linkMetric.getLastUpdated()) > deadLinkThreshold) {
                deadLinks.add(entry.getKey());
            }
        }
        
        // Xóa các dead links
        for (String linkId : deadLinks) {
            LinkMetric deadLink = linkMetrics.remove(linkId);
            if (deadLink != null) {
                removeFromNeighborCache(deadLink.getSourceNodeId(), deadLink.getDestinationNodeId());
            }
        }
        
        logger.info("Cleaned up {} dead links", deadLinks.size());
        return deadLinks.size();
    }

    /**
     * Tạo ID duy nhất cho link
     */
    private String generateLinkId(String sourceNodeId, String destinationNodeId) {
        return sourceNodeId + "-" + destinationNodeId;
    }

    /**
     * Tính latency cơ bản dựa trên khoảng cách
     * Giả sử tốc độ ánh sáng trong không gian
     */
    private double calculateBasicLatency(double distanceKm) {
        double speedOfLight = 299792.458; // km/ms
        return distanceKm / speedOfLight * 2; // Round trip
    }

    /**
     * Cập nhật neighbor cache
     */
    private void updateNeighborCache(String sourceNodeId, String destinationNodeId) {
        neighborsCache.computeIfAbsent(sourceNodeId, k -> new HashSet<>()).add(destinationNodeId);
        neighborsCache.computeIfAbsent(destinationNodeId, k -> new HashSet<>()).add(sourceNodeId);
    }

    /**
     * Xóa khỏi neighbor cache
     */
    private void removeFromNeighborCache(String sourceNodeId, String destinationNodeId) {
        Set<String> sourceNeighbors = neighborsCache.get(sourceNodeId);
        if (sourceNeighbors != null) {
            sourceNeighbors.remove(destinationNodeId);
            if (sourceNeighbors.isEmpty()) {
                neighborsCache.remove(sourceNodeId);
            }
        }
        
        Set<String> destNeighbors = neighborsCache.get(destinationNodeId);
        if (destNeighbors != null) {
            destNeighbors.remove(sourceNodeId);
            if (destNeighbors.isEmpty()) {
                neighborsCache.remove(destinationNodeId);
            }
        }
    }
}