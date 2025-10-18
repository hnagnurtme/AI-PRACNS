package com.sagsins.core.service.implement;

import com.sagsins.core.DTOs.*;
import com.sagsins.core.DTOs.request.UpdateStatusRequest;
import com.sagsins.core.exception.NotFoundException;
import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.repository.INodeRepository;
import com.sagsins.core.service.*;

import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class NodeService implements INodeService {

    private static final Logger log = LoggerFactory.getLogger(NodeService.class);

    private final INodeRepository nodeRepository;
    
    public NodeService(INodeRepository nodeRepository) {
        this.nodeRepository = nodeRepository;
    }

    @Override
    public List<NodeDTO> getAllNodes() {
        log.info("Fetching all nodes from repository");
        List<NodeInfo> nodes = nodeRepository.findAll();
        log.info("Found {} nodes", nodes.size());
        
        return nodes.stream()
                .map(NodeDTO::fromEntity)
                .collect(Collectors.toList());      
    }

    @Override
    public Optional<NodeDTO> getNodeById(String nodeId) {
        log.info("Fetching node with ID: {}", nodeId);
        return nodeRepository.findById(nodeId)
                .map(NodeDTO::fromEntity);
    }

    @Override
    public NodeDTO updateNodeStatus(String nodeId, UpdateStatusRequest request) {
        log.info("Updating node status with ID: {}", nodeId);
        
        // Tìm node hiện tại
        NodeInfo existingNode = nodeRepository.findById(nodeId)
                .orElseThrow(() -> {
                    log.error("Node not found with ID: {}", nodeId);
                    return new NotFoundException("Node not found with ID: " + nodeId);
                });
        
        log.debug("Found existing node: {}", existingNode.getNodeId());
        
        // Cập nhật các trường từ request (chỉ cập nhật các trường không null)
        updateNodeFromRequest(existingNode, request);
        
        // Lưu vào database
        NodeInfo updatedNode = nodeRepository.save(existingNode);
        log.info("Successfully updated node with ID: {}", nodeId);
        
        return NodeDTO.fromEntity(updatedNode);
    }

    /**
     * Helper method để cập nhật NodeInfo từ UpdateStatusRequest
     */
    private void updateNodeFromRequest(NodeInfo node, UpdateStatusRequest request) {
        if (request.getNodeName() != null) {
            node.setNodeName(request.getNodeName());
        }
        if (request.getOrbit() != null) {
            node.setOrbit(request.getOrbit());
        }
        if (request.getVelocity() != null) {
            node.setVelocity(request.getVelocity());
        }
        if (request.getCommunication() != null) {
            node.setCommunication(request.getCommunication());
        }
        if (request.getIsOperational() != null) {
            node.setOperational(request.getIsOperational());
        }
        if (request.getBatteryChargePercent() != null) {
            node.setBatteryChargePercent(request.getBatteryChargePercent());
        }
        if (request.getNodeProcessingDelayMs() != null) {
            node.setNodeProcessingDelayMs(request.getNodeProcessingDelayMs());
        }
        if (request.getPacketLossRate() != null) {
            node.setPacketLossRate(request.getPacketLossRate());
        }
        if (request.getResourceUtilization() != null) {
            node.setResourceUtilization(request.getResourceUtilization());
        }
        if (request.getPacketBufferCapacity() != null) {
            node.setPacketBufferCapacity(request.getPacketBufferCapacity());
        }
        if (request.getCurrentPacketCount() != null) {
            node.setCurrentPacketCount(request.getCurrentPacketCount());
        }
        if (request.getWeather() != null) {
            node.setWeather(request.getWeather());
        }
        if (request.getHost() != null) {
            node.setHost(request.getHost());
        }
        if (request.getPort() != null) {
            node.setPort(request.getPort());
        }
        
        // Luôn cập nhật lastUpdated
        node.setLastUpdated(Instant.now());
    }
}
