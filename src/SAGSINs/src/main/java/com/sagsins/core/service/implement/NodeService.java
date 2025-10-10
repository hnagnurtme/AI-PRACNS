package com.sagsins.core.service.implement;

import com.sagsins.core.DTOs.*;
import com.sagsins.core.exception.*;
import com.sagsins.core.model.*;
import com.sagsins.core.repository.INodeRepository;
import com.sagsins.core.service.*;

import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class NodeService implements INodeService {

    private static final Logger log = LoggerFactory.getLogger(NodeService.class);

    private final INodeRepository nodeRepository;
    private final IDockerService dockerService;

    public NodeService(INodeRepository nodeRepository, IDockerService dockerService) {
        this.nodeRepository = nodeRepository;
        this.dockerService = dockerService;
    }

    // ----------------------------------------------------------------------
    @Override
    public NodeDTO createNode(CreateNodeRequest request) {
        if (nodeRepository.existsById(request.getNodeId())) {
            throw new DuplicateKeyException("Node with ID " + request.getNodeId() + " already exists.");
        }

        NodeInfo newNode = new NodeInfo();
        newNode.setNodeId(request.getNodeId());

        try {
            newNode.setNodeType(NodeType.valueOf(request.getNodeType()));
        } catch (IllegalArgumentException e) {
            log.warn("Invalid nodeType '{}', set default to GROUND_STATION", request.getNodeType());
            newNode.setNodeType(NodeType.GROUND_STATION);
        }

        newNode.setOperational(request.isOperational());
        newNode.setPosition(request.getPosition());
        newNode.setOrbit(request.getOrbit());
        newNode.setVelocity(request.getVelocity());
        newNode.setBatteryChargePercent(request.getBatteryChargePercent());
        newNode.setNodeProcessingDelayMs(request.getNodeProcessingDelayMs());
        newNode.setPacketLossRate(request.getPacketLossRate());
        newNode.setResourceUtilization(request.getResourceUtilization());
        newNode.setPacketBufferCapacity(request.getPacketBufferCapacity());
        newNode.setWeather(request.getWeather());
        newNode.setHost(request.getHost());
        newNode.setPort(request.getPort());
        newNode.setCurrentPacketCount(0);
        newNode.setLastUpdated(System.currentTimeMillis());

        NodeInfo savedNode = nodeRepository.save(newNode);

        try {
            dockerService.runContainerForNode(savedNode);
        } catch (Exception e) {
            log.error("❌ Lỗi khi khởi tạo container cho Node {}: {}", savedNode.getNodeId(), e.getMessage());
            throw new DockerException("Failed to create Docker container for Node " + savedNode.getNodeId() + ": " + e.getMessage());
        }

        log.info("✅ Node {} created successfully.", savedNode.getNodeId());
        return convertToDTO(savedNode);
    }

    // ----------------------------------------------------------------------
    @Override
    public List<NodeDTO> getAllNodes() {
        return nodeRepository.findAll()
                .stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    @Override
    public Optional<NodeDTO> getNodeById(String nodeId) {
        return nodeRepository.findById(nodeId)
                .map(this::convertToDTO);
    }

    // ----------------------------------------------------------------------
    @Override
    public Optional<NodeDTO> updateNode(String nodeId, UpdateNodeRequest request) {
        return nodeRepository.findById(nodeId).map(existingNode -> {

            if (request.getNodeType() != null) {
                try {
                    existingNode.setNodeType(NodeType.valueOf(request.getNodeType()));
                } catch (IllegalArgumentException e) {
                    log.warn("Invalid nodeType '{}' ignored.", request.getNodeType());
                }
            }

            if (request.getIsOperational() != null)
                existingNode.setOperational(request.getIsOperational());

            if (request.getPosition() != null) existingNode.setPosition(request.getPosition());
            if (request.getOrbit() != null) existingNode.setOrbit(request.getOrbit());
            if (request.getVelocity() != null) existingNode.setVelocity(request.getVelocity());

            if (request.getBatteryChargePercent() != null)
                existingNode.setBatteryChargePercent(request.getBatteryChargePercent());
            if (request.getNodeProcessingDelayMs() != null)
                existingNode.setNodeProcessingDelayMs(request.getNodeProcessingDelayMs());
            if (request.getPacketLossRate() != null)
                existingNode.setPacketLossRate(request.getPacketLossRate());
            if (request.getResourceUtilization() != null)
                existingNode.setResourceUtilization(request.getResourceUtilization());
            if (request.getPacketBufferCapacity() != null)
                existingNode.setPacketBufferCapacity(request.getPacketBufferCapacity());
            if (request.getPacketBufferLoad() != null)
                existingNode.setCurrentPacketCount(request.getPacketBufferLoad());
            if (request.getWeather() != null)
                existingNode.setWeather(request.getWeather());
            if (request.getHost() != null)
                existingNode.setHost(request.getHost());
            if (request.getPort() != null)
                existingNode.setPort(request.getPort());

            existingNode.setLastUpdated(System.currentTimeMillis());

            NodeInfo updatedNode = nodeRepository.save(existingNode);
            log.info("♻️ Node {} updated successfully.", nodeId);
            return convertToDTO(updatedNode);
        });
    }

    // ----------------------------------------------------------------------
    @Override
    public boolean deleteNode(String nodeId) {
        if (nodeRepository.existsById(nodeId)) {
            nodeRepository.deleteById(nodeId);
            try {
                dockerService.stopAndRemoveContainer(nodeId);
            } catch (Exception e) {
                log.warn("⚠️ Docker container for Node {} could not be removed: {}", nodeId, e.getMessage());
            }
            log.info("🗑️ Node {} deleted successfully.", nodeId);
            return true;
        }
        return false;
    }

    // ----------------------------------------------------------------------
    private NodeDTO convertToDTO(NodeInfo nodeInfo) {
        if (nodeInfo == null) return null;

        NodeDTO dto = new NodeDTO();
        dto.setNodeId(nodeInfo.getNodeId());
        dto.setNodeType(nodeInfo.getNodeType() != null ? nodeInfo.getNodeType().name() : null);
        dto.setPosition(nodeInfo.getPosition());
        dto.setOrbit(nodeInfo.getOrbit());
        dto.setVelocity(nodeInfo.getVelocity());
        dto.setOperational(nodeInfo.isOperational());
        dto.setIsHealthy(nodeInfo.isHealthy());
        dto.setBatteryChargePercent(nodeInfo.getBatteryChargePercent());
        dto.setNodeProcessingDelayMs(nodeInfo.getNodeProcessingDelayMs());
        dto.setPacketLossRate(nodeInfo.getPacketLossRate());
        dto.setResourceUtilization(nodeInfo.getResourceUtilization());
        dto.setPacketBufferCapacity(nodeInfo.getPacketBufferCapacity());
        dto.setCurrentPacketCount(nodeInfo.getCurrentPacketCount());
        dto.setWeather(nodeInfo.getWeather());
        dto.setHost(nodeInfo.getHost());
        dto.setPort(nodeInfo.getPort());
        dto.setLastUpdated(nodeInfo.getLastUpdated());
        return dto;
    }

    @Override
    public boolean runNodeProcess(String nodeId) {
        Optional<NodeInfo> nodeOpt = nodeRepository.findById(nodeId);
        if (nodeOpt.isPresent()) {
            NodeInfo node = nodeOpt.get();
            try {
                dockerService.runContainerForNode(node);
                log.info("Node process for {} started successfully.", nodeId);
                return true;
            } catch (Exception e) {
                log.error("Failed to start node process for {}: {}", nodeId, e.getMessage());
                return false;
            }
        }
        log.warn(" Node {} not found. Cannot start process.", nodeId);
        return false;
    }
}
