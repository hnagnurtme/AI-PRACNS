package com.sagsins.core.service.implement;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.sagsins.core.DTOs.CreateNodeRequest;
import com.sagsins.core.DTOs.UpdateNodeRequest;
import com.sagsins.core.model.Geo3D;
import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.model.Orbit;
import com.sagsins.core.model.Velocity;
import com.sagsins.core.repository.INodeRepository;
import com.sagsins.core.service.INodeService;

@Service
public class NodeService implements INodeService{
    private static final Logger logger = LoggerFactory.getLogger(NodeService.class);
    private final INodeRepository nodeRepository;
    
    public NodeService(INodeRepository nodeService) {
        this.nodeRepository = nodeService;
    }

    @Override
    public List<NodeInfo> getAllNodeIds() {
        List<NodeInfo> nodes = new ArrayList<>();
        nodes = nodeRepository.getAllNodes();
        if (nodes.isEmpty()) {
            logger.warn("No nodes found in the repository.");
        }
        return nodes;
    }

    @Override
    public NodeInfo getNodeById(String nodeId) {
        return nodeRepository.getNodeById(nodeId);
    }

    @Override
    public NodeInfo createNode(CreateNodeRequest request) {
        // Convert DTO to model
        Geo3D position = new Geo3D(
            request.getPosition().getLongitude(),
            request.getPosition().getLatitude(),
            request.getPosition().getAltitude()
        );
        
        Orbit orbit = null;
        if (request.getOrbit() != null) {
            orbit = new Orbit(
                "LEO", // default type
                request.getOrbit().getInclination(),
                95.0, // default period
                request.getOrbit().getAltitude() / 1000.0 // convert to km
            );
        }
        
        Velocity velocity = null;
        if (request.getVelocity() != null) {
            velocity = new Velocity(
                0.0, // x component
                request.getVelocity().getSpeed(), // y component
                0.0  // z component
            );
        }
        
        NodeInfo newNode = new NodeInfo(request.getNodeType(), position, orbit, velocity);
        nodeRepository.saveNode(newNode);
        return newNode;
    }

    @Override
    public NodeInfo updateNode(String nodeId, UpdateNodeRequest request) {
        NodeInfo existingNode = nodeRepository.getNodeById(nodeId);
        if (existingNode == null) {
            throw new RuntimeException("Node not found: " + nodeId);
        }
        
        // Update fields if provided
        if (request.getNodeType() != null) {
            existingNode.setNodeType(request.getNodeType());
        }
        
        if (request.getPosition() != null) {
            Geo3D newPosition = new Geo3D(
                request.getPosition().getLongitude(),
                request.getPosition().getLatitude(),
                request.getPosition().getAltitude()
            );
            existingNode.setPosition(newPosition);
        }
        
        if (request.getOrbit() != null) {
            Orbit newOrbit = new Orbit(
                "LEO", // default type
                request.getOrbit().getInclination(),
                95.0, // default period
                request.getOrbit().getAltitude() / 1000.0 // convert to km
            );
            existingNode.setOrbit(newOrbit);
        }
        
        if (request.getVelocity() != null) {
            Velocity newVelocity = new Velocity(
                0.0, // x component
                request.getVelocity().getSpeed(), // y component
                0.0  // z component
            );
            existingNode.setVelocity(newVelocity);
        }
        
        nodeRepository.saveNode(existingNode);
        return existingNode;
    }

    @Override
    public void deleteNode(String nodeId) {
        NodeInfo node = nodeRepository.getNodeById(nodeId);
        if (node == null) {
            throw new RuntimeException("Node not found: " + nodeId);
        }
        nodeRepository.deleteNode(nodeId);
    }

    @Override
    public NodeInfo addNode() {
        // Keep existing implementation for backward compatibility
        // Create a default node for testing
        Geo3D position = new Geo3D(0.0, 0.0, 1000.0);
        NodeInfo newNode = new NodeInfo("UE", position);
        nodeRepository.saveNode(newNode);
        return newNode;
    }
}
