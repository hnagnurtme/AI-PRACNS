package com.example.service;

import java.util.Map;
import java.util.Optional; 

import com.example.factory.PositionFactory;
import com.example.model.NodeInfo;
import com.example.model.Position;
import com.example.repository.INodeRepository;
import com.example.repository.IUserRepository;
import com.example.util.GeoUtils;

public class NodeService {
    private final IUserRepository userRepository;
    Map<String, Position> cities = PositionFactory.createWorldCities();
    private final Map<String, NodeInfo> allNodes;
    
    public NodeService(IUserRepository userRepository , INodeRepository nodeRepository) {
        this.userRepository = userRepository;
        this.allNodes = nodeRepository.loadAllNodeConfigs();
    }

    public Optional<NodeInfo> getNearestNode(String userId) { 
        var userOpt = userRepository.findByUserId(userId);
        if (userOpt.isEmpty()) {
            return Optional.empty(); 
        }
        var user = userOpt.get();

        Position userPos = this.cities.get(user.getCityName());
        if (userPos == null) {
            return Optional.empty(); 
        }

        NodeInfo nearestNode = null;
        double minDistance = Double.MAX_VALUE;
        for (NodeInfo node : allNodes.values()) {
            Position nodePos = node.getPosition();
            if (nodePos == null) continue;
            double distance = GeoUtils.calculateDistance3D(userPos, nodePos);
            if (distance < minDistance) {
                minDistance = distance;
                nearestNode = node;
            }
        }
        return Optional.ofNullable(nearestNode); 
    }
}