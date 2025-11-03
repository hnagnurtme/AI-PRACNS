package com.example.service;

import java.util.Map;

import com.example.factory.PositionFactory;
import com.example.model.NodeInfo;
import com.example.model.Position;
import com.example.repository.INodeRepository;
import com.example.repository.IUserRepository;
import com.example.util.GeoUtils;

public class NodeService {
    private final IUserRepository userRepository;
    private final INodeRepository nodeRepository;
    public NodeService(IUserRepository userRepository , INodeRepository nodeRepository) {
        this.userRepository = userRepository;
        this.nodeRepository = nodeRepository;
    }

    public NodeInfo getNearestNode(String userId) {
        var userOpt = userRepository.findByUserId(userId);
        if (userOpt.isEmpty()) {
            return null;
        }
        var user = userOpt.get();
        
        Map<String, Position> cities = PositionFactory.createWorldCities();
        Position userPos = cities.get(user.getCityName());
        if (userPos == null) {
            return null;
        }
        Map<String, NodeInfo> allNodes = nodeRepository.loadAllNodeConfigs();
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
        return nearestNode;
    }
}
