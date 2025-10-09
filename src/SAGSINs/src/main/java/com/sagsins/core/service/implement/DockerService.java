package com.sagsins.core.service.implement;

import java.util.Optional;

import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.service.IDockerService;

public class DockerService implements IDockerService {
    @Override
    public Optional<String> runContainerForNode(NodeInfo nodeInfo) {
        // Extract NodeInfo details
        String nodeId = nodeInfo.getNodeId();
        String nodeType = nodeInfo.getNodeType();
        
        // TODO Auto-generated method stub
        return Optional.empty();
    }

    @Override
    public boolean stopAndRemoveContainer(String nodeId) {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public Optional<String> getContainerStatus(String nodeId) {
        // TODO Auto-generated method stub
        return Optional.empty();
    }
    
}
