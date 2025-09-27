package com.sagsins.core.service;

import java.util.List;

import com.sagsins.core.DTOs.CreateNodeRequest;
import com.sagsins.core.DTOs.UpdateNodeRequest;
import com.sagsins.core.model.NodeInfo;

public interface INodeService {
    List<NodeInfo> getAllNodeIds();
    NodeInfo getNodeById(String nodeId);
    NodeInfo createNode(CreateNodeRequest request);
    NodeInfo updateNode(String nodeId, UpdateNodeRequest request);
    void deleteNode(String nodeId);
    NodeInfo addNode(); // Keep for backward compatibility
}
