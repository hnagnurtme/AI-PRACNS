package com.sagsins.core.repository;

import java.util.List;
import com.sagsins.core.model.NodeInfo;

public interface INodeRepository {
    List<NodeInfo> getAllNodes();
    NodeInfo getNodeById(String nodeId);
    void saveNode(NodeInfo node);
}
