package com.sagsins.core.service.implement;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.sagsins.core.model.NodeInfo;
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
}
