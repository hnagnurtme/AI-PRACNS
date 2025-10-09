package com.sagsins.core.seed;
import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.repository.NodeRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import jakarta.annotation.PostConstruct;

import java.util.ArrayList;
import java.util.List;

@Component
public class SagsinsSeeder {
    private static final Logger logger = LoggerFactory.getLogger(SagsinsSeeder.class);
    private final NodeRepository nodeRepository;

    public SagsinsSeeder(NodeRepository nodeRepository) {
        this.nodeRepository = nodeRepository;
    }

    @PostConstruct
    public void seedNetworkNodes() {
        if (!nodeRepository.findAll().isEmpty()) {
            return; 
        }

        List<NodeInfo> nodes = new ArrayList<>();
        nodes.forEach(nodeRepository::save);

        logger.info("Seeded {} network nodes into Firestore.", nodes.size());
    }

}
