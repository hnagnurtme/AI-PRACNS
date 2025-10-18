package com.sagin.util;

import com.sagin.model.NodeInfo;
import com.sagin.network.implement.NodeGateway;
import com.sagin.network.implement.TCP_Service;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.repository.MongoNodeRepository;
import com.sagin.repository.MongoUserRepository;
import com.sagin.routing.DynamicRoutingService;
import com.sagin.service.INodeService;
import com.sagin.service.NodeService;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;

public class SimulationMain {

    private static final Logger logger = LoggerFactory.getLogger(SimulationMain.class);

    public static void main(String[] args) {
        INodeRepository nodeRepository = new MongoNodeRepository();
        IUserRepository userRepository = new MongoUserRepository();
        INodeService nodeService = new NodeService(nodeRepository);
        DynamicRoutingService routingService = new DynamicRoutingService(nodeRepository, nodeService);

        String[] nodeIds = { "N-TOKYO", "N-SINGAPORE" };
        for (String nodeId : nodeIds) {
            Optional<NodeInfo> nodeInfoOptional = nodeRepository.getNodeInfo(nodeId);
            nodeInfoOptional.ifPresent(nodeInfo -> {
                TCP_Service tcpService = new TCP_Service(nodeRepository, nodeService, userRepository, routingService);
                NodeGateway nodeGateway = new NodeGateway(tcpService);

                new Thread(() -> {
                    nodeGateway.startListening(nodeInfo, nodeInfo.getCommunication().port());
                    logger.info("Node Gateway started for node: {}", nodeInfo.getNodeId());

                    try {
                        Thread.currentThread().join();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        logger.warn("Node {} listener interrupted", nodeInfo.getNodeId(), e);
                    }
                }).start();
            });
        }

        // Shutdown hook cho tất cả node
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Shutting down all NodeGateways...");
            routingService.shutdown();
        }));

        // Force update routing table một lần
        routingService.forceUpdateRoutingTables();
        logger.info("Initial routing tables updated");
    }
}
