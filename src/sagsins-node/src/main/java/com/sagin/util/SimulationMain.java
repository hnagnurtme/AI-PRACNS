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

import java.util.Map;

public class SimulationMain {

    private static final Logger logger = LoggerFactory.getLogger(SimulationMain.class);

    public static void main(String[] args) {
        logger.info("=== Starting full SAGIN network simulation ===");

        INodeRepository nodeRepository = new MongoNodeRepository();
        IUserRepository userRepository = new MongoUserRepository();
        INodeService nodeService = new NodeService(nodeRepository);
        DynamicRoutingService routingService = new DynamicRoutingService(nodeRepository, nodeService);

        Map<String, NodeInfo> nodeInfoMap = nodeRepository.loadAllNodeConfigs();
        logger.info("Loaded {} node configurations from repository.", nodeInfoMap.size());

        String envHost = "127.0.0.1";
        nodeInfoMap.values().forEach(nodeInfo -> {
            nodeService.updateNodeIpAddress(nodeInfo.getNodeId(), envHost);
            nodeService.flushToDatabase();
            TCP_Service tcpService = new TCP_Service(nodeRepository, nodeService, userRepository, routingService);
            NodeGateway nodeGateway = new NodeGateway(tcpService);

            new Thread(() -> {
                try {
                    nodeGateway.startListening(nodeInfo, nodeInfo.getCommunication().getPort());
                    logger.info("Node Gateway started for node: {} ({}) on port {}",
                            nodeInfo.getNodeId(), nodeInfo.getNodeType(), nodeInfo.getCommunication().getPort());

                    Thread.currentThread().join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    logger.warn("Node {} listener interrupted", nodeInfo.getNodeId(), e);
                } catch (Exception e) {
                    logger.error("Error starting node {}: {}", nodeInfo.getNodeId(), e.getMessage(), e);
                }
            }, "Node-" + nodeInfo.getNodeId()).start();
        });

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Shutting down all NodeGateways...");
            routingService.shutdown();
        }));

        routingService.forceUpdateRoutingTables();
        logger.info("Initial routing tables updated");

        logger.info("=== All nodes initialized successfully ===");
    }
}
