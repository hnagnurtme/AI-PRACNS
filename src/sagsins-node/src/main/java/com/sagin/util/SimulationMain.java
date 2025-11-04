package com.sagin.util;

import com.sagin.model.NodeInfo;
import com.sagin.network.implement.NodeGateway;
import com.sagin.network.implement.TCP_Service;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.IPacketComparisonRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.repository.MongoNodeRepository;
import com.sagin.repository.MongoPacketComparisonRepository;
import com.sagin.repository.MongoUserRepository;
import com.sagin.routing.DynamicRoutingService;
import com.sagin.service.INodeService;
import com.sagin.service.NodeService;
import com.sagin.service.PacketComparisonService;

import org.slf4j.Logger;

import java.util.Map;

public class SimulationMain {

    private static final Logger logger = AppLogger.getLogger(SimulationMain.class);

    public static void main(String[] args) {
        String simulationId = String.valueOf(System.currentTimeMillis());
        AppLogger.putMdc("simulationId", simulationId);
        
        logger.info("=== Starting full SAGIN network simulation ===");
        logger.info("Simulation ID: {}", simulationId);

        INodeRepository nodeRepository = new MongoNodeRepository();
        IUserRepository userRepository = new MongoUserRepository();
        IPacketComparisonRepository packetComparisonRepository = new MongoPacketComparisonRepository();
        INodeService nodeService = new NodeService(nodeRepository);
        DynamicRoutingService routingService = new DynamicRoutingService(nodeRepository, nodeService);
        PacketComparisonService packetComparisonService = new PacketComparisonService(packetComparisonRepository);

        Map<String, NodeInfo> nodeInfoMap = nodeRepository.loadAllNodeConfigs();
        logger.info("Loaded {} node configurations from repository.", nodeInfoMap.size());

        String envHost = "127.0.0.1";
        nodeInfoMap.values().forEach(nodeInfo -> {
            nodeService.updateNodeIpAddress(nodeInfo.getNodeId(), envHost);
            nodeService.flushToDatabase();
            TCP_Service tcpService = new TCP_Service(nodeRepository, nodeService, userRepository, routingService, packetComparisonService);
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
            AppLogger.clearMdc();
        }));

        routingService.forceUpdateRoutingTables();
        logger.info("Initial routing tables updated");

        logger.info("=== All nodes initialized successfully ===");
    }
}
