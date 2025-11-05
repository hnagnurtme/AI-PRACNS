package com.sagin.util;

import com.sagin.model.NodeInfo;
import com.sagin.network.implement.NodeGateway;
import com.sagin.network.implement.TCP_Service;
import com.sagin.repository.IBatchPacketRepository;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.ITwoPacketRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.repository.MongoBatchPacketRepository;
import com.sagin.repository.MongoNodeRepository;
import com.sagin.repository.MongoTwoPacketRepository;
import com.sagin.repository.MongoUserRepository;
import com.sagin.routing.DynamicRoutingService;
import com.sagin.routing.RLRoutingService;
import com.sagin.service.BatchPacketService;
import com.sagin.service.INodeService;
import com.sagin.service.NodeService;

import org.slf4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Main entry point for the SAGIN network simulation.
 * This class initializes all singleton services (Repositories, Routing, TCP),
 * configures all nodes, and launches a NodeGateway for each node on its own
 * thread, managed by an ExecutorService.
 */
public class SimulationMain {

    private static final Logger logger = AppLogger.getLogger(SimulationMain.class);

    private static final String NODE_HOST_IP = System.getenv().getOrDefault("NODE_HOST_IP", "127.0.0.1");
    private static final String RL_SERVICE_HOST_DEFAULT = "127.0.0.1";
    private static final int RL_SERVICE_PORT_DEFAULT = 6000;

    public static void main(String[] args) {
        String simulationId = String.valueOf(System.currentTimeMillis());
        AppLogger.putMdc("simulationId", simulationId);

        INodeRepository nodeRepository = new MongoNodeRepository();
        IUserRepository userRepository = new MongoUserRepository();
        INodeService nodeService = new NodeService(nodeRepository);
        DynamicRoutingService routingService = new DynamicRoutingService(nodeRepository, nodeService);
        ITwoPacketRepository twoPacketRepository = new MongoTwoPacketRepository();
        IBatchPacketRepository batchPacketRepository = new MongoBatchPacketRepository();
        BatchPacketService batchPacketService = new BatchPacketService(batchPacketRepository, twoPacketRepository);

        String rlHost = System.getenv().getOrDefault("RL_SERVICE_HOST", RL_SERVICE_HOST_DEFAULT);
        int rlPort = getPortFromEnv("RL_SERVICE_PORT", RL_SERVICE_PORT_DEFAULT);
        RLRoutingService rlRoutingService = new RLRoutingService(rlHost, rlPort);
        logger.info("RL Routing Service configured for {}:{}", rlHost, rlPort);

        TCP_Service tcpService = new TCP_Service(
                nodeRepository,
                nodeService,
                userRepository,
                routingService,
                batchPacketService,
                rlRoutingService);

        Map<String, NodeInfo> nodeInfoMap = nodeRepository.loadAllNodeConfigs();
        logger.info("Loaded {} node configurations.", nodeInfoMap.size());

        nodeInfoMap.values().forEach(nodeInfo -> {
            nodeService.updateNodeIpAddress(nodeInfo.getNodeId(), NODE_HOST_IP);
        });

        nodeService.flushToDatabase();
        logger.info("Node IP addresses flushed to database.");

        ExecutorService nodeLauncherPool = Executors.newFixedThreadPool(nodeInfoMap.size());

        List<NodeGateway> runningGateways = new ArrayList<>();

        logger.info("Starting {} NodeGateways...", nodeInfoMap.size());
        for (NodeInfo nodeInfo : nodeInfoMap.values()) {
            NodeGateway nodeGateway = new NodeGateway(tcpService);
            runningGateways.add(nodeGateway);

            nodeLauncherPool.submit(() -> {
                try {
                    AppLogger.putMdc("simulationId", simulationId);
                    AppLogger.putMdc("nodeId", nodeInfo.getNodeId());

                    nodeGateway.startListening(nodeInfo, nodeInfo.getCommunication().getPort());

                    logger.info("Node Gateway started for node: {} ({}) on port {}",
                            nodeInfo.getNodeId(), nodeInfo.getNodeType(), nodeInfo.getCommunication().getPort());
                    Thread.currentThread().join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    logger.warn("Node {} listener interrupted", nodeInfo.getNodeId(), e);
                } catch (Exception e) {
                    logger.error("Error starting node {}: {}", nodeInfo.getNodeId(), e.getMessage(), e);
                } finally {
                    AppLogger.clearMdc(); 
                }
            });
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("=== SHUTDOWN HOOK: Stopping all services... ===");
            try {
                logger.info("Stopping all NodeGateways ({})...", runningGateways.size());
                for (NodeGateway gateway : runningGateways) {
                    gateway.stopListening();
                }

                logger.info("Shutting down Node Launcher pool...");
                nodeLauncherPool.shutdown();
                if (!nodeLauncherPool.awaitTermination(5, TimeUnit.SECONDS)) {
                    nodeLauncherPool.shutdownNow(); 
                }

                logger.info("Stopping core services (Routing, TCP)...");
                routingService.shutdown();
                tcpService.stop(); 

                logger.info("=== All services stopped. Shutdown complete. ===");
                AppLogger.clearMdc();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.error("Shutdown hook interrupted.", e);
            }
        }));

        routingService.forceUpdateRoutingTables();
        logger.info("Initial routing tables updated.");

        logger.info("=== All {} nodes initialized successfully ===", nodeInfoMap.size());
    }

    /**
     * A helper method to safely parse a port number from an environment
     * variable.
     *
     * @param envVariable  The name of the environment variable (e.g.,
     * "RL_SERVICE_PORT").
     * @param defaultPort  The port to use if the variable is missing or invalid.
     * @return A valid port number.
     */
    private static int getPortFromEnv(String envVariable, int defaultPort) {
        String portStr = System.getenv(envVariable);
        if (portStr == null) {
            logger.warn("Environment variable {} not set. Using default port {}.", envVariable, defaultPort);
            return defaultPort;
        }
        try {
            return Integer.parseInt(portStr);
        } catch (NumberFormatException e) {
            logger.error(
                    "Invalid port specified in {}: '{}'. Must be a number. Using default port {}.",
                    envVariable, portStr, defaultPort);
            return defaultPort;
        }
    }
}