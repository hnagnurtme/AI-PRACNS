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

    private static final String RL_SERVICE_HOST_DEFAULT = "127.0.0.1";
    private static final int RL_SERVICE_PORT_DEFAULT = 6000;

    public static void main(String[] args) {
        // Set up centralized logging for this simulation run
        String simulationId = String.valueOf(System.currentTimeMillis());
        AppLogger.putMdc("simulationId", simulationId);

        logger.info("=== Starting full SAGIN network simulation ===");
        logger.info("Simulation ID: {}", simulationId);

        // --- 0. Auto-detect LAN IP Address ---
        // Allow override via environment variable for special cases (e.g., Docker, testing)
        String detectedIp = NetworkUtils.getLocalIpAddress();
        String nodeHostIp = System.getenv().getOrDefault("NODE_HOST_IP", detectedIp);
        
        if (!nodeHostIp.equals(detectedIp)) {
            logger.warn("⚠️ NODE_HOST_IP environment variable ({}) overrides auto-detected IP ({})", 
                nodeHostIp, detectedIp);
        } else {
            logger.info("✅ Using auto-detected LAN IP: {}", nodeHostIp);
        }

        // --- 1. Initialize Core Singleton Services ---
        // These services are created ONCE and shared by all nodes.
        logger.info("Initializing core services (singletons)...");
        INodeRepository nodeRepository = new MongoNodeRepository();
        IUserRepository userRepository = new MongoUserRepository();
        INodeService nodeService = new NodeService(nodeRepository);
        DynamicRoutingService routingService = new DynamicRoutingService(nodeRepository, nodeService);

        // Database repositories for packet analytics
        ITwoPacketRepository twoPacketRepository = new MongoTwoPacketRepository();
        IBatchPacketRepository batchPacketRepository = new MongoBatchPacketRepository();
        BatchPacketService batchPacketService = new BatchPacketService(batchPacketRepository, twoPacketRepository);

        // Initialize RL Service with safe port parsing
        String rlHost = System.getenv().getOrDefault("RL_SERVICE_HOST", RL_SERVICE_HOST_DEFAULT);
        int rlPort = getPortFromEnv("RL_SERVICE_PORT", RL_SERVICE_PORT_DEFAULT);
        RLRoutingService rlRoutingService = new RLRoutingService(rlHost, rlPort);
        logger.info("RL Routing Service configured for {}:{}", rlHost, rlPort);

        // Create ONE TCP_Service and inject it everywhere.
        // This service manages the async send/retry queue for ALL nodes.
        TCP_Service tcpService = new TCP_Service(
                nodeRepository,
                nodeService,
                userRepository,
                routingService,
                batchPacketService,
                rlRoutingService);

        // --- 2. Load and Configure Node Settings ---
        Map<String, NodeInfo> nodeInfoMap = nodeRepository.loadAllNodeConfigs();
        logger.info("Loaded {} node configurations.", nodeInfoMap.size());

        // Update IPs in memory first with auto-detected IP.
        logger.info("📝 Updating node IP addresses to {}...", nodeHostIp);
        nodeInfoMap.values().forEach(nodeInfo -> {
            nodeService.updateNodeIpAddress(nodeInfo.getNodeId(), nodeHostIp);
        });

        // Flush all IP updates to the database in ONE operation.
        nodeService.flushToDatabase();
        logger.info("✅ Node IP addresses flushed to database.");

        // --- 3. Launch Node Gateways ---
        // Use a Thread Pool to manage node threads.
        ExecutorService nodeLauncherPool = Executors.newFixedThreadPool(nodeInfoMap.size());

        // Keep a list of gateways for a graceful shutdown.
        List<NodeGateway> runningGateways = new ArrayList<>();

        logger.info("Starting {} NodeGateways...", nodeInfoMap.size());
        for (NodeInfo nodeInfo : nodeInfoMap.values()) {
            // Create ONE new gateway per node.
            // Inject the SINGLETON tcpService into it.
            NodeGateway nodeGateway = new NodeGateway(tcpService);
            runningGateways.add(nodeGateway);

            // Submit the node's main loop to the thread pool.
            nodeLauncherPool.submit(() -> {
                try {
                    // Set up MDC (Mapped Diagnostic Context) for this node's thread
                    AppLogger.putMdc("simulationId", simulationId);
                    AppLogger.putMdc("nodeId", nodeInfo.getNodeId());

                    // Start the gateway. This will throw an IOException if the port is busy.
                    nodeGateway.startListening(nodeInfo, nodeInfo.getCommunication().getPort());

                    logger.info("Node Gateway started for node: {} ({}) on port {}",
                            nodeInfo.getNodeId(), nodeInfo.getNodeType(), nodeInfo.getCommunication().getPort());

                    // This 'join' blocks the pool thread, keeping it alive indefinitely
                    // until the application is interrupted.
                    Thread.currentThread().join();

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    logger.warn("Node {} listener interrupted", nodeInfo.getNodeId(), e);

                } catch (Exception e) {
                    // ✅ --- LOGIC BẮT EXCEPTION VÀ CẬP NHẬT DATABASE --- ✅
                    // If startListening fails (e.g., BindException from IOException),
                    // log it and update the database.
                    logger.error(
                            "[SimulationMain] CRITICAL: Node {} failed to start, likely due to: {}. Marking as UNHEALTHY.",
                            nodeInfo.getNodeId(), e.getMessage());

                    try {
                        nodeService.markNodeAsUnhealthy(nodeInfo.getNodeId());                        
                        logger.warn(
                                "[SimulationMain] Node {} successfully marked as UNHEALTHY in database (logic assumed).", // Sửa: "(logic assumed)" nghĩa là "giả định logic này đúng"
                                nodeInfo.getNodeId());
                    } catch (Exception dbError) {
                        logger.error(
                                "[SimulationMain] FAILED to update database for failed node {}: {}. Network state may be inconsistent.",
                                nodeInfo.getNodeId(), dbError.getMessage(), dbError);
                    }
                    // ✅ --- KẾT THÚC LOGIC ---
                } finally {
                    AppLogger.clearMdc(); // Clean up thread-local logging info
                }
            });
        }

        // --- 4. Register Comprehensive Shutdown Hook ---
        // This hook ensures all services are stopped gracefully.
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("=== SHUTDOWN HOOK: Stopping all services... ===");
            try {
                // 1. Stop Gateways: Refuse new connections
                logger.info("Stopping all NodeGateways ({})...", runningGateways.size());
                for (NodeGateway gateway : runningGateways) {
                    gateway.stopListening();
                }

                // 2. Stop Launcher Pool: Interrupt the 'join' and stop node threads
                logger.info("Shutting down Node Launcher pool...");
                nodeLauncherPool.shutdown();
                if (!nodeLauncherPool.awaitTermination(5, TimeUnit.SECONDS)) {
                    nodeLauncherPool.shutdownNow(); // Force shutdown if threads are stuck
                }

                // 3. Stop Core Services: Stop background processing (e.g., retry scheduler)
                logger.info("Stopping core services (Routing, TCP)...");
                routingService.shutdown();
                tcpService.stop(); // Stops the TCP_Service retry scheduler

                logger.info("=== All services stopped. Shutdown complete. ===");
                AppLogger.clearMdc();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.error("Shutdown hook interrupted.", e);
            }
        }));

        // --- 5. Finalize Initialization ---
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