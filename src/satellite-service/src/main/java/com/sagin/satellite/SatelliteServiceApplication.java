package com.sagin.satellite;

import com.sagin.satellite.config.SatelliteConfiguration;
import com.sagin.satellite.controller.*;
import com.sagin.satellite.model.*;
import com.sagin.satellite.util.ProjectConstant;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Scanner;

/**
 * Main application class cho Satellite Service
 * Khởi tạo và chạy satellite service với interactive console
 */
public class SatelliteServiceApplication {

    private static final Logger logger = LoggerFactory.getLogger(SatelliteServiceApplication.class);
    
    private SatelliteConfiguration config;
    private boolean running = false;

    public static void main(String[] args) {
        logger.info("Starting Satellite Service Application...");
        
        SatelliteServiceApplication app = new SatelliteServiceApplication();
        app.start();
    }

    public void start() {
        try {
            // Load configuration
            String satelliteId = System.getProperty("satellite.id", "SAT_001");
            String nodeId = System.getProperty("satellite.node.id", "SAT_001");
            
            logger.info("Initializing satellite: {} with node ID: {}", satelliteId, nodeId);
            
            // Initialize configuration
            config = new SatelliteConfiguration(satelliteId, nodeId);
            
            // Register this satellite as a node in the network
            registerSatelliteNode(satelliteId);
            
            // Start the service
            running = true;
            logger.info("Satellite Service started successfully!");
            
            // Start interactive console
            startInteractiveConsole();
            
        } catch (Exception e) {
            logger.error("Failed to start Satellite Service: {}", e.getMessage(), e);
            System.exit(1);
        }
    }

    /**
     * Đăng ký satellite node vào network topology
     */
    private void registerSatelliteNode(String satelliteId) {
        try {
            // Tạo node info cho satellite này
            NodeInfo nodeInfo = new NodeInfo();
            nodeInfo.setNodeId(satelliteId);
            nodeInfo.setNodeType(ProjectConstant.NODE_TYPE_SATELLITE);
            
            // Set default position (có thể load từ config)
            Geo3D position = new Geo3D();
            position.setLatitude(getRandomLatitude());
            position.setLongitude(getRandomLongitude());
            position.setAltitude(550.0); // 550km altitude for LEO satellite
            nodeInfo.setPosition(position);
            
            // Set default capabilities
            nodeInfo.setLinkAvailable(true);
            nodeInfo.setBandwidth(100.0); // 100 Mbps
            nodeInfo.setLatencyMs(50.0);
            nodeInfo.setPacketLossRate(0.01); // 1%
            nodeInfo.setBufferSize(0);
            nodeInfo.setThroughput(0.0);
            nodeInfo.setLastUpdated(System.currentTimeMillis());
            
            // Register với network topology service
            config.getNetworkTopologyService().registerNode(nodeInfo);
            
            logger.info("Satellite node {} registered successfully at position: lat={}, lon={}, alt={}km",
                       satelliteId, position.getLatitude(), position.getLongitude(), position.getAltitude());
            
        } catch (Exception e) {
            logger.error("Failed to register satellite node: {}", e.getMessage(), e);
        }
    }

    /**
     * Start interactive console cho testing và debugging
     */
    private void startInteractiveConsole() {
        Scanner scanner = new Scanner(System.in);
        
        printMenu();
        
        while (running) {
            System.out.print("satellite> ");
            String input = scanner.nextLine().trim();
            
            if (input.isEmpty()) continue;
            
            String[] parts = input.split("\\s+");
            String command = parts[0].toLowerCase();
            
            try {
                switch (command) {
                    case "help":
                    case "h":
                        printMenu();
                        break;
                        
                    case "status":
                        showStatus();
                        break;
                        
                    case "health":
                        showHealth();
                        break;
                        
                    case "metrics":
                        showMetrics();
                        break;
                        
                    case "network":
                        showNetwork();
                        break;
                        
                    case "send":
                        if (parts.length >= 3) {
                            sendTestPacket(parts[1], parts[2]);
                        } else {
                            System.out.println("Usage: send <destinationUser> <message>");
                        }
                        break;
                        
                    case "topology":
                        updateTopology();
                        break;
                        
                    case "reset":
                        resetMetrics();
                        break;
                        
                    case "quit":
                    case "exit":
                    case "q":
                        shutdown();
                        break;
                        
                    default:
                        System.out.println("Unknown command: " + command + ". Type 'help' for available commands.");
                }
                
            } catch (Exception e) {
                logger.error("Error executing command '{}': {}", command, e.getMessage());
                System.out.println("Error: " + e.getMessage());
            }
        }
        
        scanner.close();
    }

    private void printMenu() {
        System.out.println("\n=== Satellite Service Console ===");
        System.out.println("Available commands:");
        System.out.println("  help (h)                  - Show this menu");
        System.out.println("  status                    - Show satellite status");
        System.out.println("  health                    - Show health status");
        System.out.println("  metrics                   - Show detailed metrics");
        System.out.println("  network                   - Show network topology");
        System.out.println("  send <user> <message>     - Send test packet");
        System.out.println("  topology                  - Update network topology");
        System.out.println("  reset                     - Reset metrics");
        System.out.println("  quit (q)                  - Shutdown satellite");
        System.out.println("================================\n");
    }

    private void showStatus() {
        SatelliteStatusController controller = config.getSatelliteStatusController();
        SatelliteStatus status = controller.getStatus();
        
        System.out.println("\n=== Satellite Status ===");
        System.out.println("Satellite ID: " + status.getSatelliteId());
        System.out.println("Buffer Size: " + status.getBufferSize());
        System.out.println("Throughput: " + String.format("%.2f Mbps", status.getThroughput()));
        System.out.println("Average Latency: " + String.format("%.2f ms", status.getAverageLatencyMs()));
        System.out.println("Packet Loss Rate: " + String.format("%.2f%%", status.getPacketLossRate() * 100));
        System.out.println("Last Updated: " + new java.util.Date(status.getLastUpdated()));
        System.out.println("========================\n");
    }

    private void showHealth() {
        SatelliteStatusController controller = config.getSatelliteStatusController();
        java.util.Map<String, Object> health = controller.getHealth();
        
        System.out.println("\n=== Health Status ===");
        health.forEach((key, value) -> {
            System.out.println(key + ": " + value);
        });
        System.out.println("=====================\n");
    }

    private void showMetrics() {
        SatelliteStatusController controller = config.getSatelliteStatusController();
        java.util.Map<String, Object> metrics = controller.getMetrics();
        
        System.out.println("\n=== Detailed Metrics ===");
        metrics.forEach((key, value) -> {
            System.out.println(key + ": " + value);
        });
        System.out.println("========================\n");
    }

    private void showNetwork() {
        java.util.Map<String, Object> network = config.getNetworkTopologyService().getNetworkSnapshot();
        
        System.out.println("\n=== Network Topology ===");
        network.forEach((key, value) -> {
            if (!"nodes".equals(key) && !"linkMetrics".equals(key)) {
                System.out.println(key + ": " + value);
            }
        });
        System.out.println("========================\n");
    }

    private void sendTestPacket(String destinationUser, String message) {
        try {
            SatelliteController controller = config.getSatelliteController();
            
            // Tạo test packet
            Packet packet = new Packet();
            packet.setPacketId("TEST_" + System.currentTimeMillis());
            packet.setSourceUserId("TEST_USER");
            packet.setDestinationUserId(destinationUser);
            packet.setMessage(message);
            packet.setTimestamp(System.currentTimeMillis());
            packet.setTTL(10);
            packet.setCurrentNode(config.getNodeId());
            packet.setPayloadSize(message.length());
            packet.setPriority(1);
            
            controller.receivePacket(packet);
            System.out.println("Test packet sent: " + packet.getPacketId());
            
        } catch (Exception e) {
            System.out.println("Failed to send test packet: " + e.getMessage());
            logger.error("Error sending test packet", e);
        }
    }

    private void updateTopology() {
        try {
            NetworkController controller = config.getNetworkController();
            java.util.Map<String, Object> result = controller.updateTopology(5000.0);
            
            System.out.println("Topology updated:");
            result.forEach((key, value) -> {
                System.out.println("  " + key + ": " + value);
            });
            
        } catch (Exception e) {
            System.out.println("Failed to update topology: " + e.getMessage());
            logger.error("Error updating topology", e);
        }
    }

    private void resetMetrics() {
        try {
            SatelliteStatusController controller = config.getSatelliteStatusController();
            java.util.Map<String, Object> result = controller.resetMetrics();
            
            System.out.println("Metrics reset: " + result.get("message"));
            
        } catch (Exception e) {
            System.out.println("Failed to reset metrics: " + e.getMessage());
            logger.error("Error resetting metrics", e);
        }
    }

    private void shutdown() {
        logger.info("Shutting down Satellite Service...");
        running = false;
        
        try {
            // Cleanup resources
            if (config != null && config.getTcpSender() instanceof com.sagin.satellite.service.implement.TcpSender) {
                ((com.sagin.satellite.service.implement.TcpSender) config.getTcpSender()).shutdown();
            }
            
            logger.info("Satellite Service shutdown completed");
            System.out.println("Goodbye!");
            
        } catch (Exception e) {
            logger.error("Error during shutdown: {}", e.getMessage(), e);
        }
    }

    // Helper methods to generate random positions
    private double getRandomLatitude() {
        return -90 + Math.random() * 180; // -90 to +90
    }

    private double getRandomLongitude() {
        return -180 + Math.random() * 360; // -180 to +180
    }
}