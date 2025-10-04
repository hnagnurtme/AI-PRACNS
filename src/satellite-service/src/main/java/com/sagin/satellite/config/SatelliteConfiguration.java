package com.sagin.satellite.config;

import com.sagin.satellite.service.*;
import com.sagin.satellite.service.implement.*;
import com.sagin.satellite.controller.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Configuration class để khởi tạo và wire các services và controllers
 */
public class SatelliteConfiguration {

    private static final Logger logger = LoggerFactory.getLogger(SatelliteConfiguration.class);
    
    // Service instances
    private IBufferManager bufferManager;
    private ILinkManagementService linkManagementService;
    private INetworkTopologyService networkTopologyService;
    private IRoutingService routingService;
    private IPacketForwardingService packetForwardingService;
    private ISatelliteStatusService satelliteStatusService;
    private ISatelliteService satelliteService;
    private ITcpSender tcpSender;
    
    // Controller instances
    private SatelliteController satelliteController;
    private SatelliteStatusController satelliteStatusController;
    private NetworkController networkController;
    
    // Configuration
    private final String satelliteId;
    private final String nodeId;
    
    public SatelliteConfiguration(String satelliteId, String nodeId) {
        this.satelliteId = satelliteId;
        this.nodeId = nodeId;
        initializeServices();
        initializeControllers();
        logger.info("Satellite configuration initialized for satellite: {}, node: {}", 
                   satelliteId, nodeId);
    }

    /**
     * Khởi tạo tất cả services
     */
    private void initializeServices() {
        logger.info("Initializing services...");
        
        // Core services - sử dụng constructor với parameters
        TcpSender tcpSenderImpl = new TcpSender();
        tcpSender = tcpSenderImpl;
        bufferManager = new BufferManagerImpl(1000, tcpSenderImpl, 1000, 3);
        
        // Management services
        linkManagementService = new LinkManagementService();
        networkTopologyService = new NetworkTopologyService(linkManagementService);
        routingService = new RoutingService();
        
        // Status service
        satelliteStatusService = new SatelliteStatusService(satelliteId);
        
        // Forwarding service
        packetForwardingService = new PacketForwardingService(
            routingService, tcpSender, nodeId);
        
        // Main satellite service
        satelliteService = new SatelliteService(
            bufferManager, packetForwardingService, satelliteStatusService);
        
        logger.info("All services initialized successfully");
    }

    /**
     * Khởi tạo tất cả controllers
     */
    private void initializeControllers() {
        logger.info("Initializing controllers...");
        
        satelliteController = new SatelliteController(satelliteService);
        satelliteStatusController = new SatelliteStatusController(
            satelliteStatusService, networkTopologyService);
        networkController = new NetworkController(
            networkTopologyService, linkManagementService, routingService);
        
        logger.info("All controllers initialized successfully");
    }

    // Getters for services
    public IBufferManager getBufferManager() { return bufferManager; }
    public ILinkManagementService getLinkManagementService() { return linkManagementService; }
    public INetworkTopologyService getNetworkTopologyService() { return networkTopologyService; }
    public IRoutingService getRoutingService() { return routingService; }
    public IPacketForwardingService getPacketForwardingService() { return packetForwardingService; }
    public ISatelliteStatusService getSatelliteStatusService() { return satelliteStatusService; }
    public ISatelliteService getSatelliteService() { return satelliteService; }
    public ITcpSender getTcpSender() { return tcpSender; }

    // Getters for controllers
    public SatelliteController getSatelliteController() { return satelliteController; }
    public SatelliteStatusController getSatelliteStatusController() { return satelliteStatusController; }
    public NetworkController getNetworkController() { return networkController; }
    
    // Configuration getters
    public String getSatelliteId() { return satelliteId; }
    public String getNodeId() { return nodeId; }
}