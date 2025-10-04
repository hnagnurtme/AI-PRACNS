package com.sagin.satellite.controller;

import com.sagin.satellite.model.SatelliteStatus;
import com.sagin.satellite.service.ISatelliteStatusService;
import com.sagin.satellite.service.INetworkTopologyService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * REST Controller cho satellite status và monitoring
 */
public class SatelliteStatusController extends BaseController {

    private static final Logger logger = LoggerFactory.getLogger(SatelliteStatusController.class);
    
    private final ISatelliteStatusService statusService;
    private final INetworkTopologyService topologyService;

    public SatelliteStatusController(ISatelliteStatusService statusService,
                                   INetworkTopologyService topologyService) {
        this.statusService = statusService;
        this.topologyService = topologyService;
    }

    /**
     * GET /api/satellite/status
     * Lấy trạng thái hiện tại của vệ tinh
     */
    public SatelliteStatus getStatus() {
        logger.debug("Getting satellite status");
        try {
            return statusService.getCurrentStatus();
        } catch (Exception e) {
            logger.error("Error getting satellite status: {}", e.getMessage());
            throw new RuntimeException("Failed to get satellite status", e);
        }
    }

    /**
     * GET /api/satellite/health
     * Kiểm tra tình trạng sức khỏe của vệ tinh
     */
    public Map<String, Object> getHealth() {
        logger.debug("Getting satellite health status");
        try {
            Map<String, Object> health = Map.of(
                "healthy", statusService.isHealthy(),
                "currentLoad", statusService.getCurrentLoad(),
                "remainingBufferCapacity", statusService.getRemainingBufferCapacity(),
                "timestamp", System.currentTimeMillis()
            );
            return health;
        } catch (Exception e) {
            logger.error("Error getting health status: {}", e.getMessage());
            throw new RuntimeException("Failed to get health status", e);
        }
    }

    /**
     * GET /api/satellite/metrics
     * Lấy metrics chi tiết
     */
    public Map<String, Object> getMetrics() {
        logger.debug("Getting detailed metrics");
        try {
            return statusService.getDetailedMetrics();
        } catch (Exception e) {
            logger.error("Error getting metrics: {}", e.getMessage());
            throw new RuntimeException("Failed to get metrics", e);
        }
    }

    /**
     * POST /api/satellite/metrics/reset
     * Reset tất cả metrics
     */
    public Map<String, Object> resetMetrics() {
        logger.info("Resetting satellite metrics");
        try {
            statusService.resetMetrics();
            return Map.of(
                "success", true,
                "message", "Metrics reset successfully",
                "timestamp", System.currentTimeMillis()
            );
        } catch (Exception e) {
            logger.error("Error resetting metrics: {}", e.getMessage());
            throw new RuntimeException("Failed to reset metrics", e);
        }
    }

    /**
     * GET /api/satellite/network
     * Lấy thông tin network topology
     */
    public Map<String, Object> getNetworkInfo() {
        logger.debug("Getting network topology information");
        try {
            return topologyService.getNetworkSnapshot();
        } catch (Exception e) {
            logger.error("Error getting network info: {}", e.getMessage());
            throw new RuntimeException("Failed to get network info", e);
        }
    }
}