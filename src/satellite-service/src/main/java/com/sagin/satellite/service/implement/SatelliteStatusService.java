package com.sagin.satellite.service.implement;

import com.sagin.satellite.model.SatelliteStatus;
import com.sagin.satellite.model.NodeInfo;
import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.ISatelliteStatusService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * SatelliteStatusService quản lý trạng thái real-time của vệ tinh
 */
public class SatelliteStatusService implements ISatelliteStatusService {

    private static final Logger logger = LoggerFactory.getLogger(SatelliteStatusService.class);
    
    private final String satelliteId;
    private SatelliteStatus currentStatus;
    
    // Configuration
    private static final int MAX_BUFFER_SIZE = 1000;
    private static final double HEALTHY_THROUGHPUT_THRESHOLD = 50.0; // Mbps
    private static final double HEALTHY_LATENCY_THRESHOLD = 500.0; // ms
    private static final double HEALTHY_LOSS_RATE_THRESHOLD = 0.05; // 5%
    
    // Metrics tracking
    private final AtomicLong totalPacketsProcessed = new AtomicLong(0);
    private final AtomicLong totalPacketsDropped = new AtomicLong(0);
    private final Queue<Double> latencyHistory = new LinkedList<>();
    private final Queue<Double> throughputHistory = new LinkedList<>();
    private static final int HISTORY_SIZE = 100;

    public SatelliteStatusService(String satelliteId) {
        this.satelliteId = satelliteId;
        initializeStatus();
    }

    @Override
    public SatelliteStatus getCurrentStatus() {
        // Update real-time computed fields
        updateComputedMetrics();
        return currentStatus;
    }

    @Override
    public void updateNodeInfo(NodeInfo nodeInfo) {
        logger.debug("Updating node info for satellite {}", satelliteId);
        currentStatus.setNodeInfo(nodeInfo);
        currentStatus.setLastUpdated(System.currentTimeMillis());
    }

    @Override
    public void updateBufferStatus(List<Packet> buffer) {
        logger.debug("Updating buffer status: {} packets", buffer.size());
        
        currentStatus.setBuffer(new ArrayList<>(buffer));
        currentStatus.setBufferSize(buffer.size());
        currentStatus.setLastUpdated(System.currentTimeMillis());
        
        // Update node info buffer size if available
        if (currentStatus.getNodeInfo() != null) {
            currentStatus.getNodeInfo().setBufferSize(buffer.size());
        }
    }

    @Override
    public void updatePerformanceMetrics(double throughput, double averageLatency, double packetLossRate) {
        logger.debug("Updating performance metrics: throughput={}Mbps, latency={}ms, loss={}%", 
                    throughput, averageLatency, packetLossRate * 100);
        
        currentStatus.setThroughput(throughput);
        currentStatus.setAverageLatencyMs(averageLatency);
        currentStatus.setPacketLossRate(packetLossRate);
        currentStatus.setLastUpdated(System.currentTimeMillis());
        
        // Update historical data
        updateHistoricalData(throughput, averageLatency);
        
        // Update node info if available
        if (currentStatus.getNodeInfo() != null) {
            currentStatus.getNodeInfo().updateMetrics(
                throughput,
                averageLatency,
                packetLossRate,
                currentStatus.getBufferSize(),
                throughput
            );
        }
    }

    @Override
    public boolean isHealthy() {
        SatelliteStatus status = getCurrentStatus();
        
        // Check buffer capacity
        boolean bufferHealthy = getRemainingBufferCapacity() > 0;
        
        // Check performance metrics
        boolean performanceHealthy = 
            status.getThroughput() >= HEALTHY_THROUGHPUT_THRESHOLD &&
            status.getAverageLatencyMs() <= HEALTHY_LATENCY_THRESHOLD &&
            status.getPacketLossRate() <= HEALTHY_LOSS_RATE_THRESHOLD;
        
        // Check node availability
        boolean nodeHealthy = status.getNodeInfo() != null && 
                             status.getNodeInfo().isHealthy();
        
        boolean healthy = bufferHealthy && performanceHealthy && nodeHealthy;
        
        logger.debug("Health check for {}: buffer={}, performance={}, node={}, overall={}", 
                    satelliteId, bufferHealthy, performanceHealthy, nodeHealthy, healthy);
        
        return healthy;
    }

    @Override
    public int getRemainingBufferCapacity() {
        return MAX_BUFFER_SIZE - currentStatus.getBufferSize();
    }

    @Override
    public double getCurrentLoad() {
        double bufferLoad = (double) currentStatus.getBufferSize() / MAX_BUFFER_SIZE;
        double throughputLoad = Math.min(1.0, currentStatus.getThroughput() / (HEALTHY_THROUGHPUT_THRESHOLD * 2));
        
        // Weighted average
        return (bufferLoad * 0.6) + (throughputLoad * 0.4);
    }

    @Override
    public Map<String, Object> getDetailedMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        
        SatelliteStatus status = getCurrentStatus();
        
        // Basic metrics
        metrics.put("satelliteId", satelliteId);
        metrics.put("bufferSize", status.getBufferSize());
        metrics.put("maxBufferSize", MAX_BUFFER_SIZE);
        metrics.put("remainingCapacity", getRemainingBufferCapacity());
        metrics.put("currentLoad", getCurrentLoad());
        metrics.put("isHealthy", isHealthy());
        
        // Performance metrics
        metrics.put("throughput", status.getThroughput());
        metrics.put("averageLatency", status.getAverageLatencyMs());
        metrics.put("packetLossRate", status.getPacketLossRate());
        
        // Historical averages
        metrics.put("avgThroughputLast100", calculateAverageThroughput());
        metrics.put("avgLatencyLast100", calculateAverageLatency());
        
        // Counters
        metrics.put("totalPacketsProcessed", totalPacketsProcessed.get());
        metrics.put("totalPacketsDropped", totalPacketsDropped.get());
        
        // Uptime
        metrics.put("lastUpdated", status.getLastUpdated());
        
        // Node position (if available)
        if (status.getNodeInfo() != null && status.getNodeInfo().getPosition() != null) {
            metrics.put("position", Map.of(
                "latitude", status.getNodeInfo().getPosition().getLatitude(),
                "longitude", status.getNodeInfo().getPosition().getLongitude(),
                "altitude", status.getNodeInfo().getPosition().getAltitude()
            ));
        }
        
        return metrics;
    }

    @Override
    public void resetMetrics() {
        logger.info("Resetting metrics for satellite {}", satelliteId);
        
        totalPacketsProcessed.set(0);
        totalPacketsDropped.set(0);
        latencyHistory.clear();
        throughputHistory.clear();
        
        // Reset status to defaults
        initializeStatus();
    }

    /**
     * Khởi tạo trạng thái ban đầu
     */
    private void initializeStatus() {
        currentStatus = new SatelliteStatus();
        currentStatus.setSatelliteId(satelliteId);
        currentStatus.setBuffer(new ArrayList<>());
        currentStatus.setBufferSize(0);
        currentStatus.setThroughput(0.0);
        currentStatus.setAverageLatencyMs(0.0);
        currentStatus.setPacketLossRate(0.0);
        currentStatus.setLastUpdated(System.currentTimeMillis());
        
        logger.info("Initialized status for satellite {}", satelliteId);
    }

    /**
     * Cập nhật dữ liệu lịch sử
     */
    private void updateHistoricalData(double throughput, double latency) {
        // Add to history
        throughputHistory.offer(throughput);
        latencyHistory.offer(latency);
        
        // Maintain history size
        while (throughputHistory.size() > HISTORY_SIZE) {
            throughputHistory.poll();
        }
        while (latencyHistory.size() > HISTORY_SIZE) {
            latencyHistory.poll();
        }
    }

    /**
     * Tính throughput trung bình
     */
    private double calculateAverageThroughput() {
        return throughputHistory.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);
    }

    /**
     * Tính latency trung bình
     */
    private double calculateAverageLatency() {
        return latencyHistory.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);
    }

    /**
     * Cập nhật các metrics được tính toán
     */
    private void updateComputedMetrics() {
        // This could include real-time calculations based on current state
        // For now, we'll just update the timestamp
        currentStatus.setLastUpdated(System.currentTimeMillis());
    }

    /**
     * Increment packet processed counter
     */
    public void incrementPacketsProcessed() {
        totalPacketsProcessed.incrementAndGet();
    }

    /**
     * Increment packet dropped counter
     */
    public void incrementPacketsDropped() {
        totalPacketsDropped.incrementAndGet();
    }
}