package com.sagsins.core.service;

import com.sagsins.core.model.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Service to manage simulation scenario templates and apply them to nodes
 */
@Service
public class SimulationScenarioFactoryService {
    
    private static final Logger logger = LoggerFactory.getLogger(SimulationScenarioFactoryService.class);
    
    private String currentScenario = "NORMAL";
    private final Random random = new Random();
    
    /**
     * Set the current simulation scenario
     */
    public void setScenario(String scenario) {
        logger.info("🎬 Simulation scenario changed: {} → {}", currentScenario, scenario);
        this.currentScenario = scenario;
    }
    
    /**
     * Get the current simulation scenario
     */
    public String getCurrentScenario() {
        return currentScenario;
    }
    
    /**
     * Apply scenario effects to a node based on scenario name
     */
    public NodeInfo applyScenarioToNode(NodeInfo node, String scenarioName) {
        if ("NORMAL".equals(scenarioName) || node == null) {
            return node;
        }
        
        logger.debug("Applying scenario {} to node {}", scenarioName, node.getNodeId());
        
        switch (scenarioName) {
            case "WEATHER_EVENT":
                applyWeatherEvent(node);
                break;
            case "NODE_OVERLOAD":
                applyNodeOverload(node);
                break;
            case "NODE_OFFLINE":
                applyNodeOffline(node);
                break;
            case "TRAFFIC_SPIKE":
                applyTrafficSpike(node);
                break;
            default:
                logger.warn("Unknown scenario: {}", scenarioName);
                break;
        }
        
        node.setLastUpdated(Instant.now());
        return node;
    }
    
    /**
     * Apply current scenario to multiple nodes
     */
    public List<NodeInfo> applyCurrentScenarioToNodes(List<NodeInfo> nodes) {
        nodes.forEach(node -> applyScenarioToNode(node, currentScenario));
        return nodes;
    }
    
    /**
     * Get list of available scenario names
     */
    public List<String> getAvailableScenarios() {
        return Arrays.asList(
            "NORMAL",
            "WEATHER_EVENT",
            "NODE_OVERLOAD",
            "NODE_OFFLINE",
            "TRAFFIC_SPIKE",
            "TTL_EXPIRED"
        );
    }
    
    /**
     * Get scenario description
     */
    public Map<String, String> getScenarioInfo(String scenarioName) {
        Map<String, String> info = new HashMap<>();
        info.put("name", scenarioName);
        
        switch (scenarioName) {
            case "NORMAL":
                info.put("displayName", "Normal");
                info.put("description", "Standard network operation with no special conditions");
                break;
            case "WEATHER_EVENT":
                info.put("displayName", "Weather Event");
                info.put("description", "Bad weather affecting transmission quality");
                break;
            case "NODE_OVERLOAD":
                info.put("displayName", "Node Overload");
                info.put("description", "Node experiencing high load and queue congestion");
                break;
            case "NODE_OFFLINE":
                info.put("displayName", "Node Offline");
                info.put("description", "Node temporarily offline or unreachable");
                break;
            case "TRAFFIC_SPIKE":
                info.put("displayName", "Traffic Spike");
                info.put("description", "Sudden burst of traffic causing congestion");
                break;
            case "TTL_EXPIRED":
                info.put("displayName", "TTL Expired");
                info.put("description", "Packet dropped due to time-to-live expiration");
                break;
            default:
                info.put("displayName", scenarioName);
                info.put("description", "Unknown scenario");
                break;
        }
        
        return info;
    }
    
    // Private helper methods to apply specific scenario effects
    
    private void applyWeatherEvent(NodeInfo node) {
        // Randomly set bad weather conditions
        double rand = random.nextDouble();
        if (rand < 0.3) {
            node.setWeather(WeatherCondition.STORM);
        } else if (rand < 0.5) {
            node.setWeather(WeatherCondition.RAIN);
        } else if (rand < 0.2) {
            node.setWeather(WeatherCondition.SEVERE_STORM);
        }
        
        // Increase packet loss rate
        node.setPacketLossRate(node.getPacketLossRate() + 0.05 + random.nextDouble() * 0.1);
        
        // Increase processing delay
        node.setNodeProcessingDelayMs(node.getNodeProcessingDelayMs() + 20 + random.nextDouble() * 30);
    }
    
    private void applyNodeOverload(NodeInfo node) {
        // Increase queue size to simulate overload
        int capacity = node.getPacketBufferCapacity();
        int overloadCount = (int) (capacity * (0.7 + random.nextDouble() * 0.25)); // 70-95% full
        node.setCurrentPacketCount(overloadCount);
        
        // Increase resource utilization
        node.setResourceUtilization(Math.min(95.0, 70.0 + random.nextDouble() * 25.0));
        
        // Increase processing delay
        node.setNodeProcessingDelayMs(node.getNodeProcessingDelayMs() + 50 + random.nextDouble() * 50);
    }
    
    private void applyNodeOffline(NodeInfo node) {
        // Randomly mark some nodes as offline
        if (random.nextDouble() < 0.2) { // 20% of nodes go offline
            node.setOperational(false);
        }
    }
    
    private void applyTrafficSpike(NodeInfo node) {
        // Simulate traffic spike by increasing queue size
        int capacity = node.getPacketBufferCapacity();
        int spikeCount = (int) (capacity * (0.5 + random.nextDouble() * 0.4)); // 50-90% full
        node.setCurrentPacketCount(Math.max(node.getCurrentPacketCount(), spikeCount));
        
        // Increase resource utilization
        node.setResourceUtilization(Math.min(90.0, 60.0 + random.nextDouble() * 30.0));
    }
}
