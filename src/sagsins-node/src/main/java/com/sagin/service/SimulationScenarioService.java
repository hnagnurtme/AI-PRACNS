package com.sagin.service;

import com.sagin.model.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Service to manage and apply simulation scenarios to network operations
 */
public class SimulationScenarioService {
    
    private static final Logger logger = LoggerFactory.getLogger(SimulationScenarioService.class);
    
    private SimulationScenario currentScenario = SimulationScenario.NORMAL;
    private final Random random = new Random();
    
    /**
     * Set the current simulation scenario
     */
    public void setScenario(SimulationScenario scenario) {
        logger.info("🎬 Simulation scenario changed: {} → {}", 
                currentScenario.getDisplayName(), scenario.getDisplayName());
        this.currentScenario = scenario;
    }
    
    /**
     * Get the current simulation scenario
     */
    public SimulationScenario getCurrentScenario() {
        return currentScenario;
    }
    
    /**
     * Apply scenario effects to a node's state
     * Returns modified NodeInfo with scenario effects applied
     */
    public NodeInfo applyScenarioToNode(NodeInfo node) {
        if (currentScenario == SimulationScenario.NORMAL || node == null) {
            return node;
        }
        
        // Apply scenario effects directly to the node
        switch (currentScenario) {
            case WEATHER_EVENT:
                applyWeatherEvent(node);
                break;
            case NODE_OVERLOAD:
                applyNodeOverload(node);
                break;
            case NODE_OFFLINE:
                applyNodeOffline(node);
                break;
            case TRAFFIC_SPIKE:
                applyTrafficSpike(node);
                break;
            default:
                break;
        }
        
        return node;
    }
    
    /**
     * Check if packet should be dropped based on current scenario
     */
    public boolean shouldDropPacket(Packet packet, NodeInfo node) {
        switch (currentScenario) {
            case NODE_OFFLINE:
                // 50% chance to drop packets when node is marked as offline
                if (!node.isOperational()) {
                    return random.nextDouble() < 0.5;
                }
                break;
            case NODE_OVERLOAD:
                // Drop packets if queue is over 90% full
                if (node.getCurrentPacketCount() >= node.getPacketBufferCapacity() * 0.9) {
                    return random.nextDouble() < 0.3;
                }
                break;
            case WEATHER_EVENT:
                // Increased packet loss during bad weather
                if (node.getWeather() == WeatherCondition.STORM || 
                    node.getWeather() == WeatherCondition.SEVERE_STORM ||
                    node.getWeather() == WeatherCondition.RAIN) {
                    return random.nextDouble() < 0.15;
                }
                break;
            case TTL_EXPIRED:
                // Check if packet TTL has expired
                if (packet.getTTL() <= 0) {
                    return true;
                }
                break;
            default:
                break;
        }
        return false;
    }
    
    /**
     * Get drop reason based on scenario
     */
    public String getDropReason(Packet packet, NodeInfo node) {
        switch (currentScenario) {
            case NODE_OFFLINE:
                return "Node offline - link failure";
            case NODE_OVERLOAD:
                return String.format("Node overload - queue %.0f%% full", 
                    (double) node.getCurrentPacketCount() / node.getPacketBufferCapacity() * 100);
            case WEATHER_EVENT:
                return "Weather event - poor transmission conditions (" + node.getWeather() + ")";
            case TTL_EXPIRED:
                return "TTL expired - packet lifetime exceeded";
            case TRAFFIC_SPIKE:
                return "Traffic spike - congestion detected";
            default:
                return "Unknown reason";
        }
    }
    
    /**
     * Calculate additional latency based on scenario
     */
    public double getScenarioLatencyMs(NodeInfo node) {
        switch (currentScenario) {
            case WEATHER_EVENT:
                // Add 20-50ms latency for bad weather
                return 20 + random.nextDouble() * 30;
            case NODE_OVERLOAD:
                // Add latency based on queue size
                double queueFillPercent = (double) node.getCurrentPacketCount() / node.getPacketBufferCapacity();
                return queueFillPercent * 100; // Up to 100ms extra
            case TRAFFIC_SPIKE:
                // Add 10-30ms latency during traffic spike
                return 10 + random.nextDouble() * 20;
            default:
                return 0.0;
        }
    }
    
    /**
     * Get current node load percentage for visualization
     */
    public double getNodeLoadPercent(NodeInfo node) {
        return (double) node.getCurrentPacketCount() / node.getPacketBufferCapacity() * 100;
    }
    
    // Private helper methods to apply specific scenario effects
    
    private void applyWeatherEvent(NodeInfo node) {
        // Randomly set bad weather conditions
        if (random.nextDouble() < 0.3) {
            node.setWeather(WeatherCondition.STORM);
        } else if (random.nextDouble() < 0.5) {
            node.setWeather(WeatherCondition.RAIN);
        } else if (random.nextDouble() < 0.2) {
            node.setWeather(WeatherCondition.SEVERE_STORM);
        }
    }
    
    private void applyNodeOverload(NodeInfo node) {
        // Increase queue size to simulate overload
        int capacity = node.getPacketBufferCapacity();
        int overloadCount = (int) (capacity * (0.7 + random.nextDouble() * 0.25)); // 70-95% full
        node.setCurrentPacketCount(overloadCount);
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
    }
    
    /**
     * Generate random simulation scenario (for testing)
     */
    public SimulationScenario getRandomScenario() {
        SimulationScenario[] scenarios = SimulationScenario.values();
        return scenarios[ThreadLocalRandom.current().nextInt(scenarios.length)];
    }
}
