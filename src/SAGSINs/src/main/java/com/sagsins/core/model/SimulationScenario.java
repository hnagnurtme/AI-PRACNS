package com.sagsins.core.model;

/**
 * Enum representing different simulation scenarios that can affect network behavior
 */
public enum SimulationScenario {
    NORMAL("Normal", "Standard network operation with no special conditions"),
    WEATHER_EVENT("Weather Event", "Bad weather affecting transmission quality"),
    NODE_OVERLOAD("Node Overload", "Node experiencing high load and queue congestion"),
    NODE_OFFLINE("Node Offline", "Node temporarily offline or unreachable"),
    TRAFFIC_SPIKE("Traffic Spike", "Sudden burst of traffic causing congestion"),
    TTL_EXPIRED("TTL Expired", "Packet dropped due to time-to-live expiration");
    
    private final String displayName;
    private final String description;
    
    SimulationScenario(String displayName, String description) {
        this.displayName = displayName;
        this.description = description;
    }
    
    public String getDisplayName() {
        return displayName;
    }
    
    public String getDescription() {
        return description;
    }
}
