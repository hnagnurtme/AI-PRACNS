package com.sagsins.core.DTOs;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.sagsins.core.model.*;
import lombok.*;

import jakarta.validation.constraints.*;
import java.time.Instant;

/**
 * Data Transfer Object for Node Information
 * Used for API requests and responses
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString
@JsonInclude(JsonInclude.Include.NON_NULL)
public class NodeDTO {

    private String id;

    @NotBlank(message = "Node ID is required")
    @Size(min = 3, max = 50, message = "Node ID must be between 3 and 50 characters")
    private String nodeId;

    @NotBlank(message = "Node name is required")
    @Size(min = 3, max = 100, message = "Node name must be between 3 and 100 characters")
    private String nodeName;

    @NotNull(message = "Node type is required")
    private NodeType nodeType;

    @NotNull(message = "Position is required")
    private Position position;

    private Orbit orbit;
    
    private Velocity velocity;

    private Communication communication;

    @NotNull(message = "Operational status is required")
    private Boolean isOperational;

    @Min(value = 0, message = "Battery charge must be at least 0")
    @Max(value = 100, message = "Battery charge must not exceed 100")
    private Double batteryChargePercent;

    @Min(value = 0, message = "Processing delay must be non-negative")
    private Double nodeProcessingDelayMs;

    @Min(value = 0, message = "Packet loss rate must be at least 0")
    @Max(value = 1, message = "Packet loss rate must not exceed 1")
    private Double packetLossRate;

    @Min(value = 0, message = "Resource utilization must be at least 0")
    @Max(value = 1, message = "Resource utilization must not exceed 1")
    private Double resourceUtilization;

    @Min(value = 0, message = "Packet buffer capacity must be non-negative")
    private Integer packetBufferCapacity;

    @Min(value = 0, message = "Current packet count must be non-negative")
    private Integer currentPacketCount;

    private WeatherCondition weather;

    private Instant lastUpdated;

    @NotBlank(message = "Host is required")
    private String host;

    @Min(value = 1, message = "Port must be at least 1")
    @Max(value = 65535, message = "Port must not exceed 65535")
    private Integer port;

    /**
     * Computed field to check if node is healthy
     */
    @JsonProperty(value = "healthy", access = JsonProperty.Access.READ_ONLY)
    public Boolean isHealthy() {
        if (isOperational == null || batteryChargePercent == null || 
            packetBufferCapacity == null || currentPacketCount == null || weather == null) {
            return null;
        }

        final double MIN_POWER = 10.0;
        final double MAX_BUFFER_LOAD_RATIO = 0.8;
        
        double bufferLoadRatio = (packetBufferCapacity > 0)
            ? (double) currentPacketCount / packetBufferCapacity
            : 0.0;

        return isOperational
            && batteryChargePercent > MIN_POWER
            && bufferLoadRatio <= MAX_BUFFER_LOAD_RATIO
            && weather != WeatherCondition.SEVERE_STORM;
    }

    /**
     * Convert NodeInfo entity to NodeDTO
     */
    public static NodeDTO fromEntity(NodeInfo nodeInfo) {
        if (nodeInfo == null) {
            return null;
        }

        return NodeDTO.builder()
            .id(nodeInfo.getId())
            .nodeId(nodeInfo.getNodeId())
            .nodeName(nodeInfo.getNodeName())
            .nodeType(nodeInfo.getNodeType())
            .position(nodeInfo.getPosition())
            .orbit(nodeInfo.getOrbit())
            .velocity(nodeInfo.getVelocity())
            .communication(nodeInfo.getCommunication())
            .isOperational(nodeInfo.isOperational())
            .batteryChargePercent(nodeInfo.getBatteryChargePercent())
            .nodeProcessingDelayMs(nodeInfo.getNodeProcessingDelayMs())
            .packetLossRate(nodeInfo.getPacketLossRate())
            .resourceUtilization(nodeInfo.getResourceUtilization())
            .packetBufferCapacity(nodeInfo.getPacketBufferCapacity())
            .currentPacketCount(nodeInfo.getCurrentPacketCount())
            .weather(nodeInfo.getWeather())
            .lastUpdated(nodeInfo.getLastUpdated())
            .host(nodeInfo.getHost())
            .port(nodeInfo.getPort())
            .build();
    }

    /**
     * Convert NodeDTO to NodeInfo entity
     */
    public NodeInfo toEntity() {
        NodeInfo nodeInfo = new NodeInfo();
        nodeInfo.setId(this.id);
        nodeInfo.setNodeId(this.nodeId);
        nodeInfo.setNodeName(this.nodeName);
        nodeInfo.setNodeType(this.nodeType);
        nodeInfo.setPosition(this.position);
        nodeInfo.setOrbit(this.orbit);
        nodeInfo.setVelocity(this.velocity);
        nodeInfo.setCommunication(this.communication);
        nodeInfo.setOperational(this.isOperational != null ? this.isOperational : false);
        nodeInfo.setBatteryChargePercent(this.batteryChargePercent != null ? this.batteryChargePercent : 100.0);
        nodeInfo.setNodeProcessingDelayMs(this.nodeProcessingDelayMs != null ? this.nodeProcessingDelayMs : 0.0);
        nodeInfo.setPacketLossRate(this.packetLossRate != null ? this.packetLossRate : 0.0);
        nodeInfo.setResourceUtilization(this.resourceUtilization != null ? this.resourceUtilization : 0.0);
        nodeInfo.setPacketBufferCapacity(this.packetBufferCapacity != null ? this.packetBufferCapacity : 100);
        nodeInfo.setCurrentPacketCount(this.currentPacketCount != null ? this.currentPacketCount : 0);
        nodeInfo.setWeather(this.weather != null ? this.weather : WeatherCondition.CLEAR);
        nodeInfo.setLastUpdated(this.lastUpdated != null ? this.lastUpdated : Instant.now());
        nodeInfo.setHost(this.host);
        nodeInfo.setPort(this.port);
        return nodeInfo;
    }

    /**
     * Update entity with DTO values (for PATCH operations)
     */
    public void updateEntity(NodeInfo nodeInfo) {
        if (this.nodeId != null) nodeInfo.setNodeId(this.nodeId);
        if (this.nodeName != null) nodeInfo.setNodeName(this.nodeName);
        if (this.nodeType != null) nodeInfo.setNodeType(this.nodeType);
        if (this.position != null) nodeInfo.setPosition(this.position);
        if (this.orbit != null) nodeInfo.setOrbit(this.orbit);
        if (this.velocity != null) nodeInfo.setVelocity(this.velocity);
        if (this.communication != null) nodeInfo.setCommunication(this.communication);
        if (this.isOperational != null) nodeInfo.setOperational(this.isOperational);
        if (this.batteryChargePercent != null) nodeInfo.setBatteryChargePercent(this.batteryChargePercent);
        if (this.nodeProcessingDelayMs != null) nodeInfo.setNodeProcessingDelayMs(this.nodeProcessingDelayMs);
        if (this.packetLossRate != null) nodeInfo.setPacketLossRate(this.packetLossRate);
        if (this.resourceUtilization != null) nodeInfo.setResourceUtilization(this.resourceUtilization);
        if (this.packetBufferCapacity != null) nodeInfo.setPacketBufferCapacity(this.packetBufferCapacity);
        if (this.currentPacketCount != null) nodeInfo.setCurrentPacketCount(this.currentPacketCount);
        if (this.weather != null) nodeInfo.setWeather(this.weather);
        if (this.host != null) nodeInfo.setHost(this.host);
        if (this.port != null) nodeInfo.setPort(this.port);
        nodeInfo.setLastUpdated(Instant.now());
    }
}
