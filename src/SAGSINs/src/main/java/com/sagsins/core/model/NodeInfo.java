package com.sagsins.core.model;

import lombok.*;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(value = {"healthy"}, ignoreUnknown = true)
@ToString
@Document(collection = "network_nodes")
public class NodeInfo {

    @Id 
    private String nodeId;

    private NodeType nodeType;
    private Geo3D position;
    private Orbit orbit;
    private Velocity velocity;

    private boolean isOperational;
    private double batteryChargePercent;
    private double nodeProcessingDelayMs;
    private double packetLossRate;
    private double resourceUtilization;
    private int packetBufferCapacity;
    private int currentPacketCount;
    private WeatherCondition weather;
    private long lastUpdated;

    private String host;
    private int port;

    @JsonIgnore
    @JsonProperty(access = Access.READ_ONLY)
    public boolean isHealthy() {
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
}
