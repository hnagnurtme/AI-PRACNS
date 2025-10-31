package com.sagin.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import org.bson.BsonType;
import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.codecs.pojo.annotations.BsonProperty;
import org.bson.codecs.pojo.annotations.BsonRepresentation;

import java.time.Instant;
import java.util.List;
import java.util.Map;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@ToString
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(value = {"healthy"}, ignoreUnknown = true)
public class NodeInfo {

    @BsonId
    @BsonRepresentation(BsonType.OBJECT_ID)
    private String id;

    @BsonProperty("nodeId")
    private String nodeId;
    private String nodeName;
    private NodeType nodeType;

    private Position position; 

    private Orbit orbit;
    private Velocity velocity;
    private Communication communication;
    
    private List<String> neigbours;
    private boolean isOperational;
    private double batteryChargePercent;
    private double nodeProcessingDelayMs;
    private double packetLossRate;
    private double resourceUtilization;
    private int packetBufferCapacity;
    private int currentPacketCount;
    private WeatherCondition weather;
    private Instant lastUpdated;
    private String host;
    private int port;

    /**
     * Kiểm tra Node có sẵn sàng xử lý/gửi/nhận không (Healthy Check).
     */
    @JsonProperty(value = "healthy", access = JsonProperty.Access.READ_ONLY)
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


    public void setLastUpdated(Instant lastUpdated) {
        this.lastUpdated = lastUpdated;
    }

    public Map<String, Object> getBufferState() {
        return Map.of(
            "packetBufferCapacity", packetBufferCapacity,
            "currentPacketCount", currentPacketCount,
            "packetLossRate", packetLossRate,
            "resourceUtilization", resourceUtilization,
            "nodeProcessingDelayMs", nodeProcessingDelayMs
        );
    }
}