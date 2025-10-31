package com.sagsins.core.model;

import com.fasterxml.jackson.annotation.*;
import lombok.*;
import org.bson.*;
import org.bson.codecs.pojo.annotations.*;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

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
@Document(collection = "network_nodes")
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

    @Field("operational")
    private boolean isOperational;

    private double batteryChargePercent;
    private double nodeProcessingDelayMs;
    private double packetLossRate;
    private double resourceUtilization;
    private int packetBufferCapacity;
    private int currentPacketCount;
    private WeatherCondition weather;
    private List<String> neighbors;

    @BsonProperty("lastUpdated")
    @BsonRepresentation(BsonType.DATE_TIME)
    @LastModifiedDate
    private Instant lastUpdated = Instant.now();

    private String host;
    private int port;

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
