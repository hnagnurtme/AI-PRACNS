package com.sagin.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import org.bson.BsonType;
import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.codecs.pojo.annotations.BsonProperty;
import org.bson.codecs.pojo.annotations.BsonRepresentation;

import java.time.Instant;
import java.util.List;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@ToString
@JsonInclude(JsonInclude.Include.NON_NULL)
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

    private List<String> neigbors;

    private double batteryChargePercent;
    private double nodeProcessingDelayMs;
    private double packetLossRate;
    private double resourceUtilization;
    private int packetBufferCapacity;
    private int currentPacketCount;
    private WeatherCondition weather;
    private Instant lastUpdated;
    private int port;

    private Boolean healthy;
    @BsonProperty("isOperational")
    private boolean operational;

    public boolean isOperational() { // getter chuẩn
        return operational;
    }

    public void setOperational(boolean operational) { // setter chuẩn
        this.operational = operational;
    }


    public void setLastUpdated(Instant lastUpdated) {
        this.lastUpdated = lastUpdated;
    }
}