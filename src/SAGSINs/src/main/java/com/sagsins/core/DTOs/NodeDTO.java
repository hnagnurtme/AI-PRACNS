package com.sagsins.core.DTOs;

import com.sagsins.core.model.Geo3D;
import com.sagsins.core.model.Orbit;
import com.sagsins.core.model.Velocity;
import com.sagsins.core.model.WeatherCondition;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class NodeDTO {

    private String nodeId;
    private String nodeType;

    private Geo3D position;
    private Orbit orbit;
    private Velocity velocity;

    private boolean isOperational;
    private Boolean isHealthy;

    private double batteryChargePercent;
    private double nodeProcessingDelayMs;
    private double packetLossRate;
    private double resourceUtilization;

    private int packetBufferCapacity;
    private int currentPacketCount;

    private WeatherCondition weather;
    private String host;
    private int port;

    private long lastUpdated;
}
