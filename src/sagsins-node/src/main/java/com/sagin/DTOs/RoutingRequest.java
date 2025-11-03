package com.sagin.DTOs;

import com.sagin.model.ServiceQoS;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class RoutingRequest {
    private String packetId;
    private String currentHoldingNodeId;
    private String stationDest;
    private double accumulatedDelayMs;
    private int ttl;
    private ServiceQoS serviceQoS;
}
