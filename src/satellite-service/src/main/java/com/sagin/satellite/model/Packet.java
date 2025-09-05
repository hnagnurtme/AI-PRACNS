package com.sagin.satellite.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class Packet {
    private String id;
    private String src;
    private String dst;
    private int payloadSize;
    private int ttl;
    private String currentNode;
    private String nextHop;
    private long createdAt;
    private long lastUpdated;
    private double delayMs;
    private double lossRate;
    private boolean dropped;
}
