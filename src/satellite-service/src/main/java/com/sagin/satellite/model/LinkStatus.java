package com.sagin.satellite.model;

import lombok.*;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class LinkStatus {
    private String srcNode;
    private String dstNode;
    private double latencyMs;
    private double lossRate;
    private double capacityMbps;
    private boolean active;
    private double currentLoad;
    private long lastUpdated;
}
