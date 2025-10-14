package com.sagin.model;

import lombok.*;
import java.util.List;

@Getter
@Setter
// @AllArgsConstructor  
@NoArgsConstructor
@ToString
public class RouteInfo {
    private String nextHopNodeId;
    private List<String> pathNodeIds; // Toàn bộ đường đi (source -> ... -> destination)
    
    // --- Metrics của tuyến đường ---
    private double totalCost;         // Chi phí tổng hợp của tuyến đường (dựa trên LinkScore)
    private double totalLatencyMs;    // Tổng độ trễ tích lũy
    private double minBandwidthMbps;  // Băng thông thấp nhất trên tuyến đường (bottleneck)
    private double avgPacketLossRate; // Tỷ lệ mất gói trung bình

    private long timestampComputed;   // Thời điểm tuyến đường này được tính toán

    public RouteInfo(String nextHopNodeId, List<String> pathNodeIds, double totalCost,
                     double totalLatencyMs, double minBandwidthMbps,
                     double avgPacketLossRate, long timestampComputed) {
        this.nextHopNodeId = nextHopNodeId;
        this.pathNodeIds = pathNodeIds;
        this.totalCost = totalCost;
        this.totalLatencyMs = totalLatencyMs;
        this.minBandwidthMbps = minBandwidthMbps;
        this.avgPacketLossRate = avgPacketLossRate;
        this.timestampComputed = timestampComputed;
    }
}