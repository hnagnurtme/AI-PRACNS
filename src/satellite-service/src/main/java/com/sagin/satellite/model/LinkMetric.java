package com.sagin.satellite.model;

import lombok.*;
import com.fasterxml.jackson.annotation.JsonInclude;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class LinkMetric {

    private String sourceNodeId;       // Node gửi
    private String destinationNodeId;  // Node nhận

    private double distanceKm;         // Khoảng cách 3D giữa 2 node
    private double bandwidthMbps;      // Băng thông tối đa đường link
    private double latencyMs;          // Delay trung bình (ms)
    private double packetLossRate;     // Tỷ lệ mất packet
    private double linkScore;          // Điểm đánh giá link tổng hợp (cho AI hoặc routing)

    private boolean linkAvailable;     // Link hiện tại có khả dụng không
    private long lastUpdated;          // Timestamp cập nhật metric

    // ----- Helper methods -----

    /**
     * Cập nhật các chỉ số đường link và tính linkScore tổng hợp
     */
    public void updateMetrics(double bandwidthMbps, double latencyMs, double packetLossRate, boolean linkAvailable) {
        this.bandwidthMbps = bandwidthMbps;
        this.latencyMs = latencyMs;
        this.packetLossRate = packetLossRate;
        this.linkAvailable = linkAvailable;
        this.lastUpdated = System.currentTimeMillis();
        calculateLinkScore();
    }

    /**
     * Tính điểm link tổng hợp dựa trên QoS
     * Có thể dùng cho routing, AI-service để chọn đường đi tốt nhất
     * Ví dụ đơn giản: score càng cao, link càng tốt
     */
    private void calculateLinkScore() {
        if(!linkAvailable) {
            linkScore = 0;
            return;
        }
        // Giả sử công thức đơn giản: bandwidth ưu tiên, latency và packetLoss trừ điểm
        linkScore = bandwidthMbps / (1 + latencyMs) * (1 - packetLossRate);
    }

    /**
     * Cập nhật khoảng cách dựa trên NodeInfo (3D)
     */
    public void updateDistance(NodeInfo sourceNode, NodeInfo destinationNode) {
        if(sourceNode != null && destinationNode != null) {
            this.distanceKm = sourceNode.distanceTo(destinationNode);
        }
    }
}
