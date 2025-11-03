package com.example.model;

import lombok.*;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Builder
public class RouteInfo {

    // Cặp nguồn - đích
    private String sourceNodeId;
    private String destinationNodeId;

    // Route chính
    private String nextHopNodeId;
    private List<String> pathNodeIds; // danh sách các NodeId trên tuyến đường

    // Các metric định tuyến
    private double totalCost;         // tổng chi phí dựa trên link scores
    private double totalLatencyMs;    // tổng độ trễ tích lũy
    private double minBandwidthMbps;  // băng thông nhỏ nhất (bottleneck)
    private double avgPacketLossRate; // tỷ lệ mất gói trung bình
    private double reliabilityScore;  // độ tin cậy (0..1)
    private double energyCost;        // năng lượng tiêu hao (Joules)

    // Thông tin quản lý
    private int hopCount;             // số hop
    private long timestampComputed;   // thời điểm tính toán
    private long validUntil;          // thời điểm hết hạn (TTL)

    // Metadata cho RL / Scheduler
    private Double lastReward;        // reward gán từ RL
    private String policyVersion;     // chính sách RL đã sử dụng

}
