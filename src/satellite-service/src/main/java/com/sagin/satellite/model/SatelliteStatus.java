package com.sagin.satellite.model;

import lombok.*;
import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class SatelliteStatus {

    private String satelliteId; // ID vệ tinh
    private NodeInfo nodeInfo; // Thông tin node
    private List<Packet> buffer; // Các packet đang giữ
    private int bufferSize; // Số packet hiện tại
    private long lastUpdated; // Timestamp cập nhật

    // Metrics tổng hợp
    private double throughput; // Mbps
    private double averageLatencyMs; // ms
    private double packetLossRate; // %

    /**
     * Cập nhật trạng thái vệ tinh dựa trên buffer và metrics
     */
    public void updateStatus(List<Packet> buffer, double throughput, double averageLatencyMs, double packetLossRate) {
        this.buffer = buffer;
        this.bufferSize = buffer != null ? buffer.size() : 0;
        this.throughput = throughput;
        this.averageLatencyMs = averageLatencyMs;
        this.packetLossRate = packetLossRate;
        this.lastUpdated = System.currentTimeMillis();
    }
}
