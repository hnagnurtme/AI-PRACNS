package com.sagin.satellite.model;

import lombok.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.sagin.satellite.util.ProjectConstant;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class NodeInfo {

    private String nodeId; // ID node
    private String nodeType = ProjectConstant.NODE_TYPE_SATELLITE; 

    private Geo3D position; // Vị trí 3D của node

    private boolean linkAvailable; // Liên kết đến node khả dụng
    private double bandwidth; // Băng thông tối đa (Mbps)
    private double latencyMs; // Độ trễ đến node (ms)
    private double packetLossRate; // Tỷ lệ mất packet

    private int bufferSize; // Số packet hiện có trong buffer
    private double throughput; // Lưu lượng thực tế đang xử lý (Mbps)

    private long lastUpdated; // Thời điểm cập nhật trạng thái

    // ----- Helper methods -----

    /**
     * Node có khả năng nhận/gửi packet không
     */
    public boolean isHealthy() {
        return linkAvailable && bufferSize < 1000; // threshold ví dụ
    }

    /**
     * Cập nhật các chỉ số QoS
     */
    public void updateMetrics(double bandwidth, double latencyMs, double packetLossRate,
            int bufferSize, double throughput) {
        this.bandwidth = bandwidth;
        this.latencyMs = latencyMs;
        this.packetLossRate = packetLossRate;
        this.bufferSize = bufferSize;
        this.throughput = throughput;
        this.lastUpdated = System.currentTimeMillis();
    }

    /**
     * Khoảng cách 3D đến node khác
     */
    public double distanceTo(NodeInfo other) {
        if (this.position == null || other.position == null)
            return Double.MAX_VALUE;
        return this.position.distanceTo(other.position);
    }

    /**
     * Vector hướng từ node này tới node khác
     */
    public double[] directionTo(NodeInfo other) {
        if (this.position == null || other.position == null)
            return new double[] { 0, 0, 0 };
        return this.position.directionTo(other.position);
    }
}
