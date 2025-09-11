package com.sagsins.core.model;

import lombok.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import java.util.concurrent.atomic.AtomicInteger;


@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)

public class NodeInfo {
    private static final AtomicInteger COUNTER = new AtomicInteger(1);
    private String nodeId; // ID node
    private String nodeType ;
    private Orbit orbit;       // ➕ quỹ đạo
    private Velocity velocity; // ➕ vận tốc
    private Geo3D position; // Vị trí 3D của node



    private boolean linkAvailable; // Liên kết đến node khả dụng
    private double bandwidth; // Băng thông tối đa (Mbps)
    private double latencyMs; // Độ trễ đến node (ms)
    private double packetLossRate; // Tỷ lệ mất packet

    private int bufferSize; // Số packet hiện có trong buffer
    private double throughput; // Lưu lượng thực tế đang xử lý (Mbps)

    private long lastUpdated; // Thời điểm cập nhật trạng thái
    private boolean healthy = true;


    // ----- Helper methods -----

    /**
     * Node có khả năng nhận/gửi packet không
     */
    public boolean isHealthy() {
        return linkAvailable && bufferSize < 1000; // threshold ví dụ
    }

    public void setHealthy(boolean healthy) {
        this.healthy = healthy;
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
        this.healthy = isHealthy();
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

    public NodeInfo(String nodeType, Geo3D position) {
        this.nodeId = nodeType + "-" + COUNTER.getAndIncrement();
        this.nodeType = nodeType;
        this.position = position;

        // Khởi tạo các giá trị mặc định
        this.linkAvailable = true;
        this.bandwidth = 100.0; // Mbps
        this.latencyMs = 10.0; // ms
        this.packetLossRate = 0.01; // 1%
        this.bufferSize = 0; // ban đầu không có packet
        this.throughput = 0.0; // ban đầu không có throughput
        this.lastUpdated = System.currentTimeMillis(); // thời điểm hiện tại    
    }

    public NodeInfo(String nodeType, Geo3D position, Orbit orbit, Velocity velocity) {
        this.nodeId = nodeType + "-" + COUNTER.getAndIncrement();
        this.nodeType = nodeType;
        this.position = position;
        this.orbit = orbit;
        this.velocity = velocity;

        // Khởi tạo các giá trị mặc định
        this.linkAvailable = true;
        this.bandwidth = 100.0; // Mbps
        this.latencyMs = 10.0; // ms
        this.packetLossRate = 0.01; // 1%
        this.bufferSize = 0; // ban đầu không có packet
        this.throughput = 0.0; // ban đầu không có throughput
        this.lastUpdated = System.currentTimeMillis(); // thời điểm hiện tại    
    }
}
