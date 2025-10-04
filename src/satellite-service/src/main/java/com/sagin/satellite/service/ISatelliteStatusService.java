package com.sagin.satellite.service;

import com.sagin.satellite.model.SatelliteStatus;
import com.sagin.satellite.model.NodeInfo;
import com.sagin.satellite.model.Packet;
import java.util.List;

/**
 * ISatelliteStatusService quản lý trạng thái và metrics của vệ tinh
 */
public interface ISatelliteStatusService {

    /**
     * Lấy trạng thái hiện tại của vệ tinh
     *
     * @return SatelliteStatus object
     */
    SatelliteStatus getCurrentStatus();

    /**
     * Cập nhật thông tin node của vệ tinh
     *
     * @param nodeInfo Thông tin node mới
     */
    void updateNodeInfo(NodeInfo nodeInfo);

    /**
     * Cập nhật buffer status
     *
     * @param buffer Danh sách packet hiện tại trong buffer
     */
    void updateBufferStatus(List<Packet> buffer);

    /**
     * Cập nhật performance metrics
     *
     * @param throughput Throughput hiện tại (Mbps)
     * @param averageLatency Latency trung bình (ms)
     * @param packetLossRate Tỷ lệ mất packet
     */
    void updatePerformanceMetrics(double throughput, double averageLatency, double packetLossRate);

    /**
     * Kiểm tra tình trạng sức khỏe của vệ tinh
     *
     * @return true nếu vệ tinh hoạt động bình thường
     */
    boolean isHealthy();

    /**
     * Lấy capacity còn lại của buffer
     *
     * @return Số packet có thể thêm vào buffer
     */
    int getRemainingBufferCapacity();

    /**
     * Tính toán load hiện tại của vệ tinh (0.0 - 1.0)
     *
     * @return Load percentage
     */
    double getCurrentLoad();

    /**
     * Lấy thống kê chi tiết về performance
     *
     * @return Map chứa các metrics chi tiết
     */
    java.util.Map<String, Object> getDetailedMetrics();

    /**
     * Reset tất cả metrics về trạng thái ban đầu
     */
    void resetMetrics();

    /**
     * Increment packet processed counter
     */
    void incrementPacketsProcessed();

    /**
     * Increment packet dropped counter
     */
    void incrementPacketsDropped();
}