package com.sagin.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString; // Thêm ToString để dễ gỡ lỗi

/**
 * Định nghĩa yêu cầu QoS mặc định cho từng loại dịch vụ.
 * Các thông số này được sử dụng để định tuyến và kiểm soát Admission Control.
 */
@Getter
@AllArgsConstructor
@ToString // Giúp hiển thị thông tin khi log
public class ServiceQoS {

    // --- Định danh Dịch vụ ---
    private final String serviceType;
    
    // --- Yêu cầu Chất lượng Dịch vụ (QoS) ---
    private final int defaultPriority;          // Độ ưu tiên trong hàng đợi (Ví dụ: 0 = cao nhất)

    // Yêu cầu về Độ trễ (Latency)
    private final double maxLatencyMs;          // Độ trễ tối đa chấp nhận được (Tính bằng miligiây)
    private final double maxJitterMs;           // Độ biến thiên độ trễ tối đa (Quan trọng cho real-time)

    // Yêu cầu về Băng thông (Bandwidth)
    private final double minBandwidthMbps;      // Băng thông tối thiểu cần thiết để hoạt động (Tính bằng Mbps)

    // Yêu cầu về Độ tin cậy (Reliability)
    private final double maxLossRate;           // Tỷ lệ mất gói tối đa chấp nhận được (0.0 đến 1.0)
    
}