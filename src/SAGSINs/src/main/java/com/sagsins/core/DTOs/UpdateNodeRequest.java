package com.sagsins.core.DTOs;

import com.sagsins.core.model.Geo3D;
import com.sagsins.core.model.Orbit;
import com.sagsins.core.model.Velocity;
import com.sagsins.core.model.WeatherCondition;
import com.fasterxml.jackson.annotation.JsonInclude;

import lombok.Getter;
import lombok.Setter;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

/**
 * DTO dùng để cập nhật thông tin NodeInfo từ client.
 * 
 * - Mọi trường đều có thể là NULL → chỉ cập nhật những trường được gửi lên.
 * - Dùng các wrapper type (Double, Integer, Boolean, v.v.) để phân biệt "null" và "giá trị thực".
 * - Cho phép cập nhật cả phần kỹ thuật, năng lượng, hiệu suất, và môi trường.
 */
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class UpdateNodeRequest {

    // --- THÔNG TIN CƠ BẢN / ĐỊNH DANH ---
    private String nodeType;            // Có thể chuyển đổi loại node (nếu cần)
    private Boolean isOperational;      // Bật / tắt node

    // --- VỊ TRÍ & QUỸ ĐẠO ---
    private Geo3D position;             // Cập nhật toạ độ 3D
    private Orbit orbit;                // Thay đổi thông số quỹ đạo
    private Velocity velocity;          // Tốc độ hiện tại

    // --- HIỆU SUẤT & CHẤT LƯỢNG DỊCH VỤ (QoS) ---
    private Double currentBandwidth;    // Băng thông hiện tại (Mbps)
    private Double avgLatencyMs;        // Độ trễ trung bình (ms)
    private Double packetLossRate;      // Tỷ lệ mất gói (%)
    private Double currentThroughput;   // Lưu lượng thực tế (Mbps)
    private Double nodeProcessingDelayMs; // Độ trễ xử lý nội bộ
    private Double resourceUtilization; // % tài nguyên sử dụng (CPU/RAM)
    private Integer packetBufferLoad;   // Số lượng gói trong buffer hiện tại
    private Integer packetBufferCapacity; // Dung lượng tối đa của buffer

    // --- NĂNG LƯỢNG & TÀI NGUYÊN ---
    private Double batteryChargePercent; // % năng lượng pin
    private Double powerLevel;           // Công suất tiêu thụ hiện tại (W)
    private Double temperatureCelsius;   // Nhiệt độ (°C)
    private Double cpuUsagePercent;      // % CPU đang sử dụng
    private Double memoryUsagePercent;   // % RAM đang sử dụng

    // --- MÔI TRƯỜNG & TÌNH TRẠNG ---
    private WeatherCondition weather;   // Điều kiện thời tiết
    private Double radiationLevel;      // Mức phóng xạ / nhiễu (nếu là vệ tinh)
    private Double signalToNoiseRatio;  // SNR (Signal-to-Noise Ratio)
    private Double linkQuality;         // Điểm chất lượng liên kết (0–1)

    // --- KẾT NỐI & GIAO TIẾP ---
    private String host;                // Tên máy / container / IP
    private Integer port;               // Cổng TCP
    private Double uplinkPower;         // Công suất phát lên (dBm)
    private Double downlinkPower;       // Công suất thu xuống (dBm)
    private Double linkLatencyMs;       // Độ trễ đường truyền (ms)

    // --- MONITORING & TRẠNG THÁI ---
    private Long lastUpdated;           // Thời điểm cập nhật cuối cùng
    private String statusMessage;       // Ghi chú trạng thái (tùy chọn)
    private Integer errorCount;         // Số lỗi xảy ra gần đây
    private Double reliabilityScore;    // Mức độ tin cậy (0–1)
}
