package com.sagin.model;

import lombok.*;

import org.bson.codecs.pojo.annotations.BsonId;
import org.bson.codecs.pojo.annotations.BsonProperty;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(value = {"healthy"}, ignoreUnknown = true)
@ToString
public class NodeInfo {

    @BsonId
    @BsonProperty("_id")
    private String nodeId; 
    private NodeType nodeType; // Sử dụng Enum

    private Geo3D position; // Vị trí 3D (không thay đổi)
    private Orbit orbit;    // Thông tin quỹ đạo (chỉ cho vệ tinh)
    private Velocity velocity; // Thông tin vận tốc (chỉ cho vệ tinh)

    // --- TRẠNG THÁI VÀ NĂNG LƯỢNG ---
    private boolean isOperational;       // Trạng thái cấu hình (On/Off)
    private double batteryChargePercent; // Mức pin hiện tại (0.0 - 100.0)

    // --- HIỆU SUẤT VÀ TẮC NGHẼN ---
    private double nodeProcessingDelayMs; // Độ trễ xử lý bên trong node
    private double packetLossRate;        // Tỷ lệ mất gói do lỗi node/thiết bị
    private double resourceUtilization;   // Tỷ lệ sử dụng tài nguyên (CPU/RAM)

    private int packetBufferCapacity;     // Dung lượng tối đa của buffer
    private int currentPacketCount;       // Số gói tin hiện tại trong buffer

    // --- ĐIỀU KIỆN MÔI TRƯỜNG ---
    private WeatherCondition weather; // Điều kiện thời tiết tại node (chủ yếu cho Ground Station)

    private long lastUpdated;

       // --- THÔNG TIN KẾT NỐI (MỚI) ---
    private String host; // Tên dịch vụ Docker hoặc IP (ví dụ: "leo_101")
    private int port;    // Cổng lắng nghe TCP (ví dụ: 8081)

    /** Kiểm tra Node có sẵn sàng xử lý/gửi/nhận không (Healthy Check). */
    @JsonIgnore
    @JsonProperty(access = Access.READ_ONLY)
    public boolean isHealthy() {
        // Ngưỡng hoạt động
        final double MIN_POWER = 10.0;
        final double MAX_BUFFER_LOAD_RATIO = 0.8; 

        double bufferLoadRatio = (packetBufferCapacity > 0) 
            ? (double) currentPacketCount / packetBufferCapacity 
            : 0.0;

        // Bị coi là không khỏe nếu có bão nghiêm trọng HOẶC tải buffer quá cao HOẶC pin yếu
        return isOperational 
            && batteryChargePercent > MIN_POWER 
            && bufferLoadRatio <= MAX_BUFFER_LOAD_RATIO
            && weather != WeatherCondition.SEVERE_STORM;
    }
}