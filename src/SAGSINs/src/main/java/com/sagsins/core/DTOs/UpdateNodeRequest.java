package com.sagsins.core.DTOs;

import com.sagsins.core.model.Geo3D;
import com.sagsins.core.model.Orbit;
import com.sagsins.core.model.Velocity;
import com.fasterxml.jackson.annotation.JsonInclude;

import lombok.Getter;
import lombok.Setter;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

/**
 * DTO (Data Transfer Object) dùng để nhận yêu cầu cập nhật thông tin NodeInfo từ client.
 * Các trường có thể là NULL, Service sẽ chỉ cập nhật những trường được cung cấp.
 */
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class UpdateNodeRequest {

    // --- Thông tin Định danh & Cấu hình ---
    private String nodeType; 
    
    // Sử dụng Boolean object để phân biệt giữa:
    // null (không cập nhật) và false (chuyển sang trạng thái không hoạt động)
    private Boolean isOperational; 

    // --- Vị trí và Cơ học ---
    private Geo3D position;
    private Orbit orbit;   
    private Velocity velocity; 

    // --- Các Metric QoS đang được theo dõi/cập nhật ---
    // Sử dụng Double object để phân biệt giữa:
    // null (không cập nhật) và 0.0 (giá trị metric thực tế bằng 0)
    private Double currentBandwidth;     
    private Double avgLatencyMs;         
    private Double packetLossRate;       
    private Integer packetBufferLoad;
    private Double currentThroughput;    
    private Double resourceUtilization;  
    private Double powerLevel;           
}