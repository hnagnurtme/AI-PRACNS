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
 * DTO dùng để trả về thông tin chi tiết của một Node cho client.
 * Lớp này chỉ bao gồm các trường cần thiết và đã được làm sạch.
 */
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class NodeDTO {

    private String nodeId; 
    private String nodeType; 

    // --- Thông tin Vị trí & Cơ học ---
    private Geo3D position; 
    private Orbit orbit;   
    private Velocity velocity; 

    // --- Trạng thái Hoạt động & Sức khỏe ---
    private boolean isOperational;       
    private Boolean isHealthy; // Thêm trường này để hiển thị trạng thái sức khỏe tính toán

    // --- Các Metric QoS hiện tại ---
    private double currentBandwidth;     
    private double avgLatencyMs;         
    private double packetLossRate;       
    private double currentThroughput;    
    private double resourceUtilization;  
    private double powerLevel;           

    private long lastUpdated;   
}