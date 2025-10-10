package com.sagsins.core.DTOs;

import com.sagsins.core.model.Geo3D;
import com.sagsins.core.model.Orbit;
import com.sagsins.core.model.Velocity;
import com.sagsins.core.model.WeatherCondition; 
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max; 
import jakarta.validation.constraints.Min; 
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

import lombok.Getter;
import lombok.Setter;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class CreateNodeRequest {
    
    // --- THÔNG TIN CƠ BẢN VÀ DANH TÍNH ---
    
    @NotBlank(message = "Node id must be provided.")
    private String nodeId;

    @NotBlank(message = "Node type must be provided.")
    @Pattern(regexp = "^(GROUND_STATION|LEO_SATELLITE|MEO_SATELLITE|GEO_SATELLITE)$", 
            message = "Node type must be one of the defined categories.")
    private String nodeType; 
    
    @JsonProperty("isOperational")
    @NotNull(message = "Operational status must be provided.")
    private boolean isOperational = true; 

    // --- VỊ TRÍ & CHUYỂN ĐỘNG ---

    @NotNull(message = "Position data is mandatory.")
    @Valid 
    private Geo3D position; 
    
    @Valid
    private Orbit orbit;    
    
    @Valid
    private Velocity velocity; 

    // --- NĂNG LƯỢNG (MỚI) ---
    
    @NotNull(message = "Battery charge must be provided.")
    @DecimalMin(value = "0.0", inclusive = true, message = "Battery charge must be non-negative.")
    @Max(value = 100, message = "Battery charge cannot exceed 100%.")
    private double batteryChargePercent = 100.0; 
    
    // --- HIỆU SUẤT VÀ TẮC NGHẼN ---
    
    @NotNull(message = "Processing delay must be provided.")
    @DecimalMin(value = "0.0", inclusive = true, message = "Processing delay cannot be negative.")
    private double nodeProcessingDelayMs = 1.0; // MỚI: Độ trễ xử lý (Được ưu tiên hơn avgLatency)
    
    @NotNull(message = "Packet loss rate must be provided.")
    @DecimalMin(value = "0.0", message = "Packet loss rate cannot be negative.")
    private double packetLossRate;       // ĐÃ CÓ
    
    @NotNull(message = "Resource utilization must be provided.")
    @DecimalMin(value = "0.0", inclusive = true, message = "Resource utilization must be non-negative.")
    private double resourceUtilization = 0.0; // MỚI: Tải ban đầu

    @NotNull(message = "Packet buffer capacity must be provided.")
    @Min(value = 1, message = "Buffer capacity must be at least 1.")
    private int packetBufferCapacity = 500; // MỚI: Dung lượng buffer

    // --- ĐIỀU KIỆN MÔI TRƯỜNG (MỚI) ---
    
    @NotNull(message = "Weather condition must be provided.")
    private WeatherCondition weather = WeatherCondition.CLEAR; 

    // --- THÔNG TIN KẾT NỐI (MỚI) ---
    
    @NotBlank(message = "Host/IP must be provided.")
    private String host = "localhost"; 
    
    @Min(value = 1024, message = "Port must be greater than 1024.")
    @Max(value = 65535, message = "Port cannot exceed 65535.")
    private int port = 8081;    
}