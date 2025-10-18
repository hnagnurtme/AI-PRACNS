package com.sagsins.core.DTOs.request;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.sagsins.core.model.*;
import lombok.*;

import jakarta.validation.constraints.*;

/**
 * Request DTO for updating node status and operational parameters
 * Used specifically for PATCH operations
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString
@JsonInclude(JsonInclude.Include.NON_NULL)
public class UpdateStatusRequest {

    // Basic Information
    private String nodeName;

    private Orbit orbit;
    private Velocity velocity;

    // Communication
    private Communication communication;

    // Operational Status
    private Boolean isOperational;

    @Min(value = 0, message = "Battery charge must be at least 0")
    @Max(value = 100, message = "Battery charge must not exceed 100")
    private Double batteryChargePercent;

    @Min(value = 0, message = "Processing delay must be non-negative")
    private Double nodeProcessingDelayMs;

    @Min(value = 0, message = "Packet loss rate must be at least 0")
    @Max(value = 1, message = "Packet loss rate must not exceed 1")
    private Double packetLossRate;

    @Min(value = 0, message = "Resource utilization must be at least 0")
    @Max(value = 1, message = "Resource utilization must not exceed 1")
    private Double resourceUtilization;

    // Buffer Management
    @Min(value = 0, message = "Packet buffer capacity must be non-negative")
    private Integer packetBufferCapacity;

    @Min(value = 0, message = "Current packet count must be non-negative")
    private Integer currentPacketCount;

    // Weather
    private WeatherCondition weather;

    // Network Configuration
    private String host;

    @Min(value = 1, message = "Port must be at least 1")
    @Max(value = 65535, message = "Port must not exceed 65535")
    private Integer port;
}
