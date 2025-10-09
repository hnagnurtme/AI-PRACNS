// src/main/java/com/sagsins.core.DTOs/CreateNodeRequest.java
package com.sagsins.core.DTOs;

import com.sagsins.core.model.Geo3D;
import com.sagsins.core.model.Orbit;
import com.sagsins.core.model.Velocity;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import jakarta.validation.Valid;     // Cần cho Geo3D, Orbit, Velocity
import jakarta.validation.constraints.DecimalMin;
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
    @NotBlank(message = "Node id must be provided.")
    private String nodeId;

    @NotBlank(message = "Node type must be provided.")
    @Pattern(regexp = "^(GROUND_STATION|LEO_SATELLITE|MEO_SATELLITE|GEO_SATELLITE)$", 
              message = "Node type must be one of the defined categories.")
    private String nodeType; 
    
    @JsonProperty("isOperational")
    @NotNull(message = "Operational status must be provided.")
    private boolean isOperational = true; 

    @NotNull(message = "Position data is mandatory.")
    @Valid 
    private Geo3D position; 
    
    @Valid
    private Orbit orbit;    
    
    @Valid
    private Velocity velocity; 

    @NotNull(message = "Current bandwidth must be provided.")
    @DecimalMin(value = "0.0", inclusive = true, message = "Bandwidth cannot be negative.")
    private double currentBandwidth;     
    
    @NotNull(message = "Average latency must be provided.")
    @DecimalMin(value = "0.0", inclusive = false, message = "Latency must be positive.")
    private double avgLatencyMs;         
    
    @NotNull(message = "Packet loss rate must be provided.")
    @DecimalMin(value = "0.0", message = "Packet loss rate cannot be negative.")
    private double packetLossRate;       
}
