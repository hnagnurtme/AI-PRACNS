package com.sagsins.core.model;

import lombok.*;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class NodeInfo {

    private String nodeId; 
    private String nodeType; 

    private Geo3D position; 
    private Orbit orbit; 
    private Velocity velocity; 


    private boolean isOperational;       
    private double currentBandwidth;     
    private double avgLatencyMs;         
    private double packetLossRate;       

    private int packetBufferLoad;        
    private double currentThroughput;    
    private double resourceUtilization;  
    private double powerLevel;           

    private long lastUpdated;            

    /** Kiểm tra Node có sẵn sàng xử lý/gửi/nhận không. */
    @JsonIgnore
    public boolean isHealthy() {
        // Giả định ngưỡng hoạt động: Pin trên 5% và buffer chưa quá tải (ví dụ: 90/100)
        return isOperational && powerLevel > 5.0 && packetBufferLoad < 90; 
    }
}