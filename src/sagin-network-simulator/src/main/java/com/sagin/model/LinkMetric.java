package com.sagin.model;

import lombok.*;
import com.fasterxml.jackson.annotation.JsonInclude;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class LinkMetric {

    private String sourceNodeId;       
    private String destinationNodeId;  

    private double distanceKm;         
    private double maxBandwidthMbps;   
    private double currentAvailableBandwidthMbps; 
    private double latencyMs;          
    private double packetLossRate;     
    
    private double linkScore;          
    private boolean isLinkActive;      
    private long lastUpdated;          

    /** * Tính điểm link tổng hợp dựa trên QoS.
     * Điểm này là chi phí/phần thưởng chính cho thuật toán định tuyến/RL.
     * @return Điểm Link đã được tính toán.
     */
    public double calculateLinkScore() {
        if (!isLinkActive) {
            this.linkScore = 0.0;
            return 0.0;
        }
        
        // Công thức đơn giản: (Bandwidth / (1 + Latency)) * (1 - PacketLossRate)
        double score = currentAvailableBandwidthMbps / (1.0 + latencyMs) * (1.0 - packetLossRate);
        
        this.linkScore = Math.max(0, score);
        return this.linkScore;
    }
}