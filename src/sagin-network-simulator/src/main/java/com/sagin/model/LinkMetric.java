package com.sagin.model;

import lombok.*;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonProperty.Access;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
@ToString
public class LinkMetric {

    private String sourceNodeId;       
    private String destinationNodeId;  

    private double distanceKm;         
    private double maxBandwidthMbps;   
    private double currentAvailableBandwidthMbps; 
    private double latencyMs;          
    private double packetLossRate;     
    private double linkAttenuationDb; 
    
    private double linkScore;          
    private boolean isLinkActive = true;      
    private long lastUpdated;          

    /** * Tính điểm link tổng hợp dựa trên QoS.
     * Điểm này dùng để đánh giá chất lượng link trong định tuyến (cao hơn = tốt hơn).
     */
    @JsonProperty(access = Access.READ_ONLY)
    public double calculateLinkScore() {
        if (!isLinkActive || currentAvailableBandwidthMbps <= 1.0) {
            this.linkScore = 0.0;
            return 0.0;
        }
        
        // Công thức tối ưu: Phạt nặng độ trễ (Log) và suy hao/mất gói.
        // LatencyCost: Đảm bảo Latency không phải 0 để tránh lỗi log(0)
        double latencyCost = 1.0 + Math.log(1.0 + latencyMs); 
        
        // LossFactor: Giảm điểm nếu có mất gói
        double lossFactor = 1.0 - packetLossRate;
        
        // AttenuationFactor: Giảm điểm dựa trên suy hao (ví dụ: mỗi 1dB giảm 5% điểm)
        double attenuationFactor = 1.0 / (1.0 + 0.05 * linkAttenuationDb);
        
        // LinkScore = (Bandwidth / Cost) * (Reliability) * (Quality)
        double score = (currentAvailableBandwidthMbps / latencyCost) * lossFactor * attenuationFactor;
        
        this.linkScore = Math.max(0, score);
        return this.linkScore;
    }
}