package com.sagsins.core.DTOs;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CreateNodeRequest {
    private String nodeType;
    private PositionDto position;
    private OrbitDto orbit;
    private VelocityDto velocity;
    
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class PositionDto {
        private double longitude;
        private double latitude;
        private double altitude;
    }
    
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class OrbitDto {
        private double altitude;
        private double inclination;
    }
    
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class VelocityDto {
        private double speed;
    }
}
