package com.example.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class AnalysisData {
    private double avgLatency;
    private double avgDistanceKm;
    private double routeSuccessRate;
    private double totalDistanceKm;
    private double totalLatencyMs;
}
