package com.example.model;

public enum NodeType {
    GROUND_STATION(0, 0.3, 0.001, 5.0),
    LEO_SATELLITE(500, 1.0, 0.005, 20.0),
    MEO_SATELLITE(10000, 0.8, 0.01, 50.0),
    GEO_SATELLITE(35786, 0.6, 0.02, 250.0);

    private final double defaultAltitudeKm;      // Độ cao trung bình
    private final double signalAttenuation;      // Hệ số suy hao tín hiệu tương đối
    private final double batteryDrainFactor;     // Hệ số hao pin
    private final double baseLatencyMs;          // Độ trễ cơ bản

    NodeType(double defaultAltitudeKm, double signalAttenuation,
            double batteryDrainFactor, double baseLatencyMs) {
        this.defaultAltitudeKm = defaultAltitudeKm;
        this.signalAttenuation = signalAttenuation;
        this.batteryDrainFactor = batteryDrainFactor;
        this.baseLatencyMs = baseLatencyMs;
    }

    public double getDefaultAltitudeKm() {
        return defaultAltitudeKm;
    }

    public double getSignalAttenuation() {
        return signalAttenuation;
    }

    public double getBatteryDrainFactor() {
        return batteryDrainFactor;
    }

    public double getBaseLatencyMs() {
        return baseLatencyMs;
    }
}
