package com.example.model;

public record ServiceQoS(
    ServiceType serviceType,
    int defaultPriority,
    double maxLatencyMs,
    double maxJitterMs,
    double minBandwidthMbps,
    double maxLossRate
) {}