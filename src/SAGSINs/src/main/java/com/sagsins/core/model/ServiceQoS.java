package com.sagsins.core.model;

public record ServiceQoS(
    ServiceType serviceType,
    int defaultPriority,
    double maxLatencyMs,
    double maxJitterMs,
    double minBandwidthMbps,
    double maxLossRate
) {}