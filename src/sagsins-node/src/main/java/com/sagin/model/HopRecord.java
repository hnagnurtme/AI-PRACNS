package com.sagin.model;

import java.util.Map;

public record HopRecord(
    String fromNodeId,
    String toNodeId,
    double latencyMs,
    long timestampMs,
    Position fromNodePosition,
    Position toNodePosition,
    double distanceKm,
    Map<String, Object> fromNodeBufferState,
    Map<String, Object> routingDecisionInfo
) {}