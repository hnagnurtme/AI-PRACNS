package com.sagin.model;


public record HopRecord(
    String fromNodeId,
    String toNodeId,
    double latencyMs,
    long timestampMs,
    Position fromNodePosition,
    Position toNodePosition,
    double distanceKm,
    BufferState fromNodeBufferState,
    RoutingDecisionInfo routingDecisionInfo
) {}