package com.sagsins.core.model;
public record HopRecord(
    String fromNodeId,
    String toNodeId,
    double latencyMs,
    long timestampMs,
    Position fromNodePosition,
    Position toNodePosition,
    double distanceKm,
    BufferState fromNodeBufferState,
    RoutingDecisionInfo routingDecisionInfo,
    SimulationScenario scenarioType,
    Double nodeLoadPercent,
    String dropReasonDetails
) {
    /**
     * Constructor for backward compatibility without scenario fields
     */
    public HopRecord(
        String fromNodeId,
        String toNodeId,
        double latencyMs,
        long timestampMs,
        Position fromNodePosition,
        Position toNodePosition,
        double distanceKm,
        BufferState fromNodeBufferState,
        RoutingDecisionInfo routingDecisionInfo
    ) {
        this(fromNodeId, toNodeId, latencyMs, timestampMs, fromNodePosition, toNodePosition,
             distanceKm, fromNodeBufferState, routingDecisionInfo, 
             SimulationScenario.NORMAL, null, null);
    }
}