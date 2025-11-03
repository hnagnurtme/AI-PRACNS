package com.example.model;


/**
 * Tương ứng với export interface RoutingDecisionInfo.
 */
public record RoutingDecisionInfo(
    Algorithm algorithm, // Sử dụng Enum đã định nghĩa bên dưới
    String metric,
    Double reward
) {
    /**
     * Tương ứng với Union Type: "Dijkstra" | "ReinforcementLearning" trong TypeScript.
     */
    public enum Algorithm {
        Dijkstra,
        ReinforcementLearning
    }
}