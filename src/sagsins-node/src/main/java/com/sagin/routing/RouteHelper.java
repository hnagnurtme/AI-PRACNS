package com.sagin.routing;

import java.util.List;

public class RouteHelper {

    /**
     * Tạo RouteInfo cơ bản.
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @param path Danh sách nodeId trên đường đi (source -> ... -> destination)
     * @return RouteInfo đã điền thông tin cơ bản
     */
    public static RouteInfo createBasicRoute(String sourceNodeId, String destinationNodeId, List<String> path) {
        if (path == null || path.isEmpty()) {
            return null;
        }

        RouteInfo route = new RouteInfo();
        route.setSourceNodeId(sourceNodeId);
        route.setDestinationNodeId(destinationNodeId);
        route.setPathNodeIds(path);
        route.setNextHopNodeId(path.size() > 1 ? path.get(1) : destinationNodeId);
        route.setHopCount(path.size() - 1);
        route.setTimestampComputed(System.currentTimeMillis());
        route.setValidUntil(System.currentTimeMillis() + 60000); // TTL 1 phút

        // Metric nâng cao mặc định
        route.setTotalCost(0);
        route.setTotalLatencyMs(0);
        route.setMinBandwidthMbps(0);
        route.setAvgPacketLossRate(0);
        route.setReliabilityScore(1.0);
        route.setEnergyCost(0.0);
        route.setLastReward(null);
        route.setPolicyVersion(null);

        return route;
    }
}
