package com.sagin.routing;

import com.sagin.model.NodeInfo;
import com.sagin.repository.INodeRepository;

import java.util.List;
import java.util.Optional;

public class RouteHelper {

    /**
     * Tạo RouteInfo cơ bản.
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @param path Danh sách nodeId trên đường đi (source -> ... -> destination)
     * @return RouteInfo đã điền thông tin cơ bản
     * @deprecated Use createRouteWithCost instead for accurate cost calculation
     */
    @Deprecated
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

    /**
     * ✅ Creates RouteInfo with ACTUAL calculated costs based on path.
     * This method calculates real metrics like distance, latency, bandwidth, etc.
     *
     * @param sourceNodeId Node nguồn
     * @param destinationNodeId Node đích
     * @param path Danh sách nodeId trên đường đi (source -> ... -> destination)
     * @param totalCost Total cost from Dijkstra algorithm
     * @param nodeRepository Repository to fetch node information
     * @return RouteInfo với các metric đã được tính toán chính xác
     */
    public static RouteInfo createRouteWithCost(
            String sourceNodeId,
            String destinationNodeId,
            List<String> path,
            double totalCost,
            INodeRepository nodeRepository) {

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

        // ✅ Calculate actual metrics along the path
        double totalDistanceKm = 0.0;
        double totalLatencyMs = 0.0;
        double minBandwidth = Double.MAX_VALUE;
        double totalPacketLoss = 0.0;
        int linkCount = 0;

        for (int i = 0; i < path.size() - 1; i++) {
            String fromNodeId = path.get(i);
            String toNodeId = path.get(i + 1);

            Optional<NodeInfo> fromNodeOpt = nodeRepository.getNodeInfo(fromNodeId);
            Optional<NodeInfo> toNodeOpt = nodeRepository.getNodeInfo(toNodeId);

            if (fromNodeOpt.isEmpty() || toNodeOpt.isEmpty()) {
                continue; // Skip invalid nodes
            }

            NodeInfo fromNode = fromNodeOpt.get();
            NodeInfo toNode = toNodeOpt.get();

            // Calculate distance using Haversine formula
            double distanceKm = calculateDistance(fromNode, toNode);
            totalDistanceKm += distanceKm;

            // Calculate propagation delay
            double propagationDelayMs = distanceKm / 200.0; // ~200 km/ms (speed of light in medium)

            // Get bandwidth (minimum along path is the bottleneck)
            double bandwidthMHz = fromNode.getCommunication().getBandwidthMHz();
            if (bandwidthMHz < minBandwidth) {
                minBandwidth = bandwidthMHz;
            }

            // Transmission delay (assume 1KB packet for estimation)
            double transmissionDelayMs = 0.0;
            if (bandwidthMHz > 0) {
                double bandwidthBps = bandwidthMHz * 1_000_000; // MHz to bps
                transmissionDelayMs = (1024.0 * 8.0) / bandwidthBps * 1000.0; // bits / bps * 1000 = ms
            }

            // Weather impact
            double weatherFactor = 1.0;
            if (fromNode.getWeather() != null) {
                weatherFactor = 1.0 + (fromNode.getWeather().getTypicalAttenuationDb() / 100.0);
            }

            // Total latency for this hop
            double hopLatency = (propagationDelayMs + transmissionDelayMs) * weatherFactor;
            totalLatencyMs += hopLatency;

            // Packet loss (if available)
            if (fromNode.getCommunication() != null) {
                // Assume 0.01% base packet loss per hop
                totalPacketLoss += 0.0001;
            }

            linkCount++;
        }

        // Set calculated metrics
        route.setTotalCost(totalCost); // From Dijkstra algorithm
        route.setTotalLatencyMs(totalLatencyMs);
        route.setMinBandwidthMbps(minBandwidth == Double.MAX_VALUE ? 0 : minBandwidth);
        route.setAvgPacketLossRate(linkCount > 0 ? totalPacketLoss / linkCount : 0.0);

        // Reliability score (1.0 = perfect, decreases with packet loss)
        route.setReliabilityScore(Math.max(0.0, 1.0 - totalPacketLoss));

        // Energy cost (proportional to distance and hops)
        route.setEnergyCost(totalDistanceKm * 0.01 + linkCount * 0.5);

        route.setLastReward(null);
        route.setPolicyVersion(null);

        return route;
    }

    /**
     * Calculates distance between two nodes using Haversine formula.
     */
    private static double calculateDistance(NodeInfo from, NodeInfo to) {
        double R = 6371; // Earth radius in km
        double dLat = Math.toRadians(to.getPosition().getLatitude() - from.getPosition().getLatitude());
        double dLon = Math.toRadians(to.getPosition().getLongitude() - from.getPosition().getLongitude());
        double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(Math.toRadians(from.getPosition().getLatitude())) *
                        Math.cos(Math.toRadians(to.getPosition().getLatitude())) *
                        Math.sin(dLon / 2) * Math.sin(dLon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }
}
