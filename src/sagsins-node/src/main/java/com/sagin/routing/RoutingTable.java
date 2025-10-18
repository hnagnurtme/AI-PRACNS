package com.sagin.routing;

import lombok.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * RoutingTable lưu danh sách các tuyến đường (RouteInfo) đến các node khác trong mạng.
 * Hỗ trợ TTL, cập nhật từ RL Agent, và chọn tuyến tốt nhất dựa trên metric linh hoạt.
 */
@Data
@ToString
@AllArgsConstructor
public class RoutingTable {

    /**
     * Mỗi đích đến (destinationNodeId) có thể có nhiều route khả thi.
     * Key: destinationNodeId
     * Value: danh sách RouteInfo (có thể nhiều tuyến đường khác nhau)
     */
    private final Map<String, List<RouteInfo>> table = new ConcurrentHashMap<>();

    /**
     * Thêm hoặc cập nhật route vào bảng định tuyến.
     * Nếu route đã tồn tại, chỉ thay thế nếu route mới tốt hơn hoặc đã hết hạn.
     */
    public synchronized void updateRoute(RouteInfo newRoute) {
        if (newRoute == null || newRoute.getDestinationNodeId() == null) return;

        String dest = newRoute.getDestinationNodeId();
        List<RouteInfo> routes = table.getOrDefault(dest, new ArrayList<>());

        // Xoá các route đã hết hạn
        long now = System.currentTimeMillis();
        routes.removeIf(r -> r.getValidUntil() > 0 && r.getValidUntil() < now);

        // Kiểm tra xem có route tương tự (cùng path hoặc nextHop)
        Optional<RouteInfo> existing = routes.stream()
                .filter(r -> Objects.equals(r.getNextHopNodeId(), newRoute.getNextHopNodeId()))
                .findFirst();

        if (existing.isPresent()) {
            RouteInfo old = existing.get();
            if (isBetterRoute(newRoute, old)) {
                routes.remove(old);
                routes.add(newRoute);
            }
        } else {
            routes.add(newRoute);
        }

        table.put(dest, routes);
    }

    /**
     * Lấy route tốt nhất đến một node đích.
     * Tiêu chí mặc định: cost thấp + latency thấp + reliability cao.
     */
    public RouteInfo getBestRoute(String destinationNodeId) {
        List<RouteInfo> routes = table.get(destinationNodeId);
        if (routes == null || routes.isEmpty()) return null;

        long now = System.currentTimeMillis();
        routes.removeIf(r -> r.getValidUntil() > 0 && r.getValidUntil() < now);

        return routes.stream()
                .min(Comparator.comparingDouble(this::routeScore))
                .orElse(null);
    }

    /**
     * Tính "điểm" cho một route: càng thấp càng tốt.
     * Công thức có thể được điều chỉnh bởi RL hoặc scheduler.
     */
    private double routeScore(RouteInfo route) {
        return route.getTotalCost() * 0.4
                + route.getTotalLatencyMs() * 0.4
                - route.getReliabilityScore() * 100.0
                + route.getAvgPacketLossRate() * 10.0;
    }

    /**
     * Kiểm tra route mới có tốt hơn route cũ không (so sánh bằng routeScore)
     */
    private boolean isBetterRoute(RouteInfo newRoute, RouteInfo oldRoute) {
        return routeScore(newRoute) < routeScore(oldRoute);
    }

    /**
     * Dọn dẹp các route đã hết hạn TTL
     */
    public synchronized void cleanupExpiredRoutes() {
        long now = System.currentTimeMillis();
        for (List<RouteInfo> routes : table.values()) {
            routes.removeIf(r -> r.getValidUntil() > 0 && r.getValidUntil() < now);
        }
    }

    /**
     * Cập nhật reward từ Reinforcement Learning Agent sau khi truyền dữ liệu thành công/thất bại
     */
    public void updateReward(String destinationNodeId, String nextHopNodeId, double reward) {
        List<RouteInfo> routes = table.get(destinationNodeId);
        if (routes == null) return;

        for (RouteInfo route : routes) {
            if (route.getNextHopNodeId().equals(nextHopNodeId)) {
                route.setLastReward(reward);
                break;
            }
        }
    }

    /**
     * Trả về toàn bộ danh sách route đang hoạt động (hữu ích cho giám sát hoặc dashboard)
     */
    public List<RouteInfo> getActiveRoutes() {
        long now = System.currentTimeMillis();
        List<RouteInfo> active = new ArrayList<>();
        for (List<RouteInfo> routes : table.values()) {
            for (RouteInfo r : routes) {
                if (r.getValidUntil() == 0 || r.getValidUntil() > now) {
                    active.add(r);
                }
            }
        }
        return active;
    }
}
