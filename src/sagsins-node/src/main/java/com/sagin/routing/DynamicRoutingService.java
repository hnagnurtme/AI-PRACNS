package com.sagin.routing;

import com.sagin.model.NodeInfo;
import com.sagin.service.INodeService;
import com.sagin.repository.INodeRepository;

import java.util.*;
import java.util.concurrent.*;

/**
 * DynamicRoutingService tối ưu:
 * - Precompute routing table cho tất cả node.
 * - Scheduler update liên tục.
 * - Packet chỉ lookup bảng định tuyến đã có.
 */
public class DynamicRoutingService implements IRoutingService {

    private final INodeRepository nodeRepository;
    private final INodeService nodeService;

    // Map nodeId -> RoutingTable
    private final Map<String, RoutingTable> routingTables = new ConcurrentHashMap<>();

    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

    public DynamicRoutingService(INodeRepository nodeRepository, INodeService nodeService) {
        this.nodeRepository = nodeRepository;
        this.nodeService = nodeService;

        scheduler.scheduleAtFixedRate(this::updateAllRoutingTables, 0, 5, TimeUnit.SECONDS);
    }

    @Override
    public RouteInfo getBestRoute(String currentNodeId, String destinationNodeId) {
        if (currentNodeId == null || destinationNodeId == null) return null;
        RoutingTable table = routingTables.get(currentNodeId);
        return table != null ? table.getBestRoute(destinationNodeId) : null;
    }

    @Override
    public RoutingTable getRoutingTableForNode(String nodeId) {
        return routingTables.getOrDefault(nodeId, new RoutingTable());
    }

    @Override
    public void updateRoute(String forNodeId, RouteInfo newRoute) {
        routingTables.computeIfAbsent(forNodeId, k -> new RoutingTable())
                        .updateRoute(newRoute);
    }

    /**
     * Scheduler: tính toàn bộ routing table cho tất cả node
     */
    private void updateAllRoutingTables() {
        List<NodeInfo> allNodes = new ArrayList<>(nodeRepository.loadAllNodeConfigs().values());

        // 1 adjacency list chung cho tất cả nodes
        Map<String, List<String>> graph = buildAdjacencyList(allNodes);

        for (NodeInfo srcNode : allNodes) {
            if (!srcNode.isHealthy()) {
                continue;
            }
            RoutingTable table = computeRoutingTable(srcNode.getNodeId(), graph);
            routingTables.put(srcNode.getNodeId(), table);
        }
    }

    private Map<String, List<String>> buildAdjacencyList(List<NodeInfo> nodes) {
        Map<String, List<String>> graph = new HashMap<>();
        for (NodeInfo node : nodes) {
            if (!node.isHealthy()) continue;
            List<String> visibleNeighbors = nodeService.getVisibleNodes(node, nodes).stream()
                    .filter(NodeInfo::isHealthy)
                    .map(NodeInfo::getNodeId)
                    .toList();
            graph.put(node.getNodeId(), visibleNeighbors);
        }
        return graph;
    }

    private RoutingTable computeRoutingTable(String sourceNodeId, Map<String, List<String>> graph) {
        RoutingTable table = new RoutingTable();

        // BFS không trọng số, precompute tất cả destination từ source
        Queue<List<String>> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        queue.add(List.of(sourceNodeId));

        while (!queue.isEmpty()) {
            List<String> path = queue.poll();
            String last = path.get(path.size() - 1);

            if (!visited.add(last)) continue;

            if (!last.equals(sourceNodeId)) {
                table.updateRoute(RouteHelper.createBasicRoute(sourceNodeId, last, path));
            }

            for (String neighbor : graph.getOrDefault(last, List.of())) {
                if (!visited.contains(neighbor)) {
                    List<String> newPath = new ArrayList<>(path);
                    newPath.add(neighbor);
                    queue.add(newPath);
                }
            }
        }

        return table;
    }
    public void forceUpdateRoutingTables() {
        updateAllRoutingTables();
    }


    public void shutdown() {
        scheduler.shutdownNow();
    }
}
