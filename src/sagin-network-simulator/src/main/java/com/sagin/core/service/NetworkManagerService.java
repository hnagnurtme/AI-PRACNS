package com.sagin.core.service;

import com.sagin.core.ILinkManagerService;
import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeService;
import com.sagin.model.*;
import com.sagin.repository.INodeRepository;
import com.sagin.routing.IRoutingEngine;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ThreadLocalRandom;

public class NetworkManagerService implements INetworkManagerService {

    // Registry của các Node đang hoạt động (NodeId -> INodeService Instance)
    private final Map<String, INodeService> activeNodesRegistry = new ConcurrentHashMap<>();
    // Cache Topology toàn mạng (Được cập nhật bởi các Node qua DB và đẩy vào
    // Manager)
    private final Map<String, NodeInfo> networkTopologyCache = new ConcurrentHashMap<>();

    // Dependencies
    private final ILinkManagerService linkManager;
    private final INodeRepository nodeRepository;
    private final IRoutingEngine routingEngine;

    // Lịch trình cho vòng lặp mô phỏng định kỳ
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
    private static final long ROUTING_UPDATE_INTERVAL_SECONDS = 20; // Cập nhật định tuyến 20 giây một lần

    // logger
    private static final Logger logger = LoggerFactory.getLogger(NetworkManagerService.class);

    public NetworkManagerService(
            ILinkManagerService linkManager,
            INodeRepository nodeRepository,
            IRoutingEngine routingEngine) {
        this.linkManager = linkManager;
        this.nodeRepository = nodeRepository;
        this.routingEngine = routingEngine;
    }

    // --- Phương thức Quản lý Vòng lặp ---

    /**
     * Khởi tạo toàn bộ mạng lưới và bắt đầu vòng lặp tính toán định tuyến định kỳ.
     * 
     * @param initialNodeConfigs Cấu hình ban đầu tải từ DB.
     * @param baseQoS            Base QoS để tính toán bảng định tuyến Proactive.
     */
    public void startNetworkSimulation(Map<String, NodeInfo> initialNodeConfigs, ServiceQoS baseQoS) {
        initializeNetwork(initialNodeConfigs);
        this.networkTopologyCache.putAll(nodeRepository.loadAllNodeConfigs());

        // --- BƯỚC 1: BUỘC TÍNH TOÁN VÀ PHÂN PHỐI ROUTING NGAY LẬP TỨC ---
        // Giúp các Node có RoutingTable ngay khi khởi động.
        syncCacheWithDatabase();
        triggerRoutingComputation(baseQoS);
        // ------------------------------------------------------------------

        // --- BƯỚC 2: ĐẶT LỊCH CHO CÁC LẦN CẬP NHẬT TIẾP THEO ---
        scheduler.scheduleAtFixedRate(
                () -> {
                    // Đồng bộ hóa Cache và Kích hoạt tính toán định tuyến
                    syncCacheWithDatabase();
                    triggerRoutingComputation(baseQoS);
                },
                ROUTING_UPDATE_INTERVAL_SECONDS, // initial delay (Chờ 20s sau lần tính toán đầu tiên)
                ROUTING_UPDATE_INTERVAL_SECONDS,
                TimeUnit.SECONDS);
        System.out.printf("[Manager] Vòng lặp tính toán định tuyến (interval %d s) đã khởi động.%n",
                ROUTING_UPDATE_INTERVAL_SECONDS);
    }

    /**
     * Dừng vòng lặp mô phỏng.
     */
    public void stopSimulation() {
        scheduler.shutdownNow();
    }

    // --- Triển khai INetworkManagerService ---

    /** @inheritdoc */
    @Override
    public void initializeNetwork(Map<String, NodeInfo> initialNodeConfigs) {
        // Tải cấu hình Node ban đầu vào cache Topology
        this.networkTopologyCache.putAll(initialNodeConfigs);
        System.out.printf("[Manager] Đã tải %d cấu hình Node vào Topology Cache.%n", initialNodeConfigs.size());
    }

    /** @inheritdoc */
    @Override
    public void registerActiveNode(String nodeId, INodeService nodeService) {
        activeNodesRegistry.put(nodeId, nodeService);
        System.out.printf("[Manager] Node %s đã đăng ký và hoạt động.%n", nodeId);
    }

    /** @inheritdoc */
    public void updateNodeCache(NodeInfo info) {
        // Cập nhật trạng thái cục bộ của node vào cache toàn mạng
        this.networkTopologyCache.put(info.getNodeId(), info);
    }

    @Override
    public void transferPacket(Packet packet, String sourceNodeId) {
        String destNodeId = packet.getNextHopNodeId();

        NodeInfo sourceInfo = getNodeInfo(sourceNodeId);
        NodeInfo destInfo = getNodeInfo(destNodeId); 

        // 1. Kiểm tra tính khả dụng của Node Info (Giữ nguyên)
        if (sourceInfo == null || destInfo == null || !sourceInfo.isOperational() || !destInfo.isOperational()) {
            logger.error("[Manager] LỖI GIAO TIẾP: Node nguồn hoặc đích không khả dụng. Gói {} bị DROP.", packet.getPacketId());
            packet.markDropped("Dest Node Info Missing/Offline");
            return;
        }

        // 2. Tính toán Link Metric thực tế
        LinkMetric linkMetric = linkManager.calculateLinkMetric(sourceInfo, destInfo);

        if (!linkMetric.isLinkActive()) {
            packet.markDropped("Link Down / Visibility Lost");
            return;
        }

        // 3. Mô phỏng Độ trễ (Trì hoãn gói tin)
        long delayMs = (long) linkMetric.getLatencyMs();

        scheduler.schedule(() -> {
            // Kiểm tra Mất gói
            if (ThreadLocalRandom.current().nextDouble() < linkMetric.getPacketLossRate()) {
                // Gói tin bị mất trên đường truyền
                logger.warn("[Manager] Gói {} bị MẤT trên đường truyền từ {} đến {}.", packet.getPacketId(), sourceNodeId, destNodeId);
                return;
            }

            boolean success = RemotePacketSender.sendPacketViaSocket(packet, destInfo, linkMetric);

            if (success) {
                // Ghi nhận thành công
                logger.info("[Manager] Gói {} đã được chuyển từ {} đến {} (Delay: {} ms).",
                        packet.getPacketId(), sourceNodeId, destNodeId, delayMs);
            } else {
                // Xử lý lỗi khi Node đích không phản hồi Socket (thực sự OFFLINE)
                logger.error("[Manager] LỖI GIAO TIẾP: Gói {} không thể gửi tới {} (Socket Unreachable).",
                        packet.getPacketId(), destNodeId);
                packet.markDropped("NextHop Socket Unreachable");
            }

        }, delayMs, TimeUnit.MILLISECONDS);
    }

    /** @inheritdoc */
    @Override
    public NodeInfo getNodeInfo(String nodeId) {
        return networkTopologyCache.get(nodeId);
    }

    /** @inheritdoc */
    @Override
    public Map<String, NodeInfo> getAllNodeInfos() {
        return new HashMap<>(networkTopologyCache);
    }

    // --- Logic Tính toán Định tuyến Định kỳ ---

    /**
     * Kích hoạt việc tính toán RoutingTable toàn mạng cho tất cả các node.
     * 
     * @param baseQoS Base QoS để tính toán.
     */
    private void triggerRoutingComputation(ServiceQoS baseQoS) {
        Map<String, NodeInfo> currentTopology = getAllNodeInfos();
        Map<String, LinkMetric> allLinks = new ConcurrentHashMap<>();

        // 1. TÍNH TOÁN TOÀN BỘ CÁC LINK GIỮA CÁC NODE ĐANG HOẠT ĐỘNG
        for (NodeInfo source : currentTopology.values()) {
            for (NodeInfo dest : currentTopology.values()) {
                if (!source.getNodeId().equals(dest.getNodeId())) {
                    LinkMetric metric = linkManager.calculateLinkMetric(source, dest);
                    if (metric.isLinkActive()) {
                        allLinks.put(source.getNodeId() + "-" + dest.getNodeId(), metric);
                    }
                }
            }
        }

        // 2. KÍCH HOẠT ROUTING ENGINE CHO MỖI NODE
        for (NodeInfo sourceNode : currentTopology.values()) {
            // Chỉ tính toán cho các node đang hoạt động và đăng ký
            if (sourceNode.isOperational() && activeNodesRegistry.containsKey(sourceNode.getNodeId())) {

                RoutingTable table = routingEngine.computeRoutes(
                        sourceNode,
                        allLinks,
                        currentTopology,
                        baseQoS);

                // 3. ĐẨY BẢNG ĐỊNH TUYẾN MỚI VỀ CHO INodeService
                INodeService sourceService = activeNodesRegistry.get(sourceNode.getNodeId());
                if (sourceService != null) {
                    for (Map.Entry<String, RouteInfo> entry : table.getRouteInfoMap().entrySet()) {
                        sourceService.updateRoute(entry.getKey(), entry.getValue());
                    }
                }
            }
        }
        System.out.printf("[Manager] Hoàn thành cập nhật định tuyến toàn mạng. %d links.%n", allLinks.size());
    }

    /**
     * Tải lại toàn bộ cấu hình Node từ Repository (Firestore) để đồng bộ hóa cache.
     * Điều này đảm bảo NetworkManager luôn có NodeInfo mới nhất (Buffer/Pin)
     * mà các INodeService khác đã ghi lên DB.
     */
    private void syncCacheWithDatabase() {
        try {
            Map<String, NodeInfo> latestConfigs = nodeRepository.loadAllNodeConfigs();
            this.networkTopologyCache.clear();
            this.networkTopologyCache.putAll(latestConfigs);
        } catch (Exception e) {
            System.err.println("[Manager] LỖI đồng bộ hóa với Database: " + e.getMessage());
        }
    }
}