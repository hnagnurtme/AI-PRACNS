package com.sagin.core.service;

import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeService;
import com.sagin.core.IUserService;
import com.sagin.model.*;
import com.sagin.repository.INodeRepository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class NodeService implements INodeService {

    private static final Logger logger = LoggerFactory.getLogger(NodeService.class);

    private NodeInfo selfInfo;
    private final Map<String, RouteInfo> routingTable = new ConcurrentHashMap<>();
    private final Map<String, Packet> packetBuffer = new ConcurrentHashMap<>();

    private final INetworkManagerService networkManager;
    private final IUserService userService;
    private final INodeRepository nodeRepository;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

    public NodeService(NodeInfo initialInfo, INetworkManagerService networkManager, IUserService userService,
            INodeRepository nodeRepository) {
        this.selfInfo = initialInfo;
        this.networkManager = networkManager;
        this.userService = userService;
        this.nodeRepository = nodeRepository;
    }

    // --- CÁC HÀM CƠ CHẾ VÀ ĐỒNG BỘ ---

    /** @inheritdoc */
    @Override
    public void startSimulationLoop() {
        networkManager.registerActiveNode(selfInfo.getNodeId(), this);

        List<String> loopbackPath = Collections.singletonList(selfInfo.getNodeId());
        RouteInfo loopbackRoute = new RouteInfo(
                selfInfo.getNodeId(), // nextHopNodeId
                loopbackPath, // pathNodeIds
                0.0, // totalCost (Chi phí bằng 0)
                0.0, // totalLatencyMs (Độ trễ bằng 0)
                1000.0, // minBandwidthMbps (Mức an toàn, không phải bottleneck)
                0.0, // avgPacketLossRate (Không mất gói)
                System.currentTimeMillis() // timestampComputed (Thời điểm hiện tại)
        );
        // create loopback route
        routingTable.put(selfInfo.getNodeId(), loopbackRoute);
        logger.info("[Node {}] ĐÃ TẠO TUYẾN NỘI BỘ: {}", selfInfo.getNodeId(), loopbackPath.toString());
        scheduler.scheduleAtFixedRate(this::updateNodeState, 1, 1, TimeUnit.SECONDS);
        logger.info("[Node {}] Bắt đầu vòng lặp xử lý trạng thái.", selfInfo.getNodeId());
    }

    /** @inheritdoc */
    @Override
    public void updateNodeState() {
        if (selfInfo.getNodeType() == NodeType.GROUND_STATION) {
            selfInfo.setBatteryChargePercent(100.0);
        }

        double currentBattery = selfInfo.getBatteryChargePercent();
        if (currentBattery > 0) {
            if (selfInfo.getNodeType() == NodeType.GEO_SATELLITE || selfInfo.getNodeType() == NodeType.LEO_SATELLITE
                    || selfInfo.getNodeType() == NodeType.MEO_SATELLITE) {
                selfInfo.setBatteryChargePercent(Math.max(0.0, currentBattery - 0.1));
            }
        }

        selfInfo.setLastUpdated(System.currentTimeMillis());
        nodeRepository.updateNodeInfo(selfInfo.getNodeId(), selfInfo);
    }

    // @inheritdoc
    @Override
    public void updateRoute(String destinationId, RouteInfo route) {
        if (route == null) {
            this.routingTable.remove(destinationId);
            logger.warn("[Node {}] Xóa tuyến đường đến {}.", selfInfo.getNodeId(), destinationId);
        } else {
            this.routingTable.put(destinationId, route);
        }
    }

    /** @inheritdoc */
    @Override
    public NodeInfo getNodeInfo() {
        return selfInfo;
    }

    /** @inheritdoc */
    @Override
    public void receivePacket(Packet packet, LinkMetric incomingLinkMetric) {
        
        if (incomingLinkMetric == null) {
            // --- GÓI TIN MỚI SINH RA (Từ Gateway/Client) ---

            // a) Khởi tạo QoS/TTL (Đảm bảo gói tin có các ràng buộc QoS và TTL)
            if (packet.getMaxAcceptableLatencyMs() == 0.0 || packet.getTTL() <= 0) {
                ServiceQoS initialQoS = userService.getQoSForPacket(packet);
                packet.setMaxAcceptableLatencyMs(initialQoS.getMaxLatencyMs());
                packet.setMaxAcceptableLossRate(initialQoS.getMaxLossRate());

                if (packet.getTTL() <= 0) {
                    packet.setTTL(15);
                }
            }
            // b) Khởi tạo Độ trễ tích lũy ban đầu
            packet.setAccumulatedDelayMs(selfInfo.getNodeProcessingDelayMs());

            logger.error("[Node {}] NHẬN: Gói mới {} (Type: {}) từ Client/Gateway. Source : {}, Dest: {}",
                    selfInfo.getNodeId(), packet.getPacketId(), packet.getType(), packet.getSourceUserId(),
                    packet.getDestinationUserId());

        } else {
            // --- GÓI TIN CHUYỂN TIẾP (Từ Node khác qua Link) ---

            double score = incomingLinkMetric.calculateLinkScore();
            if (score <= 0.0) {
                logger.warn(
                        "[Node {}] DROP: Gói {} bị từ chối. LinkScore đến bằng 0. Link không hoạt động/Không đáng tin cậy.",
                        selfInfo.getNodeId(), packet.getPacketId());
                packet.markDropped("Incoming Link Unusable");
                return; // DROP và THOÁT
            }

            // 2. Cập nhật Độ trễ tích lũy
            packet.setAccumulatedDelayMs(packet.getAccumulatedDelayMs() + incomingLinkMetric.getLatencyMs()
                    + selfInfo.getNodeProcessingDelayMs());
            // 3. GIẢM TTL VÀ KIỂM TRA (CHO GÓI CHUYỂN TIẾP)
            logger.info("[Node {}] NHẬN: Gói {} (Type: {}) từ Link. TTL ban đầu: {}.",
                    selfInfo.getNodeId(), packet.getPacketId(), packet.getType(), packet.getTTL());
            // LOG TTL TRƯỚC KHI GIẢM ĐỂ KIỂM TRA LỖI LOGIC
    
            // GIẢM TTL CHỈ SAU KHI LOG VÀ KIỂM TRA
            packet.decrementTTL();

            // A. Kiểm tra TTL (Cho gói chuyển tiếp sau khi đã giảm)
            if (!packet.isAlive()) {
                packet.markDropped("TTL Expired");
                logger.warn("[Node {}] DROP: Gói {} - TTL Expired.", selfInfo.getNodeId(), packet.getPacketId());
                return; // DROP và THOÁT
            }
        }

        // --- LOGIC XỬ LÝ CHUNG (Áp dụng cho CẢ HAI) ---

        // B. Kiểm tra Đích cuối cùng
        if (packet.getDestinationUserId().equals(selfInfo.getNodeId())) {
            try {
                handleFinalDestination(packet);
                logger.info("[Node {}] Gói {} đã đến đích cuối cùng (Dest: {}).",
                        selfInfo.getNodeId(), packet.getPacketId(), packet.getDestinationUserId());
            } catch (Exception e) {
                logger.error("[Node {}] LỖI khi xử lý Gói {} tại Đích Cuối Cùng: {}",
                        selfInfo.getNodeId(), packet.getPacketId(), e.getMessage());
            }
            // Gói tin đã được xử lý xong, thoát luồng.
            return;
        }

        // C. Kiểm tra Sức khỏe Node (CHỈ CHO GÓI MỚI - Initial Admission)
        if (incomingLinkMetric == null && !selfInfo.isHealthy()) {
            logger.warn("[Node {}] DROP: Gói {} bị từ chối. Node không khỏe (Pin/Tải quá tải).",
                    selfInfo.getNodeId(), packet.getPacketId());
            packet.markDropped("Node Unhealthy (Initial Reject)");
            return;
        }

        // D. Kiểm tra Buffer (Congestion)
        if (selfInfo.getCurrentPacketCount() >= selfInfo.getPacketBufferCapacity()) {
            packet.markDropped("Buffer Overflow");
            logger.warn("[Node {}] DROP: Gói {} - Buffer Full (Tải: {}/{})",
                    selfInfo.getNodeId(), packet.getPacketId(), selfInfo.getCurrentPacketCount(),
                    selfInfo.getPacketBufferCapacity());
            return;
        }

        // 3. Đưa gói tin vào Buffer (Tăng tải buffer và cập nhật DB)
        packetBuffer.put(packet.getPacketId(), packet);
        selfInfo.setCurrentPacketCount(selfInfo.getCurrentPacketCount() + 1);
        updateNodeState();

        // 4. Quyết định Định tuyến
        sendPacket(packet);
    }

    // --- ĐỊNH TUYẾN/CHUYỂN TIẾP (FORWARDING) ---

    /** @inheritdoc */
    @Override
    public void sendPacket(Packet packet) {

        String destinationId = packet.getDestinationUserId();
        RouteInfo routeInfo = routingTable.get(destinationId);

        logger.info("[Node {}] Định tuyến gói {}: Tra cứu tuyến đến {}. Bảng có {} mục.",
                selfInfo.getNodeId(), packet.getPacketId(), destinationId, routingTable.size());

        // 2. KIỂM TRA LỖI ROUTING (ROUTE NOT FOUND)

        boolean routeIsInvalid = routeInfo == null ||
                routeInfo.getNextHopNodeId() == null ||
                routeInfo.getNextHopNodeId().equals(selfInfo.getNodeId()) ||
                routeInfo.getTotalCost() == Double.MAX_VALUE;

        if (routeIsInvalid) {
            
            // --- LOGIC HOLD/GIỮ LẠI CHO GÓI ACK ---
            if (packet.getType() == Packet.PacketType.ACK) {
                logger.warn("[Node {}] HOLD: Gói ACK {} - Tuyến đường đến {} chưa sẵn sàng. Giữ lại buffer (Tải: {}).",
                            selfInfo.getNodeId(), packet.getPacketId(), destinationId, selfInfo.getCurrentPacketCount());
                // Gói tin vẫn nằm trong packetBuffer. Nó sẽ được xử lý lại sau.
                return; // THOÁT KHỎI HÀM
            }
            // --- END LOGIC ACK ---

            // Logic DROP cho Gói DATA/Gói tin thông thường
            logger.error("[Node {}] DROP: Gói {} - Route Not Found (Dest: {}). Bảng định tuyến trống hoặc lỗi.",
                    selfInfo.getNodeId(), packet.getPacketId(), destinationId);

            packet.markDropped("Route Not Found/Stale");
            packetBuffer.remove(packet.getPacketId());
            selfInfo.setCurrentPacketCount(selfInfo.getCurrentPacketCount() - 1);
            updateNodeState();
            return;
        }
        // if (routeInfo == null ||
        //         routeInfo.getNextHopNodeId() == null ||
        //         routeInfo.getNextHopNodeId().equals(selfInfo.getNodeId()) ||
        //         routeInfo.getTotalCost() == Double.MAX_VALUE) {
        //     logger.error("[Node {}] DROP: Gói {} - Route Not Found (Dest: {}). Bảng định tuyến trống hoặc lỗi.",
        //             selfInfo.getNodeId(), packet.getPacketId(), destinationId);

        //     packet.markDropped("Route Not Found/Stale");
        //     packetBuffer.remove(packet.getPacketId());
        //     selfInfo.setCurrentPacketCount(selfInfo.getCurrentPacketCount() - 1);
        //     updateNodeState();
        //     return;
        // }

        // 3. KIỂM TRA QoS (Admission Control)
        double maxLatency = packet.getMaxAcceptableLatencyMs();

        if (routeInfo.getTotalLatencyMs() > maxLatency) {

            logger.warn("[Node {}] DROP: Gói {} - QoS Failed (Latency: {}ms > {}ms).",
                    selfInfo.getNodeId(), packet.getPacketId(), (int) routeInfo.getTotalLatencyMs(), (int) maxLatency);

            packet.markDropped("QoS Violation on Route");
            packetBuffer.remove(packet.getPacketId());
            selfInfo.setCurrentPacketCount(selfInfo.getCurrentPacketCount() - 1);
            updateNodeState();
            return;
        }

        // 4. Chuẩn bị và Gửi gói tin
        String nextHopId = routeInfo.getNextHopNodeId();

        // Cập nhật history và next hop
        packet.addToPath(selfInfo.getNodeId());
        packet.setNextHopNodeId(nextHopId);

        // 5. Gửi gói tin qua Network Manager (Kích hoạt RPC)
        networkManager.transferPacket(packet, selfInfo.getNodeId());

        // 6. Cập nhật Trạng thái Buffer (Gói tin đã rời khỏi buffer)
        packetBuffer.remove(packet.getPacketId());
        selfInfo.setCurrentPacketCount(selfInfo.getCurrentPacketCount() - 1);
        updateNodeState();
        logger.debug("[Node {}] GỬI: Gói {} (Type: {}) tới NextHop: {}. Path Cost: {}",
                selfInfo.getNodeId(), packet.getPacketId(), packet.getType(), nextHopId, routeInfo.getTotalCost());
    }

    // --- XỬ LÝ ĐÍCH CUỐI CÙNG ---

    /**
     * Xử lý khi gói tin đến đích cuối cùng (Trạm Mặt đất Đích).
     */
    private void handleFinalDestination(Packet packet) {
        // Xóa gói tin khỏi buffer an toàn trước khi xử lý
        if (packetBuffer.containsKey(packet.getPacketId())) {
            packetBuffer.remove(packet.getPacketId());
            selfInfo.setCurrentPacketCount(selfInfo.getCurrentPacketCount() - 1);
            updateNodeState();
        }

        try {
            if (packet.getType() == Packet.PacketType.DATA) {
                logger.info("[Node {}] SUCCESS: Gói DATA {} đã đến đích (Total Delay: {}ms).",
                        selfInfo.getNodeId(), packet.getPacketId(), (int) packet.getAccumulatedDelayMs());
                Packet ackPacket = createAckPacket(packet);
                sendPacket(ackPacket);

                logger.debug("[Node {}] Đã gửi gói ACK {} ngược lại cho gói DATA {}.",
                        selfInfo.getNodeId(), ackPacket.getPacketId(), packet.getPacketId());

            } else if (packet.getType() == Packet.PacketType.ACK) {
                logger.info("[Node {}] ACK RECEIVED: Gói {} đã được xác nhận. (ACK ID: {}).",
                        selfInfo.getNodeId(), packet.getAcknowledgedPacketId(), packet.getPacketId());
            }
        } catch (Exception e) {
            logger.error("[Node {}] LỖI FATAL: Xảy ra ngoại lệ khi xử lý tại đích cuối cùng cho gói {}. {}",
                    selfInfo.getNodeId(), packet.getPacketId(), e.getMessage(), e);
        }
    }

    /**
     * Tạo gói tin ACK đảo ngược.
     */
    private Packet createAckPacket(Packet dataPacket) {
    Packet ack = new Packet();
    ack.setType(Packet.PacketType.ACK);
    ack.setPacketId("ACK_" + dataPacket.getPacketId() + "_" + System.currentTimeMillis());
    ack.setAcknowledgedPacketId(dataPacket.getPacketId());
    
    // ĐẢO NGUỒN VÀ ĐÍCH
    ack.setSourceUserId(dataPacket.getDestinationUserId()); // Dest ban đầu -> Source mới
    ack.setDestinationUserId(dataPacket.getSourceUserId()); // Source ban đầu -> Dest mới

    ack.setServiceType(dataPacket.getServiceType());
    ack.setTTL(15);
    ack.setTimeSentFromSourceMs(System.currentTimeMillis());
    ack.setCurrentHoldingNodeId(selfInfo.getNodeId());
    
    // QUAN TRỌNG: KHỞI TẠO CÁC THÔNG SỐ TRẠNG THÁI MẠNG CẦN THIẾT CHO VIỆC ĐỊNH TUYẾN
    ack.setMaxAcceptableLatencyMs(dataPacket.getMaxAcceptableLatencyMs());
    ack.setMaxAcceptableLossRate(dataPacket.getMaxAcceptableLossRate());
    ack.setAccumulatedDelayMs(selfInfo.getNodeProcessingDelayMs()); // Đặt lại độ trễ ban đầu
    
    return ack;
}
}