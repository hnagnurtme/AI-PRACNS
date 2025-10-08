package com.sagin.core.service;

import com.sagin.core.INetworkManagerService;
import com.sagin.core.INodeService;
import com.sagin.core.IUserService;
import com.sagin.model.*;
import com.sagin.repository.INodeRepository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class NodeService implements INodeService {

    private static final Logger logger = LoggerFactory.getLogger(NodeService.class);

    private NodeInfo selfInfo; // Trạng thái cục bộ của node này
    private final Map<String, RouteInfo> routingTable = new ConcurrentHashMap<>(); // Bảng định tuyến cục bộ
    private final Map<String, Packet> packetBuffer = new ConcurrentHashMap<>(); // Buffer gói tin đang chờ xử lý

    // Dependencies
    private final INetworkManagerService networkManager;
    private final IUserService userService;
    private final INodeRepository nodeRepository;

    // Lịch trình cho vòng lặp cập nhật trạng thái
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
    // private static final long STATE_UPDATE_INTERVAL_MS = 1000; // Không cần thiết
    // nếu đã dùng scheduler

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
        // Đăng ký Node này với Network Manager
        networkManager.registerActiveNode(selfInfo.getNodeId(), this);

        // Bắt đầu vòng lặp cập nhật trạng thái định kỳ
        scheduler.scheduleAtFixedRate(this::updateNodeState, 1, 1, TimeUnit.SECONDS);
        logger.info("[Node {}] Bắt đầu vòng lặp xử lý trạng thái.", selfInfo.getNodeId());
    }

    /** @inheritdoc */
    @Override
    public void updateNodeState() {
        // 1. Mô phỏng thay đổi trạng thái Pin/Tài nguyên/Vị trí
        double currentBattery = selfInfo.getBatteryChargePercent();
        if (currentBattery > 0) {
            selfInfo.setBatteryChargePercent(Math.max(0.0, currentBattery - 0.1)); // Giảm pin chậm
        }

        // 2. Cập nhật lastUpdated
        selfInfo.setLastUpdated(System.currentTimeMillis());

        // 3. Đồng bộ hóa trạng thái lên Database (Firestore)
        nodeRepository.updateNodeInfo(selfInfo.getNodeId(), selfInfo);
        logger.trace("Trạng thái Node {} được cập nhật và đồng bộ hóa (Pin: {}).", selfInfo.getNodeId(),
                selfInfo.getBatteryChargePercent());
    }

    // @inheritdoc
    @Override
    public void updateRoute(String destinationId, RouteInfo route) {
        if (route == null) {
            this.routingTable.remove(destinationId);
            logger.warn("[Node {}] Xóa tuyến đường đến {}.", selfInfo.getNodeId(), destinationId);
        } else {
            // Cập nhật hoặc thêm tuyến đường mới
            this.routingTable.put(destinationId, route);

            // --- LOG BẢNG ĐỊNH TUYẾN MỚI ---
            logger.info("[Node {}] ĐÃ CẬP NHẬT TUYẾN ĐƯỜNG MỚI:", selfInfo.getNodeId());

            // Log tuyến đường cụ thể vừa được cập nhật
            logger.info("  -> Đích {}: {}", destinationId, route.toString());

            // HOẶC log toàn bộ bảng (nếu cần thiết):
            logger.debug("Toàn bộ bảng định tuyến: {}", routingTable.toString());
        }
    }

    /** @inheritdoc */
    @Override
    public NodeInfo getNodeInfo() {
        return selfInfo;
    }

    // --- XỬ LÝ GÓI TIN ---

    /** @inheritdoc */
    @Override
    public void receivePacket(Packet packet, LinkMetric incomingLinkMetric) {
        // 1. Cập nhật Độ trễ tích lũy và TTL
        packet.decrementTTL();
        packet.setAccumulatedDelayMs(packet.getAccumulatedDelayMs() + incomingLinkMetric.getLatencyMs()
                + selfInfo.getNodeProcessingDelayMs());

        // 2. KIỂM TRA DROP CỤC BỘ

        // A. Kiểm tra TTL
        if (!packet.isAlive()) {
            packet.markDropped("TTL Expired");
            logger.warn("[Node {}] DROP: Gói {} - TTL Expired.", selfInfo.getNodeId(), packet.getPacketId());
            return;
        }

        // B. Kiểm tra Đích cuối cùng
        if (packet.getDestinationUserId().equals(selfInfo.getNodeId())) {
            handleFinalDestination(packet);
            return;
        }

        // C. Kiểm tra Buffer (Congestion)
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

        // 4. Quyết định Định tuyến (Gói tin đã vào buffer)
        sendPacket(packet);
    }

    // Trong com.sagin.core.service.NodeService.java

    /**
     * Xử lý gói tin sau khi nó đã vượt qua tất cả các kiểm tra và được lưu trong
     * Buffer (hoặc mới sinh ra).
     * Đây là điểm ra quyết định định tuyến chính.
     * 
     * @inheritdoc
     */
    @Override
    public void sendPacket(Packet packet) {

        // 0. KHỞI TẠO GÓI TIN MỚI (Nếu nhận từ Client/Gateway)
        // Kiểm tra nếu gói tin chưa có lịch sử đường đi (mới sinh ra)
        if (packet.getPathHistory() == null || packet.getPathHistory().isEmpty()) {

            // Lấy và Gán các yêu cầu QoS (Chỉ cần làm một lần)
            ServiceQoS initialQoS = userService.getQoSForPacket(packet);
            packet.setMaxAcceptableLatencyMs(initialQoS.getMaxLatencyMs());
            packet.setMaxAcceptableLossRate(initialQoS.getMaxLossRate());

            // Đặt TTL ban đầu
            if (packet.getTTL() <= 0) {
                packet.setTTL(15);
            }
        }

        // 1. Tra cứu RouteInfo từ bảng định tuyến cục bộ
        String destinationId = packet.getDestinationUserId();
        RouteInfo routeInfo = routingTable.get(destinationId);

        logger.info("[Node {}] Định tuyến gói {}: Tra cứu tuyến đến {}. Bảng có {} mục.",
                selfInfo.getNodeId(), packet.getPacketId(), destinationId, routingTable.size());

        // 2. KIỂM TRA LỖI ROUTING (ROUTE NOT FOUND)

        // Kiểm tra:
        // a) RouteInfo có tồn tại không?
        // b) Next Hop có bị trỏ về Node hiện tại (Self-loop) không?
        // c) Chi phí có phải là vô hạn (MAX_VALUE) không?
        if (routeInfo == null ||
                routeInfo.getNextHopNodeId() == null ||
                routeInfo.getNextHopNodeId().equals(selfInfo.getNodeId()) || // Ngăn chặn self-loop (Route Not Found
                                                                             // Logic)
                routeInfo.getTotalCost() == Double.MAX_VALUE) {
            // Gói tin bị DROP do không có đường đi hợp lệ.
            logger.error("[Node {}] DROP: Gói {} - Route Not Found (Dest: {}). Bảng định tuyến trống hoặc lỗi.",
                    selfInfo.getNodeId(), packet.getPacketId(), destinationId);

            packet.markDropped("Route Not Found/Stale");
            packetBuffer.remove(packet.getPacketId());
            selfInfo.setCurrentPacketCount(selfInfo.getCurrentPacketCount() - 1);
            updateNodeState();
            return;
        }

        // 3. KIỂM TRA QoS (Admission Control)

        // Lấy lại QoS (hoặc sử dụng các giá trị đã gán trong bước 0)
        double maxLatency = packet.getMaxAcceptableLatencyMs();
        double minBandwidth = routeInfo.getMinBandwidthMbps();

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

    /**
     * Xử lý khi gói tin đến đích cuối cùng (Trạm Mặt đất Đích).
     */
    private void handleFinalDestination(Packet packet) {
        // Gói tin đến đích cuối cùng (Trạm Mặt đất đích)

        // Xóa gói tin khỏi buffer (nếu nó được đưa vào)
        if (packetBuffer.containsKey(packet.getPacketId())) {
            packetBuffer.remove(packet.getPacketId());
            selfInfo.setCurrentPacketCount(selfInfo.getCurrentPacketCount() - 1);
            updateNodeState();
        }

        // 1. Xử lý Gói tin DATA
        if (packet.getType() == Packet.PacketType.DATA) {
            logger.info("[Node {}] SUCCESS: Gói DATA {} đã đến đích (Total Delay: {}ms).",
                    selfInfo.getNodeId(), packet.getPacketId(), (int) packet.getAccumulatedDelayMs());

            // 2. Tạo Gói tin ACK (Chiều ngược lại)
            Packet ackPacket = createAckPacket(packet);

            // 3. Gửi ACK đi (ACK bắt đầu hành trình từ đây)
            // LƯU Ý: ACK là gói tin mới nên nó cần được đưa vào luồng gửi đi thông qua
            // sendPacket().
            sendPacket(ackPacket);

        }
        // 4. Xử lý Gói tin ACK
        else if (packet.getType() == Packet.PacketType.ACK) {
            logger.info("[Node {}] ACK RECEIVED: Gói {} đã được xác nhận. (ACK ID: {}).",
                    selfInfo.getNodeId(), packet.getAcknowledgedPacketId(), packet.getPacketId());
            // TODO: Logic: Dừng Timer Retransmission cho gói tin ACKED_PACKET_ID
        }
    }

    /**
     * Tạo gói tin ACK đảo ngược.
     */
    private Packet createAckPacket(Packet dataPacket) {
        Packet ack = new Packet();
        // Cần truyền đủ thông tin để nó có thể được định tuyến ngược lại
        ack.setType(Packet.PacketType.ACK);
        ack.setPacketId("ACK_" + dataPacket.getPacketId() + "_" + System.currentTimeMillis());
        ack.setAcknowledgedPacketId(dataPacket.getPacketId());
        ack.setSourceUserId(dataPacket.getDestinationUserId());
        ack.setDestinationUserId(dataPacket.getSourceUserId());

        // Các giá trị QoS/TTL không quan trọng lắm, nhưng cần được đặt
        ack.setServiceType(dataPacket.getServiceType());
        ack.setTTL(15);
        ack.setTimeSentFromSourceMs(System.currentTimeMillis());
        ack.setCurrentHoldingNodeId(selfInfo.getNodeId());

        return ack;
    }
}