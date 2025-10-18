package com.sagin.network.implement;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sagin.helper.PacketHelper;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.model.UserInfo;
import com.sagin.network.interfaces.ITCP_Service;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.service.INodeService;
import com.sagin.routing.IRoutingService;
import com.sagin.routing.RouteInfo;
import com.sagin.util.SimulationConstants;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.Optional;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Dịch vụ TCP thực thi, có hỗ trợ LẬP LỊCH GỬI LẠI (Retry)
 * và logic ĐỊNH TUYẾN (Routing) phức tạp.
 *
 * Tách biệt rõ ràng hạch toán tài nguyên Nhận (RX) và Gửi (TX).
 */
public class TCP_Service implements ITCP_Service {
    private static final Logger logger = LoggerFactory.getLogger(TCP_Service.class);

    // --- Các Dependency (Dịch vụ phụ thuộc) ---
    private final INodeRepository nodeRepository;
    private final IUserRepository userRepository;
    private final INodeService nodeService;
    private final IRoutingService routingService;
    private final ObjectMapper objectMapper;

    // --- Hàng đợi Gửi lại (Retry Queue) ---
    private final BlockingQueue<RetryablePacket> sendQueue;
    private final ScheduledExecutorService retryScheduler;
    private static final int MAX_RETRIES = 5; // Số lần thử lại tối đa
    private static final long RETRY_POLL_INTERVAL_MS = 500; // Nửa giây quét hàng đợi 1 lần

    /**
     * Đối tượng nội bộ để đóng gói packet và số lần thử.
     * Chứa 'originalNodeId' để biết node nào cần bị trừ tài nguyên (TX)
     * sau khi gửi thành công.
     */
    private record RetryablePacket(
            String originalNodeId,
            Packet packet,
            String host,
            int port,
            String destinationDesc,
            int attemptCount) {
    }

    /**
     * Khởi tạo TCP_Service với tất cả các dịch vụ phụ thuộc.
     */
    public TCP_Service(INodeRepository nodeRepository,
            INodeService nodeService,
            IUserRepository userRepository,
            IRoutingService routingService) {
        this.nodeRepository = nodeRepository;
        this.nodeService = nodeService;
        this.userRepository = userRepository;
        this.routingService = routingService;

        // Khởi tạo ObjectMapper và đăng ký module JavaTime (cho Instant, v.v.)
        this.objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());

        // Khởi tạo hàng đợi và dịch vụ chạy nền
        this.sendQueue = new LinkedBlockingQueue<>();
        this.retryScheduler = Executors.newSingleThreadScheduledExecutor();

        // Bắt đầu chạy thread xử lý hàng đợi
        this.startSendScheduler();
    }

    @Override
    public void receivePacket(Packet packet) {
        if (packet == null || packet.getCurrentHoldingNodeId() == null) {
            logger.warn("[TCP_Service] Nhận được packet không hợp lệ.");
            return;
        }

        String currentNodeId = packet.getCurrentHoldingNodeId();
        logger.info("[TCP_Service] Đã nhận Packet {} tại {}.", packet.getPacketId(), currentNodeId);

        // --- Hạch toán chi phí nhận (RX) ---
        try {
            nodeService.updateNodeStatus(currentNodeId, packet);
        } catch (Exception e) {
            logger.error("[TCP_Service] Lỗi khi hạch toán RX/CPU cho {}: {}", currentNodeId, e.getMessage(), e);
        }

        // --- Node đích ---
        if (currentNodeId.equals(packet.getStationDest())) {
            if (packet.getPathHistory() != null) {
                packet.getPathHistory().add(currentNodeId);
            }
            logger.info("[TCP_Service] Packet {} đã đến trạm đích. Forward đến user...", packet.getPacketId());
            forwardPacketToUser(packet, currentNodeId);
            return;
        }

        // --- Node transit ---
        RouteInfo bestRoute = routingService.getBestRoute(currentNodeId, packet.getStationDest());
        if (bestRoute == null) {
            packet.setDropped(true);
            packet.setDropReason("NO_ROUTE_TO_HOST");
            logger.warn("[TCP_Service] Không tìm thấy đường đi cho packet {} từ {} đến {}.",
                    packet.getPacketId(), currentNodeId, packet.getStationDest());
            return;
        }

        String nextHopNodeId = bestRoute.getNextHopNodeId();
        packet.setNextHopNodeId(nextHopNodeId);

        Optional<NodeInfo> currentNodeOpt = nodeRepository.getNodeInfo(currentNodeId);
        Optional<NodeInfo> nextNodeOpt = nodeRepository.getNodeInfo(nextHopNodeId);

        if (currentNodeOpt.isPresent() && nextNodeOpt.isPresent()) {
            NodeInfo currentNode = currentNodeOpt.get();
            NodeInfo nextNode = nextNodeOpt.get();

            // --- Cập nhật packet qua helper ---
            PacketHelper.updatePacketForTransit(packet, currentNode, nextNode, bestRoute);

            // --- Nếu packet còn sống, gửi tiếp ---
            if (!packet.isDropped()) {
                packet.setCurrentHoldingNodeId(nextHopNodeId);
                sendPacket(packet, currentNodeId);
            } else {
                logger.warn("[TCP_Service] Packet {} bị drop: {}", packet.getPacketId(), packet.getDropReason());
            }
        } else {
            packet.setDropped(true);
            packet.setDropReason("NODE_INFO_NOT_FOUND");
            logger.error("[TCP_Service] Không tìm thấy thông tin node cho {} hoặc {}.", currentNodeId, nextHopNodeId);
        }
    }

    /**
     * (Hàm "Gửi đi" - Node-to-Node)
     * Thêm một packet (node-to-node) vào hàng đợi.
     * 
     * @param packet       Packet để gửi
     * @param senderNodeId Node HIỆN TẠI đang gửi packet này (để hạch toán TX)
     */
    @Override
    public void sendPacket(Packet packet, String senderNodeId) {
        String nextHopNodeId = packet.getNextHopNodeId();
        if (nextHopNodeId == null) {
            logger.warn("[TCP_Service] Packet {} từ {} bị drop: nextHopNodeId bị null (Lỗi định tuyến).",
                    packet.getPacketId(), senderNodeId);
            packet.setDropped(true);
            packet.setDropReason("ROUTING_BLACK_HOLE");
            return;
        }

        Optional<NodeInfo> nextHopOpt = nodeRepository.getNodeInfo(nextHopNodeId);
        if (nextHopOpt.isEmpty()) {
            logger.warn("[TCP_Service] Packet {} từ {} bị drop: Không tìm thấy node {} trong DB (Lỗi định tuyến).",
                    packet.getPacketId(), senderNodeId, nextHopNodeId);
            packet.setDropped(true);
            packet.setDropReason("ROUTING_NODE_NOT_FOUND");
            return;
        }

        NodeInfo nextHop = nextHopOpt.get();
        String host = nextHop.getHost();
        int port = nextHop.getPort();
        if (host == null || port <= 0) {
            logger.warn("[TCP_Service] Packet {} từ {} bị drop: Node {} có host/port không hợp lệ.",
                    packet.getPacketId(), senderNodeId, nextHopNodeId);
            return;
        }

        // Thêm vào hàng đợi, mang theo senderNodeId
        addToSendQueue(senderNodeId, packet, host, port, "NODE:" + nextHopNodeId);
    }

    /**
     * (Hàm "Gửi đi" - Node-to-User)
     * Hàm riêng: Gửi packet (node-to-user) vào hàng đợi.
     * 
     * @param packet       Packet để gửi
     * @param senderNodeId Node HIỆN TẠI đang gửi packet này (để hạch toán TX)
     */
    private void forwardPacketToUser(Packet packet, String senderNodeId) {
        String userId = packet.getDestinationUserId();
        if (userId == null || userId.isBlank()) {
            logger.warn("[TCP_Service] (forwardUser) Không thể chuyển tiếp {}: destinationUserId bị null.",
                    packet.getPacketId());
            return;
        }

        Optional<UserInfo> userOpt = userRepository.findByUserId(userId);
        if (userOpt.isEmpty()) {
            logger.error("[TCP_Service] (forwardUser) Không tìm thấy người dùng {}. Không thể giao packet {}.", userId,
                    packet.getPacketId());
            return;
        }

        UserInfo user = userOpt.get();
        String host = user.getCommunication().ipAddress();
        int port = user.getCommunication().port();
        if (host == null || port <= 0) {
            logger.error("[TCP_Service] (forwardUser) Người dùng {} có thông tin host/port không hợp lệ.", userId);
            return;
        }

        // Thêm vào hàng đợi, mang theo senderNodeId
        addToSendQueue(senderNodeId, packet, host, port, "USER:" + userId);
    }

    // ===================================================================
    // HÀM QUẢN LÝ HÀNG ĐỢI VÀ GỬI LẠI (Async Producer/Consumer)
    // ===================================================================

    /**
     * (Producer)
     * Thêm một job vào hàng đợi gửi (thread-safe).
     */
    private void addToSendQueue(String originalNodeId, Packet packet, String host, int port, String destinationDesc) {
        // Tạo job với 6 tham số
        RetryablePacket job = new RetryablePacket(originalNodeId, packet, host, port, destinationDesc, 1);
        try {
            sendQueue.put(job);
            logger.debug("[TCP_Service] ✈️ Đã thêm Packet {} (từ {}) vào hàng đợi gửi.",
                    packet.getPacketId(), originalNodeId);
        } catch (InterruptedException e) {
            logger.error("[TCP_Service] Bị gián đoạn khi thêm packet {} vào hàng đợi.", packet.getPacketId(), e);
            Thread.currentThread().interrupt();
        }
    }

    /**
     * (Consumer Setup)
     * Bắt đầu một thread nền (background thread) để xử lý `sendQueue`.
     */
    private void startSendScheduler() {
        this.retryScheduler.scheduleAtFixedRate(this::processSendQueue,
                RETRY_POLL_INTERVAL_MS,
                RETRY_POLL_INTERVAL_MS,
                TimeUnit.MILLISECONDS);
        logger.info("[TCP_Service] Dịch vụ lập lịch gửi (Send Scheduler) đã bắt đầu.");
    }

    /**
     * (Consumer Logic)
     * Hàm này được gọi định kỳ bởi `retryScheduler` để xử lý hàng đợi.
     * Sẽ gọi hạch toán `processSuccessfulSend` sau khi gửi thành công.
     */
    private void processSendQueue() {
        RetryablePacket job = sendQueue.poll(); // Lấy 1 item (không block)
        if (job == null)
            return; // Hàng đợi trống, nghỉ

        // Cố gắng gửi qua socket
        boolean success = attemptSendInternal(job);

        if (success) {
            // ====================================================
            // === GỬI THÀNH CÔNG (LOGIC QUAN TRỌNG NHẤT) ===
            // HẠCH TOÁN CHI PHÍ GỬI (TX)
            // ====================================================
            logger.debug("[TCP_Service] Gửi {} thành công. Hạch toán TX cho node {}.",
                    job.packet().getPacketId(), job.originalNodeId());
            // Gọi NodeService để hạch toán chi phí TX (pin, util,...)
            nodeService.processSuccessfulSend(job.originalNodeId(), job.packet());

        } else {
            // === GỬI THẤT BẠI (Lỗi I/O) ===
            if (job.attemptCount() < MAX_RETRIES) {
                // Vẫn còn lượt thử
                logger.warn("[TCP_Service] Gửi packet {} (lần {}) thất bại. Sẽ thử lại...",
                        job.packet().getPacketId(), job.attemptCount());

                // Tạo job mới với số lần thử tăng lên
                RetryablePacket nextAttempt = new RetryablePacket(
                        job.originalNodeId(), // Giữ nguyên
                        job.packet(),
                        job.host(),
                        job.port(),
                        job.destinationDesc(),
                        job.attemptCount() + 1 // Tăng số lần thử
                );
                sendQueue.add(nextAttempt); // Thêm lại vào cuối hàng đợi

            } else {
                // Đã hết lượt thử
                logger.error("[TCP_Service] HỦY BỎ packet {} đến {}: Đã vượt quá {} lần thử.",
                        job.packet().getPacketId(), job.destinationDesc(), MAX_RETRIES);

                if (job.destinationDesc().startsWith("NODE:")) {
                    job.packet().setDropped(true);
                    job.packet().setDropReason("TCP_SEND_FAILED_MAX_RETRIES");
                }
            }
        }
    }

    /**
     * (Consumer I/O)
     * Hàm I/O thực tế: Cố gắng serialize và gửi packet qua Socket.
     * 
     * @return true nếu thành công, false nếu thất bại (IOException).
     */
    private boolean attemptSendInternal(RetryablePacket job) {
        byte[] packetData;
        try {
            // 1. Chuyển Object thành JSON byte[]
            packetData = objectMapper.writeValueAsBytes(job.packet());
        } catch (IOException e) {
            logger.error("[TCP_Service] Lỗi serialize packet {}. Hủy bỏ (lỗi vĩnh viễn). Lỗi: {}",
                    job.packet().getPacketId(), e.getMessage(), e);
            return true; // Trả về true để *loại bỏ* khỏi hàng đợi (không thể thử lại)
        }

        logger.debug("[TCP_Service] Đang gửi (Lần {}/{}): Packet {} đến {} tại {}:{}...",
                job.attemptCount(), MAX_RETRIES,
                job.packet().getPacketId(), job.destinationDesc(), job.host(), job.port());

        // 2. Mở Socket và Gửi
        // (Sử dụng try-with-resources để đảm bảo socket và stream luôn được đóng)
        try (Socket socket = new Socket()) {
            // Thêm timeout để tránh bị treo nếu host không phản hồi
            socket.connect(
                    new InetSocketAddress(job.host(), job.port()),
                    SimulationConstants.TCP_CONNECT_TIMEOUT_MS // (ví dụ: 1000ms)
            );

            try (OutputStream out = socket.getOutputStream()) {
                out.write(packetData);
                out.flush();
                logger.info("[TCP_Service] Gửi thành công Packet {} đến {}.",
                        job.packet().getPacketId(), job.destinationDesc());
                return true; // Gửi thành công!
            }
        } catch (IOException e) {
            // Lỗi mạng (timeout, connection refused, v.v.)
            logger.warn("[TCP_Service] Lỗi I/O khi gửi packet {} (lần {}): {}",
                    job.packet().getPacketId(), job.attemptCount(), e.getMessage());
            return false; // Gửi thất bại, cần thử lại
        }
    }

    /**
     * Dừng lịch trình gửi lại (retry scheduler) khi đóng ứng dụng.
     * (Nên được gọi trong hook shutdown của ứng dụng)
     */
    public void stop() {
        logger.info("[TCP_Service] Đang dừng Send Scheduler...");
        this.retryScheduler.shutdown();
        try {
            if (!this.retryScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                this.retryScheduler.shutdownNow();
            }
            logger.info("[TCP_Service] Send Scheduler đã dừng.");
        } catch (InterruptedException e) {
            this.retryScheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}