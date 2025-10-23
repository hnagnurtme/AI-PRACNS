package com.sagin.service;

import com.sagin.model.*;
import com.sagin.repository.INodeRepository;
import com.sagin.util.OrbitProfileFactory;
import com.sagin.util.SimulationConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Quản lý logic nghiệp vụ cho Node.
 * ĐÃ REFACTOR: Tách biệt hạch toán NHẬN (updateNodeStatus)
 * và hạch toán GỬI (processSuccessfulSend).
 */
public class NodeService implements INodeService {

    private static final Logger logger = LoggerFactory.getLogger(NodeService.class);
    private final INodeRepository nodeRepository;

    private final Map<String, NodeInfo> nodeStateCache = new ConcurrentHashMap<>();
    private final Set<String> dirtyNodeIds = ConcurrentHashMap.newKeySet();

    /**
     * DTO nội bộ cho độ trễ Xử lý (RX/CPU).
     */
    private record ProcessingDelayProfile(double queuingMs, double processingMs) {
    }

    /**
     * DTO nội bộ cho độ trễ Truyền tải (TX).
     */
    private record TransmissionDelayProfile(double transmissionMs, double propagationMs) {
    }

    public NodeService(INodeRepository nodeRepository) {
        this.nodeRepository = nodeRepository;
    }

    /**
     * Hạch toán chi phí NHẬN (RX) và XỬ LÝ (CPU) khi packet đến.
     */
    @Override
    public void updateNodeStatus(String nodeId, Packet packet) {

        // === BƯỚC 1, 2, 3 (Lấy Node, Check Buffer, Check Healthy) ---
        NodeInfo node = nodeStateCache.computeIfAbsent(nodeId, id -> {
            logger.info("[NodeService] Cache miss. Đang tải Node {} từ DB...", id);
            return nodeRepository.getNodeInfo(id).orElse(null);
        });
        if (node == null) {
            logger.warn("[NodeService] Node {} không tìm thấy. Packet {} bị drop.", nodeId, packet.getPacketId());
            packet.setDropped(true);
            packet.setDropReason("NODE_NOT_FOUND_");
            return;
        }
        if (node.getCurrentPacketCount() >= node.getPacketBufferCapacity()) {
            packet.setDropped(true);
            packet.setDropReason("BUFFER_OVERFLOW_AT_" + nodeId);
            logger.warn("[NodeService] Node {} buffer đầy. Packet {} bị drop.", nodeId, packet.getPacketId());
            double newLoss = updateNodeMetricEMA(node.getPacketLossRate(), 1.0, SimulationConstants.BETA_LOSS);
            node.setPacketLossRate(Math.min(1.0, newLoss));
            dirtyNodeIds.add(nodeId);
            return;
        }
        if (!node.isHealthy()) {
            packet.setDropped(true);
            packet.setDropReason("NODE_UNHEALTHY_" + nodeId);
            logger.warn("[NodeService] Node {} không khỏe. Packet {} bị drop.", nodeId, packet.getPacketId());
            return;
        }
        node.setCurrentPacketCount(node.getCurrentPacketCount() + 1); // Chiếm buffer
        // --- Hết ---

        // === BƯỚC 4: LẤY YẾU TỐ MÔI TRƯỜNG ===
        NodeType nodeType = node.getNodeType();
        WeatherCondition weather = node.getWeather() != null ? node.getWeather() : WeatherCondition.CLEAR;
        double altitudeKm = node.getOrbit() != null
                ? OrbitProfileFactory.computeAltitudeKm(node.getOrbit())
                : nodeType.getDefaultAltitudeKm();

        // === BƯỚC 5: HẠCH TOÁN CHI PHÍ NHẬN (RX) VÀ XỬ LÝ (CPU) ===

        // 5a. Hạch toán Pin (Chỉ RX và Process)
        double drainPercent = computeRxAndProcessBatteryDrain(node, packet, altitudeKm, weather);
        double newBattery = Math.max(SimulationConstants.MIN_BATTERY,
                node.getBatteryChargePercent() - drainPercent);

        // 5b. Hạch toán Độ trễ (Chỉ Queuing và Process)
        ProcessingDelayProfile delays = computeProcessingDelay(node, packet);

        // 5c. Cập nhật độ trễ TÍCH LŨY (chỉ phần xử lý)
        packet.setAccumulatedDelayMs(packet.getAccumulatedDelayMs() + delays.queuingMs() + delays.processingMs());

        // 5d. Kiểm tra QoS (Logic giữ nguyên)
        if (packet.getAccumulatedDelayMs() > packet.getMaxAcceptableLatencyMs()) {
            packet.setDropped(true);
            packet.setDropReason("QOS_LATENCY_EXCEEDED");
            logger.warn("[NodeService] Packet {} bị drop tại {}: Vượt quá latency ({} > {}).",
                    packet.getPacketId(), nodeId,
                    String.format("%.2f", packet.getAccumulatedDelayMs()),
                    String.format("%.2f", packet.getMaxAcceptableLatencyMs()));

            node.setBatteryChargePercent(newBattery); // Vẫn cập nhật pin
            dirtyNodeIds.add(nodeId); // Đánh dấu là "dirty"
            return;
        }

        // 5e. Cập nhật Utilization (CHỈ DỰA TRÊN CPU VÀ BUFFER)
        double bufferLoad = (node.getPacketBufferCapacity() > 0)
                ? (double) node.getCurrentPacketCount() / node.getPacketBufferCapacity()
                : 0.0;
        double cpuLoad = Math.min(1.0, delays.processingMs() / SimulationConstants.SIMULATION_TIMESLOT_MS);
        double currentLoad = Math.max(bufferLoad, cpuLoad); // Lấy tải cao nhất

        double newUtilization = updateNodeMetricEMA(node.getResourceUtilization(), currentLoad,
                SimulationConstants.ALPHA_UTIL);

        double weatherLoss = weather.getTypicalAttenuationDb() / SimulationConstants.WEATHER_LOSS_FACTOR;
        double newLossRate = updateNodeMetricEMA(node.getPacketLossRate(), weatherLoss, SimulationConstants.BETA_LOSS);

        // === BƯỚC 6: GIẢI PHÓNG BUFFER ===
        node.setCurrentPacketCount(node.getCurrentPacketCount() - 1);

        // === BƯỚC 7: CẬP NHẬT TRẠNG THÁI NODE (TRONG CACHE) ===
        node.setBatteryChargePercent(newBattery);
        node.setNodeProcessingDelayMs(delays.processingMs());
        node.setPacketLossRate(Math.min(1.0, Math.max(0.0, newLossRate)));
        node.setResourceUtilization(Math.min(SimulationConstants.MAX_UTILIZATION, newUtilization));
        node.setLastUpdated(Instant.now());
        dirtyNodeIds.add(nodeId);

        flushToDatabase();
        logger.info("[NodeService] Processed (RX/CPU) Packet {} on Node: {}",
                packet.getPacketId(), nodeId);
    }

    /**
     * **HÀM MỚI**: Hạch toán chi phí GỬI (TX) SAU KHI gửi thành công.
     */
    @Override
    public void processSuccessfulSend(String nodeId, Packet packet) {
        NodeInfo node = nodeStateCache.get(nodeId);
        if (node == null) {
            logger.warn("[NodeService] (processSend) Không tìm thấy node {} trong cache.", nodeId);
            return;
        }

        // --- Lấy Yếu tố Môi trường ---
        NodeType nodeType = node.getNodeType();
        WeatherCondition weather = node.getWeather() != null ? node.getWeather() : WeatherCondition.CLEAR;
        double altitudeKm = node.getOrbit() != null
                ? OrbitProfileFactory.computeAltitudeKm(node.getOrbit())
                : nodeType.getDefaultAltitudeKm();

        // 1. Hạch toán Pin (Chỉ TX)
        double drainPercent = computeTxBatteryDrain(node, packet, altitudeKm, weather);
        double newBattery = Math.max(SimulationConstants.MIN_BATTERY,
                node.getBatteryChargePercent() - drainPercent);

        // 2. Hạch toán Độ trễ (Chỉ Transmission và Propagation)
        TransmissionDelayProfile delays = computeTransmissionDelay(node, packet, altitudeKm, weather);

        // 3. Cập nhật độ trễ TÍCH LŨY (phần truyền tải)
        packet.setAccumulatedDelayMs(packet.getAccumulatedDelayMs() + delays.transmissionMs() + delays.propagationMs());

        // 4. Cập nhật Utilization (DỰA TRÊN TẢI KÊNH TRUYỀN)
        double channelLoad = Math.min(1.0, delays.transmissionMs() / SimulationConstants.SIMULATION_TIMESLOT_MS);
        double newUtilization = updateNodeMetricEMA(node.getResourceUtilization(), channelLoad,
                SimulationConstants.ALPHA_UTIL);

        // 5. Cập nhật Node (trong cache)
        node.setBatteryChargePercent(newBattery);
        node.setResourceUtilization(Math.min(SimulationConstants.MAX_UTILIZATION, newUtilization));
        node.setLastUpdated(Instant.now());

        dirtyNodeIds.add(nodeId);

        logger.info("[NodeService] Processed (TX) Packet {} on Node: {}",
                packet.getPacketId(), nodeId);
    }

    /**
     * Xử lý một "tick" mô phỏng
     */
    @Override
    public void processTick(Map<String, NodeInfo> nodeMap, List<Packet> packets) {
        if (packets == null || packets.isEmpty()) {
            return;
        }

        if (nodeMap != null && !nodeMap.isEmpty()) {
            logger.warn("[NodeService] processTick bỏ qua tham số nodeMap vì nó dùng cache nội bộ.");
        }

        logger.debug("[NodeService] Bắt đầu xử lý tick với {} packets...", packets.size());
        for (Packet packet : packets) {
            if (packet == null || packet.getCurrentHoldingNodeId() == null) {
                logger.warn("[NodeService] Bỏ qua packet bị null hoặc không có node giữ.");
                continue;
            }

            updateNodeStatus(packet.getCurrentHoldingNodeId(), packet);
        }

        flushToDatabase();
        logger.debug("[NodeService] Kết thúc xử lý tick.");
    }

    @Override
    public void flushToDatabase() {
        if (dirtyNodeIds.isEmpty()) {
            logger.info("[NodeService] Không có thay đổi nào để lưu vào CSDL.");
            return;
        }

        // Tạo bản sao của các ID để đảm bảo an toàn luồng
        Set<String> idsToFlush = Set.copyOf(dirtyNodeIds);
        List<NodeInfo> nodesToUpdate = new ArrayList<>(idsToFlush.size());

        for (String nodeId : idsToFlush) {
            NodeInfo node = nodeStateCache.get(nodeId);
            if (node != null) {
                nodesToUpdate.add(node);
            } else {
                logger.warn("[NodeService] Node {} dirty nhưng không có trong cache. Xóa khỏi dirty set.", nodeId);
                dirtyNodeIds.remove(nodeId); // Xóa ID không hợp lệ
            }
        }

        if (nodesToUpdate.isEmpty()) {
            logger.info("[NodeService] Không có đối tượng NodeInfo hợp lệ nào để lưu.");
            return;
        }

        logger.info("[NodeService] Bắt đầu thực hiện bulk update cho {} node...", nodesToUpdate.size());

        try {
            // Gọi phương thức bulk update
            nodeRepository.bulkUpdateNodes(nodesToUpdate);

            // **QUAN TRỌNG:** Chỉ xóa khỏi dirty set SAU KHI thành công
            dirtyNodeIds.removeAll(idsToFlush);

            logger.info("[NodeService] Bulk update CSDL hoàn tất. Còn lại {} node dirty.",
                    dirtyNodeIds.size());

        } catch (Exception e) {
            logger.error("[NodeService] Lỗi nghiêm trọng khi thực hiện bulk update: {}", e.getMessage(), e);
            // Nếu lỗi, KHÔNG xóa dirtyNodeIds. Chúng sẽ được thử lại ở lần flush sau.
            logger.warn("[NodeService] Các thay đổi sẽ được giữ lại để thử lại ở lần flush sau.");
        }
    }

    // ===========================================
    // ⚙️ Các hàm hỗ trợ đã được CHIA NHỎ
    // ===========================================

    /**
     * CHỈ tính chi phí Nhận (RX) và Xử lý (CPU).
     */
    private double computeRxAndProcessBatteryDrain(NodeInfo node, Packet packet, double altitudeKm,
            WeatherCondition weather) {
        double drainRx = SimulationConstants.RX_COST_PER_PACKET;
        double drainProcess = packet.isUseRL()
                ? SimulationConstants.RL_CPU_DRAIN_COST
                : SimulationConstants.BASE_CPU_DRAIN_COST;

        double nodeFactor = node.getNodeType().getBatteryDrainFactor();
        double altitudeFactor = (1.0 + altitudeKm / SimulationConstants.ALTITUDE_DRAIN_NORMALIZATION_KM);

        // Không có drainTx, không có weatherFactor cho TX
        return (drainRx + drainProcess) * nodeFactor * altitudeFactor;
    }

    /**
     * CHỈ tính chi phí Gửi (TX).
     */
    private double computeTxBatteryDrain(NodeInfo node, Packet packet, double altitudeKm, WeatherCondition weather) {
        double drainTx = packet.getPayloadSizeByte() * SimulationConstants.TX_COST_PER_BYTE;

        double nodeFactor = node.getNodeType().getBatteryDrainFactor();
        double altitudeFactor = (1.0 + altitudeKm / SimulationConstants.ALTITUDE_DRAIN_NORMALIZATION_KM);
        double weatherFactor = 1.0
                + weather.getTypicalAttenuationDb() / SimulationConstants.WEATHER_DRAIN_IMPACT_FACTOR;

        // Chỉ có drainTx và các hệ số liên quan đến truyền tải
        return (drainTx) * nodeFactor * altitudeFactor * weatherFactor;
    }

    /**
     * CHỈ tính độ trễ Hàng đợi (Queuing) và Xử lý (Processing).
     */
    private ProcessingDelayProfile computeProcessingDelay(NodeInfo node, Packet packet) {
        double bufferLoadRatio = (node.getPacketBufferCapacity() > 0)
                ? (double) node.getCurrentPacketCount() / node.getPacketBufferCapacity()
                : 0.0;
        double queuingDelayMs = bufferLoadRatio * SimulationConstants.MAX_QUEUING_DELAY_MS;

        double processingDelayMs = packet.isUseRL()
                ? SimulationConstants.RL_PROCESSING_DELAY_MS
                : SimulationConstants.DATA_PROCESSING_DELAY_MS;

        return new ProcessingDelayProfile(queuingDelayMs, processingDelayMs);
    }

    /**
     * CHỈ tính độ trễ Truyền (Transmission) và Truyền sóng (Propagation).
     */
    private TransmissionDelayProfile computeTransmissionDelay(NodeInfo node, Packet packet, double altitudeKm,
            WeatherCondition weather) {
        double bandwidthMHz = node.getCommunication().bandwidthMHz();
        double dataRateMbps = bandwidthMHz;
        double bandwidthBps = dataRateMbps * SimulationConstants.MBPS_TO_BPS_CONVERSION;
        double bandwidthBpms = bandwidthBps / 1000.0;
        double transmissionDelayMs = (bandwidthBpms > 0)
                ? packet.getPayloadSizeByte() / bandwidthBpms
                : Double.MAX_VALUE;

        double propagationDelayMs = altitudeKm / SimulationConstants.PROPAGATION_DIVISOR_KM_MS;

        double weatherImpactFactor = 1.0 + weather.getTypicalAttenuationDb() / SimulationConstants.WEATHER_DB_TO_FACTOR;

        // Chỉ áp dụng Weather Impact vào Transmission
        return new TransmissionDelayProfile(transmissionDelayMs * weatherImpactFactor, propagationDelayMs);
    }

    private double updateNodeMetricEMA(double previousValue, double currentValue, double alpha) {
        return (1 - alpha) * previousValue + alpha * currentValue;
    }

    @Override
    public List<NodeInfo> getVisibleNodes(NodeInfo node, List<NodeInfo> allNodes) {
        return allNodes.stream()
                .filter(n -> !n.getNodeId().equals(node.getNodeId()))
                .filter(NodeInfo::isHealthy)
                .filter(n -> canSeeEachOther(node, n))
                .collect(Collectors.toList());
    }

    /**
     * Kiểm tra xem hai node có thể "nhìn thấy" nhau không.
     * Áp dụng mô hình vật lý thực tế: Line-of-Sight, góc nâng, phạm vi truyền.
     */
    private boolean canSeeEachOther(NodeInfo node1, NodeInfo node2) {
        // Bỏ qua nếu node chết
        if (!node1.isHealthy() || !node2.isHealthy())
            return false;

        Position p1 = node1.getPosition();
        Position p2 = node2.getPosition();
        double distance = distance3DKm(p1, p2);

        double R = 6371.0; // Bán kính Trái Đất (km)
        double alt1 = getEffectiveAltitude(node1);
        double alt2 = getEffectiveAltitude(node2);

        // 🚫 Kiểm tra che khuất Trái Đất
        if (!hasLineOfSight(p1, p2, R)) {
            return false;
        }

        // 🧭 Kiểm tra góc nâng nếu có Ground Station
        if (!isAboveHorizon(node1, node2, R)) {
            return false;
        }

        // 📡 Phạm vi truyền hợp lý giữa 2 loại node
        double maxRange = getVisibilityRange(node1.getNodeType(), node2.getNodeType());

        return distance <= maxRange;
    }

    /**
     * Tính altitude hiệu quả của node
     */
    private double getEffectiveAltitude(NodeInfo node) {
        if (node.getOrbit() != null) {
            return OrbitProfileFactory.computeAltitudeKm(node.getOrbit());
        }
        return node.getPosition().getAltitude();
    }

    /**
     * Tính khoảng cách 3D giữa hai vị trí địa lý (lat, lon, alt)
     */
    private double distance3DKm(Position p1, Position p2) {
        double lat1 = Math.toRadians(p1.getLatitude());
        double lon1 = Math.toRadians(p1.getLongitude());
        double lat2 = Math.toRadians(p2.getLatitude());
        double lon2 = Math.toRadians(p2.getLongitude());

        double r1 = 6371.0 + p1.getAltitude();
        double r2 = 6371.0 + p2.getAltitude();

        double x1 = r1 * Math.cos(lat1) * Math.cos(lon1);
        double y1 = r1 * Math.cos(lat1) * Math.sin(lon1);
        double z1 = r1 * Math.sin(lat1);

        double x2 = r2 * Math.cos(lat2) * Math.cos(lon2);
        double y2 = r2 * Math.cos(lat2) * Math.sin(lon2);
        double z2 = r2 * Math.sin(lat2);

        double dx = x2 - x1;
        double dy = y2 - y1;
        double dz = z2 - z1;

        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /**
     * Kiểm tra xem hai node có bị Trái Đất che khuất không (Line of Sight)
     */
    private boolean hasLineOfSight(Position p1, Position p2, double R) {
        double r1 = R + p1.getAltitude();
        double r2 = R + p2.getAltitude();
        double d = distance3DKm(p1, p2);

        // Tính góc giữa 2 vector tâm-Trái-Đất -> node
        double cosTheta = (r1 * r1 + r2 * r2 - d * d) / (2 * r1 * r2);

        // Nếu cosTheta < (R^2)/(r1*r2), nghĩa là đường thẳng giữa hai node xuyên qua
        // Trái Đất
        return cosTheta > (R * R) / (r1 * r2);
    }

    /**
     * Kiểm tra góc nâng (Elevation Angle) khi 1 node là Ground Station
     */
    private boolean isAboveHorizon(NodeInfo n1, NodeInfo n2, double R) {
        if (n1.getNodeType() != NodeType.GROUND_STATION &&
                n2.getNodeType() != NodeType.GROUND_STATION)
            return true;

        NodeInfo ground = n1.getNodeType() == NodeType.GROUND_STATION ? n1 : n2;
        NodeInfo sat = ground == n1 ? n2 : n1;

        double d = distance3DKm(ground.getPosition(), sat.getPosition());
        double altSat = getEffectiveAltitude(sat);

        // Góc nâng xấp xỉ: arctan(altitude / surface distance)
        double elevationDeg = Math.toDegrees(Math.atan2(altSat, d));

        return elevationDeg >= 5.0; // ≥5° để xem là khả thi
    }

    /**
     * Trả về phạm vi truyền tối đa tùy theo loại node
     * (được tinh chỉnh theo mô hình SAGIN thực tế)
     */
    private double getVisibilityRange(NodeType t1, NodeType t2) {
        if (t1 == NodeType.GROUND_STATION && t2 == NodeType.GEO_SATELLITE ||
                t2 == NodeType.GROUND_STATION && t1 == NodeType.GEO_SATELLITE)
            return 42000; // Ground ↔ GEO

        if (t1 == NodeType.GROUND_STATION || t2 == NodeType.GROUND_STATION)
            return 2500; // Ground ↔ LEO/MEO

        if (t1 == NodeType.LEO_SATELLITE && t2 == NodeType.MEO_SATELLITE ||
                t1 == NodeType.MEO_SATELLITE && t2 == NodeType.LEO_SATELLITE)
            return 10000;

        if (t1 == NodeType.LEO_SATELLITE && t2 == NodeType.GEO_SATELLITE ||
                t1 == NodeType.GEO_SATELLITE && t2 == NodeType.LEO_SATELLITE)
            return 40000;

        if (t1 == NodeType.MEO_SATELLITE && t2 == NodeType.GEO_SATELLITE ||
                t1 == NodeType.GEO_SATELLITE && t2 == NodeType.MEO_SATELLITE)
            return 50000;

        if (t1 == NodeType.GEO_SATELLITE && t2 == NodeType.GEO_SATELLITE)
            return 80000;

        // Mặc định (ví dụ LEO-LEO hoặc MEO-MEO)
        return 5000;
    }

}