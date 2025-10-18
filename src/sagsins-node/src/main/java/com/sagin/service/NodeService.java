package com.sagin.service;

import com.sagin.model.*;
import com.sagin.repository.INodeRepository;
import com.sagin.util.OrbitProfileFactory;
import com.sagin.util.SimulationConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

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
    private record ProcessingDelayProfile(double queuingMs, double processingMs) {}

    /**
     * DTO nội bộ cho độ trễ Truyền tải (TX).
     */
    private record TransmissionDelayProfile(double transmissionMs, double propagationMs) {}

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
            packet.setDropped(true); packet.setDropReason("NODE_NOT_FOUND_");
            return;
        }
        if (node.getCurrentPacketCount() >= node.getPacketBufferCapacity()) {
            packet.setDropped(true); packet.setDropReason("BUFFER_OVERFLOW_AT_" + nodeId);
            logger.warn("[NodeService] Node {} buffer đầy. Packet {} bị drop.", nodeId, packet.getPacketId());
            double newLoss = updateNodeMetricEMA(node.getPacketLossRate(), 1.0, SimulationConstants.BETA_LOSS);
            node.setPacketLossRate(Math.min(1.0, newLoss));
            dirtyNodeIds.add(nodeId);
            return;
        }
        if (!node.isHealthy()) {
            packet.setDropped(true); packet.setDropReason("NODE_UNHEALTHY_" + nodeId); 
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
                ? (double) node.getCurrentPacketCount() / node.getPacketBufferCapacity() : 0.0;
        double cpuLoad = Math.min(1.0, delays.processingMs() / SimulationConstants.SIMULATION_TIMESLOT_MS);
        double currentLoad = Math.max(bufferLoad, cpuLoad); // Lấy tải cao nhất
        
        double newUtilization = updateNodeMetricEMA(node.getResourceUtilization(), currentLoad, SimulationConstants.ALPHA_UTIL);

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
        double newUtilization = updateNodeMetricEMA(node.getResourceUtilization(), channelLoad, SimulationConstants.ALPHA_UTIL);

        // 5. Cập nhật Node (trong cache)
        node.setBatteryChargePercent(newBattery);
        node.setResourceUtilization(Math.min(SimulationConstants.MAX_UTILIZATION, newUtilization));
        node.setLastUpdated(Instant.now());
        
        dirtyNodeIds.add(nodeId);
        
        logger.info("[NodeService] Processed (TX) Packet {} on Node: {}", 
                            packet.getPacketId(), nodeId);
    }

    /**
     * Xử lý một "tick" mô phỏng (hiện không dùng nhiều).
     */
    public void processTick(Map<String, NodeInfo> nodeMap, List<Packet> packets) {
    if (packets == null || packets.isEmpty()) return;

    for (Packet packet : packets) {
        String nodeId = packet.getCurrentHoldingNodeId();

        // Lấy NodeInfo từ nodeMap (nếu có) hoặc từ repository
        NodeInfo node = null;
        if (nodeMap != null) {
            node = nodeMap.get(nodeId);
        }
        if (node == null) {
            node = nodeRepository.getNodeInfo(nodeId).orElse(null);
        }

        // Nếu vẫn null thì bỏ qua
        if (node == null) continue;

        // GỌI updateNodeStatus để xử lý logic
        updateNodeStatus(nodeId, packet);
    }
}


    @Override
    public void flushToDatabase() {
        if (dirtyNodeIds.isEmpty()) return;
        
        List<NodeInfo> dirtyNodes = dirtyNodeIds.stream()
                .map(nodeStateCache::get)
                .filter(java.util.Objects::nonNull)
                .toList();

        if (!dirtyNodes.isEmpty()) {
            nodeRepository.bulkUpdateNodes(dirtyNodes);
            logger.info("[NodeService] Đã flush {} nodes vào DB.", dirtyNodes.size());
        }

        dirtyNodeIds.clear();
    }


    // ===========================================
    // ⚙️ Các hàm hỗ trợ đã được CHIA NHỎ
    // ===========================================

    /**
     * CHỈ tính chi phí Nhận (RX) và Xử lý (CPU).
     */
    private double computeRxAndProcessBatteryDrain(NodeInfo node, Packet packet, double altitudeKm, WeatherCondition weather) {
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
        double weatherFactor = 1.0 + weather.getTypicalAttenuationDb() / SimulationConstants.WEATHER_DRAIN_IMPACT_FACTOR;

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
    private TransmissionDelayProfile computeTransmissionDelay(NodeInfo node, Packet packet, double altitudeKm, WeatherCondition weather) {
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
}