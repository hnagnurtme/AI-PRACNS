package com.sagin.service;

import com.sagin.model.Packet;
import com.sagin.model.PacketComparison;
import com.sagin.repository.IPacketComparisonRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.Optional;

/**
 * Service để lưu và so sánh packet Dijkstra vs RL
 */
public class PacketComparisonService {
    
    private static final Logger logger = LoggerFactory.getLogger(PacketComparisonService.class);
    private final IPacketComparisonRepository repository;
    
    public PacketComparisonService(IPacketComparisonRepository repository) {
        this.repository = repository;
    }
    
    /**
     * Lưu packet khi đến user đích thành công.
     * Tự động tìm hoặc tạo PacketComparison tương ứng.
     * 
     * @param packet Packet đã đến đích thành công
     */
    public void saveSuccessfulPacket(Packet packet) {
        saveSuccessfulPacket(packet, null);
    }
    
    /**
     * Lưu packet với batchId (bao gồm cả packet bị drop)
     * 
     * @param packet Packet (có thể bị drop hoặc thành công)
     * @param batchId ID của batch (optional)
     */
    public void saveSuccessfulPacket(Packet packet, String batchId) {
        if (packet == null) {
            logger.warn("[PacketComparisonService] Cannot save null packet");
            return;
        }
        
        // ✅ TÁI TẠO PACKET ID với prefix RL/Dijkstra để phân biệt
        String originalPacketId = packet.getPacketId();
        String prefixedPacketId = packet.isUseRL() 
            ? "RL-" + originalPacketId 
            : "Dijkstra-" + originalPacketId;
        packet.setPacketId(prefixedPacketId);
        
        logger.debug("[PacketComparisonService] Original PacketId: {} → Prefixed: {}", 
                originalPacketId, prefixedPacketId);
        
        // ✅ LƯU CẢ PACKET BỊ DROP để phân tích performance
        if (packet.isDropped()) {
            logger.info("[PacketComparisonService] Saving DROPPED packet {} | Reason: {}", 
                    packet.getPacketId(), packet.getDropReason());
        }
        
        // Tạo comparisonId từ source và destination
        String comparisonId = generateComparisonId(
            packet.getSourceUserId(), 
            packet.getDestinationUserId(),
            packet.getTimeSentFromSourceMs()
        );
        
        // Tìm PacketComparison hiện có hoặc tạo mới
        Optional<PacketComparison> existingOpt = repository.findByComparisonId(comparisonId);
        
        PacketComparison comparison;
        if (existingOpt.isPresent()) {
            // Đã có bản ghi → Cập nhật packet còn thiếu
            comparison = existingOpt.get();
            updateComparison(comparison, packet);
            logger.info("[PacketComparisonService] Updated existing comparison: {} | Status: {}", 
                    comparisonId, comparison.getStatus());
        } else {
            // Chưa có → Tạo mới
            comparison = createNewComparison(comparisonId, packet, batchId);
            logger.info("[PacketComparisonService] Created new comparison: {} | Algorithm: {}", 
                    comparisonId, packet.isUseRL() ? "RL" : "Dijkstra");
        }
        
        // Lưu vào database
        repository.save(comparison);
        
        // Log kết quả
        if ("complete".equals(comparison.getStatus())) {
            logComparisonSummary(comparison);
        }
    }
    
    /**
     * Tạo comparisonId duy nhất cho mỗi cặp packet
     * 
     * ✅ Dựa vào source_dest_timestamp để group các packets gửi cùng lúc
     * (bất kể packetId có giống nhau hay không)
     */
    public String generateComparisonId(String sourceUserId, String destinationUserId, long timestamp) {
        // Format: source_dest_timestamp
        // Timestamp làm tròn đến giây để group các packets gửi cùng khoảng thời gian
        return String.format("%s_%s_%d", sourceUserId, destinationUserId, timestamp / 1000);
    }
    
    /**
     * Tìm PacketComparison theo comparisonId
     */
    public Optional<PacketComparison> findByComparisonId(String comparisonId) {
        return repository.findByComparisonId(comparisonId);
    }
    
    /**
     * Tạo PacketComparison mới với packet đầu tiên
     */
    private PacketComparison createNewComparison(String comparisonId, Packet packet, String batchId) {
        PacketComparison comparison = new PacketComparison();
        comparison.setComparisonId(comparisonId);
        comparison.setBatchId(batchId);
        comparison.setSourceUserId(packet.getSourceUserId());
        comparison.setDestinationUserId(packet.getDestinationUserId());
        comparison.setCreatedAt(Instant.now());
        comparison.setStatus("partial"); // Chỉ có 1 packet
        
        // Gán packet vào slot tương ứng
        if (packet.isUseRL()) {
            comparison.setRlPacket(packet);
        } else {
            comparison.setDijkstraPacket(packet);
        }
        
        return comparison;
    }
    
    /**
     * Cập nhật PacketComparison với packet thứ 2
     */
    private void updateComparison(PacketComparison comparison, Packet packet) {
        if (packet.isUseRL()) {
            if (comparison.getRlPacket() == null) {
                comparison.setRlPacket(packet);
            } else {
                logger.warn("[PacketComparisonService] RL packet already exists for comparison: {}", 
                        comparison.getComparisonId());
            }
        } else {
            if (comparison.getDijkstraPacket() == null) {
                comparison.setDijkstraPacket(packet);
            } else {
                logger.warn("[PacketComparisonService] Dijkstra packet already exists for comparison: {}", 
                        comparison.getComparisonId());
            }
        }
        
        // Kiểm tra nếu đã có đủ 2 packet
        if (comparison.getDijkstraPacket() != null && comparison.getRlPacket() != null) {
            comparison.setStatus("complete");
        }
    }
    
    /**
     * Log tóm tắt so sánh khi có đủ 2 packet
     */
    private void logComparisonSummary(PacketComparison comparison) {
        Packet dijkstra = comparison.getDijkstraPacket();
        Packet rl = comparison.getRlPacket();
        
        if (dijkstra == null || rl == null) {
            return;
        }
        
        // Lấy latency từ AnalysisData (latency thực tế của route)
        double dijkstraLatency = dijkstra.getAnalysisData() != null 
            ? dijkstra.getAnalysisData().getTotalLatencyMs() 
            : dijkstra.getAccumulatedDelayMs();
        double rlLatency = rl.getAnalysisData() != null 
            ? rl.getAnalysisData().getTotalLatencyMs() 
            : rl.getAccumulatedDelayMs();
        
        // Lấy distance từ AnalysisData
        double dijkstraDistance = dijkstra.getAnalysisData() != null 
            ? dijkstra.getAnalysisData().getTotalDistanceKm() 
            : 0.0;
        double rlDistance = rl.getAnalysisData() != null 
            ? rl.getAnalysisData().getTotalDistanceKm() 
            : 0.0;
        
        logger.info("═══════════════════════════════════════════════════════════════");
        logger.info("🏁 PACKET COMPARISON COMPLETE: {}", comparison.getComparisonId());
        logger.info("───────────────────────────────────────────────────────────────");
        logger.info("📍 Route: {} → {}", comparison.getSourceUserId(), comparison.getDestinationUserId());
        logger.info("");
        logger.info("📊 DIJKSTRA:");
        logger.info("   • Route Latency:  {} ms (from AnalysisData)", String.format("%.2f", dijkstraLatency));
        logger.info("   • Route Distance: {} km", String.format("%.2f", dijkstraDistance));
        logger.info("   • Hops:           {}", dijkstra.getHopRecords() != null ? dijkstra.getHopRecords().size() : 0);
        logger.info("   • Path:           {}", dijkstra.getPathHistory());
        logger.info("   • Dropped:        {}", dijkstra.isDropped() ? "YES (" + dijkstra.getDropReason() + ")" : "NO");
        logger.info("");
        logger.info("🤖 REINFORCEMENT LEARNING:");
        logger.info("   • Route Latency:  {} ms (from AnalysisData)", String.format("%.2f", rlLatency));
        logger.info("   • Route Distance: {} km", String.format("%.2f", rlDistance));
        logger.info("   • Hops:           {}", rl.getHopRecords() != null ? rl.getHopRecords().size() : 0);
        logger.info("   • Path:           {}", rl.getPathHistory());
        logger.info("   • Dropped:        {}", rl.isDropped() ? "YES (" + rl.getDropReason() + ")" : "NO");
        logger.info("");
        
        // So sánh hiệu suất (chỉ nếu cả 2 đều không bị drop)
        if (!dijkstra.isDropped() && !rl.isDropped()) {
            double latencyDiff = dijkstraLatency - rlLatency;
            String winner = latencyDiff > 0 ? "RL" : "Dijkstra";
            double improvement = (dijkstraLatency != 0) 
                ? Math.abs(latencyDiff / dijkstraLatency * 100) 
                : 0;
            
            logger.info("🏆 Winner: {} ({}% faster)", winner, String.format("%.2f", improvement));
            
            // So sánh distance
            double distanceDiff = dijkstraDistance - rlDistance;
            String shorterPath = distanceDiff > 0 ? "RL" : "Dijkstra";
            logger.info("📏 Shorter Path: {} ({}% shorter)", shorterPath, 
                String.format("%.2f", dijkstraDistance != 0 ? Math.abs(distanceDiff / dijkstraDistance * 100) : 0));
        } else {
            logger.info("⚠️  Comparison: One or both packets were dropped");
            if (dijkstra.isDropped() && !rl.isDropped()) {
                logger.info("🏆 Winner: RL (Dijkstra packet was dropped)");
            } else if (!dijkstra.isDropped() && rl.isDropped()) {
                logger.info("🏆 Winner: Dijkstra (RL packet was dropped)");
            } else {
                logger.info("❌ Both packets were dropped");
            }
        }
        logger.info("═══════════════════════════════════════════════════════════════");
    }
}
