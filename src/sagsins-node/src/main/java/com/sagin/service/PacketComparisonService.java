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
        if (packet == null || packet.isDropped()) {
            logger.warn("[PacketComparisonService] Cannot save dropped or null packet");
            return;
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
            comparison = createNewComparison(comparisonId, packet);
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
     */
    private String generateComparisonId(String sourceUserId, String destinationUserId, long timestamp) {
        // Format: source_dest_timestamp
        // Timestamp để phân biệt các lần gửi khác nhau
        return String.format("%s_%s_%d", sourceUserId, destinationUserId, timestamp / 1000); // Làm tròn đến giây
    }
    
    /**
     * Tạo PacketComparison mới với packet đầu tiên
     */
    private PacketComparison createNewComparison(String comparisonId, Packet packet) {
        PacketComparison comparison = new PacketComparison();
        comparison.setComparisonId(comparisonId);
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
        
        logger.info("═══════════════════════════════════════════════════════════════");
        logger.info("🏁 PACKET COMPARISON COMPLETE: {}", comparison.getComparisonId());
        logger.info("───────────────────────────────────────────────────────────────");
        logger.info("📍 Route: {} → {}", comparison.getSourceUserId(), comparison.getDestinationUserId());
        logger.info("");
        logger.info("📊 DIJKSTRA:");
        logger.info("   • Total Latency:  {} ms", String.format("%.2f", dijkstra.getAccumulatedDelayMs()));
        logger.info("   • Hops:           {}", dijkstra.getHopRecords() != null ? dijkstra.getHopRecords().size() : 0);
        logger.info("   • Path:           {}", dijkstra.getPathHistory());
        logger.info("");
        logger.info("🤖 REINFORCEMENT LEARNING:");
        logger.info("   • Total Latency:  {} ms", String.format("%.2f", rl.getAccumulatedDelayMs()));
        logger.info("   • Hops:           {}", rl.getHopRecords() != null ? rl.getHopRecords().size() : 0);
        logger.info("   • Path:           {}", rl.getPathHistory());
        logger.info("");
        
        // So sánh hiệu suất
        double latencyDiff = dijkstra.getAccumulatedDelayMs() - rl.getAccumulatedDelayMs();
        String winner = latencyDiff > 0 ? "RL" : "Dijkstra";
        double improvement = Math.abs(latencyDiff / dijkstra.getAccumulatedDelayMs() * 100);
        
        logger.info("🏆 Winner: {} ({}% faster)", winner, String.format("%.2f", improvement));
        logger.info("═══════════════════════════════════════════════════════════════");
    }
}
