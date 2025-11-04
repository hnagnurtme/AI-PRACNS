package com.sagin.service;

import com.sagin.model.Packet;
import com.sagin.model.PacketComparison;
import com.sagin.model.PacketComparisonBatch;
import com.sagin.repository.IPacketComparisonBatchRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Optional;
import java.util.UUID;

/**
 * Service quản lý PacketComparisonBatch
 * Xử lý việc tạo, cập nhật và tính toán statistics cho batch
 */
public class PacketComparisonBatchService {
    
    private static final Logger logger = LoggerFactory.getLogger(PacketComparisonBatchService.class);
    private final IPacketComparisonBatchRepository batchRepository;
    private final PacketComparisonService comparisonService;
    
    public PacketComparisonBatchService(
            IPacketComparisonBatchRepository batchRepository,
            PacketComparisonService comparisonService) {
        this.batchRepository = batchRepository;
        this.comparisonService = comparisonService;
    }
    
    /**
     * Tạo batch mới
     */
    public PacketComparisonBatch createBatch(
            String sourceUserId, 
            String destinationUserId, 
            int totalPairs,
            String testScenario,
            String description) {
        
        String batchId = generateBatchId();
        
        PacketComparisonBatch batch = new PacketComparisonBatch();
        batch.setBatchId(batchId);
        batch.setTotalPairPackets(totalPairs);
        batch.setCompletedPairs(0);
        batch.setPackets(new ArrayList<>());
        batch.setCreatedAt(Instant.now());
        batch.setStatus("pending");
        
        // Metadata
        PacketComparisonBatch.BatchMetadata metadata = new PacketComparisonBatch.BatchMetadata();
        metadata.setSourceUserId(sourceUserId);
        metadata.setDestinationUserId(destinationUserId);
        metadata.setTestScenario(testScenario);
        metadata.setDescription(description);
        batch.setMetadata(metadata);
        
        batchRepository.save(batch);
        logger.info("[PacketComparisonBatchService] Created batch {} with {} pairs", batchId, totalPairs);
        
        return batch;
    }
    
    /**
     * Thêm packet vào batch (qua comparisonService)
     * Tự động cập nhật batch khi comparison complete
     */
    public void addPacketToBatch(String batchId, Packet packet) {
        try {
            Optional<PacketComparisonBatch> batchOpt = batchRepository.findByBatchId(batchId);
            if (batchOpt.isEmpty()) {
                logger.warn("[PacketComparisonBatchService] Batch {} not found, creating implicit batch", batchId);
                // Tạo implicit batch nếu chưa có
                createImplicitBatch(batchId, packet);
            }
            
            // Lưu packet vào comparison (sẽ tạo hoặc update comparison)
            comparisonService.saveSuccessfulPacket(packet);
            
            // Lấy comparison vừa được update
            String comparisonId = comparisonService.generateComparisonId(
                packet.getSourceUserId(), 
                packet.getDestinationUserId(), 
                packet.getTimeSentFromSourceMs()
            );
            
            Optional<PacketComparison> comparisonOpt = comparisonService.findByComparisonId(comparisonId);
            if (comparisonOpt.isPresent()) {
                PacketComparison comparison = comparisonOpt.get();
                
                // Cập nhật batch với comparison này
                updateBatchWithComparison(batchId, comparison);
            }
            
        } catch (Exception e) {
            logger.error("[PacketComparisonBatchService] Error adding packet to batch {}: {}", 
                batchId, e.getMessage(), e);
        }
    }
    
    /**
     * Cập nhật batch khi có comparison mới hoặc updated
     */
    private void updateBatchWithComparison(String batchId, PacketComparison comparison) {
        Optional<PacketComparisonBatch> batchOpt = batchRepository.findByBatchId(batchId);
        if (batchOpt.isEmpty()) {
            return;
        }
        
        PacketComparisonBatch batch = batchOpt.get();
        
        // Kiểm tra xem comparison đã có trong batch chưa
        boolean exists = batch.getPackets().stream()
            .anyMatch(p -> p.getComparisonId().equals(comparison.getComparisonId()));
        
        if (!exists) {
            // Thêm mới
            batch.getPackets().add(comparison);
        } else {
            // Cập nhật comparison đã có
            batch.getPackets().removeIf(p -> p.getComparisonId().equals(comparison.getComparisonId()));
            batch.getPackets().add(comparison);
        }
        
        // Đếm số completed pairs
        long completedCount = batch.getPackets().stream()
            .filter(p -> "complete".equals(p.getStatus()))
            .count();
        batch.setCompletedPairs((int) completedCount);
        
        // Cập nhật status
        if (batch.getCompletedPairs() == 0) {
            batch.setStatus("pending");
        } else if (batch.getCompletedPairs() < batch.getTotalPairPackets()) {
            batch.setStatus("in_progress");
        } else {
            batch.setStatus("completed");
            batch.setCompletedAt(Instant.now());
            
            // Tính statistics khi hoàn thành
            calculateBatchStatistics(batch);
        }
        
        batchRepository.save(batch);
        logger.debug("[PacketComparisonBatchService] Updated batch {} | Completed: {}/{}", 
            batchId, batch.getCompletedPairs(), batch.getTotalPairPackets());
        
        // Log khi batch complete
        if ("completed".equals(batch.getStatus())) {
            logBatchCompletion(batch);
        }
    }
    
    /**
     * Tính toán statistics cho toàn bộ batch
     */
    private void calculateBatchStatistics(PacketComparisonBatch batch) {
        PacketComparisonBatch.BatchStatistics stats = new PacketComparisonBatch.BatchStatistics();
        
        // Dijkstra stats
        PacketComparisonBatch.AlgorithmStats dijkstraStats = calculateAlgorithmStats(batch, true);
        stats.setDijkstra(dijkstraStats);
        
        // RL stats
        PacketComparisonBatch.AlgorithmStats rlStats = calculateAlgorithmStats(batch, false);
        stats.setRl(rlStats);
        
        // Comparison stats
        PacketComparisonBatch.ComparisonStats comparisonStats = calculateComparisonStats(batch);
        stats.setComparison(comparisonStats);
        
        batch.setBatchStatistics(stats);
    }
    
    private PacketComparisonBatch.AlgorithmStats calculateAlgorithmStats(
            PacketComparisonBatch batch, boolean isDijkstra) {
        
        PacketComparisonBatch.AlgorithmStats stats = new PacketComparisonBatch.AlgorithmStats();
        
        var packets = batch.getPackets().stream()
            .filter(p -> "complete".equals(p.getStatus()))
            .map(p -> isDijkstra ? p.getDijkstraPacket() : p.getRlPacket())
            .filter(p -> p != null && p.getAnalysisData() != null)
            .toList();
        
        if (packets.isEmpty()) {
            return stats;
        }
        
        double totalLatency = packets.stream()
            .mapToDouble(p -> p.getAnalysisData().getTotalLatencyMs())
            .sum();
        double totalDistance = packets.stream()
            .mapToDouble(p -> p.getAnalysisData().getTotalDistanceKm())
            .sum();
        double totalHops = packets.stream()
            .mapToInt(p -> p.getHopRecords() != null ? p.getHopRecords().size() : 0)
            .sum();
        
        stats.setAvgTotalLatency(totalLatency / packets.size());
        stats.setAvgTotalDistance(totalDistance / packets.size());
        stats.setAvgHopCount(totalHops / packets.size());
        stats.setSuccessCount(packets.size());
        stats.setFailureCount(batch.getTotalPairPackets() - packets.size());
        
        return stats;
    }
    
    private PacketComparisonBatch.ComparisonStats calculateComparisonStats(PacketComparisonBatch batch) {
        PacketComparisonBatch.ComparisonStats stats = new PacketComparisonBatch.ComparisonStats();
        
        var completePairs = batch.getPackets().stream()
            .filter(p -> "complete".equals(p.getStatus()))
            .filter(p -> p.getDijkstraPacket() != null && p.getRlPacket() != null)
            .filter(p -> p.getDijkstraPacket().getAnalysisData() != null 
                      && p.getRlPacket().getAnalysisData() != null)
            .toList();
        
        if (completePairs.isEmpty()) {
            return stats;
        }
        
        int dijkstraFaster = 0;
        int rlFaster = 0;
        int dijkstraShorter = 0;
        int rlShorter = 0;
        double totalLatencyDiff = 0;
        double totalDistanceDiff = 0;
        
        for (PacketComparison pair : completePairs) {
            double dijkstraLatency = pair.getDijkstraPacket().getAnalysisData().getTotalLatencyMs();
            double rlLatency = pair.getRlPacket().getAnalysisData().getTotalLatencyMs();
            double dijkstraDistance = pair.getDijkstraPacket().getAnalysisData().getTotalDistanceKm();
            double rlDistance = pair.getRlPacket().getAnalysisData().getTotalDistanceKm();
            
            if (dijkstraLatency < rlLatency) dijkstraFaster++;
            else if (rlLatency < dijkstraLatency) rlFaster++;
            
            if (dijkstraDistance < rlDistance) dijkstraShorter++;
            else if (rlDistance < dijkstraDistance) rlShorter++;
            
            totalLatencyDiff += Math.abs(dijkstraLatency - rlLatency);
            totalDistanceDiff += Math.abs(dijkstraDistance - rlDistance);
        }
        
        stats.setDijkstraFasterCount(dijkstraFaster);
        stats.setRlFasterCount(rlFaster);
        stats.setDijkstraShorterCount(dijkstraShorter);
        stats.setRlShorterCount(rlShorter);
        stats.setAvgLatencyDifference(totalLatencyDiff / completePairs.size());
        stats.setAvgDistanceDifference(totalDistanceDiff / completePairs.size());
        
        return stats;
    }
    
    /**
     * Tạo implicit batch khi không có batch được tạo trước
     */
    private void createImplicitBatch(String batchId, Packet packet) {
        PacketComparisonBatch batch = new PacketComparisonBatch();
        batch.setBatchId(batchId);
        batch.setTotalPairPackets(1); // Sẽ tăng dần khi có thêm packets
        batch.setCompletedPairs(0);
        batch.setPackets(new ArrayList<>());
        batch.setCreatedAt(Instant.now());
        batch.setStatus("pending");
        
        PacketComparisonBatch.BatchMetadata metadata = new PacketComparisonBatch.BatchMetadata();
        metadata.setSourceUserId(packet.getSourceUserId());
        metadata.setDestinationUserId(packet.getDestinationUserId());
        metadata.setTestScenario("implicit");
        metadata.setDescription("Auto-created batch");
        batch.setMetadata(metadata);
        
        batchRepository.save(batch);
    }
    
    /**
     * Log khi batch hoàn thành
     */
    private void logBatchCompletion(PacketComparisonBatch batch) {
        logger.info("=".repeat(80));
        logger.info("[PacketComparisonBatchService] 🎉 BATCH COMPLETED: {}", batch.getBatchId());
        logger.info("   Total Pairs: {} | Completed: {}", 
            batch.getTotalPairPackets(), batch.getCompletedPairs());
        
        if (batch.getBatchStatistics() != null) {
            var stats = batch.getBatchStatistics();
            
            logger.info("   📊 DIJKSTRA Stats:");
            logger.info("      Avg Latency: {:.2f} ms | Avg Distance: {:.2f} km | Avg Hops: {:.2f}",
                stats.getDijkstra().getAvgTotalLatency(),
                stats.getDijkstra().getAvgTotalDistance(),
                stats.getDijkstra().getAvgHopCount());
            
            logger.info("   📊 RL Stats:");
            logger.info("      Avg Latency: {:.2f} ms | Avg Distance: {:.2f} km | Avg Hops: {:.2f}",
                stats.getRl().getAvgTotalLatency(),
                stats.getRl().getAvgTotalDistance(),
                stats.getRl().getAvgHopCount());
            
            logger.info("   🆚 COMPARISON:");
            logger.info("      Dijkstra Faster: {} | RL Faster: {}",
                stats.getComparison().getDijkstraFasterCount(),
                stats.getComparison().getRlFasterCount());
            logger.info("      Dijkstra Shorter: {} | RL Shorter: {}",
                stats.getComparison().getDijkstraShorterCount(),
                stats.getComparison().getRlShorterCount());
            logger.info("      Avg Latency Diff: {:.2f} ms | Avg Distance Diff: {:.2f} km",
                stats.getComparison().getAvgLatencyDifference(),
                stats.getComparison().getAvgDistanceDifference());
        }
        
        logger.info("=".repeat(80));
    }
    
    /**
     * Tạo batchId duy nhất
     */
    private String generateBatchId() {
        return "batch_" + System.currentTimeMillis() + "_" + UUID.randomUUID().toString().substring(0, 8);
    }
    
    /**
     * Lấy batch theo ID
     */
    public Optional<PacketComparisonBatch> getBatch(String batchId) {
        return batchRepository.findByBatchId(batchId);
    }
}
