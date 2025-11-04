package com.sagin.repository;

import com.sagin.model.PacketComparisonBatch;

import java.util.List;
import java.util.Optional;

/**
 * Repository interface cho PacketComparisonBatch
 */
public interface IPacketComparisonBatchRepository {
    
    /**
     * Lưu hoặc cập nhật batch
     */
    void save(PacketComparisonBatch batch);
    
    /**
     * Tìm batch theo batchId
     */
    Optional<PacketComparisonBatch> findByBatchId(String batchId);
    
    /**
     * Tìm tất cả batches
     */
    List<PacketComparisonBatch> findAll();
    
    /**
     * Tìm batches theo status
     */
    List<PacketComparisonBatch> findByStatus(String status);
    
    /**
     * Tìm batches theo source và destination user
     */
    List<PacketComparisonBatch> findBySourceAndDestination(String sourceUserId, String destinationUserId);
    
    /**
     * Xóa batch theo batchId
     */
    void deleteByBatchId(String batchId);
}
