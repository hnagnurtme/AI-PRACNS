package com.sagin.repository;

import com.sagin.model.PacketComparison;

import java.util.List;
import java.util.Optional;

/**
 * Repository interface cho PacketComparison
 */
public interface IPacketComparisonRepository {
    
    /**
     * Lưu hoặc cập nhật PacketComparison
     */
    void save(PacketComparison comparison);
    
    /**
     * Tìm PacketComparison theo comparisonId
     */
    Optional<PacketComparison> findByComparisonId(String comparisonId);
    
    /**
     * Tìm tất cả PacketComparison theo source và destination
     */
    List<PacketComparison> findBySourceAndDestination(String sourceUserId, String destinationUserId);
    
    /**
     * Tìm tất cả PacketComparison có status = "complete"
     */
    List<PacketComparison> findCompleteComparisons();
    
    /**
     * Xóa PacketComparison theo comparisonId
     */
    void deleteByComparisonId(String comparisonId);
}
