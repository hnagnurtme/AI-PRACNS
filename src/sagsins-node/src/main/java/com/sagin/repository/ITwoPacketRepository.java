package com.sagin.repository;

import com.sagin.model.TwoPacket;

import java.util.List;
import java.util.Optional;

/**
 * Repository interface cho TwoPacket (cặp gói tin Dijkstra + RL)
 * ✅ Mỗi cặp user chỉ có 1 document (xóa và ghi đè)
 */
public interface ITwoPacketRepository {
    
    /**
     * Lưu hoặc cập nhật một cặp packet
     * ✅ Upsert theo pairId (format: sourceUserId_destinationUserId)
     * ✅ Nếu tồn tại → xóa và ghi đè
     */
    void save(TwoPacket twoPacket);
    
    /**
     * Tìm cặp packet theo pairId (sourceUserId_destinationUserId)
     */
    Optional<TwoPacket> findByPairId(String pairId);
    
    /**
     * Tìm tất cả cặp packets
     */
    List<TwoPacket> findAll();
    
    /**
     * Xóa cặp packet theo pairId
     */
    void deleteByPairId(String pairId);
}
