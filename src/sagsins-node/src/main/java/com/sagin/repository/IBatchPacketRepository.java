package com.sagin.repository;

import com.sagin.model.BatchPacket;

import java.util.List;
import java.util.Optional;

/**
 * Repository interface cho BatchPacket (lô gói tin)
 */
public interface IBatchPacketRepository {
    
    /**
     * Lưu hoặc cập nhật một batch
     * ✅ Nếu batchId trùng, sẽ xóa document cũ và tạo mới
     */
    void save(BatchPacket batch);
    
    /**
     * Tìm batch theo batchId
     */
    Optional<BatchPacket> findByBatchId(String batchId);
    
    /**
     * Tìm tất cả batches
     */
    List<BatchPacket> findAll();
    
    /**
     * Xóa batch theo batchId
     */
    void deleteByBatchId(String batchId);
}
