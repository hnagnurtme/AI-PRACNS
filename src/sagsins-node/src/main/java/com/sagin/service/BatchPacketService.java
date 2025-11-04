package com.sagin.service;

import com.sagin.model.BatchPacket;
import com.sagin.model.Packet;
import com.sagin.model.TwoPacket;
import com.sagin.repository.IBatchPacketRepository;
import com.sagin.repository.ITwoPacketRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Optional;

/**
 * Service quản lý BatchPacket và TwoPacket
 * Xử lý việc tạo, lưu và query batch packets
 * ✅ Tự động lưu packet vào 2 collections khi packet đến (drop/success)
 */
public class BatchPacketService {
    
    private static final Logger logger = LoggerFactory.getLogger(BatchPacketService.class);
    private final IBatchPacketRepository batchRepository;
    private final ITwoPacketRepository twoPacketRepository;
    
    public BatchPacketService(
            IBatchPacketRepository batchRepository,
            ITwoPacketRepository twoPacketRepository) {
        this.batchRepository = batchRepository;
        this.twoPacketRepository = twoPacketRepository;
    }
    
    /**
     * ✅ Lưu packet vào database (drop hoặc success)
     * Tự động tạo/update TwoPacket và append vào BatchPacket
     */
    public void savePacket(Packet packet) {
        if (packet == null) {
            logger.warn("[BatchPacketService] Cannot save null packet");
            return;
        }
        
        String sourceUserId = packet.getSourceUserId();
        String destinationUserId = packet.getDestinationUserId();
        String pairId = generatePairId(sourceUserId, destinationUserId);
        
        // ✅ 1. Lưu/Update TwoPacket (xóa và ghi đè)
        saveTwoPacket(pairId, packet);
        
        // ✅ 2. Append vào BatchPacket
        appendToBatch(pairId, sourceUserId, destinationUserId, packet);
    }
    
    /**
     * Tạo và lưu batch mới
     * ✅ BatchId = sourceUserId_destinationUserId
     * ✅ Nếu trùng ID → xóa document cũ
     */
    public BatchPacket createBatch(String sourceUserId, String destinationUserId, int totalPairs) {
        // ✅ Tạo batchId từ source và destination
        String batchId = generateBatchId(sourceUserId, destinationUserId);
        
        // ✅ Kiểm tra batch cũ
        Optional<BatchPacket> existingBatch = batchRepository.findByBatchId(batchId);
        if (existingBatch.isPresent()) {
            logger.info("[BatchPacketService] Batch {} already exists. Deleting old batch...", batchId);
            batchRepository.deleteByBatchId(batchId);
        }
        
        // Tạo batch mới
        BatchPacket batch = new BatchPacket();
        batch.setBatchId(batchId);
        batch.setTotalPairPackets(totalPairs);
        
        batchRepository.save(batch);
        logger.info("[BatchPacketService] ✅ Created batch {} with {} pairs", batchId, totalPairs);
        
        return batch;
    }
    
    /**
     * Thêm cặp packet vào batch
     * ✅ TwoPacket: Xóa và ghi đè (chỉ giữ 1 document mới nhất)
     * ✅ BatchPacket: Ghi chèn (append) TwoPacket vào array packets[]
     */
    public void addTwoPacketToBatch(String batchId, TwoPacket twoPacket) {
        try {
            // ✅ Lưu TwoPacket vào collection riêng (UPSERT - xóa và ghi đè)
            twoPacketRepository.save(twoPacket);
            logger.debug("[BatchPacketService] ✅ Saved/Replaced TwoPacket: {}", twoPacket.getPairId());
            
            // ✅ Cập nhật batch: GHI CHÈN TwoPacket vào array
            Optional<BatchPacket> batchOpt = batchRepository.findByBatchId(batchId);
            if (batchOpt.isPresent()) {
                BatchPacket batch = batchOpt.get();
                
                // ✅ LUÔN LUÔN thêm vào array (không check trùng)
                batch.getPackets().add(twoPacket);
                batchRepository.save(batch);
                
                logger.info("[BatchPacketService] ✅ Appended TwoPacket to batch {} | Total packets: {}", 
                    batchId, batch.getPackets().size());
            } else {
                logger.warn("[BatchPacketService] Batch {} not found", batchId);
            }
            
        } catch (Exception e) {
            logger.error("[BatchPacketService] Error adding TwoPacket to batch: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Lưu toàn bộ batch (bao gồm tất cả TwoPackets)
     */
    public void saveBatch(BatchPacket batch) {
        try {
            // Lưu từng TwoPacket vào collection riêng
            for (TwoPacket twoPacket : batch.getPackets()) {
                twoPacketRepository.save(twoPacket);
            }
            
            // Lưu batch
            batchRepository.save(batch);
            logger.info("[BatchPacketService] ✅ Saved batch {} with {} packets", 
                batch.getBatchId(), batch.getPackets().size());
            
        } catch (Exception e) {
            logger.error("[BatchPacketService] Error saving batch: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to save batch", e);
        }
    }
    
    /**
     * Lấy batch theo ID
     */
    public Optional<BatchPacket> getBatch(String batchId) {
        return batchRepository.findByBatchId(batchId);
    }
    
    /**
     * Tạo batchId từ sourceUserId và destinationUserId
     * ✅ Format: sourceUserId_destinationUserId
     */
    private String generateBatchId(String sourceUserId, String destinationUserId) {
        return sourceUserId + "_" + destinationUserId;
    }
    
    /**
     * Tạo pairId từ sourceUserId và destinationUserId
     * ✅ Format: sourceUserId_destinationUserId
     */
    private String generatePairId(String sourceUserId, String destinationUserId) {
        return sourceUserId + "_" + destinationUserId;
    }
    
    /**
     * ✅ Lưu/Update TwoPacket (xóa và ghi đè)
     */
    private void saveTwoPacket(String pairId, Packet packet) {
        try {
            // Tìm TwoPacket hiện tại
            Optional<TwoPacket> existingOpt = twoPacketRepository.findByPairId(pairId);
            
            TwoPacket twoPacket;
            if (existingOpt.isPresent()) {
                // Cập nhật TwoPacket hiện có
                twoPacket = existingOpt.get();
                if (packet.isUseRL()) {
                    twoPacket.setRlPacket(packet);
                    logger.debug("[BatchPacketService] Updated RL packet in TwoPacket: {}", pairId);
                } else {
                    twoPacket.setDijkstraPacket(packet);
                    logger.debug("[BatchPacketService] Updated Dijkstra packet in TwoPacket: {}", pairId);
                }
            } else {
                // Tạo TwoPacket mới
                twoPacket = new TwoPacket();
                twoPacket.setPairId(pairId);
                if (packet.isUseRL()) {
                    twoPacket.setRlPacket(packet);
                } else {
                    twoPacket.setDijkstraPacket(packet);
                }
                logger.debug("[BatchPacketService] Created new TwoPacket: {}", pairId);
            }
            
            // ✅ Lưu/Ghi đè vào collection two_packets
            twoPacketRepository.save(twoPacket);
            logger.info("[BatchPacketService] ✅ Saved TwoPacket: {} | Algorithm: {} | Dropped: {}", 
                pairId, packet.isUseRL() ? "RL" : "Dijkstra", packet.isDropped());
            
        } catch (Exception e) {
            logger.error("[BatchPacketService] Error saving TwoPacket: {}", e.getMessage(), e);
        }
    }
    
    /**
     * ✅ Append TwoPacket vào BatchPacket
     */
    private void appendToBatch(String pairId, String sourceUserId, String destinationUserId, Packet packet) {
        try {
            String batchId = generateBatchId(sourceUserId, destinationUserId);
            
            // Tìm hoặc tạo batch
            Optional<BatchPacket> batchOpt = batchRepository.findByBatchId(batchId);
            BatchPacket batch;
            
            if (batchOpt.isEmpty()) {
                // Tạo batch mới nếu chưa có
                batch = new BatchPacket();
                batch.setBatchId(batchId);
                batch.setTotalPairPackets(0); // Sẽ tăng dần
                batch.setPackets(new ArrayList<>());
                logger.debug("[BatchPacketService] Created new BatchPacket: {}", batchId);
            } else {
                batch = batchOpt.get();
            }
            
            // Lấy TwoPacket hiện tại
            Optional<TwoPacket> twoPacketOpt = twoPacketRepository.findByPairId(pairId);
            if (twoPacketOpt.isPresent()) {
                TwoPacket currentTwoPacket = twoPacketOpt.get();
                
                // ✅ LUÔN LUÔN append vào array (ghi chèn)
                batch.getPackets().add(currentTwoPacket);
                batch.setTotalPairPackets(batch.getPackets().size());
                
                // Lưu batch
                batchRepository.save(batch);
                logger.info("[BatchPacketService] ✅ Appended to BatchPacket: {} | Total packets: {} | Algorithm: {}", 
                    batchId, batch.getPackets().size(), packet.isUseRL() ? "RL" : "Dijkstra");
            }
            
        } catch (Exception e) {
            logger.error("[BatchPacketService] Error appending to batch: {}", e.getMessage(), e);
        }
    }
}
