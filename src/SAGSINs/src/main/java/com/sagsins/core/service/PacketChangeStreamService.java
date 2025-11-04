package com.sagsins.core.service;

import org.bson.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.aggregation.Aggregation;
import org.springframework.data.mongodb.core.messaging.ChangeStreamRequest;
import org.springframework.data.mongodb.core.messaging.Message;
import org.springframework.data.mongodb.core.messaging.MessageListenerContainer;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.mongodb.client.model.changestream.ChangeStreamDocument;
import com.sagsins.core.DTOs.request.BatchPacket;
import com.sagsins.core.DTOs.request.TwoPacket;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Service theo dõi realtime các thay đổi trong MongoDB collections
 * - TwoPacket: Chờ đủ 2 packets → Gửi sau 5s từ lần cuối cập nhật → Xóa sau 10s
 * - BatchPacket: Gộp và push mỗi 3 giây (batching) → Xóa sau 10s
 */
@Service
@RequiredArgsConstructor
public class PacketChangeStreamService {
    
    private static final Logger logger = LoggerFactory.getLogger(PacketChangeStreamService.class);
    private static final long SEND_DELAY_MS = 3000; // 3 giây - Chờ 3s từ lần update cuối
    private static final long DELETE_DELAY_MS = 10000; // 10 giây
    
    private final MongoTemplate mongoTemplate;
    private final MessageListenerContainer messageListenerContainer;
    private final SimpMessagingTemplate messagingTemplate;
    private final ObjectMapper objectMapper;
    
    // Scheduler để gửi packets sau khi không còn update
    private ScheduledExecutorService scheduler;
    
    // Scheduler để xóa records sau khi gửi
    private ScheduledExecutorService deleteScheduler;
    
    // Lưu trữ BatchPacket mới nhất để gửi
    private final AtomicReference<BatchPacket> latestBatchPacket = new AtomicReference<>();
    
    // Lưu trữ TwoPacket mới nhất để gửi
    private final AtomicReference<TwoPacket> latestTwoPacket = new AtomicReference<>();
    
    // Scheduled task hiện tại cho TwoPacket (để cancel và reschedule khi có update mới)
    private final AtomicReference<ScheduledFuture<?>> twoPacketSendTask = new AtomicReference<>();
    
    // Scheduled task hiện tại cho BatchPacket (để cancel và reschedule khi có update mới)
    private final AtomicReference<ScheduledFuture<?>> batchPacketSendTask = new AtomicReference<>();
    
    /**
     * Khởi tạo Change Stream listeners khi service start
     */
    @PostConstruct
    public void initChangeStreamListeners() {
        logger.info("Initializing MongoDB Change Stream listeners...");
        
        try {
            // Khởi tạo scheduler cho batch packets và two packets
            scheduler = Executors.newScheduledThreadPool(2, r -> {
                Thread t = new Thread(r, "packet-sender");
                t.setDaemon(true);
                return t;
            });
            
            // Khởi tạo scheduler cho delete tasks
            deleteScheduler = Executors.newScheduledThreadPool(2, r -> {
                Thread t = new Thread(r, "packet-deleter");
                t.setDaemon(true);
                return t;
            });
            
            // Listener cho two_packets collection
            initTwoPacketChangeStream();
            
            // Listener cho batch_packets collection
            initBatchPacketChangeStream();
            
            // Start listening
            messageListenerContainer.start();
            
            logger.info("✅ MongoDB Change Stream listeners started successfully");
        } catch (Exception e) {
            logger.error("❌ Failed to initialize Change Stream listeners: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Stop listeners khi service shutdown
     */
    @PreDestroy
    public void stopChangeStreamListeners() {
        logger.info("Stopping MongoDB Change Stream listeners...");
        
        // Cancel pending TwoPacket send task
        ScheduledFuture<?> pendingTwoTask = twoPacketSendTask.getAndSet(null);
        if (pendingTwoTask != null && !pendingTwoTask.isDone()) {
            pendingTwoTask.cancel(false);
        }
        
        // Cancel pending BatchPacket send task
        ScheduledFuture<?> pendingBatchTask = batchPacketSendTask.getAndSet(null);
        if (pendingBatchTask != null && !pendingBatchTask.isDone()) {
            pendingBatchTask.cancel(false);
        }
        
        // Shutdown scheduler
        if (scheduler != null && !scheduler.isShutdown()) {
            scheduler.shutdown();
            try {
                if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                    scheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                scheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        
        // Shutdown delete scheduler
        if (deleteScheduler != null && !deleteScheduler.isShutdown()) {
            deleteScheduler.shutdown();
            try {
                if (!deleteScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                    deleteScheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                deleteScheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        
        messageListenerContainer.stop();
        logger.info("✅ Change Stream listeners stopped");
    }
    
    /**
     * Khởi tạo Change Stream cho two_packets collection
     * Theo dõi: INSERT, UPDATE, REPLACE operations
     */
    private void initTwoPacketChangeStream() {
        Aggregation filter = Aggregation.newAggregation(
            Aggregation.match(Criteria.where("operationType")
                .in("insert", "update", "replace"))
        );
        
        ChangeStreamRequest<TwoPacket> request = ChangeStreamRequest.builder(this::handleTwoPacketChange)
            .collection("two_packets")
            .filter(filter)
            .build();
        
        messageListenerContainer.register(request, TwoPacket.class);
        logger.info("✅ Registered Change Stream listener for 'two_packets' collection (Send after {}ms from last update)", SEND_DELAY_MS);
    }
    
    /**
     * Khởi tạo Change Stream cho batch_packets collection
     * Theo dõi: INSERT, UPDATE, REPLACE operations
     */
    private void initBatchPacketChangeStream() {
        Aggregation filter = Aggregation.newAggregation(
            Aggregation.match(Criteria.where("operationType")
                .in("insert", "update", "replace"))
        );
        
        ChangeStreamRequest<BatchPacket> request = ChangeStreamRequest.builder(this::handleBatchPacketChange)
            .collection("batch_packets")
            .filter(filter)
            .build();
        
        messageListenerContainer.register(request, BatchPacket.class);
        logger.info("✅ Registered Change Stream listener for 'batch_packets' collection (Send after {}ms from last update)", SEND_DELAY_MS);
    }
    
    /**
     * Xử lý thay đổi trong two_packets collection
     * Gửi sau 3s từ lần cuối cập nhật (chỉ khi đủ 2 packets: dijkstra và rl không null)
     */
    private void handleTwoPacketChange(Message<ChangeStreamDocument<Document>, TwoPacket> message) {
        try {
            TwoPacket packet = message.getBody();
            
            if (packet == null) {
                logger.warn("Received null TwoPacket in change event");
                return;
            }
            
            ChangeStreamDocument<Document> raw = message.getRaw();
            String operationType = "unknown";
            if (raw != null && raw.getOperationType() != null) {
                operationType = raw.getOperationType().getValue();
            }
            
            // Kiểm tra có đủ 2 packets không (REQUIRED: cả dijkstra và rl phải có)
            boolean hasBothPackets = packet.getDijkstraPacket() != null && packet.getRlPacket() != null;
            
            logger.info("🔄 [{}] TwoPacket received - pairId={}, dijkstra={}, rl={}, complete={}", 
                operationType.toUpperCase(),
                packet.getPairId(),
                packet.getDijkstraPacket() != null ? "✓" : "✗",
                packet.getRlPacket() != null ? "✓" : "✗",
                hasBothPackets ? "YES" : "NO");
            
            // Cancel task cũ nếu có (reset timer vì có update mới)
            ScheduledFuture<?> oldTask = twoPacketSendTask.getAndSet(null);
            if (oldTask != null && !oldTask.isDone()) {
                oldTask.cancel(false);
                logger.debug("⏹️ Cancelled previous TwoPacket send task (reset timer due to new update)");
            }
            
            // Chỉ schedule gửi nếu đã đủ 2 packets
            if (hasBothPackets) {
                // Lưu packet để gửi
                latestTwoPacket.set(packet);
                
                // Schedule gửi sau 3 giây (sẽ bị cancel nếu có update mới)
                ScheduledFuture<?> newTask = scheduler.schedule(() -> {
                    try {
                        TwoPacket packetToSend = latestTwoPacket.getAndSet(null);
                        
                        if (packetToSend != null && packetToSend.getPairId().equals(packet.getPairId())) {
                            // Double-check: Vẫn đủ 2 packets
                            if (packetToSend.getDijkstraPacket() != null && packetToSend.getRlPacket() != null) {
                                // Push message qua WebSocket
                                messagingTemplate.convertAndSend("/topic/packets", packetToSend);
                                
                                // Log JSON đã gửi
                                String json = objectMapper.writeValueAsString(packetToSend);
                                logger.info("📤 [SENT] TwoPacket to /topic/packets - pairId={}, dijkstra={}, rl={}\n📄 JSON: {}", 
                                    packetToSend.getPairId(),
                                    packetToSend.getDijkstraPacket().getPacketId(),
                                    packetToSend.getRlPacket().getPacketId(),
                                    json);
                                
                                // Schedule xóa sau 10 giây
                                scheduleDeleteTwoPacket(packetToSend.getPairId());
                            } else {
                                logger.warn("⚠️ TwoPacket incomplete at send time - pairId={}, skipping send", 
                                    packetToSend.getPairId());
                            }
                        }
                    } catch (Exception e) {
                        logger.error("Error sending TwoPacket: {}", e.getMessage(), e);
                    }
                }, SEND_DELAY_MS, TimeUnit.MILLISECONDS);
                
                twoPacketSendTask.set(newTask);
                logger.info("⏰ Scheduled TwoPacket send in {}ms - pairId={} (will reset if updated)", SEND_DELAY_MS, packet.getPairId());
            } else {
                // Không đủ 2 packets → không lưu, không schedule
                latestTwoPacket.set(null);
                logger.info("⏸️ TwoPacket incomplete - pairId={}, waiting for both packets (dijkstra AND rl required)", packet.getPairId());
            }
                
        } catch (Exception e) {
            logger.error("Error handling TwoPacket change: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Xử lý thay đổi trong batch_packets collection
     * Gửi sau 3s từ lần cuối cập nhật
     */
    private void handleBatchPacketChange(Message<ChangeStreamDocument<Document>, BatchPacket> message) {
        try {
            BatchPacket batch = message.getBody();
            
            if (batch == null) {
                logger.warn("Received null BatchPacket in change event");
                return;
            }
            
            ChangeStreamDocument<Document> raw = message.getRaw();
            String operationType = "unknown";
            if (raw != null && raw.getOperationType() != null) {
                operationType = raw.getOperationType().getValue();
            }
            
            logger.info("🔄 [{}] BatchPacket received - batchId={}, totalPairs={}, packetsCount={}", 
                operationType.toUpperCase(),
                batch.getBatchId(),
                batch.getTotalPairPackets(),
                batch.getPackets() != null ? batch.getPackets().size() : 0);
            
            // Cancel task cũ nếu có (reset timer vì có update mới)
            ScheduledFuture<?> oldTask = batchPacketSendTask.getAndSet(null);
            if (oldTask != null && !oldTask.isDone()) {
                oldTask.cancel(false);
                logger.debug("⏹️ Cancelled previous BatchPacket send task (reset timer due to new update)");
            }
            
            // Lưu batch mới nhất
            latestBatchPacket.set(batch);
            
            // Schedule gửi sau 3 giây (sẽ bị cancel nếu có update mới)
            ScheduledFuture<?> newTask = scheduler.schedule(() -> {
                try {
                    BatchPacket batchToSend = latestBatchPacket.getAndSet(null);
                    
                    if (batchToSend != null && batchToSend.getBatchId().equals(batch.getBatchId())) {
                        // Push message qua WebSocket
                        messagingTemplate.convertAndSend("/topic/batchpacket", batchToSend);
                        
                        // Log JSON đã gửi
                        String json = objectMapper.writeValueAsString(batchToSend);
                        logger.info("📤 [SENT] BatchPacket to /topic/batchpacket - batchId={}, totalPairs={}, packetsCount={}\n📄 JSON: {}", 
                            batchToSend.getBatchId(),
                            batchToSend.getTotalPairPackets(),
                            batchToSend.getPackets() != null ? batchToSend.getPackets().size() : 0,
                            json);
                        
                        // Schedule xóa sau 10 giây
                        scheduleDeleteBatchPacket(batchToSend.getBatchId());
                    }
                } catch (Exception e) {
                    logger.error("Error sending BatchPacket: {}", e.getMessage(), e);
                }
            }, SEND_DELAY_MS, TimeUnit.MILLISECONDS);
            
            batchPacketSendTask.set(newTask);
            logger.info("⏰ Scheduled BatchPacket send in {}ms - batchId={} (will reset if updated)", SEND_DELAY_MS, batch.getBatchId());
                
        } catch (Exception e) {
            logger.error("Error handling BatchPacket change: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Schedule task để xóa TwoPacket sau 10 giây
     */
    private void scheduleDeleteTwoPacket(String pairId) {
        deleteScheduler.schedule(() -> {
            try {
                Query query = new Query(Criteria.where("pairId").is(pairId));
                long deletedCount = mongoTemplate.remove(query, "two_packets").getDeletedCount();
                
                if (deletedCount > 0) {
                    logger.info("🗑️ [DELETED] TwoPacket - pairId={} (after {}ms)", pairId, DELETE_DELAY_MS);
                } else {
                    logger.warn("⚠️ [NOT FOUND] TwoPacket - pairId={} already deleted or not exists", pairId);
                }
            } catch (Exception e) {
                logger.error("Error deleting TwoPacket pairId={}: {}", pairId, e.getMessage(), e);
            }
        }, DELETE_DELAY_MS, TimeUnit.MILLISECONDS);
    }
    
    /**
     * Schedule task để xóa BatchPacket sau 10 giây
     */
    private void scheduleDeleteBatchPacket(String batchId) {
        deleteScheduler.schedule(() -> {
            try {
                Query query = new Query(Criteria.where("batchId").is(batchId));
                long deletedCount = mongoTemplate.remove(query, "batch_packets").getDeletedCount();
                
                if (deletedCount > 0) {
                    logger.info("🗑️ [DELETED] BatchPacket - batchId={} (after {}ms)", batchId, DELETE_DELAY_MS);
                } else {
                    logger.warn("⚠️ [NOT FOUND] BatchPacket - batchId={} already deleted or not exists", batchId);
                }
            } catch (Exception e) {
                logger.error("Error deleting BatchPacket batchId={}: {}", batchId, e.getMessage(), e);
            }
        }, DELETE_DELAY_MS, TimeUnit.MILLISECONDS);
    }
}