package com.sagin.repository;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.ReplaceOptions;
import com.sagin.configuration.MongoConfiguration;
import com.sagin.model.BatchPacket;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * MongoDB implementation cho BatchPacket Repository
 * Collection: batch_packets
 */
public class MongoBatchPacketRepository implements IBatchPacketRepository {
    
    private static final Logger logger = LoggerFactory.getLogger(MongoBatchPacketRepository.class);
    private final MongoCollection<BatchPacket> collection;
    
    /**
     * Constructor mặc định - tự tạo MongoClient từ MongoConfiguration
     */
    public MongoBatchPacketRepository() {
        logger.info("[MongoBatchPacketRepository] Initializing with MongoConfiguration...");
        MongoClient mongoClient = MongoClients.create(MongoConfiguration.getMongoClientSettings());
        MongoDatabase database = mongoClient.getDatabase(MongoConfiguration.getDatabaseName());
        this.collection = database.getCollection("batch_packets", BatchPacket.class);
        logger.info("[MongoBatchPacketRepository] Initialized successfully");
    }
    
    /**
     * Constructor với custom MongoClient và database name
     */
    public MongoBatchPacketRepository(MongoClient mongoClient, String databaseName) {
        MongoDatabase database = mongoClient.getDatabase(databaseName);
        this.collection = database.getCollection("batch_packets", BatchPacket.class);
        logger.info("[MongoBatchPacketRepository] Initialized with database: {}", databaseName);
    }
    
    @Override
    public void save(BatchPacket batch) {
        try {
            // ✅ Upsert: Nếu batchId trùng → replace (xóa cũ, tạo mới)
            collection.replaceOne(
                Filters.eq("batchId", batch.getBatchId()),
                batch,
                new ReplaceOptions().upsert(true)
            );
            logger.debug("[MongoBatchPacketRepository] Saved BatchPacket: {}", batch.getBatchId());
        } catch (Exception e) {
            logger.error("[MongoBatchPacketRepository] Error saving BatchPacket: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to save BatchPacket", e);
        }
    }
    
    @Override
    public Optional<BatchPacket> findByBatchId(String batchId) {
        try {
            BatchPacket result = collection.find(Filters.eq("batchId", batchId)).first();
            return Optional.ofNullable(result);
        } catch (Exception e) {
            logger.error("[MongoBatchPacketRepository] Error finding BatchPacket: {}", e.getMessage(), e);
            return Optional.empty();
        }
    }
    
    @Override
    public List<BatchPacket> findAll() {
        try {
            List<BatchPacket> results = new ArrayList<>();
            collection.find().into(results);
            return results;
        } catch (Exception e) {
            logger.error("[MongoBatchPacketRepository] Error finding all BatchPackets: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    @Override
    public void deleteByBatchId(String batchId) {
        try {
            collection.deleteOne(Filters.eq("batchId", batchId));
            logger.debug("[MongoBatchPacketRepository] Deleted BatchPacket: {}", batchId);
        } catch (Exception e) {
            logger.error("[MongoBatchPacketRepository] Error deleting BatchPacket: {}", e.getMessage(), e);
        }
    }
}
