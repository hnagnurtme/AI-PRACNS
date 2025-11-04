package com.sagin.repository;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.ReplaceOptions;
import com.sagin.configuration.MongoConfiguration;
import com.sagin.model.PacketComparisonBatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * MongoDB implementation cho PacketComparisonBatch Repository
 */
public class MongoPacketComparisonBatchRepository implements IPacketComparisonBatchRepository {
    
    private static final Logger logger = LoggerFactory.getLogger(MongoPacketComparisonBatchRepository.class);
    private final MongoCollection<PacketComparisonBatch> collection;
    
    /**
     * Constructor mặc định - tự tạo MongoClient từ MongoConfiguration
     */
    public MongoPacketComparisonBatchRepository() {
        logger.info("[MongoPacketComparisonBatchRepository] Initializing with MongoConfiguration...");
        MongoClient mongoClient = MongoClients.create(MongoConfiguration.getMongoClientSettings());
        MongoDatabase database = mongoClient.getDatabase(MongoConfiguration.getDatabaseName());
        this.collection = database.getCollection("packet_comparison_batches", PacketComparisonBatch.class);
        logger.info("[MongoPacketComparisonBatchRepository] Initialized successfully");
    }
    
    /**
     * Constructor với custom MongoClient và database name
     */
    public MongoPacketComparisonBatchRepository(MongoClient mongoClient, String databaseName) {
        MongoDatabase database = mongoClient.getDatabase(databaseName);
        this.collection = database.getCollection("packet_comparison_batches", PacketComparisonBatch.class);
        logger.info("[MongoPacketComparisonBatchRepository] Initialized with database: {}", databaseName);
    }
    
    @Override
    public void save(PacketComparisonBatch batch) {
        try {
            collection.replaceOne(
                Filters.eq("batchId", batch.getBatchId()),
                batch,
                new ReplaceOptions().upsert(true)
            );
            logger.debug("[MongoPacketComparisonBatchRepository] Saved batch: {}", batch.getBatchId());
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonBatchRepository] Error saving batch: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to save PacketComparisonBatch", e);
        }
    }
    
    @Override
    public Optional<PacketComparisonBatch> findByBatchId(String batchId) {
        try {
            PacketComparisonBatch result = collection.find(Filters.eq("batchId", batchId)).first();
            return Optional.ofNullable(result);
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonBatchRepository] Error finding batch: {}", e.getMessage(), e);
            return Optional.empty();
        }
    }
    
    @Override
    public List<PacketComparisonBatch> findAll() {
        try {
            List<PacketComparisonBatch> results = new ArrayList<>();
            collection.find().into(results);
            return results;
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonBatchRepository] Error finding all batches: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    @Override
    public List<PacketComparisonBatch> findByStatus(String status) {
        try {
            List<PacketComparisonBatch> results = new ArrayList<>();
            collection.find(Filters.eq("status", status)).into(results);
            return results;
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonBatchRepository] Error finding by status: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    @Override
    public List<PacketComparisonBatch> findBySourceAndDestination(String sourceUserId, String destinationUserId) {
        try {
            List<PacketComparisonBatch> results = new ArrayList<>();
            collection.find(
                Filters.and(
                    Filters.eq("metadata.sourceUserId", sourceUserId),
                    Filters.eq("metadata.destinationUserId", destinationUserId)
                )
            ).into(results);
            return results;
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonBatchRepository] Error finding by source/dest: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    @Override
    public void deleteByBatchId(String batchId) {
        try {
            collection.deleteOne(Filters.eq("batchId", batchId));
            logger.debug("[MongoPacketComparisonBatchRepository] Deleted batch: {}", batchId);
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonBatchRepository] Error deleting batch: {}", e.getMessage(), e);
        }
    }
}
