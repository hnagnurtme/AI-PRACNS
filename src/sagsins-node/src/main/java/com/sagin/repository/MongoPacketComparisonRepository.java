package com.sagin.repository;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.ReplaceOptions;
import com.sagin.configuration.MongoConfiguration;
import com.sagin.model.PacketComparison;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * MongoDB implementation cho PacketComparison Repository
 */
public class MongoPacketComparisonRepository implements IPacketComparisonRepository {
    
    private static final Logger logger = LoggerFactory.getLogger(MongoPacketComparisonRepository.class);
    private final MongoCollection<PacketComparison> collection;
    
    /**
     * Constructor mặc định - tự tạo MongoClient từ MongoConfiguration
     */
    public MongoPacketComparisonRepository() {
        logger.info("[MongoPacketComparisonRepository] Initializing with MongoConfiguration...");
        MongoClient mongoClient = MongoClients.create(MongoConfiguration.getMongoClientSettings());
        MongoDatabase database = mongoClient.getDatabase(MongoConfiguration.getDatabaseName());
        this.collection = database.getCollection("packet_comparisons", PacketComparison.class);
        logger.info("[MongoPacketComparisonRepository] Initialized successfully");
    }
    
    /**
     * Constructor với custom MongoClient và database name
     */
    public MongoPacketComparisonRepository(MongoClient mongoClient, String databaseName) {
        MongoDatabase database = mongoClient.getDatabase(databaseName);
        this.collection = database.getCollection("packet_comparisons", PacketComparison.class);
        logger.info("[MongoPacketComparisonRepository] Initialized with database: {}", databaseName);
    }
    
    @Override
    public void save(PacketComparison comparison) {
        try {
            // Upsert: Nếu tồn tại (theo comparisonId) thì update, không thì insert
            collection.replaceOne(
                Filters.eq("comparisonId", comparison.getComparisonId()),
                comparison,
                new ReplaceOptions().upsert(true)
            );
            logger.debug("[MongoPacketComparisonRepository] Saved comparison: {}", comparison.getComparisonId());
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonRepository] Error saving comparison: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to save PacketComparison", e);
        }
    }
    
    @Override
    public Optional<PacketComparison> findByComparisonId(String comparisonId) {
        try {
            PacketComparison result = collection.find(Filters.eq("comparisonId", comparisonId)).first();
            return Optional.ofNullable(result);
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonRepository] Error finding comparison: {}", e.getMessage(), e);
            return Optional.empty();
        }
    }
    
    @Override
    public List<PacketComparison> findBySourceAndDestination(String sourceUserId, String destinationUserId) {
        try {
            List<PacketComparison> results = new ArrayList<>();
            collection.find(
                Filters.and(
                    Filters.eq("sourceUserId", sourceUserId),
                    Filters.eq("destinationUserId", destinationUserId)
                )
            ).into(results);
            return results;
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonRepository] Error finding by source/dest: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    @Override
    public List<PacketComparison> findCompleteComparisons() {
        try {
            List<PacketComparison> results = new ArrayList<>();
            collection.find(Filters.eq("status", "complete")).into(results);
            return results;
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonRepository] Error finding complete comparisons: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    @Override
    public void deleteByComparisonId(String comparisonId) {
        try {
            collection.deleteOne(Filters.eq("comparisonId", comparisonId));
            logger.debug("[MongoPacketComparisonRepository] Deleted comparison: {}", comparisonId);
        } catch (Exception e) {
            logger.error("[MongoPacketComparisonRepository] Error deleting comparison: {}", e.getMessage(), e);
        }
    }
}
