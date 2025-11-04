package com.sagin.repository;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.ReplaceOptions;
import com.sagin.configuration.MongoConfiguration;
import com.sagin.model.TwoPacket;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * MongoDB implementation cho TwoPacket Repository
 * Collection: two_packets
 */
public class MongoTwoPacketRepository implements ITwoPacketRepository {
    
    private static final Logger logger = LoggerFactory.getLogger(MongoTwoPacketRepository.class);
    private final MongoCollection<TwoPacket> collection;
    
    /**
     * Constructor mặc định - tự tạo MongoClient từ MongoConfiguration
     */
    public MongoTwoPacketRepository() {
        logger.info("[MongoTwoPacketRepository] Initializing with MongoConfiguration...");
        MongoClient mongoClient = MongoClients.create(MongoConfiguration.getMongoClientSettings());
        MongoDatabase database = mongoClient.getDatabase(MongoConfiguration.getDatabaseName());
        this.collection = database.getCollection("two_packets", TwoPacket.class);
        logger.info("[MongoTwoPacketRepository] Initialized successfully");
    }
    
    /**
     * Constructor với custom MongoClient và database name
     */
    public MongoTwoPacketRepository(MongoClient mongoClient, String databaseName) {
        MongoDatabase database = mongoClient.getDatabase(databaseName);
        this.collection = database.getCollection("two_packets", TwoPacket.class);
        logger.info("[MongoTwoPacketRepository] Initialized with database: {}", databaseName);
    }
    
    @Override
    public void save(TwoPacket twoPacket) {
        try {
            // Upsert: Nếu tồn tại (theo pairId) thì update, không thì insert
            collection.replaceOne(
                Filters.eq("pairId", twoPacket.getPairId()),
                twoPacket,
                new ReplaceOptions().upsert(true)
            );
            logger.debug("[MongoTwoPacketRepository] Saved TwoPacket: {}", twoPacket.getPairId());
        } catch (Exception e) {
            logger.error("[MongoTwoPacketRepository] Error saving TwoPacket: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to save TwoPacket", e);
        }
    }
    
    @Override
    public Optional<TwoPacket> findByPairId(String pairId) {
        try {
            TwoPacket result = collection.find(Filters.eq("pairId", pairId)).first();
            return Optional.ofNullable(result);
        } catch (Exception e) {
            logger.error("[MongoTwoPacketRepository] Error finding TwoPacket: {}", e.getMessage(), e);
            return Optional.empty();
        }
    }
    
    @Override
    public List<TwoPacket> findAll() {
        try {
            List<TwoPacket> results = new ArrayList<>();
            collection.find().into(results);
            return results;
        } catch (Exception e) {
            logger.error("[MongoTwoPacketRepository] Error finding all TwoPackets: {}", e.getMessage(), e);
            return new ArrayList<>();
        }
    }
    
    @Override
    public void deleteByPairId(String pairId) {
        try {
            collection.deleteOne(Filters.eq("pairId", pairId));
            logger.debug("[MongoTwoPacketRepository] Deleted TwoPacket: {}", pairId);
        } catch (Exception e) {
            logger.error("[MongoTwoPacketRepository] Error deleting TwoPacket: {}", e.getMessage(), e);
        }
    }
}
