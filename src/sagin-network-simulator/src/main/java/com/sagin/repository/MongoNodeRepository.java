package com.sagin.repository;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.ReplaceOptions;

import com.sagin.configuration.MongoConfiguration; 
import com.sagin.model.NodeInfo;
import org.bson.conversions.Bson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;


public class MongoNodeRepository implements INodeRepository {
    
    private static final Logger logger = LoggerFactory.getLogger(MongoNodeRepository.class);
    
    private final MongoClient mongoClient;
    private final MongoDatabase database;
    private final MongoCollection<NodeInfo> nodesCollection; 

    public MongoNodeRepository() {
        // 1. Kết nối với MongoDB Server
        this.mongoClient = MongoClients.create(MongoConfiguration.getMongoClientSettings()); 
        
        // 2. Tham chiếu đến Database
        this.database = mongoClient.getDatabase(MongoConfiguration.getDatabaseName());
        
        // 3. Tham chiếu đến Collection chứa NodeInfo
        // Sử dụng POJO Codec để tự động mapping giữa Document và NodeInfo.class
        this.nodesCollection = database.getCollection(
            MongoConfiguration.NODES_COLLECTION, NodeInfo.class
        );
    }

    
    @Override
    public Map<String, NodeInfo> loadAllNodeConfigs() {
        Map<String, NodeInfo> configs = new HashMap<>();
        
        try {
            
            // Tìm tất cả Document trong Collection và chuyển thành Stream
            List<NodeInfo> nodes = StreamSupport
                .stream(nodesCollection.find().spliterator(), false)
                .collect(Collectors.toList());

            for (NodeInfo node : nodes) {
                // Trong MongoDB, _id là khóa chính, NodeId phải là một trường dữ liệu
                configs.put(node.getNodeId(), node);
            }
            return configs;
            
        } catch (Exception e) {
            logger.error("LỖI TRUY VẤN MONGODB: Không thể tải cấu hình Node.", e);
            return Collections.emptyMap(); 
        }
    }

    @Override
    public void updateNodeInfo(String nodeId, NodeInfo info) {
        // Giả sử NodeInfo.lastUpdated đã được cập nhật trước đó
        info.setLastUpdated(System.currentTimeMillis());

        try {
            Bson filter = Filters.eq("_id", nodeId); 
            
            ReplaceOptions options = new ReplaceOptions().upsert(true);
            
            nodesCollection.replaceOne(filter, info, options);
            
            logger.debug("Cập nhật/Thay thế trạng thái thành công cho Node {}.", nodeId);
            
        } catch (Exception e) {
            logger.error("LỖI GHI DỮ LIỆU MONGODB cho Node {}: {}", nodeId, e.getMessage(), e);
        }
    }

    @Override
    public NodeInfo getNodeInfo(String nodeId) {
        try {
            logger.info("Đang tải thông tin Node {} từ MongoDB...", nodeId);
            
            NodeInfo node = nodesCollection.find(Filters.eq("_id", nodeId)).first();
            
            if (node != null) {
                return node;
            } else {
                logger.warn("Node {} không tồn tại trong Database.", nodeId);
                return null;
            }
            
        } catch (Exception e) {
            logger.error("LỖI TRUY VẤN MONGODB cho Node {}: {}", nodeId, e.getMessage(), e);
            return null;
        }
    }
}