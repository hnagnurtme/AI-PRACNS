package com.example.repository;

import com.mongodb.bulk.BulkWriteResult;
import com.mongodb.client.*;
import com.mongodb.client.model.*;
import com.example.configuration.MongoConfiguration;
import com.example.model.NodeInfo;
import org.bson.conversions.Bson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Triển khai INodeRepository sử dụng MongoDB (Singleton Pattern).
 * Quản lý MỘT MongoClient duy nhất cho toàn bộ ứng dụng.
 */
public class MongoNodeRepository implements INodeRepository {
    private static final Logger logger = LoggerFactory.getLogger(MongoNodeRepository.class);

    // === SỬA LỖI 1: Singleton Pattern ===
    private static final MongoNodeRepository INSTANCE = new MongoNodeRepository();

    private final MongoClient mongoClient;
    private final MongoCollection<NodeInfo> nodesCollection;

    /**
     * Constructor private để đảm bảo Singleton.
     */
    private MongoNodeRepository() {
        logger.info("Khởi tạo Singleton MongoNodeRepository, kết nối MongoDB...");
        try {
            this.mongoClient = MongoClients.create(MongoConfiguration.getMongoClientSettings());
            MongoDatabase database = mongoClient.getDatabase(MongoConfiguration.getDatabaseName());
            this.nodesCollection = database.getCollection(
                MongoConfiguration.NODES_COLLECTION, NodeInfo.class
            );

            // === SỬA LỖI 2: Đăng ký Shutdown Hook ===
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    mongoClient.close();
                    logger.info("Đã đóng kết nối MongoClient (thông qua Shutdown Hook).");
                } catch (Exception e) {
                    logger.error("Lỗi khi đóng MongoClient trong Shutdown Hook: {}", e.getMessage(), e);
                }
            }));
            
            logger.info("Kết nối thành công tới '{}'. Repository sẵn sàng.", MongoConfiguration.NODES_COLLECTION);
            
        } catch (Exception e) {
            logger.error("KHỞI TẠO MONGOUSERREPOSITORY THẤT BẠI!", e);
            throw new RuntimeException("Không thể khởi tạo kết nối MongoDB", e);
        }
    }

    /**
     * Lấy instance duy nhất của Repository.
     */
    public static MongoNodeRepository getInstance() {
        return INSTANCE;
    }

    @Override
    public Map<String, NodeInfo> loadAllNodeConfigs() {
        logger.debug("Tải tất cả cấu hình node từ MongoDB...");
        try (var cursor = nodesCollection.find().iterator()) {
            return StreamSupport
                .stream(((Iterable<NodeInfo>) () -> cursor).spliterator(), false)
                .collect(Collectors.toMap(NodeInfo::getNodeId, Function.identity()));
        } catch (Exception e) {
            logger.error("Lỗi tải node configs:", e);
            return Collections.emptyMap();
        }
    }

    @Override
    public void updateNodeInfo(String nodeId, NodeInfo info) {
        if (info == null) return;
        info.setLastUpdated(Instant.now());

        try {
            Bson filter = Filters.eq("nodeId", nodeId);
            ReplaceOptions options = new ReplaceOptions().upsert(true);
            nodesCollection.replaceOne(filter, info, options);
            logger.debug("Cập nhật Node {} thành công", nodeId);
        } catch (Exception e) {
            logger.error("Lỗi cập nhật Node {}: {}", nodeId, e.getMessage(), e);
        }
    }

    @Override
    public Optional<NodeInfo> getNodeInfo(String nodeId) {
        try {
            return Optional.ofNullable(nodesCollection.find(Filters.eq("nodeId", nodeId)).first());
        } catch (Exception e) {
            logger.error("Lỗi khi tìm Node {}: {}", nodeId, e.getMessage(), e);
            return Optional.empty();
        }
    }

    @Override
    public void bulkUpdateNodes(Collection<NodeInfo> nodes) {
        if (nodes == null || nodes.isEmpty()) return;

        List<WriteModel<NodeInfo>> operations = new ArrayList<>(nodes.size());
        ReplaceOptions options = new ReplaceOptions().upsert(true);

        for (NodeInfo info : nodes) {
            info.setLastUpdated(Instant.now());
            Bson filter = Filters.eq("nodeId", info.getNodeId());
            ReplaceOneModel<NodeInfo> replaceModel = new ReplaceOneModel<>(filter, info, options);
            operations.add(replaceModel);

            // === SỬA LỖI 3: Chuyển sang DEBUG ===
            // logger.info("Trang thai "+ info.toString()); // Gây spam
            logger.debug("Chuẩn bị bulk update: {}", info.toString());
        }

        try {
            BulkWriteResult result = nodesCollection.bulkWrite(operations, new BulkWriteOptions().ordered(false));
            logger.debug(
                "Bulk update hoàn tất: {} upserts, {} matched.",
                result.getUpserts().size(),
                result.getMatchedCount()
            );
        } catch (Exception e) {
            logger.error("Lỗi bulk update Node batch: {}", e.getMessage(), e);
        }
    }
}