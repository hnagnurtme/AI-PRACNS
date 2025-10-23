package com.sagin.repository;

import com.mongodb.bulk.BulkWriteResult;
import com.mongodb.client.*;
import com.mongodb.client.model.*;
import com.sagin.configuration.MongoConfiguration;
import com.sagin.model.NodeInfo;
import org.bson.conversions.Bson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class MongoNodeRepository implements INodeRepository, AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(MongoNodeRepository.class);

    private final MongoClient mongoClient;
    private final MongoCollection<NodeInfo> nodesCollection;

    private boolean isClosed = false;

    public MongoNodeRepository() {
        logger.info("Khởi tạo MongoNodeRepository, kết nối MongoDB...");
        this.mongoClient = MongoClients.create(MongoConfiguration.getMongoClientSettings());
        MongoDatabase database = mongoClient.getDatabase(MongoConfiguration.getDatabaseName());
        this.nodesCollection = database.getCollection(
            MongoConfiguration.NODES_COLLECTION, NodeInfo.class
        );
        logger.info("Kết nối thành công tới '{}'.", MongoConfiguration.NODES_COLLECTION);
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
    public List<NodeInfo> getAllNodes() {
        logger.info("📡 Đang tải toàn bộ danh sách Node từ MongoDB...");
        try (var cursor = nodesCollection.find().iterator()) {
            List<NodeInfo> result = new ArrayList<>();
            while (cursor.hasNext()) {
                result.add(cursor.next());
            }
            logger.info("✅ Tải thành công {} node từ MongoDB.", result.size());
            return result;
        } catch (Exception e) {
            logger.error("❌ Lỗi khi tải danh sách node: {}", e.getMessage(), e);
            return Collections.emptyList();
        }
    }

    /**
     * 🚀 Cập nhật đồng loạt danh sách Node (Batch Update)
     * Tối ưu cho mô phỏng tick-based, giảm I/O tới MongoDB ~10x
     */
    public void bulkUpdateNodes(Collection<NodeInfo> nodes) {
        if (nodes == null || nodes.isEmpty()) return;

        List<WriteModel<NodeInfo>> operations = new ArrayList<>();

        for (NodeInfo info : nodes) {
            info.setLastUpdated(Instant.now());
            Bson filter = Filters.eq("nodeId", info.getNodeId());
            ReplaceOneModel<NodeInfo> replaceModel =
                new ReplaceOneModel<>(filter, info, new ReplaceOptions().upsert(true));
            operations.add(replaceModel);


            logger.info("Trang thai "+ info.toString());
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

    @Override
    public void close() {
        if (mongoClient != null && !isClosed) {
            mongoClient.close();
            isClosed = true;
            logger.info("Đóng kết nối MongoClient.");
        }
    }
}
