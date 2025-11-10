package com.sagin.repository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters; 
import com.sagin.configuration.MongoConfiguration;
import com.sagin.model.UserInfo;

import org.bson.Document;
import org.bson.conversions.Bson;

import java.util.List;
import java.util.Optional;

/**
 * Triển khai IUserRepository sử dụng MongoDB.
 */
public class MongoUserRepository implements IUserRepository, AutoCloseable {
    
    // Sửa tên Logger cho đúng
    private static final Logger logger = LoggerFactory.getLogger(MongoUserRepository.class); 

    private final MongoClient mongoClient;
    private final MongoCollection<UserInfo> usersCollection;

    private boolean isClosed = false;

    public MongoUserRepository() {
        logger.info("Khởi tạo MongoUserRepository, kết nối MongoDB..."); 
        this.mongoClient = MongoClients.create(MongoConfiguration.getMongoClientSettings());
        MongoDatabase database = mongoClient.getDatabase(MongoConfiguration.getDatabaseName());
        this.usersCollection = database.getCollection(
            MongoConfiguration.USERS_COLLECTION, UserInfo.class
        );
        logger.info("Kết nối thành công tới '{}'.", MongoConfiguration.USERS_COLLECTION);
    }

    /**
     * Tìm người dùng trong collection 'users' bằng trường 'userId'.
     *
     * @param userId ID của người dùng (ví dụ: "USER-02")
     * @return Optional chứa UserInfo nếu tìm thấy
     */
    @Override
    public Optional<UserInfo> findByUserId(String userId) {
        if (isClosed) {
            logger.warn("Repository đã đóng. Không thể tìm userId: {}", userId);
            return Optional.empty();
        }

        try {
            // Tạo một bộ lọc BSON để tìm tài liệu có "userId" = userId
            Bson filter = Filters.eq("userId", userId);
            
            // Tìm và lấy tài liệu đầu tiên khớp
            UserInfo user = usersCollection.find(filter).first(); 
            
            // Optional.ofNullable sẽ trả về Optional.empty() nếu user là null
            return Optional.ofNullable(user); 
            
        } catch (Exception e) {
            logger.error("Lỗi khi tìm kiếm CSDL cho userId {}: {}", userId, e.getMessage(), e);
            return Optional.empty();
        }
    }

    public void bulkUpdateUsers(List<UserInfo> users) {
        for (UserInfo user : users) {
            usersCollection.replaceOne(
                    new Document("userId", user.getUserId()),
                    user,
                    new com.mongodb.client.model.ReplaceOptions().upsert(true)
            );
        }
    }

    @Override
    public void updateUserIpAddress(String userId, String ipAddress) {
        if (isClosed) {
            logger.warn("Repository đã đóng. Không thể cập nhật IP cho userId: {}", userId);
            return;
        }

        try {
            Bson filter = Filters.eq("userId", userId);
            Bson update = com.mongodb.client.model.Updates.set("ipAddress", ipAddress);
            usersCollection.updateOne(filter, update);
            logger.info("✅ Updated IP address for user {}: {}", userId, ipAddress);
        } catch (Exception e) {
            logger.error("❌ Failed to update IP address for user {}: {}", userId, e.getMessage(), e);
        }
    }


    /**
     * Đóng kết nối MongoClient.
     */
    @Override
    public void close() {
        if (!isClosed) {
            try {
                mongoClient.close();
                isClosed = true;
                logger.info("Đã đóng kết nối MongoClient cho UserRepository.");
            } catch (Exception e) {
                logger.error("Lỗi khi đóng MongoClient: {}", e.getMessage(), e);
            }
        }
    }
}