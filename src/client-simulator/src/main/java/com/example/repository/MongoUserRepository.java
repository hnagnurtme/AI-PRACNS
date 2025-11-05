package com.example.repository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.example.configuration.MongoConfiguration;
import com.example.model.UserInfo;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.ReplaceOneModel;
import com.mongodb.client.model.ReplaceOptions;
import com.mongodb.client.model.WriteModel;

import org.bson.conversions.Bson;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Triển khai IUserRepository sử dụng MongoDB (Singleton Pattern).
 * Quản lý MỘT MongoClient duy nhất cho toàn bộ ứng dụng.
 */
public class MongoUserRepository implements IUserRepository {
    
    private static final Logger logger = LoggerFactory.getLogger(MongoUserRepository.class); 

    // === SỬA LỖI 1: Singleton Pattern ===
    private static final MongoUserRepository INSTANCE = new MongoUserRepository();

    private final MongoClient mongoClient;
    private final MongoCollection<UserInfo> usersCollection;

    // private boolean isClosed = false; // XÓA CỜ NÀY

    /**
     * Constructor private để đảm bảo Singleton.
     */
    private MongoUserRepository() {
        logger.info("Khởi tạo Singleton MongoUserRepository, kết nối MongoDB..."); 
        try {
            this.mongoClient = MongoClients.create(MongoConfiguration.getMongoClientSettings());
            MongoDatabase database = mongoClient.getDatabase(MongoConfiguration.getDatabaseName());
            this.usersCollection = database.getCollection(
                MongoConfiguration.USERS_COLLECTION, UserInfo.class
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
            
            logger.info("Kết nối thành công tới '{}'. Repository sẵn sàng.", MongoConfiguration.USERS_COLLECTION);
            
        } catch (Exception e) {
            logger.error("KHỞI TẠO MONGOUSERREPOSITORY THẤT BẠI!", e);
            // Ném lỗi này để ứng dụng biết và dừng lại, thay vì chạy với repo = null
            throw new RuntimeException("Không thể khởi tạo kết nối MongoDB", e);
        }
    }

    /**
     * Lấy instance duy nhất của Repository.
     */
    public static MongoUserRepository getInstance() {
        return INSTANCE;
    }

    @Override
    public Optional<UserInfo> findByUserId(String userId) {
        // if (isClosed) { ... } // XÓA CHECK NÀY
        
        try {
            Bson filter = Filters.eq("userId", userId);
            UserInfo user = usersCollection.find(filter).first(); 
            return Optional.ofNullable(user); 
        } catch (Exception e) {
            logger.error("Lỗi khi tìm kiếm CSDL cho userId {}: {}", userId, e.getMessage(), e);
            return Optional.empty();
        }
    }

    /**
     * === SỬA LỖI 3: Dùng bulkWrite ===
     * Cập nhật/Thêm (Upsert) hàng loạt user một cách hiệu quả.
     */
    @Override
    public void bulkUpdateUsers(List<UserInfo> users) {
        if (users == null || users.isEmpty()) {
            return;
        }

        try {
            // 1. Tạo danh sách các hành động
            List<WriteModel<UserInfo>> operations = new ArrayList<>();
            ReplaceOptions options = new ReplaceOptions().upsert(true);

            for (UserInfo user : users) {
                Bson filter = Filters.eq("userId", user.getUserId());
                operations.add(new ReplaceOneModel<>(filter, user, options));
            }
            
            // 2. Gửi MỘT lệnh duy nhất
            logger.info("Thực hiện bulkWrite cho {} users...", operations.size());
            usersCollection.bulkWrite(operations);
            
        } catch (Exception e) {
            logger.error("Lỗi khi thực hiện bulkUpdateUsers: {}", e.getMessage(), e);
        }
    }

    @Override
    public List<UserInfo> findAll(){
        try {
            return usersCollection.find().into(new ArrayList<>());
        } catch (Exception e) {
            logger.error("Lỗi khi findAll users: {}", e.getMessage(), e);
            return new ArrayList<>(); // Trả về list rỗng nếu lỗi
        }
    }

}