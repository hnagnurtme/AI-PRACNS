package com.sagin.configuration;

import com.mongodb.ConnectionString;
import com.mongodb.MongoClientSettings;
import org.bson.codecs.configuration.CodecProvider;
import org.bson.codecs.configuration.CodecRegistry;
import org.bson.codecs.pojo.PojoCodecProvider;

import java.util.Optional;

import static org.bson.codecs.configuration.CodecRegistries.fromProviders;
import static org.bson.codecs.configuration.CodecRegistries.fromRegistries;

/**
 * Cấu hình tĩnh cho MongoDB, bao gồm các hằng số, tiện ích kết nối và cấu hình
 * POJO Codec.
 */
public final class MongoConfiguration { // Thêm 'final' để ngăn kế thừa

    // --- CẤU HÌNH KẾT NỐI ---
    public static final String CONNECTION_STRING = "mongodb://user:password123@localhost:27017/?authSource=admin";
    public static final String DATABASE_NAME = "sagsin_network";
    public static final String NODES_COLLECTION = "nodes";
    public static final String USERS_COLLECTION = "users";

    // --- TẠO INSTANCE SINGLETON CHO MONGO CLIENT SETTINGS ---
    // <<< CẢI TIẾN: Tạo một lần duy nhất để tối ưu hiệu suất
    private static final MongoClientSettings MONGO_CLIENT_SETTINGS = createMongoClientSettings();

    // Private constructor để ngăn việc tạo instance của lớp tiện ích này
    private MongoConfiguration() {}

    // --- TIỆN ÍCH CÔNG KHAI ---

    public static String getConnectionString() {
        return Optional.ofNullable(System.getenv("MONGO_URI"))
                .orElse(CONNECTION_STRING);
    }

    public static String getDatabaseName() {
        return DATABASE_NAME;
    }

    /**
     * Lấy về instance MongoClientSettings đã được cấu hình sẵn.
     * Đây là cách tiếp cận Singleton để đảm bảo hiệu suất và tính nhất quán.
     */
    public static MongoClientSettings getMongoClientSettings() {
        return MONGO_CLIENT_SETTINGS;
    }

    public static boolean isConfigReady() {
        // Có thể kiểm tra kỹ hơn nếu cần, ví dụ ping server
        return !getConnectionString().isEmpty() && !DATABASE_NAME.isEmpty();
    }

    // --- CÁC PHƯƠNG THỨC KHỞI TẠO NỘI BỘ ---

    /**
     * Phương thức nội bộ để tạo MongoClientSettings khi lớp được load.
     */
    private static MongoClientSettings createMongoClientSettings() {
        CodecRegistry pojoCodecRegistry = createPojoCodecRegistry();
        return MongoClientSettings.builder()
                .applyConnectionString(new ConnectionString(getConnectionString()))
                .codecRegistry(pojoCodecRegistry)
                .build();
    }

    /**
     * Tạo Codec Registry tùy chỉnh để MongoDB Driver có thể tự động
     * ánh xạ (map) giữa BSON documents và các đối tượng Java (POJO/Record).
     */
    private static CodecRegistry createPojoCodecRegistry() {
        CodecProvider pojoCodecProvider = PojoCodecProvider.builder()
                .automatic(true)
                .build();

        return fromRegistries(
                MongoClientSettings.getDefaultCodecRegistry(),
                fromProviders(pojoCodecProvider));
    }
}