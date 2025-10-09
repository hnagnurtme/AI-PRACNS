package com.sagin.configuration;

import com.mongodb.ConnectionString;
import com.mongodb.MongoClientSettings;
import com.sagin.model.Geo3D;
import com.sagin.model.NodeInfo;
import com.sagin.model.Orbit;
import com.sagin.model.Velocity;

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
public class MongoConfiguration {

    // --- CẤU HÌNH KẾT NỐI ---
    public static final String CONNECTION_STRING = "mongodb://user:password123@localhost:27017/?authSource=admin";
    public static final String DATABASE_NAME = "SAGSINS";

    // --- TÊN COLLECTIONS (Đã cập nhật theo quy ước chữ thường) ---
    public static final String NODES_COLLECTION = "network_nodes";
    public static final String ROUTING_LOGS_COLLECTION = "routing_logs";
    public static final String PACKET_LOGS_COLLECTION = "packet_logs";

    // --- TIỆN ÍCH TRUY CẬP ---

    public static String getConnectionString() {
        return Optional.ofNullable(System.getenv("MONGO_URI"))
                .orElse(CONNECTION_STRING);
    }

    public static String getDatabaseName() {
        return DATABASE_NAME;
    }

    // --- CẤU HÌNH POJO CODEC VÀ SETTINGS MONGOCLIENT ---

    /**
     * Tạo Codec Registry tùy chỉnh, bao gồm khả năng mapping POJO.
     */
    private static CodecRegistry createPojoCodecRegistry() {
        CodecProvider pojoCodecProvider = PojoCodecProvider.builder()
                // BỎ QUA .automatic(true) và thay bằng register()
                .register(NodeInfo.class,
                        Geo3D.class,
                        Orbit.class,
                        Velocity.class)
                // Nếu bạn có các lớp POJO khác (không phải Enum), hãy thêm chúng vào đây!
                .build();

        return fromRegistries(
                MongoClientSettings.getDefaultCodecRegistry(),
                fromProviders(pojoCodecProvider));
    }

    /**
     * Tạo MongoClientSettings đã cấu hình POJO Codec.
     * Lớp Repository sẽ sử dụng Settings này để tạo MongoClient.
     */
    public static MongoClientSettings getMongoClientSettings() {
        return MongoClientSettings.builder()
                .applyConnectionString(new ConnectionString(getConnectionString()))
                .codecRegistry(createPojoCodecRegistry())
                .build();
    }

    public static boolean isConfigReady() {
        return !getConnectionString().isEmpty() && !DATABASE_NAME.isEmpty();
    }
}