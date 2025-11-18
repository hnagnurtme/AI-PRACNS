package com.sagsins.core.configuration;

import com.mongodb.ConnectionString;
import com.mongodb.MongoClientSettings;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.core.MongoTemplate;

import java.util.concurrent.TimeUnit;

@Configuration
public class MongoConfiguration {

    public static final String DATABASE_NAME = "network";

    /**
     * Tạo MongoClient từ URI kết nối MongoDB với cấu hình timeout và retry
     */
    @Bean
    public MongoClient mongoClient() {
        String uri = "mongodb+srv://admin:SMILEisme0106@mongo1.ragz4ka.mongodb.net/network"
                + "?retryWrites=true&w=majority&tls=true&appName=MONGO1";

        ConnectionString connectionString = new ConnectionString(uri);

        MongoClientSettings settings = MongoClientSettings.builder()
                .applyConnectionString(connectionString)
                // Socket timeout - thời gian chờ đọc/ghi socket (30 seconds)
                .applyToSocketSettings(builder ->
                        builder.connectTimeout(10, TimeUnit.SECONDS)
                               .readTimeout(30, TimeUnit.SECONDS))
                // Server selection timeout - thời gian tìm server (15 seconds)
                .applyToClusterSettings(builder ->
                        builder.serverSelectionTimeout(15, TimeUnit.SECONDS))
                // Connection pool settings
                .applyToConnectionPoolSettings(builder ->
                        builder.maxSize(50)
                               .minSize(5)
                               .maxConnectionIdleTime(60, TimeUnit.SECONDS)
                               .maxConnectionLifeTime(120, TimeUnit.SECONDS))
                .build();

        return MongoClients.create(settings);
    }

    /**
     * Tạo MongoTemplate dùng cho Repository
     */
    @Bean
    public MongoTemplate mongoTemplate(MongoClient mongoClient) {
        return new MongoTemplate(mongoClient, DATABASE_NAME);
    }
}
