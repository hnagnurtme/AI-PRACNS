package com.sagsins.core.configuration;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.core.MongoTemplate;

@Configuration
public class MongoConfiguration {

    public static final String DATABASE_NAME = "network";

    /**
     * Tạo MongoClient từ URI kết nối MongoDB
     */
    @Bean
    public MongoClient mongoClient() {
        String uri = "mongodb+srv://admin:SMILEisme0106@mongo1.ragz4ka.mongodb.net/network"
                + "?retryWrites=true&w=majority&tls=true&appName=MONGO1";
        return MongoClients.create(uri);
    }

    /**
     * Tạo MongoTemplate dùng cho Repository
     */
    @Bean
    public MongoTemplate mongoTemplate(MongoClient mongoClient) {
        return new MongoTemplate(mongoClient, DATABASE_NAME);
    }
}
