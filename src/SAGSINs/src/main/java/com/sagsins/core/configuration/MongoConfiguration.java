package com.sagsins.core.configuration;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.core.MongoTemplate;

@Configuration
public class MongoConfiguration {

    public static final String DATABASE_NAME = "sagsin_network";

    /**
     * Tạo MongoClient từ URI kết nối MongoDB
     */
    @Bean
    public MongoClient mongoClient() {
        String uri = "mongodb://user:password123@localhost:27017/?authSource=admin";
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
