package com.sagsins.core.configuration;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.messaging.DefaultMessageListenerContainer;
import org.springframework.data.mongodb.core.messaging.MessageListenerContainer;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * Configuration cho MongoDB Change Streams
 * Cho phép theo dõi realtime các thay đổi trong database
 */
@Configuration
public class ChangeStreamConfiguration {
    
    /**
     * Tạo MessageListenerContainer để quản lý các Change Stream listeners
     * Container này sẽ kết nối với MongoDB và lắng nghe các thay đổi
     */
    @Bean
    public MessageListenerContainer messageListenerContainer(MongoTemplate mongoTemplate) {
        // Tạo thread pool cho việc xử lý change events
        Executor executor = Executors.newFixedThreadPool(2);
        
        return new DefaultMessageListenerContainer(mongoTemplate, executor) {
            @Override
            public boolean isAutoStartup() {
                // Không tự động start, sẽ start thủ công trong service
                return false;
            }
        };
    }
}
