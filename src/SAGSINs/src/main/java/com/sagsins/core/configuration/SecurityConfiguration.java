// src/main/java/com/sagsins/core/configuration/SecurityConfiguration.java
package com.sagsins.core.configuration;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.web.SecurityFilterChain;

import static org.springframework.security.config.Customizer.withDefaults;

/**
 * Cấu hình bảo mật cơ bản cho API REST.
 * Vô hiệu hóa CSRF, cho phép tất cả các yêu cầu (chưa áp dụng JWT/Authentication).
 */
@Configuration
@EnableWebSecurity
public class SecurityConfiguration {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            // 1. Vô hiệu hóa CSRF (Cần thiết cho API RESTful, tránh xung đột với POST/PUT/DELETE)
            .csrf(AbstractHttpConfigurer::disable)
            
            // 2. Định nghĩa ủy quyền (Authorization)
            .authorizeHttpRequests(authorize -> authorize
                // Cho phép truy cập công khai tới tất cả các endpoints (bao gồm /api/nodes và Swagger)
                .anyRequest().permitAll() 
            )
            // 3. Sử dụng cấu hình mặc định khác
            .httpBasic(withDefaults());

        return http.build();
    }
}