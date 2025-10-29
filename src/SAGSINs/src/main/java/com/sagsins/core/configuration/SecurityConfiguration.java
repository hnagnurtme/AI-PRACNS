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
        .csrf(AbstractHttpConfigurer::disable)
        .authorizeHttpRequests(auth -> auth
            // Swagger & API docs
            .requestMatchers(
                "/v3/api-docs/**",
                "/swagger-ui/**",
                "/swagger-ui.html"
            ).permitAll()

            // WebSocket endpoint
            .requestMatchers("/ws/**").permitAll()

            // REST API
            .requestMatchers("/api/**").permitAll()

            // Tất cả còn lại
            .anyRequest().permitAll()
        )
        .httpBasic(withDefaults());

    return http.build();
}

}