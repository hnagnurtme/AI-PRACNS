package com.sagsins.core.configuration;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.firestore.Firestore;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.cloud.FirestoreClient;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import jakarta.annotation.PostConstruct;


@Configuration
public class FireBaseConfiguration {

    @Value("${firebase.serviceAccountPath}")
    private String serviceAccountPath;

    @Value("${firebase.databaseUrl}")
    private String databaseUrl;

    @PostConstruct
    public void initFirebase() {
        try {
            if (serviceAccountPath == null || serviceAccountPath.isEmpty()) {
                throw new IllegalArgumentException("serviceAccountPath is null or empty. Please check your configuration.");
            }
            String resourcePath = serviceAccountPath.startsWith("/") ? serviceAccountPath : "/" + serviceAccountPath;
            if (getClass().getResourceAsStream(resourcePath) == null) {
                throw new IllegalArgumentException("Resource not found: " + resourcePath);
            }
            GoogleCredentials credentials = GoogleCredentials.fromStream(getClass().getResourceAsStream(resourcePath));

            FirebaseOptions options = FirebaseOptions.builder()
                    .setCredentials(credentials)
                    .setDatabaseUrl(databaseUrl)
                    .build();

            if (FirebaseApp.getApps().isEmpty()) {
                FirebaseApp.initializeApp(options);
            }

        } catch (Exception e) {
            Logger logger = LoggerFactory.getLogger(FireBaseConfiguration.class);
            logger.error("Error initializing Firebase: ", e);
        }
    }

    @Bean
    public Firestore firestore() {
        return FirestoreClient.getFirestore();
    }

}
