package com.sagin.satellite.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.logging.Logger;

public class FireBaseConfiguration {
    private static final Logger logger = Logger.getLogger(FireBaseConfiguration.class.getName());

    public static void init() throws IOException {
        Properties props = new Properties();

        // Load application.properties từ resources
        try (InputStream input = FireBaseConfiguration.class
                .getClassLoader()
                .getResourceAsStream("application-prod.properties")) {

            if (input == null) {
                throw new IOException("application.properties not found in resources!");
            }
            props.load(input);
        }

        String serviceAccountPath = props.getProperty("firebase.serviceAccountPath");
        String databaseUrl = props.getProperty("firebase.databaseUrl");

        // Load serviceAccountKey.json (vẫn từ path, có thể cải tiến đọc từ resources luôn)
        FileInputStream serviceAccount = new FileInputStream(serviceAccountPath);

        FirebaseOptions options = FirebaseOptions.builder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .setDatabaseUrl(databaseUrl)
                .build();

        if (FirebaseApp.getApps().isEmpty()) {
            FirebaseApp.initializeApp(options);
            logger.info("Firebase has been initialized");
        } else {
            logger.info("FirebaseApp already initialized");
        }
    }
}
