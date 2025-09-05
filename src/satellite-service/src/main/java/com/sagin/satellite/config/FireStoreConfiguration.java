package com.sagin.satellite.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.firestore.Firestore;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.cloud.FirestoreClient;
import com.sagin.satellite.util.ReadPropertiesUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.IOException;

public class FireStoreConfiguration {
    private static final Logger logger = LoggerFactory.getLogger(FireStoreConfiguration.class);
    private static Firestore firestore;

    public static void init() throws IOException {
        if (firestore != null) {
            logger.info("Firestore already initialized");
            return;
        }
        String serviceAccountPath = ReadPropertiesUtils.getString("firebase.serviceAccountPath");
        String databaseUrl = ReadPropertiesUtils.getString("firebase.databaseUrl");

        FileInputStream serviceAccount = new FileInputStream(serviceAccountPath);

        FirebaseOptions options = FirebaseOptions.builder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .setDatabaseUrl(databaseUrl)
                .build();
        ;

        if (FirebaseApp.getApps().isEmpty()) {
            FirebaseApp.initializeApp(options);
            logger.info("FirebaseApp initialized for Firestore");
        }

        firestore = FirestoreClient.getFirestore();
        logger.info("Firestore has been initialized");
    }

    public static Firestore getFirestore() {
        if (firestore == null) {
            throw new IllegalStateException("Firestore not initialized. Call init() first.");
        }
        return firestore;
    }

    public static void shutdown() {
        if (!FirebaseApp.getApps().isEmpty()) {
            FirebaseApp.getInstance().delete();
            logger.info("Firestore/FirebaseApp shutdown complete");
        }
    }
}
