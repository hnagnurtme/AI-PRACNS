package com.sagin.satellite.config;

import com.google.cloud.firestore.Firestore;
import com.sagin.satellite.util.ReadPropertiesUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
public class FireBaseConfiguration {
    private static final Logger logger = LoggerFactory.getLogger(FireBaseConfiguration.class);
    private static Firestore firestore;

    public static void init() throws IOException {
        if (firestore != null) {
            logger.info("Firestore already initialized");
            return;
        }

        String serviceAccountPath = ReadPropertiesUtils.getString("firebase.serviceAccountPath");

        InputStream serviceAccount;

        try {
            serviceAccount = FireBaseConfiguration.class.getClassLoader()
                    .getResourceAsStream("serviceAccountKey.json");
            if (serviceAccount == null) {
                serviceAccount = new FileInputStream(serviceAccountPath);
            } else {
                logger.info("Loading Firebase service account key from classpath");
            }
        } catch (Exception e) {
            logger.warn("Failed to load from classpath, trying file system: {}", serviceAccountPath);
            serviceAccount = new FileInputStream(serviceAccountPath);
        }

    }
}