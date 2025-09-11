package com.chatapp.chatapp.config;

import java.io.InputStream;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.firestore.Firestore;
import com.google.firebase.cloud.FirestoreClient;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;

public class FirebaseConfig {
    private static Firestore firestore;
    private static final String PROJECT_ID = "ai-pracns";
    
    // Static block để tự động khởi tạo khi class được load
    static {
        try {
            initFirebase();
        } catch (Exception e) {
            System.err.println("Failed to auto-initialize Firebase: " + e.getMessage());
        }
    }
    
    public static void initFirebase() throws Exception {
        if (firestore != null) {
            System.out.println("Firestore already initialized");
            return;
        }

        try {
            // Sử dụng classpath resource thay vì file path
            InputStream serviceAccount = FirebaseConfig.class.getClassLoader()
                    .getResourceAsStream("serviceAccountKey.json");
            
            if (serviceAccount == null) {
                throw new RuntimeException("Service account key file not found in resources");
            }

            FirebaseOptions options = FirebaseOptions.builder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .setProjectId(PROJECT_ID)
                .build();

            if (FirebaseApp.getApps().isEmpty()) {
                FirebaseApp.initializeApp(options);
                System.out.println("FirebaseApp initialized successfully");
            }

            firestore = FirestoreClient.getFirestore();
            System.out.println("Firestore initialized successfully");
            
        } catch (Exception e) {
            System.err.println("Firebase initialization failed: " + e.getMessage());
            throw e;
        }
    }

    public static Firestore getFirestore() {
        if (firestore == null) {
            try {
                initFirebase();
            } catch (Exception e) {
                throw new RuntimeException("Failed to initialize Firestore", e);
            }
        }
        return firestore;
    }

    public static void shutdown() {
        if (!FirebaseApp.getApps().isEmpty()) {
            FirebaseApp.getInstance().delete();
            firestore = null;
            System.out.println("FirebaseApp shutdown complete");
        }
    }
}
