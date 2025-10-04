package com.sagin.configuration;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.firestore.Firestore;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.cloud.FirestoreClient;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;

public class FireStoreConfiguration {
    private static final Logger logger = LoggerFactory.getLogger(FireStoreConfiguration.class);
    private static Firestore firestoreInstance; 

    /**
     * Khởi tạo kết nối FirebaseApp và Firestore. 
     * @throws IOException nếu không tìm thấy tệp Service Account.
     */
    public static void init() throws IOException {
        // Bảo vệ khỏi việc khởi tạo lại
        if (firestoreInstance != null) {
            logger.warn("Firestore đã được khởi tạo. Bỏ qua lệnh init() lặp lại.");
            return;
        }

        // Lấy tệp Service Account
        InputStream serviceAccountStream = FirebaseConfiguration.getServiceAccountStream();
        
        if (serviceAccountStream == null) {
            throw new IOException("KHÔNG TÌM THẤY: Tệp Service Account (" + 
                                  FirebaseConfiguration.SERVICE_ACCOUNT_FILE + ") không có trong resources.");
        }

        // Tạo Firebase Options
        FirebaseOptions options = FirebaseOptions.builder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccountStream))
                .build();
        
        // Khởi tạo FirebaseApp (phải kiểm tra vì Firebase SDK không cho phép khởi tạo lại)
        if (FirebaseApp.getApps().isEmpty()) {
            FirebaseApp.initializeApp(options);
            logger.info("FirebaseApp đã khởi tạo thành công.");
        }

        // Lấy instance Singleton của Firestore Client
        firestoreInstance = FirestoreClient.getFirestore();
        logger.info("Kết nối Firestore đã sẵn sàng.");
    }

    /**
     * Trả về instance Singleton của Firestore.
     * @return Đối tượng Firestore.
     */
    public static Firestore getFirestore() {
        if (firestoreInstance == null) {
            // Ném lỗi trạng thái nếu lớp gọi mà chưa init()
            throw new IllegalStateException("Firestore chưa được khởi tạo. Gọi init() từ ServiceConfiguration trước.");
        }
        return firestoreInstance;
    }

    /**
     * Đóng kết nối Firebase/Firestore.
     */
    public static void shutdown() {
        if (!FirebaseApp.getApps().isEmpty()) {
            FirebaseApp.getInstance().delete();
            logger.info("Firestore/FirebaseApp đã tắt.");
        }
    }
}