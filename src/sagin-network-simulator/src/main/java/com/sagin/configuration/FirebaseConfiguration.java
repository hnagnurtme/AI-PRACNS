package com.sagin.configuration;

import java.io.IOException;
import java.io.InputStream;

/**
 * Cấu hình tĩnh cho Firebase SDK, bao gồm các hằng số và tiện ích kết nối.
 * Được sử dụng bởi lớp FireStoreConfiguration.
 */
public class FirebaseConfiguration {
    
    // Đường dẫn tới tệp Service Account Key (Đặt trong thư mục resources)
    public static final String SERVICE_ACCOUNT_FILE = "sagin-service-account-key.json"; 
    
    // Tên Collection/Path cơ sở để lưu trữ NodeInfo
    public static final String NODES_COLLECTION_PATH = "NETWORK_NODES"; 

    /**
     * Lấy InputStream của tệp Service Account Key từ thư mục resources.
     * @return InputStream của tệp key.
     */
    public static InputStream getServiceAccountStream() {
        // Sử dụng ClassLoader để tải tệp từ thư mục resources
        return FirebaseConfiguration.class.getClassLoader().getResourceAsStream(SERVICE_ACCOUNT_FILE);
    }
    
    /**
     * Kiểm tra xem tệp Service Account Key đã tồn tại chưa.
     * @return true nếu tệp key tồn tại và có thể đọc được.
     */
    public static boolean isConfigReady() {
        // Đóng InputStream ngay lập tức sau khi kiểm tra
        try (InputStream stream = getServiceAccountStream()) {
            return stream != null;
        } catch (IOException e) {
            return false;
        }
    }
}