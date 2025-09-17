package com.chatapp.chatapp.utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public class LoadProperties {
    /**
     * Đọc file .env và trả về Map<String,String> chứa key-value
     */
    public static Map<String, String> loadEnv(Path envPath) throws IOException {
        Map<String, String> envMap = new HashMap<>();

        for (String line : Files.readAllLines(envPath)) {
            String trimmed = line.trim();
            // Bỏ qua dòng trống và comment
            if (trimmed.isEmpty() || trimmed.startsWith("#")) continue;

            int idx = trimmed.indexOf('=');
            if (idx < 0) continue; // dòng không hợp lệ

            String key = trimmed.substring(0, idx).trim();
            String value = trimmed.substring(idx + 1).trim();

            // Bỏ dấu nháy nếu có
            if ((value.startsWith("\"") && value.endsWith("\"")) ||
                (value.startsWith("'") && value.endsWith("'"))) {
                value = value.substring(1, value.length() - 1);
            }

            envMap.put(key, value);
        }

        return envMap;
    }
    
    /**
     * Overload method với String path
     */
    public static Map<String, String> loadEnv(String envPath) throws IOException {
        return loadEnv(Path.of(envPath));
    }
    
    /**
     * Load .env từ default location
     */
    public static Map<String, String> loadEnv() throws IOException {
        return loadEnv(".env");
    }
}
