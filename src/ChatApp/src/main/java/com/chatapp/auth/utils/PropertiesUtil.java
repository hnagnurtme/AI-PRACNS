package com.chatapp.auth.utils;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class PropertiesUtil {
    private static final Properties properties = new Properties();

    static {
        loadProperties();
    }

    private static void loadProperties() {
        try (InputStream input = PropertiesUtil.class.getClassLoader()
                .getResourceAsStream("application.properties")) {
            if (input != null) {
                properties.load(input);
                System.out.println("✅ Loaded application.properties successfully");
            } else {
                System.err.println("❌ application.properties file not found");
            }
        } catch (IOException e) {
            System.err.println("❌ Error loading application.properties: " + e.getMessage());
        }
    }

    public static String getString(String key) {
        return properties.getProperty(key);
    }

    public static String getString(String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
}
