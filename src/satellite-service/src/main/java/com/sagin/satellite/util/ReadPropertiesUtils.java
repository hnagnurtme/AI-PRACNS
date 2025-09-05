package com.sagin.satellite.util;

import org.yaml.snakeyaml.Yaml;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

public class ReadPropertiesUtils {

    private static final String DEFAULT_CONFIG = "/application.yml";
    private static final String PROFILE_CONFIG_PREFIX = "/application-";

    private static Map<String, Object> config = new HashMap<>();

    static {
        loadConfigs();
    }

    private static void loadConfigs() {
        try {
            Yaml yaml = new Yaml();
            InputStream inputStream = ReadPropertiesUtils.class.getResourceAsStream(DEFAULT_CONFIG);
            if (inputStream == null) {
                throw new RuntimeException("application.yml not found in resources!");
            }
            Map<String, Object> baseConfig = yaml.load(inputStream);
            if (baseConfig != null) {
                config.putAll(flatten(baseConfig, null));
            }
            String environment = (String) config.get("environment");
            if (environment != null) {
                String profileFile = PROFILE_CONFIG_PREFIX + environment + ".yml";
                InputStream profileStream = ReadPropertiesUtils.class.getResourceAsStream(profileFile);
                if (profileStream != null) {
                    Map<String, Object> profileConfig = yaml.load(profileStream);
                    if (profileConfig != null) {
                        config.putAll(flatten(profileConfig, null));
                    }
                }
            }

        } catch (Exception e) {
            throw new RuntimeException("Failed to load configs: " + e.getMessage(), e);
        }
    }
    @SuppressWarnings("unchecked")
    private static Map<String, Object> flatten(Map<String, Object> map, String parentKey) {
        Map<String, Object> flatMap = new HashMap<>();
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            String key = (parentKey == null ? entry.getKey() : parentKey + "." + entry.getKey());
            Object value = entry.getValue();
            if (value instanceof Map) {
                flatMap.putAll(flatten((Map<String, Object>) value, key));
            } else {
                flatMap.put(key, value);
            }
        }
        return flatMap;
    }

    public static String getString(String key) {
        Object value = config.get(key);
        return value != null ? value.toString() : null;
    }

    public static Integer getInt(String key) {
        Object value = config.get(key);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        if (value != null) {
            return Integer.parseInt(value.toString());
        }
        return null;
    }

    public static Long getLong(String key) {
        Object value = config.get(key);
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        if (value != null) {
            return Long.parseLong(value.toString());
        }
        return null;
    }

    public static Boolean getBoolean(String key) {
        Object value = config.get(key);
        if (value != null) {
            return Boolean.parseBoolean(value.toString());
        }
        return null;
    }
}
