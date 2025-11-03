package com.example.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

/**
 * Small utility wrapper over SLF4J Logger to centralize MDC helpers.
 */
public final class AppLogger {

    private AppLogger() {}

    public static Logger getLogger(Class<?> cls) {
        return LoggerFactory.getLogger(cls);
    }

    public static void putMdc(String key, String value) {
        if (key != null && value != null) {
            MDC.put(key, value);
        }
    }

    public static void removeMdc(String key) {
        if (key != null) MDC.remove(key);
    }

    public static void clearMdc() {
        MDC.clear();
    }
}
