package com.sagin.satellite.controller;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * BaseController chuẩn, cung cấp response success/error đồng bộ
 */
public abstract class BaseController {

    /**
     * Trả response thành công với dữ liệu typed
     */
    protected <T> Map<String, Object> success(T data) {
        Map<String, Object> response = new HashMap<>(4);
        response.put("timestamp", Instant.now().toString());
        response.put("status", "success");
        response.put("data", data);
        return response;
    }

    /**
     * Trả response khi xảy ra lỗi
     */
    protected Map<String, Object> error(Exception e) {
        Map<String, Object> response = new HashMap<>(5);
        response.put("timestamp", Instant.now().toString());
        response.put("status", "error");
        response.put("errorType", e != null ? e.getClass().getSimpleName() : "UnknownError");
        response.put("message", e != null ? e.getMessage() : "No message available");
        return response;
    }

    /**
     * Tạo response error tùy chỉnh với message
     */
    protected Map<String, Object> error(String message) {
        Map<String, Object> response = new HashMap<>(4);
        response.put("timestamp", Instant.now().toString());
        response.put("status", "error");
        response.put("errorType", "CustomError");
        response.put("message", message);
        return response;
    }
}
