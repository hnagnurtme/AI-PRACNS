package com.sagin.util;

import com.sagin.model.Geo3D;

public class GeoUtils {

    // Vận tốc ánh sáng trong chân không (gần đúng) - đơn vị: km/ms
    private static final double LIGHT_SPEED_KM_PER_MS = 299792.458 / 1000.0; 

    /**
     * Tính khoảng cách Euclidean 3D giữa hai node.
     * Công thức: distance = sqrt(dx^2 + dy^2 + dz^2)
     * Vì bạn dùng Geo3D (Lat/Lon/Alt), công thức này là đơn giản nhất cho mô phỏng.
     * LƯU Ý: Trong thực tế, cần dùng công thức Great-Circle (Haversine) và chuyển Lat/Lon/Alt về tọa độ Descartes (ECF).
     * Tuy nhiên, cho mô phỏng đơn giản, ta tính khoảng cách dựa trên tọa độ đã có.
     * Ta giả định tọa độ Lat/Lon đã được chuẩn hóa thành km trên một mặt phẳng xấp xỉ.
     * Giả sử Geo3D của bạn đã được chuyển đổi sang ECEF hoặc sử dụng công thức sau cho khoảng cách giữa hai điểm trong không gian.
     */
    public static double calculateDistance3D(Geo3D posA, Geo3D posB) {
        // Đây là công thức giả định, bạn cần thay thế bằng tính toán khoảng cách thực tế 
        // dựa trên tọa độ cầu (Spherical Coordinates) nếu muốn độ chính xác cao.
        // Tuy nhiên, để tiếp tục, ta tính toán khoảng cách 3D giả lập:
        double latDiff = (posA.getLatitude() - posB.getLatitude()) * 111.0; // xấp xỉ 111km/độ lat
        double lonDiff = (posA.getLongitude() - posB.getLongitude()) * 111.0 * Math.cos(Math.toRadians(posA.getLatitude()));
        double altDiff = posA.getAltitude() - posB.getAltitude(); 

        return Math.sqrt(latDiff * latDiff + lonDiff * lonDiff + altDiff * altDiff);
    }

    /**
     * Tính độ trễ truyền dẫn dựa trên khoảng cách vật lý.
     * @param distanceKm Khoảng cách giữa hai node (km).
     * @return Độ trễ truyền dẫn (ms).
     */
    public static double calculatePropagationDelayMs(double distanceKm) {
        return distanceKm / LIGHT_SPEED_KM_PER_MS;
    }
    
    /**
     * Kiểm tra tầm nhìn (Line of Sight) giữa hai node.
     * Chỉ là một kiểm tra đơn giản: LEO có thể thấy Ground Station nếu nó nằm trên đường chân trời.
     * Trong thực tế, cần tính toán chướng ngại vật và góc nâng (elevation angle).
     */
    public static boolean checkVisibility(Geo3D posA, Geo3D posB) {
        // Giả định đơn giản: Mọi vệ tinh LEO có thể nhìn thấy nhau. 
        // Ground Station chỉ có thể thấy vệ tinh nếu nó không quá xa (giả sử < 5000 km)
        double distance = calculateDistance3D(posA, posB);
        return distance < 5000000.0; // Ngưỡng tầm nhìn đơn giản
    }
}