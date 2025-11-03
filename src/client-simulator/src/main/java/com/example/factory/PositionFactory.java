package com.example.factory;

import java.util.HashMap;
import java.util.Map;

import com.example.model.Position;

public class PositionFactory {
    /**
     * Tạo Map từ tên thành phố -> Position (tọa độ thật)
     */
    public static Map<String, Position> createWorldCities() {
        Map<String, Position> positionMap = new HashMap<>();

        positionMap.put("New York", new Position(40.7128, -74.0060, 10));
        positionMap.put("London", new Position(51.5074, -0.1278, 11));
        positionMap.put("Paris", new Position(48.8566, 2.3522, 35));
        positionMap.put("Tokyo", new Position(35.6895, 139.6917, 40));
        positionMap.put("Beijing", new Position(39.9042, 116.4074, 43));
        positionMap.put("Moscow", new Position(55.7558, 37.6173, 144));
        positionMap.put("Sydney", new Position(-33.8688, 151.2093, 58));
        positionMap.put("Los Angeles", new Position(34.0522, -118.2437, 71));
        positionMap.put("Rio de Janeiro", new Position(-22.9068, -43.1729, 2));
        positionMap.put("Singapore", new Position(1.3521, 103.8198, 15));
        positionMap.put("Dubai", new Position(25.2048, 55.2708, 16));
        positionMap.put("Toronto", new Position(43.6532, -79.3832, 76));
        positionMap.put("Berlin", new Position(52.5200, 13.4050, 34));
        positionMap.put("Rome", new Position(41.9028, 12.4964, 21));
        positionMap.put("Bangkok", new Position(13.7563, 100.5018, 2));
        positionMap.put("Istanbul", new Position(41.0082, 28.9784, 39));
        positionMap.put("Mumbai", new Position(19.0760, 72.8777, 14));
        positionMap.put("Seoul", new Position(37.5665, 126.9780, 38));
        positionMap.put("Mexico City", new Position(19.4326, -99.1332, 2250));
        positionMap.put("Johannesburg", new Position(-26.2041, 28.0473, 1753));

        positionMap.put("Ho Chi Minh City", new Position(10.8231, 106.6297, 19));
        positionMap.put("Da Nang", new Position(16.0544, 108.2022, 5));
        positionMap.put("Hai Phong", new Position(20.8449, 106.6881, 7));
        positionMap.put("Can Tho", new Position(10.0452, 105.7469, 2));
        positionMap.put("Nha Trang", new Position(12.2388, 109.1967, 3));
        positionMap.put("Hue", new Position(16.4637, 107.5909, 3));
        positionMap.put("Vung Tau", new Position(10.4114, 107.1362, 5));

        return positionMap;
    }
}
