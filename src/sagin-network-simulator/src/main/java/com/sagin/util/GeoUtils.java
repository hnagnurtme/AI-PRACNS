package com.sagin.util;

import com.sagin.model.Geo3D;
import com.sagin.model.NodeInfo;
import com.sagin.model.NodeType;
import com.sagin.model.WeatherCondition;

public class GeoUtils {

    private static final double EARTH_RADIUS_KM = 6371.0;
        // Vận tốc ánh sáng trong chân không (gần đúng) - đơn vị: km/ms
    private static final double LIGHT_SPEED_KM_PER_MS = 299792.458 / 1000.0; 

    /**
     * Tính khoảng cách Euclidean 3D giữa hai node.


    /**
     * Kiểm tra Line of Sight giữa hai NodeInfo
     * @return true nếu có thể nhìn thấy nhau
     */
    public static boolean checkVisibility(NodeInfo sourceNode, NodeInfo destNode) {
        if (sourceNode == null || destNode == null) return false;

        Geo3D srcPos = sourceNode.getPosition();
        Geo3D dstPos = destNode.getPosition();

        if (srcPos == null || dstPos == null) return false;

        NodeType srcType = sourceNode.getNodeType();
        NodeType dstType = destNode.getNodeType();

        // Kiểm tra LOS
        double hSource = adjustHeightForType(srcType, srcPos.getAltitude());
        double hDest = adjustHeightForType(dstType, dstPos.getAltitude());

        double lat1 = Math.toRadians(srcPos.getLatitude());
        double lon1 = Math.toRadians(srcPos.getLongitude());
        double lat2 = Math.toRadians(dstPos.getLatitude());
        double lon2 = Math.toRadians(dstPos.getLongitude());

        double dLat = lat2 - lat1;
        double dLon = lon2 - lon1;

        double haversine = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                           Math.cos(lat1) * Math.cos(lat2) *
                           Math.sin(dLon / 2) * Math.sin(dLon / 2);

        double c = 2 * Math.atan2(Math.sqrt(haversine), Math.sqrt(1 - haversine));

        double groundDistance = EARTH_RADIUS_KM * c;

        double maxLOS = Math.sqrt(2 * EARTH_RADIUS_KM * hSource + hSource * hSource)
                      + Math.sqrt(2 * EARTH_RADIUS_KM * hDest + hDest * hDest);

        // Giới hạn LOS theo loại Node
        maxLOS = applyNodeTypeLimits(srcType, dstType, maxLOS);

        boolean isVisible = groundDistance <= maxLOS;

        // Kiểm tra điều kiện thời tiết GS
        if (dstType == NodeType.GROUND_STATION &&
            destNode.getWeather() == WeatherCondition.SEVERE_STORM) {
            return false;
        }

        return isVisible;
    }

    private static double adjustHeightForType(NodeType type, double altKm) {
        switch (type) {
            case GROUND_STATION: return altKm;          // sát mặt đất
            case LEO_SATELLITE: return altKm;           // 500-1200 km
            case MEO_SATELLITE: return altKm;           // 10000-20000 km
            case GEO_SATELLITE: return altKm;           // ~35786 km
            default: return altKm;
        }
    }

    private static double applyNodeTypeLimits(NodeType source, NodeType dest, double los) {
        if (source == NodeType.LEO_SATELLITE && dest == NodeType.LEO_SATELLITE) {
            return Math.min(los, 2000);
        } else if ((source == NodeType.LEO_SATELLITE && dest == NodeType.GEO_SATELLITE)
                || (source == NodeType.GEO_SATELLITE && dest == NodeType.LEO_SATELLITE)) {
            return Math.min(los, 40000);
        } else if ((source == NodeType.MEO_SATELLITE && dest == NodeType.GEO_SATELLITE)
                || (source == NodeType.GEO_SATELLITE && dest == NodeType.MEO_SATELLITE)) {
            return Math.min(los, 30000);
        }
        return los;
    }

    public static double calculateDistance3D(Geo3D posA, Geo3D posB) {
        double latDiff = (posA.getLatitude() - posB.getLatitude()) * 111.0; 
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
        return distance < 5000.0;
    }
}

