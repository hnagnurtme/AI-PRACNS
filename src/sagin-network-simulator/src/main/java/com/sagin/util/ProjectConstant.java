package com.sagin.util;

/**
 * Tập hợp các hằng số (constants) được sử dụng trong toàn bộ dự án.
 * Giúp chuẩn hóa các chuỗi quan trọng như loại Node, loại Dịch vụ và các giá trị vật lý.
 */
public class ProjectConstant {

    // ----- Hằng số cho NODE_TYPE (Sử dụng trong NodeInfo.nodeType) -----
    public static final String NODE_TYPE_SATELLITE = "SATELLITE";
    public static final String NODE_TYPE_UAV = "UAV"; 
    public static final String NODE_TYPE_GROUND_STATION = "GROUND_STATION"; 
    public static final String NODE_TYPE_USER_TERMINAL = "USER_TERMINAL"; 
    public static final String NODE_TYPE_SEA_VESSEL = "SEA_VESSEL";

    // ----- Hằng số cho SERVICE_TYPE (Sử dụng trong Packet.serviceType) -----
    public static final String SERVICE_TYPE_VOICE = "VOICE";            // Yêu cầu độ trễ cực thấp
    public static final String SERVICE_TYPE_VIDEO = "VIDEO_STREAMING"; // Yêu cầu băng thông cao
    public static final String SERVICE_TYPE_DATA_BULK = "DATA_BULK";   // Không nhạy cảm về độ trễ
    public static final String SERVICE_TYPE_CONTROL = "CONTROL_SIGNAL"; // Ưu tiên cao nhất
    
    // ----- Hằng số Định tuyến/Mô phỏng Vật lý -----
    public static final int DEFAULT_TTL = 30; // Time-To-Live mặc định
    // Tốc độ ánh sáng trong chân không (xấp xỉ 300,000 km/s) được chuyển sang km/ms
    public static final double SPEED_OF_LIGHT_KM_PER_MS = 299792.458 / 1000.0; 


}