package com.sagin.util;

import com.sagin.model.Geo3D;
import com.sagin.model.NodeInfo;

/**
 * Xử lý việc đọc và khởi tạo các đối tượng NodeInfo 
 * từ các tham số dòng lệnh (được truyền qua Docker Compose).
 */
public class Initializer {
    
    // Số lượng tham số cần thiết: (ID, Type, Priority, Lat, Lon, Alt, MaxBW, Latency, Power)
    public static final int REQUIRED_ARGS_COUNT = 9; 

    /**
     * Khởi tạo đối tượng NodeInfo từ mảng tham số dòng lệnh.
     * @param args Mảng tham số (ID, Type, Priority, Lat, Lon, Alt, MaxBW, Latency, Power)
     * @return Đối tượng NodeInfo đã được khởi tạo.
     * @throws IllegalArgumentException nếu số lượng hoặc định dạng tham số không hợp lệ.
     */
    public static NodeInfo initializeNodeFromArgs(String[] args) {
        if (args.length < REQUIRED_ARGS_COUNT) {
            throw new IllegalArgumentException(
                "Lỗi: Thiếu tham số khởi tạo. Cần ít nhất " + REQUIRED_ARGS_COUNT + " tham số."
            );
        }

        try {
            // 1. Dữ liệu Cơ bản
            String nodeId = args[0];
            String nodeType = args[1];
            // PriorityLevel của Node được lưu tạm thời, sau này dùng cho NodeController
            // int priorityLevel = Integer.parseInt(args[2]); // TODO: Implement priority level usage
            
            // 2. Dữ liệu Vị trí (Geo3D)
            double latitude = Double.parseDouble(args[3]);
            double longitude = Double.parseDouble(args[4]);
            double altitude = Double.parseDouble(args[5]);
            Geo3D initialPosition = new Geo3D(latitude, longitude, altitude);
            
            // 3. Dữ liệu Metric Khởi tạo
            double initialMaxBandwidth = Double.parseDouble(args[6]);
            double initialLatency = Double.parseDouble(args[7]);
            double initialPowerLevel = Double.parseDouble(args[8]);

            // Khởi tạo NodeInfo
            NodeInfo currentNode = new NodeInfo();
            currentNode.setNodeId(nodeId);
            currentNode.setNodeType(nodeType);
            currentNode.setPosition(initialPosition);
            
            // Thiết lập các Metric khởi tạo (là giá trị ban đầu cho mô phỏng)
            currentNode.setCurrentBandwidth(initialMaxBandwidth);
            currentNode.setAvgLatencyMs(initialLatency);
            currentNode.setPowerLevel(initialPowerLevel);
            
            // Thiết lập các giá trị mặc định cho các metric động
            currentNode.setOperational(true);
            currentNode.setPacketLossRate(0.0);
            currentNode.setPacketBufferLoad(0);
            currentNode.setCurrentThroughput(0.0);
            currentNode.setResourceUtilization(0.0); 
            currentNode.setLastUpdated(System.currentTimeMillis());

            return currentNode;

        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Lỗi: Định dạng tham số không hợp lệ. Hãy kiểm tra các giá trị số.", e);
        }
    }
}