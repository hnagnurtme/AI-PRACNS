package com.sagin;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.HashMap;
import java.util.Map;

// BẮT BUỘC: Thư viện Jackson phải được thêm vào Client Project
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;

// LỚP CHƯƠNG TRÌNH CLIENT CHÍNH
public class SimulationClient {
    
    // ObjectMapper phải được tạo một lần và tái sử dụng
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    
    private static final String SERVER_IP = "127.0.0.1";
    private static final int SERVER_PORT = 8081; 
    private static final String TARGET_NODE = "LEO_004"; 
    private static final String SOURCE_CLIENT_ID = "GS_NY";
    
    public static void main(String[] args) {
        System.out.println("--- BẮT ĐẦU CHƯƠNG TRÌNH CLIENT (JSON Generator) ---");
        System.out.printf("   Đích: %s:%d | Target Node: %s%n", SERVER_IP, SERVER_PORT, TARGET_NODE);
        
        // Gửi 10 gói tin với độ trễ 500ms
        for (int i = 0; i < 1; i++) {
            String service = (i % 3 == 0) ? "VOICE" : (i % 3 == 1) ? "VIDEO" : "DATA";
            
            // TẠO JSON DỰA TRÊN MAP
            String packetId = UUID.randomUUID().toString().substring(0, 8);
            
            sendJsonPacket(packetId, TARGET_NODE, service, SERVER_IP, SERVER_PORT);
            
            try {
                Thread.sleep(500); 
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        System.out.println("--- KẾT THÚC GỬI LƯU LƯỢNG MẪU ---");
    }

    /**
     * Tạo Map chứa dữ liệu gói tin và tuần tự hóa thành JSON String để gửi.
     */
    public static void sendJsonPacket(String packetId, String destinationNodeId, String serviceType, String host, int port) {
        
        // 1. Dựng cấu trúc dữ liệu (Map) - Ánh xạ trực tiếp tới các fields của lớp Packet Server
        Map<String, Object> packetData = new HashMap<>();
        packetData.put("packetId", packetId);
        packetData.put("sourceUserId", SOURCE_CLIENT_ID);
        packetData.put("destinationUserId", destinationNodeId);
        
        // Fields cần thiết cho Server xử lý
        packetData.put("type", "DATA"); 
        packetData.put("serviceType", serviceType);
        packetData.put("TTL", 15); // TTL ban đầu
        packetData.put("timeSentFromSourceMs", System.currentTimeMillis());
        
        // Fields QoS Cần thiết cho NodeService.sendPacket() kiểm tra
        if ("VOICE".equals(serviceType)) {
            packetData.put("maxAcceptableLatencyMs", 150.0);
            packetData.put("maxAcceptableLossRate", 0.01);
        } else if ("VIDEO".equals(serviceType)) {
            packetData.put("maxAcceptableLatencyMs", 500.0);
            packetData.put("maxAcceptableLossRate", 0.02);
        } else { // DATA
            packetData.put("maxAcceptableLatencyMs", 2000.0);
            packetData.put("maxAcceptableLossRate", 0.05);
        }
        
        // Các fields List<String> phải được khởi tạo (hoặc để null nếu Server chấp nhận)
        packetData.put("pathHistory", new java.util.ArrayList<String>()); 
        
        String jsonString;
        try {
            // 2. Tuần tự hóa Map thành JSON String
            jsonString = OBJECT_MAPPER.writeValueAsString(packetData);
        } catch (JsonProcessingException e) {
            System.err.printf("[CLIENT] LỖI tuần tự hóa JSON cho gói %s: %s%n", packetId, e.getMessage());
            return;
        }

        // 3. Gửi JSON String qua Socket
        try (
            Socket socket = new Socket(host, port);
            PrintWriter writer = new PrintWriter(socket.getOutputStream(), true) 
        ) {
            System.out.printf("[CLIENT] Gửi gói %s (%s) tới %s:%d... ", 
                              packetId, serviceType, host, port);
            
            // Ghi chuỗi JSON và thêm ký tự xuống dòng (\n)
            writer.println(jsonString); 
            
            System.out.println("THÀNH CÔNG.");
            
        } catch (IOException e) {
            System.err.printf("\n[CLIENT] LỖI KẾT NỐI/GỬI đến %s:%d: %s%n", 
                                host, port, e.getMessage());
            System.err.println("KIỂM TRA: Node Server có đang chạy và cổng có mở không?");
        }
    }
}