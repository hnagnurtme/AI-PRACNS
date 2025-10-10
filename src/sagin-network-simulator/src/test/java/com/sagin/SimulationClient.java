package com.sagin;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.UUID;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import java.lang.Math;

// LỚP CHƯƠNG TRÌNH CLIENT CHÍNH
public class SimulationClient {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private static final String SERVER_IP = "127.0.0.1";
    private static final int SERVER_PORT = 8083;
    private static final String TARGET_NODE = "GS_SINGAPORE";
    private static final String SOURCE_CLIENT_ID = "GS_HUHU";
    // Trong SimulationClient.java

    private static final int ACK_RECEIVE_PORT = 8090; // ĐỊNH NGHĨA PORT LẮNG NGHE

    public static void main(String[] args) {
        System.out.println("--- BẮT ĐẦU CHƯƠNG TRÌNH CLIENT ---");

        // 1. KHỞI ĐỘNG LUỒNG LẮNG NGHE ACK (Sử dụng ServerSocket)
        Thread listenerThread = new Thread(SimulationClient::startAckListener);
        listenerThread.start();
        System.out.printf("   Client %s đang lắng nghe ACK trên cổng: %d%n", SOURCE_CLIENT_ID, ACK_RECEIVE_PORT);

        // Chờ 1 giây để Listener sẵn sàng (Tùy chọn)
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // 2. GỬI GÓI DATA
        for (int i = 0; i < 1; i++) {
            String service = "DATA";
            String packetId = UUID.randomUUID().toString().substring(0, 8);

            // Gửi gói tin đi
            // ⚠️ CẦN SỬA: Chúng ta cần thêm cổng ACK_RECEIVE_PORT vào hàm gửi!
            sendJsonPacket(packetId, TARGET_NODE, service, SERVER_IP, SERVER_PORT);

            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        System.out.println("--- KẾT THÚC GỬI LƯU LƯỢNG MẪU ---");

        // 3. GIỮ CHƯƠNG TRÌNH CHẠY để chờ ACK
        try {
            System.out.println("Đang chờ ACK phản hồi (20 giây)...");
            Thread.sleep(20000);
        } catch (InterruptedException e) {
            // Dọn dẹp
        }
        listenerThread.interrupt();
    }

    // --- BỔ SUNG: HÀM LẮNG NGHE ACK ---
    public static void startAckListener() {
        try (ServerSocket serverSocket = new ServerSocket(ACK_RECEIVE_PORT)) {
            while (!Thread.currentThread().isInterrupted()) {
                try (
                        Socket clientSocket = serverSocket.accept(); // Lắng nghe kết nối đến
                        BufferedReader reader = new BufferedReader(
                                new InputStreamReader(clientSocket.getInputStream()))) {
                    String line = reader.readLine();
                    if (line != null) {
                        System.out.println("\n[CLIENT RECEIVER] NHẬN ĐƯỢC PHẢN HỒI (ACK):");
                        System.out.println(line);
                        System.out.println("--- GÓI ACK ĐÃ ĐẾN ĐÍCH! ---");
                    }
                } catch (IOException e) {
                    if (Thread.currentThread().isInterrupted())
                        break;
                    System.err.println("[CLIENT LISTENER] LỖI xử lý kết nối: " + e.getMessage());
                }
            }
        } catch (IOException e) {
            System.err.printf("[CLIENT LISTENER] KHÔNG THỂ MỞ CỔNG %d: %s%n", ACK_RECEIVE_PORT, e.getMessage());
        }
    }

    /**
     * Tạo Map chứa dữ liệu gói tin, bọc trong Wrapper và tuần tự hóa thành JSON
     * String để gửi.
     */
    public static void sendJsonPacket(String packetId, String destinationNodeId, String serviceType, String host,
            int port) {

        // --- CÁC THAM SỐ CỦA GÓI TIN VÀ LINK ---
        final double LATENCY_MS = 15.0;
        final double LOSS_RATE = 0.01;
        final double ATTENUATION_DB = 3.0;
        final double BANDWIDTH = 800.0;

        // 1. Dựng cấu trúc dữ liệu cho Packet (Giữ nguyên)
        Map<String, Object> packetData = new HashMap<>();
        packetData.put("packetId", packetId);
        packetData.put("sourceUserId", SOURCE_CLIENT_ID);
        packetData.put("destinationUserId", destinationNodeId);
        packetData.put("pathHistory", new ArrayList<>());
        packetData.put("type", "DATA");
        packetData.put("serviceType", serviceType);
        packetData.put("TTL", 25);
        packetData.put("timeSentFromSourceMs", System.currentTimeMillis() - 50);
        packetData.put("accumulatedDelayMs", 25.0);
        packetData.put("maxAcceptableLatencyMs", 500.0);
        packetData.put("maxAcceptableLossRate", 0.02);

        // 2. TÍNH TOÁN LINK SCORE (Tương đồng với Server)
        double latencyCost = 1.0 + Math.log(1.0 + LATENCY_MS);
        double lossFactor = 1.0 - LOSS_RATE;
        double attenuationFactor = 1.0 / (1.0 + 0.05 * ATTENUATION_DB);
        double calculatedScore = (BANDWIDTH / latencyCost) * lossFactor * attenuationFactor;

        // 3. DỰNG CẤU TRÚC LINK METRIC GIẢ ĐỊNH (Sửa lỗi ghi đè)
        Map<String, Object> linkMetricData = new HashMap<>();
        linkMetricData.put("sourceNodeId", "LEO_001");
        linkMetricData.put("destinationNodeId", "GS_LONDON");
        linkMetricData.put("distanceKm", 500.0);
        linkMetricData.put("maxBandwidthMbps", 1200.0);
        linkMetricData.put("currentAvailableBandwidthMbps", BANDWIDTH);
        linkMetricData.put("latencyMs", LATENCY_MS);
        linkMetricData.put("packetLossRate", LOSS_RATE);
        linkMetricData.put("linkAttenuationDb", ATTENUATION_DB);

        // BẮT BUỘC SỬA: Đặt là TRUE, đây là điểm cần đảm bảo Server không ghi đè thành
        // false
        linkMetricData.put("isLinkActive", true);

        linkMetricData.put("lastUpdated", System.currentTimeMillis());

        // KHÔNG GỬI linkScore TÍNH TOÁN TỪ CLIENT (để Server tự tính, tránh lỗi mâu
        // thuẫn)
        // linkMetricData.put("linkScore", calculatedScore);

        // 4. Dựng cấu trúc PacketTransferWrapper
        Map<String, Object> wrapperData = new HashMap<>();
        wrapperData.put("packet", packetData);
        wrapperData.put("linkMetric", linkMetricData);

        // 5. Serialize và Gửi
        String jsonString;
        try {
            jsonString = OBJECT_MAPPER.writeValueAsString(wrapperData);
        } catch (JsonProcessingException e) {
            System.err.printf("[CLIENT] LỖI tuần tự hóa JSON cho gói %s: %s%n", packetId, e.getMessage());
            return;
        }

        try (
                Socket socket = new Socket(host, port);
                PrintWriter writer = new PrintWriter(socket.getOutputStream(), true)) {
            System.out.printf("[CLIENT] Gửi gói %s (%s) tới %s:%d... ",
                    packetId, serviceType, host, port);
            writer.println(jsonString);
            System.out.println("THÀNH CÔNG.");

        } catch (IOException e) {
            System.err.printf("\n[CLIENT] LỖI KẾT NỐI/GỬI đến %s:%d: %s%n",
                    host, port, e.getMessage());
            System.err.println("KIỂM TRA: Node Server có đang chạy và cổng có mở không?");
        }
    }
}